"""ENGINE v3 resource-safety tests: bounded frame streaming, work-dir/job-dir cleanup (normal exit, cancel,
SIGTERM, stale dirs of hard-killed jobs), pre-flight disk check, no orphaned measurement workers, and the
independent quality report.

    PYTHONPATH=engine .venv/bin/python -m pytest tests/test_resources.py -q
"""
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

from stillpoint.framestream import FrameStream, decode_blocks
from stillpoint.workspace import InsufficientDiskSpace, JobDir, preflight, sweep_stale, work_root
from eval.footage import o3  # noqa: E402  (env-configurable, see eval/footage.py)

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
CLIP = o3('DJI_0026.MP4')
PY = sys.executable
needs_clip = pytest.mark.skipif(not os.path.exists(CLIP), reason='footage not available')


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers kill(0); ask ps for its state
    st = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith('Z')


def _wait_dead(pids, timeout=6.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.1)
    return not any(_alive(p) for p in pids)


# --------------------------------------------------------------------------------------------- pure helpers
def test_decode_blocks_spans_and_gaps():
    recs = np.r_[np.arange(0, 100), np.arange(105, 130), np.arange(400, 410)]
    bl = decode_blocks(recs, max_gap=30, block=40)
    pos = np.concatenate([b[2] for b in bl])
    assert np.array_equal(pos, np.arange(len(recs)))                 # every wanted frame exactly once, in order
    for a, n, p in bl:
        assert n <= 40 and recs[p[0]] == a and recs[p[-1]] == a + n - 1
    assert any(a == 400 for a, _, _ in bl)                            # a big gap starts a new block (seek)


def test_workdir_env_and_jobdir_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv('STILLPOINT_WORK_DIR', str(tmp_path / 'w'))
    assert work_root() == str(tmp_path / 'w')
    with JobDir('t') as jd:
        p = jd.path('x.bin')
        open(p, 'wb').write(b'0' * 1000)
        assert jd.usage_bytes() >= 1000 and jd.dir.startswith(str(tmp_path / 'w'))
    assert not os.path.exists(jd.dir)
    # a job dir left behind by a hard-killed process (owner pid gone) is swept by the next job
    dead = subprocess.Popen([PY, '-c', 'pass'])
    dead.wait()
    stale = tmp_path / 'w' / 'jobs' / f'analyze-{dead.pid}-x'
    stale.mkdir(parents=True)
    (stale / 'owner.json').write_text(json.dumps(dict(pid=dead.pid)))
    (stale / 'big.npy').write_bytes(b'0' * 1000)
    old = time.time() - 60
    os.utime(stale, (old, old))
    assert str(stale) in sweep_stale(str(tmp_path / 'w'))
    assert not stale.exists()


def test_preflight_clear_error(tmp_path):
    with pytest.raises(InsufficientDiskSpace) as e:
        preflight(str(tmp_path), need_bytes=10 ** 16, what='analysis')
    assert 'free' in str(e.value) and 'GB' in str(e.value)


# --------------------------------------------------------------------------------------------- frame stream
@needs_clip
def test_framestream_bounded_and_exact():
    fs = FrameStream(CLIP, 960, lanes=3, block=24, capacity=40)
    recs = np.r_[np.arange(0, 70), np.arange(100, 101), np.arange(150, 200)]
    got = list(fs.frames(recs))
    assert [k for k, _ in got] == recs.tolist()
    assert fs.stats['max_buffered'] <= 40                              # the reorder buffer never exceeds capacity
    # the same pictures as the M2 cache's ffmpeg decode (area downscale of the raw Y plane; the persistent PyAV
    # decoder uses OpenCV's area filter, so allow rounding differences)
    from stillpoint.video import _run_decode, probe
    info = probe(CLIP)
    ref = {k: g.copy() for k, _, g in _run_decode(CLIP, info, fs.w, fs.h, 'vt_cpu_scale', 150, 50, True)}
    n_cmp = 0
    for k, g in got:
        if k in ref:
            d = np.abs(g.astype(int) - ref[k].astype(int))
            assert d.mean() < 0.6 and np.percentile(d, 99.9) <= 2, (k, d.mean(), d.max())
            n_cmp += 1
    assert n_cmp == 50
    assert fs.stats['decoder_starts'] <= 3                             # persistent decoders (one per lane)
    got2 = list(fs.frames(np.arange(200, 230)))                         # reused decoders: no new sessions
    assert [k for k, _ in got2] == list(range(200, 230)) and fs.stats['decoder_starts'] <= 3
    pids = fs.decoder_pids()
    fs.close()
    assert _wait_dead(pids, 3.0)                                        # decoder processes exit with the stream


@needs_clip
def test_framestream_close_stops_decoders():
    fs = FrameStream(CLIP, 960, lanes=2, block=30)
    gen = fs.frames(np.arange(0, 250))
    next(gen)
    time.sleep(0.5)
    t = time.time()
    gen.close()                                   # consumer stops early -> lanes stop within ~0.2 s
    assert time.time() - t < 2.0
    time.sleep(0.3)
    assert not [t for t in threading.enumerate() if t.name.startswith('framestream-')]
    pids = fs.decoder_pids()
    fs.close()
    assert _wait_dead(pids, 3.0)
    out = subprocess.run(['pgrep', '-P', str(os.getpid()), 'ffmpeg'], capture_output=True, text=True).stdout.split()
    assert not out


# --------------------------------------------------------------------------------------------- workers
def test_no_orphan_workers_when_parent_is_killed(tmp_path):
    """A hard-killed parent (SIGKILL: no cleanup can run) must not leave spawn_main workers behind."""
    code = textwrap.dedent(f"""
        import os, sys, time, signal
        sys.path.insert(0, {os.path.join(ROOT, 'engine')!r})
        from stillpoint.residual import make_pool, pool_pids
        ex = make_pool(2)
        print(' '.join(map(str, pool_pids(ex))), flush=True)
        time.sleep(0.3)
        os.kill(os.getpid(), signal.SIGKILL)
    """)
    p = subprocess.run([PY, '-c', code], capture_output=True, text=True, timeout=60)
    pids = [int(x) for x in p.stdout.split()]
    assert len(pids) == 2
    assert _wait_dead(pids, 5.0), f'orphaned workers still alive: {[q for q in pids if _alive(q)]}'


def test_shutdown_pool_terminates_busy_workers():
    from stillpoint.residual import make_pool, pool_pids, shutdown_pool, _ping
    ex = make_pool(2)
    pids = pool_pids(ex)
    futs = [ex.submit(_ping, 30.0) for _ in range(2)]                  # busy for 30 s
    time.sleep(0.3)
    t = time.time()
    shutdown_pool(ex)
    assert time.time() - t < 5.0
    assert _wait_dead(pids, 3.0)


# --------------------------------------------------------------------------------------------- analyze()
@needs_clip
def test_analyze_cancel_cleans_up(tmp_path, monkeypatch):
    from stillpoint import residual
    from stillpoint.pipeline import AnalysisCancelled, AnalyzeParams, analyze
    wd = tmp_path / 'work'
    monkeypatch.setenv('STILLPOINT_WORK_DIR', str(wd))
    pids = []
    orig = residual.make_pool

    def spy(n=None):
        ex = orig(n)
        pids.extend(residual.pool_pids(ex))
        return ex
    monkeypatch.setattr(residual, 'make_pool', spy)
    seen = {'t': None}

    def prog(stage, f, msg):
        if stage.startswith('measure0') and 'pairs' in msg and seen['t'] is None:
            seen['t'] = time.time()

    def cancel():
        return seen['t'] is not None
    out = tmp_path / 'out'
    t0 = time.time()
    with pytest.raises(AnalysisCancelled):
        analyze(CLIP, str(out), AnalyzeParams(closed_loop_iters=1, verbose=False, processes=2),
                progress=prog, cancel=cancel)
    assert seen['t'] is not None and time.time() - seen['t'] < 3.0       # cancel honoured quickly
    assert pids and _wait_dead(pids, 3.0)                                # measurement workers stopped
    jobs = wd / 'jobs'
    assert not jobs.exists() or not os.listdir(jobs)                     # job dir removed
    assert not (out / 'plan.spplan').exists()                            # nothing written
    assert time.time() - t0 < 120


@needs_clip
def test_sigterm_stops_and_cleans_up(tmp_path):
    """SIGTERM to a process running analyze(): the cancel path runs (workers stopped, job dir removed)."""
    wd = tmp_path / 'work'
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {os.path.join(ROOT, 'engine')!r})
        from stillpoint import residual
        from stillpoint.pipeline import AnalyzeParams, analyze, AnalysisCancelled
        orig = residual.make_pool
        def spy(n=None):
            ex = orig(n)
            print('PIDS', ' '.join(map(str, residual.pool_pids(ex))), flush=True)
            return ex
        residual.make_pool = spy
        def prog(stage, f, msg):
            if stage.startswith('measure0') and 'pairs' in msg:
                print('MEASURING', flush=True)
        try:
            analyze({CLIP!r}, {str(tmp_path / 'out')!r}, AnalyzeParams(closed_loop_iters=1, verbose=False, processes=2),
                    progress=prog)
        except AnalysisCancelled:
            print('CANCELLED', flush=True)
            sys.exit(143)
    """)
    env = dict(os.environ, STILLPOINT_WORK_DIR=str(wd))
    p = subprocess.Popen([PY, '-c', code], stdout=subprocess.PIPE, text=True, env=env)
    pids = []
    t_sig = None
    for line in p.stdout:
        if line.startswith('PIDS'):
            pids = [int(x) for x in line.split()[1:]]
        if line.startswith('MEASURING') and t_sig is None:
            p.send_signal(signal.SIGTERM)
            t_sig = time.time()
        if line.startswith('CANCELLED'):
            break
    rc = p.wait(timeout=30)
    assert t_sig is not None and rc == 143 and time.time() - t_sig < 10
    assert pids and _wait_dead(pids, 3.0)
    jobs = wd / 'jobs'
    assert not jobs.exists() or not os.listdir(jobs)


@pytest.fixture(scope='module')
def analysis(tmp_path_factory):
    if not os.path.exists(CLIP):
        pytest.skip('footage not available')
    from stillpoint.pipeline import AnalyzeParams, analyze
    base = tmp_path_factory.mktemp('an')
    wd = base / 'work'
    old = os.environ.get('STILLPOINT_WORK_DIR')
    os.environ['STILLPOINT_WORK_DIR'] = str(wd)
    peak = {'disk': 0, 'files_outside': []}
    stop = threading.Event()

    def mon():
        while not stop.is_set():
            tot = 0
            for dp, _, fns in os.walk(wd):
                for f in fns:
                    try:
                        tot += os.path.getsize(os.path.join(dp, f))
                    except OSError:
                        pass
            peak['disk'] = max(peak['disk'], tot)
            time.sleep(0.2)
    th = threading.Thread(target=mon, daemon=True)
    th.start()
    try:
        rep = analyze(CLIP, str(base / 'out'), AnalyzeParams(closed_loop_iters=1, verbose=False, save_iter_plans=False))
    finally:
        stop.set()
        th.join()
        if old is None:
            os.environ.pop('STILLPOINT_WORK_DIR', None)
        else:
            os.environ['STILLPOINT_WORK_DIR'] = old
    return rep, base, peak


def test_analyze_bounded_temp_and_outputs(analysis):
    rep, base, peak = analysis
    assert peak['disk'] < 2 * 1024 ** 3                                  # temp disk bound
    assert sorted(os.listdir(base / 'out')) == ['analysis.npz', 'plan.spplan', 'report.json']
    jobs = base / 'work' / 'jobs'
    assert not jobs.exists() or not os.listdir(jobs)                     # cleaned up
    assert rep['params']['luma_cache'] is False
    assert rep['timings']['decode']['max_buffered'] <= rep['timings']['frame_buffer_mb'] * 1e6 / (960 * 540) + 1


def test_quality_report_fields(analysis):
    rep, base, _ = analysis
    q = json.load(open(base / 'out' / 'report.json'))['quality']
    for k in ('original_hf_px', 'stabilized_hf_px', 'original_calm_hf_px', 'stabilized_calm_hf_px',
              'original_b8_30_px', 'stabilized_b8_30_px', 'vision_trusted_frac', 'crop_footprint_mean', 'method',
              'new_jumps_gt_1px', 'windows', 'coverage_frac'):
        assert k in q, k
    assert 'MotionEstimator' in q['method']
    assert q['original_hf_px'] > 0 and q['stabilized_hf_px'] > 0
    assert 0.0 <= q['vision_trusted_frac'] <= 1.0
    assert 0.3 < q['crop_footprint_mean'] < 1.0
    # default crop (no Gyroflow render, no FOV given): Gyroflow-like footprint
    assert rep['crop']['target_source'] == 'default' and abs(q['crop_footprint_mean'] - 0.603) < 0.01
