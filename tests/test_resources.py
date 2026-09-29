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


# --------------------------------------------------------------------------------------------- memory (MEMORY role)
def test_plan_resources_fits_budget():
    """The worker pool / decoder lanes / buffer are sized so the modelled process-group peak stays within the
    budget for any clip length and resolution; low available RAM shrinks the pool; processes is an upper bound."""
    from stillpoint.pipeline import AnalyzeParams, plan_resources
    prm = AnalyzeParams()
    geoms = [(3840, 2160, 8, 960, 540), (3840, 2880, 10, 960, 720), (2688, 1512, 10, 960, 540)]
    for g in geoms:
        for F in (268, 3029, 30241, 120000):
            for stage in ('measure', 'quality'):
                r = plan_resources(prm, *g, F, stage=stage, avail_bytes=16e9, main_rss=0.4e9, max_workers=9)
                assert 1 <= r['processes'] <= 9 and r['lanes'] in (1, 2)
                if F <= 30241 and (stage == 'quality' or g[2] == 8):
                    assert r['lanes'] == 2 and r['processes'] >= 5, (g, F, stage, r)
                assert r['capacity'] <= prm.decode_block + 96
                if r['processes'] > 1:
                    assert r['est_gb']['total'] <= 4.5 + 1e-9, (g, F, stage, r)
    # 4K O3 clips: 5 measurement workers (the passes saturate at 4-5: the feeding process is the bottleneck);
    # 4:3 10-bit clips with measurement passes get fewer (bigger frames); quality-only OA4 runs keep 8
    o3 = plan_resources(prm, 3840, 2160, 8, 960, 540, 3029, 'measure', 16e9, 0.4e9, 9)
    oa4 = plan_resources(prm, 3840, 2880, 10, 960, 720, 30241, 'measure', 16e9, 0.4e9, 9)
    oa4q = plan_resources(prm, 3840, 2880, 10, 960, 720, 30241, 'quality', 16e9, 0.4e9, 9)
    assert o3['processes'] >= 5 and oa4['processes'] < o3['processes'] and oa4q['processes'] >= 7
    # little RAM available -> smaller pool (and one decoder lane), never more than the budget
    low = plan_resources(prm, 3840, 2160, 8, 960, 540, 3029, 'measure', 2.5e9, 0.4e9, 9)
    assert low['processes'] < o3['processes'] and low['effective_budget_gb'] < 4.5
    # explicit processes = upper bound; budget 0 = the CPU count
    assert plan_resources(AnalyzeParams(processes=2), 3840, 2160, 8, 960, 540, 3029, 'measure', 16e9, 0,
                          9)['processes'] == 2
    assert plan_resources(AnalyzeParams(mem_budget_gb=0), 3840, 2880, 10, 960, 720, 30241, 'quality', 16e9, 0,
                          9)['processes'] == 9
    assert plan_resources(AnalyzeParams(processes=0), 3840, 2160, 8, 960, 540, 3029)['processes'] == 0
    # measurement pools are capped at max_workers_measure (the passes saturate at 5-6 workers); the default worker
    # environment turns off malloc's large-block cache; with the system allocator the model budgets far more per
    # worker (the freed-block cache reached 0.94 GB per worker) -> fewer workers
    assert AnalyzeParams().worker_malloc_env.get('MallocLargeCache') == '0'
    assert o3['processes'] <= AnalyzeParams().max_workers_measure
    sysm = plan_resources(AnalyzeParams(worker_malloc_env={}), 3840, 2160, 8, 960, 540, 3029, 'measure', 16e9, 0.4e9, 9)
    assert sysm['processes'] < o3['processes'] and sysm['est_gb']['total'] <= 4.5


def _synthetic_frames(n, H=180, W=320, seed=0, produced=None):
    """A long textured stream with small random camera motion (fast to produce: the producer outruns a
    one-worker measurement pool by far -> exercises the bounded queues)."""
    import cv2
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur((rng.random((H + 64, W + 64)) * 255).astype(np.float32), (0, 0), 1.2)
    xy = np.cumsum(rng.normal(0, 0.4, (n, 2)), axis=0)
    xy = 32 + np.clip(xy - xy.mean(0), -24, 24)
    for k in range(n):
        M = np.float32([[1, 0, xy[k, 0]], [0, 1, xy[k, 1]]])
        img = cv2.warpAffine(tex, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        if produced is not None:
            produced[0] += 1
        yield k, np.clip(img, 0, 255).astype(np.uint8)


def test_measure_stream_bounded_memory_with_slow_consumer():
    """Bounded memory on a synthetic long stream: previews are produced much faster than one measurement worker
    consumes them. The memory this process holds (prefetch queue + in-flight chunks + context frames) must not
    grow with the stream length -- only the small per-pair results do (a queue that grows when the consumer
    stalls was a candidate for the unreproduced 14 GB spike)."""
    import tracemalloc
    from stillpoint.pipeline import _prefetch
    from stillpoint.residual import make_pool, measure_residuals, shutdown_pool
    H, W = 96, 160
    K = np.array([[130.0, 0, (W - 1) / 2], [0, 130.0, (H - 1) / 2], [0, 0, 1]])
    fb = H * W
    ex = make_pool(1)
    peaks, lags = {}, {}
    try:
        for n in (160, 560):
            produced = [0]
            consumed = [0]
            maxlag = [0]

            def prog(i1, _p=produced, _c=consumed, _m=maxlag):
                _c[0] = i1
                _m[0] = max(_m[0], _p[0] - i1)
            tracemalloc.start()
            try:
                res = measure_residuals(_prefetch(_synthetic_frames(n, H, W, produced=produced), 16), K,
                                        executor=ex, consecutive_only=True, progress=prog)
                _, peaks[n] = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            lags[n] = maxlag[0]
            assert len(res['k0']) == n - 1 and np.median(res['conf']) > 0.2
        # frames buffered ahead of the measurement: prefetch (16) + in-flight chunks (2 x 57) + context, whatever the
        # stream length (an unbounded queue would hold ~all 560 frames here: the producer is ~50x faster)
        assert lags[560] <= 16 + 2 * 57 + 64, lags
        assert lags[560] <= lags[160] + 64, lags
        # traced memory of this process: bounded in frames; only the per-pair result dicts (~3 KB) grow
        assert peaks[560] < peaks[160] + 400 * 4e3 + 40 * fb, (peaks, fb)
        assert peaks[560] < (16 + 2 * 2 * 57 + 64) * fb + 8e6, (peaks, fb)
    finally:
        shutdown_pool(ex)


def _synthetic_telemetry(F=600, fps=59.94, seed=1):
    from stillpoint.geom import Lens, qexp, qmul
    from stillpoint.types import Telemetry
    rng = np.random.default_rng(seed)
    rate = 2000.0
    N = int((F / fps + 1.0) * rate)
    t = np.arange(N) / rate - 0.5
    w = np.cumsum(rng.normal(0, 0.02, (N, 3)), axis=0) * 0.2 + rng.normal(0, 0.3, (N, 3))   # rad/s, drifting + jitter
    q = np.zeros((N, 4))
    q[0] = [1, 0, 0, 0]
    for i in range(1, N):
        q[i] = qmul(q[i - 1], qexp(w[i] / rate))
    pts = np.arange(F) / fps
    lens = Lens('kb4', 1405.13, 1405.13, 1919.5, 1079.5, np.array([0.2499, 0.0136, -0.0621, 0.0122]), 3840, 2160)
    return Telemetry(source='synthetic', camera='synthetic', width=3840, height=2160, fps=fps, frame_pts=pts,
                     frame_t=pts + 0.004, exposure_s=np.full(F, 0.002), readout_s=0.0097, lens=lens, imu_t=t,
                     imu_q=q, imu_rate=rate, has_highrate=True, eis_baked=False, segments=[(0, F - 1)])


def test_path_solver_child_matches_in_process_and_exits():
    """optimize_path in the short-lived solver process (long clips: its memory goes back to the OS on exit) gives
    the in-process result; progress ticks arrive; a cancel kills the child promptly."""
    from stillpoint.pipeline import AnalysisCancelled, _PathSolver, _qfn
    from stillpoint.smooth import SmoothParams, optimize_path
    from stillpoint.types import TimeModel
    tel = _synthetic_telemetry()
    tm = TimeModel()
    frames = np.arange(tel.n_frames)
    q = _qfn(tel, tm)
    sp = SmoothParams(min_out_fx=1600.0, max_out_fx=2000.0, window=1500, sqp_iters=2, repair_iters=0)
    v0, fx0, _ = optimize_path(tel, q, frames, 3840, 2160, sp, tm=tm, return_info=True)
    ticks = []
    sp.tick = ticks.append
    ps = _PathSolver(tel, tm, frames, 3840, 2160)
    try:
        v1, fx1, info1 = ps.solve(q.recipe[1], sp, lambda: None)
        pid = ps.pid()
        assert np.allclose(v0, v1, atol=1e-9) and np.allclose(fx0, fx1) and 'runtime_s' in info1
        assert ticks and max(ticks) <= 1.0
        assert _alive(pid)
    finally:
        ps.close()
    assert _wait_dead([pid], 3.0)
    # cancel while solving: check() raises -> the caller closes the solver (kills the child)
    ps = _PathSolver(tel, tm, frames, 3840, 2160)
    t_c = time.time() + 1.0

    def check():
        if time.time() > t_c:
            raise AnalysisCancelled('cancel')
    pid = ps.pid()
    sp.sqp_iters, sp.tick = 40, None
    try:
        with pytest.raises(AnalysisCancelled):
            ps.solve(None, sp, check)
    finally:
        t0 = time.time()
        ps.close()
    assert time.time() - t0 < 3.0 and _wait_dead([pid], 3.0)


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
        # path_proc_frames=0: the camera path is solved in the short-lived solver process (long-clip mode)
        rep = analyze(CLIP, str(base / 'out'), AnalyzeParams(closed_loop_iters=1, verbose=False, save_iter_plans=False,
                                                             processes=4, path_proc_frames=0))
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


def test_analyze_memory_plan_and_rss_report(analysis):
    """The resource plan (2 decoder lanes, bounded buffer, processes as an upper bound), the path-solver process
    and the process-group RSS watch are in the report; the whole group stayed within the 4.5 GB budget."""
    rep, base, _ = analysis
    r = rep['resources']
    assert r['plan']['lanes'] == 2 and r['plan']['processes'] <= 4 and r['plan']['capacity'] <= 150 + 96
    assert r['pool_workers'] == r['plan']['processes']
    assert rep['timings']['decode']['decoder_starts'] <= 2
    assert r['path_solver']['process'] and r['path_solver']['solves'] >= 2 and r['path_solver']['error'] is None
    assert r['rss']['samples'] >= 5 and 0.2 < r['rss']['peak_total_gb'] < 4.5, r['rss']


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
