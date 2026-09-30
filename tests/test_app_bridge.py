"""App bridge (JSON-lines front end for Stillpoint.app).  PYTHONPATH=engine .venv/bin/python -m pytest tests/test_app_bridge.py -q

Unit tests run anywhere; the integration tests use the short O3 clip DJI_0026 (4.5 s) and are skipped without it.
"""
import json
import math
import os
import signal
import subprocess
import sys
import time
import types

import numpy as np
import pytest

from stillpoint import app_bridge as ab
from eval.footage import o3  # noqa: E402  (env-configurable, see eval/footage.py)

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
CLIP = o3('DJI_0026.MP4')
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'engine'))
needs_clip = pytest.mark.skipif(not os.path.exists(CLIP), reason='DJI_0026 not available')


def run_bridge(args, env=None, **kw):
    return subprocess.run([PY, '-m', 'stillpoint.app_bridge', *args], capture_output=True, text=True, env=env or ENV,
                          cwd=ROOT, stdin=subprocess.DEVNULL, **kw)


def events(stdout):
    ev = [json.loads(line) for line in stdout.splitlines() if line.strip()]   # every stdout line must be JSON
    assert all('type' in e for e in ev)
    return ev


# ---------------------------------------------------------------------------------------------- unit


def test_normalize_native_progress_shapes():
    n = ab._normalize_native
    assert n(({'stage': 'measure', 'fraction': 0.5, 'message': 'x'},), {}) == ('measure', 0.5, 'x')
    assert n(('path', 0.25), {}) == ('path', 0.25, None)
    assert n(('path', 0.25, 'solving'), {}) == ('path', 0.25, 'solving')
    assert n((0.75,), {}) == (None, 0.75, None)
    assert n((), dict(stage='telemetry', fraction=0.1, message='m')) == ('telemetry', 0.1, 'm')
    assert n((np.float64(0.3),), {}) == (None, 0.3, None)


def test_progress_weighted_monotonic_and_eta(monkeypatch):
    out = []
    monkeypatch.setattr(ab, 'emit', out.append)
    p = ab.Progress([('telemetry', 'T', 1.0), ('path', 'P', 1.0), ('measure', 'M', 8.0), ('path', 'P', 1.0),
                     ('finalize', 'F', 1.0)], min_interval=0.0)
    p.start('telemetry')
    p.start('telemetry')              # re-announcing the current stage must not skip ahead
    assert p.i == 0
    p.update(1.0)
    p.start('path')
    p.update(1.0)
    p.start('measure')
    p.t0 -= 10.0                      # pretend 10 s elapsed
    p.update(0.5)
    fr = [e['fraction'] for e in out]
    assert fr == sorted(fr), 'overall fraction must never go backwards'
    assert abs(out[-1]['fraction'] - (2 + 4) / 12) < 1e-3
    assert out[-1]['eta_s'] is not None and out[-1]['eta_s'] > 0
    p.start('finalize')               # skipping the second 'path' is allowed (early stop)
    assert out[-1]['stage'] == 'finalize' and out[-1]['fraction'] >= fr[-1]
    p.finish()
    assert out[-1]['fraction'] == 1.0 and out[-1]['eta_s'] == 0.0


def test_native_stage_mapping_and_messages(monkeypatch):
    """pipeline.analyze(progress=(stage, overall, msg)) stage names -> the app's four steps, friendly messages."""
    assert ab.map_native_stage('measure1') == ('measure', 'Measuring jitter')
    assert ab.map_native_stage('fold2')[0] == 'path' and ab.map_native_stage('crop')[0] == 'path'
    assert ab.map_native_stage('calibration')[0] == 'telemetry' and ab.map_native_stage('done')[0] == 'finalize'
    assert ab.map_native_stage('something_new') == ('something_new', 'Something new')
    assert ab.map_native_stage('quality') == ('quality', 'Checking the result')
    out = []
    monkeypatch.setattr(ab, 'emit', out.append)
    prog = ab.Progress([('telemetry', 'T', 1.0), ('finalize', 'F', 1.0)], min_interval=0.0)
    prog.start('telemetry')
    cb = ab._NativeProgress(prog, n_passes=3)
    cb('crop', 0.05, 'out_fx 1611.1')
    cb('measure0', 0.30, 'measuring pass 0: 64/267 pairs')
    cb('measure0', 0.20, 'measuring pass 0: 128/267 pairs')      # a lower overall fraction must not go backwards
    cb('fold1', 0.52, 'fold 1')
    assert out[-3]['message'] == 'pass 1 of 3 · 64/267 frames'
    assert out[-2]['fraction'] >= out[-3]['fraction']
    assert out[-1]['stage'] == 'path' and out[-1]['message'] == 'Folding measured jitter back into the path'
    assert [e['fraction'] for e in out] == sorted(e['fraction'] for e in out)
    prog.start('finalize')
    prog.finish()
    assert out[-1]['fraction'] == 1.0


def test_stage_plan_counts():
    plan = ab._stage_plan(3000, 2, cached_tel=True)
    keys = [k for k, _, _ in plan]
    assert keys.count('measure') == 3 and keys.count('path') == 4
    assert keys[0] == 'telemetry' and keys[-1] == 'finalize'
    w = dict(measure=0, path=0)
    for k, _, s in plan:
        if k in w:
            w[k] += s
    assert w['measure'] > 5 * w['path']        # the vision passes dominate, as measured in M1


def test_sprender_progress_parsing():
    assert ab.parse_sprender_line('PROGRESS frame=120 total=3029') == (120, 3029)
    assert ab.parse_sprender_line('PROGRESS frame=0 total=0') is None
    assert ab.parse_sprender_line('progress 5/10') == (5, 10)
    assert ab.parse_sprender_line('frames: 7/9') == (7, 9)
    assert ab.parse_sprender_line('input 3840x2160 59.94 fps codec=avc1 8-bit') is None
    assert ab.parse_sprender_line('RESULT frames=268 wall=6.9s fps=41.6') is None


def test_expected_bytes():
    assert abs(ab._expected_bytes('hevc10', 180, 3840, 2160, 59.94, 5994) - 180e6 / 8 * 100) < 1
    pr = ab._expected_bytes('prores', 180, 3840, 2160, 60000 / 1001, 60)
    assert 1.6e9 / 8 < pr < 1.9e9 / 8              # ~1.77 Gbit/s ProRes 422 HQ at 2160p59.94


def test_fov_roundtrip():
    for deg in (80.0, 100.0, 112.5):
        assert abs(ab._fov_deg(3840, ab._fx_for_fov(3840, deg)) - deg) < 1e-9
    assert abs(ab._fx_for_fov(3840, 90.0) - 1920.0) < 1e-9


def test_gyro_status_labels():
    def tel(rate, hi, eis=False, name='EIS_OFF'):
        return types.SimpleNamespace(imu_rate=rate, has_highrate=hi, eis_baked=eis, extra=dict(eis_status_name=name))
    assert ab.gyro_status(tel(1999.0, True)) == dict(label='2 kHz gyro', level='good',
                                                     detail='1999 Hz orientation samples, EIS off.')
    assert ab.gyro_status(tel(1000.0, True))['label'] == '1 kHz gyro'
    g = ab.gyro_status(tel(60.0, False))
    assert g['label'] == '60 Hz attitude — limited' and g['level'] == 'limited'
    g = ab.gyro_status(tel(60.0, False, True, 'EIS_TRADEOFF'))
    assert g['label'] == 'In-camera EIS on — not supported yet' and g['level'] == 'unsupported'
    assert 'tradeoff' in g['detail']
    assert ab.gyro_status(None)['level'] == 'unsupported'


def test_json_cleaning():
    assert ab._clean({'a': float('nan'), 'b': [1.0, float('inf')], 'c': np.float32(2.5)}) == \
        {'a': None, 'b': [1.0, None], 'c': 2.5}


def test_cancel_token_shapes():
    t = ab.CancelToken()
    assert not t() and not t.is_set() and not bool(t)
    t.check()
    t.set()
    assert t() and t.cancelled and bool(t)
    with pytest.raises(ab.Cancelled):
        t.raise_if_cancelled()


def test_analyze_feature_detection_matches_pipeline():
    import inspect
    from stillpoint import pipeline
    ps = inspect.signature(pipeline.analyze).parameters
    assert ab._analyze_supports() == ('progress' in ps, 'cancel' in ps)


# ---------------------------------------------------------------------------------------------- integration


@needs_clip
def test_probe_protocol_and_fields():
    r = run_bridge(['probe', CLIP])
    assert r.returncode == 0, r.stderr[-2000:]
    ev = events(r.stdout)
    assert ev[0]['type'] == 'start' and ev[-1]['type'] == 'result'
    p = ev[-1]
    assert (p['width'], p['height'], p['n_frames']) == (3840, 2160, 268)
    assert p['camera'].startswith('DJI O3') and p['gyro']['label'] == '2 kHz gyro' and p['supported']
    assert p['fov']['min_deg'] < p['fov']['default_deg'] <= p['fov']['max_deg']
    assert abs(p['duration_s'] - 268 * 1001 / 60000) < 0.01


def test_probe_missing_file_is_json_error():
    r = run_bridge(['probe', '/nonexistent/clip.mp4'])
    ev = events(r.stdout)
    assert r.returncode == 1 and ev[-1]['type'] == 'error' and ev[-1]['code'] == 'not_found'


@needs_clip
def test_analyze_end_to_end(tmp_path):
    out = tmp_path / 'a'
    r = run_bridge(['analyze', CLIP, '--out', str(out), '--loop-iters', '1', '--fov', '100'], timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    ev = events(r.stdout)
    prog = [e for e in ev if e['type'] == 'progress']
    fr = [e['fraction'] for e in prog]
    assert fr == sorted(fr) and fr[-1] == 1.0 and len(prog) >= 5
    assert {'telemetry', 'path', 'measure', 'finalize'} <= {e['stage'] for e in prog}
    assert any(e['type'] == 'log' for e in ev)
    res = ev[-1]
    assert res['type'] == 'result'
    m = json.load(open(out / 'stillpoint_app.json'))
    st = os.stat(CLIP)
    assert m['clip']['size_bytes'] == st.st_size and m['clip']['mtime_ns'] == st.st_mtime_ns
    assert m['params']['fov_deg'] == 100.0 and m['params']['loop_iters'] == 1
    s = m['summary']
    assert abs(s['hfov_deg'] - 100.0) < 0.01
    assert s['orig_hf_px'] > 0 and s['final_px'] is not None and s['final_px'] >= 0
    assert len(s['win_orig_px']) == len(s['win_final_px']) == 4
    assert os.path.exists(m['plan']) and os.path.exists(out / 'report.json')
    assert not [d for d in os.listdir(out) if d.startswith('.work-')]
    from stillpoint.plan_io import read_plan
    pl = read_plan(m['plan'])
    assert pl.n_frames == 268 and abs(float(pl.out_fx.min()) - ab._fx_for_fov(3840, 100.0)) < 0.5


@needs_clip
@pytest.mark.parametrize('how', ['sigterm', 'stdin'])
def test_analyze_cancel_leaves_no_plan(tmp_path, how):
    out = tmp_path / 'c'
    p = subprocess.Popen([PY, '-m', 'stillpoint.app_bridge', 'analyze', CLIP, '--out', str(out)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         env=ENV, cwd=ROOT)
    lines = []
    t0 = time.time()
    for line in p.stdout:                       # wait until the vision loop is running
        lines.append(line)
        e = json.loads(line)
        if e['type'] == 'progress' and e['stage'] == 'measure' and e['fraction'] > 0.08:
            break
        assert time.time() - t0 < 120
    if how == 'sigterm':
        p.send_signal(signal.SIGTERM)
    else:
        p.stdin.close()                          # what the app's pipe does when the app goes away
    rest = p.stdout.read()
    p.wait(timeout=60)
    ev = [json.loads(x) for x in lines + rest.splitlines() if x.strip()]
    assert p.returncode == 130
    assert ev[-1]['type'] == 'cancelled'
    assert not any(e['type'] == 'result' for e in ev)
    assert not (out / 'stillpoint_app.json').exists() and not (out / 'plan.spplan').exists()
    assert not [d for d in os.listdir(out) if d.startswith('.work-')]


@needs_clip
def test_render_short_window_and_cancel(tmp_path):
    plan_dir = tmp_path / 'p'
    r = run_bridge(['analyze', CLIP, '--out', str(plan_dir), '--loop-iters', '0'], timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    plan = str(plan_dir / 'plan.spplan')
    out = tmp_path / 'r.mov'
    r = run_bridge(['render', CLIP, '--plan', plan, '--out', str(out), '--start-frame', '30', '--frames', '40',
                    '--codec', 'hevc10-speed'], timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    ev = events(r.stdout)
    assert ev[-1]['type'] == 'result' and ev[-1]['frames'] == 40 and ev[-1]['bytes'] == os.path.getsize(out)
    fr = [e['fraction'] for e in ev if e['type'] == 'progress']
    assert fr == sorted(fr) and fr[-1] == 1.0
    assert not [f for f in os.listdir(tmp_path) if '.partial' in f]
    n = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                        'stream=nb_read_packets,profile', '-of', 'csv=p=0', str(out)], capture_output=True, text=True).stdout
    assert '40' in n and 'Main 10' in n
    # cancel mid-render: no output, no partial file
    out2 = tmp_path / 'c.mov'
    p = subprocess.Popen([PY, '-m', 'stillpoint.app_bridge', 'render', CLIP, '--plan', plan, '--out', str(out2),
                          '--codec', 'prores'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, env=ENV, cwd=ROOT)
    for line in p.stdout:
        e = json.loads(line)
        if e['type'] == 'progress' and e['fraction'] > 0.05:
            break
        if e['type'] in ('result', 'error'):
            pytest.skip('render finished before it could be cancelled')
    p.send_signal(signal.SIGTERM)
    rest, _ = p.communicate(timeout=60)
    assert p.returncode == 130 and json.loads(rest.splitlines()[-1])['type'] == 'cancelled'
    assert not out2.exists() and not [f for f in os.listdir(tmp_path) if f.startswith('.c.mov')]


def test_cache_dir_matches_swift():
    """analysis_dir_for mirrors EngineConfig.analysisDir (FNV-1a-64 of the absolute path, first 10 hex digits)."""
    d = ab.analysis_dir_for('/Users/x/Desktop/untitled folder 4/DJI_0034.MP4', '/A/analyses')
    h = 0xcbf29ce484222325
    for b in b'/Users/x/Desktop/untitled folder 4/DJI_0034.MP4':
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    assert d == '/A/analyses/DJI_0034-' + f'{h:016x}'[:10]
    # the value Swift's EngineConfig.analysisDir computes for this path (its cache dir name)
    assert ab.analysis_dir_for('/Users/pilot/Movies/FPV/DJI_0034.MP4', '/A').endswith('DJI_0034-87d1c37c3f')


# ---------------------------------------------------------------------------------------------- app v2: scratch, disk,
#                                                                                                  honest numbers


def test_quality_normalization_shapes():
    nested = dict(method='eval.jitter_metrics on the rendered output', units='px @1080p',
                  original=dict(hf_px=1.2, calm_cruise_px=0.41, b8_30_px=0.3),
                  stabilized=dict(hf_px=0.21, calm_cruise_px=0.12, b8_30_px=0.08),
                  windows=dict(original=[1.0, 1.4], stabilized=[0.2, 0.22]), window_s=1.0)
    q = ab.normalize_quality(nested)
    assert q['method'].startswith('eval.jitter_metrics') and q['units'] == 'px @1080p'
    assert [m['key'] for m in q['metrics']] == ['hf', 'calm', 'b8_30']
    assert q['metrics'][1] == dict(key='calm', label='Calm cruise', original=0.41, stabilized=0.12)
    assert q['windows'] == [dict(t0_s=0.0, t1_s=1.0, original=1.0, stabilized=0.2),
                            dict(t0_s=1.0, t1_s=2.0, original=1.4, stabilized=0.22)]
    by_metric = dict(method='m', hf=dict(original=0.9, stabilized=0.3), calm_cruise=dict(orig=0.5, stab=0.2))
    q = ab.normalize_quality(by_metric)
    assert [(m['key'], m['original'], m['stabilized']) for m in q['metrics']] == [('hf', 0.9, 0.3), ('calm', 0.5, 0.2)]
    flat = dict(orig_hf_px=0.8, stab_hf_px=0.25, b8_30_px_original=0.2, b8_30_px_stabilized=float('nan'))
    q = ab.normalize_quality(flat)
    assert q['method'] is None and q['metrics'][0]['stabilized'] == 0.25
    assert q['metrics'][1] == dict(key='b8_30', label='Fine jitter 8–30 Hz', original=0.2, stabilized=None)
    assert ab.normalize_quality(None) is None and ab.normalize_quality({'foo': 1}) is None
    assert ab.normalize_quality(dict(original=dict(hf_px=True))) is None      # bools are not numbers
    assert ab.normalize_quality(dict(error='skipped', vision_trusted_frac=1.0)) is None


def test_quality_normalization_engine_v3_layout():
    """The layout pipeline.analyze writes (ENGINE v3, quality.aggregate + quality_report)."""
    q = dict(original_hf_px=0.1001, stabilized_hf_px=0.1128, original_b8_30_px=0.017, stabilized_b8_30_px=0.039,
             calm_frac=1.0, original_calm_hf_px=0.1, stabilized_calm_hf_px=0.11, hf_reduction=-0.127,
             original_jello_px=0.108, stabilized_jello_px=0.115, new_jumps_gt_0_5px=1, new_jumps_gt_1px=0,
             new_jumps_max_px=0.602, new_jumps=[dict(t_s=3.55, px=0.602)], method='eval.jitter_metrics ...',
             units='px @1080p-eq of the output', windows=[dict(t0_s=0.234, t1_s=4.238, original_hf_px=0.1001,
                                                               stabilized_hf_px=0.1128)],
             window_s=4.0, coverage_frac=0.8955, vision_trusted_frac=1.0, crop_footprint_mean=0.588,
             closed_loop_self_estimate_px=dict(open=0.09, final=0.034, note='...'))
    n = ab.normalize_quality(q)
    assert [(m['key'], m['original'], m['stabilized']) for m in n['metrics']] == [
        ('hf', 0.1001, 0.1128), ('calm', 0.1, 0.11), ('b8_30', 0.017, 0.039), ('jello', 0.108, 0.115)]
    assert n['windows'] == [dict(t0_s=0.234, t1_s=4.238, original=0.1001, stabilized=0.1128)]
    assert n['new_jumps_gt_1px'] == 0 and n['new_jumps_max_px'] == 0.602 and n['coverage_frac'] == 0.8955
    assert 'hf_reduction' not in n and 'closed_loop_self_estimate_px' not in n     # no derived / self-graded numbers


def test_summary_labels_self_measured(monkeypatch):
    """Without report['quality'] the summary says the Stillpoint number is the closed loop grading itself."""
    import types as _t
    tel = _t.SimpleNamespace(fps=59.94, n_frames=10, width=3840, height=2160, camera='DJI O3', frame_t=np.zeros(10))
    monkeypatch.setattr('stillpoint.telemetry.load_telemetry', lambda *a, **k: tel)
    rep = dict(out=dict(w=3840, h=2160, min_out_fx=1611.0, hfov_deg=100.0), closed_loop=dict(composite_window_hf_px=0.03))
    s = ab.summarize('/nonexistent.mp4', '/nonexistent', rep)
    assert s['final_method'] == 'closed-loop residual (self-measured)' and 'quality' not in s and 'reduction' not in s
    rep['quality'] = dict(method='independent', original=dict(hf_px=0.5), stabilized=dict(hf_px=0.2))
    s = ab.summarize('/nonexistent.mp4', '/nonexistent', rep)
    assert s['quality']['metrics'][0] == dict(key='hf', label='Shake above 2 Hz', original=0.5, stabilized=0.2)
    rep['quality'] = dict(something_else=1.0)
    s = ab.summarize('/nonexistent.mp4', '/nonexistent', rep)
    assert 'quality' not in s and s['quality_unparsed_keys'] == ['something_else']


def test_work_dir_env(monkeypatch, tmp_path):
    monkeypatch.delenv(ab.WORK_ENV, raising=False)
    assert ab.work_root() is None
    assert ab.tmp_dir_for('/x/out', 42) == '/x/out/.work-42'
    assert ab.tel_cache_dir() == os.path.join(ab.ROOT, 'work', 'cache')
    monkeypatch.setenv(ab.WORK_ENV, str(tmp_path / 'w'))
    assert ab.work_root() == str(tmp_path / 'w')
    assert ab.tmp_dir_for('/x/out', 42) == str(tmp_path / 'w' / 'tmp' / 'analyze-42')
    assert ab.tel_cache_dir() == str(tmp_path / 'w' / 'cache')


def test_sweep_stale_tmp(tmp_path):
    dead = subprocess.Popen(['/usr/bin/true'])
    dead.wait()
    base = tmp_path / 'tmp'
    for n in (f'analyze-{dead.pid}', f'analyze-{os.getpid()}', f'analyze-{os.getppid()}', 'other'):
        (base / n).mkdir(parents=True)
    (base / f'analyze-{dead.pid}' / '.luma_cache.npy').write_bytes(b'x' * 10)
    removed = ab.sweep_stale_tmp(str(tmp_path))
    assert removed == [f'analyze-{dead.pid}']
    assert sorted(os.listdir(base)) == sorted([f'analyze-{os.getpid()}', f'analyze-{os.getppid()}', 'other'])


def test_frame_cache_size_and_decision():
    # 6.7-min 4K 4:3 Osmo clip at 59.94: the 16.7 GB cache that filled Jimmy's disk
    assert abs(ab.frame_cache_bytes(24120, 3840, 2880, 960) / 1e9 - 16.67) < 0.01
    assert ab.frame_cache_bytes(268, 3840, 2160, 960) == 268 * 960 * 540 + 128
    assert ab.decide_frame_cache(int(2e9), int(15e9)) is True
    assert ab.decide_frame_cache(int(16.7e9), int(15e9)) is False
    assert ab.decide_frame_cache(int(1e9), int(3.9e9)) is False        # would leave < 3 GB


def _analyze_args(out, **kw):
    d = dict(clip=CLIP, out=str(out), smoothness=1.0, fov=100.0, crop_area=None, horizon_lock=False, loop_iters=0)
    d.update(kw)
    return types.SimpleNamespace(**d)


@needs_clip
def test_analyze_refuses_when_disk_nearly_full(monkeypatch, tmp_path):
    monkeypatch.setenv(ab.WORK_ENV, str(tmp_path / 'w'))
    monkeypatch.setattr(ab, 'emit', lambda e: None)
    monkeypatch.setattr(ab, 'free_bytes', lambda p: int(2.5e9))
    with pytest.raises(ab.BridgeError) as ei:
        ab.cmd_analyze(_analyze_args(tmp_path / 'o'))
    need = ab.engine_scratch_need()[1]
    assert ei.value.code == 'disk_full' and f'{need / 1e9:.1f} GB' in ei.value.message and need >= 3e9
    assert os.listdir(tmp_path / 'w' / 'tmp') == []                       # nothing left behind


@needs_clip
def test_analyze_scratch_and_frame_cache_policy(monkeypatch, tmp_path):
    """Temporaries go to $STILLPOINT_WORK_DIR/tmp; a whole-clip frame cache (old engines) is only used when it
    leaves MIN_FREE_BYTES free (with a notice otherwise); ENGINE v3 has none; an existing gyro parse is reused."""
    from stillpoint import pipeline
    monkeypatch.setenv(ab.WORK_ENV, str(tmp_path / 'w'))
    ev = []
    monkeypatch.setattr(ab, 'emit', ev.append)
    seen = {}

    def fake_analyze(video, out_dir, prm, **kw):
        seen.update(out_dir=out_dir, luma_cache=getattr(prm, 'luma_cache', None),
                    telemetry_cache=getattr(prm, 'telemetry_cache', None))
        raise ab.BridgeError('stop', 'stop here')
    monkeypatch.setattr(pipeline, 'analyze', fake_analyze)
    monkeypatch.setattr(ab, 'free_bytes', lambda p: int(50e9))
    with pytest.raises(ab.BridgeError):
        ab.cmd_analyze(_analyze_args(tmp_path / 'o'))
    assert seen['out_dir'] == str(tmp_path / 'w' / 'tmp' / f'analyze-{os.getpid()}')
    assert os.listdir(tmp_path / 'w' / 'tmp') == []
    fc = ab._engine_frame_cache()
    if fc is None or not fc[0]:                                            # ENGINE v3: streamed frames, no cache
        assert not seen['luma_cache'] and not any(e.get('type') == 'notice' for e in ev)
    else:
        assert seen['luma_cache'] is True
        need = ab.frame_cache_bytes(268, 3840, 2160, fc[1])
        monkeypatch.setattr(ab, 'free_bytes', lambda p: int(ab.engine_scratch_need()[1] + need - 1))
        with pytest.raises(ab.BridgeError):
            ab.cmd_analyze(_analyze_args(tmp_path / 'o'))
        assert seen['luma_cache'] is False
        assert any(e.get('type') == 'notice' and e.get('code') == 'no_frame_cache' for e in ev)
    from stillpoint.telemetry import cache_file
    legacy = os.path.join(ab.ROOT, 'work', 'cache')
    if hasattr(pipeline.AnalyzeParams(), 'telemetry_cache') and os.path.exists(cache_file(CLIP, legacy)):
        assert seen['telemetry_cache'] == legacy


@needs_clip
def test_probe_reports_scratch_need():
    r = run_bridge(['probe', CLIP])
    p = events(r.stdout)[-1]
    temp, need = ab.engine_scratch_need()
    fc = ab._engine_frame_cache()
    assert p['scratch']['frame_cache_bytes'] == (268 * 960 * 540 + 128 if fc and fc[0] else 0)
    assert p['scratch']['temp_bytes'] == max(temp, p['scratch']['frame_cache_bytes'])
    assert p['scratch']['min_free_bytes'] == need >= int(3e9)


@needs_clip
def test_analyze_with_work_dir_env_keeps_scratch_out_of_out_dir(tmp_path):
    out, wd = tmp_path / 'a', tmp_path / 'w'
    env = dict(ENV, **{ab.WORK_ENV: str(wd)})
    p = subprocess.Popen([PY, '-m', 'stillpoint.app_bridge', 'analyze', CLIP, '--out', str(out), '--loop-iters', '0'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         env=env, cwd=ROOT)
    seen_tmp = False
    lines = []
    for line in p.stdout:
        lines.append(json.loads(line))
        if not seen_tmp and (wd / 'tmp').exists():
            seen_tmp = bool(os.listdir(wd / 'tmp'))
    p.wait(timeout=300)
    assert p.returncode == 0 and lines[0]['work_root'] == str(wd) and lines[0]['pgid'] == lines[0]['pid']
    assert seen_tmp, 'the run should have used $STILLPOINT_WORK_DIR/tmp'
    assert os.listdir(wd / 'tmp') == [] and sorted(os.listdir(out)) == ['analysis.npz', 'plan.spplan', 'report.json',
                                                                        'stillpoint_app.json']
    m = json.load(open(out / 'stillpoint_app.json'))
    fc = ab._engine_frame_cache()
    assert m['timing']['frame_cache'] is bool(fc and fc[0]) and m['timing']['n_frames'] == 268 and m['timing']['wall_s'] > 0
    assert m['summary']['final_method'] == 'closed-loop residual (self-measured)'
    rep = json.load(open(out / 'report.json'))
    if isinstance(rep.get('quality'), dict) and 'original_hf_px' in rep['quality']:      # ENGINE v3 independent check
        hf = m['summary']['quality']['metrics'][0]
        assert hf['key'] == 'hf' and hf['original'] == rep['quality']['original_hf_px']
        assert hf['stabilized'] == rep['quality']['stabilized_hf_px']
    big = [os.path.join(dp, f) for dp, _, fs in os.walk(tmp_path) for f in fs
           if os.path.getsize(os.path.join(dp, f)) > 50e6]
    assert not big, big
    for sub in ('tmp', 'jobs'):
        if (wd / sub).exists():
            assert os.listdir(wd / sub) == [], sub


# ---------------------------------------------------------------------------------------------- engine v5 options
#                                                                                                  (app6, 2026-09-30)


def test_option_defaults_come_from_the_engine(monkeypatch):
    """The app's option defaults are AnalyzeParams' own, overridden per camera only by CAMERA_OPTION_DEFAULTS."""
    from stillpoint.pipeline import AnalyzeParams
    p = AnalyzeParams()
    d = ab.option_defaults('DJI Osmo Action 4')                  # no per-camera entry: the engine's own defaults
    assert d['fill'] is bool(p.fill) and d['mesh'] is bool(p.mesh_residual) and d['timecal'] is bool(p.timecal)
    assert d['horizon_lock'] == float(p.horizon_lock) and d['roll_limit_deg'] == float(p.roll_limit_deg)
    assert d['horizon_strength'] == (d['horizon_lock'] or ab.HORIZON_STRENGTH_ON)
    assert d['fill_overscan'] == (float(p.fill_overscan) or ab.FILL_OVERSCAN_ON)
    monkeypatch.setattr(ab, 'CAMERA_OPTION_DEFAULTS', {'DJI O3': {'fill': True, 'horizon_lock': 0.7, 'roll_limit_deg': 12}})
    o3, o4 = ab.option_defaults('DJI O3 (FC8383)'), ab.option_defaults('DJI O4 Pro')
    assert o3['fill'] is True and o3['horizon_lock'] == 0.7 and o3['horizon_strength'] == 0.7 and o3['roll_limit_deg'] == 12
    assert o4 == d                                               # other cameras keep the engine defaults
    assert ab.option_defaults(None) == d


def test_camera_defaults_follow_the_gate():
    """gate v6 (work/gate/v6/decision.md): fill on for the DJI O3 only; mesh and horizon lock off everywhere."""
    o3 = ab.option_defaults('DJI O3 (FC8383)')
    assert o3['fill'] is True and o3['fill_overscan'] == 0.06 and o3['mesh'] is False and o3['horizon_lock'] == 0.0
    for cam in ('DJI Osmo Action 4', 'DJI O4 Pro', None):
        d = ab.option_defaults(cam)
        assert d['fill'] is False and d['mesh'] is False and d['horizon_lock'] == 0.0 and d['timecal'] is True
    r = ab.resolve_options(types.SimpleNamespace(), 'DJI O3 (FC8383)')
    assert (r['fill'], r['fill_overscan'], r['mesh']) == (True, 0.06, False)
    r = ab.resolve_options(types.SimpleNamespace(fill=False, mesh=True), 'DJI O3 (FC8383)')    # Max quality on O3
    assert (r['fill'], r['fill_overscan'], r['mesh']) == (False, 0.0, True)
    # mesh costs ~26-34 ms/frame: x1.3-1.5 on the O3's heavy default, x5-7 on OA4 / O4 Pro
    assert ab.time_factors('DJI O3 (FC8383)')['mesh'] < 2 < ab.time_factors('DJI Osmo Action 4')['mesh']
    assert ab.time_factors('DJI O4 Pro')['mesh'] > 4


def test_cli_analyze_uses_the_camera_defaults(monkeypatch):
    """`stillpoint.cli analyze` resolves options not given exactly like the app (app_bridge.resolve_for_video)."""
    from stillpoint import cli, pipeline
    seen = []

    class Stop(Exception):
        pass

    def fake(video, out, prm, progress=None):
        seen.append(prm)
        raise Stop
    monkeypatch.setattr(pipeline, 'analyze', fake)
    monkeypatch.setattr(ab, '_camera_for', lambda v: 'DJI O3 (FC8383)')
    for argv, want in ((['analyze', 'x.mp4', '--out', 'o', '-q'], (True, 0.06, False)),
                       (['analyze', 'x.mp4', '--out', 'o', '-q', '--no-fill', '--mesh'], (False, 0.0, True)),
                       (['analyze', 'x.mp4', '--out', 'o', '-q', '--no-timecal'], (True, 0.06, False))):
        with pytest.raises(Stop):
            cli.main(argv)
        assert (seen[-1].fill, seen[-1].fill_overscan, seen[-1].mesh_residual) == want
    assert seen[0].timecal is True and seen[-1].timecal is False and seen[0].horizon_lock == 0.0
    monkeypatch.setattr(ab, '_camera_for', lambda v: 'DJI O4 Pro')
    with pytest.raises(Stop):
        cli.main(['analyze', 'x.mp4', '--out', 'o', '-q'])
    assert (seen[-1].fill, seen[-1].mesh_residual) == (False, False)


def test_option_info_support_notes_and_exclusivity(monkeypatch):
    from stillpoint import pipeline
    o4 = ab.option_info('DJI O4 Pro', gravity=True)
    assert set(o4) >= {'defaults', 'supported', 'exclusive', 'notes', 'ranges', 'time_factor'}
    assert 'O4 Pro' in o4['notes']['horizon'] and o4['supported']['horizon'] is True
    o3 = ab.option_info('DJI O3 (FC8383)', gravity=True)
    assert o3['notes'] == {} and o3['ranges']['roll_limit_deg'] == [0.0, 45.0]
    assert ab.option_info('DJI O3 (FC8383)', gravity=False)['supported']['horizon'] is False
    assert ['fill', 'mesh'] in o3['exclusive']                    # pipeline.check_params refuses fill + mesh today
    assert o3['time_factor']['mesh'] > 1.0
    for k in list(o3['supported']) + list(o3['notes']) + list(o3['time_factor']):   # single-word keys (Swift decoder)
        assert '_' not in k
    monkeypatch.setattr(pipeline, 'check_params', lambda prm: None)      # the engine lifts the restriction
    assert ab.exclusive_pairs() == []


def test_resolve_options_tristate():
    d = ab.option_defaults(None)
    ns = types.SimpleNamespace
    r = ab.resolve_options(ns(), None)                               # nothing given: the defaults
    assert (r['fill'], r['mesh'], r['horizon_lock'], r['timecal']) == (d['fill'], d['mesh'], d['horizon_lock'], d['timecal'])
    assert r['fill_overscan'] == (d['fill_overscan'] if d['fill'] else 0.0)
    r = ab.resolve_options(ns(fill=True, mesh=False, horizon_lock=0.6, roll_limit=15.0, timecal=False), None)
    assert (r['fill'], r['fill_overscan'], r['mesh'], r['horizon_lock'], r['roll_limit_deg'], r['timecal']) == \
        (True, ab.FILL_OVERSCAN_ON, False, 0.6, 15.0, False)
    assert ab.resolve_options(ns(horizon_lock=False), None)['horizon_lock'] == 0.0     # old callers: bool
    assert ab.resolve_options(ns(no_timecal=True), None)['timecal'] is False           # old namespace field
    assert ab.resolve_options(ns(fill=True, fill_overscan=0.03), None)['fill_overscan'] == 0.03


def test_analyze_parser_flags():
    ap = ab.build_parser()
    a = ap.parse_args(['analyze', 'c.mp4', '--out', 'o', '--fill', '--no-mesh', '--horizon-lock', '--roll-limit', '10',
                       '--no-timecal', '--dry-run'])
    assert (a.fill, a.mesh, a.horizon_lock, a.roll_limit, a.timecal, a.dry_run) == (True, False, 1.0, 10.0, False, True)
    a = ap.parse_args(['analyze', 'c.mp4', '--out', 'o'])
    assert (a.fill, a.mesh, a.horizon_lock, a.roll_limit, a.timecal, a.synth_blur) == (None,) * 6
    assert ap.parse_args(['analyze', 'c.mp4', '--out', 'o', '--horizon-lock', '0.4']).horizon_lock == 0.4


def test_native_stage_mapping_mesh_and_fill(monkeypatch):
    """The mesh / fill stages of a v5 analysis stay on the Vision step (not back to Gyro) with their own message."""
    out = []
    monkeypatch.setattr(ab, 'emit', out.append)
    np_ = ab._NativeProgress(ab.Progress([('telemetry', 'T', 1.0)], min_interval=0.0), 3)
    np_('mesh', 0.8, 'mesh: tracking 120/268')
    assert out[-1]['stage'] == 'measure' and 'Max quality' in out[-1]['message']
    np_('fill', 0.9, 'fill: selecting')
    assert out[-1]['stage'] == 'measure' and 'Full-frame fill' in out[-1]['message']
    np_('measure1', 0.5, 'measuring pass 1: 812/2964 pairs')
    assert out[-1]['message'] == 'pass 2 of 3 · 812/2964 frames'


def _tc_report(**tc):
    return dict(params=dict(timecal=True), calibration=dict(readout_meta_ms=11.0, timecal=tc))


def test_timecal_summary_states():
    assert ab.timecal_summary(dict(params=dict(smoothness=1.0), calibration={})) is None        # pre-v5 analysis
    assert ab.timecal_summary(dict(params=dict(timecal=False), calibration={}))['state'] == 'off'
    assert ab.timecal_summary(dict(params=dict(timecal=True), calibration={}))['state'] == 'skipped'  # no high-rate gyro
    none = dict(offset_ms=0.0, readout_s=None, focal_scale=1.0, exposure_slope=0.0, exposure_scale=1.0)
    # DJI_0034 (v5): -0.017 +- 0.041 ms, not applied -> the metadata timing is confirmed
    s = ab.timecal_summary(_tc_report(status='ok', applied=none, estimate=dict(offset_ms=-0.0173),
                                      sigma=dict(offset_ms=0.0406), decision=dict(reasons=['not worth it'])))
    assert s['state'] == 'confirmed' and s['estimate_ms'] == -0.0173 and s['sigma_ms'] == 0.0406 and s['detail'] == 'not worth it'
    # injected +0.92 ms error (merge verification): recovered and applied
    s = ab.timecal_summary(_tc_report(status='ok', applied=dict(none, offset_ms=0.928), estimate=dict(offset_ms=0.928),
                                      sigma=dict(offset_ms=0.005)))
    assert s['state'] == 'applied' and s['offset_ms'] == 0.928
    s = ab.timecal_summary(_tc_report(status='ok', applied=dict(none, readout_s=0.0111), estimate=dict(offset_ms=0.0),
                                      sigma=dict(offset_ms=0.01)))
    assert s['state'] == 'applied' and abs(s['readout_pct'] - 0.909) < 0.01 and s['offset_ms'] == 0.0
    # a fitted offset the engine did not trust: the metadata timing was KEPT, not confirmed
    s = ab.timecal_summary(_tc_report(status='ok', applied=none, estimate=dict(offset_ms=0.31), sigma=dict(offset_ms=0.02),
                                      decision=dict(reasons=['held-out folds do not confirm it'])))
    assert s['state'] == 'kept' and 'held-out' in s['detail']
    assert ab.timecal_summary(_tc_report(status='skipped: in-camera EIS baked into the picture'))['state'] == 'skipped'
    assert ab.timecal_summary(_tc_report(status="failed: ValueError('x')"))['state'] == 'failed'
    assert ab.timecal_summary(_tc_report(status='no window with enough rotation'))['state'] == 'kept'


def test_fill_mesh_horizon_summaries():
    rep = dict(fill=dict(frac_frames_filled=0.033, fill_frac_mean=0.0004, fill_frac_max=0.051, max_offset=10),
               mesh=dict(offset_rms_1080=0.21, offset_max_1080=2.97, verify=dict(accepted_frac=0.98)),
               smooth=dict(horizon=dict(strength=1.0, roll_limit_deg=0.0, frac_full_level=0.68, frac_off=0.0)))
    assert ab.fill_summary(rep)['frames_frac'] == 0.033 and ab.fill_summary(rep)['max_offset'] == 10
    assert ab.mesh_summary(rep) == dict(offset_rms_px=0.21, offset_max_px=2.97, accepted_frac=0.98)
    assert ab.horizon_summary(rep)['full_level_frac'] == 0.68
    off = dict(fill=None, mesh=None, smooth=dict(horizon=None))
    assert ab.fill_summary(off) is None and ab.mesh_summary(off) is None and ab.horizon_summary(off) is None
    assert ab.mesh_summary(dict(mesh=dict(error='boom')))['error'] == 'boom'


def test_summary_carries_v5_blocks(monkeypatch):
    tel = types.SimpleNamespace(fps=59.94, n_frames=10, width=3840, height=2160, camera='DJI O3', frame_t=np.zeros(10))
    monkeypatch.setattr('stillpoint.telemetry.load_telemetry', lambda *a, **k: tel)
    rep = dict(out=dict(w=3840, h=2160, min_out_fx=1611.0, hfov_deg=100.0), closed_loop={}, params=dict(timecal=True),
               calibration=dict(timecal=dict(status='ok', applied=dict(offset_ms=0.0), estimate=dict(offset_ms=0.01),
                                             sigma=dict(offset_ms=0.02))),
               fill=dict(frac_frames_filled=0.1))
    s = ab.summarize('/nonexistent.mp4', '/nonexistent', rep)
    assert s['timecal']['state'] == 'confirmed' and s['fill']['frames_frac'] == 0.1 and 'mesh' not in s


@needs_clip
def test_analyze_dry_run_roundtrip(tmp_path):
    """What the app sends -> the AnalyzeParams the engine would run and the manifest params the app reads back."""
    out = tmp_path / 'never'
    r = run_bridge(['analyze', CLIP, '--out', str(out), '--fov', '104', '--horizon-lock', '0.6', '--roll-limit', '15',
                    '--fill', '--no-mesh', '--dry-run'], env=dict(ENV, STILLPOINT_ANALYSIS_WORKERS='3'))
    assert r.returncode == 0, r.stderr[-2000:]
    res = events(r.stdout)[-1]
    assert res['type'] == 'result' and res['dry_run'] is True
    p, ap = res['params'], res['analyze_params']
    assert (p['horizon_lock'], p['horizon_strength'], p['roll_limit_deg'], p['fill'], p['fill_overscan'], p['mesh'],
            p['fov_deg']) == (True, 0.6, 15.0, True, ab.FILL_OVERSCAN_ON, False, 104.0)
    assert (ap['horizon_lock'], ap['roll_limit_deg'], ap['fill'], ap['fill_overscan'], ap['mesh_residual']) == \
        (0.6, 15.0, True, ab.FILL_OVERSCAN_ON, False)
    assert ap['processes'] == 3 and ap['max_workers_measure'] == 3
    assert not out.exists()                                        # a dry run writes nothing
    r = run_bridge(['analyze', CLIP, '--out', str(out), '--fill', '--mesh', '--dry-run'])
    ev = events(r.stdout)
    assert r.returncode == 1 and ev[-1]['type'] == 'error' and ev[-1]['code'] == 'bad_args'
    r = run_bridge(['analyze', CLIP, '--out', str(out), '--horizon-lock', '1.5', '--dry-run'])
    assert r.returncode == 1 and events(r.stdout)[-1]['code'] == 'bad_args'
    r = run_bridge(['analyze', CLIP, '--out', str(out), '--dry-run'])      # no flags: the camera defaults
    p = events(r.stdout)[-1]['params']
    d = ab.option_defaults('DJI O3')
    assert (p['fill'], p['mesh'], p['horizon_strength'], p['timecal']) == (d['fill'], d['mesh'], d['horizon_lock'], d['timecal'])


@needs_clip
def test_probe_reports_options():
    p = events(run_bridge(['probe', CLIP]).stdout)[-1]
    o = p['options']
    assert o['defaults'] == ab.option_defaults(p['camera'])
    assert o['supported']['horizon'] is p['horizon_lock_supported'] and ['fill', 'mesh'] in o['exclusive']
