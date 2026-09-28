"""WP-F pipeline tests.  PYTHONPATH=engine .venv/bin/python -m pytest tests/test_pipeline.py -q"""
import json
import math
import os

import numpy as np
import pytest

from stillpoint.pipeline import (AnalyzeParams, _score, _window_mask, analyze, first_frame_at, render,
                                 residual_stats, window_hf)
from stillpoint.plan_io import read_plan
from eval.footage import o3  # noqa: E402  (env-configurable, see eval/footage.py)

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
CLIP = o3('DJI_0026.MP4')


def test_first_frame_at_matches_ffmpeg_seek():
    pts = np.arange(3000) * 1001 / 60000
    assert first_frame_at(pts, 15.0) == 900        # 899 -> 14.998 s, 900 -> 15.015 s
    assert first_frame_at(pts, 8.0) == 480
    assert first_frame_at(pts, 0.0) == 0
    assert first_frame_at(pts, pts[37]) == 37


def test_residual_stats_units_and_gate():
    fs, F = 59.94, 1500
    t = np.arange(F) / fs
    f1920 = 700.0
    A_px = 0.5                                        # yaw sinusoid, 0.5 px peak at 5 Hz
    eps = np.zeros((F, 3))
    eps[:, 1] = A_px / f1920 * np.sin(2 * np.pi * 5 * t)
    eps[:, 2] = 0.2 / math.sqrt((1920 ** 2 + 1080 ** 2) / 12) * np.sin(2 * np.pi * 12 * t)   # 0.2 px roll @12 Hz
    res = dict(k0=np.arange(F - 1), k1=np.arange(1, F), err_rotvec=np.diff(eps, axis=0), conf=np.ones(F - 1))
    speed = np.where(t < 12.5, 50.0, 500.0)
    st = residual_stats(res, F, fs, f1920, speed)
    want = math.sqrt(A_px ** 2 / 2 + 0.2 ** 2 / 2)
    assert abs(st['hf_px'] - want) / want < 0.03
    assert abs(st['b2_8_px'] - A_px / math.sqrt(2)) < 0.02
    assert abs(st['b8_30_px'] - 0.2 / math.sqrt(2)) < 0.01
    assert abs(st['hf_axis_px']['yaw'] - A_px / math.sqrt(2)) < 0.01
    assert st['hf_axis_px']['pitch'] < 1e-6
    assert 0.45 < st['calm_frac'] < 0.55 and abs(st['calm_hf_px'] - want) / want < 0.05
    # untrusted pairs contribute nothing
    res['conf'][:] = 0.0
    st0 = residual_stats(res, F, fs, f1920, speed)
    assert st0['hf_px'] < 1e-9 and st0['trusted_frac'] == 0.0
    assert _score(st) > _score(st0)


def test_window_hf_and_mask():
    fs, F = 59.94, 600
    t = np.arange(F) / fs
    eps = np.zeros((F, 3))
    eps[240:360, 0] = 1e-3 * np.sin(2 * np.pi * 6 * t[240:360])          # jitter only in window 4 (1-s windows)
    res = dict(k0=np.arange(F - 1), k1=np.arange(1, F), err_rotvec=np.diff(eps, axis=0), conf=np.ones(F - 1))
    h = window_hf(res, F, fs, 700.0, 60)
    assert h.shape == (10,) and h.argmax() == 4 and h[4] > 10 * np.median(h)
    acc = np.ones(10, bool)
    acc[4] = False
    m = _window_mask(acc, F, 60, 15)
    assert m[270] < 0.01 and m[100] > 0.999 and m[500] > 0.999 and np.all(np.abs(np.diff(m)) < 0.2)


@pytest.mark.skipif(not os.path.exists(CLIP), reason='footage not available')
def test_analyze_and_render_short_clip(tmp_path):
    """End-to-end on the 268-frame DJI_0026: whole-clip analysis with one closed-loop fold, then a 4K render."""
    out = tmp_path / 'a'
    prm = AnalyzeParams(calibrate=False, closed_loop_iters=1, min_improve=-1.0, crop_area=0.7, verbose=False)
    rep = analyze(CLIP, str(out), prm)
    assert os.path.exists(out / 'plan.spplan') and os.path.exists(out / 'report.json')
    r = json.load(open(out / 'report.json'))
    it = r['closed_loop']['iterations']
    assert len(it) == 2 and 'fold' in it[0]
    # local acceptance: the composite (accepted windows) is never worse than open loop, window by window
    a = np.load(out / 'analysis.npz')
    assert np.all(a['window_hf_best'] <= a['window_hf_open'] + 1e-12)
    assert r['closed_loop']['composite_window_hf_px'] <= r['closed_loop']['open_loop_window_hf_px'] + 1e-12
    s0 = it[0]['stats']
    assert s0['trusted_frac'] > 0.8 and s0['hf_px'] < 2.0
    plan = read_plan(str(out / 'plan.spplan'))
    assert plan.n_frames == rep['n_frames'] and plan.out_w == 3840 and plan.out_h == 2160
    assert rep['smooth']['max_violation_px'] < 1.0
    mov = str(tmp_path / 'r.mov')
    rr = render(CLIP, str(out / 'plan.spplan'), mov, start_frame=100, n_frames=20, codec='hevc10-speed')
    from stillpoint.video import probe
    pr = probe(mov)
    assert pr['n_frames'] == 20 and pr['width'] == 3840
    assert abs(pr['frame_pts'][0]) < 1e-6                         # zero-based


def test_plan_footprint_matches_analytic_area():
    """Exact source footprint (verifier's fov2 method) of a camera-locked plan == fx_for_crop_area's area."""
    from stillpoint.geom import Lens
    from stillpoint.pipeline import plan_footprint
    from stillpoint.smooth import fx_for_crop_area
    from stillpoint.types import Plan
    lens = Lens('kb4', 1405.13, 1405.13, 1919.5, 1079.5, np.array([0.2499, 0.0136, -0.0621, 0.0122]), 3840, 2160)
    for area in (0.55, 0.65):
        fx = fx_for_crop_area(lens, 3840, 2160, 3840, 2160, area)
        plan = Plan(src_w=3840, src_h=2160, out_w=3840, out_h=2160, lens=lens, frame_pts=np.zeros(2),
                    out_fx=np.full(2, fx), row_mats=np.tile(np.eye(3), (2, 8, 1, 1)), virt_q=np.tile([1.0, 0, 0, 0], (2, 1)))
        fp = plan_footprint(plan, np.arange(2))
        assert abs(fp.mean() - area) < 0.004, (fp, area)


@pytest.mark.skipif(not os.path.exists(CLIP), reason='footage not available')
def test_progress_and_cancel(tmp_path):
    from stillpoint.pipeline import AnalysisCancelled
    calls = []
    prm = AnalyzeParams(closed_loop_iters=1, min_improve=-1.0, crop_area=0.7, verbose=False, processes=2)
    analyze(CLIP, str(tmp_path / 'a'), prm, progress=lambda s, f, m: calls.append((s, f, m)))
    fr = [c[1] for c in calls]
    assert len(calls) >= 8 and fr[-1] == 1.0 and all(b >= a for a, b in zip(fr, fr[1:]))
    assert {'telemetry', 'measure0', 'measure1', 'final', 'done'} <= {c[0] for c in calls}
    assert not os.path.exists(tmp_path / 'a' / '.luma_cache.npy')           # temporary cache removed
    # cancel as soon as the first measurement pass starts
    seen = []

    def cancel():
        return any(c[0] == 'measure0' for c in seen)
    with pytest.raises(AnalysisCancelled):
        analyze(CLIP, str(tmp_path / 'b'), prm, progress=lambda s, f, m: seen.append((s, f, m)), cancel=cancel)
    assert not os.path.exists(tmp_path / 'b' / '.luma_cache.npy')
    assert not os.path.exists(tmp_path / 'b' / 'plan.spplan')


def test_progress_heartbeat_and_callback_cancel():
    """analyze()'s progress keeps ticking (>= 1/s) through silent steps; a *Cancel* exception raised by the
    callback on the heartbeat thread cancels the analysis at the next check."""
    import threading
    import time
    from stillpoint.pipeline import AnalysisCancelled, _Progress
    calls = []
    pg = _Progress(lambda s, f, m: calls.append(time.perf_counter()), None)
    stop = threading.Event()
    th = threading.Thread(target=pg.heartbeat, args=(stop,), daemon=True)
    th.start()
    pg('crop', 0.5, 'solving')
    time.sleep(2.6)
    stop.set()
    th.join()
    assert len(calls) >= 3 and max(np.diff(calls)) < 1.5

    class Cancelled(Exception):
        pass
    n = []

    def cb(s, f, m):
        n.append(1)
        if len(n) > 1:
            raise Cancelled()
    pg = _Progress(cb, None)
    stop = threading.Event()
    th = threading.Thread(target=pg.heartbeat, args=(stop,), daemon=True)
    th.start()
    pg('crop', 0.5, 'solving')
    time.sleep(1.6)
    stop.set()
    th.join()
    with pytest.raises(AnalysisCancelled):
        pg.check()


def test_gyroflow_footprint_auto_lookup_is_length_checked():
    from stillpoint.pipeline import _gf_footprint_cached
    p = os.path.join(os.path.dirname(__file__), '..', 'work', 'baseline', 'DJI_0034_wholeclip_gf_footprint.json')
    if not os.path.exists(p):
        pytest.skip('no cached Gyroflow footprint')
    got = _gf_footprint_cached('/somewhere/DJI_0034.MP4', n_frames=3029)
    assert got is not None and 0.5 < got[0] < 0.75
    assert _gf_footprint_cached('/somewhere/DJI_0034.MP4', n_frames=9000) is None     # another clip, same name
