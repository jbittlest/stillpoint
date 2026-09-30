"""Synthetic shutter (engine/stillpoint/synth_blur.py, shaders/warp.metal *_blur kernels, sprender --blur).

  * .spblur round trip
  * 'angle' mode completes the exposure to the target shutter angle; taps are symmetric about the frame's own
    orientation (a constant-rate path gives D_i D_(n-1-i) = I) and equally weighted
  * the shutter is shortened where a tap would pull the output border outside the source
  * 'auto' on a static output adds no taps (nothing to mask with), and scales with the judder / output speed
  * golden (O3 DJI_0026, skipped without footage): sprender --blur pre-encode planes vs the float64 reference
"""
import os

import numpy as np
import pytest

from stillpoint.geom import Lens, qexp, qmul
from stillpoint.synth_blur import SynthBlurParams, read_blur, synth_blur_schedule, write_blur
from stillpoint.types import Telemetry

FPS = 60000 / 1001
W, H = 1920, 1080


def _tel(q_of_t, n=120, exposure=0.002, readout=0.008):
    imu_t = np.arange(-0.3, n / FPS + 0.3, 1 / 2000.0)
    lens = Lens('pinhole', 1000.0, 1000.0, (W - 1) / 2, (H - 1) / 2, np.zeros(4), W, H)
    pts = np.arange(n) / FPS
    return Telemetry(source='synthetic', camera='synthetic', width=W, height=H, fps=FPS, frame_pts=pts,
                     frame_t=pts.copy(), exposure_s=np.full(n, exposure), readout_s=readout, lens=lens,
                     imu_t=imu_t, imu_q=q_of_t(imu_t), imu_rate=2000.0, has_highrate=True, eis_baked=False,
                     segments=[(0, n - 1)])


def _yaw(w):
    return lambda t: qexp(np.stack([np.zeros_like(t), w * np.asarray(t), np.zeros_like(t)], -1))


def _sched(tel, virt, fx, prm, **kw):
    return synth_blur_schedule(tel.orientation_at, tel.frame_t, tel.exposure_s, virt, np.full(tel.n_frames, fx),
                               W, H, tel.lens, tel.readout_s, W, H, FPS, prm, **kw)


def test_roundtrip(tmp_path):
    tel = _tel(_yaw(0.5))
    s = _sched(tel, _yaw(0.5)(tel.frame_t), 1400.0, SynthBlurParams(mode='angle', max_taps=8))
    p = str(tmp_path / 'x.spblur')
    write_blur(p, tel.frame_pts, s, 1.0, 1.0)
    r = read_blur(p)
    assert np.allclose(r['frame_pts'], tel.frame_pts) and np.array_equal(r['n_taps'], s['n_taps'])
    assert np.allclose(r['D'], s['D'], atol=1e-6) and np.allclose(r['w'], s['w'], atol=1e-7)
    assert np.allclose(r['shutter_s'], s['shutter_s'], atol=1e-7)


def test_angle_mode_symmetric_taps():
    tel = _tel(_yaw(0.5), exposure=0.002)
    virt = _yaw(0.5)(tel.frame_t)
    s = _sched(tel, virt, 1400.0, SynthBlurParams(mode='angle', angle_deg=180.0, smooth_s=0.0, px_step=0.5))
    k = 60
    assert abs(s['shutter_s'][k] - (0.5 / FPS - 0.002)) < 1e-9
    n = s['n_taps'][k]
    # 0.5 rad/s x 6.34 ms at f 1400: 4.4 px at the centre, 6.6 px at the corners -> 14 taps at 0.5 px
    assert 13 <= n <= 15
    D = s['D'][k, :n]
    for i in range(n):
        assert np.allclose(D[i] @ D[n - 1 - i], np.eye(3), atol=1e-9)
    assert np.allclose(s['w'][k, :n], 1.0 / n) and np.allclose(s['w'][k, n:], 0.0)


def test_border_shrinks_shutter():
    tel = _tel(_yaw(0.5))
    virt = _yaw(0.5)(tel.frame_t)
    # out_fx 1000 == the source focal (pinhole): the output covers the whole source -> no room to turn at all
    s = _sched(tel, virt, 1000.0, SynthBlurParams(mode='angle', angle_deg=360.0, smooth_s=0.0))
    assert s['clamped'].mean() > 0.9 and np.all(s['shutter_s'] < 0.0005)
    s2 = _sched(tel, virt, 1600.0, SynthBlurParams(mode='angle', angle_deg=360.0, smooth_s=0.0))
    assert not s2['clamped'].any() and np.allclose(s2['shutter_s'], 1 / FPS - 0.002)


def test_auto_static_output_adds_nothing_and_scales():
    def shaky(t):
        return qmul(_yaw(0.3)(t), qexp(np.stack([0.003 * np.sin(2 * np.pi * 17 * np.asarray(t)),
                                                 np.zeros_like(t), np.zeros_like(t)], -1)))
    tel = _tel(shaky, exposure=0.006)
    still = np.tile(_yaw(0.3)(np.array([1.0])), (tel.n_frames, 1))
    s0 = _sched(tel, still, 1400.0, SynthBlurParams(mode='auto'))
    assert np.all(s0['n_taps'] == 1)                           # no output motion -> no synthetic streak
    pan = _yaw(0.3)(tel.frame_t)
    J = np.full(tel.n_frames, 2.0)
    s1 = _sched(tel, pan, 1400.0, SynthBlurParams(mode='auto', kappa=1.0, smooth_s=0.0), judder_len=J)
    s2 = _sched(tel, pan, 1400.0, SynthBlurParams(mode='auto', kappa=2.0, smooth_s=0.0), judder_len=J)
    k = 60
    assert s1['shutter_s'][k] > 0 and abs(s2['shutter_s'][k] / s1['shutter_s'][k] - 2.0) < 1e-6
    # synthetic streak ~ kappa x judder (output px = 1080p-eq px here, W = 1920)
    streak_rms = s1['speed_px_s'][k] * s1['shutter_s'][k]
    assert abs(streak_rms - 2.0) < 1e-6
    s3 = _sched(tel, pan, 1400.0, SynthBlurParams(mode='auto', judder_min_px=3.0), judder_len=J)
    assert np.all(s3['shutter_s'] == 0)


# ------------------------------------------------------------------------------------------ golden (sprender)
def test_golden_blur_sprender_vs_reference(tmp_path):
    from test_render_golden import V26, demux_pts, load_dump, make_plan, psnr10, run_sprender
    if not os.path.exists(V26):
        pytest.skip('O3 footage not available')
    from stillpoint.geom import qconj, quat_to_mat
    from stillpoint.plan_io import read_plan, write_plan
    from stillpoint.render_ref import render_planes_ref_blur
    pts = demux_pts(V26)
    plan = make_plan(pts, source=V26)
    pp = str(tmp_path / 'p.spplan')
    write_plan(pp, plan)
    frames = (100, 150)
    # a hand-made sidecar: 7 taps of +-1.5 deg yaw / 0.5 deg roll around each frame's own orientation
    K = 8
    D = np.tile(np.eye(3), (len(pts), K, 1, 1))
    w = np.zeros((len(pts), K))
    n = np.ones(len(pts), int)
    s = np.linspace(-1, 1, 7)
    for k in frames:
        rv = np.stack([np.zeros(7), np.deg2rad(1.5) * s, np.deg2rad(0.5) * s], -1)
        D[k, :7] = quat_to_mat(qexp(rv))
        w[k, :7] = 1 / 7
        n[k] = 7
    bp = str(tmp_path / 'p.spblur')
    write_blur(bp, pts, dict(n_taps=n, shutter_s=np.zeros(len(pts)), D=D, w=w))
    d = tmp_path / 'dump'
    res = run_sprender(V26, pp, tmp_path / 'o.mov', '--start-frame', 99, '--frames', 53, '--blur', bp,
                       '--dump-frames', ','.join(map(str, frames)), '--dump-dir', d)
    assert int(res['blurred']) == 2
    plan32 = read_plan(pp)
    B = read_blur(bp)
    for k in frames:
        Dm = load_dump(d, k)
        ry = np.arange(0, plan32.out_h, 9)                   # every 9th row (the float64 reference is slow at 4K)
        rc = np.arange(0, plan32.out_h // 2, 9)
        RY, RUV = render_planes_ref_blur(plan32, k, Dm['sy'], Dm['suv'], B['D'][k, :7], B['w'][k, :7],
                                         rows_y=ry, rows_c=rc)
        py, puv = psnr10(Dm['oy'][ry], RY), psnr10(Dm['ouv'][rc], RUV)
        print(f'\n[blur] frame {k}: PSNR Y {py:.1f} dB, CbCr {puv:.1f} dB')
        assert py > 45 and puv > 45


def test_pipeline_writes_sidecar(tmp_path):
    from stillpoint.pipeline import AnalyzeParams, _write_synth_blur
    from stillpoint.plan_build import build_plan
    tel = _tel(_yaw(0.5), n=90)
    q = tel.orientation_at
    frames = np.arange(tel.n_frames)
    plan = build_plan(tel, None, q, _yaw(0.5)(tel.frame_t), np.full(tel.n_frames, 1500.0), W, H, n_rows=8)
    assert _write_synth_blur(AnalyzeParams(), tel, q, plan, frames, str(tmp_path)) is None     # default: off
    rep = _write_synth_blur(AnalyzeParams(synth_blur='angle', verbose=False), tel, q, plan, frames, str(tmp_path))
    assert 'error' not in rep and rep['frac_frames'] > 0.9
    r = read_blur(rep['path'])
    assert np.allclose(r['frame_pts'], plan.frame_pts) and len(r['n_taps']) == tel.n_frames
