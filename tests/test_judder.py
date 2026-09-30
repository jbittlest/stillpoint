"""Blur-judder metric (eval/judder.py): analytic checks on synthetic telemetry.

  * unstabilized output (virtual path = camera path) -> 0 judder, with rolling shutter and any motion
  * locked output under a constant pan w -> judder = f*w*e at the centre (and lin = that vector)
  * a vibration completing whole cycles inside the exposure -> blur but ~no straight (lin) mismatch
  * synthetic shutter along a moving virtual path lowers the inconsistent share, never the mismatch in px
"""
import numpy as np

from eval.judder import blur_mismatch, output_points, summarize
from stillpoint.geom import Lens, qexp, qmul
from stillpoint.types import Telemetry

FPS = 60000 / 1001
W, H = 1920, 1080


def _tel(q_of_t, n=240, exposure=0.008, readout=0.008, rate=2000.0):
    imu_t = np.arange(-0.3, n / FPS + 0.3, 1.0 / rate)
    lens = Lens('pinhole', 1000.0, 1000.0, (W - 1) / 2, (H - 1) / 2, np.zeros(4), W, H)
    pts = np.arange(n) / FPS
    return Telemetry(source='synthetic', camera='synthetic', width=W, height=H, fps=FPS, frame_pts=pts,
                     frame_t=pts.copy(), exposure_s=np.full(n, exposure), readout_s=readout, lens=lens,
                     imu_t=imu_t, imu_q=q_of_t(imu_t), imu_rate=rate, has_highrate=True, eis_baked=False,
                     segments=[(0, n - 1)])


def _yaw(w):
    return lambda t: qexp(np.stack([np.zeros_like(t), w * t, np.zeros_like(t)], -1))


def _run(tel, virt, **kw):
    q = tel.orientation_at
    return blur_mismatch(q, tel.frame_t, tel.exposure_s, virt, np.full(tel.n_frames, 1000.0), W, H, tel.lens,
                         tel.readout_s, H, **kw)


def test_unstabilized_is_zero():
    # constant-rate rotation about a fixed axis: zero with rolling shutter (every row sees the same streak)
    def spin(t):
        return qexp(np.stack([0.3 * t, 0.8 * t, -0.2 * t], -1))
    tel = _tel(spin)
    r = _run(tel, tel.orientation_at(tel.frame_t))
    assert r['judder_len'].max() < 1e-3                     # px
    assert r['blur_len'].mean() > 5.0                       # but there IS blur
    # slow sway: the frame-rate path follows it -> the mismatch is a small fraction of the blur
    rng = np.random.default_rng(0)
    comps = [(f, rng.normal(size=3) * 0.02) for f in (0.7, 1.5)]

    def sway(t):
        v = np.stack([0.3 * t, 0.8 * t, -0.2 * t], -1)
        for f, a in comps:
            v = v + np.sin(2 * np.pi * f * t)[..., None] * a
        return qexp(v)
    tel = _tel(sway)
    r = _run(tel, tel.orientation_at(tel.frame_t))
    assert r['judder_len'].max() < 0.02 * r['blur_len'].mean()


def test_vibration_above_nyquist_counts():
    # prop vibration faster than the frame rate cannot be motion of ANY frame-rate path: it is counted (the ORIGINAL
    # has it too -- report the original's level next to the render's)
    def vib(t):
        return qexp(np.stack([0.002 * np.sin(2 * np.pi * 180.0 * t), np.zeros_like(t), np.zeros_like(t)], -1))
    tel = _tel(vib, exposure=0.004, rate=8000.0)
    r = _run(tel, tel.orientation_at(tel.frame_t))
    assert r['judder_len'].mean() > 1.0


def test_locked_pan_equals_f_w_e():
    w, e = 1.5, 0.008                                       # rad/s, s -> 12 px streak at f = 1000
    tel = _tel(_yaw(w), n=30, exposure=e, readout=0.0)
    pts = np.array([[(W - 1) / 2, (H - 1) / 2]])
    virt = tel.orientation_at(np.zeros(tel.n_frames))       # output locked to world yaw 0
    r = _run(tel, virt, pts=pts)
    # the world point at the output centre is smeared by the camera's world-frame yaw: x = f*tan(w*tau)
    L = 1000.0 * w * e
    assert np.all(np.abs(r['judder_len'] - L) / L < 0.01)
    assert np.all(np.abs(np.abs(r['lin'][:, 0, 0]) - L) < 0.01 * L)
    assert np.all(np.abs(r['lin'][:, 0, 1]) < 1e-6 * L)
    assert np.all(r['cons_len'] < 1e-9) and np.all(r['share'] > 0.99)   # nothing consistent about it
    assert abs(summarize(r, FPS)['share'] - 1.0) < 0.01


def test_vibration_whole_cycle_length():
    e = 0.004
    f_vib = 1.0 / e                                          # 250 Hz: one full cycle per exposure

    def vib(t):
        return qexp(np.stack([0.002 * np.sin(2 * np.pi * f_vib * t), np.zeros_like(t), np.zeros_like(t)], -1))
    tel = _tel(vib, exposure=e, readout=0.0, rate=16000.0)
    virt = np.tile(np.array([1.0, 0, 0, 0]), (tel.n_frames, 1))
    r = _run(tel, virt, taps=64, pts=np.array([[(W - 1) / 2, (H - 1) / 2]]))
    # centre: y = f*tan(A sin) ~ f*A*sin over a whole period -> RMS f*A/sqrt2 -> length sqrt(12)*f*A/sqrt(2)
    L = np.sqrt(6.0) * 1000.0 * 0.002
    assert np.all(np.abs(r['judder_len'] - L) < 0.02 * L)


def test_synthetic_shutter_masks_but_does_not_remove():
    # camera: pan 1 rad/s + 20 Hz shake; virtual path = the pan only
    def cam(t):
        return qmul(_yaw(1.0)(t), qexp(np.stack([0.004 * np.sin(2 * np.pi * 20 * t), np.zeros_like(t),
                                                 np.zeros_like(t)], -1)))
    tel = _tel(cam, exposure=0.006, readout=0.0)
    virt = _yaw(1.0)(tel.frame_t)
    r0 = _run(tel, virt)
    r1 = _run(tel, virt, synth_shutter=np.full(tel.n_frames, 0.008))
    assert np.allclose(r0['judder_len'], r1['judder_len'])
    assert r1['synth_len'].mean() > 6.0                      # 8 ms of a 1 rad/s pan at f 1000
    assert r1['share'].mean() < 0.6 * r0['share'].mean()
    s = summarize(r1, FPS)
    assert s['judder_hf_px'] > 0.5 and s['synth_px'] > 6.0
    s0 = summarize(r0, FPS)                                  # pooled (energy) share: same judder, lower share
    assert s['share'] < 0.6 * s0['share'] and abs(s['judder_px'] - s0['judder_px']) < 1e-9


def test_vision_variant_ignores_translation_scale(tmp_path):
    # judder_from_eval (renders without a plan): a camera that does not rotate, filmed in low forward flight -> the
    # render's measured motion is a pure similarity SCALE rate (radial parallax). The baked blur model is rotation-
    # only, so this is not judder (the original shows the same expansion blur); counting it (scale_term=True, the
    # old behaviour) reports a large spurious mismatch.
    from eval.judder import judder_from_eval
    tel = _tel(lambda t: qexp(np.zeros(np.shape(t) + (3,))), n=240, exposure=0.008, readout=0.0)
    N = tel.n_frames
    z = dict(path_tx=np.zeros(N), path_ty=np.zeros(N), path_rot=np.zeros(N),
             path_logs=np.cumsum(np.full(N, 0.01)), fps=np.float64(FPS))
    p = str(tmp_path / 'ev.npz')
    np.savez(p, **z)
    off = judder_from_eval(tel, p, 0.0, 1000.0, H / W)
    on = judder_from_eval(tel, p, 0.0, 1000.0, H / W, scale_term=True)
    assert off['judder_px'] < 1e-6 and off['blur_px'] < 1e-6
    assert on['judder_px'] > 1.0
    # and a real rotation still counts: a constant yaw with a LOCKED render (no measured motion) -> the plan-exact value
    tel = _tel(_yaw(1.5), n=240, exposure=0.008, readout=0.0)
    np.savez(p, **dict(z, path_logs=np.zeros(N)))
    r = judder_from_eval(tel, p, 0.0, 1000.0, H / W)
    exact = summarize(_run(tel, np.tile(np.array([1.0, 0, 0, 0]), (N, 1))), FPS)
    assert abs(r['judder_px'] / exact['judder_px'] - 1.0) < 0.01
    assert r['judder_px'] > 12.0                             # f*w*e = 12 px at the centre, more off-centre


def test_output_points_grid():
    p = output_points(100, 50, 3, 3, 0.1)
    assert p.shape == (9, 2) and np.isclose(p[4, 0], 49.5) and np.isclose(p[4, 1], 24.5)
