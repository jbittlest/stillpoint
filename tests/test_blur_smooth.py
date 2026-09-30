"""Blur-aware smoothing (SmoothParams.w_blur, smooth._blur_weights + the QP term in _solve_window).

  * short exposures (streak below blur_b0_px everywhere): the term is inactive -> the same path as w_blur = 0
  * long exposures + shake: the path follows the camera's motion along the streak more -> the plan-based blur
    judder (eval.judder) drops, and the path's own HF motion rises (the trade-off it makes)
"""
import numpy as np

from eval.judder import blur_mismatch
from stillpoint.geom import Lens, qexp, qlog, qmul, qconj
from stillpoint.smooth import SmoothParams, optimize_path
from stillpoint.types import Telemetry

FPS = 60000 / 1001
W, H = 1920, 1080


def _tel(exposure, n=360, seed=0):
    rng = np.random.default_rng(seed)
    imu_t = np.arange(-0.3, n / FPS + 0.3, 1 / 2000.0)
    yaw = 0.25 * np.sin(2 * np.pi * 0.2 * imu_t) + 0.02 * np.sin(2 * np.pi * 6.0 * imu_t + 0.3)
    pitch = 0.1 * np.sin(2 * np.pi * 0.15 * imu_t + 1.0) + 0.015 * np.sin(2 * np.pi * 9.0 * imu_t + 1.1)
    roll = 0.004 * np.sin(2 * np.pi * 4.0 * imu_t + rng.uniform(0, 6))
    q = qexp(np.stack([pitch, yaw, roll], -1))
    lens = Lens('pinhole', 1300.0, 1300.0, (W - 1) / 2, (H - 1) / 2, np.zeros(4), W, H)
    pts = np.arange(n) / FPS
    return Telemetry(source='synthetic', camera='synthetic', width=W, height=H, fps=FPS, frame_pts=pts,
                     frame_t=pts.copy(), exposure_s=np.full(n, exposure), readout_s=0.0, lens=lens,
                     imu_t=imu_t, imu_q=q, imu_rate=2000.0, has_highrate=True, eis_baked=False,
                     segments=[(0, n - 1)])


def _solve(tel, w_blur):
    prm = SmoothParams(min_out_fx=1700.0, allow_zoom=False, w_blur=w_blur, blur_b0_px=1.0, blur_b1_px=4.0, window=0)
    v, fx, info = optimize_path(tel, tel.orientation_at, np.arange(tel.n_frames), W, H, prm, return_info=True)
    return v, fx, info


def _judder(tel, v, fx):
    r = blur_mismatch(tel.orientation_at, tel.frame_t, tel.exposure_s, v, fx, W, H, tel.lens, 0.0, H)
    return float(np.sqrt(np.mean(r['judder_len'][30:-30] ** 2)))


def _hf(v):
    from scipy.signal import butter, sosfiltfilt
    w = qlog(qmul(qconj(v[:-1]), v[1:])) * 1700.0
    return float(np.sqrt(np.mean(sosfiltfilt(butter(4, 2.0, 'highpass', fs=FPS, output='sos'), w, axis=0)[30:-30] ** 2)))


def test_inactive_on_short_exposure():
    tel = _tel(0.0002)
    v0, f0, _ = _solve(tel, 0.0)
    v1, f1, info = _solve(tel, 50.0)
    assert info['blur']['frac_active'] == 0.0
    assert np.allclose(np.abs(np.einsum('ij,ij->i', v0, v1)), 1.0, atol=1e-12)


def test_long_exposure_trades_jitter_for_judder():
    tel = _tel(0.014)
    v0, f0, _ = _solve(tel, 0.0)
    v1, f1, info = _solve(tel, 500.0)
    j0, j1 = _judder(tel, v0, f0), _judder(tel, v1, f1)
    h0, h1 = _hf(v0), _hf(v1)
    print(f'\njudder {j0:.3f} -> {j1:.3f} px, path HF {h0:.4f} -> {h1:.4f} px/frame, active {info["blur"]["frac_active"]:.2f}')
    assert info['blur']['frac_active'] > 0.3
    assert j1 < 0.85 * j0                    # measured: 20.5 -> ~14 px (w 500); 10.8 at w 2000
    assert h1 > h0
