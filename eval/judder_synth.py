"""
Synthetic validation of the blur-judder metric (eval/judder.py).

A textured plane at infinity (1/f noise + rectangles) is filmed by a rotating pinhole camera (pan + 5-23 Hz shake)
with a long exposure: every source frame is the AVERAGE of n_sub sub-exposure renders (true motion blur).  It is then
"stabilized" with a known virtual path (the pan only) -> an H.264 output clip, plus the same output rendered from
sharp (single-instant) source frames.

Checks:
  1. ground truth = the exact plan-based metric (blur_mismatch, known camera + virtual path);
     vision variant  = judder_from_eval on the eval's own KLT measurement of the output clip (what a Gyroflow render
     gets); per-frame correlation and the RMS ratio.
  2. the pixels: per output frame, directional gradient energy E(theta) of the blurred output relative to the sharp
     output; measured blur axis = argmin, strength = 1 - min ratio.  Against the metric's predicted baked streak at
     the centre (direction, length): median axis error on frames with streaks >= 3 px, Spearman(length, strength).
  3. the locked (tripod) output of the same clip: judder == baked blur (share 1).

    PYTHONPATH=engine:. .venv/bin/python -m eval.judder synthetic [--out DIR]
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from .judder import _engine, blur_mismatch, judder_from_eval, summarize

_engine()
from stillpoint.geom import Lens, qexp, qmul, quat_to_mat  # noqa: E402
from stillpoint.types import Telemetry  # noqa: E402

FPS = 60000 / 1001
W, H = 960, 540
F_SRC = 700.0
ZOOM = 1.25


def _texture(seed=0, w=3000, h=2200):
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    amp = 1.0 / np.maximum(np.hypot(fx, fy), 1.0 / 400)
    ph = rng.uniform(0, 2 * np.pi, amp.shape)
    img = np.fft.irfft2(amp * np.exp(1j * ph), s=(h, w))
    img = (img - img.mean()) / img.std()
    img = 128 + 28 * img
    for _ in range(900):
        x0, y0 = rng.integers(0, w - 40), rng.integers(0, h - 40)
        dx, dy = rng.integers(6, 40), rng.integers(6, 40)
        img[y0:y0 + dy, x0:x0 + dx] = rng.uniform(30, 225)
    import cv2
    return cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.float32), (0, 0), 0.7)


def _camera(n, e, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(-0.5, n / FPS + 0.5, 1 / 4000.0)
    rv = np.zeros((len(t), 3))
    tc = 0.5 * n / FPS
    rv[:, 1] = 0.12 * (t - tc)                                       # pan (yaw) 0.12 rad/s, centred
    for f, a in ((5.0, 0.004), (7.0, 0.005), (13.0, 0.004), (23.0, 0.0025)):
        rv += np.sin(2 * np.pi * f * t[:, None] + rng.uniform(0, 6.3, 3)) * a * np.array([1.0, 1.0, 0.5])
    q = qexp(rv)
    lens = Lens('pinhole', F_SRC, F_SRC, (W - 1) / 2, (H - 1) / 2, np.zeros(4), W, H)
    pts = np.arange(n) / FPS
    tel = Telemetry(source='synthetic', camera='synthetic', width=W, height=H, fps=FPS, frame_pts=pts,
                    frame_t=pts.copy(), exposure_s=np.full(n, e), readout_s=0.0, lens=lens, imu_t=t, imu_q=q,
                    imu_rate=4000.0, has_highrate=True, eis_baked=False, segments=[(0, n - 1)])
    virt = qexp(np.stack([np.zeros(n), 0.12 * (pts - tc), np.zeros(n)], -1))   # the pan only
    return tel, virt


def _rays(fx, w, h):
    X, Y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return np.stack([(X - (w - 1) / 2) / fx, (Y - (h - 1) / 2) / fx, np.ones_like(X)], -1)


def _remap(img, pix):
    import cv2
    return cv2.remap(img, pix[..., 0].astype(np.float32), pix[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _render_source(tex, R):
    """source frame for camera orientation R (3x3, camera->world): world ray -> texture pinhole (f_T = F_SRC)."""
    th, tw = tex.shape
    d = _SRC_RAYS @ R.T
    z = np.maximum(d[..., 2], 1e-6)
    pix = np.stack([F_SRC * d[..., 0] / z + (tw - 1) / 2, F_SRC * d[..., 1] / z + (th - 1) / 2], -1)
    return _remap(tex, pix)


_SRC_RAYS = _rays(F_SRC, W, H)


def _stabilize(src, Rk, Vk):
    """output pixel -> r_v -> source ray R_k^T V_k r_v -> source px (pinhole F_SRC)."""
    d = _OUT_RAYS @ (Rk.T @ Vk).T
    z = np.maximum(d[..., 2], 1e-6)
    pix = np.stack([F_SRC * d[..., 0] / z + (W - 1) / 2, F_SRC * d[..., 1] / z + (H - 1) / 2], -1)
    return _remap(src, pix)


_OUT_RAYS = _rays(F_SRC * ZOOM, W, H)


def _grad_energy(img, thetas):
    import cv2
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    c = (slice(H // 6, H - H // 6), slice(W // 6, W - W // 6))
    gx, gy = gx[c].astype(np.float64), gy[c].astype(np.float64)
    Ixx, Iyy, Ixy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    return np.array([np.cos(a) ** 2 * Ixx + np.sin(a) ** 2 * Iyy + 2 * np.cos(a) * np.sin(a) * Ixy for a in thetas])


def _write_video(path, frames):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}',
                          '-r', '60000/1001', '-i', '-', '-c:v', 'libx264', '-crf', '10', '-preset', 'veryfast',
                          '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.clip(np.round(f), 0, 255).astype(np.uint8).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError('ffmpeg failed')


def run_synthetic(out_dir=None, n=300, e=0.012, n_sub=24, seed=0) -> dict:
    from scipy.stats import spearmanr
    out_dir = out_dir or tempfile.mkdtemp(prefix='judder_synth_')
    os.makedirs(out_dir, exist_ok=True)
    tex = _texture(seed)
    tel, virt = _camera(n, e, seed + 1)
    Rv = quat_to_mat(virt)
    blurred, sharp = [], []
    taus = ((np.arange(n_sub) + 0.5) / n_sub - 0.5) * e
    for k in range(n):
        tk = tel.frame_t[k]
        Rk = quat_to_mat(tel.orientation_at(np.array([tk])))[0]
        acc = np.zeros((H, W), np.float64)
        for tau in taus:
            acc += _render_source(tex, quat_to_mat(tel.orientation_at(np.array([tk + tau])))[0])
        src_b = (acc / n_sub).astype(np.float32)
        src_s = _render_source(tex, Rk)
        blurred.append(_stabilize(src_b, Rk, Rv[k]))
        sharp.append(_stabilize(src_s, Rk, Rv[k]))
    vid = os.path.join(out_dir, 'synthetic_stabilized.mp4')
    _write_video(vid, blurred)
    # 1. ground truth vs the vision variant
    q = tel.orientation_at
    truth = blur_mismatch(q, tel.frame_t, tel.exposure_s, virt, np.full(n, F_SRC * ZOOM), W, H, tel.lens, 0.0, H)
    s_truth = summarize(truth, FPS)
    from . import jitter_metrics as jm
    npz = os.path.join(out_dir, 'synthetic_stabilized.npz')
    jm.run(vid, 0.0, None, signals=npz, plot=None, verbose=False)
    s_vis = judder_from_eval(tel, npz, 0.0, F_SRC * ZOOM * 1920.0 / W, H / W)
    # per-frame vision series for the correlation (recompute with the same helper, keep the per-frame lengths)
    vis_len = _vision_series(tel, npz, F_SRC * ZOOM * 1920.0 / W, H / W)
    m = min(len(vis_len), n)
    tr = int(0.5 * FPS)
    corr = float(np.corrcoef(truth['judder_len'][tr:m - tr], vis_len[tr:m - tr])[0, 1])
    # 2. pixels: measured directional blur vs the predicted baked streak at the centre
    thetas = np.deg2rad(np.arange(0, 180, 3.0))
    pred_dir, pred_len = _pred_centre_streak(tel, virt, e)
    meas_dir, meas_str = [], []
    for k in range(n):
        r = _grad_energy(blurred[k], thetas) / np.maximum(_grad_energy(sharp[k], thetas), 1e-9)
        i = int(np.argmin(r))
        meas_dir.append(thetas[i])
        meas_str.append(1.0 - r[i])
    meas_dir, meas_str = np.array(meas_dir), np.array(meas_str)
    big = pred_len >= 3.0 * W / 1920.0 * 2          # >= 3 px at 1080p-eq (x2 for 960 px)
    ang = np.abs(((meas_dir - pred_dir) + np.pi / 2) % np.pi - np.pi / 2)
    rho = float(spearmanr(pred_len, meas_str).correlation)
    # 3. locked output: judder == blur
    lock = blur_mismatch(q, tel.frame_t, tel.exposure_s, np.tile(virt[:1], (n, 1)), np.full(n, F_SRC * ZOOM), W, H,
                         tel.lens, 0.0, H)
    return dict(out_dir=out_dir, n_frames=n, exposure_ms=1e3 * e,
                truth=s_truth, vision=s_vis, vision_truth_corr=corr,
                vision_truth_rms_ratio=float(s_vis['judder_px'] / max(s_truth['judder_px'], 1e-9)),
                pixel_axis_err_deg_median=float(np.rad2deg(np.median(ang[big]))) if big.any() else None,
                pixel_axis_err_deg_p90=float(np.rad2deg(np.percentile(ang[big], 90))) if big.any() else None,
                pixel_frames_used=int(big.sum()), pixel_spearman_len_vs_strength=rho,
                locked_judder_over_blur=float(np.mean(lock['judder_len'] / np.maximum(lock['blur_len'], 1e-9))),
                locked_share=float(lock['share'].mean()))


def _vision_series(tel, npz, f1920, aspect):
    """per-frame judder_len of the vision variant (same math as judder_from_eval)."""
    from .judder import SQRT12, S1920, _proj, _stats, _taps, output_points
    from stillpoint.geom import qconj, qrotate
    z = np.load(npz)
    N = min(len(z['path_tx']), tel.n_frames)
    W1, H1 = S1920, S1920 * aspect
    pts = output_points(int(round(W1)), int(round(H1)))
    d = pts - np.array([(W1 - 1) / 2.0, (H1 - 1) / 2.0])
    P = np.stack([np.asarray(z[k], float)[:N] for k in ('path_tx', 'path_ty', 'path_rot', 'path_logs')], 1)
    vel = np.gradient(P, axis=0)
    v = np.stack([vel[:, None, 0] - vel[:, None, 2] * d[None, :, 1] + vel[:, None, 3] * d[None, :, 0],
                  vel[:, None, 1] + vel[:, None, 2] * d[None, :, 0] + vel[:, None, 3] * d[None, :, 1]], -1)
    e = np.asarray(tel.exposure_s, float)[:N]
    tau = _taps(9)
    ft = tel.frame_t[:N]
    rv = np.concatenate([d / f1920, np.ones((len(d), 1))], 1)
    rel = qmul(qconj(tel.orientation_at(ft[:, None] + e[:, None] * tau[None, :])), tel.orientation_at(ft)[:, None, :])
    O = _proj(qrotate(rel[:, None], np.broadcast_to(rv[None, :, None, :], (N, len(d), 9, 3))), f1920)
    C = (e * FPS)[:, None, None, None] * tau[None, None, :, None] * v[:, :, None, :]
    vm, _ = _stats(O - C, tau)
    return SQRT12 * np.sqrt(vm.mean(1))


def _pred_centre_streak(tel, virt, e):
    """principal axis (rad, image angle) and length (px, output res) of the baked streak at the output centre."""
    from .judder import _proj
    from stillpoint.geom import qconj, qrotate
    tau = ((np.arange(33) + 0.5) / 33 - 0.5) * e
    ft = tel.frame_t
    rv = np.array([0.0, 0.0, 1.0])
    qp = tel.orientation_at(ft)
    qi = tel.orientation_at(ft[:, None] + tau[None, :])
    V = virt[:, None, :]
    wq = qmul(qmul(qmul(qconj(V), qp[:, None, :]), qconj(qi)), V)
    O = _proj(qrotate(wq, np.broadcast_to(rv, wq.shape[:-1] + (3,))), F_SRC * ZOOM)          # (n,33,2)
    c = O - O.mean(1, keepdims=True)
    C = np.einsum('nti,ntj->nij', c, c) / c.shape[1]
    ev, evec = np.linalg.eigh(C)
    main = evec[:, :, -1]
    ang = np.arctan2(main[:, 1], main[:, 0]) % np.pi
    return ang, np.sqrt(12 * ev[:, -1])
