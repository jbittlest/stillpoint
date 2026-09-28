"""WP-D tests for stillpoint.residual: synthetic rectilinear previews made from REAL O3 frames.

Scene model: a real DJI O3 frame (DJI_0026 / DJI_0034, KB4 fisheye) is undistorted ONCE with geom.Lens to a
high-res rectilinear 'world' image (2x the preview focal). Preview k = that world seen by a camera with
effective orientation W_k (pure rotation => homography), rendered at 2x then INTER_AREA-downsampled to
960x540 (motion happens before sampling, like a sensor), + Gaussian noise, uint8. Optional nuisances:
an independently moving object (a rigid patch moving 3-6 px/frame), parallax-like local motion (steady radial
expansion of the lower 'ground' band from a FOE, up to ~1.5 px/frame), black borders.
Truth for pair k: R_k = W_k^T W_{k+1}.
"""
from __future__ import annotations

import os
import subprocess

import cv2
import numpy as np
import pytest

from stillpoint.geom import Lens, pinhole_K, qexp, qlog, qmul, qconj, quat_to_mat, mat_to_quat
from stillpoint.residual import ResidualParams, measure_residuals, px_equiv
from eval.footage import O3_DIR  # noqa: E402  (env-configurable, see eval/footage.py)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, 'work', 'wpd', 'src')
FOOTAGE = O3_DIR
O3_LENS_4K = Lens('kb4', 1405.129, 1405.129, 1919.5, 1079.5,
                  np.array([0.249918, 0.013606, -0.062084, 0.012193]), 3840, 2160)
OUT_W, OUT_H, OUT_F = 960, 540, 470.0
K = pinhole_K(OUT_F, OUT_W, OUT_H)
SS = 2
FPS = 60000 / 1001
SOURCES = {'0026': ('0026', 1.0), '0034': ('0034', 20.0), '0034b': ('0034', 32.0)}


def source_frame(name: str) -> np.ndarray:
    """1920x1080 gray fisheye frame (read-only decode of Jimmy's clip, cached under work/wpd/src)."""
    clip, t = SOURCES[name]
    path = os.path.join(SRC_DIR, f'o3_{clip}_t{t:g}.png')
    if not os.path.exists(path):
        os.makedirs(SRC_DIR, exist_ok=True)
        src = os.path.join(FOOTAGE, f'DJI_{clip}.MP4')
        if not os.path.exists(src):
            pytest.skip(f'footage not available: {src}')
        subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-ss', str(t), '-i', src, '-frames:v', '1',
                        '-vf', 'scale=1920:1080:flags=area,format=gray', path], check=True)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


_WORLD_CACHE: dict = {}
MARGIN = 200  # world-image margin (px at 2x) around the preview field of view


def world_image(name: str):
    """Undistort the fisheye frame to a rectilinear image with focal SS*OUT_F (+ margin). Returns (img f32, Kw)."""
    if name in _WORLD_CACHE:
        return _WORLD_CACHE[name]
    fish = source_frame(name).astype(np.float32)
    lens = O3_LENS_4K.scaled(0.5)
    Ww, Hw = SS * OUT_W + 2 * MARGIN, SS * OUT_H + 2 * MARGIN
    Kw = pinhole_K(SS * OUT_F, Ww, Hw)
    ys, xs = np.mgrid[0:Hw, 0:Ww].astype(np.float64)
    rays = np.stack([(xs - Kw[0, 2]) / Kw[0, 0], (ys - Kw[1, 2]) / Kw[1, 1], np.ones_like(xs)], -1)
    pix = lens.project(rays).astype(np.float32)
    img = cv2.remap(fish, pix[..., 0], pix[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    _WORLD_CACHE[name] = (img, Kw)
    return img, Kw


def render_preview(name: str, Wk: np.ndarray, k: int = 0, rng=None, noise=1.0, obj=None, parallax=None,
                   black_border=False, gain=1.0) -> np.ndarray:
    """Preview (960x540 uint8) seen by a camera with effective orientation Wk (cam->world, world = the
    undistorted source camera). obj: dict(x0,y0,w,h,vx,vy) in 1x px (rigid patch moving v px/frame).
    parallax: dict(gamma, yh, fx) steady radial expansion of rows below yh (1x px) from FOE (fx, yh)."""
    world, Kw = world_image(name)
    K2 = pinhole_K(SS * OUT_F, SS * OUT_W, SS * OUT_H)
    Hm = Kw @ Wk @ np.linalg.inv(K2)          # output(2x) pixel -> world pixel
    if obj is None and parallax is None:
        img = cv2.warpPerspective(world, Hm, (SS * OUT_W, SS * OUT_H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REFLECT)
    else:
        ys2, xs2 = np.mgrid[0:SS * OUT_H, 0:SS * OUT_W].astype(np.float32)
        x1 = (xs2 + 0.5) / SS - 0.5
        y1 = (ys2 + 0.5) / SS - 0.5
        if parallax is not None:
            yh, fx0 = parallax['yh'], parallax['fx']
            g = parallax['gamma'] * np.clip((y1 - yh) / (OUT_H - 1 - yh), 0, None) ** 1.5
            fac = 1.0 + k * g
            x1 = fx0 + (x1 - fx0) / fac
            y1 = yh + (y1 - yh) / fac
        if obj is not None:
            dx, dy = obj['vx'] * k, obj['vy'] * k
            inside = ((x1 >= obj['x0'] + dx) & (x1 < obj['x0'] + dx + obj['w']) &
                      (y1 >= obj['y0'] + dy) & (y1 < obj['y0'] + dy + obj['h']))
            x1 = np.where(inside, x1 - dx, x1)
            y1 = np.where(inside, y1 - dy, y1)
        x2 = (x1 + 0.5) * SS - 0.5
        y2 = (y1 + 0.5) * SS - 0.5
        X = Hm[0, 0] * x2 + Hm[0, 1] * y2 + Hm[0, 2]
        Y = Hm[1, 0] * x2 + Hm[1, 1] * y2 + Hm[1, 2]
        Z = Hm[2, 0] * x2 + Hm[2, 1] * y2 + Hm[2, 2]
        img = cv2.remap(world, (X / Z).astype(np.float32), (Y / Z).astype(np.float32), cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT)
    img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA) * gain
    if rng is not None and noise > 0:
        img = img + rng.normal(0, noise, img.shape)
    img = np.clip(np.rint(img), 1, 255).astype(np.uint8)  # 0 is reserved for 'invalid border'
    if black_border:
        img[:, :36] = 0
        tri = np.fromfunction(lambda y, x: (x + y) < 140, img.shape)
        img[tri[::-1]] = 0
    return img


def jitter_signal(t: np.ndarray, rng, amp_deg=(0.01, 0.3), freqs=(2.0, 30.0), n=4) -> np.ndarray:
    """Sum of n random sinusoids per axis: amplitudes in amp_deg, frequencies in freqs. (F,3) rad."""
    out = np.zeros((len(t), 3))
    for ax in range(3):
        for _ in range(n):
            a = np.deg2rad(rng.uniform(*amp_deg))
            f = rng.uniform(*freqs)
            out[:, ax] += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return out


def pan_path(t: np.ndarray, rate_dps=(4.0, -2.0, 1.5)) -> np.ndarray:
    """Intended smooth virtual path V (F,4): constant-rate pan (pitch, yaw, roll deg/s)."""
    return qexp(np.deg2rad(np.asarray(rate_dps))[None] * (t - t.mean())[:, None])


def rel_quats(q: np.ndarray) -> np.ndarray:
    return qmul(qconj(q[:-1]), q[1:])


def rot_err_px(q_true_rel: np.ndarray, rotvec_meas: np.ndarray) -> np.ndarray:
    """(P,3) rotation-vector error log(R_true^T R_meas)."""
    return qlog(qmul(qconj(q_true_rel), qexp(rotvec_meas)))


def make_sequence(name, n, seed, obj=False, parallax=False, black_border=False, amp=(0.01, 0.3), freqs=(2, 30),
                  rate=(4.0, -2.0, 1.5), flicker=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    V = pan_path(t, rate)
    eps = jitter_signal(t, rng, amp, freqs)
    Wq = qmul(V, qexp(eps))
    ob = dict(x0=120.0, y0=260.0, w=180.0, h=110.0, vx=4.5, vy=-1.2) if obj else None
    par = dict(gamma=0.0075, yh=330.0, fx=470.0) if parallax else None
    frames = []
    for k in range(n):
        g = 1.0 + flicker * np.sin(2 * np.pi * 1.3 * t[k])
        frames.append((k, render_preview(name, quat_to_mat(Wq[k]), k, rng, obj=ob, parallax=par,
                                         black_border=black_border, gain=g)))
    return dict(t=t, V=V, eps=eps, W=Wq, frames=frames)


def evaluate(seq, res):
    q_true = rel_quats(seq['W'])
    e = rot_err_px(q_true, res['rotvec'])
    epx = px_equiv(e, K, OUT_W, OUT_H)
    # error after removing the constant bias (steady parallax / object pull is low-frequency -> high-passed later)
    e_d = e - e.mean(0)
    epx_d = px_equiv(e_d, K, OUT_W, OUT_H)
    # expected-vs-measured error channel (must equal eps_{k+1}-eps_k)
    return dict(rms_px=float(np.sqrt(np.mean(epx ** 2))), rms_px_debiased=float(np.sqrt(np.mean(epx_d ** 2))),
                max_px=float(epx.max()), bias_px=float(px_equiv(e.mean(0), K, OUT_W, OUT_H)),
                truth_rms_px=float(np.sqrt(np.mean(px_equiv(qlog(q_true), K, OUT_W, OUT_H) ** 2))),
                conf_med=float(np.median(res['conf'])), pps=res['timing']['pairs_per_s'])


# ----------------------------------------------------------------------------- tests

N_FR = 72


@pytest.mark.parametrize('name', ['0026', '0034'])
def test_precision_clean(name):
    seq = make_sequence(name, N_FR, seed=1)
    V = seq['V']
    res = measure_residuals(seq['frames'], K, rel_quats(V))
    ev = evaluate(seq, res)
    print(name, 'clean', ev)
    assert ev['rms_px'] < 0.03, ev
    # the error channel is log(R_exp^T R_meas) ~ eps_{k+1} - eps_k
    d_eps = seq['eps'][1:] - seq['eps'][:-1]
    de = px_equiv(res['err_rotvec'] - d_eps, K, OUT_W, OUT_H)
    assert np.sqrt(np.mean(de ** 2)) < 0.05
    assert np.median(res['conf']) > 0.5


@pytest.mark.parametrize('name', ['0026', '0034'])
def test_precision_moving_object_and_parallax(name):
    seq = make_sequence(name, N_FR, seed=2, obj=True, parallax=True, black_border=True, flicker=0.03)
    res = measure_residuals(seq['frames'], K, rel_quats(seq['V']))
    ev = evaluate(seq, res)
    print(name, 'moving+parallax+border', ev, 'parallax_px', np.median(res['parallax_px']))
    assert ev['rms_px_debiased'] < 0.03, ev
    assert ev['rms_px'] < 0.06, ev
    assert np.median(res['conf']) > 0.3


def test_sign_convention_pure_yaw():
    """Scene content moving LEFT between frames = camera yawed RIGHT (+y rotation, y down => right-handed
    about y turns +z toward +x). The measured rotvec must be +yaw and the error vs identity equal to it."""
    a = np.deg2rad(0.2)
    W0 = np.eye(3)
    W1 = quat_to_mat(qexp(np.array([0.0, a, 0.0])))
    rng = np.random.default_rng(5)
    f0 = render_preview('0034', W0, 0, rng)
    f1 = render_preview('0034', W1, 1, rng)
    # sanity on the renderer: content shifts left by ~ f*a px near the centre
    flow = cv2.DISOpticalFlow_create(2).calc(f0, f1, None)
    assert np.median(flow[200:340, 380:580, 0]) == pytest.approx(-OUT_F * a, abs=0.2)
    res = measure_residuals([(0, f0), (1, f1)], K, None, workers=1)
    truth = np.array([0.0, a, 0.0])
    assert res['rotvec'][0][1] == pytest.approx(a, rel=0.01)
    assert px_equiv(res['rotvec'][0] - truth, K, OUT_W, OUT_H) < 0.02          # px
    assert px_equiv(res['err_rotvec'][0] - truth, K, OUT_W, OUT_H) < 0.02      # expected = identity
    # and relative to an expected rotation equal to the truth, the error vanishes
    res2 = measure_residuals([(0, f0), (1, f1)], K, qexp(truth)[None], workers=1)
    assert px_equiv(res2['err_rotvec'][0], K, OUT_W, OUT_H) < 0.02


def test_speed_960x540():
    """Production path: process pool (the thread pool is GIL-bound). Same results as the thread path."""
    from stillpoint.residual import make_pool
    seq = make_sequence('0034', 50, seed=3, obj=True)
    ex = make_pool()
    try:
        measure_residuals(seq['frames'][:12], K, rel_quats(seq['V'])[:11], executor=ex)       # warm-up
        res = measure_residuals(seq['frames'], K, rel_quats(seq['V']), executor=ex, chunk=8)
    finally:
        ex.shutdown()
    print('pairs/s', res['timing'])
    ref = measure_residuals(seq['frames'], K, rel_quats(seq['V']))
    assert np.array_equal(res['k0'], ref['k0']) and np.allclose(res['err_rotvec'], ref['err_rotvec'], atol=1e-12)
    if os.getloadavg()[0] > 0.5 * (os.cpu_count() or 8):
        pytest.skip(f'machine busy (load {os.getloadavg()[0]:.0f}): throughput not meaningful')
    assert res['timing']['pairs_per_s'] >= 25.0


def test_consecutive_only_skips_pairs_across_gaps():
    seq = make_sequence('0034', 12, seed=4)
    frames = [f for f in seq['frames'] if f[0] not in (5, 6)]
    res = measure_residuals(frames, K, None, consecutive_only=True, workers=2)
    assert np.all(res['k1'] - res['k0'] == 1) and len(res['k0']) == 11 - 3
