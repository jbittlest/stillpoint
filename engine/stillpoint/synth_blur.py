"""Synthetic motion blur along the VIRTUAL path (the "synthetic shutter") -> .spblur sidecar for sprender --blur.

Why: a stabilized frame keeps the blur of the real, shaky exposure; once the picture is steady those streaks point
along the original shake ("blur judder", eval/judder.py measures it).  The blur cannot be removed geometrically, but
it can be MASKED: warping the same frame at virtual orientations V(t_k + s), s in [-T_s/2, T_s/2], and averaging adds
a streak that exactly follows the output's own motion (rotation-only blur is geometrically exact, no depth needed).

Strength (mode 'auto'): per frame, enough virtual shutter that the synthetic streak is kappa x the mismatched baked
streak (eval.judder judder_len), T_s = kappa * judder / speed_out, where speed_out is the output's own image speed
(RMS over a 3x3 grid, px/s).  Where the output barely moves the synthetic streak is ~0 whatever T_s is: synthetic blur
cannot mask judder in a static shot (reported, not hidden).  Frames with judder below judder_min_px get none.
T_s <= max_shutter_frac / fps; smoothed in time (max-filter + box over +-smooth_s) so the blur does not pump; each
frame's shutter is then shortened (bisection) until every output-border sample of every tap maps inside the source
(no black edges; the kernel additionally renormalises per pixel over the taps that land inside).
Mode 'angle': complete the real exposure to a total shutter angle (the ND-filter look): T_s = angle/360/fps - e_k.

Taps: n = clamp(ceil(max border streak px / px_step), 2, max_taps) mid-point samples of a box shutter (equal
weights); frames whose synthetic streak is < min_px everywhere get 1 tap (rendered by the plain kernels).
Tap i: D_i = R(V_k)^T R(V(t_k + s_i)), V(t) the slerp of the plan's virtual path (extrapolated at the ends).

.spblur v1 (little-endian): 64-byte header "SPBLUR01", u32 version=1, u32 header_bytes=64, u32 n_frames, u32 max_taps,
u32 record_bytes (= 16 + 40*max_taps), f32 kappa, f32 px_step, zero padding; records: f64 pts, u32 n_taps,
f32 shutter_s, max_taps x (f32 D[9] row-major, f32 weight).
"""
from __future__ import annotations

import os
import struct
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geom import qconj, qmul, quat_to_mat

MAGIC = b'SPBLUR01'
HEADER_BYTES = 64
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

__all__ = ['SynthBlurParams', 'synth_blur_schedule', 'write_blur', 'read_blur', 'blur_for_plan']


@dataclass
class SynthBlurParams:
    mode: str = 'auto'               # 'auto' (mask the baked-blur mismatch) | 'angle' (complete a shutter angle) | 'off'
    kappa: float = 1.0               # auto: synthetic streak >= kappa x the mismatched (judder) streak
    judder_min_px: float = 0.5       # auto: frames with less judder (1080p-eq px) get no synthetic blur
    angle_deg: float = 180.0         # 'angle': target total shutter angle
    max_shutter_frac: float = 1.0    # T_s <= this x the frame interval (1.0 = 360 deg)
    smooth_s: float = 0.1            # temporal smoothing of T_s (anti-pumping): max-filter then box, +-smooth_s
    px_step: float = 1.0             # output px between taps (border samples)
    max_taps: int = 16
    min_px: float = 0.35             # synthetic streak below this (output px, border max) -> 1 tap (off)
    margin_px: float = 1.0           # source-pixel margin for the border check
    judder_taps: int = 9


def _virt_at(vt, V, t):
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from eval.judder import virt_at
    return virt_at(vt, V, t)


def _border(out_w, out_h, n=5):
    xs = np.linspace(0, out_w - 1, n)
    ys = np.linspace(0, out_h - 1, n)
    return np.concatenate([np.stack([xs, np.zeros(n)], 1), np.stack([xs, np.full(n, out_h - 1.0)], 1),
                           np.stack([np.zeros(n - 2), ys[1:-1]], 1), np.stack([np.full(n - 2, out_w - 1.0), ys[1:-1]], 1)])


def _proj_rays(Vq_rel, rv, f):
    """R(Vq_rel) rv projected (rectilinear, f): Vq_rel (...,4) broadcast with rv (...,3)."""
    from .geom import qrotate
    r = qrotate(Vq_rel, rv)
    z = np.where(np.abs(r[..., 2]) < 1e-9, 1e-9, r[..., 2])
    return np.stack([f * r[..., 0] / z, f * r[..., 1] / z], -1)


def _streak_px(vt, V, ft, f, T, pts, out_w, out_h):
    """Synthetic streak length (px, max over pts) of each frame for shutters T (F,): |pi(D(+T/2) r) - pi(D(-T/2) r)|."""
    cx, cy = (out_w - 1) / 2.0, (out_h - 1) / 2.0
    rv = np.stack([(pts[:, 0] - cx), (pts[:, 1] - cy), np.zeros(len(pts))], 1)
    rv = rv[None] / f[:, None, None]
    rv[..., 2] = 1.0
    qa = qmul(qconj(_virt_at(vt, V, ft + 0.5 * T)), V)[:, None]
    qb = qmul(qconj(_virt_at(vt, V, ft - 0.5 * T)), V)[:, None]
    d = _proj_rays(qa, rv, f[:, None]) - _proj_rays(qb, rv, f[:, None])
    return np.linalg.norm(d, axis=-1)


def synth_blur_schedule(q_cam_fn, frame_t, exposure, virt_q, out_fx, out_w: int, out_h: int, lens, readout: float,
                        src_w: int, src_h: int, fps: float, params: Optional[SynthBlurParams] = None, virt_t=None,
                        judder_len: Optional[np.ndarray] = None) -> dict:
    """Per-frame synthetic shutter + taps.  Inputs per OUTPUT frame (frame_t = source centre-row mid-exposure time).
    judder_len (F,) 1080p-eq px: precomputed eval.judder mismatch (computed here when None and mode == 'auto').
    Returns dict(shutter_s (F,), n_taps (F,), D (F,K,3,3), w (F,K), speed_px_s (F,), judder_len (F,), clamped (F,))."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    from .smooth import map_output_points
    prm = params or SynthBlurParams()
    ft = np.asarray(frame_t, np.float64).reshape(-1)
    F = len(ft)
    V = np.asarray(virt_q, np.float64).reshape(F, 4)
    vt = ft if virt_t is None else np.asarray(virt_t, np.float64)
    f = np.broadcast_to(np.asarray(out_fx, np.float64), (F,)).astype(np.float64)
    e = np.clip(np.nan_to_num(np.broadcast_to(np.asarray(exposure, np.float64), (F,)), nan=0.0), 0.0, 0.1)
    Tf = 1.0 / float(fps)
    Tmax = float(prm.max_shutter_frac) * Tf
    K = int(max(2, prm.max_taps))
    grid = np.stack(np.meshgrid(np.linspace(0.1, 0.9, 3) * (out_w - 1), np.linspace(0.1, 0.9, 3) * (out_h - 1)), -1).reshape(-1, 2)
    # output image speed (px/s, RMS over the grid) from the virtual path over one frame interval
    cx, cy = (out_w - 1) / 2.0, (out_h - 1) / 2.0
    rvg = np.stack([(grid[:, 0] - cx), (grid[:, 1] - cy), np.zeros(len(grid))], 1)[None] / f[:, None, None]
    rvg[..., 2] = 1.0
    da = _proj_rays(qmul(qconj(_virt_at(vt, V, ft + 0.5 * Tf)), V)[:, None], rvg, f[:, None])
    db = _proj_rays(qmul(qconj(_virt_at(vt, V, ft - 0.5 * Tf)), V)[:, None], rvg, f[:, None])
    speed = np.sqrt(np.mean(np.sum((da - db) ** 2, -1), 1)) / Tf                   # px/s (output px)
    if prm.mode == 'off':
        T = np.zeros(F)
    elif prm.mode == 'angle':
        T = np.clip(prm.angle_deg / 360.0 * Tf - e, 0.0, Tmax)
    elif prm.mode == 'auto':
        if judder_len is None:
            if _ROOT not in sys.path:
                sys.path.insert(0, _ROOT)
            from eval.judder import blur_mismatch
            judder_len = blur_mismatch(q_cam_fn, ft, e, V, f, out_w, out_h, lens, readout, src_h, virt_t=vt,
                                       taps=prm.judder_taps)['judder_len']
        J_out = np.asarray(judder_len, np.float64) * (out_w / 1920.0)                  # output px
        T = prm.kappa * J_out / np.maximum(speed, 1e-6)
        T = np.where(np.asarray(judder_len) >= prm.judder_min_px, T, 0.0)
        T = np.clip(T, 0.0, Tmax)
    else:
        raise ValueError(f'unknown synthetic blur mode {prm.mode!r}')
    n = max(1, int(round(prm.smooth_s * fps)))
    if prm.mode != 'off' and F > 2 and n > 0:
        T = uniform_filter1d(maximum_filter1d(T, 2 * n + 1, mode='nearest'), 2 * n + 1, mode='nearest')
    T = np.clip(T, 0.0, Tmax)
    # keep every tap's border inside the source: shorten the shutter where needed (bisection, both ends symmetric)
    bpts = _border(out_w, out_h, 5)
    m = float(prm.margin_px)
    clamped = np.zeros(F, bool)

    def ok_at(idx, s):
        Vs = _virt_at(vt, V, ft[idx] + s)
        p = map_output_points(q_cam_fn, ft[idx], Vs, f[idx], bpts, out_w, out_h, lens, readout, src_h, 3)
        return np.all((p[..., 0] >= m) & (p[..., 0] <= src_w - 1 - m) & (p[..., 1] >= m) & (p[..., 1] <= src_h - 1 - m),
                      axis=1)
    act = np.flatnonzero(T > 0)
    if len(act):
        good = ok_at(act, 0.5 * T[act]) & ok_at(act, -0.5 * T[act])
        bad = act[~good]
        if len(bad):
            clamped[bad] = True
            lo, hi = np.zeros(len(bad)), T[bad].copy()
            base_ok = ok_at(bad, np.zeros(len(bad)))
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                g = ok_at(bad, 0.5 * mid) & ok_at(bad, -0.5 * mid)
                lo = np.where(g, mid, lo)
                hi = np.where(g, hi, mid)
            T[bad] = np.where(base_ok, lo, 0.0)
    # taps
    L = _streak_px(vt, V, ft, f, T, np.concatenate([bpts, grid]), out_w, out_h).max(axis=1)
    nt = np.clip(np.ceil(L / max(prm.px_step, 1e-3)).astype(int), 2, K)
    nt = np.where(L < prm.min_px, 1, nt)
    D = np.zeros((F, K, 3, 3))
    w = np.zeros((F, K))
    D[:, :, 0, 0] = D[:, :, 1, 1] = D[:, :, 2, 2] = 1.0
    for cnt in np.unique(nt):
        idx = np.flatnonzero(nt == cnt)
        s = ((np.arange(cnt) + 0.5) / cnt - 0.5)[None, :] * T[idx, None]            # (m,cnt)
        Vs = _virt_at(vt, V, ft[idx, None] + s)                                         # (m,cnt,4)
        rel = qmul(qconj(V[idx])[:, None, :], Vs)                                       # R(V_k)^T R(V(t+s))
        D[idx, :cnt] = quat_to_mat(rel)
        w[idx, :cnt] = 1.0 / cnt
    return dict(shutter_s=T, n_taps=nt, D=D, w=w, speed_px_s=speed, streak_px=L, clamped=clamped,
                judder_len=np.asarray(judder_len) if judder_len is not None else np.zeros(F))


def write_blur(path: str, frame_pts: np.ndarray, sched: dict, kappa: float = 0.0, px_step: float = 0.0) -> None:
    F = len(frame_pts)
    K = sched['D'].shape[1]
    hdr = bytearray(HEADER_BYTES)
    struct.pack_into('<8sIIIII', hdr, 0, MAGIC, 1, HEADER_BYTES, F, K, 16 + 40 * K)
    struct.pack_into('<ff', hdr, 28, float(kappa), float(px_step))
    dt = np.dtype([('pts', '<f8'), ('n', '<u4'), ('shutter', '<f4'), ('taps', '<f4', (K, 10))])
    rec = np.zeros(F, dtype=dt)
    rec['pts'] = frame_pts
    rec['n'] = sched['n_taps']
    rec['shutter'] = sched['shutter_s']
    rec['taps'][..., :9] = sched['D'].reshape(F, K, 9)
    rec['taps'][..., 9] = sched['w']
    assert dt.itemsize == 16 + 40 * K
    tmp = path + '.part'
    with open(tmp, 'wb') as fh:
        fh.write(bytes(hdr))
        fh.write(rec.tobytes())
    os.replace(tmp, path)


def read_blur(path: str) -> dict:
    with open(path, 'rb') as fh:
        b = fh.read()
    magic, ver, hb, F, K, rb = struct.unpack_from('<8sIIIII', b, 0)
    if magic != MAGIC or ver != 1 or rb != 16 + 40 * K:
        raise ValueError(f'not a v1 .spblur file: {path}')
    kappa, px_step = struct.unpack_from('<ff', b, 28)
    dt = np.dtype([('pts', '<f8'), ('n', '<u4'), ('shutter', '<f4'), ('taps', '<f4', (K, 10))])
    rec = np.frombuffer(b, dtype=dt, count=F, offset=hb)
    return dict(frame_pts=rec['pts'].astype(np.float64), n_taps=rec['n'].astype(int),
                shutter_s=rec['shutter'].astype(np.float64), D=rec['taps'][..., :9].reshape(F, K, 3, 3).astype(np.float64),
                w=rec['taps'][..., 9].astype(np.float64), kappa=kappa, px_step=px_step)


def blur_for_plan(tel, q_cam_fn, plan, frames: np.ndarray, params: Optional[SynthBlurParams] = None,
                  judder_len=None) -> dict:
    """Schedule for a Plan built from `frames` of `tel` (plan.virt_q / out_fx / geometry)."""
    frames = np.asarray(frames, np.int64)
    return synth_blur_schedule(q_cam_fn, tel.frame_t[frames], np.asarray(tel.exposure_s, np.float64)[frames],
                               plan.virt_q, plan.out_fx, plan.out_w, plan.out_h, plan.lens,
                               float(plan.meta.get('readout_s', tel.readout_s)), plan.src_w, plan.src_h,
                               float(tel.fps), params, judder_len=judder_len)
