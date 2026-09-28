"""Vision residual ROTATION measurement on rectilinear stabilized previews (WP-D, ENGINE_SPEC.md §2/§4.5).

This is the measuring half of the micro-jitter killer: render 960-wide rectilinear previews from the current
plan, then measure, for every consecutive pair, the inter-frame rotation the IMAGE CONTENT actually shows and
compare it with the rotation the virtual path intended. `closedloop.fold_residuals` turns the per-pair errors
into an orientation correction.

Geometry / conventions (ENGINE_SPEC.md §1)
------------------------------------------
Preview frame k is a pinhole image (intrinsics K, pixel centres at integers) showing what a camera with
EFFECTIVE orientation W_k (camera->world) sees.  For far-field content, pixels of frames k and k+1 satisfy

    x_{k+1} ~ K R_k^T K^-1 x_k ,      R_k = W_k^T W_{k+1}

R_k is the relative rotation of frame k+1 w.r.t. frame k, expressed in the camera frame of frame k
(x right, y down, z forward).  `rotvec[i] = log(R_k)` [rad].  With the intended virtual path V
(`expected_rel_q[i] = conj(V_k) * V_{k+1}`), the per-pair error is

    err_rotvec[i] = log( R_exp^T R_meas )            (== log(conj(q_exp) * q_meas))

If the content appears rotated by eps_k w.r.t. the intent, W_k = V_k Exp(eps_k), then to first order
err_rotvec[i] = eps_{k+1} - eps_k  (this is what closedloop integrates).

Algorithm (research/sota_ai.md §2.3-2.4, §4(1); research/metrics.md §2 on parallax)
------------------------------------------------------------------------------------
Stage A (per pair, parallel): OpenCV DIS flow (frame k+1 pre-warped by the expected rotation when that is
    large) -> robust rotation fit on a 4-px flow grid (IRLS-Cauchy on the pixel reprojection error, weighted
    Kabsch on unit rays, initialised at the expected rotation so the far field / gyro-consistent layer wins)
    -> per-grid-point residual |flow - rotational flow|.
Stage B (sequential, cheap): temporally smoothed (+-4 pairs) squared residual map -> SOFT far-field weight
    1/(1+S/c^2) (persistent parallax is down-weighted the same way in consecutive pairs, so the rotation
    estimate does not flip between depth layers -- the failure mode that fooled RANSAC in metrics.md), times
    a HARD instantaneous mask |res| < 1 px (moving objects: trains, cars, turbine blades), eroded.
Stage C (per pair, parallel): masked direct photometric refinement of the 3-parameter rotation
    (+ gain/bias), H = K R^T K^-1: ESM Gauss-Newton with IRLS-Cauchy weights, half-res then full-res,
    cubic sampling, on the strongest-gradient valid pixels. Invalid pixels (black borders, value 0, or a
    caller-supplied mask) never contribute.
Confidence combines the inlier fraction, the formal precision, and flow-vs-direct agreement.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np

from .geom import mat_to_quat, qexp, qlog, quat_to_mat

__all__ = ['ResidualParams', 'measure_residuals', 'measure_pair', 'px_equiv', 'rel_rotation_homography', 'make_pool']


# ----------------------------------------------------------------------------- parameters


@dataclass
class ResidualParams:
    dis_preset: int = 2                 # cv2.DISOPTICAL_FLOW_PRESET_MEDIUM (sharp motion boundaries)
    dis_patch_stride: int = 4           # MEDIUM uses 3. (ENGINE v3 tried stride 6 + 2 variational iterations: +38 %
                                        # pairs/CPU-s and equal on synthetic truth, but on DJI_0034 the closed loop
                                        # converged to 0.097 instead of 0.079 px (self-estimate) and the independent
                                        # eval HF rose 0.285 -> 0.305 px: reverted to (4, preset's 5))
    dis_gd_iters: int = 12
    dis_finest: int | None = None       # DIS finest pyramid scale (None = preset's, 1 for MEDIUM; 2 = half-res flow)
    dis_var_iters: int | None = None    # DIS variational refinement iterations (None = preset's 5)
    flow_step: int = 4                  # grid step (px) of the flow rotation fit / residual maps
    prewarp_px: float = 0.5             # pre-warp frame k+1 by the expected rotation if it moves > this
    hard_tol_px: float = 1.0            # instantaneous flow residual above this = independent motion
    inlier_tol_px: float = 1.0          # 'inlier' definition for inlier_frac
    parallax_c_px: float = 0.15         # Cauchy scale on the temporally smoothed residual (far-field pref.)
    temporal_radius: int = 4            # pairs each side for the smoothed residual map
    erode_cells: int = 1                # hard-mask erosion radius in grid cells (x flow_step px)
    border_px: int = 6                  # ignore this many px at the image border
    invalid_dilate_px: int = 4          # grow invalid (black) regions by this much
    blur_sigma: float = 0.8             # pre-blur for the direct stage (anti-alias, wider basin)
    max_pixels: int = 60_000            # direct-stage pixel budget at full res (1/4 of it at half res)
    min_grad: float = 0.75              # DN/px; flatter pixels carry no information
    iters_half: int = 4
    iters_full: int = 8
    photometric: bool = True            # solve gain + bias too (auto-exposure changes)
    cauchy_k: float = 2.385             # IRLS Cauchy constant (x robust sigma of the intensity residual)
    workers: int | None = None          # threads; None -> min(8, cpu_count-2)
    epipolar: bool = False              # EXPERIMENTAL translation-aware (depth-free) rotation fit from flow for
                                        # parallax / moving scenes. Off: on DJI_0028 it disagrees with the direct
                                        # fit by 0.25 px (1080p, median) where both are confident - too noisy
    epi_min_inlier: float = 0.6         # ... run it when the rotation-only inlier fraction is below this
    epi_tol_px: float = 0.5             # ... inlier tolerance on the flow component across the epipolar line
    chunk: int = 32                     # pairs per parallel batch


# ----------------------------------------------------------------------------- small helpers


def _rotm(v: np.ndarray) -> np.ndarray:
    return quat_to_mat(qexp(np.asarray(v, dtype=np.float64)))


def _logm(R: np.ndarray) -> np.ndarray:
    return qlog(mat_to_quat(np.asarray(R, dtype=np.float64)))


def rel_rotation_homography(K: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Pixel map frame k -> frame k+1 for relative rotation R = W_k^T W_{k+1}: x' ~ K R^T K^-1 x."""
    return K @ np.asarray(R, dtype=np.float64).T @ np.linalg.inv(K)


def px_equiv(rotvec: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
    """RMS image displacement (px) over the frame produced by a small rotation (…,3):
    sqrt(f^2 (wx^2 + wy^2) + wz^2 (W^2+H^2)/12). Same convention as the eval harness (metrics.md §2.1)."""
    v = np.asarray(rotvec, dtype=np.float64)
    f = 0.5 * (K[0, 0] + K[1, 1])
    return np.sqrt(f * f * (v[..., 0] ** 2 + v[..., 1] ** 2) + v[..., 2] ** 2 * (width ** 2 + height ** 2) / 12.0)


def _scaled_K(K: np.ndarray, s: float) -> np.ndarray:
    Ks = K.astype(np.float64).copy()
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return Ks


def _valid_mask(img: np.ndarray, extra: np.ndarray | None, dil: int) -> np.ndarray:
    """uint8 1 = usable. Pure-black (0) pixels are treated as outside-the-source border."""
    v = (img > 0).astype(np.uint8)
    if extra is not None:
        v &= (np.asarray(extra) > 0).astype(np.uint8)
    if dil > 0 and not v.all():
        v = cv2.erode(v, np.ones((2 * dil + 1, 2 * dil + 1), np.uint8))
    return v


_tls = threading.local()


def _dis(p: 'ResidualParams'):
    key = (p.dis_preset, p.dis_patch_stride, p.dis_gd_iters, p.dis_finest, p.dis_var_iters)
    d = getattr(_tls, 'dis', None)
    if d is None or getattr(_tls, 'dis_key', None) != key:
        d = cv2.DISOpticalFlow_create(p.dis_preset)
        if p.dis_patch_stride:
            d.setPatchStride(p.dis_patch_stride)
        if p.dis_gd_iters:
            d.setGradientDescentIterations(p.dis_gd_iters)
        if p.dis_finest is not None:
            d.setFinestScale(int(p.dis_finest))
        if p.dis_var_iters is not None:
            d.setVariationalRefinementIterations(int(p.dis_var_iters))
        _tls.dis, _tls.dis_key = d, key
    return d


def _kabsch(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Rotation Q minimising sum w |b - Q a|^2 (a, b unit rays (N,3))."""
    M = (b * w[:, None]).T @ a
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    return U @ D @ Vt


def _proj(K: np.ndarray, r: np.ndarray) -> np.ndarray:
    z = r[:, 2]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return np.stack([K[0, 0] * r[:, 0] / z + K[0, 2], K[1, 1] * r[:, 1] / z + K[1, 2]], axis=1)


# ----------------------------------------------------------------------------- stage A: flow + robust fit


def _flow_stage(A, B, vA, vB, K, R_exp, p: ResidualParams) -> dict:
    Hh, Ww = A.shape
    Kinv = np.linalg.inv(K)
    prewarp = float(px_equiv(_logm(R_exp), K, Ww, Hh)) > p.prewarp_px
    Hexp = rel_rotation_homography(K, R_exp)
    if prewarp:
        Bp = cv2.warpPerspective(B, Hexp, (Ww, Hh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
    else:
        Bp = B
    flow = _dis(p).calc(A, Bp, None)
    s = p.flow_step
    ys, xs = np.mgrid[s // 2:Hh:s, s // 2:Ww:s]
    gh, gw = ys.shape
    fl = flow[ys, xs].astype(np.float64)
    qx = xs + fl[..., 0]
    qy = ys + fl[..., 1]
    if prewarp:
        X = Hexp[0, 0] * qx + Hexp[0, 1] * qy + Hexp[0, 2]
        Y = Hexp[1, 0] * qx + Hexp[1, 1] * qy + Hexp[1, 2]
        Z = Hexp[2, 0] * qx + Hexp[2, 1] * qy + Hexp[2, 2]
        qx, qy = X / Z, Y / Z
    lam = cv2.cornerMinEigenVal(A, 5, ksize=3)[ys, xs].astype(np.float64)
    b = p.border_px
    inside = (qx >= b) & (qx <= Ww - 1 - b) & (qy >= b) & (qy <= Hh - 1 - b)
    qxi = np.clip(np.rint(qx), 0, Ww - 1).astype(np.int32)
    qyi = np.clip(np.rint(qy), 0, Hh - 1).astype(np.int32)
    valid = (vA[ys, xs] > 0) & inside & (vB[qyi, qxi] > 0)
    valid &= (xs >= b) & (xs <= Ww - 1 - b) & (ys >= b) & (ys <= Hh - 1 - b)
    lam_ref = np.median(lam[valid]) if valid.any() else 1.0
    wtex = lam / (lam + 0.3 * lam_ref + 1e-9)
    P = np.stack([xs, ys], -1).reshape(-1, 2).astype(np.float64)
    Q = np.stack([qx, qy], -1).reshape(-1, 2)
    vflat = valid.reshape(-1)
    wt = wtex.reshape(-1) * vflat
    a = np.c_[P, np.ones(len(P))] @ Kinv.T
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    bb = np.c_[Q, np.ones(len(Q))] @ Kinv.T
    bb /= np.linalg.norm(bb, axis=1, keepdims=True)
    Rt = R_exp.T.copy()  # Q_rot = R^T maps rays of frame k to rays of frame k+1
    res = np.full(len(P), 99.0)
    ok = vflat.sum() >= 30
    if ok:
        sel = np.flatnonzero(wt > 0.05)
        if len(sel) > 6000:
            sel = sel[::2]
        a_s, b_s, w_s, Q_s = a[sel], bb[sel], wt[sel], Q[sel]
        for c in (4.0, 2.0, 1.2, 0.8, 0.6, 0.6):
            r = np.linalg.norm(_proj(K, a_s @ Rt.T) - Q_s, axis=1)
            w = w_s / (1.0 + (r / c) ** 2)
            if w.sum() < 1e-6:
                ok = False
                break
            Rt = _kabsch(a_s, b_s, w)
        res = np.linalg.norm(_proj(K, a @ Rt.T) - Q, axis=1)
        res[~vflat] = 99.0
    res = res.reshape(gh, gw)
    tex_ok = valid & (wtex > 0.5)
    n_tex = max(int(tex_ok.sum()), 1)
    inl = float((tex_ok & (res < p.inlier_tol_px)).sum()) / n_tex
    moderate = tex_ok & (res < 3.0)
    par = float(np.sqrt(np.mean(res[moderate] ** 2))) if moderate.any() else float('nan')
    out = dict(R_flow=Rt.T.copy(), res=res.astype(np.float32), valid=valid, ok=ok,
               inlier_frac=inl if ok else 0.0, parallax_px=par, tex_frac=float(tex_ok.mean()),
               prewarp=prewarp)
    if p.epipolar and ok and inl < p.epi_min_inlier:
        sel = np.flatnonzero((tex_ok.reshape(-1)) & (res.reshape(-1) < 12.0))
        if len(sel) > 8000:
            sel = sel[:: len(sel) // 8000 + 1]
        if len(sel) >= 200:
            out.update(_epipolar_fit(a[sel], Q[sel], wtex.reshape(-1)[sel], K, Rt, p))
    return out


def _epipolar_fit(a: np.ndarray, Q: np.ndarray, w0: np.ndarray, K: np.ndarray, Rt0: np.ndarray,
                  p: 'ResidualParams', iters: int = 4) -> dict:
    """Depth-free rotation from flow: for a static scene (or an object moving parallel to the camera's
    translation, e.g. a train beside a drone flying along it) the flow left after removing the rotation points
    along the epipolar line through the epipole e; its component ACROSS that line depends on rotation only.
    Alternates: epipole from the residual directions (homogeneous LSQ, IRLS) <-> rotation from the across-line
    components (Gauss-Newton, Cauchy). a: unit rays of frame k (N,3); Q: matched pixels in frame k+1 (N,2).
    Returns R_epi (relative rotation, like R_flow), epi_inlier_frac, epi_sigma_px, epi_rms_px, epi_ok."""
    Rt = Rt0.copy()
    w = w0.astype(np.float64).copy()
    out = dict(epi_ok=False, epi_inlier_frac=0.0, epi_sigma_px=float('inf'), epi_rms_px=float('nan'))
    f = 0.5 * (K[0, 0] + K[1, 1])
    for it in range(iters):
        pred = _proj(K, a @ Rt.T)
        r = Q - pred
        rn = np.linalg.norm(r, axis=1)
        if np.median(rn) < 0.25:                # no measurable translation: epipole undefined
            return out
        # epipole (homogeneous, frame k+1 pixels): r_x (y ez - ey) - r_y (x ez - ex) = 0
        x, y = pred[:, 0], pred[:, 1]
        A = np.stack([r[:, 1], -r[:, 0], r[:, 0] * y - r[:, 1] * x], 1)
        A /= np.maximum(rn, 1e-6)[:, None]
        we = w * rn ** 2 / (rn ** 2 + 0.3 ** 2)
        for _ in range(3):
            _, _, Vt = np.linalg.svd(A * np.sqrt(we)[:, None], full_matrices=False)
            e = Vt[-1]
            if abs(e[2]) > 1e-9 * np.linalg.norm(e[:2]) + 1e-12:
                d = np.stack([x - e[0] / e[2], y - e[1] / e[2]], 1) * np.sign(e[2])
            else:
                d = np.tile(e[:2], (len(x), 1))
            dn = np.linalg.norm(d, axis=1)
            u = d / np.maximum(dn, 1e-9)[:, None]
            perp = r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0]
            we = w * rn ** 2 / (rn ** 2 + 0.3 ** 2) / (1.0 + (perp / 0.5) ** 2)
        n = np.stack([-u[:, 1], u[:, 0]], 1)                  # unit normal to the epipolar line
        near = dn < 0.03 * f                                   # direction undefined at the epipole
        # rotation: minimise sum w (n . (Q - pi(Rt a)))^2 ; Rt <- Rt Exp(-delta) (R <- Exp(delta) R)
        c = 0.6 if it < 2 else p.epi_tol_px * 0.8
        for _ in range(2):
            pred = _proj(K, a @ Rt.T)
            e_perp = np.sum(n * (Q - pred), 1)
            J = np.empty((len(a), 3))
            h = 1e-6
            for j in range(3):
                dv = np.zeros(3)
                dv[j] = h
                pj = _proj(K, a @ (Rt @ _rotm(-dv)).T)
                J[:, j] = np.sum(n * (pj - pred), 1) / h
            ww = w / (1.0 + (e_perp / c) ** 2)
            ww[near] = 0.0
            N = (J * ww[:, None]).T @ J
            g = (J * ww[:, None]).T @ e_perp
            try:
                dl = np.linalg.solve(N + 1e-12 * np.trace(N) * np.eye(3), g)
            except np.linalg.LinAlgError:
                return out
            Rt = Rt @ _rotm(-dl)
    pred = _proj(K, a @ Rt.T)
    e_perp = np.sum(n * (Q - pred), 1)
    use = (~near) & (w > 0.05)
    inl = float((np.abs(e_perp[use]) < p.epi_tol_px).mean()) if use.any() else 0.0
    ww = w / (1.0 + (e_perp / (p.epi_tol_px * 0.8)) ** 2)
    ww[near] = 0.0
    s2 = float(np.sum(ww * e_perp ** 2) / max(np.sum(ww) - 3, 1.0))
    try:
        C = np.linalg.inv((J * ww[:, None]).T @ J) * s2
        sig = np.sqrt(np.maximum(np.diag(C), 0))
    except np.linalg.LinAlgError:
        return out
    Hh, Ww = int(round(2 * K[1, 2] + 1)), int(round(2 * K[0, 2] + 1))
    out.update(epi_ok=True, R_epi=Rt.T.copy(), epi_inlier_frac=inl, epi_sigma_px=float(px_equiv(sig, K, Ww, Hh)),
               epi_rms_px=float(np.sqrt(np.mean(e_perp[use & (np.abs(e_perp) < 2)] ** 2))) if use.any() else float('nan'),
               epi_e=e.tolist())
    return out


# ----------------------------------------------------------------------------- stage C: direct ESM refinement


class _Level:
    """Pre-computed image data of one pyramid level for a pair."""

    def __init__(self, A, B, vA, vB, K, wfull, p: ResidualParams, scale: float, budget: int):
        if scale != 1.0:
            sz = (int(round(A.shape[1] * scale)), int(round(A.shape[0] * scale)))
            A = cv2.resize(A, sz, interpolation=cv2.INTER_AREA)
            B = cv2.resize(B, sz, interpolation=cv2.INTER_AREA)
            vA = cv2.resize(vA, sz, interpolation=cv2.INTER_NEAREST)
            vB = cv2.resize(vB, sz, interpolation=cv2.INTER_NEAREST)
            wfull = cv2.resize(wfull, sz, interpolation=cv2.INTER_AREA)
            K = _scaled_K(K, scale)
        self.K = K
        self.h, self.w = A.shape
        sig = p.blur_sigma
        A = cv2.GaussianBlur(A.astype(np.float32), (0, 0), sig)
        B = cv2.GaussianBlur(B.astype(np.float32), (0, 0), sig)
        self.B = B
        gAx = cv2.Scharr(A, cv2.CV_32F, 1, 0, scale=1.0 / 32)
        gAy = cv2.Scharr(A, cv2.CV_32F, 0, 1, scale=1.0 / 32)
        self.gBx = cv2.Scharr(B, cv2.CV_32F, 1, 0, scale=1.0 / 32)
        self.gBy = cv2.Scharr(B, cv2.CV_32F, 0, 1, scale=1.0 / 32)
        self.vB = vB
        bd = max(2, int(round(p.border_px * scale)))
        gm = np.sqrt(gAx * gAx + gAy * gAy)
        cand = (vA > 0) & (wfull > 0.02) & (gm > p.min_grad)
        cand[:bd] = False
        cand[-bd:] = False
        cand[:, :bd] = False
        cand[:, -bd:] = False
        score = np.where(cand, gm * np.sqrt(wfull), 0.0).astype(np.float32).reshape(-1)
        idx = np.flatnonzero(score > 0)
        if len(idx) > budget:
            # threshold from a strided subsample (exact top-N is not needed, a full partition is slow)
            sub = score[idx[::8]]
            thr = np.partition(sub, len(sub) - budget // 8 - 1)[len(sub) - budget // 8 - 1]
            idx = idx[score[idx] >= thr]
        ys, xs = np.divmod(idx, self.w)
        self.n = len(idx)
        self.xs = xs.astype(np.float64)
        self.ys = ys.astype(np.float64)
        self.i0 = A.reshape(-1)[idx].astype(np.float64)
        self.g0x = gAx.reshape(-1)[idx].astype(np.float64)
        self.g0y = gAy.reshape(-1)[idx].astype(np.float64)
        self.wp = wfull.reshape(-1)[idx].astype(np.float64)
        f = K[0, 0]
        u = (self.xs - K[0, 2]) / K[0, 0]
        v = (self.ys - K[1, 2]) / K[1, 1]
        # d(x')/d(delta) of x' = K Exp(delta)^T K^-1 x at delta = 0 (update R <- Exp(delta) R)
        self.Jx = np.stack([f * u * v, -f * (1 + u * u), f * v], 1)
        self.Jy = np.stack([f * (1 + v * v), -f * u * v, -f * u], 1)


def _sample(img: np.ndarray, mx: np.ndarray, my: np.ndarray, interp: int) -> np.ndarray:
    """img sampled at float positions (N,) (cv2.remap needs < 32767 columns: fold into a 2-D block)."""
    n = len(mx)
    cols = 1024
    rows = (n + cols - 1) // cols
    fx = np.zeros(rows * cols, np.float32)
    fy = np.zeros(rows * cols, np.float32)
    fx[:n] = mx
    fy[:n] = my
    out = cv2.remap(img, fx.reshape(rows, cols), fy.reshape(rows, cols), interp,
                    borderMode=cv2.BORDER_REPLICATE if img.dtype == np.float32 else cv2.BORDER_CONSTANT)
    return out.reshape(-1)[:n].astype(np.float64)


def _esm(L: _Level, R: np.ndarray, gain: float, bias: float, iters: int, p: ResidualParams):
    """Returns R, gain, bias, info. Minimises sum w (B(H(R)x) - gain*A(x) - bias)^2 over the level's pixels."""
    K = L.K
    Kinv = np.linalg.inv(K)
    info = dict(converged=False, iters=0, sigma_rot=np.full(3, np.inf), support=0.0, n=L.n, scale_dn=np.nan)
    if L.n < 50:
        return R, gain, bias, info
    npar = 5 if p.photometric else 3
    for it in range(iters):
        H = K @ R.T @ Kinv
        X = H[0, 0] * L.xs + H[0, 1] * L.ys + H[0, 2]
        Y = H[1, 0] * L.xs + H[1, 1] * L.ys + H[1, 2]
        Z = H[2, 0] * L.xs + H[2, 1] * L.ys + H[2, 2]
        mx = X / Z
        my = Y / Z
        inside = (mx >= 1) & (mx <= L.w - 2) & (my >= 1) & (my <= L.h - 2)
        i1 = _sample(L.B, mx, my, cv2.INTER_CUBIC)
        g1x = _sample(L.gBx, mx, my, cv2.INTER_LINEAR)
        g1y = _sample(L.gBy, mx, my, cv2.INTER_LINEAR)
        vb = _sample(L.vB, mx, my, cv2.INTER_NEAREST) > 0
        m = inside & vb
        if m.sum() < 50:
            break
        gx = 0.5 * (L.g0x + g1x)
        gy = 0.5 * (L.g0y + g1y)
        Jg = gx[:, None] * L.Jx + gy[:, None] * L.Jy
        r = i1 - (gain * L.i0 + bias)
        rm = r[m][::7]  # robust scale from a subsample (median is the hot spot otherwise)
        med = np.median(rm)
        s = max(1.4826 * np.median(np.abs(rm - med)), 0.5)
        wr = 1.0 / (1.0 + (r / (p.cauchy_k * s)) ** 2)
        W = L.wp * wr * m
        if npar == 5:
            Am = np.concatenate([Jg, -L.i0[:, None], -np.ones((L.n, 1))], 1)
        else:
            Am = Jg
        AW = Am * W[:, None]
        N = AW.T @ Am
        g = AW.T @ r
        N[np.diag_indices(npar)] += 1e-9 * np.trace(N) / npar
        try:
            th = -np.linalg.solve(N, g)
        except np.linalg.LinAlgError:
            break
        R = _rotm(th[:3]) @ R
        if npar == 5:
            gain += th[3]
            bias += th[4]
        info['iters'] = it + 1
        step_px = float(px_equiv(th[:3], K, L.w, L.h))
        # formal covariance (optimistic: ignores spatially correlated errors)
        sw = W.sum()
        s2 = float((W * r * r).sum() / max(sw - npar, 1.0))
        try:
            C = np.linalg.inv(N) * s2
            info['sigma_rot'] = np.sqrt(np.maximum(np.diag(C)[:3], 0))
        except np.linalg.LinAlgError:
            pass
        info['support'] = float((wr[m] > 0.5).mean())
        info['scale_dn'] = float(s)
        if step_px < 2e-4:
            info['converged'] = True
            break
    return R, gain, bias, info


def _direct_stage(A, B, vA, vB, K, R0, wfull, p: ResidualParams, coarse: bool = True):
    R = R0.copy()
    gain, bias = 1.0, 0.0
    if coarse and p.iters_half > 0:
        L1 = _Level(A, B, vA, vB, K, wfull, p, 0.5, max(p.max_pixels // 4, 2000))
        R, gain, bias, _ = _esm(L1, R, gain, bias, p.iters_half, p)
    L0 = _Level(A, B, vA, vB, K, wfull, p, 1.0, p.max_pixels)
    R, gain, bias, info = _esm(L0, R, gain, bias, p.iters_full, p)
    info['gain'], info['bias'] = gain, bias
    return R, info


# ----------------------------------------------------------------------------- per pair (public helper)


def _weight_map(res_now: np.ndarray, res_stack: list, shape, p: ResidualParams) -> np.ndarray:
    """Soft far-field weight (temporally smoothed residual) x hard instantaneous independent-motion mask."""
    S = np.mean([np.minimum(r, 3.0) ** 2 for r in res_stack], axis=0)
    soft = 1.0 / (1.0 + S / p.parallax_c_px ** 2)
    hard = (res_now < p.hard_tol_px).astype(np.uint8)
    if p.erode_cells > 0:
        e = 2 * p.erode_cells + 1
        hard = cv2.erode(hard, np.ones((e, e), np.uint8))
    wg = (soft * hard).astype(np.float32)
    return cv2.resize(wg, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _finish(fa: dict, R: np.ndarray, info: dict, K, shape, R_exp) -> dict:
    Hh, Ww = shape
    rv = _logm(R)
    agree = float(px_equiv(_logm(fa['R_flow'].T @ R), K, Ww, Hh))
    sig = info['sigma_rot']
    sigma_px = float(px_equiv(sig, K, Ww, Hh)) if np.all(np.isfinite(sig)) else float('inf')
    inl = fa['inlier_frac']
    c_inl = np.clip((inl - 0.03) / 0.2, 0.0, 1.0)
    c_sig = 1.0 / (1.0 + (sigma_px / 0.01) ** 2)
    c_agree = float(np.exp(-(max(0.0, agree - 0.3) / 1.0) ** 2))
    c_sup = np.clip((info['support'] - 0.3) / 0.3, 0.0, 1.0)
    conf = float(c_inl * c_sig * c_agree * c_sup) if (fa['ok'] and info['iters'] > 0) else 0.0
    method = 0
    epi = dict(epi_conf=0.0, epi_inlier_frac=fa.get('epi_inlier_frac', 0.0),
               epi_sigma_px=fa.get('epi_sigma_px', float('inf')), epi_agree_px=float('nan'))
    if fa.get('epi_ok'):
        # the direct (photometric) rotation fit masks everything off the far field; with strong parallax /
        # a large moving object it has too little support. The depth-free fit uses every consistent pixel.
        Re = fa['R_epi']
        ag_e = float(px_equiv(_logm(Re.T @ R), K, Ww, Hh))
        c_e = np.clip((epi['epi_inlier_frac'] - 0.35) / 0.3, 0.0, 1.0) \
            / (1.0 + (epi['epi_sigma_px'] / 0.02) ** 2)
        epi.update(epi_conf=float(c_e), epi_agree_px=ag_e)
        if 0.8 * c_e > conf:
            R = Re
            rv = _logm(R)
            conf = float(0.8 * c_e)
            method = 1
    err = _logm(R_exp.T @ R)
    return dict(rotvec=rv, rotvec_flow=_logm(fa['R_flow']), err=err, conf=conf, inlier_frac=inl,
                parallax_px=fa['parallax_px'], sigma_px=sigma_px, agree_px=agree,
                support=info['support'], converged=info['converged'], gain=info.get('gain', 1.0),
                n_pix=info['n'], method=method, **epi)


def measure_pair(A: np.ndarray, B: np.ndarray, K: np.ndarray, R_exp: np.ndarray | None = None,
                 validA: np.ndarray | None = None, validB: np.ndarray | None = None,
                 params: ResidualParams | None = None) -> dict:
    """Single-pair measurement (no temporal smoothing of the far-field weight). See measure_residuals."""
    p = params or ResidualParams()
    A, B = _as_u8(A), _as_u8(B)
    vA = _valid_mask(A, validA, p.invalid_dilate_px)
    vB = _valid_mask(B, validB, p.invalid_dilate_px)
    R_exp = np.eye(3) if R_exp is None else np.asarray(R_exp, dtype=np.float64)
    fa = _flow_stage(A, B, vA, vB, K, R_exp, p)
    w = _weight_map(fa['res'], [fa['res']], A.shape, p)
    R, info = _direct_stage(A, B, vA, vB, K, fa['R_flow'], w, p)
    return _finish(fa, R, info, K, A.shape, R_exp)


def _as_u8(img) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[..., 0]
    if img.dtype != np.uint8:
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


# ----------------------------------------------------------------------------- main API


def _exp_matrix(expected_rel_q, i: int, k0: int, k1: int) -> np.ndarray:
    if expected_rel_q is None:
        return np.eye(3)
    if callable(expected_rel_q):
        q = expected_rel_q(k0, k1)
    else:
        q = np.asarray(expected_rel_q)[i]
    return quat_to_mat(np.asarray(q, dtype=np.float64))


def _parent_watchdog(parent_pid: int, period: float = 0.25):
    """Worker side: exit as soon as the parent process is gone (macOS has no PR_SET_PDEATHSIG; an orphan is
    re-parented to launchd, so getppid() changes). Without this, a hard-killed analysis leaves spawn_main
    workers running forever."""
    def run():
        while True:
            time.sleep(period)
            if os.getppid() != parent_pid:
                os._exit(0)
    th = threading.Thread(target=run, daemon=True, name='parent-watchdog')
    th.start()


def _init_worker(parent_pid: int | None = None):
    cv2.setNumThreads(1)
    if parent_pid:
        _parent_watchdog(int(parent_pid))


def _ping(dt: float = 0.0):
    if dt > 0:
        time.sleep(dt)
    return os.getpid()


class _hide_main_file:
    """While spawning worker processes, stop them from re-running a plain calling SCRIPT (multiprocessing's
    spawn re-imports __main__ by path; the workers only need engine modules). `python -m ...`, `-c` and
    pytest are unaffected."""

    def __enter__(self):
        import sys
        self.main = sys.modules.get('__main__')
        self.saved = None
        if self.main is not None and getattr(self.main, '__spec__', None) is None and '__file__' in self.main.__dict__:
            self.saved = self.main.__dict__.pop('__file__')
        self.env = {k: os.environ.get(k) for k in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
        for k in self.env:
            os.environ[k] = '1'
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            self.main.__file__ = self.saved
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def default_processes() -> int:
    """Measurement worker count: all performance cores but one-two (the main process decodes, renders previews
    on the GPU and feeds the pool). M4 Pro (10P+4E): 9."""
    n = os.cpu_count() or 4
    try:
        import subprocess
        p = int(subprocess.run(['sysctl', '-n', 'hw.perflevel0.physicalcpu'], capture_output=True, text=True,
                               timeout=2).stdout.strip())
        if p > 0:
            return max(1, min(p - 1, n - 2, 12))
    except Exception:
        pass
    return max(1, min(8, n - 2))


def make_pool(processes: int | None = None):
    """A process pool for measure_residuals(executor=...) (reuse it across closed-loop passes; shut it down
    with shutdown_pool() when done). processes None -> default_processes(). Workers exit on their own when the
    parent dies (parent watchdog), so a hard-killed analysis leaves no orphans."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    n = processes if processes else default_processes()
    with _hide_main_file():
        ex = ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context('spawn'), initializer=_init_worker,
                                 initargs=(os.getpid(),))
        # spawn every worker now (inside the guard); ProcessPoolExecutor otherwise spawns on demand
        list(ex.map(_ping, [0.3] * n))
    ex.n_workers = n
    return ex


def pool_pids(ex) -> list:
    procs = getattr(ex, '_processes', None) or {}
    return [p.pid for p in list(procs.values()) if p is not None and p.pid]


def shutdown_pool(ex, timeout: float = 3.0):
    """Stop a make_pool() executor NOW: cancel queued work, terminate the workers (they may be in the middle of a
    multi-second chunk), and reap them. Safe to call twice."""
    if ex is None:
        return
    procs = list((getattr(ex, '_processes', None) or {}).values())
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass
    t_end = time.monotonic() + timeout
    for p in procs:
        try:
            p.join(max(0.0, t_end - time.monotonic()))
            if p.is_alive():
                p.kill()
                p.join(1.0)
        except Exception:
            pass


def _chunk_job(payload):
    """Measure pairs [lo, hi) of a chunk of consecutive items (with +-temporal_radius pairs of context)."""
    ks, imgs, vpk, K, Rexp, p, lo, hi, consec = payload
    c0 = time.process_time()
    n = len(ks)
    Hh, Ww = imgs.shape[1:]
    vs = []
    for i in range(n):
        extra = None if vpk[i] is None else np.unpackbits(vpk[i])[:Hh * Ww].reshape(Hh, Ww)
        vs.append(_valid_mask(imgs[i], extra, p.invalid_dilate_px))
    fa = {}
    for i in range(n - 1):
        if consec and ks[i + 1] != ks[i] + 1:
            continue
        fa[i] = _flow_stage(imgs[i], imgs[i + 1], vs[i], vs[i + 1], K, Rexp[i], p)
    out = []
    r = p.temporal_radius
    for i in range(lo, hi):
        if i not in fa:
            continue
        stack = [fa[j]['res'] for j in range(i - r, i + r + 1) if j in fa]
        w = _weight_map(fa[i]['res'], stack, imgs[i].shape, p)
        R, info = _direct_stage(imgs[i], imgs[i + 1], vs[i], vs[i + 1], K, fa[i]['R_flow'], w, p,
                                coarse=fa[i]['inlier_frac'] < 0.5)
        res = _finish(fa[i], R, info, K, imgs[i].shape, Rexp[i])
        res.update(k0=int(ks[i]), k1=int(ks[i + 1]), expected=_logm(Rexp[i]))
        out.append(res)
    if out:
        cpu = (time.process_time() - c0) / len(out)
        for rr in out:
            rr['cpu_s'] = cpu
    return out


def _measure_mp(frames: Iterable, K, expected_rel_q, p: ResidualParams, executor, progress, cancel,
                consecutive_only: bool, chunk: int):
    """Process-pool version of the measurement loop (the thread version is GIL-bound at ~2.6x on 8 threads)."""
    from collections import deque
    r = p.temporal_radius
    items: dict[int, tuple] = {}                 # item index -> (k, u8, packed valid | None)
    it = iter(frames)
    n_read = 0
    exhausted = False
    t_start = time.perf_counter()
    inflight: deque = deque()
    results: list = []
    # bounded: every in-flight chunk holds ~(chunk + 2r) frames (0.5-0.7 MB each) pickled in the call queue
    max_inflight = getattr(executor, 'n_workers', 8) + 1
    done_pairs = 0
    tm = dict(feed_s=0.0, pool_wait_s=0.0, submit_s=0.0)

    def read_until(n):
        nonlocal n_read, exhausted
        t0 = time.perf_counter()
        while not exhausted and n_read <= n:
            try:
                item = next(it)
            except StopIteration:
                exhausted = True
                break
            img = _as_u8(item[1])
            extra = item[2] if len(item) > 2 else None
            if extra is not None:
                extra = np.asarray(extra).astype(bool)
                if extra.all():                     # most previews are fully inside the source: send nothing
                    extra = None
            vp = None if extra is None else np.packbits(extra.reshape(-1))
            items[n_read] = (int(item[0]), img, vp)
            n_read += 1
        tm['feed_s'] += time.perf_counter() - t0

    def wait_first():
        """Block until the oldest chunk is done, polling cancel() every 0.2 s (a chunk takes seconds)."""
        from concurrent.futures import wait as _wait
        fut = inflight[0][1]
        t0 = time.perf_counter()
        while not fut.done():
            if cancel is not None and cancel():
                raise InterruptedError('measurement cancelled')
            _wait([fut], timeout=0.2)
        tm['pool_wait_s'] += time.perf_counter() - t0

    def collect(block: bool):
        nonlocal done_pairs
        while inflight and (block or inflight[0][1].done()):
            if block:
                wait_first()
            i1, fut = inflight.popleft()
            out = fut.result()
            results.extend(out)
            done_pairs = i1
            if progress is not None:
                progress(i1)
            block = False if len(inflight) < max_inflight else block

    i0 = 0
    try:
        while True:
            if cancel is not None and cancel():
                raise InterruptedError('measurement cancelled')
            read_until(i0 + chunk + r + 1)
            n_pairs = n_read - 1
            if n_pairs <= i0:
                break
            i1 = min(i0 + chunk, n_pairs)
            t_sub = time.perf_counter()
            a = max(0, i0 - r)
            b = min(n_read - 1, i1 + r)            # last item index included
            idx = list(range(a, b + 1))
            ks = np.array([items[j][0] for j in idx], dtype=np.int64)
            imgs = np.stack([items[j][1] for j in idx])
            vpk = [items[j][2] for j in idx]
            Rexp = [_exp_matrix(expected_rel_q, j, items[j][0], items[j + 1][0]) if j + 1 <= b else np.eye(3)
                    for j in idx]
            payload = (ks, imgs, vpk, K, Rexp, p, i0 - a, i1 - a, consecutive_only)
            inflight.append((i1, executor.submit(_chunk_job, payload)))
            tm['submit_s'] += time.perf_counter() - t_sub
            if len(inflight) >= max_inflight:
                collect(True)
            else:
                collect(False)
            for j in [j for j in items if j < i1 - r]:
                del items[j]
            i0 = i1
        while inflight:
            collect(True)
    finally:
        for _, f in inflight:
            f.cancel()
    _measure_mp.last_timing = tm
    return results, time.perf_counter() - t_start


def measure_residuals(frames: Iterable, K: np.ndarray, expected_rel_q=None,
                      params: ResidualParams | None = None, workers: int | None = None,
                      progress: Callable[[int], None] | None = None, *, executor=None,
                      consecutive_only: bool = False, cancel: Callable[[], bool] | None = None,
                      chunk: int = 48) -> dict:
    """Measure the inter-frame rotation of consecutive RECTILINEAR preview frames.

    frames: iterable of (k, gray) or (k, gray, valid_mask); gray uint8 (or float 0..255) HxW, all the same
        size, rendered with pinhole intrinsics K (3x3, pixel-centre convention). Pure-black pixels (0) are
        treated as invalid border. Consecutive items form the pairs (k_i, k_{i+1}), in iteration order.
    K: 3x3 pinhole matrix of the previews.
    expected_rel_q: None (identity), an array (P,4) aligned with the pairs in iteration order, or a callable
        (k0, k1) -> (4,) quaternion. Each is conj(V_k0) * V_k1 of the INTENDED virtual path (w,x,y,z).

    Returns dict of arrays over the P pairs:
        k0, k1          frame indices of each pair
        rotvec   (P,3)  measured relative rotation log(W_k0^T W_k1), camera frame of k0, rad
        err_rotvec (P,3) log(R_exp^T R_meas) (~ eps_k1 - eps_k0), rad  [= rotvec when expected is None]
        expected_rotvec (P,3)
        conf     (P,)   0..1 (0 = do not trust; closedloop interpolates across these)
        inlier_frac (P,) fraction of textured valid pixels consistent with the rotation (flow |res| < 1 px)
        parallax_px (P,) RMS flow residual (< 3 px) after the rotation fit: DIS noise ~0.1-0.2 px; larger =
                        parallax / deformation present
        sigma_px (P,)   formal 1-sigma precision (px-equivalent; optimistic, use relatively)
        agree_px (P,)   |flow-fit rotation - direct rotation| (px-equivalent)
        rotvec_flow (P,3) the flow-stage estimate (diagnostic)
        px_per_rad, width, height, timing {pairs_per_s, ...}

    executor: a process pool from make_pool() -> chunked multi-process measurement (same results; ~3x the
        pairs/s of the thread pool, which is GIL-bound). consecutive_only: skip pairs whose frame indices are
        not consecutive (partial passes over several runs of frames). cancel: polled per chunk (raises
        InterruptedError).
    """
    p = params or ResidualParams()
    K = np.asarray(K, dtype=np.float64)
    if executor is not None:
        res_list, elapsed = _measure_mp(frames, K, expected_rel_q, p, executor, progress, cancel,
                                        consecutive_only, max(8, int(chunk)))
        return _pack(res_list, K, elapsed, dict(workers=getattr(executor, 'n_workers', None), processes=True,
                                                **getattr(_measure_mp, 'last_timing', {})))
    if workers is None:
        workers = p.workers
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    t_start = time.perf_counter()
    it = iter(frames)
    buf: dict[int, tuple] = {}          # item index -> (k, u8, valid)
    n_read = 0
    exhausted = False

    def read_until(n):
        nonlocal n_read, exhausted
        while not exhausted and n_read <= n:
            try:
                item = next(it)
            except StopIteration:
                exhausted = True
                break
            k, img = item[0], _as_u8(item[1])
            extra = item[2] if len(item) > 2 else None
            buf[n_read] = (int(k), img, _valid_mask(img, extra, p.invalid_dilate_px))
            n_read += 1

    fa_res: dict[int, dict] = {}
    out: dict[int, dict] = {}
    r = p.temporal_radius
    prev_threads = cv2.getNumThreads()
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    if pool is not None:
        cv2.setNumThreads(1)

    def pmap(fn, items):
        return list(pool.map(fn, items)) if pool is not None else [fn(x) for x in items]

    def job_a(i):
        k0, A, vA = buf[i]
        k1, B, vB = buf[i + 1]
        if consecutive_only and k1 != k0 + 1:
            return i, None
        return i, _flow_stage(A, B, vA, vB, K, _exp_matrix(expected_rel_q, i, k0, k1), p)

    def job_c(i):
        k0, A, vA = buf[i]
        k1, B, vB = buf[i + 1]
        fa = fa_res[i]
        if fa is None:
            return i, None
        stack = [fa_res[j]['res'] for j in range(i - r, i + r + 1) if fa_res.get(j) is not None]
        w = _weight_map(fa['res'], stack, A.shape, p)
        R_exp = _exp_matrix(expected_rel_q, i, k0, k1)
        R, info = _direct_stage(A, B, vA, vB, K, fa['R_flow'], w, p, coarse=fa['inlier_frac'] < 0.5)
        res = _finish(fa, R, info, K, A.shape, R_exp)
        res.update(k0=k0, k1=k1, expected=_logm(R_exp))
        return i, res

    t_a = t_c = 0.0
    try:
        i0 = 0
        while True:
            if cancel is not None and cancel():
                raise InterruptedError('measurement cancelled')
            read_until(i0 + p.chunk + r + 1)
            n_pairs = n_read - 1
            if n_pairs <= i0:
                break
            i1 = min(i0 + p.chunk, n_pairs)
            need = [i for i in range(max(0, i0 - r), min(n_pairs, i1 + r)) if i not in fa_res]
            t0 = time.perf_counter()
            for i, fa in pmap(job_a, need):
                fa_res[i] = fa
            t1 = time.perf_counter()
            for i, res in pmap(job_c, list(range(i0, i1))):
                if res is not None:
                    out[i] = res
            t_c += time.perf_counter() - t1
            t_a += t1 - t0
            for j in [j for j in fa_res if j < i1 - r]:
                del fa_res[j]
            for j in [j for j in buf if j < i1]:
                del buf[j]
            if progress is not None:
                progress(i1)
            i0 = i1
    finally:
        if pool is not None:
            pool.shutdown()
            cv2.setNumThreads(prev_threads)

    idx = sorted(out)
    res_list = [out[i] for i in idx]
    elapsed = time.perf_counter() - t_start
    return _pack(res_list, K, elapsed, dict(flow_stage_s=t_a, direct_stage_s=t_c, workers=workers))


def _pack(res_list: list, K: np.ndarray, elapsed: float, timing: dict) -> dict:
    P = len(res_list)

    def stack(key, dim=None):
        if P == 0:
            return np.zeros((0, dim)) if dim else np.zeros(0)
        return np.array([rr[key] for rr in res_list], dtype=np.float64)

    cpu = float(sum(rr.get('cpu_s', 0.0) for rr in res_list))
    if cpu > 0:
        timing = dict(timing, worker_cpu_s=cpu, pairs_per_worker_cpu_s=P / cpu,
                      pool_utilization=cpu / max(elapsed * (timing.get('workers') or 1), 1e-9))
    return dict(k0=np.array([rr['k0'] for rr in res_list], dtype=np.int64),
                k1=np.array([rr['k1'] for rr in res_list], dtype=np.int64),
                rotvec=stack('rotvec', 3), err_rotvec=stack('err', 3), expected_rotvec=stack('expected', 3),
                rotvec_flow=stack('rotvec_flow', 3), conf=stack('conf'), inlier_frac=stack('inlier_frac'),
                parallax_px=stack('parallax_px'), sigma_px=stack('sigma_px'), agree_px=stack('agree_px'),
                support=stack('support'), gain=stack('gain'), method=stack('method'),
                epi_conf=stack('epi_conf'), epi_inlier_frac=stack('epi_inlier_frac'),
                epi_sigma_px=stack('epi_sigma_px'), epi_agree_px=stack('epi_agree_px'),
                px_per_rad=float(0.5 * (K[0, 0] + K[1, 1])),
                timing=dict(elapsed_s=elapsed, pairs_per_s=P / max(elapsed, 1e-9), **timing))
