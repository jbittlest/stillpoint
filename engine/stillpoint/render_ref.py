"""Reference + preview renderer — the SAME math as shaders/warp.metal (ENGINE_SPEC.md §1-§3).  Owner: WP-B.

Geometry (identical in warp.metal::sp_source_coord and _source_coord below):
    r_v = ((X - out_cx)/out_fx, (Y - out_cy)/out_fx, 1),  out_cx = (out_w-1)/2, out_cy = (out_h-1)/2
    3 evaluations: y0 = (src_h-1)/2; eval -> v0; y1 = v0; eval -> v1; y2 = secant root of g(y)=v(y)-y through
    (y0,g0),(y1,g1) [fallback y2 = v1 if the implied contraction |c| > 0.9 or y1 == y0]; eval -> final (u, v).
    Each eval: M(y) = row_mats lerped at g = clip(y*(R-1)/(src_h-1), 0, R-1); r_c = M(y) r_v; (u,v) = lens.project(r_c).
    Valid iff r_c.z > 0 and -0.5 <= u <= src_w-0.5 and -0.5 <= v <= src_h-0.5.

SCALING CONVENTION (pixel centres at integers everywhere):
    * An image scaled by s has pixel p_s = s*(p+0.5) - 0.5 for full-res pixel p; inverse p = (p_s+0.5)/s - 0.5.
    * out_scale: the output grid has Wo = round(out_w*out_scale), Ho = round(out_h*out_scale) pixels and per-axis
      scales sox = Wo/out_w, soy = Ho/out_h. Output pixel (i,j) is full-res output X = (i+0.5)/sox - 0.5, etc.
      Its camera matrix is preview_K(): fx*sox, fx*soy, cx = (out_cx+0.5)*sox-0.5 = Wo/2-0.5 (square when exact).
    * source_map() always returns FULL-RES source luma pixel coordinates.
    * Previews sample a source decoded at a reduced size (bw,bh): buffer pixel s = (bw/src_w)*(S+0.5) - 0.5.

Public API:
    source_map(plan, k, out_scale=1.0, iters=3, return_valid=False) -> (Ho,Wo,2) float64 [, (Ho,Wo) bool]
    source_points(plan, k, xy (N,2) full-res output px) -> (S (N,2), valid (N,))
    preview_K(plan, k, out_scale) -> 3x3
    render_frames(plan, video_path, frames, out_scale=0.25, gray=True, ...) -> Iterator[(k, uint8 (Ho,Wo))]
    render_planes_ref(plan, k, y, uv, kernel='lanczos3') -> (Y u16, UV u16) 10-bit MSB-aligned (golden reference)
    metal_coord_map(plan, k, out_scale=1.0) -> the Metal kernel's own map via torch MPS (float32)
    mesh_offset(mesh_k, X, Y, out_w, out_h) -> (dx, dy): the mesh-residual displacement (warp.metal sp_mesh_offset)

MESH RESIDUAL (optional, plan.mesh (F, ny, nx, 2), engine/stillpoint/mesh.py): every function above first displaces
the full-res output pixel X by the bilinear offset D_k(X) (X' = X + D_k(X)), then applies the rotation mapping —
the same as the *_mesh Metal kernels. use_mesh=False ignores plan.mesh.
"""
from __future__ import annotations

import os
import queue
import threading
from typing import Iterator, Optional

import numpy as np

from .types import Plan

SHADER_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'shaders', 'warp.metal'))

# parameter block indices — must match shaders/warp.metal
P_OUT_FX, P_OUT_CX, P_OUT_CY, P_LENS_MODEL = 0, 1, 2, 3
P_FX, P_FY, P_CX, P_CY, P_K1 = 4, 5, 6, 7, 8
P_SRC_W, P_SRC_H, P_N_ROWS, P_ITERS = 12, 13, 14, 15
P_OUT_SX, P_OUT_SY, P_SRC_SX, P_SRC_SY = 16, 17, 18, 19
P_KERNEL, P_IN_SCALE, P_BLACK_Y, P_BLACK_C = 20, 21, 22, 23
P_DST_W, P_DST_H, P_BUF_W, P_BUF_H = 24, 25, 26, 27
P_MESH_NX, P_MESH_NY = 30, 31          # (28, 29 belong to Stillpoint.app's appended kernels)
P_COUNT = 32
KERNELS = {'lanczos3': 0, 'catmullrom': 1, 'bilinear': 2}


# ============================================================================================ geometry
def output_grid(out_w: int, out_h: int, out_scale: float = 1.0) -> tuple[int, int, float, float]:
    Wo, Ho = int(round(out_w * out_scale)), int(round(out_h * out_scale))
    return Wo, Ho, Wo / out_w, Ho / out_h


def preview_K(plan: Plan, k: int, out_scale: float = 1.0) -> np.ndarray:
    """Camera matrix of the (scaled) rectilinear output of frame k."""
    Wo, Ho, sx, sy = output_grid(plan.out_w, plan.out_h, out_scale)
    fx = float(plan.out_fx[k])
    return np.array([[fx * sx, 0.0, ((plan.out_w - 1) / 2 + 0.5) * sx - 0.5],
                     [0.0, fx * sy, ((plan.out_h - 1) / 2 + 0.5) * sy - 0.5],
                     [0.0, 0.0, 1.0]])


def mesh_offset(mesh_k: np.ndarray, X: np.ndarray, Y: np.ndarray, out_w: int, out_h: int):
    """Bilinear mesh-residual offset (dx, dy) (full-res output px) at full-res output pixels X, Y (any shape) for one
    frame's vertex grid mesh_k (ny, nx, 2); clamped outside the grid. float64 twin of warp.metal sp_mesh_offset."""
    m = np.asarray(mesh_k, dtype=np.float64)
    ny, nx = m.shape[:2]
    gx = np.clip(np.asarray(X, dtype=np.float64) * (nx - 1) / max(out_w - 1, 1), 0.0, nx - 1)
    gy = np.clip(np.asarray(Y, dtype=np.float64) * (ny - 1) / max(out_h - 1, 1), 0.0, ny - 1)
    i0 = np.minimum(np.floor(gx).astype(np.int64), nx - 2)
    j0 = np.minimum(np.floor(gy).astype(np.int64), ny - 2)
    fx = (gx - i0)[..., None]
    fy = (gy - j0)[..., None]
    v00, v10 = m[j0, i0], m[j0, i0 + 1]
    v01, v11 = m[j0 + 1, i0], m[j0 + 1, i0 + 1]
    t = v00 + (v10 - v00) * fx
    u = v01 + (v11 - v01) * fx
    d = t + (u - t) * fy
    return d[..., 0], d[..., 1]


def plan_mesh(plan: Plan, use_mesh: bool = True):
    """plan.mesh if the plan has one and use_mesh, else None."""
    m = getattr(plan, 'mesh', None)
    return m if (use_mesh and m is not None) else None


def _rows_apply(M: np.ndarray, src_h: int, y: np.ndarray, rv: np.ndarray) -> np.ndarray:
    R = M.shape[0]
    g = np.clip(y * (R - 1) / (src_h - 1), 0.0, R - 1)
    j0 = np.minimum(np.floor(g).astype(np.int64), R - 2)
    f = (g - j0)[..., None]
    ra = np.einsum('...ij,...j->...i', M[j0], rv)
    rb = np.einsum('...ij,...j->...i', M[j0 + 1], rv)
    return ra + (rb - ra) * f


def _source_coord(plan: Plan, k: int, X: np.ndarray, Y: np.ndarray, iters: int = 3,
                  rows: Optional[np.ndarray] = None, secant: bool = True, use_mesh: bool = True):
    """Full-res output pixel coords (any shape) -> (S (…,2) float64, valid bool, z).
    secant=False gives the plain fixed-point iteration (diagnostics only; the kernel always uses the secant step).
    With a plan.mesh (and use_mesh) the output pixel is first displaced by the mesh-residual offset."""
    mesh = plan_mesh(plan, use_mesh)
    if mesh is not None:
        dx, dy = mesh_offset(mesh[k], X, Y, plan.out_w, plan.out_h)
        X = np.asarray(X, dtype=np.float64) + dx
        Y = np.asarray(Y, dtype=np.float64) + dy
    fx = float(plan.out_fx[k])
    cx, cy = (plan.out_w - 1) / 2.0, (plan.out_h - 1) / 2.0
    M = np.asarray(plan.row_mats[k] if rows is None else rows, dtype=np.float64)
    rv = np.stack([(X - cx) / fx, (Y - cy) / fx, np.ones_like(X, dtype=np.float64)], axis=-1)
    H = plan.src_h
    y = np.full(X.shape, 0.5 * (H - 1), dtype=np.float64)
    ya = np.zeros_like(y)
    ga = np.zeros_like(y)
    uv = z = None
    for e in range(max(int(iters), 1)):
        rc = _rows_apply(M, H, y, rv)
        uv = plan.lens.project(rc)
        z = rc[..., 2]
        g = uv[..., 1] - y
        yn = uv[..., 1].copy()
        if e >= 1 and secant:
            dy = y - ya
            with np.errstate(divide='ignore', invalid='ignore'):
                s = (g - ga) / dy
                use = (dy != 0) & (s <= -0.1) & (s >= -2.0)
                yn = np.where(use, y - g / np.where(use, s, 1.0), yn)
        ya, ga, y = y, g, yn
    valid = (z > 0) & (uv[..., 0] >= -0.5) & (uv[..., 1] >= -0.5) & \
            (uv[..., 0] <= plan.src_w - 0.5) & (uv[..., 1] <= plan.src_h - 0.5)
    return uv, valid, z


def source_map(plan: Plan, k: int, out_scale: float = 1.0, iters: int = 3, return_valid: bool = False,
               chunk_rows: int = 256, use_mesh: bool = True):
    """(Ho,Wo,2) FULL-RES source luma coordinates (float64) of the scaled output grid of plan frame k."""
    Wo, Ho, sx, sy = output_grid(plan.out_w, plan.out_h, out_scale)
    X1 = (np.arange(Wo) + 0.5) / sx - 0.5
    out = np.empty((Ho, Wo, 2))
    val = np.empty((Ho, Wo), dtype=bool)
    for r0 in range(0, Ho, chunk_rows):
        r1 = min(Ho, r0 + chunk_rows)
        Y1 = (np.arange(r0, r1) + 0.5) / sy - 0.5
        X, Y = np.meshgrid(X1, Y1)
        S, v, _ = _source_coord(plan, k, X, Y, iters, use_mesh=use_mesh)
        out[r0:r1] = S
        val[r0:r1] = v
    return (out, val) if return_valid else out


def source_points(plan: Plan, k: int, xy: np.ndarray, iters: int = 3, use_mesh: bool = True):
    """Arbitrary FULL-RES output points (N,2) of plan frame k -> (source coords (N,2) float64, valid (N,) bool).
    Same math as the kernel — use it e.g. for crop checks on boundary samples."""
    xy = np.asarray(xy, dtype=np.float64)
    S, ok, _ = _source_coord(plan, k, xy[..., 0], xy[..., 1], iters, use_mesh=use_mesh)
    return S, ok


# ============================================================================================ reference sampler
def _weights(f: np.ndarray, kernel: str):
    """f in [0,1) -> (offset, (…,N) normalised weights) — float64 twin of warp.metal sp_weights."""
    if kernel == 'lanczos3':
        i = np.arange(-2, 4)
        x = f[..., None] - i
        w = np.sinc(x) * np.sinc(x / 3.0)
        return -2, w / w.sum(-1, keepdims=True)
    if kernel == 'catmullrom':
        t, t2, t3 = f, f * f, f * f * f
        w = np.stack([-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                      0.5 * t3 - 0.5 * t2], axis=-1)
        return -1, w
    if kernel == 'bilinear':
        return 0, np.stack([1.0 - f, f], axis=-1)
    raise ValueError(kernel)


def sample_ref(img: np.ndarray, sx: np.ndarray, sy: np.ndarray, kernel: str = 'lanczos3') -> np.ndarray:
    """Separable resampling of img (h,w[,c]) at float coords (clamp-to-edge taps), float64."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    flx, fly = np.floor(sx), np.floor(sy)
    offx, wx = _weights(sx - flx, kernel)
    _, wy = _weights(sy - fly, kernel)
    bx, by = flx.astype(np.int64) + offx, fly.astype(np.int64) + offx
    N = wx.shape[-1]
    acc = 0.0
    for j in range(N):
        yy = np.clip(by + j, 0, h - 1)
        row = 0.0
        for i in range(N):
            xx = np.clip(bx + i, 0, w - 1)
            v = img[yy, xx]
            row = row + (wx[..., i, None] * v if img.ndim == 3 else wx[..., i] * v)
        acc = acc + (wy[..., j, None] * row if img.ndim == 3 else wy[..., j] * row)
    return acc


def _q10(code: np.ndarray) -> np.ndarray:
    return (np.clip(np.floor(code + 0.5), 0, 1023) * 64).astype(np.uint16)


def render_planes_ref(plan: Plan, k: int, y: np.ndarray, uv: np.ndarray, kernel: str = 'lanczos3',
                      iters: int = 3, black_y: int = 64, black_c: int = 512, rows=None):
    """Float64 reference of sprender's output planes for plan frame k.

    y: (src_h, src_w) uint8 (8-bit source) or uint16 (P010, MSB-aligned); uv: (src_h/2, src_w/2, 2) same dtype.
    Returns (Y (out_h,out_w) uint16, UV (out_h/2,out_w/2,2) uint16), 10-bit codes MSB-aligned (code*64)."""
    to_code = (lambda a: a.astype(np.float64) * 4.0) if y.dtype == np.uint8 else (lambda a: a.astype(np.float64) / 64.0)
    ycode, ccode = to_code(y), to_code(uv)
    W, H = plan.out_w, plan.out_h
    X, Y = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    S, ok, _ = _source_coord(plan, k, X, Y, iters, rows)
    out_y = _q10(sample_ref(ycode, S[..., 0], S[..., 1], kernel))
    out_y[~ok] = black_y * 64
    Xc, Yc = np.meshgrid(2.0 * np.arange(W // 2), 2.0 * np.arange(H // 2) + 0.5)
    Sc, okc, _ = _source_coord(plan, k, Xc, Yc, iters, rows)
    out_c = _q10(sample_ref(ccode, Sc[..., 0] * 0.5, (Sc[..., 1] - 0.5) * 0.5, kernel))
    out_c[~okc] = black_c * 64
    return out_y, out_c


# ============================================================================================ fill reference
def _smooth01(e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip(x / max(float(e1), 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _edge_ext_ref(img: np.ndarray, sx: np.ndarray, sy: np.ndarray, blur: float) -> np.ndarray:
    """warp.metal sp_edge_ext twin: nearest edge point, 3x3 bilinear blur of radius 1 + blur * overshoot."""
    h, w = img.shape[:2]
    cx, cy = np.clip(sx, 0, w - 1), np.clip(sy, 0, h - 1)
    r = 1.0 + blur * np.hypot(sx - cx, sy - cy)
    acc = 0.0
    for j in (-1, 0, 1):
        for i in (-1, 0, 1):
            acc = acc + sample_ref(img, cx + r * i, cy + r * j, 'bilinear')
    return acc / 9.0


def _fill_plane_ref(plan: Plan, fill, k: int, X, Y, imgs: list, rows: list, cmap, kernel: str, iters: int,
                    luma: bool, FP: np.ndarray, in_scale: float, black_out: int):
    """One output plane of the fill kernels (float64). imgs: normalised planes [main, source 0, ...]."""
    from .fill import (F_BLACK, F_DOFF, F_FB_BLUR, F_FEATHER_MAIN, F_FEATHER_NB, F_GAIN, F_N_SRC, F_PAR, F_SIGMA,
                       F_FB_W, F_WEIGHT, inside_distance, mesh_disp)
    S0, _, z0 = _source_coord(plan, k, X, Y, iters)
    d0 = inside_distance(plan, S0, z0)
    n = int(FP[F_N_SRC])
    fm = float(FP[F_FEATHER_MAIN])
    cx, cy = cmap(S0)
    vm = sample_ref(imgs[0], cx, cy, kernel)
    two = vm.ndim == X.ndim + 1
    q = lambda v: np.clip(np.floor(v * in_scale * (65535.0 / 64.0) + 0.5), 0, 1023) * 64
    valid = d0 >= 0
    out = np.where(valid[..., None] if two else valid, q(vm), black_out * 64).astype(np.float64)
    sel = d0 < fm
    if sel.any():
        Xs, Ys, d0s = X[sel], Y[sel], d0[sel]
        vms = np.where((d0s >= 0)[:, None] if two else d0s >= 0, vm[sel], 0.0)
        disp = mesh_disp(fill.mesh[k], Xs, Ys, plan.out_w, plan.out_h) if fill.mesh is not None \
            else np.zeros(Xs.shape + (2,))
        cv, cw = [np.zeros_like(vms)], [np.zeros(len(Xs))]        # a zero-weight dummy when n == 0
        for i in range(n):
            dd = float(FP[F_DOFF + i])
            Si, _, zi = _source_coord(plan, k, Xs + dd * disp[..., 0], Ys + dd * disp[..., 1], iters, rows=rows[i])
            di = inside_distance(plan, Si, zi)
            w_ = np.where(di > 0, _smooth01(FP[F_FEATHER_NB], di) * float(FP[F_WEIGHT + i]), 0.0)
            if FP[F_PAR] > 0:                                        # parallax-aware weight (sp_par_w)
                w_ = w_ * np.exp(-(abs(dd) * np.hypot(disp[..., 0], disp[..., 1]) / float(FP[F_PAR])) ** 2)
            sx_, sy_ = cmap(Si)
            v = sample_ref(imgs[i + 1], sx_, sy_, kernel)
            if luma:
                v = (v - FP[F_BLACK]) * float(FP[F_GAIN + i]) + FP[F_BLACK]
            v = np.where((di > 0)[:, None] if two else di > 0, v, 0.0)
            cv.append(v)
            cw.append(w_)
        cv = np.stack(cv[1:] if n else cv, 0)                        # (n, N[,2])
        cw = np.stack(cw[1:] if n else cw, 0)                        # (n, N)
        n_ = cv.shape[0]
        best = np.where(cw.max(0) > 0, np.argmax(cw, axis=0), -1)    # first max (the kernel uses strict >)
        sig = float(FP[F_SIGMA])
        wf = cw.copy()
        if sig > 0:
            idx = np.clip(best, 0, None)
            ref = np.take_along_axis(cv, idx[None, :, None] if two else idx[None, :], 0)[0]
            dv = cv - ref[None]
            e = (dv ** 2).sum(-1) if two else dv ** 2
            is_best = np.arange(n_)[:, None] == best[None, :]
            wf = np.where((cw > 0) & (best[None, :] >= 0) & ~is_best, cw * np.exp(-e / (sig * sig)), cw)
        c0x, c0y = cx[sel], cy[sel]
        vfb = _edge_ext_ref(imgs[0], c0x, c0y, float(FP[F_FB_BLUR]))
        vfb = np.where((d0s >= 0)[:, None] if two else d0s >= 0, vms, vfb)
        fbw = max(float(FP[F_FB_W]), 1e-6)
        num = vfb * fbw + ((wf[..., None] if two else wf) * cv).sum(0)
        den = fbw + wf.sum(0)
        f = num / (den[:, None] if two else den)
        a = np.where(d0s >= 0, _smooth01(fm, d0s), 0.0)
        v = f + (vms - f) * (a[:, None] if two else a)
        out[sel] = q(v)
    return out.astype(np.uint16)


def render_planes_ref_fill(plan: Plan, fill, k: int, planes: dict, kernel: str = 'lanczos3', iters: int = 3,
                           black_y: int = 64, black_c: int = 512, full_range: bool = False, chunk_rows: int = 128):
    """Float64 reference of sprender's FILL output planes for plan record k (engine/stillpoint/fill.py).
    planes: {record: (Y, UV)} for k and each of its fill sources (same dtypes as render_planes_ref)."""
    from .fill import composite_rows
    y, uv = planes[k]
    is8 = y.dtype == np.uint8
    nrm = (lambda a: a.astype(np.float64) / 255.0) if is8 else (lambda a: a.astype(np.float64) / 65535.0)
    in_scale = 255.0 * 256.0 / 65535.0 if is8 else 1.0
    black_tex = 0.0 if full_range else (16.0 / 255.0 if is8 else 4096.0 / 65535.0)
    n = int(fill.n_src[k])
    srcs = [int(fill.src[k, i]) for i in range(n)]
    rows = [composite_rows(plan, j, np.asarray(fill.G[k, i], np.float64)) for i, j in enumerate(srcs)]
    FP = fill.kernel_params(k, plan.out_w, plan.out_h, black=black_tex).astype(np.float64)
    W, H = plan.out_w, plan.out_h
    iy = [nrm(y)] + [nrm(planes[j][0]) for j in srcs]
    ic = [nrm(uv)] + [nrm(planes[j][1]) for j in srcs]
    out_y = np.empty((H, W), np.uint16)
    out_c = np.empty((H // 2, W // 2, 2), np.uint16)
    xs = np.arange(W, dtype=np.float64)
    for r0 in range(0, H, chunk_rows):                     # row chunks bound the float64 temporaries
        X, Y = np.meshgrid(xs, np.arange(r0, min(H, r0 + chunk_rows), dtype=np.float64))
        out_y[r0:r0 + X.shape[0]] = _fill_plane_ref(plan, fill, k, X, Y, iy, rows, lambda S: (S[..., 0], S[..., 1]),
                                                    kernel, iters, True, FP, in_scale, black_y)
    xc = 2.0 * np.arange(W // 2)
    for r0 in range(0, H // 2, chunk_rows // 2):
        Xc, Yc = np.meshgrid(xc, 2.0 * np.arange(r0, min(H // 2, r0 + chunk_rows // 2)) + 0.5)
        out_c[r0:r0 + Xc.shape[0]] = _fill_plane_ref(plan, fill, k, Xc, Yc, ic, rows,
                                                     lambda S: (S[..., 0] * 0.5, (S[..., 1] - 0.5) * 0.5), kernel,
                                                     iters, False, FP, in_scale, black_c)
    return out_y, out_c


def preview_fill(plan: Plan, fill, k: int, bufs: list, out_scale: float, kernel: str = 'catmullrom', iters: int = 3,
                 black: float = 16.0):
    """Gray preview of record k WITH border fill via the shared Metal kernel sp_preview_gray_fill.
    bufs: [main, source 0, ...] float (bh,bw) luma buffers in 8-bit levels at one reduced size (same scale).
    Returns (img float32 (Ho,Wo), d0 (Ho,Wo) main inside distance [full-res source px], wn (Ho,Wo) neighbour
    weight sum after the consistency check)."""
    import torch
    from .fill import composite_rows
    lib = _metal_lib()
    bh, bw = bufs[0].shape
    Wo, Ho, osx, osy = output_grid(plan.out_w, plan.out_h, out_scale)
    n = int(fill.n_src[k])
    if len(bufs) < n + 1:
        raise ValueError(f'record {k} has {n} fill sources, got {len(bufs) - 1} buffers')
    P = kernel_params(plan, k, dst_w=Wo, dst_h=Ho, out_sx=osx, out_sy=osy, src_sx=bw / plan.src_w,
                      src_sy=bh / plan.src_h, buf_w=bw, buf_h=bh, kernel=kernel, iters=iters)
    FP = fill.kernel_params(k, plan.out_w, plan.out_h, black=black, value_scale=255.0)
    FM = np.concatenate([composite_rows(plan, int(fill.src[k, i]), np.asarray(fill.G[k, i], np.float64)).reshape(-1)
                         for i in range(n)] or [np.zeros(9)]).astype(np.float32)
    MS = (fill.mesh[k].reshape(-1) if fill.mesh is not None else np.zeros(2)).astype(np.float32)
    dev = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to('mps')
    src = dev(np.stack([np.asarray(b, np.float32) for b in bufs[:n + 1]], 0).reshape(-1))
    dst = torch.empty((Ho, Wo), dtype=torch.float32, device='mps')
    d0 = torch.empty((Ho, Wo), dtype=torch.float32, device='mps')
    wn = torch.empty((Ho, Wo), dtype=torch.float32, device='mps')
    Mk = np.ascontiguousarray(plan.row_mats[k], dtype=np.float32).reshape(-1)
    lib.sp_preview_gray_fill(src, dst, d0, wn, dev(P), dev(Mk), dev(FP), dev(FM), dev(MS), threads=(Wo, Ho))
    return dst.cpu().numpy(), d0.cpu().numpy(), wn.cpu().numpy()


def render_planes_ref_blur(plan: Plan, k: int, y: np.ndarray, uv: np.ndarray, D: np.ndarray, w: np.ndarray,
                           kernel: str = 'lanczos3', iters: int = 3, black_y: int = 64, black_c: int = 512,
                           rows_y: Optional[np.ndarray] = None, rows_c: Optional[np.ndarray] = None):
    """Float64 reference of sprender's synthetic-shutter kernels (sp_warp_luma_blur / sp_warp_chroma_blur): taps
    D (n,3,3) = R(V_k)^T R(V(t_k+s_i)), weights w (n,).  Tap i uses the row matrices M_row @ D_i; taps outside the
    source are dropped per pixel and the weights renormalised; averaged in the coded domain, then quantised.
    rows_y / rows_c: compute only these output luma / chroma rows (the returned planes have just those rows)."""
    to_code = (lambda a: a.astype(np.float64) * 4.0) if y.dtype == np.uint8 else (lambda a: a.astype(np.float64) / 64.0)
    ycode, ccode = to_code(y), to_code(uv)
    W, H = plan.out_w, plan.out_h
    M = np.asarray(plan.row_mats[k], np.float64)
    ry = np.arange(H) if rows_y is None else np.asarray(rows_y)
    rc = np.arange(H // 2) if rows_c is None else np.asarray(rows_c)
    X, Y = np.meshgrid(np.arange(W, dtype=np.float64), ry.astype(np.float64))
    Xc, Yc = np.meshgrid(2.0 * np.arange(W // 2), 2.0 * rc + 0.5)
    ay, wy_ = np.zeros(X.shape), np.zeros(X.shape)
    ac, wc_ = np.zeros(Xc.shape + (2,)), np.zeros(Xc.shape)
    for Di, wi in zip(np.asarray(D, np.float64), np.asarray(w, np.float64)):
        rows = np.einsum('rij,jk->rik', M, Di)
        S, ok, _ = _source_coord(plan, k, X, Y, iters, rows)
        ay += np.where(ok, wi * sample_ref(ycode, S[..., 0], S[..., 1], kernel), 0.0)
        wy_ += np.where(ok, wi, 0.0)
        Sc, okc, _ = _source_coord(plan, k, Xc, Yc, iters, rows)
        ac += np.where(okc[..., None], wi * sample_ref(ccode, Sc[..., 0] * 0.5, (Sc[..., 1] - 0.5) * 0.5, kernel), 0.0)
        wc_ += np.where(okc, wi, 0.0)
    out_y = _q10(ay / np.maximum(wy_, 1e-30))
    out_y[wy_ <= 0] = black_y * 64
    out_c = _q10(ac / np.maximum(wc_, 1e-30)[..., None])
    out_c[wc_ <= 0] = black_c * 64
    return out_y, out_c


# ============================================================================================ Metal (torch MPS)
_LIB = None


def _metal_lib():
    global _LIB
    if _LIB is None:
        import torch
        with open(SHADER_PATH) as f:
            _LIB = torch.mps.compile_shader('#define SP_NO_TEXTURE_KERNELS 1\n' + f.read())
    return _LIB


def kernel_params(plan: Plan, k: int, *, dst_w: int, dst_h: int, out_sx: float = 1.0, out_sy: float = 1.0,
                  src_sx: float = 1.0, src_sy: float = 1.0, buf_w: int = 0, buf_h: int = 0,
                  kernel: str = 'catmullrom', iters: int = 3, in_scale: float = 1.0,
                  black_y: float = 64 * 64 / 65535, black_c: float = 512 * 64 / 65535,
                  use_mesh: bool = True) -> np.ndarray:
    L = plan.lens
    P = np.zeros(P_COUNT, dtype=np.float32)
    P[P_OUT_FX] = plan.out_fx[k]
    P[P_OUT_CX] = (plan.out_w - 1) / 2.0
    P[P_OUT_CY] = (plan.out_h - 1) / 2.0
    P[P_LENS_MODEL] = 1.0 if L.model == 'kb4' else 0.0
    P[P_FX:P_FX + 4] = [L.fx, L.fy, L.cx, L.cy]
    P[P_K1:P_K1 + 4] = np.asarray(L.k, dtype=np.float64)[:4]
    P[P_SRC_W], P[P_SRC_H], P[P_N_ROWS], P[P_ITERS] = plan.src_w, plan.src_h, plan.n_rows, iters
    P[P_OUT_SX], P[P_OUT_SY], P[P_SRC_SX], P[P_SRC_SY] = out_sx, out_sy, src_sx, src_sy
    P[P_KERNEL] = KERNELS[kernel]
    P[P_IN_SCALE], P[P_BLACK_Y], P[P_BLACK_C] = in_scale, black_y, black_c
    P[P_DST_W], P[P_DST_H], P[P_BUF_W], P[P_BUF_H] = dst_w, dst_h, buf_w, buf_h
    mesh = plan_mesh(plan, use_mesh)
    if mesh is not None:
        P[P_MESH_NY], P[P_MESH_NX] = mesh.shape[1], mesh.shape[2]
    return P


def mesh_buffer(plan: Plan, k: int, device: str = 'mps'):
    """Frame k's mesh offsets as a flat float32 torch tensor (a 2-float dummy without a mesh)."""
    import torch
    mesh = plan_mesh(plan)
    a = np.zeros(2, np.float32) if mesh is None else np.ascontiguousarray(mesh[k], dtype=np.float32).reshape(-1)
    return torch.from_numpy(a).to(device)


def metal_coord_map(plan: Plan, k: int, out_scale: float = 1.0, iters: int = 3):
    """The Metal kernel's own source map (float32, full-res source coords) + valid mask, via torch MPS."""
    import torch
    Wo, Ho, sx, sy = output_grid(plan.out_w, plan.out_h, out_scale)
    P = torch.from_numpy(kernel_params(plan, k, dst_w=Wo, dst_h=Ho, out_sx=sx, out_sy=sy, iters=iters)).to('mps')
    M = torch.from_numpy(np.ascontiguousarray(plan.row_mats[k], dtype=np.float32).reshape(-1)).to('mps')
    out = torch.empty((Ho, Wo, 3), dtype=torch.float32, device='mps')
    if plan_mesh(plan) is not None:
        _metal_lib().sp_coord_map_mesh(out, P, M, mesh_buffer(plan, k), threads=(Wo, Ho))
    else:
        _metal_lib().sp_coord_map(out, P, M, threads=(Wo, Ho))
    o = out.cpu().numpy()
    return o[..., :2].astype(np.float64), o[..., 2] > 0.5


# ============================================================================================ decoding
def _luma_frames(video_path: str, want_pts: np.ndarray, tol: float) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index into want_pts, luma plane as float32 in 8-bit units) for each wanted PTS (sorted)."""
    import av
    try:
        from av.codec.hwaccel import HWAccel
        c = av.open(video_path, hwaccel=HWAccel(device_type='videotoolbox', allow_software_fallback=True))
    except Exception:  # pragma: no cover
        c = av.open(video_path)
    try:
        vs = c.streams.video[0]
        vs.thread_type = 'AUTO'
        tb = float(vs.time_base)
        i = 0
        n = len(want_pts)
        c.seek(max(0, int((want_pts[0] - 1.0) / tb)), stream=vs, backward=True, any_frame=False)
        while i < n:
            reseek = False
            for fr in c.decode(vs):
                if fr.pts is None:
                    continue
                t = fr.pts * tb
                while i < n and want_pts[i] < t - tol:   # wanted frame missing from the stream
                    i += 1
                if i >= n:
                    break
                if abs(t - want_pts[i]) <= tol:
                    p0 = fr.planes[0]
                    if fr.format.name in ('p010le', 'yuv420p10le', 'p010'):
                        a = np.frombuffer(p0, np.uint16).reshape(fr.height, p0.line_size // 2)[:, :fr.width]
                        if fr.format.name == 'yuv420p10le':
                            lum = a.astype(np.float32) * (1.0 / 4.0)
                        else:
                            lum = a.astype(np.float32) * (1.0 / 256.0)
                    else:
                        a = np.frombuffer(p0, np.uint8).reshape(fr.height, p0.line_size)[:, :fr.width]
                        lum = a.astype(np.float32)
                    yield i, lum
                    i += 1
                    if i < n and want_pts[i] > t + 3.0:     # big gap -> seek
                        c.seek(max(0, int((want_pts[i] - 1.0) / tb)), stream=vs, backward=True, any_frame=False)
                        reseek = True
                        break
            if not reseek:
                break
    finally:
        c.close()


def render_frames(plan: Plan, video_path: str, frames, out_scale: float = 0.25, gray: bool = True,
                  src_scale: Optional[float] = None, kernel: str = 'catmullrom', iters: int = 3,
                  as_float: bool = False, return_valid: bool = False) -> Iterator:
    """Fast preview renderer (torch MPS + the shared Metal kernel sp_preview_gray).

    frames: plan record indices (any order; rendered in PTS order). Source frame of record k = the decoded
    frame whose PTS matches plan.frame_pts[k] within half a frame. The source luma is box-filtered on the GPU
    to src_scale (default = out_scale; exact integer box when 1/src_scale is an integer divisor, else antialiased
    bilinear with the same pixel-centre convention) and resampled with `kernel` (Catmull-Rom = exact position).
    Yields (k, uint8 (Ho,Wo)) — or float32 in 8-bit units with as_float — plus the valid mask if return_valid.
    Invalid (outside source) pixels are 0.
    """
    if not gray:
        raise NotImplementedError("preview renderer is gray-only")
    import torch
    import torch.nn.functional as Fnn
    lib = _metal_lib()
    frames = np.asarray(frames, dtype=np.int64).reshape(-1)
    order = np.argsort(plan.frame_pts[frames], kind='stable')
    ks = frames[order]
    want = plan.frame_pts[ks]
    dts = np.diff(plan.frame_pts)
    tol = 0.5 * (float(np.median(dts)) if len(dts) else 1.0 / 60.0)
    Wo, Ho, osx, osy = output_grid(plan.out_w, plan.out_h, out_scale)
    ss = out_scale if src_scale is None else float(src_scale)
    fac = int(round(1.0 / ss))
    exact = abs(1.0 / ss - fac) < 1e-9 and plan.src_w % fac == 0 and plan.src_h % fac == 0
    bw, bh = (plan.src_w // fac, plan.src_h // fac) if exact else (int(round(plan.src_w * ss)), int(round(plan.src_h * ss)))
    ssx, ssy = bw / plan.src_w, bh / plan.src_h

    q: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()

    def reader():
        try:
            for i, lum in _luma_frames(video_path, want, tol):
                if stop.is_set():
                    break
                q.put((i, lum))
        except BaseException as e:  # propagate to consumer
            q.put(e)
        finally:
            q.put(None)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    dst = torch.empty((Ho, Wo), dtype=torch.float32, device='mps')
    val = torch.empty((Ho, Wo), dtype=torch.float32, device='mps')
    try:
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            i, lum = item
            k = int(ks[i])
            if lum.shape != (plan.src_h, plan.src_w):
                raise ValueError(f"decoded frame {lum.shape} != plan source {(plan.src_h, plan.src_w)}")
            src = torch.from_numpy(lum).to('mps')[None, None]
            if exact and fac > 1:
                src = Fnn.avg_pool2d(src, fac, fac)
            elif not exact:
                src = Fnn.interpolate(src, size=(bh, bw), mode='bilinear', align_corners=False, antialias=True)
            src = src[0, 0].contiguous()
            P = kernel_params(plan, k, dst_w=Wo, dst_h=Ho, out_sx=osx, out_sy=osy, src_sx=ssx, src_sy=ssy,
                              buf_w=bw, buf_h=bh, kernel=kernel, iters=iters)
            Pt = torch.from_numpy(P).to('mps')
            Mt = torch.from_numpy(np.ascontiguousarray(plan.row_mats[k], dtype=np.float32).reshape(-1)).to('mps')
            if plan_mesh(plan) is not None:
                lib.sp_preview_gray_mesh(src, dst, val, Pt, Mt, mesh_buffer(plan, k), threads=(Wo, Ho))
            else:
                lib.sp_preview_gray(src, dst, val, Pt, Mt, threads=(Wo, Ho))
            img = dst.cpu().numpy()
            if not as_float:
                img = np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8)
            if return_valid:
                yield k, img, val.cpu().numpy() > 0.5
            else:
                yield k, img
    finally:
        stop.set()
        # drain so the reader can finish
        while th.is_alive():
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass
