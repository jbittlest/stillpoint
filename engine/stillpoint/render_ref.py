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


def _rows_apply(M: np.ndarray, src_h: int, y: np.ndarray, rv: np.ndarray) -> np.ndarray:
    R = M.shape[0]
    g = np.clip(y * (R - 1) / (src_h - 1), 0.0, R - 1)
    j0 = np.minimum(np.floor(g).astype(np.int64), R - 2)
    f = (g - j0)[..., None]
    ra = np.einsum('...ij,...j->...i', M[j0], rv)
    rb = np.einsum('...ij,...j->...i', M[j0 + 1], rv)
    return ra + (rb - ra) * f


def _source_coord(plan: Plan, k: int, X: np.ndarray, Y: np.ndarray, iters: int = 3,
                  rows: Optional[np.ndarray] = None, secant: bool = True):
    """Full-res output pixel coords (any shape) -> (S (…,2) float64, valid bool, z).
    secant=False gives the plain fixed-point iteration (diagnostics only; the kernel always uses the secant step)."""
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
               chunk_rows: int = 256):
    """(Ho,Wo,2) FULL-RES source luma coordinates (float64) of the scaled output grid of plan frame k."""
    Wo, Ho, sx, sy = output_grid(plan.out_w, plan.out_h, out_scale)
    X1 = (np.arange(Wo) + 0.5) / sx - 0.5
    out = np.empty((Ho, Wo, 2))
    val = np.empty((Ho, Wo), dtype=bool)
    for r0 in range(0, Ho, chunk_rows):
        r1 = min(Ho, r0 + chunk_rows)
        Y1 = (np.arange(r0, r1) + 0.5) / sy - 0.5
        X, Y = np.meshgrid(X1, Y1)
        S, v, _ = _source_coord(plan, k, X, Y, iters)
        out[r0:r1] = S
        val[r0:r1] = v
    return (out, val) if return_valid else out


def source_points(plan: Plan, k: int, xy: np.ndarray, iters: int = 3):
    """Arbitrary FULL-RES output points (N,2) of plan frame k -> (source coords (N,2) float64, valid (N,) bool).
    Same math as the kernel — use it e.g. for crop checks on boundary samples."""
    xy = np.asarray(xy, dtype=np.float64)
    S, ok, _ = _source_coord(plan, k, xy[..., 0], xy[..., 1], iters)
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
                  black_y: float = 64 * 64 / 65535, black_c: float = 512 * 64 / 65535) -> np.ndarray:
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
    return P


def metal_coord_map(plan: Plan, k: int, out_scale: float = 1.0, iters: int = 3):
    """The Metal kernel's own source map (float32, full-res source coords) + valid mask, via torch MPS."""
    import torch
    Wo, Ho, sx, sy = output_grid(plan.out_w, plan.out_h, out_scale)
    P = torch.from_numpy(kernel_params(plan, k, dst_w=Wo, dst_h=Ho, out_sx=sx, out_sy=sy, iters=iters)).to('mps')
    M = torch.from_numpy(np.ascontiguousarray(plan.row_mats[k], dtype=np.float32).reshape(-1)).to('mps')
    out = torch.empty((Ho, Wo, 3), dtype=torch.float32, device='mps')
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
