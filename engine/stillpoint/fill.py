"""Full-frame border fill: synthesize the output pixels that fall outside the current source frame from
neighbouring source frames (workstream C, 2026-09-29).

Why
---
Gyroflow can only zoom (crop) or fill the border with a solid colour / repeated / mirrored pixels / a feathered
margin. Every gyro stabiliser has the same trade: a wider output field of view leaves less room for the virtual
camera to deviate from the shaking real camera. But the pixels the current frame is missing were usually SEEN by a
neighbouring frame a few frames earlier or later (the camera was pointing there), and we know every frame's
orientation to ~0.05 px. So the path optimizer may let the output border leave the source by a bounded overscan,
and the renderer pulls those pixels from the neighbours.

Geometry (exact, rotation-only, same kernel code as the main warp)
------------------------------------------------------------------
Output ray of frame k: r_v. World ray: Rv_k r_v. Frame j's source ray for that world direction:
    r_c = M_j(y) . G_kj . r_v,   G_kj = Rv_j^T Rv_k          (M_j = plan.row_mats[j], RS rows of frame j)
so a neighbour is just another plan record whose row matrices are right-multiplied by G_kj -- the renderer runs
the unchanged sp_source_coord() on the composite rows (M_j G_kj). All closed-loop corrections are already inside
M_j, so G only carries the relative VIRTUAL rotation (exact). What rotation cannot model is parallax (translation)
and moving objects:
  * parallax mesh (align pass): per output frame a coarse grid (mesh_w x mesh_h) of the image-plane parallax
    VELOCITY v(x) [output px per frame], measured with DIS flow between the neighbours k-s and k+s reprojected
    into frame k's virtual camera, robust per cell, temporally smoothed. A source at record offset d is sampled
    at x + d * v(x) (translation-induced flow grows ~linearly with the frame offset for d << depth/speed).
  * per-source check (align pass): in the band of frame k's own pixels next to the fill region, the neighbour's
    reprojection must agree with frame k (NCC); disagreeing sources (moving objects / parallax the mesh missed)
    are down-weighted or dropped; a luma gain matches auto-exposure changes.
  * per-pixel temporal-consistency check (kernel): candidates whose value differs from the best (nearest,
    most interior) candidate by more than ~sigma are rejected (Gaussian re-weighting) -> no ghost blends.
Seams: the current frame is blended toward the fill over `feather_main` source px inside its own border; each
neighbour's weight ramps from 0 at its own border over `feather_nb` px. Where no neighbour saw the pixel: soft
edge extension (the source edge sampled with a blur that grows with the overshoot).

Selection (geometry pass, no decoding): per output frame, a border-dense grid of output points; the points
within feather_main of (or outside) the source need fill; greedy set cover over candidate records k +- offsets
(same segment), score = new coverage x temporal preference x hysteresis (sources of frame k-1 are preferred so
consecutive frames draw from the same pixels).

The renderer reads the table from the .spplan fill section (plan_io.write_plan(..., fill=table)); renderers
that do not know it ignore it. Nothing here changes the plan's own geometry.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

from .geom import quat_to_mat
from .types import Plan

__all__ = ['FillParams', 'FillTable', 'select_sources', 'align_sources', 'compute_fill', 'fill_stats',
           'inside_distance', 'composite_rows', 'relative_rotation', 'fill_grid', 'fallback_fraction',
           'overscan_px']

# parameter block of the fill kernels (float[FP_COUNT]) -- must match shaders/warp.metal F_*
F_N_SRC, F_FEATHER_MAIN, F_FEATHER_NB, F_SIGMA, F_FB_W, F_MESH_W, F_MESH_H, F_BLACK = 0, 1, 2, 3, 4, 5, 6, 7
F_WEIGHT, F_GAIN, F_DOFF, F_FB_BLUR, F_OUT_W, F_OUT_H, F_PAR = 8, 12, 16, 20, 21, 22, 23
FP_COUNT = 32
MAX_SOURCES = 4


# ============================================================================================ parameters
@dataclass
class FillParams:
    max_sources: int = 4                  # K (<= MAX_SOURCES: the kernels bind 4 neighbour textures)
    offsets: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20)    # candidate |record offsets| (both signs)
    tau: float = 3.0                      # temporal preference weight 1 / (1 + |d| / tau)
    grid: tuple = (40, 24)                # selection grid over the output (border-dense cosine spacing)
    feather_main: float = 0.008           # x src_w: blend band inside the current frame's border [source px]
    feather_nb: float = 0.006             # x src_w: weight ramp at a neighbour's border [source px]
    sigma: float = 0.05                   # per-pixel consistency (16-bit normalised levels); 0 = off
    fallback_weight: float = 0.002        # weight of the soft edge extension (wins only where nothing else covers)
    fallback_blur: float = 0.35           # blur radius per source px of overshoot (soft edge extension)
    parallax_tol: float = 0.0             # x out_w: kernel source weight x exp(-(|offset| |v(X)| / tol)^2): where the
                                          # predicted parallax shift is large a far source is likely misaligned; 0 = off.
                                          # OFF by default: at 0.012 the OA4 0012 judge got WORSE on 2 of 3 windows
                                          # (+16-20 % HF, more jumps; the soft-edge smear it falls back to is tracked
                                          # as frame-locked motion), see workstream C notes 2026-09-29
    hysteresis: float = 1.3               # score bonus for sources frame k-1 used
    min_gain: float = 0.02                # stop the greedy cover when a source adds less than this (coverage frac)
    grid_band: float = 0.2                # selection grid: only the outer band of the output (fraction of the size)
    # --- align pass (needs decoded frames)
    align: bool = True
    align_width: int = 480
    mesh: tuple = (8, 5)                  # parallax velocity grid (x, y)
    mesh_step: int = 2                    # measure between the views of k - s and k + s
    mesh_smooth: int = 5                  # temporal smoothing half-width [frames] of the mesh
    mesh_stride: int = 3                  # measure the mesh on every n-th record (smoothing fills the rest)
    check_stride: int = 2                 # per-source band check on every n-th record (+ on new sources)
    mesh_max: float = 0.02                # |v| cap [x out_w px per frame]
    band: float = 0.05                    # check band width (x preview width) inside frame k's border
    ncc_lo: float = 0.3                   # source weight ramps 0 -> 1 between these band NCC values
    ncc_hi: float = 0.7
    min_band_px: int = 150                # fewer band pixels -> no check (keep the source)
    min_band_std: float = 3.0             # textureless band (8-bit levels) -> no NCC check (gain only)
    gain_clip: tuple = (0.8, 1.25)


@dataclass
class FillTable:
    """Per plan record: up to K fill sources. src = absolute plan record (-1 = none)."""
    K: int
    n_src: np.ndarray                     # (F,) int32
    src: np.ndarray                       # (F,K) int32
    weight: np.ndarray                    # (F,K) float32
    gain: np.ndarray                      # (F,K) float32
    G: np.ndarray                         # (F,K,3,3) float32  output ray of k -> virtual ray of the source
    frac_fill: np.ndarray                 # (F,) fraction of the output outside the current frame's source
    frac_uncovered: np.ndarray            # (F,) fraction of the output that no source covers (soft edge only)
    feather_main_px: float = 30.0
    feather_nb_px: float = 23.0
    sigma: float = 0.05
    fallback_weight: float = 0.01
    fallback_blur: float = 0.35
    mesh: Optional[np.ndarray] = None     # (F, mh, mw, 2) float32 parallax velocity [full-res output px / frame]
    par_px: float = 0.0                   # parallax tolerance [full-res output px] (0 = off, e.g. older plans)
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(len(self.n_src))

    @property
    def max_offset(self) -> int:
        F = self.n_frames
        if F == 0:
            return 0
        d = np.abs(self.src - np.arange(F)[:, None])
        d[self.src < 0] = 0
        return int(d.max(initial=0))

    @staticmethod
    def empty(F: int, K: int = MAX_SOURCES) -> 'FillTable':
        return FillTable(K=K, n_src=np.zeros(F, np.int32), src=np.full((F, K), -1, np.int32),
                         weight=np.zeros((F, K), np.float32), gain=np.ones((F, K), np.float32),
                         G=np.tile(np.eye(3, dtype=np.float32), (F, K, 1, 1)), frac_fill=np.zeros(F, np.float32),
                         frac_uncovered=np.zeros(F, np.float32))

    def kernel_params(self, k: int, out_w: int, out_h: int, black: float = 0.0, value_scale: float = 1.0) -> np.ndarray:
        """float[FP_COUNT] parameter block of the fill kernels for record k (same as sprender's).
        black: gain pivot in the sampled values' units; value_scale: units of the sampled values per normalised
        level (1 for textures, 255 for 8-bit-level preview buffers) -- scales sigma."""
        FP = np.zeros(FP_COUNT, np.float32)
        n = int(self.n_src[k])
        FP[F_N_SRC] = n
        FP[F_FEATHER_MAIN], FP[F_FEATHER_NB] = self.feather_main_px, self.feather_nb_px
        FP[F_SIGMA], FP[F_FB_W], FP[F_FB_BLUR] = self.sigma * value_scale, self.fallback_weight, self.fallback_blur
        FP[F_BLACK] = black
        FP[F_OUT_W], FP[F_OUT_H] = out_w, out_h
        FP[F_PAR] = self.par_px
        if self.mesh is not None:
            FP[F_MESH_H], FP[F_MESH_W] = self.mesh.shape[1], self.mesh.shape[2]
        for i in range(n):
            FP[F_WEIGHT + i] = self.weight[k, i]
            FP[F_GAIN + i] = self.gain[k, i]
            FP[F_DOFF + i] = float(self.src[k, i] - k)
        return FP


def overscan_px(src_w: int, src_h: int, frac: float) -> float:
    """Path-optimizer overscan [source px] for a fraction of the source height (fill on)."""
    return float(frac) * float(min(src_w, src_h))


OVERSCAN_LEVELS = (8, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256)


def _covered(Rall: np.ndarray, ks: np.ndarray, rays: np.ndarray, offs: np.ndarray, seg: np.ndarray, lens, W: int,
             H: int, inset: float) -> np.ndarray:
    """(n,P) bool: camera-frame rays (n,P,3) of frames ks seen by some frame ks+d (d in offs, same segment) at least
    `inset` px inside it (camera orientations at the frames' centre times; rolling shutter ignored -> inset)."""
    F = len(Rall)
    out = np.zeros(rays.shape[:2], bool)
    Rk = Rall[ks]
    for d in offs:
        j = ks + int(d)
        ok = (j >= 0) & (j < F)
        jj = np.clip(j, 0, F - 1)
        ok &= seg[jj] == seg[ks]
        if not ok.any():
            continue
        Rrel = np.einsum('nba,nbc->nac', Rall[jj], Rk)              # R_j^T R_k
        rc = np.einsum('nab,npb->npa', Rrel, rays)
        uv = lens.project(rc)
        ins = (rc[..., 2] > 0) & (uv[..., 0] >= inset) & (uv[..., 0] <= W - 1 - inset) & \
              (uv[..., 1] >= inset) & (uv[..., 1] <= H - 1 - inset)
        out |= ins & ok[:, None]
    return out


def coverage_overscan(q_cam_fn, frame_t: np.ndarray, lens, W: int, H: int, cap_px: float,
                      segments: Optional[np.ndarray] = None, prm: Optional[FillParams] = None, n_along: int = 17,
                      chunk: int = 1000, stride: int = 2) -> tuple[np.ndarray, dict]:
    """Coverage-aware overscan for the path optimizer: (F,4,n_along) source px by which the output border of frame k
    may leave source k beyond the [left, top, right, bottom] edge, as a profile of n_along points along each edge
    (left/right: y = 0..H-1, top/bottom: x = 0..W-1). At each profile point the strip from the edge out to that
    distance (levels OVERSCAN_LEVELS <= cap_px) must have been seen by a neighbouring frame k +- prm.offsets (same
    segment), at least the neighbour feather + 8 px inside it; the box corners beyond two edges must be seen too
    (else both profile ends are zeroed). Pure camera-frame geometry, independent of the virtual path: a ray r of
    camera k is seen by frame j iff R_j^T R_k r lands inside source j (centre-time orientations; the inset absorbs
    rolling shutter). Every `stride`-th frame is evaluated; the others take the min of their evaluated neighbours."""
    prm = prm or FillParams()
    ft = np.asarray(frame_t, np.float64)
    F = len(ft)
    E = np.zeros((F, 4, n_along))
    levels = np.array([l for l in OVERSCAN_LEVELS if l <= cap_px + 1e-9], np.float64)
    info = dict(cap_px=float(cap_px), levels=levels.tolist(), n_along=int(n_along))
    if F == 0 or len(levels) == 0:
        return E, info
    t0 = time.perf_counter()
    Rall = quat_to_mat(np.asarray(q_cam_fn(ft), np.float64))
    seg = np.zeros(F, np.int64) if segments is None else np.asarray(segments, np.int64)
    offs = np.array(sorted(set(int(o) for o in prm.offsets if int(o) != 0)), np.int64)
    offs = np.concatenate([-offs[::-1], offs])
    inset = float(prm.feather_nb * W) + 8.0
    xs = np.linspace(0.0, W - 1.0, n_along)
    ys = np.linspace(0.0, H - 1.0, n_along)
    L = len(levels)
    pts = []                                               # (4 edges, L levels, n_along, 2)
    for e in range(4):
        for l in levels:
            if e == 0:
                pts.append(np.c_[np.full(n_along, -l), ys])
            elif e == 1:
                pts.append(np.c_[xs, np.full(n_along, -l)])
            elif e == 2:
                pts.append(np.c_[np.full(n_along, W - 1 + l), ys])
            else:
                pts.append(np.c_[xs, np.full(n_along, H - 1 + l)])
    pts = np.asarray(pts).reshape(-1, 2)
    rays = lens.unproject(pts)                             # (P,3), the same pixel positions for every frame
    st = max(1, int(stride))
    ev = np.unique(np.concatenate([np.arange(0, F, st), [F - 1]]))
    cov = np.zeros((len(ev), 4, L, n_along), bool)
    for a in range(0, len(ev), chunk):
        ks = ev[a:a + chunk]
        c = _covered(Rall, ks, np.broadcast_to(rays, (len(ks),) + rays.shape), offs, seg, lens, W, H, inset)
        cov[a:a + len(ks)] = c.reshape(len(ks), 4, L, n_along)
    nlev = np.cumprod(cov, axis=2).sum(axis=2)             # (n_ev,4,n_along): levels covered from the edge out
    lv = np.concatenate([[0.0], levels])
    Ee = lv[nlev]
    # box corners: beyond two edges at once (profile ends); zero both ends when not seen
    corners = ((0, 0, 1, 0, -1, -1), (2, 0, 1, n_along - 1, 1, -1), (2, n_along - 1, 3, n_along - 1, 1, 1),
               (0, n_along - 1, 3, 0, -1, 1))          # (x edge, its end, y edge, its end, sx, sy)
    for ex, ix, ey, iy, sx, sy in corners:
        m = (Ee[:, ex, ix] > 0) & (Ee[:, ey, iy] > 0)
        if not m.any():
            continue
        sel = np.flatnonzero(m)
        x0 = 0.0 if sx < 0 else W - 1.0
        y0 = 0.0 if sy < 0 else H - 1.0
        cp = np.stack([np.c_[x0 + sx * fa * Ee[sel, ex, ix], y0 + sy * fb * Ee[sel, ey, iy]]
                       for fa, fb in ((1.0, 1.0), (1.0, 0.5), (0.5, 1.0))], 1)      # (n,3,2)
        r = lens.unproject(cp)
        ok = np.ones(len(sel), bool)
        for a in range(0, len(sel), chunk):
            ok[a:a + chunk] = _covered(Rall, ev[sel[a:a + chunk]], r[a:a + chunk], offs, seg, lens, W, H,
                                       inset).all(-1)
        bad = sel[~ok]
        Ee[bad, ex, ix] = 0.0
        Ee[bad, ey, iy] = 0.0
    E[ev] = Ee
    if st > 1:                                             # frames between evaluated ones: min of the two nearest
        pos = np.searchsorted(ev, np.arange(F))
        lo_i = np.clip(pos - 1, 0, len(ev) - 1)
        hi_i = np.clip(pos, 0, len(ev) - 1)
        mid = ~np.isin(np.arange(F), ev)
        E[mid] = np.minimum(Ee[lo_i[mid]], Ee[hi_i[mid]])
    info.update(seconds=round(time.perf_counter() - t0, 2), mean_px=[float(v) for v in E.mean(axis=(0, 2))],
                max_mean_px=[float(v) for v in E.max(axis=2).mean(0)],
                frac_frames_any=[float(v) for v in (E.max(axis=2) > 0).mean(0)])
    return E, info


# ============================================================================================ geometry
def relative_rotation(virt_q: np.ndarray, k: int, j: int) -> np.ndarray:
    """G = Rv_j^T Rv_k: output (virtual) ray of frame k -> virtual ray of frame j."""
    Rk = quat_to_mat(np.asarray(virt_q[k], np.float64))
    Rj = quat_to_mat(np.asarray(virt_q[j], np.float64))
    return Rj.T @ Rk


def composite_rows(plan: Plan, j: int, G: np.ndarray) -> np.ndarray:
    """Row matrices M_j(y) . G (n_rows,3,3): what the kernels evaluate for a fill source."""
    return np.einsum('rab,bc->rac', np.asarray(plan.row_mats[j], np.float64), np.asarray(G, np.float64))


def inside_distance(plan: Plan, S: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Signed distance [source px] of source coords inside the valid source area (<0 outside; -1e30 behind)."""
    d = np.minimum(np.minimum(S[..., 0] + 0.5, plan.src_w - 0.5 - S[..., 0]),
                   np.minimum(S[..., 1] + 0.5, plan.src_h - 0.5 - S[..., 1]))
    return np.where(z > 0, d, -1e30)


def _proj(plan: Plan, k: int, X, Y, rows=None):
    from .render_ref import _source_coord
    S, _, z = _source_coord(plan, k, np.asarray(X, np.float64), np.asarray(Y, np.float64), 3, rows=rows)
    return S, z


def fill_grid(out_w: int, out_h: int, gx: int, gy: int):
    """Border-dense grid of output points (cosine spacing) + per-point area weights (sum 1)."""
    def ax(n, L):
        t = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n))
        p = t * (L - 1)
        w = np.gradient(p)
        return p, w / w.sum()
    xs, wx = ax(gx, out_w)
    ys, wy = ax(gy, out_h)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel(), np.outer(wy, wx).ravel()


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _segments_from_pts(pts: np.ndarray) -> np.ndarray:
    """Segment id per record: a new segment where the PTS step exceeds 1.5x the median step."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return np.zeros(len(pts), np.int64)
    d = np.diff(pts)
    med = float(np.median(d))
    return np.concatenate([[0], np.cumsum(d > 1.5 * med)]).astype(np.int64)


def temporal_weight(d, tau: float) -> np.ndarray:
    return 1.0 / (1.0 + np.abs(np.asarray(d, np.float64)) / max(float(tau), 1e-6))


def _proj_batch(plan: Plan, k, rows: np.ndarray, X: np.ndarray, Y: np.ndarray, iters: int = 3):
    """render_ref._source_coord for C candidate row stacks at once: rows (C,R,3,3), points (N,) -> uv (C,N,2), z (C,N).
    Same fixed-point + secant iteration as the kernels. k: the record whose output focal applies (int), or (C,)
    records (one per row stack)."""
    fx = np.asarray(plan.out_fx, np.float64)[np.asarray(k)]
    fx = np.broadcast_to(np.atleast_1d(fx), (rows.shape[0],))[:, None]            # (C,1)
    cx, cy = (plan.out_w - 1) / 2.0, (plan.out_h - 1) / 2.0
    C, R = rows.shape[:2]
    N = len(X)
    rv = np.stack([(X[None, :] - cx) / fx, (Y[None, :] - cy) / fx, np.ones((C, N))], -1)   # (C,N,3)
    Hs = plan.src_h
    y = np.full((C, N), 0.5 * (Hs - 1))
    ya = np.zeros_like(y)
    ga = np.zeros_like(y)
    ci = np.arange(C)[:, None]
    uv = z = None
    for e in range(max(int(iters), 1)):
        g = np.clip(y * (R - 1) / (Hs - 1), 0.0, R - 1)
        j0 = np.minimum(np.floor(g).astype(np.int64), R - 2)
        f = (g - j0)[..., None]
        ra = np.einsum('cnij,cnj->cni', rows[ci, j0], rv)
        rb = np.einsum('cnij,cnj->cni', rows[ci, j0 + 1], rv)
        rc = ra + (rb - ra) * f
        uv = plan.lens.project(rc)
        z = rc[..., 2]
        gg = uv[..., 1] - y
        yn = uv[..., 1].copy()
        if e >= 1:
            dy = y - ya
            with np.errstate(divide='ignore', invalid='ignore'):
                s = (gg - ga) / dy
                use = (dy != 0) & (s <= -0.1) & (s >= -2.0)
                yn = np.where(use, y - gg / np.where(use, s, 1.0), yn)
        ya, ga, y = y, gg, yn
    return uv, z


def select_sources(plan: Plan, virt_q: np.ndarray, records=None, prm: Optional[FillParams] = None,
                   progress: Optional[Callable[[float], None]] = None) -> FillTable:
    """Geometry pass: per record whose output leaves its source frame, the fill sources (greedy cover of the output
    within feather_main of / outside the source) + G matrices. Records entirely inside their source get none (the
    renderer then samples only the frame itself; the fill kernel's soft edge extension covers slivers the grid missed)."""
    prm = prm or FillParams()
    F = plan.n_frames
    K = int(min(prm.max_sources, MAX_SOURCES))
    virt_q = np.asarray(virt_q, np.float64)
    if len(virt_q) != F:
        raise ValueError(f'virt_q has {len(virt_q)} rows, plan has {F} records')
    tab = FillTable.empty(F, MAX_SOURCES)
    tab.K = K
    tab.feather_main_px = float(prm.feather_main * plan.src_w)
    tab.feather_nb_px = float(prm.feather_nb * plan.src_w)
    tab.sigma, tab.fallback_weight, tab.fallback_blur = prm.sigma, prm.fallback_weight, prm.fallback_blur
    tab.par_px = float(prm.parallax_tol * plan.out_w)
    recs = np.arange(F) if records is None else np.unique(np.asarray(records, np.int64))
    X, Y, A = fill_grid(plan.out_w, plan.out_h, *prm.grid)
    edge = np.minimum(np.minimum(X / (plan.out_w - 1), 1 - X / (plan.out_w - 1)),
                      np.minimum(Y / (plan.out_h - 1), 1 - Y / (plan.out_h - 1)))
    keep = edge <= prm.grid_band                          # the output border band (the interior never leaves the source)
    X, Y, A = X[keep], Y[keep], A[keep]
    seg = _segments_from_pts(plan.frame_pts)
    Rv = quat_to_mat(virt_q)                                   # (F,3,3)
    offs = np.array(sorted(set(int(o) for o in prm.offsets if int(o) != 0)), np.int64)
    offs = np.concatenate([-offs[::-1], offs])
    fm, fnb = tab.feather_main_px, tab.feather_nb_px
    rm = np.asarray(plan.row_mats, np.float64)
    prev_src: set = set()
    prev_k = -10
    t0 = time.perf_counter()
    # batched geometry: per chunk of records, the main projection of every record and, for the records whose
    # output leaves the source, all candidate projections in one vectorised call
    B = 32
    geo: dict = {}

    def _batch(chunk):
        S0, z0 = _proj_batch(plan, chunk, rm[chunk], X, Y)
        d0s = inside_distance(plan, S0, z0)                                  # (b,N)
        stacks, owner, info = [], [], {}
        for b, k in enumerate(chunk.tolist()):
            if not (d0s[b] < 0).any():
                info[k] = (d0s[b], None, None, None)
                continue
            cands = k + offs
            cands = cands[(cands >= 0) & (cands < F)]
            cands = cands[seg[cands] == seg[k]]
            Gs = np.einsum('cba,bd->cad', Rv[cands], Rv[k])                 # Rv_j^T Rv_k
            if len(cands):
                stacks.append(np.einsum('crab,cbd->crad', rm[cands], Gs))
                owner.append(np.full(len(cands), k))
            info[k] = (d0s[b], cands, Gs, len(cands))
        if stacks:
            rows_all = np.concatenate(stacks)
            ks_all = np.concatenate(owner)
            uv, z = _proj_batch(plan, ks_all, rows_all, X, Y)
            cov_all = _smoothstep(0.0, fnb, inside_distance(plan, uv, z))   # (sumC,N)
            a = 0
            for k in chunk.tolist():
                d0, cands, Gs, nc = info[k]
                if cands is not None and nc:
                    info[k] = (d0, cands, Gs, cov_all[a:a + nc])
                    a += nc
        return info

    for n_, k in enumerate(recs):
        k = int(k)
        if k not in geo:
            geo = _batch(recs[n_:n_ + B])
        d0, cands, Gs, cov_full = geo[k]
        out = d0 < 0
        if not out.any():
            prev_src, prev_k = set(), k
            continue
        tab.frac_fill[k] = float(A[out].sum())
        need = d0 < fm
        An = A[need]
        # demand: 1 outside, ramping to 0 at feather_main inside (what the main frame does not provide)
        dem = 1.0 - _smoothstep(0.0, fm, np.maximum(d0[need], 0.0))
        if cands is None or len(cands) == 0:
            tab.frac_uncovered[k] = tab.frac_fill[k]
            prev_src, prev_k = set(), k
            continue
        cov = cov_full[:, need]                                              # (C,N)
        tw = temporal_weight(cands - k, prm.tau)
        hy = np.array([prm.hysteresis if (int(j) in prev_src and prev_k == k - 1) else 1.0 for j in cands])
        c = np.zeros(cov.shape[1])
        chosen = []
        tot = float((dem * An).sum())
        for _ in range(K):
            gain = (np.maximum(cov - c[None, :], 0.0) * dem[None, :] * An[None, :]).sum(axis=1)
            score = gain * tw * hy
            if chosen:
                score[chosen] = -1.0
            ci = int(np.argmax(score))
            if score[ci] <= 0 or gain[ci] < prm.min_gain * max(tot, 1e-12):
                break
            chosen.append(ci)
            c = np.maximum(c, cov[ci])
        # order: highest temporal weight first (the kernel's reference candidate is the max-weight one);
        # weights relative to the nearest source (only their ratios matter, vs the tiny fallback weight)
        chosen.sort(key=lambda ci: -tw[ci])
        n = len(chosen)
        tab.n_src[k] = n
        for i, ci in enumerate(chosen):
            tab.src[k, i] = int(cands[ci])
            tab.weight[k, i] = float(tw[ci] / tw[chosen[0]])
            tab.G[k, i] = Gs[ci].astype(np.float32)
        o = out[need]
        tab.frac_uncovered[k] = float(((1.0 - c[o]) * An[o]).sum())
        prev_src, prev_k = set(int(cands[ci]) for ci in chosen), k
        if progress is not None and (n_ % 200 == 0):
            progress(n_ / max(len(recs), 1))
    tab.meta.update(select_s=round(time.perf_counter() - t0, 2), n_records=int(len(recs)),
                    offsets=offs.tolist(), K=K, grid=list(prm.grid), grid_points=int(len(X)))
    return tab


# ============================================================================================ align pass
class _Previewer:
    """Gray previews of arbitrary row matrices at the analysis size (shared Metal kernel sp_preview_gray)."""

    def __init__(self, plan: Plan, bw: int, bh: int):
        import torch
        from .render_ref import _metal_lib, output_grid
        self.torch, self.lib, self.plan = torch, _metal_lib(), plan
        self.bw, self.bh = bw, bh
        self.out_scale = bw / plan.out_w
        self.Wo, self.Ho, self.osx, self.osy = output_grid(plan.out_w, plan.out_h, self.out_scale)
        self.ssx, self.ssy = bw / plan.src_w, bh / plan.src_h
        self.dst = torch.empty((self.Ho, self.Wo), dtype=torch.float32, device='mps')
        self.val = torch.empty((self.Ho, self.Wo), dtype=torch.float32, device='mps')

    def upload(self, buf: np.ndarray):
        return self.torch.from_numpy(np.ascontiguousarray(buf, dtype=np.float32)).to('mps')

    def render(self, src, k: int, rows: np.ndarray):
        from .render_ref import kernel_params
        torch = self.torch
        P = kernel_params(self.plan, k, dst_w=self.Wo, dst_h=self.Ho, out_sx=self.osx, out_sy=self.osy,
                          src_sx=self.ssx, src_sy=self.ssy, buf_w=self.bw, buf_h=self.bh, kernel='catmullrom')
        Pt = torch.from_numpy(P).to('mps')
        Mt = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32).reshape(-1)).to('mps')
        self.lib.sp_preview_gray(src, self.dst, self.val, Pt, Mt, threads=(self.Wo, self.Ho))
        return self.dst.cpu().numpy(), self.val.cpu().numpy() > 0.5


def _cell_median(flow: np.ndarray, ok: np.ndarray, mw: int, mh: int, min_n: int = 30):
    """Robust per-cell median of a flow field -> (mh,mw,2), (mh,mw) bool (cell has data)."""
    H, W = ok.shape
    out = np.zeros((mh, mw, 2), np.float64)
    has = np.zeros((mh, mw), bool)
    ys = np.linspace(0, H, mh + 1).astype(int)
    xs = np.linspace(0, W, mw + 1).astype(int)
    for a in range(mh):
        for b in range(mw):
            m = ok[ys[a]:ys[a + 1], xs[b]:xs[b + 1]]
            if m.sum() >= min_n:
                f = flow[ys[a]:ys[a + 1], xs[b]:xs[b + 1]][m]
                out[a, b] = np.median(f, axis=0)
                has[a, b] = True
    return out, has


def _fill_cells(v: np.ndarray, has: np.ndarray) -> np.ndarray:
    """Cells without data take the mean of their filled neighbours (repeated), or 0 when nothing has data."""
    if has.all():
        return v
    if not has.any():
        return np.zeros_like(v)
    v = v.copy()
    h = has.copy()
    mh, mw = h.shape
    for _ in range(mh + mw):
        if h.all():
            break
        nh = h.copy()
        for a in range(mh):
            for b in range(mw):
                if h[a, b]:
                    continue
                acc, n = np.zeros(2), 0
                for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    aa, bb = a + da, b + db
                    if 0 <= aa < mh and 0 <= bb < mw and h[aa, bb]:
                        acc += v[aa, bb]
                        n += 1
                if n:
                    v[a, b] = acc / n
                    nh[a, b] = True
        h = nh
    return v


def mesh_disp(mesh: np.ndarray, X: np.ndarray, Y: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Bilinear lookup of a (mh,mw,2) mesh at full-res output coords (clamped) -- warp.metal sp_mesh_disp twin."""
    mh, mw = mesh.shape[:2]
    fx = np.clip((X + 0.5) * mw / out_w - 0.5, 0.0, mw - 1)
    fy = np.clip((Y + 0.5) * mh / out_h - 0.5, 0.0, mh - 1)
    x0 = np.minimum(np.floor(fx).astype(np.int64), max(mw - 2, 0))
    y0 = np.minimum(np.floor(fy).astype(np.int64), max(mh - 2, 0))
    x1 = np.minimum(x0 + 1, mw - 1)
    y1 = np.minimum(y0 + 1, mh - 1)
    ax_ = (fx - x0)[..., None]
    ay_ = (fy - y0)[..., None]
    m = mesh.astype(np.float64)
    top = m[y0, x0] * (1 - ax_) + m[y0, x1] * ax_
    bot = m[y1, x0] * (1 - ax_) + m[y1, x1] * ax_
    return top * (1 - ay_) + bot * ay_


def align_sources(tab: FillTable, plan: Plan, video: str, records=None, prm: Optional[FillParams] = None,
                  stream=None, progress: Optional[Callable[[float], None]] = None,
                  cancel: Optional[Callable[[], bool]] = None) -> FillTable:
    """Align pass (decodes frames at prm.align_width): parallax mesh, per-source NCC check + luma gain.
    Only records with fill sources are processed. Returns the updated table (a copy)."""
    import cv2

    prm = prm or FillParams()
    tab = replace(tab, n_src=tab.n_src.copy(), src=tab.src.copy(), weight=tab.weight.copy(),
                  gain=tab.gain.copy(), G=tab.G.copy(), meta=dict(tab.meta))
    F = plan.n_frames
    recs = np.arange(F) if records is None else np.unique(np.asarray(records, np.int64))
    work = recs[tab.n_src[recs] > 0]
    mw, mh = int(prm.mesh[0]), int(prm.mesh[1])
    mesh = np.zeros((F, mh, mw, 2), np.float32) if tab.mesh is None else tab.mesh.copy()
    tab.mesh = mesh
    if len(work) == 0:
        return tab
    s = max(1, int(prm.mesh_step))
    need = set()
    for k in work.tolist():
        need.add(k)
        for j in (k - s, k + s):
            if 0 <= j < F:
                need.add(j)
        need.update(int(j) for j in tab.src[k, :tab.n_src[k]])
    need = np.array(sorted(need), np.int64)
    src_idx = np.asarray(plan.meta.get('frames', np.arange(F)), np.int64)
    own = False
    if stream is None:
        from .framestream import FrameStream
        stream = FrameStream(video, int(prm.align_width), lanes=1, block=150)
        own = True
    t0 = time.perf_counter()
    # a wider stream (e.g. the analysis' own 960-px decoder) is area-downscaled to align_width
    bw, bh = int(stream.w), int(stream.h)
    if bw > 1.05 * int(prm.align_width):
        bh = max(2, int(round(bh * int(prm.align_width) / bw / 2.0)) * 2)
        bw = int(prm.align_width)
    pv = _Previewer(plan, bw, bh)
    scale = pv.Wo / plan.out_w                                  # preview px per full-res output px
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    buf: dict = {}                                              # record -> mps tensor (analysis-size luma)
    raw_v = np.zeros((F, mh, mw, 2), np.float64)
    raw_has = np.zeros((F, mh, mw), bool)
    band_px = max(4, int(round(prm.band * pv.Wo)))
    vmax = prm.mesh_max * plan.out_w
    rk = {int(src_idx[r]): int(r) for r in need}
    wi = 0
    nwork = len(work)
    stats = dict(n_checked=0, n_dropped=0, n_textureless=0, ncc=[], gain=[])

    def view(k, j, G):
        return pv.render(buf[j], k, composite_rows(plan, j, G) if j != k else plan.row_mats[k])

    H_, W_ = pv.Ho, pv.Wo
    gx, gy = np.meshgrid(np.arange(W_, dtype=np.float32), np.arange(H_, dtype=np.float32))
    last_mesh = {'k': -10 ** 9, 'v': None}
    checked: dict = {}                                          # (record, source record) -> (gain, weight factor)
    last_checked = {'k': -10 ** 9, 'srcs': set()}
    cs = max(1, int(prm.check_stride))

    def process(k):
        # ---- parallax velocity mesh between the views of k-s and k+s (both reprojected into k's camera)
        a, b = k - s, k + s
        if k % max(1, int(prm.mesh_stride)) == 0 and a >= 0 and b < F and a in buf and b in buf:
            Ia, va = view(k, a, _G(k, a))
            Ib, vb = view(k, b, _G(k, b))
            ok = va & vb
            if ok.mean() > 0.2:
                # outside a view's valid area show the OTHER view (else the black border is an edge that DIS
                # tracks: a spurious flow near the borders); where neither is valid, a flat mid level
                flat = float(Ia[ok].mean())
                A = np.where(va, Ia, np.where(vb, Ib, flat))
                B = np.where(vb, Ib, np.where(va, Ia, flat))
                A8 = np.clip(A, 0, 255).astype(np.uint8)
                B8 = np.clip(B, 0, 255).astype(np.uint8)
                fl = dis.calc(A8, B8, None)
                okf = cv2.erode(ok.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                cell_px = okf.size / float(mw * mh)
                v, has = _cell_median(fl / (b - a) / scale, okf, mw, mh, min_n=max(30, int(0.2 * cell_px)))
                raw_v[k], raw_has[k] = np.clip(v, -vmax, vmax), has
                if has.any():
                    last_mesh.update(k=k, v=_fill_cells(raw_v[k], has))
        # ---- per-source check (band of k's own pixels next to the fill region) + gain; every check_stride-th
        # record, and whenever a source appears that the last checked record did not have (the others reuse the
        # result of the same source at the nearest checked record)
        n = int(tab.n_src[k])
        if n == 0 or k not in buf:
            return
        srcs = [int(j) for j in tab.src[k, :n]]
        if k % cs != 0 and k - last_checked['k'] < cs and set(srcs) <= last_checked['srcs']:
            return
        last_checked.update(k=k, srcs=set(srcs))
        Ik, vk = view(k, k, None)
        if vk.all():
            return
        dist = cv2.distanceTransform(vk.astype(np.uint8), cv2.DIST_L2, 3)
        band = vk & (dist <= band_px) & (dist > 1.5)
        dfield = None                                       # parallax velocity field [preview px / frame]
        if last_mesh['v'] is not None and abs(k - last_mesh['k']) <= max(1, int(prm.mesh_stride)):
            dfield = cv2.resize(last_mesh['v'].astype(np.float32), (W_, H_), interpolation=cv2.INTER_LINEAR) * scale
        for i in range(n):
            j = srcs[i]
            if j not in buf:
                continue
            Ij, vj = view(k, j, tab.G[k, i].astype(np.float64))
            if dfield is not None:
                disp = dfield * float(j - k)
                mx = (gx + disp[..., 0]).astype(np.float32)
                my = (gy + disp[..., 1]).astype(np.float32)
                Ij = cv2.remap(Ij, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                vj = cv2.remap(vj.astype(np.float32), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0) > 0.5
            m = band & vj
            if m.sum() < prm.min_band_px:
                continue
            a_, b_ = Ik[m].astype(np.float64), Ij[m].astype(np.float64)
            # luma gain about the video black level (the kernels apply (v - black) * gain + black)
            g = float(np.clip((a_.mean() - 16.0) / max(b_.mean() - 16.0, 1.0), *prm.gain_clip))
            tab.gain[k, i] = g
            stats['gain'].append(g)
            if a_.std() < prm.min_band_std:
                stats['n_textureless'] += 1
                checked[(k, j)] = (g, 1.0)
                continue
            bb = b_ * g
            ncc = float(((a_ - a_.mean()) * (bb - bb.mean())).mean() / max(a_.std() * bb.std(), 1e-9))
            stats['ncc'].append(ncc)
            stats['n_checked'] += 1
            f = float(_smoothstep(prm.ncc_lo, prm.ncc_hi, np.array(ncc)))
            if f <= 0.0:
                stats['n_dropped'] += 1
            tab.weight[k, i] = float(tab.weight[k, i]) * f
            checked[(k, j)] = (g, f)

    vq = np.asarray(plan.virt_q, np.float64)
    if not np.any(vq):
        raise ValueError('align_sources needs plan.virt_q')
    Rv = quat_to_mat(vq)

    def _G(k, j):
        return Rv[j].T @ Rv[k]

    maxoff = max(tab.max_offset, s)
    try:
        src_need = src_idx[need]
        for sk, lum in stream.frames(src_need, cancel=cancel):
            r = rk[int(sk)]
            if lum.shape[1] != bw:
                lum = cv2.resize(lum, (bw, bh), interpolation=cv2.INTER_AREA)
            buf[r] = pv.upload(lum)
            # process every work record whose inputs have all arrived (inputs lie at most maxoff ahead)
            while wi < nwork and (work[wi] + maxoff <= r or r == need[-1]):
                process(int(work[wi]))
                wi += 1
                if progress is not None and wi % 100 == 0:
                    progress(wi / nwork)
            lo = int(work[wi]) - maxoff if wi < nwork else r
            for key in [q for q in buf if q < lo]:
                del buf[key]
        while wi < nwork:
            process(int(work[wi]))
            wi += 1
    finally:
        if own:
            stream.close()
    # records skipped by check_stride: the result of the same source at the nearest checked record
    n_reused = 0
    if cs > 1:
        for k in work.tolist():
            for i in range(int(tab.n_src[k])):
                j = int(tab.src[k, i])
                if (k, j) in checked:
                    continue
                for dk in sorted(range(-2 * cs, 2 * cs + 1), key=abs):
                    r_ = checked.get((k + dk, j))
                    if r_ is not None:
                        tab.gain[k, i] = r_[0]
                        tab.weight[k, i] = float(tab.weight[k, i]) * r_[1]
                        n_reused += 1
                        break
    # temporal smoothing of the mesh (weighted by data availability), over the work records only
    h = max(0, int(prm.mesh_smooth))
    wmask = np.zeros(F, bool)
    wmask[work] = True
    for k in work.tolist():
        a, b = max(0, k - h), min(F, k + h + 1)
        hs = raw_has[a:b] & wmask[a:b, None, None]
        cnt = hs.sum(0)
        acc = (raw_v[a:b] * hs[..., None]).sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            v = np.where(cnt[..., None] > 0, acc / np.maximum(cnt[..., None], 1), 0.0)
        mesh[k] = _fill_cells(v, cnt > 0).astype(np.float32)
    # re-drop zero-weight sources, keep order by weight
    for k in work.tolist():
        n = int(tab.n_src[k])
        keep = [i for i in range(n) if tab.weight[k, i] > 1e-4]
        if len(keep) != n:
            order = keep
            tab.src[k, :len(order)] = tab.src[k, order]
            tab.weight[k, :len(order)] = tab.weight[k, order]
            tab.gain[k, :len(order)] = tab.gain[k, order]
            tab.G[k, :len(order)] = tab.G[k, order]
            tab.src[k, len(order):] = -1
            tab.weight[k, len(order):] = 0
            tab.gain[k, len(order):] = 1
            tab.G[k, len(order):] = np.eye(3, dtype=np.float32)
            tab.n_src[k] = len(order)
    nc = np.array(stats['ncc']) if stats['ncc'] else np.zeros(0)
    tab.meta.update(align_s=round(time.perf_counter() - t0, 2), align_width=int(bw),
                    align=dict(n_work=int(nwork), n_checked=stats['n_checked'], n_reused=n_reused,
                               n_dropped=stats['n_dropped'],
                               n_textureless=stats['n_textureless'],
                               ncc_median=float(np.median(nc)) if len(nc) else None,
                               ncc_p10=float(np.percentile(nc, 10)) if len(nc) else None,
                               gain_p5_p95=[float(np.percentile(stats['gain'], 5)),
                                            float(np.percentile(stats['gain'], 95))] if stats['gain'] else None,
                               mesh_abs_p95_px=float(np.percentile(np.abs(mesh[work]), 95)) if len(work) else 0.0))
    return tab


# ============================================================================================ driver + stats
def compute_fill(plan: Plan, video: Optional[str], virt_q: Optional[np.ndarray] = None, records=None,
                 prm: Optional[FillParams] = None, stream=None, progress=None, cancel=None) -> FillTable:
    """select_sources (+ align_sources when prm.align and a video is given)."""
    prm = prm or FillParams()
    vq = plan.virt_q if virt_q is None else np.asarray(virt_q, np.float64)
    if not np.any(vq):
        raise ValueError('compute_fill needs the virtual path (plan.virt_q or virt_q=analysis.npz virt_q)')
    if not np.any(plan.virt_q):
        plan = replace(plan, virt_q=vq)
    do_align = bool(prm.align and (video or stream is not None))
    p_sel = None if progress is None else (lambda f: progress((0.25 if do_align else 1.0) * f))
    tab = select_sources(plan, vq, records, prm, progress=p_sel)
    if do_align:
        p_al = None if progress is None else (lambda f: progress(0.25 + 0.75 * f))
        tab = align_sources(tab, plan, video, records, prm, stream=stream, cancel=cancel, progress=p_al)
    tab.meta['stats'] = fill_stats(tab, records)
    return tab


def fallback_fraction(tab: FillTable, records=None) -> float:
    r = np.arange(tab.n_frames) if records is None else np.asarray(records, np.int64)
    return float(np.mean(tab.frac_uncovered[r])) if len(r) else 0.0


def fill_stats(tab: FillTable, records=None) -> dict:
    r = np.arange(tab.n_frames) if records is None else np.unique(np.asarray(records, np.int64))
    if len(r) == 0:
        return {}
    ff, fu, ns = tab.frac_fill[r], tab.frac_uncovered[r], tab.n_src[r]
    return dict(records=int(len(r)), frac_frames_filled=float(np.mean(ff > 0)), fill_frac_mean=float(ff.mean()),
                fill_frac_p95=float(np.percentile(ff, 95)), fill_frac_max=float(ff.max()),
                uncovered_frac_mean=float(fu.mean()), uncovered_frac_p99=float(np.percentile(fu, 99)),
                uncovered_frac_max=float(fu.max()), n_src_mean=float(ns.mean()), max_offset=tab.max_offset)


def output_footprint(plan: Plan, records, n_edge: int = 64) -> np.ndarray:
    """UNCLIPPED output field of view as a fraction of the source area: the output's pixel-edge border mapped
    into the source (renderer maths, rolling-shutter fixed point), shoelace area / (src_w * src_h). Equals
    eval.footprint.plan_footprint while the output stays inside the source; larger when fill supplies the rest."""
    t = np.linspace(0, 1, n_edge, endpoint=False)
    W, H = plan.out_w, plan.out_h
    x0, x1, y0, y1 = -0.5, W - 0.5, -0.5, H - 0.5
    poly = np.concatenate([np.c_[x0 + t * (x1 - x0), y0 + 0 * t], np.c_[x1 + 0 * t, y0 + t * (y1 - y0)],
                           np.c_[x1 - t * (x1 - x0), y1 + 0 * t], np.c_[x0 + 0 * t, y1 - t * (y1 - y0)]])
    out = []
    for k in np.asarray(records, np.int64):
        S, _ = _proj(plan, int(k), poly[:, 0], poly[:, 1])
        x, y = S[:, 0], S[:, 1]
        out.append(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0 / (plan.src_w * plan.src_h))
    return np.asarray(out)


def main(argv=None):
    """python -m stillpoint.fill PLAN VIDEO --out PLAN_FILL [--virt analysis.npz] [--start S --dur D] [--no-align]
    Adds a FILL section to an existing plan (records of the window +- the offsets, or the whole plan)."""
    import argparse
    import json
    from .plan_io import read_plan, write_plan
    ap = argparse.ArgumentParser()
    ap.add_argument('plan')
    ap.add_argument('video')
    ap.add_argument('--out', required=True)
    ap.add_argument('--virt', default='', help='analysis.npz with virt_q (plan files do not store it)')
    ap.add_argument('--start', type=float, default=None, help='window start (s, source PTS)')
    ap.add_argument('--dur', type=float, default=None)
    ap.add_argument('--no-align', action='store_true')
    ap.add_argument('--set', action='append', default=[], help='FillParams field=value (python literal)')
    a = ap.parse_args(argv)
    import ast
    plan = read_plan(a.plan)
    vq = np.load(a.virt)['virt_q'] if a.virt else plan.virt_q
    kw = {}
    for kv in a.set:
        k, v = kv.split('=', 1)
        kw[k] = ast.literal_eval(v)
    prm = FillParams(align=not a.no_align, **kw)
    recs = None
    if a.start is not None:
        t0 = float(a.start)
        t1 = t0 + (float(a.dur) if a.dur else 1e9)
        recs = np.flatnonzero((plan.frame_pts >= t0 - 1e-6) & (plan.frame_pts < t1 - 1e-6))
    t = time.perf_counter()
    tab = compute_fill(plan, a.video, virt_q=vq, records=recs, prm=prm)
    write_plan(a.out, plan, fill=tab)
    info = dict(tab.meta.get('stats', {}), select_s=tab.meta.get('select_s'), align_s=tab.meta.get('align_s'),
                align=tab.meta.get('align'), wall_s=round(time.perf_counter() - t, 2),
                output_footprint_mean=float(output_footprint(replace(plan, virt_q=vq),
                                                             recs if recs is not None else np.arange(plan.n_frames)).mean())
                if True else None)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
