"""Parallax-aware MESH RESIDUAL on top of the rotation plan (workstream A; research/sota_algorithms.md §7.2).

Why: a rotation-only stabiliser (Stillpoint's plan, Gyroflow) is exact for far-field content only. On low FPV flights
the drone's small translational bob (a 2 cm bob at 2 m is ~16 px at 4K) makes the near-field ground / bushes bounce
against a steady horizon: depth-dependent motion that no rotation can remove. This stage measures the residual image
motion that is left in the STABILISED output, keeps only its high-frequency part (hp_hz..lp_hz: the jitter; the
intentional parallax of forward flight is low-frequency and stays) and bakes it into the plan as a bounded, spatially
smooth per-frame offset field (the .spplan mesh extension, plan_io.py). The Metal kernel applies it before the rotation
mapping (shaders/warp.metal sp_mesh_offset): output pixel X samples the rotation render at X + D_t(X).

Estimator (build_mesh):
 1. 960-wide previews of the plan (the shared Metal kernel) -> PERSISTENT KLT tracks (one Shi-Tomasi corner per 16 px
    cell, pyramidal LK with a forward-backward check, replenished every frame), in consecutive chunks on the process
    pool.
 2. Per frame pair a smooth POLYNOMIAL flow field is fitted to the track flows (robust Huber IRLS) -- and, as in the
    judge (eval/jitter_metrics.py step 3), its THREE-FRAME DELTA: the change of the field between (k-1 -> k) and
    (k -> k+1) fitted on the SAME tracks with the same weights (designs at the pairs' mid positions, fit_delta). In
    forward flight each track's own depth-dependent flow is nearly constant over two frames and largely cancels, and
    since the tracks persist their tracking errors telescope instead of random-walking. Default order 1 (AFFINE, 6
    parameters): the first-order image motion of a ground plane under a vertical / lateral bob is exactly
    translation + vertical stretch / horizontal skew about the horizon row (inverse depth of a plane is linear in the
    image and ~0 at the horizon), i.e. the near-ground bounce -- and the judge's HF and skew/stretch "jello" -- while
    the horizon row barely moves. Order 2 (quadratic, 12 parameters: the plane-induced flow incl. forward-speed
    terms) is available but its extra terms were too noisy on the OA4 low-flight proxy (HF and jello got worse).
    ~1000-2000 tracks per fit keep the estimate far below the per-track noise; the per-vertex estimators of the first
    attempt (grid LK re-seeded every pair, fixed-position vertex motion) were noise-dominated: their HF field was 2-3x
    the judge's HF and nearly uncorrelated with it.
 3. Field coefficient velocities = cumsum(deltas) with the < lf_hz part replaced by the single-pair fits (drift-free);
    coefficient paths = cumsum; jitter = zero-phase Butterworth band-pass [hp_hz, lp_hz] (the band where the residual
    lives; above lp_hz the estimate is mostly tracking noise). Frames with few tracks fade the field out.
 4. Offsets D at the mesh vertices = the jitter field (content at X is displaced by +D, so X samples X + D), soft
    clamped PER FRAME to clamp_px (1080p-equivalent; the frame's largest vertex offset, uniform scaling so an affine
    field stays affine -- a per-vertex clamp bends it and creates wobble), converted to full-res output px. Per frame,
    the field is scaled down (smoothly in time) where the displaced output border would sample outside the source
    (limit_border: no black, no extra crop).
 5. Wiener shrinkage (wiener=True): per field coefficient, the band-passed path is scaled by max(0, 1 - N^2/P) over
    1 s, N from split-half (odd / even track id) estimates -- estimator noise stays out of calm stretches.
 6. Verification (verify=True): previews WITH the mesh are tracked and fitted again; per 1-s window the mesh is kept
    only where the measured jitter field drops by >= accept_gain (35 %: a self-consistency test); elsewhere it is
    removed with xfade_s crossfades.

build_mesh() returns (mesh (F, ny, nx, 2) float32, diagnostics). pipeline.AnalyzeParams.mesh_residual runs it on the
final plan; `python -m stillpoint.cli mesh` adds a mesh to an existing plan (whole clip or windows).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

__all__ = ['MeshParams', 'Tracker', 'vertex_grid', 'mesh_shape', 'poly_basis', 'fit_single', 'fit_delta',
           'reconstruct_velocity', 'band_field', 'solve_fields', 'soft_clamp', 'limit_border', 'measure_motion',
           'window_accept', 'build_mesh', 'mesh_plan_file']


@dataclass
class MeshParams:
    nx: int = 17                     # vertex columns (16 cells)
    ny: int = 0                      # vertex rows; 0 -> round((nx-1)*out_h/out_w)+1 (16:9 -> 10, 4:3 -> 13)
    order: int = 1                   # polynomial order of the per-pair flow field (1 = affine, 2 = quadratic)
    model: str = 'poly'              # 'poly': the full order-`order` field; 'sim': its similarity part only
                                     # (translation + rotation + scale, first-column convention of the judge's
                                     # rolling-shutter-aware affine decomposition; needs order 1)
    hp_hz: float = 2.0               # keep the field-path part above this (the jitter) ...
    lp_hz: float = 6.0               # ... and below this (0 = no upper limit; the residual lives at 2-6 Hz, above it the
                                     # estimate is mostly tracking noise and reads as single-frame jumps)
    filt_order: int = 4              # Butterworth order (zero-phase filtfilt doubles it)
    lf_hz: float = 1.0               # velocity below this comes from the single-pair fits
    clamp_px: float = 3.0            # soft clamp of |offset| (1080p-equivalent px) ...
    clamp_mode: str = 'frame'        # ... 'frame': the whole frame's field is scaled by c*tanh(m/c)/m (m = its
                                     # largest vertex offset) so its shape is kept; 'vertex': per-vertex tanh (bends a
                                     # large affine / quadratic field into a non-rigid one -> wobble)
    # tracker (preview px; the values of the judge's tracker at 960 px)
    cell: int = 16                   # one corner per cell
    lk_win: int = 21
    lk_levels: int = 4
    fb_thr: float = 0.3              # forward-backward check (preview px)
    erode_px: int = 4                # grow the invalid (outside-source) region by this much
    # fits
    huber_k: float = 2.0             # single-pair Huber scale = k * median residual
    delta_k: float = 4.0             # delta Huber scale = k * median residual change
    parallax_c: float = 1.0          # delta: first-iteration Cauchy de-weighting of tracks off the field (preview px
                                     # at 960 wide: near-field outliers, moving objects)
    irls_iters: int = 6
    delta_design: str = 'mid'        # 'mid' | 'start' (the judge's convention) | 'common' (see fit_delta)
    min_tracks: int = 40             # a pair / delta needs this many tracks
    support_lo: int = 60             # frame support (tracks in the delta fit) at which the mesh is off ...
    support_hi: int = 200            # ... and fully on
    edge_s: float = 0.25             # fade the field in/out at the ends of every measured run (filter transients)
    wiener: bool = True              # per coefficient, shrink the band-passed field path by its local SNR:
    wiener_s: float = 1.0            # gain = max(0, 1 - noise^2 / power) over +-wiener_s/2, the noise from split-half
                                     # (odd / even track) estimates -- keeps estimator noise out of calm stretches
    border_levels: tuple = (1.0, 0.75, 0.5, 0.25, 0.0)
    border_samples: int = 8          # border test points per cell edge
    verify: bool = True              # re-measure with the mesh and accept per window
    window_s: float = 1.0
    accept_gain: float = 0.35        # keep a window only if its measured jitter field drops by >= this fraction: a
                                     # SELF-CONSISTENCY test -- where the estimate is right, re-measuring the corrected
                                     # previews finds most of it gone (calm / moderate seconds: -50..-90 %); in violent
                                     # low-flight seconds (strong parallax, 15-20 px/frame flow) it drops only 20-30 %
                                     # and the judge measured those corrections as added jitter (OA4 178-179 s)
    xfade_s: float = 0.25
    chunk: int = 90                  # frames per tracking job (consecutive chunks overlap by one frame)


# ============================================================================================ geometry helpers
def mesh_shape(out_w: int, out_h: int, p: MeshParams) -> tuple[int, int]:
    nx = max(2, int(p.nx))
    ny = int(p.ny) if p.ny and p.ny >= 2 else max(2, int(round((nx - 1) * out_h / out_w)) + 1)
    return nx, ny


def vertex_grid(out_w: int, out_h: int, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-res OUTPUT pixel positions (ny, nx) of the mesh vertices (warp.metal sp_mesh_offset convention)."""
    X = np.arange(nx) * (out_w - 1) / (nx - 1)
    Y = np.arange(ny) * (out_h - 1) / (ny - 1)
    return np.meshgrid(X, Y)


def poly_basis(q: np.ndarray, order: int, L: float) -> np.ndarray:
    """Field basis at centred preview points q (N,2): (N, nb) with nb = 3 (order 1) or 6 (order 2); coordinates / L."""
    x, y = q[:, 0] / L, q[:, 1] / L
    cols = [np.ones_like(x), x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    return np.stack(cols, 1)


def n_basis(order: int) -> int:
    return 3 if order <= 1 else 6


# ============================================================================================ tracking
_LK_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


class Tracker:
    """Persistent KLT tracks: step(img, valid) -> (ids, positions) of the tracks alive in this frame."""

    def __init__(self, W: int, H: int, p: MeshParams, max_per_cell: int = 2, min_eig_rel: float = 0.002,
                 min_eig_abs: float = 2e-5):
        self.W, self.H, self.p = W, H, p
        c = int(p.cell)
        self.c = c
        self.ny, self.nx = H // c, W // c
        gy, gx = np.mgrid[0:self.ny, 0:self.nx]
        self.cell_ok = ((gy > 0) & (gy < self.ny - 1) & (gx > 0) & (gx < self.nx - 1)).ravel()
        self.max_per_cell, self.min_eig_rel, self.min_eig_abs = max_per_cell, min_eig_rel, min_eig_abs
        self.ids = np.zeros(0, np.int64)
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.next_id = 0
        self.prev = None

    def _cells(self, pts):
        cx = np.clip((pts[:, 0] // self.c).astype(int), 0, self.nx - 1)
        cy = np.clip((pts[:, 1] // self.c).astype(int), 0, self.ny - 1)
        return cy * self.nx + cx

    def _vmask(self, img, valid):
        v = (img > 0) if valid is None else (np.asarray(valid, bool) & (img > 0))
        v = v.astype(np.uint8)
        e = int(self.p.erode_px)
        if e > 0 and not v.all():
            v = cv2.erode(v, np.ones((2 * e + 1, 2 * e + 1), np.uint8))
        return v

    def _replenish(self, img, vmask):
        c = self.c
        eig = cv2.cornerMinEigenVal(img, 7, 3)
        e = eig[: self.ny * c, : self.nx * c].reshape(self.ny, c, self.nx, c).transpose(0, 2, 1, 3)
        e = e.reshape(self.ny * self.nx, c * c)
        idx = e.argmax(1)
        emax = e[np.arange(len(idx)), idx]
        gy, gx = np.divmod(np.arange(len(idx)), self.nx)
        pts = np.column_stack([gx * c + idx % c, gy * c + idx // c]).astype(np.float32)
        thr = max(self.min_eig_abs, self.min_eig_rel * float(emax.max()))
        ok = (emax > thr) & self.cell_ok & (vmask[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 0)
        occ = np.zeros(self.ny * self.nx, int)
        if len(self.pos):
            ci = self._cells(self.pos)
            occ = np.bincount(ci, minlength=self.ny * self.nx)
            order = np.argsort(ci, kind='stable')
            rank = np.empty(len(ci), int)
            _, first = np.unique(ci[order], return_index=True)
            counts = np.diff(np.append(first, len(ci)))
            rank[order] = np.arange(len(ci)) - np.repeat(first, counts)
            keep = rank < self.max_per_cell
            self.ids, self.pos, self.vel = self.ids[keep], self.pos[keep], self.vel[keep]
        new = ok & (occ == 0)
        n = int(new.sum())
        if n:
            self.ids = np.concatenate([self.ids, self.next_id + np.arange(n)])
            self.next_id += n
            self.pos = np.vstack([self.pos, pts[new]]).astype(np.float32)
            self.vel = np.vstack([self.vel, np.zeros((n, 2), np.float32)])

    def step(self, img: np.ndarray, valid: Optional[np.ndarray] = None):
        p = self.p
        vmask = self._vmask(img, valid)
        if self.prev is not None and len(self.pos):
            win = (int(p.lk_win), int(p.lk_win))
            p0 = self.pos.reshape(-1, 1, 2)
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev, img, p0, (self.pos + self.vel).reshape(-1, 1, 2),
                                                  winSize=win, maxLevel=int(p.lk_levels), criteria=_LK_CRIT,
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(img, self.prev, p1, p0.copy(), winSize=win,
                                                   maxLevel=int(p.lk_levels), criteria=_LK_CRIT,
                                                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            p1 = p1.reshape(-1, 2)
            fb = np.linalg.norm(p0r.reshape(-1, 2) - self.pos, axis=1)
            ins = (p1[:, 0] >= 1) & (p1[:, 0] <= self.W - 2) & (p1[:, 1] >= 1) & (p1[:, 1] <= self.H - 2)
            good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < p.fb_thr) & ins & np.isfinite(p1).all(1)
            if good.any():
                gi = np.flatnonzero(good)
                q = np.clip(np.rint(p1[gi]), 0, [self.W - 1, self.H - 1]).astype(int)
                good[gi[vmask[q[:, 1], q[:, 0]] == 0]] = False
            self.vel = (p1 - self.pos)[good].astype(np.float32)
            self.ids, self.pos = self.ids[good], p1[good].astype(np.float32)
        self._replenish(img, vmask)                 # new corners exist in this frame too
        self.prev = img
        return self.ids.copy(), self.pos.astype(np.float64)


# ============================================================================================ fits
def fit_single(Phi: np.ndarray, F: np.ndarray, p: MeshParams) -> np.ndarray:
    """Robust (Huber IRLS) field fit: flows F (N,2) ~ Phi (N,nb) @ C (nb,2) -> C."""
    w = np.ones(len(Phi))
    C = np.zeros((Phi.shape[1], 2))
    for _ in range(int(p.irls_iters)):
        sw = np.sqrt(w)[:, None]
        C = np.linalg.lstsq(Phi * sw, F * sw, rcond=None)[0]
        d = np.hypot(*(F - Phi @ C).T)
        c = p.huber_k * float(np.median(d)) + 0.02
        w = np.minimum(1.0, c / np.maximum(d, 1e-9))
    return C


def fit_delta(Phi_a: np.ndarray, Phi_b: np.ndarray, F0: np.ndarray, F1: np.ndarray, p: MeshParams,
              par_c: float) -> np.ndarray:
    """Three-frame delta: the fields of pair a (k-1 -> k, flows F0, design Phi_a) and pair b (k -> k+1, flows F1,
    design Phi_b) fitted on the SAME tracks with the SAME weights (Huber on the residual change; the first iteration
    also Cauchy-de-weights tracks off the fields by par_c) -> C_b - C_a.
    Designs (MeshParams.delta_design): 'start' = each pair's field at its own start positions (the judge's convention:
    exact for a field that is exactly polynomial, however fast the content moves) or 'common' = both at the middle
    frame (exact cancellation of each track's own non-polynomial flow, but it picks up the convective term
    grad(u).u, which on fast low flights (15-20 px/frame, strongly varying flow) is large and noisy)."""
    w = np.ones(len(Phi_a))
    wpar = np.ones(len(Phi_a))
    Ca = Cb = np.zeros((Phi_a.shape[1], 2))
    for it in range(int(p.irls_iters)):
        sw = np.sqrt(w * wpar)[:, None]
        Ca = np.linalg.lstsq(Phi_a * sw, F0 * sw, rcond=None)[0]
        Cb = np.linalg.lstsq(Phi_b * sw, F1 * sw, rcond=None)[0]
        Ra, Rb = F0 - Phi_a @ Ca, F1 - Phi_b @ Cb
        dd = np.hypot(*(Rb - Ra).T)
        c = p.delta_k * float(np.median(dd)) + 0.02
        w = np.minimum(1.0, c / np.maximum(dd, 1e-9))
        if it == 0 and par_c > 0:
            rm = 0.5 * (np.hypot(*Ra.T) + np.hypot(*Rb.T))
            wpar = 1.0 / (1.0 + (rm / par_c) ** 2)
    return Cb - Ca


def delta_designs(Q0: np.ndarray, Q1: np.ndarray, Q2: np.ndarray, p: MeshParams, L: float):
    """(Phi_a, Phi_b) for the three positions of the delta tracks (see fit_delta)."""
    if p.delta_design == 'common':
        P1 = poly_basis(Q1, p.order, L)
        return P1, P1
    if p.delta_design == 'mid':
        return poly_basis(0.5 * (Q0 + Q1), p.order, L), poly_basis(0.5 * (Q1 + Q2), p.order, L)
    return poly_basis(Q0, p.order, L), poly_basis(Q1, p.order, L)


def _common(a, b):
    _, ia, ib = np.intersect1d(a, b, assume_unique=True, return_indices=True)
    return ia, ib


def _track_chunk_job(payload):
    """Frames [0, n) of a chunk of consecutive preview frames: a fresh tracker runs over all of them; pair i = (i, i+1)
    for i in [lo, n-1) gets a single-pair field fit, and a delta fit when frame i-1 precedes it in the chunk."""
    ks, imgs, vpk, p, lo, L = payload
    n, Hp, Wp = imgs.shape
    t0 = time.process_time()
    tr = Tracker(Wp, Hp, p)
    trk = []
    for i in range(n):
        valid = None
        if vpk[i] is not None:
            valid = np.unpackbits(vpk[i])[:Hp * Wp].reshape(Hp, Wp).astype(bool)
        if i > 0 and ks[i] != ks[i - 1] + 1:
            tr = Tracker(Wp, Hp, p)                                       # gap: restart
        trk.append(tr.step(np.ascontiguousarray(imgs[i]), valid))
    cc = np.array([(Wp - 1) / 2.0, (Hp - 1) / 2.0])
    par_c = p.parallax_c * Wp / 960.0
    nb = n_basis(p.order)
    out = []
    for i in range(lo, n - 1):
        if ks[i + 1] != ks[i] + 1:
            continue
        ia, ib = _common(trk[i][0], trk[i + 1][0])
        nan = np.full((nb, 2), np.nan)
        r = dict(k0=int(ks[i]), vp=nan.copy(), vd=nan.copy(), vph=np.stack([nan, nan]), vdh=np.stack([nan, nan]),
                 n_pair=int(len(ia)), n_delta=0)
        if len(ia) >= p.min_tracks:
            q0, q1 = trk[i][1][ia] - cc, trk[i + 1][1][ib] - cc
            Phi = poly_basis(q0, p.order, L)
            r['vp'] = fit_single(Phi, q1 - q0, p)
            par = trk[i][0][ia] % 2                                      # split halves (noise estimate)
            for h in (0, 1):
                m = par == h
                if m.sum() >= p.min_tracks // 2:
                    r['vph'][h] = fit_single(Phi[m], (q1 - q0)[m], p)
        if i >= 1 and ks[i - 1] == ks[i] - 1:
            ids01, i0, i1 = np.intersect1d(trk[i - 1][0], trk[i][0], assume_unique=True, return_indices=True)
            j1, j2 = _common(ids01, trk[i + 1][0])
            if len(j1) >= p.min_tracks:
                Q0 = trk[i - 1][1][i0[j1]] - cc
                Q1 = trk[i][1][i1[j1]] - cc
                Q2 = trk[i + 1][1][j2] - cc
                Pa, Pb = delta_designs(Q0, Q1, Q2, p, L)
                r['vd'] = fit_delta(Pa, Pb, Q1 - Q0, Q2 - Q1, p, par_c)
                r['n_delta'] = int(len(j1))
                par = ids01[j1] % 2
                for h in (0, 1):
                    m = par == h
                    if m.sum() >= p.min_tracks // 2:
                        r['vdh'][h] = fit_delta(Pa[m], Pb[m], (Q1 - Q0)[m], (Q2 - Q1)[m], p, par_c)
        out.append(r)
    cpu = (time.process_time() - t0) / max(len(out), 1)
    for r in out:
        r['cpu_s'] = cpu
    return out


def measure_motion(frames: Iterable, p: MeshParams, L: float, executor=None,
                   cancel: Optional[Callable[[], bool]] = None,
                   progress: Optional[Callable[[int], None]] = None) -> dict:
    """frames: iterator of (k, uint8 (Hp,Wp) preview, valid) in increasing k. Returns per measured pair (sorted k0):
    k0 (P,), vp, vd (P, nb, 2) field-coefficient velocity / delta (preview px per frame; NaN = not measured),
    n_pair, n_delta (P,)."""
    from collections import deque
    items: dict = {}
    it = iter(frames)
    n_read = 0
    exhausted = False
    results: list = []
    inflight: deque = deque()
    max_inflight = (getattr(executor, 'n_workers', 1) + 1) if executor is not None else 1
    t_start = time.perf_counter()

    def read_until(nn):
        nonlocal n_read, exhausted
        while not exhausted and n_read <= nn:
            try:
                item = next(it)
            except StopIteration:
                exhausted = True
                break
            img = np.ascontiguousarray(item[1], dtype=np.uint8)
            extra = np.asarray(item[2]).astype(bool) if len(item) > 2 and item[2] is not None else None
            vp = None if (extra is None or extra.all()) else np.packbits(extra.reshape(-1))
            items[n_read] = (int(item[0]), img, vp)
            n_read += 1

    def collect(block: bool):
        while inflight and (block or inflight[0].done()):
            fut = inflight.popleft()
            while block and not fut.done():
                if cancel is not None and cancel():
                    raise InterruptedError('mesh measurement cancelled')
                from concurrent.futures import wait as _wait
                _wait([fut], timeout=0.2)
            results.extend(fut.result())
            if progress is not None:
                progress(len(results))
            block = False

    i0 = 0
    chunk = max(8, int(p.chunk))
    try:
        while True:
            if cancel is not None and cancel():
                raise InterruptedError('mesh measurement cancelled')
            read_until(i0 + chunk + 1)
            n_pairs = n_read - 1
            if n_pairs <= i0:
                break
            i1 = min(i0 + chunk, n_pairs)
            a = max(0, i0 - 1)
            idx = list(range(a, i1 + 1))
            ks = np.array([items[j][0] for j in idx], dtype=np.int64)
            imgs = np.stack([items[j][1] for j in idx])
            vpk = [items[j][2] for j in idx]
            payload = (ks, imgs, vpk, p, i0 - a, L)
            if executor is None:
                results.extend(_track_chunk_job(payload))
                if progress is not None:
                    progress(len(results))
            else:
                inflight.append(executor.submit(_track_chunk_job, payload))
                collect(len(inflight) >= max_inflight)
            for j in [j for j in items if j < i1 - 1]:
                del items[j]
            i0 = i1
        while inflight:
            collect(True)
    finally:
        for f in inflight:
            f.cancel()
    results.sort(key=lambda r: r['k0'])
    nb = n_basis(p.order)
    P = len(results)
    return dict(k0=np.array([r['k0'] for r in results], np.int64),
                vp=np.stack([r['vp'] for r in results]) if P else np.zeros((0, nb, 2)),
                vd=np.stack([r['vd'] for r in results]) if P else np.zeros((0, nb, 2)),
                vph=np.stack([r['vph'] for r in results]) if P else np.zeros((0, 2, nb, 2)),
                vdh=np.stack([r['vdh'] for r in results]) if P else np.zeros((0, 2, nb, 2)),
                n_pair=np.array([r['n_pair'] for r in results]), n_delta=np.array([r['n_delta'] for r in results]),
                cpu_s=float(np.mean([r['cpu_s'] for r in results])) if P else 0.0,
                wall_s=time.perf_counter() - t_start)


# ============================================================================================ solving
def _filt(x: np.ndarray, fs: float, hz: float, kind: str, order: int) -> np.ndarray:
    sos = butter(order, min(hz / (fs / 2), 0.95), kind, output='sos')
    return sosfiltfilt(sos, x, axis=0, padtype='odd', padlen=min(len(x) - 1, int(3 * fs)))


def reconstruct_velocity(vp: np.ndarray, vd: np.ndarray, fs: float, lf_hz: float, order: int = 4) -> np.ndarray:
    """Single-pair velocities vp (P, ...) and three-frame changes vd (P, ...) (NaN = missing) -> velocity: cumsum of
    the changes with its < lf_hz part replaced by the single-pair estimate (eval/jitter_metrics.reconstruct_velocity)."""
    P = len(vp)
    shp = vp.shape
    a = vp.reshape(P, -1).astype(np.float64).copy()
    d = vd.reshape(P, -1).astype(np.float64).copy()
    for j in range(a.shape[1]):
        bad = ~np.isfinite(a[:, j])
        if bad.all():
            a[:, j] = 0.0
        elif bad.any():
            t = np.arange(P)
            a[bad, j] = np.interp(t[bad], t[~bad], a[~bad, j])
    dp = np.diff(a, axis=0, prepend=a[:1])
    bad = ~np.isfinite(d)
    d[bad] = dp[bad]
    d[0] = 0.0
    v = a[0] + np.cumsum(d, axis=0)
    if P >= 16:
        v = v + _filt(a - v, fs, lf_hz, 'lowpass', order)
    return v.reshape(shp)


def band_field(path: np.ndarray, fs: float, p: MeshParams) -> np.ndarray:
    """Band-pass [hp_hz, lp_hz] of coefficient paths (T, ...) along time."""
    T = len(path)
    x = path.reshape(T, -1)
    if T < 16:
        return np.zeros_like(path)
    h = x - _filt(x, fs, p.hp_hz, 'lowpass', p.filt_order)
    if p.lp_hz and p.lp_hz < 0.95 * fs / 2:
        h = _filt(h, fs, p.lp_hz, 'lowpass', p.filt_order)
    return h.reshape(path.shape)


def _runs(k0: np.ndarray, cut: set):
    """Consecutive pair runs (start, stop) into k0 (k0 increasing by 1), split where a pair crosses a shot cut."""
    if len(k0) == 0:
        return []
    br = set((np.flatnonzero(np.diff(k0) != 1) + 1).tolist())
    for i, kk in enumerate(k0):
        if int(kk) in cut:
            br.update((i, i + 1))
    edges = sorted(b for b in br if 0 < b < len(k0))
    starts = [0] + edges
    stops = edges + [len(k0)]
    return [(a, b) for a, b in zip(starts, stops) if b > a]


def solve_fields(meas: dict, F: int, fs: float, qv: np.ndarray, L: float, p: MeshParams,
                 segments: Optional[list] = None) -> tuple[np.ndarray, dict]:
    """Measurement -> jitter field J (F, nv, 2) in PREVIEW px at the centred vertex positions qv (nv, 2); 0 where not
    measured. Also returns diagnostics (per-frame support gain)."""
    nv = len(qv)
    J = np.zeros((F, nv, 2))
    gain = np.zeros(F)
    k0 = meas['k0']
    if len(k0) == 0:
        return J, dict(n_pairs=0)
    Phi = poly_basis(qv, p.order, L)
    cut = set(int(s_) - 1 for s_, _e in (segments or []))
    gains: list = []
    hw = max(1, int(round(0.25 * fs)))
    ne = max(1, int(round(p.edge_s * fs)))
    for a0, a1 in _runs(k0, cut):
        n = a1 - a0
        if n < 16:
            continue
        v = reconstruct_velocity(meas['vp'][a0:a1], meas['vd'][a0:a1], fs, p.lf_hz)
        path = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(v, axis=0)], axis=0)   # frames k0..k0+n
        C = band_field(path, fs, p)
        if p.wiener and meas.get('vdh') is not None:
            C, g = wiener_shrink(C, meas, a0, a1, fs, p)
            gains.append(g)
        if p.model == 'sim':
            C = similarity_part(C)
        Jr = np.einsum('vb,tbc->tvc', Phi, C)
        sup = meas['n_delta'][a0:a1].astype(float)
        sup = np.convolve(sup, np.ones(2 * hw + 1) / (2 * hw + 1), mode='same')
        sup = np.r_[sup, sup[-1]]
        g = np.clip((sup - p.support_lo) / max(p.support_hi - p.support_lo, 1e-9), 0.0, 1.0)
        ramp = np.minimum(1.0, np.minimum(np.arange(n + 1), np.arange(n, -1, -1)) / ne)
        g = g * (0.5 - 0.5 * np.cos(np.pi * ramp))
        f0 = int(k0[a0])
        J[f0:f0 + n + 1] = Jr * g[:, None, None]
        gain[f0:f0 + n + 1] = g
    diag = dict(n_pairs=int(len(k0)), tracks_pair_median=float(np.median(meas['n_pair'])),
                tracks_delta_median=float(np.median(meas['n_delta'])),
                gain_mean=float(gain[gain > 0].mean()) if (gain > 0).any() else 0.0)
    if gains:
        gg = np.concatenate([g.reshape(len(g), -1) for g in gains])
        diag['wiener_gain_mean'] = [round(float(x), 3) for x in gg.mean(0)]
    return J, diag


def wiener_shrink(C: np.ndarray, meas: dict, a0: int, a1: int, fs: float, p: MeshParams):
    """Band-passed coefficient paths C (T, nb, 2) of pairs [a0, a1) -> (C * gain, gain): per coefficient and time,
    gain = max(0, 1 - N^2 / E[C^2]) with N the noise of the full estimate from the split-half paths ((C_odd - C_even)/2,
    each half reconstructed and band-passed like C), both powers averaged over wiener_s."""
    halves = []
    for h in (0, 1):
        vh = reconstruct_velocity(meas['vph'][a0:a1, h], meas['vdh'][a0:a1, h], fs, p.lf_hz)
        ph = np.concatenate([np.zeros((1,) + vh.shape[1:]), np.cumsum(vh, axis=0)], axis=0)
        halves.append(band_field(ph, fs, p))
    N = 0.5 * (halves[0] - halves[1])
    from scipy.ndimage import uniform_filter1d
    w = max(3, int(round(p.wiener_s * fs)) | 1)
    S2 = uniform_filter1d(C ** 2, w, axis=0, mode='nearest')
    N2 = uniform_filter1d(N ** 2, w, axis=0, mode='nearest')
    g = np.clip(1.0 - N2 / np.maximum(S2, 1e-12), 0.0, 1.0)
    g = uniform_filter1d(g, w, axis=0, mode='nearest')                  # no gain steps
    return C * g, g


def soft_clamp(J: np.ndarray, c: float, mode: str = 'vertex') -> np.ndarray:
    """J (..., nv, 2): 'vertex': |J| -> c*tanh(|J|/c) per vertex vector; 'frame': every frame (leading index) scaled
    uniformly by c*tanh(m/c)/m, m = its largest vertex |J| (keeps the field's shape)."""
    r = np.linalg.norm(J, axis=-1, keepdims=True)
    if mode == 'frame':
        r = r.max(axis=-2, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.where(r > 1e-12, c * np.tanh(r / c) / np.maximum(r, 1e-12), 1.0)
    return J * s


def similarity_part(C: np.ndarray) -> np.ndarray:
    """Order-1 field coefficients (..., 3, 2) (rows 1, x, y; columns dx, dy) -> their similarity part with the
    judge's first-column convention: the x-column (a00, a10) is kept, the y-column becomes (-a10, a00)."""
    out = C.copy()
    out[..., 2, 0] = -C[..., 1, 1]
    out[..., 2, 1] = C[..., 1, 0]
    return out


# ============================================================================================ border / acceptance
def _border_points(out_w: int, out_h: int, nx: int, ny: int, per_cell: int):
    xs = np.linspace(0, out_w - 1, (nx - 1) * per_cell + 1)
    ys = np.linspace(0, out_h - 1, (ny - 1) * per_cell + 1)
    return np.concatenate([np.c_[xs, np.zeros_like(xs)], np.c_[xs, np.full_like(xs, out_h - 1)],
                           np.c_[np.zeros_like(ys), ys], np.c_[np.full_like(ys, out_w - 1), ys]])


def limit_border(plan, mesh: np.ndarray, records: np.ndarray, p: MeshParams, fs: float) -> tuple[np.ndarray, dict]:
    """Scale the mesh of each frame (uniform per frame, smoothed in time) so that the displaced output border never
    samples outside the source where the rotation-only plan did not. Returns (mesh, diag)."""
    from .render_ref import mesh_offset, source_points
    from dataclasses import replace as _replace
    F, ny, nx = mesh.shape[:3]
    pts = _border_points(plan.out_w, plan.out_h, nx, ny, int(p.border_samples))
    alpha = np.ones(F)
    base_plan = _replace(plan, mesh=None)
    n_lim = 0
    for k in np.asarray(records, dtype=np.int64):
        mk = mesh[k]
        if not np.any(mk):
            continue
        _, ok0 = source_points(base_plan, int(k), pts)
        dx, dy = mesh_offset(mk, pts[:, 0], pts[:, 1], plan.out_w, plan.out_h)
        for lev in p.border_levels:
            if lev <= 0:
                alpha[k] = 0.0
                break
            _, ok1 = source_points(base_plan, int(k), pts + lev * np.c_[dx, dy])
            if np.all(ok1 | ~ok0):
                alpha[k] = lev
                break
        if alpha[k] < 1.0:
            n_lim += 1
    if n_lim:
        from scipy.ndimage import minimum_filter1d, uniform_filter1d
        r = max(1, int(round(0.1 * fs)))
        a2 = minimum_filter1d(alpha, 2 * r + 1, mode='nearest')
        a2 = np.minimum(uniform_filter1d(a2, 2 * r + 1, mode='nearest'), alpha)
        mesh = mesh * a2[:, None, None, None]
        alpha = a2
    recs = np.asarray(records, dtype=np.int64)
    return mesh, dict(frames_limited=int(n_lim), alpha_mean=float(alpha[recs].mean()) if len(recs) else 1.0,
                      alpha_min=float(alpha.min()) if F else 1.0)


def window_rms(J: np.ndarray, valid_frames: np.ndarray, wl: int, to_1080: float) -> np.ndarray:
    """Per window of wl frames: RMS of the jitter field (1080p px) over frames x vertices. J (F, nv, 2)."""
    F = len(J)
    nw = max(1, F // wl)
    out = np.full(nw, np.nan)
    for w in range(nw):
        a, b = w * wl, (F if w == nw - 1 else (w + 1) * wl)
        vf = valid_frames[a:b]
        if vf.sum() < 0.5 * (b - a):
            continue
        out[w] = np.sqrt(np.mean(np.sum(J[a:b][vf] ** 2, -1))) * to_1080
    return out


def window_accept(h0: np.ndarray, h1: np.ndarray, p: MeshParams) -> np.ndarray:
    ok = np.isfinite(h0) & np.isfinite(h1)
    return ok & (h1 <= (1.0 - p.accept_gain) * h0)


def _window_mask(acc: np.ndarray, F: int, wl: int, xf: int) -> np.ndarray:
    m = np.zeros(F)
    for w, a in enumerate(acc):
        if a:
            m[w * wl:(F if w == len(acc) - 1 else (w + 1) * wl)] = 1.0
    if xf > 0:
        from scipy.ndimage import uniform_filter1d
        m = np.minimum(uniform_filter1d(m, 2 * xf + 1, mode='nearest'), 1.0)
    return m


# ============================================================================================ orchestration
def build_mesh(plan, stream, records: np.ndarray, render_fn, fs: float, p: Optional[MeshParams] = None,
               executor=None, cancel: Optional[Callable[[], bool]] = None,
               progress: Optional[Callable[[float, str], None]] = None, segments: Optional[list] = None,
               log: Optional[Callable[[str], None]] = None, meas: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """Mesh-residual offsets (F, ny, nx, 2) float32 (full-res output px; zero outside `records`) for `plan` + diag.

    stream: FrameStream (analysis width = preview width); render_fn: pipeline._render_stream (plan, stream, records,
    cancel=...) -> (k, u8, valid); records: sorted plan record indices to measure (consecutive runs; windows).
    meas: a measurement of this plan's previews from an earlier run (diag['_meas'], same tracker / order / delta design)
    -> the tracking pass is skipped (re-solve with other band / clamp settings)."""
    p = p or MeshParams()
    if p.model == 'sim' and p.order != 1:
        p = replace(p, order=1)
    t0 = time.perf_counter()
    F = plan.n_frames
    base = replace(plan, mesh=None)
    nx, ny = mesh_shape(plan.out_w, plan.out_h, p)
    X, Y = vertex_grid(plan.out_w, plan.out_h, nx, ny)
    Wp, Hp = int(stream.w), int(round(plan.out_h * stream.w / plan.out_w))
    sx, sy = Wp / plan.out_w, Hp / plan.out_h
    vx, vy = (X + 0.5) * sx - 0.5, (Y + 0.5) * sy - 0.5
    qv = np.stack([vx.reshape(-1) - (Wp - 1) / 2.0, vy.reshape(-1) - (Hp - 1) / 2.0], 1)
    L = Wp / 2.0
    to_1080 = 1920.0 / Wp
    recs = np.unique(np.asarray(records, dtype=np.int64))
    diag = dict(params=asdict(p), nx=nx, ny=ny, preview=[Wp, Hp], n_records=int(len(recs)))

    def prog(frac, msg):
        if progress is not None:
            progress(frac, msg)

    n_pass = 2 if p.verify else 1
    prog(0.0, 'mesh: tracking residual motion')
    if meas is None:
        meas = measure_motion(render_fn(base, stream, recs, cancel=cancel), p, L, executor, cancel,
                              lambda n: prog(0.9 * n / max(len(recs), 1) / n_pass, 'mesh: tracking'))
    diag['_meas'] = meas
    diag['measure_s'] = meas['wall_s']
    diag['pairs_per_s'] = len(meas['k0']) / max(meas['wall_s'], 1e-9)
    J, sd = solve_fields(meas, F, fs, qv, L, p, segments)
    diag['solve'] = sd
    J = soft_clamp(J, p.clamp_px / to_1080, p.clamp_mode)
    mesh = np.zeros((F, ny, nx, 2), np.float64)
    mesh[..., 0] = J[..., 0].reshape(F, ny, nx) / sx                     # preview px -> full-res output px
    mesh[..., 1] = J[..., 1].reshape(F, ny, nx) / sy
    mesh, bd = limit_border(base, mesh, recs, p, fs)
    diag['border'] = bd
    wl = max(1, int(round(p.window_s * fs)))
    vf = np.zeros(F, bool)
    vf[recs] = True
    Jraw, _ = solve_fields(meas, F, fs, qv, L, replace(p, support_lo=-1, support_hi=0, edge_s=0.0), segments)
    h0 = window_rms(Jraw, vf, wl, to_1080)
    if p.verify and np.any(mesh):
        prog(0.5, 'mesh: verifying')
        cand = replace(base, mesh=mesh.astype(np.float32))
        meas1 = measure_motion(render_fn(cand, stream, recs, cancel=cancel), p, L, executor, cancel,
                               lambda n: prog(0.45 + 0.45 * n / max(len(recs), 1), 'mesh: verifying'))
        J1, _ = solve_fields(meas1, F, fs, qv, L, replace(p, support_lo=-1, support_hi=0, edge_s=0.0), segments)
        h1 = window_rms(J1, vf, wl, to_1080)
        acc = window_accept(h0, h1, p)
        measured_w = np.isfinite(h0)
        m = _window_mask(acc, F, wl, int(round(p.xfade_s * fs)))
        mesh = mesh * m[:, None, None, None]
        diag['verify'] = dict(windows=int(measured_w.sum()), accepted=int(acc.sum()),
                              accepted_frac=float(acc[measured_w].mean()) if measured_w.any() else 0.0,
                              hf_before=float(np.sqrt(np.nanmean(h0 ** 2))),
                              hf_after_all=float(np.sqrt(np.nanmean(h1 ** 2))),
                              hf_after_kept=float(np.sqrt(np.nanmean(np.where(acc, h1, h0) ** 2))),
                              per_window=dict(h0=h0.tolist(), h1=h1.tolist(), acc=acc.tolist()),
                              measure_s=meas1['wall_s'])
    else:
        diag['verify'] = None
        diag['hf_before'] = float(np.sqrt(np.nanmean(h0 ** 2))) if np.isfinite(h0).any() else None
    mag = np.linalg.norm(mesh, axis=-1) * (1920.0 / plan.out_w)
    diag['offset_rms_1080'] = float(np.sqrt(np.mean(mag[recs] ** 2))) if len(recs) else 0.0
    diag['offset_max_1080'] = float(mag.max()) if mag.size else 0.0
    if np.any(mesh):                      # local strain: max neighbour offset difference / vertex spacing
        dxs = np.abs(np.diff(mesh, axis=2)).max() / ((plan.out_w - 1) / (nx - 1))
        dys = np.abs(np.diff(mesh, axis=1)).max() / ((plan.out_h - 1) / (ny - 1))
        diag['max_strain'] = float(max(dxs, dys))
    diag['total_s'] = time.perf_counter() - t0
    try:
        import resource
        import sys as _sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        diag['main_max_rss_mb'] = float(rss / (1 << 20) if _sys.platform == 'darwin' else rss / 1024)
    except Exception:                                   # noqa: BLE001 -- diagnostics only
        pass
    if log is not None:
        v = diag.get('verify') or {}
        log(f"mesh {nx}x{ny} order {p.order}: {diag['n_records']} frames, {diag['pairs_per_s']:.0f} pairs/s, "
            f"{sd.get('tracks_delta_median', 0):.0f} tracks/delta; offsets rms {diag['offset_rms_1080']:.2f} max "
            f"{diag['offset_max_1080']:.2f} px (1080p); verify: accepted {v.get('accepted')}/{v.get('windows')} "
            f"windows, field HF {v.get('hf_before', float('nan')):.3f} -> {v.get('hf_after_kept', float('nan')):.3f} "
            f"({diag['total_s']:.0f}s)")
    prog(1.0, 'mesh done')
    return mesh.astype(np.float32), diag


# ============================================================================================ plan files (CLI)
_MEAS_KEYS = ('order', 'delta_design', 'cell', 'lk_win', 'lk_levels', 'fb_thr', 'erode_px', 'huber_k', 'delta_k',
              'parallax_c', 'irls_iters', 'min_tracks')


def mesh_plan_file(video: str, plan_in: str, plan_out: str, start=None, dur=None, pad: float = 2.0,
                   processes: int = 3, overrides: Optional[dict] = None, width: int = 960, log=print,
                   meas_file: Optional[str] = None) -> dict:
    """Add a mesh residual to an existing plan -- whole clip, or the records within [start-pad, start+dur+pad] s of the
    SOURCE timeline (start/dur may be lists: several windows) -- and write it to plan_out. Returns the diagnostics
    (also written to plan_out + '.json'). meas_file: cache of the tracking pass (.npz): written if missing, reused
    (skipping the tracking pass) if present and made with the same plan, width, records and tracker/fit settings."""
    import json
    import os
    from .framestream import FrameStream
    from .pipeline import _env, _render_stream
    from .plan_io import read_plan, write_plan
    from .residual import make_pool, shutdown_pool
    from .video import probe
    p = MeshParams(**(overrides or {}))
    plan = read_plan(plan_in)
    info = probe(video)
    fpts = np.asarray(info['frame_pts'], dtype=np.float64)
    fdur = float(np.median(np.diff(fpts))) if len(fpts) > 1 else 1 / 60.0
    j = np.clip(np.searchsorted(fpts, plan.frame_pts), 1, len(fpts) - 1)
    j = np.where(np.abs(fpts[j - 1] - plan.frame_pts) < np.abs(fpts[j] - plan.frame_pts), j - 1, j)
    if np.any(np.abs(fpts[j] - plan.frame_pts) > 0.5 * fdur) or np.any(np.diff(j) <= 0):
        raise ValueError('plan records do not map 1:1 onto the video frames')
    plan.meta['frames'] = j
    recs = np.arange(plan.n_frames)
    if start is not None:
        starts = list(start) if np.ndim(start) else [start]
        durs = list(dur) if np.ndim(dur) else [dur] * len(starts)
        sel = np.zeros(plan.n_frames, bool)
        for s_, d_ in zip(starts, durs):
            t1 = s_ + (d_ if d_ else 1e9)
            sel |= (plan.frame_pts >= s_ - pad) & (plan.frame_pts <= t1 + pad)
        recs = recs[sel]
    fs = 1.0 / fdur
    if p.model == 'sim' and p.order != 1:
        p = replace(p, order=1)
    import hashlib
    with open(plan_in, 'rb') as fh:
        plan_sha = hashlib.sha1(fh.read()).hexdigest()[:16]
    key = json.dumps(dict(v=2, plan=plan_sha, width=int(width), recs=[int(recs[0]), int(recs[-1]), int(len(recs))]
                          if len(recs) else [], **{k: getattr(p, k) for k in _MEAS_KEYS}), sort_keys=True)
    meas = None
    if meas_file and os.path.exists(meas_file):
        z = np.load(meas_file, allow_pickle=False)
        if str(z['key']) == key:
            meas = {k: z[k] for k in ('k0', 'vp', 'vd', 'vph', 'vdh', 'n_pair', 'n_delta') if k in z.files}
            meas.update(wall_s=float(z['wall_s']), cpu_s=float(z['cpu_s']))
            log(f'mesh: reusing the tracking pass {meas_file}')
        else:
            log(f'mesh: {meas_file} was made with other settings -- measuring again (and replacing it)')
    stream = FrameStream(video, width, lanes=2, block=150, info=info)
    pool = None
    try:
        if processes and processes > 0 and (meas is None or p.verify):
            with _env({'MallocLargeCache': '0'}):
                pool = make_pool(processes)
        mesh, diag = build_mesh(plan, stream, recs, _render_stream, fs, p, executor=pool, log=log, meas=meas)
    finally:
        if pool is not None:
            shutdown_pool(pool)
        stream.close()
    m_ = diag.pop('_meas', None)
    if meas_file and meas is None and m_ is not None:
        np.savez(meas_file, key=np.array(key), **{k: m_[k] for k in ('k0', 'vp', 'vd', 'vph', 'vdh', 'n_pair',
                                                                     'n_delta')},
                 wall_s=m_['wall_s'], cpu_s=m_['cpu_s'])
    plan.mesh = mesh if np.any(mesh) else None
    plan.meta['mesh_clamp_px'] = float(p.clamp_px * plan.out_w / 1920.0)
    write_plan(plan_out, plan)
    diag.update(video=os.path.abspath(video), plan_in=os.path.abspath(plan_in), start=start, dur=dur, pad=pad,
                records=[int(recs[0]), int(recs[-1])] if len(recs) else None)
    with open(plan_out + '.json', 'w') as fh:
        json.dump(diag, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return diag
