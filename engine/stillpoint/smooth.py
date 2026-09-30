"""Crop-constrained camera-path optimizer (WP-C). ENGINE_SPEC.md §2, research/sota_algorithms.md §5.2.

The virtual camera is parameterized in CORRECTION space around the real camera:

    V_k = R_k · Exp(phi_k),   R_k = q_cam(frame_t[k])  (centre-row, mid-exposure orientation)
    out_fx_k = min_out_fx · exp(zeta_k),  0 <= zeta_k <= log(max_out_fx / min_out_fx)

and solved by SQP: every iteration linearizes around the current iterate (V⁰, zeta⁰) with
V = V⁰·Exp(delta), zeta = zeta⁰ + dzeta and solves ONE sparse QP (Clarabel) over the whole clip
(parallel overlapping windows for long clips — Clarabel releases the GIL):

  objective (all terms in OUTPUT pixels, f = min_out_fx px/rad):
      w_acc_l1·Σ|a_t|₁ + w_jerk_l1·Σ|j_t|₁ + w_acc_l2·Σ|a_t|² + w_vel_l2·Σ|w_t|²    (virtual motion)
    + w_jerk_l2·Σ|S²j_t|²             (S = 2-tap average: an ~omega^6 penalty on 2-8 Hz, well conditioned)
    + w_fidelity·Σ|f·phi_t|²                                                      (weak proximity)
    + w_horizon·Σ a_t·Huber(f·dist(rho_t, [lo_t, hi_t]))          (horizon lock v2, stage 2 only; see below)
    + zoom: w_zoom·Σzeta + w_zoom_rate_l1·Σ|Δzeta|₁ + w_zoom_acc_l2·Σ(Δ²zeta)²  (piecewise-constant)
    + w_crop_slack·Σ s_k                                    (exact penalty: s_k = 0 when feasible)
  w_t = Log(V_tᵀV_{t+1}) (body-frame virtual angular velocity, rad/frame, linearized with the SO(3)
  Jacobians J_r⁻¹/J_l⁻¹), a_t = w_{t+1}-w_t, j_t = a_{t+1}-a_t.

  HARD crop constraints: for every frame, 16 output-border samples (corners + 3 per edge) are
  mapped exactly as the renderer does (ENGINE_SPEC §1: rectilinear output ray -> R_cam(t_row)ᵀ·V ->
  KB4 fisheye, source row found by fixed-point iteration from the centre row) and must land inside
  [m, W-1-m] x [m, H-1-m]. They are linearized around the iterate including the rolling-shutter
  row/time coupling (dp = J·dθ / (1 - ∂p_y/∂y)); rows that cannot become active inside the trust
  region are pruned exactly (reach test). A per-frame slack with a large L1 weight makes the QP always
  feasible: when the crop truly cannot be satisfied (e.g. RS during a 900°/s flip with zoom capped),
  the path follows the camera as closely as possible instead of failing.

Units in the QP are scaled: rotation variables in output px (u = f·delta), zoom in px of border
motion (z = Z·dzeta, Z = output half-diagonal), so every weight is "per output pixel".

Start and trust region (2026-09-28): the SQP starts from a crop-feasible zero-phase low-pass of the camera path
(_warm_start), not from the camera, and the per-iteration trust radius is per frame (the schedule near the crop
border, a fraction of the distance to it elsewhere). Starting at the camera, the radii (0.06+0.03+0.012+0.005 rad)
capped |V - camera| at 6.1 deg per axis; in fast moves the path sat on that cap far from any crop border and copied
the camera's 2-4 Hz shake (DJI_20260927091931_0012 365 s: 11.9 px of 2-8 Hz; DJI_0027 26-27 s: 1.4 px).

Horizon lock v2 (2026-09-29; `horizon_lock` = strength 0..1, `roll_limit_deg`). Quaternion/vector native: the
horizon angle of a virtual camera is rho = atan2(g_x, g_y) with g = Vᵀ·g_w the gravity direction in camera
coordinates (x right, y down, z forward) -- the angle of the image-plane projection of 'down' against the image's
down axis. It is defined for every attitude except looking exactly straight up/down (no Euler angles, no yaw/pitch
decomposition: Gyroflow's rebuild from asin/atan2 of the view vector is singular at +-90 deg pitch and flips 180 deg
through loops). Its exact linearization for V -> V·Exp(delta) is d rho = delta_z - g_z (g_x delta_x + g_y delta_y)/n²,
n² = g_x² + g_y² (a roll about the optical axis changes rho one-to-one).
  1. Availability a_t in [0,1] from the CAMERA path: 1 below / 0 above a steepness (|asin g_z|, the optical axis
     elevation), an inverted-attitude (|rho|) and a tilt-rate (angular rate minus its component about gravity: a flat
     yaw spin keeps the lock) threshold, then eroded and ramped over +-horizon_fade_s. Offline, so the lock fades out
     BEFORE a flip / dive / roll-through-inverted and re-enters smoothly after it.
  2. Stage 1: the ordinary (unlocked) path V1 -- the lock never changes the smoothing itself.
  3. Target: from V1's own (smooth) horizon angle rho1, the ideal roll change is Delta = -a·s·dz_L(rho1) (dz_L:
     the part of the bank beyond the roll limit L; s = strength). Leveling costs crop: the largest fraction beta of
     Delta that keeps V1·Exp(beta·Delta·e_z) inside the crop (exact RS-aware mapping, an extra inner margin), eroded
     and ramped in time like a_t, so the leveled target is smooth AND feasible with room to spare. A target on the
     crop border would make the path hug the border and copy the camera's shake along it.
  4. Stage 2: the SQP again from V1·Exp(beta·Delta·e_z) with the horizon term w_horizon·a_t·Huber(f·dist(rho_t,
     [min(-L, tau_t), max(L, tau_t)])), tau = rho1 + beta·Delta: no pull inside the allowed interval (between the
     reachable target and the roll-limit band), quadratic up to horizon_huber_deg, linear beyond (bounded gradient, so
     the exact-penalty crop constraint always wins). Modelled exactly in the QP with two variables per frame. The
     zoom may not exceed stage 1's at any zoom knot (horizon_zoom=False): leveling never costs crop.
Cost: one extra SQP solve (the path QP is a few % of an analysis). Off (strength 0) the optimizer is unchanged.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from .geom import Lens, qconj, qexp, qfix_sign, qlog, qmul, qnormalize, quat_to_mat, slerp_series

try:  # imported lazily in tests if missing
    import clarabel
except ImportError:  # pragma: no cover
    clarabel = None

__all__ = ['SmoothParams', 'optimize_path', 'check_crop', 'fx_for_crop_area', 'map_output_points',
           'border_samples', 'gravity_world', 'horizon_angles', 'horizon_jacobian', 'horizon_availability']


# ============================================================================= parameters


@dataclass
class SmoothParams:
    """Knobs for optimize_path. `smoothness` is the one user-facing control (0 = follow the camera
    closely, 1 = cinematic FPV default, 2 = very floaty); the weights below are derived from it unless
    set explicitly."""
    smoothness: float = 1.0
    min_out_fx: float = 0.0            # widest output focal (output px). <= 0 -> auto from crop_area
    max_out_fx: Optional[float] = None  # tightest allowed zoom; None -> 1.5 * min_out_fx
    allow_zoom: bool = True             # False -> out_fx == min_out_fx everywhere
    horizon_lock: float = 0.0           # horizon lock strength 0..1 (fraction of the bank beyond roll_limit_deg that
                                        # is removed; True = 1). Needs tel.gravity_q (silently off when absent)
    roll_limit_deg: float = 0.0         # > 0: bank up to this many degrees is left to the smoother (natural FPV feel);
                                        # only the part beyond it is leveled
    margin_px: float = 8.0              # source-pixel margin (resampling kernel + border sampling)
    crop_extend: Optional[np.ndarray] = None   # (F,4[,n]) per frame of frame_ids: how far [source px] the output border
                                        # may leave the source beyond the [left, top, right, bottom] edge, optionally as
                                        # a profile of n points along each edge (left/right: top->bottom, top/bottom:
                                        # left->right; a sample uses the min of the two nearest points). Full-frame
                                        # fill supplies those pixels (fill.coverage_overscan). None = 0
    crop_area: float = 0.75             # only used when min_out_fx <= 0
    max_zoom_rate: float = 0.04         # max |d(log zoom)/dt| [1/s]
    zoom_knot: int = 15                 # zoom variable every N frames (linear in between)
    # --- solver
    sqp_iters: int = 4
    trust_rad: tuple = (0.06, 0.03, 0.012, 0.005)          # per-iteration |delta|_inf bound [rad]
    trust_zoom: tuple = (0.25, 0.08, 0.03, 0.01)           # per-iteration |dzeta| bound
    trust_far: tuple = (0.35, 0.2, 0.08, 0.03)             # per-iteration cap of the per-frame radius of frames far
                                                           # from the crop border (fraction-to-the-boundary, below)
    trust_frac: float = 0.5             # far frames may move this fraction of their (linearized) distance to the crop
                                        # border per iteration. With one global radius, trust_rad summed to 6.1 deg,
                                        # which capped |V - camera| per axis: in fast FPV moves (250-350 deg/s) the path
                                        # sat on that cap and followed the camera's 2-8 Hz shake (DJI_0027 26-27 s: 1.4 px)
    repair_iters: int = 3               # extra iterations if the exact check still finds violations
    viol_tol_px: float = 0.25           # ... larger than this (source px, beyond margin_px)
    samples_long: int = 3               # interior border samples per long edge
    samples_short: int = 3              # interior border samples per short edge
    rs_iters: int = 3                   # fixed-point iterations for the source row (ENGINE_SPEC §1)
    window: int = -1                    # frames per parallel QP window core; 0 = one global QP;
                                        # -1 = auto (one window per worker, >= 1500 frames)
    overlap: int = 300                  # extra frames on each side of a window core
    workers: int = 0                    # 0 -> min(8, cpu_count)
    lens: Optional[Lens] = None         # override tel.lens (e.g. after TimeModel.focal_scale)
    readout_s: Optional[float] = None   # override tel.readout_s (e.g. TimeModel.readout_s)
    # --- weights (px units). None -> derived from smoothness
    w_acc_l1: float = 1.0
    w_jerk_l1: float = 10.0
    w_acc_l2: Optional[float] = None    # L2 on acceleration: the main 'smooth ease-in/out' term (tuned
                                        # on DJI_0025: 30 cuts calm >2 Hz path content 0.09 -> 0.02 px)
    w_jerk_l2: float = 1000.0           # L2 on (smoothed) jerk: weighs 2-8 Hz path motion ~w*omega^6 vs LF, so a path
                                        # pressed against the crop border no longer copies the camera's shake along it
                                        # (DJI_0034 23 s: 0.93 -> 0.41 px of 2-8 Hz; +0-2 IPM iterations)
    jerk_l2_smooth: int = 2             # 2-tap velocity averages applied inside the jerk L2 term (conditioning)
    w_vel_l2: Optional[float] = None
    w_fidelity: Optional[float] = None
    w_horizon: float = 50.0             # per output px² of horizon-angle excess (quadratic up to horizon_huber_deg)
    # --- horizon lock v2 (module docstring). Angles in degrees, rates in deg/s, (full lock, no lock) pairs
    horizon_huber_deg: float = 0.5      # linear beyond: gradient <= 2·w·f·0.5° (~1.4e3/px) < crop slack weight
    horizon_steep_deg: tuple = (60.0, 78.0)      # |optical-axis elevation|: near nadir/zenith rho is ill-defined
                                                 # and a level roll would spin with every pan (coupling tan(elev))
    horizon_inverted_deg: tuple = (95.0, 135.0)  # |camera horizon angle|: rolling through / hanging inverted
    horizon_rate_dps: tuple = (300.0, 600.0)     # camera tilt rate (angular rate minus its part about gravity)
    horizon_fade_s: float = 0.35        # availability (steep / inverted / flip): eroded then ramped over +-this
    horizon_crop_fade_s: float = 1.0    # the crop-feasible leveling fraction: eroded + smoothed over +-this (the
                                        # per-frame feasibility follows the camera's shake; a fast-varying target
                                        # would put that shake into the roll: 2-8 Hz x4.6 on a 35-deg synthetic turn)
    horizon_margin_px: float = 16.0     # extra source-px margin the leveled target keeps from the crop border
    horizon_fracs: int = 10             # crop-fraction search resolution (beta in 0, 1/n, ..., 1)
    horizon_stage2_iters: int = 2       # stage-2 main SQP iterations (its start is stage 1's optimum, leveled): the
                                        # last entries of the trust schedule, then the usual repair iterations
    horizon_zoom: bool = False          # False: stage 2 may not zoom in beyond the unlocked (stage-1) path at any zoom
                                        # knot -- leveling never costs crop. (Free zoom: the Huber horizon term is far
                                        # dearer than w_zoom, so where the path strayed from the target the QP zoomed:
                                        # DJI_0033 80-106 s mean zoom 1.068 -> 1.072, max 1.25 -> 1.29.)
    w_zoom: float = 200.0               # per unit log-zoom per frame (prefer wide). 10x the M1 value: once the path may
                                        # use the whole crop (warm start), 20/2000 zoomed DJI_0027 26 % of the time
                                        # (max 1.24x, was 12 % / 1.13x); 200/20000: 4 % / 1.10x, 0012 max 1.01x
    w_zoom_rate_l1: float = 20000.0     # per unit |Δ log-zoom| -> piecewise-constant zoom
    w_zoom_acc_l2: float = 2e5          # rounds the corners of zoom ramps
    w_crop_slack: float = 1e4           # per source px of crop violation per frame (exact penalty)
    prox: float = 1e-3                  # SQP proximal weight on the step (px^-2): picks the minimal step
                                        # on the flat optimal faces of the L1 objective (no wandering)
    axis_w: tuple = (1.0, 1.0, 1.0)     # (pitch x, yaw y, roll z) weights on motion terms
    # --- warm start: the SQP starts from a crop-feasible low-passed camera path instead of the camera itself
    warm_start: bool = True
    warm_fc: float = 0.5                # [Hz] zero-phase low-pass of the camera path used as the start
    warm_alphas: tuple = (1.0, 0.75, 0.5, 0.25)   # tried blends camera -> low-pass (largest crop-feasible one per frame)
    warm_erode_s: float = 0.4           # alpha is eroded + box-averaged over +-this (stays <= the feasible alpha)
    # --- blur-aware smoothing: in frames whose exposure holds a long motion streak, the path follows the camera's
    #     own (exposure-averaged) motion ALONG the streak a little -- the blur already shows that motion, so matching
    #     it hides the baked blur instead of making it judder (research/sota_algorithms.md §7.3 eq. 15; Apple
    #     US9674438 strength modulation). Term: w_blur * beta_t * (e_t/T_f)^2 * (bhat_t . (vbar_virt,t - v_cam,t))^2
    #     [px^2], vbar = mean of the virtual velocities into / out of frame t, v_cam = the camera's rate averaged over
    #     the exposure (px/frame), bhat its direction, beta_t = ramp of the streak length (1080p-eq px) from
    #     blur_b0_px (0) to blur_b1_px (1). Only the along-streak component is pulled: cross-streak shake is still
    #     removed. 0 = off.
    w_blur: float = 0.0
    blur_b0_px: float = 2.0
    blur_b1_px: float = 8.0
    verbose: bool = False
    cancel: Optional[Callable[[], bool]] = None      # polled per SQP iteration and per solved window;
                                                     # True -> InterruptedError (analysis cancel <= ~1 s)
    tick: Optional[Callable[[float], None]] = None   # progress within optimize_path (0..1), per iteration

    def fidelity(self) -> float:
        if self.w_fidelity is not None:
            return float(self.w_fidelity)
        s = max(0.0, float(self.smoothness))
        return 1e-4 * 10.0 ** (2.0 * (1.0 - s))

    def acc_l2(self) -> float:
        if self.w_acc_l2 is not None:
            return float(self.w_acc_l2)
        return 30.0 * 10.0 ** (max(0.0, float(self.smoothness)) - 1.0)

    def vel_l2(self) -> float:
        if self.w_vel_l2 is not None:
            return float(self.w_vel_l2)
        return 1e-4 * max(0.0, float(self.smoothness))

    def lock_strength(self) -> float:
        """Horizon lock strength in [0, 1] (bool True -> 1)."""
        try:
            s = float(self.horizon_lock or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, s)) if math.isfinite(s) else 0.0


# ============================================================================= SO(3) helpers


def _hat(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    z = np.zeros(v.shape[:-1])
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([np.stack([z, -w, y], -1), np.stack([w, z, -x], -1), np.stack([-y, x, z], -1)], -2)


def _jr_inv(th: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian of SO(3): Exp(th)·Exp(d) ≈ Exp(th + J_r⁻¹(th)·d)."""
    th = np.asarray(th, dtype=np.float64)
    a = np.linalg.norm(th, axis=-1)
    H = _hat(th)
    small = a < 1e-3
    a_s = np.where(small, 1.0, a)
    c = np.where(small, 1.0 / 12.0 + a * a / 720.0,
                 1.0 / a_s ** 2 - (1.0 + np.cos(a_s)) / (2.0 * a_s * np.sin(np.where(small, 1.0, a_s))))
    return np.eye(3) + 0.5 * H + c[..., None, None] * (H @ H)


# ============================================================================= exact mapping


def border_samples(out_w: int, out_h: int, n_long: int = 3, n_short: int = 3) -> np.ndarray:
    """Output-border sample points (S,2): 4 corners + n interior points per edge (pixel centres)."""
    W1, H1 = out_w - 1.0, out_h - 1.0
    nx, ny = (n_long, n_short) if out_w >= out_h else (n_short, n_long)
    fx = np.arange(1, nx + 1) / (nx + 1) * W1
    fy = np.arange(1, ny + 1) / (ny + 1) * H1
    pts = [(0, 0), (W1, 0), (W1, H1), (0, H1)]
    pts += [(x, 0.0) for x in fx] + [(x, H1) for x in fx] + [(0.0, y) for y in fy] + [(W1, y) for y in fy]
    return np.asarray(pts, dtype=np.float64)


def _edge_layout(out_w: int, out_h: int, n_long: int, n_short: int):
    """For each output edge: (sample indices ordered along it incl. corners, varying axis, fixed value,
    normal component, outward sign). Matches border_samples() ordering."""
    nx, ny = (n_long, n_short) if out_w >= out_h else (n_short, n_long)
    top = [0] + list(range(4, 4 + nx)) + [1]
    bot = [3] + list(range(4 + nx, 4 + 2 * nx)) + [2]
    lef = [0] + list(range(4 + 2 * nx, 4 + 2 * nx + ny)) + [3]
    rig = [1] + list(range(4 + 2 * nx + ny, 4 + 2 * nx + 2 * ny)) + [2]
    return [(np.array(top), 0, 0.0, 1, -1.0), (np.array(bot), 0, out_h - 1.0, 1, 1.0),
            (np.array(lef), 1, 0.0, 0, -1.0), (np.array(rig), 1, out_w - 1.0, 0, 1.0)]


def _adaptive_points(p_fixed: np.ndarray, layout, out_w: int, out_h: int) -> np.ndarray:
    """Per frame and edge, the output-border point whose source image lies furthest OUT along the edge
    normal (parabolic fit through the outermost fixed sample and its neighbours). The mapped border is
    curved (rectilinear output of a fisheye) and its extreme wanders along the edge with roll/RS, so the
    fixed samples alone can miss a violation by ~10+ px. Returns (F, 4, 2) output pixel coords."""
    F = p_fixed.shape[0]
    out = np.empty((F, len(layout), 2))
    for e, (idx, ax, fixed, comp, sgn) in enumerate(layout):
        vals = sgn * p_fixed[:, idx, comp]                       # (F, n) outward distance-like
        n = len(idx)
        L = (out_w - 1.0) if ax == 0 else (out_h - 1.0)
        spacing = L / (n - 1)
        i_star = np.argmax(vals, axis=1)
        i0 = np.clip(i_star, 1, n - 2)
        r = np.arange(F)
        ym, y0, yp = vals[r, i0 - 1], vals[r, i0], vals[r, i0 + 1]
        den = ym - 2.0 * y0 + yp
        t_vertex = 0.5 * (ym - yp) / np.where(den < -1e-12, den, -1.0)
        t_end = np.where(yp >= ym, 0.95, -0.95)
        t = np.where(den < -1e-12, np.clip(t_vertex, -0.95, 0.95), t_end)
        pos = np.clip((i0 + t) * spacing, 0.0, L)
        out[:, e, ax] = pos
        out[:, e, 1 - ax] = fixed
    return out


def _q_at(q_cam_fn, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q_cam_fn(t.reshape(-1)), dtype=np.float64)
    return q.reshape(t.shape + (4,))


def map_output_points(q_cam_fn, frame_t, virt_q, out_fx, pts, out_w, out_h, lens: Lens, readout_s: float,
                      src_h: int, iters: int = 3, jac: bool = False):
    """Exact (renderer-equivalent) mapping of output pixels to source pixels.

    frame_t (N,), virt_q (N,4), out_fx (N,), pts (S,2) or (N,S,2) output pixel coords.
    Returns p (N,S,2) source pixels [and, with jac=True, (Gd (N,S,2,3) px/rad w.r.t. a right-multiplied
    rotation delta of V, Gz (N,S,2) px per unit log-zoom), both including the RS row coupling]."""
    frame_t = np.asarray(frame_t, dtype=np.float64)
    N = len(frame_t)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 2:
        pts = np.broadcast_to(pts, (N,) + pts.shape)
    fo = np.asarray(out_fx, dtype=np.float64).reshape(N, 1)
    cx, cy = (out_w - 1) / 2.0, (out_h - 1) / 2.0
    d = np.stack([(pts[..., 0] - cx) / fo, (pts[..., 1] - cy) / fo, np.ones(pts.shape[:2])], -1)  # (N,S,3)
    Vm = quat_to_mat(virt_q)                                    # (N,3,3)
    w = np.einsum('nij,nsj->nsi', Vm, d)                        # world rays
    y = np.full(pts.shape[:2], (src_h - 1) / 2.0)

    def proj_at(yrow):
        t = frame_t[:, None] + readout_s * ((yrow + 0.5) / src_h - 0.5)
        Rr = quat_to_mat(_q_at(q_cam_fn, t))                    # (N,S,3,3)
        X = np.einsum('nsji,nsj->nsi', Rr, w)                   # Rrowᵀ w
        return lens.project(X), X, Rr

    for _ in range(iters):
        y = proj_at(y)[0][..., 1]
    p, X, Rr = proj_at(y)
    if not jac:
        return p
    # ∂π/∂X by central differences (scale-aware)
    h = 1e-6 * np.linalg.norm(X, axis=-1, keepdims=True)
    Jpi = np.empty(X.shape[:-1] + (2, 3))
    for a in range(3):
        e = np.zeros(3)
        e[a] = 1.0
        Jpi[..., :, a] = (lens.project(X + h * e) - lens.project(X - h * e)) / (2.0 * h)
    M = np.einsum('nsji,njk->nsik', Rr, Vm)                     # Rrowᵀ V
    dXd = -np.einsum('nsij,nsjk->nsik', M, _hat(d))            # ∂X/∂delta
    dXz = np.einsum('nsij,nsj->nsi', M, np.stack([-d[..., 0], -d[..., 1], np.zeros_like(d[..., 0])], -1))
    Gd = np.einsum('nsij,nsjk->nsik', Jpi, dXd)                 # (N,S,2,3)
    Gz = np.einsum('nsij,nsj->nsi', Jpi, dXz)                   # (N,S,2)
    # rolling-shutter coupling: p = P(theta, y(p)); dp_y = ∂P_y/(1 - g_y), dp_x = ∂P_x + g_x dp_y
    hy = 1.0
    g = (proj_at(y + hy)[0] - proj_at(y - hy)[0]) / (2.0 * hy)   # (N,S,2)
    den = np.clip(1.0 - g[..., 1], 0.2, None)
    Gd_y = Gd[..., 1, :] / den[..., None]
    Gd[..., 0, :] += g[..., 0:1] * Gd_y
    Gd[..., 1, :] = Gd_y
    Gz_y = Gz[..., 1] / den
    Gz[..., 0] += g[..., 0] * Gz_y
    Gz[..., 1] = Gz_y
    return p, Gd, Gz


def check_crop(tel, q_cam_fn, frame_ids, virt_q, out_fx, out_w, out_h, n_per_edge: int = 64,
               margin_px: float = 0.0, lens: Optional[Lens] = None, readout_s: Optional[float] = None,
               iters: int = 3, chunk: int = 2000) -> np.ndarray:
    """Per-frame maximum crop violation [source px] over a DENSE output border (n_per_edge samples per
    edge), exact RS-aware mapping. <= 0 means every border sample lands inside [m, W-1-m]x[m, H-1-m]."""
    lens = lens or tel.lens
    ro = tel.readout_s if readout_s is None else readout_s
    W, H = tel.width, tel.height
    xs = np.linspace(0, out_w - 1, n_per_edge)
    ys = np.linspace(0, out_h - 1, n_per_edge)
    pts = np.concatenate([np.stack([xs, np.zeros_like(xs)], 1), np.stack([xs, np.full_like(xs, out_h - 1)], 1),
                          np.stack([np.zeros_like(ys), ys], 1), np.stack([np.full_like(ys, out_w - 1), ys], 1)])
    frame_ids = np.asarray(frame_ids)
    ft = tel.frame_t[frame_ids]
    out = np.empty(len(frame_ids))
    for a in range(0, len(frame_ids), chunk):
        b = min(len(frame_ids), a + chunk)
        p = map_output_points(q_cam_fn, ft[a:b], virt_q[a:b], out_fx[a:b], pts, out_w, out_h, lens, ro, H, iters)
        m = margin_px
        v = np.maximum.reduce([m - p[..., 0], p[..., 0] - (W - 1 - m), m - p[..., 1], p[..., 1] - (H - 1 - m)])
        out[a:b] = v.max(axis=1)
    return out


def fx_for_crop_area(lens: Lens, src_w: int, src_h: int, out_w: int, out_h: int, area_frac: float = 0.75,
                     n_per_edge: int = 128) -> float:
    """Output focal (px) for which the output rectangle, mapped at identity (no rotation, no RS), covers
    `area_frac` of the source image area."""
    xs = np.linspace(0, out_w - 1, n_per_edge)
    ys = np.linspace(0, out_h - 1, n_per_edge)
    poly = np.concatenate([np.stack([xs, np.zeros_like(xs)], 1), np.stack([np.full_like(ys, out_w - 1), ys], 1),
                           np.stack([xs[::-1], np.full_like(xs, out_h - 1)], 1),
                           np.stack([np.zeros_like(ys), ys[::-1]], 1)])
    cx, cy = (out_w - 1) / 2.0, (out_h - 1) / 2.0

    def area(fx):
        d = np.stack([(poly[:, 0] - cx) / fx, (poly[:, 1] - cy) / fx, np.ones(len(poly))], 1)
        p = lens.project(d)
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / (src_w * src_h)

    lo, hi = 0.05 * out_w, 50.0 * out_w
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if area(mid) > area_frac:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ============================================================================= QP (one window)


def _diff_op(n: int, order: int) -> sp.csr_matrix:
    c = np.array([1.0])
    for _ in range(order):
        c = np.convolve(c, [-1.0, 1.0])
    m = n - order
    if m <= 0:
        return sp.csr_matrix((0, n))
    return sp.diags([np.full(m, c[i]) for i in range(order + 1)], list(range(order + 1)), shape=(m, n), format='csr')


def _solve_window(job: dict) -> dict:
    """Build and solve one SQP QP for a contiguous window. Pure numpy/scipy/clarabel (thread-safe)."""
    t0 = time.perf_counter()
    Fw = job['n']
    f = job['f']
    Zs = job['Zs']
    wax = np.asarray(job['axis_w'], dtype=np.float64)
    zoom = job['zoom']
    x = job['x']                      # (Fw-1,3) rad/frame
    Jr = _jr_inv(x)
    Jl = _jr_inv(-x)
    # ---- virtual velocity operator in scaled variables u = f*delta:  w_px = w0 + Wop u
    nb = Fw - 1
    rr = np.repeat(np.arange(3 * nb).reshape(nb, 3), 3, axis=1).reshape(nb, 3, 3)  # row index
    cc_next = (3 * (np.arange(nb) + 1))[:, None, None] + np.arange(3)[None, None, :]
    cc_cur = (3 * np.arange(nb))[:, None, None] + np.arange(3)[None, None, :]
    cc_next = np.broadcast_to(cc_next, (nb, 3, 3))
    cc_cur = np.broadcast_to(cc_cur, (nb, 3, 3))
    vals_next = wax[None, :, None] * Jr
    vals_cur = -wax[None, :, None] * Jl
    Wop = sp.csr_matrix((np.concatenate([vals_next.ravel(), vals_cur.ravel()]),
                         (np.concatenate([rr.ravel(), rr.ravel()]), np.concatenate([cc_next.ravel(), cc_cur.ravel()]))),
                        shape=(3 * nb, 3 * Fw))
    w0 = (f * wax[None, :] * x).ravel()
    I3 = sp.identity(3, format='csr')
    D1 = sp.kron(_diff_op(nb, 1), I3, format='csr')
    D2 = sp.kron(_diff_op(nb, 2), I3, format='csr')
    Aop = (D1 @ Wop).tocsr()
    a0 = D1 @ w0
    Jop = (D2 @ Wop).tocsr()
    j0 = D2 @ w0
    na, nj = Aop.shape[0], Jop.shape[0]
    # ---- quadratic terms on u
    mu_a, mu_v, mu_f = job['w_acc_l2'], job['w_vel_l2'], job['w_fid']
    mu_j = job.get('w_jerk_l2', 0.0)
    Puu = 2.0 * mu_a * (Aop.T @ Aop) + 2.0 * mu_v * (Wop.T @ Wop)
    qu = 2.0 * mu_a * (Aop.T @ a0) + 2.0 * mu_v * (Wop.T @ w0)
    if mu_j > 0:
        # jerk L2 on a 2-tap-averaged velocity (x jerk_l2_smooth): same weight at 2-8 Hz, but the Nyquist gain of
        # the 3rd difference (64 -> 2.2 with 2 averages) no longer blows up the KKT condition number (plain Jop:
        # 25 -> 100 IPM iterations at w=1000)
        Js, js0 = Jop, j0
        for _ in range(int(job.get('jerk_l2_smooth', 2))):
            m_ = Js.shape[0] // 3
            if m_ < 2:
                break
            Sm = sp.kron(sp.diags([np.full(m_ - 1, 0.5), np.full(m_ - 1, 0.5)], [0, 1], shape=(m_ - 1, m_)),
                         sp.identity(3), format='csr')
            Js, js0 = (Sm @ Js).tocsr(), Sm @ js0
        Puu = Puu + 2.0 * mu_j * (Js.T @ Js)
        qu = qu + 2.0 * mu_j * (Js.T @ js0)
    bs = job.get('blur_s')
    if bs is not None and Fw >= 3 and np.any(bs[1:Fw - 1] > 0):
        # blur-aware term: rows t = 1..Fw-2, s_t * bhat_t . (0.5 (w_{t-1} + w_t) - r_t)   (px, linear in u)
        t_ = np.arange(1, Fw - 1)
        s_ = bs[t_]
        bd = job['blur_dir'][t_]                                   # (m,3) unit, QP px space
        cols = np.concatenate([3 * (t_ - 1)[:, None] + np.arange(3)[None, :], 3 * t_[:, None] + np.arange(3)[None, :]], 1)
        vals = np.concatenate([0.5 * s_[:, None] * bd, 0.5 * s_[:, None] * bd], 1)
        Bm = sp.csr_matrix((vals.ravel(), (np.repeat(np.arange(len(t_)), 6), cols.ravel())), shape=(len(t_), 3 * nb))
        Bop = (Bm @ Wop).tocsr()
        b0 = Bm @ w0 - s_ * np.einsum('ki,ki->k', bd, job['blur_r'][t_])
        Puu = Puu + 2.0 * (Bop.T @ Bop)
        qu = qu + 2.0 * (Bop.T @ b0)
    B = _jr_inv(job['phi0'])          # phi_new ≈ phi0 + B delta
    BtB = np.einsum('kji,kjl->kil', B, B)
    Bt_phi = np.einsum('kji,kj->ki', B, f * job['phi0'])
    diag_blocks = 2.0 * mu_f * BtB
    qu = qu + (2.0 * mu_f * Bt_phi).ravel()
    ri =(3 * np.arange(Fw))[:, None, None] + np.arange(3)[None, :, None]
    ci = (3 * np.arange(Fw))[:, None, None] + np.arange(3)[None, None, :]
    Pblk = sp.csr_matrix((diag_blocks.ravel(), (np.broadcast_to(ri, (Fw, 3, 3)).ravel(),
                                                np.broadcast_to(ci, (Fw, 3, 3)).ravel())), shape=(3 * Fw, 3 * Fw))
    Puu = (Puu + Pblk + 2.0 * job['prox'] * sp.identity(3 * Fw, format='csr')).tocsc()
    # ---- variable layout
    nu = 3 * Fw
    if zoom:   # zoom lives on knots every K frames, linearly interpolated (Zi: frames x knots)
        K = max(1, int(job.get('zoom_knot', 15)))
        kn = np.unique(np.concatenate([np.arange(0, Fw, K), [Fw - 1]]))
        nk = len(kn)
        jz = np.clip(np.searchsorted(kn, np.arange(Fw), side='right') - 1, 0, nk - 2)
        wz = (np.arange(Fw) - kn[jz]) / (kn[jz + 1] - kn[jz])
        Zi = sp.csr_matrix((np.concatenate([1.0 - wz, wz]), (np.tile(np.arange(Fw), 2), np.concatenate([jz, jz + 1]))),
                           shape=(Fw, nk))
    nz = nk if zoom else 0
    nsz = nk - 1 if zoom else 0
    off_u, off_z = 0, nu
    off_sa = nu + nz
    off_sj = off_sa + na
    off_sz = off_sj + nj
    off_sc = off_sz + nsz
    # horizon lock (stage 2): per active frame Q (quadratic part, 0..c) and T (linear part, >= 0) of the Huber
    # penalty on the excess of the (linearized) horizon angle beyond its allowed interval
    hz = job.get('hz')
    hz_act = np.flatnonzero(hz['w'] > 0) if hz is not None else np.zeros(0, dtype=np.int64)
    nh = len(hz_act)
    off_hq = off_sc + Fw
    off_ht = off_hq + nh
    nvar = off_ht + nh

    rows_A, rows_b = [], []

    def add(blocks, b):
        """blocks: list of (col_offset, sparse matrix) sharing the same row count."""
        m = len(b)
        mats = []
        for off, M in blocks:
            M = M.tocoo()
            mats.append((M.row, M.col + off, M.data))
        r = np.concatenate([a[0] for a in mats])
        c = np.concatenate([a[1] for a in mats])
        v = np.concatenate([a[2] for a in mats])
        rows_A.append(sp.csr_matrix((v, (r, c)), shape=(m, nvar)))
        rows_b.append(np.asarray(b, dtype=np.float64))

    Ia = sp.identity(na, format='csr')
    Ij = sp.identity(nj, format='csr')
    add([(off_u, Aop), (off_sa, -Ia)], -a0)
    add([(off_u, -Aop), (off_sa, -Ia)], a0)
    add([(off_u, Jop), (off_sj, -Ij)], -j0)
    add([(off_u, -Jop), (off_sj, -Ij)], j0)
    # trust region |u| <= f*tr, only on frames that carry crop rows (elsewhere the prox term keeps the step
    # small and the exact post-check + repair iterations catch anything the pruning missed)
    trf = job.get('tr_frames')
    tr_idx = np.arange(nu) if trf is None else (3 * np.asarray(trf)[:, None] + np.arange(3)[None, :]).ravel()
    tr_u = np.repeat(np.broadcast_to(np.asarray(job['tr'], dtype=np.float64), (Fw,)), 3)   # per-frame radius [rad]
    if len(tr_idx):
        Iu = sp.csr_matrix((np.ones(len(tr_idx)), (np.arange(len(tr_idx)), tr_idx)), shape=(len(tr_idx), nu))
        tr_px = f * tr_u[tr_idx]
        add([(off_u, Iu)], tr_px)
        add([(off_u, -Iu)], tr_px)
    q = np.zeros(nvar)
    q[off_u:off_u + nu] = qu
    q[off_sa:off_sa + na] = job['w_acc_l1']
    q[off_sj:off_sj + nj] = job['w_jerk_l1']
    q[off_sc:off_sc + Fw] = job['w_crop']
    Pz = None
    if zoom:
        zp0 = Zs * job['zeta0']
        zk0 = zp0[kn]
        zpmax = Zs * job['zeta_max']
        trz = Zs * job['trz']
        Dk1 = _diff_op(nk, 1)
        Isz = sp.identity(nsz, format='csr')
        add([(off_z, Dk1), (off_sz, -Isz)], -(Dk1 @ zk0))
        add([(off_z, -Dk1), (off_sz, -Isz)], Dk1 @ zk0)
        add([(off_sz, Isz)], Zs * job['zrate'] * np.diff(kn).astype(np.float64))
        Iz = sp.identity(nk, format='csr')
        zub = np.full(nk, zpmax)
        if job.get('zcap') is not None:        # per-knot zoom ceiling (horizon stage 2: the stage-1 zoom)
            zub = np.minimum(zub, Zs * np.asarray(job['zcap'], dtype=np.float64)[kn])
        add([(off_z, Iz)], np.maximum(np.minimum(zub - zk0, trz), 0.0))
        add([(off_z, -Iz)], np.maximum(np.minimum(zk0, trz), 0.0))
        q[off_z:off_z + nz] = job['w_zoom'] / Zs * np.asarray(Zi.sum(axis=0)).ravel()
        q[off_sz:off_sz + nsz] = job['w_zrate'] / Zs
        mu_zz = job['w_zacc'] / Zs ** 2 / float(K) ** 3
        if mu_zz > 0 and nk > 2:
            Dk2 = _diff_op(nk, 2)
            Pz = 2.0 * mu_zz * (Dk2.T @ Dk2)
            q[off_z:off_z + nz] += 2.0 * mu_zz * (Dk2.T @ (Dk2 @ zk0))
    # ---- crop rows: C u + cz z - s_k <= rhs
    ck = job['crop_k']
    nc = len(ck)
    if nc:
        cvec = job['crop_c'] / f          # per px of u
        r3 = np.repeat(np.arange(nc), 3)
        c3 = (off_u + 3 * ck[:, None] + np.arange(3)[None, :]).ravel()
        rr_ = [r3, np.arange(nc)]
        cc_ = [c3, off_sc + ck]
        vv_ = [cvec.ravel(), -np.ones(nc)]
        if zoom:
            rr_ += [np.arange(nc), np.arange(nc)]
            cc_ += [off_z + jz[ck], off_z + jz[ck] + 1]
            czs = job['crop_cz'] / Zs
            vv_ += [czs * (1.0 - wz[ck]), czs * wz[ck]]
        rows_A.append(sp.csr_matrix((np.concatenate(vv_), (np.concatenate(rr_), np.concatenate(cc_))), shape=(nc, nvar)))
        rows_b.append(job['crop_rhs'])
    Isc = sp.identity(Fw, format='csr')
    add([(off_sc, -Isc)], np.zeros(Fw))
    if nh:
        # e(u) = e0 + J·u (px; e0 = f·wrap(rho0 - centre)); excess beyond the half-width h: Q + T >= +-e(u) - h
        Jh = np.asarray(hz['J'], dtype=np.float64)[hz_act]                   # (nh,3) d rho / d delta
        e0 = np.asarray(hz['e0'], dtype=np.float64)[hz_act]
        hh = np.asarray(hz['h'], dtype=np.float64)[hz_act]
        ch = float(hz['c'])
        Ju = sp.csr_matrix((Jh.ravel(), (np.repeat(np.arange(nh), 3),
                                         (3 * hz_act[:, None] + np.arange(3)[None, :]).ravel())), shape=(nh, nu))
        Ih = sp.identity(nh, format='csr')
        add([(off_u, Ju), (off_hq, -Ih), (off_ht, -Ih)], hh - e0)
        add([(off_u, -Ju), (off_hq, -Ih), (off_ht, -Ih)], hh + e0)
        add([(off_hq, Ih)], np.full(nh, ch))
        add([(off_ht, -Ih)], np.zeros(nh))
        wh = np.asarray(hz['w'], dtype=np.float64)[hz_act]
        q[off_ht:off_ht + nh] = 2.0 * ch * wh

    A = sp.vstack(rows_A, format='csc')
    b = np.concatenate(rows_b)
    Pfull = sp.block_diag([Puu, Pz if Pz is not None else sp.csr_matrix((nz, nz)),
                           sp.csr_matrix((nvar - nu - nz, nvar - nu - nz))], format='csc') if nz else \
        sp.block_diag([Puu, sp.csr_matrix((nvar - nu, nvar - nu))], format='csc')
    if nh:
        ih = off_hq + np.arange(nh)
        Pfull = (Pfull + sp.csc_matrix((2.0 * wh, (ih, ih)), shape=(nvar, nvar))).tocsc()
    P = sp.triu(Pfull, format='csc')
    t1 = time.perf_counter()
    s = clarabel.DefaultSettings()
    s.verbose = False
    tol = job.get('qp_tol', 1e-7)
    s.tol_gap_abs = tol
    s.tol_gap_rel = tol
    s.tol_feas = tol
    s.equilibrate_enable = job.get('equilibrate', True)
    s.max_iter = 100
    sol = clarabel.DefaultSolver(P, q, A, b, [clarabel.NonnegativeConeT(A.shape[0])], s).solve()
    t2 = time.perf_counter()
    status = str(sol.status)
    ok = ('Solved' in status)
    xs = np.asarray(sol.x) if ok else np.zeros(nvar)
    delta = xs[off_u:off_u + nu].reshape(Fw, 3) / f
    dz = (Zi @ xs[off_z:off_z + nz]) / Zs if zoom else np.zeros(Fw)
    sc = xs[off_sc:off_sc + Fw]
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(dz)):
        ok = False
        delta, dz, sc = np.zeros((Fw, 3)), np.zeros(Fw), np.zeros(Fw)
    # enforce the trust region exactly (IPM tolerance)
    trk = tr_u.reshape(Fw, 3)
    delta = np.clip(delta, -trk, trk)
    return dict(delta=delta, dz=dz, slack=sc, status=status, ok=ok, iters=sol.iterations,
                build_s=t1 - t0, solve_s=t2 - t1, n_rows=A.shape[0], n_var=nvar, n_crop=nc)


# ============================================================================= driver


def _objective(V, zeta, R, f, Zs, prm, wfid, wvel, hz=None) -> dict:
    """Exact (nonlinear) value of the path objective terms for diagnostics. hz: horizon target (stage 2) or None."""
    wax = np.asarray(prm.axis_w, dtype=np.float64)
    w = f * wax * qlog(qmul(qconj(V[:-1]), V[1:]))
    a = np.diff(w, axis=0)
    j = np.diff(a, axis=0)
    phi = qlog(qmul(qconj(R), V))
    js = j
    for _ in range(int(prm.jerk_l2_smooth)):
        if len(js) > 1:
            js = 0.5 * (js[:-1] + js[1:])
    o = dict(acc_l1=prm.w_acc_l1 * np.abs(a).sum(), jerk_l1=prm.w_jerk_l1 * np.abs(j).sum(),
             acc_l2=prm.acc_l2() * (a * a).sum(), jerk_l2=prm.w_jerk_l2 * (js * js).sum(), vel_l2=wvel * (w * w).sum(), fid=wfid * ((f * phi) ** 2).sum())
    if hz is not None:
        ex = f * _horizon_excess(V, hz)
        c = f * np.deg2rad(prm.horizon_huber_deg)
        hub = np.where(ex <= c, ex * ex, 2.0 * c * ex - c * c)
        o['horizon'] = float((hz['w'] * hub).sum())
    zp = Zs * zeta
    o['zoom'] = prm.w_zoom / Zs * zp.sum() + prm.w_zoom_rate_l1 / Zs * np.abs(np.diff(zp)).sum() + \
        prm.w_zoom_acc_l2 / Zs ** 2 * (np.diff(zp, 2) ** 2).sum()
    o['total'] = float(sum(o.values()))
    return o


def _lowpass_path(R: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """Zero-phase low-pass of an orientation sequence (F,4): the body-frame integrated rotation P is filtered and the
    camera is rotated back by its high-pass part, R_lp = R * Exp(-(P - LP(P))) (exact to first order in the removed
    part; only used as the SQP's starting point)."""
    from scipy.signal import butter, sosfiltfilt
    F = len(R)
    if F < 16 or fc <= 0 or fc >= 0.45 * fs:
        return R.copy()
    w = qlog(qmul(qconj(R[:-1]), R[1:]))
    P = np.concatenate([np.zeros((1, 3)), np.cumsum(w, 0)])
    sos = butter(2, fc, 'lowpass', fs=fs, output='sos')
    h = P - sosfiltfilt(sos, P, axis=0, padtype='odd', padlen=min(F - 1, int(3 * fs / fc)))
    return qnormalize(qmul(R, qexp(-h)))


def _warm_start(R, groups, fps, prm, q_cam_fn, ft, fx0, pts, out_w, out_h, lens, readout, H, bnd):
    """SQP start: per frame the largest blend alpha (of warm_alphas, else 0) from the camera R toward its low-passed
    path whose output border maps inside the crop (exact RS-aware mapping), eroded then box-averaged over
    +-warm_erode_s (the average of an eroded sequence never exceeds the original, so a feasible alpha stays feasible
    up to the blend's non-monotonicity, which the SQP's exact check and repair iterations absorb).
    Why: the trust radii sum to ~6 deg, so an SQP started at the camera cannot move farther than that from it; in
    fast moves the path sat on that cap and copied the camera's 2-8 Hz shake. From a smooth start the cap is
    relative to a smooth path."""
    F = len(R)
    Rl = R.copy()
    for (a, b) in groups:
        if b - a >= 16:
            Rl[a:b] = _lowpass_path(R[a:b], fps, prm.warm_fc)
    d = qlog(qmul(qconj(R), Rl))                           # (F,3) camera -> low-passed
    fo = np.full(F, fx0)
    alpha = np.zeros(F)
    for al in sorted(prm.warm_alphas):
        V = qmul(R, qexp(al * d))
        p = map_output_points(q_cam_fn, ft, V, fo, pts, out_w, out_h, lens, readout, H, prm.rs_iters)
        lo, hi = bnd(p)
        ok = (np.maximum(lo - p, p - hi).max(axis=-1).max(axis=1) <= -1.0)
        alpha = np.where(ok, al, alpha)
    n = max(1, int(round(prm.warm_erode_s * fps)))
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    a_s = np.zeros(F)
    for (a, b) in groups:
        if b - a < 1:
            continue
        e = minimum_filter1d(alpha[a:b], 2 * n + 1, mode='nearest')
        a_s[a:b] = np.minimum(uniform_filter1d(e, 2 * n + 1, mode='nearest'), alpha[a:b])
    V = qnormalize(qmul(R, qexp(a_s[:, None] * d)))
    dev = np.rad2deg(np.linalg.norm(d, axis=1))
    return V, dict(mean_alpha=float(a_s.mean()), frac_full=float((a_s >= 0.999).mean()),
                   lp_dev_p99_deg=float(np.percentile(dev, 99)) if F else 0.0)


def _groups(tel, frame_ids: np.ndarray) -> list:
    """Split positions 0..F-1 into runs that lie in the same telemetry segment."""
    F = len(frame_ids)
    seg_id = np.zeros(F, dtype=np.int64)
    segs = getattr(tel, 'segments', None) or []
    for i, (a, b) in enumerate(segs):
        seg_id[(frame_ids >= a) & (frame_ids <= b)] = i
    cuts = np.flatnonzero(np.diff(seg_id) != 0) + 1
    edges = np.concatenate([[0], cuts, [F]])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def _windows(a: int, b: int, core: int, ov: int) -> list:
    """Overlapping windows over [a,b): list of (win_lo, win_hi, core_lo, core_hi)."""
    L = b - a
    if core <= 0 or L <= core + 2 * ov:
        return [(a, b, a, b)]
    n = int(math.ceil(L / core))
    edges = np.round(np.linspace(a, b, n + 1)).astype(int)
    return [(max(a, edges[i] - ov), min(b, edges[i + 1] + ov), int(edges[i]), int(edges[i + 1])) for i in range(n)]


# ============================================================================= horizon lock v2 (module docstring)


def _wrap(a):
    """Angle(s) wrapped to [-pi, pi)."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _smoothstep(x, lo: float, hi: float) -> np.ndarray:
    """0 at/below lo, 1 at/above hi, C1 cubic in between."""
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _erode_ramp(x: np.ndarray, n: int, groups) -> np.ndarray:
    """Per group: min-filter over +-n, then a Gaussian (sigma n/3, truncated at +-n) -- never above x, since every
    output is a weighted average of minima over windows that contain the centre. A drop of x is anticipated n frames
    early and ramped smoothly (C-infinity; at sigma = n/3 a 1-s ramp leaves < 1e-3 of a step above 2 Hz)."""
    out = np.array(x, dtype=np.float64, copy=True)
    if n <= 0:
        return out
    from scipy.ndimage import gaussian_filter1d, minimum_filter1d
    for (a, b) in groups:
        if b - a < 1:
            continue
        e = minimum_filter1d(out[a:b], 2 * n + 1, mode='nearest')
        e = gaussian_filter1d(e, n / 3.0, mode='nearest', truncate=3.0)
        out[a:b] = np.minimum(e, out[a:b])
    return out


def gravity_world(tel, t: np.ndarray) -> Optional[np.ndarray]:
    """Gravity 'down' (unit) in the IMU world frame at video times t, (N,3); None without a gravity-referenced
    attitude. tel.gravity_q maps camera -> a gravity-aligned world (z down): O3 / O4 Pro fused attitude (the same
    array as imu_q), OA4 per-frame cam_quat re-levelling of the 1 kHz stream (a constant world rotation)."""
    t = np.asarray(t, dtype=np.float64)
    if getattr(tel, 'gravity_q', None) is None:
        return None
    if tel.gravity_q is tel.imu_q:
        return np.tile(np.array([0.0, 0.0, 1.0]), (len(t), 1))
    qi = slerp_series(tel.imu_t, tel.imu_q, t)
    qg = slerp_series(tel.imu_t, tel.gravity_q, t)
    G = quat_to_mat(qmul(qg, qconj(qi)))                   # imu-world -> gravity-world
    return np.einsum('kji,j->ki', G, np.array([0.0, 0.0, 1.0]))


def horizon_angles(V: np.ndarray, g_w: np.ndarray):
    """For orientations V (N,4) camera->world and gravity 'down' g_w (N,3) in that world: (rho, elev, g) with
    g = Vᵀ g_w in camera coordinates (x right, y down, z forward), rho = atan2(g_x, g_y) the horizon angle [rad]
    (0 = level; the horizon line is drawn at -rho in the image, y down) and elev = asin(g_z) the optical-axis
    elevation below the horizon [rad] (+90 deg = looking straight down)."""
    g = np.einsum('kji,kj->ki', quat_to_mat(V), g_w)
    return np.arctan2(g[:, 0], g[:, 1]), np.arcsin(np.clip(g[:, 2], -1.0, 1.0)), g


def horizon_jacobian(g: np.ndarray, n2_min: float = 0.04) -> np.ndarray:
    """d rho / d delta (N,3) for V -> V·Exp(delta), g = Vᵀ g_w: (-g_z g_x, -g_z g_y, n²)/n², n² = g_x² + g_y²
    (floored at n2_min, i.e. |elevation| <= 78 deg, where the lock has faded out anyway). A roll about the optical
    axis moves rho one-to-one; the pan/tilt coupling grows like tan(elevation)."""
    g = np.asarray(g, dtype=np.float64)
    n2 = np.maximum(g[:, 0] ** 2 + g[:, 1] ** 2, n2_min)
    return np.stack([-g[:, 2] * g[:, 0] / n2, -g[:, 2] * g[:, 1] / n2, np.ones(len(g))], 1)


def horizon_availability(R: np.ndarray, g_w: np.ndarray, fps: float, groups, prm: SmoothParams):
    """Per-frame lock availability a in [0,1] of the camera path R: steepness x inverted attitude x tilt rate, each a
    smoothstep between its (full, none) thresholds, then eroded + ramped over +-horizon_fade_s. Returns (a, diag)."""
    F = len(R)
    rho, elev, _ = horizon_angles(R, g_w)
    tilt = np.zeros(F)
    if F > 1:
        ww = qlog(qmul(R[1:], qconj(R[:-1]))) * fps                     # world-frame angular velocity [rad/s]
        gm = g_w[1:] + g_w[:-1]
        gm = gm / np.maximum(np.linalg.norm(gm, axis=1, keepdims=True), 1e-12)
        tl = np.linalg.norm(ww - np.einsum('ki,ki->k', ww, gm)[:, None] * gm, axis=1)   # rate minus yaw about g
        for (a, b) in groups[1:]:
            if a > 0:
                tl[a - 1] = 0.0                                             # no rate across a segment cut
        tilt[:-1] = tl
        tilt[1:] = np.maximum(tilt[1:], tl)
    s0, s1 = prm.horizon_steep_deg
    i0, i1 = prm.horizon_inverted_deg
    r0, r1 = prm.horizon_rate_dps
    a_steep = 1.0 - _smoothstep(np.degrees(np.abs(elev)), s0, s1)
    a_inv = 1.0 - _smoothstep(np.degrees(np.abs(rho)), i0, i1)
    a_rate = 1.0 - _smoothstep(np.degrees(tilt), r0, r1)
    a0 = a_steep * a_inv * a_rate
    a = _erode_ramp(a0, int(round(float(prm.horizon_fade_s) * fps)), groups)
    diag = dict(frac_steep=float((a_steep < 0.5).mean()), frac_inverted=float((a_inv < 0.5).mean()),
                frac_fast=float((a_rate < 0.5).mean()), avail_mean=float(a.mean()),
                frac_available=float((a >= 0.99).mean()), frac_off=float((a <= 0.01).mean()),
                tilt_rate_p99_dps=float(np.degrees(np.percentile(tilt, 99))) if F else 0.0)
    return a, diag


def _crop_ok(q_cam_fn, ft, V, fo, pts, out_w, out_h, lens, readout, H, iters, bnd, idx, extra, chunk=4000):
    """Per frame: every output-border sample maps inside [lo+extra, hi-extra] (exact RS-aware mapping); the crop box
    (lo, hi) = bnd(p, idx) of the frames idx (of frame_ids) that ft / V / fo belong to."""
    ok = np.empty(len(ft), dtype=bool)
    idx = np.asarray(idx)
    for a in range(0, len(ft), chunk):
        b = min(len(ft), a + chunk)
        p = map_output_points(q_cam_fn, ft[a:b], V[a:b], fo[a:b], pts, out_w, out_h, lens, readout, H, iters)
        lo, hi = bnd(p, idx[a:b])
        ok[a:b] = np.maximum(lo - p, p - hi).max(axis=-1).max(axis=1) <= -extra
    return ok


def _horizon_target(V1, zeta1, R, g_w, groups, fps, prm, s_lock, q_cam_fn, ft, fx0, out_w, out_h, lens, readout, H,
                    bnd):
    """Stage-2 start and horizon target from the stage-1 path V1 (module docstring, steps 1 and 3).
    Returns (V2, hz or None, diag); hz = dict(w, centre, half, a, g_w) (angles in rad)."""
    F = len(V1)
    a, diag = horizon_availability(R, g_w, fps, groups, prm)
    rho1, _, _ = horizon_angles(V1, g_w)
    L = math.radians(max(0.0, float(prm.roll_limit_deg)))
    excess = np.sign(rho1) * np.maximum(np.abs(rho1) - L, 0.0)
    D = -a * s_lock * excess                                          # ideal roll change (rad)
    beta_raw = np.ones(F)
    need = np.flatnonzero(np.abs(D) > 1e-5)
    fo = fx0 * np.exp(zeta1)
    pts = border_samples(out_w, out_h, 5, 5)
    extra = float(prm.horizon_margin_px)
    ez = np.array([0.0, 0.0, 1.0])

    def feasible(idx, frac):
        Vk = qmul(V1[idx], qexp((frac * D[idx])[:, None] * ez))
        return _crop_ok(q_cam_fn, ft[idx], Vk, fo[idx], pts, out_w, out_h, lens, readout, H, prm.rs_iters, bnd, idx,
                        extra)

    if len(need):
        full = feasible(need, np.ones(len(need)))
        rest = need[~full]
        blo, bhi = np.zeros(len(rest)), np.ones(len(rest))
        for _ in range(max(1, int(math.ceil(math.log2(max(2, int(prm.horizon_fracs))))))):
            mid = 0.5 * (blo + bhi)
            ok = feasible(rest, mid) if len(rest) else np.zeros(0, dtype=bool)
            blo = np.where(ok, mid, blo)
            bhi = np.where(ok, bhi, mid)
        beta_raw[rest] = blo
    # the leveled FRACTION is eroded + ramped over +-horizon_crop_fade_s: smooth, and never above the per-frame
    # feasible fraction. (Tried 2026-09-29: eroding the absolute deficit instead, and a 0.6 s fade: the deficit form
    # re-enters no faster after a loop -- the smoother itself hugs the crop right after it -- and 0.6 s put the
    # camera's shake into the roll of a crop-limited 35 deg bank: 2-8 Hz 0.0006 -> 0.0037 deg, x6.)
    beta = _erode_ramp(beta_raw, int(round(float(prm.horizon_crop_fade_s) * fps)), groups)
    tau = rho1 + beta * D
    lo_i = np.minimum(-L, tau)
    hi_i = np.maximum(L, tau)
    w = float(prm.w_horizon) * np.where(a >= 0.02, a, 0.0)          # (a < 2 %: effectively off; no QP variables)
    V2 = qnormalize(qmul(V1, qexp((beta * D)[:, None] * ez)))
    act = a >= 0.99
    diag.update(strength=s_lock, roll_limit_deg=float(prm.roll_limit_deg),
                beta_mean_active=float(beta[act].mean()) if act.any() else None,
                frac_full_level=float((beta[act & (np.abs(D) > 1e-5)] >= 0.99).mean())
                if (act & (np.abs(D) > 1e-5)).any() else None,
                stage1_abs_roll_deg=_pcts(np.degrees(np.abs(rho1[act]))),
                target_abs_roll_deg=_pcts(np.degrees(np.abs(tau[act]))))
    if not (w > 0).any():
        return V1, None, diag
    return V2, dict(w=w, centre=0.5 * (lo_i + hi_i), half=0.5 * (hi_i - lo_i), a=a, g_w=g_w, tau=tau, beta=beta), diag


def _pcts(x, ps=(50, 90, 95, 99)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return [round(float(np.percentile(x, p_)), 3) for p_ in ps] if len(x) else None


def _horizon_excess(V, hz) -> np.ndarray:
    """Per frame: |horizon angle| beyond the allowed interval [rad] (0 inside)."""
    rho, _, _ = horizon_angles(V, hz['g_w'])
    return np.maximum(np.abs(_wrap(rho - hz['centre'])) - hz['half'], 0.0)


def _horizon_stats(V, hz, g_w) -> dict:
    rho, _, _ = horizon_angles(V, g_w)
    act = hz['a'] >= 0.99
    ex = np.degrees(_horizon_excess(V, hz))
    return dict(final_abs_roll_deg=_pcts(np.degrees(np.abs(rho[act]))),
                excess_deg=_pcts(ex[act]), max_excess_deg=float(ex[act].max()) if act.any() else 0.0)


def _blur_weights(tel, q_cam_fn, frame_ids, ft, fps, f, out_w, out_h, prm) -> Optional[dict]:
    """Per-frame inputs of the blur-aware term (SmoothParams.w_blur): s_t = sqrt(w_blur*beta_t)*e_t/T_f, the camera's
    exposure-averaged rate r_t (QP units: output px/frame, axis-weighted) and its direction. None when off."""
    if not prm.w_blur or prm.w_blur <= 0:
        return None
    ex = np.asarray(getattr(tel, 'exposure_s', np.zeros(0)), np.float64)
    if ex.ndim != 1 or len(ex) != tel.n_frames:
        return None
    e = np.clip(np.nan_to_num(ex[frame_ids], nan=0.0), 0.0, 0.1)
    eh = np.maximum(e, 2e-4)
    qa = _q_at(q_cam_fn, ft - 0.5 * eh)
    qb = _q_at(q_cam_fn, ft + 0.5 * eh)
    om = qlog(qmul(qconj(qa), qb)) / eh[:, None]                 # rad/s, camera frame, averaged over the exposure
    Tf = 1.0 / float(fps)
    wax = np.asarray(prm.axis_w, np.float64)
    r = f * wax[None, :] * om * Tf                               # px/frame (QP velocity units)
    rho = math.sqrt((out_w ** 2 + out_h ** 2) / 12.0)
    blen = e * np.sqrt((f * om[:, 0]) ** 2 + (f * om[:, 1]) ** 2 + (rho * om[:, 2]) ** 2) * (1920.0 / out_w)
    b0, b1 = float(prm.blur_b0_px), max(float(prm.blur_b1_px), float(prm.blur_b0_px) + 1e-6)
    beta = np.clip((blen - b0) / (b1 - b0), 0.0, 1.0)
    nrm = np.linalg.norm(r, axis=1)
    beta = np.where(nrm > 1e-9, beta, 0.0)
    d = r / np.maximum(nrm, 1e-9)[:, None]
    s = np.sqrt(float(prm.w_blur) * beta) * (e / Tf)
    return dict(s=s, dir=d, r=r, beta=beta, streak_px=blen, frac_active=float((beta > 0).mean()),
                beta_mean=float(beta.mean()), streak_p95_px=float(np.percentile(blen, 95)) if len(blen) else 0.0)


def optimize_path(tel, q_cam_fn: Callable[[np.ndarray], np.ndarray], frame_ids: np.ndarray, out_w: int, out_h: int,
                  params: Optional[SmoothParams] = None, tm=None, return_info: bool = False):
    """Crop-constrained virtual camera path.

    Returns (virt_q (F,4) camera->world, out_fx (F,)) for the frames `frame_ids` (consecutive frames of
    tel; optimization never crosses a tel.segments boundary). With return_info=True a third value, a
    diagnostics dict, is returned. `tm` (TimeModel, optional) supplies readout/focal overrides."""
    prm = params or SmoothParams()
    if clarabel is None:  # pragma: no cover
        raise ImportError('clarabel is required for optimize_path')
    T0 = time.perf_counter()
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    F = len(frame_ids)
    lens = prm.lens or tel.lens
    readout = prm.readout_s if prm.readout_s is not None else tel.readout_s
    if tm is not None:
        if prm.readout_s is None and getattr(tm, 'readout_s', None) is not None:
            readout = float(tm.readout_s)
        if prm.lens is None and getattr(tm, 'focal_scale', 1.0) != 1.0:
            s_ = float(tm.focal_scale)
            lens = Lens(lens.model, lens.fx * s_, lens.fy * s_, lens.cx, lens.cy, np.array(lens.k), lens.width, lens.height)
    W, H = int(tel.width), int(tel.height)
    fx0 = float(prm.min_out_fx) if prm.min_out_fx and prm.min_out_fx > 0 else \
        fx_for_crop_area(lens, W, H, out_w, out_h, prm.crop_area)
    fxmax = float(prm.max_out_fx) if prm.max_out_fx else 1.5 * fx0
    zoom = bool(prm.allow_zoom) and fxmax > fx0 * (1 + 1e-9)
    zeta_max = math.log(fxmax / fx0) if zoom else 0.0
    fps = float(tel.fps) if tel.fps else 59.94
    ft = np.asarray(tel.frame_t, dtype=np.float64)[frame_ids]
    R = qfix_sign(_q_at(q_cam_fn, ft))
    info = dict(F=F, min_out_fx=fx0, max_out_fx=fxmax, iters=[], zoom=zoom)
    if F < 4:
        virt = R.copy()
        out = (virt, np.full(F, fx0)) + ((info,) if return_info else ())
        return out
    # horizon lock v2 (module docstring): gravity 'down' in the camera/imu world, per frame
    s_lock = prm.lock_strength()
    g_w = None
    if s_lock > 0 and getattr(tel, 'gravity_q', None) is not None:
        try:
            g_w = gravity_world(tel, ft)
        except Exception as e:  # pragma: no cover - never crash on optional horizon data
            info['horizon_error'] = repr(e)
            g_w = None
    info['horizon_active'] = g_w is not None
    f = fx0
    Zs = 0.5 * math.hypot(out_w, out_h)
    pts = border_samples(out_w, out_h, prm.samples_long, prm.samples_short)
    layout = _edge_layout(out_w, out_h, prm.samples_long, prm.samples_short)
    m = float(prm.margin_px)
    lo = np.array([m, m])
    hi = np.array([W - 1 - m, H - 1 - m])
    ext = None
    if prm.crop_extend is not None:                        # full-frame fill: the crop box grows per frame / edge
        ext = np.maximum(np.asarray(prm.crop_extend, np.float64), 0.0)
        ext = ext.reshape(F, 4, -1)
        info['crop_extend_mean_px'] = [float(v) for v in ext.mean(axis=(0, 2))]

    def bounds(p, idx=None):
        """Crop box (lo, hi) of every border sample p (F,S,2) [source px], broadcastable to p. idx: the frames (of
        frame_ids) p belongs to when p covers a subset of them (None = all frames, in order)."""
        if ext is None:
            return lo, hi
        e = ext if idx is None else ext[np.asarray(idx)]
        n = e.shape[2]
        rr = np.arange(p.shape[0])[:, None]

        def prof(edge, t):
            if n == 1:
                return np.broadcast_to(e[:, edge, 0][:, None], t.shape)
            g = np.clip(t, 0.0, 1.0) * (n - 1)
            i0 = np.minimum(np.floor(g).astype(np.int64), n - 2)
            return np.minimum(e[rr, edge, i0], e[rr, edge, i0 + 1])
        tx, ty = p[..., 0] / (W - 1.0), p[..., 1] / (H - 1.0)
        return (np.stack([m - prof(0, ty), m - prof(1, tx)], -1),
                np.stack([W - 1 - m + prof(2, ty), H - 1 - m + prof(3, tx)], -1))
    wfid, wvel, wacc = prm.fidelity(), prm.vel_l2(), prm.acc_l2()
    groups = _groups(tel, frame_ids)
    workers = prm.workers or min(8, os.cpu_count() or 4)
    wins = []
    for (a, b) in groups:
        if b - a < 4:
            continue
        core = prm.window if prm.window >= 0 else max(1500, int(math.ceil((b - a) / workers)))
        wins += [(a, b) + w for w in _windows(a, b, core, prm.overlap)]

    blur = _blur_weights(tel, q_cam_fn, frame_ids, ft, fps, f, out_w, out_h, prm)
    if blur is not None:
        info['blur'] = {k_: v_ for k_, v_ in blur.items() if np.isscalar(v_)}
    trs = list(prm.trust_rad)
    trzs = list(prm.trust_zoom)
    n_main = int(prm.sqp_iters)
    total_iters = n_main + int(prm.repair_iters)
    n_stages = 2 if g_w is not None else 1
    c_hub = f * math.radians(float(prm.horizon_huber_deg))

    def _check():
        if prm.cancel is not None and prm.cancel():
            raise InterruptedError('path optimisation cancelled')

    def sqp(V, zeta, hz, stage, it0=0, zcap=None):
        """SQP iterations from (V, zeta); hz: horizon target (stage 2) or None; it0 > 0 enters the trust-radius
        schedule later (a start that is already close); zcap: per-frame log-zoom ceiling or None.
        Returns (V, zeta, p)."""
        it = it0
        rec = None
        while True:
            _check()
            if prm.tick is not None:
                prm.tick((stage + (min(it, total_iters) - it0) / max(total_iters - it0, 1)) / n_stages)
            Ti = time.perf_counter()
            # ---------------- exact mapping + linearization of the current iterate (global, vectorized)
            fo = fx0 * np.exp(zeta)
            p_f, Gd_f, Gz_f = map_output_points(q_cam_fn, ft, V, fo, pts, out_w, out_h, lens, readout, H,
                                                prm.rs_iters, jac=True)
            p_a, Gd_a, Gz_a = map_output_points(q_cam_fn, ft, V, fo, _adaptive_points(p_f, layout, out_w, out_h),
                                                out_w, out_h, lens, readout, H, prm.rs_iters, jac=True)
            p = np.concatenate([p_f, p_a], 1)
            lo_p, hi_p = bounds(p)
            viol = np.maximum(lo_p - p, p - hi_p).max(axis=-1).max(axis=1)
            if rec is not None:
                rec.update(max_viol_px=float(viol.max()), viol_frame=int(viol.argmax()), n_viol=int((viol > 0.05).sum()))
                if prm.verbose:
                    print('[smooth]', rec, flush=True)
            if it >= total_iters or (it >= n_main and viol.max() <= prm.viol_tol_px):
                break
            tr = trs[min(it, len(trs) - 1)]
            trz = trzs[min(it, len(trzs) - 1)]
            Gd = np.concatenate([Gd_f, Gd_a], 1)
            Gz = np.concatenate([Gz_f, Gz_a], 1)
            phi0 = qlog(qmul(qconj(R), V))
            xrel = qlog(qmul(qconj(V[:-1]), V[1:]))
            hz_e0 = hz_J = None
            if hz is not None:    # horizon angle of the iterate and its exact first-order sensitivity
                rho0, _, g0 = horizon_angles(V, g_w)
                hz_J = horizon_jacobian(g0)
                hz_e0 = f * _wrap(rho0 - hz['centre'])
            # crop rows (4 bounds per sample), pruned by exact reach under the trust region
            slack = np.stack([p[..., 0] - lo_p[..., 0], hi_p[..., 0] - p[..., 0], p[..., 1] - lo_p[..., 1],
                              hi_p[..., 1] - p[..., 1]], -1)                                                   # (F,S,4)
            coef = np.stack([-Gd[..., 0, :], Gd[..., 0, :], -Gd[..., 1, :], Gd[..., 1, :]], -2)            # (F,S,4,3)
            coefz = np.stack([-Gz[..., 0], Gz[..., 0], -Gz[..., 1], Gz[..., 1]], -1)                       # (F,S,4)
            # per-frame trust radius: near the crop border the schedule `tr` (linearization accuracy of the crop
            # rows); far from it a fraction of the first-order distance to the border (no crop row of the frame can
            # become active, so none is needed), capped by trust_far. The radii never sum to a hard cap on
            # |V - camera|.
            csum = np.abs(coef).sum(-1)                                                                    # (F,S,4)
            zr = (trz * np.abs(coefz) if zoom else 0.0)
            tfar = prm.trust_far[min(it, len(prm.trust_far) - 1)] if prm.trust_far else tr
            r_free = ((slack - 2.0 - zr) / np.maximum(csum, 1e-9)).min(axis=(1, 2))                         # (F,)
            tr_k = np.clip(float(prm.trust_frac) * r_free, tr, max(tr, tfar))
            reach = tr_k[:, None, None] * csum + zr + 2.0
            kk, ss, bb = np.nonzero(slack < reach)
            crop_k_all = kk
            crop_c_all = coef[kk, ss, bb]
            crop_cz_all = coefz[kk, ss, bb]
            crop_rhs_all = slack[kk, ss, bb]
            t_lin = time.perf_counter() - Ti
            # ---------------- solve windows in parallel (Clarabel releases the GIL)
            jobs = []
            for (ga, gb, wa, wb, ca, cb) in wins:
                sel = (crop_k_all >= wa) & (crop_k_all < wb)
                jobs.append(dict(
                    n=wb - wa, f=f, Zs=Zs, axis_w=prm.axis_w, zoom=zoom, x=xrel[wa:wb - 1], phi0=phi0[wa:wb],
                    zeta0=zeta[wa:wb], zeta_max=zeta_max, trz=trz, zrate=prm.max_zoom_rate / fps, tr=tr_k[wa:wb],
                    zcap=None if zcap is None else zcap[wa:wb],
                    hz=None if hz is None else dict(w=hz['w'][wa:wb], e0=hz_e0[wa:wb], h=f * hz['half'][wa:wb],
                                                    J=hz_J[wa:wb], c=c_hub),
                    w_acc_l1=prm.w_acc_l1, w_jerk_l1=prm.w_jerk_l1, w_acc_l2=wacc, w_vel_l2=wvel,
                    w_jerk_l2=prm.w_jerk_l2, jerk_l2_smooth=prm.jerk_l2_smooth,
                    w_fid=wfid, w_zoom=prm.w_zoom,
                    w_zrate=prm.w_zoom_rate_l1, w_zacc=prm.w_zoom_acc_l2, w_crop=prm.w_crop_slack, prox=prm.prox,
                    zoom_knot=prm.zoom_knot, crop_k=crop_k_all[sel] - wa, crop_c=crop_c_all[sel],
                    crop_cz=crop_cz_all[sel], crop_rhs=crop_rhs_all[sel],
                    blur_s=None if blur is None else blur['s'][wa:wb],
                    blur_dir=None if blur is None else blur['dir'][wa:wb],
                    blur_r=None if blur is None else blur['r'][wa:wb]))
            Ts = time.perf_counter()
            if len(jobs) == 1 or workers <= 1:
                res = []
                for j in jobs:
                    _check()
                    res.append(_solve_window(j))
            else:
                from concurrent.futures import wait as _wait
                ex = ThreadPoolExecutor(max_workers=min(workers, len(jobs)))
                futs = [ex.submit(_solve_window, j) for j in jobs]
                try:
                    while True:
                        done, pend = _wait(futs, timeout=0.25)
                        if not pend:
                            break
                        _check()
                    res = [f_.result() for f_ in futs]
                finally:
                    ex.shutdown(wait=False, cancel_futures=True)      # a cancel does not wait for running solves
            t_solve = time.perf_counter() - Ts
            # ---------------- blend windows (linear cross-fade over the middle of each overlap)
            delta = np.zeros((F, 3))
            dz = np.zeros(F)
            wsum = np.zeros(F)
            slack_c = np.zeros(F)
            half = max(1, prm.overlap // 2)
            for (ga, gb, wa, wb, ca, cb), r in zip(wins, res):
                idx = np.arange(wa, wb)
                wgt = np.ones(wb - wa)
                if ca > ga:   # ramp up across [ca - half/2, ca + half/2)
                    wgt = np.minimum(wgt, np.clip((idx - (ca - half / 2)) / half, 0, 1))
                if cb < gb:
                    wgt = np.minimum(wgt, np.clip(((cb + half / 2) - idx) / half, 0, 1))
                delta[wa:wb] += wgt[:, None] * r['delta']
                dz[wa:wb] += wgt * r['dz']
                slack_c[wa:wb] = np.maximum(slack_c[wa:wb], np.where(wgt > 0, r['slack'], 0))
                wsum[wa:wb] += wgt
            good = wsum > 0
            delta[good] /= wsum[good, None]
            dz[good] /= wsum[good]
            V = qnormalize(qmul(V, qexp(delta)))
            zeta = np.clip(zeta + dz, 0.0, zeta_max)
            sat = np.zeros(F, dtype=bool)                   # frames whose step hit their trust radius (not converged)
            for (ga, gb, wa, wb, ca, cb), r in zip(wins, res):
                sat[ca:cb] |= (np.abs(r['delta'][ca - wa:cb - wa]) >= 0.999 * tr_k[ca:cb, None]).any(1)
            rec = dict(it=it, tr=tr, tr_far_frac=float((tr_k > tr * (1 + 1e-9)).mean()), n_sat=int(sat.sum()),
                       obj=round(_objective(V, zeta, R, f, Zs, prm, wfid, wvel, hz)['total'], 2),
                       lin_s=round(t_lin, 3), solve_s=round(t_solve, 3), n_crop_rows=int(len(crop_k_all)),
                       max_step_deg=float(np.rad2deg(np.abs(delta).max())),
                       step_frame=int(np.abs(delta).max(1).argmax()), max_slack=float(slack_c.max()),
                       status=sorted(set(r['status'] for r in res)), qp_iters=int(max(r['iters'] for r in res)),
                       qp_solve_s=round(max(r['solve_s'] for r in res), 3), total_s=0.0)
            if stage:
                rec['stage'] = stage
            info['iters'].append(rec)
            rec['total_s'] = round(time.perf_counter() - Ti, 3)
            it += 1
        return V, zeta, p

    V = R.copy()
    if prm.warm_start:
        V, info['warm'] = _warm_start(R, groups, fps, prm, q_cam_fn, ft, fx0, pts, out_w, out_h, lens, readout, H,
                                      bounds)
    zeta = np.zeros(F)
    V, zeta, p = sqp(V, zeta, None, 0)
    if g_w is not None:
        # ---------------- horizon lock stage 2: a smooth, crop-feasible leveled target from the stage-1 path
        Th = time.perf_counter()
        V2, hz, hinfo = _horizon_target(V, zeta, R, g_w, groups, fps, prm, s_lock, q_cam_fn, ft, fx0, out_w, out_h,
                                        lens, readout, H, bounds)
        hinfo['target_s'] = round(time.perf_counter() - Th, 3)
        if hz is not None:
            Th = time.perf_counter()
            zcap = None if (prm.horizon_zoom or not zoom) else zeta + 1e-6     # leveling never zooms in further
            V, zeta, p = sqp(V2, zeta, hz, 1, it0=max(0, n_main - int(prm.horizon_stage2_iters)), zcap=zcap)
            hinfo['stage2_s'] = round(time.perf_counter() - Th, 3)
            hinfo.update(_horizon_stats(V, hz, g_w))
        info['horizon'] = hinfo
    out_fx = fx0 * np.exp(zeta)
    virt_q = qfix_sign(V)
    # diagnostics
    phi = qlog(qmul(qconj(R), V))
    fo = out_fx
    lo_p, hi_p = bounds(p)
    slack_fin = np.minimum(p - lo_p, hi_p - p).min(-1).min(axis=1)     # p: exact mapping of the final iterate
    info.update(dict(
        runtime_s=time.perf_counter() - T0,
        max_violation_px=float(-slack_fin.min()) if len(slack_fin) else 0.0,
        n_frames_violating=int((slack_fin < -0.05).sum()),
        frac_binding=float((slack_fin < 0.5).mean()),
        n_binding=int((slack_fin < 0.5).sum()),
        max_phi_deg=float(np.rad2deg(np.linalg.norm(phi, axis=1).max())),
        zoom_changes=int((np.abs(np.diff(zeta)) > 1e-4).sum()),
        max_zoom=float(out_fx.max() / fx0),
        windows=len(wins), fidelity=wfid, slack_min_px=slack_fin,
    ))
    if return_info:
        return virt_q, out_fx, info
    return virt_q, out_fx
