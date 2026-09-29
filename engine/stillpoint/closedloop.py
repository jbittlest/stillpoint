"""Fold measured preview residuals back into the camera orientation (WP-D closed loop, ENGINE_SPEC.md §4.5).

Model (see residual.py for the measurement conventions)
-------------------------------------------------------
Preview k was rendered with camera estimate R_est(t) and virtual orientation V_k. If the true camera is
R_true = R_est * Exp(c) (c: small camera-frame error), the preview content looks like a camera with
orientation W_k = R_true R_est^T V_k = V_k * Exp(eps_k) with eps_k = M_k^T c_k, M_k = R_est_k^T V_k
(the virtual->camera rotation). residual.measure_residuals gives per pair
    rel_err[k] = log(R_exp^T R_meas) ~= eps_{k+1} - eps_k.
So eps = integral(rel_err) (up to a constant), and the correction to RIGHT-multiply onto q_cam is
    correction(t) = Exp( M(t) * HP(eps)(t) )           q_cam_new(t) = q_cam(t) * correction(t)
(no sign flip: content appearing rotated by +eps means the camera really was rotated by +eps more than the
estimate said). Low frequencies (< hp_hz) are intent / parallax / calibration drift and are absorbed by path
smoothing, so they are removed with a zero-phase Butterworth high-pass (order 4: the integrated error carries
degrees of LF drift during manoeuvres, an order-2 fold leaked it as 1-3 Hz wander).

Robustness (M1 verifier findings: single-frame vision errors became lasting 1-3 px jumps)
-----------------------------------------------------------------------------------------
1. Pair weights: confidence ramp conf_lo -> conf_min (squared), then ROBUST OUTLIER REJECTION before
   integration (`reject_outliers`): a Hampel test of every per-pair error against the running median of its
   trusted neighbours (+-4 pairs), with a local robust scale (+-0.5 s) and an absolute floor; a spike is only
   kept when the gyro itself shows a comparable one-frame event at that pair (`pred_px`, the gyro-predicted
   residual magnitude). An isolated spike is a step in the integrated error (a lasting offset); two opposite
   spikes are a one-frame blip; both are rejected and bridged.
2. Weighted integration (`weighted_integrate`): eps minimising sum w (D1 eps - rel)^2 + lam |D2 eps|^2, so
   untrusted pairs are bridged by a C1 continuation instead of line-fit links (which injected 1-3 Hz steps).
3. Smooth support (`support_weight`): the correction is multiplied by a per-frame weight that is a smoothstep
   of the local mean pair weight, Hann-smoothed over +-fade_s, so it can never switch on or off within a frame
   (the M1 fold faded only on the trusted side and then dropped to exactly 0: the 2.9 px jump on DJI_0032).
4. Jump guard (`jump_guard`): per-frame change of the correction that is a velocity outlier (vs its 7-tap
   running median) larger than jump_px at a pair without strong local measurement support is reverted
   locally (smooth Hann notch of +-notch_s).
5. Gyro events the image does not confirm (`gyro_event_vectors`, O4 Pro 0004 frame-step diagnosis): the
   one-frame event the gyro path imposes on the stabilized view is the velocity outlier of the virtual->camera
   rotation series; if the image does not share it, the preview shows exactly that vector as a pair residual.
   Such a vision spike (aligned with the gyro event, up to explain_max x its size) is not an outlier: it is the
   image refuting the gyro, and folding it cancels the false gyro spike. (Before, it was rejected, because a
   spike larger than pred_gain x the gyro event was taken for a vision failure, and the false spike stayed.)
6. Step limiter (`limit_steps`): the correction may step from one frame to the next by at most
   max(step_floor_px, step_gyro_gain x the gyro's own one-frame event at that pair); larger steps are turned
   into smooth ramps (+-step_relax_s), keeping the correction's position. A correction that changes faster
   than the gyro's own events is the vision chasing its own noise / parallax: on O4 Pro 0004 the unbacked
   correction steps (median 2.3x the gyro event) coincided with the independent eval's new 0.6-0.8 px jumps,
   while on O3 DJI_0034 correction steps are 0.8x the (over-reported) gyro events (the limiter removes 0.2 %
   of their energy there).

Loop bookkeeping helpers (moved here from pipeline.py): `hf_path_px`, `window_hf`, `window_mask`,
`accept_windows`, `step_series`.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from .geom import qconj, qexp, qlog, qmul, qrotate

__all__ = ['fold_residuals', 'ResidualCorrection', 'compose_corrections', 'identity_correction',
           'pair_weights', 'reject_outliers', 'weighted_integrate', 'support_weight', 'jump_guard',
           'gyro_event_vectors', 'limit_steps', 'step_series', 'hf_path_px', 'window_hf', 'window_mask', 'accept_windows', 'axis_px_scale',
           'R_ROLL_1920']

R_ROLL_1920 = math.sqrt((1920.0 ** 2 + 1080.0 ** 2) / 12.0)   # eval's roll -> px factor (16:9, 1920 wide)


class ResidualCorrection:
    """Callable t -> correction quaternion(s) (…,4) (w,x,y,z), to right-multiply onto q_cam(t).

    Smooth in t (natural cubic spline of camera-frame rotation vectors between frame times), so it can be
    evaluated at per-row capture times. Outside [t0, t_end] the end value is held.
    `corr, diag = fold_residuals(...)` also works (iterates as (self, diagnostics)).
    """

    def __init__(self, t_knots: np.ndarray, rotvec: np.ndarray, diagnostics: dict | None = None):
        self.t_knots = np.asarray(t_knots, dtype=np.float64)
        self.rotvec_knots = np.asarray(rotvec, dtype=np.float64)
        self.diagnostics = diagnostics if diagnostics is not None else {}
        if len(self.t_knots) >= 2:
            self._spl = CubicSpline(self.t_knots, self.rotvec_knots, axis=0, bc_type='natural')
        else:
            self._spl = None

    def rotvec(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        if self._spl is None:
            return np.zeros(t.shape + (3,))
        tc = np.clip(t, self.t_knots[0], self.t_knots[-1])
        return self._spl(tc)

    def scaled(self, m: np.ndarray) -> 'ResidualCorrection':
        """Same correction with its knots multiplied by a per-knot factor m (F,) (local acceptance / revert)."""
        return ResidualCorrection(self.t_knots, self.rotvec_knots * np.asarray(m, dtype=np.float64)[:, None],
                                  self.diagnostics)

    def __call__(self, t) -> np.ndarray:
        return qexp(self.rotvec(t))

    def __iter__(self):
        yield self
        yield self.diagnostics


class _Composed:
    def __init__(self, parts):
        self.parts = list(parts)

    def __call__(self, t):
        q = self.parts[0](t)
        for c in self.parts[1:]:
            q = qmul(q, c(t))
        return q


def compose_corrections(*corrections):
    """Chain closed-loop iterations: q_cam * c1 * c2 * ... (c1 applied first, i.e. c2 was measured on
    previews rendered with q_cam * c1)."""
    parts = [c for c in corrections if c is not None]
    if not parts:
        return identity_correction()
    return parts[0] if len(parts) == 1 else _Composed(parts)


def identity_correction() -> ResidualCorrection:
    return ResidualCorrection(np.array([0.0]), np.zeros((1, 3)), {'identity': True})


# ============================================================================================ small helpers


def axis_px_scale(px_per_rad: float, roll_px_per_rad: float | None = None) -> np.ndarray:
    """(3,) factors turning a small rotation (pitch x, yaw y, roll z; rad) into image displacement px:
    f for x/y, sqrt((W^2+H^2)/12) for roll (eval convention). Default roll factor: the 1080p-eq ratio."""
    f = float(px_per_rad)
    # typical O3 output (hfov ~100 deg): f1920 ~ 810 px/rad vs roll factor 636 px/rad -> 0.785 f
    r = float(roll_px_per_rad) if roll_px_per_rad else 0.785 * f
    return np.array([f, f, r])


def _highpass(x: np.ndarray, fs: float, hp_hz: float, order: int) -> np.ndarray:
    n = len(x)
    if hp_hz <= 0:
        return x - x.mean(0)
    if n < 8:
        A = np.c_[np.ones(n), np.arange(n)]
        return x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    sos = butter(order, hp_hz, btype='highpass', fs=fs, output='sos')
    padlen = int(min(n - 1, max(3 * fs / hp_hz, 3 * (2 * len(sos) + 1))))
    return sosfiltfilt(sos, x, axis=0, padtype='odd', padlen=padlen)


def _runs(mask: np.ndarray):
    """(start, stop) of runs of True."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _running_nanmedian(x: np.ndarray, half: int) -> np.ndarray:
    """Running median over +-half samples along axis 0, ignoring NaN (NaN where the window is all-NaN)."""
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    pad = np.full((half,) + x.shape[1:], np.nan)
    xp = np.concatenate([pad, x, pad], 0)
    win = np.lib.stride_tricks.sliding_window_view(xp, 2 * half + 1, axis=0)   # (n, C, 2h+1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        m = np.nanmedian(win, axis=-1)
    return m[:, 0] if squeeze else m


def _running_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Centred running mean over n samples (edge-replicated)."""
    n = max(1, int(n))
    if n == 1:
        return np.asarray(x, dtype=np.float64).copy()
    k = np.ones(n) / n
    lp, rp = n // 2, n - 1 - n // 2
    return np.convolve(np.pad(np.asarray(x, dtype=np.float64), (lp, rp), mode='edge'), k, mode='valid')


def _hann_smooth(x: np.ndarray, half: int) -> np.ndarray:
    """Hann-weighted running mean over +-half samples (edge-replicated); exact zeros stay zero."""
    if half < 1:
        return np.asarray(x, dtype=np.float64).copy()
    ker = np.hanning(2 * half + 3)[1:-1]
    ker /= ker.sum()
    return np.convolve(np.pad(np.asarray(x, dtype=np.float64), half + 0, mode='edge'), ker, mode='valid')


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ============================================================================================ robust folding


def pair_weights(conf: np.ndarray, conf_lo: float = 0.3, conf_hi: float = 0.6) -> np.ndarray:
    """Confidence -> integration weight: 0 below conf_lo, 1 above conf_hi, squared ramp in between."""
    cf = np.asarray(conf, dtype=np.float64)
    w = np.clip((cf - conf_lo) / max(conf_hi - conf_lo, 1e-6), 0.0, 1.0) ** 2
    w[~np.isfinite(cf)] = 0.0
    return w


def step_series(x: np.ndarray, axis_px: np.ndarray, half: int = 3) -> np.ndarray:
    """Velocity-outlier ('step') size per pair of a per-frame position series x (F,3) [rad]: |v - median7(v)|
    in px, v = diff(x) * axis_px. An isolated velocity outlier is a position jump (what a viewer sees)."""
    v = np.diff(np.asarray(x, dtype=np.float64), axis=0) * np.asarray(axis_px)[None, :]
    if len(v) == 0:
        return np.zeros(0)
    med = median_filter(v, size=(2 * half + 1, 1), mode='nearest')
    return np.linalg.norm(v - med, axis=1)


def reject_outliers(rel: np.ndarray, w: np.ndarray, axis_px: np.ndarray, *, pred_px: np.ndarray | None = None,
                    half: int = 4, scale_half: int = 30, k_sigma: float = 5.0, floor_px: float = 0.15,
                    pred_gain: float = 0.35, pred_ceiling: float | None = None, pred_min_px: float = 0.4,
                    w_min: float = 0.02, iters: int = 2, pred_vec: np.ndarray | None = None,
                    explain_cos: float = 0.7, explain_max: float = 1.4,
                    explain_min_px: float = 0.3) -> tuple[np.ndarray, dict]:
    """Hampel-type rejection of per-pair error spikes BEFORE integration.

    rel (P,3) per-pair errors [rad]; w (P,) weights (pairs with w < w_min are not used as neighbours and are
    never flagged); axis_px (3,) rad -> px. A pair is an outlier when its deviation from the running median
    of its trusted neighbours (+-half pairs) exceeds max(k_sigma * local robust scale, floor_px), unless it is
    small against the gyro's own one-frame event at that pair (dev <= pred_gain * pred_px: a gyro timing / gain
    error leaves a residual that is a fraction of the jolt). pred_px (P,) is that gyro-predicted magnitude (same
    operator on the gyro motion). Conversely a spike LARGER than pred_ceiling * pred_px (+floor) at a jolt
    (pred_px > pred_min_px) is rejected already at half the Hampel threshold: the gyro cannot be that wrong about
    its own jolt, so it is the vision fit that failed (blur / intra-frame RS during the jolt; DJI_0028 57.9 s).
    That ceiling is OFF by default (pred_ceiling None): on O3 footage the gyro over-reports HF roll, legit
    residuals are ~50 % of the gyro's own events, and the ceiling rejected 257 good pairs on DJI_0028 (loop HF
    0.41 -> 0.52 px). The pipeline's verification veto handles that case instead.
    pred_vec (P,3) px (optional, `gyro_event_vectors`): the gyro's one-frame event as it appears in the stabilized
    view if the image does not share it. A flagged spike whose deviation points along pred_vec (cosine >=
    explain_cos) and is at most explain_max x |pred_vec| (with |pred_vec| >= explain_min_px) is the image refuting
    a gyro spike: kept, so the fold cancels it (unless pred_ceiling is set, which takes precedence).
    Returns (outlier bool (P,), info)."""
    rel = np.asarray(rel, dtype=np.float64)
    P = len(rel)
    out = np.zeros(P, bool)
    info = dict(n_outliers=0)
    if P < 2 * half + 1:
        return out, info
    d = rel * np.asarray(axis_px)[None, :]
    use = (np.asarray(w) >= w_min) & np.isfinite(d).all(1)
    dev = np.zeros(P)
    pv = None
    if pred_vec is not None:
        pv = np.nan_to_num(np.asarray(pred_vec, dtype=np.float64))
        pvn = np.linalg.norm(pv, axis=1)
    explained = np.zeros(P, bool)
    for _ in range(max(1, iters)):
        dn = np.where((use & ~out)[:, None], d, np.nan)
        med = _running_nanmedian(dn, half)
        med = np.where(np.isfinite(med), med, 0.0)
        dvec = d - med
        dev = np.linalg.norm(dvec, axis=1)
        devn = np.where(use & ~out, dev, np.nan)
        scale = _running_nanmedian(devn, scale_half)
        g = np.nanmedian(devn) if np.isfinite(devn).any() else floor_px
        scale = np.where(np.isfinite(scale), scale, g)
        thr = np.maximum(k_sigma * scale, floor_px)
        new = use & (dev > thr)
        if pred_px is not None:
            pr = np.asarray(pred_px, dtype=np.float64)
            new &= ~(dev <= pred_gain * pr)
        if pv is not None:
            # the image refutes the gyro's one-frame event: the residual IS that event (same direction, <= its size)
            cos = np.einsum('ij,ij->i', dvec, pv) / np.maximum(dev * pvn, 1e-12)
            explained = new & (pvn >= explain_min_px) & (cos >= explain_cos) & (dev <= explain_max * pvn)
            new &= ~explained
        if pred_px is not None and pred_ceiling is not None:
            new |= use & (pr > pred_min_px) & (dev > pred_ceiling * pr + floor_px) & (dev > 0.5 * k_sigma * scale)
        if np.array_equal(new, out):
            break
        out = new
    info.update(n_outliers=int(out.sum()), dev_px=dev, n_gyro_refuted=int(explained.sum()),
                gyro_refuted=np.flatnonzero(explained))
    return out, info


def weighted_integrate(rel: np.ndarray, w: np.ndarray, lam: float = 0.01, *, fs: float | None = None,
                       anchor_s: float | None = 0.3, lf_hz: float = 1.0) -> np.ndarray:
    """Integrate per-pair increments rel (P,3) with weights w (P,) into eps (P+1,3) (zero-mean).

    anchor_s None: eps minimising sum w_i (D1 eps - rel)_i^2 + lam |D2 eps|^2 (untrusted pairs bridged by the
    C1 continuation). Across a gap the HF increments are unknown, so eps after the gap is offset by their sum:
    after the fold high-pass that offset is a ~1 s transient (a wrong correction around every rejected pair).

    anchor_s > 0 (default): eps = L + H with L low-frequency (mu |D2 L|^2, cutoff ~lf_hz) and H the
    zero-mean HF part (ridge rho |H|^2): after a gap the offset is re-anchored so the LF stays continuous and the
    HF stays zero-mean (a soft, global version of linking runs by their trends), with an anchoring time
    ~anchor_s; lam keeps eps C1 across gaps. In trusted stretches the result equals plain integration up to a
    few-% shrink of < 2 Hz content where the weights are low."""
    import scipy.sparse as sps
    from scipy.sparse.linalg import splu
    rel = np.asarray(rel, dtype=np.float64)
    P = len(rel)
    n = P + 1
    if P == 0:
        return np.zeros((1, rel.shape[1] if rel.ndim == 2 else 3))
    w = np.where(np.isfinite(rel).all(1), np.asarray(w, dtype=np.float64), 0.0)
    rel = np.where(np.isfinite(rel), rel, 0.0)
    D1 = sps.diags([-np.ones(P), np.ones(P)], [0, 1], shape=(P, n), format='csr')
    D2 = sps.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(max(n - 2, 0), n),
                   format='csr') if n >= 3 else sps.csr_matrix((0, n))
    Wd = sps.diags(w)
    if anchor_s is None or anchor_s <= 0 or n < 8:
        A = (D1.T @ Wd @ D1 + lam * (D2.T @ D2) + 1e-9 * sps.eye(n)).tocsc()
        rhs = D1.T @ (w[:, None] * rel)
        lu = splu(A)
        eps = np.column_stack([lu.solve(np.ascontiguousarray(rhs[:, j])) for j in range(rel.shape[1])])
        return eps - eps.mean(0)
    fs = 60.0 if fs is None else float(fs)
    mu = (fs / (2 * np.pi * lf_hz)) ** 2              # data w*omega^2 == mu*omega^4 at lf_hz
    T = max(anchor_s * fs, 2.0)
    rho = mu / T ** 4
    # z = [L; H]; eps = L + H
    I = sps.eye(n, format='csr')
    DD = D1.T @ Wd @ D1
    S2 = D2.T @ D2
    A = sps.bmat([[DD + (lam + mu) * S2 + 1e-9 * I, DD + lam * S2],
                  [DD + lam * S2, DD + lam * S2 + rho * I]], format='csc')
    r1 = D1.T @ (w[:, None] * rel)
    rhs = np.vstack([r1, r1])
    lu = splu(A)
    z = np.column_stack([lu.solve(np.ascontiguousarray(rhs[:, j])) for j in range(rel.shape[1])])
    eps = z[:n] + z[n:]
    return eps - eps.mean(0)


def support_weight(w: np.ndarray, fs: float, win_s: float = 0.25, lo: float = 0.15, hi: float = 0.35,
                   fade_s: float = 0.15) -> np.ndarray:
    """Per-FRAME weight (P+1,) in [0,1] from per-pair weights (P,): smoothstep of the running mean pair weight
    over win_s, Hann-smoothed over +-fade_s so it never changes by more than ~1/(fade_s*fs) per frame."""
    w = np.asarray(w, dtype=np.float64)
    P = len(w)
    if P == 0:
        return np.zeros(1)
    s = _running_mean(w, int(round(win_s * fs)) | 1)
    sf = np.r_[s[:1], 0.5 * (s[:-1] + s[1:]), s[-1:]]                   # pairs -> frames
    g = _smoothstep((sf - lo) / max(hi - lo, 1e-6))
    g[sf <= lo] = 0.0
    return np.clip(_hann_smooth(g, max(1, int(round(fade_s * fs)))), 0.0, 1.0)


def _notch(F: int, centres: np.ndarray, half: int) -> np.ndarray:
    """Product of Hann notches (1 at +-half frames away, ~0 at the centre) at fractional frame positions."""
    m = np.ones(F)
    j = np.arange(F, dtype=np.float64)
    for c in np.atleast_1d(centres):
        u = np.abs(j - c) / max(half, 1)
        b = np.where(u < 1.0, 0.5 * (1.0 + np.cos(np.pi * u)), 0.0)
        m *= 1.0 - b
    return m


def jump_guard(c: np.ndarray, support: np.ndarray, axis_px: np.ndarray, fs: float, *, jump_px: float = 0.5,
               support_min: float = 0.5, notch_s: float = 0.2, max_iter: int = 3) -> tuple[np.ndarray, dict]:
    """Revert unsupported jumps of a correction series locally.

    c (F,3) correction [rad] in the VIRTUAL (viewer) frame, support (F-1,) per-pair measurement support 0..1
    (e.g. the 5-pair running mean of the integration weights). A pair whose correction change is a velocity
    outlier (step_series) larger than jump_px while support < support_min is not backed by consistent
    measurements: the correction is faded to zero around it with a +-notch_s Hann notch.
    Returns (c_new, info{n_fixed, pairs, factor (F,)})."""
    c = np.asarray(c, dtype=np.float64)
    F = len(c)
    fac = np.ones(F)
    fixed: list[int] = []
    if F < 8:
        return c.copy(), dict(n_fixed=0, pairs=[], factor=fac)
    sup = np.asarray(support, dtype=np.float64)
    half = max(2, int(round(notch_s * fs)))
    cur = c.copy()
    for _ in range(max_iter):
        J = step_series(cur, axis_px)
        bad = np.flatnonzero((J > jump_px) & (sup < support_min))
        bad = np.setdiff1d(bad, fixed)
        if len(bad) == 0:
            break
        fixed.extend(int(b) for b in bad)
        fac *= _notch(F, bad + 0.5, half)
        cur = c * fac[:, None]
    return cur, dict(n_fixed=len(fixed), pairs=sorted(fixed), factor=fac)


def gyro_event_vectors(virt_to_cam_q: np.ndarray, axis_px: np.ndarray, half: int = 3) -> np.ndarray:
    """(P,3) px, virtual frame: the one-frame event the gyro-driven camera path imposes on the stabilized view at
    each pair -- the velocity outlier (vs the +-half running median) of log(conj(M_k) M_{k+1}), M = conj(q_cam) V
    (virtual -> camera). The virtual path is smooth, so its outliers are the camera path's. If the image does not
    share the event (a false gyro spike), a perfect preview measurement returns exactly this vector as the pair's
    residual (err_rotvec * axis_px)."""
    M = np.asarray(virt_to_cam_q, dtype=np.float64)
    if len(M) < 2:
        return np.zeros((0, 3))
    d = qlog(qmul(qconj(M[:-1]), M[1:])) * np.asarray(axis_px)[None, :]
    if len(d) < 2 * half + 1:
        return np.zeros_like(d)
    return d - median_filter(d, size=(2 * half + 1, 1), mode='nearest')


def _hann_smooth_cols(x: np.ndarray, half: int) -> np.ndarray:
    return np.column_stack([_hann_smooth(x[:, j], half) for j in range(x.shape[1])])


def limit_steps(c: np.ndarray, axis_px: np.ndarray, allow_px: np.ndarray, fs: float, relax_s: float = 0.2,
                half: int = 3, iters: int = 3) -> tuple[np.ndarray, dict]:
    """Turn frame-to-frame steps of a correction series into smooth ramps.

    c (F,3) [rad], allow_px (F-1,) the largest step (velocity outlier vs the +-half running median, px) allowed at
    each pair. The part of an outlier above its allowance is removed from the velocity, the series re-integrated,
    and the lasting offset that removal leaves is given back as a Hann ramp over +-relax_s: the correction reaches
    the same values a few frames later instead of jumping there. Returns (c_new, info{n_limited, pairs})."""
    c = np.asarray(c, dtype=np.float64).copy()
    F = len(c)
    ax = np.asarray(axis_px, dtype=np.float64)
    allow = np.asarray(allow_px, dtype=np.float64)
    limited: set = set()
    if F < 2 * half + 3:
        return c, dict(n_limited=0, pairs=[])
    hs = max(1, int(round(relax_s * fs)))
    for _ in range(max(1, iters)):
        v = np.diff(c, axis=0) * ax[None, :]
        med = median_filter(v, size=(2 * half + 1, 1), mode='nearest')
        dv = v - med
        n = np.linalg.norm(dv, axis=1)
        over = n > allow * 1.02
        if not over.any():
            break
        limited.update(int(k) for k in np.flatnonzero(over))
        s = np.where(over, allow / np.maximum(n, 1e-12), 1.0)
        v_new = med + dv * s[:, None]
        c_new = np.vstack([c[:1], c[:1] + np.cumsum(v_new / ax[None, :], axis=0)])
        e = c_new - c                                    # the removed steps, as lasting offsets
        c = c + e - _hann_smooth_cols(e, hs)              # ... given back smoothly
    return c, dict(n_limited=len(limited), pairs=sorted(limited))


def fold_residuals(frame_t, rel_err_rotvec, conf, fs=None, hp_hz: float = 1.0, clamp_deg: float = 0.5, *,
                   virt_to_cam_q: np.ndarray | None = None, conf_min: float = 0.6, conf_lo: float = 0.3,
                   gain: float = 1.0, hp_order: int = 4, lam: float = 0.01, weights: np.ndarray | None = None,
                   support_lo: float = 0.15, support_hi: float = 0.35, support_win_s: float = 0.25,
                   fade_s: float = 0.15, px_per_rad: float | None = None, roll_px_per_rad: float | None = None,
                   outlier_k: float | None = 5.0, outlier_floor_px: float = 0.15, pred_px: np.ndarray | None = None,
                   pred_gain: float = 0.35, jump_px: float | None = 0.5, jump_support: float = 0.25,
                   notch_s: float = 0.2, anchor_s: float | None = 0.3,
                   max_gap_s: float | None = None, explain_gyro: bool = True,
                   step_floor_px: float | None = 0.3, step_gyro_gain: float = 1.5,
                   step_relax_s: float = 0.2) -> ResidualCorrection:
    """Integrate per-pair rotation errors into a robust, high-passed, support-weighted, clamped correction.

    frame_t: (F,) times of the preview frames (centre-row mid-exposure, video timeline) whose consecutive
        pairs were measured. rel_err_rotvec: (F-1,3) residual.measure_residuals(...)['err_rotvec'] [rad],
        virtual-camera frame of the first frame of each pair (NaN = not measured). conf: (F-1,) 0..1.
    fs: frame rate [Hz] (default 1/median(diff(frame_t))). hp_hz: keep content above this. clamp_deg:
        soft limit on |correction|.
    virt_to_cam_q: optional (F,4) = conj(q_cam(frame_t[k])) * virt_q[k] used to render the previews; rotates
        the virtual-frame error into the camera frame (None = identity, first-order-correct when small).
    px_per_rad: px per rad of the units the px thresholds (outlier_floor_px, jump_px) are given in — pass the
        1080p-equivalent output focal (f1920) to get 1080p-eq px. Default 810 (typical O3 output).
    weights: optional (F-1,) pair weights overriding the confidence ramp (conf is then only reported).
    pred_px: optional (F-1,) gyro-predicted one-frame event size per pair (px, same units) for the outlier test.
    outlier_k / jump_px: None disables outlier rejection / the jump guard. anchor_s: see weighted_integrate.
    max_gap_s: ignored (M1 compatibility).
    explain_gyro: with virt_to_cam_q, keep vision spikes that equal the gyro's own one-frame event (the image
        refuting a gyro spike; module docstring 5) so the fold cancels false gyro spikes.
    step_floor_px / step_gyro_gain / step_relax_s: step limiter (module docstring 6): the correction steps by at
        most max(step_floor_px, step_gyro_gain x the gyro's one-frame event at the pair, +-1 pair) per frame;
        larger steps become ramps over +-step_relax_s. Needs a gyro reference (virt_to_cam_q or pred_px);
        step_floor_px None disables it.
    Returns a ResidualCorrection (callable, .diagnostics; `corr, diag = fold_residuals(...)` also works).
    """
    t = np.asarray(frame_t, dtype=np.float64)
    rel = np.array(rel_err_rotvec, dtype=np.float64, copy=True)
    cf = np.asarray(conf, dtype=np.float64)
    F = len(t)
    if rel.shape != (F - 1, 3) or cf.shape != (F - 1,):
        raise ValueError(f'expected rel_err (F-1,3)={F - 1, 3} and conf (F-1,), got {rel.shape}, {cf.shape}')
    if fs is None:
        fs = 1.0 / float(np.median(np.diff(t))) if F > 1 else 60.0
    ppr = float(px_per_rad) if px_per_rad else 810.0
    ax = axis_px_scale(ppr, roll_px_per_rad)
    w = pair_weights(cf, conf_lo, conf_min) if weights is None else np.asarray(weights, dtype=np.float64).copy()
    finite = np.isfinite(rel).all(1)
    w[~finite] = 0.0
    n_bad = int((w <= 0).sum())
    diag = dict(n_frames=F, n_pairs=F - 1, n_bad=n_bad, bad_frac=float(n_bad / max(F - 1, 1)),
                fs=float(fs), hp_hz=hp_hz, clamp_deg=clamp_deg, conf_min=conf_min, conf_lo=conf_lo)
    if F < 3 or (w > 0).sum() < 2:
        diag.update(identity=True, reason='too few trusted pairs', weight=np.zeros(F), n_outliers=0, n_jump_fixed=0)
        return ResidualCorrection(t, np.zeros((F, 3)), diag)
    rel[~finite] = 0.0
    M = None
    gvec = None
    if virt_to_cam_q is not None:
        M = np.asarray(virt_to_cam_q, dtype=np.float64)
        if M.shape != (F, 4):
            raise ValueError('virt_to_cam_q must be (F,4)')
        gvec = gyro_event_vectors(M, ax)

    # 1) robust outlier rejection on the per-pair errors (before integration)
    oinfo = {}
    if outlier_k is not None and outlier_k > 0:
        out, oinfo = reject_outliers(rel, w, ax, pred_px=pred_px, k_sigma=outlier_k, floor_px=outlier_floor_px,
                                     pred_gain=pred_gain, pred_vec=gvec if explain_gyro else None)
        w = np.where(out, 0.0, w)
    else:
        out = np.zeros(F - 1, bool)
    # 2) weighted integration (untrusted / rejected pairs are bridged smoothly)
    eps = weighted_integrate(rel, w, lam, fs=fs, anchor_s=anchor_s)
    # 3) zero-phase high-pass
    eps_hp = _highpass(eps, fs, hp_hz, hp_order)
    # 4) smooth per-frame support weight (never switches within a frame)
    wf = support_weight(w, fs, support_win_s, support_lo, support_hi, fade_s)
    c_v = gain * wf[:, None] * eps_hp
    # 5) soft clamp to clamp_deg: identity up to 80 % of the limit, then a C1 tanh knee
    lim = np.deg2rad(clamp_deg)
    n = np.linalg.norm(c_v, axis=1)
    knee = 0.8 * lim
    n_cl = np.where(n <= knee, n, knee + (lim - knee) * np.tanh((n - knee) / (lim - knee)))
    c_v = c_v * np.where(n > 1e-12, n_cl / np.maximum(n, 1e-12), 1.0)[:, None]
    # 6) step limiter: no frame-to-frame step larger than the gyro's own one-frame event allows
    sinfo = dict(n_limited=0, pairs=[])
    if step_floor_px is not None and (gvec is not None or pred_px is not None):
        G = np.zeros(F - 1)
        if gvec is not None:
            G = np.linalg.norm(gvec, axis=1)
        if pred_px is not None:
            G = np.maximum(G, np.nan_to_num(np.asarray(pred_px, dtype=np.float64)))
        G3 = np.maximum(G, np.maximum(np.r_[G[1:], 0.0], np.r_[0.0, G[:-1]]))
        c_v, sinfo = limit_steps(c_v, ax, np.maximum(step_floor_px, step_gyro_gain * G3), fs, step_relax_s)
    # 7) jump guard in the viewer (virtual) frame
    sup = _running_mean(w, 5)
    jinfo = dict(n_fixed=0, pairs=[], factor=np.ones(F))
    if jump_px is not None and jump_px > 0:
        c_v, jinfo = jump_guard(c_v, sup, ax, fs, jump_px=jump_px, support_min=jump_support, notch_s=notch_s)
    # 8) virtual frame -> camera frame
    c = c_v
    if M is not None:
        c = qrotate(M, c_v)

    rms = lambda x: float(np.sqrt(np.mean(np.sum(np.asarray(x) ** 2, axis=-1))))
    steps = step_series(c_v, ax)
    diag.update(identity=False, eps=eps, eps_hp=eps_hp, weight=wf, pair_weight=w, outliers=np.flatnonzero(out),
                n_outliers=int(out.sum()), n_jump_fixed=int(jinfo['n_fixed']), jump_pairs=jinfo['pairs'],
                n_gyro_refuted=int(oinfo.get('n_gyro_refuted', 0)), gyro_refuted=oinfo.get('gyro_refuted', []),
                n_steps_limited=int(sinfo['n_limited']), steps_limited=sinfo['pairs'],
                corr_rotvec=c, corr_virt=c_v,
                hf_rms_deg=np.rad2deg(rms(eps_hp)), lf_rms_deg=np.rad2deg(rms(eps - eps_hp - (eps - eps_hp).mean(0))),
                corr_rms_deg=np.rad2deg(rms(c)), corr_max_deg=float(np.rad2deg(n.max())),
                clamp_frac=float((n > 0.8 * lim).mean()), max_step_px=float(steps.max()) if len(steps) else 0.0,
                support_frac=float((wf > 0.5).mean()))
    diag['hf_rms_px'] = rms(eps_hp * ax)
    diag['corr_rms_px'] = rms(c_v * ax)
    return ResidualCorrection(t, c, diag)


# ============================================================================================ loop bookkeeping


def _pairs_to_series(res: dict, n_frames: int, conf_min: float, weights: np.ndarray | None = None):
    P = n_frames - 1
    rel = np.zeros((P, 3))
    cf = np.zeros(P)
    k0, k1 = np.asarray(res['k0']), np.asarray(res['k1'])
    ok = (k1 == k0 + 1) & (k0 >= 0) & (k0 < P)
    rel[k0[ok]] = np.asarray(res['err_rotvec'])[ok]
    cf[k0[ok]] = np.asarray(res['conf'])[ok]
    rel[~np.isfinite(rel).all(1)] = 0.0
    rel[cf < conf_min] = 0.0
    if weights is not None:
        rel[np.asarray(weights) <= 0] = 0.0
    measured = np.zeros(P, bool)
    measured[k0[ok]] = True
    return rel, cf, measured


def hf_path_px(res: dict, n_frames: int, fs: float, axis_px: np.ndarray, conf_min: float = 0.3,
               fc: float = 2.0, weights: np.ndarray | None = None) -> np.ndarray:
    """Per-frame squared >fc Hz displacement (px^2) of the measured residual path (untrusted pairs = 0)."""
    rel, _, _ = _pairs_to_series(res, n_frames, conf_min, weights)
    eps = np.vstack([np.zeros((1, 3)), np.cumsum(rel, axis=0)])
    if n_frames < 16:
        return np.zeros(n_frames)
    hp = sosfiltfilt(butter(4, fc, btype='highpass', fs=fs, output='sos'), eps, axis=0)
    return np.sum((hp * np.asarray(axis_px)[None, :]) ** 2, axis=1)


def window_hf(res: dict, n_frames: int, fs: float, f1920: float, wl: int, conf_min: float = 0.3,
              axis_px: np.ndarray | None = None, weights: np.ndarray | None = None) -> np.ndarray:
    """Per-window (wl frames, aligned to frame 0; the tail joins the last window) RMS of the measured >2 Hz
    residual path, 1080p-eq px (roll with eval's sqrt((W^2+H^2)/12) factor)."""
    ax = np.array([f1920, f1920, R_ROLL_1920]) if axis_px is None else np.asarray(axis_px)
    e2 = hf_path_px(res, n_frames, fs, ax, conf_min, weights=weights)
    nw = max(1, n_frames // wl)
    idx = np.minimum(np.arange(n_frames) // wl, nw - 1)
    return np.sqrt(np.bincount(idx, e2, nw) / np.bincount(idx, None, nw))


def window_mask(acc: np.ndarray, n_frames: int, wl: int, xfade: int) -> np.ndarray:
    """Per-frame 0..1 mask of accepted windows with a Hann crossfade of xfade frames at every transition."""
    nw = len(acc)
    idx = np.minimum(np.arange(n_frames) // wl, nw - 1)
    m = np.asarray(acc)[idx].astype(np.float64)
    if xfade > 1 and not m.all():
        ker = np.hanning(xfade + 2)[1:-1]
        ker /= ker.sum()
        m = np.convolve(np.pad(m, xfade, mode='edge'), ker, mode='same')[xfade:-xfade]
    return np.clip(m, 0.0, 1.0)


def accept_windows(h_new: np.ndarray, h_best: np.ndarray, tol: float = 0.05, abs_px: float = 0.005,
                   veto: np.ndarray | None = None) -> np.ndarray:
    """Local acceptance: keep an increment in a window if its measured HF is not worse than the best so far
    (h_new <= h_best*(1+tol) + abs_px) and no veto (e.g. a new jump) was raised there."""
    acc = np.asarray(h_new) <= np.asarray(h_best) * (1 + tol) + abs_px
    if veto is not None:
        acc &= ~np.asarray(veto, bool)
    return acc
