"""Per-clip timing self-calibration for every camera (engine v5, workstream E; replaces calib.self_calibrate in the
pipeline).

Why: the per-row rolling-shutter correction is only as good as the picture <-> gyro timing. Two manual diagnoses
found camera/firmware quirks the metadata hides (research/notes/oa4_0012_timing_diagnosis.md: OA4 picture 0.92 ms
off -> at 250 Hz prop vibration the RS correction ADDED wobble; the O4 Pro's "2 kHz" stream is 1 kHz written twice).
This module does that diagnosis automatically on every clip, bounded in time, and applies a correction only when it
is confident.

Model (the renderer's own row-time model, plan_build.build_plan):
    row time   t(k, y) = frame_t[k] + offset + gamma*(e_k - e_ref) + readout*((y+0.5)/H - 0.5)
    camera     q(t)    = gyro orientation, averaged over the row's exposure window exposure_avg_window(e_k)
                         (the same box model the renderer uses), lens focal x focal_scale.
Measurement: KLT tracks on the ORIGINAL (fisheye) analysis frames of a few automatically chosen windows (most
    gyro excitation per second, little motion blur, spread over the clip). For every track through three consecutive
    frames the rotation-only transfer error of the pair (k+1, k+2) MINUS that of (k, k+1): the "delta residual".
    The plain transfer error of low fast flight is dominated by translation parallax (4-5 px @960, nearly flat in the
    offset); the delta cancels each track's own parallax and keeps the timing signal (research note above).
Fit (Cauchy-robust):
    1. coarse: offset only, on a +-5 ms grid, with the gyro's HF (> lf_hz) removed -> unimodal (prop vibration at
       200-300 Hz makes the full-band cost periodic with ~4 ms local minima);
    2. fine: full-band offset sweep +-1 ms around it (the readout fit is aliased at a wrong offset, so the offset
       comes first);
    3. joint Gauss-Newton IRLS on (offset, readout, focal, exposure slope, exposure-box width) -> which nuisance
       parameters are clearly there;
    4. conditional fit: offset + only those (the others fixed at the metadata, as the renderer will use them).
    Uncertainty: jackknife over 1-s blocks (captures the correlated track errors), per-window estimates for a
    consistency check.
Decision: applied only when (a) the offset's jackknife sigma is small, it is inside physical bounds and the windows
    agree, and (b) the correction is worth it and real: it lowers the delta cost by >= gain_min in sample AND, fitted
    without a window, lowers that held-out window's cost (most windows). Otherwise the metadata is kept. (b) exists
    because on O3 the jackknife said -0.29..+0.14 ms at 3-6 sigma while held-out windows did not improve (cost
    changes of +-0.4 %); a 1 ms error changes the cost by several % (OA4: 27 %). (c) The offset must also be where
    the gyro's HF part (> lf_hz, prop vibration) fits best: the maneuver-driven LF information is coupled to
    translation / parallax (O3 DJI_0027, 2-s windows: -0.81 ms at 18 sigma, passing (a) and (b); the HF part put it at
    +0.69 with no HF gain, 5-s windows around the same moments said -0.27). Without HF content (handheld, 1/61 s) no
    offset is applied. Nuisances additionally need a clearly significant change (the old calib's +11.8 % readout on
    DJI_0025 raised eval jello 0.56 -> 0.96).
Long exposures (OA4 1/61 s = 16.4 ms): the rows are box-averaged over the exposure (plan_build); the box width is
    a fitted parameter there (TimeModel.exposure_scale; 0005: the picture looks averaged over ~2 exposures and
    ~2.6 ms late, consistent with temporal noise reduction at ISO 12800).
Cost: a few hundred decoded frames (default 3 x 5 s) + KLT + a few s of fitting; see TimecalParams.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .calib import _max_theta_d, unproject_robust
from .geom import Lens, qconj, qexp, qfix_sign, qlog, qmul
from .plan_build import exposure_avg_window
from .types import Telemetry, TimeModel

__all__ = ['TimecalParams', 'calibrate_timing', 'choose_windows', 'track_frames', 'build_deltas', 'fit_timing', 'PNAMES',
           'apply_exposure_slope', 'TIMECAL_VERSION']

TIMECAL_VERSION = 2


@dataclass
class TimecalParams:
    # ---- data
    n_windows: int = 3                 # analysis windows (chosen automatically, spread over the clip)
    window_s: float = 5.0              # window length (s)
    min_window_s: float = 2.0          # ... shortened down to this on short clips so that n_windows fit
    budget_frac: float = 0.03          # ... and so that the windows total <= this fraction of the clip (run time)
    width: int = 960                   # analysis width (px)
    max_points: int = 450              # KLT features per frame
    max_deltas: int = 30000            # delta residuals used by the fit (random subset)
    coarse_deltas: int = 10000         # ... by the coarse / fine offset sweeps
    min_deltas: int = 3000             # fewer -> no calibration (metadata kept)
    # ---- model / fit
    exposure_avg: bool = True          # rows averaged over the exposure like the renderer (plan_build)
    box_mode: str = 'engine'           # studies: 'engine' (plan_build.exposure_avg_window ramp) | 'full' (= e)
    box_scale: float = 1.0             # studies: box width x this
    lf_hz: float = 40.0                # coarse stage: gyro HF above this removed
    coarse_ms: float = 5.0             # coarse offset search range (+-)
    coarse_step_ms: float = 0.25
    fine_ms: float = 1.0               # full-band offset sweep range (+-) around the coarse optimum
    fine_step_ms: float = 0.1
    cauchy_px: float = 0.5             # Cauchy scale of the delta residual at 960 px
    fit_readout: bool = True
    fit_focal: bool = True
    fit_exposure: bool = True          # exposure slope, only when the exposure varies inside the windows
    exposure_spread_min_s: float = 0.5e-3
    fit_box: bool = True               # exposure-box width scale, only when the rows are box-averaged over a long
    box_fit_min_s: float = 6e-3        # exposure (median averaging window >= this; 1/61 s = 16.4 ms)
    block_s: float = 1.0               # jackknife block length
    gn_iters: int = 6
    # ---- decision (what gets applied)
    offset_bound_ms: float = 5.0
    offset_sigma_max_ms: float = 0.10  # ... or offset_sigma_rel x |offset| for large offsets (1/61 s: weak, blurred)
    offset_sigma_rel: float = 0.10
    window_agree_ms: float = 0.25      # per-window offsets must agree to max(this, 4 sigma_w)
    readout_bound_rel: float = 0.25
    readout_min_rel: float = 0.015     # apply a readout change only if >= 1.5 % ...
    readout_sigma_max_rel: float = 0.01   # ... and its sigma <= 1 % and |change| >= 4 sigma
    focal_bound_rel: float = 0.05
    apply_focal: bool = False          # focal is fitted and reported, not applied: on OA4 0012 it tracks the exposure
                                       # model (-2.4 +- 0.4 % with the renderer's box ramp, 0.93-0.98 instantaneous,
                                       # 0.987 +- 0.010 with a full box; diag note), i.e. it absorbs model error
    focal_min_rel: float = 0.01
    focal_sigma_max_rel: float = 0.003
    slope_bound: float = 0.5           # exposure slope (s of timing per s of exposure)
    slope_sigma_max: float = 0.05
    slope_min: float = 0.08
    box_bound: tuple = (-50.0, 250.0)  # exposure-box width change, % (0.5x .. 3.5x the exposure; 0005: 2.5-3x)
    box_min_pct: float = 10.0          # apply a box change only if >= 10 % ...
    box_sigma_max_pct: float = 15.0    # ... with sigma <= 15 % and |change| >= 3 sigma
    min_windows_agree: int = 2
    # practical significance + out-of-sample check (the jackknife is too optimistic on O3: per-clip offsets of
    # -0.27..+0.15 ms at 3-6 sigma that do NOT lower the cost of held-out windows): a correction is applied only
    # when it lowers the in-sample delta cost by >= gain_min AND, fitted without window w, lowers window w's cost
    # for at least heldout_frac of the windows with a mean held-out gain >= heldout_gain_min.
    # Calibration (saved production tracks, 2026-09-29): real clips with correct metadata: in-sample 0.00-0.10 %,
    # held-out means -0.03..+0.05 % (5 O3, 2 OA4, 2 O4 Pro clips); errors that matter: O4 Pro pre-v4 held-sample
    # parser 0.28 % / held-out 0.32 % mean; OA4 0.3 ms 5-6 %; O3 1 ms 1.1-2.0 %; OA4 1/61 s clips 0.9-1.1 %.
    gain_min: float = 0.002
    heldout_frac: float = 0.66
    heldout_gain_min: float = 0.0015
    # HF consistency (see fit_timing step 6): the offset must also be where the gyro's HF part fits best
    hf_check: bool = True
    hf_check_ms: float = 1.5           # HF-offset sweep +- around the conditional offset
    hf_agree_ms: float = 0.15          # |HF optimum - offset| <= max(this, hf_agree_rel x |offset|)
    hf_agree_rel: float = 0.25
    hf_gain_min: float = 0.001         # ... and the HF part at that offset must fit better than at the metadata


# ============================================================================ orientation with exposure box

class _BoxOrient:
    """Orientation on a (sub)series with O(1) exposure-box averaging: the chordal mean of the (sign-continuous,
    linearly interpolated) quaternion over [t - w/2, t + w/2] from a cumulative integral. w = 0 -> nlerp.
    Uniform grids (every Stillpoint IMU series inside one segment) are indexed arithmetically, others by bisection."""

    normalize = True

    def __init__(self, t: np.ndarray, q: np.ndarray):
        self.t = np.asarray(t, np.float64)
        self.q = qfix_sign(np.asarray(q, np.float64)) if self.normalize else np.asarray(q, np.float64)
        self.dt = np.maximum(np.diff(self.t), 1e-12)
        nc = self.q.shape[1]
        self.C = np.concatenate([np.zeros((1, nc)), np.cumsum(0.5 * (self.q[1:] + self.q[:-1]) * self.dt[:, None],
                                                              axis=0)])
        med = float(np.median(self.dt))
        self.min_w = 0.25 * med
        self.uniform = bool(np.max(np.abs(self.dt - med)) < 0.02 * med)
        self.t0, self.inv = float(self.t[0]), 1.0 / med
        self.n2 = len(self.t) - 2

    def _iu(self, tq):
        if self.uniform:
            i = np.clip(((tq - self.t0) * self.inv).astype(np.int64), 0, self.n2)
            i = np.clip(i - (self.t[i] > tq) + (self.t[i + 1] <= tq), 0, self.n2)
        else:
            i = np.clip(np.searchsorted(self.t, tq, side='right') - 1, 0, self.n2)
        du = np.clip(tq - self.t[i], 0.0, self.dt[i])
        return i, du

    def _lerp(self, tq):
        i, du = self._iu(tq)
        u = (du / self.dt[i])[:, None]
        q0 = self.q[i]
        return q0 + (self.q[i + 1] - q0) * u

    def _cum(self, tq):
        i, du = self._iu(tq)
        q0 = self.q[i]
        return self.C[i] + (q0 + (self.q[i + 1] - q0) * (0.5 * du / self.dt[i])[:, None]) * du[:, None]

    def __call__(self, tq: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        tq = np.asarray(tq, np.float64)
        if w is None:
            q = self._lerp(tq)
        else:
            w = np.broadcast_to(np.asarray(w, np.float64), tq.shape)
            m = w >= self.min_w
            if m.all():
                q = (self._cum(tq + 0.5 * w) - self._cum(tq - 0.5 * w)) / w[:, None]
            else:
                q = self._lerp(tq)
                if m.any():
                    tm, wm = tq[m], w[m]
                    q[m] = (self._cum(tm + 0.5 * wm) - self._cum(tm - 0.5 * wm)) / wm[:, None]
        if not self.normalize:
            return q
        return q / np.sqrt(np.einsum('ij,ij->i', q, q))[:, None]


class _BoxVec(_BoxOrient):
    """The same box average for a vector series (the gyro's HF rotation h, rad): no sign fix, no normalisation."""
    normalize = False


def _lowpass_orientation(t: np.ndarray, q: np.ndarray, fc: float, return_h: bool = False):
    """q (x) exp(-h): the orientation with the body-frame rotation above fc removed (h = integrated high-passed body
    rate, drift removed by a second, 4x lower high-pass). return_h: also h (N,3), so that q = q_lf (x) exp(h)."""
    q = qfix_sign(q)
    dt = np.maximum(np.diff(t), 1e-9)
    fs = 1.0 / float(np.median(dt))
    if fc >= 0.45 * fs or len(t) < 64:
        return (q, np.zeros((len(q), 3))) if return_h else q
    w = qlog(qmul(qconj(q[:-1]), q[1:])) / dt[:, None]
    wh = sosfiltfilt(butter(4, fc, 'highpass', fs=fs, output='sos'), w, axis=0)
    h = np.concatenate([np.zeros((1, 3)), np.cumsum(wh * dt[:, None], axis=0)])
    h = sosfiltfilt(butter(2, fc / 4.0, 'highpass', fs=fs, output='sos'), h, axis=0)
    q_lf = qfix_sign(qmul(q, qexp(-h)))
    return (q_lf, h) if return_h else q_lf


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0, a1, a2 = a[:, 0], a[:, 1], a[:, 2]
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    return np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=1)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v (N,3) by unit quaternions q (N,4): R(q) v."""
    u = q[:, 1:]
    t = 2.0 * _cross(u, v)
    return v + q[:, :1] * t + _cross(u, t)


# ============================================================================ window choice

def _frame_rate_series(tel: Telemetry, frames: np.ndarray) -> np.ndarray:
    """(n,3) instantaneous angular rate (camera frame, rad/s) at frame_t of `frames` (central difference, 1 ms)."""
    t = tel.frame_t[frames]
    h = 0.5e-3
    qa = tel.orientation_at(t - h)
    qb = tel.orientation_at(t + h)
    return qlog(qmul(qconj(qa), qb)) / (2 * h)


def choose_windows(tel: Telemetry, n: int, window_frames: int, width: int = 960, f_lo: int = 0,
                   f_hi: Optional[int] = None, guard_s: float = 0.03) -> list[dict]:
    """Pick up to n windows [start, start + window_frames) with the most timing information per frame.

    Information of the delta residual about a time offset: f * |w(k+2) - 2 w(k+1) + w(k)| (px per s of offset; the
    second difference of the angular rate at frame rate; prop vibration aliases into it), plus the rate itself for the
    readout/focal. KLT dies in motion blur: x 1/(1 + (blur/4 px)^2). Static windows (< 3 deg/s) are skipped. Windows
    stay inside one segment and the IMU span (+- guard_s), and are spread over the clip (greedy, >= 1.5 windows
    apart; the clip is split into n equal parts and each part's best window is preferred)."""
    F = tel.n_frames
    f_hi = F if f_hi is None else min(F, int(f_hi))
    L = int(window_frames)
    if f_hi - f_lo < L + 4:
        return []
    fr = np.arange(F)
    ok = np.ones(F, bool)
    ro = float(tel.readout_s or 0.0)
    ex = np.asarray(tel.exposure_s, np.float64)
    ex = np.broadcast_to(ex, (F,)) if ex.ndim == 0 else np.nan_to_num(ex)
    exg = 1.75 * np.clip(ex, 0.0, 0.05)                      # half of the widest exposure box the fit may try
    ok &= (tel.frame_t - 0.5 * ro - exg - guard_s > tel.imu_t[0]) & (tel.frame_t + 0.5 * ro + exg + guard_s < tel.imu_t[-1])
    ok[:f_lo] = False
    ok[f_hi:] = False
    margin = int(round(0.25 * tel.fps))
    ok[:margin] = False
    ok[F - margin:] = False
    seg = np.full(F, -1)
    for i, (a, b) in enumerate(tel.segments or [(0, F - 1)]):
        seg[int(a):int(b) + 1] = i
    w = _frame_rate_series(tel, fr)
    fa = float(tel.lens.fx) * width / float(tel.width)
    speed = np.linalg.norm(w, axis=1)
    d2 = np.zeros(F)
    d2[1:-1] = np.linalg.norm(w[2:] - 2 * w[1:-1] + w[:-2], axis=1)
    blur = fa * speed * ex
    pen = 1.0 / (1.0 + (blur / 4.0) ** 2)
    info = fa * d2 * 1e-3 * pen                       # px per ms of offset error, blur-penalised
    cs = lambda x: np.concatenate([[0.0], np.cumsum(x)])  # noqa: E731
    C_info, C_sp, C_ok = cs(info), cs(speed ** 2), cs((~ok).astype(float))
    cands = []
    stride = max(1, L // 6)
    for s in range(f_lo, f_hi - L, stride):
        e = s + L
        if C_ok[e] - C_ok[s] > 0 or seg[s] < 0 or seg[s] != seg[e - 1]:
            continue
        rate_rms = np.sqrt((C_sp[e] - C_sp[s]) / L)
        if rate_rms < np.deg2rad(3.0):
            continue
        score = (C_info[e] - C_info[s]) / L * min(1.0, rate_rms / 0.5)
        cands.append(dict(start=int(s), n=L, score=float(score), rate_rms_dps=float(np.rad2deg(rate_rms)),
                          info_px_per_ms=float((C_info[e] - C_info[s]) / L),
                          blur_px_med=float(np.median(blur[s:e])), exposure_ms_med=float(np.median(ex[s:e]) * 1e3)))
    if not cands:
        return []
    cands.sort(key=lambda c: -c['score'])
    chosen: list[dict] = []

    def far(c):
        return all(abs(c['start'] - o['start']) >= (3 * L) // 2 for o in chosen)
    # one per part first (temporal spread: drifts / exposure changes), then the best remaining
    edges = np.linspace(f_lo, f_hi, n + 1)
    for pi in range(n):
        best = [c for c in cands if edges[pi] <= c['start'] < edges[pi + 1] and far(c)]
        if best and best[0]['score'] >= 0.35 * cands[0]['score']:
            chosen.append(best[0])
    for c in cands:
        if len(chosen) >= n:
            break
        if far(c) and c not in chosen:
            chosen.append(c)
    return sorted(chosen[:n], key=lambda c: c['start'])


# ============================================================================ tracking

def _gyro_predictor(tel: Telemetry, lens: Lens, frames: Optional[np.ndarray] = None):
    """KLT initial guess from the gyro. `frames`: the frames that will be tracked -> their orientations are looked up
    once (tel.orientation_at re-signs the whole IMU series per call: ~20 ms at 1 kHz x 3 min, per frame)."""
    qf = {}
    if frames is not None and len(frames):
        fr = np.unique(np.asarray(frames, np.int64))
        qf = dict(zip(fr.tolist(), tel.orientation_at(tel.frame_t[fr])))

    def q_of(k):
        q = qf.get(int(k))
        return q if q is not None else tel.orientation_at(np.array([tel.frame_t[k]]))[0]

    def predict(ka: int, kb: int, pts: np.ndarray) -> np.ndarray:
        q = np.stack([q_of(ka), q_of(kb)])
        rel = qmul(qconj(q[1:2]), q[0:1])                 # R_b^T R_a
        r = unproject_robust(lens, pts.astype(np.float64))
        return lens.project(_qrot(np.repeat(rel, len(r), 0), r)).astype(np.float32)
    return predict


def track_frames(frames: Iterable[tuple[int, np.ndarray]], predict: Callable, max_points: int = 450,
                 grid: tuple = (12, 9), fb_thr: float = 0.5, border: int = 8, min_dist: int = 9,
                 cancel: Optional[Callable[[], bool]] = None) -> dict:
    """Gyro-predicted, forward-backward checked KLT through a frame sequence (a gap in k restarts the tracks).
    Returns observations dict(tid, k, xy) sorted by (tid, k), plus n_frames / per-frame counts."""
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
              minEigThreshold=1e-4)
    gx, gy = grid
    quota = max(2, int(np.ceil(max_points / (gx * gy))))
    pts = np.zeros((0, 2), np.float32)
    ids = np.zeros(0, np.int64)
    nid = 0
    prev_pyr = None
    prev_k = None
    O_t, O_k, O_xy = [], [], []
    nfr = 0
    for k, img in frames:
        if cancel is not None and nfr % 30 == 0 and cancel():
            raise InterruptedError('cancelled')
        img = np.ascontiguousarray(img)
        H, W = img.shape
        pyr = img      # (OpenCV 5's Python binding takes no prebuilt pyramids)
        if prev_pyr is not None and len(pts) and k == prev_k + 1:
            init = predict(prev_k, k, pts)
            p1, s1, _ = cv2.calcOpticalFlowPyrLK(prev_pyr, pyr, pts, init.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            p0b, s2, _ = cv2.calcOpticalFlowPyrLK(pyr, prev_pyr, p1, pts.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            good = (s1[:, 0] == 1) & (s2[:, 0] == 1) & (np.linalg.norm(p0b - pts, axis=1) < fb_thr)
            good &= (p1[:, 0] > border) & (p1[:, 0] < W - 1 - border) & (p1[:, 1] > border) & (p1[:, 1] < H - 1 - border)
            pts, ids = p1[good], ids[good]
        elif prev_k is None or k != prev_k + 1:
            pts, ids = np.zeros((0, 2), np.float32), np.zeros(0, np.int64)
        # top up: per-cell quota, away from existing points
        cw, ch = W / gx, H / gy
        if len(pts) < 0.7 * max_points or (nfr % 3 == 0 and len(pts) < 0.9 * max_points):
            cx = np.clip((pts[:, 0] / cw).astype(int), 0, gx - 1)
            cy = np.clip((pts[:, 1] / ch).astype(int), 0, gy - 1)
            cnt = np.zeros((gy, gx), int)
            np.add.at(cnt, (cy, cx), 1)
            s8 = 8
            small = np.full(((H + s8 - 1) // s8, (W + s8 - 1) // s8), 255, np.uint8)
            if len(pts):
                small[np.clip((pts[:, 1] / s8).astype(int), 0, small.shape[0] - 1),
                      np.clip((pts[:, 0] / s8).astype(int), 0, small.shape[1] - 1)] = 0
                small = cv2.erode(small, np.ones((3, 3), np.uint8))
            mask = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
            full = cnt >= quota
            if full.any():
                cm = cv2.resize((~full).astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = cv2.bitwise_and(mask, cm)
            mask[:border + 2] = 0
            mask[H - border - 2:] = 0
            mask[:, :border + 2] = 0
            mask[:, W - border - 2:] = 0
            need = int(np.sum(np.maximum(quota - cnt, 0)))
            c = cv2.goodFeaturesToTrack(img, maxCorners=int(2 * need + 10), qualityLevel=0.01, minDistance=min_dist,
                                        mask=mask, blockSize=7) if need > 0 else None
            if c is not None and len(c):
                c = c.reshape(-1, 2)
                ncx = np.clip((c[:, 0] / cw).astype(int), 0, gx - 1)
                ncy = np.clip((c[:, 1] / ch).astype(int), 0, gy - 1)
                cell = ncy * gx + ncx
                order = np.argsort(cell, kind='stable')          # keeps quality order inside a cell
                cs_ = cell[order]
                first = np.r_[0, np.flatnonzero(np.diff(cs_)) + 1]
                rank = np.arange(len(cs_)) - np.repeat(first, np.diff(np.r_[first, len(cs_)]))
                room = np.maximum(quota - cnt.ravel(), 0)[cs_]
                take = order[rank < room]
                if len(take):
                    nw = c[take].astype(np.float32)
                    nw = cv2.cornerSubPix(img, nw.reshape(-1, 1, 2), (4, 4), (-1, -1),
                                          (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 15, 0.02)).reshape(-1, 2)
                    pts = np.concatenate([pts, nw]).astype(np.float32)
                    ids = np.concatenate([ids, nid + np.arange(len(nw))])
                    nid += len(nw)
        O_t.append(ids.copy())
        O_k.append(np.full(len(ids), int(k), np.int64))
        O_xy.append(pts.astype(np.float64).copy())
        prev_pyr, prev_k = pyr, int(k)
        nfr += 1
    if not O_t:
        return dict(tid=np.zeros(0, np.int64), k=np.zeros(0, np.int64), xy=np.zeros((0, 2)), n_frames=0)
    tid = np.concatenate(O_t)
    kk = np.concatenate(O_k)
    xy = np.concatenate(O_xy)
    o = np.lexsort((kk, tid))
    return dict(tid=tid[o], k=kk[o], xy=xy[o], n_frames=nfr)


def build_deltas(tid: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observations sorted by (tid, k) -> (pair_a, pair_b, delta (m,2) pair indices): span-1 pairs and the
    three-frame deltas (pair (k+1,k+2) minus pair (k,k+1) of the same track)."""
    n = len(tid)
    if n < 3:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 2), np.int64)
    link = (tid[1:] == tid[:-1]) & (k[1:] == k[:-1] + 1)          # obs i -> i+1
    pa = np.flatnonzero(link)
    pid = np.full(n, -1, np.int64)
    pid[pa] = np.arange(len(pa))
    tri = np.flatnonzero(link[:-1] & link[1:])                     # obs i, i+1, i+2
    return pa, pa + 1, np.stack([pid[tri], pid[tri + 1]], axis=1)


# ============================================================================ the model

class _Model:
    """Delta residuals for a parameter vector x = (offset ms, readout %, focal %, exposure slope, box width %)."""

    NP = 5

    def __init__(self, tel: Telemetry, lens_a: Lens, H: int, obs: dict, pa, pb, dl, win_of_obs, spans,
                 lf_hz: float, exposure_avg: bool, e_ref: float, box_mode: str = 'engine', box_scale: float = 1.0):
        self.lens0 = lens_a
        self.H = H
        k = obs['k']
        self.xy = obs['xy']
        self.ft = tel.frame_t[k]
        ex = np.asarray(tel.exposure_s, np.float64)
        ex = np.broadcast_to(ex, (tel.n_frames,)) if ex.ndim == 0 else np.nan_to_num(ex)
        self.e = ex[k]
        self.e_ref = e_ref
        if not (exposure_avg and tel.has_highrate):
            self.w_box = np.zeros_like(self.e)
        else:
            self.w_box = box_scale * (exposure_avg_window(self.e) if box_mode == 'engine' else self.e)
        self.row = (self.xy[:, 1] + 0.5) / H - 0.5
        self.tr0 = float(tel.readout_s)
        self.pa, self.pb, self.dl = pa, pb, dl
        self.win = win_of_obs
        self.orients, self.orients_lf, self.hvec = [], [], []
        for (a, b) in spans:
            i0 = max(0, int(np.searchsorted(tel.imu_t, a)) - 2)
            i1 = min(len(tel.imu_t), int(np.searchsorted(tel.imu_t, b)) + 2)
            t, q = tel.imu_t[i0:i1], tel.imu_q[i0:i1]
            self.orients.append(_BoxOrient(t, q))
            if lf_hz:
                q_lf, h = _lowpass_orientation(t, q, lf_hz, return_h=True)
                self.orients_lf.append(_BoxOrient(t, q_lf))
                self.hvec.append(_BoxVec(t, h))
            else:
                self.orients_lf.append(None)
                self.hvec.append(None)
        self.win_idx = [np.flatnonzero(win_of_obs == w) for w in range(len(spans))]
        self._rays = {}

    def rays(self, s: float) -> np.ndarray:
        key = round(float(s), 9)
        if key not in self._rays:
            if len(self._rays) > 6:
                self._rays.clear()
            L = self.lens0
            self._rays[key] = unproject_robust(Lens(L.model, L.fx * s, L.fy * s, L.cx, L.cy, L.k, L.width, L.height),
                                               self.xy)
        return self._rays[key]

    def quats(self, x: np.ndarray, lf: bool = False, hf_ms: Optional[float] = None) -> np.ndarray:
        """Row orientations. lf: the gyro HF (> lf_hz) removed. hf_ms: the LF part at the offset x[0] and the HF part
        at the offset hf_ms instead (q = q_lf(t + x0) (x) exp(h(t + hf)), box-averaged): a pure timing error shifts
        both; a parallax-coupled bias of the (maneuver-driven) LF part does not move the HF optimum."""
        t = self.ft + x[0] * 1e-3 + x[3] * (self.e - self.e_ref) + self.tr0 * (1 + x[1] * 1e-2) * self.row
        wb = self.w_box * max(0.0, 1.0 + x[4] * 1e-2) if len(x) > 4 and x[4] else self.w_box
        q = np.empty((len(t), 4))
        ors = self.orients_lf if (lf or hf_ms is not None) else self.orients
        for w, idx in enumerate(self.win_idx):
            if len(idx):
                q[idx] = ors[w](t[idx], wb[idx])
                if hf_ms is not None:
                    h = self.hvec[w](t[idx] + (hf_ms - x[0]) * 1e-3, wb[idx])
                    q[idx] = qmul(q[idx], qexp(h))
        return q

    def pair_res(self, x: np.ndarray, lf: bool = False, hf_ms: Optional[float] = None) -> np.ndarray:
        s = 1.0 + x[2] * 1e-2
        b = self.rays(s)
        q = self.quats(x, lf, hf_ms)
        u = _qrot(q[self.pa], b[self.pa])
        rc = _qrot(qconj(q[self.pb]), u)
        L = self.lens0
        pred = Lens(L.model, L.fx * s, L.fy * s, L.cx, L.cy, L.k, L.width, L.height).project(rc)
        r = pred - self.xy[self.pb]
        r[rc[:, 2] < 0.05] = 0.0
        return r

    def delta_res(self, x: np.ndarray, lf: bool = False, hf_ms: Optional[float] = None) -> np.ndarray:
        r = self.pair_res(x, lf, hf_ms)
        return r[self.dl[:, 1]] - r[self.dl[:, 0]]


def _cauchy_cost(d: np.ndarray, c: float) -> float:
    return float(np.mean(np.log1p(np.sum(d * d, axis=1) / (c * c))))


def _vertex(xs: np.ndarray, ys: np.ndarray) -> tuple[float, bool]:
    i = int(np.argmin(ys))
    if i == 0 or i == len(ys) - 1:
        return float(xs[i]), False
    a, b, _ = np.polyfit(xs[i - 1:i + 2], ys[i - 1:i + 2], 2)
    if a <= 0:
        return float(xs[i]), True
    v = -b / (2 * a)
    return float(np.clip(v, xs[i - 1], xs[i + 1])), True


# ============================================================================ the fit

def _compact(o: dict, pa: np.ndarray, pb: np.ndarray, dl: np.ndarray):
    """Keep only the pairs / observations that the deltas `dl` use (indices remapped)."""
    need_p = np.unique(dl)
    remap_p = np.full(len(pa), -1, np.int64)
    remap_p[need_p] = np.arange(len(need_p))
    dl = remap_p[dl]
    pa, pb = pa[need_p], pb[need_p]
    need_o = np.unique(np.concatenate([pa, pb]))
    remap_o = np.full(len(o['k']), -1, np.int64)
    remap_o[need_o] = np.arange(len(need_o))
    return {kk: v[need_o] for kk, v in o.items()}, remap_o[pa], remap_o[pb], dl


PNAMES = ('offset_ms', 'readout_pct', 'focal_pct', 'exposure_slope', 'box_pct')
NUISANCE = PNAMES[1:]
_PRIOR_SIG = np.array([10.0, 20.0, 3.0, 0.5, 60.0])     # priors pull readout/focal/slope/box to the metadata (0)
_STEPS = np.array([0.02, 0.2, 0.1, 0.02, 2.0])          # numeric-derivative steps
_TOL = np.array([0.002, 0.02, 0.01, 0.002, 0.5])        # GN convergence


def fit_timing(tel: Telemetry, obs: dict, H: int, width: int, windows: list[dict],
               params: Optional[TimecalParams] = None, notes: Optional[dict] = None) -> dict:
    """Fit the clip's timing to tracked observations (dict tid, k, xy, win at analysis width; windows: list of
    dict(start, n)).

    1. coarse offset sweep with the gyro HF removed, 2. full-band fine sweep, 3. joint robust Gauss-Newton on
    (offset, readout, focal, exposure slope, exposure-box width) -> `joint` (reported); the nuisance parameters that
    are clearly significant are selected (_decide_nuisance), 4. CONDITIONAL fit: offset + the selected nuisances
    only, the rest fixed at the metadata (an offset fitted jointly with, e.g., a focal that is then not applied is
    biased: OA4 0012 joint +0.068 ms with focal -3.5 %, conditional +0.020 ms), 5. uncertainty from 1-s block
    jackknife, per-window estimates (outlier windows dropped), and validation: in-sample cost gain and the cost of
    each held-out fold (window) at the parameters fitted WITHOUT it (linearised), vs the metadata.
    If the offset fails its checks but selected nuisances exist, 4-5 are repeated for the nuisances alone
    (`nuisance_only`). Returns a result dict; _decide() turns it into what gets applied."""
    p = params or TimecalParams()
    t0 = time.perf_counter()
    ws = width / 960.0
    c_px = p.cauchy_px * ws
    c2 = c_px * c_px
    la = tel.lens.scaled(width / float(tel.width))
    lens_a = Lens(la.model, la.fx, la.fy, la.cx, la.cy, la.k, int(width), int(H))
    lim = _max_theta_d(lens_a)                                     # reliable KB4 domain
    rn = np.hypot((obs['xy'][:, 0] - lens_a.cx) / lens_a.fx, (obs['xy'][:, 1] - lens_a.cy) / lens_a.fy)
    keep = rn < lim
    o = {kk: v[keep] for kk, v in obs.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(keep)}
    pa, pb, dl = build_deltas(o['tid'], o['k'])
    res = dict(n_obs=int(len(o['k'])), n_pairs=int(len(pa)), n_deltas_all=int(len(dl)))
    if len(dl) < p.min_deltas:
        res['status'] = f'too few delta residuals ({len(dl)} < {p.min_deltas})'
        return res
    rng = np.random.default_rng(0)
    # subsample whole TRACKS (consecutive deltas share pairs and observations: ~1 observation per delta instead of
    # ~4 for random deltas -> 3-4x cheaper evaluations for the same number of deltas)
    d_tid = o['tid'][pa[dl[:, 0]]]
    ut, inv = np.unique(d_tid, return_inverse=True)
    perm = rng.permutation(len(ut))
    rank = np.empty(len(ut), np.int64)
    rank[perm] = np.arange(len(ut))
    cnt = np.bincount(inv, minlength=len(ut))[perm]
    cum = np.cumsum(cnt)
    d_rank = rank[inv]

    def n_tracks(nmax):
        return int(np.searchsorted(cum, nmax)) + 1
    if len(dl) > p.max_deltas:
        keep = d_rank < n_tracks(p.max_deltas)
        dl, d_rank = dl[keep], d_rank[keep]
    sub = np.flatnonzero(d_rank < n_tracks(p.coarse_deltas)) if len(dl) > p.coarse_deltas else None
    oc, pac, pbc, dlc = _compact(o, pa, pb, dl if sub is None else dl[sub])
    o, pa, pb, dl = _compact(o, pa, pb, dl)
    ex = np.asarray(tel.exposure_s, np.float64)
    ex = np.broadcast_to(ex, (tel.n_frames,)) if ex.ndim == 0 else np.nan_to_num(ex)
    spans = []
    for w in windows:
        f1 = min(tel.n_frames - 1, w['start'] + w['n'] - 1)
        e_max = float(np.max(ex[w['start']:f1 + 1], initial=0.0))
        pad = 0.5 * tel.readout_s + 0.03 + (p.coarse_ms + p.fine_ms) * 1e-3 + 1.75 * min(e_max, 0.05)
        spans.append((tel.frame_t[w['start']] - pad, tel.frame_t[f1] + pad))
    k_mid = o['k'][pb[dl[:, 0]]]                                    # middle frame of each delta
    e_mid = ex[k_mid]
    e_ref = float(np.median(e_mid))
    M = _Model(tel, lens_a, H, o, pa, pb, dl, o['win'], spans, 0.0, p.exposure_avg, e_ref, p.box_mode, p.box_scale)
    Mc = _Model(tel, lens_a, H, oc, pac, pbc, dlc, oc['win'], spans, p.lf_hz, p.exposure_avg, e_ref, p.box_mode,
                p.box_scale)
    res.update(n_deltas=int(len(dl)), exposure_ref_ms=e_ref * 1e3,
               exposure_ms_pct=np.percentile(e_mid * 1e3, [5, 50, 95]).round(3).tolist())
    NP = _Model.NP
    x0 = np.zeros(NP)

    def xoff(v):
        x = x0.copy()
        x[0] = v
        return x

    # ---- 1. coarse offset grid, gyro HF removed (unimodal)
    grid = np.arange(-p.coarse_ms, p.coarse_ms + 1e-9, p.coarse_step_ms)
    cc = np.array([_cauchy_cost(Mc.delta_res(xoff(g), lf=True), c_px) for g in grid])
    off1, interior1 = _vertex(grid, cc)
    res['coarse'] = dict(offset_ms=off1, interior=interior1, cost_min=float(cc.min()),
                         cost_at_0=float(cc[int(np.argmin(np.abs(grid)))]),
                         contrast=float((np.median(cc) - cc.min()) / max(cc.min(), 1e-9)))
    # ---- 2. full-band fine sweep around it (the readout fit is aliased at a wrong offset: offset first)
    fgrid = off1 + np.arange(-p.fine_ms, p.fine_ms + 1e-9, p.fine_step_ms)
    cf = np.array([_cauchy_cost(Mc.delta_res(xoff(g)), c_px) for g in fgrid])
    off2, interior2 = _vertex(fgrid, cf)
    res['fine'] = dict(offset_ms=off2, interior=interior2, cost_min=float(cf.min()), cost_curve=cf.round(5).tolist(),
                       grid0=float(fgrid[0]))
    # (Mc is kept for the HF-consistency check below)

    lo = np.array([-(p.coarse_ms + p.fine_ms), -100 * p.readout_bound_rel, -100 * p.focal_bound_rel, -p.slope_bound,
                   float(p.box_bound[0])])
    hi = np.array([p.coarse_ms + p.fine_ms, 100 * p.readout_bound_rel, 100 * p.focal_bound_rel, p.slope_bound,
                   float(p.box_bound[1])])

    def lin(x, act):
        d = M.delta_res(x)
        wt = 1.0 / (1.0 + np.sum(d * d, axis=1) / c2)
        J = np.empty((len(d), 2, len(act)))
        for j, a in enumerate(act):
            xp = x.copy()
            xp[a] += _STEPS[a]
            J[:, :, j] = (M.delta_res(xp) - d) / _STEPS[a]
        return d, wt, J

    def gn(x, act):
        """Robust (Cauchy IRLS) Gauss-Newton on the parameters `act`; the others stay at x."""
        act = np.asarray(act, np.int64)
        ps = _PRIOR_SIG[act]
        pm = (act > 0).astype(float)
        x = np.clip(x, lo, hi)
        cur = lin(x, act)
        c_cur = _cauchy_cost(cur[0], c_px)
        hist = []
        for it in range(p.gn_iters):
            d, wt, J = cur
            A = np.einsum('n,nci,ncj->ij', wt, J, J) / c2 + np.diag(1.0 / ps ** 2)
            g = np.einsum('n,nci,nc->i', wt, J, d) / c2 + pm * x[act] / ps ** 2
            step = -np.linalg.solve(A, g)
            if np.all(np.abs(step) < _TOL[act]):
                hist.append(dict(it=it, converged=True, cost=c_cur))
                break
            moved = False
            for damp in (1.0, 0.3):
                x_new = x.copy()
                x_new[act] = np.clip(x[act] + damp * step, lo[act], hi[act])
                new = lin(x_new, act)
                c_new = _cauchy_cost(new[0], c_px)
                if c_new <= c_cur + 1e-7:
                    x, cur, c_cur, moved = x_new, new, c_new, True
                    break
            hist.append(dict(it=it, x=np.round(x, 5).tolist(), cost=c_cur, moved=moved))
            if not moved:
                break
        return x, cur, hist

    # 1-s blocks (jackknife) and validation folds (windows; one window: every third block)
    t_mid = tel.frame_t[k_mid]
    wid = o['win'][pb[dl[:, 0]]]
    blk = np.zeros(len(dl), np.int64)
    bj = np.zeros(len(dl), np.int64)
    # >= 3 jackknife blocks per window (else the per-window sigma is undefined and the agreement check is vacuous)
    block_s = min(p.block_s, min(w['n'] for w in windows) / float(tel.fps) / 3.0)
    res['block_s'] = block_s
    for w in np.unique(wid):
        m = wid == w
        bj[m] = np.floor((t_mid[m] - t_mid[m].min()) / block_s).astype(np.int64)
        blk[m] = int(w) * 100000 + bj[m]
    ublk, bi = np.unique(blk, return_inverse=True)
    nb = len(ublk)
    bwin = ublk // 100000
    n_win_data = len(np.unique(bwin))
    fold_of_delta = wid if n_win_data >= 2 else (bj % 3)
    fold_blk = bwin if n_win_data >= 2 else (ublk % 100000) % 3
    cost_x0_d = None

    def stage(x_start, act, name):
        """GN on `act` from x_start, then jackknife sigmas, per-window offsets (outlier windows dropped), the final
        estimate from the agreeing windows, and the held-out validation of that estimate vs the metadata."""
        nonlocal cost_x0_d
        act = np.asarray(act, np.int64)
        x, cur, hist = gn(x_start.copy(), act)
        d, wt, J = cur
        Ab = np.zeros((nb, len(act), len(act)))
        gb = np.zeros((nb, len(act)))
        np.add.at(Ab, bi, np.einsum('n,nci,ncj->nij', wt, J, J) / c2)
        np.add.at(gb, bi, np.einsum('n,nci,nc->ni', wt, J, d) / c2)
        Pm = np.diag(1.0 / _PRIOR_SIG[act] ** 2)
        gP = (act > 0) * x[act] / _PRIOR_SIG[act] ** 2

        def solve_blocks(sel, jack=True):
            A_, g_ = Ab[sel].sum(0) + Pm, gb[sel].sum(0) + gP
            xs = x[act] - np.linalg.solve(A_, g_)
            idx = np.flatnonzero(sel)
            if jack and len(idx) >= 3:
                ests = np.array([x[act] - np.linalg.solve(A_ - Ab[b], g_ - gb[b]) for b in idx])
                sj = np.sqrt((len(idx) - 1) / len(idx) * np.sum((ests - ests.mean(0)) ** 2, axis=0))
            else:
                sj = np.full(len(act), np.inf)
            try:
                sf = np.sqrt(np.diag(np.linalg.inv(A_)))
            except np.linalg.LinAlgError:
                sf = np.full(len(act), np.inf)
            return xs, sj, sf

        per_win = []
        for w in range(len(windows)):
            sel = bwin == w
            if not sel.any():
                continue
            xw, sjw, sfw = solve_blocks(sel)
            j0 = int(np.flatnonzero(act == 0)[0]) if 0 in act else None
            per_win.append(dict(window=w, start=int(windows[w]['start']),
                                t_s=float(tel.frame_pts[windows[w]['start']]),
                                offset_ms=float(xw[j0]) if j0 is not None else float(x[0]),
                                offset_sigma_ms=float(max(sjw[j0], sfw[j0])) if j0 is not None else float('nan'),
                                estimate=dict(zip([PNAMES[a] for a in act], np.round(xw, 5).tolist())),
                                n_deltas=int(np.sum(wid == w)), exposure_ms=float(np.median(e_mid[wid == w]) * 1e3)))
        # windows that agree with the weighted median offset; an outlier window is excluded (scene-dependent bias:
        # a moving subject, a window full of blur), the estimate is re-solved from the rest
        used_w = [pw['window'] for pw in per_win]
        if len(per_win) >= 3 and 0 in act:
            offs = np.array([pw['offset_ms'] for pw in per_win])
            sg = np.array([pw['offset_sigma_ms'] for pw in per_win])
            wts = 1.0 / np.maximum(np.nan_to_num(sg, nan=1e3, posinf=1e3), 0.01) ** 2
            o_ = np.argsort(offs)
            cw = np.cumsum(wts[o_]) / wts.sum()
            med = float(offs[o_][min(len(offs) - 1, int(np.searchsorted(cw, 0.5)))])
            agree = np.abs(offs - med) <= np.maximum(p.window_agree_ms, 4 * np.nan_to_num(sg, nan=np.inf))
            used_w = [pw['window'] for pw, a in zip(per_win, agree) if a]
        for pw in per_win:
            pw['used'] = pw['window'] in used_w
        selu = np.isin(bwin, used_w)
        xs, sj, sf = solve_blocks(selu)
        x_fin = x.copy()
        x_fin[act] = np.clip(xs, lo[act], hi[act])
        sigma = np.full(NP, np.nan)
        sigma[act] = np.maximum(sj, sf)
        # ---- validation: in-sample gain and held-out folds (fitted without the fold, linearised) vs metadata
        if cost_x0_d is None:
            d0 = M.delta_res(x0)
            cost_x0_d = np.log1p(np.sum(d0 * d0, axis=1) / c2)
        dx = M.delta_res(x_fin)
        cost_x_d = np.log1p(np.sum(dx * dx, axis=1) / c2)
        gain_in = float(1.0 - cost_x_d.mean() / max(cost_x0_d.mean(), 1e-12))
        held = []
        for f in np.unique(fold_blk):
            tr = selu & (fold_blk != f)
            te = fold_of_delta == f
            if not tr.any() or not te.any() or (n_win_data >= 2 and f not in used_w):
                continue
            xf = x.copy()
            xf[act] = np.clip(solve_blocks(tr, jack=False)[0], lo[act], hi[act])
            dfv = M.delta_res(xf)
            c_f = float(np.log1p(np.sum(dfv[te] ** 2, axis=1) / c2).mean())
            c_0 = float(cost_x0_d[te].mean())
            held.append(dict(fold=int(f), gain=float(1.0 - c_f / max(c_0, 1e-12)),
                             x=dict(zip([PNAMES[a] for a in act], np.round(xf[act], 4).tolist()))))
        hg = np.array([h['gain'] for h in held])
        val = dict(gain_in=gain_in, heldout=held,
                   heldout_pos_frac=float(np.mean(hg > 0)) if len(hg) else float('nan'),
                   heldout_mean=float(hg.mean()) if len(hg) else float('nan'))
        return dict(name=name, active=[PNAMES[a] for a in act], estimate=dict(zip(PNAMES, x_fin.tolist())),
                    estimate_gn=dict(zip(PNAMES, x.tolist())), sigma=dict(zip(PNAMES, sigma.tolist())),
                    sigma_jackknife=dict(zip([PNAMES[a] for a in act], sj.tolist())),
                    sigma_formal=dict(zip([PNAMES[a] for a in act], sf.tolist())),
                    per_window=per_win, windows_used=used_w, n_blocks=int(np.sum(selu)), gn_history=hist,
                    cost_nominal=float(cost_x0_d.mean()), cost_final=float(cost_x_d.mean()), validation=val,
                    delta_med_px_final=float(np.median(np.linalg.norm(dx, axis=1))))

    # ---- 3. joint fit (reported; selects the nuisance parameters that are clearly there)
    spread = float(np.std(e_mid))
    w_med = float(np.median(M.w_box[M.pb[M.dl[:, 0]]])) if len(M.w_box) else 0.0
    # focal only when it may be applied: a free focal absorbs exposure-model error and biases the other parameters
    # (OA4 0012: joint offset +0.068 ms with focal -3.5 %, conditional +0.022 ms)
    active = np.array([True, bool(p.fit_readout), bool(p.fit_focal and p.apply_focal),
                       bool(p.fit_exposure and spread >= p.exposure_spread_min_s),
                       bool(p.fit_box and w_med >= p.box_fit_min_s)])
    res['active'] = dict(zip(PNAMES, active.tolist()))
    res['exposure_spread_ms'] = spread * 1e3
    res['box_window_med_ms'] = w_med * 1e3
    joint = stage(xoff(off2), np.flatnonzero(active), 'joint')
    res['joint'] = joint
    nd = _decide_nuisance(joint, res['active'], p)
    res['nuisance_decision'] = nd
    sel_n = [i for i, nm in enumerate(PNAMES) if i > 0 and nd.get(nm)]
    # ---- 4./5. conditional fit: offset + the selected nuisances, the others at the metadata
    xs0 = xoff(joint['estimate']['offset_ms'])
    for i in sel_n:
        xs0[i] = joint['estimate'][PNAMES[i]]
    cond = stage(xs0, [0] + sel_n, 'conditional')
    for kk in ('estimate', 'estimate_gn', 'sigma', 'sigma_jackknife', 'sigma_formal', 'per_window', 'windows_used',
               'n_blocks', 'gn_history', 'cost_nominal', 'cost_final', 'validation', 'delta_med_px_final'):
        res[kk] = cond[kk]
    res['status'] = 'ok'
    # ---- 6. HF consistency: with the LF part at the conditional offset, where does the gyro's HF part (> lf_hz: prop
    #         vibration) fit best? A real timing error shifts every frequency; a bias of the maneuver-driven LF part
    #         (translation / parallax coupled to the attitude, O3: the same 2-s windows said -0.81 ms, the 5-s
    #         windows around them -0.27) leaves the HF optimum at the true offset.
    xc = np.array([res['estimate'][nm] for nm in PNAMES])
    hg = xc[0] + np.arange(-p.hf_check_ms, p.hf_check_ms + 1e-9, p.fine_step_ms)
    ch = np.array([_cauchy_cost(Mc.delta_res(xc, hf_ms=g), c_px) for g in hg])
    off_hf, int_hf = _vertex(hg, ch)
    res['hf_check'] = dict(offset_ms=off_hf, interior=int_hf, cost_min=float(ch.min()),
                           cost_hf_at_0=_cauchy_cost(Mc.delta_res(xc, hf_ms=0.0), c_px),
                           contrast=float((np.median(ch) - ch.min()) / max(ch.min(), 1e-9)),
                           curve=ch.round(5).tolist(), grid0=float(hg[0]))
    res['hf_check']['gain'] = float(1.0 - res['hf_check']['cost_min'] / max(res['hf_check']['cost_hf_at_0'], 1e-12))
    ok_off, why = _offset_checks(res, p)
    res['offset_checks'] = dict(ok=ok_off, reasons=why)
    if not ok_off and sel_n:
        xs1 = x0.copy()
        for i in sel_n:
            xs1[i] = joint['estimate'][PNAMES[i]]
        nu = stage(xs1, sel_n, 'nuisance_only')
        res['nuisance_only'] = {kk: nu[kk] for kk in ('active', 'estimate', 'sigma', 'validation', 'cost_final')}
    d0 = M.delta_res(x0)
    res.update(delta_med_px_nominal=float(np.median(np.linalg.norm(d0, axis=1))),
               cost_gain=float(1 - res['cost_final'] / max(res['cost_nominal'], 1e-12)),
               fit_s=time.perf_counter() - t0, readout_meta_ms=float(tel.readout_s) * 1e3)
    return res


def _decide_nuisance(joint: dict, active: dict, p: TimecalParams) -> dict:
    """Which nuisance parameters are clearly significant in the joint fit (then fitted conditionally and applied if
    the result validates). The readout / focal / slope / box need a clear, significant change (the old calib's
    +11.8 % readout on DJI_0025 raised eval jello 0.56 -> 0.96)."""
    e, s = joint['estimate'], joint['sigma']
    dec = dict(readout_pct=False, focal_pct=False, exposure_slope=False, box_pct=False)
    if active.get('readout_pct'):
        r, sr = e['readout_pct'] / 100.0, s['readout_pct'] / 100.0
        dec['readout_pct'] = bool(abs(r) >= p.readout_min_rel and sr <= p.readout_sigma_max_rel and abs(r) >= 4 * sr
                                  and abs(r) <= p.readout_bound_rel * 0.98)
    if p.apply_focal and active.get('focal_pct'):
        f, sf = e['focal_pct'] / 100.0, s['focal_pct'] / 100.0
        dec['focal_pct'] = bool(abs(f) >= p.focal_min_rel and sf <= p.focal_sigma_max_rel and abs(f) >= 4 * sf
                                and abs(f) <= p.focal_bound_rel * 0.98)
    if active.get('exposure_slope'):
        g, sg = e['exposure_slope'], s['exposure_slope']
        dec['exposure_slope'] = bool(abs(g) >= p.slope_min and sg <= p.slope_sigma_max and abs(g) >= 4 * sg
                                     and abs(g) <= 0.98 * p.slope_bound)
    if active.get('box_pct'):
        b, sb = e['box_pct'], s['box_pct']
        lo_b, hi_b = p.box_bound
        dec['box_pct'] = bool(abs(b) >= p.box_min_pct and sb <= p.box_sigma_max_pct and abs(b) >= 3 * sb
                              and 0.98 * lo_b <= b <= 0.98 * hi_b)
    return dec


def _offset_checks(res: dict, p: TimecalParams) -> tuple[bool, list]:
    """Statistical checks of the conditional offset: sigma, physical bound, inside the coarse basin, windows agree."""
    why = []
    e, s = res['estimate'], res['sigma']
    off, so = e['offset_ms'], s['offset_ms']
    smax = max(p.offset_sigma_max_ms, p.offset_sigma_rel * abs(off))
    if not np.isfinite(so) or so > smax:
        why.append(f'offset sigma {so:.3f} ms > {smax:.3f}')
    if abs(off) > p.offset_bound_ms:
        why.append(f'offset {off:+.2f} ms outside +-{p.offset_bound_ms}')
    if not res['fine']['interior'] or abs(off - res['coarse']['offset_ms']) > p.fine_ms + 0.25:
        why.append('offset left the coarse basin (fine sweep minimum on its edge)')
    npw = len(res['per_window'])
    need = max(min(p.min_windows_agree, npw), int(np.ceil(2 * npw / 3)))
    if len(res['windows_used']) < need:
        why.append('windows disagree: ' + ', '.join(f"{w['offset_ms']:+.3f}+-{w['offset_sigma_ms']:.3f}"
                                                   for w in res['per_window']))
    hc = res.get('hf_check')
    if p.hf_check and hc is not None:
        tol = max(p.hf_agree_ms, p.hf_agree_rel * abs(off))
        if not hc['interior'] or abs(hc['offset_ms'] - off) > tol or hc['gain'] < p.hf_gain_min:
            why.append(f"the gyro's HF part does not confirm it (HF optimum {hc['offset_ms']:+.3f} ms, HF gain "
                       f"{hc['gain'] * 100:.2f} %)")
    return (not why), why


def _validated(val: Optional[dict], p: TimecalParams) -> tuple[bool, list]:
    """Practical significance + out-of-sample check of a fitted parameter vector vs the metadata."""
    if not val:
        return False, ['no validation']
    why = []
    if not val['gain_in'] >= p.gain_min:
        why.append(f"in-sample cost gain {val['gain_in'] * 100:.2f} % < {p.gain_min * 100:.2f} % (not worth applying)")
    if val['heldout']:
        if not (val['heldout_pos_frac'] >= p.heldout_frac and val['heldout_mean'] >= p.heldout_gain_min):
            why.append('held-out folds do not confirm it: ' + ', '.join(f"{h['gain'] * 100:+.2f} %"
                                                                     for h in val['heldout']))
    else:
        why.append('no held-out fold')
    return (not why), why


def _decide(res: dict, tel: Telemetry, p: TimecalParams) -> dict:
    """What gets applied (the rest keeps the metadata value): the conditional vector (offset + selected nuisances)
    when the offset passes its statistical checks and the vector validates; else the nuisance-only vector when it
    validates; else nothing."""
    dec = dict(offset=False, readout=False, focal=False, exposure_slope=False, box=False, source=None, reasons=[])
    if res.get('status') != 'ok':
        dec['reasons'].append(res.get('status', 'no fit'))
        return dec
    nd = res.get('nuisance_decision') or {}
    keymap = dict(readout_pct='readout', focal_pct='focal', exposure_slope='exposure_slope', box_pct='box')
    ok_off, why = _offset_checks(res, p)
    if ok_off:
        v_ok, vwhy = _validated(res.get('validation'), p)
        if v_ok:
            dec['offset'] = True
            dec['source'] = 'conditional'
            for k, nm in keymap.items():
                dec[nm] = bool(nd.get(k))
            return dec
        dec['reasons'] += vwhy
    else:
        dec['reasons'] += why
    nu = res.get('nuisance_only')
    if nu:
        v_ok, vwhy = _validated(nu.get('validation'), p)
        if v_ok:
            dec['source'] = 'nuisance_only'
            for k, nm in keymap.items():
                dec[nm] = bool(nd.get(k))
        else:
            dec['reasons'] += ['nuisance-only: ' + w for w in vwhy]
    return dec


def apply_exposure_slope(tel: Telemetry, tm: TimeModel) -> Telemetry:
    """Telemetry with frame_t shifted by the calibrated exposure slope (TimeModel has no exposure term):
    frame_t[k] += slope * (e_k - e_ref). No-op unless tm.notes['timecal'] applied one."""
    from dataclasses import replace
    n = (tm.notes or {}).get('timecal', {})
    sl = n.get('applied', {}).get('exposure_slope')
    if not sl:
        return tel
    ex = np.asarray(tel.exposure_s, np.float64)
    if ex.ndim == 0:
        return tel
    shift = float(sl) * (np.nan_to_num(ex) - float(n['applied']['exposure_ref_s']))
    return replace(tel, frame_t=np.asarray(tel.frame_t, np.float64) + shift)


# ============================================================================ public entry

def calibrate_timing(tel: Telemetry, *, stream=None, video: Optional[str] = None,
                     frame_reader: Optional[Callable] = None, params: Optional[TimecalParams] = None,
                     cancel: Optional[Callable[[], bool]] = None,
                     progress: Optional[Callable[[float, str], None]] = None,
                     obs: Optional[dict] = None, windows: Optional[list] = None,
                     debug: Optional[dict] = None) -> TimeModel:
    """Self-calibrate the clip's picture <-> gyro timing. Frames come from `stream` (a FrameStream-like object with
    .frames(records, cancel) yielding (k, uint8 luma at params.width)), else frame_reader(f0, n) -> (idx, frames),
    else a FrameStream opened on `video`. `obs`/`windows`: precomputed tracks (dict tid, k, xy, win, H, W) for
    studies; `debug` (a dict) receives the tracks. Always returns a TimeModel; parameters that are not confidently
    observable, or whose correction does not measurably lower the delta cost (in sample and on held-out windows),
    stay at the metadata (offset 0, readout None, focal 1, exposure_scale 1). The exposure slope, when applied, is
    not a TimeModel field: apply it with apply_exposure_slope(tel, tm)."""
    p = params or TimecalParams()
    t_all = time.perf_counter()
    notes: dict = dict(version=TIMECAL_VERSION, params=asdict(p), camera=tel.camera)
    tm_default = lambda: TimeModel(notes={'timecal': notes})  # noqa: E731
    if tel.eis_baked:
        notes['status'] = 'skipped: in-camera EIS baked into the picture'
        return tm_default()
    if not tel.has_highrate or not tel.readout_s or tel.readout_s <= 0:
        notes['status'] = 'skipped: needs the high-rate gyro and a readout time'
        return tm_default()
    t_dec = 0.0
    if obs is None:
        # short clips: shorter windows so that n_windows still fit (n windows need (1.5 n - 0.5) x their length)
        usable_s = max(0.0, tel.n_frames / tel.fps - 0.6)
        win_s = min(p.window_s, max(p.min_window_s, usable_s / (1.5 * p.n_windows - 0.5)))
        # run-time budget: the calibration decodes + tracks ~budget_frac of the clip (>= n x min_window_s)
        if p.budget_frac and p.budget_frac > 0:
            win_s = min(win_s, max(p.min_window_s, p.budget_frac * tel.n_frames / tel.fps / p.n_windows))
        L = int(round(win_s * tel.fps))
        windows = choose_windows(tel, p.n_windows, L, p.width, guard_s=0.03 + p.coarse_ms * 1e-3)
        notes['windows'] = windows
        if not windows:
            notes['status'] = 'no window with enough rotation'
            return tm_default()
        records = np.concatenate([np.arange(w['start'], w['start'] + w['n']) for w in windows])
        win_of_frame = {int(k): wi for wi, w in enumerate(windows) for k in range(w['start'], w['start'] + w['n'])}
        own = None
        if stream is None and frame_reader is None:
            from .framestream import FrameStream
            own = stream = FrameStream(video, p.width, lanes=2, block=150)
        try:
            if stream is not None:
                src = stream.frames(records, cancel=cancel)
            else:
                def _gen():
                    for w in windows:
                        idx, fr = frame_reader(w['start'], w['n'])
                        for kk, im in zip(idx, fr):
                            yield int(kk), im
                src = _gen()
            H = None
            lens_a = None
            predict = None
            done = [0]

            def counted(it):
                nonlocal H, lens_a, predict
                for kk, im in it:
                    if H is None:
                        H = im.shape[0]
                        la = tel.lens.scaled(im.shape[1] / float(tel.width))
                        lens_a = Lens(la.model, la.fx, la.fy, la.cx, la.cy, la.k, im.shape[1], H)
                        predict = _gyro_predictor(tel, lens_a, records)
                    done[0] += 1
                    if progress is not None and done[0] % 60 == 0:
                        progress(0.85 * done[0] / len(records), 'timing calibration: tracking')
                    yield kk, im

            t0 = time.perf_counter()
            # the predictor needs the analysis lens before the first LK call: wrap it lazily
            tr = track_frames(counted(src), lambda a, b, q: predict(a, b, q), max_points=p.max_points, cancel=cancel)
            t_dec = time.perf_counter() - t0
        finally:
            if own is not None:
                own.close()
        if H is None:
            notes['status'] = 'no frames decoded'
            return tm_default()
        width = int(lens_a.width)
        obs = dict(tid=tr['tid'], k=tr['k'], xy=tr['xy'],
                   win=np.array([win_of_frame.get(int(kk), 0) for kk in tr['k']], np.int64))
        notes['tracks_per_frame'] = float(len(tr['k']) / max(1, tr['n_frames']))
    else:
        H = int(obs['H'])
        width = int(obs.get('W', p.width))
        notes['windows'] = windows
    if debug is not None:
        debug.update(obs=obs, windows=windows, H=H, W=width)
    res = fit_timing(tel, obs, H, width, windows, p, notes)
    dec = _decide(res, tel, p)
    notes['fit'] = res
    notes['decision'] = dec
    notes['status'] = res.get('status', 'failed')
    applied = dict(offset_ms=0.0, readout_s=None, focal_scale=1.0, exposure_slope=0.0,
                   exposure_ref_s=float(res.get('exposure_ref_ms', 0.0)) * 1e-3, exposure_scale=1.0)
    if res.get('status') == 'ok' and dec.get('source'):
        e = res['estimate'] if dec['source'] == 'conditional' else res['nuisance_only']['estimate']
        if dec['offset']:
            applied['offset_ms'] = e['offset_ms']
        if dec['readout']:
            applied['readout_s'] = float(tel.readout_s) * (1 + e['readout_pct'] * 1e-2)
        if dec['focal']:
            applied['focal_scale'] = 1 + e['focal_pct'] * 1e-2
        if dec['exposure_slope']:
            applied['exposure_slope'] = e['exposure_slope']
        if dec['box']:
            applied['exposure_scale'] = max(0.0, 1 + e['box_pct'] * 1e-2) * float(p.box_scale)
    notes['applied'] = applied
    notes['runtime_s'] = dict(total=time.perf_counter() - t_all, decode_track=t_dec, fit=res.get('fit_s'))
    return TimeModel(offset_s=applied['offset_ms'] * 1e-3, readout_s=applied['readout_s'],
                     focal_scale=float(applied['focal_scale']), exposure_scale=float(applied['exposure_scale']),
                     notes={'timecal': notes})
