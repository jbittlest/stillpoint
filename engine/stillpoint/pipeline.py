"""Stillpoint pipeline (WP-F / ENGINE v3): analyze -> plan -> closed loop -> quality -> render -> eval.
ENGINE_SPEC.md §2, §4, §5.

analyze(video, out_dir, params, progress=None, cancel=None):
    load_telemetry -> [self_calibrate] -> q_cam_fn -> output focal (crop: whole-clip exact source footprint >=
    Gyroflow's, or a Gyroflow-like default) -> optimize_path -> build_plan
    -> CLOSED LOOP: render 960-wide rectilinear previews of the current plan -> measure_residuals against the
       intended virtual relative rotations -> closedloop.fold_residuals (robust: outlier rejection, anchored
       weighted integration, smooth support, jump guard) -> q_cam * correction -> re-optimize -> rebuild plan.
       Every increment is verified by the next measurement and kept only in the 1-s windows where it lowered
       the measured HF residual and introduced no new jump (local acceptance, 0.25 s crossfades). Passes after
       the first only re-measure windows whose plan changed; the 2nd+ fold is limited to the windows with the
       largest increments (refine_max_frac).
    -> final composite jump guard -> INDEPENDENT quality report (eval's KLT estimator on the rectified original
       vs the final plan, sampled windows; report.json['quality']) -> plan.spplan + report.json (+ analysis.npz)
render(video, plan_path, out_path, ...): app/renderer/.build/sprender

Resources (ENGINE v3): no whole-clip frame cache any more. Every pass streams exactly the frames it needs through
a bounded FrameStream (parallel ffmpeg VideoToolbox decoders, <= lanes*block+64 frames buffered: ~270-360 MB for
any clip length, no disk); previews are rendered on the GPU with the shared Metal kernel; residuals are measured
in a process pool whose workers exit when this process dies. Temporary files (only the quality pass's bounded
window files) live in a job dir under $STILLPOINT_WORK_DIR (default ~/Library/Application Support/Stillpoint/work),
removed on exit / error / cancel / SIGTERM (stale ones from hard kills are swept by the next job).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .closedloop import (R_ROLL_1920, ResidualCorrection, accept_windows, compose_corrections, fold_residuals,
                         jump_guard, pair_weights, step_series, window_hf, window_mask)
from .geom import qconj, qlog, qmul
from .plan_build import build_plan, camera_orientation_fn, effective_lens
from .plan_io import write_plan
from .types import Plan, Telemetry, TimeModel

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SPRENDER = os.path.join(ROOT, 'app', 'renderer', '.build', 'sprender')

__all__ = ['AnalyzeParams', 'AnalysisCancelled', 'analyze', 'render', 'residual_stats', 'measure_plan',
           'measure_crop_area', 'first_frame_at', 'gyroflow_crop_area', 'plan_footprint', 'gyroflow_footprint',
           'LumaCache', 'window_hf']      # LumaCache: kept for scripts only (analyze() streams frames)

ProgressFn = Callable[[str, float, str], None]
CancelFn = Callable[[], bool]


class AnalysisCancelled(Exception):
    """Raised by analyze() when its cancel() callback returns True."""


# ============================================================================================ parameters


@dataclass
class AnalyzeParams:
    smoothness: float = 1.0
    out_w: int = 0                       # 0 -> source width
    out_h: int = 0                       # 0 -> source height
    out_fx: float = 0.0                  # widest output focal (output px); 0 -> crop search (below)
    crop_area: float = 0.0               # >0: analytic crop area (M1) when no fov_match / target is given
    fov_match: Optional[str] = None      # a Gyroflow render: pick out_fx so Stillpoint's whole-clip mean SOURCE
                                         # FOOTPRINT >= Gyroflow's (eval's cached whole-clip measurement when present)
    crop_mode: str = 'footprint'         # 'footprint' (exact source area, whole clip) | 'eval' (M1: eval's
                                         # homography area over a window)
    footprint_margin: float = 0.003      # aim this much above the target footprint
    target_footprint: float = 0.0        # >0: use this whole-clip mean footprint target instead of measuring
    auto_gf_footprint: bool = True       # no fov_match: use eval's cached whole-clip Gyroflow footprint of this clip
                                         # (work/baseline/<clip>_wholeclip_gf_footprint.json, length-checked) if any
    default_footprint: float = 0.60      # no out_fx / crop_area / fov_match / target: Gyroflow-like footprint
                                         # (Gyroflow's whole-clip footprint on the 5 O3 clips: 0.59-0.63)
    match_start: float = 0.0             # crop_mode 'eval': window (s) over which crop areas are compared
    match_dur: float = 0.0               # 0 -> whole clip
    match_step: int = 6                  # frames between crop measurements (eval --ref-step)
    match_margin: float = 0.012          # crop_mode 'eval': aim this much above the Gyroflow area
    target_area: float = 0.0             # crop_mode 'eval': >0 use this eval-style area instead of measuring
    allow_zoom: bool = True
    max_zoom: float = 1.5                # max_out_fx / min_out_fx
    calibrate: bool = False              # M1 A/B on DJI_0025: WP-E's +11.8% readout raised eval jello 0.555 -> 0.958
    calib_keep: tuple = ('offset_s', 'readout_s', 'focal_scale', 'extrinsic_rotvec', 'skew')
    closed_loop_iters: int = 2           # folds; the loop measures closed_loop_iters + 1 plans at most
    loop_iters_oa4: int = 0              # Osmo Action 4 clips with the in-camera 1 kHz attitude: at most this many
                                         # folds (-1: closed_loop_iters). With the corrected OA4 timing the vision
                                         # fold only chased near-ground parallax on DJI_20260927091931_0012 (3 windows,
                                         # loop off vs on: HF 0.25/0.52/1.20 vs 0.28/0.65/1.19 px, 8-30 Hz 0.049/0.101/
                                         # 0.081 vs 0.048/0.101/0.106, calm 0.109 vs 0.165, jello 0.83 vs 1.25 @146 s)
    measure_open_loop_oa4: bool = False  # ... and skip the (report-only) open-loop measurement pass there (~2x faster)
    min_improve: float = 0.03            # stop when the composite HF residual improves less than this fraction
    refine_max_frac: float = 0.15        # 2nd+ folds re-measure at most this fraction of the 1-s windows (largest
                                         # increments first); M2: a 2nd fold gained <= 1-5 % for up to 35 % of a pass
    preview_width: int = 960
    hp_hz: float = 1.0                   # fold high-pass
    hp_order: int = 4                    # fold high-pass order (the integrated error has degrees of LF drift)
    clamp_deg: float = 0.5
    conf_min: float = 0.6                # pair weight ramps from 0 at conf_lo to 1 at conf_min (squared)
    conf_lo: float = 0.3
    wi_lambda: float = 0.01              # C1 prior of the weighted integration
    anchor_s: float = 0.3                # re-anchoring time of the LF/HF split across untrusted gaps
    support_min: float = 0.25            # mean pair weight over 0.25 s at which the correction is half on
    outlier_k: float = 5.0               # Hampel threshold (x local robust scale); 0 disables
    outlier_floor_px: float = 0.15       # ... never below this (1080p-eq px)
    pred_gain: float = 0.35              # a vision spike is kept if the gyro shows a one-frame event >= 1/0.35 its size
                                         # (and rejected if > 0.5x a gyro jolt: see closedloop.reject_outliers)
    jump_px: float = 0.5                 # jump guard / acceptance veto threshold (1080p-eq px); 0 disables
    jump_support: float = 0.25           # the jump guard reverts steps only where the 5-pair mean weight is below
                                         # this (0.5 notched well-measured legit corrections on DJI_0034)
    window_s: float = 1.0                # local acceptance windows
    accept_tol: float = 0.05             # accept an increment in a window if HF <= best*(1+tol) + abs
    accept_abs_px: float = 0.005
    xfade_s: float = 0.25
    floor_px: float = 0.07               # windows whose measured HF is at/below this are converged: no more folds
    refine_floor_px: float = 0.12        # ... the same for the 2nd+ fold (M2 runs on 0025/0032: a 2nd fold below
                                         # this was accepted in only 30-45 % of windows and moved HF by < 5 %)
    inc_min_px: float = 0.02             # windows whose increment is below this (RMS, 1080p px) ...
    inc_rel_min: float = 0.5             # ... or below this fraction of the window's measured HF are not re-measured
                                         # (the increment is dropped there instead of being kept unverified)
    parallax_gate: float = 0.0           # >0: drop pairs whose parallax_px exceeds this x the clip median
    agree_gate_px: float = 0.0           # >0: drop pairs whose flow-vs-direct disagreement exceeds this (preview px)
    n_rows: int = 32
    workers: Optional[int] = None        # thread workers (only when processes == 0)
    processes: Optional[int] = None      # measurement process pool; None -> residual.default_processes(); 0 -> threads
    residual_overrides: dict = field(default_factory=dict)   # extra ResidualParams fields
    # ---- resources (ENGINE v3): frames are streamed with a bounded buffer; nothing big is written to disk
    decode_lanes: int = 3                # parallel ffmpeg decoders (one VT H.264 4K60 stream: ~55-70 fps)
    decode_block: int = 150              # frames per decode block (buffer = lanes*block+64 frames, ~270-360 MB)
    work_dir: Optional[str] = None       # overrides $STILLPOINT_WORK_DIR for the job's temporary files
    max_temp_bytes: int = 2 * 1024 ** 3  # pre-flight: free space needed on the work volume (+2 GB reserve)
    telemetry_cache: Optional[str] = None   # telemetry cache dir (default <work root>/cache)
    quality: bool = True                 # independent quality report (eval estimator) in report.json['quality']
    quality_frac: float = 0.2            # ... fraction of the clip it samples (<= ~3000 frames; >= 600)
    quality_window_s: float = 3.0
    # ---- deprecated (M2 whole-clip luma cache; ignored: it grew with the clip and froze the app)
    luma_cache: bool = False
    cache_dir: Optional[str] = None
    keep_cache: bool = False
    decode_segments: int = 4
    smooth_overrides: dict = field(default_factory=dict)   # extra SmoothParams fields
    readout_scale: float = 1.0           # multiply the metadata readout (diagnostics / A-B tests)
    measure_open_loop: bool = True       # with closed_loop_iters == 0: still measure the open-loop residual
    save_iter_plans: bool = True         # also write plan_iter<i>.spplan for every measured loop iteration
    max_frames: int = 0                  # >0: analyse only max_frames frames (diagnostics / resource tests)
    start_frame: int = 0                 # >0: the analysed window starts at this source frame (with max_frames: a
                                         # window of the clip; plan records carry the source PTS, so `render` with
                                         # --start-frame renders it)
    exposure_avg: bool = True            # row orientations averaged over each row's exposure (plan_build)
    verbose: bool = True


def _log(prm, *a):
    if prm.verbose:
        print('[stillpoint]', *a, flush=True)


class _Progress:
    """Maps (stage, local fraction) to a monotonic overall fraction and calls the user's progress callback
    (at most every 0.2 s per stage unless the fraction hits 1); checks cancel() on every call.
    Stage weights ~ measured share of the run time (M2/v3 runs): the measurement passes dominate."""
    WEIGHTS = (('telemetry', 0.02), ('calibration', 0.005), ('decode', 0.0), ('crop', 0.05), ('path', 0.015),
               ('measure0', 0.40), ('fold1', 0.03), ('measure1', 0.33), ('fold2', 0.02), ('measure2', 0.05),
               ('final', 0.03), ('quality', 0.08), ('write', 0.01))

    def __init__(self, fn: Optional[ProgressFn], cancel: Optional[CancelFn]):
        self.fn, self.cancel = fn, cancel
        self.base = {}
        acc = 0.0
        for k, w in self.WEIGHTS:
            self.base[k] = (acc, w)
            acc += w
        self.total = acc
        self.last = 0.0
        self.t_last = 0.0
        self.last_stage, self.last_msg = 'telemetry', ''
        self.lock = threading.RLock()
        self.cb_cancelled = False          # the callback raised a *Cancel* exception on the heartbeat thread
        self.finished = False

    def cancel_fn(self) -> bool:
        return bool(self.cb_cancelled or (self.cancel is not None and self.cancel()))

    def __call__(self, stage: str, frac: float = 0.0, msg: str = ''):
        self.check()
        if self.fn is None:
            return
        with self.lock:
            b, w = self.base.get(stage, (self.last * self.total, 0.0))
            f = min(1.0, max(self.last, (b + w * min(max(frac, 0.0), 1.0)) / self.total))
            now = time.perf_counter()
            if not (f >= 1.0 or now - self.t_last > 0.2 or frac <= 0.0):
                return
            self.t_last = now
            self.last = f
            self.last_stage, self.last_msg = stage, msg
            try:
                self.fn(stage, f, msg)
            except Exception as e:  # a broken UI callback must not kill the analysis ...
                if 'cancel' in type(e).__name__.lower():      # ... but a callback may cancel by raising
                    raise

    def heartbeat(self, stop: threading.Event, every: float = 1.0):
        """Background thread: re-send the last progress when nothing was reported for `every` s (stages with
        long silent steps: decoder start-up, pool warm-up, QP solves, writing) -> a callback at least every
        ~1 s. A cancelling exception raised by the callback here is turned into cancel_fn() == True."""
        while not stop.wait(0.25):
            if self.fn is None:
                return
            with self.lock:
                if self.finished or time.perf_counter() - self.t_last < every:
                    continue
                self.t_last = time.perf_counter()
                try:
                    self.fn(self.last_stage, self.last, self.last_msg)
                except Exception as e:  # noqa: BLE001
                    if 'cancel' in type(e).__name__.lower():
                        self.cb_cancelled = True

    def done(self, msg: str = 'done'):
        if self.fn is not None:
            with self.lock:
                self.finished = True
                self.last = 1.0
                try:
                    self.fn('done', 1.0, msg)
                except Exception:
                    pass

    def check(self):
        if self.cancel_fn():
            raise AnalysisCancelled('analysis cancelled')


# ============================================================================================ helpers


def first_frame_at(frame_pts: np.ndarray, t: float) -> int:
    """Index of the first frame with PTS >= t (what `ffmpeg -ss t -i` starts at)."""
    return int(np.searchsorted(np.asarray(frame_pts), t - 1e-6, side='left'))


def _hp(x: np.ndarray, fs: float, lo: float, hi: Optional[float] = None, order: int = 4) -> np.ndarray:
    if hi is None or hi >= 0.499 * fs:
        sos = butter(order, lo, btype='highpass', fs=fs, output='sos')
    else:
        sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x, axis=0)


def _lp(x: np.ndarray, fs: float, hz: float, order: int = 4) -> np.ndarray:
    return sosfiltfilt(butter(order, hz, btype='lowpass', fs=fs, output='sos'), x, axis=0)


def _px(theta: np.ndarray, f1920: float, aspect: float = 9 / 16) -> np.ndarray:
    """Per-sample combined displacement^2 of small rotations (…,3) (camera frame, rad) in 1920-eq px,
    matching eval.jitter_metrics: pitch(x)/yaw(y) -> f*theta translation, roll(z) -> theta*sqrt((W²+H²)/12)."""
    r = math.sqrt((1920.0 ** 2 + (1920.0 * aspect) ** 2) / 12.0)
    return (f1920 * theta[..., 0]) ** 2 + (f1920 * theta[..., 1]) ** 2 + (r * theta[..., 2]) ** 2


def camera_lf_speed(tel: Telemetry, frames: np.ndarray, q_cam_fn, fs: float) -> np.ndarray:
    """Low-passed (<1 Hz) camera angular speed per frame in ORIGINAL-image 1920-eq px/s (the eval gate)."""
    q = q_cam_fn(tel.frame_t[frames])
    w = qlog(qmul(qconj(q[:-1]), q[1:])) * fs                       # rad/s, camera frame
    w = np.vstack([w[:1], w])
    f = float(tel.lens.fx) * 1920.0 / tel.width
    wl = _lp(w, fs, 1.0) if len(w) > 30 else w
    return np.sqrt(_px(wl, f, tel.height / tel.width))


def gyro_event_px(q_cam_fn, ft: np.ndarray, axis_px: np.ndarray) -> np.ndarray:
    """Per pair: the gyro's own one-frame event size (velocity minus its 7-tap running median, px) — the
    'gyro-predicted residual magnitude' for the outlier test (a vision spike where the gyro shows a jolt of
    comparable size is plausible, one on a quiet gyro is not)."""
    from scipy.ndimage import median_filter
    q = q_cam_fn(ft)
    w = qlog(qmul(qconj(q[:-1]), q[1:])) * np.asarray(axis_px)[None, :]
    if len(w) < 8:
        return np.zeros(len(w))
    return np.linalg.norm(w - median_filter(w, size=(7, 1), mode='nearest'), axis=1)


def residual_stats(res: dict, n_frames: int, fs: float, f1920: float, lf_speed: Optional[np.ndarray] = None,
                   conf_min: float = 0.3, trim_s: float = 0.5, gate_px_s: float = 150.0, aspect: float = 9 / 16,
                   window_s: float = 1.0) -> dict:
    """High-frequency residual of a measured plan (vision): integrate the per-pair error rotations into a path
    (untrusted pairs contribute 0), high-pass at 2 Hz like the eval metric, report RMS in 1080p-eq px overall,
    in calm frames (camera LF speed < gate), per band and per axis, plus 1-s window stats."""
    P = n_frames - 1
    rel = np.zeros((P, 3))
    cf = np.zeros(P)
    k0, k1 = np.asarray(res['k0']), np.asarray(res['k1'])
    ok = (k1 == k0 + 1) & (k0 >= 0) & (k0 < P)
    rel[k0[ok]] = np.asarray(res['err_rotvec'])[ok]
    cf[k0[ok]] = np.asarray(res['conf'])[ok]
    rel[~np.isfinite(rel).all(1)] = 0.0
    good = cf >= conf_min
    rel[~good] = 0.0
    eps = np.vstack([np.zeros((1, 3)), np.cumsum(rel, axis=0)])
    tr = int(round(trim_s * fs))
    sl = slice(tr, n_frames - tr) if n_frames > 2 * tr + 30 else slice(0, n_frames)
    out = dict(n_pairs=int(P), trusted_frac=float(good.mean()) if P else 0.0,
               conf_median=float(np.median(cf[ok])) if ok.any() else 0.0)
    if n_frames < 60:
        return out
    hp = _hp(eps, fs, 2.0)
    e2 = _px(hp, f1920, aspect)[sl]
    out['hf_px'] = float(np.sqrt(e2.mean()))
    for nm, lo, hi in (('b2_8', 2.0, 8.0), ('b8_30', 8.0, None)):
        b = _hp(eps, fs, lo, hi)[sl]
        out[nm + '_px'] = float(np.sqrt(_px(b, f1920, aspect).mean()))
    r = math.sqrt((1920.0 ** 2 + (1920.0 * aspect) ** 2) / 12.0)
    h = hp[sl]
    out['hf_axis_px'] = dict(pitch=float(f1920 * np.sqrt(np.mean(h[:, 0] ** 2))),
                             yaw=float(f1920 * np.sqrt(np.mean(h[:, 1] ** 2))),
                             roll=float(r * np.sqrt(np.mean(h[:, 2] ** 2))))
    if lf_speed is not None:
        calm = (np.asarray(lf_speed)[:n_frames] < gate_px_s)[sl]
        out['calm_frac'] = float(calm.mean())
        if calm.sum() >= 60:
            out['calm_hf_px'] = float(np.sqrt(e2[calm].mean()))
            b = _hp(eps, fs, 8.0)[sl][calm]
            out['calm_b8_30_px'] = float(np.sqrt(_px(b, f1920, aspect).mean()))
    wl = max(4, int(round(window_s * fs)))
    nw = len(e2) // wl
    if nw:
        win = np.sqrt(e2[:nw * wl].reshape(nw, wl).mean(1))
        out['win_median_px'] = float(np.median(win))
        out['win_p90_px'] = float(np.percentile(win, 90))
    return out


_window_mask = window_mask          # M1 name (tests / scripts)


def _score(st: dict) -> float:
    """Loop objective: calm-cruise HF when there is enough calm flight, else the 1-s window median, blended
    with the overall HF (so a calm win cannot hide a regression elsewhere)."""
    a = st.get('calm_hf_px', st.get('win_median_px', st.get('hf_px', np.inf)))
    return float(0.7 * a + 0.3 * st.get('hf_px', a))


def _dense(res: dict, n_frames: int) -> dict:
    """Measurement dict (pairs in any order / subset) -> dense per-pair arrays over the clip's F-1 pairs
    (err NaN and conf 0 where not measured)."""
    P = n_frames - 1
    k0, k1 = np.asarray(res['k0']), np.asarray(res['k1'])
    ok = (k1 == k0 + 1) & (k0 >= 0) & (k0 < P)
    d = dict(err=np.full((P, 3), np.nan), conf=np.zeros(P), measured=np.zeros(P, bool))
    d['err'][k0[ok]] = np.asarray(res['err_rotvec'])[ok]
    d['conf'][k0[ok]] = np.asarray(res['conf'])[ok]
    d['measured'][k0[ok]] = True
    for key in ('inlier_frac', 'parallax_px', 'agree_px', 'sigma_px', 'support'):
        if key in res:
            a = np.full(P, np.nan)
            a[k0[ok]] = np.asarray(res[key])[ok]
            d[key] = a
    for key in ('rotvec', 'expected_rotvec'):
        if key in res:
            a = np.full((P, 3), np.nan)
            a[k0[ok]] = np.asarray(res[key])[ok]
            d[key] = a
    d['px_per_rad'] = res.get('px_per_rad')
    return d


def _merge(old: dict, new: dict) -> dict:
    """Dense measurement: new values where the new pass measured, else the old ones."""
    m = new['measured']
    out = {}
    for k, v in old.items():
        if isinstance(v, np.ndarray) and k in new:
            nv = new[k]
            out[k] = np.where(m[:, None] if v.ndim == 2 else m, nv, v)
        else:
            out[k] = v
    out['measured'] = old['measured'] | m
    return out


def _as_res(d: dict) -> dict:
    """Dense -> measure_residuals-like dict over all pairs (unmeasured: err 0, conf 0)."""
    P = len(d['conf'])
    err = np.where(np.isfinite(d['err']), d['err'], 0.0)
    return dict(k0=np.arange(P), k1=np.arange(1, P + 1), err_rotvec=err, conf=np.where(d['measured'], d['conf'], 0.0),
                px_per_rad=d.get('px_per_rad'))


# ============================================================================================ luma cache


class LumaCache:
    """Exact 4x4-box (ffmpeg `scale=...:flags=area`, bit-identical to the preview renderer's GPU box filter
    before 8-bit rounding) 8-bit luma of EVERY frame at preview size, decoded once by parallel ffmpeg segments
    (VideoToolbox decode, per-frame PTS verified against probe()), stored in a memmap file.
    get(k) blocks until frame k is decoded."""

    def __init__(self, video: str, width: int, path: str, segments: int = 4):
        from .video import gray_size, probe
        self.video = video
        self.info = probe(video)
        self.n = int(self.info['n_frames'])
        self.w, self.h = gray_size(self.info['width'], self.info['height'], width)
        self.path = path
        self.mm = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=(self.n, self.h, self.w))
        self.done = np.zeros(self.n, bool)
        self.n_done = 0
        self.cond = threading.Condition()
        self.err: Optional[BaseException] = None
        self.stop = threading.Event()
        self.segments = max(1, int(segments))
        self.threads: list = []
        self.t0 = None
        self.t_done = None

    def start(self):
        self.t0 = time.perf_counter()
        b = np.linspace(0, self.n, self.segments + 1).astype(int)
        for a, e in zip(b[:-1], b[1:]):
            if e > a:
                th = threading.Thread(target=self._run, args=(int(a), int(e)), daemon=True)
                th.start()
                self.threads.append(th)
        return self

    def _run(self, a: int, e: int):
        from .video import DecodeError, _run_decode
        k = a
        try:
            for mode in ('vt_cpu_scale', 'sw'):
                try:
                    for kk, _, img in _run_decode(self.video, self.info, self.w, self.h, mode, k, e - k, True):
                        if self.stop.is_set():
                            return
                        self.mm[kk] = img
                        with self.cond:
                            self.done[kk] = True
                            self.n_done += 1
                            if self.n_done == self.n:
                                self.t_done = time.perf_counter()
                            self.cond.notify_all()
                        k = kk + 1
                    break
                except DecodeError:
                    if mode == 'sw':
                        raise
                    continue                 # resume from the next undecoded frame in software
        except BaseException as ex:  # noqa: BLE001
            with self.cond:
                self.err = ex
                self.cond.notify_all()

    def get(self, k: int) -> np.ndarray:
        if not self.done[k]:
            with self.cond:
                while not self.done[k]:
                    if self.err is not None:
                        raise RuntimeError(f'luma cache decode failed: {self.err!r}') from self.err
                    self.cond.wait(1.0)
        return self.mm[k]

    def fraction(self) -> float:
        return self.n_done / max(self.n, 1)

    def wait_all(self, poll: Optional[Callable[[float], None]] = None):
        while True:
            with self.cond:
                if self.n_done >= self.n:
                    return
                if self.err is not None:
                    raise RuntimeError(f'luma cache decode failed: {self.err!r}') from self.err
                self.cond.wait(0.5)
            if poll is not None:
                poll(self.fraction())

    def close(self, delete: bool = True):
        self.stop.set()
        for th in self.threads:
            th.join(timeout=10)
        try:
            del self.mm
        except AttributeError:
            pass
        if delete:
            try:
                os.remove(self.path)
            except OSError:
                pass


class _GpuPreview:
    """Preview renderer on the GPU (the shared Metal kernel sp_preview_gray via torch MPS) for source luma
    buffers of size (bw, bh) = the analysis size. Output: uint8 (Ho,Wo) (rounded like the old path) + valid."""

    def __init__(self, plan: Plan, bw: int, bh: int, kernel: str = 'catmullrom', iters: int = 3):
        import torch
        from .render_ref import _metal_lib, output_grid
        self.torch = torch
        self.lib = _metal_lib()
        self.plan = plan
        self.bw, self.bh = bw, bh
        self.out_scale = bw / plan.out_w
        self.Wo, self.Ho, self.osx, self.osy = output_grid(plan.out_w, plan.out_h, self.out_scale)
        self.ssx, self.ssy = bw / plan.src_w, bh / plan.src_h
        self.kernel, self.iters = kernel, iters
        self.dst = torch.empty((self.Ho, self.Wo), dtype=torch.float32, device='mps')
        self.val = torch.empty((self.Ho, self.Wo), dtype=torch.float32, device='mps')

    def upload(self, buf: np.ndarray):
        return self.torch.from_numpy(np.ascontiguousarray(buf)).to('mps').to(self.torch.float32)

    def render(self, src, k: int, plan: Optional[Plan] = None):
        from .render_ref import kernel_params
        torch = self.torch
        pl = plan if plan is not None else self.plan
        P = kernel_params(pl, k, dst_w=self.Wo, dst_h=self.Ho, out_sx=self.osx, out_sy=self.osy, src_sx=self.ssx,
                          src_sy=self.ssy, buf_w=self.bw, buf_h=self.bh, kernel=self.kernel, iters=self.iters)
        Pt = torch.from_numpy(P).to('mps')
        Mt = torch.from_numpy(np.ascontiguousarray(pl.row_mats[k], dtype=np.float32).reshape(-1)).to('mps')
        self.lib.sp_preview_gray(src, self.dst, self.val, Pt, Mt, threads=(self.Wo, self.Ho))
        img = torch.clamp(torch.floor(self.dst + 0.5), 0, 255).to(torch.uint8).cpu().numpy()
        return img, (self.val > 0.5).cpu().numpy()


def _render_stream(plan: Plan, stream, records, cancel: Optional[CancelFn] = None, extra_plans: tuple = (),
                   kernel: str = 'catmullrom', iters: int = 3):
    """Preview frames of `plan` (output width = the stream's analysis width) decoded on demand by a bounded
    FrameStream: yields (k, uint8 (Ho,Wo), valid) — or, with extra_plans, (k, [(u8, valid) for plan and each
    extra plan]) rendered from the same decoded frame. Same pixels as _render_cached()."""
    recs = np.asarray(records, dtype=np.int64).reshape(-1)
    src_idx = np.asarray(plan.meta.get('frames', np.arange(plan.n_frames)))
    src = src_idx[recs]
    if len(src) > 1 and np.any(np.diff(src) <= 0):
        raise ValueError('_render_stream: records must map to increasing source frames')
    rk = dict(zip(src.tolist(), recs.tolist()))
    gp = _GpuPreview(plan, stream.w, stream.h, kernel, iters)
    for sk, buf in stream.frames(src, cancel=cancel):
        k = rk[int(sk)]
        s = gp.upload(buf)
        if extra_plans:
            yield k, [gp.render(s, k, pl) for pl in (plan,) + tuple(extra_plans)]
        else:
            img, val = gp.render(s, k)
            yield k, img, val


def _render_cached(plan: Plan, cache: LumaCache, records, kernel: str = 'catmullrom', iters: int = 3):
    """Preview frames of `plan` (output size = cache size) from the luma cache with the shared Metal kernel:
    yields (k, uint8 (Ho,Wo), valid bool). Same math as render_ref.render_frames(out_scale = cache.w/out_w)."""
    import torch
    from .render_ref import _metal_lib, kernel_params, output_grid
    lib = _metal_lib()
    out_scale = cache.w / plan.out_w
    Wo, Ho, osx, osy = output_grid(plan.out_w, plan.out_h, out_scale)
    bw, bh = cache.w, cache.h
    ssx, ssy = bw / plan.src_w, bh / plan.src_h
    src_idx = np.asarray(plan.meta.get('frames', np.arange(plan.n_frames)))
    dst = torch.empty((Ho, Wo), dtype=torch.float32, device='mps')
    val = torch.empty((Ho, Wo), dtype=torch.float32, device='mps')
    for k in np.asarray(records, dtype=np.int64):
        k = int(k)
        buf = cache.get(int(src_idx[k]))
        src = torch.from_numpy(np.ascontiguousarray(buf)).to('mps').to(torch.float32)
        P = kernel_params(plan, k, dst_w=Wo, dst_h=Ho, out_sx=osx, out_sy=osy, src_sx=ssx, src_sy=ssy,
                          buf_w=bw, buf_h=bh, kernel=kernel, iters=iters)
        Pt = torch.from_numpy(P).to('mps')
        Mt = torch.from_numpy(np.ascontiguousarray(plan.row_mats[k], dtype=np.float32).reshape(-1)).to('mps')
        lib.sp_preview_gray(src, dst, val, Pt, Mt, threads=(Wo, Ho))
        img = dst.cpu().numpy()
        yield k, np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8), val.cpu().numpy() > 0.5


# ============================================================================================ measuring


def _preview_plan(plan: Plan) -> Plan:
    """Same plan with a constant (widest) focal, so every preview shares one K (zoom only crops)."""
    return replace(plan, out_fx=np.full(plan.n_frames, float(np.min(plan.out_fx))))


def measure_plan(plan: Plan, video: str, preview_width: int = 960, workers: Optional[int] = None,
                 records: Optional[np.ndarray] = None, progress: Optional[Callable] = None, *,
                 cache: Optional[LumaCache] = None, executor=None,
                 cancel: Optional[CancelFn] = None, stream=None, params=None) -> tuple[dict, np.ndarray]:
    """Render 960-wide rectilinear previews of `plan` and measure the per-pair residual rotation vs the plan's
    intended virtual relative rotation. Returns (measure_residuals dict, preview K). records may contain
    several runs of consecutive frames (pairs across runs are skipped). Frames come from `stream` (a bounded
    FrameStream, the analyze() path), a random-access `cache` (.get(k), .w, .h; scripts) or, with neither, a
    full-resolution PyAV decode (render_ref.render_frames)."""
    from .render_ref import preview_K, render_frames
    from .residual import measure_residuals
    pplan = _preview_plan(plan)
    bw = stream.w if stream is not None else (cache.w if cache is not None else preview_width)
    scale = bw / plan.out_w
    K = preview_K(pplan, 0, scale)
    V = plan.virt_q
    recs = np.arange(plan.n_frames) if records is None else np.asarray(records)

    def expected(k0, k1):
        return qmul(qconj(V[k0]), V[k1])

    if stream is not None:
        gen = _render_stream(pplan, stream, recs, cancel=cancel)
    elif cache is not None:
        gen = _render_cached(pplan, cache, recs)
    else:
        gen = render_frames(pplan, video, recs, out_scale=scale, return_valid=True)
    it = _prefetch(gen, 48)
    try:
        res = measure_residuals(it, K, expected, params=params, workers=workers, progress=progress,
                                executor=executor, consecutive_only=True, cancel=cancel)
    except InterruptedError as e:
        raise AnalysisCancelled(str(e)) from e
    return res, K


def _prefetch(iterable, depth: int = 128):
    """Run a generator in a background thread (preview rendering overlaps the residual measurement)."""
    import queue
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    END = object()

    def worker():
        try:
            for item in iterable:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    break
        except BaseException as e:  # propagate
            q.put(e)
        finally:
            # stop the producer now (a FrameStream generator stops its decoders + ffmpeg on close)
            close = getattr(iterable, 'close', None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
            q.put(END)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is END:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while th.is_alive():
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass


def _fold(tel, plan, dense: dict, q_cam_fn, frames, fs, prm, f1920: float, weights: np.ndarray,
          pred_px: Optional[np.ndarray] = None):
    """Robust fold of a dense measurement (closedloop.fold_residuals) with explicit pair weights."""
    F = len(frames)
    rel = np.where(np.isfinite(dense['err']), dense['err'], 0.0)
    w = np.asarray(weights, dtype=np.float64).copy()
    # parallax-dominated pairs bias a rotation-only fit (the fold would inject the bias as real jitter)
    if prm.parallax_gate > 0 and 'parallax_px' in dense:
        par = dense['parallax_px']
        med = np.nanmedian(par)
        w[np.isfinite(par) & (par > prm.parallax_gate * med)] = 0.0
    if prm.agree_gate_px > 0 and 'agree_px' in dense:
        ag = np.nan_to_num(dense['agree_px'])
        w[ag > prm.agree_gate_px] = 0.0
    # never integrate across a shot boundary
    for a, b in (tel.segments or []):
        if 0 < a < F:
            w[a - 1] = 0.0
    ft = tel.frame_t[frames]
    v2c = qmul(qconj(q_cam_fn(ft)), plan.virt_q)
    return fold_residuals(ft, rel, dense['conf'], fs=fs, hp_hz=prm.hp_hz, clamp_deg=prm.clamp_deg,
                          virt_to_cam_q=v2c, conf_min=prm.conf_min, conf_lo=prm.conf_lo, hp_order=prm.hp_order,
                          lam=prm.wi_lambda, weights=w, support_lo=max(0.0, prm.support_min - 0.1),
                          support_hi=prm.support_min + 0.1, px_per_rad=f1920,
                          roll_px_per_rad=R_ROLL_1920, outlier_k=prm.outlier_k or None,
                          outlier_floor_px=prm.outlier_floor_px, pred_px=pred_px, pred_gain=prm.pred_gain,
                          jump_px=prm.jump_px or None, jump_support=prm.jump_support, anchor_s=prm.anchor_s)


def _weighted_integrate(rel: np.ndarray, w: np.ndarray, lam: float = 0.01) -> np.ndarray:
    """M1 name, kept for scripts: plain weighted integration (closedloop.weighted_integrate, no anchoring)."""
    from .closedloop import weighted_integrate
    return weighted_integrate(rel, w, lam, anchor_s=None)


def _runs(mask: np.ndarray):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# ============================================================================================ crop matching


def _orig_gray(video: str, frames: np.ndarray, width: int = 960) -> np.ndarray:
    from .video import read_gray_frames
    return read_gray_frames(video, np.asarray(frames), width=width)


def measure_crop_area(plan: Plan, video: str, records: np.ndarray, orig_frames: np.ndarray,
                      width: int = 960, cache: Optional[LumaCache] = None) -> dict:
    """eval.jitter_metrics-style visible-area fraction (SIFT homography original->stabilised, output rectangle
    back-projected and clipped to the original) of `plan` at the given plan records, from 960 previews."""
    sys.path.insert(0, ROOT) if ROOT not in sys.path else None
    from eval.jitter_metrics import crop_distortion_from_H, match_homography
    from .render_ref import render_frames
    if cache is not None:
        prev = {k: img for k, img, _ in _render_cached(plan, cache, records)}
    else:
        prev = {k: img for k, img in render_frames(plan, video, records, out_scale=width / plan.out_w)}
    areas = []
    for i, k in enumerate(np.asarray(records)):
        o = orig_frames[i]
        s = prev.get(int(k))
        if s is None:
            continue
        Hm, n, err = match_homography(o, s)
        if Hm is None:
            continue
        _, a, _ = crop_distortion_from_H(Hm, o.shape[0] / o.shape[1], s.shape[0] / s.shape[1])
        areas.append(a)
    areas = np.asarray(areas)
    return dict(area_mean=float(areas.mean()) if len(areas) else float('nan'),
                area_min=float(areas.min()) if len(areas) else float('nan'), n=int(len(areas)))


def gyroflow_crop_area(gf_render: str, orig: str, start: float, dur: Optional[float], step: int = 6) -> float:
    """Visible-area fraction of a Gyroflow render measured exactly like eval (--ref original)."""
    sys.path.insert(0, ROOT) if ROOT not in sys.path else None
    from eval import jitter_metrics as jm
    r = jm.reference_metrics(gf_render, orig, start, dur, start, 960, step)
    return float(r['visible_area_frac_mean'])


def _border_points(ow: int, oh: int, n: int = 41) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return np.concatenate([np.c_[t * (ow - 1), 0 * t], np.c_[(ow - 1) + 0 * t, t * (oh - 1)],
                           np.c_[(1 - t) * (ow - 1), (oh - 1) + 0 * t], np.c_[0 * t, (1 - t) * (oh - 1)]])


def _footprint_from_rays(plan: Plan, k: int, rv: np.ndarray) -> float:
    """Fraction of the source image covered by the polygon of VIRTUAL rays rv (N,3) of plan frame k (row-timed
    mapping, fixed-point source row, 3 iterations; rasterised at 1/4 res and clipped to the source) — the
    verifier's exact method (work/verify/metric/fov2.py)."""
    import cv2
    mats = plan.row_mats[k].astype(np.float64)
    nr = mats.shape[0]
    y = np.full(len(rv), (plan.src_h - 1) / 2)
    uv = None
    for _ in range(3):
        g = np.clip(y * (nr - 1) / (plan.src_h - 1), 0, nr - 1)
        j0 = np.minimum(np.floor(g).astype(int), nr - 2)
        a = (g - j0)[:, None, None]
        Mi = mats[j0] * (1 - a) + mats[j0 + 1] * a
        rc = np.einsum('nij,nj->ni', Mi, rv)
        uv = plan.lens.project(rc)
        y = uv[:, 1]
    m = np.zeros((plan.src_h // 4, plan.src_w // 4), np.uint8)
    cv2.fillPoly(m, [np.round(uv.astype(np.float32) / 4).astype(np.int32)], 1)
    return float(m.mean())


def plan_footprint(plan: Plan, records: np.ndarray, n: int = 41) -> np.ndarray:
    """Exact source footprint (fraction of the source area the output uses) of plan records."""
    ow, oh = plan.out_w, plan.out_h
    e = _border_points(ow, oh, n)
    cx, cy = (ow - 1) / 2, (oh - 1) / 2
    out = []
    for k in np.asarray(records, dtype=np.int64):
        fx = float(plan.out_fx[k])
        rv = np.c_[(e[:, 0] - cx) / fx, (e[:, 1] - cy) / fx, np.ones(len(e))]
        out.append(_footprint_from_rays(plan, int(k), rv))
    return np.asarray(out)


def _decompose_gf(Hm: np.ndarray, fs: float, asp: float):
    """H(GF -> SP) in centred width-normalised coords = Ks R Kg^-1: solve the GF focal fg (width-normalised)
    and R given the SP focal fs (fov2.decompose)."""
    from scipy.optimize import least_squares
    T = np.array([[1, 0, -0.5], [0, 1, -asp / 2], [0, 0, 1.0]])
    Hc = T @ Hm @ np.linalg.inv(T)

    def mk(p):
        fg = math.exp(p[0])
        M = np.diag([1 / fs, 1 / fs, 1]) @ Hc @ np.diag([fg, fg, 1])
        return M / np.cbrt(np.linalg.det(M))

    def res(p):
        M = mk(p)
        return (M.T @ M - np.eye(3)).ravel()[[0, 1, 2, 4, 5, 8]]
    best = None
    for g in (0.7, 0.85, 1.0, 1.15, 1.3):
        r = least_squares(res, [math.log(fs * g)])
        if best is None or r.cost < best.cost:
            best = r
    M = mk(best.x)
    U, _, Vt = np.linalg.svd(M)
    return math.exp(best.x[0]), U @ Vt, float(np.sqrt(best.cost * 2 / 6))


def _gf_keyframes(gf_render: str, w: int, h: int):
    """(pts (N,), gray u8 (N,h,w)) of the key frames of a render (fast: -skip_frame nokey)."""
    import re
    cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'info', '-skip_frame', 'nokey', '-hwaccel',
           'videotoolbox', '-i', gf_render, '-an', '-vf', f'extractplanes=y,scale={w}:{h}:flags=area,showinfo',
           '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f'ffmpeg keyframe decode failed: {p.stderr[-800:]!r}')
    fr = np.frombuffer(p.stdout, np.uint8)
    n = len(fr) // (w * h)
    pts = [float(m.group(1).decode()) for m in re.finditer(rb'pts_time:\s*(\S+)', p.stderr)]
    pts = np.asarray(pts[:n], dtype=np.float64)
    return pts, fr[:n * w * h].reshape(n, h, w)


def gyroflow_footprint(gf_render: str, plan: Plan, cache, frame_pts: np.ndarray,
                       cache_json: Optional[str] = None, cancel: Optional[CancelFn] = None) -> dict:
    """Whole-clip mean source footprint of a Gyroflow render: at every key frame of the render, SIFT-match it
    to a Stillpoint preview of `plan` (same source frame), decompose H = Ks R Kg^-1 (Gyroflow's focal and
    rotation relative to the preview), and map Gyroflow's output border through the plan's row-timed mapping
    to the source (verifier's fov2.py method, whole clip instead of a window). Cached in cache_json."""
    import cv2
    key = dict(path=os.path.abspath(gf_render), size=os.path.getsize(gf_render),
               mtime=os.path.getmtime(gf_render))
    if cache_json and os.path.exists(cache_json):
        try:
            d = json.load(open(cache_json))
            if d.get('key') == key:
                return d
        except Exception:
            pass
    W, H = cache.w, cache.h
    asp = H / W
    pts, gfr = _gf_keyframes(gf_render, W, H)
    ks = np.array([int(np.argmin(np.abs(frame_pts - t))) for t in pts])
    okk = np.abs(frame_pts[ks] - pts) < 0.25 / 59.94 * 2
    sift = cv2.SIFT_create()
    if hasattr(cache, 'frames'):          # a bounded FrameStream (analyze) ...
        prev = {k: img for k, img, _ in _render_stream(plan, cache, np.unique(ks[okk]), cancel=cancel)}
    else:                                 # ... or a random-access cache (.get)
        prev = {k: img for k, img, _ in _render_cached(plan, cache, np.unique(ks[okk]))}
    rows = []
    ow, oh = plan.out_w, plan.out_h
    e = _border_points(ow, oh)
    cx, cy = (ow - 1) / 2, (oh - 1) / 2
    for i in np.flatnonzero(okk):
        k = int(ks[i])
        a, b = prev[k], gfr[i]
        k1, d1 = sift.detectAndCompute(b, None)
        k2, d2 = sift.detectAndCompute(a, None)
        if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
            continue
        good = [m for m, nn in (x for x in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(x) == 2)
                if m.distance < 0.75 * nn.distance]
        if len(good) < 30:
            continue
        p1 = np.float32([k1[m.queryIdx].pt for m in good]) / W
        p2 = np.float32([k2[m.trainIdx].pt for m in good]) / W
        Hm, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0 / 960, maxIters=5000)
        if Hm is None or inl.sum() < 30:
            continue
        inl = inl.ravel().astype(bool)
        pr = cv2.perspectiveTransform(p1[inl].reshape(-1, 1, 2), Hm).reshape(-1, 2)
        herr = float(np.median(np.linalg.norm(pr - p2[inl], axis=1)) * 960)
        fs_ = float(plan.out_fx[k]) / ow
        fg, R, orth = _decompose_gf(Hm, fs_, asp)
        if herr >= 0.6 or orth >= 0.02:
            continue
        f = fg * ow
        rg = np.c_[(e[:, 0] - cx) / f, (e[:, 1] - cy) / f, np.ones(len(e))]
        rv = rg @ R.T
        rv = rv / rv[:, 2:3]
        rows.append(dict(k=k, t=float(pts[i]), fg=fg, herr=herr, orth=orth,
                         gf=_footprint_from_rays(plan, k, rv)))
    if not rows:
        raise RuntimeError('Gyroflow footprint: no usable key frame matches')
    gf = np.array([r['gf'] for r in rows])
    d = dict(key=key, n=len(rows), n_keyframes=int(len(pts)), mean=float(gf.mean()), median=float(np.median(gf)),
             min=float(gf.min()), max=float(gf.max()),
             fg_1920_median=float(np.median([r['fg'] for r in rows]) * 1920), rows=rows)
    if cache_json:
        os.makedirs(os.path.dirname(cache_json), exist_ok=True)
        with open(cache_json, 'w') as fh:
            json.dump(d, fh, indent=1, default=float)
    return d


# ============================================================================================ analyze


def _smooth_params(prm: AnalyzeParams, fx0: float, tm: TimeModel, tel: Telemetry, pg=None, stage: str = 'path',
                   lo: float = 0.0, hi: float = 1.0):
    from .smooth import SmoothParams
    sp = SmoothParams(smoothness=prm.smoothness, min_out_fx=fx0, max_out_fx=fx0 * prm.max_zoom,
                      allow_zoom=prm.allow_zoom)
    for k, v in (prm.smooth_overrides or {}).items():
        setattr(sp, k, v)
    if pg is not None:
        sp.cancel = pg.cancel_fn
        sp.tick = lambda f, _s=stage, _lo=lo, _hi=hi: pg(_s, _lo + (_hi - _lo) * f, 'solving the camera path')
    return sp


def _gf_footprint_cached(fov_match: str, n_frames: Optional[int] = None) -> Optional[tuple]:
    """Whole-clip Gyroflow footprint measured by eval (work/baseline/<clip>_wholeclip_gf_footprint.json) for a
    Gyroflow render named <clip>_stabilized.* (or the original <clip>.MP4) -> (mean, path) or None.
    With n_frames (auto lookup by the ORIGINAL's name: DJI numbers repeat across cards) the measurement must
    have sampled exactly this clip length (n == ceil(n_frames / step_frames)), else it is ignored."""
    from .workspace import work_root
    b = os.path.basename(fov_match)
    clip = b.split('_stabilized')[0] if '_stabilized' in b else os.path.splitext(b)[0]
    for d in (os.path.join(ROOT, 'work', 'baseline'), os.path.join(work_root(create=False), 'baseline')):
        p = os.path.join(d, f'{clip}_wholeclip_gf_footprint.json')
        if os.path.exists(p):
            try:
                js = json.load(open(p))
                if n_frames is not None:
                    step = int(js.get('step_frames') or 0)
                    if step <= 0 or int(js.get('n_sampled', js.get('n', -1))) != -(-int(n_frames) // step):
                        continue
                return float(js['mean']), p
            except Exception:
                continue
    return None


def analyze(video: str, out_dir: str, params: Optional[AnalyzeParams] = None,
            progress: Optional[ProgressFn] = None, cancel: Optional[CancelFn] = None) -> dict:
    """Full analysis of one clip. Writes <out_dir>/plan.spplan, report.json, analysis.npz. Returns the report.

    progress(stage, fraction, message): called at least every ~2 s; fraction is the OVERALL analysis progress
        0..1 (monotonic, roughly proportional to time), stage one of telemetry, calibration, crop, path,
        measure0, fold1, measure1, fold2, measure2, final, quality, write, done.
    cancel(): polled at least every ~0.25 s in the long stages; when it returns True, AnalysisCancelled is raised
        (worker processes and decoders are stopped, temporary files removed, nothing is written).

    Resources (ENGINE v3): frames are streamed (no whole-clip cache: temporary disk <= ~1.3 GB during the quality
    pass, ~0 otherwise; memory bounded independent of clip length); temporary files live in a private job dir
    under $STILLPOINT_WORK_DIR (default ~/Library/Application Support/Stillpoint/work), removed on exit, error,
    cancel and SIGTERM; measurement workers die with this process."""
    from .workspace import JobDir, combine_cancel, preflight, sigterm_cancels, work_root
    prm = params or AnalyzeParams()
    if prm.work_dir:
        os.environ['STILLPOINT_WORK_DIR'] = os.path.abspath(os.path.expanduser(prm.work_dir))
    term = threading.Event()
    pg = _Progress(progress, combine_cancel(cancel, term.is_set))
    root = work_root()
    preflight(root, prm.max_temp_bytes, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    hb_stop = threading.Event()
    hb = threading.Thread(target=pg.heartbeat, args=(hb_stop,), daemon=True, name='stillpoint-progress')
    hb.start()
    try:
        with sigterm_cancels(term), JobDir('analyze', root=root, video=video) as job:
            try:
                return _analyze(video, out_dir, prm, pg, job)
            except InterruptedError as e:
                raise AnalysisCancelled(str(e)) from e
    finally:
        hb_stop.set()
        hb.join(2.0)


class _OffsetStream:
    """A FrameStream seen through a window: local frame k is source frame k + offset (AnalyzeParams.start_frame)."""

    def __init__(self, stream, offset: int):
        self._s, self._off = stream, int(offset)

    def frames(self, records, cancel=None):
        for sk, buf in self._s.frames(np.asarray(records, dtype=np.int64) + self._off, cancel=cancel):
            yield int(sk) - self._off, buf

    def __getattr__(self, name):
        return getattr(self._s, name)


def _analyze(video, out_dir, prm: AnalyzeParams, pg: _Progress, job) -> dict:
    from .calib import self_calibrate
    from .framestream import FrameStream
    from .residual import ResidualParams, make_pool, shutdown_pool
    from .smooth import fx_for_crop_area, optimize_path
    from .telemetry import load_telemetry
    from .workspace import cache_dir

    T = {}
    t_all = time.perf_counter()
    pg('telemetry', 0.0, 'reading telemetry')
    t0 = time.perf_counter()
    tel = load_telemetry(video, cache_dir=prm.telemetry_cache or cache_dir())
    T['telemetry_s'] = time.perf_counter() - t0
    if tel.eis_baked:
        raise NotImplementedError(f'{video}: in-camera EIS is baked into the picture; the gyro no longer '
                                  'describes the pixels. Stillpoint needs EIS-off footage.')
    s_ = max(0, int(prm.start_frame or 0))
    if s_ >= tel.n_frames:
        raise ValueError(f'start_frame {s_} >= {tel.n_frames} frames')
    n_ = tel.n_frames - s_
    if prm.max_frames and int(prm.max_frames) > 0:
        n_ = min(n_, int(prm.max_frames))
    if s_ > 0 or n_ < tel.n_frames:                     # diagnostics / windows: frames [s_, s_ + n_)
        e_ = s_ + n_
        ex_ = np.asarray(tel.exposure_s)
        tel = replace(tel, frame_pts=np.asarray(tel.frame_pts)[s_:e_], frame_t=np.asarray(tel.frame_t)[s_:e_],
                      exposure_s=ex_[s_:e_] if ex_.ndim else ex_,
                      segments=[(max(int(a), s_) - s_, min(int(b), e_ - 1) - s_) for a, b in (tel.segments or [])
                                if int(b) >= s_ and int(a) < e_])
    loop_note = None
    if (tel.has_highrate and 'Osmo Action 4' in str(tel.camera) and prm.loop_iters_oa4 >= 0
            and prm.closed_loop_iters > prm.loop_iters_oa4):
        loop_note = (f'OA4 1 kHz attitude: closed loop {prm.closed_loop_iters} -> {prm.loop_iters_oa4} folds '
                     '(AnalyzeParams.loop_iters_oa4)')
        prm = replace(prm, closed_loop_iters=int(prm.loop_iters_oa4),
                      measure_open_loop=bool(prm.measure_open_loop and prm.measure_open_loop_oa4))
        _log(prm, loop_note)
    out_w = int(prm.out_w or tel.width)
    out_h = int(prm.out_h or tel.height)
    fs = float(tel.fps)
    frames = np.arange(tel.n_frames)
    F = tel.n_frames
    _log(prm, f'{os.path.basename(video)}: {tel.camera}, {F} frames, imu {tel.imu_rate:.0f} Hz, '
              f'segments {tel.segments}')
    pg('telemetry', 1.0, f'{F} frames')

    stream = None
    pool = None
    res_prm = ResidualParams(**(prm.residual_overrides or {}))
    try:
        stream = FrameStream(video, prm.preview_width, lanes=prm.decode_lanes, block=prm.decode_block)
        if s_ > 0:
            stream = _OffsetStream(stream, s_)
        T['frame_buffer_mb'] = stream.max_buffer_bytes() / 1e6
        if prm.processes is None or prm.processes > 0:
            try:
                pool = make_pool(prm.processes)
            except Exception as e:  # pragma: no cover - fall back to threads
                _log(prm, f'process pool unavailable ({e!r}); using threads')
                pool = None
        pg.check()

        # ---- calibration
        pg('calibration', 0.0, 'calibration')
        t0 = time.perf_counter()
        if prm.calibrate and tel.has_highrate:
            tm = self_calibrate(tel, video)
            tm_full = tm
            tm = TimeModel(offset_s=tm.offset_s if 'offset_s' in prm.calib_keep else 0.0,
                           skew=tm.skew if 'skew' in prm.calib_keep else 0.0,
                           readout_s=tm.readout_s if 'readout_s' in prm.calib_keep else None,
                           focal_scale=tm.focal_scale if 'focal_scale' in prm.calib_keep else 1.0,
                           extrinsic_rotvec=(np.asarray(tm.extrinsic_rotvec) if 'extrinsic_rotvec' in prm.calib_keep
                                             else np.zeros(3)), notes=tm_full.notes)
        else:
            tm = TimeModel()
        if prm.readout_scale != 1.0:
            tm.readout_s = float(tel.readout_s) * prm.readout_scale
        T['calib_s'] = time.perf_counter() - t0
        calib = dict(offset_ms=tm.offset_s * 1e3, skew_ppm=tm.skew * 1e6,
                     readout_ms=(tm.readout_s if tm.readout_s is not None else tel.readout_s) * 1e3,
                     readout_meta_ms=tel.readout_s * 1e3, focal_scale=tm.focal_scale,
                     extrinsic_deg=np.rad2deg(np.asarray(tm.extrinsic_rotvec)).tolist(),
                     status=tm.notes.get('calib', {}).get('status'),
                     final=tm.notes.get('calib', {}).get('final'))
        _log(prm, f'calib ({T["calib_s"]:.0f}s): offset {calib["offset_ms"]:+.2f} ms, readout {calib["readout_ms"]:.3f} ms '
                  f'(meta {calib["readout_meta_ms"]:.3f}), focal x{tm.focal_scale:.4f}, ext {np.round(calib["extrinsic_deg"], 3)}')
        pg('calibration', 1.0, 'calibration done')

        q_fn = camera_orientation_fn(tel, tm)
        lens = effective_lens(tel, tm)
        lf_speed = camera_lf_speed(tel, frames, q_fn, fs)
        info_holder = {}

        def optimize(qf, fx0, stage='path', lo=0.0, hi=1.0):
            sp = _smooth_params(prm, fx0, tm, tel, pg, stage, lo, hi)
            v, fx, info = optimize_path(tel, qf, frames, out_w, out_h, sp, tm=tm, return_info=True)
            info_holder['info'] = info
            return v, fx, info

        def mkplan(qf, v, fx):
            return build_plan(tel, tm, qf, v, fx, out_w, out_h, n_rows=prm.n_rows, frames=frames,
                              exposure_avg=prm.exposure_avg)

        # ---- output focal / crop
        pg('crop', 0.0, 'choosing the output field of view')
        t0 = time.perf_counter()
        crop = dict(mode=None)
        if prm.out_fx and prm.out_fx > 0:
            fx0 = float(prm.out_fx)
            crop['mode'] = 'explicit'
            v, fx, info = optimize(q_fn, fx0, 'crop', 0.1, 0.9)
        elif prm.crop_mode == 'eval' and (prm.fov_match or prm.target_area > 0):
            fx0, v, fx, info = _crop_eval(prm, tel, lens, out_w, out_h, fs, q_fn, optimize, mkplan, None,
                                          video, crop)
        elif prm.crop_area and prm.crop_area > 0 and not (prm.fov_match or prm.target_footprint > 0):
            fx0 = fx_for_crop_area(lens, tel.width, tel.height, out_w, out_h, prm.crop_area)
            crop['mode'] = 'analytic_area'
            crop['analytic_area'] = prm.crop_area
            v, fx, info = optimize(q_fn, fx0, 'crop', 0.1, 0.9)
        else:
            fx0, v, fx, info = _crop_footprint(prm, tel, lens, out_w, out_h, q_fn, optimize, mkplan, stream,
                                               video, crop, pg)
        crop['min_out_fx'] = fx0
        T['crop_s'] = time.perf_counter() - t0
        pg('crop', 1.0, f'out_fx {fx0:.1f}')
        pg('path', 0.0, 'building the plan')
        plan = mkplan(q_fn, v, fx)
        f1920 = fx0 * 1920.0 / out_w
        ax = np.array([f1920, f1920, R_ROLL_1920])
        ft = tel.frame_t[frames]
        pred = gyro_event_px(q_fn, ft, ax)
        pg('path', 1.0, 'plan ready')

        # ---- closed loop
        t_loop = time.perf_counter()
        wl = int(round(prm.window_s * fs))
        nw = max(1, F // wl)
        widx = np.minimum(np.arange(F) // wl, nw - 1)          # frame -> window
        pidx = widx[:-1]                                        # pair -> window (first frame)
        xf = int(round(prm.xfade_s * fs))
        iters = []
        incs: list[ResidualCorrection] = []
        inc_support: list[np.ndarray] = []
        dense_all = None
        dense0 = None
        h_best = None
        h_open = None
        frozen_w = np.zeros(nw, bool)                          # windows whose last increment was reverted
        records = frames.copy()
        stop_reason = 'iterations'
        n_meas = prm.closed_loop_iters + 1
        if prm.closed_loop_iters == 0 and not prm.measure_open_loop:
            n_meas = 0
            iters.append(dict(iter=0, measured=False))
        measured_w = np.ones(nw, bool)
        veto_pair_mask = np.zeros(max(F - 1, 0), bool)
        for i in range(n_meas):
            ti = time.perf_counter()
            stage = f'measure{min(i, 2)}'
            nrec = len(records)
            pg(stage, 0.0, f'measuring pass {i} ({nrec} frames)')

            def prog(n_done, _s=stage, _n=nrec, _i=i):
                pg(_s, n_done / max(_n - 1, 1), f'measuring pass {_i}: {n_done}/{_n - 1} pairs')
            res, K = measure_plan(plan, video, prm.preview_width, prm.workers, records=records, progress=prog,
                                  stream=stream, executor=pool, cancel=pg.cancel_fn, params=res_prm)
            tm_meas = time.perf_counter() - ti
            dn = _dense(res, F)
            rec = dict(iter=i, measure_s=tm_meas, pairs_per_s=res['timing']['pairs_per_s'],
                       timing={k_: v_ for k_, v_ in res['timing'].items() if np.isscalar(v_)},
                       n_pairs_measured=int(dn['measured'].sum()),
                       smooth=dict(runtime_s=info['runtime_s'], frac_binding=info['frac_binding'],
                                   max_violation_px=info['max_violation_px'], max_zoom=info['max_zoom'],
                                   n_frames_violating=info['n_frames_violating']))
            if prm.save_iter_plans:
                write_plan(os.path.join(out_dir, f'plan_iter{i}.spplan'), plan)
            if i == 0:
                dense_all = dn
                dense0 = dn
                h_best = window_hf(_as_res(dn), F, fs, f1920, wl, conf_min=prm.conf_lo)
                h_open = h_best.copy()
            else:
                merged = _merge(dense_all, dn)
                h_new = window_hf(_as_res(merged), F, fs, f1920, wl, conf_min=prm.conf_lo)
                # the re-measured windows (the merged series supplies HP context at their edges); the increment
                # is removed where it was not re-measured (it was below inc_min there: nothing unverified stays)
                inner = measured_w.copy()
                # new-jump veto: a velocity outlier in the new measurement that the old one did not have, at a
                # pair where this increment changed the correction noticeably
                c_last = incs[-1].diagnostics.get('corr_virt')
                if prm.jump_px and c_last is not None:
                    e_new = np.where(np.isfinite(merged['err']), merged['err'], 0.0)
                    e_old = np.where(np.isfinite(dense_all['err']), dense_all['err'], 0.0)
                    trust = (merged['conf'] >= 0.15) & dn['measured']
                    j_new = _vel_steps(e_new, ax)
                    j_old = _vel_steps(e_old, ax)
                    dc = np.linalg.norm(np.diff(c_last, axis=0) * ax[None, :], axis=1)
                    # (a) a new velocity outlier where the increment changed the plan, or (b) the increment made a
                    # large one-pair change that did NOT remove the measured step it was meant to cancel (then the
                    # old measurement was the outlier and the change is a real jump in the picture)
                    newj = trust & (((j_new > prm.jump_px) & (j_new > j_old + prm.jump_px) & (dc > 0.3 * prm.jump_px))
                                    | ((dc > prm.jump_px) & (j_new > np.maximum(prm.jump_px, 0.8 * j_old))))
                    rec['veto_pairs'] = np.flatnonzero(newj).tolist()
                acc = accept_windows(h_new, h_best, prm.accept_tol, prm.accept_abs_px) & inner
                m = window_mask(acc, F, wl, xf)
                if rec.get('veto_pairs'):          # revert the increment locally (+-0.2 s) around vetoed pairs
                    nt = _notch_frames(F, np.asarray(rec['veto_pairs']) + 0.5, int(round(0.2 * fs)))
                    m = m * nt
                    veto_pair_mask |= (0.5 * (nt[:-1] + nt[1:])) < 0.99      # never folded again
                incs[-1] = incs[-1].scaled(m)
                inc_support[-1] = inc_support[-1] * 0.5 * (m[:-1] + m[1:])
                frozen_w |= ~acc
                h_prev = h_best.copy()
                h_best = np.where(acc & inner, h_new, h_best)
                # the accepted windows' state is now the new measurement; rejected windows keep the old one
                keep_new = (acc & inner)[pidx] & dn['measured']
                dense_all = _merge(dense_all, dict(dn, measured=keep_new))
                rec['accepted_window_frac'] = float(acc[inner].mean()) if inner.any() else 0.0
                rec['measured_window_frac'] = float(inner.mean())
                rec['rejected_windows'] = np.flatnonzero(~acc).tolist()
                rec['composite_hf_px'] = float(np.sqrt(np.mean(h_best ** 2)))
                rec['composite_gain'] = float(1 - np.sqrt(np.mean(h_best ** 2)) / max(np.sqrt(np.mean(h_prev ** 2)), 1e-12))
            st = residual_stats(_as_res(dense_all), F, fs, f1920, lf_speed, conf_min=prm.conf_lo)
            rec['stats'] = st
            _log(prm, f'loop {i}: measured {rec["n_pairs_measured"]} pairs, composite HF {st.get("hf_px", float("nan")):.3f} px, '
                      f'calm {st.get("calm_hf_px", float("nan")):.3f}, 8-30 {st.get("b8_30_px", float("nan")):.3f}, '
                      f'trusted {st["trusted_frac"]:.2f}'
                      + (f'; accepted {rec["accepted_window_frac"]:.0%} of re-measured windows' if i > 0 else '')
                      + f' ({tm_meas:.0f}s, {res["timing"]["pairs_per_s"]:.0f} pairs/s)')
            last = i == n_meas - 1
            if i > 0 and rec['composite_gain'] < prm.min_improve:
                stop_reason = 'converged (composite gain < min_improve)'
                last = True
            if last:
                iters.append(rec)
                break
            # ---- fold the measured residual where it can still help
            pg(f'fold{i + 1}', 0.0, f'fold {i + 1}')
            tf = time.perf_counter()
            conv_w = h_best <= (prm.floor_px if i == 0 else max(prm.floor_px, prm.refine_floor_px))
            wts = pair_weights(dn['conf'], prm.conf_lo, prm.conf_min) * dn['measured']
            wts = wts * ~(frozen_w | conv_w)[pidx] * ~veto_pair_mask
            corr = _fold(tel, plan, dn, q_fn, frames, fs, prm, f1920, wts, pred)
            d = corr.diagnostics
            rec['fold'] = {k: (float(d[k]) if np.isscalar(d.get(k)) else None) for k in
                           ('hf_rms_deg', 'lf_rms_deg', 'corr_rms_deg', 'corr_max_deg', 'clamp_frac', 'bad_frac',
                            'hf_rms_px', 'corr_rms_px', 'n_outliers', 'n_jump_fixed', 'max_step_px',
                            'support_frac') if k in d}
            rec['fold']['converged_windows'] = int(conv_w.sum())
            rec['fold']['frozen_windows'] = int(frozen_w.sum())
            if d.get('identity'):
                iters.append(rec)
                stop_reason = 'nothing to fold'
                break
            cv = d['corr_virt'] * ax[None, :]
            inc_w = np.sqrt(np.bincount(widx, np.sum(cv ** 2, 1), nw) / np.bincount(widx, None, nw))
            need = inc_w > np.maximum(prm.inc_min_px, prm.inc_rel_min * h_best)
            if i >= 1 and prm.refine_max_frac < 1.0:
                # 2nd+ folds: M2 runs accepted them in only 16-38 % of windows for a <= 1-5 % gain, at up to 35 %
                # of a pass -> measure only the windows with the largest increments, within a frame budget
                budget = max(1, int(prm.refine_max_frac * nw))
                if need.sum() > budget:
                    order = np.argsort(-np.where(need, inc_w, -np.inf))
                    keep = np.zeros(nw, bool)
                    keep[order[:budget]] = True
                    rec['refine_budget_dropped'] = int(need.sum() - budget)
                    need &= keep
            if not need.any():
                iters.append(rec)
                stop_reason = 'increment below inc_min_px everywhere'
                break
            measured_w = need
            records = frames[need[widx]]
            rec['next_measure_windows'] = int(need.sum())
            incs.append(corr)
            inc_support.append(np.convolve(d['pair_weight'], np.ones(5) / 5, mode='same'))
            q_fn = camera_orientation_fn(tel, tm, correction=compose_corrections(*incs))
            v, fx, info = optimize(q_fn, fx0, f'fold{i + 1}', 0.1, 0.9)
            plan = mkplan(q_fn, v, fx)
            rec['replan_s'] = time.perf_counter() - tf
            rec['next_measure_frac'] = float(len(records) / F)
            iters.append(rec)
            pg(f'fold{i + 1}', 1.0, f'fold {i + 1} done')

        # ---- final composite: jump guard over the sum of the accepted increments, then the final plan
        pg('final', 0.0, 'final plan')
        tf = time.perf_counter()
        guard = dict(n_fixed=0, pairs=[])
        if incs:
            c_tot = np.zeros((F, 3))
            sup = np.zeros(F - 1)
            v2c_now = qmul(qconj(q_fn(ft)), plan.virt_q)
            for c_, s_ in zip(incs, inc_support):
                c_tot += qmul_rot(qconj(v2c_now), c_.rotvec(ft))
                sup = np.maximum(sup, s_)
            if prm.jump_px:
                _, g = jump_guard(c_tot, sup, ax, fs, jump_px=prm.jump_px, support_min=prm.jump_support)
                guard = dict(n_fixed=g['n_fixed'], pairs=g['pairs'])
                if g['n_fixed']:
                    incs = [c_.scaled(g['factor']) for c_ in incs]
            c_steps = step_series(c_tot, ax)
            guard['max_step_px_before'] = float(c_steps.max()) if len(c_steps) else 0.0
        corr_best = compose_corrections(*incs) if incs else None
        if incs and len(iters) > 0:
            q_fn = camera_orientation_fn(tel, tm, correction=corr_best)
            v, fx, info = optimize(q_fn, fx0, 'final', 0.1, 0.9)
            plan = mkplan(q_fn, v, fx)
        T['closed_loop_s'] = time.perf_counter() - t_loop
        T['final_replan_s'] = time.perf_counter() - tf
        pg('final', 1.0, 'final plan ready')
        plan.meta.update(dict(video=os.path.abspath(video)))

        # ---- independent quality report (eval's estimator on the original vs the final plan)
        quality = None
        if prm.quality and F >= 90:
            pg('quality', 0.0, 'measuring the result independently')
            tq = time.perf_counter()
            try:
                from .quality import quality_report
                quality = quality_report(plan, stream, pool, fs, fx0, job.dir, frac=prm.quality_frac,
                                         window_s=prm.quality_window_s, cancel=pg.cancel_fn,
                                         progress=lambda f, m: pg('quality', f, m), render_fn=_render_stream)
            except (InterruptedError, AnalysisCancelled):
                raise
            except Exception as e:  # never lose an analysis to the report
                quality = dict(error=repr(e))
                _log(prm, f'quality report failed: {e!r}')
            T['quality_s'] = time.perf_counter() - tq
            pg('quality', 1.0, 'quality measured')
        if quality is None:
            quality = dict(error='skipped')
        quality['vision_trusted_frac'] = float(np.mean(dense_all['conf'] >= prm.conf_lo)) if dense_all is not None else None
        try:
            fp = _eval_plan_footprint(plan, np.arange(0, F, max(1, F // 1200)))
            quality['crop_footprint_mean'] = float(fp.mean())
            quality['crop_footprint_min'] = float(fp.min())
        except Exception as e:
            quality['crop_footprint_error'] = repr(e)
        quality['closed_loop_self_estimate_px'] = dict(
            open=float(np.sqrt(np.mean(h_open ** 2))) if h_open is not None else None,
            final=float(np.sqrt(np.mean(h_best ** 2))) if h_best is not None else None,
            note='the closed loop grading itself (optimistic); use stabilized_* for the honest number')

        # ---- outputs
        pg('write', 0.0, 'writing')
        plan_path = os.path.join(out_dir, 'plan.spplan')
        write_plan(plan_path, plan)
        corr_rv = np.zeros((F, 3))
        for c in incs:
            corr_rv += c.rotvec(ft)
        corr_virt = qmul_rot(qconj(qmul(qconj(q_fn(ft)), plan.virt_q)), corr_rv) if incs else corr_rv
        final_steps = step_series(corr_virt, ax) if incs else np.zeros(max(F - 1, 0))
        guard['max_step_px_final'] = float(final_steps.max()) if len(final_steps) else 0.0
        guard['n_steps_over_jump_px'] = int((final_steps > (prm.jump_px or 0.5)).sum())
        extra = {}
        if dense0 is not None:
            for k_ in ('err', 'conf', 'inlier_frac', 'parallax_px', 'agree_px', 'rotvec'):
                if k_ in dense0:
                    extra['res0_' + ('err_rotvec' if k_ == 'err' else k_)] = dense0[k_]
            extra['res0_k0'] = np.arange(F - 1)
        if dense_all is not None:
            extra.update(res_err=dense_all['err'], res_conf=dense_all['conf'], res_k0=np.arange(F - 1),
                         res_measured=dense_all['measured'])
        np.savez_compressed(os.path.join(out_dir, 'analysis.npz'), frame_t=ft, frame_pts=tel.frame_pts,
                            virt_q=plan.virt_q, out_fx=plan.out_fx, corr_rotvec=corr_rv, corr_virt=corr_virt,
                            lf_speed=lf_speed, gyro_event_px=pred,
                            window_hf_open=h_open if h_open is not None else np.zeros(0),
                            window_hf_best=h_best if h_best is not None else np.zeros(0), **extra)
        T['total_s'] = time.perf_counter() - t_all
        T['x_realtime'] = T['total_s'] / max(F / fs, 1e-9)
        T['decode'] = dict(stream.stats) if stream is not None else None
        rep_cl = dict(iterations=iters, n_increments=len(incs), stop_reason=stop_reason, jump_guard=guard,
                      note=loop_note)
        if dense0 is not None:
            rep_cl.update(open_loop_window_hf_px=float(np.sqrt(np.mean(h_open ** 2))),
                          composite_window_hf_px=float(np.sqrt(np.mean(h_best ** 2))),
                          window_median_open_px=float(np.median(h_open)),
                          window_median_composite_px=float(np.median(h_best)))
        else:
            rep_cl.update(open_loop_window_hf_px=None, composite_window_hf_px=None)
        report = dict(
            video=os.path.abspath(video), camera=tel.camera, n_frames=int(F), fps=fs,
            out=dict(w=out_w, h=out_h, min_out_fx=fx0, hfov_deg=float(np.rad2deg(2 * np.arctan(out_w / 2 / fx0)))),
            calibration=calib, crop=crop,
            zoom=dict(max=float(plan.out_fx.max() / fx0), mean=float(plan.out_fx.mean() / fx0),
                      frac_zoomed=float((plan.out_fx > fx0 * 1.001).mean())),
            smooth=dict(runtime_s=info['runtime_s'], frac_binding=info['frac_binding'],
                        max_violation_px=info['max_violation_px'], n_frames_violating=info['n_frames_violating']),
            closed_loop=rep_cl, quality=quality, timings=T, plan=plan_path,
            resources=dict(work_dir=job.dir, frame_buffer_mb=T.get('frame_buffer_mb'),
                           pool_workers=getattr(pool, 'n_workers', 0)),
            params={k: (v_ if not isinstance(v_, tuple) else list(v_)) for k, v_ in asdict(prm).items()})
        with open(os.path.join(out_dir, 'report.json'), 'w') as fh:
            json.dump(report, fh, indent=1, default=_json_default)
        q_ = quality or {}
        _log(prm, f'done in {T["total_s"]:.0f}s ({T["x_realtime"]:.2f}x realtime); quality (eval estimator, sampled): '
                  f'HF {q_.get("original_hf_px")} -> {q_.get("stabilized_hf_px")} px, calm '
                  f'{q_.get("original_calm_hf_px")} -> {q_.get("stabilized_calm_hf_px")}, new jumps >1px '
                  f'{q_.get("new_jumps_gt_1px")}; self-estimate {rep_cl["open_loop_window_hf_px"]} -> '
                  f'{rep_cl["composite_window_hf_px"]}; jump guard {guard}')
        pg('write', 1.0, 'written')
        pg.done(f'analysis done in {T["total_s"]:.0f}s')
        return report
    finally:
        if pool is not None:
            shutdown_pool(pool)
        if stream is not None:
            stream.close()


def _eval_plan_footprint(plan: Plan, records: np.ndarray) -> np.ndarray:
    """eval.footprint.plan_footprint (exact polygon clipping; the gate's crop measure)."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from eval.footprint import plan_footprint as _pf
    return _pf(plan, records)


def qmul_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (F,3) by quaternions q (F,4)."""
    from .geom import qrotate
    return qrotate(q, v)


def _notch_frames(F: int, centres: np.ndarray, half: int) -> np.ndarray:
    from .closedloop import _notch
    return _notch(F, centres, half)


def _vel_steps(err: np.ndarray, ax: np.ndarray) -> np.ndarray:
    """Velocity-outlier size (px) of a per-pair error series (P,3): |e*ax - median7(e*ax)|."""
    from scipy.ndimage import median_filter
    v = err * ax[None, :]
    if len(v) < 8:
        return np.zeros(len(v))
    return np.linalg.norm(v - median_filter(v, size=(7, 1), mode='nearest'), axis=1)


def _crop_footprint(prm, tel, lens, out_w, out_h, q_fn, optimize, mkplan, stream, video, crop, pg):
    """out_fx such that the whole-clip mean EXACT source footprint (eval.footprint.plan_footprint, the gate's
    crop measure) >= the target (+margin). Target: target_footprint if given; else the Gyroflow render's
    whole-clip footprint (eval's cached work/baseline/<clip>_wholeclip_gf_footprint.json, else measured from the
    render's key frames); else default_footprint (Gyroflow-like ~0.60).
    The focal is searched on the SAME virtual path with a scaled focal (the row matrices do not depend on the
    focal: no rebuild), then the path is re-optimised at that focal and checked (zoom where the crop constraint
    binds lowers the footprint)."""
    F = tel.n_frames
    recs = np.arange(0, F, max(1, min(15, F // 400 or 1)))
    guess = lambda a: 1650.0 * math.sqrt(0.58 / a) * (lens.fx / 1405.13) * (out_w / 3840.0)
    target = float(prm.target_footprint) if prm.target_footprint > 0 else None
    if target is None and prm.fov_match:
        c = _gf_footprint_cached(prm.fov_match)
        if c is not None:
            target = c[0]
            crop.update(gyroflow_render=prm.fov_match, gyroflow_footprint_source=c[1])
    if target is None and not prm.fov_match and prm.auto_gf_footprint:
        c = _gf_footprint_cached(video, n_frames=F)       # Gyroflow's whole-clip footprint, measured by eval
        if c is not None:
            target = c[0]
            crop.update(gyroflow_footprint_source=c[1], gyroflow_auto=True)
    source = 'target' if prm.target_footprint > 0 else ('gyroflow_cached' if target is not None else None)
    fx0 = guess(target + prm.footprint_margin) if target else guess(max(prm.default_footprint, 0.3))
    pg('crop', 0.05, f'path at fx {fx0:.1f}')
    v, fx, info = optimize(q_fn, fx0, 'crop', 0.05, 0.3)
    pl = mkplan(q_fn, v, fx)
    if target is None and prm.fov_match:
        pg('crop', 0.3, 'measuring the Gyroflow render footprint')
        from .workspace import cache_dir
        cj = os.path.join(cache_dir(), 'gf_footprint',
                          os.path.basename(prm.fov_match) + f'.{os.path.getsize(prm.fov_match)}.json')
        gfd = gyroflow_footprint(prm.fov_match, pl, stream, tel.frame_pts, cache_json=cj, cancel=pg.cancel_fn)
        target = gfd['mean']
        source = 'gyroflow_measured'
        crop.update(gyroflow_render=prm.fov_match, gyroflow_n=gfd['n'], gyroflow_footprint_min=gfd['min'],
                    gyroflow_fx_1920_median=gfd['fg_1920_median'])
    if target is None:
        target = float(prm.default_footprint)
        source = 'default'
    crop['target_footprint'] = target
    crop['target_source'] = source
    if source.startswith('gyroflow'):
        crop['gyroflow_footprint'] = target
    goal = target + prm.footprint_margin
    trials = []
    s = 1.0
    for it in range(5):                             # cheap: same path, scaled focal (row matrices unchanged)
        pg.check()
        a = float(_eval_plan_footprint(replace(pl, out_fx=fx * s), recs).mean())
        trials.append(dict(kind='scaled', fx0=fx0 * s, footprint=a))
        if abs(a - goal) < 3e-4:
            break
        s *= math.sqrt(a / goal)
    best = None
    fx_try = fx0 * s
    a = float('nan')
    v2 = fx2 = info2 = None
    for it in range(3):                             # re-optimised path at the chosen focal
        pg('crop', 0.4 + 0.18 * it, f'path at fx {fx_try:.1f}')
        v2, fx2, info2 = optimize(q_fn, fx_try, 'crop', 0.4 + 0.18 * it, 0.55 + 0.18 * it)
        a = float(_eval_plan_footprint(mkplan(q_fn, v2, fx2), recs).mean())
        trials.append(dict(kind='optimized', fx0=fx_try, footprint=a, max_zoom=info2['max_zoom'],
                           binding=info2['frac_binding']))
        _log(prm, f'crop: fx0 {fx_try:.1f} -> mean footprint {a:.4f} (target {target:.4f} [{source}], goal {goal:.4f})')
        if a >= target and (best is None or a >= goal - 1e-3 or best[1] < goal - 1e-3):
            best = (fx_try, a, v2, fx2, info2)
        if goal - 1e-3 <= a < goal + 0.004:
            break
        if a >= target:
            fx_try *= math.sqrt(a / goal)
        else:
            fx_try *= math.sqrt(a / goal) * 0.998
    if best is None:
        best = (fx_try, a, v2, fx2, info2)
        crop['warning'] = 'target footprint not reached'
    fx0, a, v, fx, info = best
    crop.update(mode='footprint', trials=trials, footprint_mean=a, footprint_goal=goal,
                footprint_method='eval.footprint.plan_footprint (every %d frames)' % (recs[1] - recs[0] if len(recs) > 1 else 1))
    return fx0, v, fx, info


def _crop_eval(prm, tel, lens, out_w, out_h, fs, q_fn, optimize, mkplan, cache, video, crop):
    """M1 method: eval's homography-based visible area over a window >= Gyroflow's (+margin)."""
    if prm.target_area > 0:
        target = float(prm.target_area)
        crop['gyroflow_area'] = target
    else:
        dur = prm.match_dur if prm.match_dur > 0 else None
        target = gyroflow_crop_area(prm.fov_match, video, prm.match_start, dur, prm.match_step)
        crop['gyroflow_area'] = target
        crop['gyroflow_render'] = prm.fov_match
    crop['mode'] = 'match_eval'
    goal = target + prm.match_margin
    a0 = first_frame_at(tel.frame_pts, prm.match_start)
    a1 = tel.n_frames if prm.match_dur <= 0 else min(tel.n_frames, a0 + int(math.ceil(prm.match_dur * fs)))
    recs = np.arange(a0, a1, prm.match_step)
    if len(recs) > 400:
        recs = recs[np.linspace(0, len(recs) - 1, 400).astype(int)]
    orig = _orig_gray(video, recs)
    fx0 = float(np.interp(goal, [0.511, 0.792, 0.945, 0.996], [2000.0, 1600.0, 1400.0, 1250.0])) \
        * (lens.fx / 1405.13) * (out_w / 3840.0) * 0.965
    trials = []
    best = None
    m = None
    for trial in range(5):
        v, fx, info = optimize(q_fn, fx0)
        pl = mkplan(q_fn, v, fx)
        m = measure_crop_area(pl, video, recs, orig, cache=cache)
        trials.append(dict(fx0=fx0, area=m['area_mean'], area_min=m['area_min'], n=m['n'],
                           max_zoom=info['max_zoom'], binding=info['frac_binding']))
        _log(prm, f'crop trial {trial}: fx0 {fx0:.1f} -> eval-style area {m["area_mean"]:.4f} (goal {goal:.4f})')
        okay = m['area_mean'] >= target + 0.5 * prm.match_margin
        if okay and (best is None or m['area_mean'] < best[1]):
            best = (fx0, m['area_mean'], v, fx, info)
        if okay and m['area_mean'] - goal < 0.006:
            break
        ok_ = [t_ for t_ in trials if np.isfinite(t_['area'])]
        if len(ok_) >= 2 and abs(ok_[-1]['area'] - ok_[-2]['area']) > 1e-4:
            l1, l2 = math.log(ok_[-1]['fx0']), math.log(ok_[-2]['fx0'])
            a1_, a2_ = ok_[-1]['area'], ok_[-2]['area']
            lnew = l1 + (goal - a1_) * (l1 - l2) / (a1_ - a2_)
            fx0 = float(np.clip(math.exp(lnew), fx0 * 0.8, fx0 * 1.25))
        else:
            fx0 = fx0 * (m['area_mean'] / goal) ** 1.5
    if best is None:
        best = (fx0, m['area_mean'], v, fx, info)
        crop['warning'] = 'Gyroflow crop area not reached'
    fx0, _, v, fx, info = best
    crop['trials'] = trials
    return fx0, v, fx, info


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return repr(o)


# ============================================================================================ render


def render(video: str, plan_path: str, out_path: str, start_frame: int = 0, n_frames: Optional[int] = None,
           codec: str = 'hevc10', zero_base: bool = True, kernel: str = 'lanczos3', extra: tuple = (),
           quiet: bool = True) -> dict:
    """Render with the Metal renderer (app/renderer/.build/sprender). zero_base: output starts at t=0
    (deliverables people watch); False keeps the source timestamps (empty edit at the start)."""
    if not os.path.exists(SPRENDER):
        raise FileNotFoundError(f'{SPRENDER} missing: run app/renderer/build.sh')
    cmd = [SPRENDER, video, plan_path, out_path, '--start-frame', str(int(start_frame)), '--codec', codec,
           '--kernel', kernel]
    if n_frames:
        cmd += ['--frames', str(int(n_frames))]
    if zero_base:
        cmd.append('--zero-base')
    cmd += list(extra)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f'sprender failed ({p.returncode}): {p.stderr[-2000:]}\n{p.stdout[-2000:]}')
    n = n_frames or 0
    return dict(cmd=cmd, seconds=dt, fps=(n / dt if n else None), stdout=p.stdout[-3000:])


# ============================================================================================ evaluation summary


def _calm_axes(npz_ref: str, npz: str, thr: float = 150.0, trim: int = 30) -> dict:
    """Calm-cruise (reference LF speed < thr) band x axis RMS (1920-eq px) of an eval signals file."""
    a, b = np.load(npz_ref), np.load(npz)
    fps = float(b['fps'])
    N = min(len(a['path_tx']), len(b['path_tx']))
    sp = a['lp_speed']
    sp = np.concatenate([[sp[0]], sp])[:N]
    base = np.arange(trim, N - trim)
    idx = base[sp[base] < thr]
    out = {}
    if len(idx) < 60:
        return out
    for nm, lo, hi in (('2-8', 2.0, 8.0), ('8-30', 8.0, None), ('hf', 2.0, None)):
        row = {}
        for ax_, key, s in (('x', 'path_tx', 1.0), ('y', 'path_ty', 1.0), ('roll', 'path_rot', R_ROLL_1920)):
            x = _hp(np.asarray(b[key][:N], float), fps, lo, hi) * s
            row[ax_] = float(np.sqrt(np.mean(x[idx] ** 2)))
        out[nm] = row
    return out


def summarize_eval(runs: dict, gate: float = 150.0) -> tuple[str, dict]:
    """runs: name -> base path (eval json/npz without extension); needs 'original'. Returns (markdown, dict)."""
    sys.path.insert(0, ROOT) if ROOT not in sys.path else None
    from eval.compare import load, paired
    from eval.run_baseline import gated_paired
    rows = {}
    for nm, base in runs.items():
        r = json.load(open(base + '.json'))
        m = r['metrics']
        b8 = [v_ for k_, v_ in m['bands'].items() if k_.startswith('8-')][0]
        rows[nm] = dict(hf=m['hf_jitter_px'], hf_dn=m['hf_jitter_px_denoised'], b28=m['bands']['2-8Hz']['combined_px'],
                        b8=b8['combined_px'], b8_rot=b8['rot_deg'], b8_tx=b8['tx_px'], b8_ty=b8['ty_px'],
                        roll=m['hf_rot_deg'], jello=m['jello']['jello_px'],
                        crop=r.get('ref', {}).get('visible_area_frac_mean', float('nan')),
                        lag=r.get('ref_alignment', {}).get('best_lag_frames'),
                        med1s=m.get('window', {}).get('median_px', float('nan')),
                        active=m.get('window', {}).get('median_px_active', float('nan')),
                        noise=m['noise_floor_px'])
    o = runs['original'] + '.npz'
    for nm, base in runs.items():
        g = gated_paired(o, base + '.npz', gate)
        rows[nm]['calm'] = g['b'] if g else float('nan')
        rows[nm]['calm_frac'] = g['frac'] if g else float('nan')
        rows[nm]['calm_axes'] = _calm_axes(o, base + '.npz', gate)
    comp = {}
    if 'stillpoint' in runs:
        sp_ = load(runs['stillpoint'] + '.json')
        for nm in runs:
            if nm != 'stillpoint':
                comp[nm] = paired(load(runs[nm] + '.json'), sp_)
    lines = ['| video | HF px | calm-cruise HF px | 2-8 Hz | 8-30 Hz | roll deg | 8-30 roll deg | jello | 1-s median | '
             'crop area | lag | win-rate (SP better) |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for nm, x in rows.items():
        w = comp.get(nm, {})
        wr = f"{w['b_better_frac'] * 100:.0f}% (p={w['wilcoxon_p']:.1g})" if w else '-'
        lines.append(f"| {nm} | {x['hf']:.3f} | {x['calm']:.3f} | {x['b28']:.3f} | {x['b8']:.3f} | {x['roll']:.4f} | "
                     f"{x['b8_rot']:.4f} | {x['jello']:.3f} | {x['med1s']:.3f} | {x['crop']:.3f} | {x['lag']} | {wr} |")
    return '\n'.join(lines), dict(rows=rows, paired=comp)
