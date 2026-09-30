"""Orientation + virtual path -> render Plan (ENGINE_SPEC.md §1-§3).  Owner: WP-B.

Row-matrix convention (what the Metal kernel and render_ref consume):
    For output frame k and row sample j (source row y_j = j*(src_h-1)/(n_rows-1)):
        t_kj   = frame_t[src_k] + readout*((y_j+0.5)/src_h - 0.5)          (video timeline, s)
        M[k,j] = R(q_cam(t_kj))^T @ R(virt_q[k])
    so that an output ray r_v = ((px-out_cx)/out_fx, (py-out_cy)/out_fx, 1) maps to the source-camera ray
    r_c = M(y) @ r_v, with M(y) linearly interpolated between row samples at the (fixed-point) source row y.

Time model (TimeModel): offset/skew are applied inside the orientation function (the IMU is sampled at
t*(1+skew)+offset for a video-timeline time t), the readout comes from tm.readout_s (or the telemetry), the
extrinsic is right-multiplied (camera-frame misalignment), and focal_scale multiplies the lens fx/fy.
Note: applying (1+skew) to the full row time (including the readout term) differs from the TimeModel docstring
formula by readout*skew (~1 ns for |skew| < 1e-4) — negligible, and it keeps q_cam a pure function of t.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .geom import Lens, qexp, qmul, qnormalize, quat_to_mat
from .types import Plan, Telemetry, TimeModel

__all__ = ['camera_orientation_fn', 'build_plan', 'effective_lens', 'effective_readout', 'row_samples_y',
           'exposure_averaged_q', 'exposure_avg_window']


def effective_readout(tel: Telemetry, tm: Optional[TimeModel]) -> float:
    if tm is not None and tm.readout_s is not None:
        return float(tm.readout_s)
    return float(tel.readout_s)


def effective_lens(tel: Telemetry, tm: Optional[TimeModel]) -> Lens:
    """Source lens with TimeModel.focal_scale applied to fx/fy (principal point and k unchanged)."""
    L = tel.lens
    s = 1.0 if tm is None else float(tm.focal_scale)
    return Lens(L.model, float(L.fx) * s, float(L.fy) * s, float(L.cx), float(L.cy),
                np.array(L.k, dtype=np.float64), int(L.width or tel.width), int(L.height or tel.height))


def row_samples_y(src_h: int, n_rows: int) -> np.ndarray:
    """Source rows of the plan's row samples: y_j = j*(src_h-1)/(n_rows-1)."""
    return np.arange(n_rows, dtype=np.float64) * (src_h - 1) / (n_rows - 1)


def camera_orientation_fn(tel: Telemetry, tm: Optional[TimeModel] = None,
                          correction: Optional[Callable[[np.ndarray], np.ndarray]] = None
                          ) -> Callable[[np.ndarray], np.ndarray]:
    """t (video timeline, any shape) -> q_cam(t) (…,4), camera->world.

    q_cam(t) = imu_orientation(t*(1+skew) + offset) * qexp(extrinsic_rotvec) [* correction(t)]
    `correction(t)` (e.g. from closedloop.fold_residuals) is evaluated at the VIDEO-timeline time t and
    right-multiplied (a camera-frame correction)."""
    tm = tm if tm is not None else TimeModel()
    skew, off = float(tm.skew), float(tm.offset_s)
    ext_q = qexp(np.asarray(tm.extrinsic_rotvec, dtype=np.float64).reshape(3))
    has_ext = bool(np.any(np.asarray(tm.extrinsic_rotvec) != 0))

    def q_cam(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        q = tel.orientation_at(t * (1.0 + skew) + off)
        if has_ext:
            q = qmul(q, ext_q)
        if correction is not None:
            q = qmul(q, np.asarray(correction(t), dtype=np.float64))
        return qnormalize(q)

    return q_cam


# Exposure averaging is ramped in with the exposure: window = e * clip((e - AVG_E0) / (AVG_E1 - AVG_E0), 0, 1).
# A/B on DJI_20260927091931_0012 with the same virtual path (eval quality estimator, 1080p-eq px, HF / 8-30 Hz / jello):
#   100-110 s (e ~4.3 ms): instantaneous 0.284 / 0.055 / 0.71  ->  full averaging 0.107 / 0.023 / 0.43
#   176-188 s (e ~1.7 ms): instantaneous 0.98 / 0.083 / 1.97   ->  full averaging 1.20 / 0.113 / 2.05  (worse)
#   300-312 s (e ~1.3 ms): 0.747 / 0.101 / 1.78 -> 0.811 / 0.104 / 1.87;  146-158 s (e ~1.9 ms): mixed.
# i.e. a large win at long exposures and a small loss below ~2 ms (the 1 kHz fused attitude probably under-reports
# the 200-300 Hz prop vibration a little, which the full box filter then over-attenuates). Ramp: none <= 2 ms, full
# >= 3.5 ms, continuous in between (auto exposure moves smoothly, so no timing steps).
AVG_E0_S = 0.0020
AVG_E1_S = 0.0035


def exposure_avg_window(exposure: np.ndarray) -> np.ndarray:
    """Averaging window length (s) per frame for a given exposure (s)."""
    e = np.clip(np.nan_to_num(np.asarray(exposure, np.float64), nan=0.0), 0.0, 0.05)
    return e * np.clip((e - AVG_E0_S) / (AVG_E1_S - AVG_E0_S), 0.0, 1.0)


def exposure_averaged_q(q_cam_fn: Callable[[np.ndarray], np.ndarray], t_rows: np.ndarray, exposure: np.ndarray,
                        taps: int = 9, chunk: int = 2048, sample_rate: float = 0.0) -> np.ndarray:
    """Mean camera orientation over each row's exposure window [t - e/2, t + e/2] (t = row mid-exposure time).

    A row records the time-average of the image motion during its exposure; at short shutters and 200-300 Hz prop
    vibration the high-rate gyro swings through a sizeable part of a vibration cycle inside one exposure, and
    correcting with the instantaneous orientation re-injects vibration that the picture only shows as blur
    (DJI_20260927091931_0012: gyro-vs-image row misfit -10..35 % at 3.5-4.5 ms exposures; gyro HF gain fits 1.0
    only with this). t_rows (F,R), exposure (F,) s -> (F,R,4). Box filter with `taps` mid-point samples,
    sign-aligned to the centre sample, normalized (chordal L2 mean: exact to O(angle^3) for these tiny spreads).
    sample_rate > 0 (the gyro rate): frames whose window spans more gyro samples than `taps` get more taps (at least
    one per gyro sample, in steps of 2x) so long exposures (1/61 s = 16 samples at 1 kHz) are not aliased; windows up
    to (taps - 1) samples keep exactly `taps` taps."""
    t_rows = np.asarray(t_rows, np.float64)
    e = np.nan_to_num(np.asarray(exposure, np.float64).reshape(-1), nan=0.0)
    e = np.clip(e, 0.0, 0.1)                                                  # (timecal may widen the box)
    need = np.full(len(e), int(taps), np.int64)
    if sample_rate and sample_rate > 0:
        n_s = np.ceil(e * float(sample_rate)).astype(np.int64) + 1
        lvl = np.maximum(0, np.ceil(np.log2(np.maximum(n_s, 1) / float(taps)))).astype(np.int64)
        need = np.minimum(int(taps) * (2 ** np.minimum(lvl, 4)), 16 * int(taps))
    out = np.empty(t_rows.shape + (4,), np.float64)
    for n_t in np.unique(need):
        u = (np.arange(n_t, dtype=np.float64) + 0.5) / n_t - 0.5            # (n,) in (-0.5, 0.5)
        idx = np.flatnonzero(need == n_t)
        ch = max(16, int(chunk * taps // n_t))
        for a in range(0, len(idx), ch):
            sel = idx[a:a + ch]
            tt = t_rows[sel, :, None] + e[sel, None, None] * u[None, None, :]    # (f,R,n)
            q = np.asarray(q_cam_fn(tt), np.float64)                              # (f,R,n,4)
            qc = np.asarray(q_cam_fn(t_rows[sel]), np.float64)                    # (f,R,4) centre (sign reference)
            sgn = np.sign(np.einsum('frnc,frc->frn', q, qc))
            sgn[sgn == 0] = 1.0
            out[sel] = qnormalize((q * sgn[..., None]).sum(axis=2))
    return out


def build_plan(tel: Telemetry, tm: Optional[TimeModel], q_cam_fn: Callable[[np.ndarray], np.ndarray],
               virt_q: np.ndarray, out_fx: np.ndarray, out_w: int, out_h: int, n_rows: int = 32,
               frames: Optional[np.ndarray] = None, exposure_avg: bool = True, exposure_taps: int = 9) -> Plan:
    """Build the render plan.

    frames: source frame indices of the output frames (default: all frames). virt_q (F,4) and out_fx (F,)
    (or a scalar) are per OUTPUT frame, F = len(frames). Output frame k is rendered from source frame
    frames[k] and carries its container PTS. Vectorized: 23k frames x 32 rows build in ~1-2 s.
    exposure_avg: with a high-rate gyro (>= 500 Hz), each row matrix uses the orientation averaged over that row's
    exposure (exposure_averaged_q; window exposure_avg_window(e): off below 2 ms, full above 3.5 ms) instead of the
    instantaneous mid-exposure orientation.
    """
    if n_rows < 2:
        raise ValueError("n_rows must be >= 2")
    if out_w % 2 or out_h % 2:
        raise ValueError("out_w/out_h must be even (4:2:0 output)")
    frames = np.arange(tel.n_frames) if frames is None else np.asarray(frames, dtype=np.int64).reshape(-1)
    F = len(frames)
    virt_q = qnormalize(np.asarray(virt_q, dtype=np.float64).reshape(-1, 4))
    if len(virt_q) != F:
        raise ValueError(f"virt_q has {len(virt_q)} rows, expected {F}")
    out_fx = np.broadcast_to(np.asarray(out_fx, dtype=np.float64), (F,)).copy()
    readout = effective_readout(tel, tm)
    lens = effective_lens(tel, tm)
    H = int(tel.height)

    ys = row_samples_y(H, n_rows)                                     # (R,)
    t_rows = tel.frame_t[frames][:, None] + readout * ((ys[None, :] + 0.5) / H - 0.5)   # (F,R)
    ex = np.asarray(tel.exposure_s, np.float64)
    ex_scale = float(getattr(tm, 'exposure_scale', 1.0) or 1.0) if tm is not None else 1.0
    use_avg = bool(exposure_avg and tel.has_highrate and ex.ndim == 1 and len(ex) == tel.n_frames
                   and np.max(exposure_avg_window(ex[frames]), initial=0.0) * ex_scale * tel.imu_rate >= 1.0)
    if use_avg:
        q_cam = exposure_averaged_q(q_cam_fn, t_rows, exposure_avg_window(ex[frames]) * ex_scale, taps=exposure_taps,
                                    sample_rate=float(tel.imu_rate))
    else:
        q_cam = q_cam_fn(t_rows)                                      # (F,R,4)
    Rc = quat_to_mat(q_cam)                                           # (F,R,3,3)
    Rv = quat_to_mat(virt_q)                                          # (F,3,3)
    M = np.einsum('fjki,fkl->fjil', Rc, Rv)                           # Rc^T @ Rv

    tmd = {}
    if tm is not None:
        tmd = {'offset_s': float(tm.offset_s), 'skew': float(tm.skew), 'readout_s': readout,
               'focal_scale': float(tm.focal_scale), 'exposure_scale': ex_scale,
               'extrinsic_rotvec': [float(v) for v in np.asarray(tm.extrinsic_rotvec).reshape(3)]}
    return Plan(src_w=int(tel.width), src_h=H, out_w=int(out_w), out_h=int(out_h), lens=lens,
                frame_pts=np.asarray(tel.frame_pts, dtype=np.float64)[frames].copy(), out_fx=out_fx,
                row_mats=M, virt_q=virt_q.copy(),
                meta={'readout_s': readout, 'frames': frames.copy(), 'time_model': tmd, 'n_rows': n_rows,
                      'exposure_avg': use_avg})
