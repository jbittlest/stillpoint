"""Shared data types passed between engine stages. See ENGINE_SPEC.md §2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geom import Lens, slerp_series


@dataclass
class Telemetry:
    """Everything a clip's metadata tells us, normalized to Stillpoint conventions.

    Orientation samples: imu_q[i] maps camera-frame vectors to world at time imu_t[i]
    (seconds on the VIDEO timeline, strictly increasing, uniform-grid timing — NOT the
    per-frame-spread timing telemetry-parser/Gyroflow use).
    Frames: frame_pts[k] is the container PTS of video frame k; frame_t[k] is the time the
    CENTRE row of frame k was captured, mid-exposure, on the same timeline.
    """
    source: str                      # path of the video
    camera: str                      # e.g. 'DJI O3 (FC8383)', 'DJI Osmo Action 4'
    width: int
    height: int
    fps: float                       # container nominal fps
    frame_pts: np.ndarray            # (F,) float64 s
    frame_t: np.ndarray              # (F,) float64 s, centre-row mid-exposure time
    exposure_s: np.ndarray           # (F,) float64 s
    readout_s: float                 # full-frame top->bottom readout time, s
    lens: Lens                       # source intrinsics (pixel-centre convention)
    imu_t: np.ndarray                # (N,) float64 s
    imu_q: np.ndarray                # (N,4) camera->world, sign-continuous
    imu_rate: float                  # Hz (60 when only per-frame orientation exists)
    has_highrate: bool               # True when imu_rate >= 500 Hz
    eis_baked: bool                  # True when in-camera EIS was on (picture already warped)
    gravity_q: Optional[np.ndarray] = None   # (N,4) same as imu_q but world = gravity-aligned (z down), if known
    segments: list = field(default_factory=list)  # [(first_frame, last_frame_inclusive)] continuous shots
    extra: dict = field(default_factory=dict)      # anything else (iso, zoom, raw fields, notes)

    @property
    def n_frames(self) -> int:
        return int(len(self.frame_pts))

    def orientation_at(self, t: np.ndarray) -> np.ndarray:
        """Camera->world quaternions at arbitrary times (…,) -> (…,4)."""
        return slerp_series(self.imu_t, self.imu_q, t)


@dataclass
class TimeModel:
    """Corrections found by self-calibration (defaults = trust the metadata).

    Row time used everywhere:  t(k, y) = frame_t[k]*(1+skew) + offset_s + readout_s*((y+0.5)/H - 0.5)
    Camera orientation used:   q_cam(t) = imu_orientation(t) * qexp(extrinsic_rotvec)   (right-multiply,
                               i.e. a fixed IMU->camera misalignment expressed in the camera frame)
    """
    offset_s: float = 0.0
    skew: float = 0.0
    readout_s: Optional[float] = None       # None -> Telemetry.readout_s
    focal_scale: float = 1.0                # multiplies lens.fx/fy
    extrinsic_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    notes: dict = field(default_factory=dict)


@dataclass
class Plan:
    """What the renderer needs for every output frame (see ENGINE_SPEC.md §3, .spplan format).

    For output frame k and row-sample j, row_mats[k, j] (3x3) maps an OUTPUT virtual-camera ray
    r_v = ((px-out_cx)/out_fx, (py-out_cy)/out_fx, 1) to a SOURCE camera ray r_c = M @ r_v, valid for
    source rows near row_y[j] = j*(src_h-1)/(n_rows-1). i.e. M = R_cam(t(k, row_y[j]))^T @ R_virtual[k].
    """
    src_w: int
    src_h: int
    out_w: int
    out_h: int
    lens: Lens                        # source lens actually used (after focal_scale)
    frame_pts: np.ndarray             # (F,) source PTS each output frame is rendered from
    out_fx: np.ndarray                # (F,) output focal length in output pixels (per-frame zoom)
    row_mats: np.ndarray              # (F, n_rows, 3, 3) float64 in memory, float32 on disk
    virt_q: np.ndarray                # (F,4) virtual camera->world orientation (for diagnostics)
    meta: dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.row_mats.shape[1])

    @property
    def n_frames(self) -> int:
        return int(self.row_mats.shape[0])
