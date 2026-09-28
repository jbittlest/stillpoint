"""Shared geometry: quaternions / SO(3), the KB4 fisheye lens, and row timing.

Conventions (authoritative, see ENGINE_SPEC.md §1):
  * Quaternions are (w, x, y, z), Hamilton product, unit norm, float64 in the engine.
  * An orientation quaternion q maps CAMERA-frame vectors to WORLD: v_world = R(q) @ v_cam.
  * Camera frame: x right, y down, z forward (OpenCV).
  * Pixel coordinates: pixel centres at integers, (0, 0) is the centre of the top-left pixel.
  * Time: seconds on the video timeline (comparable to frame PTS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------- quaternions


def qnormalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b, broadcasting over leading dims."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qconj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qfix_sign(q: np.ndarray) -> np.ndarray:
    """Make a quaternion sequence (N,4) sign-continuous (q and -q are the same rotation)."""
    q = np.array(q, dtype=np.float64, copy=True)
    if len(q) < 2:
        return q
    d = np.einsum('ij,ij->i', q[1:], q[:-1])
    flips = np.cumsum(d < 0) % 2
    q[1:][flips == 1] *= -1.0
    return q


def qexp(v: np.ndarray) -> np.ndarray:
    """Rotation vector (…,3) [rad] -> unit quaternion (…,4)."""
    v = np.asarray(v, dtype=np.float64)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * th
    # sin(th/2)/th with a series near 0
    small = th < 1e-8
    k = np.where(small, 0.5 - th * th / 48.0, np.sin(half) / np.where(small, 1.0, th))
    return np.concatenate([np.cos(half), v * k], axis=-1)


def qlog(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (…,4) -> rotation vector (…,3) [rad], shortest rotation (w >= 0)."""
    q = np.asarray(q, dtype=np.float64)
    q = np.where(q[..., :1] < 0, -q, q)
    w = np.clip(q[..., :1], -1.0, 1.0)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    th = 2.0 * np.arctan2(s, w)
    small = s < 1e-8
    k = np.where(small, 2.0 / np.where(w == 0, 1.0, w), th / np.where(small, 1.0, s))
    return v * k


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(…,4) -> (…,3,3) rotation matrices."""
    q = qnormalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(…,3,3) -> (…,4) with w >= 0 (Shepperd's method, vectorized)."""
    R = np.asarray(R, dtype=np.float64)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    q = np.empty((len(R), 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for i in range(len(R)):
        m = R[i]
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.where(q[:, :1] < 0, -q, q)
    return qnormalize(q).reshape(shp + (4,))


def qrotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (…,3) by q (…,4): R(q) @ v."""
    return np.einsum('...ij,...j->...i', quat_to_mat(q), np.asarray(v, dtype=np.float64))


def slerp_series(t_src: np.ndarray, q_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Interpolate a sign-continuous quaternion time series at arbitrary times.

    t_src must be strictly increasing. Queries outside the range are clamped to the ends.
    Uses exact slerp between the two bracketing samples (vectorized).
    """
    t_src = np.asarray(t_src, dtype=np.float64)
    q_src = qfix_sign(q_src)
    tq = np.asarray(t_query, dtype=np.float64)
    flat = tq.reshape(-1)
    i = np.clip(np.searchsorted(t_src, flat, side='right') - 1, 0, len(t_src) - 2)
    t0, t1 = t_src[i], t_src[i + 1]
    u = np.clip((flat - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)[:, None]
    q0, q1 = q_src[i], q_src[i + 1]
    d = qmul(qconj(q0), q1)
    out = qmul(q0, qexp(qlog(d) * u))
    return qnormalize(out).reshape(tq.shape + (4,))


# ----------------------------------------------------------------------------- lens


@dataclass
class Lens:
    """Source camera intrinsics.

    model: 'kb4'     OpenCV-fisheye / Kannala-Brandt 4-coefficient model (what DJI embeds):
                     theta = atan2(sqrt(X^2+Y^2), Z);  theta_d = theta*(1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)
                     u = fx * theta_d * X / r_xy + cx ;  v = fy * theta_d * Y / r_xy + cy
           'pinhole' rectilinear, k ignored.
    Pixel centres at integers; a lens centred on a W x H image has cx = (W-1)/2, cy = (H-1)/2.
    """
    model: str
    fx: float
    fy: float
    cx: float
    cy: float
    k: np.ndarray = field(default_factory=lambda: np.zeros(4))
    width: int = 0
    height: int = 0

    def scaled(self, s: float) -> 'Lens':
        """Same lens for an image resized by factor s (pixel-centre convention preserved)."""
        return Lens(self.model, self.fx * s, self.fy * s, (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5,
                    np.array(self.k, dtype=np.float64), int(round(self.width * s)), int(round(self.height * s)))

    # --- rays -> pixels
    def project(self, rays: np.ndarray) -> np.ndarray:
        """Camera-frame rays (…,3) -> pixels (…,2). Rays with z <= 0 are still mapped (theta > 90 deg);
        callers must check validity via `valid_ray`."""
        X, Y, Z = np.moveaxis(np.asarray(rays, dtype=np.float64), -1, 0)
        rxy = np.sqrt(X * X + Y * Y)
        if self.model == 'pinhole':
            Zs = np.where(np.abs(Z) < 1e-12, 1e-12, Z)
            return np.stack([self.fx * X / Zs + self.cx, self.fy * Y / Zs + self.cy], axis=-1)
        th = np.arctan2(rxy, Z)
        t2 = th * th
        k1, k2, k3, k4 = [float(c) for c in self.k]
        thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        s = np.where(rxy < 1e-12, 1.0 / np.maximum(Z, 1e-12), thd / np.where(rxy < 1e-12, 1.0, rxy))
        return np.stack([self.fx * X * s + self.cx, self.fy * Y * s + self.cy], axis=-1)

    # --- pixels -> unit rays
    def unproject(self, pix: np.ndarray, iters: int = 8) -> np.ndarray:
        pix = np.asarray(pix, dtype=np.float64)
        mx = (pix[..., 0] - self.cx) / self.fx
        my = (pix[..., 1] - self.cy) / self.fy
        if self.model == 'pinhole':
            r = np.stack([mx, my, np.ones_like(mx)], axis=-1)
            return r / np.linalg.norm(r, axis=-1, keepdims=True)
        thd = np.sqrt(mx * mx + my * my)
        k1, k2, k3, k4 = [float(c) for c in self.k]
        th = thd.copy()
        for _ in range(iters):  # Newton on f(th) = th*(1+k1 th^2+...) - thd
            t2 = th * th
            f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - thd
            df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
            th = th - f / df
        s = np.where(thd < 1e-12, 1.0, np.sin(th) / np.where(thd < 1e-12, 1.0, thd))
        return np.stack([mx * s, my * s, np.cos(th)], axis=-1)

    def valid_ray(self, rays: np.ndarray, max_theta_deg: float = 89.0) -> np.ndarray:
        X, Y, Z = np.moveaxis(np.asarray(rays, dtype=np.float64), -1, 0)
        return np.arctan2(np.sqrt(X * X + Y * Y), Z) < np.deg2rad(max_theta_deg)


def pinhole_K(fx: float, width: int, height: int) -> np.ndarray:
    """Output virtual camera matrix (square pixels, centred, pixel-centre convention)."""
    return np.array([[fx, 0.0, (width - 1) / 2.0], [0.0, fx, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def hfov_to_fx(hfov_deg: float, width: int) -> float:
    return (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)


# ----------------------------------------------------------------------------- row timing


def row_time(frame_t: np.ndarray, y: np.ndarray, height: int, readout_s: float) -> np.ndarray:
    """Capture time of source row y (pixel-centre coords) of a frame whose CENTRE row was captured
    (mid-exposure) at frame_t. Top-to-bottom readout; row y spans readout*((y+0.5)/H - 0.5)."""
    return np.asarray(frame_t, dtype=np.float64) + readout_s * ((np.asarray(y, dtype=np.float64) + 0.5) / height - 0.5)
