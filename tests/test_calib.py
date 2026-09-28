"""WP-E self-calibration tests.

Synthetic: frames rendered from a real still (equirectangular world at infinity, so pure rotation) through the
O3 KB4 fisheye with a known rotation path, rolling shutter and KNOWN offset / readout / focal / extrinsic errors
relative to the telemetry handed to self_calibrate; the calibration must recover them.
Real-footage validation lives in work/calib/run_real.py (too slow for a unit test).
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import pytest

from stillpoint.calib import _Orient, self_calibrate, select_windows
from stillpoint.geom import Lens, qexp, qmul
from stillpoint.types import Telemetry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STILL = os.path.join(ROOT, 'work', 'o3', 'frames', '0025_t12.jpg')

O3_LENS = Lens('kb4', 1405.1293, 1405.1293, 1919.5, 1079.5,
               np.array([0.24991769, 0.01360575, -0.06208358, 0.01219307]), 3840, 2160)
FPS = 60000 / 1001


def _texture() -> np.ndarray:
    """2560x1280 equirectangular texture (407 px/rad) built from a real still (mirror tiled)."""
    if os.path.exists(STILL):
        img = cv2.imread(STILL, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_CUBIC)
    else:  # procedural fallback
        rng = np.random.default_rng(1)
        img = cv2.GaussianBlur((rng.random((720, 1280)) * 255).astype(np.uint8), (0, 0), 2.0)
        img = cv2.equalizeHist(img)
    row = np.hstack([img, img[:, ::-1]])
    pano = np.vstack([row[::-1][360:], row, row[::-1][:360]])        # 1440 rows
    return np.ascontiguousarray(pano[80:80 + 1280]).astype(np.float32)


def _rotvec_path(t: np.ndarray) -> np.ndarray:
    comps = {  # axis: [(amp_deg, f_hz, phase)]
        0: [(5.0, 2.1, 0.3), (1.2, 7.3, 1.1), (0.3, 17.0, 0.2)],
        1: [(7.0, 1.3, 0.0), (1.5, 5.7, 2.0), (0.25, 23.0, 0.7)],
        2: [(6.0, 1.7, 0.9), (1.0, 9.1, 0.4)],
    }
    th = np.zeros((len(t), 3))
    for ax, lst in comps.items():
        for a, f, ph in lst:
            th[:, ax] += np.deg2rad(a) * np.sin(2 * np.pi * f * t + ph)
    return th


def make_synthetic(duration=6.0, readout=9.72e-3):
    F = int(duration * FPS)
    frame_t = np.arange(F) / FPS
    imu_t = np.arange(-0.2, duration + 0.2, 1 / 2000.0)
    imu_q = qexp(_rotvec_path(imu_t))
    tel = Telemetry(source='synthetic', camera='synthetic O3', width=3840, height=2160, fps=FPS,
                    frame_pts=frame_t.copy(), frame_t=frame_t.copy(), exposure_s=np.full(F, 0.002),
                    readout_s=readout, lens=O3_LENS, imu_t=imu_t, imu_q=imu_q, imu_rate=2000.0,
                    has_highrate=True, eis_baked=False)
    return tel


class Renderer:
    """frame_reader for self_calibrate: renders frames with the TRUE time model / lens / extrinsic."""

    def __init__(self, tel: Telemetry, width=960, offset=0.0, readout=None, focal=1.0, ext_deg=(0, 0, 0),
                 noise=1.0, seed=0, velocity=None, radius=40.0):
        """velocity (world m/s) != None: camera translates inside a textured sphere of `radius` m around the
        origin (parallax everywhere, like FPV forward flight); otherwise the world is at infinity."""
        self.tel = tel
        self.velocity = None if velocity is None else np.asarray(velocity, float)
        self.radius = radius
        s = width / tel.width
        L = tel.lens.scaled(s)
        self.lens = Lens(L.model, L.fx * focal, L.fy * focal, L.cx, L.cy, L.k, L.width, L.height)
        self.W, self.H = L.width, L.height
        self.offset, self.readout = offset, (readout if readout is not None else tel.readout_s)
        self.qE = qexp(np.deg2rad(np.asarray(ext_deg, float)))
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float64)
        self.beta = self.lens.unproject(np.stack([xx, yy], -1))            # (H,W,3)
        self.tex = _texture()
        self.orient = _Orient(tel.imu_t, tel.imu_q)
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def frame(self, k: int) -> np.ndarray:
        rows = (np.arange(self.H) + 0.5) / self.H - 0.5
        tau = self.tel.frame_t[k] + self.offset + self.readout * rows
        q = qmul(self.orient(tau), self.qE)
        from stillpoint.geom import quat_to_mat
        R = quat_to_mat(q)                                                  # (H,3,3)
        w = np.einsum('yij,yxj->yxi', R, self.beta)
        if self.velocity is not None:
            c = ((tau[:, None] - 3.0) * self.velocity[None, :])[:, None, :]      # (H,1,3) camera centre per row
            cw = np.sum(c * w, axis=-1, keepdims=True)
            lam = -cw + np.sqrt(cw ** 2 - np.sum(c * c, axis=-1, keepdims=True) + self.radius ** 2)
            w = c + lam * w
            w = w / np.linalg.norm(w, axis=-1, keepdims=True)
        lon = np.arctan2(w[..., 0], w[..., 2])
        lat = np.arcsin(np.clip(w[..., 1], -1, 1))
        th, tw = self.tex.shape
        u = (lon / (2 * np.pi) + 0.5) * tw - 0.5
        v = (lat / np.pi + 0.5) * th - 0.5
        img = cv2.remap(self.tex, u.astype(np.float32), v.astype(np.float32), cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT)
        if self.noise:
            img = img + self.rng.normal(0, self.noise, img.shape)
        return np.clip(np.round(img), 0, 255).astype(np.uint8)

    def __call__(self, f0: int, n: int):
        idx = np.arange(f0, min(f0 + n, self.tel.n_frames))
        return idx, np.stack([self.frame(int(k)) for k in idx])


def test_orient_matches_geom_slerp():
    tel = make_synthetic(1.0)
    t = np.linspace(0.01, 0.99, 777)
    a = _Orient(tel.imu_t, tel.imu_q)(t)
    b = tel.orientation_at(t)
    d = np.minimum(np.linalg.norm(a - b, axis=1), np.linalg.norm(a + b, axis=1))
    assert d.max() < 1e-12


def test_select_windows_prefers_motion_and_skips_static():
    tel = make_synthetic(6.0)
    # freeze the first 3 s (static) -> no window may start there
    q = tel.imu_q.copy()
    q[tel.imu_t < 3.0] = q[np.searchsorted(tel.imu_t, 3.0)]
    tel.imu_q = q
    w = select_windows(tel, 3, 45)
    assert len(w) >= 1
    assert all(tel.frame_t[c['start']] > 2.9 for c in w)


@pytest.mark.parametrize('case', ['errors'])
def test_synthetic_recovery(case):
    tel = make_synthetic(6.0)
    truth = dict(offset_ms=1.2, readout=9.72e-3 * 1.08, focal=1.015, ext_deg=(0.6, -0.9, 0.4))
    rend = Renderer(tel, offset=truth['offset_ms'] * 1e-3, readout=truth['readout'], focal=truth['focal'],
                    ext_deg=truth['ext_deg'])
    t0 = time.time()
    tm = self_calibrate(tel, None, max_windows=3, frame_reader=rend, window_s=0.75)
    dt = time.time() - t0
    n = tm.notes['calib']
    print('\nsynthetic:', n['estimate'], '\nsigma:', n['sigma'], '\nused:', n['used'], f'\n{dt:.1f}s',
          n['runtime_s'], n['residual_default'], n['residual_final'])
    assert abs(tm.offset_s * 1e3 - truth['offset_ms']) < 0.1
    assert tm.readout_s is not None and abs(tm.readout_s / truth['readout'] - 1) < 0.03
    assert abs(tm.focal_scale / truth['focal'] - 1) < 0.003
    assert np.max(np.abs(np.rad2deg(tm.extrinsic_rotvec) - np.array(truth['ext_deg']))) < 0.1


def test_synthetic_recovery_with_parallax():
    """FPV-like forward flight: every point has 1-3 px/frame parallax, so no track is 'far' and the
    depth-free coplanarity residual has to carry the calibration."""
    tel = make_synthetic(6.0)
    truth = dict(offset_ms=-0.8, readout=9.72e-3 * 0.95, focal=0.99, ext_deg=(-0.5, 0.3, 0.7))
    rend = Renderer(tel, offset=truth['offset_ms'] * 1e-3, readout=truth['readout'], focal=truth['focal'],
                    ext_deg=truth["ext_deg"], velocity=(2.0, 0.5, 8.0), radius=50.0)
    tm = self_calibrate(tel, None, max_windows=3, frame_reader=rend, window_s=0.75)
    n = tm.notes['calib']
    print('\nparallax:', n['estimate'], '\nsigma:', n['sigma'], '\nused:', n['used'], n['n_far_pairs'],
          n['n_coplanar_pairs'])
    assert n['used']['offset_ms'] and abs(tm.offset_s * 1e3 - truth['offset_ms']) < 0.1
    if n['used']['readout_pct']:
        assert abs(tm.readout_s / truth['readout'] - 1) < 0.03
    if n['used']['focal_pct']:
        assert abs(tm.focal_scale / truth['focal'] - 1) < 0.003
    assert np.max(np.abs(np.rad2deg(tm.extrinsic_rotvec) - np.array(truth['ext_deg']))) < 0.15


def test_synthetic_no_error_stays_default():
    tel = make_synthetic(6.0)
    rend = Renderer(tel)
    tm = self_calibrate(tel, None, max_windows=3, frame_reader=rend, window_s=0.75)
    assert abs(tm.offset_s) < 0.1e-3
    assert abs(tm.focal_scale - 1) < 0.003
    assert tm.readout_s is None or abs(tm.readout_s / tel.readout_s - 1) < 0.03
    assert np.max(np.abs(np.rad2deg(tm.extrinsic_rotvec))) < 0.1


def test_eis_baked_returns_default():
    tel = make_synthetic(2.0)
    tel.eis_baked = True
    tm = self_calibrate(tel, None, frame_reader=lambda a, b: None)
    assert tm.offset_s == 0.0 and tm.readout_s is None and tm.focal_scale == 1.0


def test_unproject_robust_first_branch():
    from stillpoint.calib import unproject_robust
    oa4 = Lens('kb4', 364.2684, 364.2684, 479.5, 359.5, np.array([0.1551311, 0.1371409, -0.0938614, 0.0041704]),
               960, 720)
    r = np.linspace(0.0, 1.6, 200)
    pix = np.stack([oa4.cx + r * oa4.fx, np.full_like(r, oa4.cy)], -1)
    ray = unproject_robust(oa4, pix)
    th = np.arctan2(np.hypot(ray[:, 0], ray[:, 1]), ray[:, 2])
    assert np.all(np.diff(th) > 0)                         # monotonic: never jumps to the far branch
    assert np.abs(oa4.project(ray) - pix).max() < 1e-6
    o3 = O3_LENS.scaled(0.25)
    pix = np.random.default_rng(0).uniform([0, 0], [959, 539], (500, 2))
    assert np.abs(unproject_robust(o3, pix) - o3.unproject(pix)).max() < 1e-10
