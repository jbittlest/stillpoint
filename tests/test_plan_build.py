"""WP-B tests: plan_build + render_ref geometry (no video needed).

(a) identity sanity: virtual path == camera path (readout 0) and out_fx = lens fx -> the map is exactly the
    rectilinear undistortion of the source (checked against OpenCV's independent fisheye implementation), also
    at a reduced output scale (pixel-centre scaling convention / preview_K).
(c) RS inverse accuracy at 600 deg/s: 3-evaluation (fixed point + secant) residual < 0.05 px.
plus: vectorized build == explicit per-row formula, TimeModel wiring, speed (23k x 32 rows), .spplan round trip,
the Metal kernel's coordinate map (torch MPS) == float64 reference.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from stillpoint.geom import Lens, pinhole_K, qexp, qmul, qnormalize, quat_to_mat, row_time
from stillpoint.plan_build import build_plan, camera_orientation_fn, row_samples_y
from stillpoint.plan_io import read_plan, write_plan
from stillpoint.render_ref import _source_coord, output_grid, preview_K, source_map
from stillpoint.types import Telemetry, TimeModel

W, H = 3840, 2160
FPS = 60000 / 1001
K_O3 = np.array([0.24991769, 0.01360575, -0.06208358, 0.01219307])


def o3_lens():
    return Lens('kb4', 1405.129, 1405.129, (W - 1) / 2, (H - 1) / 2, K_O3.copy(), W, H)


def synth_tel(n_frames=120, rate_fn=None, readout=0.0097247, imu_rate=2000.0, lens=None):
    pts = np.arange(n_frames) * 1001 / 60000
    t = np.arange(-0.5, pts[-1] + 0.5, 1 / imu_rate)
    if rate_fn is None:
        rv = np.stack([0.2 * np.sin(2 * np.pi * 1.7 * t), 0.3 * np.sin(2 * np.pi * 0.9 * t + 1),
                       0.1 * np.sin(2 * np.pi * 3.1 * t + 2)], -1)
    else:
        rv = rate_fn(t)
    return Telemetry(source='synthetic', camera='synthetic', width=W, height=H, fps=FPS, frame_pts=pts,
                     frame_t=pts + 0.004, exposure_s=np.full(n_frames, 0.002), readout_s=readout,
                     lens=lens or o3_lens(), imu_t=t, imu_q=qexp(rv), imu_rate=imu_rate, has_highrate=True,
                     eis_baked=False)


# ------------------------------------------------------------------------------------------ plan_build
def test_build_matches_explicit_formula_and_time_model():
    tel = synth_tel()
    tm = TimeModel(offset_s=0.0031, skew=2e-4, readout_s=0.011, focal_scale=1.01,
                   extrinsic_rotvec=np.array([0.01, -0.02, 0.005]))
    corr = lambda t: qexp(np.stack([1e-3 * np.sin(7 * t), 2e-3 * np.cos(5 * t), 0 * t], -1))
    qf = camera_orientation_fn(tel, tm, correction=corr)
    frames = np.array([3, 10, 57, 99])
    virt = qexp(np.random.default_rng(0).normal(0, 0.2, (len(frames), 3)))
    plan = build_plan(tel, tm, qf, virt, 1600.0, 3840, 2160, n_rows=16, frames=frames)
    # time model wiring of q_cam
    t = np.array([0.1, 0.77, 1.5])
    q_exp = qmul(qmul(tel.orientation_at(t * (1 + 2e-4) + 0.0031), qexp(tm.extrinsic_rotvec)), corr(t))
    q_got = qf(t)
    assert np.allclose(np.abs(np.einsum('ij,ij->i', q_exp, q_got)), 1.0, atol=1e-12)
    # explicit per-row formula
    ys = row_samples_y(H, 16)
    assert ys[0] == 0 and ys[-1] == H - 1
    for a, k in enumerate(frames):
        for j in (0, 5, 15):
            tr = row_time(tel.frame_t[k], ys[j], H, 0.011)
            M = quat_to_mat(qf(np.array(tr)))[0] if np.ndim(tr) else quat_to_mat(qf(np.array([tr])))[0]
            M = M.T @ quat_to_mat(virt[a])
            assert np.allclose(plan.row_mats[a, j], M, atol=1e-12)
    assert plan.lens.fx == pytest.approx(1405.129 * 1.01) and plan.lens.cx == tel.lens.cx
    assert np.array_equal(plan.frame_pts, tel.frame_pts[frames])
    assert plan.meta['readout_s'] == 0.011


def test_build_speed_23k_frames():
    n = 23512
    tel = synth_tel(n_frames=n, imu_rate=2000.0)
    qf = camera_orientation_fn(tel, TimeModel())
    virt = qf(tel.frame_t)
    t0 = time.time()
    plan = build_plan(tel, TimeModel(), qf, virt, 1700.0, 3840, 2160, n_rows=32)
    dt = time.time() - t0
    print(f"build_plan 23512 x 32 rows: {dt:.2f} s")
    assert plan.row_mats.shape == (n, 32, 3, 3)
    assert dt < 10.0


def test_plan_io_roundtrip(tmp_path):
    tel = synth_tel(30)
    qf = camera_orientation_fn(tel, TimeModel())
    plan = build_plan(tel, None, qf, qf(tel.frame_t), np.linspace(1500, 1600, 30), 1920, 1080, n_rows=8)
    write_plan(str(tmp_path / 'p.spplan'), plan)
    p2 = read_plan(str(tmp_path / 'p.spplan'))
    assert np.allclose(p2.row_mats, plan.row_mats, atol=1e-6)
    assert np.array_equal(p2.frame_pts, plan.frame_pts)
    assert np.allclose(p2.out_fx, plan.out_fx, rtol=1e-6)
    assert p2.lens.model == 'kb4' and p2.lens.fx == pytest.approx(plan.lens.fx, rel=1e-6)


# ------------------------------------------------------------------------------------------ (a) identity
def _cv2_undistort_map(lens: Lens, P: np.ndarray, size):
    import cv2
    K = np.array([[lens.fx, 0, lens.cx], [0, lens.fy, lens.cy], [0, 0, 1.0]])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, np.asarray(lens.k, np.float64).reshape(4, 1), np.eye(3), P,
                                                 size, cv2.CV_32FC1)
    return np.stack([m1, m2], -1).astype(np.float64)


@pytest.mark.parametrize('out_scale', [1.0, 0.25])
def test_identity_is_rectilinear_undistortion(out_scale):
    tel = synth_tel(20, readout=0.0097247)
    tm = TimeModel(readout_s=0.0)                  # no RS -> every row sees the centre-row orientation
    qf = camera_orientation_fn(tel, tm)
    plan = build_plan(tel, tm, qf, qf(tel.frame_t), tel.lens.fx, W, H, n_rows=32)
    assert np.abs(plan.row_mats - np.eye(3)).max() < 1e-12
    k = 7
    S, ok = source_map(plan, k, out_scale=out_scale, return_valid=True)
    Wo, Ho, _, _ = output_grid(W, H, out_scale)
    Kp = preview_K(plan, k, out_scale)
    ref = _cv2_undistort_map(plan.lens, Kp, (Wo, Ho))
    # the same pixels in full-res output coordinates must hit the same source pixels
    Kfull = pinhole_K(plan.out_fx[k], W, H)
    assert np.allclose(preview_K(plan, k, 1.0), Kfull)
    inside = ok & (np.abs(ref).max(-1) < 1e5)
    err = np.abs(S - ref)[inside]
    print(f"out_scale={out_scale}: identity vs cv2 fisheye undistort: mean {err.mean():.2e} max {err.max():.2e} px, "
          f"valid {ok.mean():.3f}")
    assert inside.mean() > 0.9
    assert err.max() < 2e-3                        # cv2 stores float32 maps
    # centre pixel maps to the principal point
    if out_scale == 1.0:
        assert np.allclose(S[(H - 1) // 2, (W - 1) // 2], [plan.lens.cx - 0.5, plan.lens.cy - 0.5], atol=0.51)


# ------------------------------------------------------------------------------------------ (c) RS inverse
def _rs_plan(axis, rate_dps=600.0, readout=0.0097247, zoom=1.1):
    w = np.deg2rad(rate_dps) * np.asarray(axis, float) / np.linalg.norm(axis)
    tel = synth_tel(12, rate_fn=lambda t: t[:, None] * w[None, :], readout=readout)
    tm = TimeModel()
    qf = camera_orientation_fn(tel, tm)
    return build_plan(tel, tm, qf, qf(tel.frame_t), tel.lens.fx * zoom, W, H, n_rows=32), tel, qf


@pytest.mark.parametrize('axis,name', [((1, 0, 0), 'pitch'), ((0, 1, 0), 'yaw'), ((0, 0, 1), 'roll'),
                                       ((0.6, 0.5, 0.62), 'mixed')])
def test_rs_inverse_600dps(axis, name):
    plan, tel, qf = _rs_plan(axis)
    k = 5
    Wo, Ho, sx, sy = output_grid(W, H, 0.125)
    X, Y = np.meshgrid((np.arange(Wo) + 0.5) / sx - 0.5, (np.arange(Ho) + 0.5) / sy - 0.5)
    S3, ok, _ = _source_coord(plan, k, X, Y, iters=3)
    Sinf, _, _ = _source_coord(plan, k, X, Y, iters=40)
    Sfp, _, _ = _source_coord(plan, k, X, Y, iters=3, secant=False)
    # exact geometry at the found row (no row-matrix interpolation): R_cam(t(v))^T R_virt r_v
    rv = np.stack([(X - (W - 1) / 2) / plan.out_fx[k], (Y - (H - 1) / 2) / plan.out_fx[k], np.ones_like(X)], -1)
    trow = row_time(tel.frame_t[k], Sinf[..., 1], H, tel.readout_s)
    Rc = quat_to_mat(qf(trow))
    rc = np.einsum('...ji,jk,...k->...i', Rc, quat_to_mat(plan.virt_q[k]), rv)
    Sex = plan.lens.project(rc)
    fixed_point_resid = np.linalg.norm(S3 - Sinf, axis=-1)[ok]
    plain_resid = np.linalg.norm(Sfp - Sinf, axis=-1)[ok]
    geom_err = np.linalg.norm(Sex - Sinf, axis=-1)[ok]
    print(f"{name} 600 deg/s: 3-eval residual max {fixed_point_resid.max():.4f} px (plain fixed point "
          f"{plain_resid.max():.3f} px); row-interp geometry error max {geom_err.max():.4f} px; valid {ok.mean():.2f}")
    assert ok.mean() > 0.5
    assert fixed_point_resid.max() < 0.05
    assert geom_err.max() < 0.05


def test_metal_coord_map_matches_reference():
    torch = pytest.importorskip('torch')
    if not torch.backends.mps.is_available():
        pytest.skip('no MPS')
    from stillpoint.render_ref import metal_coord_map
    plan, _, _ = _rs_plan((0.6, 0.5, 0.62))
    plan.row_mats = plan.row_mats.astype(np.float32).astype(np.float64)   # what the .spplan carries
    for s in (1.0, 0.25):
        Sm, okm = metal_coord_map(plan, 5, out_scale=s)
        Sr, okr = source_map(plan, 5, out_scale=s, return_valid=True)
        both = okm & okr
        err = np.abs(Sm - Sr)[both]
        print(f"metal vs float64 map (scale {s}): mean {err.mean():.2e} max {err.max():.2e} px, "
              f"valid mismatches {(okm != okr).sum()}")
        assert err.mean() < 2e-3 and err.max() < 0.02
        assert (okm != okr).mean() < 1e-4


# ------------------------------------------------------------------------------------------ exposure averaging
def test_exposure_avg_window_ramp():
    from stillpoint.plan_build import exposure_avg_window
    e = np.array([0.0, 0.001, 0.002, 0.00275, 0.0035, 0.0045, np.nan])
    w = exposure_avg_window(e)
    np.testing.assert_allclose(w[:6], [0, 0, 0, 0.00275 * 0.5, 0.0035, 0.0045], atol=1e-12)
    assert w[6] == 0.0


def test_exposure_averaged_q_box_filter():
    """A 250 Hz vibration averaged over a 4 ms exposure is attenuated by sinc(f e); a constant rate is unchanged."""
    from stillpoint.geom import qlog
    from stillpoint.plan_build import exposure_averaged_q
    f, A, e = 250.0, 2e-4, 0.004
    rv = lambda t: np.stack([A * np.sin(2 * np.pi * f * t), 0.05 * t, 0 * t], -1)
    qf = lambda t: qexp(rv(np.asarray(t, np.float64)))
    t = np.linspace(0.1, 0.2, 400)[:, None]
    qa = exposure_averaged_q(qf, t, np.full(400, e), taps=33)
    x = qlog(qa)[..., 0].ravel()
    gain = np.sqrt(2) * x.std() / A
    assert gain == pytest.approx(abs(np.sinc(f * e)), abs=0.02)
    assert np.allclose(qlog(qa)[..., 1].ravel(), 0.05 * t.ravel(), atol=1e-9)     # linear motion: mean = centre


def test_build_plan_exposure_avg_only_for_long_exposures():
    tel = synth_tel(n_frames=30)
    qf = camera_orientation_fn(tel)
    virt = qf(tel.frame_t)
    p0 = build_plan(tel, None, qf, virt, 1600.0, 3840, 2160, n_rows=8)
    assert p0.meta['exposure_avg'] is False                       # 2 ms: below the ramp
    tel.exposure_s = np.full(30, 0.004)
    p1 = build_plan(tel, None, qf, virt, 1600.0, 3840, 2160, n_rows=8)
    p2 = build_plan(tel, None, qf, virt, 1600.0, 3840, 2160, n_rows=8, exposure_avg=False)
    assert p1.meta['exposure_avg'] is True and p2.meta['exposure_avg'] is False
    np.testing.assert_allclose(p2.row_mats, p0.row_mats, atol=1e-12)
    d = np.abs(p1.row_mats - p2.row_mats).max()
    assert 0 < d < 1e-4                                           # slow synthetic motion: tiny change
