"""WP-C tests: crop-constrained path optimizer (engine/stillpoint/smooth.py).

Synthetic O3-like telemetry (KB4 lens, 3840x2160, 9.72 ms readout, 2 kHz orientation): a smooth
intentional path + injected 3-40 Hz jitter + 360-degree barrel rolls peaking at 800 deg/s; 2-4 Hz roll rocking (e);
fast 90-degree turns with 2-4 Hz shake (f); horizon lock v2 in a gravity-aligned (NED) world: cruise,
strength / roll limit, a crop-limited banked turn, a 1200 deg/s flip and a pitch loop through nadir/zenith.
Set STILLPOINT_FAST=1 to skip the 23,500-frame runtime test.
"""
import os
import time

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from stillpoint.geom import Lens, qexp, qlog, qmul, qconj, slerp_series
from stillpoint.smooth import (SmoothParams, _jr_inv, check_crop, fx_for_crop_area, gravity_world, horizon_angles,
                               horizon_jacobian, map_output_points, optimize_path)
from stillpoint.types import Telemetry

FPS = 60000 / 1001
O3_K = np.array([0.24991769, 0.01360575, -0.06208358, 0.01219307])
OUT_W, OUT_H = 1920, 1080


# ----------------------------------------------------------------------------- synthetic telemetry


def _intent(t, rolls=(), roll_dur=0.9, seed=0):
    rng = np.random.default_rng(seed)
    yaw, pitch, roll = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    for f, a in [(0.05, 0.6), (0.11, 0.3), (0.23, 0.15), (0.4, 0.05)]:
        yaw += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
        pitch += 0.3 * a * np.sin(2 * np.pi * f * 1.3 * t + rng.uniform(0, 6.28))
        roll += 0.25 * a * np.sin(2 * np.pi * f * 0.9 * t + rng.uniform(0, 6.28))
    for te in rolls:  # 360 deg barrel roll, raised-cosine rate profile -> peak 2*360/0.9 = 800 deg/s
        u = np.clip((t - te) / roll_dur, 0, 1)
        roll += 2 * np.pi * (u - np.sin(2 * np.pi * u) / (2 * np.pi))
    ey, ex, ez = np.eye(3)[1], np.eye(3)[0], np.eye(3)[2]
    return qmul(qmul(qexp(yaw[:, None] * ey), qexp(pitch[:, None] * ex)), qexp(roll[:, None] * ez))


def _jitter(t, rms_deg=0.3, seed=1, fmin=3.0, fmax=40.0):
    rng = np.random.default_rng(seed)
    j = np.zeros((len(t), 3))
    for f in np.exp(np.linspace(np.log(fmin), np.log(fmax), 12)):
        j += np.sin(2 * np.pi * f * t[:, None] + rng.uniform(0, 6.28, 3)) * rng.uniform(0.5, 1.0, 3)
    return j * (np.deg2rad(rms_deg) / j.std(axis=0))


def make_tel(n_frames, rolls=(), jit_rms_deg=0.3, imu_rate=2000.0, extra_jitter=None, seed=0, gravity=False):
    T = n_frames / FPS
    imu_t = np.arange(-0.2, T + 0.2, 1.0 / imu_rate)
    q_int = _intent(imu_t, rolls, seed=seed)
    J = _jitter(imu_t, jit_rms_deg, seed=seed + 1)
    if extra_jitter is not None:
        J = J + extra_jitter(imu_t)
    q = qmul(q_int, qexp(J))
    W, H = 3840, 2160
    lens = Lens('kb4', 1405.129, 1405.129, (W - 1) / 2, (H - 1) / 2, O3_K.copy(), W, H)
    pts = np.arange(n_frames) / FPS
    tel = Telemetry(source='synthetic', camera='synthetic', width=W, height=H, fps=FPS, frame_pts=pts,
                    frame_t=pts.copy(), exposure_s=np.full(n_frames, 0.002), readout_s=0.00972, lens=lens,
                    imu_t=imu_t, imu_q=q, imu_rate=imu_rate, has_highrate=True, eis_baked=False,
                    segments=[(0, n_frames - 1)])
    if gravity:   # world rotated by a fixed tilt: gravity world = Rg * imu world
        tel.gravity_q = qmul(np.broadcast_to(qexp(np.array([0.3, -0.2, 0.0])), q.shape), q)
    tel.extra['intent_q'] = q_int
    return tel


def _qfn(tel):
    return lambda t: tel.orientation_at(t)


def _hp(x, fc=2.0):
    return sosfiltfilt(butter(4, fc, 'highpass', fs=FPS, output='sos'), x, axis=0)


# ----------------------------------------------------------------------------- fixtures

N_MAIN = 3000
ROLLS = (N_MAIN / FPS * 0.35, N_MAIN / FPS * 0.7)


@pytest.fixture(scope='module')
def rolls_case():
    tel = make_tel(N_MAIN, rolls=ROLLS)
    fx = fx_for_crop_area(tel.lens, tel.width, tel.height, OUT_W, OUT_H, 0.75)
    prm = SmoothParams(min_out_fx=fx)
    t0 = time.perf_counter()
    V, ofx, info = optimize_path(tel, _qfn(tel), np.arange(N_MAIN), OUT_W, OUT_H, prm, return_info=True)
    info['wall'] = time.perf_counter() - t0
    return tel, prm, V, ofx, info


# ----------------------------------------------------------------------------- unit tests


def test_jr_inv_matches_numeric():
    rng = np.random.default_rng(3)
    for _ in range(5):
        th = rng.normal(size=3) * 0.8
        d = rng.normal(size=3) * 1e-6
        lhs = qlog(qmul(qexp(th), qexp(d)))
        rhs = th + _jr_inv(th) @ d
        assert np.allclose(lhs, rhs, atol=1e-11)
        lhs2 = qlog(qmul(qexp(-d), qexp(th)))           # left Jacobian: Exp(-d)Exp(th) = Exp(th - Jl^-1 d)
        assert np.allclose(lhs2, th - _jr_inv(-th) @ d, atol=1e-11)


def test_mapping_jacobian_matches_finite_differences():
    """Crop-constraint Jacobians (incl. the rolling-shutter row coupling) vs exact finite differences,
    during a fast 700 deg/s pitch+roll where the RS coupling matters."""
    tel = make_tel(120, rolls=(0.5,), jit_rms_deg=0.0)
    w = np.deg2rad(np.array([500.0, 0.0, 500.0]))
    t_imu = tel.imu_t
    tel.imu_q = qexp(t_imu[:, None] * w[None, :])
    qfn = _qfn(tel)
    ft = tel.frame_t[40:44]
    V = qfn(ft)
    fo = np.full(4, 900.0)
    pts = np.array([[0.0, 0.0], [OUT_W - 1.0, 0.0], [700.0, OUT_H - 1.0], [0.0, 500.0]])
    p, Gd, Gz = map_output_points(qfn, ft, V, fo, pts, OUT_W, OUT_H, tel.lens, tel.readout_s, tel.height, 6, jac=True)
    h = 1e-6
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        pp = map_output_points(qfn, ft, qmul(V, qexp(e)), fo, pts, OUT_W, OUT_H, tel.lens, tel.readout_s, tel.height, 6)
        pm = map_output_points(qfn, ft, qmul(V, qexp(-e)), fo, pts, OUT_W, OUT_H, tel.lens, tel.readout_s, tel.height, 6)
        num = (pp - pm) / (2 * h)
        assert np.max(np.abs(num - Gd[..., :, a])) < 2e-3 * np.max(np.abs(num)) + 0.5
    pz = map_output_points(qfn, ft, V, fo * np.exp(h), pts, OUT_W, OUT_H, tel.lens, tel.readout_s, tel.height, 6)
    pz2 = map_output_points(qfn, ft, V, fo * np.exp(-h), pts, OUT_W, OUT_H, tel.lens, tel.readout_s, tel.height, 6)
    assert np.max(np.abs((pz - pz2) / (2 * h) - Gz)) < 2e-3 * np.max(np.abs(Gz)) + 0.5


def test_fx_for_crop_area_monotone():
    tel = make_tel(10)
    f75 = fx_for_crop_area(tel.lens, 3840, 2160, OUT_W, OUT_H, 0.75)
    f60 = fx_for_crop_area(tel.lens, 3840, 2160, OUT_W, OUT_H, 0.60)
    assert f60 > f75 > 0


# ----------------------------------------------------------------------------- (a) jitter rejection


def test_a_jitter_rejected_when_slack(rolls_case):
    tel, prm, V, ofx, info = rolls_case
    I = slerp_series(tel.imu_t, tel.extra['intent_q'], tel.frame_t)
    R = _qfn(tel)(tel.frame_t)
    e_v = _hp(qlog(qmul(qconj(I), V)))       # >2 Hz content of the virtual path (vs the smooth intent)
    e_r = _hp(qlog(qmul(qconj(I), R)))       # injected jitter as seen at frame times
    # "constraints slack": no crop sample within 0.5 px of the border in a +-0.25 s neighbourhood, away from
    # the rolls. (At 75% area a 16:9 rectilinear crop of this fisheye has only ~1 deg of vertical headroom,
    # so with 0.3 deg rms jitter the path touches the top/bottom border fairly often.)
    slack = info['slack_min_px']
    near = np.convolve((slack < 0.5).astype(float), np.ones(31), mode='same') > 0
    t = tel.frame_t
    mask = ~near
    for te in ROLLS:
        mask &= ~((t > te - 1.0) & (t < te + 1.9))
    mask[:90] = False
    mask[-90:] = False
    assert mask.sum() > 300
    rv = np.sqrt((e_v[mask] ** 2).sum(1).mean())
    rr = np.sqrt((e_r[mask] ** 2).sum(1).mean())
    ratio = rv / rr
    print(f'\n(a) HF>2Hz residual {np.rad2deg(rv):.4f} deg vs injected {np.rad2deg(rr):.4f} deg -> {100 * ratio:.2f}% '
          f'({mask.sum()} slack frames); runtime {info["wall"]:.1f}s')
    assert ratio < 0.05
    # through the barrel rolls the virtual camera follows the camera (it must: the crop forces it)
    k = int(np.searchsorted(t, ROLLS[0] + 0.45))
    rate_v = np.rad2deg(np.linalg.norm(qlog(qmul(qconj(V[k]), V[k + 1])))) * FPS
    assert rate_v > 600.0


# ----------------------------------------------------------------------------- (b) crop never violated


def test_b_no_crop_violation_dense_rs_exact(rolls_case):
    tel, prm, V, ofx, info = rolls_case
    ids = np.arange(tel.n_frames)
    v0 = check_crop(tel, _qfn(tel), ids, V, ofx, OUT_W, OUT_H, n_per_edge=96, margin_px=0.0)
    vm = check_crop(tel, _qfn(tel), ids, V, ofx, OUT_W, OUT_H, n_per_edge=96, margin_px=prm.margin_px)
    print(f'\n(b) dense border (96/edge, RS rows exact): max excursion beyond source edge {v0.max():+.3f} px, '
          f'beyond the {prm.margin_px:g}px margin {vm.max():+.3f} px; binding frames {info["n_binding"]}')
    assert v0.max() <= 0.0            # every output border pixel maps inside the source, all frames
    assert vm.max() <= 0.5            # and (up to border-sampling / SQP tolerance) inside the margin


# ----------------------------------------------------------------------------- (c) zoom


def test_c_zoom_piecewise_constant():
    """A 3 s burst of violent 4 Hz pitch shake (±2.5°) exceeds the vertical crop margin: the optimizer must
    zoom in (rate-limited, once) instead of letting the shake through, and hold zoom constant otherwise."""
    n = 1800
    t0b, t1b = 12.0, 15.0

    def burst(t):
        env = np.clip(np.minimum(t - t0b, t1b - t) / 0.3, 0, 1)
        j = np.zeros((len(t), 3))
        j[:, 0] = np.deg2rad(2.5) * env * np.sin(2 * np.pi * 4.0 * t)
        return j

    tel = make_tel(n, extra_jitter=burst, jit_rms_deg=0.15, seed=5)
    fx = fx_for_crop_area(tel.lens, tel.width, tel.height, OUT_W, OUT_H, 0.75)
    prm = SmoothParams(min_out_fx=fx)
    V, ofx, info = optimize_path(tel, _qfn(tel), np.arange(n), OUT_W, OUT_H, prm, return_info=True)
    z = np.log(ofx / fx)
    dzs = np.diff(z)
    dz = np.abs(dzs)
    changing = dz > 1e-5
    edges = np.flatnonzero(np.diff(np.concatenate([[0], changing.astype(int), [0]])))
    nets = np.array([dzs[a:b].sum() for a, b in zip(edges[::2], edges[1::2])])
    big = np.abs(nets) > 0.005
    print(f'\n(c) max zoom {ofx.max() / fx:.4f}x, frames with zoom changing {changing.sum()} of {n - 1}, '
          f'{len(nets)} change runs (net log-zoom {np.round(nets, 4).tolist()}), max rate {dz.max() * FPS:.4f}/s')
    assert ofx.min() >= fx * (1 - 1e-9)
    assert ofx.max() > fx * 1.01                         # zoom was needed and used
    assert dz.max() * FPS <= prm.max_zoom_rate * 1.05    # rate limit (no breathing)
    assert big.sum() == 2 and nets[big][0] > 0 > nets[big][1]   # one ramp in, one ramp out
    assert info['zoom_changes'] == int((np.abs(dzs) > 1e-4).sum()) and len(nets) <= 4   # counted; no breathing
    assert np.abs(nets[~big]).sum() < 0.005              # anything else is < 0.5% total (invisible)
    assert changing.mean() < 0.35
    t = tel.frame_t
    assert np.all(np.abs(z[(t < 5.0)] - z[0]) < 1e-5)   # flat far from the burst
    v0 = check_crop(tel, _qfn(tel), np.arange(n), V, ofx, OUT_W, OUT_H, 64, 0.0)
    assert v0.max() <= 0.0


def test_zoom_disabled_and_infeasible_crop_follows_camera():
    """Crop so tight (80% area, zoom off) that even the raw camera path violates it in every frame, plus an
    800 deg/s barrel roll: must not fail — out_fx stays fixed, the path follows the roll, and the violation
    is never worse than simply using the raw camera orientation."""
    n = 600
    tel = make_tel(n, rolls=(4.0,), seed=2)
    qfn = _qfn(tel)
    fx = fx_for_crop_area(tel.lens, tel.width, tel.height, OUT_W, OUT_H, 0.80)
    V, ofx, info = optimize_path(tel, qfn, np.arange(n), OUT_W, OUT_H, SmoothParams(min_out_fx=fx, allow_zoom=False),
                                 return_info=True)
    assert np.all(np.isfinite(V)) and np.allclose(ofx, fx)
    R = qfn(tel.frame_t)
    v_raw = check_crop(tel, qfn, np.arange(n), R, np.full(n, fx), OUT_W, OUT_H, 64, 0.0)
    v_opt = check_crop(tel, qfn, np.arange(n), V, ofx, OUT_W, OUT_H, 64, 0.0)
    assert v_raw.min() > 0                       # genuinely infeasible everywhere
    assert v_opt.max() <= v_raw.max() + 0.5
    k = int(np.searchsorted(tel.frame_t, 4.45))  # mid-roll, 800 deg/s
    rv = np.linalg.norm(qlog(qmul(qconj(V[k]), V[k + 1])))
    rc = np.linalg.norm(qlog(qmul(qconj(R[k]), R[k + 1])))
    assert 0.9 < rv / rc < 1.1


# ----------------------------------------------------------------------------- horizon lock v2


def _mat_q(M):
    from stillpoint.geom import mat_to_quat
    return mat_to_quat(np.asarray(M, dtype=np.float64)[None])[0]


# level camera looking north in a NED world: camera x = east, y = down, z = north
Q_LEVEL = _mat_q([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def make_grav_tel(n_frames, yaw_fn, elev_fn, roll_fn, jit_rms_deg=0.3, seed=1):
    """Synthetic O3-like telemetry in a gravity-aligned world (NED, z down; gravity_q is imu_q as on the O3):
    yaw about world down, elevation (+ = looking down) about the camera x axis, then roll about the optical axis,
    plus 3-40 Hz jitter."""
    T = n_frames / FPS
    t = np.arange(-0.2, T + 0.2, 1.0 / 2000.0)
    ex, ez = np.eye(3)[0], np.eye(3)[2]
    q = qmul(qexp(yaw_fn(t)[:, None] * ez), np.broadcast_to(Q_LEVEL, (len(t), 4)))
    q = qmul(q, qexp(-elev_fn(t)[:, None] * ex))
    q = qmul(q, qexp(roll_fn(t)[:, None] * ez))
    q = qmul(q, qexp(_jitter(t, jit_rms_deg, seed=seed)))
    W, H = 3840, 2160
    lens = Lens('kb4', 1405.129, 1405.129, (W - 1) / 2, (H - 1) / 2, O3_K.copy(), W, H)
    pts = np.arange(n_frames) / FPS
    tel = Telemetry(source='synthetic', camera='synthetic', width=W, height=H, fps=FPS, frame_pts=pts,
                    frame_t=pts.copy(), exposure_s=np.full(n_frames, 0.002), readout_s=0.00972, lens=lens,
                    imu_t=t, imu_q=q, imu_rate=2000.0, has_highrate=True, eis_baked=False,
                    segments=[(0, n_frames - 1)])
    tel.gravity_q = tel.imu_q
    return tel


def _hz_run(tel, area=0.6, **kw):
    fx = fx_for_crop_area(tel.lens, tel.width, tel.height, OUT_W, OUT_H, area)
    V, ofx, info = optimize_path(tel, _qfn(tel), np.arange(tel.n_frames), OUT_W, OUT_H,
                                 SmoothParams(min_out_fx=fx, **kw), return_info=True)
    rho, elev, _ = horizon_angles(V, gravity_world(tel, tel.frame_t))
    crop = check_crop(tel, _qfn(tel), np.arange(tel.n_frames), V, ofx, OUT_W, OUT_H, 64, 0.0)
    return V, ofx, info, np.rad2deg(rho), crop


def _hf_deg(V, sl=slice(None)):
    b = _band(V)[sl]
    return float(np.rad2deg(np.sqrt((b ** 2).sum(1).mean())))


N_HZ = 900
DEG = np.deg2rad


@pytest.fixture(scope='module')
def cruise_case():
    """30 s of cruise: slow yaw wander, 10 deg nose-down, bank rocking +-8 deg at 0.08 Hz."""
    tel = make_grav_tel(N_HZ, lambda t: 0.3 * np.sin(2 * np.pi * 0.05 * t), lambda t: DEG(10) + 0.05 * np.sin(0.7 * t),
                        lambda t: DEG(8) * np.sin(2 * np.pi * 0.08 * t + 0.4))
    return tel, _hz_run(tel), _hz_run(tel, horizon_lock=1.0)


def test_horizon_angle_and_jacobian():
    """rho = atan2(g_x, g_y) is the horizon tilt (level camera 0, rolled camera = its roll, independent of the
    elevation) and horizon_jacobian is its exact derivative for V -> V Exp(delta)."""
    g_w = np.array([[0.0, 0.0, 1.0]])
    for el in (-40.0, 0.0, 30.0, 70.0):
        for r in (-120.0, -15.0, 0.0, 25.0, 170.0):
            V = qmul(qmul(Q_LEVEL[None], qexp(np.array([[-DEG(el), 0, 0]]))), qexp(np.array([[0, 0, DEG(r)]])))
            rho, elev, _ = horizon_angles(V, g_w)
            assert abs(np.rad2deg(rho[0]) - r) < 1e-9 and abs(np.rad2deg(elev[0]) - el) < 1e-9
    rng = np.random.default_rng(0)
    V = qexp(rng.normal(size=(50, 3)))
    gw = np.tile([0.0, 0.0, 1.0], (50, 1))
    rho, elev, g = horizon_angles(V, gw)
    keep = np.abs(np.rad2deg(elev)) < 75
    J = horizon_jacobian(g)
    h = 1e-6
    for a in range(3):
        e = np.zeros((50, 3))
        e[:, a] = h
        rp, _, _ = horizon_angles(qmul(V, qexp(e)), gw)
        rm, _, _ = horizon_angles(qmul(V, qexp(-e)), gw)
        num = np.angle(np.exp(1j * (rp - rm))) / (2 * h)
        np.testing.assert_allclose(J[keep, a], num[keep], atol=1e-5)


def test_horizon_lock_levels_cruise(cruise_case):
    """Full lock: the output horizon is level (to 0.05 deg) wherever the crop allows it (here everywhere), with no
    added 2-8 Hz motion and no crop violation; off, the path keeps the camera's bank."""
    tel, (V0, f0, i0, r0, c0), (V1, f1, i1, r1, c1) = cruise_case
    sl = slice(60, N_HZ - 60)
    print(f'\n(h1) |horizon| p95 off {np.percentile(np.abs(r0), 95):.2f} deg -> lock {np.percentile(np.abs(r1), 95):.3f}; '
          f'2-8 Hz {_hf_deg(V0, sl):.4f} -> {_hf_deg(V1, sl):.4f} deg; {i1["horizon"]}')
    assert np.percentile(np.abs(r0), 95) > 6.0
    assert i1['horizon_active'] and np.abs(r1).max() < 0.05
    assert _hf_deg(V1, sl) <= 1.1 * _hf_deg(V0, sl) + 1e-4
    assert c1.max() <= 0.0 and np.allclose(f1, f1[0])


def test_horizon_strength_and_roll_limit(cruise_case):
    """strength 0.5 halves the bank; a 5 deg roll limit leaves |bank| <= 5 deg untouched and clips the rest."""
    tel, (V0, f0, i0, r0, c0), _ = cruise_case
    _, _, ih, rh, ch = _hz_run(tel, horizon_lock=0.5)
    big = np.abs(r0) > 2.0
    ratio = rh[big] / r0[big]
    assert 0.47 < np.median(ratio) < 0.53 and ch.max() <= 0.0
    _, _, il, rl, cl = _hz_run(tel, horizon_lock=1.0, roll_limit_deg=5.0)
    assert np.abs(rl).max() < 5.05 and cl.max() <= 0.0
    inside = np.abs(r0) < 3.0
    assert np.median(np.abs(rl[inside] - r0[inside])) < 0.05  # free band: the smoother's own roll (up to the rounded
    assert np.abs(rl[inside] - r0[inside]).max() < 1.0       # corners of the clipped stretches next to it)
    assert np.abs(rl[np.abs(r0) > 6.0]).min() > 4.9           # beyond: clipped to the limit, not leveled


def test_horizon_off_and_bool():
    """Strength 0 with gravity present is bit-identical to no gravity; True means strength 1."""
    tel = make_grav_tel(300, lambda t: 0 * t, lambda t: DEG(5) + 0 * t, lambda t: DEG(4) + 0 * t, seed=3)
    fx = fx_for_crop_area(tel.lens, tel.width, tel.height, OUT_W, OUT_H, 0.6)
    Va, fa = optimize_path(tel, _qfn(tel), np.arange(300), OUT_W, OUT_H, SmoothParams(min_out_fx=fx))
    tel.gravity_q = None
    Vb, fb = optimize_path(tel, _qfn(tel), np.arange(300), OUT_W, OUT_H, SmoothParams(min_out_fx=fx, horizon_lock=1.0))
    assert np.array_equal(Va, Vb) and np.array_equal(fa, fb)
    tel.gravity_q = tel.imu_q
    assert SmoothParams(horizon_lock=True).lock_strength() == 1.0
    _, _, _, rt, _ = _hz_run(tel, horizon_lock=True)
    assert np.abs(rt).max() < 0.05


def test_horizon_lock_crop_limited_bank():
    """A sustained 35 deg banked turn at a Gyroflow-like crop: full leveling does not fit, so the lock levels as far
    as the crop allows (with margin) -- smoothly: no crop violation, no zoom, and no extra 2-8 Hz motion (a target on
    the crop border would make the path copy the camera's shake along it)."""
    n = 1200
    tel = make_grav_tel(n, lambda t: 0.4 * t, lambda t: DEG(10) + 0 * t,
                        lambda t: DEG(35) * np.clip(np.minimum(t - 4.0, 16.0 - t) / 1.5, 0, 1), seed=5)
    V0, f0, i0, r0, c0 = _hz_run(tel)
    V1, f1, i1, r1, c1 = _hz_run(tel, horizon_lock=1.0)
    t = tel.frame_t
    mid = (t > 7) & (t < 13)
    calm = (t < 3.0) | (t > 17.5)
    sl = slice(60, n - 60)
    print(f'\n(h2) mid-turn |horizon| off {np.median(np.abs(r0[mid])):.1f} -> lock {np.median(np.abs(r1[mid])):.1f} deg; '
          f'calm {np.abs(r1[calm]).max():.3f}; 2-8 Hz {_hf_deg(V0, sl):.4f} -> {_hf_deg(V1, sl):.4f}; crop '
          f'{c1.max():+.2f}px; {i1["horizon"]}')
    assert np.median(np.abs(r0[mid])) > 30.0
    assert np.median(np.abs(r1[mid])) < np.median(np.abs(r0[mid])) - 4.0      # partly leveled ...
    assert np.median(np.abs(r1[mid])) > 5.0                                     # ... not fully (crop)
    assert np.abs(r1[calm]).max() < 0.1                                         # level again after the turn
    assert c1.max() <= 0.0 and np.allclose(f1, f1[0])
    assert _hf_deg(V1, sl) <= 1.15 * _hf_deg(V0, sl) + 2e-4
    d2 = np.diff(np.unwrap(DEG(r1)), 2) * FPS ** 2                              # horizon angular acceleration
    d2o = np.diff(np.unwrap(DEG(r0)), 2) * FPS ** 2
    assert np.percentile(np.abs(d2[sl]), 99) <= 1.5 * np.percentile(np.abs(d2o[sl]), 99) + DEG(20)


def test_horizon_lock_graceful_flip():
    """A 360 deg roll in 0.6 s (peak 1200 deg/s): the lock fades out before it (lookahead), the path follows the flip
    (the crop forces it), and the lock re-enters smoothly after: level away from it, no crop violation, no extra
    2-8 Hz motion around it, no angular-rate overshoot."""
    n = 1200
    tf = 10.0

    def roll(t):
        u = np.clip((t - tf) / 0.6, 0, 1)
        return DEG(5) * np.sin(0.5 * t) + 2 * np.pi * (u - np.sin(2 * np.pi * u) / (2 * np.pi))
    tel = make_grav_tel(n, lambda t: 0.2 * np.sin(0.1 * t), lambda t: DEG(10) + 0 * t, roll, seed=6)
    V0, f0, i0, r0, c0 = _hz_run(tel)
    V1, f1, i1, r1, c1 = _hz_run(tel, horizon_lock=1.0)
    t = tel.frame_t
    k = int(np.searchsorted(t, tf + 0.3))
    rate = np.rad2deg(np.linalg.norm(qlog(qmul(qconj(V1[:-1]), V1[1:])), axis=1)) * FPS
    rate0 = np.rad2deg(np.linalg.norm(qlog(qmul(qconj(V0[:-1]), V0[1:])), axis=1)) * FPS
    near = (t > tf - 1.5) & (t < tf + 2.1)
    away = ((t < tf - 1.5) | (t > tf + 2.1)) & (t > 1) & (t < t[-1] - 1)
    print(f'\n(h3) flip: mid rate {rate[k]:.0f} deg/s, max {rate.max():.0f} (off {rate0.max():.0f}); away |horizon| max '
          f'{np.abs(r1[away]).max():.3f}; 2-8 Hz near {_hf_deg(V0[near]):.4f} -> {_hf_deg(V1[near]):.4f}; crop '
          f'{c1.max():+.2f}; {i1["horizon"]}')
    assert rate[k] > 600.0                                     # follows the flip
    assert rate.max() < 1.1 * rate0.max()                      # no overshoot / snap
    assert np.abs(r1[away]).max() < 0.1                        # level before and after
    assert c1.max() <= 0.0
    assert _hf_deg(V1[near]) <= 1.15 * _hf_deg(V0[near]) + 2e-4
    assert i1['horizon']['frac_off'] > 0.0


def test_horizon_lock_never_zooms_in_further():
    """Leveling never costs crop: with the lock the zoom stays <= the unlocked path's (horizon_zoom=False, default);
    free zoom (horizon_zoom=True) buys leveling with zoom on this crop-limited shaky roll + flip."""
    tf = 8.0

    def roll(t):
        u = np.clip((t - tf) / 0.6, 0, 1)
        return DEG(20) * np.sin(0.5 * t) + 2 * np.pi * (u - np.sin(2 * np.pi * u) / (2 * np.pi))
    tel = make_grav_tel(900, lambda t: 0.3 * np.sin(0.2 * t), lambda t: DEG(15) + 0.2 * np.sin(0.9 * t), roll,
                        seed=9, jit_rms_deg=0.8)
    V0, f0, _, r0, _ = _hz_run(tel, area=0.7)
    V1, f1, _, r1, c1 = _hz_run(tel, area=0.7, horizon_lock=1.0)
    _, f2, _, _, c2 = _hz_run(tel, area=0.7, horizon_lock=1.0, horizon_zoom=True)
    print(f'\n(h5) zoom max: off {f0.max() / f0.min():.4f}, lock {f1.max() / f0.min():.4f} (lock/off max '
          f'{(f1 / f0).max():.6f}), free zoom {f2.max() / f0.min():.4f} (/off max {(f2 / f0).max():.5f})')
    assert f0.max() > 1.002 * f0.min()                        # the unlocked path does zoom here
    assert (f1 / f0).max() < 1.0 + 1e-5                        # the lock never zooms in further
    assert (f2 / f0).max() > 1.002                             # (with free zoom it would)
    assert c1.max() <= 0.0 and c2.max() <= 0.0
    assert np.median(np.abs(r1)) < np.median(np.abs(r0))


def test_horizon_lock_loop_through_nadir():
    """A 360 deg pitch loop in 2.5 s passes straight down and straight up, where an Euler rebuild (Gyroflow) flips
    the roll by 180 deg: the lock fades out, the path stays continuous and never spins faster than the camera."""
    n = 1200
    tl = 9.0

    def elev(t):
        u = np.clip((t - tl) / 2.5, 0, 1)
        return DEG(10) + 2 * np.pi * (u - np.sin(2 * np.pi * u) / (2 * np.pi))
    tel = make_grav_tel(n, lambda t: 0.2 * np.sin(0.1 * t), elev, lambda t: DEG(6) * np.sin(0.4 * t), seed=8)
    V0, f0, i0, r0, c0 = _hz_run(tel)
    V1, f1, i1, r1, c1 = _hz_run(tel, horizon_lock=1.0)
    t = tel.frame_t
    R = _qfn(tel)(t)
    rate_c = np.rad2deg(np.linalg.norm(qlog(qmul(qconj(R[:-1]), R[1:])), axis=1)) * FPS
    rate = np.rad2deg(np.linalg.norm(qlog(qmul(qconj(V1[:-1]), V1[1:])), axis=1)) * FPS
    # fade-out / re-entry: the smoother itself uses the whole crop just before / after the loop (lookahead), so
    # leveling is infeasible there and the crop-feasible fraction ramps out / back in over horizon_crop_fade_s (starts
    # ~2 s before the loop, level again ~2 s after it ends at tl + 2.5 s; Gyroflow's Euler rebuild instead spins the
    # output 180 deg inside the loop)
    away = ((t < tl - 2.25) | (t > tl + 5.0)) & (t > 1) & (t < t[-1] - 1)
    lvl = np.flatnonzero((t > tl + 2.5) & (np.abs(r1) > 0.1))
    t_level = (t[lvl[-1]] - (tl + 2.5)) if len(lvl) else 0.0
    print(f'\n(h4) loop: max rate {rate.max():.0f} deg/s (camera {rate_c.max():.0f}); away |horizon| '
          f'{np.abs(r1[away]).max():.3f}; level again {t_level:.2f} s after the loop; crop {c1.max():+.2f}; '
          f'{i1["horizon"]}')
    assert rate.max() < 1.1 * rate_c.max()
    assert np.abs(r1[away]).max() < 0.1 and t_level < 2.5
    assert c1.max() <= 0.0
    assert i1['horizon']['frac_steep'] > 0.0


# ----------------------------------------------------------------------------- (e) 2-4 Hz roll, fast turns


def _band(Q, lo=2.0, hi=8.0):
    """Band-passed body-frame integrated rotation of a quaternion path (F,3) rad."""
    w = qlog(qmul(qconj(Q[:-1]), Q[1:]))
    P = np.concatenate([np.zeros((1, 3)), np.cumsum(w, 0)])
    return sosfiltfilt(butter(4, [lo, hi], 'bandpass', fs=FPS, output='sos'), P, axis=0)


def _shake(freqs, rms_deg, axes=(0, 1, 2), seed=11):
    def f(t):
        rng = np.random.default_rng(seed)
        j = np.zeros((len(t), 3))
        for fr in freqs:
            j[:, list(axes)] += np.sin(2 * np.pi * fr * t[:, None] + rng.uniform(0, 6.28, len(axes)))
        sd = j.std(axis=0)
        return j * np.where(sd > 0, np.deg2rad(rms_deg) / np.where(sd > 0, sd, 1.0), 0.0)
    return f


def test_e_roll_2_4hz_rejected_when_slack():
    """Roll rocking at 2-4 Hz (0.5 deg rms, the residual the OA4 visual diagnosis flagged) on a gently wandering camera,
    wide crop (60 % area: the path never needs the border): the virtual path must carry < 2 % of it, with no zoom
    change and no crop violation."""
    n = 1200
    tel = make_tel(n, jit_rms_deg=0.0, extra_jitter=_shake((2.1, 2.7, 3.2, 3.8), 0.5, axes=(2,)), seed=3)
    fx = fx_for_crop_area(tel.lens, tel.width, tel.height, OUT_W, OUT_H, 0.60)
    V, ofx, info = optimize_path(tel, _qfn(tel), np.arange(n), OUT_W, OUT_H, SmoothParams(min_out_fx=fx),
                                 return_info=True)
    R = _qfn(tel)(tel.frame_t)
    sl = slice(90, n - 90)
    bv, br = _band(V, 2.0, 4.0)[sl], _band(R, 2.0, 4.0)[sl]
    ratio = np.sqrt((bv[:, 2] ** 2).mean()) / np.sqrt((br[:, 2] ** 2).mean())
    z = np.log(ofx / fx)
    print(f'\n(e) 2-4 Hz roll: virtual/camera {100 * ratio:.2f}%, zoom changes {info["zoom_changes"]}, '
          f'min crop slack {info["slack_min_px"].min():.1f}px')
    assert ratio < 0.02
    assert info['zoom_changes'] == int((np.abs(np.diff(z)) > 1e-4).sum()) == 0 and np.allclose(ofx, fx)
    assert check_crop(tel, _qfn(tel), np.arange(n), V, ofx, OUT_W, OUT_H, 64, 0.0).max() <= 0.0


def test_f_fast_turns_do_not_copy_shake():
    """Two 90-degree yaw turns in 0.8 s (peak 225 deg/s) with 2-4 Hz shake (0.3 deg rms, all axes), 60 % crop.
    The smooth path must lag/lead the camera by more than 6 deg there. M1 started the SQP at the camera and its trust
    radii summed to 6.1 deg, so the path sat on that cap 130 px away from the crop border and copied 12 % of the
    shake (8-12 % on DJI_0027 26-27 s); now ~2.6 % (warm start + per-frame trust radius: 4.7 %; + smoothed jerk L2
    against riding the crop border: 2.6 %), still no crop violation and no zoom."""
    n = 1200
    t_turns, dur, turn = (6.0, 14.0), 0.8, np.deg2rad(90.0)
    tel = make_tel(n, jit_rms_deg=0.0, seed=7)
    t = tel.imu_t
    yaw = 0.2 * np.sin(2 * np.pi * 0.05 * t)
    for te in t_turns:
        u = np.clip((t - te) / dur, 0, 1)
        yaw += turn * (u - np.sin(2 * np.pi * u) / (2 * np.pi))
    tel.imu_q = qmul(qexp(yaw[:, None] * np.eye(3)[1]), qexp(_shake((2.3, 2.9, 3.4, 3.9), 0.3, seed=7)(t)))
    fx = fx_for_crop_area(tel.lens, tel.width, tel.height, OUT_W, OUT_H, 0.60)
    V, ofx, info = optimize_path(tel, _qfn(tel), np.arange(n), OUT_W, OUT_H, SmoothParams(min_out_fx=fx),
                                 return_info=True)
    R = _qfn(tel)(tel.frame_t)
    ft = tel.frame_t
    m = np.zeros(n, bool)
    for te in t_turns:
        m |= (ft > te - 0.5) & (ft < te + dur + 0.5)
    bv, br = _band(V), _band(R)
    leak = np.sqrt((bv[m] ** 2).sum(1).mean()) / np.sqrt((br[m] ** 2).sum(1).mean())
    phi = np.rad2deg(np.linalg.norm(qlog(qmul(qconj(R), V)), axis=1))
    v0 = check_crop(tel, _qfn(tel), np.arange(n), V, ofx, OUT_W, OUT_H, 64, 0.0)
    print(f'\n(f) fast turns: 2-8 Hz virtual/camera {100 * leak:.2f}%, max |V-camera| {phi.max():.1f} deg, '
          f'max excursion {v0.max():+.2f}px, zoom changes {info["zoom_changes"]}, warm {info.get("warm")}')
    assert leak < 0.04
    assert phi.max() > 7.0
    assert v0.max() <= 0.0
    assert info['zoom_changes'] == 0


# ----------------------------------------------------------------------------- (d) runtime


@pytest.mark.skipif(os.environ.get('STILLPOINT_FAST') == '1', reason='slow runtime test')
def test_d_runtime_23500_frames():
    n = 23500
    rolls = tuple(np.linspace(20, 370, 8))
    tel = make_tel(n, rolls=rolls, imu_rate=2000.0)
    fx = fx_for_crop_area(tel.lens, tel.width, tel.height, OUT_W, OUT_H, 0.75)
    t0 = time.perf_counter()
    V, ofx, info = optimize_path(tel, _qfn(tel), np.arange(n), OUT_W, OUT_H, SmoothParams(min_out_fx=fx),
                                 return_info=True)
    dt = time.perf_counter() - t0
    print(f'\n(d) 23,500 frames: {dt:.1f}s ({len(info["iters"])} SQP iterations, {info["windows"]} windows), '
          f'max violation {info["max_violation_px"]:.3f}px, binding {100 * info["frac_binding"]:.1f}%')
    assert dt < 60.0
    assert info['max_violation_px'] < 0.5
