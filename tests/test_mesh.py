"""Mesh residual (engine/stillpoint/mesh.py): plan format extension, reference/Metal geometry, fits, solver, tracking.

Synthetic data only, except test_sprender_mesh_golden (O3 DJI_0026, read-only; skipped without the footage).
"""
from __future__ import annotations

import os
import struct
import subprocess

import numpy as np
import pytest

from stillpoint.geom import Lens, qexp
from stillpoint.mesh import (MeshParams, _track_chunk_job, delta_designs, fit_delta, fit_single, limit_border,
                             mesh_shape, poly_basis, reconstruct_velocity, soft_clamp, solve_fields, vertex_grid,
                             window_accept)
from stillpoint.plan_io import read_plan, write_plan
from stillpoint.render_ref import mesh_offset, source_map, source_points
from stillpoint.types import Plan

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))


def _plan(F=4, W=640, H=360, zoom=1.3, seed=0, mesh=None):
    rng = np.random.default_rng(seed)
    lens = Lens('kb4', 420.0, 420.0, (W - 1) / 2, (H - 1) / 2, np.array([0.1, 0.01, -0.02, 0.005]), W, H)
    R = 8
    mats = np.zeros((F, R, 3, 3))
    for k in range(F):
        for j in range(R):
            mats[k, j] = _rot(rng.normal(0, 0.01, 3) + np.array([0, 0, 0.002 * j]))
    return Plan(src_w=W, src_h=H, out_w=W, out_h=H, lens=lens, frame_pts=np.arange(F) / 60.0,
                out_fx=np.full(F, 420.0 * zoom), row_mats=mats, virt_q=np.zeros((F, 4)), meta={}, mesh=mesh)


def _rot(v):
    from stillpoint.geom import quat_to_mat
    return quat_to_mat(qexp(np.asarray(v, float)))


# ------------------------------------------------------------------------------------------ format
def test_plan_io_v1_unchanged_and_mesh_roundtrip(tmp_path):
    p0 = _plan()
    a = tmp_path / 'a.spplan'
    write_plan(str(a), p0)
    raw = a.read_bytes()
    assert struct.unpack_from('<I', raw, 8)[0] == 1 and struct.unpack_from('<I', raw, 84)[0] == 0
    assert raw[88:256] == bytes(168)                              # v1: zero padding
    assert len(raw) == 256 + p0.n_frames * (24 + 36 * p0.n_rows)
    assert read_plan(str(a)).mesh is None
    # all-zero mesh -> still a plain v1 file
    p0z = _plan(mesh=np.zeros((4, 3, 5, 2), np.float32))
    write_plan(str(a), p0z)
    assert a.read_bytes() == raw
    # mesh extension
    mesh = np.random.default_rng(1).normal(0, 1, (4, 3, 5, 2)).astype(np.float32)
    p1 = _plan(mesh=mesh)
    p1.meta['mesh_clamp_px'] = 5.0
    b = tmp_path / 'b.spplan'
    write_plan(str(b), p1)
    rb = b.read_bytes()
    assert struct.unpack_from('<I', rb, 8)[0] == 1                # v1 readers still accept it ...
    assert rb[:256 + 4 * (24 + 36 * 8)][84:88] != raw[84:88]
    assert rb[256:256 + len(raw) - 256] == raw[256:]             # ... with byte-identical records
    q = read_plan(str(b))
    np.testing.assert_array_equal(q.mesh, mesh)
    assert q.meta['mesh_clamp_px'] == 5.0
    np.testing.assert_allclose(q.row_mats, p1.row_mats, atol=1e-6)
    with pytest.raises(ValueError):
        write_plan(str(b), _plan(mesh=np.ones((3, 3, 5, 2), np.float32)))


def test_mesh_offset_bilinear():
    W, H = 641, 361
    m = np.zeros((3, 5, 2))
    m[1, 2] = (2.0, -1.0)                                         # centre vertex
    X, Y = vertex_grid(W, H, 5, 3)
    dx, dy = mesh_offset(m, X, Y, W, H)
    np.testing.assert_allclose(dx, m[..., 0], atol=1e-12)
    np.testing.assert_allclose(dy, m[..., 1], atol=1e-12)
    # halfway between the centre vertex and its right neighbour
    dx, dy = mesh_offset(m, np.array([X[1, 2] + 80.0]), np.array([Y[1, 2]]), W, H)
    np.testing.assert_allclose([dx[0], dy[0]], [1.0, -0.5], atol=1e-12)
    # clamped outside the grid
    dx, dy = mesh_offset(m, np.array([-50.0, W + 50.0]), np.array([-9.0, H + 9.0]), W, H)
    np.testing.assert_allclose(dx, 0.0, atol=1e-12)


def test_ref_applies_mesh_before_rotation():
    mesh = np.zeros((4, 3, 5, 2), np.float32)
    mesh[2] = (1.5, -0.75)                                        # uniform shift on frame 2
    p0, p1 = _plan(), _plan(mesh=mesh)
    xy = np.array([[100.0, 50.0], [320.0, 180.0], [500.0, 300.0]])
    s1, ok1 = source_points(p1, 2, xy)
    s0, ok0 = source_points(p0, 2, xy + [1.5, -0.75])
    np.testing.assert_allclose(s1, s0, atol=1e-9)
    s2, _ = source_points(p1, 2, xy, use_mesh=False)
    s3, _ = source_points(p0, 2, xy)
    np.testing.assert_allclose(s2, s3, atol=1e-12)
    np.testing.assert_allclose(source_points(p1, 1, xy)[0], source_points(p0, 1, xy)[0], atol=1e-12)


def test_metal_mesh_matches_ref():
    torch = pytest.importorskip('torch')
    if not torch.backends.mps.is_available():
        pytest.skip('no MPS')
    from stillpoint.render_ref import metal_coord_map
    rng = np.random.default_rng(3)
    mesh = rng.normal(0, 2.0, (4, 4, 6, 2)).astype(np.float32)
    pl = _plan(mesh=mesh)
    for k in (0, 3):
        S, v = metal_coord_map(pl, k, out_scale=0.5)
        Sr, vr = source_map(pl, k, out_scale=0.5, return_valid=True)
        both = v & vr
        assert both.mean() > 0.9
        assert np.abs(S[both] - Sr[both]).max() < 2e-3
        S0, _ = metal_coord_map(_plan(), k, out_scale=0.5)     # the non-mesh kernel differs by the offsets
        assert np.abs(S0[both] - S[both]).max() > 0.5


# ------------------------------------------------------------------------------------------ fits / solver
def _quad_field(q, C, L):
    return poly_basis(q, 2, L) @ C


def test_fit_single_recovers_quadratic_field_despite_outliers():
    rng = np.random.default_rng(0)
    L = 480.0
    q = rng.uniform([-480, -270], [480, 270], (800, 2))
    C = rng.normal(0, 0.5, (6, 2))
    F = _quad_field(q, C, L) + rng.normal(0, 0.05, (800, 2))
    F[:60] += rng.normal(0, 8.0, (60, 2))                          # moving objects / broken tracks
    Ch = fit_single(poly_basis(q, 2, L), F, MeshParams())
    grid = np.stack(np.meshgrid(np.linspace(-480, 480, 9), np.linspace(-270, 270, 7)), -1).reshape(-1, 2)
    err = _quad_field(grid, Ch - C, L)
    assert np.abs(err).max() < 0.05


def test_fit_delta_designs():
    """Pair fields of a flow that is exactly quadratic in the image but FAST (tracks move ~15 px/frame): the 'start'
    design (the judge's convention) recovers the field change exactly, the 'common' design picks up the convective
    term. Conversely, per-track non-polynomial flows that stay constant over the two pairs cancel exactly in the
    'common' design and bias the 'start' design."""
    rng = np.random.default_rng(1)
    L = 480.0
    n = 800
    Q0 = rng.uniform([-450, -250], [450, 250], (n, 2))
    C0 = np.zeros((6, 2))
    C0[0, 1], C0[2, 1], C0[5, 1] = 6.0, 5.0, 3.0                    # forward-flight-like: 6..14 px/frame downwards
    dC = rng.normal(0, 0.2, (6, 2))
    field = lambda q, C: poly_basis(q, 2, L) @ C                      # noqa: E731
    Q1 = Q0 + field(Q0, C0)
    Q2 = Q1 + field(Q1, C0 + dC)
    grid = np.stack(np.meshgrid(np.linspace(-400, 400, 9), np.linspace(-220, 220, 7)), -1).reshape(-1, 2)
    truth = field(grid, dC)
    rms = lambda x: np.sqrt(np.mean(x ** 2))                          # noqa: E731
    errs = {}
    for design in ('start', 'mid', 'common'):
        p = MeshParams(order=2, delta_design=design)
        Pa, Pb = delta_designs(Q0, Q1, Q2, p, L)
        errs[design] = rms(field(grid, fit_delta(Pa, Pb, Q1 - Q0, Q2 - Q1, p, par_c=0.0)) - truth)
    assert errs['start'] < 0.01 * rms(truth)
    assert errs['common'] > 10 * errs['start'] and errs['mid'] < 0.5 * errs['common']
    # per-track constant non-polynomial flow ("own" parallax), slow field
    own = rng.normal(0, 3.0, (n, 2))
    C0s = rng.normal(0, 0.3, (6, 2))
    f0 = own + field(Q0, C0s)
    P1 = Q0 + f0
    f1 = own + field(P1, C0s + dC)
    P2 = P1 + f1
    errs = {}
    for design in ('start', 'common'):
        p = MeshParams(order=2, delta_design=design)
        Pa, Pb = delta_designs(Q0, P1, P2, p, L)
        errs[design] = rms(field(grid, fit_delta(Pa, Pb, f0, f1, p, par_c=0.0)) - truth)
    assert errs['common'] < 0.05 * rms(truth) and errs['start'] > 3 * errs['common']


def test_reconstruct_velocity_drift_free():
    fs = 60.0
    P = 1200
    t = np.arange(P) / fs
    v = (0.5 * np.sin(2 * np.pi * 0.3 * t) + 0.2 * np.sin(2 * np.pi * 6.0 * t))[:, None, None] * np.ones((1, 3, 2))
    rng = np.random.default_rng(2)
    vp = v + rng.normal(0, 0.05, v.shape) + 0.3 * np.sin(2 * np.pi * 7.0 * t)[:, None, None]   # pair-method junk
    vd = np.diff(v, axis=0, prepend=v[:1]) + rng.normal(0, 0.002, v.shape)
    vd[::97] = np.nan
    vr = reconstruct_velocity(vp, vd, fs, 1.0)
    sl = slice(120, -120)
    assert np.sqrt(np.mean((vr[sl] - v[sl]) ** 2)) < 0.05          # vs 0.31 for the pair estimate alone
    assert np.abs(vr[sl] - v[sl]).max() < 0.2                        # no drift


def _meas_from_field(Cpath, noise=0.0, seed=0):
    """Field-coefficient path (T, 6, 2) -> synthetic measurement dict (exact single-pair velocities and deltas)."""
    rng = np.random.default_rng(seed)
    v = np.diff(Cpath, axis=0)
    P = len(v)
    vp = v + rng.normal(0, noise, v.shape)
    vd = np.diff(v, axis=0, prepend=v[:1]) + rng.normal(0, noise * 0.1, v.shape)
    vd[0] = np.nan
    return dict(k0=np.arange(P), vp=vp, vd=vd, n_pair=np.full(P, 1000), n_delta=np.full(P, 800))


def test_solve_fields_band_limits_and_keeps_intent():
    fs, T = 60.0, 900
    t = np.arange(T) / fs
    L = 480.0
    C = np.zeros((T, 6, 2))
    C[:, 0, 1] = 0.6 * np.sin(2 * np.pi * 3.5 * t)                 # vertical bob (translation) ...
    C[:, 2, 1] = 0.9 * np.sin(2 * np.pi * 3.5 * t)                 # ... growing towards the bottom (ground)
    C[:, 0, 0] = 20.0 * t + 8.0 * np.sin(2 * np.pi * 0.3 * t)      # intentional pan: must stay
    C[:, 5, 1] = 5.0 * t                                           # steady forward-flight parallax: must stay
    C[:, 1, 0] = 0.05 * np.sin(2 * np.pi * 20.0 * t)               # above lp_hz: not corrected
    meas = _meas_from_field(C, noise=0.01)
    qv = np.stack(np.meshgrid(np.linspace(-480, 480, 9), np.linspace(-270, 270, 7)), -1).reshape(-1, 2)
    p = MeshParams(order=2)
    J, d = solve_fields(meas, T, fs, qv, L, p)
    truth = poly_basis(qv, 2, L)[:, 0:3:2] @ np.stack([C[:, 0, 1], C[:, 2, 1]])        # (nv, T) vertical
    sl = slice(60, -60)
    err = J[sl, :, 1] - truth.T[sl]
    assert np.sqrt(np.mean(err ** 2)) < 0.1 * np.sqrt(np.mean(truth ** 2))
    assert np.sqrt(np.mean(J[sl, :, 0] ** 2)) < 0.02               # pan + 20 Hz: nothing
    # support gating: no tracks -> no field; gaps split the runs (no integration across)
    J0, _ = solve_fields(dict(meas, n_delta=np.zeros(T - 1)), T, fs, qv, L, p)
    assert np.abs(J0).max() == 0.0
    keep = np.r_[0:400, 430:T - 1]
    J2, _ = solve_fields({k: (v[keep] if k != 'k0' else v[keep]) for k, v in meas.items()}, T, fs, qv, L, p)
    assert np.abs(J2[401:430]).max() == 0.0 and np.abs(J2[100:300]).max() > 0.3


def test_wiener_shrink_keeps_signal_and_drops_noise():
    """Split-half measurement: a 5 Hz bob during the first half of the clip, nothing in the second; each track half
    carries independent estimator noise. The Wiener gain keeps the bob and suppresses the pure-noise stretch."""
    fs, T = 60.0, 1200
    t = np.arange(T) / fs
    L = 480.0
    C = np.zeros((T, 3, 2))
    C[:, 0, 1] = np.where(t < 10, 0.8 * np.sin(2 * np.pi * 5.0 * t), 0.0)
    v = np.diff(C, axis=0)
    rng = np.random.default_rng(3)
    P = len(v)
    halves_p, halves_d = [], []
    for h in (0, 1):
        n = rng.normal(0, 0.05, v.shape)                               # per-pair noise of one half
        vh = v + n
        halves_p.append(vh)
        halves_d.append(np.diff(vh, axis=0, prepend=vh[:1]))
    vp = 0.5 * (halves_p[0] + halves_p[1])
    vd = 0.5 * (halves_d[0] + halves_d[1])
    meas = dict(k0=np.arange(P), vp=vp, vd=vd, vph=np.stack(halves_p, 1), vdh=np.stack(halves_d, 1),
                n_pair=np.full(P, 1000), n_delta=np.full(P, 800))
    qv = np.stack(np.meshgrid(np.linspace(-480, 480, 9), np.linspace(-270, 270, 7)), -1).reshape(-1, 2)
    p = MeshParams(order=1)
    J, d = solve_fields(meas, T, fs, qv, L, p)
    J0, _ = solve_fields(meas, T, fs, qv, L, MeshParams(order=1, wiener=False))
    first, second = slice(120, 540), slice(660, T - 60)
    truth = C[first, 0, 1]
    assert np.sqrt(np.mean((J[first, :, 1] - truth[:, None]) ** 2)) < 0.25 * np.sqrt(np.mean(truth ** 2))
    noise0 = np.sqrt(np.mean(J0[second] ** 2))
    assert noise0 > 0.02                                               # without shrinkage the noise goes through
    assert np.sqrt(np.mean(J[second] ** 2)) < 0.4 * noise0


def test_soft_clamp_and_accept():
    J = np.array([[[0.1, 0.0], [10.0, 0.0]]])
    c = soft_clamp(J, 2.0)
    assert abs(c[0, 0, 0] - 0.1) < 1e-3 and c[0, 1, 0] < 2.0
    # 'frame': one scale per frame (an affine field stays affine), small frames untouched
    Jf = np.array([[[0.1, 0.0], [10.0, 0.0]], [[0.1, 0.0], [0.2, 0.0]]])
    cf = soft_clamp(Jf, 2.0, 'frame')
    assert np.linalg.norm(cf[0], axis=-1).max() < 2.0
    assert abs(cf[0, 0, 0] / cf[0, 1, 0] - 0.01) < 1e-9
    np.testing.assert_allclose(cf[1], Jf[1], rtol=1e-2)
    acc = window_accept(np.array([1.0, 1.0, 1.0, np.nan]), np.array([0.5, 0.99, 1.2, 0.5]), MeshParams())
    assert acc.tolist() == [True, False, False, False]


def test_similarity_part():
    from stillpoint.mesh import similarity_part
    L = 100.0
    q = np.array([[50.0, 0.0], [0.0, 50.0], [30.0, -20.0]])
    th, ls, t = 0.01, 0.02, np.array([0.3, -0.2])
    C = np.zeros((3, 2))
    C[0] = t
    C[1] = (ls * L, th * L)                                         # x-column: scale, rotation
    C[2] = (0.5, 0.7)                                               # y-column: rotation+skew, scale+stretch
    S = similarity_part(C)
    d = poly_basis(q, 1, L) @ S
    ref = t + np.stack([ls * q[:, 0] - th * q[:, 1], th * q[:, 0] + ls * q[:, 1]], -1)
    np.testing.assert_allclose(d, ref, atol=1e-12)


def test_limit_border_prevents_black():
    # zoom 1.0 plan: the output border touches the source edge -> an outward mesh must be scaled down
    W, H = 640, 360
    lens = Lens('pinhole', 420.0, 420.0, (W - 1) / 2, (H - 1) / 2, np.zeros(4), W, H)
    F = 30
    mats = np.tile(np.eye(3), (F, 4, 1, 1))
    pl = Plan(src_w=W, src_h=H, out_w=W, out_h=H, lens=lens, frame_pts=np.arange(F) / 60.0,
              out_fx=np.full(F, 420.0), row_mats=mats, virt_q=np.zeros((F, 4)), meta={})
    mesh = np.zeros((F, 3, 5, 2))
    mesh[10:20, :, :, 0] = -2.0                                   # samples left of the left border
    out, d = limit_border(pl, mesh, np.arange(F), MeshParams(), 60.0)
    assert d['frames_limited'] == 10
    for k in range(F):
        pts = np.c_[np.zeros(20), np.linspace(0, H - 1, 20)]
        dx, dy = mesh_offset(out[k], pts[:, 0], pts[:, 1], W, H)
        _, ok = source_points(pl, k, pts + np.c_[dx, dy])
        assert ok.all()
    # a plan with slack keeps the mesh
    pl2 = Plan(**{**pl.__dict__, 'out_fx': np.full(F, 420.0 * 1.2)})
    out2, d2 = limit_border(pl2, mesh, np.arange(F), MeshParams(), 60.0)
    assert d2['frames_limited'] == 0 and np.allclose(out2, mesh)


# ------------------------------------------------------------------------------------------ measurement
def _texture(W, H, seed=0):
    """Smooth random texture (sigma 2.5 px): cubic resampling of it is accurate to ~0.01 px."""
    import cv2
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (H, W)).astype(np.float32), (0, 0), 2.5)
    return (img - img.mean()) / img.std() * 40.0 + 128.0


def _warp_seq(base, W, H, dy_fn, n, pad=20):
    """Frames whose content at row y is shifted DOWN by dy_fn(i, y) (quintic spline resampling)."""
    from scipy.ndimage import map_coordinates
    rows = np.arange(H, dtype=np.float64)
    mx, my = np.meshgrid(np.arange(W, dtype=np.float64) + pad, rows + pad)
    out = []
    for i in range(n):
        out.append(np.clip(np.rint(map_coordinates(base, [my - dy_fn(i, my - pad), mx], order=5)), 1, 255)
                   .astype(np.uint8))
    return np.stack(out)


def test_chunk_job_measures_field_velocity_and_delta():
    """Persistent tracks on frames whose content bounces vertically with an amplitude growing quadratically towards
    the bottom (static top): the per-pair field velocity and three-frame delta match the truth."""
    W, H = 480, 270
    base = _texture(W + 40, H + 40)
    n = 8
    dy = np.array([0.0, 0.6, -0.4, 0.3, 0.0, -0.5, 0.2, 0.1])
    prof = lambda y: (y / (H - 1)) ** 2                              # noqa: E731
    imgs = _warp_seq(base, W, H, lambda i, y: dy[i] * prof(y), n)
    p = MeshParams(order=2, cell=12, lk_win=15, lk_levels=3, min_tracks=30)
    L = W / 2.0
    out = _track_chunk_job((np.arange(n), imgs, [None] * n, p, 1, L))
    assert [r['k0'] for r in out] == list(range(1, n - 1))
    q = np.stack(np.meshgrid(np.linspace(-200, 200, 7), np.linspace(-110, 110, 5)), -1).reshape(-1, 2)
    Phi = poly_basis(q, 2, L)
    pr = prof(q[:, 1] + (H - 1) / 2)
    for r in out:
        i = r['k0']
        assert r['n_pair'] > 300
        truth = (dy[i + 1] - dy[i]) * pr
        v = Phi @ r['vp']
        assert np.abs(v[:, 1] - truth).max() < 0.04 and np.abs(v[:, 0]).max() < 0.03
        ta = ((dy[i + 1] - dy[i]) - (dy[i] - dy[i - 1])) * pr
        a = Phi @ r['vd']
        assert np.abs(a[:, 1] - ta).max() < 0.04


def test_build_mesh_end_to_end_sign_and_verify():
    """Orchestration with a fake preview renderer that applies plan.mesh exactly like the kernel (output pixel X
    samples X + D(X)): the bottom of a textured scene bounces at 5-8 Hz (amplitude growing quadratically with y), the
    top is static. build_mesh must find offsets that cancel the bounce (sign!), keep the top still, accept the windows
    in verification, and the re-rendered previews must be steadier."""
    from scipy.ndimage import map_coordinates
    from stillpoint.mesh import build_mesh
    Wo, Ho = 640, 360                                              # full-res output == source
    Wp, Hp = 320, 180                                              # previews
    s = Wp / Wo
    fs, T = 60.0, 300
    t = np.arange(T) / fs
    jit = 0.8 * np.sin(2 * np.pi * 5.0 * t) + 0.4 * np.sin(2 * np.pi * 8.0 * t + 1.0)   # preview px, at prof = 1
    rows = np.arange(Hp, dtype=np.float64)
    prof = lambda y: np.clip(y / (Hp - 1), 0, 1) ** 2               # noqa: E731
    base = _texture(Wp + 40, Hp + 40, seed=3)
    PXg, PYg = np.meshgrid(np.arange(Wp, dtype=np.float64), rows)

    def frame(k, mesh_k=None):
        sx_, sy_ = PXg.copy(), PYg.copy()
        if mesh_k is not None:
            dx, dy = mesh_offset(mesh_k, (PXg + 0.5) / s - 0.5, (PYg + 0.5) / s - 0.5, Wo, Ho)
            sx_, sy_ = sx_ + dx * s, sy_ + dy * s
        img = map_coordinates(base, [sy_ + 20 - jit[k] * prof(sy_), sx_ + 20], order=3)
        return np.clip(np.rint(img), 1, 255).astype(np.uint8)

    class Stream:
        w, h = Wp, Hp

    def render_fn(plan, stream, recs, cancel=None):
        for k in recs:
            mk = None if plan.mesh is None else plan.mesh[k]
            yield int(k), frame(int(k), mk), np.ones((Hp, Wp), bool)

    lens = Lens('pinhole', 500.0, 500.0, (Wo - 1) / 2, (Ho - 1) / 2, np.zeros(4), Wo, Ho)
    pl = Plan(src_w=Wo, src_h=Ho, out_w=Wo, out_h=Ho, lens=lens, frame_pts=t, out_fx=np.full(T, 500.0 * 1.25),
              row_mats=np.tile(np.eye(3), (T, 4, 1, 1)), virt_q=np.zeros((T, 4)), meta={})
    p = MeshParams(nx=9, order=2, clamp_px=20.0, cell=8, lk_win=11, lk_levels=3, min_tracks=30, support_lo=20,
                   support_hi=60, chunk=40)
    mesh, d = build_mesh(pl, Stream(), np.arange(T), render_fn, fs, p)
    ny, nx = mesh.shape[1:3]
    X, Y = vertex_grid(Wo, Ho, nx, ny)
    pr = prof((Y.reshape(-1) + 0.5) * s - 0.5)
    sl = slice(40, T - 40)
    from stillpoint.mesh import band_field
    truth = band_field((jit[:, None] * pr[None, :]) / s, fs, p)    # full-res px the mesh must displace by (+)
    bottom = pr > 0.5
    est = mesh.reshape(T, -1, 2)[..., 1]
    c = np.corrcoef(est[sl][:, bottom].reshape(-1), truth[sl][:, bottom].reshape(-1))[0, 1]
    assert c > 0.95, c
    assert np.sqrt(np.mean((est[sl][:, bottom] - truth[sl][:, bottom]) ** 2)) < \
        0.25 * np.sqrt(np.mean(truth[sl][:, bottom] ** 2))
    top = pr < 0.02
    assert np.sqrt(np.mean(mesh.reshape(T, -1, 2)[sl][:, top] ** 2)) < 0.15
    assert d['verify']['accepted'] >= 3
    assert d['verify']['hf_after_kept'] < 0.4 * d['verify']['hf_before']


# ------------------------------------------------------------------------------------------ renderer
SPRENDER = os.path.join(ROOT, 'app', 'renderer', '.build', 'sprender')


def test_sprender_mesh_golden(tmp_path):
    """sprender --dump-frames coordinate map of a mesh plan == render_ref (float64) within 0.02 px on real O3
    footage; --no-mesh reproduces the rotation-only map."""
    from eval.footage import O3_DIR
    v26 = os.path.join(O3_DIR, 'DJI_0026.MP4')
    if not os.path.exists(v26) or not os.path.exists(SPRENDER):
        pytest.skip('footage or sprender not available')
    from test_render_golden import demux_pts, make_plan
    pts = demux_pts(v26)[:6]
    pl = make_plan(pts, rate_dps=60.0, zoom=1.2)
    nx, ny = 17, 10
    rng = np.random.default_rng(5)
    pl.mesh = (rng.normal(0, 3.0, (len(pts), ny, nx, 2))).astype(np.float32)
    plan_path = tmp_path / 'm.spplan'
    write_plan(str(plan_path), pl)
    for extra, use in (([], True), (['--no-mesh'], False)):
        dd = tmp_path / ('d1' if use else 'd0')
        r = subprocess.run([SPRENDER, v26, str(plan_path), str(tmp_path / 'o.mov'), '--frames', '4', '--no-write',
                            '--dump-frames', '2', '--dump-dir', str(dd), '--quiet', *extra],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        c = np.fromfile(dd / 'frame2_coords.f32', np.float32).reshape(pl.out_h, pl.out_w, 3)
        ys = np.arange(0, pl.out_h, 37)
        xs = np.arange(0, pl.out_w, 41)
        XX, YY = np.meshgrid(xs, ys)
        S, ok = source_points(pl, 2, np.stack([XX.reshape(-1), YY.reshape(-1)], -1), use_mesh=use)
        Sm = c[YY, XX, :2].reshape(-1, 2)
        vm = c[YY, XX, 2].reshape(-1) > 0.5
        both = ok & vm
        assert both.mean() > 0.8
        assert np.abs(Sm[both] - S[both]).mean() < 0.02
        assert np.abs(Sm[both] - S[both]).max() < 0.1
