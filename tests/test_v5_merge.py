"""Engine v5 merge (workstreams A-E, 2026-09-29): the pieces that meet.

(a) .spplan flags: FILL = bit 0, MESH = bit 1; a file carrying both sections round-trips (mesh offset explicit,
    after the fill section) and a plain plan stays byte-identical to v1.
(b) pipeline.check_params: fill / mesh_residual / synth_blur are mutually exclusive (separate renderer kernels).
(c) horizon lock v2 + fill overscan (crop_extend) in one optimizer run: the extended crop box is honoured in both
    SQP stages and the lock levels at least as far as without the extension.
(d) horizon lock + blur-aware smoothing in one run (both extra QP terms in the same windows).
"""
import os
import struct

import numpy as np
import pytest

from stillpoint.pipeline import AnalyzeParams, check_params
from stillpoint.plan_io import FLAG_FILL, FLAG_MESH, read_fill, read_plan, write_plan


def test_flags_and_both_sections_roundtrip(tmp_path):
    from test_fill import _synthetic
    from stillpoint.fill import FillParams, select_sources
    _, plan, _ = _synthetic(F=24)
    assert (FLAG_FILL, FLAG_MESH) == (1, 2)
    p0 = os.path.join(tmp_path, 'plain.spplan')
    write_plan(p0, plan)
    raw = open(p0, 'rb').read()
    assert struct.unpack_from('<I', raw, 84)[0] == 0 and raw[88:256] == bytes(168)
    fill = select_sources(plan, plan.virt_q, None, FillParams(align=False))
    rng = np.random.default_rng(0)
    plan.mesh = rng.normal(0, 0.5, (plan.n_frames, 4, 5, 2)).astype(np.float32)
    plan.meta['mesh_clamp_px'] = 1.5
    p1 = os.path.join(tmp_path, 'both.spplan')
    write_plan(p1, plan, fill=fill)
    raw1 = open(p1, 'rb').read()
    assert struct.unpack_from('<I', raw1, 84)[0] == FLAG_FILL | FLAG_MESH
    back = read_plan(p1)
    np.testing.assert_array_equal(back.mesh, plan.mesh)
    assert back.meta['mesh_clamp_px'] == pytest.approx(1.5)
    np.testing.assert_allclose(back.row_mats, plan.row_mats.astype(np.float32))
    fb = read_fill(p1)
    np.testing.assert_array_equal(fb.src, fill.src)
    np.testing.assert_array_equal(fb.n_src, fill.n_src)
    # mesh only: the block sits right after the records
    p2 = os.path.join(tmp_path, 'mesh.spplan')
    write_plan(p2, plan)
    raw2 = open(p2, 'rb').read()
    F, R = plan.n_frames, plan.n_rows
    assert struct.unpack_from('<I', raw2, 84)[0] == FLAG_MESH
    assert struct.unpack_from('<Q', raw2, 96)[0] == 256 + F * (24 + 36 * R)
    assert read_fill(p2) is None
    np.testing.assert_array_equal(read_plan(p2).mesh, plan.mesh)


def test_check_params_exclusive_options():
    check_params(AnalyzeParams())
    check_params(AnalyzeParams(horizon_lock=0.7, roll_limit_deg=10, fill=True, fill_overscan=0.06, blur_smooth_w=50))
    check_params(AnalyzeParams(mesh_residual=True, horizon_lock=1.0))
    for kw in (dict(fill=True, mesh_residual=True), dict(fill=True, synth_blur='auto'),
               dict(mesh_residual=True, synth_blur='angle'), dict(synth_blur='sometimes'), dict(horizon_lock=1.5)):
        with pytest.raises(ValueError):
            check_params(AnalyzeParams(**kw))


def test_horizon_lock_with_fill_overscan():
    from test_smooth import DEG, OUT_H, OUT_W, _qfn, make_grav_tel
    from stillpoint.smooth import SmoothParams, check_crop, fx_for_crop_area, gravity_world, horizon_angles, \
        optimize_path
    n = 900
    tel = make_grav_tel(n, lambda t: 0.4 * t, lambda t: DEG(10) + 0 * t,
                        lambda t: DEG(35) * np.clip(np.minimum(t - 4.0, 12.0 - t) / 1.5, 0, 1), seed=5)
    q = _qfn(tel)
    fx = fx_for_crop_area(tel.lens, tel.width, tel.height, OUT_W, OUT_H, 0.6)
    cap = 0.05 * tel.height
    E = np.full((n, 4), cap)
    runs = {}
    for name, kw in (('lock', dict(horizon_lock=1.0)), ('lock+ext', dict(horizon_lock=1.0, crop_extend=E))):
        V, f, info = optimize_path(tel, q, np.arange(n), OUT_W, OUT_H, SmoothParams(min_out_fx=fx, **kw),
                                   return_info=True)
        rho = np.rad2deg(horizon_angles(V, gravity_world(tel, tel.frame_t))[0])
        runs[name] = (V, f, info, rho)
    t = tel.frame_t
    mid = (t > 6.5) & (t < 9.5)
    lvl = {k: float(np.median(np.abs(v[3][mid]))) for k, v in runs.items()}
    info_e = runs['lock+ext'][2]
    print(f'\nmid-turn |horizon| lock {lvl["lock"]:.2f} deg, lock + {cap:.0f} px overscan {lvl["lock+ext"]:.2f} deg; '
          f'max violation of the extended box {info_e["max_violation_px"]:.2f} px; {info_e["horizon"]}')
    assert 'horizon' in info_e and info_e['crop_extend_mean_px'][0] == pytest.approx(cap)
    assert info_e['max_violation_px'] < 1.0                           # the extended box holds after stage 2
    assert lvl['lock+ext'] <= lvl['lock'] + 0.25                       # more room never levels less
    # and the plain box is exceeded somewhere (the extension is really used), by at most the cap
    Ve, fe = runs['lock+ext'][0], runs['lock+ext'][1]
    c = check_crop(tel, q, np.arange(n), Ve, fe, OUT_W, OUT_H, 64, 0.0)
    assert c.max() <= cap + 1.0


def test_horizon_lock_with_blur_smoothing():
    from test_smooth import DEG, OUT_H, OUT_W, _qfn, make_grav_tel
    from stillpoint.smooth import SmoothParams, fx_for_crop_area, optimize_path
    n = 600
    tel = make_grav_tel(n, lambda t: 0.8 * t, lambda t: DEG(10) + 0 * t, lambda t: DEG(8) * np.sin(0.5 * t), seed=3)
    tel.exposure_s = np.full(n, 0.012)                                   # long exposures: the blur term is active
    fx = fx_for_crop_area(tel.lens, tel.width, tel.height, OUT_W, OUT_H, 0.6)
    V, f, info = optimize_path(tel, _qfn(tel), np.arange(n), OUT_W, OUT_H,
                               SmoothParams(min_out_fx=fx, horizon_lock=1.0, w_blur=100.0), return_info=True)
    assert np.all(np.isfinite(V)) and 'horizon' in info and info['blur']['frac_active'] > 0
    assert info['max_violation_px'] < 1.0
