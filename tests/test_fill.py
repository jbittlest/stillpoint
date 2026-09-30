"""Full-frame border fill (engine/stillpoint/fill.py, warp.metal *_fill kernels, sprender's frame ring).

(a) .spplan FILL section round trip; v1 readers still read the plan.
(b) geometry end to end on a synthetic spherical world (no footage): source frames are rendered from a panorama
    through the KB4 lens with rolling shutter; a wide output (out_fx = 0.85 x lens fx) of a low-passed path leaves
    the source; select_sources + the Metal preview kernel fill it from neighbours and must reproduce the IDEAL wide
    view (the panorama seen by the virtual camera) to interpolation accuracy, where the plain render is black.
(c) the Metal buffer kernel == the float64 reference (render_planes_ref_fill) incl. parallax mesh, gains, the
    per-pixel consistency check and the soft edge extension.
(d) golden (footage): sprender's fill output on DJI_0026 == render_planes_ref_fill (PSNR > 45 dB); pixels deep
    inside the source are bit-identical to the plain kernel; --no-fill == a plan without the section.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from stillpoint.fill import (FillParams, FillTable, fill_stats, inside_distance, mesh_disp, select_sources)
from stillpoint.geom import Lens, qexp, qlog, qmul, quat_to_mat
from stillpoint.plan_build import build_plan, camera_orientation_fn
from stillpoint.plan_io import read_fill, read_plan, write_plan
from stillpoint.render_ref import _source_coord, preview_fill, render_planes_ref_fill
from stillpoint.types import Telemetry, TimeModel

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))


# ------------------------------------------------------------------------------------------ synthetic world
def _tex(w):
    """Smooth panorama texture of a world direction (…,3) -> 8-bit-ish levels."""
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    lon = np.arctan2(w[..., 0], w[..., 2])
    lat = np.arcsin(np.clip(w[..., 1], -1, 1))
    return 128 + 55 * np.sin(7 * lon) * np.cos(5 * lat) + 35 * np.sin(19 * lon + 9 * lat) + 20 * np.cos(13 * lat)


def _synthetic(W=640, H=360, F=48, zoom=0.85, seed=0):
    lens = Lens('kb4', 235.0, 235.0, (W - 1) / 2, (H - 1) / 2, np.array([0.2499, 0.0136, -0.0621, 0.0122]), W, H)
    fps = 60000 / 1001
    pts = np.arange(F) / fps
    t = np.arange(-1.0, pts[-1] + 1.0, 1 / 2000.0)
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, 6)
    rv = np.stack([0.05 * np.sin(2 * np.pi * 2.3 * t + ph[0]) + 0.02 * np.sin(2 * np.pi * 5.1 * t + ph[1]),
                   0.3 * t + 0.06 * np.sin(2 * np.pi * 1.7 * t + ph[2]) + 0.02 * np.sin(2 * np.pi * 4.3 * t + ph[3]),
                   0.03 * np.sin(2 * np.pi * 3.1 * t + ph[4])], -1)
    tel = Telemetry(source='synthetic', camera='synthetic', width=W, height=H, fps=fps, frame_pts=pts,
                    frame_t=pts.copy(), exposure_s=np.full(F, 1 / 1000), readout_s=0.008, lens=lens, imu_t=t,
                    imu_q=qexp(rv), imu_rate=2000.0, has_highrate=True, eis_baked=False)
    tm = TimeModel()
    qf = camera_orientation_fn(tel, tm)
    virt = qexp(uniform_filter1d(qlog(qf(tel.frame_t)), 25, axis=0, mode='nearest'))
    plan = build_plan(tel, tm, qf, virt, np.full(F, lens.fx * zoom), W, H, n_rows=16, exposure_avg=False)
    # source frames: panorama through the lens, each source row at its own (rolling-shutter) time
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    rc = lens.unproject(np.stack([uu, vv], -1))
    frames = []
    for k in range(F):
        trow = pts[k] + tel.readout_s * ((np.arange(H) + 0.5) / H - 0.5)
        R = quat_to_mat(qf(trow))                                   # (H,3,3)
        wv = np.einsum('hij,hwj->hwi', R, rc)
        frames.append(_tex(wv).astype(np.float32))
    return tel, plan, frames


def _ideal(plan, k, out_scale=1.0):
    Wo, Ho = int(round(plan.out_w * out_scale)), int(round(plan.out_h * out_scale))
    X, Y = np.meshgrid((np.arange(Wo) + 0.5) / out_scale - 0.5, (np.arange(Ho) + 0.5) / out_scale - 0.5)
    fx = plan.out_fx[k]
    rv = np.stack([(X - (plan.out_w - 1) / 2) / fx, (Y - (plan.out_h - 1) / 2) / fx, np.ones_like(X)], -1)
    return _tex(np.einsum('ij,hwj->hwi', quat_to_mat(plan.virt_q[k]), rv))


@pytest.fixture(scope='module')
def synth():
    tel, plan, frames = _synthetic()
    tab = select_sources(plan, plan.virt_q, None, FillParams(align=False))
    return dict(tel=tel, plan=plan, frames=frames, tab=tab)


# ------------------------------------------------------------------------------------------ (a) file format
def test_fill_section_roundtrip(tmp_path, synth):
    plan, tab = synth['plan'], FillTable(**{**synth['tab'].__dict__})
    tab.mesh = np.random.default_rng(1).normal(0, 0.3, (plan.n_frames, 5, 8, 2)).astype(np.float32)
    tab.par_px = 7.5
    p1, p2 = str(tmp_path / 'a.spplan'), str(tmp_path / 'b.spplan')
    write_plan(p1, plan, fill=tab)
    write_plan(p2, plan)
    assert read_fill(p2) is None
    t2 = read_fill(p1)
    for f in ('n_src', 'src', 'weight', 'gain', 'G', 'frac_fill', 'frac_uncovered', 'mesh'):
        assert np.array_equal(getattr(t2, f), getattr(tab, f).astype(getattr(t2, f).dtype)), f
    assert abs(t2.feather_main_px - tab.feather_main_px) < 1e-4 and abs(t2.sigma - tab.sigma) < 1e-7
    assert tab.par_px > 0 and abs(t2.par_px - tab.par_px) < 1e-4
    a, b = read_plan(p1), read_plan(p2)                  # the plan itself is unchanged by the section
    assert np.array_equal(a.row_mats, b.row_mats) and np.array_equal(a.frame_pts, b.frame_pts)
    with open(p1, 'rb') as f1, open(p2, 'rb') as f2:
        x, y = f1.read(), f2.read()
    assert x[:84] == y[:84] and x[88:len(y)] == y[88:]   # only the flags word differs, then the appended section
    tab.mesh = None


# ------------------------------------------------------------------------------------------ (b) geometry
def test_select_sources_covers_out_of_source(synth):
    plan, tab = synth['plan'], synth['tab']
    st = fill_stats(tab)
    print('\n', st)
    assert st['frac_frames_filled'] > 0.9 and st['fill_frac_mean'] > 0.03      # the wide output does leave the source
    assert st['uncovered_frac_mean'] < 0.1 * st['fill_frac_mean']               # neighbours cover >90 % of it
    k = 24
    n = tab.n_src[k]
    assert 1 <= n <= 4 and np.all(np.abs(tab.src[k, :n] - k) <= 20) and np.all(tab.src[k, :n] != k)
    # G = Rv_j^T Rv_k exactly
    for i in range(n):
        j = tab.src[k, i]
        G = quat_to_mat(plan.virt_q[j]).T @ quat_to_mat(plan.virt_q[k])
        assert np.allclose(tab.G[k, i], G, atol=1e-6)


def test_select_batched_equals_per_record(synth):
    """The batched geometry (32 records per vectorised projection) == selecting each record on its own (hysteresis
    off: it is the only coupling between consecutive records)."""
    plan = synth['plan']
    prm = FillParams(align=False, hysteresis=1.0)
    a = select_sources(plan, plan.virt_q, None, prm)
    for k in (0, 5, 24, 31, 32, 33, 47):
        b = select_sources(plan, plan.virt_q, [k], prm)
        n = int(a.n_src[k])
        assert b.n_src[k] == n and np.array_equal(a.src[k], b.src[k]), k
        assert np.allclose(a.weight[k], b.weight[k]) and np.allclose(a.G[k], b.G[k], atol=1e-7)
        assert abs(a.frac_fill[k] - b.frac_fill[k]) < 1e-6 and abs(a.frac_uncovered[k] - b.frac_uncovered[k]) < 1e-6


def test_fill_reproduces_ideal_wide_view(synth):
    """The filled preview (Metal buffer kernel) vs the panorama seen by the virtual camera."""
    plan, tab, frames = synth['plan'], synth['tab'], synth['frames']
    errs_fill, errs_main, cov = [], [], []
    for k in (8, 24, 40):
        n = int(tab.n_src[k])
        bufs = [frames[k]] + [frames[int(tab.src[k, i])] for i in range(n)]
        img, d0, wn = preview_fill(plan, tab, k, bufs, 1.0, kernel='catmullrom', black=0.0)
        ideal = _ideal(plan, k)
        out = d0 < 0
        covered = out & (wn > 0.1)
        deep = d0 > 4
        errs_fill.append(np.abs(img - ideal)[covered])
        errs_main.append(np.abs(img - ideal)[deep])
        cov.append(covered.sum() / max(out.sum(), 1))
        print(f'\nframe {k}: {n} sources, outside {out.mean():.3f}, covered {cov[-1]:.3f}; |err| fill mean '
              f'{errs_fill[-1].mean():.3f} p99 {np.percentile(errs_fill[-1], 99):.2f}; main mean {errs_main[-1].mean():.3f}')
    ef, em = np.concatenate(errs_fill), np.concatenate(errs_main)
    assert np.mean(cov) > 0.85
    assert ef.mean() < 1.5 * em.mean() + 0.1 and np.percentile(ef, 99) < 6.0      # as good as the frame's own pixels


def test_fill_wrong_geometry_is_detected(synth):
    """Control: the same fill with G = identity (no relative rotation) is far off -> the test above is sensitive."""
    plan, tab, frames = synth['plan'], synth['tab'], synth['frames']
    k = 24
    bad = FillTable(**{**tab.__dict__, 'G': np.tile(np.eye(3, dtype=np.float32), (plan.n_frames, 4, 1, 1))})
    n = int(tab.n_src[k])
    bufs = [frames[k]] + [frames[int(tab.src[k, i])] for i in range(n)]
    img, d0, wn = preview_fill(plan, bad, k, bufs, 1.0, kernel='catmullrom', black=0.0)
    m = (d0 < 0) & (wn > 0.1)
    assert np.abs(img - _ideal(plan, k))[m].mean() > 5.0


# ------------------------------------------------------------------------------------------ (c) kernel == reference
def test_preview_kernel_matches_float64_reference(synth):
    plan, frames = synth['plan'], synth['frames']
    tab = FillTable(**{**synth['tab'].__dict__})
    rng = np.random.default_rng(3)
    tab.mesh = (rng.normal(0, 0.8, (plan.n_frames, 5, 8, 2))).astype(np.float32)   # exercised, not physical
    tab.gain = np.clip(rng.normal(1.0, 0.05, tab.gain.shape), 0.9, 1.1).astype(np.float32)
    tab.sigma = 0.03
    tab.par_px = 0.02 * plan.out_w                         # the parallax-aware weight exercised too
    for k in (12, 30):
        n = int(tab.n_src[k])
        srcs = [k] + [int(tab.src[k, i]) for i in range(n)]
        y8 = {j: np.clip(np.round(frames[j]), 0, 255).astype(np.uint8) for j in srcs}
        planes = {j: (y8[j], np.full((plan.src_h // 2, plan.src_w // 2, 2), 128, np.uint8)) for j in srcs}
        RY, _ = render_planes_ref_fill(plan, tab, k, planes, kernel='catmullrom', full_range=False)
        img, d0, wn = preview_fill(plan, tab, k, [y8[j].astype(np.float32) for j in srcs], 1.0, kernel='catmullrom',
                                   black=16.0)
        ref_lv = RY.astype(np.float64) / 64.0 / 4.0            # 10-bit code -> 8-bit level
        valid = d0 > -1e5
        d = np.abs(img - ref_lv)[valid]
        print(f'\nframe {k}: kernel vs float64: mean {d.mean():.3f} p99.9 {np.percentile(d, 99.9):.3f} max {d.max():.3f} levels')
        assert d.mean() < 0.15 and np.percentile(d, 99.9) < 0.6


def test_parallax_weight_falls_back_where_shift_is_large(synth):
    """sp_par_w: with a large predicted parallax shift (|offset| |v| >> par_px) the neighbours lose their weight
    (the soft edge extension takes over); with par_px = 0 they keep it."""
    plan, frames = synth['plan'], synth['frames']
    k = 24
    n = int(synth['tab'].n_src[k])
    bufs = [frames[k]] + [frames[int(synth['tab'].src[k, i])] for i in range(n)]
    tab = FillTable(**{**synth['tab'].__dict__})
    tab.mesh = np.zeros((plan.n_frames, 5, 8, 2), np.float32)
    tab.mesh[..., 0] = 0.05 * plan.out_w                  # 5 % of the width per frame
    tab.par_px = 0.01 * plan.out_w
    _, d0, wn_on = preview_fill(plan, tab, k, bufs, 0.5, kernel='catmullrom', black=0.0)
    tab.par_px = 0.0
    _, _, wn_off = preview_fill(plan, tab, k, bufs, 0.5, kernel='catmullrom', black=0.0)
    out = d0 < 0
    assert out.sum() > 100
    assert np.mean(wn_off[out] > 0.05) > 0.3 and np.mean(wn_on[out] > 0.05) < 0.02


def test_coverage_overscan_and_path_uses_it():
    """(e) coverage_overscan on shaky synthetic O3 telemetry: extensions exist where neighbours looked, never exceed
    the cap, are 0 at the fisheye corners; the path optimizer with that per-edge profile is smoother at the same
    (tight) output focal and never leaves the extended box."""
    from test_smooth import OUT_H, OUT_W, _hp, _qfn, make_tel
    from stillpoint.fill import coverage_overscan
    from stillpoint.smooth import SmoothParams, fx_for_crop_area, optimize_path
    n = 900
    tel = make_tel(n, jit_rms_deg=0.8, seed=5)
    q = _qfn(tel)
    cap = 0.05 * tel.height
    E, info = coverage_overscan(q, tel.frame_t, tel.lens, tel.width, tel.height, cap)
    print('\n', info)
    assert E.shape == (n, 4, 17) and E.max() <= cap and np.all(E >= 0)
    mid = E[:, :, 8]
    assert (mid > 0).mean() > 0.5                          # the edge centres are usually covered by neighbours
    assert np.all(E[:, 1, 0] <= mid[:, 1] + 1e-9) or (E[:, 1, 0] > 0).mean() < 0.5
    fx = fx_for_crop_area(tel.lens, tel.width, tel.height, OUT_W, OUT_H, 0.72)
    V0, f0, i0 = optimize_path(tel, q, np.arange(n), OUT_W, OUT_H, SmoothParams(min_out_fx=fx, allow_zoom=False),
                               return_info=True)
    V1, f1, i1 = optimize_path(tel, q, np.arange(n), OUT_W, OUT_H,
                               SmoothParams(min_out_fx=fx, allow_zoom=False, crop_extend=E), return_info=True)
    sl = slice(60, n - 60)
    hf = lambda V: float(np.sqrt((np.rad2deg(_hp(np.unwrap(qlog(V), axis=0)))[sl] ** 2).mean()))
    print(f"binding {i0['frac_binding']:.3f} -> {i1['frac_binding']:.3f}; path HF {hf(V0):.4f} -> {hf(V1):.4f} deg; "
          f"max violation of the extended box {i1['max_violation_px']:.2f} px; extension mean {i1['crop_extend_mean_px']}")
    assert i0['frac_binding'] > 0.05                       # the crop binds at this focal ...
    assert hf(V1) < 0.5 * hf(V0)                           # ... and the fill-backed overscan lets the path smooth more
    assert i1['max_violation_px'] < 1.0


class _FakeStream:
    """FrameStream stand-in over in-memory full-res gray frames (area-downscaled to width w)."""

    def __init__(self, frames, w):
        import cv2
        H, W = frames[0].shape
        self.w, self.h = w, int(round(H * w / W))
        self._f = [cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_AREA) for f in frames]

    def frames(self, recs, cancel=None):
        for r in recs:
            yield int(r), np.clip(np.round(self._f[int(r)]), 0, 255).astype(np.uint8)

    def close(self):
        pass


def test_align_pass_on_rotation_only_world(synth):
    """align_sources on the synthetic (rotation-only, static) world: no parallax -> mesh ~ 0, the band check keeps
    every source (high NCC), gains ~ 1; a source whose content is replaced by another frame's is dropped."""
    from stillpoint.fill import align_sources
    plan, frames = synth['plan'], synth['frames']
    tab = synth['tab']
    st = _FakeStream(frames, 320)
    t2 = align_sources(tab, plan, None, None, FillParams(), stream=st)
    a = t2.meta['align']
    print('\n', a)
    w = tab.n_src > 0
    assert a['n_checked'] > 10 and a['n_dropped'] == 0 and a['ncc_median'] > 0.9
    assert np.all(np.abs(t2.gain[w][:, 0] - 1) < 0.05)
    assert np.abs(t2.mesh[w]).max() < 0.05 * plan.out_w / 100            # < 0.05 % of the width per frame
    assert np.array_equal(t2.n_src, tab.n_src)
    # corrupt the frames a source of record 24 points at -> that source must lose its weight
    k = 24
    j = int(tab.src[k, 0])
    bad = list(frames)
    bad[j] = np.roll(frames[j], 40, axis=1)                  # 40 px shift = wrong content in the band
    t3 = align_sources(tab, plan, None, [k], FillParams(), stream=_FakeStream(bad, 320))
    assert j not in set(t3.src[k, :t3.n_src[k]].tolist()) or t3.weight[k, list(t3.src[k]).index(j)] < 0.5 * tab.weight[k, 0]
    # check_stride: every record checked vs every 2nd (+ new sources): same outcome, fewer checks, the rest reused
    t1 = align_sources(tab, plan, None, None, FillParams(check_stride=1), stream=st)
    assert t1.meta['align']['n_reused'] == 0 and a['n_reused'] > 0
    assert a['n_checked'] < t1.meta['align']['n_checked']
    assert np.array_equal(t1.n_src, t2.n_src) and np.allclose(t1.gain, t2.gain, atol=0.03)
    # a wider stream is downscaled to align_width (the analysis passes its own 960-px decoder)
    t4 = align_sources(tab, plan, None, [20, 21, 22], FillParams(align_width=320), stream=_FakeStream(frames, 640))
    assert t4.meta['align_width'] == 320


def test_mesh_disp_bilinear():
    mesh = np.zeros((3, 4, 2), np.float32)
    mesh[..., 0] = np.arange(4)[None, :]
    mesh[..., 1] = np.arange(3)[:, None] * 10
    W, H = 400, 300
    d = mesh_disp(mesh, np.array([-5.0, 49.5, 99.5, 399.0]), np.array([0.0, 49.5, 149.5, 299.0]), W, H)
    assert np.allclose(d[:, 0], [0, 0, 0.5, 3]) and np.allclose(d[:, 1], [0, 0, 10, 20])


# ------------------------------------------------------------------------------------------ (d) golden (footage)
try:
    from test_render_golden import V26, decode_frames, demux_pts, ensure_sprender, load_dump, make_plan, psnr10
    HAVE26 = os.path.exists(V26)
except Exception:  # pragma: no cover
    HAVE26 = False


@pytest.mark.skipif(not HAVE26, reason='O3 footage not available')
def test_golden_sprender_fill(tmp_path):
    import re
    pts = demux_pts(V26)
    plan = make_plan(pts, source=V26, zoom=0.8)
    tab = select_sources(plan, plan.virt_q, np.arange(90, 150), FillParams(align=False))
    tab.mesh = np.zeros((plan.n_frames, 5, 8, 2), np.float32)
    tab.mesh[..., 0] = 0.4
    tab.gain[:] = 1.02
    p = str(tmp_path / 'fill.spplan')
    write_plan(p, plan, fill=tab)
    pr = read_plan(p)
    tab = read_fill(p)
    spr = ensure_sprender()
    frames = (100, 128)

    def run(*extra, d='dump'):
        r = subprocess.run([spr, V26, p, str(tmp_path / 'o.mov'), '--start-frame', '96', '--frames', '36',
                            '--dump-frames', ','.join(map(str, frames)), '--dump-dir', str(tmp_path / d), '--no-write',
                            '--quiet', *extra], capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr + r.stdout
        return r.stdout

    out = run()
    m = re.search(r'FILL frames=(\d+) sources=(\d+) missing_sources=(\d+)', out)
    assert m and int(m.group(1)) == int((tab.n_src[96:132] > 0).sum()) and int(m.group(3)) == 0
    run('--no-fill', d='plain')
    for n in frames:
        D = load_dump(str(tmp_path / 'dump'), n)
        Pn = load_dump(str(tmp_path / 'plain'), n)
        srcs = [int(tab.src[n, i]) for i in range(tab.n_src[n])]
        dec = decode_frames(V26, [pts[j] for j in [n] + srcs])
        planes = {j: dec[pts[j]] for j in [n] + srcs}
        RY, RUV = render_planes_ref_fill(pr, tab, n, planes)
        py, puv = psnr10(D['oy'], RY), psnr10(D['ouv'], RUV)
        S, _, z = _source_coord(pr, n, *np.meshgrid(np.arange(pr.out_w, dtype=float), np.arange(pr.out_h, dtype=float)))
        deep = inside_distance(pr, S, z) >= tab.feather_main_px + 1
        same = np.mean(D['oy'][deep] == Pn['oy'][deep])
        print(f'\nframe {n}: {len(srcs)} sources, fill frac {tab.frac_fill[n]:.3f}; PSNR Y {py:.1f} CbCr {puv:.1f}; '
              f'deep-interior identical to plain {same:.5f}')
        assert py > 45 and puv > 45 and same == 1.0
        assert np.mean(Pn['oy'][~deep & (inside_distance(pr, S, z) < 0)] == 64 * 64) == 1.0   # plain: black outside


@pytest.mark.skipif(not HAVE26, reason='O3 footage not available')
def test_analyze_with_fill_short_clip(tmp_path):
    """analyze(fill=True, fill_overscan) on the 268-frame DJI_0026: the plan carries a FILL section (align pass ran),
    the report has the fill stats, and sprender renders it."""
    import json
    from dataclasses import replace
    from stillpoint.fill import output_footprint
    from stillpoint.pipeline import AnalyzeParams, analyze, render
    out = tmp_path / 'a'
    prm = AnalyzeParams(calibrate=False, closed_loop_iters=0, measure_open_loop=False, target_footprint=0.8,
                        verbose=False, quality=False, fill=True, fill_overscan=0.03, fill_overscan_mode='uniform')
    rep = analyze(V26, str(out), prm)
    r = json.load(open(out / 'report.json'))
    f = r['fill']
    print('\n', {k: f[k] for k in f if k != 'align'}, f.get('align'), r['crop'].get('footprint_mean'))
    assert 'error' not in f and f['records'] == rep['n_frames']
    # with fill + overscan the crop search targets the DISPLAYED field of view (unclipped footprint)
    assert r['crop']['footprint_method'].startswith('fill.output_footprint')
    assert abs(f['output_footprint_mean'] - 0.8) < 0.01
    if f['frac_frames_filled'] > 0:                        # the align pass ran on the analysis' own decoder
        assert f['align_s'] is not None and f['align']['n_work'] > 0
    tab = read_fill(str(out / 'plan.spplan'))
    plan = read_plan(str(out / 'plan.spplan'))
    assert tab is not None and tab.mesh is not None and tab.n_frames == plan.n_frames
    assert f['uncovered_frac_mean'] <= f['fill_frac_mean']
    assert rep['smooth']['max_violation_px'] < 1.0          # the overscan is part of the (relaxed) constraint
    a = np.load(out / 'analysis.npz')
    fp = output_footprint(replace(plan, virt_q=a['virt_q']), np.arange(0, plan.n_frames, 20))
    assert abs(fp.mean() - f['output_footprint_mean']) < 0.01
    mov = str(tmp_path / 'r.mov')
    render(V26, str(out / 'plan.spplan'), mov, start_frame=100, n_frames=30, codec='hevc10-speed')
    assert os.path.exists(mov)
