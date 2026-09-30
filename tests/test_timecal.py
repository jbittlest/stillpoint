"""Per-clip timing self-calibration (timecal.py).

Synthetic: frames rendered (tests/test_calib.Renderer: textured world at infinity or a sphere for parallax, KB4
fisheye, rolling shutter) with a KNOWN timing error relative to the telemetry handed to calibrate_timing; the fit
must recover it and must leave correct metadata alone. Real-footage validation (O3 / OA4 / O4 Pro clips, +-1 ms
injections) is in the workstream report (too slow for a unit test).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from stillpoint.geom import qexp
from stillpoint.plan_build import exposure_averaged_q
from stillpoint.timecal import (TimecalParams, _BoxOrient, _decide, _decide_nuisance, apply_exposure_slope,
                                build_deltas, calibrate_timing, choose_windows)
from stillpoint.types import TimeModel
from test_calib import Renderer, make_synthetic


class FastRenderer(Renderer):
    """test_calib.Renderer with the per-row rotation applied by broadcasting (einsum was ~1 s per frame) at 640 px;
    the world at infinity or inside a sphere (velocity) exactly as there."""

    def __init__(self, tel, **kw):
        kw.setdefault('width', 640)
        super().__init__(tel, **kw)
        self._b = [np.ascontiguousarray(self.beta[..., j]) for j in range(3)]

    def frame(self, k: int) -> np.ndarray:
        import cv2
        from stillpoint.geom import qmul, quat_to_mat
        rows = (np.arange(self.H) + 0.5) / self.H - 0.5
        tau = self.tel.frame_t[k] + self.offset + self.readout * rows
        R = quat_to_mat(qmul(self.orient(tau), np.broadcast_to(self.qE, (self.H, 4))))
        bx, by, bz = self._b
        w = [R[:, i, 0, None] * bx + R[:, i, 1, None] * by + R[:, i, 2, None] * bz for i in range(3)]
        if self.velocity is not None:
            c = (tau[:, None] - 3.0) * self.velocity[None, :]                    # (H,3) camera centre per row
            cw = c[:, 0, None] * w[0] + c[:, 1, None] * w[1] + c[:, 2, None] * w[2]
            lam = -cw + np.sqrt(cw ** 2 - np.sum(c * c, axis=1)[:, None] + self.radius ** 2)
            w = [c[:, i, None] + lam * w[i] for i in range(3)]
            n = np.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
            w = [v / n for v in w]
        lon = np.arctan2(w[0], w[2])
        lat = np.arcsin(np.clip(w[1], -1, 1))
        th, tw = self.tex.shape
        u = (lon / (2 * np.pi) + 0.5) * tw - 0.5
        v = (lat / np.pi + 0.5) * th - 0.5
        img = cv2.remap(self.tex, u.astype(np.float32), v.astype(np.float32), cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT)
        if self.noise:
            img = img + self.rng.normal(0, self.noise, img.shape)
        return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _qdist(a, b):
    return np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))


def test_box_orient_matches_tap_average_and_slerp():
    tel = make_synthetic(1.0)
    bo = _BoxOrient(tel.imu_t, tel.imu_q)
    t = np.linspace(0.1, 0.9, 301)
    assert _qdist(bo(t), tel.orientation_at(t)).max() < 1e-6            # w = 0: nlerp == slerp at 2 kHz
    w = np.full(len(t), 0.012)
    ref = exposure_averaged_q(tel.orientation_at, t[:, None], w, taps=401)[:, 0]
    assert _qdist(bo(t, w), ref).max() < 2e-7
    w2 = np.where(np.arange(len(t)) % 2 == 0, 0.004, 0.0)               # mixed: box where w > 0, point elsewhere
    got = bo(t, w2)
    assert _qdist(got[1::2], tel.orientation_at(t[1::2])).max() < 1e-6


def test_box_orient_nonuniform_grid():
    tel = make_synthetic(1.0)
    keep = np.ones(len(tel.imu_t), bool)
    keep[500:520] = False                                                # a gap -> bisection path
    bo = _BoxOrient(tel.imu_t[keep], tel.imu_q[keep])
    assert not bo.uniform
    t = np.linspace(0.1, 0.9, 101)
    assert _qdist(bo(t), tel.orientation_at(t)).max() < 1e-4


def test_build_deltas():
    tid = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])
    k = np.array([5, 6, 7, 9, 3, 4, 10, 11, 12])
    pa, pb, dl = build_deltas(tid, k)
    assert pa.tolist() == [0, 1, 4, 6, 7] and pb.tolist() == [1, 2, 5, 7, 8]
    assert dl.tolist() == [[0, 1], [3, 4]]                               # (5,6,7) and (10,11,12); 9 breaks track 0


def test_choose_windows_skips_static_and_spreads():
    tel = make_synthetic(12.0)
    q = tel.imu_q.copy()
    q[tel.imu_t < 4.0] = q[np.searchsorted(tel.imu_t, 4.0)]
    tel.imu_q = q
    w = choose_windows(tel, 3, 60)
    assert len(w) == 3
    assert all(tel.frame_t[c['start'] + c['n'] // 2] > 4.0 for c in w)     # mostly in the moving part
    s = sorted(c['start'] for c in w)
    assert all(b - a >= 90 for a, b in zip(s, s[1:]))


def make_vib(duration):
    """make_synthetic + prop-like vibration (120 / 180 / 230 Hz, 0.06-0.08 deg): the timing fit's HF-consistency
    check needs gyro HF that the picture follows (as on OA4 / O4 Pro)."""
    from test_calib import _rotvec_path
    tel = make_synthetic(duration)
    th = _rotvec_path(tel.imu_t)
    for ax, (a, f, ph) in enumerate(((0.08, 120.0, 0.3), (0.06, 230.0, 1.2), (0.06, 180.0, 2.1))):
        th[:, ax] += np.deg2rad(a) * np.sin(2 * np.pi * f * tel.imu_t + ph)
    tel.imu_q = qexp(th)
    return tel


def _params(**kw):
    base = dict(n_windows=3, window_s=1.0, min_deltas=500, max_points=400, width=640)
    base.update(kw)
    return TimecalParams(**base)


@pytest.mark.parametrize('true_ms', [0.8, -1.0])
def test_recovers_injected_offset(true_ms):
    tel = make_vib(8.0)
    rend = FastRenderer(tel, offset=true_ms * 1e-3)
    tm = calibrate_timing(tel, frame_reader=rend, params=_params())
    n = tm.notes['timecal']
    fit = n['fit']
    print('\n', true_ms, fit['estimate'], fit['sigma'], n['decision'], n['runtime_s'])
    assert n['decision']['offset']
    assert abs(tm.offset_s * 1e3 - true_ms) < 0.05
    assert tm.readout_s is None and tm.focal_scale == 1.0                # correct metadata stays


def test_correct_metadata_is_left_alone_with_parallax():
    tel = make_vib(8.0)
    rend = FastRenderer(tel, velocity=(2.0, 0.5, 8.0), radius=50.0)          # 1-3 px/frame parallax everywhere
    tm = calibrate_timing(tel, frame_reader=rend, params=_params())
    fit = tm.notes['timecal']['fit']
    print('\nparallax, no error:', fit['estimate'], fit['sigma'])
    assert abs(tm.offset_s * 1e3) < 0.05
    assert tm.readout_s is None and tm.focal_scale == 1.0


def test_readout_error_is_found_after_the_offset():
    tel = make_vib(8.0)
    rend = FastRenderer(tel, offset=0.5e-3, readout=tel.readout_s * 1.10)
    tm = calibrate_timing(tel, frame_reader=rend, params=_params())
    fit = tm.notes['timecal']['fit']
    print('\nreadout +10%:', fit['estimate'], fit['sigma'], tm.notes['timecal']['decision'])
    assert abs(tm.offset_s * 1e3 - 0.5) < 0.05
    assert tm.readout_s is not None and abs(tm.readout_s / (tel.readout_s * 1.10) - 1) < 0.02


def test_offset_without_gyro_hf_is_not_applied():
    """No gyro content above lf_hz (the synthetic path tops out at 23 Hz): the HF-consistency check cannot confirm
    an offset (on real O3 footage the maneuver-driven LF information was biased by -0.8 ms), so it is reported but
    the metadata timing is kept."""
    tel = make_synthetic(8.0)
    th = np.zeros((len(tel.imu_t), 3))                                     # <= 9 Hz only (no leak above 40 Hz)
    for ax, (a, f, ph) in enumerate(((5.0, 2.1, 0.3), (7.0, 1.3, 0.0), (6.0, 1.7, 0.9))):
        th[:, ax] += np.deg2rad(a) * np.sin(2 * np.pi * f * tel.imu_t + ph)
        th[:, ax] += np.deg2rad(a / 4) * np.sin(2 * np.pi * (4.3 + 2 * ax) * tel.imu_t + ph)
    tel.imu_q = qexp(th)
    rend = FastRenderer(tel, offset=0.8e-3)
    tm = calibrate_timing(tel, frame_reader=rend, params=_params())
    n = tm.notes['timecal']
    print('\nno HF:', n['fit']['estimate'], n['fit'].get('hf_check', {}).get('gain'), n['decision'])
    assert abs(n['fit']['estimate']['offset_ms'] - 0.8) < 0.15               # measured ...
    assert tm.offset_s == 0.0 and not n['decision']['offset']               # ... but not applied
    assert any('HF' in r for r in n['decision']['reasons'])


def test_skips_without_highrate_or_with_eis():
    tel = make_synthetic(2.0)
    tm = calibrate_timing(replace(tel, has_highrate=False), frame_reader=lambda a, b: None)
    assert tm.offset_s == 0.0 and tm.readout_s is None and 'skipped' in tm.notes['timecal']['status']
    tm = calibrate_timing(replace(tel, eis_baked=True), frame_reader=lambda a, b: None)
    assert tm.offset_s == 0.0 and 'EIS' in tm.notes['timecal']['status']


def _val(gain_in=0.05, held=(0.04, 0.05, 0.03)):
    h = [dict(fold=i, gain=g) for i, g in enumerate(held)]
    return dict(gain_in=gain_in, heldout=h, heldout_pos_frac=float(np.mean(np.array(held) > 0)) if held else np.nan,
                heldout_mean=float(np.mean(held)) if held else np.nan)


def _fake_res(offs, sig=0.02, off=None, readout=True, focal_ok=False, val=None):
    pw = [dict(window=i, offset_ms=o, offset_sigma_ms=sig) for i, o in enumerate(offs)]
    used = [i for i, o in enumerate(offs) if abs(o - np.median(offs)) <= max(0.25, 4 * sig)]
    off = float(np.mean([offs[i] for i in used])) if off is None else off
    return dict(status='ok', estimate=dict(offset_ms=off, readout_pct=5.0, focal_pct=0.0, exposure_slope=0.0,
                                           box_pct=0.0),
                sigma=dict(offset_ms=0.02, readout_pct=0.5, focal_pct=np.nan, exposure_slope=np.nan, box_pct=np.nan),
                fine=dict(interior=True), coarse=dict(offset_ms=off), per_window=pw, windows_used=used,
                nuisance_decision=dict(readout_pct=readout, focal_pct=focal_ok, exposure_slope=False, box_pct=False),
                validation=val if val is not None else _val())


def test_decision_rules():
    tel = make_synthetic(1.0)
    p = TimecalParams()
    d = _decide(_fake_res([0.90, 0.92, 0.95]), tel, p)
    assert d['offset'] and d['readout'] and not d['focal'] and d['source'] == 'conditional'
    d = _decide(_fake_res([0.9, -1.2, 2.5]), tel, p)                     # no two windows agree
    assert not d['offset'] and not d['readout'] and d['source'] is None
    d = _decide(_fake_res([0.9, 0.92, 3.0]), tel, p)                     # one outlier window of three: excluded
    assert d['offset']
    r = _fake_res([0.9, 0.92, 0.95])
    r['sigma']['offset_ms'] = 0.3
    assert not _decide(r, tel, p)['offset']


def test_nuisance_selection():
    p = TimecalParams()
    act = dict(offset_ms=True, readout_pct=True, focal_pct=True, exposure_slope=False, box_pct=True)
    joint = dict(estimate=dict(offset_ms=0.1, readout_pct=5.0, focal_pct=-2.5, exposure_slope=0.0, box_pct=80.0),
                 sigma=dict(offset_ms=0.02, readout_pct=0.5, focal_pct=0.4, exposure_slope=np.nan, box_pct=12.0))
    nd = _decide_nuisance(joint, act, p)
    assert nd['readout_pct'] and not nd['focal_pct'] and nd['box_pct']      # focal: reported, not applied (default)
    assert not _decide_nuisance(joint, act, replace(p, apply_focal=True))['focal_pct']   # sigma 0.4 % > 0.3 %
    joint['sigma']['focal_pct'] = 0.2
    assert _decide_nuisance(joint, act, replace(p, apply_focal=True))['focal_pct']
    joint['sigma']['box_pct'] = 40.0                                         # box: 80 +- 40 % is not clear
    assert not _decide_nuisance(joint, act, p)['box_pct']
    joint['estimate']['readout_pct'] = 1.0                                   # < 1.5 %: not worth it
    assert not _decide_nuisance(joint, act, p)['readout_pct']


def test_validation_gate():
    """Statistically 'significant' corrections that do not lower the cost enough in sample, or that held-out windows
    do not confirm, are NOT applied (O3: -0.29..+0.14 ms at 3-6 jackknife sigma, held-out cost +-0.4 %)."""
    tel = make_synthetic(1.0)
    p = TimecalParams()
    d = _decide(_fake_res([0.20, 0.25, 0.22], val=_val(gain_in=0.001, held=(0.002, 0.001, 0.001))), tel, p)
    assert not d['offset'] and not d['readout'] and any('not worth' in s for s in d['reasons'])
    d = _decide(_fake_res([0.20, 0.25, 0.22], val=_val(gain_in=0.02, held=(-0.004, 0.003, -0.002))), tel, p)
    assert not d['offset'] and any('held-out' in s for s in d['reasons'])
    d = _decide(_fake_res([0.20, 0.25, 0.22], val=_val(gain_in=0.02, held=(0.01, -0.001, 0.02))), tel, p)
    assert d['offset']                                                      # 2 of 3 folds, mean +1 %


def test_nuisance_only_fallback():
    """Offset not confident (e.g. 1/61 s clip with a noisy lag) but a clear exposure-box change that validates ->
    the box is applied alone; the offset stays at the metadata."""
    tel = make_synthetic(1.0)
    p = TimecalParams()
    r = _fake_res([-2.6, -2.4, -2.9], readout=False)
    r['sigma']['offset_ms'] = 0.5                                          # > max(0.1 ms, 10 % of 2.6 ms)
    r['nuisance_decision']['box_pct'] = True
    r['nuisance_only'] = dict(estimate=dict(offset_ms=0.0, readout_pct=0.0, focal_pct=0.0, exposure_slope=0.0,
                                            box_pct=90.0), validation=_val(gain_in=0.01, held=(0.01, 0.005, 0.02)))
    d = _decide(r, tel, p)
    assert d['source'] == 'nuisance_only' and d['box'] and not d['offset']
    r['nuisance_only']['validation'] = _val(gain_in=0.01, held=(-0.01, -0.005, 0.02))
    d = _decide(r, tel, p)
    assert d['source'] is None and not d['box']


def test_exposure_scale_reaches_the_plan():
    from stillpoint.plan_build import build_plan, camera_orientation_fn, exposure_avg_window
    tel = make_synthetic(1.0)
    tel.exposure_s = np.full(tel.n_frames, 1.0 / 61)
    F = tel.n_frames
    virt = np.tile([1.0, 0, 0, 0], (F, 1))
    tm1, tm2 = TimeModel(), TimeModel(exposure_scale=1.8)
    p1 = build_plan(tel, tm1, camera_orientation_fn(tel, tm1), virt, 900.0, 1280, 960, n_rows=8)
    p2 = build_plan(tel, tm2, camera_orientation_fn(tel, tm2), virt, 900.0, 1280, 960, n_rows=8)
    assert p2.meta['time_model']['exposure_scale'] == 1.8 and p1.meta['time_model']['exposure_scale'] == 1.0
    assert np.abs(p1.row_mats - p2.row_mats).max() > 1e-6                   # a wider box changes the rows
    from stillpoint.plan_build import row_samples_y
    ys = row_samples_y(tel.height, 8)
    t_rows = tel.frame_t[:, None] + tel.readout_s * ((ys[None, :] + 0.5) / tel.height - 0.5)
    ref = exposure_averaged_q(tel.orientation_at, t_rows, exposure_avg_window(tel.exposure_s) * 1.8, taps=257)
    from stillpoint.geom import quat_to_mat
    Rc = quat_to_mat(ref)
    M = np.einsum('fjki,fkl->fjil', Rc, quat_to_mat(virt))
    assert np.abs(M - p2.row_mats).max() < 2e-6


def test_adaptive_taps_keep_short_windows_and_fix_long_ones():
    tel = make_synthetic(1.0)
    t = np.linspace(0.2, 0.8, 50)[:, None]
    qf = tel.orientation_at
    short = np.full(50, 0.0045)                                             # OA4 1 kHz, <= 8 ms: 9 taps as in v4
    a = exposure_averaged_q(qf, t, short, taps=9)
    b = exposure_averaged_q(qf, t, short, taps=9, sample_rate=1000.0)
    assert np.array_equal(a, b)                                             # <= 8 samples: unchanged (v4 plans)
    long_ = np.full(50, 0.030)
    ref = exposure_averaged_q(qf, t, long_, taps=1001)
    e9 = _qdist(exposure_averaged_q(qf, t, long_, taps=9), ref).max()
    ea = _qdist(exposure_averaged_q(qf, t, long_, taps=9, sample_rate=tel.imu_rate), ref).max()
    assert ea < 0.5 * e9


def test_apply_exposure_slope():
    tel = make_synthetic(1.0)
    tel.exposure_s = np.linspace(0.001, 0.009, tel.n_frames)
    tm = TimeModel(notes={'timecal': dict(applied=dict(exposure_slope=0.25, exposure_ref_s=0.005))})
    t2 = apply_exposure_slope(tel, tm)
    assert np.allclose(t2.frame_t - tel.frame_t, 0.25 * (tel.exposure_s - 0.005))
    assert apply_exposure_slope(tel, TimeModel()) is tel
