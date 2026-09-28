"""WP-A telemetry tests on Jimmy's real clips (read-only). Run:
    cd stillpoint && PYTHONPATH=engine .venv/bin/python -m pytest tests/test_telemetry.py -q
Clips that are missing on this machine are skipped.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from stillpoint import video
from stillpoint.geom import qconj, qlog, qmul, quat_to_mat, slerp_series
from stillpoint.telemetry import Q_CAM2BODY, load_telemetry
from eval.footage import O3_DIR, oa4  # noqa: E402  (env-configurable, see eval/footage.py)

HOME = os.path.expanduser('~')
O3 = O3_DIR
CLIPS = {
    '0025': os.path.join(O3, 'DJI_0025.MP4'),
    '0026': os.path.join(O3, 'DJI_0026.MP4'),
    'joined': os.path.join(O3, 'DJI_0025_joined.MP4'),
    '0002': oa4('DJI_20260926152149_0002_D.MP4'),
    '0005': oa4('DJI_20260926153751_0005_D.MP4'),
    '0006': oa4('DJI_20260926155933_0006_D.MP4'),
    '0007': oa4('DJI_20260926155953_0007_D.MP4'),
}
O3_K = [0.24991769, 0.01360575, -0.06208358, 0.01219307]
OA4_K = [0.155131, 0.137141, -0.093861, 0.004170]

_TEL: dict = {}


def tel_of(name, cache_dir=None):
    p = CLIPS[name]
    if not os.path.exists(p):
        pytest.skip(f'missing clip {p}')
    key = (name, cache_dir)
    if key not in _TEL:
        _TEL[key] = load_telemetry(p, cache_dir=cache_dir)
    return _TEL[key]


def assert_uniform(t, dt_expected, tol_rel=1e-6):
    d = np.diff(t)
    assert np.all(d > 0)
    assert abs(np.median(d) / dt_expected - 1) < 1e-3, (np.median(d), dt_expected)
    assert (d.max() - d.min()) / np.median(d) < tol_rel, 'IMU grid is not uniform'


def assert_covers_rows(tel, edge=0):
    """Every row (t = frame_t +- readout/2) of every frame except `edge` frames at each end lies inside the IMU span
    (OA4's first 1 kHz sample comes 2-3 ms after T_0, so its frame 0 top rows are not covered)."""
    a, b = tel.extra['imu_full_coverage_frames']
    assert a <= edge and b >= tel.n_frames - 1 - edge, (a, b)
    assert tel.imu_t[0] <= tel.frame_t[edge] - tel.readout_s / 2
    assert tel.imu_t[-1] >= tel.frame_t[-1 - edge] + tel.readout_s / 2


# ----------------------------------------------------------------------------------------------- O3


@pytest.mark.parametrize('name,F,N', [('0026', 268, 8938), ('0025', 2965, 98884)])
def test_o3_basic(name, F, N):
    tel = tel_of(name)
    info = video.probe(CLIPS[name])
    assert tel.camera == 'DJI O3 (FC8383)'
    assert (tel.width, tel.height) == (3840, 2160)
    assert tel.n_frames == F == info['n_frames']
    np.testing.assert_array_equal(tel.frame_pts, info['frame_pts'])
    np.testing.assert_allclose(tel.frame_pts, np.arange(F) * 1001 / 60000, atol=1e-12)
    assert abs(tel.fps - 60000 / 1001) < 1e-9
    assert len(tel.imu_t) == N                       # 33/34 per frame, no gaps, nothing dropped
    assert tel.has_highrate and not tel.eis_baked
    assert tel.segments == [(0, F - 1)]
    s = tel.extra['segment_time_maps'][0]['scale']   # sensor clock -> video clock (59.969295 vs 59.94006)
    assert abs(s - 59.969295501708984 / (60000 / 1001)) < 2e-6
    assert_uniform(tel.imu_t, s / 2000.0)            # exact 2 kHz grid in sensor time
    assert abs(tel.imu_rate - 2000 / s) < 1e-3
    g = tel.extra['imu_grid'][-1]
    assert g['resid_rms_s'] < 30e-6 and g['resid_max_s'] < 100e-6   # blocks tile the grid (offset model incl. -exp/2)
    assert abs(g['dt_s'] - 5e-4) < 1e-9
    # lens + readout read from the file, must match DJI's O3 calibration
    L = tel.lens
    assert L.model == 'kb4' and abs(L.fx - 1405.129) < 0.01 and L.fx == L.fy
    np.testing.assert_allclose(L.k, O3_K, atol=1e-6)
    assert (L.cx, L.cy, L.width, L.height) == (1919.5, 1079.5, 3840, 2160)
    assert tel.extra['lens_source'] == 'file'
    assert abs(tel.readout_s - 9.7247e-3) < 2e-6
    # frame_t = centre-row mid-exposure: frame label T_k minus exposure/2 (O3 folds -exp/2 into `offset`)
    r = tel.frame_t - (tel.frame_pts - 0.5 * s * tel.exposure_s)
    assert np.abs(r - r.mean()).max() < 20e-6 and abs(r.mean()) < 5e-6
    assert_covers_rows(tel)
    assert tel.gravity_q is not None
    assert np.allclose(np.linalg.norm(tel.imu_q, axis=1), 1)
    assert np.all(np.einsum('ij,ij->i', tel.imu_q[1:], tel.imu_q[:-1]) > 0)


def test_o3_matches_prototype_decode():
    """Quaternions are the verified prototype's (research/proto/o3_parse.py) mapped to the camera frame; the
    timeline differs only by an affine map (+ exposure-jump fix, none in 0026)."""
    ref = os.path.join(os.path.dirname(__file__), '..', 'work', 'o3', 'DJI_0026_imu.npz')
    if not os.path.exists(ref):
        pytest.skip('prototype output missing')
    z = np.load(ref)
    tel = tel_of('0026')
    q_ref = qmul(z['q_raw'], Q_CAM2BODY)
    d = np.abs(np.einsum('ij,ij->i', q_ref / np.linalg.norm(q_ref, axis=1, keepdims=True), tel.imu_q))
    assert len(d) == len(tel.imu_q) and d.min() > 1 - 1e-12
    t_ref = z['q_t_fixed']
    A = np.vstack([t_ref, np.ones_like(t_ref)]).T
    res = tel.imu_t - A @ np.linalg.lstsq(A, tel.imu_t, rcond=None)[0]
    assert np.abs(res).max() < 30e-6
    # 0025 (exposure 1.39-7.19 ms): within each block, picture-vs-sample timing equals the prototype's verified
    # 'fixed' timeline (which puts the centre row at frame_ts and jumps at exposure changes); ours is uniform.
    ref = os.path.join(os.path.dirname(__file__), '..', 'work', 'o3', 'DJI_0025_imu.npz')
    if os.path.exists(ref):
        z = np.load(ref)
        tel = tel_of('0025')
        first = np.concatenate([[0], np.cumsum(z['att_count'])])[:-1]
        pic_fixed = (z['frame_ts_us'] - z['frame_ts_us'][0]) * 1e-6 * float(z['sensor_fps']) / float(z['meta_fps'])
        d = (pic_fixed - z['q_t_fixed'][first]) - (tel.frame_t - tel.imu_t[first])
        assert abs(d.mean()) < 2e-6 and np.abs(d).max() < 40e-6


def test_o3_joined_segments():
    """21 GB concatenation of 0025..0034: split at clock discontinuities; auto-split joins stay continuous; the
    first segment is bit-identical to parsing DJI_0025 alone. (Uses work/cache; first run ~13 s.)"""
    if not os.path.exists(CLIPS['joined']):
        pytest.skip('joined clip missing')
    tel = load_telemetry(CLIPS['joined'])
    assert tel.n_frames == 66500
    assert tel.segments == [(0, 2964), (2965, 3232), (3233, 18674), (18675, 30362), (30363, 40186), (40187, 66499)]
    cb = {b['frame']: b['continuous'] for b in tel.extra['clip_bounds']}
    assert cb == {2965: False, 3233: False, 14842: True, 18675: False, 30363: False, 40187: False,
                  51989: True, 63471: True}
    assert np.all(np.diff(tel.imu_t) > 0)
    for g in tel.extra['imu_grid']:
        assert g['resid_max_s'] < 100e-6
    a = tel_of('0025')
    n = len(a.imu_t) - 20
    np.testing.assert_allclose(tel.frame_t[:2965], a.frame_t, atol=1e-9)
    np.testing.assert_allclose(tel.imu_t[:n], a.imu_t[:n], atol=1e-9)
    np.testing.assert_allclose(tel.imu_q[:n], a.imu_q[:n], atol=1e-12)


# ----------------------------------------------------------------------------------------------- OA4


@pytest.mark.parametrize('name,F,N,Wdeg', [('0005', 519, 8659, 7.9), ('0006', 482, 8025, 23.5)])
def test_oa4_highrate(name, F, N, Wdeg):
    tel = tel_of(name)
    assert tel.camera == 'DJI Osmo Action 4'
    assert (tel.width, tel.height) == (3840, 2880)
    assert tel.n_frames == F == video.probe(CLIPS[name])['n_frames']
    assert len(tel.imu_t) == N                     # 0006: truncated last block (11 samples) dropped
    assert tel.has_highrate and not tel.eis_baked
    s = tel.extra['segment_time_maps'][0]['scale']
    assert abs(s - 1) < 1e-4
    assert_uniform(tel.imu_t, s / 1000.0)
    assert abs(tel.imu_rate - 1000) < 0.1
    assert tel.extra['imu_grid'][-1]['resid_rms_s'] < 50e-6   # 1/8-sample quantisation of `offset`
    L = tel.lens
    assert abs(L.fx - 1457.0737) < 0.01 and (L.cx, L.cy) == (1919.5, 1439.5)
    np.testing.assert_allclose(L.k, OA4_K, atol=1e-6)
    assert abs(tel.readout_s - 11.0873e-3) < 5e-6
    np.testing.assert_allclose(tel.frame_t - tel.frame_pts, -0.8e-3, atol=20e-6)   # T_k - 0.8 ms
    assert_covers_rows(tel, edge=1)
    # gravity re-levelling: constant world rotation of the 1 kHz stream, then agreement with the accelerometer
    assert abs(tel.extra['gravity_W_deg'] - Wdeg) < 1.0
    assert tel.extra['gravity_W_spread_deg']['p95'] < 1.0
    _check_gravity(tel, tel.gravity_q, 5.0)


def _check_gravity(tel, qg, max_median_deg):
    from scipy.ndimage import uniform_filter1d
    acc = uniform_filter1d(tel.extra['accel_cam_g'], 31, axis=0)
    R = quat_to_mat(slerp_series(tel.imu_t, qg, tel.frame_t))
    f_pred = -R[:, 2, :]                           # specific force at rest = -g; world z down -> -R^T e_z
    c = np.einsum('ni,ni->n', f_pred, acc) / np.linalg.norm(acc, axis=1)
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    assert np.median(ang) < max_median_deg, np.median(ang)


def test_oa4_perframe_0002():
    tel = tel_of('0002')
    assert (tel.width, tel.height) == (3840, 2160)
    assert tel.n_frames == 23512 and len(tel.imu_t) == 23512
    assert not tel.has_highrate and not tel.eis_baked
    assert abs(tel.imu_rate - 59.94) < 0.05
    assert tel.lens.fx == pytest.approx(1457.0737, abs=0.01) and tel.lens.cy == 1079.5
    assert tel.extra['lens_source'].startswith('fallback')
    assert abs(tel.readout_s - 11.0876e-3 * 2160 / 2880) < 5e-6
    assert tel.extra['timing']['cam_quat_delay_s'] == pytest.approx(0.0087)
    assert tel.segments == [(0, 23511)]
    _check_gravity(tel, tel.imu_q, 5.0)


def test_oa4_eis_0007():
    tel = tel_of('0007')
    assert tel.eis_baked and not tel.has_highrate
    assert tel.extra['eis_status_name'] == 'EIS_TRADEOFF'
    assert np.all(tel.extra['dbgi_eis_mode'] == 4)
    assert tel.extra['dbgi_q_eis_cam'].shape == (371, 4)
    assert tel.extra['timing']['cam_quat_delay_s'] == pytest.approx(0.018)
    for n in ('0005', '0006'):
        t = tel_of(n)
        assert not t.eis_baked and np.all(t.extra['dbgi_eis_mode'] == 0)


def test_cache_roundtrip(tmp_path):
    a = tel_of('0006')
    b = load_telemetry(CLIPS['0006'], cache_dir=str(tmp_path))      # parse + write
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].suffix == '.npz'
    c = load_telemetry(CLIPS['0006'], cache_dir=str(tmp_path))      # read back
    for x in (b, c):
        for k in ('frame_pts', 'frame_t', 'exposure_s', 'imu_t', 'imu_q', 'gravity_q'):
            np.testing.assert_array_equal(getattr(x, k), getattr(a, k))
        assert x.lens.fx == a.lens.fx and np.array_equal(x.lens.k, a.lens.k)
        assert (x.readout_s, x.imu_rate, x.has_highrate, x.eis_baked, x.segments) == \
               (a.readout_s, a.imu_rate, a.has_highrate, a.eis_baked, a.segments)
    np.testing.assert_array_equal(c.extra['accel_cam_g'], a.extra['accel_cam_g'])
    assert c.extra['gravity_W_deg'] == a.extra['gravity_W_deg']


# ------------------------------------------------------------------------------ axis convention vs vision

def _kabsch(P, Q):
    U, _s, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def visual_pairs(tel, path, start_frame, n_frames, width=960, seed=0):
    """KLT + fisheye unprojection (telemetry's own lens) + RANSAC Kabsch -> per consecutive pair the camera
    rotation vector in frame-a coordinates and the mean feature row (full-res px)."""
    lens = tel.lens.scaled(width / tel.width)
    rng = np.random.default_rng(seed)
    rows, prev = [], None
    for k, _pts, g in video.iter_gray(path, width, start_frame, n_frames):
        if prev is not None:
            pk, pg = prev
            p0 = cv2.goodFeaturesToTrack(pg, 500, 0.01, 10)
            if p0 is not None and len(p0) >= 40:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, p0, None, winSize=(21, 21), maxLevel=4)
                p0b, stb, _ = cv2.calcOpticalFlowPyrLK(g, pg, p1, None, winSize=(21, 21), maxLevel=4)
                ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm(p0b - p0, axis=2)[:, 0] < 0.3)
                if ok.sum() >= 30:
                    a, b = p0[ok][:, 0].astype(np.float64), p1[ok][:, 0].astype(np.float64)
                    b0, b1 = lens.unproject(a), lens.unproject(b)
                    best = None
                    for _ in range(100):
                        i3 = rng.choice(len(b0), 3, replace=False)
                        R = _kabsch(b0[i3], b1[i3])
                        inl = np.linalg.norm(b1 - b0 @ R.T, axis=1) < 1.0 / lens.fx
                        if best is None or inl.sum() > best.sum():
                            best = inl
                    if best.sum() >= 20:
                        R = _kabsch(b0[best], b1[best])          # b1 = R b0 (points); camera rotation = R^T
                        rv = cv2.Rodrigues(R.T)[0][:, 0]
                        y = float(np.mean(0.5 * (a[best, 1] + b[best, 1]))) * tel.height / g.shape[0]
                        rows.append((pk, k, rv, y))
        prev = (k, g)
    return rows


def telemetry_rotvecs(tel, rows, delta=0.0):
    ka = np.array([r[0] for r in rows])
    kb = np.array([r[1] for r in rows])
    y = np.array([r[3] for r in rows])
    rs = tel.readout_s * ((y + 0.5) / tel.height - 0.5)
    qa = tel.orientation_at(tel.frame_t[ka] + rs + delta)
    qb = tel.orientation_at(tel.frame_t[kb] + rs + delta)
    return qlog(qmul(qconj(qa), qb))


def axis_report(tel, rows, lags_ms=np.arange(-8.0, 8.01, 0.25)):
    vis = np.array([r[2] for r in rows])
    G = telemetry_rotvecs(tel, rows)
    corr = [float(np.corrcoef(vis[:, i], G[:, i])[0, 1]) for i in range(3)]
    M = np.linalg.lstsq(G, vis, rcond=None)[0].T               # vis ~ M @ G  (should be ~identity)
    res = []
    for L in lags_ms:
        e2 = np.sort(np.sum((vis - telemetry_rotvecs(tel, rows, L * 1e-3)) ** 2, axis=1))
        res.append(np.sqrt(np.mean(e2[:int(0.9 * len(e2))])))
    i = int(np.argmin(res))
    return dict(n=len(rows), corr=corr, M=M, best_lag_ms=float(lags_ms[i]), lag_at_edge=i in (0, len(res) - 1),
                corr_all=float(np.corrcoef(vis.ravel(), G.ravel())[0, 1]))


@pytest.mark.parametrize('name,start,min_corr,max_lag_ms', [
    ('0025', 1470, 0.95, 2.0),     # O3, 2 kHz, strong 3-axis move at 24.5 s
    ('0006', 60, 0.99, 2.0),       # OA4 1 kHz, waving
    ('0002', 9000, 0.95, 3.0),     # OA4 per-frame cam_quat (16:9), handheld 150 s
])
def test_axis_convention_vs_vision(name, start, min_corr, max_lag_ms):
    tel = tel_of(name)
    rows = visual_pairs(tel, CLIPS[name], start, 120)
    assert len(rows) >= 100
    r = axis_report(tel, rows)
    assert min(r['corr']) > min_corr, r
    assert r['corr_all'] > 0.98, r
    # the best linear map image<-telemetry is the identity (no axis swap / sign flip / conjugation)
    assert np.array_equal(np.round(r['M']), np.eye(3)), r['M']
    assert np.all(np.abs(np.diag(r['M']) - 1) < 0.2), r['M']
    # frame_t convention: residual timing error small (the metadata timing model is right)
    assert not r['lag_at_edge'] and abs(r['best_lag_ms']) <= max_lag_ms, r


def test_oa4_picture_offset_model():
    """Short shutters: +0.12 ms (0012 vision fit); 1/61 s: -0.8 ms (0005/0006); linear and monotonic between."""
    from stillpoint.telemetry import oa4_picture_offset
    c = oa4_picture_offset(np.array([0.0008, 0.0025, 0.0046, 0.010, 1 / 61, 0.03, np.nan]))
    np.testing.assert_allclose(c[:3], 0.00012, atol=1e-9)
    np.testing.assert_allclose(c[4:], -0.0008, atol=1e-9)        # 1/61 s, longer, unknown -> the old constant
    assert c[2] > c[3] > c[4]
