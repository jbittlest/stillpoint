"""Fast telemetry reading (pread of only the djmd/dbgi samples) vs the PyAV demux path and vs the legacy parser.
Run:
    cd stillpoint && PYTHONPATH=engine .venv/bin/python -m pytest tests/test_telemetry_fast.py -q
Clips that are missing on this machine (or an unmounted SD card) are skipped. Everything is read-only; nothing is
written next to the footage (cache_dir=None, or a tmp dir).
"""
from __future__ import annotations

import json
import os
import time
import types

import numpy as np
import pytest

from stillpoint import telemetry as T
from stillpoint import video
from eval.footage import O3_DIR, SD_DIR, oa4  # noqa: E402  (env-configurable, see eval/footage.py)

HOME = os.path.expanduser('~')
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
O3 = O3_DIR
SD = SD_DIR
CLIPS = {
    '0026': os.path.join(O3, 'DJI_0026.MP4'),                                   # O3, 268 frames
    '0025': os.path.join(O3, 'DJI_0025.MP4'),                                   # O3, 2965 frames
    '0005': oa4('DJI_20260926153751_0005_D.MP4'),     # OA4 4:3, 1 kHz, EIS off
    '0002': oa4('DJI_20260926152149_0002_D.MP4'),     # OA4 16:9, per-frame, 5.4 GB
    '0007': oa4('DJI_20260926155953_0007_D.MP4'),     # OA4, EIS on
    'sd0009': os.path.join(SD, 'DJI_20260927084845_0009_D.MP4'),                # OA4 4:3 on the SD card, 126 MB
}
SD_LONG = os.path.join(SD, 'DJI_20260927091931_0012_D.MP4')                     # 6.2 GiB, 8 min, 23950 frames


def need(p):
    if not os.path.exists(p):
        pytest.skip(f'missing {p}')
    return p


# ------------------------------------------------------------------------------------------ comparison helper


def _norm(v):
    """Extras as the npz cache stores them (JSON for everything but arrays), so parsed and cached compare alike."""
    return json.loads(json.dumps(v, default=T._jsonable))


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def telemetry_diff(a, b, dbgi_subset: bool = False, ignore=('meta_reader',)) -> list[str]:
    """Names of everything that differs between two Telemetry objects (exact: bitwise arrays, == scalars).
    dbgi_subset: b was parsed with sparse dbgi (dbgi='auto', EIS off) and a with every dbgi frame -> b's dbgi_*
    arrays must equal a's rows at b.extra['dbgi_frames'] (the dbgi_q_*_cam quaternions up to a per-row sign: they
    are made sign-continuous along whichever frames were read)."""
    out = []
    for k in ('source', 'camera', 'width', 'height', 'fps', 'readout_s', 'imu_rate', 'has_highrate', 'eis_baked'):
        if not _same(getattr(a, k), getattr(b, k)):
            out.append(k)
    for k in ('frame_pts', 'frame_t', 'exposure_s', 'imu_t', 'imu_q'):
        if not _same(getattr(a, k), getattr(b, k)):
            out.append(k)
    if (a.gravity_q is None) != (b.gravity_q is None) or (a.gravity_q is not None and not _same(a.gravity_q, b.gravity_q)):
        out.append('gravity_q')
    la, lb = a.lens, b.lens
    if not (la.model == lb.model and (la.fx, la.fy, la.cx, la.cy, la.width, la.height) ==
            (lb.fx, lb.fy, lb.cx, lb.cy, lb.width, lb.height) and _same(np.asarray(la.k), np.asarray(lb.k))):
        out.append('lens')
    if [tuple(s) for s in a.segments] != [tuple(s) for s in b.segments]:
        out.append('segments')
    sub = b.extra.get('dbgi_frames') if dbgi_subset else None
    keys = (set(a.extra) | set(b.extra)) - set(ignore) - ({'dbgi_frames'} if dbgi_subset else set())
    for k in sorted(keys):
        if k not in a.extra or k not in b.extra:
            out.append(f'extra.{k} (missing)')
            continue
        x, y = a.extra[k], b.extra[k]
        if sub is not None and k.startswith('dbgi_') and isinstance(x, np.ndarray):
            x = x[sub]
            if k.endswith('_cam') and x.shape == np.shape(y):
                flip = np.nan_to_num(np.einsum('ij,ij->i', x, y)) < 0
                x = np.where(flip[:, None], -x, x)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            ok = _same(np.asarray(x), np.asarray(y))
        else:
            ok = _same(_norm(x), _norm(y))
        if not ok:
            out.append(f'extra.{k}')
    return out


def pyav_samples(path, track_ids):
    """Independent reference: every packet of the given MP4 track ids, demuxed by PyAV."""
    import av
    with av.open(path) as c:
        sel = [s for s in c.streams if s.id in track_ids]
        idx = {s.index: s.id for s in sel}
        got = {t: [] for t in track_ids}
        for pk in c.demux(sel):
            if pk.size == 0 and pk.pts is None and pk.dts is None:
                continue
            got[idx[pk.stream.index]].append(bytes(pk))
    return got


# ------------------------------------------------------------------------------------------ byte-range reader


def test_read_ranges_matches_naive_reads(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.integers(0, 256, 300_000, dtype=np.uint8).tobytes()
    p = tmp_path / 'blob.bin'
    p.write_bytes(data)
    offs = rng.integers(0, len(data), 400)
    sizes = rng.integers(0, 3000, 400)
    offs[:5] = [0, 10, 10, 50, len(data) - 100]        # duplicates, overlap, and one running past EOF
    sizes[:5] = [100, 40, 40, 5000, 1000]
    for gap in (0, 64, 16 * 1024, 10 ** 9):             # no merging ... everything in one read
        got = video.read_ranges(str(p), offs, sizes, max_gap=gap)
        assert got == [data[o:o + s] for o, s in zip(offs, sizes)]
    assert video.read_ranges(str(p), [], []) == []


# ------------------------------------------------------------------------------------------ fast == PyAV


@pytest.mark.parametrize('name', ['0026', '0025', '0005', '0002', '0007', 'sd0009'])
def test_mp4_samples_equal_pyav_packets(name):
    """The sample table read (offsets/sizes from stsz/stco/co64/stsc) returns exactly PyAV's packets."""
    p = need(CLIPS[name])
    tracks = video.mp4_tracks(p)
    meta = [t for t in tracks if t.fourcc in ('djmd', 'dbgi')]
    assert meta and any(t.fourcc == 'djmd' for t in meta)
    ref = pyav_samples(p, [t.track_id for t in meta])
    for t in meta:
        fast = t.read_samples(p)
        assert len(fast) == t.n_samples == len(ref[t.track_id]), t.fourcc
        assert fast == ref[t.track_id], t.fourcc


@pytest.mark.parametrize('name', ['0026', '0025', '0005', '0002', '0007', 'sd0009'])
def test_fast_reader_equals_pyav(name):
    p = need(CLIPS[name])
    a = T.load_telemetry(p, cache_dir=None, reader='pyav')
    b = T.load_telemetry(p, cache_dir=None, reader='mp4')
    assert a.extra['meta_reader'] == 'pyav' and b.extra['meta_reader'] == 'mp4'
    assert telemetry_diff(a, b) == []
    if name in ('0005', 'sd0009'):          # OA4 4:3 with a dbgi track: also with every dbgi frame
        a = T.load_telemetry(p, cache_dir=None, reader='pyav', dbgi='full')
        b = T.load_telemetry(p, cache_dir=None, reader='mp4', dbgi='full')
        assert 'dbgi_frames' not in b.extra and telemetry_diff(a, b) == []


# ------------------------------------------------------------------------------------------ vs the legacy parser


def legacy_reference(p):
    """The npz cache written by the pre-fast-path parser (v7, 1 MiB-buffered seek+read of every djmd/dbgi sample),
    if this machine has one (work/cache); caches written by the new parser carry extra['meta_reader']."""
    cp = T.cache_file(p, os.path.join(ROOT, 'work', 'cache'))
    if not os.path.exists(cp):
        pytest.skip(f'no legacy telemetry cache for {os.path.basename(p)}')
    ref = T._load(cp)
    if 'meta_reader' in ref.extra:
        pytest.skip('cache was written by the new parser')
    return ref


@pytest.mark.parametrize('name', ['0026', '0025', '0005', '0002', '0007'])
def test_matches_legacy_parser(name):
    """Bit-identical to what the old implementation produced: everything with dbgi='full'; with the default dbgi
    policy everything too, except that the dbgi_* extras of EIS-off clips cover only extra['dbgi_frames']."""
    p = need(CLIPS[name])
    ref = legacy_reference(p)
    full = T.load_telemetry(p, cache_dir=None, dbgi='full')
    assert telemetry_diff(ref, full) == []
    auto = T.load_telemetry(p, cache_dir=None)
    assert telemetry_diff(ref, auto, dbgi_subset='dbgi_frames' in auto.extra) == []


def test_dbgi_policy():
    p5, p7 = need(CLIPS['0005']), need(CLIPS['0007'])
    t5 = T.load_telemetry(p5, cache_dir=None)                    # EIS off: 8 frames at the start, middle, end
    fr = t5.extra['dbgi_frames']
    assert len(fr) == 24 and fr[0] == 0 and fr[-1] == t5.n_frames - 1
    assert t5.extra['dbgi_eis_mode'].shape == (24,) and np.all(t5.extra['dbgi_eis_mode'] == 0)
    assert t5.extra['dbgi_q_phys_cam'].shape == (24, 4) and len(t5.extra['dbgi_q_phys_t']) == 24
    t7 = T.load_telemetry(p7, cache_dir=None)                    # EIS on (header): every frame
    assert 'dbgi_frames' not in t7.extra and t7.extra['dbgi_eis_mode'].shape == (371,) and t7.eis_baked
    # the cache key separates dbgi='full' from the default; the default key is the legacy (v7) one
    k_auto, k_full = T.cache_file(p5, '/x'), T.cache_file(p5, '/x', dbgi='full')
    assert k_auto != k_full


def test_reader_and_dbgi_switches(monkeypatch):
    p = need(CLIPS['0026'])
    monkeypatch.setenv(T.READER_ENV, 'pyav')
    assert T.load_telemetry(p, cache_dir=None).extra['meta_reader'] == 'pyav'
    assert T.load_telemetry(p, cache_dir=None, reader='mp4').extra['meta_reader'] == 'mp4'   # argument wins
    monkeypatch.setenv(T.READER_ENV, 'auto')
    assert T.load_telemetry(p, cache_dir=None).extra['meta_reader'] == 'mp4'
    monkeypatch.setenv(T.READER_ENV, 'bogus')
    with pytest.raises(ValueError):
        T.load_telemetry(p, cache_dir=None)
    monkeypatch.delenv(T.READER_ENV)
    monkeypatch.setenv(T.DBGI_ENV, 'full')
    assert T.cache_file(p, '/x') == T.cache_file(p, '/x', dbgi='full')


# ------------------------------------------------------------------------------------------ quick look / probe


@pytest.mark.parametrize('name', ['0025', '0005', '0002', '0007'])
def test_probe_telemetry_matches_full(name):
    p = need(CLIPS[name])
    full = T.load_telemetry(p, cache_dir=None)
    q = T.probe_telemetry(p)
    n = min(T.QUICK_FRAMES, full.n_frames)
    assert q.n_frames == n and q.extra['quick']['frames'] == n and q.extra['quick']['n_frames'] == full.n_frames
    assert q.extra['quick']['discontinuities'] == 0
    assert (q.camera, q.width, q.height, q.has_highrate, q.eis_baked) == \
           (full.camera, full.width, full.height, full.has_highrate, full.eis_baked)
    assert (q.gravity_q is None) == (full.gravity_q is None)
    assert q.lens.fx == full.lens.fx and np.array_equal(q.lens.k, full.lens.k)
    assert abs(q.imu_rate / full.imu_rate - 1) < 1e-4 and abs(q.readout_s / full.readout_s - 1) < 1e-4
    np.testing.assert_array_equal(q.frame_pts, full.frame_pts[:n])
    np.testing.assert_allclose(q.frame_t, full.frame_t[:n], atol=50e-6)


def test_bridge_probe_quick_vs_full(monkeypatch, tmp_path):
    """app_bridge probe on a clip longer than PROBE_FULL_MAX_FRAMES and not cached takes the quick look; the fields
    the app shows equal a full parse's."""
    from stillpoint import app_bridge as ab
    p = need(CLIPS['0025'])
    monkeypatch.setattr(ab, 'ROOT', str(tmp_path))               # empty telemetry cache
    res = {}
    for mode in ('quick', 'full', 'auto'):
        res[mode] = ab.cmd_probe(types.SimpleNamespace(clip=p, telemetry=mode))
    assert res['quick']['telemetry_scope'] == 'quick' and res['full']['telemetry_scope'] == 'full'
    assert res['auto']['telemetry_scope'] == 'cached'           # the full probe cached it
    for k in ('camera', 'product', 'has_highrate', 'eis_baked', 'eis_status', 'segments', 'lens_fx', 'gyro',
              'supported', 'horizon_lock_supported', 'fov', 'n_frames', 'duration_s'):
        assert res['quick'][k] == res['full'][k], k
    assert abs(res['quick']['imu_rate'] / res['full']['imu_rate'] - 1) < 1e-4


# ------------------------------------------------------------------------------------------ the long SD-card clip


def test_sd_long_clip_fast_path():
    """6.2 GiB, 8-minute OA4 clip on the SD card: quick look < 3 s; the full parse reads only the metadata samples
    (one ~0.4 ms random read per frame on that card, ~12 s; the old buffered reader took minutes) and equals the
    legacy parser's cached result when this machine has one. Never PyAV-demuxed (that would read far more)."""
    p = need(SD_LONG)
    t0 = time.perf_counter()
    q = T.probe_telemetry(p)
    t_quick = time.perf_counter() - t0
    assert q.has_highrate and abs(q.imu_rate - 1000) < 1 and not q.eis_baked
    assert t_quick < 3.0, t_quick
    t0 = time.perf_counter()
    full = T.load_telemetry(p, cache_dir=None)
    t_full = time.perf_counter() - t0
    print(f'\n{os.path.basename(p)}: quick look {t_quick:.2f} s, full telemetry {t_full:.1f} s '
          f'({full.n_frames} frames, {len(full.imu_t)} IMU samples)')
    assert full.n_frames == 23950 and full.has_highrate and abs(full.imu_rate - 1000) < 1
    assert t_full < 60.0, t_full
    cp = T.cache_file(p, os.path.join(ROOT, 'work', 'cache'))
    if os.path.exists(cp):
        ref = T._load(cp)
        if 'meta_reader' not in ref.extra:
            assert telemetry_diff(ref, full, dbgi_subset=True) == []
