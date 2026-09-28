"""WP-A video tests (probe, analysis decode, frame-exact seeking). Run:
    cd stillpoint && PYTHONPATH=engine .venv/bin/python -m pytest tests/test_video.py -q
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from stillpoint import video
from eval.footage import o3, oa4  # noqa: E402  (env-configurable, see eval/footage.py)

HOME = os.path.expanduser('~')
O3_26 = o3('DJI_0026.MP4')
O3_JOINED = o3('DJI_0025_joined.MP4')
OA4_05 = oa4('DJI_20260926153751_0005_D.MP4')
OA4_02 = oa4('DJI_20260926152149_0002_D.MP4')


def need(p):
    if not os.path.exists(p):
        pytest.skip(f'missing {p}')
    return p


def ffprobe_packet_pts(p):
    out = subprocess.run([video.FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts',
                          '-of', 'csv=p=0', p], capture_output=True, text=True, check=True).stdout.split()
    return np.sort(np.array([int(x) for x in out if x not in ('', 'N/A')]))


_FULL: dict = {}


def full_decode(p, n=None):
    key = (p, n)
    if key not in _FULL:
        _FULL[key] = list(video.iter_gray(p, 960, 0, n))
    return _FULL[key]


def test_probe_o3():
    p = need(O3_26)
    info = video.probe(p)
    assert (info['width'], info['height'], info['codec'], info['pix_fmt'], info['bit_depth']) == \
           (3840, 2160, 'h264', 'yuv420p', 8)
    assert info['n_frames'] == 268 and info['fps_fraction'] == '60000/1001' and info['cfr']
    assert info['color_primaries'] == 'bt709' and info['color_transfer'] == 'bt709' and info['color_range'] == 'tv'
    assert info['pts_source'] == 'mp4_stbl' and info['timescale'] == 60000
    np.testing.assert_array_equal(info['frame_pts_ticks'], ffprobe_packet_pts(p))   # exact, per frame
    np.testing.assert_array_equal(info['frame_pts_ticks'], np.arange(268) * 1001)
    assert list(info['keyframes'][:4]) == [0, 30, 60, 90]


def test_probe_oa4_and_joined():
    p = need(OA4_05)
    info = video.probe(p)
    assert (info['width'], info['height'], info['codec'], info['pix_fmt'], info['bit_depth']) == \
           (3840, 2880, 'hevc', 'yuv420p10le', 10)
    np.testing.assert_array_equal(info['frame_pts_ticks'], ffprobe_packet_pts(p))
    assert info['n_frames'] == 519
    if os.path.exists(O3_JOINED):            # 21 GB: sample tables only, no packet reads
        j = video.probe(O3_JOINED)
        assert j['n_frames'] == 66500 and j['cfr']
        assert abs(j['frame_pts'][-1] - 66499 * 1001 / 60000) < 1e-9


def test_iter_gray_full_count_and_pts():
    p = need(O3_26)
    info = video.probe(p)
    fr = full_decode(p)
    assert len(fr) == info['n_frames'] == 268                     # no dup / drop (-fps_mode passthrough)
    assert [f[0] for f in fr] == list(range(268))
    np.testing.assert_array_equal([f[1] for f in fr], info['frame_pts'])
    g = fr[100][2]
    assert g.shape == (540, 960) and g.dtype == np.uint8 and 20 < g.mean() < 235
    # consecutive frames differ (not duplicated)
    assert all(np.any(fr[i][2] != fr[i + 1][2]) for i in range(0, 267, 7))


@pytest.mark.parametrize('k', [1, 29, 30, 31, 137, 250])
def test_seek_is_frame_exact_o3(k):
    p = need(O3_26)
    fr = full_decode(p)
    got = list(video.iter_gray(p, 960, start_frame=k, n_frames=3))
    assert [g[0] for g in got] == [k, k + 1, k + 2]
    for j, (_i, _t, g) in enumerate(got):
        assert np.array_equal(g, fr[k + j][2]), f'start_frame={k}: frame {k + j} differs from the full decode'
    # and it is not a neighbour (sanity: neighbours are measurably different)
    assert np.abs(got[0][2].astype(int) - fr[k - 1][2]).mean() > 0.2


@pytest.mark.parametrize('k', [29, 30, 45])
def test_seek_is_frame_exact_oa4_hevc10(k):
    p = need(OA4_05)
    fr = full_decode(p, 60)
    got = list(video.iter_gray(p, 960, start_frame=k, n_frames=2))
    assert got[0][2].shape == (720, 960)
    assert [g[0] for g in got] == [k, k + 1]
    assert all(np.array_equal(g[2], fr[g[0]][2]) for g in got)


def test_deep_seek_long_hevc():
    """Seek deep into the 5.4 GB 16:9 HEVC clip == sequential decode from the previous keyframe."""
    p = need(OA4_02)
    info = video.probe(p)
    kf = info['keyframes']
    k0 = int(kf[np.searchsorted(kf, 20000) - 1])
    seq = list(video.iter_gray(p, 480, start_frame=k0, n_frames=20000 - k0 + 2))
    got = list(video.iter_gray(p, 480, start_frame=20000, n_frames=2))
    assert [g[0] for g in got] == [20000, 20001]
    assert np.array_equal(got[0][2], seq[20000 - k0][2]) and np.array_equal(got[1][2], seq[20001 - k0][2])


def test_read_gray_frames_random_access():
    p = need(O3_26)
    fr = full_decode(p)
    idx = np.array([200, 3, 3, 150, 31, 267, 0, 120])
    out = video.read_gray_frames(p, idx)
    assert out.shape == (len(idx), 540, 960)
    for i, k in enumerate(idx):
        assert np.array_equal(out[i], fr[k][2])


def test_misalignment_is_detected():
    """If ffmpeg ever delivers a different frame than expected, iteration must raise, not shift indices."""
    p = need(O3_26)
    info = dict(video.probe(p))
    info['frame_pts'] = info['frame_pts'] + 1001 / 60000          # pretend every frame is one later
    with pytest.raises(video.DecodeError, match='misalignment'):
        list(video._run_decode(p, info, 960, 540, 'vt_scale_vt', 0, 2, True))


@pytest.mark.parametrize('mode', ['vt_cpu_scale', 'sw'])
def test_fallback_modes_align(mode):
    """Fallback chains (CPU scale; software decode) deliver the same frames (they differ from scale_vt only by
    the resampling filter, so compare after a small blur: the same frame must match far better than neighbours)."""
    import cv2
    p = need(OA4_05)
    info = video.probe(p)
    ref = full_decode(p, 60)
    got = list(video._run_decode(p, info, 960, 720, mode, 30, 2, True))
    assert [g[0] for g in got] == [30, 31]
    B = lambda x: cv2.GaussianBlur(x.astype(np.float32), (0, 0), 2.0)
    for k, _t, g in got:
        d = [float(np.abs(B(g) - B(ref[j][2])).mean()) for j in (k - 1, k, k + 1)]
        assert d[1] < 0.6 and d[1] < 0.4 * min(d[0], d[2]), d
