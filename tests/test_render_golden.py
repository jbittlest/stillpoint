"""WP-B golden + throughput tests: sprender (Swift/Metal) vs render_ref (float64 numpy) on real footage.

(b) DJI_0026 (O3 4K H.264 8-bit) with a non-trivial plan (fast synthetic camera motion ~150 deg/s + RS 9.72 ms,
    low-passed virtual path + 2 deg roll, time-model offset/focal/extrinsic, 1.25x zoom, some frames partly
    outside the source): sprender --dump-frames (pre-encode planes + the kernel's coordinate map) vs render_ref.
    PSNR > 45 dB, mean |source-coordinate discrepancy| < 0.02 px; dumped source == independent PyAV decode of
    the same frame index; encoded output keeps the source PTS / tags / frame count.
(d) throughput of sprender on DJI_0026 (all 268 frames — the clip has no 300) and on a 300-frame window of
    DJI_0025.
Jimmy's footage is only read. Outputs go to pytest's tmp dir.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from stillpoint.geom import Lens, qexp, qlog, qmul
from stillpoint.plan_build import build_plan, camera_orientation_fn
from stillpoint.plan_io import read_plan, write_plan
from stillpoint.render_ref import render_frames, render_planes_ref, source_map
from stillpoint.types import Telemetry, TimeModel
from eval.footage import O3_DIR, oa4  # noqa: E402  (env-configurable, see eval/footage.py)

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
SPRENDER = os.path.join(ROOT, 'app', 'renderer', '.build', 'sprender')
O3 = O3_DIR
V26 = os.path.join(O3, 'DJI_0026.MP4')
V25 = os.path.join(O3, 'DJI_0025.MP4')
OA4 = oa4('DJI_20260926153751_0005_D.MP4')   # 3840x2880 HEVC Main10 + AAC

pytestmark = pytest.mark.skipif(not os.path.exists(V26), reason='O3 footage not available')


# ------------------------------------------------------------------------------------------ helpers
def ensure_sprender():
    src = os.path.join(ROOT, 'app', 'renderer', 'main.swift')
    if not os.path.exists(SPRENDER) or os.path.getmtime(SPRENDER) < os.path.getmtime(src):
        subprocess.run([os.path.join(ROOT, 'app', 'renderer', 'build.sh')], check=True, capture_output=True)
    return SPRENDER


def demux_pts(path):
    import av
    c = av.open(path)
    s = c.streams.video[0]
    p = sorted(float(pk.pts * s.time_base) for pk in c.demux(s) if pk.pts is not None)
    c.close()
    return np.array(p)


def o3_lens(W=3840, H=2160):
    return Lens('kb4', 1405.129, 1405.129, (W - 1) / 2, (H - 1) / 2,
                np.array([0.24991769, 0.01360575, -0.06208358, 0.01219307]), W, H)


def make_plan(pts, rate_dps=150.0, zoom=1.25, n_rows=32, source='synthetic', W=3840, H=2160, lens=None):
    lens = lens or o3_lens(W, H)
    t = np.arange(pts[0] - 1, pts[-1] + 1, 1 / 2000.0)
    w = np.deg2rad(rate_dps)
    rv = np.stack([0.05 * np.sin(2 * np.pi * 1.3 * t) + w / (2 * np.pi * 2.1) * np.sin(2 * np.pi * 2.1 * t),
                   0.08 * np.sin(2 * np.pi * 0.7 * t + 1) + w / (2 * np.pi * 3.3) * np.sin(2 * np.pi * 3.3 * t + 0.4),
                   0.03 * np.sin(2 * np.pi * 0.9 * t + 2) + 0.5 * w / (2 * np.pi * 5.0) * np.sin(2 * np.pi * 5.0 * t)], -1)
    tel = Telemetry(source=source, camera='synthetic', width=W, height=H, fps=60000 / 1001, frame_pts=pts,
                    frame_t=pts.copy(), exposure_s=np.full(len(pts), 1 / 500), readout_s=0.0097247, lens=lens,
                    imu_t=t, imu_q=qexp(rv), imu_rate=2000.0, has_highrate=True, eis_baked=False)
    tm = TimeModel(offset_s=0.0012, focal_scale=1.003, extrinsic_rotvec=np.array([0.002, -0.001, 0.003]))
    qf = camera_orientation_fn(tel, tm)
    rvs = uniform_filter1d(qlog(qf(tel.frame_t)), 31, axis=0, mode='nearest')
    virt = qmul(qexp(rvs), qexp(np.array([0, 0, np.deg2rad(2.0)])))
    return build_plan(tel, tm, qf, virt, np.full(len(pts), lens.fx * zoom), W, H, n_rows=n_rows)


def run_sprender(*a):
    r = subprocess.run([ensure_sprender(), *map(str, a), '--quiet'], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr + r.stdout
    m = re.search(r'RESULT (.*)', r.stdout)
    return dict(kv.split('=', 1) for kv in m.group(1).split())


def psnr10(a, b):
    d = (a.astype(np.float64) - b.astype(np.float64)) / 64.0
    mse = np.mean(d * d)
    return 99.0 if mse == 0 else 10 * np.log10(1023.0 ** 2 / mse)


def load_dump(d, n):
    m = json.load(open(f'{d}/frame{n}.json'))
    W, H, oW, oH = m['src_w'], m['src_h'], m['out_w'], m['out_h']
    dt = np.uint8 if m['src_bits'] == 8 else np.uint16
    return dict(meta=m,
                sy=np.fromfile(f'{d}/frame{n}_src_y.raw', dt).reshape(H, W),
                suv=np.fromfile(f'{d}/frame{n}_src_uv.raw', dt).reshape(H // 2, W // 2, 2),
                oy=np.fromfile(f'{d}/frame{n}_out_y.u16', np.uint16).reshape(oH, oW),
                ouv=np.fromfile(f'{d}/frame{n}_out_uv.u16', np.uint16).reshape(oH // 2, oW // 2, 2),
                cm=np.fromfile(f'{d}/frame{n}_coords.f32', np.float32).reshape(oH, oW, 3))


def decode_frames(path, want_pts, fmt='yuv420p'):
    """Independent software decode (PyAV) of the frames at the given PTS -> {pts: (Y, UV-interleaved)}."""
    import av
    out = {}
    c = av.open(path)
    s = c.streams.video[0]
    for fr in c.decode(s):
        t = float(fr.pts * s.time_base)
        hit = [p for p in want_pts if abs(p - t) < 1e-4]
        if hit:
            a = fr.to_ndarray(format=fmt)  # planar (H*3/2, W)
            H, W = fr.height, fr.width
            y = a[:H]
            u = a[H:H + H // 4].reshape(H // 2, W // 2)
            v = a[H + H // 4:].reshape(H // 2, W // 2)
            out[hit[0]] = (y, np.stack([u, v], -1))
        if len(out) == len(want_pts) or t > max(want_pts) + 0.1:
            break
    c.close()
    return out


@pytest.fixture(scope='module')
def plan26(tmp_path_factory):
    d = tmp_path_factory.mktemp('golden')
    pts = demux_pts(V26)
    plan = make_plan(pts, source=V26)
    path = str(d / 'p0026.spplan')
    write_plan(path, plan)
    return dict(dir=d, path=path, pts=pts, plan=read_plan(path))   # float32 matrices = what the renderer sees


# ------------------------------------------------------------------------------------------ (b) golden
@pytest.mark.parametrize('kernel,frames,window', [('lanczos3', (100, 150), (98, 55)), ('catmullrom', (150,), (149, 3))])
def test_golden_sprender_vs_reference(plan26, kernel, frames, window):
    d = plan26['dir'] / f'dump_{kernel}'
    out = plan26['dir'] / f'win_{kernel}.mov'
    res = run_sprender(V26, plan26['path'], out, '--start-frame', window[0], '--frames', window[1], '--kernel', kernel,
                       '--dump-frames', ','.join(map(str, frames)), '--dump-dir', d)
    assert int(res['frames']) == window[1] and int(res['skipped_no_plan']) == 0
    plan, pts = plan26['plan'], plan26['pts']
    ref_src = decode_frames(V26, [pts[n] for n in frames])
    for n in frames:
        D = load_dump(d, n)
        assert D['meta']['src_index'] == n and D['meta']['plan_record'] == n
        assert abs(D['meta']['pts'] - pts[n]) < 1e-6
        # the renderer decoded exactly source frame n (independent software H.264 decode is bit-exact)
        ry, ruv = ref_src[pts[n]]
        assert np.array_equal(D['sy'], ry) and np.array_equal(D['suv'], ruv)
        # geometry: the kernel's own source coordinates vs float64
        S, ok = source_map(plan, n, return_valid=True)
        okm = D['cm'][..., 2] > 0.5
        both = ok & okm
        cerr = np.abs(S - D['cm'][..., :2])[both]
        # pixels: pre-encode output planes vs float64 reference render of the same decoded source
        RY, RUV = render_planes_ref(plan, n, D['sy'], D['suv'], kernel=kernel)
        py, puv = psnr10(D['oy'], RY), psnr10(D['ouv'], RUV)
        print(f"\n[{kernel}] frame {n}: valid {ok.mean():.3f} (mismatch {(ok != okm).sum()} px); coord |d| mean "
              f"{cerr.mean():.2e} max {cerr.max():.2e} px; PSNR Y {py:.1f} dB, CbCr {puv:.1f} dB; "
              f"max |dY| {np.abs(D['oy'].astype(int) - RY.astype(int)).max() // 64} codes")
        assert cerr.mean() < 0.02 and cerr.max() < 0.05
        assert (ok != okm).mean() < 1e-4
        assert py > 45 and puv > 45


def test_encoded_output_timestamps_tags_quality(plan26):
    """The HEVC output of the lanczos3 window: frame count, SOURCE PTS kept, tags, decoded quality."""
    out = plan26['dir'] / 'win_lanczos3.mov'
    if not out.exists():
        pytest.skip('run with test_golden_sprender_vs_reference')
    pts = plan26['pts']
    info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                      'stream=codec_name,profile,pix_fmt,color_range,color_space,color_primaries,'
                                      'color_transfer,time_base', '-of', 'json', str(out)],
                                     capture_output=True, text=True).stdout)['streams'][0]
    assert info['codec_name'] == 'hevc' and info['profile'] == 'Main 10' and info['pix_fmt'] == 'yuv420p10le'
    assert info['color_range'] == 'tv' and info['color_primaries'] == 'bt709' and info['color_transfer'] == 'bt709'
    assert info['time_base'] == '1/60000'
    import av
    c = av.open(str(out))
    s = c.streams.video[0]
    frames = {}
    for fr in c.decode(s):
        t = float(fr.pts * s.time_base)
        frames[round(t, 6)] = fr
    opts = np.array(sorted(frames))
    assert len(opts) == 55
    assert np.allclose(opts, pts[98:153], atol=1e-6), 'output must keep the source PTS of the window'
    fr = frames[round(pts[100], 6)]
    a = fr.to_ndarray(format='yuv420p10le') if hasattr(fr, 'to_ndarray') else None
    y10 = a[:2160].astype(np.uint16) * 64
    c.close()
    D = load_dump(plan26['dir'] / 'dump_lanczos3', 100)
    p = psnr10(y10, D['oy'])
    print(f"\nencoded HEVC10 180 Mbit/s vs pre-encode (frame 100): Y PSNR {p:.1f} dB")
    assert p > 38


@pytest.mark.skipif(not os.path.exists(OA4), reason='OA4 clip not available')
def test_golden_10bit_hevc_with_audio(tmp_path):
    """10-bit path (x420 textures), 4:3 frame, audio passthrough trimmed to the window."""
    pts = demux_pts(OA4)
    lens = Lens('kb4', 1650.0, 1650.0, (3840 - 1) / 2, (2880 - 1) / 2, np.array([0.05, -0.01, 0.002, 0.0]), 3840, 2880)
    plan = make_plan(pts, source=OA4, W=3840, H=2880, lens=lens, zoom=1.15)
    path = str(tmp_path / 'oa4.spplan')
    write_plan(path, plan)
    plan = read_plan(path)
    out = tmp_path / 'oa4.mov'
    n = 130
    res = run_sprender(OA4, path, out, '--start-frame', 120, '--frames', 60, '--dump-frames', n, '--dump-dir', tmp_path)
    assert int(res['frames']) == 60 and res['in'] == '10bit'
    D = load_dump(tmp_path, n)
    ref = decode_frames(OA4, [pts[n]], fmt='yuv420p10le')[pts[n]]
    assert np.array_equal(D['sy'] >> 6, ref[0]) and np.array_equal(D['suv'] >> 6, ref[1])
    assert not np.any(D['sy'] & 63)
    S, ok = source_map(plan, n, return_valid=True)
    both = ok & (D['cm'][..., 2] > 0.5)
    cerr = np.abs(S - D['cm'][..., :2])[both]
    RY, RUV = render_planes_ref(plan, n, D['sy'], D['suv'], kernel='lanczos3')
    py, puv = psnr10(D['oy'], RY), psnr10(D['ouv'], RUV)
    info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,start_time,duration',
                                      '-of', 'json', str(out)], capture_output=True, text=True).stdout)['streams']
    aud = [s for s in info if s['codec_type'] == 'audio']
    vid = [s for s in info if s['codec_type'] == 'video'][0]
    print(f"\n[OA4 10-bit] frame {n}: coord |d| mean {cerr.mean():.2e} max {cerr.max():.2e} px; PSNR Y {py:.1f} "
          f"CbCr {puv:.1f} dB; video start {vid['start_time']} dur {vid['duration']}; audio "
          f"{[(a['start_time'], a['duration']) for a in aud]}")
    assert cerr.mean() < 0.02 and py > 45 and puv > 45
    # audio packets are passed through whole (AAC 21.3 ms); the writer's edit list trims them to the window
    pcm = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(out), '-map', '0:a', '-f', 's16le', '-ac', '1', '-ar', '48000',
                          '-'], capture_output=True).stdout
    a_dur = len(pcm) / 2 / 48000
    print(f"decoded (edit-list aware) audio {a_dur:.4f} s vs video window {60 * 1001 / 60000:.4f} s")
    assert len(aud) == 1 and abs(a_dur - 60 * 1001 / 60000) < 0.025
    assert abs(float(vid['start_time']) - pts[120]) < 1e-6


def test_preview_render_frames(plan26):
    """Closed-loop preview: speed (>= 1500 frames/min) and agreement with the float64 reference."""
    import time
    from stillpoint.render_ref import _luma_frames, preview_K, sample_ref
    plan = plan26['plan']
    t0 = time.time()
    got = {}
    n = 0
    for k, img in render_frames(plan, V26, np.arange(plan.n_frames), out_scale=0.25, as_float=True):
        n += 1
        if k in (40, 150):
            got[k] = img.copy()
    dt = time.time() - t0
    print(f"\npreview 960x540 gray: {n} frames in {dt:.1f} s = {60 * n / dt:.0f} frames/min")
    assert n == plan.n_frames and 60 * n / dt > 1500
    assert np.allclose(preview_K(plan, 40, 0.25)[0], [1405.129 * 1.25 * 1.0 * 0.25, 0, 479.5])
    lums = dict(_luma_frames(V26, plan.frame_pts[[40, 150]], 0.008))
    for i, k in enumerate((40, 150)):
        small = lums[i].reshape(540, 4, 960, 4).mean(axis=(1, 3))   # box filter, pixel-centre aligned
        S, ok = source_map(plan, k, out_scale=0.25, return_valid=True)
        s = 0.25 * (S + 0.5) - 0.5
        ref = sample_ref(small, s[..., 0], s[..., 1], 'catmullrom')
        ref[~ok] = 0
        d = np.abs(ref - got[k])
        print(f"preview frame {k} vs float64: mean |d| {d.mean():.2e} max {d.max():.3f} (8-bit levels)")
        assert d.mean() < 0.01 and d.max() < 0.5


# ------------------------------------------------------------------------------------------ (d) throughput
def test_throughput_dji0026(plan26):
    out = plan26['dir'] / 'full.mov'
    rows = []
    for codec in ('hevc10', 'hevc10-speed'):
        r = run_sprender(V26, plan26['path'], out, '--codec', codec)
        rows.append((codec, r))
        assert int(r['frames']) == 268
    r = run_sprender(V26, plan26['path'], out, '--no-write')
    rows.append(('gpu-only (no encode)', r))
    for name, r in rows:
        print(f"\nsprender DJI_0026 268 frames 4K lanczos3 [{name}]: {r['fps']} fps, GPU {r['gpu_ms_per_frame']} ms/frame")
    assert float(rows[0][1]['fps']) > 15


@pytest.mark.skipif(not os.path.exists(V25), reason='DJI_0025 not available')
def test_throughput_window_dji0025(tmp_path):
    pts = demux_pts(V25)
    plan = make_plan(pts, source=V25)
    path = str(tmp_path / 'p0025.spplan')
    write_plan(path, plan)
    out = tmp_path / 'w25.mov'
    r = run_sprender(V25, path, out, '--start-frame', 600, '--frames', 300)
    print(f"\nsprender DJI_0025 frames 600..899 4K lanczos3 hevc10: {r['fps']} fps, GPU {r['gpu_ms_per_frame']} ms/frame")
    assert int(r['frames']) == 300
    import av
    c = av.open(str(out))
    s = c.streams.video[0]
    opts = sorted(float(pk.pts * s.time_base) for pk in c.demux(s) if pk.pts is not None)
    c.close()
    assert len(opts) == 300 and np.allclose(opts, pts[600:900], atol=1e-6)
