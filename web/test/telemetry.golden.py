"""Golden fixtures for the web TELEMETRY module (web/src/mp4.ts, dji.ts, telemetry.ts).

Owner: TELEMETRY agent (web). Dumps what the Python engine (engine/stillpoint/telemetry.py + video.py) says about a
clip so web/test/telemetry.test.ts can check the TypeScript port against it on the real files.

    cd stillpoint && PYTHONPATH=engine .venv/bin/python web/test/telemetry.golden.py [name ...] [--full DIR]

Writes web/test/fixtures/telemetry/<name>.json.gz (small: the first HEAD frames and the IMU samples that cover them,
plus ~STRIDED frames / IMU samples spread over the whole clip, scalars, flags, lens, warnings, timing extras and the
MP4 sample-table summary of every track). Arrays are base64 little-endian float64 (bit exact, NaN-safe).
--full DIR additionally writes the COMPLETE arrays (frame_pts, frame_t, exposure_s, imu_t, imu_q, gravity_q) as raw
float64 to DIR/<name>.full.bin + DIR/<name>.full.json (off-repo; the test compares every sample when
STILLPOINT_TELEMETRY_FULL=DIR is set).

Clips missing on this machine (or an unmounted SD card) are skipped. Read-only: parses with cache_dir=None.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'engine'))
sys.path.insert(0, ROOT)

from stillpoint import telemetry as T  # noqa: E402
from stillpoint import video  # noqa: E402

HOME = os.path.expanduser('~')
SD = os.environ.get('STILLPOINT_SD_DIR', '/Volumes/Untitled/DCIM/DJI_001')
O3 = os.environ.get('STILLPOINT_O3_DIR', os.path.join(HOME, 'Desktop', 'untitled folder 4'))
DESK = os.path.join(HOME, 'Desktop')


def _first(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


CLIPS = {
    'o3_0026': lambda: os.path.join(O3, 'DJI_0026.MP4'),                                   # O3, 2 kHz, H.264
    'oa4_0005': lambda: _first(os.path.join(DESK, 'DJI_20260926153751_0005_D.MP4'),       # OA4 4:3, 1 kHz, 1/61 s
                               os.path.join(SD, 'DJI_20260926153751_0005_D.MP4')),
    'oa4_0002': lambda: _first(os.path.join(DESK, 'DJI_20260926152149_0002_D.MP4'),       # OA4 16:9, per-frame, 5 GB
                               os.path.join(SD, 'DJI_20260926152149_0002_D.MP4')),
    'oa4_0007': lambda: _first(os.path.join(DESK, 'DJI_20260926155953_0007_D.MP4'),       # OA4, EIS on
                               os.path.join(SD, 'DJI_20260926155953_0007_D.MP4')),
    'oa4_0012': lambda: _first(os.path.join(DESK, 'DJI_20260927091931_0012_D.MP4'),       # OA4 1 kHz, 6.2 GiB, AUTO
                               os.path.join(SD, 'DJI_20260927091931_0012_D.MP4')),
    'o4p_0003': lambda: os.path.join(DESK, 'DJI_20260925151130_0003_D.MP4'),              # O4 Pro (bonus)
    'o4p_0003_joined': lambda: os.path.join(DESK, 'DJI_20260925151130_0003_D_joined.MP4'),  # joined file (segments)
}
DEFAULT = ['o3_0026', 'oa4_0005', 'oa4_0002', 'oa4_0007', 'oa4_0012', 'o4p_0003', 'o4p_0003_joined']
HEAD = 600          # frames dumped completely (with their IMU samples)
HEAD_BONUS = {'o4p_0003': 240, 'o4p_0003_joined': 240}   # keep the fixtures small (< 5 MB in total)
QUICK = {'oa4_0012', 'o4p_0003_joined'}                  # also dump probe_telemetry (quick look) for these
STRIDED = 400       # frames spread over the whole clip
STRIDED_IMU = 3000  # IMU samples spread over the whole clip
TRACK_SAMPLES = 40  # first sample-table entries per track + as many strided


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, '<f8')).tobytes()).decode()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return None       # arrays are dumped explicitly
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def strided_idx(n: int, m: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, np.int64)
    return np.unique(np.linspace(0, n - 1, min(m, n)).round().astype(np.int64))


def track_summary(tr: video.Mp4Track) -> dict:
    n = tr.n_samples
    idx = np.unique(np.concatenate([np.arange(min(n, TRACK_SAMPLES)), strided_idx(n, TRACK_SAMPLES)])).astype(np.int64)
    pts = tr.pts_ticks()
    d = dict(id=int(tr.track_id), handler=tr.handler, handler_name=tr.handler_name, fourcc=tr.fourcc,
             timescale=int(tr.timescale), width=int(tr.width), height=int(tr.height), n=int(n),
             movie_timescale=int(tr.movie_timescale), elst=[list(map(float, e)) for e in tr.elst])
    if n:
        d.update(sum_sizes=int(tr.sizes.sum()), sum_pts_ticks=int(pts.sum()), sum_dts_ticks=int(tr.dts.sum()),
                 idx=idx.tolist(), sizes=tr.sizes[idx].tolist(), pts_ticks=pts[idx].tolist(),
                 dts_ticks=tr.dts[idx].tolist(),
                 n_sync=int(n if tr.sync is None else tr.sync.sum()),
                 sync_first=(list(range(min(n, 8))) if tr.sync is None else np.flatnonzero(tr.sync)[:8].tolist()))
        if len(tr.offsets) == n:
            d.update(sum_offsets=int(tr.offsets.sum()), offsets=tr.offsets[idx].tolist(),
                     max_offset=int(tr.offsets.max()))
    return d


def dump(name: str, path: str, out_dir: str, full_dir: str | None) -> dict:
    t0 = time.time()
    tel = T.load_telemetry(path, cache_dir=None)
    t_parse = time.time() - t0
    info = video.probe(path)
    F = tel.n_frames
    N = len(tel.imu_t)
    h = min(HEAD_BONUS.get(name, HEAD), F)
    t_end = tel.frame_t[h - 1] + 0.5 * tel.readout_s + 0.02
    n_imu_head = int(np.searchsorted(tel.imu_t, t_end, side='right'))
    n_imu_head = max(1, min(N, n_imu_head))
    fi = strided_idx(F, STRIDED)
    ii = strided_idx(N, STRIDED_IMU)
    grav = 'none' if tel.gravity_q is None else ('same' if tel.gravity_q is tel.imu_q else 'array')
    ex = tel.extra
    extras = {k: jsonable(ex[k]) for k in ('product', 'proto_file', 'firmware', 'lens_source', 'readout_source',
                                           'readout_raw_s', 'read_direction', 'eis_status', 'eis_status_name',
                                           'format_comment', 'sensor_fps_meta', 'imu_rate_nominal', 'fov_type',
                                           'clip_bounds', 'segment_time_maps', 'timing', 'imu_grid',
                                           'imu_full_coverage_frames', 'gravity_W_deg', 'gravity_W_spread_deg',
                                           'gravity_note', 'imu_source', 'eis_note', 'parser_version')
              if k in ex}
    if 'gravity_W' in ex:
        extras['gravity_W'] = [float(v) for v in ex['gravity_W']]
    if 'dbgi_eis_mode' in ex:
        m = np.asarray(ex['dbgi_eis_mode'])
        extras['dbgi_eis_mode_max'] = int(m.max()) if len(m) else -1
        extras['dbgi_eis_mode_n'] = int(len(m))
    tracks = video.mp4_tracks(path)
    vt = video.main_video_track(tracks)
    L = tel.lens
    fx = dict(
        name=name, file=os.path.basename(path), size=os.path.getsize(path), python_parse_s=round(t_parse, 3),
        camera=tel.camera, width=int(tel.width), height=int(tel.height), fps=float(tel.fps),
        readout_s=float(tel.readout_s), imu_rate=float(tel.imu_rate), has_highrate=bool(tel.has_highrate),
        eis_baked=bool(tel.eis_baked), gravity=grav,
        lens=dict(model=L.model, fx=float(L.fx), fy=float(L.fy), cx=float(L.cx), cy=float(L.cy),
                  k=[float(v) for v in L.k], width=int(L.width), height=int(L.height)),
        segments=[list(map(int, s)) for s in tel.segments], warnings=list(ex.get('warnings', [])),
        n_frames=int(F), n_imu=int(N), extras=extras,
        probe=dict(width=int(info['width']), height=int(info['height']), codec=info['codec'], fps=float(info['fps']),
                   fps_fraction=info['fps_fraction'], pix_fmt=info['pix_fmt'], color_range=info['color_range'],
                   color_space=info['color_space'], color_primaries=info['color_primaries'],
                   color_transfer=info['color_transfer'], comment=str(info['format_tags'].get('comment', '')),
                   encoder=str(info['format_tags'].get('encoder', '')), duration=float(info['duration']),
                   n_frames=int(info['n_frames']), main_video_track_id=int(vt.track_id),
                   n_keyframes=int(len(info['keyframes'])), bit_depth=int(info['bit_depth'])),
        tracks=[track_summary(t) for t in tracks],
        head=dict(frames=int(h), n_imu=int(n_imu_head), frame_pts=b64(tel.frame_pts[:h]), frame_t=b64(tel.frame_t[:h]),
                  exposure_s=b64(tel.exposure_s[:h]), imu_t=b64(tel.imu_t[:n_imu_head]),
                  imu_q=b64(tel.imu_q[:n_imu_head])),
        strided=dict(frame_idx=fi.tolist(), frame_pts=b64(tel.frame_pts[fi]), frame_t=b64(tel.frame_t[fi]),
                     exposure_s=b64(tel.exposure_s[fi]), imu_idx=ii.tolist(), imu_t=b64(tel.imu_t[ii]),
                     imu_q=b64(tel.imu_q[ii])),
    )
    if grav == 'array':
        fx['head']['gravity_q'] = b64(tel.gravity_q[:n_imu_head])
        fx['strided']['gravity_q'] = b64(tel.gravity_q[ii])
    if name in QUICK:
        q = T.probe_telemetry(path)
        qi = strided_idx(len(q.imu_t), 300)
        fx['quick'] = dict(n_frames=int(q.n_frames), n_imu=int(len(q.imu_t)), warnings=list(q.extra.get('warnings', [])),
                           quick=jsonable(q.extra['quick']), imu_rate=float(q.imu_rate), readout_s=float(q.readout_s),
                           eis_baked=bool(q.eis_baked), has_highrate=bool(q.has_highrate),
                           segments=[list(map(int, s_)) for s_ in q.segments],
                           frame_t=b64(q.frame_t), imu_idx=qi.tolist(), imu_t=b64(q.imu_t[qi]), imu_q=b64(q.imu_q[qi]))
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f'{name}.json.gz')
    with gzip.open(p, 'wt', encoding='utf-8') as f:
        json.dump(fx, f, separators=(',', ':'))
    res = dict(name=name, fixture=p, bytes=os.path.getsize(p), parse_s=round(t_parse, 2), frames=F, imu=N)
    if full_dir:
        os.makedirs(full_dir, exist_ok=True)
        arrs = [('frame_pts', tel.frame_pts), ('frame_t', tel.frame_t), ('exposure_s', tel.exposure_s),
                ('imu_t', tel.imu_t), ('imu_q', tel.imu_q.reshape(-1))]
        if grav == 'array':
            arrs.append(('gravity_q', tel.gravity_q.reshape(-1)))
        layout, off = {}, 0
        with open(os.path.join(full_dir, f'{name}.full.bin'), 'wb') as f:
            for k, a in arrs:
                b = np.ascontiguousarray(np.asarray(a, '<f8')).tobytes()
                f.write(b)
                layout[k] = [off, len(a)]
                off += len(b)
        with open(os.path.join(full_dir, f'{name}.full.json'), 'w') as f:
            json.dump(dict(name=name, file=path, layout=layout), f)
        res['full_bytes'] = off
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*', default=DEFAULT)
    ap.add_argument('--out', default=os.path.join(HERE, 'fixtures', 'telemetry'))
    ap.add_argument('--full', default=None, help='also dump complete arrays here (off-repo)')
    a = ap.parse_args()
    for n in a.names:
        path = CLIPS[n]()
        if not os.path.exists(path):
            print(json.dumps(dict(name=n, skipped=f'missing {path}')), flush=True)
            continue
        print(json.dumps(dump(n, path, a.out, a.full)), flush=True)


if __name__ == '__main__':
    main()
