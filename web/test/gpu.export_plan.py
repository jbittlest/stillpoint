"""Export a Stillpoint .spplan (Python engine) + the MP4 sample table the GPU golden test needs, as JSON.

Owner: GPU agent (web).  Run with the engine venv:

    PYTHONPATH=engine .venv/bin/python web/test/gpu.export_plan.py <clip.mp4> <plan.spplan> <out.json> \
        [--frames 45,77] [--auto 2]

Output (all numbers little-endian-agnostic JSON):
  { clip, srcW, srcH, outW, outH, nRows, readoutS, lens: {model, fx, fy, cx, cy, k[4], width, height},
    records: [{k, pts, outFx, rowMats: [nRows*9 float32 values, row-major]}],
    video: {codec (WebCodecs string), description (base64 avcC/hvcC payload), codedWidth, codedHeight, timescale,
            samples: [{i, offset, size, pts, key}]  -- every sample from the sync sample before each record's
                                                       frame up to that frame (decode order == presentation order
                                                       is asserted: DJI clips have no B-frames) } }
The browser harness (web/test/gpu.harness.ts) fetches those byte ranges from the dev server and decodes them with
WebCodecs; web/test/gpu.golden.py recomputes the float64 reference with engine/stillpoint/render_ref.py.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'engine')))

from stillpoint import plan_io, video  # noqa: E402


def _boxes(buf: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack('>I4s', buf[pos:pos + 8])
        hlen = 8
        if size == 1:
            size = struct.unpack('>Q', buf[pos + 8:pos + 16])[0]
            hlen = 16
        elif size == 0:
            size = end - pos
        if size < hlen:
            return
        yield typ, pos + hlen, min(pos + size, end)
        pos += size


def codec_config(path: str, track_id: int) -> tuple[str, bytes, str]:
    """(sample-entry fourcc, avcC/hvcC payload, WebCodecs codec string) of track `track_id`."""
    fsize = os.path.getsize(path)
    with open(path, 'rb') as f:
        moov = None
        for typ, p, e in video._iter_boxes(f, 0, fsize):
            if typ == b'moov':
                moov = (p, e)
                break
        f.seek(moov[0])
        mv = f.read(moov[1] - moov[0])
    for typ, p, e in _boxes(mv, 0, len(mv)):
        if typ != b'trak':
            continue
        tid = None
        stsd = None
        for t2, p2, e2 in _boxes(mv, p, e):
            if t2 == b'tkhd':
                ver = mv[p2]
                tid = struct.unpack('>I', mv[p2 + 20:p2 + 24] if ver == 1 else mv[p2 + 12:p2 + 16])[0]
            elif t2 == b'mdia':
                for t3, p3, e3 in _boxes(mv, p2, e2):
                    if t3 == b'minf':
                        for t4, p4, e4 in _boxes(mv, p3, e3):
                            if t4 == b'stbl':
                                for t5, p5, e5 in _boxes(mv, p4, e4):
                                    if t5 == b'stsd':
                                        stsd = (p5, e5)
        if tid != track_id or stsd is None:
            continue
        ent = stsd[0] + 8                      # full-box header (4) + entry_count (4)
        esize, efourcc = struct.unpack('>I4s', mv[ent:ent + 8])
        for t6, p6, e6 in _boxes(mv, ent + 8 + 78, ent + esize):
            if t6 in (b'avcC', b'hvcC'):
                cfg = mv[p6:e6]
                return efourcc.decode(), cfg, codec_string(efourcc.decode(), t6.decode(), cfg)
    raise ValueError(f'no avcC/hvcC for track {track_id}')


def codec_string(fourcc: str, kind: str, c: bytes) -> str:
    if kind == 'avcC':
        return f'{fourcc}.{c[1]:02x}{c[2]:02x}{c[3]:02x}'
    # hvcC: ISO/IEC 14496-15 Annex E codec string
    b = c[1]
    space, tier, prof = b >> 6, (b >> 5) & 1, b & 0x1F
    compat = struct.unpack('>I', c[2:6])[0]
    rev = int(f'{compat:032b}'[::-1], 2)
    cons = list(c[6:12])
    while cons and cons[-1] == 0:
        cons.pop()
    level = c[12]
    s = f"{fourcc}.{['', 'A', 'B', 'C'][space]}{prof}.{rev:x}.{'H' if tier else 'L'}{level}"
    if cons:
        s += '.' + '.'.join(f'{x:02X}' for x in cons)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clip')
    ap.add_argument('plan')
    ap.add_argument('out')
    ap.add_argument('--frames', default='', help='comma-separated plan record indices')
    ap.add_argument('--auto', type=int, default=2, help='also pick N records with the largest rolling-shutter spread')
    ap.add_argument('--max-k', type=int, default=150, help='auto-pick only among the first N records (short decode)')
    a = ap.parse_args()

    plan = plan_io.read_plan(a.plan)
    trs = video.mp4_tracks(a.clip)
    vt = video.main_video_track(trs)
    pts = vt.pts_seconds()
    assert np.all(vt.cts_off == 0), 'B-frames not supported by this exporter'
    sync = np.nonzero(vt.sync)[0] if vt.sync is not None else np.arange(vt.n_samples)

    ks = [int(x) for x in a.frames.split(',') if x.strip()]
    if a.auto:
        M = plan.row_mats[:a.max_k]
        spread = np.abs(M[:, -1] - M[:, 0]).reshape(len(M), -1).max(-1)
        spread[np.isin(np.arange(len(M)), sync)] = -1          # skip keyframes (want inter-coded frames)
        target = len(ks) + a.auto
        for k in np.argsort(-spread):
            if len(ks) >= target:
                break
            if int(k) not in ks and all(abs(int(k) - j) > 5 for j in ks):
                ks.append(int(k))
    ks = sorted(set(ks))

    fourcc, cfg, cstr = codec_config(a.clip, vt.track_id)
    need = set()
    records = []
    for k in ks:
        # source sample = the frame whose PTS matches the plan record (render rule: +-0.5 frame)
        i = int(np.argmin(np.abs(pts - plan.frame_pts[k])))
        assert abs(pts[i] - plan.frame_pts[k]) < 0.5 / 59.94, (k, i)
        s0 = int(sync[sync <= i].max())
        need.update(range(s0, i + 1))
        records.append({'k': k, 'sample': i, 'pts': float(plan.frame_pts[k]), 'outFx': float(plan.out_fx[k]),
                        'rowMats': [float(v) for v in np.asarray(plan.row_mats[k], np.float32).reshape(-1)]})
    idx = np.array(sorted(need), np.int64)
    offs, sizes = vt.sample_ranges(idx)
    samples = [{'i': int(i), 'offset': int(o), 'size': int(s), 'pts': float(pts[i]),
                'key': bool(vt.sync[i]) if vt.sync is not None else True} for i, o, s in zip(idx, offs, sizes)]
    L = plan.lens
    out = {
        'clip': os.path.basename(a.clip), 'plan': os.path.basename(a.plan),
        'srcW': plan.src_w, 'srcH': plan.src_h, 'outW': plan.out_w, 'outH': plan.out_h, 'nRows': plan.n_rows,
        'readoutS': float(plan.meta.get('readout_s', 0.0)),
        'lens': {'model': L.model, 'fx': float(L.fx), 'fy': float(L.fy), 'cx': float(L.cx), 'cy': float(L.cy),
                 'k': [float(x) for x in L.k], 'width': L.width, 'height': L.height},
        'records': records,
        'video': {'codec': cstr, 'fourcc': fourcc, 'description': base64.b64encode(cfg).decode(),
                  'codedWidth': vt.width, 'codedHeight': vt.height, 'timescale': vt.timescale, 'samples': samples},
    }
    with open(a.out, 'w') as f:
        json.dump(out, f)
    print(json.dumps({'out': a.out, 'records': ks, 'codec': cstr, 'samples': len(samples),
                      'bytes': int(sizes.sum())}))


if __name__ == '__main__':
    main()
