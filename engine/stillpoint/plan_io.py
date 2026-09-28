"""Read/write the .spplan render-plan file consumed by the Metal renderer (ENGINE_SPEC.md §3).

Little-endian. 256-byte header:
  off  type        field
    0  char[8]     magic "SPPLAN01"
    8  u32         version (=1)
   12  u32         header_bytes (=256)
   16  u32 x4      src_w, src_h, out_w, out_h
   32  u32         n_frames
   36  u32         n_rows            (row samples per frame; row j is at y = j*(src_h-1)/(n_rows-1))
   40  u32         lens_model        (0 = pinhole, 1 = kb4 fisheye)
   44  u32         record_bytes      (= 24 + 36*n_rows)
   48  f32 x8      fx, fy, cx, cy, k1, k2, k3, k4   (source lens, pixel-centre convention)
   80  f32         readout_s         (informational)
   84  u32         flags             (reserved, 0)
   88  ...         zero padding to 256
Then n_frames records of record_bytes each:
    0  f64         pts_s             source frame PTS this output frame is rendered from
    8  f32         out_fx
   12  f32         out_cx
   16  f32         out_cy
   20  f32         pad
   24  f32[n_rows][3][3]  row matrices, row-major, M @ r_virtual -> r_source
"""
from __future__ import annotations

import struct

import numpy as np

from .geom import Lens
from .types import Plan

MAGIC = b"SPPLAN01"
HEADER_BYTES = 256
_LENS_CODES = {'pinhole': 0, 'kb4': 1}
_LENS_NAMES = {v: k for k, v in _LENS_CODES.items()}


def record_dtype(n_rows: int) -> np.dtype:
    return np.dtype([('pts', '<f8'), ('out_fx', '<f4'), ('out_cx', '<f4'), ('out_cy', '<f4'), ('pad', '<f4'),
                     ('mats', '<f4', (n_rows, 3, 3))])


def write_plan(path: str, plan: Plan) -> None:
    F, R = plan.n_frames, plan.n_rows
    L = plan.lens
    hdr = bytearray(HEADER_BYTES)
    struct.pack_into('<8sII', hdr, 0, MAGIC, 1, HEADER_BYTES)
    struct.pack_into('<IIII', hdr, 16, plan.src_w, plan.src_h, plan.out_w, plan.out_h)
    struct.pack_into('<IIII', hdr, 32, F, R, _LENS_CODES[L.model], 24 + 36 * R)
    struct.pack_into('<8f', hdr, 48, L.fx, L.fy, L.cx, L.cy, *[float(c) for c in L.k])
    struct.pack_into('<fI', hdr, 80, float(plan.meta.get('readout_s', 0.0)), 0)
    rec = np.zeros(F, dtype=record_dtype(R))
    rec['pts'] = plan.frame_pts
    rec['out_fx'] = plan.out_fx
    rec['out_cx'] = (plan.out_w - 1) / 2.0
    rec['out_cy'] = (plan.out_h - 1) / 2.0
    rec['mats'] = plan.row_mats.astype(np.float32)
    assert rec.dtype.itemsize == 24 + 36 * R
    with open(path, 'wb') as f:
        f.write(bytes(hdr))
        f.write(rec.tobytes())


def read_plan(path: str) -> Plan:
    with open(path, 'rb') as f:
        hdr = f.read(HEADER_BYTES)
        magic, ver, hb = struct.unpack_from('<8sII', hdr, 0)
        if magic != MAGIC or ver != 1:
            raise ValueError(f"not a v1 .spplan file: {path}")
        src_w, src_h, out_w, out_h = struct.unpack_from('<IIII', hdr, 16)
        F, R, lens_code, rb = struct.unpack_from('<IIII', hdr, 32)
        fx, fy, cx, cy, k1, k2, k3, k4 = struct.unpack_from('<8f', hdr, 48)
        readout, _flags = struct.unpack_from('<fI', hdr, 80)
        f.seek(hb)
        rec = np.frombuffer(f.read(F * rb), dtype=record_dtype(R), count=F)
    lens = Lens(_LENS_NAMES[lens_code], fx, fy, cx, cy, np.array([k1, k2, k3, k4]), src_w, src_h)
    return Plan(src_w=src_w, src_h=src_h, out_w=out_w, out_h=out_h, lens=lens,
                frame_pts=rec['pts'].astype(np.float64), out_fx=rec['out_fx'].astype(np.float64),
                row_mats=rec['mats'].astype(np.float64), virt_q=np.zeros((F, 4)),
                meta={'readout_s': readout})
