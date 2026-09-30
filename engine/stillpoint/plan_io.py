"""Read/write the .spplan render-plan file consumed by the Metal renderer (ENGINE_SPEC.md §3).

Little-endian. 256-byte header:
  off  type        field
    0  char[8]     magic "SPPLAN01"
    8  u32         version (=1; see "mesh extension" below)
   12  u32         header_bytes (=256)
   16  u32 x4      src_w, src_h, out_w, out_h
   32  u32         n_frames
   36  u32         n_rows            (row samples per frame; row j is at y = j*(src_h-1)/(n_rows-1))
   40  u32         lens_model        (0 = pinhole, 1 = kb4 fisheye)
   44  u32         record_bytes      (= 24 + 36*n_rows)
   48  f32 x8      fx, fy, cx, cy, k1, k2, k3, k4   (source lens, pixel-centre convention)
   80  f32         readout_s         (informational)
   84  u32         flags             bit 0: a FILL section follows the records (full-frame border fill);
                                    bit 1: a MESH block is present (mesh residual, see below); other bits 0.
                                    Renderers that do not know a section ignore it (v1 readers never look past the
                                    records): a fill plan then shows black where the overscan leaves the source, a
                                    mesh plan renders rotation-only.
   88  ...         zero padding to 256 (mesh extension fields when bit 1 is set, below)
Then n_frames records of record_bytes each:
    0  f64         pts_s             source frame PTS this output frame is rendered from
    8  f32         out_fx
   12  f32         out_cx
   16  f32         out_cy
   20  f32         pad
   24  f32[n_rows][3][3]  row matrices, row-major, M @ r_virtual -> r_source

FILL section (flags bit 0; engine/stillpoint/fill.py), at header_bytes + n_frames * record_bytes:
    0  char[8]     magic "SPFILL01"
    8  u32         version (=1)
   12  u32         header_bytes (=64)
   16  u32         n_frames (= plan n_frames)
   20  u32         K (source slots per record, 4)
   24  u32         record_bytes (= 16 + 64*K + 8*mesh_w*mesh_h)
   28  u32 x2      mesh_w, mesh_h    (parallax velocity grid; 0 = none)
   36  f32 x5      feather_main_px, feather_nb_px, sigma, fallback_weight, fallback_blur
   56  u32         max_offset        (max |source record - record|: the renderer's frame-ring radius)
   60  f32         parallax_tol_px   (0 = off; written as a zero pad before 2026-09-29 evening)
  per record:
    0  u32         n_src
    4  f32 x3      frac_fill, frac_uncovered, pad
   16  K x { i32 src_record, f32 weight, f32 gain, f32 pad, f32[3][3] G (row-major), f32 pad[3] }   (64 bytes each)
       G maps an OUTPUT ray of this record to the virtual ray of src_record: the source's composite rows are
       row_mats[src_record] @ G.
   16+64K  f32[mesh_h][mesh_w][2]  parallax velocity [full-res output px per frame of record offset]

MESH EXTENSION (optional; engine/stillpoint/mesh.py). Written only when a plan carries non-zero
mesh-residual offsets. The header version stays 1 and the v1 records are unchanged, so every v1 reader
(Stillpoint.app's PlanFile, older sprender builds) still opens the file and renders it rotation-only (they ignore
the flags and the trailing block); mesh-aware readers (sprender, render_ref, read_plan) apply the offsets. Plans
without a mesh are byte-identical to v1.
   84  u32         flags |= 2 (bit 1; bit 0 is the FILL section's)
   88  u32         mesh_nx           vertex columns (>= 2)
   92  u32         mesh_ny           vertex rows (>= 2)
   96  u64         mesh_offset       byte offset of the mesh block (after the records and, if present, the FILL
                                    section)
  104  u32         mesh_frame_bytes  (= 8*mesh_nx*mesh_ny)
  108  f32         mesh_clamp_px     (informational: the offset clamp, full-res output px)
Mesh block: n_frames x mesh_ny x mesh_nx x (f32 dx, f32 dy), full-res OUTPUT pixels. Output pixel X of record k is
displaced by the bilinear interpolation D_k(X) of its vertex offsets (vertex (i,j) at X = i*(out_w-1)/(nx-1),
Y = j*(out_h-1)/(ny-1), clamped outside the grid) BEFORE the rotation mapping (shaders/warp.metal sp_mesh_offset).
(Mesh plans written by the workstream-A branch before the 2026-09-29 merge used flags bit 0 for the mesh; that
bit now means FILL, so such plans must be regenerated.)
FILL and MESH are independent in the file format, but the engine does not produce both in one plan yet (their
renderer kernels are separate; pipeline.analyze refuses the combination).
"""
from __future__ import annotations

import struct

import numpy as np

from .geom import Lens
from .types import Plan

MAGIC = b"SPPLAN01"
HEADER_BYTES = 256
FILL_MAGIC = b"SPFILL01"
FILL_HEADER_BYTES = 64
FLAG_FILL = 1
FLAG_MESH = 2
_LENS_CODES = {'pinhole': 0, 'kb4': 1}
_LENS_NAMES = {v: k for k, v in _LENS_CODES.items()}


def record_dtype(n_rows: int) -> np.dtype:
    return np.dtype([('pts', '<f8'), ('out_fx', '<f4'), ('out_cx', '<f4'), ('out_cy', '<f4'), ('pad', '<f4'),
                     ('mats', '<f4', (n_rows, 3, 3))])


def fill_record_dtype(K: int, mesh_w: int, mesh_h: int) -> np.dtype:
    slot = np.dtype([('src', '<i4'), ('weight', '<f4'), ('gain', '<f4'), ('pad', '<f4'), ('G', '<f4', (3, 3)),
                     ('pad3', '<f4', (3,))])
    fields = [('n_src', '<u4'), ('frac_fill', '<f4'), ('frac_uncovered', '<f4'), ('pad', '<f4'), ('slots', slot, (K,))]
    if mesh_w * mesh_h:
        fields.append(('mesh', '<f4', (mesh_h, mesh_w, 2)))
    return np.dtype(fields)


def _fill_bytes(fill, F: int) -> bytes:
    if fill.n_frames != F:
        raise ValueError(f'fill table has {fill.n_frames} records, plan {F}')
    K = int(fill.src.shape[1])
    mh, mw = (fill.mesh.shape[1], fill.mesh.shape[2]) if fill.mesh is not None else (0, 0)
    dt = fill_record_dtype(K, mw, mh)
    assert dt.itemsize == 16 + 64 * K + 8 * mw * mh
    hdr = bytearray(FILL_HEADER_BYTES)
    struct.pack_into('<8sIIIII', hdr, 0, FILL_MAGIC, 1, FILL_HEADER_BYTES, F, K, dt.itemsize)
    struct.pack_into('<II5fIf', hdr, 28, mw, mh, fill.feather_main_px, fill.feather_nb_px, fill.sigma,
                     fill.fallback_weight, fill.fallback_blur, fill.max_offset, float(getattr(fill, 'par_px', 0.0)))
    rec = np.zeros(F, dtype=dt)
    rec['n_src'] = fill.n_src
    rec['frac_fill'] = fill.frac_fill
    rec['frac_uncovered'] = fill.frac_uncovered
    rec['slots']['src'] = fill.src
    rec['slots']['weight'] = fill.weight
    rec['slots']['gain'] = fill.gain
    rec['slots']['G'] = fill.G
    if mw * mh:
        rec['mesh'] = fill.mesh
    return bytes(hdr) + rec.tobytes()


def has_mesh(plan: Plan) -> bool:
    m = getattr(plan, 'mesh', None)
    return m is not None and np.asarray(m).size > 0 and bool(np.any(np.asarray(m) != 0))


def write_plan(path: str, plan: Plan, fill=None) -> None:
    """fill: optional fill.FillTable -> FILL section (flags bit 0); plan.mesh (non-zero) -> MESH block (bit 1)."""
    F, R = plan.n_frames, plan.n_rows
    L = plan.lens
    mesh = np.asarray(plan.mesh, dtype='<f4') if has_mesh(plan) else None
    if mesh is not None and (mesh.ndim != 4 or mesh.shape[0] != F or mesh.shape[3] != 2 or mesh.shape[1] < 2
                             or mesh.shape[2] < 2):
        raise ValueError(f'plan.mesh must be (n_frames, ny>=2, nx>=2, 2), got {mesh.shape}')
    hdr = bytearray(HEADER_BYTES)
    struct.pack_into('<8sII', hdr, 0, MAGIC, 1, HEADER_BYTES)
    struct.pack_into('<IIII', hdr, 16, plan.src_w, plan.src_h, plan.out_w, plan.out_h)
    struct.pack_into('<IIII', hdr, 32, F, R, _LENS_CODES[L.model], 24 + 36 * R)
    struct.pack_into('<8f', hdr, 48, L.fx, L.fy, L.cx, L.cy, *[float(c) for c in L.k])
    fill_b = _fill_bytes(fill, F) if fill is not None else b''
    flags = (FLAG_FILL if fill is not None else 0) | (FLAG_MESH if mesh is not None else 0)
    struct.pack_into('<fI', hdr, 80, float(plan.meta.get('readout_s', 0.0)), flags)
    if mesh is not None:
        ny, nx = mesh.shape[1:3]
        struct.pack_into('<IIQIf', hdr, 88, nx, ny, HEADER_BYTES + F * (24 + 36 * R) + len(fill_b), 8 * nx * ny,
                         float(plan.meta.get('mesh_clamp_px', 0.0)))
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
        if fill is not None:
            f.write(fill_b)
        if mesh is not None:
            f.write(np.ascontiguousarray(mesh).tobytes())


def read_fill(path: str):
    """The plan's FILL section as a fill.FillTable, or None when the plan has none."""
    from .fill import FillTable
    with open(path, 'rb') as f:
        hdr = f.read(HEADER_BYTES)
        magic, ver, hb = struct.unpack_from('<8sII', hdr, 0)
        if magic != MAGIC or ver != 1:
            raise ValueError(f"not a v1 .spplan file: {path}")
        F, R, _lc, rb = struct.unpack_from('<IIII', hdr, 32)
        _ro, flags = struct.unpack_from('<fI', hdr, 80)
        if not flags & FLAG_FILL:
            return None
        f.seek(hb + F * rb)
        fh = f.read(FILL_HEADER_BYTES)
        fm, fv, fhb, fF, K, frb = struct.unpack_from('<8sIIIII', fh, 0)
        if fm != FILL_MAGIC or fv != 1 or fF != F:
            raise ValueError(f"corrupt fill section in {path}")
        mw, mh, fmain, fnb, sigma, fbw, fbb, _maxoff, par = struct.unpack_from('<II5fIf', fh, 28)
        dt = fill_record_dtype(K, mw, mh)
        if dt.itemsize != frb:
            raise ValueError(f"fill record size {frb} != {dt.itemsize}")
        f.seek(hb + F * rb + fhb)
        rec = np.frombuffer(f.read(F * frb), dtype=dt, count=F)
    return FillTable(K=int(K), n_src=rec['n_src'].astype(np.int32), src=rec['slots']['src'].astype(np.int32),
                     weight=rec['slots']['weight'].astype(np.float32), gain=rec['slots']['gain'].astype(np.float32),
                     G=rec['slots']['G'].astype(np.float32), frac_fill=rec['frac_fill'].astype(np.float32),
                     frac_uncovered=rec['frac_uncovered'].astype(np.float32), feather_main_px=fmain,
                     feather_nb_px=fnb, sigma=sigma, fallback_weight=fbw, fallback_blur=fbb, par_px=float(par),
                     mesh=rec['mesh'].astype(np.float32) if mw * mh else None)


def read_plan(path: str) -> Plan:
    with open(path, 'rb') as f:
        hdr = f.read(HEADER_BYTES)
        magic, ver, hb = struct.unpack_from('<8sII', hdr, 0)
        if magic != MAGIC or ver != 1:
            raise ValueError(f"not a v1 .spplan file: {path}")
        src_w, src_h, out_w, out_h = struct.unpack_from('<IIII', hdr, 16)
        F, R, lens_code, rb = struct.unpack_from('<IIII', hdr, 32)
        fx, fy, cx, cy, k1, k2, k3, k4 = struct.unpack_from('<8f', hdr, 48)
        readout, flags = struct.unpack_from('<fI', hdr, 80)
        f.seek(hb)
        rec = np.frombuffer(f.read(F * rb), dtype=record_dtype(R), count=F)
        meta = {'readout_s': readout}
        mesh = None
        if flags & FLAG_MESH:
            nx, ny, moff, mfb, mclamp = struct.unpack_from('<IIQIf', hdr, 88)
            if nx < 2 or ny < 2 or mfb != 8 * nx * ny:
                raise ValueError(f"corrupt mesh extension in {path}")
            f.seek(moff)
            raw = f.read(F * mfb)
            if len(raw) != F * mfb:
                raise ValueError(f"truncated mesh block in {path}")
            mesh = np.frombuffer(raw, dtype='<f4').reshape(F, ny, nx, 2).copy()
            meta['mesh_clamp_px'] = float(mclamp)
    lens = Lens(_LENS_NAMES[lens_code], fx, fy, cx, cy, np.array([k1, k2, k3, k4]), src_w, src_h)
    return Plan(src_w=src_w, src_h=src_h, out_w=out_w, out_h=out_h, lens=lens,
                frame_pts=rec['pts'].astype(np.float64), out_fx=rec['out_fx'].astype(np.float64),
                row_mats=rec['mats'].astype(np.float64), virt_q=np.zeros((F, 4)),
                meta=meta, mesh=mesh)
