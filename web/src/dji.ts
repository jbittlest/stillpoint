/**
 * DJI `djmd` / `dbgi` metadata decoding (protobuf wire format) — TELEMETRY module.
 *
 * Port of the decoders in engine/stillpoint/telemetry.py (_pb, _parse_clip_meta, _parse_stream_meta, _parse_djmd,
 * _parse_dbgi_ac203). Field numbers are the dvtm_wm169 (O3), dvtm_ac203 (Osmo Action 4) and dvtm_O4P schemas
 * (research/footage_*.md). Values are bit-identical to the Python decoder: f32 fields are widened exactly, integer
 * ratios divide in float64, quaternions keep proto3's "omitted component = 0".
 */

// ------------------------------------------------------------------------------------------------ wire format

/** One protobuf field occurrence. wt 0: `n` is the varint (exact below 2^53; 10-byte negatives are sign-extended),
 *  wt 1/2/5: `b` is the payload bytes. */
export interface PbField { wt: number; n: number; b: Uint8Array | null }
export type PbMsg = Map<number, PbField[]>;

const EMPTY: PbMsg = new Map();

/** Varint at buf[i]; returns [value, next]. Exact below 2^53; 10-byte varints (negative int64) are sign-decoded. */
function varint(buf: Uint8Array, i: number): [number, number] {
  let c = buf[i++];
  if (c < 0x80) return [c, i];
  let r = c & 0x7f;
  let mul = 128;
  let nbytes = 1;
  const start = i - 1;
  while (true) {
    if (i >= buf.length) throw new Error('truncated varint');
    c = buf[i++];
    nbytes++;
    r += (c & 0x7f) * mul;
    if (!(c & 0x80)) break;
    mul *= 128;
    if (nbytes > 10) throw new Error('bad varint');
  }
  if (nbytes >= 10) {
    // two's complement int64 -> signed (Python _ints semantics); only exact through BigInt
    let big = 0n;
    for (let j = 0; j < nbytes; j++) big |= BigInt(buf[start + j] & 0x7f) << BigInt(7 * j);
    return [Number(BigInt.asIntN(64, big)), i];
  }
  return [r, i];
}

/** Parse a message into field -> occurrences (Python _pb). */
export function pb(buf: Uint8Array): PbMsg {
  const d: PbMsg = new Map();
  let i = 0;
  const n = buf.length;
  while (i < n) {
    let key: number;
    [key, i] = varint(buf, i);
    const fn = Math.floor(key / 8), wt = key & 7;
    let f: PbField;
    if (wt === 0) {
      let v: number;
      [v, i] = varint(buf, i);
      f = { wt, n: v, b: null };
    } else if (wt === 2) {
      let L: number;
      [L, i] = varint(buf, i);
      f = { wt, n: 0, b: buf.subarray(i, i + L) };
      i += L;
    } else if (wt === 5) {
      f = { wt, n: 0, b: buf.subarray(i, i + 4) };
      i += 4;
    } else if (wt === 1) {
      f = { wt, n: 0, b: buf.subarray(i, i + 8) };
      i += 8;
    } else {
      throw new Error(`unsupported wire type ${wt}`);
    }
    if (i > n) throw new Error('truncated protobuf');
    const arr = d.get(fn);
    if (arr) arr.push(f); else d.set(fn, [f]);
  }
  return d;
}

/** First occurrence of field fn (Python _get), or undefined. */
function first(d: PbMsg, fn: number): PbField | undefined {
  const x = d.get(fn);
  return x ? x[0] : undefined;
}
function getInt(d: PbMsg, fn: number, dflt: number): number {
  const f = first(d, fn);
  return f && f.wt === 0 ? f.n : dflt;
}
function getBytes(d: PbMsg, fn: number): Uint8Array | null {
  const f = first(d, fn);
  return f && f.b ? f.b : null;
}
/** Sub-message fn parsed (Python _msg): empty when absent or not length-delimited. */
function msg(d: PbMsg, fn: number): PbMsg {
  const f = first(d, fn);
  return f && f.wt === 2 && f.b ? pb(f.b) : EMPTY;
}
function f32(b: Uint8Array | null | undefined, dflt = NaN): number {
  if (!b || b.length !== 4) return dflt;
  return new DataView(b.buffer, b.byteOffset, 4).getFloat32(0, true);
}
function f32field1(d: PbMsg, fn: number): number {
  const f = first(d, fn);
  return f ? f32(f.b) : NaN;
}
function f32s(b: Uint8Array | null): number[] {
  if (!b) return [];
  const dv = new DataView(b.buffer, b.byteOffset, b.length);
  const out: number[] = [];
  for (let i = 0; i + 4 <= b.length; i += 4) out.push(dv.getFloat32(i, true));
  return out;
}
/** repeated int32/int64 (packed or not), signed (Python _ints). */
function ints(d: PbMsg, fn: number): number[] {
  const out: number[] = [];
  for (const f of d.get(fn) ?? []) {
    if (f.wt === 2 && f.b) {
      let i = 0;
      while (i < f.b.length) {
        let x: number;
        [x, i] = varint(f.b, i);
        out.push(x);
      }
    } else if (f.wt === 0) {
      out.push(f.n);
    }
  }
  return out;
}
const TD = new TextDecoder('utf-8');
function str(b: Uint8Array | null): string {
  return b ? TD.decode(b) : '';
}

/** dvtm Quaternion {1:w 2:x 3:y 4:z} (f32); omitted components are 0. Writes into out[o..o+4]. */
function quatInto(b: Uint8Array, out: Float64Array | number[], o: number) {
  if (b.length === 20 && b[0] === 0x0d && b[5] === 0x15 && b[10] === 0x1d && b[15] === 0x25) {
    const dv = new DataView(b.buffer, b.byteOffset, 20);
    out[o] = dv.getFloat32(1, true);
    out[o + 1] = dv.getFloat32(6, true);
    out[o + 2] = dv.getFloat32(11, true);
    out[o + 3] = dv.getFloat32(16, true);
    return;
  }
  const d = pb(b);
  for (let k = 0; k < 4; k++) out[o + k] = f32(getBytes(d, k + 1), 0.0);
}

function vec234(b: Uint8Array): [number, number, number] {
  const d = pb(b);
  return [f32(getBytes(d, 2), 0.0), f32(getBytes(d, 3), 0.0), f32(getBytes(d, 4), 0.0)];
}

// ------------------------------------------------------------------------------------------------ clip header

export interface ClipMeta {
  proto_file: string;
  lib_version: string;
  product_proto_version: string;
  firmware: string;
  clip_timestamp_us: number;
  product_name: string;
  dist_k?: number[];
  readout_ns?: number;
  read_direction?: number;
  fx?: number;
  eis_status: number | null;
  imu_rate?: number;
  sensor_fps?: number;
  fields: number[];
}

export interface StreamMeta {
  width?: number;
  height?: number;
  meta_fps?: number;
  bit_depth?: number | null;
  fov_type?: number;
}

export function parseClipMeta(b: Uint8Array): ClipMeta {
  const d = pb(b);
  const h = msg(d, 1);
  const m: ClipMeta = {
    proto_file: str(getBytes(h, 1)), lib_version: str(getBytes(h, 2)), product_proto_version: str(getBytes(h, 3)),
    firmware: str(getBytes(h, 6)), clip_timestamp_us: getInt(h, 9, 0), product_name: str(getBytes(h, 10)),
    eis_status: null, fields: [...d.keys()].sort((a, b) => a - b),
  };
  if (d.has(3)) m.dist_k = f32s(getBytes(msg(d, 3), 1));
  if (d.has(4)) m.readout_ns = getInt(msg(d, 4), 1, 0);
  if (d.has(5)) m.read_direction = getInt(msg(d, 5), 1, 0);
  if (d.has(8)) m.fx = f32(getBytes(msg(d, 8), 1));
  m.eis_status = d.has(9) ? getInt(msg(d, 9), 1, 0) : null;
  if (d.has(10)) m.imu_rate = getInt(msg(d, 10), 1, 0);
  if (d.has(11)) m.sensor_fps = f32(getBytes(msg(d, 11), 1));
  return m;
}

export function parseStreamMeta(b: Uint8Array): StreamMeta {
  const d = pb(b);
  const m: StreamMeta = {};
  const v = msg(d, 3);
  if (v.size) {
    m.width = getInt(v, 1, 0);
    m.height = getInt(v, 2, 0);
    m.meta_fps = f32(getBytes(v, 3));
    const bd = first(v, 5);
    m.bit_depth = bd && bd.wt === 0 ? bd.n : null;
  }
  if (d.has(5)) m.fov_type = getInt(msg(d, 5), 1, 0);
  return m;
}

// ------------------------------------------------------------------------------------------------ djmd

export interface Djmd {
  /** FrameMetaHeader.frame_timestamp, s (camera clock) */
  T: Float64Array;
  seq: Float64Array;
  exposure: Float64Array;
  iso: Float64Array;
  zoom: Float64Array;
  cct: Float64Array;
  /** per-frame camera attitude (raw DJI body->world, w,x,y,z), NaN when absent; n*4 */
  camQ: Float64Array;
  /** per-frame accelerometer (body, g), NaN when absent; n*3 */
  acc: Float64Array;
  attOff: Float64Array;
  attCnt: Int32Array;
  attVsync: Float64Array;
  /** all high-rate attitude quaternions (raw DJI body->world), in file order; nQuats*4 */
  quats: Float64Array;
  nQuats: number;
  /** repeated clip headers: [frame index, clip meta, stream meta] */
  clips: Array<[number, ClipMeta, StreamMeta]>;
}

/** Decode the djmd samples of a clip (Python _parse_djmd). */
export function parseDjmd(samples: Uint8Array[], onProgress?: (f: number) => void): Djmd {
  const n = samples.length;
  const T = new Float64Array(n), seq = new Float64Array(n);
  const exposure = new Float64Array(n).fill(NaN), iso = new Float64Array(n).fill(NaN);
  const zoom = new Float64Array(n).fill(NaN), cct = new Float64Array(n);
  const camQ = new Float64Array(4 * n).fill(NaN), acc = new Float64Array(3 * n).fill(NaN);
  const attOff = new Float64Array(n).fill(NaN), attCnt = new Int32Array(n), attVsync = new Float64Array(n).fill(-1);
  let quats = new Float64Array(Math.max(64, n * 40));
  let nq = 0;
  const clips: Array<[number, ClipMeta, StreamMeta]> = [];
  const step = Math.max(1, Math.floor(n / 50));
  for (let k = 0; k < n; k++) {
    const d = pb(samples[k]);
    if (d.has(1)) {
      const cm = parseClipMeta(getBytes(d, 1) ?? new Uint8Array(0));
      const sb = d.has(2) ? getBytes(d, 2) : null;
      clips.push([k, cm, sb ? parseStreamMeta(sb) : {}]);
    }
    const fm = msg(d, 3);
    const hd = msg(fm, 1);
    seq[k] = getInt(hd, 1, 0);
    T[k] = getInt(hd, 2, 0) * 1e-6;
    const cam = msg(fm, 2);
    if (cam.size) {
      const e = ints(msg(cam, 4), 1);
      if (e.length >= 2 && e[1]) exposure[k] = e[0] / e[1];
      iso[k] = f32field1Msg(cam, 3);
      zoom[k] = f32field1Msg(cam, 5);
      cct[k] = cam.has(6) ? getInt(msg(cam, 6), 1, 0) : 0;
      const q9 = getBytes(cam, 9);
      if (cam.has(9) && q9) quatInto(q9, camQ, 4 * k);
      const a10 = getBytes(cam, 10);
      if (cam.has(10) && a10) {
        const v = vec234(a10);
        acc[3 * k] = v[0]; acc[3 * k + 1] = v[1]; acc[3 * k + 2] = v[2];
      }
    }
    const imu = msg(fm, 3);
    if (imu.has(2)) {
      const att = msg(imu, 2);
      const qs = att.get(3) ?? [];
      if (nq + qs.length > quats.length / 4) {
        const nb = new Float64Array(Math.max(quats.length * 2, (nq + qs.length) * 4));
        nb.set(quats.subarray(0, nq * 4));
        quats = nb;
      }
      for (const f of qs) {
        if (f.b) quatInto(f.b, quats, 4 * nq);
        else { quats[4 * nq] = 0; quats[4 * nq + 1] = 0; quats[4 * nq + 2] = 0; quats[4 * nq + 3] = 0; }
        nq++;
      }
      attOff[k] = f32(getBytes(att, 4), 0.0);
      attVsync[k] = getInt(att, 2, 0);
      attCnt[k] = qs.length;
    }
    if (onProgress && k % step === 0) onProgress(k / n);
  }
  return { T, seq, exposure, iso, zoom, cct, camQ, acc, attOff, attCnt, attVsync,
           quats: quats.subarray(0, nq * 4), nQuats: nq, clips };
}

/** Python _f32field: sub-message fn whose field 1 is a float -> float or NaN. */
function f32field1Msg(d: PbMsg, fn: number): number {
  const m = msg(d, fn);
  return m.size ? f32field1(m, 1) : NaN;
}

/** FrameMetaHeader.frame_timestamp (s) of one djmd sample (Python _frame_ts). */
export function frameTs(b: Uint8Array): number {
  return getInt(msg(msg(pb(b), 3), 1), 2, 0) * 1e-6;
}

// ------------------------------------------------------------------------------------------------ dbgi (OA4)

export interface DbgiAc203 {
  /** EIS mode per frame (-1 unknown, 0 off, >0 active) */
  mode: Int32Array;
  vsync: Float64Array;
  /** EIS-output / physical attitude (raw DJI body->world), n*4, NaN when absent */
  qEis: Float64Array;
  qPhys: Float64Array;
  info: { sensor_mode?: string; pipeline_topology?: string };
}

/** OA4 dbgi: per frame EIS mode, vsync, EIS-output and physical attitudes (Python _parse_dbgi_ac203). */
export function parseDbgiAc203(samples: Uint8Array[]): DbgiAc203 {
  const n = samples.length;
  const mode = new Int32Array(n).fill(-1);
  const vsync = new Float64Array(n).fill(-1);
  const qEis = new Float64Array(4 * n).fill(NaN), qPhys = new Float64Array(4 * n).fill(NaN);
  const info: DbgiAc203['info'] = {};
  for (let k = 0; k < n; k++) {
    try {
      const d = pb(samples[k]);
      const fr = msg(msg(d, 2), 1);
      if (k === 0) {
        try {
          info.sensor_mode = str(getBytes(msg(fr, 2), 2));
          info.pipeline_topology = str(getBytes(msg(fr, 3), 2));
        } catch { /* optional */ }
      }
      const e = msg(fr, 11);
      const m6 = first(e, 6);
      mode[k] = m6 !== undefined ? (m6.wt === 0 ? m6.n : -1) : (e.has(5) ? 0 : -1);
      const s7 = msg(e, 7);
      vsync[k] = getInt(s7, 1, -1);
      if (s7.has(3)) { const v = f32s(getBytes(s7, 3)).slice(0, 4); for (let j = 0; j < v.length; j++) qEis[4 * k + j] = v[j]; }
      if (s7.has(4)) { const v = f32s(getBytes(s7, 4)).slice(0, 4); for (let j = 0; j < v.length; j++) qPhys[4 * k + j] = v[j]; }
    } catch {
      continue;
    }
  }
  return { mode, vsync, qEis, qPhys, info };
}

// ------------------------------------------------------------------------------------------------ product flags

/** [isO3, isOA4, isO4P] from a parsed clip header (Python _product_flags). */
export function productFlags(clip: ClipMeta): [boolean, boolean, boolean] {
  const proto = clip.proto_file ?? '';
  const product = clip.product_name ?? '';
  return [proto.includes('wm169') || product.includes('FC8383'),
          proto.includes('ac203') || product.replace(/ /g, '').includes('OsmoAction4'),
          proto.includes('O4P') || product.trim() === 'DJI O4P'];
}

/** EIS baked into the picture according to the clip header (+ the container comment when the header has no EIS). */
export function headerEisBaked(clip: ClipMeta, fmtComment: string): boolean {
  const s = clip.eis_status;
  let baked = !(s === null || s === undefined || s === 0 || s === 7);
  if ((s === null || s === undefined) && fmtComment.toUpperCase().replace(/ /g, '').includes('EIS:ON')) baked = true;
  return baked;
}

export const EIS_STATUS: Record<number, string> = {
  0: 'EIS_OFF', 1: 'EIS_ROCK_STEADY', 2: 'EIS_HORIZON_STEADY', 3: 'EIS_HYPER', 4: 'EIS_TRADEOFF',
  5: 'EIS_HORIZON_BALANCING', 6: 'EIS_DEEPSPACE', 7: 'EIS_OFF_WITH_CROP', 8: 'EIS_HORIZON_CORRECTION',
  9: 'EIS_RS_AUTO',
};
