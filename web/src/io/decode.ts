/**
 * WebCodecs video decoding for Stillpoint, hardened for other people's computers.
 *
 *  - Stream facts + NAL parsing. Which sync samples are clean random-access points for a WebCodecs decoder:
 *    H.264 only accepts an IDR as the first chunk (Chrome rejects a non-IDR "key" chunk); HEVC accepts IDR / BLA / CRA,
 *    and when decoding starts at a CRA its leading (RASL) pictures reference pictures before it, so they are skipped
 *    and a span that needs them starts at an earlier random-access point instead. (DJI O3 / O4 Pro H.264 and Osmo
 *    Action 4 HEVC use IDR only; iPhone HEVC uses CRA + RASL.)
 *  - Decoder variants: the configs to try, in order. H.264: hardware -> automatic -> software (Chrome and Edge ship a
 *    software H.264 decoder). HEVC has no software decoder in Chrome/Edge, so its variants are hardware configs with and
 *    without colour hints, hvc1 vs hev1, and Annex B with in-band parameter sets.
 *  - SequentialDecoder: one decoder over a decode-order span with strict backpressure (bounded decode queue and
 *    decoded-frame queue, samples read in coalesced batches with one batch of read-ahead), a stall watchdog, sample-size
 *    checks (truncated files), abort, and fault injection for tests.
 *  - DecoderSession + FrameStream: presentation-ordered frames over a range that survive decoder failures: the decoder
 *    is recreated — falling back to the next variant, which then sticks for the session — and decoding resumes from the
 *    last clean random-access point before the first frame still needed. Only then does it fail, with a DecodeFailure
 *    that says exactly where and why (and carries a diagnostics report).
 *  - DecoderSession.preflight(): decodes the first frames with each variant until one works (run when a clip opens), so
 *    a decoder that isConfigSupported() calls "supported" but can't actually decode the clip is caught up front.
 *
 * Every VideoFrame handed out must be closed by the caller; frames still queued at close() are closed here.
 */
import type { Mp4Track } from '../types';

export type ReadSamples = (file: Blob, t: Mp4Track, first: number, count: number) => Promise<Uint8Array[]>;
/** bytes [offset, offset + size) for each range (used to read the head of sync samples) */
export type ReadHeads = (offsets: ArrayLike<number>, sizes: ArrayLike<number>) => Promise<Uint8Array[]>;

// ───────────────────────────── colour ─────────────────────────────

// (TS's DOM lib lists only a subset of the WebCodecs colour enums, hence the casts)
const PRIMARIES: Record<number, VideoColorPrimaries> = <any> { 1: 'bt709', 5: 'bt470bg', 6: 'smpte170m', 9: 'bt2020', 12: 'smpte432' };
const TRANSFER: Record<number, VideoTransferCharacteristics> = <any> { 1: 'bt709', 6: 'smpte170m', 8: 'linear', 13: 'iec61966-2-1', 16: 'pq', 18: 'hlg' };
const MATRIX: Record<number, VideoMatrixCoefficients> = <any> { 0: 'rgb', 1: 'bt709', 5: 'bt470bg', 6: 'smpte170m', 9: 'bt2020-ncl' };

export function colorSpaceOf(t: Mp4Track): VideoColorSpaceInit | undefined {
  const c = t.colr;
  if (!c) return undefined;
  const cs: VideoColorSpaceInit = { fullRange: c.fullRange };
  if (PRIMARIES[c.primaries]) cs.primaries = PRIMARIES[c.primaries];
  if (TRANSFER[c.transfer]) cs.transfer = TRANSFER[c.transfer];
  if (MATRIX[c.matrix]) cs.matrix = MATRIX[c.matrix];
  return cs;
}

// ───────────────────────────── stream facts ─────────────────────────────

export type CodecFamily = 'h264' | 'hevc' | 'other';

export function codecFamily(t: Pick<Mp4Track, 'fourcc' | 'codec'> & { codecString?: string }): CodecFamily {
  const s = (t.codecString || t.fourcc || '').toLowerCase();
  if (/^(avc1|avc3)/.test(s) || t.codec === 'h264') return 'h264';
  if (/^(hvc1|hev1)/.test(s) || t.codec === 'hevc') return 'hevc';
  return 'other';
}

/** NAL length-prefix size from avcC / hvcC (4 unless the config says otherwise). */
export function nalLengthSize(t: Pick<Mp4Track, 'fourcc' | 'codec' | 'codecConfig'> & { codecString?: string }): number {
  const c = t.codecConfig;
  if (!c) return 4;
  const f = codecFamily(t);
  if (f === 'hevc' && c.length > 21) return (c[21] & 3) + 1;
  if (f === 'h264' && c.length > 4) return (c[4] & 3) + 1;
  return 4;
}

export interface StreamFacts {
  family: CodecFamily;
  codec: string;
  width: number;
  height: number;
  bitDepth: number;
  /** 'High', 'Main 10', … ('' when unknown) */
  profile: string;
  /** '5.2', '5.1', … ('' when unknown) */
  level: string;
  fps: number;
}

const AVC_PROFILES: Record<number, string> = { 66: 'Baseline', 77: 'Main', 88: 'Extended', 100: 'High', 110: 'High 10', 122: 'High 4:2:2', 244: 'High 4:4:4' };
const HEVC_PROFILES: Record<number, string> = { 1: 'Main', 2: 'Main 10', 3: 'Main Still Picture', 4: 'Range Extensions' };

export function streamFacts(t: Mp4Track, fps = 0): StreamFacts {
  const family = codecFamily(t);
  const c = t.codecConfig;
  let bitDepth = 8, profile = '', level = '';
  if (family === 'h264' && c && c.length > 5) {
    profile = AVC_PROFILES[c[1]] ?? `profile ${c[1]}`;
    level = `${Math.floor(c[3] / 10)}.${c[3] % 10}`;
    if (c[1] === 110) bitDepth = 10;
    // High-profile avcC extension (after the SPS/PPS lists): chroma_format, bit_depth_luma_minus8, …
    try {
      let p = 5;
      const nSps = c[p++] & 0x1f;
      for (let k = 0; k < nSps; k++) p += 2 + ((c[p] << 8) | c[p + 1]);
      const nPps = c[p++];
      for (let k = 0; k < nPps; k++) p += 2 + ((c[p] << 8) | c[p + 1]);
      if ([100, 110, 122, 244].includes(c[1]) && p + 2 < c.length) bitDepth = (c[p + 1] & 7) + 8;
    } catch { /* keep the profile's default */ }
  } else if (family === 'hevc' && c && c.length > 18) {
    profile = HEVC_PROFILES[c[1] & 0x1f] ?? `profile ${c[1] & 0x1f}`;
    level = (c[12] / 30).toFixed(1).replace(/\.0$/, '');
    bitDepth = (c[17] & 7) + 8;
  }
  return { family, codec: t.codecString ?? t.fourcc, width: t.width ?? 0, height: t.height ?? 0, bitDepth, profile, level, fps };
}

const fpsText = (f: number) => (f > 0 ? (f >= 100 ? f.toFixed(0) : f.toFixed(2).replace(/\.?0+$/, '')) : '');

/** "3840×2160 H.264 (High, level 5.2, 59.94 fps)" / "3840×2880 HEVC 10-bit (Main 10, level 5, 59.94 fps)" */
export function describeStream(f: StreamFacts): string {
  const name = f.family === 'h264' ? 'H.264' : f.family === 'hevc' ? 'HEVC' : f.codec;
  const bits = f.bitDepth > 8 ? ` ${f.bitDepth}-bit` : '';
  const extra = [f.profile, f.level && `level ${f.level}`, f.fps && `${fpsText(f.fps)} fps`].filter(Boolean).join(', ');
  return `${f.width}×${f.height} ${name}${bits}${extra ? ` (${extra})` : ''}`;
}

// ───────────────────────────── NAL units / random access ─────────────────────────────

export interface NalRef {
  /** offset of the NAL header byte in the access unit */
  offset: number;
  /** declared NAL size (may run past the buffer when only the head of a sample was read) */
  size: number;
}

/** NAL units of one length-prefixed access unit (stops at a zero / missing length). */
export function splitNals(au: Uint8Array, lenSize = 4): NalRef[] {
  const out: NalRef[] = [];
  let p = 0;
  while (p + lenSize <= au.length) {
    let n = 0;
    for (let k = 0; k < lenSize; k++) n = n * 256 + au[p + k];
    p += lenSize;
    if (n <= 0) break;
    out.push({ offset: p, size: n });
    p += n;
  }
  return out;
}

export function nalType(au: Uint8Array, offset: number, family: CodecFamily): number {
  return family === 'hevc' ? (au[offset] >> 1) & 0x3f : au[offset] & 0x1f;
}

/** NAL unit types of one access unit (numbers; HEVC 6-bit, H.264 5-bit). */
export function nalTypes(au: Uint8Array, family: CodecFamily, lenSize = 4): number[] {
  return splitNals(au, lenSize).filter(n => n.offset < au.length).map(n => nalType(au, n.offset, family));
}

/**
 * What kind of access unit this is, as a place to start decoding:
 *  idr       IDR (H.264 5; HEVC IDR_W_RADL 19 / IDR_N_LP 20): clean.
 *  cra / bla HEVC CRA 21 / BLA 16-18: a valid start, but its leading RASL pictures must be dropped.
 *  recovery  H.264 non-IDR I slice with a recovery-point SEI (open GOP): not a valid WebCodecs start.
 *  intra     H.264 non-IDR I slice: not a valid WebCodecs start.
 *  none      a P/B picture (HEVC non-IRAP).
 *  unknown   couldn't tell (not H.264/HEVC, or the head read didn't reach a slice).
 */
export type RapKind = 'idr' | 'cra' | 'bla' | 'recovery' | 'intra' | 'none' | 'unknown';

export function accessUnitKind(au: Uint8Array, family: CodecFamily, lenSize = 4): RapKind {
  if (family === 'other') return 'unknown';
  let recovery = false;
  for (const n of splitNals(au, lenSize)) {
    if (n.offset >= au.length) break;
    const t = nalType(au, n.offset, family);
    const end = Math.min(au.length, n.offset + n.size);
    if (family === 'hevc') {
      if (t >= 32) continue; // VPS / SPS / PPS / AUD / SEI …
      if (t === 19 || t === 20) return 'idr';
      if (t === 21) return 'cra';
      if (t >= 16 && t <= 18) return 'bla';
      if (t === 22 || t === 23) return 'unknown'; // reserved IRAP
      return 'none';
    }
    if (t === 5) return 'idr';
    if (t === 6 && seiHasRecoveryPoint(au, n.offset + 1, end)) recovery = true;
    if (t === 1 || t === 2) {
      const st = h264SliceType(au, n.offset + 1, end);
      if (st < 0) return 'unknown';
      const s = st % 5;
      if (s === 2 || s === 4) return recovery ? 'recovery' : 'intra';
      return 'none';
    }
  }
  return 'unknown';
}

/** RBSP bytes of [a, b) with emulation-prevention bytes removed (at most `max` bytes). */
function rbsp(au: Uint8Array, a: number, b: number, max = 64): Uint8Array {
  const out: number[] = [];
  let zeros = 0;
  for (let p = a; p < b && out.length < max; p++) {
    const x = au[p];
    if (zeros >= 2 && x === 3) { zeros = 0; continue; }
    out.push(x);
    zeros = x === 0 ? zeros + 1 : 0;
  }
  return Uint8Array.from(out);
}

/** slice_type of an H.264 slice NAL (payload after the 1-byte header), or -1. */
function h264SliceType(au: Uint8Array, a: number, b: number): number {
  const r = rbsp(au, a, b, 16);
  let bit = 0;
  const ue = (): number => {
    let lz = 0;
    while (bit < r.length * 8 && !((r[bit >> 3] >> (7 - (bit & 7))) & 1)) { lz++; bit++; }
    if (bit >= r.length * 8 || lz > 31) return -1;
    bit++;
    let v = 0;
    for (let k = 0; k < lz; k++) {
      if (bit >= r.length * 8) return -1;
      v = v * 2 + ((r[bit >> 3] >> (7 - (bit & 7))) & 1);
      bit++;
    }
    return (1 << lz) - 1 + v;
  };
  if (ue() < 0) return -1; // first_mb_in_slice
  return ue();
}

/** Does an H.264 SEI NAL (payload [a, b)) carry a recovery point (payloadType 6)? */
function seiHasRecoveryPoint(au: Uint8Array, a: number, b: number): boolean {
  const r = rbsp(au, a, b, 4096);
  let p = 0;
  while (p < r.length && r[p] !== 0x80) {
    let type = 0, size = 0;
    while (p < r.length && r[p] === 0xff) { type += 255; p++; }
    if (p >= r.length) return false;
    type += r[p++];
    while (p < r.length && r[p] === 0xff) { size += 255; p++; }
    if (p >= r.length) return false;
    size += r[p++];
    if (type === 6) return true;
    p += size;
  }
  return false;
}

/** Parameter sets (VPS/SPS/PPS or SPS/PPS) from the hvcC / avcC description. */
export function parameterSets(t: Pick<Mp4Track, 'fourcc' | 'codec' | 'codecConfig'> & { codecString?: string }): Uint8Array[] {
  const c = t.codecConfig;
  const out: Uint8Array[] = [];
  if (!c) return out;
  const fam = codecFamily(t);
  const u16 = (p: number) => (c[p] << 8) | c[p + 1];
  if (fam === 'hevc' && c.length > 23) {
    let p = 22;
    const arrays = c[p++];
    for (let a = 0; a < arrays && p + 3 <= c.length; a++) {
      const cnt = u16(p + 1);
      p += 3;
      for (let k = 0; k < cnt && p + 2 <= c.length; k++) { const len = u16(p); p += 2; if (p + len <= c.length) out.push(c.subarray(p, p + len)); p += len; }
    }
  } else if (fam === 'h264' && c.length > 6) {
    let p = 5;
    const nSps = c[p++] & 0x1f;
    for (let k = 0; k < nSps && p + 2 <= c.length; k++) { const len = u16(p); p += 2; out.push(c.subarray(p, p + len)); p += len; }
    const nPps = c[p++] ?? 0;
    for (let k = 0; k < nPps && p + 2 <= c.length; k++) { const len = u16(p); p += 2; out.push(c.subarray(p, p + len)); p += len; }
  }
  return out;
}

/** Length-prefixed access unit -> Annex B (start codes), optionally prefixed with parameter sets. */
export function toAnnexB(au: Uint8Array, lenSize: number, prefix: Uint8Array[] = []): Uint8Array {
  const nals = splitNals(au, lenSize).map(n => au.subarray(n.offset, Math.min(au.length, n.offset + n.size)));
  let total = 0;
  for (const x of prefix) total += 4 + x.length;
  for (const x of nals) total += 4 + x.length;
  const out = new Uint8Array(total);
  let q = 0;
  for (const x of [...prefix, ...nals]) { out[q + 3] = 1; q += 4; out.set(x, q); q += x.length; }
  return out;
}

/** Random-access classes per decode index (FrameIndex.rap). */
export const RAP_NONE = 0, RAP_CLEAN = 1, RAP_LEADING = 2, RAP_BAD = 3;

export function rapClass(k: RapKind): number {
  switch (k) {
    case 'idr': case 'unknown': return RAP_CLEAN; // unknown: trust the container's sync flag, as before
    case 'cra': case 'bla': return RAP_LEADING;
    default: return RAP_BAD;
  }
}

/**
 * Classify every sync sample by reading only its head (a few KB). Returns decode index -> kind. Samples whose head
 * didn't reach a slice are re-read with a bigger head.
 */
export async function classifySyncSamples(t: Mp4Track, readHeads: ReadHeads, headBytes = 4096): Promise<Map<number, RapKind>> {
  const out = new Map<number, RapKind>();
  const fam = codecFamily(t);
  if (fam === 'other') return out;
  const ls = nalLengthSize(t);
  const idx: number[] = [];
  for (let d = 0; d < t.sampleCount; d++) if (t.sync[d]) idx.push(d);
  let todo = idx;
  for (const head of [headBytes, 1 << 20]) {
    if (!todo.length) break;
    const heads = await readHeads(todo.map(d => t.offsets[d]), todo.map(d => Math.min(t.sizes[d], head)));
    const again: number[] = [];
    todo.forEach((d, j) => {
      const k = accessUnitKind(heads[j], fam, ls);
      if (k === 'unknown' && t.sizes[d] > head && head === headBytes) again.push(d);
      else out.set(d, k);
    });
    todo = again;
  }
  return out;
}

// ───────────────────────────── truncated files ─────────────────────────────

/** Number of leading (decode-order) samples whose bytes lie entirely inside a file of `fileSize` bytes. */
export function completeSamplePrefix(t: Pick<Mp4Track, 'sampleCount' | 'offsets' | 'sizes'>, fileSize: number): number {
  let n = 0;
  while (n < t.sampleCount && t.offsets[n] + t.sizes[n] <= fileSize) n++;
  return n;
}

/** The first n samples of a track (a view; nothing is copied). */
export function trimTrack(t: Mp4Track, n: number): Mp4Track {
  return { ...t, sampleCount: n, offsets: t.offsets.subarray(0, n), sizes: t.sizes.subarray(0, n), dts: t.dts.subarray(0, n), cts: t.cts.subarray(0, n), sync: t.sync.subarray(0, n) };
}

// ───────────────────────────── frame index ─────────────────────────────

/** Presentation-order bookkeeping for one video track. "pres" = index in presentation order, "dec" = sample index. */
export class FrameIndex {
  readonly n: number;
  /** pres -> decode index */
  readonly decOfPres: Int32Array;
  /** decode index -> pres */
  readonly presOfDec: Int32Array;
  /** presentation times (s), ascending */
  readonly pts: Float64Array;
  /** chunk timestamp (µs, integer) per decode index */
  readonly tsUs: Float64Array;
  /** random-access class per decode index (RAP_*): where a decoder may start */
  readonly rap: Uint8Array;
  /** census of the sync samples' kinds (e.g. { idr: 102 }) once classified */
  rapCensus: Partial<Record<RapKind, number>> = {};
  private presOfTs = new Map<number, number>();
  /** median frame duration, s */
  readonly frameDur: number;

  constructor(readonly track: Mp4Track) {
    const n = (this.n = track.sampleCount);
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => track.cts[a] - track.cts[b] || a - b);
    this.decOfPres = Int32Array.from(order);
    this.presOfDec = new Int32Array(n);
    this.pts = new Float64Array(n);
    this.tsUs = new Float64Array(n);
    this.rap = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
      const d = order[p];
      this.presOfDec[d] = p;
      this.pts[p] = track.cts[d];
    }
    for (let d = 0; d < n; d++) {
      let ts = Math.round(track.cts[d] * 1e6);
      while (this.presOfTs.has(ts)) ts++; // guard against duplicate timestamps
      this.tsUs[d] = ts;
      this.presOfTs.set(ts, this.presOfDec[d]);
      if (track.sync[d]) this.rap[d] = RAP_CLEAN;
    }
    const deltas: number[] = [];
    for (let p = 1; p < Math.min(n, 400); p++) deltas.push(this.pts[p] - this.pts[p - 1]);
    deltas.sort((a, b) => a - b);
    this.frameDur = deltas.length ? deltas[deltas.length >> 1] : 1 / 30;
  }

  /** Apply the NAL-level classification of the sync samples (classifySyncSamples). */
  applyRapKinds(kinds: Map<number, RapKind>) {
    const census: Partial<Record<RapKind, number>> = {};
    for (const [d, k] of kinds) {
      if (d < 0 || d >= this.n) continue;
      this.rap[d] = rapClass(k);
      census[k] = (census[k] ?? 0) + 1;
    }
    this.rapCensus = census;
  }

  presOfTimestamp(tsUs: number): number {
    const p = this.presOfTs.get(tsUs);
    if (p !== undefined) return p;
    return this.presNearest(tsUs / 1e6);
  }

  presNearest(t: number): number {
    const a = this.pts;
    let lo = 0, hi = a.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < t) lo = m + 1; else hi = m; }
    if (lo > 0 && Math.abs(a[lo - 1] - t) <= Math.abs(a[lo] - t)) lo--;
    return lo;
  }

  /** End (exclusive) of the leading pictures of the random-access point d: samples (d, end) present before d. */
  leadingEnd(d: number): number {
    const c = this.track.cts;
    let e = d + 1;
    while (e < this.n && c[e] < c[d]) e++;
    return e;
  }

  /**
   * Decode index to start at so that every pres frame in [p0, p1] decodes, given the lowest decode index needed:
   * the latest clean random-access point at or before it — or a CRA/BLA whose leading pictures aren't needed. Falls
   * back to the nearest sync sample when the file has no clean point there.
   */
  startFor(lo: number, p0: number, p1: number): number {
    for (let d = Math.min(lo, this.n - 1); d >= 0; d--) {
      const r = this.rap[d];
      if (r === RAP_CLEAN) return d;
      if (r === RAP_LEADING) {
        const e = this.leadingEnd(d);
        let ok = true;
        for (let x = d + 1; x < e; x++) { const p = this.presOfDec[x]; if (p >= p0 && p <= p1) { ok = false; break; } }
        if (ok) return d;
      }
    }
    const sync = this.track.sync;
    for (let d = Math.min(lo, this.n - 1); d >= 0; d--) if (sync[d]) return d;
    return 0;
  }

  /** Decode index of the random-access point to start at to decode pres frame p. */
  keyframeFor(p: number): number {
    return this.startFor(this.decOfPres[p], p, p);
  }

  /** Decode-order span [d0, d1] that yields every pres frame in [p0, p1]. */
  decodeSpan(p0: number, p1: number): [number, number] {
    let lo = Infinity, hi = -1;
    for (let p = p0; p <= p1; p++) { const d = this.decOfPres[p]; if (d < lo) lo = d; if (d > hi) hi = d; }
    return [this.startFor(lo, p0, p1), hi];
  }

  /** Samples to skip after starting at d0: (d0, end) are leading pictures of a CRA/BLA start (none otherwise). */
  skipAfterStart(d0: number): number {
    return this.rap[d0] === RAP_LEADING ? this.leadingEnd(d0) : d0 + 1;
  }

  /** Chunk type for sample d when d0 is the span's first sample (only valid random-access points are 'key'). */
  chunkType(d: number, d0: number): EncodedVideoChunkType {
    if (d === d0) return 'key';
    const r = this.rap[d];
    return r === RAP_CLEAN || r === RAP_LEADING ? 'key' : 'delta';
  }
}

// ───────────────────────────── decoder variants ─────────────────────────────

export interface DecodeVariant {
  /** 'hw', 'hw-nocolor', 'hw-hev1', 'hw-annexb', 'auto', 'sw' */
  id: string;
  /** human label: 'hardware', 'software', … */
  label: string;
  accel: HardwareAcceleration;
  /** samples converted to Annex B, config without description (in-band parameter sets) */
  annexB: boolean;
  config: VideoDecoderConfig;
}

/** Back-compat: the plain config for a track. */
export function decoderConfig(t: Mp4Track, hw: HardwareAcceleration = 'prefer-hardware', lowLatency = false): VideoDecoderConfig {
  if (!t.codecString) throw new Error(`Unsupported video codec (${t.fourcc || t.codec || 'unknown'})`);
  const cfg: VideoDecoderConfig = { codec: t.codecString, codedWidth: t.width, codedHeight: t.height, hardwareAcceleration: hw, optimizeForLatency: lowLatency };
  if (t.codecConfig) cfg.description = t.codecConfig;
  const cs = colorSpaceOf(t);
  if (cs) cfg.colorSpace = cs;
  return cfg;
}

/** All the configs worth trying for a track, best first (not yet filtered by isConfigSupported). */
export function decoderVariants(t: Mp4Track): DecodeVariant[] {
  if (!t.codecString) throw new Error(`Unsupported video codec (${t.fourcc || t.codec || 'unknown'})`);
  const fam = codecFamily(t);
  const cs = colorSpaceOf(t);
  const mk = (id: string, label: string, accel: HardwareAcceleration, o: { noColor?: boolean; codec?: string; annexB?: boolean } = {}): DecodeVariant => {
    const config: VideoDecoderConfig = { codec: o.codec ?? t.codecString!, codedWidth: t.width, codedHeight: t.height, hardwareAcceleration: accel, optimizeForLatency: false };
    if (t.codecConfig && !o.annexB) config.description = t.codecConfig;
    if (cs && !o.noColor) config.colorSpace = cs;
    return { id, label, accel, annexB: !!o.annexB, config };
  };
  const out = [mk('hw', 'hardware', 'prefer-hardware')];
  if (fam === 'hevc') {
    const rest = t.codecString.slice(4);
    const alt = t.codecString.startsWith('hvc1') ? 'hev1' : 'hvc1';
    if (cs) out.push(mk('hw-nocolor', 'hardware, no colour hints', 'prefer-hardware', { noColor: true }));
    out.push(mk(`hw-${alt}`, `hardware, ${alt}`, 'prefer-hardware', { codec: alt + rest }));
    out.push(mk('hw-annexb', 'hardware, Annex B', 'prefer-hardware', { codec: 'hev1' + rest, annexB: true }));
  }
  out.push(mk('auto', 'automatic', 'no-preference'));
  out.push(mk('sw', 'software', 'prefer-software'));
  return out;
}

export interface SupportedVariants {
  variants: DecodeVariant[];
  /** isConfigSupported() said yes to a prefer-hardware config */
  hwSupported: boolean;
  rejected: Array<{ variant: string; reason: string }>;
}

/** decoderVariants() filtered by VideoDecoder.isConfigSupported(), order kept. */
export async function supportedVariants(t: Mp4Track): Promise<SupportedVariants> {
  if (typeof VideoDecoder === 'undefined') return { variants: [], hwSupported: false, rejected: [{ variant: '*', reason: 'This browser has no WebCodecs VideoDecoder.' }] };
  let all: DecodeVariant[];
  try { all = decoderVariants(t); } catch (e) { return { variants: [], hwSupported: false, rejected: [{ variant: '*', reason: errText(e) }] }; }
  const variants: DecodeVariant[] = [];
  const rejected: SupportedVariants['rejected'] = [];
  for (const v of all) {
    try {
      const r = await VideoDecoder.isConfigSupported(v.config);
      if (r.supported) variants.push(v); else rejected.push({ variant: v.id, reason: 'not supported' });
    } catch (e) { rejected.push({ variant: v.id, reason: errText(e) }); }
  }
  return { variants, hwSupported: variants.some(v => v.accel === 'prefer-hardware'), rejected };
}

/** Back-compat: first supported config (hardware preferred), or null with the reason. */
export async function supportedDecoderConfig(t: Mp4Track, lowLatency = false): Promise<{ config: VideoDecoderConfig | null; reason?: string }> {
  const s = await supportedVariants(t);
  if (!s.variants.length) return { config: null, reason: s.rejected[0]?.reason ?? 'Unsupported video codec.' };
  return { config: { ...s.variants[0].config, optimizeForLatency: lowLatency } };
}

/**
 * The variant to switch to after `variants[cur]` failed mid-stream: from a hardware / automatic decoder straight to
 * the first software one (an automatic decoder would just pick the failing hardware again); otherwise the next one.
 * -1 when there is none.
 */
export function fallbackIndex(variants: DecodeVariant[], cur: number): number {
  if (variants[cur] && variants[cur].accel !== 'prefer-software') {
    const j = variants.findIndex((v, k) => k > cur && v.accel === 'prefer-software');
    if (j >= 0) return j;
  }
  return cur + 1 < variants.length ? cur + 1 : -1;
}

// ───────────────────────────── platform ─────────────────────────────

export interface DecodeLimits { maxFrames: number; maxDecodeQueue: number }
export interface PlatformLimits { preview: DecodeLimits; playback: DecodeLimits; export: DecodeLimits }

export function isApplePlatform(ua = typeof navigator !== 'undefined' ? navigator.userAgent : '', platform = typeof navigator !== 'undefined' ? navigator.platform : ''): boolean {
  return /Mac|iPhone|iPad|iPod/i.test(platform) || /Mac OS X|iPhone|iPad/.test(ua);
}

/**
 * Decoded frames held + chunks in flight. Apple's VideoToolbox pools are roomy; Windows (D3D11) / Linux / ChromeOS
 * hardware decoders have small output pools that error or stall when the app holds too many frames.
 */
export function decodeLimits(apple: boolean): PlatformLimits {
  return apple
    ? { preview: { maxFrames: 4, maxDecodeQueue: 8 }, playback: { maxFrames: 4, maxDecodeQueue: 4 }, export: { maxFrames: 5, maxDecodeQueue: 4 } }
    : { preview: { maxFrames: 3, maxDecodeQueue: 4 }, playback: { maxFrames: 3, maxDecodeQueue: 3 }, export: { maxFrames: 3, maxDecodeQueue: 3 } };
}

// ───────────────────────────── errors ─────────────────────────────

export function errText(e: unknown): string {
  if (e == null) return 'unknown error';
  const x = e as { name?: string; message?: string };
  if (typeof x.message === 'string') return x.name && x.name !== 'Error' ? `${x.name}: ${x.message}` : x.message;
  return String(e);
}

export function isAbort(e: unknown): boolean {
  return (e as DOMException)?.name === 'AbortError';
}

const abortError = () => new DOMException('Aborted', 'AbortError');

/** The file ends before a sample's bytes (truncated / incompletely copied). Deterministic: never retried. */
export class TruncatedSampleError extends Error {
  constructor(readonly sample: number, readonly got: number, readonly want: number) {
    super(`frame data is missing from the file (sample ${sample}: ${got} of ${want} bytes) — the file looks truncated or incompletely copied`);
    this.name = 'TruncatedSampleError';
  }
}

/**
 * A sample whose bytes aren't a well-formed length-prefixed access unit: zeros or foreign data where the video should
 * be (a copy that didn't finish, a cloud-synced file that isn't fully downloaded, a damaged card). Deterministic: no
 * decoder can do better, so it is never retried and never blamed on the decoder.
 */
export class DamagedSampleError extends Error {
  constructor(readonly sample: number) {
    super(`the video data of sample ${sample} is blank or damaged — the file looks corrupted or incompletely copied`);
    this.name = 'DamagedSampleError';
  }
}

/**
 * Is this a well-formed length-prefixed (avcC / hvcC) access unit? The NAL lengths must chain inside the sample
 * (trailing zero padding tolerated), every NAL header's forbidden_zero_bit must be 0, and no NAL may end in a long run
 * of zero bytes (emulation prevention makes 3+ zeros impossible inside a NAL, and a NAL's last byte is never 0x00; 64
 * leaves room for sloppy muxers). Zero-filled or foreign data fails this almost surely.
 */
export function wellFormedAccessUnit(au: Uint8Array, lenSize = 4): boolean {
  let p = 0, nals = 0;
  while (p + lenSize <= au.length) {
    let n = 0;
    for (let k = 0; k < lenSize; k++) n = n * 256 + au[p + k];
    if (n === 0) break;
    const a = p + lenSize, b = a + n;
    if (b > au.length || (au[a] & 0x80)) return false;
    if (n >= 64) {
      let zeros = true;
      for (let q = b - 64; q < b; q++) if (au[q] !== 0) { zeros = false; break; }
      if (zeros) return false;
    }
    p = b;
    nals++;
  }
  if (!nals) return false;
  for (; p < au.length; p++) if (au[p] !== 0) return false;
  return true;
}

/** The decoder stopped producing frames (hung). */
export class DecoderStallError extends Error {
  constructor(readonly ms: number) {
    super(`the video decoder stopped responding (no frame for ${(ms / 1000).toFixed(0)} s)`);
    this.name = 'DecoderStallError';
  }
}

export interface DecodeEvent {
  /** ms since the session started */
  t: number;
  kind: 'preflight-ok' | 'preflight-fail' | 'switch' | 'retry' | 'fail';
  variant: string;
  to?: string;
  purpose?: string;
  pres?: number;
  error?: string;
  ms?: number;
}

export interface DecodeReport {
  kind: 'decode' | 'stall' | 'truncated' | 'damaged' | 'unsupported' | 'no-output';
  purpose: string;
  /** failing presentation frame and its time in the clip (s), -1 when not frame-specific */
  pres: number;
  timeS: number;
  sample: number;
  stream: StreamFacts & { description: string; samples: number; fileBytes: number };
  variant: string;
  config: Record<string, unknown>;
  variantsLeft: string[];
  error: { name: string; message: string };
  history: DecodeEvent[];
  rap: Partial<Record<RapKind, number>>;
  limits?: PlatformLimits;
}

/** A decode that failed for good (after fallbacks / retries). message is written for people; report for diagnostics. */
export class DecodeFailure extends Error {
  constructor(message: string, readonly report: DecodeReport) {
    super(message);
    this.name = 'DecodeFailure';
  }
}

function fmtClock(s: number): string {
  if (!Number.isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, '0')}`;
}

function configSummary(c: VideoDecoderConfig): Record<string, unknown> {
  const { description, ...rest } = c as VideoDecoderConfig & { description?: AllowSharedBufferSource };
  return { ...rest, description: description ? `${(description as ArrayBufferView).byteLength ?? (description as ArrayBuffer).byteLength} bytes` : 'none' };
}

/**
 * The friendly explanation when no decoder config can play the clip. `why`: 'unsupported' = isConfigSupported() said
 * no to everything; 'failed' = the decoders said yes but failed on the clip's first frames.
 */
export function cannotDecodeMessage(f: StreamFacts, why: 'unsupported' | 'failed'): string {
  const what = `${f.width}×${f.height} ${f.family === 'hevc' ? 'HEVC' : f.family === 'h264' ? 'H.264' : f.codec}${f.bitDepth > 8 ? ` ${f.bitDepth}-bit` : ''}${f.fps > 0 ? ` at ${fpsText(f.fps)} fps` : ''}`;
  if (f.family === 'hevc') {
    return `This computer’s video decoder can’t play ${what}. Chrome and Edge decode HEVC only with the graphics card’s hardware decoder, and ${why === 'unsupported' ? 'this one doesn’t offer it' : 'it failed on this clip'}. `
      + 'Try Chrome on a Mac with Apple Silicon, or a PC with a recent GPU (NVIDIA RTX, Intel Arc or AMD RDNA2 or newer) that has hardware HEVC decoding. '
      + 'On Windows, also make sure hardware acceleration is on (Settings → System) and the graphics driver is up to date.';
  }
  if (f.family === 'h264') {
    return `This browser couldn’t decode ${what}${f.profile ? ` (${f.profile}${f.level ? `, level ${f.level}` : ''})` : ''}, with either the graphics card or its software decoder. Try the latest Chrome or Edge.`;
  }
  return `Stillpoint can’t decode ${f.codec || 'this'} video in this browser. It works with the H.264 and HEVC clips DJI cameras record.`;
}

/** The subtle note shown when decoding runs in software: the hardware decoder failed, or there is none for this clip. */
export function softwareNote(f: StreamFacts, why: 'failed' | 'unavailable' = 'failed'): string {
  const clip = describeStream(f);
  return why === 'failed'
    ? `This computer’s hardware video decoder couldn’t handle ${clip}, so Stillpoint is decoding it in software. Preview and export still work, just slower.`
    : `This browser has no hardware decoding for ${clip} here, so Stillpoint is decoding it in software. Preview and export still work, just slower.`;
}

// ───────────────────────────── sequential decoder ─────────────────────────────

interface Batch { first: number; data: Uint8Array[] }

export interface FaultPlan {
  /** fail (or hang) once the decoder outputs presentation frame >= at */
  at: number;
  mode: 'error' | 'hang';
}

export interface SeqOptions {
  maxFrames?: number;
  maxDecodeQueue?: number;
  batchBytes?: number;
  /** fail when one next() call waits this long without an output (ms; 0 = never) */
  stallMs?: number;
  signal?: AbortSignal;
  /** test hook (sp_fault): simulate a decoder failure / hang */
  fault?: FaultPlan;
}

/**
 * Decodes samples [d0, d1] (decode order) and hands out frames (presentation order, as the decoder emits them) via
 * next(). At most `maxFrames` decoded frames are held (queued + not yet returned), and at most `maxDecodeQueue` chunks
 * wait inside the decoder. When d0 is a CRA/BLA its leading pictures are not fed (they reference earlier pictures).
 */
export class SequentialDecoder {
  private decoder: VideoDecoder;
  private out: VideoFrame[] = [];
  private wake: (() => void) | null = null;
  private error: unknown = null;
  private nextSample: number;
  private flushing = false;
  private done = false;
  private closed = false;
  private pumping = false;
  private hung = false;
  private cur: Batch | null = null;
  private ahead: Promise<Batch> | null = null;
  private readonly batchSize: number;
  private readonly skipEnd: number;
  private readonly variant: DecodeVariant;
  private readonly lenSize: number;
  private readonly psets: Uint8Array[];
  /** check each sample is a well-formed H.264 / HEVC access unit before feeding it (damaged / blank file data) */
  private readonly checkAu: boolean;
  private readonly onAbort = () => this.close();
  /** missing / damaged sample data found while feeding: raised once every frame before it has been handed out */
  private fileError: TruncatedSampleError | DamagedSampleError | null = null;
  /** frames output by the decoder so far */
  decodedCount = 0;
  /** last sample handed to decode() */
  lastFed = -1;
  /** presentation index of the last frame output */
  lastOutPres = -1;

  constructor(
    private file: Blob,
    private track: Mp4Track,
    private index: FrameIndex,
    private readSamples: ReadSamples,
    configOrVariant: VideoDecoderConfig | DecodeVariant,
    private d0: number,
    private d1: number,
    private opts: SeqOptions = {},
  ) {
    this.variant = 'config' in configOrVariant
      ? configOrVariant
      : { id: 'custom', label: String(configOrVariant.hardwareAcceleration ?? 'no-preference'), accel: configOrVariant.hardwareAcceleration ?? 'no-preference', annexB: false, config: configOrVariant };
    this.nextSample = d0;
    this.skipEnd = index.skipAfterStart(d0);
    this.lenSize = nalLengthSize(track);
    this.psets = this.variant.annexB ? parameterSets(track) : [];
    this.checkAu = codecFamily(track) !== 'other';
    const avg = track.sizes.length ? track.sizes.reduce((a, b) => a + b, 0) / track.sizes.length : 200_000;
    this.batchSize = Math.max(4, Math.min(120, Math.floor((opts.batchBytes ?? 12 * 1024 * 1024) / Math.max(1, avg))));
    this.decoder = new VideoDecoder({
      output: f => this.onOutput(f),
      error: e => this.fail(e),
    });
    this.decoder.addEventListener('dequeue', () => { this.notify(); void this.pump(); });
    this.decoder.configure(this.variant.config);
    if (opts.signal) {
      if (opts.signal.aborted) this.close();
      else opts.signal.addEventListener('abort', this.onAbort, { once: true });
    }
  }

  private onOutput(f: VideoFrame) {
    if (this.closed || this.hung) { f.close(); return; }
    const pres = this.index.presOfTimestamp(f.timestamp);
    const fault = this.opts.fault;
    if (fault && pres >= fault.at) {
      f.close();
      if (fault.mode === 'hang') { this.hung = true; return; }
      this.fail(new DOMException(`Decoding error. (simulated ${this.variant.label} decoder failure, sp_fault)`, 'EncodingError'));
      try { this.decoder.close(); } catch { /* a real decoder closes itself on error too */ }
      return;
    }
    this.decodedCount++;
    this.lastOutPres = pres;
    this.out.push(f);
    this.notify();
  }

  private fail(e: unknown) {
    if (this.error || this.closed) return;
    this.error = e ?? new Error('video decoder error');
    this.notify();
  }

  private notify() { const w = this.wake; this.wake = null; w?.(); }

  private fetchBatch(first: number): Promise<Batch> {
    const count = Math.min(this.batchSize, this.d1 + 1 - first);
    return this.readSamples(this.file, this.track, first, count).then(data => ({ first, data }));
  }

  private async sampleAt(i: number): Promise<Uint8Array> {
    if (!this.cur || i < this.cur.first || i >= this.cur.first + this.cur.data.length) {
      if (this.ahead) {
        const b = await this.ahead;
        this.ahead = null;
        this.cur = i >= b.first && i < b.first + b.data.length ? b : await this.fetchBatch(i);
      } else {
        this.cur = await this.fetchBatch(i);
      }
      const nxt = this.cur.first + this.cur.data.length;
      if (nxt <= this.d1) {
        this.ahead = this.fetchBatch(nxt);
        this.ahead.catch(() => { /* surfaced when awaited */ });
      }
    }
    return this.cur.data[i - this.cur.first];
  }

  private get maxFrames() { return this.opts.maxFrames ?? 6; }
  private get maxDecodeQueue() { return this.opts.maxDecodeQueue ?? 4; }

  private async pump() {
    if (this.pumping || this.closed || this.error || this.hung) return;
    this.pumping = true;
    try {
      while (!this.closed && !this.error && !this.hung && this.nextSample <= this.d1
        && this.decoder.decodeQueueSize < this.maxDecodeQueue && this.out.length < this.maxFrames) {
        const i = this.nextSample;
        if (i > this.d0 && i < this.skipEnd) { this.nextSample++; continue; } // leading pictures of a CRA start
        const raw = await this.sampleAt(i);
        if (this.closed || this.error) return;
        // missing / damaged data: stop feeding here, let the frames before it come out, then raise it (next())
        if (!raw || raw.length !== this.track.sizes[i]) this.fileError = new TruncatedSampleError(i, raw?.length ?? 0, this.track.sizes[i]);
        else if (this.checkAu && !wellFormedAccessUnit(raw, this.lenSize)) this.fileError = new DamagedSampleError(i);
        if (this.fileError) { this.d1 = i - 1; break; }
        if (this.decoder.state === 'closed') {
          // the decoder failed and closed itself; its error callback (the real reason) is on its way — don't replace it
          // with "Cannot call 'decode' on a closed codec"
          setTimeout(() => this.fail(new DOMException('The video decoder closed unexpectedly.', 'InvalidStateError')), 250);
          return;
        }
        const type = this.index.chunkType(i, this.d0);
        const data = this.variant.annexB ? toAnnexB(raw, this.lenSize, type === 'key' ? this.psets : []) : raw;
        this.decoder.decode(new EncodedVideoChunk({
          type,
          timestamp: this.index.tsUs[i],
          duration: Math.round(this.index.frameDur * 1e6),
          data,
        }));
        this.lastFed = i;
        this.nextSample++;
      }
      if (!this.closed && !this.error && !this.hung && this.nextSample > this.d1 && !this.flushing) {
        this.flushing = true;
        this.decoder.flush().then(
          () => { if (!this.hung) this.done = true; this.notify(); },
          e => { if (!this.closed) this.fail(e); },
        );
      }
    } catch (e) {
      if (!this.closed) this.fail(e);
    } finally {
      this.pumping = false;
    }
  }

  /** Next decoded frame (caller closes it), or null at the end of the span (or after close()). */
  async next(): Promise<VideoFrame | null> {
    const t0 = performance.now();
    const stall = this.opts.stallMs ?? 0;
    for (;;) {
      if (this.error) throw this.error;
      if (this.out.length) { const f = this.out.shift()!; void this.pump(); return f; }
      if (this.done && this.fileError && !this.closed) throw this.fileError;
      if (this.done || this.closed) return null;
      const p = new Promise<void>(r => (this.wake = r));
      void this.pump();
      if (stall > 0) {
        const left = stall - (performance.now() - t0);
        if (left <= 0) { this.fail(new DecoderStallError(stall)); continue; }
        const timer = setTimeout(() => this.notify(), left + 5);
        await p;
        clearTimeout(timer);
      } else {
        await p;
      }
    }
  }

  get queued() { return this.out.length; }

  close() {
    if (this.closed) return;
    this.closed = true;
    this.opts.signal?.removeEventListener('abort', this.onAbort);
    for (const f of this.out) f.close();
    this.out = [];
    try { if (this.decoder.state !== 'closed') this.decoder.close(); } catch { /* ignore */ }
    this.notify();
  }
}

// ───────────────────────────── session + resilient frame stream ─────────────────────────────

export interface FaultSpec {
  /** which decoders fail: every one, or only hardware / automatic ones */
  scope: 'hw' | 'all';
  mode: 'error' | 'hang';
  /** presentation frame from which they fail */
  at: number;
  /** only the first matching decoder instance */
  firstOnly: boolean;
}

/**
 * Test hook (?sp_fault=…): 'hw-first' (the first hardware decoder fails at once), 'hw' (every hardware decoder fails
 * at once), 'hw-at:N' / 'all-at:N' (fail from frame N), 'hang-at:N' / 'hw-hang-at:N' (stop producing frames from N).
 */
export function parseFault(s: string | null | undefined): FaultSpec | null {
  if (!s) return null;
  if (s === 'hw-first') return { scope: 'hw', mode: 'error', at: 0, firstOnly: true };
  if (s === 'hw') return { scope: 'hw', mode: 'error', at: 0, firstOnly: false };
  const m = /^(hw|all|hw-hang|hang)-at:(\d+)$/.exec(s);
  if (!m) return null;
  return { scope: m[1].startsWith('hw') ? 'hw' : 'all', mode: m[1].includes('hang') ? 'hang' : 'error', at: +m[2], firstOnly: false };
}

export interface SessionOptions {
  limits: PlatformLimits;
  /** runtime stall watchdog (ms) */
  stallMs: number;
  /** pre-flight stall watchdog (ms) */
  preflightStallMs?: number;
  fault?: FaultSpec | null;
}

export interface PreflightResult {
  ok: boolean;
  ms: number;
  frames: number;
  tried: Array<{ variant: string; ok: boolean; ms: number; error?: string }>;
}

export interface FrameStreamOptions {
  purpose: 'preview' | 'playback' | 'export' | 'preflight';
  signal?: AbortSignal;
  lowLatency?: boolean;
  maxFrames?: number;
  maxDecodeQueue?: number;
  /** recover from decoder errors (fallback / retry); false = fail on the first error */
  retries?: boolean;
  /** 'fail': frames must arrive in order without gaps (export); 'skip': a missing frame is skipped (playback) */
  gaps?: 'fail' | 'skip';
  stallMs?: number;
}

/**
 * The decoding state of one opened clip: the variants still worth trying (the current one first after pre-flight),
 * which one is in use (sticky after a fallback), and a history of what happened for diagnostics. One session creates
 * decoders one at a time; the engine makes sure only one is alive.
 */
export class DecoderSession {
  current = 0;
  readonly history: DecodeEvent[] = [];
  /** a fallback happened: (from, to, event) */
  onSwitch?: (from: DecodeVariant, to: DecodeVariant, ev: DecodeEvent) => void;
  hwSupported = true;
  private instances = 0;
  private hwInstances = 0;
  private readonly t0 = typeof performance !== 'undefined' ? performance.now() : 0;
  readonly facts: StreamFacts;

  constructor(
    readonly file: Blob,
    readonly track: Mp4Track,
    readonly index: FrameIndex,
    readonly readSamples: ReadSamples,
    public variants: DecodeVariant[],
    readonly opts: SessionOptions,
  ) {
    this.facts = streamFacts(track, 1 / index.frameDur);
  }

  get variant(): DecodeVariant { return this.variants[this.current]; }

  /** likely decoding in software (explicitly, or 'automatic' when no hardware config was supported) */
  get software(): boolean {
    const v = this.variant;
    return !!v && (v.accel === 'prefer-software' || (v.accel === 'no-preference' && !this.hwSupported));
  }

  private now() { return Math.round(performance.now() - this.t0); }

  log(ev: Omit<DecodeEvent, 't'>) {
    this.history.push({ t: this.now(), ...ev });
    if (this.history.length > 60) this.history.splice(0, this.history.length - 60);
  }

  /** A decoder over decode span [d0, d1] with the current variant. */
  createDecoder(d0: number, d1: number, o: SeqOptions & { lowLatency?: boolean }): SequentialDecoder {
    const v = this.variant;
    const hwish = v.accel !== 'prefer-software';
    const F = this.opts.fault;
    let fault: FaultPlan | undefined;
    if (F && (F.scope === 'all' || hwish) && (!F.firstOnly || this.hwInstances === 0)) fault = { at: F.at, mode: F.mode };
    this.instances++;
    if (hwish) this.hwInstances++;
    const variant = o.lowLatency ? { ...v, config: { ...v.config, optimizeForLatency: true } } : v;
    return new SequentialDecoder(this.file, this.track, this.index, this.readSamples, variant, d0, d1, { ...o, fault });
  }

  frames(p0: number, p1: number, o: FrameStreamOptions): FrameStream {
    return new FrameStream(this, p0, p1, o);
  }

  /** Switch to the fallback variant after a failure; false when there is none. */
  fallBack(err: unknown, pres: number, purpose: string): boolean {
    const j = fallbackIndex(this.variants, this.current);
    if (j < 0) return false;
    const from = this.variant, to = this.variants[j];
    const ev: DecodeEvent = { t: this.now(), kind: 'switch', variant: from.id, to: to.id, purpose, pres, error: errText(err) };
    this.history.push(ev);
    this.current = j;
    try { this.onSwitch?.(from, to, ev); } catch { /* informational */ }
    return true;
  }

  /** Diagnostics for a failure at pres frame `pres` (-1: not frame-specific). */
  report(err: unknown, pres: number, purpose: string, sample = -1): DecodeReport {
    const e = err as { name?: string; message?: string };
    const kind: DecodeReport['kind'] = err instanceof TruncatedSampleError ? 'truncated' : err instanceof DamagedSampleError ? 'damaged' : err instanceof DecoderStallError ? 'stall'
      : /no picture|came out/.test(String(e?.message)) ? 'no-output' : 'decode';
    const v = this.variant;
    return {
      kind, purpose, pres,
      timeS: pres >= 0 && pres < this.index.n ? +(this.index.pts[pres] - this.index.pts[0]).toFixed(3) : -1,
      sample: err instanceof TruncatedSampleError || err instanceof DamagedSampleError ? err.sample : sample >= 0 ? sample : pres >= 0 && pres < this.index.n ? this.index.decOfPres[pres] : -1,
      stream: { ...this.facts, description: describeStream(this.facts), samples: this.track.sampleCount, fileBytes: this.file.size },
      variant: v ? `${v.id} (${v.label})` : 'none',
      config: v ? configSummary(v.config) : {},
      variantsLeft: this.variants.slice(this.current + 1).map(x => x.id),
      error: { name: e?.name ?? 'Error', message: e?.message ?? String(err) },
      history: this.history.slice(),
      rap: this.index.rapCensus,
      limits: this.opts.limits,
    };
  }

  /** The DecodeFailure for an unrecoverable error, with a precise message. */
  failure(err: unknown, pres: number, purpose: string): DecodeFailure {
    if (err instanceof DecodeFailure) return err;
    const r = this.report(err, pres, purpose);
    this.log({ kind: 'fail', variant: this.variant?.id ?? 'none', purpose, pres, error: errText(err) });
    r.history = this.history.slice();
    const where = pres >= 0 ? `at ${fmtClock(r.timeS)} (frame ${pres.toLocaleString('en-US')})` : '';
    const tried = [...new Set([...this.history.filter(h => h.kind === 'switch').map(h => h.variant), this.variant?.id].filter(Boolean))]
      .map(id => this.variants.find(v => v.id === id)?.label ?? (id === 'hw' ? 'hardware' : id === 'sw' ? 'software' : id)).join(', then ');
    const clip = describeStream(this.facts);
    let what: string;
    if (err instanceof TruncatedSampleError) what = `The file ends early: the video data ${where} is missing, so it looks truncated or incompletely copied. Copy the clip from the SD card again.`;
    else if (err instanceof DamagedSampleError) what = `The video data ${where} is blank or damaged, so this copy of the file looks incomplete or corrupted (a copy that didn’t finish, or a cloud-synced file that isn’t fully downloaded). Copy the clip from the SD card again.`;
    else if (err instanceof DecoderStallError) what = `The video decoder stopped responding ${where} and didn’t recover.`;
    else what = `The video decoder failed ${where} and couldn’t recover.`;
    // (a file problem isn't the decoder's fault: no decoder / browser text for those)
    if (err instanceof TruncatedSampleError || err instanceof DamagedSampleError) return new DecodeFailure(`${what} Clip: ${clip}.`, r);
    return new DecodeFailure(`${what} Clip: ${clip}. Decoder: ${tried || 'none'}. Browser said: “${errText(err)}”.`, r);
  }

  /**
   * Pre-flight: decode the clip's first frames with each variant in order until one works; the variants that failed
   * are dropped (the working one becomes current). Catches decoders that isConfigSupported() calls supported but
   * can't decode this clip (e.g. Windows GPU H.264 decoders limited to level 5.1 / 4K30).
   */
  async preflight(signal?: AbortSignal, frames = 10, budgetMs = 15000): Promise<PreflightResult> {
    const t0 = performance.now();
    const n = Math.max(1, Math.min(this.index.n, frames));
    const tried: PreflightResult['tried'] = [];
    for (let i = 0; i < this.variants.length; i++) {
      const v = this.variants[i];
      if (performance.now() - t0 > budgetMs && v.accel !== 'prefer-software') { tried.push({ variant: v.id, ok: false, ms: 0, error: 'skipped (time budget)' }); continue; }
      this.current = i;
      const ts = performance.now();
      const st = this.frames(0, n - 1, { purpose: 'preflight', signal, retries: false, gaps: 'fail', stallMs: this.opts.preflightStallMs ?? 5000, ...this.opts.limits.preview });
      try {
        let got = 0;
        for (;;) {
          const r = await st.next();
          if (!r) break;
          const ok = r.frame.displayWidth > 0 && r.frame.displayHeight > 0;
          r.frame.close();
          if (!ok) throw new Error('the decoder returned an empty picture');
          got++;
        }
        if (got < n) throw new Error(`only ${got} of ${n} test frames came out of the decoder`);
        const ms = Math.round(performance.now() - ts);
        tried.push({ variant: v.id, ok: true, ms });
        this.log({ kind: 'preflight-ok', variant: v.id, ms });
        this.variants = this.variants.slice(i);
        this.current = 0;
        return { ok: true, ms: Math.round(performance.now() - t0), frames: n, tried };
      } catch (e) {
        if (isAbort(e) || signal?.aborted) throw abortError();
        // the file itself is damaged / incomplete right at the start: no other decoder will do better
        if (e instanceof DecodeFailure && (e.report.kind === 'damaged' || e.report.kind === 'truncated')) throw e;
        const inner = e instanceof DecodeFailure ? e.report.error : { name: (e as Error)?.name, message: (e as Error)?.message ?? String(e) };
        const msg = inner.name && inner.name !== 'Error' ? `${inner.name}: ${inner.message}` : inner.message;
        tried.push({ variant: v.id, ok: false, ms: Math.round(performance.now() - ts), error: msg });
        this.log({ kind: 'preflight-fail', variant: v.id, error: msg, ms: Math.round(performance.now() - ts) });
      } finally {
        st.close();
      }
    }
    this.current = Math.max(0, this.variants.length - 1);
    return { ok: false, ms: Math.round(performance.now() - t0), frames: n, tried };
  }
}

export interface DecodedFrame { frame: VideoFrame; pres: number }

/**
 * Presentation-ordered frames of pres range [p0, p1] that survive decoder failures. On an error the decoder is
 * closed; the session falls back to the next variant (sticky) if there is one, else the same config is retried once
 * per failing frame; decoding restarts from the last clean random-access point before the first frame still needed,
 * and frames already delivered are dropped. Truncated data and exhausted options end in a DecodeFailure.
 */
export class FrameStream {
  private dec: SequentialDecoder | null = null;
  private nextPres: number;
  private restarts = 0;
  private switches = 0;
  private failsAt = new Map<number, number>();
  private closed = false;

  constructor(private s: DecoderSession, readonly p0: number, readonly p1: number, private o: FrameStreamOptions) {
    this.nextPres = p0;
  }

  private start() {
    const [d0, d1] = this.s.index.decodeSpan(this.nextPres, this.p1);
    this.dec = this.s.createDecoder(d0, d1, {
      maxFrames: this.o.maxFrames, maxDecodeQueue: this.o.maxDecodeQueue, signal: this.o.signal,
      stallMs: this.o.stallMs ?? this.s.opts.stallMs, lowLatency: this.o.lowLatency,
    });
  }

  private async recover(err: unknown): Promise<void> {
    const fed = this.dec?.lastFed ?? -1;
    this.dec?.close();
    this.dec = null;
    const at = this.nextPres;
    const purpose = this.o.purpose;
    if (err instanceof TruncatedSampleError || err instanceof DamagedSampleError) throw this.s.failure(err, at, purpose);
    // a decoder error right where the file's data goes bad (a copy that stopped mid-sample) is a file problem: say so,
    // and don't switch this session to a slower decoder for it
    const damaged = await this.damagedNear(at, fed);
    if (damaged) throw this.s.failure(damaged, at, purpose);
    if (this.o.retries === false || this.restarts >= 12) throw this.s.failure(err, at, purpose);
    this.restarts++;
    const n = (this.failsAt.get(at) ?? 0) + 1;
    this.failsAt.set(at, n);
    if (this.switches < 2 && this.s.fallBack(err, at, purpose)) { this.switches++; return; }
    if (n <= 1) { this.s.log({ kind: 'retry', variant: this.s.variant.id, purpose, pres: at, error: errText(err) }); return; }
    throw this.s.failure(err, at, purpose);
  }

  /** The first damaged / missing sample among the next few after pres frame `at` (up to `fed` + 4), or null. */
  private async damagedNear(at: number, fed: number): Promise<TruncatedSampleError | DamagedSampleError | null> {
    const t = this.s.track, ix = this.s.index;
    if (codecFamily(t) === 'other' || at < 0 || at >= ix.n) return null;
    const d0 = ix.decOfPres[at];
    const d1 = Math.min(ix.n - 1, Math.max(d0, fed) + 4, d0 + 24);
    try {
      const data = await this.s.readSamples(this.s.file, t, d0, d1 - d0 + 1);
      const ls = nalLengthSize(t);
      for (let k = 0; k < data.length; k++) {
        const d = d0 + k;
        if (!data[k] || data[k].length !== t.sizes[d]) return new TruncatedSampleError(d, data[k]?.length ?? 0, t.sizes[d]);
        if (!wellFormedAccessUnit(data[k], ls)) return new DamagedSampleError(d);
      }
    } catch { /* can't tell: treat it as a decoder problem */ }
    return null;
  }

  async next(): Promise<DecodedFrame | null> {
    for (;;) {
      if (this.o.signal?.aborted) throw abortError();
      if (this.closed || this.nextPres > this.p1) return null;
      if (!this.dec) this.start();
      let f: VideoFrame | null;
      try {
        f = await this.dec!.next();
      } catch (e) {
        if (this.o.signal?.aborted || isAbort(e)) throw abortError();
        await this.recover(e);
        continue;
      }
      if (this.o.signal?.aborted) { f?.close(); throw abortError(); }
      if (!f) {
        if (this.closed) return null;
        // the span ended without the frames we still need
        if (this.o.gaps === 'skip' && this.o.purpose === 'playback') return null;
        await this.recover(new Error(`the decoder returned no picture for frame ${this.nextPres}`));
        continue;
      }
      const pres = this.s.index.presOfTimestamp(f.timestamp);
      if (pres < this.nextPres || pres > this.p1) { f.close(); continue; }
      if (pres > this.nextPres && this.o.gaps !== 'skip') {
        f.close();
        await this.recover(new Error(`the decoder skipped frame ${this.nextPres}`));
        continue;
      }
      this.nextPres = pres + 1;
      return { frame: f, pres };
    }
  }

  close() {
    this.closed = true;
    this.dec?.close();
    this.dec = null;
  }
}

/** Decode a single presentation frame p (random-access point -> p, then flush). Caller closes the result. */
export async function decodeOne(s: DecoderSession, p: number, signal?: AbortSignal): Promise<VideoFrame> {
  const st = s.frames(p, p, { purpose: 'preview', lowLatency: true, signal, gaps: 'fail', ...s.opts.limits.preview });
  try {
    const r = await st.next();
    if (!r) throw new Error(`Frame ${p} could not be decoded`);
    return r.frame;
  } finally { st.close(); }
}
