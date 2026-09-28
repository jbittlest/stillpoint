/**
 * MP4 / MOV sample-table reader for the browser (and Node >= 20 via fs.openAsBlob) — TELEMETRY module.
 *
 * Port of engine/stillpoint/video.py (mp4_tracks, Mp4Track.pts_ticks, read_ranges) plus what WebCodecs needs:
 * the video codec string + avcC/hvcC description, colr, and the AAC AudioSpecificConfig for audio passthrough.
 *
 *  - Works on a File/Blob with small `Blob.slice()` reads: only the top-level box headers and the moov box are read
 *    (moov may sit at the end of a multi-GB file; 64-bit box sizes and co64 chunk offsets are handled).
 *  - Sample times follow ffmpeg/Python semantics: presentation time = dts + ctts + edit-list shift (leading empty
 *    edits delay, the first media_time shifts), exactly as video.py's Mp4Track.pts_ticks().
 *  - readSamples/readRanges coalesce neighbouring byte ranges (gap < 16 KiB, run < 16 MiB) into one read and keep a
 *    few reads in flight; a DJI clip interleaves one ~0.5 KB telemetry sample per ~250 KB video frame, so reading the
 *    telemetry of a 6 GB clip is ~24k small reads, never the whole file.
 */
import type { Mp4Info, Mp4Track } from './types';

// ------------------------------------------------------------------------------------------------ byte reads

/** Bytes [start, end) of a Blob. */
export async function readBytes(file: Blob, start: number, end: number): Promise<Uint8Array> {
  const s = Math.max(0, Math.min(start, file.size));
  const e = Math.max(s, Math.min(end, file.size));
  return new Uint8Array(await file.slice(s, e).arrayBuffer());
}

export const READ_MERGE_GAP = 16 * 1024;
export const READ_MAX_RUN = 16 * 1024 * 1024;
export const READ_CONCURRENCY = 8;

export interface ReadRangesOptions {
  /** merge ranges closer than this (bytes) into one read (default 16 KiB, like video.py) */
  maxGap?: number;
  /** never merge into a single read larger than this (bytes, default 16 MiB) */
  maxRun?: number;
  /** reads in flight (default 8) */
  concurrency?: number;
  /** called with the fraction of bytes read so far (0..1) */
  onProgress?: (f: number) => void;
  /** aborts between reads */
  signal?: AbortSignal;
}

/**
 * Bytes of each (offset, size) range, returned in the order given (duplicates allowed). Ranges within `maxGap` of
 * each other are read with one Blob.slice read; up to `concurrency` reads are in flight.
 */
export async function readRanges(file: Blob, offsets: ArrayLike<number>, sizes: ArrayLike<number>,
                                 opts: ReadRangesOptions = {}): Promise<Uint8Array[]> {
  const n = offsets.length;
  if (sizes.length !== n) throw new Error('readRanges: offsets/sizes length mismatch');
  const out: Uint8Array[] = new Array(n);
  if (n === 0) return out;
  const maxGap = opts.maxGap ?? READ_MERGE_GAP;
  const maxRun = opts.maxRun ?? READ_MAX_RUN;
  const conc = Math.max(1, opts.concurrency ?? READ_CONCURRENCY);
  // sort by offset (stable), then group
  let sorted = true;
  for (let i = 1; i < n; i++) if (offsets[i] < offsets[i - 1]) { sorted = false; break; }
  let order: Uint32Array | null = null;
  if (!sorted) {
    const idx = new Array<number>(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    idx.sort((a, b) => (offsets[a] - offsets[b]) || (a - b));
    order = Uint32Array.from(idx);
  }
  const at = (j: number) => (order ? order[j] : j);
  const groups: Array<{ lo: number; hi: number; first: number; last: number }> = [];
  let g = { lo: offsets[at(0)], hi: offsets[at(0)] + sizes[at(0)], first: 0, last: 0 };
  for (let j = 1; j < n; j++) {
    const i = at(j);
    const o = offsets[i];
    const e = o + sizes[i];
    if (o - g.hi <= maxGap && Math.max(g.hi, e) - g.lo <= maxRun) {
      if (e > g.hi) g.hi = e;
      g.last = j;
    } else {
      groups.push(g);
      g = { lo: o, hi: e, first: j, last: j };
    }
  }
  groups.push(g);
  let total = 0;
  for (const gr of groups) total += gr.hi - gr.lo;
  let done = 0;
  let next = 0;
  const worker = async () => {
    while (true) {
      const gi = next++;
      if (gi >= groups.length) return;
      if (opts.signal?.aborted) throw opts.signal.reason ?? new DOMException('aborted', 'AbortError');
      const gr = groups[gi];
      const buf = await readBytes(file, gr.lo, gr.hi);
      for (let j = gr.first; j <= gr.last; j++) {
        const i = at(j);
        const s = offsets[i] - gr.lo;
        out[i] = buf.subarray(s, s + sizes[i]);
      }
      done += gr.hi - gr.lo;
      opts.onProgress?.(total ? done / total : 1);
    }
  };
  await Promise.all(Array.from({ length: Math.min(conc, groups.length) }, worker));
  return out;
}

/** Bytes of sample i of track t. */
export async function readSample(file: Blob, t: Mp4Track, i: number): Promise<Uint8Array> {
  if (!(i >= 0 && i < t.sampleCount)) throw new RangeError(`sample ${i} out of range [0, ${t.sampleCount})`);
  return readBytes(file, t.offsets[i], t.offsets[i] + t.sizes[i]);
}

/** Bytes of samples [first, first+count) of track t (coalesced range reads). */
export async function readSamples(file: Blob, t: Mp4Track, first: number, count: number,
                                  opts: ReadRangesOptions = {}): Promise<Uint8Array[]> {
  const a = Math.max(0, first | 0);
  const b = Math.min(t.sampleCount, a + Math.max(0, count | 0));
  return readRanges(file, t.offsets.subarray(a, b), t.sizes.subarray(a, b), opts);
}

/** Bytes of samples idx (any order) of track t. */
export async function readSamplesAt(file: Blob, t: Mp4Track, idx: ArrayLike<number>,
                                    opts: ReadRangesOptions = {}): Promise<Uint8Array[]> {
  const o = new Float64Array(idx.length);
  const s = new Float64Array(idx.length);
  for (let j = 0; j < idx.length; j++) {
    const i = idx[j];
    if (!(i >= 0 && i < t.sampleCount)) throw new RangeError(`sample ${i} out of range [0, ${t.sampleCount})`);
    o[j] = t.offsets[i];
    s[j] = t.sizes[i];
  }
  return readRanges(file, o, s, opts);
}

// ------------------------------------------------------------------------------------------------ box parsing

const TD = new TextDecoder('latin1');
const TD8 = new TextDecoder('utf-8');

function fourcc(b: Uint8Array, p: number): string {
  return String.fromCharCode(b[p], b[p + 1], b[p + 2], b[p + 3]);
}

function u32(b: Uint8Array, p: number): number {
  return ((b[p] << 24) >>> 0) + (b[p + 1] << 16) + (b[p + 2] << 8) + b[p + 3];
}
function i32(b: Uint8Array, p: number): number {
  return (b[p] << 24) | (b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3];
}
function u16(b: Uint8Array, p: number): number {
  return (b[p] << 8) | b[p + 1];
}
function i16(b: Uint8Array, p: number): number {
  const v = u16(b, p);
  return v >= 0x8000 ? v - 0x10000 : v;
}
/** unsigned 64-bit big endian (exact below 2^53) */
function u64(b: Uint8Array, p: number): number {
  return u32(b, p) * 4294967296 + u32(b, p + 4);
}
/** signed 64-bit big endian (exact for |v| < 2^53) */
function i64(b: Uint8Array, p: number): number {
  return i32(b, p) * 4294967296 + u32(b, p + 4);
}

interface Box { type: string; start: number; payload: number; end: number }

/** Boxes in buf[start, end) (in-memory). */
function* boxes(buf: Uint8Array, start: number, end: number): Generator<Box> {
  let pos = start;
  while (pos + 8 <= end) {
    let size = u32(buf, pos);
    const type = fourcc(buf, pos + 4);
    let hlen = 8;
    if (size === 1) {
      if (pos + 16 > end) return;
      size = u64(buf, pos + 8);
      hlen = 16;
    } else if (size === 0) {
      size = end - pos;
    }
    if (size < hlen) return;
    yield { type, start: pos, payload: pos + hlen, end: Math.min(pos + size, end) };
    pos += size;
  }
}

function child(buf: Uint8Array, a: number, b: number, type: string): Box | undefined {
  for (const x of boxes(buf, a, b)) if (x.type === type) return x;
  return undefined;
}

/** Python round(): half to even. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
}

function gcd(a: number, b: number): number {
  a = Math.abs(a); b = Math.abs(b);
  while (b) { const t = a % b; a = b; b = t; }
  return a;
}

/** libavutil av_reduce(num, den, max) for non-negative integers. */
export function avReduce(num: number, den: number, max = 2147483647): [number, number] {
  if (den === 0) return [num ? 1 : 0, 0];
  const g = gcd(num, den) || 1;
  num /= g; den /= g;
  if (num <= max && den <= max) return [num, den];
  // continued fraction approximation (as av_reduce)
  let a0n = 0, a0d = 1, a1n = 1, a1d = 0;
  let n = num, d = den;
  while (d) {
    const x = Math.floor(n / d);
    const nextN = x * a1n + a0n;
    const nextD = x * a1d + a0d;
    if (nextN > max || nextD > max) {
      let x2 = x;
      if (a1n) x2 = Math.floor((max - a0n) / a1n);
      if (a1d) x2 = Math.min(x2, Math.floor((max - a0d) / a1d));
      if (den * (2 * x2 * a1d + a0d) > num * a1d) { a1n = x2 * a1n + a0n; a1d = x2 * a1d + a0d; }
      break;
    }
    a0n = a1n; a0d = a1d; a1n = nextN; a1d = nextD;
    const r = n - d * x; n = d; d = r;
  }
  return [a1n, a1d];
}

const CODEC_NAMES: Record<string, string> = {
  avc1: 'h264', avc3: 'h264', hvc1: 'hevc', hev1: 'hevc', av01: 'av1', vp09: 'vp9', vp08: 'vp8',
  mp4v: 'mpeg4', jpeg: 'mjpeg', mjpa: 'mjpeg', apcn: 'prores', apch: 'prores', apcs: 'prores', apco: 'prores',
  ap4h: 'prores', ap4x: 'prores', mp4a: 'aac', 'ac-3': 'ac3', 'ec-3': 'eac3', Opus: 'opus', fLaC: 'flac',
  lpcm: 'pcm', sowt: 'pcm_s16le', twos: 'pcm_s16be', in24: 'pcm_s24', tmcd: 'timecode',
};

const VIDEO_FOURCCS = new Set(['avc1', 'avc3', 'hvc1', 'hev1', 'apcn', 'apch', 'apcs', 'ap4h', 'av01', 'mp4v']);

function hex2(x: number): string {
  return x.toString(16).padStart(2, '0');
}

function reverseBits32(x: number): number {
  let r = 0;
  for (let i = 0; i < 32; i++) { r = (r * 2) + (x & 1); x >>>= 1; }
  return r >>> 0;
}

/** WebCodecs codec string from an avcC / hvcC payload (ISO/IEC 14496-15 Annex E for HEVC). */
export function videoCodecString(entry: string, cfgType: string, c: Uint8Array): string {
  if (cfgType === 'avcC') return `${entry}.${hex2(c[1])}${hex2(c[2])}${hex2(c[3])}`;
  const b = c[1];
  const space = b >> 6, tier = (b >> 5) & 1, prof = b & 0x1f;
  const compat = reverseBits32(u32(c, 2));
  const cons = Array.from(c.subarray(6, 12));
  while (cons.length && cons[cons.length - 1] === 0) cons.pop();
  let s = `${entry}.${['', 'A', 'B', 'C'][space]}${prof}.${compat.toString(16).toUpperCase()}.${tier ? 'H' : 'L'}${c[12]}`;
  if (cons.length) s += '.' + cons.map((x) => x.toString(16).toUpperCase()).join('.');
  return s;
}

/** MPEG-4 descriptor header at p: [tag, payloadStart, payloadEnd]. */
function descriptor(b: Uint8Array, p: number, end: number): [number, number, number] | null {
  if (p + 2 > end) return null;
  const tag = b[p++];
  let len = 0;
  for (let i = 0; i < 4 && p < end; i++) {
    const c = b[p++];
    len = (len << 7) | (c & 0x7f);
    if (!(c & 0x80)) break;
  }
  return [tag, p, Math.min(end, p + len)];
}

interface EsdsInfo { oti: number; asc?: Uint8Array }

function parseEsds(b: Uint8Array, a: number, e: number): EsdsInfo | null {
  let p = a + 4; // FullBox
  const es = descriptor(b, p, e);
  if (!es || es[0] !== 3) return null;
  p = es[1] + 2;
  const flags = b[p++];
  if (flags & 0x80) p += 2;
  if (flags & 0x40) p += 1 + b[p];
  if (flags & 0x20) p += 2;
  while (p < es[2]) {
    const d = descriptor(b, p, es[2]);
    if (!d) break;
    if (d[0] === 4) {
      const oti = b[d[1]];
      let q = d[1] + 13;
      while (q < d[2]) {
        const s = descriptor(b, q, d[2]);
        if (!s) break;
        if (s[0] === 5) return { oti, asc: b.slice(s[1], s[2]) };
        q = s[2];
      }
      return { oti };
    }
    p = d[2];
  }
  return null;
}

const AAC_RATES = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];

/** AudioSpecificConfig -> [audioObjectType, sampleRate, channels] */
function parseAsc(asc: Uint8Array): [number, number, number] {
  let bit = 0;
  const bits = (n: number) => {
    let v = 0;
    for (let i = 0; i < n; i++) {
      const byte = asc[(bit >> 3)] ?? 0;
      v = (v << 1) | ((byte >> (7 - (bit & 7))) & 1);
      bit++;
    }
    return v;
  };
  let aot = bits(5);
  if (aot === 31) aot = 32 + bits(6);
  const fi = bits(4);
  const rate = fi === 15 ? bits(24) : (AAC_RATES[fi] ?? 0);
  const ch = bits(4);
  return [aot, rate, ch];
}

function parseColr(b: Uint8Array, a: number, e: number): Mp4Track['colr'] | undefined {
  if (e - a < 10) return undefined;
  const t = fourcc(b, a);
  if (t === 'nclx' || t === 'nclc') {
    return { primaries: u16(b, a + 4), transfer: u16(b, a + 6), matrix: u16(b, a + 8),
             fullRange: t === 'nclx' && e - a >= 11 ? (b[a + 10] & 0x80) !== 0 : false };
  }
  return undefined;
}

interface RawTrack {
  track: Mp4Track;
  elst: Array<[number, number, number]>;
  dtsTicks: Float64Array;
  ctsOff: Float64Array;
  /** ffmpeg avg_frame_rate inputs: total stts duration (ticks) and sample count */
  durationForFps: number;
  framesForFps: number;
}

function parseStsd(buf: Uint8Array, box: Box, tr: Mp4Track) {
  const n = u32(buf, box.payload + 4);
  if (!n) return;
  const ep = box.payload + 8;
  const esize = u32(buf, ep);
  const etype = fourcc(buf, ep + 4);
  const eend = Math.min(box.end, ep + Math.max(esize, 8));
  tr.fourcc = etype;
  tr.codec = CODEC_NAMES[etype] ?? '';
  const body = ep + 8;
  if (tr.handler === 'vide' && eend - body >= 28) {
    tr.width = u16(buf, body + 24);
    tr.height = u16(buf, body + 26);
    for (const c of boxes(buf, body + 78, eend)) {
      if (c.type === 'avcC' || c.type === 'hvcC') {
        tr.codecConfig = buf.slice(c.payload, c.end);
        tr.codecString = videoCodecString(etype, c.type, tr.codecConfig);
      } else if (c.type === 'av1C' || c.type === 'vpcC') {
        tr.codecConfig = buf.slice(c.payload, c.end);
      } else if (c.type === 'colr') {
        const cr = parseColr(buf, c.payload, c.end);
        if (cr) tr.colr = cr;
      }
    }
  } else if (tr.handler === 'soun' && eend - body >= 20) {
    const ver = u16(buf, body + 8);
    tr.channels = u16(buf, body + 16);
    tr.sampleRate = u32(buf, body + 24) / 65536;
    const kids = body + 28 + (ver === 1 ? 16 : ver === 2 ? 36 : 0);
    let esds: Box | undefined;
    for (const c of boxes(buf, kids, eend)) {
      if (c.type === 'esds') esds = c;
      else if (c.type === 'wave') esds = child(buf, c.payload, c.end, 'esds') ?? esds;
    }
    if (esds) {
      const info = parseEsds(buf, esds.payload, esds.end);
      if (info) {
        if (info.asc && info.asc.length >= 2) {
          const [aot, rate, ch] = parseAsc(info.asc);
          tr.codecConfig = info.asc;
          tr.codecString = `mp4a.${info.oti.toString(16)}.${aot}`;
          if (rate) tr.sampleRate = rate;
          if (ch) tr.channels = ch;
        } else {
          tr.codecString = `mp4a.${info.oti.toString(16)}`;
        }
        if (info.oti === 0x69 || info.oti === 0x6b) tr.codec = 'mp3';
      }
    }
  }
}

function parseStbl(buf: Uint8Array, a: number, b: number, raw: RawTrack) {
  const tr = raw.track;
  let stts: Uint32Array | null = null;
  let ctts: Int32Array | null = null;
  let stsz: Uint32Array | null = null;
  let stsc: Uint32Array | null = null;
  let chunkOff: Float64Array | null = null;
  let stss: Uint32Array | null = null;
  for (const x of boxes(buf, a, b)) {
    const p = x.payload;
    switch (x.type) {
      case 'stsd': parseStsd(buf, x, tr); break;
      case 'stts': {
        const n = Math.min(u32(buf, p + 4), Math.floor((x.end - p - 8) / 8));
        stts = new Uint32Array(2 * n);
        for (let i = 0; i < 2 * n; i++) stts[i] = u32(buf, p + 8 + 4 * i);
        break;
      }
      case 'ctts': {
        // signed like ffmpeg (identical to video.py for every valid file: version-0 offsets are < 2^31)
        const n = Math.min(u32(buf, p + 4), Math.floor((x.end - p - 8) / 8));
        ctts = new Int32Array(2 * n);
        for (let i = 0; i < n; i++) {
          ctts[2 * i] = u32(buf, p + 8 + 8 * i) | 0;
          ctts[2 * i + 1] = i32(buf, p + 12 + 8 * i);
        }
        break;
      }
      case 'stsz': {
        const ss = u32(buf, p + 4);
        const n = u32(buf, p + 8);
        stsz = new Uint32Array(n);
        if (ss) stsz.fill(ss);
        else for (let i = 0; i < n && p + 12 + 4 * i + 4 <= x.end; i++) stsz[i] = u32(buf, p + 12 + 4 * i);
        break;
      }
      case 'stz2': {
        const fs = buf[p + 7];
        const n = u32(buf, p + 8);
        stsz = new Uint32Array(n);
        for (let i = 0; i < n; i++) {
          const q = p + 12;
          stsz[i] = fs === 8 ? buf[q + i] : fs === 16 ? u16(buf, q + 2 * i)
            : ((buf[q + (i >> 1)] >> ((i & 1) ? 0 : 4)) & 15);
        }
        break;
      }
      case 'stsc': {
        const n = Math.min(u32(buf, p + 4), Math.floor((x.end - p - 8) / 12));
        stsc = new Uint32Array(3 * n);
        for (let i = 0; i < 3 * n; i++) stsc[i] = u32(buf, p + 8 + 4 * i);
        break;
      }
      case 'stco': {
        const n = Math.min(u32(buf, p + 4), Math.floor((x.end - p - 8) / 4));
        chunkOff = new Float64Array(n);
        for (let i = 0; i < n; i++) chunkOff[i] = u32(buf, p + 8 + 4 * i);
        break;
      }
      case 'co64': {
        const n = Math.min(u32(buf, p + 4), Math.floor((x.end - p - 8) / 8));
        chunkOff = new Float64Array(n);
        for (let i = 0; i < n; i++) chunkOff[i] = u64(buf, p + 8 + 8 * i);
        break;
      }
      case 'stss': {
        const n = Math.min(u32(buf, p + 4), Math.floor((x.end - p - 8) / 4));
        stss = new Uint32Array(n);
        for (let i = 0; i < n; i++) stss[i] = u32(buf, p + 8 + 4 * i);
        break;
      }
    }
  }
  if (!stsz || !stts) return;
  const ns = stsz.length;
  tr.sampleCount = ns;
  tr.sizes = stsz;
  // decode times (ticks) from stts; a table shorter than the sample count repeats its last delta
  const dts = new Float64Array(ns);
  let t = 0, k = 0, lastDelta = 0, durSum = 0, durCount = 0;
  const sttsArr = stts as Uint32Array;
  for (let e = 0; e < sttsArr.length / 2; e++) {
    const cnt = sttsArr[2 * e], d = sttsArr[2 * e + 1];
    durSum += cnt * d;
    durCount += cnt;
    for (let j = 0; j < cnt && k < ns; j++) { dts[k++] = t; t += d; }
    lastDelta = d;
  }
  while (k < ns) { dts[k++] = t; t += lastDelta; }
  raw.dtsTicks = dts;
  raw.durationForFps = durCount > 0 && durSum > 0 ? durSum : 0;
  raw.framesForFps = durCount;
  const co = new Float64Array(ns);
  if (ctts) {
    const c = ctts as Int32Array;
    let q = 0;
    for (let e = 0; e < c.length / 2 && q < ns; e++) {
      const cnt = c[2 * e] >>> 0, v = c[2 * e + 1];
      for (let j = 0; j < cnt && q < ns; j++) co[q++] = v;
    }
  }
  raw.ctsOff = co;
  const sync = new Uint8Array(ns);
  if (stss) {
    for (const s of stss as Uint32Array) if (s >= 1 && s <= ns) sync[s - 1] = 1;
  } else {
    sync.fill(1);
  }
  tr.sync = sync;
  // sample -> chunk -> file offset
  const offs = new Float64Array(ns).fill(NaN);
  if (chunkOff && stsc && stsc.length) {
    const nch = chunkOff.length;
    const sc = stsc as Uint32Array;
    const ne = sc.length / 3;
    let s = 0;
    for (let e = 0; e < ne && s < ns; e++) {
      const firstChunk = sc[3 * e] - 1;
      const lastChunk = e + 1 < ne ? sc[3 * (e + 1)] - 1 : nch;
      const spc = sc[3 * e + 1];
      for (let ch = Math.max(0, firstChunk); ch < Math.min(lastChunk, nch) && s < ns; ch++) {
        let o = chunkOff[ch];
        for (let j = 0; j < spc && s < ns; j++) { offs[s] = o; o += stsz[s]; s++; }
      }
    }
  }
  tr.offsets = offs;
}

function trackKind(handler: string): Mp4Track['kind'] {
  if (handler === 'vide') return 'video';
  if (handler === 'soun') return 'audio';
  if (handler === 'meta' || handler === 'data' || handler === 'text' || handler === 'sbtl' || handler === 'subt') return 'data';
  return 'other';
}

function parseTrak(buf: Uint8Array, trak: Box, movieTs: number): RawTrack {
  const tr: Mp4Track = {
    id: 0, kind: 'other', handler: '', codec: '', fourcc: '', timescale: 1, sampleCount: 0,
    offsets: new Float64Array(0), sizes: new Uint32Array(0), dts: new Float64Array(0), cts: new Float64Array(0),
    sync: new Uint8Array(0), handlerName: '',
  };
  const raw: RawTrack = { track: tr, elst: [], dtsTicks: new Float64Array(0), ctsOff: new Float64Array(0),
                         durationForFps: 0, framesForFps: 0 };
  let stbl: Box | undefined;
  let mdhdDur = 0;
  for (const x of boxes(buf, trak.payload, trak.end)) {
    const p = x.payload;
    if (x.type === 'tkhd') {
      tr.id = buf[p] === 1 ? u32(buf, p + 20) : u32(buf, p + 12);
    } else if (x.type === 'edts') {
      const el = child(buf, x.payload, x.end, 'elst');
      if (el) {
        const q = el.payload;
        const ver = buf[q];
        const n = u32(buf, q + 4);
        for (let i = 0; i < n; i++) {
          const r = q + 8 + (ver === 1 ? 20 : 12) * i;
          if (r + (ver === 1 ? 20 : 12) > el.end) break;
          const dur = ver === 1 ? u64(buf, r) : u32(buf, r);
          const mt = ver === 1 ? i64(buf, r + 8) : i32(buf, r + 4);
          const ro = r + (ver === 1 ? 16 : 8);
          raw.elst.push([dur, mt, i16(buf, ro) + i16(buf, ro + 2) / 65536]);
        }
      }
    } else if (x.type === 'mdia') {
      for (const y of boxes(buf, x.payload, x.end)) {
        const q = y.payload;
        if (y.type === 'mdhd') {
          if (buf[q] === 1) { tr.timescale = u32(buf, q + 20); mdhdDur = u64(buf, q + 24); }
          else { tr.timescale = u32(buf, q + 12); mdhdDur = u32(buf, q + 16); }
        } else if (y.type === 'hdlr') {
          tr.handler = fourcc(buf, q + 8);
          const nameBytes = buf.subarray(q + 24, y.end);
          const z = nameBytes.indexOf(0);
          tr.handlerName = TD8.decode(z >= 0 ? nameBytes.subarray(0, z) : nameBytes).trim();
        } else if (y.type === 'minf') {
          stbl = child(buf, y.payload, y.end, 'stbl');
        }
      }
    }
  }
  tr.kind = trackKind(tr.handler);
  if (tr.timescale) tr.durationS = mdhdDur / tr.timescale;
  if (stbl) parseStbl(buf, stbl.payload, stbl.end, raw);
  // presentation time = dts + ctts + edit shift (video.py Mp4Track.pts_ticks semantics)
  let shift = 0;
  for (const [dur, mt] of raw.elst) {
    if (mt === -1) shift += roundHalfEven(dur * tr.timescale / Math.max(movieTs, 1));
    else { shift -= mt; break; }
  }
  const ns = tr.sampleCount;
  const dts = new Float64Array(ns), cts = new Float64Array(ns);
  const ts = tr.timescale || 1;
  for (let i = 0; i < ns; i++) {
    dts[i] = (raw.dtsTicks[i] + shift) / ts;
    cts[i] = (raw.dtsTicks[i] + raw.ctsOff[i] + shift) / ts;
  }
  tr.dts = dts;
  tr.cts = cts;
  if (tr.kind === 'video' && raw.durationForFps > 0) {
    const [num, den] = avReduce(ts * raw.framesForFps, raw.durationForFps);
    tr.frameRate = { num, den };
  }
  return raw;
}

function parseIlst(buf: Uint8Array, meta: Box, out: { comment?: string; encoder?: string }) {
  // udta/meta is a FullBox in ISO files; QuickTime writers omit the version field
  let a = meta.payload;
  if (fourcc(buf, a + 4) !== 'hdlr') a += 4;
  const ilst = child(buf, a, meta.end, 'ilst');
  if (!ilst) return;
  for (const item of boxes(buf, ilst.payload, ilst.end)) {
    const key = item.type;
    if (key !== '©cmt' && key !== '©too' && key !== '©enc') continue;
    const data = child(buf, item.payload, item.end, 'data');
    if (!data) continue;
    const s = TD8.decode(buf.subarray(data.payload + 8, data.end)).replace(/\0+$/, '');
    if (key === '©cmt') out.comment ??= s;
    else out.encoder ??= s;
  }
}

/** Parse the moov box of an MP4/MOV file (reads only the top-level box headers and moov). */
export async function openMp4(file: Blob): Promise<Mp4Info> {
  const size = file.size;
  let pos = 0;
  let moov: Uint8Array | null = null;
  let brand = '';
  while (pos + 8 <= size) {
    const h = await readBytes(file, pos, Math.min(size, pos + 16));
    let bsize = u32(h, 0);
    const type = fourcc(h, 4);
    if (bsize === 1) bsize = u64(h, 8);
    else if (bsize === 0) bsize = size - pos;
    if (bsize < 8) break;
    if (type === 'ftyp') {
      const f = await readBytes(file, pos + 8, pos + Math.min(bsize, 64));
      brand = TD.decode(f.subarray(0, 4));
    } else if (type === 'moov') {
      moov = await readBytes(file, pos, pos + bsize);
      break;
    }
    pos += bsize;
  }
  if (!moov) throw new Error('not an MP4/MOV file (no moov box)');
  const root: Box = boxes(moov, 0, moov.length).next().value as Box;
  let movieTs = 1, movieDur = 0;
  const tracks: Mp4Track[] = [];
  const tags: { comment?: string; encoder?: string } = {};
  for (const x of boxes(moov, root.payload, root.end)) {
    if (x.type === 'mvhd') {
      const p = x.payload;
      if (moov[p] === 1) { movieTs = u32(moov, p + 20); movieDur = u64(moov, p + 24); }
      else { movieTs = u32(moov, p + 12); movieDur = u32(moov, p + 16); }
    }
  }
  for (const x of boxes(moov, root.payload, root.end)) {
    if (x.type === 'trak') tracks.push(parseTrak(moov, x, movieTs).track);
    else if (x.type === 'udta') {
      for (const y of boxes(moov, x.payload, x.end)) {
        if (y.type === 'meta') parseIlst(moov, y, tags);
      }
    }
  }
  return { tracks, durationS: movieTs ? movieDur / movieTs : 0, movieTimescale: movieTs, brand,
           comment: tags.comment ?? '', encoder: tags.encoder ?? '' };
}

// ------------------------------------------------------------------------------------------------ helpers

/** The main video track (video.py main_video_track: largest coded area, then most samples). */
export function mainVideoTrack(info: Mp4Info): Mp4Track {
  let vids = info.tracks.filter((t) => t.handler === 'vide' && VIDEO_FOURCCS.has(t.fourcc));
  if (!vids.length) vids = info.tracks.filter((t) => t.handler === 'vide');
  if (!vids.length) throw new Error('no video track');
  return vids.reduce((a, b) => {
    const ka = (a.width ?? 0) * (a.height ?? 0), kb = (b.width ?? 0) * (b.height ?? 0);
    return kb > ka || (kb === ka && b.sampleCount > a.sampleCount) ? b : a;
  });
}

/** The main audio track (first 'soun' track with samples), if any. */
export function mainAudioTrack(info: Mp4Info): Mp4Track | undefined {
  return info.tracks.find((t) => t.handler === 'soun' && t.sampleCount > 0);
}

/** video.py find_track: by sample-entry fourcc, then hdlr name, then handler type (with > 1 sample). */
export function findTrack(info: Mp4Info, q: { fourcc?: string; handlerName?: string; handler?: string }): Mp4Track | undefined {
  if (q.fourcc !== undefined) { const t = info.tracks.find((x) => x.fourcc === q.fourcc); if (t) return t; }
  if (q.handlerName !== undefined) { const t = info.tracks.find((x) => x.handlerName === q.handlerName); if (t) return t; }
  if (q.handler !== undefined) { const t = info.tracks.find((x) => x.handler === q.handler && x.sampleCount > 1); if (t) return t; }
  return undefined;
}

/** Presentation times (s) of a track's samples, sorted (= the frame PTS list for a video track). */
export function presentationTimes(t: Mp4Track): Float64Array {
  const c = Float64Array.from(t.cts);
  let sorted = true;
  for (let i = 1; i < c.length; i++) if (c[i] < c[i - 1]) { sorted = false; break; }
  if (!sorted) c.sort();
  return c;
}

/** Container frame rate as ffprobe reports avg_frame_rate (timescale * frames / total stts duration, reduced). */
export function trackFps(t: Mp4Track): number {
  if (t.frameRate && t.frameRate.den) return t.frameRate.num / t.frameRate.den;
  if (t.sampleCount > 1) return (t.sampleCount - 1) / (t.cts[t.sampleCount - 1] - t.cts[0]);
  return 0;
}

/** WebCodecs VideoDecoderConfig for a video track (codec string, description, coded size, colour space). */
export function videoDecoderConfig(t: Mp4Track): VideoDecoderConfig {
  if (!t.codecString) throw new Error(`track ${t.id} (${t.fourcc}): no WebCodecs codec string`);
  const cfg: VideoDecoderConfig = { codec: t.codecString, codedWidth: t.width, codedHeight: t.height };
  if (t.codecConfig) cfg.description = t.codecConfig;
  if (t.colr) {
    const P: Record<number, VideoColorPrimaries> = { 1: 'bt709', 5: 'bt470bg', 6: 'smpte170m', 9: 'bt2020' as VideoColorPrimaries, 12: 'smpte432' as VideoColorPrimaries };
    const T: Record<number, VideoTransferCharacteristics> = { 1: 'bt709', 6: 'smpte170m', 13: 'iec61966-2-1', 8: 'linear' as VideoTransferCharacteristics, 16: 'pq' as VideoTransferCharacteristics, 18: 'hlg' as VideoTransferCharacteristics };
    const M: Record<number, VideoMatrixCoefficients> = { 0: 'rgb', 1: 'bt709', 5: 'bt470bg', 6: 'smpte170m', 9: 'bt2020-ncl' as VideoMatrixCoefficients };
    cfg.colorSpace = { primaries: P[t.colr.primaries] ?? null, transfer: T[t.colr.transfer] ?? null,
                       matrix: M[t.colr.matrix] ?? null, fullRange: t.colr.fullRange };
  }
  return cfg;
}

/** WebCodecs AudioDecoderConfig for an AAC track (for passthrough muxing / decoding). */
export function audioDecoderConfig(t: Mp4Track): AudioDecoderConfig {
  if (!t.codecString) throw new Error(`track ${t.id} (${t.fourcc}): no WebCodecs codec string`);
  const cfg: AudioDecoderConfig = { codec: t.codecString, sampleRate: t.sampleRate ?? 48000,
                                    numberOfChannels: t.channels ?? 2 };
  if (t.codecConfig) cfg.description = t.codecConfig;
  return cfg;
}
