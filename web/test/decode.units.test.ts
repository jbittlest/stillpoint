// DECODER unit tests (node): NAL parsing + random-access classification, decode spans around CRA leading pictures,
// Annex B conversion, decoder variant / fallback ordering, messages, truncated-file helpers, and the FrameStream
// recovery logic against a mock WebCodecs VideoDecoder (fallback hardware -> software, restart at the last clean
// random-access point, exactly-once in-order delivery, precise failure, stall watchdog, truncated data).
// Real-clip checks (DJI O3 H.264, Osmo Action 4 HEVC, iPhone HEVC) run when the clips are present.
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { closeSync, existsSync, openSync, readSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import {
  DamagedSampleError, DecodeFailure, DecoderSession, FrameIndex, RAP_BAD, RAP_CLEAN, RAP_LEADING, TruncatedSampleError, accessUnitKind,
  cannotDecodeMessage, classifySyncSamples, completeSamplePrefix, decodeLimits, decoderVariants, describeStream,
  fallbackIndex, isApplePlatform, nalTypes, parameterSets, parseFault, softwareNote, splitNals, streamFacts, toAnnexB,
  trimTrack, wellFormedAccessUnit, type RapKind,
} from '../src/io/decode';
import { openMp4, mainVideoTrack, readRanges, readSamples } from '../src/mp4';
import type { Mp4Track } from '../src/types';

// ───────── helpers ─────────

/** length-prefixed access unit from NAL payloads */
function au(...nals: number[][]): Uint8Array {
  const out: number[] = [];
  for (const n of nals) out.push((n.length >>> 24) & 255, (n.length >>> 16) & 255, (n.length >>> 8) & 255, n.length & 255, ...n);
  return Uint8Array.from(out);
}
// H.264 NALs: header byte, then slice header bits (first_mb_in_slice = 0 -> '1', slice_type ue(v))
const AVC_IDR = [0x65, 0x88, 0x80];        // IDR slice, slice_type 7 (I)
const AVC_I = [0x61, 0x88, 0x80];          // non-IDR slice, slice_type 7 (I)
const AVC_P = [0x61, 0x98, 0x80];          // non-IDR slice, slice_type 5 (P): ue(5) = 00110
const AVC_SEI_RECOVERY = [0x06, 0x06, 0x01, 0x84, 0x80];
const AVC_SPS = [0x67, 0x64, 0x00, 0x34], AVC_PPS = [0x68, 0xee, 0x3c, 0x80];
const hevc = (t: number, ...rest: number[]) => [(t << 1) & 0x7e, 0x01, ...rest];

function track(cts: number[], sync: number[], extra: Partial<Mp4Track> = {}): Mp4Track {
  const n = cts.length;
  const sizes = new Uint32Array(n).fill(1000);
  return {
    id: 1, kind: 'video', handler: 'vide', codec: 'h264', fourcc: 'avc1', codecString: 'avc1.640034', timescale: 60000, sampleCount: n,
    offsets: Float64Array.from({ length: n }, (_, i) => 1000 + i * 1000), sizes,
    dts: Float64Array.from(cts.map((_, i) => i / 60)), cts: Float64Array.from(cts), sync: Uint8Array.from(sync),
    width: 3840, height: 2160, ...extra,
  };
}

/** a Blob-like view of a file on disk (Node's openAsBlob truncates sizes > 4 GiB) */
function diskBlob(path: string): Blob {
  const size = statSync(path).size;
  const mk = (a: number, b: number): any => ({
    size: Math.max(0, b - a),
    async arrayBuffer() {
      const n = Math.max(0, Math.min(b, size) - a);
      const buf = Buffer.alloc(n);
      const fd = openSync(path, 'r');
      try { let got = 0; while (got < n) { const r = readSync(fd, buf, got, n - got, a + got); if (!r) break; got += r; } } finally { closeSync(fd); }
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + n);
    },
    slice(x = 0, y = b - a) { return mk(a + x, Math.min(b, a + y)); },
  });
  return mk(0, size) as Blob;
}

// ───────── NAL parsing ─────────

describe('NAL parsing / random-access kinds', () => {
  it('splits length-prefixed NAL units', () => {
    const a = au(AVC_SPS, AVC_PPS, AVC_IDR);
    expect(splitNals(a).map(n => n.size)).toEqual([4, 4, 3]);
    expect(nalTypes(a, 'h264')).toEqual([7, 8, 5]);
    expect(nalTypes(au(hevc(32), hevc(33), hevc(34), hevc(20, 0xaf)), 'hevc')).toEqual([32, 33, 34, 20]);
  });
  it('H.264: IDR is clean; non-IDR I (with or without recovery point) and P are not starts', () => {
    expect(accessUnitKind(au(AVC_SPS, AVC_PPS, AVC_IDR), 'h264')).toBe('idr');
    expect(accessUnitKind(au(AVC_SEI_RECOVERY, AVC_I), 'h264')).toBe('recovery');
    expect(accessUnitKind(au(AVC_I), 'h264')).toBe('intra');
    expect(accessUnitKind(au(AVC_P), 'h264')).toBe('none');
    expect(accessUnitKind(new Uint8Array(0), 'h264')).toBe('unknown');
    // a head read that stops inside a big SEI can't tell
    expect(accessUnitKind(au([0x06, ...new Array(100).fill(0x05)]).subarray(0, 40), 'h264')).toBe('unknown');
  });
  it('HEVC: IDR_W_RADL / IDR_N_LP clean, CRA / BLA need leading pictures dropped, TRAIL is not a start', () => {
    expect(accessUnitKind(au(hevc(32), hevc(33), hevc(34), hevc(19)), 'hevc')).toBe('idr');
    expect(accessUnitKind(au(hevc(39), hevc(20)), 'hevc')).toBe('idr');
    expect(accessUnitKind(au(hevc(39), hevc(21)), 'hevc')).toBe('cra');
    expect(accessUnitKind(au(hevc(16)), 'hevc')).toBe('bla');
    expect(accessUnitKind(au(hevc(1)), 'hevc')).toBe('none');
    expect(accessUnitKind(au(hevc(8)), 'hevc')).toBe('none');
  });
  it('Annex B conversion with parameter sets from avcC / hvcC', () => {
    const avcC = Uint8Array.from([1, 0x64, 0, 0x34, 0xff, 0xe1, 0, 4, ...AVC_SPS, 1, 0, 4, ...AVC_PPS]);
    const t = track([0], [1], { codecConfig: avcC });
    const ps = parameterSets(t);
    expect(ps.map(p => Array.from(p))).toEqual([AVC_SPS, AVC_PPS]);
    const b = toAnnexB(au(AVC_IDR), 4, ps);
    expect(Array.from(b)).toEqual([0, 0, 0, 1, ...AVC_SPS, 0, 0, 0, 1, ...AVC_PPS, 0, 0, 0, 1, ...AVC_IDR]);
    // hvcC: 23-byte header + arrays (VPS, SPS, PPS)
    const hdr = new Array(22).fill(0); hdr[0] = 1; hdr[1] = 2; hdr[12] = 153; hdr[17] = 0xfa; hdr[21] = 0x0f;
    const arr = (t: number, nal: number[]) => [0x80 | t, 0, 1, 0, nal.length, ...nal];
    const hvcC = Uint8Array.from([...hdr, 3, ...arr(32, hevc(32, 1)), ...arr(33, hevc(33, 2)), ...arr(34, hevc(34, 3))]);
    const ht = track([0], [1], { codec: 'hevc', fourcc: 'hvc1', codecString: 'hvc1.2.4.L153.B0', codecConfig: hvcC });
    expect(parameterSets(ht).map(p => (p[0] >> 1) & 63)).toEqual([32, 33, 34]);
    const f = streamFacts(ht, 59.94);
    expect(f).toMatchObject({ family: 'hevc', bitDepth: 10, profile: 'Main 10', level: '5.1' });
  });
});

// ───────── frame index / decode spans ─────────

describe('decode spans around CRA leading pictures', () => {
  // decode order: IDR(pts0) P1 P2 P3 | CRA(pts6) RASL(pts4) RASL(pts5) | TRAIL(7) TRAIL(8)
  const t = track([0, 1, 2, 3, 6, 4, 5, 7, 8], [1, 0, 0, 0, 1, 0, 0, 0, 0], { codec: 'hevc', fourcc: 'hvc1', codecString: 'hvc1.1.6.L150.B0' });
  it('before classification every sync sample is a start (old behaviour)', () => {
    const ix = new FrameIndex(t);
    expect(ix.decodeSpan(4, 5)).toEqual([4, 6]);
  });
  it('after classification a span that needs RASL pictures starts at the previous clean point', () => {
    const ix = new FrameIndex(t);
    ix.applyRapKinds(new Map<number, RapKind>([[0, 'idr'], [4, 'cra']]));
    expect(ix.rap[4]).toBe(RAP_LEADING);
    expect(ix.leadingEnd(4)).toBe(7);
    expect(ix.decodeSpan(6, 8)).toEqual([4, 8]);  // trailing pictures only: CRA is fine
    expect(ix.skipAfterStart(4)).toBe(7);          // and its RASL pictures are not fed
    expect(ix.decodeSpan(4, 5)).toEqual([0, 6]);   // the RASL pictures themselves: from the IDR
    expect(ix.decodeSpan(5, 7)).toEqual([0, 7]);
    expect(ix.keyframeFor(5)).toBe(0);
    expect(ix.chunkType(4, 0)).toBe('key');
    expect(ix.chunkType(5, 0)).toBe('delta');
    expect(ix.rapCensus).toEqual({ idr: 1, cra: 1 });
  });
  it('H.264 recovery-point I frames are never a start; they are sent as delta', () => {
    const tt = track([0, 1, 2, 3, 4, 5], [1, 0, 0, 1, 0, 0]);
    const ix = new FrameIndex(tt);
    ix.applyRapKinds(new Map<number, RapKind>([[0, 'idr'], [3, 'recovery']]));
    expect(ix.rap[3]).toBe(RAP_BAD);
    expect(ix.decodeSpan(4, 5)).toEqual([0, 5]);
    expect(ix.chunkType(3, 0)).toBe('delta');
    // no clean point at all before the frame: fall back to the nearest sync sample
    const ix2 = new FrameIndex(tt);
    ix2.applyRapKinds(new Map<number, RapKind>([[0, 'intra'], [3, 'recovery']]));
    expect(ix2.decodeSpan(4, 5)).toEqual([3, 5]);
    expect(ix2.chunkType(3, 3)).toBe('key');
  });
  it('unknown kinds keep trusting the container', () => {
    const ix = new FrameIndex(track([0, 1, 2, 3], [1, 0, 1, 0]));
    ix.applyRapKinds(new Map<number, RapKind>([[2, 'unknown']]));
    expect(ix.rap[2]).toBe(RAP_CLEAN);
    expect(ix.decodeSpan(3, 3)).toEqual([2, 3]);
  });
});

// ───────── variants / fallback / messages ─────────

describe('decoder variants and fallback order', () => {
  const avc = track([0], [1], { codecConfig: Uint8Array.from([1, 0x64, 0, 0x34, 0xff, 0xe1, 0, 4, ...AVC_SPS, 1, 0, 4, ...AVC_PPS]), colr: { primaries: 1, transfer: 1, matrix: 1, fullRange: false } });
  const hvc = track([0], [1], { codec: 'hevc', fourcc: 'hvc1', codecString: 'hvc1.2.4.L150.90', codecConfig: new Uint8Array(23), colr: { primaries: 9, transfer: 18, matrix: 9, fullRange: false } });
  it('H.264: hardware -> automatic -> software', () => {
    const v = decoderVariants(avc);
    expect(v.map(x => x.id)).toEqual(['hw', 'auto', 'sw']);
    expect(v.map(x => x.config.hardwareAcceleration)).toEqual(['prefer-hardware', 'no-preference', 'prefer-software']);
    expect(v[0].config.description).toBe(avc.codecConfig);
    // mid-stream failure of hardware or automatic goes straight to software; software has nothing left
    expect(fallbackIndex(v, 0)).toBe(2);
    expect(fallbackIndex(v, 1)).toBe(2);
    expect(fallbackIndex(v, 2)).toBe(-1);
  });
  it('HEVC: hardware config variants (no software decoder in Chrome/Edge)', () => {
    const v = decoderVariants(hvc);
    expect(v.map(x => x.id)).toEqual(['hw', 'hw-nocolor', 'hw-hev1', 'hw-annexb', 'auto', 'sw']);
    expect(v[1].config.colorSpace).toBeUndefined();
    expect(v[2].config.codec).toBe('hev1.2.4.L150.90');
    expect(v[3]).toMatchObject({ annexB: true });
    expect(v[3].config.description).toBeUndefined();
    const noSw = v.filter(x => x.id !== 'sw'); // what isConfigSupported leaves in Chrome
    expect(fallbackIndex(noSw, 0)).toBe(1);
    expect(fallbackIndex(noSw, 4)).toBe(-1);
  });
  it('facts, messages and limits', () => {
    const f = streamFacts(avc, 60000 / 1001);
    expect(f).toMatchObject({ family: 'h264', profile: 'High', level: '5.2', bitDepth: 8 });
    expect(describeStream(f)).toBe('3840×2160 H.264 (High, level 5.2, 59.94 fps)');
    expect(softwareNote(f)).toMatch(/software.*slower/);
    const hf = { ...streamFacts(hvc, 59.94), width: 3840, height: 2880, bitDepth: 10 };
    const m = cannotDecodeMessage(hf, 'unsupported');
    expect(m).toMatch(/can’t play 3840×2880 HEVC 10-bit/);
    expect(m).toMatch(/NVIDIA RTX, Intel Arc or AMD RDNA2/);
    expect(cannotDecodeMessage(f, 'failed')).toMatch(/H\.264.*High, level 5\.2/);
    expect(decodeLimits(true).export.maxFrames).toBeGreaterThan(decodeLimits(false).export.maxFrames);
    expect(isApplePlatform('Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/153', 'Win32')).toBe(false);
    expect(isApplePlatform('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 'MacIntel')).toBe(true);
  });
  it('fault specs', () => {
    expect(parseFault('hw-first')).toEqual({ scope: 'hw', mode: 'error', at: 0, firstOnly: true });
    expect(parseFault('all-at:40')).toEqual({ scope: 'all', mode: 'error', at: 40, firstOnly: false });
    expect(parseFault('hw-hang-at:7')).toEqual({ scope: 'hw', mode: 'hang', at: 7, firstOnly: false });
    expect(parseFault('nonsense')).toBeNull();
    expect(parseFault(null)).toBeNull();
  });
  it('truncated files keep the complete prefix', () => {
    const t = track([0, 1, 2, 3, 4], [1, 0, 0, 0, 0]);
    expect(completeSamplePrefix(t, 1000 + 3 * 1000)).toBe(3);
    expect(completeSamplePrefix(t, 1e9)).toBe(5);
    const tt = trimTrack(t, 3);
    expect(tt.sampleCount).toBe(3);
    expect(Array.from(tt.cts)).toEqual([0, 1, 2]);
  });
});

// ───────── FrameStream against a mock VideoDecoder ─────────

type Behaviour = (cfg: VideoDecoderConfig, tsUs: number, instance: number) => 'ok' | 'error' | 'hang';
const G = globalThis as any;
const saved = { VideoDecoder: G.VideoDecoder, EncodedVideoChunk: G.EncodedVideoChunk };
let behaviour: Behaviour = () => 'ok';
let created: VideoDecoderConfig[] = [];
let firstChunks: Array<{ type: string; ts: number }> = [];

class MockFrame {
  closed = false;
  displayWidth = 3840; displayHeight = 2160;
  constructor(readonly timestamp: number) {}
  close() { this.closed = true; }
}
class MockChunk { constructor(readonly init: { type: string; timestamp: number; data: Uint8Array }) {} get type() { return this.init.type; } get timestamp() { return this.init.timestamp; } }
class MockDecoder {
  state = 'unconfigured';
  decodeQueueSize = 0;
  private dq: (() => void) | null = null;
  private cfg!: VideoDecoderConfig;
  private inst = -1;
  private needKey = true;
  private hung = false;
  constructor(private cb: { output: (f: any) => void; error: (e: any) => void }) {}
  addEventListener(type: string, fn: () => void) { if (type === 'dequeue') this.dq = fn; }
  configure(cfg: VideoDecoderConfig) { this.cfg = cfg; this.state = 'configured'; this.inst = created.length; created.push(cfg); }
  decode(chunk: MockChunk) {
    if (this.state !== 'configured') throw new DOMException('decoder closed', 'InvalidStateError');
    if (this.needKey) { firstChunks.push({ type: chunk.type, ts: chunk.timestamp }); if (chunk.type !== 'key') throw new DOMException('A key frame is required', 'DataError'); }
    this.needKey = false;
    this.decodeQueueSize++;
    setTimeout(() => {
      this.decodeQueueSize--;
      if (this.state === 'closed' || this.hung) return;
      this.dq?.();
      const b = behaviour(this.cfg, chunk.timestamp, this.inst);
      if (b === 'error') { this.state = 'closed'; this.cb.error(new DOMException('Decoding error.', 'EncodingError')); return; }
      if (b === 'hang') { this.hung = true; return; }
      this.cb.output(new MockFrame(chunk.timestamp));
    }, 0);
  }
  flush() {
    return new Promise<void>((res, rej) => {
      const tick = () => {
        if (this.state === 'closed') { rej(new DOMException('closed', 'AbortError')); return; }
        if (this.hung) return; // never resolves
        if (this.decodeQueueSize) { setTimeout(tick, 1); return; }
        res();
      };
      setTimeout(tick, 1);
    });
  }
  close() { this.state = 'closed'; }
}

describe('FrameStream recovery (mock decoder)', () => {
  const N = 100;
  const fps = 60000 / 1001;
  const t = track(Array.from({ length: N }, (_, i) => i / fps), Array.from({ length: N }, (_, i) => (i % 30 === 0 ? 1 : 0)));
  /** a well-formed one-NAL access unit of n bytes (the decoder mock ignores the payload) */
  const sampleBytes = (n: number) => { const b = new Uint8Array(n).fill(0x11); const len = n - 4; b.set([len >>> 24, (len >>> 16) & 255, (len >>> 8) & 255, len & 255, 0x65], 0); return b; };
  const reads = (short = -1, blank = -1) => async (_f: Blob, tr: Mp4Track, first: number, count: number) =>
    Array.from({ length: count }, (_, k) => (first + k === short ? new Uint8Array(10) : first + k >= blank && blank >= 0 ? new Uint8Array(tr.sizes[first + k]) : sampleBytes(tr.sizes[first + k])));
  const limits = decodeLimits(false);
  const mkSession = (variantsIds = ['hw', 'auto', 'sw'], short = -1, stallMs = 2000, blank = -1) => {
    const ix = new FrameIndex(t);
    const vs = decoderVariants(t).filter(v => variantsIds.includes(v.id));
    return new DecoderSession({ size: 1e9 } as Blob, t, ix, reads(short, blank), vs, { limits, stallMs, preflightStallMs: 300 });
  };
  const drain = async (s: DecoderSession, p0: number, p1: number, purpose: 'export' | 'playback' = 'export') => {
    const st = s.frames(p0, p1, { purpose, gaps: purpose === 'export' ? 'fail' : 'skip', ...limits.export });
    const got: number[] = [];
    try { for (;;) { const r = await st.next(); if (!r) break; got.push(r.pres); r.frame.close(); } } finally { st.close(); }
    return got;
  };
  const ts = (p: number) => Math.round((p / fps) * 1e6);

  beforeEach(() => {
    G.VideoDecoder = MockDecoder; G.EncodedVideoChunk = MockChunk;
    created = []; firstChunks = []; behaviour = () => 'ok';
  });
  afterEach(() => { G.VideoDecoder = saved.VideoDecoder; G.EncodedVideoChunk = saved.EncodedVideoChunk; });

  it('delivers every frame once, in order, starting spans at a key chunk', async () => {
    const s = mkSession();
    expect(await drain(s, 10, 70)).toEqual(Array.from({ length: 61 }, (_, i) => 10 + i));
    expect(firstChunks).toEqual([{ type: 'key', ts: ts(0) }]);
  });

  it('hardware fails mid-export -> software from the last keyframe, no duplicates or gaps, sticky', async () => {
    behaviour = (cfg, tsUs) => (cfg.hardwareAcceleration !== 'prefer-software' && tsUs >= ts(40) ? 'error' : 'ok');
    const s = mkSession();
    const switched: string[] = [];
    s.onSwitch = (a, b) => switched.push(`${a.id}->${b.id}`);
    expect(await drain(s, 0, 99)).toEqual(Array.from({ length: 100 }, (_, i) => i));
    expect(switched).toEqual(['hw->sw']);
    expect(s.variant.id).toBe('sw');
    expect(created.map(c => c.hardwareAcceleration)).toEqual(['prefer-hardware', 'prefer-software']);
    // the software decoder restarted at the keyframe before frame 40 (sample 30), as a key chunk
    expect(firstChunks[1]).toEqual({ type: 'key', ts: ts(30) });
    // later streams keep using software
    await drain(s, 0, 5);
    expect(created[created.length - 1].hardwareAcceleration).toBe('prefer-software');
  });

  it('every decoder fails at the same frame -> DecodeFailure with frame, time, clip and history', async () => {
    behaviour = (_cfg, tsUs) => (tsUs >= ts(40) ? 'error' : 'ok');
    const s = mkSession();
    let err: unknown;
    try { await drain(s, 0, 99); } catch (e) { err = e; }
    expect(err).toBeInstanceOf(DecodeFailure);
    const f = err as DecodeFailure;
    expect(f.message).toMatch(/frame 40/);
    expect(f.message).toMatch(/0:00\.67/);
    expect(f.message).toMatch(/3840×2160 H\.264/);
    expect(f.message).toMatch(/hardware, then software/);
    expect(f.message).toMatch(/EncodingError: Decoding error\./);
    expect(f.report).toMatchObject({ kind: 'decode', pres: 40, purpose: 'export', error: { name: 'EncodingError' } });
    expect(f.report.history.map(h => h.kind)).toEqual(['switch', 'fail']);
    // hardware, then one retry with the software fallback: 2 decoders, then a precise failure
    expect(created.length).toBe(2);
  });

  it('a transient error is retried with the same config (no software left)', async () => {
    let failed = false;
    behaviour = (_c, tsUs) => { if (!failed && tsUs >= ts(50)) { failed = true; return 'error'; } return 'ok'; };
    const s = mkSession(['sw']);
    expect(await drain(s, 0, 99)).toHaveLength(100);
    expect(s.history.map(h => h.kind)).toEqual(['retry']);
  });

  it('truncated sample data fails at once, precisely', async () => {
    const s = mkSession(['hw', 'sw'], 55);
    let err: unknown;
    try { await drain(s, 0, 99); } catch (e) { err = e; }
    expect(err).toBeInstanceOf(DecodeFailure);
    expect((err as DecodeFailure).report.kind).toBe('truncated');
    expect((err as DecodeFailure).message).toMatch(/truncated or incompletely copied/);
    expect(created.length).toBe(1);
  });

  it('blank (zero-filled) sample data fails at once as a damaged file, not a decoder failure', async () => {
    const s = mkSession(['hw', 'sw'], -1, 2000, 55);
    let err: unknown;
    const got: number[] = [];
    const st = s.frames(0, 99, { purpose: 'playback', gaps: 'skip', ...limits.playback });
    try { for (;;) { const r = await st.next(); if (!r) break; got.push(r.pres); r.frame.close(); } } catch (e) { err = e; } finally { st.close(); }
    // every good frame before the damage is still shown, then the failure names the first damaged frame
    expect(got).toEqual(Array.from({ length: 55 }, (_, i) => i));
    expect(err).toBeInstanceOf(DecodeFailure);
    const f = err as DecodeFailure;
    expect(f.report.kind).toBe('damaged');
    expect(f.report.sample).toBe(55);
    expect(f.report.pres).toBe(55);
    expect(f.message).toMatch(/blank or damaged/);
    expect(f.message).toMatch(/Copy the clip from the SD card again/);
    expect(f.message).not.toMatch(/Browser said/);
    // no pointless fallback to software for a file problem
    expect(created.length).toBe(1);
    expect(s.variant.id).toBe('hw');
  });

  it('a decoder error where the file goes blank is reported as a damaged file, without switching to software', async () => {
    // sample 55 still parses (the copy stopped inside it) but the decoder chokes on it; 56 on is zeros
    behaviour = (_c, tsUs) => (tsUs === ts(55) ? 'error' : 'ok');
    const s = mkSession(['hw', 'auto', 'sw'], -1, 2000, 56);
    const switched: string[] = [];
    s.onSwitch = (a, b) => switched.push(`${a.id}->${b.id}`);
    let err: unknown;
    try { await drain(s, 0, 99); } catch (e) { err = e; }
    expect(err).toBeInstanceOf(DecodeFailure);
    expect((err as DecodeFailure).report.kind).toBe('damaged');
    expect((err as DecodeFailure).message).toMatch(/blank or damaged/);
    expect(switched).toEqual([]);
    expect(s.variant.id).toBe('hw');
  });

  it('pre-flight on a file that is blank from the start reports the file, not the decoders', async () => {
    const s = mkSession(['hw', 'auto', 'sw'], -1, 2000, 0);
    await expect(s.preflight()).rejects.toMatchObject({ report: { kind: 'damaged' } });
    expect(created.length).toBe(1);
  });

  it('a hung decoder trips the stall watchdog and recovers in software', async () => {
    behaviour = (cfg, tsUs) => (cfg.hardwareAcceleration === 'prefer-hardware' && tsUs >= ts(20) ? 'hang' : 'ok');
    const s = mkSession(['hw', 'sw'], -1, 150);
    expect(await drain(s, 0, 40)).toHaveLength(41);
    expect(s.history[0]).toMatchObject({ kind: 'switch', variant: 'hw', to: 'sw' });
    expect(s.history[0].error).toMatch(/stopped responding/);
  });

  it('pre-flight skips a decoder that says "supported" but fails, and keeps the working one first', async () => {
    behaviour = cfg => (cfg.hardwareAcceleration === 'prefer-software' ? 'ok' : 'error');
    const s = mkSession();
    const pf = await s.preflight();
    expect(pf.ok).toBe(true);
    expect(pf.tried.map(x => `${x.variant}:${x.ok}`)).toEqual(['hw:false', 'auto:false', 'sw:true']);
    expect(s.variant.id).toBe('sw');
    expect(s.variants.map(v => v.id)).toEqual(['sw']);
    expect(s.software).toBe(true);
  });

  it('pre-flight reports failure when nothing decodes', async () => {
    behaviour = () => 'error';
    const s = mkSession(['hw', 'auto']);
    const pf = await s.preflight();
    expect(pf.ok).toBe(false);
    expect(pf.tried.every(x => !x.ok && /Decoding error/.test(x.error ?? ''))).toBe(true);
  });

  it('playback skips a frame the decoder drops instead of restarting', async () => {
    behaviour = (_c, tsUs) => (tsUs === ts(12) ? 'hang' : 'ok');
    // (a single dropped output: the mock "hangs" only that frame's output, the decoder keeps going)
    const s = mkSession(['hw']);
    const orig = MockDecoder.prototype.decode;
    MockDecoder.prototype.decode = function (this: any, chunk: MockChunk) {
      if (chunk.timestamp === ts(12)) { this.decodeQueueSize++; setTimeout(() => { this.decodeQueueSize--; }, 0); return; }
      return orig.call(this, chunk);
    };
    try {
      const got = await drain(s, 0, 20, 'playback');
      expect(got).toEqual(Array.from({ length: 21 }, (_, i) => i).filter(p => p !== 12));
      expect(created.length).toBe(1);
    } finally { MockDecoder.prototype.decode = orig; }
  });
});

// ───────── real clips (skipped when absent) ─────────

const O3 = join(process.env.STILLPOINT_O3_DIR ?? join(homedir(), 'Desktop/untitled folder 4'), 'DJI_0026.MP4');
const OA4 = join(homedir(), 'Desktop/DJI_20260927091931_0012_D.MP4');
const IPHONE = join(homedir(), 'Downloads/ScreenRecording_09-12-2026 13-52-09_1.MP4');

async function census(path: string) {
  const blob = diskBlob(path);
  const info = await openMp4(blob);
  const t = mainVideoTrack(info);
  const kinds = await classifySyncSamples(t, (o, s) => readRanges(blob, o, s, { concurrency: 8 }));
  const counts: Record<string, number> = {};
  for (const k of kinds.values()) counts[k] = (counts[k] ?? 0) + 1;
  return { t, kinds, counts, facts: streamFacts(t, t.sampleCount / (t.durationS ?? 1)), blob };
}

const codecFamilyOf = (t: Mp4Track) => (/^(hvc1|hev1)/.test(t.codecString ?? '') ? 'hevc' : 'h264');

describe('real clips: what the cameras actually use', () => {
  it.skipIf(!existsSync(O3))('DJI O3 H.264 High 5.2: every sync sample is an IDR', async () => {
    const { counts, facts, t } = await census(O3);
    console.log('[decode] O3', describeStream(facts), JSON.stringify(counts));
    expect(Object.keys(counts)).toEqual(['idr']);
    expect(facts).toMatchObject({ family: 'h264', profile: 'High', level: '5.2', bitDepth: 8, width: 3840, height: 2160 });
    expect(t.codecString).toBe('avc1.640034');
  }, 60000);
  it.skipIf(!existsSync(OA4))('Osmo Action 4 HEVC Main 10: IDR_N_LP keyframes', async () => {
    const { counts, facts } = await census(OA4);
    console.log('[decode] OA4', describeStream(facts), JSON.stringify(counts));
    expect(Object.keys(counts)).toEqual(['idr']);
    expect(facts).toMatchObject({ family: 'hevc', bitDepth: 10, profile: 'Main 10', width: 3840, height: 2880 });
  }, 120000);
  it.skipIf(!existsSync(O3) && !existsSync(OA4) && !existsSync(IPHONE))('the cameras\' samples pass the damaged-data check', async () => {
    for (const p of [O3, OA4, IPHONE].filter(x => existsSync(x))) {
      const blob = diskBlob(p);
      const t = mainVideoTrack(await openMp4(blob));
      const ls = t.codecConfig ? (codecFamilyOf(t) === 'hevc' ? (t.codecConfig[21] & 3) + 1 : (t.codecConfig[4] & 3) + 1) : 4;
      for (const first of [0, Math.max(0, (t.sampleCount >> 1) - 30), Math.max(0, t.sampleCount - 60)]) {
        const s = await readSamples(blob, t, first, Math.min(60, t.sampleCount - first));
        s.forEach((a, k) => expect(wellFormedAccessUnit(a, ls), `${p} sample ${first + k}`).toBe(true));
      }
    }
  }, 120000);
  it.skipIf(!existsSync(IPHONE))('iPhone HEVC: CRA keyframes with RASL leading pictures -> spans step back', async () => {
    const { counts, kinds, t, blob } = await census(IPHONE);
    console.log('[decode] iPhone', JSON.stringify(counts));
    expect(counts.cra).toBeGreaterThan(0);
    const ix = new FrameIndex(t);
    ix.applyRapKinds(kinds);
    const cra = [...kinds].find(([, k]) => k === 'cra')![0];
    const e = ix.leadingEnd(cra);
    expect(e).toBeGreaterThan(cra + 1);
    // the leading pictures really are RASL
    const lead = await readSamples(blob, t, cra + 1, e - cra - 1);
    for (const a of lead) expect([8, 9]).toContain(nalTypes(a, 'hevc').find(x => x < 32));
    // a span for a RASL picture starts before the CRA; a span for the CRA itself starts at it and skips them
    const rasl = ix.presOfDec[cra + 1];
    expect(ix.decodeSpan(rasl, rasl)[0]).toBeLessThan(cra);
    expect(ix.decodeSpan(ix.presOfDec[cra], ix.presOfDec[cra])[0]).toBe(cra);
    expect(ix.skipAfterStart(cra)).toBe(e);
  }, 60000);
});

describe('TruncatedSampleError', () => {
  it('names the sample', () => {
    expect(new TruncatedSampleError(7, 10, 1000).message).toMatch(/sample 7: 10 of 1000 bytes/);
    expect(new DamagedSampleError(9).message).toMatch(/sample 9 is blank or damaged/);
  });
});

describe('damaged sample data (well-formed access units)', () => {
  const big = (hdr: number, n: number, tail = 0x5a) => { const x = Array(n).fill(0x11); x[0] = hdr; x[n - 1] = tail; return x; };
  it('accepts real-shaped access units, including trailing zero padding', () => {
    expect(wellFormedAccessUnit(au(AVC_SPS, AVC_PPS, AVC_IDR))).toBe(true);
    expect(wellFormedAccessUnit(au(hevc(32, 1), hevc(33, 1), hevc(34, 1), hevc(19, 0xaf)))).toBe(true);
    expect(wellFormedAccessUnit(au(big(0x65, 5000)))).toBe(true);
    expect(wellFormedAccessUnit(Uint8Array.from([...au(AVC_IDR), 0, 0, 0, 0, 0, 0]))).toBe(true);
    // 2-byte NAL length fields (avcC lengthSizeMinusOne = 1)
    expect(wellFormedAccessUnit(Uint8Array.from([0, 3, ...AVC_IDR, 0, 3, ...AVC_P]), 2)).toBe(true);
  });
  it('rejects zeros, foreign data and zero-filled NAL tails', () => {
    expect(wellFormedAccessUnit(new Uint8Array(0))).toBe(false);
    expect(wellFormedAccessUnit(new Uint8Array(4096))).toBe(false);                       // blank (sparse / not downloaded)
    expect(wellFormedAccessUnit(Uint8Array.from([0, 0, 0x40, 0, 0x65, 1, 2, 3]))).toBe(false); // length runs past the sample
    expect(wellFormedAccessUnit(au([0xe5, 0x88, 0x80]))).toBe(false);                    // forbidden_zero_bit set
    const holed = big(0x65, 5000); for (let i = 3000; i < 5000; i++) holed[i] = 0;         // data stops mid-NAL
    expect(wellFormedAccessUnit(au(holed))).toBe(false);
    expect(wellFormedAccessUnit(Uint8Array.from([...au(AVC_IDR), 0, 0, 0, 9]))).toBe(false); // garbage after the last NAL
    // moov bytes where a sample should be (a truncated copy with the index appended)
    expect(wellFormedAccessUnit(Uint8Array.from([0, 0, 0x1c, 0x2d, ...Array.from(new TextEncoder().encode('moovmvhd')), ...Array(64).fill(1)]))).toBe(false);
  });
});
