/**
 * WebCodecs video decoding for Stillpoint: decoder configs from an Mp4Track, presentation-order frame indexing, and a
 * pull-based sequential decoder with strict backpressure (bounded decode queue + bounded decoded-frame queue, samples
 * read from the Blob in coalesced batches with one batch of read-ahead). Every VideoFrame handed out must be closed
 * by the caller; frames still queued at close() are closed here.
 */
import type { Mp4Track } from '../types';

export type ReadSamples = (file: Blob, t: Mp4Track, first: number, count: number) => Promise<Uint8Array[]>;

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

export function decoderConfig(t: Mp4Track, hw: HardwareAcceleration = 'prefer-hardware', lowLatency = false): VideoDecoderConfig {
  if (!t.codecString) throw new Error(`Unsupported video codec (${t.fourcc || t.codec || 'unknown'})`);
  const cfg: VideoDecoderConfig = {
    codec: t.codecString,
    codedWidth: t.width,
    codedHeight: t.height,
    hardwareAcceleration: hw,
    optimizeForLatency: lowLatency,
  };
  if (t.codecConfig) cfg.description = t.codecConfig;
  const cs = colorSpaceOf(t);
  if (cs) cfg.colorSpace = cs;
  return cfg;
}

/** First supported decoder config for the track (hardware preferred), or null with the reason. */
export async function supportedDecoderConfig(t: Mp4Track, lowLatency = false): Promise<{ config: VideoDecoderConfig | null; reason?: string }> {
  if (typeof VideoDecoder === 'undefined') return { config: null, reason: 'This browser has no WebCodecs VideoDecoder.' };
  let last = '';
  for (const hw of ['prefer-hardware', 'no-preference'] as HardwareAcceleration[]) {
    try {
      const cfg = decoderConfig(t, hw, lowLatency);
      const r = await VideoDecoder.isConfigSupported(cfg);
      if (r.supported) return { config: cfg };
      last = `${cfg.codec} is not supported by this browser's video decoder.`;
    } catch (e) { last = String((e as Error).message ?? e); }
  }
  return { config: null, reason: last };
}

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
    }
    const deltas: number[] = [];
    for (let p = 1; p < Math.min(n, 400); p++) deltas.push(this.pts[p] - this.pts[p - 1]);
    deltas.sort((a, b) => a - b);
    this.frameDur = deltas.length ? deltas[deltas.length >> 1] : 1 / 30;
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

  /** Decode index of the sync sample at or before the decode position of pres frame p. */
  keyframeFor(p: number): number {
    let d = this.decOfPres[p];
    const sync = this.track.sync;
    while (d > 0 && !sync[d]) d--;
    return d;
  }

  /** Decode-order span [d0, d1] that yields every pres frame in [p0, p1]. */
  decodeSpan(p0: number, p1: number): [number, number] {
    let lo = Infinity, hi = -1;
    for (let p = p0; p <= p1; p++) { const d = this.decOfPres[p]; if (d < lo) lo = d; if (d > hi) hi = d; }
    let d0 = lo;
    const sync = this.track.sync;
    while (d0 > 0 && !sync[d0]) d0--;
    return [d0, hi];
  }
}

interface Batch { first: number; data: Uint8Array[] }

/**
 * Decodes samples [d0, d1] (decode order) and hands out frames (presentation order, as the decoder emits them) via
 * next(). At most `maxFrames` decoded frames are held (queued + not yet returned), and at most `maxDecodeQueue` chunks
 * wait inside the decoder.
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
  private cur: Batch | null = null;
  private ahead: Promise<Batch> | null = null;
  private readonly batchSize: number;
  /** frames handed to the caller and not yet reported back via release() — informational */
  decodedCount = 0;

  constructor(
    private file: Blob,
    private track: Mp4Track,
    private index: FrameIndex,
    private readSamples: ReadSamples,
    config: VideoDecoderConfig,
    private d0: number,
    private d1: number,
    private opts: { maxFrames?: number; maxDecodeQueue?: number; batchBytes?: number } = {},
  ) {
    this.nextSample = d0;
    const avg = track.sizes.length ? track.sizes.reduce((a, b) => a + b, 0) / track.sizes.length : 200_000;
    this.batchSize = Math.max(4, Math.min(120, Math.floor((opts.batchBytes ?? 12 * 1024 * 1024) / Math.max(1, avg))));
    this.decoder = new VideoDecoder({
      output: f => {
        if (this.closed) { f.close(); return; }
        this.decodedCount++;
        this.out.push(f);
        this.notify();
      },
      error: e => { this.error = e; this.notify(); },
    });
    this.decoder.addEventListener('dequeue', () => { this.notify(); this.pump(); });
    this.decoder.configure(config);
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
      if (nxt <= this.d1) this.ahead = this.fetchBatch(nxt);
    }
    return this.cur.data[i - this.cur.first];
  }

  private get maxFrames() { return this.opts.maxFrames ?? 6; }
  private get maxDecodeQueue() { return this.opts.maxDecodeQueue ?? 4; }

  private async pump() {
    if (this.pumping || this.closed || this.error) return;
    this.pumping = true;
    try {
      while (!this.closed && !this.error && this.nextSample <= this.d1
        && this.decoder.decodeQueueSize < this.maxDecodeQueue && this.out.length < this.maxFrames) {
        const i = this.nextSample;
        const data = await this.sampleAt(i);
        if (this.closed) return;
        this.decoder.decode(new EncodedVideoChunk({
          type: this.track.sync[i] ? 'key' : 'delta',
          timestamp: this.index.tsUs[i],
          duration: Math.round(this.index.frameDur * 1e6),
          data,
        }));
        this.nextSample++;
      }
      if (!this.closed && !this.error && this.nextSample > this.d1 && !this.flushing) {
        this.flushing = true;
        this.decoder.flush().then(() => { this.done = true; this.notify(); }, e => { if (!this.closed) { this.error = e; this.notify(); } });
      }
    } catch (e) {
      if (!this.closed) { this.error = e; this.notify(); }
    } finally {
      this.pumping = false;
    }
  }

  /** Next decoded frame (caller closes it), or null at the end of the span. */
  async next(): Promise<VideoFrame | null> {
    for (;;) {
      if (this.error) throw this.error;
      if (this.out.length) { const f = this.out.shift()!; void this.pump(); return f; }
      if (this.done || this.closed) return null;
      const p = new Promise<void>(r => (this.wake = r));
      void this.pump();
      await p;
    }
  }

  get queued() { return this.out.length; }

  close() {
    if (this.closed) return;
    this.closed = true;
    for (const f of this.out) f.close();
    this.out = [];
    try { if (this.decoder.state !== 'closed') this.decoder.close(); } catch { /* ignore */ }
    this.notify();
  }
}

/** Decode a single presentation frame p (keyframe -> p, then flush). Caller closes the result. */
export async function decodeOne(file: Blob, track: Mp4Track, index: FrameIndex, readSamples: ReadSamples, config: VideoDecoderConfig, p: number, signal?: AbortSignal): Promise<VideoFrame> {
  const [d0] = index.decodeSpan(p, p);
  // decode enough samples past p's decode position to cover B-frame reordering
  const dp = index.decOfPres[p];
  let d1 = dp;
  for (let d = dp + 1; d < Math.min(index.n, dp + 8); d++) if (index.presOfDec[d] < p) d1 = d;
  const dec = new SequentialDecoder(file, track, index, readSamples, { ...config, optimizeForLatency: true }, d0, d1, { maxFrames: 4, maxDecodeQueue: 8 });
  let hit: VideoFrame | null = null;
  try {
    for (;;) {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      const f = await dec.next();
      if (!f) break;
      if (index.presOfTimestamp(f.timestamp) === p) { hit?.close(); hit = f; } else f.close();
    }
  } finally { dec.close(); }
  if (!hit) throw new Error(`Frame ${p} could not be decoded`);
  return hit;
}
