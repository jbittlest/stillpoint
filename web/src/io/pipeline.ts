/**
 * The export pipeline: decode (WebCodecs) -> warp (WebGPU Warper) -> encode (WebCodecs) -> MP4 mux (mediabunny) with
 * AAC passthrough. Runs in the engine worker. Memory is bounded by construction:
 *   - the frame source (a FrameStream: SequentialDecoder + fallback/recovery) holds <= maxFrames decoded frames and a
 *     few chunks in the decoder; it delivers every frame of [first, last] exactly once, in order, or throws,
 *   - the encoder queue is capped (encodeQueueSize), and the muxer chain length is capped (writer.pending),
 *   - samples are read from the file in ~12 MB coalesced batches with one batch of read-ahead,
 *   - every VideoFrame (decoded, warped, re-stamped) is closed as soon as it has been submitted.
 */
import type { Mp4Info, Mp4Track, Plan } from '../types';
import type { WarperLike } from '../ui/contracts';
import type { DecodedFrame, FrameIndex, ReadSamples } from './decode';
import type { EncoderChoice } from './encode';
import { Mp4Writer, type AudioPassthrough, type OutputSink } from './mux';

export interface RenderProgress {
  phase: 'starting' | 'rendering' | 'finalizing' | 'done';
  done: number;
  total: number;
  /** output frames per second (smoothed) */
  fps: number;
  etaS: number;
  elapsedS: number;
  bytes: number;
}

export interface RenderResult {
  frames: number;
  seconds: number;
  fps: number;
  bytes: number;
  codec: string;
  audioPackets: number;
  blob: Blob | null;
  sinkKind: OutputSink['kind'];
  name: string;
  planMisses: number;
  /** where the render loop spent its time (ms): waiting for decoded frames, for encoder/muxer room, in warp() */
  timing: { waitDecode: number; waitEncode: number; warp: number; encodeCall: number; flush: number; finalize: number };
}

export interface RenderJob {
  file: Blob;
  info: Mp4Info;
  video: Mp4Track;
  index: FrameIndex;
  plan: Plan;
  warper: WarperLike;
  readSamples: ReadSamples;
  /** decoded source frames of pres range [first, last], in order, each exactly once (a DecoderSession FrameStream) */
  frames: (first: number, last: number) => FrameSource;
  encoder: EncoderChoice;
  sink: OutputSink;
  /** presentation frame range, inclusive */
  first: number;
  last: number;
  includeAudio: boolean;
  /** key frame interval in frames (default ~1 s) */
  gop?: number;
  signal: AbortSignal;
  onProgress?: (p: RenderProgress) => void;
  /** called at most every `previewEveryMs` with the frames of the moment (do not close them) */
  onPreview?: (source: VideoFrame, warped: VideoFrame, pres: number) => void;
  previewEveryMs?: number;
  /** colour tags for the output (default: the source's, when SDR BT.709) */
  colorSpace?: VideoColorSpaceInit;
}

export interface FrameSource {
  next(): Promise<DecodedFrame | null>;
  close(): void;
}

/** plan record index for presentation frame pts t (nearest; plan.framePts ascending) */
export function planIndexer(plan: Plan) {
  const a = plan.framePts;
  return (t: number): number => {
    let lo = 0, hi = a.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < t) lo = m + 1; else hi = m; }
    if (lo > 0 && Math.abs(a[lo - 1] - t) <= Math.abs(a[lo] - t)) lo--;
    return lo;
  };
}

function abortError() { return new DOMException('Export cancelled', 'AbortError'); }

export async function renderClip(job: RenderJob): Promise<RenderResult> {
  const { file, info, video, index, plan, warper, signal } = job;
  const t0 = index.pts[job.first];
  const tEnd = index.pts[job.last] + index.frameDur;
  const total = job.last - job.first + 1;
  const gop = job.gop ?? Math.max(1, Math.round(1 / index.frameDur));
  const planIdx = planIndexer(plan);
  const halfFrame = index.frameDur * 0.5 + 1e-6;
  let planMisses = 0;

  // audio passthrough (AAC only)
  const aTrack = job.includeAudio ? info.tracks.find(t => t.kind === 'audio' && (t.codec === 'aac' || t.fourcc === 'mp4a') && t.sampleCount > 0) : undefined;
  let audio: AudioPassthrough | undefined;
  let aFirst = 0, aLast = -1;
  if (aTrack) {
    audio = { codec: 'aac', sampleRate: aTrack.sampleRate ?? 48000, channels: aTrack.channels ?? 2, codecString: aTrack.codecString ?? 'mp4a.40.2', description: aTrack.codecConfig };
    // AAC samples are in presentation order; take those that start inside [t0, tEnd)
    while (aFirst < aTrack.sampleCount && aTrack.cts[aFirst] < t0 - 1e-6) aFirst++;
    aLast = aFirst - 1;
    while (aLast + 1 < aTrack.sampleCount && aTrack.cts[aLast + 1] < tEnd - 1e-6) aLast++;
    if (aLast < aFirst) audio = undefined;
  }

  const writer = new Mp4Writer(job.sink, job.encoder.codec, { timescale: video.timescale, audio, colorSpace: job.colorSpace });
  await writer.start();

  // encoded chunk timestamp (µs) -> exact output time (s) + duration
  const outTime = new Map<number, [number, number]>();
  let encError: unknown = null;
  let encWake: (() => void) | null = null;
  const wakeEnc = () => { const w = encWake; encWake = null; w?.(); };
  const encoder = new VideoEncoder({
    output: (chunk, meta) => {
      const tt = outTime.get(chunk.timestamp);
      outTime.delete(chunk.timestamp);
      const [ts, dur] = tt ?? [chunk.timestamp / 1e6, index.frameDur];
      void writer.addVideo(chunk, meta, ts, dur).then(wakeEnc, wakeEnc);
      pumpAudio(ts + 0.5);
    },
    error: e => { encError = e; wakeEnc(); },
  });
  encoder.addEventListener('dequeue', wakeEnc);
  encoder.configure(job.encoder.config);

  // audio feeding, kept ~0.5 s ahead of the video mux position
  let aNext = aFirst;
  let aBusy = false;
  function pumpAudio(until: number) {
    if (!audio || !aTrack || aBusy || aNext > aLast) return;
    if (aTrack.cts[aNext] - t0 > until) return;
    aBusy = true;
    const count = Math.min(256, aLast + 1 - aNext);
    const start = aNext;
    aNext += count;
    job.readSamples(file, aTrack, start, count).then(datas => {
      for (let i = 0; i < datas.length; i++) {
        const s = start + i;
        const dur = s + 1 < aTrack.sampleCount ? aTrack.cts[s + 1] - aTrack.cts[s] : 1024 / (aTrack.sampleRate ?? 48000);
        void writer.addAudio(datas[i], Math.max(0, aTrack.cts[s] - t0), dur);
      }
    }).catch(e => { encError ??= e; }).finally(() => { aBusy = false; });
  }

  const dec = job.frames(job.first, job.last);

  const tStart = performance.now();
  let done = 0, lastPreview = 0, lastProgress = 0, fpsEma = 0, lastT = tStart, lastDone = 0;
  const report = (phase: RenderProgress['phase']) => {
    const now = performance.now();
    const elapsed = (now - tStart) / 1000;
    if (now - lastT > 400) {
      const inst = ((done - lastDone) * 1000) / (now - lastT);
      fpsEma = fpsEma ? fpsEma * 0.7 + inst * 0.3 : inst;
      lastT = now; lastDone = done;
    }
    const fps = fpsEma || (elapsed > 0 ? done / elapsed : 0);
    job.onProgress?.({ phase, done, total, fps, etaS: fps > 0 ? (total - done) / fps : NaN, elapsedS: elapsed, bytes: Math.max(job.sink.bytesWritten(), writer.payloadBytes) });
  };
  report('starting');

  const tm = { waitDecode: 0, waitEncode: 0, warp: 0, encodeCall: 0, flush: 0, finalize: 0 };
  const waitEncoder = async () => {
    while (!encError && !writer.error && (encoder.encodeQueueSize > 8 || writer.pending > 16)) {
      if (signal.aborted) return;
      await new Promise<void>(r => { encWake = r; setTimeout(r, 50); });
    }
    if (encError) throw encError;
    if (writer.error) throw writer.error;
  };

  try {
    for (;;) {
      if (signal.aborted) throw abortError();
      let tA = performance.now();
      const got = await dec.next();
      tm.waitDecode += performance.now() - tA;
      if (!got) break;
      const { frame: f, pres } = got;
      if (pres < job.first || pres > job.last) { f.close(); continue; }
      const pts = index.pts[pres];
      const k = planIdx(pts);
      if (Math.abs(plan.framePts[k] - pts) > halfFrame) planMisses++;
      tA = performance.now();
      await waitEncoder();
      tm.waitEncode += performance.now() - tA;
      if (signal.aborted) { f.close(); throw abortError(); }
      let warped: VideoFrame;
      tA = performance.now();
      try { warped = warper.warp(f, k); } catch (e) { f.close(); throw e; }
      tm.warp += performance.now() - tA;
      tA = performance.now();
      const tsUs = Math.round((pts - t0) * 1e6);
      const dur = pres + 1 < index.n ? index.pts[pres + 1] - pts : index.frameDur;
      const stamped = new VideoFrame(warped, { timestamp: tsUs, duration: Math.round(dur * 1e6) });
      outTime.set(tsUs, [pts - t0, dur]);
      encoder.encode(stamped, { keyFrame: done % gop === 0 });
      stamped.close();
      const now = performance.now();
      tm.encodeCall += now - tA;
      if (job.onPreview && now - lastPreview > (job.previewEveryMs ?? 250)) {
        lastPreview = now;
        try { job.onPreview(f, warped, pres); } catch { /* preview is best-effort */ }
      }
      warped.close();
      f.close();
      done++;
      if (now - lastProgress > 200) { lastProgress = now; report('rendering'); }
    }
    if (done < total) throw new Error(`Decoder produced ${done} of ${total} frames`);
    let tB = performance.now();
    await encoder.flush();
    tm.flush = performance.now() - tB;
    if (encError) throw encError;
    report('finalizing');
    tB = performance.now();
    // remaining audio
    while (audio && (aNext <= aLast || aBusy)) {
      if (aBusy) { await new Promise(r => setTimeout(r, 5)); continue; }
      pumpAudio(Infinity);
    }
    const blob = await writer.finalize();
    tm.finalize = performance.now() - tB;
    const seconds = (performance.now() - tStart) / 1000;
    report('done');
    return {
      frames: done, seconds, fps: done / seconds, bytes: job.sink.bytesWritten(), codec: job.encoder.config.codec,
      audioPackets: writer.audioPackets, blob, sinkKind: job.sink.kind, name: job.sink.name, planMisses,
      timing: Object.fromEntries(Object.entries(tm).map(([k, v]) => [k, Math.round(v)])) as RenderResult['timing'],
    };
  } catch (e) {
    await writer.cancel().catch(() => {});
    throw e;
  } finally {
    dec.close();
    try { if (encoder.state !== 'closed') encoder.close(); } catch { /* ignore */ }
  }
}
