/**
 * The export pipeline: decode (WebCodecs) -> warp (WebGPU Warper, any output size) -> retime (RetimeRenderer: frame
 * rate change, synthetic motion blur, slow motion) -> encode (WebCodecs) -> MP4 mux (mediabunny) with AAC passthrough.
 * Runs in the engine worker. Memory is bounded by construction:
 *   - the frame source (a FrameStream: SequentialDecoder + fallback/recovery) holds <= maxFrames decoded frames and a
 *     few chunks in the decoder; it delivers every frame of [first, last] exactly once, in order, or throws,
 *   - the RetimeRenderer keeps at most the schedule's maxSpan warped frames (motion blur) on the GPU,
 *   - the encoder queue is capped (encodeQueueSize), and the muxer chain length is capped (writer.pending),
 *   - samples are read from the file in ~12 MB coalesced batches with one batch of read-ahead,
 *   - every VideoFrame (decoded, warped, blended, re-stamped) is closed as soon as it has been submitted.
 *
 * Timing: the schedule (retime.ts buildSchedule) runs over the exported range's own frames (source index j = pres -
 * first). 'source' rate = identity (the source's own timestamps, the source's track timescale: bit-for-bit the
 * pre-retime export); real time = i / fps from the range start, audio kept; slow motion = i / fps, audio dropped when
 * the speed changes. Output track timescale: a multiple of the output rate (e.g. 24000 for 23.976, 25000 for 25).
 */
import type { Mp4Info, Mp4Track, OutputSchedule, Plan } from '../types';
import type { Warper } from '../gpu/warp';
import { RetimeRenderer, type RetimeStats } from '../gpu/retime_render';
import { ShutterRenderer, type ShutterStats } from '../gpu/shutter_render';
import type { BlendTransfer } from '../gpu/blend';
import type { DecodedFrame, FrameIndex, ReadSamples } from './decode';
import type { EncoderChoice } from './encode';
import { Mp4Writer, type AudioPassthrough, type OutputSink } from './mux';

export interface RenderProgress {
  phase: 'starting' | 'rendering' | 'finalizing' | 'done';
  /** output frames done / total */
  done: number;
  total: number;
  /** output frames per second (smoothed) */
  fps: number;
  etaS: number;
  elapsedS: number;
  bytes: number;
}

export interface RenderResult {
  /** output frames written */
  frames: number;
  seconds: number;
  /** output frames rendered per second of wall time */
  fps: number;
  bytes: number;
  codec: string;
  audioPackets: number;
  blob: Blob | null;
  sinkKind: OutputSink['kind'];
  name: string;
  planMisses: number;
  /** output picture and timing */
  width: number;
  height: number;
  /** output frame rate (source rate for 'source') */
  outFps: number;
  /** playback length of the video track, s */
  durationS: number;
  timeMode: 'identity' | 'realtime' | 'slowmo';
  /** source frames decoded */
  sourceFrames: number;
  /** audio requested but not written (slow motion) */
  audioDropped: boolean;
  /** single-pass or supersampled (anti-aliased downscale) warp */
  warpMode: 'direct' | 'supersample';
  /** how retimed frames were made: 'frames' (warp / re-stamp / blend of whole frames) or 'shutter' (gyro sub-frame
   *  synthetic shutter, motion blur) */
  renderer: 'frames' | 'shutter';
  retime: RetimeStats | ShutterStats;
  warnings: string[];
  /** where the render loop spent its time (ms): waiting for decoded frames, for encoder/muxer room, in warp() */
  timing: { waitDecode: number; waitEncode: number; warp: number; encodeCall: number; flush: number; finalize: number };
}

export interface RenderJob {
  file: Blob;
  info: Mp4Info;
  video: Mp4Track;
  index: FrameIndex;
  /** the plan the warper renders (output size = the export's) */
  plan: Plan;
  warper: Warper;
  /** output frames over the range's source frames (index j = pres - first); identity when `identity` */
  schedule: OutputSchedule;
  /** the schedule is the identity (source rate): keep the source's timestamps and timescale */
  identity: boolean;
  readSamples: ReadSamples;
  /** decoded source frames of pres range [first, last], in order, each exactly once (a DecoderSession FrameStream) */
  frames: (first: number, last: number) => FrameSource;
  encoder: EncoderChoice;
  sink: OutputSink;
  /** presentation frame range, inclusive */
  first: number;
  last: number;
  includeAudio: boolean;
  /** key frame interval in output frames (default ~1 s) */
  gop?: number;
  signal: AbortSignal;
  onProgress?: (p: RenderProgress) => void;
  /** called at most every `previewEveryMs` with the frames of the moment (do not close them) */
  onPreview?: (source: VideoFrame, warped: VideoFrame, pres: number) => void;
  previewEveryMs?: number;
  /** colour tags for the output (default: the source's, when SDR BT.709) */
  colorSpace?: VideoColorSpaceInit;
  /** transfer of the warped pictures, for motion-blur blending in linear light (default 'bt709') */
  blendTransfer?: BlendTransfer;
  /** motion blur: exposure window per output frame (s). With a plan that carries its virtual path, every output is
   *  rendered by the gyro sub-frame synthetic shutter (ShutterRenderer; also below 2x the output rate, where a blend
   *  of whole frames adds nothing) instead of a blend of whole frames. */
  shutterS?: number;
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

/** Output track timescale for a frame rate: N*1000 for N*1000/1001 rates, fps*1000 for integer rates, else a
 *  multiple of the rate's rational form (frame times stay exact integers). */
export function timescaleFor(fps: number): number {
  const n = Math.round(fps * 1.001);
  if (Math.abs(fps - n * 1000 / 1001) < 1e-6 && Math.abs(fps - Math.round(fps)) > 1e-6) return n * 1000;
  if (Math.abs(fps - Math.round(fps)) < 1e-9) return Math.round(fps) * 1000;
  // continued fraction p/q ~= fps, q <= 1001
  let h0 = 0, h1 = 1, k0 = 1, k1 = 0, x = fps;
  for (let i = 0; i < 20; i++) {
    const a = Math.floor(x);
    const h2 = a * h1 + h0, k2 = a * k1 + k0;
    if (k2 > 1001) break;
    h0 = h1; h1 = h2; k0 = k1; k1 = k2;
    if (Math.abs(x - a) < 1e-9) break;
    x = 1 / (x - a);
  }
  const p = Math.max(1, h1);
  return p * Math.max(1, Math.ceil(10000 / p));
}

export async function renderClip(job: RenderJob): Promise<RenderResult> {
  const { file, info, video, index, plan, warper, signal, schedule: sched } = job;
  const t0 = index.pts[job.first];
  const tEnd = index.pts[job.last] + index.frameDur;
  const nSrc = job.last - job.first + 1;
  const total = sched.n;
  const outFps = sched.fps ?? 1 / index.frameDur;
  const timeMode: RenderResult['timeMode'] = job.identity ? 'identity' : sched.dropAudio || (sched.speed ?? 1) !== 1 ? 'slowmo' : 'realtime';
  const gop = job.gop ?? Math.max(1, Math.round(outFps));
  const planIdx = planIndexer(plan);
  const halfFrame = index.frameDur * 0.5 + 1e-6;
  let planMisses = 0;
  // schedule source frame j -> plan record (nearest plan pts)
  const record = new Int32Array(nSrc);
  for (let j = 0; j < nSrc; j++) {
    const pts = index.pts[job.first + j];
    const k = planIdx(pts);
    if (Math.abs(plan.framePts[k] - pts) > halfFrame) planMisses++;
    record[j] = k;
  }
  // exact output time (s, from the range start) and duration of output frame i
  const outTimeOf = (i: number): [number, number] => {
    if (job.identity) {
      const pres = job.first + sched.taps[sched.tapStart[i]];
      const pts = index.pts[pres];
      return [pts - t0, pres + 1 < index.n ? index.pts[pres + 1] - pts : index.frameDur];
    }
    return [i / outFps, 1 / outFps];
  };
  const durationS = total ? (() => { const [a, d] = outTimeOf(total - 1); return a + d; })() : 0;

  // audio passthrough (AAC only) — not for slow motion / speed changes
  const audioDropped = job.includeAudio && !!sched.dropAudio;
  const aTrack = job.includeAudio && !sched.dropAudio ? info.tracks.find(t => t.kind === 'audio' && (t.codec === 'aac' || t.fourcc === 'mp4a') && t.sampleCount > 0) : undefined;
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

  const timescale = job.identity ? video.timescale : timescaleFor(outFps);
  const writer = new Mp4Writer(job.sink, job.encoder.codec, { timescale, audio, colorSpace: job.colorSpace });
  await writer.start();

  // encoded chunk timestamp (µs) -> output frame index (exact time + duration)
  const outIndex = new Map<number, number>();
  let encError: unknown = null;
  let encWake: (() => void) | null = null;
  const wakeEnc = () => { const w = encWake; encWake = null; w?.(); };
  const encoder = new VideoEncoder({
    output: (chunk, meta) => {
      const i = outIndex.get(chunk.timestamp);
      outIndex.delete(chunk.timestamp);
      const [ts, dur] = i !== undefined ? outTimeOf(i) : [chunk.timestamp / 1e6 - t0, 1 / outFps];
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

  let rr: RetimeRenderer | ShutterRenderer | null = null;
  let dec: FrameSource | null = null;

  const tStart = performance.now();
  let done = 0, srcDone = 0, lastPreview = 0, lastProgress = 0, fpsEma = 0, lastT = tStart, lastDone = 0;
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
    const recordOf = (j: number) => record[j];
    rr = job.shutterS && ShutterRenderer.supported(plan)
      ? await ShutterRenderer.create(warper, sched, { pts: index.pts.subarray(job.first, job.last + 1), shutterS: job.shutterS, record: recordOf, transfer: job.blendTransfer ?? 'bt709' })
      : await RetimeRenderer.create(warper, sched, { transfer: job.blendTransfer ?? 'bt709', record: recordOf });
    dec = job.frames(job.first, job.last);
    for (;;) {
      if (signal.aborted) throw abortError();
      let tA = performance.now();
      const got = await dec.next();
      tm.waitDecode += performance.now() - tA;
      if (!got) break;
      const { frame: f, pres } = got;
      if (pres < job.first || pres > job.last) { f.close(); continue; }
      srcDone++;
      tA = performance.now();
      await waitEncoder();
      tm.waitEncode += performance.now() - tA;
      if (signal.aborted) { f.close(); throw abortError(); }
      let outs: VideoFrame[];
      tA = performance.now();
      try { outs = rr.push(f, pres - job.first); } catch (e) { f.close(); throw e; }
      tm.warp += performance.now() - tA;
      tA = performance.now();
      let shown = false;
      try {
        for (const o of outs) {
          outIndex.set(o.timestamp, done);
          encoder.encode(o, { keyFrame: done % gop === 0 });
          done++;
        }
        const now = performance.now();
        tm.encodeCall += now - tA;
        if (outs.length && job.onPreview && now - lastPreview > (job.previewEveryMs ?? 250)) {
          lastPreview = now; shown = true;
          try { job.onPreview(f, outs[outs.length - 1], pres); } catch { /* preview is best-effort */ }
        }
      } finally {
        for (const o of outs) o.close();
        f.close();
      }
      const now = performance.now();
      if (shown || now - lastProgress > 200) { lastProgress = now; report('rendering'); }
    }
    if (srcDone < nSrc || !rr.done) throw new Error(`Decoder produced ${srcDone} of ${nSrc} frames`);
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
      width: warper.width, height: warper.height, outFps, durationS, timeMode, sourceFrames: srcDone, audioDropped,
      warpMode: warper.scaling.mode, renderer: rr instanceof ShutterRenderer ? 'shutter' : 'frames', retime: { ...rr.stats },
      // (the schedule's "no blur at this rate" note does not apply to the gyro shutter)
      warnings: (sched.warnings ?? []).filter(w => !(rr instanceof ShutterRenderer && /^motion blur/.test(w))),
      timing: Object.fromEntries(Object.entries(tm).map(([k, v]) => [k, Math.round(v)])) as RenderResult['timing'],
    };
  } catch (e) {
    await writer.cancel().catch(() => {});
    throw e;
  } finally {
    dec?.close();
    rr?.destroy();
    try { if (encoder.state !== 'closed') encoder.close(); } catch { /* ignore */ }
  }
}
