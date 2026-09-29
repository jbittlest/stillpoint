/// <reference lib="webworker" />
/**
 * Engine worker: owns the GPUDevice, the Warper, the two preview canvases (transferred from the page), and runs
 * scrubbing, real-time preview playback and the export pipeline — so the page stays smooth whatever the GPU or the
 * codecs are doing.
 *
 * Decoding is hardened for machines other than the one it was written on (see decode.ts):
 *  - one video decoder at a time: seek / play / export / pre-flight take the decoder lock, a newer request aborts the
 *    older one, and the held preview frame is closed before a new decoder starts (hardware decoders on Windows have
 *    small frame pools, and a frame from an old decoder keeps its pool alive);
 *  - a pre-flight test decode when a clip opens picks a decoder config that actually works (hardware -> software for
 *    H.264), or explains up front why this computer can't play the clip;
 *  - playback, scrubbing and export recover from decoder errors (fallback + restart at the last clean random-access
 *    point) and otherwise fail with a precise message plus diagnostics the UI can copy.
 */
import type { Mp4Info, Mp4Track, Plan } from '../types';
import { readRanges, readSamples } from '../mp4';
import { Warper } from '../gpu/warp';
import { blendTransferFor } from '../gpu/blend';
import { outputGeometry, rescalePlan } from '../plan';
import { buildSchedule } from '../retime';
import {
  DecodeFailure, DecoderSession, FrameIndex, type FrameStream, cannotDecodeMessage, classifySyncSamples, colorSpaceOf,
  completeSamplePrefix, decodeLimits, decodeOne, describeStream, errText, isAbort, isApplePlatform, parseFault,
  softwareNote, streamFacts, supportedVariants, trimTrack,
} from './decode';
import { chooseEncoder, codecOptions, type EncoderChoice } from './encode';
import { clearOpfsExports, createSink } from './mux';
import { planIndexer, renderClip } from './pipeline';
import type { DecoderInfo, EngineDebug, EngineIn, EngineOut, ErrorDetails, ExportSettings } from './protocol';

const scope = self as unknown as DedicatedWorkerGlobalScope;
const post = (m: EngineOut, transfer: Transferable[] = []) => scope.postMessage(m, transfer);

let device: GPUDevice | null = null;
let before: OffscreenCanvas | null = null;
let after: OffscreenCanvas | null = null;
let bctx: OffscreenCanvasRenderingContext2D | null = null;
let actx: OffscreenCanvasRenderingContext2D | null = null;
let debug: EngineDebug = {};

let file: File | null = null;
let info: Mp4Info | null = null;
let video: Mp4Track | null = null;
let index: FrameIndex | null = null;
let session: DecoderSession | null = null;
let plan: Plan | null = null;
let planIdx: ((t: number) => number) | null = null;
let warper: Warper | null = null;
let warperPlanId = -1;

/** the last source frame shown (kept to re-warp instantly when the plan changes; closed before a new decoder starts) */
let heldSrc: VideoFrame | null = null;
let heldPres = -1;
let lastWarped: VideoFrame | null = null;

let seekGen = 0;
let seekCtl: AbortController | null = null;
let playing = false;
let playGen = 0;
let playCtl: AbortController | null = null;
let exporting: AbortController | null = null;
/** warpers replaced while an export was using them; destroyed when it ends */
const retired: Warper[] = [];

// ─────────── one decoder at a time ───────────

let lockTail: Promise<void> = Promise.resolve();
/** Wait for the decoder lock; call the returned function to release it. */
function acquireDecoder(): Promise<() => void> {
  let release!: () => void;
  const mine = new Promise<void>(r => (release = r));
  const prev = lockTail;
  lockTail = prev.then(() => mine);
  let released = false;
  return prev.then(() => () => { if (!released) { released = true; release(); } });
}

// ─────────── drawing ───────────

/** Draw `img` centred: 'contain' (letterboxed) or 'cover' (filling the canvas, centre-cropped). */
function drawFit(ctx: OffscreenCanvasRenderingContext2D, img: CanvasImageSource, iw: number, ih: number, fit: 'contain' | 'cover' = 'contain') {
  const cw = ctx.canvas.width, ch = ctx.canvas.height;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, cw, ch);
  const s = fit === 'cover' ? Math.max(cw / iw, ch / ih) : Math.min(cw / iw, ch / ih);
  const w = iw * s, h = ih * s;
  ctx.drawImage(img, (cw - w) / 2, (ch - h) / 2, w, h);
}
const drawContain = (ctx: OffscreenCanvasRenderingContext2D, img: CanvasImageSource, iw: number, ih: number) => drawFit(ctx, img, iw, ih, 'contain');

function drawPair(src: VideoFrame | null, warped: VideoFrame | null) {
  // the viewer takes the OUTPUT's aspect: when that is not the source's (e.g. a 9:16 export of 16:9 footage) the
  // original fills the frame centre-cropped, so both sides of the split show the same framing
  if (bctx && src) drawFit(bctx, src, src.displayWidth, src.displayHeight, 'cover');
  if (actx) {
    if (warped) drawContain(actx, warped, warped.displayWidth, warped.displayHeight);
    else if (src) drawContain(actx, src, src.displayWidth, src.displayHeight);
  }
}

function hold(src: VideoFrame, pres: number, warped: VideoFrame | null) {
  if (heldSrc && heldSrc !== src) heldSrc.close();
  heldSrc = src;
  heldPres = pres;
  if (lastWarped && lastWarped !== warped) lastWarped.close();
  lastWarped = warped;
}

/** Close the held frames (the canvases keep showing them). heldPres is kept so the view can be restored. */
function releaseHeld() {
  heldSrc?.close(); heldSrc = null;
  lastWarped?.close(); lastWarped = null;
}

function warpFor(src: VideoFrame, pres: number, w: Warper | null = warper): VideoFrame | null {
  if (!w || !plan || !index || !planIdx) return null;
  return w.warp(src, planIdx(index.pts[pres]));
}

/**
 * Playback warper: the same plan rendered at (about) the preview canvas size — identical geometry, because the
 * row matrices act on normalized rays and only outW/outH/outFx scale — so real-time preview doesn't warp 8-11 MP per
 * frame just to downscale it. Built lazily per plan and canvas size.
 */
let pvWarper: Warper | null = null;
let pvKey = '';
async function previewWarper(): Promise<Warper | null> {
  if (!plan || !device || !after) return warper;
  const s = Math.min(1, Math.max(after.width / plan.outW, after.height / plan.outH));
  if (s > 0.8) return warper;
  const outW = Math.max(2, Math.round((plan.outW * s) / 2) * 2), outH = Math.max(2, Math.round((plan.outH * s) / 2) * 2);
  const key = `${warperPlanId}:${outW}x${outH}`;
  if (pvWarper && key === pvKey) return pvWarper;
  try {
    const scaled: Plan = rescalePlan(plan, outW, outH);
    // single-pass (no supersampled downscale): real-time preview, the canvas scales it for display anyway
    const w = await Warper.create(device, scaled, { kernel: 'catmullrom', antialias: 'off' });
    pvWarper?.destroy();
    pvWarper = w; pvKey = key;
    return w;
  } catch (e) {
    console.warn('[stillpoint] preview warper failed, using the full one', e);
    return warper;
  }
}
function dropPreviewWarper() { pvWarper?.destroy(); pvWarper = null; pvKey = ''; }

function redrawHeld() {
  if (!heldSrc) return;
  let w: VideoFrame | null = null;
  try { w = warpFor(heldSrc, heldPres); } catch (e) { post({ type: 'error', message: 'Warp failed: ' + (e as Error).message }); }
  drawPair(heldSrc, w);
  if (lastWarped) lastWarped.close();
  lastWarped = w;
  post({ type: 'frame', pres: heldPres, t: index!.pts[heldPres], hasWarp: !!w, ms: 0 });
}

// ─────────── errors ───────────

function detailsOf(e: unknown, purpose: string): ErrorDetails {
  if (e instanceof DecodeFailure) return { ...e.report };
  if (session) return { ...session.report(e, -1, purpose) };
  return { purpose, error: { name: (e as Error)?.name ?? 'Error', message: errText(e) } };
}

/** A decode problem during scrubbing / playback: a precise message + diagnostics (the UI shows "Copy details"). */
function postDecodeError(prefix: string, e: unknown, purpose: string) {
  post({ type: 'error', message: `${prefix}: ${e instanceof DecodeFailure ? e.message : errText(e)}`, details: detailsOf(e, purpose) });
}

function decoderInfo(s: DecoderSession, preflight?: DecoderInfo['preflight']): DecoderInfo {
  const v = s.variant;
  const fallback = s.history.some(h => h.kind === 'switch' || h.kind === 'preflight-fail');
  return {
    codec: v.config.codec, variant: v.id, label: s.software ? 'software' : v.label, software: s.software, fallback,
    note: s.software && (fallback || !s.hwSupported) ? softwareNote(s.facts, s.hwSupported ? 'failed' : 'unavailable') : undefined,
    facts: s.facts, rap: s.index.rapCensus, preflight, hwSupported: s.hwSupported,
  };
}

// ─────────── init / open / plan ───────────

async function init(m: Extract<EngineIn, { type: 'init' }>) {
  debug = m.debug ?? {};
  if (m.before && m.after) {
    before = m.before; after = m.after;
    bctx = before.getContext('2d', { alpha: false }) as OffscreenCanvasRenderingContext2D;
    actx = after.getContext('2d', { alpha: false }) as OffscreenCanvasRenderingContext2D;
    for (const c of [bctx, actx]) { c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high'; }
  }
  let adapterName = '';
  let adapterInfo: { vendor?: string; architecture?: string; device?: string; description?: string } | undefined;
  let error: string | undefined;
  try {
    const gpu = (navigator as any).gpu as GPU | undefined;
    if (!gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter (GPU blocked or unsupported).');
    const ai = (adapter as any).info ?? {};
    adapterName = [ai.vendor, ai.architecture, ai.description].filter(Boolean).join(' ');
    adapterInfo = { vendor: ai.vendor, architecture: ai.architecture, device: ai.device, description: ai.description };
    const want = (k: keyof GPUSupportedLimits, v: number) => Math.min(v, (adapter.limits as any)[k] as number);
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: want('maxStorageBufferBindingSize', 1 << 30),
        maxBufferSize: want('maxBufferSize', 1 << 30),
        maxTextureDimension2D: want('maxTextureDimension2D', 16384),
      },
    });
    device.lost.then(i => post({ type: 'error', message: `The GPU device was lost (${i.reason}): ${i.message}` }));
    device.onuncapturederror = e => console.error('[webgpu]', e.error.message);
  } catch (e) {
    error = (e as Error).message ?? String(e);
  }
  post({ type: 'ready', caps: { webgpu: !!device, adapter: adapterName, error, adapterInfo, platform: navigator.platform, cores: navigator.hardwareConcurrency } });
}

async function open(m: Extract<EngineIn, { type: 'open' }>) {
  stopPlayback();
  seekCtl?.abort();
  seekGen++;
  const release = await acquireDecoder();
  let ok = false;
  try {
    releaseHeld(); heldPres = -1;
    warper?.destroy(); warper = null; plan = null; planIdx = null; warperPlanId = -1;
    dropPreviewWarper();
    session = null; index = null; video = null;
    file = m.file; info = m.info;
    let v = info.tracks.find(t => t.kind === 'video' && !!t.codecString && (t.width ?? 0) >= 320) ?? info.tracks.find(t => t.kind === 'video') ?? null;
    if (!v) { post({ type: 'open-error', message: 'This file has no video track.' }); return; }
    const notes: string[] = [];
    // truncated / incompletely copied files: only the samples whose bytes are all in the file can be decoded
    const complete = completeSamplePrefix(v, file.size);
    if (complete < v.sampleCount) {
      if (complete < 2) {
        post({ type: 'open-error', message: 'This file is incomplete: its video data is missing (it may not have finished copying). Copy the clip from the SD card again.', details: { kind: 'truncated', samples: v.sampleCount, completeSamples: complete, fileBytes: file.size } });
        return;
      }
      const whole = v.cts[v.sampleCount - 1] - v.cts[0];
      v = trimTrack(v, complete);
      const have = v.cts[complete - 1] - v.cts[0];
      notes.push(`This file is incomplete — only the first ${have.toFixed(1)} s of ${whole.toFixed(1)} s of video is there (it may not have finished copying). Stillpoint will use that part.`);
    }
    video = v;
    const idx = new FrameIndex(v);
    const facts = streamFacts(v, 1 / idx.frameDur);
    // which sync samples are clean random-access points (reads a few KB per keyframe)
    try {
      const f = file;
      idx.applyRapKinds(await classifySyncSamples(v, (o, s) => readRanges(f, o, s, { concurrency: 16 })));
    } catch (e) { console.warn('[stillpoint] keyframe scan failed', e); }
    const sup = await supportedVariants(v);
    if (!sup.variants.length) {
      post({ type: 'open-error', message: cannotDecodeMessage(facts, 'unsupported'), details: { kind: 'unsupported', stream: { ...facts, description: describeStream(facts), samples: v.sampleCount, fileBytes: file.size }, rejected: sup.rejected, rap: idx.rapCensus } });
      return;
    }
    const s = new DecoderSession(file, v, idx, readSamples, sup.variants, {
      limits: decodeLimits(isApplePlatform()), stallMs: debug.stallMs ?? 15000, preflightStallMs: Math.min(5000, debug.stallMs ?? 5000), fault: parseFault(debug.fault),
    });
    s.hwSupported = sup.hwSupported;
    const pf = await s.preflight();
    if (!pf.ok) {
      const last = pf.tried.filter(t => t.error).pop()?.error ?? '';
      post({ type: 'open-error', message: cannotDecodeMessage(facts, 'failed') + (last ? ` (Browser said: “${last}”.)` : ''), details: { ...s.report(new Error(last || 'pre-flight failed'), 0, 'preflight'), kind: 'unsupported', preflight: pf, rejected: sup.rejected } });
      return;
    }
    index = idx;
    session = s;
    s.onSwitch = () => { if (session === s) post({ type: 'decoder', decoder: decoderInfo(s) }); };
    post({
      type: 'opened', frames: idx.n, fps: 1 / idx.frameDur, duration: idx.pts[idx.n - 1] - idx.pts[0] + idx.frameDur,
      decoder: decoderInfo(s, { ms: pf.ms, frames: pf.frames, tried: pf.tried }), notes,
    });
    ok = true;
  } catch (e) {
    if (!isAbort(e)) post({ type: 'open-error', message: e instanceof DecodeFailure ? e.message : 'Couldn’t open this clip for decoding: ' + errText(e), details: detailsOf(e, 'open') });
  } finally {
    release();
  }
  if (ok) await seek(0);
}

async function applyPlan(p: Plan, id: number) {
  if (!device) return;
  const w = await Warper.create(device, p, { kernel: 'lanczos3' });
  // swap only if nothing newer arrived meanwhile
  if (id < warperPlanId) { w.destroy(); return; }
  const old = warper;
  warper = w; plan = p; planIdx = planIndexer(p); warperPlanId = id;
  if (!playing) dropPreviewWarper(); // (while playing, the loop swaps it for the new plan's)
  if (old) { if (exporting) retired.push(old); else old.destroy(); }
  post({ type: 'plan-applied', id });
  if (!playing && !exporting) redrawHeld();
}

// ─────────── scrub / play ───────────

async function seek(pres: number) {
  if (!session || !index) return;
  seekCtl?.abort();
  const ctl = (seekCtl = new AbortController());
  const gen = ++seekGen;
  const release = await acquireDecoder();
  try {
    const s = session, idx = index;
    if (ctl.signal.aborted || gen !== seekGen || playing || exporting || !s || !idx) return;
    pres = Math.max(0, Math.min(idx.n - 1, pres | 0));
    const t0 = performance.now();
    releaseHeld(); // never hold a frame from a previous decoder while a new one runs
    let src: VideoFrame;
    try {
      src = await decodeOne(s, pres, ctl.signal);
    } catch (e) {
      if (!isAbort(e) && !ctl.signal.aborted && gen === seekGen) postDecodeError('Could not show that frame', e, 'preview');
      return;
    }
    if (gen !== seekGen || playing || ctl.signal.aborted || s !== session) { src.close(); return; }
    let w: VideoFrame | null = null;
    try { w = warpFor(src, pres); } catch (e) { post({ type: 'error', message: 'Warp failed: ' + (e as Error).message }); }
    drawPair(src, w);
    hold(src, pres, w);
    post({ type: 'frame', pres, t: idx.pts[pres], hasWarp: !!w, ms: performance.now() - t0 });
  } finally {
    release();
    if (seekCtl === ctl) seekCtl = null;
  }
}

let playStats = { drawn: 0, dropped: 0, t0: 0 };

function playingMsg(on: boolean): EngineOut {
  const secs = (performance.now() - playStats.t0) / 1000;
  return { type: 'playing', playing: on, stats: on ? undefined : { drawn: playStats.drawn, dropped: playStats.dropped, seconds: secs } };
}

function stopPlayback() {
  playGen++;
  playCtl?.abort();
  playCtl = null;
  if (playing) { playing = false; post(playingMsg(false)); }
}

async function play(from: number, to: number, loop: boolean) {
  if (!session || !index || exporting) return;
  stopPlayback();
  seekCtl?.abort();
  const gen = ++playGen;
  const ctl = (playCtl = new AbortController());
  playing = true;
  playStats = { drawn: 0, dropped: 0, t0: performance.now() };
  post(playingMsg(true));
  const s = session, idx = index;
  let start = Math.max(0, Math.min(idx.n - 1, from));
  const end = Math.max(start, Math.min(idx.n - 1, to));
  const frameMs = idx.frameDur * 1000;
  const release = await acquireDecoder();
  let stream: FrameStream | null = null;
  try {
    if (gen !== playGen || s !== session) return;
    releaseHeld();
    let pw = await previewWarper();
    let pwPlan = warperPlanId;
    if (gen !== playGen) return;
    do {
      stream = s.frames(start, end, { purpose: 'playback', signal: ctl.signal, gaps: 'skip', ...s.opts.limits.playback });
      let wall0 = -1;
      const media0 = idx.pts[start];
      let lastPost = 0, lastDraw = 0;
      for (;;) {
        if (gen !== playGen) return;
        const got = await stream.next();
        if (!got) break;
        const { frame: f, pres } = got;
        // real-time pacing: wait when early, drop (skip warp + draw) when more than a frame late; after a long hiccup
        // (a decoder recovering from an error or a stall) re-anchor the clock instead of dropping everything to catch up
        if (wall0 < 0 || performance.now() - (wall0 + (idx.pts[pres] - media0) * 1000) > 1000) wall0 = performance.now() - (idx.pts[pres] - media0) * 1000;
        const due = wall0 + (idx.pts[pres] - media0) * 1000;
        const wait = due - performance.now();
        if (wait > 2) await new Promise(r => setTimeout(r, wait));
        if (gen !== playGen) { f.close(); return; }
        const late = performance.now() - due;
        if (late > frameMs && pres < end && performance.now() - lastDraw < 250) { f.close(); playStats.dropped++; continue; }
        if (pwPlan !== warperPlanId) { pwPlan = warperPlanId; pw = await previewWarper(); if (gen !== playGen) { f.close(); return; } }
        let w: VideoFrame | null = null;
        try { w = warpFor(f, pres, pw); } catch { w = null; }
        drawPair(f, w);
        hold(f, pres, w);
        playStats.drawn++;
        const now = (lastDraw = performance.now());
        if (now - lastPost > 66) { lastPost = now; post({ type: 'frame', pres, t: idx.pts[pres], hasWarp: !!w, ms: 0 }); }
      }
      stream.close();
      stream = null;
      start = Math.max(0, Math.min(idx.n - 1, from));
    } while (loop && gen === playGen);
  } catch (e) {
    if (!isAbort(e) && gen === playGen) postDecodeError('Playback stopped', e, 'playback');
  } finally {
    stream?.close();
    release();
    if (gen === playGen) {
      playing = false;
      playCtl = null;
      post(playingMsg(false));
      if (heldSrc) post({ type: 'frame', pres: heldPres, t: idx.pts[heldPres], hasWarp: !!lastWarped, ms: 0 });
    }
    // stopped before a frame was drawn: bring back a held frame (re-warpable) unless a seek is already on its way
    if (!heldSrc && heldPres >= 0 && !playing && !exporting && !seekCtl && s === session) void seek(heldPres);
  }
}

// ─────────── export ───────────

async function probeEncoder(m: Extract<EngineIn, { type: 'probe-encoder' }>) {
  const req = { width: m.width, height: m.height, fps: m.fps, bitrate: m.bitrate };
  const [e, options] = await Promise.all([chooseEncoder({ ...req, prefer: m.prefer }), codecOptions(req)]);
  post({ type: 'encoder', id: m.id, codec: e?.config.codec ?? null, label: e?.label ?? '', hardware: e?.hardware === 'prefer-hardware', width: m.width, height: m.height, fps: m.fps, options });
}

async function runExport(s: ExportSettings) {
  if (!file || !info || !video || !index || !session || !plan || !warper || !device) {
    post({ type: 'export-error', message: session ? 'Nothing to export yet — the camera path is still being planned.' : 'This clip couldn’t be opened for decoding, so it can’t be exported.', cancelled: false });
    return;
  }
  stopPlayback();
  seekCtl?.abort();
  const ctl = new AbortController();
  exporting = ctl;
  const lastPres = heldPres;
  const release = await acquireDecoder();
  const sess = session;
  const dev = device;
  let own: Warper | null = null;
  try {
    // free the held preview frame before the export's decoder starts (the hardware decoder pool is small)
    releaseHeld();
    if (ctl.signal.aborted) throw new DOMException('Export cancelled', 'AbortError');
    const o = s.output ?? { size: 'source', aspect: 'source', fps: 'source', timing: 'realtime', motionBlur: false };
    // output size: the plan (built for this aspect) rescaled — same camera path, focal x outW / plan.outW
    const base = plan, baseWarper = warper;
    const geo = outputGeometry(base.srcW, base.srcH, o.size, o.aspect);
    let expPlan: Plan = base, expWarper: Warper = baseWarper;
    if (geo.outW !== base.outW || geo.outH !== base.outH) {
      try { expPlan = rescalePlan(base, geo.outW, geo.outH); } catch {
        throw new Error('The camera path is still being re-planned for this aspect ratio — try again in a moment.');
      }
      own = expWarper = await Warper.create(dev, expPlan, { kernel: 'lanczos3' });
    }
    const first = Math.max(0, s.first), last = Math.min(index.n - 1, s.last);
    const identity = o.fps === 'source';
    const schedule = buildSchedule(index.pts.subarray(first, last + 1), {
      fps: o.fps, timing: o.timing, motionBlur: o.motionBlur && o.timing === 'realtime' ? 'natural' : 'off', shutterDeg: 180,
    });
    const outFps = schedule.fps ?? 1 / index.frameDur;
    const enc = await chooseEncoder({ width: geo.outW, height: geo.outH, fps: outFps, bitrate: s.bitrate, prefer: s.prefer });
    if (!enc) throw new Error(`This browser cannot encode video at ${geo.outW}×${geo.outH} ${outFps.toFixed(2)} fps.`);
    if (s.sink.kind !== 'fsa') await clearOpfsExports();
    const sink = await createSink(s.sink);
    const cs = colorSpaceOf(video);
    const sdr709 = !cs || ((cs.primaries ?? 'bt709') === 'bt709' && (cs.transfer ?? 'bt709') === 'bt709');
    const result = await renderClip({
      file, info, video, index, plan: expPlan, warper: expWarper, schedule, identity, readSamples, encoder: enc, sink,
      frames: (a, b) => sess.frames(a, b, { purpose: 'export', signal: ctl.signal, gaps: 'fail', ...sess.opts.limits.export }),
      first, last, includeAudio: s.includeAudio, signal: ctl.signal,
      colorSpace: sdr709 ? { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: false } : undefined,
      blendTransfer: blendTransferFor(cs),
      // natural motion blur: 180° synthetic shutter (gyro sub-frame warps between the blended frames)
      shutterS: o.motionBlur && o.timing === 'realtime' && o.fps !== 'source' && debug.blur !== 'frames' ? 0.5 / outFps : undefined,
      onProgress: p => post({ type: 'export-progress', p }),
      onPreview: (src, w, pres) => { drawPair(src, w); post({ type: 'frame', pres, t: index!.pts[pres], hasWarp: true, ms: 0 }); },
      previewEveryMs: 200,
    });
    post({ type: 'export-done', result });
  } catch (e) {
    const cancelled = isAbort(e) || ctl.signal.aborted;
    if (!cancelled) console.error(e);
    const decodeFail = e instanceof DecodeFailure;
    post({
      type: 'export-error', cancelled,
      message: cancelled ? 'Export cancelled.' : decodeFail ? `Export stopped: ${e.message}` : ((e as Error).message ?? String(e)),
      details: cancelled ? undefined : detailsOf(e, 'export'),
    });
  } finally {
    own?.destroy();
    release();
    exporting = null;
    for (const w of retired.splice(0)) if (w !== warper) w.destroy();
    if (lastPres >= 0) void seek(lastPres);
  }
}

// ─────────── dispatch ───────────

let chain: Promise<unknown> = Promise.resolve();
const serial = (fn: () => Promise<unknown>) => (chain = chain.then(fn, fn));

scope.onmessage = (ev: MessageEvent<EngineIn>) => {
  const m = ev.data;
  switch (m.type) {
    case 'init': serial(() => init(m)); break;
    case 'resize':
      if (before && after) {
        for (const c of [before, after]) { c.width = Math.max(2, m.w | 0); c.height = Math.max(2, m.h | 0); }
        for (const c of [bctx, actx]) if (c) { c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high'; }
        drawPair(heldSrc, lastWarped);
      }
      break;
    case 'open': serial(() => open(m)); break;
    case 'plan': serial(() => applyPlan(m.plan, m.id)); break;
    case 'seek': if (!exporting) { stopPlayback(); void seek(m.pres); } break;
    case 'play': void play(m.from, m.to, m.loop); break;
    case 'pause': stopPlayback(); break;
    case 'probe-encoder':
      serial(() => probeEncoder(m));
      break;
    case 'export': void runExport(m.settings); break;
    case 'cancel': exporting?.abort(); break;
  }
};
