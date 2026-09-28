/// <reference lib="webworker" />
/**
 * Engine worker: owns the GPUDevice, the Warper, the two preview canvases (transferred from the page), and runs
 * scrubbing, real-time preview playback and the export pipeline — so the page stays smooth whatever the GPU or the
 * codecs are doing.
 */
import type { Mp4Info, Mp4Track, Plan } from '../types';
import type { WarperLike } from '../ui/contracts';
import { readSamples } from '../mp4';
import { Warper } from '../gpu/warp';
import { FrameIndex, SequentialDecoder, colorSpaceOf, decodeOne, supportedDecoderConfig } from './decode';
import { chooseEncoder, type EncoderChoice } from './encode';
import { clearOpfsExports, createSink } from './mux';
import { planIndexer, renderClip } from './pipeline';
import type { EngineIn, EngineOut, ExportSettings } from './protocol';

const scope = self as unknown as DedicatedWorkerGlobalScope;
const post = (m: EngineOut, transfer: Transferable[] = []) => scope.postMessage(m, transfer);

let device: GPUDevice | null = null;
let before: OffscreenCanvas | null = null;
let after: OffscreenCanvas | null = null;
let bctx: OffscreenCanvasRenderingContext2D | null = null;
let actx: OffscreenCanvasRenderingContext2D | null = null;

let file: File | null = null;
let info: Mp4Info | null = null;
let video: Mp4Track | null = null;
let index: FrameIndex | null = null;
let decCfg: VideoDecoderConfig | null = null;
let plan: Plan | null = null;
let planIdx: ((t: number) => number) | null = null;
let warper: WarperLike | null = null;
let warperPlanId = -1;

/** the last source frame shown (kept to re-warp instantly when the plan changes) */
let heldSrc: VideoFrame | null = null;
let heldPres = -1;
let lastWarped: VideoFrame | null = null;

let seekGen = 0;
let playing = false;
let playGen = 0;
let exporting: AbortController | null = null;
/** warpers replaced while an export was using them; destroyed when it ends */
const retired: WarperLike[] = [];

// ─────────── drawing ───────────

function drawContain(ctx: OffscreenCanvasRenderingContext2D, img: CanvasImageSource, iw: number, ih: number) {
  const cw = ctx.canvas.width, ch = ctx.canvas.height;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, cw, ch);
  const s = Math.min(cw / iw, ch / ih);
  const w = iw * s, h = ih * s;
  ctx.drawImage(img, (cw - w) / 2, (ch - h) / 2, w, h);
}

function drawPair(src: VideoFrame | null, warped: VideoFrame | null) {
  if (bctx && src) drawContain(bctx, src, src.displayWidth, src.displayHeight);
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

function warpFor(src: VideoFrame, pres: number, w: WarperLike | null = warper): VideoFrame | null {
  if (!w || !plan || !index || !planIdx) return null;
  return w.warp(src, planIdx(index.pts[pres]));
}

/**
 * Playback warper: the same plan rendered at (about) the preview canvas size — identical geometry, because the
 * row matrices act on normalized rays and only outW/outH/outFx scale — so real-time preview doesn't warp 8-11 MP per
 * frame just to downscale it. Built lazily per plan and canvas size.
 */
let pvWarper: WarperLike | null = null;
let pvKey = '';
async function previewWarper(): Promise<WarperLike | null> {
  if (!plan || !device || !after) return warper;
  const s = Math.min(1, Math.max(after.width / plan.outW, after.height / plan.outH));
  if (s > 0.8) return warper;
  const outW = Math.max(2, Math.round((plan.outW * s) / 2) * 2), outH = Math.max(2, Math.round((plan.outH * s) / 2) * 2);
  const key = `${warperPlanId}:${outW}x${outH}`;
  if (pvWarper && key === pvKey) return pvWarper;
  const k = outW / plan.outW;
  const scaled: Plan = { ...plan, outW, outH, outFx: plan.outFx.map(f => f * k) };
  try {
    const w = await Warper.create(device, scaled, { kernel: 'catmullrom' });
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

// ─────────── init / open / plan ───────────

async function init(m: Extract<EngineIn, { type: 'init' }>) {
  if (m.before && m.after) {
    before = m.before; after = m.after;
    bctx = before.getContext('2d', { alpha: false }) as OffscreenCanvasRenderingContext2D;
    actx = after.getContext('2d', { alpha: false }) as OffscreenCanvasRenderingContext2D;
    for (const c of [bctx, actx]) { c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high'; }
  }
  let adapterName = '';
  let error: string | undefined;
  try {
    const gpu = (navigator as any).gpu as GPU | undefined;
    if (!gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter (GPU blocked or unsupported).');
    const ai = (adapter as any).info ?? {};
    adapterName = [ai.vendor, ai.architecture, ai.description].filter(Boolean).join(' ');
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
  post({ type: 'ready', caps: { webgpu: !!device, adapter: adapterName, error } });
}

async function open(m: Extract<EngineIn, { type: 'open' }>) {
  stopPlayback();
  heldSrc?.close(); heldSrc = null; heldPres = -1;
  lastWarped?.close(); lastWarped = null;
  warper?.destroy(); warper = null; plan = null; planIdx = null; warperPlanId = -1;
  dropPreviewWarper();
  file = m.file; info = m.info;
  video = info.tracks.find(t => t.kind === 'video' && !!t.codecString && (t.width ?? 0) >= 320) ?? info.tracks.find(t => t.kind === 'video') ?? null;
  if (!video) { post({ type: 'open-error', message: 'This file has no video track.' }); return; }
  index = new FrameIndex(video);
  const s = await supportedDecoderConfig(video);
  if (!s.config) { post({ type: 'open-error', message: s.reason ?? 'Unsupported video codec.' }); return; }
  decCfg = s.config;
  post({ type: 'opened', frames: index.n, fps: 1 / index.frameDur, duration: index.pts[index.n - 1] - index.pts[0] + index.frameDur, decoder: decCfg.codec, hardware: decCfg.hardwareAcceleration === 'prefer-hardware' });
  await seek(0);
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
  if (!file || !video || !index || !decCfg) return;
  const gen = ++seekGen;
  pres = Math.max(0, Math.min(index.n - 1, pres | 0));
  const t0 = performance.now();
  let src: VideoFrame;
  try {
    src = await decodeOne(file, video, index, readSamples, decCfg, pres);
  } catch (e) {
    if (gen === seekGen) post({ type: 'error', message: 'Could not decode that frame: ' + (e as Error).message });
    return;
  }
  if (gen !== seekGen || playing) { src.close(); return; }
  let w: VideoFrame | null = null;
  try { w = warpFor(src, pres); } catch (e) { post({ type: 'error', message: 'Warp failed: ' + (e as Error).message }); }
  drawPair(src, w);
  hold(src, pres, w);
  post({ type: 'frame', pres, t: index.pts[pres], hasWarp: !!w, ms: performance.now() - t0 });
}

let playStats = { drawn: 0, dropped: 0, t0: 0 };

function playingMsg(on: boolean): EngineOut {
  const secs = (performance.now() - playStats.t0) / 1000;
  return { type: 'playing', playing: on, stats: on ? undefined : { drawn: playStats.drawn, dropped: playStats.dropped, seconds: secs } };
}

function stopPlayback() {
  playGen++;
  if (playing) { playing = false; post(playingMsg(false)); }
}

async function play(from: number, to: number, loop: boolean) {
  if (!file || !video || !index || !decCfg || exporting) return;
  stopPlayback();
  const gen = ++playGen;
  playing = true;
  playStats = { drawn: 0, dropped: 0, t0: performance.now() };
  post(playingMsg(true));
  const fileRef = file, trackRef = video, idx = index;
  let start = Math.max(0, Math.min(idx.n - 1, from));
  const end = Math.max(start, Math.min(idx.n - 1, to));
  const frameMs = idx.frameDur * 1000;
  let pw = await previewWarper();
  let pwPlan = warperPlanId;
  if (gen !== playGen) return;
  try {
    do {
      const [d0, d1] = idx.decodeSpan(start, end);
      const dec = new SequentialDecoder(fileRef, trackRef, idx, readSamples, decCfg, d0, d1, { maxFrames: 4 });
      let wall0 = -1;
      const media0 = idx.pts[start];
      let lastPost = 0, lastDraw = 0;
      try {
        for (;;) {
          if (gen !== playGen) return;
          const f = await dec.next();
          if (!f) break;
          const pres = idx.presOfTimestamp(f.timestamp);
          if (pres < start || pres > end) { f.close(); continue; }
          if (wall0 < 0) wall0 = performance.now();
          // real-time pacing: wait when early, drop (skip warp + draw) when more than a frame late
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
      } finally { dec.close(); }
      start = Math.max(0, Math.min(idx.n - 1, from));
    } while (loop && gen === playGen);
  } catch (e) {
    post({ type: 'error', message: 'Playback stopped: ' + (e as Error).message });
  } finally {
    if (gen === playGen) {
      playing = false;
      post(playingMsg(false));
      if (heldSrc) post({ type: 'frame', pres: heldPres, t: idx.pts[heldPres], hasWarp: !!lastWarped, ms: 0 });
    }
  }
}

// ─────────── export ───────────

async function encoderFor(bitrate: number, prefer?: ExportSettings['prefer']): Promise<EncoderChoice | null> {
  if (!plan || !index) return null;
  return chooseEncoder({ width: plan.outW, height: plan.outH, fps: 1 / index.frameDur, bitrate, prefer });
}

async function runExport(s: ExportSettings) {
  if (!file || !info || !video || !index || !decCfg || !plan || !warper) {
    post({ type: 'export-error', message: 'Nothing to export yet — the camera path is still being planned.', cancelled: false });
    return;
  }
  stopPlayback();
  const ctl = new AbortController();
  exporting = ctl;
  // free the held preview frame (the hardware decoder pool is small)
  heldSrc?.close(); heldSrc = null;
  lastWarped?.close(); lastWarped = null;
  const lastPres = heldPres;
  try {
    const enc = await encoderFor(s.bitrate, s.prefer);
    if (!enc) throw new Error('This browser cannot encode video at ' + plan.outW + '×' + plan.outH + '.');
    if (s.sink.kind !== 'fsa') await clearOpfsExports();
    const sink = await createSink(s.sink);
    const cs = colorSpaceOf(video);
    const sdr709 = !cs || ((cs.primaries ?? 'bt709') === 'bt709' && (cs.transfer ?? 'bt709') === 'bt709');
    const result = await renderClip({
      file, info, video, index, plan, warper, readSamples: readSamples, decoderConfig: decCfg, encoder: enc, sink,
      first: Math.max(0, s.first), last: Math.min(index.n - 1, s.last), includeAudio: s.includeAudio, signal: ctl.signal,
      colorSpace: sdr709 ? { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: false } : undefined,
      onProgress: p => post({ type: 'export-progress', p }),
      onPreview: (src, w, pres) => { drawPair(src, w); post({ type: 'frame', pres, t: index!.pts[pres], hasWarp: true, ms: 0 }); },
      previewEveryMs: 200,
    });
    post({ type: 'export-done', result });
  } catch (e) {
    const cancelled = (e as DOMException)?.name === 'AbortError' || ctl.signal.aborted;
    if (!cancelled) console.error(e);
    post({ type: 'export-error', message: cancelled ? 'Export cancelled.' : ((e as Error).message ?? String(e)), cancelled });
  } finally {
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
      // serialized behind 'plan' so the encoder is probed for the plan's output size
      serial(() => encoderFor(m.bitrate, m.prefer).then(e => post({ type: 'encoder', codec: e?.config.codec ?? null, label: e?.label ?? '', hardware: e?.hardware === 'prefer-hardware', width: plan?.outW ?? 0, height: plan?.outH ?? 0, fps: index ? 1 / index.frameDur : 0 })));
      break;
    case 'export': void runExport(m.settings); break;
    case 'cancel': exporting?.abort(); break;
  }
};
