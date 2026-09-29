/**
 * Stillpoint web — WebGPU warp (owner: GPU agent). The browser twin of shaders/warp.metal; see warp.wgsl for the math.
 *
 *   const warper = await Warper.create(device, plan, { kernel: 'lanczos3' });
 *   const out = warper.warp(decodedFrame, k);   // VideoFrame, same timestamp/duration, ready for VideoEncoder
 *   decodedFrame.close(); encoder.encode(out); out.close();
 *
 * Per frame (one submit, no CPU readback): importExternalTexture(frame) (zero copy) -> compute: blind colour-transfer
 * calibration (accumulated on the GPU) + per-frame chroma-mode test + conversion to camera-domain float intermediates
 * (Y' full res; CbCr per 2x2 block for frames imported as 4:2:0 planes, per pixel for frames the browser hands over
 * as RGB, e.g. 10-bit HEVC on macOS) -> compute warp (fp32 geometry, luma per pixel, chroma per 2x2 block at its
 * chroma site, Lanczos-3 / Catmull-Rom taps via textureGather) straight into the OffscreenCanvas texture (bgra8unorm
 * when the device has 'bgra8unorm-storage' — request it, it saves Chrome a copy on macOS — else rgba8unorm) ->
 * new VideoFrame(canvas, {timestamp, duration}). Each VideoFrame is a snapshot: warp() can be called again at once,
 * and several Warpers (preview + export) may be used interleaved on one device.
 * GPU memory: intermediates ~10 B/source pixel (83 MB at 3840x2160), shared by all Warpers of a device and source
 * size, + the canvas (4 B/output pixel, x2-3 in flight). Tests: test/gpu.run.mjs (selftest | golden | bench |
 * scaled | shimmer | blend | benchscaled).
 *
 * Output size != source size: the Warper renders whatever plan.outW x plan.outH (+ plan.outFx) describe. When the
 * output is SPARSER than the source (s = source px per output px at the densest point > 1, e.g. 4K -> 1080p / 720p)
 * one interpolation tap per output pixel would alias and shimmer, so the warp runs on a supersampled grid of the
 * output (>= source density) and a separable Lanczos-3 downscale (kernel stretched to the output pixel footprint)
 * produces the output — the same as warping at full resolution and then downscaling, without the full-res frame.
 * Extra GPU memory for that path: 6 B per intermediate pixel + 6 B per (output column x intermediate row), e.g. from
 * a 4K O3 plan: 1080p and 720p both use a 3082 x 1734 grid (~ source density over the output's field of view):
 * 52 MB / 45 MB. Measured (M-series, Chrome): 1080p 2.6 ms, 720p 2.8 ms GPU per frame vs 3.0 ms for a 4K output.
 * Magnified or same-density outputs keep the single-pass warp (unchanged, bit-identical).
 * opts.antialias: 'auto' (default) | 'off' (single-pass even when minifying: aliases; tests) | factor (force s).
 * Retimed exports: warp(frame, k, {timestamp, duration}) stamps the output; warpTo() + Blender (blend.ts) make
 * motion-blurred frames; retime_render.ts drives both from an OutputSchedule.
 *
 * Colour classes. What Chrome does to an imported frame follows THAT frame's colorSpace tag, and one decoder does not tag
 * all its frames alike: VideoToolbox (Chrome, macOS) hands out most frames of a BT.709 clip as 'bt709' (converted with
 * Apple's gamma 1.961) but, in retimed exports, a few as 'iec61966-2-1' (sRGB: no transfer change) with the same
 * pixel data. So the transfer calibration is kept per class of frames — pixel format + transfer tag — each with its
 * own GPU evidence and a default derived from the tag ('iec61966-2-1' -> sRGB, 'linear' -> linear, else the
 * platform guess), and every frame is converted with its own class's transfer. One calibration for all frames
 * applied the 1.961 inverse to the sRGB-tagged frames: ~9 luma codes darker midtones = flicker (test/export.flicker.mjs).
 *
 * Colour contract (measured in Chrome 153 / macOS, test/gpu.harness.ts 'extfit' + 'enccolor' + 'golden'):
 *   the canvas holds R'G'B' = source Y'CbCr through the source matrix (no transfer change), and VideoEncoder converts
 *   a canvas frame with the BT.709 matrix to limited range, so a BT.709 source round-trips to the same Y'CbCr codes
 *   (+-0.5 LSB of 8-bit quantisation; 4:2:0 chroma is computed once per 2x2 block at the chroma site, like the Metal
 *   kernel). The encoder's decoderConfig.colorSpace reports transfer 'iec61966-2-1'; a muxer should write
 *   colr = bt709/bt709/bt709 (limited) to describe the pixels faithfully. What an 8-bit RGB canvas cannot carry:
 *   super-whites / sub-blacks and out-of-gamut Y'CbCr clip (as any player clips them on display); HDR (PQ/HLG) is
 *   tone-mapped by the browser before we see it.
 *
 * Frame k of the plan: the caller picks the plan record for the decoded frame (plan.framePts[k] ~ frame.timestamp).
 */
import type { Plan } from '../types';
import WGSL from './warp.wgsl?raw';

export type WarpKernel = 'lanczos3' | 'catmullrom' | 'bilinear';
/** Source transfer candidates (index = shader code). 'srgb' = Chrome applied no transfer change. */
export const TRANSFERS = ['srgb', 'gamma1.961', 'bt709', 'gamma2.2', 'gamma2.4', 'linear'] as const;
export type TransferName = (typeof TRANSFERS)[number];

export interface WarperOptions {
  kernel: WarpKernel;
  /** geometry evaluations per pixel (default 3 = centre row, fixed point, secant — as warp.metal) */
  iters?: number;
  /** 'auto' (default) = detect on the GPU; otherwise force the transfer Chrome applied to imported frames */
  transfer?: 'auto' | TransferName;
  /** debug: GPU timestamps per pass (device must have 'timestamp-query'); read with profile() */
  profile?: boolean;
  /** debug/benchmark: fetch taps with textureGather (default true) or plain textureLoad */
  gather?: boolean;
  /** debug: force the chroma mode instead of the per-frame GPU test */
  chroma?: 'auto' | 'planar' | 'rgb';
  /** debug: 'copy' forces warp -> intermediate texture -> copy into the canvas (the fallback path) */
  canvasPath?: 'auto' | 'copy';
  /** anti-aliased downscaling for outputs sparser than the source: 'auto' (default: supersampled warp + Lanczos-3
   *  downscale whenever planMinification(plan) > 1), 'off' (always single-pass; aliases when minifying), or a
   *  supersampling factor to force (<= 1 = single-pass) */
  antialias?: 'auto' | 'off' | number;
}

/** How the Warper renders its output: single-pass warp, or supersampled warp (iW x iH) + Lanczos-3 downscale. */
export interface WarpScaling {
  mode: 'direct' | 'supersample';
  /** source px per output px at the densest point of the plan (max over records) */
  minification: number;
  /** supersampled grid (== outW/outH when direct) and its per-axis scale over the output */
  interW: number;
  interH: number;
  sx: number;
  sy: number;
}

/** Timing of an output VideoFrame (microseconds, as VideoFrame.timestamp / duration). */
export interface FrameTiming { timestamp: number; duration?: number }

const MAX_SUPERSAMPLE = 8;

/** Source pixels per output pixel at the densest point of the mapping, max over all records: the lens centre, where
 *  the KB4 / pinhole projection has (fx, fy) px/rad and the rectilinear output outFx px/rad (away from the centre the
 *  fisheye compresses and the rectilinear output stretches, so the ratio only drops). > 1 = the output minifies. */
export function planMinification(plan: Plan): number {
  let minFx = Infinity;
  const o = plan.outFx;
  for (let i = 0; i < o.length; i++) if (o[i] > 0 && o[i] < minFx) minFx = o[i];
  if (!Number.isFinite(minFx)) return 1;
  return Math.max(plan.lens.fx, plan.lens.fy) / minFx;
}

/** The scaling the Warper uses for `plan` (pure function of the plan, the option and the texture size limit). */
export function warpScaling(plan: Plan, antialias: WarperOptions['antialias'] = 'auto', maxDim = 8192): WarpScaling {
  const m = planMinification(plan);
  const s = antialias === 'off' ? 1 : typeof antialias === 'number' ? antialias : m;
  const direct: WarpScaling = { mode: 'direct', minification: m, interW: plan.outW, interH: plan.outH, sx: 1, sy: 1 };
  if (!(s > 1 + 1e-6)) return direct;
  const sc = Math.min(s, MAX_SUPERSAMPLE, maxDim / plan.outW, maxDim / plan.outH);
  if (!(sc > 1 + 1e-6)) return direct;
  const even = (v: number) => Math.min(maxDim, v + (v & 1));
  const interW = even(Math.ceil(plan.outW * sc - 1e-6)), interH = even(Math.ceil(plan.outH * sc - 1e-6));
  return { mode: 'supersample', minification: m, interW, interH, sx: interW / plan.outW, sy: interH / plan.outH };
}

/** Lanczos-3 window sinc(x) sinc(x/3), |x| < 3. */
function lanczos3(x: number): number {
  const a = Math.abs(x);
  if (a < 1e-12) return 1;
  if (a >= 3) return 0;
  return 3 * Math.sin(Math.PI * a) * Math.sin(Math.PI * a / 3) / (Math.PI * Math.PI * a * a);
}

/** The downscale taps of the supersampled path (float64, normalised; see warp.wgsl `DT`): per output luma column,
 *  chroma column, luma row, chroma row: [first texel, nTaps weights]. Centres (pixel-centre convention):
 *  luma c = s(p+0.5)-0.5 intermediate px; chroma site (2a, 2b+0.5) -> chroma texel (s(2a+0.5)-0.5)/2,
 *  (s(2b+1)-1)/2; kernel stretch s = sx (columns) / sy (rows) in both texel units. */
export function downscaleTable(outW: number, outH: number, sx: number, sy: number): { nTaps: number; data: Float32Array<ArrayBuffer> } {
  const nTaps = Math.floor(6 * Math.max(sx, sy)) + 2;
  const cW = Math.ceil(outW / 2), cH = Math.ceil(outH / 2);
  const entries: [number, number][] = [];                   // (centre, stretch)
  for (let x = 0; x < outW; x++) entries.push([sx * (x + 0.5) - 0.5, sx]);
  for (let a = 0; a < cW; a++) entries.push([0.5 * (sx * (2 * a + 0.5) - 0.5), sx]);
  for (let y = 0; y < outH; y++) entries.push([sy * (y + 0.5) - 0.5, sy]);
  for (let b = 0; b < cH; b++) entries.push([0.5 * (sy * (2 * b + 1) - 1), sy]);
  const data = new Float32Array(entries.length * (nTaps + 1));
  const w = new Float64Array(nTaps);
  entries.forEach(([c, s], e) => {
    const i0 = Math.ceil(c - 3 * s), i1 = Math.floor(c + 3 * s);
    if (i1 - i0 + 1 > nTaps) throw new Error('downscaleTable: tap overflow');
    let sum = 0;
    w.fill(0);
    for (let i = i0; i <= i1; i++) { w[i - i0] = lanczos3((i - c) / s); sum += w[i - i0]; }
    const o = e * (nTaps + 1);
    data[o] = i0;
    for (let t = 0; t < nTaps; t++) data[o + 1 + t] = w[t] / sum;
  });
  return { nTaps, data };
}

/** Configure a WebGPU canvas context: storage writes straight into the canvas texture when the browser allows it
 *  (returns true), else a COPY_DST canvas (the caller renders into its own texture and copies). */
export async function configureCanvas(device: GPUDevice, ctx: GPUCanvasContext, format: GPUTextureFormat,
                                      canvasPath: 'auto' | 'copy' = 'auto'): Promise<boolean> {
  const T = GPUTextureUsage;
  let direct = false;
  if (canvasPath !== 'copy') {
    device.pushErrorScope('validation');
    let threw = false;
    try {
      ctx.configure({ device, format, alphaMode: 'opaque', usage: T.STORAGE_BINDING | T.COPY_SRC | T.RENDER_ATTACHMENT });
      ctx.getCurrentTexture();
    } catch { threw = true; }
    direct = !(await device.popErrorScope()) && !threw;
  }
  if (!direct) ctx.configure({ device, format, alphaMode: 'opaque', usage: T.COPY_DST | T.COPY_SRC | T.RENDER_ATTACHMENT });
  return direct;
}

export interface ColorInfo {
  /** chroma mode of the last frame: 'planar420' = imported as 4:2:0 planes (exact source chroma, left-sited);
   *  'rgb' = the browser converted it to RGB first (e.g. 10-bit HEVC or CPU-memory frames on macOS) */
  mode: 'planar420' | 'rgb' | 'none';
  /** transfer calibration of the last frame's colour class (see `classes` for all of them) */
  transfer: TransferName;
  how: 'default' | 'V-test' | 'Q-test' | 'forced';
  /** frames of the last frame's class */
  frames: number;
  matrix: string;
  accV: number[];
  accQ: number[];
  warnings: string[];
  /** true when the warp writes straight into the canvas texture (else: intermediate + copy) */
  directCanvas: boolean;
  canvasFormat: GPUTextureFormat;
  /** colour class of the last frame ('<pixel format>|<transfer tag>') and every class seen since the last resetColor() */
  colorClass: string;
  classes: { key: string; frames: number; transfer: TransferName; how: ColorInfo['how'] }[];
}

/** Colour class of a frame: frames of one class get the same import conversion from the browser (it follows the
 *  frame's own pixel format and transfer tag), so they share one transfer calibration. */
export function colorClassOf(frame: { format?: string | null; colorSpace?: { transfer?: string | null } | null }): string {
  return `${frame.format ?? '-'}|${frame.colorSpace?.transfer ?? '-'}`;
}

/** Default T_src (index into TRANSFERS) of a colour class until its own blind tests are decisive: the transfer the
 *  browser applies on import for that tag ('iec61966-2-1' = sRGB = no change; 'linear'; bt709 / smpte170m / untagged:
 *  Apple's gamma 1.961 on macOS VideoToolbox frames, sRGB elsewhere = the platform guess). */
export function defaultTransferFor(transferTag: string | null | undefined, mac: boolean): number {
  switch (transferTag) {
    case 'iec61966-2-1': return TRANSFERS.indexOf('srgb');
    case 'linear': return TRANSFERS.indexOf('linear');
    default: return mac ? TRANSFERS.indexOf('gamma1.961') : TRANSFERS.indexOf('srgb');
  }
}

/** Calibration state of one colour class (GPU ColorState + the defaults the CPU knows). */
interface ColorClass { key: string; tag: string | null; buf: GPUBuffer; def: number; frames: number }

const KERNEL_TAPS: Record<WarpKernel, number> = { lanczos3: 6, catmullrom: 4, bilinear: 2 };
const HOW = ['default', 'V-test', 'Q-test', 'forced'] as const;
const CALIB_X = 16, CALIB_Y = 9;          // 8x8-thread workgroups -> 128 x 72 sampled 2x2 blocks per frame
const PARAM_BYTES = 128;
const CSTATE_BYTES = 80;

function matrixCoeffs(m: string | null | undefined): [number, number] {
  switch (m) {
    case 'bt470bg': case 'smpte170m': return [0.299, 0.114];
    case 'bt2020-ncl': return [0.2627, 0.0593];
    default: return [0.2126, 0.0722];     // bt709 (and unknown / rgb: only used for planar frames)
  }
}

interface Pipes {
  bglConvert: GPUBindGroupLayout; bglWarp: GPUBindGroupLayout;
  bglSS: GPUBindGroupLayout; bglDH: GPUBindGroupLayout; bglDV: GPUBindGroupLayout;
  calib: GPUComputePipeline; reduce: GPUComputePipeline; convert: GPUComputePipeline;
  warp: GPUComputePipeline; coord: GPUComputePipeline;
  warpSS: GPUComputePipeline; downH: GPUComputePipeline; downV: GPUComputePipeline;
}
const pipeCache = new WeakMap<GPUDevice, Map<string, Promise<Pipes>>>();

async function buildPipes(device: GPUDevice, constants: Record<string, number>, outFormat: 'rgba8unorm' | 'bgra8unorm'): Promise<Pipes> {
  const code = WGSL.replace('texture_storage_2d<rgba8unorm, write>', `texture_storage_2d<${outFormat}, write>`);
  const module = device.createShaderModule({ label: 'sp.warp', code });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter(m => m.type === 'error');
  if (errs.length) throw new Error('warp.wgsl: ' + errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
  const C = GPUShaderStage.COMPUTE;
  const bglConvert = device.createBindGroupLayout({ label: 'sp.bgl.convert', entries: [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 1, visibility: C, externalTexture: {} },
    { binding: 2, visibility: C, buffer: { type: 'storage' } },
    { binding: 3, visibility: C, buffer: { type: 'storage' } },
    { binding: 4, visibility: C, storageTexture: { access: 'write-only', format: 'r32float' } },
    { binding: 5, visibility: C, storageTexture: { access: 'write-only', format: 'rg32float' } },
    { binding: 6, visibility: C, storageTexture: { access: 'write-only', format: 'r32uint' } },
  ] });
  const bglWarp = device.createBindGroupLayout({ label: 'sp.bgl.warp', entries: [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 7, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 8, visibility: C, texture: { sampleType: 'unfilterable-float' } },
    { binding: 9, visibility: C, texture: { sampleType: 'unfilterable-float' } },
    { binding: 10, visibility: C, texture: { sampleType: 'uint' } },
    { binding: 11, visibility: C, buffer: { type: 'storage' } },
    { binding: 12, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 13, visibility: C, storageTexture: { access: 'write-only', format: outFormat } },
    { binding: 14, visibility: C, sampler: { type: 'non-filtering' } },
  ] });
  const uf = { sampleType: 'unfilterable-float' } as const;
  const bglSS = device.createBindGroupLayout({ label: 'sp.bgl.warp_ss', entries: [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 7, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 8, visibility: C, texture: uf },
    { binding: 9, visibility: C, texture: uf },
    { binding: 10, visibility: C, texture: { sampleType: 'uint' } },
    { binding: 12, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 14, visibility: C, sampler: { type: 'non-filtering' } },
    { binding: 21, visibility: C, storageTexture: { access: 'write-only', format: 'r32float' } },
    { binding: 22, visibility: C, storageTexture: { access: 'write-only', format: 'rg32float' } },
  ] });
  const bglDH = device.createBindGroupLayout({ label: 'sp.bgl.down_h', entries: [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 23, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 15, visibility: C, texture: uf },
    { binding: 16, visibility: C, texture: uf },
    { binding: 17, visibility: C, storageTexture: { access: 'write-only', format: 'r32float' } },
    { binding: 18, visibility: C, storageTexture: { access: 'write-only', format: 'rg32float' } },
  ] });
  const bglDV = device.createBindGroupLayout({ label: 'sp.bgl.down_v', entries: [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 23, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 11, visibility: C, buffer: { type: 'storage' } },
    { binding: 13, visibility: C, storageTexture: { access: 'write-only', format: outFormat } },
    { binding: 19, visibility: C, texture: uf },
    { binding: 20, visibility: C, texture: uf },
  ] });
  const lc = device.createPipelineLayout({ bindGroupLayouts: [bglConvert] });
  const lw = device.createPipelineLayout({ bindGroupLayouts: [bglWarp] });
  const comp = (layout: GPUPipelineLayout, entryPoint: string) =>
    device.createComputePipelineAsync({ label: 'sp.' + entryPoint, layout, compute: { module, entryPoint, constants } });
  const L = (b: GPUBindGroupLayout) => device.createPipelineLayout({ bindGroupLayouts: [b] });
  const [calib, reduce, convert, warp, coord, warpSS, downH, downV] = await Promise.all([
    comp(lc, 'calib_blocks'), comp(lc, 'calib_reduce'), comp(lc, 'convert'), comp(lw, 'warp'), comp(lw, 'coord_map'),
    comp(L(bglSS), 'warp_ss'), comp(L(bglDH), 'down_h'), comp(L(bglDV), 'down_v')]);
  return { bglConvert, bglWarp, bglSS, bglDH, bglDV, calib, reduce, convert, warp, coord, warpSS, downH, downV };
}

function pipesFor(device: GPUDevice, constants: Record<string, number>, outFormat: 'rgba8unorm' | 'bgra8unorm'): Promise<Pipes> {
  let m = pipeCache.get(device);
  if (!m) { m = new Map(); pipeCache.set(device, m); }
  const key = JSON.stringify(constants) + outFormat;
  let p = m.get(key);
  if (!p) { p = buildPipes(device, constants, outFormat); m.set(key, p); p.catch(() => m!.delete(key)); }
  return p;
}

/** Camera-domain intermediates, shared by all Warpers of a device with the same source size. Safe because every
 *  frame's convert + warp go to the queue in ONE submit, so frames of different Warpers never interleave on them. */
interface Inter { refs: number; Y: GPUTexture; CH: GPUTexture; CF: GPUTexture;
  views: { Y: GPUTextureView; CH: GPUTextureView; CF: GPUTextureView };
  /** the frame the intermediates currently hold (and the Warper that converted it): sub-frame re-warps skip the
   *  conversion while it is still this one */
  last?: { frame: VideoFrame; warper: object } | null }
const interCache = new WeakMap<GPUDevice, Map<string, Inter>>();

function acquireInter(device: GPUDevice, w: number, h: number): Inter {
  let m = interCache.get(device);
  if (!m) { m = new Map(); interCache.set(device, m); }
  const key = `${w}x${h}`;
  let it = m.get(key);
  if (!it) {
    const T = GPUTextureUsage;
    const tex = (tw: number, th: number, format: GPUTextureFormat, label: string) =>
      device.createTexture({ label, size: [tw, th], format, usage: T.TEXTURE_BINDING | T.STORAGE_BINDING });
    const Y = tex(w, h, 'r32float', 'sp.Y');                                       // Y' per pixel
    const CH = tex(Math.ceil(w / 2), Math.ceil(h / 2), 'rg32float', 'sp.CbCr.block'); // CbCr per 2x2 block
    const CF = tex(w, h, 'r32uint', 'sp.CbCr.pixel');                               // CbCr per pixel (2 x f16)
    it = { refs: 0, Y, CH, CF, views: { Y: Y.createView(), CH: CH.createView(), CF: CF.createView() } };
    m.set(key, it);
  }
  it.refs++;
  return it;
}

function releaseInter(device: GPUDevice, it: Inter) {
  if (--it.refs > 0) return;
  const m = interCache.get(device);
  if (m) for (const [k, v] of m) if (v === it) m.delete(k);
  it.Y.destroy(); it.CH.destroy(); it.CF.destroy();
}

interface SSState {
  iW: number; iH: number; sx: number; sy: number; nTaps: number;
  Y: GPUTexture; C: GPUTexture; hY: GPUTexture; hC: GPUTexture; table: GPUBuffer;
  /** warp_ss bind group per colour class (it reads that class's chroma mode) */
  bgWarp: Map<ColorClass, GPUBindGroup>; bgH: GPUBindGroup; hYv: GPUTextureView; hCv: GPUTextureView;
  Yv: GPUTextureView; Cv: GPUTextureView;
}

export class Warper {
  readonly device: GPUDevice;
  readonly canvas: OffscreenCanvas;
  readonly kernel: WarpKernel;
  private plan: Plan;
  private readonly iters: number;
  private readonly pipes: Pipes;
  private readonly ctx: GPUCanvasContext;
  private readonly directCanvas: boolean;
  private readonly outFormat: 'rgba8unorm' | 'bgra8unorm';
  private readonly forceTransfer: number;
  private readonly mac: boolean;
  private readonly params: GPUBuffer;
  private readonly rows: GPUBuffer;
  /** colour classes seen (see colorClassOf) and the class of the frame being / last converted */
  private readonly classes = new Map<string, ColorClass>();
  private cls: ColorClass;
  private readonly partials: GPUBuffer;
  private readonly dummyBuf: GPUBuffer;
  private readonly sampler: GPUSampler;
  private readonly dummyOut: GPUTexture;
  private readonly forceMode: number;
  private readonly inter: Inter;
  private readonly views: Inter['views'];
  private readonly paramData = new ArrayBuffer(PARAM_BYTES);
  private outTex: GPUTexture | null = null;       // intermediate when the canvas texture cannot be a storage target
  private matrix = 'bt709';
  private frames = 0;
  private warnings = new Set<string>();
  private destroyed = false;
  private qset: GPUQuerySet | null = null;
  private qbuf: GPUBuffer | null = null;
  private readonly antialias: WarperOptions['antialias'];
  private scalingInfo: WarpScaling;
  private ss: SSState | null = null;              // supersampled-path textures (lazy, sized by scalingInfo)

  private constructor(device: GPUDevice, plan: Plan, opts: WarperOptions, pipes: Pipes, canvas: OffscreenCanvas,
                      ctx: GPUCanvasContext, directCanvas: boolean, outFormat: 'rgba8unorm' | 'bgra8unorm') {
    this.device = device;
    this.plan = plan;
    this.kernel = opts.kernel;
    this.iters = opts.iters ?? 3;
    this.pipes = pipes;
    this.canvas = canvas;
    this.ctx = ctx;
    this.directCanvas = directCanvas;
    this.outFormat = outFormat;
    const t = opts.transfer ?? 'auto';
    this.forceTransfer = t === 'auto' ? -1 : TRANSFERS.indexOf(t);
    this.forceMode = opts.chroma === 'planar' ? 1 : opts.chroma === 'rgb' ? 2 : 0;
    // Chrome on macOS tags VideoToolbox frames with Apple's BT.709 gamma (1.961); elsewhere BT.709 ~ sRGB (no-op).
    this.mac = typeof navigator !== 'undefined' && /Mac OS X|Macintosh/.test(navigator.userAgent);
    const U = GPUBufferUsage;
    this.params = device.createBuffer({ label: 'sp.params', size: PARAM_BYTES, usage: U.UNIFORM | U.COPY_DST });
    this.rows = device.createBuffer({ label: 'sp.rows', size: Math.max(16, plan.nRows * 36), usage: U.STORAGE | U.COPY_DST });
    this.cls = this.colorClass('-|-', null);
    this.partials = device.createBuffer({ label: 'sp.partials', size: CALIB_X * CALIB_Y * 12 * 4, usage: U.STORAGE });
    this.dummyBuf = device.createBuffer({ label: 'sp.dummy', size: 16, usage: U.STORAGE });
    this.sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    const T = GPUTextureUsage;
    const tex = (w: number, h: number, format: GPUTextureFormat, label: string) =>
      device.createTexture({ label, size: [w, h], format, usage: T.TEXTURE_BINDING | T.STORAGE_BINDING });
    this.dummyOut = tex(1, 1, outFormat, 'sp.dummyOut');
    this.inter = acquireInter(device, plan.srcW, plan.srcH);
    this.views = this.inter.views;
    this.antialias = opts.antialias ?? 'auto';
    this.scalingInfo = warpScaling(plan, this.antialias, device.limits.maxTextureDimension2D);
    if (opts.profile && device.features.has('timestamp-query')) {
      this.qset = device.createQuerySet({ type: 'timestamp', count: 4 });
      this.qbuf = device.createBuffer({ size: 32, usage: U.QUERY_RESOLVE | U.COPY_SRC });
    }
    this.resetColor();
  }

  /** debug (opts.profile): GPU ms of the last frame's convert (+calibration) and warp passes */
  async profile(): Promise<{ convertMs: number; warpMs: number } | null> {
    if (!this.qbuf) return null;
    const rb = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.qbuf, 0, rb, 0, 32);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    return { convertMs: Number(t[1] - t[0]) / 1e6, warpMs: Number(t[3] - t[2]) / 1e6 };
  }

  static async create(device: GPUDevice, plan: Plan, opts: WarperOptions): Promise<Warper> {
    if (plan.nRows < 2) throw new Error('plan.nRows must be >= 2');
    if (plan.rowMats.length < plan.framePts.length * plan.nRows * 9) throw new Error('plan.rowMats too short');
    const taps = KERNEL_TAPS[opts.kernel];
    if (!taps) throw new Error(`unknown kernel ${opts.kernel}`);
    // bgra8unorm is the canvas format Chrome/macOS composites without a copy; storage writes to it need the feature.
    const outFormat = device.features.has('bgra8unorm-storage') ? 'bgra8unorm' : 'rgba8unorm';
    const pipes = await pipesFor(device, { TAPS: taps, GATHER: opts.gather === false ? 0 : 1 }, outFormat);
    if (!(plan.outW >= 1 && plan.outH >= 1)) throw new Error(`bad output size ${plan.outW}x${plan.outH}`);
    const canvas = new OffscreenCanvas(plan.outW, plan.outH);
    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('WebGPU canvas context unavailable');
    // Prefer writing the warp straight into the canvas texture (storage usage on the canvas).
    const direct = await configureCanvas(device, ctx, outFormat, opts.canvasPath ?? 'auto');
    return new Warper(device, plan, opts, pipes, canvas, ctx, direct, outFormat);
  }

  /** Output size (== plan.outW x plan.outH). */
  get width(): number { return this.plan.outW; }
  get height(): number { return this.plan.outH; }
  /** Texture format of the canvas / warpTo() targets. */
  get format(): 'rgba8unorm' | 'bgra8unorm' { return this.outFormat; }
  /** Single-pass or supersampled + Lanczos-downscaled rendering (see WarperOptions.antialias). */
  get scaling(): WarpScaling { return { ...this.scalingInfo }; }

  /** Swap in a rebuilt plan (same source/output size, same nRows) without recreating GPU state. A different outFx may
   *  change the supersampling grid (textures are re-allocated lazily). */
  setPlan(plan: Plan): void {
    const q = this.plan;
    if (plan.srcW !== q.srcW || plan.srcH !== q.srcH || plan.outW !== q.outW || plan.outH !== q.outH || plan.nRows !== q.nRows)
      throw new Error('setPlan: size / nRows change needs a new Warper');
    this.plan = plan;
    this.scalingInfo = warpScaling(plan, this.antialias, this.device.limits.maxTextureDimension2D);
    if (this.scalingInfo.mode === 'direct') this.destroySS();
  }

  /** Forget the accumulated colour-transfer evidence of every colour class (e.g. a different decoder / clip). */
  resetColor(): void {
    for (const c of this.classes.values()) {
      this.device.queue.writeBuffer(c.buf, 0, new Uint32Array(CSTATE_BYTES / 4));
      c.frames = 0;
    }
  }

  /** Warp decoded frame `frame` with plan record k. Returns a new VideoFrame (caller closes both) with the source's
   *  timestamp/duration, or `timing` (retimed exports: the output schedule's pts). */
  warp(frame: VideoFrame, k: number, timing?: FrameTiming): VideoFrame {
    this.encodeFrame(frame, k, null);
    const ts = timing ?? { timestamp: frame.timestamp, duration: frame.duration ?? undefined };
    return new VideoFrame(this.canvas, { timestamp: ts.timestamp, ...(ts.duration != null ? { duration: ts.duration } : {}) });
  }

  /** Warp into `target` (a texture from createTarget(): this.format, outW x outH, STORAGE_BINDING) instead of the
   *  canvas — e.g. a ring of warped frames for the Blender (no VideoFrame snapshot per source frame). */
  warpTo(frame: VideoFrame, k: number, target: GPUTexture): void {
    const p = this.plan;
    if (target.width !== p.outW || target.height !== p.outH || target.format !== this.outFormat)
      throw new Error(`warpTo: target must be ${p.outW}x${p.outH} ${this.outFormat}`);
    this.encodeFrame(frame, k, null, target);
  }

  /** warpTo() with explicit row matrices (nRows*9 floats, the rows record k would use — e.g. rotated to a sub-frame
   *  virtual orientation: M_row · R(virt_k)ᵀR(virt(t))) and output focal. Repeated calls for the SAME frame skip the
   *  YCbCr conversion + colour calibration (the intermediates still hold it): a synthetic shutter re-warps one decoded
   *  frame many times. */
  warpToWith(frame: VideoFrame, k: number, target: GPUTexture, rows: Float32Array, outFx: number): void {
    const p = this.plan;
    if (target.width !== p.outW || target.height !== p.outH || target.format !== this.outFormat)
      throw new Error(`warpToWith: target must be ${p.outW}x${p.outH} ${this.outFormat}`);
    if (rows.length !== p.nRows * 9) throw new Error(`warpToWith: rows must be ${p.nRows * 9} floats`);
    if (!(outFx > 0)) throw new Error(`warpToWith: bad focal ${outFx}`);
    this.encodeFrame(frame, k, null, target, { rows, outFx });
  }

  /** A texture warpTo() can render into (and the Blender can read): outW x outH, this.format. */
  createTarget(label = 'sp.target'): GPUTexture {
    const T = GPUTextureUsage;
    return this.device.createTexture({ label, size: [this.plan.outW, this.plan.outH], format: this.outFormat,
      usage: T.STORAGE_BINDING | T.TEXTURE_BINDING | T.COPY_SRC | T.COPY_DST });
  }

  /** Warp into `this.canvas` only (no VideoFrame) — e.g. a live preview that draws the canvas itself. */
  render(frame: VideoFrame, k: number): void {
    this.encodeFrame(frame, k, null);
  }

  /** Test/debug: warped luma Y' (nominal 0..1, float32, outW*outH) of the frame with record k, before 8-bit output. */
  async readbackLuma(frame: VideoFrame, k: number): Promise<Float32Array> {
    const { outW, outH } = this.plan;
    const n = outW * outH * 4;
    const out = this.device.createBuffer({ size: n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
      this.encodeFrame(frame, k, out);
      return await this.readBuffer(out, n);
    } finally { out.destroy(); }
  }

  /** Test/debug: source-coordinate map of record k on a scaled output grid (Wo = round(outW*outScale)):
   *  Float32Array (Ho*Wo*3) of (S.x, S.y, valid) in full-res source luma pixels — same as warp.metal sp_coord_map. */
  async coordMap(k: number, outScale = 1): Promise<{ width: number; height: number; data: Float32Array }> {
    const { outW, outH } = this.plan;
    const Wo = Math.round(outW * outScale), Ho = Math.round(outH * outScale);
    this.writeParams(k, { dstW: Wo, dstH: Ho, outSx: Wo / outW, outSy: Ho / outH, dbg: 0 });
    this.writeRows(k);
    const n = Wo * Ho * 12;
    const out = this.device.createBuffer({ size: n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
      const bg = this.device.createBindGroup({ layout: this.pipes.bglWarp, entries: this.warpEntries(this.dummyOut, out) });
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipes.coord); pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(Wo / 8), Math.ceil(Ho / 8));
      pass.end();
      this.device.queue.submit([enc.finish()]);
      return { width: Wo, height: Ho, data: await this.readBuffer(out, n) };
    } finally { out.destroy(); }
  }

  /** What the colour calibration concluded so far (async readback of the GPU state): the last frame's colour class,
   *  plus a summary of every class. */
  async colorInfo(): Promise<ColorInfo> {
    const read = async (c: ColorClass) => {
      const buf = (await this.readBuffer(c.buf, CSTATE_BYTES)).buffer as ArrayBuffer;
      return { f: new Float32Array(buf), u: new Uint32Array(buf) };
    };
    const classes: ColorInfo['classes'] = [];
    for (const c of this.classes.values()) {
      if (!c.frames) continue;
      const { u } = await read(c);
      classes.push({ key: c.key, frames: u[16], transfer: TRANSFERS[u[17]] ?? 'srgb', how: HOW[u[18]] ?? 'default' });
    }
    const { f, u } = await read(this.cls);
    return {
      mode: this.frames === 0 ? 'none' : u[19] === 1 ? 'planar420' : 'rgb',
      transfer: TRANSFERS[u[17]] ?? 'srgb', how: HOW[u[18]] ?? 'default', frames: u[16], matrix: this.matrix,
      accV: Array.from(f.subarray(0, TRANSFERS.length)), accQ: Array.from(f.subarray(8, 8 + TRANSFERS.length)),
      warnings: [...this.warnings], directCanvas: this.directCanvas, canvasFormat: this.outFormat,
      colorClass: this.cls.key, classes,
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const b of [this.params, this.rows, this.partials, this.dummyBuf, this.qbuf]) b?.destroy();
    for (const c of this.classes.values()) c.buf.destroy();
    this.classes.clear();
    this.qset?.destroy();
    for (const t of [this.outTex, this.dummyOut]) t?.destroy();
    this.destroySS();
    releaseInter(this.device, this.inter);
    this.outTex = null;
    try { this.ctx.unconfigure(); } catch { /* already gone */ }
  }

  // ------------------------------------------------------------------------------------------ internals
  /** The calibration state of colour class `key` (created on first use, zeroed = no evidence yet). */
  private colorClass(key: string, tag: string | null): ColorClass {
    let c = this.classes.get(key);
    if (!c) {
      const buf = this.device.createBuffer({ label: `sp.cstate ${key}`, size: CSTATE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(buf, 0, new Uint32Array(CSTATE_BYTES / 4));
      c = { key, tag, buf, def: defaultTransferFor(tag, this.mac), frames: 0 };
      this.classes.set(key, c);
    }
    return c;
  }

  private async readBuffer(src: GPUBuffer, n: number): Promise<Float32Array> {
    const rb = this.device.createBuffer({ size: n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, rb, 0, n);
      this.device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const res = new Float32Array(rb.getMappedRange().slice(0));
      rb.unmap();
      return res;
    } finally { rb.destroy(); }
  }

  private destroySS() {
    if (!this.ss) return;
    for (const t of [this.ss.Y, this.ss.C, this.ss.hY, this.ss.hC]) t.destroy();
    this.ss.table.destroy();
    this.ss = null;
  }

  /** Supersampled-path textures + the bind groups that do not change per frame, for the current scaling. */
  private ssState(): SSState {
    const sc = this.scalingInfo, p = this.plan;
    if (this.ss && this.ss.iW === sc.interW && this.ss.iH === sc.interH && this.ss.sx === sc.sx && this.ss.sy === sc.sy) return this.ss;
    this.destroySS();
    const T = GPUTextureUsage;
    const tex = (w: number, h: number, format: GPUTextureFormat, label: string) =>
      this.device.createTexture({ label, size: [w, h], format, usage: T.TEXTURE_BINDING | T.STORAGE_BINDING });
    const iW = sc.interW, iH = sc.interH;
    const Y = tex(iW, iH, 'r32float', 'sp.ss.Y');
    const C = tex(Math.ceil(iW / 2), Math.ceil(iH / 2), 'rg32float', 'sp.ss.CbCr');
    const hY = tex(p.outW, iH, 'r32float', 'sp.ss.hY');
    const hC = tex(Math.ceil(p.outW / 2), Math.ceil(iH / 2), 'rg32float', 'sp.ss.hCbCr');
    const tab = downscaleTable(p.outW, p.outH, sc.sx, sc.sy);
    const table = this.device.createBuffer({ label: 'sp.ss.taps', size: tab.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(table, 0, tab.data);
    const P = this.pipes;
    const bgH = this.device.createBindGroup({ layout: P.bglDH, entries: [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 23, resource: { buffer: table } },
      { binding: 15, resource: Y.createView() },
      { binding: 16, resource: C.createView() },
      { binding: 17, resource: hY.createView() },
      { binding: 18, resource: hC.createView() },
    ] });
    this.ss = { iW, iH, sx: sc.sx, sy: sc.sy, nTaps: tab.nTaps, Y, C, hY, hC, table, bgWarp: new Map(), bgH,
      hYv: hY.createView(), hCv: hC.createView(), Yv: Y.createView(), Cv: C.createView() };
    return this.ss;
  }

  /** warp_ss bind group for colour class c (cached per class). */
  private ssWarpGroup(ss: SSState, c: ColorClass): GPUBindGroup {
    let bg = ss.bgWarp.get(c);
    if (!bg) {
      const v = this.views;
      bg = this.device.createBindGroup({ layout: this.pipes.bglSS, entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 7, resource: { buffer: this.rows } },
        { binding: 8, resource: v.Y },
        { binding: 9, resource: v.CH },
        { binding: 10, resource: v.CF },
        { binding: 12, resource: { buffer: c.buf } },
        { binding: 14, resource: this.sampler },
        { binding: 21, resource: ss.Yv },
        { binding: 22, resource: ss.Cv },
      ] });
      ss.bgWarp.set(c, bg);
    }
    return bg;
  }

  private writeParams(k: number, o: { dstW: number; dstH: number; outSx: number; outSy: number; dbg: number; nTaps?: number; outFx?: number }) {
    const p = this.plan, L = p.lens;
    const f = new Float32Array(this.paramData), u = new Uint32Array(this.paramData), i = new Int32Array(this.paramData);
    const [kr, kb] = matrixCoeffs(this.matrix);
    f[0] = o.outFx ?? p.outFx[k]; f[1] = (p.outW - 1) / 2; f[2] = (p.outH - 1) / 2; u[3] = L.model === 'kb4' ? 1 : 0;
    f[4] = L.fx; f[5] = L.fy; f[6] = L.cx; f[7] = L.cy;
    f[8] = L.k[0]; f[9] = L.k[1]; f[10] = L.k[2]; f[11] = L.k[3];
    f[12] = p.srcW; f[13] = p.srcH; u[14] = p.nRows; u[15] = this.iters;
    f[16] = o.outSx; f[17] = o.outSy; u[18] = o.dstW; u[19] = o.dstH;
    f[20] = kr; f[21] = kb; u[22] = this.cls.def; i[23] = this.forceTransfer;
    u[24] = CALIB_X; u[25] = CALIB_Y; u[26] = this.forceMode; u[27] = o.dbg;
    u[28] = p.outW; u[29] = p.outH; u[30] = o.nTaps ?? 0; u[31] = 0;
    this.device.queue.writeBuffer(this.params, 0, this.paramData);
  }

  private writeRows(k: number, rows?: Float32Array) {
    const p = this.plan;
    if (!(k >= 0 && k < p.framePts.length)) throw new RangeError(`plan record ${k} out of range (0..${p.framePts.length - 1})`);
    const n = p.nRows * 9;
    if (rows) this.device.queue.writeBuffer(this.rows, 0, rows as Float32Array<ArrayBuffer>, 0, n);
    else this.device.queue.writeBuffer(this.rows, 0, p.rowMats as Float32Array<ArrayBuffer>, k * n, n);
  }

  private warpEntries(out: GPUTexture, dbg: GPUBuffer | null): GPUBindGroupEntry[] {
    const v = this.views;
    return [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 7, resource: { buffer: this.rows } },
      { binding: 8, resource: v.Y },
      { binding: 9, resource: v.CH },
      { binding: 10, resource: v.CF },
      { binding: 11, resource: { buffer: dbg ?? this.dummyBuf } },
      { binding: 12, resource: { buffer: this.cls.buf } },
      { binding: 13, resource: out.createView() },
      { binding: 14, resource: this.sampler },
    ];
  }

  /** Encode + submit convert/calibrate + warp for one frame, into the canvas, `into`, or (debug) float luma into
   *  `lumaOut`. Supersampled path: warp_ss (intermediate grid) -> down_h -> down_v (-> output), same pass. */
  private encodeFrame(frame: VideoFrame, k: number, lumaOut: GPUBuffer | null, into: GPUTexture | null = null,
                      ov?: { rows: Float32Array; outFx: number }) {
    if (this.destroyed) throw new Error('Warper destroyed');
    const p = this.plan;
    if (frame.displayWidth !== p.srcW || frame.displayHeight !== p.srcH)
      throw new Error(`frame ${frame.displayWidth}x${frame.displayHeight} != plan source ${p.srcW}x${p.srcH}`);
    const cs = frame.colorSpace;
    this.matrix = cs?.matrix ?? 'bt709';
    const tr = (cs?.transfer ?? '') as string;
    if (tr === 'pq' || tr === 'hlg') this.warnings.add(`HDR transfer '${tr}' is tone-mapped by the browser; colours are not preserved`);
    // this frame's colour class: its own transfer calibration (the browser's conversion follows the frame's own tag)
    this.cls = this.colorClass(colorClassOf(frame), cs?.transfer ?? null);
    // sub-frame re-warp of the frame the shared intermediates already hold: no second conversion / calibration
    const reuse = !!ov && this.inter.last?.frame === frame && this.inter.last.warper === this;
    if (!reuse) { this.frames++; this.cls.frames++; }
    const sc = this.scalingInfo;
    const ss = sc.mode === 'supersample' ? this.ssState() : null;
    if (ss) this.writeParams(k, { dstW: ss.iW, dstH: ss.iH, outSx: sc.sx, outSy: sc.sy, dbg: lumaOut ? 1 : 0, nTaps: ss.nTaps, outFx: ov?.outFx });
    else this.writeParams(k, { dstW: p.outW, dstH: p.outH, outSx: 1, outSy: 1, dbg: lumaOut ? 1 : 0, outFx: ov?.outFx });
    this.writeRows(k, ov?.rows);
    const P = this.pipes;
    const cbg = reuse ? null : this.device.createBindGroup({ layout: P.bglConvert, entries: [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: this.device.importExternalTexture({ source: frame, colorSpace: 'srgb' }) },
      { binding: 2, resource: { buffer: this.cls.buf } },
      { binding: 3, resource: { buffer: this.partials } },
      { binding: 4, resource: this.views.Y },
      { binding: 5, resource: this.views.CH },
      { binding: 6, resource: this.views.CF },
    ] });
    let target: GPUTexture;
    let canvasTex: GPUTexture | null = null;
    if (lumaOut) target = this.dummyOut;
    else if (into) target = into;
    else if (this.directCanvas) target = canvasTex = this.ctx.getCurrentTexture();
    else {
      canvasTex = this.ctx.getCurrentTexture();
      if (!this.outTex) this.outTex = this.device.createTexture({ label: 'sp.out', size: [p.outW, p.outH], format: this.outFormat,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
      target = this.outTex;
    }
    const wbg = ss ? null : this.device.createBindGroup({ layout: P.bglWarp, entries: this.warpEntries(target, lumaOut) });
    const enc = this.device.createCommandEncoder({ label: 'sp.frame' });
    const ts = (i: number): GPUComputePassDescriptor['timestampWrites'] =>
      this.qset ? { querySet: this.qset, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 } : undefined;
    if (cbg) {
      const cp = enc.beginComputePass({ label: 'sp.convert', timestampWrites: ts(0) });
      cp.setBindGroup(0, cbg);
      cp.setPipeline(P.calib); cp.dispatchWorkgroups(CALIB_X, CALIB_Y);
      cp.setPipeline(P.reduce); cp.dispatchWorkgroups(1);
      cp.setPipeline(P.convert);
      cp.dispatchWorkgroups(Math.ceil(Math.ceil(p.srcW / 2) / 8), Math.ceil(Math.ceil(p.srcH / 2) / 8));
      cp.end();
    }
    this.inter.last = { frame, warper: this };
    const wp = enc.beginComputePass({ label: 'sp.warp', timestampWrites: ts(2) });
    if (!ss) {
      wp.setBindGroup(0, wbg!);
      wp.setPipeline(P.warp);
      wp.dispatchWorkgroups(Math.ceil(p.outW / 16), Math.ceil(p.outH / 16));
    } else {
      const bgV = this.device.createBindGroup({ layout: P.bglDV, entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 23, resource: { buffer: ss.table } },
        { binding: 11, resource: { buffer: lumaOut ?? this.dummyBuf } },
        { binding: 13, resource: target.createView() },
        { binding: 19, resource: ss.hYv },
        { binding: 20, resource: ss.hCv },
      ] });
      wp.setBindGroup(0, this.ssWarpGroup(ss, this.cls));
      wp.setPipeline(P.warpSS);
      wp.dispatchWorkgroups(Math.ceil(ss.iW / 16), Math.ceil(ss.iH / 16));
      wp.setBindGroup(0, ss.bgH);
      wp.setPipeline(P.downH);
      wp.dispatchWorkgroups(Math.ceil(p.outW / 8), Math.ceil(ss.iH / 8));
      wp.setBindGroup(0, bgV);
      wp.setPipeline(P.downV);
      wp.dispatchWorkgroups(Math.ceil(p.outW / 16), Math.ceil(p.outH / 16));
    }
    wp.end();
    if (this.qset) enc.resolveQuerySet(this.qset, 0, 4, this.qbuf!, 0);
    if (canvasTex && target !== canvasTex) enc.copyTextureToTexture({ texture: target }, { texture: canvasTex }, [p.outW, p.outH]);
    this.device.queue.submit([enc.finish()]);
  }
}
