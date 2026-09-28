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
 * size, + the canvas (4 B/output pixel, x2-3 in flight). Tests: test/gpu.run.mjs (selftest | golden | bench).
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
}

export interface ColorInfo {
  /** chroma mode of the last frame: 'planar420' = imported as 4:2:0 planes (exact source chroma, left-sited);
   *  'rgb' = the browser converted it to RGB first (e.g. 10-bit HEVC or CPU-memory frames on macOS) */
  mode: 'planar420' | 'rgb' | 'none';
  transfer: TransferName;
  how: 'default' | 'V-test' | 'Q-test' | 'forced';
  frames: number;
  matrix: string;
  accV: number[];
  accQ: number[];
  warnings: string[];
  /** true when the warp writes straight into the canvas texture (else: intermediate + copy) */
  directCanvas: boolean;
  canvasFormat: GPUTextureFormat;
}

const KERNEL_TAPS: Record<WarpKernel, number> = { lanczos3: 6, catmullrom: 4, bilinear: 2 };
const HOW = ['default', 'V-test', 'Q-test', 'forced'] as const;
const CALIB_X = 16, CALIB_Y = 9;          // 8x8-thread workgroups -> 128 x 72 sampled 2x2 blocks per frame
const PARAM_BYTES = 112;
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
  calib: GPUComputePipeline; reduce: GPUComputePipeline; convert: GPUComputePipeline;
  warp: GPUComputePipeline; coord: GPUComputePipeline;
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
  const lc = device.createPipelineLayout({ bindGroupLayouts: [bglConvert] });
  const lw = device.createPipelineLayout({ bindGroupLayouts: [bglWarp] });
  const comp = (layout: GPUPipelineLayout, entryPoint: string) =>
    device.createComputePipelineAsync({ label: 'sp.' + entryPoint, layout, compute: { module, entryPoint, constants } });
  const [calib, reduce, convert, warp, coord] = await Promise.all([
    comp(lc, 'calib_blocks'), comp(lc, 'calib_reduce'), comp(lc, 'convert'), comp(lw, 'warp'), comp(lw, 'coord_map')]);
  return { bglConvert, bglWarp, calib, reduce, convert, warp, coord };
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
  views: { Y: GPUTextureView; CH: GPUTextureView; CF: GPUTextureView } }
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
  private readonly defTransfer: number;
  private readonly params: GPUBuffer;
  private readonly rows: GPUBuffer;
  private readonly cstate: GPUBuffer;
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
    this.defTransfer = typeof navigator !== 'undefined' && /Mac OS X|Macintosh/.test(navigator.userAgent) ? 1 : 0;
    const U = GPUBufferUsage;
    this.params = device.createBuffer({ label: 'sp.params', size: PARAM_BYTES, usage: U.UNIFORM | U.COPY_DST });
    this.rows = device.createBuffer({ label: 'sp.rows', size: Math.max(16, plan.nRows * 36), usage: U.STORAGE | U.COPY_DST });
    this.cstate = device.createBuffer({ label: 'sp.cstate', size: CSTATE_BYTES, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this.partials = device.createBuffer({ label: 'sp.partials', size: CALIB_X * CALIB_Y * 12 * 4, usage: U.STORAGE });
    this.dummyBuf = device.createBuffer({ label: 'sp.dummy', size: 16, usage: U.STORAGE });
    this.sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    const T = GPUTextureUsage;
    const tex = (w: number, h: number, format: GPUTextureFormat, label: string) =>
      device.createTexture({ label, size: [w, h], format, usage: T.TEXTURE_BINDING | T.STORAGE_BINDING });
    this.dummyOut = tex(1, 1, outFormat, 'sp.dummyOut');
    this.inter = acquireInter(device, plan.srcW, plan.srcH);
    this.views = this.inter.views;
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
    const canvas = new OffscreenCanvas(plan.outW, plan.outH);
    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('WebGPU canvas context unavailable');
    // Prefer writing the warp straight into the canvas texture (storage usage on the canvas).
    const T = GPUTextureUsage;
    let direct = false;
    if (opts.canvasPath !== 'copy') {
      device.pushErrorScope('validation');
      let threw = false;
      try {
        ctx.configure({ device, format: outFormat, alphaMode: 'opaque', usage: T.STORAGE_BINDING | T.COPY_SRC | T.RENDER_ATTACHMENT });
        ctx.getCurrentTexture();
      } catch { threw = true; }
      direct = !(await device.popErrorScope()) && !threw;
    }
    if (!direct) ctx.configure({ device, format: outFormat, alphaMode: 'opaque', usage: T.COPY_DST | T.COPY_SRC | T.RENDER_ATTACHMENT });
    return new Warper(device, plan, opts, pipes, canvas, ctx, direct, outFormat);
  }

  /** Swap in a rebuilt plan (same source/output size, same nRows) without recreating GPU state. */
  setPlan(plan: Plan): void {
    const q = this.plan;
    if (plan.srcW !== q.srcW || plan.srcH !== q.srcH || plan.outW !== q.outW || plan.outH !== q.outH || plan.nRows !== q.nRows)
      throw new Error('setPlan: size / nRows change needs a new Warper');
    this.plan = plan;
  }

  /** Forget the accumulated colour-transfer evidence (e.g. a different decoder / clip). */
  resetColor(): void {
    this.device.queue.writeBuffer(this.cstate, 0, new Uint32Array(CSTATE_BYTES / 4));
  }

  /** Warp decoded frame `frame` with plan record k. Returns a new VideoFrame (caller closes both). */
  warp(frame: VideoFrame, k: number): VideoFrame {
    this.encodeFrame(frame, k, null);
    return new VideoFrame(this.canvas, { timestamp: frame.timestamp, ...(frame.duration != null ? { duration: frame.duration } : {}) });
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

  /** What the colour calibration concluded so far (async readback of the GPU state). */
  async colorInfo(): Promise<ColorInfo> {
    const buf = (await this.readBuffer(this.cstate, CSTATE_BYTES)).buffer as ArrayBuffer;
    const f = new Float32Array(buf), u = new Uint32Array(buf);
    return {
      mode: this.frames === 0 ? 'none' : u[19] === 1 ? 'planar420' : 'rgb',
      transfer: TRANSFERS[u[17]] ?? 'srgb', how: HOW[u[18]] ?? 'default', frames: u[16], matrix: this.matrix,
      accV: Array.from(f.subarray(0, TRANSFERS.length)), accQ: Array.from(f.subarray(8, 8 + TRANSFERS.length)),
      warnings: [...this.warnings], directCanvas: this.directCanvas, canvasFormat: this.outFormat,
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const b of [this.params, this.rows, this.cstate, this.partials, this.dummyBuf, this.qbuf]) b?.destroy();
    this.qset?.destroy();
    for (const t of [this.outTex, this.dummyOut]) t?.destroy();
    releaseInter(this.device, this.inter);
    this.outTex = null;
    try { this.ctx.unconfigure(); } catch { /* already gone */ }
  }

  // ------------------------------------------------------------------------------------------ internals
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

  private writeParams(k: number, o: { dstW: number; dstH: number; outSx: number; outSy: number; dbg: number }) {
    const p = this.plan, L = p.lens;
    const f = new Float32Array(this.paramData), u = new Uint32Array(this.paramData), i = new Int32Array(this.paramData);
    const [kr, kb] = matrixCoeffs(this.matrix);
    f[0] = p.outFx[k]; f[1] = (p.outW - 1) / 2; f[2] = (p.outH - 1) / 2; u[3] = L.model === 'kb4' ? 1 : 0;
    f[4] = L.fx; f[5] = L.fy; f[6] = L.cx; f[7] = L.cy;
    f[8] = L.k[0]; f[9] = L.k[1]; f[10] = L.k[2]; f[11] = L.k[3];
    f[12] = p.srcW; f[13] = p.srcH; u[14] = p.nRows; u[15] = this.iters;
    f[16] = o.outSx; f[17] = o.outSy; u[18] = o.dstW; u[19] = o.dstH;
    f[20] = kr; f[21] = kb; u[22] = this.defTransfer; i[23] = this.forceTransfer;
    u[24] = CALIB_X; u[25] = CALIB_Y; u[26] = this.forceMode; u[27] = o.dbg;
    this.device.queue.writeBuffer(this.params, 0, this.paramData);
  }

  private writeRows(k: number) {
    const p = this.plan;
    if (!(k >= 0 && k < p.framePts.length)) throw new RangeError(`plan record ${k} out of range (0..${p.framePts.length - 1})`);
    const n = p.nRows * 9;
    this.device.queue.writeBuffer(this.rows, 0, p.rowMats as Float32Array<ArrayBuffer>, k * n, n);
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
      { binding: 12, resource: { buffer: this.cstate } },
      { binding: 13, resource: out.createView() },
      { binding: 14, resource: this.sampler },
    ];
  }

  /** Encode + submit convert/calibrate + warp for one frame, into the canvas or (debug) float luma into `lumaOut`. */
  private encodeFrame(frame: VideoFrame, k: number, lumaOut: GPUBuffer | null) {
    if (this.destroyed) throw new Error('Warper destroyed');
    const p = this.plan;
    if (frame.displayWidth !== p.srcW || frame.displayHeight !== p.srcH)
      throw new Error(`frame ${frame.displayWidth}x${frame.displayHeight} != plan source ${p.srcW}x${p.srcH}`);
    const cs = frame.colorSpace;
    this.matrix = cs?.matrix ?? 'bt709';
    const tr = (cs?.transfer ?? '') as string;
    if (tr === 'pq' || tr === 'hlg') this.warnings.add(`HDR transfer '${tr}' is tone-mapped by the browser; colours are not preserved`);
    this.frames++;
    this.writeParams(k, { dstW: p.outW, dstH: p.outH, outSx: 1, outSy: 1, dbg: lumaOut ? 1 : 0 });
    this.writeRows(k);
    const P = this.pipes;
    const ext = this.device.importExternalTexture({ source: frame, colorSpace: 'srgb' });
    const cbg = this.device.createBindGroup({ layout: P.bglConvert, entries: [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: ext },
      { binding: 2, resource: { buffer: this.cstate } },
      { binding: 3, resource: { buffer: this.partials } },
      { binding: 4, resource: this.views.Y },
      { binding: 5, resource: this.views.CH },
      { binding: 6, resource: this.views.CF },
    ] });
    let target: GPUTexture;
    let canvasTex: GPUTexture | null = null;
    if (lumaOut) target = this.dummyOut;
    else if (this.directCanvas) target = canvasTex = this.ctx.getCurrentTexture();
    else {
      canvasTex = this.ctx.getCurrentTexture();
      if (!this.outTex) this.outTex = this.device.createTexture({ label: 'sp.out', size: [p.outW, p.outH], format: this.outFormat,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
      target = this.outTex;
    }
    const wbg = this.device.createBindGroup({ layout: P.bglWarp, entries: this.warpEntries(target, lumaOut) });
    const enc = this.device.createCommandEncoder({ label: 'sp.frame' });
    const ts = (i: number): GPUComputePassDescriptor['timestampWrites'] =>
      this.qset ? { querySet: this.qset, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 } : undefined;
    const cp = enc.beginComputePass({ label: 'sp.convert', timestampWrites: ts(0) });
    cp.setBindGroup(0, cbg);
    cp.setPipeline(P.calib); cp.dispatchWorkgroups(CALIB_X, CALIB_Y);
    cp.setPipeline(P.reduce); cp.dispatchWorkgroups(1);
    cp.setPipeline(P.convert);
    cp.dispatchWorkgroups(Math.ceil(Math.ceil(p.srcW / 2) / 8), Math.ceil(Math.ceil(p.srcH / 2) / 8));
    cp.end();
    const wp = enc.beginComputePass({ label: 'sp.warp', timestampWrites: ts(2) });
    wp.setBindGroup(0, wbg);
    wp.setPipeline(P.warp);
    wp.dispatchWorkgroups(Math.ceil(p.outW / 16), Math.ceil(p.outH / 16));
    wp.end();
    if (this.qset) enc.resolveQuerySet(this.qset, 0, 4, this.qbuf!, 0);
    if (canvasTex && target !== canvasTex) enc.copyTextureToTexture({ texture: target }, { texture: canvasTex }, [p.outW, p.outH]);
    this.device.queue.submit([enc.finish()]);
  }
}
