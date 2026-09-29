/**
 * Stillpoint web — linear-light frame Blender (owner: GPU agent): synthetic shutter / motion blur for retimed exports.
 *
 *   const blender = await Blender.create(device, plan.outW, plan.outH, { transfer: blendTransferFor(src.colorSpace) });
 *   const out = blender.blend([warpedA, warpedB], [0.35, 0.65], { timestamp: outPtsUs, duration: frameDurUs });
 *   encoder.encode(out); out.close();
 *
 * out = enc( sum_i w_i dec(in_i) ) per pixel and channel, where dec/enc is the pictures' transfer function (default the
 * camera's BT.709 OETF: Warper outputs hold the source R'G'B' without a transfer change), i.e. the average happens in
 * LINEAR light — what a longer physical shutter integrates (a code-value average darkens bright streaks). Weights are
 * normalised to sum 1 (non-positive weights are dropped, repeated inputs merged).
 *
 * Inputs (all outW x outH): VideoFrames (canvas snapshots from Warper.warp(); imported zero-copy as external
 * textures) and/or GPUTextures (Warper.warpTo() targets, or any rgba8unorm / bgra8unorm / rgba16float / rgba32float
 * texture with TEXTURE_BINDING). One tap = exact passthrough: a VideoFrame is re-stamped without a copy
 * (passthrough 'clone', default), otherwise it is copied through the GPU with the identity transfer (bit-exact).
 * Up to 3 taps per dispatch; more taps accumulate in a float32 buffer (16 B/px, allocated on first use).
 * Output: a VideoFrame of this.canvas (8-bit R'G'B', same colour contract as warp.ts: VideoEncoder-ready).
 * One submit per blend(); each returned VideoFrame is a snapshot, so blend() may be called again immediately.
 */
import BLEND_WGSL from './blend.wgsl?raw';
import { configureCanvas, type FrameTiming } from './warp';

export const BLEND_TRANSFERS = ['bt709', 'srgb', 'gamma2.2', 'gamma2.4', 'linear'] as const;
export type BlendTransfer = (typeof BLEND_TRANSFERS)[number];
export type BlendInput = VideoFrame | GPUTexture;

export interface BlenderOptions {
  /** transfer of the pictures being blended (default 'bt709' = camera OETF; 'linear' = plain code-value average) */
  transfer?: BlendTransfer;
  /** single-tap VideoFrame input: 'clone' (default: re-stamped, zero copy, bit-exact) or 'gpu' (copied through the
   *  shader with the identity transfer — also bit-exact; tests) */
  passthrough?: 'clone' | 'gpu';
  /** debug: 'copy' forces render -> own texture -> copy into the canvas */
  canvasPath?: 'auto' | 'copy';
}

/** Blend transfer for a source VideoColorSpace (the Warper keeps the source transfer in its R'G'B' output). */
export function blendTransferFor(cs?: { transfer?: string | null } | null): BlendTransfer {
  switch (cs?.transfer) {
    case 'iec61966-2-1': return 'srgb';
    case 'linear': return 'linear';
    // bt709 / smpte170m / unknown. PQ / HLG sources reach the Warper already tone-mapped to SDR by the browser;
    // BT.709 is the closest description of what is left.
    default: return 'bt709';
  }
}

/** Reference (float64) of the blend of 8-bit-normalised values, for tests/tools: enc(sum w_i dec(v_i)). */
export function blendRef(values: number[], weights: number[], transfer: BlendTransfer = 'bt709'): number {
  const dec = (e0: number) => {
    const e = Math.min(Math.max(e0, 0), 1);
    switch (transfer) {
      case 'bt709': return e < 0.081 ? e / 4.5 : ((e + 0.099) / 1.099) ** (1 / 0.45);
      case 'srgb': return e <= 0.04045 ? e / 12.92 : ((e + 0.055) / 1.055) ** 2.4;
      case 'gamma2.2': return e ** 2.2;
      case 'gamma2.4': return e ** 2.4;
      default: return e;
    }
  };
  const enc = (l0: number) => {
    const l = Math.max(l0, 0);
    switch (transfer) {
      case 'bt709': return l < 0.018 ? 4.5 * l : 1.099 * l ** 0.45 - 0.099;
      case 'srgb': return l <= 0.0031308 ? 12.92 * l : 1.055 * l ** (1 / 2.4) - 0.055;
      case 'gamma2.2': return l ** (1 / 2.2);
      case 'gamma2.4': return l ** (1 / 2.4);
      default: return l;
    }
  };
  const ws = weights.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (let i = 0; i < values.length; i++) acc += (weights[i] / ws) * dec(values[i]);
  return enc(acc);
}

const UNI = 256;            // uniform slot stride (minUniformBufferOffsetAlignment)
const UNI_BYTES = 48;
const TAPS_PER_PASS = 3;    // 3 external textures = 12 sampled-texture slots of the 16 guaranteed

interface BlendPipes { bglExt: GPUBindGroupLayout; bglTex: GPUBindGroupLayout; ext: GPUComputePipeline; tex: GPUComputePipeline }
const pipeCache = new WeakMap<GPUDevice, Map<string, Promise<BlendPipes>>>();

async function buildPipes(device: GPUDevice, outFormat: GPUTextureFormat): Promise<BlendPipes> {
  const code = BLEND_WGSL.replace('texture_storage_2d<rgba8unorm, write>', `texture_storage_2d<${outFormat}, write>`);
  const module = device.createShaderModule({ label: 'sp.blend', code });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter(m => m.type === 'error');
  if (errs.length) throw new Error('blend.wgsl: ' + errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
  const C = GPUShaderStage.COMPUTE;
  const common: GPUBindGroupLayoutEntry[] = [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 7, visibility: C, buffer: { type: 'storage' } },
    { binding: 8, visibility: C, storageTexture: { access: 'write-only', format: outFormat } },
  ];
  const bglExt = device.createBindGroupLayout({ label: 'sp.bgl.blend_ext', entries: [...common,
    ...[1, 2, 3].map(b => ({ binding: b, visibility: C, externalTexture: {} }))] });
  const bglTex = device.createBindGroupLayout({ label: 'sp.bgl.blend_tex', entries: [...common,
    ...[4, 5, 6].map(b => ({ binding: b, visibility: C, texture: { sampleType: 'unfilterable-float' as const } }))] });
  const mk = (bgl: GPUBindGroupLayout, entryPoint: string) => device.createComputePipelineAsync({ label: 'sp.' + entryPoint,
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module, entryPoint } });
  const [ext, tex] = await Promise.all([mk(bglExt, 'blend_ext'), mk(bglTex, 'blend_tex')]);
  return { bglExt, bglTex, ext, tex };
}

function pipesFor(device: GPUDevice, outFormat: GPUTextureFormat): Promise<BlendPipes> {
  let m = pipeCache.get(device);
  if (!m) { m = new Map(); pipeCache.set(device, m); }
  let p = m.get(outFormat);
  if (!p) { p = buildPipes(device, outFormat); m.set(outFormat, p); p.catch(() => m!.delete(outFormat)); }
  return p;
}

const isFrame = (x: BlendInput): x is VideoFrame => typeof VideoFrame !== 'undefined' && x instanceof VideoFrame;

export class Blender {
  readonly device: GPUDevice;
  readonly canvas: OffscreenCanvas;
  readonly width: number;
  readonly height: number;
  private transferIdx: number;
  private readonly passthrough: 'clone' | 'gpu';
  private readonly pipes: BlendPipes;
  private readonly ctx: GPUCanvasContext;
  private readonly directCanvas: boolean;
  private readonly outFormat: GPUTextureFormat;
  private uni: GPUBuffer;
  private uniSlots: number;
  private acc: GPUBuffer | null = null;
  private readonly dummy: GPUBuffer;
  private outTex: GPUTexture | null = null;
  private destroyed = false;

  private constructor(device: GPUDevice, width: number, height: number, opts: BlenderOptions, pipes: BlendPipes,
                      canvas: OffscreenCanvas, ctx: GPUCanvasContext, direct: boolean, outFormat: GPUTextureFormat) {
    this.device = device; this.width = width; this.height = height;
    this.transferIdx = BLEND_TRANSFERS.indexOf(opts.transfer ?? 'bt709');
    if (this.transferIdx < 0) throw new Error(`unknown transfer ${opts.transfer}`);
    this.passthrough = opts.passthrough ?? 'clone';
    this.pipes = pipes; this.canvas = canvas; this.ctx = ctx; this.directCanvas = direct; this.outFormat = outFormat;
    this.uniSlots = 4;
    this.uni = device.createBuffer({ label: 'sp.blend.uni', size: UNI * this.uniSlots, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dummy = device.createBuffer({ label: 'sp.blend.dummy', size: 16, usage: GPUBufferUsage.STORAGE });
  }

  static async create(device: GPUDevice, width: number, height: number, opts: BlenderOptions = {}): Promise<Blender> {
    if (!(width >= 1 && height >= 1)) throw new Error(`bad blend size ${width}x${height}`);
    const outFormat: GPUTextureFormat = device.features.has('bgra8unorm-storage') ? 'bgra8unorm' : 'rgba8unorm';
    const pipes = await pipesFor(device, outFormat);
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('WebGPU canvas context unavailable');
    const direct = await configureCanvas(device, ctx, outFormat, opts.canvasPath ?? 'auto');
    return new Blender(device, width, height, opts, pipes, canvas, ctx, direct, outFormat);
  }

  get transfer(): BlendTransfer { return BLEND_TRANSFERS[this.transferIdx]; }
  set transfer(t: BlendTransfer) {
    const i = BLEND_TRANSFERS.indexOf(t);
    if (i < 0) throw new Error(`unknown transfer ${t}`);
    this.transferIdx = i;
  }

  /** Blend `inputs` with `weights` (normalised) into a new VideoFrame stamped `timing` (caller closes it). */
  blend(inputs: BlendInput[], weights: ArrayLike<number>, timing: FrameTiming): VideoFrame {
    const taps = this.taps(inputs, weights);
    const init = { timestamp: timing.timestamp, ...(timing.duration != null ? { duration: timing.duration } : {}) };
    if (taps.length === 1 && this.passthrough === 'clone' && isFrame(taps[0].input))
      return new VideoFrame(taps[0].input, init);
    this.encode(taps);
    return new VideoFrame(this.canvas, init);
  }

  /** Blend into this.canvas only (no VideoFrame). */
  render(inputs: BlendInput[], weights: ArrayLike<number>): void {
    this.encode(this.taps(inputs, weights));
  }

  // ---- streaming accumulation (synthetic shutter: many weighted pictures without holding them all)

  /** A linear-light float accumulator (16 B per pixel) for accumulate() / resolve(). The caller destroys it. */
  createAccumulator(label = 'sp.blend.accum'): GPUBuffer {
    return this.device.createBuffer({ label, size: this.width * this.height * 16, usage: GPUBufferUsage.STORAGE });
  }

  /** acc = (fresh ? 0 : acc) + Σ w_i · dec(in_i) over up to 3 textures (outW x outH, e.g. Warper.warpTo targets).
   *  Weights are used as given (NOT normalised: a whole output's weights over all calls should sum to 1). */
  accumulate(acc: GPUBuffer, inputs: GPUTexture[], weights: ArrayLike<number>, fresh: boolean): void {
    if (!inputs.length || inputs.length > TAPS_PER_PASS || inputs.length !== weights.length) throw new Error('accumulate: 1-3 inputs with weights');
    this.dispatchTex(acc, inputs, Array.from(weights), fresh ? 0 : 1, 0);
  }

  /** enc(acc) into this.canvas -> a VideoFrame stamped `timing` (caller closes it). `like`: any outW x outH texture
   *  (bound with weight 0, not needed for the result). */
  resolve(acc: GPUBuffer, timing: FrameTiming, like: GPUTexture): VideoFrame {
    this.dispatchTex(acc, [like], [0], 1, 1);
    return new VideoFrame(this.canvas, { timestamp: timing.timestamp, ...(timing.duration != null ? { duration: timing.duration } : {}) });
  }

  private scratchTex: GPUTexture | null = null;
  private dispatchTex(acc: GPUBuffer, inputs: GPUTexture[], w: number[], accIn: number, fin: number): void {
    if (this.destroyed) throw new Error('Blender destroyed');
    const dev = this.device, W = this.width, H = this.height;
    for (const t of inputs) if (t.width !== W || t.height !== H) throw new Error(`blend input ${t.width}x${t.height} != ${W}x${H}`);
    if (acc.size < W * H * 16) throw new Error('accumulator too small');
    const data = new ArrayBuffer(UNI_BYTES);
    const f = new Float32Array(data), u = new Uint32Array(data);
    w.forEach((x, j) => { f[j] = x; });
    u[4] = inputs.length; u[5] = this.transferIdx; u[6] = accIn; u[7] = fin; u[8] = W; u[9] = H;
    dev.queue.writeBuffer(this.uni, 0, data);
    let target: GPUTexture, canvasTex: GPUTexture | null = null;
    if (!fin) {
      // not written (fin = 0), but the layout needs a storage texture bound
      if (!this.scratchTex) this.scratchTex = dev.createTexture({ label: 'sp.blend.scratch', size: [W, H], format: this.outFormat, usage: GPUTextureUsage.STORAGE_BINDING });
      target = this.scratchTex;
    } else if (this.directCanvas) target = canvasTex = this.ctx.getCurrentTexture();
    else {
      canvasTex = this.ctx.getCurrentTexture();
      if (!this.outTex) this.outTex = dev.createTexture({ label: 'sp.blend.out', size: [W, H], format: this.outFormat,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
      target = this.outTex;
    }
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.uni, offset: 0, size: UNI_BYTES } },
      { binding: 7, resource: { buffer: acc } },
      { binding: 8, resource: target.createView() },
    ];
    for (let j = 0; j < TAPS_PER_PASS; j++) entries.push({ binding: 4 + j, resource: inputs[Math.min(j, inputs.length - 1)].createView() });
    const enc = dev.createCommandEncoder({ label: 'sp.blend.accum' });
    const pass = enc.beginComputePass({ label: 'sp.blend.accum' });
    pass.setPipeline(this.pipes.tex);
    pass.setBindGroup(0, dev.createBindGroup({ layout: this.pipes.bglTex, entries }));
    pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
    pass.end();
    if (canvasTex && target !== canvasTex) enc.copyTextureToTexture({ texture: target }, { texture: canvasTex }, [W, H]);
    dev.queue.submit([enc.finish()]);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const b of [this.uni, this.acc, this.dummy]) b?.destroy();
    this.outTex?.destroy();
    this.scratchTex?.destroy();
    this.outTex = null; this.acc = null; this.scratchTex = null;
    try { this.ctx.unconfigure(); } catch { /* already gone */ }
  }

  // ------------------------------------------------------------------------------------------ internals
  private taps(inputs: BlendInput[], weights: ArrayLike<number>): { input: BlendInput; w: number }[] {
    if (this.destroyed) throw new Error('Blender destroyed');
    if (inputs.length !== weights.length) throw new Error(`blend: ${inputs.length} inputs vs ${weights.length} weights`);
    const merged = new Map<BlendInput, number>();
    for (let i = 0; i < inputs.length; i++) {
      const w = +weights[i];
      if (!(w > 0)) continue;
      const x = inputs[i];
      const [iw, ih] = isFrame(x) ? [x.displayWidth, x.displayHeight] : [x.width, x.height];
      if (iw !== this.width || ih !== this.height) throw new Error(`blend input ${iw}x${ih} != ${this.width}x${this.height}`);
      merged.set(x, (merged.get(x) ?? 0) + w);
    }
    let sum = 0;
    for (const w of merged.values()) sum += w;
    if (!(sum > 0) || !Number.isFinite(sum)) throw new Error('blend: no positive weight');
    return [...merged].map(([input, w]) => ({ input, w: w / sum }));
  }

  private encode(taps: { input: BlendInput; w: number }[]): void {
    const dev = this.device, W = this.width, H = this.height;
    // group by input kind, TAPS_PER_PASS per dispatch
    const chunks: { ext: boolean; taps: { input: BlendInput; w: number }[] }[] = [];
    for (const ext of [true, false]) {
      const list = taps.filter(t => isFrame(t.input) === ext);
      for (let i = 0; i < list.length; i += TAPS_PER_PASS) chunks.push({ ext, taps: list.slice(i, i + TAPS_PER_PASS) });
    }
    const n = chunks.length;
    if (n > this.uniSlots) {
      this.uni.destroy();
      this.uniSlots = n;
      this.uni = dev.createBuffer({ label: 'sp.blend.uni', size: UNI * n, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    if (n > 1 && !this.acc)
      this.acc = dev.createBuffer({ label: 'sp.blend.acc', size: W * H * 16, usage: GPUBufferUsage.STORAGE });
    // one tap: identity transfer -> exact copy (weights are normalised, so w == 1)
    const tr = taps.length === 1 ? BLEND_TRANSFERS.indexOf('linear') : this.transferIdx;
    const data = new ArrayBuffer(UNI * n);
    chunks.forEach((c, i) => {
      const f = new Float32Array(data, UNI * i, UNI_BYTES / 4), u = new Uint32Array(data, UNI * i, UNI_BYTES / 4);
      c.taps.forEach((t, j) => { f[j] = t.w; });
      u[4] = c.taps.length; u[5] = tr; u[6] = i > 0 ? 1 : 0; u[7] = i === n - 1 ? 1 : 0;
      u[8] = W; u[9] = H;
    });
    dev.queue.writeBuffer(this.uni, 0, data);

    let target: GPUTexture, canvasTex: GPUTexture;
    if (this.directCanvas) target = canvasTex = this.ctx.getCurrentTexture();
    else {
      canvasTex = this.ctx.getCurrentTexture();
      if (!this.outTex) this.outTex = dev.createTexture({ label: 'sp.blend.out', size: [W, H], format: this.outFormat,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
      target = this.outTex;
    }
    const outView = target.createView();
    const imported = new Map<VideoFrame, GPUExternalTexture>();
    const enc = dev.createCommandEncoder({ label: 'sp.blend' });
    const pass = enc.beginComputePass({ label: 'sp.blend' });
    chunks.forEach((c, i) => {
      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: this.uni, offset: UNI * i, size: UNI_BYTES } },
        { binding: 7, resource: { buffer: this.acc ?? this.dummy } },
        { binding: 8, resource: outView },
      ];
      for (let j = 0; j < TAPS_PER_PASS; j++) {
        const x = c.taps[Math.min(j, c.taps.length - 1)].input;
        if (c.ext) {
          const f = x as VideoFrame;
          let e = imported.get(f);
          if (!e) { e = dev.importExternalTexture({ source: f, colorSpace: 'srgb' }); imported.set(f, e); }
          entries.push({ binding: 1 + j, resource: e });
        } else {
          entries.push({ binding: 4 + j, resource: (x as GPUTexture).createView() });
        }
      }
      const bg = dev.createBindGroup({ layout: c.ext ? this.pipes.bglExt : this.pipes.bglTex, entries });
      pass.setPipeline(c.ext ? this.pipes.ext : this.pipes.tex);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
    });
    pass.end();
    if (target !== canvasTex) enc.copyTextureToTexture({ texture: target }, { texture: canvasTex }, [W, H]);
    dev.queue.submit([enc.finish()]);
  }
}
