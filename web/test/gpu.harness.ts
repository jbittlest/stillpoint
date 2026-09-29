// Browser side of the GPU module tests (owner: GPU agent). Driven by test/gpu.run.mjs in headless Chrome:
//   selftest  footage-free: synthetic NV12/I420/RGBX CPU frames + a real decoder frame (encode->decode round trip),
//             float64 JS reference of geometry + Lanczos/Catmull-Rom; thresholds -> exit code (run this in CI)
//   golden    real footage (fixtures/gpu/*.json + clips via /__clips/): coord maps, float luma, 8-bit VideoFrame
//             output and an encoder round trip, dumped for test/gpu.golden.py (float64 render_ref.py reference)
//   bench     4K throughput of warp() -> VideoFrame, GPU pass times (timestamp queries);  variants: tap-fetch A/B
//   probe / extfit / copyfit / enccolor   the browser-behaviour measurements behind warp.ts's colour contract
//             (extfit / copyfit [fixture...]: default all fixtures)
//   scaledself / scaled / shimmer / blend / benchscaled   outputs != source size + the Blender (test/gpu.scaled.ts)
import o3Fixture from './fixtures/gpu/o3_0026.json';
import oa4Fixture from './fixtures/gpu/oa4_0005.json';
import { Warper, type WarpKernel } from '../src/gpu/warp';
import type { Plan } from '../src/types';
import { runScaled, SCALED_MODES } from './gpu.scaled';

interface FixSample { i: number; offset: number; size: number; pts: number; key: boolean }
interface FixRecord { k: number; sample: number; pts: number; outFx: number; rowMats: number[] }
interface Fixture {
  clip: string; srcW: number; srcH: number; outW: number; outH: number; nRows: number; readoutS: number;
  lens: { model: 'kb4' | 'pinhole'; fx: number; fy: number; cx: number; cy: number; k: number[]; width: number; height: number };
  records: FixRecord[];
  video: { codec: string; fourcc: string; description: string; codedWidth: number; codedHeight: number; timescale: number; samples: FixSample[] };
}
const FIXTURES: Record<string, Fixture> = { o3_0026: o3Fixture as Fixture, oa4_0005: oa4Fixture as Fixture };

const logEl = document.getElementById('log')!;
function log(...a: unknown[]) { const s = a.map(x => typeof x === 'string' ? x : JSON.stringify(x)).join(' '); console.log(s); logEl.textContent += s + '\n'; }
function done(result: unknown) { (window as any).__result = result; (window as any).__done = true; }

async function post(name: string, data: ArrayBufferView | ArrayBuffer | string) {
  const r = await fetch(`/__gpu_out/${encodeURIComponent(name)}`, { method: 'POST', body: data as BodyInit });
  if (!r.ok) throw new Error(`post ${name}: ${r.status}`);
}

function b64(s: string): Uint8Array { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }

async function fetchRange(clip: string, off: number, size: number): Promise<Uint8Array> {
  const r = await fetch(`/__clips/${encodeURIComponent(clip)}`, { headers: { Range: `bytes=${off}-${off + size - 1}` } });
  if (r.status !== 206) throw new Error(`range fetch ${clip}: HTTP ${r.status}`);
  const u = new Uint8Array(await r.arrayBuffer());
  if (u.length !== size) throw new Error(`range fetch short ${u.length} != ${size}`);
  return u;
}

/** Decode the fixture samples needed for `sampleIdx` (from its sync sample) and return that VideoFrame. */
async function decodeSample(fx: Fixture, sampleIdx: number, hw: HardwareAcceleration = 'no-preference'): Promise<VideoFrame> {
  const v = fx.video;
  const byI = new Map(v.samples.map(s => [s.i, s]));
  let s0 = sampleIdx;
  while (!byI.get(s0)?.key) { s0--; if (s0 < 0 || !byI.has(s0)) throw new Error('no sync sample'); }
  const want = byI.get(sampleIdx)!;
  let result: VideoFrame | null = null;
  let err: unknown = null;
  const dec = new VideoDecoder({
    output: f => { if (Math.abs(f.timestamp - Math.round(want.pts * 1e6)) <= 2 && !result) result = f; else f.close(); },
    error: e => { err = e; },
  });
  dec.configure({ codec: v.codec, description: b64(v.description), codedWidth: v.codedWidth, codedHeight: v.codedHeight, hardwareAcceleration: hw });
  for (let i = s0; i <= sampleIdx; i++) {
    const s = byI.get(i)!;
    const data = await fetchRange(fx.clip, s.offset, s.size);
    dec.decode(new EncodedVideoChunk({ type: s.key ? 'key' : 'delta', timestamp: Math.round(s.pts * 1e6), duration: Math.round(1e6 / 59.94), data }));
  }
  await dec.flush();
  dec.close();
  if (err) throw err;
  if (!result) throw new Error('target frame not produced');
  return result;
}

async function probe() {
  const r: any = { ua: navigator.userAgent, gpu: !!navigator.gpu };
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { r.adapter = null; return r; }
  r.adapter = { info: { vendor: adapter.info?.vendor, arch: adapter.info?.architecture, desc: adapter.info?.description },
    features: [...adapter.features], maxTex: adapter.limits.maxTextureDimension2D, maxSB: adapter.limits.maxStorageBufferBindingSize,
    maxBuf: adapter.limits.maxBufferSize };
  r.prefFmt = navigator.gpu.getPreferredCanvasFormat();
  const device = await adapter.requestDevice();
  for (const [name, fx] of Object.entries(FIXTURES)) {
    const o: any = {};
    r[name] = o;
    try {
      for (const hw of ['prefer-hardware', 'prefer-software', 'no-preference'] as HardwareAcceleration[]) {
        const s = await VideoDecoder.isConfigSupported({ codec: fx.video.codec, description: b64(fx.video.description), codedWidth: fx.video.codedWidth, codedHeight: fx.video.codedHeight, hardwareAcceleration: hw });
        o['decode_' + hw] = s.supported;
      }
      const key = fx.video.samples.find(s => s.key)!;
      const t0 = performance.now();
      const f = await decodeSample(fx, key.i);
      o.decodeMs = performance.now() - t0;
      o.frame = { format: f.format, coded: [f.codedWidth, f.codedHeight], display: [f.displayWidth, f.displayHeight],
        visible: f.visibleRect && [f.visibleRect.x, f.visibleRect.y, f.visibleRect.width, f.visibleRect.height],
        colorSpace: f.colorSpace.toJSON(), ts: f.timestamp };
      try {
        const size = f.allocationSize();
        const buf = new Uint8Array(size);
        const layout = await f.copyTo(buf);
        o.copyTo = { size, layout };
      } catch (e) { o.copyTo = 'ERR ' + e; }
      try {
        const size = f.allocationSize({ format: 'RGBA' } as any);
        o.copyToRGBA = { size };
      } catch (e) { o.copyToRGBA = 'ERR ' + e; }
      // external texture textureLoad vs sampling
      try {
        const ext = device.importExternalTexture({ source: f });
        const mod = device.createShaderModule({ code: `
          @group(0) @binding(0) var t: texture_external;
          @group(0) @binding(1) var<storage, read_write> o: array<vec4f>;
          @compute @workgroup_size(1) fn main() {
            let d = textureDimensions(t);
            o[0] = vec4f(f32(d.x), f32(d.y), 0, 0);
            for (var i = 0u; i < 8u; i++) {
              o[1u + i] = textureLoad(t, vec2u(1000u + 2u*i, 700u + i));
            }
          }` });
        const info = await mod.getCompilationInfo();
        if (info.messages.length) o.extCompile = info.messages.map(m => m.message);
        const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
        const ob = device.createBuffer({ size: 16 * 9, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const rb = device.createBuffer({ size: 16 * 9, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: ext }, { binding: 1, resource: { buffer: ob } }] });
        const enc = device.createCommandEncoder();
        const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(1); p.end();
        enc.copyBufferToBuffer(ob, 0, rb, 0, 16 * 9);
        device.queue.submit([enc.finish()]);
        await rb.mapAsync(GPUMapMode.READ);
        const v = new Float32Array(rb.getMappedRange().slice(0));
        rb.unmap();
        o.ext = { dims: [v[0], v[1]], px: Array.from({ length: 8 }, (_, i) => Array.from(v.subarray(4 + 4 * i, 8 + 4 * i))) };
        // raw YUV at the same pixels (from copyTo, if planar)
        const size = f.allocationSize();
        const buf = new Uint8Array(size);
        const layout = await f.copyTo(buf);
        const W = f.codedWidth;
        const is16 = (f.format || '').includes('P10') || (f.format || '').includes('P12');
        const rd = (plane: number, x: number, y: number, comp = 0, ncomp = 1) => {
          const L = layout[plane];
          const off = L.offset + y * L.stride + (x * ncomp + comp) * (is16 ? 2 : 1);
          return is16 ? buf[off] | (buf[off + 1] << 8) : buf[off];
        };
        o.raw = [];
        for (let i = 0; i < 8; i++) {
          const x = 1000 + 2 * i, y = 700 + i;
          let Y = 0, U = 0, V = 0;
          if (f.format === 'NV12') { Y = rd(0, x, y); U = rd(1, x >> 1, y >> 1, 0, 2); V = rd(1, x >> 1, y >> 1, 1, 2); }
          else { Y = rd(0, x, y); U = rd(1, x >> 1, y >> 1); V = rd(2, x >> 1, y >> 1); }
          o.raw.push([Y, U, V]);
        }
        o.W = W;
      } catch (e) { o.ext = 'ERR ' + e; }
      f.close();
    } catch (e) { o.error = String(e); }
  }
  for (const [k, c] of Object.entries({
    hevc_main_4k: { codec: 'hvc1.1.6.L153.B0', width: 3840, height: 2160 },
    hevc_main10_4k: { codec: 'hvc1.2.4.L153.B0', width: 3840, height: 2160 },
    h264_4k: { codec: 'avc1.640034', width: 3840, height: 2160 },
  })) {
    try { r['enc_' + k] = (await VideoEncoder.isConfigSupported({ ...c, bitrate: 80e6, framerate: 60 })).supported; } catch (e) { r['enc_' + k] = 'ERR ' + e; }
  }
  device.destroy();
  return r;
}

/** Dump a crop of importExternalTexture(textureLoad) RGBA (float32) + the raw planes (when copyTo works). */
async function extFit(names: string[]) {
  const adapter = (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }))!;
  const device = await adapter.requestDevice();
  const X0 = 1000, Y0 = 700, N = 256;
  const r: any = {};
  for (const name of names) {
    const fx = FIXTURES[name];
    const key = fx.video.samples.find(s => s.key)!;
    const f = await decodeSample(fx, key.i);
    r[name] = { sample: key.i, format: f.format, colorSpace: f.colorSpace.toJSON() };
    for (const cs of ['srgb', 'display-p3'] as PredefinedColorSpace[]) {
      const ext = device.importExternalTexture({ source: f, colorSpace: cs });
      const mod = device.createShaderModule({ code: `
        @group(0) @binding(0) var t: texture_external;
        @group(0) @binding(1) var<storage, read_write> o: array<vec4f>;
        @compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
          if (g.x >= ${N}u || g.y >= ${N}u) { return; }
          o[g.y * ${N}u + g.x] = textureLoad(t, vec2u(${X0}u + g.x, ${Y0}u + g.y));
          o[${N * N}u + g.y * ${N}u + g.x] = textureSampleBaseClampToEdge(t, smp, (vec2f(f32(${X0}u + g.x), f32(${Y0}u + g.y)) + 0.5) / vec2f(textureDimensions(t)));
        }
        @group(0) @binding(2) var smp: sampler;` });
      const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
      const bytes = 2 * N * N * 16;
      const ob = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const rb = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: ext }, { binding: 1, resource: { buffer: ob } }, { binding: 2, resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }) }] });
      const enc = device.createCommandEncoder();
      const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(N / 16, N / 16); p.end();
      enc.copyBufferToBuffer(ob, 0, rb, 0, bytes);
      device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      await post(`extfit_${name}_${cs}.f32`, new Float32Array(rb.getMappedRange().slice(0)));
      rb.unmap();
    }
    if (f.format) {
      const buf = new Uint8Array(f.allocationSize());
      r[name].layout = await f.copyTo(buf);
      await post(`extfit_${name}_raw.bin`, buf);
    }
    f.close();
  }
  r.crop = [X0, Y0, N];
  device.destroy();
  return r;
}

/** copyExternalImageToTexture(VideoFrame) into float textures: which precision / conversion survives? */
async function copyFit(names: string[]) {
  const adapter = (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }))!;
  const device = await adapter.requestDevice();
  const X0 = 1000, Y0 = 700, N = 256;
  const r: any = {};
  for (const name of names) {
    const fx = FIXTURES[name];
    const key = fx.video.samples.find(s => s.key)!;
    const f = await decodeSample(fx, key.i);
    r[name] = {};
    for (const fmt of ['rgba16float', 'rgba32float'] as GPUTextureFormat[]) {
      try {
        const tex = device.createTexture({ size: [f.displayWidth, f.displayHeight], format: fmt, usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
        const t0 = performance.now();
        device.queue.copyExternalImageToTexture({ source: f }, { texture: tex, colorSpace: 'srgb' }, [f.displayWidth, f.displayHeight]);
        await device.queue.onSubmittedWorkDone();
        r[name][fmt] = { ms: performance.now() - t0 };
        const bpp = fmt === 'rgba16float' ? 8 : 16;
        const bpr = Math.ceil(N * bpp / 256) * 256;
        const rb = device.createBuffer({ size: bpr * N, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const enc = device.createCommandEncoder();
        enc.copyTextureToBuffer({ texture: tex, origin: [X0, Y0] }, { buffer: rb, bytesPerRow: bpr }, [N, N]);
        device.queue.submit([enc.finish()]);
        await rb.mapAsync(GPUMapMode.READ);
        await post(`copyfit_${name}_${fmt}.bin`, new Uint8Array(rb.getMappedRange().slice(0)));
        rb.unmap(); tex.destroy();
      } catch (e) { r[name][fmt] = 'ERR ' + e; }
    }
    f.close();
  }
  device.destroy();
  return r;
}

/** How does VideoEncoder convert a WebGPU-canvas VideoFrame (RGB) to Y'CbCr? Known RGB patches -> encode -> decode. */
async function encColor() {
  const adapter = (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }))!;
  const device = await adapter.requestDevice();
  const S = 512, P = 32;   // 16x16 patches of 32x32 px
  const r: any = {};
  for (const fmt of ['bgra8unorm', 'rgba8unorm', 'rgba16float'] as GPUTextureFormat[]) {
    const canvas = new OffscreenCanvas(S, S);
    const ctx = canvas.getContext('webgpu')!;
    ctx.configure({ device, format: fmt, alphaMode: 'opaque' });
    const mod = device.createShaderModule({ code: `
      @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)); return vec4f(p[i], 0, 1); }
      @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let i = u32(pos.x) / ${P}u; let j = u32(pos.y) / ${P}u;
        return vec4f(f32(i) / 15.0, f32(j) / 15.0, f32((i * 7u + j * 3u) % 16u) / 15.0, 1.0); }` });
    const pipe = device.createRenderPipeline({ layout: 'auto', vertex: { module: mod, entryPoint: 'vs' }, fragment: { module: mod, entryPoint: 'fs', targets: [{ format: fmt }] } });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(pipe); pass.draw(3); pass.end();
    device.queue.submit([enc.finish()]);
    const vf = new VideoFrame(canvas, { timestamp: 0, duration: 16683 });
    const o: any = { format: vf.format, colorSpace: vf.colorSpace.toJSON(), coded: [vf.codedWidth, vf.codedHeight] };
    r[fmt] = o;
    for (const [cname, codec] of [['h264', 'avc1.640034'], ['hevc', 'hvc1.1.6.L153.B0']]) {
      try {
        const chunks: EncodedVideoChunk[] = [];
        let desc: any = null; let ecs: any = null;
        const venc = new VideoEncoder({ output: (c, md) => { chunks.push(c); if (md?.decoderConfig) { desc = md.decoderConfig; ecs = md.decoderConfig.colorSpace; } }, error: e => { throw e; } });
        venc.configure({ codec, width: S, height: S, bitrate: 40e6, framerate: 60, ...(cname === 'hevc' ? { hevc: { format: 'annexb' } } as any : { avc: { format: 'avc' } }) });
        venc.encode(vf, { keyFrame: true });
        await venc.flush(); venc.close();
        let out: VideoFrame | null = null;
        const vdec = new VideoDecoder({ output: f => { out = f; }, error: e => { throw e; } });
        vdec.configure({ ...desc, hardwareAcceleration: 'no-preference' });
        for (const c of chunks) vdec.decode(c);
        await vdec.flush(); vdec.close();
        const f = out! as VideoFrame;
        const buf = new Uint8Array(f.allocationSize());
        const layout = await f.copyTo(buf);
        o[cname] = { decFormat: f.format, decColorSpace: f.colorSpace.toJSON(), encColorSpace: ecs, layout };
        await post(`enccolor_${fmt}_${cname}.bin`, buf);
        f.close();
      } catch (e) { o[cname] = 'ERR ' + e; }
    }
    vf.close();
  }
  device.destroy();
  return r;
}


/** Plan holding just the fixture records (record r of the fixture = plan index r). */
function fixturePlan(fx: Fixture): Plan {
  const R = fx.records.length, n = fx.nRows * 9;
  const rowMats = new Float32Array(R * n);
  fx.records.forEach((r, i) => rowMats.set(r.rowMats, i * n));
  return {
    srcW: fx.srcW, srcH: fx.srcH, outW: fx.outW, outH: fx.outH,
    lens: { ...fx.lens, k: fx.lens.k.slice(0, 4) as [number, number, number, number] },
    framePts: Float64Array.from(fx.records.map(r => r.pts)), outFx: Float32Array.from(fx.records.map(r => r.outFx)),
    nRows: fx.nRows, rowMats, readoutS: fx.readoutS,
  };
}

async function gpuDevice(features: GPUFeatureName[] = []): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const device = await adapter.requestDevice({ requiredFeatures: features.filter(f => adapter.features.has(f)) });
  device.addEventListener('uncapturederror', (e: any) => log('GPU ERROR', e.error?.message));
  return device;
}

async function frameBytes(f: VideoFrame, format?: VideoPixelFormat): Promise<{ buf: Uint8Array; layout: PlaneLayout[]; format: string | null }> {
  const opts = format ? { format } as VideoFrameCopyToOptions : undefined;
  const buf = new Uint8Array(f.allocationSize(opts));
  const layout = await f.copyTo(buf, opts);
  return { buf, layout, format: format ?? f.format };
}

/** Golden test: decode fixture frames, warp, dump coordinate maps / float luma / 8-bit output for test/gpu.golden.py. */
async function golden(names: string[]) {
  const device = await gpuDevice(argFlags.has('bgra') ? ['bgra8unorm-storage'] : []);
  const res: any = { ua: navigator.userAgent };
  for (const name of names) {
    const fx = FIXTURES[name];
    const plan = fixturePlan(fx);
    const o: any = { records: [] };
    res[name] = o;
    const warpers: Partial<Record<WarpKernel, Warper>> = {};
    for (const kern of ['lanczos3', 'catmullrom'] as WarpKernel[])
      warpers[kern] = await Warper.create(device, plan, { kernel: kern, canvasPath: argFlags.has('copy') ? 'copy' : 'auto' });
    const outs: VideoFrame[] = [];
    const srcs: VideoFrame[] = [];
    for (let r = 0; r < fx.records.length; r++) {
      const rec = fx.records[r];
      const f = await decodeSample(fx, rec.sample);
      srcs.push(f);
      const ro: any = { k: rec.k, sample: rec.sample, ts: f.timestamp, format: f.format, colorSpace: f.colorSpace.toJSON() };
      o.records.push(ro);
      if (f.format) {
        const raw = await frameBytes(f);
        ro.rawLayout = raw.layout;
        await post(`golden_${name}_r${r}_src.bin`, raw.buf);
      }
      const cm = await warpers.lanczos3!.coordMap(r, 0.5);
      ro.coordMap = [cm.width, cm.height];
      await post(`golden_${name}_r${r}_coord050.f32`, cm.data);
      for (const kern of ['lanczos3', 'catmullrom'] as WarpKernel[]) {
        const t0 = performance.now();
        const luma = await warpers[kern]!.readbackLuma(f, r);
        ro['lumaMs_' + kern] = performance.now() - t0;
        await post(`golden_${name}_r${r}_luma_${kern}.f32`, luma);
      }
      ro.color = await warpers.lanczos3!.colorInfo();
    }
    // Deliverable path: warp() -> VideoFrame (8-bit canvas). All records back-to-back before reading any of them
    // (checks that each VideoFrame snapshots the canvas), then read the 8-bit pixels.
    for (let r = 0; r < srcs.length; r++) outs.push(warpers.lanczos3!.warp(srcs[r], r));
    for (let r = 0; r < outs.length; r++) {
      const v = outs[r];
      o.records[r].out = { format: v.format, colorSpace: v.colorSpace.toJSON(), ts: v.timestamp, dur: v.duration, size: [v.displayWidth, v.displayHeight] };
      const px = await frameBytes(v, 'RGBA');
      o.records[r].out.layout = px.layout;
      await post(`golden_${name}_r${r}_out_rgba.bin`, px.buf);
      v.close();
    }
    // End to end: warp() -> VideoEncoder (H.264 / HEVC, what the app ships) -> VideoDecoder -> NV12 bytes.
    o.e2e = {};
    for (const [cname, codec] of [['h264', 'avc1.640034'], ['hevc', 'hvc1.1.6.L156.B0']] as const) {
      try {
        const chunks: EncodedVideoChunk[] = [];
        let cfg: VideoDecoderConfig | null = null;
        let encErr: unknown = null;
        const venc = new VideoEncoder({ output: (c, md) => { chunks.push(c); if (md?.decoderConfig) cfg = md.decoderConfig; }, error: e => { encErr = e; } });
        venc.configure({ codec, width: plan.outW, height: plan.outH, bitrate: 150e6, framerate: 60, latencyMode: 'quality',
          ...(cname === 'h264' ? { avc: { format: 'avc' } } : {}) } as VideoEncoderConfig);
        const v = warpers.lanczos3!.warp(srcs[0], 0);
        venc.encode(v, { keyFrame: true });
        v.close();
        await venc.flush(); venc.close();
        if (encErr) throw encErr;
        let dec: VideoFrame | null = null;
        const vdec = new VideoDecoder({ output: fr => { dec = fr; }, error: e => { encErr = e; } });
        vdec.configure({ ...cfg!, hardwareAcceleration: 'no-preference' });
        for (const c of chunks) vdec.decode(c);
        await vdec.flush(); vdec.close();
        if (encErr) throw encErr;
        const df = dec! as VideoFrame;
        const px = await frameBytes(df);
        o.e2e[cname] = { bytes: chunks.reduce((a, c) => a + c.byteLength, 0), decFormat: df.format, layout: px.layout,
          encColorSpace: (cfg as any)?.colorSpace ?? null };
        if (df.format === 'NV12' || df.format === 'I420') await post(`golden_${name}_r0_e2e_${cname}.bin`, px.buf);
        df.close();
      } catch (e) { o.e2e[cname] = 'ERR ' + e; }
    }
    for (const f of srcs) f.close();
    for (const w of Object.values(warpers)) w!.destroy();
  }
  device.destroy();
  return res;
}

/** Warp throughput at the fixture resolution: frames/s for warp() -> VideoFrame (closed immediately), GPU pass times. */
async function bench(names: string[]) {
  const device = await gpuDevice(['timestamp-query', ...(argFlags.has('bgra') ? ['bgra8unorm-storage' as GPUFeatureName] : [])]);
  const res: any = {};
  const med = (a: number[]) => { const b = [...a].sort((x, y) => x - y); return +b[b.length >> 1].toFixed(2); };
  for (const name of names) {
    const fx = FIXTURES[name];
    const plan = fixturePlan(fx);
    const o: any = {};
    res[name] = o;
    const f = await decodeSample(fx, fx.records[0].sample);
    o.src = [f.displayWidth, f.displayHeight, f.format];
    for (const kern of ['lanczos3', 'catmullrom', 'bilinear'] as WarpKernel[]) {
      const w = await Warper.create(device, plan, { kernel: kern, profile: true });
      const R = plan.framePts.length;
      for (let i = 0; i < 20; i++) w.warp(f, i % R).close();       // warm-up
      await device.queue.onSubmittedWorkDone();
      const N = 300;
      let t0 = performance.now();
      for (let i = 0; i < N; i++) {
        w.warp(f, i % R).close();
        if (i % 8 === 7) await device.queue.onSubmittedWorkDone();   // keep the queue bounded like a real pipeline
      }
      await device.queue.onSubmittedWorkDone();
      const dt = performance.now() - t0;
      t0 = performance.now();
      for (let i = 0; i < N; i++) { w.render(f, i % R); if (i % 8 === 7) await device.queue.onSubmittedWorkDone(); }
      await device.queue.onSubmittedWorkDone();
      const dt3 = performance.now() - t0;
      const conv: number[] = [], wp: number[] = [];
      for (let i = 0; i < 31; i++) { w.render(f, i % R); const p = (await w.profile())!; conv.push(p.convertMs); wp.push(p.warpMs); }
      t0 = performance.now();
      for (let i = 0; i < N; i++) new VideoFrame(w.canvas, { timestamp: f.timestamp }).close();
      const dt5 = performance.now() - t0;
      o[kern] = { fps: +(N / dt * 1000).toFixed(1), msPerFrame: +(dt / N).toFixed(2), fpsRenderOnly: +(N / dt3 * 1000).toFixed(1),
        gpuConvertMsMedian: med(conv), gpuWarpMsMedian: med(wp), gpuWarpMsMin: +Math.min(...wp).toFixed(2), snapshotMs: +(dt5 / N).toFixed(3) };
      log(name, kern, JSON.stringify(o[kern]));
      w.destroy();
    }
    f.close();
  }
  device.destroy();
  return res;
}

/** Interleaved A/B of tap-fetch variants: min GPU warp ms over repeated rounds. */
async function variants(names: string[]) {
  const device = await gpuDevice(['timestamp-query']);
  const res: any = {};
  for (const name of names) {
    const fx = FIXTURES[name];
    const plan = fixturePlan(fx);
    const f = await decodeSample(fx, fx.records[0].sample);
    const cfgs: Record<string, any> = {
      lz_gather: { kernel: 'lanczos3', gather: true }, lz_load: { kernel: 'lanczos3', gather: false },
      cr_gather: { kernel: 'catmullrom', gather: true }, cr_load: { kernel: 'catmullrom', gather: false },
    };
    const ws: Record<string, Warper> = {};
    for (const [k, c] of Object.entries(cfgs)) ws[k] = await Warper.create(device, plan, { ...c, profile: true });
    const times: Record<string, number[]> = {};
    for (let round = 0; round < 5; round++) {
      for (const [k, w] of Object.entries(ws)) {
        for (let i = 0; i < 12; i++) {
          w.render(f, i % plan.framePts.length);
          const p = (await w.profile())!;
          (times[k] ??= []).push(p.warpMs);
        }
      }
    }
    res[name] = Object.fromEntries(Object.entries(times).map(([k, a]) => { const b = [...a].sort((x, y) => x - y); return [k, { min: +b[0].toFixed(2), p25: +b[b.length >> 2].toFixed(2), med: +b[b.length >> 1].toFixed(2) }]; }));
    log(name, JSON.stringify(res[name]));
    for (const w of Object.values(ws)) w.destroy();
    f.close();
  }
  device.destroy();
  return res;
}


// ------------------------------------------------------------------------------------------ portable self-test
// No footage needed: synthetic NV12 / I420 / RGBX frames, a KB4 lens + rolling-shutter row matrices from the O3
// fixture (scaled to 1920x1080), float64 JS reference of the geometry (warp.metal math) and of the resampling.
function refProject(L: Plan['lens'], r: number[]): [number, number] {
  const [X, Y, Z] = r;
  if (L.model === 'pinhole') { const z = Z > 0 ? Math.max(Z, 1e-12) : Math.min(Z, -1e-12); return [L.fx * X / z + L.cx, L.fy * Y / z + L.cy]; }
  const rxy = Math.hypot(X, Y), th = Math.atan2(rxy, Z), t2 = th * th;
  const [k1, k2, k3, k4] = L.k;
  const thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
  const sc = rxy < 1e-12 ? 1 / Math.max(Z, 1e-12) : thd / rxy;
  return [L.fx * X * sc + L.cx, L.fy * Y * sc + L.cy];
}
function refSourceCoord(p: Plan, k: number, X: number, Y: number): [number, number, boolean] {
  const fx = p.outFx[k], cx = (p.outW - 1) / 2, cy = (p.outH - 1) / 2, R = p.nRows, H = p.srcH;
  const rv = [(X - cx) / fx, (Y - cy) / fx, 1];
  const M = p.rowMats.subarray(k * R * 9, (k + 1) * R * 9);
  const apply = (y: number) => {
    const g = Math.min(Math.max(y * (R - 1) / (H - 1), 0), R - 1);
    const j0 = Math.min(Math.floor(g), R - 2), f = g - j0;
    const out = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      const a = M[9 * j0 + 3 * i] * rv[0] + M[9 * j0 + 3 * i + 1] * rv[1] + M[9 * j0 + 3 * i + 2] * rv[2];
      const b = M[9 * j0 + 9 + 3 * i] * rv[0] + M[9 * j0 + 9 + 3 * i + 1] * rv[1] + M[9 * j0 + 9 + 3 * i + 2] * rv[2];
      out[i] = a + (b - a) * f;
    }
    return out;
  };
  let y = 0.5 * (H - 1), ya = 0, ga = 0, uv: [number, number] = [0, 0], z = 1;
  for (let e = 0; e < 3; e++) {
    const rc = apply(y);
    uv = refProject(p.lens, rc); z = rc[2];
    const g = uv[1] - y;
    let yn = uv[1];
    if (e >= 1) { const dy = y - ya; if (dy !== 0) { const sl = (g - ga) / dy; if (sl <= -0.1 && sl >= -2) yn = y - g / sl; } }
    ya = y; ga = g; y = yn;
  }
  const ok = z > 0 && uv[0] >= -0.5 && uv[1] >= -0.5 && uv[0] <= p.srcW - 0.5 && uv[1] <= p.srcH - 0.5;
  return [uv[0], uv[1], ok];
}
function refWeights(f: number, kern: WarpKernel): [number, number[]] {
  if (kern === 'lanczos3') {
    const sinc = (x: number) => x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    const w = [-2, -1, 0, 1, 2, 3].map(i => sinc(f - i) * sinc((f - i) / 3));
    const s = w.reduce((a, b) => a + b, 0);
    return [-2, w.map(v => v / s)];
  }
  if (kern === 'catmullrom') {
    const t = f, t2 = f * f, t3 = t2 * f;
    return [-1, [-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2]];
  }
  return [0, [1 - f, f]];
}
function refSample(img: Float64Array, w: number, h: number, sx: number, sy: number, kern: WarpKernel, ch = 1, c = 0): number {
  const fx = Math.floor(sx), fy = Math.floor(sy);
  const [off, wx] = refWeights(sx - fx, kern);
  const [, wy] = refWeights(sy - fy, kern);
  let acc = 0;
  for (let j = 0; j < wy.length; j++) {
    const yy = Math.min(Math.max(fy + off + j, 0), h - 1);
    let row = 0;
    for (let i = 0; i < wx.length; i++) row += wx[i] * img[(yy * w + Math.min(Math.max(fx + off + i, 0), w - 1)) * ch + c];
    acc += wy[j] * row;
  }
  return acc;
}

async function selftest() {
  const device = await gpuDevice(argFlags.has('bgra') ? ['bgra8unorm-storage'] : []);
  const fx0 = FIXTURES.o3_0026;
  const W = 1920, H = 1080, s = 0.5;
  const plan: Plan = {
    srcW: W, srcH: H, outW: W, outH: H,
    lens: { model: 'kb4', fx: fx0.lens.fx * s, fy: fx0.lens.fy * s, cx: (W - 1) / 2, cy: (H - 1) / 2, k: fx0.lens.k.slice(0, 4) as [number, number, number, number], width: W, height: H },
    framePts: Float64Array.from(fx0.records.map(r => r.pts)), outFx: Float32Array.from(fx0.records.map(r => r.outFx * s)),
    nRows: fx0.nRows, rowMats: Float32Array.from(fx0.records.flatMap(r => r.rowMats)), readoutS: fx0.readoutS,
  };
  // synthetic 8-bit planes (codes), kept inside the legal/in-gamut range
  const Yc = new Float64Array(W * H), Cc = new Float64Array((W / 2) * (H / 2) * 2);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const edge = ((x >> 5) + (y >> 5)) & 1 ? 18 : -18;
    Yc[y * W + x] = Math.round(120 + 45 * Math.sin(x * 0.071 + 0.4 * Math.sin(y * 0.023)) + 25 * Math.cos(y * 0.113 + x * 0.004) + edge);
  }
  for (let y = 0; y < H / 2; y++) for (let x = 0; x < W / 2; x++) {
    Cc[(y * W / 2 + x) * 2] = Math.round(128 + 14 * Math.sin(x * 0.05 + y * 0.013));
    Cc[(y * W / 2 + x) * 2 + 1] = Math.round(128 + 12 * Math.cos(y * 0.061 - x * 0.02));
  }
  const cs: VideoColorSpaceInit = { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: false };
  const nv12 = new Uint8Array(W * H * 3 / 2), i420 = new Uint8Array(W * H * 3 / 2);
  for (let i = 0; i < W * H; i++) nv12[i] = i420[i] = Yc[i];
  for (let i = 0; i < W * H / 4; i++) {
    nv12[W * H + 2 * i] = Cc[2 * i]; nv12[W * H + 2 * i + 1] = Cc[2 * i + 1];
    i420[W * H + i] = Cc[2 * i]; i420[W * H * 5 / 4 + i] = Cc[2 * i + 1];
  }
  // RGBX frame = the same picture as full-res R'G'B' (chroma replicated), BT.709 limited -> RGB
  const rgbx = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const Yn = (Yc[y * W + x] - 16) / 219, ci = ((y >> 1) * W / 2 + (x >> 1)) * 2;
    const cb = (Cc[ci] - 128) / 224, cr = (Cc[ci + 1] - 128) / 224;
    const R = Yn + 1.5748 * cr, B = Yn + 1.8556 * cb, G = (Yn - 0.2126 * R - 0.0722 * B) / 0.7152;
    const o = (y * W + x) * 4;
    rgbx[o] = Math.round(Math.min(Math.max(R, 0), 1) * 255); rgbx[o + 1] = Math.round(Math.min(Math.max(G, 0), 1) * 255);
    rgbx[o + 2] = Math.round(Math.min(Math.max(B, 0), 1) * 255); rgbx[o + 3] = 255;
  }
  const rgbLuma = new Float64Array(W * H);   // luma of the 8-bit RGB picture, code units
  for (let i = 0; i < W * H; i++) rgbLuma[i] = 16 + 219 * (0.2126 * rgbx[4 * i] + 0.7152 * rgbx[4 * i + 1] + 0.0722 * rgbx[4 * i + 2]) / 255;
  const frames: Record<string, () => VideoFrame | Promise<VideoFrame>> = {
    NV12cpu: () => new VideoFrame(nv12, { format: 'NV12', codedWidth: W, codedHeight: H, timestamp: 1000, duration: 16683, colorSpace: cs }),
    I420cpu: () => new VideoFrame(i420, { format: 'I420', codedWidth: W, codedHeight: H, timestamp: 1000, duration: 16683, colorSpace: cs }),
    RGBXcpu: () => new VideoFrame(rgbx, { format: 'RGBX', codedWidth: W, codedHeight: H, timestamp: 1000, duration: 16683 }),
    // a real decoder frame (GPU-backed, imported as planes): encode the synthetic picture, decode it again
    NV12dec: async () => {
      const src = new VideoFrame(nv12, { format: 'NV12', codedWidth: W, codedHeight: H, timestamp: 1000, duration: 16683, colorSpace: cs });
      const chunks: EncodedVideoChunk[] = [];
      let cfg: VideoDecoderConfig | null = null;
      const enc = new VideoEncoder({ output: (c, md) => { chunks.push(c); if (md?.decoderConfig) cfg = md.decoderConfig; }, error: e => { throw e; } });
      enc.configure({ codec: 'avc1.640033', width: W, height: H, bitrate: 60e6, framerate: 60, avc: { format: 'avc' } });
      enc.encode(src, { keyFrame: true }); await enc.flush(); enc.close(); src.close();
      let out: VideoFrame | null = null;
      const dec = new VideoDecoder({ output: f => { out = f; }, error: e => { throw e; } });
      dec.configure({ ...cfg!, hardwareAcceleration: 'prefer-hardware' });
      for (const c of chunks) dec.decode(c);
      await dec.flush(); dec.close();
      return out! as VideoFrame;
    },
  };
  const lumaLimit: Record<string, number> = { NV12cpu: 1.0, I420cpu: 1.0, RGBXcpu: 0.05, NV12dec: 0.05 };
  // random sample of output pixels (fixed seed)
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  const pts: [number, number][] = [];
  for (let i = 0; i < 4000; i++) pts.push([Math.floor(rnd() * W), Math.floor(rnd() * H)]);
  const res: any = { size: [W, H], checks: {} };
  const fail: string[] = [];
  const check = (name: string, v: number, lim: number, lower = false) => {
    res.checks[name] = { value: +v.toPrecision(4), limit: lim };
    if (!(lower ? v > lim : v < lim)) fail.push(`${name}=${v} (limit ${lower ? '>' : '<'} ${lim})`);
  };
  for (const kern of ['lanczos3', 'catmullrom'] as WarpKernel[]) {
    const w = await Warper.create(device, plan, { kernel: kern, canvasPath: argFlags.has('copy') ? 'copy' : 'auto' });
    for (let k = 0; k < plan.framePts.length; k++) {
      // geometry
      const cm = await w.coordMap(k, 1);
      let gmax = 0, gsum = 0, vmis = 0;
      const refs = pts.map(([x, y]) => refSourceCoord(plan, k, x, y));
      pts.forEach(([x, y], i) => {
        const o = 3 * (y * W + x), [u, v, ok] = refs[i];
        if ((cm.data[o + 2] > 0.5) !== ok) vmis++;
        if (!ok) return;
        const d = Math.max(Math.abs(cm.data[o] - u), Math.abs(cm.data[o + 1] - v));
        gmax = Math.max(gmax, d); gsum += d;
      });
      if (kern === 'lanczos3') { check(`geom_k${k}_max_px`, gmax, 0.005); check(`geom_k${k}_valid_mismatch`, vmis, 1); res.checks[`geom_k${k}_mean_px`] = +(gsum / pts.length).toPrecision(3); }
      for (const [fmt, mk] of Object.entries(frames)) {
        const f = await mk();
        let refY = fmt === 'RGBXcpu' ? rgbLuma : Yc;
        if (fmt === 'NV12dec') {
          const raw = await frameBytes(f);
          refY = new Float64Array(W * H);
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) refY[y * W + x] = raw.buf[raw.layout[0].offset + y * raw.layout[0].stride + x];
        }
        w.resetColor();
        const luma = await w.readbackLuma(f, k);
        const ci = await w.colorInfo();
        res[`color_${fmt}`] = `${ci.mode} ${ci.transfer} (${ci.how})`;
        if (fmt === 'NV12dec' && ci.mode !== 'planar420') fail.push(`decoder frame not detected as planar (${ci.mode})`);
        let emax = 0, e2 = 0, n = 0;
        pts.forEach(([x, y], i) => {
          const [u, v, ok] = refs[i];
          const ref = ok ? refSample(refY, W, H, u, v, kern) : 16;
          const g = 16 + 219 * luma[y * W + x];
          const e = g - ref; emax = Math.max(emax, Math.abs(e)); e2 += e * e; n++;
        });
        check(`luma_${kern}_${fmt}_k${k}_max_err_codes`, emax, lumaLimit[fmt]);
        // deliverable: 8-bit canvas VideoFrame -> encoder's BT.709 conversion -> luma codes
        const out = w.warp(f, k);
        if (out.timestamp !== f.timestamp || out.duration !== f.duration) fail.push(`timestamp/duration not preserved (${fmt})`);
        const px = await frameBytes(out, 'RGBA');
        out.close();
        let o2 = 0;
        pts.forEach(([x, y], i) => {
          const [u, v, ok] = refs[i];
          const ref = ok ? refSample(refY, W, H, u, v, kern) : 16;
          const q = y * px.layout[0].stride + 4 * x;
          const Y8 = 16 + 219 * (0.2126 * px.buf[q] + 0.7152 * px.buf[q + 1] + 0.0722 * px.buf[q + 2]) / 255;
          o2 += (Y8 - ref) ** 2;
        });
        check(`out8_${kern}_${fmt}_k${k}_luma_psnr_db`, 20 * Math.log10(255 / Math.sqrt(o2 / n)), 50, true);
        f.close();
      }
    }
    const ci = await w.colorInfo();
    res.canvas = { directCanvas: ci.directCanvas, canvasFormat: ci.canvasFormat };
    w.destroy();
  }
  // Two Warpers alive at once (shared intermediates, e.g. preview + export), calls interleaved without awaiting.
  {
    const wA = await Warper.create(device, plan, { kernel: 'lanczos3', canvasPath: argFlags.has('copy') ? 'copy' : 'auto' });
    const wB = await Warper.create(device, plan, { kernel: 'catmullrom' });
    const fA = await frames.NV12dec(), fB = await frames.RGBXcpu();
    const rawA = await frameBytes(fA);
    const refA = new Float64Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) refA[y * W + x] = rawA.buf[rawA.layout[0].offset + y * rawA.layout[0].stride + x];
    const outs = [wA.warp(fA, 0), wB.warp(fB, 1), wA.warp(fA, 1), wB.warp(fB, 0)];
    const want: [Float64Array, WarpKernel, number][] = [[refA, 'lanczos3', 0], [rgbLuma, 'catmullrom', 1], [refA, 'lanczos3', 1], [rgbLuma, 'catmullrom', 0]];
    for (let i = 0; i < outs.length; i++) {
      const px = await frameBytes(outs[i], 'RGBA');
      outs[i].close();
      const [refY, kern, k] = want[i];
      let o2 = 0;
      pts.forEach(([x, y]) => {
        const [u, v, ok] = refSourceCoord(plan, k, x, y);
        const ref = ok ? refSample(refY, W, H, u, v, kern) : 16;
        const q = y * px.layout[0].stride + 4 * x;
        o2 += (16 + 219 * (0.2126 * px.buf[q] + 0.7152 * px.buf[q + 1] + 0.0722 * px.buf[q + 2]) / 255 - ref) ** 2;
      });
      check(`interleaved_${i}_luma_psnr_db`, 20 * Math.log10(255 / Math.sqrt(o2 / pts.length)), 50, true);
    }
    fA.close(); fB.close(); wA.destroy(); wB.destroy();
  }
  // Mixed colour tags on ONE Warper (the retimed-export flicker): VideoToolbox handed a few frames of a clip to the
  // Warper tagged 'iec61966-2-1' instead of 'bt709' with the same pixels, and the browser converts each frame by its
  // own tag. Warp the decoder frame (bt709) several times, then copies of its pixels tagged 'iec61966-2-1' / 'linear'
  // in between: every one must give the decoder frame's luma (one calibration for all frames gave the sRGB-tagged
  // copy the gamma-1.961 inverse on macOS: ~9 codes darker midtones).
  {
    const w = await Warper.create(device, plan, { kernel: 'lanczos3' });
    const dec = await frames.NV12dec();
    const raw = await frameBytes(dec);
    const copy = (transfer: string) => new VideoFrame(raw.buf, { format: raw.format as VideoPixelFormat, codedWidth: W, codedHeight: H,
      layout: raw.layout, timestamp: 2000, colorSpace: { primaries: 'bt709', transfer: transfer as VideoTransferCharacteristics, matrix: 'bt709', fullRange: false } });
    const ref = await w.readbackLuma(dec, 0);
    const seq: [string, () => VideoFrame][] = [['bt709', () => dec], ['bt709', () => dec], ['iec61966-2-1', () => copy('iec61966-2-1')],
      ['bt709', () => dec], ['linear', () => copy('linear')], ['iec61966-2-1', () => copy('iec61966-2-1')], ['bt709', () => dec]];
    const mixed: Record<string, number> = {};
    for (const [i, [tag, mk]] of seq.entries()) {
      const f = mk();
      const luma = await w.readbackLuma(f, 0);
      if (f !== dec) f.close();
      let emax = 0, esum = 0;
      for (let p = 0; p < luma.length; p++) { const e = (luma[p] - ref[p]) * 219; emax = Math.max(emax, Math.abs(e)); esum += e; }
      const emean = esum / luma.length;
      mixed[`${i}_${tag}`] = +emax.toPrecision(3);
      // the decoder frame itself: exact; CPU copies may reach the shader as RGBA8 (browser-converted, +-0.5 LSB per
      // channel) instead of planes: small unbiased differences — a wrong transfer shifts the MEAN by several codes
      if (f === dec) check(`mixedtags_${i}_${tag}_max_luma_diff_codes`, emax, 0.05);
      else { check(`mixedtags_${i}_${tag}_max_luma_diff_codes`, emax, 1.5); check(`mixedtags_${i}_${tag}_mean_luma_diff_codes`, Math.abs(emean), 0.1); }
    }
    const ci = await w.colorInfo();
    res.mixedTags = { maxDiff: mixed, classes: ci.classes, decodedTag: `${dec.format}|${dec.colorSpace.transfer}` };
    const cls = (k: string) => ci.classes.find(c => c.key === k);
    if (cls('NV12|iec61966-2-1')?.transfer !== 'srgb') fail.push(`sRGB-tagged frames calibrated as ${cls('NV12|iec61966-2-1')?.transfer}`);
    if (cls('NV12|linear')?.transfer !== 'linear') fail.push(`linear-tagged frames calibrated as ${cls('NV12|linear')?.transfer}`);
    dec.close(); w.destroy();
  }
  device.destroy();
  res.pass = fail.length === 0;
  if (fail.length) res.error = 'selftest failed: ' + fail.join('; ');
  return res;
}

const params = new URLSearchParams(location.search);
const mode = params.get('mode') || 'probe';
const argFlags = new Set((params.get('args') || '').split(',').filter(a => a.startsWith('+')).map(a => a.slice(1)));
const argList = (params.get('args') || '').split(',').filter(a => a && !a.startsWith('+'));
(async () => {
  try {
    if (mode === 'probe') done(await probe());
    else if (mode === 'extfit') done(await extFit(argList.length ? argList : Object.keys(FIXTURES)));
    else if (mode === 'enccolor') done(await encColor());
    else if (mode === 'copyfit') done(await copyFit(argList.length ? argList : Object.keys(FIXTURES)));
    else if (mode === 'golden') done(await golden(argList.length ? argList : Object.keys(FIXTURES)));
    else if (mode === 'bench') done(await bench(argList.length ? argList : Object.keys(FIXTURES)));
    else if (mode === 'variants') done(await variants(argList.length ? argList : Object.keys(FIXTURES)));
    else if (mode === 'selftest') done(await selftest());
    else if (SCALED_MODES.includes(mode)) done(await runScaled(mode, { FIXTURES, decodeSample, fixturePlan, gpuDevice,
      frameBytes, post, log, refSourceCoord, refSample, flags: argFlags, args: argList }));
    else throw new Error('unknown mode ' + mode);
  } catch (e) {
    log('ERROR', String(e), (e as Error).stack || '');
    done({ error: String(e) });
  }
})();
