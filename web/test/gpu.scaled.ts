// GPU module tests for scaled outputs + the Blender (owner: GPU agent). Driven through test/gpu.harness.ts modes:
//   scaledself   footage-free, pass/fail: supersampled warp + Lanczos-3 downscale vs a float64 JS reference of
//                "warp on the supersampled grid, then downscale" (luma float + 8-bit luma/chroma), several sizes /
//                aspects / forced factors, planar (decoder) and rgb (RGBX) chroma modes; scaling decisions
//   scaled       real footage (O3 fixture): 1920x1080 and 1280x720 from 4K — float luma (auto / forced integer factor
//                / single-tap 'off') + 8-bit output dumped for test/gpu.golden_scaled.py (float64 reference: warp at
//                full 4K resolution, then Lanczos-3 downscale)
//   shimmer      slowly panning synthetic high-frequency texture, 3840x2160 -> 1280x720: frame-to-frame
//                high-frequency energy variation, supersampled vs naive single-tap sampling (pass/fail on the ratio)
//   blend        Blender: single-tap exact passthrough, weight normalisation, linear-light average vs float64
//                reference (textures, RGBX VideoFrames, Warper canvas frames, > 3 taps / mixed kinds) (pass/fail)
//   benchscaled  throughput of warp() at 4K / 1080p / 720p output from a 4K O3 frame (+ single-tap 'off' for
//                comparison, + catmull-rom preview sizes) and of blend() at 1080p / 4K
//   retime       RetimeRenderer (Warper + Blender streamed by retime.ts buildSchedule): 59.94 fps synthetic clip ->
//                24/25 natural blur, 30/120 nearest, slowmo, source; every output vs float64 blend of the warped
//                taps, timestamps, warp/blend counts, ring size (pass/fail)
//   oa4scaled    real OA4 4:3 HEVC 10-bit frame (clip read through a range-request Blob shim + src/mp4.ts): 16:9
//                1920x1080 and 9:16 1080x1920 outputs vs "direct warp at 2x the output size, then Lanczos-3
//                downscale in JS" (the direct path is float64-verified by gpu.golden.py) + throughput (pass/fail)
import { Warper, warpScaling, planMinification, type WarpKernel, type WarperOptions } from '../src/gpu/warp';
import { Blender, blendRef, type BlendInput, type BlendTransfer } from '../src/gpu/blend';
import type { Plan } from '../src/types';
import { openMp4, mainVideoTrack, videoDecoderConfig } from '../src/mp4';
import { RetimeRenderer } from '../src/gpu/retime_render';
import { buildSchedule } from '../src/retime';
import type { RetimeParams } from '../src/types';

export interface HarnessCtx {
  FIXTURES: Record<string, any>;
  decodeSample(fx: any, sampleIdx: number, hw?: HardwareAcceleration): Promise<VideoFrame>;
  fixturePlan(fx: any): Plan;
  gpuDevice(features?: GPUFeatureName[]): Promise<GPUDevice>;
  frameBytes(f: VideoFrame, format?: VideoPixelFormat): Promise<{ buf: Uint8Array; layout: PlaneLayout[]; format: string | null }>;
  post(name: string, data: ArrayBufferView | ArrayBuffer | string): Promise<void>;
  log(...a: unknown[]): void;
  refSourceCoord(p: Plan, k: number, X: number, Y: number): [number, number, boolean];
  refSample(img: Float64Array, w: number, h: number, sx: number, sy: number, kern: WarpKernel, ch?: number, c?: number): number;
  flags: Set<string>;
  args: string[];
}

const KR = 0.2126, KB = 0.0722, KG = 1 - KR - KB;

/** Same plan (rotations, FOV) rendered at another output size (aspect kept when outW/outH scale alike; otherwise a
 *  centred crop / extension of the field of view at the same focal-per-width ratio). */
export function resizePlan(plan: Plan, outW: number, outH: number, fxScale = outW / plan.outW): Plan {
  return { ...plan, outW, outH, outFx: plan.outFx.map(f => f * fxScale) };
}

function lz3(x: number): number {
  const a = Math.abs(x);
  if (a < 1e-9) return 1;
  if (a >= 3) return 0;
  return 3 * Math.sin(Math.PI * a) * Math.sin(Math.PI * a / 3) / (Math.PI * Math.PI * a * a);
}
/** Normalised 1-D Lanczos-3 downscale taps (index clamped to [0, n-1]) around centre c with stretch s. */
function dtaps(c: number, s: number, n: number): [number[], number[]] {
  const i0 = Math.ceil(c - 3 * s), i1 = Math.floor(c + 3 * s);
  const idx: number[] = [], w: number[] = [];
  let sum = 0;
  for (let i = i0; i <= i1; i++) { const v = lz3((i - c) / s); idx.push(Math.min(Math.max(i, 0), n - 1)); w.push(v); sum += v; }
  return [idx, w.map(v => v / sum)];
}

function rgbToYcc8(r: number, g: number, b: number): [number, number, number] {
  const R = r / 255, G = g / 255, B = b / 255;
  const y = KR * R + KG * G + KB * B;
  return [16 + 219 * y, 128 + 224 * (B - y) / (2 * (1 - KB)), 128 + 224 * (R - y) / (2 * (1 - KR))];
}

const psnr = (mse: number, peak = 255) => mse <= 0 ? Infinity : 20 * Math.log10(peak / Math.sqrt(mse));
const r4 = (v: number) => +v.toPrecision(4);

// ================================================================================================ scaledself
async function scaledSelf(ctx: HarnessCtx) {
  const { log } = ctx;
  const device = await ctx.gpuDevice(ctx.flags.has('bgra') ? ['bgra8unorm-storage'] : []);
  const fx0 = ctx.FIXTURES.o3_0026;
  const W = 1920, H = 1080, s = 0.5;
  const base: Plan = {
    srcW: W, srcH: H, outW: W, outH: H,
    lens: { model: 'kb4', fx: fx0.lens.fx * s, fy: fx0.lens.fy * s, cx: (W - 1) / 2, cy: (H - 1) / 2, k: fx0.lens.k.slice(0, 4), width: W, height: H },
    framePts: Float64Array.from(fx0.records.map((r: any) => r.pts)), outFx: Float32Array.from(fx0.records.map((r: any) => r.outFx * s)),
    nRows: fx0.nRows, rowMats: Float32Array.from(fx0.records.flatMap((r: any) => r.rowMats)), readoutS: fx0.readoutS,
  };
  // synthetic source (codes): smooth + fine detail + edges, legal range
  const Yc = new Float64Array(W * H), Cc = new Float64Array((W / 2) * (H / 2) * 2);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const edge = ((x >> 4) + (y >> 4)) & 1 ? 14 : -14;
    Yc[y * W + x] = Math.round(122 + 38 * Math.sin(x * 0.071 + 0.4 * Math.sin(y * 0.023)) + 22 * Math.cos(y * 0.113 + x * 0.004) +
      12 * Math.sin(x * 0.93 + y * 0.61) + edge);
  }
  for (let y = 0; y < H / 2; y++) for (let x = 0; x < W / 2; x++) {
    Cc[(y * W / 2 + x) * 2] = Math.round(128 + 14 * Math.sin(x * 0.05 + y * 0.013) + 6 * Math.sin(x * 0.9));
    Cc[(y * W / 2 + x) * 2 + 1] = Math.round(128 + 12 * Math.cos(y * 0.061 - x * 0.02) + 5 * Math.cos(y * 1.1));
  }
  const cs: VideoColorSpaceInit = { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: false };
  const nv12 = new Uint8Array(W * H * 3 / 2);
  for (let i = 0; i < W * H; i++) nv12[i] = Yc[i];
  for (let i = 0; i < W * H / 4; i++) { nv12[W * H + 2 * i] = Cc[2 * i]; nv12[W * H + 2 * i + 1] = Cc[2 * i + 1]; }
  // RGBX = the same picture as full-res R'G'B' (chroma replicated); reference planes = its exact Y'CbCr (codes)
  const rgbx = new Uint8Array(W * H * 4);
  const rY = new Float64Array(W * H), rC = new Float64Array(W * H * 2);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const Yn = (Yc[y * W + x] - 16) / 219, ci = ((y >> 1) * W / 2 + (x >> 1)) * 2;
    const cb = (Cc[ci] - 128) / 224, cr = (Cc[ci + 1] - 128) / 224;
    const R = Yn + 1.5748 * cr, B = Yn + 1.8556 * cb, G = (Yn - 0.2126 * R - 0.0722 * B) / 0.7152;
    const o = (y * W + x) * 4, i = y * W + x;
    rgbx[o] = Math.round(Math.min(Math.max(R, 0), 1) * 255); rgbx[o + 1] = Math.round(Math.min(Math.max(G, 0), 1) * 255);
    rgbx[o + 2] = Math.round(Math.min(Math.max(B, 0), 1) * 255); rgbx[o + 3] = 255;
    const [yy, cbb, crr] = rgbToYcc8(rgbx[o], rgbx[o + 1], rgbx[o + 2]);
    rY[i] = yy; rC[2 * i] = cbb; rC[2 * i + 1] = crr;
  }
  // decoder frame (planar 4:2:0 import): encode the synthetic NV12 picture, decode; reference = decoded bytes
  const decFrame = async (): Promise<VideoFrame> => {
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
  };
  const fDec = await decFrame();
  const raw = await ctx.frameBytes(fDec);
  const dY = new Float64Array(W * H), dC = new Float64Array((W / 2) * (H / 2) * 2);
  {
    const L0 = raw.layout[0], L1 = raw.layout[1];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) dY[y * W + x] = raw.buf[L0.offset + y * L0.stride + x];
    for (let y = 0; y < H / 2; y++) for (let x = 0; x < W / 2; x++) {
      dC[(y * W / 2 + x) * 2] = raw.buf[L1.offset + y * L1.stride + 2 * x];
      dC[(y * W / 2 + x) * 2 + 1] = raw.buf[L1.offset + y * L1.stride + 2 * x + 1];
    }
  }
  const sources: Record<string, { mk: () => VideoFrame; Y: Float64Array; C: Float64Array; planar: boolean }> = {
    NV12dec: { mk: () => new VideoFrame(fDec, { timestamp: 1000, duration: 16683 }), Y: dY, C: dC, planar: true },
    RGBXcpu: { mk: () => new VideoFrame(rgbx, { format: 'RGBX', codedWidth: W, codedHeight: H, timestamp: 1000, duration: 16683 }), Y: rY, C: rC, planar: false },
  };

  const cases: { name: string; plan: Plan; opts?: Partial<WarperOptions>; expect: 'direct' | 'supersample' }[] = [
    { name: '960x540', plan: resizePlan(base, 960, 540), expect: 'supersample' },
    { name: '640x360', plan: resizePlan(base, 640, 360), expect: 'supersample' },
    { name: '960x540_x2', plan: resizePlan(base, 960, 540), opts: { antialias: 2 }, expect: 'supersample' },
    // 9:16 vertical: same focal as 1080-high output, narrower horizontal field (the plan agent's job; any outW/outH)
    { name: '304x540_v', plan: resizePlan(base, 304, 540, 0.5), expect: 'supersample' },
    { name: '1920x1080_direct', plan: base, expect: 'direct' },
    { name: '960x540_off', plan: resizePlan(base, 960, 540), opts: { antialias: 'off' }, expect: 'direct' },
  ];
  let seed = 4242;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  const res: any = { checks: {}, scaling: {} };
  const fail: string[] = [];
  const check = (name: string, v: number, lim: number, lower = false) => {
    res.checks[name] = { value: r4(v), limit: lim };
    if (!(lower ? v > lim : v < lim)) fail.push(`${name}=${v} (limit ${lower ? '>' : '<'} ${lim})`);
  };
  for (const cse of cases) {
    const plan = cse.plan;
    const w = await Warper.create(device, plan, { kernel: 'lanczos3', canvasPath: ctx.flags.has('copy') ? 'copy' : 'auto', ...cse.opts });
    const sc = w.scaling;
    res.scaling[cse.name] = sc;
    if (sc.mode !== cse.expect) fail.push(`${cse.name}: scaling ${sc.mode} != ${cse.expect}`);
    const oW = plan.outW, oH = plan.outH;
    const pts: [number, number][] = [];
    for (let i = 0; i < 250; i++) pts.push([Math.floor(rnd() * oW), Math.floor(rnd() * oH)]);
    // chroma blocks to check (8-bit output): block (a,b) -> pixel (2a, 2b)
    const blocks: [number, number][] = [];
    for (let i = 0; i < 120; i++) blocks.push([Math.floor(rnd() * (oW / 2)), Math.floor(rnd() * (oH / 2))]);
    for (const k of [0, 1]) {
      for (const [sname, src] of Object.entries(sources)) {
        // float64 reference of the intermediate grid (or the output grid when direct)
        const iW = sc.interW, iH = sc.interH, sx = sc.sx, sy = sc.sy;
        const lumaAt = new Map<number, number>();
        const interY = (i: number, j: number) => {
          const key = j * iW + i;
          let v = lumaAt.get(key);
          if (v === undefined) {
            const [u, vv, ok] = ctx.refSourceCoord(plan, k, (i + 0.5) / sx - 0.5, (j + 0.5) / sy - 0.5);
            v = ok ? ctx.refSample(src.Y, W, H, u, vv, 'lanczos3') : 16;
            lumaAt.set(key, v);
          }
          return v;
        };
        const chromaAt = new Map<number, [number, number]>();
        const interC = (a: number, b: number): [number, number] => {   // chroma texel (a,b) of the grid
          const key = b * iW + a;
          let v = chromaAt.get(key);
          if (!v) {
            const [u, vv, ok] = ctx.refSourceCoord(plan, k, (2 * a + 0.5) / sx - 0.5, (2 * b + 1) / sy - 0.5);
            if (!ok) v = [128, 128];
            else if (src.planar) v = [ctx.refSample(src.C, W / 2, H / 2, u * 0.5, (vv - 0.5) * 0.5, 'lanczos3', 2, 0),
              ctx.refSample(src.C, W / 2, H / 2, u * 0.5, (vv - 0.5) * 0.5, 'lanczos3', 2, 1)];
            else v = [ctx.refSample(src.C, W, H, u, vv, 'lanczos3', 2, 0), ctx.refSample(src.C, W, H, u, vv, 'lanczos3', 2, 1)];
            chromaAt.set(key, v);
          }
          return v;
        };
        const refLuma = (x: number, y: number) => {
          if (sc.mode === 'direct') return interY(x, y);
          const [ix, wx] = dtaps(sx * (x + 0.5) - 0.5, sx, iW), [iy, wy] = dtaps(sy * (y + 0.5) - 0.5, sy, iH);
          let acc = 0;
          for (let b = 0; b < iy.length; b++) { let row = 0; for (let a = 0; a < ix.length; a++) row += wx[a] * interY(ix[a], iy[b]); acc += wy[b] * row; }
          return acc;
        };
        const refChroma = (a: number, b: number): [number, number] => {
          if (sc.mode === 'direct') return interC(a, b);
          const cW = Math.ceil(iW / 2), cH = Math.ceil(iH / 2);
          const [ix, wx] = dtaps(0.5 * (sx * (2 * a + 0.5) - 0.5), sx, cW), [iy, wy] = dtaps(0.5 * (sy * (2 * b + 1) - 1), sy, cH);
          const acc: [number, number] = [0, 0];
          for (let j = 0; j < iy.length; j++) for (let i = 0; i < ix.length; i++) {
            const c = interC(ix[i], iy[j]);
            acc[0] += wy[j] * wx[i] * c[0]; acc[1] += wy[j] * wx[i] * c[1];
          }
          return acc;
        };
        const f = src.mk();
        w.resetColor();       // the colour-transfer evidence is per clip; these are two different "clips"
        const luma = await w.readbackLuma(f, k);
        let emax = 0, e2 = 0;
        for (const [x, y] of pts) {
          const e = 16 + 219 * luma[y * oW + x] - refLuma(x, y);
          emax = Math.max(emax, Math.abs(e)); e2 += e * e;
        }
        const tag = `${cse.name}_${sname}_k${k}`;
        check(`luma_${tag}_max_err_codes`, emax, 0.05);
        // deliverable: 8-bit canvas frame (timing override) -> BT.709 limited Y'CbCr codes
        const out = w.warp(f, k, { timestamp: 777, duration: 1234 });
        if (out.timestamp !== 777 || out.duration !== 1234) fail.push(`${tag}: timing override not applied`);
        if (out.displayWidth !== oW || out.displayHeight !== oH) fail.push(`${tag}: output ${out.displayWidth}x${out.displayHeight}`);
        const px = await ctx.frameBytes(out, 'RGBA');
        out.close();
        const stride = px.layout[0].stride;
        let o2 = 0, c2 = 0, cn = 0, cbl = 0;
        for (const [x, y] of pts) {
          const q = y * stride + 4 * x;
          o2 += (rgbToYcc8(px.buf[q], px.buf[q + 1], px.buf[q + 2])[0] - refLuma(x, y)) ** 2;
        }
        for (const [a, b] of blocks) {
          const [rcb, rcr] = refChroma(a, b);
          const yref = (refLuma(2 * a, 2 * b) - 16) / 219, cbn = (rcb - 128) / 224, crn = (rcr - 128) / 224;
          const R = yref + 2 * (1 - KR) * crn, B = yref + 2 * (1 - KB) * cbn, G = (yref - KR * R - KB * B) / KG;
          if (Math.min(R, G, B) < 0.002 || Math.max(R, G, B) > 0.998) { cbl++; continue; }   // clipped in 8-bit RGB
          const q = 2 * b * stride + 8 * a;
          const [, cb, cr] = rgbToYcc8(px.buf[q], px.buf[q + 1], px.buf[q + 2]);
          c2 += (cb - rcb) ** 2 + (cr - rcr) ** 2; cn += 2;
        }
        check(`out8_${tag}_luma_psnr_db`, psnr(o2 / pts.length), 50, true);
        check(`out8_${tag}_chroma_psnr_db`, psnr(c2 / Math.max(cn, 1)), 46, true);
        res.checks[`out8_${tag}_chroma_clipped_blocks`] = cbl;
        f.close();
      }
    }
    w.destroy();
  }
  // decision logic (pure)
  const m = planMinification(resizePlan(base, 640, 360));
  const s1 = warpScaling(resizePlan(base, 640, 360));
  res.decision = { minification640: r4(m), grid640: [s1.interW, s1.interH, r4(s1.sx), r4(s1.sy)],
    fullRes: warpScaling(base).mode, capped: warpScaling(resizePlan(base, 64, 36)).sx };
  if (Math.abs(s1.interW / 640 - m) > 0.01) fail.push('supersample grid does not follow the minification');
  fDec.close();
  device.destroy();
  res.pass = fail.length === 0;
  if (fail.length) res.error = 'scaledself failed: ' + fail.join('; ');
  log('scaledself', res.pass ? 'PASS' : res.error);
  return res;
}

// ================================================================================================ scaled (golden dumps)
async function scaledGolden(ctx: HarnessCtx, names: string[]) {
  const device = await ctx.gpuDevice(ctx.flags.has('bgra') ? ['bgra8unorm-storage'] : []);
  const res: any = { ua: navigator.userAgent };
  const sizes: [number, number][] = [[1920, 1080], [1280, 720]];
  for (const name of names) {
    const fx = ctx.FIXTURES[name];
    const plan = ctx.fixturePlan(fx);
    const o: any = { records: [], sizes: {} };
    res[name] = o;
    const srcs: VideoFrame[] = [];
    for (let r = 0; r < fx.records.length; r++) {
      const rec = fx.records[r];
      const f = await ctx.decodeSample(fx, rec.sample);
      srcs.push(f);
      const ro: any = { k: rec.k, sample: rec.sample, ts: f.timestamp, format: f.format, colorSpace: f.colorSpace.toJSON() };
      if (f.format) { const raw = await ctx.frameBytes(f); ro.rawLayout = raw.layout; await ctx.post(`scaled_${name}_r${r}_src.bin`, raw.buf); }
      o.records.push(ro);
    }
    for (const [oW, oH] of sizes) {
      const sp = resizePlan(plan, oW, oH);
      const factor = Math.round(plan.outW / oW);
      const variants: Record<string, Partial<WarperOptions>> = { auto: {}, [`x${factor}`]: { antialias: factor }, off: { antialias: 'off' } };
      const so: any = { plan: [sp.outW, sp.outH, sp.outFx[0]], variants: {} };
      o.sizes[`${oW}x${oH}`] = so;
      for (const [vname, vo] of Object.entries(variants)) {
        const w = await Warper.create(device, sp, { kernel: 'lanczos3', ...vo });
        so.variants[vname] = { scaling: w.scaling };
        for (let r = 0; r < srcs.length; r++) {
          const t0 = performance.now();
          const luma = await w.readbackLuma(srcs[r], r);
          so.variants[vname][`lumaMs_r${r}`] = +(performance.now() - t0).toFixed(1);
          await ctx.post(`scaled_${name}_${oW}x${oH}_${vname}_r${r}_luma.f32`, luma);
          if (vname === 'auto') {
            const v = w.warp(srcs[r], r);
            const px = await ctx.frameBytes(v, 'RGBA');
            so[`out_r${r}`] = { layout: px.layout, format: v.format, size: [v.displayWidth, v.displayHeight] };
            await ctx.post(`scaled_${name}_${oW}x${oH}_auto_r${r}_out_rgba.bin`, px.buf);
            v.close();
          }
        }
        so.variants[vname].color = (await w.colorInfo()).mode;
        w.destroy();
      }
      ctx.log(name, `${oW}x${oH}`, JSON.stringify(so.variants));
    }
    for (const f of srcs) f.close();
  }
  device.destroy();
  return res;
}

// ================================================================================================ shimmer
/** High-pass energy statistics of a sequence of float luma frames over a central window. */
function hpStats(frames: Float32Array[], W: number, H: number) {
  const x0 = Math.round(W * 0.2), x1 = Math.round(W * 0.8), y0 = Math.round(H * 0.2), y1 = Math.round(H * 0.8);
  const R = 2;   // 5x5 box
  const hps: Float32Array[] = [];
  const E: number[] = [];
  for (const f of frames) {
    // integral image (codes)
    const I = new Float64Array((W + 1) * (H + 1));
    for (let y = 0; y < H; y++) { let row = 0; for (let x = 0; x < W; x++) { row += 219 * f[y * W + x]; I[(y + 1) * (W + 1) + x + 1] = I[y * (W + 1) + x + 1] + row; } }
    const hp = new Float32Array((x1 - x0) * (y1 - y0));
    let e = 0, n = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const a = y - R, b = y + R + 1, c = x - R, d = x + R + 1;
      const box = (I[b * (W + 1) + d] - I[a * (W + 1) + d] - I[b * (W + 1) + c] + I[a * (W + 1) + c]) / 25;
      const v = 219 * f[y * W + x] - box;
      hp[n++] = v; e += v * v;
    }
    hps.push(hp); E.push(e / n);
  }
  const mean = E.reduce((a, b) => a + b, 0) / E.length;
  const sd = Math.sqrt(E.reduce((a, b) => a + (b - mean) ** 2, 0) / E.length);
  let dsum = 0;
  for (let i = 1; i < hps.length; i++) { let d = 0; const A = hps[i], B = hps[i - 1]; for (let j = 0; j < A.length; j++) d += (A[j] - B[j]) ** 2; dsum += d / A.length; }
  const dmean = dsum / (hps.length - 1);
  let dE = 0;
  for (let i = 1; i < E.length; i++) dE += Math.abs(E[i] - E[i - 1]);
  // hfEnergy*: mean HP energy per frame (code^2) and how it varies frame to frame (CV, mean |step| absolute and
  // relative); temporalHf*: per-pixel frame-to-frame change of the HP image (code^2, RMS in codes) = flicker
  return { hfEnergyMean: r4(mean), hfEnergyCV: r4(sd / mean), hfEnergyStepAbs: r4(dE / (E.length - 1)),
    hfEnergyStepRel: r4(dE / (E.length - 1) / mean), temporalHfDiffMS: r4(dmean), temporalHfFlickerRmsCodes: r4(Math.sqrt(dmean)),
    temporalHfDiffRel: r4(dmean / mean) };
}

async function shimmer(ctx: HarnessCtx) {
  const device = await ctx.gpuDevice();
  const fx0 = ctx.FIXTURES.o3_0026;
  const W = 3840, H = 2160, OW = 1280, OH = 720, N = 30;
  // texture: a legit low-frequency component (survives at 720p) + two gratings and a fine checker ABOVE the 720p
  // Nyquist (below the source Nyquist): an ideal downscale removes them, single-tap sampling aliases them into moire
  const Y = new Uint8Array(W * H * 3 / 2);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = 126 + 28 * Math.sin(2 * Math.PI * (0.011 * x + 0.008 * y)) + 22 * Math.sin(2 * Math.PI * (0.31 * x + 0.07 * y)) +
      22 * Math.sin(2 * Math.PI * (0.13 * x - 0.38 * y)) + (((x >> 1) + (y >> 1)) & 1 ? 12 : -12);
    Y[y * W + x] = Math.round(Math.min(Math.max(v, 16), 235));
  }
  Y.fill(128, W * H);
  const cs: VideoColorSpaceInit = { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: false };
  const f = new VideoFrame(Y, { format: 'NV12', codedWidth: W, codedHeight: H, timestamp: 0, duration: 16683, colorSpace: cs });
  // slow pan: yaw 0.3 source px per frame at the centre (+ a slight tilt), no rolling shutter
  const lens = { model: 'kb4' as const, fx: fx0.lens.fx, fy: fx0.lens.fy, cx: (W - 1) / 2, cy: (H - 1) / 2, k: fx0.lens.k.slice(0, 4) as [number, number, number, number], width: W, height: H };
  const nRows = 2;
  const rowMats = new Float32Array(N * nRows * 9);
  for (let k = 0; k < N; k++) {
    const a = k * 0.3 / lens.fx, b = k * 0.07 / lens.fx;
    const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
    // R = Ry(a) * Rx(b)
    const M = [ca, sa * sb, sa * cb, 0, cb, -sb, -sa, ca * sb, ca * cb];
    for (let j = 0; j < nRows; j++) rowMats.set(M, (k * nRows + j) * 9);
  }
  const outFx = 1756.41 * OW / W;   // the O3 fixture's field of view at 1280 wide
  const plan: Plan = { srcW: W, srcH: H, outW: OW, outH: OH, lens, framePts: Float64Array.from({ length: N }, (_, k) => k / 60),
    outFx: new Float32Array(N).fill(outFx), nRows, rowMats, readoutS: 0 };
  const variants: Record<string, WarperOptions> = {
    supersampled: { kernel: 'lanczos3' },
    single_lanczos3: { kernel: 'lanczos3', antialias: 'off' },
    single_bilinear: { kernel: 'bilinear', antialias: 'off' },
  };
  const res: any = { src: [W, H], out: [OW, OH], frames: N, panPxPerFrame: 0.3, minification: r4(planMinification(plan)) };
  for (const [name, o] of Object.entries(variants)) {
    const w = await Warper.create(device, plan, o);
    const frames: Float32Array[] = [];
    for (let k = 0; k < N; k++) frames.push(await w.readbackLuma(f, k));
    res[name] = { ...hpStats(frames, OW, OH), scaling: w.scaling.mode };
    if (name === 'supersampled') await ctx.post('shimmer_ss_k0.f32', frames[0]);
    if (name === 'single_lanczos3') await ctx.post('shimmer_naive_k0.f32', frames[0]);
    ctx.log('shimmer', name, JSON.stringify(res[name]));
    w.destroy();
  }
  const ss = res.supersampled, nv = res.single_lanczos3;
  res.ratio = {
    temporalHfDiff_naive_over_ss: r4(nv.temporalHfDiffMS / ss.temporalHfDiffMS),
    hfEnergy_naive_over_ss: r4(nv.hfEnergyMean / ss.hfEnergyMean),
    hfEnergyStepAbs_naive_over_ss: r4(nv.hfEnergyStepAbs / ss.hfEnergyStepAbs),
    temporalHfDiff_bilinear_over_ss: r4(res.single_bilinear.temporalHfDiffMS / ss.temporalHfDiffMS),
  };
  f.close();
  device.destroy();
  const fail: string[] = [];
  if (!(res.ratio.temporalHfDiff_naive_over_ss > 10)) fail.push(`temporal HF difference ratio ${res.ratio.temporalHfDiff_naive_over_ss} <= 10`);
  if (!(ss.hfEnergyCV < 0.02)) fail.push(`supersampled HF energy CV ${ss.hfEnergyCV} >= 0.02`);
  res.pass = fail.length === 0;
  if (fail.length) res.error = 'shimmer failed: ' + fail.join('; ');
  return res;
}

// ================================================================================================ blend
async function blendTest(ctx: HarnessCtx) {
  const device = await ctx.gpuDevice(ctx.flags.has('bgra') ? ['bgra8unorm-storage'] : []);
  const res: any = { checks: {} };
  const fail: string[] = [];
  const check = (name: string, v: number, lim: number, lower = false) => {
    res.checks[name] = { value: r4(v), limit: lim };
    if (!(lower ? v > lim : v <= lim)) fail.push(`${name}=${v} (limit ${lower ? '>' : '<='} ${lim})`);
  };
  const W = 640, H = 360;
  // synthetic RGBA pictures covering the full code range (gradients, patches, noise)
  let seed = 99;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  const pic = (kind: number) => {
    const u = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4;
      if (kind === 0) { u[o] = x * 255 / (W - 1); u[o + 1] = y * 255 / (H - 1); u[o + 2] = ((x >> 5) + (y >> 5)) & 1 ? 255 : 0; }
      else if (kind === 1) { u[o] = 255 - x * 255 / (W - 1); u[o + 1] = rnd() * 256; u[o + 2] = ((x >> 3) & 1) ? 230 : 20; }
      else { u[o] = rnd() * 256; u[o + 1] = rnd() * 256; u[o + 2] = rnd() * 256; }
      u[o + 3] = 255;
    }
    return u;
  };
  const pics = [pic(0), pic(1), pic(2), pic(2), pic(2)];
  const texOf = (u: Uint8Array<ArrayBuffer>) => {
    const t = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: t }, u, { bytesPerRow: W * 4 }, [W, H]);
    return t;
  };
  const texs = pics.map(texOf);
  const frameOf = (u: Uint8Array<ArrayBuffer>) => new VideoFrame(u, { format: 'RGBX', codedWidth: W, codedHeight: H, timestamp: 0 });
  const vfs = pics.map(frameOf);
  const bytes = async (v: VideoFrame) => { const p = await ctx.frameBytes(v, 'RGBA'); return { buf: p.buf, stride: p.layout[0].stride }; };
  const maxDiff = (a: { buf: Uint8Array; stride: number }, ref: (x: number, y: number, c: number) => number) => {
    let m = 0, s = 0, n = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) {
      const d = Math.abs(a.buf[y * a.stride + 4 * x + c] - ref(x, y, c)); m = Math.max(m, d); s += d; n++;
    }
    return { max: m, mean: s / n };
  };
  const src = (i: number) => (x: number, y: number, c: number) => pics[i][(y * W + x) * 4 + c];
  const canvasPath = ctx.flags.has('copy') ? 'copy' as const : 'auto' as const;
  const blender = await Blender.create(device, W, H, { transfer: 'bt709', canvasPath });
  const gpuPass = await Blender.create(device, W, H, { transfer: 'bt709', passthrough: 'gpu', canvasPath });
  res.canvas = { width: blender.width, height: blender.height };

  // (1) single tap: exact passthrough (clone path, GPU path for VideoFrames, textures)
  for (const [nm, b, input] of [['clone_frame', blender, vfs[0]], ['gpu_frame', gpuPass, vfs[0]], ['gpu_texture', blender, texs[0]]] as [string, Blender, BlendInput][]) {
    const v = b.blend([input], [1], { timestamp: 5000, duration: 16683 });
    if (v.timestamp !== 5000 || v.duration !== 16683) fail.push(`${nm}: timing`);
    check(`passthrough_${nm}_max_code_diff`, maxDiff(await bytes(v), src(0)).max, 0);
    v.close();
  }
  // weight 0 taps are dropped (still a passthrough)
  { const v = gpuPass.blend([vfs[0], vfs[1]], [1, 0], { timestamp: 0 }); check('passthrough_zero_weight_max_code_diff', maxDiff(await bytes(v), src(0)).max, 0); v.close(); }

  // (2) weights: same picture twice with weights summing to 1 -> itself (linear round trip within 1 code);
  //     unnormalised weights == normalised
  {
    const a = blender.blend([texs[0], vfs[0]], [0.3, 0.7], { timestamp: 0 });
    check('same_picture_0.3_0.7_max_code_diff', maxDiff(await bytes(a), src(0)).max, 1);
    const b1 = blender.blend([texs[1], texs[2]], [0.25, 0.75], { timestamp: 0 });
    const b2 = blender.blend([texs[1], texs[2]], [2.5, 7.5], { timestamp: 0 });
    const B1 = await bytes(b1), B2 = await bytes(b2);
    check('unnormalised_weights_max_code_diff', maxDiff(B1, (x, y, c) => B2.buf[y * B2.stride + 4 * x + c]).max, 0);
    a.close(); b1.close(); b2.close();
  }
  // (3) linear-light average vs float64 reference; and how far a code-value average would be
  const cases: { name: string; inputs: BlendInput[]; idx: number[]; w: number[]; tr?: BlendTransfer }[] = [
    { name: 'tex_2tap_half', inputs: [texs[0], texs[1]], idx: [0, 1], w: [0.5, 0.5] },
    { name: 'frame_2tap_25_75', inputs: [vfs[0], vfs[1]], idx: [0, 1], w: [0.25, 0.75] },
    { name: 'mixed_5tap', inputs: [vfs[0], texs[1], vfs[2], texs[3], vfs[4]], idx: [0, 1, 2, 3, 4], w: [0.1, 0.3, 0.2, 0.15, 0.25] },
    { name: 'tex_3tap_srgb', inputs: [texs[0], texs[1], texs[2]], idx: [0, 1, 2], w: [0.2, 0.5, 0.3], tr: 'srgb' },
  ];
  for (const c of cases) {
    blender.transfer = c.tr ?? 'bt709';
    const v = blender.blend(c.inputs, c.w, { timestamp: 0 });
    const B = await bytes(v);
    v.close();
    const ref = (x: number, y: number, ch: number) =>
      Math.round(255 * blendRef(c.idx.map(i => pics[i][(y * W + x) * 4 + ch] / 255), c.w, c.tr ?? 'bt709'));
    const d = maxDiff(B, ref);
    check(`linear_${c.name}_max_code_diff`, d.max, 1);
    res.checks[`linear_${c.name}_mean_code_diff`] = r4(d.mean);
    const naive = maxDiff(B, (x, y, ch) => c.idx.reduce((a, i, j) => a + c.w[j] * pics[i][(y * W + x) * 4 + ch], 0));
    res.checks[`linear_${c.name}_vs_code_value_average`] = { max: r4(naive.max), mean: r4(naive.mean) };
  }
  blender.transfer = 'bt709';
  // black/white 50:50 -> BT.709 OETF(0.5) = 0.7055 -> code 180 (a code-value average would give 128)
  {
    const bw = [new Uint8Array(W * H * 4).fill(0), new Uint8Array(W * H * 4).fill(255)];
    const tb = bw.map(texOf);
    const v = blender.blend(tb, [0.5, 0.5], { timestamp: 0 });
    const B = await bytes(v);
    res.checks.black_white_half = { got: B.buf[0], expect: Math.round(255 * blendRef([0, 1], [0.5, 0.5])) };
    if (B.buf[0] !== res.checks.black_white_half.expect) fail.push('black/white 50:50');
    v.close(); tb.forEach(t => t.destroy());
  }
  // (4) Warper canvas frames (the real inputs) + warpTo() targets: blend of two warped frames vs the reference
  {
    const fx0 = ctx.FIXTURES.o3_0026;
    const SW = 1920, SH = 1080;
    const plan: Plan = {
      srcW: SW, srcH: SH, outW: W, outH: H,
      lens: { model: 'kb4', fx: fx0.lens.fx / 2, fy: fx0.lens.fy / 2, cx: (SW - 1) / 2, cy: (SH - 1) / 2, k: fx0.lens.k.slice(0, 4), width: SW, height: SH },
      framePts: Float64Array.from(fx0.records.map((r: any) => r.pts)), outFx: Float32Array.from(fx0.records.map((r: any) => r.outFx / 2 * W / SW)),
      nRows: fx0.nRows, rowMats: Float32Array.from(fx0.records.flatMap((r: any) => r.rowMats)), readoutS: fx0.readoutS,
    };
    const srcPix = new Uint8Array(SW * SH * 4);
    for (let i = 0; i < SW * SH; i++) { const x = i % SW, y = (i / SW) | 0; srcPix[4 * i] = (x * 7 + y * 3) & 255; srcPix[4 * i + 1] = (x ^ y) & 255; srcPix[4 * i + 2] = (y * 5) & 255; srcPix[4 * i + 3] = 255; }
    const sf = new VideoFrame(srcPix, { format: 'RGBX', codedWidth: SW, codedHeight: SH, timestamp: 0 });
    const w = await Warper.create(device, plan, { kernel: 'lanczos3' });
    res.warpScaling = w.scaling.mode;
    const a = w.warp(sf, 0), b = w.warp(sf, 1);
    const A = await bytes(a), B = await bytes(b);
    const v = blender.blend([a, b], [0.4, 0.6], { timestamp: 1 });
    const V = await bytes(v);
    const ref = (x: number, y: number, c: number) => Math.round(255 * blendRef([A.buf[y * A.stride + 4 * x + c] / 255, B.buf[y * B.stride + 4 * x + c] / 255], [0.4, 0.6]));
    check('warped_frames_linear_max_code_diff', maxDiff(V, ref).max, 1);
    const t0 = w.createTarget(), t1 = w.createTarget();
    w.warpTo(sf, 0, t0); w.warpTo(sf, 1, t1);
    const v1 = blender.blend([t0], [1], { timestamp: 2 });
    check('warpTo_target_equals_warp_max_code_diff', maxDiff(await bytes(v1), (x, y, c) => A.buf[y * A.stride + 4 * x + c]).max, 0);
    const v2 = blender.blend([t0, t1], [0.4, 0.6], { timestamp: 3 });
    check('warpTo_targets_blend_equals_frames_blend_max_code_diff', maxDiff(await bytes(v2), (x, y, c) => V.buf[y * V.stride + 4 * x + c]).max, 0);
    for (const x of [a, b, v, v1, v2, sf]) x.close();
    t0.destroy(); t1.destroy(); w.destroy();
  }
  // errors
  try { blender.blend([texs[0]], [0], { timestamp: 0 }); fail.push('zero weights accepted'); } catch { /* expected */ }
  try { blender.blend([texs[0], texs[1]], [1], { timestamp: 0 }); fail.push('length mismatch accepted'); } catch { /* expected */ }
  vfs.forEach(v => v.close()); texs.forEach(t => t.destroy());
  blender.destroy(); gpuPass.destroy();
  device.destroy();
  res.pass = fail.length === 0;
  if (fail.length) res.error = 'blend failed: ' + fail.join('; ');
  ctx.log('blend', res.pass ? 'PASS' : res.error);
  return res;
}

// ================================================================================================ benchscaled
async function benchScaled(ctx: HarnessCtx, names: string[]) {
  const device = await ctx.gpuDevice(['timestamp-query', ...(ctx.flags.has('bgra') ? ['bgra8unorm-storage' as GPUFeatureName] : [])]);
  const res: any = {};
  const med = (a: number[]) => { const b = [...a].sort((x, y) => x - y); return +b[b.length >> 1].toFixed(2); };
  for (const name of names) {
    const fx = ctx.FIXTURES[name];
    const plan = ctx.fixturePlan(fx);
    const o: any = {};
    res[name] = o;
    const f = await ctx.decodeSample(fx, fx.records[0].sample);
    o.src = [f.displayWidth, f.displayHeight, f.format];
    const R = plan.framePts.length;
    const cfgs: [string, Plan, WarperOptions][] = [
      [`${plan.outW}x${plan.outH}_lanczos3`, plan, { kernel: 'lanczos3' }],
      ['1920x1080_lanczos3', resizePlan(plan, 1920, 1080), { kernel: 'lanczos3' }],
      ['1920x1080_lanczos3_off', resizePlan(plan, 1920, 1080), { kernel: 'lanczos3', antialias: 'off' }],
      ['1280x720_lanczos3', resizePlan(plan, 1280, 720), { kernel: 'lanczos3' }],
      ['1280x720_lanczos3_off', resizePlan(plan, 1280, 720), { kernel: 'lanczos3', antialias: 'off' }],
      ['960x540_catmullrom', resizePlan(plan, 960, 540), { kernel: 'catmullrom' }],
      ['960x540_catmullrom_off', resizePlan(plan, 960, 540), { kernel: 'catmullrom', antialias: 'off' }],
    ];
    for (const [cname, p, opts] of cfgs) {
      const w = await Warper.create(device, p, { ...opts, profile: true });
      for (let i = 0; i < 20; i++) w.warp(f, i % R).close();
      await device.queue.onSubmittedWorkDone();
      const N = 240;
      const t0 = performance.now();
      for (let i = 0; i < N; i++) { w.warp(f, i % R).close(); if (i % 8 === 7) await device.queue.onSubmittedWorkDone(); }
      await device.queue.onSubmittedWorkDone();
      const dt = performance.now() - t0;
      const conv: number[] = [], wp: number[] = [];
      for (let i = 0; i < 31; i++) { w.render(f, i % R); const q = (await w.profile())!; conv.push(q.convertMs); wp.push(q.warpMs); }
      o[cname] = { scaling: `${w.scaling.mode} ${w.scaling.interW}x${w.scaling.interH}`, fps: +(N / dt * 1000).toFixed(1),
        msPerFrame: +(dt / N).toFixed(2), gpuConvertMsMedian: med(conv), gpuWarpMsMedian: med(wp) };
      ctx.log(name, cname, JSON.stringify(o[cname]));
      w.destroy();
    }
    // blend throughput: 2 taps (canvas VideoFrames) and 3 taps (warpTo textures) at 1080p and 4K
    for (const [bw, bh] of [[1920, 1080], [plan.outW, plan.outH]]) {
      const w = await Warper.create(device, resizePlan(plan, bw, bh), { kernel: 'lanczos3' });
      const b = await Blender.create(device, bw, bh);
      const a0 = w.warp(f, 0), a1 = w.warp(f, R > 1 ? 1 : 0);
      const ts = [w.createTarget(), w.createTarget(), w.createTarget()];
      ts.forEach((t, i) => w.warpTo(f, i % R, t));
      await device.queue.onSubmittedWorkDone();
      for (const [bn, inputs] of [['2tap_frames', [a0, a1]], ['3tap_textures', ts]] as [string, BlendInput[]][]) {
        const wts = inputs.map((_, i) => i + 1);
        for (let i = 0; i < 10; i++) b.blend(inputs, wts, { timestamp: i }).close();
        await device.queue.onSubmittedWorkDone();
        const N = 200;
        const t0 = performance.now();
        for (let i = 0; i < N; i++) { b.blend(inputs, wts, { timestamp: i }).close(); if (i % 8 === 7) await device.queue.onSubmittedWorkDone(); }
        await device.queue.onSubmittedWorkDone();
        const dt = performance.now() - t0;
        o[`blend_${bw}x${bh}_${bn}`] = { fps: +(N / dt * 1000).toFixed(1), msPerFrame: +(dt / N).toFixed(3) };
        ctx.log(name, `blend ${bw}x${bh} ${bn}`, JSON.stringify(o[`blend_${bw}x${bh}_${bn}`]));
      }
      a0.close(); a1.close(); ts.forEach(t => t.destroy()); b.destroy(); w.destroy();
    }
    f.close();
  }
  device.destroy();
  return res;
}

// ================================================================================================ oa4scaled
/** A read-only Blob look-alike over HTTP range requests (only what src/mp4.ts uses: size + slice().arrayBuffer()). */
async function rangeBlob(url: string): Promise<Blob> {
  const r = await fetch(url, { headers: { Range: 'bytes=0-0' } });
  const m = /\/(\d+)$/.exec(r.headers.get('Content-Range') || '');
  if (r.status !== 206 || !m) throw new Error(`range probe ${url}: HTTP ${r.status}`);
  const size = +m[1];
  const mk = (s: number, e: number): any => ({
    size: e - s,
    slice: (a = 0, b = e - s) => mk(s + a, s + Math.min(b, e - s)),
    arrayBuffer: async () => {
      if (e <= s) return new ArrayBuffer(0);
      const q = await fetch(url, { headers: { Range: `bytes=${s}-${e - 1}` } });
      if (q.status !== 206) throw new Error(`range ${s}-${e}: HTTP ${q.status}`);
      return q.arrayBuffer();
    },
  });
  return mk(0, size) as Blob;
}

/** Separable Lanczos-3 downscale of a float image by an integer factor f (pixel-centre convention, clamp). */
function downscaleF(img: Float32Array, W: number, H: number, f: number): Float32Array {
  const oW = W / f, oH = H / f;
  const tapsX = Array.from({ length: oW }, (_, x) => dtaps(f * (x + 0.5) - 0.5, f, W));
  const tapsY = Array.from({ length: oH }, (_, y) => dtaps(f * (y + 0.5) - 0.5, f, H));
  const tmp = new Float32Array(oW * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < oW; x++) {
    const [ix, wx] = tapsX[x]; let a = 0;
    for (let t = 0; t < ix.length; t++) a += wx[t] * img[y * W + ix[t]];
    tmp[y * oW + x] = a;
  }
  const out = new Float32Array(oW * oH);
  for (let y = 0; y < oH; y++) {
    const [iy, wy] = tapsY[y];
    for (let x = 0; x < oW; x++) { let a = 0; for (let t = 0; t < iy.length; t++) a += wy[t] * tmp[iy[t] * oW + x]; out[y * oW + x] = a; }
  }
  return out;
}

async function oa4Scaled(ctx: HarnessCtx) {
  const clip = ctx.args[0] || 'DJI_20260927091931_0012_D.MP4';
  const device = await ctx.gpuDevice(['timestamp-query']);
  const blob = await rangeBlob(`/__clips/${encodeURIComponent(clip)}`);
  const info = await openMp4(blob);
  const vt = mainVideoTrack(info);
  const cfg = videoDecoderConfig(vt);
  const key = Array.from(vt.sync).findIndex((v, i) => v === 1 && i > 0) ;
  const k0 = key > 0 ? key : 0;
  const data = new Uint8Array(await blob.slice(vt.offsets[k0], vt.offsets[k0] + vt.sizes[k0]).arrayBuffer());
  let frame: VideoFrame | null = null;
  let err: unknown = null;
  const dec = new VideoDecoder({ output: f => { if (!frame) frame = f; else f.close(); }, error: e => { err = e; } });
  dec.configure({ ...cfg, hardwareAcceleration: 'no-preference' });
  dec.decode(new EncodedVideoChunk({ type: 'key', timestamp: Math.round(vt.cts[k0] * 1e6), data }));
  await dec.flush(); dec.close();
  if (err || !frame) throw err ?? new Error('no frame decoded');
  const f = frame as VideoFrame;
  const SW = f.displayWidth, SH = f.displayHeight;
  const res: any = { clip, codec: cfg.codec, sample: k0, src: [SW, SH, f.format], colorSpace: f.colorSpace.toJSON(), checks: {} };
  const fail: string[] = [];
  const check = (name: string, v: number, lim: number) => { res.checks[name] = { value: r4(v), limit: lim }; if (!(v > lim)) fail.push(`${name}=${v} (limit > ${lim})`); };
  // plan: OA4 intrinsics + rotations of the oa4_0005 fixture (same camera model), centred on this frame's size
  const fx = ctx.FIXTURES.oa4_0005;
  const L = fx.lens, sc = SW / fx.srcW;
  const base: Plan = {
    srcW: SW, srcH: SH, outW: SW, outH: SH,
    lens: { model: 'kb4', fx: L.fx * sc, fy: L.fy * sc, cx: (SW - 1) / 2, cy: (SH - 1) / 2, k: L.k.slice(0, 4), width: SW, height: SH },
    framePts: Float64Array.from(fx.records.map((r: any) => r.pts)), outFx: Float32Array.from(fx.records.map((r: any) => r.outFx * sc)),
    nRows: fx.nRows, rowMats: Float32Array.from(fx.records.flatMap((r: any) => r.rowMats)), readoutS: fx.readoutS,
  };
  const fx4 = base.outFx[0];                 // the fixture's 4:3 output focal (at the source width)
  const vfov = 2 * Math.atan((SH / 2) / fx4);
  const outputs: [string, number, number, number][] = [
    ['16:9_1920x1080', 1920, 1080, fx4 * 1920 / SW],                      // same horizontal FOV, 16:9 crop
    ['9:16_1080x1920', 1080, 1920, 960 / Math.tan(vfov / 2)],             // same vertical FOV, vertical crop
  ];
  for (const [nm, oW, oH, ofx] of outputs) {
    const p = { ...base, outW: oW, outH: oH, outFx: new Float32Array(base.outFx.length).fill(ofx) };
    const big = { ...p, outW: 2 * oW, outH: 2 * oH, outFx: new Float32Array(p.outFx.length).fill(2 * ofx) };
    const wBig = await Warper.create(device, big, { kernel: 'lanczos3' });
    const ref = downscaleF(await wBig.readbackLuma(f, 0), 2 * oW, 2 * oH, 2);
    const o: any = { outFx: r4(ofx), bigScaling: wBig.scaling.mode, color: (await wBig.colorInfo()).mode };
    wBig.destroy();
    // 'x': supersampled on exactly the reference's 2x grid (same pipeline -> ~exact); 'auto': the shipped grid
    for (const [vn, vo] of [['auto', {}], ['x', { antialias: 2 }], ['off', { antialias: 'off' }]] as [string, Partial<WarperOptions>][]) {
      const w = await Warper.create(device, p, { kernel: 'lanczos3', ...vo });
      const g = await w.readbackLuma(f, 0);
      let e2 = 0, emax = 0;
      for (let i = 0; i < g.length; i++) { const e = 219 * (g[i] - ref[i]); e2 += e * e; emax = Math.max(emax, Math.abs(e)); }
      o[vn] = { scaling: `${w.scaling.mode} ${w.scaling.interW}x${w.scaling.interH}`, psnrDb: r4(psnr(e2 / g.length)), maxErrCodes: r4(emax) };
      if (vn === 'auto') {
        const N = 120;
        for (let i = 0; i < 10; i++) w.warp(f, 0).close();
        await device.queue.onSubmittedWorkDone();
        const t0 = performance.now();
        for (let i = 0; i < N; i++) { w.warp(f, i & 1).close(); if (i % 8 === 7) await device.queue.onSubmittedWorkDone(); }
        await device.queue.onSubmittedWorkDone();
        o.auto.fps = +(N / (performance.now() - t0) * 1000).toFixed(1);
      }
      w.destroy();
    }
    res[nm] = o;
    check(`${nm}_x2_vs_ref_psnr_db`, o.x.psnrDb, 60);
    check(`${nm}_auto_vs_ref_psnr_db`, o.auto.psnrDb, 40);
    ctx.log('oa4', nm, JSON.stringify(o));
  }
  f.close();
  device.destroy();
  res.pass = fail.length === 0;
  if (fail.length) res.error = 'oa4scaled failed: ' + fail.join('; ');
  return res;
}

// ================================================================================================ retime
async function retimeTest(ctx: HarnessCtx) {
  const device = await ctx.gpuDevice(ctx.flags.has('bgra') ? ['bgra8unorm-storage'] : []);
  const fx0 = ctx.FIXTURES.o3_0026;
  const SW = 1280, SH = 720, OW = 640, OH = 360, N = 30, FPS = 60000 / 1001;
  const sc = SW / fx0.srcW;
  // N records: the fixture's two rotations alternating, plus a slow yaw (so consecutive frames differ)
  const rowMats = new Float32Array(N * fx0.nRows * 9);
  for (let k = 0; k < N; k++) {
    const a = k * 0.004, ca = Math.cos(a), sa = Math.sin(a);
    const R = fx0.records[k % 2].rowMats as number[];
    for (let j = 0; j < fx0.nRows; j++) {
      const m = R.slice(9 * j, 9 * j + 9);
      // M' = Ry(a) * M
      const o = (k * fx0.nRows + j) * 9;
      for (let c = 0; c < 3; c++) {
        rowMats[o + c] = ca * m[c] + sa * m[6 + c];
        rowMats[o + 3 + c] = m[3 + c];
        rowMats[o + 6 + c] = -sa * m[c] + ca * m[6 + c];
      }
    }
  }
  const plan: Plan = {
    srcW: SW, srcH: SH, outW: OW, outH: OH,
    lens: { model: 'kb4', fx: fx0.lens.fx * sc, fy: fx0.lens.fy * sc, cx: (SW - 1) / 2, cy: (SH - 1) / 2, k: fx0.lens.k.slice(0, 4), width: SW, height: SH },
    framePts: Float64Array.from({ length: N }, (_, k) => k / FPS), outFx: new Float32Array(N).fill(fx0.records[0].outFx * sc * OW / SW),
    nRows: fx0.nRows, rowMats, readoutS: fx0.readoutS,
  };
  // source frames: texture + a bright bar moving 9 px/frame
  const frames: VideoFrame[] = [];
  for (let k = 0; k < N; k++) {
    const u = new Uint8Array(SW * SH * 4);
    for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
      const o = (y * SW + x) * 4, bar = Math.abs(x - (200 + 9 * k)) < 40;
      u[o] = bar ? 250 : (x * 3 + y) & 255; u[o + 1] = bar ? 245 : ((x >> 3) ^ (y >> 3)) * 9 & 255; u[o + 2] = bar ? 240 : (y * 2) & 255; u[o + 3] = 255;
    }
    frames.push(new VideoFrame(u, { format: 'RGBX', codedWidth: SW, codedHeight: SH, timestamp: Math.round(plan.framePts[k] * 1e6), duration: 16683 }));
  }
  const warper = await Warper.create(device, plan, { kernel: 'lanczos3' });
  // reference: every source frame warped on its own (8-bit RGBA)
  const warped: Uint8Array[] = [];
  let stride = 0;
  for (let k = 0; k < N; k++) {
    const v = warper.warp(frames[k], k);
    const p = await ctx.frameBytes(v, 'RGBA'); stride = p.layout[0].stride; warped.push(p.buf); v.close();
  }
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  const samp = Array.from({ length: 4000 }, () => [Math.floor(rnd() * OW), Math.floor(rnd() * OH)]);
  const configs: Record<string, RetimeParams> = {
    '24_natural': { fps: 24, timing: 'realtime', motionBlur: 'natural' },
    '25_natural_360': { fps: 25, timing: 'realtime', motionBlur: 'natural', shutterDeg: 360 },
    '30_off': { fps: 29.97, timing: 'realtime', motionBlur: 'off' },
    '120_off': { fps: 120, timing: 'realtime', motionBlur: 'off' },
    '24_slowmo': { fps: 24, timing: 'slowmo', motionBlur: 'off' },
    'source': { fps: 'source', timing: 'realtime', motionBlur: 'off' },
  };
  const res: any = { src: [SW, SH, N, +FPS.toFixed(3)], out: [OW, OH], scaling: warper.scaling.mode, runs: {} };
  const fail: string[] = [];
  for (const [name, rp] of Object.entries(configs)) {
    const sch = buildSchedule(plan.framePts, rp);
    const rr = await RetimeRenderer.create(warper, sch, { transfer: 'bt709' });
    const outs: VideoFrame[] = [];
    for (let k = 0; k < N; k++) outs.push(...rr.push(frames[k], k));
    const o: any = { n: sch.n, emitted: outs.length, done: rr.done, maxTaps: sch.maxTaps, maxSpan: sch.maxSpan, stats: { ...rr.stats } };
    let maxd = 0, single0 = 0, tsBad = 0;
    for (let i = 0; i < outs.length; i++) {
      const v = outs[i];
      if (v.timestamp !== Math.round(sch.outPtsUs[i])) tsBad++;
      const p = await ctx.frameBytes(v, 'RGBA');
      v.close();
      const a = sch.tapStart[i], c = sch.tapsPerFrame[i];
      for (const [x, y] of samp) for (let ch = 0; ch < 3; ch++) {
        const q = y * stride + 4 * x + ch;
        let ref: number;
        if (c === 1) ref = warped[sch.taps[a]][q];
        else {
          const vals: number[] = [], w: number[] = [];
          for (let j = a; j < a + c; j++) { vals.push(warped[sch.taps[j]][q] / 255); w.push(sch.weights[j]); }
          ref = Math.round(255 * blendRef(vals, w, 'bt709'));
        }
        const d = Math.abs(p.buf[y * p.layout[0].stride + 4 * x + ch] - ref);
        maxd = Math.max(maxd, d);
        if (c === 1) single0 = Math.max(single0, d);
      }
    }
    o.maxCodeDiff = maxd; o.singleTapMaxCodeDiff = single0; o.badTimestamps = tsBad;
    res.runs[name] = o;
    ctx.log('retime', name, JSON.stringify(o));
    if (outs.length !== sch.n || !rr.done) fail.push(`${name}: emitted ${outs.length}/${sch.n}`);
    if (maxd > 1 || single0 > 0) fail.push(`${name}: pixel diff ${maxd} (single-tap ${single0})`);
    if (tsBad) fail.push(`${name}: ${tsBad} timestamps`);
    if (sch.maxSpan != null && rr.stats.peakRing > sch.maxSpan + 1) fail.push(`${name}: ring ${rr.stats.peakRing} > maxSpan+1`);
    rr.destroy();
  }
  frames.forEach(f => f.close());
  warper.destroy();
  device.destroy();
  res.pass = fail.length === 0;
  if (fail.length) res.error = 'retime failed: ' + fail.join('; ');
  return res;
}

export async function runScaled(mode: string, ctx: HarnessCtx): Promise<unknown> {
  const names = ctx.args.length ? ctx.args : ['o3_0026'];
  switch (mode) {
    case 'scaledself': return scaledSelf(ctx);
    case 'scaled': return scaledGolden(ctx, names);
    case 'shimmer': return shimmer(ctx);
    case 'blend': return blendTest(ctx);
    case 'benchscaled': return benchScaled(ctx, names);
    case 'oa4scaled': return oa4Scaled(ctx);
    case 'retime': return retimeTest(ctx);
    case 'shutter': return (await import('./gpu.shutter')).shutterTest(ctx);
    default: throw new Error('unknown mode ' + mode);
  }
}
export const SCALED_MODES = ['scaledself', 'scaled', 'shimmer', 'blend', 'benchscaled', 'oa4scaled', 'retime', 'shutter'];
