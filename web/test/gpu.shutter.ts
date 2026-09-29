// GPU check of the gyro sub-frame synthetic shutter (src/gpu/shutter_render.ts). Headless: node test/gpu.run.mjs shutter
//
// A static textured source frame seen by a virtual camera that yaws at a constant rate (plan rows M_k = M0·Ry(a·k),
// virtual path Ry(a·k)) — the image motion is exactly the virtual rotation, so the physically correct 180° motion blur
// of an output frame is known: the linear-light average of the views over the exposure window.
//   direction   frame k re-warped at frame k+1's instant (subframeRows) == warp of record k+1 (<= 1 code), and
//               clearly != the warp of record k (the motion is visible)
//   streak      ShutterRenderer output vs a dense reference (96 views per window, linear-light mean) at 24 fps (2-3
//               taps per output) and 30 fps (one tap: a frame blend cannot blur it): mean |diff| < 1 code and well
//               below the plain frame blend's (RetimeRenderer: 2-3 discrete copies / none), every output emitted,
//               timestamps = the schedule's
//   reuse       two sub-frame warps of one frame with the frame's own rows == warp() (the skipped conversion is exact)
import { Warper } from '../src/gpu/warp';
import { blendRef } from '../src/gpu/blend';
import { RetimeRenderer } from '../src/gpu/retime_render';
import { ShutterRenderer, subframeRows } from '../src/gpu/shutter_render';
import { buildSchedule } from '../src/retime';
import type { Plan } from '../src/types';
import type { HarnessCtx } from './gpu.scaled';

async function readTex(device: GPUDevice, t: GPUTexture): Promise<Uint8Array> {
  const W = t.width, H = t.height, bpr = Math.ceil((W * 4) / 256) * 256;
  const buf = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: t }, { buffer: buf, bytesPerRow: bpr }, [W, H]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buf.getMappedRange());
  const out = new Uint8Array(W * H * 4);
  const bgra = t.format === 'bgra8unorm';
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * bpr + 4 * x, o = (y * W + x) * 4;
    out[o] = src[i + (bgra ? 2 : 0)]; out[o + 1] = src[i + 1]; out[o + 2] = src[i + (bgra ? 0 : 2)]; out[o + 3] = 255;
  }
  buf.unmap(); buf.destroy();
  return out;
}

export async function shutterTest(ctx: HarnessCtx) {
  const device = await ctx.gpuDevice(ctx.flags.has('bgra') ? ['bgra8unorm-storage'] : []);
  const fx0 = ctx.FIXTURES.o3_0026;
  const SW = 1280, SH = 720, OW = 640, OH = 360, N = 24, FPS = 60000 / 1001;
  const sc = SW / fx0.srcW;
  const A = 0.02;                                    // yaw per source frame (rad), centred on the fixture's view: ±14°
  const M0 = fx0.records[0].rowMats as number[];
  const nr = fx0.nRows;
  const rowMats = new Float32Array(N * nr * 9);
  const virtQ = new Float64Array(N * 4);
  const ry = (th: number) => { const c = Math.cos(th), s = Math.sin(th); return [c, 0, s, 0, 1, 0, -s, 0, c]; };
  const mulRows = (out: Float32Array, o: number, th: number) => {
    const R = ry(th);
    for (let j = 0; j < nr; j++) for (let i = 0; i < 3; i++) for (let c = 0; c < 3; c++) {
      out[o + 9 * j + 3 * i + c] = M0[9 * j + 3 * i] * R[c] + M0[9 * j + 3 * i + 1] * R[3 + c] + M0[9 * j + 3 * i + 2] * R[6 + c];
    }
  };
  for (let k = 0; k < N; k++) {
    const th = A * (k - N / 2);
    mulRows(rowMats, k * nr * 9, th);
    virtQ[4 * k] = Math.cos(th / 2); virtQ[4 * k + 2] = Math.sin(th / 2);
  }
  const outFx = fx0.records[0].outFx * sc * OW / SW;
  const plan: Plan & { virtQ: Float64Array } = {
    srcW: SW, srcH: SH, outW: OW, outH: OH,
    lens: { model: 'kb4', fx: fx0.lens.fx * sc, fy: fx0.lens.fy * sc, cx: (SW - 1) / 2, cy: (SH - 1) / 2, k: fx0.lens.k.slice(0, 4), width: SW, height: SH },
    framePts: Float64Array.from({ length: N }, (_, k) => k / FPS), outFx: new Float32Array(N).fill(outFx),
    nRows: nr, rowMats, readoutS: fx0.readoutS, virtQ,
  };
  // one static source picture: fine vertical detail (so horizontal motion shows) + blocks
  const u = new Uint8Array(SW * SH * 4);
  for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
    const o = (y * SW + x) * 4, bar = (x % 23) < 3;
    u[o] = bar ? 250 : ((x >> 4) ^ (y >> 4)) & 1 ? 180 : 40; u[o + 1] = bar ? 245 : (y * 2) & 255; u[o + 2] = bar ? 240 : 90; u[o + 3] = 255;
  }
  const F = new VideoFrame(u, { format: 'RGBX', codedWidth: SW, codedHeight: SH, timestamp: 0, duration: 16683 });
  const warper = await Warper.create(device, plan, { kernel: 'lanczos3' });
  const tex = warper.createTarget('test.shutter');
  const res: any = { src: [SW, SH, N], out: [OW, OH], yawPerFrameDeg: +(A * 180 / Math.PI).toFixed(3), scaling: warper.scaling.mode };
  const fail: string[] = [];
  const rows = new Float32Array(nr * 9);
  const diff = (a: Uint8Array, b: Uint8Array) => { let m = 0, s = 0; for (let i = 0; i < a.length; i++) { if ((i & 3) === 3) continue; const d = Math.abs(a[i] - b[i]); m = Math.max(m, d); s += d; } return { max: m, mean: s / (a.length * 0.75) }; };
  const warpBytes = async (k: number) => { warper.warpTo(F, k, tex); return readTex(device, tex); };

  // ---- reuse: sub-frame warp with the record's own rows == warp()
  const w10 = await warpBytes(10);
  const f10 = subframeRows(plan, 10, plan.framePts[10], rows);
  warper.warpToWith(F, 10, tex, rows, f10); warper.warpToWith(F, 10, tex, rows, f10);
  const reuse = diff(await readTex(device, tex), w10);
  res.reuse = reuse;
  if (reuse.max > 0) fail.push(`reuse: own-rows sub-frame warp differs from warp() by ${reuse.max}`);

  // ---- direction: frame k at record k+1's instant == record k+1
  res.direction = [];
  for (const k of [5, 20]) {
    const want = await warpBytes(k + 1), own = await warpBytes(k);
    const f = subframeRows(plan, k, plan.framePts[k + 1], rows);
    warper.warpToWith(F, k, tex, rows, f);
    const got = await readTex(device, tex);
    const d = diff(got, want), moved = diff(own, want);
    res.direction.push({ k, vsNext: d, nextVsOwn: moved });
    if (d.max > 1) fail.push(`direction k=${k}: sub-frame at t(k+1) differs from record k+1 by ${d.max}`);
    if (moved.mean < 2) fail.push(`direction k=${k}: no visible motion between records (${moved.mean.toFixed(2)})`);
  }

  // ---- streak: ShutterRenderer vs dense reference vs plain frame blend, at 24 fps (2-3 taps per output) and 30 fps
  // (a 180° window no longer than a source frame: one tap, which the frame blend cannot blur at all)
  res.streak = {};
  const DENSE = 96;
  const t0 = plan.framePts[0] - 0.5 / FPS, t1 = plan.framePts[N - 1] + 0.5 / FPS;
  for (const fps of [24, 30]) {
    const sch = buildSchedule(plan.framePts, { fps, timing: 'realtime', motionBlur: 'natural' });
    const d = 0.5 / fps;
    const sr = await ShutterRenderer.create(warper, sch, { pts: plan.framePts, shutterS: d, transfer: 'bt709', pxStep: 1 });
    const rr = await RetimeRenderer.create(warper, sch, { transfer: 'bt709' });
    const outS: VideoFrame[] = [], outR: VideoFrame[] = [];
    for (let k = 0; k < N; k++) { outS.push(...sr.push(F, k)); outR.push(...rr.push(F, k)); }
    const run: any = { n: sch.n, maxTaps: sch.maxTaps, emitted: outS.length, stats: { ...sr.stats }, frames: [] as any[] };
    res.streak[fps] = run;
    if (outS.length !== sch.n || !sr.done) fail.push(`streak ${fps}: emitted ${outS.length}/${sch.n}`);
    let tsBad = 0, sumS = 0, sumR = 0, cnt = 0;
    for (let i = 0; i < outS.length; i++) {
      if (outS[i].timestamp !== Math.round(sch.outPtsUs[i])) tsBad++;
      const T = sch.outPtsUs[i] / 1e6;
      if (T - d / 2 < t0 || T + d / 2 > t1) continue;            // window must lie inside the clip
      // dense reference: views at DENSE instants across the window (continuous yaw), mean in linear light
      const acc = new Float64Array(OW * OH * 3);
      for (let m = 0; m < DENSE; m++) {
        const t = T - d / 2 + ((m + 0.5) / DENSE) * d;
        const r = Math.min(N - 1, Math.max(0, Math.round(t * FPS)));
        const f = subframeRows(plan, r, t, rows);
        warper.warpToWith(F, r, tex, rows, f);
        const b = await readTex(device, tex);
        for (let p = 0, q = 0; p < b.length; p += 4, q += 3) for (let c = 0; c < 3; c++) {
          const e = b[p + c] / 255;
          acc[q + c] += e < 0.081 ? e / 4.5 : ((e + 0.099) / 1.099) ** (1 / 0.45);
        }
      }
      const ref = new Uint8Array(OW * OH * 4);
      for (let p = 0, q = 0; p < ref.length; p += 4, q += 3) {
        for (let c = 0; c < 3; c++) { const l = acc[q + c] / DENSE; ref[p + c] = Math.round(255 * (l < 0.018 ? 4.5 * l : 1.099 * l ** 0.45 - 0.099)); }
        ref[p + 3] = 255;
      }
      const pack = (x: { buf: Uint8Array; layout: PlaneLayout[] }) => {
        const o = new Uint8Array(OW * OH * 4), st = x.layout[0].stride;
        for (let y = 0; y < OH; y++) o.set(x.buf.subarray(y * st, y * st + OW * 4), y * OW * 4);
        return o;
      };
      const ds = diff(pack(await ctx.frameBytes(outS[i], 'RGBA')), ref), dr = diff(pack(await ctx.frameBytes(outR[i], 'RGBA')), ref);
      run.frames.push({ i, taps: sch.tapsPerFrame[i], shutter: ds, frameBlend: dr });
      sumS += ds.mean; sumR += dr.mean; cnt++;
    }
    outS.forEach(f => f.close()); outR.forEach(f => f.close());
    run.meanShutter = +(sumS / cnt).toFixed(3);
    run.meanFrameBlend = +(sumR / cnt).toFixed(3);
    run.badTimestamps = tsBad;
    if (!cnt) fail.push(`streak ${fps}: no output window inside the clip`);
    if (tsBad) fail.push(`streak ${fps}: ${tsBad} timestamps`);
    if (!(sumS / cnt < 1.0)) fail.push(`streak ${fps}: mean |shutter - reference| ${(sumS / cnt).toFixed(2)} codes`);
    if (!(sumS < sumR / 3)) fail.push(`streak ${fps}: shutter (${(sumS / cnt).toFixed(2)}) not clearly closer to the reference than the frame blend (${(sumR / cnt).toFixed(2)})`);
    sr.destroy(); rr.destroy();
  }
  void blendRef;
  tex.destroy(); F.close(); warper.destroy(); device.destroy();
  res.pass = fail.length === 0;
  if (fail.length) res.error = 'shutter failed: ' + fail.join('; ');
  ctx.log('shutter', JSON.stringify({ reuse: res.reuse, direction: res.direction,
    streak: Object.fromEntries(Object.entries(res.streak).map(([k, v]: [string, any]) => [k, { meanShutter: v.meanShutter, meanFrameBlend: v.meanFrameBlend, maxTaps: v.maxTaps, stats: v.stats }])) }));
  return res;
}
