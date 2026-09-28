/**
 * PATH browser benchmark page (owner: PATH agent). Loads a PATH fixture (json + bin.gz) over HTTP, runs the real
 * buildPlan (default parameters) and the smoother at the Python engine's focal in the page's main thread, compares
 * the virtual path with the Python engine's, and publishes the numbers on window.__pathResult.
 *   ?fixture=<url prefix without extension>   e.g. /test/fixtures/path/o3_0034 or /__clips/oa4_0012_full
 */
import type { Telemetry } from '../src/types';
import { buildPlan, cameraRowMats, orientationSeries, planFootprint, useExposureAvg } from '../src/plan';
import { checkCrop, optimizePath, pathJitterPx, type PathProblem } from '../src/smooth';

async function load(prefix: string) {
  const meta = await (await fetch(prefix + '.json')).json();
  const gz = await fetch(prefix + '.bin.gz');
  // static servers may send .gz with Content-Encoding: gzip (then fetch has already inflated it)
  const buf = /gzip/.test(gz.headers.get('content-encoding') ?? '') ? await gz.arrayBuffer()
    : await new Response(gz.body!.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const arr: Record<string, Float64Array | Float32Array> = {};
  for (const [k, v] of Object.entries<any>(meta.layout)) {
    arr[k] = v.dtype === 'f32' ? new Float32Array(buf, v.offset, v.length) : new Float64Array(buf, v.offset, v.length);
  }
  const T = meta.tel;
  const imuT = new Float64Array(T.nImu);
  for (let i = 0; i < T.nImu; i++) imuT[i] = T.imuT0 + i * T.imuDt;
  const tel: Telemetry = {
    camera: T.camera, width: T.width, height: T.height, fps: T.fps,
    framePts: arr.framePts as Float64Array, frameT: arr.frameT as Float64Array, exposureS: arr.exposureS as Float64Array,
    readoutS: T.readoutS, lens: { ...T.lens }, imuT, imuQ: Float64Array.from(arr.imuQ32), imuRate: T.imuRate,
    hasHighrate: T.hasHighrate, eisBaked: T.eisBaked, warnings: [], segments: T.segments,
  };
  return { meta, arr, tel };
}

async function main() {
  const out: any = { ua: navigator.userAgent, hc: navigator.hardwareConcurrency };
  try {
    const prefix = new URLSearchParams(location.search).get('fixture') ?? '/test/fixtures/path/o3_0034';
    const t0 = performance.now();
    const { meta, arr, tel } = await load(prefix);
    out.loadMs = performance.now() - t0;
    out.frames = tel.framePts.length;
    // 1) the real entry point, default parameters (footprint 0.60)
    let t = performance.now();
    const plan = buildPlan(tel, { smoothness: 1 });
    out.buildPlanMs = performance.now() - t;
    out.planTimings = plan.timings;
    out.fx0 = plan.fx0;
    out.hfovDeg = plan.hfovDeg;
    out.smoothInfo = { ...plan.smoothInfo, iters: undefined };
    let fp = 0, n = 0;
    for (let k = 0; k < tel.framePts.length; k += Math.max(1, Math.floor(tel.framePts.length / 300))) { fp += planFootprint(plan, k); n++; }
    out.footprintMean = fp / n;
    // 2) the smoother at the Python engine's focal, compared with the Python path
    const F = tel.framePts.length;
    const ser = orientationSeries(tel);
    const rows = new Float32Array(F * 32 * 9);
    cameraRowMats(tel, ser, 32, useExposureAvg(tel), rows);
    const camQ = new Float64Array(4 * F);
    for (let k = 0; k < F; k++) ser.at(tel.frameT[k], camQ, 4 * k);
    const P: PathProblem = { nFrames: F, fps: tel.fps, srcW: tel.width, srcH: tel.height, outW: tel.width,
      outH: tel.height, lens: tel.lens, camQ, camRows: rows, nRows: 32, segments: tel.segments };
    const fx0 = meta.python.fx0;
    t = performance.now();
    const res = optimizePath(P, { fx0, fxMax: fx0 * 1.5, smoothness: 1 });
    out.smoothMs = performance.now() - t;
    const f1080 = fx0 * 1920 / tel.width;
    out.jitterTS = pathJitterPx(res.virtQ, tel.fps, f1080, 1.5, tel.segments).rms;
    out.jitterPy = pathJitterPx(arr.virtQ as Float64Array, tel.fps, f1080, 1.5, tel.segments).rms;
    out.jitterCam = pathJitterPx(camQ, tel.fps, f1080, 1.5, tel.segments).rms;
    out.ratio = out.jitterTS / out.jitterPy;
    const v = checkCrop(P, res.virtQ, res.outFx, 32, 0);
    out.maxViolPx = v.reduce((m, x) => Math.max(m, x), -Infinity);
    out.pythonOptimizeS = meta.python.optimize_s;
    out.ok = true;
  } catch (e: any) {
    out.ok = false;
    out.error = String(e?.stack ?? e);
  }
  (window as any).__pathResult = out;
  document.body.textContent = JSON.stringify(out, null, 1);
}

main();
