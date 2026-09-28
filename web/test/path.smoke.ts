/**
 * PATH smoke run (owner: PATH agent): buildPlan with default parameters on telemetry-only fixtures (no Python path),
 * reports runtime, zoom, exact crop check and HF jitter camera -> virtual. Bundled + run by path.dev.run.mjs-style:
 *   node test/path.dev.run.mjs --smoke <fixture> dir=<dir>
 */
import { loadFixture } from './path.fixtures';
import { buildPlan, mapPlanPoint, planFootprint, orientationSeries } from '../src/plan';
import { pathJitterPx } from '../src/smooth';

const argv = process.argv.slice(2);
const kv: Record<string, string> = {};
for (const a of argv.slice(1)) { const [k, v] = a.split('='); kv[k] = v; }
const f = loadFixture(argv[0], kv.dir);
const tel = f.tel;
const F = tel.framePts.length;
const c0 = process.cpuUsage();
const t0 = performance.now();
const plan = buildPlan(tel, { smoothness: kv.s ? Number(kv.s) : 1, footprint: kv.fp ? Number(kv.fp) : undefined },
  undefined);
const ms = performance.now() - t0;
const cpu = process.cpuUsage(c0).user / 1e6;
// exact kernel-equivalent border check on every frame (32 samples per edge)
const uv = new Float64Array(2);
let worst = -Infinity, bad = 0;
for (let k = 0; k < F; k++) {
  let w = -Infinity;
  for (let i = 0; i <= 32; i++) {
    for (const [x, y] of [[i * (plan.outW - 1) / 32, 0], [i * (plan.outW - 1) / 32, plan.outH - 1],
      [0, i * (plan.outH - 1) / 32], [plan.outW - 1, i * (plan.outH - 1) / 32]]) {
      mapPlanPoint(plan, k, x, y, uv);
      w = Math.max(w, -uv[0], uv[0] - (plan.srcW - 1), -uv[1], uv[1] - (plan.srcH - 1));
    }
  }
  worst = Math.max(worst, w);
  if (w > 0) bad++;
}
let fp = 0, n = 0;
for (let k = 0; k < F; k += Math.max(1, Math.floor(F / 400))) { fp += planFootprint(plan, k); n++; }
const ser = orientationSeries(tel);
const camQ = new Float64Array(4 * F);
for (let k = 0; k < F; k++) ser.at(tel.frameT[k], camQ, 4 * k);
const f1080 = plan.fx0 * 1920 / tel.width;
console.log(JSON.stringify({
  name: argv[0], camera: tel.camera, F, imuRate: Math.round(tel.imuRate), highrate: tel.hasHighrate, wallMs: Math.round(ms), cpuS: cpu,
  timings: plan.timings, exposureAvg: plan.exposureAvg, hfov: +plan.hfovDeg.toFixed(1), footprint: +(fp / n).toFixed(4),
  hfCam: +pathJitterPx(camQ, tel.fps, f1080, 1.5, tel.segments).rms.toFixed(3),
  hfVirt: +pathJitterPx(plan.virtQ, tel.fps, f1080, 1.5, tel.segments).rms.toFixed(3),
  worstBorderPx: +worst.toFixed(2), framesOutside: bad, zoom: { max: plan.smoothInfo.maxZoom, global: plan.smoothInfo.globalZoom,
    events: plan.smoothInfo.zoomEvents, changes: plan.smoothInfo.zoomChanges }, sqp: plan.smoothInfo.sqpIters, inner: plan.smoothInfo.innerIters,
}));
