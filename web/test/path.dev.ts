/**
 * PATH dev harness (owner: PATH agent): runs the TS smoother on a fixture and prints quality/runtime numbers.
 * Bundle + run (Node):  node test/path.dev.run.mjs <fixture> [key=value ...]
 */
import { loadFixture } from './path.fixtures';
import { cameraRowMats, orientationSeries, rowMatricesF64, useExposureAvg, buildPlan, planFootprint } from '../src/plan';
import { checkCrop, optimizePath, pathJitterPx, type PathProblem } from '../src/smooth';
import { identityFootprint } from '../src/lens';
import { qconjMulInto, qlogInto } from '../src/so3';

/** smooth._objective (Python's exact nonlinear objective, component-wise L1) for a path. */
function pyObjective(V: Float64Array, R: Float64Array, f: number, s = 1) {
  const F = V.length >> 2, tmp = new Float64Array(8);
  const w = new Float64Array(3 * (F - 1));
  for (let t = 0; t < F - 1; t++) { qconjMulInto(tmp, 0, V, 4 * t, V, 4 * t + 4); qlogInto(tmp, 4, tmp, 0); for (let c = 0; c < 3; c++) w[3 * t + c] = f * tmp[4 + c]; }
  let accL1 = 0, jerkL1 = 0, accL2 = 0, vel = 0, fid = 0;
  for (let t = 0; t < F - 2; t++) for (let c = 0; c < 3; c++) { const a = w[3 * (t + 1) + c] - w[3 * t + c]; accL1 += Math.abs(a); accL2 += a * a; }
  for (let t = 0; t < F - 3; t++) for (let c = 0; c < 3; c++) { const j = w[3 * (t + 2) + c] - 2 * w[3 * (t + 1) + c] + w[3 * t + c]; jerkL1 += Math.abs(j); }
  for (let i = 0; i < w.length; i++) vel += w[i] * w[i];
  for (let k = 0; k < F; k++) { qconjMulInto(tmp, 0, R, 4 * k, V, 4 * k); qlogInto(tmp, 4, tmp, 0); for (let c = 0; c < 3; c++) fid += (f * tmp[4 + c]) ** 2; }
  const W = { accL1: 1, jerkL1: 10, accL2: 30 * 10 ** (s - 1), vel: 1e-4 * s, fid: 1e-4 * 10 ** (2 * (1 - s)) };
  const o = { accL1: W.accL1 * accL1, jerkL1: W.jerkL1 * jerkL1, accL2: W.accL2 * accL2, vel: W.vel * vel, fid: W.fid * fid };
  return { ...o, total: o.accL1 + o.jerkL1 + o.accL2 + o.vel + o.fid };
}

const argv = process.argv.slice(2);
const name = argv[0] ?? 'o3_0034';
const kv: Record<string, string> = {};
for (const a of argv.slice(1)) { const [k, v] = a.split('='); kv[k] = v; }
const dir = kv.dir ?? (process.cwd() + '/test/fixtures/path');
const fx = loadFixture(name, dir);
const { tel, meta, arr } = fx;
const F = tel.framePts.length;
const t0 = performance.now();
const c0 = process.cpuUsage();
const ser = orientationSeries(tel);
const nRows = 32;
const avg = useExposureAvg(tel);
const rows = new Float32Array(F * nRows * 9);
cameraRowMats(tel, ser, nRows, avg, rows);
const camQ = new Float64Array(4 * F);
for (let k = 0; k < F; k++) ser.at(tel.frameT[k], camQ, 4 * k);
const tRows = (performance.now() - t0) / 1000;
const P: PathProblem = {
  nFrames: F, fps: tel.fps, srcW: tel.width, srcH: tel.height, outW: tel.width, outH: tel.height, lens: tel.lens,
  camQ, camRows: rows, nRows, segments: tel.segments,
};
const fx0 = kv.fx0 ? Number(kv.fx0) : meta.python.fx0;
const opts: any = { fx0, fxMax: fx0 * 1.5, smoothness: kv.s ? Number(kv.s) : 1, log: kv.v ? console.log : undefined };
for (const k of ['innerIters', 'sqpIters', 'repairIters', 'epsL1', 'rhoCrop', 'wJerkL1', 'wAccL1', 'wAccL2', 'rhoBox', 'reachCapPx', 'prox', 'boxMode', 'tauCropPx']) if (kv[k]) opts[k] = Number(kv[k]);
if (kv.dbg) { const hist: string[] = []; opts.debugInner = (it: number, mx: number, x: any) => hist.push(kv.dbg === '2' ? `\n   ${it}: ${mx.toExponential(2)} ${JSON.stringify(x)}` : `${it}:${mx.toExponential(1)}`); opts.log = (m: string) => { console.log(m, hist.join(' ')); hist.length = 0; }; }
if (kv.epsSched) opts.epsSchedule = kv.epsSched.split(',').map(Number);
if (kv.tolSched) opts.tolSchedule = kv.tolSched.split(',').map(Number);
if (kv.tr) opts.trustRad = kv.tr.split(',').map(Number);
const c1 = process.cpuUsage();
const res = optimizePath(P, opts);
const c2 = process.cpuUsage(c1);
const cpuRows = (c1.user - c0.user) / 1e6, cpuPath = c2.user / 1e6;
const f1080 = fx0 * 1920 / tel.width;
const py = arr.virtQ as Float64Array;
const jt = pathJitterPx(res.virtQ, tel.fps, f1080, 1.5, tel.segments);
const jp = pathJitterPx(py, tel.fps, f1080, 1.5, tel.segments);
const jc = pathJitterPx(camQ, tel.fps, f1080, 1.5, tel.segments);
const jt8 = pathJitterPx(res.virtQ, tel.fps, f1080, 8, tel.segments);
const jp8 = pathJitterPx(py, tel.fps, f1080, 8, tel.segments);
const tc = performance.now();
const v0 = checkCrop(P, res.virtQ, res.outFx, 64, 0);
const v8 = checkCrop(P, res.virtQ, res.outFx, 64, 8);
const pyFx = arr.outFx as Float64Array;
const pv0 = checkCrop(P, py, pyFx, 64, 0);
const tcheck = (performance.now() - tc) / 1000;
const mx = (a: Float64Array) => a.reduce((m, x) => Math.max(m, x), -Infinity);
const cnt = (a: Float64Array) => a.reduce((m, x) => m + (x > 0 ? 1 : 0), 0);
// deviation between paths
let dmax = 0, dsum = 0;
for (let k = 0; k < F; k++) {
  const d = Math.abs(res.virtQ[4 * k] * py[4 * k] + res.virtQ[4 * k + 1] * py[4 * k + 1] + res.virtQ[4 * k + 2] * py[4 * k + 2] + res.virtQ[4 * k + 3] * py[4 * k + 3]);
  const ang = 2 * Math.acos(Math.min(1, d)) * 180 / Math.PI;
  dmax = Math.max(dmax, ang); dsum += ang;
}
console.log(JSON.stringify({
  name, F, avg, tRows, cpuRows, cpuPath, runtime: res.info.runtimeS, timings: res.info.timings, sqp: res.info.sqpIters, inner: res.info.innerIters,
  jitterTS: jt.rms, jitterPy: jp.rms, ratio: jt.rms / jp.rms, jitterCam: jc.rms, ratio8Hz: jt8.rms / jp8.rms,
  perAxisTS: jt.perAxis, perAxisPy: jp.perAxis,
  viol0TS: mx(v0), nViol0TS: cnt(v0), viol8TS: mx(v8), nViol8TS: cnt(v8), viol0Py: mx(pv0), tcheck,
  zoomChanges: res.info.zoomChanges, zoomEvents: res.info.zoomEvents, globalZoom: res.info.globalZoom, maxZoom: res.info.maxZoom, pyZoomChanges: meta.python.zoom_changes, pyMaxZoom: meta.python.max_zoom,
  maxPhiTS: res.info.maxPhiDeg, maxPhiPy: meta.python.max_phi_deg, devMaxDeg: dmax, devMeanDeg: dsum / F,
  objTS: pyObjective(res.virtQ, camQ, fx0), objPy: pyObjective(py, camQ, fx0),
  pyRuntime: meta.python.optimize_s, jacBound: res.info.jacBound, idFootprint: identityFootprint(tel.lens, tel.width, tel.height, tel.width, tel.height, fx0),
}, null, 1));
if (kv.iters) console.log(res.info.iters);
