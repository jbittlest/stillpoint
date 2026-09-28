/**
 * PATH module tests (owner: PATH agent): so3 / lens unit checks, golden row matrices vs the Python engine
 * (plan_build.build_plan, 1e-6), and smoother quality vs the Python engine (smooth.optimize_path, Clarabel SQP)
 * on O3 DJI_0034 (whole clip) and Osmo Action 4 DJI_..._0012 (60 s window). Fixtures: web/test/path.export.py.
 *
 *   cd web && npx vitest run test/path.test.ts
 */
import { describe, expect, it } from 'vitest';
import { hasFixture, loadFixture, type PathFixture } from './path.fixtures';
import {
  buildPlan, cameraRowMats, mapPlanPoint, orientationSeries, planFootprint, rowMatricesF64, rowMatsFromCam,
  useExposureAvg,
} from '../src/plan';
import { BandSPD, checkCrop, optimizePath, pathJitterPx, zoomEnvelope, type PathProblem } from '../src/smooth';
import { fxForFootprint, identityFootprint, project, projectJac, unproject } from '../src/lens';
import { jrInvInto, OrientationSeries, qexpInto, qlogInto, qmulInto, quatToMatInto } from '../src/so3';
import type { Lens } from '../src/types';

const CASES = ['o3_0034', 'oa4_0012_w100'] as const;
const fixtures = new Map<string, PathFixture>();
const fx = (n: string) => {
  if (!fixtures.has(n)) fixtures.set(n, loadFixture(n));
  return fixtures.get(n)!;
};
const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

function problem(f: PathFixture, nRows = 32): PathProblem {
  const tel = f.tel;
  const F = tel.framePts.length;
  const ser = orientationSeries(tel);
  const rows = new Float32Array(F * nRows * 9);
  cameraRowMats(tel, ser, nRows, useExposureAvg(tel), rows);
  const camQ = new Float64Array(4 * F);
  for (let k = 0; k < F; k++) ser.at(tel.frameT[k], camQ, 4 * k);
  return { nFrames: F, fps: tel.fps, srcW: tel.width, srcH: tel.height, outW: tel.width, outH: tel.height,
    lens: tel.lens, camQ, camRows: rows, nRows, segments: tel.segments };
}

// ------------------------------------------------------------------------------------------------ unit
describe('so3 / lens', () => {
  it('qexp/qlog round trip and Jr⁻¹ is the inverse right Jacobian', () => {
    const q = new Float64Array(8), v = new Float64Array(3);
    for (const th of [[0.3, -0.2, 0.1], [1e-9, 0, 2e-9], [2.5, 0.4, -1.0]]) {
      qexpInto(q, 0, th[0], th[1], th[2]);
      qlogInto(v, 0, q, 0);
      expect(maxAbsDiff(v, th)).toBeLessThan(1e-12);
    }
    // Exp(th)·Exp(d) ≈ Exp(th + Jr⁻¹ d)
    const th = [0.4, -0.3, 0.2], d = [1e-6, -2e-6, 1.5e-6];
    const J = new Float64Array(9);
    jrInvInto(J, 0, th[0], th[1], th[2]);
    qexpInto(q, 0, th[0], th[1], th[2]);
    qexpInto(q, 4, d[0], d[1], d[2]);
    const p = new Float64Array(4);
    qmulInto(p, 0, q, 0, q, 4);
    qlogInto(v, 0, p, 0);
    for (let c = 0; c < 3; c++) {
      const pred = th[c] + J[3 * c] * d[0] + J[3 * c + 1] * d[1] + J[3 * c + 2] * d[2];
      expect(Math.abs(v[c] - pred)).toBeLessThan(1e-11);
    }
  });

  it('KB4 project/unproject round trip and analytic Jacobian', () => {
    const L: Lens = { model: 'kb4', fx: 1405.13, fy: 1405.13, cx: 1919.5, cy: 1079.5,
      k: [0.2499, 0.0136, -0.0621, 0.0122], width: 3840, height: 2160 };
    const uv = new Float64Array(2), r = new Float64Array(3), J = new Float64Array(6), up = new Float64Array(2);
    for (const [u, v] of [[100, 50], [1919.5, 1079.5], [3700, 2100], [2500, 300]]) {
      unproject(L, u, v, r, 0);
      project(L, r[0], r[1], r[2], uv, 0);
      expect(Math.hypot(uv[0] - u, uv[1] - v)).toBeLessThan(1e-6);
      projectJac(L, r[0] * 2, r[1] * 2, r[2] * 2, uv, 0, J, 0);
      for (let a = 0; a < 3; a++) {
        const h = 1e-6, X = [r[0] * 2, r[1] * 2, r[2] * 2];
        X[a] += h; project(L, X[0], X[1], X[2], up, 0);
        const X2 = [r[0] * 2, r[1] * 2, r[2] * 2];
        X2[a] -= h; project(L, X2[0], X2[1], X2[2], uv, 0);
        expect(Math.abs((up[0] - uv[0]) / (2 * h) - J[a])).toBeLessThan(1e-3);
        expect(Math.abs((up[1] - uv[1]) / (2 * h) - J[3 + a])).toBeLessThan(1e-3);
      }
    }
    // identity footprint is monotone in the focal and fxForFootprint inverts it
    const f60 = fxForFootprint(L, 3840, 2160, 3840, 2160, 0.60);
    expect(Math.abs(identityFootprint(L, 3840, 2160, 3840, 2160, f60) - 0.60)).toBeLessThan(1e-6);
  });

  it('banded Cholesky solves a random SPD band system', () => {
    const n = 60, b = 11, M = new BandSPD(n, b);
    const dense = Array.from({ length: n }, () => new Float64Array(n));
    let seed = 1;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5;
    for (let i = 0; i < n; i++) for (let j = Math.max(0, i - b); j <= i; j++) {
      const v = i === j ? 20 + rnd() : rnd();
      dense[i][j] = dense[j][i] = v;
      M.add(i, j, v);
    }
    const x = Float64Array.from({ length: n }, rnd), rhs = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) rhs[i] += dense[i][j] * x[j];
    M.factor();
    const y = new Float64Array(n);
    M.solve(rhs, y);
    expect(maxAbsDiff(x, y)).toBeLessThan(1e-12);
  });

  it('slerp series matches geom.slerp_series semantics (clamped ends, exact at samples)', () => {
    const t = Float64Array.from([0, 0.001, 0.002, 0.003]);
    const q = new Float64Array(16);
    for (let i = 0; i < 4; i++) qexpInto(q, 4 * i, 0.1 * i, -0.05 * i, 0.02 * i);
    q.set([-q[8], -q[9], -q[10], -q[11]], 8); // a sign flip must be undone
    const s = new OrientationSeries(t, q);
    const o = new Float64Array(4), e = new Float64Array(4);
    s.at(0.002, o, 0); qexpInto(e, 0, 0.2, -0.1, 0.04);
    expect(Math.min(maxAbsDiff(o, e), maxAbsDiff(o, e.map((v) => -v)))).toBeLessThan(1e-14);
    s.at(-1, o, 0); expect(maxAbsDiff(o, [1, 0, 0, 0])).toBeLessThan(1e-14);
    s.at(0.0015, o, 0); qexpInto(e, 0, 0.15, -0.075, 0.03);
    expect(Math.min(maxAbsDiff(o, e), maxAbsDiff(o, e.map((v) => -v)))).toBeLessThan(1e-12);
  });

  it('zoom envelope is slope limited, bridges short gaps and covers the need', () => {
    const need = new Float64Array(200);
    need[50] = 0.1; need[70] = 0.08; need[180] = 0.05;
    const z = zoomEnvelope(need, 0.004, 30, 0.4, new Int32Array(200));
    for (let k = 1; k < 200; k++) expect(Math.abs(z[k] - z[k - 1])).toBeLessThanOrEqual(0.004 + 1e-12);
    for (let k = 0; k < 200; k++) expect(z[k]).toBeGreaterThanOrEqual(need[k]);
    for (let k = 50; k <= 70; k++) expect(z[k]).toBeGreaterThanOrEqual(0.08 - 1e-12); // no dip between the events
  });
});

// ------------------------------------------------------------------------------------------------ golden
describe.each(CASES)('golden %s', (name) => {
  const ok = hasFixture(name);
  it.runIf(ok)('(a) row matrices of the Python virtual path match plan_build.build_plan to 1e-6', () => {
    const f = fx(name);
    const gold = Array.from(f.arr.goldFrames as Float64Array, (v) => Math.round(v));
    const G = gold.length, R = f.meta.python.n_rows;
    const pv = f.arr.virtQ as Float64Array;
    const vq = new Float64Array(4 * G);
    gold.forEach((k, i) => vq.set(pv.subarray(4 * k, 4 * k + 4), 4 * i));
    // exact float64 path (exposure averaging as the engine decided)
    const { rowMats, exposureAvg } = rowMatricesF64(f.tel, vq, R, gold);
    expect(exposureAvg).toBe(f.meta.python.exposure_avg);
    const d64 = maxAbsDiff(rowMats, f.arr.goldRowMats);
    // instantaneous orientation (exposure averaging off)
    const inst = rowMatricesF64(f.tel, vq, R, gold, false);
    const dInst = maxAbsDiff(inst.rowMats, f.arr.goldRowMatsInst);
    // the renderer path: float32 camera rows -> float32 plan row matrices
    const cam = new Float32Array(G * R * 9);
    cameraRowMats(f.tel, orientationSeries(f.tel), R, exposureAvg, cam, gold);
    rowMatsFromCam(cam, vq, R, cam);
    const d32 = maxAbsDiff(cam, f.arr.goldRowMats);
    console.log(`[path] ${name} row matrices vs Python: f64 ${d64.toExponential(2)}, instantaneous ${dInst.toExponential(2)}, ` +
      `f32 plan ${d32.toExponential(2)} (exposure_avg=${exposureAvg}, ${G} frames x ${R} rows)`);
    expect(d64).toBeLessThan(1e-9);
    expect(dInst).toBeLessThan(1e-9);
    expect(d32).toBeLessThan(1e-6);
  });

  it.runIf(ok && name === 'o3_0034')('planFootprint agrees with eval.footprint.plan_footprint (0.6266 on the Python path)', () => {
    // reference: eval/footprint.py plan_footprint of the gyro-only Python plan at every 15th record = 0.62657
    const f = fx(name);
    const F = f.tel.framePts.length, R = 32;
    const vq = f.arr.virtQ as Float64Array;
    const { rowMats } = rowMatricesF64(f.tel, vq, R);
    const plan = { srcW: f.tel.width, srcH: f.tel.height, outW: f.tel.width, outH: f.tel.height, lens: f.tel.lens,
      framePts: f.tel.framePts, outFx: Float32Array.from(f.arr.outFx), nRows: R, rowMats: Float32Array.from(rowMats),
      readoutS: f.tel.readoutS };
    let fp = 0, n = 0;
    for (let k = 0; k < F; k += 15) { fp += planFootprint(plan, k); n++; }
    console.log(`[path] planFootprint of the Python path: ${(fp / n).toFixed(4)} (eval.footprint 0.6266)`);
    expect(Math.abs(fp / n - 0.62657)).toBeLessThan(0.002);
  });

  it.runIf(ok)('(b) smoother: HF jitter <= 1.3x Python, zero crop violations, bounded zoom, fast', () => {
    const f = fx(name);
    const P = problem(f);
    const fx0 = f.meta.python.fx0;
    const t0 = performance.now();
    const res = optimizePath(P, { fx0, fxMax: fx0 * 1.5, smoothness: 1 });
    const ms = performance.now() - t0;
    const f1080 = fx0 * 1920 / f.tel.width;
    const jt = pathJitterPx(res.virtQ, f.tel.fps, f1080, 1.5, f.tel.segments);
    const jp = pathJitterPx(f.arr.virtQ as Float64Array, f.tel.fps, f1080, 1.5, f.tel.segments);
    const jc = pathJitterPx(P.camQ, f.tel.fps, f1080, 1.5, f.tel.segments);
    const v0 = checkCrop(P, res.virtQ, res.outFx, 64, 0);
    const v8 = checkCrop(P, res.virtQ, res.outFx, 64, 8);
    const mx = (a: Float64Array) => a.reduce((m, x) => Math.max(m, x), -Infinity);
    console.log(`[path] ${name}: HF(>1.5 Hz, 1080p px) camera ${jc.rms.toFixed(3)} | Python ${jp.rms.toFixed(3)} | TS ` +
      `${jt.rms.toFixed(3)} (ratio ${(jt.rms / jp.rms).toFixed(3)}); crop max viol ${mx(v0).toFixed(2)} px (margin 0), ` +
      `${mx(v8).toFixed(2)} px (margin 8); zoom changes ${res.info.zoomChanges} (Python ${f.meta.python.zoom_changes}), ` +
      `max zoom ${res.info.maxZoom.toFixed(4)}; ${F(res.info)} ; ${ms.toFixed(0)} ms (Python ${f.meta.python.optimize_s.toFixed(1)} s)`);
    expect(jt.rms / jp.rms).toBeLessThanOrEqual(1.3);
    expect(mx(v0)).toBeLessThanOrEqual(0);
    expect(res.info.maxZoom).toBeLessThanOrEqual(1.5 + 1e-9);
  }, 120_000);
});

const F = (i: { sqpIters: number; innerIters: number }) => `${i.sqpIters} SQP / ${i.innerIters} inner iterations`;

// ------------------------------------------------------------------------------------------------ plan
describe('buildPlan', () => {
  const ok = hasFixture('o3_0034');
  it.runIf(ok)('default plan: shared Plan shape, footprint ~0.60, kernel-exact border inside the source', () => {
    const f = fx('o3_0034');
    let lastP = -1, mono = true;
    const plan = buildPlan(f.tel, { smoothness: 1 }, (p) => { if (p < lastP - 1e-12) mono = false; lastP = p; });
    const Fn = f.tel.framePts.length;
    expect(mono).toBe(true);
    expect(plan.rowMats).toBeInstanceOf(Float32Array);
    expect(plan.rowMats.length).toBe(Fn * plan.nRows * 9);
    expect(plan.outFx.length).toBe(Fn);
    expect(plan.outW).toBe(f.tel.width);
    expect(plan.outH).toBe(f.tel.height);
    expect(plan.framePts[10]).toBe(f.tel.framePts[10]);
    let fp = 0, n = 0;
    for (let k = 0; k < Fn; k += 30) { fp += planFootprint(plan, k); n++; }
    fp /= n;
    // every border pixel of a few hundred records lands inside the source through the kernel-exact mapping
    const uv = new Float64Array(2);
    let worst = -Infinity;
    for (let k = 0; k < Fn; k += 7) {
      for (let i = 0; i <= 32; i++) {
        for (const [x, y] of [[i * (plan.outW - 1) / 32, 0], [i * (plan.outW - 1) / 32, plan.outH - 1],
          [0, i * (plan.outH - 1) / 32], [plan.outW - 1, i * (plan.outH - 1) / 32]]) {
          mapPlanPoint(plan, k, x, y, uv);
          worst = Math.max(worst, -uv[0], uv[0] - (plan.srcW - 1), -uv[1], uv[1] - (plan.srcH - 1));
        }
      }
    }
    console.log(`[path] buildPlan o3_0034: fx0 ${plan.fx0.toFixed(1)} (hfov ${plan.hfovDeg.toFixed(1)} deg), mean footprint ` +
      `${fp.toFixed(4)}, worst border ${worst.toFixed(2)} px, timings ${JSON.stringify(plan.timings)}`);
    expect(Math.abs(fp - 0.60)).toBeLessThan(0.03);
    expect(worst).toBeLessThanOrEqual(0);
  }, 120_000);

  it.runIf(ok)('smoothness knob: 0 follows the camera, 2 is floatier; segments are never coupled', () => {
    const f = fx('o3_0034');
    // a 1200-frame piece keeps this fast
    const n = 1200, tel = { ...f.tel, framePts: f.tel.framePts.subarray(0, n), frameT: f.tel.frameT.subarray(0, n),
      exposureS: f.tel.exposureS.subarray(0, n), segments: [[0, n - 1]] as Array<[number, number]> };
    const P = problem({ ...f, tel });
    const fx0 = f.meta.python.fx0, f1080 = fx0 * 1920 / tel.width;
    const j = [0, 1, 2].map((s) => {
      const r = optimizePath(P, { fx0, smoothness: s });
      return pathJitterPx(r.virtQ, tel.fps, f1080, 1.5).rms;
    });
    console.log(`[path] smoothness 0/1/2 -> HF ${j.map((v) => v.toFixed(3)).join(' / ')} px`);
    expect(j[0]).toBeGreaterThan(j[1]);
    expect(j[1]).toBeGreaterThan(j[2]);
    // two shots: the path is continuous inside each, free at the cut (a cut in the camera path stays a cut)
    const P2 = { ...P, segments: [[0, 599], [600, n - 1]] as Array<[number, number]> };
    const r2 = optimizePath(P2, { fx0, smoothness: 1 });
    const v = checkCrop(P2, r2.virtQ, r2.outFx, 32, 0);
    expect(v.reduce((m, x) => Math.max(m, x), -Infinity)).toBeLessThanOrEqual(0);
  }, 120_000);

  it('horizon lock levels a tilted horizon (synthetic telemetry with gravity)', () => {
    // camera looks along world +x with y = world z (down), rolled 4° about the optical axis plus a 2 Hz wobble
    const N = 3000, fps = 60, F = 400;
    const imuT = new Float64Array(N), imuQ = new Float64Array(4 * N);
    const r0 = new Float64Array(4), rz = new Float64Array(4);
    // R0 columns [e_y, e_z, e_x]  <=> rotation taking cam x->world y, cam y->world z, cam z->world x
    const M0 = [0, 0, 1, 1, 0, 0, 0, 1, 0];
    const tr = M0[0] + M0[4] + M0[8];
    const w = Math.sqrt(1 + tr) / 2;
    r0.set([w, (M0[7] - M0[5]) / (4 * w), (M0[2] - M0[6]) / (4 * w), (M0[3] - M0[1]) / (4 * w)]);
    for (let i = 0; i < N; i++) {
      imuT[i] = i / 400;
      const roll = (4 + 0.5 * Math.sin(2 * Math.PI * 2 * imuT[i])) * Math.PI / 180;
      qexpInto(rz, 0, 0, 0, roll);
      qmulInto(imuQ, 4 * i, r0, 0, rz, 0);
    }
    const lens: Lens = { model: 'kb4', fx: 1405.13, fy: 1405.13, cx: 1919.5, cy: 1079.5,
      k: [0.2499, 0.0136, -0.0621, 0.0122], width: 3840, height: 2160 };
    const framePts = Float64Array.from({ length: F }, (_, k) => k / fps);
    const tel: any = { camera: 'synthetic', width: 3840, height: 2160, fps, framePts, frameT: Float64Array.from(framePts, (t) => t + 0.2),
      exposureS: new Float64Array(F).fill(0.001), readoutS: 0.009, lens, imuT, imuQ, imuRate: 400, hasHighrate: false,
      eisBaked: false, warnings: [],
      // the IMU world is already gravity aligned (z down): gravity orientation == IMU orientation
      gravityQ: Float64Array.from(imuQ) };
    const tilt = (plan: ReturnType<typeof buildPlan>) => {
      const M = new Float64Array(9);
      let acc = 0;
      for (let k = 50; k < F - 50; k++) {
        quatToMatInto(M, 0, plan.virtQ, 4 * k);
        acc += Math.abs(Math.asin(M[8 - 2] /* (Vᵀ e_z)_x = V[2][0] */));
      }
      return acc / (F - 100) * 180 / Math.PI;
    };
    const free = buildPlan(tel, { smoothness: 1 });
    const locked = buildPlan(tel, { smoothness: 1, horizonLock: true });
    console.log(`[path] horizon tilt: free ${tilt(free).toFixed(2)} deg, locked ${tilt(locked).toFixed(2)} deg`);
    expect(tilt(free)).toBeGreaterThan(3);
    expect(tilt(locked)).toBeLessThan(0.5);
  }, 60_000);

  it('rejects EIS-baked clips', () => {
    const tel: any = { eisBaked: true, framePts: new Float64Array(0) };
    expect(() => buildPlan(tel, { smoothness: 1 })).toThrow(/EIS/);
  });
});
