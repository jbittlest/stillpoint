/**
 * PATH output targeting tests (owner: PLAN agent): outputGeometry presets/aspects, buildPlan(params.output) at other
 * sizes (same path, scaled focal) and other aspects (aspect-aware crop constraint, footprint/FOV semantics) on the
 * O3 DJI_0034 and OA4 0012 fixtures.
 *
 *   cd web && npx vitest run test/path.output.test.ts
 */
import { describe, expect, it } from 'vitest';
import { hasFixture, loadFixture, type PathFixture } from './path.fixtures';
import {
  aspectAreaFraction, buildPlan, mapPlanPoint, outputGeometry, planFootprint, referenceRect, rescalePlan, type PlanEx,
} from '../src/plan';
import { checkCrop, pathJitterPx, type PathProblem } from '../src/smooth';
import type { AspectChoice, SizeChoice } from '../src/types';

const SIZES: SizeChoice[] = ['source', '2.7k', '1440p', '1080p', '720p', { width: 1000 }, { width: 1001 }];
const ASPECTS: AspectChoice[] = ['source', '16:9', '4:3', '1:1', '9:16'];
const label = (s: SizeChoice) => (typeof s === 'object' ? `w${s.width}` : s);

const fixtures = new Map<string, PathFixture>();
const fx = (n: string) => {
  if (!fixtures.has(n)) fixtures.set(n, loadFixture(n));
  return fixtures.get(n)!;
};
const plans = new Map<string, PlanEx>();
/** memoized plan per (fixture, size, aspect) */
function planFor(name: string, size: SizeChoice | null, aspect: AspectChoice = 'source'): PlanEx {
  const key = `${name}|${size === null ? 'default' : label(size)}|${aspect}`;
  if (!plans.has(key)) {
    const f = fx(name);
    const output = size === null ? undefined : outputGeometry(f.tel.width, f.tel.height, size, aspect);
    plans.set(key, buildPlan(f.tel, { smoothness: 1, output }));
  }
  return plans.get(key)!;
}

/** worst kernel-exact border position (source px beyond the image, <= 0 = inside) over every `step`-th record */
function worstBorder(plan: PlanEx, step = 5, nPerEdge = 32): number {
  const uv = new Float64Array(2);
  let worst = -Infinity;
  const F = plan.framePts.length;
  for (let k = 0; k < F; k += step) {
    for (let i = 0; i <= nPerEdge; i++) {
      for (const [x, y] of [[i * (plan.outW - 1) / nPerEdge, 0], [i * (plan.outW - 1) / nPerEdge, plan.outH - 1],
        [0, i * (plan.outH - 1) / nPerEdge], [plan.outW - 1, i * (plan.outH - 1) / nPerEdge]]) {
        mapPlanPoint(plan, k, x, y, uv);
        worst = Math.max(worst, -uv[0], uv[0] - (plan.srcW - 1), -uv[1], uv[1] - (plan.srcH - 1));
      }
    }
  }
  return worst;
}

/** the path problem the plan was solved for (reference rectangle), for smooth.checkCrop */
function refProblem(f: PathFixture, plan: PlanEx): PathProblem {
  // camera rows from the plan: M = Rᵀ V  ->  R = V Mᵀ ; checkCrop only needs rows + V, so rebuild rows exactly
  const F = plan.framePts.length, nr = plan.nRows;
  const rows = new Float64Array(F * nr * 9);
  const V = new Float64Array(9);
  for (let k = 0; k < F; k++) {
    const q = plan.virtQ.subarray(4 * k, 4 * k + 4);
    const [w, x, y, z] = [q[0], q[1], q[2], q[3]];
    V.set([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]);
    for (let j = 0; j < nr; j++) {
      const o = (k * nr + j) * 9, M = plan.rowMats;
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        // R[a][b] = Σ_c V[a][c] M[b][c]
        rows[o + 3 * a + b] = V[3 * a] * M[o + 3 * b] + V[3 * a + 1] * M[o + 3 * b + 1] + V[3 * a + 2] * M[o + 3 * b + 2];
      }
    }
  }
  return { nFrames: F, fps: f.tel.fps, srcW: f.tel.width, srcH: f.tel.height, outW: plan.refW, outH: plan.refH,
    lens: plan.lens, camQ: new Float64Array(4 * F), camRows: rows, nRows: nr, segments: f.tel.segments };
}

// ------------------------------------------------------------------------------------------------ geometry
describe('outputGeometry', () => {
  const table = (w: number, h: number) => {
    const rows: string[] = [];
    for (const a of ASPECTS) {
      rows.push(`${a.padEnd(6)} ` + SIZES.map((s) => {
        const g = outputGeometry(w, h, s, a);
        return `${label(s)} ${g.outW}x${g.outH}${g.upscale ? '^' : ''}`;
      }).join(' | '));
    }
    return rows.join('\n');
  };

  it('16:9 source (3840x2160): every preset/aspect, even, never zero, exact values', () => {
    const G = (s: SizeChoice, a: AspectChoice) => { const g = outputGeometry(3840, 2160, s, a); return [g.outW, g.outH]; };
    console.log(`[geom] 3840x2160 (^ = upscale flag)\n${table(3840, 2160)}`);
    for (const a of ASPECTS) for (const s of SIZES) {
      const g = outputGeometry(3840, 2160, s, a);
      expect(g.outW % 2).toBe(0); expect(g.outH % 2).toBe(0);
      expect(g.outW).toBeGreaterThanOrEqual(2); expect(g.outH).toBeGreaterThanOrEqual(2);
      expect(g.aspect).toBe(a);
    }
    expect(G('source', 'source')).toEqual([3840, 2160]);
    expect(G('1080p', 'source')).toEqual([1920, 1080]);
    expect(G('1440p', '16:9')).toEqual([2560, 1440]);
    expect(G('720p', '16:9')).toEqual([1280, 720]);
    expect(G('2.7k', '16:9')).toEqual([2704, 1520]);
    expect(G('1080p', '4:3')).toEqual([1440, 1080]);
    expect(G('2.7k', '4:3')).toEqual([2704, 2028]);
    expect(G('source', '4:3')).toEqual([2880, 2160]);
    expect(G('1080p', '1:1')).toEqual([1080, 1080]);
    expect(G('source', '1:1')).toEqual([2160, 2160]);
    expect(G('1080p', '9:16')).toEqual([1080, 1920]);
    expect(G('720p', '9:16')).toEqual([720, 1280]);
    expect(G('2.7k', '9:16')).toEqual([1520, 2704]);
    expect(G('source', '9:16')).toEqual([1214, 2160]);
    expect(G({ width: 1000 }, '16:9')).toEqual([1000, 562]);
    expect(G({ width: 1001 }, '16:9')).toEqual([1000, 562]);
    expect(G({ width: 1000 }, '9:16')).toEqual([1000, 1776]);
    // upscale flag: more pixels than the largest rectangle of that aspect inside the source
    expect(outputGeometry(3840, 2160, '1080p', '16:9').upscale).toBe(false);
    expect(outputGeometry(3840, 2160, '1440p', '9:16').upscale).toBe(true);    // 1440 > 1214 wide
    expect(outputGeometry(3840, 2160, '2.7k', '1:1').upscale).toBe(true);      // 2704 > 2160
    expect(outputGeometry(3840, 2160, { width: 5000 }, 'source').upscale).toBe(true);
    expect(outputGeometry(3840, 2160, '1080p', '9:16').scale).toBeCloseTo(1080 / 1214, 12);
  });

  it('4:3 source (3840x2880) and a 1080p source', () => {
    const G = (s: SizeChoice, a: AspectChoice, w = 3840, h = 2880) => { const g = outputGeometry(w, h, s, a); return [g.outW, g.outH]; };
    console.log(`[geom] 3840x2880\n${table(3840, 2880)}`);
    for (const a of ASPECTS) for (const s of SIZES) {
      const g = outputGeometry(3840, 2880, s, a);
      expect(g.outW % 2 + g.outH % 2).toBe(0);
      expect(Math.min(g.outW, g.outH)).toBeGreaterThanOrEqual(2);
    }
    expect(G('source', 'source')).toEqual([3840, 2880]);
    expect(G('1080p', 'source')).toEqual([1440, 1080]);
    expect(G('2.7k', 'source')).toEqual([2704, 2028]);
    expect(G('source', '16:9')).toEqual([3840, 2160]);
    expect(G('1080p', '16:9')).toEqual([1920, 1080]);
    expect(G('source', '1:1')).toEqual([2880, 2880]);
    expect(G('source', '9:16')).toEqual([1620, 2880]);
    expect(G('1440p', '9:16')).toEqual([1440, 2560]);
    expect(outputGeometry(3840, 2880, '1440p', '9:16').upscale).toBe(false);
    // 1080p source: 1440p / 2.7k are upscales, 720p is not; odd / tiny sources never give zero or odd sizes
    expect(outputGeometry(1920, 1080, '1440p', 'source').upscale).toBe(true);
    expect(outputGeometry(1920, 1080, '2.7k', '16:9').upscale).toBe(true);
    expect(outputGeometry(1920, 1080, '720p', '16:9').upscale).toBe(false);
    expect(G('source', 'source', 1921, 1081)).toEqual([1920, 1080]);
    expect(G('source', '9:16', 2, 2)).toEqual([2, 2]);
    expect(G({ width: 2 }, '16:9')).toEqual([2, 2]);
    // portrait source (2160x3840): 'NNNNp' is still the short side
    expect(G('1080p', 'source', 2160, 3840)).toEqual([1080, 1920]);
    expect(G('source', '16:9', 2160, 3840)).toEqual([2160, 1214]);
    expect(G('2.7k', 'source', 2160, 3840)).toEqual([1520, 2704]);
    expect(() => outputGeometry(3840, 2160, { width: 0 }, '16:9')).toThrow();
    expect(() => outputGeometry(0, 2160, '1080p', '16:9')).toThrow();
  });

  it('reference rectangle / aspect area fraction', () => {
    expect(referenceRect(3840, 2160, 1920, 1080)).toEqual({ refW: 3840, refH: 2160, pxScale: 0.5 });
    expect(referenceRect(3840, 2160, 1280, 720)).toEqual({ refW: 3840, refH: 2160, pxScale: 1280 / 3840 });
    expect(referenceRect(3840, 2160, 1080, 1920)).toEqual({ refW: 1215, refH: 2160, pxScale: 1080 / 1215 });
    const r = referenceRect(3840, 2160, 2704, 1520);
    expect(r.refW).toBe(3840); expect(r.refH).toBeCloseTo(2158.58, 2);
    expect(aspectAreaFraction(3840, 2160, 1920, 1080)).toBe(1);
    expect(aspectAreaFraction(3840, 2160, 1080, 1920)).toBeCloseTo(1215 / 3840, 12);
    expect(aspectAreaFraction(3840, 2880, 1920, 1080)).toBeCloseTo(0.75, 12);
  });
});

// ------------------------------------------------------------------------------------------------ plans
const CASES = ['o3_0034', 'oa4_0012_w100'] as const;

describe.each(CASES)('buildPlan output targeting %s', (name) => {
  const ok = hasFixture(name);

  it.runIf(ok)('a smaller output at the source aspect = the full-size plan with the focal scaled', () => {
    const full = planFor(name, null);
    const sizes: SizeChoice[] = ['1080p', '720p'];
    for (const s of sizes) {
      const p = planFor(name, s, 'source');
      const sc = p.outW / full.outW;
      expect(p.outH / full.outH).toBeCloseTo(sc, 12);
      expect(p.rowMats.length).toBe(full.rowMats.length);
      let dRow = 0, dFx = 0;
      for (let i = 0; i < p.rowMats.length; i++) dRow = Math.max(dRow, Math.abs(p.rowMats[i] - full.rowMats[i]));
      for (let k = 0; k < p.outFx.length; k++) dFx = Math.max(dFx, Math.abs(p.outFx[k] / (full.outFx[k] * sc) - 1));
      // kernel mapping: output pixel (x,y) -> the same source point as the full-size pixel at the same image position
      const uv = new Float64Array(2), uw = new Float64Array(2);
      let dMap = 0;
      for (let k = 0; k < full.framePts.length; k += 97) {
        for (const [x, y] of [[0, 0], [p.outW - 1, 0], [p.outW / 2, p.outH / 3], [p.outW - 1, p.outH - 1], [17, p.outH - 5]]) {
          mapPlanPoint(p, k, x, y, uv);
          mapPlanPoint(full, k, (x + 0.5) / sc - 0.5, (y + 0.5) / sc - 0.5, uw);
          dMap = Math.max(dMap, Math.hypot(uv[0] - uw[0], uv[1] - uw[1]));
        }
      }
      console.log(`[path] ${name} ${label(s)} ${p.outW}x${p.outH} vs ${full.outW}x${full.outH}: row matrices max |Δ| ` +
        `${dRow.toExponential(2)}, outFx/(full·${sc.toFixed(4)}) - 1 max ${dFx.toExponential(2)}, same-point mapping ` +
        `${dMap.toExponential(2)} src px, hfov ${p.hfovDeg.toFixed(3)} vs ${full.hfovDeg.toFixed(3)} deg`);
      expect(dRow).toBe(0);                  // identical virtual path
      expect(dFx).toBeLessThan(1e-6);        // float32 rounding of the scaled focal
      expect(dMap).toBeLessThan(2e-3);
      expect(Math.abs(p.hfovDeg - full.hfovDeg)).toBeLessThan(1e-6);
      expect(p.output.outW).toBe(p.outW);
    }
  }, 120_000);

  it.runIf(ok)('9:16 and 1:1 (and 2.7k): zero crop violations, smooth path, footprint/FOV semantics', () => {
    const f = fx(name);
    const base = planFor(name, 'source', 'source');
    const fCommon = base.fx0 * 1920 / base.outW;   // compare angular jitter in the same px (source-aspect 1080p)
    const jBase = pathJitterPx(base.virtQ, f.tel.fps, fCommon, 1.5, f.tel.segments).rms;
    const lines: string[] = [];
    const cases: Array<[SizeChoice, AspectChoice]> = [['1080p', '9:16'], ['1080p', '1:1'], ['source', '9:16'], ['2.7k', 'source']];
    if (f.tel.width * 3 === f.tel.height * 4) cases.push(['1080p', '16:9']);
    for (const [s, a] of cases) {
      const p = planFor(name, s, a);
      const worst = worstBorder(p, 3);
      const viol = checkCrop(refProblem(f, p), p.virtQ, Float64Array.from(p.outFx, (v) => v / p.pxScale), 48, 0);
      const vmax = viol.reduce((m, x) => Math.max(m, x), -Infinity);
      const j = pathJitterPx(p.virtQ, f.tel.fps, fCommon, 1.5, f.tel.segments).rms;
      let fp = 0, n = 0;
      for (let k = 0; k < p.framePts.length; k += 60) { fp += planFootprint(p, k); n++; }
      fp /= n;
      const frac = aspectAreaFraction(f.tel.width, f.tel.height, p.outW, p.outH);
      lines.push(`${label(s)}/${a} ${p.outW}x${p.outH} (ref ${p.refW}x${p.refH.toFixed(1)}): worst border ` +
        `${worst.toFixed(2)} px, crop viol ${vmax.toFixed(2)} px, HF ${j.toFixed(3)} px (source-aspect ${jBase.toFixed(3)}, ` +
        `ratio ${(j / jBase).toFixed(3)}), hfov ${p.hfovDeg.toFixed(1)} deg, footprint ${fp.toFixed(4)} = ` +
        `${(fp / frac).toFixed(3)} of the aspect's max, zoom max ${p.smoothInfo.maxZoom.toFixed(4)} ` +
        `(global ${p.smoothInfo.globalZoom.toFixed(4)}), ${p.timings.total.toFixed(2)} s`);
      expect(worst).toBeLessThanOrEqual(0);
      expect(vmax).toBeLessThanOrEqual(0);
      expect(p.smoothInfo.nFramesViolating).toBe(0);
      expect(j / jBase).toBeLessThan(1.25);                       // at least about as smooth as the source aspect
      expect(Math.abs(fp / frac - 0.60)).toBeLessThan(0.04);      // footprint relative to the aspect's rectangle
      expect(p.outW % 2 + p.outH % 2).toBe(0);
    }
    console.log(`[path] ${name} (source-aspect plan: HF ${jBase.toFixed(3)} px, hfov ${base.hfovDeg.toFixed(1)} deg)\n  ` +
      lines.join('\n  '));
  }, 180_000);

  it.runIf(ok && name === 'o3_0034')('constraint-bound: a wide footprint, 9:16 vs the source aspect', () => {
    const f = fx(name);
    const fCommon = planFor(name, null).fx0 * 1920 / f.tel.width;
    const out: string[] = [];
    const hf: Record<string, number> = {};
    for (const [a, footprint] of [['source', 0.75], ['9:16', 0.75], ['9:16', 0.85]] as const) {
      const p = buildPlan(f.tel, { smoothness: 1, footprint, output: outputGeometry(f.tel.width, f.tel.height, '1080p', a) });
      const worst = worstBorder(p, 3);
      hf[`${a}@${footprint}`] = pathJitterPx(p.virtQ, f.tel.fps, fCommon, 1.5, f.tel.segments).rms;
      out.push(`${a}@${footprint} ${p.outW}x${p.outH}: worst border ${worst.toFixed(2)} px, HF ${hf[`${a}@${footprint}`].toFixed(3)} px, ` +
        `max zoom ${p.smoothInfo.maxZoom.toFixed(4)}, violating frames ${p.smoothInfo.nFramesViolating}`);
      expect(worst).toBeLessThanOrEqual(0);
      expect(p.smoothInfo.nFramesViolating).toBe(0);
    }
    console.log(`[path] ${name} wide footprints: ${out.join('; ')}`);
    expect(hf['9:16@0.75']).toBeLessThan(hf['source@0.75']);   // the narrow aspect has more room: smoother
  }, 120_000);

  it.runIf(ok)('rescalePlan: another size of the same aspect without re-solving', () => {
    const full = planFor(name, null);
    const p720 = planFor(name, '720p', 'source');
    const r = rescalePlan(full, p720.outW, p720.outH);
    expect(r.outW).toBe(p720.outW); expect(r.outH).toBe(p720.outH);
    expect(r.rowMats).toBe(full.rowMats);
    let d = 0;
    for (let k = 0; k < r.outFx.length; k++) d = Math.max(d, Math.abs(r.outFx[k] / p720.outFx[k] - 1));
    expect(d).toBeLessThan(1e-6);
    expect(r.hfovDeg).toBeCloseTo(full.hfovDeg, 9);
    expect(() => rescalePlan(full, 1080, 1920)).toThrow(/aspect/);
  }, 60_000);

  it.runIf(ok && name === 'o3_0034')('fovDeg is the OUTPUT horizontal FOV; outFx is in output px', () => {
    const f = fx(name);
    const g = outputGeometry(f.tel.width, f.tel.height, '1080p', '9:16');
    const p = buildPlan(f.tel, { smoothness: 1, output: g, fovDeg: 60 });
    expect(p.hfovDeg).toBeCloseTo(60, 6);
    expect(p.fx0).toBeCloseTo(540 / Math.tan(Math.PI / 6), 6);
    const q = buildPlan(f.tel, { smoothness: 1, output: g, outFx: 900 });
    expect(q.fx0).toBeCloseTo(900, 9);
    expect(q.outFx[0]).toBeGreaterThanOrEqual(900 - 1e-3);
    expect(worstBorder(p, 9)).toBeLessThanOrEqual(0);
  }, 120_000);
});
