/**
 * RETIME tests (owner: PLAN agent): buildSchedule (web/src/retime.ts) on ideal 59.94 PTS, the real O3 DJI_0034 /
 * OA4 0012 fixture PTS (OA4 window starts at t0 = 100 s) and jittered PTS (90 kHz timescale + random jitter).
 *
 *   cd web && npx vitest run test/retime.test.ts
 */
import { describe, expect, it } from 'vitest';
import { buildSchedule, estimateFps, nearestFrames, normalizeFps, sourceLastUse } from '../src/retime';
import type { OutputSchedule, RetimeParams } from '../src/types';
import { hasFixture, loadFixture } from './path.fixtures';

const NTSC60 = 60000 / 1001;

/** ideal PTS at `fps` (s) starting at t0 */
const ideal = (N: number, fps: number, t0 = 0) => Float64Array.from({ length: N }, (_, k) => t0 + k / fps);

/** PTS on a 90 kHz timescale with uniform random jitter of ±jitterFrac frame periods (seeded) */
function jittered(N: number, fps: number, jitterFrac: number, seed = 7): Float64Array {
  let s = seed;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647) * 2 - 1;
  const p = new Float64Array(N);
  for (let k = 0; k < N; k++) p[k] = Math.round((k / fps + rnd() * jitterFrac / fps) * 90000) / 90000;
  for (let k = 1; k < N; k++) p[k] = Math.max(p[k], p[k - 1] + 1 / 90000);
  return p;
}

/** structural invariants of every schedule (collects violations, one expect: fast on 30k-frame schedules) */
function validate(s: OutputSchedule, N: number): { maxTaps: number; meanTaps: number } {
  const bad: string[] = [];
  const chk = (c: boolean, m: () => string) => { if (!c && bad.length < 10) bad.push(m()); };
  chk(s.outPtsUs.length === s.n && s.tapsPerFrame.length === s.n && s.tapStart.length === s.n &&
    s.lastSourceFrame.length === s.n, () => 'array lengths');
  let maxTaps = 0, tot = 0;
  for (let i = 0; i < s.n; i++) {
    const a = s.tapStart[i], c = s.tapsPerFrame[i];
    chk(c >= 1, () => `frame ${i}: no taps`);
    if (i + 1 < s.n) chk(s.tapStart[i + 1] === a + c, () => `frame ${i}: tapStart`);
    let w = 0, mx = -1;
    for (let j = a; j < a + c; j++) {
      chk(s.taps[j] >= 0 && s.taps[j] < N, () => `frame ${i}: tap ${s.taps[j]} out of range`);
      if (j > a) chk(s.taps[j] > s.taps[j - 1], () => `frame ${i}: taps not ascending`);
      chk(s.weights[j] > 0, () => `frame ${i}: weight ${s.weights[j]}`);
      w += s.weights[j];
      mx = Math.max(mx, s.taps[j]);
    }
    chk(Math.abs(w - 1) < 1e-6, () => `frame ${i}: weights sum ${w}`);
    chk(s.lastSourceFrame[i] === mx, () => `frame ${i}: lastSourceFrame`);
    if (i > 0) {
      chk(s.lastSourceFrame[i] >= s.lastSourceFrame[i - 1], () => `frame ${i}: lastSourceFrame not monotone`);
      chk(s.firstSourceFrame![i] >= s.firstSourceFrame![i - 1], () => `frame ${i}: firstSourceFrame not monotone`);
      chk(s.outPtsUs[i] > s.outPtsUs[i - 1], () => `frame ${i}: outPts not increasing`);
    }
    chk(Number.isInteger(s.outPtsUs[i]), () => `frame ${i}: outPts not integer µs`);
    maxTaps = Math.max(maxTaps, c); tot += c;
  }
  chk(s.taps.length === a0(s) && s.weights.length === a0(s), () => 'taps/weights length');
  const uses = new Int32Array(N);
  for (let j = 0; j < s.taps.length; j++) if (s.taps[j] >= 0 && s.taps[j] < N) uses[s.taps[j]]++;
  const mu = s.n ? uses.reduce((m, x) => Math.max(m, x), 0) : 0;
  chk(s.maxUses === mu, () => `maxUses ${s.maxUses} != ${mu}`);
  expect(bad).toEqual([]);
  return { maxTaps, meanTaps: tot / Math.max(1, s.n) };
}
const a0 = (s: OutputSchedule) => (s.n ? s.tapStart[s.n - 1] + s.tapsPerFrame[s.n - 1] : 0);

/** increments of a one-tap schedule's source index */
function incStats(s: OutputSchedule) {
  const h = new Map<number, number>();
  for (let i = 1; i < s.n; i++) {
    const d = s.taps[s.tapStart[i]] - s.taps[s.tapStart[i - 1]];
    h.set(d, (h.get(d) ?? 0) + 1);
  }
  return h;
}

/** brute-force nearest source frame, ties -> earlier */
function bruteNearest(pts: Float64Array, t: number): number {
  let best = 0;
  for (let k = 1; k < pts.length; k++) if (Math.abs(pts[k] - t) < Math.abs(pts[best] - t) - 1e-9) best = k;
  return best;
}

const sources: Array<[string, () => Float64Array]> = [
  ['ideal 59.94 (3029 frames)', () => ideal(3029, NTSC60)],
  ...(hasFixture('o3_0034') ? [['O3 DJI_0034 PTS', () => loadFixture('o3_0034').tel.framePts] as [string, () => Float64Array]] : []),
  ...(hasFixture('oa4_0012_w100') ? [['OA4 0012 PTS (t0 100 s)', () => loadFixture('oa4_0012_w100').tel.framePts] as [string, () => Float64Array]] : []),
  ['90 kHz + ±15% jitter', () => jittered(3001, NTSC60, 0.15)],
];

describe('frame rates', () => {
  it('NTSC rates snap to N·1000/1001, integer rates stay', () => {
    expect(normalizeFps(59.94)).toBe(60000 / 1001);
    expect(normalizeFps(29.97)).toBe(30000 / 1001);
    expect(normalizeFps(23.976)).toBe(24000 / 1001);
    expect(normalizeFps(119.88)).toBe(120000 / 1001);
    expect(normalizeFps(60000 / 1001)).toBe(60000 / 1001);
    for (const f of [60, 50, 30, 25, 24, 48, 59.9, 12.5]) expect(normalizeFps(f)).toBe(f);
    expect(estimateFps(ideal(500, NTSC60))).toBe(NTSC60);
    expect(estimateFps(ideal(500, 25, 3))).toBe(25);
    expect(() => normalizeFps(0)).toThrow();
  });
});

describe.each(sources)('buildSchedule on %s', (_label, mk) => {
  const pts = mk();
  const N = pts.length;
  const t0 = pts[0];
  const D = pts[N - 1] - pts[0] + 1 / NTSC60;

  it('59.94 -> 59.94 (numeric and "source") is the identity, with or without blur', () => {
    for (const p of [{ fps: 59.94 }, { fps: 'source' as const }, { fps: 59.94, motionBlur: 'natural' as const }]) {
      const s = buildSchedule(pts, { timing: 'realtime', motionBlur: 'off', ...p });
      validate(s, N);
      expect(s.n).toBe(N);
      for (let i = 0; i < N; i++) {
        expect(s.tapsPerFrame[i]).toBe(1);
        expect(s.taps[i]).toBe(i);
        expect(s.weights[i]).toBe(1);
      }
      expect(s.dropAudio).toBe(false);
    }
  });

  it('59.94 -> 29.97: every other frame exactly (nearest and 180° blur)', () => {
    for (const motionBlur of ['off', 'natural'] as const) {
      const s = buildSchedule(pts, { fps: 29.97, timing: 'realtime', motionBlur });
      validate(s, N);
      expect(s.n).toBe(Math.ceil(N / 2));
      let multi = 0;
      for (let i = 0; i < s.n; i++) {
        if (s.tapsPerFrame[i] !== 1) multi++;
        expect(s.taps[s.tapStart[i]]).toBe(2 * i);
      }
      // jittered PTS: the 1-frame shutter window can clip a neighbour's slab by a few % -> a second (small) tap
      if (!_label.includes('jitter')) expect(multi).toBe(0);
      expect(Math.abs(s.outPtsUs[1] - s.outPtsUs[0] - 1e6 * 1001 / 30000)).toBeLessThanOrEqual(1);
    }
  });

  it('59.94 -> 24 realtime nearest: brute-force nearest, 2/3 cadence, real-time duration', () => {
    const s = buildSchedule(pts, { fps: 24, timing: 'realtime', motionBlur: 'off' });
    validate(s, N);
    expect(s.n).toBe(Math.floor(D * 24 + 0.5));
    let maxErr = 0, mism = 0;
    for (let i = 0; i < s.n; i++) {
      const T = t0 + i / 24;
      if (s.taps[i] !== bruteNearest(pts, T)) mism++;
      maxErr = Math.max(maxErr, Math.abs(pts[s.taps[i]] - T));
      expect(Math.abs(s.outPtsUs[i] - Math.round(T * 1e6))).toBe(0);
    }
    const inc = incStats(s);
    console.log(`[retime] ${_label} 59.94->24 nearest: n ${s.n}, increments ${JSON.stringify([...inc])}, ` +
      `max |T - pts| ${(maxErr * 1e3).toFixed(2)} ms, mismatches vs brute force ${mism}`);
    if (!_label.includes('jitter')) {
      expect(mism).toBe(0);
      expect([...inc.keys()].sort()).toEqual([2, 3]);
      expect(maxErr).toBeLessThanOrEqual(0.5 / NTSC60 + 1e-9);
    }
    expect(Math.abs(s.n / 24 - D)).toBeLessThanOrEqual(0.5 / 24 + 1e-9);
    expect(s.dropAudio).toBe(false);
  });

  it('59.94 -> 24 realtime with natural motion blur (180°, 360°; 90° is shorter than a source frame)', () => {
    for (const shutterDeg of [180, 360, 90]) {
      const s = buildSchedule(pts, { fps: 24, timing: 'realtime', motionBlur: 'natural', shutterDeg });
      const v = validate(s, N);
      const d = shutterDeg / 360 / 24;
      // time centroid of the taps vs the output instant, and total covered time inside the clip
      let maxC = 0;
      for (let i = 2; i < s.n - 2; i++) {
        let c = 0;
        for (let j = s.tapStart[i]; j < s.tapStart[i] + s.tapsPerFrame[i]; j++) c += s.weights[j] * pts[s.taps[j]];
        maxC = Math.max(maxC, Math.abs(c - (t0 + i / 24)));
      }
      console.log(`[retime] ${_label} 59.94->24 blur ${shutterDeg}°: taps mean ${v.meanTaps.toFixed(3)} (expect ~` +
        `${(1 + d * NTSC60).toFixed(3)}), max ${v.maxTaps}, maxSpan ${s.maxSpan}, max |centroid - T| ${(maxC * 1e3).toFixed(2)} ms`);
      if (d * NTSC60 <= 1) {
        // a window no longer than a source frame adds no blur: nearest frame, and the user is told
        expect(v.maxTaps).toBe(1);
        expect(s.warnings!.join(' ')).toMatch(/none applied/);
      } else {
        expect(v.maxTaps).toBeLessThanOrEqual(Math.ceil(d * NTSC60) + 1);
        expect(Math.abs(v.meanTaps - (1 + d * NTSC60))).toBeLessThan(0.1);
        expect(maxC).toBeLessThan(0.25 / NTSC60);
      }
      expect(s.n).toBe(Math.floor(D * 24 + 0.5));
    }
  });

  it('59.94 -> 60: only the necessary duplicates, no skips; 60 -> 59.94 on the reverse', () => {
    const s = buildSchedule(pts, { fps: 60, timing: 'realtime', motionBlur: 'off' });
    validate(s, N);
    const inc = incStats(s);
    const dups = inc.get(0) ?? 0;
    let skips = 0;
    for (const [d, c] of inc) if (d > 1) skips += (d - 1) * c;
    const raw = nearestFrames(pts, Float64Array.from({ length: s.n }, (_, i) => t0 + i / 60), s.srcFps!, 60, false);
    let rawDup = 0, rawSkip = 0;
    for (let i = 1; i < raw.length; i++) { const d = raw[i] - raw[i - 1]; if (d === 0) rawDup++; if (d > 1) rawSkip += d - 1; }
    console.log(`[retime] ${_label} 59.94->60: n ${s.n} (source ${N}), duplicates ${dups}, skips ${skips} ` +
      `(plain nearest: ${rawDup} dup / ${rawSkip} skip)`);
    expect(skips).toBe(0);
    expect(dups).toBe(s.n - N);          // every source frame is shown; exactly n - N repeats
    expect(s.taps[s.n - 1]).toBe(N - 1);
    // reverse direction on a 60 fps version of the same timeline
    const p60 = Float64Array.from(pts, (t) => t0 + (t - t0) * NTSC60 / 60);
    const r = buildSchedule(p60, { fps: 59.94, timing: 'realtime', motionBlur: 'off' });
    validate(r, N);
    const ri = incStats(r);
    let rs = 0;
    for (const [d, c] of ri) if (d > 1) rs += (d - 1) * c;
    console.log(`[retime] ${_label} 60->59.94: n ${r.n}, duplicates ${ri.get(0) ?? 0}, skips ${rs}`);
    expect(ri.get(0) ?? 0).toBe(0);
    expect(rs).toBe(N - r.n);
  });

  it('slowmo 59.94 -> 24: every source frame, duration n/24, audio dropped', () => {
    const s = buildSchedule(pts, { fps: 24, timing: 'slowmo', motionBlur: 'natural' });
    validate(s, N);
    expect(s.n).toBe(N);
    for (let i = 0; i < N; i++) {
      expect(s.taps[i]).toBe(i);
      expect(s.outPtsUs[i]).toBe(Math.round(i / 24 * 1e6));
    }
    expect((s.outPtsUs[N - 1] + s.frameDurUs) / 1e6).toBeCloseTo(N / 24, 5);
    expect(s.dropAudio).toBe(true);
    expect(s.speed).toBeCloseTo(24 / NTSC60, 12);
    expect(s.warnings!.join(' ')).toMatch(/motion blur/);
  });

  it('sourceLastUse: skipped frames are marked unused, release order follows the output', () => {
    const s = buildSchedule(pts, { fps: 24, timing: 'realtime', motionBlur: 'off' });
    const lu = sourceLastUse(s, N);
    let used = 0;
    for (let k = 0; k < N; k++) if (lu[k] >= 0) used++;
    expect(used).toBe(new Set(Array.from(s.taps)).size);
    for (let k = 1; k < N; k++) if (lu[k] >= 0 && lu[k - 1] >= 0) expect(lu[k]).toBeGreaterThanOrEqual(lu[k - 1]);
  });
});

describe('edge cases and runtime', () => {
  it('empty, single-frame and custom rates', () => {
    const e = buildSchedule(new Float64Array(0), { fps: 24, timing: 'realtime', motionBlur: 'off' });
    expect(e.n).toBe(0);
    const one = buildSchedule(Float64Array.of(0.5), { fps: 30, timing: 'realtime', motionBlur: 'natural' });
    validate(one, 1);
    expect(one.n).toBe(1);
    expect(one.outPtsUs[0]).toBe(500000);
    // 25 fps PAL source -> 23.976 and 50 -> 25 blur (2 taps of 0.5 each at 360°)
    const pal = ideal(250, 50);
    const b = buildSchedule(pal, { fps: 25, timing: 'realtime', motionBlur: 'natural', shutterDeg: 360 });
    validate(b, 250);
    for (let i = 1; i < b.n - 1; i++) {
      expect(b.tapsPerFrame[i]).toBe(3);   // centred 1/25 s window: half, full, half slab
      expect(b.weights[b.tapStart[i] + 1]).toBeCloseTo(0.5, 6);
    }
    const c = buildSchedule(pal, { fps: 23.976, timing: 'realtime', motionBlur: 'off' });
    validate(c, 250);
    expect(c.fps).toBe(24000 / 1001);
    expect(() => buildSchedule(Float64Array.of(1, 0.5), { fps: 24, timing: 'realtime', motionBlur: 'off' })).toThrow();
  });

  it('30k frames: every mode in a few ms', () => {
    const pts = jittered(30000, NTSC60, 0.05);
    const modes: RetimeParams[] = [
      { fps: 24, timing: 'realtime', motionBlur: 'off' },
      { fps: 24, timing: 'realtime', motionBlur: 'natural' },
      { fps: 60, timing: 'realtime', motionBlur: 'off' },
      { fps: 29.97, timing: 'realtime', motionBlur: 'natural', shutterDeg: 360 },
      { fps: 24, timing: 'slowmo', motionBlur: 'off' },
      { fps: 'source', timing: 'realtime', motionBlur: 'off' },
    ];
    const ms: string[] = [];
    for (const m of modes) {
      const t = performance.now();
      const s = buildSchedule(pts, m);
      const dt = performance.now() - t;
      ms.push(`${m.fps}/${m.timing}/${m.motionBlur}: ${dt.toFixed(1)} ms (n ${s.n})`);
      expect(dt).toBeLessThan(500);
      validate(s, pts.length);
    }
    console.log(`[retime] 30k frames: ${ms.join('; ')}`);
  }, 60_000);

  it('buffer sizing table (maxTaps / maxSpan) for the export pipeline', () => {
    const pts = ideal(6000, NTSC60);
    const rows: string[] = [];
    for (const fps of [24, 25, 29.97, 30, 50]) {
      for (const shutterDeg of [180, 360]) {
        const s = buildSchedule(pts, { fps, timing: 'realtime', motionBlur: 'natural', shutterDeg });
        validate(s, pts.length);
        rows.push(`${fps}@${shutterDeg}°: taps<=${s.maxTaps} span<=${s.maxSpan} uses<=${s.maxUses}`);
        expect(s.maxTaps).toBeLessThanOrEqual(4);
        expect(s.maxUses).toBeLessThanOrEqual(2);         // add-as-you-go blending needs <= 2 accumulators
      }
    }
    console.log(`[retime] 59.94 source, natural blur: ${rows.join(', ')}`);
  });
});
