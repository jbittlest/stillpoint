/**
 * Output frame-rate schedule (owner: PLAN agent): which stabilized source frames make up each output frame.
 *
 *   const s = buildSchedule(plan.framePts, { fps: 24, timing: 'realtime', motionBlur: 'natural' });
 *   output frame i = Σ_j s.weights[j] · warp(source frame s.taps[j]),  j in [s.tapStart[i], s.tapStart[i] + s.tapsPerFrame[i])
 *   timestamp s.outPtsUs[i] (µs); source frames are plan records (presentation order), so warp(k) uses plan record k.
 *
 * Timing modes
 *   realtime   output instants T_i = t0 + i/fps over the source's duration (t0 = first source PTS, so the source
 *              timeline and the audio stay aligned; n = round(duration·fps)).
 *                motionBlur 'off'     : the NEAREST source frame (|T_i - pts_k| minimal, ties -> the earlier frame),
 *                                       then drift clean-up: a duplicate and a skip within a few frames of each other
 *                                       (PTS jitter around a nearest-frame crossing, e.g. 59.94 -> 60) are merged
 *                                       when every frame in between stays within 0.75 source frame of its instant,
 *                                       so only the duplicates / skips the rate change needs remain.
 *                motionBlur 'natural' : synthetic shutter of `shutterDeg` (default 180°) centred on T_i: every source
 *                                       frame whose time slab (midpoint to midpoint between neighbouring PTS)
 *                                       overlaps [T_i - d/2, T_i + d/2], d = shutterDeg/360/fps, weighted by the
 *                                       overlap (box filter); taps under 0.5 % are dropped, weights renormalized.
 *                                       A window not longer than one source frame (d <= 1/srcFps, e.g. 59.94 ->
 *                                       29.97 at 180°) adds no blur: nearest frame (every other frame, exactly).
 *   slowmo     every source frame, played back at `fps`: outPts = i/fps, duration n/fps; audio dropped. A rate within
 *              SAME_RATE_TOL of the source's (60 from 59.94) is no slow motion: it is exported in real time (sound kept).
 *   fps 'source' (either mode) = identity: one tap per source frame at its own PTS.
 * NTSC-style rates snap to N·1000/1001 (59.94, 29.97, 23.976, 119.88, ...).
 */
import type { OutputSchedule, RetimeParams } from './types';

/** rates closer than this (relative) play at the same speed: 59.94 <-> 60 (0.1 %) is a real-time conform, not slow
 *  motion (no sound dropped for it) */
export const SAME_RATE_TOL = 0.005;

/** taps whose normalized weight is below this are dropped (motion blur) */
export const MIN_TAP_WEIGHT = 0.005;
/** drift clean-up: max distance (output frames) between a duplicate and a skip that get merged: at least this, and
 *  up to the frames the rate drift needs to move 0.5 source frame (59.94 -> 60: 500) */
const PAIR_WINDOW_MIN = 12;
const PAIR_WINDOW_MAX = 2000;
/** drift clean-up: max |T_i - pts_k| after a merge, in local source frame periods */
const PAIR_MAX_ERR = 0.75;

/** Snap NTSC-style rates (within 0.005 fps of N·1000/1001 and not an integer rate) to N·1000/1001 exactly. */
export function normalizeFps(fps: number): number {
  if (!(fps > 0) || !Number.isFinite(fps)) throw new RangeError(`bad frame rate ${fps}`);
  const n = Math.round(fps * 1.001);
  const ntsc = n * 1000 / 1001;
  if (n > 0 && Math.abs(fps - ntsc) < 0.005 && Math.abs(fps - Math.round(fps)) > 0.02) return ntsc;
  return fps;
}

/** Source frame rate from the PTS (1 / median frame interval, NTSC-snapped); `fallback` for < 2 frames. */
export function estimateFps(framePts: ArrayLike<number>, fallback = 30): number {
  const N = framePts.length;
  if (N < 2) return fallback;
  const d: number[] = [];
  const step = Math.max(1, Math.floor((N - 1) / 4096));    // plenty for a median, O(1) memory on long clips
  for (let k = 1; k < N; k += step) {
    const x = framePts[k] - framePts[k - 1];
    if (x > 0) d.push(x);
  }
  if (!d.length) return fallback;
  d.sort((a, b) => a - b);
  const med = d[d.length >> 1];
  // mean over the clip refines the median when the PTS are regular (no drops): keeps 59.94 exact-ish
  const mean = (framePts[N - 1] - framePts[0]) / (N - 1);
  const r = Math.abs(mean / med - 1) < 0.01 ? 1 / mean : 1 / med;
  return normalizeFps(r);
}

interface Builder {
  counts: Uint8Array;
  start: Uint32Array;
  taps: number[];
  weights: number[];
}

function finish(n: number, outPts: Float64Array, fps: number, srcFps: number, b: Builder, extra: {
  speed: number; dropAudio: boolean; warnings: string[];
}): OutputSchedule {
  const taps = Int32Array.from(b.taps);
  const weights = Float32Array.from(b.weights);
  const last = new Int32Array(n), first = new Int32Array(n);
  let maxTaps = 0, maxSpan = 0, maxUses = 0;
  // taps are ascending and the frames' tap ranges are monotone: uses of one source frame are consecutive frames
  let prevTap = -1, uses = 0;
  for (let i = 0; i < n; i++) {
    const s = b.start[i], c = b.counts[i];
    first[i] = taps[s];
    last[i] = taps[s + c - 1];
    if (c > maxTaps) maxTaps = c;
    if (last[i] - first[i] + 1 > maxSpan) maxSpan = last[i] - first[i] + 1;
    // a source frame shared with the previous output frame can only be this frame's first tap
    if (taps[s] === prevTap) uses++; else uses = 1;
    if (uses > maxUses) maxUses = uses;
    if (c > 1) uses = 1;
    prevTap = taps[s + c - 1];
  }
  return {
    n, outPtsUs: outPts, frameDurUs: 1e6 / fps, tapsPerFrame: b.counts, tapStart: b.start, taps, weights,
    lastSourceFrame: last, fps, srcFps, speed: extra.speed, dropAudio: extra.dropAudio, firstSourceFrame: first,
    maxTaps, maxSpan, maxUses, warnings: extra.warnings,
  };
}

/** One tap per frame from an index array. */
function singleTaps(idx: Int32Array): Builder {
  const n = idx.length;
  const start = new Uint32Array(n);
  for (let i = 0; i < n; i++) start[i] = i;
  return { counts: new Uint8Array(n).fill(1), start, taps: Array.from(idx), weights: new Array(n).fill(1) };
}

/**
 * Nearest source frame for every output instant (ties -> earlier), then the drift clean-up (see the header).
 * Returns the index per output frame (non-decreasing).
 */
export function nearestFrames(pts: ArrayLike<number>, T: Float64Array, srcFps: number, fps: number,
                              cleanup = true): Int32Array {
  const N = pts.length, n = T.length;
  const k = new Int32Array(n);
  const tieEps = 1e-9;
  let j = 0;
  for (let i = 0; i < n; i++) {
    const t = T[i];
    while (j + 1 < N && (pts[j + 1] - t) < (t - pts[j]) - tieEps) j++;
    k[i] = j;
  }
  // ---- drift clean-up: merge a duplicate/skip pair that the rate change does not require
  const r = srcFps / fps;
  const ri = Math.round(r);
  const integer = Math.abs(r - ri) < 1e-6;
  const lo = integer ? ri : Math.floor(r), hi = integer ? ri : Math.ceil(r);
  if (!cleanup || n < 3 || N < 3) return k;
  // near a crossing the error T - pts drifts by |r - round(r)| source frames per output frame; PTS jitter makes plain
  // nearest flip back and forth over the whole zone where |error| is within the jitter of 0.5
  const drift = Math.abs(r - ri);
  const window = Math.min(PAIR_WINDOW_MAX, Math.max(PAIR_WINDOW_MIN, Math.ceil(0.5 / Math.max(drift, 1e-6))));
  const localDt = (m: number) => {
    const a = Math.max(0, m - 1), b = Math.min(N - 1, m + 1);
    return b > a ? (pts[b] - pts[a]) / (b - a) : 1 / srcFps;
  };
  const okShift = (a: number, b: number, d: number): boolean => {   // frames [a, b) shifted by d stay close enough
    for (let i = a; i < b; i++) {
      const m = k[i] + d;
      if (m < 0 || m >= N) return false;
      if (Math.abs(T[i] - pts[m]) > PAIR_MAX_ERR * localDt(m)) return false;
    }
    return true;
  };
  const inc = (i: number) => k[i] - k[i - 1];
  for (let a = 1; a < n; a++) {
    const ia = inc(a);
    if (ia >= lo && ia <= hi) continue;
    // irregular increment at a (a duplicate the rate does not need, or a skip): the nearest increment on the other
    // side (a skip, or a duplicate / short step) within the window absorbs it, both end up inside [lo, hi]
    const low = ia < lo;
    // candidates by distance; the first whose shift keeps every frame in between close enough wins
    let tries = 0;
    search: for (let dd = 1; dd <= window; dd++) {
      for (const b of [a + dd, a - dd]) {
        if (b < 1 || b >= n) continue;
        const ib = inc(b);
        if (!(low ? ib >= lo + 1 : ib <= hi - 1)) continue;
        // low at a (too few source frames advanced) + high at b: the frames between move towards the high side
        let s0: number, s1: number, d: number;
        if (low) { if (a < b) { s0 = a; s1 = b; d = 1; } else { s0 = b; s1 = a; d = -1; } }
        else { if (a < b) { s0 = a; s1 = b; d = -1; } else { s0 = b; s1 = a; d = 1; } }
        if (okShift(s0, s1, d)) { for (let i = s0; i < s1; i++) k[i] += d; break search; }
        if (++tries >= 8) break search;
      }
    }
  }
  return k;
}

/**
 * Output schedule for retiming `framePts` (source presentation times, s, ascending) to `p` (see the header).
 */
export function buildSchedule(framePts: Float64Array | ArrayLike<number>, p: RetimeParams): OutputSchedule {
  const N = framePts.length;
  const warnings: string[] = [];
  for (let k = 1; k < N; k++) {
    if (!(framePts[k] >= framePts[k - 1])) throw new RangeError(`buildSchedule: framePts not ascending at ${k}`);
  }
  const srcFps = estimateFps(framePts, typeof p.fps === 'number' ? normalizeFps(p.fps) : 30);
  const fps = p.fps === 'source' ? srcFps : normalizeFps(p.fps);
  const timing = p.timing ?? 'realtime';
  const blur = p.motionBlur === 'natural';
  if (N === 0) {
    const e = new Uint8Array(0);
    return finish(0, new Float64Array(0), fps, srcFps, { counts: e, start: new Uint32Array(0), taps: [], weights: [] },
      { speed: 1, dropAudio: false, warnings });
  }
  const t0 = framePts[0];

  // ---- identity: every source frame at its own time
  if (p.fps === 'source') {
    if (blur) warnings.push('motion blur needs a lower output frame rate than the source: none applied');
    const idx = Int32Array.from({ length: N }, (_, i) => i);
    const outPts = Float64Array.from({ length: N }, (_, i) => Math.round(framePts[i] * 1e6));
    return finish(N, outPts, fps, srcFps, singleTaps(idx), { speed: 1, dropAudio: false, warnings });
  }

  // ---- slow motion: every source frame at the new rate (a rate within SAME_RATE_TOL of the source's: real time)
  if (timing === 'slowmo' && Math.abs(fps / srcFps - 1) >= SAME_RATE_TOL) {
    const speed = fps / srcFps;
    if (blur) warnings.push('motion blur does not apply to slow motion (every source frame is shown): none applied');
    warnings.push(speed < 1
      ? `slow motion ${(1 / speed).toFixed(2)}x: audio dropped`
      : `plays ${speed.toFixed(2)}x faster than real time: audio dropped`);
    const idx = Int32Array.from({ length: N }, (_, i) => i);
    const outPts = Float64Array.from({ length: N }, (_, i) => Math.round(i / fps * 1e6));
    return finish(N, outPts, fps, srcFps, singleTaps(idx), { speed, dropAudio: true, warnings });
  }

  // ---- real time
  const dtN = 1 / srcFps;
  const D = (framePts[N - 1] - t0) + dtN;
  const n = Math.max(1, Math.floor(D * fps + 0.5));
  const T = new Float64Array(n);
  const outPts = new Float64Array(n);
  for (let i = 0; i < n; i++) { T[i] = t0 + i / fps; outPts[i] = Math.round(T[i] * 1e6); }
  if (fps > srcFps * 1.01) warnings.push(`output rate ${fps.toFixed(3)} > source ${srcFps.toFixed(3)}: frames repeat`);

  // motion blur: box-filter the source frames' time slabs over the shutter window. A window no longer than one source
  // frame cannot add blur (each source frame already stands for a whole frame interval), only cross-fade two
  // frames: then the nearest frame is used (59.94 -> 29.97 at 180° = every other frame, exactly).
  const shutter = Math.min(360, Math.max(1e-3, p.shutterDeg ?? 180));
  const d = shutter / 360 / fps;
  if (blur && N > 1 && d <= dtN * 1.001) {
    warnings.push(`motion blur: the ${shutter}° shutter at ${fps.toFixed(3)} fps is not longer than a source frame: none applied`);
  }
  if (!blur || N === 1 || d <= dtN * 1.001) {
    const idx = nearestFrames(framePts, T, srcFps, fps);
    return finish(n, outPts, fps, srcFps, singleTaps(idx), { speed: 1, dropAudio: false, warnings });
  }
  const slabLo = (k: number) => (k === 0 ? framePts[0] - 0.5 * (framePts[1] - framePts[0]) : 0.5 * (framePts[k - 1] + framePts[k]));
  const slabHi = (k: number) => (k === N - 1 ? framePts[N - 1] + 0.5 * (framePts[N - 1] - framePts[N - 2]) : 0.5 * (framePts[k] + framePts[k + 1]));
  const b: Builder = { counts: new Uint8Array(n), start: new Uint32Array(n), taps: [], weights: [] };
  const nearest = nearestFrames(framePts, T, srcFps, fps);
  let j0 = 0;
  const kk: number[] = [], ww: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = T[i] - 0.5 * d, e = T[i] + 0.5 * d;
    while (j0 + 1 < N && slabHi(j0) <= a) j0++;
    kk.length = 0; ww.length = 0;
    let sum = 0;
    for (let k = j0; k < N; k++) {
      const lo = slabLo(k);
      if (lo >= e) break;
      const ov = Math.min(e, slabHi(k)) - Math.max(a, lo);
      if (ov > 1e-12) { kk.push(k); ww.push(ov); sum += ov; }
    }
    if (sum > 0) {
      // drop negligible taps, renormalize
      let s2 = 0;
      for (let m = 0; m < kk.length; m++) if (ww[m] / sum >= MIN_TAP_WEIGHT) s2 += ww[m];
      if (s2 <= 0) { kk.length = 0; }
      else {
        let w = 0;
        for (let m = 0; m < kk.length; m++) if (ww[m] / sum >= MIN_TAP_WEIGHT) { kk[w] = kk[m]; ww[w] = ww[m] / s2; w++; }
        kk.length = w; ww.length = w;
      }
    }
    if (!kk.length) { kk.push(nearest[i]); ww.push(1); }       // window outside every slab (clip ends)
    if (kk.length > 255) throw new RangeError(`buildSchedule: ${kk.length} taps per output frame (max 255)`);
    b.start[i] = b.taps.length;
    b.counts[i] = kk.length;
    // exact passthrough when there is one tap
    if (kk.length === 1) ww[0] = 1;
    for (let m = 0; m < kk.length; m++) { b.taps.push(kk[m]); b.weights.push(ww[m]); }
  }
  return finish(n, outPts, fps, srcFps, b, { speed: 1, dropAudio: false, warnings });
}

/**
 * For each source frame, the last output frame that uses it (-1 = unused: decode it but skip the warp).
 * Streaming: after output frame i is emitted, warped source frames with lastUse <= i can be released.
 */
export function sourceLastUse(s: OutputSchedule, nSrc: number): Int32Array {
  const last = new Int32Array(nSrc).fill(-1);
  for (let i = 0; i < s.n; i++) {
    const a = s.tapStart[i], c = s.tapsPerFrame[i];
    for (let j = a; j < a + c; j++) { const k = s.taps[j]; if (k >= 0 && k < nSrc) last[k] = i; }
  }
  return last;
}
