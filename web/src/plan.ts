/**
 * Telemetry -> render Plan (owner: PATH agent). TypeScript port of engine/stillpoint/plan_build.py + the path
 * optimizer (smooth.ts), browser + Node.
 *
 *   orientation      q_cam(t) = slerp on the uniform IMU grid (geom.slerp_series), video timeline
 *   row times        t_kj = frameT[k] + readout·((y_j+0.5)/H - 0.5),  y_j = j·(H-1)/(nRows-1)
 *   exposure avg.    high-rate gyro: each row uses the orientation averaged over its exposure window
 *                    (plan_build.exposure_averaged_q, 9 mid-point taps; window ramp: none <= 2 ms, full >= 3.5 ms)
 *   row matrices     M_kj = R(q_row)ᵀ · R(virt_k)   (output ray -> source camera ray; the renderer lerps rows)
 *   virtual path     smooth.ts optimizePath (crop-constrained, exact renderer mapping)
 *
 * Output: rectilinear, the source size (even) or `output` (any size / aspect, see outputGeometry), widest focal from
 * `outFx` | `fovDeg` | `footprint` (default 0.60 of the source area at identity; for another aspect: of the largest
 * rectangle of that aspect inside the source), zoom only where the crop is otherwise infeasible (<= maxZoom).
 *
 * Output targeting: the path is solved for the REFERENCE rectangle = the output rectangle scaled to the source's
 * pixel density (the largest rectangle of the output's aspect inside the source), so every output size of one
 * aspect gets the SAME virtual path and row matrices; only outFx scales (x outW/refW). The crop constraint uses the
 * output's own aspect (a 9:16 output from a 16:9 source has lots of horizontal room, the same vertical room as the
 * 16:9 default).
 */
import type { AspectChoice, Lens, OutputGeometry, Plan, SizeChoice, Telemetry } from './types';
import { fxForFootprint, hfovToFx, identityFootprint, project } from './lens';
import { OrientationSeries, qconjMulInto, qmulInto, qnormalizeInPlace, quatToMatInto } from './so3';
import { optimizePath, type PathProblem, type SmoothInfo, type SmoothOptions } from './smooth';

export interface StabParams {
  /** 0..2, default 1 */
  smoothness: number;
  /** horizontal field of view of the output (deg); overrides footprint */
  fovDeg?: number;
  /** source area fraction used by the output at identity, default ~0.60 */
  footprint?: number;
  horizonLock?: boolean;
  /** output frame size / aspect (outputGeometry); default = the source size (even-rounded). `fovDeg` is the OUTPUT's
   *  horizontal FOV; `footprint` is relative to the largest rectangle of the output aspect inside the source (so the
   *  default 0.60 means the same crop tightness for every aspect); `outFx` is in output px. */
  output?: OutputGeometry;
  // ---- extensions (optional; nothing in the shared contract depends on them)
  /** explicit widest output focal (output px); overrides fovDeg/footprint */
  outFx?: number;
  /** max zoom factor over the widest focal where the crop needs it (default 1.5; 1 disables zoom) */
  maxZoom?: number;
  /** plan row samples (default 32) */
  nRows?: number;
  /** exposure-averaged row orientations with a high-rate gyro (default true, as the engine) */
  exposureAvg?: boolean;
  /** advanced solver overrides */
  smooth?: Partial<SmoothOptions>;
  log?: (msg: string) => void;
}

/** buildPlan's result: the shared Plan plus diagnostics (extra fields; consumers may ignore them). */
export interface PlanEx extends Plan {
  /** F*4 virtual camera->world orientation per output frame */
  virtQ: Float64Array;
  /** widest output focal (px) and its horizontal FOV (deg) */
  fx0: number;
  hfovDeg: number;
  exposureAvg: boolean;
  smoothInfo: SmoothInfo;
  timings: Record<string, number>;
  /** reference rectangle the path was solved for (source pixel density; may be fractional for rounded outputs) and
   *  the output scale outW/refW (outFx = solver focal x pxScale) */
  refW: number;
  refH: number;
  pxScale: number;
  /** the output geometry used (outputGeometry(src, 'source', 'source') when none was requested) */
  output: OutputGeometry;
}

export const DEFAULT_FOOTPRINT = 0.60;
export const AVG_E0_S = 0.0020;
export const AVG_E1_S = 0.0035;
const EXPOSURE_TAPS = 9;

/** Averaging window length (s) for a frame exposure e (s): none <= 2 ms, full >= 3.5 ms (plan_build.exposure_avg_window). */
export function exposureAvgWindow(e: number): number {
  let x = Number.isFinite(e) ? e : 0;
  x = Math.min(Math.max(x, 0), 0.05);
  return x * Math.min(Math.max((x - AVG_E0_S) / (AVG_E1_S - AVG_E0_S), 0), 1);
}

export function orientationSeries(tel: Telemetry): OrientationSeries {
  return new OrientationSeries(tel.imuT, tel.imuQ);
}

/** Row sample y_j = j·(H-1)/(nRows-1). */
export function rowSampleY(srcH: number, nRows: number, j: number): number {
  return j * (srcH - 1) / (nRows - 1);
}

/** Would build_plan use exposure averaging for these frames? (plan_build.build_plan use_avg) */
export function useExposureAvg(tel: Telemetry, frames?: ArrayLike<number>, enabled = true): boolean {
  const ex = tel.exposureS;
  const F = tel.framePts.length;
  if (!enabled || !tel.hasHighrate || !ex || ex.length !== F) return false;
  let mx = 0;
  const n = frames ? frames.length : F;
  for (let i = 0; i < n; i++) mx = Math.max(mx, exposureAvgWindow(ex[frames ? frames[i] : i]));
  return mx * tel.imuRate >= 1.0;
}

/**
 * Camera orientation of every plan row sample, as camera->world rotation matrices (F*nRows*9 row-major) into
 * `out` (Float32Array for the renderer path, Float64Array for exact tests). Exposure-averaged when `avg`.
 */
export function cameraRowMats(tel: Telemetry, ser: OrientationSeries, nRows: number, avg: boolean,
                              out: Float32Array | Float64Array, frames?: ArrayLike<number>,
                              onProgress?: (f: number) => void): void {
  const H = tel.height, readout = tel.readoutS;
  const F = frames ? frames.length : tel.framePts.length;
  const q = new Float64Array(4), qc = new Float64Array(4), qs = new Float64Array(4), M = new Float64Array(9);
  const uu = Array.from({ length: EXPOSURE_TAPS }, (_, i) => (i + 0.5) / EXPOSURE_TAPS - 0.5);
  const rowOff = Array.from({ length: nRows }, (_, j) => readout * ((rowSampleY(H, nRows, j) + 0.5) / H - 0.5));
  for (let i = 0; i < F; i++) {
    const k = frames ? frames[i] : i;
    const ft = tel.frameT[k];
    const e = avg ? exposureAvgWindow(tel.exposureS[k]) : 0;
    for (let j = 0; j < nRows; j++) {
      const t = ft + rowOff[j];
      ser.at(t, qc, 0);
      if (avg && e > 0) {
        qs[0] = qs[1] = qs[2] = qs[3] = 0;
        for (let m = 0; m < EXPOSURE_TAPS; m++) {
          ser.at(t + e * uu[m], q, 0);
          const d = q[0] * qc[0] + q[1] * qc[1] + q[2] * qc[2] + q[3] * qc[3];
          const s = d < 0 ? -1 : 1;
          qs[0] += s * q[0]; qs[1] += s * q[1]; qs[2] += s * q[2]; qs[3] += s * q[3];
        }
        qnormalizeInPlace(qs, 0);
        quatToMatInto(M, 0, qs, 0);
      } else {
        quatToMatInto(M, 0, qc, 0);
      }
      const o = (i * nRows + j) * 9;
      for (let c = 0; c < 9; c++) out[o + c] = M[c];
    }
    if (onProgress && (i & 1023) === 0) onProgress(i / F);
  }
}

/**
 * Row matrices M_kj = R_kjᵀ · R(virt_k) from camera row rotations (F*nRows*9) and the virtual path (F*4).
 * `out` may be the same array as `cam` (in-place).
 */
export function rowMatsFromCam(cam: Float32Array | Float64Array, virtQ: Float64Array, nRows: number,
                               out: Float32Array | Float64Array): void {
  const F = virtQ.length >> 2;
  const Vm = new Float64Array(9);
  for (let k = 0; k < F; k++) {
    quatToMatInto(Vm, 0, virtQ, 4 * k);
    for (let j = 0; j < nRows; j++) {
      const o = (k * nRows + j) * 9;
      const r0 = cam[o], r1 = cam[o + 1], r2 = cam[o + 2], r3 = cam[o + 3], r4 = cam[o + 4], r5 = cam[o + 5];
      const r6 = cam[o + 6], r7 = cam[o + 7], r8 = cam[o + 8];
      for (let c = 0; c < 3; c++) {
        const v0 = Vm[c], v1 = Vm[3 + c], v2 = Vm[6 + c];
        out[o + c] = r0 * v0 + r3 * v1 + r6 * v2;
        out[o + 3 + c] = r1 * v0 + r4 * v1 + r7 * v2;
        out[o + 6 + c] = r2 * v0 + r5 * v1 + r8 * v2;
      }
    }
  }
}

/** Exact float64 row matrices for a given virtual path (plan_build.build_plan's row_mats); for tests/tools. */
export function rowMatricesF64(tel: Telemetry, virtQ: Float64Array, nRows = 32, frames?: ArrayLike<number>,
                               exposureAvg = true): { rowMats: Float64Array; exposureAvg: boolean } {
  const ser = orientationSeries(tel);
  const avg = useExposureAvg(tel, frames, exposureAvg);
  const F = frames ? frames.length : tel.framePts.length;
  const cam = new Float64Array(F * nRows * 9);
  cameraRowMats(tel, ser, nRows, avg, cam, frames);
  const vq = new Float64Array(4 * F);
  for (let i = 0; i < F; i++) {
    for (let c = 0; c < 4; c++) vq[4 * i + c] = virtQ[4 * i + c];
    qnormalizeInPlace(vq, 4 * i);
  }
  rowMatsFromCam(cam, vq, nRows, cam);
  return { rowMats: cam, exposureAvg: avg };
}

/** Horizon lock: gravity 'down' in the camera world per frame (F*3), or undefined without gravity data. */
function gravityPerFrame(tel: Telemetry, ser: OrientationSeries): Float64Array | undefined {
  if (!tel.gravityQ || tel.gravityQ.length !== tel.imuQ.length) return undefined;
  const gser = new OrientationSeries(tel.imuT, tel.gravityQ);
  const F = tel.framePts.length;
  const out = new Float64Array(3 * F);
  const qi = new Float64Array(4), qg = new Float64Array(4), d = new Float64Array(4), G = new Float64Array(9);
  for (let k = 0; k < F; k++) {
    ser.at(tel.frameT[k], qi, 0);
    gser.at(tel.frameT[k], qg, 0);
    // G = R(qg · conj(qi)) : imu-world -> gravity-world ; g_w = Gᵀ e_z
    const ci = [qi[0], -qi[1], -qi[2], -qi[3]];
    qmulInto(d, 0, qg, 0, ci, 0);
    quatToMatInto(G, 0, d, 0);
    out[3 * k] = G[6]; out[3 * k + 1] = G[7]; out[3 * k + 2] = G[8];
  }
  return out;
}

// ================================================================================================= output geometry

const ASPECTS: Record<Exclude<AspectChoice, 'source'>, [number, number]> = {
  '16:9': [16, 9], '4:3': [4, 3], '1:1': [1, 1], '9:16': [9, 16],
};
/** short side (px) of the 'NNNNp' presets; '2.7k' is a LONG side */
const SHORT_SIDE: Record<string, number> = { '1440p': 1440, '1080p': 1080, '720p': 720 };
const LONG_27K = 2704;

/** largest even integer <= x (>= 2); exact for the integer ratios used here */
function evenFloor(x: number): number {
  return Math.max(2, 2 * Math.floor(x / 2 + 1e-9));
}

/** aspect as an integer ratio [w, h] (the even-rounded source size for 'source') */
export function aspectRatio(srcW: number, srcH: number, aspect: AspectChoice): [number, number] {
  if (aspect === 'source') return [evenFloor(srcW), evenFloor(srcH)];
  const r = ASPECTS[aspect];
  if (!r) throw new RangeError(`unknown aspect '${aspect}'`);
  return r;
}

/**
 * Output frame size for a size preset and an aspect ratio (all dims even, >= 2):
 *   'source'              the source size (even) at 'source' aspect, else the largest rectangle of the aspect that
 *                         fits inside the source (3840x2160 -> 9:16 1214x2160, 1:1 2160x2160, 4:3 2880x2160)
 *   '1440p'/'1080p'/'720p' the SHORT side: height of landscape/square outputs (1080p -> 1920x1080 at 16:9,
 *                         1440x1080 at 4:3, 1080x1080 at 1:1), width of portrait ones (9:16 -> 1080x1920)
 *   '2.7k'                the LONG side 2704: 2704x1520 at 16:9, 2704x2028 at 4:3, 2704x2704 at 1:1, 1520x2704 at 9:16
 *   {width}               explicit width (even-floored), height from the aspect
 * The derived side is floored to even (2704*9/16 = 1521 -> 1520). `scale` = outW / width of the 'source' rectangle
 * of the aspect; `upscale` flags outputs larger than that (more pixels than the source can resolve: warn).
 */
export function outputGeometry(srcW: number, srcH: number, size: SizeChoice, aspect: AspectChoice): OutputGeometry {
  if (!(srcW >= 2 && srcH >= 2 && Number.isFinite(srcW) && Number.isFinite(srcH))) {
    throw new RangeError(`outputGeometry: bad source size ${srcW}x${srcH}`);
  }
  const [rn, rd] = aspectRatio(srcW, srcH, aspect);
  const sw = evenFloor(srcW), sh = evenFloor(srcH);
  const portrait = rn < rd;
  // largest rectangle of the aspect inside the source
  let refW: number, refH: number;
  if (aspect === 'source') { refW = sw; refH = sh; }
  else if (rn * sh >= rd * sw) { refW = sw; refH = Math.min(sh, evenFloor(sw * rd / rn)); }   // wider than the source
  else { refH = sh; refW = Math.min(sw, evenFloor(sh * rn / rd)); }                          // narrower
  let W: number, H: number;
  if (size === 'source') { W = refW; H = refH; }
  else if (typeof size === 'object' && size !== null) {
    const w = Number(size.width);
    if (!(w >= 2) || !Number.isFinite(w)) throw new RangeError(`outputGeometry: bad custom width ${size.width}`);
    W = evenFloor(w); H = evenFloor(W * rd / rn);
  } else if (size === '2.7k') {
    if (portrait) { H = LONG_27K; W = evenFloor(LONG_27K * rn / rd); }
    else { W = LONG_27K; H = evenFloor(LONG_27K * rd / rn); }
  } else {
    const n = SHORT_SIDE[size as string];
    if (!n) throw new RangeError(`outputGeometry: unknown size '${String(size)}'`);
    if (portrait) { W = n; H = evenFloor(n * rd / rn); }
    else { H = n; W = evenFloor(n * rn / rd); }
  }
  const scale = W / refW;
  return { outW: W, outH: H, aspect, scale, upscale: scale > 1 + 1e-9 };
}

/**
 * Reference rectangle of an output (outW x outH) for a source: the output rectangle scaled to the largest size that
 * fits the (even) source. Exactly the even source size when the output has the source's aspect; otherwise one side
 * is the source's and the other may be fractional (2704x1520 from 3840x2160 -> 3840x2158.58).
 */
export function referenceRect(srcW: number, srcH: number, outW: number, outH: number): { refW: number; refH: number; pxScale: number } {
  const sw = evenFloor(srcW), sh = evenFloor(srcH);
  let refW: number, refH: number;
  if (outW * sh === outH * sw) { refW = sw; refH = sh; }
  else if (outW * sh > outH * sw) { refW = sw; refH = sw * outH / outW; }
  else { refH = sh; refW = sh * outW / outH; }
  const snap = (v: number) => (Math.abs(v - Math.round(v)) < 1e-9 ? Math.round(v) : v);
  refW = snap(refW); refH = snap(refH);
  return { refW, refH, pxScale: outW / refW };
}

/** Widest output focal (px of the rectangle outW x outH) for the given parameters. `footprint` is scaled by the
 *  fraction of the source area the largest rectangle of this aspect covers (1 at the source's aspect). */
export function outputFocal(tel: Telemetry, p: StabParams, outW = tel.width, outH = tel.height): number {
  if (p.outFx && p.outFx > 0) return p.outFx;
  if (p.fovDeg && p.fovDeg > 0) return hfovToFx(p.fovDeg, outW);
  const a = p.footprint && p.footprint > 0 ? p.footprint : DEFAULT_FOOTPRINT;
  return fxForFootprint(tel.lens, tel.width, tel.height, outW, outH, a * aspectAreaFraction(tel.width, tel.height, outW, outH));
}

/** Area fraction of the (even) source covered by the largest rectangle of the aspect outW:outH (1 at the source's). */
export function aspectAreaFraction(srcW: number, srcH: number, outW: number, outH: number): number {
  const { refW, refH } = referenceRect(srcW, srcH, outW, outH);
  return (refW * refH) / (evenFloor(srcW) * evenFloor(srcH));
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

export function buildPlan(tel: Telemetry, p: StabParams, onProgress?: (f: number) => void): PlanEx {
  const T0 = now();
  const timings: Record<string, number> = {};
  if (tel.eisBaked) {
    throw new Error('In-camera EIS (RockSteady/HorizonSteady) is baked into this clip: the gyro no longer ' +
      'describes the pixels. Stillpoint needs footage recorded with EIS off.');
  }
  const F = tel.framePts.length;
  const srcW = tel.width, srcH = tel.height;
  const geo: OutputGeometry = p.output
    ? { ...p.output }
    : outputGeometry(srcW, srcH, 'source', 'source');
  if (!(geo.outW >= 2 && geo.outH >= 2)) throw new RangeError(`buildPlan: bad output size ${geo.outW}x${geo.outH}`);
  const outW = evenFloor(geo.outW), outH = evenFloor(geo.outH);   // 4:2:0 encoders need even sizes
  // the path is solved at the source's pixel density (reference rectangle); the output focal is that x pxScale
  const { refW, refH, pxScale } = referenceRect(srcW, srcH, outW, outH);
  const nRows = Math.max(2, Math.round(p.nRows ?? 32));
  const lens: Lens = { ...tel.lens, k: [...tel.lens.k] as [number, number, number, number],
    width: tel.lens.width || srcW, height: tel.lens.height || srcH };
  const fx0Ref = p.outFx && p.outFx > 0 ? p.outFx / pxScale : outputFocal(tel, p, refW, refH);
  const maxZoom = Math.max(1, p.maxZoom ?? 1.5);
  const prog = (a: number, b: number) => (f: number) => onProgress?.(a + (b - a) * f);

  // ---- camera orientation: frame centres + plan rows
  let t = now();
  const ser = orientationSeries(tel);
  const camQ = new Float64Array(4 * F);
  for (let k = 0; k < F; k++) ser.at(tel.frameT[k], camQ, 4 * k);
  const avg = useExposureAvg(tel, undefined, p.exposureAvg ?? true);
  const rows = new Float32Array(F * nRows * 9);
  cameraRowMats(tel, ser, nRows, avg, rows, undefined, prog(0, 0.2));
  timings.rows = (now() - t) / 1000;

  // ---- virtual path
  t = now();
  const prob: PathProblem = {
    nFrames: F, fps: tel.fps, srcW, srcH, outW: refW, outH: refH, lens, camQ, camRows: rows, nRows,
    segments: tel.segments, gravityW: p.horizonLock ? gravityPerFrame(tel, ser) : undefined,
  };
  const res = optimizePath(prob, {
    smoothness: p.smoothness ?? 1, fx0: fx0Ref, fxMax: fx0Ref * maxZoom, allowZoom: maxZoom > 1,
    horizonLock: !!p.horizonLock, onProgress: prog(0.2, 0.97), log: p.log, ...(p.smooth ?? {}),
  });
  timings.path = (now() - t) / 1000;

  // ---- row matrices (in place: camera rows -> M = Rᵀ V)
  t = now();
  rowMatsFromCam(rows, res.virtQ, nRows, rows);
  timings.rowMats = (now() - t) / 1000;
  timings.total = (now() - T0) / 1000;
  onProgress?.(1);
  const outFx = new Float32Array(F);
  for (let k = 0; k < F; k++) outFx[k] = res.outFx[k] * pxScale;
  const fx0 = fx0Ref * pxScale;
  return {
    srcW, srcH, outW, outH, lens, framePts: Float64Array.from(tel.framePts), outFx,
    nRows, rowMats: rows, readoutS: tel.readoutS,
    virtQ: res.virtQ, fx0, hfovDeg: 2 * Math.atan(outW / 2 / fx0) * 180 / Math.PI, exposureAvg: avg,
    smoothInfo: res.info, timings, refW, refH, pxScale,
    output: { ...geo, outW, outH },
  };
}

/**
 * The same plan for another output size of the same aspect (e.g. a small preview of a 4K export plan): identical
 * virtual path and row matrices, outFx scaled by outW / plan.outW. Throws when the aspect differs by more than the
 * even-rounding tolerance (a different aspect needs its own buildPlan: the crop constraint depends on it).
 */
export function rescalePlan<P extends Plan>(plan: P, outW: number, outH: number): P {
  const w = evenFloor(outW), h = evenFloor(outH);
  const a0 = plan.outW / plan.outH, a1 = w / h;
  const tol = 2 / Math.min(w, h, plan.outW, plan.outH) + 1e-9;   // one even-rounding step on either side
  if (Math.abs(a1 / a0 - 1) > tol) {
    throw new RangeError(`rescalePlan: ${w}x${h} has another aspect than ${plan.outW}x${plan.outH}; build a new plan`);
  }
  const sc = w / plan.outW;
  const outFx = new Float32Array(plan.outFx.length);
  for (let k = 0; k < outFx.length; k++) outFx[k] = plan.outFx[k] * sc;
  const ex = plan as unknown as Partial<PlanEx>;
  const extra: Partial<PlanEx> = {};
  if (typeof ex.fx0 === 'number') {
    extra.fx0 = ex.fx0 * sc;
    extra.hfovDeg = 2 * Math.atan(w / 2 / extra.fx0) * 180 / Math.PI;
  }
  if (typeof ex.pxScale === 'number' && typeof ex.refW === 'number') extra.pxScale = w / ex.refW;
  if (ex.output) extra.output = { ...ex.output, outW: w, outH: h, scale: typeof ex.refW === 'number' ? w / ex.refW : undefined,
    upscale: typeof ex.refW === 'number' ? w / ex.refW > 1 + 1e-9 : undefined };
  return { ...plan, ...extra, outW: w, outH: h, outFx };
}

// ================================================================================================= diagnostics

/**
 * Exact source footprint (fraction of the source area) of plan record k: the output image's pixel-edge border
 * (nPerEdge samples per edge) mapped through the plan exactly like the kernel (3-evaluation RS iteration), polygon
 * clipped to the source rectangle.
 */
export function planFootprint(plan: Plan, k: number, nPerEdge = 41): number {
  const pts: number[] = [];
  const W = plan.outW, H = plan.outH;
  const edge = (x0: number, y0: number, x1: number, y1: number) => {
    for (let i = 0; i < nPerEdge; i++) { const a = i / nPerEdge; pts.push(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a); }
  };
  edge(-0.5, -0.5, W - 0.5, -0.5); edge(W - 0.5, -0.5, W - 0.5, H - 0.5);
  edge(W - 0.5, H - 0.5, -0.5, H - 0.5); edge(-0.5, H - 0.5, -0.5, -0.5);
  const poly: number[] = [];
  const uv = new Float64Array(2);
  for (let i = 0; i < pts.length; i += 2) {
    mapPlanPoint(plan, k, pts[i], pts[i + 1], uv);
    poly.push(uv[0], uv[1]);
  }
  const clipped = clipPolygon(poly, -0.5, -0.5, plan.srcW - 0.5, plan.srcH - 0.5);
  let a = 0;
  const n = clipped.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    a += clipped[2 * i] * clipped[2 * j + 1] - clipped[2 * i + 1] * clipped[2 * j];
  }
  return 0.5 * Math.abs(a) / (plan.srcW * plan.srcH);
}

/** The kernel's mapping of output pixel (x,y) of plan record k -> source pixel (float64 on the f32 plan). */
export function mapPlanPoint(plan: Plan, k: number, x: number, y: number, out: Float64Array): number {
  const fo = plan.outFx[k], nr = plan.nRows, M = plan.rowMats, srch = plan.srcH;
  const rx = (x - (plan.outW - 1) / 2) / fo, ry = (y - (plan.outH - 1) / 2) / fo;
  let yy = 0.5 * (srch - 1), ya = 0, ga = 0, z = 1;
  for (let e = 0; e < 3; e++) {
    const g = Math.min(Math.max(yy * (nr - 1) / (srch - 1), 0), nr - 1);
    const j0 = Math.min(Math.floor(g), nr - 2);
    const f = g - j0;
    const A = (k * nr + j0) * 9, B = A + 9;
    const ax = M[A] * rx + M[A + 1] * ry + M[A + 2], ay = M[A + 3] * rx + M[A + 4] * ry + M[A + 5];
    const az = M[A + 6] * rx + M[A + 7] * ry + M[A + 8];
    const bx = M[B] * rx + M[B + 1] * ry + M[B + 2], by = M[B + 3] * rx + M[B + 4] * ry + M[B + 5];
    const bz = M[B + 6] * rx + M[B + 7] * ry + M[B + 8];
    const X = ax + (bx - ax) * f, Y = ay + (by - ay) * f;
    z = az + (bz - az) * f;
    project(plan.lens, X, Y, z, out, 0);
    const gg = out[1] - yy;
    let yn = out[1];
    if (e >= 1) {
      const d = yy - ya;
      if (d !== 0) { const s = (gg - ga) / d; if (s <= -0.1 && s >= -2.0) yn = yy - gg / s; }
    }
    ya = yy; ga = gg; yy = yn;
  }
  return z;
}

/** Sutherland–Hodgman clip of a flat polygon to an axis-aligned rectangle. */
function clipPolygon(poly: number[], x0: number, y0: number, x1: number, y1: number): number[] {
  let P = poly;
  const clip = (inside: (x: number, y: number) => boolean, inter: (ax: number, ay: number, bx: number, by: number) => [number, number]) => {
    const out: number[] = [];
    const n = P.length / 2;
    for (let i = 0; i < n; i++) {
      const ax = P[2 * i], ay = P[2 * i + 1];
      const j = (i + n - 1) % n;
      const bx = P[2 * j], by = P[2 * j + 1];
      const ia = inside(ax, ay), ib = inside(bx, by);
      if (ia) { if (!ib) out.push(...inter(bx, by, ax, ay)); out.push(ax, ay); }
      else if (ib) out.push(...inter(bx, by, ax, ay));
    }
    P = out;
  };
  const lerpX = (xc: number) => (ax: number, ay: number, bx: number, by: number): [number, number] =>
    [xc, ay + (by - ay) * (xc - ax) / (bx - ax)];
  const lerpY = (yc: number) => (ax: number, ay: number, bx: number, by: number): [number, number] =>
    [ax + (bx - ax) * (yc - ay) / (by - ay), yc];
  clip((x) => x >= x0, lerpX(x0));
  clip((x) => x <= x1, lerpX(x1));
  clip((_, y) => y >= y0, lerpY(y0));
  clip((_, y) => y <= y1, lerpY(y1));
  return P;
}

export { identityFootprint, fxForFootprint, hfovToFx };
