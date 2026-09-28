/**
 * Crop-constrained virtual camera path for the browser (owner: PATH agent).
 *
 * Same problem as the Python engine's smooth.py (Clarabel SQP), solved without a QP solver:
 *
 *   V_k = R_k · Exp(phi_k)                (virtual camera, correction space around the real camera R_k)
 *   objective (output px, f = widest output focal):
 *       Σ w_acc_l1·|a_t| + w_jerk_l1·|j_t| + w_acc_l2·|a_t|² + w_vel_l2·|w_t|² + w_fid·|f·phi_k|² [+ horizon]
 *       w_t = f·Log(V_tᵀV_{t+1}) (body-frame angular velocity, px/frame), a = Δw, j = Δ²w
 *   HARD crop constraints: every output-border sample maps inside the source (minus a margin) through EXACTLY the
 *   renderer's mapping (plan row matrices lerped over source rows, rolling-shutter row found by the kernel's
 *   3-evaluation fixed-point/secant iteration, KB4 fisheye).
 *
 * Solver (all O(F), banded):
 *   SQP outer loop (trust radii 0.25, 0.1, 0.04, 0.015, 0.006 rad, then violation-sized repairs): linearize around
 *   the current path (V ← V·Exp(u/f)) with exact per-frame SO(3) Jacobians (Jr⁻¹/Jl⁻¹ on the velocity, Jr⁻¹(phi) on
 *   the fidelity) and exact crop-border Jacobians incl. the RS row coupling; crop rows pruned by reach (smooth.py).
 *   Inner loop (per linearization, <= 10 steps): Newton/IRLS on the linearized problem:
 *     - L1 terms by IRLS (group L1 per frame = rotation invariant, Huber-smoothed with ε continued 0.02 -> 0.001 px
 *       over the SQP iterations): the quadratic surrogate majorizes the objective (MM),
 *     - crop rows by a C² exterior penalty (ρ = 1e4 per source px², curvature ramp over the last 1 px inside the
 *       margin: no active-set cycling; the solution stays ~τ inside the margin),
 *     - the trust region as a box by projected Newton (Bertsekas: bound-held variables drop out of the step),
 *     - exact Gauss-Newton Hessian Wᵀ(Q⊗I3)W + per-frame 3x3 blocks -> ONE banded Cholesky (half-bandwidth 11) per
 *       step, projected Armijo line search on the exact merit.
 *   Every iterate is checked with the exact (renderer) mapping; a final dense border check adds constraint samples
 *   where the sparse samples missed a bulge. The result is a lower value of smooth.py's own objective than the
 *   Python engine reaches (its 4 small trust-region steps stop short), with the same constraints.
 *   Zoom: (1) if the requested field of view does not fit most frames even when the virtual camera follows the real
 *   one, a CONSTANT zoom (98th-percentile need) is applied up front; (2) where the crop is still infeasible after the
 *   rotation solve (rolling shutter during violent moves), the minimal per-frame zoom is found by bisection on the
 *   exact mapping and the zoom profile is its gap-bridged (2 s), slope-limited (4%/s) envelope: slow monotone ramps
 *   and plateaus, no breathing; the path is then repaired at that zoom.
 *
 * Units: rotations in output px (u = f·delta), crop rows in SOURCE px, weights "per output pixel" as smooth.py.
 */
import type { Lens } from './types';
import { project, projectJac } from './lens';
import {
  jrInvInto, qconjMulInto, qexpInto, qfixSignInPlace, qlogInto, qmulInto, qnormalizeInPlace, quatToMatInto,
} from './so3';

// ================================================================================================= interfaces

export interface PathProblem {
  nFrames: number;
  fps: number;
  srcW: number;
  srcH: number;
  outW: number;
  outH: number;
  lens: Lens;
  /** F*4 camera->world orientation at frameT (centre row, mid exposure) */
  camQ: Float64Array;
  /** F*nRows*9 camera->world row-major rotation of each plan row sample (the rows the renderer lerps) */
  camRows: Float32Array | Float64Array;
  nRows: number;
  /** continuous shots [first, lastInclusive]; the path never couples frames across them */
  segments?: Array<[number, number]>;
  /** F*3 gravity 'down' direction in the camera world (horizon lock), optional */
  gravityW?: Float64Array;
}

export interface SmoothOptions {
  /** 0 = follow the camera closely, 1 = cinematic FPV default, 2 = very floaty */
  smoothness?: number;
  /** widest output focal (output px) */
  fx0: number;
  /** tightest allowed zoom (default 1.5·fx0) */
  fxMax?: number;
  allowZoom?: boolean;
  horizonLock?: boolean;
  /** source-pixel margin kept inside the image (resampling kernel + border sampling), default 8 */
  marginPx?: number;
  /** max |d log(zoom)/dt| in 1/s, default 0.04 (as smooth.py) */
  maxZoomRate?: number;
  /** zoom plateaus bridge gaps shorter than this (s), default 2 */
  zoomHoldS?: number;
  // ---- weights (px units); defaults derived from smoothness exactly as smooth.SmoothParams
  wAccL1?: number;
  wJerkL1?: number;
  wAccL2?: number;
  wJerkL2?: number;
  wVelL2?: number;
  wFidelity?: number;
  wHorizon?: number;
  prox?: number;
  // ---- solver
  /** per-SQP-iteration trust region (rad, box on the rotation step), default [0.25, 0.1, 0.04, 0.015, 0.006] */
  trustRad?: number[];
  /** main SQP iterations (default trustRad.length) and extra repair iterations while the crop is violated (3) */
  sqpIters?: number;
  repairIters?: number;
  /** Newton/IRLS steps per SQP iteration (default 10) */
  innerIters?: number;
  /** crop violation (source px beyond the margin) accepted without repair/zoom, default 0.25 */
  violTolPx?: number;
  /** stiffness of the crop penalty (objective units per source px²; violation at the optimum ≈ force/ρ) */
  rhoCrop?: number;
  /** Huber/IRLS smoothing of the L1 terms (px) at the end, and its per-SQP-iteration continuation */
  epsL1?: number;
  epsSchedule?: number[];
  /** inner-loop convergence tolerance (max |Δu| px) per SQP iteration */
  tolSchedule?: number[];
  /** crop rows further than this from the border (source px) are not linearized */
  reachCapPx?: number;
  /** width (source px) of the crop penalty's curvature ramp inside the boundary */
  tauCropPx?: number;
  onProgress?: (f: number) => void;
  log?: (msg: string) => void;
  /** debug: called after every inner iteration with the max step (px) and details */
  debugInner?: (it: number, maxStepPx: number, detail: Record<string, number>) => void;
}

export interface SmoothInfo {
  runtimeS: number;
  fx0: number;
  fxMax: number;
  sqpIters: number;
  innerIters: number;
  /** final per-frame max crop violation (source px beyond the margin; <= 0 is inside), fixed+adaptive samples */
  maxViolationPx: number;
  nFramesViolating: number;
  fracBinding: number;
  maxPhiDeg: number;
  zoomChanges: number;
  maxZoom: number;
  zoomFrames: number;
  infeasibleFrames: number;
  /** constant zoom applied because the requested field of view does not fit most frames (1 = none) */
  globalZoom: number;
  /** number of separate local zoom ramps */
  zoomEvents: number;
  /** largest |∂p/∂delta|_1 seen on a border sample / source focal (JAC_BOUND sanity) */
  jacBound: number;
  timings: Record<string, number>;
  iters: Array<Record<string, number>>;
}

export interface SmoothResult {
  virtQ: Float64Array;
  outFx: Float64Array;
  info: SmoothInfo;
}

// ================================================================================================= weights

export function smoothWeights(o: SmoothOptions) {
  const s = Math.max(0, o.smoothness ?? 1);
  return {
    accL1: o.wAccL1 ?? 1.0,
    jerkL1: o.wJerkL1 ?? 10.0,
    accL2: o.wAccL2 ?? 30.0 * Math.pow(10, s - 1),
    jerkL2: o.wJerkL2 ?? 0.0,
    velL2: o.wVelL2 ?? 1e-4 * s,
    fid: o.wFidelity ?? 1e-4 * Math.pow(10, 2 * (1 - s)),
    horizon: o.wHorizon ?? 50.0,
    prox: o.prox ?? 1e-3,
  };
}

// ================================================================================================= border samples

/** 4 corners + n interior points per edge (smooth.border_samples order). Returns flat (x,y) pairs. */
export function borderSamples(outW: number, outH: number, nLong = 3, nShort = 3): Float64Array {
  const W1 = outW - 1, H1 = outH - 1;
  const [nx, ny] = outW >= outH ? [nLong, nShort] : [nShort, nLong];
  const pts: number[] = [0, 0, W1, 0, W1, H1, 0, H1];
  for (let i = 1; i <= nx; i++) pts.push(i / (nx + 1) * W1, 0);
  for (let i = 1; i <= nx; i++) pts.push(i / (nx + 1) * W1, H1);
  for (let i = 1; i <= ny; i++) pts.push(0, i / (ny + 1) * H1);
  for (let i = 1; i <= ny; i++) pts.push(W1, i / (ny + 1) * H1);
  return Float64Array.from(pts);
}

interface Edge { idx: number[]; ax: 0 | 1; fixed: number; comp: 0 | 1; sgn: number }

/** Per output edge: sample indices along it (incl. corners), varying axis, fixed coordinate, normal component, outward sign. */
function edgeLayout(outW: number, outH: number, nLong = 3, nShort = 3): Edge[] {
  const [nx, ny] = outW >= outH ? [nLong, nShort] : [nShort, nLong];
  const r = (a: number, n: number) => Array.from({ length: n }, (_, i) => a + i);
  return [
    { idx: [0, ...r(4, nx), 1], ax: 0, fixed: 0, comp: 1, sgn: -1 },
    { idx: [3, ...r(4 + nx, nx), 2], ax: 0, fixed: outH - 1, comp: 1, sgn: 1 },
    { idx: [0, ...r(4 + 2 * nx, ny), 3], ax: 1, fixed: 0, comp: 0, sgn: -1 },
    { idx: [1, ...r(4 + 2 * nx + ny, ny), 2], ax: 1, fixed: outW - 1, comp: 0, sgn: 1 },
  ];
}

/** Dense border (nPerEdge per edge, corners included on every edge) as flat (x,y) pairs (smooth.check_crop). */
export function denseBorder(outW: number, outH: number, n: number): Float64Array {
  const out = new Float64Array(8 * n);
  let m = 0;
  for (let i = 0; i < n; i++) { out[m++] = i * (outW - 1) / (n - 1); out[m++] = 0; }
  for (let i = 0; i < n; i++) { out[m++] = i * (outW - 1) / (n - 1); out[m++] = outH - 1; }
  for (let i = 0; i < n; i++) { out[m++] = 0; out[m++] = i * (outH - 1) / (n - 1); }
  for (let i = 0; i < n; i++) { out[m++] = outW - 1; out[m++] = i * (outH - 1) / (n - 1); }
  return out;
}

// ================================================================================================= exact mapping

/**
 * The renderer's output->source mapping (shaders/warp.metal sp_source_coord, in float64): output pixel ->
 * virtual ray -> world (V) -> source camera through the row matrices lerped at source row y, with y found by
 * 3 evaluations (centre row, fixed point, secant) -> KB4. Optional Jacobians w.r.t. a right-multiplied rotation
 * delta of V (px/rad, 2x3) and the log-zoom (px), including the rolling-shutter row coupling.
 */
export class CropMapper {
  readonly P: PathProblem;
  private readonly ocx: number;
  private readonly ocy: number;
  private readonly scale: number;
  private readonly nr: number;
  private readonly H: number;
  private readonly J = new Float64Array(6);
  private readonly uv = new Float64Array(2);
  private readonly kb4: boolean;
  private readonly lfx: number;
  private readonly lfy: number;
  private readonly lcx: number;
  private readonly lcy: number;
  private readonly k1: number;
  private readonly k2: number;
  private readonly k3: number;
  private readonly k4: number;
  /** result of the last map(): u, v, rayZ, Gd (6, row-major 2x3), Gz (2) */
  readonly r = new Float64Array(11);

  constructor(P: PathProblem) {
    this.P = P;
    const L = P.lens;
    this.kb4 = L.model !== 'pinhole';
    this.lfx = L.fx; this.lfy = L.fy; this.lcx = L.cx; this.lcy = L.cy;
    this.k1 = L.k[0]; this.k2 = L.k[1]; this.k3 = L.k[2]; this.k4 = L.k[3];
    this.ocx = (P.outW - 1) / 2;
    this.ocy = (P.outH - 1) / 2;
    this.nr = P.nRows;
    this.H = P.srcH;
    this.scale = (P.nRows - 1) / (P.srcH - 1);
  }

  /** Map output pixel (px,py) of frame k with virtual rotation matrix Vm[vo..vo+8] and focal fo. */
  map(k: number, Vm: Float64Array, vo: number, fo: number, px: number, py: number, jac: boolean, farPx = 0): void {
    const rows = this.P.camRows, L = this.P.lens, nr = this.nr, uv = this.uv, r = this.r;
    const dx = (px - this.ocx) / fo, dy = (py - this.ocy) / fo;
    const wx = Vm[vo] * dx + Vm[vo + 1] * dy + Vm[vo + 2];
    const wy = Vm[vo + 3] * dx + Vm[vo + 4] * dy + Vm[vo + 5];
    const wz = Vm[vo + 6] * dx + Vm[vo + 7] * dy + Vm[vo + 8];
    let y = 0.5 * (this.H - 1), ya = 0, ga = 0;
    let X = 0, Y = 0, Z = 0, fr = 0, base = 0, clamped = false;
    let Xa = 0, Ya = 0, Za = 0, Xb = 0, Yb = 0, Zb = 0, u0 = 0, v0 = 0;
    const kb = k * nr * 9;
    const kb4 = this.kb4, k1 = this.k1, k2 = this.k2, k3 = this.k3, k4 = this.k4;
    const lfx = this.lfx, lfy = this.lfy, lcx = this.lcx, lcy = this.lcy;
    for (let e = 0; e < 3; e++) {
      let g = y * this.scale;
      clamped = false;
      if (g < 0) { g = 0; clamped = true; } else if (g > nr - 1) { g = nr - 1; clamped = true; }
      let j0 = g | 0;
      if (j0 > nr - 2) j0 = nr - 2;
      fr = g - j0;
      base = kb + 9 * j0;
      // Xa = Aᵀ w, Xb = Bᵀ w  (A = camera row j0 -> camera->world, so Aᵀ maps world -> camera)
      Xa = rows[base] * wx + rows[base + 3] * wy + rows[base + 6] * wz;
      Ya = rows[base + 1] * wx + rows[base + 4] * wy + rows[base + 7] * wz;
      Za = rows[base + 2] * wx + rows[base + 5] * wy + rows[base + 8] * wz;
      Xb = rows[base + 9] * wx + rows[base + 12] * wy + rows[base + 15] * wz;
      Yb = rows[base + 10] * wx + rows[base + 13] * wy + rows[base + 16] * wz;
      Zb = rows[base + 11] * wx + rows[base + 14] * wy + rows[base + 17] * wz;
      X = Xa + (Xb - Xa) * fr; Y = Ya + (Yb - Ya) * fr; Z = Za + (Zb - Za) * fr;
      if (kb4) {   // inlined lens.project (KB4), same formula
        const rxy = Math.sqrt(X * X + Y * Y);
        const th = Math.atan2(rxy, Z);
        const t2 = th * th;
        const thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
        const sc = rxy < 1e-12 ? 1.0 / (Z > 1e-12 ? Z : 1e-12) : thd / rxy;
        u0 = lfx * X * sc + lcx; v0 = lfy * Y * sc + lcy;
      } else {
        project(L, X, Y, Z, uv, 0);
        u0 = uv[0]; v0 = uv[1];
      }
      // value-only callers may accept the 2-evaluation estimate when the point is far inside the source anyway
      if (e === 1 && farPx > 0 && !jac && u0 > farPx && v0 > farPx && u0 < this.P.srcW - 1 - farPx && v0 < this.H - 1 - farPx) break;
      const gg = v0 - y;
      let yn = v0;
      if (e >= 1) {
        const d = y - ya;
        if (d !== 0) {
          const s = (gg - ga) / d;
          if (s <= -0.1 && s >= -2.0) yn = y - gg / s;
        }
      }
      ya = y; ga = gg; y = yn;
    }
    r[0] = u0; r[1] = v0; r[2] = Z;
    if (!jac) return;
    const J = this.J;
    projectJac(L, X, Y, Z, uv, 0, J, 0);
    // interpolated camera rotation R~ (camera->world) at the final row
    const a0 = base, b0 = base + 9;
    const R0 = rows[a0] + (rows[b0] - rows[a0]) * fr, R1 = rows[a0 + 1] + (rows[b0 + 1] - rows[a0 + 1]) * fr;
    const R2 = rows[a0 + 2] + (rows[b0 + 2] - rows[a0 + 2]) * fr, R3 = rows[a0 + 3] + (rows[b0 + 3] - rows[a0 + 3]) * fr;
    const R4 = rows[a0 + 4] + (rows[b0 + 4] - rows[a0 + 4]) * fr, R5 = rows[a0 + 5] + (rows[b0 + 5] - rows[a0 + 5]) * fr;
    const R6 = rows[a0 + 6] + (rows[b0 + 6] - rows[a0 + 6]) * fr, R7 = rows[a0 + 7] + (rows[b0 + 7] - rows[a0 + 7]) * fr;
    const R8 = rows[a0 + 8] + (rows[b0 + 8] - rows[a0 + 8]) * fr;
    // columns of ∂w/∂delta = V (e_i × d):  e_x×d = (0,-1,dy), e_y×d = (1,0,-dx), e_z×d = (-dy,dx,0); zoom: V(-dx,-dy,0)
    const V0 = Vm[vo], V1 = Vm[vo + 1], V2 = Vm[vo + 2], V3 = Vm[vo + 3], V4 = Vm[vo + 4], V5 = Vm[vo + 5];
    const V6 = Vm[vo + 6], V7 = Vm[vo + 7], V8 = Vm[vo + 8];
    const c0x = -V1 + V2 * dy, c0y = -V4 + V5 * dy, c0z = -V7 + V8 * dy;
    const c1x = V0 - V2 * dx, c1y = V3 - V5 * dx, c1z = V6 - V8 * dx;
    const c2x = -V0 * dy + V1 * dx, c2y = -V3 * dy + V4 * dx, c2z = -V6 * dy + V7 * dx;
    const czx = -(V0 * dx + V1 * dy), czy = -(V3 * dx + V4 * dy), czz = -(V6 * dx + V7 * dy);
    // camera space: R~ᵀ ·
    const d0x = R0 * c0x + R3 * c0y + R6 * c0z, d0y = R1 * c0x + R4 * c0y + R7 * c0z, d0z = R2 * c0x + R5 * c0y + R8 * c0z;
    const d1x = R0 * c1x + R3 * c1y + R6 * c1z, d1y = R1 * c1x + R4 * c1y + R7 * c1z, d1z = R2 * c1x + R5 * c1y + R8 * c1z;
    const d2x = R0 * c2x + R3 * c2y + R6 * c2z, d2y = R1 * c2x + R4 * c2y + R7 * c2z, d2z = R2 * c2x + R5 * c2y + R8 * c2z;
    const dzx = R0 * czx + R3 * czy + R6 * czz, dzy = R1 * czx + R4 * czy + R7 * czz, dzz = R2 * czx + R5 * czy + R8 * czz;
    let G00 = J[0] * d0x + J[1] * d0y + J[2] * d0z;
    let G01 = J[0] * d1x + J[1] * d1y + J[2] * d1z;
    let G02 = J[0] * d2x + J[1] * d2y + J[2] * d2z;
    let G10 = J[3] * d0x + J[4] * d0y + J[5] * d0z;
    let G11 = J[3] * d1x + J[4] * d1y + J[5] * d1z;
    let G12 = J[3] * d2x + J[4] * d2y + J[5] * d2z;
    let Gz0 = J[0] * dzx + J[1] * dzy + J[2] * dzz;
    let Gz1 = J[3] * dzx + J[4] * dzy + J[5] * dzz;
    if (!clamped) {
      // rolling-shutter coupling: p = P(theta, y(p)); dp_y = ∂P_y/(1 - g_y), dp_x = ∂P_x + g_x dp_y
      const ex = (Xb - Xa) * this.scale, ey = (Yb - Ya) * this.scale, ez = (Zb - Za) * this.scale;
      const gx = J[0] * ex + J[1] * ey + J[2] * ez;
      const gy = J[3] * ex + J[4] * ey + J[5] * ez;
      const den = Math.max(1 - gy, 0.2);
      G10 /= den; G11 /= den; G12 /= den; Gz1 /= den;
      G00 += gx * G10; G01 += gx * G11; G02 += gx * G12; Gz0 += gx * Gz1;
    }
    r[3] = G00; r[4] = G01; r[5] = G02; r[6] = G10; r[7] = G11; r[8] = G12; r[9] = Gz0; r[10] = Gz1;
  }
}

// ================================================================================================= banded SPD solver

/** Symmetric positive definite banded matrix (lower band storage) with in-place Cholesky. */
export class BandSPD {
  readonly n: number;
  readonly b: number;
  readonly W: number;
  readonly a: Float64Array;
  private readonly y: Float64Array;

  constructor(n: number, b: number) {
    this.n = n; this.b = b; this.W = b + 1;
    this.a = new Float64Array(n * this.W);
    this.y = new Float64Array(n);
  }

  clear(): void { this.a.fill(0); }

  /** A[i][j] += v for i >= j >= i-b */
  add(i: number, j: number, v: number): void { this.a[i * this.W + j - i + this.b] += v; }

  factor(): void {
    const { n, b, W, a } = this;
    for (let i = 0; i < n; i++) {
      const ri = i * W - i + b; // a[ri + j] = L[i][j]
      const j0 = i - b > 0 ? i - b : 0;
      for (let j = j0; j <= i; j++) {
        const rj = j * W - j + b;
        let s = a[ri + j];
        for (let k = j0; k < j; k++) s -= a[ri + k] * a[rj + k];
        if (j < i) a[ri + j] = s / a[rj + j];
        else a[ri + i] = Math.sqrt(s > 1e-300 ? s : 1e-300);
      }
    }
  }

  /** Solve L Lᵀ x = rhs (after factor()); x may alias rhs. */
  solve(rhs: Float64Array, x: Float64Array): void {
    const { n, b, W, a, y } = this;
    for (let i = 0; i < n; i++) {
      const ri = i * W - i + b;
      const k0 = i - b > 0 ? i - b : 0;
      let s = rhs[i];
      for (let k = k0; k < i; k++) s -= a[ri + k] * y[k];
      y[i] = s / a[ri + i];
    }
    for (let i = n - 1; i >= 0; i--) {
      const k1 = i + b < n - 1 ? i + b : n - 1;
      let s = y[i];
      for (let k = i + 1; k <= k1; k++) s -= a[k * W - k + b + i] * x[k];
      x[i] = s / a[i * W + b];
    }
  }
}

// ================================================================================================= helpers

class Growable {
  k: Int32Array; c: Float64Array; rhs: Float64Array; n = 0;
  constructor(cap = 1 << 14) { this.k = new Int32Array(cap); this.c = new Float64Array(3 * cap); this.rhs = new Float64Array(cap); }
  push(k: number, c0: number, c1: number, c2: number, rhs: number): void {
    if (this.n === this.k.length) {
      const cap = this.k.length * 2;
      const k = new Int32Array(cap); k.set(this.k); this.k = k;
      const c = new Float64Array(3 * cap); c.set(this.c); this.c = c;
      const r = new Float64Array(cap); r.set(this.rhs); this.rhs = r;
    }
    const i = this.n++;
    this.k[i] = k; this.c[3 * i] = c0; this.c[3 * i + 1] = c1; this.c[3 * i + 2] = c2; this.rhs[i] = rhs;
  }
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
/**
 * C² exterior penalty for a crop row g = c·u - rhs <= 0 (source px): zero for g <= -τ, curvature ramping 0 -> ρ over
 * the last τ px inside the boundary, quadratic ρ beyond. Smooth second derivative = no active-set cycling in Newton.
 */
function penVal(g: number, rho: number, tau: number): number {
  if (g <= -tau) return 0;
  if (g < 0) { const x = g + tau; return rho * x * x * x / (6 * tau); }
  return rho * (tau * tau / 6 + 0.5 * tau * g + 0.5 * g * g);
}
function penD1(g: number, rho: number, tau: number): number {
  if (g <= -tau) return 0;
  if (g < 0) { const x = g + tau; return rho * x * x / (2 * tau); }
  return rho * (0.5 * tau + g);
}
function penD2(g: number, rho: number, tau: number): number {
  if (g <= -tau) return 0;
  return g < 0 ? rho * (g + tau) / tau : rho;
}
/** assumed bound on a border sample's |∂p/∂delta|_1 in units of the source focal (only decides which samples get an
 * exact Jacobian; rows are then pruned with their exact reach). Measured max on O3/OA4 clips: see smoothDiag. */
const JAC_BOUND = 4.0;
/** a 2-evaluation estimate is within a few px of the renderer's 3-evaluation value; keep this much extra slack */
const FAR_EXTRA_PX = 60;
/** frames whose sparse-sample slack exceeds this are not densely re-checked */
const DENSE_NEAR_PX = 40;

function segIds(F: number, segments?: Array<[number, number]>): Int32Array {
  const id = new Int32Array(F);
  if (segments && segments.length) {
    id.fill(-1);
    segments.forEach(([a, b], i) => { for (let k = Math.max(0, a); k <= Math.min(F - 1, b); k++) id[k] = i; });
    // frames outside every segment: own ids (never coupled)
    let nxt = segments.length;
    for (let k = 0; k < F; k++) if (id[k] < 0) id[k] = nxt++;
  }
  return id;
}

/** Zoom profile: closing (bridge gaps < hold frames) then slope-limited upper envelope (rate per frame), clamp. */
export function zoomEnvelope(need: Float64Array, rate: number, hold: number, zmax: number, seg: Int32Array): Float64Array {
  const F = need.length;
  const h = Math.max(0, Math.round(hold / 2));
  // morphological closing with a (2h+1) window: dilate then erode (within segments)
  const dil = new Float64Array(F), clo = new Float64Array(F);
  const win = (src: Float64Array, dst: Float64Array, op: (a: number, b: number) => number, init: number) => {
    for (let k = 0; k < F; k++) {
      let v = init;
      for (let d = -h; d <= h; d++) {
        const i = k + d;
        if (i < 0 || i >= F || seg[i] !== seg[k]) continue;
        v = op(v, src[i]);
      }
      dst[k] = v;
    }
  };
  if (h > 0) { win(need, dil, Math.max, 0); win(dil, clo, Math.min, Infinity); } else clo.set(need);
  const z = new Float64Array(F);
  for (let k = 0; k < F; k++) z[k] = Math.max(need[k], clo[k]);
  for (let k = 1; k < F; k++) if (seg[k] === seg[k - 1]) z[k] = Math.max(z[k], z[k - 1] - rate);
  for (let k = F - 2; k >= 0; k--) if (seg[k] === seg[k + 1]) z[k] = Math.max(z[k], z[k + 1] - rate);
  for (let k = 0; k < F; k++) z[k] = Math.min(Math.max(z[k], 0), zmax);
  return z;
}

// ================================================================================================= the optimizer

export function optimizePath(P: PathProblem, opt: SmoothOptions): SmoothResult {
  const T0 = now();
  const timings: Record<string, number> = { map: 0, lin: 0, inner: 0, zoom: 0 };
  const F = P.nFrames;
  const log = opt.log ?? (() => {});
  const W = smoothWeights(opt);
  const fx0 = opt.fx0;
  const fxMax = opt.fxMax ?? 1.5 * fx0;
  const allowZoom = (opt.allowZoom ?? true) && fxMax > fx0 * (1 + 1e-9);
  const zetaMax = allowZoom ? Math.log(fxMax / fx0) : 0;
  const f = fx0;
  const m = opt.marginPx ?? 8.0;
  const loX = m, hiX = P.srcW - 1 - m, loY = m, hiY = P.srcH - 1 - m;
  const trs = opt.trustRad ?? [0.25, 0.1, 0.04, 0.015, 0.006];
  const nMain = opt.sqpIters ?? trs.length;
  const nRepair = opt.repairIters ?? 3;
  const nInner = opt.innerIters ?? 10;
  const violTol = opt.violTolPx ?? 0.25;
  const rho = opt.rhoCrop ?? 1e4;
  const tauCrop = opt.tauCropPx ?? 1.0;
  let epsL1 = opt.epsL1 ?? 1e-3;
  const epsFinal = epsL1;
  const epsSched: number[] = opt.epsSchedule ?? [0.02, 0.008, 0.003, 0.001];
  const tolSched: number[] = opt.tolSchedule ?? [0.2, 0.1, 0.05, 0.02, 0.01];
  const reachCap = opt.reachCapPx ?? 400;
  let tolNow = 0.01;
  const fps = P.fps > 0 ? P.fps : 59.94;
  const seg = segIds(F, P.segments);
  const srcF = Math.max(P.lens.fx, P.lens.fy);
  const horizon = !!opt.horizonLock && !!P.gravityW;

  const V = Float64Array.from(P.camQ);
  qfixSignInPlace(V);
  const R = Float64Array.from(V);
  const zeta = new Float64Array(F);
  const info: SmoothInfo = {
    runtimeS: 0, fx0, fxMax, sqpIters: 0, innerIters: 0, maxViolationPx: 0, nFramesViolating: 0, fracBinding: 0,
    maxPhiDeg: 0, zoomChanges: 0, maxZoom: 1, zoomFrames: 0, infeasibleFrames: 0, globalZoom: 1, zoomEvents: 0, jacBound: 0, timings, iters: [],
  };
  if (F < 4) {
    info.runtimeS = (now() - T0) / 1000;
    return { virtQ: V, outFx: new Float64Array(F).fill(fx0), info };
  }

  const mapper = new CropMapper(P);
  const mr = mapper.r;
  const samples = borderSamples(P.outW, P.outH);
  const nFixed = samples.length / 2;
  const layout = edgeLayout(P.outW, P.outH);
  const Vm = new Float64Array(9 * F);
  const pf = new Float64Array(2 * nFixed);      // fixed-sample source positions of the current frame
  const adapt = new Float64Array(8);             // 4 adaptive points (x,y)
  const extra = new Map<number, Float64Array>(); // frame -> extra constraint points (dense-check repair)
  const minSlack = new Float64Array(F);          // per frame: min slack over samples (source px, margin applied)
  const rows = new Growable();
  let jacMax = 0;

  const updateVm = () => { for (let k = 0; k < F; k++) quatToMatInto(Vm, 9 * k, V, 4 * k); };

  /** Adaptive (per edge) output points where the mapped border is furthest out (smooth._adaptive_points). */
  const adaptivePoints = () => {
    for (let e = 0; e < 4; e++) {
      const E = layout[e];
      const n = E.idx.length;
      const Lw = E.ax === 0 ? P.outW - 1 : P.outH - 1;
      const spacing = Lw / (n - 1);
      let iStar = 0, best = -Infinity;
      for (let i = 0; i < n; i++) {
        const v = E.sgn * pf[2 * E.idx[i] + E.comp];
        if (v > best) { best = v; iStar = i; }
      }
      const i0 = Math.min(Math.max(iStar, 1), n - 2);
      const ym = E.sgn * pf[2 * E.idx[i0 - 1] + E.comp], y0 = E.sgn * pf[2 * E.idx[i0] + E.comp];
      const yp = E.sgn * pf[2 * E.idx[i0 + 1] + E.comp];
      const den = ym - 2 * y0 + yp;
      let t: number;
      if (den < -1e-12) t = Math.min(Math.max(0.5 * (ym - yp) / den, -0.95), 0.95);
      else t = yp >= ym ? 0.95 : -0.95;
      const pos = Math.min(Math.max((i0 + t) * spacing, 0), Lw);
      adapt[2 * e + E.ax] = pos;
      adapt[2 * e + 1 - E.ax] = E.fixed;
    }
  };

  /**
   * Exact mapping of every frame's constraint samples; pushes linearized crop rows (c·u <= rhs, c per output px of u)
   * whose slack is within reach of the trust region. Returns the max violation (source px beyond the margin).
   */
  const linearizeCrop = (tr: number | null): number => {
    const t0 = now();
    rows.n = 0;
    let maxViol = -Infinity;
    const doRow = (k: number, jac: boolean) => {
      const sx0 = mr[0] - loX, sx1 = hiX - mr[0], sy0 = mr[1] - loY, sy1 = hiY - mr[1];
      let s = Math.min(sx0, sx1, sy0, sy1);
      if (mr[2] <= 0) s = Math.min(s, -1e3); // behind the camera: hard violation
      if (s < minSlack[k]) minSlack[k] = s;
      if (!jac || tr === null) return;
      const G00 = mr[3], G01 = mr[4], G02 = mr[5], G10 = mr[6], G11 = mr[7], G12 = mr[8];
      const gb = Math.max(Math.abs(G00) + Math.abs(G01) + Math.abs(G02), Math.abs(G10) + Math.abs(G11) + Math.abs(G12));
      if (gb > jacMax) jacMax = gb;
      const rx = Math.min(tr * (Math.abs(G00) + Math.abs(G01) + Math.abs(G02)) + 2.0, reachCap);
      const ry = Math.min(tr * (Math.abs(G10) + Math.abs(G11) + Math.abs(G12)) + 2.0, reachCap);
      if (sx0 < rx) rows.push(k, -G00 / f, -G01 / f, -G02 / f, sx0);
      if (sx1 < rx) rows.push(k, G00 / f, G01 / f, G02 / f, sx1);
      if (sy0 < ry) rows.push(k, -G10 / f, -G11 / f, -G12 / f, sy0);
      if (sy1 < ry) rows.push(k, G10 / f, G11 / f, G12 / f, sy1);
    };
    // generous reach bound for deciding whether a sample needs its Jacobian (|G|_1 is ~1.5-3·lens.fx per rad)
    const reachAll = tr === null ? -Infinity : Math.min(tr * JAC_BOUND * Math.max(P.lens.fx, P.lens.fy) + 4.0, reachCap);
    // samples further inside than this may skip the third RS evaluation (value only; the slack stays >> reach)
    const farPx = Math.max(m, 0) + Math.max(reachAll, 0) + FAR_EXTRA_PX;
    for (let k = 0; k < F; k++) {
      const fo = fx0 * Math.exp(zeta[k]);
      minSlack[k] = Infinity;
      for (let s = 0; s < nFixed; s++) {
        mapper.map(k, Vm, 9 * k, fo, samples[2 * s], samples[2 * s + 1], false, farPx);
        pf[2 * s] = mr[0]; pf[2 * s + 1] = mr[1];
        const sl = Math.min(mr[0] - loX, hiX - mr[0], mr[1] - loY, hiY - mr[1]);
        if (sl < reachAll || mr[2] <= 0) { mapper.map(k, Vm, 9 * k, fo, samples[2 * s], samples[2 * s + 1], true); doRow(k, true); }
        else doRow(k, false);
      }
      adaptivePoints();
      for (let e = 0; e < 4; e++) {
        mapper.map(k, Vm, 9 * k, fo, adapt[2 * e], adapt[2 * e + 1], false, farPx);
        const sl = Math.min(mr[0] - loX, hiX - mr[0], mr[1] - loY, hiY - mr[1]);
        if (sl < reachAll || mr[2] <= 0) { mapper.map(k, Vm, 9 * k, fo, adapt[2 * e], adapt[2 * e + 1], true); doRow(k, true); }
        else doRow(k, false);
      }
      const ex = extra.get(k);
      if (ex) {
        for (let s = 0; s < ex.length / 2; s++) {
          mapper.map(k, Vm, 9 * k, fo, ex[2 * s], ex[2 * s + 1], tr !== null);
          doRow(k, tr !== null);
        }
      }
      if (-minSlack[k] > maxViol) maxViol = -minSlack[k];
    }
    timings.map += now() - t0;
    return maxViol;
  };

  // ------------------------------------------------------------------ linearization buffers
  const Jr = new Float64Array(9 * F), Jl = new Float64Array(9 * F); // per pair t (t, t+1)
  const x0 = new Float64Array(3 * F);   // f·Log(V_tᵀV_{t+1}) (px/frame) per pair
  const B = new Float64Array(9 * F);    // Jr⁻¹(phi0_k)
  const fphi = new Float64Array(3 * F); // f·phi0_k
  const hz = new Float64Array(3 * F);   // horizon: ∂g_x/∂delta (per rad)
  const g0x = new Float64Array(F);      // horizon: f·g_x
  const Hfid = new Float64Array(9 * F); // 2·w_fid·BᵀB + 2·prox·I
  // Exact Gauss-Newton products of the velocity operator (w_t = x0_t + E1_t u_{t+1} + E0_t u_t, E0 = -Jl, E1 = Jr):
  // per pair t, 11 3x3 blocks: [0] E0ᵀE0 [1] E1ᵀE1 [2] E1ᵀE0 (s = t) ; [3..6] E_tᵀ(a) E_{t-1}(b) for (a,b) =
  // (0,0),(0,1),(1,0),(1,1) ; [7..10] the same with E_{t-2}.
  const NPP = 99;
  const PP = new Float64Array(NPP * F);
  const vOK = new Uint8Array(F), aOK = new Uint8Array(F), jOK = new Uint8Array(F);
  for (let t = 0; t < F; t++) {
    vOK[t] = t + 1 < F && seg[t] === seg[t + 1] ? 1 : 0;
    aOK[t] = t + 2 < F && seg[t] === seg[t + 2] ? 1 : 0;
    jOK[t] = t + 3 < F && seg[t] === seg[t + 3] ? 1 : 0;
  }
  const tmpQ = new Float64Array(8);
  /** PP[o..o+8] = s·Aᵀ·B for row-major 3x3 blocks A = Am[ao..], B = Bm[bo..] */
  const tprod = (o: number, s: number, Am: Float64Array, ao: number, Bm: Float64Array, bo: number) => {
    for (let i = 0; i < 3; i++) {
      const a0 = s * Am[ao + i], a1 = s * Am[ao + 3 + i], a2 = s * Am[ao + 6 + i];
      PP[o + 3 * i] = a0 * Bm[bo] + a1 * Bm[bo + 3] + a2 * Bm[bo + 6];
      PP[o + 3 * i + 1] = a0 * Bm[bo + 1] + a1 * Bm[bo + 4] + a2 * Bm[bo + 7];
      PP[o + 3 * i + 2] = a0 * Bm[bo + 2] + a1 * Bm[bo + 5] + a2 * Bm[bo + 8];
    }
  };
  const linearizeMotion = () => {
    const t0 = now();
    for (let t = 0; t < F; t++) {
      if (vOK[t]) {
        qconjMulInto(tmpQ, 0, V, 4 * t, V, 4 * t + 4);
        qlogInto(tmpQ, 4, tmpQ, 0);
        const ax = tmpQ[4], ay = tmpQ[5], az = tmpQ[6];
        x0[3 * t] = f * ax; x0[3 * t + 1] = f * ay; x0[3 * t + 2] = f * az;
        jrInvInto(Jr, 9 * t, ax, ay, az);
        jrInvInto(Jl, 9 * t, -ax, -ay, -az);
      }
      qconjMulInto(tmpQ, 0, R, 4 * t, V, 4 * t);
      qlogInto(tmpQ, 4, tmpQ, 0);
      fphi[3 * t] = f * tmpQ[4]; fphi[3 * t + 1] = f * tmpQ[5]; fphi[3 * t + 2] = f * tmpQ[6];
      jrInvInto(B, 9 * t, tmpQ[4], tmpQ[5], tmpQ[6]);
      {
        const o = 9 * t, w2f = 2 * W.fid, w2p = 2 * W.prox;
        for (let c = 0; c < 3; c++) for (let d = 0; d < 3; d++) {
          Hfid[o + 3 * c + d] = w2f * (B[o + c] * B[o + d] + B[o + 3 + c] * B[o + 3 + d] + B[o + 6 + c] * B[o + 6 + d])
            + (c === d ? w2p : 0);
        }
      }
      if (horizon) {
        const G = P.gravityW!, o = 9 * t;
        // g0 = Vᵀ g_w
        const gx = Vm[o] * G[3 * t] + Vm[o + 3] * G[3 * t + 1] + Vm[o + 6] * G[3 * t + 2];
        const gy = Vm[o + 1] * G[3 * t] + Vm[o + 4] * G[3 * t + 1] + Vm[o + 7] * G[3 * t + 2];
        const gz = Vm[o + 2] * G[3 * t] + Vm[o + 5] * G[3 * t + 1] + Vm[o + 8] * G[3 * t + 2];
        hz[3 * t] = 0; hz[3 * t + 1] = -gz; hz[3 * t + 2] = gy;
        g0x[t] = f * gx;
      }
    }
    for (let t = 0; t < F; t++) {
      if (!vOK[t]) continue;
      const o = NPP * t;
      // E0 = -Jl, E1 = Jr
      const m = 9 * t;
      tprod(o, 1, Jl, m, Jl, m); tprod(o + 9, 1, Jr, m, Jr, m); tprod(o + 18, -1, Jr, m, Jl, m);
      if (t >= 1 && vOK[t - 1] && seg[t - 1] === seg[t]) {
        const n = m - 9;
        tprod(o + 27, 1, Jl, m, Jl, n); tprod(o + 36, -1, Jl, m, Jr, n);
        tprod(o + 45, -1, Jr, m, Jl, n); tprod(o + 54, 1, Jr, m, Jr, n);
      }
      if (t >= 2 && vOK[t - 2] && seg[t - 2] === seg[t]) {
        const n = m - 18;
        tprod(o + 63, 1, Jl, m, Jl, n); tprod(o + 72, -1, Jl, m, Jr, n);
        tprod(o + 81, -1, Jr, m, Jl, n); tprod(o + 90, 1, Jr, m, Jr, n);
      }
    }
    timings.lin += now() - t0;
  };

  // ------------------------------------------------------------------ inner solve
  const n3 = 3 * F;
  const band = new BandSPD(n3, 11);
  const u = new Float64Array(n3), du = new Float64Array(n3), grad = new Float64Array(n3);
  const wv = new Float64Array(3 * F), av = new Float64Array(3 * F);
  const Ga = new Float64Array(3 * F), Gw = new Float64Array(3 * F);
  const Q0 = new Float64Array(F), Q1 = new Float64Array(F), Q2 = new Float64Array(F); // pair-space (t,t),(t,t-1),(t,t-2)
  const Hb = new Float64Array(9 * F);
  const crows = rows;
  const dbgInner = opt.debugInner;
  const BW = band.W, BB = band.b, BA = band.a;
  /** lower block (p > q): A[3p+r][3q+c] += s·PP[o + 3r + c] */
  const addOff = (p: number, q: number, s: number, o: number) => {
    let ri = 3 * p * BW - 3 * p + BB + 3 * q;
    BA[ri] += s * PP[o]; BA[ri + 1] += s * PP[o + 1]; BA[ri + 2] += s * PP[o + 2];
    ri += BW - 1;
    BA[ri] += s * PP[o + 3]; BA[ri + 1] += s * PP[o + 4]; BA[ri + 2] += s * PP[o + 5];
    ri += BW - 1;
    BA[ri] += s * PP[o + 6]; BA[ri + 1] += s * PP[o + 7]; BA[ri + 2] += s * PP[o + 8];
  };
  /** diagonal block p: lower triangle of s·M (sym = M symmetric) or of s·(M + Mᵀ) */
  const addDiag = (p: number, s: number, o: number, sym: boolean) => {
    for (let r = 0; r < 3; r++) {
      const i = 3 * p + r, ri = i * BW - i + BB + 3 * p;
      for (let c = 0; c <= r; c++) BA[ri + c] += s * (sym ? PP[o + 3 * r + c] : PP[o + 3 * r + c] + PP[o + 3 * c + r]);
    }
  };

  const ut = new Float64Array(n3), gradFull = new Float64Array(n3), rhsv = new Float64Array(n3);
  let rowG = new Float64Array(1024);
  const held = new Uint8Array(n3);
  /**
   * Merit Φ(u) of the linearized problem: Huber-smoothed group L1 (= what the IRLS weights majorize) + L2 motion
   * terms + fidelity + horizon + prox + stiff crop/box penalties. The Newton/IRLS direction is a descent direction
   * of Φ (its gradient is exact), so an Armijo backtrack on Φ makes the inner loop monotone (no active-set cycling).
   */
  const merit = (uu: Float64Array, T: number): number => {
    let phi = 0;
    const hub = (n: number) => (n >= epsL1 ? n : 0.5 * (n * n / epsL1 + epsL1));
    for (let t = 0; t < F; t++) {
      if (!vOK[t]) continue;
      const r = 9 * t, p = 3 * t, q = 3 * t + 3;
      for (let c = 0; c < 3; c++) {
        wv[p + c] = x0[p + c]
          + Jr[r + 3 * c] * uu[q] + Jr[r + 3 * c + 1] * uu[q + 1] + Jr[r + 3 * c + 2] * uu[q + 2]
          - Jl[r + 3 * c] * uu[p] - Jl[r + 3 * c + 1] * uu[p + 1] - Jl[r + 3 * c + 2] * uu[p + 2];
      }
      phi += W.velL2 * (wv[p] * wv[p] + wv[p + 1] * wv[p + 1] + wv[p + 2] * wv[p + 2]);
    }
    for (let t = 0; t < F; t++) {
      if (!aOK[t]) continue;
      const p = 3 * t;
      av[p] = wv[p + 3] - wv[p]; av[p + 1] = wv[p + 4] - wv[p + 1]; av[p + 2] = wv[p + 5] - wv[p + 2];
      const n2 = av[p] * av[p] + av[p + 1] * av[p + 1] + av[p + 2] * av[p + 2];
      phi += W.accL2 * n2 + W.accL1 * hub(Math.sqrt(n2));
    }
    for (let t = 0; t < F; t++) {
      if (!jOK[t]) continue;
      const p = 3 * t;
      const j0 = av[p + 3] - av[p], j1 = av[p + 4] - av[p + 1], j2 = av[p + 5] - av[p + 2];
      const n2 = j0 * j0 + j1 * j1 + j2 * j2;
      phi += W.jerkL2 * n2 + W.jerkL1 * hub(Math.sqrt(n2));
    }
    for (let k = 0; k < F; k++) {
      const p = 3 * k, o = 9 * k;
      const u0 = uu[p], u1 = uu[p + 1], u2 = uu[p + 2];
      const r0 = fphi[p] + B[o] * u0 + B[o + 1] * u1 + B[o + 2] * u2;
      const r1 = fphi[p + 1] + B[o + 3] * u0 + B[o + 4] * u1 + B[o + 5] * u2;
      const r2 = fphi[p + 2] + B[o + 6] * u0 + B[o + 7] * u1 + B[o + 8] * u2;
      phi += W.fid * (r0 * r0 + r1 * r1 + r2 * r2) + W.prox * (u0 * u0 + u1 * u1 + u2 * u2);
      if (horizon) { const rh = g0x[k] + hz[p] * u0 + hz[p + 1] * u1 + hz[p + 2] * u2; phi += W.horizon * rh * rh; }
    }
    for (let i = 0; i < crows.n; i++) {
      const p = 3 * crows.k[i];
      const g = crows.c[3 * i] * uu[p] + crows.c[3 * i + 1] * uu[p + 1] + crows.c[3 * i + 2] * uu[p + 2] - crows.rhs[i];
      phi += penVal(g, rho, tauCrop);
    }
    return phi;
  };

  const innerSolve = (tr: number): number => {
    const t0 = now();
    u.fill(0);
    const T = f * tr;
    let it = 0;
    let phiU = NaN;
    let resFresh = false;   // wv/av already hold the residuals at u (left there by the accepted line-search trial)
    for (; it < nInner; it++) {
      const tI = now();
      // ---- residuals of the linearized motion terms at u
      if (!resFresh) {
        for (let t = 0; t < F; t++) {
          if (!vOK[t]) continue;
          const r = 9 * t, p = 3 * t, q = 3 * t + 3;
          for (let c = 0; c < 3; c++) {
            wv[p + c] = x0[p + c]
              + Jr[r + 3 * c] * u[q] + Jr[r + 3 * c + 1] * u[q + 1] + Jr[r + 3 * c + 2] * u[q + 2]
              - Jl[r + 3 * c] * u[p] - Jl[r + 3 * c + 1] * u[p + 1] - Jl[r + 3 * c + 2] * u[p + 2];
          }
        }
        for (let t = 0; t < F; t++) {
          if (!aOK[t]) continue;
          const p = 3 * t;
          av[p] = wv[p + 3] - wv[p]; av[p + 1] = wv[p + 4] - wv[p + 1]; av[p + 2] = wv[p + 5] - wv[p + 2];
        }
      }
      Q0.fill(0); Q1.fill(0); Q2.fill(0);
      Ga.fill(0); Gw.fill(0);
      // jerk j_t = w_{t+2} - 2w_{t+1} + w_t: IRLS weight, gradient into Ga, Q stencil [1,-2,1] on pairs t..t+2
      for (let t = 0; t < F; t++) {
        if (!jOK[t]) continue;
        const p = 3 * t;
        const j0 = av[p + 3] - av[p], j1 = av[p + 4] - av[p + 1], j2 = av[p + 5] - av[p + 2];
        const nj = Math.sqrt(j0 * j0 + j1 * j1 + j2 * j2);
        const w2 = 2 * (W.jerkL2 + W.jerkL1 / (2 * Math.max(nj, epsL1)));
        Ga[p + 3] += w2 * j0; Ga[p + 4] += w2 * j1; Ga[p + 5] += w2 * j2;
        Ga[p] -= w2 * j0; Ga[p + 1] -= w2 * j1; Ga[p + 2] -= w2 * j2;
        Q0[t] += w2; Q0[t + 1] += 4 * w2; Q0[t + 2] += w2;
        Q1[t + 1] += -2 * w2; Q1[t + 2] += -2 * w2;
        Q2[t + 2] += w2;
      }
      // acceleration a_t = w_{t+1} - w_t
      for (let t = 0; t < F; t++) {
        if (!aOK[t]) continue;
        const p = 3 * t;
        const a0 = av[p], a1 = av[p + 1], a2 = av[p + 2];
        const na = Math.sqrt(a0 * a0 + a1 * a1 + a2 * a2);
        const w2 = 2 * (W.accL2 + W.accL1 / (2 * Math.max(na, epsL1)));
        const g0 = Ga[p] + w2 * a0, g1 = Ga[p + 1] + w2 * a1, g2 = Ga[p + 2] + w2 * a2;
        Gw[p + 3] += g0; Gw[p + 4] += g1; Gw[p + 5] += g2;
        Gw[p] -= g0; Gw[p + 1] -= g1; Gw[p + 2] -= g2;
        Q0[t] += w2; Q0[t + 1] += w2; Q1[t + 1] += -w2;
      }
      // velocity; gradient through the exact velocity operator
      grad.fill(0);
      const w2v = 2 * W.velL2;
      for (let t = 0; t < F; t++) {
        if (!vOK[t]) continue;
        const p = 3 * t, q = p + 3, r = 9 * t;
        const g0 = Gw[p] + w2v * wv[p], g1 = Gw[p + 1] + w2v * wv[p + 1], g2 = Gw[p + 2] + w2v * wv[p + 2];
        for (let c = 0; c < 3; c++) {
          grad[q + c] += Jr[r + c] * g0 + Jr[r + 3 + c] * g1 + Jr[r + 6 + c] * g2;
          grad[p + c] -= Jl[r + c] * g0 + Jl[r + 3 + c] * g1 + Jl[r + 6 + c] * g2;
        }
        Q0[t] += w2v;
      }
      // ---- per-frame exact blocks: fidelity, horizon, prox, box
      const w2f = 2 * W.fid, w2p = 2 * W.prox, w2h = 2 * W.horizon;
      for (let k = 0; k < F; k++) {
        const p = 3 * k, o = 9 * k;
        const u0 = u[p], u1 = u[p + 1], u2 = u[p + 2];
        const r0 = fphi[p] + B[o] * u0 + B[o + 1] * u1 + B[o + 2] * u2;
        const r1 = fphi[p + 1] + B[o + 3] * u0 + B[o + 4] * u1 + B[o + 5] * u2;
        const r2 = fphi[p + 2] + B[o + 6] * u0 + B[o + 7] * u1 + B[o + 8] * u2;
        for (let c = 0; c < 3; c++) {
          grad[p + c] += w2f * (B[o + c] * r0 + B[o + 3 + c] * r1 + B[o + 6 + c] * r2) + w2p * u[p + c];
        }
        for (let c = 0; c < 9; c++) Hb[o + c] = Hfid[o + c];
        if (horizon) {
          const h0 = hz[p], h1 = hz[p + 1], h2 = hz[p + 2];
          const rh = g0x[k] + h0 * u0 + h1 * u1 + h2 * u2;
          grad[p] += w2h * h0 * rh; grad[p + 1] += w2h * h1 * rh; grad[p + 2] += w2h * h2 * rh;
          Hb[o] += w2h * h0 * h0; Hb[o + 1] += w2h * h0 * h1; Hb[o + 2] += w2h * h0 * h2;
          Hb[o + 3] += w2h * h1 * h0; Hb[o + 4] += w2h * h1 * h1; Hb[o + 5] += w2h * h1 * h2;
          Hb[o + 6] += w2h * h2 * h0; Hb[o + 7] += w2h * h2 * h1; Hb[o + 8] += w2h * h2 * h2;
        }
      }
      // ---- crop rows: smoothed penalty gradient at u (Hessian blocks added after the base assembly)
      if (rowG.length < crows.n) rowG = new Float64Array(2 * crows.n);
      for (let i = 0; i < crows.n; i++) {
        const p = 3 * crows.k[i];
        const g = crows.c[3 * i] * u[p] + crows.c[3 * i + 1] * u[p + 1] + crows.c[3 * i + 2] * u[p + 2] - crows.rhs[i];
        rowG[i] = g;
        if (g > -tauCrop) {
          const d1 = penD1(g, rho, tauCrop);
          grad[p] += d1 * crows.c[3 * i]; grad[p + 1] += d1 * crows.c[3 * i + 1]; grad[p + 2] += d1 * crows.c[3 * i + 2];
        }
      }
      for (let i = 0; i < n3; i++) gradFull[i] = -grad[i];
      const tA = now();
      timings.grad = (timings.grad ?? 0) + tA - tI;
      // ---- assemble the exact Hessian: Wᵀ(Q⊗I3)W + blockdiag(Hb) (+ crop curvature below)
      BA.fill(0);
      for (let k = 0; k < F; k++) {
        const o = 9 * k;
        for (let r = 0; r < 3; r++) {
          const i = 3 * k + r, ri = i * BW - i + BB + 3 * k;
          for (let c = 0; c <= r; c++) BA[ri + c] += Hb[o + 3 * r + c];
        }
      }
      for (let t = 0; t < F; t++) {
        if (!vOK[t]) continue;
        const o = NPP * t;
        const q0 = Q0[t];
        if (q0 !== 0) { addDiag(t, q0, o, true); addDiag(t + 1, q0, o + 9, true); addOff(t + 1, t, q0, o + 18); }
        const q1 = Q1[t];
        if (q1 !== 0) {   // pairs (t, t-1): blocks (t+a, t-1+b)
          addOff(t, t - 1, q1, o + 27);      // a=0,b=0
          addDiag(t, q1, o + 36, false);     // a=0,b=1 -> diagonal block (t,t): M + Mᵀ
          addOff(t + 1, t - 1, q1, o + 45);  // a=1,b=0
          addOff(t + 1, t, q1, o + 54);      // a=1,b=1
        }
        const q2 = Q2[t];
        if (q2 !== 0) {   // pairs (t, t-2): blocks (t+a, t-2+b)
          addOff(t, t - 2, q2, o + 63);
          addOff(t, t - 1, q2, o + 72);
          addOff(t + 1, t - 2, q2, o + 81);
          addOff(t + 1, t - 1, q2, o + 90);
        }
      }
      // projected Newton for the trust-region box |u_i| <= T (Bertsekas): variables at a bound whose descent direction
      // points outward are held (row/col -> identity, zero step); the rest take the Newton step; projected Armijo.
      let nHeld = 0;
      for (let i = 0; i < n3; i++) {
        const v = u[i], g = grad[i];
        held[i] = (v >= T - 1e-9 && g < 0) || (v <= -T + 1e-9 && g > 0) ? 1 : 0;
        nHeld += held[i];
      }
      const tF = now();
      timings.asm = (timings.asm ?? 0) + tF - tA;
      for (let i = 0; i < n3; i++) rhsv[i] = -grad[i];
      for (let i = 0; i < crows.n; i++) {
        const g = rowG[i];
        if (g <= -tauCrop) continue;
        const d2 = penD2(g, rho, tauCrop);
        const p = 3 * crows.k[i];
        const c0 = crows.c[3 * i], c1 = crows.c[3 * i + 1], c2 = crows.c[3 * i + 2];
        let ri = p * BW - p + BB + p;
        BA[ri] += d2 * c0 * c0;
        ri += BW - 1;
        BA[ri] += d2 * c1 * c0; BA[ri + 1] += d2 * c1 * c1;
        ri += BW - 1;
        BA[ri] += d2 * c2 * c0; BA[ri + 1] += d2 * c2 * c1; BA[ri + 2] += d2 * c2 * c2;
      }
      if (nHeld) {
        for (let i = 0; i < n3; i++) {
          if (!held[i]) continue;
          const ri = i * BW - i + BB;
          for (let j = Math.max(0, i - BB); j < i; j++) BA[ri + j] = 0;
          for (let r = i + 1; r <= Math.min(n3 - 1, i + BB); r++) BA[r * BW - r + BB + i] = 0;
          BA[ri + i] = 1;
          rhsv[i] = 0;
        }
      }
      band.factor();
      band.solve(rhsv, du);
      const tM = now();
      timings.chol = (timings.chol ?? 0) + tM - tF;
      if (!(phiU === phiU)) phiU = merit(u, T);                   // NaN -> first evaluation
      // projected Armijo line search on Φ
      let alpha = 1, phiT = 0;
      for (let ls = 0; ls < 12; ls++) {
        let dec = 0;   // ∇Φ·(u(α) - u)
        for (let i = 0; i < n3; i++) {
          let v = u[i] + alpha * du[i];
          if (v > T) v = T; else if (v < -T) v = -T;
          ut[i] = v;
          dec -= gradFull[i] * (v - u[i]);
        }
        phiT = merit(ut, T);
        if (phiT <= phiU + 1e-4 * dec) break;
        alpha *= 0.5;
      }
      let mx = 0, imx = 0;
      for (let i = 0; i < n3; i++) { const v = Math.abs(ut[i] - u[i]); if (v > mx) { mx = v; imx = i; } u[i] = ut[i]; }
      phiU = phiT;
      resFresh = true;
      timings.merit = (timings.merit ?? 0) + now() - tM;
      if (dbgInner) {
        let na = 0; for (let i = 0; i < crows.n; i++) na += rowG[i] > 0 ? 1 : 0;
        dbgInner(it, mx, { k: (imx / 3) | 0, alpha, nHeld, na, phi: phiT });
      }
      if (mx < tolNow && it >= 2) { it++; break; }
    }
    // enforce the trust region exactly (penalty tolerance)
    for (let i = 0; i < n3; i++) { if (u[i] > T) u[i] = T; else if (u[i] < -T) u[i] = -T; }
    timings.inner += now() - t0;
    info.innerIters += it;
    return it;
  };

  const applyStep = () => {
    let mx = 0;
    for (let k = 0; k < F; k++) {
      const p = 3 * k;
      qexpInto(tmpQ, 4, u[p] / f, u[p + 1] / f, u[p + 2] / f);
      qmulInto(tmpQ, 0, V, 4 * k, tmpQ, 4);
      V[4 * k] = tmpQ[0]; V[4 * k + 1] = tmpQ[1]; V[4 * k + 2] = tmpQ[2]; V[4 * k + 3] = tmpQ[3];
      qnormalizeInPlace(V, 4 * k);
      mx = Math.max(mx, Math.abs(u[p]), Math.abs(u[p + 1]), Math.abs(u[p + 2]));
    }
    return mx / f;
  };

  // ------------------------------------------------------------------ SQP
  const progress = (x: number) => { if (opt.onProgress) opt.onProgress(Math.min(1, Math.max(0, x))); };
  const totalPlanned = nMain + nRepair;
  let it = 0;
  const sqp = (schedule: number[], nMin: number, nMax: number, label: string) => {
    let viol = Infinity, prev = Infinity;
    for (let i = 0; ; i++) {
      updateVm();
      // repairs (past the schedule) may move as far as the current violation needs (~2x its angle), not less
      let tr = schedule[Math.min(i, schedule.length - 1)];
      if (i >= schedule.length - 1 && prev < Infinity && prev > violTol) tr = Math.max(tr, Math.min(0.05, 2 * prev / srcF));
      viol = linearizeCrop(i >= nMax ? null : tr);
      if (label === 'main' && i === 0 && globalZoomCheck()) viol = linearizeCrop(tr);
      // stop: converged, out of iterations, or (repair phase) the violation stagnates -> infeasible: zoom decides
      const stagnant = i > nMin && viol > violTol && viol > 0.8 * prev;
      if (i >= nMax || (i >= nMin && viol <= violTol) || stagnant) {
        if (i < nMax) { /* minSlack is current (exact pass with Jacobian rows is a superset of the value pass) */ }
        break;
      }
      prev = viol;
      linearizeMotion();
      epsL1 = Math.max(epsSched[Math.min(it, epsSched.length - 1)], epsFinal);
      tolNow = tolSched[Math.min(it, tolSched.length - 1)];
      const ni = innerSolve(tr);
      const step = applyStep();
      info.iters.push({ it: it, tr, rows: rows.n, inner: ni, maxStepDeg: step * 180 / Math.PI, maxViolPx: viol });
      log(`[smooth] ${label} it ${it} tr ${tr} rows ${rows.n} inner ${ni} step ${(step * 180 / Math.PI).toFixed(3)} deg viol ${viol.toFixed(3)}`);
      it++;
      progress(0.9 * it / (totalPlanned + 2));
    }
    return viol;
  };
  // ------------------------------------------------------------------ base field of view feasible at all?
  // Frames whose crop cannot fit even when the virtual camera follows the real one need zoom whatever the path does.
  // If that is a large part of the clip the requested field of view is simply too wide: zoom in ONCE (constant zoom,
  // no breathing) to the 98th percentile of the per-frame need; isolated frames are left to the local zoom below.
  const minZoomFeasible = (k: number): number => {
    const feasible = (z: number) => {
      const fo = fx0 * Math.exp(z);
      for (let s = 0; s < nFixed; s++) {
        mapper.map(k, Vm, 9 * k, fo, samples[2 * s], samples[2 * s + 1], false);
        pf[2 * s] = mr[0]; pf[2 * s + 1] = mr[1];
        if (mr[2] <= 0 || Math.min(mr[0] - loX, hiX - mr[0], mr[1] - loY, hiY - mr[1]) < -0.05) return false;
      }
      adaptivePoints();
      for (let e = 0; e < 4; e++) {
        mapper.map(k, Vm, 9 * k, fo, adapt[2 * e], adapt[2 * e + 1], false);
        if (mr[2] <= 0 || Math.min(mr[0] - loX, hiX - mr[0], mr[1] - loY, hiY - mr[1]) < -0.05) return false;
      }
      return true;
    };
    if (feasible(zeta[k])) return zeta[k];
    let lo = zeta[k], hi = zetaMax;
    if (!feasible(hi)) return zetaMax;
    for (let b = 0; b < 14; b++) { const mid = 0.5 * (lo + hi); if (feasible(mid)) hi = mid; else lo = mid; }
    return hi;
  };
  /** Uses minSlack of the current (initial) path; returns true when a constant zoom was applied. */
  const globalZoomCheck = (): boolean => {
    if (!allowZoom) return false;
    const tG = now();
    const need: number[] = [];
    let nInf = 0;
    for (let k = 0; k < F; k++) {
      if (-minSlack[k] > violTol) { nInf++; need.push(minZoomFeasible(k)); } else need.push(0);
    }
    let applied = false;
    if (nInf > 0.25 * F) {
      need.sort((a, b) => a - b);
      const zg = Math.min(zetaMax, need[Math.min(F - 1, Math.floor(0.98 * F))] * 1.02 + 1e-4);
      zeta.fill(zg);
      info.globalZoom = Math.exp(zg);
      applied = true;
      log(`[smooth] requested field of view does not fit ${nInf}/${F} frames even along the camera path: constant zoom ${Math.exp(zg).toFixed(4)}x`);
    }
    timings.zoom += now() - tG;
    return applied;
  };
  let viol = sqp(trs, nMain, nMain + nRepair, 'main');

  // ------------------------------------------------------------------ zoom where the rotation alone cannot fit
  const tZ = now();
  const rate = (opt.maxZoomRate ?? 0.04) / fps;
  const hold = (opt.zoomHoldS ?? 2.0) * fps;
  let zoomRounds = 0;
  while (allowZoom && viol > violTol && zoomRounds < 3) {
    zoomRounds++;
    const need = Float64Array.from(zeta);
    let nNeed = 0;
    for (let k = 0; k < F; k++) {
      if (-minSlack[k] <= violTol) continue;
      nNeed++;
      need[k] = Math.max(need[k], minZoomFeasible(k) * 1.02 + 1e-4);
    }
    const env = zoomEnvelope(need, rate, hold, zetaMax, seg);
    for (let k = 0; k < F; k++) zeta[k] = Math.max(zeta[k], env[k]);
    let zm = 0;
    for (let k = 0; k < F; k++) zm = Math.max(zm, zeta[k]);
    log(`[smooth] zoom round ${zoomRounds}: ${nNeed} frames need zoom, max ${Math.exp(zm).toFixed(4)}x`);
    viol = sqp([trs[trs.length - 1]], 1, nRepair, 'zoom');
  }
  timings.zoom += now() - tZ;

  // ------------------------------------------------------------------ dense verification + repair
  const dense = denseBorder(P.outW, P.outH, 24);
  const denseCheck = (): number => {
    let bad = 0;
    updateVm();
    for (let k = 0; k < F; k++) {
      if (minSlack[k] > DENSE_NEAR_PX) continue;   // the border between samples cannot bulge this far
      const fo = fx0 * Math.exp(zeta[k]);
      let worst = violTol, wi = -1;
      for (let s = 0; s < dense.length / 2; s++) {
        mapper.map(k, Vm, 9 * k, fo, dense[2 * s], dense[2 * s + 1], false);
        let v = -Math.min(mr[0] - loX, hiX - mr[0], mr[1] - loY, hiY - mr[1]);
        if (mr[2] <= 0) v = 1e3;
        if (v > worst) { worst = v; wi = s; }
      }
      if (wi >= 0) {
        bad++;
        const prev = extra.get(k);
        const add = [dense[2 * wi], dense[2 * wi + 1]];
        extra.set(k, prev ? Float64Array.from([...prev, ...add]) : Float64Array.from(add));
      }
    }
    return bad;
  };
  for (let r = 0; r < 2; r++) {
    const bad = denseCheck();
    if (!bad) break;
    log(`[smooth] dense check: ${bad} frames with violations between samples -> repair`);
    viol = sqp([trs[trs.length - 1]], 1, nRepair, 'dense');
  }

  // ------------------------------------------------------------------ diagnostics (minSlack is current: every sqp()
  // ends with an exact pass over the final path)
  const outFx = new Float64Array(F);
  let zc = 0, zf = 0, zmx = 0, nViol = 0, nBind = 0, maxPhi = 0, zev = 0;
  const zBase = Math.log(info.globalZoom);
  for (let k = 0; k < F; k++) {
    if (zeta[k] > zBase + 1e-6 && (k === 0 || zeta[k - 1] <= zBase + 1e-6)) zev++;
    outFx[k] = fx0 * Math.exp(zeta[k]);
    if (k > 0 && Math.abs(zeta[k] - zeta[k - 1]) > 1e-4) zc++;
    if (zeta[k] > zBase + 1e-6) zf++;
    zmx = Math.max(zmx, zeta[k]);
    if (-minSlack[k] > 0.05) nViol++;
    if (minSlack[k] < 0.5) nBind++;
    qconjMulInto(tmpQ, 0, R, 4 * k, V, 4 * k);
    qlogInto(tmpQ, 4, tmpQ, 0);
    maxPhi = Math.max(maxPhi, Math.hypot(tmpQ[4], tmpQ[5], tmpQ[6]));
  }
  qfixSignInPlace(V);
  Object.assign(info, {
    runtimeS: (now() - T0) / 1000, sqpIters: it, maxViolationPx: viol, nFramesViolating: nViol, fracBinding: nBind / F,
    maxPhiDeg: maxPhi * 180 / Math.PI, zoomChanges: zc, maxZoom: Math.exp(zmx), zoomFrames: zf, zoomEvents: zev,
    infeasibleFrames: nViol,
  });
  info.jacBound = jacMax / srcF;
  for (const k of Object.keys(timings)) timings[k] = Math.round(timings[k]) / 1000;
  progress(1);
  return { virtQ: V, outFx, info };
}

// ================================================================================================= diagnostics

/**
 * Per-frame max crop violation (source px beyond `marginPx`, <= 0 = inside) over a dense output border with the
 * renderer-exact mapping (smooth.check_crop). virtQ F*4, outFx F.
 */
export function checkCrop(P: PathProblem, virtQ: Float64Array, outFx: ArrayLike<number>, nPerEdge = 64,
                          marginPx = 0): Float64Array {
  const mapper = new CropMapper(P);
  const pts = denseBorder(P.outW, P.outH, nPerEdge);
  const Vm = new Float64Array(9);
  const out = new Float64Array(P.nFrames);
  const lo = marginPx, hiX = P.srcW - 1 - marginPx, hiY = P.srcH - 1 - marginPx;
  for (let k = 0; k < P.nFrames; k++) {
    quatToMatInto(Vm, 0, virtQ, 4 * k);
    let v = -Infinity;
    for (let s = 0; s < pts.length / 2; s++) {
      mapper.map(k, Vm, 0, outFx[k], pts[2 * s], pts[2 * s + 1], false);
      const r = mapper.r;
      let x = Math.max(lo - r[0], r[0] - hiX, lo - r[1], r[1] - hiY);
      if (r[2] <= 0) x = Math.max(x, 1e3);
      if (x > v) v = x;
    }
    out[k] = v;
  }
  return out;
}

/**
 * High-frequency content of an orientation path (e.g. the virtual camera): per-frame body angular velocity
 * Log(V_tᵀV_{t+1}) integrated to angles, minus a zero-phase Gaussian low-pass with -3 dB at `cutoffHz`, RMS over
 * frames and axes (as the vector norm), in px at focal `fPx`. Returns {rms, perAxis}.
 */
export function pathJitterPx(q: Float64Array, fps: number, fPx: number, cutoffHz = 1.5,
                             segments?: Array<[number, number]>): { rms: number; perAxis: [number, number, number] } {
  const F = q.length >> 2;
  const seg = segIds(F, segments);
  const sigma = Math.sqrt(Math.LN2) / (2 * Math.PI * cutoffHz) * fps; // frames
  const R = Math.ceil(4 * sigma);
  const ker = new Float64Array(2 * R + 1);
  let ks = 0;
  for (let i = -R; i <= R; i++) { ker[i + R] = Math.exp(-0.5 * (i / sigma) ** 2); ks += ker[i + R]; }
  for (let i = 0; i < ker.length; i++) ker[i] /= ks;
  const tmp = new Float64Array(8);
  const acc = [0, 0, 0];
  let n = 0;
  // per segment
  let a = 0;
  while (a < F) {
    let b = a;
    while (b + 1 < F && seg[b + 1] === seg[a]) b++;
    const L = b - a + 1;
    if (L > 2 * R + 2) {
      const th = new Float64Array(3 * L);
      for (let t = 1; t < L; t++) {
        qconjMulInto(tmp, 0, q, 4 * (a + t - 1), q, 4 * (a + t));
        qlogInto(tmp, 4, tmp, 0);
        for (let c = 0; c < 3; c++) th[3 * t + c] = th[3 * (t - 1) + c] + tmp[4 + c];
      }
      for (let t = R; t < L - R; t++) { // skip filter edges
        for (let c = 0; c < 3; c++) {
          let lp = 0;
          for (let i = -R; i <= R; i++) lp += ker[i + R] * th[3 * (t + i) + c];
          const h = (th[3 * t + c] - lp) * fPx;
          acc[c] += h * h;
        }
        n++;
      }
    }
    a = b + 1;
  }
  const pa: [number, number, number] = [Math.sqrt(acc[0] / Math.max(n, 1)), Math.sqrt(acc[1] / Math.max(n, 1)),
    Math.sqrt(acc[2] / Math.max(n, 1))];
  return { rms: Math.sqrt((acc[0] + acc[1] + acc[2]) / Math.max(n, 1)), perAxis: pa };
}
