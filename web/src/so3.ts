/**
 * SO(3) / quaternion helpers for the web PATH module (owner: PATH agent). Formula-for-formula ports of
 * engine/stillpoint/geom.py (qexp, qlog, quat_to_mat, slerp_series) and smooth.py (_jr_inv), so the TS plan matches
 * the Python engine to ~1e-12.
 *
 * Conventions (ENGINE_SPEC.md §1): quaternions (w,x,y,z), Hamilton product, float64; an orientation maps CAMERA
 * vectors to WORLD. Arrays are flat Float64Arrays (quaternion i at [4i..4i+3], 3x3 matrix i row-major at [9i..9i+8]).
 * All hot functions write into caller-provided arrays (no allocation).
 */

export type Vec = Float64Array | number[];

/** out[o..o+3] = a[ai..] * b[bi..] (Hamilton). out may alias neither a nor b at the same offset. */
export function qmulInto(out: Vec, o: number, a: Vec, ai: number, b: Vec, bi: number): void {
  const aw = a[ai], ax = a[ai + 1], ay = a[ai + 2], az = a[ai + 3];
  const bw = b[bi], bx = b[bi + 1], by = b[bi + 2], bz = b[bi + 3];
  out[o] = aw * bw - ax * bx - ay * by - az * bz;
  out[o + 1] = aw * bx + ax * bw + ay * bz - az * by;
  out[o + 2] = aw * by - ax * bz + ay * bw + az * bx;
  out[o + 3] = aw * bz + ax * by - ay * bx + az * bw;
}

/** out = conj(a) * b */
export function qconjMulInto(out: Vec, o: number, a: Vec, ai: number, b: Vec, bi: number): void {
  const aw = a[ai], ax = -a[ai + 1], ay = -a[ai + 2], az = -a[ai + 3];
  const bw = b[bi], bx = b[bi + 1], by = b[bi + 2], bz = b[bi + 3];
  out[o] = aw * bw - ax * bx - ay * by - az * bz;
  out[o + 1] = aw * bx + ax * bw + ay * bz - az * by;
  out[o + 2] = aw * by - ax * bz + ay * bw + az * bx;
  out[o + 3] = aw * bz + ax * by - ay * bx + az * bw;
}

export function qnormalizeInPlace(q: Vec, o: number): void {
  const n = Math.sqrt(q[o] * q[o] + q[o + 1] * q[o + 1] + q[o + 2] * q[o + 2] + q[o + 3] * q[o + 3]);
  q[o] /= n; q[o + 1] /= n; q[o + 2] /= n; q[o + 3] /= n;
}

/** Rotation vector (rad) -> unit quaternion (geom.qexp). */
export function qexpInto(out: Vec, o: number, vx: number, vy: number, vz: number): void {
  const th = Math.sqrt(vx * vx + vy * vy + vz * vz);
  const half = 0.5 * th;
  const k = th < 1e-8 ? 0.5 - th * th / 48.0 : Math.sin(half) / th;
  out[o] = Math.cos(half);
  out[o + 1] = vx * k;
  out[o + 2] = vy * k;
  out[o + 3] = vz * k;
}

/** Unit quaternion -> rotation vector (rad), shortest rotation (geom.qlog). */
export function qlogInto(out: Vec, o: number, q: Vec, qi: number): void {
  let w = q[qi], x = q[qi + 1], y = q[qi + 2], z = q[qi + 3];
  if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  if (w > 1) w = 1;
  const s = Math.sqrt(x * x + y * y + z * z);
  const th = 2.0 * Math.atan2(s, w);
  const k = s < 1e-8 ? 2.0 / (w === 0 ? 1.0 : w) : th / s;
  out[o] = x * k; out[o + 1] = y * k; out[o + 2] = z * k;
}

/** (w,x,y,z) -> row-major 3x3 (normalizes first, like geom.quat_to_mat). */
export function quatToMatInto(out: Vec, o: number, q: Vec, qi: number): void {
  let w = q[qi], x = q[qi + 1], y = q[qi + 2], z = q[qi + 3];
  const n = Math.sqrt(w * w + x * x + y * y + z * z);
  w /= n; x /= n; y /= n; z /= n;
  out[o] = 1 - 2 * (y * y + z * z); out[o + 1] = 2 * (x * y - w * z); out[o + 2] = 2 * (x * z + w * y);
  out[o + 3] = 2 * (x * y + w * z); out[o + 4] = 1 - 2 * (x * x + z * z); out[o + 5] = 2 * (y * z - w * x);
  out[o + 6] = 2 * (x * z - w * y); out[o + 7] = 2 * (y * z + w * x); out[o + 8] = 1 - 2 * (x * x + y * y);
}

/** Make a quaternion series (N*4) sign-continuous in place (geom.qfix_sign). */
export function qfixSignInPlace(q: Float64Array): void {
  const n = q.length >> 2;
  let flip = false;
  for (let i = 1; i < n; i++) {
    const a = 4 * i, b = a - 4;
    // compare with the ORIGINAL previous sample (as numpy does: cumulative parity of negative dots)
    const d = q[a] * q[b] + q[a + 1] * q[b + 1] + q[a + 2] * q[b + 2] + q[a + 3] * q[b + 3];
    // q[b] may already be flipped: undo that for the dot product
    const dOrig = flip ? -d : d;
    if (dOrig < 0) flip = !flip;
    if (flip) { q[a] = -q[a]; q[a + 1] = -q[a + 1]; q[a + 2] = -q[a + 2]; q[a + 3] = -q[a + 3]; }
  }
}

/**
 * Inverse right Jacobian of SO(3) (smooth.py _jr_inv): Exp(th)·Exp(d) ≈ Exp(th + Jr⁻¹(th)·d). Row-major into out.
 * Jl⁻¹(th) = Jr⁻¹(-th).
 */
export function jrInvInto(out: Vec, o: number, tx: number, ty: number, tz: number): void {
  const a2 = tx * tx + ty * ty + tz * tz;
  const a = Math.sqrt(a2);
  const c = a < 1e-3 ? 1.0 / 12.0 + a2 / 720.0 : 1.0 / a2 - (1.0 + Math.cos(a)) / (2.0 * a * Math.sin(a));
  // H = hat(t) = [[0,-z,y],[z,0,-x],[-y,x,0]];  H·H = t tᵀ - |t|² I
  out[o] = 1 + c * (tx * tx - a2); out[o + 1] = -0.5 * tz + c * tx * ty; out[o + 2] = 0.5 * ty + c * tx * tz;
  out[o + 3] = 0.5 * tz + c * ty * tx; out[o + 4] = 1 + c * (ty * ty - a2); out[o + 5] = -0.5 * tx + c * ty * tz;
  out[o + 6] = -0.5 * ty + c * tz * tx; out[o + 7] = 0.5 * tx + c * tz * ty; out[o + 8] = 1 + c * (tz * tz - a2);
}

/**
 * A sign-continuous orientation series with exact slerp between the two bracketing samples
 * (geom.slerp_series: i = clip(searchsorted(t, tq, 'right') - 1, 0, n-2), u clipped to [0,1]; queries outside the
 * range clamp to the ends). O(1) lookups on uniform grids (index guess + local walk), O(log n) otherwise.
 */
export class OrientationSeries {
  readonly t: Float64Array;
  readonly q: Float64Array;
  readonly n: number;
  /** per interval i: Log(conj(q_i)·q_{i+1}) (rad), precomputed once (same formula as the per-query slerp) */
  private readonly d: Float64Array;
  private readonly t0: number;
  private readonly invDt: number;
  private tmp = new Float64Array(8);

  constructor(t: Float64Array, q: Float64Array, copy = true) {
    if (t.length < 2 || q.length !== 4 * t.length) throw new Error('OrientationSeries: need >= 2 samples and q = 4*n');
    this.t = t;
    this.q = copy ? Float64Array.from(q) : q;
    qfixSignInPlace(this.q);
    this.n = t.length;
    this.t0 = t[0];
    this.invDt = (this.n - 1) / (t[this.n - 1] - t[0]);
    this.d = new Float64Array(3 * (this.n - 1));
    const tmp = this.tmp;
    for (let i = 0; i < this.n - 1; i++) {
      qconjMulInto(tmp, 0, this.q, 4 * i, this.q, 4 * i + 4);
      qlogInto(this.d, 3 * i, tmp, 0);
    }
  }

  /** index i with t[i] <= tq < t[i+1] (clamped to [0, n-2]) */
  index(tq: number): number {
    const t = this.t, n = this.n;
    let i = Math.floor((tq - this.t0) * this.invDt);
    if (!(i >= 0)) i = 0; // also NaN
    if (i > n - 2) i = n - 2;
    // walk to the exact searchsorted('right') - 1 position
    if (t[i] > tq) {
      let steps = 0;
      while (i > 0 && t[i] > tq) { i--; if (++steps > 8) return this.bsearch(tq); }
    } else {
      let steps = 0;
      while (i < n - 2 && t[i + 1] <= tq) { i++; if (++steps > 8) return this.bsearch(tq); }
    }
    return i;
  }

  private bsearch(tq: number): number {
    const t = this.t;
    let lo = 0, hi = this.n; // first index with t > tq
    while (lo < hi) { const m = (lo + hi) >> 1; if (t[m] <= tq) lo = m + 1; else hi = m; }
    let i = lo - 1;
    if (i < 0) i = 0;
    if (i > this.n - 2) i = this.n - 2;
    return i;
  }

  /** Orientation at time tq into out[o..o+3] (normalized): q_i · Exp(u · Log(conj(q_i) q_{i+1})). */
  at(tq: number, out: Vec, o: number): void {
    const i = this.index(tq);
    const t = this.t, tmp = this.tmp, d = this.d;
    const dt = t[i + 1] - t[i];
    let u = (tq - t[i]) / (dt > 1e-12 ? dt : 1e-12);
    if (u < 0) u = 0; else if (u > 1) u = 1;
    qexpInto(tmp, 0, d[3 * i] * u, d[3 * i + 1] * u, d[3 * i + 2] * u);
    qmulInto(out, o, this.q, 4 * i, tmp, 0);
    qnormalizeInPlace(out, o);
  }
}

/** Row-major 3x3 helpers on flat arrays. */
export function matTMulVec(M: Vec, m: number, x: number, y: number, z: number, out: Vec, o: number): void {
  out[o] = M[m] * x + M[m + 3] * y + M[m + 6] * z;
  out[o + 1] = M[m + 1] * x + M[m + 4] * y + M[m + 7] * z;
  out[o + 2] = M[m + 2] * x + M[m + 5] * y + M[m + 8] * z;
}

export function matMulVec(M: Vec, m: number, x: number, y: number, z: number, out: Vec, o: number): void {
  out[o] = M[m] * x + M[m + 1] * y + M[m + 2] * z;
  out[o + 1] = M[m + 3] * x + M[m + 4] * y + M[m + 5] * z;
  out[o + 2] = M[m + 6] * x + M[m + 7] * y + M[m + 8] * z;
}

/** out = Aᵀ·B (row-major 3x3) */
export function matTMulInto(out: Vec, o: number, A: Vec, a: number, B: Vec, b: number): void {
  for (let i = 0; i < 3; i++) {
    const a0 = A[a + i], a1 = A[a + 3 + i], a2 = A[a + 6 + i];
    out[o + 3 * i] = a0 * B[b] + a1 * B[b + 3] + a2 * B[b + 6];
    out[o + 3 * i + 1] = a0 * B[b + 1] + a1 * B[b + 4] + a2 * B[b + 7];
    out[o + 3 * i + 2] = a0 * B[b + 2] + a1 * B[b + 5] + a2 * B[b + 8];
  }
}

/** Relative rotation vector Log(conj(a)·b) of two quaternions (rad). */
export function relRotvecInto(out: Vec, o: number, a: Vec, ai: number, b: Vec, bi: number, tmp: Float64Array): void {
  qconjMulInto(tmp, 0, a, ai, b, bi);
  qlogInto(out, o, tmp, 0);
}
