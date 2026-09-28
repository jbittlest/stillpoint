/**
 * Source lens model (owner: PATH agent): DJI's embedded Kannala-Brandt 4-coefficient fisheye ('kb4') or a pinhole,
 * formula-for-formula like engine/stillpoint/geom.py Lens and shaders/warp.metal sp_project.
 *
 *   theta = atan2(sqrt(X²+Y²), Z);  theta_d = theta·(1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)
 *   u = fx·theta_d·X/r + cx ;  v = fy·theta_d·Y/r + cy          (pixel centres at integers)
 */
import type { Lens } from './types';

export type Vec = Float64Array | number[];

/** Camera ray (X,Y,Z) -> source pixel into out[o], out[o+1]. Rays behind the camera still map (theta > 90°). */
export function project(L: Lens, X: number, Y: number, Z: number, out: Vec, o: number): void {
  if (L.model === 'pinhole') {
    const Zs = Math.abs(Z) < 1e-12 ? 1e-12 : Z;
    out[o] = L.fx * X / Zs + L.cx;
    out[o + 1] = L.fy * Y / Zs + L.cy;
    return;
  }
  const rxy = Math.sqrt(X * X + Y * Y);
  const th = Math.atan2(rxy, Z);
  const t2 = th * th;
  const k = L.k;
  const thd = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
  const s = rxy < 1e-12 ? 1.0 / Math.max(Z, 1e-12) : thd / rxy;
  out[o] = L.fx * X * s + L.cx;
  out[o + 1] = L.fy * Y * s + L.cy;
}

/**
 * project() plus its analytic Jacobian ∂(u,v)/∂(X,Y,Z) (row-major 2x3 into J[j..j+5]).
 */
export function projectJac(L: Lens, X: number, Y: number, Z: number, out: Vec, o: number, J: Vec, j: number): void {
  if (L.model === 'pinhole') {
    const Zs = Math.abs(Z) < 1e-12 ? 1e-12 : Z;
    const iz = 1 / Zs;
    out[o] = L.fx * X * iz + L.cx;
    out[o + 1] = L.fy * Y * iz + L.cy;
    J[j] = L.fx * iz; J[j + 1] = 0; J[j + 2] = -L.fx * X * iz * iz;
    J[j + 3] = 0; J[j + 4] = L.fy * iz; J[j + 5] = -L.fy * Y * iz * iz;
    return;
  }
  const k = L.k;
  const r2 = X * X + Y * Y;
  const r = Math.sqrt(r2);
  if (r < 1e-9) {
    const zz = Math.max(Z, 1e-12);
    const s = 1 / zz;
    out[o] = L.fx * X * s + L.cx;
    out[o + 1] = L.fy * Y * s + L.cy;
    J[j] = L.fx * s; J[j + 1] = 0; J[j + 2] = -L.fx * X * s * s;
    J[j + 3] = 0; J[j + 4] = L.fy * s; J[j + 5] = -L.fy * Y * s * s;
    return;
  }
  const th = Math.atan2(r, Z);
  const t2 = th * th;
  const thd = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
  const dthd = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
  const rho2 = r2 + Z * Z;
  const dth_dr = Z / rho2, dth_dZ = -r / rho2;
  const s = thd / r;
  const ds_dr = (dthd * dth_dr * r - thd) / r2;
  const ds_dZ = dthd * dth_dZ / r;
  const sx = ds_dr * X / r, sy = ds_dr * Y / r; // ∂s/∂X, ∂s/∂Y
  out[o] = L.fx * X * s + L.cx;
  out[o + 1] = L.fy * Y * s + L.cy;
  J[j] = L.fx * (s + X * sx); J[j + 1] = L.fx * X * sy; J[j + 2] = L.fx * X * ds_dZ;
  J[j + 3] = L.fy * Y * sx; J[j + 4] = L.fy * (s + Y * sy); J[j + 5] = L.fy * Y * ds_dZ;
}

/** Source pixel -> unit camera ray (Newton on theta, geom.Lens.unproject). */
export function unproject(L: Lens, u: number, v: number, out: Vec, o: number, iters = 8): void {
  const mx = (u - L.cx) / L.fx, my = (v - L.cy) / L.fy;
  if (L.model === 'pinhole') {
    const n = Math.sqrt(mx * mx + my * my + 1);
    out[o] = mx / n; out[o + 1] = my / n; out[o + 2] = 1 / n;
    return;
  }
  const k = L.k;
  const thd = Math.sqrt(mx * mx + my * my);
  let th = thd;
  for (let i = 0; i < iters; i++) {
    const t2 = th * th;
    const f = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - thd;
    const df = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
    th = th - f / df;
  }
  const s = thd < 1e-12 ? 1.0 : Math.sin(th) / thd;
  out[o] = mx * s; out[o + 1] = my * s; out[o + 2] = Math.cos(th);
}

/** Output focal (px) of a rectilinear output with horizontal field of view hfovDeg. */
export function hfovToFx(hfovDeg: number, width: number): number {
  return (width / 2) / Math.tan(hfovDeg * Math.PI / 360);
}

export function fxToHfov(fx: number, width: number): number {
  return 2 * Math.atan(width / 2 / fx) * 180 / Math.PI;
}

/**
 * Fraction of the source area covered by the output rectangle (pixel-centre border polygon, `nPerEdge` samples per
 * edge) mapped at identity orientation, no rolling shutter (smooth.fx_for_crop_area's area()).
 */
export function identityFootprint(L: Lens, srcW: number, srcH: number, outW: number, outH: number, fx: number,
                                  nPerEdge = 128): number {
  const cx = (outW - 1) / 2, cy = (outH - 1) / 2;
  const n = nPerEdge;
  const P = new Float64Array(8 * n);
  let m = 0;
  const push = (px: number, py: number) => { project(L, (px - cx) / fx, (py - cy) / fx, 1, P, m); m += 2; };
  for (let i = 0; i < n; i++) push(i * (outW - 1) / (n - 1), 0);
  for (let i = 0; i < n; i++) push(outW - 1, i * (outH - 1) / (n - 1));
  for (let i = n - 1; i >= 0; i--) push(i * (outW - 1) / (n - 1), outH - 1);
  for (let i = n - 1; i >= 0; i--) push(0, i * (outH - 1) / (n - 1));
  let a = 0;
  const N = 4 * n;
  for (let i = 0; i < N; i++) {
    const i2 = (i + 1) % N;
    a += P[2 * i] * P[2 * i2 + 1] - P[2 * i + 1] * P[2 * i2];
  }
  return 0.5 * Math.abs(a) / (srcW * srcH);
}

/** Output focal (px) whose identity-mapped output covers `area` of the source (smooth.fx_for_crop_area). */
export function fxForFootprint(L: Lens, srcW: number, srcH: number, outW: number, outH: number, area: number,
                               nPerEdge = 128): number {
  let lo = 0.05 * outW, hi = 50.0 * outW;
  for (let i = 0; i < 80; i++) {
    const mid = Math.sqrt(lo * hi);
    if (identityFootprint(L, srcW, srcH, outW, outH, mid, nPerEdge) > area) lo = mid; else hi = mid;
  }
  return Math.sqrt(lo * hi);
}
