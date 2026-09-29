/**
 * Stillpoint web — gyro-accurate synthetic shutter (natural motion blur) for retimed exports.
 *
 * The motion-blur schedule (retime.ts buildSchedule, motionBlur 'natural') gives every output frame i a shutter window
 * [T_i - d/2, T_i + d/2] (d = 180°/360°/fps) and the source frames whose time slabs overlap it, weighted by the
 * overlap. Blending those frames as they are (RetimeRenderer) integrates the light of 2-3 DISCRETE instants: with
 * DJI footage's short exposure, fast motion shows as 2-3 ghost copies instead of a streak.
 *
 * Here each tap k is spread over its own part of the window, [max(T-d/2, slabLo_k), min(T+d/2, slabHi_k)] (the whole
 * window when it is the output's only tap: a shutter no longer than a source frame, e.g. 59.94 -> 30 fps): frame k
 * is re-warped at S sub-frame instants t inside it with the virtual camera's orientation AT t,
 *     M'_row = M_row · R(virt_k)ᵀ R(virt(t))      (virt(t): slerp of the plan's virtual path, outFx lerped)
 * i.e. what the stabilized camera saw while it kept turning during the exposure, from frame k's pixels (exact for
 * rotation, the dominant fast image motion in FPV; parallax from translation stays as it is in the plain blend).
 * Neighbouring taps meet at the slab boundaries with the same virtual orientation, so the streak is continuous.
 * S follows the image motion (<= pxStep output px between sub-frames at the frame corners, <= maxPerTap), so calm
 * frames cost one warp per tap and fast turns get a smooth streak. Sub-frame orientations that would pull the output
 * outside the source picture are pulled back towards the frame's own time (no black edges).
 * Samples are accumulated in linear light (Blender.accumulate: float, 16 B/px per output in flight — one at 180°).
 *
 *   const sr = await ShutterRenderer.create(warper, schedule, { pts, shutterS: 0.5 / fps, record, transfer });
 *   for (each decoded frame f, schedule index k) { for (const out of sr.push(f, k)) { encode(out); out.close(); } f.close(); }
 */
import type { OutputSchedule, Plan } from '../types';
import type { Warper } from './warp';
import { Blender, type BlendTransfer } from './blend';

export interface ShutterOptions {
  /** presentation times (s) of the schedule's source frames (index = schedule tap index) */
  pts: ArrayLike<number>;
  /** exposure window of an output frame, s (180° at fps: 0.5 / fps) */
  shutterS: number;
  /** plan record of schedule source frame k (default k) */
  record?: (k: number) => number;
  transfer?: BlendTransfer;
  /** max image motion between sub-frames at the frame corners, output px (default 1.5) */
  pxStep?: number;
  /** sub-frames per tap / per output frame (defaults 16 / 32) */
  maxPerTap?: number;
  maxPerFrame?: number;
}

export interface ShutterStats {
  outputs: number;
  /** sub-frame warps in total, and the most in one output frame */
  subframes: number;
  maxSubframes: number;
  /** taps whose sub-frame span was shortened to stay inside the source picture */
  clamped: number;
  /** source frames no output uses */
  skipped: number;
  /** accumulators in flight at most */
  peakAccumulators: number;
}

type PlanWithPath = Plan & { virtQ?: Float64Array };

/** q = slerp(a, b, u) for unit quaternions (w,x,y,z) at a[ai], b[bi] */
function slerpInto(out: Float64Array, a: ArrayLike<number>, ai: number, b: ArrayLike<number>, bi: number, u: number) {
  let bw = b[bi], bx = b[bi + 1], by = b[bi + 2], bz = b[bi + 3];
  let d = a[ai] * bw + a[ai + 1] * bx + a[ai + 2] * by + a[ai + 3] * bz;
  if (d < 0) { d = -d; bw = -bw; bx = -bx; by = -by; bz = -bz; }
  let s0: number, s1: number;
  if (d > 0.9995) { s0 = 1 - u; s1 = u; }
  else { const th = Math.acos(d), sn = Math.sin(th); s0 = Math.sin((1 - u) * th) / sn; s1 = Math.sin(u * th) / sn; }
  const w = s0 * a[ai] + s1 * bw, x = s0 * a[ai + 1] + s1 * bx, y = s0 * a[ai + 2] + s1 * by, z = s0 * a[ai + 3] + s1 * bz;
  const n = Math.hypot(w, x, y, z) || 1;
  out[0] = w / n; out[1] = x / n; out[2] = y / n; out[3] = z / n;
}

/** row-major rotation matrix of a unit quaternion (so3.quatToMatInto) */
function quatMat(out: Float64Array, q: ArrayLike<number>, o = 0) {
  const w = q[o], x = q[o + 1], y = q[o + 2], z = q[o + 3];
  out[0] = 1 - 2 * (y * y + z * z); out[1] = 2 * (x * y - w * z); out[2] = 2 * (x * z + w * y);
  out[3] = 2 * (x * y + w * z); out[4] = 1 - 2 * (x * x + z * z); out[5] = 2 * (y * z - w * x);
  out[6] = 2 * (x * z - w * y); out[7] = 2 * (y * z + w * x); out[8] = 1 - 2 * (x * x + y * y);
}

/**
 * Row matrices of plan record r turned to the virtual camera orientation at plan time tPlan (s):
 * M'_row = M_row · R(virt_r)ᵀ R(virt(tPlan)), virt(t) = slerp of the plan's virtual path between the records around
 * tPlan (held at the ends). Writes nRows*9 floats into `out`; returns the output focal at tPlan (lerped outFx) and
 * the unit quaternion of virt(tPlan) in `q` (when given).
 */
export function subframeRows(plan: Plan & { virtQ?: Float64Array }, r: number, tPlan: number, out: Float32Array, q?: Float64Array): number {
  const fp = plan.framePts, F = fp.length, V = plan.virtQ!, fx = plan.outFx;
  let a = r, b = r;
  if (tPlan >= fp[r] && r + 1 < F) b = r + 1;
  else if (tPlan < fp[r] && r > 0) a = r - 1;
  const u = a === b ? 0 : Math.min(1, Math.max(0, (tPlan - fp[a]) / (fp[b] - fp[a] || 1)));
  const qt = q ?? new Float64Array(4);
  slerpInto(qt, V, 4 * a, V, 4 * b, u);
  const Vt = new Float64Array(9), Vk = new Float64Array(9), D = new Float64Array(9);
  quatMat(Vt, qt);
  quatMat(Vk, V, 4 * r);
  // D = Vkᵀ Vt
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) D[3 * i + j] = Vk[i] * Vt[j] + Vk[3 + i] * Vt[3 + j] + Vk[6 + i] * Vt[6 + j];
  const M = plan.rowMats, nr = plan.nRows, base = r * nr * 9;
  for (let row = 0; row < nr; row++) {
    const o = base + 9 * row, w = 9 * row;
    for (let i = 0; i < 3; i++) {
      const m0 = M[o + 3 * i], m1 = M[o + 3 * i + 1], m2 = M[o + 3 * i + 2];
      out[w + 3 * i] = m0 * D[0] + m1 * D[3] + m2 * D[6];
      out[w + 3 * i + 1] = m0 * D[1] + m1 * D[4] + m2 * D[7];
      out[w + 3 * i + 2] = m0 * D[2] + m1 * D[5] + m2 * D[8];
    }
  }
  return fx[a] + (fx[b] - fx[a]) * u;
}

/** The Warper's output -> source mapping (warp.wgsl source_coord), CPU float64, for the edge-validity check. */
class EdgeCheck {
  private readonly pts: Float64Array;
  constructor(private readonly plan: Plan, nPerEdge = 12) {
    const W = plan.outW, H = plan.outH, n = nPerEdge;
    const a: number[] = [];
    for (let i = 0; i < n; i++) { const x = i * (W - 1) / (n - 1); a.push(x, 0, x, H - 1); }
    for (let i = 1; i < n - 1; i++) { const y = i * (H - 1) / (n - 1); a.push(0, y, W - 1, y); }
    this.pts = Float64Array.from(a);
  }

  /** every border sample maps inside the source picture */
  ok(rows: Float32Array | Float64Array, fx: number): boolean {
    const p = this.plan, L = p.lens, nr = p.nRows, H = p.srcH;
    const ocx = (p.outW - 1) / 2, ocy = (p.outH - 1) / 2, sc = (nr - 1) / (H - 1);
    const kb4 = L.model !== 'pinhole';
    for (let s = 0; s < this.pts.length; s += 2) {
      const rx = (this.pts[s] - ocx) / fx, ry = (this.pts[s + 1] - ocy) / fx;
      let y = 0.5 * (H - 1), ya = 0, ga = 0, u = 0, v = 0, Z = 1;
      for (let e = 0; e < 3; e++) {
        const g = Math.min(Math.max(y * sc, 0), nr - 1);
        const j0 = Math.min(g | 0, nr - 2), f = g - j0, a = 9 * j0, b = a + 9;
        const X0 = rows[a] * rx + rows[a + 1] * ry + rows[a + 2], Y0 = rows[a + 3] * rx + rows[a + 4] * ry + rows[a + 5], Z0 = rows[a + 6] * rx + rows[a + 7] * ry + rows[a + 8];
        const X1 = rows[b] * rx + rows[b + 1] * ry + rows[b + 2], Y1 = rows[b + 3] * rx + rows[b + 4] * ry + rows[b + 5], Z1 = rows[b + 6] * rx + rows[b + 7] * ry + rows[b + 8];
        const X = X0 + (X1 - X0) * f, Y = Y0 + (Y1 - Y0) * f;
        Z = Z0 + (Z1 - Z0) * f;
        if (kb4) {
          const rxy = Math.hypot(X, Y), th = Math.atan2(rxy, Z), t2 = th * th;
          const thd = th * (1 + t2 * (L.k[0] + t2 * (L.k[1] + t2 * (L.k[2] + t2 * L.k[3]))));
          const k = rxy < 1e-12 ? 1 / Math.max(Z, 1e-12) : thd / rxy;
          u = L.fx * X * k + L.cx; v = L.fy * Y * k + L.cy;
        } else {
          const z = Z > 0 ? Math.max(Z, 1e-12) : Math.min(Z, -1e-12);
          u = L.fx * X / z + L.cx; v = L.fy * Y / z + L.cy;
        }
        const gg = v - y;
        let yn = v;
        if (e >= 1) { const d = y - ya; if (d !== 0) { const sl = (gg - ga) / d; if (sl <= -0.1 && sl >= -2.0) yn = y - gg / sl; } }
        ya = y; ga = gg; y = yn;
      }
      if (!(Z > 0 && u >= -0.5 && v >= -0.5 && u <= p.srcW - 0.5 && v <= H - 0.5)) return false;
    }
    return true;
  }
}

export class ShutterRenderer {
  readonly schedule: OutputSchedule;
  readonly stats: ShutterStats = { outputs: 0, subframes: 0, maxSubframes: 0, clamped: 0, skipped: 0, peakAccumulators: 0 };
  private readonly warper: Warper;
  private readonly blender: Blender;
  private readonly plan: PlanWithPath;
  private readonly o: Required<Omit<ShutterOptions, 'transfer' | 'record'>> & { record: (k: number) => number };
  private readonly edge: EdgeCheck;
  /** per source frame: [output, tap index] pairs */
  private readonly uses: Array<Array<[number, number]>>;
  private readonly slabLo: Float64Array;
  private readonly slabHi: Float64Array;
  private readonly tmp: GPUTexture[];
  private readonly accs = new Map<number, { buf: GPUBuffer; n: number; fresh: boolean; pend: GPUTexture[]; pendW: number[] }>();
  private readonly pool: GPUBuffer[] = [];
  private tmpNext = 0;
  private next = 0;
  private lastK = -1;
  private destroyed = false;
  // scratch
  private readonly q = new Float64Array(4);
  private readonly rows: Float32Array;

  private constructor(warper: Warper, schedule: OutputSchedule, blender: Blender, plan: PlanWithPath, o: ShutterRenderer['o']) {
    this.warper = warper; this.schedule = schedule; this.blender = blender; this.plan = plan; this.o = o;
    this.edge = new EdgeCheck(plan);
    this.rows = new Float32Array(plan.nRows * 9);
    const N = o.pts.length, s = schedule;
    this.uses = Array.from({ length: N }, () => []);
    for (let i = 0; i < s.n; i++) for (let j = s.tapStart[i]; j < s.tapStart[i] + s.tapsPerFrame[i]; j++) {
      const k = s.taps[j];
      if (!(k >= 0 && k < N)) throw new Error(`schedule: output ${i} taps source frame ${k} of ${N}`);
      this.uses[k].push([i, j]);
    }
    const P = o.pts;
    this.slabLo = new Float64Array(N); this.slabHi = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      this.slabLo[k] = k === 0 ? (N > 1 ? P[0] - 0.5 * (P[1] - P[0]) : P[0] - 0.5 * o.shutterS) : 0.5 * (P[k - 1] + P[k]);
      this.slabHi[k] = k === N - 1 ? (N > 1 ? P[N - 1] + 0.5 * (P[N - 1] - P[N - 2]) : P[0] + 0.5 * o.shutterS) : 0.5 * (P[k] + P[k + 1]);
    }
    this.tmp = [0, 1, 2].map(i => warper.createTarget(`sp.shutter.sub${i}`));
  }

  /** false when the plan carries no virtual path (then use RetimeRenderer's frame blend) */
  static supported(plan: Plan): boolean {
    const v = (plan as PlanWithPath).virtQ;
    return !!v && v.length === plan.framePts.length * 4;
  }

  static async create(warper: Warper, schedule: OutputSchedule, opts: ShutterOptions): Promise<ShutterRenderer> {
    const plan = (warper as unknown as { plan: PlanWithPath }).plan;
    if (!ShutterRenderer.supported(plan)) throw new Error('synthetic shutter: the plan has no virtual camera path');
    if (!(opts.shutterS > 0)) throw new Error('synthetic shutter: bad shutter time');
    const blender = await Blender.create(warper.device, warper.width, warper.height, { transfer: opts.transfer ?? 'bt709' });
    return new ShutterRenderer(warper, schedule, blender, plan, {
      pts: opts.pts, shutterS: opts.shutterS, record: opts.record ?? (k => k),
      pxStep: opts.pxStep ?? 1.5, maxPerTap: Math.max(1, opts.maxPerTap ?? 16), maxPerFrame: Math.max(1, opts.maxPerFrame ?? 32),
    });
  }

  get emitted(): number { return this.next; }
  get done(): boolean { return this.next >= this.schedule.n; }

  /** rows of record r turned to the virtual orientation at source time t (into this.rows; this.q = virt(t)); returns
   *  the focal. t is on the index clock of frame k (pts ptsK); the plan's own clock may be offset. */
  private rowsAt(r: number, t: number, ptsK: number): number {
    return subframeRows(this.plan, r, this.plan.framePts[r] + (t - ptsK), this.rows, this.q);
  }

  /** angle (rad) the virtual camera turns between t0 and t1 */
  private turn(r: number, t0: number, t1: number, ptsK: number): number {
    this.rowsAt(r, t0, ptsK); const a = Float64Array.from(this.q);
    this.rowsAt(r, t1, ptsK); const b = this.q;
    const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
    return 2 * Math.acos(Math.min(1, d));
  }

  private validAt(r: number, t: number, ptsK: number): boolean {
    const f = this.rowsAt(r, t, ptsK);
    return this.edge.ok(this.rows, f);
  }

  /** Push source frame k (schedule index, increasing). Returns the output frames it completes. */
  push(frame: VideoFrame, k: number): VideoFrame[] {
    if (this.destroyed) throw new Error('ShutterRenderer destroyed');
    if (!(k > this.lastK)) throw new Error(`push: frame ${k} after ${this.lastK} (presentation order required)`);
    this.lastK = k;
    const s = this.schedule, o = this.o, out: VideoFrame[] = [];
    const uses = k < this.uses.length ? this.uses[k] : [];
    if (!uses.length) this.stats.skipped++;
    const r = o.record(k), ptsK = o.pts[k];
    try {
      for (const [i, j] of uses) {
        const T = s.outPtsUs[i] / 1e6, half = 0.5 * o.shutterS;
        // a frame that is an output's only tap (shutter not longer than a source frame, e.g. 59.94 -> 30 fps) covers
        // the whole window; several taps share it slab by slab
        const only = s.tapsPerFrame[i] === 1;
        let ta = only ? T - half : Math.max(T - half, this.slabLo[k]), tb = only ? T + half : Math.min(T + half, this.slabHi[k]);
        if (!(tb > ta)) { ta = tb = Math.min(Math.max(ptsK, T - half), T + half); }
        // keep every sub-frame inside the source picture: pull the ends back towards the frame's own time
        const anchor = Math.min(Math.max(ptsK, ta), tb);
        let clamped = false;
        const pull = (end: number): number => {
          if (end === anchor || this.validAt(r, end, ptsK)) return end;
          clamped = true;
          let good = anchor, bad = end;
          for (let it = 0; it < 7; it++) { const m = 0.5 * (good + bad); if (this.validAt(r, m, ptsK)) good = m; else bad = m; }
          return good;
        };
        const anchorOk = this.validAt(r, anchor, ptsK);
        if (anchorOk) { ta = pull(ta); tb = pull(tb); }
        if (clamped) this.stats.clamped++;
        const f0 = this.plan.outFx[r];
        const reach = Math.hypot(this.plan.outW, this.plan.outH) / 2;
        const px = this.turn(r, ta, tb, ptsK) * f0 * (1 + (reach / f0) ** 2);
        const taps = s.tapsPerFrame[i];
        const cap = Math.max(1, Math.min(o.maxPerTap, Math.floor(o.maxPerFrame / taps)));
        const S = anchorOk ? Math.max(1, Math.min(cap, Math.ceil(px / o.pxStep))) : 1;
        const w = s.weights[j] / S;
        let acc = this.accs.get(i);
        if (!acc) {
          acc = { buf: this.pool.pop() ?? this.blender.createAccumulator(), n: 0, fresh: true, pend: [], pendW: [] };
          this.accs.set(i, acc);
          this.stats.peakAccumulators = Math.max(this.stats.peakAccumulators, this.accs.size);
        }
        for (let m = 0; m < S; m++) {
          const t = S === 1 && !anchorOk ? ptsK : ta + ((m + 0.5) / S) * (tb - ta);
          const tex = this.tmp[this.tmpNext];
          this.tmpNext = (this.tmpNext + 1) % this.tmp.length;
          if (!anchorOk) this.warper.warpTo(frame, r, tex);      // the plan's own rows: valid by construction
          else { const f = this.rowsAt(r, t, ptsK); this.warper.warpToWith(frame, r, tex, this.rows, f); }
          acc.pend.push(tex); acc.pendW.push(w); acc.n++;
          this.stats.subframes++;
          if (acc.pend.length === this.tmp.length) this.flush(acc);
        }
        this.flush(acc);
      }
      while (this.next < s.n && s.lastSourceFrame[this.next] <= k) {
        const i = this.next, acc = this.accs.get(i);
        if (!acc || acc.fresh) throw new Error(`output ${i}: none of its source frames was pushed`);
        out.push(this.blender.resolve(acc.buf, { timestamp: Math.round(s.outPtsUs[i]), duration: Math.round(s.frameDurUs) }, this.tmp[0]));
        this.stats.outputs++;
        this.stats.maxSubframes = Math.max(this.stats.maxSubframes, acc.n);
        this.accs.delete(i);
        this.pool.push(acc.buf);
        this.next++;
      }
    } catch (e) {
      for (const f of out) f.close();
      throw e;
    }
    return out;
  }

  private flush(acc: { buf: GPUBuffer; fresh: boolean; pend: GPUTexture[]; pendW: number[] }) {
    if (!acc.pend.length) return;
    this.blender.accumulate(acc.buf, acc.pend, acc.pendW, acc.fresh);
    acc.fresh = false;
    acc.pend = []; acc.pendW = [];
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const a of this.accs.values()) a.buf.destroy();
    for (const b of this.pool) b.destroy();
    for (const t of this.tmp) t.destroy();
    this.accs.clear(); this.pool.length = 0;
    this.blender.destroy();
  }
}
