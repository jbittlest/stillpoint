/**
 * DJI telemetry -> Telemetry, in the browser — TELEMETRY module.
 *
 * TypeScript port of engine/stillpoint/telemetry.py `_parse` (PARSER_VERSION 8). Same timing model, same numbers
 * (checked against the Python engine on the real clips by web/test/telemetry.test.ts: timestamps within 1 us,
 * quaternions within 1e-6, identical flags / lens / warnings):
 *
 *   T_n                 FrameMetaHeader.frame_timestamp of frame n (camera clock, s)
 *   IMU block samples   T_n + (i - offset_n)/rate, fitted per continuous shot onto ONE uniform grid
 *                       T_n - offset_n/rate = a + N_n*dt + beta*exposure_n   (O3/O4P beta = 0.5: DJI folds
 *                       -exposure/2 into `offset`; OA4 beta = 0, or fitted when the exposure varies)
 *   picture of frame n  T_n - beta*(exposure_n - e_ref) + c_pic       (centre row, mid-exposure)
 *                       c_pic: O3/O4P 0; OA4 1 kHz oa4PictureOffset(e): +0.12 ms at <= 4.6 ms, -0.8 ms at 1/61 s
 *   OA4 per-frame attitude (16:9 / EIS on): cam_quat at T_n + 18 ms (4:3) or + 8.7 ms (16:9)
 *   camera clock -> video timeline: affine per segment, least squares PTS_n ~ alpha + s*T_n (never n/fps)
 *   axes: DJI body FRD -> camera (x right, y down, z fwd): q_cam = q_raw (x) (0.5, 0.5, 0.5, 0.5)
 *
 * Reads only the djmd samples (one small read per frame, coalesced when adjacent) plus 24 dbgi samples on OA4 (EIS
 * cross-check). Differences from the Python engine (none affect the Telemetry fields or flags): no pyav fallback, no
 * cache, dbgi EIS attitudes (diagnostic extras) are not decoded, accelerometer/cam_quat extras are omitted.
 */
import type { Lens, Mp4Info, Mp4Track, Telemetry } from './types';
import { findTrack, mainVideoTrack, openMp4, presentationTimes, readRanges, readSamplesAt, trackFps } from './mp4';
import { type ClipMeta, EIS_STATUS, frameTs, headerEisBaked, parseClipMeta, parseDbgiAc203, parseDjmd, pb,
         productFlags } from './dji';

export const PARSER_VERSION = 8;

// Reference values (research/footage_*.md): sanity checks of what the file says, and the lens fallback for modes that
// store none (OA4 16:9 / EIS on). File values are never silently overridden.
const KNOWN_LENSES: Record<string, { fx: number; k: number[]; width: number; height: number; readout_ns: number }> = {
  FC8383: { fx: 1405.1293, k: [0.24991769, 0.01360575, -0.06208358, 0.01219307], width: 3840, height: 2160,
            readout_ns: 9719999 },
  OsmoAction4: { fx: 1457.0737, k: [0.155131, 0.137141, -0.093861, 0.004170], width: 3840, height: 2880,
                 readout_ns: 11087603 },
};
export const OA4_FRAME_CENTER_OFFSET_S = -0.0008;
export const OA4_SHORT_SHUTTER_EXTRA_S = 0.00092;
export const OA4_SHORT_SHUTTER_MAX_S = 0.0046;
export const OA4_LONG_SHUTTER_S = 1.0 / 61.0;
export const OA4_CAM_QUAT_DELAY_S = 0.018;
export const OA4_CAM_QUAT_DELAY_16X9_S = 0.0087;
const DBGI_SPARSE_N = 8;
const QUICK_CLOCK_CHECKS = 32;
const Q_CAM2BODY: [number, number, number, number] = [0.5, 0.5, 0.5, 0.5];   // mat_to_quat([[0,0,1],[1,0,0],[0,1,0]])

/** OA4 (1 kHz stream) picture offset c_pic(e) in s (telemetry.py oa4_picture_offset). */
export function oa4PictureOffset(e: number): number {
  const ee = Number.isFinite(e) && e > 0 ? e : OA4_LONG_SHUTTER_S;
  let w = (OA4_LONG_SHUTTER_S - ee) / (OA4_LONG_SHUTTER_S - OA4_SHORT_SHUTTER_MAX_S);
  w = Math.min(1.0, Math.max(0.0, w));
  return OA4_FRAME_CENTER_OFFSET_S + OA4_SHORT_SHUTTER_EXTRA_S * w;
}

export interface LoadTelemetryOptions {
  /** quick look: parse only the first maxFrames frames (+ a sparse whole-clip clock check), like probe_telemetry */
  maxFrames?: number;
  signal?: AbortSignal;
  /** reads in flight (default 8: ~0.2 ms per small Blob read in Chrome; more in flight does not help) */
  concurrency?: number;
}

// ================================================================================================= small numerics

function median(a: ArrayLike<number>): number {
  const n = a.length;
  if (!n) return NaN;
  const s = Float64Array.from(a as ArrayLike<number>).sort();
  if (Number.isNaN(s[n - 1])) {
    // numpy: any NaN -> NaN
    return NaN;
  }
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

function nanFilter(a: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < a.length; i++) if (!Number.isNaN(a[i])) out.push(a[i]);
  return out;
}
function mean(a: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}
function std(a: ArrayLike<number>): number {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
  return Math.sqrt(s / a.length);
}
function nanstd(a: ArrayLike<number>): number {
  const b = nanFilter(a);
  return b.length ? std(b) : NaN;
}
function nanmaxAbs(a: ArrayLike<number>): number {
  let m = NaN;
  for (let i = 0; i < a.length; i++) {
    const v = Math.abs(a[i]);
    if (!Number.isNaN(v) && (Number.isNaN(m) || v > m)) m = v;
  }
  return m;
}
const nz = (x: number) => (Number.isNaN(x) ? 0 : x);   // np.nan_to_num for finite data

/** y = c0 + s*x least squares (telemetry.py _affine). */
function affine(x: ArrayLike<number>, y: ArrayLike<number>): [number, number] {
  const n = x.length;
  const x0 = x[0], y0 = y[0];
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i] - x0; my += y[i] - y0; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - x0 - mx, dy = y[i] - y0 - my;
    sxy += dx * dy; sxx += dx * dx;
  }
  const s = sxx > 0 ? sxy / sxx : 0;
  const c = (y0 + my) - s * mx;          // intercept in (x - x0)
  return [s, c - s * x0];
}

/** Weighted line fit y ~ a + dt*N (weights W = w^2). */
function wline(N: ArrayLike<number>, y: ArrayLike<number>, W: ArrayLike<number>): [number, number] {
  const n = N.length;
  const y0 = y[0];
  let sw = 0, mN = 0, my = 0;
  for (let i = 0; i < n; i++) { sw += W[i]; mN += W[i] * N[i]; my += W[i] * (y[i] - y0); }
  mN /= sw; my /= sw;
  let sNy = 0, sNN = 0;
  for (let i = 0; i < n; i++) {
    const dN = N[i] - mN, dy = y[i] - y0 - my;
    sNy += W[i] * dN * dy; sNN += W[i] * dN * dN;
  }
  const dt = sNN > 0 ? sNy / sNN : 0;
  return [y0 + (my - dt * mN), dt];
}

/** Uniform-grid fit over the blocks of one segment (telemetry.py _fit_grid). Sample j at a + j*dt. */
function fitGrid(T: Float64Array, off: Float64Array, cnt: Int32Array, exp: Float64Array, rate: number, beta: number) {
  const n = T.length;
  const Ts = 1.0 / rate;
  const Nfirst = new Float64Array(n);
  for (let i = 1; i < n; i++) Nfirst[i] = Nfirst[i - 1] + cnt[i - 1];
  const hasIdx: number[] = [];
  for (let i = 0; i < n; i++) if (cnt[i] > 0) hasIdx.push(i);
  const m = hasIdx.length;
  const Nh = new Float64Array(m), yh = new Float64Array(m), W = new Float64Array(m).fill(1);
  for (let j = 0; j < m; j++) {
    const i = hasIdx[j];
    Nh[j] = Nfirst[i];
    yh[j] = T[i] - off[i] * Ts - beta * nz(exp[i]);
  }
  let a = 0, dt = 0;
  const r = new Float64Array(m);
  for (let it = 0; it < 3; it++) {
    [a, dt] = wline(Nh, yh, W);
    for (let j = 0; j < m; j++) r[j] = yh[j] - (a + Nh[j] * dt);
    const absr = r.map(Math.abs);
    const sc = 1.4826 * median(absr) + 1e-9;
    for (let j = 0; j < m; j++) {
      const w = 1.0 / Math.max(Math.abs(r[j]) / (3 * sc), 1.0);
      W[j] = w * w;
    }
  }
  const res = new Float64Array(n).fill(NaN);
  for (let j = 0; j < m; j++) res[hasIdx[j]] = yh[j] - (a + Nh[j] * dt);
  return { a, dt, res };
}

/** Free fit of the exposure coefficient over the longest segment (telemetry.py _fit_beta). */
function fitBeta(T: Float64Array, off: Float64Array, cnt: Int32Array, exp: Float64Array, rate: number,
                 segs: Array<[number, number]>): number | null {
  let [a, b] = segs[0];
  for (const s of segs) if (s[1] - s[0] > b - a) [a, b] = s;
  const Ns: number[] = [], es: number[] = [], ys: number[] = [];
  let acc = 0;
  for (let i = a; i <= b; i++) {
    if (cnt[i] > 0) { Ns.push(acc); es.push(nz(exp[i])); ys.push(T[i] - off[i] / rate); }
    acc += cnt[i];
  }
  if (Ns.length < 20) return null;
  if (std(es) < 2e-4) return null;
  const n = Ns.length;
  const mN = mean(Ns), me = mean(es), y0 = ys[0];
  let my = 0;
  for (let i = 0; i < n; i++) my += ys[i] - y0;
  my /= n;
  let sNN = 0, see = 0, sNe = 0, sNy = 0, sey = 0;
  for (let i = 0; i < n; i++) {
    const dN = Ns[i] - mN, de = es[i] - me, dy = ys[i] - y0 - my;
    sNN += dN * dN; see += de * de; sNe += dN * de; sNy += dN * dy; sey += de * dy;
  }
  const det = sNN * see - sNe * sNe;
  if (!(Math.abs(det) > 0)) return null;
  return (sNN * sey - sNe * sNy) / det;
}

// ------------------------------------------------------------------------------------------------ quaternions

type Q = [number, number, number, number];

function qmul(a: ArrayLike<number>, b: ArrayLike<number>, out: Float64Array | number[] = [0, 0, 0, 0], o = 0) {
  const aw = a[0], ax = a[1], ay = a[2], az = a[3];
  const bw = b[0], bx = b[1], by = b[2], bz = b[3];
  out[o] = aw * bw - ax * bx - ay * by - az * bz;
  out[o + 1] = aw * bx + ax * bw + ay * bz - az * by;
  out[o + 2] = aw * by - ax * bz + ay * bw + az * bx;
  out[o + 3] = aw * bz + ax * by - ay * bx + az * bw;
  return out;
}
function qconj(q: ArrayLike<number>): Q {
  return [q[0], -q[1], -q[2], -q[3]];
}
function qnorm(q: ArrayLike<number>, o = 0): number {
  return Math.sqrt(q[o] * q[o] + q[o + 1] * q[o + 1] + q[o + 2] * q[o + 2] + q[o + 3] * q[o + 3]);
}
function qnormalizeQ(q: ArrayLike<number>): Q {
  const n = qnorm(q);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}
/** In place: rows of an (n,4) array made sign-continuous (geom.qfix_sign). */
function qfixSign(q: Float64Array) {
  // numpy semantics: flips = cumsum(dot(orig[i], orig[i-1]) < 0) % 2 (dots of the ORIGINAL neighbours)
  const n = q.length / 4;
  let parity = 0;
  let p0 = q[0], p1 = q[1], p2 = q[2], p3 = q[3];
  for (let i = 1; i < n; i++) {
    const o = 4 * i;
    const c0 = q[o], c1 = q[o + 1], c2 = q[o + 2], c3 = q[o + 3];
    if (c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3 < 0) parity ^= 1;
    p0 = c0; p1 = c1; p2 = c2; p3 = c3;
    if (parity) { q[o] = -c0; q[o + 1] = -c1; q[o + 2] = -c2; q[o + 3] = -c3; }
  }
}
function qlog(q: ArrayLike<number>): [number, number, number] {
  let w = q[0], x = q[1], y = q[2], z = q[3];
  if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  w = Math.min(1, Math.max(-1, w));
  const s = Math.sqrt(x * x + y * y + z * z);
  const th = 2.0 * Math.atan2(s, w);
  const k = s < 1e-8 ? 2.0 / (w === 0 ? 1.0 : w) : th / s;
  return [x * k, y * k, z * k];
}
function qexp(v: ArrayLike<number>): Q {
  const th = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const half = 0.5 * th;
  const k = th < 1e-8 ? 0.5 - th * th / 48.0 : Math.sin(half) / th;
  return [Math.cos(half), v[0] * k, v[1] * k, v[2] * k];
}

/** geom.slerp_series (t_src strictly increasing, q_src sign-continuous (n,4)); queries clamp to the ends. */
function slerpAt(tSrc: Float64Array, qSrc: Float64Array, t: number): Q {
  const n = tSrc.length;
  // searchsorted(side='right') - 1
  let lo = 0, hi = n;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (tSrc[mid] <= t) lo = mid + 1; else hi = mid; }
  const i = Math.min(Math.max(lo - 1, 0), n - 2);
  const t0 = tSrc[i], t1 = tSrc[i + 1];
  const u = Math.min(1, Math.max(0, (t - t0) / Math.max(t1 - t0, 1e-12)));
  const q0 = qSrc.subarray(4 * i, 4 * i + 4), q1 = qSrc.subarray(4 * i + 4, 4 * i + 8);
  const d = qmul(qconj(q0), q1) as number[];
  const lg = qlog(d);
  const out = qmul(q0, qexp([lg[0] * u, lg[1] * u, lg[2] * u])) as number[];
  return qnormalizeQ(out);
}

/** Eigenvector of the largest eigenvalue of a symmetric 4x4 matrix (cyclic Jacobi). */
function topEigenvector4(M: number[][]): Q {
  const a = M.map((r) => r.slice());
  const V = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += a[p][q] * a[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < 4; p++) {
      for (let q = p + 1; q < 4; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < 4; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i++) if (a[i][i] > a[best][best]) best = i;
  return [V[0][best], V[1][best], V[2][best], V[3][best]];
}

// ------------------------------------------------------------------------------------------------ formatting

/** Python repr() of a float / list / None (for warnings that must match the engine's text). */
function pyRepr(v: unknown): string {
  if (v === null || v === undefined) return 'None';
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return 'nan';
    if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
    if (Number.isInteger(v) && Math.abs(v) < 1e16) return `${v}`;
    const s = `${v}`;
    // JS uses 1e-7 / 1e+21 style; Python uses 1e-07 / 1e+21
    return s.replace(/e([+-])(\d)$/, 'e$10$2');
  }
  if (Array.isArray(v)) return `[${v.map(pyFloatRepr).join(', ')}]`;
  return String(v);
}
function pyFloatRepr(v: unknown): string {
  if (typeof v === 'number' && Number.isInteger(v) && Math.abs(v) < 1e16) return `${v}.0`;
  return pyRepr(v);
}
/** Python f'{x:.{d}f}' (round-half-even on exact ties). */
function pyFixed(x: number, d: number): string {
  const s = x.toFixed(d);
  const scaled = x * 10 ** d;
  if (Number.isFinite(scaled) && Math.abs(scaled % 1) === 0.5 && Number.isInteger(scaled * 2)) {
    const r = Math.round(scaled);            // half up
    if (r % 2 !== 0) return ((r - Math.sign(scaled)) / 10 ** d).toFixed(d);
  }
  return s;
}

function deepEq(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => deepEq(x, b[i]));
  return a === b;   // NaN != NaN, like Python
}

// ================================================================================================= main entry

function sparseFrames(m: number, n = DBGI_SPARSE_N): number[] {
  if (m <= 3 * n) return Array.from({ length: m }, (_, i) => i);
  const mid = Math.floor(m / 2) - Math.floor(n / 2);
  const s = new Set<number>();
  for (let i = 0; i < n; i++) { s.add(i); s.add(mid + i); s.add(m - n + i); }
  return [...s].sort((a, b) => a - b);
}

function segmentsFromT(T: Float64Array, clipStarts: number[]) {
  const n = T.length;
  const dT = new Float64Array(Math.max(0, n - 1));
  for (let i = 0; i + 1 < n; i++) dT[i] = T[i + 1] - T[i];
  const med = dT.length ? median(dT) : 1 / 60;
  const disc = new Set<number>();
  for (let i = 0; i < dT.length; i++) if (dT[i] < 0.5 * med || dT[i] > 1.5 * med) disc.add(i + 1);
  const starts = [...new Set([0, ...clipStarts.filter((s) => s > 0 && s < n), ...disc])].sort((a, b) => a - b);
  const clipBounds = starts.slice(1).map((s) => ({ frame: s, continuous: !disc.has(s), dT_s: T[s] - T[s - 1],
                                                    header_repeat: clipStarts.includes(s) }));
  const segStarts = [0, ...[...disc].sort((a, b) => a - b)];
  const segs: Array<[number, number]> = segStarts.map((a, i) => [a, (i + 1 < segStarts.length ? segStarts[i + 1] : n) - 1]);
  return { segs, clipBounds };
}

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) throw signal.reason ?? new DOMException('aborted', 'AbortError');
}

/**
 * Parse a DJI O3 / O4 Pro / Osmo Action 4 clip's telemetry. `info` is openMp4(file). onProgress gets 0..1.
 */
export async function loadTelemetry(file: Blob, info: Mp4Info, onProgress?: (f: number) => void,
                                    opts: LoadTelemetryOptions = {}): Promise<Telemetry> {
  const signal = opts.signal;
  const userProgress = onProgress;
  let lastP = -1;
  onProgress = userProgress && ((f: number) => {        // throttle: ~100 calls per parse
    if (f >= 1 || f - lastP >= 0.01) { lastP = f; userProgress(f); }
  });
  const vt = mainVideoTrack(info);
  const framePtsAll = presentationTimes(vt);
  const F_video = framePtsAll.length;
  const fps = trackFps(vt);
  const W = vt.width ?? 0, H = vt.height ?? 0;
  let djmd = findTrack(info, { fourcc: 'djmd', handlerName: 'DJI meta' });
  if (!djmd) djmd = findTrack(info, { handlerName: 'CAM meta' });
  if (!djmd) throw new Error('no DJI djmd telemetry track (not a DJI O3 / O4 / Osmo Action 4 clip?)');
  let dbgi: Mp4Track | undefined = findTrack(info, { fourcc: 'dbgi', handlerName: 'DJI dbgi' });
  if (dbgi && !dbgi.sampleCount) dbgi = undefined;
  for (const t of [djmd, dbgi]) {
    if (t && t.offsets.some((o) => Number.isNaN(o))) throw new Error(`track ${t.id} (${t.fourcc}): sample table has no chunk offsets`);
  }
  const maxFrames = opts.maxFrames !== undefined ? Math.max(2, Math.floor(opts.maxFrames)) : undefined;
  const nRead = maxFrames === undefined ? djmd.sampleCount : Math.min(djmd.sampleCount, maxFrames);
  const conc = opts.concurrency ?? 8;

  // ---- peek at the clip header (djmd sample 0): OA4 -> read the sparse dbgi frames with the djmd samples
  const mDbgi = dbgi ? Math.min(dbgi.sampleCount, nRead, F_video) : 0;
  let dbgiIdx: number[] | null = null;
  if (mDbgi) {
    let hdr: ClipMeta | null = null;
    try {
      const d0 = pb((await readSamplesAt(file, djmd, [0]))[0]);
      const h = d0.get(1);
      if (h && h[0].b) hdr = parseClipMeta(h[0].b);
    } catch { /* decided after the parse */ }
    if (!hdr || productFlags(hdr)[1]) dbgiIdx = sparseFrames(mDbgi);
  }
  throwIfAborted(signal);
  const nDb = dbgiIdx ? dbgiIdx.length : 0;
  const offs = new Float64Array(nRead + nDb), sizes = new Float64Array(nRead + nDb);
  offs.set(djmd.offsets.subarray(0, nRead));
  for (let i = 0; i < nRead; i++) sizes[i] = djmd.sizes[i];
  for (let j = 0; j < nDb; j++) { offs[nRead + j] = dbgi!.offsets[dbgiIdx![j]]; sizes[nRead + j] = dbgi!.sizes[dbgiIdx![j]]; }
  const raw = await readRanges(file, offs, sizes, { concurrency: conc, signal,
                                                    onProgress: (f) => onProgress?.(0.85 * f) });
  const D = parseDjmd(raw.slice(0, nRead), (f) => onProgress?.(0.85 + 0.1 * f));
  const dbgiRaw = raw.slice(nRead);
  throwIfAborted(signal);
  if (!D.clips.length) throw new Error('djmd has no clip header');
  const [, clip, smeta] = D.clips[0];
  const warn: string[] = [];
  let F = F_video;
  const nd = D.T.length;
  let framePts = framePtsAll;
  let quick: Record<string, unknown> | null = null;
  if (maxFrames !== undefined) {
    quick = { frames: Math.min(nd, F), n_frames: F, djmd_samples: djmd.sampleCount };
    if (djmd.sampleCount !== F) warn.push(`djmd samples ${djmd.sampleCount} != video frames ${F}; paired by index, truncated to min`);
    F = Math.min(F, nd);
    framePts = framePts.subarray(0, F);
  }
  let T = D.T, exp = D.exposure, attOff = D.attOff, cnt = D.attCnt, camQ = D.camQ;
  let quats = D.quats;
  if (nd !== F) {
    warn.push(`djmd samples ${nd} != video frames ${F}; paired by index, truncated to min`);
    const m = Math.min(nd, F);
    T = T.subarray(0, m); exp = exp.subarray(0, m); attOff = attOff.subarray(0, m); cnt = cnt.subarray(0, m);
    camQ = camQ.subarray(0, 4 * m);
    let keep = 0;
    for (let i = 0; i < m; i++) keep += cnt[i];
    quats = quats.subarray(0, 4 * keep);
    framePts = framePts.subarray(0, m);
    F = m;
  }
  const proto = clip.proto_file ?? '';
  const product = clip.product_name ?? '';
  const [isO3, isOA4, isO4P] = productFlags(clip);
  if (!(isO3 || isOA4 || isO4P)) {
    warn.push(`unknown DJI product '${product}' (${proto}); generic DJI timing model (fitted beta, c_pic = 0)`);
  }
  if (isO4P) {
    warn.push('DJI O4 Pro: timing model assumed from the O3 (beta fitted = 0.5, picture at frame_ts); ' +
              'axes/timing checked only by the WP-A smoke test');
  }
  const camera = isO3 ? `DJI O3 (${product.replaceAll('DJI ', '')})` : isOA4 ? 'DJI Osmo Action 4' : isO4P ? 'DJI O4 Pro' : product;
  for (const [i, cm] of D.clips.slice(1)) {
    for (const key of ['fx', 'dist_k', 'readout_ns', 'imu_rate', 'eis_status'] as const) {
      const a = (cm as any)[key] ?? null, b = (clip as any)[key] ?? null;
      if (!deepEq(a, b)) warn.push(`clip header at frame ${i} differs in ${key}: ${pyRepr(a)} vs ${pyRepr(b)}`);
    }
  }
  const { segs, clipBounds } = segmentsFromT(T, D.clips.map((c) => c[0]));

  // ---- per-segment affine camera clock -> video timeline
  const segMap = segs.map(([a, b]) => {
    let s: number, c0: number;
    if (b - a >= 1) [s, c0] = affine(T.subarray(a, b + 1), framePts.subarray(a, b + 1));
    else { s = 1.0; c0 = framePts[a] - T[a]; }
    const r: number[] = [];
    for (let i = a; i <= b; i++) r.push(framePts[i] - (c0 + s * T[i]));
    return { frames: [a, b], scale: s, offset: c0, pts_resid_rms_s: std(r), pts_resid_max_s: nanmaxAbs(r) };
  });
  let longest = 0;
  for (let i = 1; i < segMap.length; i++) {
    if (segMap[i].frames[1] - segMap[i].frames[0] > segMap[longest].frames[1] - segMap[longest].frames[0]) longest = i;
  }
  const sMain = segMap[longest].scale;
  const toVideo = (tCam: number, si: number) => segMap[si].offset + segMap[si].scale * tCam;

  if (quick) {
    const nAll = Math.min(djmd.sampleCount, F_video);
    const chk: number[] = [];
    if (nAll > F) {
      // np.unique(np.linspace(F, nAll - 1, K).round())
      const set = new Set<number>();
      const K = QUICK_CLOCK_CHECKS, step = (nAll - 1 - F) / (K - 1);
      for (let i = 0; i < K; i++) set.add(roundHalfEven(i === K - 1 ? nAll - 1 : i * step + F));
      chk.push(...[...set].sort((a, b) => a - b));
    }
    let disc = 0;
    if (chk.length) {
      const bufs = await readSamplesAt(file, djmd, chk, { concurrency: conc, signal });
      for (let j = 0; j < chk.length; j++) {
        const dev = Math.abs(toVideo(frameTs(bufs[j]), segs.length - 1) - framePtsAll[chk[j]]);
        if (dev > 0.5 / fps) disc++;
      }
      if (disc) warn.push(`quick look: frame clock jumps after the first ${F} frames (${disc} of ${chk.length} ` +
                          'sampled frames); a joined file? the full parse splits it into segments');
    }
    quick.clock_checks = chk.length;
    quick.discontinuities = disc;
  }

  // ---- lens / readout
  const ref = KNOWN_LENSES[isO3 ? 'FC8383' : isOA4 ? 'OsmoAction4' : ''];
  let lensSource = 'file';
  let fx: number, kk: number[];
  const fxTruthy = clip.fx !== undefined && clip.fx !== 0;       // Python truthiness: NaN is truthy
  if (fxTruthy && clip.dist_k && clip.dist_k.length === 4) {
    fx = clip.fx as number;
    kk = clip.dist_k.slice();
    if (ref && W === ref.width && H === ref.height) {
      const dk = Math.max(...kk.map((v, i) => Math.abs(v - ref.k[i])));
      if (Math.abs(fx / ref.fx - 1) > 0.01 || dk > 0.01) {
        warn.push(`lens differs from the known ${product} calibration: fx ${pyRepr(fx)} k ${pyRepr(kk)}`);
      }
    }
  } else if (ref) {
    fx = ref.fx;
    kk = ref.k.slice();
    if (W === ref.width && H !== ref.height) {
      lensSource = `fallback: ${product} ${ref.width}x${ref.height} lens centre-cropped to ${W}x${H} ` +
                   '(no lens metadata in this mode; unverified)';
    } else {
      lensSource = `fallback: ${product} ${ref.width}x${ref.height} lens scaled (unverified)`;
      fx *= W / ref.width;
    }
    warn.push(lensSource);
  } else {
    throw new Error(`no lens metadata and no reference lens for '${product}'`);
  }
  const lens: Lens = { model: 'kb4', fx, fy: fx, cx: (W - 1) / 2.0, cy: (H - 1) / 2.0,
                       k: [kk[0], kk[1], kk[2], kk[3]], width: W, height: H };
  let readoutNs = clip.readout_ns ?? 0;
  let readoutSource = 'file';
  if (!readoutNs) {
    if (ref) {
      readoutNs = ref.readout_ns * (W === ref.width ? H / ref.height : 1.0);
      readoutSource = 'fallback: reference readout scaled by row count (unverified)';
      warn.push(readoutSource);
    } else {
      readoutNs = 0;
    }
  }
  const readoutS = readoutNs * 1e-9 * sMain;

  // ---- EIS
  const eisStatus = clip.eis_status;
  const fmtComment = info.comment ?? '';
  let eisBaked = headerEisBaked(clip, fmtComment);
  const extra: Record<string, unknown> = {
    product, proto_file: proto, firmware: clip.firmware, lens_source: lensSource, readout_source: readoutSource,
    readout_raw_s: readoutNs * 1e-9, read_direction: clip.read_direction ?? null, eis_status: eisStatus,
    eis_status_name: eisStatus === null ? 'None' : (EIS_STATUS[eisStatus] ?? String(eisStatus)),
    format_comment: fmtComment, sensor_fps_meta: clip.sensor_fps ?? null, imu_rate_nominal: clip.imu_rate ?? null,
    fov_type: smeta.fov_type ?? null, clip_bounds: clipBounds, segment_time_maps: segMap, warnings: warn,
    parser_version: PARSER_VERSION, meta_reader: 'mp4 (web)', iso: D.iso.subarray(0, F), frame_ts_cam: T,
  };
  if (quick) extra.quick = quick;

  // ---- per-frame camera attitude (OA4 3.2.9, gravity aligned)
  let haveCq = F > 0;
  let anyFinite = false;
  for (let k = 0; k < F; k++) {
    const n = qnorm(camQ, 4 * k);
    if (!(Number.isFinite(n) && Math.abs(n - 1) < 0.05)) haveCq = false;
    for (let j = 0; j < 4; j++) if (Number.isFinite(camQ[4 * k + j])) anyFinite = true;
  }
  if (F > 0 && !haveCq && anyFinite) warn.push('per-frame camera attitude present but invalid (zero/non-unit); ignored');

  // ---- high-rate attitude
  const rateNom = Number(clip.imu_rate || 0);
  const haveHr = D.nQuats > 0 && quats.length > 0 && rateNom > 0;
  let beta = isO3 || isO4P ? 0.5 : 0.0;
  let expRef = 0.0;
  let betaFit: number | null = null;
  if (haveHr && nanstd(exp) > 2e-4) {
    betaFit = fitBeta(T, attOff, cnt, exp, rateNom, segs);
    if (!(isO3 || isO4P) && betaFit !== null) {
      beta = Math.abs(betaFit - 0.5) < 0.05 ? 0.5 : Math.min(1, Math.max(0, betaFit));
      if (isOA4) {
        expRef = 1.0 / 61.0;
        warn.push(`OA4 clip with varying exposure: fitted beta=${pyFixed(betaFit, 3)}; picture offset from the ` +
                  'exposure model (short shutters: 0012 calibration, 1/61 s: 0005/0006)');
      }
    } else if (betaFit !== null && Math.abs(betaFit - beta) > 0.05) {
      warn.push(`fitted exposure coefficient ${pyFixed(betaFit, 3)} differs from the model ${pyFloatRepr(beta)}`);
    }
  }
  const cPic = new Float64Array(F).fill(isOA4 ? OA4_FRAME_CENTER_OFFSET_S : 0.0);
  const oa4Model = isOA4 && haveHr;
  if (oa4Model) {
    for (let k = 0; k < F; k++) cPic[k] = oa4PictureOffset(exp[k]);
    let nOk = 0, nUnval = 0;
    for (let k = 0; k < F; k++) {
      const e = exp[k];
      if (Number.isFinite(e) && e > 0) {
        nOk++;
        if (e > OA4_SHORT_SHUTTER_MAX_S + 2e-4 && e < OA4_LONG_SHUTTER_S - 5e-4) nUnval++;
      }
    }
    if (nOk) {
      const unval = nUnval / nOk;
      if (unval > 0.05) {
        warn.push(`OA4: ${pyFixed(unval * 100, 0)}% of frames have exposures between 4.6 ms and 1/61 s, where the ` +
                  'picture timing is interpolated (unvalidated)');
      }
    }
  }
  const timing: Record<string, unknown> = {
    beta_exposure: beta, beta_fitted: betaFit, exposure_ref_s: expRef,
    picture_offset_s: oa4Model ? median(cPic) : (isOA4 ? OA4_FRAME_CENTER_OFFSET_S : 0.0),
    picture_offset_model: oa4Model ? 'oa4_exposure_v8' : 'constant',
  };
  extra.timing = timing;
  const frameT = new Float64Array(F);
  segs.forEach(([a, b], si) => {
    for (let k = a; k <= b; k++) frameT[k] = toVideo(T[k] - beta * (nz(exp[k]) - expRef) + cPic[k], si);
  });

  let imuT: Float64Array, imuQ: Float64Array, imuRate: number, hasHighrate: boolean;
  let gravityQ: Float64Array | undefined;
  if (haveHr) {
    const firstQ = new Float64Array(F + 1);            // index of frame k's first quaternion
    for (let k = 0; k < F; k++) firstQ[k + 1] = firstQ[k] + cnt[k];
    const tParts: Float64Array[] = [], qParts: Float64Array[] = [];
    const gridDiag: Array<Record<string, unknown>> = [];
    const expShift = new Float64Array(F);
    for (let k = 0; k < F; k++) expShift[k] = nz(exp[k]) - expRef;
    segs.forEach(([a, b], si) => {
      const c = cnt.subarray(a, b + 1);
      let csum = 0;
      for (const v of c) csum += v;
      if (csum === 0) return;
      const g = fitGrid(T.subarray(a, b + 1), attOff.subarray(a, b + 1), c, expShift.subarray(a, b + 1), rateNom, beta);
      const q0 = firstQ[a], q1 = firstQ[b + 1];
      const nSeg = q1 - q0;
      const keep = new Uint8Array(nSeg).fill(1);
      const medC = median(Array.from(c).filter((v) => v > 0));
      const cl = c[c.length - 1];
      if (cl > 0 && cl < 0.8 * medC) {
        keep.fill(0, nSeg - cl);
        gridDiag.push({ segment: si, dropped_truncated_last_block: cl });
      }
      let nKeep = 0;
      for (let j = 0; j < nSeg; j++) {
        const nr = qnorm(quats, 4 * (q0 + j));
        if (!(Number.isFinite(nr) && Math.abs(nr - 1) < 0.05)) keep[j] = 0;
        if (keep[j]) nKeep++;
      }
      const tp = new Float64Array(nKeep), qp = new Float64Array(4 * nKeep);
      let o = 0;
      for (let j = 0; j < nSeg; j++) {
        if (!keep[j]) continue;
        tp[o] = toVideo(g.a + j * g.dt, si);
        const p = 4 * (q0 + j);
        const nr = qnorm(quats, p);
        qp[4 * o] = quats[p] / nr; qp[4 * o + 1] = quats[p + 1] / nr;
        qp[4 * o + 2] = quats[p + 2] / nr; qp[4 * o + 3] = quats[p + 3] / nr;
        o++;
      }
      const maxRes = nanmaxAbs(g.res);
      gridDiag.push({ segment: si, a_s: g.a, dt_s: g.dt, resid_rms_s: nanstd(g.res), resid_max_s: maxRes, samples: nKeep });
      if (maxRes > 0.45 / rateNom) {
        warn.push(`segment ${si}: IMU blocks do not tile a uniform grid (max resid ${pyFixed(maxRes * 1e6, 0)} us) -- ` +
                  'samples may be missing');
      }
      tParts.push(tp);
      qParts.push(qp);
    });
    // trim overlaps at internal segment boundaries (two clocks spliced): cut at the frame_t midpoint
    for (let i = 0; i < tParts.length; i++) {
      const [a, b] = segs[i];
      const lo = a > 0 ? 0.5 * (frameT[a - 1] + frameT[a]) : -Infinity;
      const hi = b < F - 1 ? 0.5 * (frameT[b] + frameT[b + 1]) : Infinity;
      const tp = tParts[i], qp = qParts[i];
      let n = 0;
      for (let j = 0; j < tp.length; j++) if (tp[j] > lo && tp[j] <= hi) n++;
      if (n === tp.length) continue;
      const t2 = new Float64Array(n), q2 = new Float64Array(4 * n);
      let o = 0;
      for (let j = 0; j < tp.length; j++) {
        if (!(tp[j] > lo && tp[j] <= hi)) continue;
        t2[o] = tp[j];
        q2.set(qp.subarray(4 * j, 4 * j + 4), 4 * o);
        o++;
      }
      tParts[i] = t2; qParts[i] = q2;
    }
    let total = 0;
    for (const p of tParts) total += p.length;
    imuT = new Float64Array(total);
    const qRaw = new Float64Array(4 * total);
    let o = 0;
    for (let i = 0; i < tParts.length; i++) { imuT.set(tParts[i], o); qRaw.set(qParts[i], 4 * o); o += tParts[i].length; }
    qfixSign(qRaw);
    imuQ = new Float64Array(4 * total);
    const tmp = [0, 0, 0, 0];
    for (let i = 0; i < total; i++) {
      qmul(qRaw.subarray(4 * i, 4 * i + 4), Q_CAM2BODY, tmp);
      const nr = qnorm(tmp);
      imuQ[4 * i] = tmp[0] / nr; imuQ[4 * i + 1] = tmp[1] / nr; imuQ[4 * i + 2] = tmp[2] / nr; imuQ[4 * i + 3] = tmp[3] / nr;
    }
    qfixSign(imuQ);
    extra.imu_grid = gridDiag;
    const dts = new Float64Array(Math.max(0, total - 1));
    for (let i = 0; i + 1 < total; i++) dts[i] = imuT[i + 1] - imuT[i];
    imuRate = 1.0 / median(dts);
    hasHighrate = imuRate >= 500;
    if (isO3 || (isO4P && !haveCq)) {
      gravityQ = imuQ;
      extra.gravity_note = 'fused attitude world frame is NED (inferred for the O3; assumed for the O4 Pro); ' +
                           'gravity_q is imu_q';
    } else if (haveCq && segs.length === 1) {
      // OA4: 1 kHz world is NOT gravity aligned; per-frame cam_quat is. W = cq (x) conj(q_raw(t_cq)).
      const Wk: Q[] = [];
      for (let k = 0; k < F; k++) {
        const tc = toVideo(T[k] + OA4_CAM_QUAT_DELAY_S, 0);
        if (!(tc >= imuT[0] && tc <= imuT[total - 1])) continue;
        const qi = slerpAt(imuT, qRaw, tc);
        const cqn = qnormalizeQ(camQ.subarray(4 * k, 4 * k + 4));
        let w = qmul(cqn, qconj(qi)) as number[];
        if (w[0] < 0) w = w.map((v) => -v);
        Wk.push(w as Q);
      }
      if (Wk.length) {
        let Wm = qnormalizeQ([0, 1, 2, 3].map((j) => median(Wk.map((w) => w[j]))));
        const angles = (Wref: Q) => Wk.map((w) => {
          const l = qlog(qmul(qconj(Wref), w));
          return Math.sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
        });
        for (let it = 0; it < 2; it++) {
          const ang = angles(Wm);
          const thr = Math.max(3 * median(ang), 0.5 * (Math.PI / 180));
          const M = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
          Wk.forEach((w, i) => {
            if (!(ang[i] < thr)) return;
            for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) M[r][c] += w[r] * w[c];
          });
          Wm = topEigenvector4(M);
          if (Wm[0] < 0) Wm = Wm.map((v) => -v) as Q;
        }
        const angDeg = angles(Wm).map((a) => a * (180 / Math.PI));
        gravityQ = new Float64Array(4 * total);
        for (let i = 0; i < total; i++) qmul(Wm, imuQ.subarray(4 * i, 4 * i + 4), gravityQ, 4 * i);
        qfixSign(gravityQ);
        const lw = qlog(Wm);
        const sortedDeg = Float64Array.from(angDeg).sort();
        const pct = (p: number) => {         // numpy percentile (linear)
          const x = (sortedDeg.length - 1) * p / 100, lo = Math.floor(x), hi = Math.ceil(x);
          return sortedDeg[lo] + (sortedDeg[hi] - sortedDeg[lo]) * (x - lo);
        };
        extra.gravity_W = Wm;
        extra.gravity_W_deg = Math.sqrt(lw[0] * lw[0] + lw[1] * lw[1] + lw[2] * lw[2]) * (180 / Math.PI);
        extra.gravity_W_spread_deg = { mean: mean(angDeg), p95: pct(95), max: sortedDeg[sortedDeg.length - 1] };
      }
    }
  } else {
    if (!haveCq) throw new Error('no high-rate attitude and no per-frame camera attitude');
    // per-frame gravity-aligned attitude (OA4 16:9 / EIS-on), sampled after T_k (mode-dependent delay)
    const wide = Math.abs(W / H - 16 / 9) < 0.02;
    const cqDelay = wide ? OA4_CAM_QUAT_DELAY_16X9_S : OA4_CAM_QUAT_DELAY_S;
    const tAll = new Float64Array(F);
    segs.forEach(([a, b], si) => { for (let k = a; k <= b; k++) tAll[k] = toVideo(T[k] + cqDelay, si); });
    timing.cam_quat_delay_s = cqDelay;
    timing.cam_quat_delay_source = wide ? 'measured vs video on OA4 16:9 clip 0002 (+-0.7 ms); refine by vision'
      : 'measured vs the 1 kHz stream on OA4 4:3 clips 0005/0006';
    const qAll = new Float64Array(4 * F);
    const tmp = [0, 0, 0, 0];
    for (let k = 0; k < F; k++) {
      const cqn = qnormalizeQ(camQ.subarray(4 * k, 4 * k + 4));
      qmul(cqn, Q_CAM2BODY, tmp);
      const nr = qnorm(tmp);
      qAll[4 * k] = tmp[0] / nr; qAll[4 * k + 1] = tmp[1] / nr; qAll[4 * k + 2] = tmp[2] / nr; qAll[4 * k + 3] = tmp[3] / nr;
    }
    qfixSign(qAll);
    const good: number[] = [];
    for (let k = 0; k < F; k++) if (k === 0 || tAll[k] - tAll[k - 1] > 0) good.push(k);
    imuT = new Float64Array(good.length);
    imuQ = new Float64Array(4 * good.length);
    good.forEach((k, i) => { imuT[i] = tAll[k]; imuQ.set(qAll.subarray(4 * k, 4 * k + 4), 4 * i); });
    const dts = new Float64Array(Math.max(0, imuT.length - 1));
    for (let i = 0; i + 1 < imuT.length; i++) dts[i] = imuT[i + 1] - imuT[i];
    imuRate = 1.0 / median(dts);
    hasHighrate = false;
    gravityQ = imuQ;
    extra.gravity_note = 'per-frame cam_quat is gravity aligned (world z down)';
    extra.imu_source = `per-frame camera_attitude (djmd 3.2.9) at T_k + ${pyFixed(cqDelay * 1e3, 1)} ms`;
  }

  // ---- OA4 dbgi: EIS cross-check on the sparse frames read with djmd (mode > 0 anywhere -> EIS active)
  const m = dbgi ? Math.min(dbgi.sampleCount, F) : 0;
  if (isOA4 && m) {
    let idx = dbgiIdx ?? [];
    let bufs = dbgiRaw;
    if (idx.some((i) => i >= m)) {
      const k2 = idx.map((i) => i < m);
      idx = idx.filter((_, j) => k2[j]);
      bufs = bufs.filter((_, j) => k2[j]);
    }
    if (!idx.length) {
      idx = sparseFrames(m);
      bufs = await readSamplesAt(file, dbgi!, idx, { concurrency: conc, signal });
    }
    const G = parseDbgiAc203(bufs);
    extra.dbgi_eis_mode = G.mode;
    if (idx.length < m) extra.dbgi_frames = idx;
    extra.dbgi_info = G.info;
    if (G.mode.some((v) => v > 0) && !eisBaked) {
      warn.push('dbgi reports EIS active although clip header says EIS off; marking eis_baked');
      eisBaked = true;
    }
    extra.eis_note = eisBaked ? 'dbgi_q_eis_cam = EIS output (virtual camera) attitude the picture follows; ' +
                                'imu_q is the PHYSICAL camera' : 'EIS off';
  }

  // frames whose every row time (frame_t +- readout/2) lies inside the IMU span
  let firstOk = -1, lastOk = -1;
  for (let k = 0; k < F; k++) {
    if (frameT[k] - readoutS / 2 >= imuT[0] && frameT[k] + readoutS / 2 <= imuT[imuT.length - 1]) {
      if (firstOk < 0) firstOk = k;
      lastOk = k;
    }
  }
  extra.imu_full_coverage_frames = firstOk >= 0 ? [firstOk, lastOk] : [0, -1];
  const exposureS = Float64Array.from(exp);
  const tel: Telemetry = {
    camera, width: W, height: H, fps, framePts: Float64Array.from(framePts), frameT, exposureS, readoutS, lens,
    imuT, imuQ, imuRate, hasHighrate, eisBaked, gravityQ, warnings: warn,
    segments: segs.map(([a, b]) => [a, b] as [number, number]), extra,
  };
  check(tel);
  onProgress?.(1);
  return tel;
}

function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
}

function check(tel: Telemetry) {
  const F = tel.framePts.length;
  if (tel.frameT.length !== F || tel.exposureS.length !== F) throw new Error('telemetry: frame array lengths differ');
  for (let i = 1; i < tel.imuT.length; i++) if (!(tel.imuT[i] > tel.imuT[i - 1])) throw new Error('imu_t must be strictly increasing');
  for (let i = 1; i < F; i++) if (!(tel.framePts[i] > tel.framePts[i - 1])) throw new Error('frame_pts must be strictly increasing');
  if (tel.imuQ.length !== 4 * tel.imuT.length) throw new Error('imu_q shape');
  for (let i = 0; i < tel.imuT.length; i++) {
    if (Math.abs(qnorm(tel.imuQ, 4 * i) - 1) > 1e-9) throw new Error('imu_q must be unit');
    if (i && tel.imuQ[4 * i] * tel.imuQ[4 * i - 4] + tel.imuQ[4 * i + 1] * tel.imuQ[4 * i - 3] +
        tel.imuQ[4 * i + 2] * tel.imuQ[4 * i - 2] + tel.imuQ[4 * i + 3] * tel.imuQ[4 * i - 1] < 0) {
      throw new Error('imu_q must be sign-continuous');
    }
  }
}

/** Camera -> world orientation at time t (slerp on the uniform IMU grid; clamps outside). */
export function orientationAt(tel: Telemetry, t: number): [number, number, number, number] {
  return slerpAt(tel.imuT, tel.imuQ, t);
}

/** Convenience: open + parse in one call. */
export async function loadTelemetryFromFile(file: Blob, onProgress?: (f: number) => void,
                                            opts: LoadTelemetryOptions = {}): Promise<{ info: Mp4Info; tel: Telemetry }> {
  const info = await openMp4(file);
  const tel = await loadTelemetry(file, info, onProgress, opts);
  return { info, tel };
}
