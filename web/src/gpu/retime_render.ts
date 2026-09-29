/**
 * Stillpoint web — streaming renderer of a retimed export (owner: GPU agent): Warper + Blender driven by an
 * OutputSchedule (retime.ts buildSchedule). Feed every decoded source frame in presentation order; get back the output
 * frames that became complete, in output order, stamped with the schedule's pts.
 *
 *   const rr = await RetimeRenderer.create(warper, schedule, { transfer: blendTransferFor(src.colorSpace) });
 *   for (each decoded frame f, presentation index k) {          // k = plan record = schedule tap index
 *     for (const out of rr.push(f, k)) { encoder.encode(out); out.close(); }
 *     f.close();
 *   }
 *   if (!rr.done) ...;  rr.destroy();
 *
 * Per source frame k:
 *   - not tapped by any output (e.g. 60 -> 24 fps without blur): nothing is warped;
 *   - tapped only by single-tap outputs: warper.warp(frame, k, timing) straight into an output VideoFrame (bit-identical
 *     to a non-retimed export); duplicates (output fps > source fps) are re-stamped clones of it, not re-warped;
 *   - tapped by a multi-tap (motion-blur) output: warped once into a pooled texture (Warper.warpTo, 4 B/output px)
 *     that stays in a ring until the last output using it is emitted; outputs are Blender.blend() of their taps in
 *     linear light (single-tap outputs of such frames = exact copies of the texture).
 * Output i is emitted as soon as frame lastSourceFrame[i] has been pushed. Warped textures held at once <= the
 * schedule's maxSpan (+1 while the next is warped) — e.g. 2-3 for 60 -> 24/25/30 at 180°; the Blender adds a float
 * accumulator (16 B/px) only for outputs with > 3 taps.
 */
import type { OutputSchedule } from '../types';
import type { Warper } from './warp';
import { Blender, type BlendTransfer } from './blend';

export interface RetimeRendererOptions {
  /** transfer of the warped pictures for the linear-light blend (default 'bt709'; see blendTransferFor) */
  transfer?: BlendTransfer;
  /** plan record the Warper uses for schedule source frame k (default k). An export of a range schedules the range's
   *  frames 0..n-1 and maps them to their plan records here. */
  record?: (k: number) => number;
}

export interface RetimeStats {
  /** warp() calls straight into output frames / warpTo() calls into ring textures */
  directWarps: number;
  ringWarps: number;
  /** outputs made by blend() / by re-stamping a previous output (duplicate frames) */
  blends: number;
  clones: number;
  /** source frames skipped (tapped by no output) */
  skipped: number;
  /** most ring textures alive at once, and textures allocated in total */
  peakRing: number;
  texturesAllocated: number;
}

export class RetimeRenderer {
  readonly schedule: OutputSchedule;
  readonly stats: RetimeStats = { directWarps: 0, ringWarps: 0, blends: 0, clones: 0, skipped: 0, peakRing: 0, texturesAllocated: 0 };
  private readonly warper: Warper;
  private readonly blender: Blender | null;
  private readonly record: (k: number) => number;
  /** per source frame: last output that uses it (-1 = none) */
  private readonly lastUse: Int32Array;
  /** per source frame: tapped by an output with > 1 tap (-> keep a warped texture) */
  private readonly needsTex: Uint8Array;
  private readonly ring = new Map<number, GPUTexture>();
  private readonly pool: GPUTexture[] = [];
  private next = 0;
  private lastK = -1;
  private destroyed = false;

  private constructor(warper: Warper, schedule: OutputSchedule, blender: Blender | null, lastUse: Int32Array, needsTex: Uint8Array,
                      record: (k: number) => number) {
    this.warper = warper; this.schedule = schedule; this.blender = blender; this.lastUse = lastUse; this.needsTex = needsTex;
    this.record = record;
  }

  static async create(warper: Warper, schedule: OutputSchedule, opts: RetimeRendererOptions = {}): Promise<RetimeRenderer> {
    const s = schedule;
    let nSrc = 0, multi = false;
    for (let j = 0; j < s.taps.length; j++) nSrc = Math.max(nSrc, s.taps[j] + 1);
    const lastUse = new Int32Array(nSrc).fill(-1);
    const needsTex = new Uint8Array(nSrc);
    for (let i = 0; i < s.n; i++) {
      const a = s.tapStart[i], c = s.tapsPerFrame[i];
      if (c < 1) throw new Error(`schedule: output ${i} has no taps`);
      for (let j = a; j < a + c; j++) {
        const k = s.taps[j];
        if (k < 0) throw new Error(`schedule: output ${i} taps source frame ${k}`);
        lastUse[k] = i;
        if (c > 1) { needsTex[k] = 1; multi = true; }
        if (k > s.lastSourceFrame[i]) throw new Error(`schedule: output ${i} taps ${k} > lastSourceFrame ${s.lastSourceFrame[i]}`);
      }
      if (i > 0 && s.lastSourceFrame[i] < s.lastSourceFrame[i - 1]) throw new Error('schedule: lastSourceFrame must not decrease');
    }
    const blender = multi ? await Blender.create(warper.device, warper.width, warper.height, { transfer: opts.transfer ?? 'bt709' }) : null;
    return new RetimeRenderer(warper, schedule, blender, lastUse, needsTex, opts.record ?? (k => k));
  }

  /** Output frames emitted so far / all output frames done. */
  get emitted(): number { return this.next; }
  get done(): boolean { return this.next >= this.schedule.n; }

  /** Push source frame k (presentation order, increasing; every frame an output taps must be pushed). Returns the
   *  output frames completed by it (caller encodes and closes them; `frame` may be closed right after). */
  push(frame: VideoFrame, k: number): VideoFrame[] {
    if (this.destroyed) throw new Error('RetimeRenderer destroyed');
    if (!(k > this.lastK)) throw new Error(`push: frame ${k} after ${this.lastK} (presentation order required)`);
    this.lastK = k;
    const s = this.schedule, out: VideoFrame[] = [];
    const used = k < this.lastUse.length && this.lastUse[k] >= 0;
    if (!used) this.stats.skipped++;
    if (used && this.needsTex[k]) {
      let t = this.pool.pop();
      if (!t) { t = this.warper.createTarget('sp.retime.ring'); this.stats.texturesAllocated++; }
      this.warper.warpTo(frame, this.record(k), t);
      this.ring.set(k, t);
      this.stats.ringWarps++;
      this.stats.peakRing = Math.max(this.stats.peakRing, this.ring.size);
    }
    let direct: VideoFrame | null = null;
    try {
      while (this.next < s.n && s.lastSourceFrame[this.next] <= k) {
        const i = this.next;
        const a = s.tapStart[i], c = s.tapsPerFrame[i];
        const timing = { timestamp: Math.round(s.outPtsUs[i]), duration: Math.round(s.frameDurUs) };
        if (c === 1 && s.taps[a] === k && !this.ring.has(k)) {
          if (!direct) { direct = this.warper.warp(frame, this.record(k), timing); out.push(direct); this.stats.directWarps++; }
          else { out.push(new VideoFrame(direct, timing)); this.stats.clones++; }
        } else {
          const inputs: GPUTexture[] = [], w: number[] = [];
          for (let j = a; j < a + c; j++) {
            const tex = this.ring.get(s.taps[j]);
            if (!tex) throw new Error(`output ${i}: source frame ${s.taps[j]} was not pushed`);
            inputs.push(tex); w.push(s.weights[j]);
          }
          out.push(this.blender!.blend(inputs, w, timing));
          this.stats.blends++;
        }
        this.next++;
        for (const [kk, tex] of this.ring) if (this.lastUse[kk] <= i) { this.ring.delete(kk); this.pool.push(tex); }
      }
    } catch (e) {
      for (const f of out) f.close();
      throw e;
    }
    return out;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const t of this.ring.values()) t.destroy();
    for (const t of this.pool) t.destroy();
    this.ring.clear(); this.pool.length = 0;
    this.blender?.destroy();
  }
}
