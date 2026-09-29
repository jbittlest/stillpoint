/**
 * Export-panel rules that do not need the DOM (unit-tested in test/app.units.test.ts): the custom width / frame-rate
 * fields, which timing (real time / slow motion) an output rate allows, and what the done card says about sound.
 */
import { SAME_RATE_TOL } from '../retime';

export const CUSTOM_W = [160, 7680] as const;
export const CUSTOM_FPS = [1, 240] as const;
/** custom width: even, inside CUSTOM_W */
export const evenClamp = (w: number) => Math.max(CUSTOM_W[0], Math.min(CUSTOM_W[1], 2 * Math.round(w / 2)));
/** custom rate: 3 decimals, inside CUSTOM_FPS */
export const fpsClamp = (f: number) => Math.max(CUSTOM_FPS[0], Math.min(CUSTOM_FPS[1], Math.round(f * 1000) / 1000));

export interface FieldRead {
  /** value to apply while the user is still typing (null: nothing usable yet — keep the current value) */
  live: number | null;
  /** value a commit (change / Enter / blur) applies and writes back into the field: the typed number clamped to the
   *  range (9000 px -> 7680, 300 fps -> 240, 50 px -> 160), or the current value when nothing numeric was typed */
  commit: number;
  /** error state while typing: above the maximum (no amount of further typing makes it valid). Below the minimum is
   *  treated as unfinished (1 -> 19 -> 1920), not an error; a commit clears it either way */
  bad: boolean;
  /** show the valid range next to the field (typed value outside it) */
  hint: boolean;
}

/** Read a custom number field (`raw` = input.value; `last` = the value in use). */
export function readNumberField(raw: string, range: readonly [number, number], last: number, clamp: (v: number) => number): FieldRead {
  const t = raw.trim();
  const v = t === '' ? NaN : Number(t);
  if (!Number.isFinite(v)) return { live: null, commit: clamp(last), bad: false, hint: t !== '' };
  if (v >= range[0] && v <= range[1]) { const c = clamp(v); return { live: c, commit: c, bad: false, hint: false }; }
  return { live: null, commit: clamp(v), bad: v > range[1], hint: true };
}

export type Timing = 'realtime' | 'slowmo';

export interface TimingState {
  /** timing the export uses: slow motion only when the output rate is really below the source's */
  timing: Timing;
  /** the Slow motion option can be chosen at this output rate */
  slowmoOk: boolean;
  /** playback speed of the export (output / source rate in slow motion, else 1) */
  speed: number;
  /** speed the Slow motion option gives at this rate (1 when it is not available) */
  slowmoSpeed: number;
  /** output and source rates are the same within SAME_RATE_TOL (e.g. 60 from 59.94: a real-time conform) */
  sameRate: boolean;
}

/**
 * Timing for an export at `outFps` from a `srcFps` clip when the user prefers `want`. Slow motion (every source frame
 * at the output rate, sound dropped) needs an output rate below the source's by more than SAME_RATE_TOL: at the same
 * rate (59.94 -> 60 is 0.1 %) it would only drop the sound, and above it it would play faster, not slower — then the
 * export is real time (sound kept) and the option is disabled with a hint.
 */
export function exportTiming(srcFps: number, outFps: number, fps: number | 'source', want: Timing): TimingState {
  const known = srcFps > 0 && outFps > 0;
  const sameRate = fps === 'source' || (known && Math.abs(outFps / srcFps - 1) < SAME_RATE_TOL);
  const slowmoOk = fps !== 'source' && known && !sameRate && outFps < srcFps;
  const slowmoSpeed = slowmoOk ? outFps / srcFps : 1;
  const timing: Timing = want === 'slowmo' && slowmoOk ? 'slowmo' : 'realtime';
  return { timing, slowmoOk, speed: timing === 'slowmo' ? slowmoSpeed : 1, slowmoSpeed, sameRate };
}

/** Sub-label of the Slow motion option. */
export function slowmoLabel(t: TimingState, fmt: (x: number) => string): string {
  if (t.slowmoOk) return `${fmt(Math.round((1 / t.slowmoSpeed) * 100) / 100)}× slower`;
  return 'needs a lower rate';
}

/** What the done card says about sound ('' = nothing to say: the source's sound was copied). */
export function doneAudioNote(r: { sourceAudio?: boolean; audioDropped?: boolean; audioPackets?: number; timeMode?: string }): string {
  if (r.audioDropped) return r.timeMode === 'slowmo' ? 'no sound (slow motion)' : 'no sound (speed changed)';
  if (r.sourceAudio === false) return 'no sound in the source clip';
  if (r.sourceAudio && r.audioPackets === 0) return 'sound not copied';
  return '';
}
