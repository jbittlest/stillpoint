/** Messages between the UI (main thread) and the two workers. */
import type { Mp4Info, Plan } from '../types';
import type { StabParams } from '../ui/contracts';
import type { RenderProgress, RenderResult } from './pipeline';
import type { SinkRequest } from './mux';
import type { OutCodec } from './encode';
import type { DecodeReport, RapKind, StreamFacts } from './decode';

/** What the UI needs to know about the telemetry (the full arrays stay in the analysis worker). */
export interface TelemetrySummary {
  camera: string;
  width: number;
  height: number;
  fps: number;
  frames: number;
  imuRate: number;
  hasHighrate: boolean;
  eisBaked: boolean;
  readoutS: number;
  lensModel: string;
  warnings: string[];
  medianExposureS: number;
}

export interface PlanSummary {
  outW: number;
  outH: number;
  srcW: number;
  srcH: number;
  frames: number;
  /** median output focal length, px */
  outFx: number;
  /** approximate horizontal field of view of the output, degrees */
  hfovDeg: number;
}

// ── analysis worker ──
export type AnalysisIn =
  | { type: 'analyze'; gen: number; file: File }
  | { type: 'plan'; gen: number; id: number; params: StabParams };

/** every analysis message carries `gen`: the clip it belongs to (messages for an older clip are dropped) */
export type AnalysisOut = ({ gen: number }) & (
  | { type: 'info'; info: Mp4Info }
  | { type: 'progress'; stage: 'telemetry' | 'plan'; f: number; id?: number }
  | { type: 'telemetry'; summary: TelemetrySummary }
  | { type: 'plan'; id: number; plan: Plan; summary: PlanSummary; ms: number }
  | { type: 'error'; stage: 'open' | 'telemetry' | 'plan'; message: string; id?: number });

// ── engine worker ──
export interface ExportSettings {
  bitrate: number;
  prefer?: OutCodec;
  first: number;
  last: number;
  includeAudio: boolean;
  sink: SinkRequest;
}

/** test / support switches, from the page URL (?sp_fault=hw-first&sp_stall=3000) */
export interface EngineDebug {
  /** simulate decoder failures (see decode.ts parseFault) */
  fault?: string;
  /** decoder stall watchdog, ms */
  stallMs?: number;
}

export type EngineIn =
  | { type: 'init'; before?: OffscreenCanvas; after?: OffscreenCanvas; debug?: EngineDebug }
  | { type: 'resize'; w: number; h: number }
  | { type: 'open'; file: File; info: Mp4Info }
  | { type: 'plan'; plan: Plan; id: number }
  | { type: 'seek'; pres: number }
  | { type: 'play'; from: number; to: number; loop: boolean }
  | { type: 'pause' }
  | { type: 'probe-encoder'; bitrate: number; prefer?: OutCodec }
  | { type: 'export'; settings: ExportSettings }
  | { type: 'cancel' };

export interface EngineCaps {
  webgpu: boolean;
  adapter: string;
  error?: string;
  /** GPUAdapterInfo fields, for diagnostics */
  adapterInfo?: { vendor?: string; architecture?: string; device?: string; description?: string };
  /** worker navigator.platform / hardwareConcurrency, for diagnostics */
  platform?: string;
  cores?: number;
}

/** Which video decoder the engine uses for the open clip (and why). */
export interface DecoderInfo {
  codec: string;
  /** decode variant id: 'hw', 'hw-nocolor', 'auto', 'sw', … */
  variant: string;
  /** 'hardware' | 'software' | 'automatic' | … */
  label: string;
  /** decoding (most likely) in software: slower */
  software: boolean;
  /** not the first choice: a pre-flight test or a runtime error made us fall back */
  fallback: boolean;
  /** the friendly one-line note to show when software decoding is in use because hardware failed */
  note?: string;
  facts: StreamFacts;
  /** what the clip's sync samples are (e.g. { idr: 102 }) */
  rap: Partial<Record<RapKind, number>>;
  preflight?: { ms: number; frames: number; tried: Array<{ variant: string; ok: boolean; ms: number; error?: string }> };
  hwSupported: boolean;
}

/** Diagnostics attached to a decode failure (the UI adds browser/GPU facts and offers "Copy details"). */
export type ErrorDetails = Partial<DecodeReport> & Record<string, unknown>;

export type EngineOut =
  | { type: 'ready'; caps: EngineCaps }
  | { type: 'opened'; frames: number; fps: number; duration: number; decoder: DecoderInfo; notes: string[] }
  | { type: 'open-error'; message: string; details?: ErrorDetails }
  | { type: 'decoder'; decoder: DecoderInfo }
  | { type: 'frame'; pres: number; t: number; hasWarp: boolean; ms: number }
  | { type: 'playing'; playing: boolean; stats?: { drawn: number; dropped: number; seconds: number } }
  | { type: 'plan-applied'; id: number }
  | { type: 'encoder'; codec: string | null; label: string; hardware: boolean; width: number; height: number; fps: number }
  | { type: 'export-progress'; p: RenderProgress }
  | { type: 'export-done'; result: RenderResult }
  | { type: 'export-error'; message: string; cancelled: boolean; details?: ErrorDetails }
  | { type: 'error'; message: string; details?: ErrorDetails };
