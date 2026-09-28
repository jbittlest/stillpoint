/** Messages between the UI (main thread) and the two workers. */
import type { Mp4Info, Plan } from '../types';
import type { StabParams } from '../ui/contracts';
import type { RenderProgress, RenderResult } from './pipeline';
import type { SinkRequest } from './mux';
import type { OutCodec } from './encode';

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

export type EngineIn =
  | { type: 'init'; before?: OffscreenCanvas; after?: OffscreenCanvas }
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
}

export type EngineOut =
  | { type: 'ready'; caps: EngineCaps }
  | { type: 'opened'; frames: number; fps: number; duration: number; decoder: string; hardware: boolean }
  | { type: 'open-error'; message: string }
  | { type: 'frame'; pres: number; t: number; hasWarp: boolean; ms: number }
  | { type: 'playing'; playing: boolean; stats?: { drawn: number; dropped: number; seconds: number } }
  | { type: 'plan-applied'; id: number }
  | { type: 'encoder'; codec: string | null; label: string; hardware: boolean; width: number; height: number; fps: number }
  | { type: 'export-progress'; p: RenderProgress }
  | { type: 'export-done'; result: RenderResult }
  | { type: 'export-error'; message: string; cancelled: boolean }
  | { type: 'error'; message: string };
