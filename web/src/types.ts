/**
 * Shared interfaces of the Stillpoint web stabilizer (owned by the TELEMETRY module; everyone codes against this).
 *
 * Conventions (ENGINE_SPEC.md §1, identical to the Python engine):
 *  - Quaternions (w,x,y,z), Hamilton, float64. An orientation maps CAMERA vectors to WORLD.
 *  - Camera frame: x right, y down, z forward. Pixel centres at integers (a centred lens has cx = (W-1)/2).
 *  - Time: seconds on the VIDEO timeline (frame presentation time). frameT[k] = centre-row mid-exposure of frame k;
 *    row y of frame k is captured at frameT[k] + readoutS*((y+0.5)/H - 0.5).
 *
 * The fields marked "extension" are optional additions beyond the agreed contract (nothing depends on them).
 */

export interface Lens {
  model: 'kb4' | 'pinhole';
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  k: [number, number, number, number];
  width: number;
  height: number;
}

export interface Mp4Track {
  id: number;
  kind: 'video' | 'audio' | 'data' | 'other';
  /** hdlr handler type: 'vide', 'soun', 'meta', 'tmcd', ... */
  handler: string;
  /** codec name: 'h264' | 'hevc' | 'aac' | 'mjpeg' | ... ('' when unknown; then fourcc says it) */
  codec: string;
  /** first sample-description entry type, e.g. 'avc1', 'hvc1', 'mp4a', 'djmd', 'dbgi' */
  fourcc: string;
  timescale: number;
  sampleCount: number;
  /** file offset of every sample (decode order); exact up to 2^53 */
  offsets: Float64Array;
  sizes: Uint32Array;
  /** decode time, s (edit-list shift applied, same shift as cts) */
  dts: Float64Array /*s*/;
  /** presentation time, s (edit list applied: what ffmpeg reports as pts) */
  cts: Float64Array /*s, presentation*/;
  /** 1 = sync sample (keyframe) */
  sync: Uint8Array;
  /** video: avcC / hvcC box payload (WebCodecs VideoDecoderConfig.description); audio (mp4a): AudioSpecificConfig */
  codecConfig?: Uint8Array /*avcC/hvcC box payload*/;
  /** WebCodecs codec string: 'avc1.640034', 'hvc1.2.4.H150.B0', 'mp4a.40.2', ... */
  codecString?: string /*WebCodecs codec string*/;
  width?: number;
  height?: number;
  sampleRate?: number;
  channels?: number;
  colr?: { primaries: number; transfer: number; matrix: number; fullRange: boolean };
  /** extension: hdlr name, e.g. 'VideoHandler', 'DJI meta' */
  handlerName?: string;
  /** extension: track duration in s (media header) */
  durationS?: number;
  /** extension (video): container frame rate as ffprobe's avg_frame_rate (timescale*frames / stts total, reduced) */
  frameRate?: { num: number; den: number };
}

export interface Mp4Info {
  tracks: Mp4Track[];
  durationS: number;
  movieTimescale: number;
  brand: string;
  /** extension: moov/udta/meta/ilst '©cmt' (DJI puts e.g. 'EIS:OFF,FOV:Wide' there), '' when absent */
  comment?: string;
  /** extension: '©too' / '©enc' encoder tag, '' when absent */
  encoder?: string;
}

export interface Telemetry {
  camera: string;
  width: number;
  height: number;
  /** container nominal frame rate (ffmpeg avg_frame_rate) */
  fps: number;
  /** container PTS of every video frame, s, presentation order */
  framePts: Float64Array;
  /** centre-row mid-exposure time of every frame, s (video timeline) */
  frameT: Float64Array;
  exposureS: Float64Array;
  /** full-frame top->bottom readout time, s (video timeline) */
  readoutS: number;
  lens: Lens;
  /** IMU sample times, s, strictly increasing (uniform grid per shot) */
  imuT: Float64Array;
  imuQ: Float64Array /*N*4 w,x,y,z camera->world, sign-continuous*/;
  imuRate: number;
  hasHighrate: boolean;
  eisBaked: boolean;
  /** same layout as imuQ, world = gravity aligned (z down), when known */
  gravityQ?: Float64Array;
  warnings: string[];
  /** extension: continuous shots [firstFrame, lastFrameInclusive] (joined files have several) */
  segments?: Array<[number, number]>;
  /** extension: diagnostics (timing model, grid fit, lens source, ...) — informational only */
  extra?: Record<string, unknown>;
}

export interface Plan {
  srcW: number;
  srcH: number;
  outW: number;
  outH: number;
  lens: Lens;
  framePts: Float64Array;
  outFx: Float32Array;
  nRows: number;
  rowMats: Float32Array /*F*nRows*9 row-major, M·r_virtual -> r_source; row j at y_j = j*(srcH-1)/(nRows-1)*/;
  readoutS: number;
}

// ================================================================================================ export options
// Output size / aspect / frame-rate contract (owner: PLAN agent; additive, nothing above depends on it).
// Functions: outputGeometry + buildPlan(params.output) in plan.ts, buildSchedule in retime.ts.

/** Output aspect ratio: 'source' keeps the source's ratio (even-rounded). */
export type AspectChoice = 'source' | '16:9' | '4:3' | '1:1' | '9:16';

/** Output size preset: 'NNNNp' sets the SHORT side (height of landscape/square outputs, width of 9:16), '2.7k' the
 *  LONG side (2704 -> 2704x1520 at 16:9), 'source' the largest rectangle of the aspect inside the source,
 *  {width} an explicit output width (aspect kept). */
export type SizeChoice = 'source' | '2.7k' | '1440p' | '1080p' | '720p' | { width: number };

/** Output frame size (both even, >= 2). */
export interface OutputGeometry {
  outW: number;
  outH: number;
  aspect: AspectChoice;
  // ---- extensions (set by outputGeometry; optional so callers may build the object by hand)
  /** outW / (width of the largest rectangle of this aspect that fits the source): > 1 = upscale beyond the source's
   *  pixel density (plus whatever the stabilization crop already magnifies) */
  scale?: number;
  /** true when scale > 1 (output larger than the source can resolve at this aspect): warn the user */
  upscale?: boolean;
}

export interface RetimeParams {
  /** output frame rate; NTSC-style values (59.94, 29.97, 23.976, ...) snap to N*1000/1001 */
  fps: number | 'source';
  /** 'realtime' = resample in time (duration kept, audio kept); 'slowmo' = every source frame at the new fps
   *  (duration n/fps, audio dropped) */
  timing: 'realtime' | 'slowmo';
  /** realtime only: 'natural' = synthetic shutter, blend of the stabilized source frames inside the exposure window */
  motionBlur: 'off' | 'natural';
  /** synthetic shutter angle (deg) for motionBlur 'natural', default 180 */
  shutterDeg?: number;
}

/**
 * Output frame schedule: output frame i = Σ weights[j]·warp(source frame taps[j]) for j in
 * [tapStart[i], tapStart[i] + tapsPerFrame[i]); taps ascending within a frame, weights sum to 1.
 */
export interface OutputSchedule {
  n: number;
  /** output presentation time of every frame, integer µs */
  outPtsUs: Float64Array;
  /** nominal output frame duration, µs (1e6 / fps, not rounded) */
  frameDurUs: number;
  tapsPerFrame: Uint8Array;
  tapStart: Uint32Array;
  /** source presentation-order frame index (= plan record index) */
  taps: Int32Array;
  weights: Float32Array;
  /** max tap of each output frame (monotone non-decreasing): decode/warp up to this before emitting frame i */
  lastSourceFrame: Int32Array;
  // ---- extensions (set by buildSchedule)
  /** output frame rate actually used (after NTSC snapping / 'source') */
  fps?: number;
  /** source frame rate estimated from the PTS */
  srcFps?: number;
  /** playback speed relative to real time: 1 for realtime, fps/srcFps for slowmo */
  speed?: number;
  /** true when the audio track cannot be kept (slowmo) */
  dropAudio?: boolean;
  /** first tap of each output frame (monotone non-decreasing): warped frames below firstSourceFrame[i] are no
   *  longer needed once output frame i is being built */
  firstSourceFrame?: Int32Array;
  /** largest tapsPerFrame and largest (lastSourceFrame - firstSourceFrame + 1) over all frames (buffer sizing) */
  maxTaps?: number;
  maxSpan?: number;
  /** most output frames any one source frame contributes to (= accumulators an add-as-you-go blender needs) */
  maxUses?: number;
  warnings?: string[];
}
