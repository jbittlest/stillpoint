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
