/**
 * Encoder selection for Stillpoint exports: HEVC (Main, 8-bit) when the browser can encode it, else H.264 High,
 * else AV1 / VP9 (usually software — slow at 4K). Codec strings carry a level that actually fits the output size,
 * frame rate and bitrate, so players and editors accept the file.
 */

export type OutCodec = 'hevc' | 'avc' | 'av1' | 'vp9';

export interface EncoderChoice {
  codec: OutCodec;
  config: VideoEncoderConfig;
  /** human label, e.g. "HEVC (H.265)" */
  label: string;
  hardware: 'prefer-hardware' | 'no-preference';
}

// HEVC general levels (ITU-T H.265 Table A.8): level_idc*30 -> [MaxLumaPs, MaxLumaSr, MaxBr main tier kbps, high tier kbps]
const HEVC_LEVELS: Array<[number, number, number, number, number]> = [
  [120, 2228224, 66846720, 12000, 30000],
  [123, 2228224, 133693440, 20000, 50000],
  [150, 8912896, 267386880, 25000, 100000],
  [153, 8912896, 534773760, 40000, 160000],
  [156, 8912896, 1069547520, 60000, 240000],
  [180, 35651584, 1069547520, 60000, 240000],
  [183, 35651584, 2139095040, 120000, 480000],
  [186, 35651584, 4278190080, 240000, 800000],
];

export function hevcCodecString(w: number, h: number, fps: number, bitrate: number): string {
  const ps = w * h, sr = ps * fps, kbps = bitrate / 1000;
  for (const [lvl, maxPs, maxSr, mainBr, highBr] of HEVC_LEVELS) {
    if (ps > maxPs || sr > maxSr || Math.max(w, h) > Math.sqrt(8 * maxPs)) continue;
    if (kbps * 1.0 <= mainBr) return `hvc1.1.6.L${lvl}.B0`;
    if (kbps <= highBr) return `hvc1.1.6.H${lvl}.B0`;
  }
  return 'hvc1.1.6.H186.B0';
}

// H.264 levels (Table A-1): [level_idc, MaxMBPS, MaxFS, MaxBR(kbps, x1.25 for High)]
const AVC_LEVELS: Array<[number, number, number, number]> = [
  [0x28, 245760, 8192, 20000], [0x2a, 522240, 8704, 50000], [0x32, 589824, 22080, 135000],
  [0x33, 983040, 36864, 240000], [0x34, 2073600, 36864, 240000], [0x3c, 4177920, 139264, 240000],
  [0x3d, 8355840, 139264, 480000], [0x3e, 16711680, 139264, 800000],
];

export function avcCodecString(w: number, h: number, fps: number, bitrate: number): string {
  const mbw = Math.ceil(w / 16), mbh = Math.ceil(h / 16), fs = mbw * mbh, mbps = fs * fps;
  for (const [lvl, maxMbps, maxFs, maxBr] of AVC_LEVELS) {
    if (fs > maxFs || mbps > maxMbps || mbw > Math.sqrt(8 * maxFs) || mbh > Math.sqrt(8 * maxFs)) continue;
    if (bitrate / 1000 > maxBr * 1.25) continue;
    return 'avc1.6400' + lvl.toString(16).padStart(2, '0');
  }
  return 'avc1.64003E';
}

export function av1CodecString(w: number, h: number, fps: number): string {
  const ps = w * h, sr = ps * fps;
  // seq_level_idx: 12=5.0, 13=5.1, 14=5.2, 16=6.0, 17=6.1, 18=6.2 (MaxPicSize 8912896 for 5.x, 35651584 for 6.x)
  const lvl = ps <= 8912896 ? (sr <= 267386880 ? 12 : sr <= 534773760 ? 13 : 14) : (sr <= 1069547520 ? 16 : sr <= 2139095040 ? 17 : 18);
  return `av01.0.${String(lvl).padStart(2, '0')}H.08`;
}

export function vp9CodecString(w: number, h: number, fps: number): string {
  const ps = w * h, sr = ps * fps;
  const lvl = ps <= 8912896 ? (sr <= 311951360 ? 50 : sr <= 588251136 ? 51 : 52) : (sr <= 1176502272 ? 60 : sr <= 2353004544 ? 61 : 62);
  return `vp09.00.${lvl}.08`;
}

export interface EncoderRequest {
  width: number;
  height: number;
  fps: number;
  bitrate: number;
  /** try this codec first */
  prefer?: OutCodec;
  /** skip software-only codecs (AV1/VP9 at 4K are ~1-5 fps) */
  allowSoftware?: boolean;
}

function configFor(codec: OutCodec, r: EncoderRequest, hw: EncoderChoice['hardware']): VideoEncoderConfig {
  const base: VideoEncoderConfig = {
    codec: '',
    width: r.width,
    height: r.height,
    bitrate: Math.round(r.bitrate),
    bitrateMode: 'variable',
    framerate: r.fps,
    latencyMode: 'quality',
    hardwareAcceleration: hw,
    alpha: 'discard',
  };
  switch (codec) {
    case 'hevc': return { ...base, codec: hevcCodecString(r.width, r.height, r.fps, r.bitrate), hevc: { format: 'hevc' } } as VideoEncoderConfig;
    case 'avc': return { ...base, codec: avcCodecString(r.width, r.height, r.fps, r.bitrate), avc: { format: 'avc' } };
    case 'av1': return { ...base, codec: av1CodecString(r.width, r.height, r.fps) };
    case 'vp9': return { ...base, codec: vp9CodecString(r.width, r.height, r.fps) };
  }
}

export const CODEC_LABEL: Record<OutCodec, string> = { hevc: 'HEVC (H.265)', avc: 'H.264', av1: 'AV1', vp9: 'VP9' };

export async function chooseEncoder(r: EncoderRequest): Promise<EncoderChoice | null> {
  if (typeof VideoEncoder === 'undefined') return null;
  const order: OutCodec[] = ['hevc', 'avc', 'av1', 'vp9'];
  if (r.prefer) order.sort((a, b) => (a === r.prefer ? -1 : b === r.prefer ? 1 : 0));
  // hardware first across all codecs, then anything
  for (const hw of ['prefer-hardware', 'no-preference'] as const) {
    for (const codec of order) {
      if (hw === 'no-preference' && (codec === 'av1' || codec === 'vp9') && r.allowSoftware === false) continue;
      const config = configFor(codec, r, hw);
      try {
        const s = await VideoEncoder.isConfigSupported(config);
        if (s.supported) return { codec, config: s.config ?? config, label: CODEC_LABEL[codec], hardware: hw };
      } catch { /* try the next one */ }
    }
  }
  return null;
}

/** Suggested bitrates (bits/s) for the Quality control, scaled from a 3840x2160 reference by pixel rate. */
export function bitratePresets(w: number, h: number, fps: number): Record<'balanced' | 'high' | 'max', number> {
  const scale = Math.max(0.15, (w * h * fps) / (3840 * 2160 * 59.94));
  const r = (mbps: number) => Math.round(mbps * scale) * 1e6;
  return { balanced: r(60), high: r(100), max: r(150) };
}
