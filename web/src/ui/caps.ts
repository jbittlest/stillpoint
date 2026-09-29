/**
 * Browser capability check: WebGPU, WebCodecs (decode HEVC Main10 / H.264, encode HEVC / H.264), workers +
 * OffscreenCanvas, and where exports can be written. Produces a verdict with a friendly explanation.
 */

export interface Caps {
  webgpu: boolean;
  webcodecs: boolean;
  offscreen: boolean;
  decodeHevc10: boolean;
  decodeHevc: boolean;
  decodeH264: boolean;
  /** isConfigSupported() with prefer-hardware (diagnostics: a "yes" here doesn't guarantee the GPU can decode the clip) */
  hwDecodeH264?: boolean;
  hwDecodeHevc10?: boolean;
  encodeHevc: boolean;
  encodeH264: boolean;
  encodeAny: boolean;
  saveToDisk: boolean;
  opfs: boolean;
  browser: 'chrome' | 'edge' | 'safari' | 'firefox' | 'other';
  platform?: 'mac' | 'windows' | 'linux' | 'chromeos' | 'android' | 'ios' | 'other';
  mobile: boolean;
}

export interface Verdict {
  ok: boolean;
  /** usable but with caveats */
  partial: boolean;
  title: string;
  body: string;
}

export function detectBrowser(): Caps['browser'] {
  const ua = navigator.userAgent;
  if (/Firefox\//.test(ua)) return 'firefox';
  if (/Edg\//.test(ua)) return 'edge';
  if (/Chrome\//.test(ua) || /CriOS\//.test(ua)) return 'chrome';
  if (/Safari\//.test(ua)) return 'safari';
  return 'other';
}

export function detectPlatform(ua = navigator.userAgent, platform = navigator.platform ?? ''): NonNullable<Caps['platform']> {
  if (/Android/i.test(ua)) return 'android';
  if (/iPhone|iPad|iPod/.test(ua) || (/Mac/.test(platform) && (navigator as any).maxTouchPoints > 1)) return 'ios';
  if (/CrOS/.test(ua)) return 'chromeos';
  if (/Win/i.test(platform) || /Windows/.test(ua)) return 'windows';
  if (/Mac/i.test(platform) || /Mac OS X/.test(ua)) return 'mac';
  if (/Linux/i.test(platform) || /Linux/.test(ua)) return 'linux';
  return 'other';
}

async function dec(codec: string, w = 3840, h = 2160, modes: HardwareAcceleration[] = ['prefer-hardware', 'no-preference', 'prefer-software']): Promise<boolean> {
  if (typeof VideoDecoder === 'undefined') return false;
  for (const hw of modes) {
    try { if ((await VideoDecoder.isConfigSupported({ codec, codedWidth: w, codedHeight: h, hardwareAcceleration: hw })).supported) return true; } catch { /* next */ }
  }
  return false;
}

async function enc(codec: string, extra: Record<string, unknown>): Promise<boolean> {
  if (typeof VideoEncoder === 'undefined') return false;
  try {
    const r = await VideoEncoder.isConfigSupported({ codec, width: 3840, height: 2160, framerate: 59.94, bitrate: 100e6, hardwareAcceleration: 'prefer-hardware', latencyMode: 'quality', ...extra } as VideoEncoderConfig);
    return !!r.supported;
  } catch { return false; }
}

export async function detectCaps(): Promise<Caps> {
  const webcodecs = typeof VideoDecoder !== 'undefined' && typeof VideoEncoder !== 'undefined' && typeof VideoFrame !== 'undefined';
  let webgpu = false;
  try {
    const gpu = (navigator as any).gpu;
    webgpu = !!gpu && !!(await gpu.requestAdapter());
  } catch { webgpu = false; }
  const offscreen = typeof OffscreenCanvas !== 'undefined' && typeof HTMLCanvasElement.prototype.transferControlToOffscreen === 'function' && typeof Worker !== 'undefined';
  // avc1.640034 = H.264 High level 5.2, what DJI O3 / O4 Pro record at 4K60
  const [decodeHevc10, decodeHevc, decodeH264, hwDecodeH264, hwDecodeHevc10, encodeHevc, encodeH264, encodeAv1] = await Promise.all([
    dec('hvc1.2.4.L153.B0'), dec('hvc1.1.6.L153.B0'), dec('avc1.640034'),
    dec('avc1.640034', 3840, 2160, ['prefer-hardware']), dec('hvc1.2.4.L153.B0', 3840, 2160, ['prefer-hardware']),
    enc('hvc1.1.6.L153.B0', { hevc: { format: 'hevc' } }), enc('avc1.640034', { avc: { format: 'avc' } }), enc('av01.0.13M.08', { hardwareAcceleration: 'no-preference' }),
  ]);
  const ua = navigator.userAgent;
  return {
    webgpu, webcodecs, offscreen, decodeHevc10, decodeHevc, decodeH264, hwDecodeH264, hwDecodeHevc10, encodeHevc, encodeH264,
    encodeAny: encodeHevc || encodeH264 || encodeAv1,
    saveToDisk: typeof (window as any).showSaveFilePicker === 'function',
    opfs: !!navigator.storage?.getDirectory,
    browser: detectBrowser(),
    platform: detectPlatform(),
    mobile: /Android|iPhone|iPad|Mobile/i.test(ua),
  };
}

export function verdict(c: Caps): Verdict {
  const missing: string[] = [];
  if (!c.webgpu) missing.push('WebGPU');
  if (!c.webcodecs) missing.push('WebCodecs video');
  if (!c.offscreen) missing.push('OffscreenCanvas in workers');
  if (c.webcodecs && !c.encodeAny) missing.push('a video encoder');
  if (c.webcodecs && !c.decodeH264 && !c.decodeHevc) missing.push('an H.264 / HEVC decoder');
  const suggest = 'Please open this page in <b>Chrome</b>, <b>Edge</b> or <b>Arc</b> on a Mac or Windows PC (or Safari 26 on a Mac).';
  if (missing.length) {
    let body: string;
    if (c.browser === 'firefox') {
      body = `Firefox can’t decode the HEVC video DJI cameras record, and its WebGPU support is still rolling out. ${suggest}`;
    } else if (c.browser === 'safari' && !c.webgpu) {
      body = `This version of Safari doesn’t have WebGPU. Update to Safari 26 or later, or use Chrome / Edge.`;
    } else {
      body = `Stillpoint needs ${missing.join(', ')}, which this browser doesn’t provide. ${suggest}`;
    }
    if (!c.webgpu && (c.browser === 'chrome' || c.browser === 'edge')) {
      body = `WebGPU is turned off or blocked for your graphics card in this browser. Make sure it’s up to date and hardware acceleration is on (Settings → System). ${suggest}`;
    }
    return { ok: false, partial: false, title: 'This browser can’t run Stillpoint yet', body };
  }
  const notes: string[] = [];
  if (!c.decodeHevc10) notes.push('This computer has no hardware HEVC (H.265) decoding in this browser, so HEVC clips (Osmo Action 4, or O4 Pro set to H.265) won’t open here; H.264 clips (O3, O4 Pro in H.264) will.');
  if (!c.encodeHevc && !c.encodeH264) notes.push('No hardware video encoder was found, so exports will be slow.');
  if (c.mobile) notes.push('Phones and tablets can run it, but 4K exports are much faster on a laptop or desktop.');
  if (notes.length) return { ok: true, partial: true, title: 'Heads up', body: notes.join(' ') };
  return { ok: true, partial: false, title: '', body: '' };
}
