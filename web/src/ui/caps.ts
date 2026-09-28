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
  encodeHevc: boolean;
  encodeH264: boolean;
  encodeAny: boolean;
  saveToDisk: boolean;
  opfs: boolean;
  browser: 'chrome' | 'edge' | 'safari' | 'firefox' | 'other';
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

async function dec(codec: string, w = 3840, h = 2160): Promise<boolean> {
  if (typeof VideoDecoder === 'undefined') return false;
  for (const hw of ['prefer-hardware', 'no-preference'] as HardwareAcceleration[]) {
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
  const [decodeHevc10, decodeHevc, decodeH264, encodeHevc, encodeH264, encodeAv1] = await Promise.all([
    dec('hvc1.2.4.L153.B0'), dec('hvc1.1.6.L153.B0'), dec('avc1.640033'),
    enc('hvc1.1.6.L153.B0', { hevc: { format: 'hevc' } }), enc('avc1.640034', { avc: { format: 'avc' } }), enc('av01.0.13M.08', { hardwareAcceleration: 'no-preference' }),
  ]);
  const ua = navigator.userAgent;
  return {
    webgpu, webcodecs, offscreen, decodeHevc10, decodeHevc, decodeH264, encodeHevc, encodeH264,
    encodeAny: encodeHevc || encodeH264 || encodeAv1,
    saveToDisk: typeof (window as any).showSaveFilePicker === 'function',
    opfs: !!navigator.storage?.getDirectory,
    browser: detectBrowser(),
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
  if (!c.decodeHevc10) notes.push('HEVC clips (Osmo Action 4, O4 Pro) won’t open here; H.264 clips (O3) will.');
  if (!c.encodeHevc && !c.encodeH264) notes.push('No hardware video encoder was found, so exports will be slow.');
  if (c.mobile) notes.push('Phones and tablets can run it, but 4K exports are much faster on a laptop or desktop.');
  if (notes.length) return { ok: true, partial: true, title: 'Heads up', body: notes.join(' ') };
  return { ok: true, partial: false, title: '', body: '' };
}
