// Robustness harness: the prelude injected at the top of the engine worker script (via request interception, so the
// app under test is unmodified). It
//   1. instruments WebCodecs VideoDecoder: every instance, its config, how many decoders are open at once, how many
//      decoded VideoFrames are alive at once (overall / hardware-path / per decoder), and every error (real or
//      injected) -> self.__spDiag, plus a console.warn('[spdiag] …') line per error;
//   2. optionally injects faults that model decoders on other machines (see FAULTS). A "hardware-path" decoder is one
//      configured with hardwareAcceleration !== 'prefer-software' — in Chrome 'no-preference' also picks the GPU
//      decoder whenever one claims support, so only 'prefer-software' is immune to a GPU decoder that fails mid-stream.
//
// Injected errors mimic Chrome's own: the error callback gets DOMException(message, name), the decoder is closed, a
// pending flush() rejects with the same exception, and later decode() calls throw InvalidStateError natively.

export const FAULTS = {
  'none': 'no fault (instrumentation only)',
  'hwfail-first': 'the first hardware-path decoder instance fails on its first decode',
  'hwfail-all': 'every hardware-path decoder fails on its first decode (GPU decoder rejects the stream)',
  'hwfail-late:N': 'every hardware-path decoder fails at its N-th decode (default 45): runtime failure mid-stream',
  'hwlevel': 'typical Windows iGPU (D3D11): hardware-path decoders fail on the first decode when H.264 level > 5.1 or the coded size exceeds 4096x2304',
  'hwinit': 'like hwlevel but at configure(): prefer-hardware -> NotSupportedError; no-preference silently falls back to software (Chrome DecoderSelector)',
  'pool:N': 'per-decoder hardware frame pool of N: a hardware-path decoder errors when it must output while N of its frames are still open',
  'gpool:N': 'shared hardware frame pool of N across all hardware-path decoders',
};

/** Parse "hwlevel,pool:6" -> { hwlevel: true, pool: 6 } */
export function parseFault(spec = 'none') {
  const f = {};
  for (const part of String(spec).split(/[,+]/).map(s => s.trim()).filter(Boolean)) {
    const [k, v] = part.split(':');
    if (k === 'none') continue;
    if (!Object.keys(FAULTS).some(x => x.split(':')[0] === k)) throw new Error(`unknown fault "${k}" (known: ${Object.keys(FAULTS).join(', ')})`);
    f[k] = v !== undefined ? +v : (k === 'hwfail-late' ? 45 : k === 'pool' || k === 'gpool' ? 6 : true);
  }
  return f;
}

/**
 * JS source prepended to the engine worker. `msg` / `name`: the exception injected decode faults raise. Default: what
 * Chrome 153 reports when a GPU decoder fails on data (measured with --probe: VideoToolbox gives exactly
 * "EncodingError: Decoder error."; the FFmpeg software decoder adds a reason in parentheses; a decoder that cannot be
 * created at all gives "OperationError: Unsupported configuration. Check isConfigSupported() prior to calling configure().").
 */
export function prelude(faultSpec = 'none', { msg = 'Decoder error.', name = 'EncodingError' } = {}) {
  const FAULT = parseFault(faultSpec);
  return `/* stillpoint robustness prelude: fault=${JSON.stringify(faultSpec)} */
;(() => {
  const Native = self.VideoDecoder;
  if (!Native || Native.__spWrapped) return;
  const FAULT = ${JSON.stringify(FAULT)};
  const MSG = ${JSON.stringify(msg)}, NAME = ${JSON.stringify(name)};
  const now = () => +(performance.now() / 1000).toFixed(3);
  const D = self.__spDiag = { fault: ${JSON.stringify(faultSpec)}, created: 0, open: 0, maxOpen: 0, hwOpen: 0, maxHwOpen: 0,
    aliveFrames: 0, maxAliveFrames: 0, aliveHw: 0, maxAliveHw: 0, maxAlivePerDecoder: 0, decodes: 0, outputs: 0,
    errors: [], injected: [], configs: [] };
  const hwPath = c => (c && c.hardwareAcceleration || 'no-preference') !== 'prefer-software';
  const avcLevel = codec => { const m = /^avc[13]\\.([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(codec || ''); return m ? parseInt(m[3], 16) : 0; };
  const overLimits = c => avcLevel(c.codec) > 51 || (c.codedWidth || 0) > 4096 || (c.codedHeight || 0) > 2304 || ((c.codedWidth || 0) * (c.codedHeight || 0)) > 4096 * 2304;
  let firstHwTaken = false;
  const markClosed = s => { if (s.closed) return; s.closed = true; D.open--; if (s.hw) D.hwOpen--; };
  function record(dec, e, injected) {
    const s = dec && dec.__sp;
    const r = { t: now(), id: s ? s.id : 0, name: e && e.name, message: e && e.message, injected, codec: s && s.cfg ? s.cfg.codec : '', hwAccel: s && s.cfg ? s.cfg.hardwareAcceleration : '', decodes: s ? s.decodes : 0, outputs: s ? s.outputs : 0, alive: s ? s.alive : 0, open: D.open };
    D.errors.push(r);
    console.warn('[spdiag] decoder #' + r.id + (injected ? ' INJECTED ' : ' ') + 'error ' + r.name + ': ' + r.message + ' (' + r.codec + ' ' + r.hwAccel + ', ' + r.decodes + ' decodes, ' + r.outputs + ' outputs, ' + r.alive + ' own frames open, ' + D.aliveFrames + ' frames open in total, ' + D.open + ' decoders open)');
  }
  function inject(dec, why, name = NAME, msg = MSG) {
    const s = dec.__sp;
    if (s.failed) return;
    s.failed = true;
    const e = new DOMException(msg, name);
    s.failErr = e;
    D.injected.push({ t: now(), id: s.id, why });
    setTimeout(() => {
      try { Native.prototype.close.call(dec); } catch (_) {}
      markClosed(s);
      record(dec, e, why);
      try { s.userError(e); } catch (err) { console.error(err); }
    }, 2);
  }
  function track(f, s) {
    const close = f.close.bind(f);
    let open = true;
    D.aliveFrames++; s.alive++; if (s.hw) D.aliveHw++;
    D.maxAliveFrames = Math.max(D.maxAliveFrames, D.aliveFrames);
    D.maxAliveHw = Math.max(D.maxAliveHw, D.aliveHw);
    D.maxAlivePerDecoder = Math.max(D.maxAlivePerDecoder, s.alive);
    f.close = () => { if (open) { open = false; D.aliveFrames--; s.alive--; if (s.hw) D.aliveHw--; } close(); };
  }
  class VideoDecoder extends Native {
    constructor(init) {
      let me = null;
      super({
        output: f => {
          const s = me && me.__sp;
          if (!s || s.failed) { f.close(); return; }
          if (s.hw && FAULT.pool && s.alive >= FAULT.pool) { f.close(); inject(me, 'pool:' + FAULT.pool + ' (' + s.alive + ' of this decoder\\'s frames still open)'); return; }
          if (s.hw && FAULT.gpool && D.aliveHw >= FAULT.gpool) { f.close(); inject(me, 'gpool:' + FAULT.gpool + ' (' + D.aliveHw + ' hardware frames still open)'); return; }
          track(f, s);
          s.outputs++; D.outputs++;
          init.output(f);
        },
        error: e => { const s = me && me.__sp; if (s) { if (s.failed) return; s.failed = true; markClosed(s); } record(me, e, false); init.error(e); },
      });
      me = this;
      this.__sp = { id: ++D.created, t: now(), hw: false, cfg: null, decodes: 0, outputs: 0, alive: 0, failed: false, failAt: Infinity, closed: false, userError: init.error };
      D.open++; D.maxOpen = Math.max(D.maxOpen, D.open);
    }
    configure(cfg) {
      const s = this.__sp;
      let c = cfg;
      const wasHw = s.hw;
      s.hw = hwPath(cfg);
      if (s.hw && !wasHw) { D.hwOpen++; D.maxHwOpen = Math.max(D.maxHwOpen, D.hwOpen); }
      if (!s.hw && wasHw) D.hwOpen--;
      s.cfg = { codec: cfg.codec, hardwareAcceleration: cfg.hardwareAcceleration, codedWidth: cfg.codedWidth, codedHeight: cfg.codedHeight, optimizeForLatency: cfg.optimizeForLatency };
      D.configs.push({ t: now(), id: s.id, ...s.cfg });
      if (s.hw) {
        if (FAULT['hwfail-all'] || (FAULT['hwfail-first'] && !firstHwTaken)) s.failAt = 1;
        if (FAULT['hwfail-late']) s.failAt = FAULT['hwfail-late'];
        if (FAULT.hwlevel && overLimits(cfg)) s.failAt = 1;
        firstHwTaken = true;
        if (FAULT.hwinit && overLimits(cfg)) {
          if (cfg.hardwareAcceleration === 'prefer-hardware') {
            super.configure({ ...cfg, hardwareAcceleration: 'prefer-software' });
            inject(this, 'hwinit: prefer-hardware over limits', 'OperationError', 'Unsupported configuration. Check isConfigSupported() prior to calling configure().');
            return;
          }
          // no-preference: Chrome falls back to its software decoder when the GPU decoder refuses to initialize
          c = { ...cfg, hardwareAcceleration: 'prefer-software' };
          s.hw = false; D.hwOpen--;
          s.cfg.fellBackToSoftware = true;
        }
      }
      return super.configure(c);
    }
    decode(chunk) {
      const s = this.__sp;
      s.decodes++; D.decodes++;
      const r = super.decode(chunk);
      if (s.hw && !s.failed && s.decodes >= s.failAt) inject(this, 'decode #' + s.decodes + ' on a hardware-path decoder');
      return r;
    }
    flush() {
      const s = this.__sp;
      return super.flush().catch(err => { throw s.failErr || err; });
    }
    close() { markClosed(this.__sp); return super.close(); }
  }
  VideoDecoder.__spWrapped = true;
  Object.defineProperty(self, 'VideoDecoder', { value: VideoDecoder, writable: true, configurable: true, enumerable: false });
})();
`;
}
