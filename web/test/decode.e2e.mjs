// DECODER end-to-end scenarios: headless Chrome (never a visible window) drives the real app (vite dev server on
// 127.0.0.1) through open -> pre-flight -> seek -> play -> export on real clips, with default flags, with hardware
// decoding disabled (--disable-accelerated-video-decode) and with simulated decoder failures (?sp_fault=…):
//
//   o3-default        O3 H.264 4K60 L5.2: hardware decoder, seek / play / export
//   oa4-default       Osmo Action 4 HEVC 10-bit 3840x2880: hardware decoder, seek / play / export
//   o3-nohw           --disable-accelerated-video-decode: H.264 falls back to software; play + export still work
//   oa4-nohw          --disable-accelerated-video-decode: HEVC gets the friendly "can't play" message up front
//   oa4-hw-first      the first HEVC hardware config fails in pre-flight -> the next hardware config (no colour hints)
//   oa4-hw            every HEVC hardware config fails (no software HEVC in Chrome) -> friendly message, Copy details
//   o3-hw-first       the first hardware decoder instance fails in pre-flight -> next config
//   o3-hw             every hardware decoder fails -> pre-flight picks software; note shown; export works
//   o3-hw-at-40-exp   hardware fails mid-export at frame 40 -> software from the last keyframe; all frames exported
//   o3-hw-at-40-play  hardware fails mid-playback -> playback continues in software
//   o3-hang           hardware decoder hangs at frame 30 -> stall watchdog -> software
//   o3-all-at-40      every decoder fails at frame 40 -> precise message + Copy details (export and playback); UI unstuck
//   iphone-cra        HEVC with CRA + RASL leading pictures: seeking RASL frames works (span starts earlier)
//   trunc-faststart   a truncated copy with the index up front: opens with an "incomplete" note, plays what's there
//   trunc-nomoov      a truncated DJI copy (index at the end missing): friendly "looks incomplete" message
//
//   node test/decode.e2e.mjs [--only a,b] [--skip a,b] [--keep]
// Outputs: ~/Library/Application Support/Stillpoint/scratch/decoder/ (truncated copies + downloads are deleted after).
import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, rmSync, statSync, openSync, readSync, writeSync, closeSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { launch } from './chrome.mjs';
import { openMp4, mainVideoTrack } from '../src/mp4.ts';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? (args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : true) : d; };
const ONLY = opt('only', '') ? String(opt('only')).split(',') : null;
const SKIP = opt('skip', '') ? String(opt('skip')).split(',') : [];
const KEEP = !!opt('keep', false);
const O3 = join(process.env.STILLPOINT_O3_DIR ?? join(homedir(), 'Desktop/untitled folder 4'), 'DJI_0026.MP4');
const OA4 = process.env.STILLPOINT_OA4 ?? join(homedir(), 'Desktop/DJI_20260927091931_0012_D.MP4');
const IPHONE = join(homedir(), 'Downloads/ScreenRecording_09-12-2026 13-52-09_1.MP4');
const OUT = join(homedir(), 'Library/Application Support/Stillpoint/scratch/decoder');
const DL = join(OUT, 'downloads');
mkdirSync(DL, { recursive: true });

const vt = () => { try { return +execFileSync('sh', ['-c', 'pgrep -x VTDecoderXPCService | wc -l']).toString().trim(); } catch { return -1; } };
const T0 = performance.now();
const log = (...a) => console.log(`[dec ${((performance.now() - T0) / 1000).toFixed(1)}s]`, ...a);
const sleep = ms => new Promise(r => setTimeout(r, ms));

/** a Blob-like view of a file (Node's openAsBlob truncates sizes > 4 GiB) */
function diskBlob(path) {
  const size = statSync(path).size;
  const mk = (a, b) => ({
    size: Math.max(0, b - a),
    async arrayBuffer() {
      const n = Math.max(0, Math.min(b, size) - a);
      const buf = Buffer.alloc(n);
      const fd = openSync(path, 'r');
      try { let got = 0; while (got < n) { const r = readSync(fd, buf, got, n - got, a + got); if (!r) break; got += r; } } finally { closeSync(fd); }
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + n);
    },
    slice(x = 0, y = b - a) { return mk(a + x, Math.min(b, a + y)); },
  });
  return mk(0, size);
}

function copyHead(src, dst, bytes) {
  const fi = openSync(src, 'r'), fo = openSync(dst, 'w');
  try {
    const buf = Buffer.alloc(4 << 20);
    let done = 0;
    while (done < bytes) { const r = readSync(fi, buf, 0, Math.min(buf.length, bytes - done), done); if (!r) break; writeSync(fo, buf, 0, r); done += r; }
  } finally { closeSync(fi); closeSync(fo); }
}

// ───────── scenario helpers ─────────

class Ctx {
  constructor(page, cdp, name) { this.page = page; this.cdp = cdp; this.name = name; this.evShown = 0; this.downloads = new Map(); }
  async state() { return this.page.evaluate(() => { const d = window.__stillpoint; return { error: d.error, view: d.view, decoder: d.decoder, clip: d.clip, planReady: d.planReady, exportState: d.exportState, lastFrame: d.lastFrame, playing: d.playing, playStats: d.playStats, result: d.result, details: d.errorDetails, events: d.events.slice() }; }); }
  async waitFor(label, src, timeout, { allowError = false } = {}) {
    const t0 = performance.now();
    let lastPrint = 0;
    for (;;) {
      const s = await this.page.evaluate(`(() => { const d = window.__stillpoint; return { ok: !!(${src}), err: d.error, ev: d.events.slice() }; })()`);
      for (const e of s.ev.slice(this.evShown)) log(`  [${this.name}]`, e.slice(0, 400));
      this.evShown = s.ev.length;
      if (s.ok) return;
      if (s.err && !allowError) throw new Error(`${label}: page error: ${s.err}`);
      if (performance.now() - t0 > timeout) throw new Error(`${label}: timeout after ${timeout} ms`);
      if (performance.now() - lastPrint > 15000) { lastPrint = performance.now(); log(`  [${this.name}] waiting for ${label}…`); }
      await sleep(250);
    }
  }
  async open(clip) {
    const input = await this.page.$('#file-input');
    await input.uploadFile(clip);
    await this.waitFor('opened or error', 'd.clip?.frames > 0 || d.error', 180000, { allowError: true });
    return this.state();
  }
  async seek(p) {
    await this.page.evaluate(() => { window.__stillpoint.lastFrame = undefined; });
    const t = performance.now();
    await this.page.evaluate(p => window.__sp_seek(p), p);
    await this.waitFor(`seek ${p}`, `d.lastFrame?.pres === ${p}`, 30000);
    return Math.round(performance.now() - t);
  }
  async play(ms, { allowError = false } = {}) {
    await this.page.evaluate(() => { window.__stillpoint.playStats = undefined; });
    await this.page.click('#btn-play');
    const t0 = performance.now();
    while (performance.now() - t0 < ms) {
      await sleep(250);
      const s = await this.state();
      if (s.error && !allowError) throw new Error('playback: ' + s.error);
      if (!s.playing && performance.now() - t0 > 500) break; // ended (or failed)
    }
    if ((await this.state()).playing) await this.page.click('#btn-play');
    await this.waitFor('play stats', 'd.playStats && !d.playing', 20000, { allowError });
    return (await this.state()).playStats;
  }
  async waitExportable() {
    await this.waitFor('plan', 'd.planReady', 900000);
    await this.waitFor('export enabled', `!document.getElementById('btn-export').disabled`, 120000);
  }
  async exportRange(a, b, { expectError = false } = {}) {
    await this.page.evaluate((a, b) => window.__sp_range(a, b), a, b);
    await this.waitExportable();
    for (const f of readdirSync(DL)) rmSync(join(DL, f), { force: true });
    this.downloads.clear();
    await this.page.click('#btn-export');
    const t0 = performance.now();
    for (;;) {
      const s = await this.page.evaluate(() => ({ st: window.__stillpoint.exportState, r: window.__stillpoint.result, e: window.__stillpoint.error, p: window.__stillpoint.progress }));
      if (s.st === 'done') break;
      if (s.st === 'error') { if (expectError) return { error: s.e }; throw new Error('export failed: ' + s.e); }
      if (performance.now() - t0 > 600000) throw new Error('export timeout');
      await sleep(400);
    }
    const r = (await this.state()).result;
    // the download (OPFS -> Downloads) and an independent frame count
    const t1 = performance.now();
    let file = null;
    while (performance.now() - t1 < 60000) {
      const done = [...this.downloads.values()].find(d => d.state === 'completed');
      if (done && existsSync(join(DL, done.name))) { file = join(DL, done.name); break; }
      await sleep(250);
    }
    if (!file) throw new Error('download did not complete');
    const probe = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width,height,nb_read_frames', '-of', 'json', file]).toString()).streams[0];
    rmSync(file, { force: true });
    return { frames: r.frames, fps: +r.fps.toFixed(1), seconds: +r.seconds.toFixed(1), probe: `${probe.codec_name} ${probe.width}x${probe.height} ${probe.nb_read_frames} frames`, decodedFrames: +probe.nb_read_frames };
  }
  async errorCard() {
    return this.page.evaluate(() => ({
      shown: !document.getElementById('card-error').hidden,
      text: document.getElementById('error-text').textContent,
      copyShown: !!document.getElementById('btn-error-copy') && !document.getElementById('btn-error-copy').hidden,
      exportDisabled: document.getElementById('btn-export').disabled,
      playDisabled: document.getElementById('btn-play').disabled,
      warnings: [...document.querySelectorAll('#warnings li')].map(li => `${li.className}: ${li.textContent}`),
      decodingFact: [...document.querySelectorAll('#facts dt')].find(dt => dt.textContent === 'Decoding')?.nextElementSibling?.textContent ?? null,
    }));
  }
  async copyDetails() {
    await this.page.click('#btn-error-copy');
    await this.waitFor('diagnostics', 'd.lastDiagnostics', 10000, { allowError: true });
    const diag = await this.page.evaluate(() => window.__stillpoint.lastDiagnostics);
    let clip = null;
    try { clip = await this.page.evaluate(() => navigator.clipboard.readText()); } catch (e) { clip = 'clipboard read failed: ' + e.message; }
    return { diag: JSON.parse(diag), clipboardMatches: clip === diag, label: await this.page.$eval('#btn-error-copy', b => b.textContent) };
  }
}

const expect = (cond, msg) => { if (!cond) throw new Error('expectation failed: ' + msg); };

// ───────── scenarios ─────────

const scenarios = [
  {
    name: 'o3-default', clip: O3,
    async run(c) {
      const s = await c.open(O3);
      expect(!s.error, 'no error: ' + s.error);
      expect(s.decoder.variant === 'hw' && !s.decoder.software, 'hardware decoder: ' + JSON.stringify(s.decoder?.variant));
      const seeks = [await c.seek(100), await c.seek(29), await c.seek(267)];
      const play = await c.play(2500);
      expect(play.drawn > 60, 'played frames: ' + JSON.stringify(play));
      const exp = await c.exportRange(0, 119);
      expect(exp.decodedFrames === 120, 'exported frames: ' + JSON.stringify(exp));
      return { decoder: s.decoder.variant, rap: s.decoder.rap, preflight: s.decoder.preflight, seeks, play, exp };
    },
  },
  {
    name: 'oa4-default', clip: OA4,
    async run(c) {
      const s = await c.open(OA4);
      expect(!s.error, 'no error: ' + s.error);
      expect(!s.decoder.software, 'hardware HEVC');
      const seeks = [await c.seek(1000), await c.seek(59), await c.seek(20000)];
      const play = await c.play(2500);
      expect(play.drawn > 30, 'played frames: ' + JSON.stringify(play));
      const exp = await c.exportRange(0, 119);
      expect(exp.decodedFrames === 120, 'exported frames: ' + JSON.stringify(exp));
      return { decoder: s.decoder.variant, facts: s.decoder.facts, rap: s.decoder.rap, preflight: s.decoder.preflight, seeks, play, exp };
    },
  },
  {
    name: 'o3-nohw', clip: O3, flags: ['--disable-accelerated-video-decode'],
    async run(c) {
      const s = await c.open(O3);
      expect(!s.error, 'no error: ' + s.error);
      expect(s.decoder.software && !s.decoder.hwSupported, 'software decoder: ' + JSON.stringify(s.decoder));
      const card = await c.errorCard();
      expect(card.warnings.some(w => /is-info.*software/.test(w)), 'software note shown: ' + JSON.stringify(card.warnings));
      expect(card.decodingFact === 'Software · slower', 'facts row: ' + card.decodingFact);
      const seeks = [await c.seek(100), await c.seek(29)];
      const play = await c.play(2500);
      expect(play.drawn > 10, 'played frames: ' + JSON.stringify(play));
      const exp = await c.exportRange(0, 59);
      expect(exp.decodedFrames === 60, 'exported frames: ' + JSON.stringify(exp));
      return { decoder: s.decoder.variant, note: s.decoder.note, seeks, play, exp };
    },
  },
  {
    name: 'oa4-nohw', clip: OA4, flags: ['--disable-accelerated-video-decode'],
    async run(c) {
      const s = await c.open(OA4);
      expect(/can’t play 3840×2880 HEVC 10-bit/.test(s.error ?? ''), 'friendly HEVC message: ' + s.error);
      const card = await c.errorCard();
      expect(card.shown && card.copyShown, 'error card with Copy details');
      expect(card.exportDisabled && card.playDisabled, 'export + play disabled');
      const cp = await c.copyDetails();
      expect(cp.diag.error.details.kind === 'unsupported' && cp.diag.clip.codec.startsWith('hvc1'), 'diagnostics content');
      // telemetry still arrives, but no plan is requested and export stays off
      await sleep(3000);
      const card2 = await c.errorCard();
      expect(card2.exportDisabled, 'export still disabled');
      return { message: s.error, copied: cp.label, clipboardMatches: cp.clipboardMatches, diagKeys: Object.keys(cp.diag), stream: cp.diag.error.details.stream?.description };
    },
  },
  {
    name: 'oa4-hw-first', clip: OA4, query: 'sp_fault=hw-first',
    async run(c) {
      const s = await c.open(OA4);
      expect(!s.error, 'no error: ' + s.error);
      const tried = s.decoder.preflight.tried.map(t => `${t.variant}:${t.ok ? 'ok' : 'fail'}`);
      expect(tried[0] === 'hw:fail' && tried[1] === 'hw-nocolor:ok' && !s.decoder.software, 'HEVC hardware config ladder: ' + tried);
      const seeks = [await c.seek(300)];
      return { tried, decoder: s.decoder.variant, seeks };
    },
  },
  {
    name: 'oa4-hw', clip: OA4, query: 'sp_fault=hw',
    async run(c) {
      const s = await c.open(OA4);
      expect(/can’t play 3840×2880 HEVC 10-bit.*it failed on this clip/.test(s.error ?? ''), 'friendly message: ' + s.error);
      const card = await c.errorCard();
      expect(card.shown && card.copyShown && card.exportDisabled && card.playDisabled, 'blocked with details: ' + JSON.stringify(card));
      const d = (await c.state()).details;
      return { message: s.error, tried: d?.preflight?.tried?.map(t => `${t.variant}:${t.ok ? 'ok' : 'fail'}`) };
    },
  },
  {
    name: 'o3-hw-first', clip: O3, query: 'sp_fault=hw-first',
    async run(c) {
      const s = await c.open(O3);
      expect(!s.error, 'no error: ' + s.error);
      const tried = s.decoder.preflight.tried.map(t => `${t.variant}:${t.ok ? 'ok' : 'fail'}`);
      expect(tried[0] === 'hw:fail' && tried[1] === 'auto:ok', 'pre-flight ladder: ' + tried);
      const play = await c.play(2000);
      expect(play.drawn > 30, 'played: ' + JSON.stringify(play));
      return { tried, decoder: s.decoder.variant, fallback: s.decoder.fallback, play };
    },
  },
  {
    name: 'o3-hw', clip: O3, query: 'sp_fault=hw',
    async run(c) {
      const s = await c.open(O3);
      expect(!s.error, 'no error: ' + s.error);
      const tried = s.decoder.preflight.tried.map(t => `${t.variant}:${t.ok ? 'ok' : 'fail'}`);
      expect(s.decoder.variant === 'sw' && s.decoder.software && s.decoder.fallback, 'software after pre-flight: ' + tried);
      const card = await c.errorCard();
      expect(card.warnings.some(w => /is-info.*couldn’t handle 3840×2160 H\.264/.test(w)), 'note: ' + JSON.stringify(card.warnings));
      const exp = await c.exportRange(0, 59);
      expect(exp.decodedFrames === 60, 'exported: ' + JSON.stringify(exp));
      return { tried, note: s.decoder.note, exp };
    },
  },
  {
    name: 'o3-hw-at-40-exp', clip: O3, query: 'sp_fault=hw-at:40',
    async run(c) {
      const s = await c.open(O3);
      expect(!s.error && s.decoder.variant === 'hw', 'hardware passes pre-flight');
      const exp = await c.exportRange(0, 119);
      expect(exp.decodedFrames === 120, 'every frame exported: ' + JSON.stringify(exp));
      const st = await c.state();
      expect(st.decoder.variant === 'sw', 'switched to software: ' + st.decoder.variant);
      expect(st.events.some(e => /decoder switched hw -> sw/.test(e)), 'switch logged');
      return { exp, decoder: st.decoder.variant };
    },
  },
  {
    name: 'o3-hw-at-40-play', clip: O3, query: 'sp_fault=hw-at:40',
    async run(c) {
      const s = await c.open(O3);
      expect(!s.error, 'no error');
      await c.seek(0);
      const play = await c.play(4000);
      const st = await c.state();
      expect(!st.error, 'no error: ' + st.error);
      expect(st.decoder.variant === 'sw' && play.drawn > 45, 'continued in software: ' + JSON.stringify(play));
      return { play, decoder: st.decoder.variant, lastFrame: st.lastFrame };
    },
  },
  {
    name: 'o3-hang', clip: O3, query: 'sp_fault=hw-hang-at:30&sp_stall=2500',
    async run(c) {
      const s = await c.open(O3);
      expect(!s.error, 'no error');
      await c.seek(0);
      const play = await c.play(7000);
      const st = await c.state();
      expect(!st.error, 'no error: ' + st.error);
      expect(st.decoder.variant === 'sw' && play.drawn > 35, 'recovered after the stall: ' + JSON.stringify(play));
      return { play, decoder: st.decoder.variant, events: st.events.filter(e => /switch|stall/.test(e)) };
    },
  },
  {
    name: 'o3-all-at-40', clip: O3, query: 'sp_fault=all-at:40',
    async run(c) {
      const s = await c.open(O3);
      expect(!s.error, 'no error at open');
      const exp = await c.exportRange(0, 119, { expectError: true });
      expect(/Export stopped: The video decoder failed at 0:00\.67 \(frame 40\)/.test(exp.error ?? ''), 'precise export message: ' + exp.error);
      const card = await c.errorCard();
      expect(card.shown && card.copyShown, 'error card + copy');
      const cp = await c.copyDetails();
      expect(cp.diag.error.details.pres === 40 && cp.diag.error.details.purpose === 'export', 'diagnostics: pres/purpose');
      expect(cp.diag.gpu && cp.diag.browser.userAgent && cp.diag.decoder, 'diagnostics: gpu/browser/decoder');
      await c.page.click('#btn-error-ok');
      await c.waitFor('export button back', `!document.getElementById('btn-export').disabled && !document.getElementById('card-export').hidden`, 30000, { allowError: true });
      // playback into the failing frame: stops with a precise message, UI not stuck
      await c.page.evaluate(() => { window.__stillpoint.error = undefined; });
      await c.seek(0);
      const play = await c.play(4000, { allowError: true });
      const st = await c.state();
      const card2 = await c.errorCard();
      expect(/Playback stopped: .*frame 40/.test(st.error ?? ''), 'precise playback message: ' + st.error);
      expect(!st.playing && !card2.playDisabled && card2.copyShown, 'not stuck: ' + JSON.stringify({ playing: st.playing, card2 }));
      // and the app still scrubs
      await c.page.click('#btn-error-ok');
      await c.page.evaluate(() => { window.__stillpoint.error = undefined; });
      const seekMs = await c.seek(20);
      return { exportError: exp.error, playError: st.error, play, copied: cp.label, clipboardMatches: cp.clipboardMatches, seekAfterMs: seekMs, history: cp.diag.error.details.history?.map(h => `${h.kind}:${h.variant}${h.to ? '->' + h.to : ''}@${h.pres ?? ''}`) };
    },
  },
  {
    name: 'iphone-cra', clip: IPHONE,
    async prepare() {
      // RASL pictures = samples after a sync sample that present before it
      const info = await openMp4(diskBlob(IPHONE));
      const t = mainVideoTrack(info);
      const order = Array.from({ length: t.sampleCount }, (_, i) => i).sort((a, b) => t.cts[a] - t.cts[b] || a - b);
      const presOf = new Int32Array(t.sampleCount); order.forEach((d, p) => (presOf[d] = p));
      const targets = [];
      for (let d = 1; d < t.sampleCount && targets.length < 6; d++) {
        if (!t.sync[d]) continue;
        if (d + 1 < t.sampleCount && t.cts[d + 1] < t.cts[d]) targets.push(presOf[d + 1], presOf[d]);
      }
      return { targets };
    },
    async run(c, prep) {
      const s = await c.open(IPHONE);
      expect(!s.error, 'no error: ' + s.error);
      expect(s.decoder.rap.cra > 0, 'CRA keyframes found: ' + JSON.stringify(s.decoder.rap));
      const seeks = [];
      for (const p of prep.targets) seeks.push([p, await c.seek(p)]);
      return { rap: s.decoder.rap, seeks };
    },
  },
  {
    name: 'trunc-faststart', clip: O3,
    async prepare() {
      const full = join(OUT, 'DJI_0026_faststart.mp4');
      const cut = join(OUT, 'DJI_0026_faststart_trunc.mp4');
      execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', O3, '-map', '0:v:0', '-map', '0:a:0?', '-c', 'copy', '-movflags', '+faststart', full]);
      copyHead(full, cut, 40e6);
      rmSync(full, { force: true });
      return { cut, cleanup: [cut] };
    },
    async run(c, prep) {
      const s = await c.open(prep.cut);
      expect(!s.error || /gyro/i.test(s.error), 'opened: ' + s.error);
      expect(s.clip.frames > 50 && s.clip.frames < 268, 'trimmed frames: ' + s.clip.frames);
      const card = await c.errorCard();
      expect(card.warnings.some(w => /incomplete/.test(w)), 'incomplete note: ' + JSON.stringify(card.warnings));
      const last = s.clip.frames - 1;
      const seeks = [await c.seek(last), await c.seek(10)];
      return { frames: s.clip.frames, note: card.warnings.find(w => /incomplete/.test(w)), seeks };
    },
  },
  {
    name: 'trunc-nomoov', clip: O3,
    async prepare() {
      const cut = join(OUT, 'DJI_0026_trunc.MP4');
      copyHead(O3, cut, 40e6);
      return { cut, cleanup: [cut] };
    },
    async run(c, prep) {
      const s = await c.open(prep.cut);
      expect(/looks incomplete/.test(s.error ?? ''), 'friendly incomplete message: ' + s.error);
      return { message: s.error };
    },
  },
];

// ───────── runner ─────────

const root = fileURLToPath(new URL('..', import.meta.url));
const server = await createServer({ configFile: join(root, 'vite.config.ts'), root, server: { port: 0, host: '127.0.0.1', hmr: false, watch: { ignored: ['**/*'] } }, logLevel: 'error' });
await server.listen();
const base = server.resolvedUrls.local[0];
const origin = new URL(base).origin;
const report = { base, vtBefore: vt(), results: {} };
log('serving', base, 'VT before', report.vtBefore);
let failures = 0;
try {
  for (const sc of scenarios) {
    if (ONLY && !ONLY.includes(sc.name)) continue;
    if (SKIP.includes(sc.name)) continue;
    if (!existsSync(sc.clip)) { report.results[sc.name] = { skipped: 'clip missing' }; log(sc.name, 'SKIPPED (clip missing)'); continue; }
    log('──', sc.name, sc.flags?.join(' ') ?? '', sc.query ?? '');
    const t0 = performance.now();
    let prep = {};
    const browser = await launch(sc.flags ?? []);
    try {
      prep = sc.prepare ? await sc.prepare() : {};
      await browser.defaultBrowserContext().overridePermissions(origin, ['clipboard-read', 'clipboard-write', 'clipboard-sanitized-write']).catch(() => {});
      const page = await browser.newPage();
      page.on('console', m => { const t = m.text(); if (m.type() === 'error' || /stillpoint|webgpu/i.test(t)) log(`  [${sc.name} page ${m.type()}]`, t.slice(0, 300)); });
      page.on('pageerror', e => log(`  [${sc.name} pageerror]`, e.message));
      const cdp = await browser.target().createCDPSession();
      await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: DL, eventsEnabled: true });
      const c = new Ctx(page, cdp, sc.name);
      cdp.on('Browser.downloadWillBegin', e => c.downloads.set(e.guid, { name: e.suggestedFilename, state: 'started' }));
      cdp.on('Browser.downloadProgress', e => { const d = c.downloads.get(e.guid); if (d) d.state = e.state; });
      await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
      const q = new URLSearchParams({ sink: 'opfs' });
      if (sc.query) for (const [k, v] of new URLSearchParams(sc.query)) q.set(k, v);
      await page.goto(`${base}?${q}`, { waitUntil: 'load' });
      await page.waitForFunction(() => window.__stillpoint?.view === 'landing' && window.__stillpoint?.gpu, { timeout: 60000 });
      const out = await sc.run(c, prep);
      report.results[sc.name] = { ok: true, s: +((performance.now() - t0) / 1000).toFixed(1), ...out };
      log(sc.name, 'PASS', JSON.stringify(out).slice(0, 1200));
    } catch (e) {
      failures++;
      report.results[sc.name] = { ok: false, error: String(e?.message ?? e) };
      log(sc.name, 'FAIL', e?.message ?? e);
    } finally {
      await browser.close().catch(() => {});
      if (!KEEP) for (const f of prep.cleanup ?? []) rmSync(f, { force: true });
    }
  }
} finally {
  await server.close();
  for (const f of readdirSync(DL)) rmSync(join(DL, f), { force: true });
  report.vtAfter = vt();
  console.log(JSON.stringify(report, null, 1));
  log(`${failures ? failures + ' FAILED' : 'all passed'}; VT before ${report.vtBefore} after ${report.vtAfter}`);
  process.exit(failures ? 1 : 0);
}
