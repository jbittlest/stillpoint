// Robustness matrix: headless Chrome drives the REAL app UI through a scripted user session per (Chrome flags x fault
// x clip) and records every error the UI shows (toasts, error card), every console error (page + workers) and every
// WebCodecs decoder error (via the instrumentation prelude in inject.mjs), step by step.
//
//   node test/robustness/matrix.mjs [--target docs|dev|build|URL] [--clips o3,oa4,...] [--flags default,nohwdec,...]
//        [--faults none,hwlevel,...] [--steps load,play-immediate,...] [--probe] [--strict] [--keep] [--out DIR]
//   npm --prefix test/robustness run matrix            (the default matrix; see package.json here for presets)
//
//   --target  docs (default): ../docs/app served on 127.0.0.1 = byte-identical to the live GitHub Pages app
//             dev: Vite dev server on src/ (test a fix);  build: fresh vite build of src/;  or a URL (e.g. the live site)
//   --flags   comma list of Chrome flag sets, '+' combines:  default | nohwdec (--disable-accelerated-video-decode) |
//             nohevc (--disable-features=PlatformHEVCDecoderSupport) | nogpu (--disable-gpu)
//   --faults  comma list of injected decoder faults (see inject.mjs FAULTS), '+' combines, e.g. hwlevel+pool:6
//   --clips   names from clips.mjs CLIPS (or paths). Damaged variants are generated in scratch and deleted afterwards.
//   --probe   also measure Chrome's own error messages for bad input (garbage / P-frame-as-key / empty / truncated
//             chunks, software HEVC) and isConfigSupported answers, per flag set
//   --strict  exit 1 when an undamaged, supported clip shows any error or misses a step goal (regression mode)
//   --strict-faults  exit 1 when an H.264 clip misses a step goal under an injected hardware-decoder fault
//
// The user session (steps): load (file input) -> play-immediate (Play as soon as the clip opens, while the first
// preview frame is still decoding) -> analysis (gyro + plan) -> play-held (seek to 30 %, then Play with that preview
// frame held) -> seek-during-play (timeline clicks + a scrub drag + keyboard while playing) -> near-end (seek/step to
// the last frames and play into the end) -> seek-leading (seek to the leading/RASL pictures of CRA / open-GOP sync
// samples, when the clip has any: they cannot be decoded from that sync sample) -> play-export (Play, then Stabilize while playing: last ~2 s of the clip,
// memory sink, no download) -> after-export (the post-export re-seek, then play again).
// Never opens a visible window (--headless=new). Results: <scratch>/robustness/matrix-<time>.json
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { launch } from '../chrome.mjs';
import { serve } from './server.mjs';
import { prelude, parseFault } from './inject.mjs';
import { CLIPS, DAMAGED, SCRATCH, removeDamaged, resolveClip } from './clips.mjs';
import { leadingFrames } from './nal-scan.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? (args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : true) : d; };
const TARGET = opt('target', 'docs');
const FLAGSETS = String(opt('flags', 'default')).split(',');
const FAULTLIST = String(opt('faults', 'none')).split(',');
const CLIPLIST = String(opt('clips', 'o3')).split(',');
const ALL_STEPS = ['load', 'play-immediate', 'analysis', 'play-held', 'seek-during-play', 'near-end', 'seek-leading', 'play-export', 'after-export'];
const STEPS = new Set(opt('steps', '') ? String(opt('steps')).split(',').concat(['load']) : ALL_STEPS);
const PROBE = !!opt('probe', false);
const STRICT = !!opt('strict', false);
// --strict-faults: with an injected hardware-decoder fault, an H.264 clip must still reach every step goal (Chrome has a
// software H.264 decoder to fall back to); HEVC has no software decoder in Chrome, so HEVC fault runs are report-only
const STRICT_FAULTS = !!opt('strict-faults', false);
const KEEP = !!opt('keep', false);
const OUT = opt('out', SCRATCH);
const ANALYSIS_TIMEOUT = +opt('analysis-timeout', 900) * 1000;
const EXPORT_TIMEOUT = +opt('export-timeout', 300) * 1000;
const INJECT_MSG = opt('inject-msg', '');
mkdirSync(OUT, { recursive: true });

const FLAGS = {
  default: [],
  nohwdec: ['--disable-accelerated-video-decode'],
  nohevc: ['--disable-features=PlatformHEVCDecoderSupport'],
  nogpu: ['--disable-gpu'],
};
const flagArgs = set => set.split('+').flatMap(f => { if (!FLAGS[f]) throw new Error('unknown flag set ' + f); return FLAGS[f]; });
for (const f of FAULTLIST) parseFault(f); // validate early

const t00 = performance.now();
const log = (...a) => console.log(`[rob ${((performance.now() - t00) / 1000).toFixed(1)}s]`, ...a);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const vt = () => { try { return +execFileSync('sh', ['-c', 'pgrep -x VTDecoderXPCService | wc -l']).toString().trim(); } catch { return -1; } };

// UI observers, installed in the page before the app runs
const PAGE_HOOK = () => {
  window.__spUi = [];
  const t = () => +(performance.now() / 1000).toFixed(3);
  const watch = (id, kind) => {
    const el = document.getElementById(id);
    if (!el) return;
    let last = '';
    const rec = () => { const s = (el.textContent || '').trim(); const vis = !el.hidden && !el.closest('[hidden]'); if (s && vis && s !== last) { last = s; window.__spUi.push({ t: t(), kind, text: s }); } if (!vis) last = ''; };
    new MutationObserver(rec).observe(el, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['hidden'] });
    const card = el.closest('section');
    if (card && card !== el) new MutationObserver(rec).observe(card, { attributes: true, attributeFilter: ['hidden'] });
  };
  document.addEventListener('DOMContentLoaded', () => { watch('toast', 'toast'); watch('error-text', 'error-card'); watch('unsupported-body', 'unsupported'); });
};

const ERR_EVENT = /error|fail|stopped|could not|couldn’t|unsupported|not supported/i;
// the app's own log lines for every error it surfaces (a toast repeating the same text is not a new DOM change, so the
// event log — one line per error — is the reliable per-step record; the DOM observer adds the error card / notices)
const UI_ERR = /^[\d.]+ (engine error|export error|fail|analysis error [a-z]+): (.*)$/;
const uiErrors = s => [...new Set([
  ...s.events.map(e => UI_ERR.exec(e)).filter(Boolean).map(m => (m[1] === 'engine error' ? 'toast' : m[1] === 'fail' ? 'error-card' : m[1]) + ': ' + m[2]),
  ...s.ui.filter(u => /^error-card: /.test(u)),
])];
const decErrors = s => s.decoderErrors.map(e => `decoder ${e.injected ? '(injected) ' : ''}${e.name}: ${e.message}`);

async function runOne(browser, base, flagSet, fault, clip) {
  const res = { flags: flagSet, fault, clip: clip.name, desc: clip.desc, steps: [], vtBefore: vt() };
  const ctx = await browser.createBrowserContext();
  const page = await ctx.newPage();
  const consoleLines = [];
  page.on('console', m => { const ty = m.type(); const tx = m.text(); if (ty === 'error' || ty === 'warn' || ty === 'warning' || /\[spdiag\]|stillpoint/i.test(tx)) consoleLines.push({ t: +((performance.now() - t00) / 1000).toFixed(2), type: ty, text: tx.slice(0, 500) }); });
  page.on('pageerror', e => consoleLines.push({ t: +((performance.now() - t00) / 1000).toFixed(2), type: 'pageerror', text: String(e.message).slice(0, 500) }));
  const pre = prelude(fault, INJECT_MSG ? { msg: INJECT_MSG } : {});
  // Inject the prelude into the engine worker's script response. A narrow CDP Fetch pattern on the page session pauses
  // only that one response (puppeteer's global request interception would also pause the worker's own module imports
  // under the Vite dev server, and the worker would never start).
  const cdp = await page.createCDPSession();
  await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*engine.worker*', requestStage: 'Response' }] });
  cdp.on('Fetch.requestPaused', async e => {
    try {
      const { body, base64Encoded } = await cdp.send('Fetch.getResponseBody', { requestId: e.requestId });
      const text = base64Encoded ? Buffer.from(body, 'base64').toString('utf8') : body;
      const headers = (e.responseHeaders || []).filter(h => !/^(content-length|content-encoding|etag|last-modified)$/i.test(h.name));
      await cdp.send('Fetch.fulfillRequest', { requestId: e.requestId, responseCode: e.responseStatusCode || 200, responseHeaders: headers, body: Buffer.from(pre + text, 'utf8').toString('base64') });
      res.injectedInto = e.request.url.replace(base, '');
    } catch (err) {
      res.injectError = String(err?.message ?? err);
      await cdp.send('Fetch.continueRequest', { requestId: e.requestId }).catch(() => {});
    }
  });
  await page.evaluateOnNewDocument(PAGE_HOOK);
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
  const url = `${base}${base.includes('?') ? '&' : '?'}sink=memory&autodownload=0`;
  await page.goto(url, { waitUntil: 'load' });

  const engine = async () => page.workers().find(w => /engine\.worker/.test(w.url()));
  const diag = async (reset = false) => {
    const w = await engine();
    if (!w) return null;
    return w.evaluate(reset => {
      const D = self.__spDiag; if (!D) return null;
      const snap = JSON.parse(JSON.stringify({ created: D.created, open: D.open, maxOpen: D.maxOpen, maxHwOpen: D.maxHwOpen, aliveFrames: D.aliveFrames, maxAliveFrames: D.maxAliveFrames, maxAliveHw: D.maxAliveHw, maxAlivePerDecoder: D.maxAlivePerDecoder, decodes: D.decodes, outputs: D.outputs, errors: D.errors.length, injected: D.injected.length, configs: D.configs.length }));
      if (reset) { D.maxOpen = D.open; D.maxHwOpen = D.hwOpen; D.maxAliveFrames = D.aliveFrames; D.maxAliveHw = D.aliveHw; D.maxAlivePerDecoder = 0; }
      return snap;
    }, reset).catch(() => null);
  };
  const diagDetail = async () => { const w = await engine(); return w ? w.evaluate(() => self.__spDiag && JSON.parse(JSON.stringify({ errors: self.__spDiag.errors, injected: self.__spDiag.injected, configs: self.__spDiag.configs }))).catch(() => null) : null; };
  const st = () => page.evaluate(() => {
    const d = window.__stillpoint || {};
    return { view: d.view, caps: d.caps, gpu: d.gpu, frames: d.clip?.frames ?? 0, telemetry: !!d.clip?.telemetry, planReady: d.planReady, encoder: d.clip?.encoder?.codec ?? null,
      error: d.error ?? null, events: d.events ?? [], lastFrame: d.lastFrame ?? null, playStats: d.playStats ?? null, exportState: d.exportState, result: d.result ? { frames: d.result.frames, fps: d.result.fps, codec: d.result.codec } : null,
      ui: window.__spUi ?? [], playing: document.getElementById('btn-play')?.classList.contains('is-playing') ?? false,
      badge: document.getElementById('gyro-badge')?.textContent?.trim() ?? '', inputDisabled: document.getElementById('file-input')?.disabled ?? false,
      playDisabled: document.getElementById('btn-play')?.disabled ?? true, exportDisabled: document.getElementById('btn-export')?.disabled ?? true };
  });
  const waitFor = async (pred, timeout, every = 250) => { const t0 = performance.now(); for (;;) { const s = await st(); if (pred(s)) return s; if (performance.now() - t0 > timeout) return null; await sleep(every); } };

  // ── step bookkeeping ──
  let evSeen = 0, uiSeen = 0, conSeen = 0, diagErrSeen = 0;
  const step = async (name, fn) => {
    if (!STEPS.has(name)) return null;
    const r = { step: name, t0: +((performance.now() - t00) / 1000).toFixed(2) };
    const d0 = await diag(true);
    try { Object.assign(r, (await fn()) ?? {}); } catch (e) { r.harnessError = String(e?.message ?? e); }
    await sleep(300);
    const s = await st();
    const dd = await diagDetail();
    r.events = s.events.slice(evSeen).filter(e => ERR_EVENT.test(e)); evSeen = s.events.length;
    r.ui = s.ui.slice(uiSeen).map(u => `${u.kind}: ${u.text}`); uiSeen = s.ui.length;
    r.console = consoleLines.slice(conSeen).filter(c => c.type !== 'log' || /\[spdiag\]/.test(c.text)).map(c => `${c.type}: ${c.text}`); conSeen = consoleLines.length;
    r.decoderErrors = dd ? dd.errors.slice(diagErrSeen) : []; diagErrSeen = dd ? dd.errors.length : 0;
    const d1 = await diag();
    if (d0 && d1) r.decoders = { created: d1.created - d0.created, maxOpen: d1.maxOpen, maxHwOpen: d1.maxHwOpen, maxAliveFrames: d1.maxAliveFrames, maxAlivePerDecoder: d1.maxAlivePerDecoder, decodes: d1.decodes - d0.decodes, outputs: d1.outputs - d0.outputs };
    r.dt = +((performance.now() - t00) / 1000 - r.t0).toFixed(2);
    // a play step only meets its goal when playback was not cut short by an error
    if (r.play && r.goal && uiErrors(r).some(e => /Playback stopped/.test(e))) { r.goal = false; r.note = (r.note ?? '') + ' — playback stopped by an error'; }
    res.steps.push(r);
    const errs = [...new Set([...uiErrors(r), ...decErrors(r)])];
    log(`  ${name.padEnd(16)} ${r.goal === false ? 'GOAL MISSED' : r.goal === true ? 'ok' : r.goal ?? ''} ${r.note ?? ''}${errs.length ? '  ERR ' + errs.join(' | ') : ''}`);
    return r;
  };

  // (portrait clips make the viewer taller than the window: bring the timeline into view before pointer input)
  const tlBox = async () => { await page.$eval('#timeline', el => el.scrollIntoView({ block: 'center', inline: 'nearest' })); return (await page.$('#timeline')).boundingBox(); };
  const timeline = async frac => {
    const box = await tlBox();
    await page.mouse.click(box.x + Math.max(1, Math.min(box.width - 1, box.width * frac)), box.y + box.height / 2);
  };
  const clickPlay = () => page.evaluate(() => document.getElementById('btn-play').click());
  const playFor = async ms => {
    const s0 = await st();
    const ps0 = JSON.stringify(s0.playStats);
    await clickPlay();
    await sleep(ms);
    const s1 = await st();
    if (s1.playing) await clickPlay();
    const s2 = await waitFor(s => !s.playing && JSON.stringify(s.playStats) !== ps0, 8000);
    const ps = (s2 ?? (await st())).playStats;
    const fresh = JSON.stringify(ps) !== ps0;
    return { drawn: fresh ? ps?.drawn ?? 0 : 0, dropped: fresh ? ps?.dropped ?? 0 : 0, seconds: fresh ? +(ps?.seconds ?? 0).toFixed(2) : 0 };
  };
  // seek through the app's hook; done when that frame is shown, or early when the engine reports an error for it
  const seekTo = async p => {
    const ev0 = (await st()).events.length;
    await page.evaluate(() => { window.__stillpoint.lastFrame = undefined; });
    await page.evaluate(p => window.__sp_seek(p), p);
    const s = await waitFor(s => s.lastFrame?.pres === p || s.events.slice(ev0).some(e => /engine error/.test(e)), 20000);
    return !!s && s.lastFrame?.pres === p;
  };

  let frames = 0, opened = false, planned = false;
  try {
    const boot = await waitFor(s => s.view === 'landing' && s.caps && s.gpu, 60000);
    res.caps = boot?.caps; res.gpu = boot?.gpu;
    if (!boot) throw new Error('app did not boot');

    await step('load', async () => {
      if (boot.inputDisabled) return { goal: false, note: 'file input disabled: browser refused (' + (boot.ui.map(u => u.text).join(' ') || 'unsupported') + ')', blocked: true };
      const input = await page.$('#file-input');
      await input.uploadFile(clip.path);
      const s = await waitFor(s => s.frames > 0 || !!s.error, 120000);
      if (!s) return { goal: false, note: 'timeout waiting for opened' };
      frames = s.frames;
      res.codec = (/decoder (\S+)/.exec(s.events.find(e => / opened /.test(e)) ?? '') ?? [])[1] ?? null;
      opened = s.frames > 0 && !s.error;
      const ev = s.events.find(e => / opened /.test(e)) ?? '';
      return { goal: opened, note: opened ? ev.replace(/^[\d.]+ /, '') : 'open failed: ' + s.error, frames };
    });
    if (!opened) throw Object.assign(new Error('not opened'), { quiet: true });

    await step('play-immediate', async () => {
      const r = await playFor(2500);
      return { goal: r.drawn > 0, note: `drawn ${r.drawn}, dropped ${r.dropped} in ${r.seconds}s`, play: r };
    });

    await step('analysis', async () => {
      const t0 = performance.now();
      let lastLog = 0;
      const s = await waitFor(s => {
        if (performance.now() - lastLog > 15000) { lastLog = performance.now(); if (performance.now() - t0 > 14000) log('    …analysis', s.badge); }
        return s.planReady || /No gyro|not supported/.test(s.badge) || (s.error && /plan|read this file/i.test(s.error));
      }, ANALYSIS_TIMEOUT, 500);
      if (!s) return { goal: false, note: 'analysis timeout' };
      planned = s.planReady;
      if (planned) await waitFor(s => !!s.encoder, 60000);
      return { goal: true, note: `${s.badge}${planned ? ', plan ready' : ', no plan'} (${((performance.now() - t0) / 1000).toFixed(1)} s)` };
    });

    await step('play-held', async () => {
      const p = Math.round(frames * 0.3);
      await timeline(0.3);
      const ev0 = (await st()).events.length;
      const held = await waitFor(s => (s.lastFrame && Math.abs(s.lastFrame.pres - p) <= 1) || s.events.slice(ev0).some(e => /engine error/.test(e)), 20000).then(s => s && s.lastFrame && Math.abs(s.lastFrame.pres - p) <= 1);
      const r = await playFor(3000);
      return { goal: !!held && r.drawn > 0, note: `preview frame ${held ? 'held' : 'NOT shown'}; play drawn ${r.drawn}, dropped ${r.dropped}`, play: r };
    });

    await step('seek-during-play', async () => {
      const draws = [];
      await clickPlay(); await sleep(1000);
      await timeline(0.6); await sleep(150);
      await clickPlay(); await sleep(1000);
      await timeline(0.9); await sleep(150);
      await clickPlay(); await sleep(600);
      // scrub drag while (maybe) playing
      const box = await tlBox();
      await page.mouse.move(box.x + box.width * 0.2, box.y + box.height / 2);
      await page.mouse.down();
      for (let i = 0; i <= 20; i++) { await page.mouse.move(box.x + box.width * (0.2 + 0.6 * i / 20), box.y + box.height / 2); await sleep(25); }
      await page.mouse.up();
      await sleep(800);
      await page.keyboard.press('Space'); await sleep(700);
      for (let i = 0; i < 3; i++) { await page.keyboard.press('ArrowRight'); await sleep(120); }
      const s = await waitFor(s => !s.playing && s.lastFrame, 8000);
      const before = s?.lastFrame?.pres ?? -1;
      // one clean play afterwards: does playback still work?
      const r = await playFor(1500);
      draws.push(r.drawn);
      return { goal: r.drawn > 0, note: `after the seek storm: frame ${before} shown, play drawn ${r.drawn}`, play: r };
    });

    await step('near-end', async () => {
      const okLast = await seekTo(frames - 1);
      const okM2 = await seekTo(frames - 2);
      const p0 = Math.max(0, frames - 20);
      const ok0 = await seekTo(p0);
      await clickPlay();
      const s = await waitFor(s => !s.playing, 8000, 200);
      await sleep(300);
      const s2 = await st();
      await page.keyboard.press('ArrowRight'); // step past the end (clamped)
      await sleep(400);
      return { goal: okLast && okM2 && ok0 && !!s, note: `seek last ${okLast ? 'ok' : 'FAILED'}, last-1 ${okM2 ? 'ok' : 'FAILED'}; played ${p0}..end -> stopped at ${s2.lastFrame?.pres} (drawn ${s2.playStats?.drawn})` };
    });

    await step('seek-leading', async () => {
      let lead = [];
      try { lead = await leadingFrames(clip.path, { limit: 3 }); } catch (e) { return { goal: null, note: 'n/a: ' + e.message }; }
      if (!lead.length) return { goal: null, note: 'n/a: every sync sample is an IDR (closed GOP)' };
      const got = [];
      for (const l of lead) got.push(`${l.pres}(${l.syncNal}@${l.sync}): ${(await seekTo(l.pres)) ? 'shown' : 'NOT shown'}`);
      return { goal: got.every(g => /shown$/.test(g) && !/NOT/.test(g)), note: 'leading frames ' + got.join(', ') };
    });

    await step('play-export', async () => {
      if (!planned) return { goal: null, note: 'skipped: no plan (no gyro / analysis failed)' };
      const s0 = await waitFor(s => !s.exportDisabled, 60000);
      if (!s0) return { goal: false, note: 'export button never enabled (encoder=' + (await st()).encoder + ')' };
      const first = Math.max(0, frames - 120);
      await page.evaluate((a, b) => window.__sp_range(a, b), first, frames - 1);
      await seekTo(first);
      await clickPlay(); await sleep(1200);
      const t0 = performance.now();
      await page.evaluate(() => document.getElementById('btn-export').click());
      const s = await waitFor(s => s.exportState === 'done' || s.exportState === 'error' || (s.exportState === 'idle' && s.error), EXPORT_TIMEOUT, 500);
      if (!s) return { goal: false, note: 'export timeout' };
      return { goal: s.exportState === 'done', note: s.exportState === 'done' ? `exported ${s.result?.frames} frames at ${s.result?.fps?.toFixed(1)} fps (${s.result?.codec}) in ${((performance.now() - t0) / 1000).toFixed(1)} s` : `export ${s.exportState}: ${s.error}`, export: s.result };
    });

    await step('after-export', async () => {
      if (!planned) return { goal: null, note: 'skipped' };
      await sleep(1500); // runExport's finally re-seeks the last preview frame
      await page.evaluate(() => document.getElementById('btn-error-ok')?.click());
      await page.evaluate(() => { const c = document.getElementById('card-done'); if (c && !c.hidden) document.getElementById('btn-again')?.click(); });
      await page.evaluate(() => window.__sp_range(0, 1e9));
      await seekTo(Math.round(frames * 0.5));
      const r = await playFor(1500);
      return { goal: r.drawn > 0, note: `play after export drawn ${r.drawn}`, play: r };
    });
  } catch (e) {
    if (!e.quiet) { res.harnessError = String(e?.stack ?? e); log('  harness error', e?.message ?? e); }
  } finally {
    res.decoderDiag = await diagDetail();
    res.finalError = (await st().catch(() => ({}))).error ?? null;
    await ctx.close().catch(() => {});
    res.vtAfter = vt();
  }
  // verdict
  const goals = res.steps.map(s => s.goal).filter(g => g === true || g === false);
  const allErr = res.steps.flatMap(s => [...uiErrors(s), ...decErrors(s)]);
  res.notices = [...new Set(res.steps.flatMap(s => s.ui.filter(u => /^unsupported: /.test(u))))];
  res.errors = [...new Set(allErr)];
  res.verdict = res.steps[0]?.blocked ? 'blocked' : goals.includes(false) ? 'FAIL' : res.errors.length ? 'degraded' : 'ok';
  return res;
}

// ── Chrome's own error messages for bad input (run in the page's main thread) ──
async function probe(browser, base, flagSet) {
  const { fileBlob } = await import('./nal-scan.mjs');
  const { openMp4, mainVideoTrack, readSamples } = await import('../../src/mp4.ts');
  const o3 = resolveClip('o3'), oa4 = resolveClip('oa4');
  const get = async (clip, n) => { const b = fileBlob(clip.path); const info = await openMp4(b); const t = mainVideoTrack(info); const s = await readSamples(b, t, 0, n); return { codec: t.codecString, w: t.width, h: t.height, desc: Buffer.from(t.codecConfig).toString('base64'), samples: s.map(x => Buffer.from(x).toString('base64')) }; };
  const avc = o3 ? await get(o3, 2) : null;
  const hevc = oa4 ? await get(oa4, 1) : null;
  const page = await (await browser.createBrowserContext()).newPage();
  await page.goto(base, { waitUntil: 'load' });
  const out = await page.evaluate(async (avc, hevc) => {
    const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
    const run = async (cfg, chunks) => new Promise(resolve => {
      const res = { outputs: 0, error: null, thrown: null };
      let done = false;
      const fin = () => { if (!done) { done = true; try { dec.close(); } catch {} resolve(res); } };
      const dec = new VideoDecoder({ output: f => { res.outputs++; f.close(); }, error: e => { res.error = `${e.name}: ${e.message}`; fin(); } });
      try {
        dec.configure(cfg);
        for (const [type, data] of chunks) dec.decode(new EncodedVideoChunk({ type, timestamp: 0, data }));
        dec.flush().then(fin, e => { res.error ??= `flush ${e.name}: ${e.message}`; setTimeout(fin, 50); });
      } catch (e) { res.thrown = `${e.name}: ${e.message}`; fin(); }
      setTimeout(fin, 8000);
    });
    const r = {};
    const sup = async cfg => { try { return (await VideoDecoder.isConfigSupported(cfg)).supported; } catch (e) { return 'throws ' + e.name; } };
    for (const [name, t] of [['avc', avc], ['hevc', hevc]]) {
      if (!t) continue;
      for (const hw of ['prefer-hardware', 'no-preference', 'prefer-software']) r[`isConfigSupported ${t.codec} ${t.w}x${t.h} ${hw}`] = await sup({ codec: t.codec, codedWidth: t.w, codedHeight: t.h, description: b64(t.desc), hardwareAcceleration: hw });
    }
    if (avc) {
      const desc = b64(avc.desc), idr = b64(avc.samples[0]), p1 = b64(avc.samples[1]);
      const garbage = new Uint8Array(idr.length); for (let i = 0; i < garbage.length; i++) garbage[i] = (i * 2654435761) >>> 24;
      garbage.set([0, 0, 0x10, 0, 0x65], 0); // a length prefix + IDR NAL header, then noise
      const zeros = new Uint8Array(idr.length);
      for (const hw of ['prefer-hardware', 'prefer-software']) {
        const cfg = { codec: avc.codec, codedWidth: avc.w, codedHeight: avc.h, description: desc, hardwareAcceleration: hw };
        r[`avc ${hw}: valid IDR`] = await run(cfg, [['key', idr]]);
        r[`avc ${hw}: noise marked key`] = await run(cfg, [['key', garbage]]);
        r[`avc ${hw}: all-zero sample marked key`] = await run(cfg, [['key', zeros]]);
        r[`avc ${hw}: empty chunk`] = await run(cfg, [['key', new Uint8Array(0)]]);
        r[`avc ${hw}: P-frame marked key`] = await run(cfg, [['key', p1]]);
        r[`avc ${hw}: IDR cut to 1/3`] = await run(cfg, [['key', idr.slice(0, idr.length / 3 | 0)]]);
        r[`avc ${hw}: IDR then all-zero delta`] = await run(cfg, [['key', idr], ['delta', zeros]]);
        r[`avc ${hw}: no description (Annex B expected)`] = await run({ ...cfg, description: undefined }, [['key', idr]]);
      }
    }
    if (hevc) {
      const cfg = { codec: hevc.codec, codedWidth: hevc.w, codedHeight: hevc.h, description: b64(hevc.desc) };
      for (const hw of ['prefer-hardware', 'no-preference', 'prefer-software']) r[`hevc ${hw}: valid IDR`] = await run({ ...cfg, hardwareAcceleration: hw }, [['key', b64(hevc.samples[0])]]);
    }
    return r;
  }, avc, hevc);
  await page.close();
  return { flags: flagSet, ...out };
}

// ── main ──
const server = await serve(TARGET, { buildDir: join(SCRATCH, 'build') });
log('target', server.target, server.base, '| flags', FLAGSETS.join(' '), '| faults', FAULTLIST.join(' '), '| clips', CLIPLIST.join(' '));
const results = { target: server.target, base: server.base, chrome: '', started: new Date().toISOString(), vtBefore: vt(), probes: [], runs: [] };
const clips = [];
for (const n of CLIPLIST) { const c = resolveClip(n); if (c) clips.push(c); else { log('clip not available, skipped:', n); results.runs.push({ clip: n, verdict: 'unavailable' }); } }
let exitCode = 0;
try {
  for (const fs of FLAGSETS) {
    const browser = await launch(flagArgs(fs));
    results.chrome ||= await browser.version();
    try {
      if (PROBE) { log(`probe [${fs}]`); const p = await probe(browser, server.base, fs); results.probes.push(p); for (const [k, v] of Object.entries(p)) if (k !== 'flags') log('  ', k, '->', JSON.stringify(v)); }
      for (const fault of FAULTLIST) {
        for (const clip of clips) {
          log(`run [${fs}] [fault ${fault}] ${clip.name}: ${clip.desc}`);
          const r = await runOne(browser, server.base, fs, fault, clip);
          results.runs.push(r);
          log(`  => ${r.verdict}${r.errors?.length ? ': ' + r.errors.join(' | ') : ''}  (VT ${r.vtBefore}->${r.vtAfter})`);
          if (STRICT && !DAMAGED.has(clip.name) && fault === 'none' && (r.verdict === 'FAIL' || r.verdict === 'degraded')) {
            const openRefused = r.steps[0] && r.steps[0].goal === false && /not supported|won’t open|decoder/i.test(r.steps[0].note ?? '');
            if (!openRefused) exitCode = 1;
          }
          if (STRICT_FAULTS && fault !== 'none' && !DAMAGED.has(clip.name) && /^avc/.test(r.codec ?? '') && r.verdict === 'FAIL') exitCode = 1;
        }
      }
    } finally { await browser.close(); }
  }
} finally {
  await server.close();
  if (!KEEP) removeDamaged();
  results.vtAfter = vt();
  results.finished = new Date().toISOString();
  const file = join(OUT, `matrix-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
  writeFileSync(file, JSON.stringify(results, null, 1));
  // compact summary: which combinations show which errors, first seen at which step
  console.log('\n=== SUMMARY ===');
  for (const r of results.runs) {
    if (!r.steps) { console.log(`${r.clip}: ${r.verdict}`); continue; }
    const first = new Map();
    for (const s of r.steps) for (const e of [...uiErrors(s), ...decErrors(s)]) if (!first.has(e)) first.set(e, s.step);
    const missed = r.steps.filter(s => s.goal === false).map(s => s.step);
    console.log(`[${r.flags}] [${r.fault}] ${r.clip}: ${r.verdict}${missed.length ? ' (missed: ' + missed.join(',') + ')' : ''}${r.notices?.length ? '  [landing notice: ' + r.notices.join(' ').replace(/unsupported: /g, '') + ']' : ''}`);
    for (const [e, s] of first) console.log(`    @${s}: ${e}`);
  }
  console.log(`VT ${results.vtBefore} -> ${results.vtAfter}; results: ${file}`);
  process.exit(exitCode);
}
