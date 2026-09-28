// APP end-to-end test: headless Chrome drives the real page (vite dev server on 127.0.0.1), opens a clip through the
// file input, waits for telemetry + plan, exports through the real UI (OPFS sink -> download), then validates the
// MP4 with ffprobe (frame count, duration, audio) and reports throughput.
//
//   node test/app.e2e.mjs [--clip PATH] [--range first:last] [--params '{"outFx":1548.8}'] [--quality high]
//                         [--shots] [--no-export] [--out DIR] [--keep] [--psnr] [--prod] [--url URL]
//   --url: skip the local server and test an already-deployed app, e.g. https://jbittlest.github.io/stillpoint/app/
//
// Never opens a visible window (--headless=new). Outputs go to ~/Library/Application Support/Stillpoint/scratch/webapp.
import { createServer, build as viteBuild, preview } from 'vite';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { mkdirSync, existsSync, readdirSync, statSync, rmSync } from 'node:fs';
import { join, basename } from 'node:path';
import { homedir } from 'node:os';
import { launch } from './chrome.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? (args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : true) : d; };
const CLIP = opt('clip', join(process.env.STILLPOINT_O3_DIR ?? join(homedir(), 'Desktop/untitled folder 4'), 'DJI_0026.MP4'));
const RANGE = opt('range', '');
// StabParams overrides applied before the export (e.g. the Python plan's focal for A/B quality runs)
const PARAMS = opt('params', '');
const QUALITY = opt('quality', 'high');
const CODEC = opt('codec', 'hevc');
const SHOTS = !!opt('shots', false);
const EXPORT = !opt('no-export', false);
const OUT = opt('out', join(homedir(), 'Library/Application Support/Stillpoint/scratch/webapp'));
const KEEP = !!opt('keep', false);
const PSNR = !!opt('psnr', false);
const SINK = opt('sink', 'opfs');
const PLAY = !!opt('play', false);
const CANCEL = !!opt('cancel', false);
// --prod: test the production build exactly as GitHub Pages serves it (base /stillpoint/app/)
const PROD = !!opt('prod', false);
// --url: test a deployed copy (e.g. the live GitHub Pages app) instead of starting a server
const URL_ = opt('url', '');
const TIMEOUT = +opt('timeout', 30 * 60) * 1000;
mkdirSync(OUT, { recursive: true });
const dlDir = join(OUT, 'downloads');
mkdirSync(dlDir, { recursive: true });

const vt = () => { try { return +execFileSync('sh', ['-c', 'pgrep -x VTDecoderXPCService | wc -l']).toString().trim(); } catch { return -1; } };
const vt0 = vt();
const log = (...a) => console.log(`[e2e ${(performance.now() / 1000).toFixed(1)}s]`, ...a);

const root = fileURLToPath(new URL('..', import.meta.url));
let server, base;
if (URL_) {
  base = URL_;
  server = { close: async () => {} };
} else if (PROD) {
  const outDir = join(OUT, 'build/stillpoint/app');
  await viteBuild({ root, configFile: join(root, 'vite.config.ts'), base: '/stillpoint/app/', logLevel: 'error', build: { outDir, emptyOutDir: true } });
  server = await preview({ root, configFile: join(root, 'vite.config.ts'), base: '/stillpoint/app/', build: { outDir }, preview: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
  base = server.resolvedUrls.local[0];
  if (!base.endsWith('/stillpoint/app/')) base = base.replace(/\/?$/, '/') + 'stillpoint/app/';
} else {
  // hmr off: other agents edit src/ concurrently and a full reload would reset the page mid-test
  server = await createServer({ configFile: join(root, 'vite.config.ts'), root, server: { port: 0, host: '127.0.0.1', hmr: false, watch: { ignored: ['**/*'] } }, logLevel: 'error' });
  await server.listen();
  base = server.resolvedUrls.local[0];
}
log('serving', base);
const qs = new URLSearchParams({ sink: SINK });
const url = `${base}${base.includes('?') ? '&' : '?'}${qs}`;
const browser = await launch();
// RSS of the headless browser's whole process tree (MB), sampled during the export
const treeRss = () => {
  try {
    const rows = execFileSync('ps', ['-Ao', 'pid=,ppid=,rss=']).toString().trim().split('\n').map(l => l.trim().split(/\s+/).map(Number));
    const root = browser.process()?.pid; if (!root) return -1;
    const kids = new Map(); for (const [pid, ppid] of rows) { if (!kids.has(ppid)) kids.set(ppid, []); kids.get(ppid).push(pid); }
    const rss = new Map(rows.map(r => [r[0], r[2]]));
    let sum = 0; const st = [root];
    while (st.length) { const p = st.pop(); sum += rss.get(p) ?? 0; for (const c of kids.get(p) ?? []) st.push(c); }
    return Math.round(sum / 1024);
  } catch { return -1; }
};
const memSamples = [];
const result = { clip: CLIP, vtBefore: vt0 };
let exitCode = 0;
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('console', m => { const t = m.text(); if (m.type() === 'error' || /stillpoint|warn/i.test(t)) console.log('[page]', m.type(), t.slice(0, 300)); if (m.type() === 'error') errors.push(t); });
  page.on('pageerror', e => { console.log('[pageerror]', e.message); errors.push(e.message); });
  const cdp = await browser.target().createCDPSession();
  await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: dlDir, eventsEnabled: true });
  const downloads = new Map();
  cdp.on('Browser.downloadWillBegin', e => downloads.set(e.guid, { name: e.suggestedFilename, state: 'started' }));
  cdp.on('Browser.downloadProgress', e => { const d = downloads.get(e.guid); if (d) d.state = e.state; });

  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__stillpoint?.view === 'landing' && window.__stillpoint?.gpu, { timeout: 60000 });
  const boot = await page.evaluate(() => ({ caps: window.__stillpoint.caps, gpu: window.__stillpoint.gpu }));
  result.caps = boot.caps; result.gpu = boot.gpu;
  log('caps', JSON.stringify(boot));
  if (SHOTS) { await page.screenshot({ path: join(OUT, 'landing-desktop.png') }); }

  let evShown = 0;
  const waitFor = async (label, fnSrc, timeout) => {
    const t0 = performance.now();
    let lastPrint = 0;
    for (;;) {
      const s = await page.evaluate(`(() => { const d = window.__stillpoint; return { ok: !!(${fnSrc}), err: d.error, ev: d.events.slice() }; })()`);
      for (const e of s.ev.slice(evShown)) log('[app]', e);
      evShown = s.ev.length;
      if (s.ok) return;
      if (s.err) throw new Error(`${label}: page error: ${s.err}`);
      if (performance.now() - t0 > timeout) throw new Error(`${label}: timeout`);
      if (performance.now() - lastPrint > 10000) { lastPrint = performance.now(); log(`waiting for ${label}…`); }
      await new Promise(r => setTimeout(r, 500));
    }
  };
  const input = await page.$('#file-input');
  const tOpen = performance.now();
  await input.uploadFile(CLIP);
  await waitFor('opened', 'd.clip?.frames > 0', 120000);
  await waitFor('telemetry', 'd.clip?.telemetry', 600000);
  const telDone = performance.now();
  await waitFor('plan', 'd.planReady', 900000);
  const planDone = performance.now();
  await waitFor('encoder', 'd.clip?.encoder?.codec', 60000);
  const st = await page.evaluate(() => ({ clip: window.__stillpoint.clip, error: window.__stillpoint.error, lastFrame: window.__stillpoint.lastFrame, events: window.__stillpoint.events }));
  if (st.error) throw new Error('page error: ' + st.error);
  result.openToTelemetryS = +((telDone - tOpen) / 1000).toFixed(2);
  result.openToPlanReadyS = +((planDone - tOpen) / 1000).toFixed(2);
  result.telemetry = st.clip.telemetry; result.plan = st.clip.plan; result.planMs = st.clip.planMs; result.encoder = st.clip.encoder;
  log('ready', JSON.stringify({ frames: st.clip.frames, fps: st.clip.fps, plan: st.clip.plan, planMs: st.clip.planMs, enc: st.clip.encoder, lastFrame: st.lastFrame }));

  // seek timing (single-frame preview decode + warp)
  const seekMs = [];
  for (const f of [0.25, 0.5, 0.75]) {
    const p = Math.round(f * (st.clip.frames - 1));
    await page.evaluate(() => { window.__stillpoint.lastFrame = undefined; });
    const t = performance.now();
    await page.evaluate(p => window.__sp_seek(p), p);
    await page.waitForFunction(p => window.__stillpoint.lastFrame?.pres === p, { timeout: 20000 }, p).catch(() => {});
    seekMs.push(Math.round(performance.now() - t));
  }
  result.seekMs = seekMs;

  if (SHOTS) {
    // drive the timeline to ~40 % for a representative frame
    const box = await (await page.$('#timeline')).boundingBox();
    await page.mouse.click(box.x + box.width * 0.4, box.y + box.height / 2);
    await page.waitForFunction(() => window.__stillpoint.lastFrame, { timeout: 20000 }).catch(() => {});
    await new Promise(r => setTimeout(r, 800));
    await page.screenshot({ path: join(OUT, 'work-desktop.png') });
  }

  if (PLAY) {
    // real-time preview playback for ~3 s
    await page.evaluate(() => window.__sp_seek(0));
    await new Promise(r => setTimeout(r, 500));
    await page.click('#btn-play');
    await new Promise(r => setTimeout(r, 3000));
    await page.click('#btn-play');
    await page.waitForFunction(() => window.__stillpoint.playStats, { timeout: 10000 });
    const ps = await page.evaluate(() => window.__stillpoint.playStats);
    result.preview = { ...ps, drawnFps: +(ps.drawn / ps.seconds).toFixed(1), sourceFps: st.clip.fps };
    log('preview playback', JSON.stringify(result.preview));
  }

  if (CANCEL) {
    await page.waitForFunction(() => !document.getElementById('btn-export').disabled, { timeout: 60000 });
    await page.click('#btn-export');
    await page.waitForFunction(() => (window.__stillpoint.progress?.done ?? 0) >= 30, { timeout: 120000 });
    const tc = performance.now();
    await page.click('#btn-cancel');
    await page.waitForFunction(() => window.__stillpoint.exportState === 'idle', { timeout: 30000 });
    const left = await page.evaluate(async () => {
      const root = await navigator.storage.getDirectory();
      const names = [];
      for await (const [n] of root.entries()) names.push(n);
      try { const d = await root.getDirectoryHandle('stillpoint-exports'); for await (const [n] of d.entries()) names.push('stillpoint-exports/' + n); } catch {}
      return names;
    });
    result.cancel = { ms: Math.round(performance.now() - tc), opfsLeft: left, error: await page.evaluate(() => window.__stillpoint.error) };
    log('cancel', JSON.stringify(result.cancel));
    await page.waitForFunction(() => !document.getElementById('btn-export').disabled, { timeout: 60000 });
  }

  if (EXPORT) {
    if (PSNR) {
      // PSNR check: footprint 1 (widest output) — only meaningful vs an identity plan
      await page.evaluate(() => window.__sp_params({ footprint: 1 }));
      await page.waitForFunction(() => window.__stillpoint.planReady, { timeout: 60000 });
    }
    if (PARAMS) {
      await page.evaluate(p => window.__sp_params(p), JSON.parse(PARAMS));
      await page.waitForFunction(() => window.__stillpoint.planReady, { timeout: 900000 });
      result.planWithParams = await page.evaluate(() => ({ plan: window.__stillpoint.clip?.plan, planMs: window.__stillpoint.clip?.planMs }));
      log('re-planned with', PARAMS, JSON.stringify(result.planWithParams));
    }
    if (RANGE) {
      const [a, b] = RANGE.split(':').map(Number);
      await page.evaluate((a, b) => window.__sp_range(a, b), a, b);
    }
    await page.evaluate(c => document.querySelector(`#seg-codec button[data-c="${c}"]`).click(), CODEC);
    await page.evaluate(q => document.querySelector(`#seg-quality button[data-q="${q}"]`).click(), QUALITY);
    await page.waitForFunction(c => (window.__stillpoint.clip?.encoder?.codec ?? '').startsWith(c === 'avc' ? 'avc' : 'hvc') || window.__stillpoint.clip?.encoder?.codec, { timeout: 30000 }, CODEC);
    await page.waitForFunction(() => !document.getElementById('btn-export').disabled, { timeout: 60000 });
    for (const f of readdirSync(dlDir)) rmSync(join(dlDir, f), { force: true });
    const t0 = performance.now();
    await page.click('#btn-export');
    let lastLog = 0;
    for (;;) {
      const s = await page.evaluate(() => ({ st: window.__stillpoint.exportState, p: window.__stillpoint.progress, r: window.__stillpoint.result, e: window.__stillpoint.error }));
      if (s.st === 'done' || s.st === 'error') { result.export = s.r; if (s.st === 'error') throw new Error('export failed: ' + s.e); break; }
      // a pre-flight refusal (e.g. not enough browser storage) never leaves 'idle': fail instead of waiting forever
      if (s.st === 'idle' && s.e && performance.now() - t0 > 1000) throw new Error('export refused: ' + s.e);
      if (performance.now() - lastLog > 5000 && s.p) { lastLog = performance.now(); const m = treeRss(); memSamples.push([s.p.done, m]); log(`export ${s.p.done}/${s.p.total} ${s.p.fps.toFixed(1)} fps eta ${s.p.etaS.toFixed(0)} s ${(s.p.bytes / 1e6).toFixed(0)} MB, chrome RSS ${m} MB, VT=${vt()}`); }
      if (performance.now() - t0 > TIMEOUT) throw new Error('export timeout');
      await new Promise(r => setTimeout(r, 500));
    }
    result.exportWallS = +((performance.now() - t0) / 1000).toFixed(2);
    if (memSamples.length) { const ms = memSamples.map(x => x[1]); result.chromeRssMB = { min: Math.min(...ms), max: Math.max(...ms), first: ms[0], last: ms[ms.length - 1], samples: ms.length }; }
    if (SHOTS) { await page.screenshot({ path: join(OUT, 'done-desktop.png') }); }
    // wait for the download to land
    const t1 = performance.now();
    let file = null;
    while (performance.now() - t1 < 120000) {
      const done = [...downloads.values()].find(d => d.state === 'completed');
      if (done && existsSync(join(dlDir, done.name))) { file = join(dlDir, done.name); break; }
      await new Promise(r => setTimeout(r, 250));
    }
    if (!file) throw new Error('download did not complete: ' + JSON.stringify([...downloads.values()]));
    result.output = { path: file, bytes: statSync(file).size };
    // decode-count every frame for normal outputs; for multi-GB outputs trust the sample table + decode the tail
    const big = statSync(file).size > 1.5e9;
    const probe = JSON.parse(execFileSync('ffprobe', ['-v', 'error', ...(big ? [] : ['-count_frames']), '-show_entries',
      'stream=index,codec_type,codec_name,profile,width,height,pix_fmt,avg_frame_rate,r_frame_rate,time_base,nb_frames,nb_read_frames,duration,sample_rate,channels,color_primaries,color_transfer,color_space,color_range:format=duration,size',
      '-of', 'json', file]).toString());
    result.ffprobe = probe;
    const v = probe.streams.find(s => s.codec_type === 'video');
    const a = probe.streams.find(s => s.codec_type === 'audio');
    const expected = result.export.frames;
    if (big) {
      execFileSync('ffmpeg', ['-v', 'error', '-sseof', '-2', '-i', file, '-f', 'null', '-']);
      if (v) v.nb_read_frames = v.nb_frames;
      result.bigOutputTailDecoded = true;
    }
    result.checks = {
      frames: `${v?.nb_read_frames} decoded / ${expected} expected`,
      framesOk: +v?.nb_read_frames === expected,
      durationS: +probe.format.duration,
      videoDurS: +v?.duration,
      audio: a ? `${a.codec_name} ${a.sample_rate} Hz ${a.channels} ch, ${a.duration} s` : 'none',
      color: v ? `${v.color_primaries}/${v.color_transfer}/${v.color_space}/${v.color_range}` : '',
    };
    if (!result.checks.framesOk) exitCode = 1;
    if (PSNR) {
      // identity plan only: compare output to the source (same geometry) over the exported range
      const first = RANGE ? +RANGE.split(':')[0] : 0;
      const ss = (first / (+result.export.frames ? st.clip.fps : 60)).toFixed(4);
      const out = execFileSync('sh', ['-c', `ffmpeg -hide_banner -nostats -ss ${ss} -i "${CLIP}" -i "${file}" -frames:v ${Math.min(60, expected)} -lavfi "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]psnr" -f null - 2>&1 | grep -o 'PSNR.*' | tail -1`]).toString();
      result.psnr = out.trim();
    }
    if (!KEEP) rmSync(file, { force: true });
  }
  result.pageErrors = errors.slice(0, 10);
  result.events = (await page.evaluate(() => window.__stillpoint.events)).slice(-40);
} catch (e) {
  console.error('[e2e] FAILED', e);
  result.failure = String(e?.message ?? e);
  exitCode = 1;
} finally {
  await browser.close();
  await (server.close?.() ?? new Promise(r => server.httpServer.close(r)));
  result.vtAfter = vt();
  console.log(JSON.stringify(result, null, 1));
  process.exit(exitCode);
}
