// EXPORT OPTIONS end-to-end: headless Chrome (never a visible window) drives the real app UI — size / aspect / frame
// rate / timing / motion blur / quality / codec — exports each configuration of a list through the OPFS sink, then
// checks every MP4 with ffprobe (size, rate, frame count, duration, uniform timestamps, audio present / absent as
// expected), scans all frames for black edges, and extracts stills (full frame + 1:1 crop) to look at.
//
//   node test/export.e2e.mjs [--clip PATH] [--window START:DUR (s)] [--configs a,b,…] [--target dev|build|docs]
//                            [--out DIR] [--keep a,b|all] [--shots] [--tag NAME]
//
// Configurations (CONFIGS below): hd169, v720, k27blur, sq1440slow, src4k, …  Outputs and stills go to
// ~/Library/Application Support/Stillpoint/scratch/export-ui (exports are deleted unless --keep names them).
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, rmSync, statSync, writeFileSync, renameSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { launch } from './chrome.mjs';
import { serve } from './robustness/server.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? (args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : true) : d; };
const CLIP = opt('clip', join(process.env.STILLPOINT_O3_DIR ?? join(homedir(), 'Desktop/untitled folder 4'), 'DJI_0026.MP4'));
const WINDOW = opt('window', '');
const TARGET = opt('target', 'dev');
const OUT = opt('out', join(homedir(), 'Library/Application Support/Stillpoint/scratch/export-ui'));
const KEEP = String(opt('keep', '')).split(',').filter(Boolean);
const SHOTS = !!opt('shots', false);
const TAG = opt('tag', '');
const TIMEOUT = +opt('timeout', 20 * 60) * 1000;
// extra page query, e.g. --query sp_blur=frames (plain frame-blend motion blur, for A/B)
const QUERY = opt('query', '');

// size / aspect / fps (select value) / timing / blur / quality / codec, and what the file must be
export const CONFIGS = {
  hd169: { size: '1080p', aspect: '16:9', fps: 'source', timing: 'realtime', blur: false },
  v720: { size: '720p', aspect: '9:16', fps: '30', timing: 'realtime', blur: false },
  k27blur: { size: '2.7k', aspect: 'source', fps: '24', timing: 'realtime', blur: true },
  sq1440slow: { size: '1440p', aspect: '1:1', fps: '24', timing: 'slowmo', blur: false },
  src4k: { size: 'source', aspect: 'source', fps: 'source', timing: 'realtime', blur: false },
  src5994: { size: 'source', aspect: 'source', fps: '59.94', timing: 'realtime', blur: false },
  k27: { size: '2.7k', aspect: 'source', fps: '24', timing: 'realtime', blur: false },
  hd169avc: { size: '1080p', aspect: '16:9', fps: 'source', timing: 'realtime', blur: false, codec: 'avc' },
  hd4325: { size: '1080p', aspect: '4:3', fps: '25', timing: 'realtime', blur: true },
  custom: { size: 'custom', width: 1600, aspect: '16:9', fps: 'custom', customFps: 48, timing: 'realtime', blur: false },
  hd30blur: { size: '1080p', aspect: '16:9', fps: '30', timing: 'realtime', blur: true },
  av1720: { size: '720p', aspect: '16:9', fps: '30', timing: 'realtime', blur: false, codec: 'av1' },
};
const NAMES = String(opt('configs', 'hd169,v720,k27blur,sq1440slow,src4k')).split(',');

const shotsDir = join(OUT, 'shots'), framesDir = join(OUT, 'frames'), dlDir = join(OUT, 'downloads'), keepDir = join(OUT, 'kept');
for (const d of [OUT, shotsDir, framesDir, dlDir, keepDir]) mkdirSync(d, { recursive: true });
const vt = () => { try { return +execFileSync('sh', ['-c', 'pgrep -x VTDecoderXPCService | wc -l']).toString().trim(); } catch { return -1; } };
const log = (...a) => console.log(`[exp ${(performance.now() / 1000).toFixed(1)}s]`, ...a);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const sh = (cmd, a) => execFileSync(cmd, a, { maxBuffer: 256 << 20 }).toString();

/** min / count of dark frames of the mean luma in a 2 px strip along each edge, over ALL frames */
function edgeScan(file, w, h) {
  const edges = { left: `crop=2:${h}:0:0`, right: `crop=2:${h}:${w - 2}:0`, top: `crop=${w}:2:0:0`, bottom: `crop=${w}:2:0:${h - 2}` };
  const out = {};
  for (const [k, crop] of Object.entries(edges)) {
    const txt = sh('ffmpeg', ['-v', 'error', '-i', file, '-vf', `${crop},signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-`, '-f', 'null', '-']);
    const ys = [...txt.matchAll(/YAVG=([\d.]+)/g)].map(m => +m[1]);
    out[k] = { min: +Math.min(...ys).toFixed(1), dark: ys.filter(y => y < 20).length, frames: ys.length };
  }
  return out;
}

const result = { clip: CLIP, window: WINDOW, target: TARGET, vtBefore: vt(), runs: [] };
let exitCode = 0;
const srv = await serve(TARGET, { buildDir: join(OUT, 'build') });
log('serving', srv.base, `(${srv.target})`);
const browser = await launch();
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') { errors.push(m.text()); console.log('[page error]', m.text().slice(0, 300)); } });
  page.on('pageerror', e => { errors.push(e.message); console.log('[pageerror]', e.message); });
  const cdp = await browser.target().createCDPSession();
  await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: dlDir, eventsEnabled: true });
  const downloads = new Map();
  cdp.on('Browser.downloadWillBegin', e => downloads.set(e.guid, { name: e.suggestedFilename, state: 'started' }));
  cdp.on('Browser.downloadProgress', e => { const d = downloads.get(e.guid); if (d) d.state = e.state; });
  // --shots: a tall window so the whole side panel (stabilization + export cards) is on screen for the screenshots
  await page.setViewport({ width: 1440, height: SHOTS ? 1720 : 900, deviceScaleFactor: 2 });
  await page.goto(`${srv.base}?sink=opfs${QUERY ? '&' + QUERY : ''}`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__stillpoint?.view === 'landing' && window.__stillpoint?.gpu, { timeout: 60000 });

  let evShown = 0;
  const waitFor = async (label, src, timeout) => {
    const t0 = performance.now();
    for (;;) {
      const s = await page.evaluate(`(() => { const d = window.__stillpoint; return { ok: !!(${src}), err: d.error, ev: d.events.slice() }; })()`);
      for (const e of s.ev.slice(evShown)) log('[app]', e);
      evShown = s.ev.length;
      if (s.ok) return;
      if (s.err) throw new Error(`${label}: page error: ${s.err}`);
      if (performance.now() - t0 > timeout) throw new Error(`${label}: timeout`);
      await sleep(300);
    }
  };
  await (await page.$('#file-input')).uploadFile(CLIP);
  await waitFor('opened', 'd.clip?.frames > 0', 120000);
  await waitFor('plan', 'd.planReady', 900000);
  const clip = await page.evaluate(() => ({ frames: window.__stillpoint.clip.frames, fps: window.__stillpoint.clip.fps }));
  let range = null;
  if (WINDOW) {
    const [s0, dur] = WINDOW.split(':').map(Number);
    const a = Math.round(s0 * clip.fps), b = Math.min(clip.frames - 1, a + Math.round(dur * clip.fps) - 1);
    await page.evaluate((a, b) => window.__sp_range(a, b), a, b);
    range = [a, b];
  }
  result.clipInfo = { ...clip, range };
  log('clip', JSON.stringify(result.clipInfo));

  // click like a user: scroll the control to the middle of its scroller first (the sticky Stabilize bar must not cover it)
  const tap = async sel => {
    await page.$eval(sel, el => el.scrollIntoView({ block: 'center', inline: 'nearest' }));
    await sleep(60);
    await page.click(sel);
  };
  for (const name of NAMES) {
    const cfg = CONFIGS[name];
    if (!cfg) throw new Error('unknown config ' + name);
    const run = { name, cfg };
    result.runs.push(run);
    log(`── ${name}`, JSON.stringify(cfg));
    try {
      // drive the controls like a user
      await tap(`#seg-size button[data-size="${cfg.size}"]`);
      if (cfg.size === 'custom') {
        await tap('#in-custom-w');
        await page.click('#in-custom-w', { clickCount: 3 });
        await page.type('#in-custom-w', String(cfg.width));
        await page.$eval('#in-custom-w', el => el.dispatchEvent(new Event('change', { bubbles: true })));
      }
      await tap(`#seg-aspect button[data-aspect="${cfg.aspect}"]`);
      await page.select('#sel-fps', cfg.fps);
      if (cfg.fps === 'custom') {
        await tap('#in-custom-fps');
        await page.click('#in-custom-fps', { clickCount: 3 });
        await page.type('#in-custom-fps', String(cfg.customFps));
        await page.$eval('#in-custom-fps', el => el.dispatchEvent(new Event('change', { bubbles: true })));
      }
      await tap(`#seg-timing button[data-timing="${cfg.timing}"]`);
      const blur = await page.$eval('#in-blur', el => ({ checked: el.checked, disabled: el.disabled, why: document.getElementById('blur-sub').textContent }));
      if (!!cfg.blur !== blur.checked) {
        if (blur.disabled) throw new Error(`motion blur switch disabled: ${blur.why}`);
        await tap('#in-blur');
      }
      run.blurHint = blur.why;
      await tap(`#seg-quality button[data-q="${cfg.quality ?? 'high'}"]`);
      await tap(`#seg-codec button[data-c="${cfg.codec ?? 'hevc'}"]`);
      // re-plan (aspect) + encoder probe for exactly these settings, then the button
      await waitFor('ready', `d.planReady && d.exportPlan && d.clip?.encoder && d.clip.encoder.width === d.exportPlan.width && d.clip.encoder.height === d.exportPlan.height && Math.abs(d.clip.encoder.fps - d.exportPlan.fps) < 1e-6 && !document.getElementById('btn-export').disabled`, 900000);
      await sleep(400);
      const ui = await page.evaluate(() => ({
        exportPlan: window.__stillpoint.exportPlan, encoder: window.__stillpoint.clip.encoder, plan: window.__stillpoint.clip.plan,
        facts: [...document.querySelectorAll('#export-facts dt')].map(dt => `${dt.textContent}: ${dt.nextElementSibling.textContent}`),
        dims: document.getElementById('out-dims').textContent, summary: document.getElementById('out-summary').textContent,
        blurSub: document.getElementById('blur-sub').textContent, timingSm: document.getElementById('timing-sm').textContent,
        codecs: [...document.querySelectorAll('#seg-codec button')].filter(b => !b.hidden).map(b => b.dataset.c),
      }));
      run.ui = ui;
      log('ui', JSON.stringify({ dims: ui.dims, facts: ui.facts, codecs: ui.codecs, plan: `${ui.plan.outW}x${ui.plan.outH}` }));
      if (SHOTS) {
        await page.evaluate(() => { window.scrollTo(0, 0); document.querySelector('.panel').scrollTop = 0; });
        await sleep(700);   // the preview re-draws at the new aspect
        await page.screenshot({ path: join(shotsDir, `${TAG}${name}-work.png`) });
        const panel = await page.$('#card-export');
        await panel.screenshot({ path: join(shotsDir, `${TAG}${name}-panel.png`) });
      }
      for (const f of readdirSync(dlDir)) rmSync(join(dlDir, f), { force: true });
      downloads.clear();
      const t0 = performance.now();
      await page.evaluate(() => { const d = window.__stillpoint; d.exportState = 'idle'; d.result = undefined; d.error = undefined; d.progress = undefined; });
      await tap('#btn-export');
      let lastLog = 0;
      for (;;) {
        const s = await page.evaluate(() => ({ st: window.__stillpoint.exportState, p: window.__stillpoint.progress, r: window.__stillpoint.result, e: window.__stillpoint.error }));
        if (s.st === 'done') { run.export = s.r; break; }
        if (s.st === 'error') throw new Error('export failed: ' + s.e);
        if (s.st === 'idle' && s.e && performance.now() - t0 > 1500) throw new Error('export refused: ' + s.e);
        if (performance.now() - lastLog > 5000 && s.p) { lastLog = performance.now(); log(`export ${s.p.done}/${s.p.total} ${s.p.fps.toFixed(1)} fps VT=${vt()}`); }
        if (performance.now() - t0 > TIMEOUT) throw new Error('export timeout');
        await sleep(400);
      }
      run.wallS = +((performance.now() - t0) / 1000).toFixed(2);
      if (SHOTS && name === NAMES[0]) await page.screenshot({ path: join(shotsDir, `${TAG}done-desktop.png`) });
      let file = null;
      const t1 = performance.now();
      while (performance.now() - t1 < 120000) {
        const done = [...downloads.values()].find(d => d.state === 'completed');
        if (done && existsSync(join(dlDir, done.name))) { file = join(dlDir, done.name); break; }
        await sleep(250);
      }
      if (!file) throw new Error('download did not complete');
      const kept = join(keepDir, `${TAG}${name}.mp4`);
      renameSync(file, kept);
      file = kept;
      run.bytes = statSync(file).size;

      // ── ffprobe
      const probe = JSON.parse(sh('ffprobe', ['-v', 'error', '-count_frames', '-show_entries',
        'stream=index,codec_type,codec_name,profile,width,height,pix_fmt,avg_frame_rate,r_frame_rate,time_base,nb_frames,nb_read_frames,duration,sample_rate,channels:format=duration,size', '-of', 'json', file]));
      const v = probe.streams.find(s => s.codec_type === 'video');
      const a = probe.streams.find(s => s.codec_type === 'audio');
      const pts = sh('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts', '-of', 'csv=p=0', file])
        .trim().split('\n').map(Number).sort((x, y) => x - y);
      const tb = v.time_base.split('/').map(Number);
      const deltas = pts.slice(1).map((p, i) => p - pts[i]);
      const dmin = Math.min(...deltas), dmax = Math.max(...deltas);
      const ep = ui.exportPlan;
      const rate = v.avg_frame_rate.split('/').map(Number);
      const fr = rate[0] / rate[1];
      run.probe = {
        codec: `${v.codec_name} ${v.profile ?? ''} ${v.pix_fmt}`, size: `${v.width}x${v.height}`, avgFps: +fr.toFixed(4), timeBase: v.time_base,
        frames: +v.nb_read_frames, durationS: +v.duration, formatDurS: +probe.format.duration,
        frameStepTicks: dmin === dmax ? dmin : `${dmin}..${dmax}`, firstPts: pts[0],
        audio: a ? `${a.codec_name} ${a.sample_rate} Hz ${a.channels} ch ${(+a.duration).toFixed(3)} s` : 'none',
      };
      const expectDims = `${ep.width}x${ep.height}`;
      const tick = tb[0] / tb[1];
      run.checks = {
        size: `${run.probe.size} (expected ${expectDims})`, sizeOk: run.probe.size === expectDims,
        fps: `${run.probe.avgFps} (expected ${ep.fps.toFixed(4)})`, fpsOk: Math.abs(fr - ep.fps) < 0.01,
        frames: `${run.probe.frames} (expected ${ep.frames}, export says ${run.export.frames})`, framesOk: run.probe.frames === ep.frames && run.export.frames === ep.frames,
        duration: `${run.probe.durationS} s (expected ${ep.seconds.toFixed(4)})`, durationOk: Math.abs(run.probe.durationS - ep.seconds) < 1.5 / ep.fps,
        // constant frame step (the source's own jitter-free timeline for 'source'; 1 tick of rounding allowed)
        uniformOk: cfg.fps === 'source' ? true : dmax - dmin <= 1 && Math.abs(dmin * tick - 1 / ep.fps) < 1.5 * tick,
        startsAtZero: pts[0] === 0,
        audio: `${run.probe.audio} (expected ${ep.audio ? 'present' : 'absent'})`, audioOk: !!a === !!ep.audio,
      };
      // ── black edges over every frame
      run.edges = edgeScan(file, v.width, v.height);
      run.checks.edgesOk = Object.values(run.edges).every(e => e.dark === 0);
      // ── stills: 3 full frames (<= 1600 px wide) + a 1:1 crop of the middle one
      const dur = run.probe.durationS;
      run.stills = [];
      for (const [i, f] of [0.25, 0.5, 0.8].entries()) {
        const png = join(framesDir, `${TAG}${name}_${i}.jpg`);
        sh('ffmpeg', ['-v', 'error', '-y', '-ss', (dur * f).toFixed(3), '-i', file, '-frames:v', '1', '-vf', `scale='min(1600,iw)':-2:flags=lanczos`, '-q:v', '2', png]);
        run.stills.push(png);
      }
      const cw = Math.min(800, v.width), ch = Math.min(600, v.height);
      const crop = join(framesDir, `${TAG}${name}_crop.png`);
      sh('ffmpeg', ['-v', 'error', '-y', '-ss', (dur * 0.5).toFixed(3), '-i', file, '-frames:v', '1', '-vf', `crop=${cw}:${ch}:${Math.round((v.width - cw) * 0.3)}:${Math.round((v.height - ch) * 0.75)}`, crop]);
      run.stills.push(crop);
      const ok = Object.entries(run.checks).filter(([k]) => k.endsWith('Ok') || k === 'startsAtZero').every(([, x]) => x === true);
      run.pass = ok;
      if (!ok) exitCode = 1;
      log(name, ok ? 'PASS' : 'FAIL', JSON.stringify(run.checks), JSON.stringify(run.edges));
      if (!KEEP.includes(name) && !KEEP.includes('all')) rmSync(file, { force: true });
      else run.file = file;
      // back to the export card for the next run
      await page.evaluate(() => document.getElementById('btn-again')?.click());
      await sleep(300);
    } catch (e) {
      run.failure = String(e?.message ?? e);
      run.pass = false;
      exitCode = 1;
      log(name, 'FAILED', run.failure);
      // never leave an export running into the next configuration
      if (await page.evaluate(() => window.__stillpoint.exportState === 'running')) {
        await page.evaluate(() => document.getElementById('btn-cancel')?.click());
        await page.waitForFunction(() => window.__stillpoint.exportState !== 'running', { timeout: 60000 }).catch(() => {});
      }
      await page.evaluate(() => { document.getElementById('btn-error-ok')?.click(); document.getElementById('btn-again')?.click(); });
      await sleep(500);
    }
  }
  if (SHOTS) {
    // phone width: the export card must not overflow
    await page.setViewport({ width: 375, height: 812, deviceScaleFactor: 3 }); // (isMobile would reload the page)
    await sleep(800);
    await page.evaluate(() => document.getElementById('card-export').scrollIntoView({ block: 'start' }));
    await sleep(400);
    await page.screenshot({ path: join(shotsDir, `${TAG}phone-export.png`) });
    result.phone = await page.evaluate(() => ({
      docScrollW: document.documentElement.scrollWidth, innerW: window.innerWidth,
      overflow: [...document.querySelectorAll('#card-export *')].filter(el => el.getBoundingClientRect().right > window.innerWidth + 0.5).map(el => el.id || el.className).slice(0, 10),
    }));
    log('phone', JSON.stringify(result.phone));
    await page.evaluate(() => window.scrollTo(0, 0));
    await sleep(300);
    await page.screenshot({ path: join(shotsDir, `${TAG}phone-top.png`) });
  }
  result.pageErrors = errors.slice(0, 10);
} catch (e) {
  console.error('[export e2e] FAILED', e);
  result.failure = String(e?.message ?? e);
  exitCode = 1;
} finally {
  await browser.close();
  await srv.close();
  result.vtAfter = vt();
  writeFileSync(join(OUT, `result${TAG ? '-' + TAG.replace(/[^\w]+$/, '') : ''}.json`), JSON.stringify(result, null, 1));
  console.log(JSON.stringify(result.runs.map(r => ({ name: r.name, pass: r.pass, failure: r.failure, checks: r.checks, edges: r.edges, probe: r.probe, wallS: r.wallS, retime: r.export?.retime, renderer: r.export?.renderer, warpMode: r.export?.warpMode, codec: r.export?.codec, bytes: r.bytes })), null, 1));
  console.log(`VT ${result.vtBefore} -> ${result.vtAfter}`);
  process.exit(exitCode);
}
