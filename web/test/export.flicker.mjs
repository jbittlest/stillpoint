// BRIGHTNESS-FLICKER regression (retimed exports on hardware decode). Headless Chrome (never a visible window) drives
// the real app UI and exports, per clip, a same-rate reference (1080p 16:9 at the source rate) and retimed versions
// (30 / 24 fps, with and without natural motion blur), then checks EVERY output frame's mean luma (ffmpeg signalstats
// YAVG, 8-bit Y' codes of the MP4) against the reference export's frames the output is made of — the exact taps and
// weights of the export's schedule (src/retime.ts buildSchedule over the reference's own frame times):
//     |YAVG(out_i) - sum_j w_ij YAVG(ref_tap_ij)| <= 1 code
// (measured with the fix: max 0.17 code incl. motion blur, whose sub-frame re-warps barely move the frame mean).
// Before the fix VideoToolbox handed a few frames of every retimed export to the Warper tagged 'iec61966-2-1' instead
// of 'bt709', which got the gamma-1.961 inverse of the other frames: 1-4 frames ~9 codes darker (warp.ts, colour classes).
// The export result's colorTags (decoded frames per '<format>|<transfer tag>') show whether a run met such frames.
//
//   node test/export.flicker.mjs [--clips o3,oa4] [--configs r30,r24,r30b,r24b] [--target dev|build|docs] [--keep]
//
// Clips (read-only, never copied): o3 = DJI_0026.MP4 (whole clip), oa4 = the Osmo Action 4 clip's first 8 s.
// Exports go to ~/Library/Application Support/Stillpoint/scratch/fixer/flicker and are deleted after the check.
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, rmSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { launch } from './chrome.mjs';
import { serve } from './robustness/server.mjs';
import { buildSchedule } from '../src/retime.ts';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? (args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : true) : d; };
const TARGET = opt('target', 'dev');
const OUT = opt('out', join(homedir(), 'Library/Application Support/Stillpoint/scratch/fixer/flicker'));
const KEEP = !!opt('keep', false);
const CLIPS = {
  o3: { path: join(process.env.STILLPOINT_O3_DIR ?? join(homedir(), 'Desktop/untitled folder 4'), 'DJI_0026.MP4') },
  oa4: { path: process.env.CLIP_OA4 ?? join(homedir(), 'Desktop/DJI_20260927091931_0012_D.MP4'), window: [0, 8] },
};
const CONFIGS = {
  ref: { size: '1080p', aspect: '16:9', fps: 'source', timing: 'realtime', blur: false },
  r30: { size: '1080p', aspect: '16:9', fps: '30', timing: 'realtime', blur: false },
  r24: { size: '1080p', aspect: '16:9', fps: '24', timing: 'realtime', blur: false },
  r30b: { size: '1080p', aspect: '16:9', fps: '30', timing: 'realtime', blur: true },
  r24b: { size: '1080p', aspect: '16:9', fps: '24', timing: 'realtime', blur: true },
};
const CLIP_NAMES = String(opt('clips', 'o3,oa4')).split(',').filter(Boolean);
const RUNS = String(opt('configs', 'r30,r24,r30b,r24b')).split(',').filter(Boolean);
/** max |mean luma difference| (8-bit codes), also for motion blur; the flicker it guards against is 7-9 codes */
const LIMIT = 1.0;

const dlDir = join(OUT, 'downloads'), expDir = join(OUT, 'exports');
for (const d of [OUT, dlDir, expDir]) mkdirSync(d, { recursive: true });
const vt = () => { try { return +execFileSync('sh', ['-c', 'pgrep -x VTDecoderXPCService | wc -l']).toString().trim(); } catch { return -1; } };
const log = (...a) => console.log(`[flicker ${(performance.now() / 1000).toFixed(1)}s]`, ...a);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const sh = (cmd, a) => execFileSync(cmd, a, { maxBuffer: 256 << 20 }).toString();

/** mean luma (Y' codes) of every frame, in presentation order, and the frame times (s) */
function lumaTrack(file) {
  const txt = sh('ffmpeg', ['-v', 'error', '-i', file, '-vf', 'signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-', '-f', 'null', '-']);
  const y = [...txt.matchAll(/YAVG=([\d.]+)/g)].map(m => +m[1]);
  const t = [...txt.matchAll(/pts_time:([\d.]+)/g)].map(m => +m[1]);
  return { y, t };
}

const result = { target: TARGET, vtBefore: vt(), clips: {}, limit: LIMIT };
let exitCode = 0;
const fail = msg => { exitCode = 1; log('FAIL', msg); };
const srv = await serve(TARGET, { buildDir: join(OUT, 'build') });
log('serving', srv.base, `(${srv.target})`);
const browser = await launch();
try {
  const cdp = await browser.target().createCDPSession();
  await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: dlDir, eventsEnabled: true });
  const downloads = new Map();
  cdp.on('Browser.downloadWillBegin', e => downloads.set(e.guid, { name: e.suggestedFilename, state: 'started' }));
  cdp.on('Browser.downloadProgress', e => { const d = downloads.get(e.guid); if (d) d.state = e.state; });

  for (const cname of CLIP_NAMES) {
    const clipDef = CLIPS[cname];
    if (!clipDef) throw new Error('unknown clip ' + cname);
    const R = (result.clips[cname] = { path: clipDef.path, runs: {} });
    if (!existsSync(clipDef.path)) { fail(`${cname}: clip missing (${clipDef.path})`); continue; }
    const page = await browser.newPage();
    const errors = [];
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('pageerror', e => errors.push(e.message));
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });
    await page.goto(`${srv.base}?sink=opfs`, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__stillpoint?.view === 'landing' && window.__stillpoint?.gpu, { timeout: 60000 });
    let evShown = 0;
    const waitFor = async (label, src, timeout) => {
      const t0 = performance.now();
      for (;;) {
        const s = await page.evaluate(`(() => { const d = window.__stillpoint; return { ok: !!(${src}), err: d.error, ev: d.events.slice() }; })()`);
        for (const e of s.ev.slice(evShown)) log(`[${cname}]`, e);
        evShown = s.ev.length;
        if (s.ok) return;
        if (s.err) throw new Error(`${label}: page error: ${s.err}`);
        if (performance.now() - t0 > timeout) throw new Error(`${label}: timeout`);
        await sleep(300);
      }
    };
    const tap = async sel => {
      await page.$eval(sel, el => el.scrollIntoView({ block: 'center', inline: 'nearest' }));
      await sleep(60);
      await page.click(sel);
    };
    try {
      await (await page.$('#file-input')).uploadFile(clipDef.path);
      await waitFor('opened', 'd.clip?.frames > 0', 120000);
      await waitFor('plan', 'd.planReady', 900000);
      const clip = await page.evaluate(() => ({ frames: window.__stillpoint.clip.frames, fps: window.__stillpoint.clip.fps, decoder: window.__stillpoint.decoder }));
      R.decoder = { variant: clip.decoder?.variant, label: clip.decoder?.label, software: clip.decoder?.software, fallback: clip.decoder?.fallback };
      if (!clip.decoder || clip.decoder.software || !/^hw/.test(clip.decoder.variant)) fail(`${cname}: not on the hardware decoder (${JSON.stringify(R.decoder)})`);
      if (clipDef.window) {
        const [s0, dur] = clipDef.window;
        const a = Math.round(s0 * clip.fps), b = Math.min(clip.frames - 1, a + Math.round(dur * clip.fps) - 1);
        await page.evaluate((a, b) => window.__sp_range(a, b), a, b);
        R.range = [a, b];
      }
      log(cname, JSON.stringify({ frames: clip.frames, fps: clip.fps, decoder: R.decoder, range: R.range }));

      const exportOne = async name => {
        const cfg = CONFIGS[name];
        await tap(`#seg-size button[data-size="${cfg.size}"]`);
        await tap(`#seg-aspect button[data-aspect="${cfg.aspect}"]`);
        await page.select('#sel-fps', cfg.fps);
        await tap(`#seg-timing button[data-timing="${cfg.timing}"]`);
        const blur = await page.$eval('#in-blur', el => ({ checked: el.checked, disabled: el.disabled }));
        if (!!cfg.blur !== blur.checked) {
          if (blur.disabled) throw new Error('motion blur switch disabled');
          await tap('#in-blur');
        }
        await tap('#seg-quality button[data-q="max"]');
        await tap('#seg-codec button[data-c="hevc"]');
        await waitFor('ready', `d.planReady && d.exportPlan && d.clip?.encoder && d.clip.encoder.width === d.exportPlan.width && d.clip.encoder.height === d.exportPlan.height && Math.abs(d.clip.encoder.fps - d.exportPlan.fps) < 1e-6 && !document.getElementById('btn-export').disabled`, 900000);
        await sleep(300);
        for (const f of readdirSync(dlDir)) rmSync(join(dlDir, f), { force: true });
        downloads.clear();
        await page.evaluate(() => { const d = window.__stillpoint; d.exportState = 'idle'; d.result = undefined; d.error = undefined; d.progress = undefined; });
        const t0 = performance.now();
        await tap('#btn-export');
        let res;
        for (;;) {
          const s = await page.evaluate(() => ({ st: window.__stillpoint.exportState, r: window.__stillpoint.result, e: window.__stillpoint.error }));
          if (s.st === 'done') { res = s.r; break; }
          if (s.st === 'error') throw new Error('export failed: ' + s.e);
          if (s.st === 'idle' && s.e && performance.now() - t0 > 1500) throw new Error('export refused: ' + s.e);
          if (performance.now() - t0 > 20 * 60e3) throw new Error('export timeout');
          await sleep(300);
        }
        let file = null;
        const t1 = performance.now();
        while (performance.now() - t1 < 120000) {
          const d = [...downloads.values()].find(x => x.state === 'completed');
          if (d && existsSync(join(dlDir, d.name))) { file = join(dlDir, d.name); break; }
          await sleep(250);
        }
        if (!file) throw new Error('download did not complete');
        const dst = join(expDir, `${cname}-${name}.mp4`);
        renameSync(file, dst);
        const decoder = await page.evaluate(() => window.__stillpoint.decoder);
        await page.evaluate(() => document.getElementById('btn-again')?.click());
        await sleep(300);
        return { file: dst, res, decoder, wallS: +((performance.now() - t0) / 1000).toFixed(2) };
      };

      // same-rate reference
      const ref = await exportOne('ref');
      const refL = lumaTrack(ref.file);
      R.ref = { frames: refL.y.length, colorTags: ref.res.colorTags, wallS: ref.wallS };
      log(cname, 'ref', JSON.stringify(R.ref));
      if (refL.y.length !== ref.res.frames) fail(`${cname} ref: ${refL.y.length} frames decoded, export says ${ref.res.frames}`);
      if (!KEEP) rmSync(ref.file, { force: true });

      for (const name of RUNS) {
        const cfg = CONFIGS[name];
        const run = (R.runs[name] = { cfg });
        try {
          const ex = await exportOne(name);
          const outL = lumaTrack(ex.file);
          if (!KEEP) rmSync(ex.file, { force: true });
          // the export's own schedule over the reference's frame times (the source frames of the range)
          const sch = buildSchedule(refL.t, { fps: +cfg.fps, timing: cfg.timing, motionBlur: cfg.blur ? 'natural' : 'off', shutterDeg: 180 });
          const d = [], bad = [];
          for (let i = 0; i < outL.y.length && i < sch.n; i++) {
            let e = 0;
            for (let j = sch.tapStart[i]; j < sch.tapStart[i] + sch.tapsPerFrame[i]; j++) e += sch.weights[j] * refL.y[sch.taps[j]];
            d.push(outL.y[i] - e);
          }
          d.forEach((x, i) => { if (Math.abs(x) > LIMIT) bad.push([i, +x.toFixed(2)]); });
          const abs = d.map(Math.abs).sort((a, b) => a - b);
          Object.assign(run, {
            frames: outL.y.length, scheduleFrames: sch.n, renderer: ex.res.renderer, retime: ex.res.retime, colorTags: ex.res.colorTags,
            decoder: ex.decoder?.variant, maxAbsDiff: +abs[abs.length - 1].toFixed(3), p99AbsDiff: +abs[Math.floor(0.99 * (abs.length - 1))].toFixed(3),
            medianDiff: +[...d].sort((a, b) => a - b)[d.length >> 1].toFixed(3), badFrames: bad.slice(0, 20), nBad: bad.length, wallS: ex.wallS,
          });
          run.pass = bad.length === 0 && outL.y.length === sch.n && ex.res.frames === sch.n && !ex.decoder?.software;
          log(cname, name, run.pass ? 'PASS' : 'FAIL', JSON.stringify({ frames: run.frames, sched: sch.n, max: run.maxAbsDiff, p99: run.p99AbsDiff, median: run.medianDiff, bad: run.badFrames, tags: run.colorTags, renderer: run.renderer, decoder: run.decoder }));
          if (!run.pass) exitCode = 1;
        } catch (e) {
          run.failure = String(e?.message ?? e);
          run.pass = false;
          fail(`${cname} ${name}: ${run.failure}`);
          if (await page.evaluate(() => window.__stillpoint.exportState === 'running')) {
            await page.evaluate(() => document.getElementById('btn-cancel')?.click());
            await page.waitForFunction(() => window.__stillpoint.exportState !== 'running', { timeout: 60000 }).catch(() => {});
          }
          await page.evaluate(() => { document.getElementById('btn-error-ok')?.click(); document.getElementById('btn-again')?.click(); });
          await sleep(400);
        }
      }
    } catch (e) {
      R.failure = String(e?.message ?? e);
      fail(`${cname}: ${R.failure}`);
    } finally {
      R.pageErrors = errors.slice(0, 10);
      await page.close().catch(() => {});
    }
  }
} catch (e) {
  result.failure = String(e?.message ?? e);
  fail(result.failure);
} finally {
  await browser.close();
  await srv.close();
  if (!KEEP) rmSync(expDir, { recursive: true, force: true });
  rmSync(dlDir, { recursive: true, force: true });
  result.vtAfter = vt();
  writeFileSync(join(OUT, 'result.json'), JSON.stringify(result, null, 1));
  log(`VT ${result.vtBefore} -> ${result.vtAfter}`, exitCode ? 'FAILED' : 'ALL PASS');
  process.exit(exitCode);
}
