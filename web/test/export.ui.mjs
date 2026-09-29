// EXPORT PANEL rules end-to-end (headless Chrome, never a visible window), typed and clicked like a user:
//   1. custom width / frame rate: an out-of-range value is clamped to the range when the field is committed (Tab),
//      not replaced by a number typed on the way (9000 px -> 7680, not 900; 300 fps -> 240, not 30); the error state
//      shows while typing and is gone after the commit;
//   2. 60 fps from a 59.94 source: Slow motion is disabled (with a hint) even when it was the remembered choice, and
//      the export is real time with the sound kept (OA4 clip: AAC) — not "1x faster" without sound;
//   3. a silent clip (O3 DJI_0026) exported in slow motion: the done card says the source has no sound, not
//      "no sound (slow motion)".
//
//   node test/export.ui.mjs [--target dev|build|docs]
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { launch } from './chrome.mjs';
import { serve } from './robustness/server.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? (args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : true) : d; };
const TARGET = opt('target', 'dev');
const OUT = opt('out', join(homedir(), 'Library/Application Support/Stillpoint/scratch/fixer/export-ui'));
const O3 = join(process.env.STILLPOINT_O3_DIR ?? join(homedir(), 'Desktop/untitled folder 4'), 'DJI_0026.MP4');
const OA4 = process.env.CLIP_OA4 ?? join(homedir(), 'Desktop/DJI_20260927091931_0012_D.MP4');
const dlDir = join(OUT, 'downloads');
mkdirSync(dlDir, { recursive: true });
const vt = () => { try { return +execFileSync('sh', ['-c', 'pgrep -x VTDecoderXPCService | wc -l']).toString().trim(); } catch { return -1; } };
const log = (...a) => console.log(`[ui ${(performance.now() / 1000).toFixed(1)}s]`, ...a);
const sleep = ms => new Promise(r => setTimeout(r, ms));

const result = { target: TARGET, vtBefore: vt(), checks: {} };
const failures = [];
const check = (name, ok, detail) => { result.checks[name] = { ok: !!ok, detail }; if (!ok) failures.push(`${name}: ${JSON.stringify(detail)}`); log(ok ? 'ok  ' : 'FAIL', name, JSON.stringify(detail)); };

const srv = await serve(TARGET, { buildDir: join(OUT, 'build') });
const browser = await launch();
try {
  const cdp = await browser.target().createCDPSession();
  await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: dlDir, eventsEnabled: true });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });
  await page.goto(`${srv.base}?sink=opfs`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__stillpoint?.view === 'landing' && window.__stillpoint?.gpu, { timeout: 60000 });
  await page.evaluate(() => localStorage.removeItem('stillpoint.export.v1'));
  const tap = async sel => { await page.$eval(sel, el => el.scrollIntoView({ block: 'center' })); await sleep(60); await page.click(sel); };
  const open = async (clip, window) => {
    await (await page.$('#file-input')).uploadFile(clip);
    const name = clip.split('/').pop();
    await page.waitForFunction(n => window.__stillpoint.clip?.name === n && window.__stillpoint.clip.frames > 0, { timeout: 120000 }, name);
    await page.waitForFunction(n => window.__stillpoint.clip?.name === n && window.__stillpoint.planReady && window.__stillpoint.clip.encoder, { timeout: 900000 }, name);
    if (window) {
      const { frames, fps } = await page.evaluate(() => ({ frames: window.__stillpoint.clip.frames, fps: window.__stillpoint.clip.fps }));
      await page.evaluate((a, b) => window.__sp_range(a, b), Math.round(window[0] * fps), Math.min(frames - 1, Math.round((window[0] + window[1]) * fps) - 1));
    }
  };
  const ready = () => page.waitForFunction(() => { const d = window.__stillpoint; return d.planReady && d.exportPlan && d.clip?.encoder && d.clip.encoder.width === d.exportPlan.width && d.clip.encoder.height === d.exportPlan.height && Math.abs(d.clip.encoder.fps - d.exportPlan.fps) < 1e-6 && !document.getElementById('btn-export').disabled; }, { timeout: 120000 });
  const state = () => page.evaluate(() => ({
    w: document.getElementById('in-custom-w').value, wBad: document.getElementById('in-custom-w').classList.contains('is-bad'),
    f: document.getElementById('in-custom-fps').value, fBad: document.getElementById('in-custom-fps').classList.contains('is-bad'),
    dims: document.getElementById('out-dims').textContent, customH: document.getElementById('custom-h').textContent,
    fpsNote: document.getElementById('fps-note').textContent, summary: document.getElementById('out-summary').textContent,
    smDisabled: document.querySelector('#seg-timing button[data-timing="slowmo"]').disabled,
    smLabel: document.getElementById('timing-sm').textContent, smTitle: document.querySelector('#seg-timing button[data-timing="slowmo"]').title,
    checked: [...document.querySelectorAll('#seg-timing button')].filter(b => b.getAttribute('aria-checked') === 'true').map(b => b.dataset.timing),
    facts: [...document.querySelectorAll('#export-facts dt')].map(dt => `${dt.textContent}: ${dt.nextElementSibling.textContent}`),
    plan: window.__stillpoint.exportPlan,
  }));
  /** select the field's text, type `v` key by key (state read before the commit), Tab out (commit) */
  const typeInto = async (sel, v) => {
    await tap(sel);
    await page.$eval(sel, el => { el.focus(); el.select(); });
    await page.keyboard.press('Backspace');
    await page.type(sel, String(v), { delay: 15 });
    await sleep(150);
    const typing = await state();
    await page.keyboard.press('Tab');
    await sleep(300);
    return { typing, committed: await state() };
  };
  const exportNow = async () => {
    for (const f of readdirSync(dlDir)) rmSync(join(dlDir, f), { force: true });
    await ready();
    await page.evaluate(() => { const d = window.__stillpoint; d.exportState = 'idle'; d.result = undefined; d.error = undefined; });
    await tap('#btn-export');
    await page.waitForFunction(() => ['done', 'error'].includes(window.__stillpoint.exportState), { timeout: 600000 });
    const r = await page.evaluate(() => ({ st: window.__stillpoint.exportState, err: window.__stillpoint.error, r: window.__stillpoint.result, sub: document.getElementById('done-sub').textContent, settings: window.__stillpoint.exportSettings?.output }));
    await page.evaluate(() => document.getElementById('btn-again')?.click());
    await sleep(300);
    return r;
  };

  // ── O3 (no audio track)
  await open(O3);
  await tap('#seg-aspect button[data-aspect="16:9"]');
  await tap('#seg-size button[data-size="custom"]');
  let s = await typeInto('#in-custom-w', 9000);
  check('width 9000: error while typing', s.typing.wBad && /160–7680/.test(s.typing.customH), s.typing);
  check('width 9000: commit clamps to 7680 and clears the error', s.committed.w === '7680' && !s.committed.wBad && /^7680 ×/.test(s.committed.dims), s.committed);
  s = await typeInto('#in-custom-w', 50);
  check('width 50: commit clamps to 160, no error', s.committed.w === '160' && !s.committed.wBad && /^160 ×/.test(s.committed.dims), s.committed);
  s = await typeInto('#in-custom-w', 1601);
  check('width 1601: even 1602, no error', s.committed.w === '1602' && !s.committed.wBad && /^1602 ×/.test(s.committed.dims), s.committed);
  await page.select('#sel-fps', 'custom');
  s = await typeInto('#in-custom-fps', 300);
  check('fps 300: error while typing', s.typing.fBad && /1–240/.test(s.typing.fpsNote), s.typing);
  check('fps 300: commit clamps to 240 and clears the error', s.committed.f === '240' && !s.committed.fBad && /240 fps/.test(s.committed.summary), s.committed);
  s = await typeInto('#in-custom-fps', 0);
  check('fps 0: commit clamps to 1, no error', s.committed.f === '1' && !s.committed.fBad, s.committed);
  // slow motion availability follows the rate
  await tap('#seg-size button[data-size="720p"]');
  await page.select('#sel-fps', '24');
  await tap('#seg-timing button[data-timing="slowmo"]');
  s = await state();
  check('24 from 59.94: slow motion offered', !s.smDisabled && s.checked[0] === 'slowmo' && /2\.5× slower/.test(s.smLabel), s);
  // silent clip, slow motion: the done card must not blame slow motion for the missing sound
  const e1 = await exportNow();
  check('silent clip slow motion: export done', e1.st === 'done' && e1.r?.timeMode === 'slowmo', { st: e1.st, err: e1.err, timeMode: e1.r?.timeMode });
  check('silent clip: done card says the source has no sound', /no sound in the source clip/.test(e1.sub) && !/slow motion/.test(e1.sub) && e1.r?.sourceAudio === false && e1.r?.audioDropped === false, e1.sub);
  await page.select('#sel-fps', '60');
  s = await state();
  check('60 from 59.94: slow motion disabled with a hint, real time checked', s.smDisabled && s.checked[0] === 'realtime' && s.smLabel === 'needs a lower rate' && /below the source/.test(s.smTitle), s);
  await page.select('#sel-fps', '30');
  s = await state();
  check('back to 30: the remembered slow motion comes back', !s.smDisabled && s.checked[0] === 'slowmo', s);

  // ── OA4 (AAC audio): 60 fps from 59.94 with Slow motion remembered -> real time, sound kept
  await page.select('#sel-fps', '60');
  await open(OA4, [0, 3]);
  await tap('#seg-size button[data-size="720p"]');
  await tap('#seg-aspect button[data-aspect="16:9"]');
  await page.select('#sel-fps', '60');
  s = await state();
  check('OA4 60 from 59.94: real time, sound kept in the plan', s.smDisabled && s.checked[0] === 'realtime' && s.plan?.audio === true && s.facts.some(f => /^Audio: Original/.test(f)) && !s.facts.some(f => /slow|sped/.test(f)), s);
  const e2 = await exportNow();
  const r2 = e2.r ?? {};
  check('OA4 60 from 59.94: exported real time with sound', e2.st === 'done' && e2.settings?.timing === 'realtime' && r2.timeMode === 'realtime' && r2.audioPackets > 0 && !r2.audioDropped && Math.abs(r2.frames - 180) <= 1 && !/no sound/.test(e2.sub),
    { st: e2.st, err: e2.err, timing: e2.settings?.timing, timeMode: r2.timeMode, audioPackets: r2.audioPackets, audioDropped: r2.audioDropped, frames: r2.frames, sub: e2.sub });
  check('no page errors', errors.length === 0, errors.slice(0, 5));
} catch (e) {
  failures.push(String(e?.stack ?? e));
  log('FAILED', e);
} finally {
  await browser.close();
  await srv.close();
  rmSync(dlDir, { recursive: true, force: true });
  result.vtAfter = vt();
  result.failures = failures;
  writeFileSync(join(OUT, 'result.json'), JSON.stringify(result, null, 1));
  log(`VT ${result.vtBefore} -> ${result.vtAfter}`, failures.length ? `${failures.length} FAILED` : 'ALL PASS');
  process.exit(failures.length ? 1 : 0);
}
