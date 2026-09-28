// TELEMETRY in headless Chrome: parse real clips from <input type=file> Files (the app's path) and compare against the
// Python golden fixtures; report WebGPU / WebCodecs support and whether each clip's own decoder config decodes.
//   cd stillpoint/web && node test/telemetry.browser.mjs [clip paths...]
// Never opens a visible window (test/chrome.mjs launches --headless=new). Files are read in place.
import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { join, basename } from 'node:path';
import { homedir } from 'node:os';
import { launch } from './chrome.mjs';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const FIX = join(HERE, 'fixtures', 'telemetry');
const HOME = homedir();
const SEARCH = [join(HOME, 'Desktop'), join(HOME, 'Desktop', 'untitled folder 4'), '/Volumes/Untitled/DCIM/DJI_001'];
const fixtures = Object.fromEntries(readdirSync(FIX).filter((f) => f.endsWith('.json.gz')).map((f) => {
  const fx = JSON.parse(gunzipSync(readFileSync(join(FIX, f))).toString('utf-8'));
  return [fx.file, fx];
}));
const want = process.argv.slice(2).length ? process.argv.slice(2)
  : ['DJI_0026.MP4', 'DJI_20260926153751_0005_D.MP4', 'DJI_20260926155953_0007_D.MP4', 'DJI_20260927091931_0012_D.MP4',
     'DJI_20260926152149_0002_D.MP4', 'DJI_20260925151130_0003_D_joined.MP4'];
const clips = want.map((n) => (n.includes('/') ? n : SEARCH.map((d) => join(d, n)).find(existsSync))).filter(Boolean);
const b64 = (s) => { const b = Buffer.from(s, 'base64'); return Array.from(new Float64Array(b.buffer, b.byteOffset, b.length / 8)); };
const maxAbs = (a, b) => a.reduce((m, x, i) => Math.max(m, Number.isNaN(x) && Number.isNaN(b[i]) ? 0 : Math.abs(x - b[i])), 0);

const server = await createServer({ configFile: join(HERE, '..', 'vite.config.ts'), root: join(HERE, '..'),
  server: { port: 0, host: '127.0.0.1', hmr: false, watch: { ignored: ['**/*'] } }, logLevel: 'error' });
await server.listen();
const url = server.resolvedUrls.local[0];
const browser = await launch();
const report = { clips: [] };
let failed = 0;
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error') console.log('[page]', m.text()); });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(url + 'test/telemetry.html');
  await page.waitForFunction(() => window.__ready, { timeout: 60000 });
  report.env = await page.evaluate(() => window.__env());
  const input = await page.$('#f');
  await input.uploadFile(...clips);
  for (let i = 0; i < clips.length; i++) {
    const fx = fixtures[basename(clips[i])];
    const frameIdx = fx ? fx.strided.frame_idx : [0];
    const imuIdx = fx ? fx.strided.imu_idx : [0];
    const r = await page.evaluate((i, a, b) => window.__telemetry(i, a, b, undefined), i, frameIdx, imuIdx);
    const wr = await page.evaluate((i, a, b) => window.__worker(i, a, b), i, frameIdx, imuIdx);
    if (wr.error) { console.log('worker error', wr.error); failed++; }
    const row = { file: r.name, sizeGiB: +(r.size / 2 ** 30).toFixed(2), frames: r.frames, imu: r.imu, openMs: Math.round(r.openMs),
      parseMs: Math.round(r.parseMs), workerTotalMs: Math.round(wr.ms), camera: r.camera };
    if (fx) {
      row.maxErr = { frameT: maxAbs(r.frameT, b64(fx.strided.frame_t)), framePts: maxAbs(r.framePts, b64(fx.strided.frame_pts)),
        imuT: maxAbs(r.imuT, b64(fx.strided.imu_t)), imuQ: maxAbs(r.imuQ, b64(fx.strided.imu_q)) };
      const same = r.camera === fx.camera && r.fps === fx.fps && r.hasHighrate === fx.has_highrate && r.eisBaked === fx.eis_baked &&
        JSON.stringify(r.lens) === JSON.stringify(fx.lens) && JSON.stringify(r.segments) === JSON.stringify(fx.segments) &&
        JSON.stringify(r.warnings) === JSON.stringify(fx.warnings) && r.frames === fx.n_frames && r.imu === fx.n_imu;
      row.flagsIdentical = same;
      const ok = same && row.maxErr.frameT <= 1e-6 && row.maxErr.imuT <= 1e-6 && row.maxErr.imuQ <= 1e-6 && row.maxErr.framePts <= 1e-6;
      const wok = !wr.error && wr.frames === fx.n_frames && wr.imu === fx.n_imu && maxAbs(wr.frameT, b64(fx.strided.frame_t)) <= 1e-6 &&
        maxAbs(wr.imuT, b64(fx.strided.imu_t)) <= 1e-6 && maxAbs(wr.imuQ, b64(fx.strided.imu_q)) <= 1e-6;
      row.workerOk = wok;
      if (!wok) failed++;
      row.ok = ok;
      if (!ok) failed++;
    }
    row.decode = await page.evaluate((i) => window.__decode(i), i);
    if (!row.decode.frames?.length) failed++;
    report.clips.push(row);
    console.log(JSON.stringify(row));
  }
} finally {
  await browser.close();
  await server.close();
}
console.log(JSON.stringify({ env: report.env }, null, 1));
if (failed) { console.error(`FAILED: ${failed}`); process.exit(1); }
