// UI screenshots of every state at desktop (1440x900) and phone (390x844) sizes, headless.
//   node test/app.shots.mjs [--clip PATH] [--build DIR]   -> ~/Library/Application Support/Stillpoint/scratch/webapp/shots/
// With --build, serves a production build (vite build output dir) instead of the dev server.
import { createServer, preview } from 'vite';
import { fileURLToPath } from 'node:url';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { launch } from './chrome.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? (args[i + 1] ?? true) : d; };
const CLIP = opt('clip', join(process.env.STILLPOINT_O3_DIR ?? join(homedir(), 'Desktop/untitled folder 4'), 'DJI_0026.MP4'));
const OUT = join(homedir(), 'Library/Application Support/Stillpoint/scratch/webapp/shots');
const BUILD = opt('build', '');
const NOEXPORT = !!opt('no-export', false);
const PREFIX = opt('prefix', '');
const ONLY = opt('only', '');
mkdirSync(OUT, { recursive: true });
const root = fileURLToPath(new URL('..', import.meta.url));

let server, base;
if (BUILD) {
  server = await preview({ root, configFile: join(root, 'vite.config.ts'), build: { outDir: BUILD }, preview: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
  base = server.resolvedUrls.local[0];
} else {
  server = await createServer({ configFile: join(root, 'vite.config.ts'), root, server: { port: 0, host: '127.0.0.1', hmr: false, watch: { ignored: ['**/*'] } }, logLevel: 'error' });
  await server.listen();
  base = server.resolvedUrls.local[0];
}
const url = base + '?sink=opfs&autodownload=0';
const browser = await launch();
const sleep = ms => new Promise(r => setTimeout(r, ms));
const shot = async (page, name, full = false) => { await page.screenshot({ path: join(OUT, PREFIX + name + '.png'), fullPage: full }); console.log('shot', PREFIX + name); };

async function session(label, viewport) {
  const page = await browser.newPage();
  page.on('pageerror', e => console.log('[pageerror]', e.message));
  page.on('console', m => { if (m.type() === 'error') console.log('[page error]', m.text()); });
  await page.setViewport(viewport);
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__stillpoint?.view === 'landing' && window.__stillpoint?.gpu, { timeout: 60000 });
  await sleep(2800); // entrance animations
  await shot(page, `${label}-1-landing`);
  const input = await page.$('#file-input');
  await input.uploadFile(CLIP);
  await page.waitForFunction(() => window.__stillpoint.clip?.frames > 0, { timeout: 120000 });
  await sleep(NOEXPORT ? 2500 : 150);
  await shot(page, `${label}-2-analyzing`);
  await page.waitForFunction(() => window.__stillpoint.planReady && window.__stillpoint.clip?.encoder?.codec, { timeout: 600000 });
  await page.evaluate(() => window.__sp_seek(Math.round(window.__stillpoint.clip.frames * 0.42)));
  await page.waitForFunction(() => window.__stillpoint.lastFrame?.pres === Math.round(window.__stillpoint.clip.frames * 0.42), { timeout: 20000 }).catch(() => {});
  await sleep(600);
  await shot(page, `${label}-3-ready`);
  if (viewport.width < 800) await shot(page, `${label}-3-ready-full`, true);
  if (NOEXPORT) { await page.close(); return; }
  // export progress, then done
  await page.click('#btn-export');
  await page.waitForFunction(() => (window.__stillpoint.progress?.done ?? 0) > 40, { timeout: 60000 });
  await shot(page, `${label}-4-exporting`, viewport.width < 800);
  await page.waitForFunction(() => window.__stillpoint.exportState === 'done', { timeout: 600000 });
  await sleep(400);
  await shot(page, `${label}-5-done`, viewport.width < 800);
  await page.close();
}

try {
  if (ONLY !== 'phone') await session('desktop', { width: 1440, height: 900, deviceScaleFactor: 2 });
  if (ONLY !== 'desktop') await session('phone', { width: 390, height: 844, deviceScaleFactor: 3, isMobile: true, hasTouch: true });
} finally {
  await browser.close();
  await (server.close?.() ?? server.httpServer?.close());
}
