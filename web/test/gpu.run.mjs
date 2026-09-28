// GPU module browser tests (owner: GPU agent). Headless Chrome only — never opens a window.
//
//   node test/gpu.run.mjs probe                 capability probe (WebGPU adapter, WebCodecs HEVC/H.264, external textures)
//   node test/gpu.run.mjs golden [fixture...]   decode + warp the fixture frames, dump readbacks for test/gpu.golden.py
//   node test/gpu.run.mjs bench                 4K warp throughput
//
// Clips are served by the dev server from $STILLPOINT_CLIPS_DIR (a scratch dir of symlinks — never copies).
// Binary results are POSTed by the page to /__gpu_out/<name> and land in $GPU_OUT_DIR.
import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';
import { mkdirSync, writeFileSync, symlinkSync, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import os from 'node:os';
import { launch } from './chrome.mjs';

const SCRATCH = process.env.GPU_SCRATCH || join(os.homedir(), 'Library/Application Support/Stillpoint/scratch/gpu');
const OUT = process.env.GPU_OUT_DIR || join(SCRATCH, 'out');
const CLIPS = process.env.STILLPOINT_CLIPS_DIR || join(SCRATCH, 'clips');
mkdirSync(OUT, { recursive: true });
mkdirSync(CLIPS, { recursive: true });
const CLIP_SRC = {
  'DJI_0026.MP4': [join(process.env.STILLPOINT_O3_DIR ?? join(os.homedir(), 'Desktop/untitled folder 4'), 'DJI_0026.MP4')],
  'DJI_20260926153751_0005_D.MP4': [join(os.homedir(), 'Desktop/DJI_20260926153751_0005_D.MP4'),
    '/Volumes/Untitled/DCIM/DJI_001/DJI_20260926153751_0005_D.MP4'],
};
for (const [name, cands] of Object.entries(CLIP_SRC)) {
  const link = join(CLIPS, name);
  if (existsSync(link)) continue;
  const src = cands.find(p => existsSync(p));
  if (src) symlinkSync(src, link);
}
process.env.STILLPOINT_CLIPS_DIR = CLIPS;

const mode = process.argv[2] || 'probe';
const args = process.argv.slice(3);
const root = fileURLToPath(new URL('..', import.meta.url));

const outPlugin = {
  name: 'gpu-test-out',
  configureServer(server) {
    server.middlewares.use('/__gpu_out/', (req, res) => {
      const name = decodeURIComponent((req.url || '').split('?')[0].replace(/^\/+/, ''));
      if (!/^[\w.\-]+$/.test(name)) { res.statusCode = 400; return res.end(); }
      const chunks = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => { writeFileSync(join(OUT, name), Buffer.concat(chunks)); res.end('ok'); });
    });
  },
};

const server = await createServer({
  configFile: join(root, 'vite.config.ts'), root, plugins: [outPlugin],
  server: { port: 0, host: '127.0.0.1' }, logLevel: 'error',
});
await server.listen();
const url = server.resolvedUrls.local[0];
const browser = await launch();
let code = 0;
try {
  const page = await browser.newPage();
  page.on('console', m => console.log('[page]', m.text()));
  page.on('pageerror', e => console.log('[pageerror]', e.message));
  const q = new URLSearchParams({ mode, args: args.join(',') });
  await page.goto(`${url}test/gpu.harness.html?${q}`);
  await page.waitForFunction(() => window.__done, { timeout: 20 * 60 * 1000, polling: 250 });
  const result = await page.evaluate(() => window.__result);
  writeFileSync(join(OUT, `result_${mode}.json`), JSON.stringify(result, null, 1));
  console.log(JSON.stringify(result, null, 1));
  if (result && result.error) code = 1;
} catch (e) {
  console.error(e);
  code = 1;
} finally {
  await browser.close();
  await server.close();
}
process.exit(code);
