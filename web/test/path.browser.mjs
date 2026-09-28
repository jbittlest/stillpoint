// PATH browser benchmark runner (owner: PATH agent): Vite dev server on localhost + HEADLESS Chrome.
//   node test/path.browser.mjs [fixture-url-prefix ...]
// Default fixtures: the two repo fixtures. A big scratch fixture can be exposed through the dev server's /__clips/
// route: STILLPOINT_CLIPS_DIR=<dir with the .json/.bin.gz> node test/path.browser.mjs /__clips/oa4_0012_full
import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launch } from './chrome.mjs';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const port = 5000 + Math.floor(Math.random() * 3000);
const fixtures = process.argv.slice(2);
if (!fixtures.length) fixtures.push('/test/fixtures/path/o3_0034', '/test/fixtures/path/oa4_0012_w100');

const vite = spawn(join(web, 'node_modules', '.bin', 'vite'), ['--port', String(port), '--strictPort', '--host', '127.0.0.1'],
  { cwd: web, env: { ...process.env, VITE_CONFIG_NATIVE_IGNORE_WARNING: 'true' }, stdio: ['ignore', 'pipe', 'pipe'] });
await new Promise((resolve, reject) => {
  const to = setTimeout(() => reject(new Error('vite did not start')), 60000);
  vite.stdout.on('data', (d) => { if (String(d).includes('Local')) { clearTimeout(to); resolve(); } });
  vite.on('exit', (c) => reject(new Error('vite exited ' + c)));
});
const browser = await launch();
try {
  for (const fx of fixtures) {
    const page = await browser.newPage();
    page.on('console', (m) => { if (m.type() === 'error') console.error('[page]', m.text()); });
    await page.goto(`http://127.0.0.1:${port}/test/path.browser.html?fixture=${encodeURIComponent(fx)}`);
    await page.waitForFunction('window.__pathResult', { timeout: 20 * 60 * 1000, polling: 500 });
    const r = await page.evaluate('window.__pathResult');
    console.log(JSON.stringify({ fixture: fx, ...r }));
    await page.close();
  }
} finally {
  await browser.close();
  vite.kill('SIGTERM');
}
