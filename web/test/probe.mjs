import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';
import { launch } from './chrome.mjs';
const server = await createServer({ configFile: fileURLToPath(new URL('../vite.config.ts', import.meta.url)), root: fileURLToPath(new URL('..', import.meta.url)), server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const url = server.resolvedUrls.local[0];
const browser = await launch();
try {
  const page = await browser.newPage();
  page.on('console', m => console.log('[page]', m.text()));
  await page.goto(url + 'test/probe.html');
  await page.waitForFunction(() => window.__probe, { timeout: 60000 });
  console.log(JSON.stringify(await page.evaluate(() => window.__probe), null, 1));
} finally { await browser.close(); await server.close(); }
