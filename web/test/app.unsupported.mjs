// Unsupported-browser landing: simulates Firefox (Firefox UA, no WebGPU, no HEVC decode) and old Safari (no WebGPU) in
// headless Chrome and screenshots the friendly message; asserts the drop zone is disabled.
//   node test/app.unsupported.mjs   -> ~/Library/Application Support/Stillpoint/scratch/webapp/shots/unsupported-*.png
import { createServer } from 'vite';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { launch } from './chrome.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const OUT = join(homedir(), 'Library/Application Support/Stillpoint/scratch/webapp/shots');
const server = await createServer({ configFile: join(root, 'vite.config.ts'), root, server: { port: 0, host: '127.0.0.1', hmr: false, watch: { ignored: ['**/*'] } }, logLevel: 'error' });
await server.listen();
const browser = await launch();
const cases = {
  firefox: { ua: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.6; rv:140.0) Gecko/20100101 Firefox/140.0', noHevc: true },
  safari: { ua: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15', noHevc: false },
};
let fail = 0;
try {
  for (const [name, c] of Object.entries(cases)) {
    for (const [vp, size] of [['desktop', { width: 1440, height: 900, deviceScaleFactor: 2 }], ['phone', { width: 390, height: 844, deviceScaleFactor: 3, isMobile: true, hasTouch: true }]]) {
      const page = await browser.newPage();
      await page.setUserAgent(c.ua);
      await page.setViewport(size);
      await page.evaluateOnNewDocument(noHevc => {
        Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true });
        if (noHevc && self.VideoDecoder) {
          const orig = VideoDecoder.isConfigSupported.bind(VideoDecoder);
          VideoDecoder.isConfigSupported = cfg => (/^(hvc1|hev1)/.test(cfg.codec) ? Promise.resolve({ supported: false, config: cfg }) : orig(cfg));
        }
      }, c.noHevc);
      await page.goto(server.resolvedUrls.local[0], { waitUntil: 'load' });
      await page.waitForFunction(() => window.__stillpoint?.caps, { timeout: 30000 });
      await new Promise(r => setTimeout(r, 2500));
      const st = await page.evaluate(() => ({
        shown: !document.getElementById('unsupported').hidden,
        title: document.getElementById('unsupported-title').textContent,
        body: document.getElementById('unsupported-body').textContent,
        dropDisabled: document.getElementById('drop').getAttribute('aria-disabled') === 'true' && document.getElementById('file-input').disabled,
      }));
      console.log(name, vp, JSON.stringify(st));
      if (!st.shown || !st.dropDisabled) fail = 1;
      await page.screenshot({ path: join(OUT, `unsupported-${name}-${vp}.png`) });
      await page.close();
    }
  }
} finally {
  await browser.close();
  await server.close();
}
process.exit(fail);
