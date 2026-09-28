// Clip-state check: opens a clip, waits for telemetry (or its error) and the first preview frame, and reports the gyro
// badge, notes and whether export is allowed. Used for the error paths: a file without DJI telemetry (e.g. a Gyroflow
// render) and an in-camera-EIS clip must open, preview, explain, and block export; a 60 Hz clip shows "limited".
//   node test/app.nogyro.mjs <clip> <screenshot.png>
import { createServer } from 'vite';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('..', import.meta.url)), clip = process.argv[2], out = process.argv[3];
import { launch } from './chrome.mjs';
const server = await createServer({ configFile: join(root, 'vite.config.ts'), root, server: { port: 0, host: '127.0.0.1', hmr: false, watch: { ignored: ['**/*'] } }, logLevel: 'error' });
await server.listen();
const browser = await launch();
try {
  const page = await browser.newPage();
  await page.goto(server.resolvedUrls.local[0], { waitUntil: 'load' });
  await page.waitForFunction(() => window.__stillpoint?.gpu, { timeout: 60000 });
  await (await page.$('#file-input')).uploadFile(clip);
  await page.waitForFunction(() => { const d = window.__stillpoint; return (d.clip?.telemetry || d.events.some(e => /analysis error|fail/.test(e))) && d.lastFrame; }, { timeout: 180000 });
  await new Promise(r => setTimeout(r, 1500));
  const st = await page.evaluate(() => ({ ev: window.__stillpoint.events, err: window.__stillpoint.error, badge: document.getElementById('gyro-badge').textContent, notes: document.getElementById('warnings').textContent, btn: document.getElementById('btn-export').disabled, frame: window.__stillpoint.lastFrame }));
  // a usable clip (telemetry, no EIS) must go on to a plan and an enabled Stabilize button
  const tel = await page.evaluate(() => window.__stillpoint.clip?.telemetry);
  if (tel && !tel.eisBaked) {
    await page.waitForFunction(() => window.__stillpoint.planReady && window.__stillpoint.clip?.encoder && !document.getElementById('btn-export').disabled, { timeout: 180000 }).catch(() => {});
    Object.assign(st, await page.evaluate(() => ({ planReady: window.__stillpoint.planReady, plan: window.__stillpoint.clip?.plan, btnAfterPlan: document.getElementById('btn-export').disabled, frame: window.__stillpoint.lastFrame })));
  }
  console.log(JSON.stringify(st, null, 1));
  await page.screenshot({ path: out });
} finally { await browser.close(); await server.close(); }
