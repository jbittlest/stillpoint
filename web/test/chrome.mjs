// Shared headless-Chrome launcher for Stillpoint web tests (APP agent).
// Never opens a visible window: always --headless=new.
import puppeteer from 'puppeteer-core';

export const CHROME = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

export async function launch(extraArgs = []) {
  return puppeteer.launch({
    executablePath: CHROME,
    headless: true, // puppeteer >= 22: new headless
    args: [
      '--headless=new',
      '--enable-unsafe-webgpu',
      '--enable-features=WebGPU,Vulkan,SharedArrayBuffer',
      '--use-angle=metal',
      '--ignore-gpu-blocklist',
      '--enable-gpu',
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows',
      ...extraArgs,
    ],
    defaultViewport: { width: 1440, height: 900, deviceScaleFactor: 2 },
    protocolTimeout: 30 * 60 * 1000,
  });
}
