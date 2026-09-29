// DECODER probe: what WebCodecs reports for the DJI codec configs under different Chrome flags (headless, localhost).
//   node test/decode.probe.mjs [--disable-accelerated-video-decode] [...chrome flags]
import { createServer } from 'node:http';
import { launch } from './chrome.mjs';

const flags = process.argv.slice(2);
const srv = createServer((_, res) => { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>probe</title>'); });
await new Promise(r => srv.listen(0, '127.0.0.1', r));
const browser = await launch(flags);
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${srv.address().port}/`);
  const out = await page.evaluate(async () => {
    const res = {};
    const codecs = { 'avc1.640034': [3840, 2160], 'avc1.640033': [3840, 2160], 'hvc1.2.4.L150.90': [3840, 2880], 'hvc1.2.4.L153.B0': [3840, 2160], 'hvc1.1.6.L150.B0': [1920, 1080] };
    for (const [codec, [w, h]] of Object.entries(codecs)) {
      for (const hw of ['prefer-hardware', 'no-preference', 'prefer-software']) {
        try { res[`${codec} ${hw}`] = (await VideoDecoder.isConfigSupported({ codec, codedWidth: w, codedHeight: h, hardwareAcceleration: hw })).supported; }
        catch (e) { res[`${codec} ${hw}`] = 'throws ' + e.name + ': ' + e.message; }
      }
    }
    res.ua = navigator.userAgent;
    res.platform = navigator.platform;
    try { const ad = await navigator.gpu?.requestAdapter(); res.gpu = ad ? JSON.stringify({ vendor: ad.info?.vendor, arch: ad.info?.architecture, desc: ad.info?.description }) : null; } catch (e) { res.gpu = String(e); }
    return res;
  });
  console.log(JSON.stringify({ flags, ...out }, null, 1));
} finally { await browser.close(); srv.close(); }
