// TELEMETRY: Blob read-strategy benchmark in headless Chrome (djmd samples of one clip; read-only).
//   node test/telemetry.bench.mjs [clip.mp4]
import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';
import { launch } from './chrome.mjs';
const WEB = fileURLToPath(new URL('..', import.meta.url)).replace(/\/$/, '');
const server = await createServer({ configFile: WEB + '/vite.config.ts', root: WEB, server: { port: 0, host: '127.0.0.1', hmr: false, watch: { ignored: ['**/*'] } }, logLevel: 'error' });
await server.listen();
const url = server.resolvedUrls.local[0];
const browser = await launch();
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(url + 'test/telemetry.html');
  await page.waitForFunction(() => window.__ready, { timeout: 60000 });
  const input = await page.$('#f');
  await input.uploadFile(process.argv[2] ?? `${process.env.HOME}/Desktop/DJI_20260927091931_0012_D.MP4`);
  const r = await page.evaluate(async () => {
    const m = await import('/src/mp4.ts');
    const d = await import('/src/dji.ts');
    const f = document.getElementById('f').files[0];
    const info = await m.openMp4(f);
    const t = m.findTrack(info, { fourcc: 'djmd' });
    const out = {};
    for (const c of [1, 4, 8, 16, 32]) {
      const t0 = performance.now();
      const bufs = await m.readRanges(f, t.offsets, t.sizes, { concurrency: c });
      out['read_conc' + c] = Math.round(performance.now() - t0);
      if (c === 8) {
        const t1 = performance.now();
        d.parseDjmd(bufs);
        out.parseDjmd = Math.round(performance.now() - t1);
      }
    }
    // worker + FileReaderSync
    const src = `onmessage = (e) => { const { f, offs, sizes } = e.data; const r = new FileReaderSync(); const t0 = performance.now(); let n = 0;
      for (let i = 0; i < offs.length; i++) { const b = r.readAsArrayBuffer(f.slice(offs[i], offs[i] + sizes[i])); n += b.byteLength; }
      postMessage({ ms: performance.now() - t0, n }); };`;
    const w = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
    const res = await new Promise((ok) => { w.onmessage = (e) => ok(e.data); w.postMessage({ f, offs: Array.from(t.offsets), sizes: Array.from(t.sizes) }); });
    out.frs_worker = Math.round(res.ms);
    // worker + async arrayBuffer conc 8
    const src2 = `onmessage = async (e) => { const { f, offs, sizes, c } = e.data; const t0 = performance.now(); let next = 0;
      const wk = async () => { while (true) { const i = next++; if (i >= offs.length) return; await f.slice(offs[i], offs[i] + sizes[i]).arrayBuffer(); } };
      await Promise.all(Array.from({ length: c }, wk)); postMessage({ ms: performance.now() - t0 }); };`;
    for (const c of [4, 8, 16]) {
      const w2 = new Worker(URL.createObjectURL(new Blob([src2], { type: 'text/javascript' })));
      const res2 = await new Promise((ok) => { w2.onmessage = (e) => ok(e.data); w2.postMessage({ f, offs: Array.from(t.offsets), sizes: Array.from(t.sizes), c }); });
      out['async_worker_conc' + c] = Math.round(res2.ms);
      w2.terminate();
    }
    return out;
  });
  console.log(JSON.stringify(r));
} finally { await browser.close(); await server.close(); }
