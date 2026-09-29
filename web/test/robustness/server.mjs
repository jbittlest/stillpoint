// Robustness harness: serve the app under test on 127.0.0.1.
//   target 'docs' (default): the built GitHub Pages copy in ../docs/app — byte-identical to what
//                            https://jbittlest.github.io/stillpoint/app/ serves — at /stillpoint/app/
//   target 'dev'           : the Vite dev server on src/ (to test a fix before it is built)
//   target 'build'         : a fresh production build of src/ (vite build to scratch), served like 'docs'
//   target 'http(s)://…'   : an already-deployed copy (e.g. the live site); nothing is served
import { createServer as httpServer } from 'node:http';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const WEB = fileURLToPath(new URL('../..', import.meta.url));
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg', '.png': 'image/png', '.json': 'application/json', '.wasm': 'application/wasm', '.map': 'application/json' };

/** Static server: GET /stillpoint/<path> -> <docsRoot>/<path> (docsRoot contains app/). */
function staticServer(docsRoot) {
  const srv = httpServer((req, res) => {
    const u = new URL(req.url, 'http://x');
    let p = decodeURIComponent(u.pathname);
    if (!p.startsWith('/stillpoint/')) { res.statusCode = 404; return res.end(); }
    p = normalize(p.slice('/stillpoint/'.length));
    if (p.includes('..')) { res.statusCode = 400; return res.end(); }
    let f = join(docsRoot, p);
    if (existsSync(f) && statSync(f).isDirectory()) f = join(f, 'index.html');
    if (!existsSync(f)) { res.statusCode = 404; return res.end(); }
    res.setHeader('Content-Type', MIME[extname(f)] ?? 'application/octet-stream');
    res.setHeader('Cache-Control', 'no-store');
    createReadStream(f).pipe(res);
  });
  return new Promise(r => srv.listen(0, '127.0.0.1', () => r({ base: `http://127.0.0.1:${srv.address().port}/stillpoint/app/`, close: () => new Promise(k => srv.close(k)) })));
}

export async function serve(target = 'docs', { buildDir } = {}) {
  if (/^https?:\/\//.test(target)) return { base: target.replace(/\/?$/, '/'), close: async () => {} , target };
  if (target === 'docs') return { ...(await staticServer(resolve(WEB, '../docs'))), target };
  const vite = await import('vite');
  if (target === 'build') {
    const outDir = join(buildDir, 'stillpoint/app');
    await vite.build({ root: WEB, configFile: join(WEB, 'vite.config.ts'), base: '/stillpoint/app/', mode: 'pages', logLevel: 'error', build: { outDir, emptyOutDir: true } });
    return { ...(await staticServer(join(buildDir, 'stillpoint'))), target };
  }
  if (target === 'dev') {
    // hmr/watch off: other agents may edit src/ while a matrix runs
    const server = await vite.createServer({ configFile: join(WEB, 'vite.config.ts'), root: WEB, server: { port: 0, host: '127.0.0.1', hmr: false, watch: { ignored: ['**/*'] } }, logLevel: 'error' });
    await server.listen();
    return { base: server.resolvedUrls.local[0], close: () => server.close(), target };
  }
  throw new Error('unknown target ' + target);
}
