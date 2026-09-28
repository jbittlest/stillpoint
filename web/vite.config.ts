import { defineConfig, type Plugin } from 'vite';
import { realpathSync, statSync, createReadStream, existsSync } from 'node:fs';
import { resolve, join } from 'node:path';

// Stillpoint web — Vite config.
//  * `vite` (dev) serves at '/', `vite build` targets GitHub Pages at '/stillpoint/app/'
//    (override with STILLPOINT_BASE, e.g. STILLPOINT_BASE=/ for a local static server).
//  * node_modules is a symlink to ~/Library/Application Support/Stillpoint/web/node_modules
//    (keeps ~100 MB of deps off the iCloud-synced Desktop), so its real path is allowed below.
//  * Dev only: /__clips/<name> serves files from $STILLPOINT_CLIPS_DIR (a scratch dir of symlinks
//    to test footage) with HTTP Range support, for tests that want URL access to a clip.

const root = import.meta.dirname;
const nm = (() => { try { return realpathSync(resolve(root, 'node_modules')); } catch { return resolve(root, 'node_modules'); } })();

function clipsPlugin(): Plugin {
  const dir = process.env.STILLPOINT_CLIPS_DIR;
  return {
    name: 'stillpoint-dev-clips',
    apply: 'serve',
    configureServer(server) {
      if (!dir) return;
      server.middlewares.use('/__clips/', (req, res, next) => {
        const name = decodeURIComponent((req.url || '').split('?')[0].replace(/^\/+/, ''));
        if (!name || name.includes('..') || name.includes('/')) return next();
        const p = join(dir, name);
        if (!existsSync(p)) { res.statusCode = 404; return res.end(); }
        const size = statSync(p).size;
        const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range || '');
        res.setHeader('Accept-Ranges', 'bytes');
        res.setHeader('Content-Type', 'video/mp4');
        if (range) {
          const start = range[1] ? parseInt(range[1], 10) : Math.max(0, size - parseInt(range[2], 10));
          const end = range[1] && range[2] ? Math.min(size - 1, parseInt(range[2], 10)) : size - 1;
          res.statusCode = 206;
          res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
          res.setHeader('Content-Length', String(end - start + 1));
          createReadStream(p, { start, end }).pipe(res);
        } else {
          res.setHeader('Content-Length', String(size));
          createReadStream(p).pipe(res);
        }
      });
    },
  };
}

export default defineConfig(({ command, mode }) => ({
  base: process.env.STILLPOINT_BASE ?? (command === 'build' ? '/stillpoint/app/' : '/'),
  plugins: [clipsPlugin()],
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    // no source maps in the GitHub Pages build (docs/app): smaller deploy, no local dependency paths
    sourcemap: mode !== 'pages',
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 1500,
  },
  server: {
    fs: { allow: [root, nm] },
  },
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
  },
} as any));
