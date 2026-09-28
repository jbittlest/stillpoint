// Bundles web/test/path.dev.ts with rolldown into the PATH scratch dir and runs it with node (owner: PATH agent).
//   node test/path.dev.run.mjs <fixture> [key=value ...]
//   node test/path.dev.run.mjs --smoke <fixture> dir=<fixture dir>     (test/path.smoke.ts)
import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';

const here = dirname(fileURLToPath(import.meta.url));
const out = join(homedir(), 'Library/Application Support/Stillpoint/scratch/pathweb');
mkdirSync(out, { recursive: true });
const bin = join(here, '..', 'node_modules', '.bin', 'rolldown');
const smoke = process.argv[2] === '--smoke';
const entry = smoke ? 'path.smoke.ts' : 'path.dev.ts', bundle = join(out, smoke ? 'smoke.mjs' : 'dev.mjs');
execFileSync(bin, [join(here, entry), '--format', 'esm', '--platform', 'node', '--file', bundle],
  { stdio: ['ignore', 'ignore', 'inherit'] });
execFileSync(process.execPath, ['--max-old-space-size=4096', bundle, ...process.argv.slice(smoke ? 3 : 2)], { stdio: 'inherit' });
