// Bundles test/plan.dump.ts with rolldown into scratch and runs it with node (see plan.dump.ts).
import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';

const here = dirname(fileURLToPath(import.meta.url));
const out = join(homedir(), 'Library/Application Support/Stillpoint/scratch/integ');
mkdirSync(out, { recursive: true });
const bundle = join(out, 'plan.dump.mjs');
execFileSync(join(here, '..', 'node_modules', '.bin', 'rolldown'), [join(here, 'plan.dump.ts'), '--format', 'esm', '--platform', 'node', '--file', bundle],
  { stdio: ['ignore', 'ignore', 'inherit'] });
execFileSync(process.execPath, ['--max-old-space-size=4096', bundle, ...process.argv.slice(2)], { stdio: 'inherit' });
