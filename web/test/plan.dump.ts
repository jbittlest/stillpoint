/**
 * A/B tool (integration): parse a clip with the web telemetry port and build the web Plan in Node, then write it for
 * the Python side (test/plan.dump.py turns it into a .spplan the Metal renderer `sprender` can render, so the web
 * path can be rendered by the engine's renderer and vice versa).
 *   node test/plan.dump.run.mjs CLIP OUT_PREFIX ['{"outFx":1548.78}']
 * Writes OUT_PREFIX.json (meta) and OUT_PREFIX.bin (framePts f64 | outFx f32 | rowMats f32 | virtQ f64).
 */
import { openAsBlob } from 'node:fs';
import { writeFileSync } from 'node:fs';
import { loadTelemetryFromFile } from '../src/telemetry';
import { buildPlan, type StabParams } from '../src/plan';

const [clip, out, extra] = process.argv.slice(2);
const blob = await openAsBlob(clip);
const t0 = performance.now();
const { tel } = await loadTelemetryFromFile(blob);
const t1 = performance.now();
const params: StabParams = { smoothness: 1, footprint: 0.6, horizonLock: false, ...(extra ? JSON.parse(extra) : {}) };
const plan = buildPlan(tel, params);
const t2 = performance.now();
const parts = [plan.framePts, plan.outFx, plan.rowMats, plan.virtQ];
writeFileSync(out + '.bin', Buffer.concat(parts.map(a => Buffer.from(a.buffer, a.byteOffset, a.byteLength))));
writeFileSync(out + '.json', JSON.stringify({
  clip, params, srcW: plan.srcW, srcH: plan.srcH, outW: plan.outW, outH: plan.outH, nRows: plan.nRows, frames: plan.framePts.length,
  readoutS: plan.readoutS, lens: plan.lens, fx0: plan.fx0, hfovDeg: plan.hfovDeg, exposureAvg: plan.exposureAvg,
  smoothInfo: plan.smoothInfo, telemetryS: (t1 - t0) / 1000, planS: (t2 - t1) / 1000,
  frameT: Array.from(tel.frameT), readout: tel.readoutS,
}));
console.log(JSON.stringify({ frames: plan.framePts.length, fx0: plan.fx0, telemetryS: (t1 - t0) / 1000, planS: (t2 - t1) / 1000 }));
