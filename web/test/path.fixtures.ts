/**
 * Loader for the PATH golden fixtures written by web/test/path.export.py (owner: PATH agent).
 * <name>.json = meta + Python results, <name>.bin.gz = little-endian arrays at layout offsets.
 */
import { readFileSync, existsSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Telemetry } from '../src/types';

export interface PathFixture {
  name: string;
  meta: any;
  tel: Telemetry;
  arr: Record<string, Float64Array | Float32Array>;
}

export const FIXTURE_DIR = process.env.PATH_FIXTURE_DIR ?? fileURLToPath(new URL('./fixtures/path/', import.meta.url));

export function hasFixture(name: string, dir = FIXTURE_DIR): boolean {
  return existsSync(join(dir, name + '.json')) && existsSync(join(dir, name + '.bin.gz'));
}

export function loadFixture(name: string, dir = FIXTURE_DIR): PathFixture {
  const meta = JSON.parse(readFileSync(join(dir, name + '.json'), 'utf8'));
  const raw = gunzipSync(readFileSync(join(dir, name + '.bin.gz')));
  const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  const arr: Record<string, Float64Array | Float32Array> = {};
  for (const [k, v] of Object.entries<any>(meta.layout)) {
    arr[k] = v.dtype === 'f32' ? new Float32Array(buf, v.offset, v.length) : new Float64Array(buf, v.offset, v.length);
  }
  const T = meta.tel;
  let imuT: Float64Array;
  if (arr.imuT) imuT = arr.imuT as Float64Array;          // explicit times (non-uniform grids)
  else {
    imuT = new Float64Array(T.nImu);
    for (let i = 0; i < T.nImu; i++) imuT[i] = T.imuT0 + i * T.imuDt;
  }
  const tel: Telemetry = {
    camera: T.camera, width: T.width, height: T.height, fps: T.fps,
    framePts: arr.framePts as Float64Array, frameT: arr.frameT as Float64Array, exposureS: arr.exposureS as Float64Array,
    readoutS: T.readoutS,
    lens: { model: T.lens.model, fx: T.lens.fx, fy: T.lens.fy, cx: T.lens.cx, cy: T.lens.cy, k: T.lens.k,
      width: T.lens.width, height: T.lens.height },
    imuT, imuQ: Float64Array.from(arr.imuQ32), imuRate: T.imuRate, hasHighrate: T.hasHighrate, eisBaked: T.eisBaked,
    warnings: [], segments: T.segments,
  };
  return { name, meta, tel, arr };
}
