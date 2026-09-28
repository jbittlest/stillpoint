/**
 * TELEMETRY golden test: the TypeScript port (src/mp4.ts, src/dji.ts, src/telemetry.ts) vs the Python engine, on the
 * real clips (read in place through fs.openAsBlob, or a FileHandle stand-in for files >= 4 GiB because Node 26's
 * openAsBlob truncates their size to 32 bits -- see telemetry.nodeblob.ts; nothing is copied).
 *
 *   cd stillpoint/web && npx vitest run test/telemetry.test.ts
 *
 * Fixtures: test/fixtures/telemetry/<name>.json.gz, written by test/telemetry.golden.py (first 600 frames + their IMU
 * samples, ~400 frames / ~3000 IMU samples spread over the whole clip, flags, lens, warnings, timing extras, MP4
 * sample-table summaries). With STILLPOINT_TELEMETRY_FULL=<dir> (telemetry.golden.py --full <dir>) EVERY frame and
 * IMU sample is compared. Clips missing on this machine (unmounted SD card, ...) are skipped.
 * Tolerances: timestamps 1 us, quaternions 1e-6, flags / lens / segments / warnings identical.
 * Codec strings + decoder descriptions are cross-checked against mediabunny's independent MP4 demuxer.
 */
import { afterAll, describe, expect, it } from 'vitest';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mainAudioTrack, mainVideoTrack, openMp4, presentationTimes, readSamples, trackFps } from '../src/mp4';
import { loadTelemetry } from '../src/telemetry';
import type { Mp4Info, Telemetry } from '../src/types';
import { closeFileBlobs, fileBlob } from './telemetry.nodeblob';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const FIX = join(HERE, 'fixtures', 'telemetry');
const HOME = homedir();
const SEARCH = [
  process.env.STILLPOINT_OA4_DIR ?? join(HOME, 'Desktop'),
  process.env.STILLPOINT_O3_DIR ?? join(HOME, 'Desktop', 'untitled folder 4'),
  process.env.STILLPOINT_SD_DIR ?? '/Volumes/Untitled/DCIM/DJI_001',
];
const FULL_DIR = process.env.STILLPOINT_TELEMETRY_FULL ?? '';
const REPORT_DIR = process.env.STILLPOINT_TELEMETRY_REPORT ?? '';

const US = 1e-6;       // timestamp tolerance (s)
const QTOL = 1e-6;     // quaternion tolerance

function b64f64(s: string): Float64Array {
  const b = Buffer.from(s, 'base64');
  const out = new Float64Array(b.length / 8);
  new Uint8Array(out.buffer).set(b);
  return out;
}

function findClip(name: string, size: number): string | null {
  for (const d of SEARCH) {
    const p = join(d, name);
    if (existsSync(p)) return p;
  }
  void size;
  return null;
}

/** max |a-b| (NaN == NaN) */
function maxAbs(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if (Number.isNaN(x) && Number.isNaN(y)) continue;
    const d = Math.abs(x - y);
    if (!(d <= m)) m = Number.isNaN(d) ? Infinity : d;
  }
  return m;
}

function pick(a: Float64Array, idx: number[], width = 1): Float64Array {
  const out = new Float64Array(idx.length * width);
  idx.forEach((i, j) => { for (let c = 0; c < width; c++) out[j * width + c] = a[i * width + c]; });
  return out;
}

const fixtures = existsSync(FIX) ? readdirSync(FIX).filter((f) => f.endsWith('.json.gz')).sort() : [];
const results: Record<string, unknown>[] = [];
afterAll(closeFileBlobs);

describe('telemetry port vs Python engine', () => {
  for (const f of fixtures) {
    const fx = JSON.parse(gunzipSync(readFileSync(join(FIX, f))).toString('utf-8'));
    const path = findClip(fx.file, fx.size);
    const run = path ? it : it.skip;
    run(`${fx.name} (${fx.file}, ${(fx.size / 2 ** 30).toFixed(2)} GiB)`, async () => {
      const blob = await fileBlob(path!);
      expect(blob.size).toBe(fx.size);
      const t0 = performance.now();
      const info: Mp4Info = await openMp4(blob);
      const t1 = performance.now();
      const tel: Telemetry = await loadTelemetry(blob, info);
      const t2 = performance.now();
      const res: Record<string, unknown> = { name: fx.name, file: fx.file, sizeGiB: +(fx.size / 2 ** 30).toFixed(2),
        frames: tel.framePts.length, imu: tel.imuT.length, openMs: +(t1 - t0).toFixed(1),
        parseMs: +(t2 - t1).toFixed(1), pythonParseS: fx.python_parse_s };

      // ---------------------------------------------------------------- MP4 sample tables
      for (const pt of fx.tracks) {
        const tt = info.tracks.find((t) => t.id === pt.id)!;
        expect(tt, `track ${pt.id}`).toBeTruthy();
        expect([tt.fourcc, tt.handler, tt.handlerName, tt.timescale, tt.sampleCount])
          .toEqual([pt.fourcc, pt.handler, pt.handler_name, pt.timescale, pt.n]);
        if (pt.handler === 'vide') expect([tt.width, tt.height]).toEqual([pt.width, pt.height]);
        if (!pt.n) continue;
        let ss = 0, so = 0, sp = 0;
        for (let i = 0; i < tt.sampleCount; i++) {
          ss += tt.sizes[i]; so += tt.offsets[i]; sp += Math.round(tt.cts[i] * tt.timescale);
        }
        expect(ss).toBe(pt.sum_sizes);
        if (pt.sum_offsets !== undefined) expect(so).toBe(pt.sum_offsets);
        expect(sp).toBe(pt.sum_pts_ticks);
        for (let j = 0; j < pt.idx.length; j++) {
          const i = pt.idx[j];
          expect(tt.sizes[i]).toBe(pt.sizes[j]);
          if (pt.offsets) expect(tt.offsets[i]).toBe(pt.offsets[j]);
          expect(tt.cts[i]).toBe(pt.pts_ticks[j] / pt.timescale);
        }
        let nSync = 0;
        for (let i = 0; i < tt.sampleCount; i++) nSync += tt.sync[i];
        expect(nSync).toBe(pt.n_sync);
      }
      const vt = mainVideoTrack(info);
      expect(vt.id).toBe(fx.probe.main_video_track_id);
      expect(presentationTimes(vt).length).toBe(fx.probe.n_frames);
      expect(trackFps(vt)).toBe(fx.probe.fps);
      expect(info.comment).toBe(fx.probe.comment);
      expect(info.encoder).toBe(fx.probe.encoder);
      expect(vt.codec).toBe(fx.probe.codec);
      res.codec = vt.codecString;
      res.colr = vt.colr;
      const at = mainAudioTrack(info);
      if (at) res.audio = { codec: at.codecString, sampleRate: at.sampleRate, channels: at.channels };

      // ---------------------------------------------------------------- scalars, flags, lens
      expect(tel.camera).toBe(fx.camera);
      expect([tel.width, tel.height]).toEqual([fx.width, fx.height]);
      expect(tel.fps).toBe(fx.fps);
      expect(Math.abs(tel.readoutS - fx.readout_s)).toBeLessThan(1e-9);
      expect(Math.abs(tel.imuRate / fx.imu_rate - 1)).toBeLessThan(1e-9);
      expect(tel.hasHighrate).toBe(fx.has_highrate);
      expect(tel.eisBaked).toBe(fx.eis_baked);
      const grav = tel.gravityQ === undefined ? 'none' : (tel.gravityQ === tel.imuQ ? 'same' : 'array');
      expect(grav).toBe(fx.gravity);
      expect(tel.lens).toEqual(fx.lens);
      expect(tel.segments).toEqual(fx.segments);
      expect(tel.warnings).toEqual(fx.warnings);
      expect(tel.framePts.length).toBe(fx.n_frames);
      expect(tel.imuT.length).toBe(fx.n_imu);
      const tim = tel.extra!.timing as Record<string, number | null>;
      expect(tim.beta_exposure).toBe(fx.extras.timing.beta_exposure);
      expect(tim.exposure_ref_s).toBe(fx.extras.timing.exposure_ref_s);
      expect(tim.picture_offset_model).toBe(fx.extras.timing.picture_offset_model);
      expect(Math.abs((tim.picture_offset_s as number) - fx.extras.timing.picture_offset_s)).toBeLessThan(1e-12);
      if (fx.extras.timing.beta_fitted !== null) {
        expect(Math.abs((tim.beta_fitted as number) - fx.extras.timing.beta_fitted)).toBeLessThan(1e-6);
      } else expect(tim.beta_fitted).toBeNull();
      expect(tel.extra!.imu_full_coverage_frames).toEqual(fx.extras.imu_full_coverage_frames);
      expect(tel.extra!.eis_status).toBe(fx.extras.eis_status ?? null);
      if (fx.extras.gravity_W) {
        expect(maxAbs(tel.extra!.gravity_W as number[], fx.extras.gravity_W)).toBeLessThan(QTOL);
      }

      // ---------------------------------------------------------------- head (first frames + their IMU samples)
      const h = fx.head;
      const errs: Record<string, number> = {};
      const acc = (k: string, v: number) => { errs[k] = Math.max(errs[k] ?? 0, v); };
      acc('framePts', maxAbs(tel.framePts.subarray(0, h.frames), b64f64(h.frame_pts)));
      acc('frameT', maxAbs(tel.frameT.subarray(0, h.frames), b64f64(h.frame_t)));
      acc('exposure', maxAbs(tel.exposureS.subarray(0, h.frames), b64f64(h.exposure_s)));
      acc('imuT', maxAbs(tel.imuT.subarray(0, h.n_imu), b64f64(h.imu_t)));
      acc('imuQ', maxAbs(tel.imuQ.subarray(0, 4 * h.n_imu), b64f64(h.imu_q)));
      if (h.gravity_q) acc('gravityQ', maxAbs(tel.gravityQ!.subarray(0, 4 * h.n_imu), b64f64(h.gravity_q)));
      // ---------------------------------------------------------------- strided over the whole clip
      const s = fx.strided;
      acc('framePts', maxAbs(pick(tel.framePts, s.frame_idx), b64f64(s.frame_pts)));
      acc('frameT', maxAbs(pick(tel.frameT, s.frame_idx), b64f64(s.frame_t)));
      acc('exposure', maxAbs(pick(tel.exposureS, s.frame_idx), b64f64(s.exposure_s)));
      acc('imuT', maxAbs(pick(tel.imuT, s.imu_idx), b64f64(s.imu_t)));
      acc('imuQ', maxAbs(pick(tel.imuQ, s.imu_idx, 4), b64f64(s.imu_q)));
      if (s.gravity_q) acc('gravityQ', maxAbs(pick(tel.gravityQ!, s.imu_idx, 4), b64f64(s.gravity_q)));
      // ---------------------------------------------------------------- optional: every sample
      const full = FULL_DIR && join(FULL_DIR, `${fx.name}.full.json`);
      if (full && existsSync(full)) {
        const lay = JSON.parse(readFileSync(full, 'utf-8')).layout as Record<string, [number, number]>;
        const bin = readFileSync(join(FULL_DIR, `${fx.name}.full.bin`));
        const arr = (k: string) => {
          const [off, n] = lay[k];
          const o = new Float64Array(n);
          new Uint8Array(o.buffer).set(bin.subarray(off, off + 8 * n));
          return o;
        };
        acc('framePts', maxAbs(tel.framePts, arr('frame_pts')));
        acc('frameT', maxAbs(tel.frameT, arr('frame_t')));
        acc('exposure', maxAbs(tel.exposureS, arr('exposure_s')));
        acc('imuT', maxAbs(tel.imuT, arr('imu_t')));
        acc('imuQ', maxAbs(tel.imuQ, arr('imu_q')));
        if (lay.gravity_q) acc('gravityQ', maxAbs(tel.gravityQ!, arr('gravity_q')));
        res.fullCompare = true;
      }
      res.maxErr = Object.fromEntries(Object.entries(errs).map(([k, v]) => [k, Number(v.toExponential(2))]));
      expect(errs.framePts).toBeLessThanOrEqual(US);
      expect(errs.frameT).toBeLessThanOrEqual(US);
      expect(errs.exposure).toBe(0);
      expect(errs.imuT).toBeLessThanOrEqual(US);
      expect(errs.imuQ).toBeLessThanOrEqual(QTOL);
      if (errs.gravityQ !== undefined) expect(errs.gravityQ).toBeLessThanOrEqual(QTOL);
      // ---------------------------------------------------------------- quick look (probe_telemetry, first 180 frames)
      if (fx.quick) {
        const qk = fx.quick;
        const tq0 = performance.now();
        const q = await loadTelemetry(blob, info, undefined, { maxFrames: 180 });
        res.quickMs = +(performance.now() - tq0).toFixed(1);
        expect(q.framePts.length).toBe(qk.n_frames);
        expect(q.imuT.length).toBe(qk.n_imu);
        expect(q.warnings).toEqual(qk.warnings);
        expect(q.extra!.quick).toEqual(qk.quick);
        expect(q.segments).toEqual(qk.segments);
        expect([q.eisBaked, q.hasHighrate]).toEqual([qk.eis_baked, qk.has_highrate]);
        expect(Math.abs(q.imuRate / qk.imu_rate - 1)).toBeLessThan(1e-9);
        expect(Math.abs(q.readoutS - qk.readout_s)).toBeLessThan(1e-9);
        const qe = { frameT: maxAbs(q.frameT, b64f64(qk.frame_t)), imuT: maxAbs(pick(q.imuT, qk.imu_idx), b64f64(qk.imu_t)),
                     imuQ: maxAbs(pick(q.imuQ, qk.imu_idx, 4), b64f64(qk.imu_q)) };
        res.quickMaxErr = qe;
        expect(qe.frameT).toBeLessThanOrEqual(US);
        expect(qe.imuT).toBeLessThanOrEqual(US);
        expect(qe.imuQ).toBeLessThanOrEqual(QTOL);
      }
      results.push(res);
      console.log(JSON.stringify(res));
      if (REPORT_DIR) {
        mkdirSync(REPORT_DIR, { recursive: true });
        writeFileSync(join(REPORT_DIR, 'telemetry_results.json'), JSON.stringify(results, null, 1));
      }
    }, 300_000);
  }
});

describe('WebCodecs configs vs mediabunny (independent demuxer)', () => {
  const clips = ['DJI_0026.MP4', 'DJI_20260926153751_0005_D.MP4', 'DJI_20260927091931_0012_D.MP4']
    .map((n) => findClip(n, 0)).filter((p): p is string => !!p);
  for (const p of clips) {
    it(`codec string / description / colour / audio: ${p.split('/').pop()}`, async () => {
      const mb = await import('mediabunny');
      const blob = await fileBlob(p);
      const info = await openMp4(blob);
      const vt = mainVideoTrack(info);
      const input = new mb.Input({ source: new mb.FilePathSource(p), formats: mb.ALL_FORMATS });
      const mv = await input.getPrimaryVideoTrack();
      const cfg = await mv!.getDecoderConfig();
      expect(vt.codecString!.replace(/^hvc1|^hev1/, 'hevX')).toBe(cfg!.codec.replace(/^hvc1|^hev1/, 'hevX'));
      expect(Buffer.from(vt.codecConfig!).equals(Buffer.from(cfg!.description as Uint8Array))).toBe(true);
      expect(vt.width).toBe(mv!.codedWidth);
      expect(vt.height).toBe(mv!.codedHeight);
      const cs = await mv!.getColorSpace();
      expect(vt.colr?.fullRange ?? false).toBe(!!cs.fullRange);
      const ma = await input.getPrimaryAudioTrack();
      const at = mainAudioTrack(info);
      expect(!!ma).toBe(!!at);
      if (ma && at) {
        const acfg = await ma.getDecoderConfig();
        expect(at.codecString).toBe(acfg!.codec);
        expect(at.sampleRate).toBe(acfg!.sampleRate);
        expect(at.channels).toBe(acfg!.numberOfChannels);
        expect(Buffer.from(at.codecConfig!).equals(Buffer.from(acfg!.description as Uint8Array))).toBe(true);
      }
      // the first video samples are real access units (length-prefixed NAL units filling the sample exactly)
      const smp = await readSamples(blob, vt, 0, 3);
      for (const b of smp) {
        let q = 0;
        while (q + 4 <= b.length) q += 4 + ((b[q] << 24) >>> 0) + (b[q + 1] << 16) + (b[q + 2] << 8) + b[q + 3];
        expect(q).toBe(b.length);
      }
    }, 120_000);
  }
});

describe('non-DJI MP4 (ffmpeg-muxed comparison render)', () => {
  const p = join(HOME, 'Library', 'Application Support', 'Stillpoint', 'scratch', 'deliver', 'compare_0009.mp4');
  (existsSync(p) ? it : it.skip)('parses the container, matches mediabunny, and refuses telemetry clearly', async () => {
    const mb = await import('mediabunny');
    const blob = await fileBlob(p);
    const info = await openMp4(blob);
    const vt = mainVideoTrack(info);
    const input = new mb.Input({ source: new mb.FilePathSource(p), formats: mb.ALL_FORMATS });
    const mv = await input.getPrimaryVideoTrack();
    const cfg = await mv!.getDecoderConfig();
    expect(vt.codecString!.replace(/^hvc1|^hev1/, 'hevX')).toBe(cfg!.codec.replace(/^hvc1|^hev1/, 'hevX'));
    expect(Buffer.from(vt.codecConfig!).equals(Buffer.from(cfg!.description as Uint8Array))).toBe(true);
    expect(vt.sampleCount).toBe(await mv!.computePacketStats().then((st) => st.packetCount));
    await expect(loadTelemetry(blob, info)).rejects.toThrow(/no DJI djmd telemetry track/);
  }, 120_000);
});
