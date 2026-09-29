// APP unit tests (node): formatting, encoder codec strings/levels, capability verdicts, frame indexing.
import { describe, expect, it } from 'vitest';
import { fmtBytes, fmtEta, fmtTime, fmtRate, aspectLabel } from '../src/ui/format';
import { avcCodecString, hevcCodecString, av1CodecString, bitratePresets } from '../src/io/encode';
import { verdict, type Caps } from '../src/ui/caps';
import { FrameIndex } from '../src/io/decode';
import { planIndexer } from '../src/io/pipeline';
import type { Mp4Track, Plan } from '../src/types';

describe('format', () => {
  it('times', () => {
    expect(fmtTime(0)).toBe('0:00.0');
    expect(fmtTime(75.25)).toBe('1:15.3');
    expect(fmtTime(59.97)).toBe('1:00.0');   // rounds into the next minute, never 0:60.0
    expect(fmtTime(399.5, false)).toBe('6:39');
    expect(fmtTime(3725, false)).toBe('1:02:05');
  });
  it('eta / bytes / rate / aspect', () => {
    expect(fmtEta(42.2)).toBe('43 s left');
    expect(fmtEta(125)).toBe('2:05 left');
    expect(fmtBytes(57_299_584)).toBe('57.3 MB');
    expect(fmtBytes(4.2e9)).toBe('4.20 GB');
    expect(fmtRate(1999.02)).toBe('2.0 kHz');
    expect(fmtRate(60)).toBe('60 Hz');
    expect(aspectLabel(3840, 2880)).toBe('4:3');
    expect(aspectLabel(3840, 2160)).toBe('16:9');
  });
});

describe('encoder codec strings', () => {
  it('HEVC level fits size, rate and bitrate', () => {
    // 4K60 @ 100 Mb/s: L5.1 luma rate fits; 100 Mb/s > 40 Mb/s main tier -> high tier
    expect(hevcCodecString(3840, 2160, 59.94, 100e6)).toBe('hvc1.1.6.H153.B0');
    expect(hevcCodecString(3840, 2160, 59.94, 35e6)).toBe('hvc1.1.6.L153.B0');
    // 3840x2880 exceeds MaxLumaPs of level 5.x -> level 6
    expect(hevcCodecString(3840, 2880, 59.94, 133e6)).toBe('hvc1.1.6.H180.B0');
    expect(hevcCodecString(1920, 1080, 30, 20e6)).toBe('hvc1.1.6.H120.B0'); // lowest level; high tier when the rate needs it
    expect(hevcCodecString(1920, 1080, 30, 10e6)).toBe('hvc1.1.6.L120.B0');
  });
  it('H.264 level', () => {
    expect(avcCodecString(3840, 2160, 59.94, 100e6)).toBe('avc1.640034'); // 5.2
    expect(avcCodecString(3840, 2160, 30, 100e6)).toBe('avc1.640033');   // 5.1
    expect(avcCodecString(3840, 2880, 59.94, 100e6)).toBe('avc1.64003c'); // 6.0
    expect(avcCodecString(1920, 1080, 60, 20e6)).toBe('avc1.64002a');    // 4.2
  });
  it('AV1 level', () => {
    expect(av1CodecString(3840, 2160, 59.94)).toBe('av01.0.13H.08');
    expect(av1CodecString(3840, 2880, 59.94)).toBe('av01.0.16H.08');
  });
  it('bitrate presets scale with pixels x fps (sub-linearly) and codec', () => {
    const p = bitratePresets(3840, 2160, 59.94);
    expect(p).toEqual({ small: 60e6, high: 100e6, max: 150e6 });
    expect(bitratePresets(3840, 2880, 59.94).high).toBe(126e6);
    const hd30 = bitratePresets(1920, 1080, 30);
    expect(hd30.high).toBe(21e6);
    expect(bitratePresets(1920, 1080, 30, 'avc').high).toBe(29e6);
    expect(bitratePresets(1920, 1080, 30, 'av1').high).toBe(17e6);
    // monotone in size and rate, and never below 2 Mb/s
    expect(bitratePresets(1280, 720, 30).high).toBeLessThan(hd30.high);
    expect(bitratePresets(1920, 1080, 60).high).toBeGreaterThan(hd30.high);
    expect(bitratePresets(64, 64, 1).small).toBe(2e6);
  });
});

const base: Caps = {
  webgpu: true, webcodecs: true, offscreen: true, decodeHevc10: true, decodeHevc: true, decodeH264: true, encodeHevc: true,
  encodeH264: true, encodeAny: true, saveToDisk: true, opfs: true, browser: 'chrome', mobile: false,
};

describe('capability verdict', () => {
  it('desktop Chrome is fully supported', () => {
    expect(verdict(base)).toMatchObject({ ok: true, partial: false });
  });
  it('Firefox without HEVC / WebGPU gets a friendly explanation', () => {
    const v = verdict({ ...base, browser: 'firefox', webgpu: false, decodeHevc10: false, decodeHevc: false, encodeHevc: false });
    expect(v.ok).toBe(false);
    expect(v.body).toMatch(/Firefox/);
    expect(v.body).toMatch(/Chrome/);
  });
  it('old Safari without WebGPU', () => {
    const v = verdict({ ...base, browser: 'safari', webgpu: false });
    expect(v.ok).toBe(false);
    expect(v.body).toMatch(/Safari 26/);
  });
  it('no HEVC decode is a partial (O3 H.264 clips still work)', () => {
    const v = verdict({ ...base, decodeHevc10: false, decodeHevc: false });
    expect(v).toMatchObject({ ok: true, partial: true });
    expect(v.body).toMatch(/H\.264/);
  });
  it('no encoder at all is unsupported', () => {
    expect(verdict({ ...base, encodeHevc: false, encodeH264: false, encodeAny: false }).ok).toBe(false);
  });
});

function track(cts: number[], sync: number[]): Mp4Track {
  const n = cts.length;
  return {
    id: 1, kind: 'video', handler: 'vide', codec: 'hevc', fourcc: 'hvc1', timescale: 60000, sampleCount: n,
    offsets: new Float64Array(n), sizes: new Uint32Array(n).fill(1000), dts: Float64Array.from(cts.map((_, i) => i / 60)),
    cts: Float64Array.from(cts), sync: Uint8Array.from(sync),
  };
}

describe('FrameIndex', () => {
  // decode order I P B B P B B (presentation 0 3 1 2 6 4 5)
  const d = 1001 / 60000;
  const t = track([0, 3, 1, 2, 6, 4, 5].map(p => p * d), [1, 0, 0, 0, 0, 0, 0]);
  const ix = new FrameIndex(t);
  it('maps decode <-> presentation order', () => {
    expect(Array.from(ix.decOfPres)).toEqual([0, 2, 3, 1, 5, 6, 4]);
    expect(Array.from(ix.presOfDec)).toEqual([0, 3, 1, 2, 6, 4, 5]);
    expect(ix.frameDur).toBeCloseTo(d, 9);
  });
  it('timestamps round-trip exactly', () => {
    for (let dd = 0; dd < t.sampleCount; dd++) expect(ix.presOfTimestamp(ix.tsUs[dd])).toBe(ix.presOfDec[dd]);
  });
  it('decode span covers B-frame reordering and starts at a keyframe', () => {
    expect(ix.decodeSpan(1, 2)).toEqual([0, 3]);
    expect(ix.decodeSpan(3, 3)).toEqual([0, 1]);
    expect(ix.decodeSpan(4, 6)).toEqual([0, 6]);
  });
});

describe('planIndexer', () => {
  it('picks the nearest plan record', () => {
    const plan = { framePts: Float64Array.from([0, 0.1, 0.2, 0.3]) } as unknown as Plan;
    const k = planIndexer(plan);
    expect(k(0.149)).toBe(1);
    expect(k(0.151)).toBe(2);
    expect(k(-1)).toBe(0);
    expect(k(9)).toBe(3);
  });
});

describe('export output timing', () => {
  it('track timescale is a multiple of the output rate', async () => {
    const { timescaleFor } = await import('../src/io/pipeline');
    expect(timescaleFor(24000 / 1001)).toBe(24000);
    expect(timescaleFor(30000 / 1001)).toBe(30000);
    expect(timescaleFor(60000 / 1001)).toBe(60000);
    expect(timescaleFor(24)).toBe(24000);
    expect(timescaleFor(25)).toBe(25000);
    expect(timescaleFor(50)).toBe(50000);
    for (const f of [12.5, 48, 47.952, 17.3]) {
      const ts = timescaleFor(f);
      // one frame is (close to) a whole number of ticks
      expect(Math.abs(ts / f - Math.round(ts / f))).toBeLessThan(0.02);
      expect(ts).toBeGreaterThanOrEqual(1000);
    }
  });
});
