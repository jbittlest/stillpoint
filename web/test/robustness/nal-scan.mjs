// Robustness: NAL-unit census of a clip's sync samples (and the samples right after them), using the app's own MP4
// parser (src/mp4.ts, loaded with Node's TypeScript type stripping). Answers: are the samples the app treats as
// keyframes clean random-access points (IDR) for WebCodecs, or CRA / non-IDR recovery points that some decoders
// reject as the first chunk after configure()/flush()? Are there leading pictures (RASL/RADL) after a CRA? Does any
// non-sync sample carry an IRAP (incomplete stss)? Does the file end before the sample table says it should?
//
//   node test/robustness/nal-scan.mjs CLIP [CLIP...] [--max 60] [--json]
//
// Read-only on the clips; reads at most ~max sync samples (+3 followers each) per clip.
import { openSync, readSync, statSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { openMp4, mainVideoTrack, readSamples } from '../../src/mp4.ts';

const HEVC = { 0: 'TRAIL_N', 1: 'TRAIL_R', 2: 'TSA_N', 3: 'TSA_R', 4: 'STSA_N', 5: 'STSA_R', 6: 'RADL_N', 7: 'RADL_R', 8: 'RASL_N', 9: 'RASL_R',
  16: 'BLA_W_LP', 17: 'BLA_W_RADL', 18: 'BLA_N_LP', 19: 'IDR_W_RADL', 20: 'IDR_N_LP', 21: 'CRA', 32: 'VPS', 33: 'SPS', 34: 'PPS', 35: 'AUD', 36: 'EOS', 37: 'EOB', 38: 'FD', 39: 'SEI_PREFIX', 40: 'SEI_SUFFIX' };
const AVC = { 1: 'NON_IDR', 2: 'DPA', 5: 'IDR', 6: 'SEI', 7: 'SPS', 8: 'PPS', 9: 'AUD', 10: 'EOSEQ', 11: 'EOSTREAM', 12: 'FILLER', 14: 'PREFIX', 20: 'SLICE_EXT' };

/** NAL unit types of one length-prefixed access unit. For H.264 SEI, also reports recovery-point SEI (payload 6). */
export function nalTypes(au, codec, lenSize = 4) {
  const out = [];
  let p = 0;
  while (p + lenSize <= au.length) {
    let n = 0;
    for (let k = 0; k < lenSize; k++) n = n * 256 + au[p + k];
    p += lenSize;
    if (n <= 0 || p + n > au.length) { out.push(`BAD_LEN(${n})`); break; }
    const b0 = au[p];
    if (codec === 'hevc') {
      const t = (b0 >> 1) & 0x3f;
      out.push(HEVC[t] ?? `hevc${t}`);
    } else {
      const t = b0 & 0x1f;
      let name = AVC[t] ?? `avc${t}`;
      if (t === 6 && au[p + 1] === 6) name = 'SEI(recovery_point)';
      out.push(name);
    }
    p += n;
  }
  if (!au.length) out.push('EMPTY');
  return out;
}

function lengthSize(t) {
  const c = t.codecConfig;
  if (!c) return 4;
  if (t.codec === 'hevc' || t.fourcc === 'hvc1' || t.fourcc === 'hev1') return (c[21] & 3) + 1;
  return (c[4] & 3) + 1;
}

const IRAP_HEVC = new Set(['BLA_W_LP', 'BLA_W_RADL', 'BLA_N_LP', 'IDR_W_RADL', 'IDR_N_LP', 'CRA']);
const summarize = arr => { const m = new Map(); for (const k of arr) m.set(k, (m.get(k) ?? 0) + 1); return Object.fromEntries(m); };

/** Minimal Blob stand-in over a file descriptor: Node's fs.openAsBlob wraps .size at 4 GiB (Node 26), and the OA4 /
 *  O4 Pro clips are 3-8 GB. Only what src/mp4.ts uses (size, slice().arrayBuffer()). */
export function fileBlob(path) {
  const fd = openSync(path, 'r');
  const size = statSync(path).size;
  const mk = (a, b) => ({
    size: b - a,
    slice: (s = 0, e = b - a) => mk(a + Math.max(0, s), a + Math.min(b - a, e)),
    arrayBuffer: async () => { const buf = Buffer.alloc(Math.max(0, b - a)); let got = 0; while (got < buf.length) { const n = readSync(fd, buf, got, buf.length - got, a + got); if (!n) break; got += n; } return buf.buffer.slice(buf.byteOffset, buf.byteOffset + got); },
  });
  return mk(0, size);
}

export async function scan(path, { max = 60 } = {}) {
  const size = statSync(path).size;
  const blob = fileBlob(path);
  const info = await openMp4(blob);
  const t = mainVideoTrack(info);
  const codec = /^(hvc1|hev1)$/.test(t.fourcc) ? 'hevc' : /^(avc1|avc3)$/.test(t.fourcc) ? 'h264' : t.fourcc;
  const ls = lengthSize(t);
  const syncIdx = [];
  for (let i = 0; i < t.sampleCount; i++) if (t.sync[i]) syncIdx.push(i);
  const gops = syncIdx.slice(1).map((s, i) => s - syncIdx[i]);
  let lastByte = 0;
  for (let i = 0; i < t.sampleCount; i++) lastByte = Math.max(lastByte, t.offsets[i] + t.sizes[i]);
  const missing = [];
  for (let i = 0; i < t.sampleCount; i++) if (t.offsets[i] + t.sizes[i] > size) missing.push(i);
  // evenly spaced sync samples (always the first two and the last), each with its 3 decode-order followers
  const pick = new Set([0, 1, syncIdx.length - 1].filter(k => k >= 0 && k < syncIdx.length));
  const m = Math.min(max, syncIdx.length);
  for (let k = 0; k < m; k++) pick.add(Math.round(k * (syncIdx.length - 1) / Math.max(1, m - 1)));
  const syncTypes = [], followerTypes = [], perSync = [];
  for (const k of [...pick].sort((a, b) => a - b)) {
    const s = syncIdx[k];
    if (t.offsets[s] + t.sizes[s] > size) { perSync.push({ sample: s, nal: ['BEYOND_EOF'] }); continue; }
    const cnt = Math.min(4, t.sampleCount - s);
    const data = await readSamples(blob, t, s, cnt);
    const types = nalTypes(data[0], codec, ls);
    const vcl = types.filter(x => codec === 'hevc' ? IRAP_HEVC.has(x) || /TRAIL|TSA|RADL|RASL/.test(x) : /IDR|NON_IDR/.test(x));
    syncTypes.push(vcl.join('+') || types.join('+'));
    const fol = data.slice(1).map(d => nalTypes(d, codec, ls).filter(x => !/SEI|AUD|SPS|PPS|VPS/.test(x)).join('+'));
    followerTypes.push(...fol);
    if (perSync.length < 6) perSync.push({ sample: s, nal: types, next: fol });
  }
  // non-sync samples that carry an IRAP (stss incomplete) — check a sample of non-sync frames
  const irapInNonSync = [];
  const stride = Math.max(1, Math.floor(t.sampleCount / 400));
  for (let i = 0; i < t.sampleCount; i += stride) {
    if (t.sync[i] || t.offsets[i] + t.sizes[i] > size) continue;
    const [d] = await readSamples(blob, t, i, 1);
    const ty = nalTypes(d, codec, ls);
    if (ty.some(x => IRAP_HEVC.has(x) || x === 'IDR')) irapInNonSync.push(i);
  }
  const cts = Array.from(t.cts), dts = Array.from(t.dts);
  let reordered = 0;
  for (let i = 1; i < cts.length; i++) if (cts[i] < cts[i - 1]) reordered++;
  return {
    clip: basename(path), bytes: size, codec: t.codecString, fourcc: t.fourcc, size: `${t.width}x${t.height}`, samples: t.sampleCount,
    nalLengthSize: ls, syncSamples: syncIdx.length, firstSampleIsSync: !!t.sync[0],
    gop: gops.length ? { min: Math.min(...gops), max: Math.max(...gops), median: gops.sort((a, b) => a - b)[gops.length >> 1] } : null,
    bFrames: reordered > 0, ctsMinusDtsFirst: +(cts[0] - dts[0]).toFixed(4),
    syncNal: summarize(syncTypes), followerNal: summarize(followerTypes),
    irapInNonSyncSampled: irapInNonSync.length, samplesBeyondEof: missing.length, firstMissingSample: missing[0] ?? null,
    sampleDataEnd: lastByte, examples: perSync,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  const mi = args.indexOf('--max');
  const max = mi >= 0 ? +args[mi + 1] : 60;
  const json = args.includes('--json');
  const clips = args.filter((a, i) => !a.startsWith('--') && !(mi >= 0 && i === mi + 1));
  const res = [];
  for (const c of clips) {
    try { res.push(await scan(c, { max })); } catch (e) { res.push({ clip: basename(c), error: String(e?.message ?? e) }); }
    if (!json) { const r = res[res.length - 1]; const { examples, ...rest } = r; console.log(JSON.stringify(rest)); if (examples) for (const e of examples.slice(0, 2)) console.log('   ', JSON.stringify(e)); }
  }
  if (json) console.log(JSON.stringify(res, null, 1));
}

/**
 * Presentation-order frames that are LEADING pictures of a CRA / non-IDR sync sample (decode order after the sync
 * sample, presentation before it — RASL in HEVC, open-GOP B-frames in H.264). Decoding from that sync sample cannot
 * produce them; the app's seek (decode from the preceding sync sample) will miss them. Returns up to `limit` of them,
 * with their pts and the sync sample they hang off. Empty for closed-GOP clips (all DJI footage).
 */
export async function leadingFrames(path, { limit = 6, maxSync = 12 } = {}) {
  const blob = fileBlob(path);
  const info = await openMp4(blob);
  const t = mainVideoTrack(info);
  const codec = /^(hvc1|hev1)$/.test(t.fourcc) ? 'hevc' : /^(avc1|avc3)$/.test(t.fourcc) ? 'h264' : t.fourcc;
  const ls = lengthSize(t);
  const n = t.sampleCount;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => t.cts[a] - t.cts[b] || a - b);
  const presOfDec = new Int32Array(n);
  order.forEach((d, p) => (presOfDec[d] = p));
  const out = [];
  let seen = 0;
  for (let s = 1; s < n && seen < maxSync && out.length < limit; s++) {
    if (!t.sync[s]) continue;
    seen++;
    const [d] = await readSamples(blob, t, s, 1);
    const ty = nalTypes(d, codec, ls);
    const isIdr = ty.some(x => /^IDR/.test(x));
    if (isIdr) continue;
    for (let k = s + 1; k < Math.min(n, s + 16) && out.length < limit; k++) if (t.cts[k] < t.cts[s]) out.push({ pres: presOfDec[k], dec: k, sync: s, syncNal: ty.filter(x => !/SEI|AUD|SPS|PPS|VPS/.test(x)).join('+') });
  }
  return out;
}
