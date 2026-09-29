// Robustness harness: the test-clip catalogue, and damaged copies made on demand (never full copies of the footage).
//
// Damaged variants (made from a source clip into <scratch>/robustness/damaged/<pid>/ — per process, so concurrent runs
// never delete each other's files — and deleted after the run unless --keep):
//   *-head  : the first 40 MB only (moov is at the end of DJI files, so it is lost) — an interrupted copy
//   *-trunc : the first 40 MB + the original moov appended (mdat size patched): the sample table is intact but every
//             sample past 40 MB points into the moov box or past EOF — "moov present, sample data missing"
//   *-hole  : a SPARSE file of the original size: the first 40 MB and the moov at their original offsets, zeros in
//             between (an interrupted / not-yet-downloaded cloud-sync copy). Uses ~42 MB of disk, not the file size.
import { closeSync, existsSync, ftruncateSync, mkdirSync, openSync, readdirSync, readSync, rmSync, statSync, writeSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const HOME = homedir();
export const SCRATCH = process.env.STILLPOINT_SCRATCH ?? join(HOME, 'Library/Application Support/Stillpoint/scratch/robustness');
const O3 = process.env.STILLPOINT_O3_DIR ?? join(HOME, 'Desktop/untitled folder 4');
const SD = process.env.STILLPOINT_SD_DIR ?? '/Volumes/Untitled/DCIM/DJI_001';
const firstMatch = (dir, re) => { try { const f = readdirSync(dir).filter(n => re.test(n)).sort()[0]; return f ? join(dir, f) : ''; } catch { return ''; } };

export const CLIPS = {
  o3: { path: process.env.CLIP_O3 ?? join(O3, 'DJI_0026.MP4'), desc: 'O3 Air Unit, H.264 High L5.2 3840x2160 59.94p ~150 Mb/s (4.5 s)' },
  'o3-long': { path: process.env.CLIP_O3_LONG ?? join(O3, 'DJI_0034.MP4'), desc: 'O3 Air Unit, H.264 High L5.2 4K60 (50 s, 1 GB)' },
  oa4: { path: process.env.CLIP_OA4 ?? join(HOME, 'Desktop/DJI_20260927091931_0012_D.MP4'), desc: 'Osmo Action 4, HEVC Main10 3840x2880 4:3 59.94p (6.7 GB)' },
  'oa4-169': { path: process.env.CLIP_OA4_169 ?? firstMatch(SD, /_0002_D\.MP4$/i), desc: 'Osmo Action 4, HEVC Main10 16:9 (SD card)' },
  'oa4-short': { path: process.env.CLIP_OA4_SHORT ?? firstMatch(SD, /_0005_D\.MP4$/i), desc: 'Osmo Action 4, HEVC Main10 4:3 short (SD card)' },
  o4p: { path: process.env.CLIP_O4P ?? join(HOME, 'Desktop/DJI_20260925151512_0004_D.MP4'), desc: 'O4 Pro, H.264 High L5.2 3840x2160 59.94p (3.3 GB)' },
  nondji: { path: process.env.CLIP_NONDJI ?? firstMatch(join(HOME, 'Downloads'), /^ScreenRecording.*\.MP4$/i), desc: 'non-DJI: iPhone screen recording, HEVC Main with B-frames, CRA+RASL GOPs, no gyro' },
  'o3-head': { from: 'o3-long', kind: 'head', desc: 'O3 clip, first 40 MB only (no moov)' },
  'o3-trunc': { from: 'o3-long', kind: 'trunc', desc: 'O3 clip, first 40 MB + moov (sample data missing after ~2 s)' },
  'o3-hole': { from: 'o3-long', kind: 'hole', desc: 'O3 clip, sparse: zeros after 40 MB, moov intact' },
  'oa4-trunc': { from: 'oa4', kind: 'trunc', desc: 'OA4 clip, first 40 MB + moov' },
  'oa4-hole': { from: 'oa4', kind: 'hole', desc: 'OA4 clip, sparse: zeros after 40 MB, moov intact' },
};
export const DAMAGED = new Set(Object.keys(CLIPS).filter(k => CLIPS[k].from));

const HEAD = 40 * 1024 * 1024;

function topBoxes(fd, size) {
  const out = [];
  const h = Buffer.alloc(16);
  let off = 0;
  while (off + 8 <= size) {
    readSync(fd, h, 0, 16, off);
    let n = h.readUInt32BE(0);
    const type = h.toString('latin1', 4, 8);
    let hdr = 8;
    if (n === 1) { n = Number(h.readBigUInt64BE(8)); hdr = 16; } else if (n === 0) n = size - off;
    if (n < 8) break;
    out.push({ type, off, size: n, hdr });
    off += n;
  }
  return out;
}

function copyRange(src, dst, from, len, to) {
  const buf = Buffer.alloc(8 * 1024 * 1024);
  let done = 0;
  while (done < len) {
    const n = readSync(src, buf, 0, Math.min(buf.length, len - done), from + done);
    if (!n) break;
    writeSync(dst, buf, 0, n, to + done);
    done += n;
  }
}

/** Make (or reuse) a damaged variant; returns its path. */
export function makeDamaged(name) {
  const c = CLIPS[name];
  const src = CLIPS[c.from].path;
  if (!src || !existsSync(src)) throw new Error(`source clip for ${name} not found: ${src}`);
  const dir = join(SCRATCH, 'damaged', String(process.pid));
  mkdirSync(dir, { recursive: true });
  const out = join(dir, `${name}.MP4`);
  if (existsSync(out)) return out;
  const size = statSync(src).size;
  const s = openSync(src, 'r');
  const d = openSync(out, 'w');
  try {
    const boxes = topBoxes(s, size);
    const moov = boxes.find(b => b.type === 'moov');
    const mdat = boxes.find(b => b.type === 'mdat');
    if (!moov || !mdat) throw new Error(`${src}: no moov/mdat`);
    copyRange(s, d, 0, Math.min(HEAD, size), 0);
    if (c.kind === 'trunc') {
      // mdat now ends at HEAD; moov follows it. Chunk offsets are absolute, so the first ~40 MB of samples still resolve.
      const newSize = HEAD - mdat.off;
      const hb = Buffer.alloc(8);
      if (mdat.hdr === 16) { hb.writeBigUInt64BE(BigInt(newSize)); writeSync(d, hb, 0, 8, mdat.off + 8); }
      else { hb.writeUInt32BE(newSize); writeSync(d, hb, 0, 4, mdat.off); }
      copyRange(s, d, moov.off, moov.size, HEAD);
    } else if (c.kind === 'hole') {
      ftruncateSync(d, size); // sparse on APFS
      copyRange(s, d, moov.off, moov.size, moov.off);
    }
  } finally { closeSync(s); closeSync(d); }
  return out;
}

export function removeDamaged() { rmSync(join(SCRATCH, 'damaged', String(process.pid)), { recursive: true, force: true }); }

/** Resolve a clip name (or a path) to { name, path, desc, damaged }; null when the clip is not available. */
export function resolveClip(name) {
  if (!CLIPS[name]) return existsSync(name) ? { name, path: name, desc: name, damaged: false } : null;
  const c = CLIPS[name];
  if (c.from) {
    const srcOk = CLIPS[c.from].path && existsSync(CLIPS[c.from].path);
    return srcOk ? { name, path: makeDamaged(name), desc: c.desc, damaged: true } : null;
  }
  return c.path && existsSync(c.path) ? { name, path: c.path, desc: c.desc, damaged: false } : null;
}
