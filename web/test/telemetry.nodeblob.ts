/**
 * Blob access to a local file for Node tests (TELEMETRY agent).
 *
 * fs.openAsBlob() is used when it works, but Node 26.0 reports `size` modulo 2^32 for files >= 4 GiB (a 6.2 GiB clip
 * shows up as 2.2 GiB and slices past that are empty), so bigger files get a minimal Blob stand-in backed by a
 * FileHandle: size / slice / arrayBuffer -- exactly what src/mp4.ts uses (File.slice + arrayBuffer in the browser).
 * Reads go straight to the file (pread); nothing is copied.
 */
import { openAsBlob, statSync } from 'node:fs';
import { open, type FileHandle } from 'node:fs/promises';

class FileRangeBlob {
  constructor(private fh: FileHandle, private start: number, private end: number) {}
  get size(): number { return this.end - this.start; }
  get type(): string { return ''; }
  slice(a = 0, b: number = this.size): FileRangeBlob {
    const n = this.size;
    const s = Math.min(n, a < 0 ? Math.max(0, n + a) : a);
    const e = Math.max(s, Math.min(n, b < 0 ? Math.max(0, n + b) : b));
    return new FileRangeBlob(this.fh, this.start + s, this.start + e);
  }
  async arrayBuffer(): Promise<ArrayBuffer> {
    const n = this.size;
    const buf = new Uint8Array(n);
    let got = 0;
    while (got < n) {
      const { bytesRead } = await this.fh.read(buf, got, n - got, this.start + got);
      if (!bytesRead) break;
      got += bytesRead;
    }
    return buf.buffer;
  }
}

const handles: FileHandle[] = [];

/** A Blob for `path` (fs.openAsBlob below 4 GiB, a FileHandle-backed stand-in above). */
export async function fileBlob(path: string): Promise<Blob> {
  const size = statSync(path).size;
  if (size < 2 ** 32) {
    const b = await openAsBlob(path);
    if (b.size === size) return b;
  }
  const fh = await open(path, 'r');
  handles.push(fh);
  return new FileRangeBlob(fh, 0, size) as unknown as Blob;
}

export async function closeFileBlobs() {
  while (handles.length) await handles.pop()!.close();
}
