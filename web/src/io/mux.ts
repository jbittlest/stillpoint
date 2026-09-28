/**
 * MP4 output for Stillpoint: output sinks (File System Access file picked by the user, an OPFS temp file that is then
 * downloaded, or an in-memory buffer for small clips) and a serialized mediabunny muxer that takes encoded video
 * chunks plus passthrough AAC samples. All writes go through one promise chain so muxer backpressure propagates to
 * the render loop (see `pending`).
 */
import {
  Output, Mp4OutputFormat, StreamTarget, BufferTarget, EncodedVideoPacketSource, EncodedAudioPacketSource, EncodedPacket,
  type StreamTargetChunk,
} from 'mediabunny';
import type { OutCodec } from './encode';

export type SinkKind = 'fsa' | 'opfs' | 'memory';

export interface SinkRequest {
  kind: SinkKind;
  /** file name (OPFS / download name) */
  name: string;
  /** for 'fsa': the handle from showSaveFilePicker (main thread), posted to the worker */
  handle?: FileSystemFileHandle;
}

export interface OutputSink {
  kind: SinkKind;
  name: string;
  target: StreamTarget | BufferTarget;
  bytesWritten(): number;
  /** after finalize: close the file; returns the Blob for 'memory', nothing else */
  finish(): Promise<Blob | null>;
  abort(): Promise<void>;
}

export const OPFS_DIR = 'stillpoint-exports';

async function opfsDir(): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(OPFS_DIR, { create: true });
}

/** Remove old OPFS exports (call when nothing is being downloaded, e.g. before a new export). */
export async function clearOpfsExports(keep?: string): Promise<void> {
  try {
    const dir = await opfsDir();
    const names: string[] = [];
    // @ts-expect-error async iterator on FileSystemDirectoryHandle
    for await (const [name] of dir.entries()) names.push(name);
    for (const n of names) if (n !== keep) await dir.removeEntry(n).catch(() => {});
  } catch { /* OPFS unavailable */ }
}

function countingStream(inner: WritableStream<StreamTargetChunk> | null, write: (c: StreamTargetChunk) => void | Promise<void>, close: () => Promise<void>, abort: () => Promise<void>, counter: { bytes: number; end: number }) {
  void inner;
  return new WritableStream<StreamTargetChunk>({
    async write(c) {
      counter.bytes += c.data.byteLength;
      counter.end = Math.max(counter.end, c.position + c.data.byteLength);
      await write(c);
    },
    close, abort,
  });
}

export async function createSink(req: SinkRequest): Promise<OutputSink> {
  const counter = { bytes: 0, end: 0 };
  if (req.kind === 'memory') {
    const target = new BufferTarget();
    return {
      kind: 'memory', name: req.name, target,
      bytesWritten: () => counter.end,
      async finish() { return target.buffer ? new Blob([target.buffer], { type: 'video/mp4' }) : null; },
      async abort() { /* garbage collected */ },
    };
  }
  if (req.kind === 'fsa') {
    if (!req.handle) throw new Error('No file handle for the save location');
    const w = await req.handle.createWritable();
    let closed = false;
    const stream = countingStream(null, c => w.write({ type: 'write', position: c.position, data: c.data }), async () => { closed = true; await w.close(); }, async () => { closed = true; await w.abort(); }, counter);
    return {
      kind: 'fsa', name: req.handle.name, target: new StreamTarget(stream, { chunked: true, chunkSize: 8 * 1024 * 1024 }),
      bytesWritten: () => counter.end,
      async finish() { return null; },
      async abort() { if (!closed) { closed = true; await w.abort().catch(() => {}); } await (req.handle as any).remove?.().catch?.(() => {}); },
    };
  }
  // OPFS: prefer a sync access handle (dedicated workers, all engines), else createWritable
  const dir = await opfsDir();
  await dir.removeEntry(req.name).catch(() => {});
  const fh = await dir.getFileHandle(req.name, { create: true });
  let stream: WritableStream<StreamTargetChunk>;
  let abortFn: () => Promise<void>;
  const sah = typeof (fh as any).createSyncAccessHandle === 'function' && typeof (globalThis as any).WorkerGlobalScope !== 'undefined'
    ? await (fh as any).createSyncAccessHandle().catch(() => null) : null;
  if (sah) {
    let open = true;
    stream = countingStream(null, c => { sah.write(c.data, { at: c.position }); }, async () => { if (open) { open = false; sah.flush(); sah.close(); } }, async () => { if (open) { open = false; sah.close(); } }, counter);
    abortFn = async () => { if (open) { open = false; try { sah.close(); } catch { /* */ } } await dir.removeEntry(req.name).catch(() => {}); };
  } else {
    const w = await (fh as any).createWritable();
    let open = true;
    stream = countingStream(null, c => w.write({ type: 'write', position: c.position, data: c.data }), async () => { if (open) { open = false; await w.close(); } }, async () => { if (open) { open = false; await w.abort(); } }, counter);
    abortFn = async () => { if (open) { open = false; await w.abort().catch(() => {}); } await dir.removeEntry(req.name).catch(() => {}); };
  }
  return {
    kind: 'opfs', name: req.name, target: new StreamTarget(stream, { chunked: true, chunkSize: 8 * 1024 * 1024 }),
    bytesWritten: () => counter.end,
    async finish() { return null; },
    abort: abortFn,
  };
}

/** Main thread: the finished OPFS export as a disk-backed File (for a download link). */
export async function getOpfsExport(name: string): Promise<File> {
  const dir = await opfsDir();
  return (await dir.getFileHandle(name)).getFile();
}

export interface AudioPassthrough {
  codec: 'aac';
  sampleRate: number;
  channels: number;
  codecString: string;
  description?: Uint8Array;
}

const MB_CODEC: Record<OutCodec, 'hevc' | 'avc' | 'av1' | 'vp9'> = { hevc: 'hevc', avc: 'avc', av1: 'av1', vp9: 'vp9' };

export class Mp4Writer {
  readonly output: Output;
  private video: EncodedVideoPacketSource;
  private audio?: EncodedAudioPacketSource;
  private chain: Promise<void> = Promise.resolve();
  private failed: unknown = null;
  private sentVideoMeta = false;
  private sentAudioMeta = false;
  /** mux operations queued but not finished */
  pending = 0;
  videoPackets = 0;
  audioPackets = 0;
  /** encoded payload bytes handed to the muxer (the sink may still be buffering them) */
  payloadBytes = 0;

  constructor(private sink: OutputSink, codec: OutCodec, opts: { timescale: number; audio?: AudioPassthrough; colorSpace?: VideoColorSpaceInit }) {
    this.output = new Output({ format: new Mp4OutputFormat({ fastStart: sink.kind === 'memory' ? 'in-memory' : false }), target: sink.target });
    this.video = new EncodedVideoPacketSource(MB_CODEC[codec]);
    // frameRate = source timescale -> the track timescale equals the source's (exact frame times, no 1/57600 rounding)
    this.output.addVideoTrack(this.video, { frameRate: opts.timescale });
    if (opts.audio) {
      this.audio = new EncodedAudioPacketSource('aac');
      this.output.addAudioTrack(this.audio);
      this.audioCfg = opts.audio;
    }
    this.colorSpace = opts.colorSpace;
    this.output.setMetadataTags({ comment: 'Stabilized with Stillpoint (web)' } as any);
  }
  private audioCfg?: AudioPassthrough;
  private colorSpace?: VideoColorSpaceInit;

  start() { return this.output.start(); }

  get error() { return this.failed; }

  private enqueue(fn: () => Promise<void>) {
    this.pending++;
    this.chain = this.chain.then(async () => {
      if (this.failed) return;
      try { await fn(); } catch (e) { this.failed = e; }
    }).finally(() => { this.pending--; });
    return this.chain;
  }

  addVideo(chunk: EncodedVideoChunk, meta: EncodedVideoChunkMetadata | undefined, tsS: number, durS: number) {
    const data = new Uint8Array(chunk.byteLength);
    chunk.copyTo(data);
    const pkt = new EncodedPacket(data, chunk.type, tsS, durS);
    let m: EncodedVideoChunkMetadata | undefined;
    if (!this.sentVideoMeta && meta?.decoderConfig) {
      m = { ...meta, decoderConfig: { ...meta.decoderConfig, colorSpace: this.colorSpace ?? meta.decoderConfig.colorSpace } };
      this.sentVideoMeta = true;
    }
    this.videoPackets++;
    this.payloadBytes += data.byteLength;
    return this.enqueue(() => this.video.add(pkt, m));
  }

  addAudio(data: Uint8Array, tsS: number, durS: number) {
    if (!this.audio || !this.audioCfg) return Promise.resolve();
    const pkt = new EncodedPacket(data, 'key', tsS, durS);
    let m: EncodedAudioChunkMetadata | undefined;
    if (!this.sentAudioMeta) {
      const a = this.audioCfg;
      m = { decoderConfig: { codec: a.codecString, sampleRate: a.sampleRate, numberOfChannels: a.channels, description: a.description } };
      this.sentAudioMeta = true;
    }
    this.audioPackets++;
    this.payloadBytes += data.byteLength;
    return this.enqueue(() => this.audio!.add(pkt, m));
  }

  /** resolves when every queued mux op has finished */
  async drain() { await this.chain; if (this.failed) throw this.failed; }

  async finalize(): Promise<Blob | null> {
    await this.drain();
    await this.output.finalize();
    return this.sink.finish();
  }

  async cancel() {
    try { await this.output.cancel(); } catch { /* ignore */ }
    await this.sink.abort();
  }
}
