"""Bounded streaming decode of analysis frames (ENGINE v3; replaces the whole-clip luma cache).  Owner: ENGINE.

The M2 pipeline decoded every frame of the clip ONCE into a memmap cache at preview size and rendered every
closed-loop pass from it. That cache grows with clip length (960x720 luma of a 6.7-min 4:3 clip = 16.5 GB) and
it was written next to the analysis output (an iCloud-synced folder): it filled the disk and froze the app.

FrameStream instead decodes exactly the frames a pass needs, when it needs them, with a hard memory bound:

    fs = FrameStream(video, width=960)
    for k, luma in fs.frames(records, cancel=cancel):      # records: sorted frame indices (any subset)
        ...
    fs.close()                                              # releases the decoders (also on garbage collection)

* records are split into decode BLOCKS (<= `block` frames of the video timeline; gaps up to a GOP are decoded
  through) that `lanes` threads decode in parallel. One VideoToolbox H.264 stream of a 4K60 O3 clip decodes at
  only ~55-110 fps; 3 lanes give ~200+ fps, well above what the measurement consumes.
* Each lane owns ONE persistent decoder PROCESS for the whole analysis (PyAV + VideoToolbox; it seeks between
  blocks and continues without a seek when the next block follows on). Why a process per lane and not a
  process per block or an in-process decoder: on this Mac every VideoToolbox session leaves a
  VTDecoderXPCService (+ an IOSurface client) behind; the first version (one ffmpeg per 150-frame block,
  ~50 sessions per analysis) helped exhaust the IOSurface client table, after which new sessions take ~15 s
  to start and in-process VideoToolbox init ABORTS the process ("unable to open IOSurface kernel service").
  A decoder process isolates that crash: if it dies or does not deliver its first frame within
  `hw_timeout` s, the lane restarts it with software decoding and re-requests the block. Decoder processes
  exit with the stream (close()), and on their own when this process dies (parent watchdog).
  backend='ffmpeg': the old ffmpeg-per-block lanes (fallback when PyAV is missing).
* Luma = the decoded Y plane (10-bit sources reduced to 8 bits), area-downscaled to the analysis size.
* Decoded frames wait in a reorder buffer that never holds more than `capacity` frames: a lane blocks while its
  next frame lies `capacity` or more positions ahead of the consumer. Memory = capacity x frame bytes (default
  3x150+64 frames: ~270 MB at 960x540, ~360 MB at 960x720), independent of the clip length. No disk is used.
  Deadlock-free for any capacity >= 1: a lane only waits for frames AHEAD of the consumer.
* Closing the generator (or cancel() returning True, or an exception) stops every lane within ~0.2 s
  (a lane that is mid-block kills its decoder process; it is restarted on the next request).
"""
from __future__ import annotations

import math
import os
import threading
import time
from typing import Callable, Iterator, Optional

import numpy as np

__all__ = ['FrameStream', 'decode_blocks', 'StreamCancelled']


class StreamCancelled(InterruptedError):
    pass


class _Stop(Exception):
    pass


def decode_blocks(records: np.ndarray, max_gap: int, block: int) -> list:
    """Sorted unique frame indices -> [(start_frame, n_decode, positions (np.int64 indices into records))].
    Frames between wanted ones are decoded through when the gap is <= max_gap (a seek costs ~1 GOP)."""
    r = np.asarray(records, dtype=np.int64)
    out = []
    if len(r) == 0:
        return out
    i = 0
    n = len(r)
    while i < n:
        a = int(r[i])
        j = i
        # extend the block while the span stays <= block and the gaps stay small
        while j + 1 < n and r[j + 1] - r[j] <= max_gap + 1 and r[j + 1] - a < block:
            j += 1
        out.append((a, int(r[j] - a + 1), np.arange(i, j + 1, dtype=np.int64)))
        i = j + 1
    return out


def _to_luma(fr, w: int, h: int) -> np.ndarray:
    """PyAV frame -> uint8 (h, w) luma (area downscale; 10-bit codes -> 8 bits)."""
    import cv2
    p0 = fr.planes[0]
    name = fr.format.name
    if name in ('p010le', 'p010', 'yuv420p10le', 'p016le'):
        a = np.frombuffer(p0, np.uint16).reshape(fr.height, p0.line_size // 2)[:, :fr.width]
        g = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        if name == 'yuv420p10le':                    # LSB-aligned 10-bit
            return ((g.astype(np.uint32) + 2) >> 2).clip(0, 255).astype(np.uint8)
        return ((g.astype(np.uint32) + 128) >> 8).clip(0, 255).astype(np.uint8)
    a = np.frombuffer(p0, np.uint8).reshape(fr.height, p0.line_size)[:, :fr.width]
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)


class _AVLane:
    """One persistent PyAV (VideoToolbox) decoder. decode(a, n, want, emit) delivers the wanted frames of
    [a, a+n) (want: frame index -> position); emit(k, pos, img) raises _Stop to abort."""

    def __init__(self, video: str, info: dict, w: int, h: int, max_gap: int, hw: bool = True):
        self.video, self.info, self.w, self.h = video, info, w, h
        self.hw = hw
        self.pts = np.asarray(info['frame_pts'], dtype=np.float64)
        self.fps = float(info.get('fps') or 59.94)
        self.tol = 0.25 / self.fps
        self.max_gap = max_gap
        self.c = None
        self.it = None
        self.next_k = None          # frame index the iterator yields next (None: unknown -> seek)
        self.pending = None         # a frame read past the end of the previous block
        self.unmapped = 0

    def open(self):
        import av
        if self.hw:
            try:
                from av.codec.hwaccel import HWAccel
                self.c = av.open(self.video, hwaccel=HWAccel(device_type='videotoolbox', allow_software_fallback=True))
            except Exception:  # pragma: no cover - PyAV without hwaccel support
                self.c = av.open(self.video)
        else:
            self.c = av.open(self.video)
        self.vs = self.c.streams.video[0]
        self.vs.thread_type = 'AUTO'
        if not self.hw:
            self.vs.thread_count = 4
        else:
            # VideoToolbox does the decoding; FFmpeg's default frame threading (cpu_count + 1 = 15 threads) only adds
            # surfaces: 1.32 GB RSS per HEVC 10-bit 3840x2880 decoder process (0.54 GB for H.264 4K) vs 0.30 GB
            # (0.18 GB) with one thread -- and one thread decodes FASTER (106 vs 78 fps HEVC, 96 vs 73 fps H.264;
            # MEMORY role 2026-09-28). STILLPOINT_DECODER_THREADS overrides (0 = FFmpeg auto, the old behaviour).
            self.vs.thread_count = max(0, int(os.environ.get('STILLPOINT_DECODER_THREADS', '1') or 1))
        self.tb = float(self.vs.time_base)

    def close(self):
        if self.c is not None:
            try:
                self.c.close()
            except Exception:
                pass
        self.c = self.it = self.next_k = self.pending = None

    def _seek(self, k: int):
        t = self.pts[k] - 0.5 / self.fps
        self.c.seek(max(0, int(math.floor(t / self.tb))), stream=self.vs, backward=True, any_frame=False)
        self.it = self.c.decode(self.vs)
        self.next_k = None
        self.pending = None

    def _index(self, fr) -> Optional[int]:
        if fr.pts is None:
            return None
        t = fr.pts * self.tb
        j = int(np.searchsorted(self.pts, t - self.tol))
        if j < len(self.pts) and abs(self.pts[j] - t) <= self.tol:
            return j
        self.unmapped += 1
        return None

    def _frames(self):
        if self.pending is not None:
            item, self.pending = self.pending, None
            yield item
        for fr in self.it:
            k = self._index(fr)
            if k is not None:
                yield k, fr
        self.next_k = None                      # EOF: the next block seeks

    def decode(self, a: int, n: int, want: dict, emit, stop: threading.Event):
        if self.c is None:
            self.open()
        seq = (self.next_k is not None and self.next_k <= a and a - self.next_k <= self.max_gap)
        for attempt in range(2):
            if not seq:
                self._seek(a)
            got = set()
            for k, fr in self._frames():
                if stop.is_set():
                    raise _Stop()
                self.next_k = k + 1
                if k < a:
                    continue
                if k >= a + n:
                    self.pending = (k, fr)      # belongs to a later block
                    self.next_k = k
                    break
                p = want.get(k)
                if p is not None:
                    emit(k, p, _to_luma(fr, self.w, self.h))
                    got.add(k)
                if k == a + n - 1:
                    break
            missing = [k for k in want if k not in got]
            if not missing:
                return
            seq = False                         # retry once with an explicit seek
        raise RuntimeError(f'decoder did not deliver frames {missing[:5]}... of block [{a}, {a + n})')


def _decoder_main(conn, video: str, pts: np.ndarray, fps: float, keyframes, w: int, h: int, max_gap: int,
                  parent_pid: int, hw: bool):
    """Decoder process: serve block requests (a, n, wanted frame indices) with (k, luma) messages."""
    try:
        from .residual import _parent_watchdog
        _parent_watchdog(parent_pid)
    except Exception:
        pass
    import cv2
    cv2.setNumThreads(2)
    info = dict(frame_pts=pts, fps=fps, keyframes=keyframes)
    lane = _AVLane(video, info, w, h, max_gap, hw=hw)
    never = threading.Event()
    while True:
        try:
            req = conn.recv()
        except (EOFError, OSError):
            break
        if req is None:
            break
        a, n, keys = req
        want = {int(k): i for i, k in enumerate(keys)}
        try:
            def emit(k, p, img):
                conn.send_bytes(b'F' + int(k).to_bytes(8, 'little') + np.ascontiguousarray(img).tobytes())
            lane.decode(int(a), int(n), want, emit, never)
            conn.send_bytes(b'D')
        except (BrokenPipeError, EOFError, OSError):
            break
        except Exception as e:  # noqa: BLE001
            try:
                conn.send_bytes(b'E' + repr(e).encode('utf-8', 'replace'))
            except Exception:
                break
    lane.close()


class _DecoderDied(RuntimeError):
    pass


class _DecProc:
    """Main-side handle of one decoder process."""

    def __init__(self, fs: 'FrameStream', hw: bool):
        import multiprocessing as mp
        from .residual import _hide_main_file
        self.fs, self.hw = fs, hw
        ctx = mp.get_context('spawn')
        self.conn, child = ctx.Pipe(duplex=True)
        info = fs.info
        with _hide_main_file():
            self.proc = ctx.Process(target=_decoder_main, daemon=True, name='stillpoint-decoder',
                                    args=(child, fs.video, np.asarray(info['frame_pts'], np.float64),
                                          float(info.get('fps') or 59.94), np.asarray(info.get('keyframes', [])),
                                          fs.w, fs.h, max(fs.gop, 8), os.getpid(), hw))
            self.proc.start()
        child.close()
        self.first = True                      # the first frame of a new process includes VideoToolbox init

    def alive(self) -> bool:
        return self.proc.is_alive()

    def kill(self):
        try:
            self.proc.kill()
            self.proc.join(2.0)
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass

    def close(self):
        try:
            self.conn.send(None)
            self.proc.join(2.0)
        except Exception:
            pass
        if self.proc.is_alive():
            self.kill()
        else:
            try:
                self.conn.close()
            except Exception:
                pass

    def decode(self, a: int, n: int, want: dict, put, stop: threading.Event, hw_timeout: float,
               frame_timeout: float = 30.0):
        """Request block [a, a+n); put(pos, img) for every wanted frame. Raises _Stop (process killed),
        _DecoderDied (crash / timeout: caller falls back) or RuntimeError (decoder error)."""
        self.conn.send((int(a), int(n), list(want)))
        h, w = self.fs.h, self.fs.w
        t_last = time.perf_counter()
        remaining = len(want)
        while True:
            if stop.is_set():
                self.kill()
                raise _Stop()
            try:
                ready = self.conn.poll(0.2)
            except (EOFError, OSError) as e:
                raise _DecoderDied(f'decoder pipe closed: {e!r}') from e
            if not ready:
                lim = hw_timeout if (self.first and self.hw) else frame_timeout
                if not self.proc.is_alive():
                    raise _DecoderDied(f'decoder process exited (code {self.proc.exitcode})')
                if time.perf_counter() - t_last > lim:
                    self.kill()
                    raise _DecoderDied(f'decoder silent for {lim:.0f} s (hw={self.hw})')
                continue
            try:
                msg = self.conn.recv_bytes()
            except (EOFError, OSError) as e:
                raise _DecoderDied(f'decoder died: {e!r} (exit code {self.proc.exitcode})') from e
            t_last = time.perf_counter()
            tag = msg[:1]
            if tag == b'F':
                self.first = False
                k = int.from_bytes(msg[1:9], 'little')
                img = np.frombuffer(msg, np.uint8, offset=9).reshape(h, w)
                p = want.get(k)
                if p is not None:
                    put(p, img)
                    remaining -= 1
            elif tag == b'D':
                return
            elif tag == b'E':
                raise RuntimeError('decoder error: ' + msg[1:].decode('utf-8', 'replace'))



class FrameStream:
    def __init__(self, video: str, width: int = 960, lanes: int = 3, block: int = 150,
                 capacity: Optional[int] = None, info: Optional[dict] = None, backend: str = 'auto',
                 hw: bool = True, hw_timeout: float = 30.0, modes=('vt_cpu_scale', 'sw')):
        from .video import gray_size, probe
        self.video = video
        self.info = info or probe(video)
        self.n = int(self.info['n_frames'])
        self.w, self.h = gray_size(self.info['width'], self.info['height'], width)
        self.lanes = max(1, int(lanes))
        self.block = max(8, int(block))
        self.capacity = int(capacity) if capacity else self.lanes * self.block + 64
        self.modes = tuple(modes)
        kf = np.asarray(self.info.get('keyframes', []))
        self.gop = int(np.median(np.diff(kf))) if len(kf) > 2 else 60
        if os.environ.get('STILLPOINT_DECODE_HW', '1').strip().lower() in ('0', 'false', 'no', 'off'):
            hw = False                         # e.g. while the VideoToolbox/IOSurface client table is exhausted
        self.hw_ok = bool(hw)                  # VideoToolbox usable (cleared after a crash / init timeout)
        self.hw_timeout = float(hw_timeout)
        self._dec: dict = {}
        self._lock = threading.Lock()
        self.backend = backend
        if backend in ('auto', 'proc'):
            try:
                import av  # noqa: F401
                self.backend = 'proc'
            except Exception:
                if backend == 'proc':
                    raise
                self.backend = 'ffmpeg'
        self.stats = dict(frames=0, decoded=0, blocks=0, decode_s=0.0, wait_s=0.0, max_buffered=0,
                          decoder_starts=0, hw_fallbacks=0, backend=self.backend)

    @property
    def frame_bytes(self) -> int:
        return self.w * self.h

    def max_buffer_bytes(self) -> int:
        return self.capacity * self.frame_bytes

    def close(self):
        with self._lock:
            decs, self._dec = list(self._dec.values()), {}
        for d in decs:
            d.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def decoder_pids(self) -> list:
        return [d.proc.pid for d in list(self._dec.values()) if d.proc.pid]

    def _decoder(self, li: int) -> _DecProc:
        with self._lock:
            d = self._dec.get(li)
            if d is not None and (not d.alive() or d.hw != self.hw_ok):
                d.kill()
                d = None
            if d is None:
                d = _DecProc(self, self.hw_ok)
                self._dec[li] = d
                self.stats['decoder_starts'] += 1
            return d

    def _drop(self, li: int):
        with self._lock:
            d = self._dec.pop(li, None)
        if d is not None:
            d.kill()

    def frames(self, records, cancel: Optional[Callable[[], bool]] = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, uint8 (h,w) luma) for the sorted unique `records`, in order."""
        recs = np.asarray(records, dtype=np.int64).reshape(-1)
        if len(recs) == 0:
            return
        if np.any(np.diff(recs) <= 0):
            raise ValueError('FrameStream.frames: records must be sorted and unique')
        if recs[0] < 0 or recs[-1] >= self.n:
            raise IndexError('frame index out of range')
        blocks = decode_blocks(recs, max(self.gop, 8), self.block)
        n = len(recs)
        buf: dict = {}
        cond = threading.Condition()
        state = dict(front=0, err=None)
        stop = threading.Event()
        cap = self.capacity
        st = self.stats

        def put(p: int, img: np.ndarray):
            with cond:
                if p < state['front'] or p in buf:          # a retried block re-delivers frames: keep the first
                    return
                while p >= state['front'] + cap and not stop.is_set():
                    cond.wait(0.2)
                if stop.is_set():
                    raise _Stop()
                buf[p] = img
                if len(buf) > st['max_buffered']:
                    st['max_buffered'] = len(buf)
                cond.notify_all()

        def lane_proc(li: int):
            for bi in range(li, len(blocks), self.lanes):
                if stop.is_set():
                    return
                a, cnt, pos = blocks[bi]
                want = dict(zip(recs[pos].tolist(), pos.tolist()))
                t0 = time.perf_counter()
                for attempt in range(3):
                    dec = self._decoder(li)
                    try:
                        dec.decode(a, cnt, want, put, stop, self.hw_timeout)
                        break
                    except _Stop:
                        self._drop(li)
                        raise
                    except _DecoderDied:
                        self._drop(li)
                        if dec.hw:                 # VideoToolbox crashed / hung: software from now on
                            with cond:
                                st['hw_fallbacks'] += 1
                            self.hw_ok = False
                        elif attempt == 2:
                            raise
                with cond:
                    st['decoded'] += cnt
                    st['blocks'] += 1
                    st['decode_s'] += time.perf_counter() - t0

        def lane_ffmpeg(li: int):
            from .video import DecodeError, _run_decode
            for bi in range(li, len(blocks), self.lanes):
                if stop.is_set():
                    return
                a, cnt, pos = blocks[bi]
                want = dict(zip(recs[pos].tolist(), pos.tolist()))
                k_next = a
                t0 = time.perf_counter()
                for mode in self.modes:
                    try:
                        with cond:
                            st['decoder_starts'] += 1
                        for kk, _, img in _run_decode(self.video, self.info, self.w, self.h, mode, k_next,
                                                      a + cnt - k_next, True):
                            k_next = kk + 1
                            p = want.get(kk)
                            if p is None:
                                if stop.is_set():
                                    return
                                continue
                            put(p, img)
                        break
                    except DecodeError:
                        if mode == self.modes[-1] or stop.is_set():
                            raise
                        continue            # resume from the next undecoded frame with the next mode
                with cond:
                    st['decoded'] += cnt
                    st['blocks'] += 1
                    st['decode_s'] += time.perf_counter() - t0

        def lane(li: int):
            try:
                (lane_proc if self.backend == 'proc' else lane_ffmpeg)(li)
            except _Stop:
                return
            except BaseException as e:  # noqa: BLE001 - hand to the consumer
                with cond:
                    if state['err'] is None:
                        state['err'] = e
                    cond.notify_all()

        threads = [threading.Thread(target=lane, args=(i,), daemon=True, name=f'framestream-{i}')
                   for i in range(min(self.lanes, len(blocks)))]
        for th in threads:
            th.start()
        try:
            for p in range(n):
                t0 = time.perf_counter()
                with cond:
                    while p not in buf:
                        if state['err'] is not None:
                            raise RuntimeError(f'frame decode failed: {state["err"]!r}') from state['err']
                        if cancel is not None and cancel():
                            raise StreamCancelled('decode cancelled')
                        cond.wait(0.2)
                    img = buf.pop(p)
                    state['front'] = p + 1
                    cond.notify_all()
                st['wait_s'] += time.perf_counter() - t0
                st['frames'] += 1
                yield int(recs[p]), img
        finally:
            stop.set()
            with cond:
                cond.notify_all()
            for th in threads:
                th.join(timeout=10)
            buf.clear()
