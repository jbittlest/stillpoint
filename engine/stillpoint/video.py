"""Video probing and fast analysis-frame decoding (WP-A).

  probe(path)            -> dict with exact per-frame PTS (from the MP4 sample tables, no packet reads),
                            stream geometry, codec, pix_fmt and colour tags.
  iter_gray(path, ...)   -> yields (frame_index, pts, uint8 HxW) using ffmpeg VideoToolbox decode + scale_vt
                            (falls back to CPU scale), `-fps_mode passthrough`. Every yielded frame's PTS is read
                            back from ffmpeg (showinfo) and checked against probe()['frame_pts'], so frame indices
                            are exact (a mismatch raises instead of silently shifting).
  read_gray_frames(path, idx, width) -> (N,H,W) uint8, random access (clusters nearby indices into one decode run).

MP4 sample tables are parsed directly (`mp4_tracks`), which also lets telemetry.py read the DJI `djmd`/`dbgi`
samples by file offset (`read_ranges`: unbuffered pread of exactly those bytes; adjacent samples merge into one read).
Why unbuffered matters: Python's buffered reader fills st_blksize bytes per seek+read, and on Jimmy's exFAT SD card
st_blksize is 1 MiB while DJI interleaves one ~0.5 KB djmd sample per ~245 KB video frame -- so the old seek+read
loop pulled (almost) the whole 6-8 GB file through the card reader, twice (djmd, then dbgi).
Measured on that card (built-in SDXC reader, fskit exFAT): a random small read costs ~0.4 ms (~2.5k IOPS; parallel
reads, F_NOCACHE or F_RDADVISE do not help), sequential ~92 MB/s.
"""
from __future__ import annotations

import json
import os
import queue
import re
import struct
import subprocess
import sys
import threading
from fractions import Fraction
from typing import Iterator

import numpy as np

FFMPEG = os.environ.get('STILLPOINT_FFMPEG', '/opt/homebrew/bin/ffmpeg')
FFPROBE = os.environ.get('STILLPOINT_FFPROBE', '/opt/homebrew/bin/ffprobe')
if not os.path.exists(FFMPEG):
    FFMPEG = 'ffmpeg'
if not os.path.exists(FFPROBE):
    FFPROBE = 'ffprobe'

# ============================================================================ MP4 box / sample-table parser

_CONTAINERS = {b'moov', b'trak', b'mdia', b'minf', b'stbl', b'edts', b'dinf', b'udta'}


def _iter_boxes(f, start: int, end: int):
    """Yield (type, payload_start, box_end) for boxes in [start, end) of an open binary file."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        hdr = f.read(16)
        if len(hdr) < 8:
            return
        size, typ = struct.unpack('>I4s', hdr[:8])
        hlen = 8
        if size == 1:
            size = struct.unpack('>Q', hdr[8:16])[0]
            hlen = 16
        elif size == 0:
            size = end - pos
        if size < hlen:
            return
        yield typ, pos + hlen, min(pos + size, end)
        pos += size


class Mp4Track:
    """Sample table of one MP4 track (all arrays in decode order)."""

    def __init__(self):
        self.track_id = 0
        self.handler = ''          # 'vide', 'soun', 'meta', 'data', ...
        self.handler_name = ''
        self.fourcc = ''           # first stsd entry, e.g. 'avc1', 'hvc1', 'djmd'
        self.timescale = 1
        self.width = 0
        self.height = 0
        self.dts = np.zeros(0, np.int64)       # media timescale ticks
        self.cts_off = np.zeros(0, np.int64)
        self.sizes = np.zeros(0, np.int64)
        self.offsets = np.zeros(0, np.int64)
        self.sync = None                        # np bool array or None (= all sync)
        self.elst = []                          # [(segment_duration_movie_ts, media_time, rate)]
        self.movie_timescale = 1

    @property
    def n_samples(self) -> int:
        return int(len(self.sizes))

    def pts_ticks(self) -> np.ndarray:
        """Presentation time in media ticks after the edit list (ffmpeg semantics for the simple
        cases: leading empty edits delay, first media_time shifts)."""
        shift = 0
        for dur, mt, _rate in self.elst:
            if mt == -1:
                shift += int(round(dur * self.timescale / max(self.movie_timescale, 1)))
            else:
                shift -= mt
                break
        return self.dts + self.cts_off + shift

    def pts_seconds(self) -> np.ndarray:
        return self.pts_ticks().astype(np.float64) / float(self.timescale)

    def sample_ranges(self, idx=None) -> tuple[np.ndarray, np.ndarray]:
        """(file offsets, sizes) of samples idx (all when None), in the order of idx."""
        if len(self.offsets) != self.n_samples:
            raise ValueError(f'track {self.track_id} ({self.fourcc}): sample table has no chunk offsets')
        if idx is None:
            return self.offsets, self.sizes
        idx = np.asarray(idx, np.int64).reshape(-1)
        return self.offsets[idx], self.sizes[idx]

    def read_samples(self, path: str, idx=None) -> list[bytes]:
        """Bytes of samples idx (all when None), reading only those bytes (see read_ranges)."""
        return read_ranges(path, *self.sample_ranges(idx))


# ============================================================================ byte-range reads

READ_MERGE_GAP = 16 * 1024     # merge ranges closer than this into one read (below the cost of one extra random read)
_F_RDAHEAD = 45                # <sys/fcntl.h> (Darwin): turn off speculative read-ahead for this fd


def _open_ranges_fd(path: str) -> int:
    fd = os.open(path, os.O_RDONLY)
    if sys.platform == 'darwin':
        try:
            import fcntl
            fcntl.fcntl(fd, _F_RDAHEAD, 0)     # random small reads: read-ahead would only fetch video bytes
        except OSError:
            pass
    return fd


def _pread_full(fd: int, n: int, off: int) -> bytes:
    """pread n bytes at off (short only at EOF, like file.read)."""
    b = os.pread(fd, n, off)
    if len(b) == n or not b:
        return b
    parts = [b]
    got = len(b)
    while got < n:
        c = os.pread(fd, n - got, off + got)
        if not c:
            break
        parts.append(c)
        got += len(c)
    return b''.join(parts)


def _read_coalesced(fd: int, offsets: np.ndarray, sizes: np.ndarray, max_gap: int) -> list[bytes]:
    """Read ranges (in the given order) with one pread per group of ranges that lie within max_gap of each other."""
    n = len(offsets)
    if n == 0:
        return []
    order = np.argsort(offsets, kind='stable')
    o = offsets[order]
    e = o + sizes[order]
    reach = np.maximum.accumulate(e)
    new = np.ones(n, bool)
    new[1:] = o[1:] - reach[:-1] > max_gap
    run_of = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    lo = o[starts]
    hi = np.maximum.reduceat(e, starts)
    bufs = [_pread_full(fd, int(h - l), int(l)) for l, h in zip(lo.tolist(), hi.tolist())]
    out: list = [None] * n
    rel = (o - lo[run_of]).tolist()
    sz = sizes[order].tolist()
    for j, (i, r) in enumerate(zip(order.tolist(), run_of.tolist())):
        out[i] = bufs[r][rel[j]:rel[j] + sz[j]]
    return out


def read_ranges(path: str, offsets, sizes, max_gap: int = READ_MERGE_GAP) -> list[bytes]:
    """Bytes of each (offset, size) range, in the order given; reads only those bytes (plus gaps < max_gap)
    with unbuffered pread -- no read-ahead, no st_blksize buffer fills."""
    offsets = np.asarray(offsets, np.int64).reshape(-1)
    sizes = np.asarray(sizes, np.int64).reshape(-1)
    if len(offsets) != len(sizes):
        raise ValueError('offsets/sizes length mismatch')
    fd = _open_ranges_fd(path)
    try:
        return _read_coalesced(fd, offsets, sizes, max_gap)
    finally:
        os.close(fd)


def _parse_stbl(f, a, b, tr: Mp4Track):
    stts = ctts = stsz = stsc = chunk_off = stss = None
    for typ, p, e in _iter_boxes(f, a, b):
        f.seek(p)
        data = f.read(e - p)
        if typ == b'stsd':
            n = struct.unpack('>I', data[4:8])[0]
            if n:
                tr.fourcc = data[12:16].decode('latin-1')
                if tr.handler == 'vide' and len(data) >= 8 + 8 + 36:
                    # VisualSampleEntry: 6 reserved + 2 dref + 16 pre-defined, then width/height u16
                    tr.width, tr.height = struct.unpack('>HH', data[8 + 8 + 24: 8 + 8 + 28])
        elif typ == b'stts':
            n = struct.unpack('>I', data[4:8])[0]
            stts = np.frombuffer(data[8:8 + 8 * n], '>u4').reshape(n, 2).astype(np.int64)
        elif typ == b'ctts':
            ver = data[0]
            n = struct.unpack('>I', data[4:8])[0]
            arr = np.frombuffer(data[8:8 + 8 * n], '>u4').reshape(n, 2).astype(np.int64)
            if ver == 1:
                arr[:, 1] = np.where(arr[:, 1] >= 2 ** 31, arr[:, 1] - 2 ** 32, arr[:, 1])
            ctts = arr
        elif typ == b'stsz':
            ssize, n = struct.unpack('>II', data[4:12])
            stsz = np.full(n, ssize, np.int64) if ssize else np.frombuffer(data[12:12 + 4 * n], '>u4').astype(np.int64)
        elif typ == b'stsc':
            n = struct.unpack('>I', data[4:8])[0]
            stsc = np.frombuffer(data[8:8 + 12 * n], '>u4').reshape(n, 3).astype(np.int64)
        elif typ == b'stco':
            n = struct.unpack('>I', data[4:8])[0]
            chunk_off = np.frombuffer(data[8:8 + 4 * n], '>u4').astype(np.int64)
        elif typ == b'co64':
            n = struct.unpack('>I', data[4:8])[0]
            chunk_off = np.frombuffer(data[8:8 + 8 * n], '>u8').astype(np.int64)
        elif typ == b'stss':
            n = struct.unpack('>I', data[4:8])[0]
            stss = np.frombuffer(data[8:8 + 4 * n], '>u4').astype(np.int64)
    if stsz is None or stts is None:
        return
    ns = len(stsz)
    tr.sizes = stsz
    deltas = np.repeat(stts[:, 1], stts[:, 0])[:ns]
    tr.dts = np.concatenate([[0], np.cumsum(deltas)[:-1]]).astype(np.int64) if ns else np.zeros(0, np.int64)
    tr.cts_off = np.repeat(ctts[:, 1], ctts[:, 0])[:ns] if ctts is not None else np.zeros(ns, np.int64)
    if len(tr.cts_off) < ns:
        tr.cts_off = np.concatenate([tr.cts_off, np.zeros(ns - len(tr.cts_off), np.int64)])
    if stss is not None:
        tr.sync = np.zeros(ns, bool)
        tr.sync[stss[(stss >= 1) & (stss <= ns)] - 1] = True
    # sample -> chunk -> file offset
    if chunk_off is not None and stsc is not None and len(stsc):
        nch = len(chunk_off)
        first = stsc[:, 0] - 1
        last = np.concatenate([first[1:], [nch]])
        per_chunk = np.zeros(nch, np.int64)
        for fc, lc, spc in zip(first, last, stsc[:, 1]):
            per_chunk[fc:lc] = spc
        chunk_of_sample = np.repeat(np.arange(nch), per_chunk)[:ns]
        csum = np.concatenate([[0], np.cumsum(stsz)])
        first_sample_of_chunk = np.concatenate([[0], np.cumsum(per_chunk)[:-1]])
        within = csum[:ns] - csum[first_sample_of_chunk[chunk_of_sample]]
        tr.offsets = chunk_off[chunk_of_sample] + within


_TRACKS_CACHE: dict = {}


def mp4_tracks(path: str) -> list[Mp4Track]:
    """Parse every track's sample table of an MP4/MOV file (reads only the moov box). Cached per process by
    (path, size, mtime); the returned tracks are shared -- treat them as read-only."""
    path = os.path.abspath(path)
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    tr = _TRACKS_CACHE.get(key)
    if tr is None:
        tr = _parse_mp4_tracks(path)
        _TRACKS_CACHE.clear()          # keep one file's tables (a 21 GB joined clip's are ~10 MB)
        _TRACKS_CACHE[key] = tr
    return list(tr)


def _parse_mp4_tracks(path: str) -> list[Mp4Track]:
    tracks = []
    fsize = os.path.getsize(path)
    with open(path, 'rb') as f:
        moov = None
        for typ, p, e in _iter_boxes(f, 0, fsize):
            if typ == b'moov':
                moov = (p, e)
                break
        if moov is None:
            raise ValueError(f'no moov box in {path}')
        movie_ts = 1
        for typ, p, e in _iter_boxes(f, *moov):
            if typ == b'mvhd':
                f.seek(p)
                d = f.read(32)
                movie_ts = struct.unpack('>I', d[20:24] if d[0] == 1 else d[12:16])[0]
        for typ, p, e in _iter_boxes(f, *moov):
            if typ != b'trak':
                continue
            tr = Mp4Track()
            tr.movie_timescale = movie_ts
            stbl = None
            for t2, p2, e2 in _iter_boxes(f, p, e):
                if t2 == b'tkhd':
                    f.seek(p2)
                    d = f.read(24)
                    tr.track_id = struct.unpack('>I', d[20:24] if d[0] == 1 else d[12:16])[0]
                elif t2 == b'edts':
                    for t3, p3, e3 in _iter_boxes(f, p2, e2):
                        if t3 == b'elst':
                            f.seek(p3)
                            d = f.read(e3 - p3)
                            ver, n = d[0], struct.unpack('>I', d[4:8])[0]
                            for i in range(n):
                                if ver == 1:
                                    dur, mt = struct.unpack('>Qq', d[8 + 20 * i: 8 + 20 * i + 16])
                                    ri, rf = struct.unpack('>hh', d[8 + 20 * i + 16: 8 + 20 * i + 20])
                                else:
                                    dur, mt = struct.unpack('>Ii', d[8 + 12 * i: 8 + 12 * i + 8])
                                    ri, rf = struct.unpack('>hh', d[8 + 12 * i + 8: 8 + 12 * i + 12])
                                tr.elst.append((dur, mt, ri + rf / 65536.0))
                elif t2 == b'mdia':
                    for t3, p3, e3 in _iter_boxes(f, p2, e2):
                        f.seek(p3)
                        if t3 == b'mdhd':
                            d = f.read(32)
                            tr.timescale = struct.unpack('>I', d[20:24] if d[0] == 1 else d[12:16])[0]
                        elif t3 == b'hdlr':
                            d = f.read(e3 - p3)
                            tr.handler = d[8:12].decode('latin-1')
                            tr.handler_name = d[24:].split(b'\x00')[0].decode('utf-8', 'replace').strip()
                        elif t3 == b'minf':
                            for t4, p4, e4 in _iter_boxes(f, p3, e3):
                                if t4 == b'stbl':
                                    stbl = (p4, e4)
            if stbl is not None:
                _parse_stbl(f, stbl[0], stbl[1], tr)
            tracks.append(tr)
    return tracks


def find_track(tracks: list[Mp4Track], fourcc: str | None = None, handler_name: str | None = None,
               handler: str | None = None) -> Mp4Track | None:
    for t in tracks:
        if fourcc is not None and t.fourcc == fourcc:
            return t
    for t in tracks:
        if handler_name is not None and t.handler_name == handler_name:
            return t
    for t in tracks:
        if handler is not None and t.handler == handler and t.n_samples > 1:
            return t
    return None


def main_video_track(tracks: list[Mp4Track]) -> Mp4Track:
    vids = [t for t in tracks if t.handler == 'vide' and t.fourcc in ('avc1', 'avc3', 'hvc1', 'hev1', 'apcn',
                                                                         'apch', 'apcs', 'ap4h', 'av01', 'mp4v')]
    if not vids:
        vids = [t for t in tracks if t.handler == 'vide']
    if not vids:
        raise ValueError('no video track')
    return max(vids, key=lambda t: (t.width * t.height, t.n_samples))


# ============================================================================ probe

_PROBE_CACHE: dict = {}


def _ffprobe_stream(path: str) -> dict:
    cmd = [FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
           'stream=index,codec_name,codec_tag_string,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,'
           'time_base,start_pts,nb_frames,color_range,color_space,color_primaries,color_transfer,chroma_location,'
           'bits_per_raw_sample:format=start_time,duration,format_name:format_tags=comment,encoder',
           '-of', 'json', path]
    j = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    return j


def probe(path: str) -> dict:
    """Exact per-frame PTS + stream info.

    frame_pts: float64 seconds, presentation order, one entry per decodable video frame, on ffmpeg's
    timeline (edit list applied). Taken from the MP4 sample tables (no packet reads, instant even on 21 GB);
    non-MP4 inputs fall back to ffprobe packet PTS.
    """
    path = os.path.abspath(path)
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key in _PROBE_CACHE:
        return dict(_PROBE_CACHE[key])
    j = _ffprobe_stream(path)
    s = j['streams'][0]
    fmt = j.get('format', {})
    info = dict(path=path, width=int(s['width']), height=int(s['height']), codec=s.get('codec_name'),
                codec_tag=s.get('codec_tag_string'), profile=s.get('profile'), pix_fmt=s.get('pix_fmt'),
                color_range=s.get('color_range'), color_space=s.get('color_space'),
                color_primaries=s.get('color_primaries'), color_transfer=s.get('color_transfer'),
                chroma_location=s.get('chroma_location'), stream_index=int(s['index']),
                r_frame_rate=s.get('r_frame_rate'), avg_frame_rate=s.get('avg_frame_rate'),
                time_base=s.get('time_base'), format_start_time=float(fmt.get('start_time', 0.0) or 0.0),
                duration=float(fmt.get('duration', 0.0) or 0.0), format_tags=fmt.get('tags', {}))
    bit_depth = 10 if (info['pix_fmt'] or '').find('10') >= 0 else 8
    info['bit_depth'] = bit_depth
    pts = None
    keyframes = None
    try:
        tracks = mp4_tracks(path)
        vt = main_video_track(tracks)
        order = np.argsort(vt.pts_ticks(), kind='stable')
        ticks = vt.pts_ticks()[order]
        pts = ticks.astype(np.float64) / vt.timescale
        info['timescale'] = vt.timescale
        info['frame_pts_ticks'] = ticks
        if vt.sync is not None:
            # keyframe flags in presentation order
            keyframes = np.flatnonzero(vt.sync[order])
        else:
            keyframes = np.arange(len(pts))
        info['pts_source'] = 'mp4_stbl'
    except Exception as e:  # not an MP4 or odd layout -> ffprobe packets (reads the whole file)
        info['pts_source'] = f'ffprobe_packets ({type(e).__name__}: {e})'
        cmd = [FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts,flags',
               '-of', 'csv=p=0', path]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()
        rows = [ln.split(',') for ln in out if ln and ln.split(',')[0] not in ('N/A', '')]
        tb = Fraction(info['time_base'])
        tk = np.array(sorted(int(r[0]) for r in rows), np.int64)
        pts = tk * float(tb)
        info['frame_pts_ticks'] = tk
        info['timescale'] = tb.denominator if tb.numerator == 1 else None
        kf = sorted(int(r[0]) for r in rows if 'K' in (r[1] if len(r) > 1 else ''))
        keyframes = np.searchsorted(tk, np.array(kf, np.int64))
    info['frame_pts'] = pts
    info['n_frames'] = int(len(pts))
    info['keyframes'] = np.asarray(keyframes, np.int64)
    fr = Fraction(info['avg_frame_rate'] if info['avg_frame_rate'] not in (None, '0/0') else info['r_frame_rate'])
    info['fps'] = float(fr)
    info['fps_fraction'] = f'{fr.numerator}/{fr.denominator}'
    if len(pts) > 1:
        d = np.diff(pts)
        info['cfr'] = bool(np.all(np.abs(d - d[0]) < 1e-6))
    else:
        info['cfr'] = True
    for k in ('frame_pts', 'frame_pts_ticks', 'keyframes'):
        info[k].flags.writeable = False          # shared with the probe cache
    _PROBE_CACHE[key] = dict(info)
    return info


# ============================================================================ decoding

def gray_size(width_src: int, height_src: int, width: int) -> tuple[int, int]:
    """Analysis-frame size for a requested width (even dims, aspect preserved)."""
    w = int(width) // 2 * 2
    h = int(round(height_src * w / width_src / 2.0)) * 2
    return w, h


def _vf_chain(mode: str, w: int, h: int, bit_depth: int) -> tuple[list[str], str]:
    """Return (input args, -vf string) for the decode mode."""
    # extractplanes=y keeps the raw luma codes (a plain `format=gray` would range-expand tv->pc and clip the
    # 236-255 / 0-15 codes DJI files do use); 10-bit luma is reduced to 8 bits by a plain shift.
    if mode == 'vt_scale_vt':
        dl = 'p010le' if bit_depth > 8 else 'nv12'
        return (['-hwaccel', 'videotoolbox', '-hwaccel_output_format', 'videotoolbox_vld'],
                f'scale_vt=w={w}:h={h},hwdownload,format={dl},extractplanes=y,format=gray,showinfo')
    if mode == 'vt_cpu_scale':
        return (['-hwaccel', 'videotoolbox'], f'extractplanes=y,scale={w}:{h}:flags=area,format=gray,showinfo')
    return ([], f'extractplanes=y,scale={w}:{h}:flags=area,format=gray,showinfo')


_SHOWINFO_RE = re.compile(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(\S+)')
_TB_RE = re.compile(r'config in time_base:\s*(\d+)/(\d+)')


def _stderr_reader(pipe, q: 'queue.Queue', log: list):
    tb = None
    try:
        for raw in iter(pipe.readline, b''):
            line = raw.decode('utf-8', 'replace')
            if 'Parsed_showinfo' in line:
                m = _TB_RE.search(line)
                if m:
                    tb = Fraction(int(m.group(1)), int(m.group(2)))
                    continue
                m = _SHOWINFO_RE.search(line)
                if m:
                    q.put((int(m.group(1)), int(m.group(2)), tb))
                    continue
            if len(log) < 400:
                log.append(line.rstrip())
    finally:
        q.put(None)


class DecodeError(RuntimeError):
    pass


def _run_decode(path: str, info: dict, w: int, h: int, mode: str, start_frame: int, n: int,
                verify: bool) -> Iterator[tuple[int, float, np.ndarray]]:
    pts_all = info['frame_pts']
    fps = info['fps']
    in_args, vf = _vf_chain(mode, w, h, info['bit_depth'])
    cmd = [FFMPEG, '-hide_banner', '-nostdin', '-loglevel', 'info', '-threads', '4'] + in_args
    if start_frame > 0:
        # Accurate input seek (ffmpeg decodes from the previous keyframe and drops frames before -ss).
        # -ss is relative to the file start time; aim half a frame before the target PTS.
        ss = pts_all[start_frame] - info['format_start_time'] - 0.5 / fps
        cmd += ['-ss', f'{max(ss, 0.0):.6f}']
    cmd += ['-copyts', '-i', info['path'], '-map', f"0:{info['stream_index']}", '-an', '-sn', '-dn',
            '-vf', vf, '-fps_mode', 'passthrough', '-frames:v', str(n), '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    q: queue.Queue = queue.Queue()
    log: list = []
    th = threading.Thread(target=_stderr_reader, args=(proc.stderr, q, log), daemon=True)
    th.start()
    fb = w * h
    got = 0
    tol = 0.25 / fps
    try:
        while got < n:
            buf = bytearray(fb)
            mv = memoryview(buf)
            r = 0
            while r < fb:
                k = proc.stdout.readinto(mv[r:])
                if not k:
                    break
                r += k
            if r < fb:
                break
            item = q.get(timeout=60)
            if item is None:
                raise DecodeError(f'ffmpeg frame/showinfo desync at frame {got}: ' + '\n'.join(log[-15:]))
            _n, pts_i, tb = item
            if tb is None:
                raise DecodeError('showinfo time base not found')
            t = float(pts_i * tb) + 0.0
            k = start_frame + got
            # map ffmpeg PTS back onto probe's frame list (copyts keeps the container timeline)
            if verify and abs(t - pts_all[k]) > tol:
                j = int(np.argmin(np.abs(pts_all - t)))
                raise DecodeError(f'frame index misalignment: expected frame {k} (pts {pts_all[k]:.6f}) but '
                                  f'ffmpeg delivered pts {t:.6f} (= frame {j}); mode={mode}')
            yield k, float(pts_all[k]), np.frombuffer(buf, np.uint8).reshape(h, w)
            got += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()
        th.join(timeout=5)
        proc.stderr.close()
    if got < n:
        raise DecodeError(f'ffmpeg ({mode}) delivered {got}/{n} frames: ' + '\n'.join(log[-15:]))


_MODE_OK: dict = {}


def iter_gray(path: str, width: int = 960, start_frame: int = 0, n_frames: int | None = None,
              verify: bool = True) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, pts, uint8 HxW luma) for frames [start_frame, start_frame+n_frames).

    Luma is the raw Y plane (limited-range codes, 10-bit sources shifted to 8 bits), downscaled to `width`
    (height keeps the aspect ratio, even). Analysis-only: no colour conversion.

    Uses ffmpeg -hwaccel videotoolbox + scale_vt (GPU downscale), falling back to CPU scale and then to
    software decode if a mode fails before delivering its first frame. -fps_mode passthrough (no dup/drop).
    Every frame's PTS is read back (showinfo) and must equal probe()['frame_pts'][index] (verify=True).
    """
    info = probe(path)
    F = info['n_frames']
    start_frame = int(start_frame)
    if not 0 <= start_frame < F:
        raise IndexError(f'start_frame {start_frame} out of range [0, {F})')
    n = F - start_frame if n_frames is None else int(min(n_frames, F - start_frame))
    if n <= 0:
        return
    w, h = gray_size(info['width'], info['height'], width)
    modes = ['vt_scale_vt', 'vt_cpu_scale', 'sw']
    if info['path'] in _MODE_OK:
        modes = [_MODE_OK[info['path']]] + [m for m in modes if m != _MODE_OK[info['path']]]
    last_err = None
    for mode in modes:
        delivered = 0
        try:
            for item in _run_decode(path, info, w, h, mode, start_frame, n, verify):
                delivered += 1
                _MODE_OK[info['path']] = mode
                yield item
            return
        except DecodeError as e:
            if delivered:
                raise
            last_err = e
            continue
    raise DecodeError(f'all decode modes failed for {path}: {last_err}')


def read_gray_frames(path: str, idx, width: int = 960, max_gap: int | None = None) -> np.ndarray:
    """Random access: frames idx (any order, duplicates allowed) -> (N,H,W) uint8 in the order of idx.

    Nearby requests are served from one sequential decode run; a new (accurate) seek is started when the gap to
    the next wanted frame exceeds max_gap (default: ~1 GOP)."""
    info = probe(path)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    w, h = gray_size(info['width'], info['height'], width)
    out = np.empty((len(idx), h, w), np.uint8)
    if len(idx) == 0:
        return out
    if idx.min() < 0 or idx.max() >= info['n_frames']:
        raise IndexError('frame index out of range')
    if max_gap is None:
        kf = info['keyframes']
        gop = int(np.median(np.diff(kf))) if len(kf) > 2 else 60
        max_gap = max(gop, 8)
    uniq = np.unique(idx)
    runs = []
    s = uniq[0]
    prev = uniq[0]
    for v in uniq[1:]:
        if v - prev > max_gap:
            runs.append((s, prev))
            s = v
        prev = v
    runs.append((s, prev))
    frames = {}
    want = set(uniq.tolist())
    for a, b in runs:
        for k, _t, g in iter_gray(path, width, start_frame=int(a), n_frames=int(b - a + 1)):
            if k in want:
                frames[k] = g
    for i, k in enumerate(idx):
        out[i] = frames[int(k)]
    return out
