"""DJI telemetry -> stillpoint.types.Telemetry (WP-A).

Supported:
  * DJI O3 Air Unit (camera "DJI FC8383", djmd schema dvtm_wm169.proto): fused attitude quaternions at 2 kHz,
    embedded KB4 lens, readout, exposure. Joined files (several clips concatenated) are split into segments.
  * DJI Osmo Action 4 ("DJI OsmoAction4", dvtm_ac203.proto): 1 kHz fused attitude in 4:3 EIS-off clips;
    otherwise only the per-frame (59.94 Hz) gravity-aligned camera attitude, which is sampled ~18 ms after the
    picture. dbgi EIS attitudes (4:3 clips) are kept in `extra`.

Timing model (research/footage_o3.md §2.2, research/footage_oa4.md §5) -- all on the CAMERA clock first:
  T_n           FrameMetaHeader.frame_timestamp of frame n (us -> s)
  block sample  s_{n,i} = T_n + (i - offset_n)/rate_nominal       (DeviceAttitude, i = 0..count_n-1)
  The blocks tile one gap-free uniform series. We fit, per continuous segment,
        T_n - offset_n/rate = a + N_n*dt + beta*exposure_n          (N_n = global index of block n's first sample)
  O3: beta = 0.5 (DJI folds -exposure/2 into `offset`; fitted 0.49-0.52 on all O3 clips, residual 10-26 us rms).
  O4 Pro (dvtm_O4P, seen in ~/Desktop/DJI_20260925*): same, fitted beta = 0.5001, residual 9 us rms. Its "2 kHz"
       stream is a 1 kHz stream with every sample written twice (sample 2j+1 == sample 2j bit for bit): the parser
       keeps one sample per pair, at the first copy's grid time + O4P_HOLD_SAMPLE_DELAY_S (vision-measured).
  OA4: beta = 0 (only 1/61 s clips carry the 1 kHz stream); if exposure varies, beta is fitted and applied relative
       to the 1/61 s calibration exposure. Unknown DJI products: beta fitted (snapped to 0.5 when within 0.05).
  IMU sample j        : a + j*dt                                    (UNIFORM grid -- no per-frame sawtooth)
  picture of frame n  : T_n - beta*exposure_n + c_pic               (centre row, mid-exposure)
                        c_pic = 0 (O3, verified by optical flow and Gyroflow render fits); OA4 1 kHz: -0.8 ms at 1/61 s
                        (0005/0006), +0.12 ms at exposures <= 4.6 ms (0012), linear in between (oa4_picture_offset),
                        0 for O4 Pro (verified by vision on 6 windows, together with the held-sample delay above)
  OA4 per-frame cam_quat sample time: T_n + 18 ms.
Camera clock -> video timeline: affine per segment, least squares PTS_n ~ alpha + s*T_n (s ~ sensor_fps/container_fps;
never n/fps). Affine keeps the IMU grid uniform. readout_s = readout_raw * s.

Axes: DJI body is FRD, q_raw maps body->world. Camera (x right, y down, z fwd): cam_x = body_y, cam_y = body_z,
cam_z = body_x, i.e. v_body = M v_cam with M = [[0,0,1],[1,0,0],[0,1,0]] and q_cam = q_raw (x) quat(M).

Reading the metadata samples (`reader=` / env STILLPOINT_TELEMETRY_READER = auto | mp4 | pyav):
  mp4  : sample tables from the moov box (video.mp4_tracks), then an unbuffered pread of exactly the djmd (and needed
         dbgi) bytes, streamed by a background thread so the protobuf decoding overlaps the I/O. One small read per
         frame: ~0.4 ms each on Jimmy's SD card (the card's random-read limit), so an 8.5-min 4K clip (30k frames)
         takes ~12 s there instead of ~2 min (which was the old 1 MiB-buffered seek+read reading the whole file twice),
         and well under a second from an internal SSD.
  pyav : PyAV demux of the djmd/dbgi data streams (fallback for containers the box parser cannot read).
  auto : mp4 when the moov parses, else pyav. Both feed identical bytes to the same decoders (tests/test_telemetry_fast.py).
dbgi (OA4 only; `dbgi=` / env STILLPOINT_TELEMETRY_DBGI = auto | full): ~7.4 KB per frame (3.5-5 Mbit/s), used only
  for the EIS cross-check (`eis_baked`) and diagnostic extras (dbgi_q_eis_cam / dbgi_q_phys_cam; nothing downstream
  reads them). auto: when the clip header says EIS off, only DBGI_SPARSE_N frames at the start, middle and end are
  read (extra['dbgi_frames'] lists them; the dbgi_* extras cover just those frames); if the header says EIS on, or
  any sampled frame reports EIS active, every frame is read, as before. full: always every frame.
  Nothing in the Telemetry fields (or in any non-dbgi extra) depends on this choice.
Quick look (`probe_telemetry`): the same parser over the first QUICK_FRAMES frames plus a sparse whole-clip clock
  check -- what the app's clip list needs (camera, gyro rate, EIS, lens / FOV range) in well under a second even on
  the SD card; never cached. Full telemetry (analysis) always parses every frame.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import warnings

import numpy as np

from .geom import Lens, mat_to_quat, qconj, qfix_sign, qlog, qmul, qnormalize, slerp_series
from .types import Telemetry
from . import video as _video

PARSER_VERSION = 9                    # v9: O4 Pro held (duplicated) 2 kHz samples -> 1 kHz at the measured time
                                      # v8: OA4 exposure-dependent picture offset (short shutters)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

M_CAM2BODY = np.array([[0.0, 0.0, 1.0],   # body x (forward) = cam z
                       [1.0, 0.0, 0.0],   # body y (right)   = cam x
                       [0.0, 1.0, 0.0]])  # body z (down)    = cam y
Q_CAM2BODY = mat_to_quat(M_CAM2BODY)      # q_cam = q_raw (x) Q_CAM2BODY

# Reference values (research/footage_*.md). Used only to (a) sanity-check what the file says and (b) fall back
# when a mode stores no lens (OA4 16:9 / EIS on). Never silently override file values.
KNOWN_LENSES = {
    'FC8383': dict(fx=1405.1293, k=[0.24991769, 0.01360575, -0.06208358, 0.01219307], width=3840, height=2160,
                   readout_ns=9719999),
    'OsmoAction4': dict(fx=1457.0737, k=[0.155131, 0.137141, -0.093861, 0.004170], width=3840, height=2880,
                        readout_ns=11087603),
}
OA4_FRAME_CENTER_OFFSET_S = -0.0008   # picture (centre row, mid-exposure) relative to T_k (0005/0006, 1/61 s)
# 1 kHz clips at SHORT shutters (AUTO 1/1198..1/220 s, DJI_20260927091931_0012, 6 x 12-s windows, exposures 0.8-4.6 ms):
# the picture sits 0.92 ms LATER than the 1/61 s constant above predicts (+0.921 ms, SD 0.010 ms between windows, no
# exposure trend inside 1.0-4.5 ms; DJI's block offsets already carry the -exposure/2 term: beta = 0.500).
# Confirmed independently by row-band flow fits (+0.66..+0.85 ms) and calib.self_calibrate (+0.98 +- 0.12 ms).
# research/notes/oa4_0012_timing_diagnosis.md. The two measurements disagree at 1/61 s, so the picture offset is
# c_pic(e) = OA4_FRAME_CENTER_OFFSET_S + OA4_SHORT_SHUTTER_EXTRA_S * w(e), w = 1 for e <= 4.6 ms, 0 at e >= 1/61 s,
# linear in between (UNVALIDATED between 4.6 and 16.4 ms: flagged in the warnings).
# 1/61 s re-check with the same three-frame-delta fit (timecal.py, 2026-09-29, tracks of the whole 0005 / 0006 clips,
# ISO 12800 indoor handheld): 0006 -0.14 +- 0.06 ms relative to the -0.8 ms constant (i.e. it holds; the short-shutter
# value would be +0.92), 0005 -2.6 +- 0.14 ms (consistent over the clip, but its row-slope jello proxy does not
# improve at that offset: a clip-specific lag, e.g. motion-adaptive temporal NR at ISO 12800). Kept as is; per-clip
# timecal applies a correction when it is confident. Rows at 1/61 s must be exposure-box averaged (plan_build): with
# instantaneous rows the RS correction made the 0005 jello proxy WORSE than no RS correction (skew 0.63 vs 0.58 px),
# with the box it is better (0.56; 0006: 0.63 none -> 0.40 instantaneous -> 0.36 box).
# timecal v2 (box width fitted per clip, 3 windows of each whole clip): 0006 box x1.38, offset -0.19 ms (delta cost
# -0.9 %, all 3 held-out windows better); 0005 cost minimum at box x2.5-3 with offset -3.9..-4.5 ms (jello proxy
# skew/stretch 0.53/0.50 at x1 -> 0.41/0.46; the ORIGINAL 0.56/0.52): the 0005 picture behaves as if averaged over
# ~3 frames and ~4 ms late -- motion-adaptive temporal noise reduction at ISO 12800 on a slow pan is the likely
# cause, i.e. a per-clip property, not a timing constant. Hence per-clip calibration, no constant change here.
OA4_SHORT_SHUTTER_EXTRA_S = 0.00092
OA4_SHORT_SHUTTER_MAX_S = 0.0046      # longest exposure the short-shutter offset was measured at
OA4_LONG_SHUTTER_S = 1.0 / 61.0       # exposure of the 0005/0006 calibration


def oa4_picture_offset(exposure_s) -> np.ndarray:
    """OA4 (1 kHz stream) picture offset c_pic(e) in s, per frame (see OA4_SHORT_SHUTTER_EXTRA_S)."""
    e = np.asarray(exposure_s, np.float64)
    e = np.where(np.isfinite(e) & (e > 0), e, OA4_LONG_SHUTTER_S)
    w = np.clip((OA4_LONG_SHUTTER_S - e) / (OA4_LONG_SHUTTER_S - OA4_SHORT_SHUTTER_MAX_S), 0.0, 1.0)
    return OA4_FRAME_CENTER_OFFSET_S + OA4_SHORT_SHUTTER_EXTRA_S * w
OA4_CAM_QUAT_DELAY_S = 0.018          # per-frame cam_quat sampled at T_k + 18.0..18.25 ms (4:3 mode, vs the 1 kHz stream)
# 16:9 mode (no 1 kHz stream): the per-frame cam_quat lags the picture by only ~9.5 ms (vs 18.8 ms in 4:3).
# Measured on DJI_20260926152149_0002 by KLT/Kabsch rotation vs telemetry lag scans over 7 windows (47-135 deg/s,
# exposures 2-4 ms, no exposure trend): median-residual estimator +9.6 ms, trimmed-RMS estimator +9.0 ms relative to
# the 4:3 model -> use 18.0 - 9.3 = 8.7 ms (+-1 ms). Modelled as a shorter cam_quat delay so frame_t keeps T_k - 0.8 ms.
OA4_CAM_QUAT_DELAY_16X9_S = 0.0087
# DJI O4 Pro: the 2 kHz DeviceAttitude stream is a 1 kHz stream with every sample written twice (samples 2j and 2j+1
# are bit-identical in every O4P clip: 0003 and 0004, 100 % of pairs, no phase slips). Interpolating the held series
# gives a staircase (still for 0.5 ms, then double speed): a 0..0.5 ms sawtooth time error at 1 kHz (0.25 ms mean
# lag) on every row and frame -- jitter and row wobble proportional to the angular rate. The parser keeps one sample
# per pair at the first copy's grid time + O4P_HOLD_SAMPLE_DELAY_S. Measured by the 3-frame-delta track fit on the
# ORIGINAL frames (6 x 12-s windows, 0003 @20/60/112 s + 0004 @45/88/177 s; median exposures 0.14-1.19 ms, 1-143
# deg/s): offset of the de-duplicated series -0.138..-0.180 ms (mean -0.168, SD 0.016) at the metadata readout,
# the same in every exposure tercile (no exposure term beyond the -e/2 already in `offset`: slope 0.00); readout
# 15.31-15.63 ms (metadata 15.38 ms kept); top->bottom (reversed readout: +40-50 % cost); focal unobservable
# (0.90-1.12, kept 1.0). The de-duplicated series fits better than the held one in every window (delta cost -0.1..
# -9 %, the most in fast windows). Scripts: <support dir>/Stillpoint/scratch/o4timing (focus_o4.py).
O4P_HOLD_SAMPLE_DELAY_S = 0.00017
O4P_HOLD_MIN_FRAC = 0.4               # a stream is 'held' when >= this fraction of consecutive samples repeat exactly


def dedup_held_samples(t: np.ndarray, q: np.ndarray, delay_s: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Collapse runs of bit-identical consecutive samples (a lower-rate stream written at a higher rate) to one
    sample each, at the run's first time + delay_s. Returns (t, q, info)."""
    t = np.asarray(t, np.float64)
    q = np.asarray(q)
    if len(q) < 2:
        return t, q, dict(held_frac=0.0, n_in=int(len(q)), n_out=int(len(q)))
    same = np.all(q[1:] == q[:-1], axis=1)
    first = np.r_[True, ~same]
    runs = np.diff(np.r_[np.flatnonzero(first), len(q)])
    info = dict(held_frac=float(same.mean()), n_in=int(len(q)), n_out=int(first.sum()),
                run_lengths={int(u): int(c) for u, c in zip(*np.unique(runs, return_counts=True))}, delay_s=delay_s)
    return t[first] + delay_s, q[first], info


OA4_DBGI_EIS_DELAY_S = 0.0095         # dbgi 2.1.11.7.3 EIS-output attitude at T_k + 9.5 ms
OA4_DBGI_PHYS_DELAY_S = 0.0105        # dbgi 2.1.11.7.4 physical attitude at T_k + 10.5 ms
EIS_STATUS = {0: 'EIS_OFF', 1: 'EIS_ROCK_STEADY', 2: 'EIS_HORIZON_STEADY', 3: 'EIS_HYPER', 4: 'EIS_TRADEOFF',
              5: 'EIS_HORIZON_BALANCING', 6: 'EIS_DEEPSPACE', 7: 'EIS_OFF_WITH_CROP', 8: 'EIS_HORIZON_CORRECTION',
              9: 'EIS_RS_AUTO'}
READER_ENV = 'STILLPOINT_TELEMETRY_READER'    # auto | mp4 | pyav
DBGI_ENV = 'STILLPOINT_TELEMETRY_DBGI'        # auto | full
DBGI_SPARSE_N = 8                             # dbgi frames read at the start, middle and end when EIS is off (auto)
QUICK_FRAMES = 180                            # probe_telemetry: frames parsed from the start of the clip (3 s)
QUICK_CLOCK_CHECKS = 32                       # probe_telemetry: frames spread over the clip for the joined-file check

# ============================================================================ protobuf wire format


def _varint(b, i):
    r = 0
    s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if not c & 0x80:
            return r, i
        s += 7


def _pb(b) -> dict:
    """field -> list of (wire_type, value); value int (wt0) or bytes (wt1/2/5)."""
    d: dict = {}
    i = 0
    n = len(b)
    while i < n:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            L, i = _varint(b, i)
            v = b[i:i + L]
            i += L
        elif wt == 5:
            v = b[i:i + 4]
            i += 4
        elif wt == 1:
            v = b[i:i + 8]
            i += 8
        else:
            raise ValueError(f'unsupported wire type {wt}')
        if i > n:
            raise ValueError('truncated protobuf')
        d.setdefault(fn, []).append((wt, v))
    return d


def _get(d, fn, default=None):
    x = d.get(fn) if d else None
    return x[0][1] if x else default


def _msg(d, fn):
    v = _get(d, fn)
    return _pb(v) if isinstance(v, (bytes, bytearray)) else {}


def _f32(v, default=float('nan')):
    return struct.unpack('<f', v)[0] if isinstance(v, (bytes, bytearray)) and len(v) == 4 else default


def _f32s(v):
    return list(struct.unpack('<%df' % (len(v) // 4), v)) if isinstance(v, (bytes, bytearray)) else []


def _ints(d, fn):
    """repeated int32/int64 (packed or not), two's complement -> signed."""
    out = []
    for wt, v in (d.get(fn, []) if d else []):
        if wt == 2:
            i = 0
            while i < len(v):
                x, i = _varint(v, i)
                out.append(x - (1 << 64) if x >= (1 << 63) else x)
        elif wt == 0:
            out.append(v - (1 << 64) if v >= (1 << 63) else v)
    return out


def _str(v):
    return v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else ''


_Q20 = struct.Struct('<xfxfxfxf')


def _quat(v):
    """dvtm Quaternion {1:w 2:x 3:y 4:z} (f32); proto3-omitted components are 0."""
    if len(v) == 20 and v[0] == 0x0D and v[5] == 0x15 and v[10] == 0x1D and v[15] == 0x25:
        return _Q20.unpack(v)
    d = _pb(v)
    return tuple(_f32(_get(d, k), 0.0) for k in (1, 2, 3, 4))


def _vec234(v):
    d = _pb(v)
    return tuple(_f32(_get(d, k), 0.0) for k in (2, 3, 4))


def _f32field(d, fn):
    """sub-message fn whose field 1 is a float -> float or nan."""
    m = _msg(d, fn)
    return _f32(_get(m, 1)) if m else float('nan')


# ============================================================================ djmd / dbgi decoding


def _parse_clip_meta(b) -> dict:
    d = _pb(b)
    h = _msg(d, 1)
    m = dict(proto_file=_str(_get(h, 1, b'')), lib_version=_str(_get(h, 2, b'')),
             product_proto_version=_str(_get(h, 3, b'')), firmware=_str(_get(h, 6, b'')),
             clip_timestamp_us=_get(h, 9, 0), product_name=_str(_get(h, 10, b'')))
    if 3 in d:
        m['dist_k'] = _f32s(_get(_msg(d, 3), 1, b''))
    if 4 in d:
        m['readout_ns'] = _get(_msg(d, 4), 1, 0)
    if 5 in d:
        m['read_direction'] = _get(_msg(d, 5), 1, 0)
    if 8 in d:
        m['fx'] = _f32(_get(_msg(d, 8), 1))
    m['eis_status'] = _get(_msg(d, 9), 1, 0) if 9 in d else None
    if 10 in d:
        m['imu_rate'] = _get(_msg(d, 10), 1, 0)
    if 11 in d:
        m['sensor_fps'] = _f32(_get(_msg(d, 11), 1))
    m['fields'] = sorted(d.keys())
    return m


def _parse_stream_meta(b) -> dict:
    d = _pb(b)
    m = {}
    v = _msg(d, 3)
    if v:
        m.update(width=_get(v, 1, 0), height=_get(v, 2, 0), meta_fps=_f32(_get(v, 3)), bit_depth=_get(v, 5))
    if 5 in d:
        m['fov_type'] = _get(_msg(d, 5), 1, 0)
    return m


def _parse_djmd(samples: list[bytes]):
    n = len(samples)
    T = np.zeros(n)
    seq = np.zeros(n, np.int64)
    exp = np.full(n, np.nan)
    iso = np.full(n, np.nan)
    zoom = np.full(n, np.nan)
    cct = np.zeros(n)
    cam_q = np.full((n, 4), np.nan)
    acc = np.full((n, 3), np.nan)
    att_off = np.full(n, np.nan)
    att_cnt = np.zeros(n, np.int64)
    att_vsync = np.full(n, -1, np.int64)
    clips = []           # [(frame_index, clip_meta, stream_meta)]
    quats = []
    for k, b in enumerate(samples):
        d = _pb(b)
        if 1 in d:
            cm = _parse_clip_meta(_get(d, 1))
            sm = _parse_stream_meta(_get(d, 2)) if 2 in d else {}
            clips.append((k, cm, sm))
        fm = _msg(d, 3)
        hd = _msg(fm, 1)
        seq[k] = _get(hd, 1, 0)
        T[k] = _get(hd, 2, 0) * 1e-6
        cam = _msg(fm, 2)
        if cam:
            e = _ints(_msg(cam, 4), 1)
            if len(e) >= 2 and e[1]:
                exp[k] = e[0] / e[1]
            iso[k] = _f32field(cam, 3)
            zoom[k] = _f32field(cam, 5)
            cct[k] = _get(_msg(cam, 6), 1, 0) if 6 in cam else 0
            if 9 in cam:
                cam_q[k] = _quat(_get(cam, 9))
            if 10 in cam:
                acc[k] = _vec234(_get(cam, 10))
        imu = _msg(fm, 3)
        if 2 in imu:
            att = _msg(imu, 2)
            qs = [_quat(v) for _wt, v in att.get(3, [])]
            att_off[k] = _f32(_get(att, 4), 0.0)
            att_vsync[k] = _get(att, 2, 0)
            att_cnt[k] = len(qs)
            quats.extend(qs)
    return dict(T=T, seq=seq, exposure=exp, iso=iso, zoom=zoom, cct=cct, cam_q=cam_q, acc=acc, att_off=att_off,
                att_cnt=att_cnt, att_vsync=att_vsync, quats=np.asarray(quats, np.float64).reshape(-1, 4), clips=clips)


def _frame_ts(b) -> float:
    """FrameMetaHeader.frame_timestamp (s) of one djmd sample."""
    return _get(_msg(_msg(_pb(b), 3), 1), 2, 0) * 1e-6


def _parse_dbgi_ac203(samples: list[bytes]):
    """OA4 dbgi: per frame EIS mode, vsync, EIS-output and physical attitudes (raw DJI body->world, w,x,y,z)."""
    n = len(samples)
    mode = np.full(n, -1, np.int64)
    vs = np.full(n, -1, np.int64)
    q_eis = np.full((n, 4), np.nan)
    q_phys = np.full((n, 4), np.nan)
    info = {}
    for k, b in enumerate(samples):
        try:
            d = _pb(b)
            fr = _msg(_msg(d, 2), 1)
            if k == 0:
                try:
                    info['sensor_mode'] = _str(_get(_msg(fr, 2), 2, b''))
                    info['pipeline_topology'] = _str(_get(_msg(fr, 3), 2, b''))
                except Exception:
                    pass
            e = _msg(fr, 11)
            m = _get(e, 6)
            mode[k] = m if m is not None else (0 if 5 in e else -1)
            s7 = _msg(e, 7)
            vs[k] = _get(s7, 1, -1)
            if 3 in s7:
                q_eis[k] = _f32s(_get(s7, 3))[:4]
            if 4 in s7:
                q_phys[k] = _f32s(_get(s7, 4))[:4]
        except Exception:
            continue
    return dict(mode=mode, vsync=vs, q_eis=q_eis, q_phys=q_phys, info=info)


def _product_flags(clip: dict) -> tuple[bool, bool, bool]:
    """(is_o3, is_oa4, is_o4p) from a parsed clip header."""
    proto = clip.get('proto_file', '')
    product = clip.get('product_name', '')
    return ('wm169' in proto or 'FC8383' in product,
            'ac203' in proto or 'OsmoAction4' in product.replace(' ', ''),
            'O4P' in proto or product.strip() == 'DJI O4P')


def _header_eis_baked(clip: dict, fmt_comment: str) -> bool:
    eis_status = clip.get('eis_status')
    baked = bool(eis_status not in (None, 0, 7))
    if eis_status is None and 'EIS:ON' in fmt_comment.upper().replace(' ', ''):
        baked = True
    return baked


def _dbgi_sparse_frames(m: int, n: int = DBGI_SPARSE_N) -> np.ndarray:
    """Frames whose dbgi is read when EIS is off: the first, middle and last n of m."""
    if m <= 3 * n:
        return np.arange(m, dtype=np.int64)
    mid = m // 2 - n // 2
    return np.unique(np.concatenate([np.arange(n), np.arange(mid, mid + n), np.arange(m - n, m)])).astype(np.int64)


# ============================================================================ reading the metadata samples


def _reader_mode(reader: str | None) -> str:
    mode = (reader or os.environ.get(READER_ENV) or 'auto').strip().lower()
    if mode not in ('auto', 'mp4', 'pyav'):
        raise ValueError(f'telemetry reader {mode!r}: expected auto, mp4 or pyav')
    return mode


def _dbgi_mode(dbgi: str | None) -> str:
    mode = (dbgi or os.environ.get(DBGI_ENV) or 'auto').strip().lower()
    if mode not in ('auto', 'full'):
        raise ValueError(f'dbgi mode {mode!r}: expected auto or full')
    return mode


class _MetaSource:
    """The djmd / dbgi samples of one file, from the MP4 sample tables (mode 'mp4') or a PyAV demux ('pyav').
    (Reading on a thread while decoding was tried: GIL hand-offs made it ~10% slower than read-then-decode.)"""

    def __init__(self, path: str, reader: str | None = None):
        self.path = path
        self.requested = _reader_mode(reader)
        self.mode = self.requested
        self.djmd = self.dbgi = None
        self._pyav_cache = None
        tracks = None
        if self.mode in ('auto', 'mp4'):
            try:
                tracks = _video.mp4_tracks(path)
            except Exception as e:        # not an MP4/MOV the box parser understands
                if self.mode == 'mp4':
                    raise
                self.fallback_reason = f'{type(e).__name__}: {e}'
                self.mode = 'pyav'
        if tracks is not None:
            self.djmd = _video.find_track(tracks, fourcc='djmd', handler_name='DJI meta')
            if self.djmd is None:
                self.djmd = _video.find_track(tracks, handler_name='CAM meta')
            if self.djmd is None:
                raise ValueError(f'{path}: no DJI djmd telemetry track')
            self.dbgi = _video.find_track(tracks, fourcc='dbgi', handler_name='DJI dbgi')
            if self.dbgi is not None and not self.dbgi.n_samples:
                self.dbgi = None
            if self.mode == 'auto':
                ok = len(self.djmd.offsets) == self.djmd.n_samples and (
                    self.dbgi is None or len(self.dbgi.offsets) == self.dbgi.n_samples)
                self.mode = 'mp4' if ok else 'pyav'
                if not ok:
                    self.fallback_reason = 'sample table without chunk offsets'
        if self.mode == 'pyav':
            self._pyav_demux()

    # ------------------------------------------------------------------ PyAV
    def _pyav_demux(self):
        """Demux the djmd (+ dbgi) data streams with PyAV (ffmpeg's mov demuxer reads just those samples when the
        other streams are discarded; other containers are read through)."""
        import av
        tid_djmd = self.djmd.track_id if self.djmd is not None else None
        tid_dbgi = self.dbgi.track_id if self.dbgi is not None else None
        with av.open(self.path) as c:
            def hn(s):
                return str(s.metadata.get('handler_name', '')).strip()
            data = [s for s in c.streams if s.type == 'data']
            if tid_djmd is not None:
                sj = next((s for s in data if s.id == tid_djmd), None)
                sb = next((s for s in data if s.id == tid_dbgi), None) if tid_dbgi is not None else None
            else:
                sj = next((s for s in data if hn(s) == 'DJI meta'), None) or \
                     next((s for s in data if hn(s) == 'CAM meta'), None)
                sb = next((s for s in data if hn(s) == 'DJI dbgi'), None)
            if sj is None:
                raise ValueError(f'{self.path}: no DJI djmd telemetry stream')
            sel = [s for s in (sj, sb) if s is not None]
            ij, ib = sj.index, (sb.index if sb is not None else None)
            got = {s.index: [] for s in sel}
            for pk in c.demux(sel):
                if pk.size == 0 and pk.pts is None and pk.dts is None:      # demuxer flush packet
                    continue
                got[pk.stream.index].append(bytes(pk))
        self._pyav_cache = (got[ij], got[ib] if ib is not None else None)

    # ------------------------------------------------------------------ common
    @property
    def n_djmd(self) -> int:
        return self.djmd.n_samples if self.mode == 'mp4' else len(self._pyav_cache[0])

    @property
    def n_dbgi(self) -> int:
        if self.mode == 'mp4':
            return self.dbgi.n_samples if self.dbgi is not None else 0
        return len(self._pyav_cache[1]) if self._pyav_cache[1] is not None else 0

    def djmd_samples(self, idx) -> list[bytes]:
        idx = np.asarray(idx, np.int64).reshape(-1)
        if self.mode == 'mp4':
            return self.djmd.read_samples(self.path, idx)
        return [self._pyav_cache[0][int(i)] for i in idx]

    def read(self, n_djmd: int, dbgi_idx=None) -> tuple[list[bytes], list[bytes]]:
        """(djmd samples 0..n_djmd-1, dbgi samples at dbgi_idx). In 'mp4' mode ONE pass of preads: each frame's djmd
        and dbgi samples are adjacent in DJI files, so a frame's pair merges into a single read."""
        dbgi_idx = np.zeros(0, np.int64) if dbgi_idx is None else np.asarray(dbgi_idx, np.int64).reshape(-1)
        if self.mode == 'pyav':
            dj, db = self._pyav_cache
            return dj[:n_djmd], [db[int(i)] for i in dbgi_idx]
        oj, sj = self.djmd.sample_ranges(np.arange(n_djmd))
        if not len(dbgi_idx):
            return _video.read_ranges(self.path, oj, sj), []
        ob, sb = self.dbgi.sample_ranges(dbgi_idx)
        got = _video.read_ranges(self.path, np.concatenate([oj, ob]), np.concatenate([sj, sb]))
        return got[:n_djmd], got[n_djmd:]

    def read_dbgi(self, idx) -> list[bytes]:
        idx = np.asarray(idx, np.int64).reshape(-1)
        if self.mode == 'mp4':
            return self.dbgi.read_samples(self.path, idx)
        return [self._pyav_cache[1][int(i)] for i in idx]

    def describe(self) -> str:
        reason = getattr(self, 'fallback_reason', None)
        return f'{self.mode} (auto fallback: {reason})' if reason else self.mode


# ============================================================================ timing helpers


def _segments_from_T(T: np.ndarray, clip_starts: list[int]):
    """Split frames into continuous shots. Returns (segments, clip_bounds).

    clip_bounds: every clip boundary (repeated header or clock discontinuity) with a continuity flag;
    segments: runs of frames whose frame clock is continuous (continuous clip joins are merged)."""
    n = len(T)
    dT = np.diff(T)
    med = np.median(dT) if len(dT) else 1 / 60
    disc = set((np.flatnonzero((dT < 0.5 * med) | (dT > 1.5 * med)) + 1).tolist())
    starts = sorted(set([0] + [int(s) for s in clip_starts if 0 < s < n]) | disc)
    clip_bounds = []
    for s in starts[1:]:
        cont = s not in disc
        clip_bounds.append(dict(frame=int(s), continuous=bool(cont), dT_s=float(T[s] - T[s - 1]),
                                header_repeat=bool(s in clip_starts)))
    seg_starts = [0] + sorted(disc)
    segs = [(int(a), int(b) - 1) for a, b in zip(seg_starts, seg_starts[1:] + [n])]
    return segs, clip_bounds


def _affine(x, y):
    A = np.vstack([x - x[0], np.ones_like(x)]).T
    (s, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(s), float(c - s * x[0])   # y = c0 + s*x


def _fit_grid(T, off, cnt, exp, rate, beta):
    """Uniform-grid fit over the blocks of one segment. Returns (a, dt, resid, N_first) with sample j at a + j*dt."""
    Ts = 1.0 / rate
    has = cnt > 0
    N = np.concatenate([[0], np.cumsum(cnt)])[:-1]
    y = T - off * Ts - beta * np.nan_to_num(exp)
    A = np.vstack([N[has].astype(np.float64), np.ones(int(has.sum()))]).T
    w = np.ones(int(has.sum()))
    for _ in range(3):  # mild robustness (IRLS, Huber-ish)
        (dt, a), *_ = np.linalg.lstsq(A * w[:, None], y[has] * w, rcond=None)
        r = y[has] - (a + N[has] * dt)
        sc = 1.4826 * np.median(np.abs(r)) + 1e-9
        w = 1.0 / np.maximum(np.abs(r) / (3 * sc), 1.0)
    res = np.full(len(T), np.nan)
    res[has] = y[has] - (a + N[has] * dt)
    return float(a), float(dt), res, N


def _fit_beta(T, off, cnt, exp, rate, segs):
    """Free fit of the exposure coefficient over the longest segment (None if not identifiable)."""
    a, b = max(segs, key=lambda s: s[1] - s[0])
    sl = slice(a, b + 1)
    c = cnt[sl]
    has = c > 0
    if has.sum() < 20:
        return None
    N = np.concatenate([[0], np.cumsum(c)])[:-1][has].astype(np.float64)
    e = np.nan_to_num(exp[sl])[has]
    if e.std() < 2e-4:
        return None
    y = (T[sl] - off[sl] / rate)[has]
    A = np.vstack([N, np.ones_like(N), e]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][2])


def _body_rate_deg(q, t):
    dq = qmul(qconj(q[:-1]), q[1:])
    return np.rad2deg(np.linalg.norm(qlog(dq), axis=1) / np.maximum(np.diff(t), 1e-9))


# ============================================================================ main entry


def _cache_path(path: str, cache_dir: str, dbgi: str | None = None) -> str:
    """Cache file for a full parse. The reader (mp4 / pyav) does not enter the key: both give identical results.
    dbgi='full' gets its own entry; the default (auto) key is unchanged from parser v7, whose entries -- written
    with every dbgi frame -- are a superset of what auto produces and stay valid."""
    st = os.stat(path)
    key = f'{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|v{PARSER_VERSION}'
    if _dbgi_mode(dbgi) == 'full':
        key += '|dbgi=full'
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(cache_dir, f'telemetry_{stem}_{h}.npz')


_SCALARS = ('source', 'camera', 'width', 'height', 'fps', 'readout_s', 'imu_rate', 'has_highrate', 'eis_baked')
_ARRAYS = ('frame_pts', 'frame_t', 'exposure_s', 'imu_t', 'imu_q')


def _save(tel: Telemetry, p: str):
    arrs = {k: getattr(tel, k) for k in _ARRAYS}
    meta = {k: getattr(tel, k) for k in _SCALARS}
    L = tel.lens
    meta['lens'] = dict(model=L.model, fx=L.fx, fy=L.fy, cx=L.cx, cy=L.cy, k=[float(c) for c in L.k],
                        width=L.width, height=L.height)
    meta['segments'] = [list(map(int, s)) for s in tel.segments]
    grav_same = tel.gravity_q is not None and tel.gravity_q is tel.imu_q
    meta['gravity'] = 'same' if grav_same else ('none' if tel.gravity_q is None else 'array')
    if meta['gravity'] == 'array':
        arrs['gravity_q'] = tel.gravity_q
    ex = {}
    for k, v in tel.extra.items():
        if isinstance(v, np.ndarray):
            arrs['x__' + k] = v
        else:
            ex[k] = v
    meta['extra'] = ex
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f'.tmp{os.getpid()}.npz'
    np.savez(tmp, meta_json=np.array(json.dumps(meta, default=_jsonable)), **arrs)
    os.replace(tmp, p)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _load(p: str) -> Telemetry:
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z['meta_json']))
    L = meta['lens']
    lens = Lens(L['model'], L['fx'], L['fy'], L['cx'], L['cy'], np.array(L['k'], np.float64), L['width'], L['height'])
    extra = dict(meta['extra'])
    for k in z.files:
        if k.startswith('x__'):
            extra[k[3:]] = z[k]
    imu_q = z['imu_q']
    grav = imu_q if meta['gravity'] == 'same' else (z['gravity_q'] if meta['gravity'] == 'array' else None)
    return Telemetry(source=meta['source'], camera=meta['camera'], width=int(meta['width']),
                     height=int(meta['height']), fps=float(meta['fps']), frame_pts=z['frame_pts'],
                     frame_t=z['frame_t'], exposure_s=z['exposure_s'], readout_s=float(meta['readout_s']),
                     lens=lens, imu_t=z['imu_t'], imu_q=imu_q, imu_rate=float(meta['imu_rate']),
                     has_highrate=bool(meta['has_highrate']), eis_baked=bool(meta['eis_baked']), gravity_q=grav,
                     segments=[tuple(s) for s in meta['segments']], extra=extra)


def cache_file(path: str, cache_dir: str | None = 'work/cache', dbgi: str | None = None) -> str | None:
    """Where load_telemetry(path, cache_dir, dbgi=dbgi) keeps its cache (None when caching is off)."""
    if cache_dir is None:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    cdir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_ROOT, cache_dir)
    return _cache_path(path, cdir, dbgi)


def load_telemetry(path: str, cache_dir: str | None = 'work/cache', reader: str | None = None,
                   dbgi: str | None = None) -> Telemetry:
    """Parse a DJI O3 / Osmo Action 4 clip into a Telemetry (see module docstring for the timing model).

    cache_dir: relative paths are resolved against the stillpoint project root; None disables caching.
    reader: 'auto' | 'mp4' | 'pyav' (default: env STILLPOINT_TELEMETRY_READER, else auto) -- how the djmd/dbgi
            samples are read; results are identical.
    dbgi:   'auto' | 'full' (default: env STILLPOINT_TELEMETRY_DBGI, else auto) -- see the module docstring.
    """
    path = os.path.abspath(os.path.expanduser(path))
    cp = cache_file(path, cache_dir, dbgi)
    if cp is not None and os.path.exists(cp):
        try:
            return _load(cp)
        except Exception as e:  # corrupt / old cache -> re-parse
            warnings.warn(f'telemetry cache unreadable ({e}); re-parsing')
    tel = _parse(path, reader=reader, dbgi=dbgi)
    if cp is not None:
        _save(tel, cp)
    return tel


def probe_telemetry(path: str, n_frames: int = QUICK_FRAMES, reader: str | None = None) -> Telemetry:
    """Quick look for the app's clip list: the full parser run on the first n_frames frames only (reads ~n_frames
    small samples + QUICK_CLOCK_CHECKS spread over the clip), never cached. Telemetry fields describe that prefix;
    camera / lens / readout / imu_rate / has_highrate / eis_baked / gravity availability are what the full parse
    reports (to ~1e-4 for imu_rate). extra['quick'] = dict(frames, n_frames, clock_checks, discontinuities):
    discontinuities counts sampled frames whose frame clock does not continue the prefix's (a joined file; the
    full parse splits it into segments)."""
    path = os.path.abspath(os.path.expanduser(path))
    return _parse(path, reader=reader, dbgi='auto', max_frames=max(int(n_frames), 2))


def _parse(path: str, reader: str | None = None, dbgi: str | None = None, max_frames: int | None = None) -> Telemetry:
    info = _video.probe(path)
    src = _MetaSource(path, reader)
    dbgi_mode = _dbgi_mode(dbgi)
    F_video = int(info['n_frames'])
    n_read = src.n_djmd if max_frames is None else min(src.n_djmd, int(max_frames))
    # peek at the clip header (djmd sample 0) to decide whether dbgi is needed and how much of it
    m_dbgi = min(src.n_dbgi, n_read, F_video)
    dbgi_idx = None
    if m_dbgi:
        hdr = {}
        try:
            d0 = _pb(src.djmd_samples([0])[0])
            if 1 in d0:
                hdr = _parse_clip_meta(_get(d0, 1))
        except Exception:
            pass
        fmt_comment0 = str(info.get('format_tags', {}).get('comment', ''))
        if not hdr:
            dbgi_idx = _dbgi_sparse_frames(m_dbgi)            # unknown product yet: decide after the parse
        elif _product_flags(hdr)[1]:
            full = dbgi_mode == 'full' or _header_eis_baked(hdr, fmt_comment0)
            dbgi_idx = np.arange(m_dbgi, dtype=np.int64) if full else _dbgi_sparse_frames(m_dbgi)
    djmd_raw, dbgi_raw = src.read(n_read, dbgi_idx)
    D = _parse_djmd(djmd_raw)
    del djmd_raw
    dbgi_read = (dbgi_idx, dbgi_raw) if dbgi_idx is not None else None
    if not D['clips']:
        raise ValueError(f'{path}: djmd has no clip header')
    k0, clip, smeta = D['clips'][0]
    warn = []
    F = F_video
    nd = len(D['T'])
    frame_pts = info['frame_pts']
    quick = None
    if max_frames is not None:
        quick = dict(frames=int(min(nd, F)), n_frames=F, djmd_samples=int(src.n_djmd))
        if src.n_djmd != F:
            warn.append(f'djmd samples {src.n_djmd} != video frames {F}; paired by index, truncated to min')
        F = min(F, nd)
        frame_pts = frame_pts[:F]
    if nd != F:
        warn.append(f'djmd samples {nd} != video frames {F}; paired by index, truncated to min')
        m = min(nd, F)
        for key in ('T', 'seq', 'exposure', 'iso', 'zoom', 'cct', 'cam_q', 'acc', 'att_off', 'att_cnt', 'att_vsync'):
            D[key] = D[key][:m]
        keep = int(D['att_cnt'].sum())
        D['quats'] = D['quats'][:keep]
        frame_pts = frame_pts[:m]
        F = m
    proto = clip.get('proto_file', '')
    product = clip.get('product_name', '')
    is_o3, is_oa4, is_o4p = _product_flags(clip)
    if not (is_o3 or is_oa4 or is_o4p):
        warn.append(f'unknown DJI product {product!r} ({proto}); generic DJI timing model (fitted beta, c_pic = 0)')
    camera = (f'DJI O3 ({product.replace("DJI ", "")})' if is_o3 else 'DJI Osmo Action 4' if is_oa4 else
              'DJI O4 Pro' if is_o4p else product)
    W, H = info['width'], info['height']
    for i, cm, _sm in D['clips'][1:]:
        for key in ('fx', 'dist_k', 'readout_ns', 'imu_rate', 'eis_status'):
            if cm.get(key) != clip.get(key):
                warn.append(f'clip header at frame {i} differs in {key}: {cm.get(key)} vs {clip.get(key)}')

    T = D['T']
    exp = D['exposure']
    segs, clip_bounds = _segments_from_T(T, [c[0] for c in D['clips']])

    # ---- per-segment affine camera clock -> video timeline
    seg_map = []
    for a, b in segs:
        if b - a >= 1:
            s, c0 = _affine(T[a:b + 1], frame_pts[a:b + 1])
        else:
            s, c0 = 1.0, float(frame_pts[a] - T[a])
        r = frame_pts[a:b + 1] - (c0 + s * T[a:b + 1])
        seg_map.append(dict(frames=[a, b], scale=s, offset=c0, pts_resid_rms_s=float(r.std()),
                            pts_resid_max_s=float(np.abs(r).max())))
    s_main = seg_map[int(np.argmax([m['frames'][1] - m['frames'][0] for m in seg_map]))]['scale']

    def to_video(t_cam, seg_i):
        m = seg_map[seg_i]
        return m['offset'] + m['scale'] * np.asarray(t_cam, np.float64)

    if quick is not None:
        # joined-file check: does the frame clock of frames spread over the rest of the clip continue the prefix's?
        n_all = min(src.n_djmd, F_video)
        chk = np.unique(np.linspace(F, n_all - 1, QUICK_CLOCK_CHECKS).round().astype(np.int64)) if n_all > F \
            else np.zeros(0, np.int64)
        disc = 0
        if len(chk):
            Tc = np.array([_frame_ts(b) for b in src.djmd_samples(chk)])
            dev = np.abs(to_video(Tc, len(segs) - 1) - info['frame_pts'][chk])
            disc = int(np.sum(dev > 0.5 / float(info['fps'])))
            if disc:
                warn.append(f'quick look: frame clock jumps after the first {F} frames ({disc} of {len(chk)} '
                            'sampled frames); a joined file? the full parse splits it into segments')
        quick.update(clock_checks=int(len(chk)), discontinuities=disc)

    # ---- lens / readout
    ref = KNOWN_LENSES.get('FC8383' if is_o3 else ('OsmoAction4' if is_oa4 else ''), None)
    lens_source = 'file'
    if clip.get('fx') and clip.get('dist_k') and len(clip['dist_k']) == 4:
        fx = float(clip['fx'])
        kk = np.array(clip['dist_k'], np.float64)
        if ref is not None and (W, H) == (ref['width'], ref['height']):
            if abs(fx / ref['fx'] - 1) > 0.01 or np.abs(kk - np.array(ref['k'])).max() > 0.01:
                warn.append(f'lens differs from the known {product} calibration: fx {fx} k {kk.tolist()}')
    elif ref is not None:
        fx = ref['fx']
        kk = np.array(ref['k'], np.float64)
        if W == ref['width'] and H != ref['height']:
            lens_source = (f'fallback: {product} {ref["width"]}x{ref["height"]} lens centre-cropped to {W}x{H} '
                           '(no lens metadata in this mode; unverified)')
        else:
            lens_source = f'fallback: {product} {ref["width"]}x{ref["height"]} lens scaled (unverified)'
            fx *= W / ref['width']
        warn.append(lens_source)
    else:
        raise ValueError(f'{path}: no lens metadata and no reference lens for {product!r}')
    lens = Lens('kb4', fx, fx, (W - 1) / 2.0, (H - 1) / 2.0, kk, W, H)
    readout_ns = clip.get('readout_ns')
    readout_source = 'file'
    if not readout_ns:
        if ref is not None:
            readout_ns = ref['readout_ns'] * (H / ref['height'] if W == ref['width'] else 1.0)
            readout_source = 'fallback: reference readout scaled by row count (unverified)'
            warn.append(readout_source)
        else:
            readout_ns = 0
    readout_s = float(readout_ns) * 1e-9 * s_main

    # ---- EIS
    eis_status = clip.get('eis_status')
    fmt_comment = str(info.get('format_tags', {}).get('comment', ''))
    eis_baked = _header_eis_baked(clip, fmt_comment)
    extra = dict(product=product, proto_file=proto, firmware=clip.get('firmware'), lens_source=lens_source,
                 readout_source=readout_source, readout_raw_s=float(readout_ns) * 1e-9,
                 read_direction=clip.get('read_direction'), eis_status=eis_status,
                 eis_status_name=EIS_STATUS.get(eis_status, str(eis_status)), format_comment=fmt_comment,
                 sensor_fps_meta=clip.get('sensor_fps'), imu_rate_nominal=clip.get('imu_rate'),
                 fov_type=smeta.get('fov_type'), clip_bounds=clip_bounds, segment_time_maps=seg_map,
                 warnings=warn, parser_version=PARSER_VERSION, meta_reader=src.describe())
    if quick is not None:
        extra['quick'] = quick
    extra['iso'] = D['iso']
    extra['frame_ts_cam'] = T
    extra['frame_seq'] = D['seq']
    extra['wb_cct'] = D['cct']

    # ---- camera-frame per-frame attitude (OA4 3.2.9, gravity aligned) and accelerometer (3.2.10)
    cq = D['cam_q']
    cq_n = np.linalg.norm(cq, axis=1) if len(cq) else np.zeros(0)
    have_cq = bool(len(cq) and np.all(np.isfinite(cq_n) & (np.abs(cq_n - 1) < 0.05)))   # O4P writes all-zero quats
    if len(cq) and not have_cq and np.isfinite(cq).any():
        warn.append('per-frame camera attitude present but invalid (zero/non-unit); ignored')
    if np.isfinite(D['acc']).any():
        extra['accel_cam_g'] = D['acc'] @ M_CAM2BODY    # rows: v_cam = M^T v_body

    # ---- high-rate attitude
    cnt = D['att_cnt']
    rate_nom = float(clip.get('imu_rate') or 0)
    have_hr = D['quats'].shape[0] > 0 and rate_nom > 0
    beta, exp_ref = (0.5 if (is_o3 or is_o4p) else 0.0), 0.0
    c_pic = OA4_FRAME_CENTER_OFFSET_S if is_oa4 else 0.0
    beta_fit = None
    if have_hr and np.nanstd(exp) > 2e-4:
        beta_fit = _fit_beta(T, D['att_off'], cnt, exp, rate_nom, segs)
        if not (is_o3 or is_o4p) and beta_fit is not None:
            beta = 0.5 if abs(beta_fit - 0.5) < 0.05 else float(np.clip(beta_fit, 0.0, 1.0))
            if is_oa4:
                exp_ref = 1.0 / 61.0     # OA4 picture offset was measured at 1/61 s
                warn.append(f'OA4 clip with varying exposure: fitted beta={beta_fit:.3f}; picture offset from the '
                            'exposure model (short shutters: 0012 calibration, 1/61 s: 0005/0006)')
        elif beta_fit is not None and abs(beta_fit - beta) > 0.05:
            warn.append(f'fitted exposure coefficient {beta_fit:.3f} differs from the model {beta}')
    if is_oa4 and have_hr:
        # exposure-dependent picture offset (short shutters measured on 0012; see oa4_picture_offset). Relative to
        # the uniform IMU grid the picture sits at block anchor + offset_n/rate + c_pic whatever beta is, so the same
        # c_pic(e) applies to constant- and varying-exposure clips.
        c_pic = oa4_picture_offset(exp)
        e_ok = np.asarray(exp, np.float64)
        e_ok = e_ok[np.isfinite(e_ok) & (e_ok > 0)]
        if len(e_ok):
            unval = float(np.mean((e_ok > OA4_SHORT_SHUTTER_MAX_S + 2e-4) & (e_ok < OA4_LONG_SHUTTER_S - 5e-4)))
            if unval > 0.05:
                warn.append(f'OA4: {unval * 100:.0f}% of frames have exposures between 4.6 ms and 1/61 s, where the '
                            'picture timing is interpolated (unvalidated)')
    extra['timing'] = dict(beta_exposure=beta, beta_fitted=beta_fit, exposure_ref_s=exp_ref,
                           picture_offset_s=(float(np.median(c_pic)) if np.ndim(c_pic) else c_pic),
                           picture_offset_model=('oa4_exposure_v8' if (is_oa4 and have_hr) else 'constant'))
    if is_o4p:
        extra['timing']['o4p_hold_sample_delay_s'] = O4P_HOLD_SAMPLE_DELAY_S
        extra['timing']['o4p_note'] = ('vision-verified on 6 windows of 0003/0004: picture = T - e/2, readout '
                                       'top->bottom = metadata, 1 kHz held attitude de-duplicated')
    pic_cam = T - beta * (np.nan_to_num(exp) - exp_ref) + c_pic
    frame_t = np.empty(F)
    for si, (a, b) in enumerate(segs):
        frame_t[a:b + 1] = to_video(pic_cam[a:b + 1], si)

    gravity_q = None
    if have_hr:
        q_raw_all = D['quats']
        blk_frame = np.repeat(np.arange(F), cnt[:F])
        imu_t_parts, imu_q_parts, grid_diag = [], [], []
        cam_t_parts = []
        for si, (a, b) in enumerate(segs):
            sl_blocks = slice(a, b + 1)
            c = cnt[sl_blocks]
            if c.sum() == 0:
                continue
            ga, gdt, res, N = _fit_grid(T[sl_blocks], D['att_off'][sl_blocks], c,
                                        np.nan_to_num(exp[sl_blocks]) - exp_ref, rate_nom, beta)
            sel = (blk_frame >= a) & (blk_frame <= b)
            q_seg = q_raw_all[sel]
            j = np.arange(len(q_seg))
            t_cam = ga + j * gdt
            # drop a truncated final block (seen at the end of recordings: 11 of 16 samples, 0.8 deg glitch)
            med_c = np.median(c[c > 0])
            keep = np.ones(len(q_seg), bool)
            if c[-1] > 0 and c[-1] < 0.8 * med_c:
                keep[-int(c[-1]):] = False
                grid_diag.append(dict(segment=si, dropped_truncated_last_block=int(c[-1])))
            nrm = np.linalg.norm(q_seg, axis=1)
            keep &= np.isfinite(nrm) & (np.abs(nrm - 1) < 0.05)
            q_seg, t_cam = q_seg[keep], t_cam[keep]
            # held samples (O4 Pro: each 1 kHz sample written twice) -> one sample per run at its measured time
            held = float(np.mean(np.all(q_seg[1:] == q_seg[:-1], axis=1))) if len(q_seg) > 1 else 0.0
            hold_info = None
            if held >= O4P_HOLD_MIN_FRAC:
                if is_o4p:
                    delay = O4P_HOLD_SAMPLE_DELAY_S
                else:
                    delay = 0.5 * gdt * (1.0 / max(1.0 - held, 1e-3) - 1.0)      # run midpoint (unmeasured)
                    warn.append(f'segment {si}: {held * 100:.0f}% of attitude samples repeat (held stream); '
                                f'de-duplicated at the run midpoint (+{delay * 1e3:.2f} ms, unverified)')
                t_cam, q_seg, hold_info = dedup_held_samples(t_cam, q_seg, delay)
            grid_diag.append(dict(segment=si, a_s=ga, dt_s=gdt, resid_rms_s=float(np.nanstd(res)),
                                  resid_max_s=float(np.nanmax(np.abs(res))), samples=int(len(q_seg)),
                                  held_frac=held, hold_dedup=hold_info))
            if np.nanmax(np.abs(res)) > 0.45 / rate_nom:
                warn.append(f'segment {si}: IMU blocks do not tile a uniform grid (max resid '
                            f'{np.nanmax(np.abs(res)) * 1e6:.0f} us) -- samples may be missing')
            imu_t_parts.append(to_video(t_cam, si))
            cam_t_parts.append(t_cam)
            imu_q_parts.append(q_seg / np.linalg.norm(q_seg, axis=1, keepdims=True))
        # trim overlaps at internal segment boundaries (two clocks spliced): cut at the frame_t midpoint
        for i in range(len(imu_t_parts)):
            lo, hi = -np.inf, np.inf
            a, b = segs[i]
            if a > 0:
                lo = 0.5 * (frame_t[a - 1] + frame_t[a])
            if b < F - 1:
                hi = 0.5 * (frame_t[b] + frame_t[b + 1])
            m = (imu_t_parts[i] > lo) & (imu_t_parts[i] <= hi)
            imu_t_parts[i], imu_q_parts[i], cam_t_parts[i] = imu_t_parts[i][m], imu_q_parts[i][m], cam_t_parts[i][m]
        imu_t = np.concatenate(imu_t_parts)
        q_raw = qfix_sign(np.concatenate(imu_q_parts))
        imu_q = qfix_sign(qnormalize(qmul(q_raw, Q_CAM2BODY)))
        extra['imu_grid'] = grid_diag
        imu_rate = float(1.0 / np.median(np.diff(imu_t)))
        has_highrate = imu_rate >= 500
        if is_o3 or (is_o4p and not have_cq):
            gravity_q = imu_q      # fused attitude, world NED (gravity-referenced tilt) [research: strongly supported]
            extra['gravity_note'] = ('fused attitude world frame is NED (inferred for the O3; assumed for the O4 Pro);'
                                     ' gravity_q is imu_q')
        elif have_cq:
            # OA4: 1 kHz world is NOT gravity aligned; per-frame cam_quat is. W = cq (x) conj(q_raw(t_cq)).
            t_cq = to_video(T + OA4_CAM_QUAT_DELAY_S, 0) if len(segs) == 1 else None
            if t_cq is not None:
                ok = (t_cq >= imu_t[0]) & (t_cq <= imu_t[-1])
                qi = slerp_series(imu_t, q_raw, t_cq[ok])
                cqn = qnormalize(cq[ok])
                Wk = qmul(cqn, qconj(qi))
                Wk = np.where(Wk[:, :1] < 0, -Wk, Wk)
                Wm = qnormalize(np.median(Wk, axis=0))
                for _ in range(2):
                    ang = np.linalg.norm(qlog(qmul(qconj(Wm), Wk)), axis=1)
                    inl = ang < max(3 * np.median(ang), np.deg2rad(0.5))
                    Mq = Wk[inl].T @ Wk[inl]
                    Wm = np.linalg.eigh(Mq)[1][:, -1]
                    Wm = Wm if Wm[0] >= 0 else -Wm
                ang = np.rad2deg(np.linalg.norm(qlog(qmul(qconj(Wm), Wk)), axis=1))
                gravity_q = qfix_sign(qmul(Wm[None, :], imu_q))
                extra['gravity_W'] = Wm
                extra['gravity_W_deg'] = float(np.rad2deg(np.linalg.norm(qlog(Wm))))
                extra['gravity_W_spread_deg'] = dict(mean=float(ang.mean()), p95=float(np.percentile(ang, 95)),
                                                     max=float(ang.max()))
    else:
        if not have_cq:
            raise ValueError(f'{path}: no high-rate attitude and no per-frame camera attitude')
        # per-frame gravity-aligned attitude (OA4 16:9 / EIS-on), sampled after T_k (mode-dependent delay)
        wide = abs(W / H - 16 / 9) < 0.02
        cq_delay = OA4_CAM_QUAT_DELAY_16X9_S if wide else OA4_CAM_QUAT_DELAY_S
        imu_t = np.empty(F)
        for si, (a, b) in enumerate(segs):
            imu_t[a:b + 1] = to_video(T[a:b + 1] + cq_delay, si)
        extra['timing']['cam_quat_delay_s'] = cq_delay
        extra['timing']['cam_quat_delay_source'] = (
            'measured vs video on OA4 16:9 clip 0002 (+-0.7 ms); refine by vision' if wide else
            'measured vs the 1 kHz stream on OA4 4:3 clips 0005/0006')
        imu_q = qfix_sign(qnormalize(qmul(qnormalize(cq), Q_CAM2BODY)))
        good = np.concatenate([[True], np.diff(imu_t) > 0])
        imu_t, imu_q = imu_t[good], imu_q[good]
        imu_rate = float(1.0 / np.median(np.diff(imu_t)))
        has_highrate = False
        gravity_q = imu_q
        extra['gravity_note'] = 'per-frame cam_quat is gravity aligned (world z down)'
        extra['imu_source'] = f'per-frame camera_attitude (djmd 3.2.9) at T_k + {cq_delay * 1e3:.1f} ms'
    if have_cq and have_hr:
        extra['cam_quat_cam'] = qfix_sign(qnormalize(qmul(qnormalize(cq), Q_CAM2BODY)))
        extra['cam_quat_t'] = np.concatenate([to_video(T[a:b + 1] + OA4_CAM_QUAT_DELAY_S, si)
                                              for si, (a, b) in enumerate(segs)])

    # ---- OA4 dbgi (EIS attitudes). Every frame when EIS is on (or dbgi_mode='full'); otherwise the sparse frames
    # read together with djmd -- and every frame after all if any of those reports EIS active (module docstring).
    m = min(src.n_dbgi, F)
    if is_oa4 and m:
        idx, raw = dbgi_read if dbgi_read is not None else (None, None)
        if idx is not None and np.any(idx >= m):
            keep = idx < m
            idx, raw = idx[keep], [r for r, k_ in zip(raw, keep) if k_]
        if idx is None or len(idx) == 0 or ((dbgi_mode == 'full' or eis_baked) and len(idx) < m):
            idx = np.arange(m, dtype=np.int64)
            raw = src.read_dbgi(idx)
        G = _parse_dbgi_ac203(raw)
        if len(idx) < m and np.any(G['mode'] > 0):
            idx = np.arange(m, dtype=np.int64)
            G = _parse_dbgi_ac203(src.read_dbgi(idx))
        extra['dbgi_eis_mode'] = G['mode']
        if len(idx) < m:
            extra['dbgi_frames'] = idx                 # the dbgi_* arrays cover only these frames
        extra['dbgi_info'] = G['info']
        for key, delay in (('q_eis', OA4_DBGI_EIS_DELAY_S), ('q_phys', OA4_DBGI_PHYS_DELAY_S)):
            q = G[key]
            nrm = np.linalg.norm(q, axis=1)
            ok = np.isfinite(nrm) & (np.abs(nrm - 1) < 0.05)
            if ok.mean() > 0.9:
                qc = np.full_like(q, np.nan)
                qc[ok] = qfix_sign(qnormalize(qmul(q[ok] / nrm[ok, None], Q_CAM2BODY)))
                extra[f'dbgi_{key}_cam'] = qc          # NaN rows = missing/invalid in the file
                extra[f'dbgi_{key}_t'] = to_video(T[idx] + delay, 0)
        modes = G['mode']
        if np.any(modes > 0) and not eis_baked:
            warn.append('dbgi reports EIS active although clip header says EIS off; marking eis_baked')
            eis_baked = True
        extra['eis_note'] = ('dbgi_q_eis_cam = EIS output (virtual camera) attitude the picture follows; '
                             'imu_q is the PHYSICAL camera' if eis_baked else 'EIS off')

    # frames whose every row time (frame_t +- readout/2) lies inside the IMU span (orientation_at clamps outside)
    ok = (frame_t - readout_s / 2 >= imu_t[0]) & (frame_t + readout_s / 2 <= imu_t[-1])
    extra['imu_full_coverage_frames'] = [int(np.argmax(ok)), int(len(ok) - 1 - np.argmax(ok[::-1]))] if ok.any() \
        else [0, -1]
    tel = Telemetry(source=path, camera=camera, width=W, height=H, fps=float(info['fps']),
                    frame_pts=np.array(frame_pts, np.float64, copy=True), frame_t=frame_t,
                    exposure_s=np.asarray(exp, np.float64), readout_s=readout_s, lens=lens,
                    imu_t=np.asarray(imu_t, np.float64), imu_q=imu_q, imu_rate=imu_rate,
                    has_highrate=bool(has_highrate), eis_baked=bool(eis_baked), gravity_q=gravity_q,
                    segments=[(int(a), int(b)) for a, b in segs], extra=extra)
    _check(tel)
    return tel


def _check(tel: Telemetry):
    assert len(tel.frame_pts) == len(tel.frame_t) == len(tel.exposure_s)
    assert np.all(np.diff(tel.imu_t) > 0), 'imu_t must be strictly increasing'
    assert np.all(np.diff(tel.frame_pts) > 0), 'frame_pts must be strictly increasing'
    assert tel.imu_q.shape == (len(tel.imu_t), 4)
    assert np.allclose(np.linalg.norm(tel.imu_q, axis=1), 1.0, atol=1e-9)
    assert np.all(np.einsum('ij,ij->i', tel.imu_q[1:], tel.imu_q[:-1]) >= 0), 'imu_q must be sign-continuous'


def angular_rate(tel: Telemetry) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame angular velocity [rad/s] at sample midpoints: (t_mid, w_cam (N-1,3))."""
    dq = qmul(qconj(tel.imu_q[:-1]), tel.imu_q[1:])
    dt = np.diff(tel.imu_t)
    return 0.5 * (tel.imu_t[1:] + tel.imu_t[:-1]), qlog(dq) / dt[:, None]
