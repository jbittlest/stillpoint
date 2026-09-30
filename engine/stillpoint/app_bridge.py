"""Stillpoint app bridge: the JSON-lines front end the Mac app (app/Stillpoint) drives.  Owner: APP package.

    python -m stillpoint.app_bridge probe   CLIP [--telemetry auto|quick|full]
    python -m stillpoint.app_bridge analyze CLIP --out DIR [--smoothness S] [--fov DEG | --crop-area A]
                                            [--horizon-lock [S]] [--roll-limit DEG] [--loop-iters N]
                                            [--fill | --no-fill] [--fill-overscan F] [--mesh | --no-mesh]
                                            [--synth-blur off|auto|angle] [--blur-smooth W] [--timecal | --no-timecal]
                                            [--dry-run] [--start-frame N --max-frames M]
    python -m stillpoint.app_bridge render  CLIP --plan PLAN --out OUT.mov|mp4 [--codec hevc10|hevc10-speed|prores]
                                            [--bitrate-mbps 180] [--start-frame N] [--frames M] [--kernel lanczos3]
                                            [--blur SIDECAR | --no-blur]
Engine v5 analysis options (AnalyzeParams). An option that is not given takes its DEFAULT: option_defaults(camera),
i.e. AnalyzeParams' own default, overridden per camera by CAMERA_OPTION_DEFAULTS below -- the one place to change what
a clip starts with (probe reports the resolved defaults per camera in result['options'], and the app starts every clip
from them). Today: timing self-calibration on for every camera; full-frame fill on for the DJI O3 (gate v6).
    --horizon-lock [S]  horizon lock v2, strength S 0..1 (bare flag = 1.0, 0 = off); --roll-limit DEG keeps bank up
                        to DEG (0 = fully level)
    --fill / --no-fill  full-frame border fill (neighbouring frames; overscan --fill-overscan, default 0.06)
    --mesh / --no-mesh  parallax mesh residual ("Max quality": two extra tracking passes; ~+40 % analysis time on
                        the O3, ~5-7x on OA4 / O4 Pro)
    --synth-blur MODE   synthetic shutter sidecar plan.spblur ('auto' | 'angle'); render uses it automatically
    --blur-smooth W     blur-aware smoothing weight
    --timecal / --no-timecal  per-clip timing self-calibration (--no-timecal keeps the metadata timing)
    --dry-run           resolve + check the options and return them (result['params'] = the manifest params,
                        result['analyze_params'] = the AnalyzeParams fields) without analysing or writing anything
    --start-frame N --max-frames M   (diagnostics / self-tests) analyse only that window of the clip; the plan's
                        records carry the source PTS, so it renders and previews like a whole-clip plan
  fill, mesh and synth-blur cannot be combined yet (bad_args; probe's options.exclusive lists the pairs the engine's
  pipeline.check_params refuses).
  Env STILLPOINT_ANALYSIS_WORKERS=N caps the measurement worker pool (AnalyzeParams.processes / max_workers_measure).
    python -m stillpoint.app_bridge summary DIR [--recompute]
    python -m stillpoint.app_bridge adopt   CLI_ANALYSIS_DIR --clip CLIP --out DIR

stdout carries ONLY JSON objects, one per line:
    {"type":"start","cmd":..,"pid":..,"protocol":2}
    {"type":"progress","stage":"measure","label":"Measuring jitter","fraction":0.42,"stage_fraction":0.6,
     "message":"pass 1 of 3 · 812/2964 frames","elapsed_s":31.2,"eta_s":44.0}
    {"type":"log","message":"..."}                     engine log lines (also echoed to stderr)
    {"type":"notice","code":"...","message":"..."}     something the user should know (e.g. no_frame_cache)
    {"type":"result", ...}                              exactly once on success (exit 0)
    {"type":"error","code":"...","message":"..."}       exit 1
    {"type":"cancelled"}                                exit 130
Everything else (engine prints, ffmpeg, sprender chatter) goes to stderr.

Cancel: SIGTERM / SIGINT, or closing stdin when stdin is a pipe (the app holds it open for the job's lifetime).
The engine is asked to stop cooperatively (pipeline.analyze(cancel=...) when it has that parameter, otherwise at
the next progress checkpoint); if it has not stopped after --grace seconds the bridge kills its process group.
The app starts the bridge as the leader of its own process group and SIGKILLs the whole group 5 s after a cancel.

Scratch space: with env STILLPOINT_WORK_DIR set (the app sets ~/Library/Application Support/Stillpoint/work, which
is not synced to iCloud), every temporary file lives in $STILLPOINT_WORK_DIR/tmp/analyze-<pid> (the preview-size
frame cache is ~n_frames * 0.7 MB for 4K 4:3: 16.7 GB for a 6.7-min clip) and new telemetry caches go to
$STILLPOINT_WORK_DIR/cache. Without it: --out/.work-<pid> and <root>/work/cache (the CLI/test behaviour).
analyze refuses to start with < 3 GB free there, and skips the frame cache (decoding each pass instead; slower,
notice 'no_frame_cache') when it would leave < 3 GB free. Stale tmp dirs of dead bridges are removed on start.

analyze writes into that private work dir and moves plan.spplan / report.json / analysis.npz into --out only when
the analysis finished, then writes --out/stillpoint_app.json (the cache manifest: clip identity, parameters and the
jitter summary shown in the app). A cancelled or failed run never leaves a half-written plan behind.
The summary carries report.json['quality'] (the engine's independent original-vs-stabilized measurement) normalized
to summary['quality'] = {method, units, metrics: [{key, label, original, stabilized}], windows?} when present, and
the engine v5 facts: summary['timecal'] (what the timing self-calibration did: state applied | confirmed | kept |
skipped | failed | off, the offset it found and its sigma), summary['fill'], summary['mesh'], summary['horizon']
(only for analyses that used them).

render reads sprender's `PROGRESS frame=i total=n` lines (no file-size guessing) and refuses to start when the
output volume cannot hold the expected file plus 1 GB.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

PROTOCOL = 2
MANIFEST = 'stillpoint_app.json'
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SPRENDER = os.path.join(ROOT, 'app', 'renderer', '.build', 'sprender')
WORK_ENV = 'STILLPOINT_WORK_DIR'
WORKERS_ENV = 'STILLPOINT_ANALYSIS_WORKERS'  # optional cap on the measurement worker pool
GB = 1e9
MIN_FREE_BYTES = 3 * GB          # analyze refuses to start below this; the frame cache must leave this much free
RENDER_MARGIN_BYTES = 1 * GB     # render: free space beyond the expected output size


# ============================================================================================ scratch space


def work_root() -> Optional[str]:
    """$STILLPOINT_WORK_DIR (absolute), or None when unset."""
    v = os.environ.get(WORK_ENV, '').strip()
    return os.path.abspath(os.path.expanduser(v)) if v else None


def tel_cache_dir(video: Optional[str] = None) -> str:
    """Telemetry cache dir: an existing cache in <root>/work/cache is reused (read-only use); new ones go to
    $STILLPOINT_WORK_DIR/cache when that is set."""
    legacy = os.path.join(ROOT, 'work', 'cache')
    wr = work_root()
    if not wr:
        return legacy
    if video is not None:
        try:
            from .telemetry import cache_file
            if os.path.exists(cache_file(video, legacy)):
                return legacy
        except Exception:
            pass
    return os.path.join(wr, 'cache')


def tmp_dir_for(out_dir: str, pid: Optional[int] = None) -> str:
    """The private work dir of an analyze run (see module docstring)."""
    pid = os.getpid() if pid is None else pid
    wr = work_root()
    return os.path.join(wr, 'tmp', f'analyze-{pid}') if wr else os.path.join(out_dir, f'.work-{pid}')


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale_tmp(root: Optional[str] = None) -> list[str]:
    """Remove analyze-<pid> dirs (frame caches can be many GB) of bridges that are no longer running."""
    wr = root or work_root()
    if not wr:
        return []
    base = os.path.join(wr, 'tmp')
    removed = []
    try:
        names = os.listdir(base)
    except OSError:
        return []
    for n in names:
        m = re.fullmatch(r'analyze-(\d+)', n)
        if not m or int(m.group(1)) == os.getpid() or _pid_alive(int(m.group(1))):
            continue
        shutil.rmtree(os.path.join(base, n), ignore_errors=True)
        removed.append(n)
    return removed


def free_bytes(path: str) -> int:
    p = os.path.abspath(path)
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    return int(shutil.disk_usage(p).free)


def frame_cache_bytes(n_frames: int, width: int, height: int, preview_width: int) -> int:
    """Size of pipeline.LumaCache's memmap: n_frames x preview-size 8-bit luma (+ the .npy header)."""
    from .video import gray_size
    gw, gh = gray_size(int(width), int(height), int(preview_width))
    return int(n_frames) * gw * gh + 128


def decide_frame_cache(cache_bytes: int, free: int, min_free: float = MIN_FREE_BYTES) -> bool:
    """Use the frame cache only when it leaves at least min_free bytes free."""
    return free - cache_bytes >= min_free


def engine_scratch_need() -> tuple[int, int]:
    """(temporary bytes the engine may write, free bytes it needs to start) — feature-detected from the engine
    (ENGINE v3: AnalyzeParams.max_temp_bytes + workspace.MIN_FREE_BYTES), else (0, MIN_FREE_BYTES)."""
    temp, need = 0, int(MIN_FREE_BYTES)
    try:
        from .pipeline import AnalyzeParams
        temp = int(getattr(AnalyzeParams(), 'max_temp_bytes', 0) or 0)
        from . import workspace
        need = max(need, temp + int(getattr(workspace, 'MIN_FREE_BYTES', 0)))
    except Exception:
        pass
    return temp, need


def _engine_frame_cache() -> Optional[tuple[bool, int]]:
    """(luma_cache default, preview_width) when the engine has a frame cache (feature-detected), else None."""
    try:
        from .pipeline import AnalyzeParams
        p = AnalyzeParams()
        if hasattr(p, 'luma_cache') and hasattr(p, 'preview_width'):
            return bool(p.luma_cache), int(p.preview_width)
    except Exception:
        pass
    return None


def analysis_dir_for(clip: str, analyses_root: str) -> str:
    """Mirror of the app's EngineConfig.analysisDir: <root>/<stem>-<first 10 hex of FNV-1a-64(abs path)>."""
    path = os.path.abspath(os.path.expanduser(clip))
    h = 0xcbf29ce484222325
    for b in path.encode('utf-8'):
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(analyses_root, f'{stem}-{h:016x}'[:len(stem) + 11])

# ============================================================================================ output channel

_OUT = None                      # the real stdout (JSON only)
_OUT_LOCK = threading.Lock()


def _setup_channels():
    """Keep fd 1 for JSON; point fd 1 of everything else (print, subprocesses) at stderr."""
    global _OUT
    if _OUT is not None:
        return
    try:
        _OUT = os.fdopen(os.dup(1), 'w', buffering=1)
        os.dup2(2, 1)
    except OSError:              # e.g. an embedded interpreter without real fds
        _OUT = sys.__stdout__
    sys.stdout = _LogTap(sys.stderr)


def emit(obj: dict) -> None:
    line = json.dumps(obj, default=_json_default, allow_nan=False, separators=(',', ':'))
    with _OUT_LOCK:
        (_OUT or sys.__stdout__).write(line + '\n')
        (_OUT or sys.__stdout__).flush()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return [_clean(x) for x in o.tolist()]
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return repr(o)


def _clean(x):
    """JSON has no NaN/inf: map them to None (recursively)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, np.floating):
        return _clean(float(x))
    if isinstance(x, np.integer):
        return int(x)
    return x


class _LogTap:
    """sys.stdout replacement: echoes to stderr and turns complete lines into {"type":"log"} events."""

    def __init__(self, err):
        self.err = err
        self.buf = ''
        self.listeners: list[Callable[[str], None]] = []

    def write(self, s):
        self.err.write(s)
        self.buf += s
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            line = line.rstrip()
            if line:
                emit(dict(type='log', message=line))
                for fn in list(self.listeners):
                    try:
                        fn(line)
                    except Exception:
                        pass
        return len(s)

    def flush(self):
        self.err.flush()

    def isatty(self):
        return False


# ============================================================================================ cancellation


class Cancelled(Exception):
    pass


class CancelToken(threading.Event):
    """Truthy when cancellation was requested. Works as a callable (`cancel()`), an Event (`is_set()`), and
    has `check()` / `raise_if_cancelled()` that raise Cancelled."""

    def __call__(self) -> bool:
        return self.is_set()

    def __bool__(self) -> bool:
        return self.is_set()

    def check(self):
        if self.is_set():
            raise Cancelled()

    raise_if_cancelled = check
    cancelled = property(lambda self: self.is_set())


_CANCEL = CancelToken()
_OWN_GROUP = False
_GRACE_S = 10.0


def _stop_children():
    """SIGTERM everything else in our process group (ffmpeg decoders, sprender), when we lead the group."""
    if not _OWN_GROUP:
        return
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except Exception:
        pass


def _exit_cancelled(forced: bool = False):
    emit(dict(type='cancelled', forced=forced) if forced else dict(type='cancelled'))
    try:
        (_OUT or sys.__stdout__).flush()
    except Exception:
        pass
    _stop_children()
    os._exit(130)


def _hard_exit():
    _exit_cancelled(forced=True)


def request_cancel(reason: str = 'signal'):
    if _CANCEL.is_set():
        return
    _CANCEL.set()
    emit(dict(type='log', message=f'cancelling ({reason})'))
    t = threading.Timer(_GRACE_S, _hard_exit)
    t.daemon = True
    t.start()


def _install_cancel_handlers(watch_stdin: Optional[bool]):
    global _OWN_GROUP

    def on_sig(signum, _frame):
        request_cancel(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    try:
        mode = os.fstat(0).st_mode
        is_pipe = stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)
    except OSError:
        is_pipe = False
    if watch_stdin is None:
        watch_stdin = is_pipe
    if watch_stdin:
        # launched by the app: lead our own process group so a forced cancel also stops ffmpeg / sprender
        try:
            if os.getpgrp() != os.getpid():
                os.setpgrp()
            _OWN_GROUP = True
        except OSError:
            pass

        def watch():
            try:
                while os.read(0, 4096):
                    pass
            except OSError:
                pass
            request_cancel('stdin closed')

        threading.Thread(target=watch, daemon=True, name='stdin-watch').start()


# ============================================================================================ progress


class Progress:
    """Stage-weighted overall progress with ETA. Stages are (key, label, est_seconds)."""

    def __init__(self, plan: list[tuple[str, str, float]], min_interval: float = 0.2):
        self.plan = list(plan)
        self.i = -1
        self.t0 = time.perf_counter()
        self.t_emit = 0.0
        self.min_interval = min_interval
        self.last_overall = 0.0
        self.eta = None
        self.stage_f = 0.0
        self.message = ''

    def _total(self):
        return sum(max(w, 1e-6) for _, _, w in self.plan)


    def _done(self):
        return sum(max(w, 1e-6) for _, _, w in self.plan[:max(self.i, 0)])

    def start(self, key: str, message: str = '', label: Optional[str] = None, est: Optional[float] = None):
        """Advance to the next planned stage with this key (skipping stages that did not happen), or append."""
        first = self.i if (self.i >= 0 and self.plan[self.i][0] == key and self.stage_f == 0.0) else self.i + 1
        j = next((j for j in range(first, len(self.plan)) if self.plan[j][0] == key), None)
        if j is None:
            self.plan.insert(self.i + 1, (key, label or key, est if est is not None else 1.0))
            j = self.i + 1
        elif label or est is not None:
            k, lb, w = self.plan[j]
            self.plan[j] = (k, label or lb, est if est is not None else w)
        self.i = j
        self.stage_f = 0.0
        self.message = message
        self._emit(force=True)

    def update(self, stage_fraction: float, message: Optional[str] = None):
        self.stage_f = float(min(max(stage_fraction, 0.0), 1.0))
        if message is not None:
            self.message = message
        self._emit()

    def overall(self) -> float:
        if self.i < 0:
            return 0.0
        w = max(self.plan[self.i][2], 1e-6)
        return min(0.999, (self._done() + w * self.stage_f) / self._total())

    def _emit(self, force: bool = False):
        now = time.perf_counter()
        if not force and now - self.t_emit < self.min_interval:
            return
        self.t_emit = now
        f = max(self.overall(), self.last_overall)       # never goes backwards
        self.last_overall = f
        el = now - self.t0
        eta = None
        if f >= 0.02 and el >= 2.0:
            raw = el * (1.0 - f) / f
            self.eta = raw if self.eta is None else 0.7 * self.eta + 0.3 * raw
            eta = self.eta
        key, label, _ = self.plan[self.i] if self.i >= 0 else ('start', 'Starting', 0)
        emit(dict(type='progress', stage=key, label=label, fraction=round(f, 4), stage_fraction=round(self.stage_f, 4),
                  message=self.message, elapsed_s=round(el, 2), eta_s=None if eta is None else round(eta, 1)))

    def native(self, key: str, label: str, overall: Optional[float], message: Optional[str]):
        """An overall fraction reported by the engine itself."""
        if overall is not None:
            self.last_overall = max(self.last_overall, min(max(float(overall), 0.0), 0.999))
        changed = self.i < 0 or self.plan[self.i][0] != key or self.plan[self.i][1] != label
        if changed:
            self.plan.insert(self.i + 1, (key, label, 0.0))
            self.i += 1
        if message is not None:
            self.message = message
        self.stage_f = 0.0
        self._emit(force=changed)

    def finish(self, message: str = 'Done'):
        self.i = len(self.plan) - 1
        self.stage_f = 1.0
        self.last_overall = 1.0
        el = time.perf_counter() - self.t0
        key, label, _ = self.plan[-1] if self.plan else ('done', 'Done', 0)
        emit(dict(type='progress', stage=key, label=label, fraction=1.0, stage_fraction=1.0, message=message,
                  elapsed_s=round(el, 2), eta_s=0.0))


# ============================================================================================ probe

LABELS = dict(telemetry='Reading gyro', path='Smoothing the path', measure='Measuring jitter',
              finalize='Writing plan', render='Rendering')


def _fov_deg(out_w: float, fx: float) -> float:
    return math.degrees(2.0 * math.atan(out_w / 2.0 / fx))


def _fx_for_fov(out_w: float, deg: float) -> float:
    return out_w / 2.0 / math.tan(math.radians(deg) / 2.0)


def gyro_status(tel) -> dict:
    """Badge for the clip list: label, level (good | limited | unsupported) and a one-line detail."""
    if tel is None:
        return dict(label='No gyro data', level='unsupported',
                    detail='No DJI motion metadata in this file, so it cannot be stabilized.')
    if tel.eis_baked:
        name = str(tel.extra.get('eis_status_name') or 'on').replace('EIS_', '').replace('_', ' ').lower()
        return dict(label='In-camera EIS on — not supported yet', level='unsupported',
                    detail=f'The camera already warped the picture (EIS mode: {name}). Record with EIS off to stabilize it in Stillpoint.')
    if tel.has_highrate:
        khz = tel.imu_rate / 1000.0
        lab = f'{khz:.0f} kHz gyro' if abs(khz - round(khz)) < 0.05 else f'{khz:.1f} kHz gyro'
        return dict(label=lab, level='good', detail=f'{tel.imu_rate:.0f} Hz orientation samples, EIS off.')
    return dict(label=f'{tel.imu_rate:.0f} Hz attitude — limited', level='limited',
                detail='Only per-frame camera attitude in this mode: slow shake is removed, fine jitter less so.')


PROBE_TELEMETRY_ENV = 'STILLPOINT_PROBE_TELEMETRY'   # auto | quick | full
PROBE_FULL_MAX_FRAMES = 1800     # auto: parse (and cache) the whole gyro track of clips up to 30 s at 60p


def _probe_telemetry(path: str, n_frames: int, mode: str):
    """Telemetry for the clip list. Returns (tel, scope) with scope 'full' | 'cached' | 'quick'.

    Every frame's metadata is one small random read (~0.4 ms on an SD card: ~12 s for an 8-minute clip), so for
    long clips probe takes telemetry.probe_telemetry's quick look (first 3 s + a sparse clock check, well under a
    second) unless a full parse is already cached; the analysis parses (and caches) the full track anyway.
    auto: cached -> that; short clip -> full parse; else quick (full after all if the quick look finds a joined file).
    quick / full: force (a cached full parse is still used by quick)."""
    from . import telemetry as _tel
    cache = tel_cache_dir(path)
    if os.path.exists(_tel.cache_file(path, cache)):
        return _tel.load_telemetry(path, cache_dir=cache), 'cached'
    if mode == 'full' or (mode == 'auto' and n_frames <= PROBE_FULL_MAX_FRAMES):
        return _tel.load_telemetry(path, cache_dir=cache), 'full'
    tel = _tel.probe_telemetry(path)
    if mode == 'auto' and tel.extra.get('quick', {}).get('discontinuities'):
        return _tel.load_telemetry(path, cache_dir=cache), 'full'       # joined file: segments need every frame
    return tel, 'quick'


def cmd_probe(a) -> dict:
    from . import video as _video
    path = os.path.abspath(os.path.expanduser(a.clip))
    if not os.path.isfile(path):
        raise BridgeError('not_found', f'No such file: {path}')
    mode = (getattr(a, 'telemetry', None) or os.environ.get(PROBE_TELEMETRY_ENV) or 'auto').strip().lower()
    if mode not in ('auto', 'quick', 'full'):
        raise BridgeError('bad_args', f'--telemetry {mode}: expected auto, quick or full')
    st = os.stat(path)
    info = _video.probe(path)
    tel, tel_err, scope = None, None, None
    try:
        tel, scope = _probe_telemetry(path, int(info['n_frames']), mode)
    except Exception as e:                       # not a DJI clip / no djmd track
        tel_err = f'{type(e).__name__}: {e}'
    W, H = int(info['width']), int(info['height'])
    fps = float(info['fps'])
    n = int(info['n_frames'])
    dur = float(info['frame_pts'][-1] - info['frame_pts'][0] + 1.0 / fps) if n else float(info.get('duration', 0))
    gs = gyro_status(tel)
    out = dict(path=path, name=os.path.basename(path), size_bytes=int(st.st_size), mtime_ns=int(st.st_mtime_ns),
               width=W, height=H, fps=fps, fps_fraction=info.get('fps_fraction'), n_frames=n, duration_s=dur,
               codec=info.get('codec'), bit_depth=int(info.get('bit_depth') or 8), pix_fmt=info.get('pix_fmt'),
               color_transfer=info.get('color_transfer'), color_primaries=info.get('color_primaries'),
               gyro=gs, supported=gs['level'] != 'unsupported', telemetry_error=tel_err, telemetry_scope=scope)
    if tel is not None:
        from .smooth import fx_for_crop_area
        L = tel.lens
        fov_lo = _fov_deg(W, fx_for_crop_area(L, W, H, W, H, 0.45))
        fov_hi = _fov_deg(W, fx_for_crop_area(L, W, H, W, H, 0.85))
        default = float(min(max(100.0, fov_lo), fov_hi))   # M1-validated framing on the O3 (~Gyroflow's crop)
        out.update(camera=tel.camera, product=tel.extra.get('product'), imu_rate=float(tel.imu_rate),
                   has_highrate=bool(tel.has_highrate), eis_baked=bool(tel.eis_baked),
                   eis_status=tel.extra.get('eis_status_name'), segments=len(tel.segments or []),
                   lens_fx=float(L.fx), warnings=list(tel.extra.get('warnings') or []),
                   horizon_lock_supported=bool(_engine_has_horizon_lock() and tel.gravity_q is not None),
                   fov=dict(min_deg=round(fov_lo, 1), max_deg=round(fov_hi, 1), default_deg=round(default, 1)))
    else:
        out.update(camera=_camera_from_tags(info), imu_rate=0.0, has_highrate=False, eis_baked=False,
                   warnings=[], horizon_lock_supported=False, fov=None)
    try:
        out['options'] = option_info(tel.camera if tel is not None else None, bool(out['horizon_lock_supported']))
    except Exception as e:                       # an engine without the v5 options: the app hides them
        out['options'] = None
        emit(dict(type='log', message=f'options unavailable: {type(e).__name__}: {e}'))
    fc = _engine_frame_cache()
    temp, need = engine_scratch_need()
    fcb = frame_cache_bytes(n, W, H, fc[1]) if (fc and fc[0] and n) else 0
    out['scratch'] = dict(frame_cache_bytes=fcb, temp_bytes=max(temp, fcb), min_free_bytes=need)
    return out


def _camera_from_tags(info: dict) -> str:
    enc = str((info.get('format_tags') or {}).get('encoder', '') or '')
    return enc or 'Unknown camera'


def _engine_has_horizon_lock() -> bool:
    try:
        from .smooth import SmoothParams
        return 'horizon_lock' in {f.name for f in __import__('dataclasses').fields(SmoothParams)}
    except Exception:
        return False


# ============================================================================================ engine v5 options
#
# The analysis options Stillpoint.app shows, their defaults per camera, what the engine supports and which of them
# cannot be combined. The app never hard-codes any of this: it starts every clip from probe's result['options'].
#
#   option (protocol name)  AnalyzeParams field(s)            app control
#   horizon                 horizon_lock (strength, 0 = off)  'Horizon lock' switch + Strength slider
#                           roll_limit_deg                    'Bank limit' slider (0-45 deg, 0 = fully level)
#   fill                    fill, fill_overscan               'Full-frame fill' switch
#   mesh                    mesh_residual                     'Max quality' switch
#   timecal                 timecal                           readout only ('Timing auto-calibration')
#   blur                    synth_blur                        (not in the app)

# Per-camera overrides of AnalyzeParams' defaults: {substring of Telemetry.camera: {option: value}} with option in
# horizon_lock (strength 0..1), roll_limit_deg, fill (bool), fill_overscan, mesh (bool), timecal (bool).
# THIS is the one place to change the defaults a camera's clips start with in the app (and in `app_bridge analyze`
# without flags); cameras / options not listed keep AnalyzeParams' own defaults, so the CLI agrees unless listed.
# The stillpoint CLI (`stillpoint.cli analyze`) resolves its defaults through the same helper (resolve_for_video);
# AnalyzeParams' own defaults (what scripts / the gate scoreboard construct directly) stay camera-independent.
# Only options the ProRes decision gate shows as a real win with no meaningful regression go here
# (work/gate/v6/decision.md, 2026-09-30):
#   DJI O3: full-frame fill -- HF -14.5 % [-24, -4], 2-8 Hz -15.7 %, roll -7.9 %, jumps >1 px 13 -> 8, nothing worse,
#           +0-3 % analysis time. OA4 / O4 Pro: fill does not pass (O4 Pro calm +11.7 %), so no entry.
#   Mesh ('Max quality') is not a default on any camera (O3: no decisive HF edge over fill, more >1 px jumps, slower).
CAMERA_OPTION_DEFAULTS: dict[str, dict] = {
    'DJI O3': {'fill': True, 'fill_overscan': 0.06},
}

# Caveats the app shows next to an option for a camera (engine knowledge, one short sentence).
CAMERA_OPTION_NOTES: dict[str, dict[str, str]] = {
    'O4 Pro': {'horizon': "Not reliable on O4 Pro yet: its gravity estimate is 6–10° off in turns."},
}

HORIZON_STRENGTH_ON = 1.0        # strength of a bare --horizon-lock (the switch without touching the slider)
FILL_OVERSCAN_ON = 0.06          # fill_overscan with --fill when not given (the 2026-09-29 scoreboard setting)
ROLL_LIMIT_RANGE = (0.0, 45.0)   # 'Bank limit' slider, degrees
# analysis time relative to the same run without the option (pre-flight estimates): mesh = two extra tracking passes
# (+41 % on DJI_0027, +138 s on DJI_0034 under load, 2026-09-30); fill = selection + alignment (+2.5-25 s per clip)
OPTION_TIME_FACTORS = {'mesh': 1.4, 'fill': 1.05}
# ... per camera (substring of Telemetry.camera). The mesh stage adds ~26-34 ms per frame whatever the camera, so the
# factor depends on how heavy the default analysis is: O3 (closed loop) x1.3-1.5, OA4 / O4 Pro (light) x5-7
# (gate v6: OA4_0012 962 s vs 145 s clean; O4_0004 673 s vs 95 s under load; OA4 fill 185 s vs 145 s).
CAMERA_TIME_FACTORS: dict[str, dict[str, float]] = {
    'Osmo Action 4': {'mesh': 6.6, 'fill': 1.25},
    'O4 Pro': {'mesh': 6.5, 'fill': 1.1},
}
_EXCLUSIVE_CANDIDATES = (('fill', 'fill', True), ('mesh', 'mesh_residual', True), ('blur', 'synth_blur', 'auto'))


def _camera_match(camera: Optional[str], table: dict) -> dict:
    """Merge the entries of `table` whose key is a substring of `camera` (in table order)."""
    out: dict = {}
    for key, vals in table.items():
        if camera and key and key in camera:
            out.update(vals)
    return out


def option_defaults(camera: Optional[str] = None) -> dict:
    """Resolved defaults of the app's options for a camera: AnalyzeParams' defaults + CAMERA_OPTION_DEFAULTS.
    horizon_lock is the strength (0 = off); horizon_strength is the strength the switch turns on with."""
    from .pipeline import AnalyzeParams
    p = AnalyzeParams()
    d = dict(horizon_lock=float(getattr(p, 'horizon_lock', 0.0) or 0.0),
             roll_limit_deg=float(getattr(p, 'roll_limit_deg', 0.0) or 0.0),
             fill=bool(getattr(p, 'fill', False)),
             fill_overscan=float(getattr(p, 'fill_overscan', 0.0) or 0.0) or FILL_OVERSCAN_ON,
             mesh=bool(getattr(p, 'mesh_residual', False)),
             timecal=bool(getattr(p, 'timecal', False)),
             synth_blur=str(getattr(p, 'synth_blur', 'off') or 'off'))
    d.update(_camera_match(camera, CAMERA_OPTION_DEFAULTS))
    d['horizon_lock'] = float(min(max(float(d['horizon_lock'] or 0.0), 0.0), 1.0))
    d['horizon_strength'] = d['horizon_lock'] if d['horizon_lock'] > 0 else HORIZON_STRENGTH_ON
    return d


def option_support() -> dict:
    """Which options this engine has (feature-detected on AnalyzeParams / SmoothParams)."""
    from .pipeline import AnalyzeParams
    p = AnalyzeParams()
    return dict(horizon=bool(_engine_has_horizon_lock() and hasattr(p, 'roll_limit_deg')), fill=hasattr(p, 'fill'),
                mesh=hasattr(p, 'mesh_residual'), timecal=hasattr(p, 'timecal'), blur=hasattr(p, 'synth_blur'))


def exclusive_pairs() -> list[list[str]]:
    """Option pairs the engine refuses to combine, found by asking pipeline.check_params itself (so the app follows
    the engine when the restriction is lifted)."""
    from . import pipeline
    check = getattr(pipeline, 'check_params', None)
    if check is None:
        return []
    cands = [(n, f, v) for n, f, v in _EXCLUSIVE_CANDIDATES if hasattr(pipeline.AnalyzeParams(), f)]
    out = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            prm = pipeline.AnalyzeParams()
            setattr(prm, cands[i][1], cands[i][2])
            setattr(prm, cands[j][1], cands[j][2])
            try:
                check(prm)
            except ValueError:
                out.append([cands[i][0], cands[j][0]])
    return out


def option_info(camera: Optional[str], gravity: bool) -> dict:
    """probe's result['options']: defaults for this camera, support, exclusive pairs, per-camera notes, the slider
    ranges and the analysis-time factors. Dict keys that are option names are single words (the app decodes with
    a snake_case -> camelCase strategy that also rewrites dictionary keys)."""
    sup = option_support()
    sup['horizon'] = bool(sup['horizon'] and gravity)
    return dict(defaults=option_defaults(camera), supported=sup, exclusive=exclusive_pairs(),
                notes=_camera_match(camera, CAMERA_OPTION_NOTES),
                ranges=dict(horizon_strength=[0.1, 1.0], roll_limit_deg=list(ROLL_LIMIT_RANGE)),
                time_factor=time_factors(camera))


def time_factors(camera: Optional[str] = None) -> dict:
    """Analysis-time factors of the options for a camera (OPTION_TIME_FACTORS + CAMERA_TIME_FACTORS)."""
    return dict(OPTION_TIME_FACTORS, **_camera_match(camera, CAMERA_TIME_FACTORS))


def _camera_for(video: str) -> Optional[str]:
    """Camera name for option defaults (cached gyro parse, else telemetry's quick look). Only needed when
    CAMERA_OPTION_DEFAULTS has entries."""
    try:
        from . import telemetry as _tel
        cache = tel_cache_dir(video)
        if os.path.exists(_tel.cache_file(video, cache)):
            return _tel.load_telemetry(video, cache_dir=cache).camera
        return _tel.probe_telemetry(video).camera
    except Exception:
        return None


def resolve_options(a, camera: Optional[str]) -> dict:
    """The analyze options from the command line, with every option that was not given taken from
    option_defaults(camera). Tolerates namespaces without the v5 fields (older callers / tests)."""
    d = option_defaults(camera)
    opt = lambda k: getattr(a, k, None)                          # noqa: E731
    hl = opt('horizon_lock')
    strength = d['horizon_lock'] if hl is None else float(hl or 0.0)     # False / 0 -> off
    fill = d['fill'] if opt('fill') is None else bool(opt('fill'))
    # fill / mesh / synth-blur are exclusive: an option the caller asked for explicitly wins over a fill that is
    # only on because it is this camera's default (e.g. `--mesh` on an O3 clip).
    explicit_exclusive = bool(opt('mesh')) or (opt('synth_blur') not in (None, '', 'off'))
    if opt('fill') is None and explicit_exclusive:
        fill = False
    no_tc = opt('no_timecal')                                     # legacy namespaces
    timecal = d['timecal'] if opt('timecal') is None else bool(opt('timecal'))
    if no_tc:
        timecal = False
    return dict(horizon_lock=strength,
                roll_limit_deg=d['roll_limit_deg'] if opt('roll_limit') is None else float(opt('roll_limit') or 0.0),
                fill=fill,
                fill_overscan=(float(opt('fill_overscan')) if opt('fill_overscan') is not None
                               else (d['fill_overscan'] if fill else 0.0)),
                mesh=d['mesh'] if opt('mesh') is None else bool(opt('mesh')),
                synth_blur=opt('synth_blur') or d['synth_blur'],
                blur_smooth=float(opt('blur_smooth') or 0.0),
                timecal=timecal)


def resolve_for_video(a, video: str) -> tuple[Optional[str], dict]:
    """(camera, resolve_options(a, camera)) for a clip: the one per-camera resolution used by `app_bridge analyze`
    and `stillpoint.cli analyze` (the camera is only looked up when CAMERA_OPTION_DEFAULTS has entries)."""
    camera = _camera_for(video) if CAMERA_OPTION_DEFAULTS else None
    return camera, resolve_options(a, camera)


# ============================================================================================ analyze


class BridgeError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _analyze_supports() -> tuple[bool, bool]:
    from . import pipeline
    try:
        ps = inspect.signature(pipeline.analyze).parameters
    except (TypeError, ValueError):
        return False, False
    return 'progress' in ps, 'cancel' in ps


def _normalize_native(args, kwargs) -> tuple[Optional[str], Optional[float], Optional[str]]:
    stage = frac = msg = None
    if args and isinstance(args[0], dict):
        d = args[0]
        stage = d.get('stage')
        frac = d.get('fraction', d.get('frac', d.get('progress')))
        msg = d.get('message', d.get('msg'))
    else:
        for x in args:
            if isinstance(x, str):
                if stage is None:
                    stage = x
                elif msg is None:
                    msg = x
            elif isinstance(x, (int, float, np.floating)) and not isinstance(x, bool) and frac is None:
                frac = float(x)
    stage = kwargs.get('stage', stage)
    frac = kwargs.get('fraction', kwargs.get('frac', frac))
    msg = kwargs.get('message', kwargs.get('msg', msg))
    return (None if stage is None else str(stage)), (None if frac is None else float(frac)), \
        (None if msg is None else str(msg))


# pipeline.analyze(progress=...) stage names -> the four steps the app shows (key, label)
_NATIVE_STAGES = {
    'telemetry': ('telemetry', 'Reading gyro'), 'calibration': ('telemetry', 'Calibrating'),
    'decode': ('telemetry', 'Decoding frames'), 'crop': ('path', 'Choosing the crop'),
    'path': ('path', 'Smoothing the path'), 'fold': ('path', 'Refining the path'), 'final': ('path', 'Final path'),
    'measure': ('measure', 'Measuring jitter'), 'quality': ('quality', 'Checking the result'),
    'fill': ('measure', 'Planning the border fill'), 'mesh': ('measure', 'Measuring micro-jitter'),
    'write': ('finalize', 'Writing plan'), 'done': ('finalize', 'Writing plan'),
}
# user-facing sub-messages (the engine's own messages are for logs)
_NATIVE_MESSAGES = {
    'telemetry': 'Reading the gyro track', 'calibration': 'Checking gyro timing against the picture',
    'decode': 'Decoding analysis frames', 'crop': 'Fitting the crop inside every frame',
    'path': 'Solving the smooth camera path', 'fold': 'Folding measured jitter back into the path',
    'final': 'Solving the final path', 'quality': 'Measuring original vs stabilized, independently of the loop',
    'fill': 'Choosing neighbouring frames for the corners (Full-frame fill)',
    'mesh': 'Tracking the picture for the micro-jitter mesh (Max quality)',
    'write': 'Writing the plan', 'done': 'Writing the plan',
}


def map_native_stage(stage: Optional[str]) -> tuple[str, str]:
    st = (stage or '').lower()
    for pre, kl in _NATIVE_STAGES.items():
        if st.startswith(pre):
            return kl
    return (st or 'analyze'), (st.replace('_', ' ').capitalize() or 'Analyzing')


class _NativeProgress:
    """Adapter for pipeline.analyze(progress=...) callbacks (stage, OVERALL fraction, message) — tolerant of other
    shapes. Emits through the shared Progress so the fraction stays monotonic when the bridge adds its own steps."""

    def __init__(self, prog: Progress, n_passes: int = 3):
        self.prog = prog
        self.n_passes = n_passes

    def __call__(self, *args, **kwargs):
        stage, frac, msg = _normalize_native(args, kwargs)
        key, label = map_native_stage(stage)
        if (stage or '').lower().startswith('measure'):
            msg = self.pass_message(stage or '', msg or '')
        else:
            st = (stage or '').lower()
            msg = next((m for pre, m in _NATIVE_MESSAGES.items() if st.startswith(pre)), msg)
        self.prog.native(key, label, frac, msg)
        if _CANCEL.is_set():
            raise Cancelled()


    def pass_message(self, stage: str, msg: str) -> str:
        """'measure1' + 'measuring pass 1: 812/2964 pairs' -> 'pass 2 of 3 · 812/2964 frames'."""
        m = re.search(r'(\d+)$', stage)
        k = int(m.group(1)) + 1 if m else None
        tag = f'pass {k} of {max(self.n_passes, k)}' if k else 'measuring'
        c = re.search(r'(\d+)\s*/\s*(\d+)', msg)
        return f'{tag} · {c.group(1)}/{c.group(2)} frames' if c else tag


class _Patched:
    """Fallback progress for engines without analyze(progress=): wrap the stage functions pipeline.analyze calls
    (load_telemetry, smooth.optimize_path, pipeline.measure_plan) to learn where it is, and use the per-pair
    progress callback measure_plan already exposes. Cancellation is honoured at those checkpoints."""

    def __init__(self, prog: Progress, n_meas: int, n_frames: int):
        self.prog = prog
        self.n_meas = n_meas
        self.n_frames = n_frames
        self.i_meas = 0
        self.saved = []

    def __enter__(self):
        from . import pipeline, smooth, telemetry
        prog = self

        def patch(mod, name, wrapper):
            orig = getattr(mod, name)
            self.saved.append((mod, name, orig))
            setattr(mod, name, wrapper(orig))

        def w_tel(orig):
            def f(*a, **k):
                _CANCEL.check()
                prog.prog.start('telemetry', 'Reading the gyro track')
                r = orig(*a, **k)
                prog.prog.update(1.0)
                return r
            return f

        def w_opt(orig):
            def f(*a, **k):
                _CANCEL.check()
                prog.prog.start('path', 'Solving the crop-constrained path')
                r = orig(*a, **k)
                prog.prog.update(1.0)
                _CANCEL.check()
                return r
            return f

        def w_meas(orig):
            def f(plan, video, *a, **k):
                _CANCEL.check()
                prog.i_meas += 1
                n_pairs = max(1, int(getattr(plan, 'n_frames', prog.n_frames)) - 1)
                i, n = prog.i_meas, prog.n_meas
                tag = f'pass {i} of {n}' if n > 1 else 'measuring'
                prog.prog.start('measure', f'{tag} · starting')
                user = k.pop('progress', None)

                def cb(done_pairs, *rest):
                    prog.prog.update(done_pairs / n_pairs, f'{tag} · {min(int(done_pairs), n_pairs)}/{n_pairs} frames')
                    if user is not None:
                        user(done_pairs, *rest)
                    _CANCEL.check()
                k['progress'] = cb
                r = orig(plan, video, *a, **k)
                prog.prog.update(1.0)
                return r
            return f

        patch(telemetry, 'load_telemetry', w_tel)
        patch(smooth, 'optimize_path', w_opt)
        patch(pipeline, 'measure_plan', w_meas)
        return self

    def __exit__(self, *exc):
        for mod, name, orig in reversed(self.saved):
            setattr(mod, name, orig)
        return False


def _stage_plan(n_frames: int, loop_iters: int, cached_tel: bool) -> list[tuple[str, str, float]]:
    F = max(n_frames, 2)
    t_tel = 0.5 if cached_tel else 2.0 + F / 4000.0
    t_opt = 0.3 + 0.0025 * F
    t_meas = 2.0 + F / 28.0
    plan = [('telemetry', LABELS['telemetry'], t_tel), ('path', LABELS['path'], t_opt)]
    for i in range(loop_iters + 1):
        plan.append(('measure', LABELS['measure'], t_meas))
        if i < loop_iters:
            plan.append(('path', LABELS['path'], t_opt))
    if loop_iters > 0:
        plan.append(('path', LABELS['path'], t_opt))
    plan.append(('finalize', LABELS['finalize'], 0.5 + F / 5000.0))
    return plan


def _clip_identity(path: str) -> dict:
    st = os.stat(path)
    return dict(path=os.path.abspath(path), size_bytes=int(st.st_size), mtime_ns=int(st.st_mtime_ns))


def cmd_analyze(a) -> dict:
    from . import pipeline
    from .pipeline import AnalyzeParams
    video = os.path.abspath(os.path.expanduser(a.clip))
    if not os.path.isfile(video):
        raise BridgeError('not_found', f'No such file: {video}')
    out_dir = os.path.abspath(os.path.expanduser(a.out))
    dry = bool(getattr(a, 'dry_run', False))

    from . import video as _video
    info = _video.probe(video)
    n_frames = int(info['n_frames'])

    prm = AnalyzeParams(smoothness=float(a.smoothness), closed_loop_iters=int(a.loop_iters), verbose=True,
                        save_iter_plans=False)
    W, H = int(info['width']), int(info['height'])
    fov_used = None
    if a.fov is not None:
        if not (30.0 <= a.fov <= 150.0):
            raise BridgeError('bad_args', f'--fov {a.fov} out of range (30..150 degrees)')
        prm.out_fx = _fx_for_fov(W, float(a.fov))
        fov_used = float(a.fov)
    elif a.crop_area is not None:
        prm.crop_area = float(a.crop_area)
    else:
        prm.out_fx = _fx_for_fov(W, 100.0)
        fov_used = 100.0
    # options not given on the command line take this camera's defaults (option_defaults)
    _cam, v5 = resolve_for_video(a, video)
    if not 0.0 <= v5['horizon_lock'] <= 1.0:
        raise BridgeError('bad_args', f'--horizon-lock {v5["horizon_lock"]}: the strength is 0..1')
    if not ROLL_LIMIT_RANGE[0] <= v5['roll_limit_deg'] <= 90.0:
        raise BridgeError('bad_args', f'--roll-limit {v5["roll_limit_deg"]}: expected 0..90 degrees')
    if v5['horizon_lock'] > 0:
        if not _engine_has_horizon_lock():
            raise BridgeError('unsupported', 'This engine has no horizon lock.')
        if hasattr(prm, 'roll_limit_deg'):            # engine v5: horizon lock v2 strength
            prm.horizon_lock = v5['horizon_lock']
            prm.roll_limit_deg = v5['roll_limit_deg']
        else:
            prm.smooth_overrides = dict(prm.smooth_overrides or {}, horizon_lock=True)
    fields = dict(fill=v5['fill'], fill_overscan=v5['fill_overscan'], mesh_residual=v5['mesh'],
                  synth_blur=v5['synth_blur'], blur_smooth_w=v5['blur_smooth'], timecal=v5['timecal'])
    for k, v in fields.items():
        if hasattr(prm, k):
            setattr(prm, k, v)
        elif v not in (False, 0.0, 'off') and not (k == 'timecal' and v):
            raise BridgeError('unsupported', f'This engine has no {k} option.')
    if hasattr(pipeline, 'check_params'):
        try:
            pipeline.check_params(prm)
        except ValueError as e:
            raise BridgeError('bad_args', str(e))
    sf, mf = int(getattr(a, 'start_frame', 0) or 0), int(getattr(a, 'max_frames', 0) or 0)
    if sf or mf:                                      # a window of the clip (diagnostics, self-tests)
        if sf < 0 or mf < 0 or sf >= n_frames or not hasattr(prm, 'start_frame'):
            raise BridgeError('bad_args', f'--start-frame {sf} --max-frames {mf}: not a window of this clip')
        prm.start_frame, prm.max_frames = sf, mf
    workers = os.environ.get(WORKERS_ENV, '').strip()
    if workers:                                       # machine rule / tests: cap the measurement pool
        try:
            nw = max(1, int(workers))
        except ValueError:
            raise BridgeError('bad_args', f'{WORKERS_ENV}={workers!r} is not a worker count')
        if hasattr(prm, 'processes'):
            prm.processes = nw
        if hasattr(prm, 'max_workers_measure'):
            prm.max_workers_measure = min(int(prm.max_workers_measure or nw), nw)
    params = dict(smoothness=float(a.smoothness), fov_deg=fov_used,
                  crop_area=None if a.crop_area is None or fov_used is not None else float(a.crop_area),
                  horizon_lock=v5['horizon_lock'] > 0, loop_iters=int(a.loop_iters),
                  horizon_strength=v5['horizon_lock'], roll_limit_deg=v5['roll_limit_deg'],
                  fill=v5['fill'], fill_overscan=v5['fill_overscan'], mesh=v5['mesh'],
                  synth_blur=v5['synth_blur'], blur_smooth=v5['blur_smooth'], timecal=v5['timecal'])
    if sf or mf:
        params['window'] = [sf, mf]
    if dry:
        keys = ('smoothness', 'out_fx', 'crop_area', 'closed_loop_iters', 'horizon_lock', 'roll_limit_deg', 'fill',
                'fill_overscan', 'mesh_residual', 'synth_blur', 'blur_smooth_w', 'timecal', 'processes',
                'max_workers_measure', 'start_frame', 'max_frames')
        return dict(dry_run=True, params=params, analyze_params={k: getattr(prm, k) for k in keys if hasattr(prm, k)},
                    smooth_overrides=dict(prm.smooth_overrides or {}))

    os.makedirs(out_dir, exist_ok=True)
    swept = sweep_stale_tmp()
    if swept:
        emit(dict(type='log', message=f'removed stale scratch dirs: {", ".join(swept)}'))
    try:                                   # cache presence only changes the time estimate
        from .telemetry import cache_file
        cached = os.path.exists(cache_file(video, tel_cache_dir(video)))
    except Exception:
        cached = False
    prog = Progress(_stage_plan(n_frames, a.loop_iters, cached))
    prog.start('telemetry', 'Reading the gyro track')

    work = tmp_dir_for(out_dir)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    scratch = dict(dir=work, free_bytes=free_bytes(work), frame_cache_bytes=0, frame_cache=False)
    need = engine_scratch_need()[1]
    if scratch['free_bytes'] < need:
        shutil.rmtree(work, ignore_errors=True)
        raise BridgeError('disk_full', f'Only {scratch["free_bytes"] / GB:.1f} GB free on the disk that holds '
                                       f'Stillpoint\'s scratch folder ({os.path.dirname(work)}). Analysis needs at '
                                       f'least {need / GB:.1f} GB. Free up space and try again.')
    if hasattr(prm, 'telemetry_cache') and not getattr(prm, 'telemetry_cache', None):
        legacy = os.path.join(ROOT, 'work', 'cache')          # reuse a gyro parse made before the work dir moved
        try:
            from .telemetry import cache_file
            if work_root() and os.path.exists(cache_file(video, legacy)):
                prm.telemetry_cache = legacy
        except Exception:
            pass
    if getattr(prm, 'luma_cache', False) and hasattr(prm, 'preview_width'):
        need = frame_cache_bytes(n_frames, W, H, prm.preview_width)
        scratch['frame_cache_bytes'] = need
        if hasattr(prm, 'cache_dir'):
            prm.cache_dir = work
        if decide_frame_cache(need, scratch['free_bytes']):
            scratch['frame_cache'] = True
        else:
            prm.luma_cache = False
            emit(dict(type='notice', code='no_frame_cache',
                      message=f'Not enough free space for the frame cache ({need / GB:.1f} GB needed, '
                              f'{scratch["free_bytes"] / GB:.1f} GB free): frames are decoded again for each pass, '
                              f'which is slower.'))
    has_progress, has_cancel = _analyze_supports()
    kwargs = {}
    ctx = None
    if has_progress:
        kwargs['progress'] = _NativeProgress(prog, a.loop_iters + 1)
    else:
        ctx = _Patched(prog, a.loop_iters + 1, n_frames)
    if has_cancel:
        kwargs['cancel'] = _CANCEL
    t0 = time.perf_counter()
    try:
        _CANCEL.check()
        if ctx is not None:
            with ctx:
                report = pipeline.analyze(video, work, prm, **kwargs)
        else:
            report = pipeline.analyze(video, work, prm, **kwargs)
        if _CANCEL.is_set():
            raise Cancelled()
        prog.start('finalize', 'Summarizing jitter')
        summary = summarize(video, work, report)
        # publish atomically-ish: data files first, the manifest (what the app trusts) last
        for fn in ('plan.spplan', 'plan.spblur', 'report.json', 'analysis.npz'):
            src = os.path.join(work, fn)
            if os.path.exists(src):
                _move(src, os.path.join(out_dir, fn))
            elif fn == 'plan.spblur':
                _rm(os.path.join(out_dir, fn))        # a stale sidecar of an earlier analysis must not be used
        report_path = os.path.join(out_dir, 'report.json')
        rep = json.load(open(report_path))
        rep['plan'] = os.path.join(out_dir, 'plan.spplan')
        _write_json(report_path, rep)
        wall = time.perf_counter() - t0
        manifest = dict(protocol=PROTOCOL, kind='stillpoint-analysis', clip=_clip_identity(video), params=params,
                        created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), seconds=wall,
                        plan=os.path.join(out_dir, 'plan.spplan'), report=report_path,
                        timing=dict(wall_s=wall, engine_total_s=(rep.get('timings') or {}).get('total_s'),
                                    n_frames=n_frames, duration_s=_duration(info),
                                    frame_cache=scratch['frame_cache']),
                        engine=dict(native_progress=has_progress, native_cancel=has_cancel), summary=summary)
        _write_json(os.path.join(out_dir, MANIFEST), manifest)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    prog.finish('Analysis complete')
    return dict(manifest=os.path.join(out_dir, MANIFEST), plan=os.path.join(out_dir, 'plan.spplan'),
                summary=summary, seconds=time.perf_counter() - t0, frame_cache=scratch['frame_cache'])


def _duration(info: dict) -> float:
    n = int(info.get('n_frames') or 0)
    fps = float(info.get('fps') or 0)
    pts = info.get('frame_pts')
    if n and fps and pts is not None and len(pts):
        return float(pts[-1] - pts[0] + 1.0 / fps)
    return float(info.get('duration') or 0.0)


def _move(src: str, dst: str):
    """os.replace when possible, else copy + replace (scratch and --out may be on different volumes)."""
    try:
        os.replace(src, dst)
    except OSError:
        shutil.copy2(src, dst + '.tmp')
        os.replace(dst + '.tmp', dst)
        _rm(src)


def _write_json(path: str, obj):
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as fh:
        json.dump(_clean(obj), fh, indent=1, default=_json_default)
    os.replace(tmp, path)


# ============================================================================================ jitter summary


def _hp(x, fs, lo, hi=None, order=4):
    from scipy.signal import butter, sosfiltfilt
    if hi is None or hi >= 0.499 * fs:
        sos = butter(order, lo, btype='highpass', fs=fs, output='sos')
    else:
        sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x, axis=0)


def _px2(theta, f1920, aspect):
    """Combined displacement^2 (1080p-eq px) of small camera-frame rotations, like eval.jitter_metrics."""
    r = math.sqrt((1920.0 ** 2 + (1920.0 * aspect) ** 2) / 12.0)
    return (f1920 * theta[..., 0]) ** 2 + (f1920 * theta[..., 1]) ** 2 + (r * theta[..., 2]) ** 2


def summarize(video: str, analysis_dir: str, report: Optional[dict] = None) -> dict:
    """Jitter readout for the app. Original = the camera's own rotational shake (gyro, >2 Hz, and 8-30 Hz),
    expressed in 1080p-equivalent pixels of the stabilised output. Gyro-only and Stillpoint = the residual the
    closed loop MEASURED on the picture (vision; only where vision was trusted). Per-second windows included."""
    from .geom import qconj, qlog, qmul
    from .plan_build import camera_orientation_fn
    from .telemetry import load_telemetry
    from .types import TimeModel
    if report is None:
        report = json.load(open(os.path.join(analysis_dir, 'report.json')))
    tel = load_telemetry(video, cache_dir=tel_cache_dir(video))
    fs = float(tel.fps)
    F = int(tel.n_frames)
    out = report.get('out', {})
    out_w, out_h = float(out.get('w', tel.width)), float(out.get('h', tel.height))
    fx0 = float(out.get('min_out_fx'))
    f1920 = fx0 * 1920.0 / out_w
    aspect = out_h / out_w
    s = dict(units='px @1080p', f1920=f1920, fps=fs, n_frames=F, camera=tel.camera,
             hfov_deg=float(out.get('hfov_deg', _fov_deg(out_w, fx0))), out_w=int(out_w), out_h=int(out_h),
             zoom_max=float(report.get('zoom', {}).get('max', 1.0)),
             zoom_frac=float(report.get('zoom', {}).get('frac_zoomed', 0.0)))
    wl = max(4, int(round(fs * float(report.get('params', {}).get('window_s', 1.0)))))
    s['window_s'] = wl / fs
    if F >= 60:
        q = camera_orientation_fn(tel, TimeModel())(tel.frame_t)
        rel = qlog(qmul(qconj(q[:-1]), q[1:]))
        eps = np.vstack([np.zeros((1, 3)), np.cumsum(rel, axis=0)])
        e2 = _px2(_hp(eps, fs, 2.0), f1920, aspect)
        b8 = _px2(_hp(eps, fs, 8.0, 30.0 if fs > 61 else None), f1920, aspect)
        tr = int(round(0.5 * fs)) if F > int(fs) + 30 else 0
        sl = slice(tr, F - tr) if tr else slice(0, F)
        s['orig_hf_px'] = float(np.sqrt(e2[sl].mean()))
        s['orig_b8_30_px'] = float(np.sqrt(b8[sl].mean()))
        nw = max(1, F // wl)
        idx = np.minimum(np.arange(F) // wl, nw - 1)
        s['win_orig_px'] = np.sqrt(np.bincount(idx, e2, nw) / np.bincount(idx, None, nw)).tolist()
        s['orig_window_hf_px'] = float(np.sqrt(np.mean(np.square(s['win_orig_px']))))
    cl = report.get('closed_loop', {})
    s['gyro_only_px'] = cl.get('open_loop_window_hf_px')
    s['final_px'] = cl.get('composite_window_hf_px')
    its = [it for it in cl.get('iterations', []) if isinstance(it, dict) and it.get('stats')]
    if its:
        st = its[-1]['stats']
        s['final_b8_30_px'] = st.get('b8_30_px')
        s['trusted_frac'] = st.get('trusted_frac')
        s['gyro_only_b8_30_px'] = its[0]['stats'].get('b8_30_px')
    npz = os.path.join(analysis_dir, 'analysis.npz')
    if os.path.exists(npz):
        z = np.load(npz)
        if 'window_hf_open' in z.files and len(z['window_hf_open']):
            s['win_gyro_only_px'] = z['window_hf_open'].tolist()
        if 'window_hf_best' in z.files and len(z['window_hf_best']):
            s['win_final_px'] = z['window_hf_best'].tolist()
    # the closed loop grades itself (vision residual on its own output): label it so, never as the result
    s['final_method'] = 'closed-loop residual (self-measured)'
    for k, fn in (('timecal', timecal_summary), ('fill', fill_summary), ('mesh', mesh_summary),
                  ('horizon', horizon_summary)):
        v = fn(report)
        if v is not None:
            s[k] = v
    rq = report.get('quality')
    q = normalize_quality(rq)
    if q is not None:
        s['quality'] = q
    elif isinstance(rq, dict):
        if rq.get('error'):
            s['quality_error'] = str(rq['error'])[:300]
        s['quality_unparsed_keys'] = sorted(rq.keys())[:40]
    return _clean(s)


def _fnum(v) -> Optional[float]:
    """A finite (signed) number, else None."""
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


TIMECAL_CONFIRM_MS = 0.05        # an estimate within max(2 sigma, this) of the metadata timing "confirms" it


def timecal_summary(report: dict) -> Optional[dict]:
    """What the per-clip timing self-calibration did (report['calibration']['timecal']) for the app's readout:
      state  applied    a correction was applied (offset_ms; readout_pct / focal_pct / exposure_slope / box_pct if so)
             confirmed  the fit ran and its offset agrees with the metadata timing (|estimate| <= max(2 sigma, 0.05 ms))
             kept       the fit ran but was not confident / not confirmed, so the metadata timing was kept
             skipped    not run for this clip (no high-rate gyro, in-camera EIS, ...)
             failed     the calibration raised; the metadata timing was kept
             off        turned off for this analysis (--no-timecal)
    plus estimate_ms / sigma_ms (the fitted offset, also when it was not applied) and detail (the engine's reasons).
    None for analyses made before engine v5."""
    prm = report.get('params') or {}
    cal = report.get('calibration') or {}
    tc = cal.get('timecal')
    if not isinstance(tc, dict):
        if 'timecal' not in prm:
            return None
        if not prm.get('timecal'):
            return dict(state='off', detail='Turned off for this analysis: the metadata timing was used.')
        if prm.get('calibrate'):
            return dict(state='skipped', detail='The legacy calibration ran instead.')
        return dict(state='skipped', detail='Needs the high-rate gyro track; this clip only has per-frame attitude.')
    status = str(tc.get('status') or '')
    est, sig = tc.get('estimate') or {}, tc.get('sigma') or {}
    reasons = [str(r) for r in ((tc.get('decision') or {}).get('reasons') or [])]
    out = dict(estimate_ms=_fnum(est.get('offset_ms')), sigma_ms=_fnum(sig.get('offset_ms')),
               detail=('; '.join(reasons)[:400] or None))
    if status.startswith('skipped'):
        return dict(out, state='skipped', detail=status.partition(':')[2].strip() or status)
    if status.startswith('failed'):
        return dict(out, state='failed', detail=status[:300])
    if status != 'ok':
        return dict(out, state='kept', detail=status[:300] or out['detail'])
    app = tc.get('applied') or {}
    done = {}
    off = _fnum(app.get('offset_ms')) or 0.0
    if off != 0.0:
        done['offset_ms'] = off
    meta = _fnum(cal.get('readout_meta_ms'))
    if _fnum(app.get('readout_s')) and meta:
        done['readout_pct'] = (float(app['readout_s']) * 1e3 / meta - 1.0) * 100.0
    if (_fnum(app.get('focal_scale')) or 1.0) != 1.0:
        done['focal_pct'] = (float(app['focal_scale']) - 1.0) * 100.0
    if _fnum(app.get('exposure_slope')):
        done['exposure_slope'] = float(app['exposure_slope'])
    if (_fnum(app.get('exposure_scale')) or 1.0) != 1.0:
        done['box_pct'] = (float(app['exposure_scale']) - 1.0) * 100.0
    if done:
        return dict(out, state='applied', offset_ms=off, **{k: v for k, v in done.items() if k != 'offset_ms'})
    e, s = out['estimate_ms'], out['sigma_ms']
    ok = e is not None and abs(e) <= max(2.0 * (s or 0.0), TIMECAL_CONFIRM_MS)
    return dict(out, state='confirmed' if ok else 'kept')


def fill_summary(report: dict) -> Optional[dict]:
    """report['fill'] (full-frame border fill) -> how much of the output it synthesised; None when fill was off."""
    f = report.get('fill')
    if not isinstance(f, dict):
        return None
    if f.get('error'):
        return dict(error=str(f['error'])[:300])
    return _clean(dict(frames_frac=_num(f.get('frac_frames_filled')), pixels_frac_mean=_num(f.get('fill_frac_mean')),
                       pixels_frac_max=_num(f.get('fill_frac_max')), uncovered_frac_max=_num(f.get('uncovered_frac_max')),
                       max_offset=_num(f.get('max_offset'))))


def mesh_summary(report: dict) -> Optional[dict]:
    """report['mesh'] (mesh residual, 'Max quality') -> the size of the correction it baked in (1080p px)."""
    m = report.get('mesh')
    if not isinstance(m, dict):
        return None
    if m.get('error'):
        return dict(error=str(m['error'])[:300])
    return _clean(dict(offset_rms_px=_num(m.get('offset_rms_1080')), offset_max_px=_num(m.get('offset_max_1080')),
                       accepted_frac=_num((m.get('verify') or {}).get('accepted_frac'))))


def horizon_summary(report: dict) -> Optional[dict]:
    """report['smooth']['horizon'] (horizon lock v2) -> how much of the clip it levelled; None when it was off."""
    h = (report.get('smooth') or {}).get('horizon')
    if not isinstance(h, dict) or not (_num(h.get('strength')) or 0.0) > 0:
        return None
    return _clean(dict(strength=_num(h.get('strength')), roll_limit_deg=_num(h.get('roll_limit_deg')),
                       full_level_frac=_num(h.get('frac_full_level')), off_frac=_num(h.get('frac_off')),
                       available_frac=_num(h.get('frac_available'))))


# report.json['quality'] (the engine's independent measurement) -> summary['quality'], tolerant of layout:
#   {'original': {'hf_px':..}, 'stabilized': {..}} | {'hf': {'original':.., 'stabilized':..}} | {'orig_hf_px':..}
Q_METRICS = (
    ('hf', 'Shake above 2 Hz', ('hf_px', 'hf', 'window_hf_px', 'window_hf', 'hf2_px', 'hf_2hz_px', 'rms_hf_px')),
    ('calm', 'Calm cruise', ('calm_hf_px', 'calm_cruise_px', 'calm_cruise', 'calm_px', 'calm', 'cruise_px')),
    ('b8_30', 'Fine jitter 8–30 Hz', ('b8_30_px', 'b8_30', 'band_8_30_px', 'band_8_30', 'hf_8_30_px', 'b830_px')),
    ('jello', 'Wobble (jello)', ('jello_px', 'jello')),
)
Q_ORIG = ('original', 'orig', 'source', 'src', 'before', 'input')
Q_STAB = ('stabilized', 'stabilised', 'stab', 'stillpoint', 'output', 'out', 'after', 'final')
Q_FACTS = ('new_jumps_gt_1px', 'new_jumps_gt_0_5px', 'new_jumps_max_px', 'coverage_frac', 'crop_footprint_mean',
           'n_frames_scored', 'window_s')


def _num(v) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        return None
    v = float(v)
    return v if math.isfinite(v) and v >= 0 else None


def _q_value(q: dict, sides: tuple, aliases: tuple) -> Optional[float]:
    for sd in sides:                                   # nested by side
        d = q.get(sd)
        if isinstance(d, dict):
            for al in aliases:
                if _num(d.get(al)) is not None:
                    return _num(d.get(al))
    for al in aliases:                                 # nested by metric
        d = q.get(al)
        if isinstance(d, dict):
            for sd in sides:
                if _num(d.get(sd)) is not None:
                    return _num(d.get(sd))
    for al in aliases:                                 # flat: original_hf_px (engine v3) / hf_px_original
        for sd in sides:
            for k in (f'{sd}_{al}', f'{al}_{sd}'):
                if _num(q.get(k)) is not None:
                    return _num(q.get(k))
    return None


def _q_windows(q: dict) -> Optional[list]:
    """Per-window rows [{t0_s, t1_s, original, stabilized}] (the engine samples windows; t in clip seconds)."""
    w = q.get('windows') or q.get('per_window')
    rows = []
    if isinstance(w, list):
        for r in w:
            if not isinstance(r, dict):
                continue
            t0, t1 = _num(r.get('t0_s', r.get('t0'))), _num(r.get('t1_s', r.get('t1')))
            o, st = _q_value(r, Q_ORIG, Q_METRICS[0][2]), _q_value(r, Q_STAB, Q_METRICS[0][2])
            if t0 is not None and t1 is not None and t1 > t0 and (o is not None or st is not None):
                rows.append(dict(t0_s=t0, t1_s=t1, original=o, stabilized=st))
    elif isinstance(w, dict):                          # {'original': [...], 'stabilized': [...]} per window_s
        ws = _num(q.get('window_s')) or 1.0
        wo = next((w[k] for k in Q_ORIG if isinstance(w.get(k), (list, tuple))), None)
        wt = next((w[k] for k in Q_STAB if isinstance(w.get(k), (list, tuple))), None)
        if wo is not None and wt is not None and len(wo) == len(wt):
            rows = [dict(t0_s=i * ws, t1_s=(i + 1) * ws, original=_num(a), stabilized=_num(b))
                    for i, (a, b) in enumerate(zip(wo, wt))]
    return rows or None


def normalize_quality(q) -> Optional[dict]:
    """Canonical form of report.json['quality'] for the app, or None when absent / nothing recognizable.
    Only numbers the engine wrote are passed on (no derived percentages)."""
    if not isinstance(q, dict):
        return None
    metrics = []
    for key, label, aliases in Q_METRICS:
        o, st = _q_value(q, Q_ORIG, aliases), _q_value(q, Q_STAB, aliases)
        if o is not None or st is not None:
            metrics.append(dict(key=key, label=label, original=o, stabilized=st))
    if not metrics:
        return None
    method = q.get('method')
    if isinstance(method, dict):
        method = method.get('name') or method.get('description') or json.dumps(method)[:200]
    out = dict(method=str(method) if method else None, units=str(q.get('units') or 'px @1080p'), metrics=metrics)
    rows = _q_windows(q)
    if rows:
        out['windows'] = rows
    for k in Q_FACTS:
        if _num(q.get(k)) is not None:
            out[k] = _num(q.get(k))
    return out


def cmd_summary(a) -> dict:
    d = os.path.abspath(os.path.expanduser(a.dir))
    mp = os.path.join(d, MANIFEST)
    if not os.path.exists(mp):
        raise BridgeError('not_found', f'No analysis in {d}')
    m = json.load(open(mp))
    if a.recompute:
        m['summary'] = summarize(m['clip']['path'], d)
        _write_json(mp, m)
    return dict(manifest=mp, summary=m.get('summary'), params=m.get('params'), clip=m.get('clip'))


def cmd_adopt(a) -> dict:
    """Wrap an analysis made with the CLI (DIR/plan.spplan + report.json [+ analysis.npz]) as an app cache
    entry in --out (copies the files, writes the manifest + jitter summary)."""
    src = os.path.abspath(os.path.expanduser(a.dir))
    video = os.path.abspath(os.path.expanduser(a.clip))
    out_dir = os.path.abspath(os.path.expanduser(a.out))
    for fn in ('plan.spplan', 'report.json'):
        if not os.path.exists(os.path.join(src, fn)):
            raise BridgeError('not_found', f'{src} has no {fn}')
    rep = json.load(open(os.path.join(src, 'report.json')))
    if os.path.realpath(rep.get('video', '')) != os.path.realpath(video):
        raise BridgeError('mismatch', f"{src} was made from {rep.get('video')}, not {video}")
    os.makedirs(out_dir, exist_ok=True)
    for fn in ('plan.spplan', 'report.json', 'analysis.npz'):
        s_, d_ = os.path.join(src, fn), os.path.join(out_dir, fn)
        if os.path.exists(s_) and os.path.realpath(s_) != os.path.realpath(d_):
            shutil.copy2(s_, d_ + '.tmp')
            os.replace(d_ + '.tmp', d_)
    rep['plan'] = os.path.join(out_dir, 'plan.spplan')
    _write_json(os.path.join(out_dir, 'report.json'), rep)
    summary = summarize(video, out_dir, rep)
    prm = rep.get('params', {})
    hl = float(prm.get('horizon_lock') or 0.0)
    if not hl and (prm.get('smooth_overrides') or {}).get('horizon_lock'):
        hl = HORIZON_STRENGTH_ON                                  # pre-v5 boolean horizon lock
    params = dict(smoothness=float(prm.get('smoothness', 1.0)),
                  fov_deg=round(float(rep.get('out', {}).get('hfov_deg', 100.0)), 1), crop_area=None,
                  horizon_lock=hl > 0, loop_iters=int(prm.get('closed_loop_iters', 2)),
                  horizon_strength=hl, roll_limit_deg=float(prm.get('roll_limit_deg') or 0.0),
                  fill=bool(prm.get('fill', False)), fill_overscan=float(prm.get('fill_overscan') or 0.0),
                  mesh=bool(prm.get('mesh_residual', False)), synth_blur=str(prm.get('synth_blur') or 'off'),
                  blur_smooth=float(prm.get('blur_smooth_w') or 0.0), timecal=bool(prm.get('timecal', False)))
    manifest = dict(protocol=PROTOCOL, kind='stillpoint-analysis', clip=_clip_identity(video), params=params,
                    created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), seconds=float(rep.get('timings', {}).get('total_s', 0)),
                    plan=rep['plan'], report=os.path.join(out_dir, 'report.json'), adopted_from=src,
                    engine=dict(native_progress=None, native_cancel=None), summary=summary)
    _write_json(os.path.join(out_dir, MANIFEST), manifest)
    return dict(manifest=os.path.join(out_dir, MANIFEST), plan=rep['plan'], summary=summary)


# ============================================================================================ render

_PROGRESS_RE = [re.compile(r'(?i)\bprogress\b\D*(\d+)\s*/\s*(\d+)'),
                re.compile(r'(?i)\bframe[s]?\s*[=:]\s*(\d+)\s*/\s*(\d+)'),
                re.compile(r'(?i)\bPROGRESS\b.*?\bframe=(\d+)\b.*?\btotal=(\d+)')]
_RESULT_RE = re.compile(r'RESULT frames=(\d+) wall=([\d.]+)s fps=([\d.]+)')


def parse_sprender_line(line: str) -> Optional[tuple[int, int]]:
    """(done, total) from a sprender progress line, if it has one."""
    for rx in _PROGRESS_RE:
        m = rx.search(line)
        if m:
            d, t = int(m.group(1)), int(m.group(2))
            if t > 0:
                return d, t
    return None


def _expected_bytes(codec: str, bitrate_mbps: float, w: int, h: int, fps: float, n: int) -> float:
    dur = n / max(fps, 1e-6)
    if codec == 'prores':
        bps = 3.55 * w * h * fps               # ProRes 422 HQ ~ 1.77 Gbit/s at 3840x2160p59.94 (Apple white paper)
    else:
        bps = bitrate_mbps * 1e6               # VideoToolbox ABR lands within ~1% (M1 renders: 178-181 Mbit/s)
    return bps / 8.0 * dur


def cmd_render(a) -> dict:
    from . import video as _video
    from .plan_io import read_plan
    video = os.path.abspath(os.path.expanduser(a.clip))
    plan = os.path.abspath(os.path.expanduser(a.plan))
    out = os.path.abspath(os.path.expanduser(a.out))
    exe = a.sprender or SPRENDER
    for p, what in ((video, 'clip'), (plan, 'plan'), (exe, 'renderer (app/renderer/build.sh)')):
        if not os.path.exists(p):
            raise BridgeError('not_found', f'Missing {what}: {p}')
    if a.codec not in ('hevc10', 'hevc10-speed', 'prores'):
        raise BridgeError('bad_args', f'unknown codec {a.codec}')
    info = _video.probe(video)
    pl = read_plan(plan)
    n_src = int(info['n_frames'])
    start = max(0, int(a.start_frame))
    n = (min(int(a.frames), n_src - start) if a.frames else n_src - start)
    n = min(n, pl.n_frames)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    ext = os.path.splitext(out)[1] or '.mov'
    part = os.path.join(os.path.dirname(out), '.' + os.path.basename(out) + '.partial' + ext)
    cmd = [exe, video, plan, part, '--start-frame', str(start), '--codec', a.codec,
           '--bitrate-mbps', str(a.bitrate_mbps), '--kernel', a.kernel, '--zero-base']
    a_blur = getattr(a, 'blur', None)
    blur = a_blur or (None if getattr(a, 'no_blur', False) else os.path.join(os.path.dirname(plan), 'plan.spblur'))
    if blur and os.path.exists(blur):                 # synthetic shutter sidecar (analyze --synth-blur)
        cmd += ['--blur', blur]
    elif a_blur:
        raise BridgeError('not_found', f'Missing synthetic-shutter sidecar: {a_blur}')
    if a.frames:
        cmd += ['--frames', str(int(a.frames))]
    expected = _expected_bytes(a.codec, a.bitrate_mbps, pl.out_w, pl.out_h, float(info['fps']), n)
    free = free_bytes(os.path.dirname(out) or '.')
    if free < expected * 1.1 + RENDER_MARGIN_BYTES:
        raise BridgeError('disk_full', f'Not enough free space in {os.path.dirname(out)}: this export needs about '
                                       f'{expected * 1.1 / GB:.1f} GB (+{RENDER_MARGIN_BYTES / GB:.0f} GB margin), '
                                       f'{free / GB:.1f} GB free. Pick another folder or a smaller format.')
    prog = Progress([('render', LABELS['render'], 1.0)], min_interval=0.25)
    prog.start('render', 'Starting the renderer')
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                            start_new_session=False)
    lines: list[str] = []
    errs: list[str] = []
    state = dict(parsed=None, result=None)

    def rd_out():
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            pr = parse_sprender_line(line)
            if pr:
                state['parsed'] = pr
            m = _RESULT_RE.search(line)
            if m:
                state['result'] = dict(frames=int(m.group(1)), wall_s=float(m.group(2)), fps=float(m.group(3)))
            elif line and not pr:
                sys.stderr.write('[sprender] ' + line + '\n')

    def rd_err():
        for line in proc.stderr:
            errs.append(line.rstrip())
            sys.stderr.write('[sprender] ' + line)

    th = [threading.Thread(target=rd_out, daemon=True), threading.Thread(target=rd_err, daemon=True)]
    for t in th:
        t.start()
    cancelled = False
    last = None
    while proc.poll() is None:
        if _CANCEL.is_set() and not cancelled:
            cancelled = True
            proc.terminate()
        time.sleep(0.1)
        pr = state['parsed']
        if pr is None or pr == last:        # progress only from sprender's own frame counter
            continue
        last = pr
        d, tot = pr
        el = time.perf_counter() - t0
        msg = f'{d}/{tot} frames'
        if el > 1.0 and d > 0:
            msg += f' · {d / el:.0f} fps'
        prog.update(d / max(tot, 1), msg)
    for t in th:
        t.join(timeout=2)
    if cancelled or _CANCEL.is_set():
        _rm(part)
        raise Cancelled()
    if proc.returncode != 0:
        _rm(part)
        tail = '\n'.join((errs or lines)[-8:])
        raise BridgeError('render_failed', f'sprender exited {proc.returncode}: {tail}')
    os.replace(part, out)
    prog.finish('Export complete')
    r = state['result'] or {}
    return dict(out=out, frames=r.get('frames', n), seconds=time.perf_counter() - t0, fps=r.get('fps'),
                bytes=os.path.getsize(out), codec=a.codec)


def _rm(p):
    try:
        os.remove(p)
    except OSError:
        pass


# ============================================================================================ main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='stillpoint.app_bridge')
    ap.add_argument('--grace', type=float, default=10.0, help='seconds to wait for a cooperative cancel')
    ap.add_argument('--watch-stdin', dest='watch_stdin', action='store_true', default=None)
    ap.add_argument('--no-watch-stdin', dest='watch_stdin', action='store_false')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('probe')
    p.add_argument('clip')
    p.add_argument('--telemetry', choices=('auto', 'quick', 'full'), default=None,
                   help='gyro-track depth (default: env STILLPOINT_PROBE_TELEMETRY, else auto)')
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser('analyze')
    p.add_argument('clip')
    p.add_argument('--out', required=True)
    p.add_argument('--smoothness', type=float, default=1.0)
    p.add_argument('--fov', type=float, default=None, help='output horizontal field of view, degrees')
    p.add_argument('--crop-area', type=float, default=None, help='analytic crop area (instead of --fov)')
    p.add_argument('--horizon-lock', type=float, nargs='?', const=HORIZON_STRENGTH_ON, default=None, metavar='S',
                   help='horizon lock strength 0..1 (the bare flag = 1.0, 0 = off; default: the camera default)')
    p.add_argument('--roll-limit', type=float, default=None, metavar='DEG',
                   help='with --horizon-lock: bank up to DEG degrees is kept (0 = fully level)')
    p.add_argument('--loop-iters', type=int, default=2)
    p.add_argument('--fill', action=argparse.BooleanOptionalAction, default=None,
                   help='full-frame border fill (engine v5; default: the camera default)')
    p.add_argument('--fill-overscan', type=float, default=None, metavar='F', help=f'with --fill (default {FILL_OVERSCAN_ON})')
    p.add_argument('--mesh', action=argparse.BooleanOptionalAction, default=None,
                   help='parallax mesh residual ("Max quality"; ~+40%% analysis time)')
    p.add_argument('--synth-blur', choices=('off', 'auto', 'angle'), default=None)
    p.add_argument('--blur-smooth', type=float, default=0.0, metavar='W')
    p.add_argument('--timecal', action=argparse.BooleanOptionalAction, default=None,
                   help='per-clip timing self-calibration (--no-timecal keeps the metadata timing)')
    p.add_argument('--dry-run', action='store_true', help='resolve and check the options, analyse nothing')
    p.add_argument('--start-frame', type=int, default=0, help='(diagnostics) first source frame of the window')
    p.add_argument('--max-frames', type=int, default=0, help='(diagnostics) window length in frames (0 = to the end)')
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser('render')
    p.add_argument('clip')
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--codec', default='hevc10')
    p.add_argument('--bitrate-mbps', type=float, default=180.0)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--frames', type=int, default=0)
    p.add_argument('--kernel', default='lanczos3')
    p.add_argument('--sprender', default=None)
    p.add_argument('--blur', default=None, help='synthetic shutter sidecar (default: plan.spblur next to the plan)')
    p.add_argument('--no-blur', action='store_true', help='ignore a plan.spblur next to the plan')
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser('summary')
    p.add_argument('dir')
    p.add_argument('--recompute', action='store_true')
    p.set_defaults(fn=cmd_summary)
    p = sub.add_parser('adopt')
    p.add_argument('dir')
    p.add_argument('--clip', required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(fn=cmd_adopt)
    return ap


def main(argv=None) -> int:
    global _GRACE_S
    a = build_parser().parse_args(argv)
    _setup_channels()
    _GRACE_S = float(a.grace)
    _install_cancel_handlers(a.watch_stdin)
    emit(dict(type='start', cmd=a.cmd, pid=os.getpid(), pgid=os.getpgrp(), protocol=PROTOCOL, work_root=work_root()))
    try:
        res = a.fn(a)
        emit(dict(type='result', cmd=a.cmd, **_clean(res)))
        return 0
    except Cancelled:
        _exit_cancelled()
    except BridgeError as e:
        emit(dict(type='error', code=e.code, message=e.message))
        return 1
    except NotImplementedError as e:
        emit(dict(type='error', code='unsupported', message=str(e)))
        return 1
    except Exception as e:  # noqa: BLE001 - report everything to the app
        if _CANCEL.is_set() or type(e).__name__ == 'AnalysisCancelled':
            _exit_cancelled()
        import traceback
        traceback.print_exc(file=sys.stderr)
        code = type(e).__name__
        if code == 'InsufficientDiskSpace' or getattr(e, 'errno', None) == 28:        # ENOSPC
            code = 'disk_full'
        emit(dict(type='error', code=code, message=str(e)))
        return 1
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(main())
