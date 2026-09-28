"""Where the engine may put temporary files, and how it cleans them up (ENGINE v3 resource safety).  Owner: ENGINE.

Rules (learned the hard way: a whole-clip frame cache of a 6.7-min clip wrote 16.5 GB next to the analysis output,
which lived in an iCloud-synced Desktop folder, filled the disk and froze the app):

* Every temporary file goes into a per-job directory under the WORK ROOT:
      $STILLPOINT_WORK_DIR  (default ~/Library/Application Support/Stillpoint/work — not synced by iCloud)
      <root>/jobs/<kind>-<pid>-<time>/
  never next to the video and never inside the output directory.
* Before heavy work the free space of the work root is checked (`preflight`) and a clear error is raised when it is
  too low (InsufficientDiskSpace).
* A JobDir is removed on normal exit, on exceptions and on cancel (context manager / try-finally), on interpreter
  exit (atexit) and on SIGTERM when the engine owns the SIGTERM handler (`sigterm_cancels`).  A hard kill (SIGKILL,
  os._exit) cannot run cleanup, so every new JobDir first sweeps job dirs whose owner process is gone.
* The engine keeps no whole-clip caches on disk any more (frames are streamed; see pipeline.FrameStream), so a job
  dir only ever holds small files.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import threading
import time
from typing import Callable, Optional

__all__ = ['work_root', 'JobDir', 'InsufficientDiskSpace', 'preflight', 'free_bytes', 'sweep_stale',
           'sigterm_cancels', 'is_under_desktop', 'cache_dir']

DEFAULT_WORK = os.path.join('~', 'Library', 'Application Support', 'Stillpoint', 'work')
MIN_FREE_BYTES = 2 * 1024 ** 3          # never start heavy work with less than this free on the work volume


class InsufficientDiskSpace(OSError):
    """Not enough free disk space for the requested work (message says how much is needed / free and where)."""


_REPO_WORK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'work'))
PENDING_MOVE_FALLBACK = os.path.join('~', 'Library', 'Application Support', 'Stillpoint', 'engine-work')


def work_root(create: bool = True) -> str:
    """The engine's work root ($STILLPOINT_WORK_DIR or ~/Library/Application Support/Stillpoint/work).

    Guard: the repo's old scratch dir (stillpoint/work, on the iCloud Desktop) is due to be MOVED to exactly the
    default path. While that move is pending (default path absent, stillpoint/work present) the engine must not
    create the move target (a `mv` onto an existing non-empty directory fails), so it uses
    ~/Library/Application Support/Stillpoint/engine-work instead."""
    env = os.environ.get('STILLPOINT_WORK_DIR')
    if env:
        p = os.path.abspath(os.path.expanduser(env))
    else:
        p = os.path.abspath(os.path.expanduser(DEFAULT_WORK))
        if not os.path.exists(p) and os.path.isdir(_REPO_WORK):
            p = os.path.abspath(os.path.expanduser(PENDING_MOVE_FALLBACK))
    if create:
        os.makedirs(p, exist_ok=True)
    return p


def cache_dir() -> str:
    """Small persistent caches (telemetry): <work root>/cache."""
    p = os.path.join(work_root(), 'cache')
    os.makedirs(p, exist_ok=True)
    return p


def is_under_desktop(path: str) -> bool:
    """True when `path` lies in ~/Desktop or ~/Documents (iCloud 'Desktop & Documents' sync)."""
    p = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    home = os.path.realpath(os.path.expanduser('~'))
    for d in ('Desktop', 'Documents'):
        base = os.path.join(home, d)
        if p == base or p.startswith(base + os.sep):
            return True
    return False


def free_bytes(path: str) -> int:
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    st = os.statvfs(p)
    return int(st.f_bavail) * int(st.f_frsize)


def preflight(path: str, need_bytes: int = 0, what: str = 'analysis', min_free: int = MIN_FREE_BYTES) -> int:
    """Raise InsufficientDiskSpace unless `path`'s volume has need_bytes + min_free free. Returns the free bytes."""
    fb = free_bytes(path)
    if fb < int(need_bytes) + int(min_free):
        raise InsufficientDiskSpace(
            f'Not enough free disk space for the {what}: {fb / 1e9:.1f} GB free on the volume of {path}, '
            f'need {(int(need_bytes) + int(min_free)) / 1e9:.1f} GB ({int(need_bytes) / 1e9:.1f} GB of temporary files '
            f'+ {int(min_free) / 1e9:.1f} GB reserve). Free some space and try again.')
    return fb


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale(root: Optional[str] = None, min_age_s: float = 5.0) -> list:
    """Remove job dirs under <root>/jobs whose owner process no longer exists (left behind by a hard kill).
    Returns the removed paths."""
    jobs = os.path.join(root or work_root(), 'jobs')
    removed = []
    try:
        names = os.listdir(jobs)
    except OSError:
        return removed
    now = time.time()
    for nm in names:
        p = os.path.join(jobs, nm)
        if not os.path.isdir(p):
            continue
        pid = None
        try:
            pid = int(json.load(open(os.path.join(p, 'owner.json')))['pid'])
        except Exception:
            try:
                pid = int(nm.split('-')[1])
            except Exception:
                pid = None
        try:
            age = now - os.path.getmtime(p)
        except OSError:
            continue
        if (pid is None or not _pid_alive(pid)) and age >= min_age_s:
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p)
    return removed


_LIVE: dict = {}
_LIVE_LOCK = threading.Lock()


def _atexit_cleanup():
    with _LIVE_LOCK:
        items = list(_LIVE.values())
    for jd in items:
        jd.cleanup()


atexit.register(_atexit_cleanup)


class JobDir:
    """A private temporary directory for one engine job; removed on exit (see module docstring).

        with JobDir('analyze', video=path) as jd:
            p = jd.path('scratch.npy')
    """

    def __init__(self, kind: str = 'job', root: Optional[str] = None, **owner):
        self.root = root or work_root()
        jobs = os.path.join(self.root, 'jobs')
        os.makedirs(jobs, exist_ok=True)
        try:
            sweep_stale(self.root)
        except Exception:
            pass
        name = f'{kind}-{os.getpid()}-{time.strftime("%Y%m%dT%H%M%S")}-{threading.get_ident() % 100000:05d}'
        self.dir = os.path.join(jobs, name)
        os.makedirs(self.dir, exist_ok=False)
        with open(os.path.join(self.dir, 'owner.json'), 'w') as fh:
            json.dump(dict(pid=os.getpid(), started=time.time(), kind=kind, **{k: str(v) for k, v in owner.items()}),
                      fh)
        self._closed = False
        with _LIVE_LOCK:
            _LIVE[id(self)] = self

    def path(self, *parts: str) -> str:
        return os.path.join(self.dir, *parts)

    def usage_bytes(self) -> int:
        tot = 0
        for dp, _, fns in os.walk(self.dir):
            for f in fns:
                try:
                    tot += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
        return tot

    def cleanup(self):
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self.dir, ignore_errors=True)
        with _LIVE_LOCK:
            _LIVE.pop(id(self), None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False


class sigterm_cancels:
    """While active, SIGTERM (and SIGHUP) set `event` instead of killing the interpreter outright — so the
    pipeline's cancel checks raise, and every try/finally (pool shutdown, decoder shutdown, JobDir removal) runs.
    Only installed in the main thread and only when nobody else handles the signal (the app bridge installs its
    own cooperative handler; that one is left alone). A second SIGTERM restores the default action and re-sends
    the signal (so a stuck job can still be terminated)."""

    SIGNALS = ('SIGTERM', 'SIGHUP')

    def __init__(self, event: threading.Event):
        self.event = event
        self.saved = {}

    def _handler(self, signum, _frame):
        if self.event.is_set():
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        self.event.set()

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            return self
        for nm in self.SIGNALS:
            sig = getattr(signal, nm, None)
            if sig is None:
                continue
            try:
                cur = signal.getsignal(sig)
            except (ValueError, OSError):
                continue
            if cur in (signal.SIG_DFL, None):
                try:
                    signal.signal(sig, self._handler)
                    self.saved[sig] = cur
                except (ValueError, OSError):
                    pass
        return self

    def __exit__(self, *exc):
        for sig, cur in self.saved.items():
            try:
                signal.signal(sig, cur if cur is not None else signal.SIG_DFL)
            except (ValueError, OSError):
                pass
        self.saved = {}
        return False


def combine_cancel(*fns: Optional[Callable[[], bool]]) -> Callable[[], bool]:
    fns = [f for f in fns if f is not None]

    def cancel() -> bool:
        for f in fns:
            try:
                if f():
                    return True
            except Exception:
                return True
        return False
    return cancel
