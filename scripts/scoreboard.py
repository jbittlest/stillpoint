"""Stillpoint scoreboard: analyze each clip ONCE with a given engine checkout, render the scoreboard windows with
sprender, score them with the MAIN tree's judge (eval/gate.py), delete the renders, write scoreboard.json + .md.

    # main tree, all windows (O3 vs Gyroflow, OA4 0012 + O4 Pro 0004 vs the original)
    .venv/bin/python scripts/scoreboard.py --engine . --label v4 --out work/gate/v4
    # an agent's worktree, only O3 + OA4, own sprender build, compared with the main v4 scoreboard automatically
    <repo>/.venv/bin/python <repo>/scripts/scoreboard.py \\
        --engine <worktree> --windows o3,oa4 --label myfix --out <dir> [--sprender <bin>] [--params '{"smoothness": 1.2}']
    # rebuild the tables from finished windows only (no heavy work)
    ... scripts/scoreboard.py --engine <path> --label X --out DIR --report-only

--windows: comma list of groups (all, o3, o3gate, o3heldout, oa4, o4) and/or single windows CLIP:S-E
(DJI_0025:15-40, OA4_0012:146-158, O4_0004:134-146).  Run it with nohup in the background and poll DIR/scoreboard.md
(rewritten after every window) and DIR/scoreboard.log.  Resumable: re-running the same command skips windows already
scored with the same engine content / params / codec / sprender / judge.

CODEC + CACHE (2026-09-30): renders are ProRes 422 HQ by default (--codec prores): bit-identical run to run, so a
re-render of the same plan scores exactly the same.  (--codec hevc10 = the VideoToolbox HEVC encode every baseline
before v6 used; it is intermittently non-deterministic: HF moves up to ~20 %, jello / corner more, on the SAME plan.)
ProRes and HEVC scores are NOT comparable with each other.  Scored windows are cached machine-wide in
.../scratch/scoreboard/wincache, keyed by (engine source content, clip, params, codec, sprender, shader, judge,
perturbation); any run (any --label / --out) with the same key reuses the result instead of re-rendering.  --force
re-renders anyway.

JUDGE NOISE (ProRes): re-rendering is exact, so the remaining noise is the judge's sensitivity to tiny path changes,
measured with --perturb-px 0.05,0.05 (every plan record's output centre shifted by a constant 0.05 px 1080p-eq in x
and y: no motion change at all) on the 19 windows of the v6 default (work/gate/v6/perturb).  Per-window changes are
flagged only beyond 2.5x that sensitivity; per-camera means carry 95 % bootstrap CIs over windows, and a mean change
is called real only when its CI excludes 0 AND it is larger than the perturbation's own mean change band.

Rules it enforces (machine shared by several agents):
  * every heavy job (one clip analysis, one window render+score) holds one of the 3 machine-wide slots
    ~/Library/Application Support/Stillpoint/scratch/locks/slot{1,2,3} (mkdir; retry every 20 s; released on exit)
  * the analysis runs with <= 3 measurement workers (--processes), eval with <= 2 jobs
  * renders go to ~/Library/Application Support/Stillpoint/scratch/scoreboard/<label>/renders and are deleted as soon
    as the window is scored (a ProRes 4K window is 3-6 GB: a render waits until >= 25 GB + its size are free);
    analyses are cached in .../scratch/scoreboard/analyses/<clip>-<engine content hash>-<key> (the key covers the
    engine source content (engine/stillpoint/*.py + shaders/warp.metal), the clip and params).  --reanalyze runs
    cached analyses again (clean timings; the old dir is kept as <dir>.prev-<time>, plan bytes compared)
  * the analysis records wall time, peak RSS of its process tree (engine mem watch, 1 s) and how many OTHER heavy
    slots were busy meanwhile (timings are only comparable when that is 0)
  * the judge is always this file's tree (eval/), never the engine checkout's

Per window: HF (>2 Hz), calm-cruise HF, 2-8 Hz, 8-30 Hz, roll, 8-30 Hz roll, jello, row wobble, corner wobble
(paired 1-s median vs the reference + self RMS), Stillpoint-only jumps > 0.5 / > 1 px, exact crop footprint (plan-exact
at the sampled frames + fitted), win-rate vs Gyroflow (O3) / vs the original, the official gate checks, and an axis
breakdown (tx / ty / roll / scale per band, and on calm-cruise frames).  All in 1080p-equivalent px.
"""
from __future__ import annotations

import argparse
import atexit
import glob
import hashlib
import json
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import time

import numpy as np

MAIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(MAIN, 'engine'))
sys.path.insert(0, MAIN)
from eval import boot  # noqa: E402
from eval import gate as G  # noqa: E402   (the judge: always the main tree's)
from eval import jitter_metrics as jm  # noqa: E402
from eval.compare import paired as paired_windows  # noqa: E402
from eval.events import stillpoint_only_jumps  # noqa: E402
from eval.footage import GYROFLOW_DIR, O3_DIR  # noqa: E402

HOME = os.path.expanduser('~')
SCR = os.path.join(HOME, 'Library', 'Application Support', 'Stillpoint', 'scratch')
LOCKS = os.path.join(SCR, 'locks')
SB = os.path.join(SCR, 'scoreboard')
AN_CACHE = os.path.join(SB, 'analyses')
WIN_CACHE = os.path.join(SB, 'wincache')                 # scored windows, keyed by engine content / params / codec ...
FILL_ART_MAX_FRAMES = 720                                # fill artifacts: first 12 s of each window (time bound)
MIN_FREE_GB = 25.0                                       # never let a render push the data volume below this
V6_DEFAULT_SB = os.path.join(MAIN, 'work', 'gate', 'v6', 'default', 'scoreboard.json')   # ProRes reference run
BASE_NOGF = os.path.join(SB, 'base')                     # original-video measurements of the OA4 / O4 windows
OLD_BASE_NOGF = os.path.join(SCR, 'gatev4', 'eval', 'base')   # gate v4 (2026-09-28) cache, reused when still valid
PY = os.path.join(MAIN, '.venv', 'bin', 'python')
N_SLOTS = 3
SLOT_RETRY_S = 20

OA4_VIDEO = os.path.join(HOME, 'Desktop', 'DJI_20260927091931_0012_D.MP4')
O4_VIDEO = os.path.join(HOME, 'Desktop', 'DJI_20260925151512_0004_D.MP4')

# ------------------------------------------------------------------------------------------------ registry
# Analysis params per clip.  O3: output focal from Gyroflow's whole-clip source footprint (fov_match, the engine's
# default for the gate).  OA4 / O4 Pro (no Gyroflow render): fixed output focal = the v3 app analysis's (crop held
# constant across engines: no crop gaming) and the app's smoothness 1.5 -- same as gate v4 of 2026-09-28.
CLIPS = {}
for _c in ('DJI_0025', 'DJI_0027', 'DJI_0028', 'DJI_0032', 'DJI_0034'):
    CLIPS[_c] = dict(cam='o3', video=os.path.join(O3_DIR, _c + '.MP4'),
                     gf=os.path.join(GYROFLOW_DIR, _c + '_stabilized.mp4'),
                     params=dict(fov_match=os.path.join(GYROFLOW_DIR, _c + '_stabilized.mp4')))
CLIPS['OA4_0012'] = dict(cam='oa4', video=OA4_VIDEO, gf=None, params=dict(smoothness=1.5, out_fx=1596.847144089084))
CLIPS['O4_0004'] = dict(cam='o4', video=O4_VIDEO, gf=None, params=dict(smoothness=1.5, out_fx=1611.0712918603779))

WINDOWS = []            # (clip, start, dur, set)
for _c, _s, _d in G.GATE_WINDOWS:
    WINDOWS.append((_c, _s, _d, 'gate'))
for _c, _s, _d in G.HELDOUT_WINDOWS:
    WINDOWS.append((_c, _s, _d, 'heldout'))
for _s in (146.0, 176.0, 196.0, 300.0, 330.0):
    WINDOWS.append(('OA4_0012', _s, 12.0, 'oa4'))
for _s, _set in ((45.0, 'o4'), (60.0, 'o4'), (177.0, 'o4'), (134.0, 'o4 calm')):
    WINDOWS.append(('O4_0004', _s, 12.0, _set))
CLIP_ORDER = ['OA4_0012', 'DJI_0034', 'DJI_0025', 'DJI_0028', 'O4_0004', 'DJI_0027', 'DJI_0032']
GROUPS = {'all': lambda w: True, 'o3': lambda w: CLIPS[w[0]]['cam'] == 'o3', 'o3gate': lambda w: w[3] == 'gate',
          'o3heldout': lambda w: w[3] == 'heldout', 'oa4': lambda w: CLIPS[w[0]]['cam'] == 'oa4',
          'o4': lambda w: CLIPS[w[0]]['cam'] == 'o4'}
CAM_NAME = {'o3': 'DJI O3 (vs Gyroflow)', 'oa4': 'Osmo Action 4 0012 (vs original)', 'o4': 'O4 Pro 0004 (vs original)'}
COMMON_PARAMS = dict(processes=3, save_iter_plans=False, verbose=True)


def wkey(clip, s, d):
    return f'{clip}:{G.wtag(s, d)}'


# ------------------------------------------------------------------------------------------------ small utils
_LOGFH = [None]


def log(*a):
    msg = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(msg, flush=True)
    if _LOGFH[0]:
        _LOGFH[0].write(msg + '\n')
        _LOGFH[0].flush()


def vt_count():
    return len(subprocess.run(['pgrep', '-x', 'VTDecoderXPCService'], capture_output=True, text=True).stdout.split())


def sha1_file(p, n=16):
    h = hashlib.sha1()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:n]


def redact_home(text: str) -> str:
    """Published result files must not carry the user's home path (the repo is public): /Users/<name> -> ~."""
    home = os.path.expanduser('~')
    return text.replace(home, '~') if home and home != '~' else text


def jdump(obj, path):
    tmp = path + '.part'
    with open(tmp, 'w') as fh:
        fh.write(redact_home(json.dumps(obj, indent=1, default=_json_default)))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return repr(o)


def fnum(x, nd=3):
    if x is None:
        return '-'
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return '-' if not math.isfinite(x) else f'{x:.{nd}f}'


def fin(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------------------------ heavy-job slots
_HELD = set()
_CHILD = [None]
_LAST_RELEASE = [0.0]


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _slot_owner(p):
    try:
        return open(os.path.join(p, 'owner')).read().strip()
    except OSError:
        return ''


def clean_stale_slots():
    """Remove slots written by THIS script whose orchestrator pid is gone (a crashed scoreboard run). Slots taken by
    anything else are never touched."""
    for i in range(1, N_SLOTS + 1):
        p = os.path.join(LOCKS, f'slot{i}')
        o = _slot_owner(p)
        if o.startswith('scoreboard.py ') and ' pid=' in o:
            try:
                pid = int(o.split(' pid=')[1].split()[0])
            except (ValueError, IndexError):
                continue
            if pid != os.getpid() and not _pid_alive(pid):
                shutil.rmtree(p, ignore_errors=True)
                log(f'removed stale scoreboard slot {p} ({o})')


class Slot:
    def __init__(self, label, job):
        self.label, self.job, self.path = label, job, None

    def __enter__(self):
        os.makedirs(LOCKS, exist_ok=True)
        # fairness: other jobs poll every SLOT_RETRY_S, so right after releasing a slot give them one polling period
        # to take it before this run grabs the next one (else a long scoreboard run starves every waiter)
        gap = time.time() - _LAST_RELEASE[0]
        if gap < SLOT_RETRY_S + 5:
            time.sleep(SLOT_RETRY_S + 5 - gap)
        t0, said = time.time(), False
        while True:
            for i in range(1, N_SLOTS + 1):
                p = os.path.join(LOCKS, f'slot{i}')
                try:
                    os.mkdir(p)
                except FileExistsError:
                    continue
                with open(os.path.join(p, 'owner'), 'w') as fh:
                    fh.write(f'scoreboard.py label={self.label} pid={os.getpid()} job={self.job} '
                             f'since={time.strftime("%Y-%m-%dT%H:%M:%S")}\n')
                self.path = p
                _HELD.add(p)
                if said:
                    log(f'got {os.path.basename(p)} after {time.time() - t0:.0f}s')
                return self
            if not said:
                owners = '; '.join(f'slot{i}: {_slot_owner(os.path.join(LOCKS, f"slot{i}"))[:80]}'
                                   for i in range(1, N_SLOTS + 1))
                log(f'waiting for a heavy-job slot ({self.job}) -- {owners}')
                said = True
            time.sleep(SLOT_RETRY_S)

    def __exit__(self, *exc):
        release(self.path)
        return False


def release(p):
    if p and p in _HELD:
        if f'pid={os.getpid()} ' in _slot_owner(p):
            shutil.rmtree(p, ignore_errors=True)
        _HELD.discard(p)
        _LAST_RELEASE[0] = time.time()


def _cleanup():
    c = _CHILD[0]
    if c is not None and c.poll() is None:
        try:
            os.killpg(c.pid, signal.SIGTERM)
            c.wait(20)
        except Exception:
            try:
                os.killpg(c.pid, signal.SIGKILL)
            except Exception:
                pass
    for p in list(_HELD):
        release(p)


def _on_signal(sig, frm):
    log(f'signal {sig}: stopping (child killed, slots released)')
    _cleanup()
    sys.exit(128 + sig)


atexit.register(_cleanup)
signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGHUP, _on_signal)


# ------------------------------------------------------------------------------------------------ engine state
def git(root, *args):
    p = subprocess.run(['git', '-C', root, *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ''


def engine_state(root):
    root = os.path.abspath(root)
    files = sorted(glob.glob(os.path.join(root, 'engine', 'stillpoint', '**', '*.py'), recursive=True))
    shader = os.path.join(root, 'shaders', 'warp.metal')
    if not files or not os.path.exists(shader):
        raise SystemExit(f'{root}: no engine/stillpoint/*.py or shaders/warp.metal')
    h = hashlib.sha1()
    for f in files + [shader]:
        h.update(os.path.relpath(f, root).encode())
        with open(f, 'rb') as fh:
            h.update(fh.read())
    head = git(root, 'rev-parse', 'HEAD')
    dirty = git(root, 'status', '--porcelain', '--', 'engine', 'shaders')
    return dict(root=root, head=head or 'nogit', head_short=(head or 'nogit')[:7], branch=git(root, 'rev-parse',
                '--abbrev-ref', 'HEAD'), dirty=bool(dirty), dirty_files=dirty.splitlines()[:20],
                content=h.hexdigest()[:12], shader=shader, shader_sha=sha1_file(shader, 12))


# ------------------------------------------------------------------------------------------------ analysis
CHILD_CODE = r'''
import json, os, sys, time
ENGINE = %(engine)r
import stillpoint
assert os.path.realpath(stillpoint.__file__).startswith(os.path.realpath(os.path.join(ENGINE, 'engine'))), stillpoint.__file__
from stillpoint.pipeline import AnalyzeParams, analyze
prm = AnalyzeParams(**json.loads(%(params)r))
t0 = time.time()
_last = {'stage': None, 'f': -1.0, 't': 0.0}
def _prog(stage, f, msg=''):
    now = time.time()
    if stage != _last['stage'] or f - _last['f'] >= 0.02 or now - _last['t'] >= 60:
        _last.update(stage=stage, f=f, t=now)
        print('SBPROG %%.0fs %%s %%.3f %%s' %% (now - t0, stage, f, str(msg)[:120]), flush=True)
rep = analyze(%(video)r, %(out)r, prm, progress=_prog)
print('SCOREBOARD_ANALYSIS_DONE %%.1f' %% (time.time() - t0), flush=True)
'''


def analysis_params(clip, overrides):
    p = dict(COMMON_PARAMS)
    p.update(CLIPS[clip]['params'])
    p.update(overrides or {})
    return p


def analysis_dir(clip, eng, params):
    key = hashlib.sha1(json.dumps(dict(content=eng['content'], clip=clip, video=CLIPS[clip]['video'],
                                       params={k: v for k, v in params.items() if k != 'verbose'}),
                                  sort_keys=True).encode()).hexdigest()[:10]
    return os.path.join(AN_CACHE, f'{clip}-{eng["content"][:8]}-{key}'), key


def group_cpu_s(pgid):
    """Total CPU seconds of the live processes in process group pgid (the analysis child and its workers/decoders)."""
    out = subprocess.run(['ps', '-Ao', 'pgid=,time='], capture_output=True, text=True).stdout
    tot = 0.0
    for ln in out.splitlines():
        p = ln.split()
        if len(p) != 2 or p[0] != str(pgid):
            continue
        sec = 0.0
        for part in p[1].split(':'):
            try:
                sec = sec * 60 + float(part)
            except ValueError:
                pass
        tot += sec
    return tot


def other_slots_busy():
    """Heavy-job slots held by anything other than this process right now."""
    n = 0
    for i in range(1, N_SLOTS + 1):
        p = os.path.join(LOCKS, f'slot{i}')
        if os.path.isdir(p) and f'pid={os.getpid()} ' not in _slot_owner(p):
            n += 1
    return n


def run_analysis(clip, eng, params, label, stall_s=1800, max_s=6 * 3600, reanalyze_since=None):
    """reanalyze_since: epoch s -- a cached analysis finished before it is run again (clean timings); the old dir is
    kept as <dir>.prev-<time> and the new plan is compared byte-for-byte with it."""
    adir, key = analysis_dir(clip, eng, params)
    meta_p = os.path.join(adir, 'scoreboard_meta.json')
    lockd = adir + '.inprogress'
    prev_sha = None
    while True:
        if os.path.exists(meta_p) and os.path.exists(os.path.join(adir, 'plan.spplan')):
            m = json.load(open(meta_p))
            fin_t = time.mktime(time.strptime(m['finished'], '%Y-%m-%dT%H:%M:%S')) if m.get('finished') else 0
            if m.get('ok') and reanalyze_since is not None and fin_t < reanalyze_since:
                prev = f'{adir}.prev-{time.strftime("%Y%m%dT%H%M%S")}'
                prev_sha = m.get('plan_sha') or sha1_file(os.path.join(adir, 'plan.spplan'))
                os.rename(adir, prev)
                log(f'{clip}: --reanalyze: cached analysis moved to {prev}')
                continue
            if m.get('ok'):
                log(f'{clip}: cached analysis {adir}')
                return adir, m
        try:
            os.makedirs(AN_CACHE, exist_ok=True)
            os.mkdir(lockd)
            with open(os.path.join(lockd, 'owner'), 'w') as fh:
                fh.write(f'pid={os.getpid()} label={label}\n')
            break
        except FileExistsError:
            o = _slot_owner(lockd)
            try:
                pid = int(o.split('pid=')[1].split()[0])
            except (ValueError, IndexError):
                pid = -1
            if pid > 0 and not _pid_alive(pid):
                shutil.rmtree(lockd, ignore_errors=True)
                continue
            log(f'{clip}: another scoreboard run is analysing the same engine/params ({o}); waiting')
            time.sleep(30)
    try:
        code = CHILD_CODE % dict(engine=eng['root'], params=json.dumps(params), video=CLIPS[clip]['video'], out=adir)
        env = dict(os.environ, PYTHONPATH=os.path.join(eng['root'], 'engine') + os.pathsep + eng['root'],
                   PYTHONUNBUFFERED='1')
        logp = os.path.join(adir, 'analysis.log')
        with Slot(label, f'analyze {clip}'):
            if os.path.exists(adir):
                shutil.rmtree(adir)               # an unfinished earlier attempt
            os.makedirs(adir)
            vt0 = vt_count()
            log(f'{clip}: analysing with {eng["root"]} @ {eng["head_short"]} (content {eng["content"]}) params '
                f'{json.dumps({k: v for k, v in params.items() if k not in ("verbose",)})} -> {adir} (VT {vt0})')
            t0 = time.time()
            busy = [other_slots_busy()]
            with open(logp, 'w') as lf:
                p = subprocess.Popen([PY, '-c', code], stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=eng['root'],
                                     start_new_session=True)
                _CHILD[0] = p
                # stalled = neither new log output (the child prints a progress line at least every 60 s while
                # the engine calls back) nor >= 5 CPU-s of the process group for stall_s
                last_sz, last_t, last_cpu, n = 0, time.time(), 0.0, 0
                while p.poll() is None:
                    time.sleep(5)
                    n += 1
                    sz = os.path.getsize(logp)
                    if sz != last_sz:
                        last_sz, last_t = sz, time.time()
                    if n % 6 == 0:
                        busy.append(other_slots_busy())
                        cpu = group_cpu_s(p.pid)
                        if cpu - last_cpu >= 5.0:
                            last_cpu, last_t = cpu, time.time()
                    if n % 60 == 0:                     # every 5 min: the child's latest progress line
                        try:
                            with open(logp, 'rb') as fh:
                                fh.seek(max(0, sz - 20000))
                                tl = [x for x in fh.read().decode(errors='replace').splitlines()
                                      if x.startswith('SBPROG') or x.startswith('[stillpoint]')]
                            log(f'{clip}: {tl[-1][:160] if tl else "(no progress line yet)"} (group CPU {cpu:.0f}s)')
                        except OSError:
                            pass
                    stalled = time.time() - last_t > stall_s
                    if stalled or time.time() - t0 > max_s:
                        log(f'{clip}: analysis {"stalled" if stalled else "over time"}; killing')
                        os.killpg(p.pid, signal.SIGTERM)
                        try:
                            p.wait(30)
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid, signal.SIGKILL)
                            p.wait()
                _CHILD[0] = None
            wall = time.time() - t0
            vt1 = vt_count()
        ok = p.returncode == 0 and os.path.exists(os.path.join(adir, 'plan.spplan')) and \
            os.path.exists(os.path.join(adir, 'report.json'))
        tail = open(logp, errors='replace').read()[-3000:]
        if not ok:
            log(f'{clip}: ANALYSIS FAILED rc={p.returncode} after {wall:.0f}s\n{tail}')
            jdump(dict(ok=False, rc=p.returncode, wall_s=wall, tail=tail), meta_p + '.failed')
            return None, None
        rep = json.load(open(os.path.join(adir, 'report.json')))
        m = dict(ok=True, clip=clip, key=key, engine=eng, params=params, wall_s=wall, vt=[vt0, vt1],
                 plan_sha=sha1_file(os.path.join(adir, 'plan.spplan')),
                 finished=time.strftime('%Y-%m-%dT%H:%M:%S'), report=report_summary(rep),
                 other_slots_busy=dict(max=int(max(busy)), mean=float(np.mean(busy)), samples=len(busy)))
        if prev_sha is not None:
            m['reanalyzed'] = dict(prev_plan_sha=prev_sha, plan_identical=prev_sha == m['plan_sha'])
        jdump(m, meta_p)
        log(f'{clip}: analysis done in {wall:.0f}s (VT {vt0}->{vt1}; other heavy slots busy max {max(busy)}, '
            f'mean {np.mean(busy):.2f}; peak RSS {m["report"].get("peak_rss_gb")} GB'
            + (f'; plan {"IDENTICAL to" if prev_sha == m["plan_sha"] else "DIFFERENT from"} the cached one'
               if prev_sha else '') + f'); quality '
            f'{json.dumps(m["report"].get("quality"), default=_json_default)[:400]}')
        return adir, m
    finally:
        shutil.rmtree(lockd, ignore_errors=True)


def report_summary(rep):
    q = rep.get('quality') or {}
    res = rep.get('resources') or {}
    rss = res.get('rss') or {}
    T = rep.get('timings') or {}
    cl = rep.get('closed_loop') or {}
    return dict(camera=rep.get('camera'), n_frames=rep.get('n_frames'), fps=rep.get('fps'), out=rep.get('out'),
                zoom=rep.get('zoom'), total_s=T.get('total_s'), x_realtime=T.get('x_realtime'),
                stage_s={k: T.get(k) for k in ('calib_s', 'crop_s', 'closed_loop_s', 'final_replan_s', 'fill_s',
                                               'mesh_s', 'quality_s') if T.get(k) is not None},
                pool_workers=res.get('pool_workers'),
                peak_rss_gb=rss.get('peak_total_gb') if isinstance(rss, dict) else None,
                peak_main_gb=rss.get('peak_main_gb') if isinstance(rss, dict) else None,
                peak_stage=rss.get('peak_stage') if isinstance(rss, dict) else None,
                crop={k: v for k, v in (rep.get('crop') or {}).items() if k != 'trials'},
                closed_loop=dict(stop=cl.get('stop_reason'), n_increments=cl.get('n_increments'),
                                 open_hf=cl.get('open_loop_window_hf_px'), final_hf=cl.get('composite_window_hf_px')),
                quality={k: q.get(k) for k in ('original_hf_px', 'stabilized_hf_px', 'original_calm_hf_px',
                                               'stabilized_calm_hf_px', 'original_b8_30_px', 'stabilized_b8_30_px',
                                               'original_jello_px', 'stabilized_jello_px', 'new_jumps_gt_0_5px',
                                               'new_jumps_gt_1px', 'new_jumps_max_px', 'crop_footprint_mean',
                                               'crop_footprint_min', 'error')})


# ------------------------------------------------------------------------------------------------ render
_PROBE = {}


def frame_pts(video):
    if video not in _PROBE:
        from stillpoint.video import probe
        pr = probe(video)
        _PROBE[video] = (np.asarray(pr['frame_pts'], float), float(pr['fps']))
    return _PROBE[video]


def free_gb(path=SCR):
    return shutil.disk_usage(path).free / 1e9


def est_render_gb(plan, n, codec):
    """Rough output size: ProRes 422 HQ ~ 1.1 bytes/pixel/frame (conservative), HEVC ~ 0.05."""
    try:
        with open(plan, 'rb') as fh:
            ow, oh = struct.unpack_from('<II', fh.read(32), 24)
    except (OSError, struct.error):
        ow, oh = 3840, 2880
    return ow * oh * n * (1.1 if codec == 'prores' else 0.05) / 1e9


def wait_for_disk(need_gb, what, max_wait_s=1800):
    """Block (polling every 30 s) until the data volume has MIN_FREE_GB + need_gb free; raise after max_wait_s."""
    t0, said = time.time(), False
    while free_gb() < MIN_FREE_GB + need_gb:
        if not said:
            log(f'{what}: waiting for disk: {free_gb():.1f} GB free, need {MIN_FREE_GB:.0f} + {need_gb:.1f} GB')
            said = True
        if time.time() - t0 > max_wait_s:
            raise RuntimeError(f'not enough disk for {what}: {free_gb():.1f} GB free')
        time.sleep(30)


def video_md5(path):
    """MD5 of the encoded video packets (stream copy, no decode): equal for bit-identical encodes, independent of the
    container's creation-time metadata."""
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:v:0', '-c', 'copy', '-f', 'md5', '-'],
                       capture_output=True, text=True, timeout=600)
    out = p.stdout.strip()
    return out.split('=', 1)[1] if p.returncode == 0 and out.startswith('MD5=') else None


def perturbed_plan(src, dst, dx, dy):
    """Copy of plan src whose every record's output centre (out_cx, out_cy) is shifted by (dx, dy) 1080p-eq px: a
    constant sub-pixel translation of the whole output, i.e. no motion change at all -- any change in the judge's
    numbers is its sensitivity to resampling / feature selection (the noise floor of a tiny plan change).  Byte patch:
    everything else (records, FILL / MESH sections) is kept."""
    b = bytearray(open(src, 'rb').read())
    if bytes(b[:8]) != b'SPPLAN01':
        raise ValueError(f'not a .spplan: {src}')
    hb, = struct.unpack_from('<I', b, 12)
    out_w, = struct.unpack_from('<I', b, 24)
    n, = struct.unpack_from('<I', b, 32)
    rb, = struct.unpack_from('<I', b, 44)
    sc = out_w / 1920.0
    rec = np.frombuffer(bytes(b[hb:hb + n * rb]), dtype=np.uint8).reshape(n, rb).copy()
    c = rec[:, 12:20].copy().view('<f4').reshape(n, 2).astype(np.float64)
    c[:, 0] += dx * sc
    c[:, 1] += dy * sc
    rec[:, 12:20] = c.astype('<f4').view(np.uint8).reshape(n, 8)
    b[hb:hb + n * rb] = rec.tobytes()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + '.part', 'wb') as fh:
        fh.write(bytes(b))
    os.replace(dst + '.part', dst)
    return dst


def render_window(sprender, shader, video, plan, s, d, out, timeout=1800, codec='hevc10'):
    pts, fps = frame_pts(video)
    f0 = int(np.searchsorted(pts, s - 1e-6, side='left'))
    n = int(math.ceil(d * fps)) + 2
    wait_for_disk(est_render_gb(plan, n, codec), f'render {os.path.basename(out)}')
    cmd = [sprender, video, plan, out, '--start-frame', str(f0), '--frames', str(n), '--codec', codec,
           '--kernel', 'lanczos3', '--zero-base', '--shader', shader, '--quiet', '--no-progress']
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    _CHILD[0] = p
    try:
        outp, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        raise RuntimeError(f'sprender timed out after {timeout}s')
    finally:
        _CHILD[0] = None
    if p.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f'sprender failed ({p.returncode}): {outp[-2000:]}')
    dt = time.time() - t0
    return dict(start_frame=f0, frames=n, seconds=dt, fps=n / dt, codec=codec, gb=os.path.getsize(out) / 1e9,
                free_gb_after=free_gb(), md5=video_md5(out))


# ------------------------------------------------------------------------------------------------ scoring
def evaluate_nogf(tag, video, s, d, label, render, wdir, jobs=2, plan=None):
    """Same measurements as eval.gate.evaluate_window, reference = the original (no Gyroflow render exists)."""
    os.makedirs(BASE_NOGF, exist_ok=True)
    bo = os.path.join(BASE_NOGF, f'{tag}_{G.wtag(s, d)}_orig')
    old = os.path.join(OLD_BASE_NOGF, f'{tag}_{G.wtag(s, d)}_orig')
    if not G.cache_valid(bo, video, s, d) and G.cache_valid(old, video, s, d):
        for ext in ('.json', '.npz'):
            shutil.copy2(old + ext, bo + ext)
    todo = [] if G.cache_valid(bo, video, s, d) else [(video, s, d, None, None, bo)]
    b = os.path.join(wdir, f'{tag}_{G.wtag(s, d)}_{label}')
    todo.append((render, 0.0, d, video, s, b))
    G.run_jobs(todo, min(jobs, len(todo)))
    ro, zo = G._load(bo)
    rs, zs = G._load(b)
    rows = {'original': G._row(ro, zo, zo)}
    row = G._row(rs, zs, zo)
    rows[label] = row
    o = rows['original']
    fps = float(zo['fps'])
    pw_o = paired_windows(G._cmp_load(ro), G._cmp_load(rs))
    jumps = stillpoint_only_jumps(np.asarray(zs['jump_dev']), {'original': np.asarray(zo['jump_dev'])}, fps, 0.0)
    cwp = G.paired_corner(zs, zo, fps)
    checks = {
        '8-30Hz_vs_original': {'value': row['b830'], 'limit': o['b830'], 'pass': row['b830'] <= o['b830']},
        'winrate_vs_original': {'value': pw_o.get('b_better_frac', float('nan')), 'limit': G.RULES['winrate'],
                                'pass': pw_o.get('b_better_frac', 0) >= G.RULES['winrate'],
                                'windows': pw_o.get('windows'), 'wilcoxon_p': pw_o.get('wilcoxon_p')},
        'jumps_gt_1px': {'value': jumps['only_n_gt_1px'], 'max_px': jumps['only_max_px'], 'limit': 0,
                         'pass': jumps['only_n_gt_1px'] == 0},
        'jello': {'value': row['jello'], 'limit': G.RULES['jello_rel'] * o['jello'],
                  'pass': row['jello'] <= G.RULES['jello_rel'] * o['jello']},
        'corner_wobble': {'value': cwp['med_a'], 'original': cwp['med_b'], 'rms_sp': cwp['rms_a'],
                          'rms_orig': cwp['rms_b'], 'windows': cwp.get('windows'), 'paired_frac': cwp['frac'],
                          'note': 'informational (no pass rule without a Gyroflow reference)'},
        'pairing': {'value': row['lag'], 'limit': 0, 'pass': row['lag'] == 0},
    }
    entry = {'checks': checks, 'pass': all(c['pass'] for c in checks.values() if 'pass' in c),
             'winrate_vs_original': pw_o, 'jumps': {k: v for k, v in jumps.items() if k != 'events'},
             'jump_events': [e for e in jumps['events'] if e['only'] or e['px'] > 1.0][:20]}
    if plan:
        entry['plan_footprint_exact'] = G._plan_fp(plan, rs)
    g = {'clip': tag, 'start': s, 'dur': d, 'window': G.wtag(s, d), 'rules': G.RULES, 'code': jm.code_version(),
         'reference': 'original', 'rows': rows, 'stillpoint': {label: entry}}
    base = os.path.join(wdir, f'gate_{tag}_{G.wtag(s, d)}')
    jdump(g, base + '.json')
    with open(base + '.md', 'w') as fh:
        fh.write(markdown_nogf(g) + '\n')
    return g


def markdown_nogf(g):
    f = G._f
    L = [f"### {g['clip']} {g['window']} s  (vs the original; eval {g['code']['eval_version']} / {g['code']['sha1']})",
         '', '| video | HF px | calm-cruise HF | 2-8 Hz | 8-30 Hz | roll deg | 8-30 roll deg | jello | row wobble | '
         'corner wobble | spikes >1px (self) | crop footprint | lag | 1-s median |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for nm, r in g['rows'].items():
        L.append(f"| {nm} | {f(r['hf'])} | {f(r['calm'])} | {f(r['b28'])} | {f(r['b830'])} | {f(r['roll_deg'], 4)} | "
                 f"{f(r['roll830_deg'], 4)} | {f(r['jello'])} | {f(r['row_wobble'])} | {f(r['corner_wobble'])} | "
                 f"{f(r['spikes_gt1'])} | {f(r['footprint'], 4)} | {f(r['lag'])} | {f(r['med1s'])} |")
    L += ['', '| stillpoint | 8-30 Hz <= orig | win-rate vs orig | new jumps >1px (>0.5) | jello <= 1.05 orig | '
          'corner wobble (paired 1-s median) vs orig | lag 0 | GATE |', '|---|---|---|---|---|---|---|---|']
    P = lambda c: 'PASS' if c['pass'] else 'FAIL'
    for nm, e in g['stillpoint'].items():
        c = e['checks']
        L.append(f"| {nm} | {P(c['8-30Hz_vs_original'])} {f(c['8-30Hz_vs_original']['value'])} vs "
                 f"{f(c['8-30Hz_vs_original']['limit'])} | {P(c['winrate_vs_original'])} "
                 f"{f(100 * c['winrate_vs_original']['value'], 0)}% of {c['winrate_vs_original']['windows']} | "
                 f"{P(c['jumps_gt_1px'])} {c['jumps_gt_1px']['value']} ({e['jumps']['only_n_gt_0.5px']}), max "
                 f"{f(c['jumps_gt_1px']['max_px'], 2)} | {P(c['jello'])} {f(c['jello']['value'])} vs "
                 f"{f(c['jello']['limit'] / G.RULES['jello_rel'])} | {f(c['corner_wobble']['value'])} vs "
                 f"{f(c['corner_wobble']['original'])} | {P(c['pairing'])} {c['pairing']['value']} | "
                 f"**{'PASS' if e['pass'] else 'FAIL'}** |")
        if e['jump_events']:
            L += ['', f"{nm} jumps (frame after, window time, px, only):  " + '; '.join(
                f"{j['frame_after']} @{j['t_s']:.2f}s {j['px']:.2f}px{' ONLY' if j['only'] else ''}"
                for j in e['jump_events'][:10])]
    return '\n'.join(L)


def _R(z):
    W = float(z['analysis_w'])
    H = float(z['analysis_h']) if 'analysis_h' in z.files else W * 9 / 16
    return math.sqrt((1920.0 ** 2 + (1920.0 * H / W) ** 2) / 12)


def axes(r, z, z_orig, trim=30, thr=G.CALM_THR):
    """Per-axis breakdown (1080p-eq px): tx / ty / roll / scale of HF, 2-8 Hz, 8-30 Hz, and HF on calm frames."""
    m = r['metrics']
    R = _R(z)
    out = {'hf': {'tx': m.get('hf_tx_px'), 'ty': m.get('hf_ty_px'), 'roll': m.get('hf_rot_equiv_px'),
                  'scale': m.get('hf_scale_equiv_px')}}
    for k, v in m['bands'].items():
        nm = '2-8' if k.startswith('2-') else ('8-30' if k.startswith('8-') else k)
        out[nm] = {'tx': v.get('tx_px'), 'ty': v.get('ty_px'),
                   'roll': (v['rot_deg'] * math.pi / 180 * R) if v.get('rot_deg') is not None else None,
                   'scale': (v['scale_pct'] / 100 * R) if v.get('scale_pct') is not None else None}
    N = min(len(z_orig['hp_tx']), len(z['hp_tx']))
    sp = np.asarray(z_orig['lp_speed'], float)
    sp = np.concatenate([[sp[0]], sp])[:N]
    base = np.arange(trim, N - trim)
    idx = base[sp[base] < thr]
    if len(idx) >= 60:
        rms = lambda a, s=1.0: float(np.sqrt(np.mean((np.asarray(a, float)[idx] * s) ** 2)))
        out['calm'] = {'tx': rms(z['hp_tx']), 'ty': rms(z['hp_ty']), 'roll': rms(z['hp_rot'], R),
                       'scale': rms(z['hp_logs'], R)}
    return out


def lost_seconds(w, wdir, label):
    """1-s windows (the judge's win-rate windows, 0.5 s trim) where Stillpoint's HF RMS is above the reference's
    (Gyroflow on O3, the original otherwise): [(t0 source s, sp px, ref px, ref <1 Hz speed px/s)], worst first."""
    clip, s, d = w['clip'], w['start'], w['dur']
    spj = os.path.join(wdir, f'{clip}_{G.wtag(s, d)}_{label}.json')
    refj = (G.base_path(clip, s, d, 'gf') if w['cam'] == 'o3' else
            os.path.join(BASE_NOGF, f'{clip}_{G.wtag(s, d)}_orig')) + '.json'
    oj = (G.base_path(clip, s, d, 'orig') if w['cam'] == 'o3' else
          os.path.join(BASE_NOGF, f'{clip}_{G.wtag(s, d)}_orig')) + '.json'
    try:
        a = json.load(open(spj))['metrics']['window']
        b = json.load(open(refj))['metrics']['window']
        o = json.load(open(oj))['metrics']['window']
    except (OSError, KeyError, ValueError):
        return None
    ra, rb, sp = a.get('rms_px', []), b.get('rms_px', []), o.get('lf_speed_px_s', [])
    n = min(len(ra), len(rb))
    out = [(round(s + 0.5 + i, 1), round(ra[i], 3), round(rb[i], 3), round(sp[i], 0) if i < len(sp) else None)
           for i in range(n) if ra[i] > rb[i]]
    return sorted(out, key=lambda x: -(x[1] - x[2]))


def extract(g, name, ref_name):
    """Flat metrics of row `name` of a gate dict (eval.gate O3 format or the no-Gyroflow format)."""
    if name not in g.get('rows', {}) or name not in g.get('stillpoint', {}):
        return None
    row, e = g['rows'][name], g['stillpoint'][name]
    ch = e['checks']
    cw = ch.get('corner_wobble', {})
    win_ref = ch.get('winrate_vs_gyroflow') or ch.get('winrate_vs_original') or {}
    wo = e.get('winrate_vs_original') or (ch.get('winrate_vs_original') or {})
    crop = ch.get('crop_footprint') or {}
    return dict(
        hf=row['hf'], calm=row['calm'], calm_frac=row.get('calm_frac'), b28=row['b28'], b830=row['b830'],
        roll_deg=row['roll_deg'], roll830_deg=row.get('roll830_deg'), jello=row['jello'],
        row_wobble=row.get('row_wobble'), corner_self=row.get('corner_wobble'),
        corner=cw.get('value'), corner_ref=cw.get('gyroflow', cw.get('original')),
        jumps1=ch['jumps_gt_1px']['value'], jumps05=e.get('jumps', {}).get('only_n_gt_0.5px'),
        jump_max=ch['jumps_gt_1px'].get('max_px'),
        crop_fit=row.get('footprint'), crop_paired=crop.get('value'), crop_ref=crop.get('gyroflow'),
        crop_plan=(e.get('plan_footprint_exact') or {}).get('mean'),
        win=win_ref.get('value', win_ref.get('b_better_frac')), win_n=win_ref.get('windows'),
        win_p=win_ref.get('wilcoxon_p'), win_orig=wo.get('b_better_frac', wo.get('value')),
        lag=row.get('lag'), med1s=row.get('med1s'), noise=row.get('noise'),
        fails=[k for k, c in ch.items() if 'pass' in c and not c['pass']], passed=bool(e.get('pass')),
        calm_limit=(ch.get('calm_cruise') or {}).get('limit'))


def ref_metrics(row):
    return {k: row.get(k) for k in ('hf', 'calm', 'b28', 'b830', 'roll_deg', 'roll830_deg', 'jello', 'row_wobble',
                                    'corner_wobble', 'footprint', 'med1s', 'spikes_gt1')}


def plan_has_fill(plan):
    try:
        with open(plan, 'rb') as fh:
            return bool(struct.unpack_from('<I', fh.read(88), 84)[0] & 1)
    except (OSError, struct.error):
        return False


def fill_artifacts(mov, plan, adir, video, f0, n, timeout=1500):
    """eval/fill_artifacts.py on the window render (fill fraction, seam visibility, flicker of the filled pixels);
    a subprocess with a timeout, summary only (no per-frame rows)."""
    virt = os.path.join(adir, 'analysis.npz')
    if not os.path.exists(virt):
        return dict(error='no analysis.npz')
    outj = mov + '.fillart.json'
    env = dict(os.environ, PYTHONPATH=os.path.join(MAIN, 'engine') + os.pathsep + MAIN)
    t0 = time.time()
    p = subprocess.Popen([PY, '-m', 'eval.fill_artifacts', mov, plan, '--virt', virt, '--video', video,
                          '--src-start-frame', str(f0), '--frames', str(n), '--out', outj], cwd=MAIN, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    _CHILD[0] = p
    try:
        outp, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        return dict(error=f'timeout after {timeout}s')
    finally:
        _CHILD[0] = None
    try:
        r = json.load(open(outj))
        os.remove(outj)
    except (OSError, ValueError):
        return dict(error=f'rc={p.returncode}: {outp[-500:]}')
    r.pop('rows', None)
    r['wall_s'] = time.time() - t0
    return r


def score_window(clip, s, d, wset, label, adir, eng, sprender, wdir, rdir, jobs, keep_render=False, codec='hevc10',
                 plan=None, fill_art='auto'):
    """plan: the plan to render (default adir/plan.spplan; a perturbed copy for the noise control)."""
    c = CLIPS[clip]
    plan = plan or os.path.join(adir, 'plan.spplan')
    plan_sha = sha1_file(plan)
    mov = os.path.join(rdir, f'{clip}_{G.wtag(s, d)}_{label}.mov')
    fa = None
    with Slot(label, f'render+score {wkey(clip, s, d)}'):
        vt0 = vt_count()
        t0 = time.time()
        try:
            rinfo = render_window(sprender, eng['shader'], c['video'], plan, s, d, mov, codec=codec)
            log(f'{wkey(clip, s, d)}: rendered {rinfo["frames"]} frames in {rinfo["seconds"]:.0f}s '
                f'({rinfo["gb"]:.1f} GB, {rinfo["free_gb_after"]:.0f} GB free, md5 {str(rinfo["md5"])[:12]})')
            t1 = time.time()
            if c['cam'] == 'o3':
                g = G.evaluate_window(clip, s, d, {label: (mov, 0.0)}, wdir, jobs=jobs, plan=plan, reuse_sp=False)
                ref_name = 'gyroflow'
            else:
                g = evaluate_nogf(clip, c['video'], s, d, label, mov, wdir, jobs=jobs, plan=plan)
                ref_name = 'original'
            ev_s = time.time() - t1
            if fill_art == 'on' or (fill_art == 'auto' and plan_has_fill(plan)):
                fa = fill_artifacts(mov, plan, adir, c['video'], rinfo['start_frame'],
                                    min(rinfo['frames'], FILL_ART_MAX_FRAMES))
                log(f'{wkey(clip, s, d)}: fill artifacts ' + json.dumps(
                    {k: fa.get(k) for k in ('fill_frac_mean', 'fill_frac_p95', 'seam_ratio_pooled',
                                            'flicker_ratio_pooled', 'plan_uncovered_mean', 'wall_s', 'error')},
                    default=_json_default))
        finally:
            if not keep_render:
                for p in (mov,):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        vt1 = vt_count()
    tag = f'{clip}_{G.wtag(s, d)}'
    zo = np.load(G.base_path(clip, s, d, 'orig') + '.npz') if c['cam'] == 'o3' else \
        np.load(os.path.join(BASE_NOGF, f'{tag}_orig.npz'))
    rs, zs = G._load(os.path.join(wdir, f'{tag}_{label}'))
    ax = {'sp': axes(rs, zs, zo)}
    ro = json.load(open((G.base_path(clip, s, d, 'orig') if c['cam'] == 'o3'
                         else os.path.join(BASE_NOGF, f'{tag}_orig')) + '.json'))
    ax['orig'] = axes(ro, zo, zo)
    if c['cam'] == 'o3':
        rg, zg = G._load(G.base_path(clip, s, d, 'gf'))
        ax['gf'] = axes(rg, zg, zo)
    res = dict(window=wkey(clip, s, d), clip=clip, cam=c['cam'], set=wset, start=s, dur=d, reference=ref_name,
               sp=extract(g, label, ref_name), orig=ref_metrics(g['rows']['original']),
               gf=ref_metrics(g['rows']['gyroflow']) if 'gyroflow' in g['rows'] else None, axes=ax,
               jump_events=g['stillpoint'][label].get('jump_events', [])[:10],
               render=rinfo, eval_s=ev_s, wall_s=time.time() - t0, vt=[vt0, vt1],
               gate_json=os.path.join(wdir, f'gate_{tag}.json'))
    res['lost'] = lost_seconds(res, wdir, label)
    res['plan'], res['plan_sha'], res['codec'] = plan, plan_sha, codec
    if fa is not None:
        res['fill_artifacts'] = fa
    if res['sp'] is not None:
        res['sp']['plan_sha'] = plan_sha
        res['sp']['render_md5'] = rinfo.get('md5')
    return res


# ------------------------------------------------------------------------------------------------ comparisons
def load_compare(spec):
    """label=path[#row1,row2]: path = a scoreboard.json, or a dir of gate_<clip>_<window>.json files (first row found
    wins). -> (label, {window: metrics})."""
    lab, rest = spec.split('=', 1)
    path, rows = (rest.split('#', 1) + [''])[:2]
    rows = [r for r in rows.split(',') if r]
    path = os.path.abspath(os.path.expanduser(path))
    out = {}
    if os.path.isfile(path):
        sb = json.load(open(path))
        for k, w in sb.get('windows', {}).items():
            if w.get('sp'):
                out[k] = w['sp']
        return lab, out, dict(path=path, label=sb.get('label'), engine=sb.get('engine', {}).get('head_short'),
                              codecs=sorted({w.get('codec', 'hevc10') for w in sb.get('windows', {}).values()}))
    for clip, s, d, _ in WINDOWS:
        p = os.path.join(path, f'gate_{clip}_{G.wtag(s, d)}.json')
        if not os.path.exists(p):
            continue
        g = json.load(open(p))
        for r in (rows or list(g.get('stillpoint', {}))):
            m = extract(g, r, None)
            if m:
                out[wkey(clip, s, d)] = dict(m, row=r)
                break
    return lab, out, dict(path=path, rows=rows, codecs=['hevc10'])


# JUDGE NOISE per render codec.  Per metric: tol = (abs, rel) per-window flag tolerance (a change is flagged only
# when it exceeds max(abs, rel x value)), rel = 1-sigma relative per-window difference (the band of per-camera mean
# changes is 2 x rel / sqrt(n)); jumps: per-window count tolerance; win: 1-s win-rate tolerance.
# hevc10 (measured 2026-09-29): the same plan rendered twice differs only by HEVC encoder noise (VideoToolbox is
#   intermittently non-deterministic: 7 of 18 same-plan windows bit-identical, 11 not), yet the judge moves.  11 pairs
#   (gate v4 renders of 09-28 vs this script's of 09-29): RMS (max) HF 4.8% (10%), 2-8 Hz 5.5% (10%), 8-30 Hz 1.5%
#   (4.2%), roll 3.4% (7.2%), jello / row wobble ~17% (41%), paired corner wobble 7.4% (14%); jumps >1 px -3..+1,
#   >0.5 px -2..+4, win-rate -0.17..+0.13.  The v5 run later saw +20% HF on one same-plan pair (O4 60-72).
# prores (measured 2026-09-30, work/gate/v6/perturb): re-rendering the same plan is bit-identical (render md5 and
#   every judge number equal), so re-render noise is 0.  What remains is the judge's sensitivity to a tiny plan change:
#   the v6 default plans with every output centre shifted by a constant 0.05 px (x and y, 1080p-eq; no motion change)
#   vs unshifted, 19 windows.  See NOISE['prores']['source'] for the numbers behind the tolerances.
NOISE = {
    'hevc10': dict(
        source='11 same-plan HEVC re-render pairs (2026-09-28 vs 09-29); 2.5-sigma per-window tolerances',
        tol={'hf': (0.01, 0.12), 'calm': (0.01, 0.11), 'b28': (0.01, 0.14), 'b830': (0.005, 0.045),
             'roll_deg': (0.0005, 0.09), 'jello': (0.02, 0.42), 'corner': (0.02, 0.19)},
        rel={'hf': 0.048, 'calm': 0.045, 'b28': 0.055, 'b830': 0.015, 'roll_deg': 0.034, 'jello': 0.167,
             'corner': 0.074},
        jumps1=1, jumps05=4, win=0.2),
    'prores': dict(
        source='PLACEHOLDER (not measured yet): HEVC values',
        tol={'hf': (0.01, 0.12), 'calm': (0.01, 0.11), 'b28': (0.01, 0.14), 'b830': (0.005, 0.045),
             'roll_deg': (0.0005, 0.09), 'jello': (0.02, 0.42), 'corner': (0.02, 0.19)},
        rel={'hf': 0.048, 'calm': 0.045, 'b28': 0.055, 'b830': 0.015, 'roll_deg': 0.034, 'jello': 0.167,
             'corner': 0.074},
        jumps1=1, jumps05=4, win=0.2),
}
_NOISE_CODEC = ['prores']               # set in main() from --codec
CMP_METRICS = [('hf', 'HF'), ('calm', 'calm'), ('b28', '2-8'), ('b830', '8-30'), ('roll_deg', 'roll deg'),
               ('jello', 'jello'), ('corner', 'corner'), ('jumps1', 'jumps>1'), ('jumps05', 'jumps>0.5')]


def noise():
    return NOISE[_NOISE_CODEC[0]]


def crop_of(m):
    m = m or {}
    for k in ('crop_plan', 'crop_paired', 'crop_fit'):
        if fin(m.get(k)):
            return float(m[k])
    return None


def regressions(a, b, flip=False):
    """b vs a (a = comparison, b = this run): list of 'metric a->b' that got worse beyond the judge-noise tolerance.
    flip=True (called as regressions(this, other)) still prints 'other->this' for the improvements column."""
    if flip:
        return [_flip(x) for x in regressions(a, b)]
    out = []
    nz = noise()
    for k, nm in CMP_METRICS:
        x, y = (a or {}).get(k), (b or {}).get(k)
        if not (fin(x) and fin(y)):
            continue
        tol = nz[k] if k[:5] == 'jumps' else max(nz['tol'][k][0], nz['tol'][k][1] * abs(x))
        if y - x > tol:
            out.append(f'{nm} {fnum(x) if k[:5] != "jumps" else int(x)}->{fnum(y) if k[:5] != "jumps" else int(y)}')
    x, y = crop_of(a), crop_of(b)
    if fin(x) and fin(y) and x - y > 0.003:
        out.append(f'crop {fnum(x, 4)}->{fnum(y, 4)}')
    x, y = (a or {}).get('win'), (b or {}).get('win')
    if fin(x) and fin(y) and x - y > nz['win']:
        out.append(f'win {fnum(x, 2)}->{fnum(y, 2)}')
    return out


def mean_shift(cw, k):
    """Geo-mean relative change this/other over windows for metric k with its 95 % bootstrap CI over windows, and the
    judge-noise band of such a mean (2.5 x per-window sigma / sqrt(n); 2.0 flagged the null shift control).
    -> (ratio_ci dict, band) or None."""
    pairs = [(x['this'].get(k), x['other'].get(k)) for x in cw]
    c = boot.ratio_ci([p[0] for p in pairs], [p[1] for p in pairs])
    if not c['n'] or k not in noise()['rel']:
        return None
    return c, 2.5 * noise()['rel'][k] / math.sqrt(c['n'])


def shift_txt(c, band, nm):
    real = boot.excludes_zero(c) and abs(c['mean']) > band
    ci = f" [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]" if fin(c['lo']) else ''
    return (f"{nm} {100 * c['mean']:+.1f}%{ci}" + (f" **{'worse' if c['mean'] > 0 else 'better'}**" if real else '')
            + f" (noise +-{100 * band:.1f})")


def _flip(txt):
    nm, rng = txt.rsplit(' ', 1)
    x, y = rng.split('->')
    return f'{nm} {y}->{x}'


# ------------------------------------------------------------------------------------------------ summary + markdown
def summarize(wins):
    out = {}
    for cam in ('o3', 'oa4', 'o4'):
        ws = [w for w in wins if w['cam'] == cam and w.get('sp')]
        if not ws:
            continue
        mean = lambda k: float(np.nanmean([w['sp'][k] for w in ws if fin(w['sp'].get(k))])) \
            if any(fin(w['sp'].get(k)) for w in ws) else float('nan')
        s = dict(n=len(ws), passed=sum(w['sp']['passed'] for w in ws),
                 **{f'mean_{k}': mean(k) for k in ('hf', 'calm', 'b28', 'b830', 'roll_deg', 'roll830_deg', 'jello',
                                                     'row_wobble', 'corner', 'crop_plan', 'crop_fit')},
                 jumps1=int(sum(w['sp']['jumps1'] or 0 for w in ws)), jumps05=int(sum(w['sp']['jumps05'] or 0 for w in ws)),
                 check_fails={})
        for w in ws:
            for f in w['sp']['fails']:
                s['check_fails'][f] = s['check_fails'].get(f, 0) + 1
        # per-camera means with 95 % bootstrap CIs over windows (eval/boot.py; fixed seed)
        s['ci'] = {k: boot.mean_ci([w['sp'].get(k) for w in ws])
                   for k in ('hf', 'calm', 'b28', 'b830', 'roll_deg', 'jello', 'row_wobble', 'corner', 'jumps1',
                             'jumps05', 'crop_plan', 'win')}
        ref = 'gf' if cam == 'o3' else 'orig'
        rk = {'hf': 'hf', 'calm': 'calm', 'b28': 'b28', 'b830': 'b830', 'roll_deg': 'roll_deg', 'jello': 'jello'}
        s['ref'] = ref
        s['ref_mean'] = {k: float(np.nanmean([w[ref][v] for w in ws if fin(w[ref].get(v))]))
                         if any(fin(w[ref].get(v)) for w in ws) else float('nan') for k, v in rk.items()}
        s['geo_ratio_vs_ref'] = {}
        for k, v in rk.items():
            r = [w['sp'][k] / w[ref][v] for w in ws if fin(w['sp'].get(k)) and fin(w[ref].get(v)) and w[ref][v] > 0]
            s['geo_ratio_vs_ref'][k] = float(np.exp(np.mean(np.log(r)))) if r else float('nan')
        s['beats_ref_hf'] = sum(1 for w in ws if fin(w['sp']['hf']) and w['sp']['hf'] < w[ref]['hf'])
        tot = sum(w['sp']['win_n'] or 0 for w in ws)
        won = sum((w['sp']['win'] or 0) * (w['sp']['win_n'] or 0) for w in ws if fin(w['sp'].get('win')))
        s['win_pooled'] = won / tot if tot else float('nan')
        s['win_ge_90'] = sum(1 for w in ws if fin(w['sp'].get('win')) and w['sp']['win'] >= 0.9)
        axm = {}
        for vid in ('sp', ref):
            axm[vid] = {}
            for band in ('hf', '2-8', '8-30', 'calm'):
                axm[vid][band] = {}
                for a in ('tx', 'ty', 'roll', 'scale'):
                    vals = [w['axes'].get('gf' if vid == 'gf' else vid, {}).get(band, {}).get(a) for w in ws
                            if w.get('axes')]
                    vals = [v for v in vals if fin(v)]
                    axm[vid][band][a] = float(np.sqrt(np.mean(np.square(vals)))) if vals else float('nan')
        s['axes_rms_over_windows'] = axm
        out[cam] = s
    return out


def md_table(wins, cam, compare_names):
    ref = 'gf' if cam == 'o3' else 'orig'
    rn = 'GF' if cam == 'o3' else 'orig'
    L = [f'| window | HF ({rn}) | calm-cruise ({rn}) | 2-8 Hz ({rn}) | 8-30 Hz (orig) | roll deg ({rn}) | '
         f'8-30 roll deg ({rn}) | jello ({rn}) | row wobble ({rn}) | corner wobble p-med ({rn}) | '
         f'SP-only jumps >1 (>0.5), max | crop plan-exact / fitted{" (GF)" if cam == "o3" else ""} | '
         f'win vs {rn} | failed checks |', '|' + '---|' * 14]
    for w in wins:
        if w['cam'] != cam or not w.get('sp'):
            continue
        a, r, o = w['sp'], w[ref], w['orig']
        L.append(
            f"| {w['window']}{' (' + w['set'] + ')' if cam == 'o3' else (' (calm)' if 'calm' in w['set'] else '')} | "
            f"**{fnum(a['hf'])}** ({fnum(r['hf'])}) | **{fnum(a['calm'])}** ({fnum(r['calm'])}) | "
            f"{fnum(a['b28'])} ({fnum(r['b28'])}) | {fnum(a['b830'])} ({fnum(o['b830'])}) | "
            f"{fnum(a['roll_deg'], 4)} ({fnum(r['roll_deg'], 4)}) | {fnum(a['roll830_deg'], 4)} ({fnum(r['roll830_deg'], 4)}) | "
            f"{fnum(a['jello'])} ({fnum(r['jello'])}) | {fnum(a['row_wobble'])} ({fnum(r['row_wobble'])}) | "
            f"{fnum(a['corner'])} ({fnum(a['corner_ref'])}) | {a['jumps1']} ({a['jumps05']}), {fnum(a['jump_max'], 2)} | "
            f"{fnum(a['crop_plan'], 4)} / {fnum(a['crop_fit'], 4)}"
            f"{' (' + fnum(a['crop_ref'], 4) + ')' if cam == 'o3' else ''} | "
            f"{fnum(100 * a['win'], 0) if fin(a['win']) else '-'}% | {', '.join(a['fails']) or '**PASS**'} |")
    return L


def md_axes(wins, cam):
    ref = 'gf' if cam == 'o3' else 'orig'
    L = ['| window | video | HF tx / ty / roll / scale | 2-8 Hz tx / ty / roll / scale | 8-30 Hz tx / ty / roll / scale | '
         'calm HF tx / ty / roll / scale |', '|---|---|---|---|---|---|']
    q = lambda d: ' / '.join(fnum((d or {}).get(k)) for k in ('tx', 'ty', 'roll', 'scale'))
    for w in wins:
        if w['cam'] != cam or not w.get('axes'):
            continue
        for vid, nm in (('sp', 'SP'), (ref, 'GF' if ref == 'gf' else 'orig')):
            ax = w['axes'].get(vid) or {}
            L.append(f"| {w['window'] if vid == 'sp' else ''} | {nm} | {q(ax.get('hf'))} | {q(ax.get('2-8'))} | "
                     f"{q(ax.get('8-30'))} | {q(ax.get('calm'))} |")
    return L


def write_outputs(out_dir, label, eng, sprender, params_over, sel, results, analyses, compares, started):
    wins = [results[wkey(c, s, d)] for c, s, d, _ in sel if wkey(c, s, d) in results]
    for w in wins:
        if w.get('lost') is None:
            w['lost'] = lost_seconds(w, os.path.join(out_dir, 'windows'), label)
    summ = summarize(wins)
    cmp_out = {}
    for lab, data, info in compares:
        per = {}
        for w in wins:
            b = data.get(w['window'])
            if b:
                per[w['window']] = dict(this=w['sp'], other=b, regressions=regressions(b, w['sp']),
                                        improvements=regressions(w['sp'], b, flip=True))
        cmp_out[lab] = dict(info=info, windows=per)
    sb = dict(label=label, generated=time.strftime('%Y-%m-%dT%H:%M:%S'), started=started, engine=eng,
              sprender=dict(path=sprender, sha=sha1_file(sprender, 12)), eval=jm.code_version(),
              params_overrides=params_over, params={c: analysis_params(c, params_over) for c in {x[0] for x in sel}},
              n_windows_selected=len(sel), n_windows_done=len(wins),
              windows={w['window']: w for w in wins}, analyses=analyses, summary=summ, compare=cmp_out,
              codec=_NOISE_CODEC[0], noise=noise(),
              perturb_px=next((w.get('perturb_px') for w in wins if w.get('perturb_px')), None))
    jdump(sb, os.path.join(out_dir, 'scoreboard.json'))

    L = [f'# Stillpoint scoreboard: {label}', '',
         f"engine `{eng['root']}` @ {eng['head_short']} ({eng['branch']}{', DIRTY' if eng['dirty'] else ''}), "
         f"engine content {eng['content']}; sprender `{sprender}` ({sb['sprender']['sha']}); judge eval "
         f"{sb['eval']['eval_version']} / {sb['eval']['sha1']} (main tree); param overrides "
         f"`{json.dumps(params_over) if params_over else 'none'}`.  {len(wins)}/{len(sel)} windows done "
         f"({sb['generated']}).", '',
         '1080p-equivalent px. O3 references = Jimmy\'s Gyroflow renders (paired frames, exact footprint); OA4 / O4 Pro '
         '= the original (no Gyroflow render). calm-cruise = frames where the ORIGINAL\'s <1 Hz speed < 150 px/s. corner '
         'wobble = median over 1-s windows on frames x corners measured in both. crop = source-area footprint '
         '(plan-exact at the judge\'s sampled frames / fitted by the judge; GF = Gyroflow fitted at the same frames). '
         'jumps = Stillpoint-only one-frame jumps (absent from the reference(s)).', '',
         f"**Renders:** {', '.join(sorted({w.get('codec', 'hevc10') for w in wins})) or '-'}.  **Judge noise "
         f"({_NOISE_CODEC[0]}):** {noise()['source']}.  Per-window changes are flagged only beyond the tolerances "
         f"behind it; per-camera mean changes carry 95 % bootstrap CIs over windows and count as real only when "
         f"the CI excludes 0 and the change exceeds the noise band.  Only compare runs rendered with the same codec.",
         '']
    for cam in ('o3', 'oa4', 'o4'):
        if not any(w['cam'] == cam for w in wins):
            continue
        s = summ.get(cam, {})
        L += [f'## {CAM_NAME[cam]}', ''] + md_table(wins, cam, [c[0] for c in compares]) + ['']
        if s:
            rn = 'Gyroflow' if cam == 'o3' else 'original'
            ci = s['ci']
            cit = lambda k, nd=3, sc=1.0: (f"{fnum(sc * ci[k]['mean'], nd)} [{fnum(sc * ci[k]['lo'], nd)}, "
                                           f"{fnum(sc * ci[k]['hi'], nd)}]")
            L.append(f"Per-camera means [95 % bootstrap CI over {s['n']} windows]: HF {cit('hf')}, calm "
                     f"{cit('calm')}, 2-8 {cit('b28')}, 8-30 {cit('b830')}, roll deg {cit('roll_deg', 4)}, jello "
                     f"{cit('jello')}, corner {cit('corner')}, jumps >1 px per window {cit('jumps1', 2)} (>0.5: "
                     f"{cit('jumps05', 2)}), crop {cit('crop_plan', 4)}, 1-s win-rate vs {rn} {cit('win', 1, 100)} %.")
            L.append('')
            gr = s['geo_ratio_vs_ref']
            L.append(f"**{s['passed']}/{s['n']} windows pass every check.** Means: HF {fnum(s['mean_hf'])} "
                     f"({rn} {fnum(s['ref_mean']['hf'])}), calm {fnum(s['mean_calm'])} ({fnum(s['ref_mean']['calm'])}), "
                     f"2-8 {fnum(s['mean_b28'])} ({fnum(s['ref_mean']['b28'])}), 8-30 {fnum(s['mean_b830'])} "
                     f"({fnum(s['ref_mean']['b830'])}), roll {fnum(s['mean_roll_deg'], 4)} deg, jello "
                     f"{fnum(s['mean_jello'])} ({fnum(s['ref_mean']['jello'])}), corner {fnum(s['mean_corner'])}, "
                     f"crop {fnum(s['mean_crop_plan'], 4)}; SP-only jumps >1 px {s['jumps1']} (>0.5: {s['jumps05']}). "
                     f"Geo-mean ratio SP/{rn}: HF {fnum(gr['hf'], 2)}, calm {fnum(gr['calm'], 2)}, 2-8 "
                     f"{fnum(gr['b28'], 2)}, 8-30 {fnum(gr['b830'], 2)}, roll {fnum(gr['roll_deg'], 2)}, jello "
                     f"{fnum(gr['jello'], 2)}. SP HF lower than {rn} in {s['beats_ref_hf']}/{s['n']}; pooled 1-s "
                     f"win-rate vs {rn} {fnum(100 * s['win_pooled'], 0)}% ({s['win_ge_90']}/{s['n']} windows >= 90%). "
                     f"Check failures: {', '.join(f'{k} {v}' for k, v in sorted(s['check_fails'].items())) or 'none'}.")
            L.append('')
        rn = 'Gyroflow' if cam == 'o3' else 'the original'
        L += [f'<details><summary>{CAM_NAME[cam]}: seconds where Stillpoint loses to {rn} (1-s HF RMS, worst first; '
              f'source time s: SP vs {rn} px @ original <1 Hz speed px/s) and jump times</summary>', '']
        for w in wins:
            if w['cam'] != cam:
                continue
            lo = w.get('lost') or []
            jt = '; '.join(f"{w['start'] + j['t_s']:.2f}s {j['px']:.2f}px{'' if j.get('only') else ' (also in ref)'}"
                           for j in (w.get('jump_events') or [])[:6])
            L.append(f"- {w['window']}: lost {len(lo)} s" + (': ' + '; '.join(
                f'{t:g}s {a:.2f} vs {b:.2f} @{fnum(v, 0)}' for t, a, b, v in lo[:6]) if lo else '') +
                (f'. Jumps: {jt}' if jt else ''))
        L += ['', '</details>', '']
        L += [f'<details><summary>{CAM_NAME[cam]}: axis breakdown (tx / ty / roll / scale, px)</summary>', '']
        L += md_axes(wins, cam) + ['', '</details>', '']
    for lab, info in cmp_out.items():
        per = info['windows']
        if not per:
            continue
        my_codecs = sorted({w.get('codec', 'hevc10') for w in wins})
        their = info['info'].get('codecs') or ['hevc10']
        warn = (f' **WARNING: codec mismatch ({",".join(their)} vs {",".join(my_codecs)}): these scores are not '
                f'comparable.**' if set(their) != set(my_codecs) else '')
        L += [f'## vs {lab} (`{info["info"].get("path")}`'
              f'{" rows " + ",".join(info["info"]["rows"]) if info["info"].get("rows") else ""})', '',
              f'`{lab} -> {label}`; the last two columns list only changes beyond the per-window judge-noise tolerance '
              f'(see the note at the top); "same plan" = identical plan.spplan bytes.{warn}', '',
              f'| window | HF | calm | 2-8 | 8-30 | roll deg | jello | corner | jumps >1 (>0.5) | crop | win | '
              f'worse in {label} | better in {label} |', '|' + '---|' * 13]
        for wk, x in per.items():
            a, b = x['other'], x['this']
            arrow = lambda k, nd=3: f"{fnum(a.get(k), nd)} -> {fnum(b.get(k), nd)}"
            same = ' (same plan)' if a.get('plan_sha') and a.get('plan_sha') == b.get('plan_sha') else ''
            L.append(f"| {wk}{same} | {arrow('hf')} | {arrow('calm')} | {arrow('b28')} | {arrow('b830')} | "
                     f"{arrow('roll_deg', 4)} | {arrow('jello')} | {arrow('corner')} | {a.get('jumps1')} ({a.get('jumps05')}) "
                     f"-> {b.get('jumps1')} ({b.get('jumps05')}) | {fnum(crop_of(a), 4)} -> "
                     f"{fnum(crop_of(b), 4)} | {fnum(100 * a['win'], 0) if fin(a.get('win')) else '-'}% -> "
                     f"{fnum(100 * b['win'], 0) if fin(b.get('win')) else '-'}% | {'; '.join(x['regressions']) or '-'} | "
                     f"{'; '.join(x['improvements']) or '-'} |")
        common = list(per.values())
        shifts = []
        for cam in ('o3', 'oa4', 'o4'):
            cw = [x for k, x in per.items() if CLIPS[k.split(':')[0]]['cam'] == cam]
            if not cw:
                continue
            sh = []
            for k, nm in (('hf', 'HF'), ('calm', 'calm'), ('b28', '2-8'), ('b830', '8-30'), ('roll_deg', 'roll'),
                          ('jello', 'jello'), ('corner', 'corner')):
                r = mean_shift(cw, k)
                if r is not None:
                    sh.append(shift_txt(r[0], r[1], nm))
            dj = boot.diff_ci([x['this'].get('jumps1') for x in cw], [x['other'].get('jumps1') for x in cw])
            if dj['n']:
                sh.append(f"jumps>1 per window {dj['mean']:+.2f} [{fnum(dj['lo'], 2)}, {fnum(dj['hi'], 2)}]")
            shifts.append(f'{cam} ({len(cw)} windows): ' + ', '.join(sh))
        for cam in ('o3', 'oa4', 'o4'):
            cw = [x for k, x in per.items() if CLIPS[k.split(':')[0]]['cam'] == cam]
            if not cw:
                continue
            mm = lambda side, k: float(np.nanmean([x[side][k] for x in cw if fin(x[side].get(k))])) \
                if any(fin(x[side].get(k)) for x in cw) else float('nan')
            L.append(f"| **{cam} mean ({len(cw)} common)** | {fnum(mm('other', 'hf'))} -> {fnum(mm('this', 'hf'))} | "
                     f"{fnum(mm('other', 'calm'))} -> {fnum(mm('this', 'calm'))} | {fnum(mm('other', 'b28'))} -> "
                     f"{fnum(mm('this', 'b28'))} | {fnum(mm('other', 'b830'))} -> {fnum(mm('this', 'b830'))} | "
                     f"{fnum(mm('other', 'roll_deg'), 4)} -> {fnum(mm('this', 'roll_deg'), 4)} | "
                     f"{fnum(mm('other', 'jello'))} -> {fnum(mm('this', 'jello'))} | {fnum(mm('other', 'corner'))} -> "
                     f"{fnum(mm('this', 'corner'))} | {sum(x['other'].get('jumps1') or 0 for x in cw)} -> "
                     f"{sum(x['this'].get('jumps1') or 0 for x in cw)} | | | "
                     f"{sum(bool(x['regressions']) for x in cw)} windows | {sum(bool(x['improvements']) for x in cw)} windows |")
        del common
        L += ['', f'Geo-mean per-window change {lab} -> {label} (+ = worse; [95 % bootstrap CI over windows]; +- = '
                  f'2.5-sigma judge-noise band of the mean; bold = CI excludes 0 and beyond the band): '
                  + '; '.join(shifts), '']
    if analyses:
        L += ['## Whole-clip analyses', '', '| clip | frames | analysis s (x realtime) | other heavy slots busy '
              '(max / mean) | stages s | workers | peak RSS GB tree / main [stage] | '
              'loop | quality HF orig -> SP | calm | 8-30 | jello | new jumps >1 (>0.5), max | crop mean (min) | '
              'zoom frac / max | VT before -> after | cache dir |', '|' + '---|' * 17]
        seeded = []
        for clip, m in analyses.items():
            r = m.get('report') or {}
            q = r.get('quality') or {}
            z = r.get('zoom') or {}
            ob = m.get('other_slots_busy') or {}
            st = ', '.join(f"{k[:-2]} {v:.0f}" for k, v in (r.get('stage_s') or {}).items()
                           if k in ('closed_loop_s', 'fill_s', 'mesh_s', 'quality_s') and fin(v))
            L.append(f"| {clip} | {r.get('n_frames')} | {fnum(m.get('wall_s'), 0)} ({fnum(r.get('x_realtime'), 2)}) | "
                     f"{ob.get('max', '?')} / {fnum(ob.get('mean'), 2)} | {st or '-'} | "
                     f"{r.get('pool_workers')} | {fnum(r.get('peak_rss_gb'), 2)} / {fnum(r.get('peak_main_gb'), 2)} "
                     f"[{r.get('peak_stage')}] | "
                     f"{(r.get('closed_loop') or {}).get('n_increments')} folds | {fnum(q.get('original_hf_px'))} -> "
                     f"{fnum(q.get('stabilized_hf_px'))} | {fnum(q.get('original_calm_hf_px'))} -> "
                     f"{fnum(q.get('stabilized_calm_hf_px'))} | {fnum(q.get('original_b8_30_px'))} -> "
                     f"{fnum(q.get('stabilized_b8_30_px'))} | {fnum(q.get('original_jello_px'))} -> "
                     f"{fnum(q.get('stabilized_jello_px'))} | {q.get('new_jumps_gt_1px')} ({q.get('new_jumps_gt_0_5px')}), "
                     f"{fnum(q.get('new_jumps_max_px'), 2)} | {fnum(q.get('crop_footprint_mean'), 4)} "
                     f"({fnum(q.get('crop_footprint_min'), 4)}) | {fnum(z.get('frac_zoomed'))} / {fnum(z.get('max'))} | "
                     f"{(m.get('vt') or ['-', '-'])[0]} -> {(m.get('vt') or ['-', '-'])[1]} | "
                     f"`{os.path.basename(m.get('dir', ''))}`{' (seeded*)' if m.get('seeded') else ''} |")
            if m.get('seeded'):
                seeded.append(f"{clip} <- {m['seeded'].get('source')} ({m['seeded'].get('source_time')}, params "
                              f"{json.dumps(m['seeded'].get('source_params'))})")
        if seeded:
            first = next(m['seeded'] for m in analyses.values() if m.get('seeded'))
            L += ['', f"*seeded: plan + report copied from an earlier analysis instead of re-analysing -- "
                      f"{first.get('reason')}; verified by {first.get('verified_by')}. " + '; '.join(seeded)]
        L.append('')
    with open(os.path.join(out_dir, 'scoreboard.md.part'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    os.replace(os.path.join(out_dir, 'scoreboard.md.part'), os.path.join(out_dir, 'scoreboard.md'))


# ------------------------------------------------------------------------------------------------ main
def select_windows(spec):
    sel = []
    for tok in [t.strip() for t in spec.split(',') if t.strip()]:
        if tok in GROUPS:
            sel += [w for w in WINDOWS if GROUPS[tok](w)]
        elif ':' in tok:
            clip, rng = tok.split(':', 1)
            a, b = (float(x) for x in rng.split('-'))
            hit = [w for w in WINDOWS if w[0] == clip and abs(w[1] - a) < 1e-6 and abs(w[1] + w[2] - b) < 1e-6]
            if not hit:
                if clip not in CLIPS:
                    raise SystemExit(f'unknown clip {clip} (known: {", ".join(CLIPS)})')
                hit = [(clip, a, b - a, 'extra')]
            sel += hit
        else:
            raise SystemExit(f'unknown window/group {tok!r} (groups: {", ".join(GROUPS)})')
    seen, out = set(), []
    for w in sel:
        if wkey(w[0], w[1], w[2]) not in seen:
            seen.add(wkey(w[0], w[1], w[2]))
            out.append(w)
    order = {c: i for i, c in enumerate(CLIP_ORDER)}
    return sorted(out, key=lambda w: (order.get(w[0], 99), 0 if w[3] in ('gate', 'oa4', 'o4') else 1, w[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--engine', required=True, help='engine checkout root (has engine/stillpoint and shaders/)')
    ap.add_argument('--windows', default='all', help='groups (all,o3,o3gate,o3heldout,oa4,o4) and/or CLIP:S-E')
    ap.add_argument('--label', required=True, help='row name / run label (also written into the slot owner file)')
    ap.add_argument('--out', required=True, help='output dir: scoreboard.{json,md}, windows/ (gate json/md), log')
    ap.add_argument('--sprender', default=None, help='sprender binary (default: <engine>/app/renderer/.build/sprender '
                                                     'if built, else the main tree\'s)')
    ap.add_argument('--params', default='{}', help='JSON AnalyzeParams overrides for every clip (part of the cache key)')
    ap.add_argument('--plan', action='append', default=[], help='CLIP=plan.spplan: score this plan, skip analysis')
    ap.add_argument('--compare', action='append', default=None,
                    help='label=scoreboard.json | label=dir_of_gate_jsons[#row1,row2] (repeatable). Default: '
                         'm1=work/gate/m1#stillpoint, v3=work/gate/v4#v3 (the 2026-09-28 gate: v3 plans of OA4 / '
                         'O4 / DJI_0034) and main_v4=work/gate/v4/scoreboard.json (the main v4 scoreboard; for the '
                         'main run itself: v4_0928=work/gate/v4#v4loopoff,v4 = the same plans rendered on 09-28)')
    ap.add_argument('--codec', default='prores', choices=('hevc10', 'prores'),
                    help='sprender codec for the scored renders.  prores (default since 2026-09-30) = ProRes 422 HQ: '
                         'bit-identical frames on every run, so identical plans score identically; 3-6 GB per 4K '
                         'window (deleted after scoring), ~2x slower to render.  hevc10 = what every baseline before '
                         'v6 used; its VideoToolbox encode is intermittently non-deterministic (HF up to ~20 %% on the '
                         'same plan).  The two are NOT comparable: score the reference with the same codec')
    ap.add_argument('--perturb-px', default='', help='DX,DY: render the plans with every output centre shifted by '
                    'this constant (1080p-eq px) -- the judge-sensitivity control (no motion change)')
    ap.add_argument('--reanalyze', action='store_true', help='re-run analyses even when cached (clean timings; the '
                    'old analysis dir is kept as .prev-<time> and the plans are compared)')
    ap.add_argument('--fill-artifacts', default='auto', choices=('auto', 'on', 'off'),
                    help='run eval/fill_artifacts.py on each render (auto: plans with a FILL section)')
    ap.add_argument('--jobs', type=int, default=2, help='eval measurement processes (<= 3)')
    ap.add_argument('--force', action='store_true', help='re-render + re-score windows even when a result exists '
                                                         '(this run or the machine-wide window cache)')
    ap.add_argument('--keep-renders', action='store_true')
    ap.add_argument('--report-only', action='store_true', help='only rebuild scoreboard.{json,md} from finished windows')
    a = ap.parse_args()
    a.jobs = max(1, min(3, a.jobs))
    _NOISE_CODEC[0] = a.codec
    out_dir = os.path.abspath(a.out)
    wdir = os.path.join(out_dir, 'windows')
    rdir = os.path.join(SB, a.label, 'renders')
    for d in (out_dir, wdir, rdir, AN_CACHE, WIN_CACHE):
        os.makedirs(d, exist_ok=True)
    _LOGFH[0] = open(os.path.join(out_dir, 'scoreboard.log'), 'a')
    eng = engine_state(a.engine)
    sprender = os.path.abspath(a.sprender) if a.sprender else next(
        (p for p in (os.path.join(eng['root'], 'app', 'renderer', '.build', 'sprender'),
                     os.path.join(MAIN, 'app', 'renderer', '.build', 'sprender')) if os.path.exists(p)), None)
    if not sprender or not os.path.exists(sprender):
        raise SystemExit('no sprender binary: build app/renderer (build.sh) or pass --sprender')
    params_over = json.loads(a.params)
    plans = dict(x.split('=', 1) for x in a.plan)
    perturb = [float(x) for x in a.perturb_px.split(',')] if a.perturb_px else None
    if perturb is not None and len(perturb) != 2:
        raise SystemExit('--perturb-px DX,DY')
    sel = select_windows(a.windows)
    started = time.strftime('%Y-%m-%dT%H:%M:%S')
    t_start = time.time()
    own_sb = os.path.join(out_dir, 'scoreboard.json')
    if a.codec == 'prores':                                  # only ProRes references are comparable
        default_cmp = [f'v6_default={V6_DEFAULT_SB}'] if (os.path.exists(V6_DEFAULT_SB) and
                                                          os.path.abspath(V6_DEFAULT_SB) != own_sb) else []
    else:
        default_cmp = [f"m1={os.path.join(MAIN, 'work', 'gate', 'm1')}#stillpoint"]
        g0928 = os.path.join(MAIN, 'work', 'gate', 'v4')       # gate v4 of 2026-09-28: gate_<clip>_<window>.json
        if os.path.exists(os.path.join(g0928, 'gate_OA4_0012_146-158.json')):
            default_cmp.append(f'v3={g0928}#v3')
        main_sb = os.path.join(MAIN, 'work', 'gate', 'v4', 'scoreboard.json')
        if os.path.exists(main_sb) and os.path.abspath(main_sb) != own_sb:
            default_cmp.append(f'main_v4={main_sb}')
        elif os.path.exists(os.path.join(g0928, 'gate_OA4_0012_146-158.json')):
            default_cmp.append(f'v4_0928={g0928}#v4loopoff,v4')
    compares = [load_compare(c) for c in (a.compare if a.compare is not None else default_cmp)]
    ev_sha = jm.code_version()['sha1']
    # result key: engine content + clip + params (= the analysis key), codec, renderer, shader, judge, perturbation.
    # NOT the label: the machine-wide window cache shares results between runs.
    rkey_base = dict(sprender=sha1_file(sprender, 12), shader=eng['shader_sha'], eval=ev_sha, codec=a.codec)
    if perturb is not None:
        rkey_base['perturb_px'] = perturb
    log(f'== scoreboard {a.label}: engine {eng["root"]} @ {eng["head_short"]} content {eng["content"]} '
        f'dirty={eng["dirty"]}; sprender {sprender}; codec {a.codec}{"; perturb " + str(perturb) if perturb else ""}; '
        f'{len(sel)} windows; VT {vt_count()}; {free_gb():.0f} GB free; pid {os.getpid()}')
    clean_stale_slots()

    def akey_of(c):
        return plans.get(c) or analysis_dir(c, eng, analysis_params(c, params_over))[1]

    def base_plan_of(c):
        return plans.get(c) or os.path.join(analysis_dir(c, eng, analysis_params(c, params_over))[0], 'plan.spplan')

    def wcache_path(c, s, d):
        rk = dict(rkey_base, analysis=akey_of(c))
        h = hashlib.sha1(json.dumps(dict(rk, window=wkey(c, s, d)), sort_keys=True).encode()).hexdigest()[:16]
        return os.path.join(WIN_CACHE, f'{c}_{G.wtag(s, d)}', f'{h}.json'), rk

    def wcache_get(c, s, d):
        p, rk = wcache_path(c, s, d)
        if not os.path.exists(p):
            return None
        r = json.load(open(p))
        if r.get('rkey') != rk:
            return None
        bp = base_plan_of(c)
        if os.path.exists(bp) and r.get('base_plan_sha') and r['base_plan_sha'] != sha1_file(bp):
            log(f'{wkey(c, s, d)}: window cache entry is for a different plan (re-analysed, not deterministic?)')
            return None
        return dict(r, cached_from=p)

    results, analyses = {}, {}
    for c, s, d, _ in sel:                                  # resume: finished windows of this engine state
        rp = os.path.join(wdir, f'{c}_{G.wtag(s, d)}.result.json')
        want = dict(rkey_base, analysis=akey_of(c))
        if os.path.exists(rp) and not a.force:
            r = json.load(open(rp))
            if r.get('rkey') == want or a.report_only:
                results[wkey(c, s, d)] = r
    for c in {x[0] for x in sel}:
        mp = os.path.join(analysis_dir(c, eng, analysis_params(c, params_over))[0], 'scoreboard_meta.json')
        if os.path.exists(mp):
            analyses[c] = dict(json.load(open(mp)), dir=os.path.dirname(mp))
    write_outputs(out_dir, a.label, eng, sprender, params_over, sel, results, analyses, compares, started)
    if a.report_only:
        log('report-only: tables rebuilt')
        return

    failures = []
    for clip in [c for c in CLIP_ORDER if any(w[0] == c for w in sel)]:
        todo = [w for w in sel if w[0] == clip and (a.force or wkey(*w[:3]) not in results)]
        if not a.force:                                     # machine-wide window cache
            for w in list(todo):
                r = wcache_get(*w[:3])
                if r is not None:
                    jdump(r, os.path.join(wdir, f'{w[0]}_{G.wtag(w[1], w[2])}.result.json'))
                    for ext in ('.json', '.md'):
                        src = (r.get('gate_json') or '')[:-5] + ext
                        if src and os.path.exists(src) and os.path.dirname(src) != wdir:
                            shutil.copy2(src, os.path.join(wdir, os.path.basename(src)))
                    results[wkey(*w[:3])] = r
                    todo.remove(w)
                    log(f'{wkey(*w[:3])}: from the window cache ({r["cached_from"]})')
        if not todo and not (a.reanalyze and clip not in plans):
            continue
        prm = analysis_params(clip, params_over)
        if clip in plans:
            adir, akey = os.path.dirname(os.path.abspath(plans[clip])), plans[clip]
            if os.path.basename(plans[clip]) != 'plan.spplan':
                raise SystemExit('--plan must point at a plan.spplan inside its analysis dir')
        else:
            adir, meta = run_analysis(clip, eng, prm, a.label, reanalyze_since=t_start if a.reanalyze else None)
            if adir is None:
                failures.append(f'{clip}: analysis failed')
                continue
            akey = meta['key']
            analyses[clip] = dict(meta, dir=adir)
        base_plan = os.path.join(adir, 'plan.spplan')
        base_sha = sha1_file(base_plan)
        plan_r = base_plan
        if perturb is not None and todo:
            plan_r = perturbed_plan(base_plan, os.path.join(SB, a.label, 'plans', f'{clip}-{akey}.spplan'), *perturb)
            log(f'{clip}: perturbed plan {plan_r} (output centre +{perturb[0]:g}, +{perturb[1]:g} px 1080p-eq)')
        for c, s, d, wset in todo:
            k = wkey(c, s, d)
            try:
                r = score_window(c, s, d, wset, a.label, adir, eng, sprender, wdir, rdir, a.jobs, a.keep_renders,
                                 codec=a.codec, plan=plan_r, fill_art=a.fill_artifacts)
            except Exception as e:                           # noqa: BLE001 -- report, keep going
                log(f'{k}: SCORING FAILED: {e!r}')
                failures.append(f'{k}: {e!r}')
                continue
            r['rkey'] = dict(rkey_base, analysis=akey)
            r['base_plan_sha'] = base_sha
            if perturb is not None:
                r['perturb_px'] = perturb
            jdump(r, os.path.join(wdir, f'{c}_{G.wtag(s, d)}.result.json'))
            cp, _rk = wcache_path(c, s, d)
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            jdump(r, cp)
            results[k] = r
            sp = r['sp']
            log(f"{k}: HF {fnum(sp['hf'])} calm {fnum(sp['calm'])} 2-8 {fnum(sp['b28'])} 8-30 {fnum(sp['b830'])} "
                f"jello {fnum(sp['jello'])} corner {fnum(sp['corner'])} jumps {sp['jumps1']} ({sp['jumps05']}) crop "
                f"{fnum(sp['crop_plan'], 4)} win {fnum(100 * sp['win'], 0) if fin(sp['win']) else '-'}% fails "
                f"{','.join(sp['fails']) or 'none'} (render {r['render']['seconds']:.0f}s eval {r['eval_s']:.0f}s, "
                f"VT {r['vt'][0]}->{r['vt'][1]})")
            write_outputs(out_dir, a.label, eng, sprender, params_over, sel, results, analyses, compares, started)
        if plan_r != base_plan:
            try:
                os.remove(plan_r)
            except OSError:
                pass
        log(f'{clip}: done -> {os.path.join(out_dir, "scoreboard.md")}')
    write_outputs(out_dir, a.label, eng, sprender, params_over, sel, results, analyses, compares, started)
    log(f'== scoreboard {a.label} finished: {len(results)}/{len(sel)} windows; VT {vt_count()}; {free_gb():.0f} GB free'
        + (f'; FAILURES: {failures}' if failures else ''))


if __name__ == '__main__':
    main()
