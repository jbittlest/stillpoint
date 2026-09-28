"""Honest quality report for analyze() (ENGINE v3).  Owner: ENGINE.

The closed loop grades itself with its own residual estimator (DIS + direct photometric rotation fit), which is
optimistic: it measures exactly what it just folded away (the app showed '< 0.05 px, -99 %'). This module
measures the result INDEPENDENTLY with the eval harness's estimator (eval.jitter_metrics.MotionEstimator:
persistent KLT tracks, robust RS-aware affine, three-frame delta for the high band; eval.jitter_metrics.
compute_metrics for the HF path) on:

    ORIGINAL   the source frames rectified to the output's pinhole geometry with NO rotation compensation
               (plan row matrices = identity, the output focal) -- the unstabilised camera, same pixels scale;
    STABILIZED the final plan's previews (per-frame focal = the real render's zoom),

both rendered at the analysis width (960) from the SAME decoded frames by the shared Metal kernel. Units are
eval's: 1920-wide-equivalent px of the output frame (1080p-eq for 16:9). Calm-cruise = frames where the
ORIGINAL's < 1 Hz camera speed is below 150 px/s (eval gate definition, measured on the rectified original).

Cost control: windows of `window_s` seconds spread evenly over the clip, covering ~20 % of short clips and
<= ~3000 frames on long ones; each window's two sequences are estimated in the measurement process pool.
Frames reach the workers through small temporary .npy files in the job dir (deleted as soon as the window is
done; at most ~n_workers windows on disk), not through pickles (bounded memory, no big IPC copies).
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import replace
from typing import Callable, Optional

import numpy as np

__all__ = ['choose_windows', 'identity_plan', 'quality_report', 'aggregate', 'METHOD']

METHOD = ('eval.jitter_metrics MotionEstimator (persistent KLT, RS-aware affine, 3-frame delta) + compute_metrics, '
          'independent of the closed loop; ORIGINAL = source rectified to the output geometry without stabilisation, '
          'STABILIZED = final plan (real zoom); both 960-wide previews of the same decoded frames; '
          '1920-eq px of the output; calm = original <1 Hz speed < 150 px/s; sampled windows')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
CALM_THR = 150.0


def _eval_mod():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from eval import jitter_metrics as jm
    return jm


def choose_windows(F: int, fs: float, frac: float = 0.2, window_s: float = 3.0, min_frames: int = 600,
                   max_frames: int = 3000) -> list:
    """Evenly spread windows [(a, b)) covering ~frac of F frames (clamped to [min_frames, max_frames])."""
    L = max(60, int(round(window_s * fs)))
    if F <= L + 2:
        return [(0, F)] if F >= 30 else []
    want = int(min(F, max(min_frames, min(frac * F, max_frames))))
    n = max(1, min(F // L, int(round(want / L))))
    out = []
    for i in range(n):
        c = (i + 0.5) * F / n
        a = int(max(0, min(F - L, round(c - L / 2))))
        out.append((a, a + L))
    # merge overlaps (short clips)
    out.sort()
    merged = []
    for a, b in out:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def identity_plan(plan, fx: float):
    """The unstabilised camera in the output geometry: r_c = r_v (no rotation, no RS correction), focal fx."""
    F = plan.n_frames
    I = np.broadcast_to(np.eye(3), (F, plan.n_rows, 3, 3)).copy()
    return replace(plan, row_mats=I, out_fx=np.full(F, float(fx)), meta=dict(plan.meta))


def _estimate_job(path: str, fps: float, start: float = 0.0) -> dict:
    """Worker: run the eval estimator over the frames stored in `path` (uint8 (n,h,w) .npy)."""
    jm = _eval_mod()
    import cv2
    cv2.setNumThreads(1)
    imgs = np.load(path, mmap_mode='r')
    n, H, W = imgs.shape
    est = jm.MotionEstimator(W, H)
    sig = {k: [] for k in jm.PAIR_KEYS}
    for kk in ('H', 'band_rx', 'band_ry', 'd_band_rx', 'd_band_ry', 'cor_rx', 'cor_ry', 'd_cor_rx', 'd_cor_ry'):
        sig[kk] = []
    fst = {'luma': [], 'sharp': [], 'black': []}
    prev = None
    nb = est.n_bands
    for i in range(n):
        img = np.ascontiguousarray(imgs[i])
        fst['luma'].append(float(img.mean()))
        cy0, cy1, cx0, cx1 = H // 8, H - H // 8, W // 8, W - W // 8
        fst['sharp'].append(float(cv2.Laplacian(img[cy0:cy1, cx0:cx1], cv2.CV_32F).var()))
        fst['black'].append(float((img < 4).mean()))
        if prev is not None:
            r = est.estimate(prev, img, prev)
            for k in jm.PAIR_KEYS:
                okk = r.get('d_ok') if k.startswith('d_') else r.get('ok')
                sig[k].append(r.get(k, float('nan')) if okk or k.startswith('n_') else float('nan'))
            sig['H'].append(r.get('H', [float('nan')] * 9) if r.get('ok') else [float('nan')] * 9)
            sig['band_rx'].append(r.get('band_rx', np.full(nb, np.nan)) if r.get('ok') else np.full(nb, np.nan))
            sig['band_ry'].append(r.get('band_ry', np.full(nb, np.nan)) if r.get('ok') else np.full(nb, np.nan))
            sig['d_band_rx'].append(r.get('d_band_rx', np.full(nb, np.nan)) if r.get('d_ok') else np.full(nb, np.nan))
            sig['d_band_ry'].append(r.get('d_band_ry', np.full(nb, np.nan)) if r.get('d_ok') else np.full(nb, np.nan))
            for kk in ('cor_rx', 'cor_ry'):
                sig[kk].append(r.get(kk, np.full((4, 3), np.nan)) if r.get('ok') else np.full((4, 3), np.nan))
            for kk in ('d_cor_rx', 'd_cor_ry'):
                sig[kk].append(r.get(kk, np.full((4, 3), np.nan)) if r.get('d_ok') else np.full((4, 3), np.nan))
        prev = img
    del imgs
    out = {k: np.asarray(v, dtype=float) for k, v in sig.items()}
    for k, v in fst.items():
        out['frame_' + k] = np.asarray(v, float)
    meta = dict(info=dict(fps=float(fps)), analysis_w=int(W), analysis_h=int(H), n_frames=int(n), start=float(start))
    return dict(sig=out, meta=meta)


def _seq_series(res: dict, trim_s: float = 0.5) -> dict:
    """Per-frame HF energy, 8+ Hz energy, LF speed and spike deviations of one estimated sequence."""
    jm = _eval_mod()
    sig, meta = res['sig'], res['meta']
    m, d = jm.compute_metrics(sig, meta, trim_s=trim_s)
    fs = meta['info']['fps']
    W, H = meta['analysis_w'], meta['analysis_h']
    R2 = (1920.0 ** 2 + (1920.0 * H / W) ** 2) / 12.0
    N = len(sig['tx']) + 1
    tr = int(round(trim_s * fs))
    sl = slice(tr, N - tr) if N > 2 * tr + 10 else slice(0, N)
    hp = d['hp']
    e_hf = hp['tx'] ** 2 + hp['ty'] ** 2 + (hp['rot'] ** 2 + hp['logs'] ** 2) * R2
    Fl = jm.Filters(fs, 2.0)
    b = {kk: Fl.band(1, p) for kk, p in d['paths'].items()}
    e8 = b['tx'] ** 2 + b['ty'] ** 2 + (b['rot'] ** 2 + b['logs'] ** 2) * R2
    sp = np.asarray(d['lp_speed'], float)
    sp = np.concatenate([[sp[0]], sp])[:N] if len(sp) else np.zeros(N)
    return dict(e_hf=e_hf[sl], e8=e8[sl], speed=sp[sl], jump_dev=np.asarray(d['jump_dev']), metrics=m,
                jello=m.get('jello', {}).get('jello_px'), corner=m.get('corner', {}).get('corner_wobble_px'))


def aggregate(pairs: list, fs: float) -> dict:
    """pairs: [(window (a,b), orig series, stab series)] -> whole-clip numbers."""
    jm = _eval_mod()
    from importlib import import_module
    ev = import_module('eval.events')
    eo = np.concatenate([p[1]['e_hf'] for p in pairs]) if pairs else np.zeros(0)
    es = np.concatenate([p[2]['e_hf'] for p in pairs]) if pairs else np.zeros(0)
    o8 = np.concatenate([p[1]['e8'] for p in pairs]) if pairs else np.zeros(0)
    s8 = np.concatenate([p[2]['e8'] for p in pairs]) if pairs else np.zeros(0)
    spd = np.concatenate([p[1]['speed'] for p in pairs]) if pairs else np.zeros(0)
    rms = lambda e: float(np.sqrt(np.nanmean(e))) if len(e) else None
    calm = spd < CALM_THR
    out = dict(original_hf_px=rms(eo), stabilized_hf_px=rms(es), original_b8_30_px=rms(o8), stabilized_b8_30_px=rms(s8),
               calm_frac=float(calm.mean()) if len(calm) else None,
               original_calm_hf_px=rms(eo[calm]) if calm.sum() >= 60 else None,
               stabilized_calm_hf_px=rms(es[calm]) if calm.sum() >= 60 else None,
               original_calm_b8_30_px=rms(o8[calm]) if calm.sum() >= 60 else None,
               stabilized_calm_b8_30_px=rms(s8[calm]) if calm.sum() >= 60 else None,
               n_frames_scored=int(len(es)))
    if out['original_hf_px'] and out['stabilized_hf_px'] is not None:
        out['hf_reduction'] = float(1.0 - out['stabilized_hf_px'] / out['original_hf_px'])
    jo = [p[1]['jello'] for p in pairs if p[1]['jello'] is not None and np.isfinite(p[1]['jello'])]
    js = [p[2]['jello'] for p in pairs if p[2]['jello'] is not None and np.isfinite(p[2]['jello'])]
    out['original_jello_px'] = float(np.sqrt(np.mean(np.square(jo)))) if jo else None
    out['stabilized_jello_px'] = float(np.sqrt(np.mean(np.square(js)))) if js else None
    # single-frame jumps the stabiliser created (present in the stabilized previews, not in the original)
    n05 = n1 = 0
    mx = 0.0
    ev_rows = []
    for (a, b), o, s in pairs:
        r = ev.stillpoint_only_jumps(s['jump_dev'], {'original': o['jump_dev']}, fs, a / fs)
        n05 += r['only_n_gt_0.5px']
        n1 += r['only_n_gt_1px']
        mx = max(mx, r['only_max_px'])
        ev_rows += [dict(t_s=e['t_s'], px=e['px']) for e in r['events'] if e['only']][:5]
    out.update(new_jumps_gt_0_5px=int(n05), new_jumps_gt_1px=int(n1), new_jumps_max_px=float(mx),
               new_jumps=sorted(ev_rows, key=lambda e: -e['px'])[:10])
    return out


def quality_report(final_plan, stream, pool, fs: float, fx0: float, job_dir: str, *,
                   frac: float = 0.2, window_s: float = 3.0, cancel: Optional[Callable[[], bool]] = None,
                   progress: Optional[Callable[[float, str], None]] = None, render_fn=None) -> dict:
    """Independent quality numbers of `final_plan` (see module docstring). render_fn(plan, stream, records,
    cancel, extra_plans) -> the pipeline's _render_stream (passed in to avoid an import cycle)."""
    t0 = time.perf_counter()
    F = final_plan.n_frames
    wins = choose_windows(F, fs, frac, window_s)
    if not wins:
        return dict(method=METHOD, error='clip too short for the quality estimator')
    orig = identity_plan(final_plan, fx0)
    recs = np.concatenate([np.arange(a, b) for a, b in wins])
    starts = {a: i for i, (a, b) in enumerate(wins)}
    n_total = len(recs)
    futs = []                      # (window index, kind, future, path)
    results: dict = {}
    # bounded temp disk: at most 8 window sequences (<= ~1.3 GB at 960x720) on disk at any time
    max_files = min(8, 2 * (getattr(pool, 'n_workers', 4) + 1)) if pool is not None else 2
    cur = None
    done_frames = 0

    def harvest(block: bool):
        from concurrent.futures import wait as _wait
        while futs and (block or futs[0][2].done()):
            wi, kind, fut, path = futs[0]
            t_tick = time.perf_counter()
            while not fut.done():
                if cancel is not None and cancel():
                    raise InterruptedError('quality cancelled')
                _wait([fut], timeout=0.2)
                if progress is not None and time.perf_counter() - t_tick > 1.0:     # keep the UI alive
                    t_tick = time.perf_counter()
                    progress(0.7 * done_frames / n_total + 0.3 * len(results) / (2 * len(wins)),
                             'quality: estimating')
            futs.pop(0)
            try:
                results[(wi, kind)] = fut.result()
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
            block = len(futs) >= max_files

    def open_window(i):
        a, b = wins[i]
        mms = []
        for kind in ('orig', 'stab'):
            path = os.path.join(job_dir, f'quality_{i:03d}_{kind}.npy')
            mms.append((path, np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                                                        shape=(b - a, stream.h, stream.w))))
        return mms

    def flush(i, mms):
        for kind, (path, mm) in zip(('orig', 'stab'), mms):
            mm.flush()
            del mm
            if pool is not None:
                futs.append((i, kind, pool.submit(_estimate_job, path, fs, wins[i][0] / fs), path))
            else:
                from concurrent.futures import Future
                f = Future()
                f.set_result(_estimate_job(path, fs, wins[i][0] / fs))
                futs.append((i, kind, f, path))
        mms.clear()
        harvest(len(futs) >= max_files)

    try:
        wi = -1
        mms = None
        j = 0
        for k, outs in render_fn(final_plan, stream, recs, cancel=cancel, extra_plans=(orig,)):
            if k in starts:
                if mms is not None:
                    flush(wi, mms)
                wi = starts[k]
                mms = open_window(wi)
                j = 0
            (s_img, _), (o_img, _) = outs
            mms[0][1][j] = o_img
            mms[1][1][j] = s_img
            j += 1
            done_frames += 1
            if progress is not None and done_frames % 60 == 0:
                progress(0.7 * done_frames / n_total, f'quality: {done_frames}/{n_total} frames')
        if mms is not None:
            flush(wi, mms)
        while futs:
            harvest(True)
            if progress is not None:
                progress(0.7 + 0.3 * len(results) / (2 * len(wins)), 'quality: estimating')
    finally:
        for _, _, f, path in futs:
            f.cancel()
            try:
                os.remove(path)
            except OSError:
                pass
    pairs = []
    rows = []
    for i, (a, b) in enumerate(wins):
        ro, rs = results.get((i, 'orig')), results.get((i, 'stab'))
        if ro is None or rs is None:
            continue
        so, ss = _seq_series(ro), _seq_series(rs)
        pairs.append(((a, b), so, ss))
        rows.append(dict(t0_s=round(a / fs, 3), t1_s=round(b / fs, 3),
                         original_hf_px=round(float(np.sqrt(np.mean(so['e_hf']))), 4),
                         stabilized_hf_px=round(float(np.sqrt(np.mean(ss['e_hf']))), 4)))
    q = aggregate(pairs, fs)
    q.update(method=METHOD, units='px @1080p-eq of the output (1920-wide equivalent)', windows=rows,
             window_s=window_s, coverage_frac=float(n_total / max(F, 1)), elapsed_s=time.perf_counter() - t0)
    return q
