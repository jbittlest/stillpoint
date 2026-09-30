"""Vision check of the output horizon of a Stillpoint analysis (horizon lock evaluation, gravity verification).

    PYTHONPATH=engine .venv/bin/python scripts/horizon_vision.py VIDEO ANALYSIS_DIR START_S DUR_S OUT.npz \
        [--step 4] [--scale 0.25] [--keep-frames]

Renders gray previews of the OUTPUT for every step-th frame of the window (render_ref.render_frames: the renderer's
mapping) and measures, per frame, the horizon tilt a viewer sees:
  rho_pred  predicted from telemetry gravity and the plan's virtual camera (ANALYSIS_DIR/analysis.npz virt_q):
            rho = atan2(g_x, g_y), g = V^T g_w (0 = level; the horizon line is drawn at image angle -rho)
  rho_h     the sky/ground edge near the predicted line (bright above, dark below, straight): robust line fit
  rho_v     vanishing geometry: long line segments (LSD) within +-10 deg of the predicted image direction of
            world-vertical at their midpoint (the line through the midpoint and the image of 'down', K g); the
            length-weighted median deviation (turbine towers, poles, buildings, trunks)
  rho_vis   rho_h where found, else rho_v.  rho_vis - rho_pred = the gravity error (verification of the
            camera's gravity reference); |rho_vis| = the residual horizon tilt of the output.
Single frames are noisy (sloped ridges, leaning trees, roads): use medians over many frames.  Holds one machine-wide
heavy-job slot (~/Library/Application Support/Stillpoint/scratch/locks/slot1..3) while rendering.
"""
import argparse
import json
import os
import sys
import time
from contextlib import contextmanager

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from stillpoint.plan_io import read_plan
from stillpoint.render_ref import preview_K, render_frames
from stillpoint.smooth import gravity_world, horizon_angles
from stillpoint.telemetry import load_telemetry
from stillpoint.workspace import cache_dir

LOCKS = os.path.expanduser('~/Library/Application Support/Stillpoint/scratch/locks')


@contextmanager
def heavy_slot(label: str):
    os.makedirs(LOCKS, exist_ok=True)
    p = None
    while p is None:
        for i in (1, 2, 3):
            try:
                os.mkdir(os.path.join(LOCKS, f'slot{i}'))
            except FileExistsError:
                continue
            p = os.path.join(LOCKS, f'slot{i}')
            with open(os.path.join(p, 'owner'), 'w') as fh:
                fh.write(f'{label} pid={os.getpid()} t={time.strftime("%H:%M:%S")}\n')
            break
        else:
            time.sleep(20)
    try:
        yield p
    finally:
        for f in os.listdir(p):
            os.remove(os.path.join(p, f))
        os.rmdir(p)


def detect_horizon(img, K, g, band_px, n_cols=48):
    """Sky/ground edge near the predicted horizon line -> (image angle of the edge minus the predicted line [rad],
    quality dict) or (nan, dict(reason=...))."""
    Ho, Wo = img.shape
    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    gx, gy, gz = g
    if gy < 0.3:
        return np.nan, dict(reason='steep_or_inverted')
    v0 = cy + fx * (-gz / gy)                         # predicted line gx*x + gy*y + gz = 0 at the centre column
    th = np.arctan2(-gx, gy)                          # its image angle (y down)
    d = np.array([np.cos(th), np.sin(th)])
    nrm = np.array([-np.sin(th), np.cos(th)])
    s = np.linspace(-0.36 * Wo, 0.36 * Wo, n_cols)
    t = np.arange(-band_px, band_px + 0.5, 0.5)
    X = cx + s[:, None] * d[0] + t[None, :] * nrm[0]
    Y = v0 + s[:, None] * d[1] + t[None, :] * nrm[1]
    colok = ((X >= 2) & (X <= Wo - 3) & (Y >= 2) & (Y <= Ho - 3)).mean(1) > 0.98
    if colok.mean() < 0.6:
        return np.nan, dict(reason='out_of_frame')
    im = gaussian_filter(img.astype(np.float32), 1.2)
    prof = map_coordinates(im, [Y.ravel(), X.ravel()], order=1, mode='nearest').reshape(X.shape)
    neg = -np.gradient(prof, 0.5, axis=1)             # bright above -> dark below: positive
    r_ = np.arange(len(s))
    j = np.argmax(neg, axis=1)
    strength = neg[r_, j]
    second = np.where(np.abs(t[None, :] - t[j][:, None]) > 3.0, neg, -np.inf).max(1)
    jj = np.clip(j, 1, len(t) - 2)
    a_, b_, c_ = neg[r_, jj - 1], neg[r_, jj], neg[r_, jj + 1]
    den = a_ - 2 * b_ + c_
    off = 0.5 * (a_ - c_) / np.where(np.abs(den) > 1e-6, den, 1.0) * (np.abs(den) > 1e-6)
    te = t[jj] + 0.5 * np.clip(off, -1, 1)
    good = colok & (strength > 2.0) & (strength > 1.1 * np.maximum(second, 0)) & (j > 2) & (j < len(t) - 3)
    if good.sum() < 0.45 * n_cols:
        return np.nan, dict(reason='weak')
    ss_, tt_ = s[good], te[good]
    w = np.ones(len(ss_))
    for _ in range(8):                                # Tukey IRLS line fit te = a + b s
        coef = np.linalg.lstsq(np.stack([np.ones_like(ss_), ss_], 1) * w[:, None], tt_ * w, rcond=None)[0]
        r = tt_ - (coef[0] + coef[1] * ss_)
        u = r / (4.685 * max(1.4826 * np.median(np.abs(r)), 1.0))
        w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
    inl = np.abs(r) < 5.0
    rms = float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else 99.0
    if inl.sum() < 0.4 * n_cols or rms > 3.5:
        return np.nan, dict(reason='not_straight')
    gi = np.flatnonzero(good)
    above = [np.median(prof[i, (t < te[i] - 3) & (t > te[i] - band_px)]) for i in gi
             if ((t < te[i] - 3) & (t > te[i] - band_px)).any()]
    below = [np.median(prof[i, (t > te[i] + 3) & (t < te[i] + band_px)]) for i in gi
             if ((t > te[i] + 3) & (t < te[i] + band_px)).any()]
    if not (len(above) and len(below) and np.median(above) - np.median(below) > 5.0):
        return np.nan, dict(reason='no_sky_contrast')
    return float(np.arctan(coef[1])), dict(reason='ok', rms=rms)


_LSD = cv2.createLineSegmentDetector()


def detect_verticals(img, K, g, min_len_frac=0.10, max_dev_deg=10.0, min_total_frac=0.35):
    """World-vertical segments -> (image angle of the observed verticals minus the predicted ones [rad], quality)
    or (nan, dict(reason=...))."""
    Ho, Wo = img.shape
    gx, gy, gz = g
    if abs(gz) > 0.85:
        return np.nan, dict(reason='steep')
    segs = _LSD.detect(img)[0]
    if segs is None or len(segs) == 0:
        return np.nan, dict(reason='no_segments')
    s = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    d = s[:, 2:] - s[:, :2]
    L = np.hypot(d[:, 0], d[:, 1])
    keep = L > min_len_frac * Ho
    if not keep.any():
        return np.nan, dict(reason='no_long')
    s, d, L = s[keep], d[keep], L[keep]
    m = 0.5 * (s[:, :2] + s[:, 2:])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mx, my = (m[:, 0] - cx) / fx, (m[:, 1] - cy) / fy
    # image direction of the world-vertical line through the midpoint's ray: d/dt proj(X + t g) ~ g_xy - m' g_z
    a_pred = np.arctan2((gy - my * gz) * fy, (gx - mx * gz) * fx)
    dev = (np.arctan2(d[:, 1], d[:, 0]) - a_pred + np.pi / 2) % np.pi - np.pi / 2      # undirected
    ok = np.abs(dev) < np.deg2rad(max_dev_deg)
    if L[ok].sum() < min_total_frac * Ho or ok.sum() < 2:
        return np.nan, dict(reason='few_verticals')
    dv, w = dev[ok], L[ok]
    o = np.argsort(dv)
    cw = np.cumsum(w[o])
    med = float(dv[o][np.searchsorted(cw, 0.5 * cw[-1])])
    return med, dict(reason='ok', n=int(ok.sum()),
                     spread_deg=float(np.degrees(np.sqrt(np.average((dv - med) ** 2, weights=w)))))


def _summary(res, key):
    v = np.isfinite(res[key])
    if not v.any():
        return dict(n_valid=0)
    e = np.degrees(res[key] - res['rho_pred'])[v]
    pc = lambda x: [round(float(np.percentile(np.abs(np.degrees(x)), p)), 2) for p in (50, 90, 95)]  # noqa: E731
    return dict(n_valid=int(v.sum()), abs_rho_vis_deg=pc(res[key][v]), abs_rho_pred_deg=pc(res['rho_pred'][v]),
                vis_minus_pred_median_deg=round(float(np.median(e)), 2),
                vis_minus_pred_mad_deg=round(float(1.4826 * np.median(np.abs(e - np.median(e)))), 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('adir')
    ap.add_argument('start', type=float)
    ap.add_argument('dur', type=float)
    ap.add_argument('out')
    ap.add_argument('--step', type=int, default=4)
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--band-deg', type=float, default=10.0)
    ap.add_argument('--keep-frames', action='store_true', help='keep the previews next to OUT (re-runs skip render)')
    a = ap.parse_args(argv)
    T0 = time.time()
    plan = read_plan(os.path.join(a.adir, 'plan.spplan'))
    V = np.load(os.path.join(a.adir, 'analysis.npz'))['virt_q']
    tel = load_telemetry(a.video, cache_dir=cache_dir())
    if tel.gravity_q is None:
        raise SystemExit('no gravity-referenced attitude in this clip')
    pts = plan.frame_pts
    sel = np.flatnonzero((pts >= a.start) & (pts < a.start + a.dur))[::a.step]
    src = np.searchsorted(tel.frame_pts, pts[sel] - 1e-4)              # plan record -> telemetry frame (by pts)
    rho_p, elev_p, g = horizon_angles(V[sel], gravity_world(tel, tel.frame_t[src]))
    n = len(sel)
    res = dict(t=pts[sel], rho_pred=rho_p, elev_pred=elev_p, rho_h=np.full(n, np.nan), rho_v=np.full(n, np.nan))
    cache = a.out.replace('.npz', '_frames.npy')
    if os.path.exists(cache):
        imgs = np.load(cache, mmap_mode='r')
    else:
        Wo, Ho = int(round(plan.out_w * a.scale)), int(round(plan.out_h * a.scale))
        imgs = np.zeros((n, Ho, Wo), np.uint8)
        pos = {int(k): i for i, k in enumerate(sel)}
        with heavy_slot('horizon_vision'):
            for k, img in render_frames(plan, a.video, sel, out_scale=a.scale):
                imgs[pos[int(k)]] = img
        if a.keep_frames:
            np.save(cache, imgs)
    reasons = dict(horizon={}, verticals={})
    for i, k in enumerate(sel):
        K = preview_K(plan, int(k), a.scale)
        im = np.asarray(imgs[i])
        ex, q = detect_horizon(im, K, g[i], K[0, 0] * np.deg2rad(a.band_deg))
        reasons['horizon'][q['reason']] = reasons['horizon'].get(q['reason'], 0) + 1
        if np.isfinite(ex):
            res['rho_h'][i] = rho_p[i] - ex
        dv, q = detect_verticals(im, K, g[i])
        reasons['verticals'][q['reason']] = reasons['verticals'].get(q['reason'], 0) + 1
        if np.isfinite(dv):
            res['rho_v'][i] = rho_p[i] - dv
    res['rho_vis'] = np.where(np.isfinite(res['rho_h']), res['rho_h'], res['rho_v'])
    np.savez(a.out, **res)
    out = dict(n=n, horizon=dict(_summary(res, 'rho_h'), reasons=reasons['horizon']),
               verticals=dict(_summary(res, 'rho_v'), reasons=reasons['verticals']),
               combined=_summary(res, 'rho_vis'),
               abs_rho_pred_all_deg=[round(float(np.percentile(np.abs(np.degrees(rho_p)), p)), 2)
                                     for p in (50, 90, 95)], wall_s=round(time.time() - T0, 1))
    print(json.dumps(out), flush=True)
    with open(a.out.replace('.npz', '.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    sys.exit(main())
