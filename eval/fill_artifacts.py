"""
Artifacts of full-frame border fill (engine/stillpoint/fill.py) measured on a RENDERED window.

    PYTHONPATH=engine .venv/bin/python -m eval.fill_artifacts RENDER PLAN --virt analysis.npz \
        (--start-frame N | --video ORIGINAL --src-start-frame F) [--frames M] [--width 960] [--out result.json]
        [--snap-dir DIR]

RENDER is a window render of PLAN (sprender --zero-base): its frame i is plan record N + i (--start-frame N), or
the plan record of source frame F + i of ORIGINAL, matched by PTS (--video/--src-start-frame, what the scoreboard
renders). The geometry of every frame is known exactly from the plan (which output pixels lie outside the current
source frame, the virtual rotation between consecutive output frames), so no matching is needed:

* fill fraction      share of the output outside the current source frame (synthesised pixels), and the plan's
                     share that no neighbour covered (soft edge extension only)
* seam visibility    mean gradient magnitude ON the seam (the current frame's border inside the output) divided by
                     the mean gradient in parallel bands 3-8 output px (at --width) on both sides of it; 1.0 = no
                     visible seam. Reported per frame (median / p95) and pooled.
* temporal flicker   consecutive output frames aligned with the exact virtual rotation (homography K R K^-1):
                     mean |I_k - I_{k-1}(warped)| over synthesised pixels vs over an inner band of real pixels
                     4-16 px (at --width) inside the seam. Parallax, noise and moving objects are in both;
                     the ratio is the extra temporal instability of the fill (1.0 = as stable as real pixels).
Units: 8-bit gray levels of the render decoded at --width (default 960 px).
--snap-dir: JPEGs of the hardest frames (largest fill fraction, seam ratio, flicker ratio) with the current
frame's source border drawn (for eyeballing).
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _engine():
    eng = os.path.join(ROOT, 'engine')
    if eng not in sys.path:
        sys.path.insert(0, eng)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


def _d0_map(plan, k, out_scale):
    """Main-source inside distance [full-res source px] on the scaled output grid (Metal coord map)."""
    from stillpoint.render_ref import metal_coord_map
    S, _ok = metal_coord_map(plan, k, out_scale)
    d = np.minimum(np.minimum(S[..., 0] + 0.5, plan.src_w - 0.5 - S[..., 0]),
                   np.minimum(S[..., 1] + 0.5, plan.src_h - 0.5 - S[..., 1]))
    return d


def _records(plan, n, start_frame=None, video=None, src_start_frame=None):
    """Plan record of each render frame i < n."""
    if start_frame is not None:
        return np.arange(start_frame, start_frame + n)
    from stillpoint.video import probe
    pts = np.asarray(probe(video)['frame_pts'], float)
    sp = pts[src_start_frame:src_start_frame + n]
    r = np.clip(np.searchsorted(plan.frame_pts, sp), 1, plan.n_frames - 1)
    r = np.where(np.abs(plan.frame_pts[r - 1] - sp) < np.abs(plan.frame_pts[r] - sp), r - 1, r)
    return r


def measure(render: str, plan_path: str, virt_npz: str, start_frame=None, n_frames=None, width: int = 960,
            video=None, src_start_frame=None, snap_dir: str = '', n_snap: int = 3) -> dict:
    _engine()
    from eval.jitter_metrics import FrameReader
    from stillpoint.geom import quat_to_mat
    from stillpoint.plan_io import read_fill, read_plan
    from stillpoint.render_ref import output_grid, preview_K
    from stillpoint.video import probe
    plan = read_plan(plan_path)
    fill = read_fill(plan_path)
    vq = np.load(virt_npz)['virt_q']
    Rv = quat_to_mat(vq)
    pr = probe(render)
    n_r = len(pr['frame_pts'])
    N = min(n_r, n_frames or n_r)
    recs = _records(plan, N, start_frame, video, src_start_frame)
    out_scale = width / plan.out_w
    Wo, Ho, _, _ = output_grid(plan.out_w, plan.out_h, out_scale)
    t0 = time.time()
    rows = []
    prev = None
    heaps = {'fill': [], 'seam': [], 'flicker': []}

    def keep(kind, score, i, img, d0):
        h = heaps[kind]
        item = (float(score), int(i), img, d0)
        if len(h) < n_snap:
            heapq.heappush(h, item)
        elif score > h[0][0]:
            heapq.heapreplace(h, item)

    rd = FrameReader(render, 0, 0, Wo, Ho, hwaccel=True)
    try:
        for i, img in enumerate(rd):
            if i >= N:
                break
            k = int(recs[i])
            I = img.astype(np.float32)
            d0 = _d0_map(plan, k, out_scale)
            filled = d0 < 0
            row = dict(i=i, k=k, fill_frac=float(filled.mean()),
                       plan_fill_frac=float(fill.frac_fill[k]) if fill is not None else 0.0,
                       plan_uncovered=float(fill.frac_uncovered[k]) if fill is not None else 0.0,
                       n_src=int(fill.n_src[k]) if fill is not None else 0)
            if filled.any() and (~filled).any():
                inside = (~filled).astype(np.uint8)
                din = cv2.distanceTransform(inside, cv2.DIST_L2, 3)            # preview px inside the seam
                dout = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 3)
                seam = (din > 0) & (din <= 1.0)
                band = ((din >= 3) & (din <= 8)) | ((dout >= 3) & (dout <= 8))
                gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)
                g = np.hypot(gx, gy)
                if seam.sum() > 50 and band.sum() > 50:
                    row.update(seam_g=float(g[seam].mean()), band_g=float(g[band].mean()))
                # temporal flicker vs the previous output frame (exact virtual rotation)
                if prev is not None and prev[0] == i - 1:
                    K1 = preview_K(plan, k, out_scale)
                    K0 = preview_K(plan, prev[1], out_scale)
                    Hm = K0 @ Rv[prev[1]].T @ Rv[k] @ np.linalg.inv(K1)        # pixel of k -> pixel of k-1
                    warped = cv2.warpPerspective(prev[2], Hm, (I.shape[1], I.shape[0]),
                                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=-1)
                    ok = warped >= 0
                    D = np.abs(I - warped)
                    inner = (din >= 4) & (din <= 16) & ok
                    fm = filled & ok & (dout >= 1)
                    if fm.sum() > 50 and inner.sum() > 50:
                        row.update(flick_fill=float(D[fm].mean()), flick_inner=float(D[inner].mean()))
                if snap_dir:
                    keep('fill', row['fill_frac'], i, img, d0)
                    if 'seam_g' in row:
                        keep('seam', row['seam_g'] / max(row['band_g'], 1e-6), i, img, d0)
                    if 'flick_fill' in row:
                        keep('flicker', row['flick_fill'] / max(row['flick_inner'], 1e-6), i, img, d0)
            rows.append(row)
            prev = (i, k, I)
    finally:
        rd.close()
    meta = dict(render=os.path.abspath(render), plan=os.path.abspath(plan_path), width=width,
                start_record=int(recs[0]) if len(recs) else None, seconds=round(time.time() - t0, 1))
    if snap_dir:
        os.makedirs(snap_dir, exist_ok=True)
        snaps = []
        for kind, h in heaps.items():
            for score, i, img, d0 in sorted(h, reverse=True):
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cs, _ = cv2.findContours((d0 >= 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(vis, cs, -1, (0, 0, 255), 1)
                p = os.path.join(snap_dir, f'{kind}_{score:.3f}_f{i:04d}.jpg')
                cv2.imwrite(p, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
                snaps.append(p)
        meta['snaps'] = snaps
    return summarize(rows, meta)


def summarize(rows, meta):
    def arr(key):
        return np.array([r[key] for r in rows if key in r], float)
    ff = arr('fill_frac')
    out = dict(meta, frames=len(rows), fill_frac_mean=float(ff.mean()) if len(ff) else 0.0,
               fill_frac_p95=float(np.percentile(ff, 95)) if len(ff) else 0.0,
               fill_frac_max=float(ff.max()) if len(ff) else 0.0,
               frames_with_fill=float((ff > 0).mean()) if len(ff) else 0.0,
               plan_uncovered_mean=float(arr('plan_uncovered').mean()) if len(rows) else 0.0,
               plan_uncovered_p99=float(np.percentile(arr('plan_uncovered'), 99)) if len(rows) else 0.0,
               plan_uncovered_max=float(arr('plan_uncovered').max()) if len(rows) else 0.0)
    sg, bg = arr('seam_g'), arr('band_g')
    if len(sg):
        r = sg / np.maximum(bg, 1e-6)
        out.update(seam_ratio_pooled=float(sg.mean() / max(bg.mean(), 1e-6)), seam_ratio_median=float(np.median(r)),
                   seam_ratio_p95=float(np.percentile(r, 95)), seam_frames=int(len(sg)))
    fl, fi = arr('flick_fill'), arr('flick_inner')
    if len(fl):
        r = fl / np.maximum(fi, 1e-6)
        out.update(flicker_fill=float(fl.mean()), flicker_inner=float(fi.mean()),
                   flicker_ratio_pooled=float(fl.mean() / max(fi.mean(), 1e-6)), flicker_ratio_median=float(np.median(r)),
                   flicker_ratio_p95=float(np.percentile(r, 95)), flicker_frames=int(len(fl)))
    out['rows'] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('render')
    ap.add_argument('plan')
    ap.add_argument('--virt', required=True)
    ap.add_argument('--start-frame', type=int, default=None, help='plan record of render frame 0')
    ap.add_argument('--video', default=None, help='original video (with --src-start-frame: records matched by PTS)')
    ap.add_argument('--src-start-frame', type=int, default=None)
    ap.add_argument('--frames', type=int, default=None)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--snap-dir', default='')
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    if a.start_frame is None and (a.video is None or a.src_start_frame is None):
        ap.error('--start-frame or --video + --src-start-frame')
    res = measure(a.render, a.plan, a.virt, a.start_frame, a.frames, a.width, video=a.video,
                  src_start_frame=a.src_start_frame, snap_dir=a.snap_dir)
    short = {k: v for k, v in res.items() if k != 'rows'}
    print(json.dumps(short, indent=1))
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
