"""
Exact crop measure (SOURCE FOOTPRINT) and exact frame-pairing check for stabilised renders.

    .venv/bin/python -m eval.footprint RENDER ORIGINAL [--start S] [--orig-start S2] [--dur D]
                                        [--step-s 1.0] [--out fp.json] [--check-align]

Why
---
The old eval crop number (`visible_area_frac_mean`) fits a HOMOGRAPHY original->render. The O3 original is a
KB4 fisheye and the render is rectilinear, so that fit is only approximate; it read ~0.19 too high against the
true footprint (analytic 75% area read ~0.95).  This module instead models the render exactly the way every
gyro stabiliser (Stillpoint, Gyroflow) produces it:

    source ray   r_c = lens.unproject(p_orig)                       (embedded DJI KB4 lens, from telemetry)
    virtual ray  r_v = Rv . Exp(w * tau) . r_c                      (tau = source row time, -0.5..0.5 of readout)
    render px    p_r = f * (r_v.x / r_v.z, r_v.y / r_v.z) + c_r     (centred pinhole, square pixels)

i.e. a rotation Rv (virtual camera vs the source camera at its centre row), a rolling-shutter angular term w
(rotation of the source camera over the readout), and the output focal f.  The 7 parameters are fitted per
sampled frame pair from SIFT matches (homography-RANSAC on lens-undistorted coordinates for the start, then
robust Levenberg-Marquardt, soft-L1).  The SOURCE FOOTPRINT of that output frame is the render's image border
(pixel-edge extent) mapped back through the fitted model into the source (row-time fixed point, 4 iterations),
clipped to the source rectangle, as a fraction of the source area -- the same quantity as
work/verify/metric/fov2.py `src_area` (which used the Stillpoint plan and rasterisation).  When the plan is
available, `plan_footprint` computes it exactly from the plan's row matrices (no image matching at all).

Validation (work/eval_fix/validate_footprint.py -> validate_footprint.json): 18 renders of real Stillpoint
plans (DJI_0025/0034) with out_fx scaled by a known factor 0.90/1.00/1.10 give the plan's exact footprint to
<= 0.0004 (without the RS term: 0.003); on the real M1 renders the fit agrees with the plan-exact footprint to
<= 0.0004 per frame, focal to 0.1-0.4 px (1920), and reproduces the verifier's Gyroflow footprint (0032: 0.5446 vs
fov2.py 0.5444).

Frame pairing (`check_alignment`): at a few sample times the render frame is compared with original frames
k-4..k+4.  For each candidate the model above is fitted and the original is warped into the render's geometry;
the score is the zero-mean normalised cross-correlation of band-passed images over the overlap.  Motion blur,
sensor noise, parallax and moving objects make the true pair stand out (the old blur-series cross-correlation
gave false -6/-3 frame lags on 0028/0034; this method gives 0 on every render tested).

Units: areas are fractions of the source frame area; residuals are px at the analysis width (960).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN_W = 960


# ----------------------------------------------------------------------------------------------- lens / engine
def _engine():
    eng = os.path.join(ROOT, "engine")
    if eng not in sys.path:
        sys.path.insert(0, eng)


_LENS_CACHE: dict = {}


def source_lens(orig_path: str):
    """Embedded source lens (stillpoint.geom.Lens, full resolution) of an original clip, via the engine's
    telemetry parser (cached in work/cache).  Raises if the clip carries no usable lens."""
    key = os.path.abspath(orig_path)
    if key not in _LENS_CACHE:
        _engine()
        from stillpoint.telemetry import load_telemetry
        tel = load_telemetry(orig_path, cache_dir=os.path.join(ROOT, "work", "cache"))
        _LENS_CACHE[key] = tel.lens
    return _LENS_CACHE[key]


def lens_at_width(lens, width: int):
    return lens.scaled(width / float(lens.width))


# ----------------------------------------------------------------------------------------------- rotations
def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])


def rotvec_to_mat(v):
    v = np.asarray(v, float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3) + _skew(v)
    k = v / th
    K = _skew(k)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def rotvec_to_mats(V):
    """(N,3) -> (N,3,3) Rodrigues, vectorised."""
    V = np.asarray(V, float)
    th = np.linalg.norm(V, axis=1)
    small = th < 1e-12
    ths = np.where(small, 1.0, th)
    k = V / ths[:, None]
    K = np.zeros((len(V), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.where(small, 0.0, np.sin(th))[:, None, None]
    c = np.where(small, 0.0, 1 - np.cos(th))[:, None, None]
    R = np.eye(3)[None] + s * K + c * (K @ K)
    if small.any():  # first order for tiny angles
        Ks = np.zeros((int(small.sum()), 3, 3))
        v = V[small]
        Ks[:, 0, 1], Ks[:, 0, 2] = -v[:, 2], v[:, 1]
        Ks[:, 1, 0], Ks[:, 1, 2] = v[:, 2], -v[:, 0]
        Ks[:, 2, 0], Ks[:, 2, 1] = -v[:, 1], v[:, 0]
        R[small] = np.eye(3)[None] + Ks
    return R


def mat_to_rotvec(R):
    rv, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    return rv.ravel()


# ----------------------------------------------------------------------------------------------- matching
_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=3000)
    return _SIFT


def sift_features(img):
    return _sift().detectAndCompute(img, None)


def match_features(fa, fb, ratio=0.75):
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return np.zeros((0, 2)), np.zeros((0, 2))
    good = [m for m, n in (x for x in cv2.BFMatcher().knnMatch(da, db, k=2) if len(x) == 2)
            if m.distance < ratio * n.distance]
    pa = np.float64([ka[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pb = np.float64([kb[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pa, pb


# ----------------------------------------------------------------------------------------------- the model
class RenderModel:
    """Fitted mapping source(original) <-> render for one frame pair (analysis scale).

    Rv (3x3): virtual(render) ray = Rv @ Exp(w*tau) @ source ray;  f: render focal (analysis px);
    lens: source lens at analysis scale; (Wr, Hr): render analysis size."""

    def __init__(self, lens, Wr, Hr, Rv, w, f):
        self.lens, self.Wr, self.Hr = lens, Wr, Hr
        self.Rv, self.w, self.f = np.asarray(Rv, float), np.asarray(w, float), float(f)
        self.c = np.array([(Wr - 1) / 2.0, (Hr - 1) / 2.0])

    def tau(self, y_src):
        return (np.asarray(y_src, float) + 0.5) / self.lens.height - 0.5

    def forward(self, p_src):
        """source px (N,2) -> render px (N,2) (NaN behind the virtual camera)."""
        rc = self.lens.unproject(p_src)
        return self._fwd_rays(rc, self.tau(p_src[:, 1]), self.Rv, self.w, self.f)

    def _fwd_rays(self, rc, tau, Rv, w, f):
        # rolling-shutter term to first order: Exp(w tau) r = r + tau (w x r)  (|w tau| < 0.05 rad -> < 1e-3 error)
        rr = rc + tau[:, None] * np.cross(w[None, :], rc)
        rv = rr @ Rv.T
        z = rv[:, 2]
        zs = np.where(z > 1e-6, z, np.nan)
        return np.column_stack([f * rv[:, 0] / zs + self.c[0], f * rv[:, 1] / zs + self.c[1]])

    def inverse(self, p_r, iters=4):
        """render px (N,2) -> source px (N,2) (row-time fixed point)."""
        p_r = np.asarray(p_r, float)
        rv = np.column_stack([(p_r[:, 0] - self.c[0]) / self.f, (p_r[:, 1] - self.c[1]) / self.f, np.ones(len(p_r))])
        r0 = rv @ self.Rv  # Rv^T rv
        tau = np.zeros(len(p_r))
        uv = None
        for _ in range(iters):
            rc = r0 - tau[:, None] * np.cross(self.w[None, :], r0)
            uv = self.lens.project(rc)
            tau = self.tau(uv[:, 1])
        return uv

    def footprint(self, n_edge=64):
        """Fraction of the source frame covered by the render frame (render image extent, clipped to source)."""
        poly_r = border_polygon(self.Wr, self.Hr, n_edge)
        poly_s = self.inverse(poly_r)
        W, H = self.lens.width, self.lens.height
        return clipped_area(poly_s, W, H) / float(W * H), poly_s


def border_polygon(W, H, n=64):
    """Image extent of a W x H image in pixel-centre coords (edges at -0.5 and W-0.5), clockwise."""
    t = np.linspace(0, 1, n, endpoint=False)
    x0, x1, y0, y1 = -0.5, W - 0.5, -0.5, H - 0.5
    return np.concatenate([np.c_[x0 + t * (x1 - x0), y0 + 0 * t], np.c_[x1 + 0 * t, y0 + t * (y1 - y0)],
                           np.c_[x1 - t * (x1 - x0), y1 + 0 * t], np.c_[x0 + 0 * t, y1 - t * (y1 - y0)]])


def _clip_halfplane(P, axis, val, keep_greater):
    if len(P) == 0:
        return P
    out = []
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        ina = (a[axis] >= val) if keep_greater else (a[axis] <= val)
        inb = (b[axis] >= val) if keep_greater else (b[axis] <= val)
        if ina:
            out.append(a)
        if ina != inb:
            t = (val - a[axis]) / (b[axis] - a[axis])
            out.append(a + t * (b - a))
    return np.array(out).reshape(-1, 2)


def clipped_area(poly, W, H):
    """Area of polygon `poly` (source px) intersected with the source image extent [-0.5, W-0.5] x [-0.5, H-0.5]
    (Sutherland-Hodgman against the convex rectangle; exact for any simple subject polygon)."""
    P = np.asarray(poly, float)
    if not np.all(np.isfinite(P)):
        return float("nan")
    for axis, val, g in ((0, -0.5, True), (0, W - 0.5, False), (1, -0.5, True), (1, H - 0.5, False)):
        P = _clip_halfplane(P, axis, val, g)
    if len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def fit_render_model(p_src, p_r, lens, Wr, Hr, rs=True, init=None):
    """Fit RenderModel to matched points (source px, render px; analysis scale). Returns (model, info) or
    (None, info).  Robust: homography-RANSAC on undistorted coords -> (f, Rv) -> soft-L1 LM over
    (Rv, w, log f) -> inlier re-selection -> LM again."""
    info = {"n_matches": int(len(p_src))}
    if len(p_src) < 20:
        return None, info
    rc = lens.unproject(p_src)
    okz = rc[:, 2] > 0.08
    p_src, p_r, rc = p_src[okz], p_r[okz], rc[okz]
    if len(p_src) < 20:
        return None, info
    c = np.array([(Wr - 1) / 2.0, (Hr - 1) / 2.0])
    if init is None:
        u = (rc[:, :2] / rc[:, 2:3]).astype(np.float32)
        Hm, inl = cv2.findHomography(u, (p_r - c).astype(np.float32), cv2.RANSAC, 4.0, maxIters=4000, confidence=0.999)
        if Hm is None or inl is None or inl.sum() < 15:
            return None, info
        inl = inl.ravel().astype(bool)
        h1, h2, h3 = Hm
        n3 = np.linalg.norm(h3)
        f0 = math.sqrt(np.linalg.norm(h1) * np.linalg.norm(h2)) / max(n3, 1e-12)
        M = np.diag([1 / f0, 1 / f0, 1.0]) @ Hm / n3
        if np.linalg.det(M) < 0:
            M = -M
        U, _, Vt = np.linalg.svd(M)
        Rv0 = U @ Vt
        w0 = np.zeros(3)
    else:
        Rv0, w0, f0 = init
        inl = np.ones(len(p_src), bool)
    tau = (p_src[:, 1] + 0.5) / lens.height - 0.5
    model = RenderModel(lens, Wr, Hr, Rv0, w0, f0)

    def unpack(x):
        Rv = rotvec_to_mat(x[:3]) @ Rv0
        w = x[3:6] if rs else np.zeros(3)
        return Rv, w, f0 * math.exp(x[6])

    def resid(x, m):
        Rv, w, f = unpack(x)
        pr = model._fwd_rays(rc[m], tau[m], Rv, w, f)
        r = (pr - p_r[m]).ravel()
        return np.where(np.isfinite(r), r, 50.0)

    x = np.zeros(7)
    m = inl
    rng = np.random.default_rng(0)
    for it in range(3):
        if m.sum() < 15:
            return None, info
        mf = m
        if m.sum() > 600:          # LM on a fixed random subset (speed); inliers are re-selected on all points
            idx = np.flatnonzero(m)
            mf = np.zeros_like(m)
            mf[rng.choice(idx, 600, replace=False)] = True
        sol = least_squares(resid, x, args=(mf,), loss="soft_l1", f_scale=0.7, method="trf", max_nfev=40,
                            x_scale=np.array([1e-3] * 3 + [1e-2] * 3 + [1e-3]))
        x = sol.x
        Rv, w, f = unpack(x)
        pr = model._fwd_rays(rc, tau, Rv, w, f)
        e = np.hypot(*(pr - p_r).T)
        e = np.where(np.isfinite(e), e, 1e9)
        med = float(np.median(e[m]))
        m = e < max(1.0, 3.0 * 1.4826 * med)
    Rv, w, f = unpack(x)
    fit = RenderModel(lens, Wr, Hr, Rv, w, f)
    info.update(n_inliers=int(m.sum()), resid_median_px=float(np.median(e[m])),
                resid_rms_px=float(np.sqrt(np.mean(e[m] ** 2))), f=float(f),
                rs_rate=float(np.linalg.norm(w)), rot_deg=float(np.degrees(np.linalg.norm(mat_to_rotvec(Rv)))))
    return fit, info


# ----------------------------------------------------------------------------------------------- frame reading
def read_select(path, start, dur, width=AN_W, step=None, ranges=(), hwaccel=True):
    """Decode ONE sequential pass over the window (first frame at/after `start`, `dur` s) and return the frames
    whose window index n is a multiple of `step` or lies in one of the inclusive `ranges` [(a, b), ...]:
    dict {n: gray uint8}.  Selection happens in ffmpeg's select filter on the exact decoded frame index -- no
    seeking, so indices are exact (seek-based access can be off by one frame when a PTS equals the seek time)."""
    from .jitter_metrics import FFMPEG, analysis_size, probe
    import subprocess
    info = probe(path)
    w, h = analysis_size(info, width)
    terms = []
    if step:
        terms.append(f"not(mod(n\\,{int(step)}))")
    for a, b in ranges:
        terms.append(f"between(n\\,{int(a)}\\,{int(b)})")
    if not terms:
        return {}
    sel = "+".join(terms)
    cmd = [FFMPEG, "-v", "error", "-nostdin"] + (["-hwaccel", "videotoolbox"] if hwaccel else [])
    if start and start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", path]
    if dur and dur > 0:
        cmd += ["-t", f"{dur:.6f}"]
    # frames come out in decode order; their indices are recovered by replaying the same predicate in python
    cmd += ["-an", "-sn", "-dn", "-vf", f"select={sel},scale={w}:{h}:flags=area,format=gray",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    want = []
    n_total = int(round((dur if dur else max(info.duration - (start or 0), 0)) * info.fps)) + 4
    for n in range(n_total):
        if (step and n % int(step) == 0) or any(a <= n <= b for a, b in ranges):
            want.append(n)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 4)
    out = {}
    try:
        for n in want:
            buf = proc.stdout.read(w * h)
            if len(buf) < w * h:
                break
            out[n] = np.frombuffer(buf, np.uint8).reshape(h, w).copy()
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.terminate()
        proc.wait()
    return out


# ----------------------------------------------------------------------------------------------- footprint series
def footprint_rows(render_frames: dict, orig_frames: dict, la, wr, hr, legacy=False):
    """Fit the render model on every frame index present in both dicts -> list of row dicts (area, f_1920 ...).
    legacy=True also returns the old homography visible-area fraction from the same matches."""
    rows = []
    for i in sorted(set(render_frames) & set(orig_frames)):
        a, b = orig_frames[i], render_frames[i]
        ps, pr = match_features(sift_features(a), sift_features(b))
        model, info = fit_render_model(ps, pr, la, wr, hr)
        row = {"frame": int(i), **info}
        if model is not None:
            row["area"], _ = model.footprint()
            row["f_1920"] = model.f * 1920.0 / wr
        if legacy and len(ps) >= 12:
            from .jitter_metrics import crop_distortion_from_H
            wo = a.shape[1]
            p1, p2 = (ps / wo).astype(np.float32), (pr / wr).astype(np.float32)
            Hm, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0 / 960, maxIters=3000, confidence=0.999)
            if Hm is not None:
                Hm = Hm / Hm[2, 2]
                inl = inl.ravel().astype(bool)
                pp = cv2.perspectiveTransform(p1[inl].reshape(-1, 1, 2), Hm).reshape(-1, 2)
                c, ar, d = crop_distortion_from_H(Hm, a.shape[0] / wo, hr / wr)
                row.update(legacy_liu=c, legacy_area=ar, legacy_dv=d, legacy_herr=float(np.median(np.linalg.norm(pp - p2[inl], axis=1)) * 960),
                           legacy_inliers=int(inl.sum()))
        rows.append(row)
    return rows


def summarize_rows(rows, max_resid=0.8, min_inliers=40):
    good = [r for r in rows if "area" in r and r["resid_median_px"] <= max_resid and r["n_inliers"] >= min_inliers]
    if not good:
        return {"n": 0, "n_sampled": len(rows)}
    a = np.array([r["area"] for r in good])
    f = np.array([r["f_1920"] for r in good])
    return {"n": len(good), "n_sampled": len(rows), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()), "p10": float(np.percentile(a, 10)),
            "focal_1920_median": float(np.median(f)), "focal_1920_min": float(f.min()), "focal_1920_max": float(f.max()),
            "resid_median_px960": float(np.median([r["resid_median_px"] for r in good]))}


def footprint_series(render, orig, render_start=0.0, orig_start=0.0, dur=None, step=30, lens=None,
                     width=AN_W, verbose=False):
    """Source footprint of `render` (vs its original) at every `step`-th frame of the window.

    Frame i of the render window pairs with frame i of the original window (both decoded from the first frame
    at/after their start).  Returns dict(mean, median, min, max, n, rows=[...]) -- only fits with median
    residual <= 0.8 px (960) and >= 40 inliers are used."""
    from concurrent.futures import ThreadPoolExecutor
    from .jitter_metrics import analysis_size, probe
    t0 = time.time()
    lens = lens or source_lens(orig)
    wo, _ = analysis_size(probe(orig), width)
    wr, hr = analysis_size(probe(render), width)
    la = lens_at_width(lens, wo)
    with ThreadPoolExecutor(2) as ex:
        fo = ex.submit(read_select, orig, orig_start, dur, width, step)
        fr = ex.submit(read_select, render, render_start, dur, width, step)
        of, rf = fo.result(), fr.result()
    rows = footprint_rows(rf, of, la, wr, hr)
    out = {"method": "fitted KB4-source -> pinhole+RS render model (eval.footprint)", "step_frames": step,
           **summarize_rows(rows), "elapsed_s": time.time() - t0, "rows": rows}
    if verbose:
        print(f"  footprint {os.path.basename(render)}: mean {out.get('mean', float('nan')):.4f} (n={out['n']}/{len(rows)}) "
              f"in {out['elapsed_s']:.1f}s", file=sys.stderr)
    return out


def plan_footprint(plan, records, n_edge=64):
    """EXACT source footprint of a Stillpoint plan at plan records `records` (fraction of the source area), from
    the plan's row matrices (same maths as the renderer; row-time fixed point, 3 iterations).  No decoding."""
    lens = plan.lens
    out = []
    ow, oh = plan.out_w, plan.out_h
    poly = border_polygon(ow, oh, n_edge)
    for k in np.asarray(records, int):
        fx = float(plan.out_fx[k])
        cx, cy = (ow - 1) / 2.0, (oh - 1) / 2.0
        rv = np.c_[(poly[:, 0] - cx) / fx, (poly[:, 1] - cy) / fx, np.ones(len(poly))]
        mats = plan.row_mats[k]
        nr = mats.shape[0]
        y = np.full(len(rv), (plan.src_h - 1) / 2.0)
        uv = None
        for _ in range(3):
            g = np.clip(y * (nr - 1) / (plan.src_h - 1), 0, nr - 1)
            j0 = np.minimum(np.floor(g).astype(int), nr - 2)
            a = (g - j0)[:, None, None]
            Mi = mats[j0] * (1 - a) + mats[j0 + 1] * a
            rc = np.einsum("nij,nj->ni", Mi, rv)
            uv = lens.project(rc)
            y = uv[:, 1]
        out.append(clipped_area(uv, plan.src_w, plan.src_h) / float(plan.src_w * plan.src_h))
    return np.asarray(out)


# ----------------------------------------------------------------------------------------------- alignment
def _bandpass(img):
    x = img.astype(np.float32)
    return cv2.GaussianBlur(x, (0, 0), 1.0) - cv2.GaussianBlur(x, (0, 0), 4.0)


def warp_to_render(model, img_src, Wr, Hr):
    """Original frame resampled into the render's geometry by the fitted model (+ validity mask)."""
    # the mapping is smooth: evaluate it on an 8-px lattice and interpolate bilinearly (error << 0.01 px)
    s = 8
    gh, gw = (Hr - 1) // s + 2, (Wr - 1) // s + 2
    ys, xs = np.mgrid[0:gh, 0:gw] * s
    p = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    uvc = model.inverse(p).reshape(gh, gw, 2).astype(np.float32)
    big = cv2.resize(uvc, (gw * s, gh * s), interpolation=cv2.INTER_LINEAR)
    # cv2.resize puts sample i at (i + 0.5) * s - 0.5 -> shift so lattice point i sits at pixel i*s
    shift = (s - 1) / 2.0
    M = np.float32([[1, 0, shift], [0, 1, shift]])
    uv = cv2.warpAffine(big, M, (Wr, Hr), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REPLICATE)
    out = cv2.remap(img_src, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (uv[..., 0] > 1) & (uv[..., 0] < img_src.shape[1] - 2) & (uv[..., 1] > 1) & (uv[..., 1] < img_src.shape[0] - 2)
    return out, valid


def _ncc(a, b, m):
    a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def alignment_samples(n_total, n_samples=4, search=6):
    """Render frame indices used for the pairing check (spread over the window, away from the ends)."""
    lo, hi = search + 30, n_total - search - 30
    if hi <= lo:
        return [max(n_total // 2, 0)]
    return sorted({int(x) for x in np.linspace(max(lo, 0.08 * n_total), hi, n_samples)})


def alignment_from_frames(render_frames: dict, orig_frames: dict, js, search, la, wr, hr):
    """For each render frame j in js: fit the render model against original frames j+d (|d| <= search), warp the
    original into the render geometry and score band-passed NCC.  -> result dict (lag = consensus best d)."""
    samples = []
    for j in js:
        rf = render_frames.get(j)
        if rf is None:
            continue
        fr = sift_features(rf)
        bpr = _bandpass(rf)
        scores, resid = {}, {}
        for d in range(-search, search + 1):
            o = orig_frames.get(j + d)
            if o is None:
                continue
            ps, pr = match_features(sift_features(o), fr)
            model, info = fit_render_model(ps, pr, la, wr, hr)
            if model is None:
                continue
            wimg, valid = warp_to_render(model, o, wr, hr)
            valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            if valid.mean() < 0.3:
                continue
            scores[d] = _ncc(_bandpass(wimg), bpr, valid)
            resid[d] = info["resid_median_px"]
        if not scores:
            continue
        best = max(scores, key=scores.get)
        srt = sorted(scores.values(), reverse=True)
        samples.append({"frame": int(j), "best_offset": int(best), "ncc_best": srt[0],
                        "ncc_second": srt[1] if len(srt) > 1 else float("nan"),
                        "ncc_by_offset": {int(k): round(v, 4) for k, v in scores.items()},
                        "resid_by_offset": {int(k): round(v, 3) for k, v in resid.items()}})
    return alignment_consensus(samples, search)


def alignment_consensus(samples, search):
    """Consensus lag (most frequent best offset) over per-sample results of alignment_from_frames."""
    samples = sorted(samples, key=lambda s_: s_["frame"])
    offs = [s_["best_offset"] for s_ in samples]
    if offs:
        vals, cnt = np.unique(offs, return_counts=True)
        lag, agree = int(vals[np.argmax(cnt)]), float(cnt.max() / len(offs))
    else:
        lag, agree = None, 0.0
    return {"method": "per-sample frame matching (fitted render model + band-passed NCC)", "lag_frames": lag,
            "agreement": agree, "n_samples": len(samples), "search": search, "samples": samples}


def check_alignment(render, orig, render_start=0.0, orig_start=0.0, dur=None, n_samples=4, search=6,
                    lens=None, width=AN_W):
    """Exact frame-pairing check (render frame j shows original frame j + lag).  One sequential decode of each
    window (exact indices, no seeking)."""
    from concurrent.futures import ThreadPoolExecutor
    from .jitter_metrics import analysis_size, probe
    ir = probe(render)
    n_total = int(round((dur if dur else (ir.duration - render_start)) * ir.fps))
    lens = lens or source_lens(orig)
    wo, _ = analysis_size(probe(orig), width)
    wr, hr = analysis_size(ir, width)
    js = alignment_samples(n_total, n_samples, search)
    with ThreadPoolExecutor(2) as ex:
        fo = ex.submit(read_select, orig, orig_start, dur, width, None, [(j - search, j + search) for j in js])
        fr = ex.submit(read_select, render, render_start, dur, width, None, [(j, j) for j in js])
        of, rf = fo.result(), fr.result()
    return alignment_from_frames(rf, of, js, search, lens_at_width(lens, wo), wr, hr)


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("render")
    ap.add_argument("original")
    ap.add_argument("--start", type=float, default=0.0, help="window start in the render (s)")
    ap.add_argument("--orig-start", type=float, default=None, help="window start in the original (default --start)")
    ap.add_argument("--dur", type=float, default=None, help="window length (default: whole render)")
    ap.add_argument("--step-s", type=float, default=0.5, help="sample spacing (s)")
    ap.add_argument("--check-align", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from .jitter_metrics import probe
    fps = probe(a.render).fps
    os_ = a.start if a.orig_start is None else a.orig_start
    res = {"render": os.path.abspath(a.render), "original": os.path.abspath(a.original), "start": a.start,
           "orig_start": os_, "dur": a.dur,
           "footprint": footprint_series(a.render, a.original, a.start, os_, a.dur, max(int(round(a.step_s * fps)), 1),
                                         verbose=True)}
    if a.check_align:
        res["alignment"] = check_alignment(a.render, a.original, a.start, os_, a.dur)
    fp = res["footprint"]
    if not fp.get("n"):
        print("footprint: no usable frames"); return res
    print(f"footprint mean {fp['mean']:.4f} median {fp['median']:.4f} min {fp['min']:.4f} n={fp['n']} "
          f"focal(1920) {fp['focal_1920_median']:.1f} [{fp['focal_1920_min']:.1f}, {fp['focal_1920_max']:.1f}]"
          + (f"  lag {res['alignment']['lag_frames']} (agreement {res['alignment']['agreement']:.2f})" if a.check_align else ""))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    return res


if __name__ == "__main__":
    main()
