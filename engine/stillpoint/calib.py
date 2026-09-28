"""Self-calibration of the gyro <-> camera time model from the clip itself (WP-E).

What is estimated (research/sota_algorithms.md §3, ENGINE_SPEC.md §2):
    offset_s      time offset added to every row time              (bounded +-5 ms)
    readout_s     top->bottom readout time                          (bounded 0.5x..1.5x metadata)
    focal_scale   multiplies lens.fx/fy                             (bounded 0.95..1.05)
    extrinsic     small IMU->camera rotation, right-multiplied      (bounded +-3 deg per axis)
    skew          clock skew  t*(1+skew)                            (only when the windows span > 15 s)

Model (identical to TimeModel / plan_build):
    row time   tau(k, y) = frame_t[k]*(1+skew) + offset + readout*((y+0.5)/H - 0.5)
    camera     R_cam(tau) = R_imu(tau) @ R(extrinsic)

Data: KLT tracks on the ORIGINAL fisheye frames decoded at `width` px (gyro-predicted LK, forward-backward
checked), pair spans of 1, 2 and 4 frames, every observation carrying its own row time.

Residuals (both in analysis-width pixels):
  * de-rotated coplanarity (depth-free, parallax-safe): u_a = R_cam(tau_a) b_a, u_b = R_cam(tau_b) b_b,
    r = f * e.(u_a x u_b) / sqrt((|e x u_a|^2 + |e x u_b|^2)/2), with the per-frame-pair translation direction e
    eliminated in closed form (VarPro: smallest eigenvector of sum m m^T). Used only for frame pairs whose
    translation is actually observable (eigen-gap + parallax test), otherwise e is ill-defined.
  * rotation-only transfer error pi(R_cam(tau_b)^T u_a) - x_b for tracks classified FAR (low parallax).

Solver: coarse offset grid -> staged IRLS (Cauchy) Gauss-Newton via scipy least_squares (trf, bounded) ->
far/translation re-classification -> final solve. Uncertainty = formal covariance (inv(J^T J) * s^2, inflated
x2 for correlated track errors) combined with a leave-one-window-out jackknife. Any parameter whose
uncertainty is above its threshold (or that ends on a bound) falls back to the metadata default and the rest
are re-solved. Everything is reported in TimeModel.notes['calib'].
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import replace

import cv2
import numpy as np
from scipy.optimize import least_squares

from .geom import Lens, qconj, qexp, qfix_sign, qlog, qmul
from .types import Telemetry, TimeModel

PARAM_NAMES = ('offset_ms', 'readout_pct', 'focal_pct', 'ext_x_deg', 'ext_y_deg', 'ext_z_deg', 'skew_ppm')
BOUND = np.array([5.0, 50.0, 5.0, 3.0, 3.0, 3.0, 200.0])          # symmetric bounds, parameter units
PRIOR_SIGMA = np.array([5.0, 25.0, 3.0, 2.0, 2.0, 2.0, 100.0])    # weak priors (only matter when unobservable)
SIGMA_MAX = np.array([0.30, 4.0, 0.40, 0.30, 0.30, 0.30, 15.0])   # fall back to the default above this
NP = len(PARAM_NAMES)


# ============================================================================ small helpers

def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v (N,3) by unit quaternions q (N,4): R(q) v."""
    w = q[:, :1]
    u = q[:, 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _rotvec_to_mat(rv: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64).reshape(3, 1))
    return R


class _Orient:
    """Fast slerp on a (sub)series of camera->world quaternions (precomputed per-interval logs)."""

    def __init__(self, t: np.ndarray, q: np.ndarray):
        self.t = np.asarray(t, dtype=np.float64)
        self.q = qfix_sign(q)
        self.d = qlog(qmul(qconj(self.q[:-1]), self.q[1:]))
        self.dt = np.maximum(np.diff(self.t), 1e-12)

    def __call__(self, tq: np.ndarray) -> np.ndarray:
        tq = np.asarray(tq, dtype=np.float64)
        i = np.clip(np.searchsorted(self.t, tq, side='right') - 1, 0, len(self.t) - 2)
        u = np.clip((tq - self.t[i]) / self.dt[i], 0.0, 1.0)
        return qmul(self.q[i], qexp(self.d[i] * u[:, None]))


def _orient_for_spans(tel: Telemetry, spans: list[tuple[float, float]]) -> _Orient:
    """_Orient restricted to the union of time spans (keeps searchsorted/slerp arrays small)."""
    idx = []
    for a, b in sorted(spans):
        i0 = max(0, int(np.searchsorted(tel.imu_t, a, side='right')) - 2)
        i1 = min(len(tel.imu_t), int(np.searchsorted(tel.imu_t, b, side='left')) + 2)
        idx.append(np.arange(i0, i1))
    idx = np.unique(np.concatenate(idx))
    return _Orient(tel.imu_t[idx], tel.imu_q[idx])


def _lens_scaled_focal(lens: Lens, s: float) -> Lens:
    return replace(lens, fx=lens.fx * s, fy=lens.fy * s)


_LUT_CACHE: dict = {}


def _kb4_lut(k: np.ndarray):
    """theta_d -> theta lookup over the monotonic part of the KB4 polynomial."""
    key = tuple(float(c) for c in k)
    if key not in _LUT_CACHE:
        th = np.linspace(0.0, np.deg2rad(100.0), 8192)
        k1, k2, k3, k4 = key
        t2 = th * th
        thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        d = np.diff(thd)
        stop = np.flatnonzero(d <= 0)
        n = stop[0] + 1 if len(stop) else len(th)
        _LUT_CACHE[key] = (thd[:n], th[:n])
    return _LUT_CACHE[key]


def unproject_robust(lens: Lens, pix: np.ndarray) -> np.ndarray:
    """Like geom.Lens.unproject but always on the monotonic (first) branch of the KB4 polynomial.

    geom.Lens.unproject starts Newton at theta = theta_d, which for strongly non-linear lenses (Osmo Action 4:
    k = 0.155, 0.137, -0.094, 0.004) jumps to the far branch (theta ~ 90 deg) once r/f >~ 1.45 — the outer
    image corners. Here: LUT initial guess + 3 guarded Newton steps (exact to ~1e-12 rad)."""
    pix = np.asarray(pix, dtype=np.float64)
    if lens.model != 'kb4':
        return lens.unproject(pix)
    mx = (pix[..., 0] - lens.cx) / lens.fx
    my = (pix[..., 1] - lens.cy) / lens.fy
    thd = np.sqrt(mx * mx + my * my)
    lut_d, lut_t = _kb4_lut(lens.k)
    th = np.interp(thd, lut_d, lut_t)
    k1, k2, k3, k4 = [float(c) for c in lens.k]
    for _ in range(3):
        t2 = th * th
        f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - thd
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        th = np.clip(th - f / np.maximum(df, 1e-3), 0.0, lut_t[-1])
    s = np.where(thd < 1e-12, 1.0, np.sin(th) / np.where(thd < 1e-12, 1.0, thd))
    return np.stack([mx * s, my * s, np.cos(th)], axis=-1)


def _max_theta_d(lens: Lens, focal_margin: float = 0.95, max_theta_deg: float = 85.0) -> float:
    """Largest normalized fisheye radius r/f that unprojects reliably for any focal scale >= focal_margin
    (KB4 polynomials turn over near the image corners; Newton diverges past the turnover)."""
    if lens.model != 'kb4':
        return np.tan(np.deg2rad(max_theta_deg)) * focal_margin
    th = np.linspace(0, np.deg2rad(max_theta_deg), 2000)
    k1, k2, k3, k4 = [float(c) for c in lens.k]
    t2 = th * th
    thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    d = np.gradient(thd, th)
    bad = np.flatnonzero(d < 0.3)
    lim = thd[bad[0] - 1] if len(bad) else thd[-1]
    return float(lim * focal_margin)


def _x_to_timemodel_parts(x: np.ndarray, tr0: float):
    off = x[0] * 1e-3
    tr = tr0 * (1.0 + x[1] * 1e-2)
    s = 1.0 + x[2] * 1e-2
    ex = np.deg2rad(x[3:6])
    skew = x[6] * 1e-6
    return off, tr, s, ex, skew


# ============================================================================ decoding

_PROBE_CACHE: dict = {}


def _probe_stream(path: str) -> dict:
    if path not in _PROBE_CACHE:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                              'stream=width,height,pix_fmt,start_time', '-of', 'default=nw=1', path],
                             capture_output=True, text=True, check=True).stdout
        d = dict(line.split('=', 1) for line in out.strip().splitlines() if '=' in line)
        st = d.get('start_time', '0')
        _PROBE_CACHE[path] = dict(width=int(d['width']), height=int(d['height']), pix_fmt=d.get('pix_fmt', ''),
                                  start_time=float(st) if st not in ('N/A', '') else 0.0)
    return _PROBE_CACHE[path]


def decode_gray_window(path: str, frame_pts: np.ndarray, f0: int, n: int, width: int = 960
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Decode frames [f0, f0+n) as uint8 gray at `width` px. Returns (frame_indices (m,), frames (m,H,W)).

    ffmpeg -hwaccel videotoolbox + scale_vt, -fps_mode passthrough, input seek with -copyts; every decoded
    frame is mapped back to its index by PTS (showinfo), so indices always align with frame_pts.
    Local helper (stillpoint.video was being written concurrently); CPU-scale fallback if scale_vt fails.
    """
    pr = _probe_stream(path)
    W = int(width)
    H = int(round(pr['height'] * W / pr['width'] / 2.0)) * 2
    fp = np.asarray(frame_pts, dtype=np.float64)
    rel = fp - fp[0]
    dt = float(np.median(np.diff(fp))) if len(fp) > 1 else 1 / 60
    ss = max(0.0, rel[f0] - 0.5 * dt)
    hwfmt = 'p010le' if ('10' in pr['pix_fmt'] or '12' in pr['pix_fmt']) else 'nv12'
    chains = [
        (['-hwaccel', 'videotoolbox', '-hwaccel_output_format', 'videotoolbox_vld'],
         f'scale_vt=w={W}:h={H},hwdownload,format={hwfmt},format=gray,showinfo'),
        (['-hwaccel', 'videotoolbox'], f'scale={W}:{H}:flags=area,format=gray,showinfo'),
    ]
    for pre, vf in chains:
        cmd = (['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'info'] + pre +
               ['-ss', f'{ss:.6f}', '-copyts', '-i', path, '-map', '0:v:0', '-an', '-sn', '-dn',
                '-frames:v', str(int(n)), '-fps_mode', 'passthrough', '-vf', vf,
                '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'])
        p = subprocess.run(cmd, capture_output=True)
        m = len(p.stdout) // (W * H)
        if p.returncode != 0 or m == 0:
            continue
        frames = np.frombuffer(p.stdout[:m * W * H], np.uint8).reshape(m, H, W)
        pts = [float(v) for v in re.findall(rb'pts_time:\s*([-0-9.eE+]+)', p.stderr)]
        if len(pts) >= m:
            pts = np.array(pts[:m]) - pr['start_time']
            idx = np.searchsorted(rel, pts)
            idx = np.clip(idx, 1, len(rel) - 1)
            idx = np.where(np.abs(rel[idx - 1] - pts) < np.abs(rel[idx] - pts), idx - 1, idx)
            if np.max(np.abs(rel[idx] - pts)) > 0.25 * dt:
                raise RuntimeError('decoded PTS do not match telemetry frame_pts')
        else:  # no showinfo output (should not happen): assume sequential from f0
            idx = f0 + np.arange(m)
        return idx.astype(np.int64), frames
    raise RuntimeError(f'ffmpeg decode failed for {path} frames {f0}..{f0 + n}')


# ============================================================================ window selection

def _frame_rates(tel: Telemetry) -> np.ndarray:
    """(F-1,3) mean angular velocity (camera frame, rad/s) between consecutive frame centres."""
    orient = _Orient(tel.imu_t, tel.imu_q)
    t = np.clip(tel.frame_t, tel.imu_t[0], tel.imu_t[-1])
    q = orient(t)
    d = qlog(qmul(qconj(q[:-1]), q[1:]))
    return d / np.maximum(np.diff(tel.frame_t), 1e-6)[:, None]


def select_windows(tel: Telemetry, n_windows: int, window_frames: int, width: int = 960) -> list[dict]:
    """Rank candidate windows by how observable the calibration parameters are.

    score = rms(frame-to-frame change of angular velocity)       (offset / readout excitation)
            * min(1, rms|w| / 0.6 rad/s)                         (readout / focal / extrinsic need rotation)
            * mean 1/(1+(blur/4px)^2)                            (tracking dies in motion blur)
    Static (rms|w| < 3 deg/s) windows are skipped. Windows never straddle a segment boundary and keep 0.5 s
    away from the clip ends. Greedy non-maximum suppression (no overlap, >= 1 window length apart).
    """
    F = tel.n_frames
    L = int(window_frames)
    if F < L + 4:
        return []
    w = _frame_rates(tel)                                        # (F-1,3)
    speed = np.linalg.norm(w, axis=1)
    dw = np.linalg.norm(np.diff(w, axis=0), axis=1)              # (F-2,)
    fa = tel.lens.fx * width / tel.width
    exp = np.asarray(tel.exposure_s, dtype=np.float64) if tel.exposure_s is not None else np.zeros(F)
    exp = np.broadcast_to(exp, (F,)) if np.ndim(exp) == 0 else exp
    blur = fa * speed * exp[:-1]
    pen = 1.0 / (1.0 + (blur / 4.0) ** 2)

    ok = np.ones(F, bool)
    margin = int(round(0.5 * tel.fps))
    ok[:margin] = False
    ok[F - margin:] = False
    ok &= (tel.frame_t - 0.03 > tel.imu_t[0]) & (tel.frame_t + 0.03 < tel.imu_t[-1])
    seg_id = np.full(F, -1)
    segs = tel.segments or [(0, F - 1)]
    for i, (a, b) in enumerate(segs):
        seg_id[int(a):int(b) + 1] = i

    cands = []
    step = max(1, L // 4)
    for s in range(0, F - L - 1, step):
        e = s + L
        if not ok[s:e].all() or seg_id[s] < 0 or seg_id[s] != seg_id[e - 1]:
            continue
        rate_rms = float(np.sqrt(np.mean(speed[s:e - 1] ** 2)))
        if rate_rms < np.deg2rad(3.0):
            continue
        acc_rms = float(np.sqrt(np.mean(dw[s:e - 2] ** 2)))
        score = acc_rms * min(1.0, rate_rms / 0.6) * float(np.mean(pen[s:e - 1]))
        cands.append(dict(start=s, n=L, score=score, rate_rms_dps=np.rad2deg(rate_rms),
                          dw_rms_dps=np.rad2deg(acc_rms), blur_px_med=float(np.median(blur[s:e - 1]))))
    cands.sort(key=lambda c: -c['score'])
    chosen: list[dict] = []
    for c in cands:
        if all(abs(c['start'] - o['start']) >= (3 * L) // 2 for o in chosen):
            chosen.append(c)
        if len(chosen) >= n_windows:
            break
    return sorted(chosen, key=lambda c: c['start'])


# ============================================================================ tracking

def _track_window(frames: np.ndarray, fidx: np.ndarray, predict, grid=(8, 6), per_cell=10,
                  fb_thr=0.35, border=6) -> dict:
    """KLT tracks through a window. predict(ka, kb, pts) -> gyro-predicted positions in frame kb.
    Returns observation arrays (track id, local frame index, x, y) with tracks contiguous in time."""
    n, H, W = frames.shape
    gx, gy = grid
    cw, ch = W / gx, H / gy
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.005),
              minEigThreshold=1e-4)
    pts = np.zeros((0, 2), np.float32)
    ids = np.zeros(0, np.int64)
    next_id = 0
    obs_id, obs_j, obs_xy = [], [], []
    for j in range(n):
        img = frames[j]
        if j > 0 and len(pts):
            prev = frames[j - 1]
            init = predict(int(fidx[j - 1]), int(fidx[j]), pts.astype(np.float64)).astype(np.float32)
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, img, pts, init.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            p0b, st2, _ = cv2.calcOpticalFlowPyrLK(img, prev, p1, pts.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            good = (st1[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(p0b - pts, axis=1) < fb_thr)
            good &= (p1[:, 0] > border) & (p1[:, 0] < W - 1 - border) & (p1[:, 1] > border) & (p1[:, 1] < H - 1 - border)
            pts, ids = p1[good], ids[good]
        # top up per grid cell
        cx = np.clip((pts[:, 0] / cw).astype(int), 0, gx - 1) if len(pts) else np.zeros(0, int)
        cy = np.clip((pts[:, 1] / ch).astype(int), 0, gy - 1) if len(pts) else np.zeros(0, int)
        counts = np.zeros((gy, gx), int)
        np.add.at(counts, (cy, cx), 1)
        mask = np.full((H, W), 255, np.uint8)
        for p in pts:
            cv2.circle(mask, (int(p[0]), int(p[1])), 8, 0, -1)
        new = []
        for iy in range(gy):
            for ix in range(gx):
                need = per_cell - counts[iy, ix]
                if need <= 2:
                    continue
                x0, x1 = int(ix * cw), int((ix + 1) * cw)
                y0, y1 = int(iy * ch), int((iy + 1) * ch)
                x0b, y0b = max(x0, border + 2), max(y0, border + 2)
                x1b, y1b = min(x1, W - border - 2), min(y1, H - border - 2)
                if x1b - x0b < 8 or y1b - y0b < 8:
                    continue
                c = cv2.goodFeaturesToTrack(img[y0b:y1b, x0b:x1b], maxCorners=int(need), qualityLevel=0.02,
                                            minDistance=8, mask=mask[y0b:y1b, x0b:x1b], blockSize=7)
                if c is not None:
                    new.append(c.reshape(-1, 2) + np.array([x0b, y0b], np.float32))
        if new:
            nw = np.concatenate(new).astype(np.float32)
            nw = cv2.cornerSubPix(img, nw.reshape(-1, 1, 2), (4, 4), (-1, -1),
                                  (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.01)).reshape(-1, 2)
            pts = np.concatenate([pts, nw])
            ids = np.concatenate([ids, next_id + np.arange(len(nw))])
            next_id += len(nw)
        pts = pts.reshape(-1, 2).astype(np.float32)
        obs_id.append(ids.copy())
        obs_j.append(np.full(len(ids), j))
        obs_xy.append(pts.astype(np.float64).copy())
    oid = np.concatenate(obs_id)
    oj = np.concatenate(obs_j)
    oxy = np.concatenate(obs_xy)
    order = np.lexsort((oj, oid))
    return dict(tid=oid[order], j=oj[order], xy=oxy[order], n_frames=n)


def _make_pairs(tr: dict, fidx: np.ndarray, spans=(1, 2, 4)) -> dict:
    tid, j, xy = tr['tid'], tr['j'], tr['xy']
    out = {k: [] for k in ('ia', 'ib', 'span')}
    N = len(tid)
    for s in spans:
        if N <= s:
            continue
        a = np.arange(N - s)
        b = a + s
        ok = (tid[b] == tid[a]) & (j[b] == j[a] + s)
        out['ia'].append(a[ok])
        out['ib'].append(b[ok])
        out['span'].append(np.full(int(ok.sum()), s))
    ia = np.concatenate(out['ia']) if out['ia'] else np.zeros(0, int)
    ib = np.concatenate(out['ib']) if out['ib'] else np.zeros(0, int)
    return dict(xa=xy[ia], xb=xy[ib], ka=fidx[j[ia]], kb=fidx[j[ib]], tid=tid[ia],
                span=np.concatenate(out['span']) if out['span'] else np.zeros(0, int))


# ============================================================================ the problem

class _Problem:
    """Residual machinery over all track pairs (analysis-resolution pixels)."""

    def __init__(self, tel: Telemetry, lens_a: Lens, orient: _Orient, pairs: dict, H: int):
        self.lens0 = lens_a
        self.f0 = float(lens_a.fx)
        self.orient = orient
        self.tr0 = float(tel.readout_s)
        self.H = H
        self.xa, self.xb = pairs['xa'], pairs['xb']
        self.fta = tel.frame_t[pairs['ka']]
        self.ftb = tel.frame_t[pairs['kb']]
        self.rowa = (self.xa[:, 1] + 0.5) / H - 0.5
        self.rowb = (self.xb[:, 1] + 0.5) / H - 0.5
        self.span = pairs['span']
        self.win = pairs['win']
        self.blk = pairs['blk']
        self.tid = pairs['tid']
        grp_key = pairs['ka'].astype(np.int64) * 16 + pairs['span']
        _, self.grp = np.unique(grp_key, return_inverse=True)
        self.G = int(self.grp.max()) + 1 if len(self.grp) else 0
        self.P = len(self.xa)
        self._ray_cache: dict = {}
        self._q_cache: dict = {}

    # --- cached pieces
    def rays(self, s: float):
        if s not in self._ray_cache:
            if len(self._ray_cache) > 4:
                self._ray_cache.clear()
            lens = _lens_scaled_focal(self.lens0, s)
            self._ray_cache[s] = (unproject_robust(lens, self.xa), unproject_robust(lens, self.xb))
        return self._ray_cache[s]

    def quats(self, off: float, tr: float, skew: float):
        key = (off, tr, skew)
        if key not in self._q_cache:
            if len(self._q_cache) > 4:
                self._q_cache.clear()
            ta = self.fta * (1.0 + skew) + off + tr * self.rowa
            tb = self.ftb * (1.0 + skew) + off + tr * self.rowb
            self._q_cache[key] = (self.orient(ta), self.orient(tb))
        return self._q_cache[key]

    def world_rays(self, x: np.ndarray, sel: np.ndarray):
        off, tr, s, ex, skew = _x_to_timemodel_parts(x, self.tr0)
        ba, bb = self.rays(s)
        qa, qb = self.quats(off, tr, skew)
        RE = _rotvec_to_mat(ex)
        ua = _qrot(qa[sel], ba[sel] @ RE.T)
        ub = _qrot(qb[sel], bb[sel] @ RE.T)
        return ua, ub, qb[sel], RE, s

    # --- residuals
    def transfer(self, x: np.ndarray, sel: np.ndarray) -> np.ndarray:
        """(n,2) predicted - observed position in frame b (rotation-only)."""
        ua, _, qb, RE, s = self.world_rays(x, sel)
        rc = _qrot(qconj(qb), ua) @ RE                     # R_E^T R_imu_b^T u_a
        lens = _lens_scaled_focal(self.lens0, s)
        pred = lens.project(rc)
        bad = ~lens.valid_ray(rc, 88.0)
        r = pred - self.xb[sel]
        r[bad] = 0.0
        return r

    def coplanar(self, x: np.ndarray, sel: np.ndarray, w: np.ndarray, gvalid: np.ndarray,
                 e_ref: np.ndarray | None):
        """(n,) epipolar-plane distance (px), per-group translation eliminated (VarPro). Returns (r, e)."""
        ua, ub, _, _, _ = self.world_rays(x, sel)
        m = np.cross(ua, ub)
        g = self.grp[sel]
        S = np.zeros((self.G, 3, 3))
        for a in range(3):
            for b in range(a, 3):
                v = np.bincount(g, weights=w * m[:, a] * m[:, b], minlength=self.G)
                S[:, a, b] = v
                S[:, b, a] = v
        S[~gvalid] = np.eye(3)
        _, V = np.linalg.eigh(S)
        e = V[:, :, 0]
        if e_ref is not None:
            e = e * np.where(np.sum(e * e_ref, axis=1) < 0, -1.0, 1.0)[:, None]
        eg = e[g]
        num = np.sum(eg * m, axis=1)
        den = np.sqrt(0.5 * (np.sum(np.cross(eg, ua) ** 2, axis=1) + np.sum(np.cross(eg, ub) ** 2, axis=1)))
        r = self.f0 * num / np.maximum(den, 0.05)
        return r, e

    def group_translation(self, x: np.ndarray, sel: np.ndarray, w: np.ndarray, thr_px: float):
        """Which frame pairs have an observable translation direction (eigen-gap + parallax)."""
        ua, ub, _, _, _ = self.world_rays(x, sel)
        m = np.cross(ua, ub)
        g = self.grp[sel]
        S = np.zeros((self.G, 3, 3))
        for a in range(3):
            for b in range(a, 3):
                v = np.bincount(g, weights=w * m[:, a] * m[:, b], minlength=self.G)
                S[:, a, b] = v
                S[:, b, a] = v
        cnt = np.bincount(g, weights=(w > 0).astype(float), minlength=self.G)
        wsum = np.bincount(g, weights=w, minlength=self.G)
        lam, V = np.linalg.eigh(S + 1e-18 * np.eye(3))
        par = self.f0 * np.sqrt(np.maximum(lam[:, 1], 0) / np.maximum(wsum, 1e-9))
        valid = (cnt >= 12) & (par > thr_px) & (lam[:, 1] > 10.0 * np.maximum(lam[:, 0], 1e-30))
        return valid, V[:, :, 0], par


# ============================================================================ solver pieces

def _cauchy_w(r: np.ndarray, c: float) -> np.ndarray:
    return 1.0 / (1.0 + (r / c) ** 2)


def _robust_scale(r: np.ndarray, floor: float) -> float:
    if len(r) == 0:
        return floor
    return max(floor, 1.4826 * float(np.median(np.abs(r))))


class _Fit:
    def __init__(self, prob: _Problem, width_scale: float):
        self.p = prob
        self.ws = width_scale     # analysis width / 960
        self.nfev = 0

    def solve(self, x0, active, sel_t, w_t, c_t, sel_c, w_c, c_c, gvalid, e_ref, max_nfev=40):
        p = self.p
        x0 = np.asarray(x0, float).copy()
        act = np.flatnonzero(active)
        sw_t = np.sqrt(w_t)[:, None] / c_t
        sw_c = np.sqrt(w_c) / c_c

        def full(z):
            x = x0.copy()
            x[act] = z
            return x

        def fun(z):
            self.nfev += 1
            x = full(z)
            out = []
            if len(sel_t):
                out.append((p.transfer(x, sel_t) * sw_t).ravel())
            if len(sel_c):
                rc, _ = p.coplanar(x, sel_c, w_c, gvalid, e_ref)
                out.append(rc * sw_c)
            out.append(x[act] / PRIOR_SIGMA[act])
            return np.concatenate(out)

        lo, hi = -BOUND[act], BOUND[act]
        z0 = np.clip(x0[act], lo + 1e-9, hi - 1e-9)
        res = least_squares(fun, z0, bounds=(lo, hi), method='trf', x_scale=1.0, diff_step=1e-4,
                            max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        x = full(res.x)
        n_data = len(res.fun) - len(act)
        J = res.jac
        chi2 = float(np.sum(res.fun[:n_data] ** 2))
        s2 = max(1.0, chi2 / max(1, n_data - len(act)))
        try:
            cov = np.linalg.inv(J.T @ J) * s2
        except np.linalg.LinAlgError:
            cov = np.full((len(act), len(act)), np.inf)
        covf = np.full((NP, NP), 0.0)
        covf[np.ix_(act, act)] = cov
        return x, covf


# ============================================================================ public API

def self_calibrate(tel: Telemetry, video_path: str, max_windows: int = 6, width: int = 960, *,
                   window_s: float = 1.0, fit_skew: bool | None = None, max_pairs: int = 30000,
                   frame_reader=None, verbose: bool = False) -> TimeModel:
    """Estimate a TimeModel (offset, readout, focal scale, extrinsic, skew) from the clip itself.

    frame_reader: optional callable (f0, n) -> (frame_indices, frames uint8 (m,H,W) at `width`) used instead of
    ffmpeg decoding (tests / other front-ends). Always returns a TimeModel; parameters that are not confidently
    observable stay at their metadata defaults. Diagnostics in tm.notes['calib'].
    """
    t_start = time.time()
    notes: dict = dict(version=1, width=width, param_names=list(PARAM_NAMES))
    if tel.eis_baked:
        notes['status'] = 'skipped: EIS baked into the picture (gyro no longer describes the pixels)'
        return TimeModel(notes={'calib': notes})
    no_readout_meta = tel.readout_s is None or not np.isfinite(tel.readout_s) or tel.readout_s <= 0
    if no_readout_meta:
        tel = replace(tel, readout_s=0.5 / tel.fps)   # no metadata: start from half a frame period
        notes['readout_default'] = 'half frame period (no metadata)'

    ws = width / 960.0
    L = int(round(window_s * tel.fps))
    cands = select_windows(tel, max_windows + 1, L, width)
    notes['candidates'] = [dict(c) for c in cands]
    if not cands:
        notes['status'] = 'no window with enough rotation'
        return TimeModel(notes={'calib': notes})

    scale = width / tel.width
    lens_a = tel.lens.scaled(scale)
    spans_t = []
    for c in cands:
        a, b = tel.frame_t[c['start']], tel.frame_t[c['start'] + c['n'] - 1]
        spans_t.append((a - 0.015, b + 0.015))
    orient = _orient_for_spans(tel, [(a - 0.01, b + 0.01) for a, b in spans_t])

    # gyro predictor at metadata defaults (row time of the source point used for both frames)
    tr0 = float(tel.readout_s)

    def predict(ka, kb, pts):
        H_ = lens_a.height if lens_a.height else int(round(tel.height * scale))
        rows = (pts[:, 1] + 0.5) / H_ - 0.5
        qa = orient(tel.frame_t[ka] + tr0 * rows)
        qb = orient(tel.frame_t[kb] + tr0 * rows)
        u = _qrot(qa, unproject_robust(lens_a, pts))
        return lens_a.project(_qrot(qconj(qb), u))

    # decode + track each candidate, keep the textured ones
    t_dec = t_trk = 0.0
    wins = []
    Hs = None
    for wi, c in enumerate(cands):
        t0 = time.time()
        if frame_reader is not None:
            fidx, frames = frame_reader(c['start'], c['n'])
        else:
            fidx, frames = decode_gray_window(video_path, tel.frame_pts, c['start'], c['n'], width)
        t_dec += time.time() - t0
        t0 = time.time()
        Hs = frames.shape[1]
        tr = _track_window(frames, np.asarray(fidx), predict)
        pr = _make_pairs(tr, np.asarray(fidx))
        t_trk += time.time() - t0
        per_frame = len(tr['tid']) / max(1, tr['n_frames'])
        c = dict(c, tracks_per_frame=float(per_frame), n_pairs=int(len(pr['xa'])))
        if per_frame < 40 or len(pr['xa']) < 500:
            c['rejected'] = 'low texture'
            wins.append((c, None))
            continue
        wins.append((c, pr))
    good = [(c, pr) for c, pr in wins if pr is not None]
    good.sort(key=lambda cp: -cp[0]['score'] * min(1.0, cp[0]['tracks_per_frame'] / 150.0))
    good = sorted(good[:max_windows], key=lambda cp: cp[0]['start'])
    notes['windows'] = [c for c, _ in good]
    notes['rejected_windows'] = [c for c, pr in wins if pr is None]
    if not good:
        notes['status'] = 'no textured window'
        return TimeModel(notes={'calib': notes})
    if lens_a.height != Hs:
        lens_a = replace(lens_a, height=Hs)

    # assemble pairs (subsample to max_pairs); drop observations outside the reliable lens domain
    keys = ('xa', 'xb', 'ka', 'kb', 'tid', 'span')
    pairs = {k: np.concatenate([pr[k] for _, pr in good]) for k in keys}
    pairs['win'] = np.concatenate([np.full(len(pr['xa']), i) for i, (_, pr) in enumerate(good)])
    lim = _max_theta_d(lens_a)

    def _rn(xy):
        return np.hypot((xy[:, 0] - lens_a.cx) / lens_a.fx, (xy[:, 1] - lens_a.cy) / lens_a.fy)
    inside = (_rn(pairs['xa']) < lim) & (_rn(pairs['xb']) < lim)
    notes['lens_domain_dropped_pairs'] = int((~inside).sum())
    pairs = {k: v[inside] for k, v in pairs.items()}
    pairs['tid'] = pairs['tid'] + pairs['win'] * 10_000_000
    # jackknife blocks: each window split into two halves in time
    mids = np.array([c['start'] + c['n'] // 2 for c, _ in good])
    pairs['blk'] = 2 * pairs['win'] + (pairs['ka'] >= mids[pairs['win']])
    rng = np.random.default_rng(0)
    if len(pairs['xa']) > max_pairs:
        keep = np.sort(rng.choice(len(pairs['xa']), max_pairs, replace=False))
        pairs = {k: v[keep] for k, v in pairs.items()}
    prob = _Problem(tel, lens_a, orient, pairs, Hs)
    fit = _Fit(prob, ws)
    P = prob.P
    allp = np.arange(P)
    notes['n_pairs'] = int(P)

    n_win = len(good)
    t_span = tel.frame_t[good[-1][0]['start']] - tel.frame_t[good[0][0]['start']]
    if fit_skew is None:
        fit_skew = bool(n_win >= 3 and t_span > 15.0 and tel.has_highrate)
    active = np.array([True, True, True, True, True, True, bool(fit_skew)])
    if not tel.has_highrate:
        # 60 Hz orientation cannot resolve intra-frame (row) timing: keep readout at its default
        active[1] = False
        notes['warning'] = 'per-frame orientation only: readout not estimated; offset/extrinsic only'

    # ---- stage A: coarse offset grid on the robust transfer cost (all tracks)
    x = np.zeros(NP)
    grid = np.arange(-5.0, 5.0001, 0.25)
    cost = []
    for o in grid:
        xg = x.copy()
        xg[0] = o
        r = np.linalg.norm(prob.transfer(xg, allp), axis=1) / prob.span
        cost.append(float(np.sum(np.log1p((r / (1.0 * ws)) ** 2))))
    cost = np.array(cost)
    i = int(np.argmin(cost))
    x[0] = grid[i]
    if 0 < i < len(grid) - 1:
        c0, c1, c2 = cost[i - 1], cost[i], cost[i + 1]
        den = c0 - 2 * c1 + c2
        if den > 0:
            x[0] += 0.25 * 0.5 * (c0 - c2) / den
    notes['coarse_offset_ms'] = float(x[0])
    r0 = np.linalg.norm(prob.transfer(np.zeros(NP), allp), axis=1)
    notes['median_transfer_px_default'] = float(np.median(r0))

    # ---- stage B: all params, transfer residual on all tracks, loose Cauchy (parallax tolerant)
    empty = np.zeros(0, int)
    for _ in range(1):
        rt = np.linalg.norm(prob.transfer(x, allp), axis=1)
        c_t = _robust_scale(rt, 0.15 * ws)
        w_t = _cauchy_w(rt, max(c_t, 1.0 * ws))
        x, cov = fit.solve(x, active, allp, w_t, max(c_t, 1.0 * ws), empty, np.zeros(0), 1.0,
                           np.zeros(prob.G, bool), None, max_nfev=30)
    notes['stageB'] = x.tolist()

    # ---- stages D/E: classify far tracks + translating frame pairs, solve with both residuals
    def classify(x):
        rt = np.linalg.norm(prob.transfer(x, allp), axis=1) / prob.span
        # per-track median (tracks are id'd uniquely across windows)
        u, inv = np.unique(prob.tid, return_inverse=True)
        med = np.zeros(len(u))
        order = np.argsort(inv, kind='stable')
        bounds = np.searchsorted(inv[order], np.arange(len(u) + 1))
        vals = rt[order]
        for k in range(len(u)):
            med[k] = np.median(vals[bounds[k]:bounds[k + 1]])
        far = med[inv] < 0.3 * ws
        gvalid, e_ref, par = prob.group_translation(x, allp, (rt < 20 * ws).astype(float), 0.5 * ws)
        cop = gvalid[prob.grp]
        return far, cop, gvalid, e_ref, par

    def irls(x, far, cop, gvalid, e_ref, rounds, active, subset=None, fixed_w=None, max_nfev=40):
        sel_t = np.flatnonzero(far if subset is None else (far & subset))
        sel_c = np.flatnonzero(cop if subset is None else (cop & subset))
        cov = np.zeros((NP, NP))
        wts = fixed_w
        for _ in range(rounds):
            if fixed_w is None:
                rt = np.linalg.norm(prob.transfer(x, sel_t), axis=1) if len(sel_t) else np.zeros(0)
                rc = prob.coplanar(x, sel_c, np.ones(len(sel_c)), gvalid, e_ref)[0] if len(sel_c) else np.zeros(0)
                c_t = _robust_scale(rt / np.sqrt(2), 0.08 * ws)
                c_c = _robust_scale(rc, 0.08 * ws)
                wts = (_cauchy_w(rt, 2 * c_t), c_t, _cauchy_w(rc, 2 * c_c), c_c)
            w_t, c_t, w_c, c_c = wts
            x, cov = fit.solve(x, active, sel_t, w_t, c_t, sel_c, w_c, c_c, gvalid, e_ref, max_nfev=max_nfev)
        return x, cov, wts, sel_t, sel_c

    far, cop, gvalid, e_ref, par = classify(x)
    x, cov, wts, sel_t, sel_c = irls(x, far, cop, gvalid, e_ref, 2, active)
    far, cop, gvalid, e_ref, par = classify(x)
    x, cov, wts, sel_t, sel_c = irls(x, far, cop, gvalid, e_ref, 2, active)
    notes['n_far_pairs'] = int(far.sum())
    notes['n_coplanar_pairs'] = int(cop.sum())
    notes['n_translating_groups'] = int(gvalid.sum())
    notes['n_groups'] = int(prob.G)
    notes['robust_scale_px'] = dict(transfer=float(wts[1]), coplanar=float(wts[3]))

    sig_formal = np.sqrt(np.maximum(np.diag(cov), 0))

    # ---- jackknife (fixed weights / classification): leave-one-WINDOW-out when >= 3 windows (captures
    # scene-dependent bias such as parallax / moving subjects that is shared inside a window), otherwise
    # leave-one-half-window-out.
    x_full = x.copy()
    sig_jack = np.zeros(NP)
    groups = prob.win if n_win >= 3 else prob.blk
    blocks = np.unique(groups)
    nb = len(blocks)
    if nb >= 3:
        xs = []
        for wi in blocks:
            sub = groups != wi
            wt_sub = (wts[0][sub[sel_t]], wts[1], wts[2][sub[sel_c]], wts[3])
            xj, _, _, _, _ = irls(x_full, far, cop, gvalid, e_ref, 1, active, subset=sub, fixed_w=wt_sub,
                                  max_nfev=12)
            xs.append(xj)
        xs = np.array(xs)
        sig_jack = np.sqrt((nb - 1) / nb * np.sum((xs - xs.mean(0)) ** 2, axis=0))
        notes['jackknife_unit'] = 'window' if n_win >= 3 else 'half-window'
        notes['jackknife_estimates'] = xs.tolist()
    sigma = np.sqrt((2.0 * sig_formal) ** 2 + sig_jack ** 2)
    if nb < 3:
        sigma = 3.0 * sig_formal

    # ---- fallback per parameter
    at_bound = np.abs(np.abs(x) - BOUND) < 0.02 * BOUND
    use = active & (sigma <= SIGMA_MAX) & ~at_bound
    x_est = x.copy()
    x_final = x.copy()
    for _ in range(3):   # re-solve the confident subset; anything that then lands on a bound also falls back
        if np.array_equal(use, active) and np.array_equal(x_final, x):
            break
        if not use.any():
            x_final = np.zeros(NP)
            break
        x2, _, _, _, _ = irls(np.where(use, x_final, 0.0), far, cop, gvalid, e_ref, 2, use)
        x_final = np.where(use, x2, 0.0)
        hit = use & (np.abs(np.abs(x_final) - BOUND) < 0.02 * BOUND)
        if not hit.any():
            break
        use = use & ~hit
        x_final = np.where(use, x_final, 0.0)
    if not use.any():
        x_final = np.zeros(NP)

    # ---- residual summary at final params
    def summary(xv):
        d = {}
        if len(sel_t):
            d['transfer_med_px'] = float(np.median(np.linalg.norm(prob.transfer(xv, sel_t), axis=1)))
        if len(sel_c):
            d['coplanar_med_abs_px'] = float(np.median(np.abs(prob.coplanar(xv, sel_c, np.ones(len(sel_c)), gvalid,
                                                                              e_ref)[0])))
        return d
    notes['residual_default'] = summary(np.zeros(NP))
    notes['residual_final'] = summary(x_final)

    off, tr, s, ex, skew = _x_to_timemodel_parts(x_final, tr0)
    notes.update(
        status='ok',
        estimate=dict(zip(PARAM_NAMES, x_est.tolist())),
        sigma=dict(zip(PARAM_NAMES, sigma.tolist())),
        sigma_formal=dict(zip(PARAM_NAMES, sig_formal.tolist())),
        sigma_jackknife=dict(zip(PARAM_NAMES, sig_jack.tolist())),
        used={k: bool(u) for k, u in zip(PARAM_NAMES, use)},
        final=dict(zip(PARAM_NAMES, x_final.tolist())),
        covariance_formal=cov.tolist(),
        metadata_readout_s=tr0,
        runtime_s=dict(total=time.time() - t_start, decode=t_dec, track=t_trk, nfev=fit.nfev),
    )
    if verbose:
        print({k: notes[k] for k in ('estimate', 'sigma', 'used', 'runtime_s')})
    return TimeModel(offset_s=float(off), skew=float(skew),
                     readout_s=float(tr) if (use[1] or no_readout_meta) else None,
                     focal_scale=float(s), extrinsic_rotvec=np.asarray(ex, dtype=np.float64),
                     notes={'calib': notes})
