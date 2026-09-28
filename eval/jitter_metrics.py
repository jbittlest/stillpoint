"""
Stillpoint jitter metrics -- no-reference measurement of residual camera shake,
micro-jitter, rolling-shutter jello and (optionally, with --ref) cropping /
distortion relative to the unstabilised original.

Usage
-----
    cd stillpoint && .venv/bin/python -m eval.jitter_metrics VIDEO \
        [--start S] [--dur D] [--ref ORIGINAL] [--ref-start S2] \
        --out result.json [--plot result.png]

Everything is measured on downscaled grey frames (default 960 px wide) decoded
by ffmpeg (-hwaccel videotoolbox), streamed one frame at a time.

Units
-----
* "px" = pixels of the analysed video rescaled so that its width is 1920
  (i.e. "1080p-equivalent" for 16:9, 1920x1440 for 4:3).
* rotation in degrees, scale in percent.
* "combined" displacement numbers (hf_jitter_px, jello_px ...) are the RMS,
  over time AND over every point of the frame, of the image displacement
  produced by the high-passed motion parameters.  For a similarity motion
  (tx, ty, rot, log-scale) that is
        sqrt(tx^2 + ty^2 + (rot^2 + logs^2) * (W^2 + H^2) / 12)
  (W, H in 1920-equivalent px; (W^2+H^2)/12 is E|p-c|^2 over the frame).
  For pure translation it is simply the RMS length of the 2-D shake vector,
  so a sinusoid of peak amplitude A on both x and y (any phase) gives A.

Method (see research/metrics.md for the full rationale and validation)
------
1.  Persistent KLT tracks: one best Shi-Tomasi corner per 16x16 cell (texture gated),
    pyramidal LK with forward-backward check (<0.3 px), replenished in empty cells.
2.  Per frame pair, a robust affine in frame-centred coordinates (RANSAC for gross
    outliers, then Huber IRLS over ALL consistent tracks), decomposed "rolling-shutter
    aware":   A = s*R(theta) + [[0, kx], [0, ky]]
    theta and s come from the first column (how the image of a horizontal line moves;
    rows are read out at one instant, so RS does not affect it), kx (horizontal skew) and
    ky (vertical stretch) absorb row-dependent (rolling-shutter) distortion.
    tx, ty = displacement of the frame centre.  A RANSAC similarity and homography are
    also fitted (comparison / Liu-compatibility).
3.  Three-frame DELTA estimator (primary for the high-frequency part): the CHANGE of the
    affine between (t-2->t-1) and (t-1->t) fitted on the same persistent tracks with the
    same weights.  Under parallax (FPV forward flight) the single-pair estimate is a
    different weighted average of depth-dependent flow every frame (tracks enter/leave,
    weights change) -> broadband false "jitter"; each track's own parallax is nearly
    constant over two frames and cancels in the delta.  Velocity = cumsum(delta) with
    its <1 Hz part replaced by the single-pair estimate (reconstruct_velocity).
4.  Camera path = cumulative sum of the inter-frame parameters.  Jitter = zero-phase
    Butterworth (order 4, forward+backward) high-pass of the path at --fc (2 Hz).
    Smooth intentional motion (pans, constant-velocity flight, turns slower than ~1 Hz)
    lies below the cut-off; the first/last --trim s are discarded (filter transients).
5.  Band RMS in 2-8 Hz and 8-Nyquist, 1-s window statistics (median / p90, calm vs
    active), velocity-domain jitter, dominant HF frequency, correlation with LF speed.
6.  Jello: kx/ky and the per-row-band median residual change (after the affine) are
    accumulated and high-passed the same way; "wobble" is the non-linear per-row shift.
7.  Noise floor: tracks are split into two fixed checkerboard halves, each fitted
    independently; noise of the full estimate ~ RMS(HP-path(A) - HP-path(B)) / 2.
    hf_jitter_px_denoised = sqrt(hf^2 - noise^2).
8.  Liu et al. 2013 stability score (FFT energy in bins 1..5 / bins 1..N/2 of the path),
    DIFRINT-metrics.py-compatible (accumulated homographies) and per-component additive.
9.  With --ref ORIGINAL (v2.0):
    * EXACT CROP = source footprint: every --ref-step frames the render is matched to the original (SIFT) and a
      KB4-source -> pinhole(+rolling-shutter) render model is fitted (eval/footprint.py); the render border is
      mapped back into the source -> fraction of the source area it shows (`ref.footprint_area_mean`).  The
      legacy homography crop (`visible_area_frac_*`, ~0.19 too high on fisheye originals) is still reported.
    * frame pairing verified by direct frame matching (`ref_alignment`): the original is warped into the
      render geometry at candidate offsets -6..+6 and scored by band-passed NCC (the old blur-series
      correlation, --alignment blur, gave false -6/-3 lags).
10. Corner wobble (v2.0): per pair, a homography fitted on the central 50%x50% tracks predicts the 4 corner
    regions (outer 20% x 20%); the corners' median residual (three-frame delta version for parallax
    robustness) is integrated, high-passed at fc -> `corner.corner_wobble_px` (lens-model / non-linear RS /
    warp errors that make the periphery swim; a perfect rectilinear rotation-only render gives ~0).
11. Spikes (v2.0): deviation of each pair's velocity [tx, ty, roll*rho, logs*rho] from the median of its
    +-3 neighbours = single-frame position steps/spikes (`spikes`, per-pair series `jump_dev` in the NPZ);
    eval.gate / eval.events count the ones not present in a reference render or the original.
12. Every result records `code` = {eval_version, sha1 of the measurement code}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import signal

EVAL_VERSION = "2.0"   # 2.0: corner wobble, spikes, exact source-footprint crop, frame-matching alignment
_MEASUREMENT_FILES = ("jitter_metrics.py", "footprint.py")


def _code_digest(src: str) -> bytes:
    """Normalised code of a module: the AST without docstrings (comment/docstring edits do not change it)."""
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False).encode()


def code_version() -> dict:
    """Identity of the measurement code (every result JSON records it; eval.gate refuses stale baselines).
    sha1 over the normalised code (AST without docstrings/comments) of the files that define the per-video
    measurement (not gate/compare/report code)."""
    import hashlib
    d = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    per = {}
    for f in _MEASUREMENT_FILES:
        data = _code_digest(open(os.path.join(d, f), encoding="utf-8").read())
        per[f] = hashlib.sha1(data).hexdigest()[:12]
        h.update(data)
    return {"eval_version": EVAL_VERSION, "sha1": h.hexdigest()[:16], "files": per}


FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
REF_WIDTH = 1920.0

cv2.setNumThreads(4)


# --------------------------------------------------------------------------
# video I/O
# --------------------------------------------------------------------------
@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    fps_str: str
    duration: float
    nb_frames: int
    codec: str
    rotation: int


def probe(path: str) -> VideoInfo:
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:"
           "stream_side_data=rotation:format=duration", "-of", "json", path]
    d = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    st = d["streams"][0]
    fr = st.get("avg_frame_rate") or st.get("r_frame_rate")
    if fr in (None, "0/0"):
        fr = st["r_frame_rate"]
    num, den = (int(x) for x in fr.split("/"))
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    dur = float(st.get("duration") or d.get("format", {}).get("duration") or 0)
    w, h = int(st["width"]), int(st["height"])
    if abs(rot) in (90, 270):
        w, h = h, w
    return VideoInfo(path, w, h, num / den, fr, dur, int(st.get("nb_frames") or 0),
                     st.get("codec_name", "?"), rot)


def analysis_size(info: VideoInfo, width: int) -> tuple[int, int]:
    w = int(width) // 2 * 2
    h = int(round(w * info.height / info.width / 2.0)) * 2
    return w, h


class FrameReader:
    """Stream grey frames (uint8 HxW) from ffmpeg, decoded with VideoToolbox."""

    def __init__(self, path, start, dur, w, h, hwaccel=True, select_every=1):
        vf = []
        if select_every > 1:
            vf.append(f"select=not(mod(n\\,{select_every}))")
        vf.append(f"scale={w}:{h}:flags=area")
        vf.append("format=gray")
        cmd = [FFMPEG, "-v", "error", "-nostdin"]
        if hwaccel:
            cmd += ["-hwaccel", "videotoolbox"]
        if start and start > 0:
            cmd += ["-ss", f"{start:.6f}"]
        cmd += ["-i", path]
        if dur and dur > 0:
            cmd += ["-t", f"{dur:.6f}"]
        cmd += ["-an", "-sn", "-dn", "-vf", ",".join(vf), "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", "gray", "-"]
        self.w, self.h = w, h
        self.nbytes = w * h
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=self.nbytes * 4)

    def __iter__(self):
        while True:
            buf = self.proc.stdout.read(self.nbytes)
            if len(buf) < self.nbytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w)
        self.close()

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdout.close()
            except Exception:
                pass
            self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# --------------------------------------------------------------------------
# per-pair motion estimation
# --------------------------------------------------------------------------
LK_WIN = (21, 21)
LK_LEVELS = 4
LK_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
CENTRE_FRAC = 0.5        # corner wobble: homography fitted on the central 50% x 50% of the frame
CORNER_FRAC = 0.2        # corner regions: outer 20% of the width AND of the height (4 boxes 0.2W x 0.2H)
CORNER_MIN_TRACKS = 5


def _lsq_affine(q0, q1):
    """Least-squares 2x3 affine q1 ~ A q0 (q: Nx2)."""
    X = np.column_stack([q0, np.ones(len(q0))])
    sol, *_ = np.linalg.lstsq(X, q1, rcond=None)  # 3x2
    return sol.T  # 2x3


def decompose_rs_affine(A):
    """A (2x3, centred coords) -> tx, ty, theta, log s, kx, ky.

    A[:, :2] = s R(theta) + [[0, kx], [0, ky]]
    """
    a00, a01, tx = A[0]
    a10, a11, ty = A[1]
    s = math.hypot(a00, a10)
    th = math.atan2(a10, a00)
    kx = a01 + s * math.sin(th)
    ky = a11 - s * math.cos(th)
    return tx, ty, th, math.log(max(s, 1e-9)), kx, ky


def decompose_similarity(M):
    s = math.hypot(M[0, 0], M[1, 0])
    return M[0, 2], M[1, 2], math.atan2(M[1, 0], M[0, 0]), math.log(max(s, 1e-9))


class MotionEstimator:
    """Persistent-track KLT motion estimator.

    Tracks live across frames (one best Shi-Tomasi corner per 16x16 cell, replenished in
    empty cells).  For each frame pair it produces
      * a single-pair robust affine (Huber IRLS over all forward/backward-consistent tracks),
      * a three-frame DELTA: the change of the affine between (t-2->t-1) and (t-1->t),
        fitted on the SAME persistent tracks with the SAME weights.  Each track's own
        depth-dependent (parallax) flow is almost constant over two frames and cancels, and
        because the track set is persistent the per-track tracking errors telescope when
        the deltas are summed, so the reconstruction does not random-walk.
    """

    def __init__(self, W, H, cell=16, n_bands=12, ransac_thr=1.5, fb_thr=0.3,
                 min_eig_abs=2e-5, min_eig_rel=0.002, gross_thr=8.0, huber_k=2.0, irls_iters=6,
                 max_per_cell=2, delta_k=4.0, delta_weight="huber", parallax_c=1.0):
        self.W, self.H = W, H
        self.cell = cell
        self.n_bands = n_bands
        self.ransac_thr = ransac_thr
        self.fb_thr = fb_thr
        self.min_eig_abs = min_eig_abs
        self.min_eig_rel = min_eig_rel
        self.gross_thr = gross_thr      # analysis px; tracks further than this from the RANSAC model are dropped
        self.huber_k = huber_k
        self.irls_iters = irls_iters
        self.max_per_cell = max_per_cell
        self.delta_k = delta_k
        self.delta_weight = delta_weight
        self.parallax_c = parallax_c
        self.c = np.array([(W - 1) / 2.0, (H - 1) / 2.0], np.float32)
        self.ny, self.nx = H // cell, W // cell
        gy, gx = np.mgrid[0:self.ny, 0:self.nx]
        self.cell_parity = ((gy + gx) % 2).ravel()
        self.cell_valid = ((gy > 0) & (gy < self.ny - 1) & (gx > 0) & (gx < self.nx - 1)).ravel()
        self.prev_A = None
        # persistent tracks
        self.pos = np.zeros((0, 2), np.float32)       # position in the most recent frame
        self.pos_prev = np.zeros((0, 2), np.float32)  # position one frame earlier (NaN if newborn)
        self.par = np.zeros(0, int)

    # ---------------------------------------------------------------- helpers
    def _cell_index(self, pts):
        cx = np.clip((pts[:, 0] // self.cell).astype(int), 0, self.nx - 1)
        cy = np.clip((pts[:, 1] // self.cell).astype(int), 0, self.ny - 1)
        return cy * self.nx + cx

    def _detect(self, img):
        eig = cv2.cornerMinEigenVal(img, 7, 3)
        c = self.cell
        e = eig[: self.ny * c, : self.nx * c].reshape(self.ny, c, self.nx, c)
        e = e.transpose(0, 2, 1, 3).reshape(self.ny * self.nx, c * c)
        idx = e.argmax(1)
        emax = e[np.arange(len(idx)), idx]
        gy, gx = np.divmod(np.arange(len(idx)), self.nx)
        pts = np.column_stack([gx * c + idx % c, gy * c + idx // c]).astype(np.float32)
        thr = max(self.min_eig_abs, self.min_eig_rel * float(emax.max()))
        ok = (emax > thr) & self.cell_valid
        return pts, ok

    def _replenish(self, img):
        pts, ok = self._detect(img)
        occ = np.bincount(self._cell_index(self.pos), minlength=self.ny * self.nx) if len(self.pos) else np.zeros(self.ny * self.nx, int)
        # thin over-crowded cells (keep the oldest = first occurrences)
        if len(self.pos):
            ci = self._cell_index(self.pos)
            order = np.argsort(ci, kind="stable")
            rank = np.empty(len(ci), int)
            _, first = np.unique(ci[order], return_index=True)
            counts = np.diff(np.append(first, len(ci)))
            rank[order] = np.arange(len(ci)) - np.repeat(first, counts)
            keep = rank < self.max_per_cell
            self.pos, self.pos_prev, self.par = self.pos[keep], self.pos_prev[keep], self.par[keep]
        new = ok & (occ == 0)
        if new.any():
            self.pos = np.vstack([self.pos, pts[new]]).astype(np.float32)
            self.pos_prev = np.vstack([self.pos_prev, np.full((int(new.sum()), 2), np.nan, np.float32)])
            self.par = np.concatenate([self.par, self.cell_parity[new]])

    def _track(self, prev, cur, pts, A_pred=None):
        p0 = pts.reshape(-1, 1, 2).astype(np.float32)
        flags, p1_init = 0, None
        if A_pred is not None:
            q = pts - self.c
            p1_init = (q @ A_pred[:, :2].T + A_pred[:, 2] + self.c).astype(np.float32).reshape(-1, 1, 2)
            flags = cv2.OPTFLOW_USE_INITIAL_FLOW
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, p1_init, winSize=LK_WIN,
                                              maxLevel=LK_LEVELS, criteria=LK_CRIT, flags=flags)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, p0.copy(), winSize=LK_WIN,
                                               maxLevel=LK_LEVELS, criteria=LK_CRIT,
                                               flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        p1 = p1.reshape(-1, 2)
        fb = np.linalg.norm(p0r.reshape(-1, 2) - pts, axis=1)
        inside = (p1[:, 0] >= 0) & (p1[:, 0] <= self.W - 1) & (p1[:, 1] >= 0) & (p1[:, 1] <= self.H - 1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_thr) & inside
        return p1, good

    def _robust_fit(self, q0, q1, A0):
        r = q1 - (q0 @ A0[:, :2].T + A0[:, 2])
        m = np.hypot(r[:, 0], r[:, 1]) < self.gross_thr
        if m.sum() < 10:
            return None
        Q0, Q1 = q0[m], q1[m]
        X = np.column_stack([Q0, np.ones(len(Q0))])
        w = np.ones(len(Q0))
        A = A0
        for _ in range(self.irls_iters):
            sw = np.sqrt(w)[:, None]
            sol, *_ = np.linalg.lstsq(X * sw, Q1 * sw, rcond=None)
            A = sol.T
            d = np.hypot(*(Q1 - X @ sol).T)
            c = self.huber_k * float(np.median(d)) + 0.02
            w = np.minimum(1.0, c / np.maximum(d, 1e-9))
        return A, m, w

    def _bands(self, rows, res):
        band = np.clip(((rows + self.c[1]) / self.H * self.n_bands).astype(int), 0, self.n_bands - 1)
        rx = np.full(self.n_bands, np.nan)
        ry = np.full(self.n_bands, np.nan)
        for b in range(self.n_bands):
            mm = band == b
            if mm.sum() >= 6:
                rx[b] = np.median(res[mm, 0])
                ry[b] = np.median(res[mm, 1])
        return rx, ry

    def _centre_mask(self, q):
        return (np.abs(q[:, 0]) < self.W * CENTRE_FRAC / 2) & (np.abs(q[:, 1]) < self.H * CENTRE_FRAC / 2)

    def _corner_ids(self, q):
        """-> corner id per point (0 TL, 1 TR, 2 BL, 3 BR) or -1 (not in a corner region)."""
        cx = np.abs(q[:, 0]) > self.W * (0.5 - CORNER_FRAC)
        cy = np.abs(q[:, 1]) > self.H * (0.5 - CORNER_FRAC)
        cid = (q[:, 0] > 0).astype(int) + 2 * (q[:, 1] > 0).astype(int)
        return np.where(cx & cy, cid, -1)

    def _corner_medians(self, qpos, res, par=None):
        """-> (rx, ry) per corner (4,), or with par: (4, 3) columns [all tracks, parity-a half, parity-b half]
        (the halves give the split-half noise floor of the corner metric)."""
        cid = self._corner_ids(qpos)
        if par is None:
            rx, ry = np.full(4, np.nan), np.full(4, np.nan)
            for c in range(4):
                mm = cid == c
                if mm.sum() >= CORNER_MIN_TRACKS:
                    rx[c], ry[c] = np.median(res[mm, 0]), np.median(res[mm, 1])
            return rx, ry
        rx, ry = np.full((4, 3), np.nan), np.full((4, 3), np.nan)
        for c in range(4):
            mm = cid == c
            if mm.sum() >= CORNER_MIN_TRACKS:
                rx[c, 0], ry[c, 0] = np.median(res[mm, 0]), np.median(res[mm, 1])
                for j, pv in ((1, 0), (2, 1)):
                    mh = mm & (par == pv)
                    if mh.sum() >= 3:
                        rx[c, j], ry[c, j] = np.median(res[mh, 0]), np.median(res[mh, 1])
        return rx, ry

    def _corner_resid(self, q0, q1, par=None):
        """Corner-vs-centre residual of one pair: homography fitted to the CENTRE tracks only, median residual
        of the tracks in each corner region (analysis px).  A rectilinear frame under pure rotation moves by a
        homography, so this is the non-rigid (lens-model / non-linear RS / warp) motion of the corners."""
        nan4 = np.full(4, np.nan) if par is None else np.full((4, 3), np.nan)
        cm = self._centre_mask(q0)
        if cm.sum() < 20:
            return nan4, nan4.copy()
        Hc, inl = cv2.findHomography(q0[cm].astype(np.float32), q1[cm].astype(np.float32), cv2.RANSAC, 1.0,
                                     maxIters=500, confidence=0.99)
        if Hc is None:
            return nan4, nan4.copy()
        pred = cv2.perspectiveTransform(q0.reshape(-1, 1, 2).astype(np.float64), Hc).reshape(-1, 2)
        return self._corner_medians(q0, q1 - pred, par)

    def _corner_delta(self, q0, q1, q2, par=None):
        """Three-frame version: change of each corner track's residual between (t-2->t-1) and (t-1->t), with both
        centre homographies fitted by least squares on the SAME centre tracks (common RANSAC inliers), so track
        births/deaths cancel (same idea as the jitter delta estimator)."""
        nan4 = np.full(4, np.nan) if par is None else np.full((4, 3), np.nan)
        cm = self._centre_mask(q1)
        if cm.sum() < 20:
            return nan4, nan4.copy()
        a0, a1, a2 = (x[cm].astype(np.float32) for x in (q0, q1, q2))
        Ha, ia = cv2.findHomography(a0, a1, cv2.RANSAC, 1.0, maxIters=500, confidence=0.99)
        Hb, ib = cv2.findHomography(a1, a2, cv2.RANSAC, 1.0, maxIters=500, confidence=0.99)
        if Ha is None or Hb is None:
            return nan4, nan4.copy()
        both = (ia.ravel() > 0) & (ib.ravel() > 0)
        if both.sum() < 12:
            return nan4, nan4.copy()
        Ha, _ = cv2.findHomography(a0[both], a1[both], 0)
        Hb, _ = cv2.findHomography(a1[both], a2[both], 0)
        if Ha is None or Hb is None:
            return nan4, nan4.copy()
        ra = q1 - cv2.perspectiveTransform(q0.reshape(-1, 1, 2).astype(np.float64), Ha).reshape(-1, 2)
        rb = q2 - cv2.perspectiveTransform(q1.reshape(-1, 1, 2).astype(np.float64), Hb).reshape(-1, 2)
        return self._corner_medians(q1, rb - ra, par)

    # ---------------------------------------------------------------- main step
    def estimate(self, prev_img, cur_img, _unused=None):
        out = {"ok": False, "d_ok": False}
        if len(self.pos) < 12:
            self.pos = np.zeros((0, 2), np.float32)
            self.pos_prev = np.zeros((0, 2), np.float32)
            self.par = np.zeros(0, int)
            self._replenish(prev_img)
        out["n_points"] = int(len(self.pos))
        p1, good = self._track(prev_img, cur_img, self.pos, self.prev_A)
        if good.sum() < 12 and self.prev_A is not None:
            p1, good = self._track(prev_img, cur_img, self.pos, None)
        out["n_tracked"] = int(good.sum())
        p_prev, p_cur, p_pp, par = self.pos[good], p1[good], self.pos_prev[good], self.par[good]
        # advance the persistent tracks regardless of fit success
        self.pos, self.pos_prev, self.par = p_cur.astype(np.float32), p_prev.astype(np.float32), par
        try:
            if good.sum() >= 12:
                self._pair(out, p_prev, p_cur, par)
                self._delta(out, p_pp, p_prev, p_cur, par)
        finally:
            self._replenish(cur_img)
        if not out["ok"]:
            self.prev_A = None
        return out

    def _pair(self, out, p_prev, p_cur, par):
        q0 = p_prev - self.c
        q1 = p_cur - self.c
        A0, inl = cv2.estimateAffine2D(q0, q1, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thr,
                                       maxIters=2000, confidence=0.999, refineIters=10)
        if A0 is None or inl is None or inl.sum() < 10:
            return
        inl = inl.ravel().astype(bool)
        Ar = _lsq_affine(q0[inl], q1[inl])
        h = decompose_rs_affine(Ar)
        out["ran_tx"], out["ran_ty"], out["ran_rot"], out["ran_logs"] = h[:4]
        fit = self._robust_fit(q0, q1, A0)
        if fit is None:
            return
        A, gm, w = fit
        res = q1 - (q0 @ A[:, :2].T + A[:, 2])
        tx, ty, th, ls, kx, ky = decompose_rs_affine(A)
        out.update(ok=True, n_inliers=int(inl.sum()), tx=tx, ty=ty, rot=th, logs=ls, kx=kx, ky=ky)
        r_in = res[gm]
        neff = float(w.sum() ** 2 / (w ** 2).sum())
        out["se_tx"] = float(np.sqrt(np.average(r_in[:, 0] ** 2, weights=w)) / math.sqrt(neff))
        out["se_ty"] = float(np.sqrt(np.average(r_in[:, 1] ** 2, weights=w)) / math.sqrt(neff))
        out["res_rms"] = float(np.sqrt((res[inl] ** 2).sum(1).mean()))
        out["res_med_all"] = float(np.median(np.hypot(r_in[:, 0], r_in[:, 1])))
        for tag, pv in (("a", 0), ("b", 1)):
            m = par == pv
            fh = self._robust_fit(q0[m], q1[m], A0) if m.sum() >= 12 else None
            if fh is not None:
                hh = decompose_rs_affine(fh[0])
                out[f"tx_{tag}"], out[f"ty_{tag}"], out[f"rot_{tag}"], out[f"logs_{tag}"] = hh[:4]
            else:
                out[f"tx_{tag}"] = out[f"ty_{tag}"] = out[f"rot_{tag}"] = out[f"logs_{tag}"] = float("nan")
        M, _ = cv2.estimateAffinePartial2D(q0, q1, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thr,
                                           maxIters=2000, confidence=0.999, refineIters=10)
        if M is not None:
            out["sim_tx"], out["sim_ty"], out["sim_rot"], out["sim_logs"] = decompose_similarity(M)
        Hm, _ = cv2.findHomography(q0, q1, cv2.RANSAC, self.ransac_thr, maxIters=2000, confidence=0.999)
        out["H"] = (Hm / Hm[2, 2]).ravel().tolist() if Hm is not None else [float("nan")] * 9
        out["band_rx"], out["band_ry"] = self._bands(q0[gm][:, 1], res[gm])
        out["cor_rx"], out["cor_ry"] = self._corner_resid(q0[gm], q1[gm], par[gm])
        self.prev_A = A

    def _delta(self, out, p_pp, p_prev, p_cur, par):
        have = np.isfinite(p_pp).all(1)
        if have.sum() < 12:
            return
        q0 = p_pp[have] - self.c
        q1 = p_prev[have] - self.c
        q2 = p_cur[have] - self.c
        P = par[have]
        Aa0, _ = cv2.estimateAffine2D(q0, q1, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thr,
                                      maxIters=1000, confidence=0.995, refineIters=5)
        Ab0, _ = cv2.estimateAffine2D(q1, q2, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thr,
                                      maxIters=1000, confidence=0.995, refineIters=5)
        if Aa0 is None or Ab0 is None:
            return
        ra = q1 - (q0 @ Aa0[:, :2].T + Aa0[:, 2])
        rb = q2 - (q1 @ Ab0[:, :2].T + Ab0[:, 2])
        m = (np.hypot(*(rb - ra).T) < self.gross_thr / 2) & (np.hypot(*ra.T) < 4 * self.gross_thr) \
            & (np.hypot(*rb.T) < 4 * self.gross_thr)
        if m.sum() < 12:
            return
        Q0, Q1, Q2, P = q0[m], q1[m], q2[m], P[m]
        X0 = np.column_stack([Q0, np.ones(len(Q0))])
        X1 = np.column_stack([Q1, np.ones(len(Q1))])
        w = np.ones(len(Q0))
        wpar = np.ones(len(Q0))
        for it in range(self.irls_iters):
            sw = np.sqrt(w * wpar)[:, None]
            sa, *_ = np.linalg.lstsq(X0 * sw, Q1 * sw, rcond=None)
            sb, *_ = np.linalg.lstsq(X1 * sw, Q2 * sw, rcond=None)
            Ra, Rb = Q1 - X0 @ sa, Q2 - X1 @ sb
            d = np.hypot(*(Rb - Ra).T)
            c = self.delta_k * float(np.median(d)) + 0.02
            if self.delta_weight == "cauchy":
                w = 1.0 / (1.0 + (d / c) ** 2)
            else:
                w = np.minimum(1.0, c / np.maximum(d, 1e-9))
            if self.parallax_c > 0 and it == 0:
                # optional: de-emphasise tracks with large parallax residual (near field)
                rm = 0.5 * (np.hypot(*Ra.T) + np.hypot(*Rb.T))
                wpar = 1.0 / (1.0 + (rm / self.parallax_c) ** 2)
        pa, pb = decompose_rs_affine(sa.T), decompose_rs_affine(sb.T)
        names = ("tx", "ty", "rot", "logs", "kx", "ky")
        out.update({f"d_{k}": pb[i] - pa[i] for i, k in enumerate(names)})
        out["d_ok"] = True
        out["n_delta"] = int(m.sum())
        for tag, pv in (("a", 0), ("b", 1)):
            mm = P == pv
            if mm.sum() >= 8:
                sw = np.sqrt(w[mm])[:, None]
                ha, *_ = np.linalg.lstsq(X0[mm] * sw, Q1[mm] * sw, rcond=None)
                hb, *_ = np.linalg.lstsq(X1[mm] * sw, Q2[mm] * sw, rcond=None)
                da, db = decompose_rs_affine(ha.T), decompose_rs_affine(hb.T)
                for i, k in enumerate(names[:4]):
                    out[f"d_{k}_{tag}"] = db[i] - da[i]
            else:
                for k in names[:4]:
                    out[f"d_{k}_{tag}"] = float("nan")
        out["d_band_rx"], out["d_band_ry"] = self._bands(Q1[:, 1], Rb - Ra)
        out["d_cor_rx"], out["d_cor_ry"] = self._corner_delta(Q0, Q1, Q2, P)


def build_pyr(img):
    # OpenCV 5 python bindings no longer accept pre-built pyramids; LK builds them.
    return img


# --------------------------------------------------------------------------
# pass 1: run the estimator over the clip
# --------------------------------------------------------------------------
PAIR_KEYS = ["tx", "ty", "rot", "logs", "kx", "ky", "se_tx", "se_ty", "res_rms", "res_med_all",
             "ran_tx", "ran_ty", "ran_rot", "ran_logs",
             "tx_a", "ty_a", "rot_a", "logs_a", "tx_b", "ty_b", "rot_b", "logs_b",
             "sim_tx", "sim_ty", "sim_rot", "sim_logs", "n_points", "n_tracked", "n_inliers",
             "d_tx", "d_ty", "d_rot", "d_logs", "d_kx", "d_ky", "n_delta",
             "d_tx_a", "d_ty_a", "d_rot_a", "d_logs_a", "d_tx_b", "d_ty_b", "d_rot_b", "d_logs_b"]


def extract_motion(path, start, dur, width=960, hwaccel=True, verbose=True, max_frames=None, est_kw=None,
                   frame_sink=None):
    """Motion pass over the window.  frame_sink(i, img), if given, is called with every decoded analysis frame
    (window index i) -- used to hand frames to the reference pass without decoding the video twice."""
    info = probe(path)
    W, H = analysis_size(info, width)
    est = MotionEstimator(W, H, **(est_kw or {}))
    reader = FrameReader(path, start, dur, W, H, hwaccel=hwaccel)
    sig = {k: [] for k in PAIR_KEYS}
    sig["H"], sig["band_rx"], sig["band_ry"], sig["d_band_rx"], sig["d_band_ry"] = [], [], [], [], []
    for kk in ("cor_rx", "cor_ry", "d_cor_rx", "d_cor_ry"):
        sig[kk] = []
    frame_stats = {"luma": [], "sharp": [], "black": []}
    prev = prev_pyr = None
    n = 0
    t0 = time.time()
    for img in reader:
        if frame_sink is not None:
            frame_sink(n, img)
        # photometric stats on every frame
        frame_stats["luma"].append(float(img.mean()))
        cy0, cy1, cx0, cx1 = H // 8, H - H // 8, W // 8, W - W // 8
        frame_stats["sharp"].append(float(cv2.Laplacian(img[cy0:cy1, cx0:cx1], cv2.CV_32F).var()))
        frame_stats["black"].append(float((img < 4).mean()))
        pyr = build_pyr(img)
        if prev is not None:
            r = est.estimate(prev_pyr, pyr, prev)
            for k in PAIR_KEYS:
                okk = r.get("d_ok") if k.startswith("d_") else r.get("ok")
                sig[k].append(r.get(k, float("nan")) if okk or k.startswith("n_") else float("nan"))
            sig["H"].append(r.get("H", [float("nan")] * 9) if r.get("ok") else [float("nan")] * 9)
            nb = est.n_bands
            sig["band_rx"].append(r.get("band_rx", np.full(nb, np.nan)) if r.get("ok") else np.full(nb, np.nan))
            sig["band_ry"].append(r.get("band_ry", np.full(nb, np.nan)) if r.get("ok") else np.full(nb, np.nan))
            sig["d_band_rx"].append(r.get("d_band_rx", np.full(nb, np.nan)) if r.get("d_ok") else np.full(nb, np.nan))
            sig["d_band_ry"].append(r.get("d_band_ry", np.full(nb, np.nan)) if r.get("d_ok") else np.full(nb, np.nan))
            for kk in ("cor_rx", "cor_ry"):     # (4 corners, [all, half a, half b])
                sig[kk].append(r.get(kk, np.full((4, 3), np.nan)) if r.get("ok") else np.full((4, 3), np.nan))
            for kk in ("d_cor_rx", "d_cor_ry"):
                sig[kk].append(r.get(kk, np.full((4, 3), np.nan)) if r.get("d_ok") else np.full((4, 3), np.nan))
        prev, prev_pyr = img, pyr
        n += 1
        if verbose and n % 300 == 0:
            print(f"  [{os.path.basename(path)}] {n} frames, {n / (time.time() - t0):.1f} fps", file=sys.stderr)
        if max_frames and n >= max_frames:
            reader.close()
            break
    out = {k: np.asarray(v, dtype=float) for k, v in sig.items()}
    for k, v in frame_stats.items():
        out["frame_" + k] = np.asarray(v, float)
    meta = dict(info=asdict(info), analysis_w=W, analysis_h=H, n_frames=n,
                decode_fps=n / max(time.time() - t0, 1e-9), start=start, dur=dur,
                estimator={k: v for k, v in vars(est).items() if isinstance(v, (int, float, str)) and not k.startswith("_")})
    return out, meta


# --------------------------------------------------------------------------
# pass 2: signal processing / metrics
# --------------------------------------------------------------------------
def _fill_nan(x):
    x = np.asarray(x, float).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x), int(bad.sum())
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x, int(bad.sum())


def path_from_velocity(v):
    v, _ = _fill_nan(v)
    return np.concatenate([[0.0], np.cumsum(v)])


class Filters:
    def __init__(self, fs, fc=2.0, bands=((2.0, 8.0), (8.0, None)), order=4):
        self.fs, self.fc, self.order = fs, fc, order
        nyq = fs / 2
        self.nyq = nyq
        self.hp = signal.butter(order, fc, "highpass", fs=fs, output="sos")
        self.lp = signal.butter(order, fc, "lowpass", fs=fs, output="sos")
        self.lp1 = signal.butter(order, 1.0, "lowpass", fs=fs, output="sos")
        self.band_sos = []
        for lo, hi in bands:
            if hi is None or hi >= 0.95 * nyq:
                self.band_sos.append(((lo, nyq), signal.butter(order, lo, "highpass", fs=fs, output="sos")))
            else:
                self.band_sos.append(((lo, hi), signal.butter(order, [lo, hi], "bandpass", fs=fs, output="sos")))

    @staticmethod
    def _apply(sos, x):
        x = np.asarray(x, float)
        padlen = min(len(x) - 1, int(3 * 60))
        return signal.sosfiltfilt(sos, x, axis=0, padtype="odd", padlen=padlen)

    def highpass(self, x):
        return self._apply(self.hp, x)

    def lowpass(self, x, one_hz=False):
        return self._apply(self.lp1 if one_hz else self.lp, x)

    def band(self, i, x):
        return self._apply(self.band_sos[i][1], x)


def _rms(x, axis=None):
    x = np.asarray(x, float)
    return float(np.sqrt(np.nanmean(x ** 2, axis=axis))) if axis is None else np.sqrt(np.nanmean(x ** 2, axis=axis))


def liu_stability_additive(paths, n_low=5):
    """Liu et al. 2013 stability: energy in FFT bins 1..n_low over bins 1..N/2."""
    out = {}
    for k, p in paths.items():
        p = np.asarray(p, float)
        P = np.abs(np.fft.fft(p)) ** 2
        P = P[1: len(P) // 2]
        out[k] = float(P[:n_low].sum() / max(P.sum(), 1e-30))
    return out


def liu_stability_difrint(Hs, W, H, n_low=5):
    """DIFRINT-metrics.py-compatible: accumulate inter-frame homographies in
    top-left-origin pixel coords (Pt = Pt @ M), translation magnitude and rotation
    angle of the accumulated matrix, FFT power, drop DC, first half, bins 0..4."""
    c = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
    T = np.array([[1, 0, c[0]], [0, 1, c[1]], [0, 0, 1.0]])
    Ti = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1.0]])
    Pt = np.eye(3)
    ts, rs = [], []
    for h in Hs:
        M = np.asarray(h, float).reshape(3, 3)
        if not np.all(np.isfinite(M)):
            M = np.eye(3)
        M = T @ M @ Ti
        Pt = Pt @ M
        ts.append(math.hypot(Pt[0, 2], Pt[1, 2]))
        rs.append(math.degrees(math.atan2(Pt[1, 0], Pt[0, 0])))
    res = {}
    for k, s in (("trans", ts), ("rot", rs)):
        P = np.abs(np.fft.fft(np.asarray(s))) ** 2
        P = np.delete(P, 0)
        P = P[: len(P) // 2]
        res[k] = float(P[:n_low].sum() / max(P.sum(), 1e-30))
    res["avg"] = (res["trans"] + res["rot"]) / 2
    res["min"] = min(res["trans"], res["rot"])
    return res


def reconstruct_velocity(v_pair, d, F):
    """Combine single-pair velocities (good at low frequency) with three-frame velocity
    CHANGES d[j] = v[j] - v[j-1] (immune to point-set switching under parallax):
    v = cumsum(d), with its drift below 1 Hz replaced by the single-pair estimate."""
    vp, _ = _fill_nan(v_pair)
    d = np.asarray(d, float).copy()
    dp = np.diff(vp, prepend=vp[0])
    bad = ~np.isfinite(d)
    d[bad] = dp[bad]
    d[0] = 0.0
    v_rec = vp[0] + np.cumsum(d)
    return v_rec + F.lowpass(vp - v_rec, one_hz=True)


SPIKE_HALF = 3            # neighbours on each side for the local median velocity


def spike_deviation(v, half=SPIKE_HALF):
    """v: (n_pairs, d) per-pair velocity -> deviation from the nan-median of the +-half neighbours (self
    excluded).  An isolated one-pair impulse (a position step) shows at full size; smooth motion ~0."""
    v = np.asarray(v, float)
    n = len(v)
    pad = np.full((half, v.shape[1]), np.nan)
    vp = np.vstack([pad, v, pad])
    neigh = [vp[half + o: half + o + n] for o in range(-half, half + 1) if o != 0]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.stack(neigh, 0), axis=0)
    return v - med


SPIKE_Z = 3.0            # an event must stand out from the local deviation level (vibration) by this factor
SPIKE_LOCAL = 30         # +- pairs for the local deviation level (pairs within +-2 of the event excluded)


def local_scale(dev, k, half=SPIKE_LOCAL, excl=2):
    """Typical deviation magnitude around pair k (median |dev| over +-half pairs, +-excl excluded)."""
    mag = np.linalg.norm(np.nan_to_num(np.asarray(dev, float)), axis=1) if np.ndim(dev) == 2 else np.abs(dev)
    lo, hi = max(k - half, 0), min(k + half + 1, len(mag))
    idx = np.r_[lo:max(k - excl, lo), min(k + excl + 1, hi):hi]
    return float(np.median(mag[idx])) if len(idx) else 0.0


def spike_events(dev, thr, merge=2, z=SPIKE_Z):
    """Isolated single-frame events: pairs with |dev| > thr AND |dev| >= z * local deviation level (so ongoing
    vibration is not counted), merged into events (gaps <= merge pairs).  -> list of (pair_index, magnitude)."""
    dev = np.asarray(dev, float)
    mag = np.linalg.norm(np.nan_to_num(dev), axis=1)
    idx = [i for i in np.where(mag > thr)[0] if z <= 0 or mag[i] >= z * local_scale(dev, i)]
    ev = []
    for i in idx:
        if ev and i - ev[-1][2] <= merge:
            if mag[i] > ev[-1][1]:
                ev[-1][0], ev[-1][1] = int(i), float(mag[i])
            ev[-1][2] = int(i)
        else:
            ev.append([int(i), float(mag[i]), int(i)])
    return [(e[0], e[1]) for e in ev]


def spike_summary(dev, fs, start=0.0, top=10, trim=SPIKE_HALF):
    mag = np.linalg.norm(np.nan_to_num(np.asarray(dev, float)), axis=1)
    n = len(mag)
    keep = lambda ev: [e for e in ev if trim <= e[0] < n - trim]       # edge pairs lack neighbours
    e05, e1 = keep(spike_events(dev, 0.5)), keep(spike_events(dev, 1.0))
    order = sorted(e05, key=lambda e: -e[1])[:top]
    return {"n_gt_0.5px": len(e05), "n_gt_1px": len(e1), "max_px": float(mag.max()) if len(mag) else float("nan"),
            "p99_px": float(np.percentile(mag, 99)) if len(mag) else float("nan"),
            "top": [{"pair": int(i), "frame_after": int(i) + 1, "t_s": float(start + (i + 1) / fs), "px": round(float(a), 3)}
                    for i, a in order],
            "note": "self-only spikes; Stillpoint-only jumps (not present in the reference render or the original) "
                    "are computed by eval.gate / eval.events from the signal NPZs"}


def compute_metrics(sig, meta, fc=2.0, trim_s=0.5, focal_px=None, window_s=1.0, method="delta"):
    fs = meta["info"]["fps"]
    W, H = meta["analysis_w"], meta["analysis_h"]
    k = REF_WIDTH / W                       # analysis px -> 1920-equivalent px
    Wr, Hr = REF_WIDTH, REF_WIDTH * H / W
    R2 = (Wr ** 2 + Hr ** 2) / 12.0          # E|p-c|^2 over the frame (1920-eq px^2)
    F = Filters(fs, fc)
    n_pairs = len(sig["tx"])
    N = n_pairs + 1
    trim = int(round(trim_s * fs))
    sl = slice(trim, N - trim) if N > 2 * trim + 10 else slice(0, N)

    # velocities in output units: px (1920-eq) / frame, rad / frame
    use_delta = (method == "delta" and "d_tx" in sig and np.isfinite(sig["d_tx"]).mean() > 0.5)

    def V(name, suffix=""):
        sc = k if name in ("tx", "ty") else 1.0
        vp = np.asarray(sig[name + suffix], float) * sc
        if use_delta:
            return reconstruct_velocity(vp, np.asarray(sig["d_" + name + suffix], float) * sc, F)
        return _fill_nan(vp)[0]

    vel = {kk: V(kk) for kk in ("tx", "ty", "rot", "logs")}
    vel_pair = {"tx": sig["tx"] * k, "ty": sig["ty"] * k, "rot": sig["rot"], "logs": sig["logs"]}
    n_fail = int((~np.isfinite(sig["tx"])).sum())
    paths = {kk: path_from_velocity(v) for kk, v in vel.items()}
    hp = {kk: F.highpass(p) for kk, p in paths.items()}

    def combined(d, s=sl):
        e = d["tx"][s] ** 2 + d["ty"][s] ** 2 + (d["rot"][s] ** 2 + d["logs"][s] ** 2) * R2
        return float(np.sqrt(np.mean(e))), e

    m = {"estimator": "three-frame-delta" if use_delta else "single-pair"}
    m["hf_jitter_px"], e_hf = combined(hp)
    hpp = {kk: F.highpass(path_from_velocity(v)) for kk, v in vel_pair.items()}
    m["hf_jitter_px_pair_method"], _ = combined(hpp)
    if use_delta:
        m["delta_valid_frac"] = float(np.isfinite(sig["d_tx"]).mean())
    m["hf_trans_px"] = float(np.sqrt(np.mean(hp["tx"][sl] ** 2 + hp["ty"][sl] ** 2)))
    m["hf_tx_px"] = _rms(hp["tx"][sl])
    m["hf_ty_px"] = _rms(hp["ty"][sl])
    m["hf_rot_deg"] = math.degrees(_rms(hp["rot"][sl]))
    m["hf_rot_equiv_px"] = _rms(hp["rot"][sl]) * math.sqrt(R2)
    m["hf_scale_pct"] = 100 * _rms(hp["logs"][sl])
    m["hf_scale_equiv_px"] = _rms(hp["logs"][sl]) * math.sqrt(R2)
    if focal_px:
        m["hf_trans_arcmin"] = math.degrees(math.atan(m["hf_trans_px"] / focal_px)) * 60
        m["hf_tx_arcmin"] = math.degrees(math.atan(m["hf_tx_px"] / focal_px)) * 60
        m["hf_ty_arcmin"] = math.degrees(math.atan(m["hf_ty_px"] / focal_px)) * 60
    m["hf_rot_arcmin"] = m["hf_rot_deg"] * 60

    # bands
    bands = {}
    for i, ((lo, hi), _) in enumerate(F.band_sos):
        b = {kk: F.band(i, p) for kk, p in paths.items()}
        tot, _ = combined(b)
        name = f"{lo:g}-{hi:g}Hz"
        bands[name] = {"combined_px": tot, "tx_px": _rms(b["tx"][sl]), "ty_px": _rms(b["ty"][sl]),
                       "rot_deg": math.degrees(_rms(b["rot"][sl])), "scale_pct": 100 * _rms(b["logs"][sl])}
    tot_e = sum(v["combined_px"] ** 2 for v in bands.values())
    for v in bands.values():
        v["energy_frac"] = v["combined_px"] ** 2 / tot_e if tot_e > 0 else float("nan")
    m["bands"] = bands

    # velocity-domain jitter (emphasises higher frequencies), px/frame
    hpv = {kk: F.highpass(_fill_nan(v)[0]) for kk, v in vel.items()}
    sv = slice(sl.start, (sl.stop or N) - 1)
    m["hf_vel_px_per_frame"], _ = combined(hpv, sv)

    # windowed stats (robust to a few violent intentional manoeuvres)
    wl = max(int(round(window_s * fs)), 4)
    idx = np.arange(sl.start, (sl.stop or N))
    nw = len(idx) // wl
    win_rms = np.array([math.sqrt(np.mean(e_hf[i * wl:(i + 1) * wl])) for i in range(nw)]) if nw else np.array([])
    # low-frequency (<1 Hz) speed of the camera path, px/s (translation) and deg/s
    lpv_t = np.hypot(F.lowpass(_fill_nan(vel["tx"])[0], True), F.lowpass(_fill_nan(vel["ty"])[0], True)) * fs
    lpv_r = np.abs(F.lowpass(_fill_nan(vel["rot"])[0], True)) * fs * 180 / math.pi
    lp_speed_eq = np.sqrt(lpv_t ** 2 + (np.radians(lpv_r) ** 2) * R2)   # combined px/s
    wspeed = np.array([lp_speed_eq[max(idx[0] + i * wl - 1, 0): idx[0] + (i + 1) * wl - 1].mean() for i in range(nw)]) if nw else np.array([])
    if nw >= 2:
        m["window"] = {
            "window_s": window_s, "n": int(nw),
            "median_px": float(np.median(win_rms)), "p90_px": float(np.percentile(win_rms, 90)),
            "max_px": float(win_rms.max()), "min_px": float(win_rms.min()),
        }
        if nw >= 4 and np.std(wspeed) > 0 and np.std(win_rms) > 0:
            m["window"]["corr_jitter_vs_lf_speed"] = float(np.corrcoef(win_rms, wspeed)[0, 1])
            q50, q75 = np.percentile(wspeed, [50, 75])
            m["window"]["median_px_calm"] = float(np.median(win_rms[wspeed <= q50]))
            m["window"]["median_px_active"] = float(np.median(win_rms[wspeed >= q75]))
        m["window"]["rms_px"] = win_rms.tolist()
        m["window"]["lf_speed_px_s"] = wspeed.tolist()
    m["lf_speed_px_s_median"] = float(np.median(lp_speed_eq))
    m["lf_rot_speed_deg_s_median"] = float(np.median(lpv_r))

    # PSD of path (0.5 Hz high-passed to avoid leakage from intentional motion)
    sos05 = signal.butter(4, 0.5, "highpass", fs=fs, output="sos")
    nper = int(min(512, 2 ** int(math.log2(max(len(paths["tx"]) // 2, 16)))))
    psd = {}
    for kk in ("tx", "ty", "rot"):
        x = Filters._apply(sos05, paths[kk])[sl]
        f, P = signal.welch(x, fs=fs, nperseg=min(nper, len(x)), detrend="constant")
        psd[kk] = P
    m["psd"] = {"f": f.tolist(), "tx": psd["tx"].tolist(), "ty": psd["ty"].tolist(),
                "rot_rad": psd["rot"].tolist(), "units": "px^2/Hz (1920-eq), rad^2/Hz"}
    hfm = f >= fc
    m["hf_peak_hz"] = {kk: float(f[hfm][np.argmax(psd[kk][hfm])]) for kk in psd}

    # noise floor via split-half fits
    va = {kk: V(kk, "_a") for kk in ("tx", "ty", "rot", "logs")}
    vb = {kk: V(kk, "_b") for kk in ("tx", "ty", "rot", "logs")}
    dd = {kk: F.highpass(path_from_velocity(va[kk] - vb[kk])) for kk in va}
    nf, _ = combined(dd)
    m["noise_floor_px"] = nf / 2.0
    m["noise_floor_trans_px"] = float(np.sqrt(np.mean(dd["tx"][sl] ** 2 + dd["ty"][sl] ** 2))) / 2.0
    m["noise_floor_rot_deg"] = math.degrees(_rms(dd["rot"][sl])) / 2.0
    m["hf_jitter_px_denoised"] = float(math.sqrt(max(m["hf_jitter_px"] ** 2 - m["noise_floor_px"] ** 2, 0.0)))

    # jello / intra-frame distortion
    kx_p = path_from_velocity(V("kx"))
    ky_p = path_from_velocity(V("ky"))
    sk = F.highpass(kx_p) * Hr / math.sqrt(12)
    st = F.highpass(ky_p) * Hr / math.sqrt(12)
    jel = {"skew_px": _rms(sk[sl]), "stretch_px": _rms(st[sl])}
    brx, bry = sig["band_rx"] * k, sig["band_ry"] * k
    valid = np.isfinite(brx).mean(0) >= 0.8
    wob = []
    for b in np.where(valid)[0]:
        if use_delta and "d_band_rx" in sig:
            vx = reconstruct_velocity(brx[:, b], sig["d_band_rx"][:, b] * k, F)
            vy = reconstruct_velocity(bry[:, b], sig["d_band_ry"][:, b] * k, F)
        else:
            vx, vy = brx[:, b], bry[:, b]
        wob.append(F.highpass(path_from_velocity(vx))[sl] ** 2 + F.highpass(path_from_velocity(vy))[sl] ** 2)
    jel["wobble_px"] = float(np.sqrt(np.mean(wob))) if wob else float("nan")
    jel["bands_used"] = int(valid.sum())
    jel["jello_px"] = float(math.sqrt(jel["skew_px"] ** 2 + jel["stretch_px"] ** 2 + (jel["wobble_px"] if wob else 0) ** 2))
    # high band only (8 Hz..Nyquist): typical prop-wash / vibration jello
    i8 = 1
    jel["skew_px_8hz+"] = _rms(F.band(i8, kx_p)[sl]) * Hr / math.sqrt(12)
    jel["stretch_px_8hz+"] = _rms(F.band(i8, ky_p)[sl]) * Hr / math.sqrt(12)
    m["jello"] = jel

    # corner wobble: HF motion of the 4 corner regions relative to a homography fitted on the frame centre
    # (a rectilinear frame under pure rotation moves exactly by a homography) -> lens-model, non-linear
    # rolling-shutter and warp errors that make the periphery swim.  Position domain, >fc high-passed,
    # RMS over time and the valid corners (1920-eq px).  8+ Hz band reported separately.
    corner = {"corners_used": 0, "corner_wobble_px": float("nan"), "corner_wobble_px_8hz+": float("nan")}
    corner_e = np.full((N, 4), np.nan)      # per-frame HF corner energy (px^2), NaN where not measured
    if "cor_rx" in sig and np.ndim(sig["cor_rx"]) >= 2 and len(sig["cor_rx"]):
        from scipy.ndimage import uniform_filter1d
        A3 = lambda a: (np.asarray(a, float) if np.ndim(a) == 3 else np.asarray(a, float)[:, :, None]) * k
        crx, cry = A3(sig["cor_rx"]), A3(sig["cor_ry"])
        dcx = A3(sig["d_cor_rx"]) if "d_cor_rx" in sig else np.full_like(crx, np.nan)
        dcy = A3(sig["d_cor_ry"]) if "d_cor_ry" in sig else np.full_like(cry, np.nan)
        halves = crx.shape[2] == 3
        e8 = np.full((N, 4), np.nan)
        en = np.full((N, 4), np.nan)

        def cpath(c, j):
            if use_delta and np.isfinite(dcx[:, c, j]).mean() > 0.3:
                vx = reconstruct_velocity(crx[:, c, j], dcx[:, c, j], F)
                vy = reconstruct_velocity(cry[:, c, j], dcy[:, c, j], F)
            else:
                vx, vy = _fill_nan(crx[:, c, j])[0], _fill_nan(cry[:, c, j])[0]
            return vx, vy

        for c in range(4):
            fin = np.isfinite(crx[:, c, 0])
            if fin.mean() < 0.3:
                continue
            # only frames whose corner was measured and whose +-0.5 s neighbourhood is >= 80% measured
            # (gaps are interpolated for the filters but never scored)
            okp = fin & (uniform_filter1d(fin.astype(float), max(int(round(fs)) | 1, 3), mode="nearest") >= 0.8)
            okf = np.ones(N, bool)       # path sample j lies between pair j-1 and pair j
            okf[1:] &= okp
            okf[:-1] &= okp
            vx, vy = cpath(c, 0)
            px_, py_ = path_from_velocity(vx), path_from_velocity(vy)
            corner_e[okf, c] = (F.highpass(px_) ** 2 + F.highpass(py_) ** 2)[okf]
            e8[okf, c] = (F.band(1, px_) ** 2 + F.band(1, py_) ** 2)[okf]
            if halves and np.isfinite(crx[:, c, 1]).mean() > 0.3 and np.isfinite(crx[:, c, 2]).mean() > 0.3:
                ax_, ay_ = cpath(c, 1)
                bx_, by_ = cpath(c, 2)
                dx_ = F.highpass(path_from_velocity(ax_ - bx_))
                dy_ = F.highpass(path_from_velocity(ay_ - by_))
                en[okf, c] = ((dx_ ** 2 + dy_ ** 2) / 4.0)[okf]
        ce, c8, cn = corner_e[sl], e8[sl], en[sl]
        used = np.isfinite(ce).mean(0) >= 0.3
        if used.any():
            import warnings
            vals = ce[:, used][np.isfinite(ce[:, used])]
            cw = float(np.sqrt(np.mean(vals)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                nf = float(np.sqrt(np.nanmean(cn[:, used]))) if np.isfinite(cn[:, used]).any() else float("nan")
                corner = {"corners_used": int(used.sum()), "corner_wobble_px": cw,
                          "corner_wobble_noise_px": nf,
                          "corner_wobble_px_denoised": float(math.sqrt(max(cw ** 2 - nf ** 2, 0.0))) if math.isfinite(nf) else float("nan"),
                          "corner_wobble_px_8hz+": float(np.sqrt(np.nanmean(c8[:, used]))),
                          "per_corner_px": {nm: (float(np.sqrt(np.nanmean(ce[:, i]))) if used[i] else None)
                                            for i, nm in enumerate(("TL", "TR", "BL", "BR"))},
                          "scored_frac": np.isfinite(ce).mean(0).tolist()}
                wl_ = max(int(round(window_s * fs)), 4)
                pooled = np.nanmean(np.where(used[None, :], ce, np.nan), axis=1)
                wins = [np.nanmean(pooled[i * wl_:(i + 1) * wl_]) for i in range(len(pooled) // wl_)]
            wins = np.sqrt(np.asarray([w for w in wins if np.isfinite(w)]))
            if len(wins):
                corner["window_median_px"] = float(np.median(wins))
    m["corner"] = corner

    # single-frame displacement spikes ("jumps"): deviation of each pair's velocity (the same parallax-robust
    # velocity the HF metric integrates) from the median of its +-3 neighbours.  A lasting position step at
    # frame k -> one-pair impulse; a one-frame displacement -> two adjacent impulses (merged into one event).
    # 4-vector [tx, ty, roll*rho, logscale*rho] (rho = RMS radius), magnitude in 1920-eq px.
    rho = math.sqrt(R2)
    vv = np.column_stack([vel["tx"], vel["ty"], vel["rot"] * rho, vel["logs"] * rho])
    jdev = spike_deviation(vv)
    m["spikes"] = spike_summary(jdev, fs, meta.get("start", 0.0))

    # similarity-model version of the headline (for comparison)
    vs = {"tx": sig["sim_tx"] * k, "ty": sig["sim_ty"] * k, "rot": sig["sim_rot"], "logs": sig["sim_logs"]}
    hps = {kk: F.highpass(path_from_velocity(v)) for kk, v in vs.items()}
    m["hf_jitter_px_similarity_model"], _ = combined(hps)
    vr = {"tx": sig["ran_tx"] * k, "ty": sig["ran_ty"] * k, "rot": sig["ran_rot"], "logs": sig["ran_logs"]}
    hpr = {kk: F.highpass(path_from_velocity(v)) for kk, v in vr.items()}
    m["hf_jitter_px_ransac_inlier_model"], _ = combined(hpr)
    m["hf_trans_px_ransac_inlier_model"] = float(np.sqrt(np.mean(hpr["tx"][sl] ** 2 + hpr["ty"][sl] ** 2)))

    # Liu 2013 stability
    stab_add = liu_stability_additive({"tx": paths["tx"][1:], "ty": paths["ty"][1:], "rot": paths["rot"][1:]})
    stab_add["trans"] = (stab_add["tx"] + stab_add["ty"]) / 2
    stab_add["avg"] = (stab_add["trans"] + stab_add["rot"]) / 2
    stab_add["min"] = min(stab_add["trans"], stab_add["rot"])
    m["stability_liu_additive"] = stab_add
    m["stability_liu_difrint"] = liu_stability_difrint(sig["H"], W, H)

    # photometric
    luma = sig["frame_luma"]
    sharp = np.log(np.maximum(sig["frame_sharp"], 1e-6))
    m["flicker_luma_levels"] = _rms(F.highpass(luma)[sl])
    m["blur_flicker_logvar"] = _rms(F.highpass(sharp)[sl])
    m["sharpness_median"] = float(np.median(sig["frame_sharp"]))
    m["black_border_frac_mean"] = float(np.mean(sig["frame_black"]))
    m["black_border_frac_max"] = float(np.max(sig["frame_black"]))

    # tracking health
    m["tracking"] = {
        "pairs": int(n_pairs), "failed_pairs": n_fail,
        "median_points": float(np.nanmedian(sig["n_points"])),
        "median_tracked": float(np.nanmedian(sig["n_tracked"])),
        "median_inliers": float(np.nanmedian(sig["n_inliers"])),
        "median_residual_px": float(np.nanmedian(sig["res_rms"]) * k),
        "median_abs_residual_all_px": float(np.nanmedian(sig["res_med_all"]) * k),
        "median_se_trans_px": float(np.nanmedian(np.hypot(sig["se_tx"], sig["se_ty"])) * k),
        "median_delta_tracks": float(np.nanmedian(sig["n_delta"])) if "n_delta" in sig else float("nan"),
    }
    m["units"] = {"px": f"pixels at 1920-wide equivalent (frame {Wr:.0f}x{Hr:.0f})",
                  "fc_hz": fc, "trim_s": trim_s, "fps": fs, "combined_radius_px": math.sqrt(R2)}
    derived = {"paths": paths, "hp": hp, "lp_speed": lp_speed_eq, "skew": sk, "stretch": st, "jump_dev": jdev,
               "corner_e": corner_e}
    return m, derived


def metrics_from_signals(npz_path, fc=2.0, trim_s=0.5):
    """Re-score a saved signal NPZ (from run(..., signals=...)) with the CURRENT compute_metrics, without
    decoding.  Returns (metrics, derived).  Needs an NPZ written by eval >= 2.0 for corner/spike metrics."""
    z = np.load(npz_path)
    sig = {kk: np.asarray(z[kk]) for kk in z.files}
    n = int(len(sig["tx"])) + 1
    W = int(z["analysis_w"])
    H = int(z["analysis_h"]) if "analysis_h" in z else int(round(W * 9 / 16 / 2)) * 2
    meta = {"info": {"fps": float(z["fps"])}, "analysis_w": W, "analysis_h": H, "start": float(z["start"]),
            "n_frames": n}
    return compute_metrics(sig, meta, fc=fc, trim_s=trim_s)


# --------------------------------------------------------------------------
# reference (original) comparisons: cropping / distortion / alignment
# --------------------------------------------------------------------------
_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=3000)
    return _SIFT


def match_homography(img_o, img_s, ratio=0.75, thr_norm=3.0 / 960):
    """Homography original->stabilised in width-normalised coords (x/w, y/w)."""
    sift = _sift()
    k1, d1 = sift.detectAndCompute(img_o, None)
    k2, d2 = sift.detectAndCompute(img_s, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None, 0, float("nan")
    bf = cv2.BFMatcher()
    good = [m for m, n in (x for x in bf.knnMatch(d1, d2, k=2) if len(x) == 2) if m.distance < ratio * n.distance]
    if len(good) < 12:
        return None, len(good), float("nan")
    wo, ws = img_o.shape[1], img_s.shape[1]
    p1 = np.float32([k1[m.queryIdx].pt for m in good]) / wo
    p2 = np.float32([k2[m.trainIdx].pt for m in good]) / ws
    Hm, inl = cv2.findHomography(p1, p2, cv2.RANSAC, thr_norm, maxIters=3000, confidence=0.999)
    if Hm is None:
        return None, 0, float("nan")
    inl = inl.ravel().astype(bool)
    pr = cv2.perspectiveTransform(p1[inl].reshape(-1, 1, 2), Hm).reshape(-1, 2)
    err = float(np.median(np.linalg.norm(pr - p2[inl], axis=1)) * 960)
    return Hm / Hm[2, 2], int(inl.sum()), err


def crop_distortion_from_H(Hm, aspect_o, aspect_s):
    """aspect = h/w.  Returns (liu_crop, area_frac, distortion)."""
    liu_crop = 1.0 / math.hypot(Hm[0, 0], Hm[0, 1])
    corners = np.float32([[0, 0], [1, 0], [1, aspect_s], [0, aspect_s]]).reshape(-1, 1, 2)
    back = cv2.perspectiveTransform(corners, np.linalg.inv(Hm)).reshape(-1, 2)
    # clip the back-projected output frame to the original frame, area fraction
    poly = back.astype(np.float32)
    rect = np.float32([[0, 0], [1, 0], [1, aspect_o], [0, aspect_o]])
    inter, _ = cv2.intersectConvexConvex(poly, rect)
    area_frac = float(inter) / aspect_o
    sv = np.linalg.svd(Hm[:2, :2], compute_uv=False)
    return liu_crop, area_frac, float(sv[1] / sv[0])


def _read_n(path, start, n, w, h, hwaccel=True):
    r = FrameReader(path, start, n / probe(path).fps + 0.5, w, h, hwaccel=hwaccel)
    frames = []
    for f in r:
        frames.append(f.copy())
        if len(frames) >= n:
            r.close()
            break
    return frames


def _poly_terms(p, order=3):
    x, y = p[:, 0], p[:, 1]
    cols = [x ** i * y ** j for i in range(order + 1) for j in range(order + 1 - i)]
    return np.column_stack(cols)


def warp_residual(img_o, img_s, ratio=0.75, order=3):
    """Median residual (px at 960 wide) of a robust cubic-polynomial warp original->stabilised.

    At the correct frame pairing the mapping is a smooth lens-undistortion + rotation (+RS)
    warp, which a cubic fits to sub-pixel accuracy; a 1-frame mismatch adds depth-dependent
    parallax that no smooth warp can explain.  Used to verify time alignment."""
    sift = _sift()
    k1, d1 = sift.detectAndCompute(img_o, None)
    k2, d2 = sift.detectAndCompute(img_s, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return float("nan"), 0
    good = [m for m, n in (x for x in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(x) == 2) if m.distance < ratio * n.distance]
    if len(good) < 30:
        return float("nan"), len(good)
    wo, ws = img_o.shape[1], img_s.shape[1]
    p1 = np.float64([k1[m.queryIdx].pt for m in good]) / wo
    p2 = np.float64([k2[m.trainIdx].pt for m in good]) / ws
    Hm, inl = cv2.findHomography(p1.astype(np.float32), p2.astype(np.float32), cv2.RANSAC, 10.0 / 960)
    if Hm is None:
        return float("nan"), 0
    m = inl.ravel().astype(bool)
    X = _poly_terms(p1 - 0.5, order)
    for _ in range(4):
        if m.sum() < X.shape[1] * 2:
            return float("nan"), int(m.sum())
        sol, *_ = np.linalg.lstsq(X[m], p2[m], rcond=None)
        r = np.linalg.norm(X @ sol - p2, axis=1)
        med = np.median(r[m])
        m = r < max(4 * 1.4826 * med, 0.5 / 960)
    return float(np.median(r[m]) * 960), int(m.sum())


def check_offset(stab, ref, t_list, search=3, width=960):
    """Match stab frame at time t against ref frames t+k/fps, k in [-search, search];
    best k = smallest cubic-warp residual (see warp_residual)."""
    is_, ir = probe(stab), probe(ref)
    ws, hs = analysis_size(is_, width)
    wr, hr = analysis_size(ir, width)
    results = []
    for t in t_list:
        s0 = max(t - (search + 1) / ir.fps, 0)
        rf = _read_n(ref, s0, 2 * search + 3, wr, hr)
        sf = _read_n(stab, s0, 2 * search + 3, ws, hs)
        j = int(round((t - s0) * is_.fps))
        if j >= len(sf):
            continue
        row = {}
        for kk in range(-search, search + 1):
            i = j + kk
            if 0 <= i < len(rf):
                err, ninl = warp_residual(rf[i], sf[j])
                row[kk] = {"cubic_warp_median_residual_px960": err, "inliers": ninl}
        valid = {q: v for q, v in row.items() if np.isfinite(v["cubic_warp_median_residual_px960"])}
        if not valid:
            results.append({"t": t, "best_offset_frames": None, "scores": row})
            continue
        best = min(valid, key=lambda q: valid[q]["cubic_warp_median_residual_px960"])
        errs = sorted(v["cubic_warp_median_residual_px960"] for v in valid.values())
        results.append({"t": t, "best_offset_frames": int(best),
                        "contrast": float(errs[1] / max(errs[0], 1e-6)) if len(errs) > 1 else None, "scores": row})
    return results


def sharpness_series(path, start, dur, width=960, hwaccel=True):
    info = probe(path)
    w, h = analysis_size(info, width)
    out = []
    for img in FrameReader(path, start, dur, w, h, hwaccel=hwaccel):
        out.append(float(cv2.Laplacian(img[h // 8: h - h // 8, w // 8: w - w // 8], cv2.CV_32F).var()))
    return np.asarray(out)


def align_by_sharpness(stab_sharp, ref, start, fps, max_lag=30, width=960):
    """Per-frame motion blur is baked into each captured frame and survives any warp, so the
    high-passed log-sharpness series of the stabilised clip and of the original correlate
    sharply at the true frame offset.  Returns best lag (frames; ref index = stab index + lag)."""
    n = len(stab_sharp)
    m0 = min(max_lag, int(start * fps))
    r = sharpness_series(ref, start - m0 / fps, (n + m0 + max_lag) / fps, width)
    F = Filters(fps, 2.0)
    a = F.highpass(np.log(np.maximum(stab_sharp, 1e-6)))
    b = F.highpass(np.log(np.maximum(r, 1e-6)))
    a = (a - a.mean()) / (a.std() + 1e-12)
    lags, cc = [], []
    for lag in range(-m0, max_lag + 1):
        i0 = m0 + lag
        seg = b[i0: i0 + n]
        if len(seg) < n:
            continue
        seg = (seg - seg.mean()) / (seg.std() + 1e-12)
        lags.append(lag)
        cc.append(float(np.mean(a * seg)))
    cc = np.asarray(cc)
    i = int(np.argmax(cc))
    second = float(np.max(np.delete(cc, [j for j in range(max(i - 1, 0), min(i + 2, len(cc)))]))) if len(cc) > 3 else float("nan")
    return {"best_lag_frames": int(lags[i]), "peak_corr": float(cc[i]), "next_best_corr": second,
            "corr_by_lag": dict(zip([int(x) for x in lags], [round(float(x), 4) for x in cc]))}


class FrameTap:
    """Collects the motion pass's frames that the reference pass needs (window index % step == 0, plus the
    pairing-check samples) and streams them through a queue (reference_metrics(stab_tap=...))."""

    def __init__(self, step, extra=()):
        import queue
        self.step, self.extra = int(step), set(int(x) for x in extra)
        self.q = queue.Queue()

    def __call__(self, i, img):
        if i % self.step == 0 or i in self.extra:
            self.q.put((int(i), img.copy()))

    def close(self):
        self.q.put(None)

    def __iter__(self):
        while True:
            it = self.q.get()
            if it is None:
                return
            yield it


def reference_metrics(stab, ref, start, dur, ref_start, width=960, step=30, footprint=True, align_search=6,
                      align_samples=4, stab_tap=None):
    """Crop / distortion / frame pairing of `stab` against its original `ref` (one sequential decode of each window,
    frames picked by exact index; eval.footprint.read_select).

    * EXACT crop (official): `footprint_area_*` = fraction of the SOURCE frame the stabilised frame shows, every
      `step` frames, from a fitted KB4-source -> pinhole(+rolling-shutter) render model (eval.footprint).  Needs
      the source lens (DJI embedded KB4, via the engine's telemetry parser).
    * legacy (backward compatibility; approximate, reads ~0.19 too high for fisheye originals):
      `visible_area_frac_*`, `linear_crop_mean`, `cropping_ratio_liu_*`, `distortion_value_*` (homography, same
      SIFT matches).
    * pairing (`alignment`, align_search > 0): stab frames j at align_samples positions vs original frames
      j-align_search..j+align_search, fitted model + band-passed NCC (eval.footprint.alignment_from_frames)."""
    from concurrent.futures import ThreadPoolExecutor
    from . import footprint as fpm
    is_, ir = probe(stab), probe(ref)
    ws, hs = analysis_size(is_, width)
    wr, hr = analysis_size(ir, width)
    if abs(is_.fps - ir.fps) > 1e-3:
        print("WARNING: fps differ; frame pairing by index may be wrong", file=sys.stderr)
    lens_err, la = None, None
    try:
        la = fpm.lens_at_width(fpm.source_lens(ref), wr)
    except Exception as e:  # no lens metadata -> legacy crop only, no pairing check
        lens_err = repr(e)
    n_total = int(round(dur * is_.fps))
    js = fpm.alignment_samples(n_total, align_samples, align_search) if (align_search > 0 and la is not None) else []
    lam = la if (footprint and la is not None) else _NoLens()
    if stab_tap is None:
        with ThreadPoolExecutor(2) as ex:
            f_s = ex.submit(fpm.read_select, stab, start, dur, width, step, [(j, j) for j in js])
            f_r = ex.submit(fpm.read_select, ref, ref_start, dur, width, step,
                            [(j - align_search, j + align_search) for j in js])
            sf, rf = f_s.result(), f_r.result()
        grid = sorted(i for i in sf if i % step == 0)
        rows = fpm.footprint_rows({i: sf[i] for i in grid}, {i: rf[i] for i in grid if i in rf}, lam, ws, hs,
                                  legacy=True)
    else:
        # the stabilised frames arrive from the motion pass (same decoder, same scaling, same indices)
        rf = fpm.read_select(ref, ref_start, dur, width, step, [(j - align_search, j + align_search) for j in js])
        sf, rows, al_samples = {}, [], []
        for i, img in stab_tap:
            if i in js:          # pairing check as soon as the frame arrives (overlaps the motion pass)
                al_samples += fpm.alignment_from_frames({i: img}, rf, [i], align_search, la, ws, hs)["samples"]
            if i % step == 0 and i in rf:
                rows += fpm.footprint_rows({i: img}, {i: rf[i]}, lam, ws, hs, legacy=True)
    out = {"step": step}
    leg = [r for r in rows if "legacy_area" in r]
    out["frames_measured"] = len(leg)
    if leg:
        crops = np.array([r["legacy_liu"] for r in leg])
        areas = np.array([r["legacy_area"] for r in leg])
        dvs = np.array([r["legacy_dv"] for r in leg])
        out.update({
            "cropping_ratio_liu_mean": float(min(crops.mean(), 1.0)),
            "cropping_ratio_liu_min": float(min(crops.min(), 1.0)),
            "visible_area_frac_mean": float(areas.mean()),
            "visible_area_frac_min": float(areas.min()),
            "linear_crop_mean": float(np.sqrt(areas).mean()),
            "distortion_value_min": float(dvs.min()),
            "distortion_value_mean": float(dvs.mean()),
            "homography_median_err_px960": float(np.median([r["legacy_herr"] for r in leg])),
            "median_inliers": float(np.median([r["legacy_inliers"] for r in leg])),
            "note": ("LEGACY visible_area_frac_* / linear_crop / liu / distortion: original->stabilised mapping fitted "
                     "as a homography (approximate for a fisheye original; reads ~0.19 high on O3). The official "
                     "crop number is footprint_area_mean."),
        })
    sm = fpm.summarize_rows(rows)
    if sm.get("n"):
        out.update({
            "footprint_area_mean": sm["mean"], "footprint_area_median": sm["median"], "footprint_area_min": sm["min"],
            "footprint_area_p10": sm["p10"], "footprint_n": sm["n"], "footprint_n_sampled": sm["n_sampled"],
            "footprint_focal_1920_median": sm["focal_1920_median"], "footprint_focal_1920_min": sm["focal_1920_min"],
            "footprint_focal_1920_max": sm["focal_1920_max"], "footprint_resid_median_px960": sm["resid_median_px960"],
            "footprint_method": "fitted KB4-source -> pinhole+RS render model (eval.footprint); fraction of source area",
        })
    elif lens_err:
        out["footprint_error"] = lens_err
    out["footprint_rows"] = [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
    if js:
        al = (fpm.alignment_from_frames(sf, rf, js, align_search, la, ws, hs) if stab_tap is None
              else fpm.alignment_consensus(al_samples, align_search))
        if al["lag_frames"] is not None and al["samples"]:
            al["best_lag_frames"] = int(al["lag_frames"])
            al["peak_corr"] = float(np.mean([s_["ncc_best"] for s_ in al["samples"]]))
            al["next_best_corr"] = float(np.nanmean([s_["ncc_second"] for s_ in al["samples"]]))
            al["at_search_edge"] = bool(abs(al["lag_frames"]) >= align_search)
        out["alignment"] = al
    if not leg and not sm.get("n"):
        out["error"] = "no frames matched"
    return out


class _NoLens:
    """Placeholder when the footprint is disabled/unavailable: fit_render_model then returns None."""
    height = 1
    width = 1

    def unproject(self, p):
        return np.zeros((len(p), 3))


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def make_plot(png, m, derived, meta, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = meta["info"]["fps"]
    N = len(derived["paths"]["tx"])
    t = meta["start"] + np.arange(N) / fs
    fig, ax = plt.subplots(4, 1, figsize=(13, 14))
    ax[0].plot(t, derived["paths"]["tx"], lw=0.8, label="path x")
    ax[0].plot(t, derived["paths"]["ty"], lw=0.8, label="path y")
    ax[0].set_ylabel("px (1920-eq)"); ax[0].legend(loc="upper left"); ax[0].set_title(title)
    a2 = ax[0].twinx(); a2.plot(t, np.degrees(derived["paths"]["rot"]), "k", lw=0.6, alpha=0.5); a2.set_ylabel("roll deg")
    ax[1].plot(t, derived["hp"]["tx"], lw=0.6, label=f"HP x  rms {m['hf_tx_px']:.3f}")
    ax[1].plot(t, derived["hp"]["ty"], lw=0.6, label=f"HP y  rms {m['hf_ty_px']:.3f}")
    ax[1].plot(t, np.asarray(derived["hp"]["rot"]) * m["units"]["combined_radius_px"], lw=0.6,
               label=f"HP roll x r  rms {m['hf_rot_equiv_px']:.3f}")
    ax[1].set_ylabel(f"px, >{m['units']['fc_hz']} Hz"); ax[1].legend(loc="upper left", fontsize=8)
    ax[1].set_title(f"hf_jitter_px={m['hf_jitter_px']:.3f}  (noise floor {m['noise_floor_px']:.3f})")
    f = np.asarray(m["psd"]["f"])
    ax[2].loglog(f[1:], np.asarray(m["psd"]["tx"])[1:], label="x")
    ax[2].loglog(f[1:], np.asarray(m["psd"]["ty"])[1:], label="y")
    ax[2].loglog(f[1:], np.asarray(m["psd"]["rot_rad"])[1:] * m["units"]["combined_radius_px"] ** 2, label="roll x r")
    for x in (2, 8):
        ax[2].axvline(x, color="gray", ls=":")
    ax[2].set_xlabel("Hz"); ax[2].set_ylabel("PSD px^2/Hz"); ax[2].legend(); ax[2].grid(True, which="both", alpha=0.3)
    ax[3].plot(t, derived["skew"], lw=0.6, label=f"skew rms {m['jello']['skew_px']:.3f}")
    ax[3].plot(t, derived["stretch"], lw=0.6, label=f"stretch rms {m['jello']['stretch_px']:.3f}")
    ax[3].set_ylabel("jello px (HP)"); ax[3].set_xlabel("s"); ax[3].legend(loc="upper left", fontsize=8)
    ax[3].set_title(f"jello_px={m['jello']['jello_px']:.3f}  wobble={m['jello']['wobble_px']:.3f}")
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------
def run(video, start=0.0, dur=None, ref=None, ref_start=None, width=960, fc=2.0, trim=0.5,
        focal_px=None, hfov=None, ref_step=30, ref_offset_search=6, plot=None, signals=None,
        hwaccel=True, verbose=True, alignment="match"):
    """Measure `video` (window start/dur).  With ref (the ORIGINAL): exact source-footprint crop (+ legacy
    homography crop) every ref_step frames and a frame-pairing check (alignment='match': direct frame matching,
    +-ref_offset_search frames; 'blur': the legacy blur-series cross-correlation; 'off').  The reference work
    runs in a background thread while the motion pass decodes the video."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    rstart = start if ref_start is None else ref_start
    fut_ref = None
    ex = None
    tap = None
    if ref:
        from .footprint import alignment_samples
        ex = ThreadPoolExecutor(1)
        info_ = probe(video)
        wdur = dur if dur else max(info_.duration - start, 0.0)
        srch = min(int(ref_offset_search), 8) if (ref_offset_search > 0 and alignment == "match") else 0
        js = alignment_samples(int(round(wdur * info_.fps)), 4, srch) if srch > 0 else []
        tap = FrameTap(ref_step, js)
        fut_ref = ex.submit(reference_metrics, video, ref, start, wdur, rstart, width, ref_step, True, srch, 4, tap)
    try:
        sig, meta = extract_motion(video, start, dur, width, hwaccel=hwaccel, verbose=verbose, frame_sink=tap)
    finally:
        if tap is not None:
            tap.close()
    if focal_px is None and hfov:
        focal_px = (REF_WIDTH / 2) / math.tan(math.radians(hfov) / 2)
    m, derived = compute_metrics(sig, meta, fc=fc, trim_s=trim, focal_px=focal_px)
    result = {"video": os.path.abspath(video), "start": start, "dur": dur, "meta": meta,
              "focal_px_1920": focal_px, "code": code_version(), "metrics": m}
    if ref:
        if ref_offset_search > 0 and alignment == "blur":
            try:
                result["ref_alignment"] = align_by_sharpness(sig["frame_sharp"], ref, rstart,
                                                             meta["info"]["fps"], ref_offset_search, width)
            except Exception as e:  # pragma: no cover
                result["ref_alignment"] = {"error": repr(e)}
        try:
            rm = fut_ref.result()
        except Exception as e:  # pragma: no cover
            rm = {"error": repr(e)}
        ex.shutdown()
        if "alignment" in rm:
            result["ref_alignment"] = rm.pop("alignment")
        result["ref"] = {"path": os.path.abspath(ref), "ref_start": rstart, **rm}
        lin = result["ref"].get("linear_crop_mean")
        if lin:
            # angular-equivalent: output px rescaled to the original frame's px (a stabiliser
            # cannot look better just by zooming out / cropping less)  [legacy homography scale]
            m["hf_jitter_px_orig_scale"] = m["hf_jitter_px"] * lin
    result["elapsed_s"] = time.time() - t0
    if plot:
        make_plot(plot, m, derived, meta, f"{os.path.basename(video)}  {start:.1f}+{meta['n_frames'] / meta['info']['fps']:.1f}s")
    if signals:
        np.savez_compressed(signals, **{k: v for k, v in sig.items()},
                            **{"path_" + k: v for k, v in derived["paths"].items()},
                            **{"hp_" + k: v for k, v in derived["hp"].items()},
                            lp_speed=derived["lp_speed"], skew=derived["skew"], stretch=derived["stretch"],
                            jump_dev=derived["jump_dev"], corner_e=derived["corner_e"],
                            fps=meta["info"]["fps"], start=start, analysis_w=meta["analysis_w"],
                            analysis_h=meta["analysis_h"], code_sha1=result["code"]["sha1"])
    return result


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--ref", default=None, help="original (unstabilised) video for crop/distortion")
    ap.add_argument("--ref-start", type=float, default=None, help="start time in the reference (default = --start)")
    ap.add_argument("--ref-step", type=int, default=30, help="measure crop (exact footprint + legacy) every N frames")
    ap.add_argument("--ref-offset-search", type=int, default=6,
                    help="+/- frames searched to verify frame pairing (0=off; max 8 with --alignment match)")
    ap.add_argument("--alignment", choices=["match", "blur", "off"], default="match",
                    help="pairing check: direct frame matching (default) or the legacy blur-series correlation")
    ap.add_argument("--width", type=int, default=960, help="analysis width")
    ap.add_argument("--fc", type=float, default=2.0, help="high-pass cut-off (Hz) separating jitter from intent")
    ap.add_argument("--trim", type=float, default=0.5, help="seconds discarded at each end (filter transients)")
    ap.add_argument("--focal-px", type=float, default=None, help="focal length in 1920-wide px (for arcmin)")
    ap.add_argument("--hfov", type=float, default=None, help="horizontal FOV deg, rectilinear (alt. to --focal-px)")
    ap.add_argument("--no-hwaccel", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", default=None, help="PNG path (default: <out>.png)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--signals", default=None, help="NPZ of per-frame signals (default: <out>.npz)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    base = os.path.splitext(a.out)[0]
    plot = None if a.no_plot else (a.plot or base + ".png")
    sigp = a.signals or base + ".npz"
    res = run(a.video, a.start, a.dur, a.ref, a.ref_start, a.width, a.fc, a.trim, a.focal_px, a.hfov,
              a.ref_step, a.ref_offset_search, plot, sigp, hwaccel=not a.no_hwaccel, verbose=not a.quiet,
              alignment=a.alignment)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=_json_default)
    m = res["metrics"]
    line = (f"{os.path.basename(a.video)} [{a.start:.1f}+{res['meta']['n_frames']}f] "
            f"hf_jitter={m['hf_jitter_px']:.3f}px (trans {m['hf_trans_px']:.3f}, roll {m['hf_rot_deg']:.4f}deg) "
            f"2-8Hz={m['bands']['2-8Hz']['combined_px']:.3f} 8+Hz={list(m['bands'].values())[1]['combined_px']:.3f} "
            f"noise={m['noise_floor_px']:.3f} jello={m['jello']['jello_px']:.3f} "
            f"corner_wobble={m['corner']['corner_wobble_px']:.3f} spikes>1px={m['spikes']['n_gt_1px']} "
            f"stab(add)={m['stability_liu_additive']['avg']:.3f}")
    if "ref" in res and "footprint_area_mean" in res["ref"]:
        line += f" footprint={res['ref']['footprint_area_mean']:.4f}"
    if "ref" in res and "visible_area_frac_mean" in res["ref"]:
        line += f" (legacy crop_area={res['ref']['visible_area_frac_mean']:.3f}) dv={res['ref']['distortion_value_min']:.3f}"
    if "ref_alignment" in res and "best_lag_frames" in res["ref_alignment"]:
        line += f" lag={res['ref_alignment']['best_lag_frames']}"
    print(line)
    return res


if __name__ == "__main__":
    main()
