"""
Synthetic validation clips for eval.jitter_metrics.

Two families:

* planar   -- a crop window moving over a real 4K still (default: the sharpest
              frame of the OA4 clip DJI_20260926153751_0005_D.MP4, frame 98) along a
              known smooth path + injected jitter; optional rolling shutter.
* parallax -- an FPV-like 3-D scene rendered by ray casting: textured ground
              plane + distant textured backdrop (160 m), camera flying forward at
              12 m/s, 2.5 m altitude, pitched down, slow turning, plus injected
              camera-ROTATION jitter.  This is the case that breaks naive
              global-motion estimators (depth-dependent flow).

For every clip the exact inter-frame correspondences of a uniform grid are
computed from the known geometry ("perfect tracker"), fitted with the same
least-squares affine + decomposition as the metric, and pushed through the same
compute_metrics() -> the *truth* value, in the metric's own units.  For the
parallax clips a second truth uses rotation-only flow (camera translation
frozen), i.e. the pure camera-shake signal.

    .venv/bin/python -m eval.synth render   [--only SUBSTR] [--jobs 3]
    .venv/bin/python -m eval.synth validate [--only SUBSTR] [--jobs 3]
    .venv/bin/python -m eval.synth report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy import signal

from . import jitter_metrics as jm
from .footage import o3, oa4

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work", "synth")
FPS_STR = "60000/1001"
FPS = 60000 / 1001
OUT_W, OUT_H = 1920, 1080
AN_W, AN_H = 960, 540          # metric analysis size (must match jitter_metrics default)
CELL = 16                       # analysis grid cell (matches MotionEstimator)
N_BANDS = 12
STILL_PLANAR = os.path.join(WORK, "still_0005_f98.png")
STILL_BACKDROP = os.path.join(WORK, "still_0025_t12.png")
SRC_CLIP = oa4("DJI_20260926153751_0005_D.MP4")


def ensure_stills():
    os.makedirs(WORK, exist_ok=True)
    if not os.path.exists(STILL_PLANAR):
        subprocess.run([jm.FFMPEG, "-v", "error", "-y", "-i", SRC_CLIP, "-vf", "select=eq(n\\,98)",
                        "-fps_mode", "passthrough", "-frames:v", "1", STILL_PLANAR], check=True)
    if not os.path.exists(STILL_BACKDROP):
        subprocess.run([jm.FFMPEG, "-v", "error", "-y", "-ss", "12", "-i",
                        o3("DJI_0025.MP4"),
                        "-frames:v", "1", STILL_BACKDROP], check=True)


# ---------------------------------------------------------------------------
# motion definitions (all functions of absolute time t, vectorised)
# ---------------------------------------------------------------------------
def min_jerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def planar_pan(kind, t):
    """-> px, py (output px), roll (deg), zoom."""
    t = np.asarray(t, float)
    z0 = np.zeros_like(t)
    if kind == "none":
        return z0, z0, z0, z0 + 1
    if kind == "linear":           # 112.5 px/s x, 37.5 px/s y
        return -450 + 112.5 * t, -150 + 37.5 * t, z0, z0 + 1
    if kind == "sine":             # slow wandering pan + slow roll
        return (400 * np.sin(2 * np.pi * 0.25 * t), 150 * np.sin(2 * np.pi * 0.15 * t),
                2.0 * np.sin(2 * np.pi * 0.1 * t), z0 + 1)
    if kind == "ease":             # 800 px min-jerk moves in 1.2 s, 0.8 s hold (peak 1000 px/s)
        ph = np.mod(t, 4.0)
        a = min_jerk(ph / 1.2)
        b = min_jerk((ph - 2.0) / 1.2)
        return -400 + 800 * a - 800 * b, z0, z0, z0 + 1
    if kind == "fast":             # 600 px/s constant-velocity whip
        return -1000 + 250 * t, z0, z0, z0 + 1
    if kind == "zoomroll":
        return -200 + 50 * t, z0, 4.0 * t / 8, 1 + 0.08 * t / 8
    raise ValueError(kind)


class Jitter:
    """2-D translation jitter (output px) + roll jitter (deg), continuous time."""

    def __init__(self, spec, dur):
        self.spec = spec or {}
        self.type = self.spec.get("type")
        if self.type == "noise":
            fs_hi = 2400.0
            n = int((dur + 2) * fs_hi)
            rng = np.random.default_rng(self.spec.get("seed", 1))
            lo, hi = self.spec.get("band", (3, 20))
            sos = signal.butter(4, [lo, hi], "bandpass", fs=fs_hi, output="sos")
            x = signal.sosfiltfilt(sos, rng.standard_normal((n, 2)), axis=0)
            sl = slice(int(fs_hi), n - int(fs_hi))        # drop edges
            x -= x[sl].mean(0)
            x *= self.spec["A"] / math.sqrt((x[sl] ** 2).sum(1).mean())
            self._t = np.arange(n) / fs_hi - 1.0
            self._x = x

    def __call__(self, t):
        t = np.asarray(t, float)
        z = np.zeros_like(t)
        tp = self.type
        if tp is None:
            return z, z, z
        if tp == "sine":
            A, f = self.spec["A"], self.spec["f"]
            return A * np.sin(2 * np.pi * f * t), A * np.cos(2 * np.pi * f * t), z
        if tp == "noise":
            return np.interp(t, self._t, self._x[:, 0]), np.interp(t, self._t, self._x[:, 1]), z
        if tp == "roll":
            return z, z, self.spec["A_deg"] * np.sin(2 * np.pi * self.spec["f"] * t)
        raise ValueError(tp)


# ---------------------------------------------------------------------------
# planar renderer
# ---------------------------------------------------------------------------
PLANAR_SCALE = 1.25     # source px per output px


class Planar:
    def __init__(self, spec):
        self.spec = spec
        img = cv2.imread(spec.get("still", STILL_PLANAR), cv2.IMREAD_COLOR)
        self.src = cv2.GaussianBlur(img, (0, 0), 0.6)
        self.C = np.array([(img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2])
        self.jit = Jitter(spec.get("jit"), spec["dur"])
        self.rs = spec.get("rs", 0.0)
        u, v = np.meshgrid(np.arange(OUT_W, dtype=np.float64), np.arange(OUT_H, dtype=np.float64))
        self.du, self.dv = u - (OUT_W - 1) / 2, v - (OUT_H - 1) / 2

    def pose(self, t):
        px, py, r, z = planar_pan(self.spec["pan"], t)
        jx, jy, jr = self.jit(t)
        return px + jx, py + jy, np.radians(r + jr), z

    def row_times(self, t, v):
        return t + np.asarray(v) / OUT_H * self.rs

    def src_of(self, du, dv, t):
        """output centred coords at (absolute) times t -> source px."""
        px, py, th, z = self.pose(t)
        c, s = np.cos(th), np.sin(th)
        x = self.C[0] + PLANAR_SCALE * (px + (c * du - s * dv) / z)
        y = self.C[1] + PLANAR_SCALE * (py + (s * du + c * dv) / z)
        return x, y

    def out_of(self, x, y, t):
        px, py, th, z = self.pose(t)
        a = (x - self.C[0]) / PLANAR_SCALE - px
        b = (y - self.C[1]) / PLANAR_SCALE - py
        c, s = np.cos(th), np.sin(th)
        return z * (c * a + s * b), z * (-s * a + c * b)

    def render(self, t):
        tr = self.row_times(t, np.arange(OUT_H))[:, None]
        x, y = self.src_of(self.du, self.dv, tr)
        return cv2.remap(self.src, x.astype(np.float32), y.astype(np.float32), cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT)

    def correspond(self, u0, v0, t0, t1, frozen_translation=False):
        """grid (output centred coords) in frame t0 -> positions in frame t1."""
        x, y = self.src_of(u0, v0, self.row_times(t0, v0 + (OUT_H - 1) / 2))
        v1 = v0.copy()
        for _ in range(4):
            u1, v1n = self.out_of(x, y, self.row_times(t1, v1 + (OUT_H - 1) / 2))
            v1 = v1n
        return u1, v1


# ---------------------------------------------------------------------------
# parallax (ray-cast) renderer
# ---------------------------------------------------------------------------
F_PX = 960.0          # hfov 90 deg at 1920 wide
GROUND_M_PER_PX = 0.005
WALL_M_PER_PX = 0.04
WALL_Y = 160.0
WALL_TOP = 60.0
B = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)    # camera (x r, y d, z f) -> world (x r, y f, z up)


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def _fold(x, n):
    """mirror-tile coordinate into [0, n-1]."""
    p = 2 * (n - 1)
    m = np.mod(x, p)
    return np.where(m > n - 1, p - m, m)


def _remap_points(img, x, y, row=4096):
    """bilinear sample img at N points (remap needs dst dims < 32767)."""
    n = x.size
    m = -(-n // row) * row
    mx = np.zeros(m, np.float32); my = np.zeros(m, np.float32)
    mx[:n], my[:n] = x, y
    out = cv2.remap(img, mx.reshape(-1, row), my.reshape(-1, row), cv2.INTER_LINEAR)
    return out.reshape(m, -1)[:n].astype(np.float32)


class Parallax:
    def __init__(self, spec):
        self.spec = spec
        self.rs = spec.get("rs", 0.0)
        self.jit = Jitter(spec.get("jit"), spec["dur"])
        w = cv2.imread(spec.get("backdrop", STILL_BACKDROP), cv2.IMREAD_COLOR)
        # ground texture: lower (terrain) part of the outdoor still, mirror-tiled
        g = w[int(w.shape[0] * 0.42):, :].copy()
        self.gpyr, self.wpyr = [g], [w]
        for _ in range(8):
            self.gpyr.append(cv2.pyrDown(self.gpyr[-1]))
        for _ in range(5):
            self.wpyr.append(cv2.pyrDown(self.wpyr[-1]))
        u, v = np.meshgrid(np.arange(OUT_W, dtype=np.float64), np.arange(OUT_H, dtype=np.float64))
        self.dc = np.stack([(u - (OUT_W - 1) / 2) / F_PX, (v - (OUT_H - 1) / 2) / F_PX, np.ones_like(u)], -1).astype(np.float32)

    def pose(self, t):
        """-> R_wc (...,3,3), position (...,3)."""
        t = np.asarray(t, float)
        freeze = self.spec.get("no_translation", False)
        V = 0.0 if freeze else 12.0
        pos = np.stack([1.5 * np.sin(2 * np.pi * 0.05 * t), V * t, 2.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)], -1)
        yaw = np.radians(15 * np.sin(2 * np.pi * 0.08 * t))
        pitch = np.radians(-15 + 3 * np.sin(2 * np.pi * 0.13 * t))
        roll = np.radians(10 * np.sin(2 * np.pi * 0.07 * t))
        jx, jy, jr = self.jit(t)                  # jx, jy in px-at-centre -> yaw, pitch
        yaw = yaw - jx / F_PX
        pitch = pitch - jy / F_PX
        roll = roll + np.radians(jr)
        R = _rz(yaw) @ _rx(pitch) @ B @ _rz(roll)
        return R, pos

    def hit(self, d, pos):
        """world ray dirs d (...,3) from pos (...,3) -> (is_ground, X, Y|Z, lam)."""
        dz = d[..., 2]
        dy = d[..., 1]
        lg = np.where(dz < -1e-9, pos[..., 2] / np.maximum(-dz, 1e-9), np.inf)
        lw = np.where(dy > 1e-9, (WALL_Y - pos[..., 1]) / np.maximum(dy, 1e-9), np.inf)
        ground = lg < lw
        lam = np.where(ground, lg, lw)
        P = pos + lam[..., None] * d
        return ground, P, lam

    def render(self, t):
        rows = t + np.arange(OUT_H) / OUT_H * self.rs
        R, pos = self.pose(rows)                                    # (H,3,3), (H,3)
        d = np.einsum("hij,hwj->hwi", R.astype(np.float32), self.dc)
        ground, P, lam = self.hit(d, pos[:, None, :].astype(np.float32))
        tx = np.where(ground, P[..., 0] / GROUND_M_PER_PX, P[..., 0] / WALL_M_PER_PX + self.wpyr[0].shape[1] / 2)
        ty = np.where(ground, P[..., 1] / GROUND_M_PER_PX, (WALL_TOP - P[..., 2]) / WALL_M_PER_PX)
        # texture footprint (texture px per output px) for trilinear mip selection
        gyy, gyx = np.gradient(tx)
        gxy, gxx = np.gradient(ty)
        foot = np.sqrt(np.maximum(gyy ** 2 + gxy ** 2, gyx ** 2 + gxx ** 2))
        foot = np.where(np.isfinite(foot), foot, 1.0)
        lev = np.log2(np.maximum(foot, 1.0)).ravel()
        gflat = ground.ravel()
        txf, tyf = tx.ravel(), ty.ravel()
        out = np.zeros((OUT_H * OUT_W, 3), np.float32)
        for pyr, surf in ((self.gpyr, gflat), (self.wpyr, ~gflat)):
            L = np.clip(lev, 0, len(pyr) - 1.001)
            l0 = np.floor(L).astype(np.int32)
            fr = (L - l0).astype(np.float32)
            for li in range(len(pyr)):
                for which, wv in ((0, None), (1, None)):
                    sel = np.flatnonzero(surf & (l0 + which == li))
                    if sel.size == 0:
                        continue
                    img = pyr[li]
                    h, w = img.shape[:2]
                    sc = 2.0 ** li
                    smp = _remap_points(img, _fold(txf[sel] / sc, w), _fold(tyf[sel] / sc, h))
                    wgt = (1 - fr[sel]) if which == 0 else fr[sel]
                    out[sel] += smp * wgt[:, None]
        return np.clip(out.reshape(OUT_H, OUT_W, 3), 0, 255).astype(np.uint8)

    def correspond(self, u0, v0, t0, t1, frozen_translation=False):
        """grid (output centred px) in frame t0 -> positions in frame t1 (exact geometry)."""
        r0 = t0 + (v0 + (OUT_H - 1) / 2) / OUT_H * self.rs
        R0, p0 = self.pose(r0)
        dc = np.stack([u0 / F_PX, v0 / F_PX, np.ones_like(u0)], -1)
        d = np.einsum("nij,nj->ni", R0, dc)
        _, P, _ = self.hit(d, p0)
        v1 = v0.copy()
        for _ in range(4):
            r1 = t1 + (v1 + (OUT_H - 1) / 2) / OUT_H * self.rs
            R1, p1 = self.pose(r1)
            if frozen_translation:
                p1 = p0
            q = np.einsum("nji,nj->ni", R1, P - p1)       # R^T (P - p)
            u1 = F_PX * q[:, 0] / q[:, 2]
            v1 = F_PX * q[:, 1] / q[:, 2]
        return u1, v1


# ---------------------------------------------------------------------------
# clip specs
# ---------------------------------------------------------------------------
def specs():
    S = []
    D = 8.0
    for pan in ("none", "linear", "sine", "ease", "fast", "zoomroll"):
        S.append({"name": f"planar_ctrl_{pan}", "kind": "planar", "pan": pan, "dur": D, "jit": None})
    for f in (5, 12, 25):
        for A in (0.5, 1, 2, 4):
            S.append({"name": f"planar_sine_A{A}_f{f}", "kind": "planar", "pan": "sine", "dur": D,
                      "jit": {"type": "sine", "A": A, "f": f}})
    S.append({"name": "planar_noise_A1_3-20Hz", "kind": "planar", "pan": "linear", "dur": D,
              "jit": {"type": "noise", "A": 1.0, "band": [3, 20], "seed": 3}})
    S.append({"name": "planar_noise_A0.25_3-20Hz", "kind": "planar", "pan": "linear", "dur": D,
              "jit": {"type": "noise", "A": 0.25, "band": [3, 20], "seed": 4}})
    S.append({"name": "planar_roll_0.1deg_f8", "kind": "planar", "pan": "linear", "dur": D,
              "jit": {"type": "roll", "A_deg": 0.1, "f": 8}})
    # rolling shutter (readout 12 ms of the 16.7 ms frame)
    S.append({"name": "planar_rs_ctrl_fast", "kind": "planar", "pan": "fast", "dur": D, "jit": None, "rs": 0.012})
    S.append({"name": "planar_rs_sine_A2_f12", "kind": "planar", "pan": "linear", "dur": D, "rs": 0.012,
              "jit": {"type": "sine", "A": 2, "f": 12}})
    S.append({"name": "planar_rs_sine_A2_f47", "kind": "planar", "pan": "linear", "dur": D, "rs": 0.012,
              "jit": {"type": "sine", "A": 2, "f": 47}})
    # content check on an outdoor still
    S.append({"name": "planar_outdoor_ctrl_sine", "kind": "planar", "pan": "sine", "dur": D, "jit": None,
              "still": STILL_BACKDROP})
    S.append({"name": "planar_outdoor_sine_A1_f12", "kind": "planar", "pan": "sine", "dur": D,
              "jit": {"type": "sine", "A": 1, "f": 12}, "still": STILL_BACKDROP})
    # FPV-like parallax scenes (jitter A = px at image centre, applied as yaw/pitch)
    S.append({"name": "parallax_ctrl", "kind": "parallax", "dur": D, "jit": None})
    for A, f in ((0.5, 12), (1, 5), (1, 12), (2, 25)):
        S.append({"name": f"parallax_rot_A{A}_f{f}", "kind": "parallax", "dur": D,
                  "jit": {"type": "sine", "A": A, "f": f}})
    S.append({"name": "parallax_noise_A1_3-20Hz", "kind": "parallax", "dur": D,
              "jit": {"type": "noise", "A": 1.0, "band": [3, 20], "seed": 5}})
    S.append({"name": "parallax_rs_ctrl", "kind": "parallax", "dur": D, "jit": None, "rs": 0.012})
    S.append({"name": "parallax_rs_rot_A1_f12", "kind": "parallax", "dur": D, "rs": 0.012,
              "jit": {"type": "sine", "A": 1, "f": 12}})
    return S


def make_renderer(spec):
    return Planar(spec) if spec["kind"] == "planar" else Parallax(spec)


# ---------------------------------------------------------------------------
# truth via exact correspondences
# ---------------------------------------------------------------------------
def truth_signals(spec, frozen_translation=False):
    r = make_renderer(spec)
    n = int(round(spec["dur"] * FPS))
    k = OUT_W / AN_W
    cx, cy = (AN_W - 1) / 2, (AN_H - 1) / 2
    gy, gx = np.mgrid[1:AN_H // CELL - 1, 1:AN_W // CELL - 1]
    pa = np.column_stack([(gx.ravel() + 0.5) * CELL, (gy.ravel() + 0.5) * CELL]).astype(float)  # analysis px
    # centred output coords of the grid
    u0 = (pa[:, 0] * k + (k - 1) / 2) - (OUT_W - 1) / 2
    v0 = (pa[:, 1] * k + (k - 1) / 2) - (OUT_H - 1) / 2
    keys = ["tx", "ty", "rot", "logs", "kx", "ky"]
    out = {kk: [] for kk in keys}
    brx, bry = [], []
    for i in range(1, n):
        t0, t1 = (i - 1) / FPS, i / FPS
        u1, v1 = r.correspond(u0, v0, t0, t1, frozen_translation)
        # back to analysis centred coords
        q0 = np.column_stack([(u0 + (OUT_W - 1) / 2 - (k - 1) / 2) / k - cx, (v0 + (OUT_H - 1) / 2 - (k - 1) / 2) / k - cy])
        q1 = np.column_stack([(u1 + (OUT_W - 1) / 2 - (k - 1) / 2) / k - cx, (v1 + (OUT_H - 1) / 2 - (k - 1) / 2) / k - cy])
        ok = (np.abs(q1[:, 0]) < AN_W / 2 - 1) & (np.abs(q1[:, 1]) < AN_H / 2 - 1) & np.isfinite(q1).all(1)
        A = jm._lsq_affine(q0[ok], q1[ok])
        for kk, val in zip(keys, jm.decompose_rs_affine(A)):
            out[kk].append(val)
        res = q1 - (q0 @ A[:, :2].T + A[:, 2])
        band = np.clip(((q0[:, 1] + cy) / AN_H * N_BANDS).astype(int), 0, N_BANDS - 1)
        rx, ry = np.full(N_BANDS, np.nan), np.full(N_BANDS, np.nan)
        for b in range(N_BANDS):
            m = ok & (band == b)
            if m.sum() >= 6:
                rx[b], ry[b] = np.median(res[m, 0]), np.median(res[m, 1])
        brx.append(rx); bry.append(ry)
    sig = {kk: np.asarray(v) for kk, v in out.items()}
    sig["band_rx"], sig["band_ry"] = np.asarray(brx), np.asarray(bry)
    return complete_sig(sig, n)


def complete_sig(sig, n):
    """Fill the auxiliary keys compute_metrics expects (perfect, noise-free estimator)."""
    m = n - 1
    for tag in ("a", "b"):
        for kk in ("tx", "ty", "rot", "logs"):
            sig[f"{kk}_{tag}"] = sig[kk]
    for kk in ("tx", "ty", "rot", "logs"):
        sig["sim_" + kk] = sig[kk]
        sig["ran_" + kk] = sig[kk]
    H = np.zeros((m, 9))
    for i in range(m):
        th, s = sig["rot"][i], math.exp(sig["logs"][i])
        H[i] = [s * math.cos(th), -s * math.sin(th), sig["tx"][i], s * math.sin(th), s * math.cos(th), sig["ty"][i], 0, 0, 1]
    sig["H"] = H
    for kk in ("se_tx", "se_ty", "res_rms", "res_med_all"):
        sig[kk] = np.zeros(m)
    for kk in ("n_points", "n_tracked", "n_inliers"):
        sig[kk] = np.full(m, 999.0)
    for kk in ("tx", "ty", "rot", "logs", "kx", "ky"):
        sig["d_" + kk] = np.concatenate([[np.nan], np.diff(sig[kk])])
    for tag in ("a", "b"):
        for kk in ("tx", "ty", "rot", "logs"):
            sig[f"d_{kk}_{tag}"] = sig["d_" + kk]
    for kk in ("band_rx", "band_ry"):
        sig["d_" + kk] = np.vstack([np.full((1, sig[kk].shape[1]), np.nan), np.diff(sig[kk], axis=0)])
    sig["n_delta"] = np.full(m, 999.0)
    sig["frame_luma"] = np.full(n, 100.0)
    sig["frame_sharp"] = np.full(n, 100.0)
    sig["frame_black"] = np.zeros(n)
    return sig


def truth_metrics(spec, frozen_translation=False):
    sig = truth_signals(spec, frozen_translation)
    n = len(sig["tx"]) + 1
    meta = {"info": {"fps": FPS}, "analysis_w": AN_W, "analysis_h": AN_H, "start": 0.0, "n_frames": n}
    m, _ = jm.compute_metrics(sig, meta)
    return m


# ---------------------------------------------------------------------------
# rendering / encoding
# ---------------------------------------------------------------------------
def clip_path(spec):
    return os.path.join(WORK, spec["name"] + ".mp4")


def render_clip(spec, force=False):
    path = clip_path(spec)
    if os.path.exists(path) and not force:
        return path
    r = make_renderer(spec)
    n = int(round(spec["dur"] * FPS))
    tmp = path + ".part.mp4"
    cmd = [jm.FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
           "-r", FPS_STR, "-i", "-", "-c:v", "libx264", "-preset", "faster", "-crf", "14",
           "-pix_fmt", "yuv420p", "-threads", "4", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(r.render(i / FPS).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed for " + spec["name"])
    os.replace(tmp, path)
    return path


def nominal(spec):
    j = spec.get("jit") or {}
    if j.get("type") in ("sine", "noise"):
        return {"hf_jitter_px": j["A"]}
    if j.get("type") == "roll":
        return {"hf_rot_deg": j["A_deg"] / math.sqrt(2)}
    return {"hf_jitter_px": 0.0}


def _summ(m):
    return {"hf_jitter_px": m["hf_jitter_px"], "hf_trans_px": m["hf_trans_px"], "hf_rot_deg": m["hf_rot_deg"],
            "hf_scale_pct": m["hf_scale_pct"],
            "band_2_8": m["bands"]["2-8Hz"]["combined_px"],
            "band_8_nyq": list(m["bands"].values())[1]["combined_px"],
            "jello_px": m["jello"]["jello_px"], "skew_px": m["jello"]["skew_px"],
            "stretch_px": m["jello"]["stretch_px"], "wobble_px": m["jello"]["wobble_px"],
            "stab_add_avg": m["stability_liu_additive"]["avg"]}


def validate_one(spec):
    path = render_clip(spec)
    res = jm.run(path, 0.0, None, plot=os.path.join(WORK, spec["name"] + "_metric.png"),
                 signals=os.path.join(WORK, spec["name"] + "_metric.npz"), verbose=False)
    meas = res["metrics"]
    row = {"name": spec["name"], "spec": spec, "nominal": nominal(spec),
           "truth": _summ(truth_metrics(spec)), "measured": _summ(meas),
           "noise_floor_px": meas["noise_floor_px"], "hf_jitter_px_denoised": meas["hf_jitter_px_denoised"],
           "ransac_model_hf_px": meas["hf_jitter_px_ransac_inlier_model"],
           "pair_method_hf_px": meas["hf_jitter_px_pair_method"], "estimator": meas["estimator"],
           "similarity_model_hf_px": meas["hf_jitter_px_similarity_model"],
           "tracking": meas["tracking"]}
    if spec["kind"] == "parallax":
        row["truth_rotation_only"] = _summ(truth_metrics(spec, frozen_translation=True))
    with open(os.path.join(WORK, spec["name"] + "_validation.json"), "w") as fh:
        json.dump(row, fh, indent=1, default=jm._json_default)
    return row


def report(rows):
    lines = ["| clip | nominal | truth | measured | meas/truth | noise floor (split-half) | measured, noise-corrected | 2-8 Hz t/m | 8+ Hz t/m | jello t/m | single-pair method | RANSAC-inlier model | similarity model |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        key = "hf_rot_deg" if "hf_rot_deg" in r["nominal"] else "hf_jitter_px"
        tr = r.get("truth_rotation_only", r["truth"])
        t, m = tr[key], r["measured"][key]
        ratio = m / t if t > 1e-3 else float("nan")
        lines.append(
            f"| {r['name']} | {r['nominal'][key]:.3f} | {t:.3f} | {m:.3f} | {ratio:.3f} | {r['noise_floor_px']:.3f} | "
            f"{(r['hf_jitter_px_denoised'] if key == 'hf_jitter_px' else float('nan')):.3f} | "
            f"{tr['band_2_8']:.3f}/{r['measured']['band_2_8']:.3f} | {tr['band_8_nyq']:.3f}/{r['measured']['band_8_nyq']:.3f} | "
            f"{r['truth']['jello_px']:.3f}/{r['measured']['jello_px']:.3f} | {r.get('pair_method_hf_px', float('nan')):.3f} | "
            f"{r['ransac_model_hf_px']:.3f} | {r['similarity_model_hf_px']:.3f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "validate", "report", "list"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    ensure_stills()
    S = [s for s in specs() if not a.only or any(o in s["name"] for o in a.only.split(","))]
    if a.cmd == "list":
        for s in S:
            print(s["name"])
        return
    if a.cmd == "render":
        with ProcessPoolExecutor(a.jobs) as ex:
            for p in ex.map(render_clip, S, [a.force] * len(S)):
                print(p)
        return
    if a.cmd == "validate":
        rows = []
        with ProcessPoolExecutor(a.jobs) as ex:
            for row in ex.map(validate_one, S):
                rows.append(row)
                key = "hf_rot_deg" if "hf_rot_deg" in row["nominal"] else "hf_jitter_px"
                tr = row.get("truth_rotation_only", row["truth"])
                print(f"{row['name']:32s} nominal {row['nominal'][key]:.3f} truth {tr[key]:.3f} "
                      f"measured {row['measured'][key]:.3f} noise {row['noise_floor_px']:.3f} "
                      f"jello t/m {row['truth']['jello_px']:.3f}/{row['measured']['jello_px']:.3f}", flush=True)
    rows = []
    for s in S:
        p = os.path.join(WORK, s["name"] + "_validation.json")
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    txt = report(rows)
    with open(os.path.join(WORK, "validation_table.md"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
