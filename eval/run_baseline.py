"""
Baseline: Jimmy's O3 originals vs his Gyroflow renders (and any other pair list).

    .venv/bin/python -m eval.run_baseline run   [--jobs 3] [--force]
    .venv/bin/python -m eval.run_baseline table
    .venv/bin/python -m eval.run_baseline gyro      # cross-check vs O3 gyro (needs work/o3/DJI_XXXX_imu.npz)

v2.0: measures the 5 gate windows AND the 5 held-out windows (eval.gate.GATE_WINDOWS / HELDOUT_WINDOWS) with the
current measurement code (sha1 recorded in every JSON; cached files are reused only when it matches).
Outputs: work/baseline/<clip>_<s>-<e>_{orig,gf}.{json,npz}; the gate windows are also copied to the legacy names
work/baseline/<clip>_{orig,gf}.{json,npz} (read by scripts/m1_*.py); work/baseline/baseline_table.md.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import jitter_metrics as jm
from .footage import GYROFLOW_DIR, O3_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "work", "baseline")
D4 = O3_DIR
D5 = GYROFLOW_DIR

# (clip, start_s, dur_s) -- windows chosen from gyro activity (see research/metrics.md)
CLIPS = [
    ("DJI_0025", 15.0, 25.0),
    ("DJI_0028", 8.0, 25.0),
    ("DJI_0034", 15.0, 25.0),
    ("DJI_0027", 5.0, 25.0),
    ("DJI_0032", 22.0, 25.0),
]


def pair_paths(clip):
    return os.path.join(D4, clip + ".MP4"), os.path.join(D5, clip + "_stabilized.mp4")


def _job(args):
    clip, start, dur, kind = args
    orig, gf = pair_paths(clip)
    base = os.path.join(OUT, f"{clip}_{kind}")
    if os.path.exists(base + ".json"):
        return base + ".json"
    if kind == "orig":
        res = jm.run(orig, start, dur, plot=base + ".png", signals=base + ".npz", verbose=False)
    else:
        res = jm.run(gf, start, dur, ref=orig, plot=base + ".png", signals=base + ".npz", verbose=False)
    with open(base + ".json", "w") as fh:
        json.dump(res, fh, indent=1, default=jm._json_default)
    return base + ".json"


def run(jobs=3, clips=None, force=False):
    """(Re)measure the originals and Gyroflow renders of all 10 windows with the current code (v2.0)."""
    from . import gate
    import shutil
    os.makedirs(OUT, exist_ok=True)
    wins = [w for w in gate.SETS["all"] if not clips or w[0] in clips]
    for p in gate.run_jobs(gate.baseline_jobs(wins, force), jobs):
        print(p, flush=True)
    for clip, s, d in gate.GATE_WINDOWS:       # legacy names for the gate windows
        if clips and clip not in clips:
            continue
        for kind in ("orig", "gf"):
            b = gate.base_path(clip, s, d, kind)
            for ext in (".json", ".npz"):
                if os.path.exists(b + ext):
                    shutil.copyfile(b + ext, os.path.join(OUT, f"{clip}_{kind}{ext}"))


def _row(m):
    b = m["bands"]
    b8 = [v for kk, v in b.items() if kk.startswith("8-")][0]
    return {
        "hf": m["hf_jitter_px"], "trans": m["hf_trans_px"], "tx": m["hf_tx_px"], "ty": m["hf_ty_px"],
        "rot": m["hf_rot_deg"], "scale": m["hf_scale_pct"], "b28": b["2-8Hz"]["combined_px"], "b8": b8["combined_px"],
        "b28_tx": b["2-8Hz"]["tx_px"], "b28_ty": b["2-8Hz"]["ty_px"], "b28_rot": b["2-8Hz"]["rot_deg"],
        "b8_tx": b8["tx_px"], "b8_ty": b8["ty_px"], "b8_rot": b8["rot_deg"],
        "med1s": m.get("window", {}).get("median_px", float("nan")),
        "p90": m.get("window", {}).get("p90_px", float("nan")),
        "corr_speed": m.get("window", {}).get("corr_jitter_vs_lf_speed", float("nan")),
        "calm": m.get("window", {}).get("median_px_calm", float("nan")),
        "active": m.get("window", {}).get("median_px_active", float("nan")),
        "noise": m["noise_floor_px"], "jello": m["jello"]["jello_px"], "skew": m["jello"]["skew_px"],
        "stretch": m["jello"]["stretch_px"], "wobble": m["jello"]["wobble_px"],
        "stab": m["stability_liu_additive"]["avg"], "stab_d": m["stability_liu_difrint"]["avg"],
        "peak": m["hf_peak_hz"], "pair": m["hf_jitter_px_pair_method"],
        "blur": m["blur_flicker_logvar"], "flick": m["flicker_luma_levels"],
        "corner": m.get("corner", {}).get("corner_wobble_px", float("nan")),
    }


def gated_paired(npz_ref, npz_other, thr_px_s=150.0, trim=30):
    """HF jitter (combined px) of two runs over the frames where the REFERENCE run's
    low-frequency camera speed (lp_speed, 1920-eq px/s incl. roll) is below thr_px_s."""
    a, b = np.load(npz_ref), np.load(npz_other)
    N = min(len(a["hp_tx"]), len(b["hp_tx"]))
    sp = a["lp_speed"]
    sp = np.concatenate([[sp[0]], sp])[:N]
    base = np.arange(trim, N - trim)
    idx = base[sp[base] < thr_px_s]
    if len(idx) < 60:
        return None
    W = float(b["analysis_w"]) if "analysis_w" in b.files else 960.0
    Hh = float(b["analysis_h"]) if "analysis_h" in b.files else W * 9 / 16
    R2 = (1920.0 ** 2 + (1920.0 * Hh / W) ** 2) / 12

    def comb(m):
        e = m["hp_tx"][idx] ** 2 + m["hp_ty"][idx] ** 2 + (m["hp_rot"][idx] ** 2 + m["hp_logs"][idx] ** 2) * R2
        return float(np.sqrt(e.mean()))
    return {"frac": len(idx) / len(base), "a": comb(a), "b": comb(b)}


def table():
    lines = []
    hdr = ("| clip | window | video | HF jitter px | trans px (x/y) | roll deg | 2-8 Hz px | 8-30 Hz px | "
           "1-s median / p90 px | noise floor | jello px (skew/stretch/wobble) | corner wobble px | Liu stab (add / DIFRINT) | "
           "crop: exact footprint (legacy homography) | distortion min | pairing lag | eval sha1 |")
    lines += [hdr, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    from . import gate
    for clip, s, d in gate.SETS["all"]:
        for kind in ("orig", "gf"):
            p = gate.base_path(clip, s, d, kind) + ".json"
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            m = r["metrics"]
            x = _row(m)
            rows[(clip, s, kind)] = (x, r)
            ref = r.get("ref", {})
            al = r.get("ref_alignment", {})
            crop = (f"{ref.get('footprint_area_mean', float('nan')):.4f} ({ref.get('visible_area_frac_mean', float('nan')):.3f})"
                    if ref else "-")
            dv = f"{ref['distortion_value_min']:.3f}" if "distortion_value_min" in ref else "-"
            lag = f"{al['best_lag_frames']} (ncc {al['peak_corr']:.3f} vs {al['next_best_corr']:.3f})" if "best_lag_frames" in al else "-"
            lines.append(
                f"| {clip} | {s:g}-{s + d:g} s | {'original' if kind == 'orig' else 'Gyroflow'} | **{x['hf']:.3f}** | "
                f"{x['trans']:.3f} ({x['tx']:.3f}/{x['ty']:.3f}) | {x['rot']:.4f} | {x['b28']:.3f} | {x['b8']:.3f} | "
                f"{x['med1s']:.3f} / {x['p90']:.3f} | {x['noise']:.3f} | {x['jello']:.3f} ({x['skew']:.3f}/{x['stretch']:.3f}/{x['wobble']:.3f}) | "
                f"{x['corner']:.3f} | {x['stab']:.3f} / {x['stab_d']:.3f} | {crop} | {dv} | {lag} | {r.get('code', {}).get('sha1', 'STALE')} |")
    # ratios
    lines.append("")
    lines.append("| clip | GF/orig HF | GF/orig tx | GF/orig ty | GF/orig roll | GF/orig 2-8 | GF/orig 8-30 | GF/orig jello | GF corr(jitter, LF speed) | GF calm / active 1-s median px |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for clip, s, d in gate.SETS["all"]:
        if (clip, s, "orig") in rows and (clip, s, "gf") in rows:
            o, g = rows[(clip, s, "orig")][0], rows[(clip, s, "gf")][0]
            lines.append(f"| {clip} {s:g}-{s + d:g} | {g['hf'] / o['hf']:.2f} | {g['tx'] / o['tx']:.2f} | {g['ty'] / o['ty']:.2f} | {g['rot'] / o['rot']:.2f} | "
                         f"{g['b28'] / o['b28']:.2f} | {g['b8'] / o['b8']:.2f} | {g['jello'] / o['jello']:.2f} | {g['corr_speed']:.2f} | "
                         f"{g['calm']:.3f} / {g['active']:.3f} |")
    # motion-gated, paired: frames where the ORIGINAL's low-frequency (<1 Hz) camera speed is
    # below a threshold (same frames for both videos) -> micro-jitter during cruising, with
    # intentional fast manoeuvres (flips, snap turns) excluded
    lines.append("")
    lines.append("| clip | gate (orig LF speed) | frames kept | original HF px | Gyroflow HF px | ratio |")
    lines.append("|---|---|---|---|---|---|")
    for clip, s, d in gate.SETS["all"]:
        po, pg = gate.base_path(clip, s, d, "orig") + ".npz", gate.base_path(clip, s, d, "gf") + ".npz"
        if not (os.path.exists(po) and os.path.exists(pg)):
            continue
        for thr in (150, 300):
            r = gated_paired(po, pg, thr)
            if r:
                lines.append(f"| {clip} {s:g}-{s + d:g} | < {thr} px/s | {r['frac'] * 100:.0f}% | {r['a']:.3f} | {r['b']:.3f} | {r['b'] / r['a']:.2f} |")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "baseline_table.md"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    return rows


def gyro_check():
    """Real-footage validation + residual diagnosis using the O3 gyro.

    IMU npz files (work/o3/DJI_XXXX_imu.npz: gyro_t [s, video timeline], gyro_cam [rad/s],
    2 kHz) were produced by a parallel investigator; axis order differs between clips, so a
    full 3x3 linear regression is used.  For every clip and video (orig / Gyroflow):
      * lag: time offset maximising R^2 of the ORIGINAL (then reused for the render),
      * R2_angle: variance of the metric's high-passed (>2 Hz) motion [tx, ty, roll*r]
        explained by the high-passed gyro angle (3 regressors),
      * R2_angle_rate: same with the gyro rate added (6 regressors; a timing error between a
        correction and the image leaves a residual ~ delta_t * rate),
      * per band (2-8 Hz, 8-30 Hz) R2, and the RMS (combined px) of the gyro-predictable
        part and of the unexplained part.
    For the original, R2 ~ 1 validates the metric (it measures real camera rotation).  For
    the Gyroflow render, the gyro-predictable part is residual jitter that a better
    gyro->image calibration (IMU orientation / gain / time offset / RS / lens) could remove;
    the unexplained part needs vision (or is translation/parallax)."""
    from scipy import signal as sps
    out = {}
    for clip, s, d in CLIPS:
        imu = os.path.join(ROOT, "work", "o3", f"{clip}_imu.npz")
        if not os.path.exists(imu):
            continue
        g = np.load(imu, allow_pickle=True)
        gt, w = g["gyro_t"], g["gyro_cam"]
        dt = np.diff(gt, prepend=gt[0] - np.median(np.diff(gt)))
        ang = np.cumsum(w * dt[:, None], axis=0)
        res = {}
        lag = None
        for kind in ("orig", "gf"):
            p = os.path.join(OUT, f"{clip}_{kind}.npz")
            if not os.path.exists(p):
                continue
            m = np.load(p)
            fps = float(m["fps"]); st = float(m["start"])
            N = len(m["path_tx"]); t = st + np.arange(N) / fps
            F = jm.Filters(fps, 2.0)
            r = float(np.sqrt((1920.0 ** 2 + (1920.0 * 9 / 16) ** 2) / 12))
            sl = slice(60, N - 60)
            Y = np.column_stack([m["hp_tx"], m["hp_ty"], m["hp_rot"] * r])

            def regs(lag_ms, rate=False):
                a = np.stack([F.highpass(np.interp(t + lag_ms / 1000, gt, ang[:, k])) for k in range(3)], 1)
                if rate:
                    ra = np.gradient(a, axis=0) * fps
                    a = np.column_stack([a, ra * 0.01])
                return a

            def r2(X, Yy):
                c, *_ = np.linalg.lstsq(X, Yy, rcond=None)
                e = Yy - X @ c
                return 1 - (e ** 2).sum() / ((Yy - Yy.mean(0)) ** 2).sum(), X @ c, e

            if kind == "orig":
                best = max(range(-20, 41, 2), key=lambda L: r2(regs(L)[sl], Y[sl])[0])
                lag = best
            rr = {"lag_ms": lag}
            R2a, fit_a, e_a = r2(regs(lag)[sl], Y[sl])
            R2b, fit_b, e_b = r2(regs(lag, True)[sl], Y[sl])
            rr["R2_angle"] = float(R2a)
            rr["R2_angle_rate"] = float(R2b)
            rr["rms_total_px"] = float(np.sqrt((Y[sl] ** 2).sum(1).mean()))
            rr["rms_gyro_predictable_px"] = float(np.sqrt((fit_b ** 2).sum(1).mean()))
            rr["rms_unexplained_px"] = float(np.sqrt((e_b ** 2).sum(1).mean()))
            for name, (lo, hi) in (("2-8Hz", (2, 8)), ("8-30Hz", (8, None))):
                sos = sps.butter(4, [lo, hi] if hi else lo, "bandpass" if hi else "highpass", fs=fps, output="sos")
                Yb = sps.sosfiltfilt(sos, Y, axis=0)[sl]
                Xb = sps.sosfiltfilt(sos, regs(lag, True), axis=0)[sl]
                R2x, fx_, ex_ = r2(Xb, Yb)
                rr[name] = {"R2": float(R2x), "rms_px": float(np.sqrt((Yb ** 2).sum(1).mean())),
                            "rms_unexplained_px": float(np.sqrt((ex_ ** 2).sum(1).mean()))}
            per_axis = {}
            for j, name in enumerate(("tx", "ty", "roll")):
                R2j, _, _ = r2(regs(lag, True)[sl], Y[sl, j])
                per_axis[name] = float(R2j)
            rr["R2_per_axis"] = per_axis
            res[kind] = rr
            print(f"{clip} {kind:4s} lag {lag:+d}ms R2(angle) {R2a:.3f} R2(angle+rate) {R2b:.3f} "
                  f"total {rr['rms_total_px']:.3f}px predictable {rr['rms_gyro_predictable_px']:.3f} "
                  f"unexplained {rr['rms_unexplained_px']:.3f} | 2-8Hz R2 {rr['2-8Hz']['R2']:.2f} "
                  f"8-30Hz R2 {rr['8-30Hz']['R2']:.2f} unexpl {rr['8-30Hz']['rms_unexplained_px']:.3f}/{rr['8-30Hz']['rms_px']:.3f} "
                  f"| per-axis {', '.join(f'{k} {v:.2f}' for k, v in per_axis.items())}")
        out[clip] = res
    with open(os.path.join(OUT, "gyro_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "table", "gyro"])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--clips", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.jobs, a.clips.split(",") if a.clips else None, a.force)
        table()
    elif a.cmd == "table":
        table()
    else:
        gyro_check()


if __name__ == "__main__":
    main()
