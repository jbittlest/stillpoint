"""
Paired comparison of two (or more) renders of the same clip/window measured by eval.jitter_metrics.

    .venv/bin/python -m eval.compare A.json B.json [C.json ...] [--names gyroflow,stillpoint]

Reports, per file: HF jitter (raw, noise-corrected, and rescaled to the ORIGINAL frame scale when
the run had --ref, i.e. angular-equivalent so a stabiliser cannot "win" by zooming out), bands,
jello, corner wobble, crop (exact source footprint for eval >= 2.0 JSONs; the full gate is eval.gate); and for each pair the ratio plus a per-1-s-window paired test (win rate and a
Wilcoxon signed-rank p-value on log window RMS).  The per-window test is the recommended
acceptance test for "Stillpoint beats Gyroflow": it needs the SAME --start/--dur for all runs.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy import stats


def load(p):
    r = json.load(open(p))
    m = r["metrics"]
    ref = r.get("ref", {})
    lin = ref.get("linear_crop_mean")
    b8 = [v for k, v in m["bands"].items() if k.startswith("8-")][0]
    return {
        "path": p, "video": r["video"], "start": r["start"], "n": r["meta"]["n_frames"],
        "hf": m["hf_jitter_px"], "hf_dn": m["hf_jitter_px_denoised"], "noise": m["noise_floor_px"],
        "hf_orig_scale": m["hf_jitter_px"] * lin if lin else float("nan"),
        "tx": m["hf_tx_px"], "ty": m["hf_ty_px"], "rot": m["hf_rot_deg"],
        "b28": m["bands"]["2-8Hz"]["combined_px"], "b8": b8["combined_px"],
        "jello": m["jello"]["jello_px"],
        # crop = EXACT source footprint (eval >= 2.0); falls back to the legacy homography area for old JSONs
        "crop": ref.get("footprint_area_mean", ref.get("visible_area_frac_mean", float("nan"))),
        "crop_legacy": ref.get("visible_area_frac_mean", float("nan")),
        "corner": m.get("corner", {}).get("corner_wobble_px", float("nan")),
        "code": r.get("code", {}).get("sha1", "pre-2.0"),
        "dv": ref.get("distortion_value_min", float("nan")),
        "win": np.asarray(m.get("window", {}).get("rms_px", []), float),
        "med": m.get("window", {}).get("median_px", float("nan")),
    }


def paired(a, b):
    n = min(len(a["win"]), len(b["win"]))
    if n < 3:
        return {}
    wa, wb = a["win"][:n], b["win"][:n]
    d = np.log(np.maximum(wb, 1e-6)) - np.log(np.maximum(wa, 1e-6))
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return {"windows": n, "b_better_frac": float((wb < wa).mean()), "median_ratio": float(np.exp(np.median(d))),
            "wilcoxon_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--names", default=None)
    a = ap.parse_args()
    rows = [load(p) for p in a.files]
    names = a.names.split(",") if a.names else [r["video"].split("/")[-1] for r in rows]
    starts = {round(r["start"], 3) for r in rows}
    if len(starts) > 1:
        print("WARNING: different --start values; per-window comparison is not paired:", starts)
    codes = {r["code"] for r in rows}
    if len(codes) > 1:
        print("WARNING: files were measured by different eval code versions:", codes)
    print("| name | HF px | HF px noise-corr | HF px @orig scale | tx | ty | roll deg | 2-8 Hz | 8-30 Hz | 1-s median | jello | "
          "corner wobble | crop (exact footprint) | dist | eval sha1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, r in zip(names, rows):
        print(f"| {nm} | {r['hf']:.3f} | {r['hf_dn']:.3f} | {r['hf_orig_scale']:.3f} | {r['tx']:.3f} | {r['ty']:.3f} | "
              f"{r['rot']:.4f} | {r['b28']:.3f} | {r['b8']:.3f} | {r['med']:.3f} | {r['jello']:.3f} | {r['corner']:.3f} | "
              f"{r['crop']:.4f} | {r['dv']:.3f} | {r['code']} |")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            pr = paired(rows[i], rows[j])
            ratio = rows[j]["hf"] / rows[i]["hf"]
            print(f"\n{names[j]} vs {names[i]}: HF ratio {ratio:.3f}"
                  + (f", orig-scale ratio {rows[j]['hf_orig_scale'] / rows[i]['hf_orig_scale']:.3f}"
                     if math.isfinite(rows[i]["hf_orig_scale"]) and math.isfinite(rows[j]["hf_orig_scale"]) else "")
                  + (f"; per-1s-window: {names[j]} better in {pr['b_better_frac'] * 100:.0f}% of {pr['windows']} windows, "
                     f"median ratio {pr['median_ratio']:.3f}, Wilcoxon p={pr['wilcoxon_p']:.2g}" if pr else ""))


if __name__ == "__main__":
    main()
