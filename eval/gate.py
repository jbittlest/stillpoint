"""
Official Stillpoint gate: original vs Gyroflow render vs Stillpoint render(s) on one window.

    # one window (any clip); SP renders are window renders starting at the window's first source frame
    PYTHONPATH=engine .venv/bin/python -m eval.gate window --clip DJI_0025 --start 15 --dur 25 \
        --sp stillpoint=work/m1/DJI_0025_stillpoint.mov [--sp other=path.mov@0] [--out work/gate/m1]
    # the 5 gate windows / the 5 held-out windows / all 10, SP renders from a pattern
    PYTHONPATH=engine .venv/bin/python -m eval.gate set gate    --sp-pattern 'work/m1/{clip}_stillpoint.mov'
    PYTHONPATH=engine .venv/bin/python -m eval.gate set heldout --sp-pattern 'work/verify/metric/{clip}_{s}_sp.mov'
    # regenerate the cached original/Gyroflow baselines (work/baseline) with the current measurement code
    PYTHONPATH=engine .venv/bin/python -m eval.gate baseline [--jobs 3] [--force]
    # whole-clip source footprint: Gyroflow render (fitted, sampled) vs a Stillpoint plan (exact)
    PYTHONPATH=engine .venv/bin/python -m eval.gate clipcrop --clip DJI_0034 [--plan work/m1/DJI_0034/plan.spplan]

Original and Gyroflow measurements are cached in work/baseline/<clip>_<s>-<e>_{orig,gf}.{json,npz} and reused only
when their recorded measurement-code sha1 equals the current one (eval.jitter_metrics.code_version); otherwise
they are recomputed.  Stillpoint renders are always measured (with --ref original: exact footprint + pairing).

Table (1080p-equivalent px; calm-cruise = frames where the ORIGINAL's <1 Hz camera speed < 150 px/s, same frames
for every video): HF (>2 Hz) | calm-cruise HF | 2-8 Hz | 8-30 Hz | roll deg | 8-30 Hz roll | jello | row wobble
| corner wobble (paired: frames/corners measured in both SP and GF) | Stillpoint-only jumps >0.5 / >1 px (not
present in the Gyroflow render or the original, eval.events) | exact crop = source footprint (paired sampled
frames) | pairing lag | paired per-1-s win-rate + Wilcoxon p.

OFFICIAL GATE (all must pass):
  calm-cruise HF   <= max(0.15 px, 0.5 x Gyroflow's calm-cruise)   (forward-motion scale floor ~0.2 px on 0025/0027)
  8-30 Hz          <= original's
  win-rate vs GF   >= 90 % of 1-s windows
  crop footprint   >= Gyroflow's - 0.005 (0.5 pp of the source area)
  jumps            zero Stillpoint-only jumps > 1 px
  jello, wobble    <= 1.05 x Gyroflow's (jello_px; corner wobble = median over 1-s windows of the corner HF
                   residual on the frames x corners measured in both renders -- RMS reported alongside)
  pairing          render frames exactly aligned with the original (lag 0)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import jitter_metrics as jm
from .footage import GYROFLOW_DIR, O3_DIR
from .compare import paired as paired_windows
from .events import stillpoint_only_jumps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "work", "baseline")
D4 = O3_DIR
D5 = GYROFLOW_DIR
CALM_THR = 150.0

GATE_WINDOWS = [("DJI_0025", 15.0, 25.0), ("DJI_0028", 8.0, 25.0), ("DJI_0034", 15.0, 25.0),
                ("DJI_0027", 5.0, 25.0), ("DJI_0032", 22.0, 25.0)]
HELDOUT_WINDOWS = [("DJI_0025", 1.0, 14.0), ("DJI_0028", 38.0, 20.0), ("DJI_0034", 1.0, 14.0),
                   ("DJI_0027", 100.0, 20.0), ("DJI_0032", 120.0, 20.0)]
SETS = {"gate": GATE_WINDOWS, "heldout": HELDOUT_WINDOWS, "all": GATE_WINDOWS + HELDOUT_WINDOWS}

RULES = {"calm_abs_px": 0.15, "calm_rel_gf": 0.5, "winrate": 0.90, "crop_tol": 0.005, "jumps_gt_px": 1.0,
         "jello_rel": 1.05, "wobble_rel": 1.05}


def paths(clip):
    return os.path.join(D4, clip + ".MP4"), os.path.join(D5, clip + "_stabilized.mp4")


def wtag(s, d):
    return f"{s:g}-{s + d:g}"


def base_path(clip, s, d, kind):
    return os.path.join(BASE, f"{clip}_{wtag(s, d)}_{kind}")


# ------------------------------------------------------------------------------------------------ measuring
def _measure(job):
    """job: (video, start, dur, ref, ref_start, out_base) -> out_base (json + npz written)."""
    video, start, dur, ref, ref_start, out_base = job
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    res = jm.run(video, start, dur, ref=ref, ref_start=ref_start, signals=out_base + ".npz", plot=None, verbose=False)
    with open(out_base + ".json.part", "w") as fh:
        json.dump(res, fh, indent=1, default=jm._json_default)
    os.replace(out_base + ".json.part", out_base + ".json")
    return out_base


def cache_valid(out_base, video, start, dur):
    p = out_base + ".json"
    if not (os.path.exists(p) and os.path.exists(out_base + ".npz")):
        return False
    try:
        r = json.load(open(p))
    except Exception:
        return False
    try:
        newer = os.path.getmtime(p) >= os.path.getmtime(video)      # the video was not re-rendered since
    except OSError:
        return False
    return (newer and r.get("code", {}).get("sha1") == jm.code_version()["sha1"]
            and os.path.abspath(r.get("video", "")) == os.path.abspath(video)
            and abs(float(r.get("start", -1)) - start) < 1e-6 and abs(float(r.get("dur") or -1) - dur) < 1e-6)


def baseline_jobs(windows, force=False):
    jobs = []
    for clip, s, d in windows:
        orig, gf = paths(clip)
        bo, bg = base_path(clip, s, d, "orig"), base_path(clip, s, d, "gf")
        if force or not cache_valid(bo, orig, s, d):
            jobs.append((orig, s, d, None, None, bo))
        if force or not cache_valid(bg, gf, s, d):
            jobs.append((gf, s, d, orig, s, bg))
    return jobs


def run_jobs(jobs, n=3):
    if not jobs:
        return []
    if n <= 1 or len(jobs) == 1:
        return [_measure(j) for j in jobs]
    with ProcessPoolExecutor(n) as ex:
        return list(ex.map(_measure, jobs))


# ------------------------------------------------------------------------------------------------ comparisons
def _load(base):
    return json.load(open(base + ".json")), np.load(base + ".npz")


def calm_cruise(z_orig, z, thr=CALM_THR, trim=30):
    """HF (>2 Hz) combined px of video z over the frames where the ORIGINAL's <1 Hz speed < thr px/s."""
    N = min(len(z_orig["hp_tx"]), len(z["hp_tx"]))
    sp = np.asarray(z_orig["lp_speed"], float)
    sp = np.concatenate([[sp[0]], sp])[:N]
    base = np.arange(trim, N - trim)
    idx = base[sp[base] < thr]
    if len(idx) < 60:
        return float("nan"), 0.0
    W = float(z["analysis_w"])
    H = float(z["analysis_h"]) if "analysis_h" in z.files else W * 9 / 16
    R2 = (1920.0 ** 2 + (1920.0 * H / W) ** 2) / 12
    e = z["hp_tx"][idx] ** 2 + z["hp_ty"][idx] ** 2 + (z["hp_rot"][idx] ** 2 + z["hp_logs"][idx] ** 2) * R2
    return float(np.sqrt(e.mean())), len(idx) / len(base)


def paired_corner(z_a, z_b, fps=60000 / 1001):
    """Corner wobble of a and b over the frames x corners measured in BOTH.  Returns dict with
    rms_a/rms_b (pooled RMS, px), med_a/med_b (median over 1-s windows of the pooled RMS -- the gate statistic:
    a stabiliser defect such as a lens-model error shows in every second, while a few violent seconds with
    near-field parallax / tracking failure dominate the RMS), frac (share of frame x corner cells used)."""
    nanres = {"rms_a": float("nan"), "rms_b": float("nan"), "med_a": float("nan"), "med_b": float("nan"), "frac": 0.0}
    if "corner_e" not in z_a.files or "corner_e" not in z_b.files:
        return nanres
    a, b = np.asarray(z_a["corner_e"], float), np.asarray(z_b["corner_e"], float)
    n = min(len(a), len(b))
    trim = 30
    a, b = a[trim:n - trim], b[trim:n - trim]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return dict(nanres, frac=float(m.mean()) if m.size else 0.0)
    wl = max(int(round(fps)), 4)
    wa, wb = [], []
    for i in range(len(a) // wl):
        mm = m[i * wl:(i + 1) * wl]
        if mm.sum() >= 0.5 * wl:        # >= half a second x 1 corner measured in both
            wa.append(math.sqrt(a[i * wl:(i + 1) * wl][mm].mean()))
            wb.append(math.sqrt(b[i * wl:(i + 1) * wl][mm].mean()))
    return {"rms_a": float(np.sqrt(a[m].mean())), "rms_b": float(np.sqrt(b[m].mean())),
            "med_a": float(np.median(wa)) if wa else float("nan"), "med_b": float(np.median(wb)) if wb else float("nan"),
            "windows": len(wa), "frac": float(m.mean())}


def paired_footprint(r_a, r_b):
    """Mean exact footprint of a and b over the sampled frames both measured (same frame indices)."""
    fa = {x["frame"]: x["area"] for x in r_a.get("ref", {}).get("footprint_rows", [])
          if "area" in x and x.get("resid_median_px", 9) <= 0.8 and x.get("n_inliers", 0) >= 40}
    fb = {x["frame"]: x["area"] for x in r_b.get("ref", {}).get("footprint_rows", [])
          if "area" in x and x.get("resid_median_px", 9) <= 0.8 and x.get("n_inliers", 0) >= 40}
    common = sorted(set(fa) & set(fb))
    if not common:
        return float("nan"), float("nan"), 0
    return float(np.mean([fa[k] for k in common])), float(np.mean([fb[k] for k in common])), len(common)


def _row(r, z, z_orig):
    m = r["metrics"]
    b8 = [v for k, v in m["bands"].items() if k.startswith("8-")][0]
    calm, calm_frac = calm_cruise(z_orig, z)
    ref = r.get("ref", {})
    al = r.get("ref_alignment", {})
    return {
        "video": r["video"], "start": r["start"], "dur": r["dur"], "n_frames": r["meta"]["n_frames"],
        "code_sha1": r.get("code", {}).get("sha1"),
        "hf": m["hf_jitter_px"], "hf_denoised": m["hf_jitter_px_denoised"], "noise": m["noise_floor_px"],
        "calm": calm, "calm_frac": calm_frac,
        "b28": m["bands"]["2-8Hz"]["combined_px"], "b830": b8["combined_px"],
        "roll_deg": m["hf_rot_deg"], "roll830_deg": b8["rot_deg"],
        "jello": m["jello"]["jello_px"], "row_wobble": m["jello"]["wobble_px"],
        "corner_wobble": m.get("corner", {}).get("corner_wobble_px", float("nan")),
        "corner_wobble_noise": m.get("corner", {}).get("corner_wobble_noise_px", float("nan")),
        "spikes_gt1": m.get("spikes", {}).get("n_gt_1px"),
        "med1s": m.get("window", {}).get("median_px", float("nan")),
        "footprint": ref.get("footprint_area_mean", float("nan")),
        "footprint_n": ref.get("footprint_n", 0),
        "focal_1920": [ref.get("footprint_focal_1920_min"), ref.get("footprint_focal_1920_max")],
        "legacy_crop": ref.get("visible_area_frac_mean", float("nan")),
        "lag": al.get("best_lag_frames"), "lag_agreement": al.get("agreement"),
    }


def evaluate_window(clip, s, d, sps: dict, out_dir, orig=None, gf=None, gf_start=None, jobs=3, plan=None,
                    reuse_sp=True):
    """sps: name -> (render path, render start s).  Returns the gate dict (and writes json + md)."""
    t0 = time.time()
    o_def, g_def = paths(clip)
    orig, gf = orig or o_def, gf or g_def
    gf_start = s if gf_start is None else gf_start
    os.makedirs(out_dir, exist_ok=True)
    bo, bg = base_path(clip, s, d, "orig"), base_path(clip, s, d, "gf")
    todo = []
    if not cache_valid(bo, orig, s, d):
        todo.append((orig, s, d, None, None, bo))
    if not cache_valid(bg, gf, gf_start, d):
        todo.append((gf, gf_start, d, orig, s, bg))
    sp_bases = {}
    for nm, (p, st) in sps.items():
        b = os.path.join(out_dir, f"{clip}_{wtag(s, d)}_{nm}")
        sp_bases[nm] = b
        if not (reuse_sp and cache_valid(b, p, st, d)):
            todo.append((p, st, d, orig, s, b))
    run_jobs(todo, jobs)
    ro, zo = _load(bo)
    rg, zg = _load(bg)
    rows = {"original": _row(ro, zo, zo), "gyroflow": _row(rg, zg, zo)}
    gate = {"clip": clip, "start": s, "dur": d, "window": wtag(s, d), "rules": RULES,
            "code": jm.code_version(), "rows": rows, "stillpoint": {}}
    dev_o, dev_g = np.asarray(zo["jump_dev"]), np.asarray(zg["jump_dev"])
    fps = float(zo["fps"])
    for nm, b in sp_bases.items():
        rs, zs = _load(b)
        row = _row(rs, zs, zo)
        rows[nm] = row
        pw_g = paired_windows(_cmp_load(rg), _cmp_load(rs))
        pw_o = paired_windows(_cmp_load(ro), _cmp_load(rs))
        cwp = paired_corner(zs, zg, fps)
        cw_sp, cw_gf = cwp["med_a"], cwp["med_b"]
        fp_sp, fp_gf, fp_n = paired_footprint(rs, rg)
        jumps = stillpoint_only_jumps(np.asarray(zs["jump_dev"]), {"gyroflow": dev_g, "original": dev_o}, fps, 0.0)
        g = rows["gyroflow"]
        o = rows["original"]
        calm_lim = max(RULES["calm_abs_px"], RULES["calm_rel_gf"] * g["calm"])
        checks = {
            "calm_cruise": {"value": row["calm"], "limit": calm_lim, "pass": row["calm"] <= calm_lim},
            "8-30Hz_vs_original": {"value": row["b830"], "limit": o["b830"], "pass": row["b830"] <= o["b830"]},
            "winrate_vs_gyroflow": {"value": pw_g.get("b_better_frac", float("nan")), "limit": RULES["winrate"],
                                    "pass": pw_g.get("b_better_frac", 0) >= RULES["winrate"],
                                    "wilcoxon_p": pw_g.get("wilcoxon_p"), "windows": pw_g.get("windows")},
            "crop_footprint": {"value": fp_sp, "gyroflow": fp_gf, "limit": fp_gf - RULES["crop_tol"], "frames": fp_n,
                               "pass": bool(fp_n > 0 and fp_sp >= fp_gf - RULES["crop_tol"])},
            "jumps_gt_1px": {"value": jumps["only_n_gt_1px"], "max_px": jumps["only_max_px"], "limit": 0,
                             "pass": jumps["only_n_gt_1px"] == 0},
            "jello": {"value": row["jello"], "limit": RULES["jello_rel"] * g["jello"],
                      "pass": row["jello"] <= RULES["jello_rel"] * g["jello"]},
            "corner_wobble": {"value": cw_sp, "gyroflow": cw_gf, "limit": RULES["wobble_rel"] * cw_gf,
                              "statistic": "median over 1-s windows of the corner HF residual, frames x corners "
                                           "measured in both", "rms_sp": cwp["rms_a"], "rms_gf": cwp["rms_b"],
                              "windows": cwp.get("windows"), "paired_frac": cwp["frac"],
                              "pass": bool(math.isfinite(cw_sp) and cw_sp <= RULES["wobble_rel"] * cw_gf)},
            "pairing": {"value": row["lag"], "limit": 0, "pass": row["lag"] == 0},
        }
        entry = {"checks": checks, "pass": all(c["pass"] for c in checks.values()),
                 "winrate_vs_original": pw_o, "winrate_vs_gyroflow": pw_g,
                 "jumps": {k: v for k, v in jumps.items() if k != "events"},
                 "jump_events": [e for e in jumps["events"] if e["only"] or e["px"] > 1.0][:20],
                 "row_wobble_ratio_vs_gf": row["row_wobble"] / g["row_wobble"] if g["row_wobble"] else float("nan")}
        if plan:
            entry["plan_footprint_exact"] = _plan_fp(plan, rs)
        gate["stillpoint"][nm] = entry
    gate["elapsed_s"] = time.time() - t0
    base = os.path.join(out_dir, f"gate_{clip}_{wtag(s, d)}")
    with open(base + ".json", "w") as fh:
        json.dump(gate, fh, indent=1, default=jm._json_default)
    md = markdown(gate)
    with open(base + ".md", "w") as fh:
        fh.write(md + "\n")
    return gate


def _cmp_load(r):
    m = r["metrics"]
    return {"win": np.asarray(m.get("window", {}).get("rms_px", []), float)}


def _plan_fp(plan_path, r_sp):
    """Exact footprint from the plan at the SP render's sampled frames (cross-check of the fitted measure)."""
    from .footprint import _engine, plan_footprint
    _engine()
    from stillpoint.plan_io import read_plan
    from stillpoint.pipeline import first_frame_at
    plan = read_plan(plan_path)
    k0 = first_frame_at(plan.frame_pts, float(r_sp["ref"]["ref_start"]))
    fr = [x["frame"] for x in r_sp["ref"].get("footprint_rows", [])]
    a = plan_footprint(plan, [k0 + f for f in fr])
    return {"mean": float(a.mean()), "n": int(len(a))}


# ------------------------------------------------------------------------------------------------ markdown
def _f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return "-" if not math.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def markdown(g):
    L = [f"### {g['clip']} {g['window']} s  (eval {g['code']['eval_version']} / {g['code']['sha1']})", "",
         "| video | HF px | calm-cruise HF | 2-8 Hz | 8-30 Hz | roll deg | 8-30 roll deg | jello | row wobble | "
         "corner wobble | spikes >1px (self) | crop footprint | pairing lag | 1-s median |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, r in g["rows"].items():
        L.append(f"| {nm} | {_f(r['hf'])} | {_f(r['calm'])} | {_f(r['b28'])} | {_f(r['b830'])} | {_f(r['roll_deg'], 4)} | "
                 f"{_f(r['roll830_deg'], 4)} | {_f(r['jello'])} | {_f(r['row_wobble'])} | {_f(r['corner_wobble'])} | "
                 f"{_f(r['spikes_gt1'])} | {_f(r['footprint'], 4)} | {_f(r['lag'])} | {_f(r['med1s'])} |")
    L.append("")
    L.append("| stillpoint | calm-cruise | 8-30 Hz <= orig | win-rate vs GF (p) | crop >= GF-0.5pp | SP-only jumps >1px (>0.5) | "
             "jello <= 1.05 GF | corner wobble <= 1.05 GF (paired 1-s median) | lag 0 | win-rate vs orig | GATE |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    P = lambda c: "PASS" if c["pass"] else "FAIL"
    for nm, e in g["stillpoint"].items():
        c = e["checks"]
        L.append(
            f"| {nm} | {P(c['calm_cruise'])} {_f(c['calm_cruise']['value'])} (<= {_f(c['calm_cruise']['limit'])}) | "
            f"{P(c['8-30Hz_vs_original'])} {_f(c['8-30Hz_vs_original']['value'])} vs {_f(c['8-30Hz_vs_original']['limit'])} | "
            f"{P(c['winrate_vs_gyroflow'])} {_f(100 * c['winrate_vs_gyroflow']['value'], 0)}% of {c['winrate_vs_gyroflow']['windows']} "
            f"(p={_f(c['winrate_vs_gyroflow']['wilcoxon_p'], 4)}) | "
            f"{P(c['crop_footprint'])} {_f(c['crop_footprint']['value'], 4)} vs {_f(c['crop_footprint']['gyroflow'], 4)} | "
            f"{P(c['jumps_gt_1px'])} {c['jumps_gt_1px']['value']} ({e['jumps']['only_n_gt_0.5px']}), max {_f(c['jumps_gt_1px']['max_px'], 2)} | "
            f"{P(c['jello'])} {_f(c['jello']['value'])} vs {_f(c['jello']['limit'] / RULES['jello_rel'])} | "
            f"{P(c['corner_wobble'])} {_f(c['corner_wobble']['value'])} vs {_f(c['corner_wobble']['gyroflow'])} "
            f"(rms {_f(c['corner_wobble']['rms_sp'], 2)} vs {_f(c['corner_wobble']['rms_gf'], 2)}) | "
            f"{P(c['pairing'])} {c['pairing']['value']} | "
            f"{_f(100 * e['winrate_vs_original'].get('b_better_frac', float('nan')), 0)}% | **{'PASS' if e['pass'] else 'FAIL'}** |")
    for nm, e in g["stillpoint"].items():
        if e["jump_events"]:
            L.append("")
            L.append(f"{nm} jumps (pair -> frame after, window time, px, only):  " + "; ".join(
                f"{j['frame_after']} @{j['t_s']:.2f}s {j['px']:.2f}px{' ONLY' if j['only'] else ''}" for j in e["jump_events"][:10]))
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ whole-clip crop
def clip_crop(clip, plan_path=None, step_s=1.0, force=False):
    """Whole-clip source footprint: Gyroflow render vs original (fitted, every step_s) cached in work/baseline,
    and optionally a Stillpoint plan (exact, same frames)."""
    from .footprint import footprint_series, plan_footprint, _engine
    orig, gf = paths(clip)
    out = os.path.join(BASE, f"{clip}_wholeclip_gf_footprint.json")
    step = max(int(round(step_s * jm.probe(gf).fps)), 1)
    r = None
    if os.path.exists(out) and not force:
        r = json.load(open(out))
        if r.get("code", {}).get("sha1") != jm.code_version()["sha1"] or r.get("step_frames") != step:
            r = None
    if r is None:
        r = footprint_series(gf, orig, 0.0, 0.0, None, step, verbose=True)
        r["code"] = jm.code_version()
        r["clip"] = clip
        with open(out, "w") as fh:
            json.dump(r, fh, indent=1, default=float)
    res = {"clip": clip, "gyroflow_mean": r["mean"], "gyroflow_min": r["min"], "gyroflow_n": r["n"],
           "gyroflow_focal_1920": [r["focal_1920_min"], r["focal_1920_median"], r["focal_1920_max"]]}
    if plan_path:
        _engine()
        from stillpoint.plan_io import read_plan
        plan = read_plan(plan_path)
        ok = [x["frame"] for x in r["rows"] if "area" in x and x["resid_median_px"] <= 0.8 and x["n_inliers"] >= 40]
        ok = [f for f in ok if f < plan.n_frames]
        a = plan_footprint(plan, ok)
        g = np.array([x["area"] for x in r["rows"] if x["frame"] in set(ok)])
        allf = plan_footprint(plan, np.arange(0, plan.n_frames, step))
        res.update(stillpoint_mean_paired=float(a.mean()), gyroflow_mean_paired=float(g.mean()), n_paired=len(ok),
                   stillpoint_mean_all=float(allf.mean()), stillpoint_min=float(allf.min()),
                   diff_pp=100 * float(a.mean() - g.mean()))
    return res


# ------------------------------------------------------------------------------------------------ CLI
def _parse_sp(items, default_start=0.0):
    out = {}
    for it in items:
        nm, rest = it.split("=", 1) if "=" in it else ("stillpoint", it)
        p, st = (rest.rsplit("@", 1) + [None])[:2] if "@" in rest else (rest, None)
        out[nm] = (p, float(st) if st is not None else default_start)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("window")
    w.add_argument("--clip", required=True)
    w.add_argument("--start", type=float, required=True)
    w.add_argument("--dur", type=float, required=True)
    w.add_argument("--sp", action="append", default=[], help="name=render.mov[@render_start] (repeatable)")
    w.add_argument("--orig", default=None)
    w.add_argument("--gyroflow", default=None)
    w.add_argument("--gf-start", type=float, default=None)
    w.add_argument("--plan", default=None, help="Stillpoint plan: also report its exact footprint")
    w.add_argument("--out", default=os.path.join(ROOT, "work", "gate"))
    w.add_argument("--jobs", type=int, default=3)
    st = sub.add_parser("set")
    st.add_argument("which", choices=list(SETS))
    st.add_argument("--sp-pattern", action="append", default=[],
                    help="[name=]pattern with {clip} and {s} (window start, %%g), e.g. work/m1/{clip}_stillpoint.mov")
    st.add_argument("--out", default=os.path.join(ROOT, "work", "gate"))
    st.add_argument("--jobs", type=int, default=3)
    b = sub.add_parser("baseline")
    b.add_argument("--which", choices=list(SETS), default="all")
    b.add_argument("--jobs", type=int, default=3)
    b.add_argument("--force", action="store_true")
    c = sub.add_parser("clipcrop")
    c.add_argument("--clip", required=True)
    c.add_argument("--plan", default=None)
    c.add_argument("--step-s", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.cmd == "window":
        g = evaluate_window(a.clip, a.start, a.dur, _parse_sp(a.sp), a.out, a.orig, a.gyroflow, a.gf_start, a.jobs, a.plan)
        print(markdown(g))
    elif a.cmd == "set":
        mds = []
        plan_sets = {}
        for clip, s, d in SETS[a.which]:
            sps = {}
            for pat in a.sp_pattern:
                nm, pp = pat.split("=", 1) if "=" in pat and not pat.startswith("/") and "=" in pat.split("/")[0] else ("stillpoint", pat)
                p = pp.format(clip=clip, s=f"{s:g}")
                if os.path.exists(p):
                    sps[nm] = (p, 0.0)
                else:
                    print(f"missing {p}", file=sys.stderr)
            plan_sets[(clip, s, d)] = sps
        # measure everything missing in one pool (baselines + all renders), then assemble per window
        todo = baseline_jobs(SETS[a.which])
        for (clip, s, d), sps in plan_sets.items():
            orig = paths(clip)[0]
            for nm, (p, st) in sps.items():
                b = os.path.join(a.out, f"{clip}_{wtag(s, d)}_{nm}")
                if not cache_valid(b, p, st, d):
                    todo.append((p, st, d, orig, s, b))
        print(f"{len(todo)} measurements to run", file=sys.stderr, flush=True)
        run_jobs(todo, a.jobs)
        for (clip, s, d), sps in plan_sets.items():
            g = evaluate_window(clip, s, d, sps, a.out, jobs=a.jobs)
            mds.append(markdown(g))
            print(mds[-1], flush=True)
        with open(os.path.join(a.out, f"gate_{a.which}.md"), "w") as fh:
            fh.write("\n\n".join(mds) + "\n")
    elif a.cmd == "baseline":
        jobs = baseline_jobs(SETS[a.which], a.force)
        print(f"{len(jobs)} baseline measurements to run", flush=True)
        for p in run_jobs(jobs, a.jobs):
            print(p, flush=True)
    elif a.cmd == "clipcrop":
        print(json.dumps(clip_crop(a.clip, a.plan, a.step_s), indent=1))


if __name__ == "__main__":
    main()
