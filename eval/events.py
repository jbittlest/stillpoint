"""
Single-frame JUMP events that one render has and its references do not.

    .venv/bin/python -m eval.events TEST.npz --ref gyroflow=GF.npz --ref original=ORIG.npz [--start S]

Input: the per-pair spike-deviation series `jump_dev` that eval.jitter_metrics (v2.0+) stores in its signal NPZ
(deviation of each pair's velocity [tx, ty, roll*rho, logscale*rho], 1920-eq px, from the median of its +-3
neighbours; a lasting position step at frame k+1 is a one-pair impulse at pair k, a one-frame displacement is
two adjacent impulses).  All NPZs must cover the SAME frames (same window start in source frames; renders of
the same original are frame-exact, see eval.footprint.check_alignment).

An event of the test video (|dev| > 0.5 px and >= 3x the local deviation level, i.e. isolated -- ongoing
vibration is not an event; adjacent pairs merged) at pair k with deviation vector d is PRESENT in a reference
if, within +-1 pair, the reference's deviation projected on d's direction reaches 0.5*|d| (a same-direction
spike of at least half the size) and stands out from the reference's own local deviation level by 3x.  A STILLPOINT-ONLY jump is present in none of the
references (the reference render and the original) -- a jump the stabiliser created.  `not_in[name]` counts
events missing from each reference individually (e.g. present in the original but removed by Gyroflow).

Validation: work/eval_fix/validate_events.py (synthetic clips with injected 0.6/1.5 px steps and a 1.2 px
one-frame spike under 0.25 px 3-20 Hz jitter: all found, magnitudes within 0.1 px, zero false events) and the
known M1 closed-loop jumps (DJI_0032 pairs 417/1282/1327, DJI_0025 pair 959).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .jitter_metrics import SPIKE_HALF, SPIKE_Z, local_scale, spike_events

THRESHOLDS = (0.5, 1.0)


def load_dev(npz_path):
    z = np.load(npz_path)
    if "jump_dev" not in z:
        raise KeyError(f"{npz_path}: no jump_dev (measured with eval < 2.0; re-run eval.jitter_metrics)")
    return np.asarray(z["jump_dev"], float), float(z["fps"]) if "fps" in z else 60000 / 1001, \
        float(z["start"]) if "start" in z else 0.0


def present_in(dev_ref, k, d, frac=0.5, tol=1, z=SPIKE_Z):
    """Is the test event (pair k, deviation vector d) present in a reference?  -> (projection px, bool).
    Present = within +-tol pairs the reference's deviation projected on d's direction reaches frac*|d| AND stands
    out from the reference's own local deviation level by z (a heavily vibrating original/reference does not
    'contain' every test spike by coincidence)."""
    n = np.linalg.norm(d)
    if n <= 0 or dev_ref is None:
        return 0.0, False
    u = d / n
    lo, hi = max(k - tol, 0), min(k + tol + 1, len(dev_ref))
    if hi <= lo:
        return 0.0, False
    proj = np.nan_to_num(dev_ref[lo:hi]) @ u
    p = float(proj.max())
    return p, bool(p >= frac * n and p >= z * local_scale(dev_ref, k))


def stillpoint_only_jumps(dev_test, refs: dict, fps=60000 / 1001, start=0.0, frac=0.5, tol=1, trim=SPIKE_HALF):
    """dev_test (n,4); refs: name -> dev (n',4).  Returns counts/max of test events > 0.5 and > 1 px that are
    present in none of the refs (+ per-reference 'not_in' counts and the event list)."""
    n = len(dev_test)
    for nm, r in refs.items():
        n = min(n, len(r))
    dt = np.asarray(dev_test, float)[:n]
    rr = {nm: np.asarray(r, float)[:n] for nm, r in refs.items()}
    events = [(k, m) for k, m in spike_events(dt, THRESHOLDS[0]) if trim <= k < n - trim]
    rows = []
    for k, mag in events:
        d = np.nan_to_num(dt[k])
        pres = {}
        for nm, r in rr.items():
            p, ok = present_in(r, k, d, frac, tol)
            pres[nm] = {"proj_px": round(p, 3), "present": bool(ok)}
        rows.append({"pair": int(k), "frame_after": int(k) + 1, "t_s": round(start + (k + 1) / fps, 4),
                     "px": round(float(mag), 3), "vec": [round(float(x), 3) for x in d],
                     "in": pres, "only": not any(v["present"] for v in pres.values())})
    out = {"n_pairs": int(n), "frac": frac, "tol_pairs": tol}
    for thr in THRESHOLDS:
        sel = [r for r in rows if r["px"] > thr]
        key = f"{thr:g}px"
        out[f"n_gt_{key}"] = len(sel)
        only = [r for r in sel if r["only"]]
        out[f"only_n_gt_{key}"] = len(only)
        out[f"only_max_px_gt_{key}"] = max([r["px"] for r in only], default=0.0)
        for nm in rr:
            out[f"not_in_{nm}_n_gt_{key}"] = sum(1 for r in sel if not r["in"][nm]["present"])
    out["only_max_px"] = max([r["px"] for r in rows if r["only"]], default=0.0)
    out["events"] = sorted(rows, key=lambda r: -r["px"])[:40]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test")
    ap.add_argument("--ref", action="append", default=[], help="name=path.npz (repeatable)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dev, fps, start = load_dev(a.test)
    refs = {}
    for s in a.ref:
        nm, p = s.split("=", 1)
        refs[nm] = load_dev(p)[0]
    r = stillpoint_only_jumps(dev, refs, fps, start)
    print(f"test-only jumps: >0.5 px {r['only_n_gt_0.5px']}  >1 px {r['only_n_gt_1px']}  max {r['only_max_px']:.2f} px")
    for e in r["events"][:15]:
        print(f"  pair {e['pair']:5d} t={e['t_s']:.3f}s {e['px']:.2f}px only={e['only']} "
              + " ".join(f"{nm}:{v['proj_px']:.2f}" for nm, v in e["in"].items()))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    return r


if __name__ == "__main__":
    main()
