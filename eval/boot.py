"""Bootstrap confidence intervals over scoreboard windows (scripts/scoreboard.py, eval/decision.py).

Windows are the resampling unit: a per-camera mean (or a mean paired change between two configurations) is
resampled over its windows with replacement, B times, with a FIXED seed so that tables are reproducible.
Percentile intervals; with n <= 5 windows the interval is coarse (few distinct resamples) and is reported as-is.
"""
from __future__ import annotations

import math

import numpy as np

B_DEFAULT = 10000
SEED = 20260930


def _finite(vals):
    out = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return np.asarray(out, float)


def mean_ci(vals, B: int = B_DEFAULT, alpha: float = 0.05, seed: int = SEED) -> dict:
    """Mean of the finite values and its percentile bootstrap CI over them -> {mean, lo, hi, n}."""
    v = _finite(vals)
    n = len(v)
    if n == 0:
        return dict(mean=float('nan'), lo=float('nan'), hi=float('nan'), n=0)
    if n == 1:
        return dict(mean=float(v[0]), lo=float('nan'), hi=float('nan'), n=1)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return dict(mean=float(v.mean()), lo=float(np.percentile(bs, 100 * alpha / 2)),
                hi=float(np.percentile(bs, 100 * (1 - alpha / 2))), n=int(n))


def ratio_ci(this, other, B: int = B_DEFAULT, alpha: float = 0.05, seed: int = SEED) -> dict:
    """Paired geometric-mean ratio this/other over windows (both > 0) with its bootstrap CI, as a relative change:
    {mean, lo, hi} = exp(mean log ratio) - 1 (and its CI), n = pairs used, per = per-window log ratios."""
    lr = []
    for a, b in zip(this, other):
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0:
            lr.append(math.log(a / b))
    c = mean_ci(lr, B, alpha, seed)
    f = lambda x: math.exp(x) - 1 if math.isfinite(x) else float('nan')
    return dict(mean=f(c['mean']), lo=f(c['lo']), hi=f(c['hi']), n=c['n'], per=[round(x, 5) for x in lr])


def diff_ci(this, other, B: int = B_DEFAULT, alpha: float = 0.05, seed: int = SEED) -> dict:
    """Paired absolute difference this - other over windows with its bootstrap CI."""
    d = []
    for a, b in zip(this, other):
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if math.isfinite(a) and math.isfinite(b):
            d.append(a - b)
    c = mean_ci(d, B, alpha, seed)
    c['per'] = [round(x, 5) for x in d]
    return c


def excludes_zero(c: dict) -> bool:
    return math.isfinite(c.get('lo', float('nan'))) and math.isfinite(c.get('hi', float('nan'))) and \
        (c['lo'] > 0 or c['hi'] < 0)
