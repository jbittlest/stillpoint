"""Deterministic decision gate: engine options vs the default, per camera family, from ProRes scoreboards.

    .venv/bin/python -m eval.decision --root work/gate/v6 [--recommend recommend.json]
        reads <root>/<label>/scoreboard.json for label in default, perturb, default_rep, fill, mesh, hl10, hl06
        (whatever exists) and writes <root>/decision.{json,md} and <root>/noise_prores.json

Method (all renders ProRes 422 HQ via scripts/scoreboard.py, so re-rendering a plan is bit-identical):
* determinism: default_rep re-renders 3 windows of the default plans -> render md5 + every judge number must match.
* judge sensitivity ("noise"): perturb = the default plans with every output centre shifted by a constant 0.05 px
  (1080p-eq, x and y; no motion change) on all windows.  Per metric: sigma = RMS over windows of log(perturb/default)
  (relative), jumps / win-rate: the largest absolute change.  The band of a mean over n windows is 2 sigma / sqrt(n).
* per config and camera: paired per-window change vs the default -- geo-mean relative change (exp(mean log ratio) - 1)
  with a 95 % percentile bootstrap CI over windows (eval/boot.py, fixed seed); absolute paired differences for jumps,
  crop footprint and 1-s win-rate.  A change is REAL only when its CI excludes 0 AND it exceeds the noise band.
* windows better / worse = per-window changes beyond 2.5 sigma (the scoreboard's per-window flag).
Rule for a candidate default (per camera): HF really better, no other motion metric (calm, 2-8, 8-30, roll, jello,
corner) really worse, jumps >1 px not really worse, crop not smaller by > 0.5 pp, 1-s win-rate not really worse.
Analysis cost (wall time, peak RSS) is reported next to it; the final recommendation (which also weighs cost and
what the option is FOR, e.g. horizon lock is a look, not a jitter fix) is written by hand (--recommend JSON).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from . import boot

CAMS = ('o3', 'oa4', 'o4')
CAM_NAME = {'o3': 'DJI O3 (5 clips, 10 windows; reference = Jimmy\'s Gyroflow renders)',
            'oa4': 'Osmo Action 4 (clip 0012, 5 windows; reference = the original)',
            'o4': 'O4 Pro (clip 0004, 4 windows; reference = the original)'}
CAM_SHORT = {'o3': 'O3', 'oa4': 'OA4', 'o4': 'O4 Pro'}
REL = [('hf', 'HF >2 Hz'), ('calm', 'calm-cruise HF'), ('b28', '2-8 Hz'), ('b830', '8-30 Hz'), ('roll_deg', 'roll'),
       ('jello', 'jello'), ('corner', 'corner wobble')]
ABS = [('jumps1', 'jumps >1 px / window', 2), ('jumps05', 'jumps >0.5 px / window', 2),
       ('crop_plan', 'crop footprint (source area)', 4), ('win', '1-s win-rate vs reference', 3)]
CONFIGS = [('fill', 'full-frame fill (fill=True, fill_overscan=0.06)'),
           ('mesh', 'mesh residual (mesh_residual=True)'),
           ('hl10', 'horizon lock 1.0 (O3 + OA4 only)'),
           ('hl06', 'horizon lock 0.6, roll_limit_deg 15 (O3 + OA4 only)')]
CROP_TOL = 0.005
K_BAND = 2.5            # a mean change must exceed K_BAND x sigma / sqrt(n) (2.0 flagged the null control)


def fin(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def load(root, label):
    p = os.path.join(root, label, 'scoreboard.json')
    if not os.path.exists(p):
        return None
    sb = json.load(open(p))
    sb['_path'] = p
    return sb


def wins_of(sb, cam=None):
    return {k: w for k, w in (sb or {}).get('windows', {}).items() if w.get('sp') and (cam is None or w['cam'] == cam)}


# ------------------------------------------------------------------------------------------------ noise
def noise_from(default, perturbs):
    """Judge sensitivity from one or more constant-shift controls (perturb, perturb2, ...) vs the default: per metric
    the RMS over (window, control) pairs of log(control/default) -- pooled and per camera -- and its largest value;
    jumps / crop / win-rate: RMS and largest absolute change."""
    d = wins_of(default)
    out = dict(controls=[], source='constant 0.05 px output-centre shift(s) vs default, ProRes', rel={}, abs={},
               per_window={})
    samples = []                                    # (window, cam, control label, perturb wins)
    for sb in perturbs:
        p = wins_of(sb)
        common = sorted(set(d) & set(p))
        out['controls'].append(dict(label=sb.get('label'), perturb_px=sb.get('perturb_px'), windows=len(common)))
        samples += [(w, d[w]['cam'], sb.get('label'), p[w]) for w in common]
    out['windows'] = len({s_[0] for s_ in samples})
    out['pairs'] = len(samples)

    def stats(vals):
        v = np.asarray([x for x in vals if fin(x)], float)
        return dict(sigma=float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan'),
                    max_abs=float(np.max(np.abs(v))) if len(v) else float('nan'), n=int(len(v)))
    for k, _ in REL:
        lr = {}
        for w, cam, lab, pw in samples:
            a, b = pw['sp'].get(k), d[w]['sp'].get(k)
            if fin(a) and fin(b) and a > 0 and b > 0:
                lr.setdefault(cam, []).append(math.log(a / b))
        allv = [x for v in lr.values() for x in v]
        out['rel'][k] = dict(stats(allv), cam={cam: stats(v) for cam, v in lr.items()})
        first = [(pw['sp'].get(k), d[w]['sp'].get(k)) for w, cam, lab, pw in samples
                 if lab == out['controls'][0]['label']] if out['controls'] else []
        c = boot.ratio_ci([x[0] for x in first], [x[1] for x in first])
        c.pop('per', None)
        out['rel'][k]['mean_first_control'] = c
    for k, _, _ in ABS:
        dd = {}
        for w, cam, lab, pw in samples:
            a, b = pw['sp'].get(k), d[w]['sp'].get(k)
            if fin(a) and fin(b):
                dd.setdefault(cam, []).append(a - b)
        allv = [x for v in dd.values() for x in v]
        out['abs'][k] = dict(stats(allv), cam={cam: stats(v) for cam, v in dd.items()})
    for w, cam, lab, pw in samples:
        out['per_window'].setdefault(w, {})[lab] = {k: (pw['sp'].get(k), d[w]['sp'].get(k))
                                                   for k in [x for x, _ in REL] + [x for x, _, _ in ABS]}
    return out


def sigma_for(nz, k, cam, kind='rel'):
    """Per-window sigma used for camera cam: max(per-camera, pooled) (per-camera estimates rest on 4-10 windows)."""
    e = nz.get(kind, {}).get(k) or {}
    vals = [e.get('sigma'), (e.get('cam') or {}).get(cam, {}).get('sigma')]
    vals = [v for v in vals if fin(v)]
    return max(vals) if vals else float('nan')


def determinism(default, rep):
    d, r = wins_of(default), wins_of(rep)
    rows = []
    for w in sorted(set(d) & set(r)):
        a, b = d[w], r[w]
        same_md5 = bool(a['sp'].get('render_md5')) and a['sp'].get('render_md5') == b['sp'].get('render_md5')
        keys = [k for k, _ in REL] + [k for k, _, _ in ABS]
        diffs = {k: (a['sp'].get(k), b['sp'].get(k)) for k in keys
                 if not (a['sp'].get(k) == b['sp'].get(k) or (not fin(a['sp'].get(k)) and not fin(b['sp'].get(k))))}
        rows.append(dict(window=w, plan_same=a.get('plan_sha') == b.get('plan_sha'), md5_same=same_md5,
                         md5=a['sp'].get('render_md5'), metrics_identical=not diffs, diffs=diffs))
    return rows


# ------------------------------------------------------------------------------------------------ comparison
def compare(default, cfg, cam, nz):
    d, c = wins_of(default, cam), wins_of(cfg, cam)
    common = sorted(set(d) & set(c))
    if not common:
        return None
    res = dict(n=len(common), windows=common, rel={}, abs={}, verdict={})
    for k, nm in REL:
        a = [c[w]['sp'].get(k) for w in common]
        b = [d[w]['sp'].get(k) for w in common]
        rc = boot.ratio_ci(a, b)
        sig = sigma_for(nz, k, cam)
        band = K_BAND * sig / math.sqrt(rc['n']) if rc['n'] and fin(sig) else float('nan')
        per = rc.pop('per')
        better = sum(1 for x in per if fin(sig) and x < -2.5 * sig)
        worse = sum(1 for x in per if fin(sig) and x > 2.5 * sig)
        real = boot.excludes_zero(rc) and fin(band) and abs(math.log1p(rc['mean'])) > band
        v = ('better' if rc['mean'] < 0 else 'worse') if real else 'n.s.'
        res['rel'][k] = dict(rc, band=band, sigma=sig, n_better=better, n_worse=worse, verdict=v,
                             default=boot.mean_ci(b), config=boot.mean_ci(a),
                             per_window={w: (c[w]['sp'].get(k), d[w]['sp'].get(k)) for w in common})
        res['verdict'][k] = v
    for k, nm, nd in ABS:
        a = [c[w]['sp'].get(k) for w in common]
        b = [d[w]['sp'].get(k) for w in common]
        dc = boot.diff_ci(a, b)
        dc.pop('per')
        tol = sigma_for(nz, k, cam, 'abs')
        tol = tol if fin(tol) else 0.0
        if k == 'crop_plan':
            v = 'worse' if fin(dc['mean']) and dc['mean'] < -CROP_TOL else ('better' if fin(dc['mean']) and
                                                                             dc['mean'] > CROP_TOL else 'same')
        else:
            sign = -1 if k == 'win' else 1          # jumps: + is worse; win-rate: + is better
            real = boot.excludes_zero(dc) and abs(dc['mean']) > K_BAND * tol / math.sqrt(max(dc['n'], 1))
            v = ('worse' if sign * dc['mean'] > 0 else 'better') if real else 'n.s.'
        res['abs'][k] = dict(dc, verdict=v, noise_sigma=tol, default=boot.mean_ci(b), config=boot.mean_ci(a),
                             sum_default=float(np.nansum([x for x in b if fin(x)])),
                             sum_config=float(np.nansum([x for x in a if fin(x)])))
        res['verdict'][k] = v
    res['passes'] = dict(default=sum(bool(d[w]['sp'].get('passed')) for w in common),
                         config=sum(bool(c[w]['sp'].get('passed')) for w in common))
    motion_worse = [k for k, _ in REL if k != 'hf' and res['verdict'][k] == 'worse']
    ok = (res['verdict']['hf'] == 'better' and not motion_worse and res['verdict']['jumps1'] != 'worse'
          and res['verdict']['crop_plan'] != 'worse' and res['verdict']['win'] != 'worse')
    res['rule'] = dict(candidate_default=bool(ok), hf=res['verdict']['hf'], motion_worse=motion_worse,
                       jumps1=res['verdict']['jumps1'], crop=res['verdict']['crop_plan'], win=res['verdict']['win'])
    return res


def analysis_costs(sbs):
    """Per config and clip: wall s, x realtime, peak RSS tree / main, other heavy slots busy, stage times."""
    out = {}
    for lab, sb in sbs.items():
        if not sb:
            continue
        for clip, m in (sb.get('analyses') or {}).items():
            r = m.get('report') or {}
            out.setdefault(lab, {})[clip] = dict(
                wall_s=m.get('wall_s'), x_realtime=r.get('x_realtime'), n_frames=r.get('n_frames'),
                peak_rss_gb=r.get('peak_rss_gb'), peak_main_gb=r.get('peak_main_gb'), peak_stage=r.get('peak_stage'),
                other_slots_busy=m.get('other_slots_busy'), stage_s=r.get('stage_s'), plan_sha=m.get('plan_sha'),
                reanalyzed=m.get('reanalyzed'), finished=m.get('finished'))
    return out


def fill_artifacts(sb):
    rows = {}
    for k, w in wins_of(sb).items():
        fa = w.get('fill_artifacts')
        if fa:
            rows[k] = {x: fa.get(x) for x in ('frames', 'fill_frac_mean', 'fill_frac_p95', 'fill_frac_max',
                                              'frames_with_fill', 'plan_uncovered_mean', 'seam_ratio_pooled',
                                              'seam_ratio_p95', 'flicker_ratio_pooled', 'flicker_ratio_p95', 'wall_s',
                                              'error')}
    return rows


# ------------------------------------------------------------------------------------------------ markdown
def f(x, nd=3):
    return f'{float(x):.{nd}f}' if fin(x) else '-'


def pct(c):
    if not fin(c.get('mean')):
        return '-'
    ci = f" [{100 * c['lo']:+.0f}, {100 * c['hi']:+.0f}]" if fin(c.get('lo')) else ''
    return f"{100 * c['mean']:+.1f}%{ci}"


def mark(v, txt):
    return f'**{txt}**' if v in ('better', 'worse') else txt


def md(dec, recommend):
    L = ['# Stillpoint v6 decision gate (engine v5 options, ProRes)', '',
         f"Generated {dec['generated']} by `eval/decision.py` from `{dec['root']}`.  Engine {dec['engine']}; renders "
         f"ProRes 422 HQ (bit-identical re-renders); judge eval {dec['eval']}.", '']
    if recommend:
        L += ['## Recommendation', '']
        for cam in CAMS:
            r = recommend.get(cam)
            if r:
                L.append(f"- **{CAM_SHORT[cam]}**: {r}")
        if recommend.get('toggles'):
            L += ['', f"**User toggles:** {recommend['toggles']}"]
        if recommend.get('notes'):
            L += [''] + [f'- {x}' for x in recommend['notes']]
        L.append('')
    # determinism + noise
    L += ['## Determinism and judge noise', '']
    det = dec.get('determinism') or []
    if det:
        L.append('Re-render of the default plans (default_rep): ' + '; '.join(
            f"{r['window']} md5 {'same' if r['md5_same'] else 'DIFFERENT'}, metrics "
            f"{'identical' if r['metrics_identical'] else 'DIFFERENT ' + json.dumps(r['diffs'])}" for r in det) + '.')
        L.append('')
    nz = dec.get('noise')
    if nz and nz.get('rel'):
        ctl = '; '.join(f"{c['label']} {c['perturb_px']} px on {c['windows']} windows" for c in nz.get('controls', []))
        L.append(f"Judge sensitivity to a constant sub-pixel output shift (no motion change; {ctl}; {nz['pairs']} "
                 f"window pairs).  Per-window sigma = RMS of log(shifted/default), pooled (O3 / OA4 / O4) [largest]:")
        L += ['', '| metric | pooled | O3 | OA4 | O4 | largest | mean shift of the first control [95 % CI] |',
              '|---|---|---|---|---|---|---|']
        for k, nm in REL:
            e = nz['rel'].get(k)
            if not e:
                continue
            cs = lambda cam: f"{100 * e['cam'][cam]['sigma']:.1f}%" if cam in e.get('cam', {}) else '-'
            L.append(f"| {nm} | {100 * e['sigma']:.1f}% | {cs('o3')} | {cs('oa4')} | {cs('o4')} | "
                     f"{100 * e['max_abs']:.0f}% | {pct(e.get('mean_first_control', {}))} |")
        for k, nm, nd in ABS:
            e = nz['abs'].get(k)
            if not e:
                continue
            cs = lambda cam: f(e['cam'][cam]['sigma'], nd) if cam in e.get('cam', {}) else '-'
            L.append(f"| {nm} (absolute) | {f(e['sigma'], nd)} | {cs('o3')} | {cs('oa4')} | {cs('o4')} | "
                     f"{f(e['max_abs'], nd)} | |")
        cal = dec.get('calibration') or {}
        if cal:
            L += ['', 'Rule calibration (the shift controls judged like a config vs the default; must flag nothing): '
                  + '; '.join(f"{x} {cam}: {', '.join(r['real']) or 'nothing real'}" for x, cc in cal.items()
                              for cam, r in cc.items() if r) + '.']
        L += ['', 'Each camera uses max(its own sigma, pooled sigma).  A mean change is **bold** (real) only when its '
                  '95 % bootstrap CI over windows excludes 0 AND it is larger than 2.5 sigma/sqrt(n); windows better/worse '
                  'count per-window changes beyond 2.5 sigma.', '']
    # per camera tables
    for cam in CAMS:
        rows = [(lab, nm, dec['compare'][lab][cam]) for lab, nm in CONFIGS
                if dec['compare'].get(lab, {}).get(cam)]
        base = dec['default_summary'].get(cam)
        if not rows and not base:
            continue
        L += [f'## {CAM_NAME[cam]}', '']
        if base:
            ci = base['ci']
            L.append(f"Default (v5): HF {f(ci['hf']['mean'])} [{f(ci['hf']['lo'])}, {f(ci['hf']['hi'])}], calm "
                     f"{f(ci['calm']['mean'])}, 2-8 {f(ci['b28']['mean'])}, 8-30 {f(ci['b830']['mean'])}, roll "
                     f"{f(ci['roll_deg']['mean'], 4)} deg, jello {f(ci['jello']['mean'])}, corner "
                     f"{f(ci['corner']['mean'])}, jumps >1 px {base['jumps1']} (>0.5: {base['jumps05']}), crop "
                     f"{f(ci['crop_plan']['mean'], 4)}, pooled 1-s win-rate vs {'Gyroflow' if cam == 'o3' else 'original'} "
                     f"{f(100 * base['win_pooled'], 0)}%, {base['passed']}/{base['n']} windows pass every gate check; "
                     f"reference HF {f(base['ref_mean']['hf'])}.")
            L.append('')
        if rows:
            L += ['| config vs default | ' + ' | '.join(nm for _, nm in REL) + ' | jumps >1 (>0.5), sum | crop | '
                  'win-rate | gate passes | rule |', '|' + '---|' * (len(REL) + 6)]
            h2h = (dec.get('head_to_head') or {}).get(cam)
            for lab, nm, r in rows + ([('h2h', '*mesh vs fill, head-to-head (baseline = fill)*', h2h)] if h2h else []):
                cells = []
                for k, _ in REL:
                    x = r['rel'][k]
                    cells.append(mark(x['verdict'], pct(x)) + f" ({x['n_better']}+/{x['n_worse']}-)")
                j1, j05 = r['abs']['jumps1'], r['abs']['jumps05']
                cr, wr = r['abs']['crop_plan'], r['abs']['win']
                cells.append(mark(j1['verdict'], f"{j1['sum_default']:.0f} -> {j1['sum_config']:.0f}") +
                             f" ({j05['sum_default']:.0f} -> {j05['sum_config']:.0f})")
                cells.append(mark(cr['verdict'], f"{100 * cr['mean']:+.2f} pp"))
                cells.append(mark(wr['verdict'], f"{100 * wr['mean']:+.1f} pp [{100 * wr['lo']:+.0f}, "
                                                 f"{100 * wr['hi']:+.0f}]" if fin(wr['lo']) else
                                  f"{100 * wr['mean']:+.1f} pp"))
                cells.append(f"{r['passes']['default']} -> {r['passes']['config']} of {r['n']}")
                if lab == 'h2h':
                    cells.append('mesh beats fill' if r['rule']['candidate_default'] else 'no')
                else:
                    cells.append('candidate default' if r['rule']['candidate_default'] else 'no')
                L.append(f'| {nm} | ' + ' | '.join(cells) + ' |')
            L += ['', 'Cells: geo-mean change vs default [95 % CI] (windows better+/worse- beyond 2.5 sigma)'
                  + ('; the head-to-head row is mesh vs fill with the same rule (fill and mesh are mutually exclusive).'
                     if h2h else '.'), '']
            L += ['<details><summary>per-window HF (config vs default)</summary>', '',
                  '| window | ' + ' | '.join(nm.split(' (')[0] for _, nm, _ in rows) + ' |',
                  '|---|' + '---|' * len(rows)]
            wl = sorted({w for _, _, r in rows for w in r['windows']})
            for w in wl:
                cells = []
                for _, _, r in rows:
                    pv = r['rel']['hf']['per_window'].get(w)
                    cells.append(f'{f(pv[0])} vs {f(pv[1])}' if pv else '-')
                L.append(f'| {w} | ' + ' | '.join(cells) + ' |')
            L += ['', '</details>', '']
    # costs
    costs = dec.get('costs') or {}
    if costs:
        L += ['## Analysis cost (one heavy job at a time)', '',
              '| clip | ' + ' | '.join(lab for lab in costs) + ' |', '|---|' + '---|' * len(costs)]
        clips = sorted({c for v in costs.values() for c in v})
        for clip in clips:
            cells = []
            base = costs.get('default', {}).get(clip, {})
            for lab in costs:
                m = costs[lab].get(clip)
                if not m:
                    cells.append('-')
                    continue
                ov = (f" ({100 * (m['wall_s'] / base['wall_s'] - 1):+.0f}%)" if lab != 'default' and fin(m.get('wall_s'))
                      and fin(base.get('wall_s')) else '')
                busy = (m.get('other_slots_busy') or {}).get('max')
                cells.append(f"{f(m.get('wall_s'), 0)} s{ov}, {f(m.get('peak_rss_gb'), 2)} GB"
                             + (f" (busy {busy})" if busy else ''))
            L.append(f'| {clip} | ' + ' | '.join(cells) + ' |')
        L += ['', 'wall time of the whole-clip analysis (3 measurement workers) and peak RSS of its process tree '
                  '(engine mem watch, 1 s); "busy N" = other heavy jobs ran meanwhile (timing not clean).', '']
    fa = dec.get('fill_artifacts') or {}
    if fa:
        L += ['## Fill artifacts (fill config, first 12 s of each window)', '',
              '| window | fill frac mean / p95 | frames with fill | uncovered (plan) | seam ratio pooled / p95 | '
              'flicker ratio pooled / p95 |', '|---|---|---|---|---|---|']
        for w, r in fa.items():
            L.append(f"| {w} | {f(r.get('fill_frac_mean'), 4)} / {f(r.get('fill_frac_p95'), 4)} | "
                     f"{f(r.get('frames_with_fill'), 2)} | {f(r.get('plan_uncovered_mean'), 4)} | "
                     f"{f(r.get('seam_ratio_pooled'), 2)} / {f(r.get('seam_ratio_p95'), 2)} | "
                     f"{f(r.get('flicker_ratio_pooled'), 2)} / {f(r.get('flicker_ratio_p95'), 2)} |"
                     + (f" {r['error']}" if r.get('error') else ''))
        L.append('')
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   'work', 'gate', 'v6'))
    ap.add_argument('--recommend', default='', help='JSON {o3, oa4, o4, toggles, notes[]}: the hand-written verdict')
    ap.add_argument('--configs', default='', help='comma list of config labels to judge (default: every one of '
                    + ', '.join(x for x, _ in CONFIGS) + ' that exists); e.g. fill,mesh while other runs are in flight')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    want = [x.strip() for x in a.configs.split(',') if x.strip()]
    unknown = sorted(set(want) - {x for x, _ in CONFIGS})
    if unknown:
        raise SystemExit(f'unknown --configs {unknown}')
    labels = ['default', 'default_rep', 'perturb', 'perturb2'] + [lab for lab, _ in CONFIGS if not want or lab in want]
    sbs = {lab: load(root, lab) for lab in labels}
    default = sbs['default']
    if default is None:
        raise SystemExit(f'{root}/default/scoreboard.json missing')
    pert = [sbs[x] for x in ('perturb', 'perturb2') if sbs.get(x)]
    nz = noise_from(default, pert) if pert else dict(rel={}, abs={}, windows=0, pairs=0, controls=[])
    dec = dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S'), root=root,
               engine=f"{default['engine'].get('head_short')} content {default['engine'].get('content')}",
               eval=f"{default['eval'].get('eval_version')} / {default['eval'].get('sha1')}",
               codecs={lab: sb.get('codec') for lab, sb in sbs.items() if sb},
               runs={lab: dict(path=sb['_path'], label=sb.get('label'), params=sb.get('params_overrides'),
                               windows=sb.get('n_windows_done'), engine=sb.get('engine', {}).get('content'))
                     for lab, sb in sbs.items() if sb},
               determinism=determinism(default, sbs['default_rep']) if sbs['default_rep'] else [],
               noise=nz, default_summary=default.get('summary', {}),
               compare={lab: {cam: compare(default, sbs[lab], cam, nz) for cam in CAMS} for lab, _ in CONFIGS
                        if sbs.get(lab)},
               configs_judged=[lab for lab, _ in CONFIGS if sbs.get(lab)],
               calibration={x: {cam: (lambda r: r and dict(real=[k for k, v in r['verdict'].items()
                                                                 if v in ('better', 'worse')], n=r['n']))(
                   compare(default, sbs[x], cam, nz)) for cam in CAMS} for x in ('perturb', 'perturb2') if sbs.get(x)},
               head_to_head={cam: compare(sbs['fill'], sbs['mesh'], cam, nz) for cam in CAMS}
               if sbs.get('fill') and sbs.get('mesh') else {},
               costs=analysis_costs({lab: sbs.get(lab) for lab in ['default'] + [x for x, _ in CONFIGS]}),
               fill_artifacts=fill_artifacts(sbs.get('fill')))
    engines = {lab: sb['engine'].get('content') for lab, sb in sbs.items() if sb}
    if len(set(engines.values())) > 1:
        dec['warning'] = f'runs use different engine contents: {engines}'
    codecs = {c for c in dec['codecs'].values()}
    if codecs != {'prores'}:
        dec['warning'] = (dec.get('warning', '') + f' codecs {dec["codecs"]}').strip()
    rec = json.load(open(a.recommend)) if a.recommend else None
    if rec:
        dec['recommendation'] = rec
    home = os.path.expanduser('~')
    redact = (lambda t: t.replace(home, '~')) if home and home != '~' else (lambda t: t)   # public repo: no home path
    with open(os.path.join(root, 'decision.json'), 'w') as fh:
        fh.write(redact(json.dumps(dec, indent=1, default=float)))
    with open(os.path.join(root, 'noise_prores.json'), 'w') as fh:
        fh.write(redact(json.dumps(nz, indent=1, default=float)))
    txt = md(dec, rec)
    if dec.get('warning'):
        txt = txt.replace('\n\n', f"\n\n**WARNING:** {dec['warning']}\n\n", 1)
    with open(os.path.join(root, 'decision.md'), 'w') as fh:
        fh.write(redact(txt))
    print(txt)


if __name__ == '__main__':
    main()
