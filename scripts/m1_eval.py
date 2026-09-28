"""M1 gate (ENGINE_SPEC §5): analyze whole clips, render the baseline windows at 4K, eval vs original + Gyroflow.

    cd stillpoint && PYTHONPATH=engine .venv/bin/python scripts/m1_eval.py [DJI_0025 ...] [--stage analyze,render,eval]
        [--tag NAME] [--params '{"closed_loop_iters": 2}']

Outputs: work/m1/<clip>/ (plan.spplan, report.json), work/m1/<clip>_stillpoint.mov, work/m1/eval/<clip>_stillpoint.*,
work/m1/results_<tag>.json and a markdown table on stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'engine'))
sys.path.insert(0, ROOT)

from stillpoint.pipeline import AnalyzeParams, analyze, first_frame_at, render, summarize_eval  # noqa: E402
from stillpoint.video import probe  # noqa: E402
from eval.footage import GYROFLOW_DIR, O3_DIR  # noqa: E402

D4 = O3_DIR
D5 = GYROFLOW_DIR
CLIPS = [('DJI_0025', 15.0, 25.0), ('DJI_0028', 8.0, 25.0), ('DJI_0034', 15.0, 25.0), ('DJI_0027', 5.0, 25.0),
         ('DJI_0032', 22.0, 25.0)]
M1 = os.path.join(ROOT, 'work', 'm1')
BASE = os.path.join(ROOT, 'work', 'baseline')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clips', nargs='*')
    ap.add_argument('--stage', default='analyze,render,eval')
    ap.add_argument('--tag', default='')
    ap.add_argument('--params', default='{}')
    a = ap.parse_args()
    stages = set(a.stage.split(','))
    extra = json.loads(a.params)
    sel = [c for c in CLIPS if not a.clips or c[0] in a.clips]
    tag = ('_' + a.tag) if a.tag else ''
    allres = {}
    rp = os.path.join(M1, f'results{tag}.json')
    if os.path.exists(rp):
        allres = json.load(open(rp))
    from eval import jitter_metrics as jm
    for clip, start, dur in sel:
        orig = os.path.join(D4, clip + '.MP4')
        adir = os.path.join(M1, clip + tag)
        mov = os.path.join(M1, f'{clip}{tag}_stillpoint.mov')
        evdir = os.path.join(M1, 'eval')
        os.makedirs(evdir, exist_ok=True)
        ebase = os.path.join(evdir, f'{clip}{tag}_stillpoint')
        info = allres.get(clip, {})
        if 'analyze' in stages:
            gf_area = json.load(open(os.path.join(BASE, f'{clip}_gf.json')))['ref']['visible_area_frac_mean']
            prm = AnalyzeParams(target_area=gf_area, match_start=start, match_dur=dur, **extra)
            t0 = time.time()
            analyze(orig, adir, prm)
            info['analyze_s'] = time.time() - t0
        if 'render' in stages:
            pr = probe(orig)
            f0 = first_frame_at(pr['frame_pts'], start)
            n = int(math.ceil(dur * pr['fps'])) + 2
            r = render(orig, os.path.join(adir, 'plan.spplan'), mov, start_frame=f0, n_frames=n, codec='hevc10')
            info['render'] = dict(seconds=r['seconds'], fps=r['fps'], start_frame=f0, frames=n)
            print(f'{clip}: rendered {n} frames in {r["seconds"]:.1f}s ({r["fps"]:.1f} fps)', flush=True)
        if 'eval' in stages:
            t0 = time.time()
            res = jm.run(mov, 0.0, dur, ref=orig, ref_start=start, plot=ebase + '.png', signals=ebase + '.npz',
                         verbose=False)
            with open(ebase + '.json', 'w') as fh:
                json.dump(res, fh, indent=1, default=jm._json_default)
            info['eval_s'] = time.time() - t0
        if os.path.exists(ebase + '.json'):
            md, d = summarize_eval({'original': os.path.join(BASE, f'{clip}_orig'),
                                    'gyroflow': os.path.join(BASE, f'{clip}_gf'), 'stillpoint': ebase})
            info['summary'] = d
            info['table'] = md
            print(f'\n### {clip} ({start:.0f}-{start + dur:.0f} s)\n' + md, flush=True)
            ca = d['rows']['stillpoint'].get('calm_axes', {})
            cg = d['rows']['gyroflow'].get('calm_axes', {})
            co = d['rows']['original'].get('calm_axes', {})
            for b in ('2-8', '8-30'):
                if b in ca:
                    print(f'calm {b} Hz x/y/roll px: orig {co[b]["x"]:.3f}/{co[b]["y"]:.3f}/{co[b]["roll"]:.3f}  '
                          f'GF {cg[b]["x"]:.3f}/{cg[b]["y"]:.3f}/{cg[b]["roll"]:.3f}  '
                          f'SP {ca[b]["x"]:.3f}/{ca[b]["y"]:.3f}/{ca[b]["roll"]:.3f}')
        allres = json.load(open(rp)) if os.path.exists(rp) else {}     # merge (other runs may write too)
        allres[clip] = {**allres.get(clip, {}), **info}
        with open(rp, 'w') as fh:
            json.dump(allres, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else repr(o))


if __name__ == '__main__':
    main()
