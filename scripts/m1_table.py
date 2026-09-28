"""Build the M1 results table from work/m1/results<tag>.json (written by scripts/m1_eval.py)."""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
tag = sys.argv[1] if len(sys.argv) > 1 else ''
R = json.load(open(os.path.join(ROOT, 'work', 'm1', f'results{("_" + tag) if tag else ""}.json')))
ORDER = ['DJI_0025', 'DJI_0028', 'DJI_0034', 'DJI_0027', 'DJI_0032']
WIN = {'DJI_0025': '15-40 s', 'DJI_0028': '8-33 s', 'DJI_0034': '15-40 s', 'DJI_0027': '5-30 s', 'DJI_0032': '22-47 s'}
out = ['| clip (window) | video | HF px | calm-cruise HF px | 2-8 Hz | 8-30 Hz | roll deg | 8-30 Hz roll deg | jello px | '
       '1-s median | crop area | SP better than this in (1-s windows) |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
gate = ['| clip | calm-cruise <= 0.15 | 8-30 Hz <= original | win-rate vs GF >= 90% | crop >= GF | all |', '|---|---|---|---|---|---|']
for c in ORDER:
    if c not in R or 'summary' not in R[c]:
        continue
    rows, pr = R[c]['summary']['rows'], R[c]['summary']['paired']
    for nm, lab in (('original', 'original'), ('gyroflow', 'Gyroflow'), ('stillpoint', '**Stillpoint**')):
        x = rows[nm]
        w = pr.get(nm)
        wr = f"{w['b_better_frac'] * 100:.0f}% of {w['windows']} (p={w['wilcoxon_p']:.1g})" if w else '-'
        crop = '-' if x['crop'] != x['crop'] else f"{x['crop']:.3f}"
        b = (lambda s: f'**{s}**') if nm == 'stillpoint' else (lambda s: s)
        hf, calm = f"{x['hf']:.3f}", f"{x['calm']:.3f}"
        out.append(f"| {c} ({WIN[c]}) | {lab} | {b(hf)} | {b(calm)} | "
                   f"{x['b28']:.3f} | {x['b8']:.3f} | {x['roll']:.4f} | {x['b8_rot']:.4f} | {x['jello']:.3f} | {x['med1s']:.3f} | "
                   f"{crop} | {wr} |")
    s, o, g = rows['stillpoint'], rows['original'], rows['gyroflow']
    wg = pr['gyroflow']['b_better_frac']
    ok = [s['calm'] <= 0.15, s['b8'] <= o['b8'], wg >= 0.9, s['crop'] >= g['crop']]
    f = lambda b, t: ('PASS ' if b else 'FAIL ') + t
    t0, t1 = f"{s['calm']:.3f}", f"{s['b8']:.3f} vs {o['b8']:.3f}"
    t2, t3 = f"{wg * 100:.0f}%", f"{s['crop']:.3f} vs {g['crop']:.3f}"
    gate.append(f"| {c} | {f(ok[0], t0)} | {f(ok[1], t1)} | {f(ok[2], t2)} | {f(ok[3], t3)} | "
                f"{'PASS' if all(ok) else 'FAIL'} |")
print('\n'.join(out))
print()
print('\n'.join(gate))
