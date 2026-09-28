"""Calm-cruise decomposition for the M1 notes: what is left in Stillpoint's calm-cruise HF, and how much of it a
rotation-only stabiliser can remove at all.

For each clip (eval signals of original / Gyroflow / Stillpoint, same calm frames as gated_paired):
  * components of the calm HF: tx, ty, roll, log-scale (combined px);
  * scale jitter is non-rotational (a camera rotation cannot change image scale): a hard floor;
  * eval's split-half noise floor of the Stillpoint render;
  * the Stillpoint calm HF restricted to frames where the closed loop's vision measurement was trusted
    (conf >= 0.6 in the final analysis), i.e. where the residual rotation is known to be removed.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
R = np.sqrt((1920 ** 2 + 1080 ** 2) / 12)
CLIPS = [('DJI_0025', 15.0), ('DJI_0028', 8.0), ('DJI_0034', 15.0), ('DJI_0027', 5.0), ('DJI_0032', 22.0)]


def comps(e, idx):
    return {k: float(np.sqrt(np.mean((e[k][idx] * (R if k in ('hp_rot', 'hp_logs') else 1)) ** 2)))
            for k in ('hp_tx', 'hp_ty', 'hp_rot', 'hp_logs')}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ''
    rows = []
    for clip, S in CLIPS:
        sp_p = os.path.join(ROOT, 'work', 'm1', 'eval', f'{clip}{tag}_stillpoint')
        if not os.path.exists(sp_p + '.npz'):
            continue
        o = np.load(os.path.join(ROOT, 'work', 'baseline', f'{clip}_orig.npz'))
        g = np.load(os.path.join(ROOT, 'work', 'baseline', f'{clip}_gf.npz'))
        s = np.load(sp_p + '.npz')
        N = min(len(o['hp_tx']), len(s['hp_tx']), len(g['hp_tx']))
        sp = o['lp_speed']
        sp = np.concatenate([[sp[0]], sp])[:N]
        base = np.arange(30, N - 30)
        idx = base[sp[base] < 150]
        a = np.load(os.path.join(ROOT, 'work', 'm1', clip + tag, 'analysis.npz'))
        f0 = int(np.searchsorted(a['frame_pts'], S - 1e-6))
        conf = np.zeros(len(a['frame_pts']))
        conf[a['res0_k0']] = a['res0_conf']
        cw = np.convolve(conf, np.ones(31) / 31, 'same')[f0:f0 + N]
        idx_t = idx[cw[idx] >= 0.6]
        cs, co, cg = comps(s, idx), comps(o, idx), comps(g, idx)
        tot = lambda c: float(np.sqrt(sum(v ** 2 for v in c.values())))
        noise = json.load(open(sp_p + '.json'))['metrics']['noise_floor_px']
        rows.append(dict(clip=clip, calm_frac=len(idx) / len(base), sp=tot(cs), sp_c=cs, gf=tot(cg), orig_scale=co['hp_logs'],
                         sp_scale=cs['hp_logs'], sp_noise=noise, trusted_frac=len(idx_t) / max(len(idx), 1),
                         sp_trusted=tot(comps(s, idx_t)) if len(idx_t) > 30 else float('nan'),
                         sp_trusted_noscale=float(np.sqrt(sum(v ** 2 for k, v in comps(s, idx_t).items() if k != 'hp_logs')))
                         if len(idx_t) > 30 else float('nan')))
    print('| clip | calm frames | SP calm HF | tx / ty / roll / scale | scale floor (orig) | SP noise floor | '
          'calm frames with trusted vision | SP calm HF there | same without scale |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        c = r['sp_c']
        print(f"| {r['clip']} | {r['calm_frac']:.0%} | {r['sp']:.3f} | {c['hp_tx']:.3f} / {c['hp_ty']:.3f} / {c['hp_rot']:.3f} / "
              f"{c['hp_logs']:.3f} | {r['orig_scale']:.3f} | {r['sp_noise']:.3f} | {r['trusted_frac']:.0%} | "
              f"{r['sp_trusted']:.3f} | {r['sp_trusted_noscale']:.3f} |")


if __name__ == '__main__':
    main()
