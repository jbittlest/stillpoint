"""Stillpoint command line.

    PYTHONPATH=engine .venv/bin/python -m stillpoint.cli analyze VIDEO --out DIR [--smoothness S]
        [--fov-match-gyroflow RENDER] [--crop-mode footprint|eval [--match-start S --match-dur D]]
        [--target-footprint F] [--target-area A] [--out-fx FX] [--calib] [--loop-iters N] [--processes N]
        [--keep-cache] [--no-cache] [--progress]
      --fov-match-gyroflow: out_fx so the whole-clip mean source footprint >= the Gyroflow render's (M2 default;
      --crop-mode eval = M1's eval-area match over a window). Self-calibration is OFF by default (O3: it raised
      jello); --calib turns it on (--no-calib is accepted for compatibility).
    ... -m stillpoint.cli render VIDEO --plan DIR/plan.spplan --out OUT.mov [--start S --dur D | --start-frame N
        --frames M] [--codec hevc10|hevc10-speed|prores] [--keep-timestamps]
    ... -m stillpoint.cli eval RENDER --orig ORIGINAL --start S --dur D --out DIR [--gyroflow GF_RENDER]
        [--render-start 0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys


def _root():
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def cmd_analyze(a):
    from .pipeline import AnalyzeParams, analyze
    prm = AnalyzeParams(smoothness=a.smoothness, fov_match=a.fov_match_gyroflow, crop_mode=a.crop_mode,
                        match_start=a.match_start, match_dur=a.match_dur, target_area=a.target_area,
                        target_footprint=a.target_footprint, out_fx=a.out_fx,
                        calibrate=bool(a.calib) and not a.no_calib, closed_loop_iters=a.loop_iters,
                        crop_area=a.crop_area, allow_zoom=not a.no_zoom, verbose=not a.quiet,
                        processes=a.processes, keep_cache=a.keep_cache, luma_cache=not a.no_cache)

    def prog(stage, frac, msg):
        print(f'[progress] {frac * 100:5.1f}% {stage}: {msg}', flush=True)
    rep = analyze(a.video, a.out, prm, progress=prog if a.progress else None)
    cl = rep['closed_loop']
    print(json.dumps(dict(plan=rep['plan'], measured_window_hf_open_px=cl['open_loop_window_hf_px'],
                          measured_window_hf_final_px=cl['composite_window_hf_px'], folds=cl['n_increments'],
                          jump_guard=cl.get('jump_guard'), min_out_fx=rep['out']['min_out_fx'],
                          crop={k: v for k, v in rep['crop'].items() if k != 'trials'},
                          timings=rep['timings']), indent=1, default=str))


def cmd_render(a):
    from .pipeline import first_frame_at, render
    from .video import probe
    start_frame, n = a.start_frame, a.frames
    if a.start is not None:
        pr = probe(a.video)
        start_frame = first_frame_at(pr['frame_pts'], a.start)
        if a.dur:
            n = int(math.ceil(a.dur * pr['fps'])) + 1
    r = render(a.video, a.plan, a.out, start_frame=start_frame, n_frames=n or None, codec=a.codec,
               zero_base=not a.keep_timestamps, kernel=a.kernel)
    print(f"rendered {a.out} in {r['seconds']:.1f}s" + (f" ({r['fps']:.1f} fps)" if r['fps'] else ''))


def cmd_eval(a):
    sys.path.insert(0, _root())
    from eval import jitter_metrics as jm
    from .pipeline import summarize_eval
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, os.path.splitext(os.path.basename(a.render))[0])
    res = jm.run(a.render, a.render_start, a.dur, ref=a.orig, ref_start=a.start, plot=base + '.png',
                 signals=base + '.npz', verbose=False)
    with open(base + '.json', 'w') as fh:
        json.dump(res, fh, indent=1, default=jm._json_default)
    runs = {'stillpoint': base}
    for name, path, kw in (('original', a.orig, dict(start=a.start)),
                           ('gyroflow', a.gyroflow, dict(start=a.start, ref=a.orig))):
        if not path:
            continue
        b = os.path.join(a.out, name)
        if not os.path.exists(b + '.json'):
            r = jm.run(path, kw['start'], a.dur, ref=kw.get('ref'), plot=b + '.png', signals=b + '.npz', verbose=False)
            with open(b + '.json', 'w') as fh:
                json.dump(r, fh, indent=1, default=jm._json_default)
        runs[name] = b
    print(summarize_eval(runs)[0])


def main(argv=None):
    ap = argparse.ArgumentParser(prog='stillpoint')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('analyze')
    p.add_argument('video')
    p.add_argument('--out', required=True)
    p.add_argument('--smoothness', type=float, default=1.0)
    p.add_argument('--fov-match-gyroflow', default=None)
    p.add_argument('--match-start', type=float, default=0.0)
    p.add_argument('--match-dur', type=float, default=0.0)
    p.add_argument('--crop-mode', choices=('footprint', 'eval'), default='footprint')
    p.add_argument('--target-footprint', type=float, default=0.0)
    p.add_argument('--target-area', type=float, default=0.0)
    p.add_argument('--out-fx', type=float, default=0.0)
    p.add_argument('--crop-area', type=float, default=0.75)
    p.add_argument('--calib', action='store_true', help='enable self-calibration (off by default)')
    p.add_argument('--no-calib', action='store_true', help='(default) kept for compatibility')
    p.add_argument('--processes', type=int, default=None, help='measurement processes (0 = threads)')
    p.add_argument('--keep-cache', action='store_true', help='keep OUT/.luma_cache.npy')
    p.add_argument('--no-cache', action='store_true', help='re-decode 4K per pass instead of a luma cache')
    p.add_argument('--progress', action='store_true')
    p.add_argument('--no-zoom', action='store_true')
    p.add_argument('--loop-iters', type=int, default=2)
    p.add_argument('-q', '--quiet', action='store_true')
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser('render')
    p.add_argument('video')
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--start', type=float, default=None, help='seconds (first frame with PTS >= start)')
    p.add_argument('--dur', type=float, default=None)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--frames', type=int, default=0)
    p.add_argument('--codec', default='hevc10')
    p.add_argument('--kernel', default='lanczos3')
    p.add_argument('--keep-timestamps', action='store_true')
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser('eval')
    p.add_argument('render')
    p.add_argument('--orig', required=True)
    p.add_argument('--start', type=float, required=True, help='window start in the ORIGINAL (s)')
    p.add_argument('--dur', type=float, required=True)
    p.add_argument('--render-start', type=float, default=0.0, help='window start in the render (0 for zero-based)')
    p.add_argument('--gyroflow', default=None)
    p.add_argument('--out', required=True)
    p.set_defaults(fn=cmd_eval)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
