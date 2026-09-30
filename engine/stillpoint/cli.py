"""Stillpoint command line.

    PYTHONPATH=engine .venv/bin/python -m stillpoint.cli analyze VIDEO --out DIR [--smoothness S]
        [--fov-match-gyroflow RENDER] [--crop-mode footprint|eval [--match-start S --match-dur D]]
        [--target-footprint F] [--target-area A] [--out-fx FX] [--calib] [--loop-iters N] [--processes N]
        [--keep-cache] [--no-cache] [--progress] [--no-timecal] [--horizon-lock S [--roll-limit DEG]]
        [--fill | --no-fill [--fill-overscan F]] [--mesh | --no-mesh] [--synth-blur off|auto|angle] [--blur-smooth W]
      engine v5 options: an option not given takes this camera's default, resolved exactly like the app
      (app_bridge.resolve_for_video: AnalyzeParams' defaults + app_bridge.CAMERA_OPTION_DEFAULTS) -- today timecal on
      for every camera and full-frame fill on for the DJI O3 (gate v6); fill / mesh / synth-blur are mutually exclusive:
      --horizon-lock S levels the horizon with strength S (0..1) beyond --roll-limit DEG; --fill synthesises the
      border from neighbouring frames (sprender with the fill kernels); --mesh adds the parallax mesh residual (slow:
      two extra tracking passes); --synth-blur writes a synthetic-shutter sidecar; --blur-smooth W blur-aware path.
      --fov-match-gyroflow: out_fx so the whole-clip mean source footprint >= the Gyroflow render's (M2 default;
      --crop-mode eval = M1's eval-area match over a window). The per-clip timing self-calibration (timecal.py)
      runs by default and applies only confident corrections; --no-timecal turns it off. --calib selects the
      LEGACY calib.self_calibrate instead (O3: it raised jello; --no-calib is accepted for compatibility).
    ... -m stillpoint.cli render VIDEO --plan DIR/plan.spplan --out OUT.mov [--start S --dur D | --start-frame N
        --frames M] [--codec hevc10|hevc10-speed|prores] [--keep-timestamps]
    ... -m stillpoint.cli eval RENDER --orig ORIGINAL --start S --dur D --out DIR [--gyroflow GF_RENDER]
        [--render-start 0]
    ... -m stillpoint.cli mesh VIDEO --plan IN.spplan --out OUT.spplan [--start S --dur D [--start S2 --dur D2] [--pad 2]]
        [--processes 3] [--set key=value ...]
      adds the parallax-aware mesh residual (engine/stillpoint/mesh.py) to an existing plan (whole clip or windows;
      MeshParams fields via --set). analyze --mesh does the same inside the analysis.
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
    from .app_bridge import resolve_for_video
    camera, v5 = resolve_for_video(a, a.video)       # options not given -> this camera's defaults (as in the app)
    if not a.quiet:
        on = [k for k in ('fill', 'mesh') if v5[k]] + (['horizon_lock'] if v5['horizon_lock'] > 0 else [])
        print(f'[options] camera {camera or "?"}: {", ".join(on) or "no v5 options"} on, '
              f'timecal {"on" if v5["timecal"] else "off"}', flush=True)
    prm = AnalyzeParams(smoothness=a.smoothness, fov_match=a.fov_match_gyroflow, crop_mode=a.crop_mode,
                        match_start=a.match_start, match_dur=a.match_dur, target_area=a.target_area,
                        target_footprint=a.target_footprint, out_fx=a.out_fx,
                        calibrate=bool(a.calib) and not a.no_calib, closed_loop_iters=a.loop_iters,
                        timecal=v5['timecal'],
                        crop_area=a.crop_area, allow_zoom=not a.no_zoom, verbose=not a.quiet,
                        processes=a.processes, keep_cache=a.keep_cache, luma_cache=not a.no_cache,
                        horizon_lock=v5['horizon_lock'], roll_limit_deg=v5['roll_limit_deg'],
                        mesh_residual=v5['mesh'], fill=v5['fill'], fill_overscan=v5['fill_overscan'],
                        synth_blur=v5['synth_blur'], blur_smooth_w=v5['blur_smooth'])

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
               zero_base=not a.keep_timestamps, kernel=a.kernel, blur=a.blur)
    print(f"rendered {a.out} in {r['seconds']:.1f}s" + (f" ({r['fps']:.1f} fps)" if r['fps'] else ''))


def _parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def cmd_mesh(a):
    from .mesh import mesh_plan_file
    d = mesh_plan_file(a.video, a.plan, a.out, start=a.start, dur=a.dur, pad=a.pad, processes=a.processes,
                       overrides=_parse_set(a.set), meas_file=a.meas)
    v = d.get('verify') or {}
    print(json.dumps(dict(out=a.out, offset_rms_1080=d.get('offset_rms_1080'), offset_max_1080=d.get('offset_max_1080'),
                          accepted=v.get('accepted'), windows=v.get('windows'), total_s=d.get('total_s')), indent=1))


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
    p.add_argument('--no-calib', action='store_true', help='(default) no LEGACY calibration; kept for compatibility')
    p.add_argument('--no-timecal', action='store_true', help='keep the metadata timing (no per-clip timing fit)')
    p.add_argument('--processes', type=int, default=None, help='measurement processes (0 = threads)')
    p.add_argument('--keep-cache', action='store_true', help='keep OUT/.luma_cache.npy')
    p.add_argument('--no-cache', action='store_true', help='re-decode 4K per pass instead of a luma cache')
    p.add_argument('--progress', action='store_true')
    p.add_argument('--no-zoom', action='store_true')
    p.add_argument('--loop-iters', type=int, default=2)
    # v5 options default to None = "not given": resolved per camera (app_bridge.resolve_for_video)
    p.add_argument('--horizon-lock', type=float, default=None, metavar='S',
                   help='horizon lock strength 0..1 (0 = off; needs a gravity-referenced attitude)')
    p.add_argument('--roll-limit', type=float, default=None, metavar='DEG',
                   help='with --horizon-lock: bank up to DEG degrees is kept, only the excess is leveled')
    p.add_argument('--mesh', action='store_true', default=None,
                   help='parallax-aware mesh residual (mesh.py) on the final plan')
    p.add_argument('--no-mesh', dest='mesh', action='store_false')
    p.add_argument('--fill', action='store_true', default=None,
                   help='full-frame border fill from neighbouring frames (fill.py; default overscan 0.06; '
                        'on by default for the DJI O3)')
    p.add_argument('--no-fill', dest='fill', action='store_false')
    p.add_argument('--fill-overscan', type=float, default=None, metavar='F',
                   help='with --fill: the crop may leave the source by F x min(src_w, src_h) (default 0.06)')
    p.add_argument('--synth-blur', choices=('off', 'auto', 'angle'), default=None,
                   help='synthetic shutter sidecar plan.spblur (render with --blur)')
    p.add_argument('--blur-smooth', type=float, default=0.0, metavar='W', help='blur-aware smoothing weight (0 = off)')
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
    p.add_argument('--blur', default=None, help='synthetic shutter sidecar (plan.spblur, AnalyzeParams.synth_blur)')
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
    p = sub.add_parser('mesh')
    p.add_argument('video')
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--start', type=float, action='append', default=None,
                   help='window start (s, source timeline); repeat --start/--dur for several windows')
    p.add_argument('--dur', type=float, action='append', default=None)
    p.add_argument('--pad', type=float, default=2.0)
    p.add_argument('--processes', type=int, default=3)
    p.add_argument('--set', action='append', default=[], help='MeshParams field=value (JSON value)')
    p.add_argument('--meas', default=None, help='tracking-pass cache (.npz): written if missing, reused if present')
    p.set_defaults(fn=cmd_mesh)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
