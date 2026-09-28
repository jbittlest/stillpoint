"""Export golden fixtures for the web PATH module (web/src/{so3,lens,smooth,plan}.ts).  Owner: PATH agent (web).

    cd stillpoint && PYTHONPATH=engine .venv/bin/python web/test/path.export.py [--only NAME]

For every case it writes web/test/fixtures/path/<name>.json (meta + Python results) and <name>.bin.gz (arrays).
Telemetry is exported with the IMU quaternions rounded to float32 and the IMU times on their exact uniform grid
(imuT[i] = t0 + i*dt, reconstructed identically in numpy and JS); every Python number in the fixture is computed
from exactly that (rounded) telemetry, so TS and Python see bit-identical inputs.

Python results per case (gyro-only camera orientation: camera_orientation_fn(tel, TimeModel())):
  * smooth.optimize_path virt_q / out_fx at a fixed min_out_fx (the value the engine used for that clip),
    runtime and diagnostics                                                     -> quality test (b)
  * plan_build.build_plan row matrices (float64, 32 rows, exposure averaging as the engine does) for the Python
    virtual path, on a subset of frames                                         -> exactness test (a)
  * smooth.check_crop (dense 64/edge, exact RS mapping) of the Python path     -> reference violations

Cases: O3 DJI_0034 (whole clip, fx0 = 1544.80, the engine's final footprint-matched focal) and Osmo Action 4
DJI_20260927091931_0012 frames [5994, 5994+3600) (100-160 s: exposures 1-4.5 ms, crossing the exposure-averaging
ramp; fx0 = 1611.07, the value the OA4 fixer used).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'engine'))

from stillpoint.plan_build import build_plan, camera_orientation_fn  # noqa: E402
from stillpoint.smooth import SmoothParams, check_crop, optimize_path  # noqa: E402
from stillpoint.telemetry import load_telemetry  # noqa: E402
from stillpoint.types import TimeModel  # noqa: E402

OUT = os.path.join(HERE, 'fixtures', 'path')
HOME = os.path.expanduser('~')
CASES = {
    'o3_0034': dict(video=os.path.join(HOME, 'Desktop', 'untitled folder 4', 'DJI_0034.MP4'),
                    start=0, n=None, fx0=1544.8019299464606),
    'oa4_0012_w100': dict(video=os.path.join(HOME, 'Desktop', 'DJI_20260927091931_0012_D.MP4'),
                          start=5994, n=3600, fx0=1611.0712918603779),
    # benchmark only (too big for the repo: --out to a scratch dir): the whole 6.7-min OA4 clip, 23,950 frames
    'oa4_0012_full': dict(video=os.path.join(HOME, 'Desktop', 'DJI_20260927091931_0012_D.MP4'),
                          start=0, n=None, fx0=1611.0712918603779, bench=True),
}
N_GOLD = 150          # frames with exported row matrices
N_ROWS = 32


def window(tel, s: int, n: int | None):
    """Slice frames [s, s+n) and the IMU to what they need (as pipeline._analyze does for windows)."""
    e = tel.n_frames if n is None else min(tel.n_frames, s + n)
    ex = np.asarray(tel.exposure_s)
    tel = replace(tel, frame_pts=np.asarray(tel.frame_pts)[s:e], frame_t=np.asarray(tel.frame_t)[s:e],
                  exposure_s=ex[s:e] if ex.ndim else ex,
                  segments=[(max(int(a), s) - s, min(int(b), e - 1) - s) for a, b in (tel.segments or [])
                            if int(b) >= s and int(a) < e])
    t_lo, t_hi = tel.frame_t[0] - 0.08, tel.frame_t[-1] + 0.08
    i0 = max(0, int(np.searchsorted(tel.imu_t, t_lo)) - 2)
    i1 = min(len(tel.imu_t), int(np.searchsorted(tel.imu_t, t_hi)) + 2)
    return replace(tel, imu_t=tel.imu_t[i0:i1], imu_q=tel.imu_q[i0:i1],
                   gravity_q=None if tel.gravity_q is None else tel.gravity_q[i0:i1])


def rounded(tel):
    """Uniform-grid times + float32 quaternions (the exact inputs the TS side reconstructs)."""
    t = np.asarray(tel.imu_t, np.float64)
    dt = float((t[-1] - t[0]) / (len(t) - 1))
    t0 = float(t[0])
    grid = t0 + np.arange(len(t), dtype=np.float64) * dt
    dev = float(np.abs(grid - t).max())
    if dev > 1e-9:
        raise RuntimeError(f'IMU grid is not uniform (max deviation {dev:.3g} s): segments need explicit times')
    q = np.asarray(tel.imu_q, np.float64).astype(np.float32).astype(np.float64)
    return replace(tel, imu_t=grid, imu_q=q, gravity_q=None), t0, dt, dev


def run(name: str, c: dict, out: str = OUT):
    print(f'== {name}', flush=True)
    tel = load_telemetry(c['video'], cache_dir=os.path.join(ROOT, 'work', 'cache'))
    tel = window(tel, c['start'], c['n'])
    tel, t0, dt, dev = rounded(tel)
    F = tel.n_frames
    frames = np.arange(F)
    tm = TimeModel()
    q_fn = camera_orientation_fn(tel, tm)
    out_w, out_h = int(tel.width), int(tel.height)
    sp = SmoothParams(smoothness=1.0, min_out_fx=c['fx0'], max_out_fx=c['fx0'] * 1.5, allow_zoom=True)
    T0 = time.perf_counter()
    v, fx, info = optimize_path(tel, q_fn, frames, out_w, out_h, sp, tm=tm, return_info=True)
    t_opt = time.perf_counter() - T0
    print(f'   optimize_path {t_opt:.1f}s  zoom changes {info["zoom_changes"]}  max zoom {info["max_zoom"]:.4f}  '
          f'max viol {info["max_violation_px"]:.3f}', flush=True)
    viol0 = check_crop(tel, q_fn, frames, v, fx, out_w, out_h, n_per_edge=64, margin_px=0.0)
    viol8 = check_crop(tel, q_fn, frames, v, fx, out_w, out_h, n_per_edge=64, margin_px=8.0)
    gold = np.unique(np.round(np.linspace(0, F - 1, N_GOLD)).astype(np.int64))
    T1 = time.perf_counter()
    plan = build_plan(tel, tm, q_fn, v, fx, out_w, out_h, n_rows=N_ROWS, frames=frames)
    t_plan = time.perf_counter() - T1
    print(f'   build_plan {t_plan:.2f}s exposure_avg={plan.meta["exposure_avg"]}', flush=True)
    rm = np.asarray(plan.row_mats, np.float64)[gold]            # (G, R, 3, 3)
    # a second golden: the same path through the INSTANTANEOUS orientation (exposure_avg=False)
    plan_i = build_plan(tel, tm, q_fn, v[gold], fx[gold], out_w, out_h, n_rows=N_ROWS, frames=gold,
                        exposure_avg=False)
    rm_i = np.asarray(plan_i.row_mats, np.float64)

    arrays = [
        ('framePts', np.asarray(tel.frame_pts, np.float64)),
        ('frameT', np.asarray(tel.frame_t, np.float64)),
        ('exposureS', np.asarray(tel.exposure_s, np.float64)),
        ('imuQ32', np.asarray(tel.imu_q, np.float32).reshape(-1)),
        ('virtQ', np.asarray(v, np.float64).reshape(-1)),
        ('outFx', np.asarray(fx, np.float64)),
        ('goldFrames', gold.astype(np.float64)),
        ('goldRowMats', rm.reshape(-1)),
        ('goldRowMatsInst', rm_i.reshape(-1)),
        ('violPy0', viol0.astype(np.float64)),
    ]
    layout, blobs, off = {}, [], 0
    for k, a in arrays:
        b = np.ascontiguousarray(a).tobytes()
        pad = (-off) % 8
        if pad:
            blobs.append(b'\0' * pad)
            off += pad
        layout[k] = dict(offset=off, dtype='f32' if a.dtype == np.float32 else 'f64', length=int(a.size))
        blobs.append(b)
        off += len(b)
    L = tel.lens
    meta = dict(
        name=name, video=os.path.basename(c['video']), start_frame=c['start'],
        tel=dict(camera=tel.camera, width=int(tel.width), height=int(tel.height), fps=float(tel.fps), nFrames=F,
                 readoutS=float(tel.readout_s), imuRate=float(tel.imu_rate), hasHighrate=bool(tel.has_highrate),
                 eisBaked=bool(tel.eis_baked), segments=[[int(a), int(b)] for a, b in tel.segments],
                 imuT0=t0, imuDt=dt, nImu=int(len(tel.imu_t)), imuGridDev=dev,
                 lens=dict(model=L.model, fx=float(L.fx), fy=float(L.fy), cx=float(L.cx), cy=float(L.cy),
                           k=[float(x) for x in L.k], width=int(L.width), height=int(L.height))),
        python=dict(fx0=float(c['fx0']), max_out_fx=float(c['fx0'] * 1.5), smoothness=1.0, out_w=out_w, out_h=out_h,
                    n_rows=N_ROWS, optimize_s=t_opt, build_plan_s=t_plan, exposure_avg=bool(plan.meta['exposure_avg']),
                    zoom_changes=int(info['zoom_changes']), max_zoom=float(info['max_zoom']),
                    max_violation_px_margin8=float(viol8.max()), max_violation_px_margin0=float(viol0.max()),
                    n_viol_margin0=int((viol0 > 0).sum()), n_viol_margin8=int((viol8 > 0).sum()),
                    frac_binding=float(info['frac_binding']), max_phi_deg=float(info['max_phi_deg']),
                    iters=[{k: v_ for k, v_ in r.items() if k != 'status'} for r in info['iters']]),
        layout=layout)
    os.makedirs(out, exist_ok=True)
    with gzip.open(os.path.join(out, name + '.bin.gz'), 'wb', compresslevel=9) as fh:
        fh.write(b''.join(blobs))
    with open(os.path.join(out, name + '.json'), 'w') as fh:
        json.dump(meta, fh, indent=1, default=float)
    sz = os.path.getsize(os.path.join(out, name + '.bin.gz'))
    print(f'   wrote {name}.bin.gz ({sz / 1e6:.2f} MB), viol0 max {viol0.max():.3f} n>0 {(viol0 > 0).sum()}, '
          f'viol8 max {viol8.max():.3f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--out', default=OUT, help='output dir (bench cases must go to a scratch dir)')
    a = ap.parse_args()
    for name, c in CASES.items():
        if (a.only and name != a.only) or (not a.only and c.get('bench')):
            continue
        if c.get('bench') and os.path.abspath(a.out) == os.path.abspath(OUT):
            raise SystemExit(f'{name} is a benchmark case: pass --out <scratch dir>')
        run(name, c, a.out)


if __name__ == '__main__':
    main()
