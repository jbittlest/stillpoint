"""A/B tool (integration): turn a web plan dump (test/plan.dump.ts) into a .spplan for the engine's Metal renderer,
and compare its virtual camera path with a Python .spplan of the same clip over a window.

  .venv/bin/python web/test/plan.dump.py PREFIX [--compare PY.spplan --start S --dur D]

Path comparison: both plans share the camera rows (same telemetry, row times, exposure averaging), so
D_k = M_py[k,j]^T M_web[k,j] = V_py^T V_web gives V_py from the web plan's virtQ. Each path's frame-to-frame rotation
is turned into 1920-wide px (yaw/pitch * fx, roll * frame diagonal/sqrt(12)), accumulated, zero-phase high-passed
and band-RMS'd like eval/jitter_metrics (2-8 Hz, 8-Nyquist): the smoother's own residual motion, gyro errors excluded.
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'engine'))
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation
from stillpoint.plan_io import read_plan, write_plan
from stillpoint.types import Lens, Plan


def load_web(prefix):
    m = json.load(open(prefix + '.json'))
    F, R = m['frames'], m['nRows']
    raw = open(prefix + '.bin', 'rb').read()
    o = 0
    def take(dt, n):
        nonlocal o
        a = np.frombuffer(raw, dtype=dt, count=n, offset=o); o += a.nbytes; return a
    pts = take('<f8', F); fx = take('<f4', F); mats = take('<f4', F * R * 9); vq = take('<f8', F * 4)
    L = m['lens']
    lens = Lens(L['model'], L['fx'], L['fy'], L['cx'], L['cy'], np.array(L['k'], dtype=np.float64), m['srcW'], m['srcH'])
    plan = Plan(src_w=m['srcW'], src_h=m['srcH'], out_w=m['outW'], out_h=m['outH'], lens=lens, frame_pts=pts.astype(np.float64),
                out_fx=fx.astype(np.float64), row_mats=mats.astype(np.float64).reshape(F, R, 3, 3),
                virt_q=vq.reshape(F, 4).copy(), meta={'readout_s': m['readoutS']})
    return m, plan


def band_px(V, fx, fps, W, H, sl):
    """V (F,3,3) virtual cam->world; returns 2-8 Hz and 8-Nyq RMS (1920-eq px) of the path's own motion in window sl."""
    rel = np.einsum('kji,kjl->kil', V[:-1], V[1:])          # V_k^T V_{k+1}
    w = Rotation.from_matrix(rel).as_rotvec()              # camera frame: x right, y down, z forward
    s = 1920.0 / W
    tx, ty = w[:, 1] * fx[:-1] * s, w[:, 0] * fx[:-1] * s   # yaw -> horizontal, pitch -> vertical shift
    rot = w[:, 2] * np.sqrt(((W * s) ** 2 + (H * s) ** 2) / 12.0)
    out = {}
    for name, lo, hi in (('hf_2hz', 2.0, None), ('2-8Hz', 2.0, 8.0), ('8-nyq', 8.0, None)):
        sos = butter(4, [lo, hi] if hi else lo, btype='bandpass' if hi else 'highpass', fs=fps, output='sos')
        comp = [sosfiltfilt(sos, np.cumsum(c))[sl] for c in (tx, ty, rot)]
        out[name] = float(np.sqrt(np.mean(comp[0] ** 2 + comp[1] ** 2 + comp[2] ** 2)))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix'); ap.add_argument('--compare'); ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--dur', type=float, default=0); ap.add_argument('--trim', type=float, default=0.5)
    a = ap.parse_args()
    m, web = load_web(a.prefix)
    write_plan(a.prefix + '.spplan', web)
    print('wrote', a.prefix + '.spplan')
    if a.compare:
        py = read_plan(a.compare)
        assert py.n_frames == web.n_frames and np.allclose(py.frame_pts, web.frame_pts), 'plans cover different frames'
        Vw = Rotation.from_quat(web.virt_q[:, [1, 2, 3, 0]]).as_matrix()
        j = web.n_rows // 2
        D = np.einsum('kji,kjl->kil', py.row_mats[:, j], web.row_mats[:, j])   # V_py^T V_web
        Vp = np.einsum('kij,klj->kil', Vw, D)                                   # V_web D^T
        fps = 1.0 / np.median(np.diff(web.frame_pts))
        t = web.frame_pts
        t1 = a.start + (a.dur or (t[-1] - a.start))
        sl = np.where((t >= a.start + a.trim) & (t <= t1 - a.trim))[0][:-1]
        diff = Rotation.from_matrix(D).magnitude() * 180 / np.pi
        res = dict(window=[a.start, t1], web=band_px(Vw, web.out_fx, fps, web.out_w, web.out_h, sl),
                   python=band_px(Vp, py.out_fx, fps, py.out_w, py.out_h, sl),
                   path_diff_deg=dict(median=float(np.median(diff[sl])), max=float(diff[sl].max())),
                   fx=dict(web=[float(web.out_fx.min()), float(web.out_fx.max())], python=[float(py.out_fx.min()), float(py.out_fx.max())]))
        print(json.dumps(res, indent=1))
