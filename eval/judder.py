"""
Motion-blur JUDDER metric: does the blur baked into each output frame agree with the output's own motion?

A stabilized frame keeps the motion blur of the REAL (shaky) exposure.  Once the picture is steady, the streaks
still point along the original shake and change direction/length from frame to frame ("blur judder": the eye reads
it as jitter although the geometry is perfect).  Here the baked blur is predicted from the gyro (the camera rotation
during each row's exposure, projected into OUTPUT pixels through the frame's own warp) and compared with the blur
the output's motion would produce (a fixed world point smeared along the OUTPUT's motion during the same exposure).

Per output frame k (virtual orientation V_k, output focal f_k, exposure e_k), output point p (ray r_v), taps tau in
[-e/2, e/2] (mid-points):
    X      = V_k r_v                                 world ray shown at p
    t_p    = row time of the source row p maps to (rolling shutter, exact renderer mapping)
    o(tau) = pi( V_k^T R(t_p) R(t_p+tau)^T X )       baked blur: where a fixed world point is smeared in the OUTPUT
    c(tau) = pi( V(t_k+tau)^T X )                    consistent blur: the smear if the camera had moved like the
                                                     output (virtual path V(t), slerp of the plan's path)
    m(tau) = o(tau) - c(tau), centred (the mean over tau is geometry -- the jitter eval measures it)
    judder_len(k,p) = sqrt(12) * RMS_tau |m|         equivalent length of a straight mismatched streak (a uniform
                                                     streak of length L has RMS L/sqrt(12) about its centre)
    blur_len / cons_len                              the same statistic of o / c alone
    lin(k,p) = e * LSQ-slope of m over tau           straight (directional) part of the mismatch, a 2-vector (px)
pi = rectilinear projection with f_k.  Unstabilized output (V = R) gives exactly 0 (tests/test_judder.py); a locked
output under a constant pan w gives |judder| = f*w*e at the centre.

Summary (summarize()): judder_px = RMS over frames x points of judder_len; judder_hf_px = RMS over time/points of
the > fc Hz (zero-phase Butterworth, like eval.jitter_metrics) part of the lin vectors (frame-to-frame changes of the
mismatched streak: the part that reads as jitter); mean/p95, fraction of frames > 1 / 2 px; blur/cons lengths.
Units: 1080p-equivalent px (x 1920 / out_w), like eval.jitter_metrics.

Synthetic shutter (remedy for the look, engine/stillpoint/synth_blur.py): the output PSF becomes the baked one
convolved with a streak along the virtual path over T_s (exactly consistent motion).  The mismatch in px is
unchanged (the baked blur is still there) but it is embedded in more consistent blur.  `share` reports
    inconsistent share = var(m) / (var(o) + var(s))        (second moments of the streaks, per frame x point)
= the fraction of the output PSF's spread that disagrees with the output motion (1 = all of it, e.g. a locked shot
of a pan with no synthetic blur; 0 = consistent; > 1 = the output moves more than its blur shows -- "missing" blur,
the strobing look of short shutters).  Summary `share` is energy-pooled (sum var(m) / sum PSF var over frames x
points); `share_mean` (the per-frame ratio averaged) is dominated by near-sharp frames and is kept for reference only.
It is a masking INDEX, not a validated perceptual model.

Vision variant (judder_from_eval): for renders without a plan (Gyroflow), the output motion comes from the eval's
own vision measurement of the render (eval.jitter_metrics signals) and the baked blur from the gyro with V_k ~ R(t_k)
(the stabilizer's correction is a few degrees: a few % on the streak direction) and the render's fitted output focal.

    PYTHONPATH=engine:. .venv/bin/python -m eval.judder plan VIDEO ANALYSIS_DIR [--start S --dur D] [--synth plan.spblur]
    PYTHONPATH=engine:. .venv/bin/python -m eval.judder synthetic       # synthetic validation (renders a test clip)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1920 = 1920.0
SQRT12 = math.sqrt(12.0)


def _engine():
    eng = os.path.join(ROOT, "engine")
    if eng not in sys.path:
        sys.path.insert(0, eng)


_engine()
from stillpoint.geom import qconj, qmul, qrotate, slerp_series  # noqa: E402


# ------------------------------------------------------------------------------------------------ geometry
def output_points(out_w: int, out_h: int, nx: int = 3, ny: int = 3, inset: float = 0.1) -> np.ndarray:
    """(S,2) output pixel coords of an nx x ny grid (inset from the border by `inset` of the size)."""
    xs = np.linspace(inset, 1 - inset, nx) * (out_w - 1)
    ys = np.linspace(inset, 1 - inset, ny) * (out_h - 1)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel()], 1)


def _taps(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) + 0.5) / n - 0.5          # fractions of the exposure, mean 0


def _proj(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    """rectilinear projection relative to the principal point; r (...,3), f broadcastable to r[...,0]."""
    z = np.where(np.abs(r[..., 2]) < 1e-9, 1e-9, r[..., 2])
    return np.stack([f * r[..., 0] / z, f * r[..., 1] / z], -1)


def virt_at(vt: np.ndarray, V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Virtual orientation at times t: slerp of the path samples (vt, V), extrapolated at constant rate beyond the
    ends (so the first/last frames' exposure windows see the path's motion, not a hold)."""
    from stillpoint.geom import qexp, qlog
    vt = np.asarray(vt, np.float64)
    t = np.asarray(t, np.float64)
    q = slerp_series(vt, V, t)
    if len(vt) < 2:
        return q
    for i0, i1, sel in ((0, 1, t < vt[0]), (-1, -2, t > vt[-1])):
        if np.any(sel):
            w = qlog(qmul(qconj(V[i0]), V[i1])) / (vt[i1] - vt[i0])          # rate (rad/s), body frame
            q[sel] = qmul(np.broadcast_to(V[i0], q[sel].shape), qexp((t[sel] - vt[i0])[..., None] * w))
    return q


def _stats(tr: np.ndarray, tau: np.ndarray):
    """tr (..., n, 2) trajectory over taps tau (n,) -> (var (...), slope (..., 2)) about the tap mean."""
    c = tr - tr.mean(axis=-2, keepdims=True)
    var = (c ** 2).sum(-1).mean(-1)
    slope = np.einsum('...ni,n->...i', c, tau) / max(float((tau ** 2).sum()), 1e-30)
    return var, slope


def blur_mismatch(q_cam_fn, frame_t, exposure, virt_q, out_fx, out_w: int, out_h: int, lens, readout: float,
                  src_h: int, virt_t=None, pts=None, taps: int = 9, synth_shutter=None, synth_taps: int = 9,
                  chunk: int = 1024) -> dict:
    """Exact plan-based blur/motion mismatch.  frame_t, exposure, out_fx (F,), virt_q (F,4) per OUTPUT frame
    (frame_t = the source frame's centre-row mid-exposure time); virt_t: times of the virtual path samples used for
    the consistent trajectory (default frame_t).  synth_shutter (F,) s: synthetic virtual-path shutter (masking).
    Returns per-frame arrays (1080p-eq px): judder_len, blur_len, cons_len, synth_len (F,), share (F,) and lin
    (F,S,2)."""
    _engine()
    from stillpoint.smooth import map_output_points
    frame_t = np.asarray(frame_t, np.float64).reshape(-1)
    F = len(frame_t)
    ex = np.clip(np.nan_to_num(np.broadcast_to(np.asarray(exposure, np.float64), (F,)), nan=0.0), 0.0, 0.1)
    fo = np.broadcast_to(np.asarray(out_fx, np.float64), (F,)).astype(np.float64)
    V = np.asarray(virt_q, np.float64).reshape(F, 4)
    vt = frame_t if virt_t is None else np.asarray(virt_t, np.float64)
    pts = output_points(out_w, out_h) if pts is None else np.asarray(pts, np.float64)
    S = len(pts)
    tau = _taps(taps)
    ts = None if synth_shutter is None else np.clip(np.broadcast_to(np.asarray(synth_shutter, np.float64), (F,)),
                                                    0.0, 1.0)
    sc = S1920 / float(out_w)
    cx, cy = (out_w - 1) / 2.0, (out_h - 1) / 2.0
    out = dict(judder_len=np.zeros(F), blur_len=np.zeros(F), cons_len=np.zeros(F), synth_len=np.zeros(F),
               share=np.zeros(F), var_m=np.zeros(F), var_den=np.zeros(F), lin=np.zeros((F, S, 2)),
               exposure=ex.copy())
    for a in range(0, F, chunk):
        b = min(F, a + chunk)
        n = b - a
        ft, e, f, Vk = frame_t[a:b], ex[a:b], fo[a:b], V[a:b]
        p = map_output_points(q_cam_fn, ft, Vk, f, pts, out_w, out_h, lens, readout, src_h, 3)     # (n,S,2)
        tp = ft[:, None] + readout * ((p[..., 1] + 0.5) / src_h - 0.5)                            # (n,S)
        rv = np.stack([(pts[None, :, 0] - cx) / f[:, None], (pts[None, :, 1] - cy) / f[:, None],
                       np.ones((n, S))], -1)                                                       # (n,S,3)
        dt = e[:, None, None] * tau[None, None, :]                                                # (n,1,T)
        qp = np.asarray(q_cam_fn(tp), np.float64)                                                  # (n,S,4)
        qi = np.asarray(q_cam_fn(tp[..., None] + dt), np.float64)                                  # (n,S,T,4)
        Vb = Vk[:, None, None, :]
        # o: conj(V) * q_p * conj(q_i) * V applied to r_v
        wq = qmul(qmul(qmul(qconj(Vb), qp[:, :, None, :]), qconj(qi)), Vb)
        O = _proj(qrotate(wq, np.broadcast_to(rv[:, :, None, :], wq.shape[:-1] + (3,))), f[:, None, None])
        # c: conj(V(t_k + tau)) * V_k applied to r_v
        qv = virt_at(vt, V, ft[:, None] + e[:, None] * tau[None, :])                         # (n,T,4)
        uq = qmul(qconj(qv), Vk[:, None, :])[:, None, :, :]                                        # (n,1,T,4)
        uq = np.broadcast_to(uq, (n, S, taps, 4))
        C = _proj(qrotate(uq, np.broadcast_to(rv[:, :, None, :], (n, S, taps, 3))), f[:, None, None])
        vo, _ = _stats(O, tau)
        vc, _ = _stats(C, tau)
        vm, sl = _stats(O - C, tau)
        out['judder_len'][a:b] = sc * SQRT12 * np.sqrt(vm.mean(1))
        out['blur_len'][a:b] = sc * SQRT12 * np.sqrt(vo.mean(1))
        out['cons_len'][a:b] = sc * SQRT12 * np.sqrt(vc.mean(1))
        out['lin'][a:b] = sc * sl                                     # slope per unit exposure fraction = px
        vs = np.zeros((n, S))
        if ts is not None:
            st = _taps(synth_taps)
            qs = virt_at(vt, V, ft[:, None] + ts[a:b, None] * st[None, :])                   # (n,T2,4)
            us = np.broadcast_to(qmul(qconj(qs), Vk[:, None, :])[:, None], (n, S, synth_taps, 4))
            Ssyn = _proj(qrotate(us, np.broadcast_to(rv[:, :, None, :], (n, S, synth_taps, 3))), f[:, None, None])
            vs, _ = _stats(Ssyn, st)
            out['synth_len'][a:b] = sc * SQRT12 * np.sqrt(vs.mean(1))
        den = vo + vs
        out['share'][a:b] = np.where(den > 1e-12, vm / np.maximum(den, 1e-12), 0.0).mean(1)
        out['var_m'][a:b] = vm.mean(1)
        out['var_den'][a:b] = den.mean(1)
    return out


# ------------------------------------------------------------------------------------------------ summary
def _hp(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    if len(x) < 3 * max(8, int(fs / fc)) // 2 or fc <= 0:
        return x - x.mean(axis=0, keepdims=True)
    sos = butter(order, fc, 'highpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x, axis=0)


def summarize(res: dict, fps: float, fc: float = 2.0, trim_s: float = 0.5, mask=None) -> dict:
    """Scalar summary (1080p-eq px) of a blur_mismatch result.  mask (F,) bool: frames to include (after the HF
    filter, which runs on the whole series)."""
    J, B, C = res['judder_len'], res['blur_len'], res['cons_len']
    F = len(J)
    tr = int(round(trim_s * fps)) if F > 4 * int(round(trim_s * fps)) + 8 else 0
    keep = np.zeros(F, bool)
    keep[tr:F - tr] = True
    if mask is not None:
        keep &= np.asarray(mask, bool)
    lin = res['lin']
    hf = _hp(lin.reshape(F, -1), fps, fc).reshape(lin.shape)
    k = keep
    if not k.any():
        return dict(n_frames=0)
    share_w = np.where(J > 0.5, 1.0, 0.0)[k]
    d = dict(
        n_frames=int(k.sum()),
        judder_px=float(np.sqrt(np.mean(J[k] ** 2))), judder_mean_px=float(J[k].mean()),
        judder_p95_px=float(np.percentile(J[k], 95)), judder_max_px=float(J[k].max()),
        judder_hf_px=float(np.sqrt(np.mean((hf[k] ** 2).sum(-1)))),
        judder_lin_px=float(np.sqrt(np.mean((lin[k] ** 2).sum(-1)))),
        frac_gt_1px=float((J[k] > 1.0).mean()), frac_gt_2px=float((J[k] > 2.0).mean()),
        blur_px=float(np.sqrt(np.mean(B[k] ** 2))), blur_p95_px=float(np.percentile(B[k], 95)),
        cons_px=float(np.sqrt(np.mean(C[k] ** 2))),
        synth_px=float(np.sqrt(np.mean(res['synth_len'][k] ** 2))),
        share=(float(res['var_m'][k].sum() / max(res['var_den'][k].sum(), 1e-30)) if 'var_m' in res else float('nan')),
        share_mean=float(res['share'][k].mean()),
        share_judder_frames=float((res['share'][k] * share_w).sum() / max(share_w.sum(), 1.0)),
        exposure_ms_median=float(1e3 * np.median(res['exposure'][k])),
        exposure_ms_p95=float(1e3 * np.percentile(res['exposure'][k], 95)),
    )
    return d


# ------------------------------------------------------------------------------------------------ plan helpers
def plan_inputs(video: str, analysis_dir: str, start: float = 0.0, dur=None, cache_dir=None):
    """(tel, q_cam_fn, frames, virt_q, out_fx, out_w, out_h, lens, readout) for the frames of [start, start+dur) of an
    analysis (analysis.npz virt_q/out_fx/corr_rotvec; plan.spplan geometry).  The closed-loop correction is
    re-applied from the per-frame corr_rotvec (interpolated in time; exact at frame times)."""
    _engine()
    from stillpoint.geom import qexp
    from stillpoint.pipeline import first_frame_at
    from stillpoint.plan_build import camera_orientation_fn, effective_lens
    from stillpoint.plan_io import read_plan
    from stillpoint.telemetry import load_telemetry
    from stillpoint.types import TimeModel
    tel = load_telemetry(video, cache_dir=cache_dir or _default_cache())
    z = np.load(os.path.join(analysis_dir, 'analysis.npz'))
    plan = read_plan(os.path.join(analysis_dir, 'plan.spplan'))
    ft_all = np.asarray(z['frame_t'], np.float64)
    corr = np.asarray(z['corr_rotvec'], np.float64) if 'corr_rotvec' in z.files else None
    correction = None
    if corr is not None and np.abs(corr).max() > 0:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(ft_all, corr, axis=0, extrapolate=True)

        def correction(t, _cs=cs):
            t = np.asarray(t, np.float64)
            return qexp(_cs(np.clip(t, ft_all[0], ft_all[-1])))
    tm = TimeModel()
    q_fn = camera_orientation_fn(tel, tm, correction=correction)
    k0 = first_frame_at(np.asarray(z['frame_pts']), float(start)) if start else 0
    k1 = len(ft_all) if dur is None else min(len(ft_all), k0 + int(round(float(dur) * tel.fps)))
    frames = np.arange(k0, k1)
    return dict(tel=tel, q_fn=q_fn, frames=frames, frame_t=ft_all[frames], virt_q=np.asarray(z['virt_q'])[frames],
                virt_t=ft_all, virt_all=np.asarray(z['virt_q']), out_fx=np.asarray(z['out_fx'])[frames],
                out_w=plan.out_w, out_h=plan.out_h, lens=effective_lens(tel, tm), readout=float(tel.readout_s),
                exposure=np.asarray(tel.exposure_s, np.float64)[frames], src_h=int(tel.height), src_w=int(tel.width),
                frame_pts=np.asarray(tel.frame_pts, np.float64)[frames], fps=float(tel.fps))


def _default_cache():
    _engine()
    from stillpoint.workspace import cache_dir
    return cache_dir()


def judder_for_analysis(video: str, analysis_dir: str, start: float = 0.0, dur=None, synth_shutter=None,
                        taps: int = 9) -> tuple[dict, dict]:
    """(per-frame result, summary) of the plan-based metric over a window of an analysed clip."""
    I = plan_inputs(video, analysis_dir, start, dur)
    ss = None
    if synth_shutter is not None:
        ss = np.asarray(synth_shutter, np.float64)
        ss = ss[I['frames']] if len(ss) != len(I['frames']) else ss
    res = blur_mismatch(I['q_fn'], I['frame_t'], I['exposure'], I['virt_q'], I['out_fx'], I['out_w'], I['out_h'],
                        I['lens'], I['readout'], I['src_h'], virt_t=I['virt_t'][I['frames']], taps=taps,
                        synth_shutter=ss)
    return res, summarize(res, I['fps'])


# ------------------------------------------------------------------------------------------------ vision variant
def judder_from_eval(tel, eval_npz: str, start: float, focal_1920: float, out_aspect: float,
                     taps: int = 9, pts_n: int = 3, fc: float = 2.0, scale_term: bool = False,
                     return_frames: bool = False) -> dict:
    """Judder of a render WITHOUT a plan (e.g. Gyroflow): baked blur from the gyro (V_k ~ R(t_k): the camera-frame
    rotation during the exposure applied to the output ray; rolling shutter ignored), output motion from the eval's
    vision measurement of the render (path_tx/ty/rot/logs: 1920-eq px / rad per frame).  focal_1920: the render's
    fitted output focal at 1920 width (eval ref.footprint_focal_1920_median).  Returns the summary dict (same keys
    as summarize) plus 'n_pairs' (and 'frames' = the per-frame result when return_frames).
    scale_term=False (default): the similarity SCALE rate is left out of the output motion.  The baked-blur model is
    rotation-only and a rotating camera at a fixed focal produces no uniform scale change; in low forward flight the
    scale term is translation parallax (radial expansion), which blurs the original and the output alike, so counting
    it on one side only inflated the metric (vs the plan-exact metric on the same renders: see tests/test_judder.py
    and research notes in the workstream-D report)."""
    _engine()
    from stillpoint.pipeline import first_frame_at
    z = np.load(eval_npz)
    fps = float(z['fps'])
    N = len(z['path_tx'])
    k0 = first_frame_at(tel.frame_pts, float(start))
    ks = np.arange(k0, min(k0 + N, tel.n_frames))
    N = len(ks)
    W1, H1 = S1920, S1920 * out_aspect
    pts = output_points(int(round(W1)), int(round(H1)), pts_n, pts_n)
    c = np.array([(W1 - 1) / 2.0, (H1 - 1) / 2.0])
    d = pts - c                                                          # (S,2) 1920-eq px from the centre
    # output velocity (px/frame) at each point, centred difference of the measured path
    P = np.stack([np.asarray(z[k], float)[:N] for k in ('path_tx', 'path_ty', 'path_rot', 'path_logs')], 1)
    vel = np.gradient(P, axis=0)                                         # (N,4) per frame
    if not scale_term:
        vel[:, 3] = 0.0
    vx = vel[:, None, 0] - vel[:, None, 2] * d[None, :, 1] + vel[:, None, 3] * d[None, :, 0]
    vy = vel[:, None, 1] + vel[:, None, 2] * d[None, :, 0] + vel[:, None, 3] * d[None, :, 1]
    v_out = np.stack([vx, vy], -1)                                       # (N,S,2) px/frame
    e = np.clip(np.nan_to_num(np.asarray(tel.exposure_s, float)[ks]), 0, 0.1)
    tau = _taps(taps)
    ft = np.asarray(tel.frame_t, float)[ks]
    rv = np.concatenate([d / focal_1920, np.ones((len(d), 1))], 1)       # (S,3)
    q0 = tel.orientation_at(ft)                                          # (N,4)
    qi = tel.orientation_at(ft[:, None] + e[:, None] * tau[None, :])     # (N,T,4)
    rel = qmul(qconj(qi), q0[:, None, :])                                # camera-frame: R(t+tau)^T R(t)
    O = _proj(qrotate(rel[:, None], np.broadcast_to(rv[None, :, None, :], (N, len(d), taps, 3))), focal_1920)
    Cc = (e * fps)[:, None, None, None] * tau[None, None, :, None] * v_out[:, :, None, :]   # (N,S,T,2)
    vo, _ = _stats(O, tau)
    vc, _ = _stats(Cc, tau)
    vm, sl = _stats(O - Cc, tau)
    res = dict(judder_len=SQRT12 * np.sqrt(vm.mean(1)), blur_len=SQRT12 * np.sqrt(vo.mean(1)),
               cons_len=SQRT12 * np.sqrt(vc.mean(1)), synth_len=np.zeros(N),
               share=np.where(vo > 1e-12, vm / np.maximum(vo, 1e-12), 0.0).mean(1), var_m=vm.mean(1),
               var_den=vo.mean(1), lin=sl, exposure=e)
    s = summarize(res, fps, fc)
    s['n_pairs'] = int(N)
    if return_frames:
        s['frames'] = res
    return s


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('video')
    p.add_argument('analysis_dir')
    p.add_argument('--start', type=float, default=0.0)
    p.add_argument('--dur', type=float, default=None)
    p.add_argument('--synth', default=None, help='.spblur sidecar: include the synthetic shutter (masking index)')
    p.add_argument('--json', default=None)
    s = sub.add_parser('synthetic')
    s.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        ss = None
        if a.synth:
            from stillpoint.synth_blur import read_blur
            ss = read_blur(a.synth)['shutter_s']
        res, summ = judder_for_analysis(a.video, a.analysis_dir, a.start, a.dur, synth_shutter=ss)
        print(json.dumps(summ, indent=1))
        if a.json:
            with open(a.json, 'w') as fh:
                json.dump(summ, fh, indent=1)
    elif a.cmd == 'synthetic':
        from eval.judder_synth import run_synthetic
        r = run_synthetic(out_dir=a.out)
        print(json.dumps(r, indent=1, default=float))


if __name__ == '__main__':
    main()
