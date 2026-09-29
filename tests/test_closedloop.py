"""WP-D tests for stillpoint.closedloop (fold_residuals) + the full measure -> fold -> re-render loop.

Simulation (camera frame x right, y down, z forward; q maps camera -> world):
  (The math / end-to-end tests model up to 0.2-0.3 deg of 2-25 Hz camera error that the gyro path does not show
  at all; they run with the step limiter off (step_floor_px=None), which by design refuses frame-to-frame
  correction steps the gyro's own events do not back. The limiter has its own tests at the end.)
  V_k      intended virtual path (slow pan)
  R_est_k  camera orientation estimate = V_k * Exp(s_k)   (s = the large stabilisation rotation, gyro shake)
  R_true_k = R_est_k * Exp(c_k)                            (c = camera-frame orientation ERROR to be found:
                                                             HF jitter + a low-frequency part that must NOT
                                                             be corrected)
  preview k shows W_k = R_true_k R_est_k^T V_k; after the correction R_est' = R_est * correction(t_k).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from stillpoint.closedloop import compose_corrections, fold_residuals
from stillpoint.geom import qconj, qexp, qlog, qmul
from stillpoint.residual import measure_residuals, px_equiv

from test_residual import FPS, K, OUT_H, OUT_W, jitter_signal, pan_path, quat_to_mat, rel_quats, render_preview

PXR = K[0, 0]


def _lp(x, fs, fc):
    return sosfiltfilt(butter(2, fc, 'lowpass', fs=fs, output='sos'), x, axis=0)


def _hp(x, fs, fc):
    return x - _lp(x, fs, fc)


def simulate(n, seed, s_deg=8.0, lf_deg=0.4, amp=(0.01, 0.3), freqs=(2.0, 25.0)):
    rng = np.random.default_rng(seed)
    t = 10.0 + np.arange(n) / FPS
    V = pan_path(t - 10.0, (4.0, -2.0, 1.5))
    s = np.stack([np.deg2rad(s_deg) * np.sin(2 * np.pi * 1.1 * t + ph) for ph in (0.0, 1.3, 2.1)], 1)
    R_est = qmul(V, qexp(s))
    c = jitter_signal(t, rng, amp, freqs, n=3)
    c += np.deg2rad(lf_deg) * np.stack([np.sin(2 * np.pi * 0.25 * t + ph) for ph in (0.3, 1.0, 2.0)], 1)
    R_true = qmul(R_est, qexp(c))
    return dict(t=t, V=V, s=s, R_est=R_est, c=c, R_true=R_true)


def effective(sim, R_est):
    return qmul(qmul(sim['R_true'], qconj(R_est)), sim['V'])


def virt_to_cam(R_est, V):
    return qmul(qconj(R_est), V)


def exact_rel_err(W, V):
    """What a perfect measurement returns: log(R_exp^T R_meas)."""
    return qlog(qmul(qconj(rel_quats(V)), rel_quats(W)))


def test_sign_and_axes_exact_math():
    """No images: perfect per-pair measurements. The correction must reproduce the HF camera-frame error
    (correct sign, correct axes through the virtual->camera rotation) and leave the LF part alone."""
    sim = simulate(600, seed=11, s_deg=10.0, lf_deg=0.0)
    W = effective(sim, sim['R_est'])
    rel = exact_rel_err(W, sim['V'])
    corr, diag = fold_residuals(sim['t'], rel, np.ones(len(rel)), FPS, hp_hz=1.0, clamp_deg=2.0,
                                virt_to_cam_q=virt_to_cam(sim['R_est'], sim['V']), step_floor_px=None)
    cv = corr.rotvec(sim['t'])
    hf_c = _hp(sim['c'], FPS, 1.0)
    inner = slice(60, -60)  # filtfilt edge transients
    red = np.sqrt(np.mean(hf_c[inner] ** 2)) / np.sqrt(np.mean((hf_c - cv)[inner] ** 2))
    assert red > 20, red
    # sign: correlation with the true error is +1, not -1
    assert np.corrcoef(hf_c[inner].ravel(), cv[inner].ravel())[0, 1] > 0.99
    # no low-frequency content injected
    assert np.rad2deg(np.sqrt(np.mean(_lp(cv, FPS, 0.5)[inner] ** 2))) < 0.01
    # without the virtual->camera rotation the axes mix (s = 10 deg): noticeably worse, but same sign
    corr2 = fold_residuals(sim['t'], rel, np.ones(len(rel)), FPS, hp_hz=1.0, clamp_deg=2.0)
    red2 = np.sqrt(np.mean(hf_c[inner] ** 2)) / np.sqrt(np.mean((hf_c - corr2.rotvec(sim['t']))[inner] ** 2))
    assert 2 < red2 < red


def test_low_frequency_error_is_left_alone():
    """A slow (0.25 Hz, 0.4 deg) orientation error is intent/parallax/drift territory: no correction."""
    sim = simulate(600, seed=12, s_deg=0.0, lf_deg=0.4, amp=(0.0, 0.0))
    rel = exact_rel_err(effective(sim, sim['R_est']), sim['V'])
    corr, diag = fold_residuals(sim['t'], rel, np.ones(len(rel)), FPS, hp_hz=1.0)
    cv = corr.rotvec(sim['t'])[60:-60]
    assert np.rad2deg(np.sqrt(np.mean(cv ** 2))) < 0.4 * 0.05   # < 5 % of the LF error survives
    assert diag['lf_rms_deg'] > 0.2


def test_applying_correction_removes_scene_rotation():
    """If the preview shows the scene rotating by +e between frames because of an orientation error, the
    corrected camera estimate makes the preview's relative motion match the intended path."""
    sim = simulate(400, seed=3, s_deg=6.0, lf_deg=0.0)
    R0 = sim['R_est']
    W0 = effective(sim, R0)
    rel0 = exact_rel_err(W0, sim['V'])
    corr = fold_residuals(sim['t'], rel0, np.ones(len(rel0)), FPS, hp_hz=1.0, clamp_deg=2.0,
                          virt_to_cam_q=virt_to_cam(R0, sim['V']), step_floor_px=None)
    R1 = qmul(R0, corr(sim['t']))
    rel1 = exact_rel_err(effective(sim, R1), sim['V'])
    inner = slice(40, -40)
    before = np.sqrt(np.mean(px_equiv(rel0, K, OUT_W, OUT_H)[inner] ** 2))
    after = np.sqrt(np.mean(px_equiv(rel1, K, OUT_W, OUT_H)[inner] ** 2))
    assert after < before / 20, (before, after)
    # the opposite sign would double it
    R_bad = qmul(R0, qconj(corr(sim['t'])))
    rel_bad = exact_rel_err(effective(sim, R_bad), sim['V'])
    assert np.sqrt(np.mean(px_equiv(rel_bad, K, OUT_W, OUT_H)[inner] ** 2)) > 1.5 * before


def test_low_confidence_pairs_are_bridged_not_trusted():
    # realistic post-gyro residual jitter (0.005-0.05 deg per component)
    sim = simulate(300, seed=5, s_deg=0.0, lf_deg=0.0, amp=(0.005, 0.05))
    W = effective(sim, sim['R_est'])
    rel = exact_rel_err(W, sim['V'])
    conf = np.ones(len(rel))
    bad = np.r_[50, 51, 120, 200:204]
    rel_corrupt = rel.copy()
    rel_corrupt[bad] += np.deg2rad(2.0)          # garbage measurements (e.g. a train filling the frame)
    conf[bad] = 0.0
    corr, diag = fold_residuals(sim['t'], rel_corrupt, conf, FPS, clamp_deg=2.0)
    ref = fold_residuals(sim['t'], rel, np.ones(len(rel)), FPS, clamp_deg=2.0)
    d = np.rad2deg(np.abs(corr.rotvec(sim['t']) - ref.rotvec(sim['t'])))
    assert diag['n_bad'] == len(bad)
    peak = np.rad2deg(np.abs(ref.rotvec(sim['t'])).max())
    # bridged: nothing like the 2-degree garbage (inside a gap the jitter is simply left uncorrected)
    assert np.rad2deg(np.abs(corr.rotvec(sim['t'])).max()) < 1.2 * peak
    far = np.ones(len(d), bool)
    for b in bad:
        far[max(0, b - 25):b + 27] = False               # > ~0.4 s away from any untrusted pair
    assert d[far].max() < 0.1 * peak, (d[far].max(), peak)
    # trusting the garbage (no confidence, no outlier rejection, no jump guard) is badly wrong ...
    naive = fold_residuals(sim['t'], rel_corrupt, np.ones(len(rel)), FPS, clamp_deg=2.0, outlier_k=None,
                           jump_px=None)
    assert np.rad2deg(np.abs(naive.rotvec(sim['t']) - ref.rotvec(sim['t']))).max() > 0.5
    # ... but even with full confidence the robust rejection catches every garbage pair on its own
    robust, rdiag = fold_residuals(sim['t'], rel_corrupt, np.ones(len(rel)), FPS, clamp_deg=2.0)
    assert set(rdiag['outliers']) == set(bad.tolist())
    assert np.rad2deg(np.abs(robust.rotvec(sim['t'])).max()) < 1.2 * peak
    assert np.rad2deg(np.abs(robust.rotvec(sim['t']) - ref.rotvec(sim['t'])))[far].max() < 0.1 * peak
    # a long untrusted gap fades the correction to zero inside it
    conf2 = np.ones(len(rel))
    conf2[100:160] = 0.0
    corr2, diag2 = fold_residuals(sim['t'], rel, conf2, FPS, clamp_deg=2.0)
    assert np.abs(corr2.rotvec(sim['t'][115:145])).max() < 1e-9
    # clamp
    corr3, diag3 = fold_residuals(sim['t'], rel * 20, np.ones(len(rel)), FPS, clamp_deg=0.5)
    assert np.rad2deg(np.linalg.norm(corr3.rotvec(sim['t']), axis=1)).max() <= 0.5 + 1e-9
    # smooth in time: row-time evaluation between frames is finite and continuous
    tt = np.linspace(sim['t'][10], sim['t'][11], 7)
    q = corr(tt)
    assert np.all(np.isfinite(q)) and np.all(np.abs(np.diff(qlog(q), axis=0)) < 1e-3)
    # composition of two iterations
    cc = compose_corrections(corr, ref)
    assert np.allclose(cc(sim['t'][:5]), qmul(corr(sim['t'][:5]), ref(sim['t'][:5])))


def test_closed_loop_on_rendered_previews():
    """End to end on real-frame previews: measure -> fold -> re-render -> re-measure. The HF residual must
    drop > 5x and the correction must not carry low-frequency drift."""
    n = 150
    sim = simulate(n, seed=21, s_deg=6.0, lf_deg=0.4, amp=(0.01, 0.2), freqs=(2.0, 25.0))
    rng = np.random.default_rng(7)
    obj = dict(x0=600.0, y0=300.0, w=160.0, h=90.0, vx=-3.5, vy=0.8)
    qV = rel_quats(sim['V'])

    def run(R_est):
        W = effective(sim, R_est)
        frames = [(k, render_preview('0034', quat_to_mat(W[k]), k, rng, obj=obj)) for k in range(n)]
        return W, measure_residuals(frames, K, qV)

    W0, m0 = run(sim['R_est'])
    corr, diag = fold_residuals(sim['t'], m0['err_rotvec'], m0['conf'], FPS, hp_hz=1.0, clamp_deg=0.5,
                                virt_to_cam_q=virt_to_cam(sim['R_est'], sim['V']), px_per_rad=PXR,
                                step_floor_px=None)
    R1 = qmul(sim['R_est'], corr(sim['t']))
    W1, m1 = run(R1)
    inner = slice(30, -30)
    hf = lambda rv, fc=1.0: np.sqrt(np.mean(px_equiv(_hp(np.cumsum(rv, 0), FPS, fc), K, OUT_W, OUT_H)[inner] ** 2))
    before, after = hf(m0['err_rotvec']), hf(m1['err_rotvec'])
    before2, after2 = hf(m0['err_rotvec'], 2.0), hf(m1['err_rotvec'], 2.0)
    # truth: remaining HF camera error
    c_true_after = qlog(qmul(qconj(R1), sim['R_true']))
    true_before = np.sqrt(np.mean(px_equiv(_hp(sim['c'], FPS, 1.0), K, OUT_W, OUT_H)[inner] ** 2))
    true_after = np.sqrt(np.mean(px_equiv(_hp(c_true_after, FPS, 1.0), K, OUT_W, OUT_H)[inner] ** 2))
    lf = np.rad2deg(np.sqrt(np.mean(_lp(corr.rotvec(sim['t']), FPS, 0.5)[inner] ** 2)))
    print(f'closed loop: measured HF path {before:.3f} -> {after:.4f} px, truth {true_before:.3f} -> '
          f'{true_after:.4f} px, >2 Hz {before2:.3f} -> {after2:.4f} px, LF(corr) {lf:.4f} deg, '
          f'conf med {np.median(m0["conf"]):.3f}, '
          f'pairs/s {m0["timing"]["pairs_per_s"]:.1f}')
    assert after < before / 5
    assert after2 < before2 / 10       # well inside the pass band the loop is limited by measurement noise
    assert true_after < true_before / 5
    assert lf < 0.03


# ------------------------------------------------------------------------------------------ M2 robustness
from stillpoint.closedloop import (accept_windows, jump_guard, reject_outliers, step_series,  # noqa: E402
                                   support_weight, weighted_integrate, window_hf, window_mask)

F1080 = 810.0                                    # 1080p-eq px per rad used for the px thresholds below
AX = np.array([F1080, F1080, 636.0])


def _virt_px(corr, t):
    return corr.diagnostics['corr_virt'] if 'corr_virt' in corr.diagnostics else corr.rotvec(t)


def test_single_frame_outlier_measurement_leaves_no_lasting_jump():
    """M1 defect: one bad vision measurement on one frame pair was integrated into a lasting offset (a visible
    2-3 px jump). Inject a single-pair error of 2.5 px (1080p-eq) at an otherwise trusted pair. What the viewer
    sees is the residual r = HP(true error) - correction; its per-pair steps must never exceed the uncorrected
    (open-loop) steps by more than noise (no Stillpoint-only jump), and r must not carry a lasting offset."""
    sim = simulate(600, seed=31, s_deg=0.0, lf_deg=0.0, amp=(0.002, 0.03))
    rel = exact_rel_err(effective(sim, sim['R_est']), sim['V'])
    eps_true = _hp(np.vstack([np.zeros((1, 3)), np.cumsum(rel, 0)]), FPS, 1.0)
    conf = np.full(len(rel), 0.9)
    k = 300
    bad = rel.copy()
    bad[k, 1] += 2.5 / F1080                       # one pair: +2.5 px vertical (a step in the integrated error)
    kw = dict(fs=FPS, hp_hz=1.0, clamp_deg=2.0, px_per_rad=F1080)
    ref = fold_residuals(sim['t'], rel, conf, **kw)
    got = fold_residuals(sim['t'], bad, conf, **kw)
    naive = fold_residuals(sim['t'], bad, conf, outlier_k=None, jump_px=None, **kw)
    assert k in set(got.diagnostics['outliers'])
    s_open = step_series(eps_true, AX)
    inner = slice(60, -60)

    def sp_only(corr):
        r = eps_true - corr.diagnostics['corr_virt']
        s = step_series(r, AX)
        return ((s > 0.5) & (s > 2 * s_open))[inner].sum(), r

    n_ref, r_ref = sp_only(ref)
    n_got, r_got = sp_only(got)
    n_naive, r_naive = sp_only(naive)
    assert n_ref == 0 and n_got == 0, (n_ref, n_got)
    lasting = lambda r: (np.linalg.norm((r - r_ref) * AX, axis=1) > 0.5).sum()
    assert lasting(r_got) <= 3, lasting(r_got)     # at most the uncorrected pair itself, no lasting offset
    # without the robust stage the same input produces the M1 defect: a Stillpoint-only jump that lasts
    assert n_naive >= 1 and lasting(r_naive) > 10, (n_naive, lasting(r_naive))


def test_single_frame_blip_and_gyro_backed_spike():
    """A one-frame misregistration (two opposite pair spikes) is rejected too; a spike that coincides with a
    gyro one-frame event of comparable size is kept (pred_px cross-check)."""
    sim = simulate(400, seed=32, s_deg=0.0, lf_deg=0.0, amp=(0.002, 0.01))
    rel = exact_rel_err(effective(sim, sim['R_est']), sim['V'])
    conf = np.ones(len(rel))
    bad = rel.copy()
    bad[200, 0] += 2.0 / F1080
    bad[201, 0] -= 2.0 / F1080
    out, _ = reject_outliers(bad, conf, AX)
    assert set(np.flatnonzero(out)) == {200, 201}
    pred = np.zeros(len(rel))
    pred[200:202] = 6.0                                                   # gyro shows a 6 px one-frame event
    out2, _ = reject_outliers(bad, conf, AX, pred_px=pred)
    assert not out2[200] and not out2[201]
    # consistent HF oscillation (legit residual jitter at 20-25 Hz) is never flagged
    t = sim['t']
    osc = np.zeros((len(rel), 3))
    osc[:, 2] = np.diff(np.sin(2 * np.pi * 23 * t)) * 1.5 / 636.0
    assert reject_outliers(osc, conf, AX)[0].sum() == 0


def test_support_weight_is_smooth_and_correction_never_drops_within_a_frame():
    """M1 defect: the fold faded only on the trusted side and then dropped to exactly 0 (DJI_0032 2.9 px jump).
    With confidence collapsing mid-clip the correction must fade over >= ~0.2 s with no step."""
    sim = simulate(600, seed=33, s_deg=4.0, lf_deg=0.0, amp=(0.02, 0.1))
    rel = exact_rel_err(effective(sim, sim['R_est']), sim['V'])
    conf = np.ones(len(rel))
    conf[300:] = np.linspace(0.6, 0.05, len(rel) - 300)                  # vision goes blind
    conf[330:] = 0.02
    corr = fold_residuals(sim['t'], rel, conf, FPS, clamp_deg=2.0, px_per_rad=F1080,
                          virt_to_cam_q=virt_to_cam(sim['R_est'], sim['V']))
    w = corr.diagnostics['weight']
    assert np.abs(np.diff(w)).max() < 0.12
    assert w[:280].min() > 0.99 and w[345:].max() == 0.0
    cv = corr.diagnostics['corr_virt']
    mag = np.linalg.norm(cv * AX, axis=1)
    assert mag[:250].max() > 2.0                                          # a real (px-sized) correction ...
    # ... fades out without a Stillpoint-only jump in what the viewer sees (residual r = true HF error - corr)
    eps_true = qlog(qmul(qconj(sim['V']), effective(sim, sim['R_est'])))  # content error, virtual frame
    eps_true = _hp(eps_true, FPS, 1.0)
    s_open = step_series(eps_true, AX)
    s_r = step_series(eps_true - cv, AX)
    sp_only = (s_r > 0.5) & (s_r > 2 * s_open)
    assert not sp_only[60:-60].any(), np.flatnonzero(sp_only)
    s = support_weight(np.r_[np.ones(100), np.zeros(100)], FPS)
    assert s[0] == 1.0 and s[-1] == 0.0 and np.abs(np.diff(s)).max() < 0.12


def test_jump_guard_reverts_unsupported_step_only():
    F = 400
    c = np.zeros((F, 3))
    c[:, 0] = 0.3 / F1080 * np.sin(2 * np.pi * 3 * np.arange(F) / FPS)
    c[200:, 1] += 2.0 / F1080                                             # a 2 px step at pair 199
    sup = np.ones(F - 1)
    sup[195:205] = 0.1                                                    # ... not backed by measurements
    c2, info = jump_guard(c, sup, AX, FPS, jump_px=0.5)
    assert 199 in info['pairs']
    assert step_series(c2, AX).max() < 0.5
    assert np.allclose(c2[:170], c[:170]) and np.allclose(c2[230:], c[230:])   # local only
    c3, info3 = jump_guard(c, np.ones(F - 1), AX, FPS, jump_px=0.5)           # supported: kept
    assert info3['n_fixed'] == 0 and np.allclose(c3, c)


def test_anchored_integration_matches_plain_integration_when_trusted():
    rng = np.random.default_rng(4)
    rel = rng.normal(0, 1e-4, (500, 3)) + 2e-5
    w = np.ones(500)
    e0 = weighted_integrate(rel, w, anchor_s=None)
    e1 = weighted_integrate(rel, w, fs=FPS)
    hp = lambda x: _hp(x, FPS, 1.5)
    assert np.abs(hp(e1) - hp(e0))[50:-50].max() < 0.03 * np.abs(hp(e0)).max()


def test_window_helpers():
    fs, F = 59.94, 600
    t = np.arange(F) / fs
    eps = np.zeros((F, 3))
    eps[240:360, 0] = 1e-3 * np.sin(2 * np.pi * 6 * t[240:360])
    res = dict(k0=np.arange(F - 1), k1=np.arange(1, F), err_rotvec=np.diff(eps, axis=0), conf=np.ones(F - 1))
    h = window_hf(res, F, fs, 700.0, 60)
    assert h.argmax() == 4 and h[4] > 10 * np.median(h)
    acc = accept_windows(h * 0.5, h)
    assert acc.all()
    veto = np.zeros(10, bool)
    veto[3] = True
    acc = accept_windows(h * 0.5, h, veto=veto)
    assert not acc[3] and acc.sum() == 9
    m = window_mask(acc, F, 60, 15)
    assert m[210] < 0.01 and np.all(np.abs(np.diff(m)) < 0.2)


def test_vision_spike_as_large_as_the_gyro_jolt_is_rejected():
    """DJI_0028 57.9 s: at a real one-frame jolt (handled by the gyro) the vision fit returned a spike about the
    size of the jolt itself; folding it made a 1.2 px jump. A residual that large at a jolt is implausible (a gyro
    timing / gain error leaves a fraction of the jolt), so it can be rejected below the Hampel threshold
    (pred_ceiling; off by default on O3, where the gyro over-reports HF roll)."""
    rng = np.random.default_rng(9)
    P = 300
    rel = rng.normal(0, 0.15 / F1080, (P, 3))                 # noisy measurements, local scale ~0.2 px
    pred = np.abs(rng.normal(0, 0.2, P))
    pred[150] = 1.0                                            # the gyro shows a 1.0 px one-frame jolt
    bad = rel.copy()
    bad[150, 0] += 0.9 / F1080                                 # vision spike ~ the jolt's size
    w = np.ones(P)
    assert not reject_outliers(bad, w, AX)[0][150]             # plain Hampel: below 5x the local scale
    out, _ = reject_outliers(bad, w, AX, pred_px=pred, pred_ceiling=0.5)
    assert out[150] and out.sum() <= 3


# ------------------------------------------------------------------ O4 Pro frame steps: gyro spikes and step limiter
from stillpoint.closedloop import gyro_event_vectors, limit_steps  # noqa: E402


def _gyro_scene(n=600, seed=41, amp=(0.002, 0.02)):
    """Truth = gyro everywhere (no residual error), small real jitter in both; smooth virtual path."""
    sim = simulate(n, seed=seed, s_deg=3.0, lf_deg=0.0, amp=amp)
    return sim['R_true'], sim['V'], sim['t']


def _visible(R_true, R_est, V):
    """What the viewer sees: the content's orientation error vs the intended path (virtual frame), and its
    per-pair steps (velocity outlier, 1080p-eq px)."""
    e = qlog(qmul(qconj(V), qmul(qmul(R_true, qconj(R_est)), V)))
    return e, step_series(e, AX)


def test_false_single_frame_gyro_spike_is_cancelled():
    """O4 Pro 0004 diagnosis: a one-frame gyro spike the image does not share is injected into the output by the
    stabiliser (here 1.5 px). The preview measurement sees exactly that event; before this fix reject_outliers took
    it for a vision failure (larger than pred_gain x the gyro event) and the spike stayed. Now it is recognised as
    the image refuting the gyro and folded: no visible step. The real jitter the gyro shares is untouched."""
    R_true, V, t = _gyro_scene()
    k0 = 300
    spike = np.zeros((len(t), 3))
    spike[k0, 0] = 1.5 / F1080                                  # gyro claims a 1.5 px pitch blip at frame k0
    R_est = qmul(R_true, qexp(spike))
    W = qmul(qmul(R_true, qconj(R_est)), V)
    rel = exact_rel_err(W, V) + np.random.default_rng(0).normal(0, 0.05 / F1080, (len(t) - 1, 3))
    conf = np.full(len(rel), 0.8)
    M = virt_to_cam(R_est, V)
    g = gyro_event_vectors(M, AX)
    assert np.linalg.norm(g[k0 - 1]) > 1.2 and np.linalg.norm(g[k0 + 2]) < 0.6
    _, s0 = _visible(R_true, R_est, V)
    assert s0[k0 - 1:k0 + 1].min() > 1.4                        # uncorrected: a 1.5 px blip on screen
    kw = dict(fs=FPS, clamp_deg=2.0, px_per_rad=F1080, virt_to_cam_q=M)
    corr = fold_residuals(t, rel, conf, **kw)
    assert set(corr.diagnostics['gyro_refuted']) >= {k0 - 1, k0}
    _, s1 = _visible(R_true, qmul(R_est, corr(t)), V)
    assert s1[30:-30].max() < 0.3, s1[k0 - 3:k0 + 3]           # no visible step anywhere
    # the mechanism: without the gyro cross-check the same measurement is rejected and the blip stays
    old = fold_residuals(t, rel, conf, explain_gyro=False, **kw)
    _, s2 = _visible(R_true, qmul(R_est, old(t)), V)
    assert s2[k0 - 1:k0 + 1].min() > 1.2
    # a REAL one-frame jolt (image shares the gyro's event): nothing to fold, nothing changes
    R_true2 = qmul(R_true, qexp(spike))
    W2 = qmul(qmul(R_true2, qconj(R_est)), V)
    rel2 = exact_rel_err(W2, V)
    corr2 = fold_residuals(t, rel2, conf, **kw)
    _, s3 = _visible(R_true2, qmul(R_est, corr2(t)), V)
    assert s3[30:-30].max() < 0.1


def test_correction_does_not_step_where_the_gyro_shows_nothing():
    """0004 8.7 / 20.8 s: correction steps of 0.4-0.6 px at pairs where the gyro's own event was ~0.1-0.3 px (the
    vision chasing noise / parallax) were the independent eval's new 0.6-0.8 px jumps. A vision-only oscillation
    (+-0.5 px roll, 12 Hz, 0.6 s, consistent enough to pass the outlier test; gyro quiet) must not make the
    correction step by more than the floor; the same burst where the gyro shows events of that size is folded as
    before."""
    R_true, V, t = _gyro_scene(seed=43, amp=(0.0005, 0.002))        # a quiet gyro: events < 0.05 px
    R_est = R_true
    rel = exact_rel_err(qmul(qmul(R_true, qconj(R_est)), V), V)
    P = len(rel)
    burst = np.zeros(P + 1)
    kk = np.arange(280, 316)
    burst[kk] = 0.5 * np.sin(2 * np.pi * 12 * (kk - 280) / FPS) * np.hanning(len(kk) + 2)[1:-1] ** 0.25
    bad = rel + np.random.default_rng(3).normal(0, 0.05 / F1080, rel.shape)
    bad[:, 2] += np.diff(burst) / 636.0
    conf = np.full(P, 0.8)
    M = virt_to_cam(R_est, V)
    # outlier stage off: the burst stands for a consistent measurement that reaches the integration
    kw = dict(fs=FPS, clamp_deg=2.0, px_per_rad=F1080, virt_to_cam_q=M, outlier_k=None)
    free = fold_residuals(t, bad, conf, step_floor_px=None, **kw)
    lim = fold_residuals(t, bad, conf, **kw)
    s_free = step_series(free.diagnostics['corr_virt'], AX)
    s_lim = step_series(lim.diagnostics['corr_virt'], AX)
    assert s_free[280:316].max() > 0.5                          # unlimited: the correction steps like the burst
    assert s_lim[30:-30].max() <= 0.3 * 1.05, s_lim.max()      # limited to the 0.3 px floor
    assert lim.diagnostics['n_steps_limited'] >= 3
    # ... where the gyro shows 1 px one-frame events, the correction may follow (1.5 x the gyro event)
    pred = np.zeros(P)
    pred[275:320] = 1.0
    backed = fold_residuals(t, bad, conf, pred_px=pred, **kw)
    s_b = step_series(backed.diagnostics['corr_virt'], AX)
    np.testing.assert_allclose(s_b[285:311], s_free[285:311], atol=0.05)


def test_limit_steps_turns_a_step_into_a_ramp():
    F = 400
    c = np.zeros((F, 3))
    c[:, 0] = 0.3 / F1080 * np.sin(2 * np.pi * 3 * np.arange(F) / FPS)
    c[200:, 1] += 1.5 / F1080                                   # a 1.5 px step at pair 199
    c2, info = limit_steps(c, AX, np.full(F - 1, 0.3), FPS, relax_s=0.2)
    assert 199 in info['pairs']
    assert step_series(c2, AX).max() < 0.35
    np.testing.assert_allclose(c2[:170], c[:170], atol=0.02 / F1080)      # local only; the position is kept
    np.testing.assert_allclose(c2[230:], c[230:], atol=0.02 / F1080)
    c3, info3 = limit_steps(c, AX, np.full(F - 1, 2.0), FPS)              # allowed: untouched
    assert info3['n_limited'] == 0 and np.allclose(c3, c)
