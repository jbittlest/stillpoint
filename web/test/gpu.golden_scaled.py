"""GPU golden check for SCALED outputs (owner: GPU agent): 1920x1080 and 1280x720 renders of a 4K O3 frame vs a float64
reference that warps at FULL resolution (render_ref.py, exactly the 4K fixture plan) and then Lanczos-3 downscales.

    (cd web && node test/gpu.run.mjs scaled)                     # headless Chrome: decode + warp + dump
    PYTHONPATH=engine .venv/bin/python web/test/gpu.golden_scaled.py [o3_0026] [--out DIR] [--json FILE]
Exit status 1 unless every record/size has auto-antialiased float luma PSNR > 40 dB and 8-bit output luma PSNR > 40 dB.

Reference (per record): Y' at every 4K output pixel (Lanczos-3 sampling of the decoded source plane, invalid = code 16),
CbCr at every 4K chroma site (2i, 2j+0.5) (Lanczos-3 of the source chroma plane at its left-sited position, invalid =
128), then a separable Lanczos-3 downscale by s = 3840/outW (kernel stretched by s, per-pixel normalised, clamp-to-edge,
pixel centres: c = s(p+0.5)-0.5; chroma: output chroma site (2a, 2b+0.5) -> 4K chroma-texel coordinates, same kernel).
GPU variants: 'auto' (supersampled at the plan's minification, fractional grid, then Lanczos-3 downscale — the shipped
path), 'x2'/'x3' (supersampled at exactly the 4K grid: the reference pipeline itself), 'off' (one Lanczos-3 tap per
output pixel: aliases). Also reported: PSNR vs an AREA (box) downscale of the same full-res reference.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True          # importing gpu.golden.py must not leave a __pycache__ in the repo
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'engine')))

from stillpoint import render_ref  # noqa: E402

_spec = importlib.util.spec_from_file_location('gpu_golden', os.path.join(HERE, 'gpu.golden.py'))
gg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gg)

SCRATCH = os.path.expanduser('~/Library/Application Support/Stillpoint/scratch/gpu-export')
KR, KB = 0.2126, 0.0722
KG = 1 - KR - KB


def lanczos3(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) < 3, np.sinc(x) * np.sinc(x / 3.0), 0.0)


def down_matrix(n_out: int, n_in: int, centres: np.ndarray, s: float) -> np.ndarray:
    """(n_out, n_in) normalised Lanczos-3 weights around `centres` (input texel units), stretch s, clamp-to-edge."""
    M = np.zeros((n_out, n_in))
    for o, c in enumerate(centres):
        i = np.arange(int(np.ceil(c - 3 * s)), int(np.floor(c + 3 * s)) + 1)
        w = lanczos3((i - c) / s)
        np.add.at(M[o], np.clip(i, 0, n_in - 1), w / w.sum())
    return M


def area_matrix(n_out: int, n_in: int) -> np.ndarray:
    s = n_in // n_out
    M = np.zeros((n_out, n_in))
    for o in range(n_out):
        M[o, o * s:(o + 1) * s] = 1.0 / s
    return M


def full_res_reference(plan, r: int, y8: np.ndarray, c8: np.ndarray, chunk: int = 216):
    """Y' codes at every full-res output pixel, CbCr codes at every full-res chroma site (render_ref math)."""
    W, H = plan.out_w, plan.out_h
    Yo = np.empty((H, W))
    xs = np.arange(W, dtype=np.float64)
    for r0 in range(0, H, chunk):
        r1 = min(H, r0 + chunk)
        X, Y = np.meshgrid(xs, np.arange(r0, r1, dtype=np.float64))
        S, v, _ = render_ref._source_coord(plan, r, X, Y, 3)
        Yo[r0:r1] = np.where(v, render_ref.sample_ref(y8, S[..., 0], S[..., 1], 'lanczos3'), 16.0)
    cw, ch = W // 2, H // 2
    Co = np.empty((ch, cw, 2))
    xs = 2.0 * np.arange(cw)
    for r0 in range(0, ch, chunk):
        r1 = min(ch, r0 + chunk)
        X, Y = np.meshgrid(xs, 2.0 * np.arange(r0, r1) + 0.5)
        S, v, _ = render_ref._source_coord(plan, r, X, Y, 3)
        c = render_ref.sample_ref(c8, S[..., 0] * 0.5, (S[..., 1] - 0.5) * 0.5, 'lanczos3')
        Co[r0:r1] = np.where(v[..., None], c, 128.0)
    return Yo, Co


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fixtures', nargs='*', default=['o3_0026'])
    ap.add_argument('--out', default=os.path.join(SCRATCH, 'out'))
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    res = json.load(open(os.path.join(a.out, 'result_scaled.json')))
    report = {}
    bad = []
    for name in a.fixtures:
        fx = json.load(open(os.path.join(HERE, 'fixtures', 'gpu', f'{name}.json')))
        plan = gg.fixture_plan(fx)
        W, H = plan.src_w, plan.src_h
        rep = report[name] = {}
        for r, rec in enumerate(res[name]['records']):
            if rec.get('format') != 'NV12':
                raise SystemExit(f'{name} r{r}: need NV12 source bytes (got {rec.get("format")})')
            raw = np.fromfile(os.path.join(a.out, f'scaled_{name}_r{r}_src.bin'), np.uint8)
            L = rec['rawLayout']
            y8 = raw[L[0]['offset']:L[0]['offset'] + H * L[0]['stride']].reshape(H, L[0]['stride'])[:, :W].astype(np.float64)
            uv = raw[L[1]['offset']:L[1]['offset'] + (H // 2) * L[1]['stride']].reshape(H // 2, L[1]['stride'])[:, :W]
            c8 = uv.reshape(H // 2, W // 2, 2).astype(np.float64)
            Yf, Cf = full_res_reference(plan, r, y8, c8)
            for size, so in res[name]['sizes'].items():
                oW, oH = map(int, size.split('x'))
                s = plan.out_w / oW
                assert abs(plan.out_h / oH - s) < 1e-9
                My = down_matrix(oH, plan.out_h, s * (np.arange(oH) + 0.5) - 0.5, s)
                Mx = down_matrix(oW, plan.out_w, s * (np.arange(oW) + 0.5) - 0.5, s)
                Yr = My @ Yf @ Mx.T
                Mcy = down_matrix(oH // 2, plan.out_h // 2, 0.5 * (s * (2 * np.arange(oH // 2) + 1) - 1), s)
                Mcx = down_matrix(oW // 2, plan.out_w // 2, 0.5 * (s * (2 * np.arange(oW // 2) + 0.5) - 0.5), s)
                Cr = np.stack([Mcy @ Cf[..., i] @ Mcx.T for i in range(2)], -1)
                Ya = area_matrix(oH, plan.out_h) @ Yf @ area_matrix(oW, plan.out_w).T
                o = rep.setdefault(size, {})
                ro = o[f'r{r}'] = {'k': rec['k']}
                for vname, vinfo in so['variants'].items():
                    g = np.fromfile(os.path.join(a.out, f'scaled_{name}_{size}_{vname}_r{r}_luma.f32'), np.float32).reshape(oH, oW)
                    g8 = 16.0 + 219.0 * g.astype(np.float64)
                    e = g8 - Yr
                    ro[vname] = {'grid': [vinfo['scaling']['interW'], vinfo['scaling']['interH']],
                                 'luma_psnr_db': round(gg.psnr(e), 2), 'max_abs_err': round(float(np.abs(e).max()), 3),
                                 'p999_abs_err': round(float(np.percentile(np.abs(e), 99.9)), 3),
                                 'psnr_vs_area_ref_db': round(gg.psnr(g8 - Ya), 2)}
                # the deliverable: 8-bit canvas frame -> Y'CbCr (BT.709 limited, what VideoEncoder does)
                px = np.fromfile(os.path.join(a.out, f'scaled_{name}_{size}_auto_r{r}_out_rgba.bin'), np.uint8)
                Lo = so[f'out_r{r}']['layout'][0]
                px = px[Lo['offset']:Lo['offset'] + oH * Lo['stride']].reshape(oH, Lo['stride'])[:, :oW * 4]
                rgb = px.reshape(oH, oW, 4)[..., :3].astype(np.float64) / 255.0
                yy = KR * rgb[..., 0] + KG * rgb[..., 1] + KB * rgb[..., 2]
                Y8 = 16 + 219 * yy
                Cb8 = 128 + 224 * (rgb[..., 2] - yy) / (2 * (1 - KB))
                Cr8 = 128 + 224 * (rgb[..., 0] - yy) / (2 * (1 - KR))
                # reference R'G'B' (for the clip mask): luma per pixel, chroma of its 2x2 block
                Crep = np.repeat(np.repeat(Cr, 2, 0), 2, 1)
                yn, cbn, crn = (Yr - 16) / 219, (Crep[..., 0] - 128) / 224, (Crep[..., 1] - 128) / 224
                R_ = yn + 2 * (1 - KR) * crn
                B_ = yn + 2 * (1 - KB) * cbn
                G_ = (yn - KR * R_ - KB * B_) / KG
                ing = (np.minimum(np.minimum(R_, G_), B_) >= 0) & (np.maximum(np.maximum(R_, G_), B_) <= 1)
                ref_rgb = np.clip(np.stack([R_, G_, B_], -1), 0, 1)
                ey = Y8 - Yr
                ec = np.stack([Cb8, Cr8], -1)[0::2, 0::2] - Cr
                ing_b = ing[0::2, 0::2]
                ro['output8'] = {'luma_psnr_db': round(gg.psnr(ey), 2), 'luma_psnr_in_gamut_db': round(gg.psnr(ey[ing]), 2),
                                 'luma_bias': round(float(ey.mean()), 4), 'chroma_psnr_in_gamut_db': round(gg.psnr(ec[ing_b]), 2),
                                 'chroma_bias': [round(float(v), 4) for v in ec[ing_b].mean(0)],
                                 'display_rgb_psnr_db': round(gg.psnr(255 * (rgb - ref_rgb)), 2),
                                 'out_of_gamut_frac': round(float(1 - ing.mean()), 5)}
                print(name, size, f'r{r}', json.dumps(ro))
                if not (ro['auto']['luma_psnr_db'] > 40 and ro['output8']['luma_psnr_db'] > 40):
                    bad.append(f'{name} {size} r{r}')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(report, f, indent=1)
    print('PASS' if not bad else 'FAIL: ' + ', '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
