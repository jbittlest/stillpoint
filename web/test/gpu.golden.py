"""GPU golden check (owner: GPU agent): compare the WebGPU warp (dumps from `node test/gpu.run.mjs golden`) with the
float64 reference of engine/stillpoint/render_ref.py (same geometry as shaders/warp.metal, same resampling kernels).

    (cd web && node test/gpu.run.mjs golden)          # headless Chrome: decode + warp + dump
    PYTHONPATH=engine .venv/bin/python web/test/gpu.golden.py [o3_0026 oa4_0005] [--out DIR] [--clips DIR]
Exit status 1 unless every record has geometry mean |dS| < 0.02 px and Lanczos-3 float luma PSNR > 40 dB.

Per fixture record:
  geometry  GPU coord map (half-res grid) vs render_ref.source_map: mean / p99 / max |dS| px over pixels valid in both
            (target mean < 0.02 px), and how many pixels disagree on validity.
  luma      GPU float luma (warp_luma debug entry, before 8-bit quantisation) vs float64 reference sampling of the
            source luma plane: PSNR in 8-bit code units (peak 255), all pixels (target > 40 dB).
  output    the real deliverable: 8-bit canvas VideoFrame (RGBA copy) -> Y'CbCr through the BT.709 matrix, limited
            range (what VideoEncoder does, measured) vs the reference luma / chroma codes: PSNR + mean bias.
Source planes: 8-bit NV12 frames are the exact bytes Chrome decoded (VideoFrame.copyTo); 10-bit HEVC (opaque in
Chrome) is decoded here with PyAV in software (HEVC decoding is bit-exact).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'engine')))

from stillpoint import render_ref  # noqa: E402
from stillpoint.geom import Lens  # noqa: E402
from stillpoint.types import Plan  # noqa: E402

SCRATCH = os.path.expanduser('~/Library/Application Support/Stillpoint/scratch/gpu')
KR, KB = 0.2126, 0.0722
KG = 1 - KR - KB


def fixture_plan(fx: dict) -> Plan:
    L = fx['lens']
    lens = Lens(L['model'], L['fx'], L['fy'], L['cx'], L['cy'], np.array(L['k'][:4], dtype=np.float64), L['width'], L['height'])
    R, n = len(fx['records']), fx['nRows']
    mats = np.array([r['rowMats'] for r in fx['records']], dtype=np.float32).astype(np.float64).reshape(R, n, 3, 3)
    return Plan(src_w=fx['srcW'], src_h=fx['srcH'], out_w=fx['outW'], out_h=fx['outH'], lens=lens,
                frame_pts=np.array([r['pts'] for r in fx['records']]),
                out_fx=np.array([r['outFx'] for r in fx['records']], dtype=np.float32).astype(np.float64),
                row_mats=mats, virt_q=np.zeros((R, 4)), meta={'readout_s': fx['readoutS']})


def decode_planes_pyav(path: str, frame_index: int):
    """Software-decode frame `frame_index` (presentation order) -> (Y code, CbCr code (H/2,W/2,2), bits)."""
    import av
    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.thread_type = 'AUTO'
        for i, fr in enumerate(c.decode(vs)):
            if i < frame_index:
                continue
            nd = fr.to_ndarray(format='yuv420p10le' if '10' in fr.format.name else 'yuv420p')
            H, W = fr.height, fr.width
            y = nd[:H].astype(np.float64)
            u = nd[H:H + H // 4].reshape(H // 2, W // 2).astype(np.float64)
            v = nd[H + H // 4:].reshape(H // 2, W // 2).astype(np.float64)
            return y, np.stack([u, v], -1), (10 if '10' in fr.format.name else 8)
    raise ValueError('frame not found')


def psnr(err: np.ndarray, peak: float = 255.0) -> float:
    rms = float(np.sqrt(np.mean(np.square(err))))
    return float('inf') if rms == 0 else 20 * np.log10(peak / rms)


def reference(plan: Plan, r: int, ycode: np.ndarray, ccode: np.ndarray, kernel: str, chunk: int = 240):
    """float64 reference luma/chroma codes of every output pixel + validity (render_ref math, all output pixels)."""
    W, H = plan.out_w, plan.out_h
    Yo = np.zeros((H, W)); Co = np.zeros((H, W, 2)); ok = np.zeros((H, W), bool)
    xs = np.arange(W, dtype=np.float64)
    for r0 in range(0, H, chunk):
        r1 = min(H, r0 + chunk)
        X, Y = np.meshgrid(xs, np.arange(r0, r1, dtype=np.float64))
        S, v, _ = render_ref._source_coord(plan, r, X, Y, 3)
        Yo[r0:r1] = render_ref.sample_ref(ycode, S[..., 0], S[..., 1], kernel)
        Co[r0:r1] = render_ref.sample_ref(ccode, S[..., 0] * 0.5, (S[..., 1] - 0.5) * 0.5, kernel)
        ok[r0:r1] = v
    return Yo, Co, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fixtures', nargs='*', default=['o3_0026', 'oa4_0005'])
    ap.add_argument('--out', default=os.path.join(SCRATCH, 'out'))
    ap.add_argument('--clips', default=os.path.join(SCRATCH, 'clips'))
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    res = json.load(open(os.path.join(a.out, 'result_golden.json')))
    report = {}
    for name in a.fixtures:
        fx = json.load(open(os.path.join(HERE, 'fixtures', 'gpu', f'{name}.json')))
        plan = fixture_plan(fx)
        W, H = plan.src_w, plan.src_h
        rep = report[name] = []
        for r, rec in enumerate(res[name]['records']):
            o = {'k': rec['k'], 'color': {kk: rec['color'][kk] for kk in ('mode', 'transfer', 'how')}}
            # ---- geometry
            gm = np.fromfile(os.path.join(a.out, f'golden_{name}_r{r}_coord050.f32'), np.float32)
            gw, gh = rec['coordMap']
            gm = gm.reshape(gh, gw, 3).astype(np.float64)
            ref, val = render_ref.source_map(plan, r, out_scale=gw / plan.out_w, return_valid=True)
            gval = gm[..., 2] > 0.5
            both = gval & val
            d = np.abs(gm[..., :2] - ref)[both]
            o['geom'] = {'grid': [gw, gh], 'mean_px': float(d.mean()), 'p99_px': float(np.percentile(d, 99)),
                         'max_px': float(d.max()), 'valid_mismatch': int((gval != val).sum()), 'valid_frac': float(val.mean())}
            # ---- source planes
            if rec.get('format') == 'NV12':
                raw = np.fromfile(os.path.join(a.out, f'golden_{name}_r{r}_src.bin'), np.uint8)
                L = rec['rawLayout']
                y = raw[L[0]['offset']:L[0]['offset'] + H * L[0]['stride']].reshape(H, L[0]['stride'])[:, :W].astype(np.float64)
                uv = raw[L[1]['offset']:L[1]['offset'] + (H // 2) * L[1]['stride']].reshape(H // 2, L[1]['stride'])[:, :W]
                c = uv.reshape(H // 2, W // 2, 2).astype(np.float64)
                bits = 8
            else:
                y, c, bits = decode_planes_pyav(os.path.join(a.clips, fx['clip']), rec['sample'])
            sc = 1.0 if bits == 8 else 4.0          # code units per 8-bit code
            y8, c8 = y / sc, c / sc                  # reference in 8-bit code units (float)
            for kern in ('lanczos3', 'catmullrom'):
                Yr, Cr, ok = reference(plan, r, y8, c8, kern)
                Yr = np.where(ok, Yr, 16.0)
                g = np.fromfile(os.path.join(a.out, f'golden_{name}_r{r}_luma_{kern}.f32'), np.float32).reshape(plan.out_h, plan.out_w)
                g8 = 16.0 + 219.0 * g.astype(np.float64)
                e = g8 - Yr
                o[f'luma_{kern}'] = {'psnr_db': psnr(e), 'psnr_valid_db': psnr(e[ok]), 'mean_err': float(e[ok].mean()),
                                     'max_abs_err': float(np.abs(e[ok]).max()), 'p999_abs_err': float(np.percentile(np.abs(e[ok]), 99.9))}
                if kern == 'lanczos3':
                    # ---- the deliverable (8-bit canvas VideoFrame) through the encoder's (measured) BT.709 conversion
                    px = np.fromfile(os.path.join(a.out, f'golden_{name}_r{r}_out_rgba.bin'), np.uint8)
                    Lo = rec['out']['layout'][0]
                    px = px[Lo['offset']:Lo['offset'] + plan.out_h * Lo['stride']].reshape(plan.out_h, Lo['stride'])[:, :plan.out_w * 4]
                    rgb = px.reshape(plan.out_h, plan.out_w, 4)[..., :3].astype(np.float64) / 255.0
                    yy = KR * rgb[..., 0] + KG * rgb[..., 1] + KB * rgb[..., 2]
                    Y8 = 16 + 219 * yy
                    Cb8 = 128 + 224 * (rgb[..., 2] - yy) / (2 * (1 - KB))
                    Cr8 = 128 + 224 * (rgb[..., 0] - yy) / (2 * (1 - KR))
                    Cr_ = np.where(ok[..., None], Cr, 128.0)
                    ey = Y8 - Yr
                    ec = np.stack([Cb8, Cr8], -1) - Cr_
                    # clipped reference pixels (super-white / out of gamut) cannot survive an 8-bit RGB canvas
                    yref_n = (Yr - 16) / 219
                    cb_n, cr_n = (Cr_[..., 0] - 128) / 224, (Cr_[..., 1] - 128) / 224
                    R_ = yref_n + 2 * (1 - KR) * cr_n
                    B_ = yref_n + 2 * (1 - KB) * cb_n
                    G_ = (yref_n - KR * R_ - KB * B_) / KG
                    ing = ok & (np.minimum(np.minimum(R_, G_), B_) >= 0) & (np.maximum(np.maximum(R_, G_), B_) <= 1)
                    # as displayed: players clip R'G'B' to [0,1] (limited-range video), so compare clipped RGB (8-bit units)
                    ref_rgb = np.clip(np.stack([R_, G_, B_], -1), 0, 1)
                    ref_rgb[~ok] = 0
                    o['display_rgb_psnr_db'] = psnr(255 * (rgb - ref_rgb))
                    # end to end through the browser's encoder (record 0 only): decoded Y vs the display-clipped reference
                    if r == 0:
                        Yd = 16 + 219 * (KR * ref_rgb[..., 0] + KG * ref_rgb[..., 1] + KB * ref_rgb[..., 2])
                        for cname, e2 in (res[name].get('e2e') or {}).items():
                            fn = os.path.join(a.out, f'golden_{name}_r0_e2e_{cname}.bin')
                            if not isinstance(e2, dict) or not os.path.exists(fn):
                                o[f'e2e_{cname}'] = str(e2)
                                continue
                            raw = np.fromfile(fn, np.uint8)
                            L0 = e2['layout'][0]
                            yd = raw[L0['offset']:L0['offset'] + plan.out_h * L0['stride']].reshape(plan.out_h, L0['stride'])[:, :plan.out_w]
                            ee = yd.astype(np.float64) - Yd
                            o[f'e2e_{cname}'] = {'luma_psnr_db': psnr(ee), 'luma_bias': float(ee.mean()),
                                                 'mbit': e2['bytes'] * 8 / 1e6, 'enc_colorspace': e2.get('encColorSpace')}
                    o['output8'] = {'luma_psnr_db': psnr(ey), 'luma_psnr_in_gamut_db': psnr(ey[ing]),
                                    'luma_bias': float(ey[ok].mean()), 'chroma_psnr_db': psnr(ec),
                                    'chroma_psnr_in_gamut_db': psnr(ec[ing]), 'chroma_bias': ec[ok].mean(0).tolist(),
                                    'out_of_gamut_frac': float(1 - ing[ok].mean()),
                                    'ts': rec['out']['ts'], 'src_ts': rec['ts'], 'format': rec['out']['format']}
            rep.append(o)
            print(name, json.dumps(o))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(report, f, indent=1)
    bad = [f"{n} k={o['k']}" for n, recs in report.items() for o in recs
           if not (o['geom']['mean_px'] < 0.02 and o['luma_lanczos3']['psnr_db'] > 40)]
    print('PASS' if not bad else 'FAIL: ' + ', '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
