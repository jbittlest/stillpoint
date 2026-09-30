"""Compose a per-camera-defaults scoreboard from finished gate scoreboards (no rendering, no analysis).

    .venv/bin/python scripts/compose_scoreboard.py --part o3=work/gate/v6/fill --part oa4=work/gate/v6/default \
        --part o4=work/gate/v6/default --baseline work/gate/v6/default --label v6 --out work/gate/v6

Each camera's windows come from the scoreboard run whose config equals that camera's default
(engine/stillpoint/app_bridge.py CAMERA_OPTION_DEFAULTS). Writes OUT/scoreboard.{md,json}: the per-camera sections
(vs Gyroflow / vs the original) and whole-clip analysis rows copied from those runs, the change against --baseline
(the engine v5 defaults, same ProRes renders) and the older HEVC scoreboards' headline means for context.
"""
import argparse
import json
import os
import re
import time

SECTION = {'o3': '## DJI O3', 'oa4': '## Osmo Action 4', 'o4': '## O4 Pro'}
CLIP_CAM = {'DJI_': 'o3', 'OA4_': 'oa4', 'O4_': 'o4'}
NAMES = {'o3': 'DJI O3', 'oa4': 'Osmo Action 4', 'o4': 'O4 Pro'}


def cam_of(clip):
    return next(v for k, v in CLIP_CAM.items() if clip.startswith(k))


def md_section(md, head):
    lines = md.splitlines()
    i = next(k for k, x in enumerate(lines) if x.startswith(head))
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith('## ')), len(lines))
    return '\n'.join(lines[i:j]).rstrip()


def analysis_rows(md):
    sec = md_section(md, '## Whole-clip analyses').splitlines()
    head = [x for x in sec if x.startswith('| clip') or x.startswith('|---')]
    rows = {x.split('|')[1].strip(): x for x in sec if x.startswith('| ') and not x.startswith('| clip')}
    return head, rows


def geo_segment(md, cam):
    line = next((x for x in md.splitlines() if x.startswith('Geo-mean per-window change')), '')
    m = re.search(rf'\b{cam} \(\d+ windows\): [^;]*', line)
    return m.group(0) if m else ''


def compare_rows(md, cam):
    sec = md_section(md, '## vs ').splitlines()
    head = [x for x in sec if x.startswith('| window') or x.startswith('|---')]
    rows = [x for x in sec if x.startswith('| ') and not x.startswith('| window')
            and (cam_of(x.split('|')[1].strip()) == cam if not x.startswith('| **') else f'**{cam} mean' in x)]
    return head, rows


def headline(md, head):
    sec = md_section(md, head)
    return next((x for x in sec.splitlines() if x.startswith('**') and 'windows pass' in x), '')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--part', action='append', required=True, help='cam=scoreboard_dir (cam: o3, oa4, o4)')
    ap.add_argument('--baseline', required=True, help='scoreboard dir of the engine v5 defaults (same codec)')
    ap.add_argument('--hevc', action='append', default=[], help='label=scoreboard_dir of an older HEVC scoreboard')
    ap.add_argument('--label', default='v6')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    parts = dict(x.split('=', 1) for x in a.part)
    js = {c: json.load(open(os.path.join(d, 'scoreboard.json'))) for c, d in parts.items()}
    mds = {c: open(os.path.join(d, 'scoreboard.md')).read() for c, d in parts.items()}
    base_md = open(os.path.join(a.baseline, 'scoreboard.md')).read()
    base_js = json.load(open(os.path.join(a.baseline, 'scoreboard.json')))
    codecs = {j.get('codec') for j in js.values()} | {base_js.get('codec')}
    assert len(codecs) == 1, f'mixed codecs {codecs}: not comparable'
    engines = {j['engine']['content'] for j in js.values()}
    cfg = {c: (js[c].get('params_overrides') or {}) for c in parts}
    out = [f'# Stillpoint scoreboard: {a.label} (per-camera defaults)', '',
           f'Composed {time.strftime("%Y-%m-%dT%H:%M:%S")} by `scripts/compose_scoreboard.py` from finished gate runs '
           f'(no re-render): each camera\'s windows come from the run whose config equals that camera\'s default in '
           f'`engine/stillpoint/app_bridge.py` CAMERA_OPTION_DEFAULTS. Engine content {", ".join(sorted(engines))} '
           f'(@ {js[next(iter(js))]["engine"]["head_short"]}); renders {codecs.pop()} (ProRes 422 HQ, bit-identical '
           f're-renders); judge eval 2.0. Judge noise and the decision rule: `decision.md` / `noise_prores.json`.', '',
           '| camera | default options (beyond timecal) | windows from |', '|---|---|---|']
    for c in ('o3', 'oa4', 'o4'):
        if c in parts:
            opts = ', '.join(f'{k}={v}' for k, v in cfg[c].items()) or 'none (engine defaults)'
            out.append(f'| {NAMES[c]} | {opts} | `{os.path.relpath(parts[c], a.out)}/` ({js[c]["label"]}) |')
    out += ['', 'Toggles (not defaults): Max quality (mesh residual) on every camera, exclusive with fill; horizon '
            'lock (off; not reliable on O4 Pro). See `decision.md` for the numbers behind each choice.', '']
    for c in ('o3', 'oa4', 'o4'):
        if c in parts:
            out += [md_section(mds[c], SECTION[c]), '']
    # change vs the engine v5 defaults
    out += [f'## Change vs the engine v5 defaults (`{os.path.relpath(a.baseline, a.out)}/`, same ProRes renders)', '']
    for c in ('o3', 'oa4', 'o4'):
        if c not in parts:
            continue
        if os.path.abspath(parts[c]) == os.path.abspath(a.baseline):
            out += [f'- **{NAMES[c]}**: no change (the default config is the v5 default; identical plans).']
            continue
        head, rows = compare_rows(mds[c], c)
        out += [f'- **{NAMES[c]}** ({js[c]["label"]} vs v5 default): {geo_segment(mds[c], c)}', '', *head, *rows, '']
    out += ['', 'Engine v5 default plans equal v4\'s (5/7 byte-identical, the others within 0.03 px; ENGINE_SPEC.md), '
            'so the v5-default rows stand for v4 as well.', '']
    if a.hevc:
        out += ['## Older scoreboards (HEVC renders, for context only)', '',
                'Rendered with the VideoToolbox HEVC encoder, whose output is not deterministic (HF up to ~20 % on '
                'the same plan): NOT comparable number-for-number with the ProRes tables above.', '']
        for spec in a.hevc:
            lab, d = spec.split('=', 1)
            hm = open(os.path.join(d, 'scoreboard.md')).read()
            for c in ('o3', 'oa4', 'o4'):
                h = headline(hm, SECTION[c])
                if h:
                    out.append(f'- {lab} {NAMES[c]}: {h}')
        out.append('')
    # whole-clip analyses
    head, _ = analysis_rows(mds[next(iter(parts))])
    out += ['## Whole-clip analyses (the default config of each camera)', '', *head]
    for c in ('o3', 'oa4', 'o4'):
        if c in parts:
            _, rows = analysis_rows(mds[c])
            out += [r for k, r in rows.items() if cam_of(k) == c]
    os.makedirs(a.out, exist_ok=True)
    home = os.path.expanduser('~')
    redact = (lambda t: t.replace(home, '~')) if home and home != '~' else (lambda t: t)   # public repo: no home path
    open(os.path.join(a.out, 'scoreboard.md'), 'w').write(redact('\n'.join(out) + '\n'))
    comb = dict(label=a.label, composed=time.strftime('%Y-%m-%dT%H:%M:%S'), codec=base_js.get('codec'),
                parts={c: dict(dir=os.path.abspath(parts[c]), label=js[c]['label'], params_overrides=cfg[c])
                       for c in parts},
                windows={k: w for c in parts for k, w in js[c]['windows'].items() if cam_of(k) == c},
                analyses={k: v for c in parts for k, v in js[c].get('analyses', {}).items() if cam_of(k) == c},
                summary={c: js[c]['summary'].get(c) for c in parts})
    open(os.path.join(a.out, 'scoreboard.json'), 'w').write(redact(json.dumps(comb, indent=1, default=str)))
    print(os.path.join(a.out, 'scoreboard.md'))


if __name__ == '__main__':
    main()
