"""Slit-scan (kymograph) for eyeballing jitter/wobble: the centre column (or row) of every frame, stacked in time.
Vertical jitter shows as wiggles in horizontal features of the column image; wobble/jello as bending.

    .venv/bin/python scripts/kymo.py VIDEO START DUR OUT.png [--axis col|row] [--width 960] [--label TEXT]
"""
import argparse
import subprocess

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('start', type=float)
    ap.add_argument('dur', type=float)
    ap.add_argument('out')
    ap.add_argument('--axis', default='col')
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    W, H = a.width, a.width * 9 // 16
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{a.start}', '-i', a.video, '-t', f'{a.dur}', '-vf',
           f'scale={W}:{H}:flags=area,format=gray', '-fps_mode', 'passthrough', '-f', 'rawvideo', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    lines = []
    while True:
        b = p.stdout.read(W * H)
        if len(b) < W * H:
            break
        f = np.frombuffer(b, np.uint8).reshape(H, W)
        lines.append(f[:, W // 2] if a.axis == 'col' else f[H // 2, :])
    img = np.stack(lines, 1) if a.axis == 'col' else np.stack(lines, 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if a.label:
        cv2.putText(img, a.label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(a.out, img)
    print(a.out, img.shape)


if __name__ == '__main__':
    main()
