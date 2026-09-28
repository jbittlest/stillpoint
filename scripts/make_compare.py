"""Side-by-side preview: left Gyroflow render, right Stillpoint render, same window, 1920x1080 H.264.
Top row: full frames (1/4 scale); bottom row: 2x centre crops (1/2 scale) where micro-jitter is visible.

    PYTHONPATH=engine .venv/bin/python scripts/make_compare.py GF.mp4 SP.mov OUT.mp4 --gf-start 15 --sp-start 0 --dur 25
"""
import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np


def label_png(path, texts):
    img = np.zeros((1080, 1920, 4), np.uint8)
    for (x, y), t in texts:
        (w, h), b = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x, y), (x + w + 16, y + h + b + 12), (0, 0, 0, 170), -1)
        cv2.putText(img, t, (x + 8, y + h + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gf')
    ap.add_argument('sp')
    ap.add_argument('out')
    ap.add_argument('--gf-start', type=float, required=True)
    ap.add_argument('--sp-start', type=float, default=0.0)
    ap.add_argument('--dur', type=float, default=25.0)
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    lab = os.path.join(tmp, 'labels.png')
    label_png(lab, [((10, 10), 'Gyroflow (your render)'), ((970, 10), 'Stillpoint'),
                    ((10, 550), 'Gyroflow - 2x centre crop'), ((970, 550), 'Stillpoint - 2x centre crop')]
              + ([((10, 1030), a.title)] if a.title else []))
    fc = ('[0:v]split=2[g1][g2];[g1]scale=960:540:flags=area[gt];[g2]crop=1920:1080:960:540,scale=960:540:flags=area[gb];'
          '[1:v]split=2[s1][s2];[s1]scale=960:540:flags=area[st];[s2]crop=1920:1080:960:540,scale=960:540:flags=area[sb];'
          '[gt][st]hstack[top];[gb][sb]hstack[bot];[top][bot]vstack,format=yuv420p[v0];[v0][2:v]overlay=0:0,format=yuv420p[v]')
    cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', f'{a.gf_start:.6f}', '-t', f'{a.dur:.6f}', '-i', a.gf,
           '-ss', f'{a.sp_start:.6f}', '-t', f'{a.dur:.6f}', '-i', a.sp, '-i', lab, '-filter_complex', fc,
           '-map', '[v]', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', '-shortest', a.out]
    subprocess.run(cmd, check=True)
    print(a.out)


if __name__ == '__main__':
    main()
