"""Draw Resources/AppIcon.icns (dark squircle, white ring, blue still point). Run: ../../.venv/bin/python make_icon.py"""
import os, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFilter

def squircle_mask(n, r=0.225):
    m = Image.new('L', (n, n), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, n - 1, n - 1], radius=int(n * r), fill=255)
    return m

def icon(n=1024):
    S = 4 * n                                     # supersample
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    pad = int(S * 0.098)                          # macOS icon grid: 824/1024 body
    body = S - 2 * pad
    bg = Image.new('RGBA', (body, body))
    d = ImageDraw.Draw(bg)
    for y in range(body):                         # vertical gradient
        t = y / body
        c = tuple(int(a + (b - a) * t) for a, b in zip((38, 38, 46), (10, 10, 13)))
        d.line([(0, y), (body, y)], fill=c + (255,))
    img.paste(bg, (pad, pad), squircle_mask(body))
    d = ImageDraw.Draw(img)
    cx = cy = S / 2
    R = body * 0.30
    w = body * 0.052
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(244, 244, 246, 255), width=int(w))
    r = body * 0.085
    glow = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([cx - r * 2.2, cy - r * 2.2, cx + r * 2.2, cy + r * 2.2], fill=(77, 163, 255, 60))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(S * 0.03)))
    ImageDraw.Draw(img).ellipse([cx - r, cy - r, cx + r, cy + r], fill=(77, 163, 255, 255))
    return img.resize((n, n), Image.LANCZOS)

base = icon(1024)
with tempfile.TemporaryDirectory() as t:
    iset = os.path.join(t, 'AppIcon.iconset')
    os.makedirs(iset)
    for s in (16, 32, 128, 256, 512):
        base.resize((s, s), Image.LANCZOS).save(os.path.join(iset, f'icon_{s}x{s}.png'))
        base.resize((2 * s, 2 * s), Image.LANCZOS).save(os.path.join(iset, f'icon_{s}x{s}@2x.png'))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Resources', 'AppIcon.icns')
    subprocess.run(['iconutil', '-c', 'icns', iset, '-o', out], check=True)
    base.save(os.path.join(os.path.dirname(out), 'AppIcon.png'))
print('wrote', out)
