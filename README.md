# Stillpoint

**A gyro + vision video stabilizer for DJI O3 / O4 / Avata and Osmo Action 4 footage. It targets the micro-jitter and jello that other gyro stabilizers leave behind.**

**[jbittlest.github.io/stillpoint](https://jbittlest.github.io/stillpoint/)** has the before/after videos and the full results.
**Stabilize a clip in your browser: [jbittlest.github.io/stillpoint/app/](https://jbittlest.github.io/stillpoint/app/)**. It runs entirely on your own machine (WebCodecs + WebGPU), and your video is never uploaded.

## What it does

DJI cameras embed their own fused attitude quaternions in the original MP4: 2 kHz on the O3, and 1 kHz on the Osmo Action 4 in 4:3 with EIS off. Stillpoint uses them like this:

1. **Gyro timing.** Every sample goes onto the video timeline. That includes the per-camera clock drift (the O3 sensor runs at 59.9693 Hz, the container says 59.94) and the Osmo Action 4's exposure-dependent picture offset.
2. **Rolling shutter, per row.** There is a separate rotation for 32 rows of every frame. Each output pixel is mapped exactly through the Kannala–Brandt fisheye model, on the GPU (`shaders/warp.metal`), with no mesh approximation.
3. **Exposure-aware rows.** At slower shutters, each row's orientation is averaged over that row's exposure window.
4. **Crop-constrained smoothing.** The path is a convex QP (Clarabel) that keeps every output corner inside the source image.
5. **Vision check.** A closed loop measures the residual rotation on low-res previews and folds confirmed corrections back in, per 1-s window. It's on for the O3 and off for the 1 kHz Osmo Action 4, where it only chased parallax.

The conventions (quaternion frames, pixel centres, row timing) are in [`ENGINE_SPEC.md`](ENGINE_SPEC.md).

## Results (honest)

The test set is five DJI O3 clips, each with a 25-s main window and a held-out window. The comparison is against the raw footage and Gyroflow 1.6 renders of the same clips, all measured with the same no-reference metric (`eval/`). Values are residual image motion in px at 1080p, and lower is better. "Calm cruise" is the >2 Hz jitter on the frames where the raw camera moves < 150 px/s.

| clip · window | HF jitter: raw / Gyroflow / **Stillpoint** | calm-cruise: raw / Gyroflow / **Stillpoint** | crop kept: Gyroflow / **Stillpoint** |
|---|---|---|---|
| DJI_0025 · 15–40 s | 2.79 / 0.97 / **0.36** | 0.91 / 0.60 / **0.32** | 63.3% / **63.0%** |
| DJI_0028 · 8–33 s | 1.98 / 1.14 / **0.84** | 0.93 / 0.93 / **0.12** | 61.2% / **60.9%** |
| DJI_0034 · 15–40 s | 5.58 / 1.31 / **0.51** | 1.42 / 0.55 / **0.13** | 57.6% / **58.1%** |
| DJI_0027 · 5–30 s | 5.54 / 3.05 / **1.69** | 1.66 / 0.48 / **0.32** | 57.6% / **58.4%** |
| DJI_0032 · 22–47 s | 4.61 / 1.21 / **0.54** | 1.40 / 0.34 / **0.13** | 54.3% / **54.7%** |
| DJI_0025 · 1–15 s (held-out) | 3.88 / 0.99 / **0.48** | 0.96 / 0.57 / **0.12** | 59.6% / **63.0%** |
| DJI_0028 · 38–58 s (held-out) | 5.21 / 4.61 / **4.03** | 1.91 / 0.76 / **0.32** | 61.2% / **60.7%** |
| DJI_0034 · 1–15 s (held-out) | 1.39 / 0.82 / **0.13** | 1.30 / 0.77 / **0.13** | 64.4% / **58.0%** |
| DJI_0027 · 100–120 s (held-out) | 2.14 / 0.84 / **0.78** | 0.99 / 0.48 / **0.14** | 60.1% / **58.4%** |
| DJI_0032 · 120–140 s (held-out) | 3.09 / 0.64 / **0.22** | 2.78 / 0.63 / **0.14** | 61.2% / **54.6%** |

* Stillpoint has lower high-frequency jitter than Gyroflow on **10 of 10** windows, and 1.5–7.5× lower calm-cruise micro-jitter on all 10.
* 8–30 Hz vibration is below the raw footage on every window. Gyroflow's renders add 8–30 Hz roll above the raw on 6 of 10.
* **Osmo Action 4:** the latest fix corrects the exposure-dependent picture offset (`telemetry.oa4_picture_offset(e)`: the frames were placed 0.92 ms early at short shutters, which is 60–100° of phase at 200–300 Hz prop vibration) and adds exposure-averaged rows. On clip 0012 (300–312 s) it took jello from 5.07 px (worse than the raw 4.93) to 1.72 px, HF jitter from 1.78 to 0.65 px, and new single-frame jumps > 1 px from 9 to 0.

**Not solved yet:**
* The strict internal gate (eval 2.0, ten criteria per window) passes on **0 of 10** windows. The main failures are occasional single-frame jumps of 1–4 px that Gyroflow doesn't have, and more jello than Gyroflow on 3 of 10 windows (0027 5–30, 0028 38–58, 0034 1–15).
* Low flights past large near objects show parallax, which no rotation can fix. DJI_0028 beside a train is the example: there the vision check is blind and Stillpoint only matches Gyroflow.
* Osmo Action 4 timing at 5–16 ms shutters isn't validated yet, and the smoother leaks a little 2–3.5 Hz roll.
* The held-out windows use the crop chosen for the whole clip. On two of them the crop is 6 points tighter than Gyroflow's.

## Supported cameras

| camera | motion data | notes |
|---|---|---|
| DJI O3 Air Unit / Avata | 2 kHz attitude | the camera the results were measured on; record with EIS/RockSteady off |
| DJI O4 Pro | high-rate attitude | parser support; less tested |
| Osmo Action 4 | 1 kHz in 4:3 with EIS off; 60 Hz per-frame in 16:9 | shoot 4K 4:3, RockSteady/HorizonSteady off, for the best result |

Always use the original MP4 from the card. Re-exports drop the motion track.

## Web stabilizer

**[jbittlest.github.io/stillpoint/app/](https://jbittlest.github.io/stillpoint/app/)**: open it, drop in an original DJI MP4, check the preview, and click Stabilize. The stabilized MP4 is saved to your computer.

* **Gyro-precise mode.** The browser app (`web/`, TypeScript + Vite) is a port of the engine's gyro path: DJI telemetry parsing and timing, per-row rolling shutter, exposure-averaged rows, the crop-constrained smoother, and the same KB4 fisheye warp in WebGPU. The Mac app's vision pass isn't in the browser yet. On DJI_0034 (15–40 s), with the crop matched to the Python engine, the browser output measured 0.76 px HF jitter (raw 5.58, Gyroflow 1.31, Python gyro-only 0.70, Mac app with vision 0.51). Its warp matches the Metal renderer on the same plan to within 0.08 px.
* **Runs locally.** Decoding, warping and encoding all happen in your browser with WebCodecs and WebGPU. Nothing is uploaded. The site is static files on GitHub Pages.
* **Browsers.** Current Chrome or Edge on a desktop or laptop, tested in Chrome on an Apple-silicon Mac. Safari 26 on a Mac has WebGPU and WebCodecs but is untested. Firefox isn't supported. O3 clips are H.264. Osmo Action 4 and O4 Pro clips are HEVC and need hardware HEVC decoding (Apple-silicon Macs, most recent Windows GPUs). The app checks the browser when it opens and says what's missing.
* **Speed.** On an Apple-silicon Mac, 4K exports run at about 40–58 fps, limited by the hardware encoder.

Build and test:

```sh
cd web
npm install
npm run dev            # http://localhost:5173
npm run typecheck && npm test
npm run build:pages    # rebuilds docs/app/ for GitHub Pages (base /stillpoint/app/)
node test/app.e2e.mjs --clip CLIP.MP4            # headless Chrome end-to-end export, checked with ffprobe
node test/app.e2e.mjs --clip CLIP.MP4 --url https://jbittlest.github.io/stillpoint/app/   # against the live site
```

## Mac app

A native SwiftUI + Metal app for macOS 14+ on Apple silicon, driving the Python engine:

```sh
# 1. Engine (Python 3.12)
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
# ffmpeg must be on PATH (Homebrew: brew install ffmpeg)

# 2. Export renderer and app
app/renderer/build.sh
app/Stillpoint/build.sh            # -> app/Stillpoint/build/Stillpoint.app (--debug for a fast build)
open app/Stillpoint/build/Stillpoint.app
```

The app finds the engine in the repo it was built from, and you can change that in Settings (⌘,). Analyses and scratch go to `~/Library/Application Support/Stillpoint`. More detail is in [`app/Stillpoint/README.md`](app/Stillpoint/README.md).

## Engine CLI and tests

```sh
PYTHONPATH=engine .venv/bin/python -m stillpoint.cli analyze CLIP.MP4 --out work/CLIP
PYTHONPATH=engine .venv/bin/python -m stillpoint.cli render CLIP.MP4 --plan work/CLIP/plan.spplan --out out.mov
PYTHONPATH=engine .venv/bin/python -m pytest tests -q
```

Tests that need real footage skip when it's missing. To point them at your own clips, set `STILLPOINT_FOOTAGE_DIR`, or the more specific `STILLPOINT_O3_DIR`, `STILLPOINT_GYROFLOW_DIR`, `STILLPOINT_OA4_DIR` and `STILLPOINT_SD_DIR` (see [`eval/footage.py`](eval/footage.py)). Without footage: 80 passed, 78 skipped.

## Layout

```
engine/stillpoint/   telemetry (DJI parsers), MP4 reader, path smoothing, plan building, pipeline, reference renderer
shaders/warp.metal   per-pixel KB4 fisheye + per-row rolling-shutter warp (Lanczos-3 / Catmull-Rom)
app/                 SwiftUI app (Stillpoint) and the Metal export renderer (sprender)
eval/                no-reference jitter metrics, the eval 2.0 gate, footprint/crop measurement
web/                 the in-browser stabilizer
docs/                this project's GitHub Pages site
```

## Credits

* DJI protobuf field references come from [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser) (MIT / Apache-2.0).
* Ideas were studied from [Gyroflow](https://github.com/gyroflow/gyroflow) (GPL-3.0). No Gyroflow code is used or copied.
* Not affiliated with DJI or Gyroflow. All footage on the site was shot by the author.

## License

[MIT](LICENSE) © 2026 jbittlest
