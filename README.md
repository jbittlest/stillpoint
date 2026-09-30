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

The numbers come from the v6 decision gate ([`work/gate/v6/scoreboard.md`](work/gate/v6/scoreboard.md), [`decision.md`](work/gate/v6/decision.md)). Every window is rendered to ProRes 422 HQ, so re-rendering the same plan gives bit-identical frames and scores. Judge noise was measured with sub-pixel shift controls. A change counts only when its 95 % confidence interval over windows excludes 0 and it is larger than that noise.

The test set is five DJI O3 clips, each with a 25-s main window and a held-out window, compared against Gyroflow 1.6 renders of the same clips. There are also five windows of an Osmo Action 4 clip and four of an O4 Pro clip, which have no Gyroflow render and are compared against the original. Everything is measured with the same no-reference metric (`eval/`). Values are residual image motion in px at 1080p, and lower is better. "Calm cruise" is the >2 Hz jitter on the frames where the raw camera moves < 150 px/s.

**Defaults per camera.** Each clip starts from its camera's defaults, which are set in one table, `CAMERA_OPTION_DEFAULTS` in `engine/stillpoint/app_bridge.py`. The Mac app and `stillpoint.cli analyze` both read them.

| camera | on by default | toggles (off by default) |
|---|---|---|
| DJI O3 | gyro path, closed-loop vision check, timing self-calibration, **full-frame fill** | Max quality (replaces fill; the two can't be combined yet), horizon lock |
| Osmo Action 4 | gyro path, timing self-calibration | full-frame fill, Max quality, horizon lock |
| DJI O4 Pro | gyro path, timing self-calibration | full-frame fill, Max quality, horizon lock (not reliable: its gravity estimate is 6–10° off in turns) |

The table below uses the DJI O3 default (fill on):

| clip · window | HF jitter: raw / Gyroflow / **Stillpoint** | calm-cruise: raw / Gyroflow / **Stillpoint** | crop kept: Gyroflow / **Stillpoint** | 1-s win-rate vs Gyroflow |
|---|---|---|---|---|
| DJI_0025 · 15–40 s | 2.79 / 0.97 / **0.30** | 0.91 / 0.60 / **0.31** | 63.3% / **62.9%** | 100% |
| DJI_0028 · 8–33 s | 1.98 / 1.14 / **0.83** | 0.93 / 0.93 / **0.14** | 61.2% / **63.3%** | 91% |
| DJI_0034 · 15–40 s | 5.58 / 1.31 / **0.36** | 1.42 / 0.55 / **0.13** | 57.6% / **62.5%** | 100% |
| DJI_0027 · 5–30 s | 5.54 / 3.05 / **0.62** | 1.66 / 0.48 / **0.18** | 57.6% / **60.4%** | 83% |
| DJI_0032 · 22–47 s | 4.61 / 1.21 / **0.51** | 1.40 / 0.34 / **0.12** | 54.3% / **58.9%** | 96% |
| DJI_0025 · 1–15 s (held-out) | 3.88 / 0.99 / **0.39** | 0.96 / 0.57 / **0.14** | 59.6% / **62.9%** | 100% |
| DJI_0028 · 38–58 s (held-out) | 5.21 / 4.61 / **1.90** | 1.91 / 0.76 / **0.40** | 61.2% / **63.3%** | 94% |
| DJI_0034 · 1–15 s (held-out) | 1.39 / 0.82 / **0.14** | 1.30 / 0.77 / **0.14** | 64.4% / **62.4%** | 100% |
| DJI_0027 · 100–120 s (held-out) | 2.14 / 0.84 / **0.79** | 0.99 / 0.48 / **0.13** | 60.1% / **60.3%** | 61% |
| DJI_0032 · 120–140 s (held-out) | 3.09 / 0.64 / **0.19** | 2.78 / 0.63 / **0.15** | 61.2% / **58.7%** | 100% |

**DJI O3:**
* Stillpoint has lower high-frequency (HF) jitter than Gyroflow on **10 of 10** windows: 0.61 px mean against 1.56, a geo-mean ratio of 0.37.
* Calm-cruise micro-jitter is 1.9–6.7× lower than Gyroflow's (0.19 px mean against 0.61).
* 8–30 Hz vibration is below the raw footage on every window.
* Pooled over all windows, Stillpoint wins 92 % of 1-s segments against Gyroflow.

**What fill adds on the O3.** Fill was compared with the old default (the engine v5 default, fill off) on the same ProRes renders:
* HF jitter −14.5 % [−24, −4], 2–8 Hz −15.7 %, roll −7.9 %. Single-frame jumps >1 px fell from 13 to 8, and win-rate rose +2.3 points.
* Nothing got worse, and analysis time rose by 0–3 %.
* Filled pixels average at most 0.16 % of the frame.

**Osmo Action 4 and O4 Pro** (no Gyroflow render, so compared against the original):
* Osmo Action 4 clip 0012: HF 0.83 px against the raw 3.11, and 3 of 5 windows pass every gate check.
* O4 Pro clip 0004: HF 0.55 px against the raw 1.14, and 1 of 4 windows pass.
* Fill is not their default. On the Osmo Action 4 its HF gain (−7 %) is within noise. On the O4 Pro it made calm cruise worse (+11.7 % [+3, +21]) and raised jumps from 6 to 10.

**Max quality (parallax mesh residual) is a toggle, not a default:**
* On the O3 it beats the old default by more than fill does: HF −24 %, calm −20 %, roll −27 %.
* Head-to-head against fill, its HF edge (−11 % [−23, +3]) is within noise, it has more single-frame jumps (18 against 8), and it costs +30–46 % analysis time.
* On the Osmo Action 4 and O4 Pro its HF change is within noise and analysis takes 5–7× as long.

**Not tested this round:** horizon lock. It stays off by default.

The earlier v4/v5 scoreboards were rendered with the VideoToolbox HEVC encoder, whose output is not deterministic (HF up to ~20 % apart on the same plan). They aren't comparable number-for-number with these tables. The v5 default plans are identical to v4's (5 of 7 byte-identical, the rest within 0.03 px), so "the old default" above covers v4 too.

**Not solved yet:**
* The strict internal gate (eval 2.0, ten criteria per window) passes on **1 of 10** O3 windows. The failures are occasional single-frame jumps of 1–3 px that Gyroflow doesn't have (3 windows), more jello than Gyroflow (2), calm cruise (2), win-rate against Gyroflow (DJI_0027: 83 % and 61 %), corner wobble (1), and a crop tighter than Gyroflow's on two held-out windows.
* Low flights past large near objects show parallax, which no rotation can fix. DJI_0028 beside a train is the example: the vision check is blind there, and it is Stillpoint's worst clip (HF 0.83 against Gyroflow's 1.14 on the main window, 1.90 against 4.61 on the held-out one).
* Osmo Action 4 timing at 5–16 ms shutters isn't validated yet.

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
PYTHONPATH=engine .venv/bin/python -m stillpoint.cli analyze CLIP.MP4 --out work/CLIP   # camera defaults; --no-fill / --mesh / --horizon-lock S override
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
