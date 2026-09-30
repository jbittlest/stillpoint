# Stillpoint.app

A native SwiftUI Mac app (macOS 14+, Apple silicon) for the Stillpoint engine.

## Build and run

```
app/renderer/build.sh           # sprender, the export renderer (once)
app/Stillpoint/build.sh         # -> app/Stillpoint/build/Stillpoint.app (ad-hoc signed; --debug for a fast -Onone build)
open app/Stillpoint/build/Stillpoint.app
open -a app/Stillpoint/build/Stillpoint.app ~/Movies/DJI_0034.MP4   # opens with clips
```

On a first launch from a new build, Gatekeeper may ask for confirmation (the app is ad-hoc signed, not notarized).
The engine folder defaults to the repo the app was built from (`build.sh` records it in the bundle's Info.plist;
`STILLPOINT_ROOT` overrides it). You can change it in Settings (⌘,).
The headless self-tests look for clips via `STILLPOINT_O3_DIR`, `STILLPOINT_OA4_DIR`, `STILLPOINT_SD_DIR` (see `eval/footage.py`).

## What it does

**Clips**
- Add clips by dropping them on the window or the Dock icon, or with ⌘O.
- Each clip is probed through `python -m stillpoint.app_bridge probe`. The list shows the camera, resolution, frame rate and duration.
- Each clip also gets a gyro badge:
  - "2 kHz gyro" or "1 kHz gyro"
  - "60 Hz attitude — limited"
  - "In-camera EIS on — not supported yet"

**Analyze (⌘R)**
- Before you start, the clip card shows a pre-flight: clip length, an estimated analysis time (30 s + 3× the clip length until this Mac has timed runs; then the median speed of the last runs of 15 s or longer, from `speed_history.json`), and free disk space. Below the engine's minimum (≈4.3 GB with ENGINE v3) Analyze is blocked; below 10 GB it warns. A clip on removable media (SD card) gets a warning: the analysis reads it directly.
- Runs `app_bridge analyze` and reports JSON-lines progress: stage (Gyro, Path, Vision, Check, Plan), %, and time left (the engine's ETA; before it has one, the pre-flight estimate minus elapsed time, marked "est.").
- The analysis queue runs one clip at a time. The sidebar's Analysis panel lists the running and queued clips, each with its own cancel button, the running clip's time left and each queued clip's pre-flight estimate.
- Stall watchdog: if the engine sends no progress for 45 s, the card shows "Still working — no update for N s" and whether the engine's process group is using CPU (busy) or idle (maybe stuck), next to Cancel.
- If the engine exits abnormally (a crash, a kill, a non-zero exit without a result), an error panel shows what happened and the engine's last stderr lines, with Retry and Copy Details.
- Cancelling never deletes the previous result.

**Never freezes**
- The bridge is spawned (posix_spawn) as the leader of its own process group. Cancel sends SIGTERM to the whole group (the bridge stops cooperatively within 4 s), then SIGKILL to the group after 5 s. Quitting the app does the same, synchronously, for every running job. Anything left in a finished job's group is terminated too.
- Scratch of a killed run (`work/tmp/analyze-<pid>`) is removed by the app.

**Storage (not iCloud)**
- Everything lives in `~/Library/Application Support/Stillpoint/` (the Desktop is iCloud-synced):
  - `analyses/<clip>-<hash>/` holds plan.spplan, report.json, analysis.npz and stillpoint_app.json.
  - `work/` is the bridge's `STILLPOINT_WORK_DIR`: temporary job files and telemetry caches.
  - `speed_history.json` holds analysis timings.
- On first launch, finished analyses from the old `work/app/analyses` are copied over. Only the small result files are copied (nothing over 50 MB, no temporary files), and the old copies stay where they are. `Stillpoint --migrate` does this once and exits.

**Preview**
- Playback runs through AVPlayer with a custom `AVVideoCompositing` compositor. The compositor loads `shaders/warp.metal` at runtime and warps every decoded frame with the `.spplan` record for that frame's PTS.
- While paused, a frame is bit-identical to the export: the self-test measures infinite PSNR against `sprender`.
- While playing, the compositor uses the kernel's Catmull-Rom filter, which is cheaper. The geometry is the same.
- Engine v5 plans preview with the export's own kernels:
  - Mesh residual ("Max quality") plans use `sp_warp_*_mesh` with the plan's per-frame mesh block, paused and playing.
  - Full-frame fill plans use `sp_warp_*_fill`. The compositor only receives the current frame, so the neighbouring source frames the plan's FILL section names are decoded with AVAssetReader (same decoder, pixel format and PTS matching as sprender) into a small cache (≤ 480 MB).
    - Paused: the frame renders at once with the cached neighbours, the missing ones are fetched in the background and the frame is redrawn, identical to the export.
    - Playing: only cached neighbours are used; the rest of the border is the kernel's soft edge extension. The viewer says "FILL · EXACT WHEN PAUSED". It never shows black corners.
  - Split view = the stabilised frame from those kernels with the "before" half drawn over it.
- Compare modes (⌘1, ⌘2, ⌘3, or \\):
  - After
  - Split, with a draggable divider
  - Before: the original in the same projection. The View menu can switch Before to the untouched fisheye.
- Transport: space plays or pauses, ← and → step one frame.

**Jitter readout (honest numbers)**
- If `report.json` has `quality` (ENGINE v3's independent measurement: eval's KLT estimator on the rectified original vs the final plan, over sampled windows), the card is tagged INDEPENDENT. It shows original → stabilized for shake above 2 Hz, calm cruise, 8–30 Hz and jello, in 1080p px. It also shows new frame steps > 1 px, how much of the source frame is used, how much of the clip was measured, and the method.
- The timeline strip marks the windows that were measured.
- Without `quality`, the card is tagged NO INDEPENDENT CHECK. It shows the gyro's original shake, then the loop's numbers labelled "closed-loop residual (self-measured)".
- The card never shows a derived percentage, a floor such as "<0.05", or any number the engine did not produce.
- Timing auto-calibration (engine v5 `timecal`): the offset it applied, or "Metadata timing confirmed" with the fitted offset ± sigma, or "kept" / "not run" with the engine's reason (`summary['timecal']`).
- The options the analysis used, with what they did: horizon lock (strength, bank limit, share of frames fully level), fill (share of frames filled), Max quality (mesh correction size).

**Controls**
- Smoothness and field of view.
- Engine v5 options, in one group. What a clip starts with, what is supported, which options cannot be combined, per-camera caveats and slider ranges all come from the engine: `probe` returns `options` (`app_bridge.option_info`). The defaults are `AnalyzeParams`' own, overridden per camera only in `app_bridge.CAMERA_OPTION_DEFAULTS`, the one place to change them (today: full-frame fill on for the DJI O3, from the v6 ProRes gate; Max quality's time estimate is per camera, `CAMERA_TIME_FACTORS`).
  - Horizon lock (Beta): a switch, a Strength slider and a Bank limit slider (0–45°, 0 = fully level). It needs gravity data in the clip. The engine's per-camera note is shown under it (O4 Pro: "not reliable yet").
  - Full-frame fill: fills the corners from neighbouring frames, so Stillpoint can keep more of the frame.
  - Max quality (mesh residual): removes extra micro-jitter; the analysis takes longer (the pre-flight estimate includes the engine's time factor).
  - Options the engine cannot combine (today fill and Max quality) disable each other with a hint; the app never sends both.
  - Every option the panel shows is sent to `app_bridge analyze` explicitly (`--horizon-lock S --roll-limit D --fill|--no-fill --mesh|--no-mesh`); the timing self-calibration keeps the engine default.
- Re-analyze. When any setting differs from the cached analysis, a "Settings changed" banner names what changed.

**Export (⌘E, ⇧⌘E for all)**
- Formats:
  - HEVC 10-bit at 180 Mb/s (the default)
  - HEVC Fast
  - ProRes 422 HQ
- Pick the output folder in the app. The bridge won't start an export if the folder's volume can't hold the expected file plus 1 GB.
- Progress comes from sprender's own frame counter: `PROGRESS frame=i total=n` lines, about every 0.5 s. It is not estimated from the file size.
- Exports run through a batch queue in the sidebar. Each finished export has a reveal-in-Finder button.

## Headless checks (no window, no Dock icon, no screen capture)

```
Stillpoint.app/Contents/MacOS/Stillpoint --selftest [--clip C] [--plan P] [--frames 5,150] [--fill-plan P] [--mesh-plan P] [--no-v5]
STILLPOINT_SUPPORT_DIR=<scratch> Stillpoint.app/Contents/MacOS/Stillpoint --selftest-app [--clip C]
Stillpoint.app/Contents/MacOS/Stillpoint --snapshot DIR [--scale 2] [--hero CLIP] [--long CLIP] [--time 20]
Stillpoint.app/Contents/MacOS/Stillpoint --migrate
BUILD_DIR=<dir> ./build.sh --debug       # build a scratch copy instead of build/Stillpoint.app
```

- `--selftest` checks the compositor against `sprender --dump-frames`:
  - PSNR must be above 40 dB (it is infinite).
  - The split kernel must match `sp_warp_*` bit for bit.
  - The plan record lookup must be correct.
  - It also runs the AVPlayer real-time check.
  - Engine v5: a fill plan and a mesh plan of the same clip (given, or analysed once through the bridge into `<scratch>/selftest/v5` and reused; the fill plan at a wide field of view so the border fill has work). Each must match `sprender` at the frames that use it most, split(0) must equal the kernel's output, the mesh must move pixels, and the fill preview must have no black border paused (exact) or playing (soft edge).
- `--selftest-app` walks the whole app flow. Run it with `STILLPOINT_SUPPORT_DIR` pointing at a scratch folder. It checks:
  - migration
  - probe
  - pre-flight
  - engine v5 options: the clip starts from the engine's defaults; the panel's settings round-trip through `analyze --dry-run` into AnalyzeParams and back from the manifest params (not stale), and each change flags re-analysis; fill and Max quality disable each other and the bridge refuses both
  - the timing auto-calibration readout of the fresh analysis
  - a re-analysis with Full-frame fill: FILL plan, manifest, preview
  - analyze, including the speed history
  - export progress from PROGRESS lines
  - cancelling a real analysis: the whole process group must be gone, checked with `ps`
  - a fake bridge that stalls and ignores SIGTERM: watchdog, then SIGKILL of the group after 5 s
  - a fake crash: error panel, scratch cleanup, Retry
  - a structured disk-full error
  - quit while running
  - no orphan processes at the end (`ps`)
- `--selftest-app` and `--snapshot` keep the export folder / preset in a throwaway defaults suite (`com.jimmybittleston.stillpoint.headless`), so a test, even one killed half-way, never changes the app's real settings.
- `--snapshot` writes PNGs of the main views with ImageRenderer. States that need a live engine use illustrative values, and their file names say so.

## Files

| File | Contents |
|---|---|
| `Sources/Engine.swift` | Engine path, app storage + migration, process-group spawning / kill, the bridge process (JSON-lines), protocol models |
| `Sources/Warp.swift` | `warp.metal` compiled at runtime (plus appended split/before kernels), the compositor, single-frame renderer |
| `Sources/Player.swift` | AVPlayer controller (chasing exact seeks, redraw on parameter change) |
| `Sources/Models.swift` | Clips, analysis queue, stall watchdog, pre-flight, speed history, export queue |
| `Sources/Views.swift`, `Sources/Inspector.swift`, `Sources/Theme.swift` | UI, drawn in pure SwiftUI so snapshots match the app |
| `Sources/SelfTest.swift`, `Sources/AppFlowTest.swift`, `Sources/Snapshot.swift` | The headless modes |
| `../../engine/stillpoint/app_bridge.py` | The bridge (`probe`, `analyze`, `render`, `summary`, `adopt`); tests in `tests/test_app_bridge.py` |
