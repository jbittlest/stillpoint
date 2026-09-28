# Stillpoint — Engine Build Spec (M1)

Authoritative spec for implementation agents. Research behind every choice: `research/*.md`
(read the sections relevant to your package — they contain verified facts, prototypes and numbers).

Goal of M1: on Jimmy's O3 clips, rendered MP4s that beat his Gyroflow renders on the validated
metric (`eval/`): calm-cruise high-frequency (>2 Hz) jitter ≤ 0.15 px (1080p-equivalent) vs
Gyroflow 0.34–0.93; 8–30 Hz never worse than the original; per-second win rate ≥ 90%; crop area
≥ Gyroflow's (68–81% of source area); no visible wobble/warping/breathing.

## 0. Layout

```
stillpoint/
  ENGINE_SPEC.md          this file
  .venv/                  shared python (run: PYTHONPATH=engine .venv/bin/python ...)
  engine/stillpoint/      python engine package
    geom.py      [DONE]   quaternions, SO(3), KB4 lens, row timing
    types.py     [DONE]   Telemetry, TimeModel, Plan dataclasses
    plan_io.py   [DONE]   .spplan read/write
    telemetry.py [WP-A]   DJI O3 + Osmo Action 4 -> Telemetry
    video.py     [WP-A]   probe + fast analysis-frame decoding
    plan_build.py[WP-B]   orientation + virtual path -> Plan
    render_ref.py[WP-B]   python reference renderer (preview/tests), same math as the Metal kernel
    smooth.py    [WP-C]   crop-constrained path optimizer
    residual.py  [WP-D]   vision residual rotation measurement (DIS + mask + ECC)
    closedloop.py[WP-D]   fold measured residuals back into the camera orientation
    calib.py     [WP-E]   self-calibration of time offset / readout / focal / extrinsic
    pipeline.py  [WP-F]   analyze -> plan -> closed loop -> render -> eval
    cli.py       [WP-F]   `python -m stillpoint.cli analyze|render|eval ...`
  shaders/warp.metal      [WP-B] the ONE warp kernel (Swift renderer loads it at runtime)
  app/renderer/           [WP-B] Swift CLI `sprender` (AVFoundation -> Metal -> AVAssetWriter)
  tests/                  pytest; each WP owns tests/test_<module>.py
  eval/                   existing validated jitter metric (do not break its CLI)
  work/                   big intermediates (cache/, renders/, ...)
```
Each file above is owned by exactly one work package. Do not edit files you don't own; if you
need a change in a DONE/foreign file, write the request in your final report instead (tiny
additive helpers in geom.py are OK if you also add a test).

## 1. Conventions (never deviate)

- Quaternions (w,x,y,z), Hamilton, float64. Orientation q maps CAMERA vectors to WORLD: v_w = R(q) v_c.
- Camera frame: x right, y down, z forward. DJI body (FRD) -> camera: cam_x = body_y, cam_y = body_z,
  cam_z = body_x (verified on both cameras; see research/footage_*.md).
- Pixels: centres at integers, (0,0) = centre of top-left pixel. A centred lens on WxH has cx=(W-1)/2.
- Time: seconds on the VIDEO timeline (frame PTS). `frame_t[k]` = centre-row mid-exposure time of
  frame k. Row y of frame k: `t = frame_t[k] + readout*((y+0.5)/H - 0.5)` (top->bottom readout).
- IMU sample times are on a UNIFORM grid (1/imu_rate) mapped to the video timeline — not the
  per-frame-spread timing telemetry-parser/Gyroflow use (that is a known sawtooth bug).
- Source lens: DJI's embedded KB4 fisheye (`geom.Lens`, model 'kb4'). Output: rectilinear pinhole,
  square pixels, centred, focal `out_fx` (per frame; constant within a shot unless the optimizer
  needs to zoom).
- Mapping for output pixel p of frame k: `r_v = ((px-out_cx)/out_fx, (py-out_cy)/out_fx, 1)`;
  `r_c = R_cam(t_row)^T · R_virt[k] · r_v`; source pixel = `lens.project(r_c)`. RS: t_row depends on
  the SOURCE row, found by fixed-point iteration starting at the centre row — 3 iterations.

## 2. Module interfaces

```python
# telemetry.py  (WP-A)
def load_telemetry(path: str, cache_dir: str | None = 'work/cache') -> Telemetry
#   Supports DJI O3 air unit (dvtm_wm169, 2 kHz quats), DJI Osmo Action 4 (dvtm_ac203: 1 kHz quats in
#   4:3 EIS-off clips; otherwise per-frame 60 Hz cam_quat, shifted by its measured ~18 ms delay).
#   Sets has_highrate, eis_baked, lens, readout_s, frame_t (centre-row mid-exposure), segments
#   (split at repeated clip headers / timestamp discontinuities in joined files).
# video.py  (WP-A)
def probe(path) -> dict   # width,height,fps,n_frames,frame_pts(np f64, exact per-frame),codec,pix_fmt,color tags
def iter_gray(path, width=960, start_frame=0, n_frames=None) -> Iterator[tuple[int, float, np.ndarray]]
#   yields (frame_index, pts, uint8 HxW). Uses ffmpeg -hwaccel videotoolbox + scale_vt,
#   -fps_mode passthrough (no dup/drop). Frame indices MUST align with probe()['frame_pts'].
def read_gray_frames(path, idx: np.ndarray, width=960) -> np.ndarray   # random access helper (N,H,W)

# plan_build.py  (WP-B)
def camera_orientation_fn(tel: Telemetry, tm: TimeModel, correction=None) -> Callable[[np.ndarray], np.ndarray]
#   t -> q_cam(t) = imu_orientation(t*(1+skew)+offset) * qexp(extrinsic) [* correction(t)]
def build_plan(tel, tm, q_cam_fn, virt_q (F,4), out_fx (F,), out_w, out_h, n_rows=32,
               frames: np.ndarray | None = None) -> Plan
# render_ref.py  (WP-B)
def source_map(plan: Plan, k: int, out_scale=1.0, iters=3) -> np.ndarray   # (Ho,Wo,2) source px (float64)
def render_frames(plan, video_path, frames, out_scale=0.25, gray=True) -> Iterator[(k, np.ndarray)]
#   preview renderer (torch MPS grid_sample or the shared Metal kernel via torch.mps.compile_shader).

# smooth.py  (WP-C)
@dataclass class SmoothParams: smoothness: float = 1.0; min_out_fx: float; allow_zoom: bool = True; horizon_lock: bool = False; ...
def optimize_path(tel: Telemetry, q_cam_fn, frame_ids: np.ndarray, out_w, out_h, params) -> tuple[virt_q (F,4), out_fx (F,)]
#   Crop-constrained (every output corner/edge sample maps inside the source image for ALL row times),
#   SO(3) tangent-space QP (Clarabel) per research/sota_algorithms.md §6, no zoom breathing.

# residual.py  (WP-D)
def measure_residuals(frames: Iterator[(k, gray np.ndarray)], K: np.ndarray, expected_rel_q: np.ndarray | None) -> dict
#   On RECTILINEAR stabilized previews: per consecutive pair (k,k+1) the measured inter-frame rotation
#   (rotvec, camera frame, rad), the error vs expected_rel_q (what the virtual path intended),
#   a confidence, and inlier fraction. DIS flow -> robust static mask -> ECC on a 3-param rotation
#   homography H = K R K^-1 (optionally +RS shear terms). Target precision <= 0.05 px @960.
# closedloop.py  (WP-D)
def fold_residuals(frame_t, rel_err_rotvec (F-1,3), conf, fs, hp_hz=1.0, clamp_deg=0.5) -> Callable[[t], q]
#   Integrate per-pair errors into a residual orientation path, high-pass (keep > hp_hz; low-freq is
#   parallax/intent and is absorbed by smoothing), clamp, interpolate smoothly in time -> correction(t)
#   to right-multiply onto q_cam. Must never increase jitter: gate by confidence; return diagnostics.

# calib.py  (WP-E)
def self_calibrate(tel: Telemetry, video_path: str, max_windows=6, width=960) -> TimeModel
#   KLT tracks on ORIGINAL (fisheye) analysis frames; row-timed rotation-only transfer residual;
#   robust least squares over offset (±5 ms), readout, focal_scale, small extrinsic; report
#   uncertainties in TimeModel.notes. O3 metadata timing is already good (±0.7 ms): result must be
#   bounded and fall back to defaults when unobservable.

# pipeline.py / cli.py  (WP-F)
def analyze(video, out_dir, params) -> dict       # telemetry, calib, smooth, closed loop x N, writes plan.spplan + report.json
def render(video, plan_path, out_path, start_frame=0, n_frames=None, codec='hevc10') -> None  # calls app/renderer sprender
```

## 3. Files

`.spplan` v1 — see `engine/stillpoint/plan_io.py` docstring (256-byte LE header + per-frame records:
f64 pts, f32 out_fx/out_cx/out_cy/pad, n_rows × 3×3 f32 row-major matrices). Row j sits at source
row `y_j = j*(src_h-1)/(n_rows-1)`; the renderer linearly interpolates matrices between rows.

Renderer CLI (WP-B): `app/renderer/.build/sprender <in.mp4> <plan.spplan> <out.mov|mp4>
[--start-frame N] [--frames M] [--codec hevc10|hevc10-speed|prores] [--bitrate-mbps 180]
[--kernel lanczos3|catmullrom]` — AVFoundation decode (10-bit aware, 8-bit for H.264), Metal kernel
from `shaders/warp.metal` compiled at runtime, AVAssetWriter with source timescale (fixes 600-timescale
jitter), audio passthrough, colour tags copied. Output frame k uses the plan record whose pts matches
the decoded frame's PTS (±0.5 frame).

## 4. Algorithms (M1 defaults — details in research/sota_algorithms.md and research/sota_ai.md)

1. Telemetry: uniform-grid IMU timing; O3 `offset` already contains -exposure/2 (don't subtract again);
   O3 sensor clock 59.9693 fps vs 59.94 container — map per frame (never n/fps).
2. Calibration (WP-E): bounded refinement; default TimeModel() if not confident.
3. Orientation: slerp on the uniform grid (1–2 kHz). q_cam(t) per §1.
4. Path: crop-constrained QP; smoothness preset ~ "cinematic FPV": strongly reject > ~1.5 Hz, follow
   intentional moves; horizon lock off by default (Jimmy didn't use it).
5. Closed loop (the micro-jitter killer): render 960-wide rectilinear previews from the current plan ->
   measure residual inter-frame rotation vs intended -> fold_residuals -> re-optimize path ->
   rebuild plan. 2–3 iterations; stop when HF residual < floor. This also corrects the measured
   gyro-vs-image HF roll over-report (image shows only 29–59% of gyro HF roll on some O3 clips).
6. Render full-res with Metal (Lanczos-3 or Catmull-Rom; fp32 coordinates).

## 5. Evaluation (M1 gate)

Clips/windows (same as `work/baseline/baseline_table.md`): DJI_0025 15–40 s, DJI_0028 8–33 s,
DJI_0034 15–40 s, DJI_0027 5–30 s, DJI_0032 22–47 s (originals in `~/Desktop/untitled folder 4/`,
Gyroflow renders in `~/Desktop/untitled folder 5/` — READ-ONLY). Analyze whole clips, render the
windows, run `eval.jitter_metrics` + `eval/compare.py` against the Gyroflow renders at matched crop.

## 6. Rules for agents
- Never modify/move/delete Jimmy's footage or write next to it; write only inside stillpoint/.
- Do not run the Gyroflow CLI (App Store sandbox: it hangs and it overwrote Gyroflow's log once).
- Nothing online is published; no git pushes.
- Tests: `cd stillpoint && PYTHONPATH=engine .venv/bin/python -m pytest tests/test_<yours>.py -q`.
- Keep heavy decoding bounded; other agents run concurrently.
