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

`.spblur` v1 (optional synthetic-shutter sidecar, `engine/stillpoint/synth_blur.py`; `sprender --blur`,
`AnalyzeParams.synth_blur`, `render(..., blur=)`): per frame n taps D_i = R(V_k)ᵀR(V(t_k+s_i)) + weights; the
`sp_warp_*_blur` kernels average the same source frame warped along the virtual path (M(y)·D_i). Blur judder (the
baked blur vs the output's own motion) is measured by `eval/judder.py` (plan-based exact, or vision-based for
renders without a plan -- rotation-only on both sides: the eval's similarity SCALE rate, i.e. forward-flight
parallax, is not counted). Both remedies are OFF by default (`synth_blur='off'`, `blur_smooth_w=0`): on the
available footage (exposures <= 4.4 ms, baked streaks p95 < 3 px 1080p-eq) they cost more than they fix.

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

### 4.1 Full-frame border fill (optional; `engine/stillpoint/fill.py`, 2026-09-29)

Output pixels outside the current source frame are synthesised from up to 4 neighbouring source frames
(record offsets up to +-20). Output ray of record k -> neighbour j: `r_c = M_j(y) . G_kj . r_v`, `G_kj = Rv_j^T Rv_k`
(exact rotation; closed-loop corrections are already in M_j) -- the kernels run the unchanged `sp_source_coord()`
on the composite rows. Parallax: a per-record grid of image-plane parallax velocity (DIS flow between the neighbours
k-2 and k+2 reprojected into k, robust per cell, temporally smoothed); source at offset d is sampled at x + d.v(x).
Per-source NCC check + luma gain in the band next to the fill; per-pixel consistency re-weighting; feathered seams;
soft edge extension where nothing covers. Path optimizer: `AnalyzeParams.fill_overscan` (cap, fraction of the
short side) with `fill_overscan_mode='coverage'`: per frame and edge, the crop box grows only as far as
neighbouring frames actually saw (`fill.coverage_overscan`, camera-frame geometry, independent of the path).
With fill + overscan the crop search's footprint target is the DISPLAYED field of view (`fill.output_footprint`,
the unclipped output footprint), not the clipped real-pixel area, so `target_footprint` means the same FOV with and
without fill. Cost: the selection pass is vectorised over 32 records; the align pass reuses the analysis' own
decoder (downscaled to 480 px), measures the mesh on every 3rd record and the per-source check on every 2nd record
(+ whenever a new source appears; the rest reuse the same source's result at the nearest checked record).
The table lives in the `.spplan` FILL section (flags bit 0, see plan_io.py); renderers without fill ignore it.
`sprender` keeps a ring of +-max_offset decoded frames; `--no-fill` renders the plain kernels.
Parallax-aware weight (`FillParams.parallax_tol`, x out_w, plan FILL header f32 @60, kernel `sp_par_w`): each
source's weight is multiplied by exp(-(|offset| |v(X)| / tol)^2) -- where the predicted parallax shift is large (near
ground in fast low flight: the mesh reaches 40-60 px/frame at 4K on OA4 0012) a far source is likely misaligned, so
nearer sources or the soft edge extension take over. Older plans carry 0 there (= off). Default OFF: at 0.012 the
OA4 0012 judge scored 2 of 3 windows worse (+16-20 % HF, more jumps) than the same plan without it.
Evaluation: `eval/fill_artifacts.py` (fill fraction, seam visibility, temporal flicker of the filled pixels).

### 4.2 Engine v5 options (merge of workstreams A-E, 2026-09-29; `pipeline.AnalyzeParams`, `cli analyze`, `app_bridge analyze`)

| option | default | what | where |
|---|---|---|---|
| `timecal` | **on** | per-clip timing self-calibration (offset / readout / exposure box); applied only when confident, confirmed on held-out windows and, for the offset, by the gyro's HF part (none of the 2026-09 test clips) | `timecal.py` |
| `horizon_lock`, `roll_limit_deg` | 0 (off) | horizon lock v2: second SQP stage toward a crop-feasible leveled target, fades out steep / inverted / mid-flip, never zooms in beyond the unlocked path | `smooth.py` |
| `fill`, `fill_overscan` | off, 0 | full-frame border fill (4.1); plan FILL section = flags bit 0 | `fill.py` |
| `mesh_residual` | off | parallax mesh residual: 2-6 Hz affine flow field from persistent tracks, baked as per-frame vertex offsets (plan MESH block = flags bit 1, explicit offset in the header); two extra tracking passes | `mesh.py` |
| `synth_blur`, `blur_smooth_w` | 'off', 0 | synthetic shutter sidecar `plan.spblur`; blur-aware smoothing term | `synth_blur.py`, `smooth.py` |

`fill`, `mesh_residual` and `synth_blur` have separate sprender kernels and are mutually exclusive for now
(`pipeline.check_params` raises; sprender refuses a plan / call that asks for two). Everything else combines.
With these defaults the v5 plans of the 7 scoreboard clips equal engine v4's (5 byte-identical; OA4 0012 <= 0.0003 px,
DJI_0025 <= 0.03 px): timecal fitted but applied nothing on any of them.

**Per-camera defaults (gate v6, 2026-09-30).** `AnalyzeParams()` keeps the camera-independent defaults above (scripts
and the gate scoreboard construct it directly). The front ends -- `app_bridge analyze` / `probe` (the Mac app) and
`cli analyze` -- resolve every option that is not given through ONE helper, `app_bridge.resolve_for_video` =
AnalyzeParams' defaults + `app_bridge.CAMERA_OPTION_DEFAULTS` (substring of `Telemetry.camera`). Entries are added
only for a real win with no meaningful regression on the ProRes decision gate (`work/gate/v6/decision.md`):

| camera | default | gate evidence (vs the v5 default, 95 % CI over windows) |
|---|---|---|
| DJI O3 | `fill=True, fill_overscan=0.06` | HF -14.5 % [-24, -4], 2-8 Hz -15.7 %, roll -7.9 %, jumps >1 px 13 -> 8, win-rate vs Gyroflow +2.3 pp, nothing worse, +0-3 % analysis time |
| Osmo Action 4 | none | fill: HF -7 % inside noise; mesh: HF not real, roll +7.8 % borderline, 6.6x analysis time |
| O4 Pro | none | fill: calm +11.7 % [+3, +21], jumps 6 -> 10; mesh: HF -9.5 % inside noise, ~7x analysis time |

`mesh_residual` ('Max quality') stays a toggle on every camera: on the O3 it passes against the v5 default (HF -24 %)
but head-to-head against fill its HF edge (-11 % [-23, +3]) is inside noise, jumps >1 px are worse (8 -> 18) and it
costs +30-46 % analysis time. Its cost is ~26-34 ms per frame, so the pre-flight factor is per camera
(`app_bridge.CAMERA_TIME_FACTORS`: O3 x1.4, OA4 x6.6, O4 Pro x6.5). Horizon lock stays off (not re-tested in v6; O4 Pro
gravity is 6-10 deg off in turns). The composed per-camera-default scoreboard is `work/gate/v6/scoreboard.md`
(`scripts/compose_scoreboard.py`).

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
