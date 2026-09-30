# Stillpoint v6 decision gate (engine v5 options, ProRes)

Generated 2026-09-30T07:19:00 by `eval/decision.py` from `~/Desktop/Claude Assistant/stillpoint/work/gate/v6`.  Engine 03300f2 content 4c025de1e723; renders ProRes 422 HQ (bit-identical re-renders); judge eval 2.0 / edeb667e8e42d39d.

## Recommendation

- **O3**: Full-frame fill ON by default (CAMERA_OPTION_DEFAULTS['O3'] = {fill: True, fill_overscan: 0.06}). Against the default it passes the rule: HF -14.5 % [-24, -4], 2-8 Hz -15.7 %, roll -7.9 %, jumps >1 px 13 -> 8, win-rate vs Gyroflow +2.3 pp, nothing worse, and it costs ~0-3 % analysis time. Mesh residual stays the optional 'Max quality' toggle, not the default: against the default it also passes (HF -24.3 %, calm -19.6 %, 2-8 Hz -30.4 %, roll -27.3 %, win-rate +8.4 pp), and head-to-head against fill it is really smoother in calm / 2-8 Hz / roll (-18 / -18 / -21 %) with win-rate +6.1 pp. But its HF edge over fill (-11.4 % [-23, +3]) is inside noise, jumps >1 px are really worse (8 -> 18 across the 10 windows), and it costs +30 % (clean, DJI_0034) to ~+100 % analysis time. So it does not win decisively.
- **OA4**: Both OFF by default (no change). Fill does not pass: HF -7.0 % [-15, -0] is inside noise, and nothing gets worse. Mesh does not pass either: HF -4.8 % is not real, calm-cruise -35 % is real, roll +7.8 % [-0, +17] is borderline, and analysis takes 6.6x the default (962 s vs 145 s, clean). Mesh stays available as 'Max quality'.
- **O4 Pro**: Both OFF by default (no change). Fill does not pass: HF +0.4 %, calm +11.7 % [+3, +21], jumps 6 -> 10. Mesh does not pass: HF -9.5 % [-19, +1] is inside noise, roll -22 % is real, and analysis takes about 7x the default (673 s vs 95 s, with contention). Mesh stays available as 'Max quality'. Horizon lock stays off (O4 Pro gravity is 6-10 deg off in turns).

**User toggles:** Max quality (mesh residual) on every camera, mutually exclusive with fill. Selecting it on O3 replaces the fill default. Horizon lock stays a user toggle, off by default. It was not re-tested this round.

- Mesh analysis cost grows with frame count, not as a fixed factor: the mesh stage adds about 26-34 ms per frame (clean runs: DJI_0034 79 s / 3029 frames, OA4_0012 814 s / 23950 frames), which is roughly 1.6-2x the clip's duration at 60 fps. On O3, whose default analysis is heavy (closed loop), that is about x1.3-1.5. On OA4 and O4 Pro, whose default analysis is light, it is about x5-7. app_bridge OPTION_TIME_FACTORS['mesh'] = 1.4 underestimates the OA4 / O4 Pro pre-flight time.
- Mesh timings marked 'busy 2' (DJI_0025, DJI_0028, DJI_0032, O4_0004) ran beside other gate6b mesh analyses (3 heavy slots) and are upper bounds. The clean ones are OA4_0012 (962 s vs 145 s default), DJI_0034 (351 s vs 271 s) and DJI_0027 (1358 s vs 932 s, +46 %; busy mean 0.15). The --reanalyze plans are byte-identical to the cached ones (DJI_0034, DJI_0027).
- Fill artifacts are negligible: fill fraction is at most 0.16 % of the frame on average (p95 at most 1.3 %), and the seam ratio is about 1.0.
- Horizon lock was not judged this round (decision made before the run). gate6 driver.sh queues hl10/hl06 after mesh anyway. This table was built with --configs fill,mesh.

## Determinism and judge noise

Re-render of the default plans (default_rep): DJI_0034:15-40 md5 same, metrics identical; O4_0004:60-72 md5 same, metrics identical; OA4_0012:176-188 md5 same, metrics identical.

Judge sensitivity to a constant sub-pixel output shift (no motion change; perturb [0.05, 0.05] px on 19 windows; 19 window pairs).  Per-window sigma = RMS of log(shifted/default), pooled (O3 / OA4 / O4) [largest]:

| metric | pooled | O3 | OA4 | O4 | largest | mean shift of the first control [95 % CI] |
|---|---|---|---|---|---|---|
| HF >2 Hz | 15.3% | 10.2% | 23.7% | 12.2% | 42% | -3.5% [-10, +3] |
| calm-cruise HF | 6.7% | 6.8% | 8.8% | 0.3% | 15% | -3.1% [-6, -0] |
| 2-8 Hz | 18.5% | 15.4% | 24.7% | 16.3% | 46% | -4.9% [-13, +3] |
| 8-30 Hz | 3.3% | 1.0% | 4.6% | 4.8% | 9% | -0.7% [-2, +1] |
| roll | 10.2% | 9.7% | 10.4% | 11.2% | 19% | -4.8% [-9, -1] |
| jello | 13.6% | 14.0% | 15.8% | 8.8% | 25% | +2.4% [-4, +9] |
| corner wobble | 26.4% | 12.6% | 45.2% | 18.6% | 68% | -5.1% [-16, +6] |
| jumps >1 px / window (absolute) | 0.69 | 0.55 | 0.63 | 1.00 | 2.00 | |
| jumps >0.5 px / window (absolute) | 1.76 | 1.41 | 2.41 | 1.58 | 4.00 | |
| crop footprint (source area) (absolute) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | |
| 1-s win-rate vs reference (absolute) | 0.036 | 0.039 | 0.045 | 0.000 | 0.100 | |

Rule calibration (the shift controls judged like a config vs the default; must flag nothing): perturb o3: nothing real; perturb oa4: nothing real; perturb o4: nothing real.

Each camera uses max(its own sigma, pooled sigma).  A mean change is **bold** (real) only when its 95 % bootstrap CI over windows excludes 0 AND it is larger than 2.5 sigma/sqrt(n); windows better/worse count per-window changes beyond 2.5 sigma.

## DJI O3 (5 clips, 10 windows; reference = Jimmy's Gyroflow renders)

Default (v5): HF 0.709 [0.421, 1.097], calm 0.191, 2-8 0.695, 8-30 0.150, roll 0.0253 deg, jello 0.782, corner 0.675, jumps >1 px 13 (>0.5: 41), crop 0.6162, pooled 1-s win-rate vs Gyroflow 90%, 1/10 windows pass every gate check; reference HF 1.559.

| config vs default | HF >2 Hz | calm-cruise HF | 2-8 Hz | 8-30 Hz | roll | jello | corner wobble | jumps >1 (>0.5), sum | crop | win-rate | gate passes | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full-frame fill (fill=True, fill_overscan=0.06) | **-14.5% [-24, -4]** (2+/0-) | -2.2% [-8, +2] (1+/0-) | **-15.7% [-27, -4]** (2+/0-) | -1.3% [-2, -0] (0+/0-) | **-7.9% [-14, -2]** (0+/0-) | -0.2% [-5, +4] (0+/0-) | +3.2% [-3, +9] (0+/0-) | 13 -> 8 (41 -> 39) | -0.06 pp | +2.3 pp [+0, +5] | 1 -> 1 of 10 | candidate default |
| mesh residual (mesh_residual=True) | **-24.3% [-32, -16]** (3+/0-) | **-19.6% [-23, -16]** (7+/0-) | **-30.4% [-38, -22]** (3+/0-) | -0.5% [-1, +0] (0+/0-) | **-27.3% [-35, -18]** (5+/0-) | -4.2% [-10, +2] (0+/0-) | +4.8% [+0, +10] (0+/0-) | 13 -> 18 (41 -> 46) | +0.00 pp | **+8.4 pp [+1, +17]** | 1 -> 1 of 10 | candidate default |
| *mesh vs fill, head-to-head (baseline = fill)* | -11.4% [-23, +3] (2+/0-) | **-17.8% [-23, -12]** (7+/0-) | **-17.5% [-31, -1]** (2+/0-) | +0.7% [-0, +2] (0+/0-) | **-21.1% [-31, -9]** (6+/0-) | -4.0% [-8, -0] (0+/0-) | +1.6% [-6, +10] (0+/0-) | **8 -> 18** (39 -> 46) | +0.06 pp | **+6.1 pp [+0, +14]** | 1 -> 1 of 10 | no |

Cells: geo-mean change vs default [95 % CI] (windows better+/worse- beyond 2.5 sigma); the head-to-head row is mesh vs fill with the same rule (fill and mesh are mutually exclusive).

<details><summary>per-window HF (config vs default)</summary>

| window | full-frame fill | mesh residual |
|---|---|---|
| DJI_0025:1-15 | 0.395 vs 0.516 | 0.305 vs 0.516 |
| DJI_0025:15-40 | 0.303 vs 0.308 | 0.250 vs 0.308 |
| DJI_0027:100-120 | 0.791 vs 0.743 | 0.530 vs 0.743 |
| DJI_0027:5-30 | 0.625 vs 0.983 | 0.819 vs 0.983 |
| DJI_0028:38-58 | 1.902 vs 2.241 | 2.223 vs 2.241 |
| DJI_0028:8-33 | 0.830 vs 0.822 | 0.542 vs 0.822 |
| DJI_0032:120-140 | 0.194 vs 0.270 | 0.228 vs 0.270 |
| DJI_0032:22-47 | 0.510 vs 0.496 | 0.396 vs 0.496 |
| DJI_0034:1-15 | 0.138 vs 0.137 | 0.118 vs 0.137 |
| DJI_0034:15-40 | 0.365 vs 0.571 | 0.327 vs 0.571 |

</details>

## Osmo Action 4 (clip 0012, 5 windows; reference = the original)

Default (v5): HF 0.827 [0.455, 1.170], calm 0.099, 2-8 0.852, 8-30 0.079, roll 0.0255 deg, jello 1.346, corner 0.929, jumps >1 px 3 (>0.5: 15), crop 0.5941, pooled 1-s win-rate vs original 98%, 3/5 windows pass every gate check; reference HF 3.109.

| config vs default | HF >2 Hz | calm-cruise HF | 2-8 Hz | 8-30 Hz | roll | jello | corner wobble | jumps >1 (>0.5), sum | crop | win-rate | gate passes | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full-frame fill (fill=True, fill_overscan=0.06) | -7.0% [-15, -0] (0+/0-) | -13.8% [-16, -12] (0+/0-) | -6.8% [-16, +0] (0+/0-) | -2.0% [-10, +10] (1+/1-) | -1.9% [-9, +6] (0+/0-) | +2.9% [-9, +17] (0+/0-) | -5.6% [-14, +4] (0+/0-) | 3 -> 2 (15 -> 9) | +0.08 pp | +0.0 pp [-6, +6] | 3 -> 3 of 5 | no |
| mesh residual (mesh_residual=True) | -4.8% [-11, +2] (0+/0-) | **-35.1% [-49, -17]** (1+/0-) | -4.5% [-12, +4] (0+/0-) | +2.4% [-10, +17] (1+/1-) | +7.8% [-0, +17] (0+/0-) | -3.3% [-10, +4] (0+/0-) | +12.0% [-7, +46] (0+/0-) | 3 -> 1 (15 -> 14) | +0.00 pp | +2.0 pp [+0, +6] | 3 -> 4 of 5 | no |
| *mesh vs fill, head-to-head (baseline = fill)* | +2.4% [-11, +18] (0+/0-) | **-24.8% [-40, -6]** (1+/0-) | +2.5% [-12, +19] (0+/0-) | +4.4% [-3, +12] (0+/1-) | +9.9% [-2, +24] (0+/1-) | -6.1% [-19, +5] (0+/0-) | +18.7% [-6, +59] (0+/0-) | 2 -> 1 (9 -> 14) | -0.08 pp | +2.0 pp [+0, +6] | 3 -> 4 of 5 | no |

Cells: geo-mean change vs default [95 % CI] (windows better+/worse- beyond 2.5 sigma); the head-to-head row is mesh vs fill with the same rule (fill and mesh are mutually exclusive).

<details><summary>per-window HF (config vs default)</summary>

| window | full-frame fill | mesh residual |
|---|---|---|
| OA4_0012:146-158 | 0.167 vs 0.180 | 0.197 vs 0.180 |
| OA4_0012:176-188 | 1.130 vs 1.117 | 0.970 vs 1.117 |
| OA4_0012:196-208 | 1.396 vs 1.380 | 1.199 vs 1.380 |
| OA4_0012:300-312 | 0.582 vs 0.615 | 0.596 vs 0.615 |
| OA4_0012:330-342 | 0.658 vs 0.844 | 0.827 vs 0.844 |

</details>

## O4 Pro (clip 0004, 4 windows; reference = the original)

Default (v5): HF 0.545 [0.179, 1.028], calm 0.207, 2-8 0.521, 8-30 0.139, roll 0.0162 deg, jello 0.948, corner 2.977, jumps >1 px 6 (>0.5: 10), crop 0.6139, pooled 1-s win-rate vs original 90%, 1/4 windows pass every gate check; reference HF 1.139.

| config vs default | HF >2 Hz | calm-cruise HF | 2-8 Hz | 8-30 Hz | roll | jello | corner wobble | jumps >1 (>0.5), sum | crop | win-rate | gate passes | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full-frame fill (fill=True, fill_overscan=0.06) | +0.4% [-10, +18] (0+/0-) | +11.7% [+3, +21] (0+/1-) | +1.2% [-11, +21] (0+/0-) | -1.4% [-4, +1] (0+/0-) | -1.1% [-5, +2] (0+/0-) | +3.4% [-8, +15] (0+/0-) | +5.6% [-9, +23] (0+/0-) | 6 -> 10 (10 -> 15) | +0.00 pp | +5.0 pp [+0, +15] | 1 -> 1 of 4 | no |
| mesh residual (mesh_residual=True) | -9.5% [-19, +1] (0+/0-) | -4.4% [-7, -2] (0+/0-) | -14.2% [-28, +2] (0+/0-) | -2.1% [-5, -0] (0+/0-) | **-22.4% [-26, -18]** (2+/0-) | +4.1% [-7, +16] (0+/0-) | +9.2% [-14, +39] (0+/0-) | 6 -> 6 (10 -> 14) | +0.00 pp | +0.0 pp [+0, +0] | 1 -> 1 of 4 | no |
| *mesh vs fill, head-to-head (baseline = fill)* | -9.9% [-21, +8] (0+/0-) | **-14.4% [-19, -10]** (1+/0-) | -15.2% [-26, +6] (0+/0-) | -0.8% [-2, +0] (0+/0-) | **-21.6% [-24, -20]** (1+/0-) | +0.7% [-9, +8] (0+/0-) | +3.5% [-11, +13] (0+/0-) | 10 -> 6 (15 -> 14) | -0.00 pp | -5.0 pp [-15, +0] | 1 -> 1 of 4 | no |

Cells: geo-mean change vs default [95 % CI] (windows better+/worse- beyond 2.5 sigma); the head-to-head row is mesh vs fill with the same rule (fill and mesh are mutually exclusive).

<details><summary>per-window HF (config vs default)</summary>

| window | full-frame fill | mesh residual |
|---|---|---|
| O4_0004:134-146 | 0.124 vs 0.125 | 0.104 vs 0.125 |
| O4_0004:177-189 | 0.678 vs 0.528 | 0.520 vs 0.528 |
| O4_0004:45-57 | 0.212 vs 0.233 | 0.184 vs 0.233 |
| O4_0004:60-72 | 1.138 vs 1.293 | 1.344 vs 1.293 |

</details>

## Analysis cost (one heavy job at a time)

| clip | default | fill | mesh |
|---|---|---|---|
| DJI_0025 | 281 s, 2.36 GB (busy 1) | 285 s (+2%), 2.37 GB | 648 s (+131%), 2.57 GB (busy 2) |
| DJI_0027 | 932 s, 3.10 GB | 957 s (+3%), 3.19 GB | 1358 s (+46%), 3.30 GB (busy 1) |
| DJI_0028 | 341 s, 2.44 GB (busy 2) | 331 s (-3%), 2.35 GB | 602 s (+77%), 2.83 GB (busy 2) |
| DJI_0032 | 972 s, 3.10 GB | 997 s (+3%), 3.14 GB | 1901 s (+96%), 3.00 GB (busy 2) |
| DJI_0034 | 271 s, 2.31 GB (busy 1) | 270 s (-0%), 2.29 GB | 351 s (+30%), 2.63 GB |
| O4_0004 | 95 s, 2.03 GB (busy 1) | 105 s (+11%), 2.16 GB | 673 s (+606%), 2.33 GB (busy 2) |
| OA4_0012 | 145 s, 3.18 GB | 185 s (+28%), 3.51 GB | 962 s (+562%), 3.52 GB |

wall time of the whole-clip analysis (3 measurement workers) and peak RSS of its process tree (engine mem watch, 1 s); "busy N" = other heavy jobs ran meanwhile (timing not clean).

## Fill artifacts (fill config, first 12 s of each window)

| window | fill frac mean / p95 | frames with fill | uncovered (plan) | seam ratio pooled / p95 | flicker ratio pooled / p95 |
|---|---|---|---|---|---|
| OA4_0012:146-158 | 0.0000 / 0.0000 | 0.01 | 0.0000 | 0.99 / 1.26 | 73.97 / 79.89 |
| OA4_0012:176-188 | 0.0001 / 0.0000 | 0.02 | 0.0000 | 1.04 / 1.19 | 0.59 / 1.15 |
| OA4_0012:196-208 | 0.0005 / 0.0000 | 0.05 | 0.0000 | 1.09 / 1.37 | 1.00 / 2.13 |
| OA4_0012:300-312 | 0.0000 / 0.0000 | 0.00 | 0.0000 | - / - | - / - |
| OA4_0012:330-342 | 0.0003 / 0.0000 | 0.04 | 0.0000 | 1.02 / 1.17 | 0.97 / 1.33 |
| DJI_0034:15-40 | 0.0014 / 0.0088 | 0.12 | 0.0000 | 1.06 / 1.21 | 1.80 / 7.46 |
| DJI_0034:1-15 | 0.0000 / 0.0000 | 0.00 | 0.0000 | - / - | - / - |
| DJI_0025:15-40 | 0.0003 / 0.0000 | 0.02 | 0.0000 | 0.98 / 1.12 | 1.84 / 3.34 |
| DJI_0025:1-15 | 0.0003 / 0.0000 | 0.02 | 0.0000 | 0.82 / 1.05 | 5.57 / 18.23 |
| DJI_0028:8-33 | 0.0011 / 0.0098 | 0.09 | 0.0000 | 1.01 / 1.16 | 3.06 / 6.87 |
| DJI_0028:38-58 | 0.0007 / 0.0000 | 0.04 | 0.0000 | 1.02 / 1.12 | 0.92 / 1.52 |
| O4_0004:45-57 | 0.0000 / 0.0000 | 0.00 | 0.0000 | - / - | - / - |
| O4_0004:60-72 | 0.0000 / 0.0000 | 0.00 | 0.0000 | - / - | - / - |
| O4_0004:177-189 | 0.0000 / 0.0000 | 0.00 | 0.0000 | - / - | - / - |
| O4_0004:134-146 | 0.0000 / 0.0000 | 0.00 | 0.0000 | - / - | - / - |
| DJI_0027:5-30 | 0.0012 / 0.0100 | 0.09 | 0.0000 | 1.03 / 1.32 | 1.30 / 3.58 |
| DJI_0027:100-120 | 0.0000 / 0.0000 | 0.03 | 0.0000 | 1.06 / 1.27 | 0.80 / 1.13 |
| DJI_0032:22-47 | 0.0016 / 0.0131 | 0.11 | 0.0000 | 1.04 / 1.35 | 1.33 / 2.50 |
| DJI_0032:120-140 | 0.0000 / 0.0000 | 0.00 | 0.0000 | - / - | - / - |

