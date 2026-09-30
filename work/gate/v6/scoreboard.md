# Stillpoint scoreboard: v6 (per-camera defaults)

Composed 2026-09-30T07:24:50 by `scripts/compose_scoreboard.py` from finished gate runs (no re-render): each camera's windows come from the run whose config equals that camera's default in `engine/stillpoint/app_bridge.py` CAMERA_OPTION_DEFAULTS. Engine content 4c025de1e723 (@ 03300f2); renders prores (ProRes 422 HQ, bit-identical re-renders); judge eval 2.0. Judge noise and the decision rule: `decision.md` / `noise_prores.json`.

| camera | default options (beyond timecal) | windows from |
|---|---|---|
| DJI O3 | fill=True, fill_overscan=0.06 | `fill/` (fill) |
| Osmo Action 4 | none (engine defaults) | `default/` (default) |
| O4 Pro | none (engine defaults) | `default/` (default) |

Toggles (not defaults): Max quality (mesh residual) on every camera, exclusive with fill; horizon lock (off; not reliable on O4 Pro). See `decision.md` for the numbers behind each choice.

## DJI O3 (vs Gyroflow)

| window | HF (GF) | calm-cruise (GF) | 2-8 Hz (GF) | 8-30 Hz (orig) | roll deg (GF) | 8-30 roll deg (GF) | jello (GF) | row wobble (GF) | corner wobble p-med (GF) | SP-only jumps >1 (>0.5), max | crop plan-exact / fitted (GF) | win vs GF | failed checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DJI_0034:15-40 (gate) | **0.365** (1.308) | **0.133** (0.551) | 0.371 (1.261) | 0.093 (0.559) | 0.0171 (0.0451) | 0.0057 (0.0186) | 0.380 (0.493) | 0.353 (0.422) | 0.545 (0.594) | 1 (5), 1.20 | 0.6251 / 0.6251 (0.5759) | 100% | jumps_gt_1px |
| DJI_0034:1-15 (heldout) | **0.138** (0.824) | **0.139** (0.766) | 0.100 (0.757) | 0.091 (0.201) | 0.0074 (0.0707) | 0.0055 (0.0287) | 0.294 (0.293) | 0.281 (0.279) | 0.601 (0.644) | 0 (0), 0.00 | 0.6236 / 0.6235 (0.6441) | 100% | crop_footprint |
| DJI_0025:15-40 (gate) | **0.303** (0.969) | **0.314** (0.597) | 0.283 (0.962) | 0.121 (0.232) | 0.0154 (0.0513) | 0.0079 (0.0156) | 0.744 (0.608) | 0.724 (0.583) | 0.691 (0.683) | 0 (3), 0.97 | 0.6293 / 0.6293 (0.6333) | 100% | calm_cruise, jello |
| DJI_0025:1-15 (heldout) | **0.395** (0.990) | **0.144** (0.573) | 0.398 (0.965) | 0.091 (0.581) | 0.0107 (0.0412) | 0.0053 (0.0177) | 1.018 (0.842) | 0.972 (0.713) | 0.507 (0.598) | 0 (4), 0.97 | 0.6293 / 0.6292 (0.5955) | 100% | jello |
| DJI_0028:8-33 (gate) | **0.830** (1.144) | **0.139** (0.925) | 0.784 (1.079) | 0.295 (0.478) | 0.0504 (0.0703) | 0.0214 (0.0282) | 1.227 (1.284) | 1.168 (1.115) | 0.919 (0.912) | 4 (4), 1.74 | 0.6331 / 0.6327 (0.6122) | 91% | jumps_gt_1px |
| DJI_0028:38-58 (heldout) | **1.902** (4.607) | **0.405** (0.761) | 1.893 (4.473) | 0.201 (0.467) | 0.0314 (0.1251) | 0.0142 (0.0237) | 0.885 (1.189) | 0.810 (0.894) | 1.164 (1.204) | 3 (13), 1.67 | 0.6333 / 0.6333 (0.6117) | 94% | calm_cruise, jumps_gt_1px |
| DJI_0027:5-30 (gate) | **0.625** (3.053) | **0.179** (0.481) | 0.615 (2.942) | 0.122 (0.470) | 0.0199 (0.1057) | 0.0074 (0.0080) | 0.798 (1.062) | 0.730 (0.947) | 0.877 (0.791) | 0 (4), 0.87 | 0.6037 / 0.6037 (0.5757) | 83% | winrate_vs_gyroflow, corner_wobble |
| DJI_0027:100-120 (heldout) | **0.791** (0.839) | **0.130** (0.478) | 0.768 (0.805) | 0.259 (0.330) | 0.0517 (0.0569) | 0.0208 (0.0227) | 1.686 (1.624) | 1.644 (1.582) | 0.760 (0.789) | 0 (4), 0.89 | 0.6031 / 0.6031 (0.6005) | 61% | winrate_vs_gyroflow |
| DJI_0032:22-47 (gate) | **0.510** (1.214) | **0.122** (0.337) | 0.523 (1.216) | 0.116 (0.634) | 0.0226 (0.0536) | 0.0048 (0.0069) | 0.508 (0.672) | 0.429 (0.613) | 0.589 (0.607) | 0 (1), 0.56 | 0.5889 / 0.5889 (0.5432) | 96% | **PASS** |
| DJI_0032:120-140 (heldout) | **0.194** (0.644) | **0.146** (0.632) | 0.166 (0.561) | 0.098 (0.636) | 0.0074 (0.0493) | 0.0048 (0.0260) | 0.247 (0.278) | 0.217 (0.249) | 0.451 (0.447) | 0 (1), 0.59 | 0.5875 / 0.5875 (0.6119) | 100% | crop_footprint |

Per-camera means [95 % bootstrap CI over 10 windows]: HF 0.605 [0.354, 0.938], calm 0.185 [0.137, 0.246], 2-8 0.590 [0.338, 0.923], 8-30 0.149 [0.107, 0.197], roll deg 0.0234 [0.0145, 0.0334], jello 0.779 [0.522, 1.056], corner 0.710 [0.590, 0.847], jumps >1 px per window 0.80 [0.00, 1.80] (>0.5: 3.90 [2.10, 6.30]), crop 0.6157 [0.6046, 0.6258], 1-s win-rate vs Gyroflow 92.5 [84.3, 98.5] %.

**1/10 windows pass every check.** Means: HF 0.605 (Gyroflow 1.559), calm 0.185 (0.610), 2-8 0.590 (1.502), 8-30 0.149 (0.270), roll 0.0234 deg, jello 0.779 (0.835), corner 0.710, crop 0.6157; SP-only jumps >1 px 8 (>0.5: 39). Geo-mean ratio SP/Gyroflow: HF 0.37, calm 0.29, 2-8 0.36, 8-30 0.52, roll 0.30, jello 0.92. SP HF lower than Gyroflow in 10/10; pooled 1-s win-rate vs Gyroflow 92% (8/10 windows >= 90%). Check failures: calm_cruise 2, corner_wobble 1, crop_footprint 2, jello 2, jumps_gt_1px 3, winrate_vs_gyroflow 2.

<details><summary>DJI O3 (vs Gyroflow): seconds where Stillpoint loses to Gyroflow (1-s HF RMS, worst first; source time s: SP vs Gyroflow px @ original <1 Hz speed px/s) and jump times</summary>

- DJI_0034:15-40: lost 0 s. Jumps: 23.61s 1.20px; 23.49s 0.78px; 27.95s 0.71px; 28.21s 0.51px; 37.16s 0.50px
- DJI_0034:1-15: lost 0 s
- DJI_0025:15-40: lost 0 s. Jumps: 30.68s 0.97px; 29.56s 0.87px; 29.48s 0.77px
- DJI_0025:1-15: lost 0 s. Jumps: 10.08s 1.14px (also in ref); 3.27s 0.97px; 3.22s 0.82px; 3.45s 0.72px; 3.40s 0.65px
- DJI_0028:8-33: lost 2 s: 26.5s 1.39 vs 1.25 @287; 30.5s 0.77 vs 0.76 @335. Jumps: 24.58s 3.12px (also in ref); 24.40s 2.72px (also in ref); 24.48s 2.69px (also in ref); 28.99s 1.99px (also in ref); 18.28s 1.74px; 32.39s 1.56px (also in ref)
- DJI_0028:38-58: lost 1 s: 39.5s 0.53 vs 0.47 @104. Jumps: 43.11s 2.12px (also in ref); 42.30s 1.67px; 43.19s 1.61px; 42.24s 1.34px (also in ref); 43.01s 1.28px (also in ref); 56.22s 1.08px
- DJI_0027:5-30: lost 4 s: 17.5s 0.72 vs 0.64 @299; 21.5s 0.36 vs 0.30 @248; 20.5s 0.42 vs 0.38 @245; 24.5s 0.44 vs 0.41 @316. Jumps: 17.78s 0.87px; 14.14s 0.56px; 11.01s 0.55px; 17.28s 0.54px
- DJI_0027:100-120: lost 7 s: 104.5s 1.65 vs 1.47 @286; 105.5s 0.85 vs 0.69 @221; 112.5s 0.94 vs 0.92 @299; 113.5s 0.57 vs 0.55 @461; 116.5s 0.53 vs 0.51 @358; 117.5s 0.41 vs 0.41 @334. Jumps: 119.70s 2.31px (also in ref); 119.80s 1.60px (also in ref); 118.79s 1.49px (also in ref); 111.18s 1.45px (also in ref); 115.06s 1.06px (also in ref); 105.67s 0.89px
- DJI_0032:22-47: lost 1 s: 41.5s 1.85 vs 1.80 @1974. Jumps: 42.27s 1.85px (also in ref); 29.17s 1.62px (also in ref); 29.77s 1.55px (also in ref); 28.82s 0.56px
- DJI_0032:120-140: lost 0 s. Jumps: 122.70s 0.59px

</details>

<details><summary>DJI O3 (vs Gyroflow): axis breakdown (tx / ty / roll / scale, px)</summary>

| window | video | HF tx / ty / roll / scale | 2-8 Hz tx / ty / roll / scale | 8-30 Hz tx / ty / roll / scale | calm HF tx / ty / roll / scale |
|---|---|---|---|---|---|
| DJI_0034:15-40 | SP | 0.195 / 0.151 / 0.190 / 0.190 | 0.202 / 0.157 / 0.189 / 0.192 | 0.044 / 0.041 / 0.063 / 0.034 | 0.070 / 0.068 / 0.081 / 0.042 |
|  | GF | 0.883 / 0.656 / 0.501 / 0.501 | 0.885 / 0.592 / 0.437 / 0.516 | 0.083 / 0.100 / 0.206 / 0.046 | 0.148 / 0.201 / 0.489 / 0.051 |
| DJI_0034:1-15 | SP | 0.079 / 0.068 / 0.082 / 0.036 | 0.061 / 0.053 / 0.052 / 0.029 | 0.048 / 0.042 / 0.061 / 0.022 | 0.080 / 0.070 / 0.082 / 0.037 |
|  | GF | 0.123 / 0.212 / 0.785 / 0.051 | 0.103 / 0.196 / 0.722 / 0.045 | 0.065 / 0.089 / 0.318 / 0.026 | 0.122 / 0.211 / 0.724 / 0.052 |
| DJI_0025:15-40 | SP | 0.145 / 0.111 / 0.171 / 0.171 | 0.138 / 0.102 / 0.144 / 0.174 | 0.057 / 0.049 / 0.088 / 0.036 | 0.162 / 0.110 / 0.161 / 0.185 |
|  | GF | 0.498 / 0.564 / 0.569 / 0.218 | 0.501 / 0.567 / 0.550 / 0.222 | 0.086 / 0.112 / 0.173 / 0.039 | 0.235 / 0.262 / 0.440 / 0.199 |
| DJI_0025:1-15 | SP | 0.166 / 0.309 / 0.118 / 0.135 | 0.171 / 0.318 / 0.102 / 0.134 | 0.037 / 0.045 / 0.059 / 0.038 | 0.052 / 0.094 / 0.083 / 0.047 |
|  | GF | 0.339 / 0.790 / 0.457 / 0.178 | 0.339 / 0.784 / 0.412 / 0.179 | 0.065 / 0.154 / 0.196 / 0.042 | 0.118 / 0.367 / 0.420 / 0.058 |
| DJI_0028:8-33 | SP | 0.339 / 0.347 / 0.559 / 0.376 | 0.344 / 0.312 / 0.502 / 0.383 | 0.073 / 0.147 / 0.238 / 0.059 | 0.065 / 0.058 / 0.089 / 0.062 |
|  | GF | 0.488 / 0.534 / 0.780 / 0.421 | 0.477 / 0.498 / 0.717 / 0.419 | 0.090 / 0.183 / 0.313 / 0.087 | 0.243 / 0.371 / 0.809 / 0.074 |
| DJI_0028:38-58 | SP | 1.459 / 0.206 / 0.348 / 1.151 | 1.475 / 0.196 / 0.310 / 1.129 | 0.079 / 0.059 / 0.158 / 0.077 | 0.223 / 0.133 / 0.292 / 0.106 |
|  | GF | 2.677 / 0.488 / 1.389 / 3.448 | 2.525 / 0.486 / 1.338 / 3.407 | 0.079 / 0.099 / 0.263 / 0.079 | 0.142 / 0.254 / 0.656 / 0.254 |
| DJI_0027:5-30 | SP | 0.422 / 0.237 / 0.221 / 0.327 | 0.409 / 0.224 / 0.214 / 0.339 | 0.067 / 0.045 / 0.082 / 0.043 | 0.077 / 0.112 / 0.107 / 0.048 |
|  | GF | 2.210 / 0.645 / 1.173 / 1.627 | 2.105 / 0.639 / 1.112 / 1.606 | 0.082 / 0.089 / 0.088 / 0.049 | 0.282 / 0.202 / 0.248 / 0.222 |
| DJI_0027:100-120 | SP | 0.393 / 0.225 / 0.573 / 0.303 | 0.407 / 0.205 / 0.535 / 0.308 | 0.062 / 0.093 / 0.231 / 0.037 | 0.060 / 0.042 / 0.104 / 0.024 |
|  | GF | 0.364 / 0.318 / 0.632 / 0.266 | 0.370 / 0.304 / 0.590 / 0.264 | 0.057 / 0.101 / 0.252 / 0.040 | 0.101 / 0.122 / 0.450 / 0.029 |
| DJI_0032:22-47 | SP | 0.221 / 0.203 / 0.250 / 0.327 | 0.215 / 0.185 / 0.257 / 0.357 | 0.060 / 0.078 / 0.053 / 0.031 | 0.069 / 0.070 / 0.059 / 0.042 |
|  | GF | 0.768 / 0.619 / 0.595 / 0.384 | 0.761 / 0.636 / 0.584 / 0.394 | 0.096 / 0.124 / 0.077 / 0.038 | 0.194 / 0.169 / 0.203 / 0.076 |
| DJI_0032:120-140 | SP | 0.142 / 0.089 / 0.083 / 0.054 | 0.131 / 0.061 / 0.063 / 0.050 | 0.049 / 0.062 / 0.054 / 0.020 | 0.080 / 0.091 / 0.074 / 0.036 |
|  | GF | 0.208 / 0.250 / 0.547 / 0.094 | 0.196 / 0.225 / 0.466 / 0.092 | 0.073 / 0.111 / 0.289 / 0.028 | 0.123 / 0.238 / 0.569 / 0.064 |

</details>

## Osmo Action 4 0012 (vs original)

| window | HF (orig) | calm-cruise (orig) | 2-8 Hz (orig) | 8-30 Hz (orig) | roll deg (orig) | 8-30 roll deg (orig) | jello (orig) | row wobble (orig) | corner wobble p-med (orig) | SP-only jumps >1 (>0.5), max | crop plan-exact / fitted | win vs orig | failed checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OA4_0012:146-158 | **0.180** (1.769) | **0.104** (2.380) | 0.180 (1.804) | 0.039 (0.352) | 0.0069 (0.0617) | 0.0014 (0.0221) | 0.753 (0.856) | 0.717 (0.738) | 0.477 (1.127) | 0 (0), 0.00 | 0.5937 / 0.5937 | 100% | **PASS** |
| OA4_0012:176-188 | **1.117** (3.423) | **-** (-) | 1.151 (3.516) | 0.109 (0.318) | 0.0280 (0.1798) | 0.0029 (0.0191) | 2.224 (2.492) | 1.748 (1.751) | 0.489 (1.268) | 2 (3), 1.98 | 0.5945 / 0.5946 | 100% | jumps_gt_1px |
| OA4_0012:196-208 | **1.380** (4.939) | **-** (-) | 1.441 (5.043) | 0.059 (0.456) | 0.0552 (0.2670) | 0.0021 (0.0266) | 1.554 (1.762) | 1.336 (1.610) | 2.983 (3.775) | 1 (6), 1.10 | 0.5942 / 0.5942 | 100% | jumps_gt_1px |
| OA4_0012:300-312 | **0.615** (2.161) | **-** (-) | 0.606 (2.182) | 0.093 (0.337) | 0.0188 (0.1253) | 0.0033 (0.0188) | 1.399 (3.614) | 1.252 (3.248) | 0.352 (0.884) | 0 (3), 0.65 | 0.5939 / 0.5939 | 100% | **PASS** |
| OA4_0012:330-342 | **0.844** (3.253) | **0.093** (2.083) | 0.881 (3.215) | 0.095 (0.390) | 0.0185 (0.1643) | 0.0028 (0.0246) | 0.802 (1.369) | 0.510 (1.232) | 0.344 (1.559) | 0 (3), 0.58 | 0.5941 / 0.5941 | 90% | **PASS** |

Per-camera means [95 % bootstrap CI over 5 windows]: HF 0.827 [0.455, 1.170], calm 0.099 [0.093, 0.104], 2-8 0.852 [0.460, 1.216], 8-30 0.079 [0.055, 0.100], roll deg 0.0255 [0.0135, 0.0406], jello 1.346 [0.902, 1.806], corner 0.929 [0.374, 1.958], jumps >1 px per window 0.60 [0.00, 1.40] (>0.5: 3.00 [1.20, 4.80]), crop 0.5941 [0.5939, 0.5943], 1-s win-rate vs original 98.0 [94.0, 100.0] %.

**3/5 windows pass every check.** Means: HF 0.827 (original 3.109), calm 0.099 (2.231), 2-8 0.852 (3.152), 8-30 0.079 (0.370), roll 0.0255 deg, jello 1.346 (2.019), corner 0.929, crop 0.5941; SP-only jumps >1 px 3 (>0.5: 15). Geo-mean ratio SP/original: HF 0.23, calm 0.04, 2-8 0.23, 8-30 0.20, roll 0.14, jello 0.69. SP HF lower than original in 5/5; pooled 1-s win-rate vs original 98% (5/5 windows >= 90%). Check failures: jumps_gt_1px 2.

<details><summary>Osmo Action 4 0012 (vs original): seconds where Stillpoint loses to the original (1-s HF RMS, worst first; source time s: SP vs the original px @ original <1 Hz speed px/s) and jump times</summary>

- OA4_0012:146-158: lost 0 s
- OA4_0012:176-188: lost 0 s. Jumps: 178.30s 1.98px; 179.12s 1.07px; 179.84s 0.61px
- OA4_0012:196-208: lost 0 s. Jumps: 196.15s 1.10px; 197.32s 0.56px; 197.58s 0.55px; 196.28s 0.54px; 201.00s 0.53px; 196.20s 0.52px
- OA4_0012:300-312: lost 0 s. Jumps: 310.66s 0.65px; 304.47s 0.54px; 311.64s 0.50px
- OA4_0012:330-342: lost 1 s: 338.5s 1.00 vs 0.90 @431. Jumps: 340.64s 0.58px; 332.87s 0.52px; 338.51s 0.50px

</details>

<details><summary>Osmo Action 4 0012 (vs original): axis breakdown (tx / ty / roll / scale, px)</summary>

| window | video | HF tx / ty / roll / scale | 2-8 Hz tx / ty / roll / scale | 8-30 Hz tx / ty / roll / scale | calm HF tx / ty / roll / scale |
|---|---|---|---|---|---|
| OA4_0012:146-158 | SP | 0.074 / 0.066 / 0.084 / 0.125 | 0.075 / 0.070 / 0.085 / 0.121 | 0.017 / 0.015 / 0.017 / 0.027 | 0.064 / 0.053 / 0.046 / 0.043 |
|  | orig | 1.305 / 0.904 / 0.746 / 0.224 | 1.381 / 0.900 / 0.697 / 0.230 | 0.123 / 0.190 / 0.267 / 0.038 | 1.952 / 1.080 / 0.822 / 0.117 |
| OA4_0012:176-188 | SP | 0.511 / 0.423 / 0.338 / 0.834 | 0.529 / 0.434 / 0.319 / 0.869 | 0.042 / 0.041 / 0.035 / 0.085 | - / - / - / - |
|  | orig | 1.862 / 1.686 / 2.174 / 0.828 | 1.938 / 1.722 / 2.233 / 0.809 | 0.115 / 0.169 / 0.231 / 0.076 | - / - / - / - |
| OA4_0012:196-208 | SP | 0.547 / 0.590 / 0.667 / 0.901 | 0.580 / 0.607 / 0.685 / 0.950 | 0.022 / 0.025 / 0.025 / 0.041 | - / - / - / - |
|  | orig | 1.768 / 3.256 / 3.229 / 0.495 | 1.816 / 3.305 / 3.311 / 0.501 | 0.159 / 0.276 / 0.322 / 0.049 | - / - / - / - |
| OA4_0012:300-312 | SP | 0.165 / 0.169 / 0.228 / 0.520 | 0.164 / 0.162 / 0.235 / 0.509 | 0.024 / 0.034 / 0.040 / 0.074 | - / - / - / - |
|  | orig | 0.666 / 1.310 / 1.516 / 0.459 | 0.676 / 1.267 / 1.571 / 0.480 | 0.107 / 0.220 / 0.227 / 0.047 | - / - / - / - |
| OA4_0012:330-342 | SP | 0.460 / 0.171 / 0.224 / 0.649 | 0.486 / 0.168 / 0.233 / 0.677 | 0.041 / 0.028 / 0.034 / 0.073 | 0.050 / 0.072 / 0.028 / 0.017 |
|  | orig | 1.995 / 1.514 / 1.986 / 0.603 | 1.956 / 1.599 / 1.898 / 0.594 | 0.123 / 0.213 / 0.297 / 0.054 | 1.435 / 0.351 / 1.466 / 0.067 |

</details>

## O4 Pro 0004 (vs original)

| window | HF (orig) | calm-cruise (orig) | 2-8 Hz (orig) | 8-30 Hz (orig) | roll deg (orig) | 8-30 roll deg (orig) | jello (orig) | row wobble (orig) | corner wobble p-med (orig) | SP-only jumps >1 (>0.5), max | crop plan-exact / fitted | win vs orig | failed checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O4_0004:45-57 | **0.233** (1.223) | **-** (-) | 0.199 (1.229) | 0.122 (0.258) | 0.0086 (0.0713) | 0.0035 (0.0087) | 0.445 (0.754) | 0.380 (0.684) | 2.848 (2.343) | 1 (3), 1.12 | 0.6141 / 0.6141 | 100% | jumps_gt_1px |
| O4_0004:60-72 | **1.293** (1.402) | **-** (-) | 1.279 (1.403) | 0.158 (0.192) | 0.0374 (0.0572) | 0.0067 (0.0108) | 2.397 (1.921) | 1.767 (1.474) | 1.463 (1.714) | 2 (4), 2.11 | 0.6139 / 0.6139 | 60% | winrate_vs_original, jumps_gt_1px, jello |
| O4_0004:177-189 | **0.528** (1.412) | **0.300** (0.616) | 0.496 (1.406) | 0.216 (0.312) | 0.0132 (0.0598) | 0.0063 (0.0150) | 0.676 (0.933) | 0.552 (0.843) | 6.902 (4.775) | 3 (3), 1.94 | 0.6142 / 0.6150 | 100% | jumps_gt_1px |
| O4_0004:134-146 (calm) | **0.125** (0.518) | **0.115** (0.407) | 0.110 (0.528) | 0.061 (0.108) | 0.0057 (0.0157) | 0.0026 (0.0050) | 0.275 (0.295) | 0.265 (0.277) | 0.694 (1.018) | 0 (0), 0.00 | 0.6132 / 0.6133 | 100% | **PASS** |

Per-camera means [95 % bootstrap CI over 4 windows]: HF 0.545 [0.179, 1.028], calm 0.207 [0.115, 0.300], 2-8 0.521 [0.154, 1.009], 8-30 0.139 [0.085, 0.192], roll deg 0.0162 [0.0072, 0.0302], jello 0.948 [0.360, 1.909], corner 2.977 [1.079, 5.542], jumps >1 px per window 1.50 [0.50, 2.50] (>0.5: 2.50 [0.75, 3.75]), crop 0.6139 [0.6134, 0.6142], 1-s win-rate vs original 90.0 [70.0, 100.0] %.

**1/4 windows pass every check.** Means: HF 0.545 (original 1.139), calm 0.207 (0.512), 2-8 0.521 (1.142), 8-30 0.139 (0.217), roll 0.0162 deg, jello 0.948 (0.976), corner 2.977, crop 0.6139; SP-only jumps >1 px 6 (>0.5: 10). Geo-mean ratio SP/original: HF 0.35, calm 0.37, 2-8 0.32, 8-30 0.62, roll 0.28, jello 0.84. SP HF lower than original in 4/4; pooled 1-s win-rate vs original 90% (3/4 windows >= 90%). Check failures: jello 1, jumps_gt_1px 3, winrate_vs_original 1.

<details><summary>O4 Pro 0004 (vs original): seconds where Stillpoint loses to the original (1-s HF RMS, worst first; source time s: SP vs the original px @ original <1 Hz speed px/s) and jump times</summary>

- O4_0004:45-57: lost 0 s. Jumps: 51.14s 1.12px; 52.86s 0.86px; 54.61s 0.53px
- O4_0004:60-72: lost 4 s: 65.5s 3.10 vs 2.70 @238; 64.5s 1.18 vs 1.10 @268; 68.5s 0.62 vs 0.54 @212; 62.5s 1.94 vs 1.90 @315. Jumps: 65.96s 2.11px; 63.04s 1.71px; 62.34s 0.84px; 62.94s 0.75px
- O4_0004:177-189: lost 0 s. Jumps: 182.12s 1.94px; 180.07s 1.53px; 179.24s 1.30px (also in ref); 179.29s 1.18px
- O4_0004:134-146: lost 0 s

</details>

<details><summary>O4 Pro 0004 (vs original): axis breakdown (tx / ty / roll / scale, px)</summary>

| window | video | HF tx / ty / roll / scale | 2-8 Hz tx / ty / roll / scale | 8-30 Hz tx / ty / roll / scale | calm HF tx / ty / roll / scale |
|---|---|---|---|---|---|
| O4_0004:45-57 | SP | 0.122 / 0.135 / 0.096 / 0.109 | 0.114 / 0.085 / 0.089 / 0.107 | 0.046 / 0.104 / 0.039 / 0.023 | - / - / - / - |
|  | orig | 0.782 / 0.485 / 0.791 / 0.149 | 0.807 / 0.449 / 0.798 / 0.148 | 0.087 / 0.222 / 0.096 / 0.024 | - / - / - / - |
| O4_0004:60-72 | SP | 0.455 / 0.660 / 0.415 / 0.926 | 0.458 / 0.677 / 0.419 / 0.891 | 0.081 / 0.065 / 0.074 / 0.093 | - / - / - / - |
|  | orig | 0.575 / 0.910 / 0.635 / 0.635 | 0.579 / 0.916 / 0.629 / 0.631 | 0.074 / 0.116 / 0.119 / 0.060 | - / - / - / - |
| O4_0004:177-189 | SP | 0.370 / 0.295 / 0.146 / 0.185 | 0.350 / 0.272 / 0.128 / 0.182 | 0.141 / 0.136 / 0.069 / 0.058 | 0.198 / 0.189 / 0.091 / 0.082 |
|  | orig | 0.716 / 0.983 / 0.664 / 0.275 | 0.711 / 0.982 / 0.659 / 0.270 | 0.114 / 0.230 / 0.167 / 0.060 | 0.303 / 0.386 / 0.348 / 0.133 |
| O4_0004:134-146 | SP | 0.067 / 0.076 / 0.064 / 0.036 | 0.059 / 0.066 / 0.057 / 0.030 | 0.033 / 0.037 / 0.029 / 0.020 | 0.066 / 0.062 / 0.060 / 0.037 |
|  | orig | 0.334 / 0.351 / 0.174 / 0.058 | 0.341 / 0.363 / 0.167 / 0.057 | 0.055 / 0.072 / 0.056 / 0.017 | 0.267 / 0.254 / 0.167 / 0.051 |

</details>

## Change vs the engine v5 defaults (`default/`, same ProRes renders)

- **DJI O3** (fill vs v5 default): o3 (10 windows): HF -14.5% [-24.3, -4.0] **better** (noise +-3.0), calm -2.2% [-7.6, +1.7] (noise +-2.8), 2-8 -15.7% [-26.5, -3.9] **better** (noise +-3.5), 8-30 -1.3% [-2.3, -0.4] **better** (noise +-0.9), roll -7.9% [-13.9, -1.8] **better** (noise +-2.2), jello -0.2% [-4.4, +4.0] (noise +-10.6), corner +3.2% [-2.6, +9.2] (noise +-4.7), jumps>1 per window -0.50 [-1.20, 0.00]

| window | HF | calm | 2-8 | 8-30 | roll deg | jello | corner | jumps >1 (>0.5) | crop | win | worse in fill | better in fill |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DJI_0034:15-40 | 0.571 -> 0.365 | 0.134 -> 0.133 | 0.586 -> 0.371 | 0.097 -> 0.093 | 0.0216 -> 0.0171 | 0.394 -> 0.380 | 0.543 -> 0.545 | 1 (3) -> 1 (5) | 0.6252 -> 0.6251 | 100% -> 100% | - | HF 0.571->0.365; 2-8 0.586->0.371; roll deg 0.022->0.017 |
| DJI_0034:1-15 | 0.137 -> 0.138 | 0.138 -> 0.139 | 0.099 -> 0.100 | 0.091 -> 0.091 | 0.0074 -> 0.0074 | 0.286 -> 0.294 | 0.626 -> 0.601 | 0 (0) -> 0 (0) | 0.6237 -> 0.6236 | 100% -> 100% | - | - |
| DJI_0025:15-40 | 0.308 -> 0.303 | 0.312 -> 0.314 | 0.289 -> 0.283 | 0.122 -> 0.121 | 0.0155 -> 0.0154 | 0.660 -> 0.744 | 0.693 -> 0.691 | 0 (3) -> 0 (3) | 0.6293 -> 0.6293 | 100% -> 100% | - | - |
| DJI_0025:1-15 | 0.516 -> 0.395 | 0.150 -> 0.144 | 0.519 -> 0.398 | 0.094 -> 0.091 | 0.0100 -> 0.0107 | 1.005 -> 1.018 | 0.589 -> 0.507 | 1 (3) -> 0 (4) | 0.6293 -> 0.6293 | 100% -> 100% | - | HF 0.516->0.395; 2-8 0.519->0.398 |
| DJI_0028:8-33 | 0.822 -> 0.830 | 0.130 -> 0.139 | 0.771 -> 0.784 | 0.294 -> 0.295 | 0.0517 -> 0.0504 | 1.335 -> 1.227 | 0.810 -> 0.919 | 4 (5) -> 4 (4) | 0.6344 -> 0.6331 | 87% -> 91% | - | - |
| DJI_0028:38-58 | 2.241 -> 1.902 | 0.429 -> 0.405 | 2.217 -> 1.893 | 0.206 -> 0.201 | 0.0395 -> 0.0314 | 0.833 -> 0.885 | 0.957 -> 1.164 | 4 (13) -> 3 (13) | 0.6308 -> 0.6333 | 94% -> 94% | corner 0.957->1.164 | HF 2.241->1.902; 2-8 2.217->1.893; roll deg 0.039->0.031 |
| DJI_0027:5-30 | 0.983 -> 0.625 | 0.228 -> 0.179 | 0.979 -> 0.615 | 0.125 -> 0.122 | 0.0223 -> 0.0199 | 0.913 -> 0.798 | 0.769 -> 0.877 | 3 (9) -> 0 (4) | 0.6066 -> 0.6037 | 70% -> 83% | - | HF 0.983->0.625; calm 0.228->0.179; 2-8 0.979->0.615; roll deg 0.022->0.020; jumps>1 3->0; jumps>0.5 9->4 |
| DJI_0027:100-120 | 0.743 -> 0.791 | 0.126 -> 0.130 | 0.713 -> 0.768 | 0.258 -> 0.259 | 0.0522 -> 0.0517 | 1.629 -> 1.686 | 0.726 -> 0.760 | 0 (3) -> 0 (4) | 0.6062 -> 0.6031 | 61% -> 61% | crop 0.6062->0.6031 | - |
| DJI_0032:22-47 | 0.496 -> 0.510 | 0.121 -> 0.122 | 0.505 -> 0.523 | 0.116 -> 0.116 | 0.0235 -> 0.0226 | 0.510 -> 0.508 | 0.574 -> 0.589 | 0 (0) -> 0 (1) | 0.5891 -> 0.5889 | 96% -> 96% | - | - |
| DJI_0032:120-140 | 0.270 -> 0.194 | 0.147 -> 0.146 | 0.270 -> 0.166 | 0.099 -> 0.098 | 0.0094 -> 0.0074 | 0.251 -> 0.247 | 0.463 -> 0.451 | 0 (2) -> 0 (1) | 0.5877 -> 0.5875 | 94% -> 100% | - | HF 0.270->0.194; 2-8 0.270->0.166; roll deg 0.009->0.007 |
| **o3 mean (10 common)** | 0.709 -> 0.605 | 0.191 -> 0.185 | 0.695 -> 0.590 | 0.150 -> 0.149 | 0.0253 -> 0.0234 | 0.782 -> 0.779 | 0.675 -> 0.710 | 13 -> 8 | | | 2 windows | 5 windows |

- **Osmo Action 4**: no change (the default config is the v5 default; identical plans).
- **O4 Pro**: no change (the default config is the v5 default; identical plans).

Engine v5 default plans equal v4's (5/7 byte-identical, the others within 0.03 px; ENGINE_SPEC.md), so the v5-default rows stand for v4 as well.

## Older scoreboards (HEVC renders, for context only)

Rendered with the VideoToolbox HEVC encoder, whose output is not deterministic (HF up to ~20 % on the same plan): NOT comparable number-for-number with the ProRes tables above.

- v4 DJI O3: **2/10 windows pass every check.** Means: HF 0.682 (Gyroflow 1.559), calm 0.193 (0.610), 2-8 0.651 (1.502), 8-30 0.150 (0.270), roll 0.0230 deg, jello 0.792 (0.835), corner 0.672, crop 0.6162; SP-only jumps >1 px 14 (>0.5: 47). Geo-mean ratio SP/Gyroflow: HF 0.41, calm 0.30, 2-8 0.40, 8-30 0.52, roll 0.30, jello 0.95. SP HF lower than Gyroflow in 10/10; pooled 1-s win-rate vs Gyroflow 90% (8/10 windows >= 90%). Check failures: calm_cruise 1, corner_wobble 2, crop_footprint 2, jello 2, jumps_gt_1px 6, winrate_vs_gyroflow 2.
- v4 Osmo Action 4: **4/5 windows pass every check.** Means: HF 0.757 (original 3.109), calm 0.090 (2.231), 2-8 0.783 (3.152), 8-30 0.076 (0.370), roll 0.0257 deg, jello 1.571 (2.019), corner 0.613, crop 0.5941; SP-only jumps >1 px 0 (>0.5: 15). Geo-mean ratio SP/original: HF 0.22, calm 0.04, 2-8 0.22, 8-30 0.20, roll 0.15, jello 0.79. SP HF lower than original in 5/5; pooled 1-s win-rate vs original 98% (5/5 windows >= 90%). Check failures: jello 1.
- v4 O4 Pro: **1/4 windows pass every check.** Means: HF 0.480 (original 1.139), calm 0.220 (0.512), 2-8 0.457 (1.142), 8-30 0.139 (0.217), roll 0.0147 deg, jello 1.038 (0.976), corner 3.113, crop 0.6139; SP-only jumps >1 px 7 (>0.5: 14). Geo-mean ratio SP/original: HF 0.32, calm 0.39, 2-8 0.28, 8-30 0.62, roll 0.26, jello 0.92. SP HF lower than original in 4/4; pooled 1-s win-rate vs original 95% (3/4 windows >= 90%). Check failures: jello 1, jumps_gt_1px 3, winrate_vs_original 1.
- v5 DJI O3: **1/10 windows pass every check.** Means: HF 0.681 (Gyroflow 1.559), calm 0.198 (0.610), 2-8 0.653 (1.502), 8-30 0.149 (0.270), roll 0.0232 deg, jello 0.763 (0.835), corner 0.743, crop 0.6162; SP-only jumps >1 px 13 (>0.5: 45). Geo-mean ratio SP/Gyroflow: HF 0.41, calm 0.30, 2-8 0.40, 8-30 0.52, roll 0.30, jello 0.93. SP HF lower than Gyroflow in 10/10; pooled 1-s win-rate vs Gyroflow 93% (8/10 windows >= 90%). Check failures: calm_cruise 3, corner_wobble 2, crop_footprint 2, jello 3, jumps_gt_1px 5, winrate_vs_gyroflow 2.
- v5 Osmo Action 4: **4/5 windows pass every check.** Means: HF 0.791 (original 3.109), calm 0.089 (2.231), 2-8 0.818 (3.152), 8-30 0.075 (0.370), roll 0.0270 deg, jello 1.464 (2.019), corner 0.933, crop 0.5941; SP-only jumps >1 px 0 (>0.5: 13). Geo-mean ratio SP/original: HF 0.23, calm 0.04, 2-8 0.23, 8-30 0.19, roll 0.15, jello 0.76. SP HF lower than original in 5/5; pooled 1-s win-rate vs original 98% (5/5 windows >= 90%). Check failures: jello 1.
- v5 O4 Pro: **1/4 windows pass every check.** Means: HF 0.535 (original 1.139), calm 0.220 (0.512), 2-8 0.510 (1.142), 8-30 0.147 (0.217), roll 0.0182 deg, jello 1.033 (0.976), corner 3.004, crop 0.6139; SP-only jumps >1 px 6 (>0.5: 11). Geo-mean ratio SP/original: HF 0.34, calm 0.39, 2-8 0.30, 8-30 0.65, roll 0.28, jello 0.92. SP HF lower than original in 4/4; pooled 1-s win-rate vs original 95% (3/4 windows >= 90%). Check failures: jello 1, jumps_gt_1px 3, winrate_vs_original 1.

## Whole-clip analyses (the default config of each camera)

| clip | frames | analysis s (x realtime) | other heavy slots busy (max / mean) | stages s | workers | peak RSS GB tree / main [stage] | loop | quality HF orig -> SP | calm | 8-30 | jello | new jumps >1 (>0.5), max | crop mean (min) | zoom frac / max | VT before -> after | cache dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DJI_0034 | 3029 | 270 (5.31) | 0 / 0.00 | closed_loop 228, fill 3, quality 22 | 3 | 2.29 / 1.09 [quality] | 2 folds | 5.829 -> 0.234 | 1.172 -> 0.148 | 0.398 -> 0.079 | 1.062 -> 0.364 | 0 (0), 0.00 | 0.6242 (0.6122) | 0.000 / 1.000 | 4 -> 4 | `DJI_0034-4c025de1-93beefde41` |
| DJI_0025 | 2965 | 285 (5.65) | 0 / 0.00 | closed_loop 243, fill 2, quality 19 | 3 | 2.37 / 1.15 [quality] | 2 folds | 0.852 -> 0.144 | 0.652 -> 0.138 | 0.219 -> 0.090 | 0.298 -> 0.240 | 0 (0), 0.00 | 0.6291 (0.6156) | 0.000 / 1.000 | 4 -> 4 | `DJI_0025-4c025de1-d48f945dcd` |
| DJI_0028 | 3833 | 331 (5.13) | 0 / 0.00 | closed_loop 270, fill 4, quality 30 | 3 | 2.35 / 1.20 [fill] | 2 folds | 4.261 -> 1.048 | 1.970 -> 0.229 | 0.528 -> 0.271 | 1.568 -> 0.946 | 5 (10), 3.02 | 0.6303 (0.4984) | 0.047 / 1.124 | 4 -> 4 | `DJI_0028-4c025de1-0b7893635e` |
| DJI_0027 | 11609 | 957 (4.92) | 0 / 0.00 | closed_loop 817, fill 17, quality 87 | 3 | 3.19 / 1.01 [final] | 2 folds | 5.591 -> 1.874 | 1.418 -> 0.240 | 0.610 -> 0.288 | 2.040 -> 1.474 | 17 (26), 9.24 | 0.6030 (0.5654) | 0.019 / 1.048 | 4 -> 4 | `DJI_0027-4c025de1-d69d75104a` |
| DJI_0032 | 11802 | 997 (5.04) | 0 / 0.00 | closed_loop 879, fill 13, quality 70 | 3 | 3.14 / 1.04 [fold2] | 2 folds | 3.725 -> 0.223 | 1.825 -> 0.153 | 0.477 -> 0.100 | 0.638 -> 0.265 | 1 (3), 1.63 | 0.5879 (0.5772) | 0.000 / 1.000 | 4 -> 4 | `DJI_0032-4c025de1-62b4f246db` |
| OA4_0012 | 23950 | 145 (0.36) | 0 / 0.00 | closed_loop 0, quality 113 | 3 | 3.18 / 0.56 [crop] | 0 folds | 2.488 -> 0.664 | 1.955 -> 0.785 | 0.352 -> 0.065 | 1.646 -> 1.261 | 1 (12), 1.03 | 0.5924 (0.5334) | 0.008 / 1.083 | 4 -> 4 | `OA4_0012-4c025de1-e0adedd9f3` |
| O4_0004 | 11482 | 95 (0.47) | 1 / 0.75 | closed_loop 0, quality 72 | 3 | 2.03 / 0.30 [crop] | 0 folds | 2.375 -> 0.584 | 0.573 -> 0.249 | 0.297 -> 0.109 | 1.627 -> 1.370 | 1 (3), 1.08 | 0.6144 (0.6058) | 0.000 / 1.000 | 5 -> 4 | `O4_0004-4c025de1-359d931535` |
