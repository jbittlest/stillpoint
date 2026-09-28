// Stillpoint web — the warp kernel, WGSL port of shaders/warp.metal (ENGINE_SPEC.md §1, §3).  Owner: GPU agent.
//
// GEOMETRY (identical to warp.metal::sp_source_coord and engine/stillpoint/render_ref.py::_source_coord, fp32):
//   r_v = ((X - out_cx)/out_fx, (Y - out_cy)/out_fx, 1)          X,Y = full-res output pixel (centres at integers)
//   3 evaluations of { M(y) = lerp of the plan's row matrices at source row y; r_c = M(y) r_v; (u,v) = lens(r_c) }:
//   y0 = (src_h-1)/2, y1 = v0 (fixed point), y2 = secant step on g(y) = v(y) - y through (y0,g0),(y1,g1)
//   (fallback y2 = v1 when the implied slope is outside [-2, -0.1]). Row matrix j sits at source row
//   y_j = j*(src_h-1)/(n_rows-1); rows outside [0, src_h-1] clamp. Valid iff r_c.z > 0 and
//   -0.5 <= u <= src_w-0.5, -0.5 <= v <= src_h-0.5; invalid -> black.
//   atan2 / sin / cos are evaluated with our own polynomials (< 2e-7 abs error): WGSL only guarantees
//   4096 ULP for atan2 and 2^-11 absolute for sin/cos, which would be ~0.5 px at the image edge.
//
// COLOUR. Input is a GPUExternalTexture of a WebCodecs VideoFrame. textureLoad on it returns
//   E = T_dst(T_src^-1(R'G'B')) where R'G'B' = the source Y'CbCr through its matrix/range, T_dst = sRGB and T_src is
//   whatever transfer Chrome assigns the frame (macOS VideoToolbox frames: pure gamma 1.961, others: sRGB = no-op).
//   The convert pass undoes that (x = T_src(T_dst^-1(E)), sign-symmetric; measured exact to 4e-7) and splits x into
//   Y'CbCr with the source matrix, so all resampling happens on the camera's own gamma-encoded values like the Metal
//   kernel; the warp writes R'G'B' to an 8-bit canvas that VideoEncoder turns back into Y'CbCr with the BT.709 matrix
//   (measured) -> no double gamma. Two blind tests on a sparse grid of aligned 2x2 blocks (calib passes):
//     V-test: frames imported as 4:2:0 planes (decoder frames) come with nearest-neighbour chroma, so under the right
//             T_src Cb,Cr are constant inside every aligned 2x2 block (variance ~1e-16 vs >= 1e-9 otherwise);
//     Q-test: frames Chrome converts to RGBA8 before import (10-bit P010 on macOS, CPU-memory frames) are exact
//             multiples of 1/255 under the right T_src.
//   T_src = winner of the tests accumulated over frames (default: platform guess). Chroma MODE is decided per frame:
//     planar (this frame's V-test is decisive): chroma plane = one CbCr per 2x2 block (exact source chroma), sampled
//       at left-sited positions (chroma (i,j) at luma (2i, 2j+0.5), DJI/MPEG) exactly like warp.metal sp_warp_chroma;
//     rgb (otherwise): the browser already upsampled chroma; CbCr is sampled per pixel position from a full-res plane.
//   Output chroma is evaluated once per 2x2 output block at its chroma site (2i, 2j+0.5) and shared by its 4 pixels,
//   so the encoder's 4:2:0 subsampling returns exactly it; luma is evaluated per pixel.
// SAMPLING: Lanczos-3 (6x6, per-axis normalised; default), Catmull-Rom (a=-0.5) or bilinear — override TAPS = 6/4/2 —
//   manual taps clamped to the edge, fetched 2x2 at a time with textureGather from float intermediates (the external
//   texture itself is read once per source pixel, by the convert pass).

struct Params {
  outFx: f32, outCx: f32, outCy: f32, lensModel: u32,     // lensModel: 0 pinhole, 1 kb4
  fx: f32, fy: f32, cx: f32, cy: f32,
  k1: f32, k2: f32, k3: f32, k4: f32,
  srcW: f32, srcH: f32, nRows: u32, iters: u32,
  outSx: f32, outSy: f32, dstW: u32, dstH: u32,          // output grid size (+ scale for the coord map)
  kr: f32, kb: f32, defTransfer: u32, forceTransfer: i32,  // source matrix; transfer default; >= 0 forces a transfer
  calibX: u32, calibY: u32, forceMode: u32, dbg: u32,     // calib grid (8x8 workgroups); 0 auto/1 planar/2 rgb; debug
};

// Transfers T_src: 0 = sRGB (no-op), 1 = gamma 1.961 (Apple BT.709), 2 = BT.709 OETF, 3 = gamma 2.2, 4 = gamma 2.4,
// 5 = linear. Keep NT in sync with warp.ts TRANSFERS.
const NT: u32 = 6u;

struct ColorState {
  accV: array<f32, 8>,
  accQ: array<f32, 8>,
  frames: u32,
  transfer: u32,
  how: u32,          // 0 default, 1 V-test, 2 Q-test, 3 forced
  planar: u32,       // chroma mode of the current frame: 1 planar, 0 rgb
};

override TAPS: i32 = 6;
override GATHER: bool = true;     // float intermediates: textureGather quads (else textureLoad taps)

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var ext: texture_external;
@group(0) @binding(2) var<storage, read_write> cstate: ColorState;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;
@group(0) @binding(4) var outY: texture_storage_2d<r32float, write>;     // Y' full res
@group(0) @binding(5) var outCH: texture_storage_2d<rg32float, write>;   // CbCr per 2x2 block (planar chroma plane)
@group(0) @binding(6) var outCF: texture_storage_2d<r32uint, write>;     // CbCr per pixel, pack2x16float
@group(0) @binding(7) var<storage, read> M: array<f32>;
@group(0) @binding(8) var texY: texture_2d<f32>;
@group(0) @binding(9) var texCH: texture_2d<f32>;
@group(0) @binding(10) var texCF: texture_2d<u32>;
@group(0) @binding(11) var<storage, read_write> coordOut: array<f32>;
@group(0) @binding(12) var<storage, read> cstateR: ColorState;
@group(0) @binding(13) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(14) var smp: sampler;   // non-filtering, clamp-to-edge (gathers only)

const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;

// ------------------------------------------------------------------------------------------ accurate math
// sin(pi*t), cos(pi*t) for |t| <= 0.5 (Taylor to x^11 / x^12: abs error < 6e-8)
fn sinpi_h(t: f32) -> f32 {
  let x = PI * t; let x2 = x * x;
  return x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0 + x2 * (-1.0 / 39916800.0))))));
}
fn cospi_h(t: f32) -> f32 {
  let x = PI * t; let x2 = x * x;
  return 1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40320.0 + x2 * (-1.0 / 3628800.0 + x2 * (1.0 / 479001600.0))))));
}
// sin(pi*f) for f in [0,1]
fn sinpi01(f: f32) -> f32 { return sinpi_h(select(f, 1.0 - f, f > 0.5)); }

// atan(a) for a in [0, 1]: reduce to |t| <= tan(pi/8) via atan(a) = pi/4 + atan((a-1)/(a+1)); series to t^19.
fn atan01(a: f32) -> f32 {
  var base = 0.0;
  var t = a;
  if (a > 0.41421356) { t = (a - 1.0) / (a + 1.0); base = 0.78539816339745; }
  let t2 = t * t;
  let p = 1.0 + t2 * (-1.0 / 3.0 + t2 * (1.0 / 5.0 + t2 * (-1.0 / 7.0 + t2 * (1.0 / 9.0 + t2 * (-1.0 / 11.0 +
          t2 * (1.0 / 13.0 + t2 * (-1.0 / 15.0 + t2 * (1.0 / 17.0 + t2 * (-1.0 / 19.0)))))))));
  return base + t * p;
}
// atan2(y, x) for y >= 0 -> [0, pi]
fn atan2_pos(y: f32, x: f32) -> f32 {
  let ax = abs(x);
  let mx = max(y, ax);
  if (mx <= 0.0) { return 0.0; }
  var r = atan01(min(y, ax) / mx);
  if (y > ax) { r = HALF_PI - r; }
  if (x < 0.0) { r = PI - r; }
  return r;
}

// ------------------------------------------------------------------------------------------ geometry
fn rows_apply(y: f32, rv: vec3f) -> vec3f {
  let nr = f32(P.nRows);
  let g = clamp(y * (nr - 1.0) / (P.srcH - 1.0), 0.0, nr - 1.0);
  let j0 = min(i32(g), i32(P.nRows) - 2);
  let f = g - f32(j0);
  let a = u32(j0) * 9u;
  let b = a + 9u;
  let ra = vec3f(M[a] * rv.x + M[a + 1u] * rv.y + M[a + 2u] * rv.z,
                 M[a + 3u] * rv.x + M[a + 4u] * rv.y + M[a + 5u] * rv.z,
                 M[a + 6u] * rv.x + M[a + 7u] * rv.y + M[a + 8u] * rv.z);
  let rb = vec3f(M[b] * rv.x + M[b + 1u] * rv.y + M[b + 2u] * rv.z,
                 M[b + 3u] * rv.x + M[b + 4u] * rv.y + M[b + 5u] * rv.z,
                 M[b + 6u] * rv.x + M[b + 7u] * rv.y + M[b + 8u] * rv.z);
  return ra + (rb - ra) * f;
}

fn project(r: vec3f) -> vec2f {
  if (P.lensModel == 0u) {
    let z = select(min(r.z, -1e-12), max(r.z, 1e-12), r.z > 0.0);
    return vec2f(P.fx * r.x / z + P.cx, P.fy * r.y / z + P.cy);
  }
  let rxy = sqrt(r.x * r.x + r.y * r.y);
  let th = atan2_pos(rxy, r.z);
  let t2 = th * th;
  let thd = th * (1.0 + t2 * (P.k1 + t2 * (P.k2 + t2 * (P.k3 + t2 * P.k4))));
  var s: f32;
  if (rxy < 1e-12) { s = 1.0 / max(r.z, 1e-12); } else { s = thd / rxy; }
  return vec2f(P.fx * r.x * s + P.cx, P.fy * r.y * s + P.cy);
}

// full-res output pixel -> (full-res source luma pixel u, v, valid 1/0)
fn source_coord(X: vec2f) -> vec3f {
  let rv = vec3f((X.x - P.outCx) / P.outFx, (X.y - P.outCy) / P.outFx, 1.0);
  let iters = max(i32(P.iters), 1);
  var y = 0.5 * (P.srcH - 1.0);
  var ya = 0.0;
  var ga = 0.0;
  var uv = vec2f(0.0);
  var z = 1.0;
  for (var e = 0; e < iters; e++) {
    let rc = rows_apply(y, rv);
    uv = project(rc);
    z = rc.z;
    let g = uv.y - y;
    var yn = uv.y;
    if (e >= 1) {
      let dy = y - ya;
      if (dy != 0.0) {
        let s = (g - ga) / dy;               // slope of g = c - 1
        if (s <= -0.1 && s >= -2.0) { yn = y - g / s; }
      }
    }
    ya = y; ga = g; y = yn;
  }
  let ok = (z > 0.0) && (uv.x >= -0.5) && (uv.y >= -0.5) && (uv.x <= P.srcW - 0.5) && (uv.y <= P.srcH - 0.5);
  return vec3f(uv, select(0.0, 1.0, ok));
}

// ------------------------------------------------------------------------------------------ resampling
// Lanczos-3 weights of taps i = -2..3 (x = f - i), normalised, from sin(pi f), sin(pi f/3), cos(pi f/3) using
// sin(pi x) = (-1)^i sin(pi f) and sin(pi x/3) = sin(pi f/3)cos(pi i/3) - cos(pi f/3)sin(pi i/3) — as warp.metal,
// written out per tap (no dynamic indexing, so the arrays stay in registers).
fn lz(x: f32, a: f32, b: f32) -> f32 {
  return select(3.0 * a * b / (PI * PI * x * x), 1.0, abs(x) < 1e-4);
}
fn weights(f: f32) -> array<f32, 6> {
  if (TAPS == 6) {
    let s1 = sinpi01(f);
    let s3 = sinpi_h(f / 3.0);
    let c3 = cospi_h(f / 3.0);
    let h = 0.86602540378444 * c3;
    let w0 = lz(f + 2.0, s1, h - 0.5 * s3);
    let w1 = lz(f + 1.0, -s1, h + 0.5 * s3);
    let w2 = lz(f, s1, s3);
    let w3 = lz(f - 1.0, -s1, 0.5 * s3 - h);
    let w4 = lz(f - 2.0, s1, -0.5 * s3 - h);
    let w5 = lz(f - 3.0, -s1, -s3);
    let n = 1.0 / (w0 + w1 + w2 + w3 + w4 + w5);
    return array<f32, 6>(w0 * n, w1 * n, w2 * n, w3 * n, w4 * n, w5 * n);
  } else if (TAPS == 4) {
    let t2 = f * f; let t3 = t2 * f;
    return array<f32, 6>(-0.5 * t3 + t2 - 0.5 * f, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * f,
                         0.5 * t3 - 0.5 * t2, 0.0, 0.0);
  }
  return array<f32, 6>(1.0 - f, f, 0.0, 0.0, 0.0, 0.0);
}

fn tap_offset() -> f32 { return select(select(0.0, -1.0, TAPS == 4), -2.0, TAPS == 6); }

// Separable N-tap resampling at float texel coords s (pixel centres at integers), clamp-to-edge taps. With GATHER one
// textureGather fetches the 2x2 quad (x0..x0+1, y0..y0+1) when sampled at the shared corner (x0+1, y0+1)/dims —
// exact integer texel selection, clamp-to-edge addressing = clamped taps; a 6x6 Lanczos window is 9 gathers per
// channel instead of 36 loads. Gather order: .w (x0,y0) .z (x0+1,y0) .x (x0,y0+1) .y (x0+1,y0+1).
struct TapSet { c0: vec2f, inv: vec2f, wx: array<f32, 6>, wy: array<f32, 6> };
fn taps(t: texture_2d<f32>, s: vec2f) -> TapSet {
  let fl = floor(s);
  let f = s - fl;
  return TapSet(fl + tap_offset() + 1.0, 1.0 / vec2f(textureDimensions(t)), weights(f.x), weights(f.y));
}
fn quad(g: vec4f, T: TapSet, qx: i32, qy: i32) -> f32 {
  return T.wy[2 * qy] * (T.wx[2 * qx] * g.w + T.wx[2 * qx + 1] * g.z) +
         T.wy[2 * qy + 1] * (T.wx[2 * qx] * g.x + T.wx[2 * qx + 1] * g.y);
}
fn sample_loads(t: texture_2d<f32>, s: vec2f) -> vec4f {
  let mx = vec2i(textureDimensions(t)) - 1;
  let fl = floor(s);
  let f = s - fl;
  let b = vec2i(fl) + i32(tap_offset());
  let wx = weights(f.x);
  let wy = weights(f.y);
  var acc = vec4f(0.0);
  for (var j = 0; j < TAPS; j++) {
    let yy = clamp(b.y + j, 0, mx.y);
    var row = vec4f(0.0);
    for (var i = 0; i < TAPS; i++) {
      row += wx[i] * textureLoad(t, vec2i(clamp(b.x + i, 0, mx.x), yy), 0);
    }
    acc += wy[j] * row;
  }
  return acc;
}
fn sample1(t: texture_2d<f32>, s: vec2f) -> f32 {
  if (!GATHER) { return sample_loads(t, s).x; }
  let T = taps(t, s);
  var acc = 0.0;
  for (var qy = 0; qy < TAPS / 2; qy++) {
    for (var qx = 0; qx < TAPS / 2; qx++) {
      let c = (T.c0 + vec2f(f32(2 * qx), f32(2 * qy))) * T.inv;
      acc += quad(textureGather(0, t, smp, c), T, qx, qy);
    }
  }
  return acc;
}
fn sample2(t: texture_2d<f32>, s: vec2f) -> vec2f {
  if (!GATHER) { return sample_loads(t, s).xy; }
  let T = taps(t, s);
  var acc = vec2f(0.0);
  for (var qy = 0; qy < TAPS / 2; qy++) {
    for (var qx = 0; qx < TAPS / 2; qx++) {
      let c = (T.c0 + vec2f(f32(2 * qx), f32(2 * qy))) * T.inv;
      acc += vec2f(quad(textureGather(0, t, smp, c), T, qx, qy), quad(textureGather(1, t, smp, c), T, qx, qy));
    }
  }
  return acc;
}
// full-res packed CbCr (rgb mode), plain loads + unpack
fn sample_cf(s: vec2f) -> vec2f {
  let mx = vec2i(textureDimensions(texCF)) - 1;
  let fl = floor(s);
  let f = s - fl;
  let b = vec2i(fl) + i32(tap_offset());
  let wx = weights(f.x);
  let wy = weights(f.y);
  var acc = vec2f(0.0);
  for (var j = 0; j < TAPS; j++) {
    let yy = clamp(b.y + j, 0, mx.y);
    var row = vec2f(0.0);
    for (var i = 0; i < TAPS; i++) {
      row += wx[i] * unpack2x16float(textureLoad(texCF, vec2i(clamp(b.x + i, 0, mx.x), yy), 0).x);
    }
    acc += wy[j] * row;
  }
  return acc;
}

// ------------------------------------------------------------------------------------------ colour
fn srgb_dec(e: f32) -> f32 {
  let a = abs(e);
  let l = select(pow((a + 0.055) / 1.055, 2.4), a / 12.92, a <= 0.04045);
  return select(l, -l, e < 0.0);
}
fn spow(a: f32, g: f32) -> f32 { return select(pow(a, g), 0.0, a <= 0.0); }   // a >= 0
fn t_enc(l: f32, tr: u32) -> f32 {     // linear -> T_src-encoded (sign-symmetric)
  let a = abs(l);
  var r: f32;
  switch tr {
    case 1u: { r = spow(a, 1.0 / 1.961); }
    case 2u: { r = select(1.099 * spow(a, 0.45) - 0.099, 4.5 * a, a < 0.018); }
    case 3u: { r = spow(a, 1.0 / 2.2); }
    case 4u: { r = spow(a, 1.0 / 2.4); }
    default: { r = a; }
  }
  return select(r, -r, l < 0.0);
}
fn inv_transfer(e: vec3f, tr: u32) -> vec3f {
  if (tr == 0u) { return e; }
  return vec3f(t_enc(srgb_dec(e.x), tr), t_enc(srgb_dec(e.y), tr), t_enc(srgb_dec(e.z), tr));
}
fn to_ycc(x: vec3f) -> vec3f {
  let kg = 1.0 - P.kr - P.kb;
  let y = P.kr * x.x + kg * x.y + P.kb * x.z;
  return vec3f(y, (x.z - y) / (2.0 * (1.0 - P.kb)), (x.x - y) / (2.0 * (1.0 - P.kr)));
}
fn to_rgb(c: vec3f) -> vec3f {
  let kg = 1.0 - P.kr - P.kb;
  let r = c.x + 2.0 * (1.0 - P.kr) * c.z;
  let b = c.x + 2.0 * (1.0 - P.kb) * c.y;
  return vec3f(r, (c.x - P.kr * r - P.kb * b) / kg, b);
}

// ---- calibration: per-frame blind tests on a sparse grid of aligned 2x2 blocks (partials), reduced by calib_reduce
var<workgroup> red: array<array<f32, 12>, 64>;

@compute @workgroup_size(8, 8)
fn calib_blocks(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
                @builtin(local_invocation_index) li: u32) {
  let dims = textureDimensions(ext);
  let nbx = P.calibX * 8u;
  let nby = P.calibY * 8u;
  let gx = wid.x * 8u + lid.x;
  let gy = wid.y * 8u + lid.y;
  let hb = max(dims / 2u, vec2u(1u));                      // complete 2x2 blocks
  let bx = min(u32((f32(gx) + 0.5) * f32(hb.x) / f32(nbx)), hb.x - 1u);
  let by = min(u32((f32(gy) + 0.5) * f32(hb.y) / f32(nby)), hb.y - 1u);
  var e: array<vec3f, 4>;
  e[0] = textureLoad(ext, vec2u(2u * bx, 2u * by)).rgb;
  e[1] = textureLoad(ext, vec2u(2u * bx + 1u, 2u * by)).rgb;
  e[2] = textureLoad(ext, vec2u(2u * bx, 2u * by + 1u)).rgb;
  e[3] = textureLoad(ext, vec2u(2u * bx + 1u, 2u * by + 1u)).rgb;
  for (var c = 0u; c < NT; c++) {
    var cb: array<f32, 4>;
    var cr: array<f32, 4>;
    var q = 0.0;
    var mb = 0.0;
    var mr = 0.0;
    for (var i = 0; i < 4; i++) {
      let x = inv_transfer(e[i], c);
      let ycc = to_ycc(x);
      cb[i] = ycc.y; cr[i] = ycc.z;
      mb += ycc.y; mr += ycc.z;
      let d = x * 255.0 - round(x * 255.0);
      q += dot(d, d);
    }
    mb *= 0.25; mr *= 0.25;
    var v = 0.0;
    for (var i = 0; i < 4; i++) { v += (cb[i] - mb) * (cb[i] - mb) + (cr[i] - mr) * (cr[i] - mr); }
    red[li][c] = v;
    red[li][6u + c] = q;
  }
  workgroupBarrier();
  if (li == 0u) {
    var s: array<f32, 12>;
    for (var i = 0u; i < 64u; i++) { for (var c = 0u; c < 12u; c++) { s[c] += red[i][c]; } }
    let base = (wid.y * P.calibX + wid.x) * 12u;
    for (var c = 0u; c < 12u; c++) { partials[base + c] = s[c]; }
  }
}

// two smallest values of v[0..NT) -> (argmin, min, second min)
fn two_min(v: array<f32, 8>) -> vec3f {
  var c1 = 0u; var v1 = 3.4e38; var v2 = 3.4e38;
  for (var c = 0u; c < NT; c++) {
    if (v[c] < v1) { v2 = v1; v1 = v[c]; c1 = c; } else if (v[c] < v2) { v2 = v[c]; }
  }
  return vec3f(f32(c1), v1, v2);
}

@compute @workgroup_size(1)
fn calib_reduce() {
  let nwg = P.calibX * P.calibY;
  var fV: array<f32, 8>;
  var fQ: array<f32, 8>;
  for (var c = 0u; c < 12u; c++) {
    var s = 0.0;
    for (var w = 0u; w < nwg; w++) { s += partials[w * 12u + c]; }
    if (c < 6u) { fV[c] = s; cstate.accV[c] += s; } else { fQ[c - 6u] = s; cstate.accQ[c - 6u] += s; }
  }
  cstate.frames += 1u;
  // chroma mode of THIS frame: planar iff its own V-test is decisive (3 orders of magnitude)
  let fv = two_min(fV);
  var planar = fv.z > 1e-9 && fv.y < 1e-3 * fv.z;
  if (P.forceMode == 1u) { planar = true; } else if (P.forceMode == 2u) { planar = false; }
  cstate.planar = select(0u, 1u, planar);
  // transfer: accumulated evidence
  if (P.forceTransfer >= 0) { cstate.transfer = u32(P.forceTransfer); cstate.how = 3u; return; }
  var accV: array<f32, 8>;
  var accQ: array<f32, 8>;
  for (var c = 0u; c < NT; c++) { accV[c] = cstate.accV[c]; accQ[c] = cstate.accQ[c]; }
  let av = two_min(accV);
  let aq = two_min(accQ);
  if (av.z > 1e-9 && av.y < 1e-3 * av.z) { cstate.transfer = u32(av.x); cstate.how = 1u; }
  else if (aq.z > 1.0 && aq.y < 0.02 * aq.z) { cstate.transfer = u32(aq.x); cstate.how = 2u; }
  else { cstate.transfer = P.defTransfer; cstate.how = 0u; }
}

// ---- convert: external texture -> camera-domain Y' (full), CbCr per block (planar), CbCr per pixel (rgb)
@compute @workgroup_size(8, 8)
fn convert(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(ext);
  let hd = (dims + 1u) / 2u;
  if (gid.x >= hd.x || gid.y >= hd.y) { return; }
  let tr = cstate.transfer;
  var cs = vec2f(0.0);
  var n = 0.0;
  for (var dy = 0u; dy < 2u; dy++) {
    for (var dx = 0u; dx < 2u; dx++) {
      let p = vec2u(2u * gid.x + dx, 2u * gid.y + dy);
      if (p.x < dims.x && p.y < dims.y) {
        let ycc = to_ycc(inv_transfer(textureLoad(ext, p).rgb, tr));
        textureStore(outY, p, vec4f(ycc.x, 0.0, 0.0, 1.0));
        textureStore(outCF, p, vec4u(pack2x16float(ycc.yz), 0u, 0u, 0u));
        cs += ycc.yz;
        n += 1.0;
      }
    }
  }
  textureStore(outCH, gid.xy, vec4f(cs / n, 0.0, 1.0));
}

// ---- warp: one thread per 2x2 output block. Output: camera-domain R'G'B' into outTex (rgba8unorm or, via warp.ts,
//      bgra8unorm: the canvas or an intermediate copied into it), or with P.dbg == 1 the float luma Y' into
//      coordOut[y*W + x] (tests). Invalid (outside the source) -> black.
fn emit(p: vec2u, rgb: vec3f, y: f32) {
  if (P.dbg == 1u) { coordOut[p.y * P.dstW + p.x] = y; }
  else { textureStore(outTex, p, vec4f(rgb, 1.0)); }
}

@compute @workgroup_size(8, 8)
fn warp(@builtin(global_invocation_id) gid: vec3u) {
  let p0 = 2u * gid.xy;
  if (p0.x >= P.dstW || p0.y >= P.dstH) { return; }
  let scc = source_coord(vec2f(f32(p0.x), f32(p0.y) + 0.5));      // the block's chroma site
  var cc = vec2f(0.0);
  if (scc.z > 0.5) {
    if (cstateR.planar == 1u) { cc = sample2(texCH, vec2f(scc.x * 0.5, (scc.y - 0.5) * 0.5)); }
    else { cc = sample_cf(scc.xy); }
  }
  for (var dy = 0u; dy < 2u; dy++) {
    for (var dx = 0u; dx < 2u; dx++) {
      let p = p0 + vec2u(dx, dy);
      if (p.x < P.dstW && p.y < P.dstH) {
        let sc = source_coord(vec2f(p));
        if (sc.z > 0.5) {
          let y = sample1(texY, sc.xy);
          emit(p, to_rgb(vec3f(y, cc)), y);
        } else {
          emit(p, vec3f(0.0), 0.0);
        }
      }
    }
  }
}

// ---- debug: source-coordinate map, out[(y*W+x)*3 + {0,1,2}] = (S.x, S.y, valid) for dispatch-grid pixel (x,y);
//      full-res output X = (p + 0.5)/outS - 0.5 (pixel-centre convention, same as warp.metal::sp_coord_map)
@compute @workgroup_size(8, 8)
fn coord_map(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.dstW || gid.y >= P.dstH) { return; }
  let X = (vec2f(gid.xy) + 0.5) / vec2f(P.outSx, P.outSy) - 0.5;
  let sc = source_coord(X);
  let o = 3u * (gid.y * P.dstW + gid.x);
  coordOut[o] = sc.x; coordOut[o + 1u] = sc.y; coordOut[o + 2u] = sc.z;
}
