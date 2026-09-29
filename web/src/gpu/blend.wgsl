// Stillpoint web — linear-light frame blender (owner: GPU agent). See blend.ts.
//
// out = enc( sum_i w_i * dec(in_i) ),  dec/enc = the pictures' transfer (camera OETF, e.g. BT.709) and its inverse,
// so averaging neighbouring stabilized frames integrates LIGHT like a longer physical shutter (a gamma-space average
// darkens highlights' motion trails). Inputs are R'G'B' pictures (Warper outputs: canvas VideoFrames through
// texture_external, or warpTo() textures); up to 3 taps per dispatch, more via a float accumulator buffer
// (acc_in: add the previous chunks; fin: 1 = encode into outTex, 0 = store the linear sum into acc).
// tr = 4 ('linear') is the identity: a single tap with weight 1 is copied exactly.

struct BP {
  w: vec4f,
  n: u32, tr: u32, accIn: u32, fin: u32,
  W: u32, H: u32, pad0: u32, pad1: u32,
};

@group(0) @binding(0) var<uniform> B: BP;
@group(0) @binding(1) var e0: texture_external;
@group(0) @binding(2) var e1: texture_external;
@group(0) @binding(3) var e2: texture_external;
@group(0) @binding(4) var t0: texture_2d<f32>;
@group(0) @binding(5) var t1: texture_2d<f32>;
@group(0) @binding(6) var t2: texture_2d<f32>;
@group(0) @binding(7) var<storage, read_write> acc: array<vec4f>;
@group(0) @binding(8) var outTex: texture_storage_2d<rgba8unorm, write>;

// transfers (index = blend.ts BLEND_TRANSFERS): 0 bt709 (camera OETF), 1 srgb, 2 gamma 2.2, 3 gamma 2.4,
// 4 linear (identity)
fn dec1(e0_: f32, tr: u32) -> f32 {
  let e = clamp(e0_, 0.0, 1.0);
  switch tr {
    case 0u: { return select(pow((e + 0.099) / 1.099, 1.0 / 0.45), e / 4.5, e < 0.081); }
    case 1u: { return select(pow((e + 0.055) / 1.055, 2.4), e / 12.92, e <= 0.04045); }
    case 2u: { return pow(e, 2.2); }
    case 3u: { return pow(e, 2.4); }
    default: { return e; }
  }
}
fn enc1(l0: f32, tr: u32) -> f32 {
  let l = max(l0, 0.0);
  switch tr {
    case 0u: { return select(1.099 * pow(l, 0.45) - 0.099, 4.5 * l, l < 0.018); }
    case 1u: { return select(1.055 * pow(l, 1.0 / 2.4) - 0.055, 12.92 * l, l <= 0.0031308); }
    case 2u: { return pow(l, 1.0 / 2.2); }
    case 3u: { return pow(l, 1.0 / 2.4); }
    default: { return l; }
  }
}
fn dec(e: vec3f) -> vec3f {
  if (B.tr == 4u) { return e; }
  return vec3f(dec1(e.x, B.tr), dec1(e.y, B.tr), dec1(e.z, B.tr));
}
fn enc(l: vec3f) -> vec3f {
  if (B.tr == 4u) { return l; }
  return vec3f(enc1(l.x, B.tr), enc1(l.y, B.tr), enc1(l.z, B.tr));
}

fn start(i: u32) -> vec3f {
  if (B.accIn == 1u) { return acc[i].xyz; }
  return vec3f(0.0);
}
fn finish(p: vec2u, i: u32, a: vec3f) {
  if (B.fin == 1u) { textureStore(outTex, p, vec4f(enc(a), 1.0)); }
  else { acc[i] = vec4f(a, 0.0); }
}

@compute @workgroup_size(8, 8)
fn blend_ext(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= B.W || gid.y >= B.H) { return; }
  let i = gid.y * B.W + gid.x;
  var a = start(i);
  a += B.w.x * dec(textureLoad(e0, gid.xy).rgb);
  if (B.n > 1u) { a += B.w.y * dec(textureLoad(e1, gid.xy).rgb); }
  if (B.n > 2u) { a += B.w.z * dec(textureLoad(e2, gid.xy).rgb); }
  finish(gid.xy, i, a);
}

@compute @workgroup_size(8, 8)
fn blend_tex(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= B.W || gid.y >= B.H) { return; }
  let i = gid.y * B.W + gid.x;
  let p = vec2i(gid.xy);
  var a = start(i);
  a += B.w.x * dec(textureLoad(t0, p, 0).rgb);
  if (B.n > 1u) { a += B.w.y * dec(textureLoad(t1, p, 0).rgb); }
  if (B.n > 2u) { a += B.w.z * dec(textureLoad(t2, p, 0).rgb); }
  finish(gid.xy, i, a);
}
