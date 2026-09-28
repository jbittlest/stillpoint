// Stillpoint — THE warp kernel (ENGINE_SPEC.md §1, §3).  Owner: WP-B.
//
// Loaded at runtime by the Swift renderer (app/renderer, MTLDevice.makeLibrary(source:)) and by the Python
// preview renderer (engine/stillpoint/render_ref.py, torch.mps.compile_shader; Python prepends
// `#define SP_NO_TEXTURE_KERNELS 1`). Both use the SAME geometry function sp_source_coord().
//
// Geometry (per output pixel, fp32):
//   r_v = ((X - out_cx)/out_fx, (Y - out_cy)/out_fx, 1)            X,Y = FULL-RES output pixel (centres at ints)
//   3 evaluations of  { M(y) = lerp of the plan's row matrices at source row y;  r_c = M(y) r_v;
//                       (u,v) = lens.project(r_c) }
//   y_0 = (src_h-1)/2 (centre row), y_1 = v_0 (fixed point), y_2 = secant/Aitken step on g(y) = v(y) - y using
//   (y_0,g_0),(y_1,g_1) (falls back to the fixed-point step y_2 = v_1 when the local contraction |c| > 0.9 or
//   degenerate). For locally linear v(y) the secant step is exact, so 3 evaluations converge far beyond the
//   plain fixed point (which leaves c^3 * |y*-y_0| ~ 0.3 px at 600 deg/s pitch). The final (u,v) is sampled.
//   Row matrix j sits at source row y_j = j*(src_h-1)/(n_rows-1); rows outside [0, src_h-1] clamp.
//   Valid iff r_c.z > 0 and -0.5 <= u <= W-0.5, -0.5 <= v <= H-0.5 (full-res source); invalid -> black.
// Sampling: Lanczos-3 (6x6, per-axis normalised), Catmull-Rom (a=-0.5, exact position) or bilinear; taps clamp
//   to the edge. 4:2:0 left-sited chroma: chroma sample (i,j) = luma position (2i, 2j+0.5); source luma coord S ->
//   chroma texel coord (S.x/2, (S.y-0.5)/2).
// Values: textures are unorm (8-bit r8/rg8 or 16-bit r16/rg16 MSB-aligned 10-bit). value * in_scale gives the
//   16-bit-normalised level (in_scale = 255*256/65535 for 8-bit sources, i.e. code10 = code8*4), then the output is
//   quantised to 10-bit codes stored MSB-aligned: code10 = clamp(round(v*65535/64), 0, 1023).
// Scaling (preview / debug maps): dispatch-grid pixel p -> full-res output X = (p+0.5)/out_s - 0.5;
//   full-res source S -> buffer pixel s = src_s*(S+0.5) - 0.5.  (pixel-centre convention both ways)

#include <metal_stdlib>
using namespace metal;

// ---- parameter block: float array, identical layout in Swift (sprender) and Python (render_ref.PARAM_*)
#define P_OUT_FX     0
#define P_OUT_CX     1
#define P_OUT_CY     2
#define P_LENS_MODEL 3   // 0 pinhole, 1 kb4
#define P_FX         4
#define P_FY         5
#define P_CX         6
#define P_CY         7
#define P_K1         8
#define P_K2         9
#define P_K3        10
#define P_K4        11
#define P_SRC_W     12   // full-res source size (luma)
#define P_SRC_H     13
#define P_N_ROWS    14
#define P_ITERS     15   // evaluations (default 3)
#define P_OUT_SX    16   // dispatch grid / full-res output scale (1 for the final render)
#define P_OUT_SY    17
#define P_SRC_SX    18   // sampled buffer / full-res source scale (1 for the final render)
#define P_SRC_SY    19
#define P_KERNEL    20   // 0 lanczos3, 1 catmull-rom, 2 bilinear
#define P_IN_SCALE  21
#define P_BLACK_Y   22   // normalised 16-bit level written for invalid luma
#define P_BLACK_C   23   // normalised 16-bit level written for invalid chroma
#define P_DST_W     24   // dispatch grid size (luma / preview / map)
#define P_DST_H     25
#define P_BUF_W     26   // preview: size of the (reduced) gray source buffer
#define P_BUF_H     27
#define P_COUNT     32

// ------------------------------------------------------------------------------------------ geometry
static inline float3 sp_rows_apply(device const float* M, float nr, float srch, float y, float3 rv) {
    float g = clamp(y * (nr - 1.0f) / (srch - 1.0f), 0.0f, nr - 1.0f);
    int j0 = min(int(g), int(nr) - 2);
    float f = g - float(j0);
    device const float* A = M + 9 * j0;
    device const float* B = A + 9;
    float3 ra = float3(A[0] * rv.x + A[1] * rv.y + A[2] * rv.z,
                       A[3] * rv.x + A[4] * rv.y + A[5] * rv.z,
                       A[6] * rv.x + A[7] * rv.y + A[8] * rv.z);
    float3 rb = float3(B[0] * rv.x + B[1] * rv.y + B[2] * rv.z,
                       B[3] * rv.x + B[4] * rv.y + B[5] * rv.z,
                       B[6] * rv.x + B[7] * rv.y + B[8] * rv.z);
    return ra + (rb - ra) * f;
}

static inline float2 sp_project(device const float* P, float3 r) {
    if (P[P_LENS_MODEL] < 0.5f) {
        float z = r.z > 0.0f ? max(r.z, 1e-12f) : min(r.z, -1e-12f);
        return float2(P[P_FX] * r.x / z + P[P_CX], P[P_FY] * r.y / z + P[P_CY]);
    }
    float rxy = precise::sqrt(r.x * r.x + r.y * r.y);
    float th = precise::atan2(rxy, r.z);
    float t2 = th * th;
    float thd = th * (1.0f + t2 * (P[P_K1] + t2 * (P[P_K2] + t2 * (P[P_K3] + t2 * P[P_K4]))));
    float s = rxy < 1e-12f ? 1.0f / max(r.z, 1e-12f) : thd / rxy;
    return float2(P[P_FX] * r.x * s + P[P_CX], P[P_FY] * r.y * s + P[P_CY]);
}

// full-res output pixel -> full-res source luma pixel; ok = ray in front and inside the source image.
static inline float2 sp_source_coord(device const float* P, device const float* M, float2 X, thread bool& ok) {
    float3 rv = float3((X.x - P[P_OUT_CX]) / P[P_OUT_FX], (X.y - P[P_OUT_CY]) / P[P_OUT_FX], 1.0f);
    float srch = P[P_SRC_H], nr = P[P_N_ROWS];
    int iters = max(int(P[P_ITERS]), 1);
    float y = 0.5f * (srch - 1.0f), ya = 0.0f, ga = 0.0f;
    float2 uv = float2(0.0f);
    float z = 1.0f;
    for (int e = 0; e < iters; e++) {
        float3 rc = sp_rows_apply(M, nr, srch, y, rv);
        uv = sp_project(P, rc);
        z = rc.z;
        float g = uv.y - y;
        float yn = uv.y;
        if (e >= 1) {
            float dy = y - ya;
            if (dy != 0.0f) {
                float s = (g - ga) / dy;               // slope of g = c - 1
                if (s <= -0.1f && s >= -2.0f) yn = y - g / s;
            }
        }
        ya = y; ga = g; y = yn;
    }
    ok = (z > 0.0f) && (uv.x >= -0.5f) && (uv.y >= -0.5f) &&
         (uv.x <= P[P_SRC_W] - 0.5f) && (uv.y <= P[P_SRC_H] - 0.5f);
    return uv;
}

// ------------------------------------------------------------------------------------------ resampling
// cos(pi*i/3), sin(pi*i/3) for taps i = -2..3
constant float SP_C3[6] = {-0.5f, 0.5f, 1.0f, 0.5f, -0.5f, -1.0f};
constant float SP_S3[6] = {-0.86602540378444f, -0.86602540378444f, 0.0f, 0.86602540378444f, 0.86602540378444f, 0.0f};
static inline float sp_lanczos3_w(float x, float s1, float s3, float c3, int t) {
    // tap t = i + 2, x = f - i; sin(pi x) = (-1)^i sin(pi f); sin(pi x/3) = sin(pi f/3)cos(pi i/3) - cos(pi f/3)sin(pi i/3)
    if (fabs(x) < 1e-4f) return 1.0f;
    const float PI = 3.14159265358979f;
    float a = (t & 1) ? -s1 : s1;                       // i = t-2 has the parity of t
    float b = s3 * SP_C3[t] - c3 * SP_S3[t];
    return 3.0f * a * b / (PI * PI * x * x);
}

// N taps starting at floor(s) + off; weights normalised.
template <int N>
static inline void sp_weights(float f, thread float* w) {
    if (N == 6) {
        const float PI = 3.14159265358979f;
        float s1 = precise::sin(PI * f), s3 = precise::sin(PI * f / 3.0f), c3 = precise::cos(PI * f / 3.0f);
        float sum = 0.0f;
        for (int t = 0; t < 6; t++) { w[t] = sp_lanczos3_w(f - float(t - 2), s1, s3, c3, t); sum += w[t]; }
        for (int t = 0; t < 6; t++) w[t] /= sum;
    } else if (N == 4) {
        float t2 = f * f, t3 = t2 * f;
        w[0] = -0.5f * t3 + t2 - 0.5f * f;
        w[1] = 1.5f * t3 - 2.5f * t2 + 1.0f;
        w[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * f;
        w[3] = 0.5f * t3 - 0.5f * t2;
    } else {
        w[0] = 1.0f - f; w[1] = f;
    }
}

#ifndef SP_NO_TEXTURE_KERNELS
template <int N>
static inline float4 sp_sample_tex_n(texture2d<float, access::read> t, float2 s, int2 mx) {
    const int off = (N == 6) ? -2 : ((N == 4) ? -1 : 0);
    float2 fl = floor(s);
    float2 f = s - fl;
    int2 b = int2(fl) + off;
    float wx[N], wy[N];
    sp_weights<N>(f.x, wx);
    sp_weights<N>(f.y, wy);
    float4 acc = float4(0.0f);
    for (int j = 0; j < N; j++) {
        uint yy = uint(clamp(b.y + j, 0, mx.y));
        float4 row = float4(0.0f);
        for (int i = 0; i < N; i++) row += wx[i] * t.read(uint2(uint(clamp(b.x + i, 0, mx.x)), yy));
        acc += wy[j] * row;
    }
    return acc;
}
static inline float4 sp_sample_tex(texture2d<float, access::read> t, float2 s, int2 mx, int kern) {
    if (kern == 0) return sp_sample_tex_n<6>(t, s, mx);
    if (kern == 1) return sp_sample_tex_n<4>(t, s, mx);
    return sp_sample_tex_n<2>(t, s, mx);
}
static inline float4 sp_q10(float4 v) {
    return clamp(round(v * (65535.0f / 64.0f)), 0.0f, 1023.0f) * (64.0f / 65535.0f);
}

// Final render, luma plane. dst is the full-res output luma (r16Unorm), src the source luma (r8/r16Unorm).
kernel void sp_warp_luma(texture2d<float, access::read> src [[texture(0)]],
                         texture2d<float, access::write> dst [[texture(1)]],
                         device const float* P [[buffer(0)]],
                         device const float* M [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    bool ok;
    float2 S = sp_source_coord(P, M, float2(gid), ok);
    float4 v;
    if (ok) {
        int2 mx = int2(src.get_width(), src.get_height()) - 1;
        v = sp_q10(sp_sample_tex(src, S, mx, int(P[P_KERNEL])) * P[P_IN_SCALE]);
    } else {
        v = float4(P[P_BLACK_Y]);
    }
    dst.write(v, gid);
}

// Final render, interleaved CbCr plane (rg16Unorm out; rg8/rg16Unorm in).
kernel void sp_warp_chroma(texture2d<float, access::read> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           device const float* P [[buffer(0)]],
                           device const float* M [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    bool ok;
    float2 S = sp_source_coord(P, M, float2(2.0f * float(gid.x), 2.0f * float(gid.y) + 0.5f), ok);
    float4 v;
    if (ok) {
        int2 mx = int2(src.get_width(), src.get_height()) - 1;
        float2 c = float2(S.x * 0.5f, (S.y - 0.5f) * 0.5f);
        v = sp_q10(sp_sample_tex(src, c, mx, int(P[P_KERNEL])) * P[P_IN_SCALE]);
    } else {
        v = float4(P[P_BLACK_C]);
    }
    dst.write(v, gid);
}
#endif  // SP_NO_TEXTURE_KERNELS

// ------------------------------------------------------------------------------------------ buffer kernels
// (callable from Python via torch.mps.compile_shader and from Swift for debug dumps)

// Source-coordinate map: out[(y*W+x)*3 + {0,1,2}] = (S.x, S.y, valid) for dispatch-grid pixel (x,y).
kernel void sp_coord_map(device float* out, device const float* P, device const float* M,
                         uint2 gid [[thread_position_in_grid]]) {
    uint W = uint(P[P_DST_W]), H = uint(P[P_DST_H]);
    if (gid.x >= W || gid.y >= H) return;
    float2 X = (float2(gid) + 0.5f) / float2(P[P_OUT_SX], P[P_OUT_SY]) - 0.5f;
    bool ok;
    float2 S = sp_source_coord(P, M, X, ok);
    uint o = 3u * (gid.y * W + gid.x);
    out[o] = S.x; out[o + 1] = S.y; out[o + 2] = ok ? 1.0f : 0.0f;
}

template <int N>
static inline float sp_sample_buf_n(device const float* src, int bw, int bh, float2 s) {
    const int off = (N == 6) ? -2 : ((N == 4) ? -1 : 0);
    float2 fl = floor(s);
    float2 f = s - fl;
    int2 b = int2(fl) + off;
    float wx[N], wy[N];
    sp_weights<N>(f.x, wx);
    sp_weights<N>(f.y, wy);
    float acc = 0.0f;
    for (int j = 0; j < N; j++) {
        int yy = clamp(b.y + j, 0, bh - 1);
        float row = 0.0f;
        for (int i = 0; i < N; i++) row += wx[i] * src[yy * bw + clamp(b.x + i, 0, bw - 1)];
        acc += wy[j] * row;
    }
    return acc;
}

// Preview: gray float source buffer (P_BUF_W x P_BUF_H, = src_s * full-res) -> gray float output
// (P_DST_W x P_DST_H, = out_s * full-res output). Invalid -> 0.  valid[] gets 1/0.
kernel void sp_preview_gray(device const float* src, device float* dst, device float* valid,
                            device const float* P, device const float* M,
                            uint2 gid [[thread_position_in_grid]]) {
    uint W = uint(P[P_DST_W]), H = uint(P[P_DST_H]);
    if (gid.x >= W || gid.y >= H) return;
    float2 X = (float2(gid) + 0.5f) / float2(P[P_OUT_SX], P[P_OUT_SY]) - 0.5f;
    bool ok;
    float2 S = sp_source_coord(P, M, X, ok);
    float v = 0.0f;
    if (ok) {
        float2 s = float2(P[P_SRC_SX], P[P_SRC_SY]) * (S + 0.5f) - 0.5f;
        int bw = int(P[P_BUF_W]), bh = int(P[P_BUF_H]);
        int k = int(P[P_KERNEL]);
        v = (k == 0) ? sp_sample_buf_n<6>(src, bw, bh, s)
          : (k == 1) ? sp_sample_buf_n<4>(src, bw, bh, s) : sp_sample_buf_n<2>(src, bw, bh, s);
    }
    dst[gid.y * W + gid.x] = v;
    valid[gid.y * W + gid.x] = ok ? 1.0f : 0.0f;
}
