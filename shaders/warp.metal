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
// Mesh residual (optional, *_mesh kernels only; engine/stillpoint/mesh.py, .spplan mesh block): before the rotation
//   mapping, the full-res output pixel X is displaced by a bilinearly interpolated per-frame offset field:
//   X' = X + D(X). D is given on a (P_MESH_NY x P_MESH_NX) vertex grid, vertex (i,j) at X = i*(out_w-1)/(nx-1),
//   Y = j*(out_h-1)/(ny-1) (out_w-1 = 2*out_cx), float2 (dx,dy) full-res output px, row-major; clamped outside the
//   grid. P_MESH_NX < 2 = no mesh. The non-mesh kernels (sp_warp_luma, ...) are unchanged (Stillpoint.app uses them).

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
#define P_N_TAPS    28   // synthetic shutter (the *_blur kernels, sprender only): number of taps in the tap buffer
                         // (28, 29 are also used by Stillpoint.app's appended split kernels: SPAPP_SPLIT_X, SPAPP_RAW)
#define P_MESH_NX   30   // mesh vertex columns (*_mesh kernels; < 2 = no mesh)
#define P_MESH_NY   31   // mesh vertex rows
#define P_COUNT     32

// Synthetic shutter (engine/stillpoint/synth_blur.py, .spblur sidecar): the *_blur kernels average n taps of the SAME
// source frame warped at virtual orientations V(t_k + s_i) along the virtual path. Tap i = 10 floats: D_i (3x3,
// row-major) = R(V_k)^T R(V(t_k + s_i)), then its weight. Geometry per tap: r_v' = D_i r_v, then the unchanged
// fixed-point mapping (the row matrices M(y) = R_cam(t_row)^T R(V_k) become M(y) D_i = R_cam^T R(V(t_k+s_i))).
// Taps are averaged in the coded (gamma) domain; taps that fall outside the source are dropped per pixel and the
// weights renormalised (the Python side shortens the shutter so the border stays inside; this is the safety net).

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

static inline float3 sp_out_ray(device const float* P, float2 X) {
    return float3((X.x - P[P_OUT_CX]) / P[P_OUT_FX], (X.y - P[P_OUT_CY]) / P[P_OUT_FX], 1.0f);
}

static inline float3 sp_tap_ray(device const float* D, float3 rv) {
    return float3(D[0] * rv.x + D[1] * rv.y + D[2] * rv.z,
                  D[3] * rv.x + D[4] * rv.y + D[5] * rv.z,
                  D[6] * rv.x + D[7] * rv.y + D[8] * rv.z);
}

// output ray (virtual camera frame) -> full-res source luma pixel; zo = z of the final source ray (> 0: in front).
static inline float2 sp_source_coord_rvz(device const float* P, device const float* M, float3 rv, thread float& zo) {
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
    zo = z;
    return uv;
}

static inline bool sp_in_source(device const float* P, float2 uv, float z) {
    return (z > 0.0f) && (uv.x >= -0.5f) && (uv.y >= -0.5f) &&
           (uv.x <= P[P_SRC_W] - 0.5f) && (uv.y <= P[P_SRC_H] - 0.5f);
}

// full-res output pixel -> full-res source luma pixel; zo = z of the final source ray (> 0: in front).
static inline float2 sp_source_coord_z(device const float* P, device const float* M, float2 X, thread float& zo) {
    return sp_source_coord_rvz(P, M, sp_out_ray(P, X), zo);
}

// output ray (virtual camera frame) -> full-res source luma pixel; ok = ray in front and inside the source image.
static inline float2 sp_source_coord_rv(device const float* P, device const float* M, float3 rv, thread bool& ok) {
    float z;
    float2 uv = sp_source_coord_rvz(P, M, rv, z);
    ok = sp_in_source(P, uv, z);
    return uv;
}

// full-res output pixel -> full-res source luma pixel; ok = ray in front and inside the source image.
static inline float2 sp_source_coord(device const float* P, device const float* M, float2 X, thread bool& ok) {
    return sp_source_coord_rv(P, M, sp_out_ray(P, X), ok);
}

// mesh residual offset D(X) at full-res output pixel X (bilinear on the vertex grid, clamped).
static inline float2 sp_mesh_offset(device const float* P, device const float* D, float2 X) {
    int nx = int(P[P_MESH_NX]), ny = int(P[P_MESH_NY]);
    if (nx < 2 || ny < 2) return float2(0.0f);
    float gx = clamp(X.x * float(nx - 1) / max(2.0f * P[P_OUT_CX], 1.0f), 0.0f, float(nx - 1));
    float gy = clamp(X.y * float(ny - 1) / max(2.0f * P[P_OUT_CY], 1.0f), 0.0f, float(ny - 1));
    int i0 = min(int(gx), nx - 2), j0 = min(int(gy), ny - 2);
    float fx = gx - float(i0), fy = gy - float(j0);
    device const float* a = D + 2 * (j0 * nx + i0);
    device const float* b = a + 2 * nx;
    float2 v00 = float2(a[0], a[1]), v10 = float2(a[2], a[3]);
    float2 v01 = float2(b[0], b[1]), v11 = float2(b[2], b[3]);
    float2 t = v00 + (v10 - v00) * fx;
    float2 u = v01 + (v11 - v01) * fx;
    return t + (u - t) * fy;
}

// sp_source_coord after the mesh residual displacement of the output pixel.
static inline float2 sp_source_coord_mesh(device const float* P, device const float* M, device const float* D,
                                          float2 X, thread bool& ok) {
    return sp_source_coord(P, M, X + sp_mesh_offset(P, D, X), ok);
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


// ------------------------------------------------------------------------------------------ full-frame fill
// Border synthesis from neighbouring source frames (engine/stillpoint/fill.py; the .spplan FILL section).
// Per output pixel X of record k:
//   main: S0 = sp_source_coord(M_k, X), d0 = signed distance inside the source [source px] (<0 outside).
//   d0 >= FEATHER_MAIN, or no sources and d0 >= 0: exactly the plain kernel; no sources and d0 < 0: soft edge extension.
//   else, for each source i (composite rows FM_i = M_j . G_kj, record offset DOFF_i):
//     X_i = X + DOFF_i * mesh(X)  (parallax velocity grid, output px / frame),  S_i = sp_source_coord(FM_i, X_i),
//     w_i = smoothstep(0, FEATHER_NB, d_i) * WEIGHT_i  (0 outside source j), luma value gained around BLACK;
//     reference = the max-w candidate; w_i *= exp(-|v_i - v_ref|^2 / SIGMA^2)  (per-pixel consistency);
//     fill = (sum w_i v_i + FB_W * v_fb) / (sum w_i + FB_W),  v_fb = main sample if d0 >= 0, else the source edge
//     sampled with a 3x3 bilinear blur of radius 1 + FB_BLUR * overshoot (soft edge extension);
//     out = mix(fill, main, smoothstep(0, FEATHER_MAIN, d0))  (0 for d0 < 0).
#define F_N_SRC         0
#define F_FEATHER_MAIN  1
#define F_FEATHER_NB    2
#define F_SIGMA         3
#define F_FB_W          4
#define F_MESH_W        5
#define F_MESH_H        6
#define F_BLACK         7    // texture-normalised black level (gain pivot, luma)
#define F_WEIGHT        8    // 8..11
#define F_GAIN         12    // 12..15
#define F_DOFF         16    // 16..19 record offset (source - record)
#define F_FB_BLUR      20
#define F_OUT_W        21    // full-res output size (mesh lookup)
#define F_OUT_H        22
#define F_PAR          23    // parallax tolerance [full-res output px]: source weight x exp(-(|doff| |v(X)| / PAR)^2); 0 = off
#define F_COUNT        32
#define SP_MAX_FILL     4

static inline float sp_inside(device const float* P, float2 S, float z) {
    if (!(z > 0.0f)) return -1e30f;
    return min(min(S.x + 0.5f, P[P_SRC_W] - 0.5f - S.x), min(S.y + 0.5f, P[P_SRC_H] - 0.5f - S.y));
}

static inline float sp_smooth01(float e1, float x) {
    float t = clamp(x / max(e1, 1e-6f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static inline float2 sp_mesh_disp(device const float* FP, device const float* MS, float2 X) {
    int mw = int(FP[F_MESH_W]), mh = int(FP[F_MESH_H]);
    if (mw <= 0 || mh <= 0) return float2(0.0f);
    float fx = clamp((X.x + 0.5f) * float(mw) / FP[F_OUT_W] - 0.5f, 0.0f, float(mw - 1));
    float fy = clamp((X.y + 0.5f) * float(mh) / FP[F_OUT_H] - 0.5f, 0.0f, float(mh - 1));
    int x0 = min(int(fx), max(mw - 2, 0)), y0 = min(int(fy), max(mh - 2, 0));
    int x1 = min(x0 + 1, mw - 1), y1 = min(y0 + 1, mh - 1);
    float ax = fx - float(x0), ay = fy - float(y0);
    float2 a = float2(MS[2 * (y0 * mw + x0)], MS[2 * (y0 * mw + x0) + 1]);
    float2 b = float2(MS[2 * (y0 * mw + x1)], MS[2 * (y0 * mw + x1) + 1]);
    float2 c = float2(MS[2 * (y1 * mw + x0)], MS[2 * (y1 * mw + x0) + 1]);
    float2 d = float2(MS[2 * (y1 * mw + x1)], MS[2 * (y1 * mw + x1) + 1]);
    float2 top = a * (1.0f - ax) + b * ax;
    float2 bot = c * (1.0f - ax) + d * ax;
    return top * (1.0f - ay) + bot * ay;
}

// parallax-aware source weight: the linear parallax model (X + doff * v) is only as good as v; where the predicted
// parallax displacement |doff| |v| is large (near ground in fast low flight) a far source is likely misaligned, so its
// weight decays (Gaussian in the displacement) -> nearer sources, or the soft edge extension, take over.
static inline float sp_par_w(device const float* FP, float doff, float2 disp) {
    float p = FP[F_PAR];
    if (!(p > 0.0f)) return 1.0f;
    float e = fabs(doff) * length(disp) / p;
    return exp(-e * e);
}

// candidate combination (luma: .x; chroma: .xy)
static inline float4 sp_fill_combine(thread const float4* v, thread const float* w, int n, float4 vfb,
                                     device const float* FP) {
    int best = -1;
    float bw = 0.0f;
    for (int i = 0; i < n; i++) if (w[i] > bw) { bw = w[i]; best = i; }
    float sig = FP[F_SIGMA];
    float fbw = max(FP[F_FB_W], 1e-6f);
    float4 acc = vfb * fbw;
    float ws = fbw;
    for (int i = 0; i < n; i++) {
        float wi = w[i];
        if (wi <= 0.0f) continue;
        if (sig > 0.0f && best >= 0 && i != best) {
            float2 d = (v[i] - v[best]).xy;
            wi *= exp(-dot(d, d) / (sig * sig));
        }
        acc += wi * v[i];
        ws += wi;
    }
    return acc / ws;
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
// ---- fill kernels (texture): the plain kernels plus up to 4 neighbour source textures (see the fill block above)
static inline float4 sp_sample_nb(int i, texture2d<float, access::read> n0, texture2d<float, access::read> n1,
                                  texture2d<float, access::read> n2, texture2d<float, access::read> n3,
                                  float2 s, int2 mx, int kern) {
    if (i == 0) return sp_sample_tex(n0, s, mx, kern);
    if (i == 1) return sp_sample_tex(n1, s, mx, kern);
    if (i == 2) return sp_sample_tex(n2, s, mx, kern);
    return sp_sample_tex(n3, s, mx, kern);
}

// soft edge extension: the texture's edge nearest to s, 3x3 bilinear blur of radius 1 + blur * overshoot
static inline float4 sp_edge_ext(texture2d<float, access::read> t, float2 s, int2 mx, float blur) {
    float2 c = clamp(s, float2(0.0f), float2(mx));
    float r = 1.0f + blur * length(s - c);
    float4 acc = float4(0.0f);
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) acc += sp_sample_tex_n<2>(t, c + r * float2(float(i), float(j)), mx);
    return acc * (1.0f / 9.0f);
}

kernel void sp_warp_luma_fill(texture2d<float, access::read> src [[texture(0)]],
                              texture2d<float, access::write> dst [[texture(1)]],
                              texture2d<float, access::read> n0 [[texture(2)]],
                              texture2d<float, access::read> n1 [[texture(3)]],
                              texture2d<float, access::read> n2 [[texture(4)]],
                              texture2d<float, access::read> n3 [[texture(5)]],
                              device const float* P [[buffer(0)]],
                              device const float* M [[buffer(1)]],
                              device const float* FP [[buffer(2)]],
                              device const float* FM [[buffer(3)]],
                              device const float* MS [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 X = float2(gid);
    float z0;
    float2 S0 = sp_source_coord_z(P, M, X, z0);
    float d0 = sp_inside(P, S0, z0);
    int2 mx = int2(src.get_width(), src.get_height()) - 1;
    int kern = int(P[P_KERNEL]);
    int n = min(int(FP[F_N_SRC]), SP_MAX_FILL);
    float fm = FP[F_FEATHER_MAIN];
    float4 vm = (d0 >= 0.0f) ? sp_sample_tex(src, S0, mx, kern) : float4(0.0f);
    float4 v;
    if (d0 >= fm || (n <= 0 && d0 >= 0.0f)) {
        v = sp_q10(vm * P[P_IN_SCALE]);
    } else if (n <= 0) {                                   // outside, no source: soft edge extension (never black)
        v = sp_q10(sp_edge_ext(src, S0, mx, FP[F_FB_BLUR]) * P[P_IN_SCALE]);
    } else {
        int nr = int(P[P_N_ROWS]);
        float2 disp = sp_mesh_disp(FP, MS, X);
        float4 cv[SP_MAX_FILL];
        float cw[SP_MAX_FILL];
        for (int i = 0; i < n; i++) {
            float zi;
            float2 Si = sp_source_coord_z(P, FM + 9 * nr * i, X + FP[F_DOFF + i] * disp, zi);
            float di = sp_inside(P, Si, zi);
            cw[i] = 0.0f;
            cv[i] = float4(0.0f);
            if (di > 0.0f) {
                cw[i] = sp_smooth01(FP[F_FEATHER_NB], di) * FP[F_WEIGHT + i] * sp_par_w(FP, FP[F_DOFF + i], disp);
                float4 sv = sp_sample_nb(i, n0, n1, n2, n3, Si, mx, kern);
                sv.x = (sv.x - FP[F_BLACK]) * FP[F_GAIN + i] + FP[F_BLACK];
                cv[i] = sv;
            }
        }
        float4 vfb = (d0 >= 0.0f) ? vm : sp_edge_ext(src, S0, mx, FP[F_FB_BLUR]);
        float4 f = sp_fill_combine(cv, cw, n, vfb, FP);
        float a = (d0 >= 0.0f) ? sp_smooth01(fm, d0) : 0.0f;
        v = sp_q10((f + (vm - f) * a) * P[P_IN_SCALE]);
    }
    dst.write(v, gid);
}

kernel void sp_warp_chroma_fill(texture2d<float, access::read> src [[texture(0)]],
                                texture2d<float, access::write> dst [[texture(1)]],
                                texture2d<float, access::read> n0 [[texture(2)]],
                                texture2d<float, access::read> n1 [[texture(3)]],
                                texture2d<float, access::read> n2 [[texture(4)]],
                                texture2d<float, access::read> n3 [[texture(5)]],
                                device const float* P [[buffer(0)]],
                                device const float* M [[buffer(1)]],
                                device const float* FP [[buffer(2)]],
                                device const float* FM [[buffer(3)]],
                                device const float* MS [[buffer(4)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 X = float2(2.0f * float(gid.x), 2.0f * float(gid.y) + 0.5f);
    float z0;
    float2 S0 = sp_source_coord_z(P, M, X, z0);
    float d0 = sp_inside(P, S0, z0);
    int2 mx = int2(src.get_width(), src.get_height()) - 1;
    int kern = int(P[P_KERNEL]);
    int n = min(int(FP[F_N_SRC]), SP_MAX_FILL);
    float fm = FP[F_FEATHER_MAIN];
    float2 c0 = float2(S0.x * 0.5f, (S0.y - 0.5f) * 0.5f);
    float4 vm = (d0 >= 0.0f) ? sp_sample_tex(src, c0, mx, kern) : float4(0.0f);
    float4 v;
    if (d0 >= fm || (n <= 0 && d0 >= 0.0f)) {
        v = sp_q10(vm * P[P_IN_SCALE]);
    } else if (n <= 0) {
        v = sp_q10(sp_edge_ext(src, c0, mx, FP[F_FB_BLUR]) * P[P_IN_SCALE]);
    } else {
        int nr = int(P[P_N_ROWS]);
        float2 disp = sp_mesh_disp(FP, MS, X);
        float4 cv[SP_MAX_FILL];
        float cw[SP_MAX_FILL];
        for (int i = 0; i < n; i++) {
            float zi;
            float2 Si = sp_source_coord_z(P, FM + 9 * nr * i, X + FP[F_DOFF + i] * disp, zi);
            float di = sp_inside(P, Si, zi);
            cw[i] = 0.0f;
            cv[i] = float4(0.0f);
            if (di > 0.0f) {
                cw[i] = sp_smooth01(FP[F_FEATHER_NB], di) * FP[F_WEIGHT + i] * sp_par_w(FP, FP[F_DOFF + i], disp);
                cv[i] = sp_sample_nb(i, n0, n1, n2, n3, float2(Si.x * 0.5f, (Si.y - 0.5f) * 0.5f), mx, kern);
            }
        }
        float4 vfb = (d0 >= 0.0f) ? vm : sp_edge_ext(src, c0, mx, FP[F_FB_BLUR]);
        float4 f = sp_fill_combine(cv, cw, n, vfb, FP);
        float a = (d0 >= 0.0f) ? sp_smooth01(fm, d0) : 0.0f;
        v = sp_q10((f + (vm - f) * a) * P[P_IN_SCALE]);
    }
    dst.write(v, gid);
}

// Synthetic shutter, luma: average of P[P_N_TAPS] taps (buffer T, see the top of the file).
kernel void sp_warp_luma_blur(texture2d<float, access::read> src [[texture(0)]],
                              texture2d<float, access::write> dst [[texture(1)]],
                              device const float* P [[buffer(0)]],
                              device const float* M [[buffer(1)]],
                              device const float* T [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float3 rv0 = sp_out_ray(P, float2(gid));
    int n = max(int(P[P_N_TAPS]), 1);
    int2 mx = int2(src.get_width(), src.get_height()) - 1;
    int kern = int(P[P_KERNEL]);
    float4 acc = float4(0.0f);
    float ws = 0.0f;
    for (int i = 0; i < n; i++) {
        device const float* D = T + 10 * i;
        bool ok;
        float2 S = sp_source_coord_rv(P, M, sp_tap_ray(D, rv0), ok);
        if (ok) { acc += D[9] * sp_sample_tex(src, S, mx, kern); ws += D[9]; }
    }
    float4 v = ws > 0.0f ? sp_q10(acc / ws * P[P_IN_SCALE]) : float4(P[P_BLACK_Y]);
    dst.write(v, gid);
}

// Synthetic shutter, interleaved CbCr.
kernel void sp_warp_chroma_blur(texture2d<float, access::read> src [[texture(0)]],
                                texture2d<float, access::write> dst [[texture(1)]],
                                device const float* P [[buffer(0)]],
                                device const float* M [[buffer(1)]],
                                device const float* T [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float3 rv0 = sp_out_ray(P, float2(2.0f * float(gid.x), 2.0f * float(gid.y) + 0.5f));
    int n = max(int(P[P_N_TAPS]), 1);
    int2 mx = int2(src.get_width(), src.get_height()) - 1;
    int kern = int(P[P_KERNEL]);
    float4 acc = float4(0.0f);
    float ws = 0.0f;
    for (int i = 0; i < n; i++) {
        device const float* D = T + 10 * i;
        bool ok;
        float2 S = sp_source_coord_rv(P, M, sp_tap_ray(D, rv0), ok);
        if (ok) { acc += D[9] * sp_sample_tex(src, float2(S.x * 0.5f, (S.y - 0.5f) * 0.5f), mx, kern); ws += D[9]; }
    }
    float4 v = ws > 0.0f ? sp_q10(acc / ws * P[P_IN_SCALE]) : float4(P[P_BLACK_C]);
    dst.write(v, gid);
}

// Mesh-residual versions of the final-render kernels (buffer(2) = this frame's mesh offsets).
kernel void sp_warp_luma_mesh(texture2d<float, access::read> src [[texture(0)]],
                              texture2d<float, access::write> dst [[texture(1)]],
                              device const float* P [[buffer(0)]],
                              device const float* M [[buffer(1)]],
                              device const float* D [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    bool ok;
    float2 S = sp_source_coord_mesh(P, M, D, float2(gid), ok);
    float4 v;
    if (ok) {
        int2 mx = int2(src.get_width(), src.get_height()) - 1;
        v = sp_q10(sp_sample_tex(src, S, mx, int(P[P_KERNEL])) * P[P_IN_SCALE]);
    } else {
        v = float4(P[P_BLACK_Y]);
    }
    dst.write(v, gid);
}

kernel void sp_warp_chroma_mesh(texture2d<float, access::read> src [[texture(0)]],
                                texture2d<float, access::write> dst [[texture(1)]],
                                device const float* P [[buffer(0)]],
                                device const float* M [[buffer(1)]],
                                device const float* D [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    bool ok;
    float2 S = sp_source_coord_mesh(P, M, D, float2(2.0f * float(gid.x), 2.0f * float(gid.y) + 0.5f), ok);
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


// Preview with fill: gray float sources stacked in one buffer: frame 0 = the record's own source, frame i+1 =
// fill source i (each P_BUF_W x P_BUF_H). Same logic as sp_warp_luma_fill (value units: the buffer's, e.g. 8-bit
// levels; F_SIGMA / F_BLACK in the same units). dst = value (0 where nothing), d0 = main inside distance
// [full-res source px], wn = sum of the neighbour weights after the consistency check.
static inline float sp_buf_edge_ext(device const float* src, int bw, int bh, float2 s, float blur, float sx) {
    float2 c = clamp(s, float2(0.0f), float2(float(bw - 1), float(bh - 1)));
    float r = sx + blur * length(s - c);            // = sx * (1 + blur * overshoot[full-res px])
    float acc = 0.0f;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) acc += sp_sample_buf_n<2>(src, bw, bh, c + r * float2(float(i), float(j)));
    return acc * (1.0f / 9.0f);
}

static inline float sp_buf_sample(device const float* src, int bw, int bh, float2 s, int k) {
    return (k == 0) ? sp_sample_buf_n<6>(src, bw, bh, s)
         : (k == 1) ? sp_sample_buf_n<4>(src, bw, bh, s) : sp_sample_buf_n<2>(src, bw, bh, s);
}

kernel void sp_preview_gray_fill(device const float* src, device float* dst, device float* d0out, device float* wn,
                                 device const float* P, device const float* M, device const float* FP,
                                 device const float* FM, device const float* MS,
                                 uint2 gid [[thread_position_in_grid]]) {
    uint W = uint(P[P_DST_W]), H = uint(P[P_DST_H]);
    if (gid.x >= W || gid.y >= H) return;
    float2 X = (float2(gid) + 0.5f) / float2(P[P_OUT_SX], P[P_OUT_SY]) - 0.5f;
    float2 ss = float2(P[P_SRC_SX], P[P_SRC_SY]);
    int bw = int(P[P_BUF_W]), bh = int(P[P_BUF_H]);
    int kern = int(P[P_KERNEL]);
    float z0;
    float2 S0 = sp_source_coord_z(P, M, X, z0);
    float d0 = sp_inside(P, S0, z0);
    int n = min(int(FP[F_N_SRC]), SP_MAX_FILL);
    float fm = FP[F_FEATHER_MAIN];
    float2 s0 = ss * (S0 + 0.5f) - 0.5f;
    float vm = (d0 >= 0.0f) ? sp_buf_sample(src, bw, bh, s0, kern) : 0.0f;
    float v, wsum = 0.0f;
    if (d0 >= fm || (n <= 0 && d0 >= 0.0f)) {
        v = vm;
    } else if (n <= 0) {
        v = sp_buf_edge_ext(src, bw, bh, s0, FP[F_FB_BLUR], ss.x);
    } else {
        int nr = int(P[P_N_ROWS]);
        float2 disp = sp_mesh_disp(FP, MS, X);
        float4 cv[SP_MAX_FILL];
        float cw[SP_MAX_FILL];
        for (int i = 0; i < n; i++) {
            float zi;
            float2 Si = sp_source_coord_z(P, FM + 9 * nr * i, X + FP[F_DOFF + i] * disp, zi);
            float di = sp_inside(P, Si, zi);
            cw[i] = 0.0f;
            cv[i] = float4(0.0f);
            if (di > 0.0f) {
                cw[i] = sp_smooth01(FP[F_FEATHER_NB], di) * FP[F_WEIGHT + i] * sp_par_w(FP, FP[F_DOFF + i], disp);
                float sv = sp_buf_sample(src + (i + 1) * bw * bh, bw, bh, ss * (Si + 0.5f) - 0.5f, kern);
                cv[i] = float4((sv - FP[F_BLACK]) * FP[F_GAIN + i] + FP[F_BLACK], 0.0f, 0.0f, 0.0f);
            }
        }
        // weights after the consistency check (for diagnostics)
        int best = -1;
        float bwt = 0.0f;
        for (int i = 0; i < n; i++) if (cw[i] > bwt) { bwt = cw[i]; best = i; }
        for (int i = 0; i < n; i++) {
            float wi = cw[i];
            if (wi > 0.0f && FP[F_SIGMA] > 0.0f && i != best) {
                float dd = cv[i].x - cv[best].x;
                wi *= exp(-dd * dd / (FP[F_SIGMA] * FP[F_SIGMA]));
            }
            wsum += wi;
        }
        float vfb = (d0 >= 0.0f) ? vm : sp_buf_edge_ext(src, bw, bh, s0, FP[F_FB_BLUR], ss.x);
        float4 f = sp_fill_combine(cv, cw, n, float4(vfb, 0.0f, 0.0f, 0.0f), FP);
        float a = (d0 >= 0.0f) ? sp_smooth01(fm, d0) : 0.0f;
        v = f.x + (vm - f.x) * a;
    }
    uint o = gid.y * W + gid.x;
    dst[o] = v;
    d0out[o] = max(d0, -1e6f);
    wn[o] = wsum;
}

// Preview with the synthetic shutter (same tap rules as sp_warp_luma_blur). valid[] = share of the tap weight inside.
kernel void sp_preview_gray_blur(device const float* src, device float* dst, device float* valid,
                                 device const float* P, device const float* M, device const float* T,
                                 uint2 gid [[thread_position_in_grid]]) {
    uint W = uint(P[P_DST_W]), H = uint(P[P_DST_H]);
    if (gid.x >= W || gid.y >= H) return;
    float2 X = (float2(gid) + 0.5f) / float2(P[P_OUT_SX], P[P_OUT_SY]) - 0.5f;
    float3 rv0 = sp_out_ray(P, X);
    int n = max(int(P[P_N_TAPS]), 1);
    int bw = int(P[P_BUF_W]), bh = int(P[P_BUF_H]);
    int k = int(P[P_KERNEL]);
    float acc = 0.0f, ws = 0.0f, wt = 0.0f;
    for (int i = 0; i < n; i++) {
        device const float* D = T + 10 * i;
        bool ok;
        float2 S = sp_source_coord_rv(P, M, sp_tap_ray(D, rv0), ok);
        wt += D[9];
        if (ok) {
            float2 s = float2(P[P_SRC_SX], P[P_SRC_SY]) * (S + 0.5f) - 0.5f;
            float v = (k == 0) ? sp_sample_buf_n<6>(src, bw, bh, s)
                    : (k == 1) ? sp_sample_buf_n<4>(src, bw, bh, s) : sp_sample_buf_n<2>(src, bw, bh, s);
            acc += D[9] * v;
            ws += D[9];
        }
    }
    dst[gid.y * W + gid.x] = ws > 0.0f ? acc / ws : 0.0f;
    valid[gid.y * W + gid.x] = wt > 0.0f ? ws / wt : 0.0f;
}

// ------------------------------------------------------------------------------------------ mesh buffer kernels
kernel void sp_coord_map_mesh(device float* out, device const float* P, device const float* M, device const float* D,
                              uint2 gid [[thread_position_in_grid]]) {
    uint W = uint(P[P_DST_W]), H = uint(P[P_DST_H]);
    if (gid.x >= W || gid.y >= H) return;
    float2 X = (float2(gid) + 0.5f) / float2(P[P_OUT_SX], P[P_OUT_SY]) - 0.5f;
    bool ok;
    float2 S = sp_source_coord_mesh(P, M, D, X, ok);
    uint o = 3u * (gid.y * W + gid.x);
    out[o] = S.x; out[o + 1] = S.y; out[o + 2] = ok ? 1.0f : 0.0f;
}

kernel void sp_preview_gray_mesh(device const float* src, device float* dst, device float* valid,
                                 device const float* P, device const float* M, device const float* D,
                                 uint2 gid [[thread_position_in_grid]]) {
    uint W = uint(P[P_DST_W]), H = uint(P[P_DST_H]);
    if (gid.x >= W || gid.y >= H) return;
    float2 X = (float2(gid) + 0.5f) / float2(P[P_OUT_SX], P[P_OUT_SY]) - 0.5f;
    bool ok;
    float2 S = sp_source_coord_mesh(P, M, D, X, ok);
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
