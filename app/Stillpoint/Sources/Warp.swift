// The preview warp: shaders/warp.metal compiled at runtime (the SAME kernels sprender uses), an AVVideoCompositing
// compositor that warps each decoded frame with the plan record matching its PTS, and a single-frame renderer
// (thumbnails of the stabilised view, snapshots, self-test).
import AVFoundation
import CoreImage
import CoreVideo
import Foundation
import Metal

// Appended to warp.metal at compile time. They only call warp.metal's own geometry + resampling functions; the
// "After" view uses warp.metal's sp_warp_luma / sp_warp_chroma (and the engine v5 _fill / _mesh variants) unchanged.
private let appKernels = """

// ================================================================ Stillpoint.app preview kernels (appended at runtime)
// "Before" overlay: the same output projection with the camera's own orientation (identity row matrices, MB) or,
// with SPAPP_RAW, the untouched source frame (output size == source size only). Writes only the output pixels left
// of SPAPP_SPLIT_X (full-res luma x), so split view = the stabilised frame exactly as exported (any kernel: plain,
// fill, mesh) with this overlay on top; SPLIT_X = +inf gives the whole "before" frame, 0 leaves the frame untouched.
#define SPAPP_SPLIT_X 28
#define SPAPP_RAW     29

static inline float4 spapp_luma_at(texture2d<float, access::read> src, device const float* P, device const float* M, uint2 gid) {
    bool ok;
    float2 S = sp_source_coord(P, M, float2(gid), ok);
    if (!ok) return float4(P[P_BLACK_Y]);
    int2 mx = int2(src.get_width(), src.get_height()) - 1;
    return sp_q10(sp_sample_tex(src, S, mx, int(P[P_KERNEL])) * P[P_IN_SCALE]);
}
static inline float4 spapp_chroma_at(texture2d<float, access::read> src, device const float* P, device const float* M, uint2 gid) {
    bool ok;
    float2 S = sp_source_coord(P, M, float2(2.0f * float(gid.x), 2.0f * float(gid.y) + 0.5f), ok);
    if (!ok) return float4(P[P_BLACK_C]);
    int2 mx = int2(src.get_width(), src.get_height()) - 1;
    float2 c = float2(S.x * 0.5f, (S.y - 0.5f) * 0.5f);
    return sp_q10(sp_sample_tex(src, c, mx, int(P[P_KERNEL])) * P[P_IN_SCALE]);
}
kernel void spapp_before_luma(texture2d<float, access::read> src [[texture(0)]],
                              texture2d<float, access::write> dst [[texture(1)]],
                              device const float* P [[buffer(0)]], device const float* MB [[buffer(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    if (!(float(gid.x) < P[SPAPP_SPLIT_X])) return;
    float4 v;
    if (P[SPAPP_RAW] > 0.5f) {
        uint2 q = min(gid, uint2(src.get_width() - 1, src.get_height() - 1));
        v = sp_q10(src.read(q) * P[P_IN_SCALE]);
    } else {
        v = spapp_luma_at(src, P, MB, gid);
    }
    dst.write(v, gid);
}
kernel void spapp_before_chroma(texture2d<float, access::read> src [[texture(0)]],
                                texture2d<float, access::write> dst [[texture(1)]],
                                device const float* P [[buffer(0)]], device const float* MB [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    if (!(2.0f * float(gid.x) < P[SPAPP_SPLIT_X])) return;
    float4 v;
    if (P[SPAPP_RAW] > 0.5f) {
        uint2 q = min(gid, uint2(src.get_width() - 1, src.get_height() - 1));
        v = sp_q10(src.read(q) * P[P_IN_SCALE]);
    } else {
        v = spapp_chroma_at(src, P, MB, gid);
    }
    dst.write(v, gid);
}
"""

enum CompareMode: String, CaseIterable { case after, split, before }

/// Where a full-frame fill plan's neighbouring source frames come from (the compositor only receives the current one).
enum FillMode: Equatable {
    case exact          // decode them now, blocking (export-identical: stills, snapshots, self-tests)
    case progressive    // use what is cached, fetch the rest in the background, then redraw (paused preview)
    case cachedOnly     // use what is cached, never decode (playback); uncovered border -> the kernel's soft edge
}

struct WarpSettings: Equatable {
    var mode: CompareMode = .after
    var split: Float = 0.5          // fraction of the output width shown as "before" in split mode
    var rawBefore = false           // before = untouched source frame instead of the lens-matched original
    var kernel: Float = 0           // 0 lanczos3 (= export), 1 catmull-rom, 2 bilinear
    var fill: FillMode = .exact     // full-frame fill plans: neighbour frames (see FillMode)
    var ignoreV5 = false            // self-test only: render fill / mesh plans with the plain kernels
}

struct WarpError: LocalizedError { let errorDescription: String? }

final class WarpEngine: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let psoY: MTLComputePipelineState, psoC: MTLComputePipelineState
    let psoBeforeY: MTLComputePipelineState, psoBeforeC: MTLComputePipelineState
    /// Engine v5 kernels (nil with an older warp.metal: such plans then render rotation-only / with black borders).
    let psoFillY: MTLComputePipelineState?, psoFillC: MTLComputePipelineState?
    let psoMeshY: MTLComputePipelineState?, psoMeshC: MTLComputePipelineState?
    let texCache: CVMetalTextureCache
    let shaderPath: String
    private let lock = NSLock()
    private var identity: [Int: MTLBuffer] = [:]

    private static let sharedLock = NSLock()
    private static var _shared: Result<WarpEngine, Error>?
    static func shared() throws -> WarpEngine {
        sharedLock.lock(); defer { sharedLock.unlock() }
        if _shared == nil { _shared = Result { try WarpEngine(shader: EngineConfig.shader) } }
        return try _shared!.get()
    }
    static func reset() { sharedLock.lock(); _shared = nil; sharedLock.unlock() }

    init(shader: URL) throws {
        guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else {
            throw WarpError(errorDescription: "No Metal device")
        }
        device = dev
        queue = q
        shaderPath = shader.path
        let src = try String(contentsOf: shader, encoding: .utf8) + appKernels
        let opts = MTLCompileOptions()
        if #available(macOS 15.0, *) { opts.mathMode = .safe } else { opts.fastMathEnabled = false }   // = sprender
        let lib = try dev.makeLibrary(source: src, options: opts)
        func pso(_ n: String) throws -> MTLComputePipelineState {
            guard let f = lib.makeFunction(name: n) else { throw WarpError(errorDescription: "warp.metal has no \(n)") }
            return try dev.makeComputePipelineState(function: f)
        }
        psoY = try pso("sp_warp_luma")
        psoC = try pso("sp_warp_chroma")
        psoBeforeY = try pso("spapp_before_luma")
        psoBeforeC = try pso("spapp_before_chroma")
        psoFillY = try? pso("sp_warp_luma_fill")
        psoFillC = try? pso("sp_warp_chroma_fill")
        psoMeshY = try? pso("sp_warp_luma_mesh")
        psoMeshC = try? pso("sp_warp_chroma_mesh")
        var tc: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, dev, nil, &tc)
        guard let tc else { throw WarpError(errorDescription: "CVMetalTextureCache failed") }
        texCache = tc
    }

    var kernelNames: String {
        (["sp_warp_luma/chroma"] + (psoFillY != nil ? ["_fill"] : []) + (psoMeshY != nil ? ["_mesh"] : [])
            + ["spapp_before overlay"]).joined(separator: ", ")
    }

    private func identityBuffer(_ nRows: Int) -> MTLBuffer {
        lock.lock(); defer { lock.unlock() }
        if let b = identity[nRows] { return b }
        var m = [Float](repeating: 0, count: 9 * nRows)
        for j in 0..<nRows { m[9 * j] = 1; m[9 * j + 4] = 1; m[9 * j + 8] = 1 }
        let b = device.makeBuffer(bytes: m, length: 4 * m.count, options: .storageModeShared)!
        identity[nRows] = b
        return b
    }

    private func texture(_ pb: CVPixelBuffer, _ plane: Int, _ fmt: MTLPixelFormat) throws -> CVMetalTexture {
        var t: CVMetalTexture?
        let r = CVMetalTextureCacheCreateTextureFromImage(nil, texCache, pb, nil, fmt, CVPixelBufferGetWidthOfPlane(pb, plane),
                                                          CVPixelBufferGetHeightOfPlane(pb, plane), plane, &t)
        guard r == kCVReturnSuccess, let t else { throw WarpError(errorDescription: "texture (\(r))") }
        return t
    }

    static func is8bit(_ fmt: OSType) -> Bool {
        fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    }
    static func isFullRange(_ fmt: OSType) -> Bool {
        fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || fmt == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
    }

    /// Parameter block, identical to sprender's `params(record:dstW:dstH:)` (layout = warp.metal P_* indices).
    private func params(plan: PlanFile?, record r: Int?, srcW: Int, srcH: Int, dstW: Int, dstH: Int, is8: Bool,
                        full: Bool, kernel: Float) -> [Float] {
        var P = [Float](repeating: 0, count: 32)
        if let plan, let r {
            P[0] = plan.outFx(r); P[1] = plan.outCx(r); P[2] = plan.outCy(r)
            P[3] = Float(plan.lensModel)
            for i in 0..<8 { P[4 + i] = plan.lens[i] }
            P[14] = Float(plan.nRows)
        }
        P[12] = Float(srcW); P[13] = Float(srcH); P[15] = 3
        P[16] = 1; P[17] = 1; P[18] = 1; P[19] = 1
        P[20] = kernel
        P[21] = is8 ? Float(255.0 * 256.0 / 65535.0) : 1.0
        P[22] = full ? 0 : Float(64.0 * 64.0 / 65535.0)
        P[23] = Float(512.0 * 64.0 / 65535.0)
        P[24] = Float(dstW); P[25] = Float(dstH)
        return P
    }

    /// What `encode` did (the self-test checks the kernel choice and how many fill sources were used).
    struct Encoded {
        var keep: [AnyObject] = []
        var kernel = "none"            // plain | fill | mesh | before
        var fillSources = 0            // neighbour frames bound (fill plans)
        var fillWanted = 0             // fill sources the record asks for
    }

    /// Encode the warp of `src` into `dst` (x420). `neighbours`: decoded source frames of this record's fill sources
    /// (plan record -> pixel buffer; missing ones are dropped exactly as sprender drops an undecodable source).
    /// The returned objects must stay alive until `cb` completes.
    func encode(_ cb: MTLCommandBuffer, src: CVPixelBuffer, dst: CVPixelBuffer, plan: PlanFile?, time: Double,
                settings: WarpSettings, neighbours: [Int: CVPixelBuffer] = [:]) throws -> Encoded {
        let fmt = CVPixelBufferGetPixelFormatType(src)
        let is8 = Self.is8bit(fmt)
        guard is8 || fmt == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange || fmt == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange else {
            throw WarpError(errorDescription: "unexpected source pixel format \(fmt)")
        }
        let full = Self.isFullRange(fmt)
        let W = CVPixelBufferGetWidth(src), H = CVPixelBufferGetHeight(src)
        let dW = CVPixelBufferGetWidth(dst), dH = CVPixelBufferGetHeight(dst)
        var usePlan = plan
        if let p = plan, p.srcW != W || p.srcH != H || p.outW != dW || p.outH != dH { usePlan = nil }
        let rec = usePlan?.record(at: time)
        let fY: MTLPixelFormat = is8 ? .r8Unorm : .r16Unorm, fC: MTLPixelFormat = is8 ? .rg8Unorm : .rg16Unorm
        let sY = try texture(src, 0, fY), sC = try texture(src, 1, fC)
        let dY = try texture(dst, 0, .r16Unorm), dC = try texture(dst, 1, .rg16Unorm)
        var P = params(plan: usePlan, record: rec, srcW: W, srcH: H, dstW: dW, dstH: dH, is8: is8, full: full,
                       kernel: settings.kernel)
        let sameSize = W == dW && H == dH
        var out = Encoded(keep: [sY, sC, dY, dC])
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let gridY = MTLSize(width: dW, height: dH, depth: 1), gridC = MTLSize(width: dW / 2, height: dH / 2, depth: 1)

        var overlaySplit: Float?       // before-overlay extent (full-res luma x), nil = none
        if let plan = usePlan, let r = rec, settings.mode != .before {
            // the stabilised frame, exactly as sprender renders it (same kernels, parameters and inputs)
            guard let enc = cb.makeComputeCommandEncoder() else { throw WarpError(errorDescription: "encoder") }
            if let f = plan.fill, !settings.ignoreV5, let pY = psoFillY, let pC = psoFillC {
                out.kernel = "fill"
                var nb: [(tY: CVMetalTexture, tC: CVMetalTexture, rec: Int, slot: PlanFile.FillSlot)] = []
                out.fillWanted = plan.fillCount(r)
                for i in 0..<plan.fillCount(r) {
                    let s = plan.fillSlot(r, i)
                    guard s.source >= 0, s.source < plan.count, let pb = neighbours[s.source],
                          CVPixelBufferGetPixelFormatType(pb) == fmt, CVPixelBufferGetWidth(pb) == W,
                          CVPixelBufferGetHeight(pb) == H else { continue }
                    nb.append((try texture(pb, 0, fY), try texture(pb, 1, fC), s.source, s))
                }
                out.fillSources = nb.count
                let blackTex: Float = full ? 0 : (is8 ? Float(16.0 / 255.0) : Float(4096.0 / 65535.0))
                var FP = [Float](repeating: 0, count: 32)
                FP[0] = Float(nb.count); FP[1] = f.featherMain; FP[2] = f.featherNb; FP[3] = f.sigma; FP[4] = f.fallbackWeight
                FP[5] = Float(f.meshW); FP[6] = Float(f.meshH); FP[7] = blackTex; FP[20] = f.fallbackBlur
                FP[21] = Float(plan.outW); FP[22] = Float(plan.outH); FP[23] = f.parallaxTol
                let nr = plan.nRows
                var FM = [Float](repeating: 0, count: max(1, nb.count) * nr * 9)
                for (i, e) in nb.enumerated() {
                    FP[8 + i] = e.slot.weight; FP[12 + i] = e.slot.gain; FP[16 + i] = Float(e.rec - r)
                    let G = e.slot.g.map(Double.init)
                    for row in 0..<nr {
                        let M = plan.matrix(e.rec, row: row).map(Double.init)
                        for a in 0..<3 { for c in 0..<3 {
                            FM[(i * nr + row) * 9 + 3 * a + c] = Float(M[3 * a] * G[c] + M[3 * a + 1] * G[3 + c] + M[3 * a + 2] * G[6 + c])
                        } }
                    }
                    out.keep += [e.tY, e.tC]
                }
                let MS = plan.fillVelocity(r)
                guard let fmBuf = device.makeBuffer(bytes: FM, length: 4 * FM.count, options: .storageModeShared),
                      let msBuf = device.makeBuffer(bytes: MS, length: 4 * MS.count, options: .storageModeShared) else {
                    enc.endEncoding(); throw WarpError(errorDescription: "fill buffers")
                }
                out.keep += [fmBuf, msBuf]
                for (pso, isY) in [(pY, true), (pC, false)] {
                    enc.setComputePipelineState(pso)
                    enc.setTexture(CVMetalTextureGetTexture(isY ? sY : sC), index: 0)
                    enc.setTexture(CVMetalTextureGetTexture(isY ? dY : dC), index: 1)
                    for i in 0..<4 {
                        let t = i < nb.count ? (isY ? nb[i].tY : nb[i].tC) : (isY ? sY : sC)
                        enc.setTexture(CVMetalTextureGetTexture(t), index: 2 + i)
                    }
                    enc.setBytes(&P, length: 4 * P.count, index: 0)
                    plan.withMatrices(r) { p, n in enc.setBytes(p, length: n, index: 1) }
                    enc.setBytes(&FP, length: 4 * FP.count, index: 2)
                    enc.setBuffer(fmBuf, offset: 0, index: 3)
                    enc.setBuffer(msBuf, offset: 0, index: 4)
                    enc.dispatchThreads(isY ? gridY : gridC, threadsPerThreadgroup: tg)
                }
            } else {
                var pY = psoY, pC = psoC
                var meshBuf: MTLBuffer?
                if let m = plan.mesh, !settings.ignoreV5, let mY = psoMeshY, let mC = psoMeshC {
                    out.kernel = "mesh"
                    pY = mY; pC = mC
                    P[30] = Float(m.nx); P[31] = Float(m.ny)       // P_MESH_NX / P_MESH_NY
                    if m.frameBytes > 4096 {
                        meshBuf = plan.withMesh(r) { p, n in device.makeBuffer(bytes: p, length: n, options: .storageModeShared) } ?? nil
                        if let meshBuf { out.keep.append(meshBuf) }
                    }
                } else {
                    out.kernel = "plain"
                }
                for (pso, isY) in [(pY, true), (pC, false)] {
                    enc.setComputePipelineState(pso)
                    enc.setTexture(CVMetalTextureGetTexture(isY ? sY : sC), index: 0)
                    enc.setTexture(CVMetalTextureGetTexture(isY ? dY : dC), index: 1)
                    enc.setBytes(&P, length: 4 * P.count, index: 0)
                    plan.withMatrices(r) { p, n in enc.setBytes(p, length: n, index: 1) }
                    if out.kernel == "mesh" {
                        if let meshBuf { enc.setBuffer(meshBuf, offset: 0, index: 2) }
                        else { _ = plan.withMesh(r) { p, n in enc.setBytes(p, length: n, index: 2) } }
                    }
                    enc.dispatchThreads(isY ? gridY : gridC, threadsPerThreadgroup: tg)
                }
            }
            enc.endEncoding()
            if settings.mode == .split { overlaySplit = settings.split * Float(dW) }
        } else {
            // before (or no plan: the untouched source)
            if usePlan == nil || rec == nil {
                guard sameSize else { throw WarpError(errorDescription: "no plan record for this frame") }
            }
            out.kernel = "before"
            overlaySplit = Float.greatestFiniteMagnitude
        }
        if let split = overlaySplit, split > 0 {
            guard let enc = cb.makeComputeCommandEncoder() else { throw WarpError(errorDescription: "encoder") }
            let nRows = usePlan?.nRows ?? 2
            let ident = identityBuffer(nRows)
            out.keep.append(ident)
            var PB = P
            PB[28] = split
            PB[29] = (usePlan == nil || rec == nil || (settings.rawBefore && sameSize)) ? 1 : 0
            PB[30] = 0; PB[31] = 0
            for (pso, s, d, grid) in [(psoBeforeY, sY, dY, gridY), (psoBeforeC, sC, dC, gridC)] {
                enc.setComputePipelineState(pso)
                enc.setTexture(CVMetalTextureGetTexture(s), index: 0)
                enc.setTexture(CVMetalTextureGetTexture(d), index: 1)
                enc.setBytes(&PB, length: 4 * PB.count, index: 0)
                enc.setBuffer(ident, offset: 0, index: 1)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            }
            enc.endEncoding()
        }
        return out
    }
}

// MARK: - Neighbouring source frames for full-frame fill plans

/// Decoded source frames that fill plans sample for the border (the plan's FILL section names up to 4 neighbouring
/// records per frame, up to +-max_offset away). The compositor only gets the current frame, so these are decoded
/// here with AVAssetReader -- the same decoder, pixel format and PTS matching sprender uses -- and kept in a small
/// LRU cache (by source PTS, bounded in bytes). Thread-safe.
final class FillSources: @unchecked Sendable {
    private let lock = NSLock()
    private var frames: [Int64: (sb: CMSampleBuffer, pb: CVPixelBuffer, used: Int, bytes: Int)] = [:]
    private var bytes = 0
    private var tick = 0
    private var inflight = Set<Int64>()
    private var unavailable = Set<Int64>()
    private let queue = DispatchQueue(label: "stillpoint.fill-sources", qos: .userInitiated)
    static let budgetBytes = 480_000_000
    /// Called on the main queue when a background fetch added frames (the paused preview redraws).
    var onReady: (() -> Void)?
    /// Frames decoded so far (self-test / diagnostics).
    private(set) var decodedFrames = 0

    private static func key(_ t: Double) -> Int64 { Int64((t * 1e6).rounded()) }

    func clear() {
        lock.lock(); frames.removeAll(); bytes = 0; inflight.removeAll(); unavailable.removeAll(); lock.unlock()
    }

    /// The cached frames of these plan records.
    func cached(plan: PlanFile, records: [Int]) -> [Int: CVPixelBuffer] {
        lock.lock(); defer { lock.unlock() }
        var out: [Int: CVPixelBuffer] = [:]
        for r in records {
            let k = Self.key(plan.pts[r])
            if let e = frames[k] { tick += 1; frames[k] = (e.sb, e.pb, tick, e.bytes); out[r] = e.pb }
        }
        return out
    }

    /// Cached + newly decoded frames of these records (blocking; decodes only what is missing).
    func fetch(track: AVAssetTrack, plan: PlanFile, records: [Int], format: OSType,
               cancelled: () -> Bool = { false }) -> [Int: CVPixelBuffer] {
        var have = cached(plan: plan, records: records)
        let missing = records.filter { have[$0] == nil }
        if !missing.isEmpty {
            let got = decode(track: track, plan: plan, records: missing, format: format, cancelled: cancelled)
            have.merge(got) { a, _ in a }
        }
        return have
    }

    /// Background fetch of the records not cached yet; `onReady` fires when something new arrived.
    func prefetch(track: AVAssetTrack, plan: PlanFile, records: [Int], format: OSType) {
        lock.lock()
        let todo = records.filter {
            let k = Self.key(plan.pts[$0])
            return frames[k] == nil && !inflight.contains(k) && !unavailable.contains(k)
        }
        todo.forEach { inflight.insert(Self.key(plan.pts[$0])) }
        lock.unlock()
        guard !todo.isEmpty else { return }
        queue.async { [self] in
            let got = decode(track: track, plan: plan, records: todo, format: format, cancelled: { false })
            lock.lock()
            todo.forEach {
                let k = Self.key(plan.pts[$0])
                inflight.remove(k)
                if got[$0] == nil { unavailable.insert(k) }
            }
            lock.unlock()
            if !got.isEmpty { DispatchQueue.main.async { self.onReady?() } }
        }
    }

    private func store(_ k: Int64, _ sb: CMSampleBuffer, _ pb: CVPixelBuffer) {
        let b = CVPixelBufferGetDataSize(pb)
        lock.lock()
        tick += 1
        if let old = frames[k] { bytes -= old.bytes }
        frames[k] = (sb, pb, tick, b)
        bytes += b
        decodedFrames += 1
        while bytes > Self.budgetBytes, frames.count > 1,
              let victim = frames.filter({ $0.key != k }).min(by: { $0.value.used < $1.value.used }) {
            bytes -= victim.value.bytes
            frames.removeValue(forKey: victim.key)
        }
        lock.unlock()
    }

    /// AVAssetReader over [first, last] needed PTS (it decodes from the previous keyframe), keeping the frames whose
    /// PTS matches a needed record within a quarter frame. The range starts 1.5 frames early: the reader delivers the
    /// sample that overlaps the range start re-stamped WITH the range start, so a start half a frame early made the
    /// previous frame look like the needed one (DJI_0034 record 1478 got frame 1477's pixels). The range is read to
    /// its end, like sprender, and the sample buffers are kept with their pixel buffers.
    private func decode(track: AVAssetTrack, plan: PlanFile, records: [Int], format: OSType,
                        cancelled: () -> Bool) -> [Int: CVPixelBuffer] {
        guard let asset = track.asset, !records.isEmpty else { return [:] }
        let fd = plan.frameDuration
        let need = records.sorted { plan.pts[$0] < plan.pts[$1] }
        let t0 = plan.pts[need.first!], t1 = plan.pts[need.last!]
        guard let reader = try? AVAssetReader(asset: asset) else { return [:] }
        reader.timeRange = CMTimeRange(start: CMTime(seconds: max(0, t0 - 1.5 * fd), preferredTimescale: 600_000),
                                       end: CMTime(seconds: t1 + 0.5 * fd, preferredTimescale: 600_000))
        let o = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()])
        o.alwaysCopiesSampleData = false
        guard reader.canAdd(o) else { return [:] }
        reader.add(o)
        guard reader.startReading() else { return [:] }
        var got: [(Int, CMSampleBuffer, CVPixelBuffer)] = []
        var j = 0
        var aborted = false
        while let sb = o.copyNextSampleBuffer() {
            let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
            while j < need.count && plan.pts[need[j]] < t - 0.25 * fd { j += 1 }
            if j < need.count, abs(plan.pts[need[j]] - t) <= 0.25 * fd, CMSampleBufferMakeDataReady(sb) == noErr,
               let pb = CMSampleBufferGetImageBuffer(sb) {
                got.append((need[j], sb, pb))
                j += 1
            }
            if cancelled() { aborted = true; break }
        }
        if aborted || reader.status == .reading { reader.cancelReading() }
        guard !aborted else { return [:] }
        var out: [Int: CVPixelBuffer] = [:]
        for (r, sb, pb) in got {
            out[r] = pb
            store(Self.key(plan.pts[r]), sb, pb)
        }
        return out
    }
}

// MARK: - Shared preview state (UI thread writes, compositor reads)

final class PreviewState: @unchecked Sendable {
    private let lock = NSLock()
    private var _plan: PlanFile?
    private var _settings = WarpSettings()
    private var _track: AVAssetTrack?
    /// Neighbouring source frames for full-frame fill plans (per clip; valid for any plan of that clip).
    let fill = FillSources()
    var plan: PlanFile? {
        get { lock.lock(); defer { lock.unlock() }; return _plan }
        set { lock.lock(); _plan = newValue; lock.unlock() }
    }
    var settings: WarpSettings {
        get { lock.lock(); defer { lock.unlock() }; return _settings }
        set { lock.lock(); _settings = newValue; lock.unlock() }
    }
    /// The clip's video track (the fill sources are decoded from it).
    var track: AVAssetTrack? {
        get { lock.lock(); defer { lock.unlock() }; return _track }
        set { lock.lock(); _track = newValue; lock.unlock() }
    }
    func snapshot() -> (PlanFile?, WarpSettings, AVAssetTrack?) {
        lock.lock(); defer { lock.unlock() }
        return (_plan, _settings, _track)
    }

    /// The fill sources record `time` needs, per the settings' FillMode (empty for non-fill plans / "before").
    /// `cancelled` aborts a blocking (.exact) decode.
    func neighbours(plan: PlanFile?, settings: WarpSettings, track: AVAssetTrack?, time: Double, format: OSType,
                    cancelled: () -> Bool = { false }) -> [Int: CVPixelBuffer] {
        guard let plan, plan.hasFill, !settings.ignoreV5, settings.mode != .before, let r = plan.record(at: time) else { return [:] }
        let recs = plan.fillSources(r)
        guard !recs.isEmpty else { return [:] }
        switch settings.fill {
        case .cachedOnly:
            return fill.cached(plan: plan, records: recs)
        case .progressive:
            let have = fill.cached(plan: plan, records: recs)
            if have.count < recs.count, let track { fill.prefetch(track: track, plan: plan, records: recs, format: format) }
            return have
        case .exact:
            guard let track else { return fill.cached(plan: plan, records: recs) }
            return fill.fetch(track: track, plan: plan, records: recs, format: format, cancelled: cancelled)
        }
    }
}

final class PreviewInstruction: NSObject, AVVideoCompositionInstructionProtocol, @unchecked Sendable {
    let timeRange: CMTimeRange
    let enablePostProcessing = false
    let containsTweening = true
    let requiredSourceTrackIDs: [NSValue]?
    let passthroughTrackID = kCMPersistentTrackID_Invalid
    let trackID: CMPersistentTrackID
    let state: PreviewState
    init(timeRange: CMTimeRange, trackID: CMPersistentTrackID, state: PreviewState) {
        self.timeRange = timeRange
        self.trackID = trackID
        self.state = state
        requiredSourceTrackIDs = [NSNumber(value: trackID)]
    }
}

/// Base compositor; the 8-bit / 10-bit subclasses request the decoder's native format (as sprender does), so the
/// kernel sees exactly the same samples the export sees.
class StabilizingCompositor: NSObject, AVVideoCompositing, @unchecked Sendable {
    class var sourceFormat: OSType { kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange }
    private let renderQueue = DispatchQueue(label: "stillpoint.compositor", qos: .userInteractive)
    private let genLock = NSLock()
    private var generation = 0
    static let framesRendered = ManagedAtomicCounter()
    static let gpuTime = ManagedAtomicSum()
    /// What the recently composed frames used (composition time, kernel, fill sources) -- self-test diagnostics.
    static let recent = LockedBox<[(Double, WarpEngine.Encoded)]>([])
    static func encoded(at t: Double) -> WarpEngine.Encoded? {
        recent.value.last { abs($0.0 - t) < 1e-3 }?.1
    }

    var sourcePixelBufferAttributes: [String: any Sendable]? {
        [kCVPixelBufferPixelFormatTypeKey as String: [Self.sourceFormat],
         kCVPixelBufferMetalCompatibilityKey as String: true,
         kCVPixelBufferIOSurfacePropertiesKey as String: [String: Int]()]
    }
    var requiredPixelBufferAttributesForRenderContext: [String: any Sendable] {
        [kCVPixelBufferPixelFormatTypeKey as String: [kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange],
         kCVPixelBufferMetalCompatibilityKey as String: true,
         kCVPixelBufferIOSurfacePropertiesKey as String: [String: Int]()]
    }
    func renderContextChanged(_ newRenderContext: AVVideoCompositionRenderContext) {}

    func startRequest(_ req: AVAsynchronousVideoCompositionRequest) {
        genLock.lock(); let gen = generation; genLock.unlock()
        renderQueue.async { [self] in
            genLock.lock(); let stale = gen != generation; genLock.unlock()
            if stale { req.finishCancelledRequest(); return }
            render(req)
        }
    }

    func cancelAllPendingVideoCompositionRequests() {
        genLock.lock(); generation += 1; genLock.unlock()
    }

    private func render(_ req: AVAsynchronousVideoCompositionRequest) {
        guard let ins = req.videoCompositionInstruction as? PreviewInstruction,
              let src = req.sourceFrame(byTrackID: ins.trackID),
              let dst = req.renderContext.newPixelBuffer() else {
            req.finish(with: WarpError(errorDescription: "no source frame"))
            return
        }
        CVBufferPropagateAttachments(src, dst)
        do {
            let engine = try WarpEngine.shared()
            guard let cb = engine.queue.makeCommandBuffer() else { throw WarpError(errorDescription: "command buffer") }
            let (plan, settings, track) = ins.state.snapshot()
            let t = req.compositionTime.seconds
            genLock.lock(); let gen = generation; genLock.unlock()
            let nbs = ins.state.neighbours(plan: plan, settings: settings, track: track, time: t,
                                           format: CVPixelBufferGetPixelFormatType(src),
                                           cancelled: { [self] in genLock.lock(); defer { genLock.unlock() }; return gen != generation })
            let enc = try engine.encode(cb, src: src, dst: dst, plan: plan, time: t, settings: settings, neighbours: nbs)
            var rc = StabilizingCompositor.recent.value
            var info = enc
            info.keep = []                 // never pin textures / pixel buffers here (the decoder pool must recycle them)
            rc.append((t, info)); if rc.count > 24 { rc.removeFirst(rc.count - 24) }
            StabilizingCompositor.recent.set(rc)
            let keep = (enc.keep, nbs)
            cb.addCompletedHandler { c in
                _ = keep
                if c.status == .completed {
                    StabilizingCompositor.framesRendered.increment()
                    StabilizingCompositor.gpuTime.add(c.gpuEndTime - c.gpuStartTime)
                    req.finish(withComposedVideoFrame: dst)
                } else {
                    req.finish(with: c.error ?? WarpError(errorDescription: "GPU error"))
                }
            }
            cb.commit()
        } catch {
            req.finish(with: error)
        }
    }
}

final class StabilizingCompositor8: StabilizingCompositor, @unchecked Sendable {
    override class var sourceFormat: OSType { kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange }
}
final class StabilizingCompositor10: StabilizingCompositor, @unchecked Sendable {
    override class var sourceFormat: OSType { kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange }
}

final class ManagedAtomicSum: @unchecked Sendable {
    private let lock = NSLock()
    private var v = 0.0
    func add(_ x: Double) { lock.lock(); v += x; lock.unlock() }
    var value: Double { lock.lock(); defer { lock.unlock() }; return v }
}

final class LockedBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var v: T
    init(_ v: T) { self.v = v }
    func set(_ x: T) { lock.lock(); v = x; lock.unlock() }
    var value: T { lock.lock(); defer { lock.unlock() }; return v }
}

final class ManagedAtomicCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var v = 0
    func increment() { lock.lock(); v += 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return v }
}

/// Source-track facts needed to build compositions (8/10-bit choice mirrors sprender's).
struct TrackInfo {
    let track: AVAssetTrack
    let size: CGSize
    let frameDuration: CMTime
    let duration: CMTime            // end of the asset timeline (>= end of the video track)
    let is8bit: Bool
    let fps: Double

    static func load(_ asset: AVAsset) async throws -> TrackInfo {
        guard let t = try await asset.loadTracks(withMediaType: .video).first else {
            throw WarpError(errorDescription: "no video track")
        }
        let (fds, size, minFD, nominal, range) = try await t.load(.formatDescriptions, .naturalSize, .minFrameDuration,
                                                                    .nominalFrameRate, .timeRange)
        var is8 = true
        if let fd = fds.first {
            let sub = CMFormatDescriptionGetMediaSubType(fd)
            let bpc = (CMFormatDescriptionGetExtension(fd, extensionKey: kCMFormatDescriptionExtension_BitsPerComponent) as? NSNumber)?.intValue
            is8 = sub == kCMVideoCodecType_H264 || bpc == 8
        }
        let fd = minFD.isValid && minFD.seconds > 0 ? minFD : CMTime(value: 1001, timescale: 60000)
        // AVPlayer rejects a video composition whose instructions do not span the whole asset (DJI Osmo files have
        // audio that ends a few ms after the video), so cover max(asset duration, video track end).
        let assetDur = try await asset.load(.duration)
        let end = CMTimeMaximum(range.end, assetDur)
        return TrackInfo(track: t, size: size, frameDuration: fd, duration: end, is8bit: is8,
                         fps: Double(nominal))
    }

    func makeComposition(state: PreviewState, renderSize: CGSize) -> AVMutableVideoComposition {
        let c = AVMutableVideoComposition()
        c.customVideoCompositorClass = is8bit ? StabilizingCompositor8.self : StabilizingCompositor10.self
        c.frameDuration = frameDuration
        c.renderSize = renderSize
        c.instructions = [PreviewInstruction(timeRange: CMTimeRange(start: .zero, duration: duration),
                                             trackID: track.trackID, state: state)]
        return c
    }
}

// MARK: - Single frames (snapshots, still previews)

enum StillRenderer {
    static let ciContext = CIContext(options: [.cacheIntermediates: false])

    /// Decode the source frame nearest `time` (PTS) and warp it with the plan (or pass it through). Fill plans get
    /// their neighbouring source frames decoded too, so the still is the exported frame (never black borders).
    static func render(clip: URL, plan: PlanFile?, time: Double, settings: WarpSettings,
                       maxWidth: Int = 1920) async throws -> CGImage {
        let asset = AVURLAsset(url: clip)
        let info = try await TrackInfo.load(asset)
        let reader = try AVAssetReader(asset: asset)
        let fd = info.frameDuration.seconds
        reader.timeRange = CMTimeRange(start: CMTime(seconds: max(0, time - 0.5 * fd), preferredTimescale: 60000),
                                       end: CMTime(seconds: time + 3 * fd, preferredTimescale: 60000))
        let out = AVAssetReaderTrackOutput(track: info.track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: info.is8bit ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
                                                                    : kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Int]()])
        out.alwaysCopiesSampleData = false
        reader.add(out)
        guard reader.startReading() else { throw reader.error ?? WarpError(errorDescription: "reader") }
        var best: (CVPixelBuffer, Double)?
        while let sb = out.copyNextSampleBuffer() {
            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
            if best == nil || abs(t - time) < abs(best!.1 - time) { best = (pb, t) }
            if t > time + fd { break }
        }
        reader.cancelReading()
        guard let (src, pts) = best else { throw WarpError(errorDescription: "no frame at \(time)s") }
        let usable = plan.flatMap { $0.srcW == CVPixelBufferGetWidth(src) && $0.srcH == CVPixelBufferGetHeight(src) ? $0 : nil }
        let w = usable?.outW ?? CVPixelBufferGetWidth(src), h = usable?.outH ?? CVPixelBufferGetHeight(src)
        let dst = try makePixelBuffer(w, h)
        CVBufferPropagateAttachments(src, dst)
        let engine = try WarpEngine.shared()
        // fill plans: the neighbouring source frames this record samples, decoded like sprender's ring
        let state = PreviewState()
        var st = settings
        if st.fill != .exact { st.fill = .exact }
        let nbs = state.neighbours(plan: usable, settings: st, track: info.track, time: pts,
                                   format: CVPixelBufferGetPixelFormatType(src))
        let cb = engine.queue.makeCommandBuffer()!
        let keep = (try engine.encode(cb, src: src, dst: dst, plan: usable, time: pts, settings: st, neighbours: nbs), nbs)
        await withCheckedContinuation { (k: CheckedContinuation<Void, Never>) in
            cb.addCompletedHandler { _ in k.resume() }
            cb.commit()
        }
        _ = keep
        if cb.status != .completed { throw WarpError(errorDescription: "GPU error: \(String(describing: cb.error))") }
        return try cgImage(dst, maxWidth: maxWidth)
    }

    static func makePixelBuffer(_ w: Int, _ h: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true,
                                    kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
        let r = CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, attrs as CFDictionary, &pb)
        guard r == kCVReturnSuccess, let pb else { throw WarpError(errorDescription: "pixel buffer \(r)") }
        return pb
    }

    static func cgImage(_ pb: CVPixelBuffer, maxWidth: Int) throws -> CGImage {
        var ci = CIImage(cvPixelBuffer: pb)
        let w = ci.extent.width
        if w > CGFloat(maxWidth) {
            let s = CGFloat(maxWidth) / w
            ci = ci.transformed(by: CGAffineTransform(scaleX: s, y: s))
        }
        guard let cg = ciContext.createCGImage(ci, from: ci.extent.integral, format: .RGBA8,
                                               colorSpace: CGColorSpace(name: CGColorSpace.sRGB)) else {
            throw WarpError(errorDescription: "CGImage conversion failed")
        }
        return cg
    }
}
