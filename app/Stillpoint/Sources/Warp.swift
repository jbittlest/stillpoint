// The preview warp: shaders/warp.metal compiled at runtime (the SAME kernels sprender uses), an AVVideoCompositing
// compositor that warps each decoded frame with the plan record matching its PTS, and a single-frame renderer
// (thumbnails of the stabilised view, snapshots, self-test).
import AVFoundation
import CoreImage
import CoreVideo
import Foundation
import Metal

// Appended to warp.metal at compile time. They only call warp.metal's own geometry + resampling functions; the
// "After" view uses warp.metal's sp_warp_luma / sp_warp_chroma entry points unchanged.
private let appKernels = """

// ================================================================ Stillpoint.app preview kernels (appended at runtime)
// Before / split views. "Before" = the same output projection with the camera's own orientation (identity row
// matrices, MB) or, with SPAPP_RAW, the untouched source frame (output size == source size only).
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
kernel void spapp_split_luma(texture2d<float, access::read> src [[texture(0)]],
                             texture2d<float, access::write> dst [[texture(1)]],
                             device const float* P [[buffer(0)]], device const float* M [[buffer(1)]],
                             device const float* MB [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float4 v;
    if (float(gid.x) < P[SPAPP_SPLIT_X]) {
        if (P[SPAPP_RAW] > 0.5f) {
            uint2 q = min(gid, uint2(src.get_width() - 1, src.get_height() - 1));
            v = sp_q10(src.read(q) * P[P_IN_SCALE]);
        } else {
            v = spapp_luma_at(src, P, MB, gid);
        }
    } else {
        v = spapp_luma_at(src, P, M, gid);
    }
    dst.write(v, gid);
}
kernel void spapp_split_chroma(texture2d<float, access::read> src [[texture(0)]],
                               texture2d<float, access::write> dst [[texture(1)]],
                               device const float* P [[buffer(0)]], device const float* M [[buffer(1)]],
                               device const float* MB [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float4 v;
    if (2.0f * float(gid.x) < P[SPAPP_SPLIT_X]) {
        if (P[SPAPP_RAW] > 0.5f) {
            uint2 q = min(gid, uint2(src.get_width() - 1, src.get_height() - 1));
            v = sp_q10(src.read(q) * P[P_IN_SCALE]);
        } else {
            v = spapp_chroma_at(src, P, MB, gid);
        }
    } else {
        v = spapp_chroma_at(src, P, M, gid);
    }
    dst.write(v, gid);
}
"""

enum CompareMode: String, CaseIterable { case after, split, before }

struct WarpSettings: Equatable {
    var mode: CompareMode = .after
    var split: Float = 0.5          // fraction of the output width shown as "before" in split mode
    var rawBefore = false           // before = untouched source frame instead of the lens-matched original
    var kernel: Float = 0           // 0 lanczos3 (= export), 1 catmull-rom, 2 bilinear
}

struct WarpError: LocalizedError { let errorDescription: String? }

final class WarpEngine: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let psoY: MTLComputePipelineState, psoC: MTLComputePipelineState
    let psoSplitY: MTLComputePipelineState, psoSplitC: MTLComputePipelineState
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
        psoSplitY = try pso("spapp_split_luma")
        psoSplitC = try pso("spapp_split_chroma")
        var tc: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, dev, nil, &tc)
        guard let tc else { throw WarpError(errorDescription: "CVMetalTextureCache failed") }
        texCache = tc
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

    /// Encode the warp of `src` into `dst` (x420). Returns objects that must stay alive until `cb` completes.
    func encode(_ cb: MTLCommandBuffer, src: CVPixelBuffer, dst: CVPixelBuffer, plan: PlanFile?, time: Double,
                settings: WarpSettings) throws -> [AnyObject] {
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
        let sY = try texture(src, 0, is8 ? .r8Unorm : .r16Unorm), sC = try texture(src, 1, is8 ? .rg8Unorm : .rg16Unorm)
        let dY = try texture(dst, 0, .r16Unorm), dC = try texture(dst, 1, .rg16Unorm)
        var P = params(plan: usePlan, record: rec, srcW: W, srcH: H, dstW: dW, dstH: dH, is8: is8, full: full,
                       kernel: settings.kernel)
        let sameSize = W == dW && H == dH
        var keep: [AnyObject] = [sY, sC, dY, dC]
        guard let enc = cb.makeComputeCommandEncoder() else { throw WarpError(errorDescription: "encoder") }
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let gridY = MTLSize(width: dW, height: dH, depth: 1), gridC = MTLSize(width: dW / 2, height: dH / 2, depth: 1)

        if let plan = usePlan, let r = rec, settings.mode == .after {
            // exactly sprender's dispatch
            enc.setComputePipelineState(psoY)
            enc.setTexture(CVMetalTextureGetTexture(sY), index: 0)
            enc.setTexture(CVMetalTextureGetTexture(dY), index: 1)
            enc.setBytes(&P, length: 4 * P.count, index: 0)
            plan.withMatrices(r) { p, n in enc.setBytes(p, length: n, index: 1) }
            enc.dispatchThreads(gridY, threadsPerThreadgroup: tg)
            enc.setComputePipelineState(psoC)
            enc.setTexture(CVMetalTextureGetTexture(sC), index: 0)
            enc.setTexture(CVMetalTextureGetTexture(dC), index: 1)
            enc.dispatchThreads(gridC, threadsPerThreadgroup: tg)
        } else {
            // before / split (or no plan: untouched source)
            let nRows = usePlan?.nRows ?? 2
            let ident = identityBuffer(nRows)
            keep.append(ident)
            let raw = usePlan == nil || rec == nil || (settings.rawBefore && sameSize)
            if usePlan == nil || rec == nil {
                guard sameSize else { enc.endEncoding(); throw WarpError(errorDescription: "no plan record for this frame") }
                P[28] = Float.greatestFiniteMagnitude
            } else {
                P[28] = settings.mode == .split ? settings.split * Float(dW) : Float.greatestFiniteMagnitude
            }
            P[29] = raw ? 1 : 0
            for (pso, s, d, grid) in [(psoSplitY, sY, dY, gridY), (psoSplitC, sC, dC, gridC)] {
                enc.setComputePipelineState(pso)
                enc.setTexture(CVMetalTextureGetTexture(s), index: 0)
                enc.setTexture(CVMetalTextureGetTexture(d), index: 1)
                enc.setBytes(&P, length: 4 * P.count, index: 0)
                if let plan = usePlan, let r = rec {
                    plan.withMatrices(r) { p, n in enc.setBytes(p, length: n, index: 1) }
                } else {
                    enc.setBuffer(ident, offset: 0, index: 1)
                }
                enc.setBuffer(ident, offset: 0, index: 2)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            }
        }
        enc.endEncoding()
        return keep
    }
}

// MARK: - Shared preview state (UI thread writes, compositor reads)

final class PreviewState: @unchecked Sendable {
    private let lock = NSLock()
    private var _plan: PlanFile?
    private var _settings = WarpSettings()
    var plan: PlanFile? {
        get { lock.lock(); defer { lock.unlock() }; return _plan }
        set { lock.lock(); _plan = newValue; lock.unlock() }
    }
    var settings: WarpSettings {
        get { lock.lock(); defer { lock.unlock() }; return _settings }
        set { lock.lock(); _settings = newValue; lock.unlock() }
    }
    func snapshot() -> (PlanFile?, WarpSettings) {
        lock.lock(); defer { lock.unlock() }
        return (_plan, _settings)
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
            let (plan, settings) = ins.state.snapshot()
            let keep = try engine.encode(cb, src: src, dst: dst, plan: plan, time: req.compositionTime.seconds,
                                         settings: settings)
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

    /// Decode the source frame nearest `time` (PTS) and warp it with the plan (or pass it through).
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
        let cb = engine.queue.makeCommandBuffer()!
        let keep = try engine.encode(cb, src: src, dst: dst, plan: usable, time: pts, settings: settings)
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
