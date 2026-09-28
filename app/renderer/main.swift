// sprender — Stillpoint renderer (ENGINE_SPEC.md §3).  Owner: WP-B.
//
// AVAssetReader (VideoToolbox decode; '420v'/'420f' for 8-bit sources, 'x420'/'xf20' for 10-bit)
//   -> CVMetalTextureCache (zero-copy IOSurface textures)
//   -> shaders/warp.metal sp_warp_luma / sp_warp_chroma (compiled at runtime; plan geometry + Lanczos-3/Catmull-Rom)
//   -> AVAssetWriter (HEVC Main10 / HEVC Main10 speed-priority / ProRes 422 HQ), source PTS + source timescale,
//      audio passthrough trimmed to the rendered window, colour tags copied.
//
// Usage: sprender <in.mp4> <plan.spplan> <out.mov|mp4> [--start-frame N] [--frames M]
//                 [--codec hevc10|hevc10-speed|prores] [--bitrate-mbps 180] [--kernel lanczos3|catmullrom|bilinear]
//                 [--shader path/to/warp.metal] [--iters 3] [--no-audio] [--no-write] [--inflight 3]
//                 [--dump-frames i,j,...] [--dump-dir DIR] [--zero-base] [--quiet] [--no-progress]
// Progress (stdout, line-buffered, machine-readable; about every 0.5 s and once at the end unless --quiet/--no-progress):
//   PROGRESS frame=<source frames of the window handled so far> total=<frames in the window>
// Frame indices (--start-frame, --dump-frames) are SOURCE frame indices in presentation order. Output frame for a
// decoded source frame uses the plan record whose pts matches the frame's PTS within half a frame; source frames
// without a matching record are skipped (counted in the RESULT line).
// --dump-frames writes, per listed source frame, into --dump-dir: frame<N>.json, frame<N>_out_y.u16,
// frame<N>_out_uv.u16 (warped planes before encoding, 10-bit MSB-aligned), frame<N>_src_y.raw / _src_uv.raw
// (decoded source planes, 8- or 16-bit), frame<N>_coords.f32 (out_h x out_w x (sx, sy, valid) full-res source coords).

import AVFoundation
import CoreVideo
import Foundation
import Metal
import VideoToolbox

setvbuf(stdout, nil, _IOLBF, 0)
let args = CommandLine.arguments
func opt(_ name: String, _ def: String) -> String {
    if let i = args.firstIndex(of: name), i + 1 < args.count { return args[i + 1] }
    return def
}
func die(_ msg: String) -> Never { FileHandle.standardError.write((msg + "\n").data(using: .utf8)!); exit(1) }
guard args.count >= 4 else {
    print("usage: sprender <in.mp4> <plan.spplan> <out.mov|mp4> [--start-frame N] [--frames M] [--codec hevc10|hevc10-speed|prores] [--bitrate-mbps 180] [--kernel lanczos3|catmullrom|bilinear] [--shader warp.metal] [--dump-frames i,j] [--dump-dir DIR] [--no-write] [--no-audio]")
    exit(2)
}
let inURL = URL(fileURLWithPath: args[1])
let planURL = URL(fileURLWithPath: args[2])
let outURL = URL(fileURLWithPath: args[3])
let startFrame = Int(opt("--start-frame", "0")) ?? 0
let nFramesReq = Int(opt("--frames", "0")) ?? 0
let codec = opt("--codec", "hevc10")
let bitrate = Int((Double(opt("--bitrate-mbps", "180")) ?? 180) * 1_000_000)
let kernelName = opt("--kernel", "lanczos3")
let iters = Int(opt("--iters", "3")) ?? 3
let inflight = max(1, Int(opt("--inflight", "3")) ?? 3)
let noWrite = args.contains("--no-write")
let noAudio = args.contains("--no-audio")
let quiet = args.contains("--quiet")
let showProgress = !quiet && !args.contains("--no-progress")
let dumpFrames = Set(opt("--dump-frames", "").split(separator: ",").compactMap { Int($0) })
let dumpDir = URL(fileURLWithPath: opt("--dump-dir", "."))
guard ["hevc10", "hevc10-speed", "prores"].contains(codec) else { die("unknown --codec \(codec)") }
let kernelId: Float
switch kernelName {
case "lanczos3": kernelId = 0
case "catmullrom": kernelId = 1
case "bilinear": kernelId = 2
default: die("unknown --kernel \(kernelName)")
}

// MARK: - Plan (.spplan v1, see engine/stillpoint/plan_io.py)

let planData = try Data(contentsOf: planURL)
func rdU32(_ off: Int) -> UInt32 { planData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: UInt32.self) } }
func rdF32(_ off: Int) -> Float { planData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: Float.self) } }
func rdF64(_ off: Int) -> Double { planData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: Double.self) } }
guard planData.count >= 256, String(data: planData.subdata(in: 0..<8), encoding: .ascii) == "SPPLAN01", rdU32(8) == 1 else {
    die("not a v1 .spplan: \(planURL.path)")
}
let hdrBytes = Int(rdU32(12))
let planSrcW = Int(rdU32(16)), planSrcH = Int(rdU32(20)), outW = Int(rdU32(24)), outH = Int(rdU32(28))
let planN = Int(rdU32(32)), nRows = Int(rdU32(36)), lensModel = rdU32(40), recBytes = Int(rdU32(44))
let lensF = (0..<8).map { rdF32(48 + 4 * $0) }
guard recBytes == 24 + 36 * nRows, planData.count >= hdrBytes + planN * recBytes, nRows >= 2 else { die("corrupt plan") }
guard outW % 2 == 0, outH % 2 == 0 else { die("plan output size must be even") }
let planPTS: [Double] = (0..<planN).map { rdF64(hdrBytes + $0 * recBytes) }
for i in 1..<max(planN, 1) where planPTS[i] <= planPTS[i - 1] { die("plan PTS not strictly increasing at record \(i)") }

func nearest(_ arr: [Double], _ t: Double) -> Int? {
    if arr.isEmpty { return nil }
    var lo = 0, hi = arr.count - 1
    while lo < hi { let m = (lo + hi) / 2; if arr[m] < t { lo = m + 1 } else { hi = m } }
    var best = lo
    if lo > 0 && abs(arr[lo - 1] - t) < abs(arr[lo] - t) { best = lo - 1 }
    return best
}

// MARK: - Metal

let device = MTLCreateSystemDefaultDevice()!
func shaderPath() -> String {
    let explicit = opt("--shader", "")
    if !explicit.isEmpty { return explicit }
    let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
    let cands = [exe.appendingPathComponent("warp.metal"),
                 exe.appendingPathComponent("../../../shaders/warp.metal"),   // app/renderer/.build -> stillpoint/shaders
                 exe.appendingPathComponent("../../shaders/warp.metal"),
                 exe.appendingPathComponent("../shaders/warp.metal")]
    for c in cands where FileManager.default.fileExists(atPath: c.standardizedFileURL.path) { return c.standardizedFileURL.path }
    die("cannot find shaders/warp.metal (use --shader)")
}
let shaderFile = shaderPath()
let shaderSrc = try String(contentsOfFile: shaderFile, encoding: .utf8)
let copts = MTLCompileOptions()
if #available(macOS 15.0, *) { copts.mathMode = .safe } else { copts.fastMathEnabled = false }
let lib: MTLLibrary
do { lib = try device.makeLibrary(source: shaderSrc, options: copts) } catch { die("shader compile failed: \(error)") }
let psoY = try device.makeComputePipelineState(function: lib.makeFunction(name: "sp_warp_luma")!)
let psoC = try device.makeComputePipelineState(function: lib.makeFunction(name: "sp_warp_chroma")!)
let psoMap = try device.makeComputePipelineState(function: lib.makeFunction(name: "sp_coord_map")!)
let queue = device.makeCommandQueue()!
var texCache: CVMetalTextureCache?
CVMetalTextureCacheCreate(nil, nil, device, nil, &texCache)
func makeTex(_ pb: CVPixelBuffer, _ plane: Int, _ fmt: MTLPixelFormat) -> CVMetalTexture {
    let w = CVPixelBufferGetWidthOfPlane(pb, plane), h = CVPixelBufferGetHeightOfPlane(pb, plane)
    var t: CVMetalTexture?
    let r = CVMetalTextureCacheCreateTextureFromImage(nil, texCache!, pb, nil, fmt, w, h, plane, &t)
    precondition(r == kCVReturnSuccess, "texture fail \(r)")
    return t!
}

// MARK: - Asset

let asset = AVURLAsset(url: inURL)
guard let vTrack = try await asset.loadTracks(withMediaType: .video).first else { die("no video track") }
let aTrack = noAudio ? nil : try await asset.loadTracks(withMediaType: .audio).first
let vFD = try await vTrack.load(.formatDescriptions).first!
let fps = try await vTrack.load(.nominalFrameRate)
let dims = CMVideoFormatDescriptionGetDimensions(vFD)
let W = Int(dims.width), H = Int(dims.height)
guard W == planSrcW && H == planSrcH else { die("plan source size \(planSrcW)x\(planSrcH) != video \(W)x\(H)") }
func ext(_ k: CFString) -> String? { CMFormatDescriptionGetExtension(vFD, extensionKey: k) as? String }
let prim = ext(kCMFormatDescriptionExtension_ColorPrimaries) ?? (kCMFormatDescriptionColorPrimaries_ITU_R_709_2 as String)
let trc = ext(kCMFormatDescriptionExtension_TransferFunction) ?? (kCMFormatDescriptionTransferFunction_ITU_R_709_2 as String)
let mat = ext(kCMFormatDescriptionExtension_YCbCrMatrix) ?? (kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2 as String)
let fullRange = (CMFormatDescriptionGetExtension(vFD, extensionKey: kCMFormatDescriptionExtension_FullRangeVideo) as? Bool) ?? false
let subtype = CMFormatDescriptionGetMediaSubType(vFD)
let bpc = (CMFormatDescriptionGetExtension(vFD, extensionKey: kCMFormatDescriptionExtension_BitsPerComponent) as? NSNumber)?.intValue
let is8 = args.contains("--read-8bit") || (!args.contains("--read-10bit") &&
          (subtype == kCMVideoCodecType_H264 || bpc == 8))
let srcTS = try await vTrack.load(.naturalTimeScale)

// Source frame PTS list (presentation order) from sample metadata — no decoding.
var srcPTS: [Double] = []
if vTrack.canProvideSampleCursors, let cur = vTrack.makeSampleCursorAtFirstSampleInDecodeOrder() {
    repeat { srcPTS.append(cur.presentationTimeStamp.seconds) } while cur.stepInDecodeOrder(byCount: 1) == 1
    srcPTS.sort()
}
let frameDur: Double = {
    if srcPTS.count >= 2 { let d = zip(srcPTS.dropFirst(), srcPTS).map { $0 - $1 }.sorted(); return d[d.count / 2] }
    return 1.0 / Double(fps)
}()
if srcPTS.isEmpty {  // fallback: nominal CFR
    let dur = try await vTrack.load(.timeRange)
    let n = Int((dur.duration.seconds / frameDur).rounded())
    srcPTS = (0..<n).map { dur.start.seconds + Double($0) * frameDur }
}
let nSrc = srcPTS.count
guard startFrame >= 0 && startFrame < nSrc else { die("--start-frame \(startFrame) out of range (0..<\(nSrc))") }
let lastFrame = nFramesReq > 0 ? min(nSrc - 1, startFrame + nFramesReq - 1) : nSrc - 1
let windowFrames = lastFrame - startFrame + 1
let winStart = srcPTS[startFrame], winEnd = srcPTS[lastFrame]
if !quiet {
    print("input \(W)x\(H) \(fps) fps codec=\(subtype.fourcc) \(is8 ? "8" : "10")-bit fullRange=\(fullRange) primaries=\(prim) transfer=\(trc) matrix=\(mat) frames=\(nSrc) timescale=\(srcTS)")
    print("plan \(planN) records, out \(outW)x\(outH), rows \(nRows), lens \(lensModel == 1 ? "kb4" : "pinhole"); window frames \(startFrame)...\(lastFrame); kernel \(kernelName); shader \(shaderFile)")
}

// MARK: - Reader

let reader = try AVAssetReader(asset: asset)
let t0CM = CMTime(seconds: max(0, winStart - 0.5 * frameDur), preferredTimescale: srcTS)
let t1CM = CMTime(seconds: winEnd + 0.5 * frameDur, preferredTimescale: srcTS)
reader.timeRange = CMTimeRange(start: t0CM, end: t1CM)
let inPF: OSType = is8 ? (fullRange ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
                       : (fullRange ? kCVPixelFormatType_420YpCbCr10BiPlanarFullRange : kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange)
let vOut = AVAssetReaderTrackOutput(track: vTrack, outputSettings: [
    kCVPixelBufferPixelFormatTypeKey as String: inPF,
    kCVPixelBufferMetalCompatibilityKey as String: true,
    kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
])
vOut.alwaysCopiesSampleData = false
reader.add(vOut)
var aOut: AVAssetReaderTrackOutput?
if let aTrack, !noWrite {
    let o = AVAssetReaderTrackOutput(track: aTrack, outputSettings: nil)  // compressed passthrough
    o.alwaysCopiesSampleData = false
    reader.add(o); aOut = o
}

// MARK: - Writer

let outPF: OSType = fullRange ? kCVPixelFormatType_420YpCbCr10BiPlanarFullRange : kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
let pbAttrs: [String: Any] = [
    kCVPixelBufferPixelFormatTypeKey as String: outPF,
    kCVPixelBufferWidthKey as String: outW, kCVPixelBufferHeightKey as String: outH,
    kCVPixelBufferMetalCompatibilityKey as String: true,
    kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
]
var writer: AVAssetWriter?
var vIn: AVAssetWriterInput?
var aIn: AVAssetWriterInput?
var adaptor: AVAssetWriterInputPixelBufferAdaptor?
if !noWrite {
    try? FileManager.default.removeItem(at: outURL)
    let ft: AVFileType = outURL.pathExtension.lowercased() == "mp4" ? .mp4 : .mov
    let w = try AVAssetWriter(outputURL: outURL, fileType: ft)
    let color: [String: Any] = [AVVideoColorPrimariesKey: prim, AVVideoTransferFunctionKey: trc, AVVideoYCbCrMatrixKey: mat]
    var vs: [String: Any] = [AVVideoWidthKey: outW, AVVideoHeightKey: outH, AVVideoColorPropertiesKey: color]
    if codec == "prores" {
        vs[AVVideoCodecKey] = AVVideoCodecType.proRes422HQ
    } else {
        vs[AVVideoCodecKey] = AVVideoCodecType.hevc
        var cp: [String: Any] = [
            AVVideoAverageBitRateKey: bitrate,
            AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String,
            AVVideoExpectedSourceFrameRateKey: Int(fps.rounded()),
        ]
        if codec == "hevc10-speed" { cp[kVTCompressionPropertyKey_PrioritizeEncodingSpeedOverQuality as String] = true }
        vs[AVVideoCompressionPropertiesKey] = cp
    }
    let vi = AVAssetWriterInput(mediaType: .video, outputSettings: vs)
    vi.expectsMediaDataInRealTime = false
    vi.transform = try await vTrack.load(.preferredTransform)
    // AVAssetWriter's default track timescale (600) re-quantises 1001/60000 s frames -> keep the source timescale.
    vi.mediaTimeScale = srcTS
    w.movieTimeScale = srcTS
    let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: vi, sourcePixelBufferAttributes: pbAttrs)
    w.add(vi)
    if let aTrack, aOut != nil {
        let afd = try await aTrack.load(.formatDescriptions).first
        let ai = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: afd)
        ai.expectsMediaDataInRealTime = false
        w.add(ai); aIn = ai
    }
    writer = w; vIn = vi; adaptor = ad
}
var ownPool: CVPixelBufferPool?
CVPixelBufferPoolCreate(nil, nil, pbAttrs as CFDictionary, &ownPool)

// MARK: - Per-frame parameters (layout = shaders/warp.metal P_* indices)

let inScale: Float = is8 ? Float(255.0 * 256.0 / 65535.0) : 1.0
let blackY: Float = fullRange ? 0 : Float(64.0 * 64.0 / 65535.0)
let blackC: Float = Float(512.0 * 64.0 / 65535.0)
func params(record r: Int, dstW: Int, dstH: Int) -> [Float] {
    var P = [Float](repeating: 0, count: 32)
    let base = hdrBytes + r * recBytes
    P[0] = rdF32(base + 8); P[1] = rdF32(base + 12); P[2] = rdF32(base + 16)
    P[3] = Float(lensModel)
    for i in 0..<8 { P[4 + i] = lensF[i] }
    P[12] = Float(W); P[13] = Float(H); P[14] = Float(nRows); P[15] = Float(iters)
    P[16] = 1; P[17] = 1; P[18] = 1; P[19] = 1
    P[20] = kernelId; P[21] = inScale; P[22] = blackY; P[23] = blackC
    P[24] = Float(dstW); P[25] = Float(dstH)
    return P
}
func setMats(_ enc: MTLComputeCommandEncoder, record r: Int, index: Int) {
    let off = hdrBytes + r * recBytes + 24, len = 36 * nRows
    planData.withUnsafeBytes { raw in
        let p = raw.baseAddress!.advanced(by: off)
        if len <= 4096 { enc.setBytes(p, length: len, index: index) }
        else { enc.setBuffer(device.makeBuffer(bytes: p, length: len, options: .storageModeShared)!, offset: 0, index: index) }
    }
}

// MARK: - Pipeline

struct Pending { let cb: MTLCommandBuffer; let pb: CVPixelBuffer; let src: CVPixelBuffer; let pts: CMTime
                 let srcIdx: Int; let rec: Int; let map: MTLBuffer?; let keep: [AnyObject] }
var pending: [Pending] = []
var nIn = 0, nOut = 0, nSkipNoPlan = 0, nSkipWindow = 0
var lastProgress = Date.distantPast
/// Machine-readable progress for the app (app_bridge.parse_sprender_line): frames of the window handled so far.
func reportProgress(final: Bool = false) {
    guard showProgress else { return }
    let now = Date()
    guard final || now.timeIntervalSince(lastProgress) >= 0.5 else { return }
    lastProgress = now
    let done = final ? windowFrames : min(nOut + nSkipNoPlan, windowFrames)
    print("PROGRESS frame=\(done) total=\(windowFrames)")
}
var gpuTime = 0.0
var eos = false
var worstMatch = 0.0

func encodeFrame(_ sb: CMSampleBuffer) -> Pending? {
    guard let src = CMSampleBufferGetImageBuffer(sb) else { return nil }
    let pts = CMSampleBufferGetPresentationTimeStamp(sb)
    let ts = pts.seconds
    guard let si = nearest(srcPTS, ts), abs(srcPTS[si] - ts) <= 0.5 * frameDur else { nSkipWindow += 1; return nil }
    if si < startFrame || si > lastFrame { nSkipWindow += 1; return nil }
    guard let r = nearest(planPTS, ts), abs(planPTS[r] - ts) <= 0.5 * frameDur else {
        nSkipNoPlan += 1; reportProgress(); return nil
    }
    worstMatch = max(worstMatch, abs(planPTS[r] - ts))
    var dst: CVPixelBuffer?
    let pool = adaptor?.pixelBufferPool ?? ownPool!
    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &dst)
    guard let dst else { die("pixel buffer pool exhausted") }
    CVBufferPropagateAttachments(src, dst)
    let fY: MTLPixelFormat = is8 ? .r8Unorm : .r16Unorm
    let fC: MTLPixelFormat = is8 ? .rg8Unorm : .rg16Unorm
    let sY = makeTex(src, 0, fY), sC = makeTex(src, 1, fC)
    let dY = makeTex(dst, 0, .r16Unorm), dC = makeTex(dst, 1, .rg16Unorm)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    var P = params(record: r, dstW: outW, dstH: outH)
    enc.setComputePipelineState(psoY)
    enc.setTexture(CVMetalTextureGetTexture(sY), index: 0)
    enc.setTexture(CVMetalTextureGetTexture(dY), index: 1)
    enc.setBytes(&P, length: 4 * P.count, index: 0)
    setMats(enc, record: r, index: 1)
    enc.dispatchThreads(MTLSize(width: outW, height: outH, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    enc.setComputePipelineState(psoC)
    enc.setTexture(CVMetalTextureGetTexture(sC), index: 0)
    enc.setTexture(CVMetalTextureGetTexture(dC), index: 1)
    enc.dispatchThreads(MTLSize(width: outW / 2, height: outH / 2, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    var mapBuf: MTLBuffer?
    if dumpFrames.contains(si) {
        mapBuf = device.makeBuffer(length: outW * outH * 3 * 4, options: .storageModeShared)
        enc.setComputePipelineState(psoMap)
        enc.setBuffer(mapBuf, offset: 0, index: 0)
        enc.setBytes(&P, length: 4 * P.count, index: 1)
        setMats(enc, record: r, index: 2)
        enc.dispatchThreads(MTLSize(width: outW, height: outH, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }
    enc.endEncoding()
    cb.commit()
    nIn += 1
    return Pending(cb: cb, pb: dst, src: src, pts: pts, srcIdx: si, rec: r, map: mapBuf, keep: [sY, sC, dY, dC, sb])
}

func dumpPlane(_ pb: CVPixelBuffer, _ plane: Int, _ url: URL) {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let h = CVPixelBufferGetHeightOfPlane(pb, plane), w = CVPixelBufferGetWidthOfPlane(pb, plane)
    let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, plane)
    let fmt = CVPixelBufferGetPixelFormatType(pb)
    let is8b = fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    let bytesPerSample = (is8b ? 1 : 2) * (plane == 0 ? 1 : 2)
    let rowBytes = w * bytesPerSample
    var data = Data(capacity: rowBytes * h)
    let base = CVPixelBufferGetBaseAddressOfPlane(pb, plane)!
    for y in 0..<h { data.append(base.advanced(by: y * bpr).assumingMemoryBound(to: UInt8.self), count: rowBytes) }
    try! data.write(to: url)
}

func dump(_ p: Pending) {
    try? FileManager.default.createDirectory(at: dumpDir, withIntermediateDirectories: true)
    let pre = dumpDir.appendingPathComponent("frame\(p.srcIdx)")
    dumpPlane(p.pb, 0, URL(fileURLWithPath: pre.path + "_out_y.u16"))
    dumpPlane(p.pb, 1, URL(fileURLWithPath: pre.path + "_out_uv.u16"))
    dumpPlane(p.src, 0, URL(fileURLWithPath: pre.path + "_src_y.raw"))
    dumpPlane(p.src, 1, URL(fileURLWithPath: pre.path + "_src_uv.raw"))
    if let m = p.map { try! Data(bytes: m.contents(), count: m.length).write(to: URL(fileURLWithPath: pre.path + "_coords.f32")) }
    let meta: [String: Any] = ["src_index": p.srcIdx, "pts": p.pts.seconds, "plan_record": p.rec, "plan_pts": planPTS[p.rec],
                               "out_w": outW, "out_h": outH, "src_w": W, "src_h": H, "src_bits": is8 ? 8 : 10,
                               "kernel": kernelName, "iters": iters]
    try! JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: pre.path + ".json"))
}

func retire(_ p: Pending) {
    p.cb.waitUntilCompleted()
    if p.cb.status != .completed { die("GPU error \(String(describing: p.cb.error))") }
    gpuTime += p.cb.gpuEndTime - p.cb.gpuStartTime
    if dumpFrames.contains(p.srcIdx) { dump(p) }
    if let ad = adaptor {
        if !ad.append(p.pb, withPresentationTime: p.pts) { die("append failed: \(String(describing: writer?.error))") }
    }
    nOut += 1
    reportProgress()
}

let t0 = Date()
guard reader.startReading() else { die("reader: \(String(describing: reader.error))") }
// Default: the output keeps the SOURCE timeline (session starts at 0, so a window starting at frame N keeps its
// absolute PTS via an initial empty edit). --zero-base starts the output timeline at the window's first frame.
let zeroBase = args.contains("--zero-base")
let sessionStart = zeroBase ? CMTime(seconds: winStart, preferredTimescale: srcTS) : CMTime.zero
let sessionEnd = CMTime(seconds: winEnd + frameDur, preferredTimescale: srcTS)
if let writer {
    guard writer.startWriting() else { die("writer: \(String(describing: writer.error))") }
    writer.startSession(atSourceTime: sessionStart)
}

if let vIn, let writer {
    let group = DispatchGroup()
    group.enter()
    let vq = DispatchQueue(label: "video")
    var finished = false
    vIn.requestMediaDataWhenReady(on: vq) {
        while vIn.isReadyForMoreMediaData && !finished {
            if !pending.isEmpty && (pending.count >= inflight || eos) { retire(pending.removeFirst()); continue }
            if eos { vIn.markAsFinished(); finished = true; group.leave(); return }
            guard let sb = vOut.copyNextSampleBuffer() else { eos = true; continue }
            if let p = encodeFrame(sb) { pending.append(p) }
        }
    }
    if let aIn, let aOut {
        group.enter()
        let aq = DispatchQueue(label: "audio")
        var aDone = false
        aIn.requestMediaDataWhenReady(on: aq) {
            while aIn.isReadyForMoreMediaData && !aDone {
                if let sb = aOut.copyNextSampleBuffer() { aIn.append(sb) } else { aIn.markAsFinished(); aDone = true; group.leave() }
            }
        }
    }
    group.wait()
    writer.endSession(atSourceTime: sessionEnd)
    let sem = DispatchSemaphore(value: 0)
    writer.finishWriting { sem.signal() }
    sem.wait()
    if writer.status != .completed { die("writer failed: \(String(describing: writer.error))") }
} else {
    while let sb = vOut.copyNextSampleBuffer() {
        if let p = encodeFrame(sb) { pending.append(p) }
        if pending.count >= inflight { retire(pending.removeFirst()) }
    }
    while !pending.isEmpty { retire(pending.removeFirst()) }
}
if reader.status == .failed { die("reader failed: \(String(describing: reader.error))") }
let dt = Date().timeIntervalSince(t0)
if nOut > 0 { reportProgress(final: true) }
print(String(format: "RESULT frames=%d wall=%.3fs fps=%.1f gpu_ms_per_frame=%.2f skipped_no_plan=%d skipped_outside_window=%d worst_pts_match_ms=%.3f codec=%@ kernel=%@ in=%@",
             nOut, dt, Double(nOut) / dt, 1000 * gpuTime / Double(max(nOut, 1)), nSkipNoPlan, nSkipWindow, 1000 * worstMatch,
             noWrite ? "none" : codec, kernelName, is8 ? "8bit" : "10bit"))
if nOut == 0 { die("no frames rendered (plan PTS do not match the video?)") }

extension FourCharCode {
    var fourcc: String { String(bytes: [UInt8(self >> 24 & 255), UInt8(self >> 16 & 255), UInt8(self >> 8 & 255), UInt8(self & 255)], encoding: .ascii) ?? "?" }
}
