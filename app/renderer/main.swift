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
//                 [--dump-frames i,j,...] [--dump-dir DIR] [--zero-base] [--quiet] [--no-progress] [--no-fill]
//                 [--no-mesh] [--blur plan.spblur]
// --blur: synthetic shutter sidecar (engine/stillpoint/synth_blur.py): frames whose record has >= 2 taps are rendered
//   with sp_warp_luma_blur / sp_warp_chroma_blur (the same frame warped along the virtual path, averaged).
// Plans with the mesh extension (engine/stillpoint/plan_io.py, flags bit 1) are rendered with the *_mesh kernels
// (the per-frame mesh-residual offsets are bound as buffer 2); --no-mesh renders them rotation-only.
// Fill, mesh and --blur are separate kernels and not combinable yet: sprender refuses a plan / call that asks for two
// of them (--no-fill / --no-mesh drop a plan's section).
// Progress (stdout, line-buffered, machine-readable; about every 0.5 s and once at the end unless --quiet/--no-progress):
//   PROGRESS frame=<source frames of the window handled so far> total=<frames in the window>
// Frame indices (--start-frame, --dump-frames) are SOURCE frame indices in presentation order. Output frame for a
// decoded source frame uses the plan record whose pts matches the frame's PTS within half a frame; source frames
// without a matching record are skipped (counted in the RESULT line).
// --dump-frames writes, per listed source frame, into --dump-dir: frame<N>.json, frame<N>_out_y.u16,
// frame<N>_out_uv.u16 (warped planes before encoding, 10-bit MSB-aligned), frame<N>_src_y.raw / _src_uv.raw
// (decoded source planes, 8- or 16-bit), frame<N>_coords.f32 (out_h x out_w x (sx, sy, valid) full-res source coords).
// FULL-FRAME FILL (plan flags bit 0, engine/stillpoint/fill.py): output pixels outside (or within the feather band of)
// the current source frame are synthesised from up to 4 neighbouring source frames listed per record in the plan's
// FILL section (sp_warp_luma_fill / sp_warp_chroma_fill). Decoded frames are kept in a ring of +-max_offset frames
// (the decode range is widened by that radius around the window; frames outside the window are only fill sources).
// --no-fill renders the plain kernels (identical to a plan without the section).

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
    print("usage: sprender <in.mp4> <plan.spplan> <out.mov|mp4> [--start-frame N] [--frames M] [--codec hevc10|hevc10-speed|prores] [--bitrate-mbps 180] [--kernel lanczos3|catmullrom|bilinear] [--shader warp.metal] [--dump-frames i,j] [--dump-dir DIR] [--no-write] [--no-audio] [--no-fill]")
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

// MARK: - Plan (.spplan v1 + optional mesh extension, see engine/stillpoint/plan_io.py)

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
// Mesh extension (flags bit 1): per-record (meshNY x meshNX) float2 output-pixel offsets in a trailing block.
let planFlags = rdU32(84)
let hasMesh = (planFlags & 2) != 0 && !args.contains("--no-mesh")
let meshNX = hasMesh ? Int(rdU32(88)) : 0, meshNY = hasMesh ? Int(rdU32(92)) : 0
let meshOff: Int = hasMesh ? Int(planData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 96, as: UInt64.self) }) : 0
let meshFrameBytes = hasMesh ? Int(rdU32(104)) : 0
if hasMesh {
    guard meshNX >= 2, meshNY >= 2, meshFrameBytes == 8 * meshNX * meshNY,
          planData.count >= meshOff + planN * meshFrameBytes else { die("corrupt plan mesh block") }
}
for i in 1..<max(planN, 1) where planPTS[i] <= planPTS[i - 1] { die("plan PTS not strictly increasing at record \(i)") }

// FILL section (flags bit 0, see engine/stillpoint/plan_io.py)
let fillOff = hdrBytes + planN * recBytes
var useFill = (planFlags & 1) != 0 && !args.contains("--no-fill")
var fillHdr = 0, fillK = 0, fillRec = 0, meshW = 0, meshH = 0
var fFeatherMain: Float = 0, fFeatherNb: Float = 0, fSigma: Float = 0, fFbW: Float = 0.01, fFbBlur: Float = 0, fPar: Float = 0
if useFill {
    guard planData.count >= fillOff + 64,
          String(data: planData.subdata(in: fillOff..<(fillOff + 8)), encoding: .ascii) == "SPFILL01",
          rdU32(fillOff + 8) == 1 else { die("plan flags announce a fill section but none was found") }
    fillHdr = Int(rdU32(fillOff + 12)); fillK = Int(rdU32(fillOff + 20)); fillRec = Int(rdU32(fillOff + 24))
    meshW = Int(rdU32(fillOff + 28)); meshH = Int(rdU32(fillOff + 32))
    fFeatherMain = rdF32(fillOff + 36); fFeatherNb = rdF32(fillOff + 40); fSigma = rdF32(fillOff + 44)
    fFbW = rdF32(fillOff + 48); fFbBlur = rdF32(fillOff + 52); fPar = rdF32(fillOff + 60)   // 0 in older plans = off
    guard Int(rdU32(fillOff + 16)) == planN, fillRec == 16 + 64 * fillK + 8 * meshW * meshH,
          planData.count >= fillOff + fillHdr + planN * fillRec else { die("corrupt fill section") }
}
func fillBase(_ r: Int) -> Int { fillOff + fillHdr + r * fillRec }
func fillCount(_ r: Int) -> Int { useFill ? min(Int(rdU32(fillBase(r))), fillK, 4) : 0 }
func fillSrc(_ r: Int, _ i: Int) -> Int { Int(Int32(bitPattern: rdU32(fillBase(r) + 16 + 64 * i))) }
// MARK: - Synthetic shutter sidecar (.spblur v1, see engine/stillpoint/synth_blur.py)
// 64-byte header: "SPBLUR01", u32 version, u32 header_bytes, u32 n_frames, u32 max_taps, u32 record_bytes;
// records: f64 pts, u32 n_taps, f32 shutter_s, max_taps x (f32 D[9] row-major, f32 weight).
let blurPath = opt("--blur", "")
var blurData = Data()
var blurHdr = 0, blurN = 0, blurMax = 0, blurRec = 0
var blurPTS: [Double] = []
if !blurPath.isEmpty {
    blurData = try Data(contentsOf: URL(fileURLWithPath: blurPath))
    guard blurData.count >= 64, String(data: blurData.subdata(in: 0..<8), encoding: .ascii) == "SPBLUR01" else {
        die("not a .spblur file: \(blurPath)")
    }
    func bU32(_ off: Int) -> UInt32 { blurData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: UInt32.self) } }
    blurHdr = Int(bU32(12)); blurN = Int(bU32(16)); blurMax = Int(bU32(20)); blurRec = Int(bU32(24))
    guard bU32(8) == 1, blurRec == 16 + 40 * blurMax, blurData.count >= blurHdr + blurN * blurRec, blurMax >= 1,
          blurMax <= 64 else { die("corrupt .spblur") }
    blurPTS = (0..<blurN).map { i in blurData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: blurHdr + i * blurRec, as: Double.self) } }
}
if useFill && !blurPath.isEmpty {
    die("--blur (synthetic shutter) is not supported together with a fill plan yet: render with --no-fill or without --blur")
}
if hasMesh && useFill { die("plan has both a fill section and a mesh block (not combinable yet): pass --no-fill or --no-mesh") }
if hasMesh && !blurPath.isEmpty { die("--blur (synthetic shutter) is not supported with a mesh plan yet: pass --no-mesh or drop --blur") }
func blurTaps(_ r: Int) -> Int {
    blurData.withUnsafeBytes { Int($0.loadUnaligned(fromByteOffset: blurHdr + r * blurRec + 8, as: UInt32.self)) }
}

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
let psoYF = try device.makeComputePipelineState(function: lib.makeFunction(name: "sp_warp_luma_fill")!)
let psoCF = try device.makeComputePipelineState(function: lib.makeFunction(name: "sp_warp_chroma_fill")!)
let psoYB = try device.makeComputePipelineState(function: lib.makeFunction(name: "sp_warp_luma_blur")!)
let psoCB = try device.makeComputePipelineState(function: lib.makeFunction(name: "sp_warp_chroma_blur")!)
let psoYm = hasMesh ? try device.makeComputePipelineState(function: lib.makeFunction(name: "sp_warp_luma_mesh")!) : psoY
let psoCm = hasMesh ? try device.makeComputePipelineState(function: lib.makeFunction(name: "sp_warp_chroma_mesh")!) : psoC
let psoMapM = hasMesh ? try device.makeComputePipelineState(function: lib.makeFunction(name: "sp_coord_map_mesh")!) : psoMap
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
let frameDurEarly = frameDur
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

// plan record of each window frame, and the source frame of each fill source (by PTS, like the main frames)
func recOf(_ si: Int) -> Int? {
    guard let r = nearest(planPTS, srcPTS[si]), abs(planPTS[r] - srcPTS[si]) <= 0.5 * frameDurEarly else { return nil }
    return r
}
func srcOfRec(_ r: Int) -> Int? {
    guard r >= 0 && r < planN, let si = nearest(srcPTS, planPTS[r]), abs(srcPTS[si] - planPTS[r]) <= 0.5 * frameDurEarly
    else { return nil }
    return si
}
var fillRadius = 0          // max |source frame of a fill source - output frame| over the window
var needMax = [Int](repeating: 0, count: windowFrames)   // last source frame each output frame needs
var nFillFrames = 0, nFillSources = 0
var lastUse: [Int: Int] = [:]  // decoded source frame -> last output frame that samples it (the ring keeps only these)
for si in startFrame...lastFrame {
    var m = si
    lastUse[si] = max(lastUse[si] ?? si, si)
    if useFill, let r = recOf(si) {
        let n = fillCount(r)
        if n > 0 { nFillFrames += 1; nFillSources += n }
        for i in 0..<n {
            if let sj = srcOfRec(fillSrc(r, i)) {
                fillRadius = max(fillRadius, abs(sj - si)); m = max(m, sj)
                lastUse[sj] = max(lastUse[sj] ?? si, si)
            }
        }
    }
    needMax[si - startFrame] = m
}
let decFirst = max(0, startFrame - fillRadius), decLast = min(nSrc - 1, lastFrame + fillRadius)
if !quiet {
    print("input \(W)x\(H) \(fps) fps codec=\(subtype.fourcc) \(is8 ? "8" : "10")-bit fullRange=\(fullRange) primaries=\(prim) transfer=\(trc) matrix=\(mat) frames=\(nSrc) timescale=\(srcTS)")
    print("plan \(planN) records, out \(outW)x\(outH), rows \(nRows), lens \(lensModel == 1 ? "kb4" : "pinhole"); window frames \(startFrame)...\(lastFrame); kernel \(kernelName); shader \(shaderFile)" + (hasMesh ? "; mesh \(meshNX)x\(meshNY)" : ""))
    if useFill { print("fill: \(nFillFrames)/\(windowFrames) frames, \(nFillSources) sources, ring radius \(fillRadius) frames, mesh \(meshW)x\(meshH); decode \(decFirst)...\(decLast)") }
}

// MARK: - Reader

let reader = try AVAssetReader(asset: asset)
let t0CM = CMTime(seconds: max(0, srcPTS[decFirst] - 0.5 * frameDur), preferredTimescale: srcTS)
let t1CM = CMTime(seconds: srcPTS[decLast] + 0.5 * frameDur, preferredTimescale: srcTS)
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
    if hasMesh { P[30] = Float(meshNX); P[31] = Float(meshNY) }   // P_MESH_NX / P_MESH_NY (28, 29: the app's)
    return P
}
func setMesh(_ enc: MTLComputeCommandEncoder, record r: Int, index: Int) {
    let off = meshOff + r * meshFrameBytes, len = meshFrameBytes
    planData.withUnsafeBytes { raw in
        let p = raw.baseAddress!.advanced(by: off)
        if len <= 4096 { enc.setBytes(p, length: len, index: index) }
        else { enc.setBuffer(device.makeBuffer(bytes: p, length: len, options: .storageModeShared)!, offset: 0, index: index) }
    }
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
var nBlurred = 0, nBlurTaps = 0

/// A decoded source frame kept in the ring (retained sample buffer + zero-copy plane textures).
final class DecFrame {
    let si: Int; let sb: CMSampleBuffer; let pb: CVPixelBuffer; let tY: CVMetalTexture; let tC: CVMetalTexture
    init(si: Int, sb: CMSampleBuffer, pb: CVPixelBuffer, tY: CVMetalTexture, tC: CVMetalTexture) {
        self.si = si; self.sb = sb; self.pb = pb; self.tY = tY; self.tC = tC
    }
}
var ring: [Int: DecFrame] = [:]
var nextOut = startFrame, maxDecoded = -1, nMissing = 0, nFillOut = 0, nFillSrcUsed = 0, nFillSrcMissing = 0
var ringPeak = 0
let blackTex: Float = fullRange ? 0 : (is8 ? Float(16.0 / 255.0) : Float(4096.0 / 65535.0))

/// Receive one decoded sample; returns the output frames that became ready.
func ingest(_ sb: CMSampleBuffer) -> [Pending] {
    guard let pb = CMSampleBufferGetImageBuffer(sb) else { return [] }
    let ts = CMSampleBufferGetPresentationTimeStamp(sb).seconds
    guard let si = nearest(srcPTS, ts), abs(srcPTS[si] - ts) <= 0.5 * frameDur else { nSkipWindow += 1; return [] }
    if si < decFirst || si > decLast { nSkipWindow += 1; return [] }
    if si < startFrame || si > lastFrame { nSkipWindow += 1 }      // decoded only as a fill source
    if let lu = lastUse[si], lu >= nextOut {           // only frames an output frame still samples
        let fY: MTLPixelFormat = is8 ? .r8Unorm : .r16Unorm
        let fC: MTLPixelFormat = is8 ? .rg8Unorm : .rg16Unorm
        ring[si] = DecFrame(si: si, sb: sb, pb: pb, tY: makeTex(pb, 0, fY), tC: makeTex(pb, 1, fC))
        ringPeak = max(ringPeak, ring.count)
    }
    maxDecoded = max(maxDecoded, si)
    return drain(eos: false)
}

/// Encode every output frame whose inputs (its source frame and fill sources) have been decoded.
func drain(eos: Bool) -> [Pending] {
    var out: [Pending] = []
    while nextOut <= lastFrame {
        if !eos && maxDecoded < needMax[nextOut - startFrame] { break }
        if let p = encodeOut(nextOut) { out.append(p) }
        nextOut += 1
        for k in ring.keys where (lastUse[k] ?? -1) < nextOut { ring.removeValue(forKey: k) }
    }
    return out
}

func encodeOut(_ si: Int) -> Pending? {
    guard let fr = ring[si] else { nMissing += 1; return nil }
    let src = fr.pb, sb = fr.sb
    let pts = CMSampleBufferGetPresentationTimeStamp(sb)
    let ts = pts.seconds
    guard let r = nearest(planPTS, ts), abs(planPTS[r] - ts) <= 0.5 * frameDur else {
        nSkipNoPlan += 1; reportProgress(); return nil
    }
    worstMatch = max(worstMatch, abs(planPTS[r] - ts))
    var dst: CVPixelBuffer?
    let pool = adaptor?.pixelBufferPool ?? ownPool!
    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &dst)
    guard let dst else { die("pixel buffer pool exhausted") }
    CVBufferPropagateAttachments(src, dst)
    let sY = fr.tY, sC = fr.tC
    let dY = makeTex(dst, 0, .r16Unorm), dC = makeTex(dst, 1, .rg16Unorm)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    var P = params(record: r, dstW: outW, dstH: outH)
    // fill sources present in the ring (a source whose frame could not be decoded is dropped)
    var nb: [(DecFrame, Int, Float, Float, Int)] = []      // frame, record, weight, gain, slot
    for i in 0..<fillCount(r) {
        let j = fillSrc(r, i)
        guard let sj = srcOfRec(j), let f = ring[sj] else { nFillSrcMissing += 1; continue }
        let b = fillBase(r) + 16 + 64 * i
        nb.append((f, j, rdF32(b + 4), rdF32(b + 8), i))
    }
    var keep: [AnyObject] = [fr, dY, dC]
    if useFill {                               // fill plans: always the fill kernels (0 sources: soft edge, not black)
        if !nb.isEmpty { nFillOut += 1; nFillSrcUsed += nb.count }
        var FP = [Float](repeating: 0, count: 32)
        FP[0] = Float(nb.count); FP[1] = fFeatherMain; FP[2] = fFeatherNb; FP[3] = fSigma; FP[4] = fFbW
        FP[5] = Float(meshW); FP[6] = Float(meshH); FP[7] = blackTex; FP[20] = fFbBlur
        FP[21] = Float(outW); FP[22] = Float(outH); FP[23] = fPar
        var FM = [Float](repeating: 0, count: max(1, nb.count) * nRows * 9)
        for (i, e) in nb.enumerated() {
            FP[8 + i] = e.2; FP[12 + i] = e.3; FP[16 + i] = Float(e.1 - r)
            let gb = fillBase(r) + 16 + 64 * e.4 + 16
            let G = (0..<9).map { Double(rdF32(gb + 4 * $0)) }
            let mb = hdrBytes + e.1 * recBytes + 24
            for row in 0..<nRows {
                let M = (0..<9).map { Double(rdF32(mb + 36 * row + 4 * $0)) }
                for a in 0..<3 { for c in 0..<3 {
                    FM[(i * nRows + row) * 9 + 3 * a + c] = Float(M[3 * a] * G[c] + M[3 * a + 1] * G[3 + c] + M[3 * a + 2] * G[6 + c])
                } }
            }
            keep.append(e.0)
        }
        var MS: [Float] = [0, 0]
        if meshW * meshH > 0 {
            let mb = fillBase(r) + 16 + 64 * fillK
            MS = (0..<(2 * meshW * meshH)).map { rdF32(mb + 4 * $0) }
        }
        let fmBuf = device.makeBuffer(bytes: FM, length: 4 * FM.count, options: .storageModeShared)!
        let msBuf = device.makeBuffer(bytes: MS, length: 4 * MS.count, options: .storageModeShared)!
        keep += [fmBuf, msBuf]
        for (pso, planeIsY) in [(psoYF, true), (psoCF, false)] {
            enc.setComputePipelineState(pso)
            enc.setTexture(CVMetalTextureGetTexture(planeIsY ? sY : sC), index: 0)
            enc.setTexture(CVMetalTextureGetTexture(planeIsY ? dY : dC), index: 1)
            for i in 0..<4 {
                let t = i < nb.count ? (planeIsY ? nb[i].0.tY : nb[i].0.tC) : (planeIsY ? sY : sC)
                enc.setTexture(CVMetalTextureGetTexture(t), index: 2 + i)
            }
            enc.setBytes(&P, length: 4 * P.count, index: 0)
            setMats(enc, record: r, index: 1)
            enc.setBytes(&FP, length: 4 * FP.count, index: 2)
            enc.setBuffer(fmBuf, offset: 0, index: 3)
            enc.setBuffer(msBuf, offset: 0, index: 4)
            let w = planeIsY ? outW : outW / 2, h = planeIsY ? outH : outH / 2
            enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        }
    } else {
        // synthetic shutter: the .spblur record at this PTS (>= 2 taps -> the *_blur kernels)
        var br = -1, nt = 0
        if !blurPTS.isEmpty, let b = nearest(blurPTS, ts), abs(blurPTS[b] - ts) <= 0.5 * frameDur {
            nt = min(blurTaps(b), blurMax)
            if nt >= 2 { br = b; P[28] = Float(nt); nBlurred += 1; nBlurTaps += nt }
        }
        func setTaps() {
            let off = blurHdr + br * blurRec + 16
            blurData.withUnsafeBytes { raw in enc.setBytes(raw.baseAddress!.advanced(by: off), length: 40 * nt, index: 2) }
        }
        enc.setComputePipelineState(br >= 0 ? psoYB : (hasMesh ? psoYm : psoY))
        enc.setTexture(CVMetalTextureGetTexture(sY), index: 0)
        enc.setTexture(CVMetalTextureGetTexture(dY), index: 1)
        enc.setBytes(&P, length: 4 * P.count, index: 0)
        setMats(enc, record: r, index: 1)
        if br >= 0 { setTaps() }
        if hasMesh { setMesh(enc, record: r, index: 2) }         // (mesh and --blur are exclusive)
        enc.dispatchThreads(MTLSize(width: outW, height: outH, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.setComputePipelineState(br >= 0 ? psoCB : (hasMesh ? psoCm : psoC))
        enc.setTexture(CVMetalTextureGetTexture(sC), index: 0)
        enc.setTexture(CVMetalTextureGetTexture(dC), index: 1)
        enc.dispatchThreads(MTLSize(width: outW / 2, height: outH / 2, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }
    var mapBuf: MTLBuffer?
    if dumpFrames.contains(si) {
        mapBuf = device.makeBuffer(length: outW * outH * 3 * 4, options: .storageModeShared)
        enc.setComputePipelineState(hasMesh ? psoMapM : psoMap)
        enc.setBuffer(mapBuf, offset: 0, index: 0)
        enc.setBytes(&P, length: 4 * P.count, index: 1)
        setMats(enc, record: r, index: 2)
        if hasMesh { setMesh(enc, record: r, index: 3) }
        enc.dispatchThreads(MTLSize(width: outW, height: outH, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }
    enc.endEncoding()
    cb.commit()
    nIn += 1
    return Pending(cb: cb, pb: dst, src: src, pts: pts, srcIdx: si, rec: r, map: mapBuf, keep: keep)
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
            guard let sb = vOut.copyNextSampleBuffer() else { eos = true; pending += drain(eos: true); continue }
            pending += ingest(sb)
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
        pending += ingest(sb)
        while pending.count >= inflight { retire(pending.removeFirst()) }
    }
    pending += drain(eos: true)
    while !pending.isEmpty { retire(pending.removeFirst()) }
}
if reader.status == .failed { die("reader failed: \(String(describing: reader.error))") }
let dt = Date().timeIntervalSince(t0)
if nOut > 0 { reportProgress(final: true) }
print(String(format: "RESULT frames=%d wall=%.3fs fps=%.1f gpu_ms_per_frame=%.2f skipped_no_plan=%d skipped_outside_window=%d worst_pts_match_ms=%.3f codec=%@ kernel=%@ in=%@ blurred=%d blur_taps_mean=%.2f",
             nOut, dt, Double(nOut) / dt, 1000 * gpuTime / Double(max(nOut, 1)), nSkipNoPlan, nSkipWindow, 1000 * worstMatch,
             noWrite ? "none" : codec, kernelName, is8 ? "8bit" : "10bit", nBlurred,
             nBlurred > 0 ? Double(nBlurTaps) / Double(nBlurred) : 0.0))
if useFill {
    print(String(format: "FILL frames=%d sources=%d missing_sources=%d ring_radius=%d ring_peak=%d missing_frames=%d",
                 nFillOut, nFillSrcUsed, nFillSrcMissing, fillRadius, ringPeak, nMissing))
}
if nOut == 0 { die("no frames rendered (plan PTS do not match the video?)") }

extension FourCharCode {
    var fourcc: String { String(bytes: [UInt8(self >> 24 & 255), UInt8(self >> 16 & 255), UInt8(self >> 8 & 255), UInt8(self & 255)], encoding: .ascii) ?? "?" }
}
