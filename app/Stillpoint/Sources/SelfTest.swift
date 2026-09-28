// Headless check that the preview compositor renders the same stabilised frame as the export renderer.
//
//   Stillpoint --selftest [--clip CLIP] [--plan PLAN.spplan] [--frames 5,120] [--out DIR]
//
// For each frame N: sprender --no-write --dump-frames N (warped 10-bit planes before encoding) vs the app's
// AVVideoCompositing compositor driven through AVAssetReaderVideoCompositionOutput at composition time PTS(N).
// Pass: PSNR(Y) and PSNR(CbCr) > 40 dB for every frame, split(0) == after bit-exactly, the "before" view really differs,
// and plan-record lookup picks the displayed frame.
import AVFoundation
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum SelfTest {
    struct Planes { let w: Int; let h: Int; let y: [UInt16]; let uv: [UInt16] }

    static func log(_ s: String) { print(s); fflush(stdout) }

    static func run(args: [String]) -> Int32 {
        func opt(_ n: String) -> String? {
            if let i = args.firstIndex(of: n), i + 1 < args.count { return args[i + 1] }
            return nil
        }
        let clip = URL(fileURLWithPath: opt("--clip") ?? Footage.o3("DJI_0026.MP4"))
        if opt("--plan") == nil { AnalysisStore.migrateLegacy() }
        let planURL = opt("--plan").map { URL(fileURLWithPath: $0) }
            ?? ([EngineConfig.analyses] + EngineConfig.legacyAnalyses).map {
                EngineConfig.analysisDir(for: clip, in: $0).appendingPathComponent("plan.spplan")
            }.first { FileManager.default.fileExists(atPath: $0.path) }
            ?? EngineConfig.analysisDir(for: clip).appendingPathComponent("plan.spplan")
        let out = URL(fileURLWithPath: opt("--out") ?? EngineConfig.scratch.appendingPathComponent("selftest").path)
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        var ok = true
        func check(_ cond: Bool, _ msg: String) {
            log((cond ? "PASS " : "FAIL ") + msg)
            if !cond { ok = false }
        }
        log("selftest clip \(clip.path)")
        log("selftest plan \(planURL.path)")
        let plan: PlanFile
        do { plan = try PlanFile(url: planURL) } catch {
            log("FAIL cannot read plan: \(error.localizedDescription) (run an analysis first, or pass --plan)")
            return 2
        }
        do {
            let engine = try WarpEngine.shared()
            log("PASS warp.metal compiled at runtime: \(engine.shaderPath) (sp_warp_luma, sp_warp_chroma + app split kernels)")
        } catch {
            log("FAIL shader compile: \(error)")
            return 2
        }
        let frames = (opt("--frames") ?? "5,\(min(plan.count - 3, 150))").split(separator: ",").compactMap { Int($0) }
            .filter { $0 >= 0 && $0 < plan.count }

        // record lookup: what AVFoundation displays at t is the last frame with PTS <= t
        for n in frames {
            let t = plan.pts[n], fd = plan.frameDuration
            check(plan.record(at: t) == n && plan.record(at: t + 0.45 * fd) == n && plan.record(at: t + 0.99 * fd) == n
                  && (n == 0 || plan.record(at: t - 0.02 * fd) == n - 1),
                  "plan.record(at:) frame \(n) (pts \(String(format: "%.6f", t)))")
        }

        var worst = Double.infinity
        for n in frames {
            let dump = out.appendingPathComponent("sprender")
            try? FileManager.default.removeItem(at: dump)
            guard let ref = runSprender(clip: clip, plan: planURL, frame: n, dumpDir: dump, outW: plan.outW, outH: plan.outH) else {
                log("FAIL sprender dump for frame \(n)")
                ok = false
                continue
            }
            let t = plan.pts[n]
            var after = WarpSettings(); after.mode = .after
            var split0 = WarpSettings(); split0.mode = .split; split0.split = 0
            var before = WarpSettings(); before.mode = .before
            guard let a = composite(clip: clip, plan: plan, time: t, settings: after, save: out.appendingPathComponent("composited_\(n).png")),
                  let s0 = composite(clip: clip, plan: plan, time: t, settings: split0, save: nil),
                  let b = composite(clip: clip, plan: plan, time: t, settings: before, save: out.appendingPathComponent("before_\(n).png")) else {
                log("FAIL compositor produced no frame at \(t)")
                ok = false
                continue
            }
            let (py, maxY, sameY) = psnr(ref.y, a.y)
            let (pc, maxC, _) = psnr(ref.uv, a.uv)
            worst = min(worst, py, pc)
            check(py > 40 && pc > 40, String(format: "frame %d compositor vs sprender: PSNR Y %@ dB, CbCr %@ dB, max |diff| %d / %d codes, %.4f%% of luma identical",
                                            n, fmtdB(py), fmtdB(pc), maxY, maxC, 100 * sameY))
            let (ps, _, _) = psnr(a.y + a.uv, s0.y + s0.uv)
            check(ps.isInfinite, "frame \(n) split kernel at 0 == sp_warp_luma/chroma output (\(fmtdB(ps)) dB)")
            let (pb, _, _) = psnr(a.y, b.y)
            check(pb < 40, "frame \(n) 'before' view differs from stabilised (\(fmtdB(pb)) dB)")
        }
        // GPU cost per composed frame (reader path, 40 frames), export filter vs playback filter
        for (name, k) in [("lanczos3", Float(0)), ("catmull-rom", Float(1))] {
            var st = WarpSettings(); st.kernel = k
            let (n, ms, fps) = throughput(clip: clip, plan: plan, start: plan.pts[min(10, plan.count - 1)], frames: 40, settings: st)
            log(String(format: "INFO compositor %@: %d frames, GPU %.2f ms/frame, %.0f fps through AVAssetReader (machine shared with other jobs)", name, n, ms, fps))
        }
        if !args.contains("--no-player") {
            let t0 = plan.pts[min(30, plan.count - 1)]
            let secs = min(2.0, max(0.5, plan.pts[plan.count - 1] - t0 - 0.2))
            let r = playerCheck(clip: clip, plan: plan, start: t0, seconds: secs)
            let pct = 100 * Double(r.delivered) / Double(max(r.expected, 1))
            check(r.delivered > 0 && r.composited > 0,
                  String(format: "AVPlayer live path (muted, headless): %d distinct composed frames of ~%d at 1.0x (%.0f%%), compositor calls %d, exact seek -> new frame in %.0f ms",
                         r.delivered, r.expected, pct, r.composited, r.seekMs))
            if pct < 90 { log("WARN real-time delivery below 90% (other GPU/decoder jobs running?)") }
        }
        log(String(format: "RESULT selftest %@ worst PSNR %@ dB over frames %@", ok ? "PASS" : "FAIL", fmtdB(worst),
                   frames.map(String.init).joined(separator: ",")))
        return ok ? 0 : 1
    }

    static func throughput(clip: URL, plan: PlanFile, start: Double, frames: Int, settings: WarpSettings) -> (Int, Double, Double) {
        let sem = DispatchSemaphore(value: 0)
        var out = (0, 0.0, 0.0)
        Task.detached {
            defer { sem.signal() }
            guard let asset = Optional(AVURLAsset(url: clip)), let info = try? await TrackInfo.load(asset),
                  let reader = try? AVAssetReader(asset: asset) else { return }
            let state = PreviewState()
            state.plan = plan
            state.settings = settings
            reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 60000),
                                           duration: CMTime(seconds: Double(frames) * info.frameDuration.seconds, preferredTimescale: 60000))
            let o = AVAssetReaderVideoCompositionOutput(videoTracks: [info.track], videoSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange])
            o.videoComposition = info.makeComposition(state: state, renderSize: CGSize(width: plan.outW, height: plan.outH))
            o.alwaysCopiesSampleData = false
            reader.add(o)
            guard reader.startReading() else { return }
            let g0 = StabilizingCompositor.gpuTime.value, c0 = StabilizingCompositor.framesRendered.value
            let t0 = Date()
            var n = 0
            while o.copyNextSampleBuffer() != nil { n += 1 }
            let dt = Date().timeIntervalSince(t0)
            let c = StabilizingCompositor.framesRendered.value - c0
            out = (n, 1000 * (StabilizingCompositor.gpuTime.value - g0) / Double(max(c, 1)), Double(n) / max(dt, 1e-6))
        }
        sem.wait()
        return out
    }

    /// Real-time playback through AVPlayer + the compositor, headless (AVPlayerItemVideoOutput instead of a layer).
    /// Runs on the main thread, spinning the run loop (AVPlayer delivers some callbacks on the main queue).
    static func playerCheck(clip: URL, plan: PlanFile, start: Double, seconds: Double) -> (delivered: Int, expected: Int, composited: Int, seekMs: Double) {
        let asset = AVURLAsset(url: clip)
        var info: TrackInfo?
        let sem = DispatchSemaphore(value: 0)
        Task.detached { info = try? await TrackInfo.load(asset); sem.signal() }
        while sem.wait(timeout: .now()) == .timedOut { RunLoop.main.run(until: Date().addingTimeInterval(0.005)) }
        guard let info else { return (0, 1, 0, 0) }
        let state = PreviewState()
        state.plan = plan
        var live = WarpSettings(); live.kernel = 1          // what PlayerController uses while playing
        state.settings = live
        let item = AVPlayerItem(asset: asset)
        if ProcessInfo.processInfo.environment["SP_NOCOMP"] == nil {
            item.videoComposition = info.makeComposition(state: state, renderSize: CGSize(width: plan.outW, height: plan.outH))
        }
        let vo = AVPlayerItemVideoOutput(pixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange])
        item.add(vo)
        let player = AVPlayer(playerItem: item)
        player.isMuted = true
        player.volume = 0
        player.automaticallyWaitsToMinimizeStalling = false
        func spin(until cond: () -> Bool, timeout: Double) {
            let end = Date().addingTimeInterval(timeout)
            while !cond() && Date() < end { RunLoop.main.run(until: Date().addingTimeInterval(0.002)) }
        }
        spin(until: { item.status != .unknown }, timeout: 10)
        if ProcessInfo.processInfo.environment["SP_DEBUG"] != nil {
            log("  debug: frameDuration \(info.frameDuration.value)/\(info.frameDuration.timescale) track \(info.track.trackID) dur \(info.duration.seconds) is8 \(info.is8bit) size \(info.size) tracks \(item.tracks.map { "\($0.assetTrack?.trackID ?? -1):\($0.assetTrack?.mediaType.rawValue ?? "?"):\($0.isEnabled)" })")
        }
        if item.status != .readyToPlay {
            log("  player item status \(item.status.rawValue): \(String(describing: item.error)) \((item.error as NSError?)?.userInfo ?? [:])")
        }
        var seekDone = false
        let c0 = StabilizingCompositor.framesRendered.value
        let ts = Date()
        player.seek(to: CMTime(seconds: start, preferredTimescale: 60000), toleranceBefore: .zero, toleranceAfter: .zero) { _ in seekDone = true }
        spin(until: { seekDone && StabilizingCompositor.framesRendered.value > c0 }, timeout: 5)
        let seekMs = Date().timeIntervalSince(ts) * 1000
        let c1 = StabilizingCompositor.framesRendered.value
        var seen = Set<Int64>()
        player.rate = 1.0
        let end = Date().addingTimeInterval(seconds)
        while Date() < end {
            let it = vo.itemTime(forHostTime: CACurrentMediaTime())
            if vo.hasNewPixelBuffer(forItemTime: it) {
                var disp = CMTime.invalid
                if vo.copyPixelBuffer(forItemTime: it, itemTimeForDisplay: &disp) != nil, disp.isValid {
                    seen.insert(disp.value * 1_000_000 / Int64(disp.timescale))
                }
            }
            RunLoop.main.run(until: Date().addingTimeInterval(0.003))
        }
        player.rate = 0
        player.replaceCurrentItem(with: nil)
        let expected = Int((seconds / plan.frameDuration).rounded())
        return (seen.count, expected, StabilizingCompositor.framesRendered.value - c1, seekMs)
    }

    static func fmtdB(_ v: Double) -> String { v.isInfinite ? "inf" : String(format: "%.2f", v) }

    static func psnr(_ a: [UInt16], _ b: [UInt16]) -> (Double, Int, Double) {
        guard a.count == b.count, !a.isEmpty else { return (0, Int.max, 0) }
        var se = 0.0, mx = 0, same = 0
        for i in 0..<a.count {
            let d = Int(a[i] >> 6) - Int(b[i] >> 6)       // 10-bit codes (MSB-aligned in 16 bits)
            se += Double(d * d)
            mx = max(mx, abs(d))
            if d == 0 { same += 1 }
        }
        let mse = se / Double(a.count)
        return (mse == 0 ? .infinity : 10 * log10(1023.0 * 1023.0 / mse), mx, Double(same) / Double(a.count))
    }

    /// sprender --no-write --dump-frames N: the warped planes exactly as they would be encoded.
    static func runSprender(clip: URL, plan: URL, frame n: Int, dumpDir: URL, outW: Int, outH: Int) -> Planes? {
        let p = Process()
        p.executableURL = EngineConfig.sprender
        p.arguments = [clip.path, plan.path, "/dev/null", "--no-write", "--start-frame", "\(n)", "--frames", "1",
                       "--dump-frames", "\(n)", "--dump-dir", dumpDir.path, "--kernel", "lanczos3", "--quiet"]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        do { try p.run() } catch { log("sprender: \(error)"); return nil }
        let outData = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        if let s = String(data: outData, encoding: .utf8) { log("  sprender: " + s.trimmingCharacters(in: .whitespacesAndNewlines)) }
        guard p.terminationStatus == 0,
              let y = try? Data(contentsOf: dumpDir.appendingPathComponent("frame\(n)_out_y.u16")),
              let uv = try? Data(contentsOf: dumpDir.appendingPathComponent("frame\(n)_out_uv.u16")),
              y.count == outW * outH * 2, uv.count == outW * outH else { return nil }
        return Planes(w: outW, h: outH, y: y.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) },
                      uv: uv.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) })
    }

    /// One composed frame through the real preview path (custom compositor + PreviewInstruction).
    static func composite(clip: URL, plan: PlanFile, time: Double, settings: WarpSettings, save: URL?) -> Planes? {
        let sem = DispatchSemaphore(value: 0)
        var result: Planes?
        Task.detached {
            defer { sem.signal() }
            do {
                let asset = AVURLAsset(url: clip)
                let info = try await TrackInfo.load(asset)
                let state = PreviewState()
                state.plan = plan
                state.settings = settings
                let comp = info.makeComposition(state: state, renderSize: CGSize(width: plan.outW, height: plan.outH))
                let reader = try AVAssetReader(asset: asset)
                let fd = info.frameDuration.seconds
                reader.timeRange = CMTimeRange(start: CMTime(seconds: time, preferredTimescale: 60000),
                                               duration: CMTime(seconds: 3 * fd, preferredTimescale: 60000))
                let o = AVAssetReaderVideoCompositionOutput(videoTracks: [info.track], videoSettings: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
                    kCVPixelBufferIOSurfacePropertiesKey as String: [String: Int]()])
                o.videoComposition = comp
                o.alwaysCopiesSampleData = false
                reader.add(o)
                guard reader.startReading() else { throw reader.error ?? WarpError(errorDescription: "reader") }
                var best: (CVPixelBuffer, Double)?
                while let sb = o.copyNextSampleBuffer() {
                    let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
                    if let pb = CMSampleBufferGetImageBuffer(sb), best == nil || abs(t - time) < abs(best!.1 - time) {
                        best = (pb, t)
                    }
                }
                reader.cancelReading()
                guard let (pb, t) = best, abs(t - time) < 1e-3 else { return }
                if let save, let cg = try? StillRenderer.cgImage(pb, maxWidth: 1920) { writePNG(cg, save) }
                result = planes(pb)
            } catch {
                log("compositor: \(error)")
            }
        }
        sem.wait()
        return result
    }

    static func planes(_ pb: CVPixelBuffer) -> Planes {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        func plane(_ i: Int, samplesPerPixel: Int) -> [UInt16] {
            let w = CVPixelBufferGetWidthOfPlane(pb, i), h = CVPixelBufferGetHeightOfPlane(pb, i)
            let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, i)
            let base = CVPixelBufferGetBaseAddressOfPlane(pb, i)!
            var out = [UInt16](); out.reserveCapacity(w * h * samplesPerPixel)
            for y in 0..<h {
                let row = base.advanced(by: y * bpr).assumingMemoryBound(to: UInt16.self)
                out.append(contentsOf: UnsafeBufferPointer(start: row, count: w * samplesPerPixel))
            }
            return out
        }
        return Planes(w: CVPixelBufferGetWidth(pb), h: CVPixelBufferGetHeight(pb),
                      y: plane(0, samplesPerPixel: 1), uv: plane(1, samplesPerPixel: 2))
    }
}

func writePNG(_ img: CGImage, _ url: URL) {
    guard let d = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(d, img, nil)
    CGImageDestinationFinalize(d)
}
