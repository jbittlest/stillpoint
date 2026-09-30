// Headless check that the preview compositor renders the same stabilised frame as the export renderer.
//
//   Stillpoint --selftest [--clip CLIP] [--plan PLAN.spplan] [--frames 5,120] [--out DIR]
//                         [--fill-plan P] [--fill-clip C] [--fill-window START,N] [--mesh-plan P] [--no-v5] [--no-player]
//
// For each frame N: sprender --no-write --dump-frames N (warped 10-bit planes before encoding) vs the app's
// AVVideoCompositing compositor driven through AVAssetReaderVideoCompositionOutput at composition time PTS(N).
// Pass: PSNR(Y) and PSNR(CbCr) > 40 dB for every frame, split(0) == after bit-exactly, the "before" view really differs,
// and plan-record lookup picks the displayed frame.
// Engine v5 (unless --no-v5): a MESH-residual plan of --clip and a full-frame FILL plan (given, or made once via the
// bridge into <out>/v5 and reused) through the same compositor: fill / mesh kernels, PSNR vs sprender, split(0) ==
// after, the mesh really moves pixels, and a fill plan never shows black borders -- neither paused (neighbouring
// frames decoded: export-identical) nor playing (cached neighbours only: the kernel's soft edge extension). The fill
// plan needs motion (on a calm clip the neighbours never see past the frame edge and the engine zooms instead): by
// default it is DJI_0034 frames 1350-1589 at a wide field of view (--fill-clip / --fill-window change that).
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
            log("PASS warp.metal compiled at runtime: \(engine.shaderPath) (\(engine.kernelNames))")
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
        if !args.contains("--no-v5") { v5(clip: clip, out: out, args: args, check: check) }
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

    // MARK: engine v5 plans (fill, mesh)

    /// A cached analysis in `dir` made with `flags` for this exact clip, else one made now through the bridge.
    static func ensurePlan(clip: URL, dir: URL, flags: [String], matches: (AnalysisParams) -> Bool) -> URL? {
        if let m = JSONIO.loadManifest(dir), let id = fileIdentity(clip), m.clip.sizeBytes == id.size,
           m.clip.mtimeNs == id.mtimeNs, matches(m.params), FileManager.default.fileExists(atPath: m.plan) {
            log("  reusing \(dir.lastPathComponent) (\(m.created ?? "?"))")
            return URL(fileURLWithPath: m.plan)
        }
        log("  analyzing \(clip.lastPathComponent) \(flags.joined(separator: " ")) via the bridge -> \(dir.path)")
        let t0 = Date()
        switch BridgeJob.runSync(["analyze", clip.path, "--out", dir.path] + flags, timeout: 1500) {
        case .success:
            log(String(format: "  analysis done in %.0f s", Date().timeIntervalSince(t0)))
            let u = dir.appendingPathComponent("plan.spplan")
            return FileManager.default.fileExists(atPath: u.path) ? u : nil
        case .failure(let e):
            log("  analysis failed: \(e.localizedDescription)")
            return nil
        }
    }

    static func blackCount(_ y: [UInt16], at idx: [Int]) -> Int { idx.reduce(0) { $0 + (y[$1] == 4096 ? 1 : 0) } }

    static func v5(clip: URL, out: URL, args: [String], check: (Bool, String) -> Void) {
        func opt(_ n: String) -> String? {
            if let i = args.firstIndex(of: n), i + 1 < args.count { return args[i + 1] }
            return nil
        }
        let dir = out.appendingPathComponent("v5", isDirectory: true)
        let stem = clip.deletingPathExtension().lastPathComponent
        let fillClip = URL(fileURLWithPath: opt("--fill-clip") ?? (opt("--fill-plan") != nil ? clip.path : Footage.o3("DJI_0034.MP4")))
        let window = (opt("--fill-window") ?? (opt("--fill-clip") == nil && opt("--fill-plan") == nil ? "1350,240" : ""))
            .split(separator: ",").compactMap { Int($0) }
        var wide = 110.0                                  // fill plan: a wide view, so the border fill has work to do
        if case .success(let obj) = BridgeJob.runSync(["probe", fillClip.path], timeout: 120),
           let p = ProbeInfo.decode(result: obj), let f = p.fov {
            wide = max(f.defaultDeg, f.maxDeg - 2)
            check(p.options != nil, "probe reports the engine's options: defaults fill=\(p.options?.defaults.fill ?? false) mesh=\(p.options?.defaults.mesh ?? false) horizon=\(p.options?.defaults.horizonLock ?? -1) timecal=\(p.options?.defaults.timecal ?? false); exclusive \(p.options?.exclusive ?? [])")
        }
        let win = window.count == 2 ? ["--start-frame", "\(window[0])", "--max-frames", "\(window[1])"] : []
        let fillStem = fillClip.deletingPathExtension().lastPathComponent + (window.count == 2 ? "-w\(window[0])+\(window[1])" : "")
        let fillURL = opt("--fill-plan").map { URL(fileURLWithPath: $0) }
            ?? ensurePlan(clip: fillClip, dir: dir.appendingPathComponent("\(fillStem)-fill"),
                          flags: ["--fill", "--no-mesh", "--fov", String(format: "%.1f", wide)] + win,
                          matches: { $0.fill == true })
        // source frame index of a plan record (DJI clips: CFR, first PTS 0; window plans start mid-clip)
        func srcIndex(_ p: PlanFile, _ r: Int) -> Int { Int((p.pts[r] / p.frameDuration).rounded()) }
        let meshURL = opt("--mesh-plan").map { URL(fileURLWithPath: $0) }
            ?? ensurePlan(clip: clip, dir: dir.appendingPathComponent("\(stem)-mesh"),
                          flags: ["--mesh", "--no-fill"], matches: { $0.mesh == true })

        // ---- full-frame fill
        if let u = fillURL, let plan = try? PlanFile(url: u), plan.hasFill {
            let clip = fillClip
            let withSrc = (0..<plan.count).filter { plan.fillCount($0) > 0 && $0 + 1 < plan.count }
            var frames: [Int] = []
            for r in withSrc.sorted(by: { plan.fillFraction($0) > plan.fillFraction($1) }) where frames.allSatisfy({ abs($0 - r) >= 30 }) {
                frames.append(r)
                if frames.count == 2 { break }
            }
            let nFilled = (0..<plan.count).filter { plan.fillFraction($0) > 0 }.count
            log("fill plan \(u.path) (\(clip.lastPathComponent), source frames \(srcIndex(plan, 0))-\(srcIndex(plan, plan.count - 1))): \(nFilled)/\(plan.count) records synthesise border pixels, \(withSrc.count) with neighbour sources, max offset \(plan.fill?.maxOffset ?? 0); testing source frames \(frames.map { srcIndex(plan, $0) })")
            check(!frames.isEmpty, "fill plan has frames that sample neighbouring source frames")
            for r in frames {
                let t = plan.pts[r], n = srcIndex(plan, r)          // n: source frame index (messages, sprender)
                let dump = out.appendingPathComponent("sprender_fill")
                try? FileManager.default.removeItem(at: dump)
                guard let ref = runSprender(clip: clip, plan: u, frame: n, dumpDir: dump, outW: plan.outW, outH: plan.outH) else {
                    check(false, "sprender dump of fill frame \(n)"); continue
                }
                var exact = WarpSettings(); exact.fill = .exact
                var split0 = exact; split0.mode = .split; split0.split = 0
                var live = WarpSettings(); live.fill = .cachedOnly; live.kernel = 0     // playing, nothing cached yet
                var plain = WarpSettings(); plain.ignoreV5 = true
                guard let a = composite(clip: clip, plan: plan, time: t, settings: exact, save: out.appendingPathComponent("fill_exact_\(n).png")),
                      let enc = StabilizingCompositor.encoded(at: t),
                      let s0 = composite(clip: clip, plan: plan, time: t, settings: split0, save: nil),
                      let e = composite(clip: clip, plan: plan, time: t, settings: live, save: out.appendingPathComponent("fill_playing_\(n).png")),
                      let pl = composite(clip: clip, plan: plan, time: t, settings: plain, save: out.appendingPathComponent("fill_as_plain_\(n).png")) else {
                    check(false, "compositor produced the fill frame \(n)"); continue
                }
                let (py, maxY, sameY) = psnr(ref.y, a.y)
                let (pc, maxC, _) = psnr(ref.uv, a.uv)
                check(py > 40 && pc > 40 && enc.kernel == "fill" && enc.fillSources == enc.fillWanted && enc.fillSources > 0,
                      String(format: "fill frame %d (%.1f%% of the frame synthesised, %d/%d neighbour frames) compositor vs sprender: PSNR Y %@ dB, CbCr %@ dB, max |diff| %d / %d, %.4f%% identical",
                             n, 100 * Double(plan.fillFraction(r)), enc.fillSources, enc.fillWanted, fmtdB(py), fmtdB(pc), maxY, maxC, 100 * sameY))
                let (ps, _, _) = psnr(a.y + a.uv, s0.y + s0.uv)
                check(ps.isInfinite, "fill frame \(n): split at 0 == the fill kernel's output (\(fmtdB(ps)) dB)")
                let blackIdx = pl.y.indices.filter { pl.y[$0] == 4096 }
                let bExact = blackCount(a.y, at: blackIdx), bLive = blackCount(e.y, at: blackIdx)
                check(!blackIdx.isEmpty && bExact <= blackIdx.count / 200 && bLive <= blackIdx.count / 200,
                      String(format: "fill frame %d: no black borders — %d px are black with the plain kernel (%.2f%% of the frame); black there: paused/exact %d, playing/soft-edge %d",
                             n, blackIdx.count, 100 * Double(blackIdx.count) / Double(pl.y.count), bExact, bLive))
                let (pe, _, _) = psnr(a.y, e.y)
                check(!pe.isInfinite, "fill frame \(n): the neighbour frames change the border vs the soft edge (\(fmtdB(pe)) dB)")
            }
        } else {
            check(false, "fill plan available (\(fillURL?.path ?? "none"))")
        }

        // ---- mesh residual
        if let u = meshURL, let plan = try? PlanFile(url: u), let m = plan.mesh {
            var mag = [Float](repeating: 0, count: plan.count)
            for r in 0..<plan.count {
                mag[r] = plan.withMesh(r) { p, n in
                    let f = p.assumingMemoryBound(to: Float.self)
                    return (0..<(n / 4)).reduce(Float(0)) { max($0, abs(f[$1])) }
                } ?? 0
            }
            let n = (1..<(plan.count - 1)).max { mag[$0] < mag[$1] } ?? 5
            log(String(format: "mesh plan %@: %dx%d vertices, largest offset %.2f px at frame %d", u.path, m.nx, m.ny, mag[n], n))
            let t = plan.pts[n]
            let dump = out.appendingPathComponent("sprender_mesh")
            try? FileManager.default.removeItem(at: dump)
            if let ref = runSprender(clip: clip, plan: u, frame: srcIndex(plan, n), dumpDir: dump, outW: plan.outW, outH: plan.outH) {
                let after = WarpSettings()
                var split0 = after; split0.mode = .split; split0.split = 0
                var plain = after; plain.ignoreV5 = true
                if let a = composite(clip: clip, plan: plan, time: t, settings: after, save: out.appendingPathComponent("mesh_\(n).png")),
                   let enc = StabilizingCompositor.encoded(at: t),
                   let s0 = composite(clip: clip, plan: plan, time: t, settings: split0, save: nil),
                   let pl = composite(clip: clip, plan: plan, time: t, settings: plain, save: nil) {
                    let (py, maxY, sameY) = psnr(ref.y, a.y)
                    let (pc, maxC, _) = psnr(ref.uv, a.uv)
                    check(py > 40 && pc > 40 && enc.kernel == "mesh",
                          String(format: "mesh frame %d compositor vs sprender: PSNR Y %@ dB, CbCr %@ dB, max |diff| %d / %d, %.4f%% identical (kernel %@)",
                                 n, fmtdB(py), fmtdB(pc), maxY, maxC, 100 * sameY, enc.kernel))
                    let (ps, _, _) = psnr(a.y + a.uv, s0.y + s0.uv)
                    check(ps.isInfinite, "mesh frame \(n): split at 0 == the mesh kernel's output (\(fmtdB(ps)) dB)")
                    let (pm, _, _) = psnr(a.y, pl.y)
                    check(!pm.isInfinite, "mesh frame \(n): the mesh offsets move pixels vs rotation-only (\(fmtdB(pm)) dB)")
                } else {
                    check(false, "compositor produced the mesh frame \(n)")
                }
            } else {
                check(false, "sprender dump of mesh frame \(n)")
            }
        } else {
            check(false, "mesh plan available (\(meshURL?.path ?? "none"))")
        }
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
                state.track = info.track
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
                if let save, let d = ProcessInfo.processInfo.environment["SP_DUMP_DIR"], let r = result {   // debugging aid
                    let base = URL(fileURLWithPath: d).appendingPathComponent(save.deletingPathExtension().lastPathComponent)
                    try? r.y.withUnsafeBytes { Data($0) }.write(to: URL(fileURLWithPath: base.path + "_y.u16"))
                    try? r.uv.withUnsafeBytes { Data($0) }.write(to: URL(fileURLWithPath: base.path + "_uv.u16"))
                }
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
