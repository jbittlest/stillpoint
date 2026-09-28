// Offscreen UI review: `Stillpoint --snapshot DIR [--scale 1] [--hero CLIP] [--long CLIP] [--time 20]`
// Builds the real views with real clips (probe via the bridge, thumbnails, cached analyses, a frame warped by the
// preview kernel) and writes PNGs with ImageRenderer. No window is shown and nothing is captured from the screen.
// UI states that need a running engine (progress, stall, errors, an independent quality block) are set directly on
// the model with illustrative values; the file names of those say so.
import AVFoundation
import SwiftUI

@MainActor
enum Snapshot {
    static func run(dir: URL, args: [String]) -> Int32 {
        func opt(_ n: String) -> String? {
            if let i = args.firstIndex(of: n), i + 1 < args.count { return args[i + 1] }
            return nil
        }
        _ = NSApplication.shared                    // fonts / appearance for offscreen rendering
        NSApp.appearance = NSAppearance(named: .darkAqua)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let scale = CGFloat(Double(opt("--scale") ?? "1") ?? 1)
        let home = NSHomeDirectory()
        let fm = FileManager.default
        let hero = URL(fileURLWithPath: opt("--hero") ?? Footage.o3("DJI_0034.MP4"))
        let heroTime = Double(opt("--time") ?? "20") ?? 20
        // a long clip for the pre-flight / long-analysis states (the 6.7-min Osmo clip on the SD card if present)
        let longCands = [opt("--long"), Footage.sdDir + "/DJI_20260927091931_0012_D.MP4",
                         Footage.oa4("DJI_20260926152149_0002_D.MP4")].compactMap { $0 }
        let long = longCands.map { URL(fileURLWithPath: $0) }.first { fm.fileExists(atPath: $0.path) }
        let others = [
            Footage.o3("DJI_0026.MP4"),
            Footage.o3("DJI_0025.MP4"),
            Footage.oa4("DJI_20260926153751_0005_D.MP4"),
            Footage.oa4("DJI_20260925151130_0003_D.MP4"),
            Footage.oa4("DJI_20260926155953_0007_D.MP4"),
        ].compactMap { p -> URL? in       // Desktop copy, else the same file on the SD card (read-only)
            let sd = Footage.sdDir + "/" + (p as NSString).lastPathComponent
            return [p, sd].first { fm.fileExists(atPath: $0) }.map { URL(fileURLWithPath: $0) }
        }

        let model = AppModel(headless: true)
        let end = Date().addingTimeInterval(60)
        while !model.migrationDone && Date() < end { RunLoop.main.run(until: Date().addingTimeInterval(0.05)) }
        model.add([hero] + (long.map { [$0] } ?? []) + others)
        for c in model.clips {
            fill(c)
            print("snapshot clip \(c.name): \(c.probe?.gyro.label ?? c.probeError ?? "?") analyzed=\(c.isAnalyzed) removable=\(c.onRemovableMedia)")
        }
        guard let heroClip = model.clips.first else { print("no clips"); return 1 }
        let longClip = long.flatMap { u in model.clips.first { $0.url == u.standardizedFileURL } }
        model.selectedID = heroClip.id

        func frame(_ c: Clip, _ t: Double, _ s: WarpSettings) -> CGImage? {
            let sem = DispatchSemaphore(value: 0)
            var img: CGImage?
            let url = c.url, plan = c.plan
            Task.detached {
                img = try? await StillRenderer.render(clip: url, plan: plan, time: t, settings: s, maxWidth: 1920)
                sem.signal()
            }
            sem.wait()
            return img
        }
        func setPlayer(_ c: Clip, _ t: Double, _ s: WarpSettings) {
            let p = c.probe
            let rs = c.plan.map { CGSize(width: $0.outW, height: $0.outH) }
                ?? CGSize(width: p?.width ?? 3840, height: p?.height ?? 2160)
            model.player.configureForSnapshot(url: c.url, duration: p?.durationS ?? 60, time: t, fps: p?.fps ?? 59.94,
                                              plan: c.plan, renderSize: rs)
            model.player.settings = s
            model.snapshotFrame = frame(c, t, s)
        }
        func render(_ name: String) {
            let view = MainView()
                .environmentObject(model)
                .environment(\.snapshotMode, true)
                .environment(\.colorScheme, .dark)
                .frame(width: 1480, height: 900)
            let r = ImageRenderer(content: view)
            r.scale = scale
            r.isOpaque = true
            guard let cg = r.cgImage else { print("render failed: \(name)"); return }
            let url = dir.appendingPathComponent(name)
            writePNG(cg, url)
            print("wrote \(url.path)")
        }

        var after = WarpSettings(); after.mode = .after
        var split = WarpSettings(); split.mode = .split; split.split = 0.5
        var before = WarpSettings(); before.mode = .before

        // 1. stabilised view of an analysed clip (jitter card: self-measured unless the report has 'quality')
        setPlayer(heroClip, heroTime, after)
        render("01_stabilized.png")
        // 2. split compare
        setPlayer(heroClip, heroTime, split)
        render("02_split_compare.png")
        // 3. pre-flight of a long clip (real probe, real free space, real speed history)
        let lc = longClip ?? model.clips.first { !$0.isAnalyzed && $0.probe?.supported == true }
        if let c = lc {
            model.selectedID = c.id
            setPlayer(c, min(30, (c.probe?.durationS ?? 60) / 3), before)
            render("03_preflight_long_clip.png")
            // 4. long analysis running, a second clip queued (illustrative progress values)
            let queued = model.clips.first { $0.id != c.id && $0.id != heroClip.id && $0.probe?.supported == true }
            model._snapshotQueue([c] + (queued.map { [$0] } ?? []))
            c.phase = .running
            queued?.phase = .queued
            c.runEstimate = model.preflight(c)?.estimate
            c.elapsed = 312
            c.progress = BridgeProgress(stage: "measure", label: "Measuring jitter", fraction: 0.27,
                                        message: "pass 1 of 3 · 6512/24120 frames", elapsed: 312, eta: 845)
            c.notices = model.preflight(c)?.frameCacheFits == false
                ? ["Not enough free space for the frame cache (16.7 GB needed, 15.0 GB free): frames are decoded again for each pass, which is slower."] : []
            render("04_analyzing_long_clip_illustrative.png")
            // 5. stall watchdog
            c.stallSeconds = 72
            c.engineBusy = true
            render("05_stalled_illustrative.png")
            c.engineBusy = false
            c.stallSeconds = 128
            render("05b_stalled_idle_illustrative.png")
            // 6. error panel after an abnormal exit
            c.stallSeconds = 0
            c.progress = nil
            c.notices = []
            c.phase = .failed(FailureInfo(
                title: "The engine stopped unexpectedly",
                message: "The analysis process ended without a result (stopped by SIGKILL — killed, e.g. by the system when memory ran out). The last lines it printed are below.",
                details: ["[stillpoint] DJI_20260927091931_0012_D.MP4: DJI Osmo Action 4, 24120 frames, imu 1000 Hz",
                          "[stillpoint] crop: out_fx 1771.2 (hfov 94.5°)",
                          "Traceback (most recent call last):",
                          "  File \"engine/stillpoint/pipeline.py\", line 834, in _analyze",
                          "OSError: [Errno 28] No space left on device"]))
            queued?.phase = .idle
            model._snapshotQueue([])
            render("06_error_panel_illustrative.png")
            c.phase = .idle
        }
        // 7. independent measurement block: a real analysis with report['quality'] if there is one, else illustrative
        if let qc = model.clips.first(where: { $0.manifest?.summary?.quality != nil }) {
            model.selectedID = qc.id
            setPlayer(qc, min(heroTime, (qc.probe?.durationS ?? 4) / 2), after)
            render("07_quality_block.png")
        } else if let m = heroClip.manifest, m.summary?.quality == nil {
            model.selectedID = heroClip.id
            setPlayer(heroClip, heroTime, after)
            var s = m.summary ?? JitterSummary()
            s.quality = QualitySummary(method: "eval.jitter_metrics on the rendered output (illustrative values)", units: "px @1080p",
                                       metrics: [QualityMetric(key: "hf", label: "Shake above 2 Hz", original: 1.84, stabilized: 0.21),
                                                 QualityMetric(key: "calm", label: "Calm cruise", original: 0.62, stabilized: 0.12),
                                                 QualityMetric(key: "b8_30", label: "Fine jitter 8–30 Hz", original: 0.48, stabilized: 0.09)],
                                       windows: nil)
            var m2 = m
            m2.summary = s
            heroClip.manifest = m2
            render("07_quality_block_illustrative.png")
            heroClip.manifest = m
        }
        // 8. unsupported clip (in-camera EIS)
        if let c = model.clips.first(where: { $0.probe?.gyro.level == "unsupported" }) {
            model.selectedID = c.id
            setPlayer(c, 2, before)
            render("08_unsupported.png")
        }
        // 9. settings changed + export queue
        model.selectedID = heroClip.id
        setPlayer(heroClip, heroTime + 6, after)
        heroClip.smoothness = 1.35
        let outDir = URL(fileURLWithPath: "\(home)/Movies/Stillpoint")
        let jobs: [(Clip, ExportJob.State, Double)] = model.clips.filter { $0.isAnalyzed }.prefix(3).enumerated().map { i, c in
            (c, i == 0 ? .running : (i == 1 ? .done : .queued), i == 0 ? 0.42 : 1)
        }
        for (c, st, f) in jobs {
            let e = ExportJob(clip: c, preset: .hevc10, output: outDir.appendingPathComponent(c.url.deletingPathExtension().lastPathComponent + "_stillpoint.mov"),
                              planPath: c.manifest?.plan ?? "")
            e.state = st
            e.fraction = f
            e.eta = 38
            if st == .running { e.message = "1272/3029 frames · 61 fps" }
            if st == .done { e.bytes = 1_120_000_000; e.seconds = 51 }
            model.exports.append(e)
        }
        render("09_settings_changed_exports.png")
        heroClip.smoothness = heroClip.manifest?.params.smoothness ?? 1
        model.exports.removeAll()
        // 10. empty state
        let empty = AppModel(headless: true)
        let v = MainView().environmentObject(empty).environment(\.snapshotMode, true).environment(\.colorScheme, .dark)
            .frame(width: 1480, height: 900)
        let r = ImageRenderer(content: v)
        r.scale = scale
        if let cg = r.cgImage { writePNG(cg, dir.appendingPathComponent("10_empty.png")); print("wrote 10_empty.png") }
        return 0
    }

    /// Synchronous probe + thumbnail for offscreen rendering.
    static func fill(_ c: Clip) {
        switch BridgeJob.runSync(["probe", c.url.path], timeout: 120) {
        case .success(let obj):
            c.probe = ProbeInfo.decode(result: obj)
            if c.manifest == nil, let d = c.probe?.fov?.defaultDeg { c.fovDeg = d }
        case .failure(let e):
            c.probeError = e.localizedDescription
        }
        c.probing = false
        let sem = DispatchSemaphore(value: 0)
        var img: CGImage?
        let url = c.url
        Task.detached {
            let asset = AVURLAsset(url: url)
            let gen = AVAssetImageGenerator(asset: asset)
            gen.maximumSize = CGSize(width: 480, height: 270)
            gen.appliesPreferredTrackTransform = true
            gen.requestedTimeToleranceBefore = CMTime(value: 1, timescale: 2)
            gen.requestedTimeToleranceAfter = CMTime(value: 1, timescale: 2)
            let dur = (try? await asset.load(.duration).seconds) ?? 0
            img = try? await gen.image(at: CMTime(seconds: min(max(dur * 0.3, 0), 8), preferredTimescale: 600)).image
            sem.signal()
        }
        sem.wait()
        c.thumbnail = img
    }
}
