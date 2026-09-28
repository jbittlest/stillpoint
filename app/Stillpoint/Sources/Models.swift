// App state: clips (probe, thumbnail, cached analysis), the analysis queue (one clip at a time, with a stall
// watchdog), pre-flight checks, the export queue.
import AVFoundation
import AppKit
import SwiftUI
import UniformTypeIdentifiers

enum ExportPreset: String, CaseIterable, Identifiable, Codable {
    case hevc10, hevcFast, prores
    var id: String { rawValue }
    var title: String {
        switch self {
        case .hevc10: return "HEVC 10-bit"
        case .hevcFast: return "HEVC Fast"
        case .prores: return "ProRes 422 HQ"
        }
    }
    var short: String {
        switch self {
        case .hevc10: return "HEVC 10-bit"
        case .hevcFast: return "HEVC Fast"
        case .prores: return "ProRes HQ"
        }
    }
    var detail: String {
        switch self {
        case .hevc10: return "180 Mb/s, best quality. For upload and archive."
        case .hevcFast: return "180 Mb/s, speed-priority encoder. Quick turnarounds."
        case .prores: return "Edit-ready intraframe. Large files (≈13 GB/min at 4K60)."
        }
    }
    var codec: String {
        switch self {
        case .hevc10: return "hevc10"
        case .hevcFast: return "hevc10-speed"
        case .prores: return "prores"
        }
    }
}

/// Why an analysis did not finish — shown in the error panel with the engine's last stderr lines.
struct FailureInfo: Equatable {
    var title: String
    var message: String
    var details: [String] = []
    var canRetry = true
}

enum AnalysisPhase: Equatable {
    case idle, queued, running, failed(FailureInfo), cancelled
}

/// What the user sees before starting an analysis.
struct Preflight: Equatable {
    enum Level: Equatable { case info, warn, block }
    struct Item: Equatable, Hashable { var level: Level; var icon: String; var text: String }
    var duration: Double
    var estimate: Double
    var estimateBasis: String
    var freeBytes: Int64?
    var frameCacheBytes: Int64
    var tempBytes: Int64
    var minFreeBytes: Int64
    var frameCacheFits: Bool
    var items: [Item]
    var blocked: Bool { items.contains { $0.level == .block } }
}

@MainActor
final class Clip: ObservableObject, Identifiable {
    let id = UUID()
    let url: URL
    @Published var probe: ProbeInfo?
    @Published var probing = true
    @Published var probeError: String?
    @Published var thumbnail: CGImage?
    @Published var manifest: AnalysisManifest?
    @Published var plan: PlanFile?
    @Published var phase: AnalysisPhase = .idle
    @Published var progress: BridgeProgress?
    @Published var smoothness: Double = 1.0
    @Published var fovDeg: Double = 100
    @Published var horizonLock = false
    /// Seconds since the running analysis last reported progress (published once it passes the stall threshold).
    @Published var stallSeconds = 0
    /// CPU activity of the engine's process group over the last few seconds (nil = not sampled yet).
    @Published var engineBusy: Bool?
    @Published var notices: [String] = []
    @Published var elapsed: Double = 0
    /// Source volume facts (removable media warning).
    var volumeName: String?
    var onRemovableMedia = false
    var job: BridgeJob?
    var lastLog: [String] = []
    var runStartedAt: Date?
    var lastProgressAt: Date?
    var runEstimate: Double?

    init(url: URL) { self.url = url }

    var name: String { url.lastPathComponent }
    /// Short list name: DJI_20260926153751_0005_D.MP4 -> DJI_0005_D (the timestamp is in the tooltip); DJI_0034.MP4 -> DJI_0034.
    var displayName: String {
        let stem = url.deletingPathExtension().lastPathComponent
        let parts = stem.split(separator: "_")
        if parts.count == 4, parts[0] == "DJI", parts[1].count == 14, parts[1].allSatisfy(\.isNumber) {
            return "DJI_\(parts[2])_\(parts[3])"
        }
        return stem
    }
    /// "Sep 26, 15:37" from DJI's timestamped file names.
    var recordedLabel: String? {
        let parts = url.deletingPathExtension().lastPathComponent.split(separator: "_")
        guard parts.count == 4, parts[1].count == 14 else { return nil }
        let f = DateFormatter(); f.dateFormat = "yyyyMMddHHmmss"
        guard let d = f.date(from: String(parts[1])) else { return nil }
        let o = DateFormatter(); o.dateFormat = "MMM d, HH:mm"
        return o.string(from: d)
    }
    var analysisDir: URL { EngineConfig.analysisDir(for: url) }
    var isAnalyzed: Bool { manifest != nil && plan != nil }
    var isBusy: Bool { phase == .running || phase == .queued }
    var isStalled: Bool { phase == .running && stallSeconds >= Int(AppModel.stallThreshold) }
    var failure: FailureInfo? { if case .failed(let f) = phase { return f }; return nil }
    var fovRange: ClosedRange<Double> {
        guard let f = probe?.fov, f.maxDeg > f.minDeg else { return 85...115 }
        return f.minDeg...f.maxDeg
    }
    var settingsChanged: Bool {
        guard let m = manifest else { return false }
        return abs(m.params.smoothness - smoothness) > 0.005 || abs((m.params.fovDeg ?? fovDeg) - fovDeg) > 0.05
            || m.params.horizonLock != horizonLock
    }
    var canAnalyze: Bool { probe?.supported == true && !isBusy }
    /// Time left: the engine's own ETA once it has one, before that the pre-flight estimate minus elapsed time.
    var etaSeconds: Double? {
        if let e = progress?.eta { return e }
        guard let est = runEstimate, phase == .running else { return nil }
        return max(est - elapsed, 0)
    }
    var etaIsEngine: Bool { progress?.eta != nil }
}

@MainActor
final class ExportJob: ObservableObject, Identifiable {
    enum State: Equatable { case queued, running, done, failed(String), cancelled }
    let id = UUID()
    let clip: Clip
    let preset: ExportPreset
    let output: URL
    let planPath: String
    @Published var state: State = .queued
    @Published var fraction: Double = 0
    @Published var message = ""
    @Published var eta: Double?
    @Published var bytes: Int64?
    @Published var seconds: Double?
    var job: BridgeJob?

    init(clip: Clip, preset: ExportPreset, output: URL, planPath: String) {
        self.clip = clip
        self.preset = preset
        self.output = output
        self.planPath = planPath
    }
}

/// Analysis speed on this Mac: seconds of analysis per second of clip, from the timings of previous runs.
struct SpeedHistory: Codable {
    struct Entry: Codable { var durationS: Double; var seconds: Double; var frameCache: Bool?; var date: Date; var clip: String? }
    var entries: [Entry] = []
    static let defaultRatio = 3.0          // measured on the fixed engine: ~160 s per 50 s O3 clip
    static let overhead = 30.0             // fixed start-up / path / write cost of a run (a 4.5 s clip takes ~45 s)
    static let minDuration = 15.0          // shorter clips are dominated by the fixed overhead: they say little about speed

    static func load() -> SpeedHistory {
        guard let d = try? Data(contentsOf: EngineConfig.speedHistory) else { return SpeedHistory() }
        let dec = JSONDecoder(); dec.dateDecodingStrategy = .iso8601
        return (try? dec.decode(SpeedHistory.self, from: d)) ?? SpeedHistory()
    }
    func save() {
        let enc = JSONEncoder(); enc.dateEncodingStrategy = .iso8601; enc.outputFormatting = [.prettyPrinted]
        guard let d = try? enc.encode(self) else { return }
        try? FileManager.default.createDirectory(at: EngineConfig.supportRoot, withIntermediateDirectories: true)
        try? d.write(to: EngineConfig.speedHistory, options: .atomic)
    }
    mutating func add(durationS: Double, seconds: Double, frameCache: Bool?, clip: String) {
        guard durationS > 0, seconds > 0 else { return }
        entries.append(Entry(durationS: durationS, seconds: seconds, frameCache: frameCache, date: Date(), clip: clip))
        if entries.count > 40 { entries.removeFirst(entries.count - 40) }
    }
    /// Timings of earlier analyses on this Mac (report.json of cached analyses made by the current, frame-cache engine
    /// — reports with timings.decode_s; older engines were 3-10x slower and would mislead).
    static func seedFromReports() -> [Entry] {
        let fm = FileManager.default
        var out: [Entry] = []
        for d in (try? fm.contentsOfDirectory(at: EngineConfig.analyses, includingPropertiesForKeys: nil)) ?? [] {
            guard let data = try? Data(contentsOf: d.appendingPathComponent("report.json")),
                  let r = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  let t = r["timings"] as? [String: Any], t["decode_s"] != nil || t["x_realtime"] != nil,
                  let total = t["total_s"] as? Double, let n = r["n_frames"] as? Int, let fps = r["fps"] as? Double, fps > 0
            else { continue }
            let man = (try? Data(contentsOf: d.appendingPathComponent("stillpoint_app.json")))
                .flatMap { (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any] }
            let secs = (man?["seconds"] as? Double) ?? total
            let date = (try? fm.attributesOfItem(atPath: d.appendingPathComponent("report.json").path)[.modificationDate] as? Date) ?? Date()
            out.append(Entry(durationS: Double(n) / fps, seconds: secs, frameCache: true, date: date, clip: d.lastPathComponent))
        }
        return out.sorted { $0.date < $1.date }
    }

    /// (seconds of analysis per clip second beyond the fixed overhead, how many runs it is based on):
    /// median over the latest 6 runs, longer clips first (they measure the per-frame speed best).
    func ratio(frameCache: Bool) -> (Double, Int) {
        let ok = entries.filter { $0.durationS >= Self.minDuration && ($0.frameCache ?? frameCache) == frameCache }
            .suffix(12).sorted { $0.durationS > $1.durationS }.prefix(6)
        guard !ok.isEmpty else { return (Self.defaultRatio, 0) }
        let r = ok.map { max(0.5, ($0.seconds - Self.overhead) / $0.durationS) }.sorted()
        return (r[r.count / 2], ok.count)
    }
    func estimate(duration: Double, frameCache: Bool) -> (Double, Int) {
        let (r, n) = ratio(frameCache: frameCache)
        return (Self.overhead + r * duration, n)
    }
}

@MainActor
final class AppModel: ObservableObject {
    @Published var clips: [Clip] = []
    @Published var selectedID: UUID? {
        didSet { if oldValue != selectedID { selectionChanged() } }
    }
    @Published var exports: [ExportJob] = []
    @Published var preset: ExportPreset {
        didSet { Self.prefs.set(preset.rawValue, forKey: "exportPreset") }
    }
    @Published var outputFolder: URL {
        didSet { Self.prefs.set(outputFolder.path, forKey: "outputFolder") }
    }
    @Published var enginePath: String {
        didSet {
            EngineConfig.rootPath = enginePath
            WarpEngine.reset()
            engineProblems = EngineConfig.problems()
        }
    }
    @Published var engineProblems: [String] = []
    @Published var dropTargeted = false
    /// Snapshot mode only: the still (rendered by the preview kernel) shown in place of the player layer.
    @Published var snapshotFrame: CGImage?
    /// Clips waiting for / running analysis, in order (for the sidebar queue panel).
    @Published private(set) var analysisQueueView: [Clip] = []
    /// The one-time copy of the old iCloud-synced analysis cache has finished.
    @Published private(set) var migrationDone = false
    let player = PlayerController()
    var speed = SpeedHistory.load()

    /// A running analysis that has not reported progress for this long shows the "Still working" banner.
    static var stallThreshold: TimeInterval = 45
    static let minFreeBytes: Int64 = 3_000_000_000
    static let warnFreeBytes: Int64 = 10_000_000_000

    private var probeQueue: [Clip] = []
    private var probesRunning = 0
    private var analysisQueue: [Clip] = []
    private(set) var analysisRunning: Clip?
    private var exportRunning: ExportJob?
    private var watchdog: Timer?
    private var cpuSample: (Date, [pid_t: Double])?
    private var allJobs: [BridgeJob] = []
    /// Headless snapshot/selftest mode: never start processes or players.
    var headless = false

    /// Where the export preset / output folder live. The headless modes (--selftest-app, --snapshot) switch this to a
    /// throwaway suite so a test (even one killed half-way) can never change the user's real settings.
    static var prefs: UserDefaults = .standard
    static func useHeadlessPrefs() {
        let name = "com.jimmybittleston.stillpoint.headless"
        UserDefaults().removePersistentDomain(forName: name)
        prefs = UserDefaults(suiteName: name) ?? .standard
    }

    init(headless: Bool = false) {
        self.headless = headless
        let d = Self.prefs
        preset = ExportPreset(rawValue: d.string(forKey: "exportPreset") ?? "") ?? .hevc10
        let movies = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Movies")
        outputFolder = URL(fileURLWithPath: d.string(forKey: "outputFolder") ?? movies.appendingPathComponent("Stillpoint").path)
        enginePath = EngineConfig.rootPath
        engineProblems = EngineConfig.problems()
        migrateInBackground()
    }

    var selected: Clip? { clips.first { $0.id == selectedID } }
    var analyzedClips: [Clip] { clips.filter { $0.isAnalyzed } }

    /// Copy analyses from the old iCloud-synced cache (work/app/analyses) once, off the main thread (reading the
    /// synced Desktop can block), then pick them up for clips already in the list.
    private func migrateInBackground() {
        let seedHistory = !FileManager.default.fileExists(atPath: EngineConfig.speedHistory.path)
        Task.detached(priority: .utility) {
            let moved = AnalysisStore.migrateLegacy()
            let seeds = seedHistory ? SpeedHistory.seedFromReports() : []
            await MainActor.run {
                if !seeds.isEmpty && self.speed.entries.isEmpty {
                    self.speed.entries = seeds
                    self.speed.save()
                }
                if moved.isEmpty { self.migrationDone = true }
            }
            guard !moved.isEmpty else { return }
            await MainActor.run {
                for c in self.clips where c.manifest == nil && !c.isBusy { self.loadCachedAnalysis(c) }
                if let c = self.selected, c.isAnalyzed, !self.headless { self.player.setPlan(c.plan) }
                self.migrationDone = true
            }
        }
    }

    // MARK: adding clips

    static let videoExtensions: Set<String> = ["mp4", "mov", "m4v"]

    func add(_ urls: [URL]) {
        var files: [URL] = []
        let fm = FileManager.default
        for u in urls {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: u.path, isDirectory: &isDir) else { continue }
            if isDir.boolValue {
                let items = (try? fm.contentsOfDirectory(at: u, includingPropertiesForKeys: nil)) ?? []
                files += items.filter { Self.videoExtensions.contains($0.pathExtension.lowercased()) && !$0.lastPathComponent.hasPrefix("._") }
                    .sorted { $0.lastPathComponent < $1.lastPathComponent }
            } else if Self.videoExtensions.contains(u.pathExtension.lowercased()) {
                files.append(u)
            }
        }
        var firstNew: Clip?
        for f in files {
            let std = f.standardizedFileURL
            if clips.contains(where: { $0.url.standardizedFileURL == std }) { continue }
            let c = Clip(url: std)
            if let v = try? std.resourceValues(forKeys: [.volumeIsRemovableKey, .volumeIsEjectableKey, .volumeIsInternalKey,
                                                         .volumeLocalizedNameKey, .volumeIsRootFileSystemKey]) {
                c.volumeName = v.volumeLocalizedName
                c.onRemovableMedia = (v.volumeIsRootFileSystem != true)
                    && (v.volumeIsRemovable == true || v.volumeIsEjectable == true || v.volumeIsInternal == false)
            }
            clips.append(c)
            firstNew = firstNew ?? c
            loadCachedAnalysis(c)
            if !headless {
                probeQueue.append(c)
                loadThumbnail(c)
            }
        }
        pumpProbes()
        if let firstNew, selectedID == nil || selected == nil { selectedID = firstNew.id }
    }

    func remove(_ clip: Clip) {
        cancelAnalysis(clip)
        clips.removeAll { $0.id == clip.id }
        if selectedID == clip.id { selectedID = clips.first?.id }
    }

    func openPanel() {
        let p = NSOpenPanel()
        p.allowsMultipleSelection = true
        p.canChooseDirectories = true
        p.allowedContentTypes = [.mpeg4Movie, .quickTimeMovie, .movie]
        p.prompt = "Add Clips"
        p.message = "Choose DJI clips recorded with EIS off"
        if p.runModal() == .OK { add(p.urls) }
    }

    func chooseOutputFolder() {
        let p = NSOpenPanel()
        p.canChooseFiles = false
        p.canChooseDirectories = true
        p.canCreateDirectories = true
        p.directoryURL = outputFolder
        p.prompt = "Choose"
        if p.runModal() == .OK, let u = p.url { outputFolder = u }
    }

    private func track(_ job: BridgeJob) {
        allJobs.removeAll { !$0.isRunning && !$0.groupAlive }
        allJobs.append(job)
        startedGroups.append(job.pid)
    }
    /// Every process group this model started (tests check none of them survives).
    private(set) var startedGroups: [pid_t] = []

    private func pumpProbes() {
        while probesRunning < 2, !probeQueue.isEmpty {
            let c = probeQueue.removeFirst()
            probesRunning += 1
            let job = BridgeJob(["probe", c.url.path])
            job.onEvent = { [weak c] ev in
                guard let c else { return }
                switch ev {
                case .result(let obj, _):
                    if let p = ProbeInfo.decode(result: obj) {
                        c.probe = p
                        if c.manifest == nil {
                            c.fovDeg = p.fov?.defaultDeg ?? 100
                        }
                        c.fovDeg = c.fovDeg.clamped(c.fovRange.lowerBound, c.fovRange.upperBound)
                    } else {
                        c.probeError = "Could not read clip info"
                    }
                case .error(_, let msg): c.probeError = msg
                default: break
                }
            }
            job.onExit = { [weak self, weak c] exit in
                guard let self else { return }
                if let c {
                    c.probing = false
                    if !exit.sawTerminal && c.probe == nil {
                        c.probeError = "Engine failed to start (\(exit.describe)). " + (exit.stderrTail.last ?? "")
                    }
                }
                self.probesRunning -= 1
                self.pumpProbes()
            }
            do { try job.start(); track(job) } catch {
                c.probing = false
                c.probeError = "Cannot run the engine: \(error.localizedDescription)"
                probesRunning -= 1
            }
        }
    }

    private func loadThumbnail(_ c: Clip) {
        let url = c.url
        Task.detached(priority: .utility) {
            let asset = AVURLAsset(url: url)
            let gen = AVAssetImageGenerator(asset: asset)
            gen.maximumSize = CGSize(width: 480, height: 270)
            gen.appliesPreferredTrackTransform = true
            gen.requestedTimeToleranceBefore = CMTime(value: 1, timescale: 2)
            gen.requestedTimeToleranceAfter = CMTime(value: 1, timescale: 2)
            let dur = (try? await asset.load(.duration).seconds) ?? 0
            let t = CMTime(seconds: min(max(dur * 0.3, 0), 8), preferredTimescale: 600)
            if let (img, _) = try? await gen.image(at: t) {
                await MainActor.run { c.thumbnail = img }
            }
        }
    }

    /// Load a finished analysis from the per-clip cache, if it belongs to this exact file.
    func loadCachedAnalysis(_ c: Clip) {
        guard let m = JSONIO.loadManifest(c.analysisDir), let id = fileIdentity(c.url),
              m.clip.sizeBytes == id.size, m.clip.mtimeNs == id.mtimeNs else { return }
        guard let plan = try? PlanFile(url: URL(fileURLWithPath: m.plan)) else { return }
        c.manifest = m
        c.plan = plan
        c.smoothness = m.params.smoothness
        if let f = m.params.fovDeg { c.fovDeg = f }
        c.horizonLock = m.params.horizonLock
    }

    // MARK: selection / preview

    private func selectionChanged() {
        guard !headless else { return }
        if let c = selected { player.load(url: c.url, plan: c.plan) } else { player.unload() }
    }

    // MARK: pre-flight

    /// Clip length, time estimate from this Mac's measured speed, scratch space and source-volume checks.
    func preflight(_ c: Clip) -> Preflight? {
        guard let p = c.probe else { return nil }
        let free = freeBytes(at: EngineConfig.supportRoot)
        let cache = p.scratch?.frameCacheBytes ?? 0
        let temp = max(p.scratch?.tempBytes ?? 0, cache)
        let minFree = max(p.scratch?.minFreeBytes ?? 0, Self.minFreeBytes)
        let fits = free.map { $0 - cache >= minFree } ?? true
        let (estimate, n) = speed.estimate(duration: p.durationS, frameCache: cache > 0 && fits)
        var basis = n > 0 ? "from \(n) recent analys\(n == 1 ? "is" : "es") on this Mac"
                          : "30 s + 3× the clip length until this Mac has timed runs"
        var items: [Preflight.Item] = []
        if let free {
            if free < minFree {
                items.append(.init(level: .block, icon: "externaldrive.badge.xmark",
                                   text: "Only \(Fmt.gb(free)) free — analysis needs at least \(Fmt.gb(minFree)). Free up disk space first."))
            } else if !fits {
                items.append(.init(level: .warn, icon: "internaldrive",
                                   text: "Not enough room for the \(Fmt.gb(cache)) frame cache (\(Fmt.gb(free)) free): frames will be decoded again for each pass, so it will take longer than estimated."))
                if n == 0 { basis += "; longer without the frame cache" }
            } else if free < Self.warnFreeBytes {
                items.append(.init(level: .warn, icon: "internaldrive",
                                   text: "Disk is nearly full: \(Fmt.gb(free)) free\(temp > 0 ? "; the analysis uses up to \(Fmt.gb(temp)) of it temporarily" : "")."))
            }
        }
        if c.onRemovableMedia {
            items.append(.init(level: .warn, icon: "sdcard",
                               text: "The clip is on \(c.volumeName.map { "“\($0)”" } ?? "removable media"). The analysis reads it directly — keep it connected. Copying the clip to this Mac first is faster."))
        }
        return Preflight(duration: p.durationS, estimate: estimate, estimateBasis: basis, freeBytes: free,
                         frameCacheBytes: cache, tempBytes: temp, minFreeBytes: minFree, frameCacheFits: fits, items: items)
    }

    // MARK: analysis queue

    func analyze(_ c: Clip) {
        guard c.canAnalyze else { return }
        if let pf = preflight(c), pf.blocked {
            c.phase = .failed(FailureInfo(title: "Not enough disk space",
                                          message: pf.items.first { $0.level == .block }?.text ?? "Free up disk space first.",
                                          canRetry: true))
            return
        }
        c.phase = .queued
        c.progress = nil
        c.stallSeconds = 0
        c.notices = []
        analysisQueue.append(c)
        refreshQueueView()
        pumpAnalysis()
    }

    func retry(_ c: Clip) {
        if case .failed = c.phase { c.phase = .idle }
        analyze(c)
    }

    func cancelAnalysis(_ c: Clip) {
        if let i = analysisQueue.firstIndex(where: { $0.id == c.id }) {
            analysisQueue.remove(at: i)
            c.phase = .idle
            refreshQueueView()
        }
        if analysisRunning?.id == c.id { c.job?.cancel() }
    }

    /// Snapshot mode only: show these clips in the queue panel.
    func _snapshotQueue(_ cs: [Clip]) { if headless { analysisQueueView = cs } }

    private func refreshQueueView() {
        analysisQueueView = (analysisRunning.map { [$0] } ?? []) + analysisQueue
    }

    private func pumpAnalysis() {
        guard analysisRunning == nil, !analysisQueue.isEmpty, !headless else { refreshQueueView(); return }
        let c = analysisQueue.removeFirst()
        analysisRunning = c
        c.phase = .running
        c.lastLog = []
        c.runStartedAt = Date()
        c.lastProgressAt = nil
        c.elapsed = 0
        c.stallSeconds = 0
        c.engineBusy = nil
        c.runEstimate = preflight(c)?.estimate
        var args = ["analyze", c.url.path, "--out", c.analysisDir.path,
                    "--smoothness", String(format: "%.3f", c.smoothness), "--fov", String(format: "%.2f", c.fovDeg)]
        if c.horizonLock && (c.probe?.horizonLockSupported ?? false) { args.append("--horizon-lock") }
        let job = BridgeJob(args)
        c.job = job
        var failure: (String, String)?
        var cancelled = false
        job.onEvent = { [weak c] ev in
            guard let c else { return }
            switch ev {
            case .progress(let p):
                c.progress = p
                c.lastProgressAt = Date()
                if c.stallSeconds != 0 { c.stallSeconds = 0 }
            case .log(let s):
                c.lastLog.append(s)
                if c.lastLog.count > 60 { c.lastLog.removeFirst() }
            case .notice(_, let m): c.notices.append(m)
            case .result: break
            case .error(let code, let msg): failure = (code, msg)
            case .cancelled: cancelled = true
            }
        }
        job.onExit = { [weak self, weak c, weak job] exit in
            guard let self else { return }
            if let c {
                c.job = nil
                if cancelled || exit.cancelRequested {
                    c.phase = .cancelled
                } else if let (code, msg) = failure {
                    c.phase = .failed(FailureInfo(title: code == "disk_full" ? "Not enough disk space" : "Analysis failed",
                                                  message: msg, details: Array(exit.stderrTail.suffix(14))))
                } else if !exit.succeeded || !exit.sawTerminal {
                    c.phase = .failed(FailureInfo(
                        title: "The engine stopped unexpectedly",
                        message: "The analysis process ended without a result (\(exit.describe)). The last lines it printed are below.",
                        details: Array(exit.stderrTail.suffix(14))))
                } else {
                    c.phase = .idle
                    self.loadCachedAnalysis(c)
                    if !c.isAnalyzed {
                        c.phase = .failed(FailureInfo(title: "Analysis finished but its plan could not be loaded",
                                                      message: c.analysisDir.path, details: Array(exit.stderrTail.suffix(8))))
                    } else {
                        self.recordSpeed(c)
                    }
                    if self.selectedID == c.id { self.player.setPlan(c.plan) }
                }
                c.progress = nil
                c.stallSeconds = 0
                c.engineBusy = nil
            }
            // a killed bridge cannot remove its scratch (temporary job files can be GBs): do it here
            if let pid = job?.pid, pid > 0, !exit.succeeded {
                DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + BridgeJob.killDelay + 1) {
                    for d in EngineConfig.scratchDirs(forBridgePID: pid) { try? FileManager.default.removeItem(at: d) }
                }
            }
            self.analysisRunning = nil
            self.cpuSample = nil
            self.pumpAnalysis()
            self.updateWatchdog()
        }
        do {
            try job.start()
            track(job)
        } catch {
            c.phase = .failed(FailureInfo(title: "Cannot run the engine", message: error.localizedDescription))
            c.job = nil
            analysisRunning = nil
            pumpAnalysis()
        }
        refreshQueueView()
        updateWatchdog()
    }

    private func recordSpeed(_ c: Clip) {
        guard let m = c.manifest else { return }
        let dur = m.timing?.durationS ?? c.probe?.durationS ?? 0
        let secs = m.timing?.wallS ?? m.seconds ?? 0
        speed.add(durationS: dur, seconds: secs, frameCache: m.timing?.frameCache, clip: c.name)
        speed.save()
    }

    // MARK: stall watchdog

    private func updateWatchdog() {
        if analysisRunning != nil && watchdog == nil {
            let t = Timer(timeInterval: 1, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated { self?.watchdogTick() }
            }
            RunLoop.main.add(t, forMode: .common)
            watchdog = t
        } else if analysisRunning == nil, let t = watchdog {
            t.invalidate()
            watchdog = nil
        }
    }

    private func watchdogTick() {
        guard let c = analysisRunning, c.phase == .running, let start = c.runStartedAt else { return }
        let now = Date()
        c.elapsed = now.timeIntervalSince(start)
        let quiet = now.timeIntervalSince(c.lastProgressAt ?? start)
        let s = quiet >= Self.stallThreshold ? Int(quiet) : 0
        if s != c.stallSeconds { c.stallSeconds = s }
        // CPU activity of the whole process group (bridge, measurement workers, ffmpeg) over ~3 s
        guard let pid = c.job?.pid, pid > 0 else { return }
        if let (t0, before) = cpuSample {
            guard now.timeIntervalSince(t0) >= 3 else { return }
            let after = GroupSpawn.cpuSeconds(pid)
            let used = after.reduce(0.0) { acc, kv in acc + max(0, kv.value - (before[kv.key] ?? kv.value)) }
            let busy = used / now.timeIntervalSince(t0) > 0.05
            if c.engineBusy != busy { c.engineBusy = busy }
            cpuSample = (now, after)
        } else {
            cpuSample = (now, GroupSpawn.cpuSeconds(pid))
        }
    }

    // MARK: export queue

    func export(_ cs: [Clip]) {
        let fm = FileManager.default
        try? fm.createDirectory(at: outputFolder, withIntermediateDirectories: true)
        for c in cs where c.isAnalyzed {
            guard let m = c.manifest else { continue }
            let stem = c.url.deletingPathExtension().lastPathComponent + "_stillpoint"
            var out = outputFolder.appendingPathComponent(stem + ".mov")
            var k = 2
            let reserved = Set(exports.filter { $0.state == .queued || $0.state == .running }.map { $0.output.path })
            while fm.fileExists(atPath: out.path) || reserved.contains(out.path) {
                out = outputFolder.appendingPathComponent("\(stem) \(k).mov")
                k += 1
            }
            exports.append(ExportJob(clip: c, preset: preset, output: out, planPath: m.plan))
        }
        pumpExports()
    }

    func cancelExport(_ e: ExportJob) {
        if e.state == .queued { e.state = .cancelled }
        if e.state == .running { e.job?.cancel() }
    }

    func clearFinishedExports() {
        exports.removeAll { $0.state == .done || $0.state == .cancelled || { if case .failed = $0.state { return true }; return false }($0) }
    }

    func reveal(_ url: URL) { NSWorkspace.shared.activateFileViewerSelecting([url]) }

    private func pumpExports() {
        guard exportRunning == nil, !headless, let e = exports.first(where: { $0.state == .queued }) else { return }
        exportRunning = e
        e.state = .running
        e.message = "Starting the renderer…"
        let job = BridgeJob(["render", e.clip.url.path, "--plan", e.planPath, "--out", e.output.path,
                             "--codec", e.preset.codec, "--bitrate-mbps", "180"])
        e.job = job
        var failure: String?
        var cancelled = false
        job.onEvent = { [weak e] ev in
            guard let e else { return }
            switch ev {
            case .progress(let p):
                e.fraction = p.fraction
                e.message = p.message
                e.eta = p.eta
            case .result(let obj, _):
                e.bytes = (obj["bytes"] as? NSNumber)?.int64Value
                e.seconds = obj["seconds"] as? Double
            case .error(_, let msg): failure = msg
            case .cancelled: cancelled = true
            case .log, .notice: break
            }
        }
        job.onExit = { [weak self, weak e] exit in
            guard let self else { return }
            if let e {
                e.job = nil
                if cancelled || exit.cancelRequested { e.state = .cancelled }
                else if let failure { e.state = .failed(failure) }
                else if !exit.succeeded { e.state = .failed("Renderer stopped (\(exit.describe)). " + (exit.stderrTail.last ?? "")) }
                else { e.state = .done; e.fraction = 1 }
            }
            self.exportRunning = nil
            self.pumpExports()
        }
        do { try job.start(); track(job) } catch {
            e.state = .failed(error.localizedDescription)
            exportRunning = nil
            pumpExports()
        }
    }

    /// Quit-time cleanup: stop every engine process group we started (SIGTERM, then SIGKILL), remove scratch.
    func shutdown() {
        watchdog?.invalidate()
        watchdog = nil
        let jobs = allJobs.filter { $0.isRunning || $0.groupAlive }
        let group = DispatchGroup()
        for j in jobs {
            DispatchQueue.global().async(group: group) {
                j.terminateNow(grace: 2.5)
                for d in EngineConfig.scratchDirs(forBridgePID: j.pid) where j.pid > 0 { try? FileManager.default.removeItem(at: d) }
            }
        }
        _ = group.wait(timeout: .now() + 5)
    }

    /// For tests: process groups of every job this model started that still have live processes.
    var liveGroups: [pid_t] { allJobs.filter { $0.groupAlive }.map { $0.pid } }
}
