// Engine location, app storage (Application Support, never iCloud) and the JSON-lines bridge process
// (engine/stillpoint/app_bridge.py), which runs as the leader of its own process group so cancel / quit can stop
// everything it started (ffmpeg decoders, the measurement pool, sprender).
import Darwin
import Foundation

/// Test-footage locations for the headless self-tests (same env vars and defaults as eval/footage.py).
enum Footage {
    private static func dir(_ key: String, _ fallback: String) -> String {
        if let v = ProcessInfo.processInfo.environment[key], !v.isEmpty { return (v as NSString).expandingTildeInPath }
        return fallback
    }
    static var baseDir: String { dir("STILLPOINT_FOOTAGE_DIR", NSHomeDirectory() + "/Desktop") }
    static var o3Dir: String { dir("STILLPOINT_O3_DIR", baseDir + "/untitled folder 4") }
    static var oa4Dir: String { dir("STILLPOINT_OA4_DIR", baseDir) }
    static var sdDir: String { dir("STILLPOINT_SD_DIR", "/Volumes/Untitled/DCIM/DJI_001") }
    static func o3(_ name: String) -> String { o3Dir + "/" + name }
    static func oa4(_ name: String) -> String { oa4Dir + "/" + name }
}

enum EngineConfig {
    /// The repo root (the folder holding engine/, shaders/, .venv/): $STILLPOINT_ROOT, else the root build.sh baked into
    /// Info.plist (StillpointEngineRoot), else the folder four levels above an in-repo build
    /// (<root>/app/Stillpoint/build/Stillpoint.app). Settings (⌘,) can override it.
    static var defaultPath: String {
        let fm = FileManager.default
        func isRoot(_ p: String) -> Bool { fm.fileExists(atPath: p + "/engine/stillpoint") }
        if let e = ProcessInfo.processInfo.environment["STILLPOINT_ROOT"], !e.isEmpty {
            return (e as NSString).expandingTildeInPath
        }
        if let baked = Bundle.main.object(forInfoDictionaryKey: "StillpointEngineRoot") as? String, isRoot(baked) {
            return baked
        }
        var u = Bundle.main.bundleURL.standardizedFileURL
        for _ in 0..<4 { u.deleteLastPathComponent() }
        return u.path
    }
    static let key = "enginePath"

    static var rootPath: String {
        get {
            let s = UserDefaults.standard.string(forKey: key) ?? ""
            return s.isEmpty ? defaultPath : s
        }
        set { UserDefaults.standard.set(newValue, forKey: key) }
    }
    static var root: URL { URL(fileURLWithPath: rootPath, isDirectory: true) }
    static var python: URL { root.appendingPathComponent(".venv/bin/python") }
    static var shader: URL { root.appendingPathComponent("shaders/warp.metal") }
    static var sprender: URL { root.appendingPathComponent("app/renderer/.build/sprender") }

    private static var env: [String: String] { ProcessInfo.processInfo.environment }
    private static var librarySupport: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory() + "/Library/Application Support", isDirectory: true)
    }
    /// ~/Library/Application Support/Stillpoint — local, not synced to iCloud (the Desktop is).
    /// STILLPOINT_SUPPORT_DIR overrides it (headless tests).
    static var supportRoot: URL {
        if let o = env["STILLPOINT_SUPPORT_DIR"], !o.isEmpty { return URL(fileURLWithPath: o, isDirectory: true) }
        return librarySupport.appendingPathComponent("Stillpoint", isDirectory: true)
    }
    /// Finished analyses (plan.spplan, report.json, analysis.npz, stillpoint_app.json) per clip.
    static var analyses: URL { supportRoot.appendingPathComponent("analyses", isDirectory: true) }
    /// The bridge's STILLPOINT_WORK_DIR: frame caches and other temporaries (tmp/analyze-<pid>), telemetry caches.
    /// Created by the bridge when an analysis starts, never by the app at launch.
    static var workDir: URL { supportRoot.appendingPathComponent("work", isDirectory: true) }
    static func scratchDir(forBridgePID pid: Int32) -> URL {
        workDir.appendingPathComponent("tmp/analyze-\(pid)", isDirectory: true)
    }
    /// Everything a (dead) bridge process left in the work dir: the bridge's tmp/analyze-<pid> and the engine's
    /// jobs/<kind>-<pid>-<time> job dirs (ENGINE v3 workspace.JobDir; the engine runs inside the bridge process).
    static func scratchDirs(forBridgePID pid: Int32) -> [URL] {
        guard pid > 1, kill(pid, 0) != 0, errno == ESRCH else { return [] }     // only for a pid that is gone
        let jobs = workDir.appendingPathComponent("jobs", isDirectory: true)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: jobs.path)) ?? []
        return [scratchDir(forBridgePID: pid)] + names.filter {
            let parts = $0.split(separator: "-")
            return parts.count >= 3 && parts[1] == Substring(String(pid))
        }.map { jobs.appendingPathComponent($0, isDirectory: true) }
    }
    /// Analyses made by earlier builds (read-only sources for migration; never deleted).
    static var legacyAnalyses: [URL] {
        [root.appendingPathComponent("work/app/analyses", isDirectory: true),
         workDir.appendingPathComponent("app/analyses", isDirectory: true)]
    }
    static var speedHistory: URL { supportRoot.appendingPathComponent("speed_history.json") }
    /// Headless test / snapshot output (STILLPOINT_SCRATCH_DIR overrides).
    static var scratch: URL {
        if let o = env["STILLPOINT_SCRATCH_DIR"], !o.isEmpty { return URL(fileURLWithPath: o, isDirectory: true) }
        return librarySupport.appendingPathComponent("Stillpoint/scratch/app", isDirectory: true)
    }

    /// Test hook: run this executable + leading arguments instead of `python -m stillpoint.app_bridge`.
    static var bridgeOverride: [String]?

    /// Human-readable problems with the configured engine (empty = OK).
    static func problems() -> [String] {
        let fm = FileManager.default
        var p: [String] = []
        if !fm.isExecutableFile(atPath: python.path) { p.append("Python not found at .venv/bin/python") }
        if !fm.fileExists(atPath: root.appendingPathComponent("engine/stillpoint/app_bridge.py").path) {
            p.append("engine/stillpoint/app_bridge.py missing")
        }
        if !fm.fileExists(atPath: shader.path) { p.append("shaders/warp.metal missing") }
        if !fm.isExecutableFile(atPath: sprender.path) { p.append("Renderer not built (app/renderer/build.sh)") }
        return p
    }

    /// Per-clip analysis cache directory: <analyses>/<stem>-<fnv1a(path)> (app_bridge.analysis_dir_for mirrors it).
    static func analysisDir(for clip: URL, in base: URL? = nil) -> URL {
        let path = clip.standardizedFileURL.path
        var h: UInt64 = 0xcbf29ce484222325
        for b in path.utf8 { h ^= UInt64(b); h = h &* 0x100000001b3 }
        let stem = clip.deletingPathExtension().lastPathComponent
        return (base ?? analyses).appendingPathComponent("\(stem)-\(String(format: "%016llx", h).prefix(10))", isDirectory: true)
    }
}

/// File identity (size + mtime in ns) — must match the bridge's os.stat values.
func fileIdentity(_ url: URL) -> (size: Int64, mtimeNs: Int64)? {
    var st = stat()
    guard stat(url.path, &st) == 0 else { return nil }
    return (Int64(st.st_size), Int64(st.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(st.st_mtimespec.tv_nsec))
}

/// Free bytes on the volume holding `url` (or its nearest existing ancestor) — what `df` calls "Avail".
func freeBytes(at url: URL) -> Int64? {
    var p = url.standardizedFileURL.path
    let fm = FileManager.default
    while !fm.fileExists(atPath: p), p != "/" { p = (p as NSString).deletingLastPathComponent }
    var s = statfs()
    guard statfs(p, &s) == 0 else { return nil }
    return Int64(s.f_bavail) * Int64(s.f_bsize)
}

// MARK: - Storage migration

enum AnalysisStore {
    static let files = ["plan.spplan", "report.json", "analysis.npz"]
    static let maxCopyBytes: Int64 = 50_000_000

    /// Copy finished analyses from the legacy caches (work/app/analyses, iCloud-synced) into EngineConfig.analyses:
    /// only the small result files (never .luma_cache / .work-* temporaries or anything > 50 MB), with the manifest's
    /// paths rewritten. The old copies are left in place. Idempotent. Returns the migrated entry names.
    @discardableResult
    static func migrateLegacy(from sources: [URL] = EngineConfig.legacyAnalyses, to dest: URL = EngineConfig.analyses) -> [String] {
        let fm = FileManager.default
        var migrated: [String] = []
        for src in sources where fm.fileExists(atPath: src.path) && src.standardizedFileURL != dest.standardizedFileURL {
            let entries = (try? fm.contentsOfDirectory(at: src, includingPropertiesForKeys: nil)) ?? []
            for e in entries where !e.lastPathComponent.hasPrefix(".") {
                let man = e.appendingPathComponent("stillpoint_app.json"), plan = e.appendingPathComponent("plan.spplan")
                guard fm.fileExists(atPath: man.path), fm.fileExists(atPath: plan.path) else { continue }
                let out = dest.appendingPathComponent(e.lastPathComponent, isDirectory: true)
                if fm.fileExists(atPath: out.appendingPathComponent("stillpoint_app.json").path) { continue }
                guard let data = try? Data(contentsOf: man),
                      var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { continue }
                let sizes = files.map { f -> Int64? in
                    (try? fm.attributesOfItem(atPath: e.appendingPathComponent(f).path)[.size] as? NSNumber)?.int64Value
                }
                if let p = sizes[0], p > maxCopyBytes { continue }        // implausible plan: leave it alone
                do {
                    try fm.createDirectory(at: out, withIntermediateDirectories: true)
                    for (f, size) in zip(files, sizes) {
                        guard let size, size <= maxCopyBytes else { continue }
                        let tmp = out.appendingPathComponent("." + f + ".migrating")
                        try? fm.removeItem(at: tmp)
                        try fm.copyItem(at: e.appendingPathComponent(f), to: tmp)
                        _ = try fm.replaceItemAt(out.appendingPathComponent(f), withItemAt: tmp)
                    }
                    obj["plan"] = out.appendingPathComponent("plan.spplan").path
                    obj["report"] = out.appendingPathComponent("report.json").path
                    obj["migrated_from"] = e.path
                    let json = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
                    try json.write(to: out.appendingPathComponent("stillpoint_app.json"), options: .atomic)
                    migrated.append(e.lastPathComponent)
                } catch {
                    NSLog("Stillpoint: migrating \(e.path) failed: \(error)")
                }
            }
        }
        return migrated
    }
}

// MARK: - Bridge protocol models

struct BridgeProgress: Equatable {
    var stage: String
    var label: String
    var fraction: Double
    var message: String
    var elapsed: Double
    var eta: Double?
}

struct GyroStatus: Codable, Equatable {
    var label: String
    var level: String
    var detail: String
}

struct FovRange: Codable, Equatable {
    var minDeg: Double
    var maxDeg: Double
    var defaultDeg: Double
}

struct ScratchNeed: Codable, Equatable {
    var frameCacheBytes: Int64
    var tempBytes: Int64?
    var minFreeBytes: Int64
}

struct ProbeInfo: Codable, Equatable {
    var path: String
    var name: String
    var sizeBytes: Int64
    var width: Int
    var height: Int
    var fps: Double
    var nFrames: Int
    var durationS: Double
    var codec: String?
    var bitDepth: Int?
    var colorTransfer: String?
    var gyro: GyroStatus
    var supported: Bool
    var camera: String?
    var imuRate: Double?
    var eisBaked: Bool?
    var horizonLockSupported: Bool?
    var fov: FovRange?
    var warnings: [String]?
    var telemetryError: String?
    var scratch: ScratchNeed?

    /// Full camera name for headers ("DJI O3", "DJI Osmo Action 4", "DJI O4 Pro").
    var cameraName: String {
        guard let c = camera, !c.isEmpty else { return "Unknown camera" }
        if c.hasPrefix("DJI O3") { return "DJI O3" }
        return c
    }
    /// Compact name for list rows.
    var shortCamera: String {
        let c = cameraName
        if c.contains("Osmo Action") { return c.replacingOccurrences(of: "DJI Osmo ", with: "") }
        return c.replacingOccurrences(of: "DJI ", with: "")
    }
    var hdrLabel: String? {
        switch colorTransfer ?? "" {
        case "arib-std-b67": return "HLG"
        case "smpte2084": return "PQ"
        default: return nil
        }
    }
}

struct ClipIdentityJSON: Codable, Equatable {
    var path: String
    var sizeBytes: Int64
    var mtimeNs: Int64
}

struct AnalysisParams: Codable, Equatable {
    var smoothness: Double
    var fovDeg: Double?
    var cropArea: Double?
    var horizonLock: Bool
    var loopIters: Int
}

/// report.json['quality'] as normalized by the bridge: the engine's independent original-vs-stabilized measurement.
struct QualityMetric: Codable, Equatable {
    var key: String
    var label: String
    var original: Double?
    var stabilized: Double?
}

/// One independently measured window (the engine samples windows of the clip; times in clip seconds).
struct QualityWindow: Codable, Equatable {
    var t0S: Double
    var t1S: Double
    var original: Double?
    var stabilized: Double?
    enum CodingKeys: String, CodingKey { case t0S = "t0_s", t1S = "t1_s", original, stabilized }
}

struct QualitySummary: Codable, Equatable {
    var method: String?
    var units: String?
    var metrics: [QualityMetric]
    var windows: [QualityWindow]?
    var newJumpsGt1px: Double?
    var newJumpsGt05px: Double?
    var newJumpsMaxPx: Double?
    var coverageFrac: Double?
    var cropFootprintMean: Double?
    var nFramesScored: Double?
    var windowS: Double?
    enum CodingKeys: String, CodingKey {
        case method, units, metrics, windows
        case newJumpsGt1px = "new_jumps_gt_1px", newJumpsGt05px = "new_jumps_gt_0_5px", newJumpsMaxPx = "new_jumps_max_px"
        case coverageFrac = "coverage_frac", cropFootprintMean = "crop_footprint_mean", nFramesScored = "n_frames_scored"
        case windowS = "window_s"
    }
    func metric(_ key: String) -> QualityMetric? { metrics.first { $0.key == key } }
}

struct JitterSummary: Codable, Equatable {
    var hfovDeg: Double?
    var zoomMax: Double?
    var zoomFrac: Double?
    var windowS: Double?
    var origHfPx: Double?
    var origWindowHfPx: Double?
    var origB830Px: Double?
    var gyroOnlyPx: Double?
    var finalPx: Double?
    var finalB830Px: Double?
    var gyroOnlyB830Px: Double?
    var trustedFrac: Double?
    var winOrigPx: [Double?]?
    var winGyroOnlyPx: [Double?]?
    var winFinalPx: [Double?]?
    var finalMethod: String?
    var quality: QualitySummary?
    var qualityError: String?

    enum CodingKeys: String, CodingKey {
        case hfovDeg = "hfov_deg", zoomMax = "zoom_max", zoomFrac = "zoom_frac", windowS = "window_s"
        case origHfPx = "orig_hf_px", origWindowHfPx = "orig_window_hf_px", origB830Px = "orig_b8_30_px"
        case gyroOnlyPx = "gyro_only_px", finalPx = "final_px", finalB830Px = "final_b8_30_px"
        case gyroOnlyB830Px = "gyro_only_b8_30_px", trustedFrac = "trusted_frac"
        case winOrigPx = "win_orig_px", winGyroOnlyPx = "win_gyro_only_px", winFinalPx = "win_final_px"
        case finalMethod = "final_method", quality, qualityError = "quality_error"
    }
    /// Original shake on the same per-second windows the stabilised number uses.
    var origPx: Double? { origWindowHfPx ?? origHfPx }
}

struct AnalysisTiming: Codable, Equatable {
    var wallS: Double?
    var engineTotalS: Double?
    var nFrames: Int?
    var durationS: Double?
    var frameCache: Bool?
}

struct AnalysisManifest: Codable, Equatable {
    var clip: ClipIdentityJSON
    var params: AnalysisParams
    var created: String?
    var seconds: Double?
    var plan: String
    var timing: AnalysisTiming?
    var summary: JitterSummary?
}

enum JSONIO {
    static let snake: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()
    /// The manifest in `dir`. Its plan path is resolved inside `dir` first (the cache can be moved/migrated).
    static func loadManifest(_ dir: URL) -> AnalysisManifest? {
        let url = dir.appendingPathComponent("stillpoint_app.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        // summary uses explicit CodingKeys (digits in names do not round-trip through convertFromSnakeCase)
        struct Raw: Codable { var clip: ClipIdentityJSON; var params: AnalysisParams; var created: String?
                              var seconds: Double?; var plan: String; var timing: AnalysisTiming? }
        struct SummaryOnly: Codable { var summary: JitterSummary? }
        guard let raw = try? snake.decode(Raw.self, from: data) else { return nil }
        let summary = (try? JSONDecoder().decode(SummaryOnly.self, from: data))?.summary
        let local = dir.appendingPathComponent("plan.spplan").path
        let plan = FileManager.default.fileExists(atPath: local) ? local : raw.plan
        return AnalysisManifest(clip: raw.clip, params: raw.params, created: raw.created, seconds: raw.seconds,
                                plan: plan, timing: raw.timing, summary: summary)
    }
}

// MARK: - Process groups

/// posix_spawn in a NEW process group (pgid == pid) with only fds 0/1/2 inherited and default signal dispositions.
enum GroupSpawn {
    static func spawn(_ exe: String, _ args: [String], env: [String: String], cwd: String?,
                      stdin: Int32, stdout: Int32, stderr: Int32) throws -> pid_t {
        var fa: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&fa)
        defer { posix_spawn_file_actions_destroy(&fa) }
        posix_spawn_file_actions_adddup2(&fa, stdin, 0)
        posix_spawn_file_actions_adddup2(&fa, stdout, 1)
        posix_spawn_file_actions_adddup2(&fa, stderr, 2)
        if let cwd { posix_spawn_file_actions_addchdir_np(&fa, cwd) }
        var attr: posix_spawnattr_t?
        posix_spawnattr_init(&attr)
        defer { posix_spawnattr_destroy(&attr) }
        posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT
                                              | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK))
        posix_spawnattr_setpgroup(&attr, 0)
        var all = sigset_t(), none = sigset_t()
        sigfillset(&all)
        sigdelset(&all, SIGKILL)
        sigdelset(&all, SIGSTOP)
        sigemptyset(&none)
        posix_spawnattr_setsigdefault(&attr, &all)
        posix_spawnattr_setsigmask(&attr, &none)
        let argv: [UnsafeMutablePointer<CChar>?] = ([exe] + args).map { strdup($0) } + [nil]
        let envp: [UnsafeMutablePointer<CChar>?] = env.map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { argv.forEach { free($0) }; envp.forEach { free($0) } }
        var pid: pid_t = 0
        let rc = posix_spawn(&pid, exe, &fa, &attr, argv, envp)
        guard rc == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(rc),
                          userInfo: [NSLocalizedDescriptionKey: "cannot start \(exe): \(String(cString: strerror(rc)))"])
        }
        return pid
    }

    /// A close-on-exec pipe (read, write).
    static func pipe() -> (Int32, Int32) {
        var fds: [Int32] = [-1, -1]
        _ = Darwin.pipe(&fds)
        for fd in fds { _ = fcntl(fd, F_SETFD, FD_CLOEXEC) }
        return (fds[0], fds[1])
    }

    /// True while any process of the group exists (including orphans that outlived the leader).
    static func alive(_ pgid: pid_t) -> Bool {
        guard pgid > 1 else { return false }
        return kill(-pgid, 0) == 0 || errno == EPERM
    }

    static func pids(_ pgid: pid_t) -> [pid_t] {
        guard pgid > 1 else { return [] }
        let n = proc_listpgrppids(pgid, nil, 0)
        guard n > 0 else { return [] }
        var buf = [pid_t](repeating: 0, count: Int(n) + 32)
        let got = proc_listpgrppids(pgid, &buf, Int32(buf.count * MemoryLayout<pid_t>.size))
        return Array(buf.prefix(max(0, Int(got)))).filter { $0 > 0 }
    }

    private static let timebase: Double = {
        var tb = mach_timebase_info_data_t()
        mach_timebase_info(&tb)
        return Double(tb.numer) / Double(tb.denom)
    }()

    /// CPU seconds used so far by each live process of the group.
    static func cpuSeconds(_ pgid: pid_t) -> [pid_t: Double] {
        var out: [pid_t: Double] = [:]
        for p in pids(pgid) {
            var ti = proc_taskinfo()
            let r = proc_pidinfo(p, PROC_PIDTASKINFO, 0, &ti, Int32(MemoryLayout<proc_taskinfo>.size))
            if r == Int32(MemoryLayout<proc_taskinfo>.size) {
                out[p] = Double(ti.pti_total_user + ti.pti_total_system) * timebase / 1e9
            }
        }
        return out
    }

    static func startTime(_ pid: pid_t) -> Date? {
        var bi = proc_bsdinfo()
        let r = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &bi, Int32(MemoryLayout<proc_bsdinfo>.size))
        guard r == Int32(MemoryLayout<proc_bsdinfo>.size) else { return nil }
        return Date(timeIntervalSince1970: Double(bi.pbi_start_tvsec) + Double(bi.pbi_start_tvusec) / 1e6)
    }

    /// Signal the group. After its leader exited (`leaderExit`), a pgid could in principle be reused by an unrelated
    /// new group leader, so then only signal when a member started before that exit (i.e. it is ours).
    static func signal(_ pgid: pid_t, _ sig: Int32, leaderExit: Date?) {
        guard alive(pgid) else { return }
        if let t = leaderExit {
            let limit = t.addingTimeInterval(0.5)
            guard pids(pgid).contains(where: { (startTime($0) ?? .distantFuture) <= limit }) else { return }
        }
        kill(-pgid, sig)
    }

    /// SIGTERM the group, wait up to `grace` seconds, then SIGKILL whatever is left. Blocking (quit path).
    static func terminate(_ pgid: pid_t, grace: TimeInterval, leaderExit: Date?) {
        guard alive(pgid) else { return }
        signal(pgid, SIGTERM, leaderExit: leaderExit)
        let end = Date().addingTimeInterval(grace)
        while alive(pgid) && Date() < end { usleep(50_000) }
        signal(pgid, SIGKILL, leaderExit: leaderExit ?? Date())
        let end2 = Date().addingTimeInterval(1)
        while alive(pgid) && Date() < end2 { usleep(20_000) }
    }
}

// MARK: - Bridge process

/// One `python -m stillpoint.app_bridge ...` invocation, leader of its own process group.
/// Events are delivered on the main queue.
final class BridgeJob: @unchecked Sendable {
    enum Event {
        case progress(BridgeProgress)
        case log(String)
        case notice(code: String, message: String)
        case result([String: Any], Data)
        case error(code: String, message: String)
        case cancelled
    }
    struct Exit {
        var code: Int32              // exit status (valid when signal == 0)
        var signal: Int32            // terminating signal, 0 if it exited
        var sawTerminal: Bool        // a result / error / cancelled event was received
        var cancelRequested: Bool
        var stderrTail: [String]
        var succeeded: Bool { signal == 0 && code == 0 }
        var describe: String {
            if signal != 0 {
                let name: String
                switch signal {
                case SIGKILL: name = "SIGKILL — killed, e.g. by the system when memory ran out"
                case SIGTERM: name = "SIGTERM"
                case SIGSEGV: name = "SIGSEGV — crashed"
                case SIGABRT: name = "SIGABRT — aborted"
                default: name = "signal \(signal)"
                }
                return "stopped by \(name)"
            }
            return "exit code \(code)"
        }
    }

    /// Seconds between the cancel SIGTERM and the SIGKILL of the whole group.
    static var killDelay: TimeInterval = 5
    /// The bridge's own cooperative-cancel grace (must be shorter than killDelay).
    static var bridgeGrace: TimeInterval = 4

    let args: [String]
    private let lock = NSLock()
    private var terminal = false
    private var cancelRequested = false
    private var stderrTail: [String] = []
    private var stdinWrite: Int32 = -1
    private(set) var pid: pid_t = 0          // == its process group id
    private(set) var exited = false
    private var exitDate: Date?
    var onEvent: ((Event) -> Void)?
    var onExit: ((Exit) -> Void)?

    init(_ args: [String]) { self.args = args }

    var isRunning: Bool { lock.lock(); defer { lock.unlock() }; return pid > 0 && !exited }
    var wasCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelRequested }
    /// Any process of the job's group still alive (the bridge, or something it started)?
    var groupAlive: Bool { GroupSpawn.alive(pid) }
    /// This run's scratch dir (frame cache etc.) under the app's work dir.
    var scratchDir: URL? { pid > 0 ? EngineConfig.scratchDir(forBridgePID: pid) : nil }

    private func command() -> (String, [String]) {
        if let o = EngineConfig.bridgeOverride, let exe = o.first {
            return (exe, Array(o.dropFirst()) + args)
        }
        return (EngineConfig.python.path, ["-m", "stillpoint.app_bridge", "--grace", String(format: "%.1f", Self.bridgeGrace)] + args)
    }

    private func environment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = EngineConfig.root.appendingPathComponent("engine").path
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin" + (env["PATH"].map { ":" + $0 } ?? "")
        env["STILLPOINT_WORK_DIR"] = EngineConfig.workDir.path      // temporaries stay off the iCloud-synced Desktop
        return env
    }

    func start() throws {
        let (exe, argv) = command()
        let (inR, inW) = GroupSpawn.pipe(), (outR, outW) = GroupSpawn.pipe(), (errR, errW) = GroupSpawn.pipe()
        let p: pid_t
        do {
            p = try GroupSpawn.spawn(exe, argv, env: environment(), cwd: EngineConfig.root.path,
                                     stdin: inR, stdout: outW, stderr: errW)
        } catch {
            for fd in [inR, inW, outR, outW, errR, errW] { close(fd) }
            throw error
        }
        close(inR); close(outW); close(errW)
        lock.lock(); pid = p; stdinWrite = inW; lock.unlock()

        let group = DispatchGroup()
        group.enter()
        Thread.detachNewThread { [self] in
            let out = FileHandle(fileDescriptor: outR, closeOnDealloc: true)
            var buf = Data()
            while true {
                let d = out.availableData
                if d.isEmpty { break }
                buf.append(d)
                while let nl = buf.firstIndex(of: 0x0A) {
                    let line = buf.subdata(in: buf.startIndex..<nl)
                    buf.removeSubrange(buf.startIndex...nl)
                    handle(line)
                }
            }
            if !buf.isEmpty { handle(buf) }
            group.leave()
        }
        group.enter()
        Thread.detachNewThread { [self] in
            let err = FileHandle(fileDescriptor: errR, closeOnDealloc: true)
            var partial = ""
            while true {
                let d = err.availableData
                if d.isEmpty { break }
                partial += String(decoding: d, as: UTF8.self)
                var lines = partial.components(separatedBy: "\n")
                partial = lines.removeLast()
                appendStderr(lines)
            }
            if !partial.isEmpty { appendStderr([partial]) }
            group.leave()
        }
        Thread.detachNewThread { [self] in
            var status: Int32 = 0
            while waitpid(p, &status, 0) == -1 && errno == EINTR {}
            let exitAt = Date()
            lock.lock(); exited = true; exitDate = exitAt; let fd = stdinWrite; stdinWrite = -1; lock.unlock()
            if fd >= 0 { close(fd) }
            // an orphan in the group can keep the pipes open: never wait for it longer than this
            _ = group.wait(timeout: .now() + 3)
            let sig = status & 0x7f
            lock.lock()
            let exit = Exit(code: sig == 0 ? (status >> 8) & 0xff : 0, signal: sig == 0 ? 0 : sig, sawTerminal: terminal,
                            cancelRequested: cancelRequested, stderrTail: stderrTail)
            lock.unlock()
            // nothing the bridge started may outlive it: TERM then KILL any leftovers of the group
            if GroupSpawn.alive(p) {
                DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
                    GroupSpawn.signal(p, SIGTERM, leaderExit: exitAt)
                    DispatchQueue.global().asyncAfter(deadline: .now() + Self.killDelay) {
                        GroupSpawn.signal(p, SIGKILL, leaderExit: exitAt)
                    }
                }
            }
            DispatchQueue.main.async { self.onExit?(exit) }
        }
    }

    private func appendStderr(_ lines: [String]) {
        let clean = lines.map { $0.trimmingCharacters(in: .whitespacesAndNewlines.subtracting(.init(charactersIn: " "))) }
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        guard !clean.isEmpty else { return }
        lock.lock()
        stderrTail.append(contentsOf: clean)
        if stderrTail.count > 60 { stderrTail.removeFirst(stderrTail.count - 60) }
        lock.unlock()
    }

    private func handle(_ line: Data) {
        guard let obj = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any],
              let type = obj["type"] as? String else { return }
        let ev: Event?
        switch type {
        case "progress":
            ev = .progress(BridgeProgress(stage: obj["stage"] as? String ?? "",
                                          label: obj["label"] as? String ?? "",
                                          fraction: (obj["fraction"] as? Double) ?? 0,
                                          message: obj["message"] as? String ?? "",
                                          elapsed: (obj["elapsed_s"] as? Double) ?? 0,
                                          eta: obj["eta_s"] as? Double))
        case "log": ev = .log(obj["message"] as? String ?? "")
        case "notice": ev = .notice(code: obj["code"] as? String ?? "", message: obj["message"] as? String ?? "")
        case "result": ev = .result(obj, line)
        case "error": ev = .error(code: obj["code"] as? String ?? "error", message: obj["message"] as? String ?? "")
        case "cancelled": ev = .cancelled
        default: ev = nil
        }
        if let ev {
            switch ev {
            case .result, .error, .cancelled: lock.lock(); terminal = true; lock.unlock()
            default: break
            }
            DispatchQueue.main.async { self.onEvent?(ev) }
        }
    }

    /// Cancel: SIGTERM to the whole group + stdin closed (the bridge stops cooperatively and cleans its scratch),
    /// then SIGKILL to the group after `killDelay` seconds if anything in it is still alive.
    func cancel() {
        lock.lock()
        let p = pid, fd = stdinWrite
        guard p > 0, !cancelRequested else { lock.unlock(); return }
        cancelRequested = true
        stdinWrite = -1
        lock.unlock()
        GroupSpawn.signal(p, SIGTERM, leaderExit: leaderExitDate)
        if fd >= 0 { close(fd) }
        DispatchQueue.global().asyncAfter(deadline: .now() + Self.killDelay) { [self] in
            GroupSpawn.signal(p, SIGKILL, leaderExit: leaderExitDate)
        }
    }

    private var leaderExitDate: Date? { lock.lock(); defer { lock.unlock() }; return exitDate }

    /// Quit path: stop the group now (blocking, at most grace + 1 s).
    func terminateNow(grace: TimeInterval = 2.5) {
        lock.lock()
        let p = pid, fd = stdinWrite
        cancelRequested = true
        stdinWrite = -1
        lock.unlock()
        if fd >= 0 { close(fd) }
        GroupSpawn.terminate(p, grace: grace, leaderExit: leaderExitDate)
    }

    /// Synchronous helper for headless modes: run to completion and return the result object.
    static func runSync(_ args: [String], timeout: TimeInterval = 600) -> Result<[String: Any], Error> {
        let job = BridgeJob(args)
        var result: Result<[String: Any], Error> = .failure(NSError(domain: "bridge", code: -1,
                                                                    userInfo: [NSLocalizedDescriptionKey: "no result"]))
        var done = false
        job.onEvent = { ev in
            switch ev {
            case .result(let obj, _): result = .success(obj)
            case .error(_, let m): result = .failure(NSError(domain: "bridge", code: 1, userInfo: [NSLocalizedDescriptionKey: m]))
            default: break
            }
        }
        job.onExit = { _ in done = true }
        do { try job.start() } catch { return .failure(error) }
        let end = Date().addingTimeInterval(timeout)
        while !done && Date() < end { RunLoop.main.run(until: Date().addingTimeInterval(0.02)) }
        if !done {
            job.terminateNow()
            return .failure(NSError(domain: "bridge", code: 2, userInfo: [NSLocalizedDescriptionKey: "timed out"]))
        }
        return result
    }
}

extension ProbeInfo {
    static func decode(result obj: [String: Any]) -> ProbeInfo? {
        guard let data = try? JSONSerialization.data(withJSONObject: obj) else { return nil }
        return try? JSONIO.snake.decode(ProbeInfo.self, from: data)
    }
}
