// Headless end-to-end check of the app's own wiring (no window): storage migration -> AppModel -> BridgeJob probe ->
// pre-flight -> analyze (progress, cache manifest, plan, speed history) -> export queue (sprender PROGRESS lines) ->
// cancel a real re-analysis (the whole process group must be gone, checked with ps) -> a fake bridge that stalls and
// ignores SIGTERM (stall watchdog, SIGKILL after 5 s) -> a fake bridge that crashes (error panel, scratch cleanup,
// Retry) -> a structured engine error -> quit while running (shutdown kills the group) -> no orphans (ps).
//
//   Stillpoint --selftest-app [--clip CLIP] [--out DIR]
// Run it with STILLPOINT_SUPPORT_DIR=<scratch> to keep its analyses / scratch out of the real Application Support.
import AppKit
import Foundation

@MainActor
enum AppFlowTest {
    struct PS { let pid: Int32; let pgid: Int32; let ppid: Int32; let stat: String; let command: String }

    /// `ps` itself (not libproc), so the check is independent of the code under test.
    static func ps() -> [PS] {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/ps")
        p.arguments = ["-Ao", "pid=,pgid=,ppid=,stat=,command="]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        guard (try? p.run()) != nil else { return [] }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return String(decoding: data, as: UTF8.self).split(separator: "\n").compactMap { line in
            let f = line.split(separator: " ", maxSplits: 4, omittingEmptySubsequences: true)
            guard f.count == 5, let a = Int32(f[0]), let b = Int32(f[1]), let c = Int32(f[2]) else { return nil }
            return PS(pid: a, pgid: b, ppid: c, stat: String(f[3]), command: String(f[4]))
        }
    }

    /// The whole process tree under `root` (by parent pid, so it also catches children that left the group) plus
    /// everything in its process group.
    static func tree(_ root: pid_t, in all: [PS]) -> [PS] {
        var ids: Set<Int32> = [root]
        var grew = true
        while grew {
            grew = false
            for p in all where !ids.contains(p.pid) && (ids.contains(p.ppid) || p.pgid == root) { ids.insert(p.pid); grew = true }
        }
        return all.filter { ids.contains($0.pid) }
    }
    /// Processes from `before` that still exist (same pid AND same command, so a recycled pid does not count).
    static func survivors(_ before: [PS]) -> [PS] {
        let now = Dictionary(ps().map { ($0.pid, $0.command) }, uniquingKeysWith: { a, _ in a })
        return before.filter { now[$0.pid] == $0.command }
    }

    static let fakeBridge = #"""
    """Fake app_bridge for Stillpoint --selftest-app:  --mode stall | crash | error"""
    import json, os, signal, subprocess, sys, time
    mode = sys.argv[sys.argv.index('--mode') + 1] if '--mode' in sys.argv else 'stall'
    def emit(o):
        sys.stdout.write(json.dumps(o) + '\n'); sys.stdout.flush()
    wd = os.environ.get('STILLPOINT_WORK_DIR')
    emit(dict(type='start', cmd='analyze', pid=os.getpid(), pgid=os.getpgrp(), protocol=2, work_root=wd))
    emit(dict(type='progress', stage='measure', label='Measuring jitter', fraction=0.31, stage_fraction=0.2,
              message='pass 1 of 3 · 820/2964 frames', elapsed_s=1.0, eta_s=None))
    if mode == 'stall':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)          # a wedged engine that ignores the polite stop ...
        kids = [subprocess.Popen(['/bin/sleep', '600']) for _ in range(2)]    # ... and children that inherit that
        time.sleep(600)
    elif mode == 'crash':
        if wd:
            for d in (os.path.join(wd, 'tmp', f'analyze-{os.getpid()}'),
                      os.path.join(wd, 'jobs', f'analyze-{os.getpid()}-{int(time.time())}')):
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, 'frames.bin'), 'wb').write(b'\0' * 4096)
        for l in ('[stillpoint] DJI_0026.MP4: DJI O3, 268 frames, imu 2000 Hz', 'Traceback (most recent call last):',
                  '  File "engine/stillpoint/pipeline.py", line 834, in _analyze',
                  'MemoryError: Unable to allocate 16.7 GiB for an array with shape (24120, 720, 960)'):
            sys.stderr.write(l + '\n')
        sys.stderr.flush()
        os._exit(3)
    elif mode == 'error':
        sys.stderr.write('[stillpoint] checking free space\n'); sys.stderr.flush()
        emit(dict(type='error', code='disk_full', message="Only 1.2 GB free on the disk that holds Stillpoint's scratch folder."))
        sys.exit(1)
    """#

    static func run(args: [String]) -> Int32 {
        func opt(_ n: String) -> String? {
            if let i = args.firstIndex(of: n), i + 1 < args.count { return args[i + 1] }
            return nil
        }
        let clipURL = URL(fileURLWithPath: opt("--clip") ?? Footage.o3("DJI_0026.MP4"))
        let scratch = EngineConfig.scratch.appendingPathComponent("selftest-app", isDirectory: true)
        let out = URL(fileURLWithPath: opt("--out") ?? scratch.appendingPathComponent("export").path)
        let fm = FileManager.default
        try? fm.removeItem(at: out)
        try? fm.createDirectory(at: scratch, withIntermediateDirectories: true)
        var ok = true
        func check(_ c: Bool, _ m: String) { print((c ? "PASS " : "FAIL ") + m); fflush(stdout); if !c { ok = false } }
        func spin(_ timeout: Double, until cond: () -> Bool) -> Bool {
            let end = Date().addingTimeInterval(timeout)
            while !cond() && Date() < end { RunLoop.main.run(until: Date().addingTimeInterval(0.02)) }
            return cond()
        }
        check(AppModel.prefs !== UserDefaults.standard, "test settings go to a throwaway defaults suite, not the user's")
        let userPrefs = { ["outputFolder", "exportPreset"].map { UserDefaults.standard.string(forKey: $0) ?? "-" } }
        let userPrefsBefore = userPrefs()
        defer { EngineConfig.bridgeOverride = nil }
        print("support dir \(EngineConfig.supportRoot.path)")
        print("bridge work dir (STILLPOINT_WORK_DIR) \(EngineConfig.workDir.path)")

        // ---------------------------------------------------------------- storage + probe + pre-flight
        let model = AppModel()
        model.player.player.isMuted = true
        model.outputFolder = out
        model.preset = .hevcFast
        check(model.engineProblems.isEmpty, "engine found at \(EngineConfig.rootPath) \(model.engineProblems)")
        check(spin(60) { model.migrationDone }, "legacy analysis cache migrated off the iCloud-synced Desktop")
        let entries = (try? fm.contentsOfDirectory(at: EngineConfig.analyses, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        var migratedOK = !entries.isEmpty
        for e in entries {
            let files = (try? fm.contentsOfDirectory(atPath: e.path)) ?? []
            if files.contains(where: { $0.hasPrefix(".luma") || $0.hasPrefix(".work-") || $0.contains("migrating") }) { migratedOK = false }
            for f in files {
                let sz = (try? fm.attributesOfItem(atPath: e.appendingPathComponent(f).path)[.size] as? NSNumber)?.int64Value ?? 0
                if sz > AnalysisStore.maxCopyBytes { migratedOK = false }
            }
            if let m = JSONIO.loadManifest(e), !m.plan.hasPrefix(e.path) { migratedOK = false }
        }
        check(migratedOK, "migrated entries: \(entries.map(\.lastPathComponent).sorted()) (small files only, manifest paths rewritten)")

        model.add([clipURL])
        guard let clip = model.clips.first else { print("FAIL no clip"); return 1 }
        check(spin(120) { clip.probe != nil || clip.probeError != nil } && clip.probe != nil,
              "probe via bridge: \(clip.probe?.cameraName ?? "-") · \(clip.probe?.gyro.label ?? clip.probeError ?? "-")")
        check(spin(20) { clip.thumbnail != nil }, "thumbnail generated")
        if let pf = model.preflight(clip) {
            check(pf.duration > 4 && pf.estimate >= 30 && pf.freeBytes != nil && !pf.blocked && pf.minFreeBytes >= 3_000_000_000,
                  String(format: "pre-flight: clip %.1f s, estimate %.0f s (%@), %@ free, needs %@ free, temp %@",
                         pf.duration, pf.estimate, pf.estimateBasis, Fmt.gb(pf.freeBytes ?? 0), Fmt.gb(pf.minFreeBytes), Fmt.gb(pf.tempBytes)))
        } else { check(false, "pre-flight available") }
        check(!clip.onRemovableMedia, "internal-disk source: no removable-media warning")

        // ---------------------------------------------------------------- fresh analysis through the queue
        let speedBefore = model.speed.entries.count
        var progressSeen: [Double] = []
        var stages = Set<String>()
        model.analyze(clip)
        check(model.analysisQueueView.first?.id == clip.id, "clip shows in the analysis queue panel")
        let t0 = Date()
        var bridgePID: pid_t = 0
        let finished = spin(900) {
            if bridgePID == 0, let p = clip.job?.pid { bridgePID = p }
            if let p = clip.progress { if progressSeen.last != p.fraction { progressSeen.append(p.fraction) }; stages.insert(p.stage) }
            return !clip.isBusy
        }
        check(finished && clip.isAnalyzed && clip.phase == .idle,
              String(format: "analyze via bridge in %.0f s: %d progress updates, stages %@, phase %@", Date().timeIntervalSince(t0),
                     progressSeen.count, stages.sorted().joined(separator: ","), String(describing: clip.phase)))
        check(progressSeen == progressSeen.sorted() && progressSeen.count >= 5, "progress monotonic")
        check(clip.manifest?.plan.hasPrefix(EngineConfig.analyses.path) == true, "analysis cached in \(EngineConfig.analyses.path)")
        let tmpLeft = ((try? fm.contentsOfDirectory(atPath: EngineConfig.workDir.appendingPathComponent("tmp").path)) ?? [])
        check(tmpLeft.isEmpty && fm.fileExists(atPath: EngineConfig.workDir.path), "scratch used \(EngineConfig.workDir.path)/tmp and was cleaned (left: \(tmpLeft))")
        if let s = clip.manifest?.summary {
            let label = s.quality != nil ? "independent (\(s.quality?.method ?? "?"))" : (s.finalMethod ?? "unlabelled")
            check((s.origPx ?? 0) > 0 && s.finalPx != nil && s.finalMethod == "closed-loop residual (self-measured)",
                  String(format: "jitter summary: original (gyro) %.3f px, closed-loop residual %.3f px labelled self-measured", s.origPx ?? -1, s.finalPx ?? -1))
            if let q = s.quality {
                let hf = q.metric("hf")
                check(hf?.original != nil && hf?.stabilized != nil && q.metric("b8_30") != nil,
                      String(format: "independent quality from report.json: HF %.3f -> %.3f px, 8-30 Hz %.3f -> %.3f px, %d windows, jumps>1px %.0f",
                             hf?.original ?? -1, hf?.stabilized ?? -1, q.metric("b8_30")?.original ?? -1, q.metric("b8_30")?.stabilized ?? -1,
                             q.windows?.count ?? 0, q.newJumpsGt1px ?? -1) + " (\(label.prefix(60))…)")
            } else {
                print("NOTE report.json has no 'quality' (engine without the independent check): \(s.qualityError ?? "-")")
            }
        } else { check(false, "manifest summary present") }
        check(model.speed.entries.count == speedBefore + 1, "analysis time recorded for future estimates (\(model.speed.entries.last.map { String(format: "%.0f s for %.1f s of clip", $0.seconds, $0.durationS) } ?? "-"))")
        check(model.player.hasPlan, "preview player switched to the new plan")

        // ---------------------------------------------------------------- export queue: progress from sprender's frame counter
        model.export([clip])
        guard let job = model.exports.first else { print("FAIL no export job"); return 1 }
        let t1 = Date()
        var fr: [Double] = []
        var msgs = Set<String>()
        _ = spin(900) {
            if fr.last != job.fraction { fr.append(job.fraction) }
            msgs.insert(job.message)
            return job.state != .queued && job.state != .running
        }
        let size = (try? fm.attributesOfItem(atPath: job.output.path)[.size] as? Int64) ?? 0
        check(job.state == .done && size > 0, String(format: "export %@ (%@) in %.0f s, %d progress updates, %@", job.output.lastPathComponent,
                                                     job.preset.title, Date().timeIntervalSince(t1), fr.count, Fmt.bytes(size)))
        check(fr.count >= 3 && msgs.contains { $0.range(of: #"^\d+/268 frames"#, options: .regularExpression) != nil },
              "export progress from sprender PROGRESS lines (e.g. “\(msgs.filter { $0.contains("/268") }.sorted().last ?? "-")”)")
        try? fm.removeItem(at: out)

        // ---------------------------------------------------------------- cancel a real re-analysis: whole group gone
        let planBefore = clip.manifest?.plan
        clip.smoothness = 1.5
        check(clip.settingsChanged, "changing smoothness flags the analysis as stale")
        model.analyze(clip)
        _ = spin(300) { (clip.progress?.fraction ?? 0) > 0.15 || !clip.isBusy }
        let pg = clip.job?.pid ?? 0
        let before = pg > 0 ? tree(pg, in: ps()) : []
        let tc = Date()
        model.cancelAnalysis(clip)
        let stopped = spin(60) { !clip.isBusy }
        _ = spin(BridgeJob.killDelay + 4) { !GroupSpawn.alive(pg) && survivors(before).isEmpty }
        let after = ps().filter { $0.pgid == pg } + survivors(before).filter { $0.pgid != pg }
        let kinds = Set(before.map { c -> String in
            if c.command.contains("app_bridge") { return "bridge" }
            if c.command.contains("multiprocessing") { return "pool/tracker" }
            if c.command.contains("ffmpeg") { return "ffmpeg" }
            return URL(fileURLWithPath: String(c.command.split(separator: " ").first ?? "")).lastPathComponent
        })
        check(stopped && clip.phase == .cancelled && before.count >= 2 && after.isEmpty,
              String(format: "cancel-kills-group: %d processes in the bridge tree/group %d (%@) -> ps shows %d left after %.1f s (phase %@)",
                     before.count, pg, kinds.sorted().joined(separator: ", "), after.count, Date().timeIntervalSince(tc),
                     String(describing: clip.phase)))
        model.loadCachedAnalysis(clip)
        check(clip.isAnalyzed && clip.manifest?.plan == planBefore && abs((clip.manifest?.params.smoothness ?? 0) - 1.0) < 1e-6,
              "previous analysis kept after cancel (smoothness \(clip.manifest?.params.smoothness ?? -1))")
        _ = spin(BridgeJob.killDelay + 3) { !fm.fileExists(atPath: EngineConfig.scratchDir(forBridgePID: pg).path) }
        check(!fm.fileExists(atPath: EngineConfig.scratchDir(forBridgePID: pg).path), "no scratch dir left behind after cancel")

        // ---------------------------------------------------------------- fake bridges
        let fake = scratch.appendingPathComponent("fake_bridge.py")
        try? fakeBridge.write(to: fake, atomically: true, encoding: .utf8)
        let savedThreshold = AppModel.stallThreshold
        AppModel.stallThreshold = 3
        defer { AppModel.stallThreshold = savedThreshold }

        // stall watchdog + SIGTERM-ignoring group -> SIGKILL after killDelay
        EngineConfig.bridgeOverride = [EngineConfig.python.path, fake.path, "--mode", "stall"]
        model.analyze(clip)
        check(spin(20) { clip.isStalled }, "stall watchdog: “Still working — no update for \(clip.stallSeconds) s” after \(Int(AppModel.stallThreshold)) s without progress")
        check(spin(10) { clip.engineBusy != nil } && clip.engineBusy == false, "watchdog reports an idle engine (the fake only sleeps)")
        let spg = clip.job?.pid ?? 0
        let sBefore = spg > 0 ? tree(spg, in: ps()) : []
        let ts = Date()
        model.cancelAnalysis(clip)
        _ = spin(BridgeJob.killDelay + 5) { !GroupSpawn.alive(spg) }
        let dt = Date().timeIntervalSince(ts)
        let sAfter = ps().filter { $0.pgid == spg }
        check(sBefore.count == 3 && sAfter.isEmpty && dt > BridgeJob.killDelay - 0.5 && dt < BridgeJob.killDelay + 3,
              String(format: "SIGTERM ignored by bridge + 2 children -> whole group SIGKILLed after %.1f s; ps shows %d left", dt, sAfter.count))
        _ = spin(10) { !clip.isBusy }
        check(clip.phase == .cancelled && !clip.isStalled, "stalled analysis ends as cancelled")

        // abnormal exit -> error panel with the last stderr lines, scratch cleanup, Retry
        EngineConfig.bridgeOverride = [EngineConfig.python.path, fake.path, "--mode", "crash"]
        model.analyze(clip)
        var cpg: pid_t = 0
        _ = spin(30) { if cpg == 0, let p = clip.job?.pid { cpg = p }; return !clip.isBusy }
        let f1 = clip.failure
        check(f1?.title == "The engine stopped unexpectedly" && f1?.message.contains("exit code 3") == true
              && f1?.details.contains { $0.contains("MemoryError") } == true && f1?.canRetry == true,
              "error panel: “\(f1?.title ?? "-")” · \(f1?.message ?? "-") · last line “\(f1?.details.last ?? "-")”")
        let jobsDir = EngineConfig.workDir.appendingPathComponent("jobs").path
        func leftovers() -> [String] {
            (fm.fileExists(atPath: EngineConfig.scratchDir(forBridgePID: cpg).path) ? ["tmp/analyze-\(cpg)"] : [])
                + ((try? fm.contentsOfDirectory(atPath: jobsDir)) ?? []).filter { $0.contains("-\(cpg)-") }
        }
        _ = spin(BridgeJob.killDelay + 4) { leftovers().isEmpty }
        check(cpg > 0 && leftovers().isEmpty, "scratch of the crashed run (bridge tmp + engine job dir stand-ins) removed by the app")
        let groupsBefore = model.startedGroups.count
        model.retry(clip)
        _ = spin(30) { !clip.isBusy && model.startedGroups.count > groupsBefore }
        check(model.startedGroups.count == groupsBefore + 1 && clip.failure != nil, "Retry re-runs the analysis (fails again with the fake)")

        // structured engine error (e.g. disk full)
        EngineConfig.bridgeOverride = [EngineConfig.python.path, fake.path, "--mode", "error"]
        model.retry(clip)
        _ = spin(30) { !clip.isBusy }
        check(clip.failure?.title == "Not enough disk space" && clip.failure?.message.contains("1.2 GB") == true,
              "engine error event -> “\(clip.failure?.title ?? "-")”: \(clip.failure?.message ?? "-")")

        // quit while an analysis runs: shutdown() stops the whole group
        EngineConfig.bridgeOverride = [EngineConfig.python.path, fake.path, "--mode", "stall"]
        model.retry(clip)
        _ = spin(15) { clip.progress != nil }
        let qpg = clip.job?.pid ?? 0
        let tq = Date()
        model.shutdown()
        let qdt = Date().timeIntervalSince(tq)
        check(qpg > 0 && !GroupSpawn.alive(qpg) && ps().filter { $0.pgid == qpg }.isEmpty && qdt < 5,
              String(format: "app quit (shutdown) killed the running bridge group %d in %.1f s", qpg, qdt))
        EngineConfig.bridgeOverride = nil

        // ---------------------------------------------------------------- no orphans, checked with ps
        _ = spin(8) { model.startedGroups.allSatisfy { !GroupSpawn.alive($0) } }
        let groups = Set(model.startedGroups)
        let me = getpid()
        let orphans = ps().filter { groups.contains($0.pgid) || ($0.ppid == me && $0.command.contains("python")) }
            + survivors(before + sBefore).filter { !groups.contains($0.pgid) }   // children that left their group (python/ffmpeg)
        check(orphans.isEmpty, "no orphan processes from \(groups.count) bridge runs (ps): \(orphans.map { "\($0.pid) \($0.command.prefix(60))" })")
        check(userPrefs() == userPrefsBefore, "user's export folder / preset untouched (\(userPrefs().joined(separator: ", ")))")
        print("RESULT selftest-app \(ok ? "PASS" : "FAIL")")
        return ok ? 0 : 1
    }
}
