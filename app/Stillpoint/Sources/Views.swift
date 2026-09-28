// Main window: clip sidebar, viewer (stabilised preview, compare, transport) and the empty drop zone.
import AVFoundation
import SwiftUI
import UniformTypeIdentifiers

struct MainView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.snapshotMode) private var snapshot

    var body: some View {
        HStack(spacing: 0) {
            if !model.clips.isEmpty {
                Sidebar()
                    .frame(width: 272)
                Rectangle().fill(Theme.hairline).frame(width: 1)
            }
            if let clip = model.selected {
                ViewerPane(clip: clip, player: model.player)
                    .frame(minWidth: 560, maxWidth: .infinity, maxHeight: .infinity)
                Rectangle().fill(Theme.hairline).frame(width: 1)
                Inspector(clip: clip)
                    .frame(width: 324)
            } else {
                EmptyState()
            }
        }
        .background(Theme.window)
        .foregroundStyle(Theme.text)
        .onDrop(of: [UTType.fileURL], isTargeted: $model.dropTargeted) { providers in
            for p in providers {
                _ = p.loadObject(ofClass: URL.self) { url, _ in
                    if let url { DispatchQueue.main.async { model.add([url]) } }
                }
            }
            return true
        }
        .overlay {
            if model.dropTargeted && model.selected != nil {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(Theme.accent, lineWidth: 2)
                    .background(Theme.accent.opacity(0.06))
                    .overlay(Text("Drop to add clips").font(.spTitle).padding(.horizontal, 14).padding(.vertical, 8)
                        .background(Capsule().fill(Theme.raised2)))
                    .padding(6)
                    .allowsHitTesting(false)
            }
        }
    }
}

// MARK: - Logo

struct LogoMark: View {
    var size: CGFloat = 18
    var body: some View {
        ZStack {
            Circle().stroke(Theme.text.opacity(0.9), lineWidth: size * 0.09)
            Circle().fill(Theme.accent).frame(width: size * 0.3, height: size * 0.3)
        }.frame(width: size, height: size)
    }
}

// MARK: - Sidebar

struct Sidebar: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 9) {
                LogoMark()
                Text("Stillpoint").font(.system(size: 15, weight: .semibold)).tracking(0.2)
                Spacer()
                Button { model.openPanel() } label: { Image(systemName: "plus") }
                    .buttonStyle(IconButtonStyle(size: 26))
                    .help("Add clips (⌘O)")
            }
            .padding(.top, 40).padding(.horizontal, 16).padding(.bottom, 18)

            SectionLabel(text: "Clips", trailing: model.clips.isEmpty ? nil : "\(model.clips.count)")
                .padding(.horizontal, 16).padding(.bottom, 8)

            AdaptiveScroll {
                VStack(spacing: 3) {
                    ForEach(model.clips) { c in
                        ClipRow(clip: c, selected: c.id == model.selectedID)
                            .contentShape(Rectangle())
                            .onTapGesture { model.selectedID = c.id }
                            .contextMenu {
                                Button("Analyze") { model.analyze(c) }.disabled(!c.canAnalyze)
                                Button("Export") { model.export([c]) }.disabled(!c.isAnalyzed)
                                Button("Show in Finder") { model.reveal(c.url) }
                                Divider()
                                Button("Remove from List") { model.remove(c) }
                            }
                    }
                    if model.clips.isEmpty {
                        Text("Drop clips anywhere in the window, or press ⌘O.")
                            .font(.spSmall).foregroundStyle(Theme.text3)
                            .padding(.horizontal, 8).padding(.top, 4)
                    }
                }
                .padding(.horizontal, 8)
            }
            Spacer(minLength: 0)
            AnalysisQueuePanel()
            ExportQueuePanel()
        }
        .background(Theme.panel)
    }
}

struct ClipRow: View {
    @ObservedObject var clip: Clip
    let selected: Bool

    var body: some View {
        HStack(spacing: 11) {
            thumb
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 4) {
                    Text(clip.displayName).font(.system(size: 12.5, weight: .semibold)).lineLimit(1).truncationMode(.middle)
                        .foregroundStyle(Theme.text)
                    Spacer(minLength: 2)
                    if let p = clip.probe {
                        Text(Fmt.duration(p.durationS)).font(.spMono(10.5)).foregroundStyle(Theme.text3)
                    }
                }
                Text(meta).font(.system(size: 11)).foregroundStyle(Theme.text3).lineLimit(1)
                badge
            }
            Spacer(minLength: 0)
        }
        .padding(7)
        .help([clip.name, clip.recordedLabel.map { "Recorded " + $0 }].compactMap { $0 }.joined(separator: " — "))
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(selected ? Theme.raised2 : Color.clear))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(selected ? Theme.hairlineStrong : Color.clear))
    }

    private var meta: String {
        guard let p = clip.probe else { return clip.probing ? "Reading…" : (clip.probeError ?? "") }
        return [p.shortCamera, Fmt.resolution(p.width, p.height), String(format: "%.2f", p.fps)].joined(separator: " · ")
    }

    @ViewBuilder private var badge: some View {
        if let p = clip.probe {
            StatusBadge(label: rowLabel(p.gyro), level: p.gyro.level, compact: true)
        } else if clip.probeError != nil {
            StatusBadge(label: "Unreadable", level: "unsupported", compact: true)
        } else {
            Chip(text: "Reading gyro…")
        }
    }

    private func rowLabel(_ g: GyroStatus) -> String {
        if g.label.contains("EIS on") { return "EIS on · unsupported" }
        if g.label.contains("attitude") { return g.label.replacingOccurrences(of: " attitude — ", with: " · ") }
        return g.label
    }

    private var thumb: some View {
        ZStack {
            if let t = clip.thumbnail {
                Image(decorative: t, scale: 1).resizable().aspectRatio(contentMode: .fill)
            } else {
                Theme.raised2
                Image(systemName: "film").font(.system(size: 13)).foregroundStyle(Theme.text3)
            }
            if clip.phase == .running {
                Color.black.opacity(0.55)
                SPRing(fraction: clip.progress?.fraction ?? 0.02, size: 18)
            } else if clip.failure != nil {
                Color.black.opacity(0.5)
                Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 13)).foregroundStyle(Theme.bad)
            } else if clip.phase == .queued {
                Color.black.opacity(0.55)
                Image(systemName: "clock").font(.system(size: 12, weight: .semibold)).foregroundStyle(Theme.text2)
            }
        }
        .frame(width: 80, height: 45)
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(Color.white.opacity(0.08)))
        .overlay(alignment: .bottomTrailing) {
            if clip.isAnalyzed && !clip.isBusy {
                Image(systemName: "checkmark").font(.system(size: 7.5, weight: .heavy)).foregroundStyle(.white)
                    .frame(width: 14, height: 14).background(Circle().fill(Theme.accent))
                    .overlay(Circle().strokeBorder(Theme.panel, lineWidth: 1.5))
                    .offset(x: 4, y: 4)
            }
        }
    }
}

// MARK: - Analysis queue (sidebar footer): one clip at a time, each cancellable

struct AnalysisQueuePanel: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        if !model.analysisQueueView.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                SectionLabel(text: "Analysis", trailing: model.analysisQueueView.count > 1 ? "\(model.analysisQueueView.count - 1) queued" : nil)
                ForEach(model.analysisQueueView) { c in AnalysisQueueRow(clip: c) }
            }
            .padding(14)
            .background(Theme.raised.opacity(0.6))
            .overlay(alignment: .top) { Rectangle().fill(Theme.hairline).frame(height: 1) }
        }
    }
}

struct AnalysisQueueRow: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var clip: Clip
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 6) {
                Text(clip.displayName).font(.system(size: 11.5, weight: .medium)).lineLimit(1).truncationMode(.middle)
                Spacer(minLength: 4)
                Button { model.cancelAnalysis(clip) } label: { Image(systemName: "xmark") }
                    .buttonStyle(IconButtonStyle(size: 18)).font(.system(size: 9))
                    .help(clip.phase == .queued ? "Remove from queue" : "Cancel analysis")
            }
            if clip.phase == .running {
                SPProgressBar(fraction: clip.progress?.fraction ?? 0, tint: clip.isStalled ? Theme.warn : Theme.accent, height: 3)
                Text(status).font(.system(size: 10.5).monospacedDigit())
                    .foregroundStyle(clip.isStalled ? Theme.warn : Theme.text3).lineLimit(1)
            } else {
                Text(queued).font(.system(size: 10.5).monospacedDigit()).foregroundStyle(Theme.text3).lineLimit(1)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { model.selectedID = clip.id }
    }
    /// "Queued · 0:04 clip · ≈ 45 s" — the pre-flight estimate (past runs on this Mac, else 30 s + 3× clip length).
    private var queued: String {
        var s = "Queued" + (clip.probe.map { " · \(Fmt.duration($0.durationS)) clip" } ?? "")
        if let e = model.preflight(clip)?.estimate, e.isFinite, e > 0 {
            s += e < 60 ? " · ≈ \(max(5, Int((e / 5).rounded() * 5))) s" : " · ≈ \(Int((e / 60).rounded())) min"
        }
        return s
    }
    private var status: String {
        let pct = "\(Int(((clip.progress?.fraction ?? 0) * 100).rounded()))%"
        if clip.isStalled { return "\(pct) · no update for \(Fmt.duration(Double(clip.stallSeconds)))" }
        let stage = clip.progress?.label ?? "Starting"
        return "\(pct) · \(stage) · " + (clip.etaSeconds.map { Fmt.eta($0) } ?? "estimating…")
    }
}

// MARK: - Export queue (sidebar footer)

struct ExportQueuePanel: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        if !model.exports.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    SectionLabel(text: "Exports")
                    Spacer()
                    if model.exports.contains(where: { $0.state != .queued && $0.state != .running }) {
                        Button("Clear") { model.clearFinishedExports() }
                            .buttonStyle(.plain).font(.spSmall).foregroundStyle(Theme.text3)
                    }
                }
                ForEach(model.exports.suffix(5)) { e in ExportRow(job: e) }
            }
            .padding(14)
            .background(Theme.raised.opacity(0.6))
            .overlay(alignment: .top) { Rectangle().fill(Theme.hairline).frame(height: 1) }
        }
    }
}

struct ExportRow: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var job: ExportJob
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 6) {
                Text(job.output.lastPathComponent).font(.system(size: 11.5, weight: .medium)).lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 4)
                switch job.state {
                case .running, .queued:
                    Button { model.cancelExport(job) } label: { Image(systemName: "xmark") }
                        .buttonStyle(IconButtonStyle(size: 18)).font(.system(size: 9)).help("Cancel")
                case .done:
                    Button { model.reveal(job.output) } label: { Image(systemName: "magnifyingglass") }
                        .buttonStyle(IconButtonStyle(size: 18)).help("Reveal in Finder")
                default: EmptyView()
                }
            }
            switch job.state {
            case .queued:
                Text("\(job.preset.short) · Queued").font(.system(size: 10.5)).foregroundStyle(Theme.text3)
            case .running:
                SPProgressBar(fraction: job.fraction, height: 3)
                Text("\(Int((job.fraction * 100).rounded()))% · \(Fmt.eta(job.eta))").font(.system(size: 10.5).monospacedDigit())
                    .foregroundStyle(Theme.text3)
                if !job.message.isEmpty {
                    Text(job.message).font(.system(size: 10).monospacedDigit()).foregroundStyle(Theme.text3).lineLimit(1)
                }
            case .done:
                Text("\(job.preset.short) · \(job.bytes.map(Fmt.bytes) ?? "done")"
                     + (job.seconds.map { String(format: " · %.0f s", $0) } ?? ""))
                    .font(.system(size: 10.5)).foregroundStyle(Theme.good)
            case .failed(let m):
                Text(m).font(.system(size: 10.5)).foregroundStyle(Theme.bad).lineLimit(2)
            case .cancelled:
                Text("Cancelled").font(.system(size: 10.5)).foregroundStyle(Theme.text3)
            }
        }
    }
}

// MARK: - Empty state

struct EmptyState: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        ZStack {
            Theme.window
            VStack(spacing: 0) {
            HStack(spacing: 10) {
                LogoMark(size: 24)
                Text("Stillpoint").font(.system(size: 20, weight: .semibold)).tracking(0.2)
            }
            .padding(.bottom, 6)
            Text("Gyro + vision stabilization for FPV footage")
                .font(.system(size: 12.5)).foregroundStyle(Theme.text3).padding(.bottom, 26)
            VStack(spacing: 0) {
                ZStack {
                    Circle().fill(Theme.raised2).frame(width: 84, height: 84)
                    Image(systemName: "arrow.down.to.line")
                        .font(.system(size: 28, weight: .medium)).foregroundStyle(Theme.text2)
                }
                .padding(.bottom, 22)
                Text("Drop FPV clips to stabilize").font(.system(size: 24, weight: .semibold)).padding(.bottom, 8)
                Text("DJI O3 · O4 Pro · Avata · Osmo Action 4 — recorded with EIS off.")
                    .font(.system(size: 13.5)).foregroundStyle(Theme.text2).padding(.bottom, 26)
                Button { model.openPanel() } label: { Text("Open Clips…") }
                    .buttonStyle(PrimaryButtonStyle()).frame(width: 170).padding(.bottom, 26)
                HStack(spacing: 22) {
                    feature("gyroscope", "Reads the camera's own gyro")
                    feature("scope", "Measures leftover jitter in the picture")
                    feature("lock.shield", "Runs entirely on this Mac")
                }
                if !model.engineProblems.isEmpty {
                    Text("Engine: " + model.engineProblems.joined(separator: " · "))
                        .font(.spSmall).foregroundStyle(Theme.warn).padding(.top, 22)
                }
            }
            .padding(.horizontal, 56).padding(.vertical, 50)
            .background(RoundedRectangle(cornerRadius: 26, style: .continuous).fill(Theme.panel))
            .overlay(RoundedRectangle(cornerRadius: 26, style: .continuous)
                .strokeBorder(model.dropTargeted ? Theme.accent : Theme.hairlineStrong,
                              style: StrokeStyle(lineWidth: model.dropTargeted ? 2 : 1.2, dash: [8, 7])))
            }
            .padding(40)
        }
    }

    private func feature(_ icon: String, _ text: String) -> some View {
        HStack(spacing: 7) {
            Image(systemName: icon).font(.system(size: 11.5)).foregroundStyle(Theme.accent)
            Text(text).font(.system(size: 11.5)).foregroundStyle(Theme.text3)
        }
    }
}

// MARK: - Viewer

struct ViewerPane: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var clip: Clip
    @ObservedObject var player: PlayerController
    @Environment(\.snapshotMode) private var snapshot

    var body: some View {
        VStack(spacing: 0) {
            header
            GeometryReader { g in
                let transportH: CGFloat = clip.isAnalyzed ? 112 : 64
                let avail = CGSize(width: max(g.size.width - 48, 100), height: max(g.size.height - transportH - 36, 100))
                let vs = fittedSize(player.renderSize, in: avail)
                VStack(spacing: 14) {
                    VideoStage(clip: clip, player: player)
                        .frame(width: vs.width, height: vs.height)
                    TransportBar(clip: clip, player: player)
                        .frame(width: min(max(vs.width, 560), g.size.width - 48))
                }
                .frame(width: g.size.width, height: g.size.height)
                .offset(y: -8)
            }
        }
        .background(Theme.window)
        .background {
            if !snapshot {
                Group {
                    Button("") { player.togglePlay() }.keyboardShortcut(.space, modifiers: [])
                    Button("") { player.step(-1) }.keyboardShortcut(.leftArrow, modifiers: [])
                    Button("") { player.step(1) }.keyboardShortcut(.rightArrow, modifiers: [])
                    Button("") {
                        player.settings.mode = player.settings.mode == .after ? .before : .after
                    }.keyboardShortcut("\\", modifiers: [])
                }
                .opacity(0).allowsHitTesting(false)
            }
        }
    }

    private func fittedSize(_ content: CGSize, in box: CGSize) -> CGSize {
        guard content.width > 0, content.height > 0 else { return box }
        let s = min(box.width / content.width, box.height / content.height)
        return CGSize(width: (content.width * s).rounded(), height: (content.height * s).rounded())
    }

    private var header: some View {
        HStack(spacing: 10) {
            Spacer()
            Text(clip.name).font(.system(size: 13, weight: .semibold))
            if let p = clip.probe {
                Text([p.cameraName, "\(p.width)×\(p.height)", Fmt.fps(p.fps)].joined(separator: "  ·  "))
                    .font(.system(size: 11.5)).foregroundStyle(Theme.text3)
            }
            Spacer()
        }
        .frame(height: 52)
    }
}

/// The picture. Live: AVPlayerLayer fed by the Metal compositor. Snapshot: a still rendered by the same kernel.
struct VideoStage: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var clip: Clip
    @ObservedObject var player: PlayerController
    @Environment(\.snapshotMode) private var snapshot

    var body: some View {
        GeometryReader { g in
            let rect = CGRect(origin: .zero, size: g.size)
            ZStack(alignment: .topLeading) {
                Color.black
                if snapshot {
                    if let img = model.snapshotFrame {
                        Image(decorative: img, scale: 1).resizable().interpolation(.high)
                            .aspectRatio(contentMode: .fit)
                            .frame(width: rect.width, height: rect.height)
                    }
                } else {
                    PlayerLayerView(player: player.player)
                }
                overlays(rect)
            }
            .contentShape(Rectangle())
            .gesture(splitDrag(rect), including: player.settings.mode == .split && player.hasPlan ? .all : .none)
            .onTapGesture(count: 2) { player.togglePlay() }
        }
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.white.opacity(0.09)))
        .shadow(color: .black.opacity(0.55), radius: 24, y: 10)
    }

    private func splitDrag(_ rect: CGRect) -> some Gesture {
        DragGesture(minimumDistance: 0).onChanged { d in
            guard rect.width > 0 else { return }
            player.settings.split = Float(((d.location.x - rect.minX) / rect.width).clamped(0.02, 0.98))
        }
    }

    @ViewBuilder private func overlays(_ rect: CGRect) -> some View {
        let mode = player.hasPlan ? player.settings.mode : .before
        ZStack(alignment: .topLeading) {
            if mode == .split {
                let x = rect.minX + rect.width * CGFloat(player.settings.split)
                Rectangle().fill(Color.white.opacity(0.92)).frame(width: 1.5, height: rect.height)
                    .shadow(color: .black.opacity(0.5), radius: 2)
                    .offset(x: x - 0.75, y: rect.minY)
                ZStack {
                    Circle().fill(Color.black.opacity(0.55)).frame(width: 34, height: 34)
                    Circle().strokeBorder(Color.white.opacity(0.9), lineWidth: 1.5).frame(width: 34, height: 34)
                    HStack(spacing: 3) {
                        Image(systemName: "chevron.left")
                        Image(systemName: "chevron.right")
                    }.font(.system(size: 9, weight: .bold)).foregroundStyle(.white)
                }
                .offset(x: x - 17, y: rect.midY - 17)
                pill("ORIGINAL", dot: nil).offset(x: rect.minX + 14, y: rect.minY + 14)
                pill("STILLPOINT", dot: Theme.accent)
                    .frame(width: 140, alignment: .trailing)
                    .offset(x: rect.maxX - 154, y: rect.minY + 14)
            } else if mode == .after {
                pill("STABILIZED", dot: Theme.accent).offset(x: rect.minX + 14, y: rect.minY + 14)
            } else {
                let unsupported = clip.probe?.supported == false
                pill(player.hasPlan || unsupported ? "ORIGINAL" : "ORIGINAL · NOT ANALYZED", dot: nil)
                    .offset(x: rect.minX + 14, y: rect.minY + 14)
            }
            if clip.phase == .running, let p = clip.progress {
                HStack(spacing: 7) {
                    SPRing(fraction: p.fraction, size: 13)
                    Text(clip.isStalled ? "Analyzing \(Int((p.fraction * 100).rounded()))% · no update" : "Analyzing \(Int((p.fraction * 100).rounded()))%")
                        .font(.system(size: 10.5, weight: .semibold))
                        .tracking(0.3)
                }
                .padding(.horizontal, 9).padding(.vertical, 5)
                .background(Capsule().fill(Color.black.opacity(0.6)))
                .frame(width: 240, alignment: .trailing)
                .offset(x: rect.maxX - 254, y: rect.maxY - 40)
            }
            if let err = player.error {
                Text(err).font(.spSmall).foregroundStyle(Theme.bad).padding(8)
                    .background(RoundedRectangle(cornerRadius: 6).fill(Color.black.opacity(0.7)))
                    .offset(x: rect.minX + 14, y: rect.maxY - 44)
            }
        }
        .allowsHitTesting(false)
    }

    private func pill(_ text: String, dot: Color?) -> some View {
        HStack(spacing: 6) {
            if let dot { Circle().fill(dot).frame(width: 6, height: 6) }
            Text(text).font(.system(size: 10, weight: .semibold)).tracking(1.0)
        }
        .foregroundStyle(.white.opacity(0.92))
        .padding(.horizontal, 9).padding(.vertical, 5)
        .background(Capsule().fill(Color.black.opacity(0.55)))
        .fixedSize()
    }
}

// MARK: - Transport

struct TransportBar: View {
    @ObservedObject var clip: Clip
    @ObservedObject var player: PlayerController

    var body: some View {
        VStack(spacing: 10) {
            Scrubber(player: player, summary: clip.manifest?.summary, analyzed: clip.isAnalyzed)
            HStack(spacing: 4) {
                Button { player.step(-1) } label: { Image(systemName: "backward.frame.fill") }
                    .buttonStyle(IconButtonStyle(size: 30)).help("Previous frame (←)")
                Button { player.togglePlay() } label: {
                    Image(systemName: player.isPlaying ? "pause.fill" : "play.fill").font(.system(size: 14))
                        .foregroundStyle(Theme.text)
                        .frame(width: 36, height: 36)
                        .background(Circle().fill(Color.white.opacity(0.10)))
                }
                .buttonStyle(.plain).help("Play / pause (space)")
                Button { player.step(1) } label: { Image(systemName: "forward.frame.fill") }
                    .buttonStyle(IconButtonStyle(size: 30)).help("Next frame (→)")
                HStack(spacing: 5) {
                    Text(Fmt.clock(player.time)).foregroundStyle(Theme.text)
                    Text("/").foregroundStyle(Theme.text3)
                    Text(Fmt.clock(player.duration)).foregroundStyle(Theme.text3)
                    Text(verbatim: "F \(Int((player.time * player.fps + 1e-3).rounded(.down)))").foregroundStyle(Theme.text3)
                        .font(.spMono(11)).padding(.leading, 8)
                }
                .font(.spMono(12.5, .medium))
                .padding(.leading, 10)
                Spacer()
                Button { player.loop.toggle() } label: { Image(systemName: "repeat") }
                    .buttonStyle(IconButtonStyle(size: 28, active: player.loop)).help("Loop playback")
                SPSegmented(options: [(CompareMode.after, "After"), (.split, "Split"), (.before, "Before")],
                            selection: Binding(get: { player.settings.mode }, set: { player.settings.mode = $0 }))
                    .frame(width: 216)
                    .disabled(!player.hasPlan)
                    .opacity(player.hasPlan ? 1 : 0.4)
                    .help("Compare with the original (\\ toggles)")
            }
        }
    }
}

/// Timeline scrubber with a per-second jitter strip (original shake vs what Stillpoint leaves).
struct Scrubber: View {
    @ObservedObject var player: PlayerController
    let summary: JitterSummary?
    let analyzed: Bool
    @State private var dragging = false

    var body: some View {
        GeometryReader { g in
            let w = g.size.width
            let f = player.duration > 0 ? CGFloat(player.time / player.duration).clamped(0, 1) : 0
            VStack(spacing: 6) {
                strip(width: w)
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.10)).frame(height: 4)
                    Capsule().fill(Theme.text.opacity(0.85)).frame(width: max(4, w * f), height: 4)
                    RoundedRectangle(cornerRadius: 1.5).fill(Color.white)
                        .frame(width: 3, height: dragging ? 18 : 14)
                        .shadow(color: .black.opacity(0.5), radius: 2)
                        .offset(x: w * f - 1.5)
                }
                .frame(height: 18)
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { d in
                    dragging = true
                    if player.isPlaying { player.pause() }
                    player.seek(to: Double((d.location.x / max(w, 1)).clamped(0, 1)) * player.duration)
                }
                .onEnded { d in
                    dragging = false
                    player.seek(to: Double((d.location.x / max(w, 1)).clamped(0, 1)) * player.duration, exact: true)
                })
        }
        .frame(height: analyzed ? 66 : 18)
    }

    @ViewBuilder private func strip(width w: CGFloat) -> some View {
        if analyzed, let s = summary, let orig = s.winOrigPx, !orig.isEmpty {
            let qwin = s.quality?.windows ?? []
            let independent = !qwin.isEmpty
            let fin = independent ? [] : (s.winFinalPx ?? [])
            let winS = s.windowS ?? 1
            let vals = orig.compactMap { $0 }.sorted()
            // scale to the 95th percentile so typical seconds are readable; the loudest seconds clip at the top
            let top = max(vals.last ?? 1, 0.2)
            let n = orig.count
            let bw = w / CGFloat(n)
            VStack(spacing: 3) {
                HStack(spacing: 12) {
                    Text("JITTER PER SECOND").font(.system(size: 9, weight: .semibold)).tracking(0.8).foregroundStyle(Theme.text3)
                    Spacer()
                    legend(Theme.shake.opacity(0.7), "Original (gyro)")
                    legend(Theme.accent, independent ? "Stabilized (measured windows)" : "Residual (self-measured)")
                }
                Canvas { ctx, size in
                    let H = size.height
                    // square-root scale: quiet seconds stay readable next to the loudest ones
                    func y(_ v: Double) -> CGFloat { H - max(0, H * CGFloat(min((max(v, 0) / top).squareRoot(), 1))) }
                    func xc(_ i: Int) -> CGFloat { (CGFloat(i) + 0.5) * bw }
                    func curve(_ vals: [Double?], floor: Double) -> Path {
                        var p = Path()
                        p.move(to: CGPoint(x: 0, y: y(vals.first.flatMap { $0 } ?? 0)))
                        for i in 0..<vals.count {
                            let pt = CGPoint(x: xc(i), y: y(max(vals[i] ?? 0, floor)))
                            if i == 0 { p.addLine(to: pt); continue }
                            let prev = CGPoint(x: xc(i - 1), y: y(max(vals[i - 1] ?? 0, floor)))
                            let mx = (prev.x + pt.x) / 2
                            p.addCurve(to: pt, control1: CGPoint(x: mx, y: prev.y), control2: CGPoint(x: mx, y: pt.y))
                        }
                        p.addLine(to: CGPoint(x: size.width, y: y(max(vals.last.flatMap { $0 } ?? 0, floor))))
                        return p
                    }
                    let o = curve(orig, floor: 0)
                    var area = o
                    area.addLine(to: CGPoint(x: size.width, y: H))
                    area.addLine(to: CGPoint(x: 0, y: H))
                    area.closeSubpath()
                    ctx.fill(area, with: .linearGradient(Gradient(colors: [Theme.shake.opacity(0.42), Theme.shake.opacity(0.08)]),
                                                         startPoint: CGPoint(x: 0, y: 0), endPoint: CGPoint(x: 0, y: H)))
                    ctx.stroke(o, with: .color(Theme.shake.opacity(0.85)), lineWidth: 1)
                    if !fin.isEmpty {
                        ctx.stroke(curve(fin, floor: 0), with: .color(Theme.accent), lineWidth: 1.6)
                    }
                    // independent measurement: the sampled windows where it was taken, at their place in the clip
                    let clipLen = Double(n) * winS
                    for w in qwin where clipLen > 0 {
                        let x0 = CGFloat(min(max(w.t0S / clipLen, 0), 1)) * size.width
                        let x1 = CGFloat(min(max(w.t1S / clipLen, 0), 1)) * size.width
                        ctx.fill(Path(CGRect(x: x0, y: 0, width: max(x1 - x0, 2), height: H)), with: .color(Theme.accent.opacity(0.07)))
                        if let v = w.stabilized {
                            ctx.fill(Path(CGRect(x: x0, y: y(v) - 1, width: max(x1 - x0, 2), height: 2)), with: .color(Theme.accent))
                        }
                    }
                    ctx.fill(Path(CGRect(x: 0, y: H - 0.5, width: size.width, height: 0.5)), with: .color(Color.white.opacity(0.1)))
                }
                .frame(height: 26)
            }
        }
    }

    private func legend(_ c: Color, _ t: String) -> some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 1).fill(c).frame(width: 7, height: 7)
            Text(t).font(.system(size: 9.5, weight: .medium)).foregroundStyle(Theme.text3)
        }
    }
}
