// Right-hand inspector: clip facts, analysis (progress / jitter readout), stabilisation controls, export.
import AppKit
import SwiftUI

struct Inspector: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var clip: Clip

    var body: some View {
        AdaptiveScroll {
            VStack(alignment: .leading, spacing: 18) {
                InspectorHeader(clip: clip)
                AnalysisSection(clip: clip)
                if clip.probe?.supported != false {
                    StabilizationSection(clip: clip)
                    ExportSection(clip: clip)
                }
            }
            .padding(.horizontal, 18).padding(.top, 20).padding(.bottom, 20)
        }
        .background(Theme.panel)
    }
}

struct InspectorHeader: View {
    @ObservedObject var clip: Clip
    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text(clip.name).font(.system(size: 15, weight: .semibold)).lineLimit(1).truncationMode(.middle)
            if let p = clip.probe {
                HStack(spacing: 5) {
                    Chip(text: p.cameraName.replacingOccurrences(of: "DJI Osmo", with: "Osmo"))
                    Chip(text: Fmt.resolution(p.width, p.height))
                    Chip(text: String(format: "%.2f", p.fps))
                    if let b = p.bitDepth { Chip(text: "\(b)-bit") }
                    if let h = p.hdrLabel { Chip(text: h) }
                    Chip(text: Fmt.duration(p.durationS))
                }
                StatusBadge(label: p.gyro.label, level: p.gyro.level)
                    .help(p.gyro.detail)
            } else if let e = clip.probeError {
                Text(e).font(.spSmall).foregroundStyle(Theme.bad).lineLimit(3)
            } else {
                Text("Reading clip…").font(.spSmall).foregroundStyle(Theme.text3)
            }
        }
    }
}

// MARK: - Analysis

struct AnalysisSection: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var clip: Clip

    var body: some View {
        switch clip.phase {
        case .running, .queued:
            running
        default:
            if let p = clip.probe, !p.supported {
                unsupported(p)
            } else if clip.isAnalyzed, let s = clip.manifest?.summary {
                VStack(alignment: .leading, spacing: 10) {
                    if let f = clip.failure { FailurePanel(clip: clip, info: f) }
                    JitterReadout(summary: s, params: clip.manifest?.params)
                }
            } else if let f = clip.failure {
                VStack(alignment: .leading, spacing: 10) {
                    FailurePanel(clip: clip, info: f)
                    if let pf = model.preflight(clip) { Card(padding: 14) { PreflightView(pf: pf) } }
                }
            } else {
                readyCard
            }
        }
    }

    private var readyCard: some View {
        Card(padding: 16) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Image(systemName: "waveform.path.ecg").foregroundStyle(Theme.accent)
                    Text(clip.phase == .cancelled ? "Analysis cancelled" : "Ready to stabilize").font(.system(size: 13.5, weight: .semibold))
                }
                Text(readyText).font(.system(size: 12)).foregroundStyle(Theme.text2).fixedSize(horizontal: false, vertical: true)
                if let pf = model.preflight(clip) { PreflightView(pf: pf) }
                let blocked = model.preflight(clip)?.blocked ?? false
                Button { model.analyze(clip) } label: {
                    HStack(spacing: 6) { Image(systemName: "sparkles"); Text("Analyze") }
                }
                .buttonStyle(PrimaryButtonStyle(enabled: clip.canAnalyze && !blocked))
                .disabled(!clip.canAnalyze || blocked)
            }
        }
    }

    private var readyText: String {
        var s = "Stillpoint reads the camera's gyro, then measures the picture itself to remove the jitter the gyro misses."
        if clip.probe?.gyro.level == "limited" {
            s += " This clip only has per-frame attitude, so fine jitter removal is limited."
        }
        return s
    }

    private var running: some View {
        Card(padding: 16) {
            VStack(alignment: .leading, spacing: 11) {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(clip.phase == .queued ? "Waiting to analyze" : (clip.progress?.label ?? "Starting"))
                        .font(.system(size: 13.5, weight: .semibold)).lineLimit(1)
                    Spacer(minLength: 4)
                    if clip.phase == .running {
                        if !clip.isStalled {
                            Text(etaText).font(.system(size: 11).monospacedDigit()).foregroundStyle(Theme.text3).fixedSize()
                        }
                        Text("\(Int(((clip.progress?.fraction ?? 0) * 100).rounded()))%").font(.spMono(13.5, .semibold))
                            .foregroundStyle(Theme.text).fixedSize()
                    }
                }
                SPProgressBar(fraction: clip.progress?.fraction ?? 0, tint: clip.isStalled ? Theme.warn : Theme.accent, height: 5)
                Text(clip.phase == .queued ? queuedText : (clip.progress?.message ?? "Launching the engine…"))
                    .font(.system(size: 11).monospacedDigit()).foregroundStyle(Theme.text3).lineLimit(1)
                StageSteps(stage: clip.progress?.stage ?? "", message: clip.progress?.message ?? "")
                if clip.isStalled { StallBanner(clip: clip) }
                ForEach(clip.notices, id: \.self) { n in
                    NoteRow(item: .init(level: .warn, icon: "info.circle", text: n))
                }
                if clip.phase == .running, clip.elapsed > 0 {
                    Text("Running for \(Fmt.duration(clip.elapsed))" + (clip.probe.map { " · clip \(Fmt.duration($0.durationS))" } ?? ""))
                        .font(.system(size: 10.5).monospacedDigit()).foregroundStyle(Theme.text3)
                }
                Button { model.cancelAnalysis(clip) } label: {
                    Text(clip.phase == .queued ? "Remove from Queue" : (clip.isStalled ? "Cancel Analysis" : "Cancel"))
                }
                .buttonStyle(SecondaryButtonStyle(fill: true))
                .help("Stops the engine and everything it started. The previous analysis, if any, is kept.")
            }
        }
    }

    private var queuedText: String {
        if let r = model.analysisRunning, r.id != clip.id { return "After \(r.displayName) finishes." }
        return "Another clip is being analyzed."
    }

    private var etaText: String {
        guard let e = clip.etaSeconds else { return "estimating…" }
        return clip.etaIsEngine ? Fmt.eta(e) : Fmt.eta(e).replacingOccurrences(of: " left", with: " left (est.)")
    }

    private func unsupported(_ p: ProbeInfo) -> some View {
        Card(padding: 16) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(Theme.bad)
                    Text("Can't stabilize this clip yet").font(.system(size: 13.5, weight: .semibold))
                }
                Text(p.gyro.detail).font(.system(size: 12)).foregroundStyle(Theme.text2)
                    .fixedSize(horizontal: false, vertical: true)
                Text("You can still preview the original.").font(.spSmall).foregroundStyle(Theme.text3)
            }
        }
    }
}

/// Clip length, time estimate and disk / source checks shown before an analysis starts.
struct PreflightView: View {
    let pf: Preflight
    var compact = false
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !compact {
                HStack(spacing: 0) {
                    fact("Clip", Fmt.duration(pf.duration))
                    fact("Analysis", Fmt.approx(pf.estimate).replacingOccurrences(of: "about ", with: "≈ "))
                    fact("Free disk", pf.freeBytes.map(Fmt.gb) ?? "–")
                }
                .padding(.vertical, 8)
                .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Color.black.opacity(0.22)))
                Text("Estimate " + pf.estimateBasis + ".").font(.system(size: 10.5)).foregroundStyle(Theme.text3)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("Re-analyzing takes \(Fmt.approx(pf.estimate)).").font(.system(size: 10.5))
                    .foregroundStyle(Theme.text3).help("Estimate " + pf.estimateBasis + ".")
            }
            ForEach(pf.items, id: \.self) { NoteRow(item: $0) }
        }
    }

    private func fact(_ k: String, _ v: String) -> some View {
        VStack(spacing: 2) {
            Text(v).font(.spMono(13, .semibold)).foregroundStyle(Theme.text).lineLimit(1).minimumScaleFactor(0.8)
            Text(k.uppercased()).font(.system(size: 8.5, weight: .semibold)).tracking(0.7).foregroundStyle(Theme.text3)
        }
        .frame(maxWidth: .infinity)
    }
}

struct NoteRow: View {
    let item: Preflight.Item
    var body: some View {
        let c = item.level == .block ? Theme.bad : (item.level == .warn ? Theme.warn : Theme.text2)
        HStack(alignment: .top, spacing: 7) {
            Image(systemName: item.icon).font(.system(size: 11, weight: .semibold)).foregroundStyle(c).frame(width: 14)
                .padding(.top, 1)
            Text(item.text).font(.system(size: 11)).foregroundStyle(item.level == .info ? Theme.text2 : c.opacity(0.95))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 9).padding(.vertical, 7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 7, style: .continuous).fill(c.opacity(0.09)))
    }
}

/// Shown when the engine has not reported progress for a while: honest about what we know, with Cancel.
struct StallBanner: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var clip: Clip
    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 7) {
                Image(systemName: "hourglass").font(.system(size: 11.5, weight: .semibold)).foregroundStyle(Theme.warn)
                Text("Still working — no update for \(Fmt.duration(Double(clip.stallSeconds)))")
                    .font(.system(size: 12, weight: .semibold)).foregroundStyle(Theme.text)
            }
            Text(detail).font(.system(size: 11)).foregroundStyle(Theme.text2).fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 9, style: .continuous).fill(Theme.warn.opacity(0.09)))
        .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(Theme.warn.opacity(0.28)))
    }
    private var detail: String {
        switch clip.engineBusy {
        case .some(true): return "The engine is busy (using CPU) but has not reported progress. Long clips can pause like this while a path is solved."
        case .some(false): return "The engine is idle — it may be waiting on the disk or stuck. If this lasts, cancel it below: nothing is lost, the previous analysis is kept."
        case .none: return "Checking whether the engine is still busy…"
        }
    }
}

/// Readable error panel: what happened, the engine's last output lines, Retry.
struct FailurePanel: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var clip: Clip
    let info: FailureInfo
    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 7) {
                Image(systemName: "xmark.octagon.fill").font(.system(size: 12.5)).foregroundStyle(Theme.bad)
                Text(info.title).font(.system(size: 12.5, weight: .semibold)).lineLimit(2)
            }
            Text(info.message).font(.system(size: 11.5)).foregroundStyle(Theme.text2)
                .fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
            if !info.details.isEmpty {
                VStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(info.details.suffix(10).enumerated()), id: \.offset) { _, l in
                        Text(l).font(.system(size: 10, design: .monospaced)).foregroundStyle(Theme.text2)
                            .lineLimit(3).fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .textSelection(.enabled)
                .padding(8)
                .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Color.black.opacity(0.35)))
            }
            HStack(spacing: 8) {
                if info.canRetry {
                    Button { model.retry(clip) } label: { HStack(spacing: 5) { Image(systemName: "arrow.clockwise"); Text("Retry") } }
                        .buttonStyle(PrimaryButtonStyle(tint: Theme.accent, enabled: clip.probe?.supported == true))
                        .frame(width: 96)
                }
                if !info.details.isEmpty {
                    Button("Copy Details") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(([info.title, info.message] + info.details).joined(separator: "\n"), forType: .string)
                    }
                    .buttonStyle(SecondaryButtonStyle())
                }
                Spacer()
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Theme.bad.opacity(0.08)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Theme.bad.opacity(0.25)))
    }
}

/// Three-step stage indicator under the progress bar.
struct StageSteps: View {
    let stage: String
    let message: String
    var body: some View {
        let order = ["telemetry", "path", "measure", "quality", "finalize"]
        let idx = order.firstIndex(of: stage) ?? 0
        HStack(spacing: 0) {
            step("Gyro", done: idx > 0, active: idx == 0)
            line(done: idx > 0)
            step("Path", done: idx > 1, active: idx == 1)
            line(done: idx >= 2)
            step("Vision", done: idx > 2, active: idx == 2)
            line(done: idx > 2)
            step("Check", done: idx > 3, active: idx == 3)
            line(done: idx > 3)
            step("Plan", done: false, active: idx == 4)
        }
    }
    private func step(_ t: String, done: Bool, active: Bool) -> some View {
        HStack(spacing: 4) {
            Circle().fill(done ? Theme.accent : (active ? Theme.accent.opacity(0.9) : Color.white.opacity(0.15)))
                .frame(width: 6, height: 6)
                .overlay(Circle().stroke(active ? Theme.accent.opacity(0.35) : .clear, lineWidth: 4))
            Text(t).font(.system(size: 10.5, weight: active ? .semibold : .regular))
                .foregroundStyle(active ? Theme.text : (done ? Theme.text2 : Theme.text3))
        }
        .fixedSize()
    }
    private func line(done: Bool) -> some View {
        Rectangle().fill(done ? Theme.accent.opacity(0.5) : Color.white.opacity(0.1)).frame(height: 1).padding(.horizontal, 5)
    }
}

/// The jitter readout. With report.json['quality'] (the engine's independent measurement): original vs stabilized.
/// Without it: the gyro's original shake and the closed loop's own residual, labelled as self-measured.
/// Only numbers the engine produced are shown — no derived percentages, no floors.
struct JitterReadout: View {
    let summary: JitterSummary
    var params: AnalysisParams? = nil

    var body: some View {
        if let q = summary.quality, !q.metrics.isEmpty { independent(q) } else { selfMeasured }
    }

    // MARK: independent measurement

    private func independent(_ q: QualitySummary) -> some View {
        let head = q.metric("hf") ?? q.metrics[0]
        let worse = (head.stabilized ?? 0) > (head.original ?? .infinity)
        return Card(padding: 16) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 6) {
                    SectionLabel(text: "Jitter · measured")
                    Image(systemName: "info.circle").font(.system(size: 10.5)).foregroundStyle(Theme.text3)
                        .help("Measured independently of the stabilizer's own loop: " + (q.method ?? "method not reported by the engine"))
                    Spacer()
                    tag("INDEPENDENT", Theme.good)
                }
                HStack(alignment: .lastTextBaseline, spacing: 0) {
                    bigNumber(head.original, "Original", Theme.shake)
                    Image(systemName: "arrow.right").font(.system(size: 12, weight: .semibold)).foregroundStyle(Theme.text3)
                        .padding(.horizontal, 14).alignmentGuide(.lastTextBaseline) { d in d[.bottom] + 22 }
                    bigNumber(head.stabilized, "Stabilized", Theme.text)
                    Spacer(minLength: 0)
                }
                Text(head.label + (worse ? " — higher than the original on this clip" : ""))
                    .font(.system(size: 10.5)).foregroundStyle(worse ? Theme.warn : Theme.text3).padding(.top, -6)
                VStack(spacing: 6) {
                    ForEach(q.metrics.filter { $0.key != head.key }, id: \.key) { m in
                        row(m.label, "\(Fmt.pair(m.original, m.stabilized)) px",
                            warn: (m.stabilized ?? 0) > (m.original ?? .infinity))
                    }
                    if let n = q.newJumpsGt1px {
                        row("New frame steps > 1 px", "\(Int(n))" + (q.newJumpsMaxPx.map { " · largest \(Fmt.px($0)) px" } ?? ""),
                            warn: n > 0)
                    }
                    row("Field of view", fovText)
                    if let c = q.cropFootprintMean { row("Source frame used", "\(Int((c * 100).rounded()))%") }
                    if let c = q.coverageFrac { row("Measured on", "\(Int((c * 100).rounded()))% of the clip (sampled)") }
                    optionRows
                }
                if let t = summary.timecal { TimingReadout(t: t) }
                Rectangle().fill(Theme.hairline).frame(height: 1)
                Text("\(q.units ?? "px @1080p"). Method: \(q.method ?? "not reported by the engine").")
                    .font(.system(size: 10)).foregroundStyle(Theme.text3).lineLimit(4)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
        }
    }

    // MARK: self-measured fallback

    private var selfMeasured: some View {
        let o = summary.origPx
        return Card(padding: 16) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 6) {
                    SectionLabel(text: "Jitter")
                    Image(systemName: "info.circle").font(.system(size: 10.5)).foregroundStyle(Theme.text3)
                        .help(explainer)
                    Spacer()
                    tag("NO INDEPENDENT CHECK", Theme.warn)
                }
                bigNumber(o, "Original shake above 2 Hz (from the gyro)", Theme.shake)
                VStack(spacing: 6) {
                    row("Original 8–30 Hz (gyro)", "\(Fmt.px(summary.origB830Px)) px")
                    row("Gyro-only residual (self-measured)", "\(Fmt.px(summary.gyroOnlyPx)) px")
                    row("Closed-loop residual (self-measured)", "\(Fmt.px(summary.finalPx)) px")
                    row("Field of view", fovText)
                    if let t = summary.trustedFrac {
                        row("Checked by vision", "\(Int((t * 100).rounded()))% of frames", warn: t < 0.5)
                    }
                    optionRows
                }
                if let t = summary.timecal { TimingReadout(t: t) }
                Rectangle().fill(Theme.hairline).frame(height: 1)
                (Text(summary.qualityError.map { "Independent check unavailable: \($0). " }
                      ?? "This analysis predates the independent check — re-analyze to measure original vs stabilized. ")
                    .foregroundColor(Theme.warn.opacity(0.9))
                 + Text("Residuals are the \(summary.finalMethod ?? "closed-loop residual (self-measured)"): the vision loop grading its own output, so optimistic.")
                    .foregroundColor(Theme.text3))
                    .font(.system(size: 10.5)).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// The engine v5 options this analysis used, with what they did (numbers from the engine's report only).
    @ViewBuilder private var optionRows: some View {
        if params?.horizonLock == true {
            let st = Int(((params?.horizonStrength ?? 1) * 100).rounded()), bank = Int((params?.rollLimitDeg ?? 0).rounded())
            let lvl = summary.horizon?.fullLevelFrac.map { " · level \(Int(($0 * 100).rounded()))% of frames" } ?? ""
            row("Horizon lock", "\(st)%" + (bank > 0 ? " · ±\(bank)°" : "") + lvl)
        }
        if params?.fill == true {
            if let e = summary.fill?.error { row("Full-frame fill", "failed: \(e.prefix(40))", warn: true) }
            else { row("Full-frame fill", summary.fill?.framesFrac.map { "on · \(Fmt.pct($0)) of frames filled" } ?? "on") }
        }
        if params?.mesh == true {
            if let e = summary.mesh?.error { row("Max quality", "failed: \(e.prefix(40))", warn: true) }
            else { row("Max quality", summary.mesh?.offsetRmsPx.map { "mesh · \(Fmt.px($0)) px rms" } ?? "on") }
        }
    }

    private var fovText: String {
        let fov = summary.hfovDeg.map { "\(Int($0.rounded()))°" } ?? "–"
        if let z = summary.zoomMax, z > 1.005 {
            return "\(fov) · zooms ≤\(String(format: "%.2f", z))× in hard moves"
        }
        return "\(fov) · fixed crop"
    }

    private let explainer = """
    Rotational jitter above 2 Hz, in pixels of a 1080p frame. Original: the camera's own shake, from its gyro. \
    Gyro only / Residual: what the closed loop measured on its own stabilized preview before and after its vision \
    corrections (self-measured, so optimistic). Parallax and forward-motion wobble are not included.
    """

    private func tag(_ t: String, _ c: Color) -> some View {
        Text(t).font(.system(size: 8.5, weight: .bold)).tracking(0.7).foregroundStyle(c)
            .padding(.horizontal, 6).padding(.vertical, 2.5)
            .background(Capsule().fill(c.opacity(0.13)))
            .fixedSize()
    }

    private func bigNumber(_ v: Double?, _ label: String, _ color: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(Fmt.px(v)).font(.system(size: 28, weight: .semibold, design: .rounded).monospacedDigit())
                    .foregroundStyle(color)
                Text("px").font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.text3)
            }
            Text(label).font(.system(size: 11)).foregroundStyle(Theme.text3).lineLimit(1)
        }
        .fixedSize()
    }

    private func row(_ k: String, _ v: String, warn: Bool = false) -> some View {
        HStack {
            Text(k).font(.system(size: 11.5)).foregroundStyle(Theme.text3)
            Spacer()
            Text(v).font(.system(size: 11.5).monospacedDigit()).foregroundStyle(warn ? Theme.warn : Theme.text2).lineLimit(1)
        }
    }
}

/// "Timing auto-calibration": what the engine's per-clip timing fit did (summary['timecal']).
struct TimingReadout: View {
    let t: TimecalSummary
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon).font(.system(size: 11.5, weight: .semibold)).foregroundStyle(color)
                .frame(width: 14).padding(.top, 1)
            VStack(alignment: .leading, spacing: 2) {
                Text("Timing auto-calibration").font(.system(size: 11.5)).foregroundStyle(Theme.text3)
                (Text(value.capitalizedFirst).foregroundColor(Theme.text2)
                 + Text(detail.map { "  " + $0.replacingOccurrences(of: " ", with: "\u{00A0}") } ?? "").foregroundColor(Theme.text3))
                    .font(.system(size: 11.5).monospacedDigit())
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .help(help)
    }
    private var icon: String {
        switch t.state {
        case "applied": return "clock.arrow.circlepath"
        case "confirmed": return "checkmark.seal"
        case "failed": return "exclamationmark.triangle"
        default: return "minus.circle"
        }
    }
    private var color: Color {
        switch t.state {
        case "applied", "confirmed": return Theme.good
        case "failed": return Theme.warn
        default: return Theme.text3
        }
    }
    static func ms(_ v: Double) -> String {
        let a = abs(v)
        let s = a < 0.1 ? String(format: "%.3f", a) : String(format: "%.2f", a)
        return (v < 0 ? "−" : "+") + s
    }
    private var sigma: String { t.sigmaMs.map { String(format: " ± %.3f", $0) } ?? "" }
    /// The short result (right of the label).
    var value: String {
        switch t.state {
        case "applied":
            if let o = t.offsetMs, o != 0 { return "\(Self.ms(o)) ms applied" }
            return "correction applied"
        case "confirmed": return "metadata timing confirmed"
        case "kept": return "metadata timing kept"
        case "skipped": return "not run"
        case "failed": return "failed, metadata kept"
        case "off": return "off"
        default: return t.state
        }
    }
    /// The numbers behind it (second line).
    var detail: String? {
        let fit = t.estimateMs.map { "fit \(Self.ms($0))\(sigma) ms" }
        switch t.state {
        case "applied":
            var parts: [String] = []
            if let o = t.offsetMs, o != 0 { parts.append("offset \(Self.ms(o))\(sigma) ms") }
            if let r = t.readoutPct { parts.append(String(format: "readout %+.1f%%", r)) }
            if let f = t.focalPct { parts.append(String(format: "focal %+.2f%%", f)) }
            if let b = t.boxPct { parts.append(String(format: "exposure %+.0f%%", b)) }
            return parts.isEmpty ? nil : parts.joined(separator: " · ")
        case "confirmed": return fit
        case "kept": return fit.map { $0 + ", not confirmed on the clip" } ?? t.detail   // else the engine's reason
        case "skipped": return t.detail
        default: return nil
        }
    }
    private var help: String {
        var h = "Stillpoint fits the gyro-to-picture timing (offset, readout) on a few windows of the clip and applies it only when it is confident and confirmed on held-out windows; otherwise the camera's own timing metadata is used."
        if let d = t.detail, !d.isEmpty { h += "\n\nEngine: " + d }
        return h
    }
}

// MARK: - Controls

struct StabilizationSection: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var clip: Clip

    var body: some View {
        let enabled = clip.probe?.supported == true && !clip.isBusy
        VStack(alignment: .leading, spacing: 14) {
            SectionLabel(text: "Stabilization")
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Smoothness").font(.system(size: 12.5, weight: .medium))
                    Spacer()
                    Text(smoothLabel(clip.smoothness)).font(.system(size: 12)).foregroundStyle(Theme.text2)
                    Text(String(format: "%.2f", clip.smoothness)).font(.spMono(11)).foregroundStyle(Theme.text3)
                        .frame(width: 30, alignment: .trailing)
                }
                SPSlider(value: $clip.smoothness, range: 0...2, step: 0.05, enabled: enabled,
                         marker: clip.manifest.map { $0.params.smoothness })
                legend("Responsive", "Floaty")
            }
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Field of view").font(.system(size: 12.5, weight: .medium))
                    Spacer()
                    Text(String(format: "%.0f°", clip.fovDeg)).font(.spMono(12, .medium)).foregroundStyle(Theme.text2)
                }
                SPSlider(value: $clip.fovDeg, range: clip.fovRange, step: 0.5, enabled: enabled,
                         marker: clip.manifest?.params.fovDeg)
                legend("Tighter · steadier", "Wider")
            }
            OptionsGroup(clip: clip, enabled: enabled)
            if clip.isAnalyzed {
                if clip.settingsChanged {
                    HStack(alignment: .center, spacing: 10) {
                        Image(systemName: "arrow.triangle.2.circlepath").foregroundStyle(Theme.warn)
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Settings changed").font(.system(size: 12, weight: .medium))
                            Text(clip.changedSettings.joined(separator: " · ")).font(.system(size: 10.5))
                                .foregroundStyle(Theme.text3).lineLimit(2).fixedSize(horizontal: false, vertical: true)
                        }
                        Spacer(minLength: 4)
                        Button("Re-analyze") { model.analyze(clip) }
                            .buttonStyle(PrimaryButtonStyle(tint: Theme.accent, enabled: clip.canAnalyze)).frame(width: 104)
                            .disabled(!clip.canAnalyze)
                    }
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: 9, style: .continuous).fill(Theme.warn.opacity(0.09)))
                    .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(Theme.warn.opacity(0.25)))
                } else {
                    Button { model.analyze(clip) } label: {
                        HStack(spacing: 6) { Image(systemName: "arrow.clockwise"); Text("Re-analyze") }
                    }
                    .buttonStyle(SecondaryButtonStyle(fill: true))
                    .disabled(!clip.canAnalyze)
                }
                if !clip.isBusy, let pf = model.preflight(clip) { PreflightView(pf: pf, compact: true) }
            }
        }
    }

    private func legend(_ a: String, _ b: String) -> some View {
        HStack {
            Text(a)
            Spacer()
            Text(b)
        }
        .font(.system(size: 10.5)).foregroundStyle(Theme.text3)
    }

    private func smoothLabel(_ s: Double) -> String {
        switch s {
        case ..<0.35: return "Responsive"
        case ..<0.75: return "Natural"
        case ..<1.3: return "Cinematic"
        case ..<1.7: return "Smooth"
        default: return "Floaty"
        }
    }
}

/// Engine v5 options: Horizon lock (strength, bank limit), Full-frame fill, Max quality. What each clip starts with,
/// what is supported and what cannot be combined all come from the engine (probe's options); options the engine
/// cannot combine disable each other with a hint instead of failing the analysis.
struct OptionsGroup: View {
    @ObservedObject var clip: Clip
    let enabled: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            horizon
            divider
            fill
            divider
            maxQuality
        }
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.black.opacity(0.22)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Theme.hairline))
    }

    private var divider: some View { Rectangle().fill(Theme.hairline).frame(height: 1).padding(.leading, 40) }

    // MARK: horizon lock

    private var horizon: some View {
        let supported = clip.horizonSupported && clip.probe != nil
        let on = clip.horizonLock && supported
        return VStack(alignment: .leading, spacing: 10) {
            OptionRow(icon: "level", title: "Horizon lock", badge: supported ? "BETA" : nil,
                      subtitle: supported ? "Keeps the horizon level through rolls"
                                          : "Needs the camera's gravity data — not in this clip",
                      isOn: $clip.horizonLock, enabled: supported && enabled)
            if on {
                VStack(alignment: .leading, spacing: 10) {
                    slider("Strength", value: "\(Int((clip.horizonStrength * 100).rounded()))%",
                           binding: $clip.horizonStrength, range: clip.options?.strengthRange ?? 0.1...1, step: 0.05,
                           marker: markerStrength, legend: ("Gentle", "Full"))
                    slider("Bank limit", value: bankText, binding: $clip.rollLimitDeg,
                           range: clip.options?.rollLimitRange ?? 0...45, step: 1, marker: markerRoll,
                           legend: ("Fully level", "Keeps banks to \(Int((clip.options?.rollLimitRange ?? 0...45).upperBound))°"))
                }
                .padding(.leading, 40).padding(.trailing, 12)
            }
            if on || clip.horizonLock, let n = clip.note("horizon") {
                NoteRow(item: .init(level: .warn, icon: "exclamationmark.triangle", text: n))
                    .padding(.leading, 40).padding(.trailing, 10)
            }
        }
        .padding(.bottom, on || (clip.horizonLock && clip.note("horizon") != nil) ? 12 : 0)
        .help(supported ? "Levels the horizon using the camera's gravity estimate, never zooming in beyond the unlocked path. It fades out when the camera is steep, inverted or flipping."
                        : "This clip has no gravity data, so the horizon cannot be levelled.")
    }

    private var bankText: String {
        let d = Int(clip.rollLimitDeg.rounded())
        return d == 0 ? "0° · fully level" : "±\(d)° kept"
    }
    private var markerStrength: Double? {
        guard let p = clip.manifest?.params, p.horizonLock else { return nil }
        return p.horizonStrength ?? 1
    }
    private var markerRoll: Double? {
        guard let p = clip.manifest?.params, p.horizonLock else { return nil }
        return p.rollLimitDeg ?? 0
    }

    private func slider(_ title: String, value: String, binding: Binding<Double>, range: ClosedRange<Double>,
                        step: Double, marker: Double?, legend: (String, String)) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(title).font(.system(size: 11.5, weight: .medium)).foregroundStyle(Theme.text2)
                Spacer()
                Text(value).font(.spMono(11.5, .medium)).foregroundStyle(Theme.text2)
            }
            SPSlider(value: binding, range: range, step: step, enabled: enabled, marker: marker)
            HStack { Text(legend.0); Spacer(); Text(legend.1) }
                .font(.system(size: 10)).foregroundStyle(Theme.text3)
        }
    }

    // MARK: fill / max quality

    private var fill: some View {
        let blocked = clip.fillBlockedBy
        return VStack(alignment: .leading, spacing: 8) {
            OptionRow(icon: "rectangle.dashed", title: "Full-frame fill", badge: nil,
                      subtitle: clip.fillSupported ? "Fills the corners from neighbouring frames, so Stillpoint can keep more of the frame"
                                                   : "Not available with this engine",
                      isOn: $clip.fill, enabled: enabled && clip.fillSupported && blocked == nil)
            if let b = blocked, !clip.fill {
                hint("Not with \(b) yet — turn it off to use fill.")
            }
        }
        .padding(.bottom, blocked != nil && !clip.fill ? 10 : 0)
    }

    private var maxQuality: some View {
        let blocked = clip.maxQualityBlockedBy
        let factor = (clip.options?.timeFactor?["mesh"]) ?? 1.4        // per camera: ~x1.4 on O3, ~x6.5 on OA4 / O4 Pro
        let cost = factor >= 2 ? "about \(String(format: "%.0f", factor))× as long"
                               : "about +\(Int(((factor - 1) * 100).rounded()))%"
        return VStack(alignment: .leading, spacing: 8) {
            OptionRow(icon: "dial.high", title: "Max quality", badge: nil,
                      subtitle: clip.meshSupported ? "Removes extra micro-jitter; analysis takes longer (\(cost))"
                                                   : "Not available with this engine",
                      isOn: $clip.maxQuality, enabled: enabled && clip.meshSupported && blocked == nil)
            if let b = blocked, !clip.maxQuality {
                hint("Not with \(b) yet — turn it off to use Max quality.")
            }
        }
        .padding(.bottom, blocked != nil && !clip.maxQuality ? 10 : 0)
    }

    private func hint(_ t: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "info.circle").font(.system(size: 10.5, weight: .medium)).padding(.top, 1)
            Text(t).font(.system(size: 10.5)).fixedSize(horizontal: false, vertical: true)
        }
        .foregroundStyle(Theme.text3)
        .padding(.leading, 40).padding(.trailing, 12)
    }
}

/// One switch row of the options group: icon, title (+ badge), one-line explanation, toggle.
struct OptionRow: View {
    let icon: String
    let title: String
    let badge: String?
    let subtitle: String
    @Binding var isOn: Bool
    let enabled: Bool

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: icon).font(.system(size: 12.5, weight: .medium))
                .foregroundStyle(isOn && enabled ? Theme.accent : Theme.text3)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(title).font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(enabled || isOn ? Theme.text : Theme.text2)
                    if let badge {
                        Text(badge).font(.system(size: 8.5, weight: .bold)).tracking(0.6).foregroundStyle(Theme.warn)
                            .padding(.horizontal, 4).padding(.vertical, 1.5)
                            .background(RoundedRectangle(cornerRadius: 3).fill(Theme.warn.opacity(0.14)))
                    }
                }
                Text(subtitle).font(.system(size: 10.5)).foregroundStyle(Theme.text3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 6)
            SPToggle(isOn: $isOn, enabled: enabled)
        }
        .padding(.horizontal, 12).padding(.vertical, 11)
    }
}

// MARK: - Export

struct ExportSection: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var clip: Clip

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                SectionLabel(text: "Export")
                let n = model.analyzedClips.count
                if n > 1 {
                    Button("Export all \(n)") { model.export(model.analyzedClips) }
                        .buttonStyle(.plain).font(.system(size: 11, weight: .medium)).foregroundStyle(Theme.accent)
                }
            }
            SPSegmented(options: ExportPreset.allCases.map { ($0, $0.short) }, selection: $model.preset, height: 28)
            Text(model.preset.detail).font(.system(size: 11)).foregroundStyle(Theme.text3)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 8) {
                Image(systemName: "folder.fill").font(.system(size: 12)).foregroundStyle(Theme.accent.opacity(0.9))
                Text(abbreviated(model.outputFolder.path)).font(.system(size: 12)).foregroundStyle(Theme.text2)
                    .lineLimit(1).truncationMode(.middle)
                Spacer(minLength: 4)
                Button("Change…") { model.chooseOutputFolder() }.buttonStyle(SecondaryButtonStyle())
            }
            .padding(.leading, 10).padding(.trailing, 4).padding(.vertical, 4)
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Color.black.opacity(0.25)))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(Theme.hairline))
            Button { model.export([clip]) } label: {
                HStack(spacing: 6) { Image(systemName: "square.and.arrow.up"); Text("Export Clip") }
            }
            .buttonStyle(PrimaryButtonStyle(enabled: clip.isAnalyzed))
            .disabled(!clip.isAnalyzed)
            if !clip.isAnalyzed {
                Text("Analyze the clip first.").font(.system(size: 11)).foregroundStyle(Theme.text3)
            }
        }
    }

    private func abbreviated(_ p: String) -> String {
        let home = NSHomeDirectory()
        return p.hasPrefix(home) ? "~" + p.dropFirst(home.count) : p
    }
}

