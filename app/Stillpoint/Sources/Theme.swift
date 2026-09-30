// Theme + custom controls. Everything here is drawn in pure SwiftUI (no AppKit-backed controls), so the app looks the
// same in the window and in `--snapshot` renders (ImageRenderer cannot draw NSView-backed controls).
import SwiftUI

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        self.init(.sRGB, red: Double((hex >> 16) & 255) / 255, green: Double((hex >> 8) & 255) / 255,
                  blue: Double(hex & 255) / 255, opacity: alpha)
    }
}

enum Theme {
    static let window = Color(hex: 0x09090B)
    static let panel = Color(hex: 0x111114)
    static let raised = Color(hex: 0x18181C)
    static let raised2 = Color(hex: 0x222228)
    static let control = Color(hex: 0x2A2A31)
    static let hairline = Color.white.opacity(0.075)
    static let hairlineStrong = Color.white.opacity(0.13)
    static let text = Color(hex: 0xF4F4F6)
    static let text2 = Color(hex: 0xA3A3AC)
    static let text3 = Color(hex: 0x6C6C76)
    static let accent = Color(hex: 0x4DA3FF)
    static let accentSoft = Color(hex: 0x4DA3FF, alpha: 0.16)
    static let good = Color(hex: 0x3DD68C)
    static let warn = Color(hex: 0xFFB547)
    static let bad = Color(hex: 0xFF6A5C)
    static let shake = Color(hex: 0xFF8A5B)

    static func level(_ level: String) -> Color {
        switch level {
        case "good": return good
        case "limited": return warn
        default: return bad
        }
    }
}

extension Font {
    static let spTitle = Font.system(size: 15, weight: .semibold)
    static let spBody = Font.system(size: 12.5)
    static let spSmall = Font.system(size: 11)
    static let spCaps = Font.system(size: 10, weight: .semibold)
    static func spMono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .rounded).monospacedDigit()
    }
}

/// Small uppercase section label.
struct SectionLabel: View {
    let text: String
    var trailing: String? = nil
    var body: some View {
        HStack {
            Text(text.uppercased()).font(.spCaps).tracking(0.9).foregroundStyle(Theme.text3)
            Spacer()
            if let trailing { Text(trailing).font(.spSmall).foregroundStyle(Theme.text3) }
        }
    }
}

struct Card<Content: View>: View {
    var padding: CGFloat = 14
    @ViewBuilder var content: Content
    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.raised))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Theme.hairline))
    }
}

/// Gyro status pill (green / amber / red dot + label).
struct StatusBadge: View {
    let label: String
    let level: String
    var compact = false
    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(Theme.level(level)).frame(width: 6, height: 6)
            Text(label).font(.system(size: compact ? 10.5 : 11, weight: .medium)).lineLimit(1)
                .foregroundStyle(level == "good" ? Theme.text2 : Theme.level(level))
        }
        .padding(.horizontal, 7).padding(.vertical, 3)
        .background(Capsule().fill(Theme.level(level).opacity(level == "good" ? 0.10 : 0.13)))
    }
}

struct Chip: View {
    let text: String
    var body: some View {
        Text(text).font(.system(size: 10.5, weight: .medium)).foregroundStyle(Theme.text2)
            .lineLimit(1).fixedSize()
            .padding(.horizontal, 6).padding(.vertical, 2.5)
            .background(RoundedRectangle(cornerRadius: 4, style: .continuous).fill(Color.white.opacity(0.06)))
    }
}

// MARK: - Buttons

struct PrimaryButtonStyle: ButtonStyle {
    var tint: Color = Theme.accent
    var enabled = true
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12.5, weight: .semibold))
            .foregroundStyle(enabled ? Color.white : Theme.text3)
            .padding(.horizontal, 14).frame(height: 30)
            .frame(maxWidth: .infinity)
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(enabled ? tint.opacity(configuration.isPressed ? 0.75 : 1) : Theme.control))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Color.white.opacity(enabled ? 0.18 : 0.04), lineWidth: 0.5))
            .contentShape(Rectangle())
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    var fill = false
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(Theme.text)
            .padding(.horizontal, 12).frame(height: 28)
            .frame(maxWidth: fill ? .infinity : nil)
            .background(RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(Theme.control.opacity(configuration.isPressed ? 0.6 : 1)))
            .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous).strokeBorder(Theme.hairline))
            .contentShape(Rectangle())
    }
}

struct IconButtonStyle: ButtonStyle {
    var size: CGFloat = 28
    var active = false
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(active ? Theme.accent : Theme.text2)
            .frame(width: size, height: size)
            .background(Circle().fill(Color.white.opacity(configuration.isPressed ? 0.12 : (active ? 0.08 : 0))))
            .contentShape(Circle())
    }
}

// MARK: - Slider

struct SPSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    var step: Double = 0
    var enabled = true
    var marker: Double? = nil                 // e.g. the analysed value
    var onCommit: () -> Void = {}
    @State private var dragging = false

    var body: some View {
        GeometryReader { g in
            let w = max(g.size.width - 16, 1)
            let f = CGFloat((value - range.lowerBound) / (range.upperBound - range.lowerBound)).clamped(0, 1)
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.09)).frame(height: 4).padding(.horizontal, 8)
                Capsule().fill(enabled ? Theme.accent : Theme.text3).frame(width: 8 + w * f, height: 4)
                    .padding(.leading, 0)
                if let m = marker {
                    let mf = CGFloat((m - range.lowerBound) / (range.upperBound - range.lowerBound)).clamped(0, 1)
                    RoundedRectangle(cornerRadius: 1).fill(Theme.text2.opacity(0.7))
                        .frame(width: 2, height: 10).offset(x: 8 + w * mf - 1)
                }
                Circle().fill(enabled ? Color.white : Theme.text3)
                    .frame(width: dragging ? 16 : 14, height: dragging ? 16 : 14)
                    .shadow(color: .black.opacity(0.45), radius: 3, y: 1)
                    .offset(x: 8 + w * f - (dragging ? 8 : 7))
            }
            .frame(height: g.size.height)
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { d in
                    guard enabled else { return }
                    dragging = true
                    let x = Double(((d.location.x - 8) / w).clamped(0, 1))
                    var v = range.lowerBound + x * (range.upperBound - range.lowerBound)
                    if step > 0 { v = (v / step).rounded() * step }
                    value = min(max(v, range.lowerBound), range.upperBound)
                }
                .onEnded { _ in dragging = false; if enabled { onCommit() } })
        }
        .frame(height: 20)
        .opacity(enabled ? 1 : 0.5)
    }
}

// MARK: - Segmented

struct SPSegmented<T: Hashable>: View {
    let options: [(T, String)]
    @Binding var selection: T
    var height: CGFloat = 26
    var body: some View {
        HStack(spacing: 2) {
            ForEach(options.indices, id: \.self) { i in
                let (v, label) = options[i]
                let on = v == selection
                Button { selection = v } label: {
                    Text(label).font(.system(size: 11.5, weight: on ? .semibold : .medium))
                        .foregroundStyle(on ? Theme.text : Theme.text2)
                        .frame(maxWidth: .infinity).frame(height: height - 4)
                        .background(RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(on ? Theme.raised2 : Color.clear)
                            .shadow(color: .black.opacity(on ? 0.35 : 0), radius: 2, y: 1))
                        .contentShape(Rectangle())
                }.buttonStyle(.plain)
            }
        }
        .padding(2)
        .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Color.black.opacity(0.35)))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(Theme.hairline))
    }
}

// MARK: - Toggle

struct SPToggle: View {
    @Binding var isOn: Bool
    var enabled = true
    var body: some View {
        Button { if enabled { isOn.toggle() } } label: {
            ZStack(alignment: isOn ? .trailing : .leading) {
                Capsule().fill(isOn && enabled ? Theme.accent : Color.white.opacity(0.12)).frame(width: 30, height: 18)
                Circle().fill(enabled ? Color.white : Theme.text3).frame(width: 14, height: 14).padding(2)
                    .shadow(color: .black.opacity(0.3), radius: 1, y: 0.5)
            }
        }
        .buttonStyle(.plain)
        .opacity(enabled ? 1 : 0.45)
    }
}

// MARK: - Progress bar

struct SPProgressBar: View {
    let fraction: Double
    var tint: Color = Theme.accent
    var height: CGFloat = 4
    var body: some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.08))
                Capsule().fill(tint).frame(width: max(height, g.size.width * CGFloat(fraction.clamped(0, 1))))
            }
        }.frame(height: height)
    }
}

/// Circular progress used on thumbnails.
struct SPRing: View {
    let fraction: Double
    var size: CGFloat = 18
    var body: some View {
        ZStack {
            Circle().stroke(Color.white.opacity(0.18), lineWidth: 2.2)
            Circle().trim(from: 0, to: CGFloat(fraction.clamped(0.02, 1)))
                .stroke(Theme.accent, style: StrokeStyle(lineWidth: 2.2, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }.frame(width: size, height: size)
    }
}

extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}

extension Comparable {
    func clamped(_ lo: Self, _ hi: Self) -> Self { min(max(self, lo), hi) }
}

// MARK: - Snapshot environment

private struct SnapshotModeKey: EnvironmentKey { static let defaultValue = false }
extension EnvironmentValues {
    /// True while rendering with ImageRenderer: views avoid AppKit-backed containers (ScrollView, player layer).
    var snapshotMode: Bool {
        get { self[SnapshotModeKey.self] }
        set { self[SnapshotModeKey.self] = newValue }
    }
}

/// ScrollView in the app, plain VStack in snapshots.
struct AdaptiveScroll<Content: View>: View {
    @Environment(\.snapshotMode) private var snapshot
    @ViewBuilder var content: Content
    var body: some View {
        if snapshot {       // top-aligned; overflow is cut at the bottom like a scroll view at its top
            Color.clear
                .overlay(alignment: .top) { VStack(spacing: 0) { content }.fixedSize(horizontal: false, vertical: true) }
                .clipped()
        } else {
            ScrollView(.vertical, showsIndicators: false) { content }
        }
    }
}

// MARK: - Formatting

enum Fmt {
    static func duration(_ s: Double) -> String {
        guard s.isFinite, s >= 0 else { return "–" }
        let t = Int(s.rounded())
        return t >= 3600 ? String(format: "%d:%02d:%02d", t / 3600, (t / 60) % 60, t % 60)
                         : String(format: "%d:%02d", t / 60, t % 60)
    }
    /// mm:ss:ff (frames) timecode.
    static func timecode(_ s: Double, fps: Double) -> String {
        guard s.isFinite, fps > 0 else { return "00:00:00" }
        let fr = Int((s * fps).rounded(.down))
        let f = Int(fps.rounded())
        let secs = fr / max(f, 1)
        return String(format: "%02d:%02d:%02d", secs / 60, secs % 60, fr % max(f, 1))
    }
    /// mm:ss.cc
    static func clock(_ s: Double) -> String {
        guard s.isFinite, s >= 0 else { return "00:00.00" }
        let cs = Int((s * 100).rounded(.down))
        return String(format: "%02d:%02d.%02d", cs / 6000, (cs / 100) % 60, cs % 100)
    }
    static func eta(_ s: Double?) -> String {
        guard let s, s.isFinite, s >= 0 else { return "estimating…" }
        if s < 5 { return "a few seconds left" }
        if s < 60 { return "about \(Int((s / 5).rounded() * 5)) s left" }
        let m = Int((s / 60).rounded())
        return m < 60 ? "about \(m) min left" : String(format: "about %d h %02d min left", m / 60, m % 60)
    }
    /// A pixel value exactly as measured (no floors or clamping): 3 decimals below 0.1, 2 below 10.
    static func px(_ v: Double?) -> String {
        guard let v, v.isFinite else { return "–" }
        if v < 0.1 { return String(format: "%.3f", v) }
        return v < 10 ? String(format: "%.2f", v) : String(format: "%.1f", v)
    }
    /// "a → b"; when two different measurements would print the same, both get 3 decimals.
    static func pair(_ a: Double?, _ b: Double?) -> String {
        var (fa, fb) = (px(a), px(b))
        if fa == fb, let a, let b, a != b, a.isFinite, b.isFinite {
            fa = String(format: "%.3f", a)
            fb = String(format: "%.3f", b)
        }
        return "\(fa) → \(fb)"
    }
    static func gb(_ b: Int64) -> String {
        let g = Double(b) / 1e9
        return g < 10 ? String(format: "%.1f GB", g) : String(format: "%.0f GB", g)
    }
    /// "about 40 s", "about 12 min", "about 1 h 05 min".
    static func approx(_ s: Double) -> String {
        guard s.isFinite, s >= 0 else { return "–" }
        if s < 90 { return "about \(max(10, Int((s / 10).rounded()) * 10)) s" }
        let m = Int((s / 60).rounded())
        return m < 60 ? "about \(m) min" : String(format: "about %d h %02d min", m / 60, m % 60)
    }
    static func bytes(_ b: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: b, countStyle: .file)
    }
    static func resolution(_ w: Int, _ h: Int) -> String {
        if w >= 3800 { return h >= 2800 ? "4K 4:3" : "4K" }
        if w >= 2600 { return "2.7K" }
        if w >= 1900 { return "1080p" }
        return "\(w)×\(h)"
    }
    /// A fraction as a percentage: "3%", "0.4%" below 1 %, "<0.1%" for tiny non-zero values.
    static func pct(_ f: Double) -> String {
        let p = f * 100
        if p > 0 && p < 0.1 { return "<0.1%" }
        return p < 1 && p > 0 ? String(format: "%.1f%%", p) : "\(Int(p.rounded()))%"
    }
    static func fps(_ f: Double) -> String {
        abs(f - f.rounded()) < 0.01 ? String(format: "%.0f fps", f) : String(format: "%.2f fps", f)
    }
}
