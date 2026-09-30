// Stabilised playback: AVPlayer + the Metal compositor (Warp.swift), scrubbing, compare modes.
import AVFoundation
import AppKit
import SwiftUI

@MainActor
final class PlayerController: ObservableObject {
    let player = AVPlayer()
    let state = PreviewState()
    @Published private(set) var url: URL?
    @Published var isPlaying = false
    @Published var time: Double = 0
    @Published private(set) var duration: Double = 0
    @Published private(set) var fps: Double = 59.94
    @Published private(set) var hasPlan = false
    @Published private(set) var renderSize = CGSize(width: 3840, height: 2160)
    @Published var error: String?
    @Published var loop = true
    @Published var settings = WarpSettings() {
        didSet {
            applySettings()
            if !isPlaying { scheduleRedraw() }
        }
    }
    /// The loaded plan has a full-frame fill section / a mesh block (engine v5).
    @Published private(set) var planHasFill = false
    @Published private(set) var planHasMesh = false

    private var item: AVPlayerItem?
    private var info: TrackInfo?
    private var timeObserver: Any?
    private var endObserver: NSObjectProtocol?
    private var seeking = false
    private var pendingSeek: (Double, Bool)?
    private var redrawScheduled = false
    private var loadToken = 0

    init() {
        state.fill.onReady = { [weak self] in
            MainActor.assumeIsolated {       // fill sources arrived for the paused frame: show it exact
                guard let self, !self.isPlaying else { return }
                self.scheduleRedraw()
            }
        }
        player.isMuted = false
        player.automaticallyWaitsToMinimizeStalling = false
        player.actionAtItemEnd = .pause
        timeObserver = player.addPeriodicTimeObserver(forInterval: CMTime(value: 1, timescale: 30), queue: .main) { [weak self] t in
            MainActor.assumeIsolated {
                guard let self, !self.seeking else { return }
                self.time = t.seconds
            }
        }
    }

    func load(url: URL, plan: PlanFile?) {
        if self.url == url, item != nil { setPlan(plan); return }
        loadToken += 1
        let token = loadToken
        pause()
        if self.url != url { state.fill.clear(); state.track = nil }
        self.url = url
        state.plan = plan
        notePlan(plan)
        time = 0
        duration = 0
        error = nil
        Task { @MainActor in
            do {
                let asset = AVURLAsset(url: url)
                let info = try await TrackInfo.load(asset)
                guard token == self.loadToken else { return }
                self.info = info
                self.state.track = info.track
                self.fps = info.fps > 0 ? info.fps : 59.94
                self.duration = info.duration.seconds
                let it = AVPlayerItem(asset: asset)
                it.videoComposition = info.makeComposition(state: state, renderSize: renderSize(for: plan, info))
                self.renderSize = renderSize(for: plan, info)
                if let o = endObserver { NotificationCenter.default.removeObserver(o) }
                endObserver = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: it,
                                                                     queue: .main) { [weak self] _ in
                    MainActor.assumeIsolated {
                        guard let self else { return }
                        if self.loop && self.isPlaying {
                            self.player.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
                            self.player.play()
                        } else {
                            self.isPlaying = false
                        }
                    }
                }
                self.item = it
                player.replaceCurrentItem(with: it)
                await player.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
            } catch {
                self.error = error.localizedDescription
            }
        }
    }

    private func renderSize(for plan: PlanFile?, _ info: TrackInfo) -> CGSize {
        if let p = plan, p.srcW == Int(info.size.width), p.srcH == Int(info.size.height) {
            return CGSize(width: p.outW, height: p.outH)
        }
        return info.size
    }

    private func notePlan(_ plan: PlanFile?) {
        hasPlan = plan != nil
        planHasFill = plan?.hasFill ?? false
        planHasMesh = plan?.hasMesh ?? false
    }

    /// What the compositor uses: paused frames = export filter + exact fill (neighbours fetched in the background,
    /// then redrawn); playing = Catmull-Rom + whatever fill sources are cached (the rest: the kernel's soft edge).
    private func applySettings() {
        var s = settings
        if isPlaying { s.kernel = 1; s.fill = .cachedOnly } else { s.fill = .progressive }
        state.settings = s
    }

    func setPlan(_ plan: PlanFile?) {
        state.plan = plan
        notePlan(plan)
        guard let info, let item else { return }
        let size = renderSize(for: plan, info)
        renderSize = size
        item.videoComposition = info.makeComposition(state: state, renderSize: size)
    }

    func unload() {
        pause()
        player.replaceCurrentItem(with: nil)
        item = nil
        url = nil
        state.plan = nil
        state.track = nil
        state.fill.clear()
        notePlan(nil)
    }

    /// Snapshot mode: show a timeline position without an AVPlayerItem.
    func configureForSnapshot(url: URL, duration: Double, time: Double, fps: Double, plan: PlanFile?, renderSize: CGSize) {
        self.url = url
        self.duration = duration
        self.time = time
        self.fps = fps
        state.plan = plan
        notePlan(plan)
        self.renderSize = renderSize
    }

    // MARK: transport

    func togglePlay() { isPlaying ? pause() : play() }

    func play() {
        guard item != nil else { return }
        if duration > 0, time >= duration - 0.05 { seek(to: 0) }
        isPlaying = true
        applySettings()              // Catmull-Rom while playing (cheaper 4x4 taps, same geometry), cached fill only
        player.play()
    }

    func pause() {
        player.pause()
        let wasPlaying = isPlaying
        isPlaying = false
        applySettings()              // paused frames use the export filter (Lanczos-3) and exact fill: what export writes
        if wasPlaying { scheduleRedraw() }
    }

    /// Frame-exact seek with chasing (only the latest target is kept while one is in flight).
    func seek(to t: Double, exact: Bool = true) {
        let target = min(max(0, t), max(duration - 0.001, 0))
        time = target
        if seeking { pendingSeek = (target, exact); return }
        seeking = true
        let tol: CMTime = exact ? .zero : CMTime(value: 1, timescale: 10)
        player.seek(to: CMTime(seconds: target, preferredTimescale: 60000), toleranceBefore: tol, toleranceAfter: tol) { [weak self] _ in
            DispatchQueue.main.async {
                guard let self else { return }
                self.seeking = false
                if let (next, ex) = self.pendingSeek {
                    self.pendingSeek = nil
                    self.seek(to: next, exact: ex)
                }
            }
        }
    }

    func step(_ frames: Int) {
        pause()
        guard fps > 0 else { return }
        let fd = 1.0 / fps
        let k = (time / fd).rounded() + Double(frames)
        seek(to: k * fd + 0.1 * fd)
    }

    /// Re-render the paused frame after a settings change (new composition object => AVFoundation recomposes).
    func scheduleRedraw() {
        guard !redrawScheduled else { return }
        redrawScheduled = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0 / 60.0) { [weak self] in
            guard let self else { return }
            self.redrawScheduled = false
            guard let item = self.item, let c = item.videoComposition?.mutableCopy() as? AVMutableVideoComposition else { return }
            item.videoComposition = c
        }
    }
}

/// AVPlayerLayer host.
struct PlayerLayerView: NSViewRepresentable {
    let player: AVPlayer
    func makeNSView(context: Context) -> PlayerNSView { PlayerNSView(player: player) }
    func updateNSView(_ v: PlayerNSView, context: Context) {
        if v.playerLayer.player !== player { v.playerLayer.player = player }
    }
    final class PlayerNSView: NSView {
        let playerLayer = AVPlayerLayer()
        init(player: AVPlayer) {
            super.init(frame: .zero)
            wantsLayer = true
            layer = CALayer()
            layer?.backgroundColor = NSColor.black.cgColor
            playerLayer.player = player
            playerLayer.videoGravity = .resizeAspect
            playerLayer.backgroundColor = NSColor.black.cgColor
            layer?.addSublayer(playerLayer)
        }
        required init?(coder: NSCoder) { fatalError() }
        override func layout() {
            super.layout()
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            playerLayer.frame = bounds
            CATransaction.commit()
        }
    }
}
