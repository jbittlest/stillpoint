// Entry point: GUI app, or the headless `--selftest` / `--snapshot DIR` modes.
import AppKit
import SwiftUI

@main
enum Main {
    static func main() {
        let args = CommandLine.arguments
        if args.contains("--selftest-app") || args.contains("--selftest") || args.contains("--snapshot") || args.contains("--migrate") {
            NSApplication.shared.setActivationPolicy(.prohibited)      // headless: never a Dock icon or window
        }
        if args.contains("--migrate") {                                  // copy the old iCloud-synced cache, then exit
            let moved = AnalysisStore.migrateLegacy()
            print("migrated \(moved.count) analyses to \(EngineConfig.analyses.path): \(moved.joined(separator: ", "))")
            exit(0)
        }
        if args.contains("--selftest-app") || args.contains("--snapshot") {
            MainActor.assumeIsolated { AppModel.useHeadlessPrefs() }   // never touch the user's export settings
        }
        if args.contains("--selftest-app") {
            exit(MainActor.assumeIsolated { AppFlowTest.run(args: args) })
        }
        if args.contains("--selftest") {
            exit(SelfTest.run(args: args))
        }
        if let i = args.firstIndex(of: "--snapshot"), i + 1 < args.count {
            let dir = URL(fileURLWithPath: args[i + 1])
            let code = MainActor.assumeIsolated { Snapshot.run(dir: dir, args: args) }
            exit(code)
        }
        StillpointApp.main()
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: AppModel?
    var pendingURLs: [URL] = []

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func application(_ application: NSApplication, open urls: [URL]) {
        MainActor.assumeIsolated {
            if let model { model.add(urls) } else { pendingURLs += urls }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { model?.shutdown() }
    }
}

struct StillpointApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        Window("Stillpoint", id: "main") {
            MainView()
                .environmentObject(model)
                .preferredColorScheme(.dark)
                .frame(minWidth: 1180, minHeight: 720)
                .ignoresSafeArea()
                .onAppear {
                    delegate.model = model
                    let launchFiles = CommandLine.arguments.dropFirst().filter { !$0.hasPrefix("-") }
                        .map { URL(fileURLWithPath: $0) }
                    model.add(launchFiles + delegate.pendingURLs)
                    delegate.pendingURLs = []
                }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1480, height: 900)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Clips…") { model.openPanel() }.keyboardShortcut("o")
            }
            CommandMenu("Clip") {
                Button("Analyze") { if let c = model.selected { model.analyze(c) } }
                    .keyboardShortcut("r")
                Button("Cancel Analysis") { if let c = model.selected { model.cancelAnalysis(c) } }
                    .keyboardShortcut(".")
                Divider()
                Button("Export Clip") { if let c = model.selected { model.export([c]) } }
                    .keyboardShortcut("e")
                Button("Export All Analyzed") { model.export(model.analyzedClips) }
                    .keyboardShortcut("e", modifiers: [.command, .shift])
                Divider()
                Button("Show in Finder") { if let c = model.selected { model.reveal(c.url) } }
                Button("Remove from List") { if let c = model.selected { model.remove(c) } }
                    .keyboardShortcut(.delete)
            }
            CommandGroup(after: .toolbar) {
                Button("Stabilized") { model.player.settings.mode = .after }.keyboardShortcut("1")
                Button("Split Compare") { model.player.settings.mode = .split }.keyboardShortcut("2")
                Button("Original") { model.player.settings.mode = .before }.keyboardShortcut("3")
                Toggle("Original Shows Untouched Fisheye", isOn: Binding(
                    get: { model.player.settings.rawBefore }, set: { model.player.settings.rawBefore = $0 }))
                Divider()
            }
        }
        Settings {
            SettingsView().environmentObject(model).preferredColorScheme(.dark)
        }
    }
}

struct SettingsView: View {
    @EnvironmentObject var model: AppModel
    @State private var path = EngineConfig.rootPath

    var body: some View {
        Form {
            Section {
                HStack {
                    TextField("Engine folder", text: $path)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { model.enginePath = path }
                    Button("Choose…") {
                        let p = NSOpenPanel()
                        p.canChooseDirectories = true
                        p.canChooseFiles = false
                        p.directoryURL = URL(fileURLWithPath: path)
                        if p.runModal() == .OK, let u = p.url { path = u.path; model.enginePath = u.path }
                    }
                    Button("Default") { path = EngineConfig.defaultPath; model.enginePath = path }
                }
                if model.engineProblems.isEmpty {
                    Label("Engine, renderer and warp kernel found", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                } else {
                    ForEach(model.engineProblems, id: \.self) { p in
                        Label(p, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    }
                }
            } header: {
                Text("Stillpoint engine")
            } footer: {
                Text("The folder that contains .venv, engine/, shaders/warp.metal and app/renderer. Analyses and scratch files live in ~/Library/Application Support/Stillpoint (not synced to iCloud).")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Analysis cache") {
                Button("Show in Finder") {
                    try? FileManager.default.createDirectory(at: EngineConfig.analyses, withIntermediateDirectories: true)
                    NSWorkspace.shared.open(EngineConfig.analyses)
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 560, height: 320)
        .onChange(of: model.enginePath) { _, v in path = v }
    }
}
