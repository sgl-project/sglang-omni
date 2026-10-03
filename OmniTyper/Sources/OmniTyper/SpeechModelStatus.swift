// SPDX-License-Identifier: Apache-2.0
import SwiftUI

enum SpeechModelState: Equatable {
    case missing, downloaded, ready

    var symbol: String {
        switch self {
        case .missing: "arrow.down.circle"
        case .downloaded: "internaldrive"
        case .ready: "checkmark.circle.fill"
        }
    }

    var summary: String {
        switch self {
        case .missing: L("settings.speechModel.missing")
        case .downloaded: L("settings.speechModel.downloaded")
        case .ready: L("settings.speechModel.ready")
        }
    }

    static func resolve(ready: Bool, repository: String,
                        environment: [String: String] = ProcessInfo.processInfo.environment,
                        home: URL = FileManager.default.homeDirectoryForCurrentUser) -> SpeechModelState {
        if ready { return .ready }
        return isDownloaded(repository, environment: environment, home: home) ? .downloaded : .missing
    }

    // Note (Yifei Leng): Resolve the cache the way huggingface_hub does, so the state is known before the worker starts.
    static func hubCache(environment: [String: String], home: URL) -> URL {
        func directory(_ key: String) -> URL? {
            guard let value = environment[key], !value.isEmpty else { return nil }
            return URL(fileURLWithPath: (value as NSString).expandingTildeInPath)
        }
        if let cache = directory("HF_HUB_CACHE") { return cache }
        if let hub = directory("HF_HOME") { return hub.appendingPathComponent("hub") }
        return (directory("XDG_CACHE_HOME") ?? home.appendingPathComponent(".cache"))
            .appendingPathComponent("huggingface/hub")
    }

    // Note (Yifei Leng): fileExists follows the snapshot's symlink, so interrupted weights still count as missing.
    static func isDownloaded(_ repository: String, environment: [String: String], home: URL) -> Bool {
        let snapshots = hubCache(environment: environment, home: home)
            .appendingPathComponent("models--" + repository.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots")
        let revisions = (try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil)) ?? []
        return revisions.contains {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.safetensors").path)
        }
    }
}

struct SpeechModelStatusRow: View {
    @ObservedObject var worker: WorkerClient
    let repository: String

    var body: some View {
        // Note (Yifei Leng): While preparing, the card's PreparationCard already shows the live status and bar.
        if !worker.isPreparingSpeech {
            let state = SpeechModelState.resolve(ready: worker.speechModelReady, repository: repository)
            Label(state.summary, systemImage: state.symbol).font(.caption)
                .foregroundStyle(state == .ready ? AnyShapeStyle(.green) : AnyShapeStyle(.secondary))
        }
    }
}
