// LocalAPILoopbackServeTests.swift
// Serves the Local API for the loopback end-to-end run in Voxt/Mobile/docs/evidence/loopback-e2e.

import XCTest
@testable import Voxt

@MainActor
final class LocalAPILoopbackServeTests: XCTestCase {
    /// Runs only with VOXT_E2E_SERVE_SECONDS; serves the same router, pipeline and models as the app.
    func testServeLocalAPIForTheEndToEndRun() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let seconds = environment["VOXT_E2E_SERVE_SECONDS"].flatMap(Double.init) else {
            throw XCTSkip("Set VOXT_E2E_SERVE_SECONDS to serve the Local API for an end-to-end run.")
        }
        let port = try XCTUnwrap(environment["VOXT_E2E_PORT"].flatMap(Int.init), "Set VOXT_E2E_PORT.")
        let asrModel = try XCTUnwrap(environment["VOXT_E2E_ASR_MODEL"], "Set VOXT_E2E_ASR_MODEL.")
        let textModel = try XCTUnwrap(environment["VOXT_E2E_TEXT_MODEL"], "Set VOXT_E2E_TEXT_MODEL.")
        let settings: [String: Any] = [
            AppPreferenceKey.localAPIServerEnabled: true,
            AppPreferenceKey.localAPIServerPort: port,
            AppPreferenceKey.transcriptionEngine: TranscriptionEngine.mlxAudio.rawValue,
            AppPreferenceKey.mlxModelRepo: asrModel,
            AppPreferenceKey.enhancementMode: EnhancementMode.customLLM.rawValue,
            AppPreferenceKey.customLLMModelRepo: textModel,
        ]
        let defaults = UserDefaults.standard
        let previous = Dictionary(uniqueKeysWithValues: (Array(settings.keys) + [AppPreferenceKey.modelStorageRootPath, AppPreferenceKey.featureSettings]).map {
            ($0, defaults.object(forKey: $0))
        })
        defer {
            for (key, value) in previous { defaults.set(value, forKey: key) }
        }
        // The default model folder, under the home this run is given.
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootPath)
        for (key, value) in settings { defaults.set(value, forKey: key) }
        // Transcription picks its models from the feature settings, as the Settings window saves them.
        var features = FeatureSettingsStore.load(defaults: defaults)
        features.transcription.asrSelectionID = .mlx(asrModel)
        features.transcription.llmEnabled = true
        features.transcription.llmSelectionID = .localLLM(textModel)
        FeatureSettingsStore.save(features, defaults: defaults, notify: false)

        let delegate = try XCTUnwrap(AppDelegate.shared)
        delegate.applyLocalAPIServerLifecycle()
        defer { delegate.stopLocalAPIServer() }
        try await Task.sleep(for: .seconds(seconds))
        XCTAssertEqual(delegate.localAPIServerStatus, .listening(port: port))
    }
}
