// LocalAPIModelStatusTests.swift
// Capabilities describe the models a request would use, which come from the feature settings.

import XCTest
@testable import Voxt

@MainActor
final class LocalAPIModelStatusTests: XCTestCase {
    /// Every key `localAPIModelStatus()` writes while copying the feature settings onto the engine keys.
    private static let touchedKeys = [
        AppPreferenceKey.featureSettings,
        AppPreferenceKey.enhancementMode,
        AppPreferenceKey.enhancementSystemPrompt,
        AppPreferenceKey.customLLMModelRepo,
        AppPreferenceKey.remoteLLMSelectedProvider,
        AppPreferenceKey.transcriptionEngine,
        AppPreferenceKey.mlxModelRepo,
        AppPreferenceKey.remoteASRSelectedProvider,
    ]

    private var savedDefaults: [String: Any?] = [:]

    override func setUp() async throws {
        let defaults = UserDefaults.standard
        savedDefaults = Dictionary(uniqueKeysWithValues: Self.touchedKeys.map { ($0, defaults.object(forKey: $0)) })
    }

    override func tearDown() async throws {
        for (key, value) in savedDefaults { UserDefaults.standard.set(value, forKey: key) }
    }

    func testCapabilitiesFollowTheFeatureSettingsNotAStaleLegacyKey() throws {
        let delegate = try XCTUnwrap(AppDelegate.shared)
        let defaults = UserDefaults.standard
        var features = FeatureSettingsStore.load(defaults: defaults)
        features.transcription.llmEnabled = false
        FeatureSettingsStore.save(features, defaults: defaults, notify: false)
        // Left over from an earlier session; a request would reset it from the feature settings.
        defaults.set(EnhancementMode.remoteLLM.rawValue, forKey: AppPreferenceKey.enhancementMode)

        XCTAssertEqual(delegate.localAPIModelStatus().text, .off)
        XCTAssertEqual(delegate.localAPIModelLocations().text, .local, "no text model sends nothing anywhere")
    }

    func testADownloadedModelTheNativeRuntimeServesIsReady() throws {
        try XCTSkipUnless(OmniASRBackend.isAppleSilicon, "local models run only on Apple Silicon")
        let repo = MLXModelManager.defaultModelRepo
        XCTAssertNil(OmniASRBackend.unavailabilityReason(for: repo))
        let delegate = try selectLocalModel(repo)

        XCTAssertEqual(delegate.localAPIModelStatus(localASR: Self.context(repo, downloaded: true)).transcription, .ready)
        XCTAssertEqual(delegate.localAPIModelStatus(localASR: Self.context(repo, downloaded: false)).transcription, .missing)
    }

    /// A request for this model would be refused before transcribing (`RecordingStartPlanner`),
    /// so capabilities must not call it ready just because its files are on disk.
    func testADownloadedModelTheNativeRuntimeCannotRunIsNotReady() throws {
        let repo = "unknown/model"
        XCTAssertNotNil(OmniASRBackend.unavailabilityReason(for: repo))
        let delegate = try selectLocalModel(repo)

        XCTAssertEqual(delegate.localAPIModelStatus(localASR: Self.context(repo, downloaded: true)).transcription, .unsupported)
        XCTAssertEqual(delegate.localAPIModelStatus(localASR: Self.context(repo, downloaded: false)).transcription, .unsupported)
    }

    private func selectLocalModel(_ repo: String) throws -> AppDelegate {
        let delegate = try XCTUnwrap(AppDelegate.shared)
        let defaults = UserDefaults.standard
        var features = FeatureSettingsStore.load(defaults: defaults)
        features.transcription.asrSelectionID = .mlx(repo)
        features.transcription.llmEnabled = false
        FeatureSettingsStore.save(features, defaults: defaults, notify: false)
        return delegate
    }

    private static func context(_ repo: String, downloaded: Bool) -> AppDelegate.LocalASRStartContext {
        AppDelegate.LocalASRStartContext(
            selectedMLXRepo: repo,
            activeMLXDownloadRepo: nil,
            isSelectedMLXModelDownloaded: downloaded,
            mlxModelState: downloaded ? .downloaded : .notDownloaded
        )
    }
}
