// LocalAPIModelStatusTests.swift
// Capabilities describe the models a request would use, which come from the feature settings.

import XCTest
@testable import Voxt

@MainActor
final class LocalAPIModelStatusTests: XCTestCase {
    func testCapabilitiesFollowTheFeatureSettingsNotAStaleLegacyKey() throws {
        let delegate = try XCTUnwrap(AppDelegate.shared)
        let defaults = UserDefaults.standard
        let keys = [AppPreferenceKey.featureSettings, AppPreferenceKey.enhancementMode]
        let previous = Dictionary(uniqueKeysWithValues: keys.map { ($0, defaults.object(forKey: $0)) })
        defer {
            for (key, value) in previous { defaults.set(value, forKey: key) }
        }
        var features = FeatureSettingsStore.load(defaults: defaults)
        features.transcription.llmEnabled = false
        FeatureSettingsStore.save(features, defaults: defaults, notify: false)
        // Left over from an earlier session; a request would reset it from the feature settings.
        defaults.set(EnhancementMode.remoteLLM.rawValue, forKey: AppPreferenceKey.enhancementMode)

        XCTAssertEqual(delegate.localAPIModelStatus().text, .off)
        XCTAssertEqual(delegate.localAPIModelLocations().text, .local, "no text model sends nothing anywhere")
    }
}
