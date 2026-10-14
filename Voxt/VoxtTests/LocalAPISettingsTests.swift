// LocalAPISettingsTests.swift
// Provides Local API settings tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class LocalAPISettingsTests: XCTestCase {
    private func withEphemeralDefaults(_ body: (UserDefaults) throws -> Void) rethrows {
        let suiteName = "LocalAPISettingsTests.\(UUID().uuidString)"
        guard let defaults = UserDefaults(suiteName: suiteName) else {
            return XCTFail("Expected ephemeral UserDefaults suite")
        }
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }
        try body(defaults)
    }

    func testDisabledAndDefaultPortWhenUnset() {
        withEphemeralDefaults { defaults in
            XCTAssertFalse(LocalAPISettings.isEnabled(defaults: defaults))
            XCTAssertEqual(LocalAPISettings.port(defaults: defaults), LocalAPISettings.defaultPort)
        }
    }

    func testOutOfRangePortFallsBackToDefault() {
        withEphemeralDefaults { defaults in
            defaults.set(80, forKey: AppPreferenceKey.localAPIServerPort)
            XCTAssertEqual(LocalAPISettings.port(defaults: defaults), LocalAPISettings.defaultPort)
            defaults.set(65535, forKey: AppPreferenceKey.localAPIServerPort)
            XCTAssertEqual(LocalAPISettings.port(defaults: defaults), LocalAPISettings.defaultPort)
            defaults.set(52000, forKey: AppPreferenceKey.localAPIServerPort)
            XCTAssertEqual(LocalAPISettings.port(defaults: defaults), 52000)
        }
    }

    func testTokenIsStableAcrossReadsWhenKeychainWritesFail() {
        VoxtSecureStorage.clearAllForTesting()
        LocalAPISettings.resetUnsavedTokenForTesting()
        VoxtSecureStorage.setProtectedWritesFailForTesting(true)
        defer {
            VoxtSecureStorage.setProtectedWritesFailForTesting(false)
            VoxtSecureStorage.clearAllForTesting()
            LocalAPISettings.resetUnsavedTokenForTesting()
        }

        let first = LocalAPISettings.token(environment: [:])
        XCTAssertEqual(LocalAPISettings.token(environment: [:]), first)

        let regenerated = LocalAPISettings.regenerateToken()
        XCTAssertNotEqual(regenerated, first)
        XCTAssertEqual(LocalAPISettings.token(environment: [:]), regenerated)
    }

    func testTokenIsSavedToSecureStorage() {
        VoxtSecureStorage.clearAllForTesting()
        LocalAPISettings.resetUnsavedTokenForTesting()
        defer { VoxtSecureStorage.clearAllForTesting() }

        let token = LocalAPISettings.token(environment: [:])
        XCTAssertEqual(VoxtSecureStorage.string(for: LocalAPISettings.tokenAccount), token)
    }

    func testDebugBuildsAcceptTokenFromEnvironment() {
        XCTAssertEqual(
            LocalAPISettings.token(environment: ["VOXT_LOCAL_API_TOKEN": "from-env"]),
            "from-env"
        )
    }
}
