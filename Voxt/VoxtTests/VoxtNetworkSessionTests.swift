// VoxtNetworkSessionTests.swift
// Provides Voxt Network Session Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class VoxtNetworkSessionTests: XCTestCase {
    override func setUp() {
        super.setUp()
        VoxtSecureStorage.clearAllForTesting()
    }

    override func tearDown() {
        VoxtSecureStorage.clearAllForTesting()
        super.tearDown()
    }

    func testClearProcessProxyEnvironmentOverridesRemovesStandardProxyVariables() {
        let keys = [
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY"
        ]
        let originalValues = Dictionary(uniqueKeysWithValues: keys.map { key in
            (key, ProcessInfo.processInfo.environment[key])
        })

        setenv("http_proxy", "http://127.0.0.1:7897", 1)
        setenv("https_proxy", "http://127.0.0.1:7897", 1)
        setenv("all_proxy", "socks5://127.0.0.1:7897", 1)
        setenv("no_proxy", "localhost,127.0.0.1", 1)

        VoxtNetworkSession.clearProcessProxyEnvironmentOverridesIfNeeded()

        XCTAssertNil(ProcessInfo.processInfo.environment["http_proxy"])
        XCTAssertNil(ProcessInfo.processInfo.environment["https_proxy"])
        XCTAssertNil(ProcessInfo.processInfo.environment["all_proxy"])
        XCTAssertNil(ProcessInfo.processInfo.environment["no_proxy"])

        for (key, value) in originalValues {
            if let value {
                setenv(key, value, 1)
            } else {
                unsetenv(key)
            }
        }
    }

    func testLegacyProxyMigrationKeepsDefaultsWhenProtectedWriteFails() {
        let suiteName = "VoxtNetworkSessionTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        defaults.set("proxy-user", forKey: AppPreferenceKey.customProxyUsername)
        defaults.set("proxy-password", forKey: AppPreferenceKey.customProxyPassword)
        VoxtSecureStorage.setProtectedWritesFailForTesting(true)

        VoxtNetworkSession.migrateLegacyProxyCredentials(defaults: defaults)

        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.customProxyUsername), "proxy-user")
        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.customProxyPassword), "proxy-password")

        VoxtSecureStorage.setProtectedWritesFailForTesting(false)
        VoxtNetworkSession.migrateLegacyProxyCredentials(defaults: defaults)

        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.customProxyUsername) ?? "", "")
        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.customProxyPassword) ?? "", "")
        XCTAssertEqual(VoxtNetworkSession.proxyCredentials(defaults: defaults).username, "proxy-user")
        XCTAssertEqual(VoxtNetworkSession.proxyCredentials(defaults: defaults).password, "proxy-password")
    }

    func testLegacyKeychainCleanupFailureKeepsProxyCredentialsAvailable() {
        let suiteName = "VoxtNetworkSessionTests.keychainCleanup.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let usernameAccount = "network-proxy.customProxyUsername"
        let passwordAccount = "network-proxy.customProxyPassword"
        VoxtSecureStorage.setLegacyValueForTesting("proxy-user", for: usernameAccount)
        VoxtSecureStorage.setLegacyValueForTesting("proxy-password", for: passwordAccount)
        VoxtSecureStorage.setDeletesFailForTesting(true)

        let firstLoad = VoxtNetworkSession.proxyCredentials(defaults: defaults)

        XCTAssertEqual(firstLoad.username, "proxy-user")
        XCTAssertEqual(firstLoad.password, "proxy-password")
        XCTAssertTrue(VoxtSecureStorage.hasProtectedValueForTesting(for: usernameAccount))
        XCTAssertTrue(VoxtSecureStorage.hasProtectedValueForTesting(for: passwordAccount))
        XCTAssertTrue(VoxtSecureStorage.hasLegacyValueForTesting(for: usernameAccount))
        XCTAssertTrue(VoxtSecureStorage.hasLegacyValueForTesting(for: passwordAccount))

        VoxtSecureStorage.setDeletesFailForTesting(false)
        VoxtSecureStorage.clearCacheForTesting()

        let retriedLoad = VoxtNetworkSession.proxyCredentials(defaults: defaults)
        XCTAssertEqual(retriedLoad.username, "proxy-user")
        XCTAssertEqual(retriedLoad.password, "proxy-password")
        XCTAssertFalse(VoxtSecureStorage.hasLegacyValueForTesting(for: usernameAccount))
        XCTAssertFalse(VoxtSecureStorage.hasLegacyValueForTesting(for: passwordAccount))
    }
}
