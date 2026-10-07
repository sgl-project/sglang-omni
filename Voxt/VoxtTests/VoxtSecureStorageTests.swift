// VoxtSecureStorageTests.swift
// Verifies Data Protection Keychain migration behavior without touching the user's keychain.

import XCTest
import Security
@testable import Voxt

final class VoxtSecureStorageTests: XCTestCase {
    override func setUp() {
        super.setUp()
        VoxtSecureStorage.clearAllForTesting()
    }

    override func tearDown() {
        VoxtSecureStorage.clearAllForTesting()
        super.tearDown()
    }

    func testLegacyValueMigratesOnlyAfterProtectedWriteSucceeds() {
        let account = "remote-provider.doubaoASR.credentials"
        VoxtSecureStorage.setLegacyValueForTesting("legacy-secret", for: account)

        XCTAssertEqual(VoxtSecureStorage.string(for: account), "legacy-secret")
        XCTAssertTrue(VoxtSecureStorage.hasProtectedValueForTesting(for: account))
        XCTAssertFalse(VoxtSecureStorage.hasLegacyValueForTesting(for: account))
    }

    func testFailedProtectedWriteKeepsLegacyValueAvailableForCurrentRequest() {
        let account = "remote-provider.doubaoASR.credentials"
        VoxtSecureStorage.setLegacyValueForTesting("legacy-secret", for: account)
        VoxtSecureStorage.setProtectedWritesFailForTesting(true)

        XCTAssertEqual(VoxtSecureStorage.string(for: account), "legacy-secret")
        XCTAssertFalse(VoxtSecureStorage.hasProtectedValueForTesting(for: account))
        XCTAssertTrue(VoxtSecureStorage.hasLegacyValueForTesting(for: account))

        VoxtSecureStorage.clearCacheForTesting()
        VoxtSecureStorage.setProtectedWritesFailForTesting(false)

        XCTAssertEqual(VoxtSecureStorage.string(for: account), "legacy-secret")
        XCTAssertTrue(VoxtSecureStorage.hasProtectedValueForTesting(for: account))
        XCTAssertFalse(VoxtSecureStorage.hasLegacyValueForTesting(for: account))
    }

    func testFailedLegacyCleanupReturnsProtectedValueAndRetriesLater() {
        let account = "remote-provider.doubaoASR.credentials"
        VoxtSecureStorage.setLegacyValueForTesting("legacy-secret", for: account)
        VoxtSecureStorage.setDeletesFailForTesting(true)

        XCTAssertEqual(VoxtSecureStorage.string(for: account), "legacy-secret")
        XCTAssertTrue(VoxtSecureStorage.hasProtectedValueForTesting(for: account))
        XCTAssertTrue(VoxtSecureStorage.hasLegacyValueForTesting(for: account))

        VoxtSecureStorage.clearCacheForTesting()
        VoxtSecureStorage.setDeletesFailForTesting(false)

        XCTAssertEqual(VoxtSecureStorage.string(for: account), "legacy-secret")
        XCTAssertTrue(VoxtSecureStorage.hasProtectedValueForTesting(for: account))
        XCTAssertFalse(VoxtSecureStorage.hasLegacyValueForTesting(for: account))
    }

    func testProtectedValueTakesPrecedenceAndCleansUpLegacyValue() {
        let account = "remote-provider.doubaoASR.credentials"
        XCTAssertTrue(VoxtSecureStorage.set("current-secret", for: account))
        VoxtSecureStorage.setLegacyValueForTesting("legacy-secret", for: account)

        XCTAssertEqual(VoxtSecureStorage.string(for: account), "current-secret")
        XCTAssertFalse(VoxtSecureStorage.hasLegacyValueForTesting(for: account))
    }

    func testRemovingValueClearsProtectedAndLegacyCopies() {
        let account = "remote-provider.doubaoASR.credentials"
        XCTAssertTrue(VoxtSecureStorage.set("current-secret", for: account))
        VoxtSecureStorage.setLegacyValueForTesting("legacy-secret", for: account)

        VoxtSecureStorage.removeValue(for: account)

        XCTAssertNil(VoxtSecureStorage.string(for: account))
        XCTAssertFalse(VoxtSecureStorage.hasProtectedValueForTesting(for: account))
        XCTAssertFalse(VoxtSecureStorage.hasLegacyValueForTesting(for: account))
    }

    func testRemovingValueReportsFailureAndPreservesStoredCopies() {
        let account = "remote-provider.doubaoASR.credentials"
        XCTAssertTrue(VoxtSecureStorage.set("current-secret", for: account))
        VoxtSecureStorage.setLegacyValueForTesting("legacy-secret", for: account)
        VoxtSecureStorage.setDeletesFailForTesting(true)

        XCTAssertFalse(VoxtSecureStorage.removeValue(for: account))

        XCTAssertTrue(VoxtSecureStorage.hasProtectedValueForTesting(for: account))
        XCTAssertTrue(VoxtSecureStorage.hasLegacyValueForTesting(for: account))
    }

    func testLegacyQueryExplicitlyExcludesDataProtectionKeychain() {
        let query = VoxtSecureStorage.legacyQuery(
            for: "remote-provider.doubaoASR.credentials",
            interactionAllowed: false
        )

        XCTAssertEqual(query[kSecUseDataProtectionKeychain as String] as? Bool, false)
    }
}
