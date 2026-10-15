// RemoteProviderConnectivityLoggingTests.swift
// Provides Remote Provider Connectivity Logging Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class RemoteProviderConnectivityLoggingTests: XCTestCase {
    func testSanitizedHeadersRedactProviderCredentials() {
        let sanitized = RemoteProviderConnectivityTestLogging.sanitizedHeadersForLog([
            "Authorization": "Bearer authorization-secret",
            "Proxy-Authorization": "Basic proxy-secret",
            "X-Api-Key": "api-secret",
            "X-Api-Access-Key": "access-secret",
            "X-Api-App-Key": "app-secret",
            "X-Session-Token": "session-secret",
            "Cookie": "session=cookie-secret",
            "Content-Type": "application/json"
        ])

        XCTAssertEqual(sanitized["Authorization"], "<redacted>")
        XCTAssertEqual(sanitized["Proxy-Authorization"], "<redacted>")
        XCTAssertEqual(sanitized["X-Api-Key"], "<redacted>")
        XCTAssertEqual(sanitized["X-Api-Access-Key"], "<redacted>")
        XCTAssertEqual(sanitized["X-Api-App-Key"], "<redacted>")
        XCTAssertEqual(sanitized["X-Session-Token"], "<redacted>")
        XCTAssertEqual(sanitized["Cookie"], "<redacted>")
        XCTAssertEqual(sanitized["Content-Type"], "application/json")
    }

    func testSanitizedHeadersRedactSecretsEmbeddedInOtherwiseSafeValues() {
        let sanitized = RemoteProviderConnectivityTestLogging.sanitizedHeadersForLog([
            "X-Debug-Context": "region=cn access_token=embedded-secret"
        ])

        let value = sanitized["X-Debug-Context"] ?? ""
        XCTAssertFalse(value.contains("embedded-secret"))
        XCTAssertTrue(value.contains("<redacted>"))
    }
}
