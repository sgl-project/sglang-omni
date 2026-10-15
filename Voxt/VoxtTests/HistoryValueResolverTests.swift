// HistoryValueResolverTests.swift
// Provides History Value Resolver Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class HistoryValueResolverTests: XCTestCase {
    func testResolvedDurationHandlesNilAndNegativeValues() {
        let start = Date(timeIntervalSince1970: 10)
        let end = Date(timeIntervalSince1970: 15)

        XCTAssertEqual(HistoryValueResolver.resolvedDuration(from: start, to: end), 5)
        XCTAssertNil(HistoryValueResolver.resolvedDuration(from: end, to: start))
        XCTAssertNil(HistoryValueResolver.resolvedDuration(from: nil, to: end))
    }

    func testResolvedKindMapsSessionOutputModes() {
        XCTAssertEqual(HistoryValueResolver.resolvedKind(for: .transcription), .normal)
        XCTAssertEqual(HistoryValueResolver.resolvedKind(for: .translation), .translation)
        XCTAssertEqual(HistoryValueResolver.resolvedKind(for: .rewrite), .rewrite)
    }

    func testHistoryDisplayEndpointRedactsSensitiveQueryValues() {
        let endpoint = HistoryValueResolver.historyDisplayEndpoint(
            "https://example.com/v1/chat?api_key=secret&token=abc&mode=test"
        )

        XCTAssertEqual(
            endpoint,
            "https://example.com/v1/chat?api_key=%3Credacted%3E&token=%3Credacted%3E&mode=test"
        )
        XCTAssertEqual(
            HistoryValueResolver.historyDisplayEndpoint("   "),
            AppLocalization.localizedString("Default")
        )
    }
}
