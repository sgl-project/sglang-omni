// VoxtLoggerTests.swift
// Provides Voxt Logger Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class VoxtLoggerTests: XCTestCase {
    func testVerboseLogDoesNotEvaluateMessageWhenVerboseLoggingIsDisabled() {
        let previousVerboseEnabled = VoxtLog.verboseEnabled
        VoxtLog.verboseEnabled = false
        defer {
            VoxtLog.verboseEnabled = previousVerboseEnabled
        }

        let logger = VoxtLogger(category: .app)
        var didEvaluateMessage = false

        logger.info({
            didEvaluateMessage = true
            return "expensive verbose message"
        }(), verbose: true)

        XCTAssertFalse(didEvaluateMessage)
    }
}
