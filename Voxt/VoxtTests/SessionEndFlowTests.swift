// SessionEndFlowTests.swift
// Provides Session End Flow Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class SessionEndFlowTests: XCTestCase {
    func testSessionCallbackHandlingDecisionAcceptsActiveNonCancelledSession() {
        let sessionID = UUID()

        XCTAssertEqual(
            AppDelegate.sessionCallbackHandlingDecision(
                requestedSessionID: sessionID,
                activeSessionID: sessionID,
                isSessionCancellationRequested: false
            ),
            .accept
        )
    }

    func testSessionCallbackHandlingDecisionRejectsStaleSession() {
        XCTAssertEqual(
            AppDelegate.sessionCallbackHandlingDecision(
                requestedSessionID: UUID(),
                activeSessionID: UUID(),
                isSessionCancellationRequested: false
            ),
            .rejectStale
        )
    }

    func testSessionCallbackHandlingDecisionRejectsCancelledSession() {
        let sessionID = UUID()

        XCTAssertEqual(
            AppDelegate.sessionCallbackHandlingDecision(
                requestedSessionID: sessionID,
                activeSessionID: sessionID,
                isSessionCancellationRequested: true
            ),
            .rejectCancelled
        )
    }

    func testSessionEndExecutionDecisionRejectsDuplicateInFlightSession() {
        var lifecycle = RecordingSessionLifecycle()
        let sessionID = lifecycle.id
        XCTAssertEqual(lifecycle.beginEnding(sessionID), .execute)
        XCTAssertEqual(lifecycle.beginEnding(sessionID), .skipDuplicateInFlight)
    }

    func testSessionEndExecutionDecisionRejectsAlreadyCompletedSession() {
        var lifecycle = RecordingSessionLifecycle()
        let sessionID = lifecycle.id
        XCTAssertEqual(lifecycle.beginEnding(sessionID), .execute)
        lifecycle.completeEnding(sessionID)
        XCTAssertEqual(lifecycle.beginEnding(sessionID), .skipAlreadyCompleted)
    }
}
