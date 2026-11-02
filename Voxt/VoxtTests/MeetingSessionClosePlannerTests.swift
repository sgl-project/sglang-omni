// MeetingSessionClosePlannerTests.swift
// Provides Meeting Session Close Planner Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class MeetingSessionClosePlannerTests: XCTestCase {
    func testAudioOnlySessionRequiresFinishConfirmation() {
        let decision = MeetingSessionClosePlanner.resolve(
            hasTranscriptSegments: false,
            hasCapturedAudio: true
        )

        XCTAssertEqual(decision, .confirmFinish)
    }

    func testTranscriptSessionRequiresFinishConfirmation() {
        let decision = MeetingSessionClosePlanner.resolve(
            hasTranscriptSegments: true,
            hasCapturedAudio: false
        )

        XCTAssertEqual(decision, .confirmFinish)
    }

    func testEmptySessionCanBeDiscardedImmediately() {
        let decision = MeetingSessionClosePlanner.resolve(
            hasTranscriptSegments: false,
            hasCapturedAudio: false
        )

        XCTAssertEqual(decision, .discard)
    }
}
