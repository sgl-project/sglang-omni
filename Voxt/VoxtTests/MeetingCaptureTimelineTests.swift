// MeetingCaptureTimelineTests.swift
// Verifies source restart anchoring and stale-callback rejection.

import XCTest
@testable import Voxt

final class MeetingCaptureTimelineTests: XCTestCase {
    func testRestartedSourceAnchorsAtCurrentMeetingTime() {
        var tracker = MeetingCaptureTimelineTracker()
        let firstGeneration = tracker.beginEpoch(for: .me)
        tracker.anchorEpoch(for: .me, generation: firstGeneration, minimumStartSeconds: 0)
        let first = tracker.nextRange(
            for: .me,
            generation: firstGeneration,
            durationSeconds: 1,
            fallbackEndSeconds: 1
        )
        XCTAssertEqual(first?.lowerBound, 0)
        XCTAssertEqual(first?.upperBound, 1)

        let restartedGeneration = tracker.beginEpoch(for: .me)
        tracker.anchorEpoch(for: .me, generation: restartedGeneration, minimumStartSeconds: 4)
        let restarted = tracker.nextRange(
            for: .me,
            generation: restartedGeneration,
            durationSeconds: 0.5,
            fallbackEndSeconds: 4.5
        )

        XCTAssertEqual(restarted?.lowerBound, 4)
        XCTAssertEqual(restarted?.upperBound, 4.5)
    }

    func testCleanupInvalidatesCallbacksBeforeNextCaptureStarts() {
        var tracker = MeetingCaptureTimelineTracker()
        let generation = tracker.beginEpoch(for: .me)
        XCTAssertTrue(tracker.isCurrent(generation, for: .me))
        tracker.invalidateEpochs()
        XCTAssertFalse(tracker.isCurrent(generation, for: .me))
        XCTAssertNil(tracker.nextRange(for: .me, generation: generation, durationSeconds: 1, fallbackEndSeconds: 1))
        let next = tracker.beginEpoch(for: .me)
        XCTAssertNotEqual(next, generation)
        XCTAssertTrue(tracker.isCurrent(next, for: .me))
    }

    func testStaleCaptureGenerationCannotAdvanceNewTimeline() {
        var tracker = MeetingCaptureTimelineTracker()
        let staleGeneration = tracker.beginEpoch(for: .me)
        tracker.anchorEpoch(for: .me, generation: staleGeneration, minimumStartSeconds: 0)
        let currentGeneration = tracker.beginEpoch(for: .me)
        tracker.anchorEpoch(for: .me, generation: currentGeneration, minimumStartSeconds: 10)

        let stale = tracker.nextRange(
            for: .me,
            generation: staleGeneration,
            durationSeconds: 1,
            fallbackEndSeconds: 11
        )
        let current = tracker.nextRange(
            for: .me,
            generation: currentGeneration,
            durationSeconds: 1,
            fallbackEndSeconds: 11
        )

        XCTAssertNil(stale)
        XCTAssertEqual(current?.lowerBound, 10)
        XCTAssertEqual(current?.upperBound, 11)
    }

    func testRestartingMicrophoneDoesNotMoveSystemAudioCursor() {
        var tracker = MeetingCaptureTimelineTracker()
        let microphoneGeneration = tracker.beginEpoch(for: .me)
        let systemGeneration = tracker.beginEpoch(for: .them)
        tracker.anchorEpoch(for: .me, generation: microphoneGeneration, minimumStartSeconds: 0)
        tracker.anchorEpoch(for: .them, generation: systemGeneration, minimumStartSeconds: 0)
        _ = tracker.nextRange(for: .them, generation: systemGeneration, durationSeconds: 3, fallbackEndSeconds: 3)

        let restartedMicrophoneGeneration = tracker.beginEpoch(for: .me)
        tracker.anchorEpoch(for: .me, generation: restartedMicrophoneGeneration, minimumStartSeconds: 5)
        let systemNext = tracker.nextRange(
            for: .them,
            generation: systemGeneration,
            durationSeconds: 1,
            fallbackEndSeconds: 4
        )

        XCTAssertEqual(systemNext?.lowerBound, 3)
        XCTAssertEqual(systemNext?.upperBound, 4)
    }
}
