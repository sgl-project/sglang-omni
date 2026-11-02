import XCTest
@testable import Voxt

final class RecordingSessionLifecycleTests: XCTestCase {
    func testOutputCanBeClaimedOnlyOnce() {
        var lifecycle = RecordingSessionLifecycle()
        let id = lifecycle.id
        XCTAssertTrue(lifecycle.claimOutput(for: id))
        XCTAssertFalse(lifecycle.claimOutput(for: id))
        XCTAssertTrue(lifecycle.hasCommittedOutput)
    }

    func testCancelInvalidatesOutputButAllowsCancelledSessionCleanup() {
        var lifecycle = RecordingSessionLifecycle()
        let cancelledID = lifecycle.id
        lifecycle.cancel()
        XCTAssertNotEqual(lifecycle.id, cancelledID)
        XCTAssertFalse(lifecycle.accepts(cancelledID))
        XCTAssertFalse(lifecycle.claimOutput(for: cancelledID))
        XCTAssertFalse(lifecycle.claimOutput(for: lifecycle.id))
        XCTAssertEqual(lifecycle.beginEnding(cancelledID), .execute)
    }

    func testNewSessionRejectsOldOutputAndEndRequest() {
        var lifecycle = RecordingSessionLifecycle()
        let old = lifecycle.id
        lifecycle.begin()
        XCTAssertFalse(lifecycle.claimOutput(for: old))
        XCTAssertEqual(lifecycle.beginEnding(old), .skipStale)
        XCTAssertNil(lifecycle.endingID)
        XCTAssertTrue(lifecycle.claimOutput(for: lifecycle.id))
    }

    func testLateEndCompletionCannotClearNewEndingSession() {
        var lifecycle = RecordingSessionLifecycle()
        let old = lifecycle.id
        XCTAssertEqual(lifecycle.beginEnding(old), .execute)
        lifecycle.begin()
        let current = lifecycle.id
        XCTAssertEqual(lifecycle.beginEnding(current), .execute)
        lifecycle.completeEnding(old)
        XCTAssertEqual(lifecycle.endingID, current)
        XCTAssertNil(lifecycle.completedEndID)
    }

    func testCallbackInvalidationDuringEndPreservesEndAdmission() {
        var lifecycle = RecordingSessionLifecycle()
        let ending = lifecycle.id
        XCTAssertEqual(lifecycle.beginEnding(ending), .execute)
        lifecycle.invalidateCallbacks()
        XCTAssertFalse(lifecycle.accepts(ending))
        lifecycle.completeEnding(ending)
        XCTAssertEqual(lifecycle.completedEndID, ending)
        XCTAssertEqual(lifecycle.beginEnding(ending), .skipAlreadyCompleted)
    }

    func testOutputGenerationSurvivesNormalCallbackTeardown() {
        var lifecycle = RecordingSessionLifecycle()
        let sessionID = lifecycle.id
        let generation = lifecycle.outputGeneration
        XCTAssertTrue(lifecycle.claimOutput(for: sessionID))
        XCTAssertEqual(lifecycle.beginEnding(sessionID), .execute)
        lifecycle.invalidateCallbacks()
        lifecycle.completeEnding(sessionID)
        XCTAssertFalse(lifecycle.accepts(sessionID))
        XCTAssertTrue(lifecycle.acceptsOutputGeneration(generation))
    }

    func testNewSessionInvalidatesOldOutputGeneration() {
        var lifecycle = RecordingSessionLifecycle()
        let generation = lifecycle.outputGeneration
        lifecycle.begin()
        XCTAssertFalse(lifecycle.acceptsOutputGeneration(generation))
        XCTAssertTrue(lifecycle.acceptsOutputGeneration(lifecycle.outputGeneration))
    }

    func testCancellationInvalidatesOldOutputButAllowsFreshManualRequest() {
        var lifecycle = RecordingSessionLifecycle()
        let generation = lifecycle.outputGeneration
        lifecycle.cancel()
        XCTAssertFalse(lifecycle.acceptsOutputGeneration(generation))
        XCTAssertFalse(lifecycle.accepts(lifecycle.id))
        // An explicit manual paste can be requested after cancellation; it is
        // not a continuation of the cancelled session's automated delivery.
        XCTAssertTrue(lifecycle.acceptsOutputGeneration(lifecycle.outputGeneration))
    }

    func testDismissInvalidatesQueuedOutputWithoutCancellingRecording() {
        var lifecycle = RecordingSessionLifecycle()
        let sessionID = lifecycle.id
        let generation = lifecycle.outputGeneration
        lifecycle.invalidateOutputDelivery()
        XCTAssertFalse(lifecycle.acceptsOutputGeneration(generation))
        XCTAssertTrue(lifecycle.accepts(sessionID))
        XCTAssertFalse(lifecycle.isCancelled)
    }

    func testBeginResetsCancellationAndCommitTogether() {
        var lifecycle = RecordingSessionLifecycle()
        let old = lifecycle.id
        lifecycle.cancel()
        _ = lifecycle.beginEnding(old)
        lifecycle.completeEnding(old)
        lifecycle.begin()
        XCTAssertFalse(lifecycle.isCancelled)
        XCTAssertFalse(lifecycle.hasCommittedOutput)
        XCTAssertNil(lifecycle.endingID)
        XCTAssertNil(lifecycle.completedEndID)
        XCTAssertEqual(lifecycle.beginEnding(old), .skipStale)
    }
}
