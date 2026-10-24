// MeetingLocalInferenceCoordinatorTests.swift
// Verifies serialization, priority, and recording-time resource protection.

import XCTest
@testable import Voxt

final class MeetingLocalInferenceCoordinatorTests: XCTestCase {
    func testFileAnalysisUsesLowestPriorityAndYieldsWhileRecording() {
        XCTAssertLessThan(
            MeetingLocalInferenceWorkClass.fileASR.priority,
            MeetingLocalInferenceWorkClass.finalASR.priority
        )
        XCTAssertTrue(MeetingLocalInferenceWorkClass.fileASR.waitsWhileRecording)
        XCTAssertTrue(MeetingLocalInferenceWorkClass.fileASR.isThermallyDeferrable)
        XCTAssertTrue(MeetingLocalInferenceWorkClass.fileASR.isMemoryDeferrable)
    }

    func testHeavyInferenceRunsOneOperationAtATime() async throws {
        let coordinator = MeetingLocalInferenceCoordinator(readThermalState: { .nominal })
        let tracker = MeetingInferenceConcurrencyTracker()

        async let first: Void = coordinator.withPermit(.liveASRFinal) {
            await tracker.begin()
            try await Task.sleep(for: .milliseconds(20))
            await tracker.end()
        }
        async let second: Void = coordinator.withPermit(.realtimeTranslation) {
            await tracker.begin()
            try await Task.sleep(for: .milliseconds(20))
            await tracker.end()
        }

        _ = try await (first, second)
        let peakCount = await tracker.peakCount()
        XCTAssertEqual(peakCount, 1)
    }

    func testHigherPriorityWorkRunsBeforeQueuedBackgroundWork() async throws {
        let coordinator = MeetingLocalInferenceCoordinator(readThermalState: { .nominal })
        let gate = MeetingInferenceGate()
        let order = MeetingInferenceOrderRecorder()

        let active = Task {
            try await coordinator.withPermit(.liveASRFinal) {
                await gate.wait()
            }
        }
        await gate.waitUntilStarted()

        let background = Task {
            try await coordinator.withPermit(.summary) {
                await order.append("summary")
            }
        }
        let didQueueBackground = await waitForQueuedInferenceWork(1, coordinator: coordinator)
        XCTAssertTrue(didQueueBackground, "Background work did not enter the inference queue")

        let live = Task {
            try await coordinator.withPermit(.liveASRFeed) {
                await order.append("live")
            }
        }
        let didQueueLiveWork = await waitForQueuedInferenceWork(2, coordinator: coordinator)
        XCTAssertTrue(didQueueLiveWork, "Live work did not enter the inference queue")

        await gate.open()

        _ = try await (active.value, background.value, live.value)
        let values = await order.values()
        XCTAssertEqual(values, ["live", "summary"])
    }

    func testFileAnalysisWaitsThroughMemoryPressureAndResumes() async throws {
        let coordinator = MeetingLocalInferenceCoordinator(readThermalState: { .nominal })
        let tracker = MeetingInferenceConcurrencyTracker()
        await coordinator.setMemoryPressureConstrained(true)

        let task = Task {
            try await coordinator.withPermit(.fileASR) {
                await tracker.begin()
                await tracker.end()
            }
        }
        try await Task.sleep(for: .milliseconds(100))
        let peakWhilePressured = await tracker.peakCount()
        XCTAssertEqual(peakWhilePressured, 0)

        await coordinator.setMemoryPressureConstrained(false)
        try await task.value
        let peakAfterPressure = await tracker.peakCount()
        XCTAssertEqual(peakAfterPressure, 1)
    }

    func testFileAnalysisWaitsThroughRecordingAndDoesNotFail() async throws {
        let coordinator = MeetingLocalInferenceCoordinator(readThermalState: { .nominal })
        let tracker = MeetingInferenceConcurrencyTracker()
        await coordinator.setRecordingActive(true)

        let task = Task {
            try await coordinator.withPermit(.fileASR) {
                await tracker.begin()
                await tracker.end()
            }
        }
        try await Task.sleep(for: .milliseconds(100))
        let peakWhileRecording = await tracker.peakCount()
        XCTAssertEqual(peakWhileRecording, 0)

        await coordinator.setRecordingActive(false)
        try await task.value
        let peakAfterRecording = await tracker.peakCount()
        XCTAssertEqual(peakAfterRecording, 1)
    }

    func testNonFileDeferrableWorkStillReportsMemoryPressure() async throws {
        let coordinator = MeetingLocalInferenceCoordinator(readThermalState: { .nominal })
        await coordinator.setMemoryPressureConstrained(true)

        do {
            _ = try await coordinator.withPermit(.summary) {}
            XCTFail("Expected non-file background work to be rejected")
        } catch let error as MeetingLocalInferenceCoordinatorError {
            guard case .memoryConstrained = error else {
                return XCTFail("Unexpected coordinator error: \(error)")
            }
        }
    }

    func testFileSpeakerAnalysisWaitsForPressureAndCanBeCancelled() async throws {
        let coordinator = MeetingLocalInferenceCoordinator(readThermalState: { .nominal })
        let tracker = MeetingInferenceConcurrencyTracker()
        await coordinator.setMemoryPressureConstrained(true)
        let task = Task {
            try await coordinator.withPermit(.fileSpeakerAnalysis) {
                await tracker.begin()
                await tracker.end()
            }
        }
        try await Task.sleep(for: .milliseconds(50))
        task.cancel()
        do { try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError) }
        let peak = await tracker.peakCount()
        XCTAssertEqual(peak, 0)
        await coordinator.setMemoryPressureConstrained(false)
        try await coordinator.withPermit(.fileSpeakerAnalysis) {
            await tracker.begin()
            await tracker.end()
        }
        let recoveredPeak = await tracker.peakCount()
        XCTAssertEqual(recoveredPeak, 1)
    }

    func testPressureAppearingWhileFileSpeakerWaitsForLaneIsRechecked() async throws {
        let coordinator = MeetingLocalInferenceCoordinator(readThermalState: { .nominal })
        let gate = MeetingInferenceGate()
        let tracker = MeetingInferenceConcurrencyTracker()
        let first = Task {
            try await coordinator.withPermit(.liveASRFinal) { await gate.wait() }
        }
        await gate.waitUntilStarted()
        let speaker = Task {
            try await coordinator.withPermit(.fileSpeakerAnalysis) {
                await tracker.begin()
                await tracker.end()
            }
        }
        try await Task.sleep(for: .milliseconds(50))
        await coordinator.setMemoryPressureConstrained(true)
        await gate.open()
        try await first.value
        try await Task.sleep(for: .milliseconds(300))
        let peak = await tracker.peakCount()
        XCTAssertEqual(peak, 0)
        await coordinator.setMemoryPressureConstrained(false)
        try await speaker.value
        let recoveredPeak = await tracker.peakCount()
        XCTAssertEqual(recoveredPeak, 1)
    }

    func testBlockedNonFileWaiterDoesNotPreventBoundedFileWorkUnderWarning() async throws {
        let coordinator = MeetingLocalInferenceCoordinator(readMemoryPressure: { false }, readThermalState: { .nominal })
        let gate = MeetingInferenceGate()
        let order = MeetingInferenceOrderRecorder()
        let active = Task { try await coordinator.withPermit(.liveASRFinal) { await gate.wait() } }
        await gate.waitUntilStarted()
        let summary = Task { try await coordinator.withPermit(.summary) { await order.append("summary") } }
        let queued = await waitForQueuedInferenceWork(1, coordinator: coordinator)
        XCTAssertTrue(queued)
        await coordinator.setMemoryPressureConstrained(true)
        await gate.open()
        try await active.value
        try await coordinator.withPermit(.fileSpeakerAnalysis) { await order.append("file") }
        let duringWarning = await order.values()
        XCTAssertEqual(duringWarning, ["file"])
        await coordinator.setMemoryPressureConstrained(false)
        try await summary.value
        let final = await order.values()
        XCTAssertEqual(final, ["file", "summary"])
    }

    func testBackgroundWorkWaitsWhileRecording() async throws {
        let coordinator = MeetingLocalInferenceCoordinator(readThermalState: { .nominal })
        let tracker = MeetingInferenceConcurrencyTracker()
        await coordinator.setRecordingActive(true)

        let task = Task {
            try await coordinator.withPermit(.summary) {
                await tracker.begin()
                await tracker.end()
            }
        }
        try await Task.sleep(for: .milliseconds(20))
        let peakWhileRecording = await tracker.peakCount()
        XCTAssertEqual(peakWhileRecording, 0)

        await coordinator.setRecordingActive(false)
        try await task.value
        let peakAfterRecording = await tracker.peakCount()
        XCTAssertEqual(peakAfterRecording, 1)
    }

    func testOrdinaryPermitStillRestoresCacheOnErrorAndCancellation() async throws {
        for shouldCancel in [false, true] {
            let hooks = MeetingInferenceHooks()
            let coordinator = hooks.makeCoordinator()
            do {
                try await coordinator.withPermit(.fileSpeakerAnalysis) {
                    if shouldCancel { throw CancellationError() }
                    throw URLError(.cannotDecodeContentData)
                }
                XCTFail("Expected operation error")
            } catch {
                if shouldCancel { XCTAssertTrue(error is CancellationError) }
                else { XCTAssertEqual((error as? URLError)?.code, .cannotDecodeContentData) }
            }
            XCTAssertEqual(hooks.events, [.trim(false), .begin, .trim(false), .restore])
            let statistics = await coordinator.currentStatistics()
            XCTAssertEqual(statistics.completedCount, 0)
            try await coordinator.withPermit(.liveASRFinal) { hooks.record(.probe) }
            XCTAssertEqual(hooks.events.last, .probe)
        }
    }

}
