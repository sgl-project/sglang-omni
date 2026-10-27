// MeetingRealtimeTranslationSchedulerTests.swift
// Verifies bounded, single-consumer meeting translation behavior.

import XCTest
@testable import Voxt

@MainActor
final class MeetingRealtimeTranslationSchedulerTests: XCTestCase {
    func testTranslationsRunSeriallyInSubmissionOrder() async {
        let scheduler = MeetingRealtimeTranslationScheduler()
        var activeCount = 0
        var peakActiveCount = 0
        var completedTexts: [String] = []

        for index in 0..<5 {
            let source = "segment-\(index)"
            let result = scheduler.submit(
                segmentID: UUID(),
                sourceText: source,
                targetLanguage: .english,
                operation: MeetingTranslationOperation(executionScope: .localInference) {
                activeCount += 1
                peakActiveCount = max(peakActiveCount, activeCount)
                try await Task.sleep(for: .milliseconds(10))
                activeCount -= 1
                    return "translated-\(source)"
                }
            ) { result in
                if case .success(let text) = result {
                    completedTexts.append(text)
                }
            }
            XCTAssertEqual(result, .accepted)
        }

        await scheduler.flush()

        XCTAssertEqual(peakActiveCount, 1)
        XCTAssertEqual(
            completedTexts,
            (0..<5).map { "translated-segment-\($0)" }
        )
        XCTAssertEqual(scheduler.currentStatistics().completedCount, 5)
        XCTAssertLessThanOrEqual(scheduler.currentStatistics().inferenceBatchCount, 2)
        XCTAssertEqual(scheduler.currentStatistics().peakBatchSize, 4)
    }

    func testSchedulerEnforcesPendingJobSafetyBound() async {
        let scheduler = MeetingRealtimeTranslationScheduler()

        for index in 0..<9 {
            let result = scheduler.submit(
                segmentID: UUID(),
                sourceText: "segment-\(index)",
                targetLanguage: .english,
                operation: MeetingTranslationOperation(executionScope: .externalRequest) { "segment-\(index)" },
                completion: { _ in }
            )
            XCTAssertEqual(result, .accepted)
        }

        let overloaded = scheduler.submit(
            segmentID: UUID(),
            sourceText: "overloaded",
            targetLanguage: .english,
            operation: MeetingTranslationOperation(executionScope: .externalRequest) { "overloaded" },
            completion: { _ in }
        )

        XCTAssertEqual(overloaded, .overloaded)
        XCTAssertEqual(scheduler.currentStatistics().peakScheduledCount, 9)
        XCTAssertEqual(scheduler.currentStatistics().overloadedCount, 1)
        scheduler.cancelAll()
        await Task.yield()
    }

    func testExternalTranslationDoesNotHoldLocalInferencePermit() async throws {
        let scheduler = MeetingRealtimeTranslationScheduler()
        let gate = MeetingTranslationTestGate()
        let segmentID = UUID()

        _ = scheduler.submit(
            segmentID: segmentID,
            sourceText: "remote",
            targetLanguage: .english,
            operation: MeetingTranslationOperation(executionScope: .externalRequest) {
                await gate.wait()
                return "remote-result"
            },
            completion: { _ in }
        )
        await gate.waitUntilStarted()

        let localResult = try await MeetingLocalInferenceCoordinator.shared.withPermit(.liveASRFeed) {
            "asr-ran"
        }
        XCTAssertEqual(localResult, "asr-ran")

        await gate.open()
        await scheduler.flush()
    }

    func testNewTextRevisionSupersedesActiveTranslation() async {
        let scheduler = MeetingRealtimeTranslationScheduler()
        let gate = MeetingTranslationTestGate()
        let segmentID = UUID()
        var completed: [String] = []

        let first = scheduler.submit(
            segmentID: segmentID,
            sourceText: "old text",
            targetLanguage: .english,
            operation: MeetingTranslationOperation(executionScope: .externalRequest) {
                await gate.wait()
                return "old result"
            }
        ) { result in
            if case .success(let value) = result { completed.append(value) }
        }
        XCTAssertEqual(first, .accepted)
        await gate.waitUntilStarted()

        let updated = scheduler.submit(
            segmentID: segmentID,
            sourceText: "new text",
            targetLanguage: .english,
            operation: MeetingTranslationOperation(executionScope: .externalRequest) {
                "new result"
            }
        ) { result in
            if case .success(let value) = result { completed.append(value) }
        }
        XCTAssertEqual(updated, .updated)
        let newest = scheduler.submit(
            segmentID: segmentID,
            sourceText: "newest text",
            targetLanguage: .english,
            operation: MeetingTranslationOperation(executionScope: .externalRequest) {
                "newest result"
            }
        ) { result in
            if case .success(let value) = result { completed.append(value) }
        }
        XCTAssertEqual(newest, .updated)

        await gate.open()
        await scheduler.flush()
        XCTAssertEqual(completed, ["newest result"])
        XCTAssertEqual(scheduler.currentStatistics().peakScheduledCount, 1)
    }
}

private actor MeetingTranslationTestGate {
    private var started = false
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        started = true
        await withCheckedContinuation { continuation = $0 }
    }

    func waitUntilStarted() async {
        while !started { await Task.yield() }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}
