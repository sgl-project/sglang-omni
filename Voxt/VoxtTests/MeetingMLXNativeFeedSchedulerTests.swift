import Foundation
import MLXAudioSTT
import XCTest
@testable import Voxt

@MainActor
final class MeetingMLXNativeFeedSchedulerTests: XCTestCase {
    func testFinishDrainsAcceptedSamplesOnceBeforeStop() async throws {
        let session = FeedTestSession()
        let collector = SampleCollector()
        let scheduler = MeetingMLXNativeFeedScheduler(session: session, deliver: { samples in
            await collector.append(samples)
        })
        let input = (0..<7_000).map(Float.init)
        let accepted = await scheduler.submit(samples: input, sampleRate: 16_000)
        XCTAssertTrue(accepted)
        try await scheduler.finish()
        try await scheduler.finish()
        let received = await collector.samples
        XCTAssertEqual(received, input)
        XCTAssertEqual(session.stopCount, 1)
        let pending = await scheduler.pendingSampleCount
        XCTAssertEqual(pending, 0)
    }

    func testRejectedPermitDoesNotConsumeAudioOrReportSuccess() async {
        let session = FeedTestSession()
        let scheduler = MeetingMLXNativeFeedScheduler(session: session, deliver: { _ in
            throw MeetingLocalInferenceCoordinatorError.overloaded
        })
        _ = await scheduler.submit(samples: [1, 2, 3], sampleRate: 16_000)
        do {
            try await scheduler.finish()
            XCTFail("Failed delivery must not finalize successfully")
        } catch {}
        let pending = await scheduler.pendingSampleCount
        XCTAssertEqual(pending, 3)
        XCTAssertEqual(session.stopCount, 0)
        await scheduler.cancel()
    }

    func testCancelDuringSuspendedDeliveryDoesNotRestoreOffsetOrStop() async {
        let session = FeedTestSession()
        let started = expectation(description: "delivery started")
        let gate = FeedTestGate()
        let scheduler = MeetingMLXNativeFeedScheduler(session: session, deliver: { _ in
            started.fulfill()
            await gate.wait()
        })
        _ = await scheduler.submit(samples: [1, 2, 3], sampleRate: 16_000)
        await fulfillment(of: [started], timeout: 1)
        await scheduler.cancel()
        await gate.open()
        do {
            try await scheduler.finish()
            XCTFail("Cancelled delivery must not stop successfully")
        } catch {}
        let pending = await scheduler.pendingSampleCount
        XCTAssertEqual(pending, 0)
        XCTAssertEqual(session.stopCount, 0)
        XCTAssertEqual(session.cancelCount, 1)
    }

    func testQueueOverloadIsExplicitAndDoesNotSilentlySkipFrame() async {
        let session = FeedTestSession()
        let started = expectation(description: "delivery suspended")
        let gate = FeedTestGate()
        let scheduler = MeetingMLXNativeFeedScheduler(session: session, maximumPendingSamples: 4, deliver: { _ in
            started.fulfill()
            await gate.wait()
        })
        _ = await scheduler.submit(samples: [1, 2, 3], sampleRate: 16_000)
        await fulfillment(of: [started], timeout: 1)
        let accepted = await scheduler.submit(samples: [4, 5], sampleRate: 16_000)
        let failure = await scheduler.failureMessage
        XCTAssertFalse(accepted)
        XCTAssertNotNil(failure)
        await gate.open()
        do { try await scheduler.finish(); XCTFail("Overload must fail live path") } catch {}
        XCTAssertEqual(session.stopCount, 0)
        await scheduler.cancel()
    }
}

private actor SampleCollector {
    var samples: [Float] = []
    func append(_ samples: [Float]) { self.samples += samples }
}

private actor FeedTestGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var opened = false
    func wait() async {
        guard !opened else { return }
        await withCheckedContinuation { continuation = $0 }
    }
    func open() {
        opened = true
        continuation?.resume()
        continuation = nil
    }
}

nonisolated private final class FeedTestSession: MLXNativeStreamingSession, @unchecked Sendable {
    let events: AsyncStream<TranscriptionEvent> = AsyncStream { $0.finish() }
    private let lock = NSLock()
    private var stops = 0
    private var cancels = 0
    var stopCount: Int { lock.withLock { stops } }
    var cancelCount: Int { lock.withLock { cancels } }
    func feedAudio(samples: [Float]) {}
    func stop() { lock.withLock { stops += 1 } }
    func cancel() { lock.withLock { cancels += 1 } }
}
