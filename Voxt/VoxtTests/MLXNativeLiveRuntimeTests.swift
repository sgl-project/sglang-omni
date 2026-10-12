import XCTest
import MLXAudioSTT
@testable import Voxt

@MainActor
final class MLXNativeLiveRuntimeTests: XCTestCase {
    func testReleaseKeepsModelUseUntilOwnedTasksHaveExited() async {
        let runtime = MLXNativeLiveRuntime()
        let session = RuntimeTestStreamingSession()
        var releases = 0
        install(session, in: runtime, release: { releases += 1 })
        runtime.release(cancelSession: true)
        XCTAssertNil(runtime.session)
        XCTAssertTrue(runtime.hasPendingWork)
        XCTAssertEqual(releases, 0)
        await runtime.waitForRetirement()
        XCTAssertFalse(runtime.hasPendingWork)
        XCTAssertEqual(session.cancelCount, 1)
        XCTAssertEqual(releases, 1)
        runtime.release(cancelSession: true)
        await runtime.waitForRetirement()
        XCTAssertEqual(releases, 1)
    }

    func testReplacingSessionDoesNotReleaseReplacementModelUse() async {
        let runtime = MLXNativeLiveRuntime()
        let old = RuntimeTestStreamingSession()
        let current = RuntimeTestStreamingSession()
        var releases: [String] = []
        install(old, in: runtime, release: { releases.append("old") })
        install(current, in: runtime, release: { releases.append("current") })
        await runtime.waitForRetirement()
        XCTAssertTrue(runtime.session === current)
        XCTAssertEqual(releases, ["old"])
        XCTAssertEqual(current.cancelCount, 0)
        runtime.release(cancelSession: true)
        await runtime.waitForRetirement()
        XCTAssertEqual(releases, ["old", "current"])
    }

    func testQueuedOldEventsAreRejectedAfterReplacement() async {
        let runtime = MLXNativeLiveRuntime()
        let old = RuntimeTestStreamingSession()
        let current = RuntimeTestStreamingSession()
        let delivered = expectation(description: "current event")
        var texts: [String] = []
        runtime.install(old, pollInterval: .seconds(60), nextSamples: { [] }, shouldContinue: { false }, onEvent: { _ in
            XCTFail("Old session delivered an event")
        }, releaseModel: {})
        old.emit("old")
        runtime.install(current, pollInterval: .seconds(60), nextSamples: { [] }, shouldContinue: { false }, onEvent: { event in
            if case .displayUpdate(let text, _) = event {
                texts.append(text)
                delivered.fulfill()
            }
        }, releaseModel: {})
        current.emit("current")
        await fulfillment(of: [delivered], timeout: 1)
        XCTAssertEqual(texts, ["current"])
        runtime.release(cancelSession: true)
        await runtime.waitForRetirement()
    }

    func testReleaseFromEventCallbackDoesNotDeadlockOrReleaseTwice() async {
        let runtime = MLXNativeLiveRuntime()
        let session = RuntimeTestStreamingSession()
        let delivered = expectation(description: "event released stream")
        var releases = 0
        runtime.install(session, pollInterval: .seconds(60), nextSamples: { [] }, shouldContinue: { false }, onEvent: { _ in
            runtime.release(cancelSession: false)
            delivered.fulfill()
        }, releaseModel: { releases += 1 })
        session.emit("terminal")
        await fulfillment(of: [delivered], timeout: 1)
        await runtime.waitForRetirement()
        XCTAssertEqual(session.cancelCount, 0)
        XCTAssertEqual(releases, 1)
        XCTAssertFalse(runtime.hasPendingWork)
    }

    func testStopFeedingBeforeTaskRunsDoesNotSubmitAudio() async {
        let runtime = MLXNativeLiveRuntime()
        let session = RuntimeTestStreamingSession()
        var samplesRequested = 0
        runtime.install(session, pollInterval: .seconds(60), nextSamples: {
            samplesRequested += 1
            return [0.1]
        }, shouldContinue: { true }, onEvent: { _ in }, releaseModel: {})
        runtime.stopFeeding()
        runtime.release(cancelSession: true)
        await runtime.waitForRetirement()
        XCTAssertEqual(samplesRequested, 0)
        XCTAssertEqual(session.feedCount, 0)
    }

    func testAbandonedRuntimeCancelsStreamAndReleasesModelUse() async {
        var runtime: MLXNativeLiveRuntime? = MLXNativeLiveRuntime()
        weak var weakRuntime = runtime
        let session = RuntimeTestStreamingSession()
        let released = expectation(description: "abandoned model use released")
        install(session, in: runtime!, release: { released.fulfill() })
        runtime = nil
        XCTAssertNil(weakRuntime)
        await fulfillment(of: [released], timeout: 1)
        XCTAssertEqual(session.cancelCount, 1)
    }

    func testNormalDrainRetirementDoesNotCancelSessionAgain() async {
        let runtime = MLXNativeLiveRuntime()
        let session = RuntimeTestStreamingSession()
        install(session, in: runtime, release: {})
        runtime.stopFeeding()
        runtime.session?.stop()
        runtime.release(cancelSession: false)
        await runtime.waitForRetirement()
        XCTAssertEqual(session.stopCount, 1)
        XCTAssertEqual(session.cancelCount, 0)
    }

    private func install(_ session: RuntimeTestStreamingSession, in runtime: MLXNativeLiveRuntime, release: @escaping @MainActor () -> Void) {
        runtime.install(session, pollInterval: .seconds(60), nextSamples: { [] }, shouldContinue: { false }, onEvent: { _ in }, releaseModel: release)
    }
}

/// Counters are lock protected because the production session contract is nonisolated.
nonisolated private final class RuntimeTestStreamingSession: MLXNativeStreamingSession, @unchecked Sendable {
    let events: AsyncStream<TranscriptionEvent>
    private let continuation: AsyncStream<TranscriptionEvent>.Continuation
    private let lock = NSLock()
    private var cancellations = 0
    private var feeds = 0
    private var stops = 0

    init() {
        let pair = AsyncStream<TranscriptionEvent>.makeStream()
        events = pair.stream
        continuation = pair.continuation
    }

    var cancelCount: Int { lock.lock(); defer { lock.unlock() }; return cancellations }
    var feedCount: Int { lock.lock(); defer { lock.unlock() }; return feeds }
    var stopCount: Int { lock.lock(); defer { lock.unlock() }; return stops }
    func feedAudio(samples: [Float]) { lock.lock(); feeds += 1; lock.unlock() }
    func stop() { lock.lock(); stops += 1; lock.unlock(); continuation.finish() }
    func cancel() { lock.lock(); cancellations += 1; lock.unlock(); continuation.finish() }
    func emit(_ text: String) { continuation.yield(.displayUpdate(confirmedText: text, provisionalText: "")) }
}
