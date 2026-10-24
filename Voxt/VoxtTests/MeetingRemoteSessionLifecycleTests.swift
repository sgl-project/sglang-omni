import XCTest
@testable import Voxt

@MainActor
final class MeetingRemoteSessionLifecycleTests: XCTestCase {
    func testFinishAcknowledgedDuringSendIsNotLostOrEmittedTwice() async throws {
        let session = FakeMeetingRemoteSession()
        var events: [String] = []
        try await session.start(timelineOffsetSeconds: 0) { events.append(Self.label($0)) }
        await session.handleReadyForAudio()
        session.emitTranscript(text: "partial", isFinal: false)
        session.acknowledgeOnFinish = true
        await session.finish()
        session.signalFinished()
        session.emitTranscript(text: "late", isFinal: true)
        session.emitFailure(URLError(.networkConnectionLost))
        XCTAssertEqual(events, ["partial", "final", "finished"])
        XCTAssertEqual(session.finishSignals, 1)
        XCTAssertEqual(session.closes, [.normalClosure])
    }

    func testTimeoutFlushesPartialOnceAndReleasesAllFinishWaiters() async throws {
        let deadline = ManualMeetingDeadline()
        let waiting = expectation(description: "timeout registered")
        deadline.onWait = { waiting.fulfill() }
        let session = FakeMeetingRemoteSession(finishTimeout: { try await deadline.wait() })
        var events: [String] = []
        try await session.start(timelineOffsetSeconds: 0) { events.append(Self.label($0)) }
        await session.handleReadyForAudio()
        session.emitTranscript(text: "partial", isFinal: false)
        let first = Task { await session.finish() }
        let second = Task { await session.finish() }
        await fulfillment(of: [waiting], timeout: 1)
        deadline.release()
        await first.value
        await second.value
        XCTAssertEqual(events, ["partial", "final", "finished"])
        XCTAssertEqual(session.finishSignals, 1)
        XCTAssertEqual(session.closes.count, 1)
    }

    func testStopBeforeReadyDrainsBufferedAudioBeforeFinishSignal() async throws {
        let deadline = ManualMeetingDeadline()
        let waiting = expectation(description: "finish requested before handshake")
        deadline.onWait = { waiting.fulfill() }
        let session = FakeMeetingRemoteSession(finishTimeout: { try await deadline.wait() })
        try await session.start(timelineOffsetSeconds: 0) { _ in }
        await session.append(samples: [0.1, 0.2], sampleRate: 16_000)
        await session.append(samples: [0.3], sampleRate: 16_000)
        let finishing = Task { await session.finish() }
        await fulfillment(of: [waiting], timeout: 1)
        session.acknowledgeOnFinish = true
        await session.handleReadyForAudio()
        await finishing.value
        XCTAssertEqual(session.operations, ["audio", "audio", "finish", "close"])
    }

    func testCancelDuringBufferedDrainDoesNotSendRemainingPackets() async throws {
        let session = FakeMeetingRemoteSession()
        var events: [String] = []
        try await session.start(timelineOffsetSeconds: 0) { events.append(Self.label($0)) }
        await session.append(samples: [0.1], sampleRate: 16_000)
        await session.append(samples: [0.2], sampleRate: 16_000)
        session.cancelOnAudio = true
        await session.handleReadyForAudio()
        await session.append(samples: [0.3], sampleRate: 16_000)
        session.emitTranscript(text: "stale", isFinal: false)
        XCTAssertEqual(session.operations, ["audio", "close"])
        XCTAssertEqual(events, ["finished"])
        XCTAssertTrue(session.pendingAudioPackets.isEmpty)
    }

    func testSendFailureRetainsPartialBeforeFailureAndClosesTransport() async throws {
        let session = FakeMeetingRemoteSession()
        var events: [String] = []
        try await session.start(timelineOffsetSeconds: 0) { events.append(Self.label($0)) }
        await session.handleReadyForAudio()
        session.emitTranscript(text: "useful partial", isFinal: false)
        session.failOnAudio = true
        await session.append(samples: [0.2], sampleRate: 16_000)
        session.emitFailure(URLError(.timedOut))
        XCTAssertEqual(events, ["partial", "final", "failed", "finished"])
        XCTAssertEqual(session.state, .failed)
        XCTAssertEqual(session.closes.count, 1)
    }

    func testHandshakeFailureClosesTransportAndRethrows() async {
        let session = FakeMeetingRemoteSession()
        session.failOnOpen = true
        do {
            try await session.start(timelineOffsetSeconds: 0) { _ in }
            XCTFail("Expected handshake failure")
        } catch {
            XCTAssertEqual((error as NSError).code, NSURLErrorCannotConnectToHost)
        }
        XCTAssertEqual(session.state, .failed)
        XCTAssertEqual(session.closes, [.goingAway])
    }

    func testCancellingFinishWaitDoesNotPromotePartialToFinal() async throws {
        let deadline = ManualMeetingDeadline()
        let waiting = expectation(description: "finish wait started")
        deadline.onWait = { waiting.fulfill() }
        let session = FakeMeetingRemoteSession(finishTimeout: { try await deadline.wait() })
        var events: [String] = []
        try await session.start(timelineOffsetSeconds: 0) { events.append(Self.label($0)) }
        await session.handleReadyForAudio()
        session.emitTranscript(text: "cancelled partial", isFinal: false)
        let finishing = Task { await session.finish() }
        await fulfillment(of: [waiting], timeout: 1)
        finishing.cancel()
        await finishing.value
        session.signalFinished()
        XCTAssertEqual(events, ["partial", "finished"])
        XCTAssertEqual(session.closes, [.goingAway])
    }

    func testProviderFinalBeforeFinishCallStillClosesExactlyOnce() async throws {
        let session = FakeMeetingRemoteSession()
        var events: [String] = []
        try await session.start(timelineOffsetSeconds: 0) { events.append(Self.label($0)) }
        session.emitProviderPacket(MeetingLiveProviderPacket(
            units: [], fallbackText: "final answer", isFinal: true, sequence: -1
        ))
        await session.finish()
        await session.cancel()
        XCTAssertEqual(events, ["final", "finished"])
        XCTAssertEqual(session.finishSignals, 0)
        XCTAssertEqual(session.closes.count, 1)
    }

    private static func label(_ event: MeetingTranscriptEvent) -> String {
        switch event {
        case .partial: return "partial"
        case .final: return "final"
        case .failed: return "failed"
        case .finished: return "finished"
        }
    }
}

@MainActor
private final class FakeMeetingRemoteSession: BaseMeetingRemoteLiveSession {
    var acknowledgeOnFinish = false
    var cancelOnAudio = false
    var failOnAudio = false
    var failOnOpen = false
    var finishSignals = 0
    var operations: [String] = []
    var closes: [URLSessionWebSocketTask.CloseCode] = []

    init(finishTimeout: @escaping @MainActor @Sendable () async throws -> Void = {
        try await Task.sleep(for: .seconds(30))
    }) {
        super.init(
            speaker: .me,
            configuration: .init(providerID: "test", model: "test", endpoint: "", apiKey: ""),
            hintPayload: .init(language: nil, chineseOutputVariant: nil, prompt: nil),
            speechThreshold: 0.01,
            timelineOffsetSeconds: 0,
            policy: .init(
                idleKeepaliveEnabled: false, idleKeepaliveInterval: 0,
                idleKeepaliveFrameDuration: 0, providerIdleTimeoutSafetyWindow: 0,
                autoReconnectOnUnexpectedClose: false, prebufferDuration: 0,
                segmentSilenceSplitThreshold: 0
            ),
            finishTimeout: finishTimeout
        )
    }

    override func openTransport() async throws {
        if failOnOpen { throw URLError(.cannotConnectToHost) }
    }

    override func sendAudioPacket(_ pcmData: Data, isLast: Bool) async {
        operations.append("audio")
        if cancelOnAudio { await cancel() }
        if failOnAudio { emitFailure(URLError(.networkConnectionLost)) }
    }

    override func sendFinishSignal() async {
        finishSignals += 1
        operations.append("finish")
        if acknowledgeOnFinish { signalFinished() }
    }

    override func cancelTransport(closeCode: URLSessionWebSocketTask.CloseCode) {
        operations.append("close")
        closes.append(closeCode)
        super.cancelTransport(closeCode: closeCode)
    }
}

@MainActor
private final class ManualMeetingDeadline {
    var onWait: (() -> Void)?
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async throws {
        try Task.checkCancellation()
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                self.continuation = continuation
                onWait?()
                if Task.isCancelled { release() }
            }
        } onCancel: {
            Task { @MainActor in self.release() }
        }
        try Task.checkCancellation()
    }

    func release() {
        let pending = continuation
        continuation = nil
        pending?.resume()
    }
}
