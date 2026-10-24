import XCTest
@testable import Voxt

@MainActor
final class MeetingLiveSessionRegistryTests: XCTestCase {
    func testDrainKeepsFinalCallbacksValidUntilFinishReturns() {
        let registry = MeetingLiveSessionRegistry()
        let session = RegistryTestSession()
        let entry = registry.insert(session, for: .me)
        let draining = registry.beginFinishingAll()
        XCTAssertNil(registry[.me])
        XCTAssertFalse(registry.isEmpty)
        XCTAssertEqual(draining.map(\.token), [entry.token])
        XCTAssertTrue(registry.accepts(.final(segment(speaker: .me)), token: entry.token))
        registry.remove(entry.token)
        XCTAssertFalse(registry.accepts(.final(segment(speaker: .me)), token: entry.token))
        XCTAssertTrue(registry.isEmpty)
    }

    func testReplacementRejectsEveryOldEventWithoutRemovingNewSession() {
        let registry = MeetingLiveSessionRegistry()
        let old = registry.insert(RegistryTestSession(), for: .them)
        _ = registry.beginFinishing(.them)
        let current = registry.insert(RegistryTestSession(), for: .them)
        let events: [MeetingTranscriptEvent] = [
            .partial(segment(speaker: .them)), .final(segment(speaker: .them)),
            .failed(speaker: .them, message: "stale"), .finished(speaker: .them)
        ]
        for event in events {
            XCTAssertFalse(registry.accepts(event, token: old.token))
            XCTAssertTrue(registry.accepts(event, token: current.token))
        }
        registry.remove(old.token)
        XCTAssertTrue(registry.accepts(current.token, for: .them))
        XCTAssertNotNil(registry[.them])
    }

    func testSpeakerIdentityMustMatchToken() {
        let registry = MeetingLiveSessionRegistry()
        let me = registry.insert(RegistryTestSession(), for: .me)
        let them = registry.insert(RegistryTestSession(), for: .them)
        XCTAssertFalse(registry.accepts(.final(segment(speaker: .them)), token: me.token))
        registry.remove(me.token)
        XCTAssertTrue(registry.accepts(them.token, for: .them))
    }

    func testCancelSnapshotIncludesDrainingSessionsAndInvalidatesBeforeAwait() async {
        let registry = MeetingLiveSessionRegistry()
        let session = RegistryTestSession()
        let entry = registry.insert(session, for: .me)
        _ = registry.beginFinishing(.me)
        session.onCancel = {
            XCTAssertFalse(registry.accepts(.finished(speaker: .me), token: entry.token))
        }
        let retired = registry.removeAll()
        XCTAssertEqual(retired.count, 1)
        XCTAssertTrue(registry.isEmpty)
        for entry in retired { await entry.session.cancel() }
        XCTAssertEqual(session.cancelCount, 1)
    }

    func testFinishingOneSpeakerDoesNotStopOtherSpeaker() {
        let registry = MeetingLiveSessionRegistry()
        let me = registry.insert(RegistryTestSession(), for: .me)
        _ = registry.insert(RegistryTestSession(), for: .them)
        XCTAssertEqual(registry.beginFinishing(.me)?.token, me.token)
        XCTAssertNil(registry.beginFinishing(.me))
        XCTAssertNotNil(registry[.them])
        XCTAssertTrue(registry.accepts(me.token, for: .me))
    }

    private func segment(speaker: MeetingSpeaker) -> MeetingTranscriptSegment {
        .init(speaker: speaker, startSeconds: 0, endSeconds: 1, text: "speech")
    }
}

@MainActor
private final class RegistryTestSession: MeetingLiveTranscribingSession {
    var state: MeetingLiveSessionState = .active
    var onCancel: (() -> Void)?
    var cancelCount = 0
    func start(timelineOffsetSeconds: TimeInterval, eventHandler: @escaping @MainActor (MeetingTranscriptEvent) -> Void) async throws {}
    func append(samples: [Float], sampleRate: Double) async {}
    func finish() async {}
    func cancel() async {
        cancelCount += 1
        onCancel?()
    }
}
