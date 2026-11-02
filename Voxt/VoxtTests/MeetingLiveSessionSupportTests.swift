// MeetingLiveSessionSupportTests.swift
// Provides Meeting Live Session Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class MeetingLiveSessionSupportTests: XCTestCase {
    func testDoubaoPolicyEnablesKeepaliveAndReconnect() {
        let policy = MeetingLiveSessionPolicy.resolved(
            provider: .doubaoASR,
            configuration: .init(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                model: DoubaoASRConfiguration.modelV2,
                endpoint: "",
                apiKey: "",
                appID: "app-id",
                accessToken: "token"
            )
        )

        XCTAssertTrue(policy.idleKeepaliveEnabled)
        XCTAssertEqual(policy.idleKeepaliveInterval, 3.0, accuracy: 0.001)
        XCTAssertEqual(policy.idleKeepaliveFrameDuration, 0.2, accuracy: 0.001)
        XCTAssertTrue(policy.autoReconnectOnUnexpectedClose)
        XCTAssertEqual(policy.prebufferDuration, 1.0, accuracy: 0.001)
        XCTAssertEqual(policy.segmentSilenceSplitThreshold, 1.2, accuracy: 0.001)
    }

    func testAliyunNonRealtimePolicyDisablesKeepaliveAndReconnect() {
        let policy = MeetingLiveSessionPolicy.resolved(
            provider: .aliyunBailianASR,
            configuration: .init(
                providerID: RemoteASRProvider.aliyunBailianASR.rawValue,
                model: "paraformer-v2",
                endpoint: "",
                apiKey: "token"
            )
        )

        XCTAssertFalse(policy.idleKeepaliveEnabled)
        XCTAssertFalse(policy.autoReconnectOnUnexpectedClose)
        XCTAssertEqual(policy.prebufferDuration, 1.0, accuracy: 0.001)
    }

    func testAliyunRealtimePolicyEnablesKeepaliveAndReconnect() {
        let policy = MeetingLiveSessionPolicy.resolved(
            provider: .aliyunBailianASR,
            configuration: .init(
                providerID: RemoteASRProvider.aliyunBailianASR.rawValue,
                model: "fun-asr-realtime",
                endpoint: "",
                apiKey: "token"
            )
        )

        XCTAssertTrue(policy.idleKeepaliveEnabled)
        XCTAssertTrue(policy.autoReconnectOnUnexpectedClose)
        XCTAssertEqual(policy.segmentSilenceSplitThreshold, 1.2, accuracy: 0.001)
    }

    func testOpenAIPolicyDisablesLiveSessionMaintenance() {
        let policy = MeetingLiveSessionPolicy.resolved(
            provider: .openAIWhisper,
            configuration: .init(
                providerID: RemoteASRProvider.openAIWhisper.rawValue,
                model: "gpt-4o-mini-transcribe",
                endpoint: "",
                apiKey: "token",
                openAIChunkPseudoRealtimeEnabled: true
            )
        )

        XCTAssertFalse(policy.idleKeepaliveEnabled)
        XCTAssertFalse(policy.autoReconnectOnUnexpectedClose)
        XCTAssertEqual(policy.prebufferDuration, 0, accuracy: 0.001)
    }

    func testPrebufferKeepsOnlyMostRecentWindow() {
        var prebuffer = MeetingLiveAudioPrebuffer(maxDuration: 1.0)
        prebuffer.append(samples: Array(repeating: 0.1, count: 16_000 / 2), sampleRate: 16_000)
        prebuffer.append(samples: Array(repeating: 0.2, count: 16_000 / 2), sampleRate: 16_000)
        prebuffer.append(samples: Array(repeating: 0.3, count: 16_000 / 2), sampleRate: 16_000)

        let frames = prebuffer.snapshot()
        XCTAssertEqual(frames.count, 2)
        XCTAssertEqual(frames.first?.samples.first ?? 0, 0.2, accuracy: 0.0001)
        XCTAssertEqual(frames.last?.samples.first ?? 0, 0.3, accuracy: 0.0001)
    }

    func testPrebufferDrainPreservesSpeechOnsetAudio() {
        var prebuffer = MeetingLiveAudioPrebuffer(maxDuration: 0.5)
        prebuffer.append(samples: [0.1, 0.2], sampleRate: 100)
        prebuffer.append(samples: [0.3, 0.4], sampleRate: 100)

        let frames = prebuffer.drain()

        XCTAssertEqual(frames.flatMap(\.samples), [0.1, 0.2, 0.3, 0.4])
        XCTAssertTrue(prebuffer.snapshot().isEmpty)
    }

    func testPrebufferDrainReplacesConfirmedSilenceWithoutChangingTimeline() {
        var prebuffer = MeetingLiveAudioPrebuffer(maxDuration: 0.5)
        prebuffer.append(samples: [0.1, 0.2], sampleRate: 100)
        prebuffer.append(samples: [0.3, 0.4], sampleRate: 100)
        let originalDuration = prebuffer.snapshot().reduce(0) { $0 + $1.duration }

        let frames = prebuffer.drain(replacingWithSilence: true)

        XCTAssertEqual(frames.flatMap(\.samples), [0, 0, 0, 0])
        XCTAssertEqual(frames.reduce(0) { $0 + $1.duration }, originalDuration, accuracy: 0.0001)
        XCTAssertTrue(prebuffer.snapshot().isEmpty)
    }

    func testRevisionCapableNativeStreamDefersFinalizationUntilEnded() {
        XCTAssertFalse(
            MeetingNativeLiveSegmentationPolicy.shouldFinalizeBeforeStreamEnd(
                streamCanReviseEarlierText: true,
                silenceDuration: 1.0,
                segmentDuration: 12,
                silenceThreshold: 0.75,
                maximumSegmentDuration: 8
            )
        )
    }

    func testAppendOnlyNativeStreamStillFinalizesAtSilenceOrDurationBoundary() {
        XCTAssertTrue(
            MeetingNativeLiveSegmentationPolicy.shouldFinalizeBeforeStreamEnd(
                streamCanReviseEarlierText: false,
                silenceDuration: 0.75,
                segmentDuration: 2,
                silenceThreshold: 0.75,
                maximumSegmentDuration: 8
            )
        )
        XCTAssertTrue(
            MeetingNativeLiveSegmentationPolicy.shouldFinalizeBeforeStreamEnd(
                streamCanReviseEarlierText: false,
                silenceDuration: 0,
                segmentDuration: 8,
                silenceThreshold: 0.75,
                maximumSegmentDuration: 8
            )
        )
    }

    func testReliableTimingDefersSilenceFinalizationUntilEnded() {
        XCTAssertTrue(
            MeetingNativeLiveSegmentationPolicy.shouldDeferSilenceFinalization(
                timingGranularity: .sentence
            )
        )
        XCTAssertTrue(
            MeetingNativeLiveSegmentationPolicy.shouldDeferSilenceFinalization(
                timingGranularity: .word
            )
        )
        XCTAssertFalse(
            MeetingNativeLiveSegmentationPolicy.shouldDeferSilenceFinalization(
                timingGranularity: .chunk
            )
        )
        XCTAssertFalse(
            MeetingNativeLiveSegmentationPolicy.shouldDeferSilenceFinalization(
                timingGranularity: .none
            )
        )
    }

    func testStructuredLiveFinalizationMapsReliableSegmentsOntoTimeline() {
        let mapped = MeetingNativeLiveStructuredFinalization.meetingSegments(
            from: [
                ASRTranscriptSegment(text: " Hello ", startSeconds: 0.2, endSeconds: 1.1),
                ASRTranscriptSegment(text: "world", startSeconds: 1.2, endSeconds: 2.0),
                ASRTranscriptSegment(text: "", startSeconds: 2.0, endSeconds: 2.5),
            ],
            timingGranularity: .sentence,
            modelFamily: .nemotronASR,
            timelineOffsetSeconds: 10,
            speaker: .me,
            audioSource: .microphone
        )

        XCTAssertEqual(mapped.count, 2)
        XCTAssertEqual(mapped[0].text, "Hello")
        XCTAssertEqual(mapped[0].startSeconds, 10.2, accuracy: 0.0001)
        XCTAssertEqual(mapped[0].endSeconds ?? -1, 11.1, accuracy: 0.0001)
        XCTAssertEqual(mapped[1].text, "world")
        XCTAssertNil(mapped[0].speakerID)
    }

    func testStructuredLiveFinalizationRejectsUnreliableTiming() {
        let mapped = MeetingNativeLiveStructuredFinalization.meetingSegments(
            from: [
                ASRTranscriptSegment(text: "chunk", startSeconds: 0.0, endSeconds: 1.0)
            ],
            timingGranularity: .chunk,
            modelFamily: .qwen3ASR,
            timelineOffsetSeconds: 0,
            speaker: .them,
            audioSource: .systemAudio
        )
        XCTAssertTrue(mapped.isEmpty)
    }

    func testStructuredLiveFinalizationPreservesMOSSTimingSpeakersAndReplacementID() {
        let inFlightID = UUID()
        let mapped = MeetingNativeLiveStructuredFinalization.meetingSegments(
            from: [
                ASRTranscriptSegment(
                    text: "[S01] [breath]",
                    startSeconds: 0.1,
                    endSeconds: 0.3,
                    speakerID: "S01"
                ),
                ASRTranscriptSegment(text: "No speaker", startSeconds: 0.3, endSeconds: 0.4),
                ASRTranscriptSegment(
                    text: "[S01] Hello [sniff]",
                    startSeconds: 0.5,
                    endSeconds: 1.5,
                    speakerID: " S01 "
                ),
                ASRTranscriptSegment(
                    text: "[S02] Hi",
                    startSeconds: 2,
                    endSeconds: 2,
                    speakerID: "S02"
                ),
            ],
            timingGranularity: .sentence,
            modelFamily: .mossTranscribeDiarize,
            timelineOffsetSeconds: 10,
            speaker: .them,
            audioSource: .systemAudio,
            replacingSegmentID: inFlightID
        )

        XCTAssertEqual(mapped.count, 2)
        XCTAssertEqual(mapped[0].id, inFlightID)
        XCTAssertEqual(mapped[0].text, "Hello")
        XCTAssertEqual(mapped[0].speakerID, "moss:S01")
        XCTAssertEqual(mapped[0].startSeconds, 10.5, accuracy: 0.0001)
        XCTAssertEqual(mapped[0].endSeconds ?? -1, 11.5, accuracy: 0.0001)
        XCTAssertEqual(mapped[0].speaker, .them)
        XCTAssertEqual(mapped[0].audioSource, .systemAudio)
        XCTAssertTrue(mapped[0].preventsAdjacentMerge)
        XCTAssertNotEqual(mapped[1].id, inFlightID)
        XCTAssertEqual(mapped[1].text, "Hi")
        XCTAssertEqual(mapped[1].speakerID, "moss:S02")
        XCTAssertEqual(mapped[1].startSeconds, 12)
        XCTAssertEqual(mapped[1].endSeconds, 12)
    }

    func testStructuredLiveFinalizationReusesInFlightSegmentIDForFirstItem() {
        let inFlightID = UUID()
        let mapped = MeetingNativeLiveStructuredFinalization.meetingSegments(
            from: [
                ASRTranscriptSegment(text: "", startSeconds: 0.0, endSeconds: 0.1),
                ASRTranscriptSegment(text: "first", startSeconds: 12.0, endSeconds: 13.0),
                ASRTranscriptSegment(text: "second", startSeconds: 13.2, endSeconds: 14.0),
            ],
            timingGranularity: .sentence,
            modelFamily: .nemotronASR,
            timelineOffsetSeconds: 0,
            speaker: .me,
            audioSource: .microphone,
            replacingSegmentID: inFlightID
        )

        XCTAssertEqual(mapped.count, 2)
        XCTAssertEqual(mapped[0].id, inFlightID)
        XCTAssertEqual(mapped[0].text, "first")
        XCTAssertNotEqual(mapped[1].id, inFlightID)
        XCTAssertEqual(mapped[1].text, "second")
    }

    func testTranscriptStateSubtractsAllPriorFrozenText() {
        var state = MeetingLiveTranscriptState()

        XCTAssertEqual(state.normalizedVisibleText(for: "A"), "A")
        state.freezeCurrentItem(text: "A")

        XCTAssertEqual(state.normalizedVisibleText(for: "AB"), "B")
        state.freezeCurrentItem(text: "B")

        XCTAssertEqual(state.normalizedVisibleText(for: "ABC"), "C")
    }

    func testTranscriptStateIgnoresPunctuationWhenSubtractingFrozenText() {
        var state = MeetingLiveTranscriptState()
        state.freezeCurrentItem(text: "你好。")

        XCTAssertEqual(state.normalizedVisibleText(for: "你好，世界"), "世界")
    }

    func testTranscriptStateReturnsEmptyWhenOnlyFrozenPrefixIsPresent() {
        var state = MeetingLiveTranscriptState()
        state.freezeCurrentItem(text: "hello world")

        XCTAssertEqual(state.normalizedVisibleText(for: "hello world"), "")
    }

    func testTranscriptStateKeepsSpeakersIsolatedByUsingIndependentState() {
        var meState = MeetingLiveTranscriptState()
        var themState = MeetingLiveTranscriptState()

        meState.freezeCurrentItem(text: "me-one")
        themState.freezeCurrentItem(text: "them-one")

        XCTAssertEqual(meState.normalizedVisibleText(for: "me-one me-two"), "me-two")
        XCTAssertEqual(themState.normalizedVisibleText(for: "them-one them-two"), "them-two")
    }

    func testPreventAdjacentMergeBlocksFrozenLiveItemsFromMergingBack() {
        let previous = MeetingTranscriptSegment(
            speaker: .me,
            startSeconds: 0,
            endSeconds: 1.0,
            text: "aaa",
            preventsAdjacentMerge: true
        )
        let next = MeetingTranscriptSegment(
            speaker: .me,
            startSeconds: 1.3,
            endSeconds: 2.0,
            text: "bbb",
            preventsAdjacentMerge: true
        )

        XCTAssertNil(MeetingTranscriptFormatter.mergedAdjacentSegment(previous: previous, next: next))
    }
}
