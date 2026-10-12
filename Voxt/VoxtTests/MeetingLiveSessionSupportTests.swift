// MeetingLiveSessionSupportTests.swift
// Provides Meeting Live Session Support Tests for Voxt test coverage.

import XCTest
import MLXAudioSTT
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
                STTTranscriptSegment(text: " Hello ", startTime: 0.2, endTime: 1.1),
                STTTranscriptSegment(text: "world", startTime: 1.2, endTime: 2.0),
                STTTranscriptSegment(text: "", startTime: 2.0, endTime: 2.5),
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
                STTTranscriptSegment(text: "chunk", startTime: 0.0, endTime: 1.0)
            ],
            timingGranularity: .chunk,
            modelFamily: .qwen3ASR,
            timelineOffsetSeconds: 0,
            speaker: .them,
            audioSource: .systemAudio
        )
        XCTAssertTrue(mapped.isEmpty)
    }

    func testStructuredLiveFinalizationMapsMOSSSpeakerIDs() {
        let mapped = MeetingNativeLiveStructuredFinalization.meetingSegments(
            from: [
                STTTranscriptSegment(
                    text: "[S01] Hello",
                    startTime: 0.5,
                    endTime: 1.5,
                    speakerID: "S01"
                )
            ],
            timingGranularity: .sentence,
            modelFamily: .mossTranscribeDiarize,
            timelineOffsetSeconds: 0,
            speaker: .them,
            audioSource: .systemAudio
        )

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0].text, "Hello")
        XCTAssertEqual(mapped[0].speakerID, "moss:S01")
        XCTAssertEqual(mapped[0].startSeconds, 0.5, accuracy: 0.0001)
        XCTAssertEqual(mapped[0].endSeconds ?? -1, 1.5, accuracy: 0.0001)
    }

    func testStructuredLiveFinalizationReusesInFlightSegmentIDForFirstItem() {
        let inFlightID = UUID()
        let mapped = MeetingNativeLiveStructuredFinalization.meetingSegments(
            from: [
                STTTranscriptSegment(text: "", startTime: 0.0, endTime: 0.1),
                STTTranscriptSegment(text: "first", startTime: 12.0, endTime: 13.0),
                STTTranscriptSegment(text: "second", startTime: 13.2, endTime: 14.0),
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
