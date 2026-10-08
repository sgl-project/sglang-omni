// MeetingSpeakerTranscriptAssemblerTests.swift
// Provides Meeting Speaker Transcript Assembler Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class MeetingSpeakerTranscriptAssemblerTests: XCTestCase {
    func testSpeakerDisplayNameFormatterUsesInterfaceLanguagePreference() {
        let suiteName = "MeetingSpeakerTranscriptAssemblerTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }

        defaults.set("en", forKey: "interfaceLanguage")
        XCTAssertEqual(MeetingSpeakerDisplayNameFormatter.displayName(ordinal: 3, defaults: defaults), "Speaker 3")

        defaults.set("zh-Hans", forKey: "interfaceLanguage")
        XCTAssertEqual(MeetingSpeakerDisplayNameFormatter.displayName(ordinal: 3, defaults: defaults), "发言人 3")

        defaults.set("ja", forKey: "interfaceLanguage")
        XCTAssertEqual(MeetingSpeakerDisplayNameFormatter.displayName(ordinal: 3, defaults: defaults), "話者 3")
    }

    func testDominantSpeakerTurnAnnotatesTranscriptSegment() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 0,
            endSeconds: 4,
            text: "We should ship this week."
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 4,
                confidence: 0.91
            )
        ]

        let result = MeetingSpeakerTranscriptAssembler.assemble(segments: [segment], speakerTurns: turns)

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].speaker, .them)
        XCTAssertEqual(result[0].audioSource, .systemAudio)
        XCTAssertEqual(result[0].speakerID, "S1")
        XCTAssertEqual(result[0].displaySpeakerTitle, "Speaker 1")
        XCTAssertEqual(result[0].speakerConfidence, 0.91)
    }

    func testSegmentCrossingSpeakerTurnsKeepsTextTogetherByDefault() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 0,
            endSeconds: 4,
            text: "alpha beta gamma delta"
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 2
            ),
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S2",
                displayName: "Speaker 2",
                startSeconds: 2,
                endSeconds: 4
            )
        ]

        let result = MeetingSpeakerTranscriptAssembler.assemble(segments: [segment], speakerTurns: turns)

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].speakerID, "S1")
        XCTAssertEqual(result[0].text, "alpha beta gamma delta")
        XCTAssertEqual(result[0].startSeconds, 0, accuracy: 0.001)
        XCTAssertEqual(result[0].endSeconds ?? -1, 4, accuracy: 0.001)
    }

    func testSegmentCrossingSpeakerTurnsCanSplitTextAndTiming() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 0,
            endSeconds: 4,
            text: "alpha beta gamma delta"
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 2
            ),
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S2",
                displayName: "Speaker 2",
                startSeconds: 2,
                endSeconds: 4
            )
        ]

        let result = MeetingSpeakerTranscriptAssembler.assemble(
            segments: [segment],
            speakerTurns: turns,
            options: MeetingSpeakerTranscriptAssembler.Options(
                dominantSpeakerOverlapRatio: 0.9,
                splitsSegmentsOnSpeakerBoundaries: true
            )
        )

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].speakerID, "S1")
        XCTAssertEqual(result[0].text, "alpha beta")
        XCTAssertEqual(result[0].startSeconds, 0, accuracy: 0.001)
        XCTAssertEqual(result[0].endSeconds ?? -1, 2, accuracy: 0.001)
        XCTAssertEqual(result[1].speakerID, "S2")
        XCTAssertEqual(result[1].text, "gamma delta")
        XCTAssertEqual(result[1].startSeconds, 2, accuracy: 0.001)
        XCTAssertEqual(result[1].endSeconds ?? -1, 4, accuracy: 0.001)
    }

    func testSameSpeakerTurnsWithinSegmentDoNotSplitText() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 0,
            endSeconds: 8,
            text: "one two three four five six seven eight"
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 2
            ),
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 2.7,
                endSeconds: 5
            ),
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 5.4,
                endSeconds: 8
            )
        ]

        let result = MeetingSpeakerTranscriptAssembler.assemble(
            segments: [segment],
            speakerTurns: turns,
            options: MeetingSpeakerTranscriptAssembler.Options(
                dominantSpeakerOverlapRatio: 0.9,
                splitsSegmentsOnSpeakerBoundaries: true
            )
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].speakerID, "S1")
        XCTAssertEqual(result[0].text, "one two three four five six seven eight")
    }

    func testSegmentCrossingSpeakerTurnsSplitsTextProportionallyByTurnDuration() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 0,
            endSeconds: 10,
            text: "one two three four five six seven eight nine ten"
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 7
            ),
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S2",
                displayName: "Speaker 2",
                startSeconds: 7,
                endSeconds: 10
            )
        ]

        let result = MeetingSpeakerTranscriptAssembler.assemble(
            segments: [segment],
            speakerTurns: turns,
            options: MeetingSpeakerTranscriptAssembler.Options(
                dominantSpeakerOverlapRatio: 0.9,
                splitsSegmentsOnSpeakerBoundaries: true
            )
        )

        XCTAssertEqual(result.count, 2)
        guard result.count == 2 else { return }
        XCTAssertEqual(result[0].speakerID, "S1")
        XCTAssertEqual(result[0].text, "one two three four five six seven")
        XCTAssertEqual(result[1].speakerID, "S2")
        XCTAssertEqual(result[1].text, "eight nine ten")
    }

    func testDominantSpeakerDoesNotHideMeaningfulSecondarySpeakerByDefault() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 20,
            text: "第一位发言人先介绍项目背景和整体方案。第二位发言人补充一个关键风险。"
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 18
            ),
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S2",
                displayName: "Speaker 2",
                startSeconds: 18,
                endSeconds: 20
            )
        ]

        let result = MeetingSpeakerTranscriptAssembler.assemble(
            segments: [segment],
            speakerTurns: turns,
            options: MeetingSpeakerTranscriptAssembler.Options(
                dominantSpeakerOverlapRatio: 0.9,
                splitsSegmentsOnSpeakerBoundaries: true
            )
        )

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].speakerID, "S1")
        XCTAssertEqual(result[1].speakerID, "S2")
        XCTAssertEqual(result[0].startSeconds, 0, accuracy: 0.001)
        XCTAssertEqual(result[1].startSeconds, 18, accuracy: 0.001)
        XCTAssertFalse(result[0].text.isEmpty)
        XCTAssertFalse(result[1].text.isEmpty)
        XCTAssertEqual(result.map(\.text).joined(), segment.text)
    }

    func testShortSecondarySpeakerBlipStillUsesDominantSpeaker() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 8,
            text: "This sentence should stay with the main speaker."
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 7.7
            ),
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S2",
                displayName: "Speaker 2",
                startSeconds: 7.7,
                endSeconds: 8
            )
        ]

        let result = MeetingSpeakerTranscriptAssembler.assemble(
            segments: [segment],
            speakerTurns: turns,
            options: MeetingSpeakerTranscriptAssembler.Options(
                dominantSpeakerOverlapRatio: 0.9,
                splitsSegmentsOnSpeakerBoundaries: true
            )
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].speakerID, "S1")
    }

    func testSpeakerAnalysisPipelineSplitsMultiSpeakerFinalSegmentByDefault() async {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 4,
            text: "alpha beta gamma delta"
        )
        let asset = MeetingAudioAsset(
            source: .systemAudio,
            samples: [Float](repeating: 0.1, count: 400),
            sampleRate: 100,
            sessionStartOffset: 0
        )
        let engine = StubMeetingSpeakerDiarizationEngine(turns: [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 2
            ),
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S2",
                displayName: "Speaker 2",
                startSeconds: 2,
                endSeconds: 4
            )
        ])

        let result = await MeetingSpeakerAnalysisPipeline.analyzedSegments(
            from: [segment],
            assets: [asset],
            engine: engine
        )

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].speakerID, "S1")
        XCTAssertEqual(result[0].text, "alpha beta")
        XCTAssertEqual(result[1].speakerID, "S2")
        XCTAssertEqual(result[1].text, "gamma delta")
    }

    func testSpeakerAnalysisPipelineDoesNotCollapseRawSpeakersWhenConfidenceFilterIsTooStrict() async {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 6,
            text: "alpha beta gamma delta epsilon zeta"
        )
        let asset = MeetingAudioAsset(
            source: .systemAudio,
            samples: [Float](repeating: 0.1, count: 600),
            sampleRate: 100,
            sessionStartOffset: 0
        )
        let engine = StubMeetingSpeakerDiarizationEngine(turns: [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 3,
                confidence: 0.55
            ),
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S2",
                displayName: "Speaker 2",
                startSeconds: 3,
                endSeconds: 6,
                confidence: 0.12
            )
        ])

        let result = await MeetingSpeakerAnalysisPipeline.analyzedSegments(
            from: [segment],
            assets: [asset],
            options: MeetingSpeakerDiarizationOptions(minimumSpeakerConfidence: 0.28),
            engine: engine
        )

        XCTAssertEqual(Set(result.compactMap(\.speakerID)), Set(["S1", "S2"]))
    }

    func testSpeakerAnalysisPipelinePreservesMOSSDataAndAnalyzesRemainingSegments() async {
        let mossSegment = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "moss:S01",
            speakerDisplayName: "MOSS Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 2,
            text: "structured segment",
            preventsAdjacentMerge: true
        )
        let unstructuredSegment = MeetingTranscriptSegment(
            speaker: .them,
            audioSource: .systemAudio,
            startSeconds: 2,
            endSeconds: 4,
            text: "segment needing analysis"
        )
        let asset = MeetingAudioAsset(
            source: .systemAudio,
            samples: [Float](repeating: 0.1, count: 400),
            sampleRate: 100,
            sessionStartOffset: 0
        )
        let engine = StubMeetingSpeakerDiarizationEngine(turns: [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "raw-speaker",
                displayName: "Raw Speaker",
                startSeconds: 0,
                endSeconds: 4
            )
        ])

        let result = await MeetingSpeakerAnalysisPipeline.analyzedSegmentsPreservingStructuredSpeakerData(
            from: [mossSegment, unstructuredSegment],
            assets: [asset],
            engine: engine
        )

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].speakerID, "moss:S01")
        XCTAssertEqual(result[0].speakerDisplayName, "MOSS Speaker 1")
        XCTAssertEqual(result[0].text, "structured segment")
        XCTAssertEqual(result[1].speakerID, "raw-speaker")
        XCTAssertEqual(result[1].text, "segment needing analysis")
    }

    @MainActor
    func testDescriptorAnalysisRelabelsStructuredSegmentsAcrossWindows() async {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "moss:S01",
            speakerDisplayName: "MOSS Speaker 1",
            audioSource: .mixed,
            startSeconds: 0,
            endSeconds: 4,
            text: "alpha beta gamma delta",
            preventsAdjacentMerge: true
        )
        let descriptors = [
            MeetingAudioAssetDescriptor(
                source: .mixed,
                sampleRate: 100,
                startSample: 0,
                sampleCount: 200
            ),
            MeetingAudioAssetDescriptor(
                source: .mixed,
                sampleRate: 100,
                startSample: 200,
                sampleCount: 200
            ),
        ]
        let continuousAudioURL = URL(fileURLWithPath: "/tmp/continuous-meeting.wav")
        let engine = StubMeetingSpeakerDiarizationEngine(
            turns: [
                MeetingSpeakerTurn(
                    source: .mixed,
                    speakerID: "session-speaker",
                    displayName: "Speaker 1",
                    startSeconds: 0,
                    endSeconds: 4
                )
            ]
        )

        let result = await MeetingSpeakerAnalysisPipeline.analyzedSegments(
            from: [segment],
            descriptors: descriptors,
            loadAsset: { descriptor in
                MeetingAudioAsset(
                    source: descriptor.source,
                    samples: [Float](repeating: 0.1, count: descriptor.sampleCount),
                    sampleRate: descriptor.sampleRate,
                    sessionStartOffset: descriptor.sessionStartOffset
                )
            },
            continuousAudioURL: continuousAudioURL,
            engine: engine
        )

        XCTAssertEqual(result.map(\.speakerID), ["session-speaker"])
    }

    func testSegmentsWithoutMatchingTurnsKeepOriginalSpeakerAndSource() {
        let segment = MeetingTranscriptSegment(
            speaker: .me,
            startSeconds: 0,
            endSeconds: 2,
            text: "hello"
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 2
            )
        ]

        let result = MeetingSpeakerTranscriptAssembler.assemble(segments: [segment], speakerTurns: turns)

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].speaker, .me)
        XCTAssertNil(result[0].speakerID)
        XCTAssertEqual(result[0].audioSource, .microphone)
        XCTAssertEqual(result[0].displaySpeakerTitle, "Me")
    }

    func testMixedSpeakerTurnsApplyAcrossSourcesWhilePreservingAudioSource() {
        let microphoneSegment = MeetingTranscriptSegment(
            speaker: .me,
            audioSource: .microphone,
            startSeconds: 0,
            endSeconds: 2,
            text: "I can start with context."
        )
        let systemSegment = MeetingTranscriptSegment(
            speaker: .them,
            audioSource: .systemAudio,
            startSeconds: 2,
            endSeconds: 4,
            text: "Then I will add the customer impact."
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .mixed,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 4
            )
        ]

        let result = MeetingSpeakerTranscriptAssembler.assemble(
            segments: [microphoneSegment, systemSegment],
            speakerTurns: turns
        )

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result.map(\.speakerID), ["S1", "S1"])
        XCTAssertEqual(result.map(\.displaySpeakerTitle), ["Speaker 1", "Speaker 1"])
        XCTAssertEqual(result[0].speaker, .me)
        XCTAssertEqual(result[0].audioSource, .microphone)
        XCTAssertEqual(result[1].speaker, .them)
        XCTAssertEqual(result[1].audioSource, .systemAudio)
    }

    func testNearbyMixedSpeakerTurnLabelsSmallAsrTimingGap() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            audioSource: .systemAudio,
            startSeconds: 10.15,
            endSeconds: 12,
            text: "This line starts just after the diarized turn."
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .mixed,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 8,
                endSeconds: 10
            )
        ]

        let result = MeetingSpeakerTranscriptAssembler.assemble(
            segments: [segment],
            speakerTurns: turns
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].speakerID, "S1")
        XCTAssertEqual(result[0].displaySpeakerTitle, "Speaker 1")
        XCTAssertEqual(result[0].audioSource, .systemAudio)
    }

    func testDistantMixedSpeakerTurnDoesNotLabelUnmatchedSegment() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            audioSource: .systemAudio,
            startSeconds: 20,
            endSeconds: 22,
            text: "This line is too far away from speaker evidence."
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .mixed,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 8,
                endSeconds: 10
            )
        ]

        let result = MeetingSpeakerTranscriptAssembler.assemble(
            segments: [segment],
            speakerTurns: turns
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertNil(result[0].speakerID)
        XCTAssertEqual(result[0].displaySpeakerTitle, "Them")
    }

    func testSessionResultPersistsAnalyzedSegmentsWithoutSnapshotDuplicates() {
        let analyzedSegment = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 2,
            text: "We can move less-used data into CPU memory."
        )
        let snapshotSegment = MeetingTranscriptSegment(
            speaker: .them,
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 2,
            text: "We can move less-used data into CPU memory."
        )
        let result = MeetingSessionResult(
            captureMode: .meeting,
            transcriptionEngine: .remote,
            transcriptionModelDescription: "Qwen",
            segments: [analyzedSegment],
            visibleSnapshotSegments: [snapshotSegment],
            audioDurationSeconds: 2,
            archivedAudioURL: nil
        )

        XCTAssertEqual(result.persistedSegments.count, 1)
        XCTAssertEqual(result.persistedSegments.first?.speakerID, "S1")
        XCTAssertEqual(result.persistedSegments.first?.displaySpeakerTitle, "Speaker 1")
        XCTAssertEqual(
            result.combinedText,
            "00:00 Speaker 1 We can move less-used data into CPU memory."
        )
    }

    func testSpeakerTurnSmootherAbsorbsShortIsolatedTurnBetweenSameSpeaker() {
        let turns = [
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 3
            ),
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S2",
                displayName: "Speaker 2",
                startSeconds: 3,
                endSeconds: 3.3
            ),
            MeetingSpeakerTurn(
                source: .systemAudio,
                speakerID: "S1",
                displayName: "Speaker 1",
                startSeconds: 3.3,
                endSeconds: 6
            )
        ]

        let smoothed = MeetingSpeakerTurnSmoother.smooth(turns)

        XCTAssertEqual(smoothed.count, 1)
        XCTAssertEqual(smoothed[0].speakerID, "S1")
        XCTAssertEqual(smoothed[0].startSeconds, 0, accuracy: 0.001)
        XCTAssertEqual(smoothed[0].endSeconds, 6, accuracy: 0.001)
    }
}

private struct StubMeetingSpeakerDiarizationEngine: MeetingSpeakerDiarizationEngine {
    let turns: [MeetingSpeakerTurn]

    func diarize(
        asset: MeetingAudioAsset,
        options: MeetingSpeakerDiarizationOptions
    ) async throws -> [MeetingSpeakerTurn] {
        turns
    }
}
