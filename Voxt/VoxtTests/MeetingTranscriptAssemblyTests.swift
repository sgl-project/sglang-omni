// MeetingTranscriptAssemblyTests.swift
// Provides Meeting Transcript Assembly Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class MeetingTranscriptAssemblyTests: XCTestCase {
    func testPartialThenFinalReusesSegmentID() {
        let id = UUID()
        let partial = MeetingTranscriptSegment(
            id: id,
            speaker: .them,
            startSeconds: 2,
            endSeconds: 2.5,
            text: "hello"
        )
        let final = MeetingTranscriptSegment(
            id: id,
            speaker: .them,
            startSeconds: 2,
            endSeconds: 3,
            text: "hello world"
        )

        let partialResult = MeetingTranscriptAssembler.apply(.partial(partial), to: [])
        let finalResult = MeetingTranscriptAssembler.apply(.final(final), to: partialResult.segments)

        XCTAssertEqual(partialResult.segments.count, 1)
        XCTAssertEqual(finalResult.segments.count, 1)
        XCTAssertEqual(finalResult.segments[0].id, id)
        XCTAssertEqual(finalResult.segments[0].text, "hello world")
        XCTAssertEqual(finalResult.finalizedSegmentID, id)
    }

    func testStructuredEndedFinalsReplaceInFlightPartialWithoutDuplicatingText() {
        let inFlightID = UUID()
        let partial = MeetingTranscriptSegment(
            id: inFlightID,
            speaker: .me,
            audioSource: .microphone,
            startSeconds: 12.0,
            endSeconds: 13.5,
            text: "呃，现在看起来。",
            preventsAdjacentMerge: true
        )
        // Mimic MeetingNativeLiveStructuredFinalization: first final reuses the
        // in-flight partial ID; additional structured items get fresh IDs.
        let structured = [
            MeetingTranscriptSegment(
                id: inFlightID,
                speaker: .me,
                audioSource: .microphone,
                startSeconds: 12.1,
                endSeconds: 13.4,
                text: "呃，现在看起来。",
                preventsAdjacentMerge: true
            ),
            MeetingTranscriptSegment(
                id: UUID(),
                speaker: .me,
                audioSource: .microphone,
                startSeconds: 15.0,
                endSeconds: 16.2,
                text: "会出现重复的问题",
                preventsAdjacentMerge: true
            ),
        ]

        var segments = MeetingTranscriptAssembler.apply(.partial(partial), to: []).segments
        for item in structured {
            segments = MeetingTranscriptAssembler.apply(.final(item), to: segments).segments
        }

        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0].id, inFlightID)
        XCTAssertEqual(segments[0].text, "呃，现在看起来。")
        XCTAssertEqual(segments[0].startSeconds, 12.1, accuracy: 0.0001)
        XCTAssertEqual(segments[1].text, "会出现重复的问题")
        XCTAssertEqual(
            segments.filter { $0.text == "呃，现在看起来。" }.count,
            1,
            "In-flight partial must be replaced, not left beside structured finals"
        )
    }

    func testFinalUpsertAdoptsIncomingStartSeconds() {
        let id = UUID()
        let partial = MeetingTranscriptSegment(
            id: id,
            speaker: .me,
            startSeconds: 10.0,
            endSeconds: 11.0,
            text: "hello"
        )
        let final = MeetingTranscriptSegment(
            id: id,
            speaker: .me,
            startSeconds: 10.4,
            endSeconds: 11.2,
            text: "hello world"
        )

        let afterPartial = MeetingTranscriptAssembler.apply(.partial(partial), to: [])
        let afterFinal = MeetingTranscriptAssembler.apply(.final(final), to: afterPartial.segments)

        XCTAssertEqual(afterFinal.segments.count, 1)
        XCTAssertEqual(afterFinal.segments[0].startSeconds, 10.4, accuracy: 0.0001)
        XCTAssertEqual(afterFinal.segments[0].endSeconds ?? -1, 11.2, accuracy: 0.0001)
    }

    func testFinalSegmentsMergeWithinTwoSecondsForSameSpeaker() {
        let first = MeetingTranscriptSegment(
            id: UUID(),
            speaker: .me,
            startSeconds: 1,
            endSeconds: 2,
            text: "hello"
        )
        let second = MeetingTranscriptSegment(
            id: UUID(),
            speaker: .me,
            startSeconds: 3.2,
            endSeconds: 4.1,
            text: "world"
        )

        let firstResult = MeetingTranscriptAssembler.apply(.final(first), to: [])
        let secondResult = MeetingTranscriptAssembler.apply(.final(second), to: firstResult.segments)

        XCTAssertEqual(secondResult.segments.count, 1)
        XCTAssertEqual(secondResult.segments[0].text, "hello world")
        XCTAssertEqual(Set(secondResult.supersededSegmentIDs), Set([first.id, second.id]))
        XCTAssertEqual(secondResult.finalizedSegmentID, first.id)
    }

    func testDifferentSpeakersDoNotMerge() {
        let first = MeetingTranscriptSegment(
            id: UUID(),
            speaker: .me,
            startSeconds: 1,
            endSeconds: 2,
            text: "hello"
        )
        let second = MeetingTranscriptSegment(
            id: UUID(),
            speaker: .them,
            startSeconds: 2.5,
            endSeconds: 3,
            text: "world"
        )

        let firstResult = MeetingTranscriptAssembler.apply(.final(first), to: [])
        let secondResult = MeetingTranscriptAssembler.apply(.final(second), to: firstResult.segments)

        XCTAssertEqual(secondResult.segments.count, 2)
        XCTAssertTrue(secondResult.supersededSegmentIDs.isEmpty)
    }

    func testUpdatedSegmentPreservesExistingTranslationWhileRefreshIsPending() {
        let id = UUID()
        let existing = MeetingTranscriptSegment(
            id: id,
            speaker: .them,
            startSeconds: 2,
            endSeconds: 4,
            text: "hello there",
            translatedText: "你好",
            isTranslationPending: false
        )
        let updated = MeetingTranscriptSegment(
            id: id,
            speaker: .them,
            startSeconds: 2,
            endSeconds: 5,
            text: "hello there again"
        )

        let result = MeetingTranscriptAssembler.apply(.final(updated), to: [existing])

        XCTAssertEqual(result.segments.count, 1)
        XCTAssertEqual(result.segments[0].translatedText, "你好")
        XCTAssertTrue(result.segments[0].isTranslationPending)
    }

    func testUpdatedSegmentPreservesSpeakerAnalysisMetadata() {
        let id = UUID()
        let existing = MeetingTranscriptSegment(
            id: id,
            speaker: .them,
            speakerID: "speaker_a",
            speakerDisplayName: "Speaker 1",
            audioSource: .mixed,
            speakerConfidence: 0.72,
            startSeconds: 2,
            endSeconds: 4,
            text: "hello there"
        )
        let updated = MeetingTranscriptSegment(
            id: id,
            speaker: .them,
            startSeconds: 2,
            endSeconds: 5,
            text: "hello there again"
        )

        let result = MeetingTranscriptAssembler.apply(.final(updated), to: [existing])

        XCTAssertEqual(result.segments.count, 1)
        XCTAssertEqual(result.segments[0].speakerID, "speaker_a")
        XCTAssertEqual(result.segments[0].speakerDisplayName, "Speaker 1")
        XCTAssertEqual(result.segments[0].audioSource, .mixed)
        XCTAssertEqual(result.segments[0].speakerConfidence ?? -1, 0.72, accuracy: 0.001)
    }

    func testUpdatedSegmentDoesNotEnterPendingStateWithoutExistingTranslation() {
        let id = UUID()
        let existing = MeetingTranscriptSegment(
            id: id,
            speaker: .them,
            startSeconds: 2,
            endSeconds: 4,
            text: "hello there",
            translatedText: nil,
            isTranslationPending: false
        )
        let updated = MeetingTranscriptSegment(
            id: id,
            speaker: .them,
            startSeconds: 2,
            endSeconds: 5,
            text: "hello there again"
        )

        let result = MeetingTranscriptAssembler.apply(.final(updated), to: [existing])

        XCTAssertEqual(result.segments.count, 1)
        XCTAssertNil(result.segments[0].translatedText)
        XCTAssertFalse(result.segments[0].isTranslationPending)
    }

    func testFinalTextPostProcessorCollapsesSpacedAcronyms() {
        let text = MeetingTranscriptTextPostProcessor.normalizedFinalText("放进 C P U，然后做 R L 微调。")

        XCTAssertEqual(text, "放进 CPU，然后做 RL 微调。")
    }

    func testFinalPostProcessorMergesOverlappingAdjacentSegments() {
        let first = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 10,
            text: "We should move data into CPU memory"
        )
        let second = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 9.2,
            endSeconds: 18,
            text: "CPU memory before running RL"
        )

        let result = MeetingTranscriptPostProcessor.process([first, second])
        let combinedText = result.map(\.text).joined(separator: " ")

        XCTAssertGreaterThanOrEqual(result.count, 1)
        XCTAssertEqual(combinedText, "We should move data into CPU memory before running RL")
        XCTAssertEqual(result.first?.startSeconds ?? -1, 0, accuracy: 0.001)
        XCTAssertEqual(result.last?.endSeconds ?? -1, 18, accuracy: 0.001)
        XCTAssertEqual(result.count, 1)
        XCTAssertFalse(result[0].preventsAdjacentMerge)
    }

    func testFinalPostProcessorDoesNotChainLongOverlappingChunksWithoutTextOverlap() {
        let first = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 16,
            text: "We discussed the project background and model engine choices."
        )
        let second = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 15,
            endSeconds: 31,
            text: "The next part focused on safety and deployment tradeoffs."
        )

        let result = MeetingTranscriptPostProcessor.process([first, second])

        XCTAssertGreaterThanOrEqual(result.count, 2)
        XCTAssertEqual(result.map(\.text).joined(separator: " "), "\(first.text) \(second.text)")
        XCTAssertTrue(result.contains { $0.startSeconds >= 15 })
        XCTAssertTrue(result.allSatisfy { !$0.text.contains("backgr ound") })
    }

    func testFinalPostProcessorHonorsPreventAdjacentMergeBoundaries() {
        let first = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 3,
            text: "first turn",
            preventsAdjacentMerge: true
        )
        let second = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 3.1,
            endSeconds: 5,
            text: "second turn"
        )

        let result = MeetingTranscriptPostProcessor.process([first, second])

        XCTAssertEqual(result.count, 2)
    }

    func testFinalPostProcessorSplitsLongSingleSegmentIntoReadableChunks() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 60,
            text: "第一段内容主要介绍项目背景和模型选择，大家先讨论了系统音频和麦克风的采集。第二段内容继续讨论发言人识别和时间对齐，需要避免所有内容都挤在一个段落里。第三段内容强调即使只有一个发言人结果，也要按照阅读节奏拆成更短的段落。"
        )

        let result = MeetingTranscriptPostProcessor.process(
            [segment],
            options: MeetingTranscriptPostProcessor.Options(maxSegmentTextCharacters: 45)
        )

        XCTAssertGreaterThan(result.count, 1)
        XCTAssertEqual(result.first?.id, segment.id)
        XCTAssertTrue(result.allSatisfy(\.preventsAdjacentMerge))
        XCTAssertTrue(result.allSatisfy { $0.text.count <= 45 })
        XCTAssertEqual(result.first?.startSeconds ?? -1, 0, accuracy: 0.001)
        XCTAssertEqual(result.last?.endSeconds ?? -1, 60, accuracy: 0.001)
    }

    func testLiveOverlayPostProcessorSplitsLongSegmentMoreAggressivelyThanDetailWithoutOverSplitting() {
        let text = "第一句介绍当前模型选择和采集链路。第二句继续说明发言人识别的误差来源，以及短暂停顿不应该直接生成新的发言人。第三句讨论会议详情应该保持连续语义，同时实时浮层需要更快地给出可读段落和清晰反馈。第四句补充说明如果文本继续变长，实时显示仍然需要在自然句边界拆开，避免一整块内容压在同一个气泡里。"
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 24,
            text: text
        )

        let detailResult = MeetingTranscriptPostProcessor.process([segment])
        let overlayResult = MeetingTranscriptPostProcessor.process(
            [segment],
            options: .liveOverlay
        )

        XCTAssertEqual(detailResult.count, 1)
        XCTAssertGreaterThan(overlayResult.count, 1)
        XCTAssertTrue(overlayResult.allSatisfy(\.preventsAdjacentMerge))
        XCTAssertTrue(overlayResult.allSatisfy { $0.text.count <= 130 })
        XCTAssertEqual(overlayResult.map(\.text).joined(), text)
    }

    func testReadableChunksMergeShortTrailingOverlaySentences() {
        let text = "只要给你无限的时间写，是不是？是总有机，肯定会出现重复的。就是写的这个牌的顺序完全重复，就像哎，他说。说这个女士，一个女士买了啊五百个。前面这部分继续补充一些上下文，确保实时浮层需要拆成多个可读片段。这里再补一段会议里的连续说明，用来模拟远端实时模型一次返回较长内容，但最后又带上几个非常短的句子。嗯嗯，上一千双鞋。给他一千年的时间搭。"
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 18,
            text: text
        )

        let result = MeetingTranscriptPostProcessor.process(
            [segment],
            options: .liveOverlay
        )
        let chunks = result.map(\.text)

        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertFalse(chunks.contains("嗯嗯，上一千双鞋。"))
        XCTAssertFalse(chunks.contains("给他一千年的时间搭。"))
        XCTAssertTrue(chunks.contains { $0.contains("嗯嗯，上一千双鞋。给他一千年的时间搭。") })
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 130 })
        XCTAssertEqual(chunks.joined(), MeetingTranscriptTextPostProcessor.normalizedFinalText(text))
    }

    func testFinalPostProcessorSplitsRealMeetingExportSampleIntoReadableChunks() {
        let exportedSample = "哎。一些可能。不那么需要的。放在起，放进 C P U。好了，非常感谢万晨老师。我觉得我们也。可以到下一个环节了。我们啊，除了要推理方面的优化。阿西拉，相比于。啊，不一样的这个解解解决方案。我们也提供了这个。带领就。立马，我们就有了一个 R L 上的这个知识。相当于说。直接就可以拿你的数据在上面进行一些微调。啊，进行一些。标化。然后让让它能够适用在你的这个使用场。场景中。那么这里的话，我也邀请我们下一位同学。上来给我们。介绍一下这个 mouse 相关的啊，还是。Java开发。对。嗯，我觉得其实带领。二 L 这样事件本身。"
        let exportedText = "\(exportedSample)接下来我们继续确认第二部分内容。\(exportedSample)"
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 4,
            endSeconds: 45,
            text: exportedText
        )

        let result = MeetingTranscriptPostProcessor.process([segment])

        XCTAssertGreaterThan(result.count, 1)
        XCTAssertTrue(result.allSatisfy { $0.text.count <= 260 })
        XCTAssertEqual(result.first?.id, segment.id)
        XCTAssertEqual(result.first?.startSeconds ?? -1, 4, accuracy: 0.001)
        XCTAssertEqual(result.last?.endSeconds ?? -1, 45, accuracy: 0.001)
        XCTAssertEqual(result.map(\.text).joined(), MeetingTranscriptTextPostProcessor.normalizedFinalText(exportedText))
    }

    func testMeetingTranscriptSanitizerSuppressesPromptEcho() {
        let prompt = "The speaker's primary language is Simplified Chinese. Mixed-language speech is expected. Preserve names, product terms, URLs, and code-like text exactly as spoken."

        XCTAssertEqual(
            MeetingTranscriptSanitizer.sanitizedText(prompt, prompt: prompt),
            ""
        )
    }

    func testMeetingTranscriptSanitizerSuppressesHintOnlyEcho() {
        let entries = [
            DictionaryEntry(
                term: "Qwen",
                normalizedTerm: "qwen",
                source: .manual,
                status: .active
            ),
            DictionaryEntry(
                term: "FluidAudio",
                normalizedTerm: "fluidaudio",
                source: .manual,
                status: .active
            )
        ]

        XCTAssertEqual(
            MeetingTranscriptSanitizer.sanitizedText(
                "Qwen, FluidAudio",
                contextualPhrases: ["Silero VAD"],
                dictionaryEntries: entries
            ),
            ""
        )
        XCTAssertEqual(
            MeetingTranscriptSanitizer.sanitizedText(
                "我们今天讨论 Qwen 的会议转写效果。",
                contextualPhrases: ["Silero VAD"],
                dictionaryEntries: entries
            ),
            "我们今天讨论 Qwen 的会议转写效果。"
        )
    }

    func testFinalPostProcessorKeepsCoherentShortTextTogetherEvenWhenDurationIsLong() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 35,
            text: "第一句介绍背景。第二句解释方案。第三句讨论风险。第四句确认下一步。"
        )

        let result = MeetingTranscriptPostProcessor.process([segment])

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result.first?.startSeconds ?? -1, 0, accuracy: 0.001)
        XCTAssertEqual(result.last?.endSeconds ?? -1, 35, accuracy: 0.001)
    }

    func testFinalPostProcessorDoesNotMergeAcrossReadablePauseGap() {
        let first = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 2,
            text: "第一句讲完一个完整观点。"
        )
        let second = MeetingTranscriptSegment(
            speaker: .them,
            speakerID: "S1",
            speakerDisplayName: "Speaker 1",
            audioSource: .systemAudio,
            startSeconds: 2.8,
            endSeconds: 4,
            text: "停顿后开始新的观点。"
        )

        let result = MeetingTranscriptPostProcessor.process([first, second])

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result.map(\.text), [first.text, second.text])
    }

    func testFinalTranscriptionPassBuildsLongOverlappingChunksPerSource() {
        let samples = [Float](repeating: 0.1, count: 40)
        let asset = MeetingAudioAsset(
            source: .systemAudio,
            samples: samples,
            sampleRate: 10,
            sessionStartOffset: 3
        )

        let chunks = MeetingFinalTranscriptionPass.chunks(
            for: asset,
            options: MeetingFinalTranscriptionPass.Options(
                maxChunkSeconds: 2,
                overlapSeconds: 0.5,
                minimumChunkSeconds: 0.5,
                minimumRMS: 0.001
            )
        )

        XCTAssertEqual(chunks.count, 3)
        XCTAssertEqual(chunks[0].startSeconds, 3, accuracy: 0.001)
        XCTAssertEqual(chunks[0].endSeconds, 5, accuracy: 0.001)
        XCTAssertEqual(chunks[1].startSeconds, 4.5, accuracy: 0.001)
        XCTAssertEqual(chunks[1].speaker, .them)
    }

    func testFinalTranscriptionPassUsesMicrophoneSpeakerForMicrophoneAssets() {
        let asset = MeetingAudioAsset(
            source: .microphone,
            samples: [Float](repeating: 0.1, count: 20),
            sampleRate: 10,
            sessionStartOffset: 0
        )

        let chunks = MeetingFinalTranscriptionPass.chunks(
            for: asset,
            options: MeetingFinalTranscriptionPass.Options(
                maxChunkSeconds: 3,
                overlapSeconds: 0,
                minimumChunkSeconds: 0.5,
                minimumRMS: 0.001
            )
        )

        XCTAssertFalse(chunks.isEmpty)
        XCTAssertTrue(chunks.allSatisfy { $0.speaker == .me })
    }

    func testFinalTranscriptionPassSplitsChunksOnLongSilence() {
        let samples =
            [Float](repeating: 0.1, count: 12) +
            [Float](repeating: 0, count: 10) +
            [Float](repeating: 0.1, count: 12)
        let asset = MeetingAudioAsset(
            source: .systemAudio,
            samples: samples,
            sampleRate: 10,
            sessionStartOffset: 3
        )

        let chunks = MeetingFinalTranscriptionPass.chunks(
            for: asset,
            options: MeetingFinalTranscriptionPass.Options(
                maxChunkSeconds: 10,
                overlapSeconds: 0,
                minimumChunkSeconds: 0.2,
                minimumRMS: 0.001,
                silenceSplitSeconds: 0.8,
                silenceWindowSeconds: 0.1,
                speechPaddingSeconds: 0
            )
        )

        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0].startSeconds, 3, accuracy: 0.001)
        XCTAssertEqual(chunks[0].endSeconds, 4.2, accuracy: 0.001)
        XCTAssertEqual(chunks[1].startSeconds, 5.2, accuracy: 0.001)
        XCTAssertEqual(chunks[1].endSeconds, 6.4, accuracy: 0.001)
        XCTAssertTrue(chunks.allSatisfy(\.preventsAdjacentMerge))
    }

    @MainActor
    func testChunkPlanningKeepsRangesAndMaterializesOnlyTheRequestedPCM() {
        let asset = MeetingAudioAsset(
            source: .mixed, samples: Array(repeating: Float(0.1), count: 600),
            sampleRate: 10, sessionStartOffset: 120
        )
        let plans = MeetingFinalTranscriptionPass.chunkPlans(for: asset)
        XCTAssertEqual(plans.map(\.samples), [0..<220, 210..<430, 420..<600])
        let chunks = MeetingFinalTranscriptionPass.chunks(for: asset)
        XCTAssertEqual(chunks.count, plans.count)
        let middle = chunks[1]
        XCTAssertEqual(middle.samples, Array(asset.samples[210..<430]))
        XCTAssertEqual(middle.startSeconds, 141)
        XCTAssertEqual(middle.endSeconds, 163)
        XCTAssertFalse(middle.preventsAdjacentMerge)
    }

    @MainActor
    func testFinalTranscriptionPassPrefersWholeAssetTranscriptionWhenAvailable() async throws {
        let asset = MeetingAudioAsset(
            source: .systemAudio,
            samples: [Float](repeating: 0.1, count: 20),
            sampleRate: 10,
            sessionStartOffset: 3
        )
        let transcriber = WholeAssetMeetingTranscriber()

        let segments = try await MeetingFinalTranscriptionPass.transcribe(
            assets: [asset],
            transcriber: transcriber
        )

        XCTAssertEqual(transcriber.chunkTranscriptionCount, 0)
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].speakerID, "moss:S01")
        XCTAssertEqual(segments[0].text, "whole asset")
    }

    @MainActor
    func testFinalTranscriptionPassDoesNotReturnPartialWholeAssetResults() async {
        let assets = [
            MeetingAudioAsset(
                source: .microphone,
                samples: [Float](repeating: 0.1, count: 20),
                sampleRate: 10,
                sessionStartOffset: 0
            ),
            MeetingAudioAsset(
                source: .systemAudio,
                samples: [Float](repeating: 0.1, count: 20),
                sampleRate: 10,
                sessionStartOffset: 0
            ),
        ]
        let transcriber = FailingWholeAssetMeetingTranscriber(failingSource: .systemAudio)

        do {
            _ = try await MeetingFinalTranscriptionPass.transcribe(
                assets: assets,
                transcriber: transcriber
            )
            XCTFail("Expected the whole final transcription pass to fail atomically")
        } catch {
            XCTAssertEqual(error as? WholeAssetTranscriptionStubError, .inferenceFailed)
        }

        XCTAssertEqual(transcriber.successfulSources, [.microphone])
    }

    @MainActor
    func testFinalTranscriptionPassThrowsWhenDescriptorAssetCannotBeLoaded() async {
        let descriptor = MeetingAudioAssetDescriptor(
            source: .systemAudio,
            sampleRate: 10,
            startSample: 0,
            sampleCount: 20
        )

        do {
            _ = try await MeetingFinalTranscriptionPass.transcribe(
                descriptors: [descriptor],
                loadAsset: { _ in nil },
                transcriber: WholeAssetMeetingTranscriber()
            )
            XCTFail("Expected an unavailable asset error")
        } catch {
            XCTAssertEqual(
                error as? MeetingFinalTranscriptionPass.Failure,
                .assetUnavailable(.systemAudio)
            )
        }
    }

    @MainActor
    func testFinalTranscriptionPassResumesAfterCommittedDescriptor() async throws {
        let descriptors = [
            MeetingAudioAssetDescriptor(source: .mixed, sampleRate: 10, startSample: 0, sampleCount: 20),
            MeetingAudioAssetDescriptor(source: .mixed, sampleRate: 10, startSample: 20, sampleCount: 20),
        ]
        let initial = MeetingTranscriptSegment(
            speaker: .them, startSeconds: 0, endSeconds: 2, text: "already committed"
        )
        let recorder = MeetingAnalysisProgressRecorder()
        let checkpointCounts = CheckpointCountRecorder()

        let segments = try await MeetingFinalTranscriptionPass.transcribe(
            descriptors: descriptors,
            loadAsset: { descriptor in
                MeetingAudioAsset(
                    source: descriptor.source,
                    samples: [Float](repeating: 0.1, count: descriptor.sampleCount),
                    sampleRate: descriptor.sampleRate,
                    sessionStartOffset: descriptor.sessionStartOffset
                )
            },
            transcriber: WholeAssetMeetingTranscriber(),
            progress: { value in await recorder.append(value) },
            initialSegments: [initial],
            startingDescriptorIndex: 1,
            checkpoint: { segments, completedCount in
                await checkpointCounts.append(segments.count, completedCount)
            }
        )

        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments.first?.text, "already committed")
        XCTAssertEqual(segments.last?.text, "whole asset")
        let checkpointValues = await checkpointCounts.values()
        let progressValues = await recorder.values
        XCTAssertEqual(checkpointValues.count, 1)
        XCTAssertEqual(checkpointValues.first?.0, 2)
        XCTAssertEqual(checkpointValues.first?.1, 2)
        XCTAssertEqual(progressValues, [0.5, 1])
    }

    @MainActor
    func testFinalTranscriptionPassReportsDescriptorProgress() async throws {
        let descriptors = [
            MeetingAudioAssetDescriptor(
                source: .mixed,
                sampleRate: 10,
                startSample: 0,
                sampleCount: 20
            ),
            MeetingAudioAssetDescriptor(
                source: .mixed,
                sampleRate: 10,
                startSample: 20,
                sampleCount: 20
            ),
        ]
        let recorder = MeetingAnalysisProgressRecorder()

        _ = try await MeetingFinalTranscriptionPass.transcribe(
            descriptors: descriptors,
            loadAsset: { descriptor in
                MeetingAudioAsset(
                    source: descriptor.source,
                    samples: [Float](repeating: 0.1, count: descriptor.sampleCount),
                    sampleRate: descriptor.sampleRate,
                    sessionStartOffset: descriptor.sessionStartOffset
                )
            },
            transcriber: WholeAssetMeetingTranscriber(),
            progress: { value in
                await recorder.append(value)
            }
        )

        let progressValues = await recorder.values
        XCTAssertEqual(progressValues, [0, 0.5, 1])
    }

    @MainActor
    func testFinalTranscriptionPassPropagatesStrictChunkFailure() async {
        let descriptor = MeetingAudioAssetDescriptor(
            source: .mixed,
            sampleRate: 10,
            startSample: 0,
            sampleCount: 20
        )

        do {
            _ = try await MeetingFinalTranscriptionPass.transcribe(
                descriptors: [descriptor],
                loadAsset: { descriptor in
                    MeetingAudioAsset(
                        source: descriptor.source,
                        samples: [Float](repeating: 0.1, count: descriptor.sampleCount),
                        sampleRate: descriptor.sampleRate,
                        sessionStartOffset: descriptor.sessionStartOffset
                    )
                },
                transcriber: StrictFailingMeetingTranscriber(),
                requiresCompleteTranscription: true
            )
            XCTFail("Expected strict chunk transcription to fail atomically")
        } catch {
            XCTAssertEqual(error as? WholeAssetTranscriptionStubError, .inferenceFailed)
        }
    }

    func testBalancedSpeakerAssemblyKeepsContinuousSegmentTogetherAcrossShortSpeakerTurns() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            audioSource: .mixed,
            startSeconds: 0,
            endSeconds: 12,
            text: "今天我们先确认采集链路和模型选择，然后继续讨论发言人识别在短句和停顿里的稳定性，最后再看会议详情里的段落阅读效果。"
        )
        let turns = [
            MeetingSpeakerTurn(
                source: .mixed,
                speakerID: "speaker_a",
                displayName: "Speaker 1",
                startSeconds: 0,
                endSeconds: 4.4,
                confidence: 0.7
            ),
            MeetingSpeakerTurn(
                source: .mixed,
                speakerID: "speaker_b",
                displayName: "Speaker 2",
                startSeconds: 4.4,
                endSeconds: 5.1,
                confidence: 0.65
            ),
            MeetingSpeakerTurn(
                source: .mixed,
                speakerID: "speaker_a",
                displayName: "Speaker 1",
                startSeconds: 5.1,
                endSeconds: 12,
                confidence: 0.75
            )
        ]

        let smoothedTurns = MeetingSpeakerTurnSmoother.smooth(
            turns,
            options: MeetingSpeakerDiarizationSensitivity.balanced.smootherOptions
        )
        let result = MeetingSpeakerTranscriptAssembler.assemble(
            segments: [segment],
            speakerTurns: smoothedTurns,
            options: MeetingSpeakerDiarizationSensitivity.balanced.transcriptAssemblyOptions
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].text, segment.text)
        XCTAssertEqual(result[0].speakerID, "speaker_a")
        XCTAssertEqual(result[0].speakerDisplayName, "Speaker 1")
    }
}

private actor MeetingAnalysisProgressRecorder {
    private var recordedValues: [Double] = []

    func append(_ value: Double) {
        recordedValues.append(value)
    }

    var values: [Double] {
        recordedValues
    }
}

private actor CheckpointCountRecorder {
    private var recorded: [(Int, Int)] = []

    func append(_ segmentCount: Int, _ completedCount: Int) {
        recorded.append((segmentCount, completedCount))
    }

    func values() -> [(Int, Int)] { recorded }
}

@MainActor
private final class WholeAssetMeetingTranscriber: MeetingSegmentTranscribing {
    private(set) var chunkTranscriptionCount = 0

    func transcribe(chunk: BufferedMeetingChunk) async -> MeetingTranscriptSegment? {
        chunkTranscriptionCount += 1
        return nil
    }

    func transcribeWholeAsset(_ asset: MeetingAudioAsset) async throws -> [MeetingTranscriptSegment]? {
        [
            MeetingTranscriptSegment(
                speaker: asset.source.defaultSpeaker,
                speakerID: "moss:S01",
                speakerDisplayName: "Speaker 1",
                audioSource: asset.source,
                startSeconds: asset.sessionStartOffset,
                endSeconds: asset.sessionStartOffset + asset.durationSeconds,
                text: "whole asset"
            )
        ]
    }
}

private enum WholeAssetTranscriptionStubError: Error, Equatable {
    case inferenceFailed
}

@MainActor
private final class FailingWholeAssetMeetingTranscriber: MeetingSegmentTranscribing {
    private let failingSource: TranscriptAudioSource
    private(set) var successfulSources: [TranscriptAudioSource] = []

    init(failingSource: TranscriptAudioSource) {
        self.failingSource = failingSource
    }

    func transcribe(chunk: BufferedMeetingChunk) async -> MeetingTranscriptSegment? {
        nil
    }

    func transcribeWholeAsset(_ asset: MeetingAudioAsset) async throws -> [MeetingTranscriptSegment]? {
        guard asset.source != failingSource else {
            throw WholeAssetTranscriptionStubError.inferenceFailed
        }
        successfulSources.append(asset.source)
        return [
            MeetingTranscriptSegment(
                speaker: asset.source.defaultSpeaker,
                speakerID: "moss:S01",
                speakerDisplayName: "Speaker 1",
                audioSource: asset.source,
                startSeconds: asset.sessionStartOffset,
                endSeconds: asset.sessionStartOffset + asset.durationSeconds,
                text: "whole asset"
            )
        ]
    }
}

@MainActor
private final class StrictFailingMeetingTranscriber: MeetingSegmentTranscribing {
    func transcribe(chunk: BufferedMeetingChunk) async -> MeetingTranscriptSegment? {
        nil
    }

    func transcribeSegmentsStrict(chunk: BufferedMeetingChunk) async throws -> [MeetingTranscriptSegment] {
        throw WholeAssetTranscriptionStubError.inferenceFailed
    }
}
