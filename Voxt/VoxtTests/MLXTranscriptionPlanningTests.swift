// MLXTranscriptionPlanningTests.swift
// Provides MLXTranscription Planning Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class MLXTranscriptionPlanningTests: XCTestCase {
    func testVoiceActivityFilteredSamplesPreservePrerollBeforeDetectedSpeech() {
        var buffer = MLXVoiceActivitySampleContextBuffer()

        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [1], startSeconds: 0.0),
                isSpeech: false
            ),
            []
        )
        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [2], startSeconds: 0.1),
                isSpeech: false
            ),
            []
        )
        let filtered = buffer.append(
            voiceActivityFrame(samples: [3], startSeconds: 0.2),
            isSpeech: true
        )

        XCTAssertEqual(filtered, [1, 2, 3])
        XCTAssertTrue(buffer.observedFrames)
        XCTAssertTrue(buffer.observedSpeech)
    }

    func testVoiceActivityFilteredSamplesBoundLongPrerollBeforeDetectedSpeech() {
        var buffer = MLXVoiceActivitySampleContextBuffer()

        for index in 0..<6 {
            XCTAssertEqual(
                buffer.append(
                    voiceActivityFrame(samples: [Float(index)], startSeconds: Double(index) * 0.1),
                    isSpeech: false
                ),
                []
            )
        }
        let filtered = buffer.append(
            voiceActivityFrame(samples: [6], startSeconds: 0.6),
            isSpeech: true
        )

        XCTAssertEqual(filtered, [3, 4, 5, 6])
    }

    func testVoiceActivityFilteredSamplesFlushPrerollBeforeEachSpeechBurst() {
        var buffer = MLXVoiceActivitySampleContextBuffer()

        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [1], startSeconds: 0.0),
                isSpeech: false
            ),
            []
        )
        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [2], startSeconds: 0.1),
                isSpeech: true
            ),
            [1, 2]
        )
        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [3], startSeconds: 0.2),
                isSpeech: false
            ),
            []
        )
        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [4], startSeconds: 0.3),
                isSpeech: false
            ),
            []
        )
        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [5], startSeconds: 0.4),
                isSpeech: true
            ),
            [3, 4, 5]
        )
    }

    func testVoiceActivityFilteredSamplesFinishDropsTrailingSilenceAfterSpeech() {
        var buffer = MLXVoiceActivitySampleContextBuffer()

        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [1], startSeconds: 0.0),
                isSpeech: true
            ),
            [1]
        )
        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [2], startSeconds: 0.1),
                isSpeech: false
            ),
            []
        )
        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [3], startSeconds: 0.2),
                isSpeech: false
            ),
            []
        )

        // Trailing non-speech must not lengthen Final ASR input.
        XCTAssertEqual(buffer.finish(), [])
    }

    func testFinalizationSamplesPreferFilteredOnlyWhenStrictlyShorter() {
        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: [1, 2, 3, 4],
            voiceActivityFilteredSamples: [1, 2, 3, 4],
            localVADGateActive: true,
            observedVoiceActivityFrames: true,
            observedSpeech: true
        )

        XCTAssertEqual(selection.samples, [1, 2, 3, 4])
        XCTAssertEqual(selection.source, .full)
    }

    func testFinalizationSamplesUseFilteredForStandardQwenPolicyWhenShorter() {
        let capability = MLXModelCatalog.capability(for: MLXModelManager.defaultModelRepo)
        XCTAssertEqual(capability.vadPolicy, .standard)

        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: [1, 2, 3, 4, 5, 6],
            voiceActivityFilteredSamples: [2, 3, 4],
            localVADGateActive: true,
            observedVoiceActivityFrames: true,
            observedSpeech: true,
            vadPolicy: capability.vadPolicy
        )

        XCTAssertEqual(selection.samples, [2, 3, 4])
        XCTAssertEqual(selection.source, .voiceActivityFiltered)
    }

    func testVoiceActivityFilteredSamplesFinishDropsContextWhenNoSpeechWasObserved() {
        var buffer = MLXVoiceActivitySampleContextBuffer()

        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [1], startSeconds: 0.0),
                isSpeech: false
            ),
            []
        )

        XCTAssertEqual(buffer.finish(), [])
        XCTAssertTrue(buffer.observedFrames)
        XCTAssertFalse(buffer.observedSpeech)
    }

    func testVoiceActivityFilteredSamplesResetClearsPrerollAndObservedState() {
        var buffer = MLXVoiceActivitySampleContextBuffer()

        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [1], startSeconds: 0.0),
                isSpeech: false
            ),
            []
        )
        buffer.reset()

        XCTAssertFalse(buffer.observedFrames)
        XCTAssertFalse(buffer.observedSpeech)
        XCTAssertEqual(
            buffer.append(
                voiceActivityFrame(samples: [2], startSeconds: 0.1),
                isSpeech: true
            ),
            [2]
        )
    }

    func testFinalizationSamplesUseFullAudioWhenLocalVADIsInactive() {
        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: [1, 2, 3, 4],
            voiceActivityFilteredSamples: [2, 3],
            localVADGateActive: false,
            observedVoiceActivityFrames: true,
            observedSpeech: true
        )

        XCTAssertEqual(selection.samples, [1, 2, 3, 4])
        XCTAssertEqual(selection.source, .full)
    }

    func testFinalizationSamplesUseFullAudioUntilVADFramesAreAvailable() {
        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: [1, 2, 3, 4],
            voiceActivityFilteredSamples: [],
            localVADGateActive: true,
            observedVoiceActivityFrames: false,
            observedSpeech: false
        )

        XCTAssertEqual(selection.samples, [1, 2, 3, 4])
        XCTAssertEqual(selection.source, .full)
    }

    func testFinalizationSamplesUseVoiceActivityFilteredAudioWhenSpeechWasObserved() {
        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: [1, 2, 3, 4],
            voiceActivityFilteredSamples: [2, 3],
            localVADGateActive: true,
            observedVoiceActivityFrames: true,
            observedSpeech: true
        )

        XCTAssertEqual(selection.samples, [2, 3])
        XCTAssertEqual(selection.source, .voiceActivityFiltered)
    }

    func testFinalizationSamplesSkipASRWhenVADObservedNoSpeech() {
        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: [1, 2, 3, 4],
            voiceActivityFilteredSamples: [],
            localVADGateActive: true,
            observedVoiceActivityFrames: true,
            observedSpeech: false
        )

        XCTAssertEqual(selection.samples, [])
        XCTAssertEqual(selection.source, .noSpeech)
    }

    func testFinalizationSamplesPreserveTimelineUsesFullAudioEvenWhenFilteredSpeechExists() {
        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: [1, 2, 3, 4],
            voiceActivityFilteredSamples: [2, 3],
            localVADGateActive: true,
            observedVoiceActivityFrames: true,
            observedSpeech: true,
            vadPolicy: .preserveTimeline
        )

        XCTAssertEqual(selection.samples, [1, 2, 3, 4])
        XCTAssertEqual(selection.source, .full)
    }

    func testFinalizationSamplesPreserveTimelineStillSkipsWhenNoSpeech() {
        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: [1, 2, 3, 4],
            voiceActivityFilteredSamples: [],
            localVADGateActive: true,
            observedVoiceActivityFrames: true,
            observedSpeech: false,
            vadPolicy: .preserveTimeline
        )

        XCTAssertEqual(selection.samples, [])
        XCTAssertEqual(selection.source, .noSpeech)
    }

    func testPostStopFinalMaxTokensScalesWithAudioDuration() {
        XCTAssertEqual(MLXTranscriptionPlanning.postStopFinalMaxTokens(audioDurationSeconds: 1), 256)
        XCTAssertEqual(MLXTranscriptionPlanning.postStopFinalMaxTokens(audioDurationSeconds: 10), 344)
        XCTAssertEqual(MLXTranscriptionPlanning.postStopFinalMaxTokens(audioDurationSeconds: 120), 3424)
        XCTAssertEqual(MLXTranscriptionPlanning.postStopFinalMaxTokens(audioDurationSeconds: 600), 8192)
    }


    func testFinalizationSamplesModelManagedKeepsFullAudio() {
        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: [1, 2, 3, 4],
            voiceActivityFilteredSamples: [2, 3],
            localVADGateActive: true,
            observedVoiceActivityFrames: true,
            observedSpeech: true,
            vadPolicy: .modelManaged
        )

        XCTAssertEqual(selection.samples, [1, 2, 3, 4])
        XCTAssertEqual(selection.source, .full)
        XCTAssertFalse(MLXVADPolicy.modelManaged.allowsExternalFinalSpeechTrim)
    }

    func testFinalizationSamplesSenseVoiceKeepsFullAudioDespiteStandardCatalogPolicy() {
        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: [1, 2, 3, 4],
            voiceActivityFilteredSamples: [2, 3],
            localVADGateActive: true,
            observedVoiceActivityFrames: true,
            observedSpeech: true,
            vadPolicy: .standard,
            family: .senseVoice
        )

        XCTAssertEqual(selection.samples, [1, 2, 3, 4])
        XCTAssertEqual(selection.source, .full)
    }

    func testFinalizationSamplesSenseVoiceStillSkipsWhenNoSpeechObserved() {
        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: [1, 2, 3, 4],
            voiceActivityFilteredSamples: [],
            localVADGateActive: true,
            observedVoiceActivityFrames: true,
            observedSpeech: false,
            vadPolicy: .standard,
            family: .senseVoice
        )

        XCTAssertEqual(selection.samples, [])
        XCTAssertEqual(selection.source, .noSpeech)
    }

    func testPostStopFinalChunkDurationLiftsAccuracyFirstSlicesToSingleOfflineWindow() {
        XCTAssertEqual(MLXTranscriptionPlanning.postStopFinalChunkDuration(presetChunkDuration: 90), 1200)
        XCTAssertEqual(MLXTranscriptionPlanning.postStopFinalChunkDuration(presetChunkDuration: 1200), 1200)
    }

    func testPostStopFinalKVCachePolicyKeepsConservativeQwenForQwenFamily() {
        XCTAssertEqual(
            MLXTranscriptionPlanning.postStopFinalKVCachePolicy(
                family: .qwen3ASR,
                catalogPolicy: .conservativeQwen
            ),
            .conservativeQwen
        )
        XCTAssertEqual(MLXASRKVCachePolicy.conservativeQwen.quantizedStart, 256)
        XCTAssertEqual(
            MLXTranscriptionPlanning.postStopFinalKVCachePolicy(
                family: .qwen3ASR,
                catalogPolicy: nil
            ),
            .conservativeQwen
        )
        XCTAssertNil(
            MLXTranscriptionPlanning.postStopFinalKVCachePolicy(
                family: .mossTranscribeDiarize,
                catalogPolicy: nil
            )
        )
    }

    func testMOSSHotwordsAreFinalOnly() {
        XCTAssertFalse(MLXTranscriptionPlanning.shouldIncludeMOSSHotwords(for: .intermediate))
        XCTAssertTrue(MLXTranscriptionPlanning.shouldIncludeMOSSHotwords(for: .postStopQuick))
        XCTAssertTrue(MLXTranscriptionPlanning.shouldIncludeMOSSHotwords(for: .postStopFinal))

        let livePrompt = MossASRPromptSupport.resolvedPrompt(
            outputMode: .plainText,
            customPrompt: "",
            hotwords: MLXTranscriptionPlanning.shouldIncludeMOSSHotwords(for: .intermediate)
                ? "Voxt, Codex"
                : ""
        )
        let finalPrompt = MossASRPromptSupport.resolvedPrompt(
            outputMode: .plainText,
            customPrompt: "",
            hotwords: MLXTranscriptionPlanning.shouldIncludeMOSSHotwords(for: .postStopFinal)
                ? "Voxt, Codex"
                : ""
        )

        XCTAssertEqual(
            livePrompt,
            "Transcribe the audio as plain text without timestamps or speaker labels."
        )
        XCTAssertFalse(livePrompt.contains("Hotwords:"))
        XCTAssertEqual(
            finalPrompt,
            "Transcribe the audio as plain text without timestamps or speaker labels. Hotwords: Voxt, Codex"
        )
    }

    private func voiceActivityFrame(
        samples: [Float],
        startSeconds: TimeInterval,
        durationSeconds: TimeInterval = 0.1
    ) -> ASRVoiceActivityAudioFrame {
        ASRVoiceActivityAudioFrame(
            samples: samples,
            sampleRate: 10,
            startSeconds: startSeconds,
            endSeconds: startSeconds + durationSeconds,
            level: nil
        )
    }

    func testSenseVoiceUsesDirectPassForShortAudio() {
        let shouldUseVAD = MLXTranscriptionPlanning.shouldUseSenseVoiceVAD(
            sampleCount: 16000 * 12,
            sampleRate: 16000,
            directPassMaximumDurationSeconds: 30
        )

        XCTAssertFalse(shouldUseVAD)
    }

    func testSenseVoiceUsesVADForLongAudio() {
        let shouldUseVAD = MLXTranscriptionPlanning.shouldUseSenseVoiceVAD(
            sampleCount: 16000 * 40,
            sampleRate: 16000,
            directPassMaximumDurationSeconds: 30
        )

        XCTAssertTrue(shouldUseVAD)
    }

    func testSenseVoiceSplitRangeReturnsOriginalRangeWhenChunkingIsNotNeeded() {
        let ranges = MLXTranscriptionPlanning.splitSenseVoiceRange(
            start: 100,
            end: 1000,
            maxChunkSamples: 5000,
            overlapSamples: 320
        )

        XCTAssertEqual(ranges, [100..<1000])
    }

    func testSenseVoiceSplitRangeProducesOverlappingChunksForLongSegments() {
        let sampleCount = 16000 * 61
        let ranges = MLXTranscriptionPlanning.splitSenseVoiceRange(
            start: 0,
            end: sampleCount,
            maxChunkSamples: 16000 * 24,
            overlapSamples: Int(0.35 * 16000)
        )

        XCTAssertGreaterThan(ranges.count, 1)
        XCTAssertEqual(ranges.first?.lowerBound, 0)
        XCTAssertEqual(ranges.last?.upperBound, sampleCount)
        XCTAssertEqual(ranges[0].upperBound - ranges[1].lowerBound, Int(0.35 * 16000))
    }

    func testNativeLiveLanguagePreservesAutomaticSelection() {
        XCTAssertNil(MLXTranscriptionPlanning.nativeLiveLanguage(from: nil))
        XCTAssertNil(MLXTranscriptionPlanning.nativeLiveLanguage(from: "  "))
        XCTAssertEqual(MLXTranscriptionPlanning.nativeLiveLanguage(from: " Chinese "), "Chinese")
    }

    func testNativeNemotronLanguageUsesCheckpointLocaleKeys() {
        let available = ["auto", "en-US", "zh-CN", "de-DE"]

        XCTAssertEqual(
            MLXTranscriptionPlanning.nativeNemotronLanguage(
                requested: "zh",
                availableLanguages: available,
                defaultLanguage: "auto"
            ),
            "zh-CN"
        )
        XCTAssertEqual(
            MLXTranscriptionPlanning.nativeNemotronLanguage(
                requested: "de",
                availableLanguages: available,
                defaultLanguage: "auto"
            ),
            "de-DE"
        )
        XCTAssertEqual(
            MLXTranscriptionPlanning.nativeNemotronLanguage(
                requested: nil,
                availableLanguages: available,
                defaultLanguage: "auto"
            ),
            "auto"
        )
    }

    func testSenseVoiceSegmentRangesUseSharedVADPolicy() {
        let probabilities: [Float] = [
            0.01,
            0.8,
            0.82,
            0.2,
            0.1,
            0.05
        ]

        let ranges = MLXTranscriptionPlanning.senseVoiceSegmentRanges(
            probabilities: probabilities,
            sampleCount: 6 * 1600,
            sampleRate: 16_000,
            probabilityFrameSampleCount: 1600,
            vadThreshold: 0.5,
            vadMinSpeechDurationMs: 150,
            vadMinSilenceDurationMs: 200,
            vadSpeechPadMs: 100,
            maxChunkSamples: 16_000,
            overlapSamples: 0
        )

        XCTAssertEqual(ranges, [0..<6400])
    }

    func testSenseVoiceSegmentRangesSplitLongSpeechSegments() {
        let probabilities = [Float](repeating: 0.9, count: 12)

        let ranges = MLXTranscriptionPlanning.senseVoiceSegmentRanges(
            probabilities: probabilities,
            sampleCount: 12 * 1600,
            sampleRate: 16_000,
            probabilityFrameSampleCount: 1600,
            vadThreshold: 0.5,
            vadMinSpeechDurationMs: 150,
            vadMinSilenceDurationMs: 200,
            vadSpeechPadMs: 0,
            maxChunkSamples: 4800,
            overlapSamples: 1600
        )

        XCTAssertGreaterThan(ranges.count, 1)
        XCTAssertEqual(ranges.first, 0..<4800)
        XCTAssertEqual(ranges.last?.upperBound, 19200)
        XCTAssertEqual(ranges[0].upperBound - ranges[1].lowerBound, 1600)
    }

    func testSenseVoiceVisibleRealtimeCorrectionCadenceIsMoreAggressive() {
        let cadence = MLXTranscriptionPlanning.correctionCadence(
            for: "mlx-community/SenseVoiceSmall",
            sessionAllowsRealtimeTextDisplay: true
        )

        XCTAssertEqual(cadence.correctionIntervalSeconds, 4.0, accuracy: 0.0001)
        XCTAssertEqual(cadence.firstCorrectionMinimumSeconds, 2.2, accuracy: 0.0001)
        XCTAssertEqual(cadence.intermediateContextWindowSeconds, 14.0, accuracy: 0.0001)
        XCTAssertEqual(cadence.quickPassContextWindowSeconds, 24.0, accuracy: 0.0001)
    }

    func testDefaultVisibleRealtimeCorrectionCadenceRemainsUnchanged() {
        let cadence = MLXTranscriptionPlanning.correctionCadence(
            for: "mlx-community/Qwen3-ASR-0.6B-4bit",
            sessionAllowsRealtimeTextDisplay: true
        )

        XCTAssertEqual(cadence.correctionIntervalSeconds, 6.0, accuracy: 0.0001)
        XCTAssertEqual(cadence.firstCorrectionMinimumSeconds, 3.5, accuracy: 0.0001)
        XCTAssertEqual(cadence.intermediateContextWindowSeconds, 18.0, accuracy: 0.0001)
        XCTAssertEqual(cadence.quickPassContextWindowSeconds, 30.0, accuracy: 0.0001)
    }

    func testSenseVoiceHiddenRealtimeCorrectionCadenceUsesShorterIntervals() {
        let cadence = MLXTranscriptionPlanning.correctionCadence(
            for: "mlx-community/SenseVoiceSmall",
            sessionAllowsRealtimeTextDisplay: false
        )

        XCTAssertEqual(cadence.correctionIntervalSeconds, 2.6, accuracy: 0.0001)
        XCTAssertEqual(cadence.firstCorrectionMinimumSeconds, 1.8, accuracy: 0.0001)
        XCTAssertEqual(cadence.intermediateContextWindowSeconds, 18.0, accuracy: 0.0001)
        XCTAssertEqual(cadence.quickPassContextWindowSeconds, 18.0, accuracy: 0.0001)
    }

    func testSenseVoiceSequentialMergeRemovesChunkBoundaryOverlap() {
        let merged = MLXTranscriptionPlanning.mergeSequentialTranscript(
            base: "hello world",
            next: "world again"
        )

        XCTAssertEqual(merged, "hello world again")
    }

    func testSenseVoiceSequentialMergeHandlesChineseBoundaryOverlap() {
        let merged = MLXTranscriptionPlanning.mergeSequentialTranscript(
            base: "我们正在测试长音频切分",
            next: "音频切分和合并效果"
        )

        XCTAssertEqual(merged, "我们正在测试长音频切分和合并效果")
    }

    func testIntermediateSchedulingSkipsWhenAnotherPassIsInFlight() {
        let decision = MLXTranscriptionPlanning.correctionPassSchedulingDecision(
            requestedPass: .intermediate,
            inFlightPass: .intermediate
        )

        XCTAssertEqual(decision, .skipRequestedPass)
    }

    func testStopTimeSchedulingInterruptsInFlightIntermediatePass() {
        let decision = MLXTranscriptionPlanning.correctionPassSchedulingDecision(
            requestedPass: .postStopFinal,
            inFlightPass: .intermediate
        )

        XCTAssertEqual(decision, .interruptInFlightPass)
    }

    func testStopTimeSchedulingWaitsForAnotherStopPass() {
        let decision = MLXTranscriptionPlanning.correctionPassSchedulingDecision(
            requestedPass: .postStopFinal,
            inFlightPass: .postStopQuick
        )

        XCTAssertEqual(decision, .waitForInFlightPass)
    }

    func testQuickStopPassDisabledForNativeLiveModes() {
        let plan = MLXFinalizationPlan(durationSeconds: 30, quickPassSampleCount: 16000 * 30)

        XCTAssertFalse(
            MLXTranscriptionPlanning.shouldRunQuickStopPass(
                plan: plan,
                sessionAllowsRealtimeTextDisplay: true,
                liveMode: .nativeQwenLive
            )
        )
        XCTAssertFalse(
            MLXTranscriptionPlanning.shouldRunQuickStopPass(
                plan: plan,
                sessionAllowsRealtimeTextDisplay: true,
                liveMode: .nativeNemotronLive
            )
        )
        XCTAssertTrue(
            MLXTranscriptionPlanning.shouldRunQuickStopPass(
                plan: plan,
                sessionAllowsRealtimeTextDisplay: true,
                liveMode: .batchPreview
            )
        )
    }

    func testNativeLiveVisiblePreviewSuppressesSilentCollapseWhenConfirmedIsUnchanged() {
        XCTAssertNil(
            MLXTranscriptionPlanning.resolvedNativeLiveVisiblePreview(
                previousPreview: "hello world",
                previousConfirmedText: "hello",
                confirmedText: "hello",
                provisionalText: ""
            )
        )
    }

    func testNativeLiveVisiblePreviewSuppressesProvisionalThrashWhenCombinedShrinks() {
        XCTAssertNil(
            MLXTranscriptionPlanning.resolvedNativeLiveVisiblePreview(
                previousPreview: "hello world",
                previousConfirmedText: "hello",
                confirmedText: "hello",
                provisionalText: " w"
            )
        )
    }

    func testNativeLiveVisiblePreviewSuppressesTinyProvisionalOnlyFlash() {
        XCTAssertNil(
            MLXTranscriptionPlanning.resolvedNativeLiveVisiblePreview(
                previousPreview: "",
                previousConfirmedText: "",
                confirmedText: "",
                provisionalText: "a"
            )
        )
    }

    func testNativeLiveVisiblePreviewAllowsNewCombinedText() {
        XCTAssertEqual(
            MLXTranscriptionPlanning.resolvedNativeLiveVisiblePreview(
                previousPreview: "hello",
                previousConfirmedText: "hello",
                confirmedText: "hello",
                provisionalText: " world"
            ),
            "hello world"
        )
    }

    func testQwenStreamingTextHidesProtocolUntilTextMarkerCompletes() {
        XCTAssertEqual(
            MLXTranscriptionPlanning.qwenStreamingVisibleText("language Chinese<asr_te"),
            ""
        )
        XCTAssertEqual(
            MLXTranscriptionPlanning.qwenStreamingVisibleText("language Chinese<asr_text>\u{4f60}\u{597d}\u{3002}"),
            "\u{4f60}\u{597d}\u{3002}"
        )
    }

    func testQwenStreamingTextParsesMarkerSplitAcrossConfirmedAndProvisionalText() {
        let parts = MLXTranscriptionPlanning.qwenStreamingVisibleTextParts(
            confirmedText: "language Chinese<asr_",
            provisionalText: "text>\u{4f60}\u{597d}\u{3002}"
        )

        XCTAssertEqual(parts.confirmedText, "")
        XCTAssertEqual(parts.provisionalText, "\u{4f60}\u{597d}\u{3002}")
    }

    func testQwenStreamingTextRemovesProtocolAfterWindowReset() {
        XCTAssertEqual(
            MLXTranscriptionPlanning.qwenStreamingVisibleText(
                "language Chinese<asr_text>\u{4f60}\u{597d}\u{3002} language Chinese<asr_text>\u{4e16}\u{754c}\u{3002}"
            ),
            "\u{4f60}\u{597d}\u{3002} \u{4e16}\u{754c}\u{3002}"
        )
        XCTAssertEqual(
            MLXTranscriptionPlanning.qwenStreamingVisibleText(
                "language Chinese<asr_text>\u{4f60}\u{597d}\u{3002} language Chin"
            ),
            "\u{4f60}\u{597d}\u{3002}"
        )
    }

    func testQwenStreamingTextPreservesOrdinaryLanguageTextBeforeWindowHeader() {
        XCTAssertEqual(
            MLXTranscriptionPlanning.qwenStreamingVisibleText(
                "I study language models language Chinese<asr_text>hello"
            ),
            "I study language models hello"
        )

        let parts = MLXTranscriptionPlanning.qwenStreamingVisibleTextParts(
            confirmedText: "I study language models language Chin",
            provisionalText: "ese<asr_text>hello"
        )
        XCTAssertEqual(parts.confirmedText, "I study language models")
        XCTAssertEqual(parts.provisionalText, " hello")
    }

    func testQwenFinalTextPreservesOrdinaryTrailingLanguageName() {
        XCTAssertEqual(
            MLXTranscriptionPlanning.qwenStreamingVisibleText(
                "language English<asr_text>I study language Chinese",
                suppressIncompleteWindowHeader: false
            ),
            "I study language Chinese"
        )
        XCTAssertEqual(
            MLXTranscriptionPlanning.qwenStreamingVisibleText(
                "language English<asr_text>call lang",
                suppressIncompleteWindowHeader: false
            ),
            "call lang"
        )
    }

    func testQwenStreamingTextLeavesOrdinaryDecodedTextUnchanged() {
        XCTAssertEqual(
            MLXTranscriptionPlanning.qwenStreamingVisibleText("Use language models carefully."),
            "Use language models carefully."
        )
        XCTAssertEqual(
            MLXTranscriptionPlanning.qwenStreamingVisibleText(
                "language English<asr_text>language models"
            ),
            "language models"
        )
        XCTAssertEqual(
            MLXTranscriptionPlanning.qwenStreamingVisibleText(
                "language Models<asr_text>hello"
            ),
            "language Models<asr_text>hello"
        )
    }

    func testMergedHiddenPostStopPreviewKeepsLongerBaseWhenCandidateIsContained() {
        let base = "文档目录结构都能被读取。你可以在文档列表中复制一份或多份文档，直接发起对话。"
        let candidate = "你可以在文档列表中复制一份或多份文档"

        let merged = MLXTranscriptionPlanning.mergedHiddenPostStopPreview(base: base, candidate: candidate)

        XCTAssertEqual(merged, base)
    }

    func testMergedHiddenPostStopPreviewAvoidsSuspiciousDuplicateGrowth() {
        let base = "比如写周报的时候，勾选本周的项目文档，让 AI 总结并更新到周报文档里。做 PPT 时，从资料库里挑素材，起稿就发起做一份某某汇报 PPT 的任务。"
        let candidate = "比如写周报的时候，勾选本周的项目文档，让 AI 总结并更新到周报文档里。做 PPT 时，从资料库里挑素材，起稿就发起做一份某某汇报 PPT 的任务，有了准确的上下文，生成的结果自然更贴近原文。"

        let merged = MLXTranscriptionPlanning.mergedHiddenPostStopPreview(base: base, candidate: candidate)

        XCTAssertEqual(merged, candidate)
    }

    func testMergedHiddenPostStopPreviewAvoidsConcatenatingLowOverlapFragments() {
        let base = "连接 Work Body 成功后，腾讯文档里的所有内容和目录结构都能被读取。你可以在文档列表中复制一份或多份文档，直接发起对话。"
        let candidate = "文档目录结构都能被读取。你可以在文档列表中复制一份或多份文档，直接发起对话，或者在新的任务里添加文档。AI 就能基于这些文档做出真实的内容思考和输出。"

        let merged = MLXTranscriptionPlanning.mergedHiddenPostStopPreview(base: base, candidate: candidate)

        XCTAssertEqual(merged, candidate)
        XCTAssertFalse(merged.contains("连接 Work Body 成功后，腾讯文档里的所有内容和目录结构都能被读取。你可以在文档列表中复制一份或多份文档，直接发起对话。 文档目录结构都能被读取。"))
    }

    func testMergedHiddenPostStopPreviewConcatenatesDisjointSentenceFragments() {
        let base = "比如写周报的时候，勾选本周的项目文档，让AI总结并更新到周报文档里。"
        let candidate = "做PPT时，从资料库里挑素材，起稿就发起一份“某某汇报PPT”的任务。有了准确的上下文，生成的结果自然更贴近原文。"

        let merged = MLXTranscriptionPlanning.mergedHiddenPostStopPreview(base: base, candidate: candidate)

        XCTAssertEqual(
            merged,
            "比如写周报的时候，勾选本周的项目文档，让AI总结并更新到周报文档里。做PPT时，从资料库里挑素材，起稿就发起一份“某某汇报PPT”的任务。有了准确的上下文，生成的结果自然更贴近原文。"
        )
    }

    func testMergedHiddenPostStopPreviewStitchesContinuationWhenCandidateStartsWithMinorNoise() {
        let base = "比如写作报的时候，勾选本周的项目文档，让AI总结并更新到周报文档里。"
        let candidate = "来总结并更新到周报文档里。做PPT时，从资料库里挑选素材，起稿就可以发一份“某某汇报PPT”的任务。有了准确的上下文，生成的结果自然更贴近原文。"

        let merged = MLXTranscriptionPlanning.mergedHiddenPostStopPreview(base: base, candidate: candidate)

        XCTAssertEqual(
            merged,
            "比如写作报的时候，勾选本周的项目文档，让AI总结并更新到周报文档里。做PPT时，从资料库里挑选素材，起稿就可以发一份“某某汇报PPT”的任务。有了准确的上下文，生成的结果自然更贴近原文。"
        )
    }
}
