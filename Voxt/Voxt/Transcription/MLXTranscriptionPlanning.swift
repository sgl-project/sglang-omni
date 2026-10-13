// Pure transcription decisions; no capture, task, or model ownership.

import Foundation

enum MLXTranscriptionPlanning {
    nonisolated static func shouldUseSenseVoiceVAD(
        sampleCount: Int,
        sampleRate: Int,
        directPassMaximumDurationSeconds: Double
    ) -> Bool {
        let safeSampleRate = max(sampleRate, 1)
        let durationSeconds = Double(sampleCount) / Double(safeSampleRate)
        return durationSeconds > directPassMaximumDurationSeconds
    }

    nonisolated static func splitSenseVoiceRange(
        start: Int,
        end: Int,
        maxChunkSamples: Int,
        overlapSamples: Int
    ) -> [Range<Int>] {
        guard maxChunkSamples > 0, end - start > maxChunkSamples else {
            return start < end ? [start..<end] : []
        }

        var ranges: [Range<Int>] = []
        var cursor = start
        while cursor < end {
            let upperBound = min(cursor + maxChunkSamples, end)
            ranges.append(cursor..<upperBound)
            guard upperBound < end else { break }
            cursor = max(start, upperBound - overlapSamples)
        }
        return ranges
    }

    /// Whether dictation Final may replace full PCM with VAD-filtered speech.
    /// SenseVoice keeps catalog `.standard` (meeting validation) but must not also
    /// externally trim PCM before its own long-form Silero segmentation.
    nonisolated static func allowsExternalFinalSpeechTrim(
        vadPolicy: MLXVADPolicy,
        family: MLXModelFamily
    ) -> Bool {
        if family == .senseVoice {
            return false
        }
        return vadPolicy.allowsExternalFinalSpeechTrim
    }

    nonisolated static func finalizationSamples(
        fullSamples: [Float],
        voiceActivityFilteredSamples: [Float],
        localVADGateActive: Bool,
        observedVoiceActivityFrames: Bool,
        observedSpeech: Bool,
        vadPolicy: MLXVADPolicy = .standard,
        family: MLXModelFamily? = nil
    ) -> MLXFinalizationSampleSelection {
        let allowsTrim = family.map {
            allowsExternalFinalSpeechTrim(vadPolicy: vadPolicy, family: $0)
        } ?? vadPolicy.allowsExternalFinalSpeechTrim

        // Timeline-sensitive / model-managed / SenseVoice dictation: VAD is only a no-speech gate.
        guard allowsTrim else {
            if localVADGateActive, observedVoiceActivityFrames, !observedSpeech {
                return MLXFinalizationSampleSelection(samples: [], source: .noSpeech)
            }
            return MLXFinalizationSampleSelection(samples: fullSamples, source: .full)
        }

        guard localVADGateActive, observedVoiceActivityFrames else {
            return MLXFinalizationSampleSelection(samples: fullSamples, source: .full)
        }
        guard observedSpeech else {
            return MLXFinalizationSampleSelection(samples: [], source: .noSpeech)
        }
        guard !voiceActivityFilteredSamples.isEmpty,
              voiceActivityFilteredSamples.count < fullSamples.count
        else {
            return MLXFinalizationSampleSelection(samples: fullSamples, source: .full)
        }
        return MLXFinalizationSampleSelection(
            samples: voiceActivityFilteredSamples,
            source: .voiceActivityFiltered
        )
    }

    /// Caps offline Final decode budget by audio length while keeping headroom for
    /// dense Chinese/English mixed speech.
    nonisolated static func postStopFinalMaxTokens(audioDurationSeconds: Double) -> Int {
        let safeDuration = max(0, audioDurationSeconds)
        // 28 tok/s covers dense CN/EN mixed speech better than 24; EOS still ends early.
        let estimated = Int(ceil(safeDuration * 28.0)) + 64
        return min(8192, max(256, estimated))
    }

    /// Offline Final prefers one decode window. Intermediate passes may still use the
    /// recognition-preset slice (e.g. accuracyFirst=90); Final lifts to the balanced window.
    nonisolated static func postStopFinalChunkDuration(presetChunkDuration: Float) -> Float {
        max(presetChunkDuration, 1200)
    }

    nonisolated static func postStopFinalKVCachePolicy(
        family: MLXModelFamily,
        catalogPolicy: MLXASRKVCachePolicy?
    ) -> MLXASRKVCachePolicy? {
        if family == .qwen3ASR {
            // Final must use the same conservative Qwen KV policy as live decoding.
            // The automatic-language path is especially sensitive to early prompt
            // quantization; overriding the catalog policy here can make Final lose
            // earlier multilingual content even when live decoding accumulated it.
            return catalogPolicy ?? .conservativeQwen
        }
        return catalogPolicy
    }

    nonisolated static func senseVoiceSegmentRanges(
        probabilities: [Float],
        sampleCount: Int,
        sampleRate: Int,
        probabilityFrameSampleCount: Int,
        vadThreshold: Float,
        vadMinSpeechDurationMs: Int,
        vadMinSilenceDurationMs: Int,
        vadSpeechPadMs: Int,
        maxChunkSamples: Int,
        overlapSamples: Int
    ) -> [Range<Int>] {
        guard sampleCount > 0,
              sampleRate > 0,
              probabilityFrameSampleCount > 0,
              !probabilities.isEmpty
        else {
            return []
        }

        let configuration = ASRVoiceActivityConfiguration(
            onsetProbabilityThreshold: vadThreshold,
            offsetProbabilityThreshold: max(vadThreshold - 0.15, 0.01),
            minSpeechSeconds: Double(max(vadMinSpeechDurationMs, 0)) / 1000,
            minSilenceSeconds: Double(max(vadMinSilenceDurationMs, 0)) / 1000,
            speechPadSeconds: Double(max(vadSpeechPadMs, 0)) / 1000,
            maxSegmentSeconds: nil
        )
        var segmenter = ASRVoiceActivitySegmenter(configuration: configuration)
        var segments: [ASRVoiceActivitySegment] = []

        for (index, probability) in probabilities.enumerated() {
            let startSample = min(index * probabilityFrameSampleCount, sampleCount)
            let endSample = min((index + 1) * probabilityFrameSampleCount, sampleCount)
            guard endSample > startSample else { break }

            let events = segmenter.append(
                ASRVoiceActivityFrameDecision(
                    startSeconds: Double(startSample) / Double(sampleRate),
                    endSeconds: Double(endSample) / Double(sampleRate),
                    isSpeech: false,
                    probability: probability
                )
            )
            for event in events {
                switch event {
                case .speechEnded(let segment), .speechForced(let segment):
                    segments.append(segment)
                case .speechStarted, .speechRejected:
                    break
                }
            }
        }

        if let finalEvent = segmenter.finish(at: Double(sampleCount) / Double(sampleRate)) {
            switch finalEvent {
            case .speechEnded(let segment), .speechForced(let segment):
                segments.append(segment)
            case .speechStarted, .speechRejected:
                break
            }
        }

        return segments.flatMap { segment -> [Range<Int>] in
            let start = max(0, min(Int(floor(segment.startSeconds * Double(sampleRate))), sampleCount))
            let end = max(start, min(Int(ceil(segment.endSeconds * Double(sampleRate))), sampleCount))
            guard end > start else { return [] }
            return splitSenseVoiceRange(
                start: start,
                end: end,
                maxChunkSamples: maxChunkSamples,
                overlapSamples: overlapSamples
            )
        }
    }

    static func correctionCadence(
        for repo: String,
        sessionAllowsRealtimeTextDisplay: Bool
    ) -> MLXCorrectionCadence {
        if MLXModelFamily.family(for: repo) == .senseVoice {
            if sessionAllowsRealtimeTextDisplay {
                return MLXCorrectionCadence(
                    correctionIntervalSeconds: 4.0,
                    firstCorrectionMinimumSeconds: 2.2,
                    intermediateContextWindowSeconds: 14.0,
                    quickPassContextWindowSeconds: 24.0
                )
            }
            return MLXCorrectionCadence(
                correctionIntervalSeconds: 2.6,
                firstCorrectionMinimumSeconds: 1.8,
                intermediateContextWindowSeconds: 18.0,
                quickPassContextWindowSeconds: 18.0
            )
        }

        if sessionAllowsRealtimeTextDisplay {
            return MLXCorrectionCadence(
                correctionIntervalSeconds: 6.0,
                firstCorrectionMinimumSeconds: 3.5,
                intermediateContextWindowSeconds: 18.0,
                quickPassContextWindowSeconds: 30.0
            )
        }
        return MLXCorrectionCadence(
            correctionIntervalSeconds: 3.2,
            firstCorrectionMinimumSeconds: 2.2,
            intermediateContextWindowSeconds: 24.0,
            quickPassContextWindowSeconds: 18.0
        )
    }

    static func intermediateCorrectionDecision(
        sampleCount: Int,
        sampleRate: Double,
        nextCorrectionAtSeconds: Double,
        behavior: MLXModelManager.TranscriptionBehavior,
        firstCorrectionMinimumSeconds: Double,
        contextWindowSeconds: Double
    ) -> MLXIntermediateCorrectionDecision? {
        guard behavior.runsIntermediateCorrections else { return nil }
        guard sampleCount > 0 else { return nil }

        let safeSampleRate = max(sampleRate, 1)
        let elapsedSeconds = Double(sampleCount) / safeSampleRate
        guard elapsedSeconds >= firstCorrectionMinimumSeconds else { return nil }
        guard elapsedSeconds >= nextCorrectionAtSeconds else { return nil }

        return MLXIntermediateCorrectionDecision(
            elapsedSeconds: elapsedSeconds,
            contextSampleCount: Int(contextWindowSeconds * safeSampleRate)
        )
    }

    static func finalizationPlan(
        sampleCount: Int,
        sampleRate: Double,
        behavior: MLXModelManager.TranscriptionBehavior,
        quickPassMinimumDurationSeconds: Double,
        quickPassContextWindowSeconds: Double
    ) -> MLXFinalizationPlan {
        let safeSampleRate = max(sampleRate, 1)
        let durationSeconds = Double(sampleCount) / safeSampleRate
        let quickPassSampleCount: Int?

        if behavior.allowsQuickStopPass, durationSeconds >= quickPassMinimumDurationSeconds {
            quickPassSampleCount = Int(quickPassContextWindowSeconds * safeSampleRate)
        } else {
            quickPassSampleCount = nil
        }

        return MLXFinalizationPlan(
            durationSeconds: durationSeconds,
            quickPassSampleCount: quickPassSampleCount
        )
    }

    static func shouldRunQuickStopPass(
        plan: MLXFinalizationPlan,
        sessionAllowsRealtimeTextDisplay: Bool,
        liveMode: MLXLiveMode
    ) -> Bool {
        guard sessionAllowsRealtimeTextDisplay else { return false }
        guard !Self.isNativeLiveMode(liveMode) else { return false }
        return plan.shouldRunQuickPass
    }

    static func isNativeLiveMode(_ liveMode: MLXLiveMode) -> Bool {
        switch liveMode {
        case .batchPreview:
            return false
        case .nativeQwenLive, .nativeStreamingLive, .nativeNemotronLive:
            return true
        }
    }

    static func correctionPassSchedulingDecision(
        requestedPass: MLXCorrectionPassKind,
        inFlightPass: MLXCorrectionPassKind?
    ) -> MLXCorrectionPassSchedulingDecision {
        guard let inFlightPass else { return .startImmediately }
        if requestedPass == .intermediate {
            return .skipRequestedPass
        }
        if inFlightPass == .intermediate {
            return .interruptInFlightPass
        }
        return .waitForInFlightPass
    }

    /// MOSS Hotwords stay on Final only. Live/intermediate windows are short and prompt-biased,
    /// so injecting the dictionary there commonly surfaces hotword hallucinations mid-stream.
    nonisolated static func shouldIncludeMOSSHotwords(for stage: MLXCorrectionPassKind) -> Bool {
        switch stage {
        case .intermediate:
            return false
        case .postStopQuick, .postStopFinal:
            return true
        }
    }
}
