import Foundation

/// Value-only inference planning. The transcriber owns session/model resources;
/// this policy receives an explicit snapshot and never reads mutable defaults.
enum MLXInferenceConfiguration {
    typealias ResolvedInferenceConfiguration = MLXTranscriber.ResolvedInferenceConfiguration

    static func resolve(
        for stage: MLXCorrectionPassKind,
        audioDurationSeconds: Double?,
        hintPayload: ResolvedASRHintPayload,
        tuningSettings: MLXLocalTuningSettings,
        transcriptionPurpose: MLXTranscriptionPurpose,
        userLanguageCodes: [String],
        capability: MLXASRModelCapability,
        dictionaryTerms: String,
        sessionAllowsRealtimeTextDisplay: Bool
    ) -> ResolvedInferenceConfiguration {
        let mossSettings = tuningSettings.mossSettings(for: transcriptionPurpose.mossUsageScope)
        let mossGenerationOutputMode = MossASRPromptSupport.generationOutputMode(
            requestedOutputMode: mossSettings.outputMode,
            scope: transcriptionPurpose.mossUsageScope
        )
        let family = capability.family
        var chunkDurationSeconds: Float
        var minimumChunkDurationSeconds: Float
        if capability.configurationCapabilities.contains(.recognitionPreset) {
            switch tuningSettings.preset {
            case .balanced:
                chunkDurationSeconds = 1200
                minimumChunkDurationSeconds = 1
            case .accuracyFirst:
                chunkDurationSeconds = 90
                minimumChunkDurationSeconds = 2.5
            }
        } else {
            // Models without recognitionPreset (for example Whisper's fixed window) ignore
            // leftover preset values so stale settings cannot change decoding windows.
            chunkDurationSeconds = 1200
            minimumChunkDurationSeconds = 1
        }
        if stage == .postStopFinal {
            chunkDurationSeconds = MLXTranscriptionPlanning.postStopFinalChunkDuration(
                presetChunkDuration: chunkDurationSeconds
            )
            minimumChunkDurationSeconds = min(minimumChunkDurationSeconds, 1)
        }

        var languageHint = hintPayload.language
        switch family {
        case .mossTranscribeDiarize, .parakeet:
            languageHint = nil
        default:
            break
        }

        let stageMaximumTokenCount: Int
        switch stage {
        case .intermediate:
            stageMaximumTokenCount = 1024
        case .postStopQuick:
            stageMaximumTokenCount = sessionAllowsRealtimeTextDisplay ? 1024 : 512
        case .postStopFinal:
            if let audioDurationSeconds {
                stageMaximumTokenCount = MLXTranscriptionPlanning.postStopFinalMaxTokens(
                    audioDurationSeconds: audioDurationSeconds
                )
            } else {
                stageMaximumTokenCount = 8192
            }
        }
        let maximumTokenCount: Int
        let temperature: Float
        let usePunctuation: Bool?
        switch family {
        case .cohereTranscribe:
            // P3: Cohere keeps tuning budget on all stages (including Final).
            maximumTokenCount = tuningSettings.cohereMaxTokens
            temperature = Float(tuningSettings.cohereTemperature)
            usePunctuation = tuningSettings.cohereUsePunctuation
        default:
            maximumTokenCount = stageMaximumTokenCount
            temperature = family == .whisper ? Float(tuningSettings.whisperTemperature) : 0.0
            usePunctuation = nil
        }

        let kvCachePolicy: MLXASRKVCachePolicy?
        if stage == .postStopFinal {
            kvCachePolicy = MLXTranscriptionPlanning.postStopFinalKVCachePolicy(
                family: family,
                catalogPolicy: capability.kvCachePolicy
            )
        } else {
            kvCachePolicy = capability.kvCachePolicy
        }

        return ResolvedInferenceConfiguration(
            family: family,
            generationParameters: ASRGenerationOptions(
                maximumTokenCount: maximumTokenCount,
                temperature: temperature,
                topProbabilityThreshold: 0.95,
                topTokenCount: 0,
                verbose: false,
                language: languageHint,
                usePunctuation: usePunctuation,
                chunkDurationSeconds: chunkDurationSeconds,
                minimumChunkDurationSeconds: minimumChunkDurationSeconds,
                kvCacheBitWidth: kvCachePolicy?.bits,
                kvCacheGroupSize: kvCachePolicy?.groupSize ?? 64,
                kvCacheQuantizedStartTokenCount: kvCachePolicy?.quantizedStart ?? 0
            ),
            languageHint: languageHint,
            timingGranularity: capability.timingGranularity,
            qwenContextBias: resolvedBiasTemplate(
                tuningSettings.qwenContextBias,
                userLanguageCodes: userLanguageCodes,
                dictionaryTerms: dictionaryTerms
            ),
            senseVoiceUseITN: tuningSettings.senseVoiceUseITN,
            cohereLongFormStrategy: tuningSettings.cohereLongFormStrategy,
            mossPrompt: family == .mossTranscribeDiarize
                ? MossASRPromptSupport.resolvedPrompt(
                    requestedOutputMode: mossSettings.outputMode,
                    scope: transcriptionPurpose.mossUsageScope,
                    customPrompt: resolvedBiasTemplate(
                        mossSettings.customPrompt,
                        userLanguageCodes: userLanguageCodes,
                        dictionaryTerms: dictionaryTerms
                    ),
                    hotwords: MLXTranscriptionPlanning.shouldIncludeMOSSHotwords(for: stage)
                        ? resolvedBiasTemplate(
                            mossSettings.hotwords,
                            userLanguageCodes: userLanguageCodes,
                            dictionaryTerms: dictionaryTerms
                        )
                        : ""
                )
                : nil,
            mossOutputMode: mossGenerationOutputMode
        )
    }

    private static func resolvedBiasTemplate(
        _ template: String,
        userLanguageCodes: [String],
        dictionaryTerms: String
    ) -> String {
        ASRHintResolver.resolveTemplateVariables(
            in: template,
            userLanguageCodes: userLanguageCodes,
            dictionaryTerms: dictionaryTerms
        )
        .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
