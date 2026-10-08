import Foundation
import MLXAudioSTT

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
        var chunkDuration: Float
        var minChunkDuration: Float
        if capability.configurationCapabilities.contains(.recognitionPreset) {
            switch tuningSettings.preset {
            case .balanced:
                chunkDuration = 1200
                minChunkDuration = 1
            case .accuracyFirst:
                chunkDuration = 90
                minChunkDuration = 2.5
            }
        } else {
            // Models without recognitionPreset (for example Whisper's fixed window) ignore
            // leftover preset values so stale settings cannot change decoding windows.
            chunkDuration = 1200
            minChunkDuration = 1
        }
        if stage == .postStopFinal {
            chunkDuration = MLXTranscriptionPlanning.postStopFinalChunkDuration(
                presetChunkDuration: chunkDuration
            )
            minChunkDuration = min(minChunkDuration, 1)
        }

        var languageHint = hintPayload.language
        switch family {
        case .mossTranscribeDiarize, .parakeet:
            languageHint = nil
        default:
            break
        }

        let stageMaxTokens: Int
        switch stage {
        case .intermediate:
            stageMaxTokens = 1024
        case .postStopQuick:
            stageMaxTokens = sessionAllowsRealtimeTextDisplay ? 1024 : 512
        case .postStopFinal:
            if let audioDurationSeconds {
                stageMaxTokens = MLXTranscriptionPlanning.postStopFinalMaxTokens(
                    audioDurationSeconds: audioDurationSeconds
                )
            } else {
                stageMaxTokens = 8192
            }
        }
        let maxTokens: Int
        let temperature: Float
        let usePunctuation: Bool?
        switch family {
        case .cohereTranscribe:
            // P3: Cohere keeps tuning budget on all stages (including Final).
            maxTokens = tuningSettings.cohereMaxTokens
            temperature = Float(tuningSettings.cohereTemperature)
            usePunctuation = tuningSettings.cohereUsePunctuation
        default:
            maxTokens = stageMaxTokens
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
            generationParameters: STTGenerateParameters(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: 0.95,
                topK: 0,
                verbose: false,
                language: languageHint,
                usePunctuation: usePunctuation,
                chunkDuration: chunkDuration,
                minChunkDuration: minChunkDuration,
                kvBits: kvCachePolicy?.bits,
                kvGroupSize: kvCachePolicy?.groupSize ?? 64,
                quantizedKVStart: kvCachePolicy?.quantizedStart ?? 0
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
