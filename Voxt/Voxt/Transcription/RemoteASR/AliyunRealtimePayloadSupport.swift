import Foundation

enum AliyunQwenRealtimeSessionKind: Equatable {
    case qwenASR
    case omniASR

    var transcriptionModel: String? {
        switch self {
        case .qwenASR:
            return nil
        case .omniASR:
            return "qwen3-asr-flash-realtime"
        }
    }

    var shouldCommitBeforeFinish: Bool {
        switch self {
        case .qwenASR:
            return false
        case .omniASR:
            return false
        }
    }
}

enum AliyunQwenRealtimePayloadSupport {
    static func sessionUpdatePayload(
        kind: AliyunQwenRealtimeSessionKind,
        hintPayload: ResolvedASRHintPayload,
        includesTurnDetection: Bool = true,
        settings: AliyunASRModelSettings = AliyunASRModelSettings()
    ) -> [String: Any] {
        var transcriptionPayload: [String: Any] = [:]
        if let transcriptionModel = kind.transcriptionModel {
            transcriptionPayload["model"] = transcriptionModel
        }
        if let language = hintPayload.language?.trimmingCharacters(in: .whitespacesAndNewlines), !language.isEmpty {
            transcriptionPayload["language"] = language
        }
        var session: [String: Any] = [
            "modalities": ["text"],
            "input_audio_format": "pcm",
            "sample_rate": 16000,
            "input_audio_transcription": transcriptionPayload
        ]
        if includesTurnDetection && !(kind == .qwenASR && settings.useManualCommit) {
            session["turn_detection"] = [
                "type": "server_vad",
                "threshold": min(max(settings.serverVADThreshold, -1), 1),
                "silence_duration_ms": min(max(settings.serverVADSilenceDurationMilliseconds, 200), 6000)
            ]
        }
        return [
            "event_id": UUID().uuidString.lowercased(),
            "type": "session.update",
            "session": session
        ]
    }
}

enum AliyunFunRealtimePayloadSupport {
    static func supportsContext(model: String) -> Bool {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized.hasPrefix("qwen-audio-3.0-asr-flash-streaming")
            || normalized == "fun-asr-realtime"
            || normalized == "fun-asr-realtime-2025-11-07"
    }

    static func context(model: String, phrases: [String]) -> [[String: Any]] {
        guard supportsContext(model: model) else { return [] }
        let text = phrases
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        let truncated = String(text.prefix(400))
        guard !truncated.isEmpty else { return [] }
        return [[
            "role": "user",
            "content": [[
                "type": "input_text",
                "text": truncated
            ]]
        ]]
    }

    static func parameters(
        model: String,
        hintPayload: ResolvedASRHintPayload,
        settings: AliyunASRModelSettings = AliyunASRModelSettings(),
        includeHotwords: Bool = true
    ) -> [String: Any] {
        let capabilities = AliyunASRModelCapabilities.forModel(model)
        var parameters: [String: Any] = [
            "sample_rate": 16000,
            "format": "pcm"
        ]
        if capabilities.supportsLanguageHints, !hintPayload.languageHints.isEmpty {
            let languageHints = Array(hintPayload.languageHints.prefix(capabilities.maximumLanguageHints))
            if !languageHints.isEmpty {
                parameters["language_hints"] = languageHints
            }
        }
        if includeHotwords,
           capabilities.supportsInlineVocabulary,
           !hintPayload.contextualPhrases.isEmpty {
            var vocabulary: [String: Int] = [:]
            for phrase in hintPayload.contextualPhrases {
                let term = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !term.isEmpty else { continue }
                vocabulary[term] = 4
            }
            if !vocabulary.isEmpty {
                parameters["vocabulary"] = vocabulary
            }
        }
        if capabilities.supportsMaxSentenceSilence {
            parameters["max_sentence_silence"] = min(max(settings.maxSentenceSilenceMilliseconds, 200), 6000)
        }
        if capabilities.supportsServerVAD {
            parameters["speech_noise_threshold"] = min(max(settings.serverVADThreshold, -1), 1)
        }
        if capabilities.supportsSemanticPunctuation {
            parameters["semantic_punctuation_enabled"] = settings.semanticPunctuationEnabled
        }
        if capabilities.supportsPunctuationPrediction {
            parameters["punctuation_prediction_enabled"] = settings.punctuationPredictionEnabled
        }
        if capabilities.supportsInverseTextNormalization {
            parameters["inverse_text_normalization_enabled"] = settings.inverseTextNormalizationEnabled
        }
        if capabilities.supportsDisfluencyRemoval {
            parameters["disfluency_removal_enabled"] = settings.disfluencyRemovalEnabled
        }
        return parameters
    }
}
