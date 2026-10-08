// RecordingSessionSupport.swift
// Provides Recording Session Support for recording session routing.

import Foundation

enum RecordingSessionSupport {
    static func outputLabel(for outputMode: SessionOutputMode) -> String {
        switch outputMode {
        case .transcription:
            return "transcription"
        case .translation:
            return "translation"
        case .rewrite:
            return "rewrite"
        }
    }

    static func textModelRoutingDescription(
        outputMode: SessionOutputMode,
        transcriptionSettings: TranscriptionFeatureSettings,
        translationSettings: TranslationFeatureSettings,
        rewriteSettings: RewriteFeatureSettings
    ) -> String {
        switch outputMode {
        case .transcription:
            guard transcriptionSettings.llmEnabled else {
                return "transcription: none"
            }
            switch transcriptionSettings.llmSelectionID.textSelection {
            case .appleIntelligence:
                return "transcription: apple-intelligence"
            case .localLLM(let repo):
                return "transcription: local-llm(\(repo))"
            case .remoteLLM(let provider):
                return "transcription: remote-llm(\(provider.rawValue))"
            case .none:
                return "transcription: none"
            }
        case .translation:
            switch translationSettings.modelSelectionID.translationSelection {
            case .localLLM(let repo):
                return "translation: local-llm(\(repo))"
            case .localGGUF(let modelID):
                return "translation: local-gguf(\(modelID.rawValue))"
            case .remoteLLM(let provider):
                return "translation: remote-llm(\(provider.rawValue))"
            case .none:
                return "translation: none"
            }
        case .rewrite:
            switch rewriteSettings.llmSelectionID.textSelection {
            case .appleIntelligence:
                return "rewrite: apple-intelligence"
            case .localLLM(let repo):
                return "rewrite: local-llm(\(repo))"
            case .remoteLLM(let provider):
                return "rewrite: remote-llm(\(provider.rawValue))"
            case .none:
                return "rewrite: none"
            }
        }
    }

    static func overlayIconMode(
        for outputMode: SessionOutputMode,
        isNoteSession: Bool = false
    ) -> OverlaySessionIconMode {
        switch outputMode {
        case .transcription:
            return isNoteSession ? .note : .transcription
        case .translation:
            return .translation
        case .rewrite:
            return .rewrite
        }
    }

    static func fallbackInjectBundleID(
        from bundleID: String?,
        ownBundleID: String?
    ) -> String? {
        guard let bundleID,
              let ownBundleID,
              bundleID != ownBundleID
        else {
            return nil
        }
        return bundleID
    }

    static func normalizedTranscriptionDisplayText(
        _ rawText: String,
        transcriptionEngine: TranscriptionEngine,
        remoteProvider: RemoteASRProvider,
        userMainLanguage: UserMainLanguageOption
    ) -> String {
        let trimmed = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        let extractedText: String
        if transcriptionEngine == .remote, remoteProvider == .openAIWhisper {
            guard (trimmed.hasPrefix("{") && trimmed.hasSuffix("}")) ||
                  (trimmed.hasPrefix("[") && trimmed.hasSuffix("]")) else {
                extractedText = trimmed
                let normalized = ChineseScriptNormalizer.normalize(extractedText, preferredMainLanguage: userMainLanguage)
                return textAfterSuppressingPromptEcho(normalized)
            }

            guard let data = trimmed.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data),
                  let extracted = extractTranscriptionTextValue(from: object),
                  !extracted.isEmpty else {
                extractedText = trimmed
                let normalized = ChineseScriptNormalizer.normalize(extractedText, preferredMainLanguage: userMainLanguage)
                return textAfterSuppressingPromptEcho(normalized)
            }
            extractedText = extracted
        } else {
            extractedText = trimmed
        }

        let normalized = ChineseScriptNormalizer.normalize(extractedText, preferredMainLanguage: userMainLanguage)
        return textAfterSuppressingPromptEcho(normalized)
    }

    nonisolated static func textAfterSuppressingPromptEcho(_ text: String, prompt: String? = nil) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        return isLikelyPromptEcho(trimmed, prompt: prompt) ? "" : trimmed
    }

    nonisolated static func isLikelyPromptEcho(_ text: String, prompt: String? = nil) -> Bool {
        let normalized = normalizedPromptEchoDetectionText(text)
        guard normalized.count >= 24 else { return false }

        if let prompt {
            let promptKey = normalizedPromptEchoComparisonKey(prompt)
            let textKey = normalizedPromptEchoComparisonKey(text)
            if textKey.count >= 40,
               promptKey.count >= 40,
               (promptKey.contains(textKey) || textKey.contains(promptKey)) {
                return true
            }
        }

        let strongMarkers = [
            "[system_prompt]",
            "[request_content]",
            "process this asr transcription",
            "return only the final processed text",
            "你是 voxt 的转写清理助手",
            "请严格按优先级执行以下规则",
            "请直接输出清理后的文本"
        ]
        if strongMarkers.contains(where: normalized.contains) {
            return true
        }

        let markerHits = [
            "the speaker's primary language is",
            "mixed-language speech is expected",
            "preserve names, product terms",
            "primary language:",
            "other frequently used languages:",
            "mixed-language speech may appear",
            "preserve names, brands",
            "language rule:",
            "user main language:"
        ].filter { normalized.contains($0) }.count

        return markerHits >= 2
    }

    static func extractTranscriptionTextValue(from object: Any) -> String? {
        if let text = object as? String {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }

        if let dict = object as? [String: Any] {
            let preferredKeys = ["text", "transcript", "result_text", "utterance", "content", "data"]
            for key in preferredKeys {
                if let value = dict[key],
                   let extracted = extractTranscriptionTextValue(from: value),
                   !extracted.isEmpty {
                    return extracted
                }
            }

            for value in dict.values {
                if let extracted = extractTranscriptionTextValue(from: value),
                   !extracted.isEmpty {
                    return extracted
                }
            }
            return nil
        }

        if let array = object as? [Any] {
            for item in array {
                if let extracted = extractTranscriptionTextValue(from: item),
                   !extracted.isEmpty {
                    return extracted
                }
            }
        }

        return nil
    }

    private nonisolated static func normalizedPromptEchoDetectionText(_ text: String) -> String {
        text
            .folding(options: [.diacriticInsensitive, .widthInsensitive], locale: nil)
            .lowercased()
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
    }

    private nonisolated static func normalizedPromptEchoComparisonKey(_ text: String) -> String {
        normalizedPromptEchoDetectionText(text)
            .filter { !$0.isWhitespace }
    }
}
