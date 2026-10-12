import Foundation
import SwiftUI

extension RemoteProviderConfigurationSheet {
    var currentConfigurationSnapshot: RemoteProviderConfiguration {
        let snapshot = RemoteProviderConfiguration(
            providerID: configuration.providerID,
            model: resolvedModelValue(),
            endpoint: isDoubaoASRTest ? "" : endpoint.trimmingCharacters(in: .whitespacesAndNewlines),
            apiKey: (isDoubaoASRTest || isCodexLLMProvider) ? "" : apiKey.trimmingCharacters(in: .whitespacesAndNewlines),
            appID: appID.trimmingCharacters(in: .whitespacesAndNewlines),
            accessToken: accessToken.trimmingCharacters(in: .whitespacesAndNewlines),
            searchEnabled: (llmProviderForPicker?.supportsHostedSearch == true) ? searchEnabled : false,
            openAIChunkPseudoRealtimeEnabled: isOpenAIASRTest ? openAIChunkPseudoRealtimeEnabled : false,
            openAIReasoningEffort: usesOpenAIResponsesOptions ? openAIReasoningEffortSnapshot() : OpenAIReasoningEffort.automatic.rawValue,
            openAITextVerbosity: usesOpenAIResponsesOptions ? openAITextVerbosity : OpenAITextVerbosity.automatic.rawValue,
            openAIMaxOutputTokens: usesOpenAIResponsesOptions ? parsedOptionalInt(generationMaxOutputTokensText) : nil,
            doubaoDictionaryMode: doubaoDictionaryMode,
            doubaoEnableRequestHotwords: doubaoEnableRequestHotwords,
            doubaoEnableRequestCorrections: doubaoEnableRequestCorrections,
            ollamaResponseFormat: isOllamaLLMProvider ? ollamaResponseFormatSnapshot() : ollamaResponseFormat,
            ollamaJSONSchema: ollamaJSONSchema.trimmingCharacters(in: .whitespacesAndNewlines),
            ollamaThinkMode: isOllamaLLMProvider ? ollamaThinkModeSnapshot() : ollamaThinkMode,
            ollamaKeepAlive: ollamaKeepAlive.trimmingCharacters(in: .whitespacesAndNewlines),
            ollamaLogprobsEnabled: isOllamaLLMProvider ? generationLogprobsEnabled : ollamaLogprobsEnabled,
            ollamaTopLogprobs: isOllamaLLMProvider ? parsedOptionalInt(generationTopLogprobsText) : parsedOptionalInt(ollamaTopLogprobsText),
            ollamaOptionsJSON: isOllamaLLMProvider ? generationExtraOptionsJSON.trimmingCharacters(in: .whitespacesAndNewlines) : ollamaOptionsJSON.trimmingCharacters(in: .whitespacesAndNewlines),
            omlxResponseFormat: isOMLXLLMProvider ? omlxResponseFormatSnapshot() : omlxResponseFormat,
            omlxJSONSchema: omlxJSONSchema.trimmingCharacters(in: .whitespacesAndNewlines),
            omlxIncludeUsageStreamOptions: omlxIncludeUsageStreamOptions,
            omlxExtraBodyJSON: isOMLXLLMProvider ? generationExtraBodyJSON.trimmingCharacters(in: .whitespacesAndNewlines) : omlxExtraBodyJSON.trimmingCharacters(in: .whitespacesAndNewlines),
            codexAuthFilePath: isCodexLLMProvider ? codexAuthFilePath.trimmingCharacters(in: .whitespacesAndNewlines) : configuration.codexAuthFilePath,
            codexAuthFileBookmark: isCodexLLMProvider ? codexAuthFileBookmark : configuration.codexAuthFileBookmark,
            codexFastModeEnabled: isCodexLLMProvider ? codexFastModeEnabled : configuration.codexFastModeEnabled,
            aliyunASRSettings: currentAliyunASRSettingsSnapshot(),
            generationSettings: currentGenerationSettingsSnapshot()
        )
        return snapshot.applyingCredentialEditIntent(
            from: configuration,
            editedFields: editedCredentialFields
        )
    }

    func currentAliyunASRSettingsSnapshot() -> AliyunASRModelSettings {
        guard isAliyunASRProvider else {
            return configuration.aliyunASRSettings
        }
        return AliyunASRModelSettings(
            maxSentenceSilenceMilliseconds: clampedInt(aliyunMaxSentenceSilenceMillisecondsText, defaultValue: 1300, lowerBound: 200, upperBound: 6000),
            serverVADThreshold: clampedDouble(aliyunServerVADThresholdText, defaultValue: 0.35, lowerBound: -1, upperBound: 1),
            serverVADSilenceDurationMilliseconds: clampedInt(aliyunServerVADSilenceDurationMillisecondsText, defaultValue: 800, lowerBound: 200, upperBound: 6000),
            useManualCommit: aliyunUseManualCommit,
            semanticPunctuationEnabled: aliyunSemanticPunctuationEnabled,
            punctuationPredictionEnabled: aliyunPunctuationPredictionEnabled,
            inverseTextNormalizationEnabled: aliyunInverseTextNormalizationEnabled,
            disfluencyRemovalEnabled: aliyunDisfluencyRemovalEnabled
        )
    }

    private func clampedInt(_ text: String, defaultValue: Int, lowerBound: Int, upperBound: Int) -> Int {
        min(max(Int(text.trimmingCharacters(in: .whitespacesAndNewlines)) ?? defaultValue, lowerBound), upperBound)
    }

    private func clampedDouble(_ text: String, defaultValue: Double, lowerBound: Double, upperBound: Double) -> Double {
        min(max(Double(text.trimmingCharacters(in: .whitespacesAndNewlines)) ?? defaultValue, lowerBound), upperBound)
    }

    func currentGenerationSettingsSnapshot() -> LLMGenerationSettings {
        guard let provider = llmProviderForPicker else {
            return configuration.generationSettings
        }
        let capabilities = LLMProviderCapabilityRegistry.capabilities(for: provider)
        var settings = LLMGenerationSettings()
        if isStepFunLLMProvider {
            settings.thinking = .off
        }
        settings.maxOutputTokens = capabilities.supportsMaxOutputTokens ? parsedOptionalInt(generationMaxOutputTokensText) : nil
        settings.temperature = capabilities.supportsTemperature ? parsedOptionalDouble(generationTemperatureText) : nil
        settings.topP = capabilities.supportsTopP ? parsedOptionalDouble(generationTopPText) : nil
        settings.topK = capabilities.supportsTopK ? parsedOptionalInt(generationTopKText) : nil
        settings.minP = capabilities.supportsMinP ? parsedOptionalDouble(generationMinPText) : nil
        settings.seed = capabilities.supportsSeed ? parsedOptionalInt(generationSeedText) : nil
        settings.stop = capabilities.supportsStopSequences ? parsedStopSequences() : []
        if capabilities.supportsPenalties {
            settings.frequencyPenalty = parsedOptionalDouble(generationFrequencyPenaltyText)
            if !isStepFunLLMProvider {
                settings.presencePenalty = parsedOptionalDouble(generationPresencePenaltyText)
                settings.repetitionPenalty = parsedOptionalDouble(generationRepetitionPenaltyText)
            }
        }
        if capabilities.supportsLogprobs {
            settings.logprobs = generationLogprobsEnabled
            settings.topLogprobs = parsedOptionalInt(generationTopLogprobsText)
        }
        if capabilities.supportsResponseFormat {
            settings.responseFormat = LLMResponseFormat(rawValue: generationResponseFormat) ?? .plain
        }
        if shouldShowGenerationThinking {
            settings.thinking = LLMThinkingSettings(
                mode: sanitizedGenerationThinkingMode,
                effort: normalizedOptionalString(generationThinkingEffort),
                budgetTokens: parsedOptionalInt(generationThinkingBudgetText),
                exposeReasoning: false
            )
        }
        if capabilities.supportsExtraBody {
            settings.extraBodyJSON = generationExtraBodyJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if capabilities.supportsExtraOptions {
            settings.extraOptionsJSON = generationExtraOptionsJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return settings
    }

    func configureGenerationSettingsState() {
        let provider = llmProviderForPicker
        let settings = provider.map { configuration.effectiveGenerationSettings(provider: $0) } ?? configuration.generationSettings
        generationMaxOutputTokensText = settings.maxOutputTokens.map(String.init) ?? ""
        generationTemperatureText = settings.temperature.map(Self.formatOptionalDouble) ?? ""
        generationTopPText = settings.topP.map(Self.formatOptionalDouble) ?? ""
        generationTopKText = settings.topK.map(String.init) ?? ""
        generationMinPText = settings.minP.map(Self.formatOptionalDouble) ?? ""
        generationSeedText = settings.seed.map(String.init) ?? ""
        generationStopText = settings.stop.joined(separator: "\n")
        generationPresencePenaltyText = settings.presencePenalty.map(Self.formatOptionalDouble) ?? ""
        generationFrequencyPenaltyText = settings.frequencyPenalty.map(Self.formatOptionalDouble) ?? ""
        generationRepetitionPenaltyText = settings.repetitionPenalty.map(Self.formatOptionalDouble) ?? ""
        generationLogprobsEnabled = settings.logprobs
        generationTopLogprobsText = settings.topLogprobs.map(String.init) ?? ""
        generationResponseFormat = settings.responseFormat.rawValue
        generationThinkingMode = settings.thinking.mode.rawValue
        generationThinkingEffort = settings.thinking.effort ?? ""
        generationThinkingBudgetText = settings.thinking.budgetTokens.map(String.init) ?? ""
        generationExtraBodyJSON = settings.extraBodyJSON
        generationExtraOptionsJSON = settings.extraOptionsJSON
    }

    func parsedOptionalInt(_ text: String) -> Int? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return Int(trimmed)
    }

    func parsedOptionalDouble(_ text: String) -> Double? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return Double(trimmed)
    }

    func parsedStopSequences() -> [String] {
        generationStopText
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    func normalizedOptionalString(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    func openAIReasoningEffortSnapshot() -> String {
        guard LLMThinkingMode(rawValue: generationThinkingMode) == .effort,
              let effort = normalizedOptionalString(generationThinkingEffort),
              OpenAIReasoningEffort(rawValue: effort) != nil
        else {
            return OpenAIReasoningEffort.automatic.rawValue
        }
        return effort
    }

    func ollamaResponseFormatSnapshot() -> String {
        switch LLMResponseFormat(rawValue: generationResponseFormat) {
        case .json:
            return OllamaResponseFormat.json.rawValue
        case .jsonSchema:
            return OllamaResponseFormat.jsonSchema.rawValue
        case .plain, nil:
            return OllamaResponseFormat.plain.rawValue
        }
    }

    func omlxResponseFormatSnapshot() -> String {
        switch LLMResponseFormat(rawValue: generationResponseFormat) {
        case .jsonSchema:
            return OMLXResponseFormat.jsonSchema.rawValue
        case .plain, .json, nil:
            return OMLXResponseFormat.plain.rawValue
        }
    }

    func ollamaThinkModeSnapshot() -> String {
        switch LLMThinkingMode(rawValue: generationThinkingMode) {
        case .off:
            return OllamaThinkMode.off.rawValue
        case .on, .budget:
            return OllamaThinkMode.on.rawValue
        case .effort:
            switch generationThinkingEffort {
            case OllamaThinkMode.low.rawValue:
                return OllamaThinkMode.low.rawValue
            case OllamaThinkMode.medium.rawValue:
                return OllamaThinkMode.medium.rawValue
            case OllamaThinkMode.high.rawValue:
                return OllamaThinkMode.high.rawValue
            default:
                return OllamaThinkMode.on.rawValue
            }
        case .providerDefault, nil:
            return OllamaThinkMode.off.rawValue
        }
    }

    nonisolated static func formatOptionalDouble(_ value: Double) -> String {
        String(format: "%g", value)
    }
}
