import Foundation
import SwiftUI

extension RemoteProviderConfigurationSheet {
    func validationMessage() -> String? {
        let hasCredentials = switch testTarget {
        case .asr:
            RemoteEndpointSecurityPolicy.hasExplicitCredentials(currentConfigurationSnapshot)
        case .llm(let provider):
            RemoteEndpointSecurityPolicy.hasLLMCredentials(
                provider: provider,
                configuration: currentConfigurationSnapshot
            )
        }
        if let endpointMessage = RemoteEndpointSecurityPolicy.validationMessage(
            endpoint: endpoint,
            hasCredentials: hasCredentials,
            allowsWebSocket: {
                if case .asr = testTarget { return true }
                return false
            }()
        ) {
            return endpointMessage
        }
        if let generationMessage = validationMessageForGenerationSettings() {
            return generationMessage
        }
        if isOllamaLLMProvider {
            return validationMessageForOllamaSettings(
                responseFormat: ollamaResponseFormatSnapshot(),
                jsonSchema: ollamaJSONSchema,
                optionsJSON: generationExtraOptionsJSON,
                logprobsEnabled: generationLogprobsEnabled,
                topLogprobsText: generationTopLogprobsText
            )
        }
        if isOMLXLLMProvider {
            return validationMessageForOMLXSettings(
                responseFormat: omlxResponseFormatSnapshot(),
                jsonSchema: omlxJSONSchema,
                extraBodyJSON: generationExtraBodyJSON
            )
        }
        return nil
    }

    func validationMessageForGenerationSettings() -> String? {
        guard let capabilities = generationCapabilities else { return nil }
        var positiveIntFields = [(String, String)]()
        if capabilities.supportsMaxOutputTokens {
            positiveIntFields.append((generationMaxOutputTokensText, AppLocalization.localizedString("Max Output Tokens")))
        }
        if capabilities.supportsTopK {
            positiveIntFields.append((generationTopKText, AppLocalization.localizedString("Top K")))
        }
        if sanitizedGenerationThinkingMode == .budget {
            positiveIntFields.append((generationThinkingBudgetText, AppLocalization.localizedString("Thinking Budget")))
        }
        if sanitizedGenerationThinkingMode == .budget,
           generationThinkingBudgetText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return AppLocalization.format("%@ must be a positive integer.", AppLocalization.localizedString("Thinking Budget"))
        }
        if let message = validateTrimmedFields(
            positiveIntFields,
            message: "%@ must be a positive integer.",
            isValid: { Int($0).map { $0 > 0 } == true }
        ) {
            return message
        }

        var integerFields = [(String, String)]()
        if capabilities.supportsSeed {
            integerFields.append((generationSeedText, AppLocalization.localizedString("Seed")))
        }
        if capabilities.supportsLogprobs && generationLogprobsEnabled {
            integerFields.append((generationTopLogprobsText, AppLocalization.localizedString("Top Logprobs")))
        }
        if let message = validateTrimmedFields(
            integerFields,
            message: "%@ must be a non-negative integer.",
            isValid: { Int($0).map { $0 >= 0 } == true }
        ) {
            return message
        }

        var doubleFields = [(String, String)]()
        if capabilities.supportsTemperature {
            doubleFields.append((generationTemperatureText, AppLocalization.localizedString("Temperature")))
        }
        if capabilities.supportsTopP {
            doubleFields.append((generationTopPText, AppLocalization.localizedString("Top P")))
        }
        if capabilities.supportsMinP {
            doubleFields.append((generationMinPText, AppLocalization.localizedString("Min P")))
        }
        if capabilities.supportsPenalties {
            doubleFields.append((generationFrequencyPenaltyText, AppLocalization.localizedString("Frequency Penalty")))
            if !isStepFunLLMProvider {
                doubleFields.append((generationPresencePenaltyText, AppLocalization.localizedString("Presence Penalty")))
                doubleFields.append((generationRepetitionPenaltyText, AppLocalization.localizedString("Repetition Penalty")))
            }
        }
        if let message = validateTrimmedFields(
            doubleFields,
            message: "%@ must be a number.",
            isValid: { Double($0) != nil }
        ) {
            return message
        }

        if capabilities.supportsExtraBody,
           let extraBodyMessage = validateJSONObjectField(
               generationExtraBodyJSON,
               fieldName: AppLocalization.localizedString("Extra Body JSON"),
               requiresValue: false
           ) {
            return extraBodyMessage
        }
        if capabilities.supportsExtraOptions,
           let extraOptionsMessage = validateJSONObjectField(
               generationExtraOptionsJSON,
               fieldName: AppLocalization.localizedString("Options JSON"),
               requiresValue: false
           ) {
            return extraOptionsMessage
        }
        return nil
    }

    private func validateTrimmedFields(
        _ fields: [(String, String)],
        message: String,
        isValid: (String) -> Bool
    ) -> String? {
        for (text, fieldName) in fields {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            guard isValid(trimmed) else {
                return AppLocalization.format(message, fieldName)
            }
        }
        return nil
    }

    func shouldShowOllamaJSONSchemaField(for responseFormat: String) -> Bool {
        OllamaResponseFormat(rawValue: responseFormat) == .jsonSchema
    }

    func shouldShowOMLXJSONSchemaField(for responseFormat: String) -> Bool {
        OMLXResponseFormat(rawValue: responseFormat) == .jsonSchema
    }

    func validationMessageForOllamaSettings(
        responseFormat: String,
        jsonSchema: String,
        optionsJSON: String,
        logprobsEnabled: Bool,
        topLogprobsText: String
    ) -> String? {
        if let topLogprobsMessage = validateOllamaTopLogprobs(
            enabled: logprobsEnabled,
            text: topLogprobsText
        ) {
            return topLogprobsMessage
        }
        if let optionsMessage = validateJSONObjectField(
            optionsJSON,
            fieldName: AppLocalization.localizedString("Options JSON"),
            requiresValue: false
        ) {
            return optionsMessage
        }
        if shouldShowOllamaJSONSchemaField(for: responseFormat),
           let schemaMessage = validateJSONObjectField(
               jsonSchema,
               fieldName: AppLocalization.localizedString("JSON Schema"),
               requiresValue: true
           ) {
            return schemaMessage
        }
        return nil
    }

    func validationMessageForOMLXSettings(
        responseFormat: String,
        jsonSchema: String,
        extraBodyJSON: String
    ) -> String? {
        if let extraBodyMessage = validateJSONObjectField(
            extraBodyJSON,
            fieldName: AppLocalization.localizedString("Extra Body JSON"),
            requiresValue: false
        ) {
            return extraBodyMessage
        }
        if shouldShowOMLXJSONSchemaField(for: responseFormat),
           let schemaMessage = validateJSONObjectField(
               jsonSchema,
               fieldName: AppLocalization.localizedString("JSON Schema"),
               requiresValue: true
           ) {
            return schemaMessage
        }
        return nil
    }

    func validateOllamaTopLogprobs(enabled: Bool, text: String) -> String? {
        guard enabled else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard let value = Int(trimmed), value >= 0 else {
            return AppLocalization.localizedString("Top Logprobs must be a non-negative integer.")
        }
        return nil
    }

    func validateJSONObjectField(_ value: String, fieldName: String, requiresValue: Bool) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return requiresValue ? AppLocalization.format("%@ must be a JSON object.", fieldName) : nil
        }
        guard let data = trimmed.data(using: .utf8) else {
            return AppLocalization.format("%@ must be valid JSON.", fieldName)
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            object is [String: Any]
        else {
            return AppLocalization.format("%@ must be a JSON object.", fieldName)
        }
        return nil
    }
}
