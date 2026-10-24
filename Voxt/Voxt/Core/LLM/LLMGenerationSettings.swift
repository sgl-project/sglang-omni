// LLMGenerationSettings.swift
// Provides LLMGeneration Settings for LLM requests and response handling.

import Foundation

enum LLMResponseFormat: String, Codable, Hashable, CaseIterable {
    case plain
    case json
    case jsonSchema

    var title: String {
        switch self {
        case .plain:
            return AppLocalization.localizedString("Plain Text")
        case .json:
            return "JSON"
        case .jsonSchema:
            return AppLocalization.localizedString("JSON Schema")
        }
    }
}

enum LLMThinkingMode: String, Codable, Hashable, CaseIterable {
    case providerDefault
    case off
    case on
    case effort
    case budget
}

struct LLMThinkingSettings: Codable, Hashable {
    var mode: LLMThinkingMode
    var effort: String?
    var budgetTokens: Int?
    var exposeReasoning: Bool

    static let providerDefault = LLMThinkingSettings(
        mode: .providerDefault,
        effort: nil,
        budgetTokens: nil,
        exposeReasoning: false
    )

    static let off = LLMThinkingSettings(
        mode: .off,
        effort: nil,
        budgetTokens: nil,
        exposeReasoning: false
    )
}

struct LLMGenerationSettings: Codable, Hashable {
    var maxOutputTokens: Int?
    var temperature: Double?
    var topP: Double?
    var topK: Int?
    var minP: Double?
    var seed: Int?
    var stop: [String]
    var presencePenalty: Double?
    var frequencyPenalty: Double?
    var repetitionPenalty: Double?
    var logprobs: Bool
    var topLogprobs: Int?
    var responseFormat: LLMResponseFormat
    var thinking: LLMThinkingSettings
    var extraBodyJSON: String
    var extraOptionsJSON: String

    init(
        maxOutputTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        seed: Int? = nil,
        stop: [String] = [],
        presencePenalty: Double? = nil,
        frequencyPenalty: Double? = nil,
        repetitionPenalty: Double? = nil,
        logprobs: Bool = false,
        topLogprobs: Int? = nil,
        responseFormat: LLMResponseFormat = .plain,
        thinking: LLMThinkingSettings = .providerDefault,
        extraBodyJSON: String = "",
        extraOptionsJSON: String = ""
    ) {
        self.maxOutputTokens = maxOutputTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.seed = seed
        self.stop = stop
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.repetitionPenalty = repetitionPenalty
        self.logprobs = logprobs
        self.topLogprobs = topLogprobs
        self.responseFormat = responseFormat
        self.thinking = thinking
        self.extraBodyJSON = extraBodyJSON
        self.extraOptionsJSON = extraOptionsJSON
    }
}

struct LLMProviderCapabilities: Equatable {
    var supportsThinkingToggle: Bool = false
    var supportsThinkingEffort: Bool = false
    var supportsThinkingBudget: Bool = false
    var supportsMaxOutputTokens: Bool = true
    var supportsTemperature: Bool = true
    var supportsTopP: Bool = true
    var supportsTopK: Bool = false
    var supportsMinP: Bool = false
    var supportsSeed: Bool = false
    var supportsPenalties: Bool = false
    var supportsLogprobs: Bool = false
    var supportsResponseFormat: Bool = false
    var supportsStopSequences: Bool = true
    var supportsExtraBody: Bool = true
    var supportsExtraOptions: Bool = false
}

enum LLMProviderCapabilityRegistry {
    static let localMLXCapabilities = LLMProviderCapabilities(
        supportsThinkingToggle: true,
        supportsTemperature: true,
        supportsTopP: true,
        supportsTopK: true,
        supportsMinP: true,
        supportsPenalties: true,
        supportsExtraBody: false
    )

    static func capabilities(for provider: RemoteLLMProvider) -> LLMProviderCapabilities {
        switch provider {
        case .openAI:
            return LLMProviderCapabilities(
                supportsThinkingEffort: true,
                supportsLogprobs: true,
                supportsResponseFormat: true
            )
        case .codex:
            return LLMProviderCapabilities(
                supportsMaxOutputTokens: false,
                supportsTemperature: false,
                supportsTopP: false,
                supportsStopSequences: false,
                supportsExtraBody: false
            )
        case .anthropic:
            return LLMProviderCapabilities(
                supportsThinkingBudget: true,
                supportsTopK: true
            )
        case .google:
            return LLMProviderCapabilities(
                supportsThinkingBudget: true,
                supportsTopK: true,
                supportsResponseFormat: true
            )
        case .ollama:
            return LLMProviderCapabilities(
                supportsThinkingToggle: true,
                supportsThinkingEffort: true,
                supportsTopK: true,
                supportsMinP: true,
                supportsSeed: true,
                supportsPenalties: true,
                supportsLogprobs: true,
                supportsResponseFormat: true,
                supportsExtraOptions: true
            )
        case .omlx, .lmStudio:
            return LLMProviderCapabilities(
                supportsTopK: true,
                supportsMinP: true,
                supportsSeed: true,
                supportsPenalties: true,
                supportsLogprobs: true,
                supportsResponseFormat: true
            )
        case .deepseek:
            return LLMProviderCapabilities(
                supportsThinkingToggle: true,
                supportsThinkingEffort: true,
                supportsLogprobs: true,
                supportsResponseFormat: true
            )
        case .zai, .volcengine, .aliyunBailian:
            return LLMProviderCapabilities(
                supportsThinkingToggle: true,
                supportsThinkingEffort: true,
                supportsThinkingBudget: true,
                supportsPenalties: true,
                supportsLogprobs: true,
                supportsResponseFormat: true
            )
        case .xiaomiMiMo:
            return LLMProviderCapabilities(
                supportsThinkingToggle: true,
                supportsPenalties: true,
                supportsResponseFormat: true
            )
        case .stepFun:
            return LLMProviderCapabilities(
                supportsThinkingEffort: true,
                supportsPenalties: true,
                supportsResponseFormat: true
            )
        case .openrouter, .grok, .kimi:
            return LLMProviderCapabilities(
                supportsThinkingEffort: true,
                supportsThinkingBudget: true,
                supportsPenalties: true,
                supportsLogprobs: true,
                supportsResponseFormat: true
            )
        case .minimax:
            return LLMProviderCapabilities(
                supportsPenalties: true,
                supportsResponseFormat: true
            )
        }
    }
}

enum CustomLLMGenerationSettingsStore {
    static let defaultSettings = LLMGenerationSettings(thinking: .off)

    static func resolvedSettings(
        for repo: String,
        rawByRepo: String?,
        legacyRaw: String?
    ) -> LLMGenerationSettings {
        let canonicalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
        let values = decodedByRepo(from: rawByRepo)
        if let settings = values[canonicalRepo] {
            return sanitized(settings)
        }
        return resolvedSettings(from: legacyRaw)
    }

    static func resolvedSettings(from rawValue: String?) -> LLMGenerationSettings {
        guard let rawValue,
              let data = rawValue.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(LLMGenerationSettings.self, from: data)
        else {
            return defaultSettings
        }
        return sanitized(decoded)
    }

    static func storageValue(for settings: LLMGenerationSettings) -> String {
        let sanitized = sanitized(settings)
        guard let data = try? JSONEncoder().encode(sanitized),
              let text = String(data: data, encoding: .utf8) else {
            return defaultStoredValue()
        }
        return text
    }

    static func save(
        _ settings: LLMGenerationSettings,
        for repo: String,
        rawByRepo: String?
    ) -> String {
        let canonicalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
        var values = decodedByRepo(from: rawByRepo)
        values[canonicalRepo] = sanitized(settings)
        return storageValue(forByRepo: values)
    }

    static func resolvedByRepo(from rawValue: String?) -> [String: LLMGenerationSettings] {
        decodedByRepo(from: rawValue)
    }

    static func storageValue(forByRepo settingsByRepo: [String: LLMGenerationSettings]) -> String {
        var sanitizedByRepo = [String: LLMGenerationSettings]()
        for (repo, settings) in settingsByRepo {
            guard CustomLLMModelCatalog.preservesGenerationSettings(for: repo) else { continue }
            let canonicalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
            guard repo == canonicalRepo || settingsByRepo[canonicalRepo] == nil else { continue }
            sanitizedByRepo[canonicalRepo] = sanitized(settings)
        }
        guard let data = try? JSONEncoder().encode(sanitizedByRepo),
              let text = String(data: data, encoding: .utf8) else {
            return defaultByRepoStoredValue()
        }
        return text
    }

    static func defaultByRepoStoredValue() -> String {
        "{}"
    }

    static func defaultStoredValue() -> String {
        storageValue(for: defaultSettings)
    }

    static func sanitized(_ settings: LLMGenerationSettings) -> LLMGenerationSettings {
        var sanitized = settings
        sanitized.maxOutputTokens = settings.maxOutputTokens.map { max(1, min($0, 32768)) }
        sanitized.temperature = settings.temperature.map { max(0, min($0, 2)) }
        sanitized.topP = settings.topP.map { max(0, min($0, 1)) }
        sanitized.topK = settings.topK.map { max(0, min($0, 1000)) }
        sanitized.minP = settings.minP.map { max(0, min($0, 1)) }
        sanitized.repetitionPenalty = settings.repetitionPenalty.map { max(0.5, min($0, 2)) }
        sanitized.presencePenalty = nil
        sanitized.frequencyPenalty = nil
        sanitized.seed = nil
        sanitized.stop = []
        sanitized.logprobs = false
        sanitized.topLogprobs = nil
        sanitized.responseFormat = .plain
        sanitized.extraBodyJSON = ""
        sanitized.extraOptionsJSON = ""

        switch settings.thinking.mode {
        case .providerDefault, .off:
            sanitized.thinking = LLMThinkingSettings(
                mode: .off,
                effort: nil,
                budgetTokens: nil,
                exposeReasoning: false
            )
        case .on:
            sanitized.thinking = LLMThinkingSettings(
                mode: settings.thinking.mode,
                effort: nil,
                budgetTokens: nil,
                exposeReasoning: false
            )
        case .effort, .budget:
            sanitized.thinking = .providerDefault
        }
        return sanitized
    }

    private static func decodedByRepo(from rawValue: String?) -> [String: LLMGenerationSettings] {
        guard let rawValue,
              let data = rawValue.data(using: .utf8),
              let decoded = try? JSONDecoder().decode([String: LLMGenerationSettings].self, from: data)
        else {
            return [:]
        }

        var sanitizedByRepo = [String: LLMGenerationSettings]()
        for (repo, settings) in decoded {
            // Retired-model tuning must not overwrite the replacement model's settings.
            guard CustomLLMModelCatalog.preservesGenerationSettings(for: repo) else { continue }
            let canonicalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
            guard repo == canonicalRepo || decoded[canonicalRepo] == nil else { continue }
            sanitizedByRepo[canonicalRepo] = sanitized(settings)
        }
        return sanitizedByRepo
    }
}

extension RemoteProviderConfiguration {
    func effectiveGenerationSettings(provider: RemoteLLMProvider) -> LLMGenerationSettings {
        var settings = generationSettings
        if provider == .stepFun, settings.thinking.mode == .providerDefault {
            settings.thinking = .off
        }
        if provider == .deepseek {
            // Current DeepSeek models default to thinking, which can consume the
            // short output budget before producing any visible text. Keep legacy
            // reasoner semantics and explicit user choices, but use fast text
            // generation by default for all other models (including custom IDs).
            let normalizedModel = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if settings.thinking.mode == .providerDefault, normalizedModel != "deepseek-reasoner" {
                settings.thinking = .off
            }
            settings.presencePenalty = nil
            settings.frequencyPenalty = nil
        }
        guard provider == .codex else {
            return settings
        }
        let capabilities = LLMProviderCapabilityRegistry.capabilities(for: provider)
        if !capabilities.supportsMaxOutputTokens {
            settings.maxOutputTokens = nil
        }
        if !capabilities.supportsTemperature {
            settings.temperature = nil
        }
        if !capabilities.supportsTopP {
            settings.topP = nil
        }
        if !capabilities.supportsTopK {
            settings.topK = nil
        }
        if !capabilities.supportsMinP {
            settings.minP = nil
        }
        if !capabilities.supportsSeed {
            settings.seed = nil
        }
        if !capabilities.supportsPenalties {
            settings.presencePenalty = nil
            settings.frequencyPenalty = nil
            settings.repetitionPenalty = nil
        }
        if !capabilities.supportsLogprobs {
            settings.logprobs = false
            settings.topLogprobs = nil
        }
        if !capabilities.supportsResponseFormat {
            settings.responseFormat = .plain
        }
        if !capabilities.supportsStopSequences {
            settings.stop = []
        }
        if !capabilities.supportsThinkingToggle &&
            !capabilities.supportsThinkingEffort &&
            !capabilities.supportsThinkingBudget {
            settings.thinking = .providerDefault
        }
        if !capabilities.supportsExtraBody {
            settings.extraBodyJSON = ""
        }
        if !capabilities.supportsExtraOptions {
            settings.extraOptionsJSON = ""
        }
        return settings
    }
}

extension LLMGenerationSettings {
    static func legacy(
        providerID: String,
        openAIReasoningEffort: String,
        openAIMaxOutputTokens: Int?,
        ollamaResponseFormat: String,
        ollamaThinkMode: String,
        ollamaLogprobsEnabled: Bool,
        ollamaTopLogprobs: Int?,
        ollamaOptionsJSON: String,
        omlxResponseFormat: String,
        omlxExtraBodyJSON: String
    ) -> LLMGenerationSettings {
        var settings = LLMGenerationSettings()
        if providerID == RemoteLLMProvider.openAI.rawValue {
            settings.maxOutputTokens = openAIMaxOutputTokens
        }

        if providerID != RemoteLLMProvider.stepFun.rawValue,
           let effort = OpenAIReasoningEffort(rawValue: openAIReasoningEffort),
           effort != .automatic {
            settings.thinking = LLMThinkingSettings(
                mode: .effort,
                effort: effort.rawValue,
                budgetTokens: nil,
                exposeReasoning: false
            )
        }

        if providerID == RemoteLLMProvider.ollama.rawValue {
            if let responseFormat = LLMResponseFormat(ollamaResponseFormat: ollamaResponseFormat) {
                settings.responseFormat = responseFormat
            }
            if let thinkMode = OllamaThinkMode(rawValue: ollamaThinkMode) {
                settings.thinking = LLMThinkingSettings(
                    mode: thinkMode.llmThinkingMode,
                    effort: thinkMode.llmThinkingEffort,
                    budgetTokens: nil,
                    exposeReasoning: false
                )
            }
            settings.logprobs = ollamaLogprobsEnabled
            settings.topLogprobs = ollamaTopLogprobs
            settings.extraOptionsJSON = ollamaOptionsJSON
        } else if providerID == RemoteLLMProvider.omlx.rawValue {
            if let responseFormat = LLMResponseFormat(omlxResponseFormat: omlxResponseFormat) {
                settings.responseFormat = responseFormat
            }
            settings.extraBodyJSON = omlxExtraBodyJSON
        }
        if providerID == RemoteLLMProvider.stepFun.rawValue,
           settings.thinking.mode == .providerDefault {
            settings.thinking = .off
        }

        return settings
    }
}

extension LLMResponseFormat {
    init?(ollamaResponseFormat: String) {
        switch OllamaResponseFormat(rawValue: ollamaResponseFormat) {
        case .plain:
            self = .plain
        case .json:
            self = .json
        case .jsonSchema:
            self = .jsonSchema
        case nil:
            return nil
        }
    }

    init?(omlxResponseFormat: String) {
        switch OMLXResponseFormat(rawValue: omlxResponseFormat) {
        case .plain:
            self = .plain
        case .jsonSchema:
            self = .jsonSchema
        case nil:
            return nil
        }
    }
}

extension OllamaThinkMode {
    var llmThinkingMode: LLMThinkingMode {
        switch self {
        case .off:
            return .off
        case .on:
            return .on
        case .low, .medium, .high:
            return .effort
        }
    }

    var llmThinkingEffort: String? {
        switch self {
        case .low, .medium, .high:
            return rawValue
        case .off, .on:
            return nil
        }
    }
}
