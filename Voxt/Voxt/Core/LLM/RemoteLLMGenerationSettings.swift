// Maps unified generation settings to provider payload fields.

import Foundation

extension RemoteLLMRuntimeClient {
    func applyAnthropicGenerationSettings(
        to payload: inout [String: Any],
        settings: LLMGenerationSettings,
        tuning: GenerationTuning
    ) {
        var maxTokens = settings.maxOutputTokens.map { max(1, $0) } ?? tuning.maxTokens
        if settings.thinking.mode == .budget,
           let budget = settings.thinking.budgetTokens {
            maxTokens = max(maxTokens, budget + 1)
        }
        payload["max_tokens"] = maxTokens
        if let temperature = settings.temperature {
            payload["temperature"] = temperature
        }
        if let topP = settings.topP {
            payload["top_p"] = topP
        }
        if let topK = settings.topK {
            payload["top_k"] = topK
        }
        if !settings.stop.isEmpty {
            payload["stop_sequences"] = settings.stop
        }
        switch settings.thinking.mode {
        case .budget:
            guard let budget = settings.thinking.budgetTokens else { break }
            let thinking: [String: Any] = [
                "type": "enabled",
                "budget_tokens": budget,
                "display": "omitted"
            ]
            payload["thinking"] = thinking
        case .off:
            payload["thinking"] = ["type": "disabled"]
        case .on, .providerDefault, .effort:
            break
        }
    }

    func applyGoogleGenerationSettings(
        to payload: inout [String: Any],
        settings: LLMGenerationSettings,
        tuning: GenerationTuning
    ) {
        var generationConfig: [String: Any] = [
            "maxOutputTokens": settings.maxOutputTokens.map { max(1, $0) } ?? tuning.maxTokens,
            "temperature": settings.temperature ?? tuning.temperature,
            "topP": settings.topP ?? tuning.topP
        ]
        if let topK = settings.topK {
            generationConfig["topK"] = topK
        }
        if !settings.stop.isEmpty {
            generationConfig["stopSequences"] = settings.stop
        }
        switch settings.responseFormat {
        case .plain:
            break
        case .json, .jsonSchema:
            generationConfig["responseMimeType"] = "application/json"
        }
        switch settings.thinking.mode {
        case .off:
            generationConfig["thinkingConfig"] = ["thinkingBudget": 0]
        case .budget:
            if let budget = settings.thinking.budgetTokens {
                generationConfig["thinkingConfig"] = ["thinkingBudget": budget]
            }
        case .on, .providerDefault, .effort:
            break
        }
        payload["generationConfig"] = generationConfig
    }

    func applyOpenAICompatibleGenerationSettings(
        to payload: inout [String: Any],
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration,
        tuning: GenerationTuning,
        responseFormat: OpenAICompatibleResponseFormat?
    ) throws {
        let settings = configuration.effectiveGenerationSettings(provider: provider)
        if let maxOutputTokens = settings.maxOutputTokens {
            payload["max_tokens"] = max(1, maxOutputTokens)
        } else if shouldOmitDefaultMaxTokens(provider: provider, model: configuration.model) {
            payload.removeValue(forKey: "max_tokens")
        } else {
            payload["max_tokens"] = tuning.maxTokens
        }
        if let temperature = settings.temperature {
            payload["temperature"] = temperature
        }
        if let topP = settings.topP {
            payload["top_p"] = topP
        }
        if let seed = settings.seed {
            payload["seed"] = seed
        }
        if !settings.stop.isEmpty {
            payload["stop"] = settings.stop
        }
        if provider != .stepFun, let presencePenalty = settings.presencePenalty {
            payload["presence_penalty"] = presencePenalty
        }
        if let frequencyPenalty = settings.frequencyPenalty {
            payload["frequency_penalty"] = frequencyPenalty
        }
        if settings.logprobs {
            payload["logprobs"] = true
            if let topLogprobs = settings.topLogprobs {
                payload["top_logprobs"] = topLogprobs
            }
        }

        switch settings.responseFormat {
        case .plain:
            break
        case .json:
            payload["response_format"] = ["type": "json_object"]
        case .jsonSchema:
            if payload["response_format"] == nil {
                payload["response_format"] = ["type": "json_object"]
            }
        }

        if responseFormat == .jsonObject {
            payload["response_format"] = ["type": "json_object"]
        }

        if provider == .xiaomiMiMo, let maxTokens = payload.removeValue(forKey: "max_tokens") {
            payload["max_completion_tokens"] = maxTokens
        }

        applyOpenAICompatibleThinkingSettings(
            to: &payload,
            provider: provider,
            settings: settings,
            model: configuration.model
        )
        try applyCommonExtraBody(
            to: &payload,
            settings: settings,
            fieldName: AppLocalization.localizedString("Extra Body JSON")
        )
    }

    func shouldOmitDefaultMaxTokens(provider: RemoteLLMProvider, model: String) -> Bool {
        guard provider == .stepFun else { return false }
        let normalizedModel = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return [
            "step-3.5-flash",
            "step-3.5-flash-2603",
            "step-3",
            "step-r1-v-mini",
            "step-router-v1"
        ].contains(normalizedModel)
    }

    func applyOpenAICompatibleThinkingSettings(
        to payload: inout [String: Any],
        provider: RemoteLLMProvider,
        settings: LLMGenerationSettings,
        model: String
    ) {
        switch provider {
        case .openrouter:
            var reasoning: [String: Any] = ["exclude": !settings.thinking.exposeReasoning]
            switch settings.thinking.mode {
            case .effort:
                if let effort = settings.thinking.effort { reasoning["effort"] = effort }
            case .budget:
                if let budget = settings.thinking.budgetTokens { reasoning["max_tokens"] = budget }
            case .off:
                reasoning["enabled"] = false
            case .on:
                reasoning["enabled"] = true
            case .providerDefault:
                break
            }
            payload["reasoning"] = reasoning
        case .deepseek:
            switch settings.thinking.mode {
            case .off:
                payload["thinking"] = ["type": "disabled"]
            case .on, .budget:
                // DeepSeek supports effort, not a separate thinking-token budget.
                // Preserve old budget configurations as an explicit thinking opt-in.
                payload["thinking"] = ["type": "enabled"]
            case .effort:
                if let effort = settings.thinking.effort {
                    payload["reasoning_effort"] = effort
                }
            case .providerDefault:
                break
            }
        case .zai, .volcengine, .aliyunBailian:
            switch settings.thinking.mode {
            case .off:
                payload["thinking"] = ["type": "disabled"]
                if provider == .aliyunBailian {
                    payload["enable_thinking"] = false
                }
            case .on:
                payload["thinking"] = ["type": "enabled"]
                if provider == .aliyunBailian {
                    payload["enable_thinking"] = true
                }
            case .budget:
                if provider == .aliyunBailian {
                    payload["enable_thinking"] = true
                    if let budget = settings.thinking.budgetTokens {
                        payload["thinking_budget"] = budget
                    }
                } else {
                    var thinking: [String: Any] = ["type": "enabled"]
                    if let budget = settings.thinking.budgetTokens {
                        thinking["budget_tokens"] = budget
                    }
                    payload["thinking"] = thinking
                }
            case .effort:
                if let effort = settings.thinking.effort {
                    payload["reasoning_effort"] = effort
                }
            case .providerDefault:
                break
            }
        case .xiaomiMiMo:
            switch settings.thinking.mode {
            case .off:
                payload["thinking"] = ["type": "disabled"]
            case .on:
                payload["thinking"] = ["type": "enabled"]
            case .providerDefault, .effort, .budget:
                break
            }
        case .grok:
            if settings.thinking.mode == .effort, let effort = settings.thinking.effort {
                payload["reasoning_effort"] = effort
            }
        case .stepFun:
            let normalizedModel = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if settings.thinking.mode == .effort,
               let effort = settings.thinking.effort,
               normalizedModel == "step-3.5-flash-2603",
               ["low", "high"].contains(effort) {
                payload["reasoning_effort"] = effort
            }
        case .kimi:
            switch settings.thinking.mode {
            case .off:
                payload["thinking"] = ["type": "disabled"]
            case .on:
                payload["thinking"] = ["type": "enabled"]
            case .budget:
                var thinking: [String: Any] = ["type": "enabled"]
                if let budget = settings.thinking.budgetTokens {
                    thinking["budget_tokens"] = budget
                }
                payload["thinking"] = thinking
            case .effort:
                if let effort = settings.thinking.effort {
                    payload["reasoning_effort"] = effort
                }
            case .providerDefault:
                break
            }
        default:
            break
        }
    }

    func applyCommonExtraBody(
        to payload: inout [String: Any],
        settings: LLMGenerationSettings,
        fieldName: String
    ) throws {
        guard let extraBody = try optionalJSONObject(source: settings.extraBodyJSON, fieldName: fieldName) else {
            return
        }
        for (key, value) in extraBody {
            payload[key] = value
        }
    }

    func optionalJSONObject(
        source: String,
        fieldName: String
    ) throws -> [String: Any]? {
        let trimmed = source.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return try requiredJSONObject(source: trimmed, fieldName: fieldName)
    }

    func requiredJSONObject(
        source: String,
        fieldName: String
    ) throws -> [String: Any] {
        guard let data = source.data(using: .utf8) else {
            throw NSError(
                domain: "Voxt.RemoteLLM",
                code: -308,
                userInfo: [NSLocalizedDescriptionKey: "\(fieldName) is not valid UTF-8 JSON."]
            )
        }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(
                domain: "Voxt.RemoteLLM",
                code: -309,
                userInfo: [NSLocalizedDescriptionKey: "\(fieldName) must be a JSON object."]
            )
        }
        return object
    }

    func doubleValue(from value: Any?) -> Double? {
        switch value {
        case let number as NSNumber:
            return number.doubleValue
        case let string as String:
            return Double(string.trimmingCharacters(in: .whitespacesAndNewlines))
        default:
            return nil
        }
    }

    func intValue(from value: Any?) -> Int? {
        switch value {
        case let number as NSNumber:
            return number.intValue
        case let string as String:
            return Int(string.trimmingCharacters(in: .whitespacesAndNewlines))
        default:
            return nil
        }
    }
}
