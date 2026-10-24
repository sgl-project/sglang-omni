// Ollama native/compatible payloads and oMLX configuration.

import Foundation

extension RemoteLLMRuntimeClient {
    func ollamaNativePayload(
        endpointURL: URL? = nil,
        model: String,
        systemPrompt: String,
        userPrompt: String,
        messagesOverride: [[String: String]]? = nil,
        configuration: RemoteProviderConfiguration,
        tuning: GenerationTuning,
        streamingEnabled: Bool
    ) throws -> [String: Any] {
        if let endpointURL, usesNativeOllamaGenerateEndpoint(endpointURL) {
            return try ollamaNativeGeneratePayload(
                model: model,
                systemPrompt: systemPrompt,
                userPrompt: userPrompt,
                messagesOverride: messagesOverride,
                configuration: configuration,
                tuning: tuning,
                streamingEnabled: streamingEnabled
            )
        }

        var payload: [String: Any] = [
            "model": model,
            "messages": openAICompatibleMessages(systemPrompt: systemPrompt, userPrompt: userPrompt),
            "stream": streamingEnabled,
            "options": try mergedOllamaNativeOptions(configuration: configuration, tuning: tuning)
        ]

        try applyOllamaNativeConfiguration(to: &payload, configuration: configuration)
        return payload
    }

    private func ollamaNativeGeneratePayload(
        model: String,
        systemPrompt: String,
        userPrompt: String,
        messagesOverride: [[String: String]]?,
        configuration: RemoteProviderConfiguration,
        tuning: GenerationTuning,
        streamingEnabled: Bool
    ) throws -> [String: Any] {
        let promptInput = ollamaGeneratePromptInput(
            systemPrompt: systemPrompt,
            userPrompt: userPrompt,
            messagesOverride: messagesOverride
        )

        var payload: [String: Any] = [
            "model": model,
            "prompt": promptInput.prompt,
            "stream": streamingEnabled,
            "options": try mergedOllamaNativeOptions(configuration: configuration, tuning: tuning)
        ]

        if !promptInput.system.isEmpty {
            payload["system"] = promptInput.system
        }

        try applyOllamaNativeConfiguration(to: &payload, configuration: configuration)
        return payload
    }

    private func applyOllamaNativeConfiguration(
        to payload: inout [String: Any],
        configuration: RemoteProviderConfiguration
    ) throws {
        let settings = configuration.effectiveGenerationSettings(provider: .ollama)
        switch settings.responseFormat {
        case .plain:
            break
        case .json:
            payload["format"] = "json"
        case .jsonSchema:
            payload["format"] = try requiredJSONObject(
                source: configuration.ollamaJSONSchema,
                fieldName: "Ollama JSON Schema"
            )
        }

        switch settings.thinking.mode {
        case .off:
            payload["think"] = false
        case .on, .budget:
            payload["think"] = true
        case .effort:
            if let effort = settings.thinking.effort {
                payload["think"] = effort
            }
        case .providerDefault:
            break
        }

        let keepAlive = configuration.ollamaKeepAlive.trimmingCharacters(in: .whitespacesAndNewlines)
        if !keepAlive.isEmpty {
            payload["keep_alive"] = keepAlive
        }

        if settings.logprobs {
            payload["logprobs"] = true
            if let topLogprobs = settings.topLogprobs {
                payload["top_logprobs"] = topLogprobs
            }
        }
    }

    private func ollamaGeneratePromptInput(
        systemPrompt: String,
        userPrompt: String,
        messagesOverride: [[String: String]]?
    ) -> (system: String, prompt: String) {
        guard let messagesOverride, !messagesOverride.isEmpty else {
            return (
                systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines),
                userPrompt
            )
        }

        var resolvedSystem = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        var promptSegments: [String] = []

        for message in messagesOverride {
            let role = message["role"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
            let content = message["content"]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !content.isEmpty else { continue }

            if role == "system" {
                if resolvedSystem.isEmpty {
                    resolvedSystem = content
                }
                continue
            }

            let prefix: String
            switch role {
            case "assistant":
                prefix = "Assistant"
            case "user":
                prefix = "User"
            default:
                prefix = role.isEmpty ? "User" : role.capitalized
            }
            promptSegments.append("\(prefix):\n\(content)")
        }

        let prompt = promptSegments.joined(separator: "\n\n")
        return (
            resolvedSystem,
            prompt.isEmpty ? userPrompt : prompt
        )
    }

    func mergedOllamaNativeOptions(
        configuration: RemoteProviderConfiguration,
        tuning: GenerationTuning
    ) throws -> [String: Any] {
        let settings = configuration.effectiveGenerationSettings(provider: .ollama)
        var options: [String: Any] = [
            "temperature": settings.temperature ?? tuning.temperature,
            "top_p": settings.topP ?? tuning.topP,
            "num_predict": settings.maxOutputTokens.map { max(1, $0) } ?? tuning.maxTokens
        ]
        if let topK = settings.topK {
            options["top_k"] = topK
        }
        if let minP = settings.minP {
            options["min_p"] = minP
        }
        if let seed = settings.seed {
            options["seed"] = seed
        }
        if let repetitionPenalty = settings.repetitionPenalty {
            options["repeat_penalty"] = repetitionPenalty
        }
        if !settings.stop.isEmpty {
            options["stop"] = settings.stop
        }

        if let customOptions = try optionalJSONObject(
            source: settings.extraOptionsJSON,
            fieldName: "Ollama Options JSON"
        ) {
            for (key, value) in customOptions {
                options[key] = value
            }
        }

        return options
    }

    func applyOllamaCompatibleOptionOverrides(
        to payload: inout [String: Any],
        configuration: RemoteProviderConfiguration
    ) throws {
        let settings = configuration.effectiveGenerationSettings(provider: .ollama)
        guard let customOptions = try optionalJSONObject(
            source: settings.extraOptionsJSON,
            fieldName: "Ollama Options JSON"
        ) else {
            return
        }

        if let temperature = doubleValue(from: customOptions["temperature"]) {
            payload["temperature"] = temperature
        }
        if let topP = doubleValue(from: customOptions["top_p"] ?? customOptions["topP"]) {
            payload["top_p"] = topP
        }
        if let maxTokens = intValue(from: customOptions["max_tokens"] ?? customOptions["num_predict"]) {
            payload["max_tokens"] = maxTokens
        }
    }

    func applyOMLXCompatibleConfiguration(
        to payload: inout [String: Any],
        configuration: RemoteProviderConfiguration
    ) throws {
        let settings = configuration.effectiveGenerationSettings(provider: .omlx)
        if settings.responseFormat == .jsonSchema {
            payload["response_format"] = [
                "type": "json_schema",
                "json_schema": [
                    "name": "voxt_output",
                    "schema": try requiredJSONObject(
                        source: configuration.omlxJSONSchema,
                        fieldName: AppLocalization.localizedString("oMLX JSON Schema")
                    )
                ]
            ]
        }

        if configuration.omlxIncludeUsageStreamOptions,
           payload["stream"] as? Bool == true {
            var streamOptions = payload["stream_options"] as? [String: Any] ?? [:]
            streamOptions["include_usage"] = true
            payload["stream_options"] = streamOptions
        }

        if let extraBody = try optionalJSONObject(
            source: settings.extraBodyJSON,
            fieldName: AppLocalization.localizedString("oMLX Extra Body JSON")
        ) {
            for (key, value) in extraBody {
                payload[key] = value
            }
        }
    }
}
