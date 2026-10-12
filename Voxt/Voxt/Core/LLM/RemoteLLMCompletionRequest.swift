// Provider-specific chat completion requests; Responses requests live in RemoteLLMMessages.

import Foundation

extension RemoteLLMRuntimeClient {
    func makeCompletionRequest(
        provider: RemoteLLMProvider,
        runtimeConfiguration: RemoteProviderRuntimeConfiguration,
        endpointValue: String,
        model: String,
        systemPrompt: String,
        userPrompt: String,
        messagesOverride: [[String: String]]? = nil,
        openAICompatibleResponseFormat: OpenAICompatibleResponseFormat? = nil,
        tuning: GenerationTuning,
        streamingEnabled: Bool
    ) throws -> URLRequest {
        let configuration = runtimeConfiguration.value
        let resolvedEndpoint: String
        if provider == .ollama {
            resolvedEndpoint = resolvedOllamaRequestEndpoint(
                endpoint: endpointValue,
                useGenerate: false
            )
        } else {
            resolvedEndpoint = streamingEndpointValue(
                provider: provider,
                endpoint: endpointValue,
                model: model,
                streamingEnabled: streamingEnabled
            )
        }
        guard let url = URL(string: resolvedEndpoint) else {
            throw NSError(
                domain: "Voxt.RemoteLLM",
                code: -300,
                userInfo: [NSLocalizedDescriptionKey: "Invalid remote LLM endpoint URL: \(resolvedEndpoint)"]
            )
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = requestTimeoutInterval(for: provider)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(
            streamingEnabled ? "text/event-stream, application/x-ndjson, application/json" : "application/json",
            forHTTPHeaderField: "Accept"
        )

        switch provider {
        case .anthropic:
            guard !configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw NSError(domain: "Voxt.RemoteLLM", code: -301, userInfo: [NSLocalizedDescriptionKey: "Anthropic API key is empty."])
            }
            request.setValue(configuration.apiKey, forHTTPHeaderField: "x-api-key")
            request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
            var payload: [String: Any] = [
                "model": model,
                "max_tokens": tuning.maxTokens,
                "stream": streamingEnabled,
                "messages": [
                    ["role": "user", "content": userPrompt]
                ]
            ]
            applyAnthropicGenerationSettings(
                to: &payload,
                settings: configuration.effectiveGenerationSettings(provider: provider),
                tuning: tuning
            )
            let trimmedSystem = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedSystem.isEmpty {
                payload["system"] = systemPrompt
            }
            if configuration.searchEnabled && provider.supportsHostedSearch {
                payload["tools"] = [
                    [
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 5
                    ]
                ]
            }
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        case .google:
            let apiKey = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !apiKey.isEmpty else {
                throw NSError(domain: "Voxt.RemoteLLM", code: -302, userInfo: [NSLocalizedDescriptionKey: "Google API key is empty."])
            }
            guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
                throw NSError(domain: "Voxt.RemoteLLM", code: -303, userInfo: [NSLocalizedDescriptionKey: "Invalid Google endpoint URL."])
            }
            var items = components.queryItems ?? []
            if !items.contains(where: { $0.name == "key" }) {
                items.append(URLQueryItem(name: "key", value: apiKey))
            }
            if streamingEnabled && !items.contains(where: { $0.name == "alt" }) {
                items.append(URLQueryItem(name: "alt", value: "sse"))
            }
            components.queryItems = items
            request.url = components.url
            var payload: [String: Any] = [
                "contents": [
                    ["parts": [["text": userPrompt]]]
                ]
            ]
            applyGoogleGenerationSettings(
                to: &payload,
                settings: configuration.effectiveGenerationSettings(provider: provider),
                tuning: tuning
            )
            let trimmedSystem = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedSystem.isEmpty {
                payload["system_instruction"] = ["parts": [["text": systemPrompt]]]
            }
            if configuration.searchEnabled && provider.supportsHostedSearch {
                payload["tools"] = [googleSearchToolPayload(for: model)]
            }
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        case .minimax:
            let apiKey = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !apiKey.isEmpty else {
                throw NSError(domain: "Voxt.RemoteLLM", code: -304, userInfo: [NSLocalizedDescriptionKey: "MiniMax API key is empty."])
            }
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            var payload: [String: Any] = [
                "model": model,
                "stream": streamingEnabled,
                "messages": openAICompatibleMessages(systemPrompt: systemPrompt, userPrompt: userPrompt)
            ]
            try applyOpenAICompatibleGenerationSettings(
                to: &payload,
                provider: provider,
                configuration: configuration,
                tuning: tuning,
                responseFormat: openAICompatibleResponseFormat
            )
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        case .ollama where usesNativeOllamaEndpoint(url):
            let apiKey = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
            if !apiKey.isEmpty {
                request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            }
            request.httpBody = try JSONSerialization.data(
                withJSONObject: try ollamaNativePayload(
                    endpointURL: url,
                    model: model,
                    systemPrompt: systemPrompt,
                    userPrompt: userPrompt,
                    messagesOverride: messagesOverride,
                    configuration: configuration,
                    tuning: tuning,
                    streamingEnabled: streamingEnabled
                )
            )
        default:
            let apiKey = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
            if !apiKey.isEmpty {
                request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            }
            var payload = openAICompatiblePayload(
                model: model,
                systemPrompt: systemPrompt,
                userPrompt: userPrompt,
                messagesOverride: messagesOverride,
                tuning: tuning,
                streamingEnabled: streamingEnabled,
                responseFormat: openAICompatibleResponseFormat
            )
            try applyOpenAICompatibleGenerationSettings(
                to: &payload,
                provider: provider,
                configuration: configuration,
                tuning: tuning,
                responseFormat: openAICompatibleResponseFormat
            )
            if provider == .ollama {
                try applyOllamaCompatibleOptionOverrides(
                    to: &payload,
                    configuration: configuration
                )
            } else if provider == .omlx {
                try applyOMLXCompatibleConfiguration(
                    to: &payload,
                    configuration: configuration
                )
            }
            applyOpenAICompatibleSearchConfiguration(
                to: &payload,
                provider: provider,
                configuration: configuration
            )
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        }

        return request
    }

    func openAICompatiblePayload(
        model: String,
        systemPrompt: String,
        userPrompt: String,
        messagesOverride: [[String: String]]? = nil,
        tuning: GenerationTuning,
        streamingEnabled: Bool,
        responseFormat: OpenAICompatibleResponseFormat? = nil
    ) -> [String: Any] {
        var payload: [String: Any] = [
            "model": model,
            "messages": messagesOverride ?? openAICompatibleMessages(systemPrompt: systemPrompt, userPrompt: userPrompt),
            "stream": streamingEnabled,
            "max_tokens": tuning.maxTokens,
            "temperature": tuning.temperature,
            "top_p": tuning.topP
        ]

        switch responseFormat {
        case .jsonObject:
            payload["response_format"] = ["type": "json_object"]
        case nil:
            break
        }

        return payload
    }
}
