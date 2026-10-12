// RemoteLLMMessages.swift
// Provides Remote LLMMessages for LLM requests and response handling.

import Foundation

extension RemoteLLMRuntimeClient {
    func openAICompatibleMessages(systemPrompt: String, userPrompt: String) -> [[String: String]] {
        let trimmedSystem = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedSystem.isEmpty {
            return [
                ["role": "user", "content": userPrompt]
            ]
        }
        return [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": userPrompt]
        ]
    }

    func openAICompatibleConversationMessages(
        systemPrompt: String,
        currentUserPrompt: String,
        conversationHistory: [RewriteConversationPromptTurn]
    ) -> [[String: String]] {
        var messages: [[String: String]] = []

        let trimmedSystem = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedSystem.isEmpty {
            messages.append([
                "role": "system",
                "content": systemPrompt
            ])
        }

        for turn in conversationHistory {
            let userPrompt = turn.modelUserMessage
            if !userPrompt.isEmpty {
                messages.append([
                    "role": "user",
                    "content": userPrompt
                ])
            }

            let assistantMessage = turn.resultContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if !assistantMessage.isEmpty {
                messages.append([
                    "role": "assistant",
                    "content": assistantMessage
                ])
            }
        }

        messages.append([
            "role": "user",
            "content": currentUserPrompt
        ])

        return messages
    }

    func makeResponsesRequest(
        provider: RemoteLLMProvider,
        endpointValue: String,
        model: String,
        systemPrompt: String,
        inputPayload: Any,
        runtimeConfiguration: RemoteProviderRuntimeConfiguration,
        previousResponseID: String?,
        tuning: RemoteLLMRuntimeClient.GenerationTuning,
        textFormat: [String: Any]?,
        streamingEnabled: Bool,
        additionalHeaders: [String: String] = [:]
    ) throws -> URLRequest {
        let configuration = runtimeConfiguration.value
        guard let url = URL(string: endpointValue) else {
            throw NSError(
                domain: "Voxt.RemoteLLM",
                code: -300,
                userInfo: [NSLocalizedDescriptionKey: "Invalid remote LLM endpoint URL: \(endpointValue)"]
            )
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = requestTimeoutInterval(for: provider)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let effectiveStreamingEnabled = provider == .codex ? true : streamingEnabled
        request.setValue(
            effectiveStreamingEnabled ? "text/event-stream, application/json" : "application/json",
            forHTTPHeaderField: "Accept"
        )

        let apiKey = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        for (key, value) in additionalHeaders {
            request.setValue(value, forHTTPHeaderField: key)
        }

        let generationSettings = configuration.effectiveGenerationSettings(provider: provider)
        let maxOutputTokens = generationSettings.maxOutputTokens.map { max(1, $0) } ?? tuning.maxTokens

        var payload: [String: Any] = [
            "model": model,
            "stream": effectiveStreamingEnabled
        ]
        if provider == .codex {
            payload["store"] = false
            if configuration.codexFastModeEnabled {
                payload["service_tier"] = "priority"
            }
        } else {
            payload["max_output_tokens"] = maxOutputTokens
        }
        if provider != .openAI && provider != .codex {
            payload["temperature"] = tuning.temperature
            payload["top_p"] = tuning.topP
        }

        let trimmedSystemPrompt = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedSystemPrompt.isEmpty {
            payload["instructions"] = trimmedSystemPrompt
        } else if provider == .codex {
            payload["instructions"] = "Process the input and return only the final answer."
        }

        let trimmedPreviousResponseID = previousResponseID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !trimmedPreviousResponseID.isEmpty {
            payload["previous_response_id"] = trimmedPreviousResponseID
            payload["input"] = responsesInputPayload(inputPayload, provider: provider)
        } else {
            payload["input"] = responsesInputPayload(inputPayload, provider: provider)
        }

        if let reasoningEffort = responsesReasoningEffort(
            provider: provider,
            model: model,
            settings: generationSettings,
            configuration: configuration
        ) {
            payload["reasoning"] = [
                "effort": reasoningEffort
            ]
        }
        applyResponsesThinkingConfiguration(
            to: &payload,
            provider: provider,
            settings: generationSettings
        )
        applyStructuredResponsesThinkingPolicy(
            to: &payload,
            provider: provider,
            settings: generationSettings,
            textFormat: textFormat
        )

        var textPayload = textFormat ?? responsesTextPayload(for: generationSettings.responseFormat)
        if provider == .openAI,
           let verbosity = OpenAITextVerbosity.apiValue(
            selection: configuration.openAITextVerbosity,
            model: model
           ) {
            textPayload["verbosity"] = verbosity
        }
        if !textPayload.isEmpty {
            payload["text"] = textPayload
        }

        if LLMProviderCapabilityRegistry.capabilities(for: provider).supportsExtraBody {
            try applyCommonExtraBody(
                to: &payload,
                settings: generationSettings,
                fieldName: AppLocalization.localizedString("Extra Body JSON")
            )
        }

        if configuration.searchEnabled && provider.supportsHostedSearch {
            payload["tools"] = [
                [
                    "type": "web_search"
                ]
            ]
        }

        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        return request
    }

    func responsesInputPayload(_ inputPayload: Any, provider: RemoteLLMProvider) -> Any {
        guard provider == .codex else { return inputPayload }
        if inputPayload is [[String: Any]] {
            return inputPayload
        }
        if let inputText = inputPayload as? String {
            return [
                [
                    "role": "user",
                    "content": inputText
                ]
            ]
        }
        return inputPayload
    }

    func responsesTextPayload(for responseFormat: LLMResponseFormat) -> [String: Any] {
        switch responseFormat {
        case .plain:
            return [:]
        case .json:
            return [
                "format": [
                    "type": "json_object"
                ]
            ]
        case .jsonSchema:
            return [:]
        }
    }

    func responsesTextFormat(
        for responseFormat: RemoteLLMRuntimeClient.OpenAICompatibleResponseFormat?
    ) -> [String: Any]? {
        guard let responseFormat else { return nil }
        switch responseFormat {
        case .jsonObject:
            return [
                "format": [
                    "type": "json_object"
                ]
            ]
        }
    }

    func applyStructuredResponsesThinkingPolicy(
        to payload: inout [String: Any],
        provider: RemoteLLMProvider,
        settings: LLMGenerationSettings,
        textFormat: [String: Any]?
    ) {
        guard textFormat != nil, settings.thinking.mode == .providerDefault else { return }
        switch provider {
        case .volcengine:
            payload["thinking"] = ["type": "disabled"]
        case .aliyunBailian:
            payload["enable_thinking"] = false
        default:
            break
        }
    }

    func responsesReasoningEffort(
        provider: RemoteLLMProvider,
        model: String,
        settings: LLMGenerationSettings,
        configuration: RemoteProviderConfiguration
    ) -> String? {
        if provider == .codex {
            return nil
        }

        if provider == .openAI {
            if settings.thinking.mode == .effort,
               let effort = settings.thinking.effort,
               OpenAIReasoningEffort.apiValue(selection: effort, model: model) != nil {
                return effort
            }
            return OpenAIReasoningEffort.apiValue(
                selection: configuration.openAIReasoningEffort,
                model: model
            )
        }

        guard settings.thinking.mode == .effort else { return nil }
        switch provider {
        case .volcengine:
            return settings.thinking.effort
        default:
            return nil
        }
    }

    func applyResponsesThinkingConfiguration(
        to payload: inout [String: Any],
        provider: RemoteLLMProvider,
        settings: LLMGenerationSettings
    ) {
        switch provider {
        case .aliyunBailian:
            switch settings.thinking.mode {
            case .off:
                payload["enable_thinking"] = false
            case .on, .effort:
                payload["enable_thinking"] = true
            case .budget:
                payload["enable_thinking"] = true
                if let budget = settings.thinking.budgetTokens {
                    payload["thinking_budget"] = budget
                }
            case .providerDefault:
                break
            }
        case .volcengine:
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
            case .providerDefault, .effort:
                break
            }
        default:
            break
        }
    }

    func googleSearchToolPayload(for model: String) -> [String: Any] {
        let normalizedModel = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalizedModel.hasPrefix("gemini-1.5") {
            return [
                "google_search_retrieval": [
                    "dynamic_retrieval_config": [
                        "mode": "MODE_DYNAMIC",
                        "dynamic_threshold": 0.0
                    ]
                ]
            ]
        }
        return [
            "google_search": [:]
        ]
    }

    func applyOpenAICompatibleSearchConfiguration(
        to payload: inout [String: Any],
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration
    ) {
        guard configuration.searchEnabled, provider.supportsHostedSearch else { return }

        switch provider {
        case .zai:
            payload["tools"] = [
                [
                    "type": "web_search",
                    "web_search": [
                        "enable": "True",
                        "search_engine": "search_std"
                    ]
                ]
            ]
        default:
            break
        }
    }

    func responsesInputMessages(
        currentUserInput: String,
        conversationHistory: [RewriteConversationPromptTurn]
    ) -> [[String: Any]] {
        var messages: [[String: Any]] = []

        for turn in conversationHistory {
            let userPrompt = turn.modelUserMessage
            if !userPrompt.isEmpty {
                messages.append([
                    "role": "user",
                    "content": userPrompt
                ])
            }

            let assistantMessage = turn.resultContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if !assistantMessage.isEmpty {
                messages.append([
                    "role": "assistant",
                    "content": assistantMessage
                ])
            }
        }

        messages.append([
            "role": "user",
            "content": currentUserInput
        ])
        return messages
    }

    func responsesStreamingDelta(from object: [String: Any]) -> String? {
        if let type = object["type"] as? String {
            if type == "response.output_text.delta", let delta = object["delta"] as? String, !delta.isEmpty {
                return delta
            }
            if (type == "response.output_text" || type == "response.output_text.done"),
               let text = object["text"] as? String,
               !text.isEmpty {
                return text
            }
            if type == "response.completed",
               let response = object["response"],
               let text = extractPrimaryText(from: response),
               !text.isEmpty {
                return text
            }
        }

        if let delta = object["delta"] as? String, !delta.isEmpty {
            return delta
        }
        if let text = object["text"] as? String, !text.isEmpty {
            return text
        }
        return nil
    }

    func responsesResponseID(from object: [String: Any]) -> String? {
        if let response = object["response"] as? [String: Any],
           let id = response["id"] as? String,
           !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return id
        }
        if let id = object["response_id"] as? String,
           !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return id
        }
        if let id = object["id"] as? String,
           !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           (object["type"] as? String)?.hasPrefix("response.") == true {
            return id
        }
        return nil
    }

    func responsesErrorMessage(from object: [String: Any]) -> String? {
        if let type = object["type"] as? String,
           type == "error" || type == "response.failed" {
            if let error = object["error"] as? [String: Any],
               let message = error["message"] as? String,
               !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return message
            }
            if let message = object["message"] as? String,
               !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return message
            }
        }
        return nil
    }
}
