// Task entry points and compiled-request routing.

import Foundation

// Protocol execution and payload mapping live in sibling extensions. Shared helpers
// are internal for cross-file use; callers should use these task entry points.
struct RemoteLLMRuntimeClient {
    private let sessionOverride: URLSession?

    nonisolated init(session: URLSession? = nil) {
        self.sessionOverride = session
    }

    var networkSession: URLSession { sessionOverride ?? VoxtNetworkSession.active }

    enum OpenAICompatibleResponseFormat: Equatable {
        case jsonObject
    }

    struct StreamingFailure: Error {
        let underlying: Error
        let partialText: String
        let emittedChunkCount: Int
    }

    enum CompletionIntent: Equatable {
        case enhancement
        case translation
        case rewrite
        case dictionaryHistoryScan
    }

    struct GenerationTuning {
        let maxTokens: Int
        let temperature: Double
        let topP: Double

        func applying(_ settings: LLMGenerationSettings) -> GenerationTuning {
            GenerationTuning(
                maxTokens: settings.maxOutputTokens.map { max(1, $0) } ?? maxTokens,
                temperature: settings.temperature ?? temperature,
                topP: settings.topP ?? topP
            )
        }
    }

    func authorizationHeaders(
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration
    ) async throws -> [String: String] {
        guard provider == .codex else { return [:] }
        try validateEndpointSecurity(provider: provider, configuration: configuration)
        return try await CodexOAuthCredentialProvider(
            authFilePath: configuration.codexAuthFilePath,
            authFileBookmark: configuration.codexAuthFileBookmark
        ).authorizationHeaders()
    }

    func warmupConnection(
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration
    ) async throws {
        let runtimeConfiguration = try RemoteModelConfigurationStore.runtimeConfiguration(for: configuration)
        let configuration = runtimeConfiguration.value
        try validateEndpointSecurity(provider: provider, configuration: configuration)
        let model = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? provider.suggestedModel
            : configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let endpointValue = provider.usesResponsesAPI
            ? responsesEndpointValue(provider: provider, endpoint: configuration.endpoint, model: model)
            : resolvedLLMEndpoint(provider: provider, endpoint: configuration.endpoint, model: model)
        guard let url = URL(string: endpointValue) else {
            throw NSError(
                domain: "Voxt.RemoteLLM",
                code: -900,
                userInfo: [NSLocalizedDescriptionKey: "Invalid remote LLM endpoint URL: \(endpointValue)"]
            )
        }

        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        request.timeoutInterval = min(8, requestTimeoutInterval(for: provider))
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let apiKey = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        for (key, value) in try await authorizationHeaders(provider: provider, configuration: configuration) {
            request.setValue(value, forHTTPHeaderField: key)
        }

        let (_, response) = try await networkSession.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else { return }
        guard (200..<500).contains(httpResponse.statusCode) else {
            throw NSError(
                domain: "Voxt.RemoteLLM",
                code: httpResponse.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "Remote warmup failed with HTTP \(httpResponse.statusCode)."]
            )
        }
    }

    func executeCompiledRequest(
        _ request: LLMCompiledRequest,
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration,
        onPartialText: (@Sendable (String) -> Void)? = nil,
        onResponseID: ((String) -> Void)? = nil
    ) async throws -> String {
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return request.fallbackText }

        let intent: CompletionIntent
        switch request.taskLabel {
        case "enhancement":
            intent = .enhancement
        case "translation":
            intent = .translation
        case "rewrite":
            intent = .rewrite
        default:
            intent = .enhancement
        }

        let usesResponsesConversation =
            intent == .rewrite &&
            provider.usesResponsesAPI &&
            (!request.conversationHistory.isEmpty ||
             !(request.previousResponseID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "").isEmpty)
        let usesChatConversation =
            intent == .rewrite &&
            !provider.usesResponsesAPI &&
            !request.conversationHistory.isEmpty

        if provider.usesResponsesAPI {
            let trimmedPreviousResponseID = request.previousResponseID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let inputPayload: Any
            let currentUserInputPayload = prompt

            if usesResponsesConversation {
                if !trimmedPreviousResponseID.isEmpty {
                    inputPayload = currentUserInputPayload
                } else {
                    inputPayload = responsesInputMessages(
                        currentUserInput: prompt,
                        conversationHistory: request.conversationHistory
                    )
                }
            } else {
                inputPayload = currentUserInputPayload
            }

            let textFormat = responsesTextFormat(for: request.responseFormat)
            let result: ResponsesStreamingResult
            do {
                result = try await completeResponses(
                    systemPrompt: request.instructions,
                    debugInput: request.debugInput,
                    requestContentForLog: prompt,
                    inputPayload: inputPayload,
                    inputTextLength: request.inputCharacterCount,
                    intent: intent,
                    provider: provider,
                    configuration: configuration,
                    previousResponseID: usesResponsesConversation ? request.previousResponseID : nil,
                    textFormat: textFormat,
                    onPartialText: onPartialText,
                    onResponseID: onResponseID
                )
            } catch let error as NSError where
                textFormat != nil &&
                error.domain == "Voxt.RemoteLLM" &&
                [-308, -309].contains(error.code) {
                let retryConfiguration = structuredResponsesRetryConfiguration(
                    configuration,
                    provider: provider
                )
                VoxtLog.llmWarning(
                    "Remote LLM structured Responses output was incomplete or invalid; retrying with a larger output budget and thinking disabled where supported. provider=\(provider.rawValue), detail=\(error.localizedDescription)"
                )
                result = try await completeResponses(
                    systemPrompt: request.instructions,
                    debugInput: request.debugInput,
                    requestContentForLog: prompt,
                    inputPayload: inputPayload,
                    inputTextLength: request.inputCharacterCount,
                    intent: intent,
                    provider: provider,
                    configuration: retryConfiguration,
                    previousResponseID: usesResponsesConversation ? request.previousResponseID : nil,
                    textFormat: textFormat,
                    onPartialText: onPartialText,
                    onResponseID: onResponseID
                )
            }
            let trimmed = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? request.fallbackText : trimmed
        }

        let output = try await complete(
            systemPrompt: request.instructions,
            debugInput: request.debugInput,
            userPrompt: prompt,
            inputTextLength: request.inputCharacterCount,
            intent: intent,
            provider: provider,
            configuration: configuration,
            messagesOverride: usesChatConversation
                ? openAICompatibleConversationMessages(
                    systemPrompt: request.instructions,
                    currentUserPrompt: prompt,
                    conversationHistory: request.conversationHistory
                )
                : nil,
            openAICompatibleResponseFormat: request.responseFormat,
            onPartialText: onPartialText
        )
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? request.fallbackText : trimmed
    }

    func enhance(
        userPrompt: String,
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration
    ) async throws -> String {
        let input = userPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return "" }
        if provider.usesResponsesAPI {
            let result = try await completeResponses(
                systemPrompt: "",
                debugInput: input,
                requestContentForLog: input,
                inputPayload: input,
                inputTextLength: input.count,
                intent: .enhancement,
                provider: provider,
                configuration: configuration
            )
            return result.text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
        }
        let output = try await complete(
            systemPrompt: "",
            debugInput: input,
            userPrompt: input,
            inputTextLength: input.count,
            intent: .enhancement,
            provider: provider,
            configuration: configuration
        )
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func dictionaryHistoryScanTerms(
        userPrompt: String,
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration
    ) async throws -> [String] {
        let input = userPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return [] }
        if provider.usesResponsesAPI {
            let result = try await completeResponses(
                systemPrompt: "",
                debugInput: input,
                requestContentForLog: input,
                inputPayload: input,
                inputTextLength: input.count,
                intent: .dictionaryHistoryScan,
                provider: provider,
                configuration: configuration,
                textFormat: DictionaryHistoryScanResponseParser.responsesTextFormatPayload()
            )
            return try DictionaryHistoryScanResponseParser.parseTerms(from: result.text)
        }
        let output = try await complete(
            systemPrompt: "",
            debugInput: input,
            userPrompt: input,
            inputTextLength: input.count,
            intent: .dictionaryHistoryScan,
            provider: provider,
            configuration: configuration
        )
        return try DictionaryHistoryScanResponseParser.parseTerms(from: output)
    }

    private func structuredResponsesRetryConfiguration(
        _ configuration: RemoteProviderConfiguration,
        provider: RemoteLLMProvider
    ) -> RemoteProviderConfiguration {
        var retryConfiguration = configuration
        retryConfiguration.generationSettings.maxOutputTokens = max(
            retryConfiguration.generationSettings.maxOutputTokens ?? 0,
            1_536
        )
        switch provider {
        case .volcengine, .aliyunBailian:
            retryConfiguration.generationSettings.thinking = .off
        default:
            break
        }
        return retryConfiguration
    }
}
