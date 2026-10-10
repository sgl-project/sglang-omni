import Foundation

extension RemoteProviderConnectivityTester {
    func testLLMProvider(_ provider: RemoteLLMProvider, configuration: RemoteProviderConfiguration) async throws -> String {
        let model = configuration.model.isEmpty ? provider.suggestedModel : configuration.model
        let endpoint = resolvedLLMTestEndpoint(provider: provider, endpoint: configuration.endpoint, model: model)
        var headers: [String: String] = [:]
        switch provider {
        case .anthropic:
            guard !configuration.apiKey.isEmpty else {
                throw NSError(domain: "Voxt.Settings", code: -30, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Anthropic API Key is required for testing.")])
            }
            headers["x-api-key"] = configuration.apiKey
            headers["anthropic-version"] = "2023-06-01"
            return try await testAnthropicReachability(endpoint: endpoint, headers: headers, model: model)
        case .google:
            guard !configuration.apiKey.isEmpty else {
                throw NSError(domain: "Voxt.Settings", code: -31, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Google API Key is required for testing.")])
            }
            return try await testGoogleReachability(endpoint: endpoint, apiKey: configuration.apiKey)
        case .minimax:
            guard !configuration.apiKey.isEmpty else {
                throw NSError(domain: "Voxt.Settings", code: -32, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("MiniMax API Key is required for testing.")])
            }
            headers["Authorization"] = "Bearer \(configuration.apiKey)"
            return try await testMiniMaxReachability(endpoint: endpoint, headers: headers, model: model)
        case .openAI, .codex, .ollama, .omlx, .deepseek, .openrouter, .grok, .zai, .volcengine, .kimi, .lmStudio, .aliyunBailian, .stepFun, .xiaomiMiMo:
            if !configuration.apiKey.isEmpty {
                headers["Authorization"] = "Bearer \(configuration.apiKey)"
            }
            if provider == .codex {
                for (key, value) in try await RemoteLLMRuntimeClient().authorizationHeaders(
                    provider: .codex,
                    configuration: configuration
                ) {
                    headers[key] = value
                }
            }
            if provider.usesResponsesAPI {
                return try await testResponsesReachability(
                    provider: provider,
                    endpoint: endpoint,
                    headers: headers,
                    configuration: configuration,
                    model: model
                )
            }
            return try await testOpenAICompatibleReachability(
                provider: provider,
                endpoint: endpoint,
                headers: headers,
                configuration: configuration,
                model: model
            )
        }
    }

    private func testOpenAICompatibleReachability(
        provider: RemoteLLMProvider,
        endpoint: String,
        headers: [String: String],
        configuration: RemoteProviderConfiguration,
        model: String
    ) async throws -> String {
        let runtimeClient = RemoteLLMRuntimeClient()
        let requestEndpoint: String
        if provider == .ollama {
            requestEndpoint = runtimeClient.resolvedOllamaRequestEndpoint(
                endpoint: endpoint,
                useGenerate: false
            )
        } else {
            requestEndpoint = endpoint
        }
        let body = try await openAICompatibleReachabilityBody(
            provider: provider,
            endpoint: requestEndpoint,
            configuration: configuration,
            model: model
        )
        return try await testJSONPOSTReachability(endpoint: requestEndpoint, headers: headers, body: body)
    }

    func openAICompatibleReachabilityBody(
        provider: RemoteLLMProvider,
        endpoint: String,
        configuration: RemoteProviderConfiguration,
        model: String
    ) async throws -> [String: Any] {
        let runtimeClient = RemoteLLMRuntimeClient()
        if provider == .ollama,
           let url = URL(string: endpoint),
           usesNativeOllamaEndpoint(url) {
            return try runtimeClient.ollamaNativePayload(
                endpointURL: url,
                model: model,
                systemPrompt: "",
                userPrompt: "ping",
                configuration: configuration,
                tuning: .init(maxTokens: 32, temperature: 0.2, topP: 0.9),
                streamingEnabled: false
            )
        }

        if provider == .deepseek {
            return [
                "model": model,
                "messages": [
                    ["role": "user", "content": "ping"]
                ],
                "thinking": [
                    "type": "disabled"
                ],
                "max_tokens": 1,
                "stream": false
            ]
        }

        var payload: [String: Any] = [
            "model": model,
            "messages": [
                ["role": "user", "content": "ping"]
            ],
            "stream": false
        ]
        if provider == .ollama {
            try runtimeClient.applyOllamaCompatibleOptionOverrides(
                to: &payload,
                configuration: configuration
            )
        } else if provider == .omlx {
            try runtimeClient.applyOMLXCompatibleConfiguration(
                to: &payload,
                configuration: configuration
            )
        }
        return payload
    }

    private func testResponsesReachability(
        provider: RemoteLLMProvider,
        endpoint: String,
        headers: [String: String],
        configuration: RemoteProviderConfiguration,
        model: String
    ) async throws -> String {
        let runtimeClient = RemoteLLMRuntimeClient()
        let systemPrompt = provider == .codex
            ? "Reply with exactly pong."
            : ""
        let runtimeConfiguration = try RemoteModelConfigurationStore.runtimeConfiguration(
            for: configuration
        )
        let request = try runtimeClient.makeResponsesRequest(
            provider: provider,
            endpointValue: endpoint,
            model: model,
            systemPrompt: systemPrompt,
            inputPayload: "ping",
            runtimeConfiguration: runtimeConfiguration,
            previousResponseID: nil,
            tuning: .init(maxTokens: 32, temperature: 0.2, topP: 0.9),
            textFormat: nil,
            streamingEnabled: false,
            additionalHeaders: headers
        )
        return try await sendLLMTestRequest(
            request,
            context: "LLM Responses test",
            allowValidationErrorsAsReachable: provider != .codex
        )
    }

    private func testAnthropicReachability(
        endpoint: String,
        headers: [String: String],
        model: String
    ) async throws -> String {
        let body: [String: Any] = [
            "model": model,
            "messages": [
                ["role": "user", "content": "ping"]
            ],
            "stream": false
        ]
        return try await testJSONPOSTReachability(endpoint: endpoint, headers: headers, body: body)
    }

    private func testGoogleReachability(
        endpoint: String,
        apiKey: String
    ) async throws -> String {
        guard var components = URLComponents(string: endpoint) else {
            throw NSError(domain: "Voxt.Settings", code: -33, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid Google endpoint URL.")])
        }
        let hasKeyQuery = components.queryItems?.contains(where: { $0.name == "key" }) ?? false
        if !hasKeyQuery {
            var items = components.queryItems ?? []
            items.append(URLQueryItem(name: "key", value: apiKey))
            components.queryItems = items
        }
        guard let url = components.url else {
            throw NSError(domain: "Voxt.Settings", code: -34, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid Google endpoint URL.")])
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 15
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let body: [String: Any] = [
            "contents": [
                ["parts": [["text": "ping"]]]
            ]
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        return try await sendLLMTestRequest(request, context: "LLM Google test")
    }

    private func testMiniMaxReachability(
        endpoint: String,
        headers: [String: String],
        model: String
    ) async throws -> String {
        let body: [String: Any] = [
            "model": model,
            "messages": [
                ["role": "user", "content": "ping"]
            ]
        ]
        return try await testJSONPOSTReachability(endpoint: endpoint, headers: headers, body: body)
    }

    func testJSONPOSTReachability(
        endpoint: String,
        headers: [String: String],
        body: [String: Any],
        successMessage: String = ""
    ) async throws -> String {
        guard let url = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.Settings", code: -35, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid endpoint URL.")])
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 15
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        for (key, value) in headers {
            request.setValue(value, forHTTPHeaderField: key)
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        return try await sendLLMTestRequest(
            request,
            context: "LLM JSON POST test",
            successMessage: successMessage
        )
    }

    private func sendLLMTestRequest(
        _ request: URLRequest,
        context: String,
        successMessage: String = "",
        allowValidationErrorsAsReachable: Bool = true
    ) async throws -> String {
        let bodyPreview = request.httpBody.flatMap { String(data: $0, encoding: .utf8) } ?? "<empty>"
        RemoteProviderConnectivityTestLogging.logHTTPRequest(context: context, request: request, bodyPreview: bodyPreview)
        let (data, response) = try await VoxtNetworkSession.active.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "Voxt.Settings", code: -36, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid server response.")])
        }
        RemoteProviderConnectivityTestLogging.logHTTPResponse(context: context, response: http, data: data)

        return try llmTestResponseMessage(
            statusCode: http.statusCode,
            data: data,
            successMessage: successMessage,
            allowValidationErrorsAsReachable: allowValidationErrorsAsReachable
        )
    }

    func llmTestResponseMessage(
        statusCode: Int,
        data: Data,
        successMessage: String = "",
        allowValidationErrorsAsReachable: Bool = true
    ) throws -> String {
        let acceptsValidationErrors: Bool
        switch testTarget {
        case .llm(.deepseek):
            // DeepSeek's probe is a valid completion request. A 400/422 can
            // indicate an invalid custom model ID, not a usable connection.
            acceptsValidationErrors = false
        default:
            acceptsValidationErrors = allowValidationErrorsAsReachable
        }
        let payload = String(data: data.prefix(220), encoding: .utf8) ?? ""
        if (200...299).contains(statusCode) {
            if !successMessage.isEmpty {
                return successMessage
            }
            return AppLocalization.format("Connection test succeeded (HTTP %d).", statusCode)
        }
        if acceptsValidationErrors && (statusCode == 400 || statusCode == 422) {
            return AppLocalization.format("Endpoint reachable (HTTP %d). Authentication and routing look valid.", statusCode)
        }
        if statusCode == 401 || statusCode == 403 {
            throw NSError(
                domain: "Voxt.Settings",
                code: statusCode,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Server reachable, but authentication failed (HTTP %d). %@", statusCode, payload)]
            )
        }
        throw NSError(
            domain: "Voxt.Settings",
            code: statusCode,
            userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Connection failed (HTTP %d). %@", statusCode, payload)]
        )
    }

    private func usesNativeOllamaEndpoint(_ url: URL) -> Bool {
        let path = url.path.lowercased()
        return path.isEmpty ||
            path == "/" ||
            path == "/api" ||
            path.hasSuffix("/api/chat") ||
            path.hasSuffix("/api/generate") ||
            path.hasSuffix("/api/tags")
    }

    private func resolvedLLMTestEndpoint(provider: RemoteLLMProvider, endpoint: String, model: String) -> String {
        RemoteLLMRuntimeClient().resolvedLLMEndpoint(
            provider: provider,
            endpoint: endpoint,
            model: model
        )
    }
}
