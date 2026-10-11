// Shared request security, budgets, logging, and partial-delivery policy.

import Foundation

extension RemoteLLMRuntimeClient {
    struct StreamingPartialDeliveryState {
        var lastPublishedAt = Date.distantPast
        var lastPublishedLength = 0

        mutating func shouldPublish(
            aggregatedText: String,
            force: Bool,
            minimumInterval: TimeInterval = 0.05,
            minimumCharacterDelta: Int = 96
        ) -> Bool {
            guard force else {
                let elapsed = Date().timeIntervalSince(lastPublishedAt)
                let appendedCount = aggregatedText.count - lastPublishedLength
                guard elapsed >= minimumInterval || appendedCount >= minimumCharacterDelta else {
                    return false
                }
                return true
            }
            return aggregatedText.count != lastPublishedLength
        }

        mutating func markPublished(aggregatedText: String) {
            lastPublishedAt = Date()
            lastPublishedLength = aggregatedText.count
        }
    }

    func logRequest(
        request: URLRequest,
        provider: RemoteLLMProvider,
        endpointValue: String,
        model: String,
        inputTextLength: Int,
        systemPrompt: String,
        debugInput: String,
        userPrompt: String,
        tuning: GenerationTuning,
        requestMaxTokensDescription: String
    ) {
        let proxySettings = VoxtNetworkSession.currentProxySettings
        let proxyRoute = request.url.map { resolvedProxyRoute(for: $0, settings: proxySettings) } ?? "unavailable"
        let networkMode = VoxtNetworkSession.modeDescription
        VoxtLog.llmInfo(
            "Remote LLM request started. provider=\(provider.rawValue), endpoint=\(endpointValue), url=\(request.url?.absoluteString ?? endpointValue), model=\(model), timeoutSec=\(Int(request.timeoutInterval)), inputChars=\(inputTextLength), systemChars=\(systemPrompt.count), userChars=\(userPrompt.count), maxTokens=\(requestMaxTokensDescription), temp=\(tuning.temperature), topP=\(tuning.topP), networkMode=\(networkMode), proxy=\(proxyRoute)"
        )
        VoxtLog.llm(
            """
            Remote LLM request content. provider=\(provider.rawValue), endpoint=\(endpointValue), model=\(model)
            [system_prompt]
            \(VoxtLog.llmPreview(systemPrompt))
            [input]
            \(VoxtLog.llmPreview(debugInput))
            [request_content]
            \(VoxtLog.llmPreview(userPrompt))
            """
        )
    }

    func requestMaxTokensDescription(
        provider: RemoteLLMProvider,
        usesResponsesAPI: Bool,
        tuning: GenerationTuning
    ) -> String {
        if usesResponsesAPI && provider == .codex {
            return "auto"
        }
        return "\(tuning.maxTokens)"
    }

    func validateEndpointSecurity(
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration
    ) throws {
        guard let message = RemoteEndpointSecurityPolicy.validationMessage(
            endpoint: configuration.endpoint,
            hasCredentials: RemoteEndpointSecurityPolicy.hasLLMCredentials(
                provider: provider,
                configuration: configuration
            )
        ) else { return }
        throw NSError(
            domain: "Voxt.RemoteLLM",
            code: -901,
            userInfo: [NSLocalizedDescriptionKey: message]
        )
    }

    func supportsStreaming(provider: RemoteLLMProvider, intent: CompletionIntent) -> Bool {
        if requiresResponsesStreaming(provider: provider) {
            return true
        }
        guard intent == .rewrite else { return false }
        return true
    }

    func requiresResponsesStreaming(provider: RemoteLLMProvider) -> Bool {
        provider == .codex
    }

    func requestTimeoutInterval(for provider: RemoteLLMProvider) -> TimeInterval {
        switch provider {
        case .zai, .volcengine, .xiaomiMiMo:
            return 30
        default:
            return 40
        }
    }

    func guardRepeatedOutputIfNeeded(
        _ content: String,
        provider: RemoteLLMProvider,
        endpointValue: String,
        context: String
    ) -> String {
        guard let repetition = LLMOutputRepetitionGuard().repeatedSuffix(in: content) else {
            return content
        }
        VoxtLog.llmWarning(
            "Remote LLM \(context) repetition guard truncated output. provider=\(provider.rawValue), endpoint=\(endpointValue), repeatedUnitChars=\(repetition.repeatedUnit.count), repetitions=\(repetition.repetitionCount), outputChars=\(repetition.truncatedText.count)"
        )
        return repetition.truncatedText
    }

    func generationTuning(
        for provider: RemoteLLMProvider,
        inputTextLength: Int,
        systemPromptLength: Int,
        userPromptLength: Int,
        intent: CompletionIntent
    ) -> GenerationTuning {
        let outputBudget = estimatedOutputTokenBudget(
            inputTextLength: inputTextLength,
            systemPromptLength: systemPromptLength,
            userPromptLength: userPromptLength,
            intent: intent
        )
        switch provider {
        case .volcengine:
            // Favor low latency and deterministic rewrite/translation behavior.
            return GenerationTuning(maxTokens: outputBudget, temperature: 0.1, topP: 0.3)
        case .xiaomiMiMo:
            return GenerationTuning(maxTokens: outputBudget, temperature: 1.0, topP: 0.95)
        case .zai:
            return GenerationTuning(maxTokens: outputBudget, temperature: 0.2, topP: 0.7)
        default:
            return GenerationTuning(maxTokens: outputBudget, temperature: 0.2, topP: 0.9)
        }
    }

    private func estimatedOutputTokenBudget(
        inputTextLength: Int,
        systemPromptLength: Int,
        userPromptLength: Int,
        intent: CompletionIntent
    ) -> Int {
        let safeInput = max(1, inputTextLength)
        // Keep output budget mainly tied to ASR text length, and reserve a small
        // extra window for instruction overhead (system/user prompt framing).
        let instructionChars = max(0, systemPromptLength + userPromptLength - safeInput)
        let baseMultiplier: Double
        let minimumBudget: Int
        let maximumBudget: Int

        switch intent {
        case .translation:
            baseMultiplier = 1.35
            minimumBudget = 128
            maximumBudget = 1024
        case .rewrite:
            // Rewrite often needs to synthesize a fresh answer from a short spoken
            // instruction, so the budget cannot track prompt length too closely.
            baseMultiplier = safeInput < 180 ? 2.6 : 1.4
            minimumBudget = safeInput < 180 ? 384 : 256
            maximumBudget = 1536
        case .enhancement:
            baseMultiplier = 1.15
            minimumBudget = 128
            maximumBudget = 1024
        case .dictionaryHistoryScan:
            // Dictionary ingest emits compact JSON objects and can legitimately
            // return a short list of accepted terms, so it needs a wider floor.
            baseMultiplier = 1.60
            minimumBudget = 384
            maximumBudget = 2048
        }

        let contentEstimate = Int(Double(safeInput) * baseMultiplier)
        let instructionReserveLimit: Int
        switch intent {
        case .rewrite:
            instructionReserveLimit = 256
        case .dictionaryHistoryScan:
            instructionReserveLimit = 320
        case .enhancement, .translation:
            instructionReserveLimit = 192
        }
        let instructionReserve = min(instructionReserveLimit, max(32, instructionChars / 12))
        let estimate = contentEstimate + instructionReserve
        return max(minimumBudget, min(estimate, maximumBudget))
    }
}
