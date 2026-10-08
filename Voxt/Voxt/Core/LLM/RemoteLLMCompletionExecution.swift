// Chat completion execution, streaming, and endpoint retries.

import Foundation

extension RemoteLLMRuntimeClient {
    func complete(
        systemPrompt: String,
        debugInput: String,
        userPrompt: String,
        inputTextLength: Int,
        intent: CompletionIntent,
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration,
        messagesOverride: [[String: String]]? = nil,
        openAICompatibleResponseFormat: OpenAICompatibleResponseFormat? = nil,
        onPartialText: (@Sendable (String) -> Void)? = nil
    ) async throws -> String {
        try Task.checkCancellation()
        let runtimeConfiguration = try RemoteModelConfigurationStore.runtimeConfiguration(for: configuration)
        let configuration = runtimeConfiguration.value
        try validateEndpointSecurity(provider: provider, configuration: configuration)
        let model = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? provider.suggestedModel
            : configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let endpoint = resolvedLLMEndpoint(provider: provider, endpoint: configuration.endpoint, model: model)
        let endpoints = resolvedEndpointCandidates(provider: provider, primaryEndpoint: endpoint)
        var lastError: Error?
        let shouldAttemptStreaming = onPartialText != nil && supportsStreaming(provider: provider, intent: intent)

        for (index, endpointValue) in endpoints.enumerated() {
            let attemptStartedAt = Date()
            let tuning = generationTuning(
                for: provider,
                inputTextLength: inputTextLength,
                systemPromptLength: systemPrompt.count,
                userPromptLength: userPrompt.count,
                intent: intent
            ).applying(configuration.effectiveGenerationSettings(provider: provider))
            do {
                if shouldAttemptStreaming, let onPartialText {
                    do {
                        let streamingRequest = try makeCompletionRequest(
                            provider: provider,
                            runtimeConfiguration: runtimeConfiguration,
                            endpointValue: endpointValue,
                            model: model,
                            systemPrompt: systemPrompt,
                            userPrompt: userPrompt,
                            messagesOverride: messagesOverride,
                            openAICompatibleResponseFormat: openAICompatibleResponseFormat,
                            tuning: tuning,
                            streamingEnabled: true
                        )
                        let requestStartedAt = Date()
                        logRequest(
                            request: streamingRequest,
                            provider: provider,
                            endpointValue: endpointValue,
                            model: model,
                            inputTextLength: inputTextLength,
                            systemPrompt: systemPrompt,
                            debugInput: debugInput,
                            userPrompt: userPrompt,
                            tuning: tuning,
                            requestMaxTokensDescription: requestMaxTokensDescription(
                                provider: provider,
                                usesResponsesAPI: false,
                                tuning: tuning
                            )
                        )
                        let streamed = try await completeStreaming(
                            request: streamingRequest,
                            provider: provider,
                            endpointValue: endpointValue,
                            requestStartedAt: requestStartedAt,
                            attempt: index + 1,
                            endpointCount: endpoints.count,
                            onPartialText: onPartialText
                        )
                        let trimmed = streamed.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !trimmed.isEmpty else {
                            throw NSError(domain: "Voxt.RemoteLLM", code: -306, userInfo: [NSLocalizedDescriptionKey: "Remote LLM returned no text content."])
                        }
                        return trimmed
                    } catch let streamingFailure as StreamingFailure where streamingFailure.emittedChunkCount == 0 {
                        try Task.checkCancellation()
                        VoxtLog.llmWarning(
                            "Remote LLM streaming unavailable, retrying non-streaming. provider=\(provider.rawValue), endpoint=\(endpointValue), attempt=\(index + 1)/\(endpoints.count), detail=\(streamingFailure.underlying.localizedDescription)"
                        )
                    } catch {
                        throw error
                    }
                }

                let request = try makeCompletionRequest(
                    provider: provider,
                    runtimeConfiguration: runtimeConfiguration,
                    endpointValue: endpointValue,
                    model: model,
                    systemPrompt: systemPrompt,
                    userPrompt: userPrompt,
                    messagesOverride: messagesOverride,
                    openAICompatibleResponseFormat: openAICompatibleResponseFormat,
                    tuning: tuning,
                    streamingEnabled: false
                )
                let requestStartedAt = Date()
                logRequest(
                    request: request,
                    provider: provider,
                    endpointValue: endpointValue,
                    model: model,
                    inputTextLength: inputTextLength,
                    systemPrompt: systemPrompt,
                    debugInput: debugInput,
                    userPrompt: userPrompt,
                    tuning: tuning,
                    requestMaxTokensDescription: requestMaxTokensDescription(
                        provider: provider,
                        usesResponsesAPI: false,
                        tuning: tuning
                    )
                )
                let (data, response) = try await networkSession.data(for: request)
                let responseElapsedMs = Int(Date().timeIntervalSince(requestStartedAt) * 1000)
                guard let http = response as? HTTPURLResponse else {
                    throw NSError(domain: "Voxt.RemoteLLM", code: -305, userInfo: [NSLocalizedDescriptionKey: "Invalid remote LLM response."])
                }
                guard (200...299).contains(http.statusCode) else {
                    throw NSError(
                        domain: "Voxt.RemoteLLM",
                        code: http.statusCode,
                        userInfo: [NSLocalizedDescriptionKey: "Remote LLM request failed (HTTP \(http.statusCode))."]
                    )
                }

                let decodeStartedAt = Date()
                let object = try JSONSerialization.jsonObject(with: data)
                let decodeElapsedMs = Int(Date().timeIntervalSince(decodeStartedAt) * 1000)
                let totalElapsedMs = Int(Date().timeIntervalSince(requestStartedAt) * 1000)
                let attempt = index + 1
                if let errorMessage = extractStreamingErrorMessage(from: object) {
                    throw NSError(
                        domain: "Voxt.RemoteLLM",
                        code: -307,
                        userInfo: [NSLocalizedDescriptionKey: errorMessage]
                    )
                }
                if let content = extractPrimaryText(from: object), !content.isEmpty {
                    let guardedContent = guardRepeatedOutputIfNeeded(
                        content,
                        provider: provider,
                        endpointValue: endpointValue,
                        context: "response"
                    )
                    VoxtLog.llmInfo(
                        "Remote LLM response received. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode), attempt=\(attempt)/\(endpoints.count), bytes=\(data.count), networkMs=\(responseElapsedMs), decodeMs=\(decodeElapsedMs), totalMs=\(totalElapsedMs)"
                    )
                    VoxtLog.llm(
                        """
                        Remote LLM response content. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode)
                        [output]
                        \(VoxtLog.llmPreview(guardedContent))
                        """
                    )
                    return guardedContent
                }

                VoxtLog.llmWarning(
                    "Remote LLM response has no usable text. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode), attempt=\(attempt)/\(endpoints.count), bytes=\(data.count), networkMs=\(responseElapsedMs), decodeMs=\(decodeElapsedMs), totalMs=\(totalElapsedMs)"
                )
                throw NSError(domain: "Voxt.RemoteLLM", code: -306, userInfo: [NSLocalizedDescriptionKey: "Remote LLM returned no text content."])
            } catch {
                lastError = error
                let elapsedMs = Int(Date().timeIntervalSince(attemptStartedAt) * 1000)
                let nsError = error as NSError
                let isTimeout = nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorTimedOut
                let detail = networkErrorDetail(error: nsError)
                let attempt = index + 1
                let requestTimeout = requestTimeoutInterval(for: provider)
                let resolvedURL = URL(string: streamingEndpointValue(
                    provider: provider,
                    endpoint: endpointValue,
                    model: model,
                    streamingEnabled: shouldAttemptStreaming
                )) ?? URL(string: endpointValue)
                let proxyRoute = resolvedURL.map {
                    resolvedProxyRoute(for: $0, settings: VoxtNetworkSession.currentProxySettings)
                } ?? "unavailable"
                if isTimeout {
                    VoxtLog.llmWarning("Remote LLM request timeout. provider=\(provider.rawValue), endpoint=\(endpointValue), attempt=\(attempt)/\(endpoints.count), elapsedMs=\(elapsedMs), timeoutSec=\(Int(requestTimeout)), proxy=\(proxyRoute), detail=\(detail)")
                } else {
                    VoxtLog.llmWarning("Remote LLM request failed. provider=\(provider.rawValue), endpoint=\(endpointValue), attempt=\(attempt)/\(endpoints.count), elapsedMs=\(elapsedMs), proxy=\(proxyRoute), detail=\(detail)")
                }

                let hasNext = index < endpoints.count - 1
                if hasNext && shouldRetry(error: error, provider: provider) {
                    VoxtLog.llmWarning("Remote LLM request failed on endpoint \(endpointValue); retrying next endpoint. attempt=\(attempt)/\(endpoints.count), reason=\(error.localizedDescription)")
                    continue
                }
                throw error
            }
        }

        throw lastError ?? NSError(domain: "Voxt.RemoteLLM", code: -306, userInfo: [NSLocalizedDescriptionKey: "Remote LLM returned no text content."])
    }

    private func completeStreaming(
        request: URLRequest,
        provider: RemoteLLMProvider,
        endpointValue: String,
        requestStartedAt: Date,
        attempt: Int,
        endpointCount: Int,
        onPartialText: @Sendable (String) -> Void
    ) async throws -> String {
        var aggregated = ""
        var emittedChunkCount = 0
        var partialDeliveryState = StreamingPartialDeliveryState()
        let repetitionGuard = LLMOutputRepetitionGuard()
        var didStopForRepetition = false
        do {
            let (bytes, response) = try await networkSession.bytes(for: request)
            defer { bytes.task.cancel() }
            guard let http = response as? HTTPURLResponse else {
                throw NSError(domain: "Voxt.RemoteLLM", code: -305, userInfo: [NSLocalizedDescriptionKey: "Invalid remote LLM response."])
            }
            guard (200...299).contains(http.statusCode) else {
                throw NSError(
                    domain: "Voxt.RemoteLLM",
                    code: http.statusCode,
                    userInfo: [NSLocalizedDescriptionKey: "Remote LLM request failed (HTTP \(http.statusCode)) while opening stream."]
                )
            }

            var bufferedEventLines: [String] = []
            var sawEventStreamMarkers = false
            var nonEventStreamBuffer = ""

            func publishAggregated(force: Bool = false) {
                guard partialDeliveryState.shouldPublish(aggregatedText: aggregated, force: force) else { return }
                partialDeliveryState.markPublished(aggregatedText: aggregated)
                onPartialText(aggregated)
            }

            func publish(_ chunkPayload: String) throws {
                let trimmed = chunkPayload.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty, trimmed != "[DONE]" else { return }

                if let data = trimmed.data(using: .utf8),
                   let object = try? JSONSerialization.jsonObject(with: data) {
                    if let errorMessage = extractStreamingErrorMessage(from: object) {
                        throw NSError(
                            domain: "Voxt.RemoteLLM",
                            code: -307,
                            userInfo: [NSLocalizedDescriptionKey: errorMessage]
                        )
                    }
                    if let delta = extractStreamingDelta(from: object), !delta.isEmpty {
                        aggregated.append(delta)
                    } else if let snapshot = extractPrimaryText(from: object), !snapshot.isEmpty {
                        aggregated = mergedStreamingSnapshot(current: aggregated, next: snapshot)
                    } else {
                        return
                    }
                } else if let recovered = recoverStreamingDelta(fromRawPayload: trimmed), !recovered.isEmpty {
                    aggregated.append(recovered)
                } else if looksLikeStreamingEnvelopeFragment(trimmed) {
                    return
                } else {
                    aggregated = mergedStreamingSnapshot(current: aggregated, next: trimmed)
                }

                emittedChunkCount += 1
                if let repetition = repetitionGuard.repeatedSuffix(in: aggregated) {
                    aggregated = repetition.truncatedText
                    didStopForRepetition = true
                    VoxtLog.llmWarning(
                        "Remote LLM streaming repetition guard stopped generation. provider=\(provider.rawValue), endpoint=\(endpointValue), attempt=\(attempt)/\(endpointCount), repeatedUnitChars=\(repetition.repeatedUnit.count), repetitions=\(repetition.repetitionCount), outputChars=\(aggregated.count)"
                    )
                    publishAggregated(force: true)
                    return
                }
                publishAggregated()
            }

            for try await line in bytes.lines {
                try Task.checkCancellation()
                let trimmedLine = line.trimmingCharacters(in: .newlines)
                if trimmedLine.isEmpty {
                    if !bufferedEventLines.isEmpty {
                        try publish(bufferedEventLines.joined(separator: "\n"))
                        bufferedEventLines.removeAll(keepingCapacity: true)
                        if didStopForRepetition { break }
                    }
                    continue
                }

                if trimmedLine.hasPrefix(":") {
                    continue
                }

                if trimmedLine.hasPrefix("event:") || trimmedLine.hasPrefix("id:") || trimmedLine.hasPrefix("retry:") {
                    sawEventStreamMarkers = true
                    continue
                }

                if trimmedLine.hasPrefix("data:") {
                    sawEventStreamMarkers = true
                    var payload = String(trimmedLine.dropFirst(5))
                    if payload.hasPrefix(" ") {
                        payload.removeFirst()
                    }
                    bufferedEventLines.append(payload)
                    if shouldFlushBufferedEventLines(bufferedEventLines) {
                        try publish(bufferedEventLines.joined(separator: "\n"))
                        bufferedEventLines.removeAll(keepingCapacity: true)
                        if didStopForRepetition { break }
                    }
                    continue
                }

                if sawEventStreamMarkers {
                    bufferedEventLines.append(trimmedLine)
                    if shouldFlushBufferedEventLines(bufferedEventLines) {
                        try publish(bufferedEventLines.joined(separator: "\n"))
                        bufferedEventLines.removeAll(keepingCapacity: true)
                        if didStopForRepetition { break }
                    }
                } else {
                    nonEventStreamBuffer.append(line)
                    nonEventStreamBuffer.append("\n")
                    let payloads = drainNonEventStreamPayloads(buffer: &nonEventStreamBuffer)
                    for payload in payloads {
                        try publish(payload)
                        if didStopForRepetition { break }
                    }
                    if didStopForRepetition { break }
                }
            }

            if !didStopForRepetition, !bufferedEventLines.isEmpty {
                try publish(bufferedEventLines.joined(separator: "\n"))
            }
            if !didStopForRepetition {
                let trailingPayloads = drainNonEventStreamPayloads(buffer: &nonEventStreamBuffer)
                for payload in trailingPayloads {
                    try publish(payload)
                    if didStopForRepetition { break }
                }
            }
            let trailingText = didStopForRepetition ? "" : nonEventStreamBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trailingText.isEmpty {
                try publish(trailingText)
            }
            try Task.checkCancellation()
            publishAggregated(force: true)

            let totalElapsedMs = Int(Date().timeIntervalSince(requestStartedAt) * 1000)
            VoxtLog.llmInfo(
                "Remote LLM streaming response received. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode), attempt=\(attempt)/\(endpointCount), chunks=\(emittedChunkCount), totalMs=\(totalElapsedMs)"
            )
            VoxtLog.llm(
                """
                Remote LLM streaming content. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode)
                [output]
                \(VoxtLog.llmPreview(aggregated))
                """
            )
            return aggregated
        } catch {
            try Task.checkCancellation()
            let nsError = error as NSError
            if error is CancellationError || (nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled) {
                throw CancellationError()
            }
            throw StreamingFailure(
                underlying: error,
                partialText: aggregated,
                emittedChunkCount: emittedChunkCount
            )
        }
    }

    private func shouldRetry(error: Error, provider: RemoteLLMProvider) -> Bool {
        guard provider == .zai else { return false }
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain {
            return [
                NSURLErrorTimedOut,
                NSURLErrorCannotConnectToHost,
                NSURLErrorNetworkConnectionLost,
                NSURLErrorNotConnectedToInternet
            ].contains(nsError.code)
        }
        if nsError.domain == "Voxt.RemoteLLM" {
            return (500...599).contains(nsError.code)
        }
        return false
    }

    private func resolvedEndpointCandidates(provider: RemoteLLMProvider, primaryEndpoint: String) -> [String] {
        guard provider == .zai else { return [primaryEndpoint] }

        var values: [String] = [primaryEndpoint]
        if let alternate = alternateZAIEndpoint(from: primaryEndpoint),
           alternate.caseInsensitiveCompare(primaryEndpoint) != .orderedSame {
            values.append(alternate)
        }
        return values
    }

    private func alternateZAIEndpoint(from endpoint: String) -> String? {
        guard var components = URLComponents(string: endpoint) else {
            return "https://api.z.ai/api/paas/v4/chat/completions"
        }
        let host = (components.host ?? "").lowercased()
        if host == "open.bigmodel.cn" {
            components.host = "api.z.ai"
            return components.string
        }
        if host == "api.z.ai" {
            components.host = "open.bigmodel.cn"
            return components.string
        }
        if host.hasSuffix("bigmodel.cn") {
            components.host = "api.z.ai"
            return components.string
        }
        return "https://api.z.ai/api/paas/v4/chat/completions"
    }
}
