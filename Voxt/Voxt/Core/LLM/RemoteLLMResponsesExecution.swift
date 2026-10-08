// Responses API execution, response IDs, and streaming fallback.

import Foundation

extension RemoteLLMRuntimeClient {
    struct ResponsesStreamingResult {
        let text: String
        let responseID: String?
    }

    func completeResponses(
        systemPrompt: String,
        debugInput: String,
        requestContentForLog: String,
        inputPayload: Any,
        inputTextLength: Int,
        intent: CompletionIntent,
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration,
        previousResponseID: String? = nil,
        textFormat: [String: Any]? = nil,
        onPartialText: (@Sendable (String) -> Void)? = nil,
        onResponseID: ((String) -> Void)? = nil
    ) async throws -> ResponsesStreamingResult {
        try Task.checkCancellation()
        let runtimeConfiguration = try RemoteModelConfigurationStore.runtimeConfiguration(for: configuration)
        let configuration = runtimeConfiguration.value
        try validateEndpointSecurity(provider: provider, configuration: configuration)
        let model = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? provider.suggestedModel
            : configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let endpointValue = responsesEndpointValue(provider: provider, endpoint: configuration.endpoint, model: model)
        let tuning = generationTuning(
            for: provider,
            inputTextLength: inputTextLength,
            systemPromptLength: systemPrompt.count,
            userPromptLength: requestContentForLog.count,
            intent: intent
        ).applying(configuration.effectiveGenerationSettings(provider: provider))
        let requiresStreaming = requiresResponsesStreaming(provider: provider)
        let shouldAttemptStreaming = (onPartialText != nil || requiresStreaming) && supportsStreaming(provider: provider, intent: intent)
        let authHeaders = try await authorizationHeaders(provider: provider, configuration: configuration)

        if shouldAttemptStreaming {
            do {
                let streamingRequest = try makeResponsesRequest(
                    provider: provider,
                    endpointValue: endpointValue,
                    model: model,
                    systemPrompt: systemPrompt,
                    inputPayload: inputPayload,
                    runtimeConfiguration: runtimeConfiguration,
                    previousResponseID: previousResponseID,
                    tuning: tuning,
                    textFormat: textFormat,
                    streamingEnabled: true,
                    additionalHeaders: authHeaders
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
                    userPrompt: requestContentForLog,
                    tuning: tuning,
                    requestMaxTokensDescription: requestMaxTokensDescription(
                        provider: provider,
                        usesResponsesAPI: true,
                        tuning: tuning
                    )
                )
                return try await completeResponsesStreaming(
                    request: streamingRequest,
                    provider: provider,
                    endpointValue: endpointValue,
                    requestStartedAt: requestStartedAt,
                    onPartialText: onPartialText ?? { _ in },
                    onResponseID: onResponseID
                )
            } catch let streamingFailure as StreamingFailure where streamingFailure.emittedChunkCount == 0 {
                try Task.checkCancellation()
                if requiresStreaming {
                    throw streamingFailure.underlying
                }
                VoxtLog.llmWarning(
                    "Remote LLM Responses streaming unavailable, retrying non-streaming. provider=\(provider.rawValue), endpoint=\(endpointValue), detail=\(streamingFailure.underlying.localizedDescription)"
                )
            }
        }

        let request = try makeResponsesRequest(
            provider: provider,
            endpointValue: endpointValue,
            model: model,
            systemPrompt: systemPrompt,
            inputPayload: inputPayload,
            runtimeConfiguration: runtimeConfiguration,
            previousResponseID: previousResponseID,
            tuning: tuning,
            textFormat: textFormat,
            streamingEnabled: false,
            additionalHeaders: authHeaders
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
            userPrompt: requestContentForLog,
            tuning: tuning,
            requestMaxTokensDescription: requestMaxTokensDescription(
                provider: provider,
                usesResponsesAPI: true,
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
        let object: [String: Any]
        do {
            object = try decodeResponsesObject(from: data, response: http)
        } catch {
            let payloadPreview = responsePayloadPreview(from: data)
            VoxtLog.llmWarning(
                "Remote LLM Responses response rejected. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode), bytes=\(data.count), payloadChars=\(payloadPreview.count), detail=\(error.localizedDescription)"
            )
            throw error
        }
        let decodeElapsedMs = Int(Date().timeIntervalSince(decodeStartedAt) * 1000)
        let totalElapsedMs = Int(Date().timeIntervalSince(requestStartedAt) * 1000)

        if let errorMessage = extractStreamingErrorMessage(from: object) ?? responsesErrorMessage(from: object) {
            throw NSError(
                domain: "Voxt.RemoteLLM",
                code: -307,
                userInfo: [NSLocalizedDescriptionKey: errorMessage]
            )
        }

        if let completionIssue = responsesCompletionIssue(from: object) {
            VoxtLog.llmWarning(
                "Remote LLM Responses response incomplete. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode), detail=\(completionIssue)"
            )
            throw NSError(
                domain: "Voxt.RemoteLLM",
                code: -308,
                userInfo: [NSLocalizedDescriptionKey: completionIssue]
            )
        }

        guard let content = extractPrimaryText(from: object), !content.isEmpty else {
            let payloadPreview = responsePayloadPreview(from: data)
            VoxtLog.llmWarning(
                "Remote LLM Responses response has no usable text. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode), bytes=\(data.count), payloadChars=\(payloadPreview.count)"
            )
            throw NSError(domain: "Voxt.RemoteLLM", code: -306, userInfo: [NSLocalizedDescriptionKey: "Remote LLM returned no text content."])
        }
        if textFormat != nil, !isValidStructuredJSONObject(content) {
            VoxtLog.llmWarning(
                "Remote LLM Responses structured output is not a complete JSON object. provider=\(provider.rawValue), endpoint=\(endpointValue), outputChars=\(content.count)"
            )
            throw NSError(
                domain: "Voxt.RemoteLLM",
                code: -309,
                userInfo: [NSLocalizedDescriptionKey: "Remote LLM returned incomplete structured output."]
            )
        }

        if let responseID = responsesResponseID(from: object) {
            onResponseID?(responseID)
        }

        let guardedContent = guardRepeatedOutputIfNeeded(
            content,
            provider: provider,
            endpointValue: endpointValue,
            context: "Responses response"
        )

        VoxtLog.llmInfo(
            "Remote LLM Responses response received. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode), bytes=\(data.count), networkMs=\(responseElapsedMs), decodeMs=\(decodeElapsedMs), totalMs=\(totalElapsedMs)"
        )
        VoxtLog.llm(
            """
            Remote LLM Responses content. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode)
            [output]
            \(VoxtLog.llmPreview(guardedContent))
            """
        )
        return ResponsesStreamingResult(
            text: guardedContent,
            responseID: responsesResponseID(from: object)
        )
    }

    private func completeResponsesStreaming(
        request: URLRequest,
        provider: RemoteLLMProvider,
        endpointValue: String,
        requestStartedAt: Date,
        onPartialText: @escaping (String) -> Void,
        onResponseID: ((String) -> Void)?
    ) async throws -> ResponsesStreamingResult {
        var aggregated = ""
        var responseID: String?
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

            func publishAggregated(force: Bool = false) {
                guard partialDeliveryState.shouldPublish(aggregatedText: aggregated, force: force) else { return }
                partialDeliveryState.markPublished(aggregatedText: aggregated)
                onPartialText(aggregated)
            }

            func publish(_ chunkPayload: String) throws {
                let trimmed = chunkPayload.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty, trimmed != "[DONE]" else { return }
                guard let data = trimmed.data(using: .utf8),
                      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    return
                }

                if let extractedResponseID = responsesResponseID(from: object) {
                    responseID = extractedResponseID
                    onResponseID?(extractedResponseID)
                }

                if let errorMessage = extractStreamingErrorMessage(from: object) ?? responsesErrorMessage(from: object) {
                    throw NSError(
                        domain: "Voxt.RemoteLLM",
                        code: -307,
                        userInfo: [NSLocalizedDescriptionKey: errorMessage]
                    )
                }

                if let delta = responsesStreamingDelta(from: object), !delta.isEmpty {
                    let eventType = object["type"] as? String
                    if eventType == "response.output_text.delta" {
                        aggregated.append(delta)
                    } else {
                        aggregated = mergedStreamingSnapshot(current: aggregated, next: delta)
                    }
                    emittedChunkCount += 1
                    if let repetition = repetitionGuard.repeatedSuffix(in: aggregated) {
                        aggregated = repetition.truncatedText
                        didStopForRepetition = true
                        VoxtLog.llmWarning(
                            "Remote LLM Responses streaming repetition guard stopped generation. provider=\(provider.rawValue), endpoint=\(endpointValue), repeatedUnitChars=\(repetition.repeatedUnit.count), repetitions=\(repetition.repetitionCount), outputChars=\(aggregated.count)"
                        )
                        publishAggregated(force: true)
                        return
                    }
                    publishAggregated()
                }
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
                }
            }

            if !didStopForRepetition, !bufferedEventLines.isEmpty {
                try publish(bufferedEventLines.joined(separator: "\n"))
            }
            try Task.checkCancellation()
            publishAggregated(force: true)

            let totalElapsedMs = Int(Date().timeIntervalSince(requestStartedAt) * 1000)
            VoxtLog.llmInfo(
                "Remote LLM Responses streaming response received. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode), chunks=\(emittedChunkCount), totalMs=\(totalElapsedMs), responseID=\(responseID ?? "nil")"
            )
            VoxtLog.llm(
                """
                Remote LLM Responses streaming content. provider=\(provider.rawValue), endpoint=\(endpointValue), status=\(http.statusCode)
                [output]
                \(VoxtLog.llmPreview(aggregated))
                """
            )
            return ResponsesStreamingResult(text: aggregated, responseID: responseID)
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
}
