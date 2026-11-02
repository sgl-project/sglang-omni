import Foundation

extension RemoteASRTranscriber {
    func transcribeOpenAI(
        fileURL: URL,
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload
    ) async throws -> String {
        let endpoint = URL(string: RemoteASREndpointSupport.normalizedEndpoint(configuration.endpoint, defaultValue: "https://api.openai.com/v1/audio/transcriptions"))!
        let token = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw NSError(domain: "Voxt.RemoteASR", code: -1, userInfo: [NSLocalizedDescriptionKey: "OpenAI API key is empty."])
        }
        return try await transcribeOpenAIJSON(
            endpoint: endpoint,
            authorizationValue: "Bearer \(token)",
            fileURL: fileURL,
            model: configuration.model,
            hintPayload: hintPayload
        )
    }

    private func transcribeOpenAIJSON(
        endpoint: URL,
        authorizationValue: String,
        fileURL: URL,
        model: String,
        hintPayload: ResolvedASRHintPayload
    ) async throws -> String {
        let boundary = "Boundary-\(UUID().uuidString)"
        let effectiveModel = model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? RemoteASRProvider.openAIWhisper.suggestedModel
            : model
        let extraFields = RemoteASRTextSupport.openAITranscriptionMultipartFields(
            model: effectiveModel,
            hintPayload: hintPayload
        )
        let body = try makeMultipartFileBody(
            fileURL: fileURL,
            boundary: boundary,
            model: effectiveModel,
            extraFields: extraFields
        )
        defer { body.remove() }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/plain", forHTTPHeaderField: "Accept")
        request.setValue(authorizationValue, forHTTPHeaderField: "Authorization")
        request.setValue(String(body.byteCount), forHTTPHeaderField: "Content-Length")

        let (data, response) = try await VoxtNetworkSession.active.upload(for: request, fromFile: body.url)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "Voxt.RemoteASR", code: -10, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response."])
        }
        guard (200...299).contains(http.statusCode) else {
            let payload = String(data: data.prefix(500), encoding: .utf8) ?? ""
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode): \(payload)"]
            )
        }

        if let object = try? JSONSerialization.jsonObject(with: data),
           let text = RemoteASRTextSupport.extractText(in: object),
           !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let plainText = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !plainText.isEmpty, !RemoteASRTextSupport.isLikelyJSONObjectString(plainText) {
            return plainText
        }

        throw NSError(
            domain: "Voxt.RemoteASR",
            code: -11,
            userInfo: [NSLocalizedDescriptionKey: "OpenAI transcription response did not contain text."]
        )
    }

    func transcribeGLM(
        fileURL: URL,
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload
    ) async throws -> String {
        let endpoint = URL(string: RemoteASREndpointSupport.normalizedEndpoint(configuration.endpoint, defaultValue: "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"))!
        let token = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw NSError(domain: "Voxt.RemoteASR", code: -2, userInfo: [NSLocalizedDescriptionKey: "GLM API key is empty."])
        }
        var extraFields = ["stream": "true"]
        if let prompt = hintPayload.prompt?.trimmingCharacters(in: .whitespacesAndNewlines), !prompt.isEmpty {
            extraFields["prompt"] = prompt
        }
        return try await transcribeViaMultipartStream(
            endpoint: endpoint,
            authorizationValue: "Bearer \(token)",
            fileURL: fileURL,
            model: configuration.model,
            extraFields: extraFields
        )
    }

    func transcribeXiaomiMiMo(
        fileURL: URL,
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload
    ) async throws -> String {
        let endpointValue = RemoteASREndpointSupport.resolvedXiaomiMiMoASREndpoint(configuration.endpoint)
        guard let endpoint = URL(string: endpointValue) else {
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: -70,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid Xiaomi MiMo ASR endpoint URL.")]
            )
        }

        let token = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: -71,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Xiaomi MiMo API key is empty.")]
            )
        }

        let configuredModel = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let model = configuredModel.isEmpty ? RemoteASRProvider.xiaomiMiMoASR.suggestedModel : configuredModel
        let audioData = try Data(contentsOf: fileURL)
        let payload = RemoteASRTextSupport.xiaomiMiMoASRPayload(
            model: model,
            audioData: audioData,
            mimeType: RemoteASREndpointSupport.audioMIMEType(for: fileURL),
            hintPayload: hintPayload
        )

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (data, response) = try await VoxtNetworkSession.active.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: -72,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid Xiaomi MiMo ASR HTTP response.")]
            )
        }
        guard (200...299).contains(http.statusCode) else {
            let message = String(data: data.prefix(500), encoding: .utf8) ?? ""
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: http.statusCode,
                userInfo: [
                    NSLocalizedDescriptionKey: AppLocalization.format(
                        "Xiaomi MiMo ASR request failed (HTTP %d): %@",
                        http.statusCode,
                        message
                    )
                ]
            )
        }

        let object = try JSONSerialization.jsonObject(with: data)
        if let text = RemoteASRTextSupport.extractText(in: object),
           let normalized = RemoteASRTextSupport.normalizedTextFragment(text),
           !normalized.isEmpty {
            return normalized
        }
        throw NSError(
            domain: "Voxt.RemoteASR",
            code: -73,
            userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Xiaomi MiMo ASR returned no text content.")]
        )
    }

    func transcribeStepFun(
        fileURL: URL,
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload
    ) async throws -> String {
        let endpoint = URL(string: RemoteASREndpointSupport.normalizedEndpoint(
            configuration.endpoint,
            defaultValue: "https://api.stepfun.com/v1/audio/asr/sse"
        ))!

        let token = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: -6,
                userInfo: [NSLocalizedDescriptionKey: "StepFun API key is empty."]
            )
        }

        let configuredModel = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let model = configuredModel.isEmpty
            ? RemoteASRProvider.stepFunASR.suggestedModel
            : configuredModel

        let wavData = try Data(contentsOf: fileURL)
        let pcmData = try StepFunSupport.extractPCMData(fromWAV: wavData)
        let base64Audio = pcmData.base64EncodedString()

        let body: [String: Any] = [
            "audio": [
                "data": base64Audio,
                "input": [
                    "transcription": StepFunPayloadSupport.transcriptionPayload(
                        model: model,
                        hintPayload: hintPayload,
                        includePrompt: StepFunPayloadSupport.supportsSSEPrompt(model: model)
                    ),
                    "format": StepFunPayloadSupport.audioFormatPayload()
                ]
            ]
        ]

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (bytes, response) = try await VoxtNetworkSession.active.bytes(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: -10,
                userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response."]
            )
        }

        if !(200...299).contains(http.statusCode) {
            let payload = try await RemoteASRTextSupport.collectText(from: bytes)
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode): \(payload)"]
            )
        }

        var previewText = ""
        var finalText: String?
        var sseEvent: String?
        for try await rawLine in bytes.lines {
            let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }

            if trimmed.hasPrefix("event:") {
                sseEvent = String(trimmed.dropFirst(6))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                continue
            }

            let line: String
            if trimmed.hasPrefix("data:") {
                line = String(trimmed.dropFirst(5)).trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                line = trimmed
            }

            if line == "[DONE]" { break }

            if sseEvent == "error" {
                let message = RemoteASRTextSupport.extractStreamErrorMessage(fromLine: line) ?? line
                throw NSError(
                    domain: "Voxt.RemoteASR",
                    code: -11,
                    userInfo: [NSLocalizedDescriptionKey: "StepFun ASR stream error: \(message)"]
                )
            }

            if let message = RemoteASRTextSupport.extractStreamErrorMessage(fromLine: line) {
                throw NSError(
                    domain: "Voxt.RemoteASR",
                    code: -11,
                    userInfo: [NSLocalizedDescriptionKey: "StepFun ASR stream error: \(message)"]
                )
            }

            switch StepFunPayloadSupport.parseSSEDataLine(line) {
            case .delta(let fragment), .fragment(let fragment):
                previewText = RemoteASRTextSupport.mergeStreamFragment(current: previewText, incoming: fragment)
                await MainActor.run {
                    self.publishIntermediateTranscription(previewText)
                }
            case .completed(let text):
                finalText = text
                previewText = text
                await MainActor.run {
                    self.publishIntermediateTranscription(text)
                }
            case .error(let message):
                throw NSError(
                    domain: "Voxt.RemoteASR",
                    code: -11,
                    userInfo: [NSLocalizedDescriptionKey: "StepFun ASR stream error: \(message)"]
                )
            case .ignore:
                break
            }
            sseEvent = nil
        }

        if let finalText, !finalText.isEmpty { return finalText }
        if !previewText.isEmpty { return previewText }
        return transcribedText
    }

    func transcribeDoubao(
        fileURL: URL,
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload
    ) async throws -> String {
        let accessToken = configuration.accessToken.trimmingCharacters(in: .whitespacesAndNewlines)
        let appID = configuration.appID.trimmingCharacters(in: .whitespacesAndNewlines)
        let resourceID = RemoteASREndpointSupport.resolvedDoubaoResourceID(from: configuration)
        let endpoint = RemoteASREndpointSupport.resolvedDoubaoStreamingEndpoint(from: configuration)

        guard !accessToken.isEmpty else {
            throw NSError(domain: "Voxt.RemoteASR", code: -3, userInfo: [NSLocalizedDescriptionKey: "Doubao Access Token is empty."])
        }
        guard !appID.isEmpty else {
            throw NSError(domain: "Voxt.RemoteASR", code: -4, userInfo: [NSLocalizedDescriptionKey: "Doubao App ID is empty."])
        }
        if DoubaoASRConfiguration.isFlashRecognitionModel(resourceID) {
            return try await transcribeDoubaoFlashRecognition(
                fileURL: fileURL,
                appID: appID,
                accessToken: accessToken,
                resourceID: resourceID,
                endpoint: DoubaoASRConfiguration.resolvedFlashRecognitionEndpoint(configuration.endpoint),
                hintPayload: hintPayload,
                configuration: configuration
            )
        }
        return try await transcribeDoubaoStreamingFileWebSocket(
            fileURL: fileURL,
            appID: appID,
            accessToken: accessToken,
            resourceID: resourceID,
            endpoint: endpoint,
            hintPayload: hintPayload,
            configuration: configuration
        )
    }

    private func transcribeDoubaoFlashRecognition(
        fileURL: URL,
        appID: String,
        accessToken: String,
        resourceID: String,
        endpoint: String,
        hintPayload: ResolvedASRHintPayload,
        configuration: RemoteProviderConfiguration
    ) async throws -> String {
        guard let url = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -34, userInfo: [NSLocalizedDescriptionKey: "Invalid Doubao ASR endpoint URL."])
        }

        let audioData = try Data(contentsOf: fileURL)
        var body = doubaoRequestPayload(
            configuration: configuration,
            hintPayload: hintPayload,
            requestID: UUID().uuidString.lowercased(),
            userID: "voxt-transcript",
            audioFormat: DoubaoASRConfiguration.requestAudioFormat
        )
        body["audio"] = ["data": audioData.base64EncodedString()]

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(appID, forHTTPHeaderField: "X-Api-App-Key")
        request.setValue(accessToken, forHTTPHeaderField: "X-Api-Access-Key")
        request.setValue(resourceID, forHTTPHeaderField: "X-Api-Resource-Id")
        request.setValue(UUID().uuidString.lowercased(), forHTTPHeaderField: "X-Api-Request-Id")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await VoxtNetworkSession.active.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "Voxt.RemoteASR", code: -35, userInfo: [NSLocalizedDescriptionKey: "Invalid Doubao ASR HTTP response."])
        }
        guard (200...299).contains(http.statusCode) else {
            let payload = String(data: data.prefix(500), encoding: .utf8) ?? ""
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "Doubao ASR request failed (HTTP \(http.statusCode)): \(payload)"]
            )
        }

        let object = try JSONSerialization.jsonObject(with: data)
        if let text = RemoteASRTextSupport.extractDoubaoText(in: object), !text.isEmpty {
            return text
        }
        if let text = RemoteASRTextSupport.extractText(in: object), !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return ""
    }

    func transcribeAliyunBailian(fileURL: URL, configuration: RemoteProviderConfiguration) async throws -> String {
        let configuredModel = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let model = configuredModel.isEmpty
            ? RemoteASRProvider.aliyunBailianASR.suggestedModel
            : configuredModel
        guard RemoteASREndpointSupport.isAliyunFunRealtimeModel(model)
                || RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: model) != nil
                || RemoteASREndpointSupport.isAliyunFileTranscriptionModel(model)
                || AliyunRemoteASRConfiguration.routing(for: model) == .compatibleShortAudio
        else {
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: -33,
                userInfo: [NSLocalizedDescriptionKey: "Aliyun ASR in Voxt supports Qwen/Omni/Fun/Paraformer transcription models only."]
            )
        }

        let token = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw NSError(domain: "Voxt.RemoteASR", code: -30, userInfo: [NSLocalizedDescriptionKey: "Aliyun Bailian API key is empty."])
        }
        if RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: model) != nil {
            return try await transcribeAliyunQwenRealtimeFile(
                fileURL: fileURL,
                token: token,
                model: model,
                endpoint: configuration.endpoint,
                hintPayload: resolvedHintPayload(for: .aliyunBailianASR, configuration: configuration)
            )
        }
        if RemoteASREndpointSupport.isAliyunFunRealtimeModel(model) {
            return try await transcribeAliyunFunRealtimeFile(
                fileURL: fileURL,
                token: token,
                model: model,
                endpoint: configuration.endpoint,
                hintPayload: resolvedHintPayload(for: .aliyunBailianASR, configuration: configuration),
                settings: configuration.aliyunASRSettings
            )
        }
        if let validationError = AliyunRemoteASRConfiguration.validationError(model: model, endpoint: configuration.endpoint) {
            throw NSError(domain: "Voxt.RemoteASR", code: -36, userInfo: [NSLocalizedDescriptionKey: validationError])
        }
        if RemoteASREndpointSupport.isAliyunFileTranscriptionModel(model) {
            return try await AliyunRemoteASRClient.transcribe(
                fileURL: fileURL,
                apiKey: token,
                model: model,
                endpoint: configuration.endpoint
            )
        }
        let endpoint = URL(string: AliyunRemoteASRConfiguration.resolvedCompatibleEndpoint(configuration.endpoint, model: model))!
        let fileData = try Data(contentsOf: fileURL)
        let dataURI = "data:\(RemoteASREndpointSupport.audioMIMEType(for: fileURL));base64,\(fileData.base64EncodedString())"

        let payload: [String: Any] = [
            "model": model,
            "messages": [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "input_audio",
                            "input_audio": [
                                "data": dataURI,
                                "format": "wav"
                            ]
                        ]
                    ]
                ]
            ],
            "stream": false
        ]

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (data, response) = try await VoxtNetworkSession.active.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "Voxt.RemoteASR", code: -31, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun Bailian HTTP response."])
        }
        guard (200...299).contains(http.statusCode) else {
            let message = String(data: data.prefix(300), encoding: .utf8) ?? ""
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "Aliyun Bailian ASR request failed (HTTP \(http.statusCode)): \(message)"]
            )
        }

        let object = try JSONSerialization.jsonObject(with: data)
        if let text = AliyunRemoteASRClient.extractText(from: object), !text.isEmpty {
            return text
        }
        throw NSError(domain: "Voxt.RemoteASR", code: -32, userInfo: [NSLocalizedDescriptionKey: "Aliyun Bailian ASR returned no text content."])
    }


    private func transcribeViaMultipartStream(
        endpoint: URL,
        authorizationValue: String,
        fileURL: URL,
        model: String,
        extraFields: [String: String]
    ) async throws -> String {
        let boundary = "Boundary-\(UUID().uuidString)"
        let effectiveModel = model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? RemoteASRProvider.glmASR.suggestedModel
            : model
        let body = try makeMultipartFileBody(
            fileURL: fileURL,
            boundary: boundary,
            model: effectiveModel,
            extraFields: extraFields
        )
        defer { body.remove() }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream, application/json, text/plain", forHTTPHeaderField: "Accept")
        request.setValue(authorizationValue, forHTTPHeaderField: "Authorization")
        request.setValue(String(body.byteCount), forHTTPHeaderField: "Content-Length")
        request.httpBodyStream = InputStream(url: body.url)

        let (bytes, response) = try await VoxtNetworkSession.active.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "Voxt.RemoteASR", code: -10, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response."])
        }

        if !(200...299).contains(http.statusCode) {
            let payload = try await RemoteASRTextSupport.collectText(from: bytes)
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode): \(payload)"]
            )
        }

        var aggregate = ""
        for try await rawLine in bytes.lines {
            let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }

            let line: String
            if trimmed.hasPrefix("data:") {
                line = String(trimmed.dropFirst(5)).trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                line = trimmed
            }

            if line == "[DONE]" {
                break
            }

            if let fragment = RemoteASRTextSupport.extractTextFragment(fromLine: line), !fragment.isEmpty {
                aggregate = RemoteASRTextSupport.mergeStreamFragment(current: aggregate, incoming: fragment)
                await MainActor.run {
                    self.publishIntermediateTranscription(aggregate)
                }
            }
        }

        if aggregate.isEmpty {
            return transcribedText
        }
        return aggregate
    }

    private func makeMultipartFileBody(
        fileURL: URL,
        boundary: String,
        model: String,
        extraFields: [String: String]
    ) throws -> MultipartFileBody {
        let fields = [(name: "model", value: model)] + extraFields
            .sorted(by: { $0.key < $1.key })
            .map { (name: $0.key, value: $0.value) }
        return try MultipartFileBody.create(
            sourceFileURL: fileURL,
            boundary: boundary,
            fields: fields,
            mimeType: "audio/wav"
        )
    }
}
