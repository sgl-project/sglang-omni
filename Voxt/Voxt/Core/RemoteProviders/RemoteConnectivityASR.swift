import Foundation

extension RemoteProviderConnectivityTester {
    func testASRProvider(_ provider: RemoteASRProvider, configuration: RemoteProviderConfiguration) async throws -> String {
        switch provider {
        case .doubaoASR:
            let token = configuration.accessToken
            guard !token.isEmpty else {
                throw NSError(domain: "Voxt.Settings", code: -1, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Doubao Access Token is required for testing.")])
            }
            guard !configuration.appID.isEmpty else {
                throw NSError(domain: "Voxt.Settings", code: -2, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Doubao App ID is required for testing.")])
            }
            let endpoint = RemoteProviderConnectivityTestEndpoints.resolvedDoubaoASREndpoint(configuration.endpoint, model: configuration.model)
            return try await testDoubaoStreamingReachability(
                endpoint: endpoint,
                appID: configuration.appID,
                accessToken: token,
                model: configuration.model
            )
        case .openAIWhisper:
            guard !configuration.apiKey.isEmpty else {
                throw NSError(domain: "Voxt.Settings", code: -3, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("OpenAI API Key is required for testing.")])
            }
            let endpoint = RemoteProviderConnectivityTestEndpoints.resolvedASRTranscriptionEndpoint(
                endpoint: configuration.endpoint,
                defaultValue: "https://api.openai.com/v1/audio/transcriptions"
            )
            return try await testASRMultipartReachability(
                endpoint: endpoint,
                headers: ["Authorization": "Bearer \(configuration.apiKey)"],
                model: configuration.model.isEmpty ? RemoteASRProvider.openAIWhisper.suggestedModel : configuration.model
            )
        case .glmASR:
            guard !configuration.apiKey.isEmpty else {
                throw NSError(domain: "Voxt.Settings", code: -4, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("GLM API Key is required for testing.")])
            }
            let endpoint = RemoteProviderConnectivityTestEndpoints.resolvedGLMASRTranscriptionEndpoint(
                endpoint: configuration.endpoint,
                defaultValue: "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
            )
            return try await testASRMultipartReachability(
                endpoint: endpoint,
                headers: ["Authorization": "Bearer \(configuration.apiKey)"],
                model: configuration.model.isEmpty ? "glm-asr-1" : configuration.model
            )
        case .aliyunBailianASR:
            guard !configuration.apiKey.isEmpty else {
                throw NSError(domain: "Voxt.Settings", code: -5, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Aliyun Bailian API Key is required for testing.")])
            }
            let model = configuration.model.isEmpty ? "fun-asr-realtime" : configuration.model
            if let kind = RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: model) {
                let endpoint = RemoteProviderConnectivityTestEndpoints.resolvedAliyunASRQwenRealtimeWebSocketEndpoint(
                    endpoint: configuration.endpoint,
                    model: model
                )
                return try await testAliyunASRQwenRealtimeWebSocketReachability(
                    endpoint: endpoint,
                    apiKey: configuration.apiKey,
                    kind: kind
                )
            }
            let endpoint = RemoteProviderConnectivityTestEndpoints.resolvedAliyunASRRealtimeWebSocketEndpoint(
                endpoint: configuration.endpoint,
                defaultValue: "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
            )
            return try await testAliyunASRRealtimeWebSocketReachability(
                endpoint: endpoint,
                apiKey: configuration.apiKey,
                model: model
            )
        case .stepFunASR:
            guard !configuration.apiKey.isEmpty else {
                throw NSError(
                    domain: "Voxt.Settings",
                    code: -7,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("StepFun API Key is required for testing.")]
                )
            }
            let endpoint = RemoteProviderConnectivityTestEndpoints.resolvedStepFunASREndpoint(configuration.endpoint)
            return try await testStepFunReachability(
                endpoint: endpoint,
                token: configuration.apiKey,
                model: configuration.model.isEmpty ? RemoteASRProvider.stepFunASR.suggestedModel : configuration.model
            )
        case .xiaomiMiMoASR:
            guard !configuration.apiKey.isEmpty else {
                throw NSError(
                    domain: "Voxt.Settings",
                    code: -8,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Xiaomi MiMo API Key is required for testing.")]
                )
            }
            return try await testXiaomiMiMoASRReachability(
                endpoint: RemoteProviderConnectivityTestEndpoints.resolvedXiaomiMiMoASREndpoint(configuration.endpoint),
                apiKey: configuration.apiKey,
                model: configuration.model.isEmpty ? RemoteASRProvider.xiaomiMiMoASR.suggestedModel : configuration.model
            )
        case .googleGeminiASR:
            guard !configuration.apiKey.isEmpty else {
                throw NSError(
                    domain: "Voxt.Settings",
                    code: -9,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Google Gemini API Key is required for testing.")]
                )
            }
            return try await testGeminiLiveReachability(
                endpoint: configuration.endpoint,
                apiKey: configuration.apiKey,
                model: configuration.model.isEmpty ? RemoteASRProvider.googleGeminiASR.suggestedModel : configuration.model
            )
        }
    }

    private func testASRMultipartReachability(
        endpoint: String,
        headers: [String: String],
        model: String
    ) async throws -> String {
        guard let url = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.Settings", code: -20, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid ASR endpoint URL.")])
        }
        let boundary = "Boundary-\(UUID().uuidString)"
        let body = makeASRTestMultipartBody(boundary: boundary, model: model)

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 15
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream, text/plain", forHTTPHeaderField: "Accept")
        for (key, value) in headers {
            request.setValue(value, forHTTPHeaderField: key)
        }
        RemoteProviderConnectivityTestLogging.logHTTPRequest(
            context: "ASR multipart test",
            request: request,
            bodyPreview: "multipart/form-data body bytes=\(body.count)"
        )

        let (data, response) = try await VoxtNetworkSession.active.upload(for: request, from: body)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "Voxt.Settings", code: -21, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid server response.")])
        }
        RemoteProviderConnectivityTestLogging.logHTTPResponse(context: "ASR multipart test", response: http, data: data)

        let payload = String(data: data.prefix(200), encoding: .utf8) ?? ""
        if (200...299).contains(http.statusCode) {
            return AppLocalization.format("Connection test succeeded (HTTP %d).", http.statusCode)
        }
        if http.statusCode == 400 || http.statusCode == 422 {
            return AppLocalization.format("Endpoint reachable (HTTP %d). Authentication and routing look valid.", http.statusCode)
        }
        if http.statusCode == 401 || http.statusCode == 403 {
            throw NSError(
                domain: "Voxt.Settings",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Server reachable, but authentication failed (HTTP %d). %@", http.statusCode, payload)]
            )
        }
        throw NSError(
            domain: "Voxt.Settings",
            code: http.statusCode,
            userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Connection failed (HTTP %d). %@", http.statusCode, payload)]
        )
    }

    private func testStepFunReachability(
        endpoint: String,
        token: String,
        model: String
    ) async throws -> String {
        guard URL(string: endpoint) != nil else {
            throw NSError(
                domain: "Voxt.Settings",
                code: -22,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid StepFun ASR endpoint URL.")]
            )
        }

        let silentWAV = silentTestWavData()
        let pcmData = (try? StepFunSupport.extractPCMData(fromWAV: silentWAV))
            ?? silentWAV.subdata(in: 44..<silentWAV.count)
        let base64Audio = pcmData.base64EncodedString()

        let body: [String: Any] = [
            "audio": [
                "data": base64Audio,
                "input": [
                    "transcription": [
                        "model": model,
                        "language": "zh",
                        "enable_itn": false
                    ],
                    "format": [
                        "type": "pcm",
                        "codec": "pcm_s16le",
                        "rate": 16000,
                        "bits": 16,
                        "channel": 1
                    ]
                ]
            ]
        ]

        return try await testJSONPOSTReachability(
            endpoint: endpoint,
            headers: stepFunReachabilityHeaders(token: token),
            body: body,
            successMessage: AppLocalization.localizedString("Connection test succeeded (StepFun ASR reachable).")
        )
    }

    func stepFunReachabilityHeaders(token: String) -> [String: String] {
        [
            "Accept": "text/event-stream",
            "Authorization": "Bearer \(token)"
        ]
    }

    private func testXiaomiMiMoASRReachability(
        endpoint: String,
        apiKey: String,
        model: String
    ) async throws -> String {
        let body = RemoteASRTextSupport.xiaomiMiMoASRPayload(
            model: model,
            audioData: silentTestWavData(),
            mimeType: "audio/wav",
            hintPayload: ResolvedASRHintPayload(language: "auto")
        )
        return try await testJSONPOSTReachability(
            endpoint: endpoint,
            headers: ["Authorization": "Bearer \(apiKey)"],
            body: body,
            successMessage: AppLocalization.localizedString("Connection test succeeded (Xiaomi MiMo ASR reachable).")
        )
    }

    private func makeASRTestMultipartBody(boundary: String, model: String) -> Data {
        var body = Data()

        func append(_ text: String) {
            body.append(text.data(using: .utf8) ?? Data())
        }

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"model\"\r\n\r\n")
        append("\(model)\r\n")

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(silentTestWavData())
        append("\r\n")

        append("--\(boundary)--\r\n")
        return body
    }

    func silentTestWavData() -> Data {
        var data = Data()
        let sampleRate: UInt32 = 16000
        let channels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let durationMs: UInt32 = 100
        let samples = sampleRate * durationMs / 1000
        let bytesPerSample = UInt32(bitsPerSample / 8)
        let dataSize = samples * UInt32(channels) * bytesPerSample
        let byteRate = sampleRate * UInt32(channels) * bytesPerSample
        let blockAlign = channels * (bitsPerSample / 8)
        let riffSize = 36 + dataSize

        data.append("RIFF".data(using: .ascii) ?? Data())
        data.append(le32(riffSize))
        data.append("WAVE".data(using: .ascii) ?? Data())
        data.append("fmt ".data(using: .ascii) ?? Data())
        data.append(le32(16))
        data.append(le16(1))
        data.append(le16(channels))
        data.append(le32(sampleRate))
        data.append(le32(byteRate))
        data.append(le16(blockAlign))
        data.append(le16(bitsPerSample))
        data.append("data".data(using: .ascii) ?? Data())
        data.append(le32(dataSize))
        data.append(Data(count: Int(dataSize)))
        return data
    }

    private func le16(_ value: UInt16) -> Data {
        withUnsafeBytes(of: value.littleEndian) { Data($0) }
    }

    private func le32(_ value: UInt32) -> Data {
        withUnsafeBytes(of: value.littleEndian) { Data($0) }
    }
}
