import Foundation

extension RemoteProviderConnectivityTester {
    func testAliyunASRRealtimeWebSocketReachability(
        endpoint: String,
        apiKey: String,
        model: String
    ) async throws -> String {
        guard let url = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.Settings", code: -50, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid WebSocket endpoint URL.")])
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        RemoteProviderConnectivityTestLogging.logHTTPRequest(context: "Aliyun ASR realtime WebSocket test", request: request, bodyPreview: "run-task + finish-task")

        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        let ws = ConnectivityWebSocketSession(managedSocket: managedSocket)
        defer { ws.close() }

        let taskID = AliyunRemoteASRConfiguration.makeRealtimeTaskID()
        let runPayload = AliyunRemoteASRConfiguration.funRealtimeControlPayload(
            action: "run-task",
            taskID: taskID,
            model: model,
            parameters: AliyunFunRealtimePayloadSupport.parameters(
                model: model,
                hintPayload: ResolvedASRHintPayload(languageHints: ["zh", "en"])
            )
        )
        let finishPayload = AliyunRemoteASRConfiguration.funRealtimeControlPayload(
            action: "finish-task",
            taskID: taskID
        )
        let runData = try JSONSerialization.data(withJSONObject: runPayload)
        let finishData = try JSONSerialization.data(withJSONObject: finishPayload)
        guard let runText = String(data: runData, encoding: .utf8),
              let finishText = String(data: finishData, encoding: .utf8) else {
            throw NSError(domain: "Voxt.Settings", code: -51, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Failed to encode Aliyun WebSocket payload.")])
        }

        try await ws.send(.string(runText))
        try await ws.send(.string(finishText))

        for _ in 0..<6 {
            let message = try await ws.receive(timeoutSeconds: 3)
            guard case .string(let text) = message,
                  let data = text.data(using: .utf8),
                  let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                continue
            }
            let event = AliyunRemoteASRConfiguration.realtimeSocketEvent(from: object)
            if event == "task-started" || event == "task-finished" || event == "result-generated" {
                return AppLocalization.localizedString("Connection test succeeded (Aliyun ASR WebSocket reachable).")
            }
            if event == "task-failed" || event == "error" {
                let detail = AliyunRemoteASRConfiguration.realtimeSocketErrorMessage(from: object) ?? ""
                throw NSError(
                    domain: "Voxt.Settings",
                    code: 403,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Connection failed (HTTP %d). %@", 403, detail)]
                )
            }
        }

        throw NSError(
            domain: "Voxt.Settings",
            code: -52,
            userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Connection failed (HTTP %d). %@").replacingOccurrences(of: "%d", with: "0").replacingOccurrences(of: "%@", with: "No valid ASR response packet.")]
        )
    }

    func testAliyunASRQwenRealtimeWebSocketReachability(
        endpoint: String,
        apiKey: String,
        kind: AliyunQwenRealtimeSessionKind
    ) async throws -> String {
        guard let url = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.Settings", code: -53, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid WebSocket endpoint URL.")])
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("realtime=v1", forHTTPHeaderField: "OpenAI-Beta")
        RemoteProviderConnectivityTestLogging.logHTTPRequest(context: "Aliyun ASR Qwen realtime WebSocket test", request: request, bodyPreview: "session.update + session.finish")

        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        let ws = ConnectivityWebSocketSession(managedSocket: managedSocket)
        defer { ws.close() }

        let updatePayload = AliyunQwenRealtimePayloadSupport.sessionUpdatePayload(
            kind: kind,
            hintPayload: .init(language: "zh", languageHints: ["zh"])
        )
        let finishPayload: [String: Any] = [
            "event_id": UUID().uuidString.lowercased(),
            "type": "session.finish"
        ]
        let updateData = try JSONSerialization.data(withJSONObject: updatePayload)
        let finishData = try JSONSerialization.data(withJSONObject: finishPayload)
        guard let updateText = String(data: updateData, encoding: .utf8),
              let finishText = String(data: finishData, encoding: .utf8) else {
            throw NSError(domain: "Voxt.Settings", code: -54, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Failed to encode Aliyun Qwen realtime payload.")])
        }

        do {
            try await ws.send(.string(updateText))
        } catch {
            throw NSError(
                domain: "Voxt.Settings",
                code: -56,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Network connection failed before realtime handshake. %@ (Check proxy/VPN and endpoint reachability.)", error.localizedDescription)]
            )
        }

        do {
            for _ in 0..<6 {
                let message = try await ws.receive(timeoutSeconds: 3)
                guard case .string(let text) = message,
                      let data = text.data(using: .utf8),
                      let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    continue
                }
                let type = (object["type"] as? String ?? "").lowercased()
                if type == "session.created" || type == "session.updated" || type == "conversation.item.input_audio_transcription.text" {
                    try await ws.send(.string(finishText))
                    return AppLocalization.localizedString("Connection test succeeded (Aliyun Qwen realtime WebSocket reachable).")
                }
                if type == "error" {
                    let detail = (object["message"] as? String) ?? ""
                    throw NSError(
                        domain: "Voxt.Settings",
                        code: 403,
                        userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Connection failed (HTTP %d). %@", 403, detail)]
                    )
                }
            }
        } catch {
            if isWebSocketHandshakeFailure(error),
               let detailedError = await fetchAliyunQwenRealtimeHandshakeFailureDetail(
                endpoint: endpoint,
                apiKey: apiKey
               ) {
                throw detailedError
            }
            throw NSError(
                domain: "Voxt.Settings",
                code: -57,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Realtime WebSocket receive failed. %@ (Check proxy/VPN or region endpoint.)", error.localizedDescription)]
            )
        }

        throw NSError(
            domain: "Voxt.Settings",
            code: -55,
            userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Connection failed (HTTP %d). %@").replacingOccurrences(of: "%d", with: "0").replacingOccurrences(of: "%@", with: "No valid ASR response packet.")]
        )
    }

    func testGeminiLiveReachability(
        endpoint: String,
        apiKey: String,
        model: String
    ) async throws -> String {
        guard let url = RemoteASREndpointSupport.geminiLiveURL(endpoint: endpoint, apiKey: apiKey) else {
            throw NSError(domain: "Voxt.Settings", code: -58, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid WebSocket endpoint URL.")])
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        // The key rides in the query string, so only the sanitized endpoint is logged.
        RemoteProviderConnectivityTestLogging.logHTTPRequest(
            context: "Gemini live transcribe WebSocket test",
            request: URLRequest(url: URL(string: RemoteASREndpointSupport.resolvedGeminiLiveEndpoint(endpoint)) ?? url),
            bodyPreview: "setup"
        )

        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        let ws = ConnectivityWebSocketSession(managedSocket: managedSocket)
        defer { ws.close() }

        let setupPayload = GeminiLivePayloadSupport.setupPayload(
            model: model,
            hintPayload: ResolvedASRHintPayload(language: nil)
        )
        let setupData = try JSONSerialization.data(withJSONObject: setupPayload)
        guard let setupText = String(data: setupData, encoding: .utf8) else {
            throw NSError(domain: "Voxt.Settings", code: -59, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Failed to encode Gemini live payload.")])
        }

        do {
            try await ws.send(.string(setupText))
        } catch {
            throw NSError(
                domain: "Voxt.Settings",
                code: -60,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Network connection failed before realtime handshake. %@ (Check proxy/VPN and endpoint reachability.)", error.localizedDescription)]
            )
        }

        do {
            for _ in 0..<6 {
                let message = try await ws.receive(timeoutSeconds: 3)
                let text: String?
                switch message {
                case .string(let value):
                    text = value
                case .data(let data):
                    text = String(data: data, encoding: .utf8)
                @unknown default:
                    text = nil
                }
                guard let text,
                      let data = text.data(using: .utf8),
                      let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                else { continue }

                if let detail = GeminiLivePayloadSupport.errorMessage(from: object) {
                    throw NSError(
                        domain: "Voxt.Settings",
                        code: 403,
                        userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Connection failed (HTTP %d). %@", 403, detail)]
                    )
                }
                if GeminiLivePayloadSupport.isSetupComplete(object) {
                    return AppLocalization.localizedString("Connection test succeeded (Gemini live transcribe reachable).")
                }
            }
        } catch {
            throw NSError(
                domain: "Voxt.Settings",
                code: -61,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Realtime WebSocket receive failed. %@ (Check proxy/VPN or region endpoint.)", error.localizedDescription)]
            )
        }

        throw NSError(
            domain: "Voxt.Settings",
            code: -62,
            userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Connection failed (HTTP %d). %@").replacingOccurrences(of: "%d", with: "0").replacingOccurrences(of: "%@", with: "No valid ASR response packet.")]
        )
    }
}
