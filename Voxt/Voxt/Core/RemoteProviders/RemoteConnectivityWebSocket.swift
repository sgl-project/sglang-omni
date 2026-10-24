import Foundation

extension RemoteProviderConnectivityTester {
    func testDoubaoStreamingReachability(
        endpoint: String,
        appID: String,
        accessToken: String,
        model: String,
        successMessage: String = ""
    ) async throws -> String {
        guard let url = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.Settings", code: -12, userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Invalid WebSocket endpoint URL.")])
        }

        let resourceID = DoubaoConnectivityTestSupport.normalizedResourceID(model)
        var request = URLRequest(url: url)
        request.timeoutInterval = 12
        request.setValue(appID, forHTTPHeaderField: "X-Api-App-Key")
        request.setValue(accessToken, forHTTPHeaderField: "X-Api-Access-Key")
        request.setValue(resourceID, forHTTPHeaderField: "X-Api-Resource-Id")
        let requestID = UUID().uuidString.lowercased()
        request.setValue(requestID, forHTTPHeaderField: "X-Api-Request-Id")
        request.setValue(requestID, forHTTPHeaderField: "X-Api-Connect-Id")
        RemoteProviderConnectivityTestLogging.logHTTPRequest(
            context: "Doubao streaming test",
            request: request,
            bodyPreview: "full-request(audio=\(DoubaoASRConfiguration.requestAudioFormat),gzip) + silent wav bytes(gzip)"
        )

        do {
            let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
            let ws = ConnectivityWebSocketSession(managedSocket: managedSocket)
            defer { ws.close() }

            let reqID = UUID().uuidString.lowercased()
            let payloadObject = DoubaoASRConfiguration.fullRequestPayload(
                requestID: reqID,
                userID: "voxt-test",
                language: "zh-CN",
                chineseOutputVariant: nil
            )
            let initPayload = try JSONSerialization.data(withJSONObject: payloadObject)
            let (initCompression, initPacketPayload) = DoubaoConnectivityTestSupport.encodePacketPayload(initPayload, preferGzip: true)
            try await ws.send(.data(DoubaoConnectivityTestSupport.buildPacket(
                messageType: 0x1,
                messageFlags: 0x1,
                serialization: 0x1,
                compression: initCompression,
                sequence: 1,
                payload: initPacketPayload
            )))

            let (audioCompression, audioPayload) = DoubaoConnectivityTestSupport.encodePacketPayload(silentTestWavData(), preferGzip: true)
            try await ws.send(.data(DoubaoConnectivityTestSupport.buildPacket(
                messageType: 0x2,
                messageFlags: 0x3,
                serialization: 0x0,
                compression: audioCompression,
                sequence: -2,
                payload: audioPayload
            )))

            for index in 1...4 {
                let message = try await ws.receive(timeoutSeconds: 3)
                guard case .data(let packetData) = message else { continue }
                let parsed = try DoubaoConnectivityTestSupport.parseServerPacket(packetData)
                VoxtLog.network(
                    "Doubao test server packet. index=\(index), type=\(parsed.messageType), bytes=\(packetData.count), hasText=\(parsed.hasText), isFinal=\(parsed.isFinal)",
                    verbose: true
                )

                if let errorText = parsed.errorText, !errorText.isEmpty {
                    throw NSError(domain: "Voxt.Settings", code: 403, userInfo: [NSLocalizedDescriptionKey: errorText])
                }
                if parsed.hasText || parsed.isFinal || parsed.messageType == 0xB || parsed.messageType == 0x9 {
                    if !successMessage.isEmpty {
                        return successMessage
                    }
                    return AppLocalization.localizedString("Connection test succeeded (Doubao WebSocket reachable).")
                }
            }

            throw NSError(
                domain: "Voxt.Settings",
                code: -120,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Connection failed (HTTP %d). %@").replacingOccurrences(of: "%d", with: "0").replacingOccurrences(of: "%@", with: "No valid ASR response packet.")]
            )
        } catch {
            if isWebSocketHandshakeFailure(error),
               let detailedError = await fetchDoubaoHandshakeFailureDetail(
                    endpoint: endpoint,
                    appID: appID,
                    accessToken: accessToken,
                    resourceID: resourceID
               ) {
                throw detailedError
            }
            throw error
        }
    }

    func isWebSocketHandshakeFailure(_ error: Error) -> Bool {
        let nsError = error as NSError
        return nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorBadServerResponse
    }

    private func fetchDoubaoHandshakeFailureDetail(
        endpoint: String,
        appID: String,
        accessToken: String,
        resourceID: String
    ) async -> NSError? {
        guard var components = URLComponents(string: endpoint) else {
            return nil
        }
        if components.scheme == "wss" {
            components.scheme = "https"
        } else if components.scheme == "ws" {
            components.scheme = "http"
        }
        guard let probeURL = components.url else {
            return nil
        }

        var request = URLRequest(url: probeURL)
        request.httpMethod = "GET"
        request.timeoutInterval = 10
        request.setValue("websocket", forHTTPHeaderField: "Upgrade")
        request.setValue("Upgrade", forHTTPHeaderField: "Connection")
        request.setValue(appID, forHTTPHeaderField: "X-Api-App-Key")
        request.setValue(accessToken, forHTTPHeaderField: "X-Api-Access-Key")
        request.setValue(resourceID, forHTTPHeaderField: "X-Api-Resource-Id")
        let requestID = UUID().uuidString.lowercased()
        request.setValue(requestID, forHTTPHeaderField: "X-Api-Request-Id")
        request.setValue(requestID, forHTTPHeaderField: "X-Api-Connect-Id")

        do {
            let (data, response) = try await VoxtNetworkSession.active.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return nil
            }
            RemoteProviderConnectivityTestLogging.logHTTPResponse(context: "Doubao handshake probe", response: http, data: data)
            let payload = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if payload.isEmpty {
                return NSError(
                    domain: "Voxt.Settings",
                    code: http.statusCode,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Doubao handshake failed (HTTP %d).", http.statusCode)]
                )
            }
            return NSError(
                domain: "Voxt.Settings",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Doubao handshake failed (HTTP %d): %@", http.statusCode, payload)]
            )
        } catch {
            return nil
        }
    }

    func fetchAliyunQwenRealtimeHandshakeFailureDetail(
        endpoint: String,
        apiKey: String
    ) async -> NSError? {
        guard var components = URLComponents(string: endpoint) else {
            return nil
        }
        if components.scheme == "wss" {
            components.scheme = "https"
        } else if components.scheme == "ws" {
            components.scheme = "http"
        }
        guard let probeURL = components.url else {
            return nil
        }

        var request = URLRequest(url: probeURL)
        request.httpMethod = "GET"
        request.timeoutInterval = 10
        request.setValue("websocket", forHTTPHeaderField: "Upgrade")
        request.setValue("Upgrade", forHTTPHeaderField: "Connection")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("realtime=v1", forHTTPHeaderField: "OpenAI-Beta")

        do {
            let (data, response) = try await VoxtNetworkSession.active.data(for: request)
            guard let http = response as? HTTPURLResponse else { return nil }
            RemoteProviderConnectivityTestLogging.logHTTPResponse(context: "Aliyun Qwen realtime handshake probe", response: http, data: data)
            let payload = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if payload.isEmpty {
                return NSError(
                    domain: "Voxt.Settings",
                    code: http.statusCode,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Aliyun Qwen realtime handshake failed (HTTP %d).", http.statusCode)]
                )
            }
            return NSError(
                domain: "Voxt.Settings",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.format("Aliyun Qwen realtime handshake failed (HTTP %d). %@", http.statusCode, payload)]
            )
        } catch {
            return nil
        }
    }
}
