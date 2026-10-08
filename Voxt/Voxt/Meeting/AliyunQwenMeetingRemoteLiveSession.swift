import Foundation

@MainActor
final class AliyunQwenMeetingRemoteLiveSession: BaseMeetingRemoteLiveSession {
    private let endpoint: String
    private let token: String
    private let sessionKind: AliyunQwenRealtimeSessionKind

    init(
        speaker: MeetingSpeaker,
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload,
        timelineOffsetSeconds: TimeInterval,
        policy: MeetingLiveSessionPolicy
    ) {
        let configuredModel = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let model = configuredModel.isEmpty
            ? "qwen3-asr-flash-realtime"
            : configuredModel
        self.token = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        self.endpoint = RemoteASREndpointSupport.resolvedAliyunQwenRealtimeEndpoint(configuration.endpoint, model: model)
        self.sessionKind = RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: model) ?? .qwenASR
        super.init(
            speaker: speaker,
            configuration: configuration,
            hintPayload: hintPayload,
            speechThreshold: speaker == .me ? 0.015 : 0.025,
            timelineOffsetSeconds: timelineOffsetSeconds,
            policy: policy
        )
    }

    override func openTransport() async throws {
        guard !token.isEmpty else {
            throw NSError(domain: "Voxt.Meeting", code: -30, userInfo: [NSLocalizedDescriptionKey: "Aliyun Bailian API key is empty."])
        }
        guard let wsURL = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.Meeting", code: -31, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun Qwen realtime WebSocket endpoint URL."])
        }

        var request = URLRequest(url: wsURL)
        request.timeoutInterval = 45
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        VoxtLog.meeting(
            "Meeting Aliyun Qwen live connect. endpoint=\(endpoint), speaker=\(speaker.rawValue), proxyMode=\(VoxtNetworkSession.modeDescription)"
        )

        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        registerSocket(managedSocket)
        managedSocket.task.resume()
        startReceiveLoop()
        try await sendSessionUpdate(on: managedSocket.task)
        await primeTransportForAudio()
    }

    override func sendAudioPacket(_ pcmData: Data, isLast: Bool) async {
        guard !isLast, let ws = socketTask() else { return }
        let payload: [String: Any] = [
            "event_id": UUID().uuidString.lowercased(),
            "type": "input_audio_buffer.append",
            "audio": pcmData.base64EncodedString()
        ]
        do {
            let data = try JSONSerialization.data(withJSONObject: payload)
            guard let text = String(data: data, encoding: .utf8) else {
                throw NSError(domain: "Voxt.Meeting", code: -32, userInfo: [NSLocalizedDescriptionKey: "Failed to encode Aliyun Qwen audio event."])
            }
            try await ws.send(.string(text))
            if consumeShouldLogNextSpeechAudioPacket() {
                logFirstAudioPacketIfNeeded(kind: "aliyunQwen")
            }
        } catch {
            emitFailure(error)
            await cancel()
        }
    }

    override func sendFinishSignal() async {
        guard let ws = socketTask() else { return }
        let payload: [String: Any] = [
            "event_id": UUID().uuidString.lowercased(),
            "type": "session.finish"
        ]
        do {
            let data = try JSONSerialization.data(withJSONObject: payload)
            guard let text = String(data: data, encoding: .utf8) else {
                throw NSError(domain: "Voxt.Meeting", code: -33, userInfo: [NSLocalizedDescriptionKey: "Failed to encode Aliyun Qwen finish event."])
            }
            try await ws.send(.string(text))
        } catch {
            emitFailure(error)
        }
    }

    private func sendSessionUpdate(on ws: URLSessionWebSocketTask) async throws {
        let payload = AliyunQwenRealtimePayloadSupport.sessionUpdatePayload(
            kind: sessionKind,
            hintPayload: hintPayload,
            includesTurnDetection: false
        )
        let data = try JSONSerialization.data(withJSONObject: payload)
        guard let text = String(data: data, encoding: .utf8) else {
            throw NSError(domain: "Voxt.Meeting", code: -34, userInfo: [NSLocalizedDescriptionKey: "Failed to encode Aliyun Qwen session update."])
        }
        try await ws.send(.string(text))
    }

    private func startReceiveLoop() {
        guard let ws = socketTask() else { return }
        finishReceiveLoop(
            Task { @MainActor [weak self] in
                guard let self else { return }
                do {
                    while !Task.isCancelled {
                        let message = try await ws.receive()
                        let text: String?
                        switch message {
                        case .string(let string):
                            text = string
                        case .data(let data):
                            text = String(data: data, encoding: .utf8)
                        @unknown default:
                            text = nil
                        }
                        guard let text,
                              let data = text.data(using: .utf8),
                              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                        else {
                            continue
                        }
                        let type = (object["type"] as? String ?? "").lowercased()
                        if type == "session.updated" {
                            self.logServerPacketIfNeeded(kind: "aliyunQwen", parsed: nil)
                            await self.handleReadyForAudio()
                            continue
                        }
                        if type == "conversation.item.input_audio_transcription.text" {
                            let partial = (object["text"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                            if !partial.isEmpty {
                                self.logServerPacketIfNeeded(kind: "aliyunQwen", parsed: (text: partial, isFinal: false, sequence: nil))
                                if let unit = MeetingRemoteAudioSupport.makeAliyunQwenUnit(
                                    object: object,
                                    fallbackText: partial,
                                    isFinal: false
                                ) {
                                    self.emitProviderPacket(
                                        MeetingLiveProviderPacket(
                                            units: [unit],
                                            fallbackText: nil,
                                            isFinal: false,
                                            sequence: nil
                                        )
                                    )
                                } else {
                                    self.emitTranscript(text: partial, isFinal: false)
                                }
                            }
                            continue
                        }
                        if type == "conversation.item.input_audio_transcription.completed" {
                            let final = (object["transcript"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                            if !final.isEmpty {
                                self.logServerPacketIfNeeded(kind: "aliyunQwen", parsed: (text: final, isFinal: true, sequence: nil))
                                if let unit = MeetingRemoteAudioSupport.makeAliyunQwenUnit(
                                    object: object,
                                    fallbackText: final,
                                    isFinal: true
                                ) {
                                    self.emitProviderPacket(
                                        MeetingLiveProviderPacket(
                                            units: [unit],
                                            fallbackText: nil,
                                            isFinal: false,
                                            sequence: nil
                                        )
                                    )
                                } else {
                                    self.emitTranscript(text: final, isFinal: true)
                                }
                            }
                            continue
                        }
                        if type == "session.finished" {
                            self.signalFinished()
                            break
                        }
                        if type == "error" {
                            let detail = (object["message"] as? String) ?? "Aliyun Qwen realtime ASR task failed."
                            self.emitFailure(NSError(domain: "Voxt.Meeting", code: -35, userInfo: [NSLocalizedDescriptionKey: detail]))
                            break
                        }
                    }
                } catch {
                    if self.isStopping || self.isCancelled || self.shouldTreatAsBenignSocketClosure(error) {
                        self.signalFinished()
                    } else {
                        self.emitFailure(error)
                    }
                }
            }
        )
    }

    private func shouldTreatAsBenignSocketClosure(_ error: Error) -> Bool {
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain {
            return [NSURLErrorCancelled, NSURLErrorNetworkConnectionLost].contains(nsError.code)
        }
        return false
    }
}
