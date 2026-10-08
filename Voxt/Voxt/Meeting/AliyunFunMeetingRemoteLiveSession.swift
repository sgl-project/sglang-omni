import Foundation

@MainActor
final class AliyunFunMeetingRemoteLiveSession: BaseMeetingRemoteLiveSession {
    private let endpoint: String
    private let token: String
    private let model: String
    private let taskID = AliyunMeetingASRConfiguration.makeRealtimeTaskID()

    init(
        speaker: MeetingSpeaker,
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload,
        timelineOffsetSeconds: TimeInterval,
        policy: MeetingLiveSessionPolicy
    ) {
        self.token = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        let configuredModel = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        self.model = configuredModel.isEmpty
            ? RemoteASRProvider.aliyunBailianASR.suggestedModel
            : configuredModel
        self.endpoint = RemoteASREndpointSupport.resolvedAliyunFunRealtimeEndpoint(configuration.endpoint)
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
            throw NSError(domain: "Voxt.Meeting", code: -20, userInfo: [NSLocalizedDescriptionKey: "Aliyun Bailian API key is empty."])
        }
        guard let wsURL = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.Meeting", code: -21, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun realtime WebSocket endpoint URL."])
        }

        var request = URLRequest(url: wsURL)
        request.timeoutInterval = 45
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        VoxtLog.meeting(
            "Meeting Aliyun Fun live connect. endpoint=\(endpoint), model=\(model), speaker=\(speaker.rawValue), proxyMode=\(VoxtNetworkSession.modeDescription)"
        )

        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        registerSocket(managedSocket)
        managedSocket.task.resume()
        startReceiveLoop()
        try await sendRunTask(on: managedSocket.task)
        await primeTransportForAudio()
    }

    override func sendAudioPacket(_ pcmData: Data, isLast: Bool) async {
        guard !isLast, let ws = socketTask() else { return }
        do {
            try await ws.send(.data(pcmData))
            if consumeShouldLogNextSpeechAudioPacket() {
                logFirstAudioPacketIfNeeded(kind: "aliyunFun")
            }
        } catch {
            emitFailure(error)
            await cancel()
        }
    }

    override func sendFinishSignal() async {
        guard let ws = socketTask() else { return }
        do {
            try await sendControl(action: "finish-task", on: ws)
        } catch {
            emitFailure(error)
        }
    }

    private func sendRunTask(on ws: URLSessionWebSocketTask) async throws {
        var parameters: [String: Any] = [
            "sample_rate": 16000,
            "format": "pcm"
        ]
        if !hintPayload.languageHints.isEmpty {
            parameters["language_hints"] = hintPayload.languageHints
        }
        let payload = AliyunMeetingASRConfiguration.funRealtimeControlPayload(
            action: "run-task",
            taskID: taskID,
            model: model,
            parameters: parameters
        )
        let data = try JSONSerialization.data(withJSONObject: payload)
        guard let text = String(data: data, encoding: .utf8) else {
            throw NSError(domain: "Voxt.Meeting", code: -22, userInfo: [NSLocalizedDescriptionKey: "Failed to encode Aliyun run-task payload."])
        }
        try await ws.send(.string(text))
    }

    private func sendControl(action: String, on ws: URLSessionWebSocketTask) async throws {
        let payload = AliyunMeetingASRConfiguration.funRealtimeControlPayload(
            action: action,
            taskID: taskID
        )
        let data = try JSONSerialization.data(withJSONObject: payload)
        guard let text = String(data: data, encoding: .utf8) else {
            throw NSError(domain: "Voxt.Meeting", code: -23, userInfo: [NSLocalizedDescriptionKey: "Failed to encode Aliyun control payload."])
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
                        let event = AliyunMeetingASRConfiguration.realtimeSocketEvent(from: object)
                        let payload = object["payload"] as? [String: Any] ?? [:]

                        if event == "task-started" {
                            self.logServerPacketIfNeeded(kind: "aliyunFun", parsed: nil)
                            await self.handleReadyForAudio()
                            continue
                        }
                        if event == "task-finished" {
                            self.signalFinished()
                            break
                        }
                        if event == "task-failed" || event == "error" {
                            let detail = AliyunMeetingASRConfiguration.realtimeSocketErrorMessage(from: object)
                                ?? "Aliyun fun ASR task failed."
                            self.emitFailure(NSError(domain: "Voxt.Meeting", code: -24, userInfo: [NSLocalizedDescriptionKey: detail]))
                            break
                        }
                        if event == "result-generated" {
                            let sentence = (payload["output"] as? [String: Any]).flatMap { output in
                                output["sentence"] as? [String: Any]
                            } ?? [:]
                            let partialText = (sentence["text"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                            let isSentenceEnd = sentence["sentence_end"] as? Bool ?? false
                            if !partialText.isEmpty {
                                let unit = MeetingRemoteAudioSupport.makeAliyunSentenceUnit(
                                    sentence: sentence,
                                    fallbackText: partialText,
                                    isFinal: isSentenceEnd
                                )
                                self.logServerPacketIfNeeded(
                                    kind: "aliyunFun",
                                    parsed: (text: partialText, isFinal: isSentenceEnd, sequence: nil)
                                )
                                if let unit {
                                    self.emitProviderPacket(
                                        MeetingLiveProviderPacket(
                                            units: [unit],
                                            fallbackText: nil,
                                            isFinal: false,
                                            sequence: nil
                                        )
                                    )
                                } else {
                                    self.emitTranscript(text: partialText, isFinal: isSentenceEnd)
                                }
                            }
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
