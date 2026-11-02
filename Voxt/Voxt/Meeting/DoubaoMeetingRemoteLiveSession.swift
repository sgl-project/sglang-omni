import Foundation

@MainActor
final class DoubaoMeetingRemoteLiveSession: BaseMeetingRemoteLiveSession {
    private let resourceID: String
    private let endpoint: String
    private let appID: String
    private let accessToken: String
    private var bufferedPCMData = Data()

    init(
        speaker: MeetingSpeaker,
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload,
        timelineOffsetSeconds: TimeInterval,
        policy: MeetingLiveSessionPolicy
    ) {
        self.resourceID = DoubaoASRConfiguration.resolvedResourceID(configuration.model)
        self.endpoint = DoubaoASRConfiguration.resolvedStreamingEndpoint(configuration.endpoint, model: configuration.model)
        self.appID = configuration.appID.trimmingCharacters(in: .whitespacesAndNewlines)
        self.accessToken = configuration.accessToken.trimmingCharacters(in: .whitespacesAndNewlines)
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
        guard !accessToken.isEmpty else {
            throw NSError(domain: "Voxt.Meeting", code: -10, userInfo: [NSLocalizedDescriptionKey: "Doubao Access Token is empty."])
        }
        guard !appID.isEmpty else {
            throw NSError(domain: "Voxt.Meeting", code: -11, userInfo: [NSLocalizedDescriptionKey: "Doubao App ID is empty."])
        }
        guard let wsURL = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.Meeting", code: -12, userInfo: [NSLocalizedDescriptionKey: "Invalid Doubao WebSocket endpoint URL."])
        }

        var request = URLRequest(url: wsURL)
        request.timeoutInterval = 45
        request.setValue(appID, forHTTPHeaderField: "X-Api-App-Key")
        request.setValue(accessToken, forHTTPHeaderField: "X-Api-Access-Key")
        request.setValue(resourceID, forHTTPHeaderField: "X-Api-Resource-Id")
        let requestID = UUID().uuidString.lowercased()
        request.setValue(requestID, forHTTPHeaderField: "X-Api-Request-Id")
        request.setValue(requestID, forHTTPHeaderField: "X-Api-Connect-Id")
        VoxtLog.meeting(
            "Meeting Doubao live connect. endpoint=\(endpoint), resource=\(resourceID), speaker=\(speaker.rawValue), proxyMode=\(VoxtNetworkSession.modeDescription)"
        )

        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        registerSocket(managedSocket)
        managedSocket.task.resume()
        startReceiveLoop()
        try await sendDoubaoFullRequest(on: managedSocket.task)
    }

    override func sendAudioPacket(_ pcmData: Data, isLast: Bool) async {
        do {
            guard !isLast else {
                try await flushBufferedAudioIfNeeded(includeTrailingPartial: true)
                try await transmitAudioPayload(Data(), isLast: true)
                return
            }

            bufferedPCMData.append(pcmData)
            try await flushBufferedAudioIfNeeded(includeTrailingPartial: false)
        } catch {
            emitFailure(error)
            await cancel()
        }
    }

    override func sendFinishSignal() async {
        await sendAudioPacket(Data(), isLast: true)
    }

    private func sendDoubaoFullRequest(on ws: URLSessionWebSocketTask) async throws {
        let reqID = UUID().uuidString.lowercased()
        let streamingHintPayload = ResolvedASRHintPayload(
            language: nil,
            languageHints: hintPayload.languageHints,
            chineseOutputVariant: hintPayload.chineseOutputVariant,
            prompt: hintPayload.prompt
        )
        let packet = try MeetingRemoteAudioSupport.buildDoubaoFullRequestPacket(
            reqID: reqID,
            sequence: 1,
            hintPayload: streamingHintPayload,
            audioFormat: DoubaoASRConfiguration.streamingAudioFormat,
            enableNonstream: true
        )
        try await ws.send(.data(packet))
        await handleReadyForAudio()
    }

    private func flushBufferedAudioIfNeeded(includeTrailingPartial: Bool) async throws {
        while let payload = DoubaoASRConfiguration.popRecommendedStreamingChunk(
            from: &bufferedPCMData,
            includeTrailingPartial: includeTrailingPartial
        ) {
            try await transmitAudioPayload(payload, isLast: false)
        }
    }

    private func transmitAudioPayload(_ pcmData: Data, isLast: Bool) async throws {
        guard let ws = socketTask() else { return }
        let (audioCompression, audioPayload) = try DoubaoPacketCodec.encodePayload(pcmData)
        let packet = DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeAudioOnlyClientRequest,
            messageFlags: isLast
                ? DoubaoProtocol.flagLastAudioPacket
                : DoubaoProtocol.flagNoSequence,
            serialization: DoubaoProtocol.serializationNone,
            compression: audioCompression,
            sequence: 0,
            payload: audioPayload
        )
        try await ws.send(.data(packet))
        if !isLast {
            logOutgoingAudioPacketIfNeeded(kind: "doubao", sequence: 0, payloadBytes: audioPayload.count)
            if consumeShouldLogNextSpeechAudioPacket() {
                logFirstAudioPacketIfNeeded(kind: "doubao")
            }
        }
    }

    private func startReceiveLoop() {
        guard let ws = socketTask() else { return }
        finishReceiveLoop(
            Task { @MainActor [weak self] in
                guard let self else { return }
                do {
                    while !Task.isCancelled {
                        let message = try await ws.receive()
                        guard case .data(let payloadData) = message else { continue }
                        if let packet = try MeetingRemoteAudioSupport.parseDoubaoServerPacket(payloadData) {
                            self.logServerPacketIfNeeded(
                                kind: "doubao",
                                parsed: (
                                    text: packet.units.last?.text ?? packet.fallbackText,
                                    isFinal: packet.isFinal,
                                    sequence: packet.sequence
                                )
                            )
                            await self.handleReadyForAudio()
                            self.emitProviderPacket(packet)
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
        if nsError.domain == NSPOSIXErrorDomain, nsError.code == 57 {
            return true
        }
        if nsError.domain == NSURLErrorDomain {
            return [
                NSURLErrorCancelled,
                NSURLErrorNetworkConnectionLost,
                NSURLErrorCannotConnectToHost,
                NSURLErrorNotConnectedToInternet
            ].contains(nsError.code)
        }
        return false
    }
}
