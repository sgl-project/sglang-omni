import Foundation
import AVFoundation

extension RemoteASRTranscriber {
    func stopDoubaoStreaming(_ context: DoubaoStreamingContext) {
        isRecording = false
        stopDoubaoAudioCapture()
        flushBufferedDoubaoAudioIfNeeded(context: context, includeTrailingPartial: true)
        VoxtLog.asr("Doubao streaming stop requested. state=\(context.debugSummary())", verbose: true)

        let finalSequence = DoubaoASRConfiguration.finalStreamingSequence(
            nextAudioSequence: context.nextAudioSequence
        )
        VoxtLog.asr(
            "Doubao streaming final packet. lastSequence=\(context.lastAudioSequence), nextSequence=\(context.nextAudioSequence), finalSequence=\(finalSequence)",
            verbose: true
        )
        guard !context.isClosed else {
            VoxtLog.asr("Doubao streaming socket already closed before final packet, skip final send.", verbose: true)
            return
        }

        let finalPacket = DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeAudioOnlyClientRequest,
            messageFlags: DoubaoProtocol.flagNegativeAudioPacket,
            serialization: DoubaoProtocol.serializationNone,
            compression: DoubaoProtocol.compressionNone,
            sequence: finalSequence,
            payload: Data()
        )
        sendDoubaoPacket(finalPacket, through: context.ws) { error, isBenign in
            Task { [responseState = context.responseState] in
                if isBenign {
                    await responseState.markSocketClosed()
                } else {
                    await responseState.markCompletedWithError(error)
                }
            }
        }
    }


    func transcribeDoubaoStreamingFileWebSocket(
        fileURL: URL,
        appID: String,
        accessToken: String,
        resourceID: String,
        endpoint: String,
        hintPayload: ResolvedASRHintPayload,
        configuration: RemoteProviderConfiguration
    ) async throws -> String {
        guard let wsURL = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -5, userInfo: [NSLocalizedDescriptionKey: "Invalid Doubao endpoint URL."])
        }

        let (samples, sampleRate) = try DebugAudioClipIO.loadMonoSamples(from: fileURL)
        guard let pcmData = Self.makePCM16MonoData(from: samples, inputSampleRate: sampleRate),
              !pcmData.isEmpty else {
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: -52,
                userInfo: [NSLocalizedDescriptionKey: "Unable to decode audio samples."]
            )
        }

        var request = URLRequest(url: wsURL)
        request.timeoutInterval = 45
        request.setValue(appID, forHTTPHeaderField: "X-Api-App-Key")
        request.setValue(accessToken, forHTTPHeaderField: "X-Api-Access-Key")
        request.setValue(resourceID, forHTTPHeaderField: "X-Api-Resource-Id")
        let requestID = UUID().uuidString.lowercased()
        request.setValue(requestID, forHTTPHeaderField: "X-Api-Request-Id")
        request.setValue(requestID, forHTTPHeaderField: "X-Api-Connect-Id")
        VoxtLog.asr(
            "Doubao websocket connect. endpoint=\(endpoint), resource=\(resourceID)"
        )

        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        let ws = managedSocket.task
        ws.resume()
        defer {
            ws.cancel(with: .goingAway, reason: nil)
            managedSocket.session.invalidateAndCancel()
        }

        let reqID = UUID().uuidString.lowercased()
        try await sendDoubaoFullRequest(
            ws: ws,
            reqID: reqID,
            sequence: 1,
            hintPayload: hintPayload,
            audioFormat: DoubaoASRConfiguration.streamingAudioFormat,
            configuration: configuration
        )

        let responseState = DoubaoResponseState { [weak self, generationID = self.recordingGenerationID] error in
            Task { @MainActor [weak self] in
                self?.notifyRuntimeFailure(error, generationID: generationID)
            }
        }
        let receiveTask = Task {
            do {
                while !Task.isCancelled {
                    let message = try await ws.receive()
                    guard case .data(let payloadData) = message else { continue }
                    if let parsed = try Self.parseDoubaoServerPacket(payloadData) {
                        if let text = parsed.text, !text.isEmpty {
                            _ = await responseState.replace(text: text, isFinal: parsed.isFinal)
                        } else if parsed.isFinal {
                            await responseState.markFinal()
                        }
                    }
                }
            } catch {
                if let detail = await self.fetchDoubaoHandshakeFailureDetail(
                    error: error,
                    endpoint: endpoint,
                    resourceID: resourceID,
                    appID: appID,
                    accessToken: accessToken
                ) {
                    VoxtLog.asrWarning("Doubao websocket receive failed. detail=\(detail)")
                    let detailedError = NSError(
                        domain: "Voxt.RemoteASR",
                        code: (error as NSError).code,
                        userInfo: [NSLocalizedDescriptionKey: detail]
                    )
                    await responseState.markCompletedWithError(detailedError)
                } else {
                    await responseState.markCompletedWithError(error)
                }
            }
        }

        defer { receiveTask.cancel() }

        var offset = 0
        let chunkSize = DoubaoASRConfiguration.recommendedStreamingPacketBytes
        var sequence: Int32 = 2
        while offset < pcmData.count {
            let end = min(offset + chunkSize, pcmData.count)
            let chunk = pcmData[offset..<end]
            let isLast = end >= pcmData.count
            try await sendDoubaoAudioPacket(
                ws: ws,
                payload: Data(chunk),
                isLast: isLast,
                sequence: sequence
            )
            if !isLast {
                sequence += 1
            }
            offset = end
            try await Task.sleep(for: .milliseconds(24))
        }

        let finalText = await resolveStreamingResult(
            warningMessage: "Doubao async file result wait failed"
        ) {
            try await responseState.waitForFinalResult(timeoutSeconds: 20)
        } fallback: {
            await responseState.currentText()
        }
        return finalText
    }

    func startDoubaoStreaming(
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload
    ) throws {
        let accessToken = configuration.accessToken.trimmingCharacters(in: .whitespacesAndNewlines)
        let appID = configuration.appID.trimmingCharacters(in: .whitespacesAndNewlines)
        let resourceID = RemoteASREndpointSupport.resolvedDoubaoResourceID(from: configuration)

        guard !accessToken.isEmpty else {
            throw NSError(domain: "Voxt.RemoteASR", code: -3, userInfo: [NSLocalizedDescriptionKey: "Doubao Access Token is empty."])
        }
        guard !appID.isEmpty else {
            throw NSError(domain: "Voxt.RemoteASR", code: -4, userInfo: [NSLocalizedDescriptionKey: "Doubao App ID is empty."])
        }

        let endpoint = RemoteASREndpointSupport.resolvedDoubaoStreamingEndpoint(from: configuration)
        guard let wsURL = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -5, userInfo: [NSLocalizedDescriptionKey: "Invalid Doubao endpoint URL."])
        }

        var request = URLRequest(url: wsURL)
        request.timeoutInterval = 45
        request.setValue(appID, forHTTPHeaderField: "X-Api-App-Key")
        request.setValue(accessToken, forHTTPHeaderField: "X-Api-Access-Key")
        request.setValue(resourceID, forHTTPHeaderField: "X-Api-Resource-Id")
        let requestID = UUID().uuidString.lowercased()
        request.setValue(requestID, forHTTPHeaderField: "X-Api-Request-Id")
        request.setValue(requestID, forHTTPHeaderField: "X-Api-Connect-Id")
        VoxtLog.model(
            "Doubao stream connect. endpoint=\(endpoint), resource=\(resourceID)"
        )

        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        let ws = managedSocket.task
        ws.resume()
        let context = DoubaoStreamingContext(
            session: managedSocket.session,
            ws: ws,
            responseState: DoubaoResponseState { [weak self, generationID = self.recordingGenerationID] error in
                Task { @MainActor [weak self] in
                    self?.notifyRuntimeFailure(error, generationID: generationID)
                }
            },
            generationID: recordingGenerationID
        )
        doubaoStreamingContext = context
        receiveDoubaoMessages(context, endpoint: endpoint, resourceID: resourceID, appID: appID, accessToken: accessToken)

        let reqID = UUID().uuidString.lowercased()
        let streamingHintPayload = ResolvedASRHintPayload(
            language: nil,
            languageHints: hintPayload.languageHints,
            chineseOutputVariant: hintPayload.chineseOutputVariant,
            prompt: hintPayload.prompt
        )
        sendDoubaoFullRequest(
            ws: ws,
            reqID: reqID,
            sequence: 1,
            hintPayload: streamingHintPayload,
            audioFormat: DoubaoASRConfiguration.streamingAudioFormat,
            configuration: configuration,
            enableNonstream: true
        ) { error, isBenign in
            Task { [responseState = context.responseState] in
                if isBenign {
                    context.isClosed = true
                    await responseState.markSocketClosed()
                } else {
                    await responseState.markCompletedWithError(error)
                }
            }
        }
        try ensureDoubaoAudioCaptureStarted(context, reason: "request-sent")
    }

    private func sendDoubaoPacket(
        _ packet: Data,
        through ws: URLSessionWebSocketTask,
        onError: @escaping (Error, Bool) -> Void
    ) {
        ws.send(.data(packet)) { error in
            if let error {
                Task { @MainActor in
                    let nsError = error as NSError
                    let isBenign = self.isBenignDoubaoSocketError(nsError)
                    onError(error, isBenign)
                }
            }
        }
    }

    func startDoubaoAudioCapture(usePreferredInputDevice: Bool? = nil) throws {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.reset()

        let inputNode = acquireStreamingInputNode()
        let shouldUsePreferredInputDevice = usePreferredInputDevice ?? (preferredInputDeviceID != nil)
        doubaoCaptureUsesPreferredInputDevice = shouldUsePreferredInputDevice
        let didApplyPreferredInputDevice = shouldUsePreferredInputDevice
            ? applyPreferredInputDeviceIfNeeded(inputNode: inputNode)
            : false
        let activeInputDeviceID = didApplyPreferredInputDevice ? preferredInputDeviceID : AudioInputDeviceManager.defaultInputDeviceID()
        let inputFormat = inputCaptureTapFormat(
            inputNode: inputNode,
            activeInputDeviceID: activeInputDeviceID,
            logContext: "Doubao transcriber"
        )
        streamingInputSampleRate = inputFormat.sampleRate
        inputNode.removeTap(onBus: 0)
        let captureGeneration = recordingGenerationID
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }
            guard let pcmData = Self.makeDoubaoPCM16MonoData(from: buffer) else { return }
            if let samples = AudioLevelMeter.monoSamples(from: buffer), !samples.isEmpty {
                self.sampleStore.append(samples)
            }
            Task { @MainActor in
                guard self.isCurrentGeneration(captureGeneration), self.isRecording,
                      let context = self.doubaoStreamingContext,
                      !context.isClosed
                else { return }
                self.audioLevel = self.audioLevelFromPCM16(pcmData)
                self.queueDoubaoAudioData(pcmData, context: context)
            }
        }

        audioEngine.prepare()
        try audioEngine.start()
        isRecording = true
        VoxtLog.asr(
            "Doubao audio capture engine started. sampleRate=\(Int(inputFormat.sampleRate)), channels=\(inputFormat.channelCount), routing=\(shouldUsePreferredInputDevice ? "preferred" : "system-default"), deviceID=\(shouldUsePreferredInputDevice ? (preferredInputDeviceID.map(String.init(describing:)) ?? "default") : "system-default")",
            verbose: true
        )
    }

    func stopDoubaoAudioCapture() {
        doubaoCaptureStartupWatchdogTask?.cancel()
        doubaoCaptureStartupWatchdogTask = nil
        stopStreamingAudioCapture()
    }

    private func ensureDoubaoAudioCaptureStarted(
        _ context: DoubaoStreamingContext,
        reason: String
    ) throws {
        guard !context.didStartAudioStream else { return }
        guard !stopRequested else {
            VoxtLog.asr("Doubao audio capture start skipped because stop was already requested. reason=\(reason)", verbose: true)
            return
        }
        didRetryDoubaoCaptureStartup = false
        try startDoubaoAudioCapture(usePreferredInputDevice: preferredInputDeviceID != nil)
        context.didStartAudioStream = true
        context.audioCaptureStartCount += 1
        context.lastAudioCaptureStartReason = reason
        scheduleDoubaoCaptureStartupWatchdog(context)
        VoxtLog.asr("Doubao audio capture started. reason=\(reason), state=\(context.debugSummary())", verbose: true)
    }

    func scheduleDoubaoCaptureStartupWatchdog(_ context: DoubaoStreamingContext) {
        doubaoCaptureStartupWatchdogTask?.cancel()
        doubaoCaptureStartupWatchdogTask = Task { [weak self] in
            do {
                try await Task.sleep(for: self?.doubaoCaptureStartupWatchdogDelay ?? .seconds(1.2))
            } catch {
                return
            }
            await self?.recoverDoubaoCaptureIfNeeded(context)
        }
    }

    private func recoverDoubaoCaptureIfNeeded(_ context: DoubaoStreamingContext) async {
        guard doubaoStreamingContext === context else { return }
        guard isCurrentGeneration(context.generationID), isRecording, !context.isClosed else { return }
        guard context.pcmCallbackCount == 0 else { return }
        guard !didRetryDoubaoCaptureStartup else { return }

        didRetryDoubaoCaptureStartup = true
        let shouldFallbackToSystemDefault = preferredInputDeviceID != nil && doubaoCaptureUsesPreferredInputDevice
        if shouldFallbackToSystemDefault {
            VoxtLog.asrWarning(
                "Doubao audio capture produced no initial callbacks. Retrying once with system default input instead of the preferred device. state=\(context.debugSummary())"
            )
        } else {
            VoxtLog.asrWarning(
                "Doubao audio capture produced no initial callbacks. Restarting input graph once. state=\(context.debugSummary())"
            )
        }

        do {
            try startDoubaoAudioCapture(
                usePreferredInputDevice: shouldFallbackToSystemDefault ? false : doubaoCaptureUsesPreferredInputDevice
            )
            context.audioCaptureStartCount += 1
            context.lastAudioCaptureStartReason = "startup-watchdog"
            scheduleDoubaoCaptureStartupWatchdog(context)
        } catch {
            VoxtLog.asrError("Doubao audio capture recovery failed: \(error.localizedDescription)")
        }
    }


    private func receiveDoubaoMessages(
        _ context: DoubaoStreamingContext,
        endpoint: String,
        resourceID: String,
        appID: String,
        accessToken: String
    ) {
        context.ws.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    guard self.isCurrentGeneration(context.generationID),
                          self.doubaoStreamingContext === context
                    else { return }
                    do {
                        context.serverPacketCount += 1
                        let now = Date()
                        if context.firstServerPacketAt == nil {
                            context.firstServerPacketAt = now
                        }
                        context.lastServerPacketAt = now
                        if context.serverPacketCount == 1 {
                            VoxtLog.asr("Doubao first server packet received. state=\(context.debugSummary(now: now))", verbose: true)
                        }
                        if case .data(let payloadData) = message,
                           let parsed = try Self.parseDoubaoServerPacket(payloadData) {
                            if !context.didStartAudioStream {
                                do {
                                    try self.ensureDoubaoAudioCaptureStarted(context, reason: "server-packet")
                                } catch {
                                    await context.responseState.markCompletedWithError(error)
                                    self.cleanupDoubaoStreamingState()
                                    self.activeProvider = nil
                                    self.activeConfiguration = nil
                                    return
                                }
                            }
                            if let text = parsed.text, !text.isEmpty {
                                let merged = await context.responseState.replace(text: text, isFinal: parsed.isFinal)
                                await MainActor.run {
                                    self.publishIntermediateTranscription(merged)
                                }
                            } else if parsed.isFinal {
                                await context.responseState.markFinal()
                            }
                        }
                    } catch {
                        let nsError = error as NSError
                        if self.isBenignDoubaoSocketError(nsError) {
                            context.isClosed = true
                            await context.responseState.markSocketClosed()
                        } else {
                            context.isClosed = true
                            VoxtLog.asrWarning("Doubao stream receive failed. detail=\(error.localizedDescription), state=\(context.debugSummary())")
                            await context.responseState.markCompletedWithError(error)
                        }
                    }
                    if !context.isClosed {
                        self.receiveDoubaoMessages(
                            context,
                            endpoint: endpoint,
                            resourceID: resourceID,
                            appID: appID,
                            accessToken: accessToken
                        )
                    }
                }
            case .failure(let error):
                Task { @MainActor in
                    guard self.isCurrentGeneration(context.generationID),
                          self.doubaoStreamingContext === context
                    else { return }
                    let nsError = error as NSError
                    if self.isBenignDoubaoSocketError(nsError) {
                        context.isClosed = true
                        await context.responseState.markSocketClosed()
                        return
                    }

                    if let detail = await self.fetchDoubaoHandshakeFailureDetail(
                        error: error,
                        endpoint: endpoint,
                        resourceID: resourceID,
                        appID: appID,
                        accessToken: accessToken
                    ) {
                        context.isClosed = true
                        await MainActor.run {
                            VoxtLog.asrWarning("Doubao stream receive failed. detail=\(detail), state=\(context.debugSummary())")
                        }
                        let detailedError = NSError(
                            domain: "Voxt.RemoteASR",
                            code: nsError.code,
                            userInfo: [NSLocalizedDescriptionKey: detail]
                        )
                        await context.responseState.markCompletedWithError(detailedError)
                    } else {
                        context.isClosed = true
                        await context.responseState.markCompletedWithError(error)
                    }
                }
            }
        }
    }

    private func fetchDoubaoHandshakeFailureDetail(
        error: Error,
        endpoint: String,
        resourceID: String,
        appID: String,
        accessToken: String
    ) async -> String? {
        let nsError = error as NSError
        if nsError.domain != NSURLErrorDomain || nsError.code != NSURLErrorBadServerResponse {
            return nil
        }

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
            guard let http = response as? HTTPURLResponse else { return nil }
            logHTTPResponse(context: "Doubao handshake probe", response: http, data: data)
            let payload = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if payload.isEmpty {
                return "Doubao handshake failed (HTTP \(http.statusCode))."
            }
            return "Doubao handshake failed (HTTP \(http.statusCode)): \(payload)"
        } catch {
            return nil
        }
    }

    private func logHTTPResponse(context: String, response: HTTPURLResponse, data: Data) {
        let headers = response.allHeaderFields
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: ", ")
        let preview = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        VoxtLog.asr("[\(context)] status=\(response.statusCode), headers={\(headers)}, body=\(preview)", verbose: true)
    }

    private func isBenignDoubaoSocketError(_ error: NSError) -> Bool {
        if error.domain == NSPOSIXErrorDomain {
            return error.code == 57
        }

        if error.domain == NSURLErrorDomain {
            return [
                NSURLErrorCancelled,
                NSURLErrorNetworkConnectionLost,
                NSURLErrorCannotConnectToHost,
                NSURLErrorNotConnectedToInternet
            ].contains(error.code)
        }

        return false
    }

    private func sendDoubaoFullRequest(
        ws: URLSessionWebSocketTask,
        reqID: String,
        sequence: Int32,
        hintPayload: ResolvedASRHintPayload,
        audioFormat: String,
        configuration: RemoteProviderConfiguration
    ) async throws {
        let packet = try buildDoubaoFullRequestPacket(
            reqID: reqID,
            sequence: sequence,
            hintPayload: hintPayload,
            audioFormat: audioFormat,
            configuration: configuration
        )
        try await ws.send(.data(packet))
    }

    private func sendDoubaoFullRequest(
        ws: URLSessionWebSocketTask,
        reqID: String,
        sequence: Int32,
        hintPayload: ResolvedASRHintPayload,
        audioFormat: String,
        configuration: RemoteProviderConfiguration,
        enableNonstream: Bool = false,
        onError: @escaping (Error, Bool) -> Void
    ) {
        do {
            let packet = try buildDoubaoFullRequestPacket(
                reqID: reqID,
                sequence: sequence,
                hintPayload: hintPayload,
                audioFormat: audioFormat,
                configuration: configuration,
                enableNonstream: enableNonstream
            )
            sendDoubaoPacket(packet, through: ws, onError: onError)
        } catch {
            onError(error, false)
        }
    }

    private func buildDoubaoFullRequestPacket(
        reqID: String,
        sequence: Int32,
        hintPayload: ResolvedASRHintPayload,
        audioFormat: String,
        configuration: RemoteProviderConfiguration,
        enableNonstream: Bool = false
    ) throws -> Data {
        let payloadObject = doubaoRequestPayload(
            configuration: configuration,
            hintPayload: hintPayload,
            requestID: reqID,
            userID: "voxt",
            audioFormat: audioFormat,
            enableNonstream: enableNonstream
        )
        let rawPayload = try JSONSerialization.data(withJSONObject: payloadObject)
        let (payloadCompression, payload) = encodeDoubaoPacketPayload(rawPayload, preferGzip: true)
        return DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeFullClientRequest,
            messageFlags: DoubaoProtocol.flagPositiveSequence,
            serialization: DoubaoProtocol.serializationJSON,
            compression: payloadCompression,
            sequence: sequence,
            payload: payload
        )
    }

    private func queueDoubaoAudioData(_ pcmData: Data, context: DoubaoStreamingContext) {
        let now = Date()
        context.pcmCallbackCount += 1
        if context.firstPCMCallbackAt == nil {
            doubaoCaptureStartupWatchdogTask?.cancel()
            doubaoCaptureStartupWatchdogTask = nil
            context.firstPCMCallbackAt = now
            VoxtLog.asr("Doubao first PCM callback received. bytes=\(pcmData.count), state=\(context.debugSummary(now: now))", verbose: true)
        }
        context.lastPCMCallbackAt = now
        context.pendingPCMData.append(pcmData)
        flushBufferedDoubaoAudioIfNeeded(context: context, includeTrailingPartial: false)
    }

    private func flushBufferedDoubaoAudioIfNeeded(
        context: DoubaoStreamingContext,
        includeTrailingPartial: Bool
    ) {
        while let payload = DoubaoASRConfiguration.popRecommendedStreamingChunk(
            from: &context.pendingPCMData,
            includeTrailingPartial: includeTrailingPartial
        ) {
            sendBufferedDoubaoAudioPacket(payload, context: context)
        }
    }

    private func sendBufferedDoubaoAudioPacket(_ pcmData: Data, context: DoubaoStreamingContext) {
        guard !pcmData.isEmpty, !context.isClosed else { return }
        context.audioPacketCount += 1
        let now = Date()
        if context.firstAudioPacketSentAt == nil {
            context.firstAudioPacketSentAt = now
        }
        context.lastAudioPacketSentAt = now
        let sequence = context.nextAudioSequence
        context.nextAudioSequence += 1
        context.lastAudioSequence = sequence
        let (audioCompression, audioPayload) = encodeDoubaoPacketPayload(pcmData, preferGzip: true)
        let packet = DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeAudioOnlyClientRequest,
            messageFlags: DoubaoProtocol.flagPositiveSequence,
            serialization: DoubaoProtocol.serializationNone,
            compression: audioCompression,
            sequence: sequence,
            payload: audioPayload
        )
        if context.audioPacketCount == 1 {
            VoxtLog.asr("Doubao first audio packet sent. bytes=\(pcmData.count), sequence=\(sequence), state=\(context.debugSummary(now: now))", verbose: true)
        }
        sendDoubaoPacket(packet, through: context.ws) { error, isBenign in
            Task { [responseState = context.responseState] in
                if isBenign {
                    context.isClosed = true
                    await responseState.markSocketClosed()
                } else {
                    await responseState.markCompletedWithError(error)
                }
            }
        }
    }

    private func sendDoubaoAudioPacket(
        ws: URLSessionWebSocketTask,
        payload: Data,
        isLast: Bool,
        sequence: Int32
    ) async throws {
        let (audioCompression, compressedPayload) = encodeDoubaoPacketPayload(payload, preferGzip: true)
        let packet = DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeAudioOnlyClientRequest,
            messageFlags: isLast ? DoubaoProtocol.flagNegativeAudioPacket : DoubaoProtocol.flagPositiveSequence,
            serialization: DoubaoProtocol.serializationNone,
            compression: audioCompression,
            sequence: isLast ? -sequence : sequence,
            payload: compressedPayload
        )
        try await ws.send(.data(packet))
    }

    private func encodeDoubaoPacketPayload(
        _ payload: Data,
        preferGzip: Bool
    ) -> (compression: UInt8, payload: Data) {
        guard preferGzip, !payload.isEmpty else {
            return (DoubaoProtocol.compressionNone, payload)
        }

        do {
            return (DoubaoProtocol.compressionGzip, try DoubaoPacketCodec.encodePayload(payload).payload)
        } catch {
            VoxtLog.asrWarning("Doubao gzip compression failed. fallback to plain payload. error=\(error.localizedDescription)")
            return (DoubaoProtocol.compressionNone, payload)
        }
    }
}
