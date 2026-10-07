// GeminiLiveStreamingSupport.swift
// Provides Gemini live transcribe streaming for remote ASR adapters.

import AVFoundation
import Foundation

@MainActor
extension RemoteASRTranscriber {
    func startGeminiLiveStreaming(
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload
    ) throws {
        let apiKey = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !apiKey.isEmpty else {
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: -60,
                userInfo: [NSLocalizedDescriptionKey: "Google Gemini API key is empty."]
            )
        }

        let configuredModel = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let model = configuredModel.isEmpty ? GeminiLivePayloadSupport.defaultModel : configuredModel
        let loggedEndpoint = RemoteASREndpointSupport.resolvedGeminiLiveEndpoint(configuration.endpoint)
        guard let wsURL = RemoteASREndpointSupport.geminiLiveURL(
            endpoint: configuration.endpoint,
            apiKey: apiKey
        ) else {
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: -61,
                userInfo: [NSLocalizedDescriptionKey: "Invalid Gemini live transcribe WebSocket endpoint URL."]
            )
        }

        var request = URLRequest(url: wsURL)
        request.timeoutInterval = 45
        VoxtLog.model("Gemini live transcribe connect. endpoint=\(loggedEndpoint), model=\(model)")

        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        let ws = managedSocket.task
        ws.resume()

        let context = GeminiLiveStreamingContext(
            session: managedSocket.session,
            ws: ws,
            responseState: GeminiLiveResponseState { [weak self, generationID = self.recordingGenerationID] error in
                Task { @MainActor [weak self] in
                    self?.notifyRuntimeFailure(error, generationID: generationID)
                }
            },
            generationID: recordingGenerationID
        )
        geminiLiveStreamingContext = context
        receiveGeminiLiveMessages(context)
        try startGeminiLiveAudioCapture(context: context)
        context.didStartAudioStream = true

        let payload = GeminiLivePayloadSupport.setupPayload(model: model, hintPayload: hintPayload)
        VoxtLog.model(
            "Gemini live setup sent. languageCodes=\(GeminiLivePayloadSupport.languageCodes(from: hintPayload).joined(separator: ",")), vocabulary=\(GeminiLivePayloadSupport.customVocabulary(from: hintPayload).count)"
        )
        sendGeminiLiveJSON(payload, through: ws) { error in
            if let error {
                Task { [responseState = context.responseState] in
                    await responseState.markCompletedWithError(error)
                }
            }
        }
    }

    func stopGeminiLiveStreaming(_ context: GeminiLiveStreamingContext) {
        VoxtLog.model("Gemini live stop requested. stopRequested=\(stopRequested)")
        Task { @MainActor [weak self] in
            guard let self else { return }
            guard self.isCurrentGeneration(context.generationID),
                  self.geminiLiveStreamingContext === context,
                  !context.isClosed
            else { return }

            try? await Task.sleep(for: self.aliyunRealtimeStopDrainDelay)

            guard self.isCurrentGeneration(context.generationID),
                  self.geminiLiveStreamingContext === context,
                  !context.isClosed
            else { return }

            self.isRecording = false
            self.stopGeminiLiveAudioCapture()
            if context.isSetupComplete {
                self.flushPendingGeminiLiveAudio(context)
                self.sendGeminiLiveAudioStreamEnd(context)
            } else {
                context.shouldEndAudioStreamAfterSetup = true
                VoxtLog.model(
                    "Gemini live stop deferred until setupComplete. bufferedBytes=\(context.pendingAudioByteCount)"
                )
            }
        }
    }

    private func receiveGeminiLiveMessages(_ context: GeminiLiveStreamingContext) {
        context.ws.receive { [weak self, weak context] result in
            Task { @MainActor [weak self, weak context] in
                guard let self, let context else { return }
                guard self.geminiLiveStreamingContext === context, !context.isClosed else { return }

                switch result {
                case .failure(let error):
                    await self.handleGeminiLiveSocketFailure(error, context: context)
                    return
                case .success(let message):
                    let text: String?
                    switch message {
                    case .string(let value):
                        text = value
                    case .data(let data):
                        text = String(data: data, encoding: .utf8)
                    @unknown default:
                        text = nil
                    }
                    if let text {
                        await self.handleGeminiLiveMessage(text, context: context)
                    }
                    if !context.isClosed {
                        self.receiveGeminiLiveMessages(context)
                    }
                }
            }
        }
    }

    /// Google closes the socket once the stream ends, so a post-stop failure with
    /// text already in hand is a normal shutdown, not a runtime error.
    private func handleGeminiLiveSocketFailure(
        _ error: Error,
        context: GeminiLiveStreamingContext
    ) async {
        guard !context.isClosed else { return }
        if stopRequested {
            let current = await context.responseState.currentText()
            if !current.isEmpty {
                context.isClosed = true
                VoxtLog.model("Gemini live socket closed after stop. chars=\(current.count)")
                await context.responseState.markSessionFinished()
                return
            }
        }
        await context.responseState.markCompletedWithError(error)
    }

    private func handleGeminiLiveMessage(_ text: String, context: GeminiLiveStreamingContext) async {
        guard let data = text.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }

        if let message = GeminiLivePayloadSupport.errorMessage(from: object) {
            VoxtLog.asr("Gemini live error packet received. detail=\(message)", verbose: true)
            await context.responseState.markCompletedWithError(
                NSError(
                    domain: "Voxt.RemoteASR",
                    code: -62,
                    userInfo: [NSLocalizedDescriptionKey: "Gemini live transcribe error: \(message)"]
                )
            )
            return
        }

        if GeminiLivePayloadSupport.isSetupComplete(object) {
            guard !context.isSetupComplete else { return }
            context.isSetupComplete = true
            flushPendingGeminiLiveAudio(context)
            VoxtLog.model(
                "Gemini live setupComplete acknowledged. pendingEnd=\(context.shouldEndAudioStreamAfterSetup)"
            )
            if context.shouldEndAudioStreamAfterSetup {
                context.shouldEndAudioStreamAfterSetup = false
                sendGeminiLiveAudioStreamEnd(context)
            }
            return
        }

        let update = GeminiLivePayloadSupport.transcriptUpdate(from: object)
        if let final = update.final {
            let merged = await context.responseState.commit(final)
            publishIntermediateTranscription(merged)
        } else if let interim = update.interim {
            let merged = await context.responseState.setInterim(interim)
            publishIntermediateTranscription(merged)
        }

        // A turn ends on every pause, so only a terminal event after our own
        // audioStreamEnd means the session is really done.
        if context.didSendAudioStreamEnd, GeminiLivePayloadSupport.isSessionTerminal(object) {
            context.isClosed = true
            VoxtLog.model("Gemini live session terminal event received.")
            await context.responseState.markSessionFinished()
        }
    }

    func startGeminiLiveAudioCapture(context: GeminiLiveStreamingContext) throws {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.reset()

        let inputNode = acquireStreamingInputNode()
        let didApplyPreferredInputDevice = preferredInputDeviceID != nil
            ? applyPreferredInputDeviceIfNeeded(inputNode: inputNode)
            : false
        let activeInputDeviceID = didApplyPreferredInputDevice
            ? preferredInputDeviceID
            : AudioInputDeviceManager.defaultInputDeviceID()
        let inputFormat = inputCaptureTapFormat(
            inputNode: inputNode,
            activeInputDeviceID: activeInputDeviceID,
            logContext: "Gemini live transcriber"
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
                      let context = self.geminiLiveStreamingContext,
                      !context.isClosed
                else { return }
                self.audioLevel = self.audioLevelFromPCM16(pcmData)
                self.sendGeminiLiveAudio(pcmData, context: context)
            }
        }

        audioEngine.prepare()
        try audioEngine.start()
        isRecording = true
        VoxtLog.asr(
            "Gemini live audio capture started. sampleRate=\(Int(inputFormat.sampleRate)), channels=\(inputFormat.channelCount)",
            verbose: true
        )
    }

    func stopGeminiLiveAudioCapture() {
        stopStreamingAudioCapture()
    }

    private func sendGeminiLiveAudio(_ pcmData: Data, context: GeminiLiveStreamingContext) {
        guard !pcmData.isEmpty, !context.isClosed else { return }
        guard context.isSetupComplete else {
            queuePendingGeminiLiveAudio(pcmData, context: context)
            return
        }
        sendGeminiLiveAudioChunk(pcmData, context: context)
    }

    /// Speech starts before the setup handshake completes, so the first syllables
    /// are buffered instead of dropped.
    private func queuePendingGeminiLiveAudio(_ pcmData: Data, context: GeminiLiveStreamingContext) {
        context.pendingAudioChunks.append(pcmData)
        context.pendingAudioByteCount += pcmData.count

        var droppedBytes = 0
        var droppedChunks = 0
        while context.pendingAudioByteCount > realtimePendingAudioByteLimit,
              !context.pendingAudioChunks.isEmpty {
            let dropped = context.pendingAudioChunks.removeFirst()
            context.pendingAudioByteCount -= dropped.count
            droppedBytes += dropped.count
            droppedChunks += 1
        }

        if droppedChunks > 0 {
            VoxtLog.asrWarning(
                "Gemini live startup audio buffer exceeded limit; dropped oldest chunks. droppedChunks=\(droppedChunks), droppedBytes=\(droppedBytes)"
            )
        }
    }

    private func flushPendingGeminiLiveAudio(_ context: GeminiLiveStreamingContext) {
        guard context.isSetupComplete, !context.isClosed else { return }
        let chunks = context.pendingAudioChunks
        let byteCount = context.pendingAudioByteCount
        context.pendingAudioChunks.removeAll(keepingCapacity: false)
        context.pendingAudioByteCount = 0

        guard !chunks.isEmpty else { return }
        VoxtLog.model(
            "Gemini live flushing buffered startup audio. chunks=\(chunks.count), bytes=\(byteCount)"
        )
        for chunk in chunks {
            sendGeminiLiveAudioChunk(chunk, context: context)
        }
    }

    private func sendGeminiLiveAudioChunk(_ pcmData: Data, context: GeminiLiveStreamingContext) {
        sendGeminiLiveJSON(GeminiLivePayloadSupport.audioPayload(pcmData), through: context.ws) { error in
            if let error {
                Task { [responseState = context.responseState] in
                    await responseState.markCompletedWithError(error)
                }
            }
        }
    }

    private func sendGeminiLiveAudioStreamEnd(_ context: GeminiLiveStreamingContext) {
        context.didSendAudioStreamEnd = true
        sendGeminiLiveJSON(GeminiLivePayloadSupport.audioStreamEndPayload, through: context.ws) { error in
            Task { [responseState = context.responseState] in
                if let error {
                    await responseState.markCompletedWithError(error)
                } else {
                    await responseState.markFinishRequested()
                }
            }
        }
    }

    private func sendGeminiLiveJSON(
        _ payload: [String: Any],
        through ws: URLSessionWebSocketTask,
        onError: @escaping (Error?) -> Void
    ) {
        do {
            let data = try JSONSerialization.data(withJSONObject: payload)
            guard let text = String(data: data, encoding: .utf8) else {
                throw NSError(
                    domain: "Voxt.RemoteASR",
                    code: -63,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to encode Gemini live payload."]
                )
            }
            ws.send(.string(text)) { error in
                onError(error)
            }
        } catch {
            onError(error)
        }
    }
}
