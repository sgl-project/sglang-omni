// StepFunStreamingSupport.swift
// Provides Step Fun Streaming Support for remote ASR adapters.

import AVFoundation
import Foundation

@MainActor
extension RemoteASRTranscriber {
    func startStepFunStreaming(
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload
    ) throws {
        let token = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw NSError(domain: "Voxt.RemoteASR", code: -6, userInfo: [NSLocalizedDescriptionKey: "StepFun API key is empty."])
        }

        let configuredModel = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let model = configuredModel.isEmpty
            ? "step-asr-1.1-stream"
            : configuredModel
        let endpoint = RemoteASREndpointSupport.resolvedStepFunRealtimeEndpoint(configuration.endpoint)
        guard let wsURL = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -5, userInfo: [NSLocalizedDescriptionKey: "Invalid StepFun realtime endpoint URL."])
        }

        var request = URLRequest(url: wsURL)
        request.timeoutInterval = 45
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        VoxtLog.model("StepFun realtime stream connect. endpoint=\(endpoint), model=\(model)")

        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        let ws = managedSocket.task
        ws.resume()

        let context = StepFunStreamingContext(
            session: managedSocket.session,
            ws: ws,
            responseState: StepFunResponseState { [weak self, generationID = self.recordingGenerationID] error in
                Task { @MainActor [weak self] in
                    self?.notifyRuntimeFailure(error, generationID: generationID)
                }
            },
            generationID: recordingGenerationID
        )
        stepFunStreamingContext = context
        receiveStepFunMessages(context)
        try startStepFunAudioCapture(context: context)
        context.didStartAudioStream = true
        VoxtLog.model("StepFun realtime audio capture started while waiting for session.updated.")

        let payload = StepFunPayloadSupport.sessionUpdatePayload(
            model: model,
            hintPayload: hintPayload,
            useServerVAD: true
        )
        sendStepFunJSON(payload, through: ws) { error in
            if let error {
                Task { [responseState = context.responseState] in
                    await responseState.markCompletedWithError(error)
                }
            }
        }
    }

    func stopStepFunStreaming(_ context: StepFunStreamingContext) {
        VoxtLog.model("StepFun realtime stop requested. stopRequested=\(stopRequested)")
        Task { @MainActor [weak self] in
            guard let self else { return }
            guard self.isCurrentGeneration(context.generationID),
                  self.stepFunStreamingContext === context,
                  !context.isClosed
            else { return }

            try? await Task.sleep(for: self.aliyunRealtimeStopDrainDelay)

            guard self.isCurrentGeneration(context.generationID),
                  self.stepFunStreamingContext === context,
                  !context.isClosed
            else { return }

            self.isRecording = false
            self.stopStepFunAudioCapture()
            if context.isSessionUpdated {
                self.flushPendingStepFunAudio(context)
                self.sendStepFunCommit(context)
            } else {
                context.shouldCommitAfterSessionUpdate = true
                VoxtLog.model(
                    "StepFun realtime stop deferred until session.updated. bufferedBytes=\(context.pendingAudioByteCount)"
                )
            }
        }
    }

    private func receiveStepFunMessages(_ context: StepFunStreamingContext) {
        context.ws.receive { [weak self, weak context] result in
            Task { @MainActor [weak self, weak context] in
                guard let self, let context else { return }
                guard self.stepFunStreamingContext === context, !context.isClosed else { return }

                switch result {
                case .failure(let error):
                    if !context.isClosed {
                        await context.responseState.markCompletedWithError(error)
                    }
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
                        await self.handleStepFunRealtimeMessage(text, context: context)
                    }
                    self.receiveStepFunMessages(context)
                }
            }
        }
    }

    private func handleStepFunRealtimeMessage(_ text: String, context: StepFunStreamingContext) async {
        guard let data = text.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dict = object as? [String: Any] else {
            return
        }

        let type = (dict["type"] as? String) ?? ""
        switch type {
        case "session.updated":
            guard !context.isSessionUpdated else { return }
            context.isSessionUpdated = true
            if !context.didStartAudioStream, !stopRequested {
                do {
                    try startStepFunAudioCapture(context: context)
                    context.didStartAudioStream = true
                } catch {
                    await context.responseState.markCompletedWithError(error)
                    return
                }
            }
            flushPendingStepFunAudio(context)
            VoxtLog.model(
                "StepFun session.updated acknowledged. didStartAudioStream=\(context.didStartAudioStream), pendingCommit=\(context.shouldCommitAfterSessionUpdate)"
            )
            if context.shouldCommitAfterSessionUpdate {
                context.shouldCommitAfterSessionUpdate = false
                sendStepFunCommit(context)
            }
        case "conversation.item.input_audio_transcription.delta":
            let value = (dict["text"] as? String) ?? (dict["delta"] as? String) ?? ""
            guard !value.isEmpty else { return }
            let merged = await context.responseState.appendDelta(value, itemID: dict["item_id"] as? String)
            publishIntermediateTranscription(merged)
        case "conversation.item.input_audio_transcription.completed":
            let value = (dict["transcript"] as? String) ?? (dict["text"] as? String) ?? ""
            guard !value.isEmpty else { return }
            let merged = await context.responseState.commit(value, itemID: dict["item_id"] as? String)
            publishIntermediateTranscription(merged)
        case "error":
            let message = RemoteASRTextSupport.extractStreamErrorMessage(fromLine: text)
                ?? (dict["message"] as? String)
                ?? "StepFun realtime stream error."
            let error = NSError(
                domain: "Voxt.RemoteASR",
                code: -11,
                userInfo: [NSLocalizedDescriptionKey: "StepFun ASR realtime error: \(message)"]
            )
            await context.responseState.markCompletedWithError(error)
        default:
            break
        }
    }

    func startStepFunAudioCapture(context: StepFunStreamingContext) throws {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.reset()

        let inputNode = acquireStreamingInputNode()
        let didApplyPreferredInputDevice = preferredInputDeviceID != nil
            ? applyPreferredInputDeviceIfNeeded(inputNode: inputNode)
            : false
        let activeInputDeviceID = didApplyPreferredInputDevice ? preferredInputDeviceID : AudioInputDeviceManager.defaultInputDeviceID()
        let inputFormat = inputCaptureTapFormat(
            inputNode: inputNode,
            activeInputDeviceID: activeInputDeviceID,
            logContext: "StepFun transcriber"
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
                      let context = self.stepFunStreamingContext,
                      !context.isClosed
                else { return }
                self.audioLevel = self.audioLevelFromPCM16(pcmData)
                self.sendStepFunAudio(pcmData, context: context)
            }
        }

        audioEngine.prepare()
        try audioEngine.start()
        isRecording = true
        VoxtLog.asr(
            "StepFun realtime audio capture engine started. sampleRate=\(Int(inputFormat.sampleRate)), channels=\(inputFormat.channelCount)",
            verbose: true
        )
    }

    func stopStepFunAudioCapture() {
        stopStreamingAudioCapture()
    }

    private func sendStepFunAudio(_ pcmData: Data, context: StepFunStreamingContext) {
        guard !pcmData.isEmpty, !context.isClosed else { return }
        guard context.isSessionUpdated else {
            queuePendingStepFunAudio(pcmData, context: context)
            return
        }
        sendStepFunAudioChunk(pcmData, context: context)
    }

    private func queuePendingStepFunAudio(_ pcmData: Data, context: StepFunStreamingContext) {
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
                "StepFun startup audio buffer exceeded limit; dropped oldest chunks. droppedChunks=\(droppedChunks), droppedBytes=\(droppedBytes)"
            )
        }
    }

    private func flushPendingStepFunAudio(_ context: StepFunStreamingContext) {
        guard context.isSessionUpdated, !context.isClosed else { return }
        let chunks = context.pendingAudioChunks
        let byteCount = context.pendingAudioByteCount
        context.pendingAudioChunks.removeAll(keepingCapacity: false)
        context.pendingAudioByteCount = 0

        guard !chunks.isEmpty else { return }
        VoxtLog.model(
            "StepFun realtime flushing buffered startup audio. chunks=\(chunks.count), bytes=\(byteCount)"
        )
        for chunk in chunks {
            sendStepFunAudioChunk(chunk, context: context)
        }
    }

    private func sendStepFunAudioChunk(_ pcmData: Data, context: StepFunStreamingContext) {
        let payload: [String: Any] = [
            "event_id": UUID().uuidString.lowercased(),
            "type": "input_audio_buffer.append",
            "audio": pcmData.base64EncodedString()
        ]
        sendStepFunJSON(payload, through: context.ws) { error in
            if let error {
                Task { [responseState = context.responseState] in
                    await responseState.markCompletedWithError(error)
                }
            }
        }
    }

    private func sendStepFunCommit(_ context: StepFunStreamingContext) {
        let payload: [String: Any] = [
            "event_id": UUID().uuidString.lowercased(),
            "type": "input_audio_buffer.commit"
        ]
        sendStepFunJSON(payload, through: context.ws) { error in
            Task { [responseState = context.responseState] in
                if let error {
                    await responseState.markCompletedWithError(error)
                } else {
                    await responseState.markFinishRequested()
                }
            }
        }
    }

    private func sendStepFunJSON(
        _ payload: [String: Any],
        through ws: URLSessionWebSocketTask,
        onError: @escaping (Error?) -> Void
    ) {
        do {
            let data = try JSONSerialization.data(withJSONObject: payload)
            guard let text = String(data: data, encoding: .utf8) else {
                throw NSError(domain: "Voxt.RemoteASR", code: -12, userInfo: [NSLocalizedDescriptionKey: "Failed to encode StepFun realtime payload."])
            }
            ws.send(.string(text)) { error in
                onError(error)
            }
        } catch {
            onError(error)
        }
    }
}
