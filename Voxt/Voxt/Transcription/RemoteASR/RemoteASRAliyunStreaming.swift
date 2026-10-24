import Foundation
import AVFoundation

extension RemoteASRTranscriber {
    func stopAliyunFunStreaming(_ context: AliyunFunStreamingContext) {
        isRecording = false
        stopAliyunAudioCapture()
        guard !context.isClosed else { return }
        VoxtLog.model(
            "Aliyun fun stop requested. taskID=\(context.taskID), didStartAudioStream=\(context.didStartAudioStream), stopRequested=\(stopRequested)"
        )

        sendAliyunFunControl(action: "finish-task", through: context.ws, taskID: context.taskID) { error in
            Task { [responseState = context.responseState] in
                if let error {
                    await responseState.markCompletedWithError(error)
                } else {
                    await responseState.markFinishRequested()
                }
            }
        }
    }

    func stopAliyunQwenStreaming(_ context: AliyunQwenStreamingContext) {
        VoxtLog.model(
            "Aliyun qwen stop requested. kind=\(context.kind), didStartAudioStream=\(context.didStartAudioStream), stopRequested=\(stopRequested)"
        )
        Task { @MainActor [weak self] in
            guard let self else { return }
            guard self.isCurrentGeneration(context.generationID),
                  self.aliyunQwenStreamingContext === context,
                  !context.isClosed
            else { return }

            // Keep capture alive briefly so the last queued tap callbacks can append
            // trailing speech before we close the realtime session.
            try? await Task.sleep(for: self.aliyunRealtimeStopDrainDelay)

            guard self.isCurrentGeneration(context.generationID),
                  self.aliyunQwenStreamingContext === context,
                  !context.isClosed
            else { return }

            self.isRecording = false
            self.stopAliyunAudioCapture()
            self.sendAliyunQwenFinishEvent(context)
        }
    }

    private func sendAliyunQwenFinishEvent(_ context: AliyunQwenStreamingContext) {
        VoxtLog.model("Aliyun qwen sending session.finish. kind=\(context.kind)")
        let sendFinish: () -> Void = { [weak self] in
            guard let self else { return }
            self.sendAliyunQwenEvent(
                type: "session.finish",
                through: context.ws
            ) { error in
                Task { [responseState = context.responseState] in
                    if let error {
                        await responseState.markCompletedWithError(error)
                    } else {
                        await responseState.markFinishRequested()
                    }
                }
            }
        }

        if activeConfiguration?.aliyunASRSettings.useManualCommit == true,
           context.kind == .qwenASR {
            sendAliyunQwenEvent(
                type: "input_audio_buffer.commit",
                through: context.ws
            ) { error in
                if let error {
                    Task { [responseState = context.responseState] in
                        await responseState.markCompletedWithError(error)
                    }
                } else {
                    sendFinish()
                }
            }
        } else {
            sendFinish()
        }
    }


    func startAliyunFunStreaming(
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload
    ) throws {
        let token = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw NSError(domain: "Voxt.RemoteASR", code: -40, userInfo: [NSLocalizedDescriptionKey: "Aliyun Bailian API key is empty."])
        }

        let configuredModel = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let model = configuredModel.isEmpty
            ? RemoteASRProvider.aliyunBailianASR.suggestedModel
            : configuredModel
        let endpoint = RemoteASREndpointSupport.resolvedAliyunFunRealtimeEndpoint(configuration.endpoint)
        guard let wsURL = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -41, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun realtime WebSocket endpoint URL."])
        }

        var request = URLRequest(url: wsURL)
        request.timeoutInterval = 45
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        let ws = managedSocket.task
        ws.resume()

        let taskID = AliyunRemoteASRConfiguration.makeRealtimeTaskID()
        let responseState = AliyunFunResponseState { [weak self, generationID = self.recordingGenerationID] error in
            Task { @MainActor [weak self] in
                self?.notifyRuntimeFailure(error, generationID: generationID)
            }
        }
        let context = AliyunFunStreamingContext(
            session: managedSocket.session,
            ws: ws,
            taskID: taskID,
            responseState: responseState,
            generationID: recordingGenerationID
        )
        aliyunStreamingContext = context
        receiveAliyunFunMessages(context)
        VoxtLog.model(
            "Aliyun fun streaming socket ready. taskID=\(taskID), model=\(model), endpoint=\(endpoint), language=\(hintPayload.language ?? "auto"), languageHints=\(hintPayload.languageHints.joined(separator: ","))"
        )

        sendAliyunFunControl(
            action: "run-task",
            through: ws,
            taskID: taskID,
            model: model,
            parameters: AliyunFunRealtimePayloadSupport.parameters(
                model: model,
                hintPayload: hintPayload,
                settings: configuration.aliyunASRSettings
            ),
            context: AliyunFunRealtimePayloadSupport.context(
                model: model,
                phrases: hintPayload.contextualPhrases
            )
        ) { error in
            guard let error else { return }
            Task { [responseState] in
                await responseState.markCompletedWithError(error)
            }
        }
    }

    private func receiveAliyunFunMessages(_ context: AliyunFunStreamingContext) {
        context.ws.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    guard self.isCurrentGeneration(context.generationID),
                          self.aliyunStreamingContext === context
                    else { return }
                    do {
                        if case .string(let text) = message {
                            try await self.handleAliyunFunMessage(text, context: context)
                        } else if case .data(let data) = message,
                                  let text = String(data: data, encoding: .utf8) {
                            try await self.handleAliyunFunMessage(text, context: context)
                        }
                    } catch {
                        await context.responseState.markCompletedWithError(error)
                    }
                    if !context.isClosed {
                        self.receiveAliyunFunMessages(context)
                    }
                }
            case .failure(let error):
                Task {
                    guard await MainActor.run(body: { [weak self] in
                        guard let self else { return false }
                        return self.isCurrentGeneration(context.generationID) && self.aliyunStreamingContext === context
                    }) else { return }
                    await context.responseState.markCompletedWithError(error)
                }
            }
        }
    }

    private func handleAliyunFunMessage(_ text: String, context: AliyunFunStreamingContext) async throws {
        guard let data = text.data(using: .utf8),
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        let event = AliyunRemoteASRConfiguration.realtimeSocketEvent(from: object)
        let payload = object["payload"] as? [String: Any] ?? [:]
        VoxtLog.model(
            "Aliyun fun socket event received. event=\(event), didStartAudioStream=\(context.didStartAudioStream), stopRequested=\(stopRequested)"
        )

        if event == "task-failed" || event == "error" {
            let errorText = AliyunRemoteASRConfiguration.realtimeSocketErrorMessage(from: object)
                ?? "Aliyun fun ASR task failed."
            VoxtLog.model("Aliyun fun error event. event=\(event), detail=\(errorText)")
            throw NSError(domain: "Voxt.RemoteASR", code: -42, userInfo: [NSLocalizedDescriptionKey: errorText])
        }

        if event == "task-started", !context.didStartAudioStream {
            guard !stopRequested else {
                VoxtLog.asr("Aliyun fun task-started ignored because stop was already requested.", verbose: true)
                return
            }
            do {
                try startAliyunAudioCapture(context: context)
                context.didStartAudioStream = true
                VoxtLog.model("Aliyun fun task-started acknowledged. audio capture started.")
            } catch {
                throw error
            }
            return
        }

        if event == "result-generated" {
            let sentence = (payload["output"] as? [String: Any]).flatMap { output -> [String: Any]? in
                output["sentence"] as? [String: Any]
            } ?? [:]
            let partialText = (sentence["text"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let isSentenceEnd = sentence["sentence_end"] as? Bool ?? false
            if !partialText.isEmpty {
                VoxtLog.model(
                    "Aliyun fun result-generated. chars=\(partialText.count), sentenceEnd=\(isSentenceEnd)"
                )
                let merged = await context.responseState.updateWithSentence(partialText, isSentenceEnd: isSentenceEnd)
                publishIntermediateTranscription(merged)
            }
            return
        }

        if event == "task-finished" {
            context.isClosed = true
            VoxtLog.model("Aliyun fun task-finished received.")
            await context.responseState.markTaskFinished()
            return
        }
    }

    func startAliyunAudioCapture(context: AliyunFunStreamingContext) throws {
        let inputNode = acquireStreamingInputNode()
        let didApplyPreferredInputDevice = applyPreferredInputDeviceIfNeeded(inputNode: inputNode)
        let activeInputDeviceID = didApplyPreferredInputDevice ? preferredInputDeviceID : AudioInputDeviceManager.defaultInputDeviceID()
        let inputFormat = inputCaptureTapFormat(
            inputNode: inputNode,
            activeInputDeviceID: activeInputDeviceID,
            logContext: "Aliyun fun transcriber"
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
                      let ctx = self.aliyunStreamingContext,
                      !ctx.isClosed
                else { return }
                self.audioLevel = self.audioLevelFromPCM16(pcmData)
                ctx.ws.send(.data(pcmData)) { error in
                    if let error {
                        Task { [responseState = ctx.responseState] in
                            await responseState.markCompletedWithError(error)
                        }
                    }
                }
            }
        }

        audioEngine.prepare()
        try audioEngine.start()
        isRecording = true
        VoxtLog.model("Aliyun fun audio capture started. sampleRate=\(Int(streamingInputSampleRate))")
    }

    func stopAliyunAudioCapture() {
        stopStreamingAudioCapture()
    }

    func sendAliyunFunControl(
        action: String,
        through ws: URLSessionWebSocketTask,
        taskID: String,
        model: String? = nil,
        parameters: [String: Any]? = nil,
        context: [[String: Any]] = [],
        onError: @escaping (Error?) -> Void
    ) {
        let payload = AliyunRemoteASRConfiguration.funRealtimeControlPayload(
            action: action,
            taskID: taskID,
            model: model,
            parameters: parameters,
            context: context
        )
        do {
            let data = try JSONSerialization.data(withJSONObject: payload)
            guard let text = String(data: data, encoding: .utf8) else {
                onError(NSError(domain: "Voxt.RemoteASR", code: -43, userInfo: [NSLocalizedDescriptionKey: "Failed to encode Aliyun fun control message."]))
                return
            }
            ws.send(.string(text)) { error in
                onError(error)
            }
        } catch {
            onError(error)
        }
    }

    func startAliyunQwenRealtimeStreaming(
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload,
        kind: AliyunQwenRealtimeSessionKind
    ) throws {
        let token = configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw NSError(domain: "Voxt.RemoteASR", code: -44, userInfo: [NSLocalizedDescriptionKey: "Aliyun Bailian API key is empty."])
        }

        let configuredModel = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let model = configuredModel.isEmpty
            ? "qwen3-asr-flash-realtime"
            : configuredModel
        let endpoint = RemoteASREndpointSupport.resolvedAliyunQwenRealtimeEndpoint(configuration.endpoint, model: model)
        guard let wsURL = URL(string: endpoint) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -45, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun Qwen realtime WebSocket endpoint URL."])
        }

        var request = URLRequest(url: wsURL)
        request.timeoutInterval = 45
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        let ws = managedSocket.task
        ws.resume()

        let responseState = AliyunQwenResponseState { [weak self, generationID = self.recordingGenerationID] error in
            Task { @MainActor [weak self] in
                self?.notifyRuntimeFailure(error, generationID: generationID)
            }
        }
        let context = AliyunQwenStreamingContext(
            session: managedSocket.session,
            ws: ws,
            responseState: responseState,
            generationID: recordingGenerationID,
            kind: kind
        )
        aliyunQwenStreamingContext = context
        receiveAliyunQwenMessages(context)
        VoxtLog.model(
            "Aliyun qwen realtime socket ready. kind=\(kind), model=\(model), endpoint=\(endpoint), language=\(hintPayload.language ?? "auto"), languageHints=\(hintPayload.languageHints.joined(separator: ","))"
        )
        sendAliyunQwenSessionUpdate(through: ws, hintPayload: hintPayload, kind: kind) { error in
            Task { [responseState] in
                if let error {
                    await responseState.markCompletedWithError(error)
                }
            }
        }
    }

    private func receiveAliyunQwenMessages(_ context: AliyunQwenStreamingContext) {
        context.ws.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    guard self.isCurrentGeneration(context.generationID),
                          self.aliyunQwenStreamingContext === context
                    else { return }
                    do {
                        if case .string(let text) = message {
                            try await self.handleAliyunQwenMessage(text, context: context)
                        } else if case .data(let data) = message,
                                  let text = String(data: data, encoding: .utf8) {
                            try await self.handleAliyunQwenMessage(text, context: context)
                        }
                    } catch {
                        await context.responseState.markCompletedWithError(error)
                    }
                    if !context.isClosed {
                        self.receiveAliyunQwenMessages(context)
                    }
                }
            case .failure(let error):
                Task {
                    guard await MainActor.run(body: { [weak self] in
                        guard let self else { return false }
                        return self.isCurrentGeneration(context.generationID) && self.aliyunQwenStreamingContext === context
                    }) else { return }
                    await context.responseState.markCompletedWithError(error)
                }
            }
        }
    }

    private func handleAliyunQwenMessage(_ text: String, context: AliyunQwenStreamingContext) async throws {
        guard let data = text.data(using: .utf8),
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        let type = (object["type"] as? String ?? "").lowercased()
        VoxtLog.model(
            "Aliyun qwen socket event received. type=\(type), kind=\(context.kind), didStartAudioStream=\(context.didStartAudioStream), stopRequested=\(stopRequested)"
        )
        if type == "error" {
            let detail = (object["message"] as? String) ?? "Aliyun Qwen realtime ASR task failed."
            if await shouldIgnoreTrailingAliyunQwenGenericError(
                detail: detail,
                context: context
            ) {
                context.isClosed = true
                VoxtLog.model("Aliyun qwen trailing generic error ignored after stop. detail=\(detail)")
                await context.responseState.markSessionFinished()
                return
            }
            VoxtLog.model("Aliyun qwen error event. detail=\(detail)")
            VoxtLog.asr("Aliyun qwen realtime error packet received. detail=\(detail)", verbose: true)
            throw NSError(domain: "Voxt.RemoteASR", code: -46, userInfo: [NSLocalizedDescriptionKey: detail])
        }

        if type == "session.updated", !context.didStartAudioStream {
            guard !stopRequested else {
                VoxtLog.asr("Aliyun qwen session.updated ignored because stop was already requested.", verbose: true)
                return
            }
            try startAliyunQwenAudioCapture(context: context)
            context.didStartAudioStream = true
            VoxtLog.model("Aliyun qwen session.updated acknowledged. audio capture started. kind=\(context.kind)")
            return
        }

        if type.hasPrefix("response.")
            || type.hasPrefix("output_audio.")
            || (type.hasPrefix("conversation.item.") && !type.hasPrefix("conversation.item.input_audio_transcription.")) {
            return
        }

        if type == "conversation.item.input_audio_transcription.text" {
            let partial = (object["text"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if !partial.isEmpty {
                VoxtLog.model("Aliyun qwen partial text received. chars=\(partial.count)")
                let merged = await context.responseState.setPartial(partial)
                publishIntermediateTranscription(merged)
            }
            return
        }

        if type == "conversation.item.input_audio_transcription.completed" {
            let final = (object["transcript"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if !final.isEmpty {
                VoxtLog.model("Aliyun qwen transcript completed. chars=\(final.count)")
                let merged = await context.responseState.commit(final)
                publishIntermediateTranscription(merged)
            }
            return
        }

        if type == "session.finished" {
            context.isClosed = true
            VoxtLog.model("Aliyun qwen session.finished received. kind=\(context.kind)")
            await context.responseState.markSessionFinished()
            return
        }
    }

    func startAliyunQwenAudioCapture(context: AliyunQwenStreamingContext) throws {
        let inputNode = acquireStreamingInputNode()
        let didApplyPreferredInputDevice = applyPreferredInputDeviceIfNeeded(inputNode: inputNode)
        let activeInputDeviceID = didApplyPreferredInputDevice ? preferredInputDeviceID : AudioInputDeviceManager.defaultInputDeviceID()
        let inputFormat = inputCaptureTapFormat(
            inputNode: inputNode,
            activeInputDeviceID: activeInputDeviceID,
            logContext: "Aliyun qwen transcriber"
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
                      let ctx = self.aliyunQwenStreamingContext,
                      !ctx.isClosed
                else { return }
                self.audioLevel = self.audioLevelFromPCM16(pcmData)
                self.sendAliyunQwenAudioAppend(pcmData, through: ctx.ws) { error in
                    if let error {
                        Task { [responseState = ctx.responseState] in
                            await responseState.markCompletedWithError(error)
                        }
                    }
                }
            }
        }
        audioEngine.prepare()
        try audioEngine.start()
        isRecording = true
        VoxtLog.model("Aliyun qwen audio capture started. kind=\(context.kind), sampleRate=\(Int(streamingInputSampleRate))")
    }

    private func shouldIgnoreTrailingAliyunQwenGenericError(
        detail: String,
        context: AliyunQwenStreamingContext
    ) async -> Bool {
        guard stopRequested else { return false }
        let normalized = detail.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard normalized.isEmpty
                || normalized == "aliyun qwen realtime asr task failed."
                || normalized == "aliyun qwen realtime task failed."
                || normalized == "task failed"
        else { return false }
        let currentText = await context.responseState.currentText()
        return !currentText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func sendAliyunQwenSessionUpdate(
        through ws: URLSessionWebSocketTask,
        hintPayload: ResolvedASRHintPayload,
        kind: AliyunQwenRealtimeSessionKind = .qwenASR,
        onError: @escaping (Error?) -> Void
    ) {
        let payload = AliyunQwenRealtimePayloadSupport.sessionUpdatePayload(
            kind: kind,
            hintPayload: hintPayload,
            settings: activeConfiguration?.aliyunASRSettings ?? AliyunASRModelSettings()
        )
        sendAliyunQwenEvent(payload: payload, through: ws, onError: onError)
    }

    func sendAliyunQwenAudioAppend(
        _ audio: Data,
        through ws: URLSessionWebSocketTask,
        onError: @escaping (Error?) -> Void
    ) {
        let payload: [String: Any] = [
            "event_id": UUID().uuidString.lowercased(),
            "type": "input_audio_buffer.append",
            "audio": audio.base64EncodedString()
        ]
        sendAliyunQwenEvent(payload: payload, through: ws, onError: onError)
    }

    func sendAliyunQwenEvent(
        type: String,
        through ws: URLSessionWebSocketTask,
        onError: @escaping (Error?) -> Void
    ) {
        let payload: [String: Any] = [
            "event_id": UUID().uuidString.lowercased(),
            "type": type
        ]
        sendAliyunQwenEvent(payload: payload, through: ws, onError: onError)
    }

    func sendAliyunQwenEvent(
        payload: [String: Any],
        through ws: URLSessionWebSocketTask,
        onError: @escaping (Error?) -> Void
    ) {
        do {
            let data = try JSONSerialization.data(withJSONObject: payload)
            guard let text = String(data: data, encoding: .utf8) else {
                onError(NSError(domain: "Voxt.RemoteASR", code: -47, userInfo: [NSLocalizedDescriptionKey: "Failed to encode Aliyun Qwen realtime event."]))
                return
            }
            ws.send(.string(text)) { error in
                onError(error)
            }
        } catch {
            onError(error)
        }
    }
}
