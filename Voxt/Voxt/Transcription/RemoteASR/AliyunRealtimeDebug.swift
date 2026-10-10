// AliyunRealtimeDebug.swift
// Provides Aliyun Realtime Debug for remote ASR adapters.

import Foundation

extension RemoteASRTranscriber {
    func transcribeAliyunFunRealtimeFile(
        fileURL: URL,
        token: String,
        model: String,
        endpoint: String,
        hintPayload: ResolvedASRHintPayload,
        settings: AliyunASRModelSettings
    ) async throws -> String {
        guard let wsURL = URL(string: RemoteASREndpointSupport.resolvedAliyunFunRealtimeEndpoint(endpoint)) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -41, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun realtime WebSocket endpoint URL."])
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
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        let ws = managedSocket.task
        ws.resume()
        defer {
            ws.cancel(with: .goingAway, reason: nil)
            managedSocket.session.invalidateAndCancel()
        }

        let taskID = AliyunRemoteASRConfiguration.makeRealtimeTaskID()
        let responseState = AliyunFunResponseState()
        let startSignal = AsyncGate()
        let receiveTask = Task {
            do {
                while !Task.isCancelled {
                    let message = try await ws.receive()
                    let text: String
                    switch message {
                    case .string(let value):
                        text = value
                    case .data(let data):
                        guard let value = String(data: data, encoding: .utf8) else { continue }
                        text = value
                    @unknown default:
                        continue
                    }
                    try await self.handleAliyunFunDebugMessage(
                        text,
                        responseState: responseState,
                        startSignal: startSignal
                    )
                }
            } catch {
                await responseState.markCompletedWithError(error)
                await startSignal.fail(error)
            }
        }

        defer { receiveTask.cancel() }

        sendAliyunFunControl(
            action: "run-task",
            through: ws,
            taskID: taskID,
            model: model,
            parameters: AliyunFunRealtimePayloadSupport.parameters(
                model: model,
                hintPayload: hintPayload,
                settings: settings
            ),
            context: AliyunFunRealtimePayloadSupport.context(
                model: model,
                phrases: hintPayload.contextualPhrases
            )
        ) { error in
            Task {
                if let error {
                    await responseState.markCompletedWithError(error)
                    await startSignal.fail(error)
                }
            }
        }

        try await startSignal.wait()

        let chunkSize = 3200
        var offset = 0
        while offset < pcmData.count {
            let end = min(offset + chunkSize, pcmData.count)
            let chunk = Data(pcmData[offset..<end])
            try await ws.send(.data(chunk))
            offset = end
            try await Task.sleep(for: .milliseconds(24))
        }

        sendAliyunFunControl(action: "finish-task", through: ws, taskID: taskID) { error in
            Task {
                if let error {
                    await responseState.markCompletedWithError(error)
                } else {
                    await responseState.markFinishRequested()
                }
            }
        }

        let finalText = await resolveStreamingResult(
            warningMessage: "Aliyun fun file result wait failed"
        ) {
            try await responseState.waitForFinalResult(timeoutSeconds: 20)
        } fallback: {
            await responseState.currentText()
        }
        return finalText
    }

    func transcribeAliyunQwenRealtimeFile(
        fileURL: URL,
        token: String,
        model: String,
        endpoint: String,
        hintPayload: ResolvedASRHintPayload
    ) async throws -> String {
        let resolvedEndpoint = RemoteASREndpointSupport.resolvedAliyunQwenRealtimeEndpoint(endpoint, model: model)
        guard let wsURL = URL(string: resolvedEndpoint) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -45, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun Qwen realtime WebSocket endpoint URL."])
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
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let managedSocket = VoxtNetworkSession.makeWebSocketTask(with: request)
        let ws = managedSocket.task
        ws.resume()
        defer {
            ws.cancel(with: .goingAway, reason: nil)
            managedSocket.session.invalidateAndCancel()
        }

        let responseState = AliyunQwenResponseState()
        let startSignal = AsyncGate()
        let kind = RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: model) ?? .qwenASR
        let receiveTask = Task {
            do {
                while !Task.isCancelled {
                    let message = try await ws.receive()
                    let text: String
                    switch message {
                    case .string(let value):
                        text = value
                    case .data(let data):
                        guard let value = String(data: data, encoding: .utf8) else { continue }
                        text = value
                    @unknown default:
                        continue
                    }
                    try await self.handleAliyunQwenDebugMessage(
                        text,
                        responseState: responseState,
                        startSignal: startSignal
                    )
                }
            } catch {
                await responseState.markCompletedWithError(error)
                await startSignal.fail(error)
            }
        }

        defer { receiveTask.cancel() }

        sendAliyunQwenSessionUpdate(through: ws, hintPayload: hintPayload, kind: kind) { error in
            Task {
                if let error {
                    await responseState.markCompletedWithError(error)
                    await startSignal.fail(error)
                }
            }
        }

        try await startSignal.wait()

        let chunkSize = 3200
        var offset = 0
        while offset < pcmData.count {
            let end = min(offset + chunkSize, pcmData.count)
            let chunk = Data(pcmData[offset..<end])
            sendAliyunQwenAudioAppend(chunk, through: ws) { error in
                if let error {
                    Task { await responseState.markCompletedWithError(error) }
                }
            }
            offset = end
            try await Task.sleep(for: .milliseconds(24))
        }

        sendAliyunQwenEvent(type: "input_audio_buffer.commit", through: ws) { error in
            if let error {
                Task { await responseState.markCompletedWithError(error) }
            }
        }
        sendAliyunQwenEvent(type: "session.finish", through: ws) { error in
            Task {
                if let error {
                    await responseState.markCompletedWithError(error)
                } else {
                    await responseState.markFinishRequested()
                }
            }
        }

        let finalText = await resolveStreamingResult(
            warningMessage: "Aliyun qwen file result wait failed"
        ) {
            try await responseState.waitForFinalResult(timeoutSeconds: 20)
        } fallback: {
            await responseState.currentText()
        }
        return finalText
    }

    func handleAliyunFunDebugMessage(
        _ text: String,
        responseState: AliyunFunResponseState,
        startSignal: AsyncGate
    ) async throws {
        guard let data = text.data(using: .utf8),
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        let event = AliyunRemoteASRConfiguration.realtimeSocketEvent(from: object)
        let payload = object["payload"] as? [String: Any] ?? [:]

        if event == "task-failed" || event == "error" {
            let errorText = AliyunRemoteASRConfiguration.realtimeSocketErrorMessage(from: object)
                ?? "Aliyun fun ASR task failed."
            throw NSError(domain: "Voxt.RemoteASR", code: -42, userInfo: [NSLocalizedDescriptionKey: errorText])
        }

        if event == "task-started" {
            await startSignal.open()
            return
        }

        if event == "result-generated" {
            let sentence = (payload["output"] as? [String: Any]).flatMap { output -> [String: Any]? in
                output["sentence"] as? [String: Any]
            } ?? [:]
            let partialText = (sentence["text"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let isSentenceEnd = sentence["sentence_end"] as? Bool ?? false
            if !partialText.isEmpty {
                _ = await responseState.updateWithSentence(partialText, isSentenceEnd: isSentenceEnd)
            }
            return
        }

        if event == "task-finished" {
            await responseState.markTaskFinished()
            return
        }
    }

    func handleAliyunQwenDebugMessage(
        _ text: String,
        responseState: AliyunQwenResponseState,
        startSignal: AsyncGate
    ) async throws {
        guard let data = text.data(using: .utf8),
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        let type = (object["type"] as? String ?? "").lowercased()
        if type == "error" {
            let detail = (object["message"] as? String) ?? "Aliyun Qwen realtime ASR task failed."
            throw NSError(domain: "Voxt.RemoteASR", code: -46, userInfo: [NSLocalizedDescriptionKey: detail])
        }

        if type == "session.updated" {
            await startSignal.open()
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
                _ = await responseState.setPartial(partial)
            }
            return
        }

        if type == "conversation.item.input_audio_transcription.completed" {
            let final = (object["transcript"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if !final.isEmpty {
                _ = await responseState.commit(final)
            }
            return
        }

        if type == "session.finished" {
            await responseState.markSessionFinished()
            return
        }
    }
}
