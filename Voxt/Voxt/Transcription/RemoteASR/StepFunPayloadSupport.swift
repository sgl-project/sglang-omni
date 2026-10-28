import Foundation

enum StepFunSSEDataPayload: Equatable {
    case delta(String)
    case completed(String)
    case error(String)
    case fragment(String)
    case ignore
}

enum StepFunPayloadSupport {
    static func supportsSSEPrompt(model: String) -> Bool {
        let capabilities = StepFunASRModelCapabilities.forModel(model)
        return capabilities.supportsPrompt && !capabilities.usesRealtimeWebSocket
    }

    static func transcriptionPayload(
        model: String,
        hintPayload: ResolvedASRHintPayload,
        includeTimestamp: Bool = false,
        includePrompt: Bool = false,
        includeHotwords: Bool = true,
        fullRerunOnCommit: Bool? = nil
    ) -> [String: Any] {
        let capabilities = StepFunASRModelCapabilities.forModel(model)
        var payload: [String: Any] = [
            "model": model,
            "language": hintPayload.language ?? "zh",
            "enable_itn": true
        ]
        if includeTimestamp {
            payload["enable_timestamp"] = true
        }
        if includeHotwords,
           capabilities.supportsHotwords,
           !hintPayload.contextualPhrases.isEmpty {
            payload["hotwords"] = hintPayload.contextualPhrases
        }
        if includePrompt,
           capabilities.supportsPrompt,
           let prompt = hintPayload.prompt?.trimmingCharacters(in: .whitespacesAndNewlines),
           !prompt.isEmpty {
            payload["prompt"] = prompt
        }
        if let fullRerunOnCommit {
            payload["full_rerun_on_commit"] = fullRerunOnCommit
        }
        return payload
    }

    static func audioFormatPayload() -> [String: Any] {
        [
            "type": "pcm",
            "codec": "pcm_s16le",
            "rate": 16000,
            "bits": 16,
            "channel": 1
        ]
    }

    static func sessionUpdatePayload(
        model: String,
        hintPayload: ResolvedASRHintPayload,
        useServerVAD: Bool
    ) -> [String: Any] {
        var input: [String: Any] = [
            "format": audioFormatPayload(),
            "transcription": transcriptionPayload(
                model: model,
                hintPayload: hintPayload,
                includePrompt: true,
                includeHotwords: false,
                fullRerunOnCommit: true
            )
        ]
        if useServerVAD {
            input["turn_detection"] = [
                "type": "server_vad",
                "silence_duration_ms": 800,
                "threshold": 0.5
            ]
        }
        return [
            "event_id": UUID().uuidString.lowercased(),
            "type": "session.update",
            "session": [
                "audio": [
                    "input": input
                ]
            ]
        ]
    }

    static func parseSSEDataLine(_ line: String) -> StepFunSSEDataPayload {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .ignore }

        guard let data = trimmed.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) else {
            return .fragment(trimmed)
        }

        guard let dict = object as? [String: Any] else {
            if let text = RemoteASRTextSupport.extractText(in: object),
               let normalized = RemoteASRTextSupport.normalizedTextFragment(text) {
                return .fragment(normalized)
            }
            return .ignore
        }

        let type = (dict["type"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? ""
        switch type {
        case "transcript.text.delta":
            guard let value = dict["delta"],
                  let text = RemoteASRTextSupport.extractText(in: value),
                  let normalized = RemoteASRTextSupport.normalizedTextFragment(text) else {
                return .ignore
            }
            return .delta(normalized)
        case "transcript.text.done":
            guard let value = dict["text"],
                  let text = RemoteASRTextSupport.extractText(in: value),
                  let normalized = RemoteASRTextSupport.normalizedTextFragment(text) else {
                return .ignore
            }
            return .completed(normalized)
        default:
            if type == "error" || type.hasSuffix(".error") {
                return .error(RemoteASRTextSupport.extractStreamErrorMessage(fromLine: trimmed) ?? trimmed)
            }
        }

        if let text = RemoteASRTextSupport.extractTextFragment(fromLine: trimmed) {
            return .fragment(text)
        }
        return .ignore
    }
}

enum StepFunSupport {
    /// Extracts raw PCM data from a WAV file by walking RIFF chunks and
    /// returning the contents of the "data" chunk.
    static func extractPCMData(fromWAV wavData: Data) throws -> Data {
        guard wavData.count > 44 else {
            throw NSError(
                domain: "Voxt.StepFun",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "WAV file too small for StepFun ASR."]
            )
        }

        var offset = 12
        while offset + 8 <= wavData.count {
            let chunkID = String(data: wavData.subdata(in: offset..<offset + 4), encoding: .ascii) ?? ""
            let chunkSize = wavData.withUnsafeBytes { ptr in
                ptr.loadUnaligned(fromByteOffset: offset + 4, as: UInt32.self)
            }
            let size = Int(chunkSize)
            if chunkID == "data" {
                let dataStart = offset + 8
                let dataEnd = min(dataStart + size, wavData.count)
                guard dataEnd > dataStart else {
                    throw NSError(
                        domain: "Voxt.StepFun",
                        code: -2,
                        userInfo: [NSLocalizedDescriptionKey: "WAV data chunk is empty."]
                    )
                }
                return wavData.subdata(in: dataStart..<dataEnd)
            }
            offset += 8 + size
            if size % 2 != 0 { offset += 1 }
        }

        guard wavData.count > 44 else {
            throw NSError(
                domain: "Voxt.StepFun",
                code: -3,
                userInfo: [NSLocalizedDescriptionKey: "Cannot locate WAV data chunk."]
            )
        }
        return wavData.subdata(in: 44..<wavData.count)
    }
}
