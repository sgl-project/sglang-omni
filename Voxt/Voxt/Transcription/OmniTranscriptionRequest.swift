import Foundation

/// Wire formats and request planning for the local Omni server.
nonisolated enum OmniASRModelKind: String, Sendable, CaseIterable {
    case qwen3ASR = "qwen3_asr"
}

nonisolated struct OmniTranscriptionRequest: Sendable, Equatable {
    var samples: [Float]
    var sampleRate: Int
    var language: String?
    var prompt: String?
    var maxNewTokens: Int?
    var stopAtEndOfText: Bool
    var stopOnTokenLoop: Bool
    var includeGenerationMetadata = false
    /// Prompt audio layout the server builds; nil keeps the reference layout.
    var audioLayout: String? = nil
}

nonisolated struct OmniGenerationMetadata: Sendable, Equatable {
    let generatedTokenCount: Int
    let language: String?
    let hitLengthLimit: Bool
}

nonisolated struct OmniTranscriptionResult: Sendable, Equatable {
    let text: String
    var generationMetadata: OmniGenerationMetadata? = nil
}

nonisolated enum OmniTranscriptionError: LocalizedError, Equatable {
    case httpStatus(Int, String)
    case streamError(String)
    case streamEndedWithoutDone
    case malformedEvent(String)

    var errorDescription: String? {
        switch self {
        // Server bodies and events can echo the request or the transcript,
        // which must never reach a log: only the status is described.
        case .httpStatus(let status, _):
            return "Local Omni server returned HTTP \(status)."
        case .streamError(let message):
            return "Local Omni transcription failed: \(message)"
        case .streamEndedWithoutDone:
            return "Local Omni transcription ended before its final result."
        case .malformedEvent:
            return "Local Omni transcription sent an unreadable event."
        }
    }
}

nonisolated enum OmniWAVEncoding {
    /// IEEE float32 mono WAV, so the server decodes the exact capture samples.
    static func float32WAV(samples: [Float], sampleRate: Int) -> Data {
        let bytesPerSample = 4
        let dataByteCount = samples.count * bytesPerSample
        var data = Data(capacity: 44 + dataByteCount)
        func append<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36 + dataByteCount))
        data.append(contentsOf: Array("WAVEfmt ".utf8))
        append(UInt32(16))
        append(UInt16(3))
        append(UInt16(1))
        append(UInt32(sampleRate))
        append(UInt32(sampleRate * bytesPerSample))
        append(UInt16(bytesPerSample))
        append(UInt16(32))
        data.append(contentsOf: Array("data".utf8))
        append(UInt32(dataByteCount))
        samples.withUnsafeBufferPointer { buffer in
            for sample in buffer {
                append(sample.bitPattern)
            }
        }
        return data
    }

    /// Little-endian PCM16 for the realtime socket, clamped like AVAudioConverter.
    static func pcm16(samples: ArraySlice<Float>) -> Data {
        var data = Data(capacity: samples.count * 2)
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let value = Int16((clamped * Float(Int16.max)).rounded())
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }
}

nonisolated enum OmniMultipartBody {
    static func transcription(
        _ request: OmniTranscriptionRequest,
        modelName: String,
        boundary: String
    ) -> Data {
        var fields: [(String, String)] = [
            ("model", modelName),
            ("stream", "true"),
            ("response_format", "json"),
        ]
        if let language = request.language {
            fields.append(("language", language))
        }
        if let prompt = request.prompt {
            fields.append(("prompt", prompt))
        }
        if let maxNewTokens = request.maxNewTokens {
            fields.append(("max_new_tokens", String(maxNewTokens)))
        }
        if request.stopAtEndOfText {
            fields.append(("stop_at_end_of_text", "true"))
        }
        if request.stopOnTokenLoop {
            fields.append(("stop_on_token_loop", "true"))
        }
        if request.includeGenerationMetadata {
            fields.append(("include_generation_metadata", "true"))
        }
        if let audioLayout = request.audioLayout {
            fields.append(("audio_layout", audioLayout))
        }
        var body = Data()
        for (name, value) in fields {
            body.append(Data("--\(boundary)\r\n".utf8))
            body.append(Data("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8))
            body.append(Data(value.utf8))
            body.append(Data("\r\n".utf8))
        }
        body.append(Data("--\(boundary)\r\n".utf8))
        body.append(Data("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n".utf8))
        body.append(Data("Content-Type: audio/wav\r\n\r\n".utf8))
        body.append(OmniWAVEncoding.float32WAV(samples: request.samples, sampleRate: request.sampleRate))
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        return body
    }
}

/// Server-sent transcription events. Only transcript.text.done is a success.
nonisolated struct OmniTranscriptionStreamParser {
    enum Outcome: Equatable {
        case pending
        case done(OmniTranscriptionResult)
    }

    private(set) var outcome: Outcome = .pending

    /// Returns the text delta the line carried, if any.
    @discardableResult
    mutating func consume(line: String) throws -> String? {
        guard line.hasPrefix("data:") else { return nil }
        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
        if payload == "[DONE]" {
            guard case .done = outcome else { throw OmniTranscriptionError.streamEndedWithoutDone }
            return nil
        }
        guard let data = payload.data(using: .utf8),
              let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = event["type"] as? String
        else {
            throw OmniTranscriptionError.malformedEvent(payload)
        }
        switch type {
        case "transcript.text.delta":
            return event["delta"] as? String
        case "transcript.text.done":
            var result = OmniTranscriptionResult(text: event["text"] as? String ?? "")
            if let metadata = event["generation_metadata"] as? [String: Any] {
                guard let count = metadata["generated_token_count"] as? Int,
                      let finishReason = metadata["finish_reason"] as? String
                else { throw OmniTranscriptionError.malformedEvent(payload) }
                result.generationMetadata = OmniGenerationMetadata(
                    generatedTokenCount: count,
                    language: metadata["language"] as? String,
                    hitLengthLimit: finishReason == "length"
                )
            }
            outcome = .done(result)
            return nil
        case "error":
            let error = event["error"] as? [String: Any]
            throw OmniTranscriptionError.streamError(error?["message"] as? String ?? payload)
        default:
            return nil
        }
    }

    func finish() throws -> OmniTranscriptionResult {
        guard case .done(let result) = outcome else { throw OmniTranscriptionError.streamEndedWithoutDone }
        return result
    }
}

/// Splits one recording the way the original Swift model decoded it.
nonisolated enum OmniTranscriptionPlanning {
    /// How far past a chunk's nominal end the energy cut may look (MLXAudio's 5 s).
    static let energyCutSearchSeconds: Float = 5.0

    struct PaddedChunk: Equatable {
        let samples: [Float]
        let offsetSeconds: Double
    }

    /// The MLXAudio splitter Qwen3-ASR used, on plain arrays: cut near
    /// the quietest 100 ms within ±5 s of each chunk end and zero-pad any chunk
    /// shorter than the minimum duration.
    static func energySplitChunks(
        _ samples: [Float],
        sampleRate: Int,
        chunkDurationSeconds: Float,
        minChunkDurationSeconds: Float,
        searchExpandSeconds: Float = energyCutSearchSeconds,
        minWindowMilliseconds: Float = 100.0,
        allowsCutPastChunkEnd: Bool = true
    ) -> [PaddedChunk] {
        let totalSamples = samples.count
        let minSamples = Int(minChunkDurationSeconds * Float(sampleRate))
        func padded(_ range: Range<Int>) -> [Float] {
            var chunk = Array(samples[range])
            if chunk.count < minSamples {
                chunk.append(contentsOf: repeatElement(0, count: minSamples - chunk.count))
            }
            return chunk
        }
        if Float(totalSamples) / Float(sampleRate) <= chunkDurationSeconds {
            return [PaddedChunk(samples: padded(0..<totalSamples), offsetSeconds: 0)]
        }
        let maxChunkSamples = Int(chunkDurationSeconds * Float(sampleRate))
        let searchSamples = Int(searchExpandSeconds * Float(sampleRate))
        let minWindowSamples = Int(minWindowMilliseconds * Float(sampleRate) / 1000.0)
        var chunks: [PaddedChunk] = []
        var startSample = 0
        while startSample < totalSamples {
            let endSample = min(startSample + maxChunkSamples, totalSamples)
            let offsetSeconds = Double(Float(startSample) / Float(sampleRate))
            if endSample >= totalSamples {
                chunks.append(PaddedChunk(samples: padded(startSample..<totalSamples), offsetSeconds: offsetSeconds))
                break
            }
            let searchStart = max(startSample, endSample - searchSamples)
            let searchEnd = min(totalSamples, endSample + (allowsCutPastChunkEnd ? searchSamples : 0))
            var cutSample = endSample
            if searchEnd - searchStart > minWindowSamples {
                let energyCount = searchEnd - searchStart - minWindowSamples + 1
                var windowSum: Float = 0
                for index in searchStart..<(searchStart + minWindowSamples) {
                    windowSum += samples[index] * samples[index]
                }
                let inverseWindow = 1.0 / Float(minWindowSamples)
                var minimumEnergy = windowSum * inverseWindow
                var minimumIndex = 0
                for offset in 1..<energyCount {
                    let leaving = samples[searchStart + offset - 1]
                    let entering = samples[searchStart + offset + minWindowSamples - 1]
                    windowSum += entering * entering - leaving * leaving
                    let energy = windowSum * inverseWindow
                    if energy < minimumEnergy {
                        minimumEnergy = energy
                        minimumIndex = offset
                    }
                }
                cutSample = searchStart + minimumIndex + minWindowSamples / 2
            }
            cutSample = max(cutSample, startSample + sampleRate)
            let actualEnd = min(cutSample, totalSamples)
            chunks.append(PaddedChunk(samples: padded(startSample..<actualEnd), offsetSeconds: offsetSeconds))
            startSample = cutSample
        }
        return chunks
    }
}

/// Joins transcript pieces the way the Omni server joins segments: a space only
/// between two characters of scripts that separate words with spaces.
nonisolated enum OmniTranscriptJoining {
    private static let unspacedScriptRanges: [ClosedRange<UInt32>] = [
        0x0E00...0x0EFF, 0x1000...0x109F, 0x1780...0x17FF, 0x2E80...0x303F,
        0x3040...0x30FF, 0x3400...0x9FFF, 0xF900...0xFAFF, 0xFF00...0xFFEF,
        0x20000...0x2FA1F,
    ]

    static func isSpacedScript(_ character: Character) -> Bool {
        guard let scalar = character.unicodeScalars.first, !character.isWhitespace else { return false }
        return !unspacedScriptRanges.contains { $0.contains(scalar.value) }
    }

    static func join(_ parts: [String]) -> String {
        var joined = ""
        for part in parts {
            let stripped = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !stripped.isEmpty else { continue }
            if let last = joined.last, let first = stripped.first,
               isSpacedScript(last), isSpacedScript(first) {
                joined += " "
            }
            joined += stripped
        }
        return joined
    }
}
