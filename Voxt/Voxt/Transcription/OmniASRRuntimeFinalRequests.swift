import Foundation

extension OmniASRRuntime {
    /// Qwen3-ASR Final as MLXAudio decoded it: ≤1200 s energy-cut chunks share one
    /// token budget, and the first chunk's detected language is forced on the rest.
    /// One Qwen3-ASR Final chunk, built the way Voxt's Swift model built it:
    /// its stop rules, and its prompt audio layout (one more mel frame and its
    /// own audio token count), so transcripts match the original backend.
    /// Like the original, an energy cut may land up to 5 s past the chunk end,
    /// unless that could exceed the server's 1200 s per-request limit.
    nonisolated static func qwenCutMayPassChunkEnd(chunkDurationSeconds: Float) -> Bool {
        chunkDurationSeconds + OmniTranscriptionPlanning.energyCutSearchSeconds <= qwenMaximumRequestSeconds
    }

    nonisolated static func qwenFinalRequest(
        samples: [Float],
        sampleRate: Int,
        language: String?,
        context: String?,
        maxNewTokens: Int
    ) -> OmniTranscriptionRequest {
        OmniTranscriptionRequest(
            samples: samples,
            sampleRate: sampleRate,
            language: language,
            prompt: context,
            maxNewTokens: maxNewTokens,
            stopAtEndOfText: true,
            stopOnTokenLoop: true,
            includeGenerationMetadata: true,
            audioLayout: "voxt_swift"
        )
    }

    /// Whisper Final as MLXAudio decoded it: the server cuts the recording into
    /// 30 s windows and decodes each with the same budget, so one request
    /// carries the whole recording.
    nonisolated static func whisperFinalRequest(
        samples: [Float],
        sampleRate: Int,
        language: String?,
        maxNewTokens: Int,
        temperature: Float
    ) -> OmniTranscriptionRequest {
        OmniTranscriptionRequest(
            samples: samples,
            sampleRate: sampleRate,
            language: language,
            prompt: nil,
            maxNewTokens: maxNewTokens,
            stopAtEndOfText: false,
            stopOnTokenLoop: false,
            temperature: temperature
        )
    }

    /// Cohere Transcribe Final as MLXAudio decoded it: energy-cut chunks, or
    /// speech segments for long audio, share one token budget on the server.
    nonisolated static func cohereFinalRequest(
        samples: [Float],
        sampleRate: Int,
        language: String?,
        usePunctuation: Bool?,
        maxNewTokens: Int,
        temperature: Float,
        chunkDuration: Float,
        minChunkDuration: Float,
        speechSegments: OmniSpeechSegments?
    ) -> OmniTranscriptionRequest {
        OmniTranscriptionRequest(
            samples: samples,
            sampleRate: sampleRate,
            language: language,
            prompt: nil,
            maxNewTokens: maxNewTokens,
            stopAtEndOfText: false,
            stopOnTokenLoop: false,
            temperature: temperature,
            usePunctuation: usePunctuation,
            chunkDuration: chunkDuration,
            minChunkDuration: minChunkDuration,
            speechSegments: speechSegments
        )
    }

    /// Note (Dayuxiaoshui): Parakeet Final as MLXAudio's generateStream decoded it: the
    /// server cuts the chunks from Voxt's chunk length and merges them.
    nonisolated static func parakeetFinalRequest(
        samples: [Float],
        sampleRate: Int,
        chunkDuration: Float
    ) -> OmniTranscriptionRequest {
        OmniTranscriptionRequest(
            samples: samples,
            sampleRate: sampleRate,
            language: nil,
            prompt: nil,
            maxNewTokens: nil,
            stopAtEndOfText: false,
            stopOnTokenLoop: false,
            chunkDuration: chunkDuration
        )
    }

    /// Note (Dayuxiaoshui): SenseVoice Final as MLXAudio decoded it: one pass, or for long
    /// audio the speech segments' chunks; the server normalizes the hint as Voxt does.
    nonisolated static func senseVoiceFinalRequest(
        samples: [Float],
        sampleRate: Int,
        language: String?,
        useITN: Bool,
        speechSegments: OmniSpeechSegments?
    ) -> OmniTranscriptionRequest {
        OmniTranscriptionRequest(
            samples: samples,
            sampleRate: sampleRate,
            language: language,
            prompt: nil,
            maxNewTokens: nil,
            stopAtEndOfText: false,
            stopOnTokenLoop: false,
            speechSegments: speechSegments,
            useITN: useITN
        )
    }

    func transcribeQwenFinal(
        samples: [Float],
        sampleRate: Int,
        language: String?,
        context: String?,
        maxTokens: Int,
        chunkDurationSeconds: Float = 1200,
        minChunkDurationSeconds: Float = 1
    ) async throws -> (text: String, language: String?) {
        let chunkDuration = min(chunkDurationSeconds, Self.qwenMaximumRequestSeconds)
        let chunks = OmniTranscriptionPlanning.energySplitChunks(
            samples,
            sampleRate: sampleRate,
            chunkDurationSeconds: chunkDuration,
            minChunkDurationSeconds: minChunkDurationSeconds,
            allowsCutPastChunkEnd: Self.qwenCutMayPassChunkEnd(chunkDurationSeconds: chunkDuration)
        )
        let endpoint = try beginUse()
        defer { endUse() }
        var remainingTokens = maxTokens
        var resolvedLanguage = language
        var text = ""
        for chunk in chunks {
            if remainingTokens <= 0 { break }
            try Task.checkCancellation()
            let result = try await transcribe(Self.qwenFinalRequest(
                samples: chunk.samples,
                sampleRate: sampleRate,
                language: resolvedLanguage,
                context: context,
                maxNewTokens: remainingTokens
            ), holding: endpoint)
            guard let metadata = result.generationMetadata else {
                throw OmniTranscriptionError.streamError("generation metadata missing")
            }
            remainingTokens -= metadata.generatedTokenCount
            if resolvedLanguage == nil {
                resolvedLanguage = metadata.language
            }
            text += result.text
        }
        return (text.trimmingCharacters(in: .whitespacesAndNewlines), resolvedLanguage)
    }
}

extension OmniASRRuntime {
    /// MOSS-Transcribe-Diarize Final as MLXAudio decoded it: one request for the
    /// whole recording, which the server cuts into 1200 s chunks, each with the
    /// token budget, decoded greedily with both end tokens and the loop guard.
    func transcribeMossFinal(
        samples: [Float],
        sampleRate: Int,
        prompt: String?,
        maxTokens: Int
    ) async throws -> OmniTranscriptionResult {
        try await transcribe(OmniTranscriptionRequest(
            samples: samples,
            sampleRate: sampleRate,
            language: nil,
            prompt: prompt,
            maxNewTokens: maxTokens,
            stopAtEndOfText: true,
            stopOnTokenLoop: true
        ))
    }
}

extension OmniASRRuntime {
    /// Nemotron 3.5 ASR Final as MLXAudio decoded it: the whole recording through
    /// the cache-aware stream at the native chunk, with the language hint as given.
    func transcribeNemotronFinal(
        samples: [Float],
        sampleRate: Int,
        language: String?
    ) async throws -> OmniTranscriptionResult {
        try await transcribe(OmniTranscriptionRequest(
            samples: samples,
            sampleRate: sampleRate,
            language: language,
            prompt: nil,
            maxNewTokens: nil,
            stopAtEndOfText: false,
            stopOnTokenLoop: false
        ))
    }

    /// Note (Dayuxiaoshui): Voxt maps a live language onto the checkpoint's
    /// prompt languages, which the Swift backend read from the loaded model;
    /// here they come from the same config.json the server loads.
    nonisolated static func nemotronPromptLanguages(
        modelDirectory: URL
    ) throws -> (languages: [String], defaultLanguage: String) {
        let data = try Data(contentsOf: modelDirectory.appendingPathComponent("config.json"))
        let config = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let prompt = config?["prompt"] as? [String: Any]
        let dictionary = prompt?["prompt_dictionary"] as? [String: Any] ?? [:]
        return (Array(dictionary.keys), config?["default_language"] as? String ?? "auto")
    }
}
