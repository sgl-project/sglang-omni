// Final requests consume resolved snapshots; model leases and session tasks remain in MLXTranscriber.

import Foundation

struct MLXDetachedInferenceResult {
    let rawText: String
    let senseVoiceMetadata: SenseVoiceTranscriptMetadata?
    let structuredSegments: [MLXStructuredTranscriptSegment]
}

enum MLXStructuredTranscriptionError: LocalizedError {
    case senseVoiceLongFormNoSpeechSegments(Double)

    var errorDescription: String? {
        switch self {
        case .senseVoiceLongFormNoSpeechSegments:
            return AppLocalization.localizedString("SenseVoice could not detect any speech segments in this long audio clip.")
        }
    }

    var diagnosticDescription: String {
        switch self {
        case .senseVoiceLongFormNoSpeechSegments(let durationSeconds):
            return "SenseVoice long-form VAD produced no speech segments. durationSec=\(String(format: "%.2f", durationSeconds))"
        }
    }
}

extension MLXTranscriber {
    struct ResolvedInferenceConfiguration: Sendable {
        let family: MLXModelFamily
        let generationParameters: ASRGenerationOptions
        let languageHint: String?
        let timingGranularity: MLXASRTimingGranularity
        let qwenContextBias: String
        let senseVoiceUseITN: Bool
        let cohereLongFormStrategy: CohereLongFormStrategy
        let mossPrompt: String?
        let mossOutputMode: MossASROutputMode
    }

    nonisolated static func prepareInputSamplesDetached(
        _ samples: [Float],
        sampleRate: Double,
        targetSampleRate: Int
    ) throws -> [Float] {
        if abs(sampleRate - Double(targetSampleRate)) > 1.0 {
            return try ASRAudioSampleRateConverter.resample(
                samples: samples,
                from: Int(sampleRate),
                to: targetSampleRate
            )
        }
        return samples
    }

    static func runFinalInference(
        model loadedModel: LoadedASRModel,
        audioSamples: [Float],
        inferenceConfiguration: ResolvedInferenceConfiguration,
        targetSampleRate: Int,
        inferenceTaskPriority: TaskPriority
    ) async throws -> MLXDetachedInferenceResult {
        try Task.checkCancellation()
        let runtime = try loadedModel.requireOmniRuntime()
        let speechSegments = try await omniSpeechSegmentsIfNeeded(
            runtime: runtime,
            audioSamples: audioSamples,
            inferenceConfiguration: inferenceConfiguration,
            targetSampleRate: targetSampleRate
        )
        let inferenceTask = Task.detached(priority: inferenceTaskPriority) {
            try await Self.runOmniInferenceDetached(
                runtime: runtime,
                audioSamples: audioSamples,
                inferenceConfiguration: inferenceConfiguration,
                targetSampleRate: targetSampleRate,
                speechSegments: speechSegments
            )
        }
        return try await withTaskCancellationHandler {
            try await inferenceTask.value
        } onCancel: {
            inferenceTask.cancel()
        }
    }

    /// The Native server loads the provisioned Silero VAD for long-form speech cuts.
    private static func omniSpeechSegmentsIfNeeded(
        runtime: OmniASRRuntime,
        audioSamples: [Float],
        inferenceConfiguration: ResolvedInferenceConfiguration,
        targetSampleRate: Int
    ) async throws -> OmniSpeechSegments? {
        let senseVoiceDirectPassMaximumDurationSeconds: Double = 30.0
        let senseVoiceChunkMaximumDurationSeconds: Double = 24.0
        let senseVoiceChunkOverlapSeconds: Double = 0.35
        let senseVoiceVADThreshold: Float = 0.5
        let senseVoiceVADMinSpeechDurationMs = 220
        let senseVoiceVADMinSilenceDurationMs = 420
        let senseVoiceVADSpeechPadMs = 180
        let usesVoiceActivity: Bool
        switch runtime.kind {
        case .cohereTranscribe:
            usesVoiceActivity = inferenceConfiguration.cohereLongFormStrategy == .voiceActivity
        case .senseVoice:
            usesVoiceActivity = true
        default:
            usesVoiceActivity = false
        }
        guard usesVoiceActivity,
              MLXTranscriptionPlanning.shouldUseSenseVoiceVAD(
                  sampleCount: audioSamples.count,
                  sampleRate: targetSampleRate,
                  directPassMaximumDurationSeconds: senseVoiceDirectPassMaximumDurationSeconds
              )
        else {
            return nil
        }
        let modelDirectory = try await SileroVADModelProvisioner.shared.ensureModelDirectory()
        try Task.checkCancellation()
        return OmniSpeechSegments(
            vadModelDirectory: modelDirectory,
            threshold: senseVoiceVADThreshold,
            minSpeechMilliseconds: senseVoiceVADMinSpeechDurationMs,
            minSilenceMilliseconds: senseVoiceVADMinSilenceDurationMs,
            speechPadMilliseconds: senseVoiceVADSpeechPadMs,
            // Note (Dayuxiaoshui): SenseVoice splits each speech run into overlapping chunks
            // instead of merging runs across short gaps.
            mergeGapSeconds: runtime.kind == .senseVoice ? nil : 1.0,
            maxChunkSeconds: Float(senseVoiceChunkMaximumDurationSeconds),
            chunkOverlapSeconds: runtime.kind == .senseVoice ? Float(senseVoiceChunkOverlapSeconds) : nil
        )
    }

    /// Native decoding retains each family's chunking, budget and stop rules.
    nonisolated static func runOmniInferenceDetached(
        runtime: OmniASRRuntime,
        audioSamples: [Float],
        inferenceConfiguration: ResolvedInferenceConfiguration,
        targetSampleRate: Int,
        speechSegments: OmniSpeechSegments?
    ) async throws -> MLXDetachedInferenceResult {
        try Task.checkCancellation()
        let options = inferenceConfiguration.generationParameters
        let text: String
        switch runtime.kind {
        case .qwen3ASR:
            text = try await runtime.transcribeQwenFinal(
                samples: audioSamples,
                sampleRate: targetSampleRate,
                language: inferenceConfiguration.languageHint,
                context: inferenceConfiguration.qwenContextBias,
                maxTokens: options.maximumTokenCount,
                chunkDurationSeconds: options.chunkDurationSeconds,
                minChunkDurationSeconds: options.minimumChunkDurationSeconds
            ).text
        case .whisper:
            text = try await runtime.transcribe(OmniASRRuntime.whisperFinalRequest(
                samples: audioSamples,
                sampleRate: targetSampleRate,
                language: inferenceConfiguration.languageHint,
                maxNewTokens: options.maximumTokenCount,
                temperature: options.temperature
            )).text
        case .cohereTranscribe:
            text = try await runtime.transcribe(OmniASRRuntime.cohereFinalRequest(
                samples: audioSamples,
                sampleRate: targetSampleRate,
                language: inferenceConfiguration.languageHint,
                usePunctuation: options.usePunctuation,
                maxNewTokens: options.maximumTokenCount,
                temperature: options.temperature,
                chunkDuration: options.chunkDurationSeconds,
                minChunkDuration: options.minimumChunkDurationSeconds,
                speechSegments: speechSegments
            )).text
        case .mossTranscribeDiarize:
            let result = try await runtime.transcribeMossFinal(
                samples: audioSamples,
                sampleRate: targetSampleRate,
                prompt: inferenceConfiguration.mossPrompt,
                maxTokens: options.maximumTokenCount
            )
            return MLXDetachedInferenceResult(
                rawText: MossASRTranscriptRendering.renderedText(
                    result.text,
                    outputMode: inferenceConfiguration.mossOutputMode
                ),
                senseVoiceMetadata: nil,
                structuredSegments: mossStructuredSegments(from: result.segments.map(\.transcriptSegment))
            )
        case .nemotronASR:
            let result = try await runtime.transcribeNemotronFinal(
                samples: audioSamples,
                sampleRate: targetSampleRate,
                language: inferenceConfiguration.languageHint
            )
            return MLXDetachedInferenceResult(
                rawText: result.text,
                senseVoiceMetadata: nil,
                structuredSegments: reliableStructuredSegments(
                    from: result.segments.map(\.transcriptSegment),
                    timingGranularity: inferenceConfiguration.timingGranularity
                )
            )
        case .parakeet:
            let result = try await runtime.transcribe(OmniASRRuntime.parakeetFinalRequest(
                samples: audioSamples,
                sampleRate: targetSampleRate,
                chunkDuration: options.chunkDurationSeconds
            ))
            return MLXDetachedInferenceResult(
                rawText: result.text,
                senseVoiceMetadata: nil,
                structuredSegments: reliableStructuredSegments(
                    from: result.segments.map(\.transcriptSegment),
                    timingGranularity: inferenceConfiguration.timingGranularity
                )
            )
        case .senseVoice:
            let result = try await runtime.transcribe(OmniASRRuntime.senseVoiceFinalRequest(
                samples: audioSamples,
                sampleRate: targetSampleRate,
                language: inferenceConfiguration.languageHint,
                useITN: inferenceConfiguration.senseVoiceUseITN,
                speechSegments: speechSegments
            ))
            return try senseVoiceOmniResult(
                result,
                durationSeconds: Double(audioSamples.count) / Double(targetSampleRate),
                usedVADSegmentation: speechSegments != nil
            )
        case .sileroVAD, .sortformer:
            // Note (khazic): a Silero VAD or Sortformer server is never a loaded ASR model.
            preconditionFailure("a Silero VAD or Sortformer server transcribes nothing")
        }
        // Note (khazic): chunk and window segments have chunk timing, which Voxt discards.
        return MLXDetachedInferenceResult(rawText: text, senseVoiceMetadata: nil, structuredSegments: [])
    }

    /// SenseVoice metadata keeps one segment per pass, with that pass's chunk bounds.
    private nonisolated static func senseVoiceOmniResult(
        _ result: OmniTranscriptionResult,
        durationSeconds: Double,
        usedVADSegmentation: Bool
    ) throws -> MLXDetachedInferenceResult {
        if usedVADSegmentation, result.segments.isEmpty {
            let structuredError = MLXStructuredTranscriptionError.senseVoiceLongFormNoSpeechSegments(
                durationSeconds
            )
            VoxtLog.asrError(structuredError.diagnosticDescription)
            throw structuredError
        }
        let passMetadata = result.segments.map { segment in
            SenseVoiceTranscriptMetadata.fromOutput(
                ASRTranscriptionResult(
                    text: segment.text,
                    segments: [
                        ASRTranscriptSegment(
                            text: segment.text,
                            language: segment.language,
                            emotion: segment.emotion,
                            event: segment.event
                        ),
                    ],
                    language: segment.language,
                    languageProvenance: .detected
                ),
                startSeconds: usedVADSegmentation ? segment.startSeconds : 0,
                endSeconds: usedVADSegmentation ? segment.endSeconds : durationSeconds,
                usedVADSegmentation: usedVADSegmentation
            )
        }
        let metadata: SenseVoiceTranscriptMetadata?
        if usedVADSegmentation {
            var metadataSegments: [SenseVoiceSegmentMetadata] = []
            for pass in passMetadata {
                guard let pass else { continue }
                metadataSegments = SenseVoiceTranscriptMetadata.mergeSequentialSegments(
                    base: metadataSegments,
                    next: pass.segments
                )
            }
            metadata = SenseVoiceTranscriptMetadata.aggregated(
                segments: metadataSegments,
                usedVADSegmentation: true
            )
        } else {
            metadata = passMetadata.first ?? nil
        }
        return MLXDetachedInferenceResult(
            rawText: result.text,
            senseVoiceMetadata: metadata,
            structuredSegments: []
        )
    }
}
