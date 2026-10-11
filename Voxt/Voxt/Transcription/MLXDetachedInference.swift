// Off-actor inference consumes a resolved snapshot. Task cancellation and model leases remain in MLXTranscriber.

import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT
import MLXAudioVAD

private struct SenseVoiceInferenceResult {
    let output: STTOutput
    let metadata: SenseVoiceTranscriptMetadata?
}

struct MLXDetachedInferenceResult {
    let rawText: String
    let senseVoiceMetadata: SenseVoiceTranscriptMetadata?
    let structuredSegments: [MLXStructuredTranscriptSegment]
}

enum MLXStructuredTranscriptionError: LocalizedError {
    case senseVoiceLongFormVADUnavailable(String)
    case senseVoiceLongFormNoSpeechSegments(Double)

    var errorDescription: String? {
        switch self {
        case .senseVoiceLongFormVADUnavailable:
            return AppLocalization.localizedString("SenseVoice long audio processing is unavailable because the VAD model could not be prepared.")
        case .senseVoiceLongFormNoSpeechSegments:
            return AppLocalization.localizedString("SenseVoice could not detect any speech segments in this long audio clip.")
        }
    }

    var diagnosticDescription: String {
        switch self {
        case .senseVoiceLongFormVADUnavailable(let detail):
            return "SenseVoice long-form VAD unavailable. detail=\(detail)"
        case .senseVoiceLongFormNoSpeechSegments(let durationSeconds):
            return "SenseVoice long-form VAD produced no speech segments. durationSec=\(String(format: "%.2f", durationSeconds))"
        }
    }
}

extension MLXTranscriber {
    struct ResolvedInferenceConfiguration: Sendable {
        let family: MLXModelFamily
        let generationParameters: STTGenerateParameters
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
            // Keep MLXAudio AVAudioConverter resampling for Final/live quality.
            return try resampleAudio(samples, from: Int(sampleRate), to: targetSampleRate)
        }
        return samples
    }

    nonisolated static func runStreamingInferenceDetached(
        model: any STTGenerationModel,
        audioSamples: [Float],
        inferenceConfiguration: ResolvedInferenceConfiguration,
        longFormVADModel: SileroVAD?,
        targetSampleRate: Int,
        directPassMaximumDurationSeconds: Double,
        chunkMaximumDurationSeconds: Double,
        chunkOverlapSeconds: Double,
        vadThreshold: Float,
        vadMinSpeechDurationMs: Int,
        vadMinSilenceDurationMs: Int,
        vadSpeechPadMs: Int
    ) async throws -> MLXDetachedInferenceResult {
        try Task.checkCancellation()
        let audioArray = MLXArray(audioSamples)
        var streamedText = ""
        var finalOutput: STTOutput?

        let stream: AsyncThrowingStream<STTGeneration, Error>
        let generationParameters = inferenceConfiguration.generationParameters
        if let qwenModel = model as? Qwen3ASRModel {
            stream = qwenModel.generateStream(
                audio: audioArray,
                maxTokens: generationParameters.maxTokens,
                temperature: generationParameters.temperature,
                context: inferenceConfiguration.qwenContextBias,
                language: inferenceConfiguration.languageHint,
                chunkDuration: generationParameters.chunkDuration,
                minChunkDuration: generationParameters.minChunkDuration,
                kvBits: generationParameters.kvBits,
                kvGroupSize: generationParameters.kvGroupSize,
                quantizedKVStart: generationParameters.quantizedKVStart
            )
        } else if let senseVoiceModel = model as? SenseVoiceModel {
            let result = try runSenseVoiceInferenceDetached(
                model: senseVoiceModel,
                audioSamples: audioSamples,
                languageHint: inferenceConfiguration.languageHint,
                useITN: inferenceConfiguration.senseVoiceUseITN,
                verbose: generationParameters.verbose,
                vadModel: longFormVADModel,
                targetSampleRate: targetSampleRate,
                directPassMaximumDurationSeconds: directPassMaximumDurationSeconds,
                chunkMaximumDurationSeconds: chunkMaximumDurationSeconds,
                chunkOverlapSeconds: chunkOverlapSeconds,
                vadThreshold: vadThreshold,
                vadMinSpeechDurationMs: vadMinSpeechDurationMs,
                vadMinSilenceDurationMs: vadMinSilenceDurationMs,
                vadSpeechPadMs: vadSpeechPadMs
            )
            return MLXDetachedInferenceResult(
                rawText: result.output.text,
                senseVoiceMetadata: result.metadata,
                structuredSegments: []
            )
        } else if let mossModel = model as? MossTranscribeDiarizeModel {
            stream = mossModel.generateStream(
                audio: audioArray,
                generationParameters: generationParameters,
                prompt: inferenceConfiguration.mossPrompt
            )
        } else if let cohereModel = model as? CohereTranscribeModel,
                  let longFormVADModel {
            let output = try cohereModel.generateWithVAD(
                audio: audioArray,
                generationParameters: generationParameters,
                vad: (
                    model: longFormVADModel,
                    config: longFormSpeechSegmentConfig(
                        chunkMaximumDurationSeconds: chunkMaximumDurationSeconds,
                        vadThreshold: vadThreshold,
                        vadMinSpeechDurationMs: vadMinSpeechDurationMs,
                        vadMinSilenceDurationMs: vadMinSilenceDurationMs,
                        vadSpeechPadMs: vadSpeechPadMs
                    )
                )
            )
            return MLXDetachedInferenceResult(rawText: output.text, senseVoiceMetadata: nil, structuredSegments: [])
        } else {
            stream = model.generateStream(audio: audioArray, generationParameters: generationParameters)
        }

        for try await event in stream {
            try Task.checkCancellation()
            switch event {
            case .token(let token):
                streamedText += token
                await Task.yield()
            case .info:
                break
            case .result(let output):
                finalOutput = output
            }
        }

        if model is MossTranscribeDiarizeModel {
            let rawText = finalOutput?.text ?? streamedText
            return MLXDetachedInferenceResult(
                rawText: MossASRTranscriptRendering.renderedText(
                    rawText,
                    outputMode: inferenceConfiguration.mossOutputMode
                ),
                senseVoiceMetadata: nil,
                structuredSegments: mossStructuredSegments(from: finalOutput?.segments)
            )
        }

        return MLXDetachedInferenceResult(
            rawText: finalOutput?.text ?? streamedText,
            senseVoiceMetadata: nil,
            structuredSegments: reliableStructuredSegments(
                from: finalOutput?.segments,
                timingGranularity: inferenceConfiguration.timingGranularity
            )
        )
    }

    /// A model on the local Omni server decodes with the chunking, budget and
    /// stop rules its MLXAudio model used.
    nonisolated static func runOmniInferenceDetached(
        runtime: OmniASRRuntime,
        audioSamples: [Float],
        inferenceConfiguration: ResolvedInferenceConfiguration,
        targetSampleRate: Int,
        speechSegments: OmniSpeechSegments?
    ) async throws -> MLXDetachedInferenceResult {
        try Task.checkCancellation()
        let parameters = inferenceConfiguration.generationParameters
        let text: String
        switch runtime.kind {
        case .qwen3ASR:
            text = try await runtime.transcribeQwenFinal(
                samples: audioSamples,
                sampleRate: targetSampleRate,
                language: inferenceConfiguration.languageHint,
                context: inferenceConfiguration.qwenContextBias,
                maxTokens: parameters.maxTokens,
                chunkDurationSeconds: parameters.chunkDuration,
                minChunkDurationSeconds: parameters.minChunkDuration
            ).text
        case .whisper:
            text = try await runtime.transcribe(OmniASRRuntime.whisperFinalRequest(
                samples: audioSamples,
                sampleRate: targetSampleRate,
                language: inferenceConfiguration.languageHint,
                maxNewTokens: parameters.maxTokens,
                temperature: parameters.temperature
            )).text
        case .cohereTranscribe:
            text = try await runtime.transcribe(OmniASRRuntime.cohereFinalRequest(
                samples: audioSamples,
                sampleRate: targetSampleRate,
                language: inferenceConfiguration.languageHint,
                usePunctuation: parameters.usePunctuation,
                maxNewTokens: parameters.maxTokens,
                temperature: parameters.temperature,
                chunkDuration: parameters.chunkDuration,
                minChunkDuration: parameters.minChunkDuration,
                speechSegments: speechSegments
            )).text
        case .mossTranscribeDiarize:
            let result = try await runtime.transcribeMossFinal(
                samples: audioSamples,
                sampleRate: targetSampleRate,
                prompt: inferenceConfiguration.mossPrompt,
                maxTokens: parameters.maxTokens
            )
            return MLXDetachedInferenceResult(
                rawText: MossASRTranscriptRendering.renderedText(
                    result.text,
                    outputMode: inferenceConfiguration.mossOutputMode
                ),
                senseVoiceMetadata: nil,
                structuredSegments: mossStructuredSegments(from: result.segments.map(\.transcriptSegment))
            )
        case .sileroVAD, .sortformer:
            // Note (khazic): a Silero VAD or Sortformer server is never a loaded ASR model.
            preconditionFailure("a Silero VAD or Sortformer server transcribes nothing")
        }
        // Note (khazic): chunk and window segments have chunk timing, which Voxt discards.
        return MLXDetachedInferenceResult(rawText: text, senseVoiceMetadata: nil, structuredSegments: [])
    }

    nonisolated static func longFormSpeechSegmentConfig(
        chunkMaximumDurationSeconds: Double,
        vadThreshold: Float,
        vadMinSpeechDurationMs: Int,
        vadMinSilenceDurationMs: Int,
        vadSpeechPadMs: Int
    ) -> SpeechSegmentConfig {
        SpeechSegmentConfig(
            threshold: vadThreshold,
            minSpeechMs: vadMinSpeechDurationMs,
            minSilenceMs: vadMinSilenceDurationMs,
            speechPadMs: vadSpeechPadMs,
            mergeGapS: 1.0,
            maxChunkS: Float(chunkMaximumDurationSeconds),
            noSpeechPolicy: .returnEmpty
        )
    }

    private nonisolated static func runSenseVoiceInferenceDetached(
        model: SenseVoiceModel,
        audioSamples: [Float],
        languageHint: String?,
        useITN: Bool,
        verbose: Bool,
        vadModel: SileroVAD?,
        targetSampleRate: Int,
        directPassMaximumDurationSeconds: Double,
        chunkMaximumDurationSeconds: Double,
        chunkOverlapSeconds: Double,
        vadThreshold: Float,
        vadMinSpeechDurationMs: Int,
        vadMinSilenceDurationMs: Int,
        vadSpeechPadMs: Int
    ) throws -> SenseVoiceInferenceResult {
        let durationSeconds = Double(audioSamples.count) / Double(targetSampleRate)
        let resolvedLanguage = normalizedSenseVoiceLanguageHint(languageHint)

        guard MLXTranscriptionPlanning.shouldUseSenseVoiceVAD(
            sampleCount: audioSamples.count,
            sampleRate: targetSampleRate,
            directPassMaximumDurationSeconds: directPassMaximumDurationSeconds
        ) else {
            let output = model.generate(
                audio: MLXArray(audioSamples),
                language: resolvedLanguage,
                useITN: useITN,
                verbose: verbose
            )
            return SenseVoiceInferenceResult(
                output: output,
                metadata: SenseVoiceTranscriptMetadata.fromOutput(
                    output,
                    startSeconds: 0,
                    endSeconds: durationSeconds,
                    usedVADSegmentation: false
                )
            )
        }

        let ranges: [Range<Int>]
        do {
            ranges = try resolvedSenseVoiceSegmentRangesDetached(
                for: audioSamples,
                vad: vadModel,
                targetSampleRate: targetSampleRate,
                chunkMaximumDurationSeconds: chunkMaximumDurationSeconds,
                chunkOverlapSeconds: chunkOverlapSeconds,
                vadThreshold: vadThreshold,
                vadMinSpeechDurationMs: vadMinSpeechDurationMs,
                vadMinSilenceDurationMs: vadMinSilenceDurationMs,
                vadSpeechPadMs: vadSpeechPadMs
            )
        } catch {
            let structuredError = MLXStructuredTranscriptionError.senseVoiceLongFormVADUnavailable(
                error.localizedDescription
            )
            VoxtLog.asrError(structuredError.diagnosticDescription)
            throw structuredError
        }

        guard !ranges.isEmpty else {
            let structuredError = MLXStructuredTranscriptionError.senseVoiceLongFormNoSpeechSegments(
                durationSeconds
            )
            VoxtLog.asrError(structuredError.diagnosticDescription)
            throw structuredError
        }

        let rangeDurations = ranges.map {
            Double($0.upperBound - $0.lowerBound) / Double(targetSampleRate)
        }
        VoxtLog.asr(
            "SenseVoice VAD segmentation planned. audioDurationSec=\(String(format: "%.3f", durationSeconds)), segmentCount=\(ranges.count), minSegmentSec=\(String(format: "%.3f", rangeDurations.min() ?? 0)), maxSegmentSec=\(String(format: "%.3f", rangeDurations.max() ?? 0)), threshold=\(String(format: "%.3f", vadThreshold))",
            verbose: true
        )

        var mergedText = ""
        var metadataSegments: [SenseVoiceSegmentMetadata] = []

        for range in ranges {
            try Task.checkCancellation()
            let chunkSamples = Array(audioSamples[range])
            guard !chunkSamples.isEmpty else { continue }
            let output = model.generate(
                audio: MLXArray(chunkSamples),
                language: resolvedLanguage,
                useITN: useITN,
                verbose: verbose
            )
            let chunkText = output.text.trimmingCharacters(in: .whitespacesAndNewlines)
            mergedText = MLXTranscriptionPlanning.mergeSequentialTranscript(base: mergedText, next: chunkText)
            if let metadata = SenseVoiceTranscriptMetadata.fromOutput(
                output,
                startSeconds: Double(range.lowerBound) / Double(targetSampleRate),
                endSeconds: Double(range.upperBound) / Double(targetSampleRate),
                usedVADSegmentation: true
            ) {
                metadataSegments = SenseVoiceTranscriptMetadata.mergeSequentialSegments(
                    base: metadataSegments,
                    next: metadata.segments
                )
            }
        }

        let metadata = SenseVoiceTranscriptMetadata.aggregated(
            segments: metadataSegments,
            usedVADSegmentation: true
        )
        let output = STTOutput(
            text: mergedText,
            segments: metadataSegments.map { segment in
                STTTranscriptSegment(
                    text: segment.text,
                    startTime: segment.startSeconds,
                    endTime: segment.endSeconds,
                    language: segment.language,
                    emotion: segment.emotion,
                    event: segment.event
                )
            },
            language: metadata?.language,
            languageProvenance: .detected
        )
        return SenseVoiceInferenceResult(output: output, metadata: metadata)
    }

    private nonisolated static func resolvedSenseVoiceSegmentRangesDetached(
        for audioSamples: [Float],
        vad: SileroVAD?,
        targetSampleRate: Int,
        chunkMaximumDurationSeconds: Double,
        chunkOverlapSeconds: Double,
        vadThreshold: Float,
        vadMinSpeechDurationMs: Int,
        vadMinSilenceDurationMs: Int,
        vadSpeechPadMs: Int
    ) throws -> [Range<Int>] {
        guard let vad else {
            throw MLXStructuredTranscriptionError.senseVoiceLongFormVADUnavailable("VAD model is not loaded.")
        }
        let timestamps = try vad.getSpeechTimestamps(
            MLXArray(audioSamples),
            sampleRate: targetSampleRate,
            threshold: vadThreshold,
            minSpeechDurationMs: vadMinSpeechDurationMs,
            minSilenceDurationMs: vadMinSilenceDurationMs,
            speechPadMs: vadSpeechPadMs
        )
        let maxChunkSamples = Int(chunkMaximumDurationSeconds * Double(targetSampleRate))
        let overlapSamples = Int(chunkOverlapSeconds * Double(targetSampleRate))
        return timestamps.flatMap { timestamp in
            MLXTranscriptionPlanning.splitSenseVoiceRange(
                start: max(0, min(timestamp.start, audioSamples.count)),
                end: max(0, min(timestamp.end, audioSamples.count)),
                maxChunkSamples: maxChunkSamples,
                overlapSamples: overlapSamples
            )
        }
    }

    private nonisolated static func normalizedSenseVoiceLanguageHint(_ languageHint: String?) -> String {
        let normalized = languageHint?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? "auto"
        switch normalized {
        case "zh", "en", "yue", "ja", "ko", "nospeech":
            return normalized
        default:
            return "auto"
        }
    }
}
