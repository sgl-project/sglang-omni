// RemoteASRAudioUploadPreprocessing.swift
// Provides remote ASR upload audio policy and client-side VAD preprocessing.

import Foundation

enum RemoteASRAudioUploadVADPolicy: Equatable, Sendable {
    case disabled(reason: String)
    case fileUploadClientVAD
    case realtimeServerVAD
    case realtimeUnchanged(reason: String)

    var usesFileUploadClientVAD: Bool {
        if case .fileUploadClientVAD = self {
            return true
        }
        return false
    }

    var telemetryName: String {
        switch self {
        case .disabled(let reason):
            return "disabled:\(reason)"
        case .fileUploadClientVAD:
            return "file-upload-client-vad"
        case .realtimeServerVAD:
            return "realtime-server-vad"
        case .realtimeUnchanged(let reason):
            return "realtime-unchanged:\(reason)"
        }
    }
}

enum RemoteASRAudioUploadVADPolicyResolver {
    static func policy(
        provider: RemoteASRProvider,
        model: String,
        localVADMode: LocalVADMode
    ) -> RemoteASRAudioUploadVADPolicy {
        guard localVADMode != .off else {
            return .disabled(reason: "local-vad-off")
        }

        let resolvedModel = model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? provider.suggestedModel
            : model
        switch provider {
        case .openAIWhisper, .glmASR, .xiaomiMiMoASR:
            return .fileUploadClientVAD
        case .stepFunASR:
            if RemoteASRRealtimeSupport.isStepFunRealtimeModel(resolvedModel) {
                return .realtimeServerVAD
            }
            return .fileUploadClientVAD
        case .aliyunBailianASR:
            if RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: resolvedModel) != nil {
                return .realtimeServerVAD
            }
            if RemoteASREndpointSupport.isAliyunFunRealtimeModel(resolvedModel) {
                return .realtimeUnchanged(reason: "aliyun-fun-realtime")
            }
            if RemoteASREndpointSupport.isAliyunFileTranscriptionModel(resolvedModel) {
                return .fileUploadClientVAD
            }
            return .disabled(reason: "unsupported-aliyun-route")
        case .doubaoASR:
            return .realtimeUnchanged(reason: "doubao-streaming")
        case .googleGeminiASR:
            return .realtimeServerVAD
        }
    }
}

struct RemoteASRAudioUploadPreparation: Sendable {
    let uploadFileURL: URL
    let temporaryUploadFileURL: URL?
    let policy: RemoteASRAudioUploadVADPolicy
    let originalDurationSeconds: TimeInterval?
    let uploadDurationSeconds: TimeInterval?
    let speechSegmentCount: Int
    let observedSpeech: Bool?

    var shouldRequestRemoteASR: Bool {
        observedSpeech != false
    }

    func cleanupTemporaryUploadFileIfNeeded() {
        guard let temporaryUploadFileURL else { return }
        try? FileManager.default.removeItem(at: temporaryUploadFileURL)
    }

    static func original(
        fileURL: URL,
        policy: RemoteASRAudioUploadVADPolicy,
        originalDurationSeconds: TimeInterval? = nil,
        observedSpeech: Bool? = nil
    ) -> RemoteASRAudioUploadPreparation {
        RemoteASRAudioUploadPreparation(
            uploadFileURL: fileURL,
            temporaryUploadFileURL: nil,
            policy: policy,
            originalDurationSeconds: originalDurationSeconds,
            uploadDurationSeconds: originalDurationSeconds,
            speechSegmentCount: 0,
            observedSpeech: observedSpeech
        )
    }
}

enum RemoteASRAudioUploadPreprocessor {
    nonisolated static let minimumAudioDurationSeconds: TimeInterval = 3.0
    nonisolated static let minimumReductionRatioForFilteredUpload: Double = 0.15
    nonisolated static let defaultEnergyThreshold: Float = 0.06
    nonisolated static let frameSize = 1_024

    static func prepareUploadAudio(
        originalFileURL: URL,
        provider: RemoteASRProvider,
        configuration: RemoteProviderConfiguration,
        localVADMode: LocalVADMode,
        useCase: ASRVoiceActivityUseCase,
        energyThreshold: Float = defaultEnergyThreshold
    ) async throws -> RemoteASRAudioUploadPreparation {
        let policy = RemoteASRAudioUploadVADPolicyResolver.policy(
            provider: provider,
            model: configuration.model,
            localVADMode: localVADMode
        )
        guard policy.usesFileUploadClientVAD else {
            return .original(fileURL: originalFileURL, policy: policy)
        }

        let samples = try HistoryAudioArchiveSupport.readWAVSamples(from: originalFileURL)
        let sampleRate = HistoryAudioArchiveSupport.targetSampleRate
        let originalDurationSeconds = Double(samples.count) / sampleRate
        guard originalDurationSeconds >= minimumAudioDurationSeconds else {
            return .original(
                fileURL: originalFileURL,
                policy: .disabled(reason: "short-audio"),
                originalDurationSeconds: originalDurationSeconds
            )
        }

        let decisions = await voiceActivityDecisions(
            samples: samples,
            sampleRate: sampleRate,
            mode: localVADMode,
            useCase: useCase,
            energyThreshold: energyThreshold
        )
        let result = ASRVoiceActivitySampleFilter.filter(
            samples: samples,
            sampleRate: sampleRate,
            decisions: decisions,
            configuration: ASRVoiceActivityConfiguration.profile(for: useCase)
        )

        guard result.observedFrames else {
            return .original(
                fileURL: originalFileURL,
                policy: .disabled(reason: "no-vad-frames"),
                originalDurationSeconds: originalDurationSeconds
            )
        }
        guard result.observedSpeech else {
            return RemoteASRAudioUploadPreparation(
                uploadFileURL: originalFileURL,
                temporaryUploadFileURL: nil,
                policy: policy,
                originalDurationSeconds: result.originalDurationSeconds,
                uploadDurationSeconds: 0,
                speechSegmentCount: 0,
                observedSpeech: false
            )
        }
        guard !result.samples.isEmpty,
              result.reductionRatio >= minimumReductionRatioForFilteredUpload
        else {
            return .original(
                fileURL: originalFileURL,
                policy: .disabled(reason: "low-reduction"),
                originalDurationSeconds: result.originalDurationSeconds,
                observedSpeech: true
            )
        }

        let uploadURL = HistoryAudioArchiveSupport.temporaryArchiveURL(prefix: "voxt-remote-asr-upload-vad")
        do {
            let didExport = try HistoryAudioArchiveSupport.exportWAV(
                samples: result.samples,
                sampleRate: sampleRate,
                to: uploadURL
            )
            guard didExport else {
                return .original(
                    fileURL: originalFileURL,
                    policy: .disabled(reason: "empty-export"),
                    originalDurationSeconds: result.originalDurationSeconds,
                    observedSpeech: true
                )
            }
        } catch {
            try? FileManager.default.removeItem(at: uploadURL)
            throw error
        }

        return RemoteASRAudioUploadPreparation(
            uploadFileURL: uploadURL,
            temporaryUploadFileURL: uploadURL,
            policy: policy,
            originalDurationSeconds: result.originalDurationSeconds,
            uploadDurationSeconds: result.filteredDurationSeconds,
            speechSegmentCount: result.speechSegments.count,
            observedSpeech: true
        )
    }

    private static func voiceActivityDecisions(
        samples: [Float],
        sampleRate: Double,
        mode: LocalVADMode,
        useCase: ASRVoiceActivityUseCase,
        energyThreshold: Float
    ) async -> [ASRVoiceActivityFrameDecision] {
        let decider = RemoteASRAudioUploadVADFrameDecider(
            mode: mode,
            useCase: useCase,
            energyThreshold: energyThreshold
        )
        var decisions: [ASRVoiceActivityFrameDecision] = []
        decisions.reserveCapacity(max(samples.count / frameSize, 1))

        var index = 0
        while index < samples.count {
            let endIndex = min(samples.count, index + frameSize)
            let frameSamples = Array(samples[index..<endIndex])
            let startSeconds = Double(index) / sampleRate
            let endSeconds = Double(endIndex) / sampleRate
            let level = AudioLevelMeter.normalizedLevel(fromSamples: frameSamples)
            let frame = ASRVoiceActivityAudioFrame(
                samples: frameSamples,
                sampleRate: sampleRate,
                startSeconds: startSeconds,
                endSeconds: endSeconds,
                level: level
            )
            if let decision = await decider.decision(for: frame) {
                decisions.append(decision)
            }
            index = endIndex
        }
        return decisions
    }
}

private actor RemoteASRAudioUploadVADFrameDecider {
    private let mode: LocalVADMode
    private let useCase: ASRVoiceActivityUseCase
    private let energyBackend: ASREnergyVoiceActivityBackend
    private let sileroThreshold: Float
    private let sileroDetector = ASRSileroStreamingVoiceActivityDetector()
    private let omniDetector: OmniStreamVoiceActivityBackend
    private var sileroFallbackWarningLogged = false
    private var omniDegradedWarningLogged = false

    init(
        mode: LocalVADMode,
        useCase: ASRVoiceActivityUseCase,
        energyThreshold: Float
    ) {
        self.mode = mode
        self.useCase = useCase
        self.energyBackend = ASREnergyVoiceActivityBackend(threshold: energyThreshold)
        self.sileroThreshold = ASRVoiceActivityConfiguration.profile(for: useCase).onsetProbabilityThreshold
        self.omniDetector = OmniStreamVoiceActivityBackend(useCase: useCase)
    }

    func decision(for frame: ASRVoiceActivityAudioFrame) async -> ASRVoiceActivityFrameDecision? {
        switch ASRVoiceActivityRuntimePolicy.effectiveBackend(mode: mode, useCase: useCase) {
        case .off:
            return nil
        case .energy:
            return energyDecision(for: frame)
        case .mlxSilero:
            return await sileroDecision(for: frame) ?? energyDecision(for: frame)
        case .omniStream:
            return await omniDecision(for: frame) ?? energyDecision(for: frame)
        }
    }

    private func sileroDecision(for frame: ASRVoiceActivityAudioFrame) async -> ASRVoiceActivityFrameDecision? {
        do {
            if let probability = try await sileroDetector.probability(
                samples: frame.samples,
                sampleRate: frame.sampleRate,
                streamID: "remote-upload-\(useCase.rawValue)"
            ) {
                return ASRVoiceActivityFrameDecision(
                    startSeconds: frame.startSeconds,
                    endSeconds: frame.endSeconds,
                    isSpeech: probability >= sileroThreshold,
                    probability: probability
                )
            }
        } catch {
            if shouldLogSileroFallback(error) {
                VoxtLog.asrWarning("Remote upload Silero VAD failed; falling back to energy VAD. error=\(error.localizedDescription)")
                sileroFallbackWarningLogged = true
            }
            await sileroDetector.reset()
        }
        return nil
    }

    private func omniDecision(for frame: ASRVoiceActivityAudioFrame) async -> ASRVoiceActivityFrameDecision? {
        do {
            return try await omniDetector.decision(
                for: frame,
                streamID: "remote-upload-\(useCase.rawValue)"
            )
        } catch {
            if !omniDegradedWarningLogged {
                VoxtLog.asrWarning("Remote upload OmniVAD unavailable; degrading current session to energy VAD. error=\(error.localizedDescription)")
                omniDegradedWarningLogged = true
            }
            await omniDetector.reset()
        }
        return nil
    }

    private func energyDecision(for frame: ASRVoiceActivityAudioFrame) -> ASRVoiceActivityFrameDecision? {
        let level = frame.level ?? 0
        return ASRVoiceActivityFrameDecision(
            startSeconds: frame.startSeconds,
            endSeconds: frame.endSeconds,
            isSpeech: level >= energyBackend.threshold,
            probability: nil
        )
    }

    private func shouldLogSileroFallback(_ error: Error) -> Bool {
        if let modelError = error as? MeetingVADModelError {
            switch modelError {
            case .modelNotDownloaded, .runtimeUnavailable:
                return false
            }
        }
        return !sileroFallbackWarningLogged
    }
}
