// RecordingVoiceActivityFrameDecider.swift
// Provides recording voice activity decisions and debug statistics.

import Foundation

struct RecordingVoiceActivityDecisionResult: Sendable {
    let decision: ASRVoiceActivityFrameDecision
    let source: Source

    enum Source: Sendable {
        case energy
        case silero
        case omni
        case fallbackEnergy

        var telemetryName: String {
            switch self {
            case .energy:
                return "energy"
            case .silero:
                return "silero"
            case .omni:
                return "omnivad"
            case .fallbackEnergy:
                return "fallback-energy"
            }
        }
    }

    var probabilityText: String {
        decision.probability.map { String(format: "%.3f", $0) } ?? "nil"
    }
}

struct RecordingVoiceActivityDebugStats: Sendable {
    var mode: LocalVADMode = .automatic
    var useCase: ASRVoiceActivityUseCase = .transcription
    var backend: ASRVoiceActivityBackendKind = .off
    private(set) var totalFrames = 0
    private(set) var speechFrames = 0
    private(set) var silenceFrames = 0
    private(set) var fallbackFrames = 0
    private(set) var events = 0
    private(set) var speechStartedEvents = 0
    private(set) var speechEndedEvents = 0
    private(set) var speechForcedEvents = 0
    private(set) var speechRejectedEvents = 0
    private(set) var probabilityFrames = 0
    private(set) var probabilitySum: Float = 0
    private(set) var minProbability: Float?
    private(set) var maxProbability: Float?
    private(set) var levelFrames = 0
    private(set) var levelSum: Float = 0
    private(set) var minLevel: Float?
    private(set) var maxLevel: Float?
    private(set) var firstFrameStartSeconds: TimeInterval?
    private(set) var lastFrameEndSeconds: TimeInterval?
    private(set) var sourceCounts: [String: Int] = [:]

    mutating func record(
        frame: ASRVoiceActivityAudioFrame,
        result: RecordingVoiceActivityDecisionResult,
        resolvedSpeech: Bool,
        events newEvents: [ASRVoiceActivityEvent],
        fallbackSource: RecordingVoiceActivityDecisionResult.Source
    ) {
        totalFrames += 1
        if resolvedSpeech {
            speechFrames += 1
        } else {
            silenceFrames += 1
        }
        if result.source.telemetryName == fallbackSource.telemetryName {
            fallbackFrames += 1
        }
        sourceCounts[result.source.telemetryName, default: 0] += 1

        if let probability = result.decision.probability {
            probabilityFrames += 1
            probabilitySum += probability
            minProbability = min(minProbability ?? probability, probability)
            maxProbability = max(maxProbability ?? probability, probability)
        }

        if let level = frame.level {
            levelFrames += 1
            levelSum += level
            minLevel = min(minLevel ?? level, level)
            maxLevel = max(maxLevel ?? level, level)
        }

        firstFrameStartSeconds = min(firstFrameStartSeconds ?? frame.startSeconds, frame.startSeconds)
        lastFrameEndSeconds = max(lastFrameEndSeconds ?? frame.endSeconds, frame.endSeconds)

        events += newEvents.count
        for event in newEvents {
            switch event {
            case .speechStarted:
                speechStartedEvents += 1
            case .speechEnded:
                speechEndedEvents += 1
            case .speechForced:
                speechForcedEvents += 1
            case .speechRejected:
                speechRejectedEvents += 1
            }
        }
    }

    var telemetrySummary: String {
        let speechRatio = totalFrames > 0 ? Double(speechFrames) / Double(totalFrames) : 0
        let probabilityAverage = probabilityFrames > 0 ? probabilitySum / Float(probabilityFrames) : nil
        let levelAverage = levelFrames > 0 ? levelSum / Float(levelFrames) : nil
        let audioSpanMs = firstFrameStartSeconds.flatMap { first in
            lastFrameEndSeconds.map { max(0, Int((($0 - first) * 1_000).rounded())) }
        }
        return [
            "mode=\(mode.rawValue)",
            "backend=\(backend.rawValue)",
            "useCase=\(useCase.rawValue)",
            "frames=\(totalFrames)",
            "speechFrames=\(speechFrames)",
            "silenceFrames=\(silenceFrames)",
            "speechRatio=\(Self.percentText(speechRatio))",
            "sources=\(Self.sourceCountsText(sourceCounts))",
            "fallbackFrames=\(fallbackFrames)",
            "events=\(events)",
            "speechStarted=\(speechStartedEvents)",
            "speechEnded=\(speechEndedEvents)",
            "speechForced=\(speechForcedEvents)",
            "speechRejected=\(speechRejectedEvents)",
            "probabilityMin=\(Self.floatText(minProbability))",
            "probabilityAvg=\(Self.floatText(probabilityAverage))",
            "probabilityMax=\(Self.floatText(maxProbability))",
            "levelMin=\(Self.floatText(minLevel))",
            "levelAvg=\(Self.floatText(levelAverage))",
            "levelMax=\(Self.floatText(maxLevel))",
            "audioSpanMs=\(audioSpanMs.map(String.init) ?? "nil")"
        ].joined(separator: ", ")
    }

    private static func sourceCountsText(_ counts: [String: Int]) -> String {
        guard !counts.isEmpty else { return "none" }
        return counts
            .sorted { $0.key < $1.key }
            .map { "\($0.key):\($0.value)" }
            .joined(separator: "|")
    }

    private static func floatText(_ value: Float?) -> String {
        value.map { String(format: "%.3f", $0) } ?? "nil"
    }

    private static func percentText(_ value: Double) -> String {
        String(format: "%.1f%%", value * 100)
    }
}

actor RecordingVoiceActivityFrameDecider {
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

    func reset() async {
        await sileroDetector.reset()
        await omniDetector.reset()
        sileroFallbackWarningLogged = false
        omniDegradedWarningLogged = false
    }

    func releaseResources() async {
        await sileroDetector.unload()
        await omniDetector.releaseResources()
        sileroFallbackWarningLogged = false
        omniDegradedWarningLogged = false
    }

    func decision(for frame: ASRVoiceActivityAudioFrame) async -> RecordingVoiceActivityDecisionResult? {
        switch ASRVoiceActivityRuntimePolicy.effectiveBackend(mode: mode, useCase: useCase) {
        case .off:
            return nil
        case .energy:
            return await energyDecision(for: frame, source: .energy)
        case .mlxSilero:
            if let sileroDecision = await sileroDecision(for: frame) {
                return sileroDecision
            }
            return await energyDecision(for: frame, source: .fallbackEnergy)
        case .omniStream:
            if let omniDecision = await omniDecision(for: frame) {
                return omniDecision
            }
            return await energyDecision(for: frame, source: .fallbackEnergy)
        }
    }

    private func sileroDecision(for frame: ASRVoiceActivityAudioFrame) async -> RecordingVoiceActivityDecisionResult? {
        do {
            if let probability = try await sileroDetector.probability(
                samples: frame.samples,
                sampleRate: frame.sampleRate,
                streamID: "recording-\(useCase.rawValue)"
            ) {
                return RecordingVoiceActivityDecisionResult(
                    decision: ASRVoiceActivityFrameDecision(
                        startSeconds: frame.startSeconds,
                        endSeconds: frame.endSeconds,
                        isSpeech: probability >= sileroThreshold,
                        probability: probability
                    ),
                    source: .silero
                )
            }
        } catch {
            if shouldLogSileroFallback(error) {
                VoxtLog.asrWarning("Recording Silero VAD failed; falling back to energy VAD. error=\(error.localizedDescription)")
                sileroFallbackWarningLogged = true
            }
            await sileroDetector.reset()
        }
        return nil
    }

    private func omniDecision(for frame: ASRVoiceActivityAudioFrame) async -> RecordingVoiceActivityDecisionResult? {
        do {
            if let decision = try await omniDetector.decision(
                for: frame,
                streamID: "recording-\(useCase.rawValue)"
            ) {
                return RecordingVoiceActivityDecisionResult(
                    decision: decision,
                    source: .omni
                )
            }
        } catch {
            if !omniDegradedWarningLogged {
                VoxtLog.asrWarning("Recording OmniVAD unavailable; degrading current session to energy VAD. error=\(error.localizedDescription)")
                omniDegradedWarningLogged = true
            }
            await omniDetector.reset()
        }
        return nil
    }

    private func energyDecision(
        for frame: ASRVoiceActivityAudioFrame,
        source: RecordingVoiceActivityDecisionResult.Source
    ) async -> RecordingVoiceActivityDecisionResult? {
        guard let decision = try? await energyBackend.decision(for: frame) else {
            return nil
        }
        return RecordingVoiceActivityDecisionResult(decision: decision, source: source)
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
