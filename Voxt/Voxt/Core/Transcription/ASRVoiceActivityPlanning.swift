// ASRVoiceActivityPlanning.swift
// Shared voice activity segmentation primitives for ASR capture and batching.

import Foundation

enum SileroVADModelSupport {
    nonisolated static let repo = "mlx-community/silero-vad-v6"
}

enum LocalVADMode: String, CaseIterable, Identifiable, Codable, Hashable, Sendable {
    case automatic
    case silero
    case omni
    case energy
    case off

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic:
            return AppLocalization.localizedString("Automatic")
        case .silero:
            return AppLocalization.localizedString("Silero")
        case .omni:
            return AppLocalization.localizedString("OmniVAD")
        case .energy:
            return AppLocalization.localizedString("Energy")
        case .off:
            return AppLocalization.localizedString("Off")
        }
    }

    var detail: String {
        switch self {
        case .automatic:
            return AppLocalization.localizedString("Let Voxt choose the local VAD path for the current workflow.")
        case .silero:
            return AppLocalization.localizedString("Use the local Silero VAD model for ASR gating.")
        case .omni:
            return AppLocalization.localizedString("Use OmniVAD-Kit streaming VAD for local ASR gating.")
        case .energy:
            return AppLocalization.localizedString("Use fast local level detection with no extra model.")
        case .off:
            return AppLocalization.localizedString("Disable local VAD gating.")
        }
    }

    nonisolated static let defaultMode: LocalVADMode = .automatic

    nonisolated static func resolved(rawValue: String?) -> LocalVADMode {
        switch rawValue {
        case "automatic", "auto":
            return .automatic
        case "silero", "mlxSilero":
            return .silero
        case "omni", "omnivad", "omniVAD":
            return .omni
        case "energy":
            return .energy
        case "off", "disabled":
            return .off
        default:
            return defaultMode
        }
    }

    nonisolated static func stored(defaults: UserDefaults = .standard) -> LocalVADMode {
        resolved(rawValue: defaults.string(forKey: AppPreferenceKey.localVADMode))
    }

    nonisolated static func save(_ mode: LocalVADMode, defaults: UserDefaults = .standard) {
        defaults.set(mode.rawValue, forKey: AppPreferenceKey.localVADMode)
        NotificationCenter.default.post(name: .voxtFeatureSettingsDidChange, object: nil)
    }
}

enum MeetingSileroVADSensitivity: String, CaseIterable, Identifiable, Codable, Hashable, Sendable {
    case responsive
    case balanced
    case stable

    var id: String { rawValue }

    var title: String {
        switch self {
        case .responsive:
            // Use "Sensitive" rather than "Responsive" so zh/ja are not confused with
            // the dictation latency preset that already localizes "Responsive" as low-latency.
            return AppLocalization.localizedString("Sensitive")
        case .balanced:
            return AppLocalization.localizedString("Balanced")
        case .stable:
            return AppLocalization.localizedString("Stable")
        }
    }

    var detail: String {
        switch self {
        case .responsive:
            return AppLocalization.localizedString("Detect quieter speech sooner; may include more background sound.")
        case .balanced:
            return AppLocalization.localizedString("Balance speech pickup with background-noise rejection.")
        case .stable:
            return AppLocalization.localizedString("Prefer stronger speech signals in noisy environments.")
        }
    }

    /// Streaming-path onset thresholds. Balanced keeps the historical live default of 0.5
    /// (not the offline meeting profile 0.45) so upgrading does not silently increase noise pickup.
    nonisolated var onsetProbabilityThreshold: Float {
        switch self {
        case .responsive:
            return 0.38
        case .balanced:
            return 0.5
        case .stable:
            return 0.55
        }
    }

    nonisolated static func resolved(rawValue: String?) -> MeetingSileroVADSensitivity {
        MeetingSileroVADSensitivity(rawValue: rawValue ?? "") ?? .balanced
    }

    nonisolated static func stored(defaults: UserDefaults = .standard) -> MeetingSileroVADSensitivity {
        resolved(rawValue: defaults.string(forKey: AppPreferenceKey.meetingSileroVADSensitivity))
    }

    nonisolated func configuration(
        base: ASRVoiceActivityConfiguration = .meeting
    ) -> ASRVoiceActivityConfiguration {
        ASRVoiceActivityConfiguration(
            onsetProbabilityThreshold: onsetProbabilityThreshold,
            offsetProbabilityThreshold: base.offsetProbabilityThreshold,
            minSpeechSeconds: base.minSpeechSeconds,
            minSilenceSeconds: base.minSilenceSeconds,
            speechPadSeconds: base.speechPadSeconds,
            maxSegmentSeconds: base.maxSegmentSeconds
        )
    }
}

enum ASRVoiceActivityBackendKind: String, CaseIterable, Identifiable, Codable, Hashable, Sendable {
    case off
    case energy
    case mlxSilero
    case omniStream

    var id: String { rawValue }

    var title: String {
        switch self {
        case .off:
            return AppLocalization.localizedString("Off")
        case .energy:
            return AppLocalization.localizedString("Energy")
        case .mlxSilero:
            return AppLocalization.localizedString("Silero")
        case .omniStream:
            return AppLocalization.localizedString("OmniVAD")
        }
    }

    var detail: String {
        switch self {
        case .off:
            return AppLocalization.localizedString("Disable local VAD gating.")
        case .energy:
            return AppLocalization.localizedString("Use fast local level detection with no extra model.")
        case .mlxSilero:
            return AppLocalization.localizedString("Use the current local MLX Silero VAD model.")
        case .omniStream:
            return AppLocalization.localizedString("Use the local OmniVAD-Kit streaming VAD model.")
        }
    }

    nonisolated static let defaultMeetingBackend: ASRVoiceActivityBackendKind = .mlxSilero

    nonisolated static func resolved(rawValue: String?) -> ASRVoiceActivityBackendKind {
        switch rawValue {
        case "off", "disabled":
            return .off
        case "energy":
            return .energy
        case "silero", "mlxSilero":
            return .mlxSilero
        case "omni", "omnivad", "omniVAD", "omniStream":
            return .omniStream
        default:
            return defaultMeetingBackend
        }
    }

}

enum ASRVoiceActivityUseCase: String, Codable, Hashable, Sendable {
    case transcription
    case translation
    case rewrite
    case meeting
}

nonisolated struct ASRVoiceActivityAudioFrame: Equatable, Sendable {
    let samples: [Float]
    let sampleRate: Double
    let startSeconds: TimeInterval
    let endSeconds: TimeInterval
    let level: Float?

    init(
        samples: [Float],
        sampleRate: Double,
        startSeconds: TimeInterval,
        endSeconds: TimeInterval,
        level: Float? = nil
    ) {
        self.samples = samples
        self.sampleRate = sampleRate.isFinite ? max(0, sampleRate) : 0
        self.startSeconds = startSeconds.isFinite ? max(0, startSeconds) : 0
        self.endSeconds = endSeconds.isFinite ? max(self.startSeconds, endSeconds) : self.startSeconds
        if let level, level.isFinite {
            self.level = max(0, level)
        } else {
            self.level = nil
        }
    }
}

nonisolated protocol ASRVoiceActivityBackend: Sendable {
    var kind: ASRVoiceActivityBackendKind { get }

    func reset() async
    func decision(for frame: ASRVoiceActivityAudioFrame) async throws -> ASRVoiceActivityFrameDecision
}

extension ASRVoiceActivityBackend {
    func reset() async {}
}

nonisolated struct ASREnergyVoiceActivityBackend: ASRVoiceActivityBackend {
    let kind: ASRVoiceActivityBackendKind = .energy
    let threshold: Float

    init(threshold: Float) {
        self.threshold = max(0, threshold)
    }

    func decision(for frame: ASRVoiceActivityAudioFrame) async throws -> ASRVoiceActivityFrameDecision {
        let level = frame.level ?? Self.rmsLevel(samples: frame.samples)
        return ASRVoiceActivityFrameDecision(
            startSeconds: frame.startSeconds,
            endSeconds: frame.endSeconds,
            isSpeech: level >= threshold,
            probability: nil
        )
    }

    private static func rmsLevel(samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var finiteSampleCount = 0
        let sumOfSquares = samples.reduce(Float(0)) { partial, sample in
            guard sample.isFinite else { return partial }
            finiteSampleCount += 1
            return partial + sample * sample
        }
        guard finiteSampleCount > 0 else { return 0 }
        return sqrt(sumOfSquares / Float(finiteSampleCount))
    }
}

nonisolated enum ASRVoiceActivityLocalGatePolicy: Equatable, Sendable {
    case enabled
    case disabled(reason: String)

    var isEnabled: Bool {
        if case .enabled = self {
            return true
        }
        return false
    }
}

nonisolated enum ASRVoiceActivityRuntimePolicy {
    nonisolated static func requiresSileroModel(mode: LocalVADMode) -> Bool {
        switch mode {
        case .automatic, .silero:
            return true
        case .omni, .energy, .off:
            return false
        }
    }

    nonisolated static func effectiveBackend(
        mode: LocalVADMode,
        useCase: ASRVoiceActivityUseCase
    ) -> ASRVoiceActivityBackendKind {
        switch mode {
        case .automatic, .silero:
            return .mlxSilero
        case .omni:
            return .omniStream
        case .energy:
            return .energy
        case .off:
            return .off
        }
    }

    nonisolated static func localGatePolicy(
        transcriptionEngine: TranscriptionEngine,
        mode: LocalVADMode
    ) -> ASRVoiceActivityLocalGatePolicy {
        guard mode != .off else {
            return .disabled(reason: "local-vad-off")
        }
        guard transcriptionEngine == .mlxAudio else {
            return .disabled(reason: "non-local-asr")
        }
        return .enabled
    }

    nonisolated static func shouldUseLevelTiming(
        localVADGateActive: Bool,
        hasVoiceActivityFrames: Bool
    ) -> Bool {
        !localVADGateActive || !hasVoiceActivityFrames
    }

    nonisolated static func shouldSuppressFinalTranscription(
        localVADGateActive: Bool,
        observedVoiceActivityFrames: Bool,
        observedSpeech: Bool
    ) -> Bool {
        localVADGateActive && observedVoiceActivityFrames && !observedSpeech
    }
}

nonisolated enum ASRLocalIntermediateGatePolicy {
    static let defaultMinimumTrailingSilenceSeconds: TimeInterval = 2.0

    nonisolated static func shouldTriggerIntermediateTranscription(
        transcriptionEngine: TranscriptionEngine,
        localVADGateActive: Bool,
        silentDuration: TimeInterval,
        didTriggerPauseTranscription: Bool,
        observedSpeechEnd: Bool,
        minimumTrailingSilenceSeconds: TimeInterval = defaultMinimumTrailingSilenceSeconds
    ) -> Bool {
        transcriptionEngine == .mlxAudio
            && localVADGateActive
            && silentDuration >= max(0, minimumTrailingSilenceSeconds)
            && !didTriggerPauseTranscription
            && observedSpeechEnd
    }
}

nonisolated struct ASRVoiceActivityConfiguration: Equatable, Sendable {
    let onsetProbabilityThreshold: Float
    let offsetProbabilityThreshold: Float
    let minSpeechSeconds: TimeInterval
    let minSilenceSeconds: TimeInterval
    let speechPadSeconds: TimeInterval
    let maxSegmentSeconds: TimeInterval?

    init(
        onsetProbabilityThreshold: Float,
        offsetProbabilityThreshold: Float,
        minSpeechSeconds: TimeInterval,
        minSilenceSeconds: TimeInterval,
        speechPadSeconds: TimeInterval,
        maxSegmentSeconds: TimeInterval?
    ) {
        self.onsetProbabilityThreshold = max(0, min(onsetProbabilityThreshold, 1))
        self.offsetProbabilityThreshold = max(0, min(offsetProbabilityThreshold, 1))
        self.minSpeechSeconds = max(0, minSpeechSeconds)
        self.minSilenceSeconds = max(0, minSilenceSeconds)
        self.speechPadSeconds = max(0, speechPadSeconds)
        self.maxSegmentSeconds = maxSegmentSeconds.map { max(0.01, $0) }
    }

    static let balanced = ASRVoiceActivityConfiguration(
        onsetProbabilityThreshold: 0.5,
        offsetProbabilityThreshold: 0.35,
        minSpeechSeconds: 0.22,
        minSilenceSeconds: 0.42,
        speechPadSeconds: 0.18,
        maxSegmentSeconds: 24
    )

    static let realtime = ASRVoiceActivityConfiguration(
        onsetProbabilityThreshold: 0.45,
        offsetProbabilityThreshold: 0.3,
        minSpeechSeconds: 0.14,
        minSilenceSeconds: 0.24,
        speechPadSeconds: 0.12,
        maxSegmentSeconds: 8
    )

    static let transcription = ASRVoiceActivityConfiguration(
        onsetProbabilityThreshold: 0.46,
        offsetProbabilityThreshold: 0.32,
        minSpeechSeconds: 0.16,
        minSilenceSeconds: 0.32,
        speechPadSeconds: 0.14,
        maxSegmentSeconds: 10
    )

    static let translation = ASRVoiceActivityConfiguration(
        onsetProbabilityThreshold: 0.52,
        offsetProbabilityThreshold: 0.38,
        minSpeechSeconds: 0.28,
        minSilenceSeconds: 0.52,
        speechPadSeconds: 0.20,
        maxSegmentSeconds: 18
    )

    static let rewrite = ASRVoiceActivityConfiguration(
        onsetProbabilityThreshold: 0.54,
        offsetProbabilityThreshold: 0.40,
        minSpeechSeconds: 0.30,
        minSilenceSeconds: 0.56,
        speechPadSeconds: 0.22,
        maxSegmentSeconds: 18
    )

    static let meeting = ASRVoiceActivityConfiguration(
        onsetProbabilityThreshold: 0.45,
        offsetProbabilityThreshold: 0.30,
        minSpeechSeconds: 0.14,
        minSilenceSeconds: 0.24,
        speechPadSeconds: 0.12,
        maxSegmentSeconds: 8
    )

    nonisolated static func profile(for useCase: ASRVoiceActivityUseCase) -> ASRVoiceActivityConfiguration {
        switch useCase {
        case .transcription:
            return .transcription
        case .translation:
            return .translation
        case .rewrite:
            return .rewrite
        case .meeting:
            return .meeting
        }
    }
}

nonisolated struct ASRVoiceActivityFrameDecision: Equatable, Sendable {
    let startSeconds: TimeInterval
    let endSeconds: TimeInterval
    let isSpeech: Bool
    let probability: Float?

    init(
        startSeconds: TimeInterval,
        endSeconds: TimeInterval,
        isSpeech: Bool,
        probability: Float? = nil
    ) {
        self.startSeconds = startSeconds.isFinite ? max(0, startSeconds) : 0
        self.endSeconds = endSeconds.isFinite ? max(self.startSeconds, endSeconds) : self.startSeconds
        self.isSpeech = isSpeech
        if let probability, probability.isFinite {
            self.probability = max(0, min(probability, 1))
        } else {
            self.probability = nil
        }
    }

    var durationSeconds: TimeInterval {
        max(0, endSeconds - startSeconds)
    }
}

nonisolated struct ASROfflineSpeechRange: Equatable, Sendable {
    let startSeconds: TimeInterval
    let endSeconds: TimeInterval

    nonisolated init(startSeconds: TimeInterval, endSeconds: TimeInterval) {
        self.startSeconds = max(0, startSeconds)
        self.endSeconds = max(self.startSeconds, endSeconds)
    }

    nonisolated func intersects(
        startSeconds otherStart: TimeInterval,
        endSeconds otherEnd: TimeInterval,
        tolerance: TimeInterval = 0
    ) -> Bool {
        let resolvedTolerance = max(0, tolerance)
        return endSeconds + resolvedTolerance > otherStart
            && startSeconds - resolvedTolerance < otherEnd
    }
}

protocol ASROfflineVoiceActivityBackend: Sendable {
    func speechRanges(samples: [Float], sampleRate: Double) async throws -> [ASROfflineSpeechRange]
}

nonisolated enum ASRVoiceActivitySampleRateConverter {
    private static let maximumExpansionRatio: Double = 32

    nonisolated static func resample(samples: [Float], from inputRate: Double, to outputRate: Double) -> [Float] {
        let finiteSamples = samples.map(sanitizedSample)
        guard !finiteSamples.isEmpty,
              inputRate.isFinite,
              outputRate.isFinite,
              inputRate > 0,
              outputRate > 0
        else {
            return finiteSamples
        }
        if abs(inputRate - outputRate) <= 1 {
            return finiteSamples
        }

        let ratio = outputRate / inputRate
        guard ratio.isFinite,
              ratio > 0,
              ratio <= maximumExpansionRatio
        else {
            return finiteSamples
        }
        let rawOutputCount = Double(finiteSamples.count) * ratio
        guard rawOutputCount.isFinite,
              rawOutputCount <= Double(Int.max)
        else {
            return finiteSamples
        }
        let outputCount = max(Int(rawOutputCount), 1)
        var output = [Float](repeating: 0, count: outputCount)

        for index in 0..<outputCount {
            let position = Double(index) / ratio
            let lowerIndex = min(Int(position), finiteSamples.count - 1)
            let upperIndex = min(lowerIndex + 1, finiteSamples.count - 1)
            let fraction = Float(position - Double(lowerIndex))
            let lower = finiteSamples[lowerIndex]
            let upper = finiteSamples[upperIndex]
            output[index] = lower + (upper - lower) * fraction
        }

        return output
    }

    private nonisolated static func sanitizedSample(_ sample: Float) -> Float {
        sample.isFinite ? sample : 0
    }
}

nonisolated struct ASRVoiceActivitySegment: Equatable, Sendable {
    let startSeconds: TimeInterval
    let endSeconds: TimeInterval
    let speechSeconds: TimeInterval
    let frameCount: Int

    var durationSeconds: TimeInterval {
        max(0, endSeconds - startSeconds)
    }

    var telemetrySummary: String {
        "startSec=\(Self.formatSeconds(startSeconds)), endSec=\(Self.formatSeconds(endSeconds)), durationSec=\(Self.formatSeconds(durationSeconds)), speechSec=\(Self.formatSeconds(speechSeconds)), frames=\(frameCount)"
    }

    private static func formatSeconds(_ value: TimeInterval) -> String {
        String(format: "%.3f", value)
    }
}

nonisolated enum ASRVoiceActivityEventReason: String, Equatable, Sendable {
    case onsetThresholdReached = "onset-threshold-reached"
    case minSilenceReached = "min-silence-reached"
    case maxSegmentDurationReached = "max-segment-duration-reached"
    case minSpeechDurationNotMet = "min-speech-duration-not-met"
}

nonisolated enum ASRVoiceActivityEvent: Equatable, Sendable {
    case speechStarted(startSeconds: TimeInterval)
    case speechEnded(ASRVoiceActivitySegment)
    case speechForced(ASRVoiceActivitySegment)
    case speechRejected(ASRVoiceActivitySegment)

    var reason: ASRVoiceActivityEventReason {
        switch self {
        case .speechStarted:
            return .onsetThresholdReached
        case .speechEnded:
            return .minSilenceReached
        case .speechForced:
            return .maxSegmentDurationReached
        case .speechRejected:
            return .minSpeechDurationNotMet
        }
    }

    var segment: ASRVoiceActivitySegment? {
        switch self {
        case .speechStarted:
            return nil
        case .speechEnded(let segment), .speechForced(let segment), .speechRejected(let segment):
            return segment
        }
    }

    var telemetrySummary: String {
        switch self {
        case .speechStarted(let startSeconds):
            return "event=speechStarted, reason=\(reason.rawValue), startSec=\(String(format: "%.3f", startSeconds))"
        case .speechEnded(let segment):
            return "event=speechEnded, reason=\(reason.rawValue), \(segment.telemetrySummary)"
        case .speechForced(let segment):
            return "event=speechForced, reason=\(reason.rawValue), \(segment.telemetrySummary)"
        case .speechRejected(let segment):
            return "event=speechRejected, reason=\(reason.rawValue), \(segment.telemetrySummary)"
        }
    }
}

nonisolated struct ASRVoiceActivitySegmenter: Sendable {
    private let configuration: ASRVoiceActivityConfiguration
    private var activeStartSeconds: TimeInterval?
    private var lastSpeechEndSeconds: TimeInterval?
    private var silenceStartSeconds: TimeInterval?
    private var speechSeconds: TimeInterval = 0
    private var frameCount: Int = 0

    init(configuration: ASRVoiceActivityConfiguration = .balanced) {
        self.configuration = configuration
    }

    mutating func reset() {
        activeStartSeconds = nil
        lastSpeechEndSeconds = nil
        silenceStartSeconds = nil
        speechSeconds = 0
        frameCount = 0
    }

    mutating func append(_ decision: ASRVoiceActivityFrameDecision) -> [ASRVoiceActivityEvent] {
        appendWithResolvedSpeechState(decision).events
    }

    mutating func appendWithResolvedSpeechState(
        _ decision: ASRVoiceActivityFrameDecision
    ) -> (events: [ASRVoiceActivityEvent], isSpeech: Bool) {
        guard decision.endSeconds > decision.startSeconds else { return ([], false) }

        let speech = resolvedSpeechState(for: decision)
        if speech {
            return (appendSpeechFrame(decision), true)
        }
        return (appendSilenceFrame(decision), false)
    }

    mutating func finish(at endSeconds: TimeInterval? = nil) -> ASRVoiceActivityEvent? {
        guard let segment = currentSegment(endSeconds: endSeconds) else { return nil }
        reset()
        if segment.speechSeconds >= configuration.minSpeechSeconds {
            return .speechEnded(segment)
        }
        return .speechRejected(segment)
    }

    private func resolvedSpeechState(for decision: ASRVoiceActivityFrameDecision) -> Bool {
        guard let probability = decision.probability else {
            return decision.isSpeech
        }
        if activeStartSeconds == nil {
            return probability >= configuration.onsetProbabilityThreshold
        }
        return probability >= configuration.offsetProbabilityThreshold
    }

    private mutating func appendSpeechFrame(_ decision: ASRVoiceActivityFrameDecision) -> [ASRVoiceActivityEvent] {
        var events: [ASRVoiceActivityEvent] = []
        if activeStartSeconds == nil {
            activeStartSeconds = max(0, decision.startSeconds - configuration.speechPadSeconds)
            events.append(.speechStarted(startSeconds: activeStartSeconds ?? decision.startSeconds))
        }

        silenceStartSeconds = nil
        lastSpeechEndSeconds = decision.endSeconds
        speechSeconds += decision.durationSeconds
        frameCount += 1

        if let maxSegmentSeconds = configuration.maxSegmentSeconds,
           let activeStartSeconds,
           decision.endSeconds - activeStartSeconds >= maxSegmentSeconds,
           let segment = currentSegment(endSeconds: decision.endSeconds) {
            reset()
            events.append(.speechForced(segment))
        }
        return events
    }

    private mutating func appendSilenceFrame(_ decision: ASRVoiceActivityFrameDecision) -> [ASRVoiceActivityEvent] {
        guard activeStartSeconds != nil else { return [] }
        frameCount += 1
        if silenceStartSeconds == nil {
            silenceStartSeconds = decision.startSeconds
        }
        guard let silenceStartSeconds,
              decision.endSeconds - silenceStartSeconds >= configuration.minSilenceSeconds,
              let segment = currentSegment(endSeconds: nil)
        else {
            return []
        }

        reset()
        if segment.speechSeconds >= configuration.minSpeechSeconds {
            return [.speechEnded(segment)]
        }
        return [.speechRejected(segment)]
    }

    private func currentSegment(endSeconds requestedEndSeconds: TimeInterval?) -> ASRVoiceActivitySegment? {
        guard let activeStartSeconds else { return nil }
        let speechEndSeconds = lastSpeechEndSeconds ?? activeStartSeconds
        let paddedEndSeconds = min(
            requestedEndSeconds ?? speechEndSeconds + configuration.speechPadSeconds,
            speechEndSeconds + configuration.speechPadSeconds
        )
        let endSeconds = max(activeStartSeconds, paddedEndSeconds)
        return ASRVoiceActivitySegment(
            startSeconds: activeStartSeconds,
            endSeconds: endSeconds,
            speechSeconds: speechSeconds,
            frameCount: frameCount
        )
    }
}
