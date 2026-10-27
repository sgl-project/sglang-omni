// SpeakerDiarizationSettings.swift
// Shared speaker-label post-processing settings; inference uses Sortformer.

import Foundation

enum MeetingSpeakerDiarizationSensitivity: String, CaseIterable, Identifiable, Codable, Hashable, Sendable {
    case stable
    case balanced
    case sensitive

    var id: String { rawValue }

    var title: String {
        switch self {
        case .stable:
            return AppLocalization.localizedString("Stable")
        case .balanced:
            return AppLocalization.localizedString("Balanced")
        case .sensitive:
            return AppLocalization.localizedString("Sensitive")
        }
    }

    var detail: String {
        switch self {
        case .stable:
            return AppLocalization.localizedString("Prefer fewer false speaker switches.")
        case .balanced:
            return AppLocalization.localizedString("Balance speaker recall and label stability.")
        case .sensitive:
            return AppLocalization.localizedString("Detect shorter speaker turns with a higher risk of extra speaker labels.")
        }
    }

    nonisolated var minimumSpeakerConfidence: Double {
        switch self {
        case .stable: return 0.42
        case .balanced: return 0.28
        case .sensitive: return 0.18
        }
    }

    nonisolated var smootherOptions: MeetingSpeakerTurnSmoother.Options {
        switch self {
        case .stable:
            return .init(minimumTurnDurationSeconds: 1.2, sameSpeakerMergeGapSeconds: 1.8)
        case .balanced:
            return .init(minimumTurnDurationSeconds: 1.0, sameSpeakerMergeGapSeconds: 1.4)
        case .sensitive:
            return .init(minimumTurnDurationSeconds: 0.35, sameSpeakerMergeGapSeconds: 0.5)
        }
    }

    nonisolated var transcriptAssemblyOptions: MeetingSpeakerTranscriptAssembler.Options {
        switch self {
        case .stable:
            return .init(
                dominantSpeakerOverlapRatio: 0.94,
                minimumTurnOverlapSeconds: 0.22,
                minimumSecondarySpeakerOverlapSeconds: 1.4,
                minimumSecondarySpeakerOverlapRatio: 0.18,
                splitsSegmentsOnSpeakerBoundaries: false
            )
        case .balanced:
            return .init(
                dominantSpeakerOverlapRatio: 0.92,
                minimumTurnOverlapSeconds: 0.18,
                minimumSecondarySpeakerOverlapSeconds: 1.0,
                minimumSecondarySpeakerOverlapRatio: 0.14,
                splitsSegmentsOnSpeakerBoundaries: true
            )
        case .sensitive:
            return .init(
                dominantSpeakerOverlapRatio: 0.72,
                minimumTurnOverlapSeconds: 0.08,
                minimumSecondarySpeakerOverlapSeconds: 0.35,
                minimumSecondarySpeakerOverlapRatio: 0.04,
                splitsSegmentsOnSpeakerBoundaries: true
            )
        }
    }
}

// Keep the persisted identifier for existing feature settings, not an engine picker.
enum MeetingDiarizationMode: String, CaseIterable, Identifiable, Codable, Hashable, Sendable {
    case sortformerV2

    var id: String { rawValue }

    var title: String {
        AppLocalization.localizedString("Sortformer v2")
    }

    var detail: String {
        AppLocalization.localizedString("Use NVIDIA Sortformer v2 for final meeting speaker analysis.")
    }

    var fallbackRemoteSizeText: String {
        MeetingVADModelStorage.sortformerFallbackRemoteSizeText
    }

    nonisolated static func stored(in defaults: UserDefaults = .standard) -> MeetingDiarizationMode {
        let rawValue = defaults.string(forKey: AppPreferenceKey.meetingSpeakerDiarizationModel) ?? ""
        return MeetingDiarizationMode(rawValue: rawValue) ?? .sortformerV2
    }
}

struct MeetingSpeakerDiarizationOptions: Equatable, Sendable {
    var minimumAudioDurationSeconds: TimeInterval
    var minimumSpeakerConfidence: Double
    var smoothing: MeetingSpeakerTurnSmoother.Options
    var transcriptAssembly: MeetingSpeakerTranscriptAssembler.Options
    var sensitivity: MeetingSpeakerDiarizationSensitivity
    var debugLoggingEnabled: Bool

    nonisolated init(
        minimumAudioDurationSeconds: TimeInterval = 2.0,
        sensitivity: MeetingSpeakerDiarizationSensitivity = .balanced,
        minimumSpeakerConfidence: Double? = nil,
        smoothing: MeetingSpeakerTurnSmoother.Options? = nil,
        transcriptAssembly: MeetingSpeakerTranscriptAssembler.Options? = nil,
        debugLoggingEnabled: Bool = false
    ) {
        self.minimumAudioDurationSeconds = minimumAudioDurationSeconds
        self.sensitivity = sensitivity
        self.minimumSpeakerConfidence = minimumSpeakerConfidence ?? sensitivity.minimumSpeakerConfidence
        self.smoothing = smoothing ?? sensitivity.smootherOptions
        self.transcriptAssembly = transcriptAssembly ?? sensitivity.transcriptAssemblyOptions
        self.debugLoggingEnabled = debugLoggingEnabled
    }

    static func fromPreferences(defaults _: UserDefaults = .standard) -> MeetingSpeakerDiarizationOptions {
        MeetingSpeakerDiarizationOptions()
    }
}
