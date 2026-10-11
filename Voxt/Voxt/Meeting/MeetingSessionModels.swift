// MeetingSessionModels.swift
// Provides Meeting Session Models for meeting session behavior.

import Foundation
import Combine

struct MeetingSessionResult {
    let recoverySessionID: UUID?
    let captureMode: MeetingCaptureMode
    let transcriptionEngine: TranscriptionEngine
    let transcriptionModelDescription: String
    let segments: [MeetingTranscriptSegment]
    let visibleSnapshotSegments: [MeetingTranscriptSegment]
    let audioDurationSeconds: TimeInterval
    let archivedAudioURL: URL?
    let captureFailureMessage: String?

    init(
        recoverySessionID: UUID? = nil,
        captureMode: MeetingCaptureMode = .meeting,
        transcriptionEngine: TranscriptionEngine,
        transcriptionModelDescription: String,
        segments: [MeetingTranscriptSegment],
        visibleSnapshotSegments: [MeetingTranscriptSegment],
        audioDurationSeconds: TimeInterval,
        archivedAudioURL: URL?,
        captureFailureMessage: String? = nil
    ) {
        self.recoverySessionID = recoverySessionID
        self.captureMode = captureMode
        self.transcriptionEngine = transcriptionEngine
        self.transcriptionModelDescription = transcriptionModelDescription
        self.segments = segments
        self.visibleSnapshotSegments = visibleSnapshotSegments
        self.audioDurationSeconds = audioDurationSeconds
        self.archivedAudioURL = archivedAudioURL
        self.captureFailureMessage = captureFailureMessage
    }

    var persistedSegments: [MeetingTranscriptSegment] {
        let primarySegments = MeetingTranscriptFormatter.meaningfulSegments(for: segments)
        if !primarySegments.isEmpty {
            return primarySegments
        }
        return MeetingTranscriptFormatter.mergedSegmentsForPersistence(
            primarySegments: segments,
            fallbackSegments: visibleSnapshotSegments
        )
    }

    var combinedText: String {
        MeetingTranscriptFormatter.joinedText(for: persistedSegments)
    }
}

enum MeetingCaptureMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case meeting
    case subtitles
    case recording

    var id: String { rawValue }

    nonisolated var usesMicrophone: Bool {
        switch self {
        case .meeting, .recording:
            return true
        case .subtitles:
            return false
        }
    }

    nonisolated var usesSystemAudio: Bool {
        switch self {
        case .meeting, .subtitles:
            return true
        case .recording:
            return false
        }
    }

    func includes(speaker: MeetingSpeaker) -> Bool {
        switch speaker {
        case .me:
            return usesMicrophone
        case .them:
            return usesSystemAudio
        }
    }

    nonisolated var capabilities: MeetingModeCapabilities {
        MeetingModeCapabilities(mode: self)
    }

    var title: String {
        switch self {
        case .meeting:
            return AppLocalization.localizedString("Meeting")
        case .subtitles:
            return AppLocalization.localizedString("Subtitles")
        case .recording:
            return AppLocalization.localizedString("Recording")
        }
    }

    var detailBadgeTitle: String {
        switch self {
        case .meeting:
            return AppLocalization.localizedString("Meeting Mode")
        case .subtitles:
            return AppLocalization.localizedString("Subtitles Mode")
        case .recording:
            return AppLocalization.localizedString("Recording Mode")
        }
    }

    var accessibilityDescription: String {
        switch self {
        case .meeting:
            return AppLocalization.localizedString("Record microphone and system audio.")
        case .subtitles:
            return AppLocalization.localizedString("Record system audio only.")
        case .recording:
            return AppLocalization.localizedString("Record microphone only.")
        }
    }

    var sourceDescription: String {
        switch self {
        case .meeting:
            return AppLocalization.localizedString("Built-in audio + microphone")
        case .subtitles:
            return AppLocalization.localizedString("Built-in audio only")
        case .recording:
            return AppLocalization.localizedString("Microphone only")
        }
    }

    static func stored(in defaults: UserDefaults = .standard) -> MeetingCaptureMode {
        let rawValue = defaults.string(forKey: AppPreferenceKey.meetingCaptureMode) ?? ""
        return MeetingCaptureMode(rawValue: rawValue) ?? .meeting
    }

    func persist(in defaults: UserDefaults = .standard) {
        defaults.set(rawValue, forKey: AppPreferenceKey.meetingCaptureMode)
    }

    nonisolated func sourceTransition(to nextMode: MeetingCaptureMode) -> MeetingCaptureSourceTransition {
        MeetingCaptureSourceTransition(
            stopsMicrophone: usesMicrophone && !nextMode.usesMicrophone,
            stopsSystemAudio: usesSystemAudio && !nextMode.usesSystemAudio,
            startsMicrophone: !usesMicrophone && nextMode.usesMicrophone,
            startsSystemAudio: !usesSystemAudio && nextMode.usesSystemAudio
        )
    }
}

nonisolated struct MeetingCaptureSourceTransition: Equatable, Sendable {
    let stopsMicrophone: Bool
    let stopsSystemAudio: Bool
    let startsMicrophone: Bool
    let startsSystemAudio: Bool
}

struct MeetingModeCapabilities: Equatable, Sendable {
    let mode: MeetingCaptureMode

    nonisolated var allowsSpeakerFeatures: Bool {
        switch mode {
        case .meeting:
            return true
        case .subtitles, .recording:
            return false
        }
    }

    nonisolated var realtimeDiarizationSources: Set<TranscriptAudioSource> {
        []
    }

    nonisolated var finalDiarizationSources: Set<TranscriptAudioSource> {
        switch mode {
        case .meeting:
            return [.systemAudio]
        case .subtitles, .recording:
            return []
        }
    }

    nonisolated func defaultSpeaker(for source: TranscriptAudioSource) -> MeetingSpeaker {
        switch (mode, source) {
        case (.meeting, .microphone):
            return .me
        case (.meeting, .systemAudio), (.meeting, .mixed):
            return .them
        case (.subtitles, _):
            return .them
        case (.recording, .microphone):
            return .me
        case (.recording, .mixed):
            return .them
        case (.recording, .systemAudio):
            return .them
        }
    }

    nonisolated func shouldRunRealtimeDiarization(for source: TranscriptAudioSource) -> Bool {
        realtimeDiarizationSources.contains(source)
    }

    nonisolated func shouldRunFinalDiarization(for source: TranscriptAudioSource) -> Bool {
        finalDiarizationSources.contains(source)
    }
}

@MainActor
final class MeetingOverlayState: ObservableObject {
    @Published var isPresented = false
    @Published var isRecording = false
    @Published var isModelInitializing = false
    @Published var isFinalizing = false
    @Published var isPaused = false
    @Published var isCollapsed = false
    @Published var realtimeTranslateEnabled = false
    @Published var captureMode: MeetingCaptureMode = .meeting
    @Published var isCaptureModePickerPresented = false
    @Published var isRealtimeTranslationLanguagePickerPresented = false
    @Published var isCloseConfirmationPresented = false
    @Published var realtimeTranslationDraftLanguageRaw = TranslationTargetLanguage.english.rawValue
    @Published var safetyMessage: String?
    @Published var segments: [MeetingTranscriptSegment] = []

    let waveformState = RecentAudioWaveformState()

    func reset() {
        isPresented = false
        isRecording = false
        isModelInitializing = false
        isFinalizing = false
        isPaused = false
        isCollapsed = false
        waveformState.reset()
        waveformState.setActive(false)
        realtimeTranslateEnabled = false
        captureMode = .meeting
        isCaptureModePickerPresented = false
        isRealtimeTranslationLanguagePickerPresented = false
        isCloseConfirmationPresented = false
        realtimeTranslationDraftLanguageRaw = TranslationTargetLanguage.english.rawValue
        safetyMessage = nil
        segments = []
    }
}
