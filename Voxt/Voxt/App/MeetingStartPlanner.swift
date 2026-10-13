// MeetingStartPlanner.swift
// Provides Meeting Start Planner for app lifecycle and routing.

import Foundation

enum MeetingStartBlockReason: Equatable {
    case dictationUnsupported
    case recording(RecordingStartBlockReason)
    case remoteASRUnavailable

    var userMessage: String {
        switch self {
        case .dictationUnsupported:
            return AppLocalization.localizedString("Meeting Notes currently supports MLX Audio and Remote ASR. Direct Dictation is not available for meetings.")
        case .recording(let reason):
            return reason.userMessage
        case .remoteASRUnavailable:
            return AppLocalization.localizedString("Remote ASR is not configured yet. Open Settings > Model to finish the provider setup.")
        }
    }

    var logDescription: String {
        switch self {
        case .dictationUnsupported:
            return "Meeting Notes does not support Direct Dictation."
        case .recording(let reason):
            return reason.logDescription
        case .remoteASRUnavailable:
            return "Remote ASR provider configuration is incomplete."
        }
    }
}

enum MeetingStartDecision: Equatable {
    case start(TranscriptionEngine)
    case blocked(MeetingStartBlockReason)
}

enum MeetingStartPlanner {
    static func resolve(
        selectedEngine: TranscriptionEngine,
        selectedMLXRepo: String? = nil,
        activeMLXDownloadRepo: String? = nil,
        isSelectedMLXModelDownloaded: Bool = false,
        mlxModelState: MLXModelManager.ModelState,
        remoteASRConfiguration: RemoteProviderConfiguration
    ) -> MeetingStartDecision {
        switch selectedEngine {
        case .dictation:
            return .blocked(.dictationUnsupported)
        case .mlxAudio:
            return resolveLocalMeetingStart(
                selectedMLXRepo: selectedMLXRepo,
                activeMLXDownloadRepo: activeMLXDownloadRepo,
                isSelectedMLXModelDownloaded: isSelectedMLXModelDownloaded,
                mlxModelState: mlxModelState
            )
        case .remote:
            guard remoteASRConfiguration.isConfigured else {
                return .blocked(.remoteASRUnavailable)
            }
            return .start(.remote)
        }
    }

    private static func resolveLocalMeetingStart(
        selectedMLXRepo: String?,
        activeMLXDownloadRepo: String?,
        isSelectedMLXModelDownloaded: Bool,
        mlxModelState: MLXModelManager.ModelState
    ) -> MeetingStartDecision {
        switch RecordingStartPlanner.resolve(
            selectedEngine: .mlxAudio,
            selectedMLXRepo: selectedMLXRepo,
            activeMLXDownloadRepo: activeMLXDownloadRepo,
            isSelectedMLXModelDownloaded: isSelectedMLXModelDownloaded,
            mlxModelState: mlxModelState
        ) {
        case .start:
            return .start(.mlxAudio)
        case .blocked(let reason):
            return .blocked(.recording(reason))
        }
    }
}
