// RecordingStartPlanner.swift
// Provides Recording Start Planner for recording session routing.

import Foundation

enum RecordingStartBlockReason: Equatable {
    case mlxModelNotInstalled
    case mlxModelDownloading
    case mlxModelUnavailable(detail: String?)

    var userMessage: String {
        switch self {
        case .mlxModelNotInstalled:
            return AppLocalization.localizedString("MLX model is not downloaded. Open Settings > Model to install it.")
        case .mlxModelDownloading:
            return AppLocalization.localizedString("MLX model is still downloading. Wait for installation to finish and try again.")
        case .mlxModelUnavailable(let detail):
            return detailedUnavailableMessage(
                base: AppLocalization.localizedString("MLX model is unavailable. Open Settings > Model to fix it."),
                detailedFormat: AppLocalization.localizedString("MLX model is unavailable. Open Settings > Model to fix it.\nReason: %@"),
                detail: detail
            )
        }
    }

    var logDescription: String {
        switch self {
        case .mlxModelNotInstalled:
            return "MLX Audio model is not downloaded."
        case .mlxModelDownloading:
            return "MLX Audio model download is still in progress."
        case .mlxModelUnavailable(let detail):
            return detailedLogDescription(
                base: "MLX Audio model is unavailable.",
                detail: detail
            )
        }
    }

    var reminderDuration: TimeInterval {
        switch self {
        case .mlxModelUnavailable(let detail):
            return normalizedDetail(detail) == nil ? 2.4 : 4.2
        default:
            return 2.4
        }
    }

    private func detailedUnavailableMessage(
        base: String,
        detailedFormat: String,
        detail: String?
    ) -> String {
        guard let detail = normalizedDetail(detail) else { return base }
        return String(format: detailedFormat, detail)
    }

    private func detailedLogDescription(base: String, detail: String?) -> String {
        guard let detail = normalizedDetail(detail) else { return base }
        return "\(base) reason=\(detail)"
    }

    private func normalizedDetail(_ detail: String?) -> String? {
        guard let detail else { return nil }
        let trimmed = detail.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

enum RecordingStartDecision: Equatable {
    case start(TranscriptionEngine)
    case blocked(RecordingStartBlockReason)
}

enum RecordingStartPlanner {
    private enum DownloadableModelAvailability {
        case ready
        case notDownloaded
        case downloadingSelectedModel
        case unavailable(String?)
    }

    private enum DownloadStatePhase {
        case ready
        case notDownloaded
        case activeDownload
        case unavailable
    }

    static func resolve(
        selectedEngine: TranscriptionEngine,
        selectedMLXRepo: String? = nil,
        activeMLXDownloadRepo: String? = nil,
        isSelectedMLXModelDownloaded: Bool = false,
        mlxModelState: MLXModelManager.ModelState,
    ) -> RecordingStartDecision {
        switch selectedEngine {
        case .dictation:
            return .start(.dictation)
        case .remote:
            return .start(.remote)
        case .mlxAudio:
            return decision(
                engine: .mlxAudio,
                availability: mlxAvailability(
                    selectedRepo: selectedMLXRepo,
                    activeDownloadRepo: activeMLXDownloadRepo,
                    isSelectedModelDownloaded: isSelectedMLXModelDownloaded,
                    state: mlxModelState
                ),
                notInstalledReason: .mlxModelNotInstalled,
                downloadingReason: .mlxModelDownloading,
                unavailableReason: { .mlxModelUnavailable(detail: $0) }
            )
        }
    }

    private static func decision(
        engine: TranscriptionEngine,
        availability: DownloadableModelAvailability,
        notInstalledReason: RecordingStartBlockReason,
        downloadingReason: RecordingStartBlockReason,
        unavailableReason: (String?) -> RecordingStartBlockReason
    ) -> RecordingStartDecision {
        switch availability {
        case .ready:
            return .start(engine)
        case .notDownloaded:
            return .blocked(notInstalledReason)
        case .downloadingSelectedModel:
            return .blocked(downloadingReason)
        case .unavailable(let detail):
            return .blocked(unavailableReason(detail))
        }
    }

    private static func mlxAvailability(
        selectedRepo: String?,
        activeDownloadRepo: String?,
        isSelectedModelDownloaded: Bool,
        state: MLXModelManager.ModelState
    ) -> DownloadableModelAvailability {
        availability(
            selectedIdentifier: selectedRepo,
            activeIdentifier: activeDownloadRepo,
            isSelectedModelDownloaded: isSelectedModelDownloaded,
            canonicalize: MLXModelManager.canonicalModelRepo,
            state: state
        )
    }


    private static func availability(
        selectedIdentifier: String?,
        activeIdentifier: String?,
        isSelectedModelDownloaded: Bool,
        canonicalize: (String) -> String,
        state: MLXModelManager.ModelState
    ) -> DownloadableModelAvailability {
        availability(
            isSelectedDownloadActive: isSelectedOperationActive(
                selectedIdentifier: selectedIdentifier,
                activeIdentifier: activeIdentifier,
                canonicalize: canonicalize
            ),
            isSelectedModelDownloaded: isSelectedModelDownloaded,
            phase: downloadStatePhase(for: state),
            unavailableDetail: modelErrorDetail(from: state)
        )
    }

    private static func availability(
        isSelectedDownloadActive: Bool,
        isSelectedModelDownloaded: Bool,
        phase: DownloadStatePhase,
        unavailableDetail: String? = nil
    ) -> DownloadableModelAvailability {
        switch phase {
        case .ready:
            return .ready
        case .notDownloaded:
            return .notDownloaded
        case .activeDownload:
            if isSelectedDownloadActive {
                return .downloadingSelectedModel
            }
            return isSelectedModelDownloaded ? .ready : .notDownloaded
        case .unavailable:
            return .unavailable(unavailableDetail)
        }
    }

    private static func downloadStatePhase(for state: MLXModelManager.ModelState) -> DownloadStatePhase {
        switch state {
        case .downloaded, .ready, .loading:
            return .ready
        case .notDownloaded:
            return .notDownloaded
        case .downloading, .paused:
            return .activeDownload
        case .error:
            return .unavailable
        }
    }

    private static func isSelectedOperationActive(
        selectedIdentifier: String?,
        activeIdentifier: String?,
        canonicalize: (String) -> String
    ) -> Bool {
        guard let selectedIdentifier, let activeIdentifier else { return false }
        return canonicalize(selectedIdentifier) == canonicalize(activeIdentifier)
    }

    private static func modelErrorDetail(from state: MLXModelManager.ModelState) -> String? {
        guard case .error(let message) = state else { return nil }
        return message
    }
}
