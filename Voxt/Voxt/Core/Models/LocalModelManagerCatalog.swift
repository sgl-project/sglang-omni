import Foundation

// Catalog values and forwarding only. Each manager retains mutable state and lifetime.
extension MLXModelManager {
    enum ModelState: Equatable {
        case notDownloaded
        case downloading(
            progress: Double,
            completed: Int64,
            total: Int64,
            currentFile: String?,
            completedFiles: Int,
            totalFiles: Int
        )
        case paused(
            progress: Double,
            completed: Int64,
            total: Int64,
            currentFile: String?,
            completedFiles: Int,
            totalFiles: Int
        )
        case downloaded
        case loading
        case ready
        case error(String)
    }

    struct CatalogSnapshot: Equatable {
        let repo: String
        let isDownloaded: Bool
        let hasResumableDownload: Bool
        let state: ModelState
        let pausedStatusMessage: String?
        let hasActiveDownloadTask: Bool

        var isDownloading: Bool {
            if hasActiveDownloadTask {
                return true
            }
            if case .downloading = state {
                return true
            }
            return false
        }

        var isPaused: Bool {
            if case .paused = state {
                return true
            }
            return hasResumableDownload
        }
    }

    nonisolated enum ModelSizeState: Equatable, Sendable {
        case unknown
        case loading
        case ready(bytes: Int64, text: String)
        case error(String)
    }

    struct TranscriptionBehavior: Equatable {
        enum CorrectionMode: Equatable {
            case incremental
            case finalizationOnly
        }

        let correctionMode: CorrectionMode
        let allowsQuickStopPass: Bool
        let preloadsOnRecordingStart: Bool

        var runsIntermediateCorrections: Bool {
            correctionMode == .incremental
        }
    }

    func displayTitle(for repo: String) -> String {
        MLXModelCatalog.displayTitle(for: repo)
    }

    nonisolated static func ratingText(for repo: String) -> String {
        MLXModelCatalog.ratingText(for: repo)
    }

    nonisolated static func catalogTagKeys(for repo: String) -> [String] {
        MLXModelCatalog.catalogTagKeys(for: repo)
    }

    nonisolated static func fallbackRemoteSizeText(repo: String) -> String? {
        MLXModelCatalog.fallbackRemoteSizeText(repo: repo)
    }

    nonisolated static func canonicalModelRepo(_ repo: String) -> String {
        MLXModelCatalog.canonicalModelRepo(repo)
    }

    func displayModelsIncludingInstalled() -> [ModelOption] {
        Self.availableModels
    }

    nonisolated static func isMultilingualModelRepo(_ repo: String) -> Bool {
        MLXModelCatalog.isMultilingualModelRepo(repo)
    }

    /// Auxiliary VAD artifacts share download/storage handling, not the ASR
    /// catalog or STT loader. Never reopen arbitrary retired repo loading.
    nonisolated static func isManagedArtifactRepo(_ repo: String) -> Bool {
        repo == SileroVADModelSupport.repo || availableModels.contains { $0.id == repo }
    }

    nonisolated static func isAvailableModelRepo(_ repo: String) -> Bool {
        MLXModelCatalog.isAvailableModelRepo(repo)
    }

    nonisolated static func isRealtimeCapableModelRepo(_ repo: String) -> Bool {
        MLXModelCatalog.isRealtimeCapableModelRepo(repo)
    }

    nonisolated static func liveMode(for repo: String) -> MLXLiveMode {
        MLXModelCatalog.liveMode(for: repo)
    }

    nonisolated static func transcriptionBehavior(for _: String) -> TranscriptionBehavior {
        TranscriptionBehavior(
            correctionMode: .incremental,
            allowsQuickStopPass: true,
            preloadsOnRecordingStart: true
        )
    }
}

extension CustomLLMModelManager {
    enum ModelState: Equatable {
        case notDownloaded
        case downloading(
            progress: Double,
            completed: Int64,
            total: Int64,
            currentFile: String?,
            completedFiles: Int,
            totalFiles: Int
        )
        case paused(
            progress: Double,
            completed: Int64,
            total: Int64,
            currentFile: String?,
            completedFiles: Int,
            totalFiles: Int
        )
        case downloaded
        case error(String)
    }

    struct CatalogSnapshot: Equatable {
        let repo: String
        let isDownloaded: Bool
        let hasResumableDownload: Bool
        let state: ModelState
        let pausedStatusMessage: String?
        let hasActiveDownloadTask: Bool

        var isDownloading: Bool {
            if hasActiveDownloadTask {
                return true
            }
            if case .downloading = state {
                return true
            }
            return false
        }

        var isPaused: Bool {
            if case .paused = state {
                return true
            }
            return hasResumableDownload
        }
    }

    enum ModelSizeState: Equatable {
        case unknown
        case loading
        case ready(bytes: Int64, text: String)
        case error(String)
    }

    func displayTitle(for repo: String) -> String {
        CustomLLMModelCatalog.displayTitle(for: repo)
    }

    nonisolated static func ratingText(for repo: String) -> String {
        CustomLLMModelCatalog.ratingText(for: repo)
    }

    nonisolated static func catalogTagKeys(for repo: String) -> [String] {
        CustomLLMModelCatalog.catalogTagKeys(for: repo)
    }

    nonisolated static func fallbackRemoteSizeText(repo: String) -> String? {
        CustomLLMModelCatalog.fallbackRemoteSizeText(repo: repo)
    }

    nonisolated static func canonicalModelRepo(_ repo: String) -> String {
        CustomLLMModelCatalog.canonicalModelRepo(repo)
    }

    func displayModelsIncludingInstalled() -> [ModelOption] {
        // Catalog membership no longer depends on disk state.
        Self.availableModels
    }

    func description(for repo: String) -> String? {
        CustomLLMModelCatalog.description(for: repo)
    }

    nonisolated static func displayModels(including repo: String? = nil) -> [ModelOption] {
        CustomLLMModelCatalog.displayModels(including: repo)
    }

    nonisolated static func displayModels(includingInstalled repos: Set<String>) -> [ModelOption] {
        CustomLLMModelCatalog.displayModels(includingInstalled: repos)
    }

    static func isSupportedModelRepo(_ repo: String) -> Bool {
        CustomLLMModelCatalog.isSupportedModelRepo(repo)
    }
}
