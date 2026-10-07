// ModelManagerRefreshSupport.swift
// Provides Model Manager Refresh Support for model settings.

import Foundation

enum ModelSettingsManagerActivityPhase: Equatable {
    case idle
    case downloading
    case paused
    case loading
    case downloaded
    case error
}

struct ModelSettingsDownloadLifecycleToken: Equatable {
    let mlxPhase: ModelSettingsManagerActivityPhase
    let mlxActiveDownloadRepos: [String]
    let customLLMPhase: ModelSettingsManagerActivityPhase
    let customLLMActiveDownloadRepos: [String]
    let ggufPhase: ModelSettingsManagerActivityPhase
    let ggufActiveDownloadModelID: String?
}

enum ModelSettingsManagerRefreshSupport {
    static func phase(for state: MLXModelManager.ModelState) -> ModelSettingsManagerActivityPhase {
        switch state {
        case .notDownloaded:
            return .idle
        case .downloading:
            return .downloading
        case .paused:
            return .paused
        case .downloaded:
            return .downloaded
        case .loading:
            return .loading
        case .ready:
            return .downloaded
        case .error:
            return .error
        }
    }

    static func phase(for state: CustomLLMModelManager.ModelState) -> ModelSettingsManagerActivityPhase {
        switch state {
        case .notDownloaded:
            return .idle
        case .downloading:
            return .downloading
        case .paused:
            return .paused
        case .downloaded:
            return .downloaded
        case .error:
            return .error
        }
    }

    static func phase(for stateByRepo: [String: CustomLLMModelManager.ModelState]) -> ModelSettingsManagerActivityPhase {
        if stateByRepo.values.contains(where: {
            if case .downloading = $0 { return true }
            return false
        }) {
            return .downloading
        }

        if stateByRepo.values.contains(where: {
            if case .paused = $0 { return true }
            return false
        }) {
            return .paused
        }

        if stateByRepo.values.contains(where: {
            if case .error = $0 { return true }
            return false
        }) {
            return .error
        }

        if stateByRepo.values.contains(.downloaded) {
            return .downloaded
        }

        return .idle
    }

    static func phase(for state: GGUFTranslationModelManager.ModelState) -> ModelSettingsManagerActivityPhase {
        switch state {
        case .notDownloaded:
            return .idle
        case .downloading:
            return .downloading
        case .paused:
            return .paused
        case .downloaded:
            return .downloaded
        case .error:
            return .error
        }
    }

    static func phase(for stateByID: [GGUFTranslationModelID: GGUFTranslationModelManager.ModelState]) -> ModelSettingsManagerActivityPhase {
        if stateByID.values.contains(where: {
            if case .downloading = $0 { return true }
            return false
        }) {
            return .downloading
        }

        if stateByID.values.contains(where: {
            if case .paused = $0 { return true }
            return false
        }) {
            return .paused
        }

        if stateByID.values.contains(where: {
            if case .error = $0 { return true }
            return false
        }) {
            return .error
        }

        if stateByID.values.contains(.downloaded) {
            return .downloaded
        }

        return .idle
    }

    static func downloadLifecycleToken(
        mlxState: MLXModelManager.ModelState,
        mlxActiveDownloadRepos: Set<String>,
        customLLMState: CustomLLMModelManager.ModelState,
        customLLMStateByRepo: [String: CustomLLMModelManager.ModelState],
        customLLMActiveDownloadRepos: Set<String>,
        ggufStateByID: [GGUFTranslationModelID: GGUFTranslationModelManager.ModelState],
        ggufActiveDownloadModelID: GGUFTranslationModelID?
    ) -> ModelSettingsDownloadLifecycleToken {
        let customLLMPhase = customLLMActiveDownloadRepos.isEmpty
            ? phase(for: customLLMStateByRepo)
            : .downloading
        return ModelSettingsDownloadLifecycleToken(
            mlxPhase: phase(for: mlxState),
            mlxActiveDownloadRepos: mlxActiveDownloadRepos.sorted(),
            customLLMPhase: customLLMPhase == .idle ? phase(for: customLLMState) : customLLMPhase,
            customLLMActiveDownloadRepos: customLLMActiveDownloadRepos.sorted(),
            ggufPhase: phase(for: ggufStateByID),
            ggufActiveDownloadModelID: ggufActiveDownloadModelID?.rawValue
        )
    }
}
