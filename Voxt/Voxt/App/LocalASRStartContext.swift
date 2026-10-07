// LocalASRStartContext.swift
// Provides Local ASRStart Context for app lifecycle and routing.

import Foundation

extension AppDelegate {
    struct LocalASRStartContext {
        let selectedMLXRepo: String
        let activeMLXDownloadRepo: String?
        let isSelectedMLXModelDownloaded: Bool
        let mlxModelState: MLXModelManager.ModelState
    }

    func currentLocalASRStartContext() -> LocalASRStartContext {
        let selectedMLXRepo = mlxModelManager.currentModelRepo
        return LocalASRStartContext(
            selectedMLXRepo: selectedMLXRepo,
            activeMLXDownloadRepo: mlxModelManager.isDownloadOperationActive(repo: selectedMLXRepo)
                ? selectedMLXRepo
                : nil,
            isSelectedMLXModelDownloaded: mlxModelManager.isModelDownloaded(repo: selectedMLXRepo),
            mlxModelState: mlxModelManager.state
        )
    }
}
