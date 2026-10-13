// ModelInstallSnapshots.swift
// Provides Model Install Snapshots for model settings.

import SwiftUI

@MainActor
extension ModelSettingsView {
    func isCancellationPending(for target: LocalModelInstallTarget) -> Bool {
        cancellingInstallTargets.contains(target)
    }

    func reconcileCancellingInstallTargets() {
        let retained = cancellingInstallTargets.filter(isCancellationStillPending(for:))
        if retained.count != cancellingInstallTargets.count {
            cancellingInstallTargets = Set(retained)
        }
    }

    func mlxInstallSnapshot(for repo: String) -> LocalModelInstallSnapshot {
        let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
        let catalogSnapshot = mlxModelManager.catalogSnapshot(for: canonicalRepo)
        let isUninstalling = isUninstallingModel(canonicalRepo)
        let target = LocalModelInstallTarget.mlx(canonicalRepo)
        let state: LocalModelInstallState

        if isUninstalling {
            state = .uninstalling
        } else if isCancellationPending(for: target) {
            state = .cancelling
        } else if catalogSnapshot.isDownloading {
            state = .downloading
        } else if catalogSnapshot.isPaused {
            state = .paused
        } else if catalogSnapshot.isDownloaded {
            state = .installed
        } else if mlxModelManager.isCheckingInstallation(repo: canonicalRepo) {
            state = .checking
        } else {
            state = .installable(isEnabled: true)
        }

        return LocalModelInstallSnapshot(
            target: target,
            state: state,
            isInstalled: catalogSnapshot.isDownloaded,
            isCurrentSelection: isCurrentModel(canonicalRepo),
            statusText: mlxInstallStatusText(
                for: catalogSnapshot,
                isUninstalling: isUninstalling,
                isCancelling: isCancellationPending(for: target)
            ),
            badgeText: hasIssue(for: .mlxModel(canonicalRepo)) ? AppLocalization.localizedString("Needs Setup") : nil,
            downloadStatus: isCancellationPending(for: target) ? nil : ModelDownloadStatusSnapshot.fromMLXState(
                catalogSnapshot.state,
                pauseMessage: catalogSnapshot.pausedStatusMessage
            ),
            canOpenLocation: catalogSnapshot.isDownloaded && !isUninstalling && !isCancellationPending(for: target),
            canConfigure: !isUninstalling && !isCancellationPending(for: target),
            configureActionTitle: AppLocalization.localizedString("Settings")
        )
    }

    func customLLMInstallSnapshot(for repo: String) -> LocalModelInstallSnapshot {
        let canonicalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
        let catalogSnapshot = customLLMManager.catalogSnapshot(for: canonicalRepo)
        let isInstalled = catalogSnapshot.isDownloaded
        let isUninstalling = isUninstallingCustomLLM(canonicalRepo)
        let target = LocalModelInstallTarget.customLLM(canonicalRepo)
        let isDownloading = catalogSnapshot.isDownloading
        let isPaused = catalogSnapshot.isPaused

        let state: LocalModelInstallState
        if isUninstalling {
            state = .uninstalling
        } else if isCancellationPending(for: target) {
            state = .cancelling
        } else if isDownloading {
            state = .downloading
        } else if isPaused {
            state = .paused
        } else if isInstalled {
            state = .installed
        } else if customLLMManager.isCheckingInstallation(repo: canonicalRepo) {
            state = .checking
        } else {
            state = .installable(isEnabled: true)
        }

        let resolvedDownloadStatus: ModelDownloadStatusSnapshot?
        switch catalogSnapshot.state {
        case .downloading, .paused:
            resolvedDownloadStatus = ModelDownloadStatusSnapshot.fromCustomLLMState(
                catalogSnapshot.state,
                pauseMessage: catalogSnapshot.pausedStatusMessage
            )
        default:
            resolvedDownloadStatus = nil
        }

        return LocalModelInstallSnapshot(
            target: target,
            state: state,
            isInstalled: isInstalled,
            isCurrentSelection: isCurrentCustomLLM(canonicalRepo),
            statusText: customLLMInstallStatusText(
                isUninstalling: isUninstalling,
                isCancelling: isCancellationPending(for: target),
                isDownloading: isDownloading,
                isPaused: isPaused,
                state: catalogSnapshot.state,
                pauseMessage: catalogSnapshot.pausedStatusMessage
            ),
            badgeText: customLLMBadgeText(for: canonicalRepo),
            downloadStatus: isCancellationPending(for: target) ? nil : resolvedDownloadStatus,
            canOpenLocation: isInstalled && !isUninstalling && !isCancellationPending(for: target),
            canConfigure: isInstalled && !isUninstalling && !isCancellationPending(for: target),
            configureActionTitle: AppLocalization.localizedString("Configure")
        )
    }

    func ggufTranslationInstallSnapshot(for modelID: GGUFTranslationModelID) -> LocalModelInstallSnapshot {
        let modelState = ggufTranslationModelManager.state(for: modelID)
        let isInstalled = ggufTranslationModelManager.isModelDownloaded(id: modelID)
        let isUninstalling = isUninstallingGGUFTranslationModel(modelID)
        let target = LocalModelInstallTarget.ggufTranslation(modelID)
        let pauseMessage = ggufTranslationModelManager.pausedStatusMessage(for: modelID)
            ?? (ggufTranslationModelManager.hasResumableDownload(id: modelID)
                ? AppLocalization.localizedString("Paused. Ready to continue.")
                : nil)

        let state: LocalModelInstallState
        if isUninstalling {
            state = .uninstalling
        } else if isCancellationPending(for: target) {
            state = .cancelling
        } else {
            switch modelState {
            case .downloading:
                state = .downloading
            case .paused:
                state = .paused
            case .downloaded:
                state = .installed
            case .notDownloaded, .error:
                state = ggufTranslationModelManager.isCheckingInstallation(id: modelID)
                    ? .checking
                    : .installable(isEnabled: ggufTranslationModelManager.activeDownloadModelID == nil)
            }
        }

        return LocalModelInstallSnapshot(
            target: target,
            state: state,
            isInstalled: isInstalled,
            isCurrentSelection: ggufTranslationModelManager.selectedModelID == modelID,
            statusText: ggufTranslationInstallStatusText(
                state: modelState,
                pauseMessage: pauseMessage,
                isUninstalling: isUninstalling,
                isCancelling: isCancellationPending(for: target)
            ),
            badgeText: ggufTranslationModelManager.option(for: modelID).badgeText,
            downloadStatus: isCancellationPending(for: target)
                ? nil
                : ModelDownloadStatusSnapshot.fromGGUFState(
                    modelState,
                    pauseMessage: pauseMessage
                ),
            canOpenLocation: isInstalled && !isUninstalling && !isCancellationPending(for: target),
            canConfigure: false,
            configureActionTitle: nil
        )
    }

    func performInstallAction(_ target: LocalModelInstallTarget, kind: LocalModelInstallActionKind) {
        switch (target, kind) {
        case (.mlx(let repo), .use):
            useModel(repo)
        case (.mlx(let repo), .install), (.mlx(let repo), .resume):
            downloadModel(repo)
        case (.mlx(let repo), .pause):
            mlxModelManager.pauseDownload(repo: repo)
            refreshCatalogSnapshot()
        case (.mlx(let repo), .cancel):
            cancellingInstallTargets.insert(.mlx(repo))
            mlxModelManager.cancelDownload(repo: repo)
            refreshCatalogSnapshot()
        case (.mlx(let repo), .uninstall):
            requestDeleteModel(repo)
        case (.mlx(let repo), .openLocation):
            openMLXModelDirectory(repo)
        case (.mlx(let repo), .configure):
            activeLocalASRConfigurationTarget = .mlx(repo: repo)

        case (.customLLM(let repo), .use):
            useCustomLLM(repo)
        case (.customLLM(let repo), .install), (.customLLM(let repo), .resume):
            downloadCustomLLM(repo)
        case (.customLLM(let repo), .pause):
            customLLMManager.pauseDownload(repo: repo)
            refreshCatalogSnapshot()
        case (.customLLM(let repo), .cancel):
            cancellingInstallTargets.insert(.customLLM(repo))
            customLLMManager.cancelDownload(repo: repo)
            refreshCatalogSnapshot()
        case (.customLLM(let repo), .uninstall):
            requestDeleteCustomLLM(repo)
        case (.customLLM(let repo), .openLocation):
            openCustomLLMModelDirectory(repo)
        case (.customLLM(let repo), .configure):
            customLLMConfigurationRepo = repo
            isCustomLLMConfigurationPresented = true

        case (.ggufTranslation(let modelID), .use):
            useGGUFTranslationModel(modelID)
        case (.ggufTranslation(let modelID), .install), (.ggufTranslation(let modelID), .resume):
            downloadGGUFTranslationModel(modelID)
        case (.ggufTranslation(let modelID), .pause):
            ggufTranslationModelManager.pauseDownload(id: modelID)
            refreshCatalogSnapshot()
        case (.ggufTranslation(let modelID), .cancel):
            cancellingInstallTargets.insert(.ggufTranslation(modelID))
            ggufTranslationModelManager.cancelDownload(id: modelID)
            refreshCatalogSnapshot()
        case (.ggufTranslation(let modelID), .uninstall):
            requestDeleteGGUFTranslationModel(modelID)
        case (.ggufTranslation(let modelID), .openLocation):
            openGGUFTranslationModelDirectory(modelID)
        case (.ggufTranslation, .configure):
            break

        case (_, .inactive):
            break
        }
    }

    func modelTableRow(
        id: String,
        title: String,
        snapshot: LocalModelInstallSnapshot
    ) -> ModelTableRow {
        ModelTableRow(
            id: id,
            title: title,
            isActive: snapshot.isCurrentSelection,
            status: snapshot.statusText,
            badgeText: snapshot.badgeText,
            isTitleUnderlined: snapshot.canOpenLocation,
            onTapTitle: snapshot.canOpenLocation ? {
                performInstallAction(snapshot.target, kind: .openLocation)
            } : nil,
            actions: ModelSettingsInstallActionResolver.tableActions(
                for: snapshot,
                perform: performInstallAction(_:kind:)
            )
        )
    }

    private func mlxInstallStatusText(
        for snapshot: MLXModelManager.CatalogSnapshot,
        isUninstalling: Bool,
        isCancelling: Bool
    ) -> String {
        if isUninstalling {
            return AppLocalization.localizedString("Uninstalling…")
        }

        if isCancelling {
            return AppLocalization.localizedString("Cancelling…")
        }

        if case .downloading(_, let completed, let total, _, _, _) = snapshot.state {
            return ModelDownloadPresentationSupport.statusText(
                downloadState: .downloading(completed: completed, total: total)
            )
        }

        if case .paused(_, let completed, let total, _, _, _) = snapshot.state {
            return ModelDownloadPresentationSupport.statusText(
                downloadState: .paused(
                    completed: completed,
                    total: total,
                    pauseMessage: snapshot.pausedStatusMessage
                )
            )
        }

        if snapshot.isPaused {
            return ModelDownloadPresentationSupport.statusText(
                downloadState: .paused(
                    completed: 0,
                    total: 0,
                    pauseMessage: AppLocalization.localizedString("Paused. Ready to continue.")
                )
            )
        }

        if case .error(let message) = snapshot.state {
            return ModelDownloadPresentationSupport.statusText(
                downloadState: .idle,
                errorMessage: message
            )
        }

        return ""
    }

    private func customLLMInstallStatusText(
        isUninstalling: Bool,
        isCancelling: Bool,
        isDownloading: Bool,
        isPaused: Bool,
        state: CustomLLMModelManager.ModelState,
        pauseMessage: String?
    ) -> String {
        if isUninstalling {
            return AppLocalization.localizedString("Uninstalling…")
        }

        if isCancelling {
            return AppLocalization.localizedString("Cancelling…")
        }

        if isDownloading,
           case .downloading(_, let completed, let total, _, _, _) = state {
            return ModelDownloadPresentationSupport.statusText(
                downloadState: .downloading(completed: completed, total: total)
            )
        }

        if isPaused,
           case .paused(_, let completed, let total, _, _, _) = state {
            return ModelDownloadPresentationSupport.statusText(
                downloadState: .paused(
                    completed: completed,
                    total: total,
                    pauseMessage: pauseMessage
                )
            )
        }

        if isPaused {
            return ModelDownloadPresentationSupport.statusText(
                downloadState: .paused(
                    completed: 0,
                    total: 0,
                    pauseMessage: AppLocalization.localizedString("Paused. Ready to continue.")
                )
            )
        }

        if case .error(let message) = state {
            return ModelDownloadPresentationSupport.statusText(
                downloadState: .idle,
                errorMessage: message
            )
        }
        return ""
    }

    private func isCancellationStillPending(for target: LocalModelInstallTarget) -> Bool {
        switch target {
        case .mlx(let repo):
            let snapshot = mlxModelManager.catalogSnapshot(for: repo)
            return snapshot.isDownloading || snapshot.isPaused
        case .customLLM(let repo):
            let snapshot = customLLMManager.catalogSnapshot(for: repo)
            return snapshot.isDownloading || snapshot.isPaused
        case .ggufTranslation(let modelID):
            let state = ggufTranslationModelManager.state(for: modelID)
            if case .downloading = state {
                return true
            }
            if case .paused = state {
                return true
            }
            return false
        }
    }

    private func ggufTranslationInstallStatusText(
        state: GGUFTranslationModelManager.ModelState,
        pauseMessage: String?,
        isUninstalling: Bool,
        isCancelling: Bool
    ) -> String {
        if isUninstalling {
            return AppLocalization.localizedString("Uninstalling…")
        }

        if isCancelling {
            return AppLocalization.localizedString("Cancelling…")
        }

        switch state {
        case .downloading(_, let completed, let total, _, _, _):
            return ModelDownloadPresentationSupport.statusText(
                downloadState: .downloading(completed: completed, total: total)
            )
        case .paused(_, let completed, let total, _, _, _):
            return ModelDownloadPresentationSupport.statusText(
                downloadState: .paused(
                    completed: completed,
                    total: total,
                    pauseMessage: pauseMessage
                )
            )
        case .error(let message):
            return ModelDownloadPresentationSupport.statusText(
                downloadState: .idle,
                errorMessage: message
            )
        case .notDownloaded, .downloaded:
            return ""
        }
    }

}
