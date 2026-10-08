// ModelSettingsObservation.swift
// Provides Model Settings Observation for model settings.

import SwiftUI
import Combine

@MainActor
extension ModelSettingsView {
    private var downloadLifecycleRefreshPublisher: AnyPublisher<Void, Never> {
        let mlx = Publishers.CombineLatest(
            mlxModelManager.$activeDownloadRepos.removeDuplicates(),
            mlxModelManager.$state
                .map(ModelSettingsManagerRefreshSupport.phase(for:))
                .removeDuplicates()
        )
        .map { _ in () }
        .eraseToAnyPublisher()

        let customLLM = Publishers.CombineLatest(
            customLLMManager.$activeDownloadRepos.removeDuplicates(),
            customLLMManager.$state
                .map(ModelSettingsManagerRefreshSupport.phase(for:))
                .removeDuplicates()
        )
        .map { _ in () }
        .eraseToAnyPublisher()

        let gguf = Publishers.CombineLatest(
            ggufTranslationModelManager.$activeDownloadModelID
                .map { $0?.rawValue }
                .removeDuplicates(),
            ggufTranslationModelManager.$stateByID
                .map(ModelSettingsManagerRefreshSupport.phase(for:))
                .removeDuplicates()
        )
        .map { _, _ in () }
        .eraseToAnyPublisher()

        return Publishers.Merge(
            Publishers.Merge(mlx, customLLM),
            gguf
        )
        .dropFirst()
        .debounce(for: .milliseconds(100), scheduler: RunLoop.main)
        .eraseToAnyPublisher()
    }

    private var downloadMetadataRefreshPublisher: AnyPublisher<Void, Never> {
        let mlx = mlxModelManager.$remoteSizeTextByRepo
            .removeDuplicates()
            .map { _ in () }
            .eraseToAnyPublisher()

        let mlxPauseMessage = mlxModelManager.$pausedStatusMessage
            .removeDuplicates()
            .map { _ in () }
            .eraseToAnyPublisher()

        let customLLM = customLLMManager.$remoteSizeTextByRepo
            .removeDuplicates()
            .map { _ in () }
            .eraseToAnyPublisher()

        let customLLMPauseMessage = Publishers.Merge(
            customLLMManager.$pausedStatusMessage
                .removeDuplicates()
                .map { _ in () }
                .eraseToAnyPublisher(),
            customLLMManager.$pausedStatusMessageByRepo
                .removeDuplicates()
                .map { _ in () }
                .eraseToAnyPublisher()
        )
        .eraseToAnyPublisher()

        let gguf = ggufTranslationModelManager.$stateByID
            .removeDuplicates()
            .map { _ in () }
            .eraseToAnyPublisher()

        let ggufPauseMessage = ggufTranslationModelManager.$pausedStatusMessageByID
            .removeDuplicates()
            .map { _ in () }
            .eraseToAnyPublisher()

        return Publishers.Merge(
            Publishers.Merge(
                Publishers.Merge(mlx, mlxPauseMessage),
                customLLM
            ),
            Publishers.Merge(
                customLLMPauseMessage,
                Publishers.Merge(gguf, ggufPauseMessage)
            )
        )
        .dropFirst()
        .debounce(for: .milliseconds(150), scheduler: RunLoop.main)
        .eraseToAnyPublisher()
    }

    private var appearanceObservedContent: some View {
        mainContent
            .onAppear(perform: handleOnAppear)
            .onAppear(perform: reloadCachedConfigurationState)
            .onAppear(perform: refreshModelStorageDisplayPath)
            .onAppear(perform: refreshCatalogSnapshot)
            .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
                refreshAppleIntelligenceAvailability()
            }
    }

    private var selectionObservedContent: some View {
        appearanceObservedContent
            .onChange(of: modelRepo) { _, newValue in
                handleModelRepoChange(newValue)
            }
            .onChange(of: localModelIdleUnloadDelaySeconds) { _, _ in
                handleLocalModelIdleUnloadDelayChange()
            }
            .onChange(of: customLLMRepo) { _, newValue in
                handleCustomLLMRepoChange(newValue)
            }
    }

    private var configurationObservedContent: some View {
        selectionObservedContent
            .onChange(of: translationModelProviderRaw) { _, _ in
                handleTranslationProviderChange()
            }
            .onChange(of: rewriteModelProviderRaw) { _, _ in
                handleRewriteProviderChange()
            }
            .onChange(of: remoteLLMProviderConfigurationsRaw) { _, _ in
                handleRemoteLLMConfigurationsChange()
            }
            .onChange(of: remoteASRProviderConfigurationsRaw) { _, _ in
                handleRemoteASRConfigurationsChange()
            }
            .onChange(of: modelStorageRootPath) { _, _ in
                handleModelStorageRootPathChange()
            }
            .onChange(of: featureSettingsRaw) { _, _ in
                handleFeatureSettingsChange()
            }
            .onChange(of: catalogTab) { _, _ in
                handleCatalogFilterSelectionChange()
            }
            .onChange(of: selectedTags) { _, _ in
                handleCatalogFilterSelectionChange()
            }
            .onChange(of: hiddenModelVisibilityIDsRaw) { _, _ in
                refreshCatalogSnapshot()
            }
            .onChange(of: isActive) { _, _ in
                handleModelSettingsVisibilityChange()
            }
            .onChange(of: mainWindowState.isVisible) { _, _ in
                handleModelSettingsVisibilityChange()
            }
    }

    private var stateObservedContent: some View {
        configurationObservedContent
            .onReceive(downloadLifecycleRefreshPublisher) { _ in
                handleImmediateDownloadLifecycleChange()
            }
            .onReceive(downloadMetadataRefreshPublisher) { _ in
                handleDownloadMetadataChange()
            }
    }

    private var installationRefreshPublisher: AnyPublisher<Void, Never> {
        ModelInstallationObservation.changes(mlx: mlxModelManager, customLLM: customLLMManager)
    }

    var contentWithLifecycle: some View {
        stateObservedContent
            .onReceive(installationRefreshPublisher) { _ in
                guard isActive, mainWindowState.isVisible else { return }
                refreshCatalogSnapshot()
            }
            .onReceive(modelStateRefreshTimer) { _ in
                handleModelStateRefreshTick()
            }
    }

    func handleModelRepoChange(_ newValue: String) {
        let canonicalRepo = MLXModelManager.canonicalModelRepo(newValue)
        if canonicalRepo != newValue {
            modelRepo = canonicalRepo
            return
        }
        mlxModelManager.updateModel(repo: canonicalRepo)
        refreshCatalogSnapshot()
    }

    func handleLocalModelIdleUnloadDelayChange() {
        let clamped = AppPreferenceKey.clampedLocalModelIdleUnloadDelaySeconds(localModelIdleUnloadDelaySeconds)
        if localModelIdleUnloadDelaySeconds != clamped {
            localModelIdleUnloadDelaySeconds = clamped
        }
        mlxModelManager.refreshMemoryOptimizationPolicy()
        customLLMManager.refreshMemoryOptimizationPolicy()
        AppDelegate.shared?.scheduleLLMIdleWarmupIfNeeded()
    }

    func handleCustomLLMRepoChange(_ newValue: String) {
        customLLMManager.updateModel(repo: newValue)
        ensureTranslationModelSelectionConsistency()
        ensureRewriteModelSelectionConsistency()
        refreshCatalogSnapshot()
        AppDelegate.shared?.scheduleLLMIdleWarmupIfNeeded()
        AppDelegate.shared?.prewarmLLMForCurrentActiveSessionIfNeeded()
    }

    func handleTranslationProviderChange() {
        syncTranslationFallbackProvider()
        ensureTranslationModelSelectionConsistency()
        refreshCatalogSnapshot()
    }

    func handleRewriteProviderChange() {
        ensureRewriteModelSelectionConsistency()
        refreshCatalogSnapshot()
    }

    func handleRemoteLLMConfigurationsChange() {
        cachedRemoteLLMConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: remoteLLMProviderConfigurationsRaw,
            sensitiveValueLoading: .metadataOnly
        )
        ensureTranslationModelSelectionConsistency()
        ensureRewriteModelSelectionConsistency()
        refreshCatalogSnapshot()
    }

    func handleRemoteASRConfigurationsChange() {
        cachedRemoteASRConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: remoteASRProviderConfigurationsRaw,
            sensitiveValueLoading: .metadataOnly
        )
        refreshCatalogSnapshot()
    }

    func handleModelStorageRootPathChange() {
        refreshAllModelStorageRoots()
        refreshModelStorageDisplayPath()
        refreshCatalogSnapshot()
    }

    func handleFeatureSettingsChange() {
        cachedFeatureSettings = FeatureSettingsStore.load(defaults: .standard)
        pruneSelectedTags()
        refreshCatalogSnapshot()
    }

    func handleCatalogFilterSelectionChange() {
        pruneSelectedTags()
        refreshCatalogSnapshot()
    }

    func handleModelSettingsVisibilityChange() {
        guard isActive else { return }
        guard mainWindowState.isVisible else { return }
        refreshModelInstallStateIfNeeded()
        pruneSelectedTags()
        refreshCatalogSnapshot()
    }

    func handleImmediateDownloadLifecycleChange() {
        guard ModelSettingsProgressRefreshSupport.shouldRefreshCatalogForLifecycleChange(
            isActive: isActive,
            isWindowVisible: mainWindowState.isVisible
        ) else {
            return
        }

        let token = ModelSettingsManagerRefreshSupport.downloadLifecycleToken(
            mlxState: mlxModelManager.state,
            mlxActiveDownloadRepos: mlxModelManager.activeDownloadRepos,
            customLLMState: customLLMManager.state,
            customLLMStateByRepo: customLLMManager.stateByRepo,
            customLLMActiveDownloadRepos: customLLMManager.activeDownloadRepos,
            ggufStateByID: ggufTranslationModelManager.stateByID,
            ggufActiveDownloadModelID: ggufTranslationModelManager.activeDownloadModelID
        )
        guard lastHandledDownloadLifecycleToken != token else {
            return
        }
        lastHandledDownloadLifecycleToken = token

        refreshModelInstallStateIfNeeded()
        pruneSelectedTags()
        refreshCatalogSnapshot()
    }

    func handleDownloadMetadataChange() {
        guard ModelSettingsProgressRefreshSupport.shouldRefreshCatalogForMetadataChange(
            isActive: isActive,
            isWindowVisible: mainWindowState.isVisible,
            shouldPollModelState: shouldPollModelState
        ) else {
            return
        }
        refreshCatalogSnapshot()
    }

    func handleModelStateRefreshTick() {
        guard isActive else { return }
        guard mainWindowState.isVisible else { return }
        guard shouldPollModelState else { return }
        refreshModelInstallStateIfNeeded()
        pruneSelectedTags()
        refreshCatalogSnapshot()
    }
}
