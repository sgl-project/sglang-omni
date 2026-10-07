// ModelSettingsLifecycle.swift
// Provides Model Settings Lifecycle for model settings.

import Foundation

extension ModelSettingsView {
    func refreshAppleIntelligenceAvailability() {
        let availability = AppleIntelligenceAvailability.current
        VoxtLog.info("Apple Intelligence availability: \(availability.logDescription)")
        AppDelegate.shared?.refreshTextEnhancerAvailability()
        appleIntelligenceRefreshRevision += 1
    }

    func refreshAllModelStorageRoots() {
        mlxModelManager.refreshStorageRoot()
        customLLMManager.refreshStorageRoot()
        ggufTranslationModelManager.refreshStorageRoot()
    }

    func handleOnAppear() {
        refreshAppleIntelligenceAvailability()
        let canonicalRepo = MLXModelManager.canonicalModelRepo(modelRepo)
        if canonicalRepo != modelRepo {
            modelRepo = canonicalRepo
        }
        mlxModelManager.updateModel(repo: canonicalRepo)
        let resolvedIdleUnloadDelay = AppPreferenceKey.resolvedLocalModelIdleUnloadDelaySeconds()
        if UserDefaults.standard.object(forKey: AppPreferenceKey.localModelIdleUnloadDelaySeconds) == nil
            || localModelIdleUnloadDelaySeconds != resolvedIdleUnloadDelay
        {
            localModelIdleUnloadDelaySeconds = resolvedIdleUnloadDelay
        }

        if customLLMRepo.isEmpty {
            customLLMRepo = CustomLLMModelManager.defaultModelRepo
        }
        if !CustomLLMModelManager.isSupportedModelRepo(customLLMRepo) {
            customLLMRepo = CustomLLMModelManager.defaultModelRepo
        }
        if translationCustomLLMRepo.isEmpty {
            translationCustomLLMRepo = customLLMRepo
        }
        if !CustomLLMModelManager.isSupportedModelRepo(translationCustomLLMRepo) {
            translationCustomLLMRepo = customLLMRepo
        }
        translationGGUFModelIDRaw = GGUFTranslationModelCatalog.resolvedModelID(translationGGUFModelIDRaw).rawValue
        ggufTranslationModelManager.updateModel(
            id: GGUFTranslationModelCatalog.resolvedModelID(translationGGUFModelIDRaw)
        )
        if !TranslationModelProvider.allCases.contains(where: { $0.rawValue == translationModelProviderRaw }) {
            translationModelProviderRaw = TranslationModelProvider.customLLM.rawValue
        }
        if !TranslationModelProvider.allCases.contains(where: { $0.rawValue == translationFallbackModelProviderRaw }) {
            translationFallbackModelProviderRaw = TranslationModelProvider.customLLM.rawValue
        }
        if !RewriteModelProvider.allCases.contains(where: { $0.rawValue == rewriteModelProviderRaw }) {
            rewriteModelProviderRaw = RewriteModelProvider.customLLM.rawValue
        }
        if rewriteCustomLLMRepo.isEmpty {
            rewriteCustomLLMRepo = customLLMRepo
        }
        if !CustomLLMModelManager.isSupportedModelRepo(rewriteCustomLLMRepo) {
            rewriteCustomLLMRepo = customLLMRepo
        }
        customLLMManager.updateModel(repo: customLLMRepo)
        if !RemoteASRProvider.allCases.contains(where: { $0.rawValue == remoteASRSelectedProviderRaw }) {
            remoteASRSelectedProviderRaw = RemoteASRProvider.openAIWhisper.rawValue
        }
        if !RemoteLLMProvider.allCases.contains(where: { $0.rawValue == remoteLLMSelectedProviderRaw }) {
            remoteLLMSelectedProviderRaw = RemoteLLMProvider.openAI.rawValue
        }
        syncTranslationFallbackProvider()
        ensureTranslationModelSelectionConsistency()
        ensureRewriteModelSelectionConsistency()
        mlxModelManager.refreshMemoryOptimizationPolicy()
        customLLMManager.refreshMemoryOptimizationPolicy()
        ggufTranslationModelManager.refreshStorageRoot()
        DispatchQueue.main.async {
            refreshModelInstallStateIfNeeded()
        }
    }

    func syncTranslationFallbackProvider() {
        let currentProvider = TranslationModelProvider.resolved(rawValue: translationModelProviderRaw)

        if translationFallbackModelProviderRaw != currentProvider.rawValue {
            translationFallbackModelProviderRaw = currentProvider.rawValue
        }
    }
}
