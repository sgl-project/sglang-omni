// ModelSettingsActions.swift
// Provides Model Settings Actions for model settings.

import AppKit
import SwiftUI

enum MLXConfigurationSummarySupport {
    static func summary(for repo: String, tuning: MLXLocalTuningSettings) -> String {
        let capability = MLXModelCatalog.capability(for: repo)
        let family = capability.family
        let usesRecognitionPreset = capability.configurationCapabilities.contains(.recognitionPreset)
        switch family {
        case .whisper:
            return AppLocalization.format("Temp %.2f", tuning.whisperTemperature)
        case .qwen3ASR:
            let hasContext = tuning.qwenContextBias.isEmpty
                ? AppLocalization.localizedString("Context Off")
                : AppLocalization.localizedString("Context On")
            if usesRecognitionPreset {
                return AppLocalization.format("%@ · %@", tuning.preset.title, hasContext)
            }
            return hasContext
        case .senseVoice:
            return AppLocalization.localizedString(tuning.senseVoiceUseITN ? "ITN On" : "ITN Off")
        case .mossTranscribeDiarize:
            return AppLocalization.format(
                "%@: %@ · %@: %@",
                AppLocalization.localizedString("Dictation Settings"),
                tuning.mossOutputMode.title,
                AppLocalization.localizedString("Meeting"),
                AppLocalization.localizedString("Structured Meeting Segments")
            )
        case .cohereTranscribe:
            return AppLocalization.format(
                "%@ · Temp %.2f",
                tuning.cohereLongFormStrategy.title,
                tuning.cohereTemperature
            )
        case .nemotronASR:
            return tuning.nemotronStreamLatency.title
        case .parakeet:
            return AppLocalization.localizedString("Checkpoint Defaults")
        case .generic:
            return tuning.preset.title
        }
    }
}

extension ModelSettingsView {
    func promptBinding(for storage: Binding<String>, kind: AppPromptKind) -> Binding<String> {
        Binding(
            get: {
                AppPromptDefaults.resolvedStoredText(storage.wrappedValue, kind: kind)
            },
            set: { newValue in
                storage.wrappedValue = AppPromptDefaults.canonicalStoredText(newValue, kind: kind)
            }
        )
    }

    var remoteASRRows: [ModelTableRow] {
        RemoteASRProvider.allCases.map { provider in
            let config = RemoteModelConfigurationStore.resolvedASRConfiguration(
                provider: provider,
                stored: remoteASRConfigurations
            )
            let isSelected = selectedRemoteASRProvider == provider
            let status = remoteASRStatusText(
                for: provider,
                configuration: config
            )
            return ModelTableRow(
                id: provider.rawValue,
                title: provider.title,
                isActive: isSelected,
                status: status,
                badgeText: hasIssue(for: .remoteASRProvider(provider)) ? AppLocalization.localizedString("Needs Setup") : nil,
                actions: [
                    ModelTableAction(
                        title: selectionActionTitle(isSelected: isSelected),
                        isEnabled: !isSelected
                    ) {
                        useRemoteASRProvider(provider)
                    },
                    ModelTableAction(title: AppLocalization.localizedString("Configure")) {
                        editingASRProvider = provider
                    }
                ]
            )
        }
    }

    var remoteLLMRows: [ModelTableRow] {
        RemoteLLMProvider.allCases.map { provider in
            let isConfigured = RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: provider,
                stored: remoteLLMConfigurations
            )
            let config = RemoteModelConfigurationStore.resolvedLLMConfiguration(
                provider: provider,
                stored: remoteLLMConfigurations
            )
            let status = isConfigured
                ? AppLocalization.format("Configured model: %@", config.model)
                : AppLocalization.localizedString("Not configured")
            return ModelTableRow(
                id: provider.rawValue,
                title: provider.title,
                isActive: selectedRemoteLLMProvider == provider,
                status: status,
                badgeText: remoteLLMBadgeText(for: provider),
                actions: [
                    ModelTableAction(
                        title: selectionActionTitle(isSelected: selectedRemoteLLMProvider == provider),
                        isEnabled: selectedRemoteLLMProvider != provider
                    ) {
                        useRemoteLLMProvider(provider)
                    },
                    ModelTableAction(title: AppLocalization.localizedString("Configure")) {
                        editingLLMProvider = provider
                    }
                ]
            )
        }
    }

    var mlxRows: [ModelTableRow] {
        mlxModelManager.displayModelsIncludingInstalled().map { model in
            let snapshot = mlxInstallSnapshot(for: model.id)
            return modelTableRow(
                id: model.id,
                title: model.title,
                snapshot: snapshot
            )
        }
    }

    var customLLMRows: [ModelTableRow] {
        customLLMManager.displayModelsIncludingInstalled().map { model in
            let snapshot = customLLMInstallSnapshot(for: model.id)
            return modelTableRow(
                id: model.id,
                title: model.title,
                snapshot: snapshot
            )
        }
    }

    func hasIssue(for scope: ModelConfigurationIssue.Scope) -> Bool {
        missingConfigurationIssues.contains(where: { $0.scope == scope })
    }

    func remoteLLMBadgeText(for provider: RemoteLLMProvider) -> String? {
        let scopes: [ModelConfigurationIssue.Scope] = [
            .remoteLLMProvider(provider),
            .translationRemoteLLM(provider),
            .rewriteRemoteLLM(provider)
        ]
        return missingConfigurationIssues.contains(where: { scopes.contains($0.scope) }) ? AppLocalization.localizedString("Needs Setup") : nil
    }

    func customLLMBadgeText(for repo: String) -> String? {
        let scopes: [ModelConfigurationIssue.Scope] = [
            .customLLMModel(repo),
            .translationCustomLLM(repo),
            .rewriteCustomLLM(repo)
        ]
        if missingConfigurationIssues.contains(where: { scopes.contains($0.scope) }) {
            return AppLocalization.localizedString("Needs Setup")
        }

        return nil
    }

    private func selectionActionTitle(isSelected: Bool) -> String {
        AppLocalization.localizedString(isSelected ? "Using" : "Use")
    }

    func useModel(_ repo: String) {
        let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
        modelRepo = canonicalRepo
        mlxModelManager.updateModel(repo: canonicalRepo)
    }

    func downloadModel(_ repo: String) {
        clearPresentedModelError(targetID: "mlx:\(MLXModelManager.canonicalModelRepo(repo))")
        Task {
            await mlxModelManager.downloadModel(repo: repo)
            if case .error(let message) = mlxModelManager.state(for: repo) {
                presentModelError(
                    targetID: "mlx:\(MLXModelManager.canonicalModelRepo(repo))",
                    modelName: mlxModelManager.displayTitle(for: repo),
                    message: message
                )
            }
        }
    }

    func deleteModel(_ repo: String) async -> Result<Void, Error> {
        let result = await mlxModelManager.deleteModel(repo: repo)
        if MLXModelManager.canonicalModelRepo(repo) == MLXModelManager.canonicalModelRepo(modelRepo) {
            mlxModelManager.checkExistingModel()
        }
        return result
    }

    func isCurrentModel(_ repo: String) -> Bool {
        MLXModelManager.canonicalModelRepo(repo) == MLXModelManager.canonicalModelRepo(modelRepo)
    }

    func isDownloadingModel(_ repo: String) -> Bool {
        mlxInstallSnapshot(for: repo).state == .downloading
    }

    func isPausedModel(_ repo: String) -> Bool {
        mlxInstallSnapshot(for: repo).state == .paused
    }

    func modelStatusText(for repo: String) -> String {
        mlxInstallSnapshot(for: repo).statusText
    }

    func useCustomLLM(_ repo: String) {
        customLLMRepo = repo
        customLLMManager.updateModel(repo: repo)
    }

    func downloadCustomLLM(_ repo: String) {
        clearPresentedModelError(targetID: "custom-llm:\(CustomLLMModelManager.canonicalModelRepo(repo))")
        Task {
            await customLLMManager.downloadModel(repo: repo)
            if case .error(let message) = customLLMManager.state(for: repo) {
                presentModelError(
                    targetID: "custom-llm:\(CustomLLMModelManager.canonicalModelRepo(repo))",
                    modelName: customLLMManager.displayTitle(for: repo),
                    message: message
                )
            }
        }
    }

    func deleteCustomLLM(_ repo: String) async -> Result<Void, Error> {
        let result = await customLLMManager.deleteModel(repo: repo)
        if repo == customLLMRepo {
            customLLMManager.checkExistingModel()
        }
        return result
    }

    func requestDeleteModel(_ repo: String) {
        pendingModelRemovalTarget = .mlx(repo: repo)
    }

    func requestDeleteCustomLLM(_ repo: String) {
        pendingModelRemovalTarget = .customLLM(repo: repo)
    }

    func requestDeleteGGUFTranslationModel(_ modelID: GGUFTranslationModelID) {
        pendingModelRemovalTarget = .ggufTranslation(modelID: modelID)
    }

    func confirmDeleteModel(_ target: LocalModelRemovalTarget) {
        pendingModelRemovalTarget = nil
        uninstallingModelTarget = target

        Task { @MainActor in
            await Task.yield()
            clearPresentedModelError(targetID: target.id)
            let result: Result<Void, Error>
            switch target {
            case .mlx(let repo):
                result = await deleteModel(repo)
            case .customLLM(let repo):
                result = await deleteCustomLLM(repo)
            case .ggufTranslation(let modelID):
                result = await deleteGGUFTranslationModel(modelID)
            }
            switch result {
            case .success:
                showModelOperationToast(
                    AppLocalization.format("Uninstalled %@.", modelDisplayName(for: target))
                )
            case .failure(let error):
                presentModelError(
                    targetID: target.id,
                    modelName: modelDisplayName(for: target),
                    message: AppLocalization.format(
                        "Uninstall failed: %@",
                        error.localizedDescription
                    )
                )
            }
            uninstallingModelTarget = nil
            refreshCatalogSnapshot()
        }
    }

    func isUninstallingModel(_ repo: String) -> Bool {
        guard case .mlx(let uninstallingRepo) = uninstallingModelTarget else { return false }
        return MLXModelManager.canonicalModelRepo(uninstallingRepo) == MLXModelManager.canonicalModelRepo(repo)
    }

    func isUninstallingCustomLLM(_ repo: String) -> Bool {
        guard case .customLLM(let uninstallingRepo) = uninstallingModelTarget else { return false }
        return CustomLLMModelManager.canonicalModelRepo(uninstallingRepo) == CustomLLMModelManager.canonicalModelRepo(repo)
    }

    func isUninstallingGGUFTranslationModel(_ modelID: GGUFTranslationModelID) -> Bool {
        guard case .ggufTranslation(let uninstallingModelID) = uninstallingModelTarget else { return false }
        return uninstallingModelID == modelID
    }

    func uninstallConfirmationMessage(for target: LocalModelRemovalTarget) -> String {
        let modelName = modelDisplayName(for: target)
        return AppLocalization.format(
            "Uninstall %@ from this Mac? You can download it again later.",
            modelName
        )
    }

    func modelDisplayName(for target: LocalModelRemovalTarget) -> String {
        switch target {
        case .mlx(let repo):
            return mlxModelManager.displayTitle(for: repo)
        case .customLLM(let repo):
            return customLLMManager.displayTitle(for: repo)
        case .ggufTranslation(let modelID):
            return ggufTranslationModelManager.displayTitle(for: modelID)
        }
    }

    func isCurrentCustomLLM(_ repo: String) -> Bool {
        CustomLLMModelManager.canonicalModelRepo(repo) == CustomLLMModelManager.canonicalModelRepo(customLLMRepo)
    }

    func isDownloadingCustomLLM(_ repo: String) -> Bool {
        customLLMManager.isDownloading(repo: repo)
    }

    func isPausedCustomLLM(_ repo: String) -> Bool {
        customLLMManager.isPaused(repo: repo)
    }

    func customLLMStatusText(for repo: String) -> String {
        customLLMInstallSnapshot(for: repo).statusText
    }

    func useGGUFTranslationModel(_ modelID: GGUFTranslationModelID) {
        translationGGUFModelIDRaw = modelID.rawValue
        ggufTranslationModelManager.updateModel(id: modelID)
    }

    func downloadGGUFTranslationModel(_ modelID: GGUFTranslationModelID) {
        ggufTranslationModelManager.downloadModel(id: modelID)
        refreshCatalogSnapshot()
    }

    func cancelGGUFTranslationDownload(_ modelID: GGUFTranslationModelID) {
        ggufTranslationModelManager.cancelDownload(id: modelID)
        refreshCatalogSnapshot()
    }

    func deleteGGUFTranslationModel(_ modelID: GGUFTranslationModelID) async -> Result<Void, Error> {
        let result = await ggufTranslationModelManager.deleteModel(id: modelID)
        refreshCatalogSnapshot()
        return result
    }

    func openGGUFTranslationModelDirectory(_ modelID: GGUFTranslationModelID) {
        ggufTranslationModelManager.openModelDirectory(id: modelID)
    }

    func useRemoteASRProvider(_ provider: RemoteASRProvider) {
        let resolved = RemoteModelConfigurationStore.resolvedASRConfiguration(
            provider: provider,
            from: remoteASRProviderConfigurationsRaw
        )
        switch saveRemoteASRConfiguration(resolved) {
        case .success:
            remoteASRSelectedProviderRaw = provider.rawValue
        case .failure(let error):
            showModelOperationToast(error.localizedDescription)
        }
    }

    func saveRemoteASRConfiguration(
        _ configuration: RemoteProviderConfiguration
    ) -> Result<Void, RemoteModelConfigurationStore.SaveError> {
        let result = RemoteModelConfigurationStore.saveConfiguration(
            configuration,
            updating: remoteASRProviderConfigurationsRaw
        )
        switch result {
        case .success(let raw):
            remoteASRProviderConfigurationsRaw = raw
            NotificationCenter.default.post(name: .voxtRemoteProviderConfigurationsDidChange, object: nil)
            return .success(())
        case .failure(let error):
            return .failure(error)
        }
    }

    func remoteASRStatusText(
        for provider: RemoteASRProvider,
        configuration: RemoteProviderConfiguration
    ) -> String {
        guard configuration.isConfigured else {
            return AppLocalization.localizedString("Not configured")
        }

        _ = provider
        return ""
    }

    func resolvedASRHintSettings(for target: ASRHintTarget) -> ASRHintSettings {
        ASRHintSettingsStore.resolvedSettings(for: target, rawValue: asrHintSettingsRaw)
    }

    func saveASRHintSettings(_ settings: ASRHintSettings, for target: ASRHintTarget) {
        var updated = ASRHintSettingsStore.load(from: asrHintSettingsRaw)
        updated[target] = ASRHintSettingsStore.sanitized(settings, for: target)
        asrHintSettingsRaw = ASRHintSettingsStore.storageValue(for: updated)
    }

    func asrHintSettingsBinding(for target: ASRHintTarget) -> Binding<ASRHintSettings> {
        Binding(
            get: { resolvedASRHintSettings(for: target) },
            set: { saveASRHintSettings($0, for: target) }
        )
    }

    func resolvedMLXLocalTuningSettings(for repo: String) -> MLXLocalTuningSettings {
        MLXLocalTuningSettingsStore.resolvedSettings(
            for: repo,
            rawValue: mlxLocalASRTuningSettingsRaw
        )
    }

    func saveMLXLocalTuningSettings(_ settings: MLXLocalTuningSettings, for repo: String) {
        mlxLocalASRTuningSettingsRaw = MLXLocalTuningSettingsStore.save(
            settings,
            for: repo,
            rawValue: mlxLocalASRTuningSettingsRaw
        )
    }

    func mlxLocalTuningSettingsBinding(for repo: String) -> Binding<MLXLocalTuningSettings> {
        Binding(
            get: { resolvedMLXLocalTuningSettings(for: repo) },
            set: { saveMLXLocalTuningSettings($0, for: repo) }
        )
    }

    func resolvedCustomLLMGenerationSettings(for repo: String) -> LLMGenerationSettings {
        CustomLLMGenerationSettingsStore.resolvedSettings(
            for: repo,
            rawByRepo: customLLMGenerationSettingsByRepoRaw,
            legacyRaw: customLLMGenerationSettingsRaw
        )
    }

    func saveCustomLLMGenerationSettings(_ settings: LLMGenerationSettings, for repo: String) {
        customLLMGenerationSettingsByRepoRaw = CustomLLMGenerationSettingsStore.save(
            settings,
            for: repo,
            rawByRepo: customLLMGenerationSettingsByRepoRaw
        )
    }

    func customLLMGenerationSettingsBinding(for repo: String) -> Binding<LLMGenerationSettings> {
        Binding(
            get: { resolvedCustomLLMGenerationSettings(for: repo) },
            set: { saveCustomLLMGenerationSettings($0, for: repo) }
        )
    }

    func useRemoteLLMProvider(_ provider: RemoteLLMProvider) {
        remoteLLMSelectedProviderRaw = provider.rawValue
    }

    func saveRemoteLLMConfiguration(
        _ configuration: RemoteProviderConfiguration
    ) -> Result<Void, RemoteModelConfigurationStore.SaveError> {
        let result = RemoteModelConfigurationStore.saveConfiguration(
            configuration,
            updating: remoteLLMProviderConfigurationsRaw
        )
        switch result {
        case .success(let raw):
            remoteLLMProviderConfigurationsRaw = raw
            NotificationCenter.default.post(name: .voxtRemoteProviderConfigurationsDidChange, object: nil)
            NotificationCenter.default.post(name: .voxtRemoteLLMProviderConfigurationsDidChange, object: nil)
            return .success(())
        case .failure(let error):
            return .failure(error)
        }
    }

    func refreshModelInstallStateIfNeeded() {
        if case .downloading = mlxModelManager.state {
            // Keep current transient state during active downloads.
        } else if case .paused = mlxModelManager.state {
            // Preserve paused state while download cancellation settles.
        } else if case .loading = mlxModelManager.state {
            // Avoid resetting while model is being loaded.
        } else {
            mlxModelManager.checkExistingModel(refresh: true)
        }

        if !customLLMManager.activeDownloadRepos.isEmpty {
            // Keep current transient state during active downloads.
        } else if case .paused = customLLMManager.state {
            // Preserve paused state while download cancellation settles.
        } else {
            customLLMManager.checkExistingModel(refresh: true)
        }
    }

    func openMLXModelDirectory(_ repo: String) {
        guard let folderURL = mlxModelManager.modelDirectoryURL(repo: repo) else { return }
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: folderURL.path)
    }

    func openCustomLLMModelDirectory(_ repo: String) {
        guard let folderURL = customLLMManager.modelDirectoryURL(repo: repo) else { return }
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: folderURL.path)
    }

    func modelLocalizedDescription(for repo: String) -> LocalizedStringKey {
        let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
        if let description = MLXModelCatalog.description(for: canonicalRepo) {
            return LocalizedStringKey(description)
        }
        return LocalizedStringKey("")
    }

    var mlxConfigurationSummary: String {
        let tuning = resolvedMLXLocalTuningSettings(for: modelRepo)
        return MLXConfigurationSummarySupport.summary(for: modelRepo, tuning: tuning)
    }

    var customLLMGenerationSummary: String {
        let settings = resolvedCustomLLMGenerationSettings(for: customLLMRepo)
        var parts = [String]()
        switch settings.thinking.mode {
        case .providerDefault:
            parts.append(AppLocalization.localizedString("Think: Off"))
        case .off:
            parts.append(AppLocalization.localizedString("Think: Off"))
        case .on:
            parts.append(AppLocalization.localizedString("Think: On"))
        case .effort, .budget:
            break
        }
        if let maxOutputTokens = settings.maxOutputTokens {
            parts.append(AppLocalization.format("Max Output: %@", String(maxOutputTokens)))
        }
        if let temperature = settings.temperature {
            parts.append(AppLocalization.format("Temperature: %@", String(format: "%.2f", temperature)))
        }
        if let topP = settings.topP {
            parts.append(AppLocalization.format("Top P: %@", String(format: "%.2f", topP)))
        }
        if let repetitionPenalty = settings.repetitionPenalty {
            parts.append(AppLocalization.format("Repetition Penalty: %@", String(format: "%.2f", repetitionPenalty)))
        }
        return parts.isEmpty ? AppLocalization.localizedString("Configuration: Default") : parts.joined(separator: " · ")
    }

    func asrCredentialHint(for provider: RemoteASRProvider) -> String? {
        switch provider {
        case .doubaoASR:
            return AppLocalization.localizedString("Doubao uses App ID + Access Token for streaming API.")
        case .aliyunBailianASR:
            return AppLocalization.localizedString("Aliyun ASR in Voxt uses realtime WebSocket only: Qwen models use /api-ws/v1/realtime, Fun/Paraformer models use /api-ws/v1/inference.")
        case .xiaomiMiMoASR:
            return AppLocalization.localizedString("Xiaomi MiMo ASR uses a MiMo API Key and the OpenAI-compatible chat completions audio endpoint.")
        case .googleGeminiASR:
            return AppLocalization.localizedString("Gemini live transcribe uses a Google AI Studio API key over the Live WebSocket API. Voice input only: file transcription and meeting mode are not supported.")
        case .openAIWhisper, .glmASR, .stepFunASR:
            return nil
        }
    }
}
