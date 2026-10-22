// ModelConfigurationIssueResolver.swift
// Provides model setup issue detection for settings screens.

import Foundation

struct ModelConfigurationIssue: Identifiable, Hashable {
    enum Scope: Hashable {
        case remoteASRProvider(RemoteASRProvider)
        case remoteLLMProvider(RemoteLLMProvider)
        case mlxModel(String)
        case customLLMModel(String)
        case translationRemoteLLM(RemoteLLMProvider)
        case rewriteRemoteLLM(RemoteLLMProvider)
        case translationCustomLLM(String)
        case rewriteCustomLLM(String)
    }

    let scope: Scope
    let message: String

    var id: String {
        switch scope {
        case .remoteASRProvider(let provider):
            return "asr:\(provider.rawValue)"
        case .remoteLLMProvider(let provider):
            return "llm:\(provider.rawValue)"
        case .mlxModel(let repo):
            return "mlx:\(repo)"
        case .customLLMModel(let repo):
            return "custom:\(repo)"
        case .translationRemoteLLM(let provider):
            return "translation-llm:\(provider.rawValue)"
        case .rewriteRemoteLLM(let provider):
            return "rewrite-llm:\(provider.rawValue)"
        case .translationCustomLLM(let repo):
            return "translation-custom:\(repo)"
        case .rewriteCustomLLM(let repo):
            return "rewrite-custom:\(repo)"
        }
    }
}

enum ModelConfigurationIssueResolver {
    private static var modelNeedsInstallMessage: String {
        AppLocalization.localizedString("Model needs to be installed.")
    }

    private static var configurationRequiredMessage: String {
        AppLocalization.localizedString("Configuration required.")
    }

    static func missingIssues(
        defaults: UserDefaults = .standard,
        mlxModelManager: MLXModelManager,
        customLLMManager: CustomLLMModelManager
    ) -> [ModelConfigurationIssue] {
        var issues: [ModelConfigurationIssue] = []

        let featureSettings = FeatureSettingsStore.load(defaults: defaults)
        let remoteASR = RemoteModelConfigurationStore.loadConfigurations(
            from: defaults.string(forKey: AppPreferenceKey.remoteASRProviderConfigurations) ?? "",
            sensitiveValueLoading: .metadataOnly
        )
        let remoteLLM = RemoteModelConfigurationStore.loadConfigurations(
            from: defaults.string(forKey: AppPreferenceKey.remoteLLMProviderConfigurations) ?? "",
            sensitiveValueLoading: .metadataOnly
        )

        appendASRIssues(
            for: featureSettings.transcription.asrSelectionID,
            issues: &issues,
            remoteASR: remoteASR,
            mlxModelManager: mlxModelManager,
        )
        if featureSettings.transcription.llmEnabled {
            appendTextModelIssues(
                for: featureSettings.transcription.llmSelectionID,
                issues: &issues,
                remoteLLM: remoteLLM,
                customLLMManager: customLLMManager
            )
        }

        appendASRIssues(
            for: featureSettings.translation.asrSelectionID,
            issues: &issues,
            remoteASR: remoteASR,
            mlxModelManager: mlxModelManager,
        )
        appendTranslationModelIssues(
            for: featureSettings.translation,
            issues: &issues,
            remoteLLM: remoteLLM,
            customLLMManager: customLLMManager
        )

        appendASRIssues(
            for: featureSettings.rewrite.asrSelectionID,
            issues: &issues,
            remoteASR: remoteASR,
            mlxModelManager: mlxModelManager,
        )
        appendTextModelIssues(
            for: featureSettings.rewrite.llmSelectionID,
            issues: &issues,
            remoteLLM: remoteLLM,
            customLLMManager: customLLMManager
        )

        return Array(Set(issues)).sorted { $0.id < $1.id }
    }

    private static func appendASRIssues(
        for selectionID: FeatureModelSelectionID,
        issues: inout [ModelConfigurationIssue],
        remoteASR: [String: RemoteProviderConfiguration],
        mlxModelManager: MLXModelManager
    ) {
        switch selectionID.asrSelection {
        case .dictation, .none:
            return
        case .mlx(let repo):
            let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
            // Unknown is a pending background scan, not evidence of a missing model.
            if !mlxModelManager.isCheckingInstallation(repo: canonicalRepo),
               !mlxModelManager.isModelDownloaded(repo: canonicalRepo) {
                issues.append(.init(scope: .mlxModel(canonicalRepo), message: modelNeedsInstallMessage))
            }
        case .remote(let provider):
            let configuration = RemoteModelConfigurationStore.resolvedASRConfiguration(provider: provider, stored: remoteASR)
            if !configuration.isConfigured {
                issues.append(.init(scope: .remoteASRProvider(provider), message: configurationRequiredMessage))
            }
        }
    }

    private static func appendTextModelIssues(
        for selectionID: FeatureModelSelectionID,
        issues: inout [ModelConfigurationIssue],
        remoteLLM: [String: RemoteProviderConfiguration],
        customLLMManager: CustomLLMModelManager
    ) {
        switch selectionID.textSelection {
        case .appleIntelligence, .none:
            return
        case .localLLM(let repo):
            if !customLLMManager.isCheckingInstallation(repo: repo),
               !customLLMManager.isModelDownloaded(repo: repo) {
                issues.append(.init(scope: .customLLMModel(repo), message: modelNeedsInstallMessage))
            }
        case .remoteLLM(let provider):
            if !RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: provider,
                stored: remoteLLM
            ) {
                issues.append(.init(scope: .remoteLLMProvider(provider), message: configurationRequiredMessage))
            }
        }
    }

    private static func appendTranslationModelIssues(
        for settings: TranslationFeatureSettings,
        issues: inout [ModelConfigurationIssue],
        remoteLLM: [String: RemoteProviderConfiguration],
        customLLMManager: CustomLLMModelManager
    ) {
        switch settings.modelSelectionID.translationSelection {
        case .none:
            return
        case .localLLM(let repo):
            if !customLLMManager.isCheckingInstallation(repo: repo),
               !customLLMManager.isModelDownloaded(repo: repo) {
                issues.append(.init(scope: .translationCustomLLM(repo), message: modelNeedsInstallMessage))
            }
        case .localGGUF:
            return
        case .remoteLLM(let provider):
            if !RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: provider,
                stored: remoteLLM
            ) {
                issues.append(.init(scope: .translationRemoteLLM(provider), message: configurationRequiredMessage))
            }
        }
    }
}
