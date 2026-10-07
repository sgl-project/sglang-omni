// ModelSettingsSelection.swift
// Provides Model Settings Selection for model settings.

import SwiftUI

extension ModelSettingsView {
    private func installedCustomLLMOptions(including currentRepo: String) -> [TranslationModelOption] {
        CustomLLMModelManager.displayModels(including: currentRepo).compactMap { model in
            guard customLLMManager.isModelDownloaded(repo: model.id) else {
                return nil
            }
            return TranslationModelOption(id: model.id, title: model.title)
        }
    }

    private var installedGGUFTranslationOptions: [TranslationModelOption] {
        GGUFTranslationModelCatalog.allModels.compactMap { model in
            guard ggufTranslationModelManager.isModelDownloaded(id: model.id) else {
                return nil
            }
            return TranslationModelOption(id: model.id.rawValue, title: model.shortTitle)
        }
    }

    var translationProviderOptions: [ModelSettingsProviderOption] {
        TranslationModelProvider.allCases.map {
            ModelSettingsProviderOption(id: $0.rawValue, title: $0.title)
        }
    }

    var rewriteProviderOptions: [ModelSettingsProviderOption] {
        RewriteModelProvider.allCases.map {
            ModelSettingsProviderOption(id: $0.rawValue, title: $0.title)
        }
    }

    var installedCustomLLMOptions: [TranslationModelOption] {
        installedCustomLLMOptions(including: customLLMRepo)
    }

    var configuredRemoteLLMOptions: [TranslationModelOption] {
        RemoteLLMProvider.allCases.compactMap { provider in
            guard let config = remoteLLMConfigurations[provider.rawValue] else {
                return nil
            }
            guard config.hasUsableModel else {
                return nil
            }
            return TranslationModelOption(
                id: provider.rawValue,
                title: "\(provider.title) · \(config.model)"
            )
        }
    }

    var translationModelOptions: [TranslationModelOption] {
        switch selectedTranslationModelProvider {
        case .remoteLLM:
            return configuredRemoteLLMOptions
        case .customLLM:
            return installedCustomLLMOptions(including: translationCustomLLMRepo)
        case .localGGUF:
            return installedGGUFTranslationOptions
        }
    }

    var rewriteModelOptions: [TranslationModelOption] {
        switch selectedRewriteModelProvider {
        case .remoteLLM:
            return configuredRemoteLLMOptions
        case .customLLM:
            return installedCustomLLMOptions(including: rewriteCustomLLMRepo)
        }
    }

    var translationModelSelectionBinding: Binding<String> {
        Binding(
            get: { resolvedTranslationSelection },
            set: { newValue in
                switch selectedTranslationModelProvider {
                case .remoteLLM:
                    translationRemoteLLMProviderRaw = newValue
                case .customLLM:
                    translationCustomLLMRepo = newValue
                case .localGGUF:
                    translationGGUFModelIDRaw = GGUFTranslationModelCatalog.resolvedModelID(newValue).rawValue
                }
            }
        )
    }

    var resolvedTranslationSelection: String {
        let options = translationModelOptions
        let rawSelection = currentTranslationSelectionRaw
        let canonicalSelection = CustomLLMModelManager.canonicalModelRepo(rawSelection)
        guard !options.isEmpty else {
            return selectedTranslationModelProvider == .customLLM ? canonicalSelection : rawSelection
        }

        let selectionToMatch = selectedTranslationModelProvider == .customLLM
            ? canonicalSelection
            : rawSelection
        if options.contains(where: { $0.id == selectionToMatch }) {
            return selectionToMatch
        }
        return options[0].id
    }

    var rewriteModelSelectionBinding: Binding<String> {
        Binding(
            get: { resolvedRewriteSelection },
            set: { newValue in
                switch selectedRewriteModelProvider {
                case .remoteLLM:
                    rewriteRemoteLLMProviderRaw = newValue
                case .customLLM:
                    rewriteCustomLLMRepo = newValue
                }
            }
        )
    }

    var resolvedRewriteSelection: String {
        let options = rewriteModelOptions
        let rawSelection = currentRewriteSelectionRaw
        let canonicalSelection = CustomLLMModelManager.canonicalModelRepo(rawSelection)
        guard !options.isEmpty else {
            return selectedRewriteModelProvider == .customLLM ? canonicalSelection : rawSelection
        }

        let selectionToMatch = selectedRewriteModelProvider == .customLLM
            ? canonicalSelection
            : rawSelection
        if options.contains(where: { $0.id == selectionToMatch }) {
            return selectionToMatch
        }
        return options[0].id
    }

    var currentTranslationSelectionRaw: String {
        translationSelectionRaw(for: selectedTranslationModelProvider)
    }

    func translationSelectionRaw(for provider: TranslationModelProvider) -> String {
        switch provider {
        case .remoteLLM:
            return translationRemoteLLMProviderRaw
        case .customLLM:
            return translationCustomLLMRepo
        case .localGGUF:
            return translationGGUFModelIDRaw
        }
    }

    var currentRewriteSelectionRaw: String {
        switch selectedRewriteModelProvider {
        case .remoteLLM:
            return rewriteRemoteLLMProviderRaw
        case .customLLM:
            return rewriteCustomLLMRepo
        }
    }

    var translationModelLabelText: String {
        switch selectedTranslationModelProvider {
        case .remoteLLM:
            return "Remote LLM Model"
        case .customLLM:
            return "Custom LLM Model"
        case .localGGUF:
            return "Local GGUF Model"
        }
    }

    var translationModelEmptyStateText: String {
        switch selectedTranslationModelProvider {
        case .remoteLLM:
            return "No configured remote LLM model yet. Configure a provider above."
        case .customLLM:
            return "No installed custom LLM model yet. Install one in the table above."
        case .localGGUF:
            return "No installed GGUF translation model yet. Install Hy in the model catalog first."
        }
    }

    var translationModelDisplayText: String? {
        nil
    }

    var translationProviderStatusMessage: String? {
        TranslationProviderResolver.warningMessage(
            selectedProvider: selectedTranslationModelProvider,
            transcriptionEngine: selectedEngine,
            targetLanguage: selectedTranslationTargetLanguage
        )
    }

    var translationProviderStatusIsWarning: Bool {
        TranslationProviderResolver.warningMessage(
            selectedProvider: selectedTranslationModelProvider,
            transcriptionEngine: selectedEngine,
            targetLanguage: selectedTranslationTargetLanguage
        ) != nil
    }

    var rewriteModelLabelText: String {
        selectedRewriteModelProvider == .remoteLLM ? "Remote LLM Model" : "Custom LLM Model"
    }

    var rewriteModelEmptyStateText: String {
        selectedRewriteModelProvider == .remoteLLM
            ? "No configured remote LLM model yet. Configure a provider above."
            : "No installed custom LLM model yet. Install one in the table above."
    }

    func ensureTranslationModelSelectionConsistency() {
        switch selectedTranslationModelProvider {
        case .remoteLLM:
            let options = configuredRemoteLLMOptions
            guard let first = options.first else {
                translationRemoteLLMProviderRaw = ""
                return
            }
            if !options.contains(where: { $0.id == translationRemoteLLMProviderRaw }) {
                translationRemoteLLMProviderRaw = first.id
            }
        case .customLLM:
            if translationCustomLLMRepo.isEmpty {
                translationCustomLLMRepo = customLLMRepo
            } else if !CustomLLMModelManager.isSupportedModelRepo(translationCustomLLMRepo) {
                translationCustomLLMRepo = customLLMRepo
            }
        case .localGGUF:
            translationGGUFModelIDRaw = GGUFTranslationModelCatalog.resolvedModelID(translationGGUFModelIDRaw).rawValue
        }
    }

    func ensureRewriteModelSelectionConsistency() {
        switch selectedRewriteModelProvider {
        case .remoteLLM:
            let options = configuredRemoteLLMOptions
            guard let first = options.first else {
                rewriteRemoteLLMProviderRaw = ""
                return
            }
            if !options.contains(where: { $0.id == rewriteRemoteLLMProviderRaw }) {
                rewriteRemoteLLMProviderRaw = first.id
            }
        case .customLLM:
            if rewriteCustomLLMRepo.isEmpty {
                rewriteCustomLLMRepo = customLLMRepo
            } else if !CustomLLMModelManager.isSupportedModelRepo(rewriteCustomLLMRepo) {
                rewriteCustomLLMRepo = customLLMRepo
            }
        }
    }
}

struct TranslationModelOption: Identifiable, Hashable {
    let id: String
    let title: String
}
