// ModelSettingsSelection.swift
// Provides Model Settings Selection for model settings.

import SwiftUI

extension ModelSettingsView {
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
