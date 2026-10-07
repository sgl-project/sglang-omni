// ModelSettingsCatalogBuilder.swift
// Provides Model Settings Catalog Builder for model settings.

import SwiftUI

private func localizedModelCatalog(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

@MainActor
struct ModelCatalogBuilder {
    struct CatalogDecoration {
        let filterTags: [String]
        let displayTags: [String]
        let usageLocations: [String]
    }

    let mlxModelManager: MLXModelManager
    let customLLMManager: CustomLLMModelManager
    let ggufTranslationModelManager: GGUFTranslationModelManager
    let remoteASRConfigurations: [String: RemoteProviderConfiguration]
    let remoteLLMConfigurations: [String: RemoteProviderConfiguration]
    let featureSettings: FeatureSettings
    let hasIssue: (ModelConfigurationIssue.Scope) -> Bool
    let customLLMBadgeText: (String) -> String?
    let remoteASRStatusText: (RemoteASRProvider, RemoteProviderConfiguration) -> String
    let remoteLLMBadgeText: (RemoteLLMProvider) -> String?
    let primaryUserLanguageCode: String?
    let appleIntelligenceAvailability: AppleIntelligenceAvailability
    let mlxInstallSnapshot: (String) -> LocalModelInstallSnapshot
    let customLLMInstallSnapshot: (String) -> LocalModelInstallSnapshot
    let ggufTranslationInstallSnapshot: (GGUFTranslationModelID) -> LocalModelInstallSnapshot
    let catalogPrimaryAction: (LocalModelInstallSnapshot) -> ModelTableAction?
    let catalogSecondaryActions: (LocalModelInstallSnapshot) -> [ModelTableAction]
    let configureASRProvider: (RemoteASRProvider) -> Void
    let configureLLMProvider: (RemoteLLMProvider) -> Void
    let showASRHintTarget: (ASRHintTarget) -> Void

    func asrEntries() -> [ModelCatalogEntry] {
        var entries = [ModelCatalogEntry]()

        entries.append(dictationASREntry())
        entries.append(contentsOf: mlxASREntries())

        entries.append(contentsOf: RemoteASRProvider.allCases.map { provider in
            let selectionID = FeatureModelSelectionID.remoteASR(provider)
            let configuration = RemoteModelConfigurationStore.resolvedASRConfiguration(
                provider: provider,
                stored: remoteASRConfigurations
            )
            let configured = configuration.isConfigured
            let needsSetup = hasIssue(.remoteASRProvider(provider))
            let decoration = catalogDecoration(
                base: [localizedModelCatalog("Remote")] + remoteASRCatalogTags(for: provider, configuration: configuration),
                installed: false,
                requiresConfiguration: true,
                configured: configured,
                selectionID: selectionID
            )

            return ModelCatalogEntry(
                id: "remote-asr:\(provider.rawValue)",
                title: provider.title,
                engine: localizedModelCatalog("Remote"),
                sizeText: configuration.hasUsableModel ? configuration.model : localizedModelCatalog("Cloud"),
                ratingText: provider == .openAIWhisper ? "4.6" : "4.4",
                filterTags: decoration.filterTags,
                displayTags: decoration.displayTags,
                statusText: remoteASRStatusText(provider, configuration),
                usageLocations: decoration.usageLocations,
                badgeText: needsSetup
                    ? localizedModelCatalog("Needs Setup")
                    : ModelCatalogBadgeSupport.recommendedBadgeText(forRemoteASRProvider: provider),
                primaryAction: ModelTableAction(title: localizedModelCatalog("Configure")) {
                    configureASRProvider(provider)
                },
                secondaryActions: []
            )
        })

        return entries
    }

    func llmEntries() -> [ModelCatalogEntry] {
        var entries = [ModelCatalogEntry]()

        if appleIntelligenceAvailability != .unsupportedOS {
            let selectionID = FeatureModelSelectionID.appleIntelligence
            let isAvailable = appleIntelligenceAvailability.isAvailable
            let decoration = catalogDecoration(
                base: [localizedModelCatalog("Local"), localizedModelCatalog("Multilingual")],
                installed: true,
                requiresConfiguration: false,
                configured: true,
                selectionID: selectionID
            )

            entries.append(
                ModelCatalogEntry(
                    id: "apple-intelligence",
                    title: localizedModelCatalog("Apple Intelligence"),
                    engine: localizedModelCatalog("Apple"),
                    sizeText: localizedModelCatalog("Built-in"),
                    ratingText: "4.2",
                    filterTags: decoration.filterTags,
                    displayTags: decoration.displayTags,
                    statusText: isAvailable
                        ? localizedModelCatalog("Available on this Mac")
                        : (appleIntelligenceAvailability.disabledReason
                            ?? localizedModelCatalog("Unavailable")),
                    usageLocations: decoration.usageLocations,
                    badgeText: nil,
                    primaryAction: nil,
                    secondaryActions: []
                )
            )
        }

        entries.append(contentsOf: customLLMDisplayModelsIncludingInstalled().map { model in
            let repo = model.id
            let selectionID = FeatureModelSelectionID.localLLM(repo)
            let snapshot = customLLMInstallSnapshot(repo)
            let decoration = catalogDecoration(
                base: [localizedModelCatalog("Local")] + llmCatalogTags(for: repo),
                installed: snapshot.isInstalled,
                requiresConfiguration: false,
                configured: true,
                selectionID: selectionID
            )

            return ModelCatalogEntry(
                id: "local-llm:\(repo)",
                title: customLLMManager.displayTitle(for: repo),
                engine: localizedModelCatalog("Local LLM"),
                sizeText: customLLMManager.remoteSizeText(repo: repo),
                ratingText: CustomLLMModelManager.ratingText(for: repo),
                filterTags: decoration.filterTags,
                displayTags: decoration.displayTags,
                statusText: snapshot.statusText,
                usageLocations: decoration.usageLocations,
                badgeText: snapshot.badgeText,
                primaryAction: catalogPrimaryAction(snapshot),
                secondaryActions: catalogSecondaryActions(snapshot)
            )
        })

        entries.append(contentsOf: ggufTranslationDisplayModelsIncludingInstalled().map { model in
            let selectionID = FeatureModelSelectionID.localGGUFTranslation(model.id)
            let snapshot = ggufTranslationInstallSnapshot(model.id)

            let decoration = catalogDecoration(
                base: model.tags,
                installed: snapshot.isInstalled,
                requiresConfiguration: false,
                configured: true,
                selectionID: selectionID
            )

            return ModelCatalogEntry(
                id: "local-gguf-translation:\(model.id.rawValue)",
                title: model.title,
                engine: localizedModelCatalog("Local GGUF"),
                sizeText: model.sizeText,
                ratingText: model.ratingText,
                filterTags: decoration.filterTags,
                displayTags: decoration.displayTags,
                statusText: snapshot.statusText,
                usageLocations: decoration.usageLocations,
                badgeText: snapshot.badgeText,
                primaryAction: catalogPrimaryAction(snapshot),
                secondaryActions: catalogSecondaryActions(snapshot)
            )
        })

        entries.append(contentsOf: RemoteLLMProvider.allCases.map { provider in
            let selectionID = FeatureModelSelectionID.remoteLLM(provider)
            let configured = RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: provider,
                stored: remoteLLMConfigurations
            )
            let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
                provider: provider,
                stored: remoteLLMConfigurations
            )
            let status = configured ? "" : localizedModelCatalog("Not configured")
            let decoration = catalogDecoration(
                base: [localizedModelCatalog("Remote")] + remoteLLMCatalogTags(for: provider),
                installed: false,
                requiresConfiguration: true,
                configured: configured,
                selectionID: selectionID
            )

            return ModelCatalogEntry(
                id: "remote-llm:\(provider.rawValue)",
                title: provider.title,
                engine: localizedModelCatalog("Remote LLM"),
                sizeText: configured ? configuration.model : localizedModelCatalog("Cloud"),
                ratingText: "4.5",
                filterTags: decoration.filterTags,
                displayTags: decoration.displayTags,
                statusText: status,
                usageLocations: decoration.usageLocations,
                badgeText: remoteLLMBadgeText(provider)
                    ?? ModelCatalogBadgeSupport.recommendedBadgeText(forRemoteLLMProvider: provider),
                primaryAction: ModelTableAction(title: localizedModelCatalog("Configure")) {
                    configureLLMProvider(provider)
                },
                secondaryActions: []
            )
        })

        return entries
    }

    private func customLLMDisplayModelsIncludingInstalled() -> [CustomLLMModelCatalog.Option] {
        var includedRepos = Set(CustomLLMModelManager.supportedModels.compactMap { model -> String? in
            let repo = CustomLLMModelManager.canonicalModelRepo(model.id)
            let snapshot = customLLMInstallSnapshot(repo)
            switch snapshot.state {
            case .installed, .downloading, .paused:
                return repo
            case .checking, .installable, .cancelling, .uninstalling:
                return nil
            }
        })
        includedRepos.insert(CustomLLMModelManager.canonicalModelRepo(customLLMManager.currentModelRepo))

        if featureSettings.transcription.llmEnabled,
           case .localLLM(let repo)? = featureSettings.transcription.llmSelectionID.textSelection {
            includedRepos.insert(CustomLLMModelManager.canonicalModelRepo(repo))
        }
        if case .localLLM(let repo)? = featureSettings.translation.modelSelectionID.translationSelection {
            includedRepos.insert(CustomLLMModelManager.canonicalModelRepo(repo))
        }
        if case .localLLM(let repo)? = featureSettings.rewrite.llmSelectionID.textSelection {
            includedRepos.insert(CustomLLMModelManager.canonicalModelRepo(repo))
        }
        if case .localLLM(let repo)? = featureSettings.meeting.summaryModelSelectionID.textSelection {
            includedRepos.insert(CustomLLMModelManager.canonicalModelRepo(repo))
        }

        return CustomLLMModelManager.displayModels(includingInstalled: includedRepos)
    }

    private func ggufTranslationDisplayModelsIncludingInstalled() -> [GGUFTranslationModelOption] {
        var includedIDs = Set(GGUFTranslationModelCatalog.allModels.compactMap { model -> GGUFTranslationModelID? in
            let snapshot = ggufTranslationInstallSnapshot(model.id)
            switch snapshot.state {
            case .installed, .downloading, .paused:
                return model.id
            case .checking, .installable, .cancelling, .uninstalling:
                return nil
            }
        })
        includedIDs.insert(ggufTranslationModelManager.selectedModelID)

        if case .localGGUF(let modelID)? = featureSettings.translation.modelSelectionID.translationSelection {
            includedIDs.insert(modelID)
        }

        return GGUFTranslationModelCatalog.displayModels(includingInstalled: includedIDs)
    }

    func usageLocations(for selectionID: FeatureModelSelectionID) -> [String] {
        var labels = [String]()
        if featureSettings.transcription.asrSelectionID == selectionID ||
            (featureSettings.transcription.llmEnabled && featureSettings.transcription.llmSelectionID == selectionID) {
            labels.append(localizedModelCatalog("Transcription"))
        }
        if featureSettings.translation.asrSelectionID == selectionID ||
            featureSettings.translation.modelSelectionID == selectionID {
            labels.append(localizedModelCatalog("Translation"))
        }
        if featureSettings.rewrite.asrSelectionID == selectionID ||
            featureSettings.rewrite.llmSelectionID == selectionID {
            labels.append(localizedModelCatalog("Rewrite"))
        }
        return labels
    }

    func catalogDecoration(
        base: [String],
        installed: Bool,
        requiresConfiguration: Bool,
        configured: Bool,
        selectionID: FeatureModelSelectionID
    ) -> CatalogDecoration {
        let usageLocations = usageLocations(for: selectionID)
        var filterTags = base
        if installed {
            filterTags.append(localizedModelCatalog("Installed"))
        }
        if requiresConfiguration && configured {
            filterTags.append(localizedModelCatalog("Configured"))
        }
        if !usageLocations.isEmpty {
            filterTags.append(localizedModelCatalog("In Use"))
        }

        var displayTags = base.filter { $0 != localizedModelCatalog("Multilingual") }
        if let languageSupportTag = primaryLanguageSupportTag(for: selectionID) {
            displayTags.append(languageSupportTag)
        }
        if requiresConfiguration && configured {
            displayTags.append(localizedModelCatalog("Configured"))
        }
        if !usageLocations.isEmpty {
            displayTags.append(localizedModelCatalog("In Use"))
        }

        return CatalogDecoration(
            filterTags: deduplicatedTags(filterTags),
            displayTags: deduplicatedTags(displayTags),
            usageLocations: usageLocations
        )
    }

    func catalogFilterTags(
        base: [String],
        installed: Bool,
        requiresConfiguration: Bool,
        configured: Bool,
        selectionID: FeatureModelSelectionID
    ) -> [String] {
        var tags = base
        if installed {
            tags.append(localizedModelCatalog("Installed"))
        }
        if requiresConfiguration && configured {
            tags.append(localizedModelCatalog("Configured"))
        }
        if !usageLocations(for: selectionID).isEmpty {
            tags.append(localizedModelCatalog("In Use"))
        }
        return deduplicatedTags(tags)
    }

    func catalogDisplayTags(
        base: [String],
        requiresConfiguration: Bool,
        configured: Bool,
        selectionID: FeatureModelSelectionID
    ) -> [String] {
        catalogDecoration(
            base: base,
            installed: false,
            requiresConfiguration: requiresConfiguration,
            configured: configured,
            selectionID: selectionID
        )
        .displayTags
    }

    func mlxCatalogTags(for repo: String) -> [String] {
        deduplicatedTags(MLXModelManager.catalogTagKeys(for: repo).map(localizedModelCatalog))
    }

    private func llmCatalogTags(for repo: String) -> [String] {
        deduplicatedTags(CustomLLMModelManager.catalogTagKeys(for: repo).map(localizedModelCatalog))
    }

    private func remoteASRCatalogTags(
        for provider: RemoteASRProvider,
        configuration: RemoteProviderConfiguration
    ) -> [String] {
        var tags = [String]()
        switch provider {
        case .openAIWhisper:
            tags.append(localizedModelCatalog("Multilingual"))
        case .doubaoASR:
            tags.append(contentsOf: [localizedModelCatalog("Realtime"), localizedModelCatalog("Multilingual")])
        case .glmASR:
            tags.append(contentsOf: [localizedModelCatalog("Accurate"), localizedModelCatalog("Multilingual")])
        case .aliyunBailianASR:
            tags.append(localizedModelCatalog("Multilingual"))
            if RemoteASRRealtimeSupport.isAliyunRealtimeModel(configuration.model) {
                tags.append(localizedModelCatalog("Realtime"))
            }
        case .stepFunASR:
            if RemoteASRRealtimeSupport.isStepFunRealtimeModel(configuration.model) {
                tags.append(localizedModelCatalog("Realtime"))
            }
            tags.append(contentsOf: [localizedModelCatalog("Accurate"), localizedModelCatalog("Multilingual")])
        case .xiaomiMiMoASR:
            tags.append(contentsOf: [localizedModelCatalog("Accurate"), localizedModelCatalog("Multilingual")])
        case .googleGeminiASR:
            tags.append(contentsOf: [
                localizedModelCatalog("Realtime"),
                localizedModelCatalog("Accurate"),
                localizedModelCatalog("Multilingual")
            ])
        }
        return deduplicatedTags(tags)
    }

    private func remoteLLMCatalogTags(for provider: RemoteLLMProvider) -> [String] {
        switch provider {
        case .lmStudio, .ollama, .omlx:
            return []
        default:
            return [localizedModelCatalog("Accurate")]
        }
    }

    private func primaryLanguageSupportTag(for selectionID: FeatureModelSelectionID) -> String? {
        guard let support = supportsPrimaryLanguage(for: selectionID) else { return nil }
        return localizedModelCatalog(support ? "Supports Primary Language" : "Does Not Support Primary Language")
    }

    private func supportsPrimaryLanguage(for selectionID: FeatureModelSelectionID) -> Bool? {
        guard let primaryLanguage = resolvedPrimaryLanguageOption() else { return nil }

        switch selectionID.asrSelection {
        case .dictation:
            return true
        case .mlx(let repo):
            return mlxSupportsPrimaryLanguage(repo, primaryLanguage: primaryLanguage)
        case .remote:
            return true
        case .none:
            return nil
        }
    }

    private func resolvedPrimaryLanguageOption() -> UserMainLanguageOption? {
        guard let primaryUserLanguageCode else { return nil }
        return UserMainLanguageOption.option(for: primaryUserLanguageCode)
    }

    private func mlxSupportsPrimaryLanguage(
        _ repo: String,
        primaryLanguage: UserMainLanguageOption
    ) -> Bool {
        MLXModelCatalog.supportsLanguage(primaryLanguage.baseLanguageCode, for: repo)
    }

    private func deduplicatedTags(_ tags: [String]) -> [String] {
        Array(NSOrderedSet(array: tags)) as? [String] ?? tags
    }
}
