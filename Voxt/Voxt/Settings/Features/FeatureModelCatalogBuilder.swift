// FeatureModelCatalogBuilder.swift
// Provides Feature Model Catalog Builder for feature settings.

import Foundation

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

@MainActor
struct FeatureModelCatalogBuilder {
    let mlxModelManager: MLXModelManager
    let customLLMManager: CustomLLMModelManager
    let ggufTranslationModelManager: GGUFTranslationModelManager
    let featureSettings: FeatureSettings
    let remoteASRProviderConfigurationsRaw: String
    let remoteLLMProviderConfigurationsRaw: String
    let appleIntelligenceAvailability: AppleIntelligenceAvailability
    let primaryUserLanguageCode: String?

    func entries(for sheet: FeatureModelSelectorSheet) -> [FeatureModelSelectorEntry] {
        switch sheet {
        case .transcriptionASR, .translationASR, .rewriteASR, .meetingASR:
            return asrEntries(for: sheet)
        case .transcriptionLLM, .transcriptionNoteTitle, .rewriteLLM, .meetingSummary:
            return llmEntries(includeAppleIntelligence: true)
        case .translationModel:
            return translationEntries(
                selectedASR: featureSettings.translation.asrSelectionID,
                targetLanguage: featureSettings.translation.targetLanguage
            )
        }
    }

    func asrSelectionSummary(_ selectionID: FeatureModelSelectionID) -> String {
        switch selectionID.asrSelection {
        case .dictation:
            return localized("Direct Dictation")
        case .mlx(let repo):
            return mlxModelManager.displayTitle(for: repo)
        case .remote(let provider):
            let configurations = RemoteModelConfigurationStore.loadConfigurations(
                from: remoteASRProviderConfigurationsRaw,
                sensitiveValueLoading: .metadataOnly
            )
            let configuration = RemoteModelConfigurationStore.resolvedASRConfiguration(provider: provider, stored: configurations)
            return configuration.hasUsableModel ? "\(provider.title) · \(configuration.model)" : provider.title
        case .none:
            return localized("Not selected")
        }
    }

    func asrSelectionBadgeTitle(_ selectionID: FeatureModelSelectionID) -> String {
        switch selectionID.asrSelection {
        case .dictation:
            return localized("Direct Dictation")
        case .mlx(let repo):
            return Self.compactModelBadgeTitle(from: mlxModelManager.displayTitle(for: repo))
        case .remote(let provider):
            return provider.title
        case .none:
            return localized("Not selected")
        }
    }

    func asrSelectionLogoKey(_ selectionID: FeatureModelSelectionID) -> ModelLogoKey {
        switch selectionID.asrSelection {
        case .dictation:
            return .apple
        case .mlx(let repo):
            return ModelLogoKey.resolve(title: mlxModelManager.displayTitle(for: repo), engine: "MLX")
        case .remote(let provider):
            return ModelLogoKey.resolve(title: provider.title, engine: "Remote ASR")
        case .none:
            return .generic
        }
    }

    func llmSelectionSummary(_ selectionID: FeatureModelSelectionID) -> String {
        switch selectionID.textSelection {
        case .appleIntelligence:
            return localized("Apple Intelligence")
        case .localLLM(let repo):
            return customLLMManager.displayTitle(for: repo)
        case .remoteLLM(let provider):
            let configurations = RemoteModelConfigurationStore.loadConfigurations(
                from: remoteLLMProviderConfigurationsRaw,
                sensitiveValueLoading: .metadataOnly
            )
            guard RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: provider,
                stored: configurations
            ) else {
                return provider.title
            }
            let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(provider: provider, stored: configurations)
            return "\(provider.title) · \(configuration.model)"
        case .none:
            return localized("Not selected")
        }
    }

    func llmSelectionBadgeTitle(_ selectionID: FeatureModelSelectionID) -> String {
        switch selectionID.textSelection {
        case .appleIntelligence:
            return localized("Apple Intelligence")
        case .localLLM(let repo):
            return Self.compactModelBadgeTitle(from: customLLMManager.displayTitle(for: repo))
        case .remoteLLM(let provider):
            return provider.title
        case .none:
            return localized("Not selected")
        }
    }

    func llmSelectionLogoKey(_ selectionID: FeatureModelSelectionID) -> ModelLogoKey {
        switch selectionID.textSelection {
        case .appleIntelligence:
            return .apple
        case .localLLM(let repo):
            return ModelLogoKey.resolve(title: customLLMManager.displayTitle(for: repo), engine: "Local LLM")
        case .remoteLLM(let provider):
            return ModelLogoKey.resolve(title: provider.title, engine: "Remote LLM")
        case .none:
            return .generic
        }
    }

    func translationSelectionSummary(_ selectionID: FeatureModelSelectionID) -> String {
        switch selectionID.translationSelection {
        case .localGGUF(let modelID):
            return ggufTranslationModelManager.displayTitle(for: modelID)
        case .localLLM, .remoteLLM:
            return llmSelectionSummary(selectionID)
        case .none:
            return localized("Not selected")
        }
    }

    func translationSelectionBadgeTitle(_ selectionID: FeatureModelSelectionID) -> String {
        switch selectionID.translationSelection {
        case .localGGUF(let modelID):
            return Self.compactModelBadgeTitle(from: ggufTranslationModelManager.displayTitle(for: modelID))
        case .localLLM, .remoteLLM:
            return llmSelectionBadgeTitle(selectionID)
        case .none:
            return localized("Not selected")
        }
    }

    func translationSelectionLogoKey(_ selectionID: FeatureModelSelectionID) -> ModelLogoKey {
        switch selectionID.translationSelection {
        case .localGGUF(let modelID):
            return ModelLogoKey.resolve(
                title: ggufTranslationModelManager.displayTitle(for: modelID),
                engine: "Local GGUF"
            )
        case .localLLM, .remoteLLM:
            return llmSelectionLogoKey(selectionID)
        case .none:
            return .generic
        }
    }

    /// Compact badge label: provider for remotes, family name for locals (no size / series / quant).
    nonisolated static func compactModelBadgeTitle(from title: String) -> String {
        var value = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if let openParen = value.lastIndex(of: "("), value.hasSuffix(")") {
            value = String(value[..<openParen]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard !value.isEmpty else { return title }

        let normalized = value
            .folding(options: [.diacriticInsensitive, .caseInsensitive], locale: .current)
            .lowercased()

        let families: [(match: String, display: String)] = [
            ("meta llama", "Llama"),
            ("llama", "Llama"),
            ("whisper", "Whisper"),
            ("qwen", "Qwen"),
            ("gemma", "Gemma"),
            ("mistral", "Mistral"),
            ("voxtral", "Voxtral"),
            ("parakeet", "Parakeet"),
            ("firered", "FireRed"),
            ("fire red", "FireRed"),
            ("funasr", "FunASR"),
            ("sensevoice", "SenseVoice"),
            ("moonshine", "Moonshine"),
            ("wav2vec2", "Wav2Vec2"),
            ("canary", "Canary"),
            ("nemotron", "Nemotron"),
            ("granite", "Granite"),
            ("moss", "MOSS"),
            ("cohere", "Cohere"),
            ("mms", "MMS"),
            ("hy-mt2", "Hy-MT2"),
            ("hunyuan", "Hunyuan"),
            ("glm", "GLM"),
            ("phi", "Phi"),
            ("internlm", "InternLM"),
            ("lfm", "LFM"),
            ("deepseek", "DeepSeek")
        ]

        for family in families {
            if matchesFamilyPrefix(normalized, family.match) {
                return family.display
            }
        }

        return value.split(whereSeparator: \.isWhitespace).first.map(String.init) ?? value
    }

    nonisolated private static func matchesFamilyPrefix(_ normalizedTitle: String, _ family: String) -> Bool {
        if normalizedTitle == family {
            return true
        }
        if normalizedTitle.hasPrefix(family + " ") || normalizedTitle.hasPrefix(family + "-") {
            return true
        }
        guard normalizedTitle.hasPrefix(family) else { return false }
        let remainder = normalizedTitle.dropFirst(family.count)
        guard let next = remainder.first else { return true }
        return !next.isLetter
    }

    private func asrEntries(for sheet: FeatureModelSelectorSheet) -> [FeatureModelSelectorEntry] {
        var entries = [FeatureModelSelectorEntry]()
        let dictationSelectable = sheet != .meetingASR
        entries.append(
            FeatureModelSelectorEntry(
                selectionID: .dictation,
                title: localized("Direct Dictation"),
                engine: localized("Apple"),
                sizeText: localized("Built-in"),
                ratingText: "3.4",
                filterTags: [localized("Local"), localized("Built-in"), localized("Multilingual"), localized("Installed")],
                displayTags: featureDisplayTags(
                    base: [localized("Local"), localized("Built-in"), localized("Multilingual")],
                    requiresConfiguration: false,
                    configured: true,
                    selectionID: .dictation
                ),
                statusText: localized("Works immediately with no model download."),
                usageLocations: usageLabels(for: .dictation),
                badgeText: nil,
                isSelectable: dictationSelectable,
                disabledReason: dictationSelectable ? nil : localized("Direct Dictation is not available for Meeting mode.")
            )
        )

        entries.append(contentsOf: mlxModelManager.displayModelsIncludingInstalled().map { model in
            let selectionID = FeatureModelSelectionID.mlx(model.id)
            let isInstalled = mlxModelManager.isModelDownloaded(repo: model.id)
            let isChecking = mlxModelManager.isCheckingInstallation(repo: model.id)
            let availability = Self.localSelectorAvailability(isInstalled: isInstalled, isChecking: isChecking)
            return FeatureModelSelectorEntry(
                selectionID: selectionID,
                title: model.title,
                engine: MLXWhisperMigrationSupport.isWhisperRepo(model.id)
                    ? localized("Whisper (MLX)")
                    : localized("MLX Audio"),
                sizeText: mlxModelManager.remoteSizeText(repo: model.id),
                ratingText: MLXModelManager.ratingText(for: model.id),
                filterTags: featureFilterTags(
                    base: [localized("Local")] + mlxSpeedTags(for: model.id),
                    installed: isInstalled,
                    requiresConfiguration: false,
                    configured: true,
                    usageLabels: usageLabels(for: selectionID)
                ),
                displayTags: featureDisplayTags(
                    base: [localized("Local")] + mlxSpeedTags(for: model.id),
                    requiresConfiguration: false,
                    configured: true,
                    selectionID: selectionID
                ),
                statusText: isChecking ? localized("Loading…") : (isInstalled ? localized("Installed") : localized("Not installed")),
                usageLocations: usageLabels(for: selectionID),
                badgeText: ModelCatalogBadgeSupport.recommendedBadgeText(forMLXRepo: model.id),
                isSelectable: availability.isSelectable,
                disabledReason: availability.disabledReason
            )
        })

        let remoteConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: remoteASRProviderConfigurationsRaw,
            sensitiveValueLoading: .metadataOnly
        )
        entries.append(contentsOf: RemoteASRProvider.allCases.map { provider in
            let selectionID = FeatureModelSelectionID.remoteASR(provider)
            let configuration = RemoteModelConfigurationStore.resolvedASRConfiguration(
                provider: provider,
                stored: remoteConfigurations
            )
            return FeatureModelSelectorEntry(
                selectionID: selectionID,
                title: provider.title,
                engine: localized("Remote"),
                sizeText: configuration.hasUsableModel ? configuration.model : localized("Cloud"),
                ratingText: provider == .openAIWhisper ? "4.6" : "4.4",
                filterTags: featureFilterTags(
                    base: [localized("Remote")] + remoteASRTags(for: provider, configuration: configuration),
                    installed: false,
                    requiresConfiguration: true,
                    configured: configuration.isConfigured,
                    usageLabels: usageLabels(for: selectionID)
                ),
                displayTags: featureDisplayTags(
                    base: [localized("Remote")] + remoteASRTags(for: provider, configuration: configuration),
                    requiresConfiguration: true,
                    configured: configuration.isConfigured,
                    selectionID: selectionID
                ),
                statusText: configuration.isConfigured ? localized("Configured") : localized("Not configured"),
                usageLocations: usageLabels(for: selectionID),
                badgeText: ModelCatalogBadgeSupport.recommendedBadgeText(forRemoteASRProvider: provider),
                isSelectable: configuration.isConfigured,
                disabledReason: configuration.isConfigured ? nil : localized("Configure this provider in Model settings first.")
            )
        })

        return entries
    }

    static func localSelectorAvailability(isInstalled: Bool, isChecking: Bool = false) -> (isSelectable: Bool, disabledReason: String?) {
        (
            isSelectable: isInstalled && !isChecking,
            disabledReason: isChecking ? localized("Loading…") : (isInstalled ? nil : localized("Install this model in Model settings first."))
        )
    }

    private func llmEntries(includeAppleIntelligence: Bool) -> [FeatureModelSelectorEntry] {
        var entries = [FeatureModelSelectorEntry]()
        if includeAppleIntelligence, appleIntelligenceAvailability != .unsupportedOS {
            let isAvailable = appleIntelligenceAvailability.isAvailable
            entries.append(
                FeatureModelSelectorEntry(
                    selectionID: .appleIntelligence,
                    title: localized("Apple Intelligence"),
                    engine: localized("Apple"),
                    sizeText: localized("Built-in"),
                    ratingText: "4.2",
                    filterTags: [localized("Local"), localized("Multilingual"), localized("Installed")] + inUseTags(for: .appleIntelligence),
                    displayTags: featureDisplayTags(
                        base: [localized("Local"), localized("Multilingual")],
                        requiresConfiguration: false,
                        configured: true,
                        selectionID: .appleIntelligence
                    ),
                    statusText: isAvailable
                        ? localized("Available on this Mac")
                        : localized("Unavailable"),
                    usageLocations: usageLabels(for: .appleIntelligence),
                    badgeText: nil,
                    isSelectable: isAvailable,
                    disabledReason: appleIntelligenceAvailability.disabledReason
                )
            )
        }

        entries.append(contentsOf: customLLMDisplayModelsIncludingFeatureSelections().map { model in
            let selectionID = FeatureModelSelectionID.localLLM(model.id)
            let isInstalled = customLLMManager.isModelDownloaded(repo: model.id)
            let isChecking = customLLMManager.isCheckingInstallation(repo: model.id)
            let availability = Self.localSelectorAvailability(isInstalled: isInstalled, isChecking: isChecking)
            return FeatureModelSelectorEntry(
                selectionID: selectionID,
                title: model.title,
                engine: localized("Local LLM"),
                sizeText: customLLMManager.remoteSizeText(repo: model.id),
                ratingText: CustomLLMModelManager.ratingText(for: model.id),
                filterTags: featureFilterTags(
                    base: [localized("Local")] + llmSpeedTags(for: model.id),
                    installed: isInstalled,
                    requiresConfiguration: false,
                    configured: true,
                    usageLabels: usageLabels(for: selectionID)
                ),
                displayTags: featureDisplayTags(
                    base: [localized("Local")] + llmSpeedTags(for: model.id),
                    requiresConfiguration: false,
                    configured: true,
                    selectionID: selectionID
                ),
                statusText: isChecking ? localized("Loading…") : (isInstalled ? localized("Installed") : localized("Not installed")),
                usageLocations: usageLabels(for: selectionID),
                badgeText: nil,
                isSelectable: availability.isSelectable,
                disabledReason: availability.disabledReason
            )
        })

        let remoteConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: remoteLLMProviderConfigurationsRaw,
            sensitiveValueLoading: .metadataOnly
        )
        entries.append(contentsOf: RemoteLLMProvider.allCases.map { provider in
            let selectionID = FeatureModelSelectionID.remoteLLM(provider)
            let isConfigured = RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: provider,
                stored: remoteConfigurations
            )
            let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
                provider: provider,
                stored: remoteConfigurations
            )
            return FeatureModelSelectorEntry(
                selectionID: selectionID,
                title: provider.title,
                engine: localized("Remote LLM"),
                sizeText: isConfigured ? configuration.model : localized("Cloud"),
                ratingText: "4.5",
                filterTags: featureFilterTags(
                    base: [localized("Remote")] + remoteLLMTags(for: provider),
                    installed: false,
                    requiresConfiguration: true,
                    configured: isConfigured,
                    usageLabels: usageLabels(for: selectionID)
                ),
                displayTags: featureDisplayTags(
                    base: [localized("Remote")] + remoteLLMTags(for: provider),
                    requiresConfiguration: true,
                    configured: isConfigured,
                    selectionID: selectionID
                ),
                statusText: isConfigured ? localized("Configured") : localized("Not configured"),
                usageLocations: usageLabels(for: selectionID),
                badgeText: ModelCatalogBadgeSupport.recommendedBadgeText(forRemoteLLMProvider: provider),
                isSelectable: isConfigured,
                disabledReason: isConfigured ? nil : localized("Configure this provider in Model settings first.")
            )
        })

        return entries
    }

    private func customLLMDisplayModelsIncludingFeatureSelections() -> [CustomLLMModelCatalog.Option] {
        var includedRepos = Set(
            customLLMManager.displayModelsIncludingInstalled()
                .map { CustomLLMModelManager.canonicalModelRepo($0.id) }
        )

        func includeLocalLLMSelection(_ selectionID: FeatureModelSelectionID) {
            guard case .localLLM(let repo)? = selectionID.textSelection else { return }
            includedRepos.insert(CustomLLMModelManager.canonicalModelRepo(repo))
        }

        if featureSettings.transcription.llmEnabled {
            includeLocalLLMSelection(featureSettings.transcription.llmSelectionID)
        }
        if featureSettings.transcription.notes.enabled {
            includeLocalLLMSelection(featureSettings.transcription.notes.titleModelSelectionID)
        }
        includeLocalLLMSelection(featureSettings.translation.modelSelectionID)
        includeLocalLLMSelection(featureSettings.rewrite.llmSelectionID)
        includeLocalLLMSelection(featureSettings.meeting.summaryModelSelectionID)

        return CustomLLMModelManager.displayModels(includingInstalled: includedRepos)
    }

    private func translationEntries(
        selectedASR: FeatureModelSelectionID,
        targetLanguage: TranslationTargetLanguage
    ) -> [FeatureModelSelectorEntry] {
        var entries = llmEntries(includeAppleIntelligence: false)
        entries.append(contentsOf: GGUFTranslationModelCatalog.allModels.compactMap { model in
            guard ggufTranslationModelManager.isModelDownloaded(id: model.id) else {
                return nil
            }
            let selectionID = FeatureModelSelectionID.localGGUFTranslation(model.id)
            return FeatureModelSelectorEntry(
                selectionID: selectionID,
                title: model.title,
                engine: localized("Local GGUF"),
                sizeText: model.sizeText,
                ratingText: model.ratingText,
                filterTags: featureFilterTags(
                    base: model.tags,
                    installed: true,
                    requiresConfiguration: false,
                    configured: true,
                    usageLabels: usageLabels(for: selectionID)
                ),
                displayTags: featureDisplayTags(
                    base: model.tags,
                    requiresConfiguration: false,
                    configured: true,
                    selectionID: selectionID
                ),
                statusText: localized("Installed"),
                usageLocations: usageLabels(for: selectionID),
                badgeText: model.badgeText,
                isSelectable: true,
                disabledReason: nil
            )
        })
        return entries
    }

    private func usageLabels(for selectionID: FeatureModelSelectionID) -> [String] {
        var labels = [String]()
        if featureSettings.transcription.asrSelectionID == selectionID ||
            (featureSettings.transcription.llmEnabled && featureSettings.transcription.llmSelectionID == selectionID) {
            labels.append(localized("Transcription"))
        }
        if featureSettings.transcription.notes.enabled &&
            featureSettings.transcription.notes.titleModelSelectionID == selectionID {
            labels.append(localized("Notes"))
        }
        if featureSettings.translation.asrSelectionID == selectionID ||
            featureSettings.translation.modelSelectionID == selectionID {
            labels.append(localized("Translation"))
        }
        if featureSettings.rewrite.asrSelectionID == selectionID ||
            featureSettings.rewrite.llmSelectionID == selectionID {
            labels.append(localized("Rewrite"))
        }
        if featureSettings.meeting.asrSelectionID == selectionID ||
            featureSettings.meeting.summaryModelSelectionID == selectionID {
            labels.append(localized("Meeting"))
        }
        return labels
    }

    private func inUseTags(for selectionID: FeatureModelSelectionID) -> [String] {
        usageLabels(for: selectionID).isEmpty ? [] : [localized("In Use")]
    }

    private func featureFilterTags(
        base: [String],
        installed: Bool,
        requiresConfiguration: Bool,
        configured: Bool,
        usageLabels: [String]
    ) -> [String] {
        var tags = base
        if installed {
            tags.append(localized("Installed"))
        }
        if requiresConfiguration && configured {
            tags.append(localized("Configured"))
        }
        if !usageLabels.isEmpty {
            tags.append(localized("In Use"))
        }
        return deduplicatedFeatureTags(tags)
    }

    private func featureDisplayTags(
        base: [String],
        requiresConfiguration: Bool,
        configured: Bool,
        selectionID: FeatureModelSelectionID
    ) -> [String] {
        var tags = base.filter { $0 != localized("Multilingual") }
        if let languageSupportTag = primaryLanguageSupportTag(for: selectionID) {
            tags.append(languageSupportTag)
        }
        if requiresConfiguration && configured {
            tags.append(localized("Configured"))
        }
        if !usageLabels(for: selectionID).isEmpty {
            tags.append(localized("In Use"))
        }
        return deduplicatedFeatureTags(tags)
    }

    private func mlxSpeedTags(for repo: String) -> [String] {
        deduplicatedFeatureTags(MLXModelManager.catalogTagKeys(for: repo).map(localized))
    }

    private func llmSpeedTags(for repo: String) -> [String] {
        deduplicatedFeatureTags(CustomLLMModelManager.catalogTagKeys(for: repo).map(localized))
    }

    private func remoteASRTags(
        for provider: RemoteASRProvider,
        configuration: RemoteProviderConfiguration
    ) -> [String] {
        var tags = [String]()
        switch provider {
        case .openAIWhisper:
            tags.append(localized("Multilingual"))
        case .doubaoASR:
            tags.append(contentsOf: [localized("Realtime"), localized("Multilingual")])
        case .glmASR:
            tags.append(contentsOf: [localized("Accurate"), localized("Multilingual")])
        case .aliyunBailianASR:
            tags.append(localized("Multilingual"))
            if RemoteASRRealtimeSupport.isAliyunRealtimeModel(configuration.model) {
                tags.append(localized("Realtime"))
            }
        case .stepFunASR:
            if RemoteASRRealtimeSupport.isStepFunRealtimeModel(configuration.model) {
                tags.append(localized("Realtime"))
            }
            tags.append(contentsOf: [localized("Accurate"), localized("Multilingual")])
        case .xiaomiMiMoASR:
            tags.append(contentsOf: [localized("Accurate"), localized("Multilingual")])
        case .googleGeminiASR:
            tags.append(contentsOf: [localized("Realtime"), localized("Accurate"), localized("Multilingual")])
        }
        return deduplicatedFeatureTags(tags)
    }

    private func remoteLLMTags(for provider: RemoteLLMProvider) -> [String] {
        switch provider {
        case .lmStudio, .ollama, .omlx:
            return []
        default:
            return [localized("Accurate")]
        }
    }

    private func primaryLanguageSupportTag(for selectionID: FeatureModelSelectionID) -> String? {
        guard let support = supportsPrimaryLanguage(for: selectionID) else { return nil }
        return localized(support ? "Supports Primary Language" : "Does Not Support Primary Language")
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

    private func deduplicatedFeatureTags(_ tags: [String]) -> [String] {
        Array(NSOrderedSet(array: tags)) as? [String] ?? tags
    }
}
