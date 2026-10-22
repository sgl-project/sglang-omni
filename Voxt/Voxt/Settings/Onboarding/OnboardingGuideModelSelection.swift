import SwiftUI
import AppKit

extension OnboardingGuideView {
    private var selectedRemoteASRProvider: RemoteASRProvider {
        if case .remote(let provider)? = selectedSpeechModel.asrSelection { return provider }
        return RemoteASRProvider(rawValue: remoteASRSelectedProviderRaw) ?? .openAIWhisper
    }

    private var selectedRemoteLLMProvider: RemoteLLMProvider {
        if case .remoteLLM(let provider)? = selectedTranslationModel.translationSelection { return provider }
        return RemoteLLMProvider(rawValue: remoteLLMSelectedProviderRaw) ?? .openAI
    }

    private var selectedLocalSpeechRepo: String {
        if case .mlx(let repo)? = selectedSpeechModel.asrSelection { return MLXModelManager.canonicalModelRepo(repo) }
        return MLXModelManager.canonicalModelRepo(mlxModelRepo)
    }

    private var selectedLocalTranslationRepo: String {
        if case .localLLM(let repo)? = selectedTranslationModel.translationSelection { return CustomLLMModelManager.canonicalModelRepo(repo) }
        return CustomLLMModelManager.canonicalModelRepo(customLLMRepo)
    }

    private var localASRRepos: [String] {
        var repos = mlxModelManager.displayModelsIncludingInstalled()
            .map { MLXModelManager.canonicalModelRepo($0.id) }
        let selectedRepo = selectedLocalSpeechRepo
        if !repos.contains(selectedRepo) {
            repos.insert(selectedRepo, at: 0)
        }
        return repos
    }

    private var localLLMRepos: [String] {
        var repos = customLLMManager.displayModelsIncludingInstalled()
            .map { CustomLLMModelManager.canonicalModelRepo($0.id) }
        let selectedRepo = selectedLocalTranslationRepo
        if !repos.contains(selectedRepo) {
            repos.insert(selectedRepo, at: 0)
        }
        return repos
    }

    private var defaultLocalASRRepos: [String] {
        let selectedRepo = selectedLocalSpeechRepo
        return collapsedModelOptions(
            all: localASRRepos,
            preferred: [selectedRepo] + Self.preferredLocalASRRepos
        )
    }

    private var defaultLocalLLMRepos: [String] {
        let selectedRepo = selectedLocalTranslationRepo
        return collapsedModelOptions(
            all: localLLMRepos,
            preferred: [selectedRepo] + Self.preferredLocalLLMRepos
        )
    }

    private var displayedLocalASRRepos: [String] {
        showsMoreLocalASRModels ? localASRRepos : defaultLocalASRRepos
    }

    var displayedLocalLLMRepos: [String] {
        showsMoreLocalLLMModels ? localLLMRepos : defaultLocalLLMRepos
    }

    private var displayedRemoteASRProviders: [RemoteASRProvider] {
        if showsMoreRemoteASRProviders {
            return RemoteASRProvider.allCases
        }
        return defaultRemoteASRProviders
    }

    var displayedRemoteLLMProviders: [RemoteLLMProvider] {
        if showsMoreRemoteLLMProviders {
            return RemoteLLMProvider.allCases
        }
        return defaultRemoteLLMProviders
    }

    private var defaultRemoteASRProviders: [RemoteASRProvider] {
        collapsedModelOptions(
            all: RemoteASRProvider.allCases,
            preferred: [selectedRemoteASRProvider] + Self.preferredRemoteASRProviders
        )
    }

    private var defaultRemoteLLMProviders: [RemoteLLMProvider] {
        collapsedModelOptions(
            all: RemoteLLMProvider.allCases,
            preferred: [selectedRemoteLLMProvider] + Self.preferredRemoteLLMProviders
        )
    }

    private func collapsedModelOptions<Option: Equatable>(
        all options: [Option],
        preferred: [Option]
    ) -> [Option] {
        var result: [Option] = []
        for option in preferred + options where options.contains(option) && !result.contains(option) {
            result.append(option)
            if result.count == Self.collapsedModelListLimit {
                break
            }
        }
        return result
    }

    private var selectedSpeechModel: FeatureModelSelectionID {
        modelDraft.speech ?? featureSettings.transcription.asrSelectionID
    }

    var selectedTranslationModel: FeatureModelSelectionID {
        modelDraft.translation ?? featureSettings.translation.modelSelectionID
    }

    var modelStepReady: Bool { isSpeechModelReady(selectedSpeechModel) }

    var modelGuidePanel: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                Text(AppLocalization.localizedString("Selections are applied when you continue. Other feature settings stay unchanged."))
                    .font(.caption).foregroundStyle(.secondary)
                modelSelectionContent
            }
                .padding(12)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            modelFooter
                .padding(.horizontal, 12)
                .padding(.bottom, 10)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var modelSelectionContent: some View {
        VStack(spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                modelTabItem(
                    focus: .local,
                    title: AppLocalization.localizedString("Local"),
                    subtitle: AppLocalization.localizedString("Private and offline after download. Only a speech model is required.")
                )

                modelTabItem(
                    focus: .remote,
                    title: AppLocalization.localizedString("Remote"),
                    subtitle: AppLocalization.localizedString("Requires internet and your provider credentials. Audio is sent to your chosen provider.")
                )
            }

            ScrollView {
                Group {
                    switch modelFocus {
                    case .local:
                        localModelActions
                    case .remote:
                        remoteModelActions
                    }
                }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(
                RoundedRectangle(cornerRadius: SettingsUIStyle.panelCornerRadius, style: .continuous)
                    .fill(SettingsUIStyle.panelFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: SettingsUIStyle.panelCornerRadius, style: .continuous)
                    .strokeBorder(SettingsUIStyle.panelBorderColor, lineWidth: 1)
            )
        }
        .animation(.easeInOut(duration: 0.18), value: modelFocus)
    }

    private func modelTabItem(
        focus: OnboardingGuideModelFocus,
        title: String,
        subtitle: String
    ) -> some View {
        let isActive = modelFocus == focus
        return Button {
            modelFocus = focus
        } label: {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 5) {
                    Text(title)
                        .font(.headline)
                        .foregroundStyle(OnboardingGuideStyle.primaryText)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(OnboardingGuideStyle.secondaryText)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer(minLength: 8)

                if isActive {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(Color.accentColor)
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, minHeight: 72, alignment: .topLeading)
            .background(
                RoundedRectangle(cornerRadius: SettingsUIStyle.panelCornerRadius, style: .continuous)
                    .fill(isActive ? Color.accentColor.opacity(0.08) : SettingsUIStyle.panelFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: SettingsUIStyle.panelCornerRadius, style: .continuous)
                    .strokeBorder(isActive ? Color.accentColor.opacity(0.35) : SettingsUIStyle.panelBorderColor, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }

    private var localModelActions: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 8) {
                Text(AppLocalization.localizedString("Speech-to-text Model"))
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(displayedLocalASRRepos, id: \.self) { repo in
                    localASRModelRow(repo: repo)
                }
                if localASRRepos.count > defaultLocalASRRepos.count {
                    moreListButton(
                        isExpanded: showsMoreLocalASRModels,
                        expandedCount: localASRRepos.count,
                        collapsedCount: defaultLocalASRRepos.count
                    ) {
                        showsMoreLocalASRModels.toggle()
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .topLeading)

            VStack(alignment: .leading, spacing: 8) {
                Text(AppLocalization.localizedString("Translation (Optional)"))
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(displayedLocalLLMRepos, id: \.self) { repo in
                    localLLMModelRow(repo: repo)
                }
                if localLLMRepos.count > defaultLocalLLMRepos.count {
                    moreListButton(
                        isExpanded: showsMoreLocalLLMModels,
                        expandedCount: localLLMRepos.count,
                        collapsedCount: defaultLocalLLMRepos.count
                    ) {
                        showsMoreLocalLLMModels.toggle()
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private var remoteModelActions: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 8) {
                Text(AppLocalization.localizedString("Speech-to-text Model"))
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(displayedRemoteASRProviders) { provider in
                    remoteProviderRow(
                        title: provider.title,
                        isSelected: selectedSpeechModel == .remoteASR(provider),
                        isConfigured: isRemoteASRConfigured(provider),
                        onSelect: {
                            modelDraft.speech = .remoteASR(provider)
                        },
                        onConfigure: {
                            editingASRProvider = provider
                        }
                    )
                }
                if RemoteASRProvider.allCases.count > defaultRemoteASRProviders.count {
                    moreListButton(
                        isExpanded: showsMoreRemoteASRProviders,
                        expandedCount: RemoteASRProvider.allCases.count,
                        collapsedCount: defaultRemoteASRProviders.count
                    ) {
                        showsMoreRemoteASRProviders.toggle()
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .topLeading)

            VStack(alignment: .leading, spacing: 8) {
                Text(AppLocalization.localizedString("Translation (Optional)"))
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(displayedRemoteLLMProviders) { provider in
                    remoteProviderRow(
                        title: onboardingRemoteLLMProviderTitle(provider),
                        isSelected: selectedTranslationModel == .remoteLLM(provider),
                        isConfigured: RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                            provider: provider,
                            stored: remoteLLMConfigurations
                        ),
                        onSelect: {
                            modelDraft.translation = .remoteLLM(provider)
                        },
                        onConfigure: {
                            editingLLMProvider = provider
                        }
                    )
                }
                if RemoteLLMProvider.allCases.count > defaultRemoteLLMProviders.count {
                    moreListButton(
                        isExpanded: showsMoreRemoteLLMProviders,
                        expandedCount: RemoteLLMProvider.allCases.count,
                        collapsedCount: defaultRemoteLLMProviders.count
                    ) {
                        showsMoreRemoteLLMProviders.toggle()
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private var modelFooter: some View {
        HStack(spacing: 10) {
            Spacer(minLength: 0)

            if modelFocus == .local {
                Button(AppLocalization.localizedString("Model Location")) {
                    isModelStorageDialogPresented = true
                }
                .buttonStyle(OnboardingGuideSecondaryButtonStyle())
            }

            Button {
                advanceStep()
            } label: {
                Label(AppLocalization.localizedString("Continue"), systemImage: "chevron.right")
                    .labelStyle(OnboardingGuideNextLabelStyle())
            }
            .buttonStyle(OnboardingGuidePrimaryButtonStyle())
            .disabled(!canContinue)
            .help(canContinue ? "" : continueDisabledHelp)

            Spacer(minLength: 0)
        }
        .padding(.top, 10)
    }

    private func localASRModelRow(repo: String) -> some View {
        localModelRow(
            title: mlxModelManager.displayTitle(for: repo),
            repo: repo,
            sizeText: mlxModelManager.remoteSizeText(repo: repo),
            ratingText: MLXModelManager.ratingText(for: repo),
            isSelected: selectedSpeechModel == .mlx(MLXModelManager.canonicalModelRepo(repo)),
            isInstalled: mlxModelManager.isModelDownloaded(repo: repo),
            isPaused: mlxModelManager.isPaused(repo: repo),
            status: mlxDownloadStatus(for: repo),
            errorMessage: mlxDownloadErrorMessage(for: repo),
            onSelect: {
                modelDraft.speech = .mlx(MLXModelManager.canonicalModelRepo(repo))
            },
            onInstall: {
                let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
                modelDraft.speech = .mlx(canonicalRepo)
                Task { await mlxModelManager.downloadModel(repo: canonicalRepo) }
            },
            onPause: { mlxModelManager.pauseDownload(repo: repo) },
            onCancel: { mlxModelManager.cancelDownload(repo: repo) }
        )
    }

    func localLLMModelRow(repo: String) -> some View {
        localModelRow(
            title: customLLMManager.displayTitle(for: repo),
            repo: repo,
            sizeText: customLLMManager.remoteSizeText(repo: repo),
            ratingText: CustomLLMModelManager.ratingText(for: repo),
            isSelected: selectedTranslationModel == .localLLM(CustomLLMModelManager.canonicalModelRepo(repo)),
            isInstalled: customLLMManager.isModelDownloaded(repo: repo),
            isPaused: customLLMManager.isPaused(repo: repo),
            status: customLLMDownloadStatus(for: repo),
            errorMessage: customLLMDownloadErrorMessage(for: repo),
            onSelect: {
                modelDraft.translation = .localLLM(CustomLLMModelManager.canonicalModelRepo(repo))
            },
            onInstall: {
                let canonicalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
                modelDraft.translation = .localLLM(canonicalRepo)
                Task { await customLLMManager.downloadModel(repo: canonicalRepo) }
            },
            onPause: { customLLMManager.pauseDownload(repo: repo) },
            onCancel: { customLLMManager.cancelDownload(repo: repo) }
        )
    }
}
