import SwiftUI
import AppKit

extension OnboardingGuideView {
    private var remoteASRConfigurations: [String: RemoteProviderConfiguration] {
        RemoteModelConfigurationStore.loadConfigurations(
            from: remoteASRProviderConfigurationsRaw,
            sensitiveValueLoading: .metadataOnly
        )
    }

    var remoteLLMConfigurations: [String: RemoteProviderConfiguration] {
        RemoteModelConfigurationStore.loadConfigurations(
            from: remoteLLMProviderConfigurationsRaw,
            sensitiveValueLoading: .metadataOnly
        )
    }

    var modelStorageDialog: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(AppLocalization.localizedString("Model Location"))
                .font(.headline)
            Text(AppLocalization.localizedString("Local ASR and LLM models are stored here. You can move future downloads to another folder."))
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            GeneralSettingsCard(titleText: AppLocalization.localizedString("Model Storage")) {
                SettingsPathSelectionRow(
                    title: AppLocalization.localizedString("Storage Path"),
                    displayedPath: modelStorageDisplayPath,
                    fallbackPath: ModelStorageDirectoryManager.defaultRootURL.path,
                    openButtonHelp: AppLocalization.localizedString("Open folder"),
                    chooseButtonTitle: AppLocalization.localizedString("Choose"),
                    onOpen: openModelStorageInFinder,
                    onChoose: chooseModelStorageDirectory
                )

                if let modelStorageSelectionError, !modelStorageSelectionError.isEmpty {
                    Text(modelStorageSelectionError)
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                Text(AppLocalization.localizedString("New model downloads are stored here. Switching the path will not move existing model files."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .settingsDialogChrome(width: 560, cornerRadius: OnboardingGuideStyle.modalCornerRadius, onClose: {
            isModelStorageDialogPresented = false
        })
    }

    func refreshModelStorageDisplayPath() {
        let resolution = ModelStorageDirectoryManager.resolvedRootResolution()
        modelStorageDisplayPath = resolution.writeRootURL.path
        modelStorageSelectionError = resolution.accessIssue?.localizedDescription
    }

    private func chooseModelStorageDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.directoryURL = ModelStorageDirectoryManager.resolvedRootURL()
        panel.prompt = AppLocalization.localizedString("Choose")

        guard panel.runModal() == .OK, let selectedURL = panel.url else { return }
        do {
            try ModelStorageDirectoryManager.saveUserSelectedRootURL(selectedURL)
            modelStorageSelectionError = nil
            mlxModelManager.refreshStorageRoot()
            customLLMManager.refreshStorageRoot()
            refreshModelStorageDisplayPath()
            SileroVADModelProvisioner.prefetchIfNeeded(for: LocalVADMode.stored())
        } catch {
            modelStorageSelectionError = AppLocalization.format("Failed to update model storage path: %@", error.localizedDescription)
        }
    }

    private func openModelStorageInFinder() {
        Task { @MainActor in
            ModelStorageDirectoryManager.openRootInFinder()
        }
    }

    func isSpeechModelReady(_ selection: FeatureModelSelectionID) -> Bool {
        switch selection.asrSelection {
        case .mlx(let repo): return mlxModelManager.isModelDownloaded(repo: repo)
        case .remote(let provider): return isRemoteASRConfigured(provider)
        case .dictation: return isPermissionGranted(.speechRecognition)
        case .none: return false
        }
    }

    func isTranslationModelReady(_ selection: FeatureModelSelectionID) -> Bool {
        switch selection.translationSelection {
        case .localLLM(let repo): return customLLMManager.isModelDownloaded(repo: repo)
        case .remoteLLM(let provider):
            return RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(provider: provider, stored: remoteLLMConfigurations)
        case .localGGUF(let id):
            return AppDelegate.shared?.ggufTranslationModelManager.isModelDownloaded(id: id) == true
        case .none: return false
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

    func isRemoteASRConfigured(_ provider: RemoteASRProvider) -> Bool {
        RemoteModelConfigurationStore.resolvedASRConfiguration(
            provider: provider,
            stored: remoteASRConfigurations
        )
        .isConfigured
    }

    func mlxDownloadStatus(for repo: String) -> ModelDownloadStatusSnapshot? {
        guard mlxModelManager.isDownloading(repo: repo) || mlxModelManager.isPaused(repo: repo) else { return nil }
        return ModelDownloadStatusSnapshot.fromMLXState(
            mlxModelManager.state(for: repo),
            pauseMessage: mlxModelManager.pausedStatusMessage(for: repo)
        )
    }

    func customLLMDownloadStatus(for repo: String) -> ModelDownloadStatusSnapshot? {
        switch customLLMManager.state(for: repo) {
        case .downloading, .paused:
            return ModelDownloadStatusSnapshot.fromCustomLLMState(
                customLLMManager.state(for: repo),
                pauseMessage: customLLMManager.pausedStatusMessage(for: repo)
            )
        default:
            return nil
        }
    }

    func mlxDownloadErrorMessage(for repo: String) -> String? {
        guard case .error(let message) = mlxModelManager.state(for: repo)
        else { return nil }
        return message
    }

    func customLLMDownloadErrorMessage(for repo: String) -> String? {
        guard case .error(let message) = customLLMManager.state(for: repo)
        else { return nil }
        return message
    }

    func asrSelectionSummary(_ selectionID: FeatureModelSelectionID) -> String {
        switch selectionID.asrSelection {
        case .dictation:
            return AppLocalization.localizedString("Direct Dictation")
        case .mlx(let repo):
            return mlxModelManager.displayTitle(for: repo)
        case .remote(let provider):
            let configuration = RemoteModelConfigurationStore.resolvedASRConfiguration(provider: provider, stored: remoteASRConfigurations)
            if configuration.hasUsableModel {
                return "\(provider.title) · \(configuration.model)"
            }
            return "\(provider.title) · \(AppLocalization.localizedString("Needs Setup"))"
        case .none:
            return AppLocalization.localizedString("Not selected")
        }
    }

    private func llmSelectionSummary(_ selectionID: FeatureModelSelectionID) -> String {
        switch selectionID.textSelection {
        case .appleIntelligence:
            return AppLocalization.localizedString("Apple Intelligence")
        case .localLLM(let repo):
            return customLLMManager.displayTitle(for: repo)
        case .remoteLLM(let provider):
            guard RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: provider,
                stored: remoteLLMConfigurations
            ) else {
                return "\(provider.title) · \(AppLocalization.localizedString("Needs Setup"))"
            }
            let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(provider: provider, stored: remoteLLMConfigurations)
            return "\(provider.title) · \(configuration.model)"
        case .none:
            return AppLocalization.localizedString("Not selected")
        }
    }

    func translationSelectionSummary(_ selectionID: FeatureModelSelectionID) -> String {
        switch selectionID.translationSelection {
        case .localGGUF(let modelID):
            return GGUFTranslationModelCatalog.option(for: modelID).title
        case .localLLM, .remoteLLM:
            return llmSelectionSummary(selectionID)
        case .none:
            return AppLocalization.localizedString("Not selected")
        }
    }
}
