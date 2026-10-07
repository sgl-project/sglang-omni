// ModelSettingsView.swift
// Provides Model Settings View for model settings.

import SwiftUI
import AppKit
import Combine

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct ModelSettingsView: View {
    @AppStorage(AppPreferenceKey.transcriptionEngine) var engineRaw = TranscriptionEngine.mlxAudio.rawValue
    @AppStorage(AppPreferenceKey.enhancementMode) var enhancementModeRaw = EnhancementMode.off.rawValue
    @AppStorage(AppPreferenceKey.enhancementSystemPrompt) var systemPrompt = ""
    @AppStorage(AppPreferenceKey.translationSystemPrompt) var translationPrompt = ""
    @AppStorage(AppPreferenceKey.rewriteSystemPrompt) var rewritePrompt = ""
    @AppStorage(AppPreferenceKey.mlxModelRepo) var modelRepo = MLXModelManager.defaultModelRepo
    @AppStorage(AppPreferenceKey.localModelIdleUnloadDelaySeconds)
    var localModelIdleUnloadDelaySeconds = AppPreferenceKey.defaultLocalModelIdleUnloadDelaySeconds
    @AppStorage(AppPreferenceKey.customLLMModelRepo) var customLLMRepo = CustomLLMModelManager.defaultModelRepo
    @AppStorage(AppPreferenceKey.customLLMGenerationSettings) var customLLMGenerationSettingsRaw = CustomLLMGenerationSettingsStore.defaultStoredValue()
    @AppStorage(AppPreferenceKey.customLLMGenerationSettingsByRepo) var customLLMGenerationSettingsByRepoRaw = CustomLLMGenerationSettingsStore.defaultByRepoStoredValue()
    @AppStorage(AppPreferenceKey.translationCustomLLMModelRepo) var translationCustomLLMRepo = CustomLLMModelManager.defaultModelRepo
    @AppStorage(AppPreferenceKey.translationGGUFModelID) var translationGGUFModelIDRaw = GGUFTranslationModelCatalog.defaultModelID.rawValue
    @AppStorage(AppPreferenceKey.rewriteCustomLLMModelRepo) var rewriteCustomLLMRepo = CustomLLMModelManager.defaultModelRepo
    @AppStorage(AppPreferenceKey.translationModelProvider) var translationModelProviderRaw = TranslationModelProvider.customLLM.rawValue
    @AppStorage(AppPreferenceKey.translationFallbackModelProvider) var translationFallbackModelProviderRaw = TranslationModelProvider.customLLM.rawValue
    @AppStorage(AppPreferenceKey.rewriteModelProvider) var rewriteModelProviderRaw = RewriteModelProvider.customLLM.rawValue
    @AppStorage(AppPreferenceKey.translationTargetLanguage) var translationTargetLanguageRaw = TranslationTargetLanguage.english.rawValue
    @AppStorage(AppPreferenceKey.remoteASRSelectedProvider) var remoteASRSelectedProviderRaw = RemoteASRProvider.openAIWhisper.rawValue
    @AppStorage(AppPreferenceKey.remoteASRProviderConfigurations) var remoteASRProviderConfigurationsRaw = ""
    @AppStorage(AppPreferenceKey.asrHintSettings) var asrHintSettingsRaw = ASRHintSettingsStore.defaultStoredValue()
    @AppStorage(AppPreferenceKey.mlxLocalASRTuningSettings) var mlxLocalASRTuningSettingsRaw = "{}"
    @AppStorage(AppPreferenceKey.userMainLanguageCodes) var userMainLanguageCodesRaw = UserMainLanguageOption.defaultStoredSelectionValue
    @AppStorage(AppPreferenceKey.remoteLLMSelectedProvider) var remoteLLMSelectedProviderRaw = RemoteLLMProvider.openAI.rawValue
    @AppStorage(AppPreferenceKey.remoteLLMProviderConfigurations) var remoteLLMProviderConfigurationsRaw = ""
    @AppStorage(AppPreferenceKey.translationRemoteLLMProvider) var translationRemoteLLMProviderRaw = ""
    @AppStorage(AppPreferenceKey.rewriteRemoteLLMProvider) var rewriteRemoteLLMProviderRaw = ""
    @AppStorage(AppPreferenceKey.modelStorageRootPath) var modelStorageRootPath = ""
    @AppStorage(AppPreferenceKey.interfaceLanguage) var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @AppStorage(AppPreferenceKey.featureSettings) var featureSettingsRaw = ""
    @AppStorage(AppPreferenceKey.hiddenModelVisibilityIDs) var hiddenModelVisibilityIDsRaw = ""

    let mlxModelManager: MLXModelManager
    let customLLMManager: CustomLLMModelManager
    let ggufTranslationModelManager: GGUFTranslationModelManager
    @ObservedObject var mainWindowState: MainWindowVisibilityState
    let missingConfigurationIssues: [ModelConfigurationIssue]
    let navigationRequest: SettingsNavigationRequest?
    let isActive: Bool

    @State var catalogTab: ModelCatalogTab = .asr
    @State var selectedTags = Set<String>()
    @State var cachedFeatureSettings = FeatureSettings.placeholder
    @State var cachedRemoteASRConfigurations = [String: RemoteProviderConfiguration]()
    @State var cachedRemoteLLMConfigurations = [String: RemoteProviderConfiguration]()
    @State private var modelStorageDisplayPath = ""
    @State private var modelStorageSelectionError: String?
    @State private var showIdleUnloadDelayInfo = false
    @State var editingASRProvider: RemoteASRProvider?
    @State var editingLLMProvider: RemoteLLMProvider?
    @State private var activeASRHintTarget: ASRHintTarget?
    @State var activeLocalASRConfigurationTarget: LocalASRConfigurationTarget?
    @State var isCustomLLMConfigurationPresented = false
    @State var customLLMConfigurationRepo: String?
    @State private var isModelDownloadSettingsPresented = false
    @State private var expandedModelGroupIDs = Set<String>()
    @State private var collapsedModelGroupIDs = Set<String>()
    @State var catalogSnapshot = ModelSettingsCatalogSnapshot.empty
    @State private var isCatalogRefreshScheduled = false
    @State private var isRefreshingCatalogSnapshot = false
    @State private var needsAnotherCatalogRefresh = false
    @State var lastHandledDownloadLifecycleToken: ModelSettingsDownloadLifecycleToken?
    @State var pendingModelRemovalTarget: LocalModelRemovalTarget?
    @State var uninstallingModelTarget: LocalModelRemovalTarget?
    @State var cancellingInstallTargets = Set<LocalModelInstallTarget>()
    @State private var lastHandledConfigurationNavigationRequestID: UUID?
    @State private var lastHandledStorageAuthorizationNavigationRequestID: UUID?
    @State private var pendingStorageAuthorizationPickerRequestID: UUID?
    @State private var modelOperationToastMessage = ""
    @State private var modelOperationToastDismissTask: Task<Void, Never>?
    @State private var presentedModelErrorMessagesByTarget: [String: String] = [:]
    @State var appleIntelligenceRefreshRevision = 0

    let modelStateRefreshTimer = Timer.publish(every: 2.0, on: .main, in: .common).autoconnect()

    var selectedEngine: TranscriptionEngine {
        TranscriptionEngine(rawValue: engineRaw) ?? .mlxAudio
    }

    var selectedEnhancementMode: EnhancementMode {
        EnhancementMode.resolved(
            storedRawValue: enhancementModeRaw,
            appleIntelligenceAvailable: appleIntelligenceAvailable,
            customLLMAvailable: customEnhancementModelAvailable,
            remoteLLMAvailable: remoteEnhancementModelAvailable
        )
    }

    var selectedRemoteASRProvider: RemoteASRProvider {
        RemoteASRProvider(rawValue: remoteASRSelectedProviderRaw) ?? .openAIWhisper
    }

    var selectedRemoteLLMProvider: RemoteLLMProvider {
        RemoteLLMProvider(rawValue: remoteLLMSelectedProviderRaw) ?? .openAI
    }

    var selectedTranslationModelProvider: TranslationModelProvider {
        TranslationModelProvider(rawValue: translationModelProviderRaw) ?? .customLLM
    }

    var selectedRewriteModelProvider: RewriteModelProvider {
        RewriteModelProvider(rawValue: rewriteModelProviderRaw) ?? .customLLM
    }

    var selectedTranslationFallbackModelProvider: TranslationModelProvider {
        TranslationProviderResolver.sanitizedFallbackProvider(
            TranslationModelProvider(rawValue: translationFallbackModelProviderRaw) ?? .customLLM
        )
    }

    var selectedTranslationTargetLanguage: TranslationTargetLanguage {
        TranslationTargetLanguage(rawValue: translationTargetLanguageRaw) ?? .english
    }

    var remoteASRConfigurations: [String: RemoteProviderConfiguration] {
        cachedRemoteASRConfigurations
    }

    var remoteLLMConfigurations: [String: RemoteProviderConfiguration] {
        cachedRemoteLLMConfigurations
    }

    var selectedUserLanguageCodes: [String] {
        UserMainLanguageOption.storedSelection(from: userMainLanguageCodesRaw)
    }

    var appleIntelligenceAvailable: Bool {
        appleIntelligenceAvailability.isAvailable
    }

    var appleIntelligenceAvailability: AppleIntelligenceAvailability {
        _ = appleIntelligenceRefreshRevision
        return AppleIntelligenceAvailability.current
    }

    var customEnhancementModelAvailable: Bool {
        customLLMManager.isModelDownloaded(repo: customLLMManager.currentModelRepo)
    }

    var remoteEnhancementModelAvailable: Bool {
        RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
            provider: selectedRemoteLLMProvider,
            stored: remoteLLMConfigurations
        )
    }

    private var featureSettings: FeatureSettings {
        cachedFeatureSettings
    }

    private var catalogBuilder: ModelCatalogBuilder {
        ModelCatalogBuilder(
            mlxModelManager: mlxModelManager,
            customLLMManager: customLLMManager,
            ggufTranslationModelManager: ggufTranslationModelManager,
            remoteASRConfigurations: remoteASRConfigurations,
            remoteLLMConfigurations: remoteLLMConfigurations,
            featureSettings: featureSettings,
            hasIssue: hasIssue(for:),
            customLLMBadgeText: customLLMBadgeText(for:),
            remoteASRStatusText: { provider, configuration in
                remoteASRStatusText(for: provider, configuration: configuration)
            },
            remoteLLMBadgeText: remoteLLMBadgeText(for:),
            primaryUserLanguageCode: selectedUserLanguageCodes.first,
            appleIntelligenceAvailability: appleIntelligenceAvailability,
            mlxInstallSnapshot: mlxInstallSnapshot(for:),
            customLLMInstallSnapshot: customLLMInstallSnapshot(for:),
            ggufTranslationInstallSnapshot: ggufTranslationInstallSnapshot(for:),
            catalogPrimaryAction: {
                ModelSettingsInstallActionResolver.catalogPrimaryAction(
                    for: $0,
                    perform: performInstallAction(_:kind:)
                )
            },
            catalogSecondaryActions: {
                ModelSettingsInstallActionResolver.catalogSecondaryActions(
                    for: $0,
                    perform: performInstallAction(_:kind:)
                )
            },
            configureASRProvider: { editingASRProvider = $0 },
            configureLLMProvider: { editingLLMProvider = $0 },
            showASRHintTarget: { activeASRHintTarget = $0 }
        )
    }

    private var allEntries: [ModelCatalogEntry] { catalogSnapshot.allEntries }

    private var hiddenModelVisibilityIDs: Set<String> {
        ModelVisibilityStore.decode(hiddenModelVisibilityIDsRaw)
    }

    private var availableTags: [String] { catalogSnapshot.availableTags }

    private var availableTagGroups: [[String]] { catalogSnapshot.availableTagGroups }

    private var filteredEntries: [ModelCatalogEntry] { catalogSnapshot.filteredEntries }

    private var displayItems: [ModelCatalogDisplayItem] { catalogSnapshot.displayItems }

    private var tagFilterBar: some View {
        Group {
            if !availableTags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(availableTagGroups.enumerated()), id: \.offset) { index, group in
                            HStack(spacing: 8) {
                                ForEach(group, id: \.self) { tag in
                                    ModelTagChip(
                                        title: tag,
                                        isSelected: selectedTags.contains(tag),
                                        action: { toggleTag(tag) }
                                    )
                                }
                            }

                            if index < availableTagGroups.count - 1 {
                                Rectangle()
                                    .fill(SettingsUIStyle.subtleBorderColor.opacity(0.95))
                                    .frame(width: 1, height: 20)
                                    .padding(.horizontal, 4)
                            }
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    private var modelCatalogContent: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                if filteredEntries.isEmpty {
                    ModelEmptyStateView()
                } else {
                    ForEach(displayItems) { item in
                        modelCatalogItemView(item)
                    }
                }
            }
            .padding(.vertical, 2)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func modelCatalogItemView(_ item: ModelCatalogDisplayItem) -> some View {
        switch item {
        case .row(let entry):
            ModelCatalogRow(
                entry: entry,
                visibilityAction: visibilityAction(for: entry)
            )
        case .group(let group):
            ModelCatalogGroupCard(
                group: group,
                isExpanded: isModelGroupExpanded(group),
                onToggle: { toggleModelGroup(group) },
                visibilityActionForEntry: { visibilityAction(for: $0) }
            )
        }
    }

    private func visibilityAction(for entry: ModelCatalogEntry) -> ModelTableAction? {
        guard !entry.filterTags.contains(localized("Installed")) else { return nil }
        let isHidden = ModelVisibilityStore.isModelHidden(entry.id, in: hiddenModelVisibilityIDs)
        if isHidden {
            return ModelTableAction(title: localized("Show")) {
                showModel(entry.id)
            }
        }
        return ModelTableAction(title: localized("Hide")) {
            hideModel(entry.id)
        }
    }

    private func updateHiddenModelVisibility(_ update: (inout Set<String>) -> Void) {
        var values = hiddenModelVisibilityIDs
        update(&values)
        hiddenModelVisibilityIDsRaw = ModelVisibilityStore.encode(values)
    }

    private func hideModel(_ id: String) {
        updateHiddenModelVisibility { $0.insert(ModelVisibilityStore.modelKey(id)) }
    }

    private func showModel(_ id: String) {
        updateHiddenModelVisibility { $0.remove(ModelVisibilityStore.modelKey(id)) }
    }

    private func isModelGroupExpanded(_ group: ModelCatalogGroupSection) -> Bool {
        if expandedModelGroupIDs.contains(group.id) {
            return true
        }
        if collapsedModelGroupIDs.contains(group.id) {
            return false
        }
        return group.defaultExpanded
    }

    private func toggleModelGroup(_ group: ModelCatalogGroupSection) {
        let isExpanded = isModelGroupExpanded(group)
        if group.defaultExpanded {
            if isExpanded {
                collapsedModelGroupIDs.insert(group.id)
            } else {
                collapsedModelGroupIDs.remove(group.id)
            }
            expandedModelGroupIDs.remove(group.id)
            return
        }

        if isExpanded {
            expandedModelGroupIDs.remove(group.id)
        } else {
            expandedModelGroupIDs.insert(group.id)
        }
        collapsedModelGroupIDs.remove(group.id)
    }

    var mainContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            modelTabHeader
            tagFilterBar
            modelCatalogContent
        }
    }

    private var contentWithSheets: some View {
        contentWithLifecycle
        .sheet(item: $editingASRProvider) { provider in
            RemoteProviderConfigurationSheet(
                providerTitle: provider.title,
                credentialHint: asrCredentialHint(for: provider),
                showsDoubaoFields: provider == .doubaoASR,
                testTarget: .asr(provider),
                configuration: RemoteModelConfigurationStore.resolvedASRConfiguration(
                    provider: provider,
                    from: remoteASRProviderConfigurationsRaw
                )
            ) { updated in
                let result = saveRemoteASRConfiguration(updated)
                if case .success = result {
                    showModelOperationToast(
                        AppLocalization.format("Configuration saved for %@.", provider.title)
                    )
                }
                return result
            }
        }
        .sheet(item: $editingLLMProvider) { provider in
            RemoteProviderConfigurationSheet(
                providerTitle: provider.title,
                credentialHint: nil,
                showsDoubaoFields: false,
                testTarget: .llm(provider),
                configuration: RemoteModelConfigurationStore.resolvedLLMConfiguration(
                    provider: provider,
                    from: remoteLLMProviderConfigurationsRaw
                )
            ) { updated in
                let result = saveRemoteLLMConfiguration(updated)
                if case .success = result {
                    showModelOperationToast(
                        AppLocalization.format("Configuration saved for %@.", provider.title)
                    )
                }
                return result
            }
        }
        .sheet(item: $activeASRHintTarget) { target in
            ASRHintSettingsSheet(
                target: target,
                userLanguageCodes: selectedUserLanguageCodes,
                mlxModelRepo: target == .mlxAudio ? modelRepo : nil,
                initialSettings: resolvedASRHintSettings(for: target)
            ) { updated in
                saveASRHintSettings(updated, for: target)
            }
        }
        .sheet(item: $activeLocalASRConfigurationTarget) { target in
            localASRConfigurationSheet(for: target)
        }
        .sheet(isPresented: $isCustomLLMConfigurationPresented) {
            let repo = customLLMConfigurationRepo ?? customLLMRepo
            CustomLLMGenerationSettingsSheet(
                modelTitle: customLLMManager.displayTitle(for: repo),
                settings: customLLMGenerationSettingsBinding(for: repo)
            ) {
                isCustomLLMConfigurationPresented = false
                customLLMConfigurationRepo = nil
            }
        }
        .sheet(isPresented: $isModelDownloadSettingsPresented) {
            modelDownloadSettingsSheet
                .onAppear(perform: presentPendingStorageAuthorizationPicker)
        }
        .alert(item: $pendingModelRemovalTarget) { target in
            Alert(
                title: Text(AppLocalization.localizedString("Uninstall Model?")),
                message: Text(uninstallConfirmationMessage(for: target)),
                primaryButton: .destructive(Text(AppLocalization.localizedString("Uninstall"))) {
                    confirmDeleteModel(target)
                },
                secondaryButton: .cancel(Text(AppLocalization.localizedString("Cancel")))
            )
        }
    }

    var body: some View {
        contentWithSheets
        .overlay(alignment: .top) {
            if !modelOperationToastMessage.isEmpty {
                ModelDebugToast(message: modelOperationToastMessage) {
                    dismissModelOperationToast()
                }
                .padding(.top, 12)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.16), value: modelOperationToastMessage)
        .id(interfaceLanguageRaw)
        .onAppear(perform: handleConfigurationNavigationRequest)
        .onAppear(perform: handleStorageAuthorizationNavigationRequest)
        .onChange(of: navigationRequest?.id) { _, _ in
            handleConfigurationNavigationRequest()
            handleStorageAuthorizationNavigationRequest()
        }
        .onChange(of: isActive) { _, isActive in
            guard isActive else { return }
            handleStorageAuthorizationNavigationRequest()
        }
        .onReceive(ggufTranslationModelManager.$stateByID) { states in
            presentGGUFErrors(states)
        }
    }

    func showModelOperationToast(_ message: String, duration: Duration = .seconds(4)) {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        modelOperationToastDismissTask?.cancel()
        modelOperationToastMessage = trimmed
        modelOperationToastDismissTask = Task { @MainActor in
            try? await Task.sleep(for: duration)
            guard !Task.isCancelled else { return }
            modelOperationToastMessage = ""
        }
    }

    private func dismissModelOperationToast() {
        modelOperationToastDismissTask?.cancel()
        modelOperationToastMessage = ""
    }

    private func presentGGUFErrors(_ states: [GGUFTranslationModelID: GGUFTranslationModelManager.ModelState]) {
        for (modelID, state) in states {
            let targetID = "gguf-translation:\(modelID.rawValue)"
            if case .error(let message) = state {
                presentModelError(
                    targetID: targetID,
                    modelName: ggufTranslationModelManager.displayTitle(for: modelID),
                    message: message
                )
            } else {
                clearPresentedModelError(targetID: targetID)
            }
        }
    }

    func presentModelError(targetID: String, modelName: String, message: String) {
        guard presentedModelErrorMessagesByTarget[targetID] != message else { return }
        presentedModelErrorMessagesByTarget[targetID] = message
        showModelOperationToast(
            AppLocalization.format("%@: %@", modelName, localizedModelOperationMessage(message))
        )
    }

    func clearPresentedModelError(targetID: String) {
        presentedModelErrorMessagesByTarget[targetID] = nil
    }

    private func localizedModelOperationMessage(_ message: String) -> String {
        let downloadPrefix = "Download failed: "
        if message.hasPrefix(downloadPrefix) {
            return AppLocalization.format(
                "Download failed: %@",
                String(message.dropFirst(downloadPrefix.count))
            )
        }
        return AppLocalization.localizedString(message)
    }

    private func handleConfigurationNavigationRequest() {
        guard let navigationRequest,
              navigationRequest.id != lastHandledConfigurationNavigationRequestID,
              navigationRequest.target.tab == .model,
              let selectionID = navigationRequest.target.modelSelectionID
        else { return }

        lastHandledConfigurationNavigationRequestID = navigationRequest.id
        if let selection = selectionID.asrSelection {
            catalogTab = .asr
            switch selection {
            case .dictation:
                break
            case .mlx(let repo):
                activeLocalASRConfigurationTarget = .mlx(repo: repo)
            case .remote(let provider):
                editingASRProvider = provider
            }
            return
        }

        if let selection = selectionID.textSelection {
            catalogTab = .llm
            switch selection {
            case .appleIntelligence:
                break
            case .localLLM(let repo):
                customLLMConfigurationRepo = repo
                isCustomLLMConfigurationPresented = true
            case .remoteLLM(let provider):
                editingLLMProvider = provider
            }
        }
    }

    private func handleStorageAuthorizationNavigationRequest() {
        guard let navigationRequest,
              navigationRequest.id != lastHandledStorageAuthorizationNavigationRequestID,
              navigationRequest.target.tab == .model,
              navigationRequest.target.requestsModelStorageAuthorization,
              isActive
        else { return }

        lastHandledStorageAuthorizationNavigationRequestID = navigationRequest.id
        refreshModelStorageDisplayPath()
        guard ModelStorageDirectoryManager.resolvedRootResolution().accessIssue != nil else { return }

        pendingStorageAuthorizationPickerRequestID = navigationRequest.id
        isModelDownloadSettingsPresented = true
    }

    private func presentPendingStorageAuthorizationPicker() {
        guard pendingStorageAuthorizationPickerRequestID != nil else { return }
        pendingStorageAuthorizationPickerRequestID = nil

        DispatchQueue.main.async {
            chooseModelStorageDirectory()
        }
    }

    func reloadCachedConfigurationState() {
        cachedFeatureSettings = FeatureSettingsStore.load(defaults: .standard)
        cachedRemoteASRConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: remoteASRProviderConfigurationsRaw,
            sensitiveValueLoading: .metadataOnly
        )
        cachedRemoteLLMConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: remoteLLMProviderConfigurationsRaw,
            sensitiveValueLoading: .metadataOnly
        )
        refreshCatalogSnapshot()
    }

    func refreshCatalogSnapshot() {
        if isRefreshingCatalogSnapshot {
            needsAnotherCatalogRefresh = true
            return
        }
        if catalogSnapshot.allEntries.isEmpty {
            rebuildCatalogSnapshot()
            return
        }
        guard !isCatalogRefreshScheduled else { return }
        isCatalogRefreshScheduled = true
        DispatchQueue.main.async {
            isCatalogRefreshScheduled = false
            rebuildCatalogSnapshot()
        }
    }

    private func rebuildCatalogSnapshot() {
        if isRefreshingCatalogSnapshot {
            needsAnotherCatalogRefresh = true
            return
        }
        isRefreshingCatalogSnapshot = true
        defer {
            isRefreshingCatalogSnapshot = false
            if needsAnotherCatalogRefresh {
                needsAnotherCatalogRefresh = false
                refreshCatalogSnapshot()
            }
        }

        reconcileCancellingInstallTargets()

        let entries = switch catalogTab {
        case .asr:
            catalogBuilder.asrEntries()
        case .llm:
            catalogBuilder.llmEntries()
        }

        catalogSnapshot = ModelSettingsCatalogSnapshotBuilder.build(
            entries: entries,
            selectedTags: selectedTags,
            hiddenIDs: hiddenModelVisibilityIDs
        )
    }

    private func chooseModelStorageDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.directoryURL = ModelStorageDirectoryManager.resolvedRootURL()
        panel.prompt = localized("Choose")

        guard panel.runModal() == .OK, let selectedURL = panel.url else { return }
        let currentURL = ModelStorageDirectoryManager.resolvedRootURL().standardizedFileURL
        let proposedURL = selectedURL.standardizedFileURL
        if proposedURL != currentURL {
            let alert = NSAlert()
            alert.messageText = localized("Change Model Storage Path?")
            alert.informativeText = localized("After changing the model storage path, previously downloaded local models will need to be downloaded again.")
            alert.addButton(withTitle: localized("Confirm"))
            alert.addButton(withTitle: localized("Cancel"))
            guard alert.runModal() == .alertFirstButtonReturn else { return }
        }

        do {
            try ModelStorageDirectoryManager.saveUserSelectedRootURL(selectedURL)
            modelStorageSelectionError = nil
            refreshAllModelStorageRoots()
            refreshModelStorageDisplayPath()
            refreshCatalogSnapshot()
            SileroVADModelProvisioner.prefetchIfNeeded(for: LocalVADMode.stored())
        } catch {
            modelStorageSelectionError = AppLocalization.format(
                "Failed to update model storage path: %@",
                error.localizedDescription
            )
        }
    }

    func refreshModelStorageDisplayPath() {
        let resolution = ModelStorageDirectoryManager.resolvedRootResolution()
        modelStorageDisplayPath = resolution.writeRootURL.path
        modelStorageSelectionError = resolution.accessIssue?.localizedDescription
    }

    private func openModelStorageInFinder() {
        Task { @MainActor in
            ModelStorageDirectoryManager.openRootInFinder()
        }
    }

    private var modelTabHeader: some View {
        HStack(spacing: 10) {
            ModelCatalogTabPicker(selectedTab: $catalogTab)

            Spacer(minLength: 0)

            if !missingConfigurationIssues.isEmpty {
                missingConfigurationMenu
            }

            Text(AppLocalization.format("%d items", filteredEntries.count))
                .font(.caption)
                .foregroundStyle(.secondary)

            Button {
                isModelDownloadSettingsPresented = true
            } label: {
                Image(systemName: "gearshape")
            }
            .buttonStyle(SettingsCompactIconButtonStyle(size: 32))

            Button(action: openModelDebugWindow) {
                Text(localized("Debug"))
            }
            .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 12))
        }
    }

    private var missingConfigurationMenu: some View {
        Menu {
            ForEach(missingConfigurationIssueDescriptions, id: \.self) { description in
                Text(description)
            }
        } label: {
            missingConfigurationMenuLabel
        }
        .menuStyle(.borderlessButton)
        .help(missingConfigurationIssueDescriptions.joined(separator: "\n"))
    }

    private var missingConfigurationMenuLabel: some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text(missingConfigurationIssueCountLabel)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(
            Capsule(style: .continuous)
                .fill(Color.orange.opacity(0.10))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(Color.orange.opacity(0.18), lineWidth: 1)
        )
    }

    private var missingConfigurationIssueCountLabel: String {
        missingConfigurationIssues.count == 1
            ? localized("1 model needs setup")
            : AppLocalization.format("%d models need setup", missingConfigurationIssues.count)
    }

    private var modelDownloadSettingsSheet: some View {
        ModelDownloadSettingsSheet(
            modelStorageDisplayPath: modelStorageDisplayPath,
            modelStorageFallbackPath: ModelStorageDirectoryManager.defaultRootURL.path,
            modelStorageSelectionError: modelStorageSelectionError,
            onOpenModelStorageInFinder: openModelStorageInFinder,
            onChooseModelStorageDirectory: chooseModelStorageDirectory,
            localModelIdleUnloadDelaySeconds: $localModelIdleUnloadDelaySeconds,
            showIdleUnloadDelayInfo: $showIdleUnloadDelayInfo,
            isPresented: $isModelDownloadSettingsPresented
        )
    }

    var shouldPollModelState: Bool {
        ModelSettingsProgressRefreshSupport.shouldPollModelState(
            mlxState: mlxModelManager.state,
            mlxHasActiveDownloadingRepos: mlxModelManager.activeDownloadRepos.contains { repo in
                if case .downloading = mlxModelManager.state(for: repo) {
                    return true
                }
                return false
            },
            customLLMState: customLLMManager.state,
            customLLMStateByRepo: customLLMManager.stateByRepo,
            customLLMHasActiveDownloadingRepos: !customLLMManager.activeDownloadRepos.isEmpty,
            ggufStateByID: ggufTranslationModelManager.stateByID,
            ggufActiveDownloadModelID: ggufTranslationModelManager.activeDownloadModelID
        )
    }

    private func toggleTag(_ tag: String) {
        selectedTags = ModelCatalogTag.toggledTags(current: selectedTags, tag: tag)
    }

    func pruneSelectedTags() {
        selectedTags = selectedTags.intersection(Set(availableTags))
    }

    private func openModelDebugWindow() {
        guard let appDelegate = AppDelegate.shared else { return }
        switch catalogTab {
        case .asr:
            ASRDebugWindowManager.shared.present(appDelegate: appDelegate)
        case .llm:
            LLMDebugWindowManager.shared.present(appDelegate: appDelegate)
        }
    }

    private var missingConfigurationIssueDescriptions: [String] {
        missingConfigurationIssues.map(missingConfigurationIssueDescription(for:))
    }

    private func missingConfigurationIssueDescription(
        for issue: ModelConfigurationIssue
    ) -> String {
        switch issue.scope {
        case .remoteASRProvider(let provider):
            return AppLocalization.format("%@ %@: %@", provider.title, localized("ASR"), issue.message)
        case .remoteLLMProvider(let provider):
            return AppLocalization.format("%@ %@: %@", provider.title, localized("LLM"), issue.message)
        case .mlxModel(let repo):
            return AppLocalization.format("%@ %@: %@", mlxModelManager.displayTitle(for: repo), localized("ASR"), issue.message)
        case .customLLMModel(let repo):
            return AppLocalization.format("%@ %@: %@", customLLMManager.displayTitle(for: repo), localized("LLM"), issue.message)
        case .translationRemoteLLM(let provider):
            return AppLocalization.format("%@ %@: %@", provider.title, localized("Translation"), issue.message)
        case .rewriteRemoteLLM(let provider):
            return AppLocalization.format("%@ %@: %@", provider.title, localized("Rewrite"), issue.message)
        case .translationCustomLLM(let repo):
            return AppLocalization.format("%@ %@: %@", customLLMManager.displayTitle(for: repo), localized("Translation"), issue.message)
        case .rewriteCustomLLM(let repo):
            return AppLocalization.format("%@ %@: %@", customLLMManager.displayTitle(for: repo), localized("Rewrite"), issue.message)
        }
    }
}
