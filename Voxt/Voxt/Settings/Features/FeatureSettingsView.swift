// FeatureSettingsView.swift
// Provides Feature Settings View for feature settings.

import SwiftUI
import AppKit

struct FeatureSettingsView: View {
    let selectedTab: FeatureSettingsTab
    let navigationRequest: SettingsNavigationRequest?
    var onSelectFeatureTab: ((FeatureSettingsTab) -> Void)? = nil
    @ObservedObject var mlxModelManager: MLXModelManager
    @ObservedObject var customLLMManager: CustomLLMModelManager
    @ObservedObject var ggufTranslationModelManager: GGUFTranslationModelManager
    @ObservedObject var noteStore: VoxtNoteStore
    @ObservedObject var meetingFileTaskQueue: MeetingFileTaskQueue
    @StateObject var meetingDiarizationModelManager = MeetingDiarizationModelManager()

    @AppStorage(AppPreferenceKey.featureSettings) var featureSettingsRaw = ""
    @AppStorage(AppPreferenceKey.remoteASRProviderConfigurations) var remoteASRProviderConfigurationsRaw = ""
    @AppStorage(AppPreferenceKey.remoteLLMProviderConfigurations) var remoteLLMProviderConfigurationsRaw = ""
    @AppStorage(AppPreferenceKey.userMainLanguageCodes) var userMainLanguageCodesRaw = UserMainLanguageOption.defaultStoredSelectionValue
    @AppStorage(AppPreferenceKey.interfaceLanguage) var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @AppStorage(AppPreferenceKey.appBranchGroups) var appBranchGroupsData = Data()

    @State var featureSettings = FeatureSettings.placeholder
    @State var selectorSheet: FeatureModelSelectorSheet?
    @State var remindersListDescriptors: [RemindersListDescriptor] = []
    @State var isRemindersListSheetPresented = false
    @State var isMeetingAdvancedSettingsExpanded = false
    @State private var toastMessage = ""
    @State private var toastDismissTask: Task<Void, Never>?
    @State var isFilesDisableBlockedAlertPresented = false
    @State private var appleIntelligenceRefreshRevision = 0
    @State var scrollToBottomRequestRevision = 0

    var body: some View {
        Group {
            switch selectedTab {
            case .features:
                featuresContent
            case .transcription:
                transcriptionContent
            case .meeting:
                meetingContent
            case .files:
                filesContent
            case .note:
                noteContent
            case .translation:
                translationContent
            case .rewrite:
                rewriteContent
            case .appEnhancement:
                AppEnhancementSettingsView(navigationRequest: navigationRequest)
            }
        }
        .id("\(selectedTab.rawValue)-\(interfaceLanguageRaw)")
        .overlay(alignment: .top) {
            if !toastMessage.isEmpty {
                ModelDebugToast(message: toastMessage) {
                    dismissToast()
                }
                .padding(.top, 12)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.16), value: toastMessage)
        .alert(
            featureSettingsLocalized("Cannot Disable Files"),
            isPresented: $isFilesDisableBlockedAlertPresented
        ) {
            Button(featureSettingsLocalized("OK"), role: .cancel) {}
        } message: {
            Text(featureSettingsLocalized("Finish or cancel all file tasks before disabling Files."))
        }
        .sheet(item: $selectorSheet) { sheet in
            FeatureModelSelectorDialog(
                title: sheet.title,
                entries: selectorEntries(for: sheet),
                selectedID: selectedSelectionID(for: sheet),
                onSelect: { selectionID in
                    applySelection(selectionID, for: sheet)
                }
            )
        }
        .sheet(isPresented: $isRemindersListSheetPresented) {
            RemindersListSelectorDialog(
                title: AppLocalization.localizedString("Choose Reminder List"),
                entries: remindersListDescriptors,
                selectedIdentifier: featureSettings.transcription.notes.remindersSync.selectedListIdentifier,
                onSelect: applyRemindersListSelection
            )
        }
        .onAppear {
            refreshAppleIntelligenceAvailability()
            reloadFeatureSettings()
            refreshRemindersLists()
            meetingDiarizationModelManager.refresh()
            meetingDiarizationModelManager.ensureSelectedModelInstalled()
        }
        .onChange(of: featureSettingsRaw) { _, _ in
            handleFeatureSettingsStorageChange()
            meetingDiarizationModelManager.refresh()
            meetingDiarizationModelManager.ensureSelectedModelInstalled()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtPermissionsDidChange)) { _ in
            refreshRemindersLists()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            refreshAppleIntelligenceAvailability()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtFeatureSettingsToastRequested)) { notification in
            guard let message = notification.userInfo?["message"] as? String else { return }
            if message.isEmpty {
                dismissToast()
            } else {
                showToast(message)
            }
        }
        .id(interfaceLanguageRaw)
    }

    private func showToast(_ message: String, duration: TimeInterval = 2.2) {
        toastDismissTask?.cancel()
        toastMessage = message
        toastDismissTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(duration))
            guard !Task.isCancelled else { return }
            toastMessage = ""
        }
    }

    private func dismissToast() {
        toastDismissTask?.cancel()
        toastMessage = ""
    }

    func requestScrollToBottom() {
        scrollToBottomRequestRevision += 1
    }

    func binding<Value>(
        get: @escaping () -> Value,
        set: @escaping (Value) -> Void
    ) -> Binding<Value> {
        Binding(
            get: get,
            set: { newValue in
                set(newValue)
                FeatureSettingsStore.save(featureSettings, defaults: .standard)
                reloadFeatureSettings()
                refreshRemindersLists()
            }
        )
    }

    func saveFeatureSettings() {
        FeatureSettingsStore.save(featureSettings, defaults: .standard)
        reloadFeatureSettings()
    }

    func reloadFeatureSettings() {
        featureSettings = FeatureSettingsStore.load(defaults: .standard)
    }

    func handleFeatureSettingsStorageChange() {
        reloadFeatureSettings()
        refreshRemindersLists()
    }

    func chooseObsidianVaultDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.directoryURL = SecurityScopedBookmarkSupport.resolveDirectoryURL(
            bookmarkData: featureSettings.transcription.notes.obsidianSync.vaultBookmarkData,
            fallbackPath: featureSettings.transcription.notes.obsidianSync.vaultPath
        ) ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        panel.prompt = AppLocalization.localizedString("Choose")

        guard panel.runModal() == .OK, let selectedURL = panel.url else { return }
        do {
            let bookmarkData = try SecurityScopedBookmarkSupport.createBookmark(for: selectedURL)
            featureSettings.transcription.notes.obsidianSync.vaultPath = selectedURL.standardizedFileURL.path
            featureSettings.transcription.notes.obsidianSync.vaultBookmarkData = bookmarkData
            saveFeatureSettings()
        } catch {
            VoxtLog.settingsWarning("Failed to store Obsidian vault bookmark: \(error.localizedDescription)")
        }
    }

    func presentRemindersListSelector() {
        refreshRemindersLists()
        isRemindersListSheetPresented = true
    }

    func applyRemindersListSelection(_ descriptor: RemindersListDescriptor) {
        featureSettings.transcription.notes.remindersSync.selectedListIdentifier = descriptor.identifier
        featureSettings.transcription.notes.remindersSync.selectedListTitle = descriptor.displayTitle
        saveFeatureSettings()
    }

    func refreshRemindersLists() {
        guard RemindersPermissionManager.isAuthorized() else {
            remindersListDescriptors = []
            return
        }
        remindersListDescriptors = RemindersPermissionManager.writableLists()
    }

    var selectedRemindersListTitle: String {
        let storedSettings = featureSettings.transcription.notes.remindersSync
        if let descriptor = remindersListDescriptors.first(where: { $0.identifier == storedSettings.selectedListIdentifier }) {
            return descriptor.displayTitle
        }
        let trimmedTitle = storedSettings.selectedListTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmedTitle.isEmpty ? AppLocalization.localizedString("Not configured") : trimmedTitle
    }

    var selectorBuilder: FeatureModelCatalogBuilder {
        FeatureModelCatalogBuilder(
            mlxModelManager: mlxModelManager,
            customLLMManager: customLLMManager,
            ggufTranslationModelManager: ggufTranslationModelManager,
            featureSettings: featureSettings,
            remoteASRProviderConfigurationsRaw: remoteASRProviderConfigurationsRaw,
            remoteLLMProviderConfigurationsRaw: remoteLLMProviderConfigurationsRaw,
            appleIntelligenceAvailability: appleIntelligenceAvailability,
            primaryUserLanguageCode: selectedUserLanguageCodes.first
        )
    }

    var appleIntelligenceAvailability: AppleIntelligenceAvailability {
        _ = appleIntelligenceRefreshRevision
        return AppleIntelligenceAvailability.current
    }

    func refreshAppleIntelligenceAvailability() {
        let availability = AppleIntelligenceAvailability.current
        VoxtLog.info("Apple Intelligence availability: \(availability.logDescription)")
        AppDelegate.shared?.refreshTextEnhancerAvailability()
        appleIntelligenceRefreshRevision += 1
    }

    var selectedUserLanguageCodes: [String] {
        UserMainLanguageOption.storedSelection(from: userMainLanguageCodesRaw)
    }

    func selectedSelectionID(for sheet: FeatureModelSelectorSheet) -> FeatureModelSelectionID {
        switch sheet {
        case .transcriptionASR:
            return featureSettings.transcription.asrSelectionID
        case .transcriptionLLM:
            return featureSettings.transcription.llmSelectionID
        case .transcriptionNoteTitle:
            return featureSettings.transcription.notes.titleModelSelectionID
        case .translationASR:
            return featureSettings.translation.asrSelectionID
        case .translationModel:
            return featureSettings.translation.modelSelectionID
        case .rewriteASR:
            return featureSettings.rewrite.asrSelectionID
        case .rewriteLLM:
            return featureSettings.rewrite.llmSelectionID
        case .meetingASR:
            return featureSettings.meeting.asrSelectionID
        case .meetingSummary:
            return featureSettings.meeting.summaryModelSelectionID
        }
    }

    func applySelection(_ selectionID: FeatureModelSelectionID, for sheet: FeatureModelSelectorSheet) {
        FeatureSettingsStore.update(defaults: .standard) { settings in
            switch sheet {
            case .transcriptionASR:
                settings.transcription.asrSelectionID = selectionID
            case .transcriptionLLM:
                settings.transcription.llmSelectionID = selectionID
            case .transcriptionNoteTitle:
                settings.transcription.notes.titleModelSelectionID = selectionID
            case .translationASR:
                settings.translation.asrSelectionID = selectionID
            case .translationModel:
                settings.translation.modelSelectionID = selectionID
            case .rewriteASR:
                settings.rewrite.asrSelectionID = selectionID
            case .rewriteLLM:
                settings.rewrite.llmSelectionID = selectionID
            case .meetingASR:
                settings.meeting.asrSelectionID = selectionID
            case .meetingSummary:
                settings.meeting.summaryModelSelectionID = selectionID
            }
        }
        reloadFeatureSettings()
    }

    func selectorEntries(for sheet: FeatureModelSelectorSheet) -> [FeatureModelSelectorEntry] {
        selectorBuilder.entries(for: sheet)
    }

    func asrSelectionSummary(_ selectionID: FeatureModelSelectionID) -> String {
        selectorBuilder.asrSelectionSummary(selectionID)
    }

    func llmSelectionSummary(_ selectionID: FeatureModelSelectionID) -> String {
        selectorBuilder.llmSelectionSummary(selectionID)
    }

    func translationSelectionSummary(_ selectionID: FeatureModelSelectionID) -> String {
        selectorBuilder.translationSelectionSummary(selectionID)
    }

    var appleIntelligenceAvailable: Bool {
        appleIntelligenceAvailability.isAvailable
    }
}
