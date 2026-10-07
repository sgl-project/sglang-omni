// SettingsView.swift
// Provides Settings View for settings shell.

import SwiftUI
import AppKit
import AVFoundation
import Speech
import ApplicationServices
import Combine

struct SettingsView: View {
    let availableDictionaryHistoryScanModels: () -> [DictionaryHistoryScanModelOption]
    let onIngestDictionarySuggestionsFromHistory: (DictionaryHistoryScanRequest, Bool) -> Void
    let onCancelDictionarySuggestionsFromHistory: () -> Void
    let mlxModelManager: MLXModelManager
    let customLLMManager: CustomLLMModelManager
    let ggufTranslationModelManager: GGUFTranslationModelManager
    @ObservedObject var historyStore: TranscriptionHistoryStore
    @ObservedObject var meetingFileTaskQueue: MeetingFileTaskQueue
    @ObservedObject var noteStore: VoxtNoteStore
    @ObservedObject var dictionaryStore: DictionaryStore
    @ObservedObject var dictionarySuggestionStore: DictionarySuggestionStore
    @ObservedObject var appUpdateManager: AppUpdateManager
    @ObservedObject var mainWindowState: MainWindowVisibilityState
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @AppStorage(AppPreferenceKey.appEnhancementEnabled) private var appEnhancementEnabled = true
    @AppStorage(AppPreferenceKey.transcriptionEngine) private var transcriptionEngineRaw = TranscriptionEngine.mlxAudio.rawValue
    @AppStorage(AppPreferenceKey.featureSettings) private var featureSettingsRaw = ""
    @AppStorage(AppPreferenceKey.remoteASRProviderConfigurations) private var remoteASRProviderConfigurationsRaw = ""
    @AppStorage(AppPreferenceKey.remoteLLMProviderConfigurations) private var remoteLLMProviderConfigurationsRaw = ""
    @AppStorage(AppPreferenceKey.hotkeyInputType) private var hotkeyInputType = HotkeyPreference.Hotkey.Input.Kind.keyboard.rawValue
    @AppStorage(AppPreferenceKey.hotkeyKeyCode) private var hotkeyKeyCode = Int(HotkeyPreference.defaultKeyCode)
    @AppStorage(AppPreferenceKey.hotkeyMouseButtonNumber) private var hotkeyMouseButtonNumber = HotkeyPreference.middleMouseButtonNumber
    @AppStorage(AppPreferenceKey.hotkeyModifiers) private var hotkeyModifiers = Int(HotkeyPreference.defaultModifiers.rawValue)
    @AppStorage(AppPreferenceKey.hotkeySidedModifiers) private var hotkeySidedModifiers = 0
    @AppStorage(AppPreferenceKey.hotkeyDistinguishModifierSides) private var hotkeyDistinguishModifierSides = HotkeyPreference.defaultDistinguishModifierSides
    @AppStorage(AppPreferenceKey.hotkeyPreset) private var hotkeyPreset = HotkeyPreference.defaultPreset.rawValue
    @State private var selectedTab: SettingsTab
    @State private var selectedFeatureTab: FeatureSettingsTab
    @State private var selectedHistoryFilter: HistoryFilterTab
    @State private var sidebarMode: SettingsSidebarMode
    @State private var navigationRequest: SettingsNavigationRequest?
    @State private var hasMissingPermissions = false
    @State private var hasNoAvailableMicrophones = false
    @State private var modelStorageAuthorizationIssue: String?
    @State private var missingModelConfigurationIssues: [ModelConfigurationIssue] = []
    @State private var languageRefreshToken = UUID()
    @State private var displayMode: SettingsDisplayMode
    @State private var initializedStaticTabs: Set<SettingsTab>
    @State private var activeModelDownloadCount = 0
    @State private var isFeedbackDialogPresented = false
    @State private var isHomeNotificationDialogPresented = false
    @StateObject private var notificationStore = VoxtNotificationStore()

    private static let officialWebsiteURL = URL(string: "https://voxt.actnow.dev")!
    private static let changelogURL = URL(string: "https://voxt.actnow.dev/changelog")!
    private static let sponsorURL = URL(string: "https://voxt.actnow.dev/#pricing")!
    private static let feedbackURL = URL(string: "https://github.com/hehehai/voxt/issues/new/choose")!
    private static let feedbackWeChatQRCodeURL = URL(string: "https://storage.actnow.dev/common/voxt/gw-wx.png")!
    private static let scrollBottomAnchorID = "settings-scroll-bottom-anchor"

    init(
        availableDictionaryHistoryScanModels: @escaping () -> [DictionaryHistoryScanModelOption],
        onIngestDictionarySuggestionsFromHistory: @escaping (DictionaryHistoryScanRequest, Bool) -> Void,
        onCancelDictionarySuggestionsFromHistory: @escaping () -> Void,
        mlxModelManager: MLXModelManager,
        customLLMManager: CustomLLMModelManager,
        ggufTranslationModelManager: GGUFTranslationModelManager,
        historyStore: TranscriptionHistoryStore,
        meetingFileTaskQueue: MeetingFileTaskQueue,
        noteStore: VoxtNoteStore,
        dictionaryStore: DictionaryStore,
        dictionarySuggestionStore: DictionarySuggestionStore,
        appUpdateManager: AppUpdateManager,
        mainWindowState: MainWindowVisibilityState,
        initialNavigationTarget: SettingsNavigationTarget = SettingsNavigationTarget(tab: .report),
        initialDisplayMode: SettingsDisplayMode = .normal
    ) {
        self.availableDictionaryHistoryScanModels = availableDictionaryHistoryScanModels
        self.onIngestDictionarySuggestionsFromHistory = onIngestDictionarySuggestionsFromHistory
        self.onCancelDictionarySuggestionsFromHistory = onCancelDictionarySuggestionsFromHistory
        self.mlxModelManager = mlxModelManager
        self.customLLMManager = customLLMManager
        self.ggufTranslationModelManager = ggufTranslationModelManager
        self.historyStore = historyStore
        self.meetingFileTaskQueue = meetingFileTaskQueue
        self.noteStore = noteStore
        self.dictionaryStore = dictionaryStore
        self.dictionarySuggestionStore = dictionarySuggestionStore
        self.appUpdateManager = appUpdateManager
        self.mainWindowState = mainWindowState
        _selectedTab = State(initialValue: initialNavigationTarget.tab)
        _selectedFeatureTab = State(initialValue: initialNavigationTarget.featureTab ?? .features)
        _selectedHistoryFilter = State(initialValue: initialNavigationTarget.historyFilter ?? .transcription)
        _sidebarMode = State(initialValue: Self.initialSidebarMode(for: initialNavigationTarget.tab))
        _navigationRequest = State(initialValue: SettingsNavigationRequest(target: initialNavigationTarget))
        _displayMode = State(initialValue: initialDisplayMode)
        _initializedStaticTabs = State(initialValue: Self.initializedStaticTabs(for: initialNavigationTarget.tab))
        _activeModelDownloadCount = State(
            initialValue: SettingsModelDownloadBadgeSupport.activeDownloadCount(
                mlxActiveDownloadRepos: mlxModelManager.activeDownloadRepos,
                customLLMActiveDownloadRepos: customLLMManager.activeDownloadRepos,
                ggufActiveDownloadModelID: ggufTranslationModelManager.activeDownloadModelID
            )
        )
    }

    var body: some View {
        settingsWithStateObservers
    }

    private var settingsContent: some View {
        ZStack {
            SettingsUIStyle.windowBackgroundColor
            Group {
                switch displayMode {
                case .normal:
                    normalSettingsContent
                case .onboarding:
                    onboardingContent
                }
            }
            .padding(.horizontal, 10)
            .padding(.bottom, 10)
            .padding(.top, 10)
        }
        .frame(minWidth: 820, minHeight: 560)
        .environment(\.locale, interfaceLanguage.locale)
        .groupBoxStyle(SettingsPanelGroupBoxStyle())
        .id(languageRefreshToken)
        .sheet(isPresented: $isFeedbackDialogPresented) {
            FeedbackDialogView(
                feedbackURL: Self.feedbackURL,
                qrCodeURL: Self.feedbackWeChatQRCodeURL,
                onClose: {
                    isFeedbackDialogPresented = false
                },
                onOpenFeedback: {
                    isFeedbackDialogPresented = false
                    openFeedbackPage()
                }
            )
        }
        .sheet(isPresented: $isHomeNotificationDialogPresented) {
            HomeNotificationDialogView(
                store: notificationStore,
                onClose: {
                    isHomeNotificationDialogPresented = false
                }
            )
        }
        .ignoresSafeArea(.container, edges: .top)
    }

    private var settingsWithNotifications: some View {
        settingsContent
        .onAppear {
            refreshPermissionBadge()
            refreshMicrophoneBadge()
            refreshModelStorageAuthorizationBadge()
            refreshModelConfigurationBadge()
            refreshNotifications()
        }
        .onReceive(ModelInstallationObservation.changes(mlx: mlxModelManager, customLLM: customLLMManager)) { _ in
            refreshModelConfigurationBadge()
        }
        .onReceive(modelDownloadBadgeCountPublisher) { count in
            let previousCount = activeModelDownloadCount
            activeModelDownloadCount = count
            if previousCount != count {
                refreshModelConfigurationBadge()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            refreshPermissionBadge()
            refreshMicrophoneBadge()
            refreshModelStorageAuthorizationBadge()
            refreshModelConfigurationBadge()
            refreshNotifications()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtAudioInputDevicesDidChange)) { _ in
            refreshMicrophoneBadge()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtSettingsSelectTab)) { notification in
            guard case .normal = displayMode else { return }
            guard let target = SettingsNavigationTarget(notification: notification)
            else {
                return
            }
            applyNavigationTarget(target)
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtSettingsNavigate)) { notification in
            guard case .normal = displayMode else { return }
            guard let target = SettingsNavigationTarget(notification: notification) else { return }
            applyNavigationTarget(target)
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtInterfaceLanguageDidChange)) { _ in
            AppLocalization.refreshLanguageCache()
            languageRefreshToken = UUID()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtRemoteProviderConfigurationsDidChange)) { _ in
            refreshModelConfigurationBadge()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtPermissionsDidChange)) { _ in
            refreshPermissionBadge()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtModelStorageAuthorizationDidChange)) { _ in
            refreshModelStorageAuthorizationBadge()
        }
    }

    private var settingsWithStateObservers: some View {
        settingsWithNotifications
        .onChange(of: mainWindowState.isVisible) { _, isVisible in
            guard isVisible else { return }
            refreshPermissionBadge()
            refreshMicrophoneBadge()
            refreshModelStorageAuthorizationBadge()
            refreshModelConfigurationBadge()
        }
        .onChange(of: appEnhancementEnabled) { _, isEnabled in
            if !isEnabled, selectedTab == .feature, selectedFeatureTab == .appEnhancement {
                navigationRequest = nil
                selectedFeatureTab = .features
            }
        }
        .onChange(of: transcriptionEngineRaw) { _, _ in
            refreshPermissionBadge()
        }
        .onChange(of: featureSettingsRaw) { _, _ in
            redirectIfSelectedFeatureTabHidden()
            refreshPermissionBadge()
            refreshModelConfigurationBadge()
        }
        .onChange(of: remoteASRProviderConfigurationsRaw) { _, _ in
            refreshModelConfigurationBadge()
        }
        .onChange(of: remoteLLMProviderConfigurationsRaw) { _, _ in
            refreshModelConfigurationBadge()
        }
        .onChange(of: selectedTab) { _, tab in
            if Self.isStaticTab(tab) {
                initializedStaticTabs.insert(tab)
            }
        }
    }

    private var normalSettingsContent: some View {
        HStack(alignment: .top, spacing: 8) {
            SettingsSidebar(
                sidebarMode: $sidebarMode,
                selectedTab: $selectedTab,
                selectedFeatureTab: $selectedFeatureTab,
                selectedHistoryFilter: $selectedHistoryFilter,
                onSelectTab: { tab in
                    navigationRequest = nil
                    switchToRootTab(tab)
                },
                onSelectFeatureTab: { tab in
                    navigationRequest = nil
                    switchToFeatureTab(tab)
                },
                onSelectHistoryFilter: { filter in
                    navigationRequest = nil
                    switchToHistoryFilter(filter)
                },
                onReturnToRoot: {
                    navigationRequest = nil
                    sidebarMode = .root
                    if selectedTab == .feature || selectedTab == .history || Self.isSettingsTab(selectedTab) {
                        selectedTab = .report
                    }
                },
                featureAvailability: featureAvailability,
                hasMissingPermissions: hasMissingPermissions,
                hasNoAvailableMicrophones: hasNoAvailableMicrophones,
                modelStorageAuthorizationIssue: modelStorageAuthorizationIssue,
                activeModelDownloadCount: activeModelDownloadCount,
                hasMissingModelConfigurationIssues: !missingModelConfigurationIssues.isEmpty,
                updateBadgeState: updateBadgeState,
                hasUnreadNotification: notificationStore.hasUnreadLatestNotification,
                onTapPermissionBadge: {
                    navigationRequest = nil
                    sidebarMode = .settings
                    selectedTab = .permissions
                },
                onTapMicrophoneBadge: {
                    sidebarMode = .settings
                    selectedTab = .general
                    navigationRequest = SettingsNavigationRequest(
                        target: SettingsNavigationTarget(tab: .general, section: .generalAudio)
                    )
                },
                onTapModelStorageAuthorizationBadge: {
                    sidebarMode = .settings
                    selectedTab = .model
                    navigationRequest = SettingsNavigationRequest(
                        target: SettingsNavigationTarget(
                            tab: .model,
                            requestsModelStorageAuthorization: true
                        )
                    )
                },
                onTapModelBadge: {
                    navigationRequest = nil
                    sidebarMode = .settings
                    selectedTab = .model
                },
                onTapUpdateBadge: {
                    appUpdateManager.checkForUpdatesWithUserInterface()
                },
                onTapNotification: {
                    isHomeNotificationDialogPresented = true
                },
                onTapWebsite: {
                    openOfficialWebsite()
                },
                onTapFeedback: {
                    isFeedbackDialogPresented = true
                }
            )
            .frame(width: SettingsUIStyle.sidebarWidth)
            .frame(maxHeight: .infinity, alignment: .top)

            VStack(alignment: .leading, spacing: 0) {
                if showsContentHeader {
                    HStack(alignment: selectedTab == .report && sidebarMode == .root ? .top : .center, spacing: 12) {
                        if sidebarMode == .root, selectedTab == .report {
                            VStack(alignment: .leading, spacing: 10) {
                                Text(AppLocalization.localizedString("Speak clearly. Adapt to context."))
                                    .font(.system(size: 18, weight: .bold))
                                    .lineLimit(1)
                                HomeShortcutPrompt(shortcut: currentTranscriptionHotkeyDisplayString)
                            }
                        } else {
                            HStack(alignment: .center, spacing: 8) {
                                Text(currentTitle)
                                    .font(.title3.weight(.semibold))

                                if showsFeatureExperimentalBadge {
                                    FeatureStatusBadge(text: AppLocalization.localizedString("Experimental"))
                                }
                            }
                        }

                        Spacer(minLength: 0)

                        if sidebarMode == .feature,
                           selectedFeatureTab == .files,
                           meetingFileTaskQueue.hasFinishedTasks {
                            Button(AppLocalization.localizedString("Clear Finished Tasks")) {
                                meetingFileTaskQueue.clearFinishedTasks()
                                NotificationCenter.default.post(
                                    name: .voxtFeatureSettingsToastRequested,
                                    object: nil,
                                    userInfo: [
                                        "message": AppLocalization.localizedString(
                                            "Completed tasks are available in Meeting History."
                                        )
                                    ]
                                )
                            }
                            .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 11, height: 26))
                        }

                        if sidebarMode == .root, selectedTab == .report {
                            Button(AppLocalization.localizedString("Guide")) {
                                AppDelegate.shared?.openOnboardingWindow()
                            }
                            .buttonStyle(SettingsPillButtonStyle())
                        }
                    }
                    .frame(
                        height: sidebarMode == .feature && selectedFeatureTab == .files ? 26 : nil,
                        alignment: .center
                    )
                }

                tabContent
                    .padding(.top, contentTopPadding)

                if selectedTab == .report && sidebarMode == .root {
                    HStack(spacing: 12) {
                        Spacer(minLength: 0)

                        HomeFooterLinkButton(
                            title: AppLocalization.localizedString("Changelog"),
                            action: openChangelog
                        )

                        HomeFooterLinkButton(
                            title: AppLocalization.localizedString("Sponsor"),
                            showsCoffeeIcon: true,
                            action: openSponsorPage
                        )
                    }
                    .padding(.top, 14)
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .background(
                RoundedRectangle(cornerRadius: SettingsUIStyle.panelCornerRadius, style: .continuous)
                    .fill(SettingsUIStyle.panelFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: SettingsUIStyle.panelCornerRadius, style: .continuous)
                    .strokeBorder(SettingsUIStyle.panelBorderColor, lineWidth: 1)
            )
        }
    }

    private var onboardingContent: some View {
        // Keep legacy navigation routes on the same read-only-on-open, six-step guide.
        Color.clear.onAppear {
            guard case .onboarding(let step) = displayMode else { return }
            displayMode = .normal
            AppDelegate.shared?.openOnboardingWindow(
                step: OnboardingGuideStep.restored(from: step.rawValue)
            )
        }
    }

    private var interfaceLanguage: AppInterfaceLanguage {
        AppInterfaceLanguage(rawValue: interfaceLanguageRaw) ?? .system
    }

    private var currentTranscriptionHotkeyDisplayString: String {
        _ = hotkeyInputType
        _ = hotkeyKeyCode
        _ = hotkeyMouseButtonNumber
        _ = hotkeyModifiers
        _ = hotkeySidedModifiers
        _ = hotkeyDistinguishModifierSides
        _ = hotkeyPreset

        return HotkeyPreference.displayString(
            for: HotkeyPreference.load(),
            distinguishModifierSides: HotkeyPreference.loadDistinguishModifierSides()
        )
        .replacingOccurrences(of: "fn", with: "FN")
    }

    private var featureAvailability: FeatureAvailabilitySettings {
        _ = featureSettingsRaw
        return FeatureSettingsStore.availability()
    }

    private func redirectIfSelectedFeatureTabHidden() {
        guard selectedTab == .feature else { return }
        let visible = FeatureSettingsTab.visibleTabs(availability: featureAvailability)
        guard !visible.contains(selectedFeatureTab) else { return }
        navigationRequest = nil
        selectedFeatureTab = .features
    }

    private var updateBadgeState: UpdateBadgeState {
        if appUpdateManager.isPreparingInteractiveUpdateUI {
            return .openingWindow(appUpdateManager.latestVersion)
        }
        if let issue = appUpdateManager.updateCheckIssueMessage,
           !issue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .checkFailed(issue)
        }
        if appUpdateManager.hasUpdate {
            return .newVersion(appUpdateManager.latestVersion)
        }
        return .none
    }

    private var modelDownloadBadgeCountPublisher: AnyPublisher<Int, Never> {
        Publishers.CombineLatest3(
            mlxModelManager.$activeDownloadRepos,
            customLLMManager.$activeDownloadRepos,
            ggufTranslationModelManager.$activeDownloadModelID
        )
        .map { mlxActiveDownloadRepos, customLLMActiveDownloadRepos, ggufActiveDownloadModelID in
            SettingsModelDownloadBadgeSupport.activeDownloadCount(
                mlxActiveDownloadRepos: mlxActiveDownloadRepos,
                customLLMActiveDownloadRepos: customLLMActiveDownloadRepos,
                ggufActiveDownloadModelID: ggufActiveDownloadModelID
            )
        }
        .removeDuplicates()
        .eraseToAnyPublisher()
    }

    @ViewBuilder
    private var tabContent: some View {
        if selectedTab == .history || selectedTab == .report || selectedTab == .feature || selectedTab == .dictionary || selectedTab == .model {
            staticTabContent
        } else {
            scrollableTabContent
        }
    }

    @ViewBuilder
    private var staticTabContent: some View {
        ZStack(alignment: .topLeading) {
            if initializedStaticTabs.contains(.report) {
                staticTabLayer(for: .report) {
                    ReportSettingsView(
                        historyStore: historyStore,
                        dictionaryStore: dictionaryStore,
                        mainWindowState: mainWindowState,
                        isActive: selectedTab == .report && sidebarMode == .root
                    )
                }
            }

            if initializedStaticTabs.contains(.history) {
                staticTabLayer(for: .history) {
                    HistorySettingsView(
                        historyStore: historyStore,
                        noteStore: noteStore,
                        dictionaryStore: dictionaryStore,
                        selectedFilter: $selectedHistoryFilter,
                        navigationRequest: navigationRequest
                    )
                }
            }

            if initializedStaticTabs.contains(.dictionary) {
                staticTabLayer(for: .dictionary) {
                    DictionarySettingsView(
                        historyStore: historyStore,
                        dictionaryStore: dictionaryStore,
                        dictionarySuggestionStore: dictionarySuggestionStore,
                        availableHistoryScanModels: availableDictionaryHistoryScanModels,
                        onIngestSuggestionsFromHistory: onIngestDictionarySuggestionsFromHistory,
                        onCancelIngestSuggestionsFromHistory: onCancelDictionarySuggestionsFromHistory,
                        navigationRequest: navigationRequest
                    )
                }
            }

            if initializedStaticTabs.contains(.feature) {
                staticTabLayer(for: .feature) {
                    FeatureSettingsView(
                        selectedTab: selectedFeatureTab,
                        navigationRequest: navigationRequest,
                        onSelectFeatureTab: { tab in
                            switchToFeatureTab(tab)
                        },
                        mlxModelManager: mlxModelManager,
                        customLLMManager: customLLMManager,
                        ggufTranslationModelManager: ggufTranslationModelManager,
                        noteStore: noteStore,
                        meetingFileTaskQueue: meetingFileTaskQueue
                    )
                }
            }

            if initializedStaticTabs.contains(.model) {
                staticTabLayer(for: .model) {
                    ModelSettingsView(
                        mlxModelManager: mlxModelManager,
                        customLLMManager: customLLMManager,
                        ggufTranslationModelManager: ggufTranslationModelManager,
                        mainWindowState: mainWindowState,
                        missingConfigurationIssues: missingModelConfigurationIssues,
                        navigationRequest: navigationRequest,
                        isActive: selectedTab == .model
                    )
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    @ViewBuilder
    private func staticTabLayer<Content: View>(
        for tab: SettingsTab,
        @ViewBuilder content: () -> Content
    ) -> some View {
        content()
            .opacity(selectedTab == tab ? 1 : 0)
            .allowsHitTesting(selectedTab == tab)
            .accessibilityHidden(selectedTab != tab)
    }

    private var scrollableTabContent: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    Group {
                        switch selectedTab {
                        case .general:
                            GeneralSettingsView(
                                appUpdateManager: appUpdateManager,
                                navigationRequest: navigationRequest,
                                onRequestScrollToBottom: {
                                    withAnimation(.easeInOut(duration: 0.18)) {
                                        proxy.scrollTo(Self.scrollBottomAnchorID, anchor: .bottom)
                                    }
                                }
                            )
                        case .permissions:
                            PermissionsSettingsView(navigationRequest: navigationRequest)
                        case .report:
                            EmptyView()
                        case .model:
                            ModelSettingsView(
                                mlxModelManager: mlxModelManager,
                                customLLMManager: customLLMManager,
                                ggufTranslationModelManager: ggufTranslationModelManager,
                                mainWindowState: mainWindowState,
                                missingConfigurationIssues: missingModelConfigurationIssues,
                                navigationRequest: navigationRequest,
                                isActive: true
                            )
                        case .dictionary:
                            EmptyView()
                        case .feature:
                            EmptyView()
                        case .appEnhancement:
                            EmptyView()
                        case .hotkey:
                            HotkeySettingsView()
                        case .about:
                            AboutSettingsView(
                                appUpdateManager: appUpdateManager,
                                navigationRequest: navigationRequest
                            )
                        case .history:
                            EmptyView()
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(.trailing, SettingsUIStyle.contentScrollTrailingGutter)

                    Color.clear
                        .frame(height: 1)
                        .id(Self.scrollBottomAnchorID)
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .padding(.trailing, -SettingsUIStyle.contentScrollIndicatorOutset)
            .onAppear {
                scrollScrollableContentIfNeeded(with: navigationRequest, proxy: proxy)
            }
            .onChange(of: navigationRequest?.id) { _, _ in
                scrollScrollableContentIfNeeded(with: navigationRequest, proxy: proxy)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private func scrollScrollableContentIfNeeded(with request: SettingsNavigationRequest?, proxy: ScrollViewProxy) {
        guard let request,
              request.target.tab == selectedTab,
              let section = request.target.section
        else {
            return
        }

        DispatchQueue.main.async {
            withAnimation(.easeInOut(duration: 0.18)) {
                proxy.scrollTo(section.rawValue, anchor: .top)
            }
        }
    }

    private func refreshPermissionBadge() {
        let engine = TranscriptionEngine(rawValue: transcriptionEngineRaw) ?? .mlxAudio
        let featureSettings = FeatureSettingsStore.load(defaults: .standard)
        let context = SettingsPermissionRequirementResolver.requirementContext(
            selectedEngine: engine,
            featureSettings: featureSettings
        )

        let hasMissingPermissions = SettingsPermissionRequirementResolver.hasMissingPermissions(context: context)
        guard self.hasMissingPermissions != hasMissingPermissions else { return }
        self.hasMissingPermissions = hasMissingPermissions
    }

    private func refreshModelConfigurationBadge() {
        let issues = ModelConfigurationIssueResolver.missingIssues(
            mlxModelManager: mlxModelManager,
            customLLMManager: customLLMManager
        )
        guard issues != missingModelConfigurationIssues else { return }
        missingModelConfigurationIssues = issues
    }

    private func refreshModelStorageAuthorizationBadge() {
        modelStorageAuthorizationIssue = ModelStorageDirectoryManager
            .resolvedRootResolution()
            .accessIssue?
            .localizedDescription
    }

    private func refreshNotifications() {
        Task {
            await notificationStore.refresh()
        }
    }

    private func refreshMicrophoneBadge() {
        hasNoAvailableMicrophones = AudioInputDeviceManager.availableInputDevices().isEmpty
    }

    private var currentTitle: LocalizedStringKey {
        switch sidebarMode {
        case .feature:
            if selectedFeatureTab == .files {
                return "File Tasks"
            }
            return selectedFeatureTab.titleKey
        case .history:
            return selectedHistoryFilter.titleKey
        case .root, .settings:
            return selectedTab.titleKey
        }
    }

    private var showsFeatureExperimentalBadge: Bool {
        sidebarMode == .feature && (selectedFeatureTab == .meeting || selectedFeatureTab == .note)
    }

    private var showsContentHeader: Bool {
        selectedTab != .history || sidebarMode != .history
    }

    private var contentTopPadding: CGFloat {
        if sidebarMode == .root, selectedTab == .report {
            return 24
        }
        if !showsContentHeader {
            return 0
        }
        return 12
    }

    private func applyNavigationTarget(_ target: SettingsNavigationTarget) {
        navigationRequest = SettingsNavigationRequest(target: target)
        if let featureTab = target.featureTab {
            if FeatureSettingsTab.visibleTabs(availability: featureAvailability).contains(featureTab) {
                selectedFeatureTab = featureTab
            } else {
                selectedFeatureTab = .features
            }
        }
        if let historyFilter = target.historyFilter {
            selectedHistoryFilter = historyFilter
        }
        if target.tab == .feature {
            sidebarMode = .feature
            selectedTab = .feature
        } else if target.tab == .history {
            sidebarMode = .history
            selectedTab = .history
        } else if Self.isSettingsTab(target.tab) {
            sidebarMode = .settings
            selectedTab = target.tab
        } else {
            sidebarMode = .root
            selectedTab = target.tab
        }
    }

    private func switchToRootTab(_ tab: SettingsTab) {
        if tab == .feature {
            selectedTab = .feature
            sidebarMode = .feature
            if !FeatureSettingsTab.visibleTabs(availability: featureAvailability).contains(selectedFeatureTab) {
                selectedFeatureTab = .features
            }
            return
        }
        if tab == .history {
            selectedTab = .history
            sidebarMode = .history
            return
        }
        if Self.isSettingsTab(tab) {
            sidebarMode = .settings
            selectedTab = tab
            return
        }
        sidebarMode = .root
        selectedTab = tab
    }

    private func switchToFeatureTab(_ tab: FeatureSettingsTab) {
        selectedTab = .feature
        sidebarMode = .feature
        selectedFeatureTab = tab
    }

    private func switchToHistoryFilter(_ filter: HistoryFilterTab) {
        selectedTab = .history
        sidebarMode = .history
        selectedHistoryFilter = filter
    }

    private func openOfficialWebsite() {
        NSWorkspace.shared.open(Self.officialWebsiteURL)
    }

    private func openChangelog() {
        NSWorkspace.shared.open(Self.changelogURL)
    }

    private func openSponsorPage() {
        NSWorkspace.shared.open(Self.sponsorURL)
    }

    private func openFeedbackPage() {
        NSWorkspace.shared.open(Self.feedbackURL)
    }

    private static func initializedStaticTabs(for tab: SettingsTab) -> Set<SettingsTab> {
        isStaticTab(tab) ? [tab] : []
    }

    private static func initialSidebarMode(for tab: SettingsTab) -> SettingsSidebarMode {
        if tab == .feature {
            return .feature
        }
        if tab == .history {
            return .history
        }
        if isSettingsTab(tab) {
            return .settings
        }
        return .root
    }

    private static func isStaticTab(_ tab: SettingsTab) -> Bool {
        tab == .history || tab == .report || tab == .feature || tab == .dictionary || tab == .model
    }

    private static func isSettingsTab(_ tab: SettingsTab) -> Bool {
        SettingsTab.settingsTabs.contains(tab)
    }

}
