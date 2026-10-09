// GeneralSettingsView.swift
// Provides General Settings View for settings screens.

import SwiftUI
import CoreAudio
import AppKit

struct GeneralSettingsView: View {
    let appUpdateManager: AppUpdateManager
    let navigationRequest: SettingsNavigationRequest?
    var onRequestScrollToBottom: (() -> Void)?
    @AppStorage(AppPreferenceKey.interactionSoundsEnabled) private var interactionSoundsEnabled = true
    @AppStorage(AppPreferenceKey.interactionSoundPreset) private var interactionSoundPresetRaw = InteractionSoundPreset.soft.rawValue
    @AppStorage(AppPreferenceKey.muteSystemAudioWhileRecording) private var muteSystemAudioWhileRecording = false
    @AppStorage(AppPreferenceKey.overlayPosition) private var overlayPositionRaw = OverlayPosition.bottom.rawValue
    @AppStorage(AppPreferenceKey.overlayCardOpacity) private var overlayCardOpacity = 82
    @AppStorage(AppPreferenceKey.overlayCardCornerRadius) private var overlayCardCornerRadius = 24
    @AppStorage(AppPreferenceKey.overlayScreenEdgeInset) private var overlayScreenEdgeInset = 30
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @AppStorage(AppPreferenceKey.userMainLanguageCodes) private var userMainLanguageCodesRaw = UserMainLanguageOption.defaultStoredSelectionValue
    @AppStorage(AppPreferenceKey.autoCopyWhenNoFocusedInput) private var autoCopyWhenNoFocusedInput = false
    @AppStorage(AppPreferenceKey.realtimeTextDisplayEnabled) private var realtimeTextDisplayEnabled = true
    @AppStorage(AppPreferenceKey.customPasteHotkeyEnabled) private var customPasteHotkeyEnabled = false
    @AppStorage(AppPreferenceKey.customPasteHotkeyKeyCode) private var customPasteHotkeyKeyCode = Int(HotkeyPreference.defaultCustomPasteKeyCode)
    @AppStorage(AppPreferenceKey.customPasteHotkeyModifiers) private var customPasteHotkeyModifiers = Int(HotkeyPreference.defaultCustomPasteModifiers.rawValue)
    @AppStorage(AppPreferenceKey.customPasteHotkeySidedModifiers) private var customPasteHotkeySidedModifiers = 0
    @AppStorage(AppPreferenceKey.hotkeyDistinguishModifierSides) private var hotkeyDistinguishModifierSides = HotkeyPreference.defaultDistinguishModifierSides
    @AppStorage(AppPreferenceKey.launchAtLogin) private var launchAtLogin = false
    @AppStorage(AppPreferenceKey.showInDock) private var showInDock = false
    @AppStorage(AppPreferenceKey.autoCheckForUpdates) private var autoCheckForUpdates = true
    @AppStorage(AppPreferenceKey.hotkeyDebugLoggingEnabled) private var hotkeyDebugLoggingEnabled = false
    @AppStorage(AppPreferenceKey.llmDebugLoggingEnabled) private var llmDebugLoggingEnabled = false
    @AppStorage(AppPreferenceKey.localVADMode) private var localVADModeRaw = LocalVADMode.defaultMode.rawValue
    @AppStorage(AppPreferenceKey.networkProxyMode) private var networkProxyModeRaw = VoxtNetworkSession.ProxyMode.system.rawValue
    @AppStorage(AppPreferenceKey.customProxyScheme) private var customProxySchemeRaw = VoxtNetworkSession.ProxyScheme.http.rawValue
    @AppStorage(AppPreferenceKey.customProxyHost) private var customProxyHost = ""
    @AppStorage(AppPreferenceKey.customProxyPort) private var customProxyPort = ""
    @State private var inputDevices: [AudioInputDevice] = []
    @State private var microphoneState = MicrophoneResolvedState.empty
    @State private var launchAtLoginError: String?
    @State private var isSyncingLaunchAtLoginState = false
    @State private var interactionSoundPlayer = InteractionSoundPlayer()
    @State private var isAdvancedExpanded = false
    @State private var isUserMainLanguageSheetPresented = false
    @State private var isMicrophonePriorityDialogPresented = false
    @State private var customProxyUsername = ""
    @State private var customProxyPassword = ""
    @State private var isLogsViewerPresented = false

    private var networkProxyMode: Binding<VoxtNetworkSession.ProxyMode> {
        Binding(
            get: { VoxtNetworkSession.ProxyMode(rawValue: networkProxyModeRaw) ?? .system },
            set: { networkProxyModeRaw = $0.rawValue }
        )
    }

    private var overlayPosition: Binding<OverlayPosition> {
        Binding(
            get: { OverlayPosition(rawValue: overlayPositionRaw) ?? .bottom },
            set: { overlayPositionRaw = $0.rawValue }
        )
    }

    private var interfaceLanguageSelection: Binding<AppInterfaceLanguage> {
        Binding(
            get: { AppInterfaceLanguage(rawValue: interfaceLanguageRaw) ?? .system },
            set: { interfaceLanguageRaw = $0.rawValue }
        )
    }

    private var interactionSoundPresetSelection: Binding<InteractionSoundPreset> {
        Binding(
            get: { InteractionSoundPreset(rawValue: interactionSoundPresetRaw) ?? .soft },
            set: { interactionSoundPresetRaw = $0.rawValue }
        )
    }

    private var customProxyScheme: Binding<VoxtNetworkSession.ProxyScheme> {
        Binding(
            get: { VoxtNetworkSession.ProxyScheme(rawValue: customProxySchemeRaw) ?? .http },
            set: { customProxySchemeRaw = $0.rawValue }
        )
    }

    private var localVADMode: Binding<LocalVADMode> {
        Binding(
            get: { LocalVADMode.resolved(rawValue: localVADModeRaw) },
            set: { mode in
                LocalVADMode.save(mode)
                SileroVADModelProvisioner.prefetchIfNeeded(for: mode)
            }
        )
    }

    private var selectedUserMainLanguageCodes: [String] {
        UserMainLanguageOption.storedSelection(from: userMainLanguageCodesRaw)
    }

    private var userMainLanguageSummary: String {
        GeneralSettingsData.userMainLanguageSummary(selectedCodes: selectedUserMainLanguageCodes)
    }

    private var currentCustomPasteHotkey: HotkeyPreference.Hotkey {
        GeneralSettingsData.customPasteHotkey(
            keyCode: customPasteHotkeyKeyCode,
            modifiersRawValue: customPasteHotkeyModifiers,
            sidedModifiersRawValue: customPasteHotkeySidedModifiers
        )
    }

    private var currentCustomPasteHotkeyDisplayString: String {
        GeneralSettingsData.customPasteHotkeyDisplayString(
            keyCode: customPasteHotkeyKeyCode,
            modifiersRawValue: customPasteHotkeyModifiers,
            sidedModifiersRawValue: customPasteHotkeySidedModifiers,
            distinguishModifierSides: true
        )
    }

    var body: some View {
        settingsWithSheets
            .id(interfaceLanguageRaw)
    }

    private var settingsCards: some View {
        VStack(alignment: .leading, spacing: 16) {
            GeneralTranscriptionUICard(
                overlayPosition: overlayPosition,
                overlayCardOpacity: $overlayCardOpacity,
                overlayCardCornerRadius: $overlayCardCornerRadius,
                overlayScreenEdgeInset: $overlayScreenEdgeInset
            )
            .settingsNavigationAnchor(.generalTranscriptionUI)

            GeneralSectionDivider()

            GeneralLanguagesCard(
                interfaceLanguage: interfaceLanguageSelection,
                userMainLanguageSummary: userMainLanguageSummary,
                onEditUserMainLanguage: { isUserMainLanguageSheetPresented = true }
            )
            .settingsNavigationAnchor(.generalLanguages)

            GeneralSectionDivider()

            GeneralAudioCard(
                microphoneState: microphoneState,
                interactionSoundsEnabled: $interactionSoundsEnabled,
                muteSystemAudioWhileRecording: $muteSystemAudioWhileRecording,
                interactionSoundPreset: interactionSoundPresetSelection,
                onTrySound: { interactionSoundPlayer.playPreview(preset: interactionSoundPreset) },
                onManageMicrophones: { isMicrophonePriorityDialogPresented = true },
                onViewPriorityList: { isMicrophonePriorityDialogPresented = true }
            )
            .settingsNavigationAnchor(.generalAudio)

            GeneralSectionDivider()

            GeneralAppBehaviorCard(
                autoCopyWhenNoFocusedInput: $autoCopyWhenNoFocusedInput,
                realtimeTextDisplayEnabled: $realtimeTextDisplayEnabled,
                customPasteHotkeyEnabled: $customPasteHotkeyEnabled,
                customPasteHotkeyDisplayString: currentCustomPasteHotkeyDisplayString,
                launchAtLogin: $launchAtLogin,
                showInDock: $showInDock,
                autoCheckForUpdates: $autoCheckForUpdates,
                launchAtLoginError: launchAtLoginError
            )
            .settingsNavigationAnchor(.generalAppBehavior)

            GeneralSectionDivider()

            GeneralAdvancedCard(
                isExpanded: $isAdvancedExpanded,
                onExpand: scrollToBottomAfterAdvancedExpands
            ) {
                GeneralLoggingCard(
                    hotkeyDebugLoggingEnabled: $hotkeyDebugLoggingEnabled,
                    llmDebugLoggingEnabled: $llmDebugLoggingEnabled,
                    onViewLogs: { isLogsViewerPresented = true }
                )
                .settingsNavigationAnchor(.generalLogging)

                Divider()

                GeneralVADCard(localVADMode: localVADMode)

                Divider()

                GeneralProxyCard(
                    networkProxyMode: networkProxyMode,
                    customProxyScheme: customProxyScheme,
                    customProxyHost: $customProxyHost,
                    customProxyPort: $customProxyPort,
                    customProxyUsername: $customProxyUsername,
                    customProxyPassword: $customProxyPassword
                )
                .settingsNavigationAnchor(.generalProxy)
            }
            .settingsNavigationAnchor(.generalAdvanced)
        }
    }

    private var settingsWithAppBehaviorLifecycle: some View {
        settingsCards
        .onAppear {
            refreshInputDevices()

            Task {
                let status = AppBehaviorController.launchAtLoginIsEnabled()
                await MainActor.run {
                    isSyncingLaunchAtLoginState = true
                    if status != launchAtLogin {
                        launchAtLogin = status
                    }
                    isSyncingLaunchAtLoginState = false
                }
            }
            if autoCheckForUpdates != appUpdateManager.automaticallyChecksForUpdates {
                autoCheckForUpdates = appUpdateManager.automaticallyChecksForUpdates
            }
            refreshProxyCredentials()
            syncAdvancedExpansionWithNavigationRequest()
        }
        .onChange(of: launchAtLogin) { _, newValue in
            if isSyncingLaunchAtLoginState { return }
            Task {
                do {
                    try AppBehaviorController.setLaunchAtLogin(newValue)
                    await MainActor.run { launchAtLoginError = nil }
                } catch {
                    await MainActor.run {
                        launchAtLogin.toggle()
                        let format = NSLocalizedString("Unable to change login item: %@", comment: "")
                        launchAtLoginError = String(format: format, error.localizedDescription)
                    }
                }
            }
        }
        .onChange(of: showInDock) { _, newValue in
            if let appDelegate = AppDelegate.shared {
                appDelegate.synchronizeAppActivationPolicy()
            } else {
                AppBehaviorController.applyDockVisibility(
                    showInDock: newValue,
                    mainWindowVisible: true
                )
            }
        }
        .onChange(of: autoCheckForUpdates) { _, newValue in
            appUpdateManager.syncAutomaticallyChecksForUpdates(newValue)
        }
    }

    private var settingsWithAppearanceLifecycle: some View {
        settingsWithAppBehaviorLifecycle
        .onChange(of: interfaceLanguageRaw) { _, _ in
            NotificationCenter.default.post(name: .voxtInterfaceLanguageDidChange, object: nil)
        }
        .onChange(of: overlayPositionRaw) { _, _ in
            postOverlayAppearanceDidChange()
        }
        .onChange(of: overlayCardOpacity) { _, newValue in
            overlayCardOpacity = GeneralSettingsData.clampedOverlayOpacity(newValue)
            postOverlayAppearanceDidChange()
        }
        .onChange(of: overlayCardCornerRadius) { _, newValue in
            overlayCardCornerRadius = GeneralSettingsData.clampedOverlayCornerRadius(newValue)
            postOverlayAppearanceDidChange()
        }
        .onChange(of: realtimeTextDisplayEnabled) { _, _ in
            postOverlayAppearanceDidChange()
        }
        .onChange(of: overlayScreenEdgeInset) { _, newValue in
            overlayScreenEdgeInset = GeneralSettingsData.clampedOverlayScreenEdgeInset(newValue)
            postOverlayAppearanceDidChange()
        }
        .onChange(of: customProxyUsername) { _, _ in
            persistProxyCredentials()
        }
        .onChange(of: customProxyPassword) { _, _ in
            persistProxyCredentials()
        }
        .onChange(of: navigationRequest) { _, _ in
            syncAdvancedExpansionWithNavigationRequest()
        }
    }

    private var settingsWithLifecycle: some View {
        settingsWithAppearanceLifecycle
        .onReceive(NotificationCenter.default.publisher(for: .voxtAudioInputDevicesDidChange)) { _ in
            refreshInputDevices()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtSelectedInputDeviceDidChange)) { _ in
            refreshInputDevices()
        }
    }

    private var settingsWithSheets: some View {
        settingsWithLifecycle
        .sheet(isPresented: $isUserMainLanguageSheetPresented) {
            UserMainLanguageSelectionSheet(
                selectedCodes: selectedUserMainLanguageCodes,
                localeIdentifier: interfaceLanguage.localeIdentifier
            ) { updatedCodes in
                userMainLanguageCodesRaw = UserMainLanguageOption.storageValue(for: updatedCodes)
            }
        }
        .sheet(isPresented: $isMicrophonePriorityDialogPresented) {
            MicrophonePriorityDialog(
                state: microphoneState,
                onUseNow: { uid in
                    focusMicrophone(uid: uid, source: "settings dialog")
                },
                onAutoSwitchChanged: { isEnabled in
                    setMicrophoneAutoSwitchEnabled(isEnabled)
                },
                onReorderPriority: { orderedUIDs in
                    applyMicrophonePriorityOrder(orderedUIDs)
                }
            )
        }
        .sheet(isPresented: $isLogsViewerPresented) {
            LogsViewerSheet()
        }
    }

    private func refreshInputDevices() {
        inputDevices = AudioInputDeviceManager.availableInputDevices()
        microphoneState = MicrophonePreferenceManager.syncState(
            defaults: .standard,
            availableDevices: inputDevices
        )
    }

    private var interactionSoundPreset: InteractionSoundPreset {
        InteractionSoundPreset(rawValue: interactionSoundPresetRaw) ?? .soft
    }

    private var interfaceLanguage: AppInterfaceLanguage {
        AppInterfaceLanguage(rawValue: interfaceLanguageRaw) ?? .system
    }

    private func refreshProxyCredentials() {
        let credentials = VoxtNetworkSession.currentProxyCredentials()
        customProxyUsername = credentials.username
        customProxyPassword = credentials.password
    }

    private func persistProxyCredentials() {
        VoxtNetworkSession.setCustomProxyCredentials(
            username: customProxyUsername,
            password: customProxyPassword
        )
    }

    private func postOverlayAppearanceDidChange() {
        NotificationCenter.default.post(name: .voxtOverlayAppearanceDidChange, object: nil)
    }

    private func syncAdvancedExpansionWithNavigationRequest() {
        guard let section = navigationRequest?.target.section else { return }
        if section == .generalAdvanced || section == .generalLogging || section == .generalProxy {
            isAdvancedExpanded = true
        }
    }

    private func scrollToBottomAfterAdvancedExpands() {
        Task {
            try? await Task.sleep(nanoseconds: 90_000_000)
            await MainActor.run {
                onRequestScrollToBottom?()
            }
        }
    }

    private func selectMicrophoneManually(uid: String) {
        microphoneState = MicrophonePreferenceManager.setFocusedDevice(
            uid: uid,
            defaults: .standard,
            availableDevices: inputDevices
        )
        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)
    }

    private func setMicrophoneAutoSwitchEnabled(_ isEnabled: Bool) {
        VoxtLog.settings("Microphone auto switch updated from settings dialog. enabled=\(isEnabled)")
        microphoneState = MicrophonePreferenceManager.setAutoSwitchEnabled(
            isEnabled,
            defaults: .standard,
            availableDevices: inputDevices
        )
        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)
    }

    private func applyMicrophonePriorityOrder(_ orderedUIDs: [String]) {
        VoxtLog.settings("Microphone priority updated from settings dialog. orderedUIDs=\(orderedUIDs.joined(separator: ","))", verbose: true)
        microphoneState = MicrophonePreferenceManager.reorderPriority(
            orderedUIDs: orderedUIDs,
            defaults: .standard,
            availableDevices: inputDevices
        )
        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)
    }

    private func focusMicrophone(uid: String, source: String) {
        if let entry = microphoneState.entries.first(where: { $0.uid == uid }) {
            VoxtLog.settings("Microphone focus changed from \(source). uid=\(uid), name=\(entry.name)")
        } else {
            VoxtLog.settings("Microphone focus changed from \(source). uid=\(uid)")
        }
        selectMicrophoneManually(uid: uid)
    }
}
