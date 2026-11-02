// OnboardingSettingsView.swift
// Provides Onboarding Settings View for onboarding settings.

import SwiftUI
import AppKit
import AVFoundation
import Speech
import ApplicationServices

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct OnboardingSettingsView: View {
    @Binding var currentStep: OnboardingStep

    @ObservedObject var mlxModelManager: MLXModelManager
    @ObservedObject var customLLMManager: CustomLLMModelManager
    @ObservedObject var appUpdateManager: AppUpdateManager

    let onExit: () -> Void
    let onFinish: () -> Void

    @AppStorage(AppPreferenceKey.interactionSoundsEnabled) var interactionSoundsEnabled = true
    @AppStorage(AppPreferenceKey.muteSystemAudioWhileRecording) var muteSystemAudioWhileRecording = false
    @AppStorage(AppPreferenceKey.interfaceLanguage) var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @AppStorage(AppPreferenceKey.featureSettings) var featureSettingsRaw = ""
    @AppStorage(AppPreferenceKey.translationTargetLanguage) var translationTargetLanguageRaw = TranslationTargetLanguage.english.rawValue
    @AppStorage(AppPreferenceKey.userMainLanguageCodes) var userMainLanguageCodesRaw = UserMainLanguageOption.defaultStoredSelectionValue
    @AppStorage(AppPreferenceKey.autoCopyWhenNoFocusedInput) var autoCopyWhenNoFocusedInput = false
    @AppStorage(AppPreferenceKey.modelStorageRootPath) var modelStorageRootPath = ""
    @AppStorage(AppPreferenceKey.transcriptionEngine) var engineRaw = TranscriptionEngine.mlxAudio.rawValue
    @AppStorage(AppPreferenceKey.enhancementMode) var enhancementModeRaw = EnhancementMode.off.rawValue
    @AppStorage(AppPreferenceKey.mlxModelRepo) var mlxModelRepo = MLXModelManager.defaultModelRepo
    @AppStorage(AppPreferenceKey.customLLMModelRepo) var customLLMRepo = CustomLLMModelManager.defaultModelRepo
    @AppStorage(AppPreferenceKey.translationCustomLLMModelRepo) var translationCustomLLMRepo = CustomLLMModelManager.defaultModelRepo
    @AppStorage(AppPreferenceKey.rewriteCustomLLMModelRepo) var rewriteCustomLLMRepo = CustomLLMModelManager.defaultModelRepo
    @AppStorage(AppPreferenceKey.translationModelProvider) var translationModelProviderRaw = TranslationModelProvider.customLLM.rawValue
    @AppStorage(AppPreferenceKey.translationFallbackModelProvider) var translationFallbackModelProviderRaw = TranslationModelProvider.customLLM.rawValue
    @AppStorage(AppPreferenceKey.rewriteModelProvider) var rewriteModelProviderRaw = RewriteModelProvider.customLLM.rawValue
    @AppStorage(AppPreferenceKey.remoteASRSelectedProvider) var remoteASRSelectedProviderRaw = RemoteASRProvider.openAIWhisper.rawValue
    @AppStorage(AppPreferenceKey.remoteASRProviderConfigurations) var remoteASRProviderConfigurationsRaw = ""
    @AppStorage(AppPreferenceKey.remoteLLMSelectedProvider) var remoteLLMSelectedProviderRaw = RemoteLLMProvider.openAI.rawValue
    @AppStorage(AppPreferenceKey.remoteLLMProviderConfigurations) var remoteLLMProviderConfigurationsRaw = ""
    @AppStorage(AppPreferenceKey.translationRemoteLLMProvider) var translationRemoteLLMProviderRaw = ""
    @AppStorage(AppPreferenceKey.rewriteRemoteLLMProvider) var rewriteRemoteLLMProviderRaw = ""
    @AppStorage(AppPreferenceKey.hotkeyPreset) var hotkeyPresetRaw = HotkeyPreference.defaultPreset.rawValue
    @AppStorage(AppPreferenceKey.hotkeyTriggerMode) var hotkeyTriggerModeRaw = HotkeyPreference.defaultTriggerMode.rawValue
    @AppStorage(AppPreferenceKey.hotkeyDistinguishModifierSides) var hotkeyDistinguishModifierSides = HotkeyPreference.defaultDistinguishModifierSides

    @State var inputDevices: [AudioInputDevice] = []
    @State var microphoneState = MicrophoneResolvedState.empty
    @State var modelStorageDisplayPath = ""
    @State var modelStorageSelectionError: String?
    @State var isUserMainLanguageSheetPresented = false
    @State var isMicrophonePriorityDialogPresented = false
    @State var isPermissionsDialogPresented = false
    @State var editingASRProvider: RemoteASRProvider?
    @State var editingLLMProvider: RemoteLLMProvider?
    @State var translationTestInput = OnboardingTranslationTest.defaultInput
    @State var rewriteTestPrompt = OnboardingRewriteTest.defaultPrompt
    @State var rewriteTestSourceText = OnboardingRewriteTest.defaultSourceText
    @State var appEnhancementDemoPlayer: AVPlayer?
    @State var featureSettings = FeatureSettings.placeholder
    @State private var permissionRefreshRevision = 0
    @State var permissionMonitoringKinds: Set<OnboardingContextualPermission> = []
    @State private var permissionMonitorTasks: [OnboardingContextualPermission: Task<Void, Never>] = [:]

    var interfaceLanguage: AppInterfaceLanguage {
        AppInterfaceLanguage(rawValue: interfaceLanguageRaw) ?? .system
    }

    var translationTargetLanguage: TranslationTargetLanguage {
        TranslationTargetLanguage(rawValue: translationTargetLanguageRaw) ?? .english
    }

    var selectedEngine: TranscriptionEngine {
        TranscriptionEngine(rawValue: engineRaw) ?? .mlxAudio
    }

    var selectedRemoteASRProvider: RemoteASRProvider {
        RemoteASRProvider(rawValue: remoteASRSelectedProviderRaw) ?? .openAIWhisper
    }

    var selectedRemoteLLMProvider: RemoteLLMProvider {
        RemoteLLMProvider(rawValue: remoteLLMSelectedProviderRaw) ?? .openAI
    }

    var selectedRewriteProvider: RewriteModelProvider {
        RewriteModelProvider(rawValue: rewriteModelProviderRaw) ?? .customLLM
    }

    var remoteASRConfigurations: [String: RemoteProviderConfiguration] {
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

    var selectedUserMainLanguageCodes: [String] {
        UserMainLanguageOption.storedSelection(from: userMainLanguageCodesRaw)
    }

    var userMainLanguageSummary: String {
        let codes = selectedUserMainLanguageCodes
        guard let primaryCode = codes.first,
              let primaryOption = UserMainLanguageOption.option(for: primaryCode) else {
            return UserMainLanguageOption.fallbackOption().title()
        }

        if codes.count == 1 {
            return primaryOption.title()
        }

        let format = AppLocalization.localizedString("%@ + %d more")
        return String(format: format, primaryOption.title(), codes.count - 1)
    }

    var appleIntelligenceAvailable: Bool {
        AppleIntelligenceAvailability.current.isAvailable
    }

    var modelPathChoice: Binding<OnboardingModelPathChoice> {
        Binding(
            get: {
                switch selectedEngine {
                case .mlxAudio:
                    return .local
                case .remote:
                    return .remote
                case .dictation:
                    return .dictation
                }
            },
            set: { newValue in
                switch newValue {
                case .local:
                    if selectedEngine == .remote || selectedEngine == .dictation {
                        engineRaw = TranscriptionEngine.mlxAudio.rawValue
                    }
                    enhancementModeRaw = EnhancementMode.customLLM.rawValue
                    translationModelProviderRaw = TranslationModelProvider.customLLM.rawValue
                    translationFallbackModelProviderRaw = TranslationModelProvider.customLLM.rawValue
                    rewriteModelProviderRaw = RewriteModelProvider.customLLM.rawValue
                case .remote:
                    engineRaw = TranscriptionEngine.remote.rawValue
                    enhancementModeRaw = EnhancementMode.remoteLLM.rawValue
                    translationModelProviderRaw = TranslationModelProvider.remoteLLM.rawValue
                    translationFallbackModelProviderRaw = TranslationModelProvider.remoteLLM.rawValue
                    rewriteModelProviderRaw = RewriteModelProvider.remoteLLM.rawValue
                    translationRemoteLLMProviderRaw = selectedRemoteLLMProvider.rawValue
                    rewriteRemoteLLMProviderRaw = selectedRemoteLLMProvider.rawValue
                case .dictation:
                    engineRaw = TranscriptionEngine.dictation.rawValue
                }
            }
        )
    }

    var llmPathChoice: Binding<OnboardingTextModelPathChoice> {
        Binding(
            get: {
                switch featureSettings.transcription.llmSelectionID.textSelection {
                case .remoteLLM:
                    return .remote
                case .appleIntelligence:
                    return .system
                case .localLLM, .none:
                    return .local
                }
            },
            set: { newValue in
                applyLLMPathChoice(newValue)
            }
        )
    }

    var localEngineSelection: Binding<TranscriptionEngine> {
        Binding(
            get: {
                switch selectedEngine {
                case .mlxAudio, .remote, .dictation:
                    return .mlxAudio
                }
            },
            set: { _ in engineRaw = TranscriptionEngine.mlxAudio.rawValue }
        )
    }

    var hotkeyPresetSelection: Binding<HotkeyPreference.Preset> {
        Binding(
            get: {
                switch HotkeyPreference.Preset(rawValue: hotkeyPresetRaw) ?? .fnCombo {
                case .commandCombo:
                    return .commandCombo
                case .mouseMiddleFnShift:
                    return .mouseMiddleFnShift
                case .fnCombo, .custom:
                    return .fnCombo
                }
            },
            set: { applyHotkeyPreset($0) }
        )
    }

    private var onboardingBodyContent: AnyView {
        AnyView(
            HStack(alignment: .top, spacing: 8) {
                onboardingSidebar
                    .frame(width: SettingsUIStyle.sidebarWidth)
                    .frame(maxHeight: .infinity, alignment: .top)

                VStack(alignment: .leading, spacing: 0) {
                    onboardingHeader

                    ScrollView {
                        VStack(alignment: .leading, spacing: 16) {
                            stepContent
                            currentStepPermissionSection
                        }
                        .padding(.top, 2)
                        .padding(.bottom, 14)
                        .padding(.trailing, SettingsUIStyle.contentScrollTrailingGutter)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                    }
                    .padding(.top, 16)
                    .padding(.trailing, -SettingsUIStyle.contentScrollIndicatorOutset)

                    onboardingFooter
                        .padding(.top, 12)
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
        )
    }

    private var onboardingObservedContent: AnyView {
        let localized = AnyView(onboardingBodyContent.environment(\.locale, interfaceLanguage.locale))
        let appeared = AnyView(localized.onAppear {
                refreshInputDevices()
                refreshModelStorageDisplayPath()
                featureSettings = FeatureSettingsStore.load(defaults: .standard)
                prepareDemoPlayerIfNeeded(for: currentStep)
            })
        let languageObserved = AnyView(appeared.onChange(of: interfaceLanguageRaw) { _, _ in
                syncLocalizedOnboardingSamples()
                NotificationCenter.default.post(name: .voxtInterfaceLanguageDidChange, object: nil)
            })
        let featureObserved = AnyView(languageObserved.onChange(of: featureSettingsRaw) { _, _ in
                featureSettings = FeatureSettingsStore.load(defaults: .standard)
            })
        let storageObserved = AnyView(featureObserved.onChange(of: modelStorageRootPath) { _, _ in
                refreshModelStorageDisplayPath()
            })
        let mlxObserved = AnyView(storageObserved.onChange(of: mlxModelRepo) { _, newValue in
                handleMLXRepoChange(newValue)
            })
        let llmRepoObserved = AnyView(mlxObserved.onChange(of: customLLMRepo) { _, newValue in
                handleCustomLLMRepoChange(newValue)
            })
        let engineObserved = AnyView(llmRepoObserved.onChange(of: engineRaw) { _, _ in
                syncOnboardingFeatureSelections()
            })
        let enhancementObserved = AnyView(engineObserved.onChange(of: enhancementModeRaw) { _, _ in
                syncOnboardingFeatureSelections()
            })
        let remoteASRObserved = AnyView(enhancementObserved.onChange(of: remoteASRSelectedProviderRaw) { _, _ in
                syncOnboardingFeatureSelections()
            })
        let remoteLLMObserved = AnyView(remoteASRObserved.onChange(of: remoteLLMSelectedProviderRaw) { _, _ in
                syncOnboardingFeatureSelections()
            })
        let targetLanguageObserved = AnyView(remoteLLMObserved.onChange(of: translationTargetLanguageRaw) { _, _ in
                syncOnboardingFeatureSelections()
            })
        let stepObserved = AnyView(targetLanguageObserved.onChange(of: currentStep) { _, newValue in
                OnboardingPreferenceManager.saveLastStep(newValue)
                prepareDemoPlayerIfNeeded(for: newValue)
            })
        let audioDevicesObserved = AnyView(stepObserved.onReceive(NotificationCenter.default.publisher(for: .voxtAudioInputDevicesDidChange)) { _ in
                refreshInputDevices()
            })
        let selectedInputObserved = AnyView(audioDevicesObserved.onReceive(NotificationCenter.default.publisher(for: .voxtSelectedInputDeviceDidChange)) { _ in
                refreshInputDevices()
            })
        return AnyView(selectedInputObserved.onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
                permissionRefreshRevision += 1
            })
    }

    private var onboardingSheetContent: AnyView {
        let languageSheet = AnyView(onboardingObservedContent.sheet(isPresented: $isUserMainLanguageSheetPresented) {
            UserMainLanguageSelectionSheet(
                selectedCodes: selectedUserMainLanguageCodes,
                localeIdentifier: interfaceLanguage.localeIdentifier
            ) { updatedCodes in
                userMainLanguageCodesRaw = UserMainLanguageOption.storageValue(for: updatedCodes)
            }
        })
        let microphoneSheet = AnyView(languageSheet.sheet(isPresented: $isMicrophonePriorityDialogPresented) {
            MicrophonePriorityDialog(
                state: microphoneState,
                onUseNow: { uid in
                    focusMicrophone(uid: uid)
                },
                onAutoSwitchChanged: { isEnabled in
                    setMicrophoneAutoSwitchEnabled(isEnabled)
                },
                onReorderPriority: { orderedUIDs in
                    applyMicrophonePriorityOrder(orderedUIDs)
                }
            )
        })
        let permissionsSheet = AnyView(microphoneSheet.sheet(isPresented: $isPermissionsDialogPresented) {
            ScrollView {
                PermissionsSettingsView(navigationRequest: nil)
                    .padding(16)
            }
            .settingsDialogChrome(width: 720, height: 520, onClose: {
                isPermissionsDialogPresented = false
            })
        })
        let asrSheet = AnyView(permissionsSheet.sheet(item: $editingASRProvider) { provider in
            RemoteProviderConfigurationSheet(
                providerTitle: provider.title,
                credentialHint: asrCredentialHint(for: provider),
                showsDoubaoFields: provider == .doubaoASR,
                testTarget: .asr(provider),
                configuration: RemoteModelConfigurationStore.resolvedASRConfiguration(
                    provider: provider,
                    from: remoteASRProviderConfigurationsRaw
                )
            ) { configuration in
                saveRemoteASRConfiguration(configuration)
            }
        })
        return AnyView(asrSheet.sheet(item: $editingLLMProvider) { provider in
            RemoteProviderConfigurationSheet(
                providerTitle: provider.title,
                credentialHint: nil,
                showsDoubaoFields: false,
                testTarget: .llm(provider),
                configuration: RemoteModelConfigurationStore.resolvedLLMConfiguration(
                    provider: provider,
                    from: remoteLLMProviderConfigurationsRaw
                )
            ) { configuration in
                saveRemoteLLMConfiguration(configuration)
            }
        }.onDisappear {
            for task in permissionMonitorTasks.values {
                task.cancel()
            }
            permissionMonitorTasks.removeAll()
            permissionMonitoringKinds.removeAll()
        })
    }

    var body: some View {
        onboardingSheetContent
            .id(interfaceLanguageRaw)
    }

    var onboardingSidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(localized("Setup Guide"))
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                Spacer(minLength: 0)

                Text(AppLocalization.format("%d/%d", currentStep.stepNumber, OnboardingStep.allCases.count))
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 7)
                    .frame(height: 19)
                    .background(
                        Capsule(style: .continuous)
                            .fill(Color.primary.opacity(0.055))
                    )
            }
            .padding(.top, 5)
            .padding(.bottom, 14)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(OnboardingStep.allCases) { step in
                    Button {
                        currentStep = step
                    } label: {
                        HStack(spacing: 10) {
                            SettingsSidebarIconView(kind: step.sidebarIconKind)
                                .frame(width: SettingsUIStyle.sidebarItemIconWidth)

                            Text(step.title)
                                .font(.system(size: 13, weight: .medium))
                                .lineLimit(1)
                                .truncationMode(.tail)
                                .allowsTightening(true)
                                .layoutPriority(1)

                            Spacer(minLength: 0)

                            OnboardingStatusBadge(status: stepStatus(for: step), isSelected: step == currentStep)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(SettingsSidebarItemButtonStyle(isActive: step == currentStep))
                }
            }

            Spacer(minLength: 8)

            VStack(spacing: 8) {
                if shouldShowPermissionBadge {
                    Button(action: { isPermissionsDialogPresented = true }) {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(.red)
                            Text(localized("Permissions Disabled"))
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.red)
                            Spacer(minLength: 0)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .buttonStyle(SettingsStatusButtonStyle(tint: .red))
                }

                Group {
                    if currentStep == .finish {
                        Button(action: onFinish) {
                            Label(localized("Start Voxt"), systemImage: "checkmark.circle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(SettingsPrimaryButtonStyle())
                    } else {
                        Button(action: onExit) {
                            Label(localized("Exit Guide"), systemImage: "xmark.circle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(SettingsPillButtonStyle())
                    }
                }
                .frame(maxWidth: .infinity)
            }
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    var onboardingHeader: some View {
        HStack(alignment: .top, spacing: 14) {
            SettingsSidebarIconView(kind: currentStep.sidebarIconKind)
                .foregroundStyle(Color.accentColor)
                .frame(width: 20, height: 20)
                .frame(width: 36, height: 36)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color.accentColor.opacity(0.12))
                )

            VStack(alignment: .leading, spacing: 5) {
                Text(currentStep.title)
                    .font(.title3.weight(.semibold))
                Text(currentStep.subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)

            Text(AppLocalization.format("%d/%d", currentStep.stepNumber, OnboardingStep.allCases.count))
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 9)
                .frame(height: 26)
                .background(
                    Capsule(style: .continuous)
                        .fill(SettingsUIStyle.controlFillColor)
                )
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    var onboardingFooter: some View {
        HStack(spacing: 8) {
            if let previousStep = currentStep.previous {
                Button {
                    currentStep = previousStep
                } label: {
                    Label(localized("Prev"), systemImage: "chevron.left")
                }
                .buttonStyle(SettingsPillButtonStyle())
            }

            Spacer(minLength: 0)

            if currentStep != .finish {
                Button(action: onExit) {
                    Label(localized("Exit Guide"), systemImage: "xmark.circle")
                }
                .buttonStyle(SettingsPillButtonStyle())
            }

            if let nextStep = currentStep.next {
                Button {
                    currentStep = nextStep
                } label: {
                    Label(localized("Next"), systemImage: "chevron.right")
                        .labelStyle(OnboardingNextLabelStyle())
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
            } else {
                Button(action: onFinish) {
                    Label(localized("Start Voxt"), systemImage: "checkmark.circle")
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
            }
        }
        .padding(.top, 12)
        .overlay(alignment: .top) {
            Divider()
        }
    }

    func stepStatus(for step: OnboardingStep) -> OnboardingStepStatus {
        OnboardingStepStatusResolver.resolve(step: step, snapshot: onboardingStatusSnapshot)
    }

    func isPermissionGranted(_ permission: OnboardingContextualPermission) -> Bool {
        _ = permissionRefreshRevision
        return OnboardingPermissionGrantResolver.isGranted(permission)
    }

    func requestPermission(_ permission: OnboardingContextualPermission) {
        let initialState = isPermissionGranted(permission)
        permissionRefreshRevision += 1
        startPermissionMonitoring(permission, initialState: initialState)

        switch permission {
        case .microphone:
            Task {
                _ = await AVCaptureDevice.requestAccess(for: .audio)
                await MainActor.run { permissionRefreshRevision += 1 }
            }
        case .speechRecognition:
            SFSpeechRecognizer.requestAuthorization { _ in
                Task { @MainActor in
                    self.permissionRefreshRevision += 1
                }
            }
        case .accessibility:
            let granted = AccessibilityPermissionManager.request(prompt: true)
            if !granted {
                Task { @MainActor in
                    PermissionGuidance.openSettings(for: permission)
                }
            }
        }
    }

    func openSettings(for permission: OnboardingContextualPermission) {
        PermissionGuidance.openSettings(for: permission)
    }

    private func startPermissionMonitoring(
        _ permission: OnboardingContextualPermission,
        initialState: Bool
    ) {
        permissionMonitorTasks[permission]?.cancel()
        permissionMonitoringKinds.insert(permission)

        let task = Task { @MainActor in
            defer {
                permissionMonitorTasks[permission] = nil
                permissionMonitoringKinds.remove(permission)
            }

            for _ in 0..<60 {
                try? await Task.sleep(for: .milliseconds(500))
                if Task.isCancelled { return }

                let latestState = OnboardingPermissionGrantResolver.isGranted(permission)
                permissionRefreshRevision += 1
                if latestState != initialState {
                    return
                }
            }
        }

        permissionMonitorTasks[permission] = task
    }

}

private extension OnboardingStep {
    var sidebarIconKind: SettingsSidebarIconKind {
        switch self {
        case .language:
            return .settings
        case .model:
            return .model
        case .transcription:
            return .transcription
        case .translation:
            return .translation
        case .rewrite:
            return .rewrite
        case .meeting:
            return .meeting
        case .appEnhancement:
            return .appEnhancement
        case .finish:
            return .home
        }
    }
}

private struct OnboardingNextLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 4) {
            configuration.title
            configuration.icon
        }
    }
}
