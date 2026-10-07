// OnboardingSettingsData.swift
// Provides Onboarding Settings Data for onboarding settings.

import SwiftUI
import AppKit
import AVFoundation
import UniformTypeIdentifiers

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

extension OnboardingSettingsView {
    var triggerModeSelection: Binding<HotkeyPreference.TriggerMode> {
        Binding(
            get: { HotkeyPreference.TriggerMode(rawValue: hotkeyTriggerModeRaw) ?? .tap },
            set: { hotkeyTriggerModeRaw = $0.rawValue }
        )
    }

    var formattedTranscriptionHotkey: String {
        HotkeyPreference.displayString(
            for: HotkeyPreference.load(),
            distinguishModifierSides: true
        )
    }

    var formattedTranslationHotkey: String {
        HotkeyPreference.displayString(
            for: HotkeyPreference.loadTranslation(),
            distinguishModifierSides: true
        )
    }

    var formattedRewriteHotkey: String {
        HotkeyPreference.displayString(
            for: HotkeyPreference.loadRewrite(),
            distinguishModifierSides: true
        )
    }

    var formattedMeetingHotkey: String {
        HotkeyPreference.displayString(
            for: HotkeyPreference.loadMeeting(),
            distinguishModifierSides: true
        )
    }

    var missingConfigurationIssues: [ModelConfigurationIssue] {
        ModelConfigurationIssueResolver.missingIssues(
            mlxModelManager: mlxModelManager,
            customLLMManager: customLLMManager
        )
    }

    var rewriteIssues: [ModelConfigurationIssue] {
        missingConfigurationIssues.filter { issue in
            switch issue.scope {
            case .rewriteRemoteLLM, .rewriteCustomLLM:
                return true
            default:
                return false
            }
        }
    }

    var onboardingStatusSnapshot: OnboardingStepStatusSnapshot {
        OnboardingStepStatusSnapshot(
            hasModelIssues: !missingConfigurationIssues.isEmpty,
            hasRecordingMicrophone: !inputDevices.isEmpty,
            hasRecordingPermissions: recordingPermissionsSatisfied,
            hasRewriteIssues: !rewriteIssues.isEmpty,
            appEnhancementEnabled: true
        )
    }

    var currentPermissionContext: OnboardingPermissionRequirementContext {
        OnboardingPermissionRequirementContext(
            selectedEngine: selectedEngine
        )
    }

    var currentStepMissingPermissions: [OnboardingContextualPermission] {
        OnboardingPermissionRequirementResolver.requiredPermissions(
            for: currentStep,
            context: currentPermissionContext
        )
        .filter { !OnboardingPermissionGrantResolver.isGranted($0) }
    }

    var currentStepRequiredPermissions: [OnboardingContextualPermission] {
        OnboardingPermissionRequirementResolver.requiredPermissions(
            for: currentStep,
            context: currentPermissionContext
        )
    }

    var shouldShowPermissionBadge: Bool {
        !currentStepMissingPermissions.isEmpty
    }

    func handleMLXRepoChange(_ newValue: String) {
        let canonicalRepo = MLXModelManager.canonicalModelRepo(newValue)
        if canonicalRepo != newValue {
            mlxModelRepo = canonicalRepo
            return
        }
        mlxModelManager.updateModel(repo: canonicalRepo)
        syncOnboardingFeatureSelections()
    }

    func handleCustomLLMRepoChange(_ newValue: String) {
        let sanitizedRepo = CustomLLMModelManager.isSupportedModelRepo(newValue)
            ? newValue
            : CustomLLMModelManager.defaultModelRepo
        if sanitizedRepo != newValue {
            customLLMRepo = sanitizedRepo
            return
        }
        customLLMManager.updateModel(repo: sanitizedRepo)
        syncOnboardingFeatureSelections()
    }

    func syncOnboardingModelManagers() {
        let canonicalRepo = MLXModelManager.canonicalModelRepo(mlxModelRepo)
        if canonicalRepo != mlxModelRepo {
            mlxModelRepo = canonicalRepo
        }
        mlxModelManager.updateModel(repo: canonicalRepo)

        let sanitizedCustomLLMRepo = CustomLLMModelManager.isSupportedModelRepo(customLLMRepo)
            ? customLLMRepo
            : CustomLLMModelManager.defaultModelRepo
        if sanitizedCustomLLMRepo != customLLMRepo {
            customLLMRepo = sanitizedCustomLLMRepo
        }
        customLLMManager.updateModel(repo: sanitizedCustomLLMRepo)
    }

    func prepareDemoPlayerIfNeeded(for step: OnboardingStep) {
        switch step {
        case .appEnhancement:
            if appEnhancementDemoPlayer == nil {
                appEnhancementDemoPlayer = AVPlayer(url: OnboardingVideoDemo.appEnhancementURL)
            }
        default:
            break
        }
    }

    var recordingPermissionsSatisfied: Bool {
        OnboardingPermissionRequirementResolver.requiredPermissions(
            for: .transcription,
            context: currentPermissionContext
        )
        .allSatisfy { permission in
            OnboardingPermissionGrantResolver.isGranted(permission)
        }
    }

    var recordingPermissionMessages: [String] {
        var messages: [String] = []
        if !OnboardingPermissionGrantResolver.isGranted(.microphone) {
            messages.append(localized("Microphone permission is required. Enable it in Settings > Permissions."))
        }
        if selectedEngine == .dictation,
           !OnboardingPermissionGrantResolver.isGranted(.speechRecognition) {
            messages.append(localized("Speech Recognition permission is required for Direct Dictation. Enable it in Settings > Permissions."))
        }
        if !OnboardingPermissionGrantResolver.isGranted(.accessibility) {
            messages.append(localized("Accessibility permission is required to insert text into other apps."))
        }
        return messages
    }

    var enhancementModeTitle: String {
        featureSettings.transcription.llmEnabled
            ? llmSelectionSummary(featureSettings.transcription.llmSelectionID)
            : AppLocalization.localizedString("Disabled")
    }

    var translationProviderSummary: String {
        translationSelectionSummary(featureSettings.translation.modelSelectionID)
    }

    var rewriteProviderSummary: String {
        llmSelectionSummary(featureSettings.rewrite.llmSelectionID)
    }

    var meetingASRSummary: String {
        asrSelectionSummary(featureSettings.meeting.asrSelectionID)
    }

    var meetingSummaryModelSummary: String {
        llmSelectionSummary(featureSettings.meeting.summaryModelSelectionID)
    }

    var notesStatusSummary: String {
        AppLocalization.localizedString("Enabled")
    }

    var onboardingASRSummary: String {
        asrSelectionSummary(featureSettings.transcription.asrSelectionID)
    }

    var onboardingLLMSummary: String {
        llmSelectionSummary(featureSettings.transcription.llmSelectionID)
    }

    var onboardingASRSelectionID: FeatureModelSelectionID {
        OnboardingFeatureSelectionResolver.asrSelectionID(
            selectedEngine: selectedEngine,
            mlxModelRepo: mlxModelRepo,
            remoteASRProvider: selectedRemoteASRProvider
        )
    }

    var onboardingLLMSelectionID: FeatureModelSelectionID {
        OnboardingFeatureSelectionResolver.llmSelectionID(
            choice: llmPathChoice.wrappedValue,
            localLLMRepo: customLLMRepo,
            remoteLLMProvider: selectedRemoteLLMProvider
        )
    }

    func applyLLMPathChoice(_ choice: OnboardingTextModelPathChoice) {
        switch choice {
        case .local:
            enhancementModeRaw = EnhancementMode.customLLM.rawValue
        case .remote:
            enhancementModeRaw = EnhancementMode.remoteLLM.rawValue
        case .system:
            enhancementModeRaw = EnhancementMode.appleIntelligence.rawValue
        }
        syncOnboardingFeatureSelections(usingLLMChoice: choice)
    }

    func syncOnboardingFeatureSelections(usingLLMChoice choice: OnboardingTextModelPathChoice? = nil) {
        let asrSelection = onboardingASRSelectionID
        let llmSelection = llmSelectionID(for: choice ?? llmPathChoice.wrappedValue)

        FeatureSettingsStore.update(defaults: .standard) { settings in
            settings.transcription.asrSelectionID = asrSelection
            settings.transcription.llmEnabled = true
            settings.transcription.llmSelectionID = llmSelection

            settings.translation.asrSelectionID = asrSelection
            settings.translation.modelSelectionID = translationSelectionID(
                from: llmSelection,
                asrSelection: asrSelection,
                existingSelection: settings.translation.modelSelectionID
            )
            settings.translation.targetLanguageRawValue = translationTargetLanguageRaw

            settings.rewrite.asrSelectionID = asrSelection
            settings.rewrite.llmSelectionID = llmSelection
            settings.rewrite.appEnhancementEnabled = true
        }

        featureSettings = FeatureSettingsStore.load(defaults: .standard)
    }

    func llmSelectionID(for choice: OnboardingTextModelPathChoice) -> FeatureModelSelectionID {
        OnboardingFeatureSelectionResolver.llmSelectionID(
            choice: choice,
            localLLMRepo: customLLMRepo,
            remoteLLMProvider: selectedRemoteLLMProvider
        )
    }

    func translationSelectionID(
        from llmSelection: FeatureModelSelectionID,
        asrSelection: FeatureModelSelectionID,
        existingSelection: FeatureModelSelectionID
    ) -> FeatureModelSelectionID {
        OnboardingFeatureSelectionResolver.translationSelectionID(
            llmSelection: llmSelection,
            asrSelection: asrSelection,
            existingSelection: existingSelection,
            fallbackLocalLLMRepo: customLLMRepo
        )
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

    func llmSelectionSummary(_ selectionID: FeatureModelSelectionID) -> String {
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

    func remoteASRStatusText(for provider: RemoteASRProvider) -> String {
        let configuration = RemoteModelConfigurationStore.resolvedASRConfiguration(
            provider: provider,
            stored: remoteASRConfigurations
        )
        guard configuration.isConfigured else {
            return localized("Not configured")
        }

        return AppLocalization.format("Configured model: %@", configuration.model)
    }

    func remoteLLMStatusText(for provider: RemoteLLMProvider) -> String {
        guard RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
            provider: provider,
            stored: remoteLLMConfigurations
        ) else {
            return localized("Not configured")
        }
        let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
            provider: provider,
            stored: remoteLLMConfigurations
        )
        return AppLocalization.format("Configured model: %@", configuration.model)
    }

    func refreshInputDevices() {
        inputDevices = AudioInputDeviceManager.availableInputDevices()
        microphoneState = MicrophonePreferenceManager.syncState(
            defaults: .standard,
            availableDevices: inputDevices
        )
    }

    func refreshModelStorageDisplayPath() {
        let resolution = ModelStorageDirectoryManager.resolvedRootResolution()
        modelStorageDisplayPath = resolution.writeRootURL.path
        modelStorageSelectionError = resolution.accessIssue?.localizedDescription
    }

    func chooseModelStorageDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.directoryURL = ModelStorageDirectoryManager.resolvedRootURL()
        panel.prompt = localized("Choose")

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

    func openModelStorageInFinder() {
        Task { @MainActor in
            ModelStorageDirectoryManager.openRootInFinder()
        }
    }

    func applyHotkeyPreset(_ preset: HotkeyPreference.Preset) {
        hotkeyPresetRaw = preset.rawValue
        guard HotkeyPreference.applyPreset(preset) != nil else { return }
        hotkeyDistinguishModifierSides = true
    }

    func setMicrophoneAutoSwitchEnabled(_ isEnabled: Bool) {
        microphoneState = MicrophonePreferenceManager.setAutoSwitchEnabled(
            isEnabled,
            defaults: .standard,
            availableDevices: inputDevices
        )
        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)
    }

    func applyMicrophonePriorityOrder(_ orderedUIDs: [String]) {
        microphoneState = MicrophonePreferenceManager.reorderPriority(
            orderedUIDs: orderedUIDs,
            defaults: .standard,
            availableDevices: inputDevices
        )
        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)
    }

    func focusMicrophone(uid: String) {
        microphoneState = MicrophonePreferenceManager.setFocusedDevice(
            uid: uid,
            defaults: .standard,
            availableDevices: inputDevices
        )
        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)
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

    func syncLocalizedOnboardingSamples() {
        let localeIdentifier = interfaceLanguage.localeIdentifier
        let englishIdentifier = AppInterfaceLanguage.english.localeIdentifier
        let chineseIdentifier = AppInterfaceLanguage.chineseSimplified.localeIdentifier
        let japaneseIdentifier = AppInterfaceLanguage.japanese.localeIdentifier

        let translationDefaults = Set([
            OnboardingTranslationTest.defaultInput(localeIdentifier: englishIdentifier),
            OnboardingTranslationTest.defaultInput(localeIdentifier: chineseIdentifier),
            OnboardingTranslationTest.defaultInput(localeIdentifier: japaneseIdentifier)
        ])
        if translationDefaults.contains(translationTestInput) {
            translationTestInput = OnboardingTranslationTest.defaultInput(localeIdentifier: localeIdentifier)
        }

        let rewritePromptDefaults = Set([
            OnboardingRewriteTest.defaultPrompt(localeIdentifier: englishIdentifier),
            OnboardingRewriteTest.defaultPrompt(localeIdentifier: chineseIdentifier),
            OnboardingRewriteTest.defaultPrompt(localeIdentifier: japaneseIdentifier)
        ])
        if rewritePromptDefaults.contains(rewriteTestPrompt) {
            rewriteTestPrompt = OnboardingRewriteTest.defaultPrompt(localeIdentifier: localeIdentifier)
        }

        let rewriteSourceDefaults = Set([
            OnboardingRewriteTest.defaultSourceText(localeIdentifier: englishIdentifier),
            OnboardingRewriteTest.defaultSourceText(localeIdentifier: chineseIdentifier),
            OnboardingRewriteTest.defaultSourceText(localeIdentifier: japaneseIdentifier)
        ])
        if rewriteSourceDefaults.contains(rewriteTestSourceText) {
            rewriteTestSourceText = OnboardingRewriteTest.defaultSourceText(localeIdentifier: localeIdentifier)
        }
    }
}
