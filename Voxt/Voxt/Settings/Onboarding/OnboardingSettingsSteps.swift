// OnboardingSettingsSteps.swift
// Provides Onboarding Settings Steps for onboarding settings.

import SwiftUI
import AppKit

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

extension OnboardingSettingsView {
    @ViewBuilder
    var stepContent: some View {
        switch currentStep {
        case .language:
            languageStep
        case .model:
            modelStep
        case .transcription:
            transcriptionStep
        case .translation:
            translationStep
        case .rewrite:
            rewriteStep
        case .meeting:
            meetingStep
        case .appEnhancement:
            appEnhancementStep
        case .finish:
            finishStep
        }
    }

    var languageStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            GeneralSettingsCard(title: "Language") {
                GeneralLanguageSettingBlock(
                    title: "Interface Language",
                    description: "Choose the language used by the main window and menu labels."
                ) {
                    SettingsMenuPicker(
                        selection: $interfaceLanguageRaw,
                        options: AppInterfaceLanguage.allCases.map { language in
                            SettingsMenuOption(value: language.rawValue, title: language.title)
                        },
                        selectedTitle: interfaceLanguage.title,
                        width: 220
                    )
                }

                Divider()

                GeneralLanguageSettingBlock(
                    title: "User Main Language",
                    description: "Used to bias transcription prompts, translation prompts, and text enhancement."
                ) {
                    SettingsSelectionButton(width: 220, action: { isUserMainLanguageSheetPresented = true }) {
                        HStack(spacing: 0) {
                            Text(userMainLanguageSummary)
                                .lineLimit(1)
                                .truncationMode(.tail)
                        }
                    }
                }

            }

            OnboardingSummaryCard(
                title: "Current Summary",
                lines: [
                    AppLocalization.format("Interface: %@", interfaceLanguage.title),
                    AppLocalization.format("Main language: %@", userMainLanguageSummary)
                ]
            )
        }
    }

    var modelStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            modelStorageCard

            GeneralSettingsCard(title: "ASR Model") {
                OnboardingSegmentedTabs(
                    selection: modelPathChoice,
                    items: OnboardingModelPathChoice.allCases.map { choice in
                        OnboardingTabItem(value: choice, title: choice.titleKey)
                    }
                )

                Text(modelPathDescription(modelPathChoice.wrappedValue))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                switch modelPathChoice.wrappedValue {
                case .local:
                    localASRSetupContent
                case .remote:
                    remoteASRSetupContent
                case .dictation:
                    dictationASRSetupContent
                }
            }

            GeneralSettingsCard(title: "LLM Model") {
                OnboardingSegmentedTabs(
                    selection: llmPathChoice,
                    items: OnboardingTextModelPathChoice.allCases.map { choice in
                        OnboardingTabItem(value: choice, title: choice.titleKey)
                    }
                )

                Text(llmPathDescription(llmPathChoice.wrappedValue))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                switch llmPathChoice.wrappedValue {
                case .local:
                    localLLMSetupContent
                case .remote:
                    remoteLLMSetupContent
                case .system:
                    systemLLMSetupContent
                }
            }
        }
    }

    var localASRSetupContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            OnboardingSegmentedTabs(
                selection: localEngineSelection,
                items: [
                    OnboardingTabItem(value: TranscriptionEngine.mlxAudio, title: TranscriptionEngine.mlxAudio.titleKey)
                ]
            )

            Text(selectedLocalEngineDescription)
                .font(.caption)
                .foregroundStyle(.secondary)

            LocalModelPickerCard(
                title: "Speech Model",
                selectionTitle: mlxModelManager.displayTitle(for: mlxModelRepo),
                selectionDescription: modelDescription(for: mlxModelRepo),
                isInstalled: mlxModelManager.isModelDownloaded(repo: mlxModelRepo),
                showsCardSurface: false,
                isInstalling: isSelectedMLXModelDownloading,
                isPaused: isSelectedMLXModelPaused,
                isInstallEnabled: !isAnotherMLXModelDownloading && MLXModelManager.isAvailableModelRepo(mlxModelRepo),
                installLabel: "Download",
                openLabel: "Open Folder",
                downloadStatus: selectedMLXDownloadStatus,
                errorMessage: selectedMLXDownloadErrorMessage,
                onChoose: {},
                pickerContent: {
                    SettingsMenuPicker(
                        selection: $mlxModelRepo,
                        options: mlxModelManager.displayModelsIncludingInstalled().map { model in
                            SettingsMenuOption(value: model.id, title: model.title)
                        },
                        selectedTitle: mlxModelManager.displayTitle(for: mlxModelRepo),
                        width: 280
                    )
                },
                onInstall: {
                    Task { await mlxModelManager.downloadModel(repo: mlxModelRepo) }
                },
                onOpen: {
                    guard let folderURL = mlxModelManager.modelDirectoryURL(repo: mlxModelRepo) else { return }
                    NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: folderURL.path)
                },
                onPause: {
                    mlxModelManager.pauseDownload(repo: mlxModelRepo)
                },
                onResume: {
                    Task { await mlxModelManager.downloadModel(repo: mlxModelRepo) }
                },
                onCancel: {
                    mlxModelManager.cancelDownload()
                },
                onUninstall: {
                    Task {
                        await mlxModelManager.deleteModel(repo: mlxModelRepo)
                        mlxModelManager.checkExistingModel()
                    }
                }
            )
        }
    }

    var remoteASRSetupContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            SettingsMenuPicker(
                selection: Binding(
                    get: { selectedRemoteASRProvider.rawValue },
                    set: { newValue in
                        remoteASRSelectedProviderRaw = newValue
                        engineRaw = TranscriptionEngine.remote.rawValue
                    }
                ),
                options: RemoteASRProvider.allCases.map { provider in
                    SettingsMenuOption(value: provider.rawValue, title: provider.title)
                },
                selectedTitle: selectedRemoteASRProvider.title,
                width: 280
            )

            ProviderStatusRow(
                title: selectedRemoteASRProvider.title,
                status: remoteASRStatusText(for: selectedRemoteASRProvider),
                onConfigure: {
                    editingASRProvider = selectedRemoteASRProvider
                }
            )
        }
    }

    var dictationASRSetupContent: some View {
        OnboardingSummaryCard(
            title: "System ASR",
            lines: [
                localized("Uses the macOS system speech recognizer with no download required."),
                localized("Best for the fastest setup, but advanced transcription features and language coverage are more limited.")
            ]
        )
    }

    var localLLMSetupContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            LocalModelPickerCard(
                title: LocalizedStringKey(localized("Language Model")),
                selectionTitle: customLLMManager.displayTitle(for: customLLMRepo),
                selectionDescription: customLLMDescription(for: customLLMRepo),
                isInstalled: customLLMManager.isModelDownloaded(repo: customLLMRepo),
                showsCardSurface: false,
                isInstalling: isSelectedCustomLLMDownloading,
                isPaused: isSelectedCustomLLMPaused,
                isInstallEnabled: true,
                installLabel: "Download",
                openLabel: "Open Folder",
                downloadStatus: customLLMDownloadStatus,
                errorMessage: customLLMDownloadErrorMessage,
                onChoose: {},
                pickerContent: {
                    SettingsMenuPicker(
                        selection: Binding(
                            get: { customLLMRepo },
                            set: { newValue in
                                customLLMRepo = newValue
                                translationCustomLLMRepo = newValue
                                rewriteCustomLLMRepo = newValue
                            }
                        ),
                        options: CustomLLMModelManager.displayModels(including: customLLMRepo).map { model in
                            SettingsMenuOption(value: model.id, title: model.title)
                        },
                        selectedTitle: customLLMManager.displayTitle(for: customLLMRepo),
                        width: 280
                    )
                },
                onInstall: {
                    Task { await customLLMManager.downloadModel(repo: customLLMRepo) }
                },
                onOpen: {
                    guard let folderURL = customLLMManager.modelDirectoryURL(repo: customLLMRepo) else { return }
                    NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: folderURL.path)
                },
                onPause: {
                    customLLMManager.pauseDownload(repo: customLLMRepo)
                },
                onResume: {
                    Task { await customLLMManager.downloadModel(repo: customLLMRepo) }
                },
                onCancel: {
                    customLLMManager.cancelDownload(repo: customLLMRepo)
                },
                onUninstall: {
                    Task { await customLLMManager.deleteModel(repo: customLLMRepo) }
                }
            )
        }
    }

    var remoteLLMSetupContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            SettingsMenuPicker(
                selection: Binding(
                    get: { selectedRemoteLLMProvider.rawValue },
                    set: { newValue in
                        remoteLLMSelectedProviderRaw = newValue
                        translationRemoteLLMProviderRaw = newValue
                        rewriteRemoteLLMProviderRaw = newValue
                        enhancementModeRaw = EnhancementMode.remoteLLM.rawValue
                    }
                ),
                options: RemoteLLMProvider.allCases.map { provider in
                    SettingsMenuOption(value: provider.rawValue, title: provider.title)
                },
                selectedTitle: selectedRemoteLLMProvider.title,
                width: 280
            )

            ProviderStatusRow(
                title: selectedRemoteLLMProvider.title,
                status: remoteLLMStatusText(for: selectedRemoteLLMProvider),
                onConfigure: {
                    editingLLMProvider = selectedRemoteLLMProvider
                }
            )
        }
    }

    var systemLLMSetupContent: some View {
        OnboardingSummaryCard(
            title: "Apple Intelligence",
            lines: appleIntelligenceAvailable
                ? [
                    localized("Use the system model for cleanup, rewrite, and transcript summaries on this Mac."),
                    localized("Translation will keep using the best compatible model automatically.")
                ]
                : [
                    localized("Apple Intelligence is currently unavailable on this Mac."),
                    localized("You can still keep this selected and switch later after the system becomes available.")
                ]
        )
    }

    var modelStorageCard: some View {
        GeneralSettingsCard(title: "Model Storage") {
            SettingsPathSelectionRow(
                title: localized("Storage Path"),
                displayedPath: modelStorageDisplayPath,
                fallbackPath: ModelStorageDirectoryManager.defaultRootURL.path,
                openButtonHelp: localized("Open folder"),
                chooseButtonTitle: localized("Choose"),
                onOpen: openModelStorageInFinder,
                onChoose: chooseModelStorageDirectory
            )

            Text(localized("New model downloads are stored here. Switching the path will not move existing model files."))
                .font(.caption)
                .foregroundStyle(.secondary)

            Text(localized("Model downloads automatically choose the fastest available source when installation starts."))
                .font(.caption)
                .foregroundStyle(.secondary)

            if let modelStorageSelectionError {
                Text(modelStorageSelectionError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
    }

    var transcriptionStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            GeneralSettingsCard(title: "Audio") {
                HStack(alignment: .center) {
                    Text(localized("Microphone"))
                        .foregroundStyle(.secondary)
                    Spacer()
                    if microphoneState.hasAvailableDevices {
                        SettingsSelectionButton(width: 272, action: { isMicrophonePriorityDialogPresented = true }) {
                            HStack(spacing: 0) {
                                Text(microphoneState.activeDevice?.name ?? localized("No available microphone devices"))
                                    .lineLimit(1)
                            }
                        }
                    } else {
                        Text(localized("No available microphone devices"))
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.red)
                    }
                }

                Toggle(localized("Interaction Sounds"), isOn: $interactionSoundsEnabled)
                Text(localized("Play a start chime when recording begins and an end chime when processing finishes."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle(localized("Mute output device while recording"), isOn: $muteSystemAudioWhileRecording)
                Text(localized("Mutes all sounds on the current output device, including Voxt. Some devices do not support software mute. No system audio recording permission is needed."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

            }

            GeneralSettingsCard(title: "Shortcut") {
                OnboardingSegmentedTabs(
                    selection: hotkeyPresetSelection,
                    items: [HotkeyPreference.Preset.fnCombo, .commandCombo, .mouseMiddleFnShift].map { preset in
                        OnboardingTabItem(value: preset, title: LocalizedStringKey(preset.title))
                    }
                )

                Text(hotkeyPresetDescription)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if hotkeyPresetSelection.wrappedValue == .fnCombo {
                    Text(localized("On macOS, fn shortcuts may conflict with Globe or input source switching. If needed, change that shortcut in System Settings > Keyboard > Keyboard Shortcuts > Input Sources."))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            OnboardingSummaryCard(
                title: "Quick Test",
                lines: [
                    AppLocalization.format("Current preset: %@", hotkeyPresetSelection.wrappedValue.title),
                    AppLocalization.format("Transcription shortcut: %@", formattedTranscriptionHotkey),
                    AppLocalization.format("Focus the textarea below, then press %@ to test transcription directly.", formattedTranscriptionHotkey)
                ]
            )

            TranscriptionTestSectionView()
        }
    }

    var translationStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            GeneralSettingsCard(title: "Translation") {
                HStack(alignment: .center) {
                    Text(localized("Target Language"))
                        .foregroundStyle(.secondary)
                    Spacer()
                    SettingsMenuPicker(
                        selection: $translationTargetLanguageRaw,
                        options: TranslationTargetLanguage.allCases.map { language in
                            SettingsMenuOption(value: language.rawValue, title: language.title)
                        },
                        selectedTitle: translationTargetLanguage.title,
                        width: 220
                    )
                }

                Toggle(localized("Also copy result to clipboard"), isOn: $autoCopyWhenNoFocusedInput)
                Text(localized("When enabled, translated text is kept in the clipboard in addition to being inserted into the current input."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            GeneralSettingsCard(title: "Test Translation") {
                Text(localized("Use the textarea below to test the translation shortcut directly."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                OnboardingSummaryCard(
                    title: "Quick Test",
                    lines: translationShortcutTestLines
                )

                TextEditor(text: $translationTestInput)
                    .settingsPromptEditor(height: 110, contentPadding: 6)

                HStack(spacing: 8) {
                    Button(localized("Use Sample")) {
                        translationTestInput = OnboardingTranslationTest.defaultInput
                    }
                    .buttonStyle(SettingsPillButtonStyle())

                    Button(localized("Clean")) {
                        translationTestInput = ""
                    }
                    .buttonStyle(SettingsPillButtonStyle())
                }
            }
        }
    }

    var rewriteStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            GeneralSettingsCard(title: "Rewrite") {
                Text(localized("Use voice to rewrite selected text, or speak a full prompt when nothing is selected."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            GeneralSettingsCard(title: "How It Works") {
                OnboardingExampleRow(
                    title: "Rewrite Selected Text",
                    detail: "Select a paragraph, then say something like: make this shorter and more polite."
                )
                Divider()
                OnboardingExampleRow(
                    title: "Voice Prompt Mode",
                    detail: "With no selected text, say something like: write a follow-up email to the client about tomorrow's launch."
                )
            }

            GeneralSettingsCard(title: "Test Rewrite") {
                Text(localized("Use the textarea below to test the rewrite shortcut directly."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                OnboardingSummaryCard(
                    title: "Quick Test",
                    lines: rewriteShortcutTestLines
                )

                Text(localized("Instruction"))
                    .font(.subheadline.weight(.medium))

                TextField(localized("Make this shorter and more polite."), text: $rewriteTestPrompt)
                    .textFieldStyle(.plain)
                    .settingsFieldSurface()

                Text(localized("Source Text"))
                    .font(.subheadline.weight(.medium))

                TextEditor(text: $rewriteTestSourceText)
                    .settingsPromptEditor(height: 120, contentPadding: 6)

                HStack(spacing: 8) {
                    Button(localized("Use Sample")) {
                        rewriteTestPrompt = OnboardingRewriteTest.defaultPrompt
                        rewriteTestSourceText = OnboardingRewriteTest.defaultSourceText
                    }
                    .buttonStyle(SettingsPillButtonStyle())

                    Button(localized("Clean")) {
                        rewriteTestPrompt = ""
                        rewriteTestSourceText = ""
                    }
                    .buttonStyle(SettingsPillButtonStyle())
                }
            }
        }
    }

    var appEnhancementStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            GeneralSettingsCard(title: "App Enhancement") {
                Text(localized("App Enhancement lets Voxt switch prompts based on the current app or browser tab, so translation, rewrite, and cleanup can behave differently across contexts."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            GeneralSettingsCard(title: "Demo Video") {
                Text(localized("Watch a short example of how App Enhancement behaves across apps and pages."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if let appEnhancementDemoPlayer {
                    OnboardingVideoPlayerView(player: appEnhancementDemoPlayer)
                        .frame(maxWidth: .infinity)
                        .frame(height: 220)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
                        )
                } else {
                    ProgressView()
                        .frame(maxWidth: .infinity)
                        .frame(height: 220)
                }
            }
        }
    }

    var meetingStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            GeneralSettingsCard(title: "Meeting") {
                Text(localized("Meeting captures live audio into a transcript, then uses the selected summary model and speaker separation settings for meeting details."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                OnboardingSummaryCard(
                    title: "Current Meeting Setup",
                    lines: [
                        AppLocalization.format("Shortcut: %@", formattedMeetingHotkey),
                        AppLocalization.format("Audio model: %@", meetingASRSummary),
                        AppLocalization.format("Summary model: %@", meetingSummaryModelSummary),
                        AppLocalization.format("Segmentation: %@", featureSettings.meeting.chunkingMode.title),
                        AppLocalization.format("Speaker separation: %@", featureSettings.meeting.speakerDiarizationModel.title),
                        AppLocalization.format(
                            "Auto summary: %@",
                            featureSettings.meeting.summaryAutoGenerate ? localized("Enabled") : localized("Disabled")
                        )
                    ]
                )
            }

            GeneralSettingsCard(title: "Behavior") {
                OnboardingExampleRow(
                    title: "Live Transcript",
                    detail: "The meeting audio model is separate from translation and rewrite, and Direct Dictation is not used for Meeting mode."
                )
                Divider()
                OnboardingExampleRow(
                    title: "Summary",
                    detail: "Meeting summaries use the selected text model and can be generated automatically after recording."
                )
                Divider()
                OnboardingExampleRow(
                    title: "Speaker Separation",
                    detail: "Speaker labels are produced after recording when the selected speaker separation model is ready."
                )
            }
        }
    }

    var finishStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            GeneralSettingsCard(title: "You're Ready") {
                Text(localized("Voxt is configured enough to start. You can still refine any detail later from the normal settings pages."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                OnboardingSummaryCard(
                    title: "Current Setup",
                    lines: [
                        AppLocalization.format("Language: %@", interfaceLanguage.title),
                        AppLocalization.format("ASR: %@", onboardingASRSummary),
                        AppLocalization.format("LLM: %@", onboardingLLMSummary),
                        AppLocalization.format("Translation: %@", translationProviderSummary),
                        AppLocalization.format("Rewrite: %@", rewriteProviderSummary),
                        AppLocalization.format("Meeting summary: %@", meetingSummaryModelSummary)
                    ]
                )

                Divider()

                VStack(alignment: .leading, spacing: 12) {
                    OnboardingFeatureSummaryRow(
                        iconKind: .meeting,
                        title: "Meeting",
                        detail: "Capture microphone and system audio into a speaker-labelled transcript.",
                        status: meetingASRSummary,
                        statusTint: .accentColor
                    )

                    Divider()

                    OnboardingFeatureSummaryRow(
                        iconKind: .note,
                        title: "Notes",
                        detail: "Capture key points during recording. Notes stay separate and get short AI titles.",
                        status: notesStatusSummary,
                        statusTint: .green
                    )

                    Divider()

                    OnboardingFeatureSummaryRow(
                        iconKind: .appEnhancement,
                        title: "App Enhancement",
                        detail: "Use different enhancement prompts for different apps or browser pages.",
                        status: featureSettings.rewrite.appEnhancementEnabled ? localized("Enabled") : localized("Disabled"),
                        statusTint: featureSettings.rewrite.appEnhancementEnabled ? .green : .secondary
                    )
                }

                HStack {
                    Spacer()
                    Button(localized("Start Voxt")) {
                        onFinish()
                    }
                    .buttonStyle(SettingsPrimaryButtonStyle())
                }
            }
        }
    }

    @ViewBuilder
    var currentStepPermissionSection: some View {
        let permissions = currentStepRequiredPermissions
        if !permissions.isEmpty {
            GeneralSettingsCard(title: "Permissions For This Step") {
                Text(localized("Grant the permissions required for this step. If a permission is disabled, you can request it here or jump to System Settings."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                ForEach(Array(permissions.enumerated()), id: \.offset) { index, permission in
                    if index > 0 {
                        Divider()
                    }

                    HStack(alignment: .top, spacing: 12) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(permission.titleKey)
                                .font(.subheadline)
                            Text(permission.descriptionKey)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        Spacer()

                        if permissionMonitoringKinds.contains(permission) {
                            ProgressView()
                                .controlSize(.small)
                                .frame(width: 14, height: 14)
                        }

                        OnboardingPermissionStatusBadge(isGranted: isPermissionGranted(permission))

                        if !isPermissionGranted(permission) {
                            Button(localized("Request")) {
                                requestPermission(permission)
                            }
                            .buttonStyle(SettingsCompactActionButtonStyle())

                            Button(localized("Open Settings")) {
                                openSettings(for: permission)
                            }
                            .buttonStyle(SettingsCompactActionButtonStyle())
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    var selectedLocalEngineDescription: String {
        switch localEngineSelection.wrappedValue {
        case .mlxAudio:
            return TranscriptionEngine.mlxAudio.description
        case .dictation, .remote:
            return ""
        }
    }

    func modelPathDescription(_ choice: OnboardingModelPathChoice) -> String {
        switch choice {
        case .local:
            return localized("Runs speech and text models on your Mac. Better privacy, works offline after download, but needs local disk space and setup time.")
        case .remote:
            return localized("Uses cloud providers for speech and text. Fast to start, but requires network access and provider configuration.")
        case .dictation:
            return localized("Uses Apple's built-in dictation path with the lightest setup, but fewer supported scenarios.")
        }
    }

    func llmPathDescription(_ choice: OnboardingTextModelPathChoice) -> String {
        switch choice {
        case .local:
            return localized("Uses a local text model for cleanup, translation, rewrite, and summaries after download.")
        case .remote:
            return localized("Uses a cloud text model. Fast to start, but requires network access and provider configuration.")
        case .system:
            return localized("Uses the macOS system model when available, so later onboarding can stay minimal.")
        }
    }

    func modelDescription(for repo: String) -> String {
        guard let description = MLXModelCatalog.description(for: MLXModelManager.canonicalModelRepo(repo)) else {
            return ""
        }
        return AppLocalization.localizedString(description)
    }

    func customLLMDescription(for repo: String) -> String {
        guard let description = customLLMManager.description(for: repo) else {
            return ""
        }
        return AppLocalization.localizedString(description)
    }

    var isSelectedMLXModelDownloading: Bool {
        mlxModelManager.isDownloading(repo: mlxModelRepo)
    }

    var isAnotherMLXModelDownloading: Bool {
        mlxModelManager.activeDownloadRepos.contains(where: { repo in
            MLXModelManager.canonicalModelRepo(repo) != MLXModelManager.canonicalModelRepo(mlxModelRepo)
        })
    }

    var isSelectedMLXModelPaused: Bool {
        mlxModelManager.isPaused(repo: mlxModelRepo)
    }

    var selectedMLXDownloadStatus: ModelDownloadStatusSnapshot? {
        guard isSelectedMLXModelDownloading || isSelectedMLXModelPaused else { return nil }
        return ModelDownloadStatusSnapshot.fromMLXState(
            mlxModelManager.state(for: mlxModelRepo),
            pauseMessage: mlxModelManager.pausedStatusMessage(for: mlxModelRepo)
        )
    }

    var selectedMLXDownloadErrorMessage: String? {
        guard MLXModelManager.canonicalModelRepo(mlxModelManager.currentModelRepo) == MLXModelManager.canonicalModelRepo(mlxModelRepo),
              case .error(let message) = mlxModelManager.state else {
            return nil
        }
        return message
    }

    var isSelectedCustomLLMDownloading: Bool {
        customLLMManager.isDownloading(repo: customLLMRepo)
    }

    var isSelectedCustomLLMPaused: Bool {
        customLLMManager.isPaused(repo: customLLMRepo)
    }

    var customLLMDownloadStatus: ModelDownloadStatusSnapshot? {
        guard isSelectedCustomLLMDownloading || isSelectedCustomLLMPaused else { return nil }
        return ModelDownloadStatusSnapshot.fromCustomLLMState(
            customLLMManager.state(for: customLLMRepo),
            pauseMessage: customLLMManager.pausedStatusMessage(for: customLLMRepo)
        )
    }

    var customLLMDownloadErrorMessage: String? {
        guard case .error(let message) = customLLMManager.state(for: customLLMRepo) else {
            return nil
        }
        return message
    }

    var hotkeyPresetDescription: String {
        switch hotkeyPresetSelection.wrappedValue {
        case .fnCombo:
            return localized("Recommended default: fn for transcription, fn+shift for translation, and fn+control for rewrite.")
        case .commandCombo:
            return localized("Useful when function-key combinations are already reserved by the system or keyboard tools.")
        case .mouseMiddleFnShift:
            return localized("Uses mouse middle button for transcription, fn+shift for translation, and double-tap middle button for content rewrite.")
        case .custom:
            return localized("You can fine-tune every shortcut later from the Hotkey page.")
        }
    }

    var translationShortcutTestLines: [String] {
        let lines = [
            AppLocalization.format("Current preset: %@", hotkeyPresetSelection.wrappedValue.title),
            AppLocalization.format("Translation shortcut: %@", formattedTranslationHotkey),
            AppLocalization.format("Select text in the textarea below, then press %@ to translate the selection directly.", formattedTranslationHotkey)
        ]

        return lines
    }

    var rewriteShortcutTestLines: [String] {
        [
            AppLocalization.format("Current preset: %@", hotkeyPresetSelection.wrappedValue.title),
            AppLocalization.format("Rewrite shortcut: %@", formattedRewriteHotkey),
            AppLocalization.format("Select text in the textarea below, then press %@ to rewrite the selection directly.", formattedRewriteHotkey),
            AppLocalization.format("If nothing is selected, %@ starts voice prompt mode instead.", formattedRewriteHotkey)
        ]
    }
}
