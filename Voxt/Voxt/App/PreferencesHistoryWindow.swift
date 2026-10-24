// PreferencesHistoryWindow.swift
// Provides Preferences History Window for app lifecycle and routing.

import Foundation
import AppKit
import CoreAudio

extension AppDelegate {
    private struct HistoryTextModelMetadata {
        let modeTitle: String
        let modelTitle: String
        let remoteProviderTitle: String?
        let remoteModelTitle: String?
        let remoteEndpoint: String?
    }

    private var defaults: UserDefaults {
        .standard
    }

    var featureSettings: FeatureSettings {
        FeatureSettingsStore.load(defaults: defaults)
    }

    func updateFeatureSettings(_ mutate: (inout FeatureSettings) -> Void) {
        FeatureSettingsStore.update(defaults: defaults, mutate)
    }

    var transcriptionFeatureSettings: TranscriptionFeatureSettings {
        featureSettings.transcription
    }

    var rewriteContinueShortcutSettings: TranscriptionContinueShortcutSettings {
        featureSettings.rewrite.continueShortcut
    }

    var noteFeatureSettings: TranscriptionNoteFeatureSettings {
        featureSettings.transcription.notes
    }

    var translationFeatureSettings: TranslationFeatureSettings {
        featureSettings.translation
    }

    var rewriteFeatureSettings: RewriteFeatureSettings {
        featureSettings.rewrite
    }

    var meetingFeatureSettings: MeetingFeatureSettings {
        featureSettings.meeting
    }

    func prepareLegacySettingsForSession(outputMode: SessionOutputMode) {
        FeatureSettingsStore.prepareLegacySession(
            from: featureSettings,
            outputMode: outputMode,
            defaults: defaults
        )
    }

    func prepareSettingsForMeetingRuntime() {
        FeatureSettingsStore.prepareMeetingRuntime(
            from: featureSettings,
            defaults: defaults
        )
    }

    var selectedInputDeviceUID: String? {
        MicrophonePreferenceManager.activeInputDeviceUID(defaults: defaults)
    }

    var selectedInputDeviceID: AudioDeviceID? {
        MicrophonePreferenceManager.activeInputDeviceID(
            defaults: defaults,
            availableDevices: inputDevicesSnapshot
        )
    }

    var microphoneAutoSwitchEnabled: Bool {
        MicrophonePreferenceManager.autoSwitchEnabled(defaults: defaults)
    }

    var interactionSoundsEnabled: Bool {
        defaults.bool(forKey: AppPreferenceKey.interactionSoundsEnabled)
    }

    var muteSystemAudioWhileRecording: Bool {
        defaults.bool(forKey: AppPreferenceKey.muteSystemAudioWhileRecording)
    }

    var overlayPosition: OverlayPosition {
        enumValue(forKey: AppPreferenceKey.overlayPosition, default: .bottom)
    }

    var autoCopyWhenNoFocusedInput: Bool {
        defaults.bool(forKey: AppPreferenceKey.autoCopyWhenNoFocusedInput)
    }

    var alwaysShowRewriteAnswerCard: Bool {
        defaults.bool(forKey: AppPreferenceKey.alwaysShowRewriteAnswerCard)
    }

    var translationTargetLanguage: TranslationTargetLanguage {
        enumValue(forKey: AppPreferenceKey.translationTargetLanguage, default: .english)
    }

    var userMainLanguageCodes: [String] {
        UserMainLanguageOption.storedSelection(
            from: defaults.string(forKey: AppPreferenceKey.userMainLanguageCodes)
        )
    }

    var userMainLanguage: UserMainLanguageOption {
        let selectedCodes = userMainLanguageCodes
        if let firstCode = selectedCodes.first,
           let option = UserMainLanguageOption.option(for: firstCode) {
            return option
        }
        return UserMainLanguageOption.fallbackOption()
    }

    var userMainLanguagePromptValue: String {
        userMainLanguage.promptName
    }

    var userOtherMainLanguagesPromptValue: String {
        DictionaryHistoryScanPromptLanguageSupport.otherLanguagesPromptValue(
            from: userMainLanguageCodes
        )
    }

    var showSelectedTextTranslationResultWindow: Bool {
        defaults.object(forKey: AppPreferenceKey.showSelectedTextTranslationResultWindow) as? Bool ?? true
    }

    var meetingRealtimeTranslationTargetLanguage: TranslationTargetLanguage? {
        resolvedMeetingRealtimeTranslationTargetLanguage()
    }

    var customPasteHotkeyEnabled: Bool {
        defaults.bool(forKey: AppPreferenceKey.customPasteHotkeyEnabled)
    }

    var voiceEndCommandEnabled: Bool {
        defaults.bool(forKey: AppPreferenceKey.voiceEndCommandEnabled)
    }

    var voiceEndCommandPreset: VoiceEndCommandPreset {
        if let preset = enumValue(forKey: AppPreferenceKey.voiceEndCommandPreset, default: Optional<VoiceEndCommandPreset>.none) {
            return preset
        }

        let legacyCustomValue = trimmedStringValue(forKey: AppPreferenceKey.voiceEndCommandText)
        return legacyCustomValue.isEmpty ? .over : .custom
    }

    var voiceEndCommandText: String {
        if let presetCommand = voiceEndCommandPreset.resolvedCommand {
            return presetCommand
        }
        return trimmedStringValue(forKey: AppPreferenceKey.voiceEndCommandText)
    }

    var translationSystemPrompt: String {
        translationFeatureSettings.prompt
    }

    var translationCustomLLMRepo: String {
        let value = defaults.string(forKey: AppPreferenceKey.translationCustomLLMModelRepo)
        if let value, !value.isEmpty {
            return value
        }
        return defaults.string(forKey: AppPreferenceKey.customLLMModelRepo)
            ?? CustomLLMModelManager.defaultModelRepo
    }

    var translationGGUFModelID: GGUFTranslationModelID {
        GGUFTranslationModelCatalog.resolvedModelID(
            defaults.string(forKey: AppPreferenceKey.translationGGUFModelID)
        )
    }

    var translationModelProvider: TranslationModelProvider {
        enumValue(forKey: AppPreferenceKey.translationModelProvider, default: .customLLM)
    }

    var translationFallbackModelProvider: TranslationModelProvider {
        let stored = enumValue(forKey: AppPreferenceKey.translationFallbackModelProvider, default: Optional<TranslationModelProvider>.none)
        return TranslationProviderResolver.sanitizedFallbackProvider(stored ?? .customLLM)
    }

    var remoteASRSelectedProvider: RemoteASRProvider {
        enumValue(forKey: AppPreferenceKey.remoteASRSelectedProvider, default: .openAIWhisper)
    }

    var remoteLLMSelectedProvider: RemoteLLMProvider {
        enumValue(forKey: AppPreferenceKey.remoteLLMSelectedProvider, default: .openAI)
    }

    var translationRemoteLLMProvider: RemoteLLMProvider? {
        enumValue(forKey: AppPreferenceKey.translationRemoteLLMProvider, default: Optional<RemoteLLMProvider>.none)
    }

    var rewriteSystemPrompt: String {
        rewriteFeatureSettings.prompt
    }

    var rewriteCustomLLMRepo: String {
        let value = defaults.string(forKey: AppPreferenceKey.rewriteCustomLLMModelRepo)
        if let value, !value.isEmpty {
            return value
        }
        return defaults.string(forKey: AppPreferenceKey.customLLMModelRepo)
            ?? CustomLLMModelManager.defaultModelRepo
    }

    var rewriteModelProvider: RewriteModelProvider {
        enumValue(forKey: AppPreferenceKey.rewriteModelProvider, default: .customLLM)
    }

    var rewriteRemoteLLMProvider: RemoteLLMProvider? {
        enumValue(forKey: AppPreferenceKey.rewriteRemoteLLMProvider, default: Optional<RemoteLLMProvider>.none)
    }

    var remoteASRConfigurations: [String: RemoteProviderConfiguration] {
        remoteConfigurations(forKey: AppPreferenceKey.remoteASRProviderConfigurations)
    }

    var remoteLLMConfigurations: [String: RemoteProviderConfiguration] {
        remoteConfigurations(forKey: AppPreferenceKey.remoteLLMProviderConfigurations)
    }

    var showInDock: Bool {
        defaults.bool(forKey: AppPreferenceKey.showInDock)
    }

    var realtimeTextDisplayEnabled: Bool {
        defaults.object(forKey: AppPreferenceKey.realtimeTextDisplayEnabled) as? Bool ?? true
    }

    var localModelIdleUnloadDelaySeconds: Int {
        AppPreferenceKey.resolvedLocalModelIdleUnloadDelaySeconds(defaults: defaults)
    }

    var historyEnabled: Bool {
        true
    }

    var historyAudioStorageEnabled: Bool {
        defaults.object(forKey: AppPreferenceKey.historyAudioStorageEnabled) as? Bool ?? false
    }

    var dictionaryAutoLearningEnabled: Bool {
        defaults.object(forKey: AppPreferenceKey.dictionaryAutoLearningEnabled) as? Bool ?? true
    }

    var dictionaryAutoLearningPrompt: String {
        AppPromptDefaults.resolvedStoredText(
            defaults.string(forKey: AppPreferenceKey.dictionaryAutoLearningPrompt),
            kind: .dictionaryAutoLearning,
            defaults: defaults
        )
    }

    var autoCheckForUpdates: Bool {
        defaults.bool(forKey: AppPreferenceKey.autoCheckForUpdates)
    }

    func appendHistoryIfNeeded(
        text: String,
        outputMode: SessionOutputMode,
        outputDestinationContext: OutputDestinationContext?,
        displayTitle: String? = nil,
        llmDurationSeconds: TimeInterval?,
        dictionaryHitTerms: [String],
        dictionaryCorrectedTerms: [String],
        dictionaryCorrectionSnapshots: [DictionaryCorrectionSnapshot] = [],
        dictionarySuggestedTerms: [DictionarySuggestionSnapshot],
        rewriteConversationTurns: [RewriteConversationTurn] = []
    ) -> UUID? {
        guard historyEnabled else {
            discardPendingCompletedHistoryAudio()
            return nil
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            discardPendingCompletedHistoryAudio()
            return nil
        }
        let trimmedDisplayTitle = displayTitle?.trimmingCharacters(in: .whitespacesAndNewlines)

        let transcriptionModel: String
        switch transcriptionEngine {
        case .dictation:
            transcriptionModel = "Apple Speech Recognition"
        case .mlxAudio:
            let repo = mlxModelManager.currentModelRepo
            transcriptionModel = "\(mlxModelManager.displayTitle(for: repo)) (\(repo))"
        case .remote:
            let provider = remoteASRSelectedProvider
            if let config = remoteASRConfigurations[provider.rawValue], config.hasUsableModel {
                transcriptionModel = "\(provider.title) (\(config.model))"
            } else {
                transcriptionModel = provider.title
            }
        }

        let historyKind = resolvedHistoryKind(for: outputMode)
        VoxtLog.history(
            "History append requested. kind=\(historyKind.rawValue), engine=\(transcriptionEngine.rawValue), historyEnabled=\(historyEnabled), audioStorageEnabled=\(historyAudioStorageEnabled), stashedAudio=\(pendingCompletedHistoryAudioArchiveURL != nil)",
            verbose: true
        )
        let pendingAudioArchiveURL = consumePendingCompletedHistoryAudioURL()
        if let pendingAudioArchiveURL {
            let exists = FileManager.default.fileExists(atPath: pendingAudioArchiveURL.path)
            VoxtLog.history(
                "History append consumed pending audio archive. kind=\(historyKind.rawValue), file=\(pendingAudioArchiveURL.lastPathComponent), exists=\(exists)",
                verbose: true
            )
        } else {
            VoxtLog.historyWarning(
                "History append found no pending audio archive. kind=\(historyKind.rawValue), engine=\(transcriptionEngine.rawValue)"
            )
        }
        let textModelMetadata = resolvedHistoryTextModelMetadata(for: historyKind)

        let now = Date()
        let audioDuration = resolvedDuration(from: recordingStartedAt, to: recordingStoppedAt ?? now)
        // ASR processing duration should exclude LLM enhancement time.
        // Measure from recording stop to first ASR text callback when available.
        let processingEnd = transcriptionResultReceivedAt ?? now
        let processingDuration = resolvedDuration(from: transcriptionProcessingStartedAt, to: processingEnd)
        let focusedAppName = outputDestinationContext?.appName
        let focusedAppBundleID = outputDestinationContext?.bundleID

        let remoteASRProviderInfo: String?
        let remoteASRModelInfo: String?
        let remoteASREndpointInfo: String?
        let senseVoiceMetadata: SenseVoiceTranscriptMetadata?
        if transcriptionEngine == .remote {
            let provider = remoteASRSelectedProvider
            let config = remoteASRConfigurations[provider.rawValue]
            remoteASRProviderInfo = provider.title
            remoteASRModelInfo = config?.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false ? config?.model : nil
            remoteASREndpointInfo = historyDisplayEndpoint(config?.endpoint)
            senseVoiceMetadata = nil
        } else {
            remoteASRProviderInfo = nil
            remoteASRModelInfo = nil
            remoteASREndpointInfo = nil
            senseVoiceMetadata = transcriptionEngine == .mlxAudio
                ? mlxTranscriber?.latestSenseVoiceMetadata
                : nil
        }

        if historyKind == .rewrite,
           let continuedEntryID = continueRewriteHistoryIfPossible(
                text: trimmed,
                createdAt: now,
                audioDurationSeconds: audioDuration,
                transcriptionProcessingDurationSeconds: processingDuration,
                llmDurationSeconds: llmDurationSeconds,
                pendingAudioArchiveURL: pendingAudioArchiveURL,
                whisperWordTimings: nil,
                senseVoiceMetadata: senseVoiceMetadata,
                dictionaryHitTerms: dictionaryHitTerms,
                dictionaryCorrectedTerms: dictionaryCorrectedTerms,
                dictionaryCorrectionSnapshots: dictionaryCorrectionSnapshots,
                dictionarySuggestedTerms: dictionarySuggestedTerms,
                rewriteConversationTurns: rewriteConversationTurns
           ) {
            if let outputDestinationContext {
                _ = historyStore.updateOutputDestination(
                    for: continuedEntryID,
                    focusedAppName: outputDestinationContext.appName,
                    focusedAppBundleID: outputDestinationContext.bundleID,
                    browserURLHost: outputDestinationContext.browserURLHost,
                    browserURLOrigin: outputDestinationContext.browserURLOrigin
                )
            }
            lastEnhancementPromptContext = nil
            transcriptionResultReceivedAt = nil
            return continuedEntryID
        }

        let audioRelativePath = importConsumedAudioArchiveIfNeeded(
            pendingAudioArchiveURL,
            kind: historyKind
        )
        let rewriteConversationMessages = TranscriptionHistoryConversationSupport
            .rewriteConversationMessages(from: rewriteConversationTurns, createdAt: now)

        let entryID = historyStore.append(
            text: trimmed,
            transcriptionEngine: transcriptionEngine.title,
            transcriptionModel: transcriptionModel,
            enhancementMode: textModelMetadata.modeTitle,
            enhancementModel: textModelMetadata.modelTitle,
            kind: historyKind,
            isTranslation: outputMode == .translation,
            audioDurationSeconds: audioDuration,
            transcriptionProcessingDurationSeconds: processingDuration,
            llmDurationSeconds: llmDurationSeconds,
            focusedAppName: focusedAppName,
            focusedAppBundleID: focusedAppBundleID,
            browserURLHost: outputDestinationContext?.browserURLHost,
            browserURLOrigin: outputDestinationContext?.browserURLOrigin,
            matchedGroupID: lastEnhancementPromptContext?.matchedGroupID,
            matchedGroupName: lastEnhancementPromptContext?.matchedGroupName,
            matchedAppGroupName: lastEnhancementPromptContext?.matchedAppGroupName,
            matchedURLGroupName: lastEnhancementPromptContext?.matchedURLGroupName,
            remoteASRProvider: remoteASRProviderInfo,
            remoteASRModel: remoteASRModelInfo,
            remoteASREndpoint: remoteASREndpointInfo,
            remoteLLMProvider: textModelMetadata.remoteProviderTitle,
            remoteLLMModel: textModelMetadata.remoteModelTitle,
            remoteLLMEndpoint: textModelMetadata.remoteEndpoint,
            audioRelativePath: audioRelativePath,
            whisperWordTimings: nil,
            senseVoiceMetadata: senseVoiceMetadata,
            displayTitle: trimmedDisplayTitle?.isEmpty == false ? trimmedDisplayTitle : nil,
            transcriptionChatMessages: historyKind == .rewrite
                ? (rewriteConversationMessages.isEmpty
                    ? TranscriptionHistoryConversationSupport.initialChatMessages(
                        forTranscript: trimmed,
                        createdAt: now
                    )
                    : rewriteConversationMessages)
                : nil,
            dictionaryHitTerms: dictionaryHitTerms,
            dictionaryCorrectedTerms: dictionaryCorrectedTerms,
            dictionaryCorrectionSnapshots: dictionaryCorrectionSnapshots,
            dictionarySuggestedTerms: dictionarySuggestedTerms
        )

        lastEnhancementPromptContext = nil
        transcriptionResultReceivedAt = nil

        return entryID
    }

    func captureCurrentOutputDestinationContext() -> OutputDestinationContext? {
        guard let application = NSWorkspace.shared.frontmostApplication else { return nil }
        let bundleID = application.bundleIdentifier
        guard bundleID != Bundle.main.bundleIdentifier else { return nil }

        let browserContext = outputDestinationBrowserContext(bundleID: bundleID)
        return OutputDestinationContext(
            appName: application.localizedName,
            bundleID: bundleID,
            pid: application.processIdentifier,
            browserURLHost: browserContext.host,
            browserURLOrigin: browserContext.origin
        )
    }

    private func outputDestinationBrowserContext(bundleID: String?) -> (host: String?, origin: String?) {
        guard isBrowserBundleID(bundleID),
              let activeURL = activeBrowserTabURL(frontmostBundleID: bundleID),
              let url = URL(string: activeURL),
              let host = url.host?.lowercased(),
              !host.isEmpty
        else {
            return (nil, nil)
        }

        return (
            host: host,
            origin: EnhancementOverlayIconResolver.faviconOrigin(fromPageURL: activeURL)
        )
    }

    private func continueRewriteHistoryIfPossible(
        text: String,
        createdAt: Date,
        audioDurationSeconds: TimeInterval?,
        transcriptionProcessingDurationSeconds: TimeInterval?,
        llmDurationSeconds: TimeInterval?,
        pendingAudioArchiveURL: URL?,
        whisperWordTimings: [WhisperHistoryWordTiming]?,
        senseVoiceMetadata: SenseVoiceTranscriptMetadata?,
        dictionaryHitTerms: [String],
        dictionaryCorrectedTerms: [String],
        dictionaryCorrectionSnapshots: [DictionaryCorrectionSnapshot],
        dictionarySuggestedTerms: [DictionarySuggestionSnapshot],
        rewriteConversationTurns: [RewriteConversationTurn]
    ) -> UUID? {
        guard overlayState.isRewriteConversationActive,
              let activeEntryID = overlayState.latestHistoryEntryID,
              let existingEntry = historyStore.entry(id: activeEntryID),
              existingEntry.kind == .rewrite
        else {
            return nil
        }

        let mergedSuggestedTerms = existingEntry.dictionarySuggestedTerms + dictionarySuggestedTerms.filter { incoming in
            !existingEntry.dictionarySuggestedTerms.contains(where: { $0.id == incoming.id })
        }

        let rewriteConversationMessages = TranscriptionHistoryConversationSupport
            .rewriteConversationMessages(
                from: rewriteConversationTurns,
                createdAt: createdAt
            )

        defer {
            if let pendingAudioArchiveURL,
               FileManager.default.fileExists(atPath: pendingAudioArchiveURL.path) {
                try? FileManager.default.removeItem(at: pendingAudioArchiveURL)
            }
        }

        let mergedEntry = historyStore.updateTranscriptionEntry(
            activeEntryID,
            text: text,
            createdAt: createdAt,
            audioDurationSeconds: TranscriptionHistoryConversationSupport.accumulatedDuration(
                existing: existingEntry.audioDurationSeconds,
                incoming: audioDurationSeconds
            ),
            transcriptionProcessingDurationSeconds: TranscriptionHistoryConversationSupport.accumulatedDuration(
                existing: existingEntry.transcriptionProcessingDurationSeconds,
                incoming: transcriptionProcessingDurationSeconds
            ),
            llmDurationSeconds: TranscriptionHistoryConversationSupport.accumulatedDuration(
                existing: existingEntry.llmDurationSeconds,
                incoming: llmDurationSeconds
            ),
            whisperWordTimings: whisperWordTimings ?? existingEntry.whisperWordTimings,
            senseVoiceMetadata: senseVoiceMetadata ?? existingEntry.senseVoiceMetadata,
            transcriptionChatMessages: rewriteConversationMessages.isEmpty
                ? TranscriptionHistoryConversationSupport.bootstrapChatMessages(for: existingEntry)
                : rewriteConversationMessages,
            dictionaryHitTerms: TranscriptionHistoryConversationSupport.mergedTerms(
                existing: existingEntry.dictionaryHitTerms,
                incoming: dictionaryHitTerms
            ),
            dictionaryCorrectedTerms: TranscriptionHistoryConversationSupport.mergedTerms(
                existing: existingEntry.dictionaryCorrectedTerms,
                incoming: dictionaryCorrectedTerms
            ),
            dictionaryCorrectionSnapshots: dictionaryCorrectionSnapshots,
            dictionarySuggestedTerms: mergedSuggestedTerms
        )

        guard mergedEntry != nil else { return nil }

        if let pendingAudioArchiveURL {
            let existingAudioURL = historyStore.audioURL(for: existingEntry)
            if let existingAudioURL, FileManager.default.fileExists(atPath: existingAudioURL.path) {
                if let mergedAudioURL = try? HistoryAudioArchiveSupport.mergedRewriteArchive(
                    existingArchiveURL: existingAudioURL,
                    appendedArchiveURL: pendingAudioArchiveURL
                ) {
                    do {
                        _ = try historyStore.replaceAudioArchive(for: activeEntryID, with: mergedAudioURL)
                    } catch {
                        try? FileManager.default.removeItem(at: mergedAudioURL)
                    }
                }
            } else {
                _ = try? historyStore.replaceAudioArchive(for: activeEntryID, with: pendingAudioArchiveURL)
            }
        }

        return activeEntryID
    }

    private func resolvedHistoryKind(for outputMode: SessionOutputMode) -> TranscriptionHistoryKind {
        HistoryValueResolver.resolvedKind(for: outputMode)
    }

    private func resolvedHistoryTextModelMetadata(for kind: TranscriptionHistoryKind) -> HistoryTextModelMetadata {
        switch kind {
        case .normal:
            guard transcriptionFeatureSettings.llmEnabled else {
                return HistoryTextModelMetadata(
                    modeTitle: EnhancementMode.off.title,
                    modelTitle: "None",
                    remoteProviderTitle: nil,
                    remoteModelTitle: nil,
                    remoteEndpoint: nil
                )
            }
            return resolvedHistoryTextModelMetadata(for: transcriptionFeatureSettings.llmSelectionID.textSelection)
        case .translation:
            return resolvedTranslationHistoryTextModelMetadata()
        case .rewrite:
            return resolvedHistoryTextModelMetadata(for: rewriteFeatureSettings.llmSelectionID.textSelection)
        case .transcript:
            return HistoryTextModelMetadata(
                modeTitle: EnhancementMode.off.title,
                modelTitle: "None",
                remoteProviderTitle: nil,
                remoteModelTitle: nil,
                remoteEndpoint: nil
            )
        }
    }

    private func resolvedTranslationHistoryTextModelMetadata() -> HistoryTextModelMetadata {
        switch translationFeatureSettings.modelSelectionID.translationSelection {
        case .localLLM(let repo):
            return HistoryTextModelMetadata(
                modeTitle: TranslationModelProvider.customLLM.title,
                modelTitle: "\(customLLMManager.displayTitle(for: repo)) (\(repo))",
                remoteProviderTitle: nil,
                remoteModelTitle: nil,
                remoteEndpoint: nil
            )
        case .localGGUF(let modelID):
            let option = ggufTranslationModelManager.option(for: modelID)
            return HistoryTextModelMetadata(
                modeTitle: TranslationModelProvider.localGGUF.title,
                modelTitle: option.title,
                remoteProviderTitle: nil,
                remoteModelTitle: nil,
                remoteEndpoint: nil
            )
        case .remoteLLM(let provider):
            return resolvedRemoteHistoryTextModelMetadata(
                provider: provider,
                modeTitle: TranslationModelProvider.remoteLLM.title
            )
        case .none:
            return HistoryTextModelMetadata(
                modeTitle: EnhancementMode.off.title,
                modelTitle: "None",
                remoteProviderTitle: nil,
                remoteModelTitle: nil,
                remoteEndpoint: nil
            )
        }
    }

    private func resolvedHistoryTextModelMetadata(
        for selection: FeatureModelSelectionID.TextSelection?
    ) -> HistoryTextModelMetadata {
        switch selection {
        case .appleIntelligence:
            return HistoryTextModelMetadata(
                modeTitle: EnhancementMode.appleIntelligence.title,
                modelTitle: "Apple Intelligence (Foundation Models)",
                remoteProviderTitle: nil,
                remoteModelTitle: nil,
                remoteEndpoint: nil
            )
        case .localLLM(let repo):
            return HistoryTextModelMetadata(
                modeTitle: EnhancementMode.customLLM.title,
                modelTitle: "\(customLLMManager.displayTitle(for: repo)) (\(repo))",
                remoteProviderTitle: nil,
                remoteModelTitle: nil,
                remoteEndpoint: nil
            )
        case .remoteLLM(let provider):
            return resolvedRemoteHistoryTextModelMetadata(
                provider: provider,
                modeTitle: EnhancementMode.remoteLLM.title
            )
        case .none:
            return HistoryTextModelMetadata(
                modeTitle: EnhancementMode.off.title,
                modelTitle: "None",
                remoteProviderTitle: nil,
                remoteModelTitle: nil,
                remoteEndpoint: nil
            )
        }
    }

    private func resolvedRemoteHistoryTextModelMetadata(
        provider: RemoteLLMProvider,
        modeTitle: String
    ) -> HistoryTextModelMetadata {
        guard RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
            provider: provider,
            stored: remoteLLMConfigurations
        ) else {
            return HistoryTextModelMetadata(
                modeTitle: modeTitle,
                modelTitle: provider.title,
                remoteProviderTitle: provider.title,
                remoteModelTitle: nil,
                remoteEndpoint: nil
            )
        }
        let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
            provider: provider,
            stored: remoteLLMConfigurations
        )
        let trimmedModel = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        return HistoryTextModelMetadata(
            modeTitle: modeTitle,
            modelTitle: trimmedModel.isEmpty ? provider.title : "\(provider.title) (\(trimmedModel))",
            remoteProviderTitle: provider.title,
            remoteModelTitle: trimmedModel.isEmpty ? nil : trimmedModel,
            remoteEndpoint: historyDisplayEndpoint(configuration.endpoint)
        )
    }

    private func resolvedDuration(from start: Date?, to end: Date?) -> TimeInterval? {
        HistoryValueResolver.resolvedDuration(from: start, to: end)
    }

    private func historyDisplayEndpoint(_ endpoint: String?) -> String? {
        HistoryValueResolver.historyDisplayEndpoint(endpoint)
    }

    private func trimmedStringValue(forKey key: String) -> String {
        stringValue(forKey: key).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func stringValue(forKey key: String) -> String {
        defaults.string(forKey: key) ?? ""
    }

    private func remoteConfigurations(forKey key: String) -> [String: RemoteProviderConfiguration] {
        RemoteModelConfigurationStore.loadConfigurations(
            from: stringValue(forKey: key),
            sensitiveValueLoading: .metadataOnly
        )
    }

    private func enumValue<T: RawRepresentable>(forKey key: String, default defaultValue: T) -> T where T.RawValue == String {
        T(rawValue: stringValue(forKey: key)) ?? defaultValue
    }

    private func enumValue<T: RawRepresentable>(forKey key: String, default defaultValue: T?) -> T? where T.RawValue == String {
        T(rawValue: stringValue(forKey: key)) ?? defaultValue
    }

    private func consumePendingCompletedHistoryAudioURL() -> URL? {
        if let pendingCompletedHistoryAudioArchiveURL {
            self.pendingCompletedHistoryAudioArchiveURL = nil
            let exists = FileManager.default.fileExists(atPath: pendingCompletedHistoryAudioArchiveURL.path)
            VoxtLog.history(
                "Consumed stashed history audio archive. file=\(pendingCompletedHistoryAudioArchiveURL.lastPathComponent), exists=\(exists)",
                verbose: true
            )
            return pendingCompletedHistoryAudioArchiveURL
        }
        let consumedURL: URL?
        switch transcriptionEngine {
        case .dictation:
            consumedURL = speechTranscriber.consumeCompletedAudioArchiveURL()
        case .mlxAudio:
            consumedURL = mlxTranscriber?.consumeCompletedAudioArchiveURL()
        case .remote:
            consumedURL = remoteASRTranscriber.consumeCompletedAudioArchiveURL()
        }
        if let consumedURL {
            let exists = FileManager.default.fileExists(atPath: consumedURL.path)
            VoxtLog.history(
                "Consumed transcriber history audio archive. engine=\(transcriptionEngine.rawValue), file=\(consumedURL.lastPathComponent), exists=\(exists)",
                verbose: true
            )
        } else {
            VoxtLog.historyWarning(
                "No transcriber history audio archive available. engine=\(transcriptionEngine.rawValue)"
            )
        }
        return consumedURL
    }

    func discardPendingCompletedHistoryAudio() {
        if let pendingCompletedHistoryAudioArchiveURL,
           FileManager.default.fileExists(atPath: pendingCompletedHistoryAudioArchiveURL.path) {
            try? FileManager.default.removeItem(at: pendingCompletedHistoryAudioArchiveURL)
        }
        pendingCompletedHistoryAudioArchiveURL = nil
        speechTranscriber.discardCompletedAudioArchive()
        mlxTranscriber?.discardCompletedAudioArchive()
        remoteASRTranscriber.discardCompletedAudioArchive()
    }

    func stashPendingCompletedHistoryAudioArchive(_ url: URL?) {
        guard let url else {
            VoxtLog.historyWarning("Pending history audio archive stash skipped because URL was nil.")
            return
        }
        if let pendingCompletedHistoryAudioArchiveURL,
           pendingCompletedHistoryAudioArchiveURL.path != url.path,
           FileManager.default.fileExists(atPath: pendingCompletedHistoryAudioArchiveURL.path) {
            try? FileManager.default.removeItem(at: pendingCompletedHistoryAudioArchiveURL)
        }
        pendingCompletedHistoryAudioArchiveURL = url
        VoxtLog.history("Pending history audio archive stashed. file=\(url.lastPathComponent)", verbose: true)
    }

    private func importConsumedAudioArchiveIfNeeded(
        _ sourceURL: URL?,
        kind: TranscriptionHistoryKind
    ) -> String? {
        guard let sourceURL else {
            VoxtLog.historyWarning("History audio import skipped because source URL was nil. kind=\(kind.rawValue)")
            return nil
        }
        defer {
            if FileManager.default.fileExists(atPath: sourceURL.path) {
                try? FileManager.default.removeItem(at: sourceURL)
            }
        }
        let exists = FileManager.default.fileExists(atPath: sourceURL.path)
        guard historyAudioStorageEnabled else {
            VoxtLog.history(
                "History audio import skipped because storage is disabled. kind=\(kind.rawValue), file=\(sourceURL.lastPathComponent), exists=\(exists)",
                verbose: true
            )
            return nil
        }
        guard exists else {
            VoxtLog.historyWarning(
                "History audio import skipped because source file does not exist. kind=\(kind.rawValue), file=\(sourceURL.lastPathComponent)"
            )
            return nil
        }
        do {
            let relativePath = try historyStore.importAudioArchive(from: sourceURL, kind: kind)
            VoxtLog.history(
                "History audio import succeeded. kind=\(kind.rawValue), file=\(sourceURL.lastPathComponent), relativePath=\(relativePath)"
            )
            return relativePath
        } catch {
            VoxtLog.historyWarning(
                "History audio import failed. kind=\(kind.rawValue), source=\(sourceURL.lastPathComponent), error=\(error.localizedDescription)"
            )
            return nil
        }
    }
}
