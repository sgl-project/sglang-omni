// FeatureSettingsStore.swift
// Provides Feature Settings Store for feature settings.

import Foundation

enum FeatureSettingsStore {
    static func migrateIfNeeded(defaults: UserDefaults = .standard) {
        pruneRetiredModelPreferences(defaults: defaults)
        removeObsoletePreferenceKeys(defaults: defaults)
        enforceAlwaysOnLegacyFlags(defaults: defaults)
        guard loadRaw(defaults: defaults) != nil else {
            save(deriveFromLegacy(defaults: defaults), defaults: defaults, notify: false)
            return
        }

        // Migration is the one explicit place where a normalized snapshot is
        // written. Keeping this out of load() prevents a SwiftUI @AppStorage
        // observer from turning a read into a write -> view update loop.
        save(load(defaults: defaults), defaults: defaults, notify: false)
    }

    static func load(defaults: UserDefaults = .standard) -> FeatureSettings {
        if let raw = loadRaw(defaults: defaults),
           let data = raw.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(FeatureSettings.self, from: data) {
            let fallback = deriveFromLegacy(defaults: defaults)
            return FeatureSettingsNormalization.sanitize(decoded, defaults: defaults, fallback: fallback)
        }
        let derived = deriveFromLegacy(defaults: defaults)
        return FeatureSettingsNormalization.sanitize(derived, defaults: defaults, fallback: derived)
    }

    static func save(
        _ settings: FeatureSettings,
        defaults: UserDefaults = .standard,
        notify: Bool = true
    ) {
        removeObsoletePreferenceKeys(defaults: defaults)
        enforceAlwaysOnLegacyFlags(defaults: defaults)
        let fallback = deriveFromLegacy(defaults: defaults)
        let sanitized = FeatureSettingsNormalization.sanitize(settings, defaults: defaults, fallback: fallback)
        _ = persist(sanitized, defaults: defaults, notify: notify)
    }

    private static func persist(
        _ settings: FeatureSettings,
        defaults: UserDefaults,
        notify: Bool
    ) -> Bool {
        var didChange = false
        let storageReady = storageRepresentation(for: settings)
        if let data = try? JSONEncoder().encode(storageReady),
           let raw = String(data: data, encoding: .utf8) {
            didChange = setIfChanged(raw, forKey: AppPreferenceKey.featureSettings, defaults: defaults) || didChange
        }
        // Keep feature-specific keys that runtime flows still read in sync.
        // VAD mode is global and is not read from or written to meeting settings.
        didChange = syncLegacyTranscription(storageReady.transcription, defaults: defaults) || didChange
        didChange = syncLegacyTranslation(storageReady.translation, defaults: defaults) || didChange
        didChange = syncLegacyRewrite(storageReady.rewrite, defaults: defaults) || didChange
        didChange = syncLegacyMeeting(storageReady.meeting, defaults: defaults) || didChange
        if notify, didChange {
            NotificationCenter.default.post(name: .voxtFeatureSettingsDidChange, object: nil)
        }
        return didChange
    }

    static func update(defaults: UserDefaults = .standard, _ mutate: (inout FeatureSettings) -> Void) {
        var settings = load(defaults: defaults)
        mutate(&settings)
        save(settings, defaults: defaults)
    }

    static func saveTranscriptionPrompt(
        _ prompt: String,
        presetID: String? = nil,
        defaults: UserDefaults = .standard
    ) {
        update(defaults: defaults) { settings in
            settings.transcription.prompt = AppPromptDefaults.canonicalStoredText(prompt, kind: .enhancement)
            settings.transcription.promptPresetID = presetID
        }
    }

    static func saveTranslationPrompt(
        _ prompt: String,
        presetID: String? = nil,
        defaults: UserDefaults = .standard
    ) {
        update(defaults: defaults) { settings in
            settings.translation.prompt = AppPromptDefaults.canonicalStoredText(prompt, kind: .translation)
            settings.translation.promptPresetID = presetID
        }
    }

    static func saveRewritePrompt(
        _ prompt: String,
        presetID: String? = nil,
        defaults: UserDefaults = .standard
    ) {
        update(defaults: defaults) { settings in
            settings.rewrite.prompt = AppPromptDefaults.canonicalStoredText(prompt, kind: .rewrite)
            settings.rewrite.promptPresetID = presetID
        }
    }

    private static func pruneRetiredModelPreferences(defaults: UserDefaults) {
        if let raw = defaults.string(forKey: AppPreferenceKey.mlxLocalASRTuningSettings) {
            let settings = MLXLocalTuningSettingsStore.load(from: raw)
            defaults.set(MLXLocalTuningSettingsStore.storageValue(for: settings), forKey: AppPreferenceKey.mlxLocalASRTuningSettings)
        }
        if let raw = defaults.string(forKey: AppPreferenceKey.customLLMGenerationSettingsByRepo) {
            let settings = CustomLLMGenerationSettingsStore.resolvedByRepo(from: raw)
            defaults.set(CustomLLMGenerationSettingsStore.storageValue(forByRepo: settings), forKey: AppPreferenceKey.customLLMGenerationSettingsByRepo)
        }
        let catalogs: [(String, Set<String>)] = [
            (AppPreferenceKey.mlxRemoteSizeCache, Set(MLXModelCatalog.availableModels.map(\.id))),
            (AppPreferenceKey.customLLMRemoteSizeCache, Set(CustomLLMModelCatalog.availableModels.map(\.id)))
        ]
        for (key, activeRepos) in catalogs {
            guard let data = defaults.data(forKey: key),
                  let sizes = try? JSONDecoder().decode([String: String].self, from: data),
                  let pruned = try? JSONEncoder().encode(sizes.filter { activeRepos.contains($0.key) })
            else { continue }
            defaults.set(pruned, forKey: key)
        }
    }

    private static func removeObsoletePreferenceKeys(defaults: UserDefaults) {
        defaults.removeObject(forKey: "enhancementLatencyProfile")
        defaults.removeObject(forKey: "translationLatencyProfile")
        defaults.removeObject(forKey: "rewriteLatencyProfile")
        // These legacy controls are no longer consumed by speaker analysis.
        defaults.removeObject(forKey: "meetingSpeakerCountHint")
        defaults.removeObject(forKey: "meetingSpeakerDiarizationSensitivity")
        defaults.removeObject(forKey: "meetingSpeakerDiarizationDebugEnabled")
    }

    private static func enforceAlwaysOnLegacyFlags(defaults: UserDefaults) {
        _ = setIfChanged(true, forKey: AppPreferenceKey.translateSelectedTextOnTranslationHotkey, defaults: defaults)
    }

    nonisolated static func availability(defaults: UserDefaults = .standard) -> FeatureAvailabilitySettings {
        // Prefer a side-effect-free peek so hotkey/runtime readers do not migrate/save
        // (and post `.voxtFeatureSettingsDidChange`) during early app bootstrap.
        if let raw = loadRaw(defaults: defaults),
           let data = raw.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(StoredAvailability.self, from: data) {
            return decoded.availability
        }
        let appEnhancementEnabled = defaults.object(forKey: AppPreferenceKey.appEnhancementEnabled) as? Bool ?? true
        return FeatureAvailabilitySettings(
            translationEnabled: true,
            rewriteEnabled: true,
            notesEnabled: true,
            appEnhancementEnabled: appEnhancementEnabled,
            meetingEnabled: true,
            filesEnabled: true
        )
    }

    static func deriveFromLegacy(defaults: UserDefaults = .standard) -> FeatureSettings {
        let transcriptionASR = legacyASRSelection(defaults: defaults)
        let transcriptionText = legacyTranscriptionTextSelection(defaults: defaults)
        let translationText = legacyTranslationSelection(defaults: defaults)
        let rewriteText = legacyRewriteSelection(defaults: defaults)
        let promptLanguage = AppPromptDefaults.interfaceLanguage(from: defaults)
        let transcriptionPrompt = AppPromptDefaults.resolvedStoredText(
            defaults.string(forKey: AppPreferenceKey.enhancementSystemPrompt),
            kind: .enhancement,
            defaults: defaults
        )
        let translationPrompt = AppPromptDefaults.resolvedStoredText(
            defaults.string(forKey: AppPreferenceKey.translationSystemPrompt),
            kind: .translation,
            defaults: defaults
        )
        let rewritePrompt = AppPromptDefaults.resolvedStoredText(
            defaults.string(forKey: AppPreferenceKey.rewriteSystemPrompt),
            kind: .rewrite,
            defaults: defaults
        )

        let appEnhancementEnabled = defaults.object(forKey: AppPreferenceKey.appEnhancementEnabled) as? Bool ?? true
        return FeatureSettings(
            transcription: TranscriptionFeatureSettings(
                asrSelectionID: transcriptionASR,
                llmEnabled: (EnhancementMode(rawValue: defaults.string(forKey: AppPreferenceKey.enhancementMode) ?? "") ?? .off) != .off,
                llmSelectionID: transcriptionText,
                prompt: transcriptionPrompt,
                promptPresetID: FeaturePromptPresetCatalog.inferredPresetID(
                    storedID: nil,
                    prompt: transcriptionPrompt,
                    kind: .enhancement,
                    language: promptLanguage
                ),
                notes: TranscriptionNoteFeatureSettings(
                    enabled: true,
                    triggerShortcut: .defaultShortcut,
                    titleModelSelectionID: transcriptionText,
                    obsidianSync: .init(),
                    remindersSync: .init()
                )
            ),
            translation: TranslationFeatureSettings(
                asrSelectionID: transcriptionASR,
                modelSelectionID: translationText,
                targetLanguageRawValue: (TranslationTargetLanguage(rawValue: defaults.string(forKey: AppPreferenceKey.translationTargetLanguage) ?? "") ?? .english).rawValue,
                prompt: translationPrompt,
                promptPresetID: FeaturePromptPresetCatalog.inferredPresetID(
                    storedID: nil,
                    prompt: translationPrompt,
                    kind: .translation,
                    language: promptLanguage
                ),
                showResultWindow: defaults.object(forKey: AppPreferenceKey.showSelectedTextTranslationResultWindow) as? Bool ?? true
            ),
            rewrite: RewriteFeatureSettings(
                asrSelectionID: transcriptionASR,
                llmSelectionID: rewriteText,
                prompt: rewritePrompt,
                promptPresetID: FeaturePromptPresetCatalog.inferredPresetID(
                    storedID: nil,
                    prompt: rewritePrompt,
                    kind: .rewrite,
                    language: promptLanguage
                ),
                appEnhancementEnabled: appEnhancementEnabled,
                continueShortcut: .defaultShortcut
            ),
            meeting: MeetingFeatureSettings(
                asrSelectionID: FeatureSettingsNormalization.supportedMeetingASRSelection(
                    transcriptionASR,
                    defaults: defaults
                ),
                summaryModelSelectionID: transcriptionText,
                summaryPrompt: "",
                summaryAutoGenerate: true,
                realtimeTranslateEnabled: false,
                realtimeTargetLanguageRawValue: defaults.string(forKey: AppPreferenceKey.meetingRealtimeTranslationTargetLanguage) ?? "",
                hideOverlayFromScreenSharing: defaults.object(forKey: AppPreferenceKey.hideMeetingOverlayFromScreenSharing) as? Bool ?? false,
                chunkingModeRawValue: MeetingChunkingMode.stored(in: defaults).rawValue,
                sileroVADSensitivityRawValue: MeetingSileroVADSensitivity.stored(defaults: defaults).rawValue,
                speakerDiarizationModelRawValue: MeetingDiarizationMode.stored(in: defaults).rawValue,
                finalTranscriptOptimizationEnabled: legacyFinalTranscriptOptimizationEnabled(defaults: defaults)
            ),
            availability: FeatureAvailabilitySettings(
                translationEnabled: true,
                rewriteEnabled: true,
                notesEnabled: true,
                appEnhancementEnabled: appEnhancementEnabled,
                meetingEnabled: true,
                filesEnabled: true
            )
        )
    }

    static func prepareLegacySession(
        from settings: FeatureSettings,
        outputMode: SessionOutputMode,
        defaults: UserDefaults = .standard
    ) {
        syncLegacyTranscription(settings.transcription, defaults: defaults)
        syncLegacyTranslation(settings.translation, defaults: defaults)
        syncLegacyRewrite(settings.rewrite, defaults: defaults)

        switch outputMode {
        case .transcription:
            syncLegacyASRSelection(settings.transcription.asrSelectionID, defaults: defaults)
        case .translation:
            syncLegacyASRSelection(settings.translation.asrSelectionID, defaults: defaults)
        case .rewrite:
            syncLegacyASRSelection(settings.rewrite.asrSelectionID, defaults: defaults)
        }
    }

    static func prepareMeetingRuntime(
        from settings: FeatureSettings,
        defaults: UserDefaults = .standard
    ) {
        let sanitizedMeeting = FeatureSettingsNormalization.sanitizedMeetingSettings(settings.meeting, defaults: defaults)
        syncLegacyTranslation(settings.translation, defaults: defaults)
        syncLegacyASRSelection(sanitizedMeeting.asrSelectionID, defaults: defaults)
        syncLegacyMeeting(sanitizedMeeting, defaults: defaults)
    }

    private nonisolated static func loadRaw(defaults: UserDefaults) -> String? {
        defaults.string(forKey: AppPreferenceKey.featureSettings)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private nonisolated struct StoredAvailability: Decodable, Sendable {
        let availability: FeatureAvailabilitySettings
    }

    @discardableResult
    private static func syncLegacyASRSelection(_ selectionID: FeatureModelSelectionID, defaults: UserDefaults) -> Bool {
        var didChange = false
        switch selectionID.asrSelection {
        case .dictation:
            didChange = setIfChanged(TranscriptionEngine.dictation.rawValue, forKey: AppPreferenceKey.transcriptionEngine, defaults: defaults) || didChange
        case .mlx(let repo):
            didChange = setIfChanged(TranscriptionEngine.mlxAudio.rawValue, forKey: AppPreferenceKey.transcriptionEngine, defaults: defaults) || didChange
            didChange = setIfChanged(MLXModelManager.canonicalModelRepo(repo), forKey: AppPreferenceKey.mlxModelRepo, defaults: defaults) || didChange
        case .remote(let provider):
            didChange = setIfChanged(TranscriptionEngine.remote.rawValue, forKey: AppPreferenceKey.transcriptionEngine, defaults: defaults) || didChange
            didChange = setIfChanged(provider.rawValue, forKey: AppPreferenceKey.remoteASRSelectedProvider, defaults: defaults) || didChange
        case .none:
            didChange = setIfChanged(TranscriptionEngine.mlxAudio.rawValue, forKey: AppPreferenceKey.transcriptionEngine, defaults: defaults) || didChange
        }
        return didChange
    }

    @discardableResult
    private static func syncLegacyTranscription(_ settings: TranscriptionFeatureSettings, defaults: UserDefaults) -> Bool {
        var didChange = setIfChanged(
            AppPromptDefaults.canonicalStoredText(settings.prompt, kind: .enhancement),
            forKey: AppPreferenceKey.enhancementSystemPrompt,
            defaults: defaults
        )
        guard settings.llmEnabled else {
            didChange = setIfChanged(EnhancementMode.off.rawValue, forKey: AppPreferenceKey.enhancementMode, defaults: defaults) || didChange
            return didChange
        }

        switch settings.llmSelectionID.textSelection {
        case .appleIntelligence:
            didChange = setIfChanged(EnhancementMode.appleIntelligence.rawValue, forKey: AppPreferenceKey.enhancementMode, defaults: defaults) || didChange
        case .localLLM(let repo):
            didChange = setIfChanged(EnhancementMode.customLLM.rawValue, forKey: AppPreferenceKey.enhancementMode, defaults: defaults) || didChange
            didChange = setIfChanged(repo, forKey: AppPreferenceKey.customLLMModelRepo, defaults: defaults) || didChange
        case .remoteLLM(let provider):
            didChange = setIfChanged(EnhancementMode.remoteLLM.rawValue, forKey: AppPreferenceKey.enhancementMode, defaults: defaults) || didChange
            didChange = setIfChanged(provider.rawValue, forKey: AppPreferenceKey.remoteLLMSelectedProvider, defaults: defaults) || didChange
        case .none:
            didChange = setIfChanged(EnhancementMode.off.rawValue, forKey: AppPreferenceKey.enhancementMode, defaults: defaults) || didChange
        }
        return didChange
    }

    @discardableResult
    private static func syncLegacyTranslation(_ settings: TranslationFeatureSettings, defaults: UserDefaults) -> Bool {
        var didChange = setIfChanged(
            AppPromptDefaults.canonicalStoredText(settings.prompt, kind: .translation),
            forKey: AppPreferenceKey.translationSystemPrompt,
            defaults: defaults
        )
        didChange = setIfChanged(settings.targetLanguage.rawValue, forKey: AppPreferenceKey.translationTargetLanguage, defaults: defaults) || didChange
        didChange = setIfChanged(true, forKey: AppPreferenceKey.translateSelectedTextOnTranslationHotkey, defaults: defaults) || didChange
        didChange = setIfChanged(settings.showResultWindow, forKey: AppPreferenceKey.showSelectedTextTranslationResultWindow, defaults: defaults) || didChange

        switch settings.modelSelectionID.translationSelection {
        case .localLLM(let repo):
            didChange = setIfChanged(TranslationModelProvider.customLLM.rawValue, forKey: AppPreferenceKey.translationModelProvider, defaults: defaults) || didChange
            didChange = setIfChanged(TranslationModelProvider.customLLM.rawValue, forKey: AppPreferenceKey.translationFallbackModelProvider, defaults: defaults) || didChange
            didChange = setIfChanged(repo, forKey: AppPreferenceKey.translationCustomLLMModelRepo, defaults: defaults) || didChange
        case .localGGUF(let modelID):
            didChange = setIfChanged(TranslationModelProvider.localGGUF.rawValue, forKey: AppPreferenceKey.translationModelProvider, defaults: defaults) || didChange
            didChange = setIfChanged(TranslationModelProvider.localGGUF.rawValue, forKey: AppPreferenceKey.translationFallbackModelProvider, defaults: defaults) || didChange
            didChange = setIfChanged(modelID.rawValue, forKey: AppPreferenceKey.translationGGUFModelID, defaults: defaults) || didChange
        case .remoteLLM(let provider):
            didChange = setIfChanged(TranslationModelProvider.remoteLLM.rawValue, forKey: AppPreferenceKey.translationModelProvider, defaults: defaults) || didChange
            didChange = setIfChanged(TranslationModelProvider.remoteLLM.rawValue, forKey: AppPreferenceKey.translationFallbackModelProvider, defaults: defaults) || didChange
            didChange = setIfChanged(provider.rawValue, forKey: AppPreferenceKey.translationRemoteLLMProvider, defaults: defaults) || didChange
        case .none:
            didChange = setIfChanged(TranslationModelProvider.customLLM.rawValue, forKey: AppPreferenceKey.translationModelProvider, defaults: defaults) || didChange
        }
        return didChange
    }

    @discardableResult
    private static func syncLegacyRewrite(_ settings: RewriteFeatureSettings, defaults: UserDefaults) -> Bool {
        var didChange = setIfChanged(
            AppPromptDefaults.canonicalStoredText(settings.prompt, kind: .rewrite),
            forKey: AppPreferenceKey.rewriteSystemPrompt,
            defaults: defaults
        )
        didChange = setIfChanged(settings.appEnhancementEnabled, forKey: AppPreferenceKey.appEnhancementEnabled, defaults: defaults) || didChange

        switch settings.llmSelectionID.textSelection {
        case .appleIntelligence:
            didChange = setIfChanged(RewriteModelProvider.customLLM.rawValue, forKey: AppPreferenceKey.rewriteModelProvider, defaults: defaults) || didChange
        case .localLLM(let repo):
            didChange = setIfChanged(RewriteModelProvider.customLLM.rawValue, forKey: AppPreferenceKey.rewriteModelProvider, defaults: defaults) || didChange
            didChange = setIfChanged(repo, forKey: AppPreferenceKey.rewriteCustomLLMModelRepo, defaults: defaults) || didChange
        case .remoteLLM(let provider):
            didChange = setIfChanged(RewriteModelProvider.remoteLLM.rawValue, forKey: AppPreferenceKey.rewriteModelProvider, defaults: defaults) || didChange
            didChange = setIfChanged(provider.rawValue, forKey: AppPreferenceKey.rewriteRemoteLLMProvider, defaults: defaults) || didChange
        case .none:
            didChange = setIfChanged(RewriteModelProvider.customLLM.rawValue, forKey: AppPreferenceKey.rewriteModelProvider, defaults: defaults) || didChange
        }
        return didChange
    }

    @discardableResult
    private static func syncLegacyMeeting(_ settings: MeetingFeatureSettings, defaults: UserDefaults) -> Bool {
        var didChange = setIfChanged(settings.chunkingMode.rawValue, forKey: AppPreferenceKey.meetingChunkingMode, defaults: defaults)
        didChange = setIfChanged(settings.sileroVADSensitivity.rawValue, forKey: AppPreferenceKey.meetingSileroVADSensitivity, defaults: defaults) || didChange
        didChange = setIfChanged(settings.speakerDiarizationModel.rawValue, forKey: AppPreferenceKey.meetingSpeakerDiarizationModel, defaults: defaults) || didChange
        didChange = setIfChanged(
            settings.finalTranscriptOptimizationEnabled,
            forKey: AppPreferenceKey.meetingFinalTranscriptOptimizationEnabled,
            defaults: defaults
        ) || didChange
        return didChange
    }

    @discardableResult
    private static func setIfChanged(_ value: Any, forKey key: String, defaults: UserDefaults) -> Bool {
        guard let existing = defaults.object(forKey: key) else {
            defaults.set(value, forKey: key)
            return true
        }
        if let existingObject = existing as? NSObject,
           let newObject = value as? NSObject,
           existingObject.isEqual(newObject) {
            return false
        }
        defaults.set(value, forKey: key)
        return true
    }

    private static func storageRepresentation(for settings: FeatureSettings) -> FeatureSettings {
        var stored = settings
        stored.transcription.notes.enabled = settings.availability.notesEnabled
        stored.rewrite.appEnhancementEnabled = settings.availability.appEnhancementEnabled
        stored.transcription.prompt = AppPromptDefaults.canonicalStoredText(settings.transcription.prompt, kind: .enhancement)
        stored.translation.prompt = AppPromptDefaults.canonicalStoredText(settings.translation.prompt, kind: .translation)
        stored.rewrite.prompt = AppPromptDefaults.canonicalStoredText(settings.rewrite.prompt, kind: .rewrite)
        stored.meeting.summaryPrompt = AppPromptDefaults.canonicalStoredText(settings.meeting.summaryPrompt, kind: .transcriptSummary)
        stored.meeting.chunkingModeRawValue = settings.meeting.chunkingMode.rawValue
        stored.meeting.sileroVADSensitivityRawValue = settings.meeting.sileroVADSensitivity.rawValue
        stored.meeting.speakerDiarizationModelRawValue = settings.meeting.speakerDiarizationModel.rawValue
        return stored
    }

    private static func legacyASRSelection(defaults: UserDefaults) -> FeatureModelSelectionID {
        let engineRaw = defaults.string(forKey: AppPreferenceKey.transcriptionEngine)
        let engine = TranscriptionEngine.resolved(rawValue: engineRaw)
        switch engine {
        case .dictation:
            return .dictation
        case .mlxAudio:
            if engineRaw == "whisperKit" {
                return .mlx(
                    MLXWhisperMigrationSupport.repo(
                        forLegacyWhisperModelID: defaults.string(forKey: AppPreferenceKey.legacyWhisperModelID)
                            ?? MLXWhisperMigrationSupport.defaultLegacyModelID
                    )
                )
            }
            let storedRepo = defaults.string(forKey: AppPreferenceKey.mlxModelRepo) ?? MLXModelManager.defaultModelRepo
            return .mlx(storedRepo)
        case .remote:
            let provider = RemoteASRProvider(rawValue: defaults.string(forKey: AppPreferenceKey.remoteASRSelectedProvider) ?? "") ?? .openAIWhisper
            return .remoteASR(provider)
        }
    }

    private static func legacyTranscriptionTextSelection(defaults: UserDefaults) -> FeatureModelSelectionID {
        let mode = EnhancementMode(rawValue: defaults.string(forKey: AppPreferenceKey.enhancementMode) ?? "") ?? .off
        switch mode {
        case .appleIntelligence:
            return .appleIntelligence
        case .customLLM, .off:
            return .localLLM(defaults.string(forKey: AppPreferenceKey.customLLMModelRepo) ?? CustomLLMModelManager.defaultModelRepo)
        case .remoteLLM:
            let provider = RemoteLLMProvider(rawValue: defaults.string(forKey: AppPreferenceKey.remoteLLMSelectedProvider) ?? "") ?? .openAI
            return .remoteLLM(provider)
        }
    }

    private static func legacyTranslationSelection(defaults: UserDefaults) -> FeatureModelSelectionID {
        let provider = TranslationModelProvider.resolved(
            rawValue: defaults.string(forKey: AppPreferenceKey.translationModelProvider)
        )
        switch provider {
        case .customLLM:
            return .localLLM(defaults.string(forKey: AppPreferenceKey.translationCustomLLMModelRepo) ?? CustomLLMModelManager.defaultModelRepo)
        case .localGGUF:
            return .localGGUFTranslation(
                GGUFTranslationModelCatalog.resolvedModelID(
                    defaults.string(forKey: AppPreferenceKey.translationGGUFModelID)
                )
            )
        case .remoteLLM:
            let fallback = RemoteLLMProvider(rawValue: defaults.string(forKey: AppPreferenceKey.remoteLLMSelectedProvider) ?? "") ?? .openAI
            let selected = RemoteLLMProvider(rawValue: defaults.string(forKey: AppPreferenceKey.translationRemoteLLMProvider) ?? "") ?? fallback
            return .remoteLLM(selected)
        }
    }

    private static func legacyRewriteSelection(defaults: UserDefaults) -> FeatureModelSelectionID {
        let provider = RewriteModelProvider(rawValue: defaults.string(forKey: AppPreferenceKey.rewriteModelProvider) ?? "") ?? .customLLM
        switch provider {
        case .customLLM:
            return .localLLM(defaults.string(forKey: AppPreferenceKey.rewriteCustomLLMModelRepo) ?? CustomLLMModelManager.defaultModelRepo)
        case .remoteLLM:
            let fallback = RemoteLLMProvider(rawValue: defaults.string(forKey: AppPreferenceKey.remoteLLMSelectedProvider) ?? "") ?? .openAI
            let selected = RemoteLLMProvider(rawValue: defaults.string(forKey: AppPreferenceKey.rewriteRemoteLLMProvider) ?? "") ?? fallback
            return .remoteLLM(selected)
        }
    }

    private static func legacyFinalTranscriptOptimizationEnabled(defaults: UserDefaults) -> Bool {
        defaults.object(forKey: AppPreferenceKey.meetingFinalTranscriptOptimizationEnabled) as? Bool
            ?? MeetingFeatureSettings.defaultFinalTranscriptOptimizationEnabled
    }

}
