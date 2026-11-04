import Foundation

enum FeatureSettingsNormalization {
    static func sanitize(
        _ settings: FeatureSettings,
        defaults: UserDefaults,
        fallback: FeatureSettings
    ) -> FeatureSettings {
        let promptLanguage = AppPromptDefaults.interfaceLanguage(from: defaults)
        let resolvedTranscriptionPrompt = AppPromptDefaults.resolvedStoredText(
            sanitizedPrompt(settings.transcription.prompt),
            kind: .enhancement,
            defaults: defaults
        )
        let transcriptionPrompt = FeaturePromptPresetCatalog.resolvedPrompt(
            storedID: settings.transcription.promptPresetID,
            prompt: resolvedTranscriptionPrompt,
            kind: .enhancement,
            language: promptLanguage
        )
        let resolvedTranslationPrompt = AppPromptDefaults.resolvedStoredText(
            sanitizedPrompt(settings.translation.prompt),
            kind: .translation,
            defaults: defaults
        )
        let translationPrompt = FeaturePromptPresetCatalog.resolvedPrompt(
            storedID: settings.translation.promptPresetID,
            prompt: resolvedTranslationPrompt,
            kind: .translation,
            language: promptLanguage
        )
        let resolvedRewritePrompt = AppPromptDefaults.resolvedStoredText(
            sanitizedPrompt(settings.rewrite.prompt),
            kind: .rewrite,
            defaults: defaults
        )
        let rewritePrompt = FeaturePromptPresetCatalog.resolvedPrompt(
            storedID: settings.rewrite.promptPresetID,
            prompt: resolvedRewritePrompt,
            kind: .rewrite,
            language: promptLanguage
        )
        let transcriptionASR = sanitizedASRSelection(
            settings.transcription.asrSelectionID,
            fallback: fallback.transcription.asrSelectionID
        )
        let translationASR = sanitizedASRSelection(
            settings.translation.asrSelectionID,
            fallback: fallback.translation.asrSelectionID
        )
        let rewriteASR = sanitizedASRSelection(
            settings.rewrite.asrSelectionID,
            fallback: fallback.rewrite.asrSelectionID
        )
        let availability = settings.availability
        var notes = sanitizedNotesSettings(
            settings.transcription.notes,
            fallbackSelectionID: fallback.transcription.notes.titleModelSelectionID
        )
        notes.enabled = availability.notesEnabled
        return FeatureSettings(
            transcription: TranscriptionFeatureSettings(
                asrSelectionID: transcriptionASR,
                llmEnabled: settings.transcription.llmEnabled,
                llmSelectionID: sanitizedTextSelection(settings.transcription.llmSelectionID, fallback: fallback.transcription.llmSelectionID),
                prompt: transcriptionPrompt,
                promptPresetID: FeaturePromptPresetCatalog.inferredPresetID(
                    storedID: settings.transcription.promptPresetID,
                    prompt: transcriptionPrompt,
                    kind: .enhancement,
                    language: promptLanguage
                ),
                notes: notes
            ),
            translation: TranslationFeatureSettings(
                asrSelectionID: translationASR,
                modelSelectionID: sanitizedTranslationSelection(
                    settings.translation.modelSelectionID,
                    fallback: fallback.translation.modelSelectionID
                ),
                targetLanguageRawValue: settings.translation.targetLanguage.rawValue,
                prompt: translationPrompt,
                promptPresetID: FeaturePromptPresetCatalog.inferredPresetID(
                    storedID: settings.translation.promptPresetID,
                    prompt: translationPrompt,
                    kind: .translation,
                    language: promptLanguage
                ),
                showResultWindow: settings.translation.showResultWindow
            ),
            rewrite: RewriteFeatureSettings(
                asrSelectionID: rewriteASR,
                llmSelectionID: sanitizedTextSelection(settings.rewrite.llmSelectionID, fallback: fallback.rewrite.llmSelectionID),
                prompt: rewritePrompt,
                promptPresetID: FeaturePromptPresetCatalog.inferredPresetID(
                    storedID: settings.rewrite.promptPresetID,
                    prompt: rewritePrompt,
                    kind: .rewrite,
                    language: promptLanguage
                ),
                appEnhancementEnabled: availability.appEnhancementEnabled,
                continueShortcut: sanitizedContinueShortcutSettings(settings.rewrite.continueShortcut)
            ),
            meeting: MeetingFeatureSettings(
                asrSelectionID: supportedMeetingASRSelection(
                    sanitizedASRSelection(
                        settings.meeting.asrSelectionID,
                        fallback: fallback.meeting.asrSelectionID
                    ),
                    defaults: defaults
                ),
                summaryModelSelectionID: sanitizedTextSelection(settings.meeting.summaryModelSelectionID, fallback: fallback.meeting.summaryModelSelectionID),
                summaryPrompt: AppPromptDefaults.resolvedStoredText(
                    sanitizedPrompt(settings.meeting.summaryPrompt),
                    kind: .transcriptSummary,
                    defaults: defaults
                ),
                summaryAutoGenerate: settings.meeting.summaryAutoGenerate,
                realtimeTranslateEnabled: settings.meeting.realtimeTranslateEnabled,
                realtimeTargetLanguageRawValue: settings.meeting.realtimeTargetLanguage?.rawValue ?? "",
                hideOverlayFromScreenSharing: settings.meeting.hideOverlayFromScreenSharing,
                chunkingModeRawValue: settings.meeting.chunkingMode.rawValue,
                sileroVADSensitivityRawValue: settings.meeting.sileroVADSensitivity.rawValue,
                speakerDiarizationModelRawValue: settings.meeting.speakerDiarizationModel.rawValue,
                finalTranscriptOptimizationEnabled: settings.meeting.finalTranscriptOptimizationEnabled
            ),
            availability: availability
        )
    }

    private static func sanitizedASRSelection(
        _ selectionID: FeatureModelSelectionID,
        fallback: FeatureModelSelectionID
    ) -> FeatureModelSelectionID {
        switch selectionID.asrSelection {
        case .dictation:
            return .dictation
        case let .mlx(repo):
            return .mlx(MLXModelCatalog.isAvailableModelRepo(repo) ? repo : MLXModelCatalog.defaultModelRepo)
        case let .remote(provider):
            return .remoteASR(provider)
        case .none:
            return fallback
        }
    }

    private static func sanitizedTextSelection(
        _ selectionID: FeatureModelSelectionID,
        fallback: FeatureModelSelectionID
    ) -> FeatureModelSelectionID {
        switch selectionID.textSelection {
        case .localLLM(let repo):
            return .localLLM(CustomLLMModelCatalog.isSupportedModelRepo(repo) ? repo : CustomLLMModelCatalog.defaultModelRepo)
        case .appleIntelligence:
            return .appleIntelligence
        case .remoteLLM(let provider):
            return .remoteLLM(provider)
        case .none:
            return fallback
        }
    }

    private static func sanitizedTranslationSelection(
        _ selectionID: FeatureModelSelectionID,
        fallback: FeatureModelSelectionID
    ) -> FeatureModelSelectionID {
        switch selectionID.translationSelection {
        case let .localLLM(repo):
            return .localLLM(CustomLLMModelCatalog.isSupportedModelRepo(repo) ? repo : CustomLLMModelCatalog.defaultModelRepo)
        case let .localGGUF(modelID):
            return .localGGUFTranslation(modelID)
        case let .remoteLLM(provider):
            return .remoteLLM(provider)
        case .none:
            return fallback
        }
    }

    private static func sanitizedPrompt(_ prompt: String) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "" : prompt
    }

    private static func sanitizedNotesSettings(
        _ settings: TranscriptionNoteFeatureSettings,
        fallbackSelectionID: FeatureModelSelectionID
    ) -> TranscriptionNoteFeatureSettings {
        let resolvedSelectionID = sanitizedTextSelection(settings.titleModelSelectionID, fallback: fallbackSelectionID)
        let resolvedShortcut = settings.triggerShortcut.keyCode == HotkeyPreference.modifierOnlyKeyCode
            ? TranscriptionNoteTriggerSettings.defaultShortcut
            : TranscriptionNoteTriggerSettings(
                keyCode: settings.triggerShortcut.keyCode,
                modifiers: settings.triggerShortcut.modifiers,
                sidedModifiers: settings.triggerShortcut.sidedModifiers
            )
        return TranscriptionNoteFeatureSettings(
            enabled: settings.enabled,
            triggerShortcut: resolvedShortcut,
            titleModelSelectionID: resolvedSelectionID,
            panel: VoxtNotePanelSettings(
                corner: settings.panel.corner,
                revealDelay: min(max(settings.panel.revealDelay, 0.2), 2.0),
                hideDelay: min(max(settings.panel.hideDelay, 0.1), 2.0),
                isTranslucent: settings.panel.isTranslucent
            ),
            obsidianSync: sanitizedObsidianSyncSettings(settings.obsidianSync),
            remindersSync: sanitizedRemindersSyncSettings(settings.remindersSync)
        )
    }

    private static func sanitizedObsidianSyncSettings(
        _ settings: ObsidianNoteSyncSettings
    ) -> ObsidianNoteSyncSettings {
        let trimmedPath = settings.vaultPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedFolder = settings.relativeFolder
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))

        return ObsidianNoteSyncSettings(
            enabled: settings.enabled,
            vaultPath: trimmedPath,
            vaultBookmarkData: settings.vaultBookmarkData,
            relativeFolder: trimmedFolder.isEmpty ? "Voxt" : trimmedFolder,
            groupingMode: settings.groupingMode
        )
    }

    private static func sanitizedRemindersSyncSettings(
        _ settings: RemindersNoteSyncSettings
    ) -> RemindersNoteSyncSettings {
        let trimmedIdentifier = settings.selectedListIdentifier
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedTitle = settings.selectedListTitle
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return RemindersNoteSyncSettings(
            enabled: settings.enabled,
            selectedListIdentifier: trimmedIdentifier,
            selectedListTitle: trimmedIdentifier.isEmpty ? "" : trimmedTitle
        )
    }

    private static func sanitizedContinueShortcutSettings(
        _ settings: TranscriptionContinueShortcutSettings
    ) -> TranscriptionContinueShortcutSettings {
        guard settings.keyCode != HotkeyPreference.modifierOnlyKeyCode else {
            return .defaultShortcut
        }
        return TranscriptionContinueShortcutSettings(
            keyCode: settings.keyCode,
            modifiers: settings.modifiers,
            sidedModifiers: settings.sidedModifiers
        )
    }

    static func sanitizedMeetingSettings(
        _ settings: MeetingFeatureSettings,
        defaults: UserDefaults
    ) -> MeetingFeatureSettings {
        MeetingFeatureSettings(
            asrSelectionID: supportedMeetingASRSelection(settings.asrSelectionID, defaults: defaults),
            summaryModelSelectionID: settings.summaryModelSelectionID,
            summaryPrompt: settings.summaryPrompt,
            summaryAutoGenerate: settings.summaryAutoGenerate,
            realtimeTranslateEnabled: settings.realtimeTranslateEnabled,
            realtimeTargetLanguageRawValue: settings.realtimeTargetLanguageRawValue,
            hideOverlayFromScreenSharing: settings.hideOverlayFromScreenSharing,
            chunkingModeRawValue: settings.chunkingMode.rawValue,
            sileroVADSensitivityRawValue: settings.sileroVADSensitivity.rawValue,
            speakerDiarizationModelRawValue: settings.speakerDiarizationModel.rawValue,
            finalTranscriptOptimizationEnabled: settings.finalTranscriptOptimizationEnabled
        )
    }

    static func supportedMeetingASRSelection(
        _ selectionID: FeatureModelSelectionID,
        defaults: UserDefaults
    ) -> FeatureModelSelectionID {
        let fallbackRepo = MLXModelManager.canonicalModelRepo(
            defaults.string(forKey: AppPreferenceKey.mlxModelRepo) ?? MLXModelManager.defaultModelRepo
        )
        switch selectionID.asrSelection {
        case .none:
            return .mlx(fallbackRepo)
        case .dictation, .mlx, .remote:
            return selectionID
        }
    }
}
