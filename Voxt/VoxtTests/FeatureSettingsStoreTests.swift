// FeatureSettingsStoreTests.swift
// Provides Feature Settings Store Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class FeatureSettingsStoreTests: XCTestCase {
    private func withEphemeralDefaults(
        _ body: (UserDefaults) throws -> Void
    ) rethrows {
        let suiteName = "FeatureSettingsStoreTests.\(UUID().uuidString)"
        guard let defaults = UserDefaults(suiteName: suiteName) else {
            XCTFail("Expected ephemeral UserDefaults suite")
            return
        }
        defaults.removePersistentDomain(forName: suiteName)
        defer {
            defaults.removePersistentDomain(forName: suiteName)
        }
        try body(defaults)
    }

    func testMigrateIfNeededRemovesObsoleteLatencyProfileKeys() throws {
        try withEphemeralDefaults { defaults in
            defaults.set("instant", forKey: "enhancementLatencyProfile")
            defaults.set("quality", forKey: "translationLatencyProfile")
            defaults.set("balanced", forKey: "rewriteLatencyProfile")

            FeatureSettingsStore.migrateIfNeeded(defaults: defaults)

            XCTAssertNil(defaults.object(forKey: "enhancementLatencyProfile"))
            XCTAssertNil(defaults.object(forKey: "translationLatencyProfile"))
            XCTAssertNil(defaults.object(forKey: "rewriteLatencyProfile"))
            XCTAssertNotNil(defaults.string(forKey: AppPreferenceKey.featureSettings))
        }
    }

    func testMigrateIfNeededRemovesObsoleteLatencyProfileKeysAndDerivesSettings() throws {
        try withEphemeralDefaults { defaults in
            defaults.set("quality", forKey: "enhancementLatencyProfile")
            defaults.set(EnhancementMode.customLLM.rawValue, forKey: AppPreferenceKey.enhancementMode)
            defaults.set("mlx-community/Qwen3.5-2B-4bit", forKey: AppPreferenceKey.customLLMModelRepo)

            FeatureSettingsStore.migrateIfNeeded(defaults: defaults)
            let settings = FeatureSettingsStore.load(defaults: defaults)

            XCTAssertNil(defaults.object(forKey: "enhancementLatencyProfile"))
            XCTAssertTrue(settings.transcription.llmEnabled)
            XCTAssertEqual(
                settings.transcription.llmSelectionID,
                .localLLM("mlx-community/Qwen3.5-2B-4bit")
            )
        }
    }

    func testLoadNormalizationDoesNotBroadcastFeatureSettingsChange() throws {
        try withEphemeralDefaults { defaults in
            let notificationExpectation = expectation(
                forNotification: .voxtFeatureSettingsDidChange,
                object: nil
            )
            notificationExpectation.isInverted = true

            _ = FeatureSettingsStore.load(defaults: defaults)

            wait(for: [notificationExpectation], timeout: 0.1)
        }
    }

    func testLoadDoesNotMutateStoredFeatureSettingsPayload() throws {
        try withEphemeralDefaults { defaults in
            let original = try XCTUnwrap(
                String(
                    data: JSONEncoder().encode(FeatureSettingsStore.deriveFromLegacy(defaults: defaults)),
                    encoding: .utf8
                )
            )
            defaults.set(original, forKey: AppPreferenceKey.featureSettings)

            _ = FeatureSettingsStore.load(defaults: defaults)

            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.featureSettings), original)
        }
    }

    func testLoadNormalizationIsIdempotent() throws {
        try withEphemeralDefaults { defaults in
            let first = FeatureSettingsStore.load(defaults: defaults)
            let second = FeatureSettingsStore.load(defaults: defaults)

            if first.transcription != second.transcription {
                XCTFail("transcription changed between loads")
            }
            if first.translation != second.translation {
                XCTFail("translation changed between loads")
            }
            if first.rewrite != second.rewrite {
                XCTFail("rewrite changed between loads")
            }
            if first.meeting != second.meeting {
                XCTFail("meeting changed between loads")
            }
            if first.availability != second.availability {
                XCTFail("availability changed between loads")
            }
        }
    }

    func testRemovedModelSelectionsMigrateToSupportedDefaults() {
        let legacySherpaSelection = FeatureModelSelectionID(rawValue: "sherpa:funasr-nano-int8")
        XCTAssertEqual(
            legacySherpaSelection.asrSelection,
            .mlx(repo: MLXModelManager.defaultModelRepo)
        )

        let removedGGUFSelection = FeatureModelSelectionID(
            rawValue: "local-gguf-translation:tencent/Hy-MT2-1.8B-GGUF#Hy-MT2-1.8B-Q6_K.gguf"
        )
        XCTAssertEqual(
            removedGGUFSelection.translationSelection,
            .localGGUF(modelID: .hyMT2Q4KM)
        )
    }

    func testLoadPersistsCanonicalIDsForRemovedModelSelections() throws {
        try withEphemeralDefaults { defaults in
            var settings = FeatureSettingsStore.deriveFromLegacy(defaults: defaults)
            settings.transcription.asrSelectionID = FeatureModelSelectionID(rawValue: "sherpa:funasr-nano-int8")
            settings.translation.modelSelectionID = FeatureModelSelectionID(
                rawValue: "local-gguf-translation:tencent/Hy-MT2-1.8B-GGUF#Hy-MT2-1.8B-Q6_K.gguf"
            )
            let data = try JSONEncoder().encode(settings)
            defaults.set(try XCTUnwrap(String(data: data, encoding: .utf8)), forKey: AppPreferenceKey.featureSettings)

            FeatureSettingsStore.migrateIfNeeded(defaults: defaults)
            let loaded = FeatureSettingsStore.load(defaults: defaults)
            let storedRaw = try XCTUnwrap(defaults.string(forKey: AppPreferenceKey.featureSettings))
            let stored = try JSONDecoder().decode(FeatureSettings.self, from: XCTUnwrap(storedRaw.data(using: .utf8)))

            XCTAssertEqual(loaded.transcription.asrSelectionID, .mlx(MLXModelManager.defaultModelRepo))
            XCTAssertEqual(loaded.translation.modelSelectionID, .localGGUFTranslation(.hyMT2Q4KM))
            XCTAssertEqual(stored.transcription.asrSelectionID, .mlx(MLXModelManager.defaultModelRepo))
            XCTAssertEqual(stored.translation.modelSelectionID, .localGGUFTranslation(.hyMT2Q4KM))
        }
    }

    func testSaveRemovesObsoleteLatencyProfileKeysWithoutAffectingStoredSettings() throws {
        try withEphemeralDefaults { defaults in
            defaults.set("instant", forKey: "enhancementLatencyProfile")
            defaults.set("balanced", forKey: "translationLatencyProfile")
            defaults.set("quality", forKey: "rewriteLatencyProfile")

            var settings = FeatureSettingsStore.deriveFromLegacy(defaults: defaults)
            settings.meeting.summaryModelSelectionID = .localLLM(CustomLLMModelManager.defaultModelRepo)
            settings.meeting.summaryPrompt = AppPromptDefaults.resolvedStoredText(
                "",
                kind: .transcriptSummary,
                defaults: defaults
            )
            FeatureSettingsStore.save(settings, defaults: defaults)
            let reloaded = FeatureSettingsStore.load(defaults: defaults)

            XCTAssertNil(defaults.object(forKey: "enhancementLatencyProfile"))
            XCTAssertNil(defaults.object(forKey: "translationLatencyProfile"))
            XCTAssertNil(defaults.object(forKey: "rewriteLatencyProfile"))
            XCTAssertEqual(reloaded, settings)
        }
    }

    func testSavePersistsFeatureAvailabilityAndSyncsLegacyFlags() throws {
        try withEphemeralDefaults { defaults in
            var settings = FeatureSettingsStore.deriveFromLegacy(defaults: defaults)
            settings.availability.appEnhancementEnabled = true
            settings.rewrite.appEnhancementEnabled = true

            FeatureSettingsStore.save(settings, defaults: defaults)

            XCTAssertTrue(defaults.bool(forKey: AppPreferenceKey.appEnhancementEnabled))
            XCTAssertTrue(FeatureSettingsStore.load(defaults: defaults).availability.appEnhancementEnabled)

            settings.availability.appEnhancementEnabled = false
            settings.rewrite.appEnhancementEnabled = false
            settings.availability.notesEnabled = false
            settings.transcription.notes.enabled = false
            settings.availability.meetingEnabled = false
            FeatureSettingsStore.save(settings, defaults: defaults)
            let reloaded = FeatureSettingsStore.load(defaults: defaults)

            XCTAssertFalse(defaults.bool(forKey: AppPreferenceKey.appEnhancementEnabled))
            XCTAssertFalse(reloaded.rewrite.appEnhancementEnabled)
            XCTAssertFalse(reloaded.availability.appEnhancementEnabled)
            XCTAssertFalse(reloaded.availability.notesEnabled)
            XCTAssertFalse(reloaded.transcription.notes.enabled)
            XCTAssertFalse(reloaded.availability.meetingEnabled)
        }
    }

    func testMissingAvailabilityDefaultsToAllEnabled() throws {
        try withEphemeralDefaults { defaults in
            var settings = FeatureSettingsStore.deriveFromLegacy(defaults: defaults)
            FeatureSettingsStore.save(settings, defaults: defaults)

            guard var json = try JSONSerialization.jsonObject(
                with: Data((defaults.string(forKey: AppPreferenceKey.featureSettings) ?? "").utf8)
            ) as? [String: Any] else {
                XCTFail("Expected feature settings JSON object")
                return
            }
            json.removeValue(forKey: "availability")
            let data = try JSONSerialization.data(withJSONObject: json)
            defaults.set(String(data: data, encoding: .utf8), forKey: AppPreferenceKey.featureSettings)

            let reloaded = FeatureSettingsStore.load(defaults: defaults)
            XCTAssertEqual(reloaded.availability, .allEnabled)
            XCTAssertTrue(reloaded.transcription.notes.enabled)
            XCTAssertTrue(reloaded.rewrite.appEnhancementEnabled)
        }
    }

    func testHotkeyRuntimeConfigurationSkipsDisabledFeatures() throws {
        try withEphemeralDefaults { defaults in
            var settings = FeatureSettingsStore.deriveFromLegacy(defaults: defaults)
            settings.availability.translationEnabled = false
            settings.availability.rewriteEnabled = false
            settings.availability.notesEnabled = false
            settings.availability.meetingEnabled = false
            FeatureSettingsStore.save(settings, defaults: defaults)

            let configuration = HotkeyRuntimeConfiguration.load(defaults: defaults)
            XCTAssertFalse(configuration.transcriptionBindings.isEmpty)
            XCTAssertTrue(configuration.translationBindings.isEmpty)
            XCTAssertTrue(configuration.rewriteBindings.isEmpty)
            XCTAssertTrue(configuration.noteBindings.isEmpty)
            XCTAssertTrue(configuration.meetingBindings.isEmpty)
            XCTAssertNil(configuration.meetingHotkey)
        }
    }

    func testPromptSpecificSaveHelpersPersistLatestPromptText() throws {
        try withEphemeralDefaults { defaults in
            let transcriptionPrompt = "Clean this transcript, keep it compact."
            let translationPrompt = "Translate into {{TARGET_LANGUAGE}} and keep product names in English."
            let rewritePrompt = "Rewrite the text to sound polite and concise."

            FeatureSettingsStore.saveTranscriptionPrompt(transcriptionPrompt, defaults: defaults)
            FeatureSettingsStore.saveTranslationPrompt(translationPrompt, defaults: defaults)
            FeatureSettingsStore.saveRewritePrompt(rewritePrompt, defaults: defaults)

            let reloaded = FeatureSettingsStore.load(defaults: defaults)

            XCTAssertEqual(reloaded.transcription.prompt, transcriptionPrompt)
            XCTAssertEqual(reloaded.translation.prompt, translationPrompt)
            XCTAssertEqual(reloaded.rewrite.prompt, rewritePrompt)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.enhancementSystemPrompt), transcriptionPrompt)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.translationSystemPrompt), translationPrompt)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.rewriteSystemPrompt), rewritePrompt)
        }
    }

    func testPromptSpecificSaveHelpersPersistPresetSelection() throws {
        try withEphemeralDefaults { defaults in
            let transcriptionPreset = FeaturePromptPresetCatalog.preset(
                id: "structured",
                for: .enhancement,
                language: .english
            )!
            let translationPreset = FeaturePromptPresetCatalog.preset(
                id: "natural",
                for: .translation,
                language: .english
            )!
            let rewritePreset = FeaturePromptPresetCatalog.preset(
                id: "strict",
                for: .rewrite,
                language: .english
            )!

            FeatureSettingsStore.saveTranscriptionPrompt(
                transcriptionPreset.prompt,
                presetID: transcriptionPreset.id,
                defaults: defaults
            )
            FeatureSettingsStore.saveTranslationPrompt(
                translationPreset.prompt,
                presetID: translationPreset.id,
                defaults: defaults
            )
            FeatureSettingsStore.saveRewritePrompt(
                rewritePreset.prompt,
                presetID: rewritePreset.id,
                defaults: defaults
            )

            let reloaded = FeatureSettingsStore.load(defaults: defaults)
            XCTAssertEqual(reloaded.transcription.promptPresetID, "structured")
            XCTAssertEqual(reloaded.translation.promptPresetID, "natural")
            XCTAssertEqual(reloaded.rewrite.promptPresetID, "strict")
        }
    }

    func testLegacyCustomPromptsRemainCustomAfterPresetMigration() throws {
        try withEphemeralDefaults { defaults in
            defaults.set("My custom cleanup prompt", forKey: AppPreferenceKey.enhancementSystemPrompt)
            defaults.set("My custom translation prompt", forKey: AppPreferenceKey.translationSystemPrompt)
            defaults.set("My custom rewrite prompt", forKey: AppPreferenceKey.rewriteSystemPrompt)

            let settings = FeatureSettingsStore.load(defaults: defaults)

            XCTAssertNil(settings.transcription.promptPresetID)
            XCTAssertNil(settings.translation.promptPresetID)
            XCTAssertNil(settings.rewrite.promptPresetID)
        }
    }

    func testLocalVADModeDefaultsToAutomatic() throws {
        try withEphemeralDefaults { defaults in
            _ = FeatureSettingsStore.load(defaults: defaults)

            XCTAssertEqual(LocalVADMode.stored(defaults: defaults), .automatic)
        }
    }

    func testLocalVADModeCanBeDisabledGlobally() throws {
        try withEphemeralDefaults { defaults in
            LocalVADMode.save(.off, defaults: defaults)

            XCTAssertEqual(LocalVADMode.stored(defaults: defaults), .off)
        }
    }

    func testSaveSyncsLegacyPromptKeysFromFeatureSettingsPayload() throws {
        try withEphemeralDefaults { defaults in
            var settings = FeatureSettingsStore.deriveFromLegacy(defaults: defaults)
            settings.transcription.prompt = "Enhance with my custom cleanup rules."
            settings.translation.prompt = "Translate to {{TARGET_LANGUAGE}} and preserve app names."
            settings.rewrite.prompt = "Rewrite as concise release notes."

            FeatureSettingsStore.save(settings, defaults: defaults)

            XCTAssertEqual(
                defaults.string(forKey: AppPreferenceKey.enhancementSystemPrompt),
                "Enhance with my custom cleanup rules."
            )
            XCTAssertEqual(
                defaults.string(forKey: AppPreferenceKey.translationSystemPrompt),
                "Translate to {{TARGET_LANGUAGE}} and preserve app names."
            )
            XCTAssertEqual(
                defaults.string(forKey: AppPreferenceKey.rewriteSystemPrompt),
                "Rewrite as concise release notes."
            )
        }
    }

    func testMigrationDropsRetiredContextSettingsWithoutChangingOtherPreferences() throws {
        try withEphemeralDefaults { defaults in
            var settings = FeatureSettingsStore.load(defaults: defaults)
            settings.transcription.prompt = "Keep my custom transcription prompt."
            settings.rewrite.prompt = "Keep my custom screenshot instructions verbatim."
            FeatureSettingsStore.save(settings, defaults: defaults)
            let expected = FeatureSettingsStore.load(defaults: defaults)
            let raw = try XCTUnwrap(defaults.string(forKey: AppPreferenceKey.featureSettings))
            let data = try XCTUnwrap(raw.data(using: .utf8))
            let original = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let oldContexts: [[String: Any]] = [
                ["enabled": true],
                ["enabled": true, "textEnabled": true, "screenshotEnabled": true]
            ]
            for oldContext in oldContexts {
                var legacy = original
                for feature in ["transcription", "rewrite"] {
                    var value = try XCTUnwrap(legacy[feature] as? [String: Any])
                    value["appContext"] = oldContext
                    legacy[feature] = value
                }
                let legacyData = try JSONSerialization.data(withJSONObject: legacy, options: .sortedKeys)
                let legacyRaw = try XCTUnwrap(String(data: legacyData, encoding: .utf8))
                defaults.set(legacyRaw, forKey: AppPreferenceKey.featureSettings)
                XCTAssertEqual(FeatureSettingsStore.load(defaults: defaults), expected)
                // Reads must not migrate/write through SwiftUI observation.
                XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.featureSettings), legacyRaw)
                FeatureSettingsStore.migrateIfNeeded(defaults: defaults)
                let migrated = try XCTUnwrap(defaults.string(forKey: AppPreferenceKey.featureSettings))
                XCTAssertFalse(migrated.contains("appContext"))
                XCTAssertEqual(FeatureSettingsStore.load(defaults: defaults), expected)
                FeatureSettingsStore.migrateIfNeeded(defaults: defaults)
                XCTAssertEqual(FeatureSettingsStore.load(defaults: defaults), expected)
                XCTAssertFalse(try XCTUnwrap(defaults.string(forKey: AppPreferenceKey.featureSettings)).contains("appContext"))
            }
        }
    }

    func testNotePanelSettingsPersistAndClampDelays() throws {
        try withEphemeralDefaults { defaults in
            var settings = FeatureSettingsStore.load(defaults: defaults)
            settings.transcription.notes.panel = VoxtNotePanelSettings(
                corner: .bottomLeft,
                revealDelay: 9,
                hideDelay: 0,
                isTranslucent: false
            )

            FeatureSettingsStore.save(settings, defaults: defaults)
            let reloaded = FeatureSettingsStore.load(defaults: defaults)

            XCTAssertEqual(reloaded.transcription.notes.panel.corner, .bottomLeft)
            XCTAssertEqual(reloaded.transcription.notes.panel.revealDelay, 2.0)
            XCTAssertEqual(reloaded.transcription.notes.panel.hideDelay, 0.1)
            XCTAssertFalse(reloaded.transcription.notes.panel.isTranslucent)
        }
    }

    func testMeetingRuntimePreferencesDoNotUseMeetingVADBackend() throws {
        try withEphemeralDefaults { defaults in
            XCTAssertEqual(MeetingChunkingMode.stored(in: defaults), .quality)
            XCTAssertEqual(MeetingDiarizationMode.stored(in: defaults), .sortformerV2)

            defaults.set(MeetingChunkingMode.quality.rawValue, forKey: AppPreferenceKey.meetingChunkingMode)
            defaults.set(MeetingDiarizationMode.sortformerV2.rawValue, forKey: AppPreferenceKey.meetingRealtimeDiarizationMode)
            defaults.set("responsive", forKey: AppPreferenceKey.meetingSileroVADSensitivity)
            defaults.set("stable", forKey: AppPreferenceKey.meetingServerVADMode)
            defaults.set("sensitive", forKey: "meetingSpeakerDiarizationSensitivity")
            defaults.set("maxThree", forKey: "meetingSpeakerCountHint")
            defaults.set(true, forKey: "meetingSpeakerDiarizationDebugEnabled")
            defaults.set(false, forKey: AppPreferenceKey.meetingFinalTranscriptOptimizationEnabled)

            var settings = FeatureSettingsStore.load(defaults: defaults)

            XCTAssertEqual(settings.meeting.chunkingMode, .quality)
            XCTAssertEqual(settings.meeting.sileroVADSensitivity, .responsive)
            XCTAssertEqual(settings.meeting.speakerDiarizationModel, .sortformerV2)
            XCTAssertFalse(settings.meeting.finalTranscriptOptimizationEnabled)

            settings.meeting.chunkingModeRawValue = MeetingChunkingMode.realtime.rawValue
            settings.meeting.sileroVADSensitivityRawValue = MeetingSileroVADSensitivity.stable.rawValue
            settings.meeting.speakerDiarizationModelRawValue = MeetingDiarizationMode.sortformerV2.rawValue
            settings.meeting.finalTranscriptOptimizationEnabled = false
            FeatureSettingsStore.save(settings, defaults: defaults)
            FeatureSettingsStore.prepareMeetingRuntime(from: settings, defaults: defaults)

            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.meetingChunkingMode), MeetingChunkingMode.realtime.rawValue)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.meetingSpeakerDiarizationModel), MeetingDiarizationMode.sortformerV2.rawValue)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.meetingSileroVADSensitivity), "stable")
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.meetingServerVADMode), "stable")
            XCTAssertNil(defaults.object(forKey: "meetingSpeakerDiarizationSensitivity"))
            XCTAssertNil(defaults.object(forKey: "meetingSpeakerCountHint"))
            XCTAssertNil(defaults.object(forKey: "meetingSpeakerDiarizationDebugEnabled"))
            XCTAssertFalse(defaults.bool(forKey: AppPreferenceKey.meetingFinalTranscriptOptimizationEnabled))
            XCTAssertEqual(FeatureSettingsStore.load(defaults: defaults).meeting.chunkingMode, .realtime)
            XCTAssertEqual(FeatureSettingsStore.load(defaults: defaults).meeting.sileroVADSensitivity, .stable)
            XCTAssertEqual(FeatureSettingsStore.load(defaults: defaults).meeting.speakerDiarizationModel, .sortformerV2)
            XCTAssertFalse(FeatureSettingsStore.load(defaults: defaults).meeting.finalTranscriptOptimizationEnabled)
        }
    }

    func testMeetingSettingsDecodePayloadWithoutNewRuntimeFieldsUsesDefaults() throws {
        let payload = """
        {
          "asrSelectionID": "mlx:mlx-community/SenseVoiceSmall",
          "summaryModelSelectionID": "local-llm:mlx-community/Qwen3.5-2B-4bit",
          "summaryPrompt": "",
          "summaryAutoGenerate": true,
          "realtimeTranslateEnabled": false,
          "realtimeTargetLanguageRawValue": "",
          "hideOverlayFromScreenSharing": false
        }
        """
        let data = try XCTUnwrap(payload.data(using: .utf8))

        let settings = try JSONDecoder().decode(MeetingFeatureSettings.self, from: data)

        XCTAssertEqual(settings.chunkingMode, .quality)
        XCTAssertEqual(settings.sileroVADSensitivity, .balanced)
        XCTAssertEqual(settings.speakerDiarizationModel, .sortformerV2)
        XCTAssertTrue(settings.finalTranscriptOptimizationEnabled)
    }

    func testMeetingSettingsStorageDoesNotEncodeVADBackend() throws {
        try withEphemeralDefaults { defaults in
            let settings = FeatureSettingsStore.load(defaults: defaults)
            FeatureSettingsStore.save(settings, defaults: defaults)
            let raw = try XCTUnwrap(defaults.string(forKey: AppPreferenceKey.featureSettings))

            XCTAssertFalse(raw.contains("vadBackendRawValue"))
        }
    }

    func testMeetingVADIgnoresObsoleteModeKey() throws {
        try withEphemeralDefaults { defaults in
            defaults.set("legacyExperimentalBackend", forKey: "meetingVADMode")

            let settings = FeatureSettingsStore.load(defaults: defaults)

            XCTAssertEqual(defaults.string(forKey: "meetingVADMode"), "legacyExperimentalBackend")
            FeatureSettingsStore.save(settings, defaults: defaults)
            XCTAssertFalse(defaults.string(forKey: AppPreferenceKey.featureSettings)?.contains("meetingVADMode") ?? true)
        }
    }

    func testMeetingDiarizationModeIgnoresLegacyRealtimeKey() throws {
        try withEphemeralDefaults { defaults in
            defaults.set(MeetingDiarizationMode.sortformerV2.rawValue, forKey: AppPreferenceKey.meetingRealtimeDiarizationMode)
            XCTAssertEqual(MeetingDiarizationMode.stored(in: defaults), .sortformerV2)

            defaults.set(MeetingDiarizationMode.sortformerV2.rawValue, forKey: AppPreferenceKey.meetingSpeakerDiarizationModel)
            XCTAssertEqual(MeetingDiarizationMode.stored(in: defaults), .sortformerV2)
        }
    }

    func testRetiredDiarizerMigrationIsIdempotent() throws {
        try withEphemeralDefaults { defaults in
            defaults.set("offlineVBx", forKey: AppPreferenceKey.meetingSpeakerDiarizationModel)
            defaults.set("maxSix", forKey: "meetingSpeakerCountHint")
            FeatureSettingsStore.migrateIfNeeded(defaults: defaults)
            let first = defaults.string(forKey: AppPreferenceKey.featureSettings)
            XCTAssertEqual(MeetingDiarizationMode.stored(in: defaults), .sortformerV2)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.meetingSpeakerDiarizationModel), "sortformerV2")
            XCTAssertNil(defaults.object(forKey: "meetingSpeakerCountHint"))
            XCTAssertFalse(first?.contains("offlineVBx") ?? true)
            FeatureSettingsStore.migrateIfNeeded(defaults: defaults)
            XCTAssertEqual(FeatureSettingsStore.load(defaults: defaults).meeting.speakerDiarizationModel, .sortformerV2)
        }
    }

    func testMeetingSpeakerRuntimeOptionsIgnoreLegacyPreferences() throws {
        try withEphemeralDefaults { defaults in
            defaults.set(MeetingSpeakerDiarizationSensitivity.sensitive.rawValue, forKey: "meetingSpeakerDiarizationSensitivity")
            defaults.set("maxFour", forKey: "meetingSpeakerCountHint")
            defaults.set(true, forKey: "meetingSpeakerDiarizationDebugEnabled")

            let options = MeetingSpeakerDiarizationOptions.fromPreferences(defaults: defaults)

            XCTAssertEqual(options.sensitivity, .balanced)
            XCTAssertFalse(options.debugLoggingEnabled)
            XCTAssertEqual(options.minimumSpeakerConfidence, MeetingSpeakerDiarizationSensitivity.balanced.minimumSpeakerConfidence)
            XCTAssertEqual(options.smoothing, MeetingSpeakerDiarizationSensitivity.balanced.smootherOptions)
            XCTAssertEqual(options.transcriptAssembly, MeetingSpeakerDiarizationSensitivity.balanced.transcriptAssemblyOptions)
        }
    }
}
