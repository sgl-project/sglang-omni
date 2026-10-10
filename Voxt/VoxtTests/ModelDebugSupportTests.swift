// ModelDebugSupportTests.swift
// Provides Model Debug Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class ModelDebugSupportTests: XCTestCase {
    private func withEphemeralDefaults(
        _ body: (UserDefaults) throws -> Void
    ) rethrows {
        let suiteName = "ModelDebugSupportTests.\(UUID().uuidString)"
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

    func testVADSnapshotExposesGlobalModeAndEffectiveBackend() throws {
        try withEphemeralDefaults { defaults in
            LocalVADMode.save(.automatic, defaults: defaults)

            let snapshot = ModelDebugCatalog.vadSnapshot(
                defaults: defaults,
                transcriptionEngine: .mlxAudio,
                providerUsesServerVAD: false
            )

            XCTAssertEqual(snapshot.mode, .automatic)
            XCTAssertEqual(snapshot.backend, .mlxSilero)
            XCTAssertEqual(snapshot.backendRawValue, ASRVoiceActivityBackendKind.mlxSilero.rawValue)
            XCTAssertEqual(snapshot.frameBackend, .mlxSilero)
            XCTAssertEqual(snapshot.localGatePolicy, .enabled)
            XCTAssertFalse(snapshot.providerUsesServerVAD)
        }
    }

    func testVADSnapshotShowsOffMode() throws {
        try withEphemeralDefaults { defaults in
            LocalVADMode.save(.off, defaults: defaults)

            let snapshot = ModelDebugCatalog.vadSnapshot(
                defaults: defaults,
                transcriptionEngine: .mlxAudio,
                providerUsesServerVAD: false
            )

            XCTAssertEqual(snapshot.mode, .off)
            XCTAssertEqual(snapshot.backend, .off)
            XCTAssertEqual(snapshot.frameBackend, .off)
            XCTAssertEqual(snapshot.localGatePolicy, .disabled(reason: "local-vad-off"))
        }
    }

    func testVADSnapshotShowsOmniBackend() throws {
        try withEphemeralDefaults { defaults in
            LocalVADMode.save(.omni, defaults: defaults)

            let snapshot = ModelDebugCatalog.vadSnapshot(
                defaults: defaults,
                transcriptionEngine: .mlxAudio,
                providerUsesServerVAD: false
            )

            XCTAssertEqual(snapshot.mode, .omni)
            XCTAssertEqual(snapshot.backend, .omniStream)
            XCTAssertEqual(snapshot.backendRawValue, ASRVoiceActivityBackendKind.omniStream.rawValue)
            XCTAssertEqual(snapshot.frameBackend, .omniStream)
            XCTAssertEqual(snapshot.localGatePolicy, .enabled)
        }
    }

    func testVADSnapshotDisablesLocalGateForRemoteASR() throws {
        try withEphemeralDefaults { defaults in
            LocalVADMode.save(.silero, defaults: defaults)

            let snapshot = ModelDebugCatalog.vadSnapshot(
                defaults: defaults,
                transcriptionEngine: .remote,
                providerUsesServerVAD: false
            )

            XCTAssertEqual(snapshot.mode, .silero)
            XCTAssertEqual(snapshot.backend, .mlxSilero)
            XCTAssertEqual(snapshot.localGatePolicy, .disabled(reason: "non-local-asr"))
        }
    }

    func testVADSnapshotMarksServerVADWhenProviderUsesRealtimeProfile() throws {
        try withEphemeralDefaults { defaults in
            LocalVADMode.save(.silero, defaults: defaults)

            let snapshot = ModelDebugCatalog.vadSnapshot(
                defaults: defaults,
                transcriptionEngine: .remote,
                providerUsesServerVAD: true
            )

            XCTAssertTrue(snapshot.providerUsesServerVAD)
            XCTAssertEqual(snapshot.frameBackend, .mlxSilero)
            XCTAssertEqual(snapshot.localGatePolicy, .disabled(reason: "server-vad"))
        }
    }

    func testLLMDebugPresetsIncludeBuiltinsAndSavedGroups() throws {
        let defaults = UserDefaults.standard
        let previousGroups = defaults.data(forKey: AppPreferenceKey.appBranchGroups)
        let previousPrompt = defaults.string(forKey: AppPreferenceKey.enhancementSystemPrompt)
        let previousLanguageCodes = defaults.string(forKey: AppPreferenceKey.userMainLanguageCodes)

        let groups = [
            AppBranchGroup(
                id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
                name: "Chrome",
                prompt: "Clean {{RAW_TRANSCRIPTION}} for browser work",
                appBundleIDs: ["com.google.Chrome"],
                appRefs: [AppBranchAppRef(bundleID: "com.google.Chrome", displayName: "Chrome")],
                urlPatternIDs: [],
                isExpanded: true
            )
        ]
        defaults.set(try JSONEncoder().encode(groups), forKey: AppPreferenceKey.appBranchGroups)
        defaults.set("Base {{RAW_TRANSCRIPTION}}", forKey: AppPreferenceKey.enhancementSystemPrompt)
        defaults.set("en", forKey: AppPreferenceKey.userMainLanguageCodes)
        addTeardownBlock {
            if let previousGroups {
                defaults.set(previousGroups, forKey: AppPreferenceKey.appBranchGroups)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.appBranchGroups)
            }
            if let previousPrompt {
                defaults.set(previousPrompt, forKey: AppPreferenceKey.enhancementSystemPrompt)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.enhancementSystemPrompt)
            }
            if let previousLanguageCodes {
                defaults.set(previousLanguageCodes, forKey: AppPreferenceKey.userMainLanguageCodes)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.userMainLanguageCodes)
            }
        }

        let presets = ModelDebugCatalog.availableLLMPresets(defaults: defaults)

        XCTAssertTrue(presets.contains(where: { $0.id == "builtin:enhancement" }))
        XCTAssertTrue(presets.contains(where: { $0.id == "builtin:translation" }))
        XCTAssertTrue(presets.contains(where: { $0.id == "builtin:rewrite" }))
        XCTAssertTrue(presets.contains(where: { $0.id == "builtin:transcript-summary" }))
        XCTAssertTrue(presets.contains(where: { $0.title.contains("Chrome") }))
    }

    func testBuiltInPresetsPreferGlobalPromptOverDebugOverride() throws {
        try withEphemeralDefaults { defaults in
            defaults.set("global enhancement prompt", forKey: AppPreferenceKey.enhancementSystemPrompt)
            defaults.set("global translation prompt", forKey: AppPreferenceKey.translationSystemPrompt)
            defaults.set("global rewrite prompt", forKey: AppPreferenceKey.rewriteSystemPrompt)
            defaults.set(
                try JSONEncoder().encode([
                    "builtin:enhancement": "stale debug enhancement",
                    "builtin:translation": "stale debug translation",
                    "builtin:rewrite": "stale debug rewrite"
                ]),
                forKey: AppPreferenceKey.llmDebugPresetPromptOverrides
            )

            let presets = ModelDebugCatalog.availableLLMPresets(defaults: defaults)

            XCTAssertEqual(
                presets.first(where: { $0.id == "builtin:enhancement" })?.promptTemplate,
                "global enhancement prompt"
            )
            XCTAssertEqual(
                presets.first(where: { $0.id == "builtin:translation" })?.promptTemplate,
                "global translation prompt"
            )
            XCTAssertEqual(
                presets.first(where: { $0.id == "builtin:rewrite" })?.promptTemplate,
                "global rewrite prompt"
            )
        }
    }

    func testBuiltInPresetsApplySessionPromptOverridesWhenProvided() throws {
        try withEphemeralDefaults { defaults in
            defaults.set("global enhancement prompt", forKey: AppPreferenceKey.enhancementSystemPrompt)
            defaults.set("global translation prompt", forKey: AppPreferenceKey.translationSystemPrompt)
            defaults.set("global rewrite prompt", forKey: AppPreferenceKey.rewriteSystemPrompt)

            let presets = ModelDebugCatalog.availableLLMPresets(
                defaults: defaults,
                promptOverrides: [
                    "builtin:enhancement": "session enhancement prompt",
                    "builtin:translation": "session translation prompt",
                    "builtin:rewrite": "session rewrite prompt"
                ]
            )

            XCTAssertEqual(
                presets.first(where: { $0.id == "builtin:enhancement" })?.promptTemplate,
                "session enhancement prompt"
            )
            XCTAssertEqual(
                presets.first(where: { $0.id == "builtin:translation" })?.promptTemplate,
                "session translation prompt"
            )
            XCTAssertEqual(
                presets.first(where: { $0.id == "builtin:rewrite" })?.promptTemplate,
                "session rewrite prompt"
            )
        }
    }

    func testBuiltInPresetsUseFeatureSettingsPromptsWhenLegacyKeysAreEmpty() throws {
        try withEphemeralDefaults { defaults in
            var settings = FeatureSettingsStore.deriveFromLegacy(defaults: defaults)
            settings.transcription.prompt = "feature enhancement prompt"
            settings.translation.prompt = "feature translation prompt"
            settings.rewrite.prompt = "feature rewrite prompt"

            FeatureSettingsStore.save(settings, defaults: defaults)
            defaults.set("", forKey: AppPreferenceKey.enhancementSystemPrompt)
            defaults.set("", forKey: AppPreferenceKey.translationSystemPrompt)
            defaults.set("", forKey: AppPreferenceKey.rewriteSystemPrompt)

            let presets = ModelDebugCatalog.availableLLMPresets(defaults: defaults)

            XCTAssertEqual(
                presets.first(where: { $0.id == "builtin:enhancement" })?.promptTemplate,
                "feature enhancement prompt"
            )
            XCTAssertEqual(
                presets.first(where: { $0.id == "builtin:translation" })?.promptTemplate,
                "feature translation prompt"
            )
            XCTAssertEqual(
                presets.first(where: { $0.id == "builtin:rewrite" })?.promptTemplate,
                "feature rewrite prompt"
            )
        }
    }

    func testPromptResolverInjectsEnhancementVariables() {
        let preset = LLMDebugPresetOption(
            id: "builtin:enhancement",
            title: "Enhancement",
            subtitle: "Built-in preset",
            kind: .enhancement,
            promptTemplate: "Clean {{RAW_TRANSCRIPTION}} for {{USER_MAIN_LANGUAGE}}",
            variables: ModelSettingsPromptVariables.enhancement,
            defaultValues: [
                AppDelegate.rawTranscriptionTemplateVariable: "",
                AppDelegate.userMainLanguageTemplateVariable: "English"
            ]
        )

        let resolved = ModelDebugPromptResolver.resolve(
            preset: preset,
            values: [
                AppDelegate.rawTranscriptionTemplateVariable: "hello world",
                AppDelegate.userMainLanguageTemplateVariable: "Chinese"
            ]
        )

        XCTAssertTrue(resolved.content.contains("hello world"))
        XCTAssertTrue(resolved.content.contains("Chinese"))
        XCTAssertEqual(resolved.inputSummary, "hello world")
    }

    func testPromptResolverInjectsTranscriptSummaryVariables() {
        let preset = LLMDebugPresetOption(
            id: "builtin:transcript-summary",
            title: "Transcript Summary",
            subtitle: "Built-in preset",
            kind: .transcriptSummary,
            promptTemplate: "Summary: {{TRANSCRIPT_RECORD}} | Lang: {{USER_MAIN_LANGUAGE}}",
            variables: [],
            defaultValues: [:]
        )

        let resolved = ModelDebugPromptResolver.resolve(
            preset: preset,
            values: [
                "{{TRANSCRIPT_RECORD}}": "Discuss launch blockers",
                AppPreferenceKey.asrUserMainLanguageTemplateVariable: "Japanese"
            ]
        )

        XCTAssertTrue(resolved.content.contains("Discuss launch blockers"))
        XCTAssertTrue(resolved.content.contains("Japanese"))
        XCTAssertEqual(resolved.inputSummary, "Discuss launch blockers")
    }

    func testRuntimeInputsExposeTaskTextSeparatelyFromPromptVariables() {
        let enhancement = LLMDebugPresetOption(
            id: "builtin:enhancement",
            title: "Enhancement",
            subtitle: "Built-in preset",
            kind: .enhancement,
            promptTemplate: AppPromptDefaults.text(for: .enhancement, language: .english),
            variables: ModelSettingsPromptVariables.enhancement,
            defaultValues: [:]
        )
        let translation = LLMDebugPresetOption(
            id: "builtin:translation",
            title: "Translation",
            subtitle: "Built-in preset",
            kind: .translation,
            promptTemplate: AppPromptDefaults.text(for: .translation, language: .english),
            variables: ModelSettingsPromptVariables.translation,
            defaultValues: [:]
        )
        let rewrite = LLMDebugPresetOption(
            id: "builtin:rewrite",
            title: "Rewrite",
            subtitle: "Built-in preset",
            kind: .rewrite,
            promptTemplate: AppPromptDefaults.text(for: .rewrite, language: .english),
            variables: ModelSettingsPromptVariables.rewrite,
            defaultValues: [:]
        )

        XCTAssertTrue(enhancement.runtimeInputDescriptors.contains(where: { $0.token == AppDelegate.rawTranscriptionTemplateVariable }))
        XCTAssertFalse(enhancement.variables.contains(where: { $0.token == AppDelegate.rawTranscriptionTemplateVariable }))
        XCTAssertTrue(translation.runtimeInputDescriptors.contains(where: { $0.token == "{{SOURCE_TEXT}}" }))
        XCTAssertFalse(translation.variables.contains(where: { $0.token == "{{SOURCE_TEXT}}" }))
        XCTAssertTrue(rewrite.runtimeInputDescriptors.contains(where: { $0.token == "{{DICTATED_PROMPT}}" }))
        XCTAssertTrue(rewrite.runtimeInputDescriptors.contains(where: { $0.token == "{{SOURCE_TEXT}}" }))
        XCTAssertTrue(rewrite.variables.isEmpty)
    }

    func testAppGroupDebugPresetInjectsRuntimeRawTranscriptionIntoCompiledPrompt() throws {
        let suiteName = "ModelDebugSupportTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        addTeardownBlock {
            defaults.removePersistentDomain(forName: suiteName)
        }

        let groupID = UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!
        let groups = [
            AppBranchGroup(
                id: groupID,
                name: "Chrome",
                prompt: "Clean browser dictation for {{USER_MAIN_LANGUAGE}}.",
                appBundleIDs: ["com.google.Chrome"],
                appRefs: [AppBranchAppRef(bundleID: "com.google.Chrome", displayName: "Chrome")],
                urlPatternIDs: [],
                isExpanded: true
            )
        ]
        defaults.set(try JSONEncoder().encode(groups), forKey: AppPreferenceKey.appBranchGroups)
        defaults.set("en", forKey: AppPreferenceKey.userMainLanguageCodes)

        let preset = try XCTUnwrap(
            ModelDebugCatalog.availableLLMPresets(defaults: defaults)
                .first(where: { $0.id == "group:\(groupID.uuidString)" })
        )
        let resolved = ModelDebugPromptResolver.resolve(
            preset: preset,
            values: [
                AppDelegate.rawTranscriptionTemplateVariable: "raw browser asr",
                AppDelegate.userMainLanguageTemplateVariable: "English"
            ],
            defaults: defaults
        )

        let compiled = try XCTUnwrap(resolved.compiledRequest)
        XCTAssertTrue(preset.runtimeInputDescriptors.contains(where: { $0.token == AppDelegate.rawTranscriptionTemplateVariable }))
        XCTAssertContains(compiled.instructions, "Clean browser dictation")
        XCTAssertContains(compiled.prompt, "raw browser asr")
        XCTAssertFalse(compiled.instructions.contains("raw browser asr"))
    }

    func testPromptResolverBuildsCompiledEnhancementRequestForDefaultPrompt() {
        let preset = LLMDebugPresetOption(
            id: "builtin:enhancement",
            title: "Enhancement",
            subtitle: "Built-in preset",
            kind: .enhancement,
            promptTemplate: AppPromptDefaults.text(for: .enhancement, language: .english),
            variables: ModelSettingsPromptVariables.enhancement,
            defaultValues: [
                AppDelegate.rawTranscriptionTemplateVariable: "",
                AppDelegate.userMainLanguageTemplateVariable: "Simplified Chinese"
            ]
        )

        let resolved = ModelDebugPromptResolver.resolve(
            preset: preset,
            values: [
                AppDelegate.rawTranscriptionTemplateVariable: "原始输入",
                AppDelegate.userMainLanguageTemplateVariable: "Simplified Chinese"
            ]
        )

        let compiled = try! XCTUnwrap(resolved.compiledRequest)
        XCTAssertContains(compiled.instructions, "Voxt")
        XCTAssertContains(compiled.instructions, "Runtime language preservation rules:")
        XCTAssertContains(compiled.prompt, "原始输入")
        XCTAssertFalse(compiled.instructions.contains("Raw transcription"))
    }

    func testPromptResolverBuildsCompiledTranslationRequestWithoutEmbeddingSourceTextInInstructionsByDefault() {
        let preset = LLMDebugPresetOption(
            id: "builtin:translation",
            title: "Translation",
            subtitle: "Built-in preset",
            kind: .translation,
            promptTemplate: AppPromptDefaults.text(for: .translation, language: .english),
            variables: ModelSettingsPromptVariables.translation,
            defaultValues: [
                "{{TARGET_LANGUAGE}}": "English",
                AppDelegate.userMainLanguageTemplateVariable: "Chinese",
                "{{SOURCE_TEXT}}": ""
            ]
        )

        let resolved = ModelDebugPromptResolver.resolve(
            preset: preset,
            values: [
                "{{TARGET_LANGUAGE}}": "English",
                AppDelegate.userMainLanguageTemplateVariable: "Chinese",
                "{{SOURCE_TEXT}}": "待翻译原文"
            ]
        )

        let compiled = try! XCTUnwrap(resolved.compiledRequest)
        XCTAssertContains(compiled.instructions, "cleanup and translation assistant")
        XCTAssertContains(compiled.prompt, "待翻译原文")
        XCTAssertFalse(compiled.instructions.contains("待翻译原文"))
    }

    func testPromptResolverRoutesLegacyRewriteTemplateThroughUserPromptPreview() {
        let preset = LLMDebugPresetOption(
            id: "builtin:rewrite",
            title: "Rewrite",
            subtitle: "Built-in preset",
            kind: .rewrite,
            promptTemplate: "Rewrite {{SOURCE_TEXT}} with {{DICTATED_PROMPT}}",
            variables: ModelSettingsPromptVariables.rewrite,
            defaultValues: [:]
        )

        let resolved = ModelDebugPromptResolver.resolve(
            preset: preset,
            values: [
                "{{DICTATED_PROMPT}}": "make it shorter",
                "{{SOURCE_TEXT}}": "Original paragraph"
            ]
        )

        let compiled = try! XCTUnwrap(resolved.compiledRequest)
        XCTAssertContains(compiled.prompt, "Original paragraph")
        XCTAssertContains(compiled.prompt, "make it shorter")
        XCTAssertFalse(compiled.instructions.contains("Structured answer output"))
    }

    func testPromptResolverUsesRequestedTranslationTargetLanguage() {
        let preset = LLMDebugPresetOption(
            id: "builtin:translation",
            title: "Translation",
            subtitle: "Built-in preset",
            kind: .translation,
            promptTemplate: "Translate {{SOURCE_TEXT}} into {{TARGET_LANGUAGE}} for {{USER_MAIN_LANGUAGE}}",
            variables: ModelSettingsPromptVariables.translation,
            defaultValues: [
                "{{TARGET_LANGUAGE}}": "English",
                AppDelegate.userMainLanguageTemplateVariable: "Chinese",
                "{{SOURCE_TEXT}}": ""
            ]
        )

        let resolved = ModelDebugPromptResolver.resolve(
            preset: preset,
            values: [
                "{{TARGET_LANGUAGE}}": "Japanese",
                AppDelegate.userMainLanguageTemplateVariable: "English",
                "{{SOURCE_TEXT}}": "你好"
            ]
        )

        XCTAssertTrue(resolved.content.contains("Japanese"))
        XCTAssertTrue(resolved.content.contains("你好"))
        XCTAssertEqual(resolved.inputSummary, "你好")
    }

    func testPromptResolverUsesDictatedPromptSummaryWhenRewriteSourceIsEmpty() {
        let preset = LLMDebugPresetOption(
            id: "builtin:rewrite",
            title: "Rewrite",
            subtitle: "Built-in preset",
            kind: .rewrite,
            promptTemplate: "Rewrite {{SOURCE_TEXT}} with {{DICTATED_PROMPT}}",
            variables: ModelSettingsPromptVariables.rewrite,
            defaultValues: [:]
        )

        let resolved = ModelDebugPromptResolver.resolve(
            preset: preset,
            values: [
                "{{DICTATED_PROMPT}}": "write a short reply",
                "{{SOURCE_TEXT}}": ""
            ]
        )

        XCTAssertTrue(resolved.content.contains("write a short reply"))
        XCTAssertEqual(resolved.inputSummary, "write a short reply")
        XCTAssertEqual(resolved.requestMetadata?.spokenInstruction, "write a short reply")
        XCTAssertEqual(resolved.requestMetadata?.sourceText, "")
    }

    func testPromptResolverIgnoresRetiredContextPayload() throws {
        let preset = LLMDebugPresetOption(
            id: "builtin:rewrite",
            title: "Rewrite",
            subtitle: "Built-in preset",
            kind: .rewrite,
            promptTemplate: AppPromptDefaults.text(for: .rewrite, language: .english),
            variables: ModelSettingsPromptVariables.rewrite,
            defaultValues: [:]
        )
        let payloadString = #"{"textContext":"PRIVATE_WINDOW_CONTENT","attachments":[{"type":"image","image":{"base64Data":"AQID","filename":"wechat.jpg"}}]}"#

        let resolved = ModelDebugPromptResolver.resolve(
            preset: preset,
            values: [
                "{{DICTATED_PROMPT}}": "make it warmer",
                "{{SOURCE_TEXT}}": "Can you send the update by 5?",
                "__VOXT_DEBUG_REWRITE_APP_CONTEXT_CAPTURE__": payloadString
            ]
        )

        let compiled = try XCTUnwrap(resolved.compiledRequest)
        XCTAssertFalse(compiled.instructions.contains("Active app context"))
        XCTAssertFalse(resolved.content.contains("PRIVATE_WINDOW_CONTENT"))
        XCTAssertFalse(resolved.content.contains("wechat.jpg"))
        XCTAssertFalse(resolved.content.contains("[attachments]"))
        XCTAssertContains(compiled.prompt, "Can you send the update by 5?")
        XCTAssertEqual(resolved.requestMetadata?.spokenInstruction, "make it warmer")
    }

}
