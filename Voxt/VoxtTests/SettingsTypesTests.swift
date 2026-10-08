// SettingsTypesTests.swift
// Provides Settings Types Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class SettingsTypesTests: XCTestCase {
    func testUserMainLanguageSanitizedSelectionDeduplicatesAndFallsBack() {
        XCTAssertEqual(
            UserMainLanguageOption.sanitizedSelection(["zh-CN", "zh-Hans", "EN", "unknown", "en"]),
            ["zh-hans", "en"]
        )
        XCTAssertEqual(
            UserMainLanguageOption.sanitizedSelection(["unknown"]),
            UserMainLanguageOption.defaultSelectionCodes()
        )
    }

    func testStoredSelectionAndStorageValueRoundTrip() {
        let raw = UserMainLanguageOption.storageValue(for: ["zh-Hant", "en"])

        XCTAssertEqual(UserMainLanguageOption.storedSelection(from: raw), ["zh-hant", "en"])
    }

    func testFallbackOptionUsesPreferredLanguages() {
        let option = UserMainLanguageOption.fallbackOption(preferredLanguages: ["zh-TW", "en-US"])

        XCTAssertEqual(option.code, "zh-hant")
        XCTAssertTrue(option.isChinese)
        XCTAssertTrue(option.isTraditionalChinese)
        XCTAssertEqual(option.baseLanguageCode, "zh")
    }

    func testDictionarySuggestionFilterSettingsSanitizedClampsAndDefaultsPrompt() {
        let sanitized = DictionarySuggestionFilterSettings(
            prompt: "   ",
            batchSize: 999,
            maxCandidatesPerBatch: 0
        ).sanitized()

        XCTAssertEqual(sanitized.prompt, DictionarySuggestionFilterSettings.defaultPrompt)
        XCTAssertEqual(sanitized.batchSize, DictionarySuggestionFilterSettings.maximumBatchSize)
        XCTAssertEqual(sanitized.maxCandidatesPerBatch, DictionarySuggestionFilterSettings.minimumMaxCandidates)
    }

    func testDictionarySuggestionFilterSettingsSanitizedMigratesLegacyDefaultPrompt() {
        let legacyPrompt = """
        You are building a personal dictionary for a speech-to-text app.

        Output: Structured list of recommended terms
        One term per line
        Return null if no worthy terms
        """

        XCTAssertEqual(
            DictionarySuggestionFilterSettings(prompt: legacyPrompt, batchSize: 12, maxCandidatesPerBatch: 12)
                .sanitized()
                .prompt,
            DictionarySuggestionFilterSettings.defaultPrompt
        )
    }

    func testDictionarySuggestionDefaultPromptFollowsInterfaceLanguage() {
        let englishPrompt = DictionarySuggestionFilterSettings.defaultPrompt(language: .english)
        let chinesePrompt = DictionarySuggestionFilterSettings.defaultPrompt(language: .chineseSimplified)
        let japanesePrompt = DictionarySuggestionFilterSettings.defaultPrompt(language: .japanese)

        XCTAssertTrue(englishPrompt.contains("You are building a personal dictionary"))
        XCTAssertTrue(chinesePrompt.contains("你正在为一款语音转文字应用构建个人词典"))
        XCTAssertTrue(japanesePrompt.contains("音声文字起こしアプリ向けの個人辞書"))
    }

    func testDictionarySuggestionFilterSettingsSanitizedReplacesKnownDefaultWithCurrentLanguage() {
        let englishPrompt = DictionarySuggestionFilterSettings.defaultPrompt(language: .english)
        let japanesePrompt = DictionarySuggestionFilterSettings.defaultPrompt(language: .japanese)

        XCTAssertEqual(
            DictionarySuggestionFilterSettings.sanitizedPrompt(
                englishPrompt,
                language: .japanese
            ),
            japanesePrompt
        )
    }

    func testDictionarySuggestionFilterSettingsCanonicalStoredPromptClearsLocalizedDefault() {
        XCTAssertEqual(
            DictionarySuggestionFilterSettings.canonicalStoredPrompt(
                DictionarySuggestionFilterSettings.defaultPrompt(language: .chineseSimplified)
            ),
            ""
        )
    }

    func testDictionarySuggestionDefaultPromptTightensTermSelectionRules() {
        let prompt = DictionarySuggestionFilterSettings.defaultPrompt(language: .english)

        XCTAssertTrue(prompt.contains("ASR mistakes"))
        XCTAssertTrue(prompt.contains("mixed-language speech"))
        XCTAssertTrue(prompt.contains("must not exceed 6 words"))
        XCTAssertTrue(prompt.contains("must not exceed 6 characters"))
        XCTAssertTrue(prompt.contains("JSON array"))
        XCTAssertTrue(prompt.contains("{\"term\": \"accepted term\"}"))
        XCTAssertTrue(prompt.contains("Return []"))
        XCTAssertTrue(prompt.contains("Other frequently used languages"))
        XCTAssertTrue(prompt.contains("secondary language"))
        XCTAssertTrue(prompt.contains("If a word would be familiar to most ordinary speakers of that language, exclude it"))
        XCTAssertTrue(prompt.contains("Chinese, English, Japanese, Korean, Thai"))
        XCTAssertTrue(prompt.contains("Well-known cities, countries"))
        XCTAssertTrue(prompt.contains("Three Filtering Principles"))
        XCTAssertTrue(prompt.contains("Common vocabulary never belongs in the dictionary"))
        XCTAssertTrue(prompt.contains("Context-only items do not belong in the dictionary"))
        XCTAssertTrue(prompt.contains("stable correction targets"))
        XCTAssertTrue(prompt.contains("our rule"))
        XCTAssertTrue(prompt.contains("flight"))
        XCTAssertTrue(prompt.contains("train service"))
        XCTAssertTrue(prompt.contains("token"))
        XCTAssertTrue(prompt.contains("MU5735"))
        XCTAssertNil(
            prompt.range(
                of: #"[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]"#,
                options: .regularExpression
            )
        )
    }

    func testDictionaryHistoryScanPromptLanguageSupportBuildsOtherLanguagesPromptValue() {
        XCTAssertEqual(
            DictionaryHistoryScanPromptLanguageSupport.otherLanguagesPromptValue(
                from: ["zh-hans", "en", "ja"]
            ),
            "English, Japanese"
        )
        XCTAssertEqual(
            DictionaryHistoryScanPromptLanguageSupport.otherLanguagesPromptValue(
                from: ["zh-hans"]
            ),
            "None"
        )
    }

    func testDictionaryHistoryScanCandidateValidatorRejectsLongOrNoisyTerms() {
        XCTAssertTrue(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "OpenAI"))
        XCTAssertTrue(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "旧金山"))
        XCTAssertTrue(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "MCP"))

        XCTAssertFalse(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "this is a very long generic transcript phrase"))
        XCTAssertFalse(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "今天我们要开会讨论一下"))
        XCTAssertFalse(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "20260413"))
        XCTAssertFalse(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "wrong term, maybe"))
        XCTAssertFalse(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "Company"))
        XCTAssertFalse(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "token"))
        XCTAssertFalse(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "我们的规则"))
        XCTAssertFalse(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "our rule"))
        XCTAssertFalse(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "航班"))
        XCTAssertFalse(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "车次"))
        XCTAssertTrue(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "北京"))
        XCTAssertTrue(DictionaryHistoryScanCandidateValidator.shouldAccept(term: "大同"))
    }

    func testDictionaryHistoryScanCandidateValidatorRejectsTravelRouteTermsFromContext() {
        let sample = "北京到大同今天的航班和车次有哪些？有没有 K130 航班？"

        XCTAssertFalse(
            DictionaryHistoryScanCandidateValidator.shouldAccept(
                term: "K130",
                evidenceSample: sample
            )
        )
        XCTAssertFalse(
            DictionaryHistoryScanCandidateValidator.shouldAccept(
                term: "北京",
                evidenceSample: sample
            )
        )
        XCTAssertFalse(
            DictionaryHistoryScanCandidateValidator.shouldAccept(
                term: "大同",
                evidenceSample: sample
            )
        )
        XCTAssertTrue(
            DictionaryHistoryScanCandidateValidator.shouldAccept(
                term: "OpenAI",
                evidenceSample: "我今天要给 OpenAI 的接口做联调。"
            )
        )
    }

    func testDictionaryHistoryScanResponseParserAcceptsWrappedJSONArray() throws {
        let response = """
        Here are the filtered terms:
        ```json
        [{"term":"OpenAI"},{"term":"OpenAI"},{"term":"MCP"}]
        ```
        """

        XCTAssertEqual(
            try DictionaryHistoryScanResponseParser.parseTerms(from: response),
            ["OpenAI", "MCP"]
        )
    }

    func testDictionaryHistoryScanResponseParserRejectsUnexpectedItemShape() {
        let response = """
        [{"term":"OpenAI","reason":"common company name"}]
        """

        XCTAssertEqual(
            try DictionaryHistoryScanResponseParser.parseTerms(from: response),
            ["OpenAI"]
        )
    }

    func testDictionaryHistoryScanResponseParserRejectsPlainTextResponse() {
        XCTAssertThrowsError(
            try DictionaryHistoryScanResponseParser.parseTerms(from: "新次元 词源数据")
        )
    }

    func testDictionaryHistoryScanResponseParserExtractsJSONArrayInsideWrapperObject() {
        let response = """
        {"terms":[{"term":"OpenAI"},{"term":"Claude"}]}
        """

        XCTAssertEqual(
            try DictionaryHistoryScanResponseParser.parseTerms(from: response),
            ["OpenAI", "Claude"]
        )
    }

    func testDictionaryHistoryScanResponseParserAcceptsWrappedStringArrayPayload() throws {
        let response = """
        Sure, here are the extracted terms:
        ```json
        {"terms":["OpenAI","MCP","OpenAI"]}
        ```
        These should be enough.
        """

        XCTAssertEqual(
            try DictionaryHistoryScanResponseParser.parseTerms(from: response),
            ["OpenAI", "MCP"]
        )
    }

    func testDictionaryHistoryScanResponseParserAcceptsTopLevelStringArray() throws {
        let response = """
        ["OpenAI", "MCP", "OpenAI"]
        """

        XCTAssertEqual(
            try DictionaryHistoryScanResponseParser.parseTerms(from: response),
            ["OpenAI", "MCP"]
        )
    }

    func testDictionaryHistoryScanResponseParserAcceptsCommonWrapperKeys() throws {
        for key in ["items", "results", "candidates", "data"] {
            let response = """
            {"\(key)":[{"term":"OpenAI"},{"term":"MCP"}]}
            """

            XCTAssertEqual(
                try DictionaryHistoryScanResponseParser.parseTerms(from: response),
                ["OpenAI", "MCP"],
                "Failed for wrapper key \(key)"
            )
        }
    }

    func testDictionaryHistoryScanResponseParserRejectsBlankTermsInsideStringArray() {
        let response = """
        {"terms":["OpenAI","   "]}
        """

        XCTAssertThrowsError(try DictionaryHistoryScanResponseParser.parseTerms(from: response))
    }

    func testDictionaryHistoryScanResponseParserAcceptsLegacyLineBasedTerms() throws {
        let response = """
        Recommended terms:
        1. OpenAI
        2. MCP
        """

        XCTAssertEqual(
            try DictionaryHistoryScanResponseParser.parseTerms(from: response),
            ["OpenAI", "MCP"]
        )
    }

    func testDictionaryHistoryScanResponseParserAcceptsSingleLegacyTermLine() throws {
        XCTAssertEqual(
            try DictionaryHistoryScanResponseParser.parseTerms(from: "OpenAI"),
            ["OpenAI"]
        )
    }

    func testDictionaryHistoryScanResponseParserAcceptsLegacyNullResponse() throws {
        XCTAssertEqual(
            try DictionaryHistoryScanResponseParser.parseTerms(from: "null"),
            []
        )
    }

    func testDictionaryHistoryScanResponseParserFiltersRejectedJSONArrayItems() throws {
        let response = """
        [
          {"term":"OpenAI"},
          {"term":"this is a very long generic transcript phrase"},
          {"term":"今天我们要开会讨论一下"},
          {"term":"MCP"}
        ]
        """

        XCTAssertEqual(
            try DictionaryHistoryScanResponseParser.parseTerms(from: response),
            ["OpenAI", "MCP"]
        )
    }

    func testDictionaryHistoryScanResponseParserNormalizesAcceptedTermsDirectly() {
        XCTAssertEqual(
            DictionaryHistoryScanResponseParser.normalizeAcceptedTerms(
                from: ["OpenAI", "OpenAI", "今天我们要开会讨论一下", "MCP"]
            ),
            ["OpenAI", "MCP"]
        )
    }

    func testDictionaryHistoryScanResponsesSchemaUsesStrictTopLevelObject() throws {
        let payload = DictionaryHistoryScanResponseParser.responsesTextFormatPayload()
        let format = try XCTUnwrap(payload["format"] as? [String: Any])
        let schema = try XCTUnwrap(format["schema"] as? [String: Any])
        let schemaProperties = try XCTUnwrap(schema["properties"] as? [String: Any])
        let terms = try XCTUnwrap(schemaProperties["terms"] as? [String: Any])
        let items = try XCTUnwrap(terms["items"] as? [String: Any])
        let itemProperties = try XCTUnwrap(items["properties"] as? [String: Any])

        XCTAssertEqual(format["type"] as? String, "json_schema")
        XCTAssertEqual(format["strict"] as? Bool, true)
        XCTAssertEqual(schema["type"] as? String, "object")
        XCTAssertEqual(schema["additionalProperties"] as? Bool, false)
        XCTAssertEqual(schema["required"] as? [String], ["terms"])
        XCTAssertEqual(terms["type"] as? String, "array")
        XCTAssertEqual(items["type"] as? String, "object")
        XCTAssertEqual(items["additionalProperties"] as? Bool, false)
        XCTAssertNotNil(itemProperties["term"])
        XCTAssertEqual(items["required"] as? [String], ["term"])
    }

    func testOnboardingStepStatusResolverMatchesExpectedRules() {
        let readySnapshot = OnboardingStepStatusSnapshot(
            hasModelIssues: false,
            hasRecordingMicrophone: true,
            hasRecordingPermissions: true,
            hasRewriteIssues: false,
            appEnhancementEnabled: true
        )
        let blockedSnapshot = OnboardingStepStatusSnapshot(
            hasModelIssues: true,
            hasRecordingMicrophone: false,
            hasRecordingPermissions: false,
            hasRewriteIssues: true,
            appEnhancementEnabled: false
        )

        XCTAssertEqual(OnboardingStepStatusResolver.resolve(step: .language, snapshot: blockedSnapshot), .ready)
        XCTAssertEqual(OnboardingStepStatusResolver.resolve(step: .model, snapshot: blockedSnapshot), .needsSetup)
        XCTAssertEqual(OnboardingStepStatusResolver.resolve(step: .transcription, snapshot: blockedSnapshot), .needsSetup)
        XCTAssertEqual(OnboardingStepStatusResolver.resolve(step: .translation, snapshot: blockedSnapshot), .ready)
        XCTAssertEqual(OnboardingStepStatusResolver.resolve(step: .rewrite, snapshot: blockedSnapshot), .needsSetup)
        XCTAssertEqual(OnboardingStepStatusResolver.resolve(step: .appEnhancement, snapshot: blockedSnapshot), .ready)
        XCTAssertEqual(OnboardingStepStatusResolver.resolve(step: .finish, snapshot: blockedSnapshot), .done)
    }

    func testVisibleTabsHideAppEnhancementWhenFeatureDisabled() {
        XCTAssertFalse(SettingsTab.visibleTabs(appEnhancementEnabled: false).contains(.appEnhancement))
        XCTAssertFalse(SettingsTab.visibleTabs(appEnhancementEnabled: true).contains(.appEnhancement))
        XCTAssertTrue(SettingsTab.visibleTabs(appEnhancementEnabled: true).contains(.feature))
    }

    func testFeatureVisibleTabsHideAppEnhancementWhenDisabled() {
        XCTAssertFalse(FeatureSettingsTab.visibleTabs(appEnhancementEnabled: false, noteEnabled: false).contains(.appEnhancement))
        XCTAssertTrue(FeatureSettingsTab.visibleTabs(appEnhancementEnabled: true, noteEnabled: false).contains(.appEnhancement))
    }

    func testFeatureVisibleTabsOnlyIncludeCurrentFeatureTabs() {
        XCTAssertEqual(
            FeatureSettingsTab.visibleTabs(appEnhancementEnabled: false, noteEnabled: false),
            [.features, .transcription, .translation, .rewrite, .meeting, .files]
        )
        XCTAssertEqual(
            FeatureSettingsTab.visibleTabs(appEnhancementEnabled: true, noteEnabled: true),
            [.features, .transcription, .translation, .rewrite, .appEnhancement, .note, .meeting, .files]
        )
    }

    func testFeatureVisibleTabsRespectNotesAvailability() {
        XCTAssertFalse(FeatureSettingsTab.visibleTabs(appEnhancementEnabled: true, noteEnabled: false).contains(.note))
        XCTAssertTrue(FeatureSettingsTab.visibleTabs(appEnhancementEnabled: true, noteEnabled: true).contains(.note))
    }

    func testFeatureVisibleTabsRespectFilesAvailability() {
        var availability = FeatureAvailabilitySettings.allEnabled
        availability.filesEnabled = false

        XCTAssertFalse(FeatureSettingsTab.visibleTabs(availability: availability).contains(.files))
        XCTAssertTrue(FeatureSettingsTab.visibleTabs(availability: .allEnabled).contains(.files))
        XCTAssertTrue(FeatureAvailabilityCardKind.files.showsToggle)
    }

    func testFeatureMenuPlacesAppEnhancementNoteMeetingAndFilesLastInOrder() {
        XCTAssertEqual(
            Array(FeatureAvailabilityCardKind.allCases.suffix(4)),
            [.appEnhancement, .note, .meeting, .files]
        )
        XCTAssertEqual(
            Array(FeatureSettingsTab.allCases.suffix(4)),
            [.appEnhancement, .note, .meeting, .files]
        )
    }

    func testFeatureVisibleTabsAlwaysIncludeFeaturesAndTranscription() {
        let availability = FeatureAvailabilitySettings(
            translationEnabled: false,
            rewriteEnabled: false,
            notesEnabled: false,
            appEnhancementEnabled: false,
            meetingEnabled: false,
            filesEnabled: false
        )
        XCTAssertEqual(
            FeatureSettingsTab.visibleTabs(availability: availability),
            [.features, .transcription]
        )
    }

    func testDefaultFeatureTabIsFeaturesOverview() {
        XCTAssertEqual(
            SettingsNavigationTarget.defaultFeatureTab(for: .feature, section: nil),
            .features
        )
    }

    func testHotkeyShortcutVisibilityOnlyIncludesCurrentFeatureKinds() {
        let suiteName = "SettingsTypesTests.hotkeyVisibility.\(UUID().uuidString)"
        guard let defaults = UserDefaults(suiteName: suiteName) else {
            XCTFail("Expected ephemeral UserDefaults suite")
            return
        }
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }

        var settings = FeatureSettingsStore.deriveFromLegacy(defaults: defaults)
        FeatureSettingsStore.save(settings, defaults: defaults)
        XCTAssertEqual(
            HotkeyShortcutVisibility.visibleKinds(defaults: defaults),
            [.transcription, .note, .translation, .rewrite, .meeting]
        )

        settings.availability.translationEnabled = false
        settings.availability.rewriteEnabled = false
        settings.availability.notesEnabled = false
        settings.availability.meetingEnabled = false
        FeatureSettingsStore.save(settings, defaults: defaults)
        XCTAssertEqual(
            HotkeyShortcutVisibility.visibleKinds(defaults: defaults),
            [.transcription]
        )
    }

    func testFeatureNavigationTargetMapsAppBranchSectionToFeatureMode() {
        let target = SettingsNavigationTarget(tab: .feature, section: .appBranchGroups)

        XCTAssertEqual(target.tab, .feature)
        XCTAssertEqual(target.featureTab, .appEnhancement)
    }

    func testHistoryNavigationTargetRoundTripsFilterThroughUserInfo() {
        let target = SettingsNavigationTarget(
            tab: .history,
            section: .historyEntries,
            historyFilter: .note
        )
        let notification = Notification(name: .voxtSettingsNavigate, object: nil, userInfo: target.userInfo)

        let parsedTarget = SettingsNavigationTarget(notification: notification)

        XCTAssertEqual(parsedTarget?.tab, .history)
        XCTAssertEqual(parsedTarget?.section, .historyEntries)
        XCTAssertEqual(parsedTarget?.historyFilter, .note)
    }

    func testModelNavigationTargetRoundTripsConfigurationSelection() {
        let selectionID = FeatureModelSelectionID.remoteLLM(.openAI)
        let target = SettingsNavigationTarget(tab: .model, modelSelectionID: selectionID)
        let notification = Notification(name: .voxtSettingsNavigate, object: nil, userInfo: target.userInfo)

        let parsedTarget = SettingsNavigationTarget(notification: notification)

        XCTAssertEqual(parsedTarget?.tab, .model)
        XCTAssertEqual(parsedTarget?.modelSelectionID, selectionID)
    }

    func testModelNavigationTargetRoundTripsStorageAuthorizationRequest() {
        let target = SettingsNavigationTarget(
            tab: .model,
            requestsModelStorageAuthorization: true
        )
        let notification = Notification(name: .voxtSettingsNavigate, object: nil, userInfo: target.userInfo)

        let parsedTarget = SettingsNavigationTarget(notification: notification)

        XCTAssertEqual(parsedTarget?.tab, .model)
        XCTAssertEqual(parsedTarget?.requestsModelStorageAuthorization, true)
    }

    func testPermissionRequirementResolverAggregatesFeatureSelections() {
        let context = SettingsPermissionRequirementContext(
            selectedEngine: .mlxAudio,
            featureSettings: FeatureSettings(
                transcription: .init(
                    asrSelectionID: .mlx(MLXModelManager.defaultModelRepo),
                    llmEnabled: false,
                    llmSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                    prompt: AppPreferenceKey.defaultEnhancementPrompt
                ),
                translation: .init(
                    asrSelectionID: .dictation,
                    modelSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                    targetLanguageRawValue: TranslationTargetLanguage.english.rawValue,
                    prompt: AppPreferenceKey.defaultTranslationPrompt
                ),
                rewrite: .init(
                    asrSelectionID: .mlx(MLXModelManager.defaultModelRepo),
                    llmSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                    prompt: AppPreferenceKey.defaultRewritePrompt,
                    appEnhancementEnabled: false
                )
            )
        )

        let permissions = SettingsPermissionRequirementResolver.requiredPermissions(context: context)

        XCTAssertTrue(permissions.contains(.speechRecognition))
        XCTAssertEqual(permissions, [.microphone, .accessibility, .speechRecognition, .reminders])
    }

    func testVoiceEndCommandPresetResolvesBuiltInCommands() {
        XCTAssertEqual(VoiceEndCommandPreset.over.title, "over")
        XCTAssertEqual(VoiceEndCommandPreset.over.resolvedCommand, "over")

        XCTAssertEqual(VoiceEndCommandPreset.end.title, "end")
        XCTAssertEqual(VoiceEndCommandPreset.end.resolvedCommand, "end")

        XCTAssertEqual(VoiceEndCommandPreset.wanBi.title, "完毕")
        XCTAssertEqual(VoiceEndCommandPreset.wanBi.resolvedCommand, "完毕")

        XCTAssertEqual(VoiceEndCommandPreset.haoLe.title, "好了")
        XCTAssertEqual(VoiceEndCommandPreset.haoLe.resolvedCommand, "好了")

        XCTAssertNil(VoiceEndCommandPreset.custom.resolvedCommand)
    }
}
