// FeatureModelCatalogBuilderTests.swift
// Provides Feature Model Catalog Builder Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class FeatureModelCatalogBuilderTests: XCTestCase {
    func testAppleIntelligenceEntryRemainsVisibleWhenUnavailable() throws {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(),
            appleIntelligenceAvailability: .unavailable(.modelNotReady)
        )

        let entry = try XCTUnwrap(
            builder.entries(for: .transcriptionLLM)
                .first(where: { $0.selectionID == .appleIntelligence })
        )

        XCTAssertFalse(entry.isSelectable)
        XCTAssertEqual(entry.statusText, AppLocalization.localizedString("Unavailable"))
        XCTAssertEqual(
            entry.disabledReason,
            AppLocalization.localizedString("Apple Intelligence is still preparing its on-device model. Try again shortly.")
        )
    }

    func testAppleIntelligenceEntryIsSelectableWhenAvailable() throws {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(),
            appleIntelligenceAvailability: .available
        )

        let entry = try XCTUnwrap(
            builder.entries(for: .transcriptionLLM)
                .first(where: { $0.selectionID == .appleIntelligence })
        )

        XCTAssertTrue(entry.isSelectable)
        XCTAssertNil(entry.disabledReason)
    }

    func testAppleIntelligenceEntryRemainsHiddenBeforeMacOS26() {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(),
            appleIntelligenceAvailability: .unsupportedOS
        )

        XCTAssertFalse(
            builder.entries(for: .transcriptionLLM)
                .contains(where: { $0.selectionID == .appleIntelligence })
        )
    }

    func testTranslationEntriesDoNotExposeWhisperDirectTranslate() throws {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(
                translationASR: .mlx(MLXModelManager.defaultModelRepo),
                translationModel: .remoteLLM(.openAI),
                translationTarget: .english
            )
        )

        XCTAssertNil(
            builder.entries(for: .translationModel)
                .first(where: { $0.selectionID.rawValue == "whisper-direct-translate" })
        )
    }

    func testLegacyWhisperTranslationSelectionIsNotDisplayed() throws {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(
                translationASR: .mlx(MLXWhisperMigrationSupport.defaultRepo),
                translationModel: .remoteLLM(.openAI),
                translationTarget: .japanese
            )
        )

        XCTAssertNil(
            builder.entries(for: .translationModel)
                .first(where: { $0.selectionID.rawValue == "whisper-direct-translate" })
        )
    }

    func testMeetingASREntriesDoNotExposeLegacyWhisperModels() throws {
        let builder = makeBuilder(featureSettings: makeFeatureSettings())

        let whisperEntries = builder.entries(for: .meetingASR)
            .filter { $0.selectionID.rawValue.hasPrefix("whisper:") }

        XCTAssertTrue(whisperEntries.isEmpty)
    }

    func testConfiguredRemoteEntriesExposeUsageAndSelectionSummary() throws {
        let remoteASRConfigurations = RemoteModelConfigurationStore.saveConfigurations([
            RemoteASRProvider.aliyunBailianASR.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteASRProvider.aliyunBailianASR.rawValue,
                model: "fun-asr-realtime",
                endpoint: "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
                apiKey: "token"
            )
        ])
        let remoteLLMConfigurations = RemoteModelConfigurationStore.saveConfigurations([
            RemoteLLMProvider.openAI.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.openAI.rawValue,
                model: "gpt-5.2",
                endpoint: "https://example.com/v1",
                apiKey: "secret"
            )
        ])

        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(
                translationASR: .remoteASR(.aliyunBailianASR),
                translationModel: .remoteLLM(.openAI)
            ),
            remoteASRConfigurationsRaw: remoteASRConfigurations,
            remoteLLMConfigurationsRaw: remoteLLMConfigurations
        )

        let translationASREntry = try XCTUnwrap(
            builder.entries(for: .translationASR)
                .first(where: { $0.selectionID == .remoteASR(.aliyunBailianASR) })
        )
        let translationLLMEntry = try XCTUnwrap(
            builder.entries(for: .translationModel)
                .first(where: { $0.selectionID == .remoteLLM(.openAI) })
        )

        XCTAssertTrue(translationASREntry.isSelectable)
        XCTAssertTrue(translationASREntry.filterTags.contains(AppLocalization.localizedString("Configured")))
        XCTAssertTrue(translationASREntry.usageLocations.contains(AppLocalization.localizedString("Translation")))

        XCTAssertTrue(translationLLMEntry.isSelectable)
        XCTAssertTrue(translationLLMEntry.displayTags.contains(AppLocalization.localizedString("Configured")))
        XCTAssertTrue(translationLLMEntry.usageLocations.contains(AppLocalization.localizedString("Translation")))
        XCTAssertEqual(builder.llmSelectionSummary(.remoteLLM(.openAI)), "OpenAI · gpt-5.2")
        XCTAssertEqual(
            builder.asrSelectionSummary(.remoteASR(.aliyunBailianASR)),
            "\(RemoteASRProvider.aliyunBailianASR.title) · fun-asr-realtime"
        )
    }

    func testDoubaoEntryWithOnlyAppIDRemainsUnconfigured() throws {
        let remoteASRConfigurations = RemoteModelConfigurationStore.saveConfigurations([
            RemoteASRProvider.doubaoASR.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                model: DoubaoASRConfiguration.modelV2,
                appID: "doubao-app",
                accessToken: ""
            )
        ])
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(translationASR: .remoteASR(.doubaoASR)),
            remoteASRConfigurationsRaw: remoteASRConfigurations
        )

        let entry = try XCTUnwrap(
            builder.entries(for: .translationASR)
                .first(where: { $0.selectionID == .remoteASR(.doubaoASR) })
        )

        XCTAssertFalse(entry.isSelectable)
        XCTAssertEqual(entry.statusText, AppLocalization.localizedString("Not configured"))
        XCTAssertFalse(entry.filterTags.contains(AppLocalization.localizedString("Configured")))
    }

    func testUnconfiguredRemoteLLMEntryRemainsNotConfiguredInSelector() throws {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(translationModel: .remoteLLM(.openAI))
        )

        let entry = try XCTUnwrap(
            builder.entries(for: .translationModel)
                .first(where: { $0.selectionID == .remoteLLM(.openAI) })
        )

        XCTAssertFalse(entry.isSelectable)
        XCTAssertEqual(entry.statusText, AppLocalization.localizedString("Not configured"))
        XCTAssertFalse(entry.filterTags.contains(AppLocalization.localizedString("Configured")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Configured")))
    }

    func testOllamaRemoteEntryIsSelectableWithoutAPIKey() throws {
        let remoteLLMConfigurations = RemoteModelConfigurationStore.saveConfigurations([
            RemoteLLMProvider.ollama.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.ollama.rawValue,
                model: "qwen3",
                endpoint: "http://127.0.0.1:11434/api/chat"
            )
        ])

        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(translationModel: .remoteLLM(.ollama)),
            remoteLLMConfigurationsRaw: remoteLLMConfigurations
        )

        let entry = try XCTUnwrap(
            builder.entries(for: .translationModel)
                .first(where: { $0.selectionID == .remoteLLM(.ollama) })
        )

        XCTAssertTrue(entry.isSelectable)
        XCTAssertEqual(entry.statusText, AppLocalization.localizedString("Configured"))
        XCTAssertEqual(builder.llmSelectionSummary(.remoteLLM(.ollama)), "Ollama · qwen3")
    }

    func testOMLXRemoteEntryIsSelectableWithoutAPIKey() throws {
        let remoteLLMConfigurations = RemoteModelConfigurationStore.saveConfigurations([
            RemoteLLMProvider.omlx.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.omlx.rawValue,
                model: "qwen3",
                endpoint: "http://localhost:8000/v1"
            )
        ])

        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(translationModel: .remoteLLM(.omlx)),
            remoteLLMConfigurationsRaw: remoteLLMConfigurations
        )

        let entry = try XCTUnwrap(
            builder.entries(for: .translationModel)
                .first(where: { $0.selectionID == .remoteLLM(.omlx) })
        )

        XCTAssertTrue(entry.isSelectable)
        XCTAssertEqual(entry.statusText, AppLocalization.localizedString("Configured"))
        XCTAssertEqual(builder.llmSelectionSummary(.remoteLLM(.omlx)), "oMLX · qwen3")
    }

    func testASRSelectorEntryDisplaysSupportsPrimaryLanguageTag() throws {
        let repo = "mlx-community/Qwen3-ASR-0.6B-4bit"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(repo)),
            primaryUserLanguageCode: "zh-Hans"
        )

        let entry = try XCTUnwrap(
            builder.entries(for: .transcriptionASR)
                .first(where: { $0.selectionID == .mlx(repo) })
        )

        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Supports Primary Language")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Does Not Support Primary Language")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Multilingual")))
    }

    func testParakeetV3SelectorDoesNotClaimUnsupportedChinesePrimaryLanguage() throws {
        let repo = "mlx-community/parakeet-tdt-0.6b-v3"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(repo)),
            primaryUserLanguageCode: "zh-Hans"
        )

        let entry = try XCTUnwrap(
            builder.entries(for: .transcriptionASR)
                .first(where: { $0.selectionID == .mlx(repo) })
        )

        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Does Not Support Primary Language")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Supports Primary Language")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Multilingual")))
    }

    func testParakeetCapabilityDoesNotClaimChinesePrimaryLanguageSupport() {
        let repo = "mlx-community/parakeet-tdt-0.6b-v3"

        XCTAssertFalse(MLXModelCatalog.supportsLanguage("zh", for: repo))
        XCTAssertTrue(MLXModelCatalog.supportsLanguage("de", for: repo))
    }


    func testLLMSelectorUsesCuratedRatingAndTags() throws {
        let repo = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(translationModel: .localLLM(repo))
        )

        let entry = try XCTUnwrap(
            builder.entries(for: .translationModel)
                .first(where: { $0.selectionID == .localLLM(repo) })
        )

        XCTAssertEqual(entry.ratingText, "4.5")
        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Balanced")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Fast")))
    }

    func testLLMSelectorMigratesRetiredLocalLLMFeatureSelection() throws {
        let repo = "mlx-community/gemma-2-2b-it-4bit"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(translationModel: .localLLM(repo))
        )

        XCTAssertFalse(CustomLLMModelManager.availableModels.contains { $0.id == repo })

        let entry = try XCTUnwrap(
            builder.entries(for: .translationModel)
                .first(where: { $0.selectionID == .localLLM(repo) })
        )

        XCTAssertEqual(entry.title, CustomLLMModelCatalog.displayTitle(for: repo))
        XCTAssertTrue(entry.usageLocations.contains(AppLocalization.localizedString("Translation")))
    }


    func testMLXSelectorUsesCuratedRatingAndTags() throws {
        let repo = "mlx-community/parakeet-tdt-0.6b-v3"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(repo))
        )

        let entry = try XCTUnwrap(
            builder.entries(for: .transcriptionASR)
                .first(where: { $0.selectionID == .mlx(repo) })
        )

        XCTAssertEqual(entry.ratingText, "4.3")
        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Fast")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Accurate")))
    }

    func testQwen3SelectorShowsRealtimeTag() throws {
        let repo = "mlx-community/Qwen3-ASR-0.6B-4bit"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(repo))
        )

        let entry = try XCTUnwrap(
            builder.entries(for: .transcriptionASR)
                .first(where: { $0.selectionID == .mlx(repo) })
        )

        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Realtime")))
    }


    func testLegacyWhisperSelectionSummaryUsesMigratedMLXWhisperModel() throws {
        let modelID = "medium"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(MLXWhisperMigrationSupport.repo(forLegacyWhisperModelID: modelID)))
        )

        let repo = MLXWhisperMigrationSupport.repo(forLegacyWhisperModelID: modelID)
        let entry = try XCTUnwrap(
            builder.entries(for: .transcriptionASR)
                .first(where: { $0.selectionID == .mlx(repo) })
        )

        XCTAssertEqual(builder.asrSelectionSummary(.mlx(repo)), MLXModelCatalog.displayTitle(for: repo))
        XCTAssertEqual(entry.ratingText, "4.8")
        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Fast")))
        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Balanced")))
    }

    func testSelectorEntriesShowRecommendedBadgesForTargetedSinglesAndProviders() throws {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(
                transcriptionASR: .mlx("mlx-community/SenseVoiceSmall"),
                translationModel: .remoteLLM(.deepseek)
            )
        )

        let asrEntries = builder.entries(for: .transcriptionASR)
        let llmEntries = builder.entries(for: .translationModel)
        let recommended = AppLocalization.localizedString("Recommended")

        let senseVoice = try XCTUnwrap(
            asrEntries.first(where: { $0.selectionID == .mlx("mlx-community/SenseVoiceSmall") })
        )
        let moss = try XCTUnwrap(
            asrEntries.first(where: { $0.selectionID == .mlx("OpenMOSS-Team/MOSS-Transcribe-Diarize") })
        )
        let doubaoASR = try XCTUnwrap(
            asrEntries.first(where: { $0.selectionID == .remoteASR(.doubaoASR) })
        )
        let stepFunASR = try XCTUnwrap(
            asrEntries.first(where: { $0.selectionID == .remoteASR(.stepFunASR) })
        )
        let deepSeek = try XCTUnwrap(
            llmEntries.first(where: { $0.selectionID == .remoteLLM(.deepseek) })
        )
        let ollama = try XCTUnwrap(
            llmEntries.first(where: { $0.selectionID == .remoteLLM(.ollama) })
        )
        let omlx = try XCTUnwrap(
            llmEntries.first(where: { $0.selectionID == .remoteLLM(.omlx) })
        )
        let aliyun = try XCTUnwrap(
            llmEntries.first(where: { $0.selectionID == .remoteLLM(.aliyunBailian) })
        )

        XCTAssertEqual(senseVoice.badgeText, recommended)
        XCTAssertEqual(moss.badgeText, recommended)
        XCTAssertEqual(doubaoASR.badgeText, recommended)
        XCTAssertEqual(stepFunASR.badgeText, recommended)
        XCTAssertEqual(deepSeek.badgeText, recommended)
        XCTAssertEqual(ollama.badgeText, recommended)
        XCTAssertEqual(omlx.badgeText, recommended)
        XCTAssertEqual(aliyun.badgeText, recommended)
    }

    func testSelectorGroupedFamiliesShowRecommendedBadgesForWhisperQwenASRAndQwenLLM() throws {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(
                transcriptionASR: .mlx("mlx-community/whisper-large-v3-turbo"),
                translationModel: .localLLM("mlx-community/gemma-4-e2b-it-4bit")
            )
        )

        let asrGroups = LocalModelSeriesGrouping.featureSelectorItems(
            from: builder.entries(for: .transcriptionASR),
            selectedID: .mlx("mlx-community/whisper-large-v3-turbo")
        )
        let llmGroups = LocalModelSeriesGrouping.featureSelectorItems(
            from: builder.entries(for: .translationModel),
            selectedID: .localLLM("mlx-community/gemma-4-e2b-it-4bit")
        )
        let recommended = AppLocalization.localizedString("Recommended")

        let whisperGroup = try XCTUnwrap(
            asrGroups.compactMap { item -> FeatureModelSelectorGroupSection? in
                guard case .group(let group) = item, group.title == "Whisper" else { return nil }
                return group
            }.first
        )
        let qwenGroup = try XCTUnwrap(
            asrGroups.compactMap { item -> FeatureModelSelectorGroupSection? in
                guard case .group(let group) = item, group.title == "Qwen3" else { return nil }
                return group
            }.first
        )
        let qwenLLMGroup = try XCTUnwrap(
            llmGroups.compactMap { item -> FeatureModelSelectorGroupSection? in
                guard case .group(let group) = item, group.title == "Qwen" else { return nil }
                return group
            }.first
        )

        XCTAssertEqual(whisperGroup.badgeText, recommended)
        XCTAssertEqual(whisperGroup.entries.map(\.groupedVariantTitle), ["Large v3 Turbo", "Large v3", "Small"])
        XCTAssertEqual(qwenGroup.badgeText, recommended)
        XCTAssertEqual(qwenLLMGroup.badgeText, recommended)
    }

    private func makeBuilder(
        mlxModelManager: MLXModelManager? = nil,
        featureSettings: FeatureSettings,
        remoteASRConfigurationsRaw: String = "",
        remoteLLMConfigurationsRaw: String = "",
        primaryUserLanguageCode: String? = "en",
        appleIntelligenceAvailability: AppleIntelligenceAvailability = .available
    ) -> FeatureModelCatalogBuilder {
        FeatureModelCatalogBuilder(
            mlxModelManager: mlxModelManager ?? TestModelManagers.mlx,
            customLLMManager: TestModelManagers.customLLM,
            ggufTranslationModelManager: TestModelManagers.gguf,
            featureSettings: featureSettings,
            remoteASRProviderConfigurationsRaw: remoteASRConfigurationsRaw,
            remoteLLMProviderConfigurationsRaw: remoteLLMConfigurationsRaw,
            appleIntelligenceAvailability: appleIntelligenceAvailability,
            primaryUserLanguageCode: primaryUserLanguageCode
        )
    }

    private func makeFeatureSettings(
        transcriptionASR: FeatureModelSelectionID? = nil,
        transcriptionLLM: FeatureModelSelectionID? = nil,
        translationASR: FeatureModelSelectionID? = nil,
        translationModel: FeatureModelSelectionID? = nil,
        translationTarget: TranslationTargetLanguage = .english,
        rewriteASR: FeatureModelSelectionID? = nil,
        rewriteLLM: FeatureModelSelectionID? = nil,
        meetingSummary: FeatureModelSelectionID? = nil
    ) -> FeatureSettings {
        let defaultLLM = FeatureModelSelectionID.localLLM(CustomLLMModelManager.defaultModelRepo)
        let transcriptionASR = transcriptionASR ?? .dictation
        let transcriptionLLM = transcriptionLLM ?? defaultLLM
        let translationASR = translationASR ?? .dictation
        let translationModel = translationModel ?? defaultLLM
        let rewriteASR = rewriteASR ?? .dictation
        let rewriteLLM = rewriteLLM ?? defaultLLM
        return FeatureSettings(
            transcription: .init(asrSelectionID: transcriptionASR, llmEnabled: true,
                                 llmSelectionID: transcriptionLLM,
                                 prompt: AppPreferenceKey.defaultEnhancementPrompt),
            translation: .init(asrSelectionID: translationASR, modelSelectionID: translationModel,
                               targetLanguageRawValue: translationTarget.rawValue,
                               prompt: AppPreferenceKey.defaultTranslationPrompt),
            rewrite: .init(asrSelectionID: rewriteASR, llmSelectionID: rewriteLLM,
                           prompt: AppPreferenceKey.defaultRewritePrompt, appEnhancementEnabled: true),
            meeting: .init(asrSelectionID: transcriptionASR, summaryModelSelectionID: meetingSummary ?? transcriptionLLM,
                           summaryPrompt: "", summaryAutoGenerate: true, realtimeTranslateEnabled: false,
                           realtimeTargetLanguageRawValue: "", hideOverlayFromScreenSharing: false)
        )
    }


}

@MainActor
private enum TestModelManagers {
    static let mlx = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
    static let customLLM = CustomLLMModelManager(modelRepo: CustomLLMModelManager.defaultModelRepo)
    static let gguf = GGUFTranslationModelManager(modelID: .hyMT2Q4KM)
}
