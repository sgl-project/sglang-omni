// ModelCatalogBuilderTests.swift
// Provides Model Catalog Builder Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class ModelCatalogBuilderTests: XCTestCase {
    func testLLMCatalogIncludesUnavailableAppleIntelligenceEntry() throws {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(),
            appleIntelligenceAvailability: .unavailable(.deviceNotEligible)
        )

        let entry = try XCTUnwrap(
            builder.llmEntries().first(where: { $0.id == "apple-intelligence" })
        )

        XCTAssertEqual(entry.title, AppLocalization.localizedString("Apple Intelligence"))
        XCTAssertEqual(entry.engine, AppLocalization.localizedString("Apple"))
        XCTAssertEqual(
            entry.statusText,
            AppLocalization.localizedString("Apple Intelligence is not available for this Mac or region.")
        )
        XCTAssertNil(entry.primaryAction)
    }

    func testLLMCatalogHidesAppleIntelligenceBeforeMacOS26() {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(),
            appleIntelligenceAvailability: .unsupportedOS
        )

        XCTAssertFalse(builder.llmEntries().contains(where: { $0.id == "apple-intelligence" }))
    }


    func testModelCatalogTagPriorityDoesNotExposeMultilingualFilter() {
        XCTAssertFalse(ModelCatalogTag.priority.contains(AppLocalization.localizedString("Multilingual")))
    }

    func testASRCatalogIncludesDirectDictationSettingsEntry() throws {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(
                transcriptionASR: .dictation
            )
        )

        let directDictation = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == FeatureModelSelectionID.dictation.rawValue })
        )

        XCTAssertEqual(directDictation.engine, AppLocalization.localizedString("System ASR"))
        XCTAssertEqual(directDictation.primaryAction?.title, AppLocalization.localizedString("Settings"))
        XCTAssertTrue(directDictation.usageLocations.contains(AppLocalization.localizedString("Transcription")))
        XCTAssertTrue(directDictation.displayTags.contains(AppLocalization.localizedString("In Use")))
}
    func testConfiguredRemoteASREntryShowsNeedsSetupBadgeWhenProviderHasConfigurationIssue() throws {
        let remoteASRConfigurations: [String: RemoteProviderConfiguration] = [
            RemoteASRProvider.aliyunBailianASR.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteASRProvider.aliyunBailianASR.rawValue,
                model: "fun-asr-realtime",
                endpoint: "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
                apiKey: "token"
            )
        ]
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(
                transcriptionASR: .remoteASR(.aliyunBailianASR)
            ),
            remoteASRConfigurations: remoteASRConfigurations,
            hasIssue: { scope in
                if case .remoteASRProvider(.aliyunBailianASR) = scope {
                    return true
                }
                return false
            }
        )

        let entry = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "remote-asr:\(RemoteASRProvider.aliyunBailianASR.rawValue)" })
        )

        XCTAssertEqual(entry.badgeText, AppLocalization.localizedString("Needs Setup"))
        XCTAssertTrue(entry.filterTags.contains(AppLocalization.localizedString("Configured")))
        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("In Use")))
        XCTAssertEqual(entry.primaryAction?.title, AppLocalization.localizedString("Configure"))
    }

    func testConfiguredRemoteLLMEntryShowsConfiguredTagAndUsage() throws {
        let remoteLLMConfigurations: [String: RemoteProviderConfiguration] = [
            RemoteLLMProvider.openAI.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.openAI.rawValue,
                model: "gpt-5.2",
                endpoint: "https://example.com/v1",
                apiKey: "secret"
            )
        ]
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(
                translationModel: .remoteLLM(.openAI)
            ),
            remoteLLMConfigurations: remoteLLMConfigurations
        )

        let entry = try XCTUnwrap(
            builder.llmEntries().first(where: { $0.id == "remote-llm:\(RemoteLLMProvider.openAI.rawValue)" })
        )

        XCTAssertTrue(entry.filterTags.contains(AppLocalization.localizedString("Configured")))
        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("In Use")))
        XCTAssertTrue(entry.usageLocations.contains(AppLocalization.localizedString("Translation")))
        XCTAssertEqual(entry.sizeText, "gpt-5.2")
        XCTAssertEqual(entry.primaryAction?.title, AppLocalization.localizedString("Configure"))
    }

    func testMultilingualMLXModelDisplaysSupportsPrimaryLanguageTag() throws {
        let repo = "mlx-community/Qwen3-ASR-0.6B-4bit"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(repo)),
            primaryUserLanguageCode: "zh-Hans"
        )

        let entry = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "mlx:\(repo)" })
        )

        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Supports Primary Language")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Does Not Support Primary Language")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Multilingual")))
    }

    func testParakeetV3DoesNotClaimUnsupportedChinesePrimaryLanguage() throws {
        let repo = "mlx-community/parakeet-tdt-0.6b-v3"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(repo)),
            primaryUserLanguageCode: "zh-Hans"
        )

        let entry = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "mlx:\(repo)" })
        )

        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Does Not Support Primary Language")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Supports Primary Language")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Multilingual")))
    }

    func testParakeetLanguageTagUsesItsOfficialLanguageList() {
        let repo = "mlx-community/parakeet-tdt-0.6b-v3"
        let chineseBuilder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(repo)),
            primaryUserLanguageCode: "zh-Hans"
        )
        let germanBuilder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(repo)),
            primaryUserLanguageCode: "de"
        )
        let selectionID = FeatureModelSelectionID.mlx(repo)
        let chineseTags = chineseBuilder.catalogDisplayTags(
            base: [AppLocalization.localizedString("Local")],
            requiresConfiguration: false,
            configured: true,
            selectionID: selectionID
        )
        let germanTags = germanBuilder.catalogDisplayTags(
            base: [AppLocalization.localizedString("Local")],
            requiresConfiguration: false,
            configured: true,
            selectionID: selectionID
        )

        XCTAssertTrue(chineseTags.contains(AppLocalization.localizedString("Does Not Support Primary Language")))
        XCTAssertFalse(chineseTags.contains(AppLocalization.localizedString("Supports Primary Language")))
        XCTAssertTrue(germanTags.contains(AppLocalization.localizedString("Supports Primary Language")))
    }

    func testMLXCatalogShowsPauseForDownloadingNonSelectedModel() throws {
        let selectedRepo = "mlx-community/parakeet-tdt-0.6b-v3"
        let downloadingRepo = "mlx-community/Qwen3-ASR-0.6B-4bit"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(selectedRepo)),
            isDownloadingModel: { repo in
                MLXModelManager.canonicalModelRepo(repo) == MLXModelManager.canonicalModelRepo(downloadingRepo)
            }
        )

        let entry = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "mlx:\(downloadingRepo)" })
        )

        XCTAssertEqual(entry.primaryAction?.title, AppLocalization.localizedString("Pause"))
    }

    func testMLXCatalogPauseActionTargetsDownloadingRepo() throws {
        let selectedRepo = "mlx-community/parakeet-tdt-0.6b-v3"
        let downloadingRepo = "mlx-community/Qwen3-ASR-0.6B-4bit"
        var pausedRepo: String?
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(selectedRepo)),
            isDownloadingModel: { repo in
                MLXModelManager.canonicalModelRepo(repo) == MLXModelManager.canonicalModelRepo(downloadingRepo)
            },
            pauseModelDownload: { pausedRepo = $0 }
        )

        let entry = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "mlx:\(downloadingRepo)" })
        )
        let action = try XCTUnwrap(entry.primaryAction)
        action.handler()

        XCTAssertEqual(
            MLXModelManager.canonicalModelRepo(pausedRepo ?? ""),
            MLXModelManager.canonicalModelRepo(downloadingRepo)
        )
    }

    func testMLXCatalogCancelActionTargetsDownloadingRepo() throws {
        let selectedRepo = "mlx-community/parakeet-tdt-0.6b-v3"
        let downloadingRepo = "mlx-community/Qwen3-ASR-0.6B-4bit"
        var cancelledRepo: String?
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(selectedRepo)),
            isDownloadingModel: { repo in
                MLXModelManager.canonicalModelRepo(repo) == MLXModelManager.canonicalModelRepo(downloadingRepo)
            },
            cancelModelDownload: { cancelledRepo = $0 }
        )

        let entry = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "mlx:\(downloadingRepo)" })
        )
        let cancelAction = try XCTUnwrap(
            entry.secondaryActions.first(where: { $0.title == AppLocalization.localizedString("Cancel") })
        )
        cancelAction.handler()

        XCTAssertEqual(
            MLXModelManager.canonicalModelRepo(cancelledRepo ?? ""),
            MLXModelManager.canonicalModelRepo(downloadingRepo)
        )
    }

    func testCustomLLMCatalogShowsPauseForDownloadingNonSelectedModel() throws {
        let selectedRepo = "mlx-community/Qwen3-8B-4bit"
        let downloadingRepo = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(translationModel: .localLLM(selectedRepo)),
            isDownloadingCustomLLM: { repo in
                repo == downloadingRepo
            }
        )

        let entry = try XCTUnwrap(
            builder.llmEntries().first(where: { $0.id == "local-llm:\(downloadingRepo)" })
        )

        XCTAssertEqual(entry.primaryAction?.title, AppLocalization.localizedString("Pause"))
    }

    func testCustomLLMCatalogAllowsInstallingAnotherModelWhileDownloadIsActive() throws {
        let downloadingRepo = "mlx-community/Qwen3.5-4B-4bit"
        let installableRepo = "mlx-community/LFM2-1.2B-4bit"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(translationModel: .localLLM(downloadingRepo)),
            isDownloadingCustomLLM: { repo in
                repo == downloadingRepo
            }
        )

        let entry = try XCTUnwrap(
            builder.llmEntries().first(where: { $0.id == "local-llm:\(installableRepo)" })
        )

        XCTAssertEqual(entry.primaryAction?.title, AppLocalization.localizedString("Install"))
        XCTAssertEqual(entry.primaryAction?.isEnabled, true)
    }


    func testCustomLLMCatalogInstalledModelIncludesConfigureAction() throws {
        let repo = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
        var configuredRepo: String?
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(translationModel: .localLLM(repo)),
            isCustomLLMInstalled: { candidate in
                CustomLLMModelManager.canonicalModelRepo(candidate) == CustomLLMModelManager.canonicalModelRepo(repo)
            },
            configureCustomLLMGeneration: { configuredRepo = $0 }
        )

        let entry = try XCTUnwrap(
            builder.llmEntries().first(where: { $0.id == "local-llm:\(repo)" })
        )
        let action = try XCTUnwrap(
            entry.secondaryActions.first(where: { $0.title == AppLocalization.localizedString("Configure") })
        )
        action.handler()

        XCTAssertEqual(
            CustomLLMModelManager.canonicalModelRepo(configuredRepo ?? ""),
            CustomLLMModelManager.canonicalModelRepo(repo)
        )
    }

    func testCustomLLMCatalogUninstalledModelDoesNotIncludeConfigureAction() throws {
        let repo = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(translationModel: .localLLM(repo)),
            isCustomLLMInstalled: { _ in false }
        )

        let entry = try XCTUnwrap(
            builder.llmEntries().first(where: { $0.id == "local-llm:\(repo)" })
        )

        XCTAssertFalse(entry.secondaryActions.contains(where: { $0.title == AppLocalization.localizedString("Configure") }))
    }

    func testCustomLLMCatalogUsesCuratedRatingAndTags() throws {
        let repo = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(translationModel: .localLLM(repo))
        )

        let entry = try XCTUnwrap(
            builder.llmEntries().first(where: { $0.id == "local-llm:\(repo)" })
        )

        XCTAssertEqual(entry.ratingText, "4.8")
        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Balanced")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Accurate")))
    }


    func testMLXCatalogUsesCuratedRatingAndTags() throws {
        let repo = "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(repo))
        )

        let entry = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "mlx:\(repo)" })
        )

        XCTAssertEqual(entry.ratingText, "4.8")
        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Realtime")))
        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Accurate")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Fast")))
    }

    func testWhisperCatalogUsesCuratedRatingAndTags() throws {
        let repo = "mlx-community/whisper-small-mlx"
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(transcriptionASR: .mlx(repo))
        )

        let entry = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "mlx:\(repo)" })
        )

        XCTAssertEqual(entry.ratingText, "4.5")
        XCTAssertTrue(entry.displayTags.contains(AppLocalization.localizedString("Fast")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Balanced")))
        XCTAssertFalse(entry.displayTags.contains(AppLocalization.localizedString("Accurate")))
    }

    func testCatalogShowsRecommendedBadgesForTargetedSingleEntriesAndProviders() throws {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(
                transcriptionASR: .mlx("mlx-community/SenseVoiceSmall"),
                translationModel: .remoteLLM(.deepseek)
            )
        )

        let senseVoice = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "mlx:mlx-community/SenseVoiceSmall" })
        )
        let moss = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "mlx:OpenMOSS-Team/MOSS-Transcribe-Diarize" })
        )
        let doubaoASR = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "remote-asr:\(RemoteASRProvider.doubaoASR.rawValue)" })
        )
        let stepFunASR = try XCTUnwrap(
            builder.asrEntries().first(where: { $0.id == "remote-asr:\(RemoteASRProvider.stepFunASR.rawValue)" })
        )
        let deepSeek = try XCTUnwrap(
            builder.llmEntries().first(where: { $0.id == "remote-llm:\(RemoteLLMProvider.deepseek.rawValue)" })
        )
        let ollama = try XCTUnwrap(
            builder.llmEntries().first(where: { $0.id == "remote-llm:\(RemoteLLMProvider.ollama.rawValue)" })
        )
        let omlx = try XCTUnwrap(
            builder.llmEntries().first(where: { $0.id == "remote-llm:\(RemoteLLMProvider.omlx.rawValue)" })
        )
        let aliyun = try XCTUnwrap(
            builder.llmEntries().first(where: { $0.id == "remote-llm:\(RemoteLLMProvider.aliyunBailian.rawValue)" })
        )

        let recommended = AppLocalization.localizedString("Recommended")
        XCTAssertEqual(senseVoice.badgeText, recommended)
        XCTAssertEqual(moss.badgeText, recommended)
        XCTAssertEqual(doubaoASR.badgeText, recommended)
        XCTAssertEqual(stepFunASR.badgeText, recommended)
        XCTAssertEqual(deepSeek.badgeText, recommended)
        XCTAssertEqual(ollama.badgeText, recommended)
        XCTAssertEqual(omlx.badgeText, recommended)
        XCTAssertEqual(aliyun.badgeText, recommended)
    }

    func testCatalogShowsRecommendedBadgeForWhisperQwenASRAndQwenLLMGroups() throws {
        let builder = makeBuilder(
            featureSettings: makeFeatureSettings(
                transcriptionASR: .mlx("mlx-community/whisper-large-v3-turbo"),
                translationModel: .localLLM("mlx-community/gemma-4-e2b-it-4bit")
            )
        )

        let asrGroups = LocalModelSeriesGrouping.modelCatalogItems(from: builder.asrEntries())
        let llmGroups = LocalModelSeriesGrouping.modelCatalogItems(from: builder.llmEntries())
        let recommended = AppLocalization.localizedString("Recommended")

        let whisperGroup = try XCTUnwrap(
            asrGroups.compactMap { item -> ModelCatalogGroupSection? in
                guard case .group(let group) = item, group.title == "Whisper" else { return nil }
                return group
            }.first
        )
        let qwenGroup = try XCTUnwrap(
            asrGroups.compactMap { item -> ModelCatalogGroupSection? in
                guard case .group(let group) = item, group.title == "Qwen3" else { return nil }
                return group
            }.first
        )
        let qwenLLMGroup = try XCTUnwrap(
            llmGroups.compactMap { item -> ModelCatalogGroupSection? in
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
        featureSettings: FeatureSettings,
        remoteASRConfigurations: [String: RemoteProviderConfiguration] = [:],
        remoteLLMConfigurations: [String: RemoteProviderConfiguration] = [:],
        primaryUserLanguageCode: String? = "en",
        appleIntelligenceAvailability: AppleIntelligenceAvailability = .available,
        hasIssue: @escaping (ModelConfigurationIssue.Scope) -> Bool = { _ in false },
        isDownloadingModel: @escaping (String) -> Bool = { _ in false },
        isPausedModel: @escaping (String) -> Bool = { _ in false },
        isDownloadingCustomLLM: @escaping (String) -> Bool = { _ in false },
        isPausedCustomLLM: @escaping (String) -> Bool = { _ in false },
        isCustomLLMInstalled: @escaping (String) -> Bool = { _ in false },
        isDownloadingGGUFTranslation: @escaping (GGUFTranslationModelID) -> Bool = { _ in false },
        isPausedGGUFTranslation: @escaping (GGUFTranslationModelID) -> Bool = { _ in false },
        isGGUFTranslationInstalled: @escaping (GGUFTranslationModelID) -> Bool = { _ in false },
        isUninstallingModel: @escaping (String) -> Bool = { _ in false },
        isUninstallingCustomLLM: @escaping (String) -> Bool = { _ in false },
        pauseModelDownload: @escaping (String) -> Void = { _ in },
        cancelModelDownload: @escaping (String) -> Void = { _ in },
        configureCustomLLMGeneration: @escaping (String) -> Void = { _ in }
    ) -> ModelCatalogBuilder {
        let performAction: (LocalModelInstallTarget, LocalModelInstallActionKind) -> Void = { target, action in
            switch (target, action) {
            case let (.mlx(repo), .pause):
                pauseModelDownload(repo)
            case let (.mlx(repo), .cancel):
                cancelModelDownload(repo)
            case let (.customLLM(repo), .configure):
                configureCustomLLMGeneration(repo)
            default:
                break
            }
        }

        let mlxInstallSnapshot: (String) -> LocalModelInstallSnapshot = { repo in
            let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
            let isDownloading = isDownloadingModel(canonicalRepo)
            let isPaused = isPausedModel(canonicalRepo)
            let isUninstalling = isUninstallingModel(canonicalRepo)
            let isInstalled = !isDownloading && !isPaused && !isUninstalling
            let state: LocalModelInstallState
            if isUninstalling { state = .uninstalling }
            else if isDownloading { state = .downloading }
            else if isPaused { state = .paused }
            else if isInstalled { state = .installed }
            else { state = .installable(isEnabled: true) }
            return LocalModelInstallSnapshot(
                target: .mlx(canonicalRepo), state: state, isInstalled: isInstalled,
                isCurrentSelection: featureSettings.transcription.asrSelectionID == .mlx(canonicalRepo),
                statusText: "", badgeText: nil, downloadStatus: nil, canOpenLocation: isInstalled,
                canConfigure: false, configureActionTitle: nil
            )
        }

        let customLLMInstallSnapshot: (String) -> LocalModelInstallSnapshot = { repo in
            let canonicalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
            let isDownloading = isDownloadingCustomLLM(canonicalRepo)
            let isPaused = isPausedCustomLLM(canonicalRepo)
            let isUninstalling = isUninstallingCustomLLM(canonicalRepo)
            let isInstalled = !isDownloading && !isPaused && !isUninstalling && isCustomLLMInstalled(canonicalRepo)
            let state: LocalModelInstallState
            if isUninstalling { state = .uninstalling }
            else if isDownloading { state = .downloading }
            else if isPaused { state = .paused }
            else if isInstalled { state = .installed }
            else { state = .installable(isEnabled: true) }
            return LocalModelInstallSnapshot(
                target: .customLLM(canonicalRepo), state: state, isInstalled: isInstalled,
                isCurrentSelection: featureSettings.translation.modelSelectionID == .localLLM(canonicalRepo),
                statusText: "", badgeText: nil, downloadStatus: nil, canOpenLocation: isInstalled,
                canConfigure: isInstalled,
                configureActionTitle: isInstalled ? AppLocalization.localizedString("Configure") : nil
            )
        }

        let ggufTranslationInstallSnapshot: (GGUFTranslationModelID) -> LocalModelInstallSnapshot = { modelID in
            let isDownloading = isDownloadingGGUFTranslation(modelID)
            let isPaused = isPausedGGUFTranslation(modelID)
            let isInstalled = !isDownloading && !isPaused && isGGUFTranslationInstalled(modelID)
            let state: LocalModelInstallState
            if isDownloading { state = .downloading }
            else if isPaused { state = .paused }
            else if isInstalled { state = .installed }
            else { state = .installable(isEnabled: true) }
            return LocalModelInstallSnapshot(
                target: .ggufTranslation(modelID), state: state, isInstalled: isInstalled,
                isCurrentSelection: featureSettings.translation.modelSelectionID == .localGGUFTranslation(modelID),
                statusText: "", badgeText: nil, downloadStatus: nil, canOpenLocation: isInstalled,
                canConfigure: false, configureActionTitle: nil
            )
        }

        return ModelCatalogBuilder(
            mlxModelManager: TestModelManagers.mlx,
            customLLMManager: TestModelManagers.customLLM,
            ggufTranslationModelManager: TestModelManagers.gguf,
            remoteASRConfigurations: remoteASRConfigurations,
            remoteLLMConfigurations: remoteLLMConfigurations,
            featureSettings: featureSettings,
            hasIssue: hasIssue,
            customLLMBadgeText: { _ in nil },
            remoteASRStatusText: { _, _ in "" },
            remoteLLMBadgeText: { _ in nil },
            primaryUserLanguageCode: primaryUserLanguageCode,
            appleIntelligenceAvailability: appleIntelligenceAvailability,
            mlxInstallSnapshot: mlxInstallSnapshot,
            customLLMInstallSnapshot: customLLMInstallSnapshot,
            ggufTranslationInstallSnapshot: ggufTranslationInstallSnapshot,
            catalogPrimaryAction: { snapshot in
                ModelSettingsInstallActionResolver.catalogPrimaryAction(for: snapshot, perform: performAction)
            },
            catalogSecondaryActions: { snapshot in
                ModelSettingsInstallActionResolver.catalogSecondaryActions(for: snapshot, perform: performAction)
            },
            configureASRProvider: { _ in }, configureLLMProvider: { _ in }, showASRHintTarget: { _ in }
        )
    }

    private func makeFeatureSettings(
        transcriptionASR: FeatureModelSelectionID? = nil,
        translationASR: FeatureModelSelectionID? = nil,
        translationModel: FeatureModelSelectionID? = nil
    ) -> FeatureSettings {
        let transcriptionASR = transcriptionASR ?? .dictation
        let translationASR = translationASR ?? .dictation
        let translationModel = translationModel ?? .localLLM(CustomLLMModelManager.defaultModelRepo)
        return FeatureSettings(
            transcription: .init(asrSelectionID: transcriptionASR, llmEnabled: false,
                                 llmSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                                 prompt: AppPreferenceKey.defaultEnhancementPrompt),
            translation: .init(asrSelectionID: translationASR, modelSelectionID: translationModel,
                               targetLanguageRawValue: TranslationTargetLanguage.english.rawValue,
                               prompt: AppPreferenceKey.defaultTranslationPrompt),
            rewrite: .init(asrSelectionID: .dictation,
                           llmSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                           prompt: AppPreferenceKey.defaultRewritePrompt, appEnhancementEnabled: false)
        )
    }
}

@MainActor
private enum TestModelManagers {
    static let mlx = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
    static let customLLM = CustomLLMModelManager(modelRepo: CustomLLMModelManager.defaultModelRepo)
    static let gguf = GGUFTranslationModelManager(modelID: .hyMT2Q4KM)
}
