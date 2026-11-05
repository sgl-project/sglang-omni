// HyMT2MigrationTests.swift
// Covers the GGUF-to-MLX Hy-MT2 migration and the translation-only routing it must preserve.

import XCTest
@testable import Voxt

final class HyMT2MigrationTests: MLXModelManagerTestCase {
    private let ggufDirectoryName = "gguf-translation"

    func testLaunchReclaimNeverTouchesModelStorageInsideXCTestHost() async throws {
        try await withIsolatedModelStorageRoot { root in
            let (defaults, name) = makeDefaults()
            defer { defaults.removePersistentDomain(forName: "VoxtTests.\(name)") }
            try seedGGUFWeights(root: root)
            try seedCustomLLMCheckpoint(repo: LegacyGGUFTranslationMigration.defaultRepo, root: root)

            let launchReclaim = LegacyGGUFTranslationMigration.reclaimRetiredArtifactsAfterLaunch(defaults: defaults)
            await launchReclaim?.value

            XCTAssertNil(launchReclaim)
            XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent(ggufDirectoryName).path))
        }
    }

    func testRetiredGGUFWeightsStayUntilMappedMLXCheckpointIsInstalled() async throws {
        try await withIsolatedModelStorageRoot { root in
            let (defaults, name) = makeDefaults()
            defer { defaults.removePersistentDomain(forName: "VoxtTests.\(name)") }
            let ggufDirectory = root.appendingPathComponent(ggufDirectoryName, isDirectory: true)
            try seedGGUFWeights(root: root)
            defaults.set(LegacyGGUFTranslationMigration.q8ModelID, forKey: AppPreferenceKey.legacyTranslationGGUFModelID)

            await LegacyGGUFTranslationMigration.reclaimRetiredArtifacts(defaults: defaults)
            XCTAssertTrue(FileManager.default.fileExists(atPath: ggufDirectory.path))

            try seedCustomLLMCheckpoint(repo: LegacyGGUFTranslationMigration.defaultRepo, root: root)
            await LegacyGGUFTranslationMigration.reclaimRetiredArtifacts(defaults: defaults)
            XCTAssertTrue(FileManager.default.fileExists(atPath: ggufDirectory.path), "Q8_0 weights wait for the 8-bit checkpoint")
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.legacyTranslationGGUFModelID), LegacyGGUFTranslationMigration.q8ModelID)

            try seedCustomLLMCheckpoint(repo: LegacyGGUFTranslationMigration.q8Repo, root: root)
            await LegacyGGUFTranslationMigration.reclaimRetiredArtifacts(defaults: defaults)
            XCTAssertFalse(FileManager.default.fileExists(atPath: ggufDirectory.path))
            XCTAssertNil(defaults.object(forKey: AppPreferenceKey.legacyTranslationGGUFModelID))
            XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("mlx-llm").path))
        }
    }

    func testMigratedGGUFUserWithoutCheckpointSeesInstallPromptAndKeepsWeights() async throws {
        try await withIsolatedModelStorageRoot { root in
            let (defaults, name) = makeDefaults()
            defer { defaults.removePersistentDomain(forName: "VoxtTests.\(name)") }
            let repo = LegacyGGUFTranslationMigration.q8Repo
            try seedGGUFWeights(root: root)
            defaults.set(LegacyGGUFTranslationMigration.providerRawValue, forKey: AppPreferenceKey.translationModelProvider)
            defaults.set(LegacyGGUFTranslationMigration.q8ModelID, forKey: AppPreferenceKey.legacyTranslationGGUFModelID)

            FeatureSettingsStore.migrateIfNeeded(defaults: defaults)
            await LegacyGGUFTranslationMigration.reclaimRetiredArtifacts(defaults: defaults)
            let mlx = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
            let llm = CustomLLMModelManager(modelRepo: CustomLLMModelManager.defaultModelRepo)
            _ = try await llm.refreshInstallation(repo: repo)

            XCTAssertEqual(FeatureSettingsStore.load(defaults: defaults).translation.modelSelectionID, .localLLM(repo))
            XCTAssertFalse(llm.canAttemptInference(repo: repo))
            XCTAssertTrue(llm.activeDownloadRepos.isEmpty)
            XCTAssertEqual(
                ModelConfigurationIssueResolver.missingIssues(defaults: defaults, mlxModelManager: mlx, customLLMManager: llm)
                    .filter { $0.scope == .translationCustomLLM(repo) },
                [ModelConfigurationIssue(scope: .translationCustomLLM(repo), message: AppLocalization.localizedString("Model needs to be installed."))]
            )
            XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent(ggufDirectoryName).path))
            await mlx.shutdownForApplicationTermination()
            await llm.shutdownForApplicationTermination()
        }
    }

    func testTranslationHistoryFollowUpSkipsTranslationOnlyModels() {
        var settings = FeatureSettings.placeholder
        let translationOnlySelections = [
            FeatureModelSelectionID.localLLM(LegacyGGUFTranslationMigration.defaultRepo),
            FeatureModelSelectionID(rawValue: LegacyGGUFTranslationMigration.selectionPrefix + LegacyGGUFTranslationMigration.q8ModelID),
        ]
        for selection in translationOnlySelections {
            settings.translation.modelSelectionID = selection
            XCTAssertNil(
                AppDelegate.resolvedTranscriptionFollowUpTextSelection(for: .translation, featureSettings: settings),
                selection.rawValue
            )
        }

        settings.translation.modelSelectionID = .localLLM(CustomLLMModelManager.defaultModelRepo)
        XCTAssertEqual(
            AppDelegate.resolvedTranscriptionFollowUpTextSelection(for: .translation, featureSettings: settings),
            .localLLM(repo: CustomLLMModelManager.defaultModelRepo)
        )
    }

    func testDictionaryHistoryScanSkipsInstalledTranslationOnlyModels() async throws {
        try await withIsolatedModelStorageRoot { root in
            let translationRepo = LegacyGGUFTranslationMigration.defaultRepo
            let generalRepo = CustomLLMModelManager.defaultModelRepo
            try seedCustomLLMCheckpoint(repo: translationRepo, root: root)
            try seedCustomLLMCheckpoint(repo: generalRepo, root: root)
            let llm = CustomLLMModelManager(modelRepo: generalRepo)
            _ = try await llm.refreshInstallation(repo: translationRepo)
            _ = try await llm.refreshInstallation(repo: generalRepo)

            XCTAssertTrue(llm.canAttemptInference(repo: translationRepo))
            XCTAssertFalse(AppDelegate.canRunDictionaryHistoryScan(repo: translationRepo, customLLMManager: llm))
            XCTAssertTrue(AppDelegate.canRunDictionaryHistoryScan(repo: generalRepo, customLLMManager: llm))
            await llm.shutdownForApplicationTermination()
        }
    }

    private func makeDefaults() -> (UserDefaults, String) {
        let name = UUID().uuidString
        return (TestDoubles.makeUserDefaults(testName: name), name)
    }

    private func seedGGUFWeights(root: URL) throws {
        let directory = root.appendingPathComponent("\(ggufDirectoryName)/tencent_Hy-MT2-1.8B-GGUF", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("gguf".utf8).write(to: directory.appendingPathComponent("Hy-MT2-1.8B-Q8_0.gguf"))
    }

    private func seedCustomLLMCheckpoint(repo: String, root: URL) throws {
        let directory = try XCTUnwrap(CustomLLMModelStorageSupport.cacheDirectory(for: repo, rootDirectory: root))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        try Data("weights".utf8).write(to: directory.appendingPathComponent("model.safetensors"))
    }
}
