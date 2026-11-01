import Combine
import XCTest
@testable import Voxt

@MainActor
final class ModelConfigurationIssueResolverTests: MLXModelManagerTestCase {
    private let asrRepo = "mlx-community/Qwen3-ASR-0.6B-4bit"
    private let llmRepo = CustomLLMModelManager.defaultModelRepo

    private var selectedASRAvailabilityIssues: [ModelConfigurationIssue] {
        guard let reason = OmniASRBackend.unavailabilityReason(for: asrRepo) else { return [] }
        return [.init(scope: .mlxModel(asrRepo), message: reason)]
    }

    func testNativeASRAliasPreservesSelectionAndUsesPlatformAndInstallationGates() async throws {
        let repo = "mlx-community/parakeet-tdt-0.6b-v3"
        let scope = ModelConfigurationIssue.Scope.mlxModel(repo)
        let availabilityIssues: [ModelConfigurationIssue] = OmniASRBackend.unavailabilityReason(for: repo).map {
            [ModelConfigurationIssue(scope: scope, message: $0)]
        } ?? []
        let selections: [WritableKeyPath<FeatureSettings, FeatureModelSelectionID>] = [
            \.transcription.asrSelectionID, \.translation.asrSelectionID,
            \.rewrite.asrSelectionID, \.meeting.asrSelectionID
        ]
        for selection in selections {
            try await withIsolatedModelStorageRoot { root in
                let (defaults, name) = configuredDefaults()
                defer { defaults.removePersistentDomain(forName: "VoxtTests.\(name)") }
                var settings = FeatureSettingsStore.load(defaults: defaults)
                settings[keyPath: selection] = .mlx("mlx-community/Parakeet-0.6B")
                FeatureSettingsStore.save(settings, defaults: defaults)
                let savedPreferences = defaults.dictionaryRepresentation() as NSDictionary
                let mlx = MLXModelManager(modelRepo: repo)
                let llm = CustomLLMModelManager(modelRepo: llmRepo)
                XCTAssertTrue(mlx.isCheckingInstallation(repo: repo))
                XCTAssertEqual(issues(defaults, mlx, llm).filter { $0.scope == scope }, availabilityIssues)

                _ = try await mlx.refreshInstallation(repo: repo)
                let missingIssues: [ModelConfigurationIssue] = OmniASRBackend.isAppleSilicon
                    ? [ModelConfigurationIssue(scope: scope, message: AppLocalization.localizedString("Model needs to be installed."))]
                    : availabilityIssues
                XCTAssertEqual(issues(defaults, mlx, llm).filter { $0.scope == scope }, missingIssues)
                try seedValidMLXModelDirectory(repo: repo, root: root)
                mlx.checkExistingModel(refresh: true)
                _ = try await mlx.refreshInstallation(repo: repo)
                XCTAssertTrue(mlx.isModelDownloaded(repo: repo))
                XCTAssertEqual(issues(defaults, mlx, llm).filter { $0.scope == scope }, availabilityIssues)
                XCTAssertEqual(FeatureSettingsStore.load(defaults: defaults)[keyPath: selection], .mlx(repo))
                XCTAssertEqual(defaults.dictionaryRepresentation() as NSDictionary, savedPreferences)
                await mlx.shutdownForApplicationTermination()
                await llm.shutdownForApplicationTermination()
            }
        }
    }

    func testColdInstalledModelsNeverProduceMissingIssues() async throws {
        try await withIsolatedModelStorageRoot { root in
            try seedModels(root)
            let (defaults, name) = configuredDefaults()
            defer { defaults.removePersistentDomain(forName: "VoxtTests.\(name)") }
            let mlx = MLXModelManager(modelRepo: asrRepo)
            let llm = CustomLLMModelManager(modelRepo: llmRepo)
            // No suspension: scan publication cannot have reached MainActor yet.
            XCTAssertTrue(mlx.isCheckingInstallation(repo: asrRepo))
            XCTAssertTrue(llm.isCheckingInstallation(repo: llmRepo))
            XCTAssertEqual(issues(defaults, mlx, llm), selectedASRAvailabilityIssues)
            let asr = try await mlx.refreshInstallation(repo: asrRepo)
            let text = try await llm.refreshInstallation(repo: llmRepo)
            XCTAssertTrue(asr.isInstalled)
            XCTAssertTrue(text.isInstalled)
            XCTAssertEqual(issues(defaults, mlx, llm), selectedASRAvailabilityIssues)
            await mlx.shutdownForApplicationTermination()
            await llm.shutdownForApplicationTermination()
        }
    }

    func testConfirmedMissingModelsStillWarnInAllSelectedScopes() async throws {
        try await withIsolatedModelStorageRoot { _ in
            let (defaults, name) = configuredDefaults()
            defer { defaults.removePersistentDomain(forName: "VoxtTests.\(name)") }
            let mlx = MLXModelManager(modelRepo: asrRepo)
            let llm = CustomLLMModelManager(modelRepo: llmRepo)
            XCTAssertEqual(issues(defaults, mlx, llm), selectedASRAvailabilityIssues)
            _ = try await mlx.refreshInstallation(repo: asrRepo)
            _ = try await llm.refreshInstallation(repo: llmRepo)
            XCTAssertEqual(Set(issues(defaults, mlx, llm).map(\.scope)), [
                .mlxModel(asrRepo), .customLLMModel(llmRepo), .translationCustomLLM(llmRepo)
            ])
            await mlx.shutdownForApplicationTermination()
            await llm.shutdownForApplicationTermination()
        }
    }

    func testScanCompletionRefreshesIssuesWithoutMenuInteractionOrDownloads() async throws {
        try await withIsolatedModelStorageRoot { root in
            let (defaults, name) = configuredDefaults()
            defer { defaults.removePersistentDomain(forName: "VoxtTests.\(name)") }
            let mlx = MLXModelManager(modelRepo: asrRepo)
            let llm = CustomLLMModelManager(modelRepo: llmRepo)
            let missing = expectation(description: "automatic missing notification")
            let installed = expectation(description: "automatic installed notification")
            var sawMissing = false
            var sawInstalled = false
            // Same coalesced installation-only stream used by the settings shell.
            let subscription = ModelInstallationObservation.changes(mlx: mlx, customLLM: llm).sink {
                MainActor.assumeIsolated {
                    let current = self.issues(defaults, mlx, llm)
                    if current.count == 3, !sawMissing {
                        sawMissing = true
                        missing.fulfill()
                    }
                    if sawMissing, !sawInstalled,
                       mlx.isModelDownloaded(repo: self.asrRepo), llm.isModelDownloaded(repo: self.llmRepo),
                       current == self.selectedASRAvailabilityIssues {
                        sawInstalled = true
                        installed.fulfill()
                    }
                }
            }
            defer { subscription.cancel() }
            await fulfillment(of: [missing], timeout: 5)
            try seedModels(root)
            mlx.checkExistingModel(refresh: true)
            llm.checkExistingModel(refresh: true)
            XCTAssertEqual(issues(defaults, mlx, llm), selectedASRAvailabilityIssues)
            await fulfillment(of: [installed], timeout: 5)
            XCTAssertTrue(mlx.activeDownloadRepos.isEmpty)
            XCTAssertTrue(llm.activeDownloadRepos.isEmpty)
            await mlx.shutdownForApplicationTermination()
            await llm.shutdownForApplicationTermination()
        }
    }

    func testRemoteConfigurationWarningsAreNotSuppressedByPendingLocalScans() async {
        await withIsolatedModelStorageRoot { _ in
            let (defaults, name) = configuredDefaults()
            defer { defaults.removePersistentDomain(forName: "VoxtTests.\(name)") }
            var settings = FeatureSettingsStore.load(defaults: defaults)
            settings.transcription.asrSelectionID = .remoteASR(.openAIWhisper)
            settings.transcription.llmSelectionID = .remoteLLM(.openAI)
            FeatureSettingsStore.save(settings, defaults: defaults)
            let mlx = MLXModelManager(modelRepo: asrRepo)
            let llm = CustomLLMModelManager(modelRepo: llmRepo)
            let scopes = Set(issues(defaults, mlx, llm).map(\.scope))
            XCTAssertTrue(scopes.contains(.remoteASRProvider(.openAIWhisper)))
            XCTAssertTrue(scopes.contains(.remoteLLMProvider(.openAI)))
            XCTAssertFalse(scopes.contains(.translationCustomLLM(llmRepo)))
            await mlx.shutdownForApplicationTermination()
            await llm.shutdownForApplicationTermination()
        }
    }

    func testLocalSelectorShowsLoadingWithoutOfferingInstallOrSelection() {
        let checking = FeatureModelCatalogBuilder.localSelectorAvailability(isInstalled: false, isChecking: true)
        XCTAssertFalse(checking.isSelectable)
        XCTAssertEqual(checking.disabledReason, AppLocalization.localizedString("Loading…"))
        let missing = FeatureModelCatalogBuilder.localSelectorAvailability(isInstalled: false)
        XCTAssertFalse(missing.isSelectable)
        XCTAssertEqual(missing.disabledReason, AppLocalization.localizedString("Install this model in Model settings first."))
        let installed = FeatureModelCatalogBuilder.localSelectorAvailability(isInstalled: true)
        XCTAssertTrue(installed.isSelectable)
        XCTAssertNil(installed.disabledReason)
        for isInstalled in [false, true] {
            for isChecking in [false, true] {
                let unavailable = FeatureModelCatalogBuilder.localSelectorAvailability(
                    isInstalled: isInstalled,
                    isChecking: isChecking,
                    unavailabilityReason: "Native support is unavailable"
                )
                XCTAssertFalse(unavailable.isSelectable)
                XCTAssertEqual(unavailable.disabledReason, "Native support is unavailable")
            }
        }
    }

    private func configuredDefaults() -> (UserDefaults, String) {
        let name = UUID().uuidString
        let defaults = TestDoubles.makeUserDefaults(testName: name)
        var settings = FeatureSettings.placeholder
        settings.transcription.asrSelectionID = .mlx(asrRepo)
        settings.transcription.llmEnabled = true
        settings.transcription.llmSelectionID = .localLLM(llmRepo)
        settings.translation.asrSelectionID = .mlx(asrRepo)
        settings.translation.modelSelectionID = .localLLM(llmRepo)
        settings.rewrite.asrSelectionID = .mlx(asrRepo)
        settings.rewrite.llmSelectionID = .localLLM(llmRepo)
        settings.meeting.asrSelectionID = .mlx(asrRepo)
        FeatureSettingsStore.save(settings, defaults: defaults)
        return (defaults, name)
    }

    private func seedModels(_ root: URL) throws {
        try seedValidMLXModelDirectory(repo: asrRepo, root: root)
        let directory = try XCTUnwrap(CustomLLMModelStorageSupport.cacheDirectory(for: llmRepo, rootDirectory: root))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        try Data("weights".utf8).write(to: directory.appendingPathComponent("model.safetensors"))
    }

    private func issues(_ defaults: UserDefaults, _ mlx: MLXModelManager, _ llm: CustomLLMModelManager) -> [ModelConfigurationIssue] {
        ModelConfigurationIssueResolver.missingIssues(defaults: defaults, mlxModelManager: mlx, customLLMManager: llm)
    }
}
