import XCTest
@testable import Voxt
import HuggingFace
import MLX
import OmniModelArtifacts

@MainActor
final class MLXModelManagerStorageTests: MLXModelManagerTestCase {
    func testMLXAudioActiveHubCacheUsesConfiguredModelStorageRoot() throws {
        let defaults = UserDefaults.standard
        let previousPath = defaults.string(forKey: AppPreferenceKey.modelStorageRootPath)
        let previousBookmark = defaults.data(forKey: AppPreferenceKey.modelStorageRootBookmark)
        let customRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)

        defaults.set(customRoot.path, forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
        ModelStorageDirectoryManager.resetForTesting()
        addTeardownBlock {
            if let previousPath {
                defaults.set(previousPath, forKey: AppPreferenceKey.modelStorageRootPath)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootPath)
            }
            if let previousBookmark {
                defaults.set(previousBookmark, forKey: AppPreferenceKey.modelStorageRootBookmark)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
            }
            ModelStorageDirectoryManager.resetForTesting()
        }

        let hubCache = MLXModelStorageSupport.hubCache(
            rootDirectory: ModelStorageDirectoryManager.resolvedWriteRootURL()
        )
        XCTAssertEqual(hubCache.cacheDirectory, customRoot)
    }

    func testMLXAudioClearHubCacheTargetsConfiguredModelStorageRoot() throws {
        let repoID = try XCTUnwrap(Repo.ID(rawValue: "mlx-community/Qwen3-ASR-0.6B-8bit"))
        let rootDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let cache = MLXModelStorageSupport.hubCache(rootDirectory: rootDirectory)
        let repoDirectory = cache.repoDirectory(repo: repoID, kind: .model)
        let metadataDirectory = cache.metadataDirectory(repo: repoID, kind: .model)

        try FileManager.default.createDirectory(at: repoDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: metadataDirectory, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: rootDirectory)
        }

        XCTAssertTrue(FileManager.default.fileExists(atPath: repoDirectory.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: metadataDirectory.path))

        MLXModelStorageSupport.clearHubCache(for: repoID, rootDirectory: rootDirectory)

        XCTAssertFalse(FileManager.default.fileExists(atPath: repoDirectory.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: metadataDirectory.path))
    }

    func testPartialMLXDownloadDirectoryIsNotTreatedAsInstalled() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let partialDirectory = root
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent("mlx-community_Qwen3-ASR-0.6B-4bit-download")
        let finalDirectory = root
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent("mlx-community_Qwen3-ASR-0.6B-4bit")

        try FileManager.default.createDirectory(at: partialDirectory, withIntermediateDirectories: true)
        try Data("partial".utf8).write(to: partialDirectory.appendingPathComponent("weights.bin"))
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertTrue(FileManager.default.directoryContainsRegularFiles(at: partialDirectory))
        XCTAssertFalse(MLXModelDownloadSupport.isModelDirectoryValid(finalDirectory, fileManager: .default))
    }

    func testWhisperDirectoryWithoutTokenizerAssetsIsNotTreatedAsInstalled() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let repo = "mlx-community/whisper-large-v3-turbo"
        let modelDir = root
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent("mlx-community_whisper-large-v3-turbo")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try Data(#"{"model_type":"whisper"}"#.utf8).write(to: modelDir.appendingPathComponent("config.json"))
        try Data("weights".utf8).write(to: modelDir.appendingPathComponent("weights.safetensors"))
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertFalse(
            MLXModelDownloadSupport.isModelDirectoryValid(
                modelDir,
                repo: repo,
                fileManager: .default
            )
        )
    }

    func testWhisperDirectoryWithTokenizerAssetsIsTreatedAsInstalled() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let repo = "mlx-community/whisper-large-v3-turbo"
        let modelDir = root
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent("mlx-community_whisper-large-v3-turbo")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try Data(#"{"model_type":"whisper"}"#.utf8).write(to: modelDir.appendingPathComponent("config.json"))
        try Data("weights".utf8).write(to: modelDir.appendingPathComponent("weights.safetensors"))
        for assetPath in MLXModelDownloadSupport.whisperTokenizerAssetPaths {
            try Data("asset".utf8).write(to: modelDir.appendingPathComponent(assetPath))
        }
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertTrue(
            MLXModelDownloadSupport.isModelDirectoryValid(
                modelDir,
                repo: repo,
                fileManager: .default
            )
        )
    }

    func testAuxiliarySileroArtifactIsManagedWithoutBecomingAnASRModel() async throws {
        try await withIsolatedModelStorageRoot { root in
            let repo = SileroVADModelSupport.repo
            try seedValidMLXModelDirectory(repo: repo, root: root)
            let manager = MLXModelManager(modelRepo: repo)
            XCTAssertTrue(MLXModelManager.isManagedArtifactRepo(repo))
            XCTAssertFalse(MLXModelCatalog.availableModels.contains { $0.id == repo })
            let snapshot = try await manager.refreshInstallation(repo: repo)
            XCTAssertTrue(snapshot.isInstalled)
            let directory = try await manager.ensureModelDirectory(repo: repo)
            XCTAssertEqual(directory, snapshot.directory)
        }
    }

    func testSileroProvisionerKeepsCachedDirectoryPresenceBehavior() async throws {
        try await withIsolatedModelStorageRoot { root in
            try seedValidMLXModelDirectory(repo: SileroVADModelSupport.repo, root: root)
            let cachedDirectory = try XCTUnwrap(MeetingVADModelStorage.writeModelDirectory())
            let provisioner = SileroVADModelProvisioner()
            let directory = try await provisioner.ensureModelDirectory()
            XCTAssertEqual(directory, cachedDirectory)
        }
    }

    func testStateForUnknownRepoDefaultsToNotDownloaded() async {
        await withIsolatedModelStorageRoot { _ in
            let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
            let unknownRepo = "mlx-community/some-unknown-repo"

            XCTAssertEqual(manager.state(for: unknownRepo), .notDownloaded)
            XCTAssertNil(manager.pausedStatusMessage(for: unknownRepo))
            XCTAssertFalse(manager.isDownloading(repo: unknownRepo))
            XCTAssertFalse(manager.isPaused(repo: unknownRepo))
        }
    }

    func testStateForRepoReturnsDownloadedWhenValidModelDirExists() async throws {
        try await withIsolatedModelStorageRoot { root in
            let otherRepo = "mlx-community/parakeet-tdt-0.6b-v3"
            try seedValidMLXModelDirectory(repo: otherRepo, root: root)

            let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
            _ = try await manager.refreshInstallation(repo: otherRepo)
            XCTAssertEqual(manager.state(for: otherRepo), .downloaded)
            XCTAssertFalse(manager.isDownloading(repo: otherRepo))
        }
    }

    func testCancelDownloadForUnknownRepoIsSafeAndDoesNotMutateCurrent() async {
        await withIsolatedModelStorageRoot { _ in
            let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
            let stateBefore = manager.state

            manager.cancelDownload(repo: "mlx-community/some-unknown-repo")

            XCTAssertEqual(manager.state, stateBefore)
        }
    }

    func testCancelDownloadForNonActiveRepoCleansUpPartialArtifacts() async throws {
        try await withIsolatedModelStorageRoot { root in
            let staleRepo = "mlx-community/parakeet-tdt-0.6b-v3"
            let staleDir = root
                .appendingPathComponent("mlx-audio")
                .appendingPathComponent("mlx-community_parakeet-tdt-0.6b-v3-download")
            try FileManager.default.createDirectory(at: staleDir, withIntermediateDirectories: true)
            try Data("partial".utf8).write(to: staleDir.appendingPathComponent("weights.bin"))
            XCTAssertTrue(FileManager.default.fileExists(atPath: staleDir.path))

            let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
            await manager.cancelDownloadAndWait(repo: staleRepo)

            XCTAssertFalse(FileManager.default.fileExists(atPath: staleDir.path))
        }
    }

    func testCancelPartialDownloadPreservesExistingDestinationWeights() async throws {
        try await withIsolatedModelStorageRoot { root in
            let repo = "mlx-community/parakeet-tdt-0.6b-v3"
            let destination = root.appendingPathComponent("mlx-audio/mlx-community_parakeet-tdt-0.6b-v3")
            let staging = root.appendingPathComponent("mlx-audio/mlx-community_parakeet-tdt-0.6b-v3-download")
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
            let weights = Data("previous weights awaiting repair".utf8)
            try weights.write(to: destination.appendingPathComponent("model.safetensors"))
            try Data("partial".utf8).write(to: staging.appendingPathComponent("model.safetensors"))
            let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
            await manager.cancelDownloadAndWait(repo: repo)
            XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("model.safetensors")), weights)
            XCTAssertFalse(FileManager.default.fileExists(atPath: staging.path))
        }
    }

    func testDeleteModelRejectsReaderLeaseWithoutRemovingWeights() async throws {
        try await withIsolatedModelStorageRoot { root in
            let repo = "mlx-community/parakeet-tdt-0.6b-v3"
            try seedValidMLXModelDirectory(repo: repo, root: root)
            let destination = root.appendingPathComponent("mlx-audio/mlx-community_parakeet-tdt-0.6b-v3")
            let reader = try OmniModelDirectoryLease.acquire(for: destination, mode: .sharedRead)
            let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
            let result = await manager.deleteModel(repo: repo)
            guard case .failure(let error) = result else { return XCTFail("Deletion bypassed reader lease") }
            XCTAssertEqual(error as? OmniModelStorageError, .busy(path: destination.path))
            XCTAssertTrue(FileManager.default.fileExists(atPath: destination.appendingPathComponent("weights.safetensors").path))
            withExtendedLifetime(reader) {}
        }
    }

    func testActiveDownloadReposIsEmptyWhenNothingIsRunning() async {
        await withIsolatedModelStorageRoot { _ in
            let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
            XCTAssertTrue(manager.activeDownloadRepos.isEmpty)
        }
    }

    func testDeleteModelForNonCurrentRepoClearsStoredPerRepoState() async throws {
        try await withIsolatedModelStorageRoot { root in
            let otherRepo = "mlx-community/parakeet-tdt-0.6b-v3"
            try seedValidMLXModelDirectory(repo: otherRepo, root: root)

            let manager = MLXModelManager(modelRepo: otherRepo)
            _ = try await manager.refreshInstallation(repo: otherRepo)
            XCTAssertEqual(manager.state(for: otherRepo), .downloaded)

            manager.updateModel(repo: MLXModelManager.defaultModelRepo)
            _ = try await manager.refreshInstallation(repo: otherRepo)
            XCTAssertEqual(manager.state(for: otherRepo), .downloaded)

            await manager.deleteModel(repo: otherRepo)

            XCTAssertEqual(manager.state(for: otherRepo), .notDownloaded)
            XCTAssertNil(manager.pausedStatusMessage(for: otherRepo))
        }
    }

    func testDownloadAttemptReusesInstalledDirectoryWithoutPublishingProgress() async throws {
        try await withIsolatedModelStorageRoot { root in
            let repo = "mlx-community/parakeet-tdt-0.6b-v3"
            try seedValidMLXModelDirectory(repo: repo, root: root)
            let session = URLSession(configuration: .ephemeral)
            defer { session.invalidateAndCancel() }
            let directory = try await MLXModelDownloadAttempt.download(
                repoID: try XCTUnwrap(Repo.ID(rawValue: repo)),
                session: session, baseURL: root,
                bearerToken: nil, writeRoot: root,
                userAgent: MLXModelManager.hubUserAgent, downloadSizeTolerance: 0.9
            ) { _ in
                XCTFail("An already installed model must not start transfer progress.")
            }

            XCTAssertEqual(directory.path, root.appendingPathComponent("mlx-audio/mlx-community_parakeet-tdt-0.6b-v3").path)
            XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent(
                "mlx-audio/mlx-community_parakeet-tdt-0.6b-v3-download"
            ).path))
        }
    }

    func testArtifactShadowReplacementPreservesSourceAndPartialDownload() async throws {
        try await withIsolatedModelStorageRoot { root in
            let source = root.appendingPathComponent("legacy-model", isDirectory: true)
            let shadow = root.appendingPathComponent(".derived-model-artifacts/mlx-audio-shadow/fixture", isDirectory: true)
            let partial = root.appendingPathComponent("mlx-audio/fixture-download", isDirectory: true)
            for directory in [source, shadow, partial] {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            }
            try Data("weights".utf8).write(to: source.appendingPathComponent("model.safetensors"))
            try Data("stale".utf8).write(to: shadow.appendingPathComponent("old.json"))
            try Data("partial".utf8).write(to: partial.appendingPathComponent("download.state"))

            _ = try MLXModelArtifactPreparation.prepareWritableShadowDirectory(from: source, to: shadow)

            XCTAssertFalse(FileManager.default.fileExists(atPath: shadow.appendingPathComponent("old.json").path))
            XCTAssertEqual(
                try FileManager.default.destinationOfSymbolicLink(atPath: shadow.appendingPathComponent("model.safetensors").path),
                source.appendingPathComponent("model.safetensors").path
            )
            XCTAssertEqual(try Data(contentsOf: source.appendingPathComponent("model.safetensors")), Data("weights".utf8))
            XCTAssertEqual(try Data(contentsOf: partial.appendingPathComponent("download.state")), Data("partial".utf8))
        }
    }

    func testDeleteModelRemovesQwen3ShadowDirectory() async throws {
        try await withIsolatedModelStorageRoot { writableRoot in
            let repo = "mlx-community/Qwen3-ASR-0.6B-4bit"
            try seedValidMLXModelDirectory(repo: repo, root: writableRoot)
            let sourceDirectory = writableRoot
                .appendingPathComponent("mlx-audio", isDirectory: true)
                .appendingPathComponent("mlx-community_Qwen3-ASR-0.6B-4bit", isDirectory: true)
            try? FileManager.default.removeItem(at: sourceDirectory.appendingPathComponent("tokenizer.json"))

            let shadowDirectory = writableRoot
                .appendingPathComponent(".derived-model-artifacts/mlx-audio-shadow", isDirectory: true)
                .appendingPathComponent("mlx-community_Qwen3-ASR-0.6B-4bit", isDirectory: true)
            let loadDirectory = try MLXModelArtifactPreparation.prepareWritableShadowDirectory(
                from: sourceDirectory, to: shadowDirectory
            )
            XCTAssertTrue(FileManager.default.fileExists(atPath: loadDirectory.path))
            let manager = MLXModelManager(modelRepo: repo)

            await manager.deleteModel(repo: repo)

            XCTAssertFalse(FileManager.default.fileExists(atPath: sourceDirectory.path))
            XCTAssertFalse(FileManager.default.fileExists(atPath: loadDirectory.path))
        }
    }

    func testRefreshStorageRootClearsStoredPerRepoState() async throws {
        let defaults = UserDefaults.standard
        let previousPath = defaults.string(forKey: AppPreferenceKey.modelStorageRootPath)
        let previousBookmark = defaults.data(forKey: AppPreferenceKey.modelStorageRootBookmark)

        try await withIsolatedModelStorageRoot { originalRoot in
            let otherRepo = "mlx-community/parakeet-tdt-0.6b-v3"
            try seedValidMLXModelDirectory(repo: otherRepo, root: originalRoot)

            let manager = MLXModelManager(modelRepo: otherRepo)
            _ = try await manager.refreshInstallation(repo: otherRepo)
            XCTAssertEqual(manager.state(for: otherRepo), .downloaded)

            manager.updateModel(repo: MLXModelManager.defaultModelRepo)
            _ = try await manager.refreshInstallation(repo: otherRepo)
            XCTAssertEqual(manager.state(for: otherRepo), .downloaded)

            let newRoot = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString, isDirectory: true)
            defaults.set(newRoot.path, forKey: AppPreferenceKey.modelStorageRootPath)
            defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
            ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(newRoot)
            defer {
                if let previousPath {
                    defaults.set(previousPath, forKey: AppPreferenceKey.modelStorageRootPath)
                } else {
                    defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootPath)
                }
                if let previousBookmark {
                    defaults.set(previousBookmark, forKey: AppPreferenceKey.modelStorageRootBookmark)
                } else {
                    defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
                }
                try? FileManager.default.removeItem(at: newRoot)
            }

            manager.refreshStorageRoot()
            _ = try await manager.refreshInstallation(repo: otherRepo)

            XCTAssertEqual(manager.state(for: otherRepo), .notDownloaded)
            XCTAssertNil(manager.pausedStatusMessage(for: otherRepo))
        }
    }

    /// Models in a storage root inside Downloads disappeared whenever the user cleaned that folder.
    func testStorageRootInsideDownloadsMovesOnlyVoxtFilesToTheDefaultRoot() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
        let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
        let destinationRootURL = scratch.appendingPathComponent("Application Support/model-storage", isDirectory: true)
        try writeFixture("source qwen", to: sourceRootURL.appendingPathComponent("mlx-audio/org_qwen/weights.safetensors"))
        try writeFixture("source shared", to: sourceRootURL.appendingPathComponent("mlx-audio/org_shared/weights.safetensors"))
        try writeFixture("{}", to: sourceRootURL.appendingPathComponent("mlx-audio/org_shared/config.json"))
        try writeFixture("source q8", to: sourceRootURL.appendingPathComponent("gguf-translation/hy/q8.gguf"))
        try writeFixture("source llm", to: sourceRootURL.appendingPathComponent("mlx-llm/org_llm/weights.safetensors"))
        try writeFixture("user file", to: sourceRootURL.appendingPathComponent("holiday.mov"))
        try writeFixture("default shared", to: destinationRootURL.appendingPathComponent("mlx-audio/org_shared/weights.safetensors"))
        try writeFixture("default q4", to: destinationRootURL.appendingPathComponent("gguf-translation/hy/q4.gguf"))
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)
        defer { ModelStorageDirectoryManager.resetForTesting() }

        ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
            defaults: defaults,
            downloadsURL: downloadsURL,
            destinationRootURL: destinationRootURL
        )

        XCTAssertEqual(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: destinationRootURL), "source qwen")
        XCTAssertEqual(fixtureText(at: "mlx-llm/org_llm/weights.safetensors", under: destinationRootURL), "source llm")
        XCTAssertEqual(fixtureText(at: "mlx-audio/org_shared/weights.safetensors", under: destinationRootURL), "default shared")
        XCTAssertEqual(fixtureText(at: "mlx-audio/org_shared/config.json", under: destinationRootURL), "{}")
        XCTAssertEqual(fixtureText(at: "gguf-translation/hy/q8.gguf", under: destinationRootURL), "source q8")
        XCTAssertEqual(fixtureText(at: "gguf-translation/hy/q4.gguf", under: destinationRootURL), "default q4")
        XCTAssertEqual(fixtureText(at: "holiday.mov", under: sourceRootURL), "user file")
        XCTAssertNil(fixtureText(at: "holiday.mov", under: destinationRootURL))
        XCTAssertNil(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath))
        XCTAssertNil(defaults.data(forKey: AppPreferenceKey.modelStorageRootBookmark))
    }

    /// A move that failed halfway left models outside every readable root.
    func testFailedMigrationOutOfDownloadsUndoesItsMovesAndKeepsTheSavedRoot() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
        let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
        let destinationRootURL = scratch.appendingPathComponent("model-storage", isDirectory: true)
        let readOnlyURL = destinationRootURL.appendingPathComponent("mlx-llm", isDirectory: true)
        try writeFixture("source qwen", to: sourceRootURL.appendingPathComponent("mlx-audio/org_qwen/weights.safetensors"))
        try writeFixture("source llm", to: sourceRootURL.appendingPathComponent("mlx-llm/org_llm/weights.safetensors"))
        try FileManager.default.createDirectory(at: readOnlyURL, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: readOnlyURL.path)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: readOnlyURL.path)
            try? FileManager.default.removeItem(at: scratch)
        }
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)
        defer { ModelStorageDirectoryManager.resetForTesting() }

        let moved = ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
            defaults: defaults,
            downloadsURL: downloadsURL,
            destinationRootURL: destinationRootURL
        )

        XCTAssertTrue(moved.isEmpty)
        XCTAssertEqual(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: sourceRootURL), "source qwen")
        XCTAssertEqual(fixtureText(at: "mlx-llm/org_llm/weights.safetensors", under: sourceRootURL), "source llm")
        XCTAssertNil(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: destinationRootURL))
        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath), sourceRootURL.path)
    }

    /// Merging into an incomplete copy could split one complete model into two broken halves.
    func testMigrationThatWouldLeaveAnIncompleteModelIsUndone() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
        let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
        let destinationRootURL = scratch.appendingPathComponent("model-storage", isDirectory: true)
        try writeFixture("{}", to: sourceRootURL.appendingPathComponent("mlx-audio/org_qwen/config.json"))
        try writeFixture("source weights", to: sourceRootURL.appendingPathComponent("mlx-audio/org_qwen/weights.safetensors"))
        try writeFixture("not json", to: destinationRootURL.appendingPathComponent("mlx-audio/org_qwen/config.json"))
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)
        defer { ModelStorageDirectoryManager.resetForTesting() }

        let moved = ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
            defaults: defaults,
            downloadsURL: downloadsURL,
            destinationRootURL: destinationRootURL
        )

        XCTAssertTrue(moved.isEmpty)
        XCTAssertEqual(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: sourceRootURL), "source weights")
        XCTAssertNil(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: destinationRootURL))
        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath), sourceRootURL.path)
    }

    /// Following a symbolic link would move files from wherever the link points.
    func testMigrationStopsAtASymbolicLinkWithoutMovingItsTarget() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
        let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
        let destinationRootURL = scratch.appendingPathComponent("model-storage", isDirectory: true)
        let unrelatedURL = scratch.appendingPathComponent("Documents", isDirectory: true)
        try writeFixture("user document", to: unrelatedURL.appendingPathComponent("org_notes/notes.txt"))
        try FileManager.default.createDirectory(at: sourceRootURL, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: sourceRootURL.appendingPathComponent("mlx-audio"),
            withDestinationURL: unrelatedURL
        )
        try writeFixture("default weights", to: destinationRootURL.appendingPathComponent("mlx-audio/org_qwen/weights.safetensors"))
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)
        defer { ModelStorageDirectoryManager.resetForTesting() }

        let moved = ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
            defaults: defaults,
            downloadsURL: downloadsURL,
            destinationRootURL: destinationRootURL
        )

        XCTAssertTrue(moved.isEmpty)
        XCTAssertEqual(fixtureText(at: "org_notes/notes.txt", under: unrelatedURL), "user document")
        XCTAssertNil(fixtureText(at: "mlx-audio/org_notes/notes.txt", under: destinationRootURL))
        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath), sourceRootURL.path)
    }

    /// Each case once left models in Downloads while the saved root was switched away from them.
    func testMigrationIsUndoneWhenAFolderCannotMoveCleanly() throws {
        let cases: [(name: String, prepare: (URL, URL) throws -> Void)] = [
            ("file in the way", { _, destinationRootURL in
                try self.writeFixture("not a folder", to: destinationRootURL.appendingPathComponent("mlx-llm"))
            }),
            ("link inside a moved model", { sourceRootURL, _ in
                try FileManager.default.createSymbolicLink(
                    atPath: sourceRootURL.appendingPathComponent("mlx-llm/org_llm/linked.safetensors").path,
                    withDestinationPath: "weights.safetensors"
                )
            }),
            ("file where the model folder goes", { _, destinationRootURL in
                try self.writeFixture("not a folder", to: destinationRootURL.appendingPathComponent("mlx-llm/org_llm"))
            }),
            ("link inside the existing destination model", { sourceRootURL, destinationRootURL in
                let destinationModelURL = destinationRootURL.appendingPathComponent("mlx-llm/org_llm", isDirectory: true)
                try FileManager.default.createDirectory(at: destinationModelURL, withIntermediateDirectories: true)
                try FileManager.default.createSymbolicLink(
                    atPath: destinationModelURL.appendingPathComponent("tokenizer.json").path,
                    withDestinationPath: sourceRootURL.appendingPathComponent("tokenizer.json").path
                )
            }),
        ]
        for testCase in cases {
            let scratch = try TestDoubles.canonicalTemporaryDirectory()
                .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: scratch) }
            let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
            let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
            let destinationRootURL = scratch.appendingPathComponent("model-storage", isDirectory: true)
            try writeFixture("source qwen", to: sourceRootURL.appendingPathComponent("mlx-audio/org_qwen/weights.safetensors"))
            try writeFixture("source llm", to: sourceRootURL.appendingPathComponent("mlx-llm/org_llm/weights.safetensors"))
            try testCase.prepare(sourceRootURL, destinationRootURL)
            let defaults = TestDoubles.makeUserDefaults()
            defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
            ModelStorageDirectoryManager.resetForTesting()
            ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)

            let moved = ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
                defaults: defaults,
                downloadsURL: downloadsURL,
                destinationRootURL: destinationRootURL
            )
            ModelStorageDirectoryManager.resetForTesting()

            XCTAssertTrue(moved.isEmpty, testCase.name)
            XCTAssertEqual(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: sourceRootURL), "source qwen", testCase.name)
            XCTAssertEqual(fixtureText(at: "mlx-llm/org_llm/weights.safetensors", under: sourceRootURL), "source llm", testCase.name)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath), sourceRootURL.path, testCase.name)
        }
    }

    /// Moving a model that a native server or download still uses would pull its files away mid-use.
    func testMigrationLeavesStorageUnchangedWhileAModelIsInUse() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
        let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
        let destinationRootURL = scratch.appendingPathComponent("model-storage", isDirectory: true)
        let modelURL = sourceRootURL.appendingPathComponent("mlx-audio/org_qwen", isDirectory: true)
        try writeFixture("source qwen", to: modelURL.appendingPathComponent("weights.safetensors"))
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)
        defer { ModelStorageDirectoryManager.resetForTesting() }
        let readerLease = try OmniModelDirectoryLease.acquire(for: modelURL, mode: .sharedRead)

        let moved = withExtendedLifetime(readerLease) {
            ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
                defaults: defaults,
                downloadsURL: downloadsURL,
                destinationRootURL: destinationRootURL
            )
        }

        XCTAssertTrue(moved.isEmpty)
        XCTAssertEqual(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: sourceRootURL), "source qwen")
        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath), sourceRootURL.path)
    }

    /// A publisher reserves a destination model before writing it; moving into it would corrupt that publication.
    func testMigrationLeavesStorageUnchangedWhileTheDestinationModelIsReserved() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
        let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
        let destinationRootURL = scratch.appendingPathComponent("model-storage", isDirectory: true)
        try writeFixture("source qwen", to: sourceRootURL.appendingPathComponent("mlx-audio/org_qwen/weights.safetensors"))
        let destinationFolderURL = destinationRootURL.appendingPathComponent("mlx-audio", isDirectory: true)
        try FileManager.default.createDirectory(at: destinationFolderURL, withIntermediateDirectories: true)
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)
        defer { ModelStorageDirectoryManager.resetForTesting() }
        let publisherLease = try OmniModelDirectoryLease.acquire(
            for: destinationFolderURL.appendingPathComponent("org_qwen", isDirectory: true),
            mode: .exclusiveMutation
        )

        let moved = withExtendedLifetime(publisherLease) {
            ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
                defaults: defaults,
                downloadsURL: downloadsURL,
                destinationRootURL: destinationRootURL
            )
        }

        XCTAssertTrue(moved.isEmpty)
        XCTAssertEqual(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: sourceRootURL), "source qwen")
        XCTAssertNil(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: destinationRootURL))
        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath), sourceRootURL.path)
    }

    /// Moving a whole namespace folder took its lock folder along, so new locks at the old paths
    /// would not exclude the migration that still held the moved ones.
    func testMigrationIntoAnAbsentNamespaceLeavesEachRootItsOwnLockFolder() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
        let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
        let destinationRootURL = scratch.appendingPathComponent("model-storage", isDirectory: true)
        try writeFixture("source qwen", to: sourceRootURL.appendingPathComponent("mlx-audio/org_qwen/weights.safetensors"))
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)
        defer { ModelStorageDirectoryManager.resetForTesting() }

        ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
            defaults: defaults,
            downloadsURL: downloadsURL,
            destinationRootURL: destinationRootURL
        )

        XCTAssertEqual(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: destinationRootURL), "source qwen")
        XCTAssertTrue(FileManager.default.fileExists(atPath: sourceRootURL.appendingPathComponent("mlx-audio/.omni-model-locks").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: destinationRootURL.appendingPathComponent("mlx-audio/.omni-model-locks").path))
        XCTAssertNil(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath))
    }

    /// An active download stages its files in a sibling folder; moving it would pull them from under the downloader.
    func testMigrationLeavesStorageUnchangedWhileADownloadStagesFiles() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
        let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
        let destinationRootURL = scratch.appendingPathComponent("model-storage", isDirectory: true)
        try writeFixture("source qwen", to: sourceRootURL.appendingPathComponent("mlx-audio/org_qwen/weights.safetensors"))
        let stagingURL = sourceRootURL.appendingPathComponent("mlx-audio/org_new-download", isDirectory: true)
        try writeFixture("partial", to: stagingURL.appendingPathComponent("weights.safetensors.part"))
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)
        defer { ModelStorageDirectoryManager.resetForTesting() }
        let downloaderLease = try OmniModelDirectoryLease.acquire(for: stagingURL, mode: .exclusiveMutation)

        let moved = withExtendedLifetime(downloaderLease) {
            ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
                defaults: defaults,
                downloadsURL: downloadsURL,
                destinationRootURL: destinationRootURL
            )
        }

        XCTAssertTrue(moved.isEmpty)
        XCTAssertEqual(fixtureText(at: "mlx-audio/org_new-download/weights.safetensors.part", under: sourceRootURL), "partial")
        XCTAssertEqual(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: sourceRootURL), "source qwen")
        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath), sourceRootURL.path)
    }

    /// A storage root reached through a symbolic link, at the root or any folder above it, would move
    /// files out of, or into, wherever the link points.
    func testMigrationStopsWhenEitherRootPathContainsASymbolicLink() throws {
        let cases = ["source root", "source parent", "destination root", "destination parent"]
        for linkCase in cases {
            let scratch = try TestDoubles.canonicalTemporaryDirectory()
                .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: scratch) }
            let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
            let realURL = scratch.appendingPathComponent("Elsewhere", isDirectory: true)
            try FileManager.default.createDirectory(at: downloadsURL, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: realURL, withIntermediateDirectories: true)
            var sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
            var destinationRootURL = scratch.appendingPathComponent("model-storage", isDirectory: true)
            var modelsURL = sourceRootURL
            switch linkCase {
            case "source root":
                try FileManager.default.createSymbolicLink(at: sourceRootURL, withDestinationURL: realURL)
                modelsURL = realURL
            case "source parent":
                let aliasURL = downloadsURL.appendingPathComponent("Alias", isDirectory: true)
                try FileManager.default.createSymbolicLink(at: aliasURL, withDestinationURL: realURL)
                sourceRootURL = aliasURL.appendingPathComponent("Models", isDirectory: true)
                modelsURL = realURL.appendingPathComponent("Models", isDirectory: true)
            case "destination root":
                try FileManager.default.createSymbolicLink(at: destinationRootURL, withDestinationURL: realURL)
            default:
                let aliasURL = scratch.appendingPathComponent("Linked Support", isDirectory: true)
                try FileManager.default.createSymbolicLink(at: aliasURL, withDestinationURL: realURL)
                destinationRootURL = aliasURL.appendingPathComponent("model-storage", isDirectory: true)
            }
            try writeFixture("weights", to: modelsURL.appendingPathComponent("mlx-audio/org_qwen/weights.safetensors"))
            let defaults = TestDoubles.makeUserDefaults()
            defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
            ModelStorageDirectoryManager.resetForTesting()
            ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)

            let moved = ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
                defaults: defaults,
                downloadsURL: downloadsURL,
                destinationRootURL: destinationRootURL
            )
            ModelStorageDirectoryManager.resetForTesting()

            XCTAssertTrue(moved.isEmpty, linkCase)
            XCTAssertEqual(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: modelsURL), "weights", linkCase)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath), sourceRootURL.path, linkCase)
        }
    }

    /// Released leases and Finder metadata leave files beside models in both roots; they are not models to move.
    func testLockFilesAndFinderMetadataInBothRootsDoNotBlockMigration() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
        let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
        let destinationRootURL = scratch.appendingPathComponent("model-storage", isDirectory: true)
        for rootURL in [sourceRootURL, destinationRootURL] {
            let modelURL = rootURL.appendingPathComponent("mlx-audio/org_qwen", isDirectory: true)
            try writeFixture("{}", to: modelURL.appendingPathComponent("config.json"))
            try writeFixture("weights", to: modelURL.appendingPathComponent("weights.safetensors"))
            _ = try OmniModelDirectoryLease.acquire(for: modelURL, mode: .sharedRead)
        }
        try writeFixture("source llm", to: sourceRootURL.appendingPathComponent("mlx-llm/org_llm/weights.safetensors"))
        for rootURL in [sourceRootURL, destinationRootURL] {
            try writeFixture("finder", to: rootURL.appendingPathComponent("mlx-audio/.DS_Store"))
        }
        try writeFixture("finder", to: sourceRootURL.appendingPathComponent("mlx-llm/.DS_Store"))
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)
        defer { ModelStorageDirectoryManager.resetForTesting() }

        ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
            defaults: defaults,
            downloadsURL: downloadsURL,
            destinationRootURL: destinationRootURL
        )

        XCTAssertEqual(fixtureText(at: "mlx-llm/org_llm/weights.safetensors", under: destinationRootURL), "source llm")
        XCTAssertNil(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath))
    }

    /// A Whisper model stored as weights.npz was judged incomplete, which undid every migration it was part of.
    func testDuplicateWhisperModelWithNpzWeightsDoesNotBlockMigration() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
        let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
        let destinationRootURL = scratch.appendingPathComponent("model-storage", isDirectory: true)
        let whisperRepo = "mlx-community/whisper-large-v3-turbo"
        let whisperFolder = "mlx-audio/mlx-community_whisper-large-v3-turbo"
        let destinationWhisperURL = destinationRootURL.appendingPathComponent(whisperFolder)
        try writeFixture(#"{"model_type":"whisper"}"#, to: destinationWhisperURL.appendingPathComponent("config.json"))
        try writeFixture("npz weights", to: destinationWhisperURL.appendingPathComponent("weights.npz"))
        for assetPath in MLXModelDownloadSupport.whisperTokenizerAssetPaths {
            try writeFixture("asset", to: destinationWhisperURL.appendingPathComponent(assetPath))
        }
        XCTAssertTrue(
            MLXModelDownloadSupport.isModelDirectoryValid(destinationWhisperURL, repo: whisperRepo, fileManager: .default)
        )
        try writeFixture(#"{"model_type":"whisper"}"#, to: sourceRootURL.appendingPathComponent(whisperFolder + "/config.json"))
        try writeFixture("source qwen", to: sourceRootURL.appendingPathComponent("mlx-audio/org_qwen/weights.safetensors"))
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)
        defer { ModelStorageDirectoryManager.resetForTesting() }

        ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
            defaults: defaults,
            downloadsURL: downloadsURL,
            destinationRootURL: destinationRootURL
        )

        XCTAssertEqual(fixtureText(at: "mlx-audio/org_qwen/weights.safetensors", under: destinationRootURL), "source qwen")
        XCTAssertNil(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath))
    }

    /// Without folder access every model looks absent, which must not clear the saved root.
    func testUnauthorizedStorageRootInsideDownloadsIsKept() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let downloadsURL = scratch.appendingPathComponent("Downloads", isDirectory: true)
        let sourceRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        defer { ModelStorageDirectoryManager.resetForTesting() }

        let moved = ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
            defaults: defaults,
            downloadsURL: downloadsURL,
            destinationRootURL: scratch.appendingPathComponent("model-storage", isDirectory: true)
        )

        XCTAssertTrue(moved.isEmpty)
        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath), sourceRootURL.path)
    }

    func testStorageRootOutsideDownloadsIsNeitherMovedNorReset() throws {
        let scratch = try TestDoubles.canonicalTemporaryDirectory()
            .appendingPathComponent("voxt-downloads-migration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let sourceRootURL = scratch.appendingPathComponent("External/Voxt Models", isDirectory: true)
        let weightsURL = sourceRootURL.appendingPathComponent("mlx-audio/org_qwen/weights.safetensors")
        try writeFixture("weights", to: weightsURL)
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(sourceRootURL.path, forKey: AppPreferenceKey.modelStorageRootPath)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(sourceRootURL)
        defer { ModelStorageDirectoryManager.resetForTesting() }

        let moved = ModelStorageDirectoryManager.migrateRootOutOfDownloadsIfNeeded(
            defaults: defaults,
            downloadsURL: scratch.appendingPathComponent("Downloads", isDirectory: true),
            destinationRootURL: scratch.appendingPathComponent("model-storage", isDirectory: true)
        )

        XCTAssertTrue(moved.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: weightsURL.path))
        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath), sourceRootURL.path)
    }

    private func writeFixture(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    private func fixtureText(at path: String, under root: URL) -> String? {
        (try? Data(contentsOf: root.appendingPathComponent(path))).map { String(decoding: $0, as: UTF8.self) }
    }

    /// Choosing the saved Downloads folder again is how a user restores access so its models can move out.
    func testOnlyTheSavedDownloadsFolderCanBeChosenAgain() {
        let downloadsURL = URL(fileURLWithPath: "/Users/someone/Downloads", isDirectory: true)
        let savedRootURL = downloadsURL.appendingPathComponent("Voxt Models", isDirectory: true)

        XCTAssertEqual(
            ModelStorageDirectoryManager.selectionIssue(
                for: downloadsURL.appendingPathComponent("Other Models"),
                savedRootPath: savedRootURL.path,
                downloadsURL: downloadsURL
            ),
            .insideDownloads(path: downloadsURL.path)
        )
        XCTAssertEqual(
            ModelStorageDirectoryManager.selectionIssue(
                for: savedRootURL,
                savedRootPath: nil,
                downloadsURL: downloadsURL
            ),
            .insideDownloads(path: downloadsURL.path)
        )
        XCTAssertNil(
            ModelStorageDirectoryManager.selectionIssue(
                for: savedRootURL,
                savedRootPath: savedRootURL.path,
                downloadsURL: downloadsURL
            )
        )
        XCTAssertNil(
            ModelStorageDirectoryManager.selectionIssue(
                for: URL(fileURLWithPath: "/Volumes/External/Voxt Models", isDirectory: true),
                savedRootPath: nil,
                downloadsURL: downloadsURL
            )
        )
    }

    /// Deleting an installed model while Voxt ran left it reported as installed, so every
    /// activation failed to start the server instead of reporting the missing files.
    func testLoadAfterInstalledFilesWereDeletedReportsTheModelAsNotDownloaded() async throws {
        try XCTSkipUnless(OmniASRBackend.isAppleSilicon, "Native ASR loads only run on Apple Silicon.")
        try await withIsolatedModelStorageRoot { root in
            let repo = MLXModelManager.defaultModelRepo
            try seedValidMLXModelDirectory(repo: repo, root: root)
            let manager = MLXModelManager(modelRepo: repo)
            _ = try await manager.refreshInstallation(repo: repo)
            XCTAssertTrue(manager.isModelDownloaded(repo: repo))
            try FileManager.default.removeItem(at: root.appendingPathComponent("mlx-audio"))

            do {
                _ = try await manager.loadModel()
                XCTFail("Loading a deleted model succeeded")
            } catch {
                XCTAssertEqual(manager.state, .notDownloaded)
                XCTAssertFalse(manager.isModelDownloaded(repo: repo))
            }
            await manager.shutdownForApplicationTermination()
        }
    }
}
