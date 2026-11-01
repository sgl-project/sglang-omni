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
}
