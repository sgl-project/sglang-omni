// MLXModelManager.swift
// Provides MLXModel Manager for transcription engines.

import Foundation
import Combine
import CFNetwork
import MLX
import MLXAudioCore
import MLXAudioSTT
import HuggingFace

@MainActor
class MLXModelManager: ObservableObject {
    static let defaultHubBaseURL = URL(string: "https://huggingface.co")!
    static let mirrorHubBaseURL = URL(string: "https://hf-mirror.com")!
    static let hubUserAgent = "Voxt/1.0 (MLXAudio)"
    private enum DownloadStopAction {
        case pause
        case cancel
    }

    typealias ModelOption = MLXModelCatalog.Option

    nonisolated static let defaultModelRepo = MLXModelCatalog.defaultModelRepo
    nonisolated static let availableModels = MLXModelCatalog.availableModels
    nonisolated static let supportedModels = MLXModelCatalog.supportedModels

    @Published private(set) var state: ModelState = .notDownloaded
    private(set) var stateByRepo: [String: ModelState] = [:]
    @Published private(set) var sizeState: ModelSizeState = .unknown
    @Published private(set) var remoteSizeTextByRepo: [String: String] = [:]
    @Published private(set) var pausedStatusMessage: String?
    private(set) var pausedStatusMessageByRepo: [String: String] = [:]
    @Published private(set) var activeDownloadRepos: Set<String> = []

    private var downloadedStateByRepo: [String: Bool] = [:]
    private let installationCache = ModelInstallationCache()
    @Published private(set) var installationRevision: UInt64 = 0
    private var resumableDownloadStateByRepo: [String: Bool] = [:]
    private var localSizeTextByRepo: [String: String] = [:]
    private var modelRepo: String
    private var hubBaseURL: URL
    private let omniLedger = OmniRuntimeLedger()
    private var loadedModel: LoadedASRModel? {
        didSet {
            // Any path that drops or replaces an Omni runtime retires its server.
            if let previous = oldValue?.omniRuntime, previous !== loadedModel?.omniRuntime {
                omniLedger.release(previous)
            }
            // Observe the state transition so model switching/deletion cannot bypass
            // the delayed cleanup that was originally wired only to idle timeout.
            guard ModelUnloadReclamationNotificationPolicy.shouldNotify(
                wasLoaded: oldValue != nil,
                isLoaded: loadedModel != nil,
                isApplicationTerminating: isShuttingDownForApplicationTermination
            ) else { return }
            onModelUnloaded?()
        }
    }
    private var loadedRepo: String?
    private let modelLoadCoordinator = SharedModelLoadCoordinator<MLXLoadedModelBox>()
    private let modelLoadingOverride: (@Sendable (String) async throws -> MLXLoadedModelBox)?
    private var downloadTasksByRepo: [String: Task<Void, Never>] = [:]
    private var downloadStopActionsByRepo: [String: DownloadStopAction] = [:]
    private var idleUnloadTask: Task<Void, Never>?
    private var shutdownTask: Task<Void, Never>?
    private let downloadSizeTolerance: Double = 0.9
    private var activeUseCount = 0
    private var deletingRepos: Set<String> = []
    private var storageRevision = UUID()
    private var storageRoots: [URL] = []
    private var downloadStorageRevisions: [String: UUID] = [:]
    private var activeUseWaiters: [CheckedContinuation<Void, Never>] = []
    private var isShuttingDownForApplicationTermination = false
    var onModelUnloaded: (() -> Void)?
    private var resolvedIdleUnloadDelay: Duration {
        .seconds(AppPreferenceKey.resolvedLocalModelIdleUnloadDelaySeconds())
    }

    init(
        modelRepo: String,
        hubBaseURL: URL = URL(string: "https://huggingface.co")!,
        modelLoadingOverride: (@Sendable (String) async throws -> MLXLoadedModelBox)? = nil
    ) {
        self.modelRepo = Self.canonicalModelRepo(modelRepo)
        self.hubBaseURL = hubBaseURL
        self.modelLoadingOverride = modelLoadingOverride
        self.remoteSizeTextByRepo = MLXModelStorageSupport.loadPersistedRemoteSizeCache()
        storageRoots = [writeRootURL(), derivedRootURL()] + readableRootURLs()
        installationCache.onChange = { [weak self] repo, snapshot in
            guard let self else { return }
            self.downloadedStateByRepo[repo] = snapshot.isInstalled
            self.resumableDownloadStateByRepo[repo] = snapshot.hasPartialDownload
            self.localSizeTextByRepo[repo] = snapshot.allocatedBytes > 0
                ? MLXModelStorageSupport.formatByteCount(snapshot.allocatedBytes) : nil
            if self.downloadTasksByRepo[repo] == nil {
                if self.loadedRepo == repo { self.setState(.ready, for: repo) }
                else if !self.modelLoadCoordinator.hasPendingLoad { self.applyInstallationState(snapshot, repo: repo) }
            }
            self.installationRevision &+= 1
        }
        checkExistingModel()
    }

    var currentModelRepo: String { modelRepo }
    var isCurrentModelLoaded: Bool { loadedModel != nil && loadedRepo == modelRepo }
    var hasLoadedModel: Bool { loadedModel != nil }
    var hasActiveUse: Bool { activeUseCount > 0 }
    var hasPendingModelLoad: Bool { modelLoadCoordinator.hasPendingLoad }
    var hasOutstandingModelLoad: Bool { modelLoadCoordinator.hasOutstandingLoad }

    func refreshMemoryOptimizationPolicy() {
        guard loadedModel != nil else {
            cancelIdleUnloadTask()
            return
        }
        guard activeUseCount == 0 else { return }
        scheduleIdleUnloadIfNeeded()
    }

    func isModelDownloaded(repo: String) -> Bool {
        let repo = Self.canonicalModelRepo(repo)
        guard Self.isManagedArtifactRepo(repo) else { return false }
        requestInstallation(repo)
        return downloadedStateByRepo[repo] ?? false
    }

    func isCheckingInstallation(repo: String) -> Bool {
        let repo = Self.canonicalModelRepo(repo)
        guard Self.isManagedArtifactRepo(repo) else { return false }
        requestInstallation(repo)
        return downloadedStateByRepo[repo] == nil
    }

    @discardableResult
    func refreshInstallation(repo: String) async throws -> ModelInstallationSnapshot {
        let repo = Self.canonicalModelRepo(repo)
        guard !isShuttingDownForApplicationTermination, !deletingRepos.contains(repo),
              Self.isManagedArtifactRepo(repo) else { throw CancellationError() }
        let request = installationRequest(repo)
        return try await installationCache.value(repo) { request.scan() }
    }

    func hasResumableDownload(repo: String) -> Bool {
        let canonicalRepo = Self.canonicalModelRepo(repo)
        let isDownloaded = isModelDownloaded(repo: canonicalRepo)
        return hasResumableDownload(repo: canonicalRepo, isDownloaded: isDownloaded)
    }

    func modelSizeOnDisk(repo: String) -> String {
        let repo = Self.canonicalModelRepo(repo)
        requestInstallation(repo)
        return localSizeTextByRepo[repo] ?? ""
    }

    func cachedModelSizeText(repo: String) -> String? {
        let canonicalRepo = Self.canonicalModelRepo(repo)
        return localSizeTextByRepo[canonicalRepo]
    }

    func modelDirectoryURL(repo: String) -> URL? {
        let repo = Self.canonicalModelRepo(repo)
        requestInstallation(repo)
        let snapshot = installationCache.peek(repo)
        return snapshot?.directory ?? snapshot?.existingDirectory
    }

    func ensureModelDirectory(repo: String) async throws -> URL {
        guard !isShuttingDownForApplicationTermination else { throw CancellationError() }
        let canonicalRepo = Self.canonicalModelRepo(repo)
        let installation = try await refreshInstallation(repo: canonicalRepo)
        if let modelDir = installation.directory {
            return modelDir
        }
        return try await performDownloadWithFallback(for: canonicalRepo)
    }

    @discardableResult
    func deleteModel(repo: String) async -> Result<Void, Error> {
        let repo = Self.canonicalModelRepo(repo)
        guard !hasActiveUse, !deletingRepos.contains(repo) else {
            return .failure(NSError(domain: "Voxt.MLXModelManager", code: 1005,
                userInfo: [NSLocalizedDescriptionKey: "The model is currently in use. Try again when transcription finishes."]))
        }
        deletingRepos.insert(repo)
        defer { deletingRepos.remove(repo) }
        let roots = readableRootURLs()
        let directories = roots.compactMap { MLXModelStorageSupport.cacheDirectory(for: repo, rootDirectory: $0) }
            + [writableShadowDirectory(for: repo), downloadTempDirectory(for: repo)].compactMap { $0 }
        invalidateLocalCache(for: repo)
        let download = downloadTasksByRepo[repo]
        if download != nil { cancelDownload(repo: repo) }
        let loads = invalidatePendingModelLoad(reason: "model-deleted")
        await download?.value
        for task in loads { await task.waitForCompletion() }
        if loadedRepo == repo {
            loadedModel = nil
            loadedRepo = nil
        }
        await omniLedger.waitForRetirements()
        do {
            try await ModelDiskOperations.remove(directories)
            await Task.detached(priority: .utility) {
                if let id = Repo.ID(rawValue: repo) {
                    for root in roots { MLXModelStorageSupport.clearHubCache(for: id, rootDirectory: root) }
                }
            }.value
            invalidateLocalCache(for: repo)
            clearSelectedDownloadSource(for: repo)
            clearPerRepoState(for: repo)
            setState(.notDownloaded, for: repo)
            requestInstallation(repo)
            return .success(())
        } catch {
            invalidateLocalCache(for: repo)
            setState(.error("Couldn't uninstall MLX model: \(error.localizedDescription)"), for: repo)
            return .failure(error)
        }
    }

    func downloadModel(repo: String) async {
        guard !isShuttingDownForApplicationTermination else { return }
        let canonicalRepo = Self.canonicalModelRepo(repo)
        await performDownload(forRepo: canonicalRepo)
    }

    func cancelDownload(repo: String) {
        let canonicalRepo = Self.canonicalModelRepo(repo)
        if let task = downloadTasksByRepo[canonicalRepo] {
            downloadStopActionsByRepo[canonicalRepo] = .cancel
            setPausedStatusMessage(nil, for: canonicalRepo)
            setState(.notDownloaded, for: canonicalRepo)
            clearSelectedDownloadSource(for: canonicalRepo)
            task.cancel()
            return
        }

        schedulePartialCleanup(for: canonicalRepo)
    }

    func cancelDownloadAndWait(repo: String) async {
        cancelDownload(repo: repo)
        await downloadTasksByRepo[Self.canonicalModelRepo(repo)]?.value
    }

    private func schedulePartialCleanup(for repo: String) {
        guard !isModelDownloaded(repo: repo), !deletingRepos.contains(repo) else { return }
        let directories = [writeCacheDirectory(for: repo), downloadTempDirectory(for: repo)].compactMap { $0 }
        let root = writeRootURL()
        let revision = storageRevision
        invalidateLocalCache(for: repo)
        clearSelectedDownloadSource(for: repo)
        setPausedStatusMessage(nil, for: repo)
        setState(.notDownloaded, for: repo)
        activeDownloadRepos.insert(repo)
        downloadStopActionsByRepo[repo] = .cancel
        let task = Task { [weak self] in
            guard let self else { return }
            defer {
                downloadTasksByRepo[repo] = nil
                downloadStopActionsByRepo[repo] = nil
                activeDownloadRepos.remove(repo)
                invalidateLocalCache(for: repo)
                requestInstallation(repo)
            }
            do {
                try await ModelDiskOperations.remove(directories)
                try await ModelDiskOperations.perform {
                    if let id = Repo.ID(rawValue: repo) {
                        MLXModelStorageSupport.clearHubCache(for: id, rootDirectory: root)
                    }
                }
                guard revision == storageRevision else { return }
                setState(.notDownloaded, for: repo)
            } catch {
                guard revision == storageRevision else { return }
                setState(.error("Couldn't remove partial model files: \(error.localizedDescription)"), for: repo)
            }
        }
        downloadTasksByRepo[repo] = task
    }

    func updateModel(repo: String) {
        let canonicalRepo = Self.canonicalModelRepo(repo)
        guard canonicalRepo != modelRepo else { return }
        cancelIdleUnloadTask()
        invalidatePendingModelLoad(reason: "model-updated")
        modelRepo = canonicalRepo
        loadedModel = nil
        loadedRepo = nil
        Memory.clearCache()
        checkExistingModel()
        fetchRemoteSize()
    }

    var currentTranscriptionBehavior: TranscriptionBehavior {
        Self.transcriptionBehavior(for: modelRepo)
    }

    func updateHubBaseURL(_ url: URL) {
        guard url != hubBaseURL else { return }
        hubBaseURL = url
        fetchRemoteSize()
    }

    func checkExistingModel(refresh: Bool = false) {
        guard !isShuttingDownForApplicationTermination else { return }
        if refresh, !installationCache.hasPendingRequest(modelRepo), downloadTasksByRepo[modelRepo] == nil {
            invalidateLocalCache(for: modelRepo)
        }
        guard Self.isManagedArtifactRepo(modelRepo) else {
            setState(.notDownloaded, for: modelRepo)
            return
        }
        requestInstallation(modelRepo)
        guard downloadTasksByRepo[modelRepo] == nil else { return }
        guard let snapshot = installationCache.peek(modelRepo) else {
            if loadedRepo == modelRepo, loadedModel != nil { setState(.ready, for: modelRepo) }
            else if !hasPendingModelLoad { setState(.loading, for: modelRepo) }
            return
        }
        if loadedRepo == modelRepo { setState(.ready, for: modelRepo); return }
        guard !hasPendingModelLoad else { return }
        applyInstallationState(snapshot, repo: modelRepo)
    }

    private func applyInstallationState(_ snapshot: ModelInstallationSnapshot, repo: String) {
        if snapshot.isInstalled {
            setState(.downloaded, for: repo)
        } else if snapshot.hasPartialDownload {
            setPausedState(progress: 0, completed: 0, total: 0, currentFile: nil, completedFiles: 0, totalFiles: 0, for: repo)
        } else {
            setState(.notDownloaded, for: repo)
        }
    }

    func state(for repo: String) -> ModelState {
        let canonicalRepo = Self.canonicalModelRepo(repo)
        return catalogSnapshot(for: canonicalRepo).state
    }

    func catalogSnapshot(for repo: String) -> CatalogSnapshot {
        let canonicalRepo = Self.canonicalModelRepo(repo)
        let isDownloaded = isModelDownloaded(repo: canonicalRepo)
        let hasResumableDownload = hasResumableDownload(repo: canonicalRepo, isDownloaded: isDownloaded)
        let resolvedState = MLXModelPerRepoStateSupport.resolvedState(
            for: canonicalRepo,
            currentRepo: modelRepo,
            currentState: state,
            storedStates: stateByRepo,
            isDownloaded: { _ in isDownloaded },
            hasResumableDownload: { _ in hasResumableDownload }
        )
        return CatalogSnapshot(
            repo: canonicalRepo,
            isDownloaded: isDownloaded,
            hasResumableDownload: hasResumableDownload,
            state: resolvedState,
            pausedStatusMessage: pausedStatusMessage(for: canonicalRepo),
            hasActiveDownloadTask: downloadTasksByRepo[canonicalRepo] != nil
        )
    }

    func pausedStatusMessage(for repo: String) -> String? {
        MLXModelPerRepoStateSupport.pausedStatusMessage(
            for: repo,
            storedMessages: pausedStatusMessageByRepo
        )
    }

    func isDownloading(repo: String) -> Bool {
        let canonicalRepo = Self.canonicalModelRepo(repo)
        if downloadTasksByRepo[canonicalRepo] != nil { return true }
        if case .downloading = state(for: canonicalRepo) { return true }
        return false
    }

    func isPaused(repo: String) -> Bool {
        if case .paused = state(for: repo) { return true }
        return false
    }

    func isDownloadOperationActive(repo: String) -> Bool {
        switch state(for: repo) {
        case .downloading, .paused:
            return true
        default:
            return false
        }
    }

    private func setState(_ newState: ModelState, for repo: String) {
        MLXModelPerRepoStateSupport.applyState(
            newState,
            for: repo,
            currentRepo: modelRepo,
            currentState: &state,
            storedStates: &stateByRepo
        )
    }

    private func setPausedStatusMessage(_ message: String?, for repo: String) {
        MLXModelPerRepoStateSupport.applyPausedStatusMessage(
            message,
            for: repo,
            currentRepo: modelRepo,
            currentMessage: &pausedStatusMessage,
            storedMessages: &pausedStatusMessageByRepo
        )
    }

    private func clearPerRepoState(for repo: String) {
        MLXModelPerRepoStateSupport.clearState(
            for: repo,
            currentRepo: modelRepo,
            currentPausedStatusMessage: &pausedStatusMessage,
            storedStates: &stateByRepo,
            storedMessages: &pausedStatusMessageByRepo
        )
    }

    private func downloadSourceTargetKey(for repo: String) -> String {
        ModelDownloadSourceSelectionStore.targetKey(namespace: "mlx-audio", identifier: repo)
    }

    private func clearSelectedDownloadSource(for repo: String) {
        ModelDownloadSourceSelectionStore.clearSourceID(for: downloadSourceTargetKey(for: repo))
    }

    func refreshStorageRoot() {
        let nextRoots = [writeRootURL(), derivedRootURL()] + readableRootURLs()
        if nextRoots != storageRoots {
            storageRoots = nextRoots
            storageRevision = UUID()
            // Old work may finish after cancellation. Its captured revision must
            // not publish into, clean up, or start a fallback in the new root.
            for (repo, task) in downloadTasksByRepo {
                downloadStopActionsByRepo[repo] = .pause
                task.cancel()
            }
            invalidatePendingModelLoad(reason: "storage-root-changed")
            loadedModel = nil
            loadedRepo = nil
        }
        downloadedStateByRepo.removeAll()
        installationCache.invalidateAll()
        installationRevision &+= 1
        resumableDownloadStateByRepo.removeAll()
        localSizeTextByRepo.removeAll()
        MLXModelPerRepoStateSupport.resetStorageRootState(
            currentPausedStatusMessage: &pausedStatusMessage,
            storedStates: &stateByRepo,
            storedMessages: &pausedStatusMessageByRepo
        )
        checkExistingModel()
    }

    func downloadModel() async {
        await performDownload(forRepo: modelRepo)
    }

    private func performDownload(forRepo canonicalRepo: String) async {
        if downloadTasksByRepo[canonicalRepo] != nil || deletingRepos.contains(canonicalRepo) { return }
        do {
            let installation = try await refreshInstallation(repo: canonicalRepo)
            if installation.isInstalled { return }
        } catch { return }
        guard downloadTasksByRepo[canonicalRepo] == nil, !isShuttingDownForApplicationTermination,
              !deletingRepos.contains(canonicalRepo) else { return }
        if hasPendingModelLoad { return }
        installationCache.invalidate(canonicalRepo)

        SystemNotificationSupport.requestAuthorizationIfNeeded()
        resumableDownloadStateByRepo.removeValue(forKey: canonicalRepo)
        let revision = storageRevision
        downloadStorageRevisions[canonicalRepo] = revision
        let task = Task { [weak self] in
            guard let self else { return }
            defer {
                self.downloadTasksByRepo[canonicalRepo] = nil
                self.downloadStopActionsByRepo[canonicalRepo] = nil
                self.activeDownloadRepos.remove(canonicalRepo)
                downloadStorageRevisions.removeValue(forKey: canonicalRepo)
                if revision != storageRevision {
                    invalidateLocalCache(for: canonicalRepo)
                    clearPerRepoState(for: canonicalRepo)
                    requestInstallation(canonicalRepo)
                }
            }
            guard revision == storageRevision, !Task.isCancelled else { return }
            activeDownloadRepos.insert(canonicalRepo)
            if let pausedState = pausedDownloadSnapshot(for: canonicalRepo) {
                setDownloadingState(
                    progress: pausedState.progress,
                    completed: pausedState.completed,
                    total: pausedState.total,
                    currentFile: pausedState.currentFile,
                    completedFiles: pausedState.completedFiles,
                    totalFiles: pausedState.totalFiles,
                    for: canonicalRepo
                )
            } else {
                setDownloadingState(
                    progress: 0,
                    completed: 0,
                    total: 0,
                    currentFile: nil,
                    completedFiles: 0,
                    totalFiles: 0,
                    for: canonicalRepo
                )
            }
            do {
                setPausedStatusMessage(nil, for: canonicalRepo)
                let modelDir = try await performDownloadWithFallback(for: canonicalRepo)
                try Task.checkCancellation()
                try await MLXModelDownloadSupport.validateDownloadedModelInBackground(
                    at: modelDir,
                    repo: canonicalRepo,
                    sizeState: MLXModelDownloadSupport.validationSizeState(for: canonicalRepo),
                    downloadSizeTolerance: downloadSizeTolerance,
                    fileManager: .default
                )
                try Task.checkCancellation()
                guard revision == storageRevision else { throw CancellationError() }
                markDownloadCompleted(for: canonicalRepo)
                SystemNotificationSupport.postModelDownloadSucceeded(
                    modelName: displayTitle(for: canonicalRepo)
                )
                VoxtLog.modelInfo("Download complete. repo=\(canonicalRepo)")
            } catch is CancellationError {
                guard revision == storageRevision else { return }
                switch downloadStopActionsByRepo[canonicalRepo] {
                case .pause:
                    setPausedStatusMessage(nil, for: canonicalRepo)
                    VoxtLog.modelInfo("Download paused. repo=\(canonicalRepo)")
                case .cancel, .none:
                    setPausedStatusMessage(nil, for: canonicalRepo)
                    await cleanupPartialDownload(for: canonicalRepo)
                    guard revision == storageRevision else { return }
                    await clearHubCache(for: canonicalRepo)
                    guard revision == storageRevision else { return }
                    markCancelledDownloadUnavailable(for: canonicalRepo)
                    VoxtLog.modelInfo("Download cancelled. repo=\(canonicalRepo)")
                }
            } catch {
                guard revision == storageRevision else { return }
                if pauseDownloadIfNetworkIssue(error, repo: canonicalRepo) {
                    return
                }
                setPausedStatusMessage(nil, for: canonicalRepo)
                await clearHubCache(for: canonicalRepo)
                guard revision == storageRevision else { return }
                let message = downloadErrorMessage(for: error, repo: canonicalRepo)
                setState(.error(message), for: canonicalRepo)
                SystemNotificationSupport.postModelDownloadFailed(
                    modelName: displayTitle(for: canonicalRepo),
                    message: message
                )
                VoxtLog.modelError("Download error. repo=\(canonicalRepo), error=\(error.localizedDescription)")
            }
        }
        downloadTasksByRepo[canonicalRepo] = task
        await task.value
    }

    func pauseDownload() {
        pauseDownload(repo: modelRepo)
    }

    func pauseDownload(repo: String) {
        let canonicalRepo = Self.canonicalModelRepo(repo)
        guard let task = downloadTasksByRepo[canonicalRepo] else { return }
        downloadStopActionsByRepo[canonicalRepo] = .pause
        setPausedStatusMessage(nil, for: canonicalRepo)
        if let snapshot = downloadingSnapshot(for: canonicalRepo) {
            setPausedState(
                progress: snapshot.progress,
                completed: snapshot.completed,
                total: snapshot.total,
                currentFile: snapshot.currentFile,
                completedFiles: snapshot.completedFiles,
                totalFiles: snapshot.totalFiles,
                for: canonicalRepo
            )
        }
        task.cancel()
    }

    func cancelDownload() {
        cancelDownload(repo: modelRepo)
    }

    func cancelPendingModelLoadForApplicationTermination() {
        invalidatePendingModelLoad(reason: "application-terminating")
    }

    func loadModel() async throws -> LoadedASRModel {
        guard !isShuttingDownForApplicationTermination else { throw CancellationError() }
        guard !deletingRepos.contains(modelRepo) else { throw CancellationError() }
        cancelIdleUnloadTask()
        if let model = loadedModel, loadedRepo == modelRepo {
            if let runtime = model.omniRuntime, await !runtime.isServing {
                // The server died or was killed: start a new one rather than fail
                // every request until the idle unload drops this runtime.
                if loadedModel?.omniRuntime === runtime {
                    VoxtLog.modelWarning("Omni ASR server is no longer serving; starting a new one. repo=\(modelRepo)")
                    loadedModel = nil
                    loadedRepo = nil
                }
            } else {
                VoxtLog.modelInfo("MLX Audio model reuse existing instance. repo=\(modelRepo)", verbose: true)
                return model
            }
        }

        let repo = modelRepo
        let revision = storageRevision
        let startedAt = Date()
        VoxtLog.modelInfo("MLX Audio model load started. repo=\(repo)", verbose: true)
        setState(.loading, for: repo)
        let manager = self
        do {
            let modelBox = try await modelLoadCoordinator.value(for: repo) {
                let model = try await manager.loadSTTModel(for: repo)
                return MLXLoadedModelBox(loaded: model)
            }
            try Task.checkCancellation()
            guard modelRepo == repo, storageRevision == revision else { throw CancellationError() }
            loadedModel = modelBox.loaded
            omniLedger.adopt(modelBox.loaded.omniRuntime)
            loadedRepo = repo
            setState(.ready, for: repo)
            let model = try readyModel(for: repo)
            let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
            VoxtLog.modelInfo("MLX Audio model load completed. repo=\(repo), elapsedMs=\(elapsedMs)")
            return model
        } catch {
            if !modelLoadCoordinator.hasPendingLoad, !modelLoadCoordinator.hasOutstandingLoad {
                // No load will adopt a server started by a load that finished for
                // nobody (its waiters left, or the repo changed); stop it now.
                omniLedger.releaseUnadopted()
            }
            guard revision == storageRevision else { throw CancellationError() }
            let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
            if error is CancellationError || Task.isCancelled {
                if repo == modelRepo, !modelLoadCoordinator.hasPendingLoad {
                    checkExistingModel()
                }
                VoxtLog.modelInfo(
                    "MLX Audio model load cancelled. repo=\(repo), elapsedMs=\(elapsedMs)",
                    verbose: true
                )
            } else {
                if repo == modelRepo {
                    setState(.error("Model load failed: \(error.localizedDescription)"), for: repo)
                }
                VoxtLog.modelError("MLX Audio model load failed. repo=\(repo), elapsedMs=\(elapsedMs), error=\(error.localizedDescription)")
            }
            throw error
        }
    }

    @discardableResult
    func deleteModel() async -> Result<Void, Error> {
        await deleteModel(repo: modelRepo)
    }

    func beginActiveUse() {
        activeUseCount += 1
        cancelIdleUnloadTask()
    }

    func endActiveUse() {
        activeUseCount = max(0, activeUseCount - 1)
        guard activeUseCount == 0 else { return }
        resumeActiveUseWaiters()
        scheduleIdleUnloadIfNeeded()
    }

    /// Releases an idle model at an explicit stage boundary. Callers must have
    /// already ended their active-use lease; active inference is never forced out.
    func releaseLoadedModelIfIdle(reason: String) {
        guard activeUseCount == 0 else { return }
        unloadLoadedModelIfIdle(expectedRepo: loadedRepo, reason: reason)
    }

    func shutdownForApplicationTermination() async {
        if let shutdownTask { await shutdownTask.value; return }
        isShuttingDownForApplicationTermination = true
        let task = Task { @MainActor [self] in await performApplicationTerminationShutdown() }
        shutdownTask = task
        await task.value
    }

    private func performApplicationTerminationShutdown() async {
        installationCache.invalidateAll()

        let downloadTasks = Array(downloadTasksByRepo.values)
        for repo in Array(downloadTasksByRepo.keys) {
            pauseDownload(repo: repo)
        }
        let loadTasks = invalidatePendingModelLoad(reason: "application-terminating")
        cancelIdleUnloadTask()

        for task in downloadTasks {
            await task.value
        }
        for task in loadTasks {
            await task.waitForCompletion()
        }
        await waitForActiveUsesToFinish()

        loadedModel = nil
        loadedRepo = nil
        await omniLedger.retireAll()
        Memory.clearCache()
        VoxtLog.modelInfo("MLX Audio model released for application termination.", verbose: true)
    }

    private func waitForActiveUsesToFinish() async {
        guard activeUseCount > 0 else { return }
        await withCheckedContinuation { continuation in
            activeUseWaiters.append(continuation)
        }
    }

    private func resumeActiveUseWaiters() {
        let waiters = activeUseWaiters
        activeUseWaiters.removeAll()
        for waiter in waiters {
            waiter.resume()
        }
    }

    @discardableResult
    private func invalidatePendingModelLoad(reason: String) -> [SharedModelLoadTask] {
        let tasks = modelLoadCoordinator.cancelAll()
        guard !tasks.isEmpty else { return [] }
        // Servers these loads are starting are stopped now, so a following
        // load never waits behind, or overlaps, a model nobody will use.
        omniLedger.releaseUnadopted()
        VoxtLog.modelInfo("MLX Audio pending model load invalidated. reason=\(reason)", verbose: true)
        return tasks
    }

    var modelSizeOnDisk: String {
        modelSizeOnDisk(repo: modelRepo)
    }

    private func invalidateLocalCache(for repo: String) {
        installationCache.invalidate(repo)
        installationRevision &+= 1
        downloadedStateByRepo.removeValue(forKey: repo)
        resumableDownloadStateByRepo.removeValue(forKey: repo)
        localSizeTextByRepo.removeValue(forKey: repo)
    }

    private func markDownloadCompleted(for repo: String) {
        installationCache.invalidate(repo)
        requestInstallation(repo)
        downloadedStateByRepo[repo] = true
        resumableDownloadStateByRepo[repo] = false
        localSizeTextByRepo.removeValue(forKey: repo)
        if repo == modelRepo {
            checkExistingModel()
        } else {
            setState(.downloaded, for: repo)
        }
    }

    private func markCancelledDownloadUnavailable(for repo: String) {
        invalidateLocalCache(for: repo)
        if repo == modelRepo {
            checkExistingModel()
        } else {
            setState(.notDownloaded, for: repo)
        }
    }

    private func installationRequest(_ repo: String) -> ModelInstallationRequest {
        let directories = readableRootURLs().compactMap { MLXModelStorageSupport.cacheDirectory(for: repo, rootDirectory: $0) }
        return ModelInstallationRequest(
            directories: directories,
            partialDirectories: [downloadTempDirectory(for: repo)].compactMap { $0 },
            validate: { MLXModelDownloadSupport.isModelDirectoryValid($0, repo: repo, fileManager: .default) }
        )
    }

    private func requestInstallation(_ repo: String) {
        guard !isShuttingDownForApplicationTermination, !deletingRepos.contains(repo), installationCache.needsRequest(repo) else { return }
        guard Self.isManagedArtifactRepo(repo) else { return }
        let request = installationRequest(repo)
        installationCache.request(repo) { request.scan() }
    }

    private func readyModel(for repo: String) throws -> LoadedASRModel {
        guard let model = loadedModel, loadedRepo == repo else {
            throw NSError(
                domain: "Voxt.MLXModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model load finished without a ready model instance."]
            )
        }
        return model
    }

    private func loadSTTModel(for repo: String) async throws -> LoadedASRModel {
        if let modelLoadingOverride {
            return try await modelLoadingOverride(repo).loaded
        }
        try Task.checkCancellation()
        let lower = repo.lowercased()
        let sourceModelDir: URL
        let installation = try await refreshInstallation(repo: repo)
        if let validDirectory = installation.directory {
            sourceModelDir = validDirectory
        } else if let existingDirectory = installation.existingDirectory {
            sourceModelDir = try await repairIncompleteModelDirectoryIfNeeded(
                for: repo,
                existingDirectory: existingDirectory
            )
            let repairedDirectory = sourceModelDir
            let valid = await Task.detached(priority: .utility) {
                MLXModelDownloadSupport.isModelDirectoryValid(repairedDirectory, repo: repo, fileManager: .default)
            }.value
            guard valid else {
                throw NSError(
                    domain: "MLXModelManager",
                    code: 1004,
                    userInfo: [NSLocalizedDescriptionKey: "MLX model is installed incompletely. Please download it again."]
                )
            }
        } else {
            throw NSError(
                domain: "MLXModelManager",
                code: 1004,
                userInfo: [NSLocalizedDescriptionKey: "MLX model is not installed locally."]
            )
        }
        try Task.checkCancellation()
        if let kind = OmniASRBackend.modelKind(for: repo),
           let configuration = OmniASRBackend.configuration() {
            // One server at a time: earlier runtimes stop before this one starts.
            await omniLedger.waitForRetirements()
            try Task.checkCancellation()
            let runtime = OmniASRRuntime(kind: kind, modelDirectory: sourceModelDir, configuration: configuration)
            omniLedger.track(runtime)
            do {
                let ledger = omniLedger
                _ = try await withTaskCancellationHandler {
                    try await runtime.prepare()
                } onCancel: {
                    // Every waiter left: stop the launch instead of finishing it.
                    Task { @MainActor in ledger.release(runtime) }
                }
                try Task.checkCancellation()
            } catch {
                omniLedger.release(runtime)
                throw error
            }
            VoxtLog.modelInfo("Omni ASR server ready. repo=\(repo), kind=\(kind.rawValue)")
            return .omni(runtime)
        }
        let modelDir = try await writableLoadDirectoryIfNeeded(
            for: repo,
            sourceDirectory: sourceModelDir,
            lowercasedRepo: lower
        )
        let modelLoadTask = Task.detached(priority: .userInitiated) {
            try await MLXSTTModelLoader.load(repo: repo, directory: modelDir)
        }
        let loaded = try await withTaskCancellationHandler {
            try await modelLoadTask.value
        } onCancel: {
            modelLoadTask.cancel()
        }
        return loaded.loaded
    }

    private func writeCacheDirectory(for repo: String) -> URL? {
        MLXModelStorageSupport.cacheDirectory(
            for: repo,
            rootDirectory: writeRootURL()
        )
    }

    private func downloadTempDirectory(for repo: String) -> URL? {
        guard let repoID = Repo.ID(rawValue: repo) else { return nil }
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        return writeRootURL()
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent("\(modelSubdir)-download")
    }

    func writableLoadDirectoryIfNeeded(
        for repo: String,
        sourceDirectory: URL,
        lowercasedRepo: String
    ) async throws -> URL {
        let writableDirectory = writableShadowDirectory(for: repo)
        return try await Task.detached(priority: .userInitiated) {
            guard lowercasedRepo.contains("qwen3-asr") || lowercasedRepo.contains("qwen3_asr"),
                  !FileManager.default.fileExists(atPath: sourceDirectory.appendingPathComponent("tokenizer.json").path),
                  let writableDirectory else { return sourceDirectory }
            return try Self.prepareWritableShadowDirectory(from: sourceDirectory, to: writableDirectory)
        }.value
    }

    private nonisolated static func prepareWritableShadowDirectory(from sourceDirectory: URL, to destinationDirectory: URL) throws -> URL {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: destinationDirectory.path) {
            try fileManager.removeItem(at: destinationDirectory)
        }
        try fileManager.createDirectory(
            at: destinationDirectory.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try fileManager.createDirectory(at: destinationDirectory, withIntermediateDirectories: true)
        let sourceEntries = try fileManager.contentsOfDirectory(
            at: sourceDirectory,
            includingPropertiesForKeys: nil,
            options: []
        )
        for entry in sourceEntries {
            let linkURL = destinationDirectory.appendingPathComponent(entry.lastPathComponent, isDirectory: false)
            try fileManager.createSymbolicLink(at: linkURL, withDestinationURL: entry)
        }
        return destinationDirectory
    }

    private func writeRootURL() -> URL {
        ModelStorageDirectoryManager.resolvedWriteRootURL()
    }

    private func derivedRootURL() -> URL {
        ModelStorageDirectoryManager.resolvedDerivedRootURL()
    }

    private func writableShadowDirectory(for repo: String) -> URL? {
        guard let repoID = Repo.ID(rawValue: repo) else { return nil }
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        return derivedRootURL()
            .appendingPathComponent("mlx-audio-shadow", isDirectory: true)
            .appendingPathComponent(modelSubdir)
    }

    private func readableRootURLs() -> [URL] {
        ModelStorageDirectoryManager.resolvedReadableRootURLs()
    }

    private func hasResumableDownload(repo: String, isDownloaded: Bool) -> Bool {
        guard !isDownloaded else { return false }
        requestInstallation(repo)
        return resumableDownloadStateByRepo[repo] ?? false
    }

    private func cleanupPartialDownload(for repo: String) async {
        let directories = [writeCacheDirectory(for: repo), downloadTempDirectory(for: repo)].compactMap { $0 }
        // This compensating cleanup is requested BY cancellation, so it must not
        // inherit the cancelled download task's flag. Paths are captured first.
        let result = await Task.detached(priority: .utility) {
            try await ModelDiskOperations.remove(directories)
        }.result
        if case .failure(let error) = result {
            VoxtLog.modelWarning("Partial model cleanup failed. repo=\(repo), error=\(error.localizedDescription)")
        }
    }

    private func downloadingSnapshot(for repo: String) -> (
        progress: Double,
        completed: Int64,
        total: Int64,
        currentFile: String?,
        completedFiles: Int,
        totalFiles: Int
    )? {
        guard case .downloading(
            let progress,
            let completed,
            let total,
            let currentFile,
            let completedFiles,
            let totalFiles
        ) = state(for: repo) else {
            return nil
        }
        return (progress, completed, total, currentFile, completedFiles, totalFiles)
    }

    private func pausedDownloadSnapshot(for repo: String) -> (
        progress: Double,
        completed: Int64,
        total: Int64,
        currentFile: String?,
        completedFiles: Int,
        totalFiles: Int
    )? {
        guard case .paused(
            let progress,
            let completed,
            let total,
            let currentFile,
            let completedFiles,
            let totalFiles
        ) = state(for: repo) else {
            return nil
        }
        return (progress, completed, total, currentFile, completedFiles, totalFiles)
    }

    private func setPausedState(
        progress: Double,
        completed: Int64,
        total: Int64,
        currentFile: String?,
        completedFiles: Int,
        totalFiles: Int,
        for repo: String
    ) {
        let nextState = ModelState.paused(
            progress: progress,
            completed: completed,
            total: total,
            currentFile: currentFile,
            completedFiles: completedFiles,
            totalFiles: totalFiles
        )
        setState(nextState, for: repo)
    }

    private func pauseDownloadIfNetworkIssue(_ error: Error, repo: String) -> Bool {
        guard let message = MLXModelDownloadSupport.pauseMessageForInterruptedDownload(error) else {
            return false
        }
        let snapshot = downloadingSnapshot(for: repo) ?? pausedDownloadSnapshot(for: repo)
        setPausedStatusMessage(message, for: repo)
        if let snapshot {
            setPausedState(
                progress: snapshot.progress,
                completed: snapshot.completed,
                total: snapshot.total,
                currentFile: snapshot.currentFile,
                completedFiles: snapshot.completedFiles,
                totalFiles: snapshot.totalFiles,
                for: repo
            )
        } else {
            setPausedState(
                progress: 0,
                completed: 0,
                total: 0,
                currentFile: nil,
                completedFiles: 0,
                totalFiles: 0,
                for: repo
            )
        }
        VoxtLog.modelWarning("Download auto-paused after network issue. repo=\(repo), error=\(error.localizedDescription)")
        return true
    }

    private func uniqueURLs(_ urls: [URL]) -> [URL] {
        var seenPaths = Set<String>()
        var uniqueURLs: [URL] = []
        for url in urls {
            let standardizedURL = url.standardizedFileURL
            if seenPaths.insert(standardizedURL.path).inserted {
                uniqueURLs.append(standardizedURL)
            }
        }
        return uniqueURLs
    }

    private func fetchRemoteSize() {
        sizeState = MLXModelDownloadSupport.validationSizeState(for: modelRepo)
    }

    func remoteSizeText(repo: String) -> String {
        let canonicalRepo = Self.canonicalModelRepo(repo)
        return Self.fallbackRemoteSizeText(repo: canonicalRepo) ?? "Unknown"
    }

    private func fallbackHubBaseURL(from baseURL: URL) -> URL? {
        guard !MLXModelDownloadSupport.isMirrorHost(baseURL) else { return nil }
        return Self.mirrorHubBaseURL
    }

    private func downloadSourceCandidates() -> [ModelDownloadSourceCandidate] {
        [
            ModelDownloadSourceCandidate(
                id: "huggingface",
                displayName: "Hugging Face",
                url: Self.defaultHubBaseURL
            ),
            ModelDownloadSourceCandidate(
                id: "hf-mirror",
                displayName: "HF Mirror",
                url: Self.mirrorHubBaseURL
            ),
        ]
    }

    private func shouldReuseSavedDownloadSource(for repo: String) -> Bool {
        if case .paused = state(for: repo) {
            return true
        }
        return hasResumableDownload(repo: repo, isDownloaded: downloadedStateByRepo[repo] ?? false)
    }

    private func performDownloadWithFallback(for repo: String) async throws -> URL {
        _ = try ModelStorageDirectoryManager.requireWriteRootURL()
        let selection = try await ModelDownloadSourceSelector.select(
            candidates: downloadSourceCandidates(),
            targetKey: downloadSourceTargetKey(for: repo),
            reuseSavedSource: shouldReuseSavedDownloadSource(for: repo)
        ) { candidate in
            let startedAt = Date()
            let session = MLXModelDownloadSupport.makeDownloadSession(for: candidate.url)
            let entries = try await MLXModelDownloadSupport.fetchModelEntries(
                repo: repo,
                baseURL: candidate.url,
                session: session,
                userAgent: Self.hubUserAgent
            )
            let elapsed = max(Date().timeIntervalSince(startedAt), 0.001)
            let bytes = entries.reduce(Int64(0)) { partial, entry in
                partial + max(entry.size ?? 0, 0)
            }
            return (elapsed, bytes)
        }
        VoxtLog.modelInfo(
            "Selected MLX Audio download source. repo=\(repo), source=\(selection.candidate.displayName), url=\(selection.candidate.url.absoluteString), reusedSavedSource=\(selection.reusedSavedSource), probes=\(ModelDownloadSourceSelector.logSummary(for: selection))"
        )

        var lastError: Error?
        for candidate in selection.attemptCandidates {
            try Task.checkCancellation()
            do {
                return try await performDownload(using: candidate.url, for: repo)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                try Task.checkCancellation()
                lastError = error
                VoxtLog.modelWarning(
                    "MLX Audio download source failed. repo=\(repo), source=\(candidate.displayName), error=\(error.localizedDescription)"
                )
                await clearHubCache(for: repo)
            }
        }

        clearSelectedDownloadSource(for: repo)
        throw lastError ?? NSError(
            domain: "MLXModelManager",
            code: 1004,
            userInfo: [NSLocalizedDescriptionKey: "All MLX Audio download sources failed."]
        )
    }

    private func performDownload(using baseURL: URL, for repo: String) async throws -> URL {
        guard let repoID = Repo.ID(rawValue: repo) else {
            throw NSError(
                domain: "MLXModelManager",
                code: 1000,
                userInfo: [NSLocalizedDescriptionKey: "Invalid model identifier"]
            )
        }
        let token = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
        let session = MLXModelDownloadSupport.makeDownloadSession(for: baseURL)
        return try await resolveOrDownloadModelUsingLFS(
            repoID: repoID,
            session: session,
            baseURL: baseURL,
            bearerToken: token
        )
    }

    private func resolveOrDownloadModelUsingLFS(
        repoID: Repo.ID,
        session: URLSession,
        baseURL: URL,
        bearerToken: String?
    ) async throws -> URL {
        let repo = repoID.description
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let baseDir = writeRootURL().appendingPathComponent("mlx-audio")
        let modelDir = baseDir.appendingPathComponent(modelSubdir)
        let tempDir = baseDir.appendingPathComponent("\(modelSubdir)-download")

        let alreadyInstalled = await Task.detached(priority: .utility) {
            MLXModelDownloadSupport.isModelDirectoryValid(modelDir, repo: repo, fileManager: .default)
        }.value
        if alreadyInstalled {
            return modelDir
        }

        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        VoxtLog.modelInfo("Fetching model entries: \(repoID.description)")
        let entries = try await MLXModelDownloadSupport.fetchModelEntries(
            repo: repoID.description,
            baseURL: baseURL,
            session: session,
            userAgent: Self.hubUserAgent
        )
        VoxtLog.modelInfo("Entry count: \(entries.count)")
        guard !entries.isEmpty else {
            throw MLXModelDownloadSupport.DownloadValidationError.emptyFileList
        }
        let totalBytes = max(entries.reduce(Int64(0)) { partial, entry in
            partial + max(entry.size ?? 0, 0)
        }, 1)
        let totalFiles = entries.count
        var completedBytes: Int64 = 0

        for (index, entry) in entries.enumerated() {
            let completedFiles = index
            let expectedEntryBytes = max(entry.size ?? 0, 0)
            let progress = Progress(totalUnitCount: max(expectedEntryBytes, 1))
            let baseCompletedBytes = completedBytes
            let isLastEntry = index == totalFiles - 1
            let beforeFraction = totalBytes > 0 ? Double(completedBytes) / Double(totalBytes) : 0
            setDownloadingState(
                progress: min(1, beforeFraction),
                completed: min(completedBytes, totalBytes),
                total: totalBytes,
                currentFile: entry.path,
                completedFiles: completedFiles,
                totalFiles: totalFiles,
                for: repo
            )
            VoxtLog.modelInfo("Download start: \(entry.path) (size=\(entry.size ?? -1))", verbose: true)

            let sampler = Task { [weak self] in
                let startTime = Date()
                while !Task.isCancelled {
                    let effectiveInFlight = ModelDownloadProgress.inFlightBytes(
                        progress: progress,
                        expectedFileBytes: expectedEntryBytes,
                        startTime: startTime
                    )
                    let currentCompleted = min(baseCompletedBytes + effectiveInFlight, totalBytes)
                    let fraction = totalBytes > 0 ? Double(currentCompleted) / Double(totalBytes) : 0
                    let fileTransferLooksComplete = expectedEntryBytes > 0 && effectiveInFlight >= expectedEntryBytes
                    let displayCompletedFiles = (isLastEntry && fileTransferLooksComplete) ? totalFiles : completedFiles
                    let displayCurrentFile = (isLastEntry && fileTransferLooksComplete) ? nil : entry.path
                    await MainActor.run {
                        guard !Task.isCancelled else { return }
                        self?.setDownloadingState(
                            progress: min(1, fraction),
                            completed: currentCompleted,
                            total: totalBytes,
                            currentFile: displayCurrentFile,
                            completedFiles: displayCompletedFiles,
                            totalFiles: totalFiles,
                            for: repo
                        )
                    }
                    try? await Task.sleep(for: .milliseconds(200))
                }
            }
            defer { sampler.cancel() }

            let destination = try MLXModelStorageSupport.destinationFileURL(for: entry.path, under: tempDir)
            if MLXModelDownloadSupport.canReuseExistingDownload(
                at: destination,
                expectedSize: entry.size,
                fileManager: .default
            ) {
                let delta = max(expectedEntryBytes, Int64((try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0))
                completedBytes += max(delta, 0)
                let finishedFiles = completedFiles + 1
                let fraction = totalBytes > 0 ? Double(completedBytes) / Double(totalBytes) : 1
                setDownloadingState(
                    progress: min(1, fraction),
                    completed: min(completedBytes, totalBytes),
                    total: totalBytes,
                    currentFile: nil,
                    completedFiles: finishedFiles,
                    totalFiles: totalFiles,
                    for: repo
                )
                VoxtLog.modelInfo("Download resume reused existing file: \(entry.path)", verbose: true)
                continue
            }

            try await downloadEntryWithRetry(
                repo: repoID.description,
                entryPath: entry.path,
                tempDir: tempDir,
                progress: progress,
                baseURL: baseURL,
                bearerToken: bearerToken
            )
            VoxtLog.modelInfo("Download done: \(entry.path)", verbose: true)
            let delta = max(expectedEntryBytes, max(progress.completedUnitCount, 0))
            completedBytes += max(delta, 0)
            let finishedFiles = completedFiles + 1
            let fraction = totalBytes > 0 ? Double(completedBytes) / Double(totalBytes) : 1
            setDownloadingState(
                progress: min(1, fraction),
                completed: min(completedBytes, totalBytes),
                total: totalBytes,
                currentFile: nil,
                completedFiles: finishedFiles,
                totalFiles: totalFiles,
                for: repo
            )
            VoxtLog.modelInfo(
                "Download progress: files=\(finishedFiles)/\(totalFiles), bytes=\(min(completedBytes, totalBytes))/\(totalBytes)",
                verbose: true
            )
        }

        try await downloadMissingWhisperTokenizerAssetsIfNeeded(
            for: repo,
            directory: tempDir,
            baseURL: baseURL,
            bearerToken: bearerToken
        )

        VoxtLog.modelInfo("Validating downloaded files...", verbose: true)
        try await MLXModelDownloadSupport.validateDownloadedModelInBackground(
            at: tempDir,
            repo: repo,
            sizeState: MLXModelDownloadSupport.validationSizeState(for: repo),
            downloadSizeTolerance: downloadSizeTolerance,
            fileManager: .default
        )
        VoxtLog.modelInfo("Moving downloaded files into final cache...", verbose: true)
        try await Task.detached(priority: .utility) {
            try MLXModelDownloadSupport.clearDirectory(at: modelDir, fileManager: .default)
            try FileManager.default.moveItem(at: tempDir, to: modelDir)
        }.value
        VoxtLog.modelInfo("Download files moved to final cache.", verbose: true)
        return modelDir
    }

    private func downloadEntryWithRetry(
        repo: String,
        entryPath: String,
        tempDir: URL,
        progress: Progress,
        baseURL: URL,
        bearerToken: String?
    ) async throws {
        let destination = try MLXModelStorageSupport.destinationFileURL(for: entryPath, under: tempDir)
        let remoteURL = try MLXModelDownloadSupport.fileResolveURL(
            baseURL: baseURL,
            repo: repo,
            path: entryPath
        )
        _ = try await ResumableModelDownloadSupport.download(
            ResumableDownloadDescriptor(
                sourceURL: remoteURL,
                destinationURL: destination,
                relativePath: entryPath,
                expectedSize: progress.totalUnitCount > 1 ? progress.totalUnitCount : nil,
                userAgent: Self.hubUserAgent,
                bearerToken: bearerToken,
                disableProxy: MLXModelDownloadSupport.isMirrorHost(baseURL)
            ),
            progress: progress
        )
    }

    private func downloadMissingWhisperTokenizerAssetsIfNeeded(
        for repo: String,
        directory: URL,
        baseURL: URL,
        bearerToken: String?
    ) async throws {
        guard let tokenizerRepo = MLXModelDownloadSupport.whisperTokenizerRepo(for: repo) else {
            return
        }
        let missingPaths = MLXModelDownloadSupport.missingWhisperTokenizerAssetPaths(
            at: directory,
            fileManager: .default
        )
        guard !missingPaths.isEmpty else { return }

        let session = MLXModelDownloadSupport.makeDownloadSession(for: baseURL)
        VoxtLog.modelInfo(
            "Fetching Whisper tokenizer entries. repo=\(repo), tokenizerRepo=\(tokenizerRepo)"
        )
        let availableEntries = try await MLXModelDownloadSupport.fetchModelEntries(
            repo: tokenizerRepo,
            baseURL: baseURL,
            session: session,
            userAgent: Self.hubUserAgent
        )
        let availableByPath = Dictionary(uniqueKeysWithValues: availableEntries.map { ($0.path, $0) })
        let entries = try missingPaths.map { path -> MLXModelDownloadSupport.ModelFileEntry in
            guard let entry = availableByPath[path] else {
                throw MLXModelDownloadSupport.DownloadValidationError.missingFiles
            }
            return entry
        }

        VoxtLog.modelInfo(
            "Downloading Whisper tokenizer assets. repo=\(repo), tokenizerRepo=\(tokenizerRepo), files=\(entries.map(\.path).joined(separator: ", "))"
        )
        for entry in entries {
            let progress = Progress(totalUnitCount: max(entry.size ?? 0, 1))
            try await downloadEntryWithRetry(
                repo: tokenizerRepo,
                entryPath: entry.path,
                tempDir: directory,
                progress: progress,
                baseURL: baseURL,
                bearerToken: bearerToken
            )
        }
    }

    private func setDownloadingState(
        progress: Double,
        completed: Int64,
        total: Int64,
        currentFile: String?,
        completedFiles: Int,
        totalFiles: Int,
        for repo: String
    ) {
        let canonicalRepo = Self.canonicalModelRepo(repo)
        guard downloadTasksByRepo[canonicalRepo] != nil,
              downloadStorageRevisions[canonicalRepo] == storageRevision,
              downloadStopActionsByRepo[canonicalRepo] == nil else { return }
        let nextState = ModelState.downloading(
            progress: progress,
            completed: completed,
            total: total,
            currentFile: currentFile,
            completedFiles: completedFiles,
            totalFiles: totalFiles
        )
        setState(nextState, for: canonicalRepo)
    }

    private func downloadErrorMessage(for error: Error, repo: String) -> String {
        if let validationError = error as? MLXModelDownloadSupport.DownloadValidationError,
           let text = validationError.errorDescription
        {
            return text
        }

        if let networkError = error as? MLXModelDownloadSupport.DownloadNetworkError,
           let text = networkError.errorDescription
        {
            return text
        }

        if let httpError = error as? HTTPClientError {
            switch httpError {
            case .responseError(let response, let detail):
                if MLXModelDownloadSupport.isMirrorHost(hubBaseURL), [401, 403].contains(response.statusCode) {
                    return "China mirror rejected request (HTTP \(response.statusCode))."
                }
                if [401, 404].contains(response.statusCode) {
                    return "Model repository unavailable (\(repo), HTTP \(response.statusCode))."
                }
                return "Download failed (HTTP \(response.statusCode)): \(detail)"
            case .decodingError(let response, _):
                return "Download failed while decoding server response (HTTP \(response.statusCode))."
            case .requestError(let detail):
                return "Download request failed: \(detail)"
            case .unexpectedError(let detail):
                return "Download failed: \(detail)"
            }
        }

        return "Download failed: \(error.localizedDescription)"
    }

    private func scheduleIdleUnloadIfNeeded() {
        guard loadedModel != nil else { return }
        idleUnloadTask?.cancel()
        let expectedRepo = loadedRepo
        let delay = resolvedIdleUnloadDelay
        idleUnloadTask = Task { [weak self] in
            do {
                try await Task.sleep(for: delay)
            } catch {
                return
            }
            guard let self else { return }
            await MainActor.run {
                self.unloadLoadedModelIfIdle(expectedRepo: expectedRepo, reason: "idle-timeout")
            }
        }
    }

    private func cancelIdleUnloadTask() {
        idleUnloadTask?.cancel()
        idleUnloadTask = nil
    }

    private func unloadLoadedModelIfIdle(expectedRepo: String?, reason: String) {
        guard activeUseCount == 0 else { return }
        guard loadedModel != nil, loadedRepo == expectedRepo else { return }

        loadedModel = nil
        loadedRepo = nil
        idleUnloadTask = nil
        Memory.clearCache()
        checkExistingModel()
        VoxtLog.modelInfo(
            "MLX Audio model released. repo=\(expectedRepo ?? "unknown"), reason=\(reason)"
        )
    }

    private func clearHubCache(for repo: String) async {
        let root = writeRootURL()
        await Task.detached(priority: .utility) {
            try? await ModelDiskOperations.perform {
                if let id = Repo.ID(rawValue: repo) {
                    MLXModelStorageSupport.clearHubCache(for: id, rootDirectory: root)
                }
            }
        }.value
    }

    private func repairIncompleteModelDirectoryIfNeeded(
        for repo: String,
        existingDirectory: URL
    ) async throws -> URL {
        let lowercasedRepo = repo.lowercased()
        guard lowercasedRepo.contains("sensevoice") || lowercasedRepo.contains("whisper") else {
            return existingDirectory
        }
        let alreadyValid = await Task.detached(priority: .utility) {
            MLXModelDownloadSupport.isModelDirectoryValid(existingDirectory, repo: repo, fileManager: .default)
        }.value
        guard !alreadyValid else {
            return existingDirectory
        }

        let token = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
        try Task.checkCancellation()
        let repairDirectory = try await writableRepairDirectoryIfNeeded(
            for: repo,
            existingDirectory: existingDirectory
        )

        try await repairIncompleteModelDirectoryIfNeeded(
            for: repo,
            existingDirectory: repairDirectory,
            baseURL: hubBaseURL,
            bearerToken: token
        )
        return repairDirectory
    }

    private func writableRepairDirectoryIfNeeded(
        for repo: String,
        existingDirectory: URL
    ) async throws -> URL {
        let writableDirectory = writableShadowDirectory(for: repo)
        return try await Task.detached(priority: .utility) {
            guard !FileManager.default.isWritableFile(atPath: existingDirectory.path),
                  let writableDirectory else { return existingDirectory }
            return try Self.prepareWritableShadowDirectory(from: existingDirectory, to: writableDirectory)
        }.value
    }

    private func repairIncompleteModelDirectoryIfNeeded(
        for repo: String,
        existingDirectory: URL,
        baseURL: URL,
        bearerToken: String?
    ) async throws {
        do {
            try await repairIncompleteModelDirectory(
                for: repo,
                existingDirectory: existingDirectory,
                baseURL: baseURL,
                bearerToken: bearerToken
            )
        } catch {
            try Task.checkCancellation()
            guard let fallbackBaseURL = fallbackHubBaseURL(from: baseURL) else {
                throw error
            }
            VoxtLog.modelWarning(
                "Primary model repair endpoint failed. Retrying with mirror. repo=\(repo), baseURL=\(baseURL.absoluteString), error=\(error.localizedDescription)"
            )
            try await repairIncompleteModelDirectory(
                for: repo,
                existingDirectory: existingDirectory,
                baseURL: fallbackBaseURL,
                bearerToken: bearerToken
            )
        }
    }

    private func repairIncompleteModelDirectory(
        for repo: String,
        existingDirectory: URL,
        baseURL: URL,
        bearerToken: String?
    ) async throws {
        if MLXModelDownloadSupport.whisperTokenizerRepo(for: repo) != nil {
            try await downloadMissingWhisperTokenizerAssetsIfNeeded(
                for: repo,
                directory: existingDirectory,
                baseURL: baseURL,
                bearerToken: bearerToken
            )
            return
        }

        let session = MLXModelDownloadSupport.makeDownloadSession(for: baseURL)
        let entries = try await MLXModelDownloadSupport.fetchModelEntries(
            repo: repo,
            baseURL: baseURL,
            session: session,
            userAgent: Self.hubUserAgent
        )
        let missingEntries = MLXModelDownloadSupport.missingRequiredRepairEntries(
            at: existingDirectory,
            repo: repo,
            availableEntries: entries,
            fileManager: .default
        )
        guard !missingEntries.isEmpty else { return }

        VoxtLog.modelInfo(
            "Repairing incomplete MLX model directory. repo=\(repo), files=\(missingEntries.map(\.path).joined(separator: ", "))"
        )
        for entry in missingEntries {
            let progress = Progress(totalUnitCount: max(entry.size ?? 0, 1))
            try await downloadEntryWithRetry(
                repo: repo,
                entryPath: entry.path,
                tempDir: existingDirectory,
                progress: progress,
                baseURL: baseURL,
                bearerToken: bearerToken
            )
        }
    }
}
