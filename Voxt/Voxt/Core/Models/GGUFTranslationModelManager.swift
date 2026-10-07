// GGUFTranslationModelManager.swift
// Provides translation-only GGUF model management and llama.cpp execution.

import Foundation
import AppKit
import Combine

@MainActor
final class GGUFTranslationModelManager: ObservableObject {
    enum ModelState: Equatable {
        case notDownloaded
        case downloading(
            progress: Double,
            completed: Int64,
            total: Int64,
            currentFile: String?,
            completedFiles: Int,
            totalFiles: Int
        )
        case paused(
            progress: Double,
            completed: Int64,
            total: Int64,
            currentFile: String?,
            completedFiles: Int,
            totalFiles: Int
        )
        case downloaded
        case error(String)
    }

    private enum DownloadStopAction {
        case pause
        case cancel
    }

    @Published private(set) var stateByID: [GGUFTranslationModelID: ModelState] = [:]
    @Published private(set) var pausedStatusMessageByID: [GGUFTranslationModelID: String] = [:]
    @Published private(set) var activeDownloadModelID: GGUFTranslationModelID?

    private let runtime = GGUFTranslationRuntime()
    private let installationCache = ModelInstallationCache()
    private var deletingIDs: Set<GGUFTranslationModelID> = []
    private var activeInferenceCount = 0
    private var storageRoot: URL?
    private var storageRevision = UUID()
    private var downloadStorageRevision: UUID?
    private var currentModelID: GGUFTranslationModelID
    private var downloadTask: Task<Void, Never>?
    private var downloadProgressTask: Task<Void, Never>?
    private var downloadStopAction: DownloadStopAction?
    private var isShuttingDownForApplicationTermination = false

    init(modelID: GGUFTranslationModelID) {
        self.currentModelID = modelID
        installationCache.onChange = { [weak self] key, _ in
            guard let self, let id = GGUFTranslationModelID(rawValue: key), self.activeDownloadModelID != id else { return }
            self.stateByID[id] = self.resolvedStoredState(for: id)
        }
        refreshStorageRoot()
    }

    var selectedModelID: GGUFTranslationModelID {
        currentModelID
    }

    func updateModel(id: GGUFTranslationModelID) {
        currentModelID = id
    }

    func refreshStorageRoot() {
        let root = ModelStorageDirectoryManager.resolvedWriteRootURL()
        if storageRoot != root {
            storageRoot = root
            storageRevision = UUID()
            downloadStopAction = downloadTask == nil ? nil : .pause
            downloadTask?.cancel()
            cancelDownloadProgressTask()
            stateByID.removeAll()
            pausedStatusMessageByID.removeAll()
        }
        installationCache.invalidateAll()
        for modelID in GGUFTranslationModelID.allCases {
            guard activeDownloadModelID != modelID else { continue }
            let resolvedState = resolvedStoredState(for: modelID)
            stateByID[modelID] = resolvedState
            if case .paused = resolvedState {
                if pausedStatusMessageByID[modelID] == nil {
                    pausedStatusMessageByID[modelID] = AppLocalization.localizedString("Paused. Ready to continue.")
                }
            } else {
                pausedStatusMessageByID[modelID] = nil
            }
        }
    }

    func state(for id: GGUFTranslationModelID) -> ModelState {
        stateByID[id] ?? resolvedStoredState(for: id)
    }

    func option(for id: GGUFTranslationModelID) -> GGUFTranslationModelOption {
        GGUFTranslationModelCatalog.option(for: id)
    }

    func displayModelsIncludingInstalled() -> [GGUFTranslationModelOption] {
        let localStateIDs = Set(GGUFTranslationModelID.allCases.compactMap { modelID -> GGUFTranslationModelID? in
            switch state(for: modelID) {
            case .downloaded, .downloading, .paused:
                return modelID
            case .notDownloaded, .error:
                return nil
            }
        })
        return GGUFTranslationModelCatalog.displayModels(
            includingInstalled: localStateIDs.union([selectedModelID])
        )
    }

    func displayTitle(for id: GGUFTranslationModelID) -> String {
        option(for: id).title
    }

    func modelFileURL(for id: GGUFTranslationModelID) -> URL {
        GGUFTranslationModelCatalog.modelFileURL(
            for: id,
            root: ModelStorageDirectoryManager.resolvedWriteRootURL()
        )
    }

    func isModelDownloaded(id: GGUFTranslationModelID) -> Bool {
        requestInstallation(id)
        return installationCache.peek(id.rawValue)?.isInstalled ?? false
    }

    func canAttemptInference(id: GGUFTranslationModelID) -> Bool {
        isModelDownloaded(id: id) || isCheckingInstallation(id: id)
    }

    func isCheckingInstallation(id: GGUFTranslationModelID) -> Bool {
        requestInstallation(id)
        return installationCache.isChecking(id.rawValue)
    }

    @discardableResult
    func refreshInstallation(id: GGUFTranslationModelID) async throws -> ModelInstallationSnapshot {
        guard !isShuttingDownForApplicationTermination, !deletingIDs.contains(id) else { throw CancellationError() }
        let request = installationRequest(id)
        return try await installationCache.value(id.rawValue) { request.scan() }
    }

    private func installationRequest(_ id: GGUFTranslationModelID) -> ModelInstallationRequest {
        let url = modelFileURL(for: id)
        let part = partialFileURL(for: url)
        return ModelInstallationRequest(directories: [url], partialDirectories: [part, part.appendingPathExtension("json")], validate: {
            guard let handle = try? FileHandle(forReadingFrom: $0) else { return false }
            defer { try? handle.close() }
            return (try? handle.read(upToCount: 4)) == Data("GGUF".utf8)
        })
    }

    private func requestInstallation(_ id: GGUFTranslationModelID) {
        guard !isShuttingDownForApplicationTermination, !deletingIDs.contains(id), installationCache.needsRequest(id.rawValue) else { return }
        let request = installationRequest(id)
        installationCache.request(id.rawValue) { request.scan() }
    }

    func cachedModelSizeText(id: GGUFTranslationModelID) -> String? {
        requestInstallation(id)
        guard let snapshot = installationCache.peek(id.rawValue), snapshot.allocatedBytes > 0 else { return nil }
        return ByteCountFormatter.string(fromByteCount: snapshot.allocatedBytes, countStyle: .file)
    }

    func openModelDirectory(id: GGUFTranslationModelID) {
        let url = modelFileURL(for: id).deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    @discardableResult
    func deleteModel(id: GGUFTranslationModelID) async -> Result<Void, Error> {
        guard activeInferenceCount == 0, !deletingIDs.contains(id) else {
            return .failure(NSError(domain: "Voxt.GGUFTranslation", code: 1005,
                userInfo: [NSLocalizedDescriptionKey: "The model is currently in use. Try again when generation finishes."]))
        }
        deletingIDs.insert(id)
        defer { deletingIDs.remove(id) }
        let url = modelFileURL(for: id)
        let partial = partialFileURL(for: url)
        let pending = activeDownloadModelID == id ? downloadTask : nil
        if pending != nil { cancelDownload(id: id) }
        installationCache.invalidate(id.rawValue)
        await pending?.value
        do {
            try await ModelDiskOperations.remove([url, partial, partial.appendingPathExtension("json")])
            installationCache.invalidate(id.rawValue)
            stateByID[id] = .notDownloaded
            pausedStatusMessageByID[id] = nil
            return .success(())
        } catch {
            installationCache.invalidate(id.rawValue)
            stateByID[id] = .error("Couldn't uninstall model: \(error.localizedDescription)")
            return .failure(error)
        }
    }

    func downloadModel(id: GGUFTranslationModelID) {
        guard !isShuttingDownForApplicationTermination, !deletingIDs.contains(id) else { return }
        guard downloadTask == nil else { return }
        guard activeDownloadModelID == nil || activeDownloadModelID == id else { return }
        if isCheckingInstallation(id: id) {
            let request = installationRequest(id)
            Task { [weak self] in
                guard let self else { return }
                do {
                    _ = try await self.installationCache.value(id.rawValue) { request.scan() }
                    self.downloadModel(id: id)
                } catch { return }
            }
            return
        }
        guard !isModelDownloaded(id: id) else {
            stateByID[id] = .downloaded
            return
        }
        installationCache.invalidate(id.rawValue)

        let revision = storageRevision
        downloadStorageRevision = revision
        downloadStopAction = nil
        SystemNotificationSupport.requestAuthorizationIfNeeded()
        activeDownloadModelID = id
        pausedStatusMessageByID[id] = nil

        if let snapshot = pausedDownloadSnapshot(for: id) {
            setDownloadingState(
                id: id,
                progress: snapshot.progress,
                completed: snapshot.completed,
                total: snapshot.total,
                currentFile: snapshot.currentFile,
                completedFiles: snapshot.completedFiles,
                totalFiles: snapshot.totalFiles
            )
        } else {
            let modelOption = option(for: id)
            setDownloadingState(
                id: id,
                progress: 0,
                completed: 0,
                total: modelOption.sizeBytes,
                currentFile: modelOption.filename,
                completedFiles: 0,
                totalFiles: 1
            )
        }

        downloadTask = Task { [weak self] in
            guard let self else { return }
            defer {
                cancelDownloadProgressTask()
                downloadTask = nil
                installationCache.invalidate(id.rawValue)
                requestInstallation(id)
                downloadStopAction = nil
                downloadStorageRevision = nil
                if activeDownloadModelID == id {
                    activeDownloadModelID = nil
                }
            }

            do {
                try Task.checkCancellation()
                guard revision == storageRevision else { throw CancellationError() }
                try await performDownload(id: id)
                try Task.checkCancellation()
                guard revision == storageRevision else { throw CancellationError() }
                pausedStatusMessageByID[id] = nil
                stateByID[id] = .downloaded
                SystemNotificationSupport.postModelDownloadSucceeded(
                    modelName: displayTitle(for: id)
                )
            } catch is CancellationError {
                guard revision == storageRevision else { return }
                cancelDownloadProgressTask()
                switch downloadStopAction {
                case .pause:
                    if pausedDownloadSnapshot(for: id) == nil {
                        setPausedState(
                            id: id,
                            progress: 0,
                            completed: 0,
                            total: option(for: id).sizeBytes,
                            currentFile: option(for: id).filename,
                            completedFiles: 0,
                            totalFiles: 1
                        )
                    }
                case .cancel, .none:
                    pausedStatusMessageByID[id] = nil
                    let url = modelFileURL(for: id)
                    await cleanupPartialDownload(at: url)
                    guard revision == storageRevision else { return }
                    stateByID[id] = .notDownloaded
                }
            } catch {
                guard revision == storageRevision else { return }
                cancelDownloadProgressTask()
                if pauseDownloadIfNetworkIssue(error, id: id) {
                    return
                }
                pausedStatusMessageByID[id] = nil
                let message = "Download failed: \(error.localizedDescription)"
                stateByID[id] = .error(message)
                SystemNotificationSupport.postModelDownloadFailed(
                    modelName: displayTitle(for: id),
                    message: message
                )
            }
        }
    }

    func cancelDownload(id: GGUFTranslationModelID) {
        guard activeDownloadModelID == id || hasResumableDownload(id: id) else { return }
        pausedStatusMessageByID[id] = nil

        if activeDownloadModelID == id, downloadTask != nil {
            downloadStopAction = .cancel
            stateByID[id] = .notDownloaded
            downloadTask?.cancel()
            cancelDownloadProgressTask()
            return
        }

        let url = modelFileURL(for: id)
        activeDownloadModelID = id
        installationCache.invalidate(id.rawValue)
        stateByID[id] = .notDownloaded
        downloadTask = Task { [weak self] in
            guard let self else { return }
            await cleanupPartialDownload(at: url)
            downloadTask = nil
            downloadStopAction = nil
            activeDownloadModelID = nil
            installationCache.invalidate(id.rawValue)
            requestInstallation(id)
        }
    }

    private func cleanupPartialDownload(at url: URL) async {
        let part = partialFileURL(for: url)
        let result = await Task.detached(priority: .utility) {
            try await ModelDiskOperations.remove([url, part, part.appendingPathExtension("json")])
        }.result
        if case .failure(let error) = result {
            VoxtLog.modelWarning("GGUF partial cleanup failed: \(error.localizedDescription)")
        }
    }

    func pauseDownload(id: GGUFTranslationModelID) {
        guard activeDownloadModelID == id, downloadTask != nil else { return }
        downloadStopAction = .pause
        pausedStatusMessageByID[id] = nil
        if let snapshot = downloadingSnapshot(for: id) {
            setPausedState(
                id: id,
                progress: snapshot.progress,
                completed: snapshot.completed,
                total: snapshot.total,
                currentFile: snapshot.currentFile,
                completedFiles: snapshot.completedFiles,
                totalFiles: snapshot.totalFiles
            )
        }
        downloadTask?.cancel()
        cancelDownloadProgressTask()
    }

    func shutdownForApplicationTermination() async {
        guard !isShuttingDownForApplicationTermination else { return }
        isShuttingDownForApplicationTermination = true
        installationCache.invalidateAll()
        let task = downloadTask
        if let activeDownloadModelID, task != nil {
            downloadStopAction = .pause
            if let snapshot = downloadingSnapshot(for: activeDownloadModelID) {
                setPausedState(
                    id: activeDownloadModelID,
                    progress: snapshot.progress,
                    completed: snapshot.completed,
                    total: snapshot.total,
                    currentFile: snapshot.currentFile,
                    completedFiles: snapshot.completedFiles,
                    totalFiles: snapshot.totalFiles
                )
            }
            task?.cancel()
            cancelDownloadProgressTask()
        }
        await task?.value
        await runtime.shutdownForApplicationTermination()
    }

    func hasResumableDownload(id: GGUFTranslationModelID) -> Bool {
        requestInstallation(id)
        return installationCache.peek(id.rawValue)?.hasPartialDownload ?? false
    }

    func pausedStatusMessage(for id: GGUFTranslationModelID) -> String? {
        pausedStatusMessageByID[id]
    }

    private func resolvedStoredState(for id: GGUFTranslationModelID) -> ModelState {
        if isModelDownloaded(id: id) {
            return .downloaded
        }
        if hasResumableDownload(id: id) {
            return .paused(
                progress: 0,
                completed: 0,
                total: option(for: id).sizeBytes,
                currentFile: option(for: id).filename,
                completedFiles: 0,
                totalFiles: 1
            )
        }
        return .notDownloaded
    }

    private func performDownload(id: GGUFTranslationModelID) async throws {
        _ = try ModelStorageDirectoryManager.requireWriteRootURL()
        let modelOption = option(for: id)
        let destinationURL = modelFileURL(for: id)
        let directoryURL = destinationURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)

        let progress = Progress(totalUnitCount: max(modelOption.sizeBytes, 1))
        let displayedTotal = max(modelOption.sizeBytes, 1)
        let baseCompleted = max(downloadingSnapshot(for: id)?.completed ?? pausedDownloadSnapshot(for: id)?.completed ?? 0, 0)
        setDownloadingState(
            id: id,
            progress: displayedTotal > 0 ? min(1, Double(baseCompleted) / Double(displayedTotal)) : 0,
            completed: baseCompleted,
            total: displayedTotal,
            currentFile: modelOption.filename,
            completedFiles: 0,
            totalFiles: 1
        )

        cancelDownloadProgressTask()
        downloadProgressTask = Task { [weak self] in
            while !Task.isCancelled {
                await MainActor.run {
                    guard let self else { return }
                    let total = max(progress.totalUnitCount, 1)
                    let completed = max(progress.completedUnitCount, 0)
                    self.setDownloadingState(
                        id: id,
                        progress: min(1, Double(completed) / Double(total)),
                        completed: completed,
                        total: total,
                        currentFile: modelOption.filename,
                        completedFiles: 0,
                        totalFiles: 1
                    )
                }
                try? await Task.sleep(for: .milliseconds(200))
            }
        }

        let result = try await ResumableModelDownloadSupport.download(
            ResumableDownloadDescriptor(
                sourceURL: modelOption.downloadURL,
                destinationURL: destinationURL,
                relativePath: modelOption.filename,
                expectedSize: nil,
                userAgent: "Voxt/1.0 (GGUFTranslation)",
                disableProxy: MLXModelDownloadSupport.isMirrorHost(modelOption.downloadURL)
            ),
            progress: progress
        )
        cancelDownloadProgressTask()
        let completed = max(result.bytesDownloaded, progress.completedUnitCount)
        let total = max(progress.totalUnitCount, completed)
        setDownloadingState(
            id: id,
            progress: total > 0 ? min(1, Double(completed) / Double(total)) : 1,
            completed: completed,
            total: total,
            currentFile: nil,
            completedFiles: 1,
            totalFiles: 1
        )
    }

    private func cancelDownloadProgressTask() {
        downloadProgressTask?.cancel()
        downloadProgressTask = nil
    }

    private func setDownloadingState(
        id: GGUFTranslationModelID,
        progress: Double,
        completed: Int64,
        total: Int64,
        currentFile: String?,
        completedFiles: Int,
        totalFiles: Int
    ) {
        guard activeDownloadModelID == id, downloadStopAction == nil,
              downloadStorageRevision == storageRevision else { return }
        let nextState = ModelState.downloading(
            progress: progress,
            completed: completed,
            total: total,
            currentFile: currentFile,
            completedFiles: completedFiles,
            totalFiles: totalFiles
        )
        if stateByID[id] != nextState {
            stateByID[id] = nextState
        }
    }

    private func setPausedState(
        id: GGUFTranslationModelID,
        progress: Double,
        completed: Int64,
        total: Int64,
        currentFile: String?,
        completedFiles: Int,
        totalFiles: Int
    ) {
        let nextState = ModelState.paused(
            progress: progress,
            completed: completed,
            total: total,
            currentFile: currentFile,
            completedFiles: completedFiles,
            totalFiles: totalFiles
        )
        if stateByID[id] != nextState {
            stateByID[id] = nextState
        }
    }

    private func pauseDownloadIfNetworkIssue(_ error: Error, id: GGUFTranslationModelID) -> Bool {
        guard let message = MLXModelDownloadSupport.pauseMessageForInterruptedDownload(error) else {
            return false
        }
        pausedStatusMessageByID[id] = message
        if let snapshot = downloadingSnapshot(for: id) ?? pausedDownloadSnapshot(for: id) {
            setPausedState(
                id: id,
                progress: snapshot.progress,
                completed: snapshot.completed,
                total: snapshot.total,
                currentFile: snapshot.currentFile,
                completedFiles: snapshot.completedFiles,
                totalFiles: snapshot.totalFiles
            )
        } else {
            setPausedState(
                id: id,
                progress: 0,
                completed: 0,
                total: option(for: id).sizeBytes,
                currentFile: option(for: id).filename,
                completedFiles: 0,
                totalFiles: 1
            )
        }
        return true
    }

    private func downloadingSnapshot(for id: GGUFTranslationModelID) -> (
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
        ) = stateByID[id] else {
            return nil
        }
        return (progress, completed, total, currentFile, completedFiles, totalFiles)
    }

    private func pausedDownloadSnapshot(for id: GGUFTranslationModelID) -> (
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
        ) = stateByID[id] else {
            return nil
        }
        return (progress, completed, total, currentFile, completedFiles, totalFiles)
    }

    private func partialFileURL(for destinationURL: URL) -> URL {
        destinationURL.appendingPathExtension("part")
    }

    func executeCompiledRequest(
        _ request: LLMCompiledRequest,
        modelID: GGUFTranslationModelID,
        onPartialText: (@Sendable (String) -> Void)? = nil
    ) async throws -> String {
        guard !isShuttingDownForApplicationTermination, !deletingIDs.contains(modelID) else { throw CancellationError() }
        activeInferenceCount += 1
        defer { activeInferenceCount -= 1 }
        let installed = try await refreshInstallation(id: modelID)
        guard let modelURL = installed.directory else {
            throw NSError(
                domain: "Voxt.GGUFTranslation",
                code: 404,
                userInfo: [NSLocalizedDescriptionKey: "Selected GGUF translation model is not installed."]
            )
        }

        let instructions = request.instructions.trimmingCharacters(in: .whitespacesAndNewlines)
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !instructions.isEmpty || !prompt.isEmpty else { return request.fallbackText }

        let maxTokens = resolvedOutputTokenBudget(for: request)
        VoxtLog.llmDebug(
            "GGUF compiled request start. task=\(request.taskLabel), model=\(modelID.rawValue), instructionsChars=\(instructions.count), promptChars=\(prompt.count), inputChars=\(request.inputCharacterCount), outputBudget=\(maxTokens)"
        )
        let generated = try await runtime.generate(
            instructions: instructions,
            prompt: prompt,
            modelURL: modelURL,
            maxTokens: maxTokens,
            onPartialText: onPartialText
        )
        let trimmed = generated.trimmingCharacters(in: .whitespacesAndNewlines)
        VoxtLog.llmDebug(
            "GGUF compiled request finished. task=\(request.taskLabel), model=\(modelID.rawValue), outputChars=\(trimmed.count), usedFallback=\(trimmed.isEmpty)"
        )
        return trimmed.isEmpty ? request.fallbackText : trimmed
    }

    private func resolvedOutputTokenBudget(for request: LLMCompiledRequest) -> Int {
        if let hint = request.outputTokenBudgetHint {
            return max(48, min(hint, 256))
        }

        let estimated = max(64, request.inputCharacterCount * 2)
        return min(estimated, 192)
    }
}
