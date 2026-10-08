// SileroVADModelProvisioner.swift
// Provisions the single supported Silero VAD checkpoint on demand.

import Foundation
import MLX
import MLXAudioVAD

extension SileroVADModelSupport {
    /// Load Silero with MLX C-layer errors converted to Swift throws.
    ///
    /// `SileroVAD.fromModelDirectory` uses bare `eval()` internally; without a
    /// scoped `withError` handler those failures call `fatalError`.
    nonisolated static func loadModel(from directory: URL) throws -> SileroVAD {
        try withError {
            try SileroVAD.fromModelDirectory(directory)
        }
    }
}

@MainActor
final class SileroVADModelProvisioner {
    static let shared = SileroVADModelProvisioner()

    private let modelManager = MLXModelManager(modelRepo: SileroVADModelSupport.repo)
    private var inFlightTask: Task<URL, Error>?
    private var inFlightID = UUID()
    private var storageRevision = UUID()
    private var storageRoots: [URL] = []
    private var prefetchTask: Task<Void, Never>?
    private var isShuttingDownForApplicationTermination = false

    static func prefetchIfNeeded(for mode: LocalVADMode) {
        guard ASRVoiceActivityRuntimePolicy.requiresSileroModel(mode: mode) else { return }
        shared.startPrefetchIfNeeded()
    }

    private func startPrefetchIfNeeded() {
        guard !isShuttingDownForApplicationTermination, prefetchTask == nil else { return }
        prefetchTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer { self.prefetchTask = nil }
            do {
                _ = try await self.ensureModelDirectory()
            } catch is CancellationError {
                return
            } catch {
                VoxtLog.modelWarning(
                    "Automatic Silero VAD download failed. repo=\(SileroVADModelSupport.repo), error=\(error.localizedDescription)"
                )
            }
        }
    }

    func ensureModelDirectory() async throws -> URL {
        guard !isShuttingDownForApplicationTermination else { throw CancellationError() }
        let roots = [ModelStorageDirectoryManager.resolvedWriteRootURL()]
            + ModelStorageDirectoryManager.resolvedReadableRootURLs()
        if roots != storageRoots {
            storageRoots = roots
            storageRevision = UUID()
            inFlightTask?.cancel()
            inFlightTask = nil
            inFlightID = UUID()
            modelManager.refreshStorageRoot()
        }
        let revision = storageRevision
        let cachedDirectory = await MeetingVADModelStorage.validatedModelDirectory()
        try Task.checkCancellation()
        guard revision == storageRevision else { throw CancellationError() }
        if let cachedDirectory { return cachedDirectory }
        if let inFlightTask {
            let directory = try await inFlightTask.value
            try Task.checkCancellation()
            guard revision == storageRevision else { throw CancellationError() }
            return directory
        }

        let repo = SileroVADModelSupport.repo
        let task = Task { @MainActor [modelManager] in
            let directory = try await modelManager.ensureModelDirectory(repo: repo)
            try Task.checkCancellation()
            try await Task.detached(priority: .userInitiated) {
                _ = try SileroVADModelSupport.loadModel(from: directory)
            }.value
            return directory
        }
        let taskID = UUID()
        inFlightID = taskID
        inFlightTask = task
        defer { if inFlightID == taskID { inFlightTask = nil } }

        let directory = try await task.value
        try Task.checkCancellation()
        guard revision == storageRevision else { throw CancellationError() }
        VoxtLog.modelInfo("Automatic Silero VAD download complete. repo=\(repo)")
        return directory
    }

    func shutdownForApplicationTermination() async {
        guard !isShuttingDownForApplicationTermination else { return }
        isShuttingDownForApplicationTermination = true
        let prefetchTask = prefetchTask
        let inFlightTask = inFlightTask
        prefetchTask?.cancel()
        inFlightTask?.cancel()
        await prefetchTask?.value
        _ = try? await inFlightTask?.value
        self.prefetchTask = nil
        self.inFlightTask = nil
        await modelManager.shutdownForApplicationTermination()
    }
}
