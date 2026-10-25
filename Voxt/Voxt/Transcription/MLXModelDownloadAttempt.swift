import Foundation
import HuggingFace
import OmniModelArtifacts

@MainActor
enum MLXModelDownloadAttempt {
    static func download(
        repoID: Repo.ID,
        session: URLSession,
        baseURL: URL,
        bearerToken: String?,
        writeRoot: URL,
        userAgent: String,
        downloadSizeTolerance: Double,
        updateProgress: @escaping @MainActor @Sendable (MLXModelManager.ModelState) -> Void
    ) async throws -> URL {
        let repo = repoID.description
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let baseDir = writeRoot.appendingPathComponent("mlx-audio")
        let modelDir = baseDir.appendingPathComponent(modelSubdir)
        let tempDir = baseDir.appendingPathComponent("\(modelSubdir)-download")

        let alreadyInstalled = await Task.detached(priority: .utility) {
            MLXModelDownloadSupport.isModelDirectoryValid(modelDir, repo: repo, fileManager: .default)
        }.value
        if alreadyInstalled {
            return modelDir
        }

        let stagingLease = try await ModelDiskOperations.prepareStagingDirectory(at: tempDir)
        defer { withExtendedLifetime(stagingLease) {} }

        VoxtLog.modelInfo("Fetching model entries: \(repoID.description)")
        let entries = try await MLXModelDownloadSupport.fetchModelEntries(
            repo: repoID.description,
            baseURL: baseURL,
            session: session,
            userAgent: userAgent
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
                updateProgress: updateProgress
            )
            VoxtLog.modelInfo("Download start: \(entry.path) (size=\(entry.size ?? -1))", verbose: true)

            let sampler = Task {
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
                        setDownloadingState(
                            progress: min(1, fraction),
                            completed: currentCompleted,
                            total: totalBytes,
                            currentFile: displayCurrentFile,
                            completedFiles: displayCompletedFiles,
                            totalFiles: totalFiles,
                            updateProgress: updateProgress
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
                    updateProgress: updateProgress
                )
                VoxtLog.modelInfo("Download resume reused existing file: \(entry.path)", verbose: true)
                continue
            }

            try await MLXModelArtifactPreparation.downloadEntryWithRetry(
                repo: repoID.description,
                entryPath: entry.path,
                tempDir: tempDir,
                progress: progress,
                baseURL: baseURL,
                bearerToken: bearerToken,
                userAgent: userAgent
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
                updateProgress: updateProgress
            )
            VoxtLog.modelInfo(
                "Download progress: files=\(finishedFiles)/\(totalFiles), bytes=\(min(completedBytes, totalBytes))/\(totalBytes)",
                verbose: true
            )
        }

        try await MLXModelArtifactPreparation.downloadMissingWhisperTokenizerAssetsIfNeeded(
            for: repo,
            directory: tempDir,
            baseURL: baseURL,
            bearerToken: bearerToken,
            userAgent: userAgent
        )

        let sizeState = MLXModelDownloadSupport.validationSizeState(for: repo)
        let sizeTolerance = downloadSizeTolerance
        let outcome = try await ModelDiskOperations.publish(stagingLease: stagingLease, to: modelDir) { directory in
            try MLXModelDownloadSupport.validateDownloadedModel(
                at: directory, repo: repo, sizeState: sizeState,
                downloadSizeTolerance: sizeTolerance, fileManager: .default
            )
            let entries = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            if !MLXModelDownloadSupport.weightExtensions(for: repo).contains("npz")
                || entries.contains(where: { $0.pathExtension.lowercased() == "safetensors" }) {
                try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory)
            }
        }
        if let warning = outcome.cleanupErrorDescription {
            VoxtLog.modelWarning("Model installed; cleanup incomplete. retainedPreviousDirectory=\(outcome.retainedPreviousDirectory?.path ?? "none"), error=\(warning)")
        }
        VoxtLog.modelInfo("Download published atomically.", verbose: true)
        return outcome.directory
    }

    private static func setDownloadingState(
        progress: Double,
        completed: Int64,
        total: Int64,
        currentFile: String?,
        completedFiles: Int,
        totalFiles: Int,
        updateProgress: @MainActor @Sendable (MLXModelManager.ModelState) -> Void
    ) {
        updateProgress(.downloading(
            progress: progress,
            completed: completed,
            total: total,
            currentFile: currentFile,
            completedFiles: completedFiles,
            totalFiles: totalFiles
        ))
    }
}
