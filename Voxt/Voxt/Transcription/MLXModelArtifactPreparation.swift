import Foundation

@MainActor
enum MLXModelArtifactPreparation {
    nonisolated static func prepareWritableShadowDirectory(from sourceDirectory: URL, to destinationDirectory: URL) throws -> URL {
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

    static func writableRepairDirectoryIfNeeded(
        existingDirectory: URL,
        writableDirectory: URL?
    ) async throws -> URL {
        return try await Task.detached(priority: .utility) {
            guard !FileManager.default.isWritableFile(atPath: existingDirectory.path),
                  let writableDirectory else { return existingDirectory }
            return try Self.prepareWritableShadowDirectory(from: existingDirectory, to: writableDirectory)
        }.value
    }

    static func downloadEntryWithRetry(
        repo: String,
        entryPath: String,
        tempDir: URL,
        progress: Progress,
        baseURL: URL,
        bearerToken: String?,
        userAgent: String
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
                userAgent: userAgent,
                bearerToken: bearerToken,
                disableProxy: MLXModelDownloadSupport.isMirrorHost(baseURL)
            ),
            progress: progress
        )
    }

    static func downloadMissingWhisperTokenizerAssetsIfNeeded(
        for repo: String,
        directory: URL,
        baseURL: URL,
        bearerToken: String?,
        userAgent: String
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
            userAgent: userAgent
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
                bearerToken: bearerToken,
                userAgent: userAgent
            )
        }
    }

    static func repairIncompleteModelDirectoryIfNeeded(
        for repo: String,
        existingDirectory: URL,
        baseURL: URL,
        bearerToken: String?,
        userAgent: String,
        mirrorBaseURL: URL
    ) async throws {
        do {
            try await repairIncompleteModelDirectory(
                for: repo,
                existingDirectory: existingDirectory,
                baseURL: baseURL,
                bearerToken: bearerToken,
                userAgent: userAgent
            )
        } catch {
            try Task.checkCancellation()
            guard !MLXModelDownloadSupport.isMirrorHost(baseURL) else {
                throw error
            }
            VoxtLog.modelWarning(
                "Primary model repair endpoint failed. Retrying with mirror. repo=\(repo), baseURL=\(baseURL.absoluteString), error=\(error.localizedDescription)"
            )
            try await repairIncompleteModelDirectory(
                for: repo,
                existingDirectory: existingDirectory,
                baseURL: mirrorBaseURL,
                bearerToken: bearerToken,
                userAgent: userAgent
            )
        }
    }

    private static func repairIncompleteModelDirectory(
        for repo: String,
        existingDirectory: URL,
        baseURL: URL,
        bearerToken: String?,
        userAgent: String
    ) async throws {
        if MLXModelDownloadSupport.whisperTokenizerRepo(for: repo) != nil {
            try await downloadMissingWhisperTokenizerAssetsIfNeeded(
                for: repo,
                directory: existingDirectory,
                baseURL: baseURL,
                bearerToken: bearerToken,
                userAgent: userAgent
            )
            return
        }

        let session = MLXModelDownloadSupport.makeDownloadSession(for: baseURL)
        let entries = try await MLXModelDownloadSupport.fetchModelEntries(
            repo: repo,
            baseURL: baseURL,
            session: session,
            userAgent: userAgent
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
                bearerToken: bearerToken,
                userAgent: userAgent
            )
        }
    }
}
