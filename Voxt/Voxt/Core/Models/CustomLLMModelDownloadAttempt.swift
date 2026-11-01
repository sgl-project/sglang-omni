import Foundation
import HuggingFace

@MainActor
enum CustomLLMModelDownloadAttempt {
    enum ProgressEvent {
        case updated(
            fraction: Double, completedBytes: Int64, totalBytes: Int64,
            currentFile: String?, completedFiles: Int, totalFiles: Int
        )
        case transferStarted(
            progress: Progress, expectedFileBytes: Int64, completedBytes: Int64,
            totalBytes: Int64, currentFile: String, completedFiles: Int, totalFiles: Int
        )
        case transferFinished
    }

    static func run(
        context: CustomLLMModelDownloadSupport.DownloadContext,
        baseURL: URL,
        directory: URL,
        userAgent: String,
        bearerToken: String?,
        transfer: @MainActor (ResumableDownloadDescriptor, Progress) async throws -> Void = { descriptor, progress in
            _ = try await ResumableModelDownloadSupport.download(descriptor, progress: progress)
        },
        onProgress: @MainActor (ProgressEvent) -> Void
    ) async throws -> URL {
        let totalBytes = context.totalBytes
        let totalFiles = context.entries.count
        var completedBytes: Int64 = 0

        try Task.checkCancellation()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        for (index, entry) in context.entries.enumerated() {
            let expectedFileBytes = max(entry.size ?? 0, 0)
            let progress = Progress(totalUnitCount: max(expectedFileBytes, 1))
            onProgress(.updated(
                fraction: min(1, Double(completedBytes) / Double(totalBytes)),
                completedBytes: min(completedBytes, totalBytes),
                totalBytes: totalBytes,
                currentFile: entry.path,
                completedFiles: index,
                totalFiles: totalFiles
            ))

            let destination = try CustomLLMModelStorageSupport.destinationFileURL(
                for: entry.path,
                under: directory
            )
            if MLXModelDownloadSupport.canReuseExistingDownload(
                at: destination,
                expectedSize: entry.size,
                fileManager: .default
            ) {
                let delta = max(expectedFileBytes, Int64((try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0))
                completedBytes += max(delta, 0)
                onProgress(.updated(
                    fraction: min(1, Double(completedBytes) / Double(totalBytes)),
                    completedBytes: min(completedBytes, totalBytes),
                    totalBytes: totalBytes,
                    currentFile: nil,
                    completedFiles: index + 1,
                    totalFiles: totalFiles
                ))
                VoxtLog.modelInfo("Custom LLM download resume reused existing file: \(entry.path)", verbose: true)
                continue
            }
            onProgress(.transferStarted(
                progress: progress,
                expectedFileBytes: expectedFileBytes,
                completedBytes: completedBytes,
                totalBytes: totalBytes,
                currentFile: entry.path,
                completedFiles: index,
                totalFiles: totalFiles
            ))

            let remoteURL = try MLXModelDownloadSupport.fileResolveURL(
                baseURL: baseURL,
                repo: context.repoID.description,
                path: entry.path
            )
            try await transfer(
                ResumableDownloadDescriptor(
                    sourceURL: remoteURL,
                    destinationURL: destination,
                    relativePath: entry.path,
                    expectedSize: progress.totalUnitCount > 1 ? progress.totalUnitCount : nil,
                    userAgent: userAgent,
                    bearerToken: bearerToken,
                    disableProxy: MLXModelDownloadSupport.isMirrorHost(baseURL)
                ),
                progress
            )
            onProgress(.transferFinished)

            let delta = max(expectedFileBytes, max(progress.completedUnitCount, 0))
            completedBytes += max(delta, 0)
            onProgress(.updated(
                fraction: min(1, Double(completedBytes) / Double(totalBytes)),
                completedBytes: min(completedBytes, totalBytes),
                totalBytes: totalBytes,
                currentFile: nil,
                completedFiles: index + 1,
                totalFiles: totalFiles
            ))
        }

        return directory
    }
}
