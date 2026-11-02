import Foundation
import HuggingFace
import OmniModelArtifacts

enum MeetingSortformerModelStorage {
    static let repo = MeetingVADModelStorage.sortformerV2Repo

    static func modelDirectory(requireValid: Bool) -> URL? {
        for rootDirectory in ModelStorageDirectoryManager.resolvedReadableRootURLs() {
            guard let directory = MLXModelStorageSupport.cacheDirectory(for: repo, rootDirectory: rootDirectory),
                  FileManager.default.fileExists(atPath: directory.path) else { continue }
            if requireValid && !isValidModelDirectory(directory) { continue }
            return directory
        }
        return nil
    }

    static func readableDirectories() -> [URL] {
        ModelStorageDirectoryManager.resolvedReadableRootURLs().compactMap {
            MLXModelStorageSupport.cacheDirectory(for: repo, rootDirectory: $0)
        }
    }

    static func validatedModelDirectory() async -> URL? {
        let directories = readableDirectories()
        return await Task.detached(priority: .utility) {
            directories.first { isValidModelDirectory($0) }
        }.value
    }

    static func validatedModelDirectoryLease() async throws -> OmniModelDirectoryLease? {
        let directories = readableDirectories()
        return try await ModelDiskOperations.perform {
            for directory in directories {
                guard FileManager.default.fileExists(atPath: directory.path) else { continue }
                let lease = try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead)
                if isValidModelDirectory(lease.directory) { return lease }
            }
            return nil
        }
    }

    static func writeModelDirectory() -> URL? {
        MLXModelStorageSupport.cacheDirectory(for: repo,
            rootDirectory: ModelStorageDirectoryManager.resolvedWriteRootURL())
    }

    static func downloadTempDirectory() -> URL? {
        guard let repoID = HuggingFace.Repo.ID(rawValue: repo) else { return nil }
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        return ModelStorageDirectoryManager.resolvedWriteRootURL()
            .appendingPathComponent("mlx-audio", isDirectory: true)
            .appendingPathComponent("\(modelSubdir)-download", isDirectory: true)
    }

    nonisolated static func isValidModelDirectory(
        _ directory: URL, fileManager: FileManager = .default
    ) -> Bool {
        guard let config = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              (try? JSONSerialization.jsonObject(with: config)) != nil,
              ModelWeightFileValidation.hasCompleteIndex(in: directory) else { return false }
        guard let entries = try? fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
            return false
        }
        return entries.contains {
            guard $0.pathExtension == "safetensors",
                  let values = try? $0.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]) else { return false }
            return values.isRegularFile == true && (values.fileSize ?? 0) > 0
        }
    }

    static func clearHubCache(rootDirectory: URL = ModelStorageDirectoryManager.resolvedWriteRootURL()) {
        guard let repoID = HuggingFace.Repo.ID(rawValue: repo) else { return }
        MLXModelStorageSupport.clearHubCache(for: repoID, rootDirectory: rootDirectory)
    }
}
