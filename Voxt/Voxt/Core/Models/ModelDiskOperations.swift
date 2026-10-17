import Foundation

/// Blocking file operations run on a bounded Foundation queue, not Swift's
/// cooperative executor. UI waits are cancellable at ModelInstallationCache;
/// an already executing filesystem syscall cannot be safely interrupted.
nonisolated enum ModelDiskOperations {
    private static let queue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "Voxt.ModelDiskOperations"
        queue.qualityOfService = .utility
        queue.maxConcurrentOperationCount = 2
        return queue
    }()

    static func perform<Value: Sendable>(
        _ operation: @escaping @Sendable () throws -> Value
    ) async throws -> Value {
        try Task.checkCancellation()
        return try await withCheckedThrowingContinuation { continuation in
            queue.addOperation {
                continuation.resume(with: Result(catching: operation))
            }
        }
    }

    static func remove(_ directories: [URL]) async throws {
        try await perform {
            for directory in Set(directories) {
                if FileManager.default.fileExists(atPath: directory.path) {
                    try FileManager.default.removeItem(at: directory)
                }
            }
        }
    }
}

extension FileManager {
    nonisolated func directoryContainsRegularFiles(at url: URL) -> Bool {
        guard let enumerator = self.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }

        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey])
            if values?.isRegularFile == true {
                return true
            }
        }
        return false
    }

    nonisolated func allocatedSizeOfDirectory(at url: URL) throws -> UInt64 {
        var totalSize: UInt64 = 0
        let enumerator = self.enumerator(at: url, includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey])
        while let fileURL = enumerator?.nextObject() as? URL {
            let resourceValues = try fileURL.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey])
            totalSize += UInt64(resourceValues.totalFileAllocatedSize ?? resourceValues.fileAllocatedSize ?? 0)
        }
        return totalSize
    }
}
