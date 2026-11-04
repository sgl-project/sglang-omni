import Foundation
import OmniModelArtifacts
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

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

    static func prepareStagingDirectory(at directory: URL) async throws -> OmniModelDirectoryLease {
        try await perform {
            try FileManager.default.createDirectory(at: directory.deletingLastPathComponent(), withIntermediateDirectories: true)
            let lease = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
            try FileManager.default.createDirectory(at: lease.directory, withIntermediateDirectories: true)
            return lease
        }
    }

    static func publish(
        stagingLease: OmniModelDirectoryLease,
        to destination: URL,
        validate: @escaping @Sendable (URL) throws -> Void
    ) async throws -> OmniModelPublication.Outcome {
        let cancellation = PublicationCancellation()
        return try await withTaskCancellationHandler {
            try await perform {
                try OmniModelPublication.publish(
                    stagingLease: stagingLease, to: destination,
                    validate: validate, checkCancellation: cancellation.check
                )
            }
        } onCancel: {
            cancellation.cancel()
        }
    }

    static func removeModelDirectories(_ directories: [URL]) async throws {
        try await perform {
            var leaseDirectories: [String: URL] = [:]
            var physicalDeletions: Set<String> = []
            var symbolicLinks: [String: URL] = [:]
            for directory in directories where FileManager.default.fileExists(atPath: directory.path) {
                guard directory.isFileURL, !directory.path.utf8.contains(0) else {
                    throw OmniModelStorageError.unsafePath(path: directory.path)
                }
                var targetStatus = stat()
                guard stat(directory.path, &targetStatus) == 0 else {
                    throw OmniModelStorageError.filesystem(operation: "stat deletion target", code: errno)
                }
                // note (guanwei): Inode identity also covers case aliases whose realpath spelling may differ.
                let targetIdentity = "\(targetStatus.st_dev):\(targetStatus.st_ino)"
                leaseDirectories[targetIdentity] = directory
                var entryStatus = stat()
                guard lstat(directory.path, &entryStatus) == 0 else {
                    throw OmniModelStorageError.filesystem(operation: "lstat deletion entry", code: errno)
                }
                if entryStatus.st_mode & S_IFMT == S_IFLNK {
                    guard entryStatus.st_nlink == 1 else {
                        throw OmniModelStorageError.unsafePath(path: directory.path)
                    }
                    symbolicLinks["\(entryStatus.st_dev):\(entryStatus.st_ino)"] = directory
                } else {
                    physicalDeletions.insert(targetIdentity)
                }
            }
            var leases: [String: OmniModelDirectoryLease] = [:]
            for (identity, directory) in leaseDirectories {
                leases[identity] = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
            }
            defer { withExtendedLifetime(leases) {} }
            // note (guanwei): Unlink leaf aliases before deleting any requested target; link-only requests never remove targets.
            for symbolicLink in symbolicLinks.values {
                try FileManager.default.removeItem(at: symbolicLink)
            }
            for (identity, lease) in leases where physicalDeletions.contains(identity) {
                try FileManager.default.removeItem(at: lease.directory)
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

private nonisolated final class PublicationCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var isCancelled = false

    func cancel() {
        lock.lock()
        defer { lock.unlock() }
        isCancelled = true
    }

    func check() throws {
        lock.lock()
        defer { lock.unlock() }
        if isCancelled { throw CancellationError() }
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
