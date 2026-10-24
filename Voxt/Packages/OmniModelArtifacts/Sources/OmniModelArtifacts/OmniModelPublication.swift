import Foundation
import OmniModelStoragePOSIX
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public nonisolated enum OmniModelPublication {
    public struct Outcome: Sendable {
        public let directory: URL
        public let retainedPreviousDirectory: URL?
        public let cleanupErrorDescription: String?
    }

    /// Publishes canonical siblings only, preserving regular-file symlink targets across relocation.
    /// Directory symlinks are unsupported. Success is committed even if cancellation or old-payload cleanup fails afterward.
    public static func publish(
        stagingLease: OmniModelDirectoryLease,
        to destination: URL,
        validate: (URL) throws -> Void = OmniModelArtifactValidation.validateSafetensorsDirectory(at:),
        checkCancellation: () throws -> Void = {}
    ) throws -> Outcome {
        try publish(
            stagingLease: stagingLease, to: destination, validate: validate,
            checkCancellation: checkCancellation,
            exchange: { omni_model_storage_exchange($0.path, $1.path) },
            cleanup: { try FileManager.default.removeItem(at: $0) }
        )
    }

    static func publish(
        stagingLease: OmniModelDirectoryLease,
        to destination: URL,
        validate: (URL) throws -> Void,
        checkCancellation: () throws -> Void,
        exchange: (URL, URL) -> Int32,
        cleanup: (URL) throws -> Void
    ) throws -> Outcome {
        try stagingLease.beginPublication()
        var committed = false
        defer { stagingLease.endPublication(committed: committed) }
        try checkCancellation()
        let stagingDirectory = stagingLease.directory
        let destinationDirectory = try OmniModelDirectoryLease.canonicalDirectory(destination)
        guard stagingDirectory != destinationDirectory,
              !stagingDirectory.path.hasPrefix(destinationDirectory.path + "/"),
              !destinationDirectory.path.hasPrefix(stagingDirectory.path + "/") else {
            throw OmniModelStorageError.stagingPathReuse(path: stagingDirectory.path)
        }
        let destinationLease = try OmniModelDirectoryLease.acquire(for: destinationDirectory, mode: .exclusiveMutation)
        defer { withExtendedLifetime(destinationLease) {} }
        var stagingStatus = stat()
        var parentStatus = stat()
        guard lstat(stagingDirectory.path, &stagingStatus) == 0 else {
            throw OmniModelStorageError.filesystem(operation: "stat staging directory", code: errno)
        }
        guard lstat(destinationDirectory.deletingLastPathComponent().path, &parentStatus) == 0 else {
            throw OmniModelStorageError.filesystem(operation: "stat destination parent", code: errno)
        }
        guard stagingStatus.st_mode & S_IFMT == S_IFDIR else {
            throw OmniModelStorageError.unsafePath(path: stagingDirectory.path)
        }
        guard stagingStatus.st_dev == parentStatus.st_dev else {
            throw OmniModelStorageError.crossFilesystem
        }
        var destinationStatus = stat()
        let destinationExists: Bool
        if lstat(destinationDirectory.path, &destinationStatus) == 0 {
            guard destinationStatus.st_mode & S_IFMT == S_IFDIR else {
                throw OmniModelStorageError.unsafePath(path: destinationDirectory.path)
            }
            guard stagingStatus.st_dev == destinationStatus.st_dev else {
                throw OmniModelStorageError.crossFilesystem
            }
            destinationExists = true
        } else if errno == ENOENT {
            destinationExists = false
        } else {
            throw OmniModelStorageError.filesystem(operation: "stat destination directory", code: errno)
        }
        guard stagingDirectory.deletingLastPathComponent().path == destinationDirectory.deletingLastPathComponent().path else {
            throw OmniModelStorageError.publicationRequiresSiblingDirectories
        }
        try validate(stagingDirectory)
        try OmniModelStoragePathValidation.validateRelocation(from: stagingDirectory, to: destinationDirectory)
        // note (guanwei): A crash around the swap leaves staging quarantined instead of resuming old weights as a new download.
        try stagingLease.markCommitAttempt()
        try checkCancellation()
        let errorCode: Int32
        if destinationExists {
            errorCode = exchange(stagingDirectory, destinationDirectory)
        } else {
            errorCode = rename(stagingDirectory.path, destinationDirectory.path) == 0 ? 0 : errno
        }
        if errorCode == EXDEV {
            throw OmniModelStorageError.crossFilesystem
        } else if [ENOTSUP, EOPNOTSUPP, ENOSYS, EINVAL].contains(errorCode) {
            throw OmniModelStorageError.unsupportedAtomicExchange
        } else if errorCode != 0 {
            throw OmniModelStorageError.filesystem(operation: "publish directory", code: errorCode)
        }
        committed = true
        var retainedPreviousDirectory = destinationExists ? stagingDirectory : nil
        do {
            if destinationExists {
                try cleanup(stagingDirectory)
                retainedPreviousDirectory = nil
            }
            try stagingLease.clearCommitAttempt()
            return Outcome(directory: destinationDirectory, retainedPreviousDirectory: nil, cleanupErrorDescription: nil)
        } catch {
            return Outcome(
                directory: destinationDirectory, retainedPreviousDirectory: retainedPreviousDirectory,
                cleanupErrorDescription: error.localizedDescription
            )
        }
    }
}
