import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public nonisolated enum OmniModelPreparationError: LocalizedError, Equatable, Sendable {
    case destinationExists
    case entryLimitExceeded
    case byteLimitExceeded

    public var errorDescription: String? {
        switch self {
        case .destinationExists: return "Model replacement requires explicit authorization."
        case .entryLimitExceeded: return "The model directory contains too many entries."
        case .byteLimitExceeded: return "The model directory exceeds the preparation byte budget."
        }
    }
}

/// Explicitly copies an installed model into a validated sibling publication; original weights are unchanged.
public nonisolated enum OmniModelPreparation {
    public static let maximumEntryCount = 100_000
    public static let maximumTotalFileByteCount: UInt64 = 128 * 1024 * 1024 * 1024

    public static func prepare(
        from source: URL,
        to destination: URL,
        replaceExisting: Bool = false,
        validate: (URL) throws -> Void = { _ in },
        checkCancellation: () throws -> Void = {}
    ) throws -> OmniModelPublication.Outcome {
        try checkCancellation()
        let reader = try OmniModelDirectoryLease.acquire(for: source, mode: .sharedRead)
        defer { withExtendedLifetime(reader) {} }
        let destination = try OmniModelDirectoryLease.canonicalDirectory(destination)
        guard reader.directory.path != destination.path,
              !reader.directory.path.hasPrefix(destination.path + "/"),
              !destination.path.hasPrefix(reader.directory.path + "/") else {
            throw OmniModelStorageError.stagingPathReuse(path: destination.path)
        }
        try OmniModelArtifactValidation.validateSafetensorsDirectory(at: reader.directory)
        let staging = destination.deletingLastPathComponent()
            .appendingPathComponent(".omni-prepare-" + UUID().uuidString, isDirectory: true)
        let stagingLease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        try FileManager.default.createDirectory(at: stagingLease.directory, withIntermediateDirectories: false)
        var pending = [(source: reader.directory, destination: stagingLease.directory)]
        var entryCount = 0
        var totalFileByteCount: UInt64 = 0
        while let directory = pending.popLast() {
            try checkCancellation()
            let entries = try FileManager.default.contentsOfDirectory(at: directory.source, includingPropertiesForKeys: nil)
            guard entries.count <= maximumEntryCount - entryCount else {
                throw OmniModelPreparationError.entryLimitExceeded
            }
            entryCount += entries.count
            for entry in entries {
                try checkCancellation()
                let target = directory.destination.appendingPathComponent(entry.lastPathComponent)
                var status = stat()
                guard lstat(entry.path, &status) == 0 else {
                    throw OmniModelStorageError.filesystem(operation: "inspect preparation source", code: errno)
                }
                if status.st_mode & S_IFMT == S_IFDIR {
                    try FileManager.default.createDirectory(at: target, withIntermediateDirectories: false)
                    pending.append((source: entry, destination: target))
                } else {
                    let file: URL
                    if status.st_mode & S_IFMT == S_IFLNK {
                        file = entry.resolvingSymlinksInPath().standardizedFileURL
                        guard stat(file.path, &status) == 0, status.st_mode & S_IFMT == S_IFREG else {
                            throw OmniModelStorageError.unsafePath(path: entry.path)
                        }
                    } else {
                        guard status.st_mode & S_IFMT == S_IFREG else {
                            throw OmniModelStorageError.unsafePath(path: entry.path)
                        }
                        file = entry
                    }
                    guard status.st_size >= 0 else { throw OmniModelStorageError.unsafePath(path: entry.path) }
                    let (nextByteCount, overflow) = totalFileByteCount.addingReportingOverflow(UInt64(status.st_size))
                    guard !overflow, nextByteCount <= maximumTotalFileByteCount else {
                        throw OmniModelPreparationError.byteLimitExceeded
                    }
                    totalFileByteCount = nextByteCount
                    try FileManager.default.copyItem(at: file, to: target)
                }
            }
        }
        return try OmniModelPublication.publish(stagingLease: stagingLease, to: destination, validate: { staged in
            if !replaceExisting, FileManager.default.fileExists(atPath: destination.path) {
                throw OmniModelPreparationError.destinationExists
            }
            try OmniModelArtifactValidation.validateSafetensorsDirectory(at: staged)
            try validate(staged)
        }, checkCancellation: checkCancellation)
    }
}
