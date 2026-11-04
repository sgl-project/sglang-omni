import Foundation
import OmniModelStoragePOSIX
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public nonisolated enum OmniModelStorageError: LocalizedError, Equatable, Sendable {
    case busy(path: String)
    case unsafePath(path: String)
    case unsupportedAtomicExchange
    case crossFilesystem
    case publicationRequiresSiblingDirectories
    case relocationSensitiveLink(path: String)
    case unsupportedNestedRoots(path: String)
    case stagingPathReuse(path: String)
    case exclusiveLeaseRequired
    case filesystem(operation: String, code: Int32)

    public var errorDescription: String? {
        switch self {
        case .busy(let path): return "Model directory is in use: \(path)"
        case .unsafePath(let path): return "Unsafe model storage or lock path: \(path)"
        case .unsupportedAtomicExchange: return "This filesystem does not support atomic directory exchange."
        case .crossFilesystem: return "Model publication requires staging and destination on the same filesystem."
        case .publicationRequiresSiblingDirectories: return "Model publication requires staging and destination to share a canonical parent."
        case .relocationSensitiveLink(let path): return "Model publication cannot preserve this symlink target: \(path)"
        case .unsupportedNestedRoots(let path): return "Nested coordinated model roots are unsupported: \(path)"
        case .stagingPathReuse(let path): return "Staging directory overlaps a destination or has already been published: \(path)"
        case .exclusiveLeaseRequired: return "An exclusive staging lease is required for publication."
        case .filesystem(let operation, let code): return "Model storage \(operation) failed (errno \(code))."
        }
    }
}

/// A nonblocking, cross-process lease. Retain through the last use of the model directory.
/// Nested coordinated roots are unsupported; persistent root claims outlive leases and payloads.
public nonisolated final class OmniModelDirectoryLease: @unchecked Sendable {
    public enum Mode: Sendable {
        case sharedRead
        case exclusiveMutation
    }

    public let directory: URL
    public let mode: Mode
    private let descriptor: Int32
    private let publicationLock = NSLock()
    private var isPublishing = false
    private var isPublished = false
    static let lockDirectoryName = ".omni-model-locks"

    private init(directory: URL, mode: Mode, descriptor: Int32) {
        self.directory = directory
        self.mode = mode
        self.descriptor = descriptor
    }

    deinit {
        // note (guanwei): Close once, without LOCK_UN or EINTR retry, to avoid unlocking a reused descriptor.
        _ = close(descriptor)
    }

    public static func acquire(for directory: URL, mode: Mode) throws -> OmniModelDirectoryLease {
        let canonicalDirectory = try canonicalDirectory(directory)
        try OmniModelStoragePathValidation.validateNonoverlappingRoot(canonicalDirectory)
        let lockDirectory = canonicalDirectory.deletingLastPathComponent().appendingPathComponent(lockDirectoryName)
        if mkdir(lockDirectory.path, 0o700) != 0, errno != EEXIST {
            throw OmniModelStorageError.filesystem(operation: "create lock directory", code: errno)
        }
        let parentDescriptor = open(lockDirectory.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard parentDescriptor >= 0 else {
            throw OmniModelStorageError.unsafePath(path: lockDirectory.path)
        }
        defer { _ = close(parentDescriptor) }
        var parentStatus = stat()
        guard fstat(parentDescriptor, &parentStatus) == 0,
              parentStatus.st_uid == geteuid(),
              parentStatus.st_mode & 0o077 == 0 else {
            throw OmniModelStorageError.unsafePath(path: lockDirectory.path)
        }
        let descriptor = openat(
            parentDescriptor, canonicalDirectory.lastPathComponent,
            O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK, 0o600
        )
        guard descriptor >= 0 else {
            throw OmniModelStorageError.unsafePath(path: lockDirectory.path)
        }
        var keepDescriptor = false
        defer { if !keepDescriptor { _ = close(descriptor) } }
        var status = stat()
        guard fstat(descriptor, &status) == 0,
              status.st_mode & S_IFMT == S_IFREG,
              status.st_uid == geteuid(), status.st_nlink == 1,
              status.st_mode & 0o077 == 0 else {
            throw OmniModelStorageError.unsafePath(path: lockDirectory.path)
        }
        let operation = (mode == .sharedRead ? LOCK_SH : LOCK_EX) | LOCK_NB
        let errorCode = omni_model_storage_flock(descriptor, operation)
        if errorCode == EWOULDBLOCK || errorCode == EAGAIN {
            throw OmniModelStorageError.busy(path: canonicalDirectory.path)
        } else if errorCode != 0 {
            throw OmniModelStorageError.filesystem(operation: "flock", code: errorCode)
        }
        // note (guanwei): Publish the persistent claim before rechecking, so concurrent parent/child claims cannot both succeed.
        try OmniModelStoragePathValidation.validateNonoverlappingRoot(canonicalDirectory)
        var pendingPublication: UInt8 = 0
        let markerByteCount = pread(descriptor, &pendingPublication, 1, 0)
        guard markerByteCount >= 0 else {
            throw OmniModelStorageError.filesystem(operation: "read staging marker", code: errno)
        }
        if mode == .exclusiveMutation, markerByteCount != 0 {
            throw OmniModelStorageError.stagingPathReuse(path: canonicalDirectory.path)
        }
        keepDescriptor = true
        return OmniModelDirectoryLease(directory: canonicalDirectory, mode: mode, descriptor: descriptor)
    }

    static func canonicalDirectory(_ directory: URL) throws -> URL {
        guard directory.isFileURL, !directory.path.utf8.contains(0) else {
            throw OmniModelStorageError.unsafePath(path: directory.path)
        }
        let canonicalDirectory: URL
        if let resolved = realpath(directory.path, nil) {
            defer { free(resolved) }
            canonicalDirectory = URL(fileURLWithPath: String(cString: resolved), isDirectory: false)
        } else if errno == ENOENT {
            var status = stat()
            guard lstat(directory.path, &status) != 0, errno == ENOENT,
                  let parent = realpath(directory.deletingLastPathComponent().path, nil) else {
                throw OmniModelStorageError.unsafePath(path: directory.path)
            }
            defer { free(parent) }
            canonicalDirectory = URL(fileURLWithPath: String(cString: parent), isDirectory: true)
                .appendingPathComponent(directory.lastPathComponent, isDirectory: false)
        } else {
            throw OmniModelStorageError.filesystem(operation: "resolve directory", code: errno)
        }
        guard canonicalDirectory.path != "/",
              !canonicalDirectory.pathComponents.contains(where: { $0.caseInsensitiveCompare(lockDirectoryName) == .orderedSame }) else {
            throw OmniModelStorageError.unsafePath(path: directory.path)
        }
        return canonicalDirectory
    }

    func beginPublication() throws {
        publicationLock.lock()
        defer { publicationLock.unlock() }
        guard mode == .exclusiveMutation else { throw OmniModelStorageError.exclusiveLeaseRequired }
        guard !isPublishing, !isPublished else {
            throw OmniModelStorageError.stagingPathReuse(path: directory.path)
        }
        isPublishing = true
    }

    func markCommitAttempt() throws {
        var pendingPublication: UInt8 = 1
        guard pwrite(descriptor, &pendingPublication, 1, 0) == 1 else {
            throw OmniModelStorageError.filesystem(operation: "mark staging publication", code: errno)
        }
    }

    func clearCommitAttempt() throws {
        guard ftruncate(descriptor, 0) == 0 else {
            throw OmniModelStorageError.filesystem(operation: "clear staging publication marker", code: errno)
        }
    }

    func endPublication(committed: Bool) {
        publicationLock.lock()
        defer { publicationLock.unlock() }
        isPublishing = false
        isPublished = committed
        if !committed { _ = ftruncate(descriptor, 0) }
    }
}
