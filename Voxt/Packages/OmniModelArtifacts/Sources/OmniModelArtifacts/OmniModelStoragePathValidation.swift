import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

nonisolated enum OmniModelStoragePathValidation {
    static func validateNonoverlappingRoot(_ directory: URL) throws {
        var ancestor = directory.deletingLastPathComponent()
        while ancestor.path != "/" {
            let ancestorLock = ancestor.deletingLastPathComponent()
                .appendingPathComponent(OmniModelDirectoryLease.lockDirectoryName)
                .appendingPathComponent(ancestor.lastPathComponent)
            var status = stat()
            if lstat(ancestorLock.path, &status) == 0 {
                throw OmniModelStorageError.unsupportedNestedRoots(path: directory.path)
            } else if errno != ENOENT {
                throw OmniModelStorageError.filesystem(operation: "inspect ancestor model claim", code: errno)
            }
            ancestor.deleteLastPathComponent()
        }
        var status = stat()
        if lstat(directory.path, &status) == 0 {
            guard status.st_mode & S_IFMT == S_IFDIR else {
                throw OmniModelStorageError.unsafePath(path: directory.path)
            }
            try inspectTree(at: directory) { entry, _ in
                guard entry.lastPathComponent.caseInsensitiveCompare(OmniModelDirectoryLease.lockDirectoryName) != .orderedSame else {
                    throw OmniModelStorageError.unsupportedNestedRoots(path: directory.path)
                }
            }
        } else if errno != ENOENT {
            throw OmniModelStorageError.filesystem(operation: "inspect model root", code: errno)
        }
    }

    static func validateRelocation(from staging: URL, to destination: URL) throws {
        try inspectTree(at: staging) { entry, status in
            guard entry.lastPathComponent.caseInsensitiveCompare(OmniModelDirectoryLease.lockDirectoryName) != .orderedSame else {
                throw OmniModelStorageError.unsupportedNestedRoots(path: staging.path)
            }
            if status.st_mode & S_IFMT == S_IFLNK {
                var originalTarget = stat()
                guard stat(entry.path, &originalTarget) == 0,
                      originalTarget.st_mode & S_IFMT == S_IFREG else {
                    throw OmniModelStorageError.relocationSensitiveLink(path: entry.path)
                }
                let futureTarget = try relocatedTarget(of: entry, from: staging, to: destination)
                guard originalTarget.st_dev == futureTarget.st_dev,
                      originalTarget.st_ino == futureTarget.st_ino else {
                    throw OmniModelStorageError.relocationSensitiveLink(path: entry.path)
                }
            }
        }
    }

    private static func inspectTree(at directory: URL, inspect: (URL, stat) throws -> Void) throws {
        var pendingDirectories = [directory]
        while let currentDirectory = pendingDirectories.popLast() {
            let entries: [URL]
            do {
                entries = try FileManager.default.contentsOfDirectory(at: currentDirectory, includingPropertiesForKeys: nil)
            } catch {
                throw OmniModelStorageError.unsafePath(path: currentDirectory.path)
            }
            for entry in entries {
                var status = stat()
                guard lstat(entry.path, &status) == 0 else {
                    throw OmniModelStorageError.filesystem(operation: "inspect model entry", code: errno)
                }
                try inspect(entry, status)
                if status.st_mode & S_IFMT == S_IFDIR {
                    pendingDirectories.append(entry)
                }
            }
        }
    }

    private static func relocatedTarget(of link: URL, from staging: URL, to destination: URL) throws -> stat {
        let futurePath = destination.path + link.path.dropFirst(staging.path.count)
        var pendingComponents = futurePath.split(separator: "/").map(String.init)[...]
        var resolvedComponents: [String] = []
        var followedLinkCount = 0
        let maximumFollowedLinkCount = 40
        var stagingStatus = stat()
        var destinationStatus = stat()
        guard lstat(staging.path, &stagingStatus) == 0 else {
            throw OmniModelStorageError.relocationSensitiveLink(path: link.path)
        }
        let destinationExists = lstat(destination.path, &destinationStatus) == 0
        var status = stat()
        while let component = pendingComponents.popFirst() {
            if component == "." {
                continue
            } else if component == ".." {
                if !resolvedComponents.isEmpty { resolvedComponents.removeLast() }
                continue
            }
            let futureEntry = "/" + (resolvedComponents + [component]).joined(separator: "/")
            let backingPath: String
            let isRelocated = futureEntry == destination.path || futureEntry.hasPrefix(destination.path + "/")
            // note (guanwei): Resolve the future tree without moving it; old staging is absent after cleanup, and destination maps to staged bytes.
            if futureEntry == staging.path || futureEntry.hasPrefix(staging.path + "/") {
                throw OmniModelStorageError.relocationSensitiveLink(path: link.path)
            } else if isRelocated {
                backingPath = staging.path + futureEntry.dropFirst(destination.path.count)
            } else {
                backingPath = futureEntry
            }
            guard lstat(backingPath, &status) == 0 else {
                throw OmniModelStorageError.relocationSensitiveLink(path: link.path)
            }
            if status.st_mode & S_IFMT == S_IFLNK {
                followedLinkCount += 1
                guard followedLinkCount <= maximumFollowedLinkCount else {
                    throw OmniModelStorageError.relocationSensitiveLink(path: link.path)
                }
                let target: String
                do {
                    target = try FileManager.default.destinationOfSymbolicLink(atPath: backingPath)
                } catch {
                    throw OmniModelStorageError.relocationSensitiveLink(path: link.path)
                }
                if target.hasPrefix("/") { resolvedComponents.removeAll() }
                pendingComponents = (target.split(separator: "/").map(String.init) + pendingComponents)[...]
            } else {
                guard pendingComponents.isEmpty || status.st_mode & S_IFMT == S_IFDIR else {
                    throw OmniModelStorageError.relocationSensitiveLink(path: link.path)
                }
                // note (guanwei): Directory identity also catches alternate spellings on case-insensitive filesystems.
                if !isRelocated, status.st_dev == stagingStatus.st_dev, status.st_ino == stagingStatus.st_ino {
                    throw OmniModelStorageError.relocationSensitiveLink(path: link.path)
                } else if !isRelocated, destinationExists,
                          status.st_dev == destinationStatus.st_dev, status.st_ino == destinationStatus.st_ino {
                    resolvedComponents = destination.path.split(separator: "/").map(String.init)
                } else {
                    resolvedComponents.append(component)
                }
            }
        }
        guard status.st_mode & S_IFMT == S_IFREG else {
            throw OmniModelStorageError.relocationSensitiveLink(path: link.path)
        }
        return status
    }
}
