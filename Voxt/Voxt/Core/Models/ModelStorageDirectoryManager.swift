// ModelStorageDirectoryManager.swift
// Provides Model Storage Directory Manager for model catalog and storage support.

import Foundation
import AppKit
import OmniModelArtifacts

enum ModelStorageDirectoryManager {
    enum AccessIssue: LocalizedError, Equatable {
        case authorizationRequired(path: String)
        case invalidBookmark(path: String)
        case securityScopeDenied(path: String)

        var errorDescription: String? {
            switch self {
            case .authorizationRequired(let path):
                return AppLocalization.format(
                    "Voxt needs permission to access the model storage folder at %@. Choose the folder again to reauthorize it.",
                    path
                )
            case .invalidBookmark(let path):
                return AppLocalization.format(
                    "The saved permission for the model storage folder at %@ is no longer valid. Choose the folder again to reauthorize it.",
                    path
                )
            case .securityScopeDenied(let path):
                return AppLocalization.format(
                    "macOS denied access to the model storage folder at %@. Choose the folder again to reauthorize it.",
                    path
                )
            }
        }
    }

    struct RootResolution {
        let writeRootURL: URL
        let readableRootURLs: [URL]
        let derivedRootURL: URL
        let accessIssue: AccessIssue?
    }

    enum SelectionIssue: LocalizedError, Equatable {
        case insideDownloads(path: String)

        var errorDescription: String? {
            switch self {
            case .insideDownloads(let path):
                return AppLocalization.format(
                    "Models stored in the Downloads folder are lost when it is cleaned up. Choose a folder outside %@.",
                    path
                )
            }
        }
    }

    /// Folders Voxt creates under a storage root; a migration never moves anything else there.
    static let managedFolderNames = ["mlx-audio", "mlx-llm", "gguf-translation", ".derived-model-artifacts"]
    /// The lease lock folder OmniModelDirectoryLease keeps beside model directories; it is metadata, not a model.
    private static let modelLockFolderName = ".omni-model-locks"

    private struct ResolvedRootCache {
        let bookmarkData: Data?
        let path: String?
        let resolution: RootResolution
    }

    private static let lock = NSLock()
    private static var securityScopedURL: URL?
    private static var resolvedRootCache: ResolvedRootCache?
    private static var authorizedRootURLOverrideForTesting: URL?
    private static let fileManager = FileManager.default

    static var defaultRootURL: URL {
        defaultRootURL(fileManager: fileManager)
    }

    static var legacyDefaultRootURL: URL {
        legacyDefaultRootURL(fileManager: fileManager)
    }

    static func resolvedRootURL() -> URL {
        resolvedWriteRootURL(defaults: .standard)
    }

    static func resolvedRootURL(defaults: UserDefaults) -> URL {
        resolvedWriteRootURL(defaults: defaults)
    }

    static func resolvedWriteRootURL(defaults: UserDefaults = .standard) -> URL {
        resolvedRootResolution(defaults: defaults).writeRootURL
    }

    static func resolvedReadableRootURLs(defaults: UserDefaults = .standard) -> [URL] {
        resolvedRootResolution(defaults: defaults).readableRootURLs
    }

    static func resolvedDerivedRootURL(defaults: UserDefaults = .standard) -> URL {
        resolvedRootResolution(defaults: defaults).derivedRootURL
    }

    static func resolvedRootResolution(defaults: UserDefaults = .standard) -> RootResolution {
        lock.lock()
        if let authorizedRootURLOverrideForTesting {
            lock.unlock()
            let rootURL = authorizedRootURLOverrideForTesting.standardizedFileURL
            return RootResolution(
                writeRootURL: rootURL,
                readableRootURLs: [rootURL],
                derivedRootURL: derivedRootURL(forWriteRoot: rootURL),
                accessIssue: nil
            )
        }
        lock.unlock()

        let bookmarkData = defaults.data(forKey: AppPreferenceKey.modelStorageRootBookmark)
        let storedPath = normalizedStoredPath(defaults.string(forKey: AppPreferenceKey.modelStorageRootPath))
        lock.lock()
        if let resolvedRootCache,
           resolvedRootCache.bookmarkData == bookmarkData,
           resolvedRootCache.path == storedPath {
            let cachedResolution = resolvedRootCache.resolution
            lock.unlock()
            return cachedResolution
        }
        lock.unlock()

        let resolution: RootResolution
        var effectiveBookmarkData = bookmarkData
        if let bookmarkData,
           let scopedAccess = prepareSecurityScopedURL(from: bookmarkData) {
            commitSecurityScopedAccess(scopedAccess)
            if scopedAccess.refreshedBookmarkData != nil {
                effectiveBookmarkData = scopedAccess.refreshedBookmarkData
                defaults.set(effectiveBookmarkData, forKey: AppPreferenceKey.modelStorageRootBookmark)
            }
            resolution = RootResolution(
                writeRootURL: scopedAccess.url,
                readableRootURLs: [scopedAccess.url],
                derivedRootURL: derivedRootURL(forWriteRoot: scopedAccess.url),
                accessIssue: nil
            )
        } else if let storedPath {
            let rootURL = URL(fileURLWithPath: storedPath, isDirectory: true)
            let normalizedRootPath = rootURL.standardizedFileURL.path
            let usesBuiltInRoot = normalizedRootPath == defaultRootURL.standardizedFileURL.path
                || normalizedRootPath == legacyDefaultRootURL.standardizedFileURL.path
            if usesBuiltInRoot {
                let writeRootURL = defaultRootURL
                resolution = RootResolution(
                    writeRootURL: writeRootURL,
                    readableRootURLs: uniqueRootURLs([writeRootURL, legacyDefaultRootURL]),
                    derivedRootURL: derivedRootURL(forWriteRoot: writeRootURL),
                    accessIssue: nil
                )
            } else {
                let issue: AccessIssue = bookmarkData == nil
                    ? .authorizationRequired(path: rootURL.path)
                    : .invalidBookmark(path: rootURL.path)
                resolution = RootResolution(
                    writeRootURL: rootURL,
                    readableRootURLs: [],
                    derivedRootURL: derivedRootURL(forWriteRoot: rootURL),
                    accessIssue: issue
                )
            }
        } else {
            let writeRootURL = defaultRootURL
            resolution = RootResolution(
                writeRootURL: writeRootURL,
                readableRootURLs: uniqueRootURLs([writeRootURL, legacyDefaultRootURL]),
                derivedRootURL: derivedRootURL(forWriteRoot: writeRootURL),
                accessIssue: nil
            )
        }

        updateResolvedRootCache(
            bookmarkData: effectiveBookmarkData,
            path: storedPath,
            resolution: resolution
        )
        return resolution
    }

    static func requireWriteRootURL(defaults: UserDefaults = .standard) throws -> URL {
        let resolution = resolvedRootResolution(defaults: defaults)
        if let accessIssue = resolution.accessIssue {
            throw accessIssue
        }
        try verifyWritableDirectory(resolution.writeRootURL)
        return resolution.writeRootURL
    }

    /// A folder inside Downloads is refused, except the saved root itself: choosing it again restores
    /// access, and the next launch moves its models out before any model work starts.
    static func selectionIssue(for url: URL, savedRootPath: String?, downloadsURL: URL) -> SelectionIssue? {
        if !isInsideDownloads(url, downloadsURL: downloadsURL) {
            return nil
        } else if let savedRootPath, normalizedStoredPath(savedRootPath) == url.standardizedFileURL.path {
            return nil
        } else {
            return .insideDownloads(path: downloadsURL.path)
        }
    }

    static func saveUserSelectedRootURL(_ url: URL) throws {
        let downloadsURL = userDownloadsURL
        if let issue = selectionIssue(
            for: url,
            savedRootPath: UserDefaults.standard.string(forKey: AppPreferenceKey.modelStorageRootPath),
            downloadsURL: downloadsURL
        ) {
            throw issue
        } else {
            VoxtLog.modelInfo("Model storage folder selected. path=\(url.standardizedFileURL.path)", verbose: true)
        }
        // Keep the original URL from NSOpenPanel for bookmark creation. It carries
        // the temporary security scope; standardizing it first can drop that scope.
        let bookmark = try url.bookmarkData(
            options: [.withSecurityScope],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        let normalized = url.standardizedFileURL

        guard let scopedAccess = prepareSecurityScopedURL(from: bookmark) else {
            throw AccessIssue.securityScopeDenied(path: normalized.path)
        }
        do {
            try verifyWritableDirectory(scopedAccess.url)
        } catch {
            if scopedAccess.startedNewAccess {
                scopedAccess.url.stopAccessingSecurityScopedResource()
            }
            throw error
        }
        commitSecurityScopedAccess(scopedAccess)

        let persistedBookmark = scopedAccess.refreshedBookmarkData ?? bookmark

        let defaults = UserDefaults.standard
        defaults.set(normalized.path, forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.set(persistedBookmark, forKey: AppPreferenceKey.modelStorageRootBookmark)
        updateResolvedRootCache(
            bookmarkData: persistedBookmark,
            path: normalized.path,
            resolution: RootResolution(
                writeRootURL: scopedAccess.url,
                readableRootURLs: [scopedAccess.url],
                derivedRootURL: derivedRootURL(forWriteRoot: scopedAccess.url),
                accessIssue: nil
            )
        )
        NotificationCenter.default.post(name: .voxtModelStorageAuthorizationDidChange, object: nil)
    }

    static func openRootInFinder() {
        let url = resolvedWriteRootURL()
        try? fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    /// The user's real Downloads folder; inside the sandbox the directory APIs return the container's alias.
    static var userDownloadsURL: URL {
        let homePath: String
        if let passwd = getpwuid(getuid()), let directory = passwd.pointee.pw_dir {
            homePath = String(cString: directory)
        } else {
            homePath = NSHomeDirectory()
        }
        return URL(fileURLWithPath: homePath, isDirectory: true)
            .appendingPathComponent("Downloads", isDirectory: true)
    }

    static func isInsideDownloads(_ url: URL, downloadsURL: URL) -> Bool {
        let path = url.standardizedFileURL.resolvingSymlinksInPath().path
        let downloadsPath = downloadsURL.standardizedFileURL.resolvingSymlinksInPath().path
        return path == downloadsPath || path.hasPrefix(downloadsPath + "/")
    }

    /// Moves Voxt's model folders out of a storage root inside Downloads, which users routinely
    /// clear, and switches back to the default root. Returns the moved items; on any failure every
    /// move is undone and the saved root is kept, so no model ends up outside every readable root.
    @discardableResult
    static func migrateRootOutOfDownloadsIfNeeded(
        defaults: UserDefaults = .standard,
        downloadsURL: URL = userDownloadsURL,
        destinationRootURL: URL = defaultRootURL
    ) -> [URL] {
        let resolution = resolvedRootResolution(defaults: defaults)
        let sourceRootURL = resolution.writeRootURL
        // Note (Minyi Tang): without access every existence check fails, so the models would look absent.
        guard resolution.accessIssue == nil,
              isInsideDownloads(sourceRootURL, downloadsURL: downloadsURL) else { return [] }

        var moves: [(source: URL, destination: URL)] = []
        var leases: [OmniModelDirectoryLease] = []
        defer { withExtendedLifetime(leases) {} }
        do {
            try fileManager.createDirectory(at: destinationRootURL, withIntermediateDirectories: true)
            // A link anywhere on either root's path, the root itself included, would move files it points to.
            for rootURL in [sourceRootURL, destinationRootURL] {
                if let linkPath = userSymbolicLinkOnPath(of: rootURL) {
                    throw MigrationConflict.symbolicLink(path: linkPath)
                } else {
                    continue
                }
            }
            // Note (Minyi Tang): across volumes a move copies gigabytes on the launch path; keep the root instead.
            guard try isSameVolume(sourceRootURL, destinationRootURL) else {
                VoxtLog.modelWarning("Model storage in Downloads is on another volume; not moving it. source=\(sourceRootURL.path)")
                return []
            }
            var folderNames: [String] = []
            for name in managedFolderNames {
                let sourceFolderURL = sourceRootURL.appendingPathComponent(name, isDirectory: true)
                let destinationFolderURL = destinationRootURL.appendingPathComponent(name, isDirectory: true)
                let sourceFolderType = try itemType(at: sourceFolderURL)
                let destinationFolderType = try itemType(at: destinationFolderURL)
                if sourceFolderType == nil {
                    continue
                } else if sourceFolderType != .typeDirectory || (destinationFolderType != nil && destinationFolderType != .typeDirectory) {
                    throw MigrationConflict.blockedFolder(path: destinationFolderURL.path)
                } else if destinationFolderType == nil {
                    try fileManager.createDirectory(at: destinationFolderURL, withIntermediateDirectories: false)
                }
                folderNames.append(name)
            }
            // One listing per folder: exactly these entries are leased and moved.
            var entriesByFolder: [String: [URL]] = [:]
            for name in folderNames {
                entriesByFolder[name] = try modelEntries(in: sourceRootURL.appendingPathComponent(name, isDirectory: true))
            }
            // A model in use, or a destination another publisher reserved, keeps its lease, so storage stays unchanged.
            for name in folderNames where name == "mlx-audio" || name == "mlx-llm" {
                for modelURL in entriesByFolder[name] ?? [] {
                    let destinationModelURL = destinationRootURL
                        .appendingPathComponent(name, isDirectory: true)
                        .appendingPathComponent(modelURL.lastPathComponent, isDirectory: true)
                    leases.append(try OmniModelDirectoryLease.acquire(for: modelURL, mode: .exclusiveMutation))
                    leases.append(try OmniModelDirectoryLease.acquire(for: destinationModelURL, mode: .exclusiveMutation))
                }
            }
            // Models move one at a time so each root keeps its own lock folder.
            for name in folderNames {
                let destinationFolderURL = destinationRootURL.appendingPathComponent(name, isDirectory: true)
                for modelURL in entriesByFolder[name] ?? [] {
                    let destinationModelURL = destinationFolderURL.appendingPathComponent(modelURL.lastPathComponent)
                    if try itemType(at: destinationModelURL) == .typeDirectory {
                        // A link already in the destination would break once Downloads is cleaned.
                        try throwIfContainsSymbolicLink(destinationModelURL)
                    }
                    let leftFilesBehind = try mergeWithoutOverwriting(modelURL, into: destinationModelURL, moves: &moves)
                    // Note (Minyi Tang): a merge that kept a destination file must still leave one complete model.
                    if leftFilesBehind, !isUsableModel(at: destinationModelURL, folderName: name) {
                        throw MigrationConflict.incompleteModel(path: destinationModelURL.path)
                    } else {
                        continue
                    }
                }
            }
            // Note (Minyi Tang): an entry created during the move, such as a new download, has no lease; keep the old root.
            for name in managedFolderNames {
                let listedNames = Set((entriesByFolder[name] ?? []).map(\.lastPathComponent))
                let sourceFolderURL = sourceRootURL.appendingPathComponent(name, isDirectory: true)
                let currentEntries = try itemType(at: sourceFolderURL) == nil ? [] : try modelEntries(in: sourceFolderURL)
                if let newEntry = currentEntries.first(where: { !listedNames.contains($0.lastPathComponent) }) {
                    throw MigrationConflict.newEntry(path: newEntry.path)
                } else {
                    continue
                }
            }
        } catch {
            VoxtLog.modelWarning("Model storage migration out of Downloads failed; undoing it. error=\(error.localizedDescription)")
            for move in moves.reversed() {
                try? fileManager.createDirectory(at: move.source.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? fileManager.moveItem(at: move.destination, to: move.source)
            }
            return []
        }

        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
        lock.lock()
        resolvedRootCache = nil
        lock.unlock()
        securityScopedURL?.stopAccessingSecurityScopedResource()
        securityScopedURL = nil
        VoxtLog.modelInfo("Moved model storage out of Downloads. source=\(sourceRootURL.path), destination=\(destinationRootURL.path), moved=\(moves.count)")
        return moves.map(\.destination)
    }

    private enum MigrationConflict: LocalizedError {
        case symbolicLink(path: String)
        case incompleteModel(path: String)
        case blockedFolder(path: String)
        case typeMismatch(path: String)
        case newEntry(path: String)

        var errorDescription: String? {
            switch self {
            case .symbolicLink(let path):
                return "a symbolic link would be followed at \(path)"
            case .incompleteModel(let path):
                return "merging would leave an incomplete model at \(path)"
            case .blockedFolder(let path):
                return "a link or a file is where a model folder should be at \(path)"
            case .typeMismatch(let path):
                return "a file and a folder collide at \(path)"
            case .newEntry(let path):
                return "\(path) appeared during the move"
            }
        }
    }

    /// Nil only for a missing item; any other error, such as no permission, fails the migration.
    private static func itemType(at url: URL) throws -> FileAttributeType? {
        do {
            // Note (Minyi Tang): attributesOfItem does not follow symbolic links, unlike fileExists.
            return try fileManager.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile || error.code == .fileNoSuchFile {
            return nil
        }
    }

    private static func throwIfContainsSymbolicLink(_ directoryURL: URL) throws {
        var enumerationError: Error?
        let entries = fileManager.enumerator(
            at: directoryURL,
            includingPropertiesForKeys: [.isSymbolicLinkKey],
            options: [],
            errorHandler: { _, error in
                enumerationError = error
                return false
            }
        )
        while let entryURL = entries?.nextObject() as? URL {
            if try entryURL.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink == true {
                throw MigrationConflict.symbolicLink(path: entryURL.path)
            } else {
                continue
            }
        }
        if let enumerationError {
            throw enumerationError
        } else {
            return
        }
    }

    /// Moves what the destination lacks and never overwrites; returns whether any source file stayed behind.
    private static func mergeWithoutOverwriting(
        _ sourceURL: URL,
        into destinationURL: URL,
        moves: inout [(source: URL, destination: URL)]
    ) throws -> Bool {
        let sourceType = try itemType(at: sourceURL)
        let destinationType = try itemType(at: destinationURL)
        if sourceType == .typeSymbolicLink || destinationType == .typeSymbolicLink {
            throw MigrationConflict.symbolicLink(path: sourceURL.path)
        } else if destinationType == nil {
            // A link inside a moved folder could break, or point back into Downloads.
            if sourceType == .typeDirectory {
                try throwIfContainsSymbolicLink(sourceURL)
            }
            try fileManager.moveItem(at: sourceURL, to: destinationURL)
            moves.append((source: sourceURL, destination: destinationURL))
            return false
        } else if sourceType == .typeDirectory, destinationType == .typeDirectory {
            var leftFilesBehind = false
            for entryURL in try fileManager.contentsOfDirectory(at: sourceURL, includingPropertiesForKeys: nil) {
                let entryLeftBehind = try mergeWithoutOverwriting(
                    entryURL,
                    into: destinationURL.appendingPathComponent(entryURL.lastPathComponent),
                    moves: &moves
                )
                leftFilesBehind = leftFilesBehind || entryLeftBehind
            }
            return leftFilesBehind
        } else if sourceType != destinationType {
            throw MigrationConflict.typeMismatch(path: destinationURL.path)
        } else {
            VoxtLog.modelInfo("Model storage migration kept an existing file. path=\(destinationURL.path)", verbose: true)
            return true
        }
    }

    /// The first user-created link on the path; system links owned by root, such as /var, are allowed.
    private static func userSymbolicLinkOnPath(of url: URL) -> String? {
        var path = ""
        for component in url.standardizedFileURL.pathComponents.dropFirst() {
            path += "/" + component
            var status = stat()
            if lstat(path, &status) == 0, status.st_mode & S_IFMT == S_IFLNK, status.st_uid != 0 {
                return path
            } else {
                continue
            }
        }
        return nil
    }

    /// Model entries only: lease locks and Finder metadata stay in their own root.
    private static func modelEntries(in folderURL: URL) throws -> [URL] {
        try fileManager.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent != modelLockFolderName && $0.lastPathComponent != ".DS_Store" }
    }

    private static func isSameVolume(_ firstURL: URL, _ secondURL: URL) throws -> Bool {
        let firstVolume = try firstURL.resourceValues(forKeys: [.volumeIdentifierKey]).volumeIdentifier
        let secondVolume = try secondURL.resourceValues(forKeys: [.volumeIdentifierKey]).volumeIdentifier
        return firstVolume.map { $0.isEqual(secondVolume) } ?? false
    }

    private static func isUsableModel(at directoryURL: URL, folderName: String) -> Bool {
        switch folderName {
        case "mlx-audio":
            // Some families need the repository, such as Whisper with weights.npz.
            let repo = MLXModelCatalog.supportedModels.map(\.id).first {
                $0.replacingOccurrences(of: "/", with: "_") == directoryURL.lastPathComponent
            }
            return MLXModelDownloadSupport.isModelDirectoryValid(directoryURL, repo: repo, fileManager: fileManager)
        case "mlx-llm":
            return CustomLLMModelStorageSupport.isModelDirectoryValid(directoryURL)
        default:
            // GGUF files are whole models, and derived artifacts are rebuilt on demand.
            return true
        }
    }

    static func resetForTesting() {
        lock.lock()
        resolvedRootCache = nil
        authorizedRootURLOverrideForTesting = nil
        lock.unlock()
        securityScopedURL?.stopAccessingSecurityScopedResource()
        securityScopedURL = nil
    }

    static func setAuthorizedRootURLForTesting(_ url: URL?) {
        lock.lock()
        authorizedRootURLOverrideForTesting = url?.standardizedFileURL
        resolvedRootCache = nil
        lock.unlock()
    }

    private struct ScopedAccessActivation {
        let url: URL
        let refreshedBookmarkData: Data?
        let startedNewAccess: Bool
    }

    private static func prepareSecurityScopedURL(from bookmarkData: Data) -> ScopedAccessActivation? {
        var isStale = false
        guard let resolved = try? URL(
            resolvingBookmarkData: bookmarkData,
            options: [.withSecurityScope, .withoutUI],
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        ) else {
            return nil
        }

        let normalized = resolved.standardizedFileURL
        let alreadyActive = securityScopedURL?.standardizedFileURL.path == normalized.path
        let startedNewAccess: Bool
        if alreadyActive {
            startedNewAccess = false
        } else {
            guard normalized.startAccessingSecurityScopedResource() else { return nil }
            startedNewAccess = true
        }

        let refreshedBookmarkData: Data?
        if isStale {
            refreshedBookmarkData = try? normalized.bookmarkData(
                options: [.withSecurityScope],
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
            if refreshedBookmarkData == nil {
                if startedNewAccess {
                    normalized.stopAccessingSecurityScopedResource()
                }
                return nil
            }
        } else {
            refreshedBookmarkData = nil
        }

        return ScopedAccessActivation(
            url: normalized,
            refreshedBookmarkData: refreshedBookmarkData,
            startedNewAccess: startedNewAccess
        )
    }

    private static func commitSecurityScopedAccess(_ access: ScopedAccessActivation) {
        guard access.startedNewAccess else { return }
        let previousURL = securityScopedURL
        securityScopedURL = access.url
        previousURL?.stopAccessingSecurityScopedResource()
    }

    private static func verifyWritableDirectory(_ url: URL) throws {
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        let probeURL = url.appendingPathComponent(".voxt-write-probe-\(UUID().uuidString)", isDirectory: true)
        do {
            try fileManager.createDirectory(at: probeURL, withIntermediateDirectories: false)
            try fileManager.removeItem(at: probeURL)
        } catch {
            try? fileManager.removeItem(at: probeURL)
            throw error
        }
    }

    private static func defaultRootURL(fileManager: FileManager) -> URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support", isDirectory: true)
        return appSupport
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("model-storage", isDirectory: true)
    }

    private static func legacyDefaultRootURL(fileManager: FileManager) -> URL {
        let caches = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? fileManager.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches", isDirectory: true)
        return caches
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
    }

    private static func normalizedStoredPath(_ path: String?) -> String? {
        guard let path,
              !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL.path
    }

    private static func uniqueRootURLs(_ urls: [URL]) -> [URL] {
        var seenPaths = Set<String>()
        var uniqueURLs: [URL] = []
        for url in urls {
            let normalizedURL = url.standardizedFileURL
            if seenPaths.insert(normalizedURL.path).inserted {
                uniqueURLs.append(normalizedURL)
            }
        }
        return uniqueURLs
    }

    private static func derivedRootURL(forWriteRoot writeRootURL: URL) -> URL {
        writeRootURL
            .appendingPathComponent(".derived-model-artifacts", isDirectory: true)
    }

    private static func updateResolvedRootCache(bookmarkData: Data?, path: String?, resolution: RootResolution) {
        lock.lock()
        resolvedRootCache = ResolvedRootCache(
            bookmarkData: bookmarkData,
            path: path,
            resolution: resolution
        )
        lock.unlock()
    }
}
