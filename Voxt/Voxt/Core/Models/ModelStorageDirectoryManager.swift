// ModelStorageDirectoryManager.swift
// Provides Model Storage Directory Manager for model catalog and storage support.

import Foundation
import AppKit

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

    static func saveUserSelectedRootURL(_ url: URL) throws {
        let normalized = url.standardizedFileURL
        let bookmark = try normalized.bookmarkData(
            options: [.withSecurityScope],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )

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
