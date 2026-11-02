// SecurityScopedBookmarkSupport.swift
// Provides Security Scoped Bookmark Support for permissions and secure storage.

import Foundation

enum SecurityScopedBookmarkSupport {
    final class Access: @unchecked Sendable {
        let url: URL
        private let didStartAccessing: Bool

        fileprivate init(url: URL, startsSecurityScopedAccess: Bool) {
            self.url = url
            didStartAccessing = startsSecurityScopedAccess
                ? url.startAccessingSecurityScopedResource()
                : false
        }

        deinit {
            if didStartAccessing {
                url.stopAccessingSecurityScopedResource()
            }
        }
    }

    static func createBookmark(for url: URL) throws -> Data {
        try url.standardizedFileURL.bookmarkData(
            options: [.withSecurityScope],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
    }

    static func resolveDirectoryURL(
        bookmarkData: Data?,
        fallbackPath: String
    ) -> URL? {
        if let bookmarkData,
           let resolved = resolveURL(from: bookmarkData) {
            return resolved
        }

        let trimmedPath = fallbackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPath.isEmpty else { return nil }
        return URL(fileURLWithPath: trimmedPath, isDirectory: true)
    }

    static func resolveFileURL(
        bookmarkData: Data?,
        fallbackPath: String
    ) -> URL? {
        if let bookmarkData,
           let resolved = resolveURL(from: bookmarkData) {
            return resolved
        }

        let trimmedPath = fallbackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPath.isEmpty else { return nil }
        return URL(fileURLWithPath: trimmedPath, isDirectory: false)
    }

    static func accessDirectoryURL(
        bookmarkData: Data?,
        fallbackPath: String
    ) -> Access? {
        accessURL(
            bookmarkData: bookmarkData,
            fallbackPath: fallbackPath,
            isDirectory: true
        )
    }

    static func accessFileURL(
        bookmarkData: Data?,
        fallbackPath: String
    ) -> Access? {
        accessURL(
            bookmarkData: bookmarkData,
            fallbackPath: fallbackPath,
            isDirectory: false
        )
    }

    private static func accessURL(
        bookmarkData: Data?,
        fallbackPath: String,
        isDirectory: Bool
    ) -> Access? {
        if let bookmarkData, let resolved = resolveURL(from: bookmarkData) {
            return Access(url: resolved, startsSecurityScopedAccess: true)
        }
        let trimmedPath = fallbackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPath.isEmpty else { return nil }
        return Access(
            url: URL(fileURLWithPath: trimmedPath, isDirectory: isDirectory),
            startsSecurityScopedAccess: false
        )
    }

    private static func resolveURL(from bookmarkData: Data) -> URL? {
        var isStale = false
        guard let resolved = try? URL(
            resolvingBookmarkData: bookmarkData,
            options: [.withSecurityScope, .withoutUI],
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        ) else {
            return nil
        }

        return resolved.standardizedFileURL
    }
}
