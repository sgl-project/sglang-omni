// HistoryAudioDirectoryManager.swift
// Provides History Audio Directory Manager for history storage and display support.

import Foundation
import AppKit

enum HistoryAudioStorageDirectoryManager {
    private static var securityScopedURL: URL?
    private static let fileManager = FileManager.default

    static var defaultRootURL: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support", isDirectory: true)
        return appSupport
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("transcription-history-audio", isDirectory: true)
    }

    static func resolvedRootURL() -> URL {
        let defaults = UserDefaults.standard
        if let bookmarkData = defaults.data(forKey: AppPreferenceKey.historyAudioStorageRootBookmark),
           let bookmarkedURL = resolveSecurityScopedURL(from: bookmarkData) {
            return bookmarkedURL
        }

        if let path = defaults.string(forKey: AppPreferenceKey.historyAudioStorageRootPath), !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }

        return defaultRootURL
    }

    static func saveUserSelectedRootURL(_ url: URL) throws {
        let normalized = url.standardizedFileURL
        let bookmark = try normalized.bookmarkData(
            options: [.withSecurityScope],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )

        let defaults = UserDefaults.standard
        defaults.set(normalized.path, forKey: AppPreferenceKey.historyAudioStorageRootPath)
        defaults.set(bookmark, forKey: AppPreferenceKey.historyAudioStorageRootBookmark)

        _ = resolveSecurityScopedURL(from: bookmark)
    }

    static func ensureRootDirectoryExists() throws -> URL {
        let url = resolvedRootURL()
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func openRootInFinder() {
        if let url = try? ensureRootDirectoryExists() {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        }
    }

    private static func resolveSecurityScopedURL(from bookmarkData: Data) -> URL? {
        var isStale = false
        guard let resolved = try? URL(
            resolvingBookmarkData: bookmarkData,
            options: [.withSecurityScope, .withoutUI],
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        ) else {
            return nil
        }

        if securityScopedURL?.path != resolved.path {
            securityScopedURL?.stopAccessingSecurityScopedResource()
            if resolved.startAccessingSecurityScopedResource() {
                securityScopedURL = resolved
            }
        }

        if isStale,
           let refreshed = try? resolved.bookmarkData(
                options: [.withSecurityScope],
                includingResourceValuesForKeys: nil,
                relativeTo: nil
           ) {
            UserDefaults.standard.set(refreshed, forKey: AppPreferenceKey.historyAudioStorageRootBookmark)
        }

        return resolved
    }
}
