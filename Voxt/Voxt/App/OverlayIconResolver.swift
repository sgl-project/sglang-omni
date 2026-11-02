// OverlayIconResolver.swift
// Provides Overlay Icon Resolver for app lifecycle and routing.

import Foundation
import AppKit
import FaviconFinder

enum EnhancementOverlayIconResolver {
    private static let faviconCache = NSCache<NSString, NSImage>()
    private static let faviconTimeoutNanoseconds: UInt64 = 2_000_000_000
    private static let faviconDiskCacheMaxAge: TimeInterval = 30 * 24 * 60 * 60
    private static let faviconFailureCooldown: TimeInterval = 60 * 60
    private static let faviconFailureLock = NSLock()
    private static var faviconFailureRetryAfterByOrigin: [String: Date] = [:]

    static func appIcon(bundleID: String) -> NSImage? {
        if let running = NSRunningApplication.runningApplications(withBundleIdentifier: bundleID).first,
           let icon = running.icon {
            return icon
        }
        guard let appURL = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) else {
            return nil
        }
        return NSWorkspace.shared.icon(forFile: appURL.path)
    }

    static func faviconOrigin(fromPageURL urlString: String?) -> String? {
        guard let urlString,
              let url = URL(string: urlString),
              let scheme = url.scheme?.lowercased(),
              let host = url.host?.lowercased()
        else {
            return nil
        }

        var components = URLComponents()
        components.scheme = scheme
        components.host = host
        if let port = url.port,
           !((scheme == "http" && port == 80) || (scheme == "https" && port == 443)) {
            components.port = port
        }
        return components.string
    }

    static func faviconLookupURL(forOrigin origin: String) -> URL? {
        URL(string: origin)
    }

    static func cachedFavicon(forOrigin origin: String) -> NSImage? {
        if let cached = faviconCache.object(forKey: origin as NSString) {
            return cached
        }
        guard let diskCached = diskCachedFavicon(forOrigin: origin) else {
            return nil
        }
        faviconCache.setObject(diskCached, forKey: origin as NSString)
        return diskCached
    }

    static func favicon(forOrigin origin: String) async -> NSImage? {
        if let cached = cachedFavicon(forOrigin: origin) {
            return cached
        }
        guard shouldAttemptFaviconLookup(forOrigin: origin) else {
            return nil
        }
        guard let lookupURL = faviconLookupURL(forOrigin: origin) else {
            return nil
        }

        let imageData = await withTaskGroup(of: Data?.self) { group in
            group.addTask {
                do {
                    let favicon = try await FaviconFinder(url: lookupURL)
                        .fetchFaviconURLs()
                        .download()
                        .largest()
                    return favicon.image?.data
                } catch {
                    VoxtLog.warning("Enhancement favicon lookup failed. origin=\(origin), error=\(error.localizedDescription)")
                    return nil
                }
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: faviconTimeoutNanoseconds)
                return nil
            }

            let resolvedImage = await group.next() ?? nil
            group.cancelAll()
            return resolvedImage
        }

        guard let imageData,
              let image = NSImage(data: imageData) else {
            noteFaviconLookupFailure(forOrigin: origin)
            return nil
        }

        faviconCache.setObject(image, forKey: origin as NSString)
        persistFaviconData(imageData, forOrigin: origin)
        clearFaviconLookupFailure(forOrigin: origin)
        return image
    }

    static func faviconCacheFileName(forOrigin origin: String) -> String {
        let encoded = Data(origin.utf8)
            .base64EncodedString()
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "=", with: "")
        return encoded + ".favicon"
    }

    private static func diskCachedFavicon(forOrigin origin: String) -> NSImage? {
        let fileURL = faviconCacheFileURL(forOrigin: origin)
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
              let modifiedAt = attributes[.modificationDate] as? Date,
              Date().timeIntervalSince(modifiedAt) <= faviconDiskCacheMaxAge,
              let data = try? Data(contentsOf: fileURL),
              let image = NSImage(data: data)
        else {
            return nil
        }
        return image
    }

    private static func persistFaviconData(_ data: Data, forOrigin origin: String) {
        let fileURL = faviconCacheFileURL(forOrigin: origin)
        do {
            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: fileURL, options: [.atomic])
        } catch {
            VoxtLog.warning("Enhancement favicon cache write failed. origin=\(origin), error=\(error.localizedDescription)")
        }
    }

    private static func faviconCacheFileURL(forOrigin origin: String) -> URL {
        faviconCacheDirectory()
            .appendingPathComponent(faviconCacheFileName(forOrigin: origin), isDirectory: false)
    }

    private static func faviconCacheDirectory() -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches", isDirectory: true)
        return caches
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("enhancement-favicons", isDirectory: true)
    }

    private static func shouldAttemptFaviconLookup(forOrigin origin: String, now: Date = Date()) -> Bool {
        faviconFailureLock.lock()
        defer { faviconFailureLock.unlock() }

        guard let retryAfter = faviconFailureRetryAfterByOrigin[origin] else {
            return true
        }
        if retryAfter <= now {
            faviconFailureRetryAfterByOrigin.removeValue(forKey: origin)
            return true
        }
        return false
    }

    private static func noteFaviconLookupFailure(forOrigin origin: String, now: Date = Date()) {
        faviconFailureLock.lock()
        faviconFailureRetryAfterByOrigin[origin] = now.addingTimeInterval(faviconFailureCooldown)
        faviconFailureLock.unlock()
    }

    private static func clearFaviconLookupFailure(forOrigin origin: String) {
        faviconFailureLock.lock()
        faviconFailureRetryAfterByOrigin.removeValue(forKey: origin)
        faviconFailureLock.unlock()
    }
}

@MainActor
extension AppDelegate {
    func applyEnhancementOverlayIconIfNeeded(
        match: OverlayEnhancementIconMatch?,
        sessionID: UUID
    ) {
        guard shouldAllowEnhancementOverlayIconUpdate(for: sessionID) else { return }
        overlayState.compactLeadingIconImage = nil

        guard let match else { return }

        switch match.kind {
        case .app:
            overlayState.compactLeadingIconImage = EnhancementOverlayIconResolver.appIcon(bundleID: match.bundleID)
        case .url:
            applyURLMatchOverlayIcon(match, sessionID: sessionID)
        }
    }

    private func applyURLMatchOverlayIcon(
        _ match: OverlayEnhancementIconMatch,
        sessionID: UUID
    ) {
        if let browserIcon = EnhancementOverlayIconResolver.appIcon(bundleID: match.bundleID) {
            overlayState.compactLeadingIconImage = browserIcon
        }

        guard let origin = match.urlOrigin else { return }
        if let cached = EnhancementOverlayIconResolver.cachedFavicon(forOrigin: origin) {
            overlayState.compactLeadingIconImage = cached
            return
        }

        Task { [weak self] in
            guard let self else { return }
            let favicon = await EnhancementOverlayIconResolver.favicon(forOrigin: origin)
            await MainActor.run {
                self.applyFetchedEnhancementOverlayIcon(
                    favicon,
                    expectedMatch: match,
                    sessionID: sessionID
                )
            }
        }
    }

    private func applyFetchedEnhancementOverlayIcon(
        _ image: NSImage?,
        expectedMatch: OverlayEnhancementIconMatch,
        sessionID: UUID
    ) {
        guard let image,
              shouldAllowEnhancementOverlayIconUpdate(for: sessionID),
              lastEnhancementPromptContext?.overlayIconMatch == expectedMatch
        else {
            return
        }
        overlayState.compactLeadingIconImage = image
    }

    private func shouldAllowEnhancementOverlayIconUpdate(for sessionID: UUID) -> Bool {
        activeRecordingSessionID == sessionID &&
            sessionOutputMode == .transcription &&
            overlayState.displayMode == .processing &&
            overlayState.sessionIconMode == .transcription &&
            !isSessionCancellationRequested
    }
}
