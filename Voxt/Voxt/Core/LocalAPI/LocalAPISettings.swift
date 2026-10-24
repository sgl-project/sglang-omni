// LocalAPISettings.swift
// Provides stored settings for the Local API server: on/off, port and access token.

import Foundation

nonisolated enum LocalAPISettings {
    static let defaultPort = 47910
    static let maxBodyBytes = 25 * 1024 * 1024
    static let tokenAccount = "localAPIToken"

    static func isEnabled(defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: AppPreferenceKey.localAPIServerEnabled)
    }

    static func port(defaults: UserDefaults = .standard) -> Int {
        let stored = defaults.integer(forKey: AppPreferenceKey.localAPIServerPort)
        return (1024...65534).contains(stored) ? stored : defaultPort
    }

    /// Text-history exchange with a paired Voxt Mobile. Missing means off.
    static func isHistorySyncEnabled(defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: AppPreferenceKey.historySyncEnabled)
    }

    /// The History preference itself, not the recording getter. Missing means on.
    static func isHistoryRecordingEnabled(defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: AppPreferenceKey.historyEnabled) as? Bool ?? true
    }

    /// Debug builds read the token from this variable so automated end-to-end tests know it.
    static let tokenEnvironmentKey = "VOXT_LOCAL_API_TOKEN"

    private static let lock = NSLock()
    /// A token that could not be saved (for example, an unsigned build without Keychain access).
    /// Kept for the life of the process so the server and Settings show the same one.
    nonisolated(unsafe) private static var unsavedToken: String?

    /// Returns the access token, creating one the first time.
    static func token(environment: [String: String] = ProcessInfo.processInfo.environment) -> String {
        #if DEBUG
        if let override = environment[tokenEnvironmentKey], !override.isEmpty {
            return override
        }
        #endif
        if let existing = VoxtSecureStorage.string(for: tokenAccount), !existing.isEmpty {
            return existing
        }
        if let unsaved = lock.withLock({ unsavedToken }) {
            return unsaved
        }
        return regenerateToken()
    }

    @discardableResult
    static func regenerateToken() -> String {
        let token = LocalAPIRouter.generateToken()
        let saved = VoxtSecureStorage.set(token, for: tokenAccount)
        if !saved {
            VoxtLog.warning("Local API token could not be saved to the Keychain; it lasts until Voxt quits.")
        }
        lock.withLock { unsavedToken = saved ? nil : token }
        return token
    }

    static func resetUnsavedTokenForTesting() {
        lock.withLock { unsavedToken = nil }
    }
}
