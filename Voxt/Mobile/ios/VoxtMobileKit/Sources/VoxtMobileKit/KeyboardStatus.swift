import Foundation

/// What the keyboard last saw of its own Full Access, in the App Group under `keyboardStatus`.
/// Without Full Access the keyboard can't write the App Group, so only `true` ever arrives:
/// a missing or old status means "not on, or the keyboard was never opened".
public struct KeyboardStatus: Codable, Equatable, Sendable {
    public static let storageKey = "keyboardStatus"
    public var hasFullAccess: Bool
    public var updatedAt: Date

    public init(hasFullAccess: Bool, updatedAt: Date) {
        self.hasFullAccess = hasFullAccess
        self.updatedAt = updatedAt
    }

    public static func load(from defaults: UserDefaults) -> KeyboardStatus? {
        defaults.data(forKey: storageKey).flatMap { try? JSONDecoder().decode(KeyboardStatus.self, from: $0) }
    }

    public func save(to defaults: UserDefaults) {
        defaults.set(try? JSONEncoder().encode(self), forKey: Self.storageKey)
    }

    /// Called by the keyboard each time it appears. Writes only when Full Access is on,
    /// which is also the only time the write can reach the app.
    public static func reportAppearance(hasFullAccess: Bool, to defaults: UserDefaults, now: Date = Date()) {
        guard hasFullAccess else { return }
        KeyboardStatus(hasFullAccess: true, updatedAt: now).save(to: defaults)
    }

    /// Full Access confirmed by a status written at or after `since` (when the keyboard came up).
    public static func confirmsFullAccess(_ status: KeyboardStatus?, since: Date) -> Bool {
        guard let status, status.hasFullAccess else { return false }
        return status.updatedAt >= since
    }
}
