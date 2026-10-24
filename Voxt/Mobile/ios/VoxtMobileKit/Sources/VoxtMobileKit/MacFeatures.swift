import Foundation

/// The endpoints the paired Mac listed in its capabilities, kept in the App Group so the
/// keyboard can skip a feature the Mac lacks. Until the first check, and for a Mac that lists
/// none (older Voxt), every feature is on.
public enum MacFeatures {
    static let storageKey = "mac.endpoints"

    public static func save(_ capabilities: VoxtClient.Capabilities, to defaults: UserDefaults) {
        if let endpoints = capabilities.endpoints { defaults.set(endpoints.sorted(), forKey: storageKey) }
        else { defaults.removeObject(forKey: storageKey) }
    }

    public static func serves(_ feature: VoxtClient.Feature, in defaults: UserDefaults) -> Bool {
        defaults.stringArray(forKey: storageKey)?.contains(feature.rawValue) ?? true
    }

    public static func clear(in defaults: UserDefaults) { defaults.removeObject(forKey: storageKey) }

    /// What `save` kept: nil before the first check or for a Mac that lists no endpoints.
    public static func endpoints(in defaults: UserDefaults) -> Set<String>? {
        defaults.stringArray(forKey: storageKey).map(Set.init)
    }
}
