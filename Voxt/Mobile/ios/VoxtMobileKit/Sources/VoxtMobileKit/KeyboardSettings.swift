import Foundation

/// Languages Voxt can translate into, as BCP 47 codes the Mac accepts.
public enum TranslationLanguages {
    public static let catalog = ["en", "zh-Hans", "ja", "es", "fr", "de", "ko", "pt",
                                 "it", "nl", "sv", "pl", "tr", "vi", "id", "th", "uk", "ar", "hi"]

    /// The catalog code for one of `Locale.preferredLanguages`, or nil when Voxt can't
    /// translate into it. Chinese is matched by script (zh-Hans), or by region when the
    /// identifier has no script.
    public static func code(forPreferredLanguage identifier: String) -> String? {
        let parts = identifier.replacingOccurrences(of: "_", with: "-").split(separator: "-").map(String.init)
        guard let language = parts.first?.lowercased(), !language.isEmpty else { return nil }
        let candidate: String
        if language == "zh" {
            let script = parts.dropFirst().first { $0.count == 4 }?.capitalized
            let region = parts.dropFirst().first { $0.count == 2 }?.uppercased()
            candidate = "zh-" + (script ?? (["TW", "HK", "MO"].contains(region ?? "") ? "Hant" : "Hans"))
        } else {
            candidate = language
        }
        return catalog.contains(candidate) ? candidate : nil
    }

    /// The language's name in itself, as on the keyboard's pills: "Español", "日本語".
    public static func nativeName(for code: String) -> String {
        let locale = Locale(identifier: code)
        guard let name = locale.localizedString(forIdentifier: code) else { return code }
        return name.prefix(1).uppercased(with: locale) + name.dropFirst()
    }

    static func baseLanguage(of identifier: String) -> String {
        identifier.replacingOccurrences(of: "_", with: "-").split(separator: "-").first.map { $0.lowercased() } ?? identifier
    }
}

/// Settings chosen in the app and read by the keyboard, in the App Group under
/// `keyboardSettings`. The only store for translation targets.
public struct KeyboardSettings: Codable, Equatable, Sendable {
    public static let storageKey = "keyboardSettings"
    public static let maxTargets = 3

    public enum TargetError: Error, Equatable { case full, unsupported }

    /// In the order the pills fan out above the mic: the first sits straight above it.
    public private(set) var translationTargets: [String]
    public var hapticFeedback: Bool

    public init(translationTargets: [String] = [], hapticFeedback: Bool = true) {
        self.translationTargets = Self.sanitized(translationTargets)
        self.hapticFeedback = hapticFeedback
    }

    private enum CodingKeys: String, CodingKey { case translationTargets, hapticFeedback }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(translationTargets: try c.decodeIfPresent([String].self, forKey: .translationTargets) ?? [],
                  hapticFeedback: try c.decodeIfPresent(Bool.self, forKey: .hapticFeedback) ?? true)
    }

    public var isFull: Bool { translationTargets.count >= Self.maxTargets }

    public mutating func addTarget(_ code: String) throws {
        guard TranslationLanguages.catalog.contains(code) else { throw TargetError.unsupported }
        if translationTargets.contains(code) { return }
        guard !isFull else { throw TargetError.full }
        translationTargets.append(code)
    }

    public mutating func removeTarget(_ code: String) {
        translationTargets.removeAll { $0 == code }
    }

    /// Same arguments as SwiftUI's `onMove`.
    public mutating func moveTargets(fromOffsets source: IndexSet, toOffset destination: Int) {
        let moving = source.filter(translationTargets.indices.contains).map { translationTargets[$0] }
        guard !moving.isEmpty else { return }
        let insertAt = destination - source.filter { $0 < destination }.count
        var rest = translationTargets.enumerated().filter { !source.contains($0.offset) }.map(\.element)
        rest.insert(contentsOf: moving, at: min(max(insertAt, 0), rest.count))
        translationTargets = rest
    }

    /// The phone's languages after the first, as catalog codes: deduplicated, without the
    /// first language's variants, at most 3. A phone with one language gives none.
    public static func seededTargets(preferredLanguages: [String]) -> [String] {
        guard let first = preferredLanguages.first else { return [] }
        var seenBases: Set<String> = [TranslationLanguages.baseLanguage(of: first)]
        var targets: [String] = []
        for identifier in preferredLanguages.dropFirst() {
            let base = TranslationLanguages.baseLanguage(of: identifier)
            guard !seenBases.contains(base), let code = TranslationLanguages.code(forPreferredLanguage: identifier) else { continue }
            seenBases.insert(base)
            targets.append(code)
            if targets.count == maxTargets { break }
        }
        return targets
    }

    /// The saved settings. The first load seeds the targets from the phone's other
    /// languages and saves them, so the phone's languages are read once.
    public static func load(from defaults: UserDefaults, preferredLanguages: [String] = Locale.preferredLanguages) -> KeyboardSettings {
        if let data = defaults.data(forKey: storageKey),
           let saved = try? JSONDecoder().decode(KeyboardSettings.self, from: data) {
            return saved
        }
        let settings = KeyboardSettings(translationTargets: seededTargets(preferredLanguages: preferredLanguages))
        settings.save(to: defaults)
        return settings
    }

    public func save(to defaults: UserDefaults) {
        defaults.set(try? JSONEncoder().encode(self), forKey: Self.storageKey)
    }

    private static func sanitized(_ codes: [String]) -> [String] {
        var result: [String] = []
        for code in codes where TranslationLanguages.catalog.contains(code) && !result.contains(code) {
            result.append(code)
            if result.count == maxTargets { break }
        }
        return result
    }
}
