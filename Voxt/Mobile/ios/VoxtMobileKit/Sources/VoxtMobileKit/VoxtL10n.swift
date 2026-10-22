import Foundation

/// Interface-language lookup shared by the app and the keyboard extension.
///
/// Both targets link this package, so the translation catalog lives here as a package
/// resource and every lookup resolves against `Bundle.module` — never the main bundle.
/// The chosen language is the `app.language` value in the shared App Group; absent or
/// unknown values fall back to English. Nothing here touches the system's preferred
/// languages.
///
/// Instances carry their own `UserDefaults`, so tests resolve against isolated suites
/// while production code uses the constant `shared` resolver. There is no mutable
/// global language state.
/// Instances are immutable value types wrapping a thread-safe `UserDefaults` reference.
public struct VoxtL10n: @unchecked Sendable {
    public static let preferenceKey = "app.language"
    public static let shippedLanguages = ["en", "zh-Hans", "zh-Hant", "ja", "es", "ko"]

    /// Constant resolver bound to the shared App Group; app and keyboard both use this.
    public static let shared = VoxtL10n(defaults: UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard)

    private let defaults: UserDefaults

    public init(defaults: UserDefaults) {
        self.defaults = defaults
    }

    /// The chosen interface language, or nil to follow the system (English base).
    public var languageCode: String? {
        guard let code = defaults.string(forKey: Self.preferenceKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines), !code.isEmpty else { return nil }
        return code
    }

    /// The locale matching the chosen language, for date/number formatting at display sites.
    public var locale: Locale {
        languageCode.map(Locale.init(identifier:)) ?? .current
    }

    /// Looks up a UI literal in the chosen language; nil follows the system's preferred
    /// languages (first shipped match, else English). A missing catalog entry returns
    /// the key (the English literal) and is reported by `hasTranslation`.
    public func string(_ key: String) -> String {
        string(key, language: languageCode)
    }

    public func string(_ key: String, language: String?) -> String {
        Self.cache.bundle(for: language).localizedString(forKey: key, value: key, table: nil)
    }

    /// Picks the shipped language for a preferred-languages list (e.g.
    /// `Locale.preferredLanguages`): exact match, then script/region mapping
    /// (zh-Hant for TW/HK/MO, zh-Hans otherwise), then base-language match, else "en".
    public static func resolve(preferredLanguages: [String]) -> String {
        for raw in preferredLanguages {
            let code = raw.replacingOccurrences(of: "_", with: "-")
            if shippedLanguages.contains(code) { return code }
            if code.hasPrefix("zh") {
                if code.contains("Hant") { return "zh-Hant" }
                if code.contains("Hans") { return "zh-Hans" }
                let region = code.split(separator: "-").dropFirst().last.map(String.init) ?? ""
                return ["TW", "HK", "MO"].contains(region) ? "zh-Hant" : "zh-Hans"
            }
            let base = code.split(separator: "-").first.map(String.init) ?? code
            if shippedLanguages.contains(base) { return base }
        }
        return "en"
    }

    /// Formatted lookup ("Tap to translate into %@", …) in the chosen language.
    public func format(_ key: String, _ args: CVarArg...) -> String {
        format(key, language: languageCode, arguments: args)
    }

    public func format(_ key: String, language: String?, arguments: [CVarArg]) -> String {
        String(format: string(key, language: language), locale: Self.localeFor(language), arguments: arguments)
    }

    /// Plural lookup driven by the catalog's plural variations (`.stringsdict` after
    /// compilation): English one/other, CJK other-only.
    public func plural(_ key: String, count: Int) -> String {
        plural(key, count: count, language: languageCode)
    }

    public func plural(_ key: String, count: Int, language: String?) -> String {
        String(format: string(key, language: language), locale: Self.localeFor(language), count)
    }

    /// False when the catalog has no entry for the key in that language.
    public func hasTranslation(_ key: String, language: String) -> Bool {
        Self.cache.bundle(for: language).localizedString(forKey: key, value: Self.missingSentinel, table: nil) != Self.missingSentinel
    }

    private static let missingSentinel = "\u{1}voxt-missing\u{1}"
    private static let cache = BundleCache()
    private static func localeFor(_ language: String?) -> Locale {
        language.map(Locale.init(identifier:)) ?? .current
    }

    /// Lock-protected bundle cache: constant after creation, lookups are the only
    /// mutation and happen under the lock.
    private final class BundleCache: @unchecked Sendable {
        private let lock = NSLock()
        private var bundles: [String: Bundle] = [:]

        /// The `.lproj` bundle for a language inside the package resources; unknown or
        /// missing languages fall back to English so the UI never shows keys.
        func bundle(for language: String?) -> Bundle {
            let code = language ?? VoxtL10n.resolve(preferredLanguages: Locale.preferredLanguages)
            lock.lock()
            defer { lock.unlock() }
            if let cached = bundles[code] { return cached }
            let bundle = Bundle.module.path(forResource: code, ofType: "lproj")
                .flatMap(Bundle.init(path:))
                ?? Bundle.module.path(forResource: "en", ofType: "lproj").flatMap(Bundle.init(path:))
                ?? .module
            bundles[code] = bundle
            return bundle
        }
    }
}
