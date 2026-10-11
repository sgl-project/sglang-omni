// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Localize display strings at lookup time; persisted and protocol identifiers stay unchanged.
enum L10n {
    static let development = "en"
    static let supported = ["en", "zh-Hans"]

    // Note (Jiaxin Deng): Audio and worker threads can resolve errors while the UI changes language.
    private static let state = NSLock()
    nonisolated(unsafe) private static var selected: String?
    nonisolated(unsafe) private static var cache: [String: Bundle] = [:]

    /// `nil` follows the localization macOS picked for the app.
    static var language: String? { state.withLock { selected } }

    static func use(_ language: String?) {
        let resolved = language.flatMap { supported.contains($0) ? $0 : nil }
        state.withLock { selected = resolved }
    }

    // Note (Jiaxin Deng): Native language names remain readable when the current UI language is unfamiliar.
    static func displayName(_ language: String) -> String {
        Locale(identifier: language).localizedString(forIdentifier: language)?.localizedCapitalized ?? language
    }

    // Note (Jiaxin Deng): Locate SwiftPM resources without relying on Bundle.module's absolute build path.
    private final class Anchor {}

    private static let resources: Bundle = {
        if Bundle.main.path(forResource: development, ofType: "lproj") != nil { return .main }
        // Note (Jiaxin Deng): SwiftPM puts resources beside the executable or test bundle.
        let name = "OmniTyper_OmniTyper.bundle"
        var directory = Bundle(for: Anchor.self).bundleURL
        for _ in 0..<3 {
            if let bundle = Bundle(url: directory.appendingPathComponent(name)) { return bundle }
            directory.deleteLastPathComponent()
        }
        return .main
    }()

    // Note (Jiaxin Deng): SwiftPM lowercases .lproj directory names.
    private static func bundle(_ language: String) -> Bundle? {
        state.withLock {
            if let cached = cache[language] { return cached }
            let name = resources.localizations.first { $0.caseInsensitiveCompare(language) == .orderedSame } ?? language
            guard let path = resources.path(forResource: name, ofType: "lproj"),
                  let bundle = Bundle(path: path) else { return nil }
            cache[language] = bundle
            return bundle
        }
    }

    static func string(_ key: String) -> String { string(key, in: language) }

    static func string(_ key: String, in language: String?) -> String {
        let missing = "\u{0}"
        func lookup(_ source: Bundle?) -> String? {
            guard let source else { return nil }
            let value = source.localizedString(forKey: key, value: missing, table: nil)
            return value == missing ? nil : value
        }
        // Note (Jiaxin Deng): Missing translations fall back to English before exposing a raw key.
        if let value = lookup(language.flatMap(bundle) ?? resources) { return value }
        if let value = lookup(bundle(development)) { return value }
        return key
    }
}

func L(_ key: String) -> String { L10n.string(key) }

func L(_ key: String, _ arguments: CVarArg...) -> String {
    String(format: L10n.string(key), arguments: arguments)
}
