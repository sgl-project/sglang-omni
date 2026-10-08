// SpeakerDisplayNameFormatter.swift
// Provides Speaker Display Name Formatter for meeting speaker analysis.

import Foundation

enum MeetingSpeakerDisplayNameFormatter {
    nonisolated static func displayName(ordinal: Int, defaults: UserDefaults = .standard) -> String {
        let localeIdentifier = resolvedLocaleIdentifier(defaults: defaults)
        let format = localizedFormat(localeIdentifier: localeIdentifier)
        return String(format: format, locale: Locale(identifier: localeIdentifier), ordinal)
    }

    nonisolated private static func resolvedLocaleIdentifier(defaults: UserDefaults) -> String {
        let storedLanguage = defaults.string(forKey: "interfaceLanguage") ?? ""
        switch storedLanguage {
        case "en", "zh-Hans", "ja":
            return storedLanguage
        default:
            return resolvedSystemLocaleIdentifier()
        }
    }

    nonisolated private static func resolvedSystemLocaleIdentifier() -> String {
        let preferred = Locale.preferredLanguages.first?.lowercased() ?? ""
        if preferred.hasPrefix("zh") {
            return "zh-Hans"
        }
        if preferred.hasPrefix("ja") {
            return "ja"
        }
        return "en"
    }

    nonisolated private static func localizedFormat(localeIdentifier: String) -> String {
        if let localized = localizedString("Speaker %d", localeIdentifier: localeIdentifier) {
            return localized
        }
        if let english = localizedString("Speaker %d", localeIdentifier: "en") {
            return english
        }
        return "Speaker %d"
    }

    nonisolated private static func localizedString(_ key: String, localeIdentifier: String) -> String? {
        guard let path = Bundle.main.path(forResource: localeIdentifier, ofType: "lproj"),
              let bundle = Bundle(path: path)
        else {
            return nil
        }
        let value = bundle.localizedString(forKey: key, value: key, table: nil)
        return value == key && localeIdentifier != "en" ? nil : value
    }
}
