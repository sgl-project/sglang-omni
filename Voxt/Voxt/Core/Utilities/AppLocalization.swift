// AppLocalization.swift
// Provides App Localization for shared utilities.

import Foundation

nonisolated enum AppLocalization {
    private struct LanguageSnapshot {
        let language: AppInterfaceLanguage
        let localeIdentifier: String
        let locale: Locale
    }

    private static let localizedBundles: [String: Bundle] = {
        var bundles = [String: Bundle]()
        for identifier in ["en", "zh-Hans", "ja"] {
            guard let path = Bundle.main.path(forResource: identifier, ofType: "lproj"),
                  let bundle = Bundle(path: path)
            else {
                continue
            }
            bundles[identifier] = bundle
        }
        return bundles
    }()

    private static let languageSnapshotLock = NSLock()
    private static var languageSnapshot = makeLanguageSnapshot()
    private static let localizedStringCacheLock = NSLock()
    private static var localizedStringCache = [String: String]()
    private static let interfaceLanguageObserver: NSObjectProtocol = {
        NotificationCenter.default.addObserver(
            forName: .voxtInterfaceLanguageDidChange,
            object: nil,
            queue: nil
        ) { _ in
            refreshLanguageCache()
        }
    }()

    static var language: AppInterfaceLanguage {
        currentLanguageSnapshot().language
    }

    static var locale: Locale {
        currentLanguageSnapshot().locale
    }

    static func refreshLanguageCache(defaults: UserDefaults = .standard) {
        let snapshot = makeLanguageSnapshot(defaults: defaults)
        languageSnapshotLock.lock()
        languageSnapshot = snapshot
        languageSnapshotLock.unlock()
    }

    static func localizedString(_ key: String) -> String {
        let identifier = currentLanguageSnapshot().localeIdentifier
        return cachedLocalizedString(key, localeIdentifier: identifier)
    }

    static func localizedString(_ key: String, localeIdentifier: String) -> String {
        cachedLocalizedString(key, localeIdentifier: localeIdentifier)
    }

    static func format(_ key: String, _ arguments: CVarArg...) -> String {
        String(format: localizedString(key), locale: locale, arguments: arguments)
    }

    private static func cachedLocalizedString(_ key: String, localeIdentifier: String) -> String {
        let cacheKey = "\(localeIdentifier)\u{1F}\(key)"
        localizedStringCacheLock.lock()
        if let cached = localizedStringCache[cacheKey] {
            localizedStringCacheLock.unlock()
            return cached
        }
        localizedStringCacheLock.unlock()

        let resolved = uncachedLocalizedString(key, localeIdentifier: localeIdentifier)

        localizedStringCacheLock.lock()
        localizedStringCache[cacheKey] = resolved
        localizedStringCacheLock.unlock()
        return resolved
    }

    private static func uncachedLocalizedString(_ key: String, localeIdentifier: String) -> String {
        if let localized = resolvedLocalizedString(key, localeIdentifier: localeIdentifier) {
            return localized
        }
        if let english = resolvedLocalizedString(key, localeIdentifier: "en") {
            return english
        }
        return Bundle.main.localizedString(forKey: key, value: key, table: nil)
    }

    private static func currentLanguageSnapshot() -> LanguageSnapshot {
        _ = interfaceLanguageObserver
        languageSnapshotLock.lock()
        let snapshot = languageSnapshot
        languageSnapshotLock.unlock()
        return snapshot
    }

    private static func configuredLanguage(defaults: UserDefaults = .standard) -> AppInterfaceLanguage {
        let raw = defaults.string(forKey: AppPreferenceKey.interfaceLanguage)
        return AppInterfaceLanguage(rawValue: raw ?? "") ?? .system
    }

    private static func makeLanguageSnapshot(defaults: UserDefaults = .standard) -> LanguageSnapshot {
        let language = configuredLanguage(defaults: defaults)
        let localeIdentifier = language.localeIdentifier
        return LanguageSnapshot(
            language: language,
            localeIdentifier: localeIdentifier,
            locale: Locale(identifier: localeIdentifier)
        )
    }

    private static func resolvedLocalizedString(_ key: String, localeIdentifier: String) -> String? {
        guard let bundle = localizedBundles[localeIdentifier] else {
            return nil
        }
        let value = bundle.localizedString(forKey: key, value: key, table: nil)
        if localeIdentifier == "en" {
            return value
        }
        return value == key ? nil : value
    }
}
