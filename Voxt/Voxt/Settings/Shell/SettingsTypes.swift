// SettingsTypes.swift
// Provides Settings Types for settings shell.

import SwiftUI

nonisolated extension Notification.Name {
    static let voxtSettingsSelectTab = Notification.Name("voxt.settings.selectTab")
    static let voxtSettingsNavigate = Notification.Name("voxt.settings.navigate")
    static let voxtInterfaceLanguageDidChange = Notification.Name("voxt.interfaceLanguage.didChange")
    static let voxtRemoteProviderConfigurationsDidChange = Notification.Name("voxt.remoteProviderConfigurations.didChange")
    static let voxtRemoteLLMProviderConfigurationsDidChange = Notification.Name("voxt.remoteLLMProviderConfigurations.didChange")
    static let voxtPermissionsDidChange = Notification.Name("voxt.permissions.didChange")
    static let voxtSelectedInputDeviceDidChange = Notification.Name("voxt.selectedInputDevice.didChange")
    static let voxtAudioInputDevicesDidChange = Notification.Name("voxt.audioInputDevices.didChange")
    static let voxtOverlayAppearanceDidChange = Notification.Name("voxt.overlayAppearance.didChange")
    static let voxtFeatureSettingsDidChange = Notification.Name("voxt.feature-settings.did-change")
    static let voxtMeetingFileResourceWaitDidChange = Notification.Name("voxt.meeting-file.resource-wait.did-change")
    static let voxtHotkeyDidTrigger = Notification.Name("voxt.hotkey.did-trigger")
    static let voxtModelStorageAuthorizationDidChange = Notification.Name("voxt.model-storage.authorization.did-change")
}

nonisolated enum AppInterfaceLanguage: String, CaseIterable, Identifiable {
    case system
    case english = "en"
    case chineseSimplified = "zh-Hans"
    case japanese = "ja"

    var id: String { rawValue }

    var titleKey: LocalizedStringKey {
        switch self {
        case .system: return "System Default"
        case .english: return "English"
        case .chineseSimplified: return "Chinese (Simplified)"
        case .japanese: return "Japanese"
        }
    }

    var title: String { AppLocalization.localizedString(rawTitleKey) }

    private var rawTitleKey: String {
        switch self {
        case .system: return "System Default"
        case .english: return "English"
        case .chineseSimplified: return "Chinese (Simplified)"
        case .japanese: return "Japanese"
        }
    }

    var localeIdentifier: String {
        switch self {
        case .system:
            return Self.resolvedSystemLanguage.rawValue
        case .english:
            return "en"
        case .chineseSimplified:
            return "zh-Hans"
        case .japanese:
            return "ja"
        }
    }

    var locale: Locale {
        Locale(identifier: localeIdentifier)
    }

    static var resolvedSystemLanguage: AppInterfaceLanguage {
        guard let preferred = Locale.preferredLanguages.first?.lowercased() else {
            return .english
        }
        if preferred.hasPrefix("zh") {
            return .chineseSimplified
        }
        if preferred.hasPrefix("ja") {
            return .japanese
        }
        if preferred.hasPrefix("en") {
            return .english
        }
        return .english
    }
}

struct UserMainLanguageOption: Identifiable, Hashable {
    let code: String
    let promptName: String
    let aliases: [String]

    var id: String { code }

    func title(locale: Locale = AppLocalization.locale) -> String {
        switch code {
        case "zh-hans":
            return AppLocalization.localizedString("Chinese (Simplified)")
        case "zh-hant":
            return AppLocalization.localizedString("Chinese (Traditional)")
        default:
            break
        }
        return locale.localizedString(forLanguageCode: code) ?? promptName
    }

    func matches(_ query: String, locale: Locale = AppLocalization.locale) -> Bool {
        let normalizedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalizedQuery.isEmpty else { return true }
        let haystack = ([code, promptName, title(locale: locale)] + aliases)
            .joined(separator: " ")
            .lowercased()
        return haystack.contains(normalizedQuery)
    }

    nonisolated static let fallbackCode = "en"

    nonisolated static let all: [UserMainLanguageOption] = [
        .init(code: "af", promptName: "Afrikaans", aliases: []),
        .init(code: "am", promptName: "Amharic", aliases: []),
        .init(code: "ar", promptName: "Arabic", aliases: []),
        .init(code: "as", promptName: "Assamese", aliases: []),
        .init(code: "az", promptName: "Azerbaijani", aliases: []),
        .init(code: "be", promptName: "Belarusian", aliases: []),
        .init(code: "bg", promptName: "Bulgarian", aliases: []),
        .init(code: "bn", promptName: "Bengali", aliases: []),
        .init(code: "bo", promptName: "Tibetan", aliases: []),
        .init(code: "br", promptName: "Breton", aliases: []),
        .init(code: "bs", promptName: "Bosnian", aliases: []),
        .init(code: "ca", promptName: "Catalan", aliases: []),
        .init(code: "cs", promptName: "Czech", aliases: []),
        .init(code: "cy", promptName: "Welsh", aliases: []),
        .init(code: "da", promptName: "Danish", aliases: []),
        .init(code: "de", promptName: "German", aliases: []),
        .init(code: "el", promptName: "Greek", aliases: []),
        .init(code: "en", promptName: "English", aliases: []),
        .init(code: "es", promptName: "Spanish", aliases: []),
        .init(code: "et", promptName: "Estonian", aliases: []),
        .init(code: "eu", promptName: "Basque", aliases: []),
        .init(code: "fa", promptName: "Persian", aliases: ["Farsi"]),
        .init(code: "fi", promptName: "Finnish", aliases: []),
        .init(code: "fo", promptName: "Faroese", aliases: []),
        .init(code: "fr", promptName: "French", aliases: []),
        .init(code: "gl", promptName: "Galician", aliases: []),
        .init(code: "gu", promptName: "Gujarati", aliases: []),
        .init(code: "ha", promptName: "Hausa", aliases: []),
        .init(code: "he", promptName: "Hebrew", aliases: []),
        .init(code: "hi", promptName: "Hindi", aliases: []),
        .init(code: "hr", promptName: "Croatian", aliases: []),
        .init(code: "ht", promptName: "Haitian Creole", aliases: []),
        .init(code: "hu", promptName: "Hungarian", aliases: []),
        .init(code: "hy", promptName: "Armenian", aliases: []),
        .init(code: "id", promptName: "Indonesian", aliases: ["Bahasa Indonesia"]),
        .init(code: "is", promptName: "Icelandic", aliases: []),
        .init(code: "it", promptName: "Italian", aliases: []),
        .init(code: "ja", promptName: "Japanese", aliases: []),
        .init(code: "jv", promptName: "Javanese", aliases: []),
        .init(code: "ka", promptName: "Georgian", aliases: []),
        .init(code: "kk", promptName: "Kazakh", aliases: []),
        .init(code: "km", promptName: "Khmer", aliases: []),
        .init(code: "kn", promptName: "Kannada", aliases: []),
        .init(code: "ko", promptName: "Korean", aliases: []),
        .init(code: "yue", promptName: "Cantonese", aliases: ["Yue Chinese"]),
        .init(code: "la", promptName: "Latin", aliases: []),
        .init(code: "lb", promptName: "Luxembourgish", aliases: []),
        .init(code: "lo", promptName: "Lao", aliases: []),
        .init(code: "lt", promptName: "Lithuanian", aliases: []),
        .init(code: "lv", promptName: "Latvian", aliases: []),
        .init(code: "mg", promptName: "Malagasy", aliases: []),
        .init(code: "mi", promptName: "Maori", aliases: ["Māori"]),
        .init(code: "mk", promptName: "Macedonian", aliases: []),
        .init(code: "ml", promptName: "Malayalam", aliases: []),
        .init(code: "mn", promptName: "Mongolian", aliases: []),
        .init(code: "mr", promptName: "Marathi", aliases: []),
        .init(code: "ms", promptName: "Malay", aliases: []),
        .init(code: "mt", promptName: "Maltese", aliases: []),
        .init(code: "my", promptName: "Burmese", aliases: ["Myanmar"]),
        .init(code: "ne", promptName: "Nepali", aliases: []),
        .init(code: "nl", promptName: "Dutch", aliases: []),
        .init(code: "nn", promptName: "Norwegian Nynorsk", aliases: []),
        .init(code: "no", promptName: "Norwegian", aliases: []),
        .init(code: "oc", promptName: "Occitan", aliases: []),
        .init(code: "pa", promptName: "Punjabi", aliases: []),
        .init(code: "pl", promptName: "Polish", aliases: []),
        .init(code: "ps", promptName: "Pashto", aliases: []),
        .init(code: "pt", promptName: "Portuguese", aliases: []),
        .init(code: "ro", promptName: "Romanian", aliases: []),
        .init(code: "ru", promptName: "Russian", aliases: []),
        .init(code: "sa", promptName: "Sanskrit", aliases: []),
        .init(code: "sd", promptName: "Sindhi", aliases: []),
        .init(code: "si", promptName: "Sinhala", aliases: []),
        .init(code: "sk", promptName: "Slovak", aliases: []),
        .init(code: "sl", promptName: "Slovenian", aliases: []),
        .init(code: "sn", promptName: "Shona", aliases: []),
        .init(code: "so", promptName: "Somali", aliases: []),
        .init(code: "sq", promptName: "Albanian", aliases: []),
        .init(code: "sr", promptName: "Serbian", aliases: []),
        .init(code: "su", promptName: "Sundanese", aliases: []),
        .init(code: "sv", promptName: "Swedish", aliases: []),
        .init(code: "sw", promptName: "Swahili", aliases: []),
        .init(code: "ta", promptName: "Tamil", aliases: []),
        .init(code: "te", promptName: "Telugu", aliases: []),
        .init(code: "tg", promptName: "Tajik", aliases: []),
        .init(code: "th", promptName: "Thai", aliases: []),
        .init(code: "tk", promptName: "Turkmen", aliases: []),
        .init(code: "tl", promptName: "Tagalog", aliases: ["Filipino"]),
        .init(code: "tr", promptName: "Turkish", aliases: []),
        .init(code: "tt", promptName: "Tatar", aliases: []),
        .init(code: "uk", promptName: "Ukrainian", aliases: []),
        .init(code: "ur", promptName: "Urdu", aliases: []),
        .init(code: "uz", promptName: "Uzbek", aliases: []),
        .init(code: "vi", promptName: "Vietnamese", aliases: []),
        .init(code: "yi", promptName: "Yiddish", aliases: []),
        .init(code: "yo", promptName: "Yoruba", aliases: []),
        .init(code: "zh-hans", promptName: "Simplified Chinese", aliases: ["Chinese", "Mandarin", "Chinese (Simplified)"]),
        .init(code: "zh-hant", promptName: "Traditional Chinese", aliases: ["Chinese", "Mandarin", "Chinese (Traditional)", "Traditional Chinese", "zh-TW", "zh-HK"])
    ]

    nonisolated static func option(for code: String) -> UserMainLanguageOption? {
        let normalized = normalizedCode(for: code)
        return all.first { $0.code == normalized }
    }

    static func sanitizedSelection(_ codes: [String]) -> [String] {
        var seen = Set<String>()
        let sanitized: [String] = codes.compactMap { (code: String) -> String? in
            let normalized = normalizedCode(for: code)
            guard option(for: normalized) != nil, seen.insert(normalized).inserted else { return nil }
            return normalized
        }
        return sanitized.isEmpty ? defaultSelectionCodes() : sanitized
    }

    static func storedSelection(from rawValue: String?) -> [String] {
        guard let rawValue,
              let data = rawValue.data(using: .utf8),
              let decoded = try? JSONDecoder().decode([String].self, from: data)
        else {
            return defaultSelectionCodes()
        }
        return sanitizedSelection(decoded)
    }

    static func storageValue(for codes: [String]) -> String {
        let sanitized = sanitizedSelection(codes)
        guard let data = try? JSONEncoder().encode(sanitized),
              let text = String(data: data, encoding: .utf8)
        else {
            return "[\"\(fallbackCode)\"]"
        }
        return text
    }

    static func defaultSelectionCodes(preferredLanguages: [String] = Locale.preferredLanguages) -> [String] {
        [fallbackOption(preferredLanguages: preferredLanguages).code]
    }

    static var defaultStoredSelectionValue: String {
        storageValue(for: defaultSelectionCodes())
    }

    static func fallbackOption(preferredLanguages: [String] = Locale.preferredLanguages) -> UserMainLanguageOption {
        for identifier in preferredLanguages {
            let normalized = normalizedCode(for: identifier)
            guard !normalized.isEmpty else { continue }
            if let matched = option(for: normalized) {
                return matched
            }
        }
        return option(for: fallbackCode) ?? all[0]
    }

    nonisolated var isChinese: Bool {
        code == "zh-hans" || code == "zh-hant"
    }

    nonisolated var isTraditionalChinese: Bool {
        code == "zh-hant"
    }

    var baseLanguageCode: String {
        switch code {
        case "zh-hans", "zh-hant":
            return "zh"
        default:
            return code.split(separator: "-").first.map(String.init) ?? code
        }
    }

    nonisolated private static func normalizedCode(for rawValue: String) -> String {
        let normalized = rawValue
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard !normalized.isEmpty else { return normalized }

        if normalized == "zh" || normalized.hasPrefix("zh-cn") || normalized.hasPrefix("zh-sg") || normalized.hasPrefix("zh-hans") {
            return "zh-hans"
        }
        if normalized.hasPrefix("zh-tw") || normalized.hasPrefix("zh-hk") || normalized.hasPrefix("zh-mo") || normalized.hasPrefix("zh-hant") {
            return "zh-hant"
        }

        let baseCode = normalized.split(separator: "-").first.map(String.init) ?? normalized
        if all.contains(where: { $0.code == baseCode }) {
            return baseCode
        }
        return normalized
    }
}

enum TranslationTargetLanguage: String, CaseIterable, Identifiable {
    case english
    case chineseSimplified
    case chineseTraditional
    case japanese
    case korean
    case spanish
    case french
    case german
    case dutch
    case turkish
    case polish
    case ukrainian
    case portuguese
    case italian
    case russian
    case arabic
    case hebrew
    case hindi
    case thai
    case vietnamese
    case indonesian
    case malay

    var id: String { rawValue }

    var titleKey: LocalizedStringKey {
        switch self {
        case .english: return "English"
        case .chineseSimplified: return "Chinese (Simplified)"
        case .chineseTraditional: return "Chinese (Traditional)"
        case .japanese: return "Japanese"
        case .korean: return "Korean"
        case .spanish: return "Spanish"
        case .french: return "French"
        case .german: return "German"
        case .dutch: return "Dutch"
        case .turkish: return "Turkish"
        case .polish: return "Polish"
        case .ukrainian: return "Ukrainian"
        case .portuguese: return "Portuguese"
        case .italian: return "Italian"
        case .russian: return "Russian"
        case .arabic: return "Arabic"
        case .hebrew: return "Hebrew"
        case .hindi: return "Hindi"
        case .thai: return "Thai"
        case .vietnamese: return "Vietnamese"
        case .indonesian: return "Indonesian"
        case .malay: return "Malay"
        }
    }

    var title: String { AppLocalization.localizedString(rawTitleKey) }

    private var rawTitleKey: String {
        switch self {
        case .english: return "English"
        case .chineseSimplified: return "Chinese (Simplified)"
        case .chineseTraditional: return "Chinese (Traditional)"
        case .japanese: return "Japanese"
        case .korean: return "Korean"
        case .spanish: return "Spanish"
        case .french: return "French"
        case .german: return "German"
        case .dutch: return "Dutch"
        case .turkish: return "Turkish"
        case .polish: return "Polish"
        case .ukrainian: return "Ukrainian"
        case .portuguese: return "Portuguese"
        case .italian: return "Italian"
        case .russian: return "Russian"
        case .arabic: return "Arabic"
        case .hebrew: return "Hebrew"
        case .hindi: return "Hindi"
        case .thai: return "Thai"
        case .vietnamese: return "Vietnamese"
        case .indonesian: return "Indonesian"
        case .malay: return "Malay"
        }
    }

    var instructionName: String {
        switch self {
        case .english: return "English"
        case .chineseSimplified: return "Simplified Chinese"
        case .chineseTraditional: return "Traditional Chinese"
        case .japanese: return "Japanese"
        case .korean: return "Korean"
        case .spanish: return "Spanish"
        case .french: return "French"
        case .german: return "German"
        case .dutch: return "Dutch"
        case .turkish: return "Turkish"
        case .polish: return "Polish"
        case .ukrainian: return "Ukrainian"
        case .portuguese: return "Portuguese"
        case .italian: return "Italian"
        case .russian: return "Russian"
        case .arabic: return "Arabic"
        case .hebrew: return "Hebrew"
        case .hindi: return "Hindi"
        case .thai: return "Thai"
        case .vietnamese: return "Vietnamese"
        case .indonesian: return "Indonesian"
        case .malay: return "Malay"
        }
    }

    var translationScriptConstraint: String? {
        switch self {
        case .chineseSimplified:
            return "Use Simplified Chinese characters only. Do not output Traditional Chinese characters."
        case .chineseTraditional:
            return "Use Traditional Chinese characters only. Do not output Simplified Chinese characters."
        default:
            return nil
        }
    }
}

enum HistoryRetentionPeriod: String, Identifiable {
    case oneDay
    case sevenDays
    case fifteenDays
    case thirtyDays
    case ninetyDays
    case oneHundredEightyDays
    case oneYear
    case forever

    static var allCases: [HistoryRetentionPeriod] {
        [
            .oneYear,
            .oneHundredEightyDays,
            .ninetyDays,
            .thirtyDays,
            .sevenDays
        ]
    }

    var id: String { rawValue }

    var days: Int? {
        switch self {
        case .oneDay:
            return 1
        case .sevenDays:
            return 7
        case .fifteenDays:
            return 15
        case .thirtyDays:
            return 30
        case .ninetyDays:
            return 90
        case .oneHundredEightyDays:
            return 180
        case .oneYear:
            return 365
        case .forever:
            return nil
        }
    }

    var title: String {
        switch self {
        case .oneDay:
            return AppLocalization.localizedString("1 Day")
        case .sevenDays:
            return AppLocalization.localizedString("1 Week")
        case .fifteenDays:
            return AppLocalization.localizedString("15 Days")
        case .thirtyDays:
            return AppLocalization.localizedString("30 Days")
        case .ninetyDays:
            return AppLocalization.localizedString("3 Months")
        case .oneHundredEightyDays:
            return AppLocalization.localizedString("6 Months")
        case .oneYear:
            return AppLocalization.localizedString("1 Year")
        case .forever:
            return AppLocalization.localizedString("Forever")
        }
    }
}

enum HistoryRetentionCount: String, Identifiable {
    case unlimited
    case fifty
    case oneHundred
    case threeHundred
    case fiveHundred
    case oneThousand

    static var allCases: [HistoryRetentionCount] {
        [
            .unlimited,
            .fifty,
            .oneHundred,
            .threeHundred,
            .fiveHundred,
            .oneThousand
        ]
    }

    var id: String { rawValue }

    var count: Int? {
        switch self {
        case .unlimited:
            return nil
        case .fifty:
            return 50
        case .oneHundred:
            return 100
        case .threeHundred:
            return 300
        case .fiveHundred:
            return 500
        case .oneThousand:
            return 1000
        }
    }

    var title: String {
        switch self {
        case .unlimited:
            return AppLocalization.localizedString("Unlimited")
        case .fifty:
            return "50"
        case .oneHundred:
            return "100"
        case .threeHundred:
            return "300"
        case .fiveHundred:
            return "500"
        case .oneThousand:
            return "1000"
        }
    }
}

enum InteractionSoundPreset: String, CaseIterable, Identifiable, Codable, Sendable {
    case soft
    case glass
    case funk
    case submarine
    case basso
    case bottle
    case frog
    case hero
    case purr
    case sosumi

    var id: String { rawValue }

    var titleKey: LocalizedStringKey {
        switch self {
        case .soft: return "Soft (Pop/Tink)"
        case .glass: return "Ping"
        case .funk: return "Morse"
        case .submarine: return "Submarine"
        case .basso: return "Basso"
        case .bottle: return "Bottle"
        case .frog: return "Frog"
        case .hero: return "Hero"
        case .purr: return "Purr"
        case .sosumi: return "Sosumi"
        }
    }

    var title: String { AppLocalization.localizedString(rawTitleKey) }

    private var rawTitleKey: String {
        switch self {
        case .soft: return "Soft (Pop/Tink)"
        case .glass: return "Ping"
        case .funk: return "Morse"
        case .submarine: return "Submarine"
        case .basso: return "Basso"
        case .bottle: return "Bottle"
        case .frog: return "Frog"
        case .hero: return "Hero"
        case .purr: return "Purr"
        case .sosumi: return "Sosumi"
        }
    }
}

enum VoiceEndCommandPreset: String, CaseIterable, Identifiable {
    case over
    case end
    case wanBi
    case haoLe
    case custom

    var id: String { rawValue }

    var title: String {
        switch self {
        case .over:
            return "over"
        case .end:
            return "end"
        case .wanBi:
            return "完毕"
        case .haoLe:
            return "好了"
        case .custom:
            return AppLocalization.localizedString("Custom")
        }
    }

    var resolvedCommand: String? {
        switch self {
        case .over:
            return "over"
        case .end:
            return "end"
        case .wanBi:
            return "完毕"
        case .haoLe:
            return "好了"
        case .custom:
            return nil
        }
    }
}
