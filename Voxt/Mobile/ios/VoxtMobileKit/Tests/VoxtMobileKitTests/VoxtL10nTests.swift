import Foundation
import Testing
@testable import VoxtMobileKit

@Suite(.serialized)
struct VoxtL10nTests {
    @Test func latestResultEditingHasChineseLabels() {
        let (l10n, defaults, name) = makeResolver()
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(l10n.string("Edit latest result", language: "zh-Hans") == "修改刚才的结果")
        #expect(l10n.string("Say how to change it", language: "zh-Hans") == "说出修改要求")
    }
    @Test func japaneseDictionaryFailureIsTranslated() {
        let (l10n, defaults, name) = makeResolver()
        defer { defaults.removePersistentDomain(forName: name) }
        let key = "Japanese dictionary unavailable. Kana input still works."
        for language in ["zh-Hans", "zh-Hant", "ja"] {
            #expect(l10n.hasTranslation(key, language: language))
            #expect(l10n.string(key, language: language) != key)
        }
    }
    @Test func partialSyncWarningIsTranslated() {
        let (l10n, defaults, name) = makeResolver()
        defer { defaults.removePersistentDomain(forName: name) }
        let key = "Some records are too large to sync. They remain on this iPhone."
        for language in ["zh-Hans", "zh-Hant", "ja"] {
            #expect(l10n.hasTranslation(key, language: language))
            #expect(l10n.string(key, language: language) != key)
        }
    }

    @Test func nextKeyboardAccessibilityLabelIsTranslated() {
        let (l10n, defaults, name) = makeResolver()
        defer { defaults.removePersistentDomain(forName: name) }
        for language in ["zh-Hans", "zh-Hant", "ja"] {
            #expect(l10n.hasTranslation("Next keyboard", language: language))
            #expect(l10n.string("Next keyboard", language: language) != "Next keyboard")
        }
    }

    private func makeResolver() -> (VoxtL10n, UserDefaults, String) {
        let name = "l10n.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return (VoxtL10n(defaults: defaults), defaults, name)
    }

    @Test func englishIsTheBaseAndUnknownLanguagesFallBack() {
        let (l10n, _, _) = makeResolver()
        #expect(l10n.string("Tap to speak", language: "en") == "Tap to speak")
        #expect(l10n.string("Tap to speak", language: "xx-UNKNOWN") == "Tap to speak")
    }

    @Test func systemFallbackFollowsPreferredLanguages() {
        // Positive: shipped languages are picked up from the system preference order.
        #expect(VoxtL10n.resolve(preferredLanguages: ["ja-JP", "en"]) == "ja")
        #expect(VoxtL10n.resolve(preferredLanguages: ["zh-Hans-CN"]) == "zh-Hans")
        #expect(VoxtL10n.resolve(preferredLanguages: ["zh-Hant-TW"]) == "zh-Hant")
        #expect(VoxtL10n.resolve(preferredLanguages: ["zh-HK"]) == "zh-Hant")
        #expect(VoxtL10n.resolve(preferredLanguages: ["zh-SG"]) == "zh-Hans")
        #expect(VoxtL10n.resolve(preferredLanguages: ["fr-FR", "ja-JP"]) == "ja")
        // Negative: nothing shipped → English.
        #expect(VoxtL10n.resolve(preferredLanguages: ["fr-FR", "de-DE"]) == "en")
        #expect(VoxtL10n.resolve(preferredLanguages: []) == "en")
    }

    @Test func nilLanguageResolvesThroughSystemPreferences() {
        let (l10n, _, _) = makeResolver()
        let resolved = l10n.string("Tap to speak", language: nil)
        // Machine-dependent: must be one of the shipped translations, never the raw key
        // unless English is the resolved language.
        let known: Set<String> = ["Tap to speak", "轻点说话", "輕點說話", "タップして話す"]
        #expect(known.contains(resolved))
    }

    @Test func chosenLanguagesResolveKnownStrings() {
        let (l10n, _, _) = makeResolver()
        #expect(l10n.string("Tap to speak", language: "zh-Hans") == "轻点说话")
    }

    @Test func pluralEnglishOneOther() {
        let (l10n, _, _) = makeResolver()
        #expect(l10n.plural("%d days", count: 1, language: "en") == "1 day")
        #expect(l10n.plural("%d days", count: 2, language: "en") == "2 days")
    }

    @Test func pluralCJKUsesOtherOnly() {
        let (l10n, _, _) = makeResolver()
        #expect(l10n.plural("%d days", count: 1, language: "zh-Hans") == "1 天")
        #expect(l10n.plural("%d days", count: 5, language: "zh-Hans") == "5 天")
    }

    @Test func missingKeyReturnsKeyAndIsDetected() {
        let (l10n, _, _) = makeResolver()
        #expect(l10n.string("no such key exists", language: "ja") == "no such key exists")
        #expect(!l10n.hasTranslation("no such key exists", language: "ja"))
        #expect(l10n.hasTranslation("Tap to speak", language: "zh-Hans"))
    }

    @Test func preferenceIsReadFromTheInjectedDefaultsSuite() {
        let (l10n, defaults, name) = makeResolver()
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(l10n.languageCode == nil)
        #expect(l10n.string("Tap to speak") == "Tap to speak")
        defaults.set("zh-Hant", forKey: VoxtL10n.preferenceKey)
        #expect(l10n.languageCode == "zh-Hant")
        defaults.set("  ", forKey: VoxtL10n.preferenceKey)
        #expect(l10n.languageCode == nil, "blank preference follows the system")
    }

    @Test func knownStringsPerLanguage() {
        let (l10n, _, _) = makeResolver()
        // Spot checks of common screen titles.
        #expect(l10n.string("Settings", language: "zh-Hans") == "设置")
        #expect(l10n.string("Done", language: "zh-Hans") == "完成")
        #expect(l10n.string("Home", language: "zh-Hans") == "主页")
        #expect(l10n.string("History", language: "zh-Hans") == "历史")
        #expect(l10n.string("Personal dictionary", language: "zh-Hans") == "个人词典")
        #expect(l10n.string("Settings", language: "ja") == "設定")
        // Script separation actually differs.
        #expect(l10n.string("History", language: "zh-Hant") == "歷史")
        #expect(l10n.string("History", language: "ja") == "履歴")
    }

    @Test func formattedLookupSubstitutesInChosenLanguage() {
        let (l10n, _, _) = makeResolver()
        #expect(l10n.format("Tap to translate into %@", language: "zh-Hans", arguments: ["日语"]) == "轻点翻译为 日语")
        #expect(l10n.format("Connected to Voxt %@", language: "ja", arguments: ["1.16.0"]) == "Voxt 1.16.0 に接続済み")
        #expect(l10n.format("Mac returned HTTP %d.", language: "en", arguments: [503]) == "Mac returned HTTP 503.")
    }

    /// Voxt's setting is Settings › General › Local API › "Allow other apps to dictate through Voxt";
    /// Voxt itself translates the card and toggle only into Japanese and Simplified Chinese.
    @Test func macSetupNamesVoxtsLocalAPISetting() throws {
        let (l10n, _, _) = makeResolver()
        let key = "Open Voxt Settings > General > Local API and turn on “Allow other apps to dictate through Voxt”."
        let names = ["en": "Allow other apps to dictate through Voxt", "es": "Allow other apps to dictate through Voxt",
                     "ko": "Allow other apps to dictate through Voxt", "zh-Hant": "Allow other apps to dictate through Voxt",
                     "ja": "他のアプリから Voxt で音声入力を許可", "zh-Hans": "允许其他应用通过 Voxt 语音输入"]
        for (language, toggle) in names {
            #expect(l10n.hasTranslation(key, language: language))
            #expect(l10n.string(key, language: language).contains(toggle), "\(language)")
        }
        // The source catalog, from this file in VoxtMobileKit/Tests/VoxtMobileKitTests/.
        let kit = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let catalog = try String(contentsOf: kit.appendingPathComponent("Sources/VoxtMobileKit/Resources/Localizable.xcstrings"), encoding: .utf8)
        // The setting's old name, built from parts so this file doesn't name it either.
        for old in ["Remote" + " Input", "リモート" + "入力", "远程" + "输入", "遠端" + "輸入"] { #expect(!catalog.contains(old), "\(old)") }
    }

    @Test func everyCatalogKeyIsTranslatedInEveryShippedLanguage() throws {
        // Resource-level missing-key detection: compare each language's compiled
        // .strings/.stringsdict key set against English inside the package bundle.
        func keys(_ language: String) throws -> Set<String> {
            let lproj = try #require(Bundle.module.path(forResource: language, ofType: "lproj"))
            var keys = Set<String>()
            if let path = try? #require(Bundle(path: lproj)?.path(forResource: "Localizable", ofType: "strings")),
               let dict = NSDictionary(contentsOfFile: path) as? [String: String] {
                keys.formUnion(dict.keys)
            }
            if let path = Bundle(path: lproj)?.path(forResource: "Localizable", ofType: "stringsdict"),
               let dict = NSDictionary(contentsOfFile: path) as? [String: Any] {
                keys.formUnion(dict.keys)
            }
            return keys
        }
        let base = try keys("en")
        #expect(base.count > 150, "English catalog should hold the full inventory")
        for language in VoxtL10n.shippedLanguages {
            let translated = try keys(language)
            #expect(base.subtracting(translated).isEmpty,
                    "\(language) is missing: \(base.subtracting(translated).sorted())")
        }
    }
}

@Suite(.serialized)
struct DeveloperModeL10nTests {
    @Test func developerModeTextIsTranslated() {
        let name = "l10n.\(UUID().uuidString)", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let l10n = VoxtL10n(defaults: defaults)
        for key in ["Developer mode", "Turn on Developer mode?", "Developer mode lets you edit the prompts Voxt uses. Custom prompts can lower quality.",
                    "Turning Developer mode off keeps your custom prompts, but Voxt stops using them.", "Reset to default", "Reset all to default"] {
            for language in ["zh-Hans", "zh-Hant", "ja"] {
                #expect(l10n.hasTranslation(key, language: language), "\(key) / \(language)")
            }
        }
    }
}
