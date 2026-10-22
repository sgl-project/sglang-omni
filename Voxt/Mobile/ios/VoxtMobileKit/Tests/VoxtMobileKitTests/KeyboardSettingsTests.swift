import Foundation
import Testing
@testable import VoxtMobileKit

/// Settings the app writes and the keyboard reads. Translation targets
/// are never chosen for the user: a new install takes only the phone's other languages.
@Suite struct KeyboardSettingsTests {
    // MARK: Seeding from the phone's languages

    @Test func aPhoneWithOneLanguageStartsWithNoTargets() {
        #expect(KeyboardSettings.seededTargets(preferredLanguages: ["en-US"]).isEmpty)
        #expect(KeyboardSettings.seededTargets(preferredLanguages: []).isEmpty)
    }

    @Test func theOtherLanguagesAfterTheFirstBecomeTargetsInOrder() {
        #expect(KeyboardSettings.seededTargets(preferredLanguages: ["en-US", "zh-Hans-CN", "ja-JP"]) == ["zh-Hans", "ja"])
    }

    @Test func aVariantOfTheFirstLanguageIsSkipped() {
        #expect(KeyboardSettings.seededTargets(preferredLanguages: ["en-US", "en-GB", "es-ES"]) == ["es"])
    }

    @Test func twoVariantsOfOneLanguageGiveOneTarget() {
        #expect(KeyboardSettings.seededTargets(preferredLanguages: ["en-US", "es-ES", "es-MX", "pt-BR"]) == ["es", "pt"])
    }

    @Test func aLanguageOutsideTheCatalogIsSkipped() {
        #expect(KeyboardSettings.seededTargets(preferredLanguages: ["en-US", "xx-YY", "fr-FR"]) == ["fr"])
    }

    @Test func neverMoreThanThree() {
        let seeded = KeyboardSettings.seededTargets(preferredLanguages: ["en-US", "fr-FR", "de-DE", "it-IT", "ko-KR"])
        #expect(seeded == ["fr", "de", "it"])
    }

    @Test func chineseWithoutAScriptFollowsTheRegion() {
        #expect(KeyboardSettings.seededTargets(preferredLanguages: ["en-US", "zh-CN"]) == ["zh-Hans"])
        // zh-Hant is not in the translation catalog today.
        #expect(KeyboardSettings.seededTargets(preferredLanguages: ["en-US", "zh-TW"]).isEmpty)
    }

    // MARK: Names on the pills

    @Test func pillsUseEachLanguagesOwnNameCapitalized() {
        #expect(TranslationLanguages.nativeName(for: "es") == "Español")
        #expect(TranslationLanguages.nativeName(for: "ja") == "日本語")
        #expect(TranslationLanguages.nativeName(for: "de") == "Deutsch")
        #expect(TranslationLanguages.nativeName(for: "fr") == "Français")
    }

    // MARK: Editing the list

    @Test func aFourthTargetIsRefused() throws {
        var settings = KeyboardSettings(translationTargets: ["es", "ja", "de"])
        #expect(throws: KeyboardSettings.TargetError.full) { try settings.addTarget("fr") }
        #expect(settings.translationTargets == ["es", "ja", "de"])
    }

    @Test func addingAppendsAndIgnoresDuplicatesAndUnknownCodes() throws {
        var settings = KeyboardSettings()
        try settings.addTarget("es")
        try settings.addTarget("es")
        #expect(throws: KeyboardSettings.TargetError.unsupported) { try settings.addTarget("xx") }
        try settings.addTarget("ja")
        #expect(settings.translationTargets == ["es", "ja"])
    }

    @Test func movingChangesTheOrder() {
        var settings = KeyboardSettings(translationTargets: ["es", "ja", "de"])
        settings.moveTargets(fromOffsets: IndexSet(integer: 2), toOffset: 0)
        #expect(settings.translationTargets == ["de", "es", "ja"])
        settings.moveTargets(fromOffsets: IndexSet(integer: 0), toOffset: 3)
        #expect(settings.translationTargets == ["es", "ja", "de"])
    }

    @Test func removingTheLastOneLeavesNone() {
        var settings = KeyboardSettings(translationTargets: ["es"])
        settings.removeTarget("es")
        #expect(settings.translationTargets.isEmpty)
    }

    @Test func hapticFeedbackIsOnByDefault() {
        #expect(KeyboardSettings().hapticFeedback)
    }

    // MARK: Storage

    @Test func aNewInstallSeedsOnceAndKeepsTheUsersLaterChoice() throws {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        var settings = KeyboardSettings.load(from: defaults, preferredLanguages: ["en-US", "ja-JP"])
        #expect(settings.translationTargets == ["ja"])
        settings.removeTarget("ja")
        settings.save(to: defaults)
        // The phone's languages are read only on the first load; an emptied list stays empty.
        #expect(KeyboardSettings.load(from: defaults, preferredLanguages: ["en-US", "ja-JP"]).translationTargets.isEmpty)
    }

    @Test func savedSettingsRoundTrip() {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        KeyboardSettings(translationTargets: ["de", "es"], hapticFeedback: false).save(to: defaults)
        let loaded = KeyboardSettings.load(from: defaults, preferredLanguages: ["en-US"])
        #expect(loaded == KeyboardSettings(translationTargets: ["de", "es"], hapticFeedback: false))
    }

    @Test func loadingClampsToThreeAndDropsCodesOutsideTheCatalog() throws {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        let stored = #"{"translationTargets":["es","xx","ja","es","de","fr"],"hapticFeedback":true}"#
        defaults.set(Data(stored.utf8), forKey: KeyboardSettings.storageKey)
        #expect(KeyboardSettings.load(from: defaults, preferredLanguages: ["en-US"]).translationTargets == ["es", "ja", "de"])
    }

    @Test func aMissingHapticFieldMeansOn() {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        defaults.set(Data(#"{"translationTargets":["es"]}"#.utf8), forKey: KeyboardSettings.storageKey)
        #expect(KeyboardSettings.load(from: defaults, preferredLanguages: ["en-US"]).hapticFeedback)
    }

    @Test func preferencesFromBeforeTheRedesignAreNotCarriedOver() {
        // An earlier build saved 8 starter languages without asking, so nothing
        // there is the user's choice.
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        defaults.set(Data(#"{"mode":"dictate","style":"automatic","targetLanguages":["en","zh-Hans","ja","es","fr","de","ko","pt"],"activeTargetLanguage":"en"}"#.utf8),
                     forKey: "composition.preferences")
        #expect(KeyboardSettings.load(from: defaults, preferredLanguages: ["en-US", "es-ES"]).translationTargets == ["es"])
    }

    private func makeDefaults() -> (UserDefaults, String) {
        let name = "keyboard-settings.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return (defaults, name)
    }
}
