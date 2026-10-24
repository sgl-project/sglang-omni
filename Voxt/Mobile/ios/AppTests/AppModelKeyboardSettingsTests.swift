import XCTest
@testable import VoxtMobile
import VoxtMobileKit

/// Translation languages and haptic feedback: set in the app, read by the keyboard
/// from `keyboardSettings` in the App Group.
@MainActor
final class AppModelKeyboardSettingsTests: XCTestCase {
    func testANewInstallTakesOnlyThePhonesOtherLanguages() {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        XCTAssertEqual(makeModel(defaults, languages: ["en-US", "ja-JP"]).keyboardSettings.translationTargets, ["ja"])
        let (single, singleName) = makeDefaults()
        defer { single.removePersistentDomain(forName: singleName) }
        XCTAssertEqual(makeModel(single, languages: ["en-US"]).keyboardSettings.translationTargets, [])
    }

    func testAddMoveAndRemoveReachTheKeyboard() {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        let model = makeModel(defaults, languages: ["en-US"])
        model.addTranslationTarget("es")
        model.addTranslationTarget("ja")
        model.addTranslationTarget("de")
        model.addTranslationTarget("fr")
        XCTAssertEqual(model.keyboardSettings.translationTargets, ["es", "ja", "de"], "a 4th is refused")
        model.moveTranslationTargets(fromOffsets: IndexSet(integer: 2), toOffset: 0)
        model.removeTranslationTarget("ja")
        XCTAssertEqual(KeyboardSettings.load(from: defaults, preferredLanguages: ["en-US", "fr-FR"]).translationTargets, ["de", "es"])
        XCTAssertNil(defaults.data(forKey: "composition.preferences"), "targets live only in keyboardSettings")
    }

    func testHapticFeedbackPersists() {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        let model = makeModel(defaults, languages: ["en-US"])
        XCTAssertTrue(model.keyboardSettings.hapticFeedback)
        model.setHapticFeedback(false)
        XCTAssertFalse(KeyboardSettings.load(from: defaults).hapticFeedback)
        XCTAssertFalse(makeModel(defaults, languages: ["en-US"]).keyboardSettings.hapticFeedback)
    }

    func testCleanUpMyWordsIsTheOppositeOfLiteralDictation() {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        let model = makeModel(defaults, languages: ["en-US"])
        XCTAssertTrue(model.cleansUpWords)
        model.setCleansUpWords(false)
        XCTAssertTrue(model.preferences.literalDictation)
        XCTAssertFalse(makeModel(defaults, languages: ["en-US"]).cleansUpWords)
    }

    private func makeModel(_ defaults: UserDefaults, languages: [String]) -> AppModel {
        AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedPreferredLanguages: languages)
    }

    private func makeDefaults() -> (UserDefaults, String) {
        let name = "keyboard-settings-app.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return (defaults, name)
    }
}

/// The keyboard's "Choose translation languages" pill opens the Translation screen.
@MainActor
final class TranslationRouteTests: XCTestCase {
    func testTheTranslationLinkOpensTheTranslationScreen() {
        let name = "translation-route.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults)
        XCTAssertFalse(model.showsTranslation)
        model.handle(url: URL(string: "voxtmobile://somewhere-else")!)
        XCTAssertFalse(model.showsTranslation)
        model.handle(url: URL(string: "voxtmobile://translation")!)
        XCTAssertTrue(model.showsTranslation)
    }

    /// The Voxt name at the keyboard's top left opens the app on Settings.
    func testTheSettingsLinkOpensSettings() {
        let name = "settings-route.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults)
        XCTAssertFalse(model.showsSettings)
        model.handle(url: URL(string: "voxtmobile://settings")!)
        XCTAssertTrue(model.showsSettings)
    }
}
