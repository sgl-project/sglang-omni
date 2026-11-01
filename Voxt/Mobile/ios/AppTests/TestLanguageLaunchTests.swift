import XCTest
@testable import VoxtMobile
import VoxtMobileKit

/// Screenshot runs in every language: VOXT_TEST_LANGUAGE writes the interface language into the
/// App Group at launch, so the app and the keyboard both switch.
final class TestLanguageLaunchTests: XCTestCase {
    func testAShippedLanguageIsWrittenForAppAndKeyboard() {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        defaults.set("en", forKey: VoxtL10n.preferenceKey)
        AppModel.applyTestLanguage(["VOXT_TEST_LANGUAGE": "ko"], to: defaults)
        XCTAssertEqual(defaults.string(forKey: VoxtL10n.preferenceKey), "ko")
    }

    func testAnUnknownCodeOrNoSettingChangesNothing() {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        defaults.set("ja", forKey: VoxtL10n.preferenceKey)
        AppModel.applyTestLanguage(["VOXT_TEST_LANGUAGE": "xx"], to: defaults)
        AppModel.applyTestLanguage([:], to: defaults)
        XCTAssertEqual(defaults.string(forKey: VoxtL10n.preferenceKey), "ja")
    }

    private func makeDefaults() -> (UserDefaults, String) {
        let name = "test-language.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return (defaults, name)
    }
}
