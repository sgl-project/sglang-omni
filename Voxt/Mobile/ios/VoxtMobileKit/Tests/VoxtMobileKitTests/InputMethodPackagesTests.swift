import Foundation
import Testing
@testable import VoxtMobileKit

/// Japanese flick, Zhuyin and pinyin come from packages a build can leave out (ios/InputMethods.yml).
/// The build lists the ones it has in `VoxtInputMethods`; the others are never offered or kept.
@Suite struct InputMethodPackagesTests {
    private let without = TextKeyboardLayoutID.offered(info: ["VoxtInputMethods": ""])

    @Test func aBuildThatListsNothingOffersEveryLayout() {
        #expect(TextKeyboardLayoutID.offered(info: nil) == TextKeyboardLayoutID.allCases)
        #expect(TextKeyboardLayoutID.offered(info: ["VoxtInputMethods": "$(VOXT_INPUT_METHODS)"]) == TextKeyboardLayoutID.allCases)
    }

    @Test func aBuildWithoutThePackagesOffersOnlyTheLayoutsItCarries() {
        #expect(without == [.englishQWERTY, .frenchAZERTY, .germanQWERTZ, .spanishQWERTY, .korean])
    }

    @Test func listedPackagesAreOffered() {
        let some = TextKeyboardLayoutID.offered(info: ["VoxtInputMethods": "ja-flick zh-Hans-pinyin"])
        #expect(some.contains(.japaneseFlick) && some.contains(.pinyin) && !some.contains(.zhuyin))
    }

    @Test func savedAndDefaultLayoutsFromAMissingPackageAreDropped() {
        let resolved = TextKeyboardState.resolve(enabledRaw: ["ja-flick", "ko-2beolsik"], activeRaw: "ja-flick", offered: without)
        #expect(resolved.enabled == [.korean] && resolved.active == .korean)
        let name = "input-methods.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(TextKeyboardState.load(from: defaults, preferredLanguages: ["ja-JP", "zh-Hans-CN"], offered: without).enabled == [.englishQWERTY])
    }

    @Test func aMissingPackageCannotBeEnabled() {
        var state = TextKeyboardState(enabled: [.englishQWERTY])
        #expect(throws: TextKeyboardLayoutError.unknownLayout) { try state.enable(rawValue: "zh-Hant-zhuyin", offered: without) }
        #expect(state.enabled == [.englishQWERTY])
    }
}
