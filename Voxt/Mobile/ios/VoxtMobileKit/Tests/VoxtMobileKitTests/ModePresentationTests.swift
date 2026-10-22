import Testing
@testable import VoxtMobileKit

struct ModePresentationTests {
    @Test func keyboardShowsIntentAndTranslationTarget() {
        #expect(CompositionOptions.Mode.dictate.keyboardSymbol == "mic")
        #expect(CompositionOptions.Mode.edit.keyboardSymbol == nil)
        #expect(CompositionOptions.Mode.edit.keyboardLabel(language: nil) == "Edit")
        #expect(CompositionOptions.Mode.translate.keyboardLabel(language: "en") == "→ EN")
        #expect(CompositionOptions.Mode.translate.keyboardLabel(language: "zh-Hans") == "→ 中")
        #expect(CompositionOptions.Mode.translate.keyboardLabel(language: "ja") == "→ JA")
        #expect(CompositionOptions.Mode.write.keyboardLabel(language: nil) == "Write")
        #expect(CompositionOptions.Mode.dictate.keyboardLabel(language: "en") == nil)
    }
}
