import Testing
@testable import ZhuyinInput

@Suite struct ZhuyinCompositionTests {
    @Test func buildsTwoSyllablesWithTones() {
        var state = ZhuyinComposition()
        for key in ["ㄋ", "ㄧ", "ˇ"] { #expect({ state.press(key) }()) }
        #expect(state.readingsForConversion == ["ㄋㄧˇ"])
        #expect({ state.press("ㄏ") }())
        #expect(state.readingsForConversion == nil)
        for key in ["ㄠ", "ˇ"] { #expect({ state.press(key) }()) }
        #expect(state.readingsForConversion == ["ㄋㄧˇ", "ㄏㄠˇ"])
        #expect(state.display == "ㄋㄧˇ ㄏㄠˇ")
    }
    @Test func firstToneSpaceAndNeutralTone() {
        var state = ZhuyinComposition()
        for key in ["ㄊ", "ㄚ", " ", "ㄉ", "ㄜ", "˙"] { #expect({ state.press(key) }()) }
        #expect(state.readingsForConversion == ["ㄊㄚ", "ㄉㄜ˙"])
    }
    @Test func incompleteToneIsRejectedAndComponentReplacementIsOrdered() {
        var state = ZhuyinComposition()
        #expect({ !state.press("ˇ") }())
        #expect({ state.press("ㄋ") }())
        #expect({ !state.press("ˇ") }())
        #expect({ !state.press("banana") }())
        #expect({ state.press("ㄧ") }())
        #expect({ state.press("ㄌ") }())
        #expect(state.display == "ㄌㄧ")
        #expect({ state.press("ˇ") }())
        #expect(state.readingsForConversion == ["ㄌㄧˇ"])
    }
    @Test func deleteReopensLastSyllableAndThenRemovesOneSymbol() {
        var state = ZhuyinComposition()
        for key in ["ㄋ", "ㄧ", "ˇ"] { _ = state.press(key) }
        state.deleteBackward()
        #expect(state.display == "ㄋㄧ")
        #expect(state.readingsForConversion == nil)
        state.deleteBackward()
        #expect(state.display == "ㄋ")
        state.deleteBackward()
        #expect(state.display.isEmpty)
        state.deleteBackward()
        #expect(state.display.isEmpty)
    }
    @Test func capsCompletedSyllablesAndResetClearsAll() {
        var state = ZhuyinComposition()
        for _ in 0..<32 { _ = state.press("ㄓ"); _ = state.press(" ") }
        #expect(state.readingsForConversion?.count == 32)
        #expect({ !state.press("ㄋ") }())
        state.reset()
        #expect(state.display.isEmpty)
        #expect(state.readingsForConversion == nil)
    }
}
