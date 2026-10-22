import Testing
import TypingInput
@testable import JapaneseInput

@Suite struct JapaneseEngineTests {
    private let context = InputContext(before: "")

    @Test func kanaBuildTheReadingAndEveryChangeAsksForConversion() throws {
        var engine = JapaneseEngine()
        #expect(engine.handle(.text("か"), context: context) == .mark("か"))
        let first = try #require(engine.conversion)
        #expect(engine.handle(.text("い"), context: context) == .mark("かい"))
        let second = try #require(engine.conversion)
        #expect(second.input == ["かい"])
        #expect(second != first)
        let stale = engine.receive([Candidate(text: "下")], for: first)
        #expect(!stale)
        let accepted = engine.receive([Candidate(text: "会")], for: second)
        #expect(accepted)
        #expect(engine.select(Candidate(text: "会")) == .insert("会", mark: ""))
        #expect(engine.state == CompositionState())
    }

    @Test func modifierCyclesTheLastKana() {
        var engine = JapaneseEngine()
        _ = engine.handle(.text("つ"), context: context)
        #expect(engine.handle(.modifier, context: context) == .mark("づ"))
        #expect(engine.handle(.modifier, context: context) == .mark("っ"))
        var empty = JapaneseEngine()
        #expect(empty.handle(.modifier, context: context) == .none)
    }

    @Test func spaceAndReturnKeepTheKanaAndDeleteEditsIt() {
        var engine = JapaneseEngine()
        _ = engine.handle(.text("か"), context: context)
        _ = engine.handle(.text("な"), context: context)
        #expect(engine.handle(.delete, context: context) == .mark("か"))
        #expect(engine.handle(.space, context: context) == .insert("か", mark: ""))
        _ = engine.handle(.text("あ"), context: context)
        #expect(engine.handle(.return, context: context) == .insert("あ", mark: ""))
        #expect(engine.handle(.space, context: context) == .passThrough)
        #expect(engine.handle(.return, context: context) == .passThrough)
        #expect(engine.handle(.delete, context: context) == .passThrough)
    }

    @Test func readingStopsAtTheConverterLimit() {
        var engine = JapaneseEngine()
        for _ in 0..<JapaneseConverter.maximumReadingCount { _ = engine.handle(.text("あ"), context: context) }
        #expect(engine.handle(.text("あ"), context: context) == .none)
        #expect(engine.state.markedText.count == JapaneseConverter.maximumReadingCount)
    }

    @Test func layoutKeepsItsSavedIDAndConvertsWithTheBundledDictionary() async throws {
        let layout = JapaneseLayout.descriptor
        #expect(layout.id == "ja-flick")
        #expect(layout.languages == ["ja"])
        #expect(layout.numberExtras == ["¥"])
        let source = try #require(try layout.makeCandidateSource?())
        let values = try await source.candidates(for: ConversionRequest(input: ["かいぎ"]))
        #expect(values.contains(Candidate(text: "会議")))
    }
}
