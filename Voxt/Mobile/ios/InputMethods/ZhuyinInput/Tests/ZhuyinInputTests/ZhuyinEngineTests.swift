import Testing
import TypingInput
@testable import ZhuyinInput

@Suite struct ZhuyinEngineTests {
    private let context = InputContext(before: "")

    private func typed(_ keys: String) -> (ZhuyinEngine, [InputEdit]) {
        var engine = ZhuyinEngine()
        let edits = keys.map { engine.handle(.text(String($0)), context: context) }
        return (engine, edits)
    }

    @Test func keysMarkTheReadingAndCompleteSyllablesConvert() {
        let (engine, edits) = typed("ㄋㄧˇㄏㄠ")
        #expect(edits == [.mark("ㄋ"), .mark("ㄋㄧ"), .mark("ㄋㄧˇ"), .mark("ㄋㄧˇ ㄏ"), .mark("ㄋㄧˇ ㄏㄠ")])
        #expect(engine.conversion == nil, "a half syllable is not converted")
        let (done, _) = typed("ㄋㄧˇ")
        #expect(done.conversion?.input == ["ㄋㄧˇ"])
    }

    @Test func firstToneSpaceConvertsWithoutChangingTheMark() throws {
        var (engine, _) = typed("ㄊㄚ")
        #expect(engine.conversion == nil)
        #expect(engine.handle(.space, context: context) == .none)
        let request = try #require(engine.conversion)
        #expect(request.input == ["ㄊㄚ"])
        let accepted = engine.receive([Candidate(text: "他"), Candidate(text: "她")], for: request)
        #expect(accepted)
        #expect(engine.state.candidates.map(\.text) == ["他", "她"])
        #expect(engine.handle(.space, context: context) == .insert("他", mark: ""), "space then picks the best")
        #expect(engine.state == CompositionState())
    }

    @Test func returnPicksTheBestOrKeepsTheReading() throws {
        var (engine, _) = typed("ㄋㄧˇ")
        let request = try #require(engine.conversion)
        _ = engine.receive([Candidate(text: "你")], for: request)
        #expect(engine.handle(.return, context: context) == .insert("你", mark: ""))
        var (raw, _) = typed("ㄋㄧ")
        #expect(raw.handle(.return, context: context) == .insert("ㄋㄧ", mark: ""))
    }

    @Test func deleteReopensTheSyllableAndStaleResultsAreDropped() throws {
        var (engine, _) = typed("ㄋㄧˇ")
        let request = try #require(engine.conversion)
        #expect(engine.handle(.delete, context: context) == .mark("ㄋㄧ"))
        let stale = engine.receive([Candidate(text: "你")], for: request)
        #expect(!stale)
        #expect(engine.state.candidates.isEmpty)
        #expect(engine.handle(.delete, context: context) == .mark("ㄋ"))
        #expect(engine.handle(.delete, context: context) == .mark(""))
        #expect(engine.handle(.delete, context: context) == .passThrough)
    }

    @Test func selectingTheReadingItselfKeepsIt() {
        var (engine, _) = typed("ㄋㄧˇ")
        #expect(engine.select(Candidate(text: "ㄋㄧˇ")) == .insert("ㄋㄧˇ", mark: ""))
        #expect(engine.commit() == .none)
    }

    @Test func emptyKeysAreTheKeyboardsAndUnknownKeysAreIgnored() {
        var engine = ZhuyinEngine()
        #expect(engine.handle(.space, context: context) == .passThrough)
        #expect(engine.handle(.return, context: context) == .passThrough)
        #expect(engine.handle(.delete, context: context) == .passThrough)
        #expect(engine.handle(.text("a"), context: context) == .none)
        #expect(engine.handle(.text("ˇ"), context: context) == .none, "a tone needs a syllable")
    }

    @Test func layoutKeepsItsSavedIDAndConvertsWithTheBundledData() async throws {
        let layout = ZhuyinLayout.descriptor
        #expect(layout.id == "zh-Hant-zhuyin")
        #expect(layout.languages == ["zh-Hant"])
        let source = try #require(try layout.makeCandidateSource?())
        let values = try await source.candidates(for: ConversionRequest(input: ["ㄋㄧˇ", "ㄏㄠˇ"]))
        #expect(values.first?.text == "你好")
        await #expect(throws: CandidateSourceError.unconvertible) {
            try await source.candidates(for: ConversionRequest(input: ["ㄋ"]))
        }
    }
}
