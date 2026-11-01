import Foundation
import Testing
import TypingInput
@testable import PinyinInput

@Suite struct PinyinEngineTests {
    private let context = InputContext(before: "")

    private func typed(_ letters: String) -> (PinyinEngine, [InputEdit]) {
        var engine = PinyinEngine()
        let edits = letters.map { engine.handle(.text(String($0)), context: context) }
        return (engine, edits)
    }

    @Test func lettersStayMarkedAndEveryChangeAsksForConversion() throws {
        var (engine, edits) = typed("ni")
        #expect(edits == [.mark("n"), .mark("ni")])
        let first = try #require(engine.conversion)
        #expect(first.input == ["ni"])
        _ = engine.handle(.text("h"), context: context)
        let second = try #require(engine.conversion)
        #expect(second.input == ["nih"] && second != first)
        let stale = engine.receive([Candidate(text: "你", span: 2)], for: first)
        #expect(!stale)
        #expect(engine.handle(.delete, context: context) == .mark("ni"))
        #expect(engine.conversion != second, "delete asks again")
    }

    @Test func pickingAWordCommitsItAndKeepsTheRestMarked() throws {
        var (engine, _) = typed("nihao")
        let request = try #require(engine.conversion)
        let candidates = [Candidate(text: "你好", span: 5), Candidate(text: "你", span: 2)]
        let accepted = engine.receive(candidates, for: request)
        #expect(accepted)
        #expect(engine.select(Candidate(text: "你", span: 2)) == .insert("你", mark: "hao"))
        #expect(engine.state.markedText == "hao")
        #expect(engine.conversion?.input == ["hao"])
        #expect(engine.state.candidates.isEmpty, "the old candidates are gone")
    }

    @Test func aTappedCandidateIsRememberedButSpaceIsNot() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("pinyin-engine-\(UUID().uuidString).tsv")
        let learning = PinyinLearning(fileURL: url)
        defer { learning.clear() }
        var engine = PinyinEngine(learning: learning)
        for letter in "nihao" { _ = engine.handle(.text(String(letter)), context: context) }
        let request = try #require(engine.conversion)
        _ = engine.receive([Candidate(text: "你好", span: 5, reading: "ni-hao"), Candidate(text: "拟好", span: 5, reading: "ni-hao")], for: request)
        #expect(engine.handle(.space, context: context) == .insert("你好", mark: ""))
        #expect(learning.entries.isEmpty, "space takes what the engine already ranks first")
        for letter in "nihao" { _ = engine.handle(.text(String(letter)), context: context) }
        _ = engine.receive([Candidate(text: "拟好", span: 5, reading: "ni-hao")], for: try #require(engine.conversion))
        #expect(engine.select(Candidate(text: "拟好", span: 5, reading: "ni-hao")) == .insert("拟好", mark: ""))
        #expect(learning.entries.map(\.text) == ["拟好"])
        #expect(learning.entries.map(\.reading) == ["ni-hao"])
    }

    @Test func anApostropheBetweenThePickAndTheRestIsDropped() throws {
        var (engine, _) = typed("xi'an")
        #expect(engine.state.markedText == "xi'an")
        #expect(engine.select(Candidate(text: "西", span: 2)) == .insert("西", mark: "an"))
    }

    @Test func spacePicksTheFirstCandidateAndReturnKeepsTheLetters() throws {
        var (engine, _) = typed("nihao")
        let request = try #require(engine.conversion)
        _ = engine.receive([Candidate(text: "你好", span: 5)], for: request)
        #expect(engine.handle(.space, context: context) == .insert("你好", mark: ""))
        #expect(engine.state == CompositionState())
        var (raw, _) = typed("nihao")
        #expect(raw.handle(.space, context: context) == .insert("nihao", mark: ""), "no candidates yet: the letters")
        var (letters, _) = typed("wo")
        #expect(letters.handle(.return, context: context) == .insert("wo", mark: ""))
    }

    @Test func punctuationCommitsTheBestCandidateFirst() throws {
        var (engine, _) = typed("nihao")
        let request = try #require(engine.conversion)
        _ = engine.receive([Candidate(text: "你好", span: 5)], for: request)
        #expect(engine.handle(.text("，"), context: context) == .insert("你好，", mark: ""))
    }

    /// Syllables spaced in marked text: "wo jiu shi shi" in the field, the letters
    /// still converted as typed, Return enters the letters without the spaces.
    @Test func theMarkShowsSyllablesWithSpaces() throws {
        var (engine, edits) = typed("wojiushishi")
        #expect(edits.last == .mark("wo jiu shi shi"))
        #expect(engine.state.markedText == "wo jiu shi shi")
        #expect(engine.conversion?.input == ["wojiushishi"])
        #expect(engine.handle(.delete, context: context) == .mark("wo jiu shi sh"), "a half-typed syllable stays as typed")
        #expect(engine.handle(.return, context: context) == .insert("wojiushish", mark: ""))
        #expect(typed("nh").1.last == .mark("n h"))
        #expect(typed("xi'an").1.last == .mark("xi'an"), "a typed apostrophe stays in place of the space")
        #expect(typed("nii").1.last == .mark("ni i"), "letters no syllable starts with stay as typed")
    }

    @Test func pickingTheFirstWordsLeavesTheRestSpaced() throws {
        var (engine, _) = typed("wojiushishi")
        let request = try #require(engine.conversion)
        _ = engine.receive([Candidate(text: "我就", span: 5)], for: request)
        #expect(engine.select(Candidate(text: "我就", span: 5)) == .insert("我就", mark: "shi shi"))
        #expect(engine.conversion?.input == ["shishi"])
    }

    @Test func fuzzySpellingsSplitAsOneSyllable() {
        var plain = PinyinEngine()
        var fuzzy = PinyinEngine(fuzzy: { true })
        var last: (InputEdit, InputEdit) = (.none, .none)
        for letter in "zuang" {
            last = (plain.handle(.text(String(letter)), context: context), fuzzy.handle(.text(String(letter)), context: context))
        }
        #expect(last.0 == .mark("zu ang"))
        #expect(last.1 == .mark("zuang"))
    }

    @Test func keysWithNothingComposedAreTheKeyboards() {
        var engine = PinyinEngine()
        #expect(engine.handle(.space, context: context) == .passThrough)
        #expect(engine.handle(.return, context: context) == .passThrough)
        #expect(engine.handle(.delete, context: context) == .passThrough)
        #expect(engine.handle(.text("。"), context: context) == .passThrough)
        #expect(engine.handle(.text("A"), context: context) == .passThrough, "a capital types as it is")
        #expect(engine.handle(.text("'"), context: context) == .passThrough)
        #expect(engine.conversion == nil)
    }

    @Test func deletingTheLastLetterEndsTheComposition() {
        var (engine, _) = typed("a")
        #expect(engine.handle(.delete, context: context) == .mark(""))
        #expect(engine.conversion == nil)
        #expect(engine.handle(.delete, context: context) == .passThrough)
    }
}

@Suite struct PinyinCandidateSourceTests {
    @Test func candidatesSayHowManyTypedLettersTheyReplace() async throws {
        let source = PinyinCandidateSource()
        let nihao = try await source.candidates(for: ConversionRequest(input: ["nihao"]))
        #expect(nihao.first == Candidate(text: "你好", span: 5, reading: "ni-hao"))
        #expect(nihao.contains(Candidate(text: "你", span: 2, reading: "ni")))
        let xian = try await source.candidates(for: ConversionRequest(input: ["xi'an"]))
        #expect(xian.first == Candidate(text: "西安", span: 5, reading: "xi-an"), "the apostrophe counts as typed")
        let partial = try await source.candidates(for: ConversionRequest(input: ["nii"]))
        #expect(partial.first == Candidate(text: "你", span: 2, reading: "ni"), "the trailing i stays")
        let abbreviated = try await source.candidates(for: ConversionRequest(input: ["nh"]))
        #expect(abbreviated.first == Candidate(text: "你好", span: 2, reading: "ni-hao"), "two typed letters")
        #expect(try await source.candidates(for: ConversionRequest(input: ["u"])).isEmpty)
    }
}

@Suite struct PinyinLayoutTests {
    @Test func theConfiguredLayoutRemembersPicksAndAsksForFuzzyEachConversion() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("pinyin-layout-\(UUID().uuidString).tsv")
        let learning = PinyinLearning(fileURL: url)
        defer { learning.clear() }
        let fuzzy = FuzzyFlag()
        let layout = PinyinLayout.descriptor(learning: learning, fuzzy: { fuzzy.value })
        #expect(layout.id == PinyinLayout.descriptor.id)
        let source = try #require(try layout.makeCandidateSource?())
        #expect(try await source.candidates(for: ConversionRequest(input: ["zongguo"])).first?.text != "中国")
        fuzzy.value = true
        #expect(try await source.candidates(for: ConversionRequest(input: ["zongguo"])).first?.text == "中国", "turning it on applies to the next conversion")
        var engine = layout.makeEngine()
        for letter in "nihao" { _ = engine.handle(.text(String(letter)), context: InputContext(before: "")) }
        _ = engine.select(Candidate(text: "拟好", span: 5, reading: "ni-hao"))
        #expect(learning.entries.map(\.text) == ["拟好"], "the layout's engine records into the same learning")
    }

    @Test func qwertyForSimplifiedChineseWithAConverter() throws {
        let layout = PinyinLayout.descriptor
        #expect(layout.id == "zh-Hans-pinyin")
        #expect(layout.languages == ["zh-Hans"])
        #expect(layout.numberExtras == ["，", "。"])
        #expect(layout.makeCandidateSource != nil)
        guard case let .keys(rows) = layout.surface else { Issue.record("expected keys"); return }
        let letters = rows.prefix(3).map { $0.filter { $0.role == .character }.compactMap(\.text).joined() }
        #expect(letters == ["qwertyuiop", "asdfghjkl", "zxcvbnm"])
        #expect(rows.last?.map(\.role) == [.numbers, .space, .return])
    }
}

final class FuzzyFlag: @unchecked Sendable {
    var value = false
}
