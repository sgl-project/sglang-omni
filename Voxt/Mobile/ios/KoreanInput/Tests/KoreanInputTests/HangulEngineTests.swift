import Testing
import TypingInput
@testable import KoreanInput

/// Types jamo one by one and applies each edit the way the keyboard host does:
/// committed text goes into `field`, the mark is shown after it.
private struct Field {
    var engine = HangulEngine()
    var field = ""
    var mark = ""
    var text: String { field + mark }

    mutating func type(_ jamo: String) {
        for character in jamo { apply(engine.handle(.text(String(character)), context: InputContext(before: field))) }
    }
    mutating func press(_ key: InputKey) { apply(engine.handle(key, context: InputContext(before: field))) }
    mutating func apply(_ edit: InputEdit) {
        switch edit {
        case .none: break
        case let .mark(text): mark = text
        case let .insert(text, mark: next): field += text; mark = next
        case .passThrough:
            // The host's own default: delete or insert the key's text.
            if !mark.isEmpty { Issue.record("passThrough while composing \(mark)") }
            field = String(field.dropLast())
        }
        #expect(engine.state.markedText == mark)
    }
}

@Suite struct HangulEngineTests {
    @Test(arguments: [
        ("ㅎㅏㄴㄱㅡㄹ", "한글"),
        ("ㄷㅏㄹㄱ", "닭"),
        ("ㅇㅏㄴㅈㅇㅏ", "앉아"),
        ("ㅇㅏㄴㅈㅏ", "안자"),
        ("ㄷㅏㄹㄱㅏ", "달가"),
        ("ㅎㅏㄴㅏ", "하나"),
        ("ㄱㅏㅆㅓ", "가써"),
        ("ㅇㅗㅐ", "왜"),
        ("ㄱㅜㅓㄴ", "권"),
        ("ㅇㅡㅣㅅㅏ", "의사"),
        ("ㄸㅓㄱ", "떡"),
        ("ㅂㅏㅂㅅ", "밦"),
        ("ㄱㄱ", "ㄱㄱ"),
        ("ㅏㅇ", "ㅏㅇ"),
        ("ㅏㅏ", "ㅏㅏ"),
        ("ㄱㅏㄸ", "가ㄸ"),
        ("ㅅㅏㄹㅁㅇㅡㄴ", "삶은"),
    ])
    func composes(jamo: String, expected: String) {
        var field = Field()
        field.type(jamo)
        #expect(field.text == expected)
    }

    @Test func backspaceRemovesOneJamoAtATime() {
        var field = Field()
        field.type("ㄷㅏㄹㄱ")
        var steps: [String] = []
        for _ in 0..<4 { field.press(.delete); steps.append(field.text) }
        #expect(steps == ["달", "다", "ㄷ", ""])
        field.type("ㅇㅗㅐ")
        field.press(.delete)
        #expect(field.text == "오")
    }

    @Test func backspaceAfterACommittedSyllableIsTheHostsDelete() {
        var field = Field()
        field.type("ㅎㅏㄴㄱ")
        #expect(field.field == "한" && field.mark == "ㄱ")
        field.press(.delete)
        #expect(field.text == "한" && field.mark.isEmpty)
        field.press(.delete)
        #expect(field.text == "", "the host deletes the committed 한 in one step")
    }

    @Test func spaceReturnAndPunctuationCommitTheSyllable() {
        var field = Field()
        field.type("ㅎㅏㄴ")
        field.press(.space)
        #expect(field.text == "한 " && field.mark.isEmpty)
        field.type("ㄱㅡㄹ")
        field.press(.return)
        #expect(field.text == "한 글\n")
        field.type("ㄷㅏ")
        field.press(.text("."))
        #expect(field.text == "한 글\n다.")
        #expect(field.engine.state == CompositionState())
    }

    @Test func commitKeepsWhatWasTypedAndResetForgetsIt() {
        var engine = HangulEngine()
        _ = engine.handle(.text("ㄱ"), context: InputContext(before: ""))
        _ = engine.handle(.text("ㅏ"), context: InputContext(before: ""))
        #expect(engine.commit() == .insert("가", mark: ""))
        #expect(engine.commit() == .none)
        _ = engine.handle(.text("ㄴ"), context: InputContext(before: ""))
        engine.reset()
        #expect(engine.state == CompositionState())
        #expect(engine.conversion == nil, "Korean needs no dictionary")
    }

    @Test func spaceAndDeleteWithNothingComposedAreTheHosts() {
        var engine = HangulEngine()
        #expect(engine.handle(.space, context: InputContext(before: "")) == .passThrough)
        #expect(engine.handle(.delete, context: InputContext(before: "")) == .passThrough)
        #expect(engine.handle(.text("7"), context: InputContext(before: "")) == .passThrough)
    }
}

@Suite struct KoreanLayoutTests {
    @Test func layoutIsTwoBeolsikWithShiftedDoubles() throws {
        let layout = KoreanLayout.descriptor
        #expect(layout.id == "ko-2beolsik")
        #expect(layout.languages == ["ko"])
        #expect(layout.makeCandidateSource == nil)
        guard case let .keys(rows) = layout.surface else { Issue.record("expected keys"); return }
        let letters = rows.prefix(3).map { $0.filter { $0.role == .character }.compactMap { $0.text(shifted: false) }.joined() }
        #expect(letters == ["ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔ", "ㅁㄴㅇㄹㅎㅗㅓㅏㅣ", "ㅋㅌㅊㅍㅠㅜㅡ"])
        let shifted = rows[0].compactMap { $0.text(shifted: true) }.joined()
        #expect(shifted == "ㅃㅉㄸㄲㅆㅛㅕㅑㅒㅖ")
        #expect(rows.flatMap { $0 }.contains { $0.role == .shift })
    }
}
