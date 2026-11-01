import Testing
import TypingInput
@testable import SpanishInput

@Suite struct SpanishLayoutTests {
    private var rows: [[KeySpec]] {
        guard case let .keys(rows) = SpanishLayout.descriptor.surface else { return [] }
        return rows
    }
    private func key(_ text: String) -> KeySpec? { rows.flatMap { $0 }.first { $0.text == text } }

    @Test func qwertyWithAnEnneKey() {
        let letters = rows.prefix(3).map { $0.filter { $0.role == .character }.compactMap(\.text).joined() }
        #expect(letters == ["qwertyuiop", "asdfghjklñ", "zxcvbnm"])
        #expect(key("ñ")?.text(shifted: true) == "Ñ")
        #expect(rows.last?.map(\.role) == [.numbers, .space, .return])
    }

    @Test func longPressGivesSpanishAccentsOnly() {
        #expect(key("a")?.alternates == ["á"])
        #expect(key("e")?.alternates == ["é"])
        #expect(key("i")?.alternates == ["í"])
        #expect(key("o")?.alternates == ["ó"])
        #expect(key("u")?.alternates == ["ú", "ü"])
        #expect(key("n")?.alternates == ["ñ"])
        #expect(key("y")?.alternates == [])
    }

    @Test func invertedMarksAreOnTheNumbersPlane() {
        #expect(SpanishLayout.descriptor.numberExtras == ["¿", "¡"])
        #expect(SpanishLayout.descriptor.id == "es-QWERTY", "the id already saved in users' settings")
        #expect(SpanishLayout.descriptor.languages == ["es"])
    }

    @Test(arguments: [("", true), ("¿", true), ("Hola. ¡", true), ("¿Dónde estás? ", true), ("hola ", false), ("Sí, ", false)])
    func sentencesStartWithACapitalEvenAfterInvertedMarks(before: String, shifts: Bool) {
        let engine = SpanishLayout.descriptor.makeEngine()
        #expect(engine.capitalizesNextKey(context: InputContext(before: before)) == shifts)
    }

    @Test func keysInsertDirectly() {
        var engine = SpanishLayout.descriptor.makeEngine()
        #expect(engine.handle(.text("ñ"), context: InputContext(before: "")) == .passThrough)
        #expect(engine.conversion == nil)
        #expect(SpanishLayout.descriptor.makeCandidateSource == nil)
    }
}
