import TypingInput

public enum SpanishLayout {
    private static let accents: [Character: [String]] = [
        "a": ["á"], "e": ["é"], "i": ["í"], "o": ["ó"], "u": ["ú", "ü"], "n": ["ñ"],
    ]

    public static let descriptor = LayoutDescriptor(
        id: "es-QWERTY",
        displayName: "Español",
        shortLabel: "ES",
        languages: ["es"],
        surface: .keys(rows: [
            keys("qwertyuiop"),
            keys("asdfghjklñ"),
            [.shift] + keys("zxcvbnm") + [.delete],
            [.numbers, .space, .return],
        ]),
        numberExtras: ["¿", "¡"],
        makeEngine: { DirectInputEngine(capitalization: .sentences(openers: ["¿", "¡"])) }
    )

    private static func keys(_ letters: String) -> [KeySpec] {
        letters.map { KeySpec.character(String($0), alternates: accents[$0] ?? []) }
    }
}
