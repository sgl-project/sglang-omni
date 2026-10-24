import TypingInput

public enum KoreanLayout {
    /// Standard 두벌식 rows. Shift gives the doubled consonants and ㅒ ㅖ.
    public static let descriptor = LayoutDescriptor(
        id: "ko-2beolsik",
        displayName: "한국어",
        shortLabel: "한",
        languages: ["ko"],
        surface: .keys(rows: [
            keys("ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔ", shifted: ["ㅂ": "ㅃ", "ㅈ": "ㅉ", "ㄷ": "ㄸ", "ㄱ": "ㄲ", "ㅅ": "ㅆ", "ㅐ": "ㅒ", "ㅔ": "ㅖ"]),
            keys("ㅁㄴㅇㄹㅎㅗㅓㅏㅣ"),
            [.shift] + keys("ㅋㅌㅊㅍㅠㅜㅡ") + [.delete],
            [.numbers, .space, .return],
        ]),
        makeEngine: { HangulEngine() }
    )

    private static func keys(_ jamo: String, shifted: [Character: String] = [:]) -> [KeySpec] {
        jamo.map { KeySpec.character(String($0), shifted: shifted[$0] ?? String($0)) }
    }
}
