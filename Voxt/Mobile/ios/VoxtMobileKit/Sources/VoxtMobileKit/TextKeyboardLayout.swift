import Foundation

/// Saved layout ids and their names. Four Latin layouts carry their keys here; layouts from
/// a TypingInput package (Japanese flick, Zhuyin, Korean, pinyin) are drawn from that package.
public enum TextKeyboardLayoutID: String, Codable, CaseIterable, Sendable {
    case englishQWERTY = "en-QWERTY"
    case frenchAZERTY = "fr-AZERTY"
    case germanQWERTZ = "de-QWERTZ"
    case spanishQWERTY = "es-QWERTY"
    case japaneseFlick = "ja-flick"
    case zhuyin = "zh-Hant-zhuyin"
    case korean = "ko-2beolsik"
    case pinyin = "zh-Hans-pinyin"

    public var shortLabel: String {
        switch self {
        case .englishQWERTY: "EN"
        case .frenchAZERTY: "FR"
        case .germanQWERTZ: "DE"
        case .spanishQWERTY: "ES"
        case .japaneseFlick: "日本語"
        case .zhuyin: "注音"
        case .korean: "한"
        case .pinyin: "拼"
        }
    }

    public var title: String {
        switch self {
        case .englishQWERTY: "English"
        case .frenchAZERTY: "Français"
        case .germanQWERTZ: "Deutsch"
        case .spanishQWERTY: "Español"
        case .japaneseFlick: "日本語"
        case .zhuyin: "注音"
        case .korean: "한국어"
        case .pinyin: "简体拼音"
        }
    }
}

extension TextKeyboardLayoutID {
    /// Layouts whose package a build can leave out (ios/InputMethods.yml): their dictionaries are large.
    static let fromOptionalPackages: Set<TextKeyboardLayoutID> = [.japaneseFlick, .zhuyin, .pinyin]

    /// The layouts this build carries. It lists its optional packages in `VoxtInputMethods`
    /// (build setting `VOXT_INPUT_METHODS`); a bundle without the key, such as a test host, has all.
    public static func offered(info: [String: Any]?) -> [TextKeyboardLayoutID] {
        let listKey = "VoxtInputMethods"
        guard let listed = info?[listKey] as? String, !listed.contains("$(") else { return allCases }
        let included = Set(listed.split(separator: " ").map(String.init))
        return allCases.filter { !fromOptionalPackages.contains($0) || included.contains($0.rawValue) }
    }

    public static let offered = offered(info: Bundle.main.infoDictionary)
}

public enum TextKeyboardPlane: String, Sendable {
    case letters, numbers
}

public struct TextKeyboardKey: Equatable, Identifiable, Sendable {
    public enum Kind: String, Equatable, Sendable {
        case character, shift, delete, space, `return`, numbers, letters, voice
    }

    public var id: String
    public var kind: Kind
    public var label: String
    public var text: String?
    /// Long-press menu. Empty when the key has no accents.
    public var alternates: [String]

    public init(id: String, kind: Kind, label: String, text: String? = nil, alternates: [String] = []) {
        self.id = id
        self.kind = kind
        self.label = label
        self.text = text
        self.alternates = alternates
    }
}

public struct TextKeyboardLayout: Equatable, Sendable {
    public var id: TextKeyboardLayoutID
    private var letterRows: [[TextKeyboardKey]]
    private var numberRows: [[TextKeyboardKey]]

    public static func make(_ id: TextKeyboardLayoutID) -> TextKeyboardLayout {
        switch id {
        case .englishQWERTY: english
        case .frenchAZERTY: french
        case .germanQWERTZ: german
        case .spanishQWERTY: spanish
        case .japaneseFlick: japanese
        case .zhuyin: zhuyin
        case .korean: korean
        case .pinyin: pinyin
        }
    }

    public func rows(_ plane: TextKeyboardPlane) -> [[TextKeyboardKey]] {
        plane == .letters ? letterRows : numberRows
    }

    private static let english = TextKeyboardLayout(
        id: .englishQWERTY,
        letters: [
            row("qwertyuiop", alternates: vowelAccents),
            row("asdfghjkl", alternates: ["a": ["à", "á", "â", "ä"]]),
            controls(middle: row("zxcvbnm", alternates: ["c": ["ç"], "n": ["ñ"]]), trailingDelete: true)
        ],
        numbers: numberRows(extra: [])
    )

    // The keyboard renders the kana plane with directional gestures. The shared
    // catalog supplies the common numeric plane and persisted layout identity.
    private static let japanese = TextKeyboardLayout(
        id: .japaneseFlick,
        letters: [row("あかさ"), row("たなは"), row("まやら"), row("わ、")],
        numbers: numberRows(extra: ["¥"])
    )

    private static let zhuyin = TextKeyboardLayout(
        id: .zhuyin,
        letters: [row("ㄅㄉˇˋㄓˊ˙ㄚㄞㄢㄦ"), row("ㄆㄊㄍㄐㄔㄗㄧㄛㄟㄣ"), row("ㄇㄋㄎㄑㄕㄘㄨㄜㄠㄤ"), row("ㄈㄌㄏㄒㄖㄙㄩㄝㄡㄥ")],
        numbers: numberRows(extra: ["，", "。"])
    )

    // Keys come from ios/KoreanInput (KoreanLayout); the catalog keeps the numbers plane.
    private static let korean = TextKeyboardLayout(id: .korean, letters: [], numbers: numberRows(extra: []))
    // Keys come from ios/PinyinInput (PinyinLayout).
    private static let pinyin = TextKeyboardLayout(id: .pinyin, letters: [], numbers: numberRows(extra: ["，", "。"]))

    private static let french = TextKeyboardLayout(
        id: .frenchAZERTY,
        letters: [
            row("azertyuiop", alternates: [
                "e": ["é", "è", "ê", "ë"],
                "a": ["à", "â", "ä"],
                "u": ["ù", "û", "ü"],
                "i": ["î", "ï"],
                "o": ["ô", "œ"],
                "y": ["ÿ"]
            ]),
            row("qsdfghjklm"),
            controls(middle: row("wxcvbn", alternates: ["c": ["ç"]]) + [character("é")], trailingDelete: true)
        ],
        numbers: numberRows(extra: ["€"])
    )

    private static let german = TextKeyboardLayout(
        id: .germanQWERTZ,
        letters: [
            row("qwertzuiop") + [character("ü")],
            row("asdfghjkl") + [character("ö"), character("ä"), character("ß")],
            controls(middle: row("yxcvbnm"), trailingDelete: true)
        ],
        numbers: numberRows(extra: [])
    )

    private static let spanish = TextKeyboardLayout(
        id: .spanishQWERTY,
        letters: [
            row("qwertyuiop", alternates: [
                "e": ["é", "è", "ê"],
                "a": ["á"],
                "i": ["í"],
                "o": ["ó"],
                "u": ["ú", "ü"],
                "y": ["ý"]
            ]),
            row("asdfghjkl", alternates: ["n": ["ñ"], "u": ["ú", "ü"]]) + [character("ñ")],
            controls(middle: row("zxcvbnm"), trailingDelete: true)
        ],
        numbers: numberRows(extra: ["¿", "¡"])
    )

    private static let vowelAccents = [
        "e": ["é", "è", "ê", "ë"],
        "a": ["à", "á", "â", "ä"],
        "i": ["ì", "í", "î", "ï"],
        "o": ["ò", "ó", "ô", "ö"],
        "u": ["ù", "ú", "û", "ü"],
        "y": ["ÿ"]
    ]

    private init(id: TextKeyboardLayoutID, letters: [[TextKeyboardKey]], numbers: [[TextKeyboardKey]]) {
        self.id = id
        self.letterRows = letters + [Self.bottomRow(numbers: true)]
        self.numberRows = numbers
    }

    private static func numberRows(extra: [String]) -> [[TextKeyboardKey]] {
        [
            row("1234567890"),
            row("-/:;().,?!") + extra.map { character($0) },
            [system(.letters, id: "letters", label: "ABC")] + row("@$&'\"", alternates: ["@": ["#", "_"], "$": ["€", "£", "¥", "%"], "&": ["*", "+", "=", "|"], "'": ["`", "~", "^"], "\"": ["[", "]", "{", "}", "<", ">", "\\"]]) + [system(.delete, id: "numbers-delete", label: "delete")],
            bottomRow(numbers: false)
        ]
    }

    private static func row(_ letters: String, alternates: [String: [String]] = [:]) -> [TextKeyboardKey] {
        letters.map { character(String($0), alternates: alternates[String($0)] ?? []) }
    }

    private static func character(_ text: String, alternates: [String] = []) -> TextKeyboardKey {
        TextKeyboardKey(id: "c-\(text)", kind: .character, label: text, text: text, alternates: alternates)
    }

    private static func controls(middle: [TextKeyboardKey], trailingDelete: Bool) -> [TextKeyboardKey] {
        [system(.shift, id: "shift", label: "shift")] + middle + (trailingDelete ? [system(.delete, id: "delete", label: "delete")] : [])
    }

    private static func bottomRow(numbers: Bool) -> [TextKeyboardKey] {
        let mode: TextKeyboardKey = numbers
            ? system(.numbers, id: "numbers", label: "123")
            : system(.letters, id: "letters-bottom", label: "ABC")
        return [
            mode,
            system(.space, id: "space", label: "space", text: " "),
            system(.return, id: "return", label: "return", text: "\n")
        ]
    }

    private static func system(_ kind: TextKeyboardKey.Kind, id: String, label: String, text: String? = nil) -> TextKeyboardKey {
        TextKeyboardKey(id: id, kind: kind, label: label, text: text)
    }
}
