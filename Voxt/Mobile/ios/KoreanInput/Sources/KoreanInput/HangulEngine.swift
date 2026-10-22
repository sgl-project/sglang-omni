import TypingInput

/// 두벌식 composition. The mark is the syllable being typed; it is inserted
/// for good when the next key can't join it. Backspace removes the last typed
/// jamo, so 닭 steps back through 달, 다, ㄷ.
public struct HangulEngine: InputEngine {
    /// Jamo typed for the current syllable, in order. Shifted doubles (ㄲ ㅆ)
    /// are one stroke; compound vowels and finals (ㅘ ㄺ) are two.
    private var strokes: [Character] = []
    public private(set) var state = CompositionState()
    public var conversion: ConversionRequest? { nil }

    public init() {}

    public mutating func handle(_ key: InputKey, context: InputContext) -> InputEdit {
        switch key {
        case let .text(text):
            guard text.count == 1, let jamo = text.first, Hangul.isJamo(jamo) else {
                return strokes.isEmpty ? .passThrough : finish(appending: text)
            }
            return type(jamo)
        case .delete:
            guard !strokes.isEmpty else { return .passThrough }
            strokes.removeLast()
            return mark()
        case .space:
            return strokes.isEmpty ? .passThrough : finish(appending: " ")
        case .return:
            return strokes.isEmpty ? .passThrough : finish(appending: "\n")
        case .modifier:
            return .none
        }
    }

    public mutating func select(_ candidate: Candidate) -> InputEdit { .none }

    public mutating func commit() -> InputEdit {
        strokes.isEmpty ? .none : finish(appending: "")
    }

    public mutating func reset() {
        strokes = []
        state = CompositionState()
    }

    public mutating func receive(_ candidates: [Candidate], for request: ConversionRequest) -> Bool { false }

    private mutating func type(_ jamo: Character) -> InputEdit {
        let parts = Hangul.Parts(strokes)
        let vowel = Hangul.isVowel(jamo)
        if strokes.isEmpty { return start(jamo) }
        switch (parts.initial != nil, parts.vowels.count, parts.finals.count) {
        case (_, 0, _):
            // A lone consonant takes a vowel; a second consonant starts over.
            return vowel ? append(jamo) : split(keeping: strokes.count, then: [jamo])
        case (_, 1, 0) where vowel && Hangul.compoundVowel(parts.vowels[0], jamo) != nil:
            return append(jamo)
        case (true, _, 0) where !vowel && Hangul.finalIndex(jamo) != nil:
            return append(jamo)
        case (true, _, 1) where !vowel && Hangul.compoundFinal(parts.finals[0], jamo) != nil:
            return append(jamo)
        case (true, _, let finals) where finals > 0 && vowel:
            // The last final moves to the next syllable: 앉 + ㅏ → 안자.
            return split(keeping: strokes.count - 1, then: [strokes[strokes.count - 1], jamo])
        default:
            return split(keeping: strokes.count, then: [jamo])
        }
    }

    private mutating func start(_ jamo: Character) -> InputEdit {
        strokes = [jamo]
        return mark()
    }

    private mutating func append(_ jamo: Character) -> InputEdit {
        strokes.append(jamo)
        return mark()
    }

    /// Inserts the first `count` strokes as one syllable and marks `next`.
    private mutating func split(keeping count: Int, then next: [Character]) -> InputEdit {
        let done = Hangul.compose(Array(strokes.prefix(count)))
        strokes = next
        let marked = Hangul.compose(strokes)
        state = CompositionState(markedText: marked)
        return .insert(done, mark: marked)
    }

    private mutating func finish(appending text: String) -> InputEdit {
        let done = Hangul.compose(strokes)
        reset()
        return .insert(done + text, mark: "")
    }

    private mutating func mark() -> InputEdit {
        state = CompositionState(markedText: Hangul.compose(strokes))
        return .mark(state.markedText)
    }
}

/// Unicode Hangul arithmetic over compatibility jamo (the characters on the keys).
enum Hangul {
    static let initials = Array("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
    static let vowels = Array("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
    /// Index 0 is "no final".
    static let finals: [Character?] = [nil] + Array("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ").map { $0 }

    private static let vowelPairs: [String: Character] = [
        "ㅗㅏ": "ㅘ", "ㅗㅐ": "ㅙ", "ㅗㅣ": "ㅚ", "ㅜㅓ": "ㅝ", "ㅜㅔ": "ㅞ", "ㅜㅣ": "ㅟ", "ㅡㅣ": "ㅢ",
    ]
    private static let finalPairs: [String: Character] = [
        "ㄱㅅ": "ㄳ", "ㄴㅈ": "ㄵ", "ㄴㅎ": "ㄶ", "ㄹㄱ": "ㄺ", "ㄹㅁ": "ㄻ", "ㄹㅂ": "ㄼ",
        "ㄹㅅ": "ㄽ", "ㄹㅌ": "ㄾ", "ㄹㅍ": "ㄿ", "ㄹㅎ": "ㅀ", "ㅂㅅ": "ㅄ",
    ]

    static func isVowel(_ jamo: Character) -> Bool { vowels.contains(jamo) }
    static func isJamo(_ jamo: Character) -> Bool { initials.contains(jamo) || vowels.contains(jamo) }
    static func compoundVowel(_ first: Character, _ second: Character) -> Character? { vowelPairs[String([first, second])] }
    static func compoundFinal(_ first: Character, _ second: Character) -> Character? { finalPairs[String([first, second])] }
    static func finalIndex(_ jamo: Character) -> Int? { finals.firstIndex(of: jamo) }

    /// Strokes of one syllable split into initial, vowel strokes and final strokes.
    struct Parts {
        var initial: Character?
        var vowels: [Character] = []
        var finals: [Character] = []

        init(_ strokes: [Character]) {
            var rest = strokes[...]
            if let first = rest.first, !Hangul.isVowel(first) { initial = first; rest = rest.dropFirst() }
            while let next = rest.first, Hangul.isVowel(next) { vowels.append(next); rest = rest.dropFirst() }
            finals = Array(rest)
        }
    }

    /// The text for one syllable's strokes: a full syllable when it has an
    /// initial and a vowel, otherwise the lone jamo.
    static func compose(_ strokes: [Character]) -> String {
        let parts = Parts(strokes)
        let vowel = parts.vowels.count == 2 ? compoundVowel(parts.vowels[0], parts.vowels[1]) : parts.vowels.first
        guard let initial = parts.initial, let vowel,
              let i = initials.firstIndex(of: initial), let m = vowels.firstIndex(of: vowel) else {
            // A lone vowel (compound or not) or a lone consonant.
            if parts.initial == nil, let vowel { return String(vowel) }
            return String(strokes)
        }
        let final = parts.finals.count == 2 ? compoundFinal(parts.finals[0], parts.finals[1]) : parts.finals.first
        let f = final.flatMap { finalIndex($0) } ?? 0
        guard let scalar = Unicode.Scalar(0xAC00 + (i * 21 + m) * 28 + f) else { return String(strokes) }
        return String(Character(scalar))
    }
}
