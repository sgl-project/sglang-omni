import TypingInput

public enum PinyinLayout {
    /// Without learning, fuzzy off.
    public static let descriptor = descriptor(learning: nil, fuzzy: { false })

    /// Picks are remembered in `learning`; `fuzzy` is read before every conversion,
    /// so the Settings switch applies without reopening the keyboard.
    public static func descriptor(learning: PinyinLearning?, fuzzy: @escaping @Sendable () -> Bool) -> LayoutDescriptor {
        LayoutDescriptor(
        id: "zh-Hans-pinyin",
        displayName: "简体拼音",
        shortLabel: "拼",
        languages: ["zh-Hans"],
        surface: .keys(rows: [
            keys("qwertyuiop"),
            keys("asdfghjkl"),
            [.shift] + keys("zxcvbnm") + [.delete],
            [.numbers, .space, .return],
        ]),
        numberExtras: ["，", "。"],
        makeEngine: { PinyinEngine(learning: learning, fuzzy: fuzzy) },
        makeCandidateSource: { PinyinCandidateSource(learning: learning, fuzzy: fuzzy) },
        unavailableMessage: "Pinyin conversion is unavailable."
        )
    }

    private static func keys(_ letters: String) -> [KeySpec] {
        letters.map { KeySpec.character(String($0)) }
    }
}

/// Typed letters stay marked, split into syllables with spaces ("wo jiu shi shi");
/// candidates come from `PinyinCandidateSource`. Picking a
/// word commits it and the letters it didn't use stay marked (你 from "nihao" leaves
/// "hao"), and is remembered in `learning`. Space takes the first candidate without
/// remembering it, Return enters the letters as typed, punctuation commits the first
/// candidate before itself.
public struct PinyinEngine: InputEngine {
    /// Longest pinyin a composition holds; McBopomofo's grid is fastest well below this.
    public static let maximumLetters = 64

    private var letters = ""
    public private(set) var state = CompositionState()
    public private(set) var conversion: ConversionRequest?

    private let learning: PinyinLearning?
    private let fuzzy: @Sendable () -> Bool

    public init(learning: PinyinLearning? = nil, fuzzy: @escaping @Sendable () -> Bool = { false }) {
        self.learning = learning
        self.fuzzy = fuzzy
    }

    public mutating func handle(_ key: InputKey, context: InputContext) -> InputEdit {
        let composing = !letters.isEmpty
        switch key {
        case let .text(text):
            if composing, let letter = Self.pinyinLetter(text, composing: true) {
                guard letters.count < Self.maximumLetters else { return .none }
                return update(letters + letter)
            }
            if !composing, let letter = Self.pinyinLetter(text, composing: false) {
                return update(letter)
            }
            guard composing else { return .passThrough }
            return finish(appending: text)
        case .delete:
            guard composing else { return .passThrough }
            return update(String(letters.dropLast()))
        case .space:
            guard composing else { return .passThrough }
            if let first = state.candidates.first { return take(first) }
            return finish(appending: "", keepLetters: true)
        case .return:
            guard composing else { return .passThrough }
            return finish(appending: "", keepLetters: true)
        case .modifier:
            return .none
        }
    }

    public mutating func select(_ candidate: Candidate) -> InputEdit {
        guard !letters.isEmpty else { return .none }
        if let reading = candidate.reading { learning?.record(reading: reading, text: candidate.text) }
        return take(candidate)
    }

    private mutating func take(_ candidate: Candidate) -> InputEdit {
        let used = min(candidate.span ?? letters.count, letters.count)
        var rest = String(letters.dropFirst(used))
        while rest.hasPrefix("'") { rest.removeFirst() }
        _ = update(rest)
        return .insert(candidate.text, mark: state.markedText)
    }

    public mutating func commit() -> InputEdit {
        letters.isEmpty ? .none : finish(appending: "", keepLetters: true)
    }

    public mutating func reset() {
        letters = ""
        state = CompositionState()
        conversion = nil
    }

    public mutating func receive(_ candidates: [Candidate], for request: ConversionRequest) -> Bool {
        guard request == conversion else { return false }
        state.candidates = candidates
        return true
    }

    /// a–z start or continue pinyin (capitals while composing count as small letters);
    /// an apostrophe only separates syllables inside a composition.
    private static func pinyinLetter(_ text: String, composing: Bool) -> String? {
        guard text.count == 1, let character = text.first, character.isASCII else { return nil }
        if character == "'" { return composing ? "'" : nil }
        guard character.isLetter else { return nil }
        if character.isUppercase && !composing { return nil }
        return text.lowercased()
    }

    private mutating func update(_ next: String) -> InputEdit {
        letters = next
        let marked = Self.marked(next, fuzzy: fuzzy())
        state = CompositionState(markedText: marked)
        conversion = next.isEmpty ? nil : ConversionRequest(input: [next])
        return .mark(marked)
    }

    private static let plainSegmenter = PinyinSegmenter()
    private static let fuzzySegmenter = PinyinSegmenter(fuzzy: true)

    /// The letters with a space between syllables; a typed apostrophe stays in its
    /// place, and letters that don't split (yet) follow as typed.
    static func marked(_ letters: String, fuzzy: Bool) -> String {
        let split = (fuzzy ? fuzzySegmenter : plainSegmenter).split(letters)
        var typed = Substring(letters)
        var result = ""
        for syllable in split.syllables {
            var apostrophe = false
            while typed.first == "'" { typed.removeFirst(); apostrophe = true }
            if !result.isEmpty { result += apostrophe ? "'" : " " }
            result += typed.prefix(syllable.count)
            typed = typed.dropFirst(syllable.count)
        }
        if !typed.isEmpty { result += (result.isEmpty || typed.first == "'" ? "" : " ") + String(typed) }
        return result
    }

    /// Commits the first candidate (or the letters) followed by `text`.
    private mutating func finish(appending text: String, keepLetters: Bool = false) -> InputEdit {
        var committed = letters
        if !keepLetters, let first = state.candidates.first {
            let used = min(first.span ?? letters.count, letters.count)
            committed = first.text + String(letters.dropFirst(used))
        }
        reset()
        return .insert(committed + text, mark: "")
    }
}

/// Converts on its own actor; opens and checks the bundled data on first use.
public actor PinyinCandidateSource: CandidateSource {
    private var converter: PinyinConverter?
    private let learning: PinyinLearning?
    private let fuzzy: @Sendable () -> Bool

    public init(learning: PinyinLearning? = nil, fuzzy: @escaping @Sendable () -> Bool = { false }) {
        self.learning = learning
        self.fuzzy = fuzzy
    }

    public func candidates(for request: ConversionRequest) async throws -> [Candidate] {
        try Task.checkCancellation()
        let converter = try self.converter ?? PinyinConverter(learning: learning)
        self.converter = converter
        await converter.setFuzzy(fuzzy())
        let typed = request.input.joined()
        return try await converter.candidates(typing: typed).map { candidate in
            Candidate(text: candidate.text, span: Self.typedLength(of: Array(candidate.split.prefix(candidate.syllables)), in: typed),
                      reading: candidate.reading.isEmpty ? nil : candidate.reading)
        }
    }

    /// Characters of `typed` (apostrophes included) that the first syllables cover.
    static func typedLength(of syllables: [String], in typed: String) -> Int {
        var remaining = syllables.joined().count
        var length = 0
        for character in typed {
            if remaining == 0 { break }
            length += 1
            if character != "'" { remaining -= 1 }
        }
        return length
    }
}
