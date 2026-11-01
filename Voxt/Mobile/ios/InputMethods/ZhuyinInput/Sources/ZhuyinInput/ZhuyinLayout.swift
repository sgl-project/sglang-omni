import TypingInput

public enum ZhuyinLayout {
    /// The key grid is ZhuyinKeyboardView in the keyboard target.
    public static let descriptor = LayoutDescriptor(
        id: "zh-Hant-zhuyin",
        displayName: "注音",
        shortLabel: "注音",
        languages: ["zh-Hant"],
        surface: .custom("zhuyin"),
        numberExtras: ["，", "。"],
        makeEngine: { ZhuyinEngine() },
        makeCandidateSource: { ZhuyinCandidateSource() },
        unavailableMessage: "Zhuyin conversion is unavailable."
    )
}

/// Zhuyin keys build syllables; a tone key or Space (first tone) completes one.
/// Space and Return pick the best candidate once there is one.
public struct ZhuyinEngine: InputEngine {
    private var composition = ZhuyinComposition()
    public private(set) var state = CompositionState()
    public private(set) var conversion: ConversionRequest?

    public init() {}

    public mutating func handle(_ key: InputKey, context: InputContext) -> InputEdit {
        switch key {
        case let .text(text):
            return press(text)
        case .delete:
            guard state.isComposing else { return .passThrough }
            composition.deleteBackward()
            return changed()
        case .space:
            guard state.isComposing else { return .passThrough }
            if let best = state.candidates.first, composition.readingsForConversion != nil { return select(best) }
            return press(" ")
        case .return:
            guard state.isComposing else { return .passThrough }
            if let best = state.candidates.first { return select(best) }
            return commit()
        case .modifier:
            return .none
        }
    }

    public mutating func select(_ candidate: Candidate) -> InputEdit {
        guard state.isComposing else { return .none }
        reset()
        return .insert(candidate.text, mark: "")
    }

    public mutating func commit() -> InputEdit {
        guard state.isComposing else { return .none }
        let text = state.markedText
        reset()
        return .insert(text, mark: "")
    }

    public mutating func reset() {
        composition.reset()
        state = CompositionState()
        conversion = nil
    }

    public mutating func receive(_ candidates: [Candidate], for request: ConversionRequest) -> Bool {
        guard request == conversion else { return false }
        state.candidates = candidates
        return true
    }

    private mutating func press(_ key: String) -> InputEdit {
        let before = composition.display
        guard composition.press(key) else { return .none }
        // A first tone completes the syllable without changing what is shown:
        // convert, but leave the field alone.
        if composition.display == before {
            state.candidates = []
            conversion = composition.readingsForConversion.map { ConversionRequest(input: $0) }
            return .none
        }
        return changed()
    }

    private mutating func changed() -> InputEdit {
        state = CompositionState(markedText: composition.display)
        conversion = composition.readingsForConversion.map { ConversionRequest(input: $0) }
        return .mark(state.markedText)
    }
}

/// Converts on its own actor; opens the bundled data on first use.
public actor ZhuyinCandidateSource: CandidateSource {
    private var converter: ZhuyinConverter?

    public init() {}

    public func candidates(for request: ConversionRequest) async throws -> [Candidate] {
        let converter = try self.converter ?? ZhuyinConverter()
        self.converter = converter
        do {
            return try await converter.candidates(for: request.input).map { Candidate(text: $0) }
        } catch ZhuyinConverter.ConverterError.invalidReading {
            // An unlisted syllable keeps its raw reading editable, without an error.
            throw CandidateSourceError.unconvertible
        }
    }
}
