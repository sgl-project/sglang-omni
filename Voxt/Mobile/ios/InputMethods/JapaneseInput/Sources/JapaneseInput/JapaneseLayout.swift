import TypingInput

public enum JapaneseLayout {
    /// The flick grid is JapaneseKeyboardView in the keyboard target.
    public static let descriptor = LayoutDescriptor(
        id: "ja-flick",
        displayName: "日本語",
        shortLabel: "日本語",
        languages: ["ja"],
        surface: .custom("japaneseFlick"),
        numberExtras: ["¥"],
        makeEngine: { JapaneseEngine() },
        makeCandidateSource: { JapaneseCandidateSource() },
        unavailableMessage: "Japanese dictionary unavailable. Kana input still works."
    )
}

/// Flick kana build the reading; Space and Return keep it as kana. Converting
/// is a candidate tap.
public struct JapaneseEngine: InputEngine {
    public private(set) var state = CompositionState()
    public private(set) var conversion: ConversionRequest?

    public init() {}

    public mutating func handle(_ key: InputKey, context: InputContext) -> InputEdit {
        let reading = state.markedText
        switch key {
        case let .text(kana):
            guard reading.count + kana.count <= JapaneseConverter.maximumReadingCount else { return .none }
            return mark(reading + kana)
        case .modifier:
            guard let last = reading.last else { return .none }
            return mark(String(reading.dropLast()) + JapaneseFlick.cycleModifier(String(last)))
        case .delete:
            return reading.isEmpty ? .passThrough : mark(String(reading.dropLast()))
        case .space, .return:
            return reading.isEmpty ? .passThrough : commit()
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
        state = CompositionState()
        conversion = nil
    }

    public mutating func receive(_ candidates: [Candidate], for request: ConversionRequest) -> Bool {
        guard request == conversion else { return false }
        state.candidates = candidates
        return true
    }

    private mutating func mark(_ reading: String) -> InputEdit {
        state = CompositionState(markedText: reading)
        conversion = reading.isEmpty ? nil : ConversionRequest(input: [reading])
        return .mark(reading)
    }
}

/// Converts on its own actor; opens the bundled dictionary on first use.
public actor JapaneseCandidateSource: CandidateSource {
    private var converter: JapaneseConverter?

    public init() {}

    public func candidates(for request: ConversionRequest) async throws -> [Candidate] {
        let converter = try self.converter ?? JapaneseConverter()
        self.converter = converter
        return try await converter.candidates(for: request.input.joined()).map { Candidate(text: $0) }
    }
}
