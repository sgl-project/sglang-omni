import Foundation

/// What runs the models: the executable's MLX and GGUF engines behind one
/// Runtime, or a fake in tests. The server calls load, unload and chat one
/// at a time (never two at once); status and memory may come any time.
public protocol Engine: Sendable {
    /// Loads model, unloading another. It returns the time it took (0 and
    /// cached when model was loaded already). Errors: load_failed,
    /// bad_request (an unknown format).
    func load(_ model: ModelSpec, cacheLimitMB: Int?) async throws -> LoadOutcome
    /// Frees the model; false when none was loaded.
    func unload() async -> Bool
    /// The loaded model, if any.
    func loaded() -> ModelSpec?
    /// Runs one completion with p.model loaded, calling onDelta with each
    /// piece of text in order. It stops at a cancellation of the calling
    /// task (throwing CancellationError) within a token. With maxTokens 0
    /// it only evaluates the prompt's shared prefix into the prompt cache.
    func chat(_ p: ChatParams, onDelta: @escaping @Sendable (String) -> Void) async throws -> ChatOutcome
    /// The process's memory now.
    func memory() -> Memory
}

public struct LoadOutcome: Sendable, Equatable {
    public var cached: Bool
    public var loadMs: Double

    public init(cached: Bool, loadMs: Double) {
        self.cached = cached
        self.loadMs = loadMs
    }
}

/// A finished completion, before the server adds the load time and memory.
public struct ChatOutcome: Sendable, Equatable {
    public var text: String
    public var finish: String
    public var usage: Usage
    public var promptMs: Double
    public var firstTokenMs: Double
    public var tokensPerSecond: Double
    public var cache: CacheUse?

    public init(text: String, finish: String, usage: Usage, promptMs: Double, firstTokenMs: Double, tokensPerSecond: Double, cache: CacheUse? = nil) {
        self.text = text
        self.finish = finish
        self.usage = usage
        self.promptMs = promptMs
        self.firstTokenMs = firstTokenMs
        self.tokensPerSecond = tokensPerSecond
        self.cache = cache
    }
}

/// Collects a completion's text as the engines produce it: the pieces
/// sent so far, and a stop when the answer starts looping (the repetition
/// guard, Voxt's: small models repeat a phrase until the token budget).
public struct TextCollector: Sendable {
    public private(set) var text = ""
    public private(set) var looped = false
    private let guardian = RepetitionGuard()

    public init() {}

    /// Adds piece; false when generation should stop (a loop was cut: text
    /// keeps one copy of the repeated part).
    public mutating func add(_ piece: String) -> Bool {
        text += piece
        if let r = guardian.repeatedSuffix(in: text) {
            text = r.truncatedText
            looped = true
            return false
        }
        return true
    }
}

/// Finds a phrase repeated at the end of a text (Voxt's
/// LLMOutputRepetitionGuard): six or more copies of a unit of up to 48
/// characters covering 48 or more, or ten of a unit of up to 4 covering 24.
public struct RepetitionGuard: Sendable {
    public var maximumUnitLength = 48
    public var minimumRepetitionCount = 6
    public var minimumRunCharacterCount = 48
    public var shortUnitMinimumRepetitionCount = 10
    public var shortUnitMinimumRunCharacterCount = 24

    public struct Repetition: Equatable, Sendable {
        public let repeatedUnit: String
        public let repetitionCount: Int
        public let truncatedText: String
    }

    public init() {}

    public func repeatedSuffix(in text: String) -> Repetition? {
        let characterCount = text.count
        guard characterCount >= shortUnitMinimumRunCharacterCount else { return nil }
        let longestUnit = min(maximumUnitLength, characterCount / minimumRepetitionCount)
        guard longestUnit > 0 else { return nil }
        for unitLength in 1...longestUnit {
            guard let unitStart = text.index(text.endIndex, offsetBy: -unitLength, limitedBy: text.startIndex) else {
                continue
            }
            let unit = String(text[unitStart...])
            guard !unit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            var repetitions = 1
            var runStart = unitStart
            while let previousStart = text.index(runStart, offsetBy: -unitLength, limitedBy: text.startIndex),
                  text[previousStart..<runStart] == text[unitStart..<text.endIndex] {
                repetitions += 1
                runStart = previousStart
            }
            let run = repetitions * unitLength
            let shortRun = unitLength <= 4 && repetitions >= shortUnitMinimumRepetitionCount && run >= shortUnitMinimumRunCharacterCount
            let generalRun = repetitions >= minimumRepetitionCount && run >= minimumRunCharacterCount
            guard shortRun || generalRun else { continue }
            let keepEnd = text.index(text.endIndex, offsetBy: -unitLength * (repetitions - 1))
            return Repetition(repeatedUnit: unit, repetitionCount: repetitions, truncatedText: String(text[..<keepEnd]))
        }
        return nil
    }
}

/// Bytes of UTF-8 that may end inside a character (a GGUF model's pieces
/// split them): text() is the longest valid prefix not sent yet.
public struct UTF8Pieces: Sendable {
    private var pending: [UInt8] = []

    public init() {}

    /// Adds bytes; returns the text they complete ("" while a character
    /// is still split).
    public mutating func add(_ bytes: [UInt8]) -> String {
        pending += bytes
        var cut = pending.count
        // Back off over an incomplete trailing sequence (at most 3 bytes).
        var back = 0
        while back < min(3, pending.count) {
            let b = pending[pending.count - 1 - back]
            if b & 0xC0 == 0x80 {
                back += 1
                continue
            }
            let need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1
            if need > back + 1 { cut = pending.count - 1 - back }
            break
        }
        let out = String(decoding: pending[..<cut], as: UTF8.self)
        pending.removeFirst(cut)
        return out
    }

    /// What is left (an incomplete character becomes U+FFFD).
    public mutating func flush() -> String {
        defer { pending = [] }
        return String(decoding: pending, as: UTF8.self)
    }
}
