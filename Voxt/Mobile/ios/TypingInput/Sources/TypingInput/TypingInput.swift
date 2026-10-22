import Foundation

// The contract between the Voxt keyboard and a typing layout.
//
// The keyboard owns the text field: it writes marked text, inserts, deletes and
// verifies that each write landed before the next key is applied. A layout only
// turns keys into edits and, optionally, a reading into candidates:
//
//   key tap → host calls engine.handle(key, context:) on the main actor
//           → InputEdit (mark / insert / passThrough / none) → host writes it
//           → if engine.conversion has a new request: host waits ~40 ms, asks
//             the layout's CandidateSource (an actor, off the main actor),
//             then hands the result back with engine.receive(_:for:)
//   candidate tap → engine.select(candidate) → InputEdit → host writes it
//
// Engines are value types with synchronous methods so keys apply in order;
// only conversion is async. A layout package depends on TypingInput only,
// never on VoxtMobileKit, and the voice path never imports a layout package.

/// One key press, after the host has applied shift (`KeySpec.text(shifted:)`).
public enum InputKey: Hashable, Sendable {
    /// The key's text: "a", "ñ", "ㄅ", "ㄲ", "い".
    case text(String)
    /// The layout's own modifier key, such as Japanese 小゛゜.
    case modifier
    case space
    case `return`
    case delete
}

/// Field text around the cursor, outside the engine's own marked text. Hosts may
/// give only part of the document, or nothing.
public struct InputContext: Equatable, Sendable {
    public var before: String
    public var after: String

    public init(before: String, after: String = "") {
        self.before = before
        self.after = after
    }
}

/// What the host should do with the field after a key or a candidate tap.
public enum InputEdit: Equatable, Sendable {
    /// The engine used the key; the field does not change.
    case none
    /// Replace the current marked text with this ("" removes it).
    case mark(String)
    /// Insert `text` for good, then show `mark` as the new marked text ("" = none).
    /// Pinyin picking 你 from "nihao" is `.insert("你", mark: "hao")`.
    case insert(String, mark: String)
    /// The engine did not use the key: the host does its default
    /// (insert the key's text, delete backward, insert a space or a newline).
    case passThrough
}

public struct Candidate: Hashable, Sendable {
    public var text: String
    /// How much of the reading this candidate replaces, in the engine's own
    /// units (syllables, kana). Nil: all of it.
    public var span: Int?
    /// The dictionary reading this candidate converts (pinyin "ni-hao"), for
    /// engines that remember picks. Nil when the engine doesn't learn.
    public var reading: String?

    public init(text: String, span: Int? = nil, reading: String? = nil) {
        self.text = text
        self.span = span
        self.reading = reading
    }
}

public struct CompositionState: Equatable, Sendable {
    /// What the field shows as marked text. Empty when nothing is being composed.
    public var markedText: String
    public var candidates: [Candidate]

    public init(markedText: String = "", candidates: [Candidate] = []) {
        self.markedText = markedText
        self.candidates = candidates
    }

    public var isComposing: Bool { !markedText.isEmpty }
}

/// A conversion the engine wants. A new request replaces the old one; results
/// for an old request are ignored by `receive`.
public struct ConversionRequest: Equatable, Sendable {
    public let id: UUID
    /// Engine-defined: Zhuyin syllables, one kana string, typed pinyin letters.
    public let input: [String]

    public init(id: UUID = UUID(), input: [String]) {
        self.id = id
        self.input = input
    }
}

/// Turns keys into edits. Synchronous and free of I/O: the host calls it on the
/// main actor for every key, in order.
public protocol InputEngine: Sendable {
    var state: CompositionState { get }
    /// The conversion to run now, or nil (nothing composed, or a half-typed
    /// syllable that can't be converted yet).
    var conversion: ConversionRequest? { get }

    mutating func handle(_ key: InputKey, context: InputContext) -> InputEdit
    mutating func select(_ candidate: Candidate) -> InputEdit
    /// Finish composing and keep what was typed (switching layout, voice, a field change).
    mutating func commit() -> InputEdit
    /// Forget the composition without touching the field (the host already lost it).
    mutating func reset()
    /// Candidates for `request`. False, and no change, when `request` is not the current one.
    @discardableResult
    mutating func receive(_ candidates: [Candidate], for request: ConversionRequest) -> Bool

    /// Whether the host should turn shift on for the next key.
    func capitalizesNextKey(context: InputContext) -> Bool
}

public extension InputEngine {
    func capitalizesNextKey(context: InputContext) -> Bool { false }
}

/// Converts a reading to candidates, off the main actor. May load a dictionary
/// on first use. Throw `CandidateSourceError.unconvertible` to keep the raw
/// reading without an error message; any other error shows the layout's
/// `unavailableMessage`.
public protocol CandidateSource: Actor {
    func candidates(for request: ConversionRequest) async throws -> [Candidate]
}

public enum CandidateSourceError: Error, Equatable {
    case unconvertible
}

/// A layout that inserts what its keys say: Latin layouts, Spanish.
public struct DirectInputEngine: InputEngine {
    public enum Capitalization: Equatable, Sendable {
        case none
        /// Shift on at the start of the field and after . ! ? plus a space.
        /// `openers` don't end the start of a sentence: "¿" keeps shift on.
        case sentences(openers: Set<Character> = [])
    }

    public let capitalization: Capitalization
    public let state = CompositionState()
    public var conversion: ConversionRequest? { nil }

    public init(capitalization: Capitalization = .none) {
        self.capitalization = capitalization
    }

    public mutating func handle(_ key: InputKey, context: InputContext) -> InputEdit { .passThrough }
    public mutating func select(_ candidate: Candidate) -> InputEdit { .none }
    public mutating func commit() -> InputEdit { .none }
    public mutating func reset() {}
    public mutating func receive(_ candidates: [Candidate], for request: ConversionRequest) -> Bool { false }

    public func capitalizesNextKey(context: InputContext) -> Bool {
        guard case let .sentences(openers) = capitalization else { return false }
        var text = Substring(context.before)
        while let last = text.last, openers.contains(last) { text.removeLast() }
        let endsWithSpace = text.last == " "
        while text.last == " " { text.removeLast() }
        guard let end = text.last else { return true }
        return end.isNewline || (endsWithSpace && ".!?".contains(end))
    }
}

/// One key in a layout's letter plane.
public struct KeySpec: Hashable, Sendable, Identifiable {
    public enum Role: String, Hashable, Sendable {
        case character, shift, delete, space, `return`, numbers, modifier, voice
    }

    public var id: String
    public var role: Role
    public var label: String
    public var text: String?
    /// Text with shift on. Nil: one-letter uppercase of `text`, when there is one.
    public var shiftedText: String?
    /// Long-press menu.
    public var alternates: [String]
    /// Width relative to a letter key.
    public var width: Double

    public init(id: String, role: Role, label: String, text: String? = nil, shiftedText: String? = nil,
                alternates: [String] = [], width: Double = 1) {
        self.id = id
        self.role = role
        self.label = label
        self.text = text
        self.shiftedText = shiftedText
        self.alternates = alternates
        self.width = width
    }

    public static func character(_ text: String, shifted: String? = nil, alternates: [String] = []) -> KeySpec {
        KeySpec(id: "c-\(text)", role: .character, label: text, text: text, shiftedText: shifted, alternates: alternates)
    }

    public static let shift = KeySpec(id: "shift", role: .shift, label: "shift")
    public static let delete = KeySpec(id: "delete", role: .delete, label: "delete")
    public static let space = KeySpec(id: "space", role: .space, label: "space", text: " ", width: 2)
    public static let `return` = KeySpec(id: "return", role: .return, label: "return", text: "\n")
    public static let numbers = KeySpec(id: "numbers", role: .numbers, label: "123")

    public func text(shifted: Bool) -> String? {
        guard shifted, role == .character, let text else { return text }
        if let shiftedText { return shiftedText }
        let upper = text.uppercased()
        return upper.count == 1 ? upper : text
    }
}

/// Everything the keyboard needs to offer a layout. Registered once in the
/// keyboard target; the keyboard model never names a specific engine.
public struct LayoutDescriptor: Sendable, Identifiable {
    public enum Surface: Sendable {
        /// Rows of keys drawn by the shared key grid, with a candidate bar when
        /// the layout has a candidate source.
        case keys(rows: [[KeySpec]])
        /// A view of the layout's own, looked up by this name in the keyboard
        /// target (Japanese flick).
        case custom(String)
    }

    /// Stable and persisted in the user's enabled-layout list, such as "ko-2beolsik".
    public let id: String
    /// In the layout's own language: "한국어", "Español".
    public let displayName: String
    /// Toolbar button: "한", "ES", "拼".
    public let shortLabel: String
    /// BCP 47 prefixes that enable this layout by default: ["ko"], ["zh-Hans"].
    public let languages: [String]
    public let surface: Surface
    /// Extra keys on the shared numbers plane, such as ["¿", "¡"] or ["，", "。"].
    public let numberExtras: [String]
    public let makeEngine: @Sendable () -> any InputEngine
    public let makeCandidateSource: (@Sendable () throws -> any CandidateSource)?
    /// English source string for VoxtL10n, shown when conversion fails.
    public let unavailableMessage: String?

    public init(id: String, displayName: String, shortLabel: String, languages: [String], surface: Surface,
                numberExtras: [String] = [], makeEngine: @escaping @Sendable () -> any InputEngine,
                makeCandidateSource: (@Sendable () throws -> any CandidateSource)? = nil,
                unavailableMessage: String? = nil) {
        self.id = id
        self.displayName = displayName
        self.shortLabel = shortLabel
        self.languages = languages
        self.surface = surface
        self.numberExtras = numberExtras
        self.makeEngine = makeEngine
        self.makeCandidateSource = makeCandidateSource
        self.unavailableMessage = unavailableMessage
    }
}

/// The layouts this keyboard build offers, in display order.
public struct LayoutRegistry: Sendable {
    public enum RegistryError: Error, Equatable {
        case duplicateID(String)
    }

    public let all: [LayoutDescriptor]

    public init(_ layouts: [LayoutDescriptor]) throws {
        var seen = Set<String>()
        for layout in layouts where !seen.insert(layout.id).inserted { throw RegistryError.duplicateID(layout.id) }
        all = layouts
    }

    public func layout(id: String) -> LayoutDescriptor? { all.first { $0.id == id } }

    /// Layouts for the user's languages, in the order of those languages.
    /// "zh-Hans" matches "zh-Hans-CN" and not "zh-Hant-TW".
    public func defaultLayouts(preferredLanguages: [String]) -> [LayoutDescriptor] {
        var result: [LayoutDescriptor] = []
        for preferred in preferredLanguages {
            let code = preferred.lowercased().replacingOccurrences(of: "_", with: "-")
            for layout in all where !result.contains(where: { $0.id == layout.id }) {
                if layout.languages.contains(where: { code == $0.lowercased() || code.hasPrefix($0.lowercased() + "-") }) {
                    result.append(layout)
                }
            }
        }
        return result
    }
}
