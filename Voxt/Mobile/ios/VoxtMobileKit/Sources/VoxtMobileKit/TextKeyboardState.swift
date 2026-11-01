import Foundation

public enum TextKeyboardLayoutError: Error, Equatable {
    case unknownLayout
    case cannotRemoveLastLayout
    case layoutNotEnabled
}

public enum TextKeyboardAction: Equatable, Sendable {
    case insert(String)
    case delete
    case voice
}

public struct TextKeyboardReplacement: Equatable, Sendable {
    public var deleteCount: Int
    public var text: String

    public init(deleteCount: Int, text: String) {
        self.deleteCount = deleteCount
        self.text = text
    }
}

public enum TextKeyboardSuggestions {
    /// Case-insensitive prefix matches. Shorter terms come first, then case-folded spelling, then vocabulary order.
    public static func matches(prefix: String, vocabulary: [String]) -> [String] {
        let prefix = prefix.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prefix.isEmpty else { return [] }
        let locale = Locale(identifier: "en_US_POSIX")
        return vocabulary.enumerated().filter { item in
            item.element.range(of: prefix, options: [.caseInsensitive, .anchored], locale: locale) != nil
        }.sorted { lhs, rhs in
            if lhs.element.count != rhs.element.count { return lhs.element.count < rhs.element.count }
            let left = lhs.element.folding(options: [.caseInsensitive], locale: locale)
            let right = rhs.element.folding(options: [.caseInsensitive], locale: locale)
            if left != right { return left < right }
            return lhs.offset < rhs.offset
        }.map(\.element)
    }

    /// Deletes only `currentWord` when `suggestion` still begins with that word.
    public static func replacement(currentWord: String, suggestion: String) -> TextKeyboardReplacement? {
        guard !currentWord.isEmpty, currentWord.allSatisfy({ !$0.isWhitespace }) else { return nil }
        let locale = Locale(identifier: "en_US_POSIX")
        guard suggestion.range(of: currentWord, options: [.caseInsensitive, .anchored], locale: locale) != nil,
              currentWord.count <= suggestion.count else { return nil }
        return TextKeyboardReplacement(deleteCount: currentWord.count, text: suggestion)
    }

    /// Prefix matches for the token before the caret. Empty when the next characters continue that token.
    public static func matches(before: String, after: String, vocabulary: [String]) -> [String] {
        guard let word = typedWord(before: before, after: after) else { return [] }
        return matches(prefix: word, vocabulary: vocabulary)
    }

    /// Replaces the token before the caret. Nil when the next characters continue that token.
    public static func replacement(before: String, after: String, suggestion: String) -> TextKeyboardReplacement? {
        guard let word = typedWord(before: before, after: after) else { return nil }
        return replacement(currentWord: word, suggestion: suggestion)
    }

    /// Nil when `after` continues the same token the caret is sitting in.
    private static func typedWord(before: String, after: String) -> String? {
        if let next = after.first, isWordCharacter(next) { return nil }
        return String(before.reversed().prefix(while: isWordCharacter).reversed())
    }

    private static func isWordCharacter(_ character: Character) -> Bool {
        character.isLetter || character.isNumber || character == "'" || character == "’" || character == "-"
    }
}

/// Enabled layouts, shift, and the letters or numbers plane. The host applies insert, delete, and voice.
public struct TextKeyboardState: Equatable, Sendable {
    public var enabled: [TextKeyboardLayoutID]
    public var active: TextKeyboardLayoutID
    public var shifted: Bool
    public var plane: TextKeyboardPlane

    public static let storageKey = "textkeyboard.layouts"
    /// The unversioned record of earlier builds enabled exactly these. Layouts added
    /// later must not make an old default look like a choice the user made.
    private static let legacyAllLayouts = ["en-QWERTY", "fr-AZERTY", "de-QWERTZ", "es-QWERTY", "ja-flick", "zh-Hant-zhuyin"]

    public init(
        enabled: [TextKeyboardLayoutID]? = nil,
        active: TextKeyboardLayoutID = .englishQWERTY,
        shifted: Bool = false,
        plane: TextKeyboardPlane = .letters
    ) {
        let layouts = enabled ?? Self.defaultLayouts(preferredLanguages: Locale.preferredLanguages).filter(TextKeyboardLayoutID.offered.contains)
        self.enabled = layouts.isEmpty ? [.englishQWERTY] : layouts
        self.active = self.enabled.contains(active) ? active : self.enabled[0]
        self.shifted = shifted
        self.plane = plane
    }

    public var layout: TextKeyboardLayout { TextKeyboardLayout.make(active) }

    public static func load(from defaults: UserDefaults, preferredLanguages: [String] = Locale.preferredLanguages,
                            offered: [TextKeyboardLayoutID] = TextKeyboardLayoutID.offered) -> TextKeyboardState {
        guard let data = defaults.data(forKey: storageKey),
              let record = try? JSONDecoder().decode(Record.self, from: data) else {
            return TextKeyboardState(enabled: defaultLayouts(preferredLanguages: preferredLanguages).filter(offered.contains))
        }
        let enabled = record.version == nil && record.enabled == legacyAllLayouts
            ? defaultLayouts(preferredLanguages: preferredLanguages).map(\.rawValue) : record.enabled
        let resolved = resolve(enabledRaw: enabled, activeRaw: record.active, offered: offered)
        return TextKeyboardState(enabled: resolved.enabled, active: resolved.active)
    }

    private static func defaultLayouts(preferredLanguages: [String]) -> [TextKeyboardLayoutID] {
        var result: [TextKeyboardLayoutID] = [.englishQWERTY]
        for language in preferredLanguages {
            let code = language.lowercased().replacingOccurrences(of: "_", with: "-")
            let layout: TextKeyboardLayoutID?
            switch code.split(separator: "-").first {
            case "fr": layout = .frenchAZERTY
            case "de": layout = .germanQWERTZ
            case "es": layout = .spanishQWERTY
            case "ja": layout = .japaneseFlick
            case "zh" where code.contains("hant") || code.hasSuffix("-tw") || code.hasSuffix("-hk") || code.hasSuffix("-mo"): layout = .zhuyin
            case "zh": layout = .pinyin
            case "ko": layout = .korean
            default: layout = nil
            }
            if let layout, !result.contains(layout) { result.append(layout) }
        }
        return result
    }

    public func save(to defaults: UserDefaults) {
        let record = Record(version: 1, enabled: enabled.map(\.rawValue), active: active.rawValue)
        defaults.set(try? JSONEncoder().encode(record), forKey: Self.storageKey)
    }

    /// Replaces the enabled list and active layout from storage. Shift and the current plane stay.
    public mutating func reloadEnabledLayouts(from defaults: UserDefaults) {
        let saved = Self.load(from: defaults)
        enabled = saved.enabled
        active = saved.active
    }

    public static func resolve(enabledRaw: [String], activeRaw: String,
                               offered: [TextKeyboardLayoutID] = TextKeyboardLayoutID.offered) -> (enabled: [TextKeyboardLayoutID], active: TextKeyboardLayoutID) {
        var seen = Set<TextKeyboardLayoutID>()
        let enabled = enabledRaw.compactMap(TextKeyboardLayoutID.init(rawValue:)).filter { offered.contains($0) && seen.insert($0).inserted }
        let kept = enabled.isEmpty ? [TextKeyboardLayoutID.englishQWERTY] : enabled
        let active = TextKeyboardLayoutID(rawValue: activeRaw).flatMap { kept.contains($0) ? $0 : nil } ?? kept[0]
        return (kept, active)
    }

    public mutating func enable(rawValue: String, offered: [TextKeyboardLayoutID] = TextKeyboardLayoutID.offered) throws {
        guard let id = TextKeyboardLayoutID(rawValue: rawValue), offered.contains(id) else { throw TextKeyboardLayoutError.unknownLayout }
        if !enabled.contains(id) { enabled.append(id) }
    }

    public mutating func disable(_ id: TextKeyboardLayoutID) throws {
        guard enabled.contains(id) else { throw TextKeyboardLayoutError.layoutNotEnabled }
        guard enabled.count > 1 else { throw TextKeyboardLayoutError.cannotRemoveLastLayout }
        enabled.removeAll { $0 == id }
        if active == id { active = enabled[0] }
    }

    public mutating func select(_ id: TextKeyboardLayoutID) throws {
        guard enabled.contains(id) else { throw TextKeyboardLayoutError.layoutNotEnabled }
        active = id
        shifted = false
    }

    public mutating func moveEnabled(from index: Int, to destination: Int) {
        guard enabled.indices.contains(index), index != destination else { return }
        let item = enabled.remove(at: index)
        enabled.insert(item, at: min(max(destination, 0), enabled.count))
    }

    /// Toolbar swipe. Positive moves to the next enabled layout.
    public mutating func cycle(by offset: Int) {
        guard let index = enabled.firstIndex(of: active), !enabled.isEmpty, offset != 0 else { return }
        let next = (index + offset) % enabled.count
        active = enabled[next >= 0 ? next : next + enabled.count]
        shifted = false
    }

    public func visibleKeys() -> [TextKeyboardKey] {
        layout.rows(plane).flatMap { row in
            row.map { key in
                guard plane == .letters, shifted, key.kind == .character, let text = key.text else { return key }
                var copy = key
                copy.text = Self.oneCharacterUppercase(text)
                copy.label = copy.text ?? copy.label
                return copy
            }
        }
    }

    public mutating func press(_ key: TextKeyboardKey) -> TextKeyboardAction? {
        switch key.kind {
        case .shift:
            shifted.toggle()
            return nil
        case .numbers:
            plane = .numbers
            shifted = false
            return nil
        case .letters:
            plane = .letters
            return nil
        case .character, .space, .return:
            shifted = false
            return .insert(key.text ?? "")
        case .delete:
            return .delete
        case .voice:
            return .voice
        }
    }

    public mutating func insertAlternate(_ text: String) -> TextKeyboardAction {
        let value = shifted ? Self.oneCharacterUppercase(text) : text
        shifted = false
        return .insert(value)
    }

    private static func oneCharacterUppercase(_ text: String) -> String {
        let upper = text.uppercased()
        return upper.count == 1 ? upper : text
    }

    private struct Record: Codable {
        var version: Int?
        var enabled: [String]
        var active: String
    }
}
