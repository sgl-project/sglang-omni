// SystemSelectionTextSupport.swift
// Shared selection-text policies for hotkey preconditions (dictionary / note / translation / rewrite).

import Foundation

/// Pure policy helpers over probed selection strings.
///
/// Keep AppKit / AX / AppleScript probing in `SelectedTextProbe`; put accept/reject
/// rules here so unit tests can cover hotkey preconditions without accessibility.
enum SystemSelectionTextSupport {
    static let maxDictionaryWhitespaceSeparatedWordCount = 5
    static let maxDictionaryConsecutiveCJKCharacterCount = 10

    /// Trimmed selection that contains at least one dictionary word character
    /// (letters / digits / CJK). Punctuation-only and invisible-only payloads are
    /// rejected — they are a common browser false positive while a caret sits in
    /// an empty input.
    static func meaningfulContent(from selectedText: String?) -> String? {
        let trimmed = selectedText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmed.isEmpty else { return nil }
        let normalized = DictionaryStore.normalizeTerm(trimmed)
        guard !normalized.isEmpty else { return nil }
        return trimmed
    }

    /// Selected text usable for note capture, selected-text translation, and rewrite source.
    /// Same gate as `meaningfulContent` today (no dictionary length/word caps).
    static func contentSelection(from selectedText: String?) -> String? {
        meaningfulContent(from: selectedText)
    }

    /// Selected text usable as a dictionary hotkey term.
    /// Applies meaningful-content first, then short-term caps (≤5 words / ≤10 CJK run).
    static func dictionaryCandidateTerm(from selectedText: String?) -> String? {
        guard let content = meaningfulContent(from: selectedText) else { return nil }
        guard whitespaceSeparatedWordCount(in: content) <= maxDictionaryWhitespaceSeparatedWordCount else {
            return nil
        }
        guard maxConsecutiveCJKRunLength(in: content) <= maxDictionaryConsecutiveCJKCharacterCount else {
            return nil
        }
        return content
    }

    static func whitespaceSeparatedWordCount(in text: String) -> Int {
        text.split(whereSeparator: \.isWhitespace).count
    }

    static func maxConsecutiveCJKRunLength(in text: String) -> Int {
        var longest = 0
        var current = 0
        for scalar in text.unicodeScalars {
            if isCJKDictionaryScalar(scalar) {
                current += 1
                longest = max(longest, current)
            } else {
                current = 0
            }
        }
        return longest
    }

    private static func isCJKDictionaryScalar(_ scalar: UnicodeScalar) -> Bool {
        dictionaryIsHanLike(scalar) || dictionaryIsKana(scalar) || dictionaryIsHangul(scalar)
    }
}

/// Compatibility facade for existing dictionary hotkey call sites / tests.
enum SelectedTextDictionaryHotkeySupport {
    static let maxWhitespaceSeparatedWordCount =
        SystemSelectionTextSupport.maxDictionaryWhitespaceSeparatedWordCount
    static let maxConsecutiveCJKCharacterCount =
        SystemSelectionTextSupport.maxDictionaryConsecutiveCJKCharacterCount

    static func candidateTerm(from selectedText: String?) -> String? {
        SystemSelectionTextSupport.dictionaryCandidateTerm(from: selectedText)
    }
}
