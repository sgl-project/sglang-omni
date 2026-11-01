// LocalAPICorrection.swift
// Extracts one full corrected word from a passage the user edited after a dictation. The selected text model
// proposes the pair; this file only accepts a pair that is actually in both passages.

import Foundation

nonisolated struct LocalAPICorrectionRequest: Equatable, Sendable {
    var original: String
    var edited: String
}

nonisolated struct LocalAPICorrectionPair: Equatable, Sendable, Codable {
    var original: String
    var replacement: String
}

nonisolated enum LocalAPICorrectionFailure: Error {
    case invalidJSON
    case mismatch
}

nonisolated enum LocalAPICorrectionPrompt {
    /// Instruction for the selected text model. The passages are the only evidence.
    /// It has no few-shot examples, so an answer cannot be copied from one.
    static func instruction(original: String, edited: String) -> String {
        """
        Extract only one full corrected word or name from the edit below.
        Return the complete word, never a fragment of a larger word. A Chinese name is the whole name, not one character from it. An alphanumeric name such as Nova2 is the whole name, not the digit 2.
        Ignore appended text, a deletion, a standalone number, punctuation, and a semantic rewrite.
        If the edit is not one full-word correction, return only {"correction":null}.
        Otherwise return only {"correction":{"original":"<full original word>","replacement":"<full corrected word>"}}.
        Original passage:
        \(original)
        Edited passage:
        \(edited)
        """
    }
}

nonisolated enum LocalAPICorrection {
    /// Appends, complete-word deletions, punctuation, standalone numbers, and multi-word
    /// rewrites are not corrections. A letter or digit inserted or deleted inside a word,
    /// including Qwen2 to Qwen3, still goes to the model.
    static func isIgnorable(original: String, edited: String) -> Bool {
        let old = Array(original)
        let new = Array(edited)
        var prefix = 0
        while prefix < min(old.count, new.count), old[prefix] == new[prefix] { prefix += 1 }
        var suffix = 0
        while suffix < min(old.count - prefix, new.count - prefix),
              old[old.count - 1 - suffix] == new[new.count - 1 - suffix] { suffix += 1 }
        let removed = Array(old[prefix..<(old.count - suffix)])
        let added = Array(new[prefix..<(new.count - suffix)])
        if removed.isEmpty || added.isEmpty {
            let changed = removed + added
            let spellingEdit = !changed.isEmpty && changed.allSatisfy { $0.isLetter || $0.isNumber }
            let hasPreviousLetter = prefix > 0 && old[prefix - 1].isLetter && new[prefix - 1].isLetter
            let hasFollowingLetter = prefix < old.count && prefix < new.count
                && old[prefix].isLetter && new[prefix].isLetter
            return !(spellingEdit && (hasPreviousLetter || hasFollowingLetter))
        }
        if removed.contains(where: \.isWhitespace) || added.contains(where: \.isWhitespace) { return true }
        if removed.contains(where: \.isLetter) || added.contains(where: \.isLetter) { return false }
        let numeric = removed.contains(where: \.isNumber) || added.contains(where: \.isNumber)
        if numeric && (touchesLetter(old, start: prefix, end: old.count - suffix)
            || touchesLetter(new, start: prefix, end: new.count - suffix)) {
            return false
        }
        return true
    }

    /// The whole trimmed model string must be `{"correction":null}` or one pair object.
    /// A pair that is not the full changed word throws instead of being guessed.
    static func interpret(modelOutput: String, original: String, edited: String) throws -> LocalAPICorrectionPair? {
        let trimmed = modelOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys) == ["correction"] else {
            throw LocalAPICorrectionFailure.invalidJSON
        }
        if object["correction"] is NSNull { return nil }
        guard let fields = object["correction"] as? [String: Any],
              Set(fields.keys) == ["original", "replacement"],
              let pairOriginal = fields["original"] as? String,
              let pairReplacement = fields["replacement"] as? String else {
            throw LocalAPICorrectionFailure.invalidJSON
        }
        let pair = LocalAPICorrectionPair(original: pairOriginal, replacement: pairReplacement)
        guard matchesEdit(pair, original: original, edited: edited) else {
            throw LocalAPICorrectionFailure.mismatch
        }
        return pair
    }

    /// `left + original + right` and `left + replacement + right` must rebuild the two
    /// passages, and that span must cover the character diff. A Latin fragment and a
    /// single Han character taken from a longer name do not.
    private static func matchesEdit(_ pair: LocalAPICorrectionPair, original: String, edited: String) -> Bool {
        let old = Array(original)
        let new = Array(edited)
        let removed = Array(pair.original)
        let added = Array(pair.replacement)
        guard !removed.isEmpty, !added.isEmpty,
              !removed.contains(where: \.isWhitespace), !added.contains(where: \.isWhitespace) else { return false }

        var prefix = 0
        while prefix < min(old.count, new.count), old[prefix] == new[prefix] { prefix += 1 }
        var suffix = 0
        while suffix < min(old.count - prefix, new.count - prefix),
              old[old.count - 1 - suffix] == new[new.count - 1 - suffix] { suffix += 1 }
        let oldEnd = old.count - suffix
        let newEnd = new.count - suffix

        guard removed.count <= old.count else { return false }
        for start in 0...(old.count - removed.count) where Array(old[start..<(start + removed.count)]) == removed {
            let left = Array(old[..<start])
            let right = Array(old[(start + removed.count)...])
            guard left + added + right == new else { continue }
            let addedEnd = start + added.count
            guard start <= prefix, start + removed.count >= oldEnd, addedEnd >= newEnd else { continue }
            let before = start > 0 ? old[start - 1] : nil
            let after = start + removed.count < old.count ? old[start + removed.count] : nil
            guard !joinsLatinWord(before, removed.first), !joinsLatinWord(removed.last, after),
                  !joinsLatinWord(before, added.first), !joinsLatinWord(added.last, after),
                  !isSingleHanFragment(removed, before: before, after: after),
                  !isSingleHanFragment(added, before: before, after: after) else { continue }
            return true
        }
        return false
    }

    private static func joinsLatinWord(_ outside: Character?, _ inside: Character?) -> Bool {
        guard let outside, let inside else { return false }
        return continuesLatinWord(outside) && continuesLatinWord(inside)
    }

    private static func touchesLetter(_ characters: [Character], start: Int, end: Int) -> Bool {
        if start > 0, characters[start - 1].isLetter { return true }
        if end < characters.count, characters[end].isLetter { return true }
        return false
    }

    private static func continuesLatinWord(_ character: Character) -> Bool {
        if character == "'" || character == "’" || character == "-" { return true }
        return (character.isLetter || character.isNumber) && !isHan(character)
    }

    /// One Han character beside another Han character is a fragment, not the name.
    private static func isSingleHanFragment(_ term: [Character], before: Character?, after: Character?) -> Bool {
        guard term.count == 1, let only = term.first, isHan(only) else { return false }
        return (before.map(isHan) ?? false) || (after.map(isHan) ?? false)
    }

    private static func isHan(_ character: Character) -> Bool {
        !character.unicodeScalars.isEmpty && character.unicodeScalars.allSatisfy { scalar in
            switch scalar.value {
            case 0x3400...0x9FFF, 0xF900...0xFAFF: true
            default: false
            }
        }
    }
}

nonisolated struct LocalAPICorrectionResponse: Encodable {
    var correction: LocalAPICorrectionPair?

    private enum CodingKeys: String, CodingKey { case correction }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        if let correction {
            try container.encode(correction, forKey: .correction)
        } else {
            try container.encodeNil(forKey: .correction)
        }
    }
}
