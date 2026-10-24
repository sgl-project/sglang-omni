import Foundation
import NaturalLanguage

/// Character-index diff between two passage versions, shared by vocabulary shaping and
/// style candidacy.
enum PassageDiff {
    static func changedSpans(original: String, edited: String) -> (original: Range<Int>, edited: Range<Int>) {
        let old = Array(original), new = Array(edited)
        var first = 0
        while first < min(old.count, new.count), old[first] == new[first] { first += 1 }
        var suffix = 0
        while suffix < min(old.count - first, new.count - first),
              old[old.count - 1 - suffix] == new[new.count - 1 - suffix] { suffix += 1 }
        return (first..<(old.count - suffix), first..<(new.count - suffix))
    }
}

public struct LearnedCorrection: Codable, Equatable, Sendable {
    public var original: String
    public var replacement: String

    public init(original: String, replacement: String) {
        self.original = original
        self.replacement = replacement
    }

    /// Compare a bounded inserted passage, not the rest of the user's document.
    public static func detect(original: String, edited: String) -> Self? {
        guard original != edited, !original.isEmpty, !edited.isEmpty,
              original.count <= 2_000, edited.count <= 2_000 else { return nil }
        let old = Array(original), new = Array(edited)
        let spans = PassageDiff.changedSpans(original: original, edited: edited)
        let first = spans.original.lowerBound
        var oldRange = spans.original
        var newRange = spans.edited
        if oldRange.isEmpty || newRange.isEmpty {
            // A letter inserted/deleted inside an existing word is a spelling edit.
            // A new sentence, word, or deletion of a complete word is not.
            let changedLetters = Array(old[oldRange]) + Array(new[newRange])
            guard !changedLetters.isEmpty, changedLetters.allSatisfy({ $0.isLetter || $0.isNumber }) else { return nil }
            let hasPreviousLetter = first > 0 && old[first - 1].isLetter && new[first - 1].isLetter
            let hasFollowingLetter = first < old.count && first < new.count && old[first].isLetter && new[first].isLetter
            guard hasPreviousLetter || hasFollowingLetter else { return nil }
            if oldRange.isEmpty {
                oldRange = hasPreviousLetter ? (first - 1)..<first : first..<(first + 1)
            }
            if newRange.isEmpty {
                newRange = hasPreviousLetter ? (first - 1)..<first : first..<(first + 1)
            }
        }
        guard let before = wordSpan(in: original, overlapping: oldRange),
              let after = wordSpan(in: edited, overlapping: newRange),
              before != after, before.contains(where: \.isLetter), after.contains(where: \.isLetter),
              before.count <= 100, after.count <= 100,
              !before.contains("\n"), !after.contains("\n") else { return nil }
        return Self(original: before, replacement: after)
    }

    private static func wordSpan(in text: String, overlapping changed: Range<Int>) -> String? {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        let words = tokenizer.tokens(for: text.startIndex..<text.endIndex).filter { range in
            let lower = text.distance(from: text.startIndex, to: range.lowerBound)
            let upper = text.distance(from: text.startIndex, to: range.upperBound)
            return lower < changed.upperBound && upper > changed.lowerBound
        }
        guard !words.isEmpty, words.count <= 4, let first = words.first, let last = words.last else { return nil }
        return String(text[first.lowerBound..<last.upperBound])
    }
}

/// Only the inserted passage crosses to the Mac; outer document context stays in the extension.
public struct CorrectionLearningRequest: Codable, Equatable, Sendable {
    public var original: String
    public var edited: String

    public init(original: String, edited: String) { self.original = original; self.edited = edited }

    public var isValid: Bool {
        !original.isEmpty && !edited.isEmpty && original != edited && original.count <= 2_000 && edited.count <= 2_000
    }

    /// A model may select a complete term, but cannot invent it or omit other edits.
    public func validated(_ correction: LearnedCorrection?) -> LearnedCorrection? {
        guard isValid, !hasUnsupportedWordChanges, let correction, correction.original != correction.replacement else { return nil }
        for term in [correction.original, correction.replacement] {
            guard !term.isEmpty, term.count <= 100, term.contains(where: \.isLetter),
                  !term.contains("\n"), term.split(whereSeparator: \.isWhitespace).count <= 4,
                  term == term.trimmingCharacters(in: .whitespacesAndNewlines) else { return nil }
            // NLTokenizer splits 星浪/星朗 into single characters. Never learn its
            // single-Han diff; the model must supply the complete name.
            if term.unicodeScalars.allSatisfy({ (0x3400...0x9FFF).contains($0.value) }), term.count < 2 { return nil }
        }
        var search = original.startIndex..<original.endIndex
        while let oldRange = original.range(of: correction.original, range: search) {
            let prefix = String(original[..<oldRange.lowerBound])
            let suffix = String(original[oldRange.upperBound...])
            if prefix + correction.replacement + suffix == edited,
               Self.latinBoundary(prefix.last, correction.original.first),
               Self.latinBoundary(correction.original.last, suffix.first),
               Self.latinBoundary(prefix.last, correction.replacement.first),
               Self.latinBoundary(correction.replacement.last, suffix.first) { return correction }
            search = original.index(after: oldRange.lowerBound)..<original.endIndex
        }
        return nil
    }

    var hasUnsupportedWordChanges: Bool {
        let old = original.split(whereSeparator: \.isWhitespace)
        let new = edited.split(whereSeparator: \.isWhitespace)
        // A spelling correction followed by newly typed words must not become one term.
        guard old.count == new.count else { return true }
        let changed = old.indices.filter { old[$0] != new[$0] }
        guard let first = changed.first, let last = changed.last else { return false }
        // One full name may span adjacent words. Unchanged words between two edits
        // indicate distinct corrections, not one dictionary entry.
        return last - first + 1 != changed.count
    }

    /// The word-only shaping gate the original `observe` applied: true when this edit
    /// belongs on the vocabulary path (`/v1/corrections`). Style candidacy is checked
    /// first by the caller — `PersonalStyleExample.candidate` — because `detect` also
    /// returns pairs for phrasing edits like Thanks→Cheers.
    public var isVocabularyCandidate: Bool {
        !hasUnsupportedWordChanges && LearnedCorrection.detect(original: original, edited: edited) != nil
    }

    private static func latinBoundary(_ lhs: Character?, _ rhs: Character?) -> Bool {
        func isLatinWord(_ c: Character?) -> Bool {
            guard let c else { return false }
            return c.unicodeScalars.allSatisfy { $0.value < 0x3000 } && (c.isLetter || c.isNumber)
        }
        return !(isLatinWord(lhs) && isLatinWord(rhs))
    }
}

/// The full observed passage stays in memory and expires; only a learned word pair is stored.
public struct CorrectionLearningSession {
    private let documentID: UUID
    private let prefix: String
    private let suffix: String
    private let expires: Date
    private var baseline: String
    private var pending: String?
    private var pendingSince: Date?
    private var valid: Bool

    public init(documentID: UUID, before: String, inserted: String, after: String, now: Date = Date()) {
        self.documentID = documentID
        prefix = before
        suffix = after
        baseline = inserted
        expires = now.addingTimeInterval(120)
        valid = !inserted.isEmpty && inserted.count <= 2_000 && before.count <= 2_000 && after.count <= 2_000
    }

    /// Word-only shaping, unchanged: only vocabulary-shaped edits are returned, and an
    /// append or broad rewrite ends the session without becoming a new baseline.
    public mutating func observe(documentID: UUID, before: String, after: String, selectedText: String, now: Date = Date()) -> CorrectionLearningRequest? {
        edit(documentID: documentID, before: before, after: after, selectedText: selectedText, now: now) { $0.isVocabularyCandidate }
    }

    /// The raw stable edit with the same field, boundary, caret, and expiry protections,
    /// before word-only filtering. The caller routes it: `PersonalStyleExample.candidate`
    /// first, then `isVocabularyCandidate`. The baseline advances on every returned edit so
    /// chained edits stay comparable.
    public mutating func observeEdit(documentID: UUID, before: String, after: String, selectedText: String, now: Date = Date()) -> CorrectionLearningRequest? {
        edit(documentID: documentID, before: before, after: after, selectedText: selectedText, now: now) { _ in true }
    }

    private mutating func edit(documentID: UUID, before: String, after: String, selectedText: String, now: Date, shaping: (CorrectionLearningRequest) -> Bool) -> CorrectionLearningRequest? {
        guard valid, now < expires else { valid = false; return nil }
        guard documentID == self.documentID else { valid = false; return nil }
        guard selectedText.isEmpty else { pending = nil; pendingSince = nil; return nil }
        let complete = before + after
        guard complete.hasPrefix(prefix), complete.hasSuffix(suffix),
              complete.count >= prefix.count + suffix.count, complete.count <= 6_000 else {
            valid = false
            return nil
        }
        let edited = String(complete.dropFirst(prefix.count).dropLast(suffix.count))
        // A pause while spelling a word is not a completed correction. Require the caret
        // to leave word characters before starting the stable interval.
        if before.last.map({ $0.isLetter || $0.isNumber }) == true {
            pending = nil
            pendingSince = nil
            return nil
        }
        guard edited != baseline else { pending = nil; pendingSince = nil; return nil }
        guard pending == edited, let since = pendingSince else {
            pending = edited
            pendingSince = now
            return nil
        }
        guard now.timeIntervalSince(since) >= 1.5 else { return nil }
        let request = CorrectionLearningRequest(original: baseline, edited: edited)
        guard request.isValid, shaping(request) else {
            // Appends and broad rewrites are outside the inserted passage contract.
            // Never turn those ignored words into a new learnable baseline.
            valid = false
            pending = nil
            pendingSince = nil
            return nil
        }
        baseline = edited
        pending = nil
        pendingSince = nil
        return request
    }

    public func accepts(_ request: CorrectionLearningRequest, documentID: UUID, before: String, after: String, selectedText: String, now: Date = Date()) -> Bool {
        valid && now < expires && documentID == self.documentID && selectedText.isEmpty
            && before + after == prefix + request.edited + suffix
    }

}

public final class CorrectionLearningStore {
    public struct Record: Codable, Equatable, Identifiable, Sendable {
        public var id: UUID
        public var correction: LearnedCorrection
        public var dictionaryEntryID: UUID
        public var date: Date
    }
    public static let enabledKey = "dictionary.learnCorrections"
    public static let recordsKey = "dictionary.learnedCorrections"
    public static let revisionKey = "dictionary.learningConsentRevision"
    private let defaults: UserDefaults
    private var dictionary: PersonalDictionary { PersonalDictionary(defaults: defaults) }

    public init(defaults: UserDefaults) { self.defaults = defaults }

    public var isEnabled: Bool {
        get { defaults.bool(forKey: Self.enabledKey) }
        set {
            if newValue != isEnabled { defaults.set(UUID().uuidString, forKey: Self.revisionKey) }
            defaults.set(newValue, forKey: Self.enabledKey)
            if !newValue { CorrectionLearningChannel(defaults: defaults).clear() }
        }
    }

    public var consentRevision: String { defaults.string(forKey: Self.revisionKey) ?? "" }

    public var records: [Record] {
        guard let data = defaults.data(forKey: Self.recordsKey) else { return [] }
        return (try? JSONDecoder().decode([Record].self, from: data)) ?? []
    }

    @discardableResult
    public func learn(_ correction: LearnedCorrection, now: Date = Date()) throws -> Record? {
        guard isEnabled, !dictionary.vocabulary.contains(where: { $0.caseInsensitiveCompare(correction.replacement) == .orderedSame }) else { return nil }
        let entry = try dictionary.add(correction.replacement)
        let record = Record(id: UUID(), correction: correction, dictionaryEntryID: entry.id, date: now)
        let existingIDs = Set(dictionary.entries.map(\.id))
        save([record] + records.filter { existingIDs.contains($0.dictionaryEntryID) })
        return record
    }

    public func undo(_ id: UUID) {
        guard let record = records.first(where: { $0.id == id }) else { return }
        if let entry = dictionary.entries.first(where: { $0.id == record.dictionaryEntryID }), entry.term == record.correction.replacement {
            dictionary.delete(entry.id)
        }
        save(records.filter { $0.id != id })
    }

    private func save(_ records: [Record]) {
        defaults.set(try? JSONEncoder().encode(records), forKey: Self.recordsKey)
    }
}

/// Expected document text after edits made by this keyboard. A matching field ID and
/// repeated prefix/suffix alone cannot identify a line in a truncated proxy window.
public struct CorrectionEditTracker {
    private var text: String?
    public init(text: String) { self.text = text }

    public func matches(before: String, selected: String, after: String) -> Bool {
        text == before + selected + after
    }

    @discardableResult
    public mutating func replace(before: String, selected: String, after: String, with replacement: String) -> Bool {
        guard matches(before: before, selected: selected, after: after) else { text = nil; return false }
        text = before + replacement + after
        return true
    }

    @discardableResult
    public mutating func delete(before: String, selected: String, after: String) -> Bool {
        guard matches(before: before, selected: selected, after: after) else { text = nil; return false }
        text = (selected.isEmpty ? String(before.dropLast()) : before) + after
        return true
    }
}
