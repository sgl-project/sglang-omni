import Foundation

/// A wording example the user reviewed and explicitly kept.
/// Word spellings never come here — they belong to the vocabulary path. The stored pair is
/// the source of truth; no model-written style label is trusted or sent.
public struct PersonalStyleExample: Codable, Equatable, Sendable {
    public static let maxLength = 120

    public var before: String
    public var after: String

    public init(before: String, after: String) {
        self.before = before
        self.after = after
    }

    /// Bounds the Mac re-enforces: non-blank, different, 1…120 characters each, and the
    /// same digit sequences. These checks do not establish semantic equivalence.
    public var isValid: Bool {
        !before.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !after.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && before != after
            && before.count <= Self.maxLength && after.count <= Self.maxLength
            && Self.digitSequences(before) == Self.digitSequences(after)
    }

    /// Local eligibility for a review candidate. Longer passages return nil for now (no
    /// trimming yet). Suffix appends, whole deletions, punctuation-only, and digit-changing
    /// edits are rejected. The changed span must contain whitespace: single-word swaps are
    /// reserved for the vocabulary path, so `Thanks,`→`Cheers —` qualifies here while
    /// `Thanks,`→`Cheers,` and `gray`→`grey` stay with `/v1/corrections`.
    public static func candidate(original: String, edited: String) -> PersonalStyleExample? {
        let example = PersonalStyleExample(before: original, after: edited)
        guard example.isValid else { return nil }
        let spans = PassageDiff.changedSpans(original: original, edited: edited)
        guard !edited.hasPrefix(original) else { return nil }
        let old = Array(original), new = Array(edited)
        guard old.filter({ $0.isLetter || $0.isNumber }) != new.filter({ $0.isLetter || $0.isNumber }) else { return nil }
        let changed = Array(old[spans.original]) + Array(new[spans.edited])
        guard changed.contains(where: { $0.isLetter }) else { return nil }
        guard changed.contains(where: { $0.isWhitespace }) else { return nil }
        return example
    }

    /// Runs of numeric characters, in order. `$50`→`$60` and `$50`→`fifty` both fail:
    /// a disappearing digit is treated as a fact change even when a reader disagrees.
    static func digitSequences(_ text: String) -> [Substring] {
        text.split(whereSeparator: { !$0.isNumber })
    }
}

/// The kept examples, stored in the App Group on the phone only — never in the personal
/// dictionary, never in history sync. Pending candidates are not persisted here; the
/// keyboard holds one in memory and only `keep` writes.
public final class PersonalStyleStore {
    public struct Record: Codable, Equatable, Identifiable, Sendable {
        public var id: UUID
        public var example: PersonalStyleExample
        public var date: Date
    }

    public enum KeepError: Error, Equatable {
        case disabled, invalid, full, duplicate, staleConsent
    }

    public static let enabledKey = "styleExamples.enabled"
    private static let revisionKey = "styleExamples.consentRevision"
    private static let recordsKey = "styleExamples.records"
    public static let maxRecords = 8

    private let defaults: UserDefaults

    public init(defaults: UserDefaults) { self.defaults = defaults }

    public var isEnabled: Bool {
        get { defaults.bool(forKey: Self.enabledKey) }
        set {
            if newValue != isEnabled { defaults.set(UUID().uuidString, forKey: Self.revisionKey) }
            defaults.set(newValue, forKey: Self.enabledKey)
        }
    }

    /// Changes when the switch changes or records are removed or cleared. Pending
    /// candidates captured before those actions must be discarded.
    public var consentRevision: String { defaults.string(forKey: Self.revisionKey) ?? "" }

    public var records: [Record] {
        guard let data = defaults.data(forKey: Self.recordsKey) else { return [] }
        return (try? JSONDecoder().decode([Record].self, from: data)) ?? []
    }

    /// Keep is the only way in. The 9th fails loudly; nothing is silently evicted.
    @discardableResult
    public func keep(_ example: PersonalStyleExample, expectedRevision: String? = nil, now: Date = Date()) throws -> Record {
        guard isEnabled else { throw KeepError.disabled }
        if let expectedRevision, expectedRevision != consentRevision { throw KeepError.staleConsent }
        guard example.isValid else { throw KeepError.invalid }
        var all = records
        guard !all.contains(where: { $0.example == example }) else { throw KeepError.duplicate }
        guard all.count < Self.maxRecords else { throw KeepError.full }
        let record = Record(id: UUID(), example: example, date: now)
        all.append(record)
        save(all)
        return record
    }

    /// Removes one record only — never dictionary entries, correction records, or history.
    public func remove(_ id: UUID) {
        defaults.set(UUID().uuidString, forKey: Self.revisionKey)
        save(records.filter { $0.id != id })
    }

    public func clear() {
        defaults.set(UUID().uuidString, forKey: Self.revisionKey)
        save([])
    }

    /// What the next compose may send: only while enabled, only automatic style, never
    /// translate. Off preserves the records but sends none.
    public func examples(for mode: CompositionOptions.Mode, style: CompositionOptions.Style) -> [PersonalStyleExample] {
        guard isEnabled, style == .automatic, mode != .translate else { return [] }
        return records.map(\.example)
    }

    private func save(_ records: [Record]) {
        defaults.set(try? JSONEncoder().encode(records), forKey: Self.recordsKey)
    }
}
