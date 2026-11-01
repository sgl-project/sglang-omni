import Foundation

/// A snapshot from the active text proxy. Nil context is unavailable, not an empty field.
public struct FieldContext: Equatable, Sendable {
    public let documentID: UUID
    public let before: String
    public let after: String
    public let selected: String

    public init(documentID: UUID, before: String, after: String, selected: String) {
        self.documentID = documentID
        self.before = before
        self.after = after
        self.selected = selected
    }
}

/// Owns only the current reading. The keyboard owns marked-text writes and reports
/// the proxy's actual context after each write; no assumed marked-text offset is used.
public struct MarkedComposition: Sendable {
    public struct Query: Equatable, Sendable {
        public let id: UUID
        public let reading: String
    }

    /// Enough to fill the expanded candidate grid.
    public static let maximumCandidates = 40

    public private(set) var reading = ""
    public private(set) var candidates: [String] = []
    private var context: FieldContext?
    private var markBase: FieldContext?
    private var pendingMarkedContext: FieldContext?
    private var query: Query?

    public init() {}

    public func matches(_ current: FieldContext?) -> Bool {
        guard let current, current.selected.isEmpty, let context else { return false }
        return current == context
    }

    public mutating func update(reading: String, context current: FieldContext?) -> Query? {
        guard let current, current.selected.isEmpty,
              context == nil || matches(current) else {
            reset()
            return nil
        }
        guard reading.count <= 128 else { return nil }
        if markBase == nil { markBase = current }
        let next = Query(id: UUID(), reading: reading)
        self.reading = reading
        context = current
        query = next
        candidates = reading.isEmpty ? [] : [reading]
        return next
    }

    @discardableResult
    public mutating func didMark(context current: FieldContext?, for request: Query) -> Bool {
        guard request == query else { return false }
        guard let current, current.documentID == context?.documentID, current.selected.isEmpty else {
            reset()
            return false
        }
        context = current
        if let base = markBase {
            let expected = FieldContext(documentID: base.documentID,
                before: base.before + reading, after: base.after, selected: "")
            pendingMarkedContext = current == expected ? nil : expected
        }
        return true
    }

    /// iOS 26.3: textDidChange acknowledges our marked write 20–25ms after
    /// setMarkedText returns its old context. Accept only that exact
    /// write once; caret movement, field changes and unrelated edits still fail.
    public mutating func acknowledgeMarkedContext(_ current: FieldContext?) -> Bool {
        if matches(current) { return true }
        guard let current, current == pendingMarkedContext else { return false }
        context = current
        pendingMarkedContext = nil
        return true
    }

    /// Some hosts expose the composition as ordinary text and do not retain a
    /// replaceable mark. Only return a deletable suffix for our exact saved range.
    public func visibleReadingCount(in current: FieldContext?) -> Int? {
        guard matches(current), let current, let base = markBase,
              current.after == base.after else { return nil }
        if current.before == base.before + reading { return reading.count }
        if current.before == base.before { return 0 }
        return nil
    }

    @discardableResult
    public mutating func receive(_ values: [String], for request: Query) -> Bool {
        guard request == query, !reading.isEmpty else { return false }
        var seen: Set<String> = [reading]
        candidates = [reading] + values.filter { !$0.isEmpty && $0.count <= 256 && seen.insert($0).inserted }.prefix(Self.maximumCandidates)
        return true
    }

    public mutating func commit(_ candidate: String, context current: FieldContext?) -> String? {
        guard matches(current) else { reset(); return nil }
        guard !reading.isEmpty, candidates.contains(candidate) else { return nil }
        reset()
        return candidate
    }

    /// Commits text the layout's engine made rather than a listed candidate
    /// (a finished Hangul syllable, or a syllable plus a space). Same field check as `commit`.
    public mutating func commitText(_ text: String, context current: FieldContext?) -> String? {
        guard matches(current) else { reset(); return nil }
        guard !reading.isEmpty, !text.isEmpty else { return nil }
        reset()
        return text
    }

    public mutating func reset() {
        reading = ""
        candidates = []
        context = nil
        markBase = nil
        pendingMarkedContext = nil
        query = nil
    }
}
