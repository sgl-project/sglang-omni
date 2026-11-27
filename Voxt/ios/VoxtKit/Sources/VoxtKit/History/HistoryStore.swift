import Foundation

/// A finished session, as the History tab lists it.
public struct HistoryEntry: Codable, Equatable, Identifiable, Sendable {
    public var id: UUID
    public var date: Date
    public var feature: Feature
    /// What was recognised (Transcribe's and Translate's speech, Rewrite's
    /// instruction); "" for typed text.
    public var speech: String
    /// The text a Translate or Rewrite worked on, when it was not spoken.
    public var source: String
    public var result: String
    public var title: String
    /// Why the result is the speech as recognised, when it is.
    public var warning: String
    public var seconds: Double

    public init(id: UUID = UUID(), date: Date = Date(), feature: Feature, speech: String, source: String = "",
                result: String, title: String = "", warning: String = "", seconds: Double = 0) {
        self.id = id
        self.date = date
        self.feature = feature
        self.speech = speech
        self.source = source
        self.result = result
        self.title = title
        self.warning = warning
        self.seconds = seconds
    }
}

/// The history, newest first, in one JSON file; at most limit entries.
public final class HistoryStore: @unchecked Sendable {
    private let url: URL
    private let limit: Int
    private let lock = NSLock()

    public init(directory: URL, limit: Int = 500) {
        url = directory.appendingPathComponent("history.json")
        self.limit = limit
    }

    public func entries() -> [HistoryEntry] {
        lock.lock()
        defer { lock.unlock() }
        return read()
    }

    @discardableResult
    public func add(_ entry: HistoryEntry) throws -> [HistoryEntry] {
        try update { list in
            list.insert(entry, at: 0)
            if list.count > limit {
                list.removeLast(list.count - limit)
            }
        }
    }

    @discardableResult
    public func delete(_ ids: Set<UUID>) throws -> [HistoryEntry] {
        try update { $0.removeAll { ids.contains($0.id) } }
    }

    @discardableResult
    public func clear() throws -> [HistoryEntry] {
        try update { $0.removeAll() }
    }

    private func update(_ change: (inout [HistoryEntry]) -> Void) throws -> [HistoryEntry] {
        lock.lock()
        defer { lock.unlock() }
        var list = read()
        change(&list)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(list).write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        return list
    }

    private func read() -> [HistoryEntry] {
        guard let data = try? Data(contentsOf: url) else {
            return []
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode([HistoryEntry].self, from: data)) ?? []
    }
}
