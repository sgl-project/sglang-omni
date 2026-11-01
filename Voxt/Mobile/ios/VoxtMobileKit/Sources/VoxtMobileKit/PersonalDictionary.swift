import Foundation

/// Phone vocabulary is read afresh for each recording and sent in CompositionOptions.
/// Entries take effect only when the paired Mac supports compose requests.
public final class PersonalDictionary {
    public struct Entry: Codable, Equatable, Identifiable, Sendable {
        public let id: UUID
        public var term: String
    }

    private let defaults: UserDefaults
    private let key = "dictionary.entries"

    public init(defaults: UserDefaults) { self.defaults = defaults }

    public var entries: [Entry] {
        guard let data = defaults.data(forKey: key) else { return [] }
        return (try? JSONDecoder().decode([Entry].self, from: data)) ?? []
    }

    public var vocabulary: [String] { entries.map(\.term) }

    public func search(_ query: String) -> [Entry] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return entries.filter { query.isEmpty || $0.term.localizedStandardContains(query) }
    }

    @discardableResult
    public func add(_ term: String) throws -> Entry {
        var items = entries
        let entry = Entry(id: UUID(), term: term.trimmingCharacters(in: .whitespacesAndNewlines))
        items.append(entry)
        try save(items)
        return entry
    }

    public func rename(_ id: UUID, to term: String) throws {
        var items = entries
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].term = term.trimmingCharacters(in: .whitespacesAndNewlines)
        try save(items)
    }

    public func delete(_ id: UUID) {
        let items = entries.filter { $0.id != id }
        defaults.set(try? JSONEncoder().encode(items), forKey: key)
    }

    private func save(_ items: [Entry]) throws {
        try CompositionOptions(vocabulary: items.map(\.term)).validate()
        let normalized = items.map { $0.term.folding(options: [.caseInsensitive], locale: Locale(identifier: "en_US_POSIX")) }
        guard Set(normalized).count == items.count else {
            throw CompositionOptions.ValidationError(message: "This word is already in your dictionary.")
        }
        defaults.set(try JSONEncoder().encode(items), forKey: key)
    }
}
