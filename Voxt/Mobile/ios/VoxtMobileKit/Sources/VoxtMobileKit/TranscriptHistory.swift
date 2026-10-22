import Foundation

public struct TranscriptRecord: Codable, Identifiable, Equatable, Sendable {
    public var id: UUID
    public var text: String
    public var date: Date
    public var mode: CompositionOptions.Mode?
    /// Measured time from recording start until the successful result, including retries.
    /// Older records have no measurement and are excluded from time estimates.
    public var elapsedSeconds: TimeInterval?

    public init(id: UUID = UUID(), text: String, date: Date = Date(), mode: CompositionOptions.Mode? = nil, elapsedSeconds: TimeInterval? = nil) {
        self.id = id
        self.text = text
        self.date = date
        self.mode = mode
        self.elapsedSeconds = elapsedSeconds
    }
}

public final class TranscriptHistory {
    public enum Retention: String, Codable, CaseIterable, Sendable {
        case off, sevenDays, thirtyDays, forever
        public var title: String {
            switch self {
            case .off: VoxtL10n.shared.string("Do not save")
            case .sevenDays: VoxtL10n.shared.string("7 days")
            case .thirtyDays: VoxtL10n.shared.string("30 days")
            case .forever: VoxtL10n.shared.string("Until I delete")
            }
        }
    }
    private let defaults: UserDefaults
    private let key = "history.items"

    public init(defaults: UserDefaults) {
        self.defaults = defaults
        prune()
    }

    public var retention: Retention {
        defaults.string(forKey: "history.retention").flatMap(Retention.init(rawValue:)) ?? .forever
    }

    /// Turning history off also removes existing phone history.
    public func setRetention(_ retention: Retention, now: Date = Date()) {
        if retention == .off { HistorySyncStore.disableForHistoryOff(defaults: defaults) }
        defaults.set(retention.rawValue, forKey: "history.retention")
        prune(now: now)
    }

    public func clear() {
        HistorySyncStore.queueDeletions(Set(items.map(\.id)), defaults: defaults)
        defaults.removeObject(forKey: key)
    }

    public func prune(now: Date = Date()) {
        switch retention {
        case .off: clear()
        case .forever: break
        case .sevenDays, .thirtyDays:
            let days = retention == .sevenDays ? 7 : 30
            save(items.filter { $0.date >= now.addingTimeInterval(-Double(days) * 86400) })
        }
    }

    public var items: [TranscriptRecord] {
        guard let data = defaults.data(forKey: key) else { return [] }
        return ((try? JSONDecoder().decode([TranscriptRecord].self, from: data)) ?? []).sorted { $0.date > $1.date }
    }

    public func search(_ query: String) -> [TranscriptRecord] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return items.filter { query.isEmpty || $0.text.localizedStandardContains(query) }
    }

    public func append(id: UUID = UUID(), text: String, mode: CompositionOptions.Mode? = nil, elapsedSeconds: TimeInterval? = nil, now: Date = Date()) {
        guard retention != .off, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard !items.contains(where: { $0.id == id }) else { return }
        let duration = elapsedSeconds.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil }
        save([TranscriptRecord(id: id, text: text, date: now, mode: mode, elapsedSeconds: duration)] + items)
        HistorySyncStore.queueUpsert(id, defaults: defaults)
        prune(now: now)
    }

    public func delete(ids: Set<UUID>) {
        HistorySyncStore.queueDeletions(ids.intersection(items.map(\.id)), defaults: defaults)
        save(items.filter { !ids.contains($0.id) })
    }

    func mergeSynced(_ incoming: [HistorySyncItem], deleting deleted: Set<UUID>) {
        guard retention != .off else { return }
        var byID = Dictionary(items.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        for item in incoming where !deleted.contains(item.id) {
            let existing = byID[item.id]
            byID[item.id] = TranscriptRecord(id: item.id, text: item.text, date: Date(timeIntervalSince1970: item.createdAt),
                                             mode: existing?.mode ?? item.mode, elapsedSeconds: existing?.elapsedSeconds ?? item.elapsedSeconds)
        }
        for id in deleted { byID.removeValue(forKey: id) }
        save(Array(byID.values))
        prune()
    }

    private func save(_ items: [TranscriptRecord]) {
        defaults.set(try? JSONEncoder().encode(items), forKey: key)
    }
}
