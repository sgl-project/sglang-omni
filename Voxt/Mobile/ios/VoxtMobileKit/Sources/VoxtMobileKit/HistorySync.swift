import Foundation

public enum HistorySyncError: Error, Equatable, LocalizedError {
    case invalidRequest, invalidResponse, staleRequest, differentStore
    public var errorDescription: String? {
        switch self {
        case .invalidRequest: "The history sync request is invalid."
        case .invalidResponse: "The Mac sent invalid history data."
        case .staleRequest: "History sync settings changed."
        case .differentStore: "This Mac's history database changed. Turn sync off and pair again."
        }
    }
}

public struct HistorySyncItem: Codable, Equatable, Sendable {
    public var id: UUID
    public var text: String
    public var createdAt: Double
    public var mode: CompositionOptions.Mode?
    public var elapsedSeconds: Double?

    public init(id: UUID, text: String, createdAt: Double, mode: CompositionOptions.Mode? = nil, elapsedSeconds: Double? = nil) {
        self.id = id; self.text = text; self.createdAt = createdAt
        self.mode = mode; self.elapsedSeconds = elapsedSeconds
    }

    public init(_ record: TranscriptRecord) {
        self.init(id: record.id, text: record.text, createdAt: record.date.timeIntervalSince1970,
                  mode: record.mode, elapsedSeconds: record.elapsedSeconds)
    }

    var isValid: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && text.count <= 100_000
            && createdAt.isFinite
            && elapsedSeconds.map { $0.isFinite && $0 >= 0 } != false
    }
}

public struct HistorySyncRequest: Codable, Equatable, Sendable {
    public var storeID: UUID?
    public var cursor: Int64
    public var limit: Int
    public var items: [HistorySyncItem]
    public var deletedIDs: [UUID]
    // Local consent identity never crosses the network.
    var localRevision: String = ""
    private enum CodingKeys: String, CodingKey { case storeID, cursor, limit, items, deletedIDs }

    public init(storeID: UUID? = nil, cursor: Int64 = 0, limit: Int = 200, items: [HistorySyncItem] = [], deletedIDs: [UUID] = []) {
        self.storeID = storeID; self.cursor = cursor; self.limit = limit
        self.items = items; self.deletedIDs = deletedIDs
    }

    public func validate() throws {
        guard cursor >= 0, (1...200).contains(limit), storeID != nil || cursor == 0,
              items.count <= 200, deletedIDs.count <= 200,
              items.allSatisfy(\.isValid), Set(items.map(\.id)).count == items.count,
              Set(deletedIDs).count == deletedIDs.count,
              Set(items.map(\.id)).isDisjoint(with: deletedIDs) else { throw HistorySyncError.invalidRequest }
    }
}

public struct HistorySyncResponse: Codable, Equatable, Sendable {
    public var storeID: UUID
    public var cursor: Int64
    public var hasMore: Bool
    public var items: [HistorySyncItem]
    public var deletedIDs: [UUID]
    public init(storeID: UUID, cursor: Int64, hasMore: Bool, items: [HistorySyncItem], deletedIDs: [UUID]) {
        self.storeID = storeID; self.cursor = cursor; self.hasMore = hasMore
        self.items = items; self.deletedIDs = deletedIDs
    }

    public func validate(for request: HistorySyncRequest) throws {
        if let expected = request.storeID, expected != storeID { throw HistorySyncError.differentStore }
        let containsChanges = !items.isEmpty || !deletedIDs.isEmpty
        guard cursor >= request.cursor, !(hasMore || containsChanges) || cursor > request.cursor,
              items.count + deletedIDs.count <= request.limit, items.allSatisfy(\.isValid),
              Set(items.map(\.id)).count == items.count, Set(deletedIDs).count == deletedIDs.count,
              Set(items.map(\.id)).isDisjoint(with: deletedIDs) else { throw HistorySyncError.invalidResponse }
    }
}

/// Durable pending changes belong to the current pairing. Only the host app uses this store.
/// Queue acknowledgements remove the sent ids, never changes made while a request was running.
public final class HistorySyncStore {
    public static let enabledKey = "history.sync.enabled"
    /// The Mac rejects request bodies over 25 MiB; legal records (up to 100k chars each)
    /// can push a 200-record batch far past that. Cap the exact encoded payload lower so a
    /// batch can never 413-loop forever.
    public static let maxRequestBytes = 24 * 1024 * 1024
    private static let revisionKey = "history.sync.consentRevision"
    private static let uploadsKey = "history.sync.uploads"
    private static let deletionsKey = "history.sync.deletions"
    private static let serverKey = "history.sync.serverID"
    private static let cursorKey = "history.sync.cursor"
    private let defaults: UserDefaults
    private var history: TranscriptHistory { TranscriptHistory(defaults: defaults) }
    public var isEnabled: Bool { defaults.bool(forKey: Self.enabledKey) && history.retention != .off }
    private var revision: String { defaults.string(forKey: Self.revisionKey) ?? "" }
    /// Records skipped by the last `makeRequest` because they are contract-invalid or over
    /// the byte budget. They stay local and pending — never acked, never deleted.
    public private(set) var lastSkippedRecordCount = 0

    public init(defaults: UserDefaults) { self.defaults = defaults }

    public func setEnabled(_ enabled: Bool) {
        let enabled = enabled && history.retention != .off
        guard enabled != defaults.bool(forKey: Self.enabledKey) else { return }
        defaults.set(UUID().uuidString, forKey: Self.revisionKey)
        defaults.set(enabled, forKey: Self.enabledKey)
        Self.saveIDs([], key: Self.uploadsKey, defaults: defaults)
        // Pausing sync must preserve offline deletions until acknowledged or unpaired.
        if enabled {
            // Off means the stores can diverge; merge the current server state on reenable.
            defaults.set(0, forKey: Self.cursorKey)
            Self.saveIDs(Set(history.items.map(\.id)), key: Self.uploadsKey, defaults: defaults)
        }
    }

    public func resetPairing() {
        setEnabled(false)
        defaults.set(UUID().uuidString, forKey: Self.revisionKey)
        for key in [Self.serverKey, Self.cursorKey, Self.uploadsKey, Self.deletionsKey] { defaults.removeObject(forKey: key) }
    }

    public func makeRequest(limit: Int = 200) -> HistorySyncRequest? {
        makeRequest(limit: limit, byteBudget: Self.maxRequestBytes)
    }

    /// Byte-budgeted variant; tests inject a smaller budget. Counts the encoded base
    /// (storeID/cursor/limit/deletedIDs) plus each individually encoded item and its comma,
    /// so the growing request is never re-encoded per candidate.
    func makeRequest(limit: Int, byteBudget: Int) -> HistorySyncRequest? {
        guard isEnabled else { return nil }
        let current = history.items
        let uploads = Self.ids(Self.uploadsKey, defaults: defaults).intersection(current.map(\.id))
        Self.saveIDs(uploads, key: Self.uploadsKey, defaults: defaults)
        var request = HistorySyncRequest(
            storeID: defaults.string(forKey: Self.serverKey).flatMap(UUID.init(uuidString:)),
            cursor: Int64(defaults.integer(forKey: Self.cursorKey)), limit: limit,
            items: [],
            deletedIDs: Array(Self.ids(Self.deletionsKey, defaults: defaults).sorted { $0.uuidString < $1.uuidString }.prefix(200))
        )
        request.localRevision = revision
        let encoder = JSONEncoder()
        guard let baseCount = try? encoder.encode(request).count, baseCount <= byteBudget else {
            // Even the empty envelope exceeds the budget: sending anything would be
            // oversized. Report everything pending as skipped so the coordinator warns.
            lastSkippedRecordCount = uploads.count
            return nil
        }
        var remaining = byteBudget - baseCount
        var items: [HistorySyncItem] = []
        var skipped = 0
        for record in current where uploads.contains(record.id) && items.count < 200 {
            let item = HistorySyncItem(record)
            guard item.isValid, let encoded = try? encoder.encode(item) else {
                skipped += 1
                continue
            }
            // Inserting into the shell's empty array adds one comma per extra element.
            let cost = encoded.count + (items.isEmpty ? 0 : 1)
            guard cost <= remaining else {
                // Too large for this page — or for any page. Skip without acking; it stays
                // local and pending while smaller valid records still ship.
                skipped += 1
                continue
            }
            items.append(item)
            remaining -= cost
        }
        request.items = items
        lastSkippedRecordCount = skipped
        return request
    }

    public func apply(_ response: HistorySyncResponse, sent request: HistorySyncRequest) throws {
        guard isEnabled, request.localRevision == revision else { throw HistorySyncError.staleRequest }
        try request.validate()
        try response.validate(for: request)
        let currentServer = defaults.string(forKey: Self.serverKey).flatMap(UUID.init(uuidString:))
        guard currentServer == request.storeID,
              Int64(defaults.integer(forKey: Self.cursorKey)) == request.cursor else { throw HistorySyncError.staleRequest }
        let localDeletions = Self.ids(Self.deletionsKey, defaults: defaults)
        let deleted = localDeletions.union(response.deletedIDs)
        history.mergeSynced(response.items, deleting: deleted)
        var uploads = Self.ids(Self.uploadsKey, defaults: defaults)
        uploads.subtract(request.items.map(\.id))
        uploads.subtract(deleted)
        Self.saveIDs(uploads, key: Self.uploadsKey, defaults: defaults)
        Self.saveIDs(localDeletions.subtracting(request.deletedIDs), key: Self.deletionsKey, defaults: defaults)
        defaults.set(response.storeID.uuidString, forKey: Self.serverKey)
        defaults.set(response.cursor, forKey: Self.cursorKey)
    }

    static func queueUpsert(_ id: UUID, defaults: UserDefaults) {
        guard defaults.bool(forKey: enabledKey) else { return }
        var ids = ids(uploadsKey, defaults: defaults)
        guard !Self.ids(deletionsKey, defaults: defaults).contains(id) else { return }
        ids.insert(id)
        saveIDs(ids, key: uploadsKey, defaults: defaults)
    }

    static func queueDeletions(_ deleted: Set<UUID>, defaults: UserDefaults) {
        guard defaults.bool(forKey: enabledKey) else { return }
        saveIDs(ids(uploadsKey, defaults: defaults).subtracting(deleted), key: uploadsKey, defaults: defaults)
        saveIDs(ids(deletionsKey, defaults: defaults).union(deleted), key: deletionsKey, defaults: defaults)
    }

    static func disableForHistoryOff(defaults: UserDefaults) {
        defaults.set(false, forKey: enabledKey)
        defaults.set(UUID().uuidString, forKey: revisionKey)
        defaults.removeObject(forKey: uploadsKey)
        defaults.removeObject(forKey: deletionsKey)
    }

    private static func ids(_ key: String, defaults: UserDefaults) -> Set<UUID> {
        Set((defaults.stringArray(forKey: key) ?? []).compactMap(UUID.init(uuidString:)))
    }
    private static func saveIDs(_ ids: Set<UUID>, key: String, defaults: UserDefaults) {
        defaults.set(ids.map(\.uuidString).sorted(), forKey: key)
    }
}

@MainActor
public final class HistorySyncCoordinator {
    private let store: HistorySyncStore
    private let exchange: @MainActor (HistorySyncRequest) async throws -> HistorySyncResponse
    private var running = false
    public private(set) var lastError: String?
    public private(set) var lastSyncedAt: Date?

    public init(store: HistorySyncStore, exchange: @escaping @MainActor (HistorySyncRequest) async throws -> HistorySyncResponse) {
        self.store = store; self.exchange = exchange
    }

    public func sync() async {
        guard !running, store.isEnabled else { return }
        running = true
        defer { running = false }
        do {
            while !Task.isCancelled, let request = store.makeRequest() {
                let response = try await exchange(request)
                guard !Task.isCancelled else { return }
                try store.apply(response, sent: request)
                lastError = nil
                lastSyncedAt = Date()
                let next = store.makeRequest()
                if !response.hasMore, next?.items.isEmpty != false, next?.deletedIDs.isEmpty != false { break }
            }
            if store.lastSkippedRecordCount > 0 {
                // Honest partial sync: some records never left this phone.
                lastError = VoxtL10n.shared.string("Some records are too large to sync. They remain on this iPhone.")
            }
        } catch HistorySyncError.staleRequest {
            // Turning sync off or replacing the pairing invalidates the in-flight response.
        } catch { lastError = error.localizedDescription }
    }
}
