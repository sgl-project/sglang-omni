// HistorySync.swift
// Exchanges text history with Voxt Mobile. Revisions and deletion tombstones live in the Mac database.

import Foundation

struct HistorySyncStoreMismatch: Error {}

enum HistorySyncParsed<Value> {
    case success(Value)
    case failure(String)
}

nonisolated struct HistorySyncItem: Equatable, Sendable {
    var id: UUID
    var text: String
    var createdAt: TimeInterval
    var mode: String?
    var elapsedSeconds: TimeInterval?
}

nonisolated struct HistorySyncRequest: Equatable, Sendable {
    var storeID: UUID?
    var cursor: Int
    var limit: Int
    var items: [HistorySyncItem]
    var deletedIDs: [UUID]

    static func parse(_ data: Data) -> HistorySyncParsed<HistorySyncRequest> {
        let object: Any
        do {
            guard !data.isEmpty else {
                return .failure("The history sync request is not valid JSON.")
            }
            object = try JSONSerialization.jsonObject(with: data)
        } catch {
            return .failure("The history sync request is not valid JSON.")
        }
        guard let fields = object as? [String: Any] else {
            return .failure("The history sync request is not valid JSON.")
        }
        let allowed = Set(["storeID", "cursor", "limit", "items", "deletedIDs"])
        let required = ["cursor", "limit", "items", "deletedIDs"]
        guard Set(fields.keys).isSubset(of: allowed), required.allSatisfy(fields.keys.contains) else {
            return .failure("The history sync request has invalid fields.")
        }
        guard let cursor = jsonInteger(fields["cursor"]) else {
            return .failure("History sync cursor must be a non-negative integer.")
        }
        guard let limit = jsonInteger(fields["limit"]), (1...200).contains(limit) else {
            return .failure("History sync limit must be from 1 to 200.")
        }
        guard let rawItems = fields["items"] as? [Any], let rawDeleted = fields["deletedIDs"] as? [Any] else {
            return .failure("The history sync request has invalid fields.")
        }
        if rawItems.count > 200 {
            return .failure("History sync accepts at most 200 items.")
        }
        if rawDeleted.count > 200 {
            return .failure("History sync accepts at most 200 deletions.")
        }
        var items: [HistorySyncItem] = []
        items.reserveCapacity(rawItems.count)
        for raw in rawItems {
            switch parseItem(raw) {
            case let .success(item):
                items.append(item)
            case let .failure(message):
                return .failure(message)
            }
        }
        var deletedIDs: [UUID] = []
        deletedIDs.reserveCapacity(rawDeleted.count)
        for raw in rawDeleted {
            guard let id = jsonUUID(raw) else {
                return .failure("History sync ids must be UUIDs.")
            }
            deletedIDs.append(id)
        }
        let itemIDs = items.map(\.id)
        guard Set(itemIDs).count == itemIDs.count,
              Set(deletedIDs).count == deletedIDs.count,
              Set(itemIDs).isDisjoint(with: deletedIDs) else {
            return .failure("History sync ids must be unique.")
        }
        switch parseStoreID(fields["storeID"]) {
        case let .success(storeID):
            if storeID == nil, cursor > 0 {
                return .failure("History sync must start at cursor 0 when storeID is null.")
            }
            return .success(HistorySyncRequest(
                storeID: storeID,
                cursor: cursor,
                limit: limit,
                items: items,
                deletedIDs: deletedIDs
            ))
        case let .failure(message):
            return .failure(message)
        }
    }

    private static func parseStoreID(_ value: Any?) -> HistorySyncParsed<UUID?> {
        guard let value, !(value is NSNull) else { return .success(nil) }
        guard let id = jsonUUID(value) else {
            return .failure("History sync storeID must be a UUID or null.")
        }
        return .success(id)
    }

    private static func parseItem(_ raw: Any) -> HistorySyncParsed<HistorySyncItem> {
        guard let fields = raw as? [String: Any] else {
            return .failure("The history sync request has invalid fields.")
        }
        let allowed = Set(["id", "text", "createdAt", "mode", "elapsedSeconds"])
        let required = ["id", "text", "createdAt"]
        guard Set(fields.keys).isSubset(of: allowed), required.allSatisfy(fields.keys.contains) else {
            return .failure("The history sync request has invalid fields.")
        }
        guard let text = fields["text"] as? String else {
            return .failure("The history sync request has invalid fields.")
        }
        guard text.count <= 100_000, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return .failure("History sync text must be 1 to 100000 characters.")
        }
        guard let id = jsonUUID(fields["id"]) else {
            return .failure("History sync ids must be UUIDs.")
        }
        guard let createdAt = jsonFinite(fields["createdAt"]) else {
            return .failure("History sync times must be finite.")
        }
        let mode: String?
        if let value = fields["mode"], !(value is NSNull) {
            guard let parsed = value as? String, Self.modes.contains(parsed) else {
                return .failure("History sync mode is invalid.")
            }
            mode = parsed
        } else {
            mode = nil
        }
        let elapsedSeconds: TimeInterval?
        if let value = fields["elapsedSeconds"], !(value is NSNull) {
            guard let parsed = jsonFinite(value), parsed >= 0 else {
                return .failure("History sync elapsed time must be a non-negative finite number.")
            }
            elapsedSeconds = parsed
        } else {
            elapsedSeconds = nil
        }
        return .success(HistorySyncItem(
            id: id,
            text: text,
            createdAt: createdAt,
            mode: mode,
            elapsedSeconds: elapsedSeconds
        ))
    }

    private static let modes: Set<String> = ["dictate", "translate", "edit", "write"]

    private static func jsonUUID(_ value: Any?) -> UUID? {
        guard let string = value as? String else { return nil }
        return UUID(uuidString: string)
    }

    private static func jsonInteger(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber, !isJSONBool(number) else { return nil }
        guard let integer = Int(exactly: number.doubleValue), integer >= 0 else { return nil }
        return integer
    }

    private static func jsonFinite(_ value: Any?) -> Double? {
        guard let number = value as? NSNumber, !isJSONBool(number) else { return nil }
        let double = number.doubleValue
        guard double.isFinite else { return nil }
        return double
    }

    private static func isJSONBool(_ number: NSNumber) -> Bool {
        CFGetTypeID(number) == CFBooleanGetTypeID()
    }
}

nonisolated struct HistorySyncResponse: Encodable, Sendable {
    var storeID: UUID
    var cursor: Int
    var hasMore: Bool
    var items: [HistorySyncItem]
    var deletedIDs: [UUID]
}

nonisolated struct HistorySyncState: Equatable, Sendable {
    var storeID: UUID
    var revision: Int
}

nonisolated struct HistorySyncChange: Equatable, Sendable {
    var revision: Int
    var deleted: Bool
}

extension HistorySyncItem: Encodable {
    private enum CodingKeys: String, CodingKey {
        case id, text, createdAt, mode, elapsedSeconds
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(text, forKey: .text)
        try container.encode(createdAt, forKey: .createdAt)
        if let mode {
            try container.encode(mode, forKey: .mode)
        } else {
            try container.encodeNil(forKey: .mode)
        }
        if let elapsedSeconds {
            try container.encode(elapsedSeconds, forKey: .elapsedSeconds)
        } else {
            try container.encodeNil(forKey: .elapsedSeconds)
        }
    }
}

nonisolated struct HistorySyncService: Sendable {
    var repository: HistoryRepository
    var syncEnabled: Bool
    var historyEnabled: Bool

    func httpResponse(for request: HistorySyncRequest) -> LocalAPIHTTPResponse {
        guard syncEnabled else {
            return .error(status: 403, "History sync is turned off.")
        }
        guard historyEnabled else {
            return .error(status: 403, "History is turned off on this Mac.")
        }
        do {
            return .json(status: 200, try repository.applyHistorySync(request))
        } catch is HistorySyncStoreMismatch {
            return .error(status: 409, "This history sync store does not match this Mac.")
        } catch {
            return .error(status: 500, "History sync failed.")
        }
    }
}
