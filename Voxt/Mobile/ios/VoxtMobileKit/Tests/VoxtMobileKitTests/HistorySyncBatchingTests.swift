import Foundation
import Testing
@testable import VoxtMobileKit

/// Byte-budget batching for history sync: legal records (up to 100k chars each) can make a
/// 200-record batch exceed the Mac's 25 MiB body cap, which would 413 forever. The store
/// must cap the exact encoded payload under the cap, skip invalid/oversize records without
/// deleting them locally, keep them pending for future repair, and the coordinator must
/// surface an explicit partial-sync warning instead of "Up to date".
struct HistorySyncBatchingTests {
    private func isolated(_ body: (UserDefaults, TranscriptHistory, HistorySyncStore) throws -> Void) rethrows {
        let name = "history-sync-budget.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        try body(defaults, TranscriptHistory(defaults: defaults), HistorySyncStore(defaults: defaults))
    }

    @Test func defaultBudgetIsBelowTheServerCap() {
        #expect(HistorySyncStore.maxRequestBytes == 24 * 1024 * 1024)
        #expect(HistorySyncStore.maxRequestBytes < 25 * 1024 * 1024)
    }

    @Test func batchStaysWithinByteBudgetWithExactEncodedSize() throws {
        try isolated { _, history, store in
            for index in 0..<30 { history.append(text: String(repeating: "字", count: 500) + "\(index)") }
            // Escaped characters inflate JSON: quotes, backslashes and newlines must be
            // counted exactly, not as their raw character count.
            history.append(text: "quoted \"text\" with \\backslash\\ and\nnewlines\n" + String(repeating: "x", count: 300))
            store.setEnabled(true)
            let budget = 4 * 1024
            let request = try #require(store.makeRequest(limit: 200, byteBudget: budget))
            let encoded = try JSONEncoder().encode(request)
            #expect(encoded.count <= budget, "encoded payload must fit the budget, not just the estimate")
            #expect(!request.items.isEmpty)
            #expect(request.items.count < 31)
        }
    }

    @Test func laterPagesContinueUntilAllFittingRecordsSync() throws {
        try isolated { defaults, history, store in
            var validIDs: [UUID] = []
            for index in 0..<12 {
                let id = UUID()
                validIDs.append(id)
                history.append(id: id, text: String(repeating: "字", count: 400) + "\(index)")
            }
            // Valid per contract (100k chars) but far over the test budget.
            let bigID = UUID()
            history.append(id: bigID, text: String(repeating: "a", count: 100_000))
            store.setEnabled(true)
            let serverID = UUID()
            var synced: Set<UUID> = []
            var cursor: Int64 = 0
            // Repeated pages: each apply acks only what was sent; the big record never ships.
            for _ in 0..<20 {
                guard let request = store.makeRequest(limit: 200, byteBudget: 4 * 1024) else { break }
                #expect(!request.items.contains { $0.id == bigID })
                #expect(request.items.allSatisfy { !synced.contains($0.id) }, "no record is uploaded twice")
                synced.formUnion(request.items.map(\.id))
                cursor += 1
                try store.apply(.init(storeID: serverID, cursor: cursor, hasMore: false, items: [], deletedIDs: []), sent: request)
                if request.items.isEmpty { break }
            }
            #expect(synced == Set(validIDs), "every fitting record syncs across pages")
            // The excluded record stays pending: raising the budget must pick it up.
            let repaired = try #require(store.makeRequest(limit: 200, byteBudget: HistorySyncStore.maxRequestBytes))
            #expect(repaired.items.map(\.id) == [bigID])
            _ = defaults
        }
    }

    @Test func contractInvalidRecordStaysLocalAndNeverShips() throws {
        try isolated { _, history, store in
            let good = UUID()
            history.append(id: good, text: "short")
            let invalidID = UUID()
            history.append(id: invalidID, text: String(repeating: "字", count: 100_001))
            store.setEnabled(true)
            let request = try #require(store.makeRequest(limit: 200, byteBudget: HistorySyncStore.maxRequestBytes))
            #expect(request.items.map(\.id) == [good], "over-100k record violates the contract and is skipped even with full budget")
            let serverID = UUID()
            try store.apply(.init(storeID: serverID, cursor: 1, hasMore: false, items: [], deletedIDs: []), sent: request)
            // Local text is untouched, and the invalid record remains pending (never acked).
            #expect(history.items.count == 2)
            #expect(history.items.first { $0.id == invalidID }?.text.count == 100_001)
            let next = try #require(store.makeRequest(limit: 200, byteBudget: HistorySyncStore.maxRequestBytes))
            #expect(next.items.isEmpty)
            #expect(store.lastSkippedRecordCount == 1)
        }
    }

    @Test func singlePathologicalGraphemeRecordIsSkippedNotForceAddedOverBudget() throws {
        try isolated { _, history, store in
            // One grapheme cluster: a base letter plus thousands of combining marks.
            // count == 1, so it is contract-valid — only the byte budget can stop it.
            let pathological = "e" + String(repeating: "\u{0301}", count: 5_000)
            #expect(pathological.count == 1)
            history.append(text: pathological)
            store.setEnabled(true)
            let budget = 4 * 1024
            let request = try #require(store.makeRequest(limit: 200, byteBudget: budget))
            #expect(request.items.isEmpty, "a record bigger than the whole budget must not be force-added")
            #expect(try JSONEncoder().encode(request).count <= budget)
            #expect(store.lastSkippedRecordCount == 1)
        }
    }

    @Test func impossibleEnvelopeReturnsNilAndMarksEverythingSkipped() throws {
        try isolated { _, history, store in
            history.append(text: "short")
            store.setEnabled(true)
            // A budget below the base envelope (storeID/cursor/limit/items/deletedIDs)
            // cannot carry even an empty page: no request may go out oversized.
            let request = store.makeRequest(limit: 200, byteBudget: 10)
            #expect(request == nil)
            #expect(store.lastSkippedRecordCount == 1)
        }
    }

    @Test func deletionsStillShipWhenItemsAreSkipped() throws {
        try isolated { _, history, store in
            let doomed = UUID()
            history.append(id: doomed, text: "delete me")
            history.append(text: String(repeating: "字", count: 500))
            store.setEnabled(true)
            history.delete(ids: [doomed])
            let request = try #require(store.makeRequest(limit: 200, byteBudget: 256))
            #expect(request.items.isEmpty)
            #expect(request.deletedIDs == [doomed])
        }
    }

    @Test @MainActor func coordinatorWarnsAboutSkippedRecordsInsteadOfUpToDate() async throws {
        let name = "history-sync-warn.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let history = TranscriptHistory(defaults: defaults), store = HistorySyncStore(defaults: defaults)
        history.append(text: "short")
        history.append(text: String(repeating: "字", count: 100_001))
        store.setEnabled(true)
        let coordinator = HistorySyncCoordinator(store: store) { request in
            .init(storeID: UUID(), cursor: request.cursor + 1, hasMore: false, items: [], deletedIDs: [])
        }
        await coordinator.sync()
        #expect(coordinator.lastError == "Some records are too large to sync. They remain on this iPhone.")
        #expect(coordinator.lastSyncedAt != nil, "valid records still synced")
    }

    @Test @MainActor func coordinatorStaysQuietWhenEverythingShips() async throws {
        let name = "history-sync-quiet.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let history = TranscriptHistory(defaults: defaults), store = HistorySyncStore(defaults: defaults)
        history.append(text: "short")
        store.setEnabled(true)
        let coordinator = HistorySyncCoordinator(store: store) { request in
            .init(storeID: UUID(), cursor: request.cursor + 1, hasMore: false, items: [], deletedIDs: [])
        }
        await coordinator.sync()
        #expect(coordinator.lastError == nil)
    }
}
