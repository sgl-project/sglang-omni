import Foundation
import Testing
@testable import VoxtMobileKit

struct HistorySyncTests {
    private func isolated(_ body: (UserDefaults, TranscriptHistory, HistorySyncStore) throws -> Void) rethrows {
        let name = "history-sync.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        try body(defaults, TranscriptHistory(defaults: defaults), HistorySyncStore(defaults: defaults))
    }

    @Test func optInQueuesExistingHistoryAndPreservesServerIdentity() throws {
        try isolated { defaults, history, store in
            let id = UUID()
            history.append(id: id, text: "Phone text", mode: .translate, elapsedSeconds: 7)
            #expect(store.makeRequest() == nil)
            store.setEnabled(true)
            let request = try #require(store.makeRequest())
            #expect(request.items.map(\.id) == [id])
            let serverID = UUID()
            try store.apply(.init(storeID: serverID, cursor: 1, hasMore: false, items: [.init(id: id, text: "Mac correction", createdAt: history.items[0].date.timeIntervalSince1970)], deletedIDs: []), sent: request)
            #expect(history.items.count == 1)
            #expect(history.items[0].id == id)
            #expect(history.items[0].text == "Mac correction")
            #expect(history.items[0].elapsedSeconds == 7)
            #expect(history.items[0].mode == .translate)
            let reloaded = HistorySyncStore(defaults: defaults)
            #expect(reloaded.makeRequest()?.storeID == serverID)
            #expect(reloaded.makeRequest()?.cursor == 1)
            #expect(reloaded.makeRequest()?.items.isEmpty == true)
            reloaded.setEnabled(false)
            reloaded.setEnabled(true)
            #expect(reloaded.makeRequest()?.cursor == 0, "reenabling must merge current Mac history, including unchanged rows")
        }
    }

    @Test func localDeletionWinsOverLateDownloadAndSurvivesRestart() throws {
        try isolated { defaults, history, store in
            let id = UUID()
            history.append(id: id, text: "Do not resurrect")
            store.setEnabled(true)
            let inFlight = try #require(store.makeRequest())
            history.delete(ids: [id])
            let response = HistorySyncResponse(storeID: UUID(), cursor: 1, hasMore: false, items: inFlight.items, deletedIDs: [])
            try store.apply(response, sent: inFlight)
            #expect(history.items.isEmpty)
            let reloaded = HistorySyncStore(defaults: defaults)
            let deletion = try #require(reloaded.makeRequest())
            #expect(deletion.deletedIDs == [id])
            try reloaded.apply(.init(storeID: response.storeID, cursor: 2, hasMore: false, items: [], deletedIDs: [id]), sent: deletion)
            #expect(reloaded.makeRequest()?.deletedIDs.isEmpty == true)
        }
    }

    @Test func retentionStaysLocalAndOptOutRejectsAnOldResponseAfterReenable() throws {
        try isolated { _, history, store in
            let now = Date(), oldID = UUID()
            history.append(id: oldID, text: "Old", now: now.addingTimeInterval(-8 * 86400))
            store.setEnabled(true)
            history.setRetention(.sevenDays)
            #expect(store.makeRequest()?.items.isEmpty == true)
            #expect(store.makeRequest()?.deletedIDs.isEmpty == true)
            let oldRequest = try #require(store.makeRequest())
            store.setEnabled(false)
            #expect(store.makeRequest() == nil)
            store.setEnabled(true)
            #expect(throws: HistorySyncError.staleRequest) {
                try store.apply(.init(storeID: UUID(), cursor: 1, hasMore: false, items: [.init(id: UUID(), text: "Late", createdAt: now.timeIntervalSince1970)], deletedIDs: []), sent: oldRequest)
            }
            #expect(history.items.isEmpty)
            history.setRetention(.off)
            #expect(!store.isEnabled)
            #expect(store.makeRequest() == nil)
        }
    }

    @Test func pendingDeletionSurvivesSyncToggleUntilAcknowledgedOrUnpaired() throws {
        try isolated { defaults, history, store in
            let id = UUID(), serverID = UUID()
            history.append(id: id, text: "Delete while offline")
            store.setEnabled(true)
            let upload = try #require(store.makeRequest())
            try store.apply(.init(storeID: serverID, cursor: 1, hasMore: false, items: upload.items, deletedIDs: []), sent: upload)
            history.delete(ids: [id])
            store.setEnabled(false)
            let restarted = HistorySyncStore(defaults: defaults)
            restarted.setEnabled(true)
            let retry = try #require(restarted.makeRequest())
            #expect(retry.deletedIDs == [id])
            try restarted.apply(.init(storeID: serverID, cursor: 2, hasMore: false, items: upload.items, deletedIDs: []), sent: retry)
            #expect(history.items.isEmpty)
            restarted.resetPairing()
            restarted.setEnabled(true)
            #expect(restarted.makeRequest()?.deletedIDs.isEmpty == true)
        }
    }

    @Test func rejectsMalformedPagesBeforeChangingHistoryOrCursor() throws {
        try isolated { _, history, store in
            store.setEnabled(true)
            let request = try #require(store.makeRequest()), id = UUID(), serverID = UUID()
            let item = HistorySyncItem(id: id, text: "Valid", createdAt: Date().timeIntervalSince1970)
            for response in [
                HistorySyncResponse(storeID: serverID, cursor: 0, hasMore: true, items: [], deletedIDs: []),
                .init(storeID: serverID, cursor: 0, hasMore: false, items: [item], deletedIDs: []),
                .init(storeID: serverID, cursor: 0, hasMore: false, items: [], deletedIDs: [id]),
                .init(storeID: serverID, cursor: -1, hasMore: false, items: [], deletedIDs: []),
                .init(storeID: serverID, cursor: 1, hasMore: false, items: [item, item], deletedIDs: []),
                .init(storeID: serverID, cursor: 1, hasMore: false, items: [item], deletedIDs: [id])
            ] {
                #expect(throws: HistorySyncError.invalidResponse) { try store.apply(response, sent: request) }
                #expect(history.items.isEmpty)
                #expect(store.makeRequest()?.cursor == 0)
            }
        }
    }

    @Test @MainActor func coordinatorMakesNoRequestWhenDisabledAndDiscardsLateReply() async throws {
        let name = "history-sync-coordinator.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let store = HistorySyncStore(defaults: defaults)
        var calls = 0
        let coordinator = HistorySyncCoordinator(store: store) { request in
            calls += 1
            store.setEnabled(false)
            store.setEnabled(true)
            return .init(storeID: UUID(), cursor: 1, hasMore: false, items: [.init(id: UUID(), text: "Late", createdAt: Date().timeIntervalSince1970)], deletedIDs: [])
        }
        await coordinator.sync()
        #expect(calls == 0)
        store.setEnabled(true)
        await coordinator.sync()
        #expect(calls == 1)
        #expect(TranscriptHistory(defaults: defaults).items.isEmpty)
    }

    @Test @MainActor func paginationAndPendingUploadsResumeAfterNetworkFailure() async throws {
        let name = "history-sync-pages.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let history = TranscriptHistory(defaults: defaults), store = HistorySyncStore(defaults: defaults)
        for index in 0..<201 { history.append(text: "Local \(index)") }
        store.setEnabled(true)
        let serverID = UUID(), importedID = UUID()
        var calls = 0
        let interrupted = HistorySyncCoordinator(store: store) { request in
            calls += 1
            if calls == 2 { throw URLError(.notConnectedToInternet) }
            #expect(request.items.count == 200)
            return .init(storeID: serverID, cursor: 1, hasMore: true,
                         items: [.init(id: importedID, text: "From Mac", createdAt: Date().timeIntervalSince1970)], deletedIDs: [])
        }
        await interrupted.sync()
        #expect(calls == 2)
        #expect(interrupted.lastError != nil)
        let restored = HistorySyncStore(defaults: defaults)
        let resumed = HistorySyncCoordinator(store: restored) { request in
            #expect(request.storeID == serverID)
            #expect(request.cursor == 1)
            #expect(request.items.count == 1)
            return .init(storeID: serverID, cursor: 2, hasMore: false, items: [], deletedIDs: [])
        }
        await resumed.sync()
        #expect(resumed.lastError == nil)
        #expect(restored.makeRequest()?.items.isEmpty == true)
        #expect(history.items.count == 202)
        #expect(history.items.filter { $0.id == importedID }.count == 1)
    }
}
