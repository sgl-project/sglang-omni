// HistorySyncTests.swift
// Covers POST /v1/history/sync: auth, validation, opt-in guards, revision pages, and tombstones.

import XCTest
@testable import Voxt

final class HistorySyncTests: XCTestCase {
    private static var retainedObjects: [AnyObject] = []
    private let router = LocalAPIRouter(token: "s3cret-token", maxBodyBytes: LocalAPISettings.maxBodyBytes)

    func testHistorySyncAuthenticatesBeforeReadingTheBody() throws {
        let (status, message) = rejection(token: nil, body: Data("nope".utf8))
        XCTAssertEqual(status, 401)
        XCTAssertEqual(message, "Missing or wrong access token.")
        let wrong = rejection(token: "other", body: Data())
        XCTAssertEqual(wrong.0, 401)
    }

    func testHistorySyncRejectsTheWrongMethodAndContentType() throws {
        let method = rejection(method: "GET", body: try syncJSON())
        XCTAssertEqual(method.0, 405)
        XCTAssertEqual(method.1, "Use POST.")
        let type = rejection(contentType: "text/plain", body: try syncJSON())
        XCTAssertEqual(type.0, 415)
        XCTAssertEqual(type.1, "Send application/json.")
        let missing = rejection(contentType: nil, body: try syncJSON())
        XCTAssertEqual(missing.0, 415)
    }

    func testHistorySyncRejectsInvalidJSONAndFields() throws {
        XCTAssertEqual(rejection(body: Data("nope".utf8)).1, "The history sync request is not valid JSON.")
        XCTAssertEqual(rejection(body: Data("[]".utf8)).1, "The history sync request is not valid JSON.")

        var fields = try baseFields()
        fields.removeValue(forKey: "cursor")
        XCTAssertEqual(rejection(body: try data(fields)).1, "The history sync request has invalid fields.")
        fields = try baseFields()
        fields["audio"] = "secret"
        XCTAssertEqual(rejection(body: try data(fields)).1, "The history sync request has invalid fields.")

        XCTAssertEqual(rejection(body: try syncJSON(cursor: -1)).1, "History sync cursor must be a non-negative integer.")
        XCTAssertEqual(rejection(body: try syncJSON(cursor: 1.5)).1, "History sync cursor must be a non-negative integer.")
        XCTAssertEqual(rejection(body: try syncJSON(cursor: true)).1, "History sync cursor must be a non-negative integer.")
        XCTAssertEqual(rejection(body: try syncJSON(limit: 0)).1, "History sync limit must be from 1 to 200.")
        XCTAssertEqual(rejection(body: try syncJSON(limit: 201)).1, "History sync limit must be from 1 to 200.")
        XCTAssertEqual(rejection(body: try syncJSON(items: Array(repeating: [:], count: 201))).1, "History sync accepts at most 200 items.")
        XCTAssertEqual(
            rejection(body: try syncJSON(deletedIDs: Array(repeating: UUID().uuidString, count: 201))).1,
            "History sync accepts at most 200 deletions."
        )
        XCTAssertEqual(rejection(body: try syncJSON(items: [itemJSON(text: "")])).1, "History sync text must be 1 to 100000 characters.")
        XCTAssertEqual(
            rejection(body: try syncJSON(items: [itemJSON(text: String(repeating: "a", count: 100_001))])).1,
            "History sync text must be 1 to 100000 characters."
        )
        XCTAssertEqual(rejection(body: try syncJSON(items: [itemJSON(id: "not-a-uuid")])).1, "History sync ids must be UUIDs.")
        let duplicated = UUID()
        XCTAssertEqual(
            rejection(body: try syncJSON(items: [itemJSON(id: duplicated.uuidString), itemJSON(id: duplicated.uuidString)])).1,
            "History sync ids must be unique."
        )
        XCTAssertEqual(
            rejection(body: try syncJSON(items: [itemJSON(id: duplicated.uuidString)], deletedIDs: [duplicated.uuidString])).1,
            "History sync ids must be unique."
        )
        XCTAssertEqual(rejection(body: try syncJSON(items: [itemJSON(createdAt: "now")])).1, "History sync times must be finite.")
        XCTAssertEqual(rejection(body: try syncJSON(items: [itemJSON(createdAt: NSNull())])).1, "History sync times must be finite.")
        XCTAssertEqual(rejection(body: try syncJSON(items: [itemJSON(mode: "compose")])).1, "History sync mode is invalid.")
        XCTAssertEqual(rejection(body: try syncJSON(items: [itemJSON(elapsedSeconds: -1)])).1, "History sync elapsed time must be a non-negative finite number.")
        XCTAssertEqual(rejection(body: try syncJSON(storeID: "nope")).1, "History sync storeID must be a UUID or null.")

        let repeated = UUID().uuidString
        XCTAssertEqual(rejection(body: try syncJSON(deletedIDs: [repeated, repeated])).1, "History sync ids must be unique.")
        var item = itemJSON()
        item.removeValue(forKey: "text")
        XCTAssertEqual(rejection(body: try syncJSON(items: [item])).1, "The history sync request has invalid fields.")
        item = itemJSON()
        item["audioRelativePath"] = "recording.wav"
        XCTAssertEqual(rejection(body: try syncJSON(items: [item])).1, "The history sync request has invalid fields.")
    }

    func testHistorySyncRejectsWhitespaceAndAnUnboundCursor() throws {
        XCTAssertEqual(
            rejection(body: try syncJSON(items: [itemJSON(text: " \n\t ")])).1,
            "History sync text must be 1 to 100000 characters."
        )
        XCTAssertEqual(
            rejection(body: try syncJSON(storeID: NSNull(), cursor: 4)).1,
            "History sync must start at cursor 0 when storeID is null."
        )
        let omitted = try data(["cursor": 4, "limit": 1, "items": [], "deletedIDs": []])
        XCTAssertEqual(rejection(body: omitted).1, "History sync must start at cursor 0 when storeID is null.")
        guard case .historySync = router.route(httpRequest(body: try syncJSON(items: [itemJSON(text: " hello ")]))) else {
            return XCTFail("words with surrounding spaces are still text")
        }
    }

    func testHistorySyncRejectsACursorPastIntMax() throws {
        let body = Data(#"{"storeID":null,"cursor":9223372036854775808,"limit":1,"items":[],"deletedIDs":[]}"#.utf8)
        let (status, message) = rejection(body: body)
        XCTAssertEqual(status, 400)
        XCTAssertEqual(message, "History sync cursor must be a non-negative integer.")
    }

    func testReplayAfterRetentionDoesNotConflictOnMetadata() throws {
        let (service, repository) = try makeService()
        let id = UUID()
        _ = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 10,
            items: [HistorySyncItem(id: id, text: "phone", createdAt: 50, mode: "write", elapsedSeconds: 3)],
            deletedIDs: []
        ))
        _ = try repository.deleteEntries(olderThan: Date(timeIntervalSince1970: 100), kinds: [.normal])
        XCTAssertNil(try repository.entry(id: id))
        XCTAssertEqual(try change(repository, id).deleted, false)
        let replay = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 10,
            items: [HistorySyncItem(id: id, text: "phone again", createdAt: 60, mode: "edit", elapsedSeconds: 9)],
            deletedIDs: []
        ))
        let entry = try XCTUnwrap(repository.entry(id: id))
        XCTAssertEqual(entry.text, "phone again")
        XCTAssertEqual(entry.createdAt.timeIntervalSince1970, 60, accuracy: 0)
        let item = try XCTUnwrap(itemDicts(replay).first {
            UUID(uuidString: $0["id"] as? String ?? "") == id
        })
        XCTAssertEqual(item["mode"] as? String, "write")
        XCTAssertEqual(number(item, "elapsedSeconds"), 3, accuracy: 0)
    }

    func testExportSkipsTextThePhoneWouldRejectAndAdvancesTheCursor() throws {
        let (service, repository) = try makeService()
        try repository.upsert(makeEntry(text: " \n\t ", createdAt: Date(timeIntervalSince1970: 10)))
        try repository.upsert(makeEntry(text: String(repeating: "a", count: 100_001), createdAt: Date(timeIntervalSince1970: 11)))
        try repository.upsert(makeEntry(text: "kept", createdAt: Date(timeIntervalSince1970: 12)))
        let first = try sync(service, HistorySyncRequest(storeID: nil, cursor: 0, limit: 2, items: [], deletedIDs: []))
        XCTAssertTrue(try itemDicts(first).isEmpty)
        XCTAssertTrue(try deletedIDs(first).isEmpty)
        XCTAssertEqual(first["hasMore"] as? Bool, true)
        let cursor = try XCTUnwrap(first["cursor"] as? Int)
        XCTAssertGreaterThan(cursor, 0)
        let second = try sync(service, HistorySyncRequest(storeID: nil, cursor: cursor, limit: 2, items: [], deletedIDs: []))
        XCTAssertEqual(try itemDicts(second).map { $0["text"] as? String }, ["kept"])
        XCTAssertEqual(second["hasMore"] as? Bool, false)
    }

    func testHistorySyncParsesAValidRequest() throws {
        let id = UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!
        let deleted = UUID(uuidString: "11111111-2222-3333-4444-555555555555")!
        let body = try syncJSON(
            cursor: 0,
            limit: 200,
            items: [itemJSON(id: id.uuidString, text: String(repeating: "a", count: 100_000), createdAt: -10, mode: "dictate", elapsedSeconds: 0)],
            deletedIDs: [deleted.uuidString]
        )
        let route = router.route(httpRequest(contentType: "application/json; charset=utf-8", body: body))
        XCTAssertEqual(
            route,
            .historySync(HistorySyncRequest(
                storeID: nil,
                cursor: 0,
                limit: 200,
                items: [HistorySyncItem(id: id, text: String(repeating: "a", count: 100_000), createdAt: -10, mode: "dictate", elapsedSeconds: 0)],
                deletedIDs: [deleted]
            ))
        )

        let omitted = try data([
            "cursor": 0,
            "limit": 1,
            "items": [["id": id.uuidString, "text": "hi", "createdAt": 10]],
            "deletedIDs": []
        ])
        XCTAssertEqual(
            router.route(httpRequest(body: omitted)),
            .historySync(HistorySyncRequest(
                storeID: nil,
                cursor: 0,
                limit: 1,
                items: [HistorySyncItem(id: id, text: "hi", createdAt: 10, mode: nil, elapsedSeconds: nil)],
                deletedIDs: []
            ))
        )
    }

    func testFreshDatabaseCreatesHistorySyncTables() throws {
        let directory = try makeTemporaryDirectory()
        let database = VoxtDatabase(databaseURL: directory.appendingPathComponent("voxt.sqlite"))
        retain(database)
        let tables = try database.debugSQLiteObjectNames(type: "table")
        XCTAssertTrue(tables.contains("history_sync_state"))
        XCTAssertTrue(tables.contains("history_sync_changes"))
        XCTAssertTrue(tables.contains("history_sync_metadata"))
        let repository = HistoryRepository(
            database: database,
            legacyJSONURL: directory.appendingPathComponent("missing-history.json"),
            migrateLegacyJSON: false
        )
        retain(repository)
        let state = try repository.historySyncState()
        XCTAssertEqual(state.revision, 0)
        XCTAssertEqual(try repository.historySyncState().storeID, state.storeID)
    }

    func testSyncAndHistoryGuardsRefuseWithoutWriting() throws {
        let item = HistorySyncItem(id: UUID(), text: "hidden", createdAt: 10, mode: "dictate", elapsedSeconds: nil)
        let request = HistorySyncRequest(storeID: nil, cursor: 0, limit: 20, items: [item], deletedIDs: [])

        let (syncOff, syncRepository) = try makeService(syncEnabled: false, historyEnabled: true)
        let syncResponse = syncOff.httpResponse(for: request)
        XCTAssertEqual(syncResponse.status, 403)
        XCTAssertEqual(try errorText(syncResponse), "History sync is turned off.")
        XCTAssertEqual(try syncRepository.historySyncState().revision, 0)
        XCTAssertNil(try syncRepository.entry(id: item.id))

        let (historyOff, historyRepository) = try makeService(syncEnabled: true, historyEnabled: false)
        let historyResponse = historyOff.httpResponse(for: request)
        XCTAssertEqual(historyResponse.status, 403)
        XCTAssertEqual(try errorText(historyResponse), "History is turned off on this Mac.")
        XCTAssertEqual(try historyRepository.historySyncState().revision, 0)
        XCTAssertNil(try historyRepository.entry(id: item.id))
    }

    func testWrongStoreIDDoesNotWrite() throws {
        let (service, repository) = try makeService()
        let before = try repository.historySyncState()
        let item = HistorySyncItem(id: UUID(), text: "nope", createdAt: 10, mode: nil, elapsedSeconds: nil)
        let response = service.httpResponse(for: HistorySyncRequest(
            storeID: UUID(),
            cursor: 0,
            limit: 20,
            items: [item],
            deletedIDs: [UUID()]
        ))
        XCTAssertEqual(response.status, 409)
        XCTAssertEqual(try errorText(response), "This history sync store does not match this Mac.")
        XCTAssertEqual(try repository.historySyncState(), before)
        XCTAssertNil(try repository.entry(id: item.id))
        XCTAssertNil(try repository.historySyncChange(id: item.id))
    }

    func testImportPreservesIdentityTimeAndLeavesAudioOffTheRow() throws {
        let (service, repository) = try makeService()
        let translated = UUID()
        let written = UUID()
        let edited = UUID()
        let page = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 20,
            items: [
                HistorySyncItem(id: translated, text: "hola", createdAt: 1_000_000_000, mode: "translate", elapsedSeconds: 2),
                HistorySyncItem(id: written, text: "note", createdAt: 1_000_000_100, mode: "write", elapsedSeconds: 0),
                HistorySyncItem(id: edited, text: "clearer", createdAt: 1_000_000_200, mode: "edit", elapsedSeconds: nil)
            ],
            deletedIDs: []
        ))
        let translatedEntry = try XCTUnwrap(repository.entry(id: translated))
        XCTAssertEqual(translatedEntry.text, "hola")
        XCTAssertEqual(translatedEntry.createdAt.timeIntervalSince1970, 1_000_000_000, accuracy: 0)
        XCTAssertEqual(translatedEntry.kind, .translation)
        XCTAssertTrue(translatedEntry.isTranslation)
        XCTAssertEqual(translatedEntry.focusedAppName, "Voxt Mobile")
        XCTAssertNil(translatedEntry.audioDurationSeconds)
        XCTAssertNil(translatedEntry.llmDurationSeconds)
        XCTAssertNil(translatedEntry.transcriptionProcessingDurationSeconds)
        XCTAssertNil(translatedEntry.audioRelativePath)
        XCTAssertNil(translatedEntry.transcriptAudioRelativePath)
        XCTAssertNil(translatedEntry.remoteASREndpoint)
        XCTAssertNil(translatedEntry.remoteLLMEndpoint)
        let writtenEntry = try XCTUnwrap(repository.entry(id: written))
        XCTAssertEqual(writtenEntry.kind, .normal)
        XCTAssertFalse(writtenEntry.isTranslation)
        XCTAssertEqual(writtenEntry.createdAt.timeIntervalSince1970, 1_000_000_100, accuracy: 0)
        XCTAssertEqual(try XCTUnwrap(repository.entry(id: edited)).kind, .rewrite)

        let items = try itemDicts(page)
        XCTAssertEqual(items.map { $0["text"] as? String }, ["hola", "note", "clearer"])
        XCTAssertEqual(items[0]["mode"] as? String, "translate")
        XCTAssertEqual(number(items[0], "elapsedSeconds"), 2, accuracy: 0)
        XCTAssertEqual(number(items[1], "elapsedSeconds"), 0, accuracy: 0)
        XCTAssertTrue(items[2]["elapsedSeconds"] is NSNull)
    }

    func testExistingRowKeepsMacTextAndTimeAndFillsMissingMetadataOnce() throws {
        let (service, repository) = try makeService()
        let id = UUID()
        try repository.upsert(makeEntry(
            id: id,
            text: "mac",
            createdAt: Date(timeIntervalSince1970: 1_111),
            llmDurationSeconds: 9,
            audioDurationSeconds: 8
        ))
        let first = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 10,
            items: [HistorySyncItem(id: id, text: "phone", createdAt: 2_222, mode: "write", elapsedSeconds: 4)],
            deletedIDs: []
        ))
        let entry = try XCTUnwrap(repository.entry(id: id))
        XCTAssertEqual(entry.text, "mac")
        XCTAssertEqual(entry.createdAt.timeIntervalSince1970, 1_111, accuracy: 0)
        XCTAssertEqual(entry.llmDurationSeconds, 9)
        XCTAssertEqual(entry.audioDurationSeconds, 8)
        let echoed = try XCTUnwrap(itemDicts(first).first)
        XCTAssertEqual(echoed["text"] as? String, "mac")
        XCTAssertEqual(number(echoed, "createdAt"), 1_111, accuracy: 0)
        XCTAssertEqual(echoed["mode"] as? String, "write")
        XCTAssertEqual(number(echoed, "elapsedSeconds"), 4, accuracy: 0)

        let revision = try repository.historySyncState().revision
        _ = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: revision,
            limit: 10,
            items: [HistorySyncItem(id: id, text: "phone again", createdAt: 3_333, mode: "edit", elapsedSeconds: 9)],
            deletedIDs: []
        ))
        XCTAssertEqual(try repository.historySyncState().revision, revision)
        XCTAssertEqual(try XCTUnwrap(repository.entry(id: id)).text, "mac")
        let again = try sync(service, HistorySyncRequest(storeID: nil, cursor: 0, limit: 10, items: [], deletedIDs: []))
        let item = try XCTUnwrap(itemDicts(again).first)
        XCTAssertEqual(item["mode"] as? String, "write")
        XCTAssertEqual(number(item, "elapsedSeconds"), 4, accuracy: 0)
    }

    func testMacDurationsAreNotExportedAsPhoneMetadata() throws {
        let (service, repository) = try makeService()
        let entry = makeEntry(
            text: "translated",
            createdAt: Date(timeIntervalSince1970: 50),
            kind: .translation,
            llmDurationSeconds: 12,
            audioDurationSeconds: 3
        )
        try repository.upsert(entry)
        let page = try sync(service, HistorySyncRequest(storeID: nil, cursor: 0, limit: 10, items: [], deletedIDs: []))
        let item = try XCTUnwrap(itemDicts(page).first)
        XCTAssertEqual(item["text"] as? String, "translated")
        XCTAssertTrue(item["mode"] is NSNull)
        XCTAssertTrue(item["elapsedSeconds"] is NSNull)
        XCTAssertEqual(try repository.entry(id: entry.id)?.llmDurationSeconds, 12)
    }

    func testTombstoneBlocksResurrectionAndSurvivesRetention() throws {
        let (service, repository) = try makeService()
        let id = UUID()
        _ = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 10,
            items: [HistorySyncItem(id: id, text: "gone", createdAt: 20, mode: "dictate", elapsedSeconds: nil)],
            deletedIDs: []
        ))
        _ = try sync(service, HistorySyncRequest(storeID: nil, cursor: 0, limit: 10, items: [], deletedIDs: [id]))
        XCTAssertNil(try repository.entry(id: id))
        XCTAssertEqual(try change(repository, id).deleted, true)
        let frozen = try repository.historySyncState().revision

        _ = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 10,
            items: [HistorySyncItem(id: id, text: "gone", createdAt: 20, mode: "dictate", elapsedSeconds: nil)],
            deletedIDs: [id]
        ))
        XCTAssertNil(try repository.entry(id: id))
        XCTAssertEqual(try repository.historySyncState().revision, frozen)
        try repository.upsert(makeEntry(id: id, text: "local again", createdAt: Date(timeIntervalSince1970: 30)))
        XCTAssertNil(try repository.entry(id: id))
        XCTAssertEqual(try change(repository, id).deleted, true)

        let other = makeEntry(text: "aged", createdAt: Date(timeIntervalSince1970: 5))
        try repository.upsert(other)
        _ = try repository.deleteEntries(olderThan: Date(timeIntervalSince1970: 10), kinds: [.normal])
        XCTAssertEqual(try change(repository, id).deleted, true)
        XCTAssertEqual(try change(repository, id).revision, frozen)
        XCTAssertEqual(try change(repository, other.id).deleted, false)
        let page = try sync(service, HistorySyncRequest(storeID: nil, cursor: 0, limit: 20, items: [], deletedIDs: []))
        XCTAssertTrue(try deletedIDs(page).contains(id))
        XCTAssertFalse(try deletedIDs(page).contains(other.id))
    }

    func testUnknownDeletionBlocksALaterUpload() throws {
        let (service, repository) = try makeService()
        let id = UUID()
        _ = try sync(service, HistorySyncRequest(storeID: nil, cursor: 0, limit: 10, items: [], deletedIDs: [id]))
        XCTAssertEqual(try change(repository, id).deleted, true)
        let revision = try repository.historySyncState().revision
        _ = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 10,
            items: [HistorySyncItem(id: id, text: "later", createdAt: 40, mode: nil, elapsedSeconds: nil)],
            deletedIDs: []
        ))
        XCTAssertNil(try repository.entry(id: id))
        XCTAssertEqual(try repository.historySyncState().revision, revision)
    }

    func testRetentionAdvancesTheCursorWithoutPropagatingDeletion() throws {
        let (service, repository) = try makeService()
        let newest = makeEntry(text: "newest", createdAt: Date(timeIntervalSince1970: 9_000))
        let oldest = makeEntry(text: "oldest", createdAt: Date(timeIntervalSince1970: 100))
        let middle = makeEntry(text: "middle", createdAt: Date(timeIntervalSince1970: 500))
        try repository.upsert(newest)
        try repository.upsert(oldest)
        try repository.upsert(middle)
        _ = try repository.deleteEntries(olderThan: Date(timeIntervalSince1970: 200), kinds: [.normal])

        let first = try sync(service, HistorySyncRequest(storeID: nil, cursor: 0, limit: 2, items: [], deletedIDs: []))
        XCTAssertEqual(try itemDicts(first).map { $0["text"] as? String }, ["newest", "middle"])
        XCTAssertEqual(first["cursor"] as? Int, 3)
        XCTAssertEqual(first["hasMore"] as? Bool, true)
        XCTAssertTrue(try deletedIDs(first).isEmpty)
        XCTAssertNil(try repository.entry(id: oldest.id))
        XCTAssertEqual(try change(repository, oldest.id).deleted, false)

        let second = try sync(service, HistorySyncRequest(storeID: nil, cursor: 3, limit: 2, items: [], deletedIDs: []))
        XCTAssertTrue(try itemDicts(second).isEmpty)
        XCTAssertTrue(try deletedIDs(second).isEmpty)
        XCTAssertEqual(second["cursor"] as? Int, try repository.historySyncState().revision)
        XCTAssertEqual(second["hasMore"] as? Bool, false)

        let (countService, countRepository) = try makeService()
        let aged = makeEntry(text: "aged", createdAt: Date(timeIntervalSince1970: 1_000))
        let kept = makeEntry(text: "kept", createdAt: Date(timeIntervalSince1970: 5_000))
        try countRepository.upsert(aged)
        try countRepository.upsert(kept)
        _ = try countRepository.deleteEntries(keepingNewest: 1, kind: .normal)
        XCTAssertNil(try countRepository.entry(id: aged.id))
        XCTAssertEqual(try change(countRepository, aged.id).deleted, false)
        XCTAssertEqual(try countRepository.entry(id: kept.id)?.text, "kept")
        let restored = try sync(countService, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 50,
            items: [HistorySyncItem(id: aged.id, text: "restored", createdAt: 1_000, mode: "dictate", elapsedSeconds: nil)],
            deletedIDs: []
        ))
        XCTAssertEqual(try countRepository.entry(id: aged.id)?.text, "restored")
        XCTAssertFalse(try deletedIDs(restored).contains(aged.id))
    }

    func testOldCreatedAtIsNotUsedAsTheCursor() throws {
        let (service, repository) = try makeService()
        let recent = makeEntry(text: "recent", createdAt: Date(timeIntervalSince1970: 2_000_000_000))
        try repository.upsert(recent)
        let oldID = UUID()
        let first = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 1,
            items: [HistorySyncItem(id: oldID, text: "from phone", createdAt: 1_000, mode: nil, elapsedSeconds: nil)],
            deletedIDs: []
        ))
        XCTAssertEqual(try itemDicts(first).map { $0["text"] as? String }, ["recent"])
        XCTAssertEqual(first["cursor"] as? Int, 1)
        XCTAssertEqual(first["hasMore"] as? Bool, true)
        let stored = try XCTUnwrap(repository.entry(id: oldID))
        XCTAssertEqual(stored.text, "from phone")
        XCTAssertEqual(stored.createdAt.timeIntervalSince1970, 1_000, accuracy: 0)
        let second = try sync(service, HistorySyncRequest(storeID: nil, cursor: 1, limit: 1, items: [], deletedIDs: []))
        let item = try XCTUnwrap(itemDicts(second).first)
        XCTAssertEqual(item["text"] as? String, "from phone")
        XCTAssertEqual(number(item, "createdAt"), 1_000, accuracy: 0)
        XCTAssertEqual(second["hasMore"] as? Bool, false)
    }

    func testTranscriptsAreNotExportedOrDeletedByThePhone() throws {
        let (service, repository) = try makeService()
        let meeting = makeEntry(text: "meeting", createdAt: Date(timeIntervalSince1970: 80), kind: .transcript)
        try repository.upsert(meeting)
        XCTAssertEqual(try repository.historySyncState().revision, 0)
        XCTAssertNil(try repository.historySyncChange(id: meeting.id))
        let page = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 10,
            items: [],
            deletedIDs: [meeting.id]
        ))
        XCTAssertEqual(page["cursor"] as? Int, 0)
        XCTAssertEqual(page["hasMore"] as? Bool, false)
        XCTAssertTrue(try itemDicts(page).isEmpty)
        XCTAssertTrue(try deletedIDs(page).isEmpty)
        XCTAssertEqual(try repository.entry(id: meeting.id)?.text, "meeting")
        XCTAssertNil(try repository.historySyncChange(id: meeting.id))
        XCTAssertEqual(try repository.delete(id: meeting.id)?.text, "meeting")
        XCTAssertNil(try repository.historySyncChange(id: meeting.id))
        XCTAssertEqual(try repository.historySyncState().revision, 0)
    }

    func testLocalEditsReviseAndExplicitDeletesTombstone() throws {
        let (service, repository) = try makeService()
        let normal = makeEntry(text: "local", createdAt: Date(timeIntervalSince1970: 70))
        let meeting = makeEntry(text: "meeting", createdAt: Date(timeIntervalSince1970: 71), kind: .transcript)
        try repository.upsert(normal)
        let afterInsert = try repository.historySyncState().revision
        XCTAssertGreaterThan(afterInsert, 0)
        try repository.upsert(meeting)
        XCTAssertEqual(try repository.historySyncState().revision, afterInsert)
        try repository.upsert(makeEntry(id: normal.id, text: "edited", createdAt: normal.createdAt))
        XCTAssertGreaterThan(try XCTUnwrap(repository.historySyncChange(id: normal.id)).revision, afterInsert)
        XCTAssertEqual(try repository.delete(id: normal.id)?.text, "edited")
        let frozen = try repository.historySyncState().revision
        XCTAssertEqual(try change(repository, normal.id).deleted, true)
        _ = try repository.delete(id: normal.id)
        XCTAssertEqual(try repository.historySyncState().revision, frozen)
        _ = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 10,
            items: [HistorySyncItem(id: normal.id, text: "edited", createdAt: 70, mode: nil, elapsedSeconds: nil)],
            deletedIDs: []
        ))
        XCTAssertNil(try repository.entry(id: normal.id))
        XCTAssertEqual(try repository.historySyncState().revision, frozen)
    }

    func testClearAllAndKindDeleteTombstoneExportableRowsOnly() throws {
        let (service, repository) = try makeService()
        let normal = makeEntry(text: "normal", createdAt: Date(timeIntervalSince1970: 10))
        let translation = makeEntry(text: "translation", createdAt: Date(timeIntervalSince1970: 11), kind: .translation)
        let meeting = makeEntry(text: "meeting", createdAt: Date(timeIntervalSince1970: 12), kind: .transcript)
        try repository.upsert(normal)
        try repository.upsert(translation)
        try repository.upsert(meeting)
        _ = try repository.deleteEntries(kind: .translation)
        XCTAssertEqual(try change(repository, translation.id).deleted, true)
        XCTAssertEqual(try change(repository, normal.id).deleted, false)
        XCTAssertNil(try repository.historySyncChange(id: meeting.id))
        try repository.clearAll()
        XCTAssertEqual(try change(repository, normal.id).deleted, true)
        XCTAssertNil(try repository.entry(id: meeting.id))
        XCTAssertNil(try repository.historySyncChange(id: meeting.id))
        let page = try sync(service, HistorySyncRequest(storeID: nil, cursor: 0, limit: 20, items: [], deletedIDs: []))
        let deleted = try deletedIDs(page)
        XCTAssertTrue(deleted.contains(normal.id))
        XCTAssertTrue(deleted.contains(translation.id))
        XCTAssertFalse(deleted.contains(meeting.id))
    }

    func testStoreIDStaysStableWhenTheDatabaseReopens() throws {
        let directory = try makeTemporaryDirectory()
        let url = directory.appendingPathComponent("voxt.sqlite")
        let firstDatabase = VoxtDatabase(databaseURL: url)
        retain(firstDatabase)
        let first = HistoryRepository(
            database: firstDatabase,
            legacyJSONURL: directory.appendingPathComponent("missing-history.json"),
            migrateLegacyJSON: false
        )
        retain(first)
        let service = HistorySyncService(repository: first, syncEnabled: true, historyEnabled: true)
        let id = UUID()
        let page = try sync(service, HistorySyncRequest(
            storeID: nil,
            cursor: 0,
            limit: 10,
            items: [HistorySyncItem(id: id, text: "phone", createdAt: 15, mode: "dictate", elapsedSeconds: nil)],
            deletedIDs: []
        ))
        let state = try first.historySyncState()
        XCTAssertEqual(UUID(uuidString: page["storeID"] as? String ?? ""), state.storeID)
        let secondDatabase = VoxtDatabase(databaseURL: url)
        retain(secondDatabase)
        let second = HistoryRepository(
            database: secondDatabase,
            legacyJSONURL: directory.appendingPathComponent("missing-history.json"),
            migrateLegacyJSON: false
        )
        retain(second)
        XCTAssertEqual(try second.historySyncState(), state)
        XCTAssertEqual(try second.entry(id: id)?.text, "phone")
    }

    func testEmptySyncKeepsTheRequestCursor() throws {
        let (service, repository) = try makeService()
        let state = try repository.historySyncState()
        let page = try sync(service, HistorySyncRequest(storeID: state.storeID, cursor: 7, limit: 1, items: [], deletedIDs: []))
        XCTAssertEqual(page["cursor"] as? Int, 7)
        XCTAssertEqual(page["hasMore"] as? Bool, false)
        XCTAssertEqual(UUID(uuidString: page["storeID"] as? String ?? ""), state.storeID)
        XCTAssertEqual(try repository.historySyncState().revision, 0)
    }

    func testHistorySyncDefaultsOffAndHistoryDefaultsOn() {
        let suite = "voxt-history-sync-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        XCTAssertFalse(LocalAPISettings.isHistorySyncEnabled(defaults: defaults))
        XCTAssertTrue(LocalAPISettings.isHistoryRecordingEnabled(defaults: defaults))
        defaults.set(false, forKey: AppPreferenceKey.historyEnabled)
        defaults.set(true, forKey: AppPreferenceKey.historySyncEnabled)
        XCTAssertFalse(LocalAPISettings.isHistoryRecordingEnabled(defaults: defaults))
        XCTAssertTrue(LocalAPISettings.isHistorySyncEnabled(defaults: defaults))
    }

    func testInvalidHistorySyncDoesNotReachTheSyncHandler() async throws {
        let flag = SyncCallFlag()
        let handler = LocalAPIRequestHandler.make(
            router: LocalAPIRouter(token: "t", maxBodyBytes: 1_000),
            version: "1",
            modelStatus: { LocalAPIModelStatus(transcription: .ready, text: .off) },
            sync: { _ in
                flag.called = true
                return .error(status: 500, "should not run")
            },
            dictate: { @MainActor _ in .json(status: 200, ["text": "x"]) },
            compose: { @MainActor _ in .json(status: 200, ["text": "x"]) }
        )
        let response = await handler(httpRequest(token: "t", body: Data("nope".utf8)))
        XCTAssertEqual(response.status, 400)
        XCTAssertFalse(flag.called)
    }

    func testUnwiredHistorySyncIsTurnedOff() async throws {
        let handler = LocalAPIRequestHandler.make(
            router: LocalAPIRouter(token: "t", maxBodyBytes: 1_000),
            version: "1",
            modelStatus: { LocalAPIModelStatus(transcription: .ready, text: .off) },
            dictate: { @MainActor _ in .json(status: 200, ["text": "x"]) },
            compose: { @MainActor _ in .json(status: 200, ["text": "x"]) }
        )
        let response = await handler(httpRequest(token: "t", body: try syncJSON()))
        XCTAssertEqual(response.status, 403)
        XCTAssertEqual(try errorText(response), "History sync is turned off.")
    }

    func testHistorySyncDoesNotWaitForTheMainActor() async throws {
        let handler = LocalAPIRequestHandler.make(
            router: LocalAPIRouter(token: "t", maxBodyBytes: 1_000_000),
            version: "1",
            modelStatus: { LocalAPIModelStatus(transcription: .ready, text: .off) },
            sync: { _ in .json(status: 200, ["ok": true]) },
            dictate: { @MainActor _ in .json(status: 200, ["text": "x"]) },
            compose: { @MainActor _ in .json(status: 200, ["text": "x"]) }
        )
        let mainBusy = Task { @MainActor in Thread.sleep(forTimeInterval: 1.5) }
        try await Task.sleep(for: .milliseconds(100))
        let body = try syncJSON()
        let started = Date()
        let response = await Task.detached {
            await handler(self.httpRequest(token: "t", body: body))
        }.value
        XCTAssertEqual(response.status, 200)
        XCTAssertLessThan(Date().timeIntervalSince(started), 0.5)
        await mainBusy.value
    }

    private func makeService(syncEnabled: Bool = true, historyEnabled: Bool = true) throws -> (HistorySyncService, HistoryRepository) {
        let directory = try makeTemporaryDirectory()
        let database = VoxtDatabase(databaseURL: directory.appendingPathComponent("voxt.sqlite"))
        retain(database)
        let repository = HistoryRepository(
            database: database,
            legacyJSONURL: directory.appendingPathComponent("missing-history.json"),
            migrateLegacyJSON: false
        )
        retain(repository)
        return (HistorySyncService(repository: repository, syncEnabled: syncEnabled, historyEnabled: historyEnabled), repository)
    }

    private func sync(_ service: HistorySyncService, _ request: HistorySyncRequest) throws -> [String: Any] {
        let response = service.httpResponse(for: request)
        let message = try errorText(response) ?? "status \(response.status)"
        XCTAssertEqual(response.status, 200, message)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: response.body) as? [String: Any])
    }

    private func errorText(_ response: LocalAPIHTTPResponse) throws -> String? {
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: response.body) as? [String: Any])
        return object["error"] as? String
    }

    private func itemDicts(_ object: [String: Any]) throws -> [[String: Any]] {
        let raw = try XCTUnwrap(object["items"] as? [Any])
        return try raw.map { try XCTUnwrap($0 as? [String: Any]) }
    }

    private func deletedIDs(_ object: [String: Any]) throws -> [UUID] {
        let raw = try XCTUnwrap(object["deletedIDs"] as? [String])
        return try raw.map { try XCTUnwrap(UUID(uuidString: $0)) }
    }

    private func change(_ repository: HistoryRepository, _ id: UUID) throws -> HistorySyncChange {
        try XCTUnwrap(repository.historySyncChange(id: id))
    }

    private func number(_ object: [String: Any], _ key: String) -> Double {
        (object[key] as? NSNumber)?.doubleValue ?? -1
    }

    private func rejection(
        method: String = "POST",
        token: String? = "s3cret-token",
        contentType: String? = "application/json",
        body: Data
    ) -> (Int, String) {
        let route = router.route(httpRequest(method, token: token, contentType: contentType, body: body))
        guard case let .reject(response) = route else {
            XCTFail("expected a rejection")
            return (0, "")
        }
        let message = (try? JSONSerialization.jsonObject(with: response.body) as? [String: Any])?["error"] as? String ?? ""
        return (response.status, message)
    }

    private func httpRequest(
        _ method: String = "POST",
        _ path: String = "/v1/history/sync",
        token: String? = "s3cret-token",
        contentType: String? = "application/json",
        body: Data = Data()
    ) -> LocalAPIHTTPRequest {
        var headers: [String: String] = [:]
        if let token { headers["authorization"] = "Bearer \(token)" }
        if let contentType { headers["content-type"] = contentType }
        return LocalAPIHTTPRequest(method: method, path: path, headers: headers, body: body)
    }

    private func syncJSON(
        storeID: Any = NSNull(),
        cursor: Any = 0,
        limit: Any = 50,
        items: [Any] = [],
        deletedIDs: [Any] = []
    ) throws -> Data {
        try data([
            "storeID": storeID,
            "cursor": cursor,
            "limit": limit,
            "items": items,
            "deletedIDs": deletedIDs
        ])
    }

    private func baseFields() throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: syncJSON()) as? [String: Any])
    }

    private func data(_ object: Any) throws -> Data {
        try JSONSerialization.data(withJSONObject: object)
    }

    private func itemJSON(
        id: Any = UUID().uuidString,
        text: String = "hello",
        createdAt: Any = 10,
        mode: Any = NSNull(),
        elapsedSeconds: Any = NSNull()
    ) -> [String: Any] {
        [
            "id": id,
            "text": text,
            "createdAt": createdAt,
            "mode": mode,
            "elapsedSeconds": elapsedSeconds
        ]
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-history-sync-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func retain<Value: AnyObject>(_ value: Value) -> Value {
        Self.retainedObjects.append(value)
        return value
    }

    private func makeEntry(
        id: UUID = UUID(),
        text: String,
        createdAt: Date,
        kind: TranscriptionHistoryKind = .normal,
        llmDurationSeconds: TimeInterval? = nil,
        audioDurationSeconds: TimeInterval? = nil
    ) -> TranscriptionHistoryEntry {
        TranscriptionHistoryEntry(
            id: id,
            text: text,
            createdAt: createdAt,
            transcriptionEngine: "engine",
            transcriptionModel: "model",
            enhancementMode: "mode",
            enhancementModel: "enhanced",
            kind: kind,
            isTranslation: kind == .translation,
            audioDurationSeconds: audioDurationSeconds,
            transcriptionProcessingDurationSeconds: nil,
            llmDurationSeconds: llmDurationSeconds,
            focusedAppName: "Notes",
            focusedAppBundleID: nil,
            matchedGroupID: nil,
            matchedGroupName: nil,
            matchedAppGroupName: nil,
            matchedURLGroupName: nil,
            remoteASRProvider: nil,
            remoteASRModel: nil,
            remoteASREndpoint: nil,
            remoteLLMProvider: nil,
            remoteLLMModel: nil,
            remoteLLMEndpoint: nil,
            whisperWordTimings: nil,
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        )
    }
}

private final class SyncCallFlag: @unchecked Sendable {
    var called = false
}
