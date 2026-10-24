import Foundation
import XCTest
@testable import Voxt

@MainActor
final class DictionarySuggestionStoreTests: XCTestCase {
    func testLegacySuggestionCodableRetainsRawValuesAndEvidence() throws {
        let data = Data(#"{"id":"00000000-0000-0000-0000-000000000001","term":"Voxt","normalizedTerm":"voxt","sourceContext":"repeatObservation","status":"dismissed","firstSeenAt":10,"lastSeenAt":20,"seenCount":3,"lastHistoryEntryID":"00000000-0000-0000-0000-000000000002","groupID":"00000000-0000-0000-0000-000000000003","groupNameSnapshot":"Legacy group","evidenceSamples":["legacy evidence"]}"#.utf8)
        let suggestion = try JSONDecoder().decode(DictionarySuggestion.self, from: data)
        XCTAssertEqual(suggestion.sourceContext, .repeatObservation)
        XCTAssertEqual(suggestion.status, .dismissed)
        XCTAssertEqual(suggestion.seenCount, 3)
        XCTAssertEqual(suggestion.evidenceSamples, ["legacy evidence"])
        let originalObject = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? NSDictionary)
        let encodedObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(suggestion)) as? NSDictionary
        )
        XCTAssertEqual(encodedObject, originalObject)
    }

    func testLegacyHistoryStillDecodesSuggestionSnapshots() throws {
        let entry = try legacyHistoryEntry()
        let snapshot = try XCTUnwrap(entry.dictionarySuggestedTerms.first)
        XCTAssertEqual(snapshot.term, "Voxt")
        XCTAssertEqual(snapshot.normalizedTerm, "voxt")
        XCTAssertEqual(snapshot.groupNameSnapshot, "Legacy group")
        XCTAssertEqual(snapshot.id, "voxt|00000000-0000-0000-0000-000000000003")
        let reloaded = try JSONDecoder().decode(
            TranscriptionHistoryEntry.self, from: JSONEncoder().encode(entry)
        )
        XCTAssertEqual(reloaded.dictionarySuggestedTerms, entry.dictionarySuggestedTerms)
    }

    func testReloadMergesLegacyDuplicatesWithoutLosingDismissalOrEvidence() throws {
        var older = legacySuggestion()
        older.status = .dismissed
        older.seenCount = 3
        older.evidenceSamples = ["old", "duplicate"]
        older.lastHistoryEntryID = UUID()
        var newer = legacySuggestion()
        newer.term = "VOXT"
        newer.sourceContext = .correction
        newer.lastSeenAt = Date(timeIntervalSinceReferenceDate: 30)
        newer.seenCount = 2
        newer.evidenceSamples = [" new ", "duplicate", ""]
        try withStore(items: [older, newer]) { store, url, _ in
            let merged = try XCTUnwrap(store.suggestions.first)
            XCTAssertEqual(store.suggestions.count, 1)
            XCTAssertEqual(merged.id, older.id)
            XCTAssertEqual(merged.term, "VOXT")
            XCTAssertEqual(merged.sourceContext, .correction)
            XCTAssertEqual(merged.status, .dismissed)
            XCTAssertEqual(merged.seenCount, 5)
            XCTAssertEqual(merged.firstSeenAt, older.firstSeenAt)
            XCTAssertEqual(merged.lastSeenAt, newer.lastSeenAt)
            XCTAssertEqual(merged.lastHistoryEntryID, older.lastHistoryEntryID)
            XCTAssertEqual(merged.evidenceSamples, ["new", "duplicate", "old"])
            XCTAssertEqual(
                try JSONDecoder().decode([DictionarySuggestion].self, from: Data(contentsOf: url)),
                store.suggestions
            )
            store.reload()
            XCTAssertEqual(store.suggestions.first?.seenCount, 5)
        }
    }

    func testAddedStatusWinsWhileDifferentScopesRemainSeparate() throws {
        let pending = legacySuggestion()
        var added = legacySuggestion()
        added.status = .added
        var scoped = legacySuggestion()
        scoped.groupID = UUID()
        scoped.status = .dismissed
        try withStore(items: [pending, added, scoped]) { store, _, _ in
            XCTAssertEqual(store.suggestions.count, 2)
            XCTAssertEqual(store.suggestions.first(where: { $0.groupID == nil })?.status, .added)
            XCTAssertEqual(store.suggestions.first(where: { $0.groupID == scoped.groupID })?.status, .dismissed)
        }
    }

    func testManualScanAddsDirectlyWithoutGrowingLegacySuggestionFile() throws {
        try withStore(items: [legacySuggestion()]) { store, url, defaults in
            let originalFile = try Data(contentsOf: url)
            let dictionary = DictionaryStore(
                defaults: defaults, fileManager: .default,
                initialEntries: [], persistenceEnabled: false
            )
            let candidates = ["OpenAI", "openai", " "].map {
                DictionaryHistoryScanCandidate(
                    term: $0, historyEntryIDs: [], groupID: nil,
                    groupNameSnapshot: nil, evidenceSample: "explicit scan"
                )
            }
            let result = store.applyHistoryScanCandidates(candidates, dictionaryStore: dictionary)
            XCTAssertEqual(result.newSuggestionCount, 1)
            XCTAssertEqual(result.duplicateCount, 1)
            XCTAssertEqual(dictionary.entries.map(\.term), ["OpenAI"])
            XCTAssertEqual(store.suggestions.count, 1)
            XCTAssertEqual(try Data(contentsOf: url), originalFile)
        }
    }

    func testSuccessfulScanPersistsCheckpointAndLastRunCounters() throws {
        let entry = try legacyHistoryEntry()
        try withStore { store, url, defaults in
            store.beginHistoryScan(totalCount: 3)
            store.updateHistoryScan(processedCount: 3, newSuggestionCount: 2, duplicateCount: 1)
            store.finishHistoryScan(
                processedCount: 3, newSuggestionCount: 2, duplicateCount: 1, checkpointEntry: entry
            )
            XCTAssertFalse(store.historyScanProgress.isRunning)
            XCTAssertEqual(store.historyScanProgress.lastProcessedCount, 3)
            XCTAssertEqual(store.historyScanProgress.lastNewSuggestionCount, 2)
            XCTAssertEqual(store.historyScanProgress.lastDuplicateCount, 1)
            XCTAssertNotNil(store.historyScanProgress.lastRunAt)
            let reloaded = DictionarySuggestionStore(defaults: defaults, legacySuggestionsURL: url)
            XCTAssertEqual(reloaded.historyScanCheckpoint, DictionaryHistoryScanCheckpoint(
                lastProcessedAt: entry.createdAt, lastHistoryEntryID: entry.id
            ))
        }
    }

    func testFailureAndCancellationKeepLastSuccessfulCheckpoint() throws {
        let entry = try legacyHistoryEntry()
        try withStore { store, _, _ in
            store.finishHistoryScan(
                processedCount: 3, newSuggestionCount: 2, duplicateCount: 1, checkpointEntry: entry
            )
            let checkpoint = store.historyScanCheckpoint
            store.beginHistoryScan(totalCount: 2)
            store.failHistoryScan(
                processedCount: 1, totalCount: 2, newSuggestionCount: 0, duplicateCount: 1,
                errorMessage: "failed"
            )
            XCTAssertFalse(store.historyScanProgress.isRunning)
            XCTAssertEqual(store.historyScanProgress.lastProcessedCount, 3)
            XCTAssertEqual(store.historyScanCheckpoint, checkpoint)
            store.beginHistoryScan(totalCount: 2)
            store.requestHistoryScanCancellation()
            XCTAssertTrue(store.historyScanProgress.isCancellationRequested)
            store.cancelHistoryScan(
                processedCount: 1, totalCount: 2, newSuggestionCount: 0, duplicateCount: 1,
                message: "cancelled"
            )
            XCTAssertFalse(store.historyScanProgress.isRunning)
            XCTAssertFalse(store.historyScanProgress.isCancellationRequested)
            XCTAssertEqual(store.historyScanProgress.lastProcessedCount, 1)
            XCTAssertEqual(store.historyScanProgress.errorMessage, "cancelled")
            XCTAssertEqual(store.historyScanCheckpoint, checkpoint)
        }
    }

    func testFilterSettingsPersistThroughIsolatedDefaults() throws {
        try withStore { store, url, defaults in
            store.saveFilterSettings(.init(prompt: " Custom prompt ", batchSize: 999, maxCandidatesPerBatch: 0))
            let reloaded = DictionarySuggestionStore(defaults: defaults, legacySuggestionsURL: url)
            XCTAssertEqual(reloaded.filterSettings.prompt, "Custom prompt")
            XCTAssertEqual(reloaded.filterSettings.batchSize, DictionarySuggestionFilterSettings.maximumBatchSize)
            XCTAssertEqual(reloaded.filterSettings.maxCandidatesPerBatch, DictionarySuggestionFilterSettings.minimumMaxCandidates)
            XCTAssertEqual(reloaded.filterSettings, store.filterSettings)
        }
    }

    private func legacySuggestion() -> DictionarySuggestion {
        DictionarySuggestion(
            term: "Voxt", normalizedTerm: "voxt", sourceContext: .history,
            firstSeenAt: Date(timeIntervalSinceReferenceDate: 10),
            lastSeenAt: Date(timeIntervalSinceReferenceDate: 20)
        )
    }

    private func legacyHistoryEntry() throws -> TranscriptionHistoryEntry {
        let data = Data(#"{"id":"00000000-0000-0000-0000-000000000002","text":"legacy","createdAt":20,"transcriptionEngine":"local","transcriptionModel":"legacy","enhancementMode":"off","enhancementModel":"","dictionarySuggestedTerms":[{"term":"Voxt","normalizedTerm":"voxt","groupID":"00000000-0000-0000-0000-000000000003","groupNameSnapshot":"Legacy group"}]}"#.utf8)
        return try JSONDecoder().decode(TranscriptionHistoryEntry.self, from: data)
    }

    private func withStore(
        items: [DictionarySuggestion] = [],
        _ body: (DictionarySuggestionStore, URL, UserDefaults) throws -> Void
    ) throws {
        let directory = try TemporaryDirectory()
        defer { withExtendedLifetime(directory) {} }
        let name = UUID().uuidString
        let defaults = TestDoubles.makeUserDefaults(testName: name)
        defer { defaults.removePersistentDomain(forName: "VoxtTests.\(name)") }
        let url = directory.url.appendingPathComponent("dictionary-suggestions.json")
        try JSONEncoder().encode(items).write(to: url)
        let store = DictionarySuggestionStore(defaults: defaults, legacySuggestionsURL: url)
        try body(store, url, defaults)
    }
}
