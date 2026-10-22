// MeetingHistoryDurabilityTests.swift
// Verifies meeting history reports durable database failures and uses idempotent IDs.

import XCTest
@testable import Voxt

@MainActor
final class MeetingHistoryDurabilityTests: XCTestCase {
    private static var retainedObjects: [AnyObject] = []

    func testAppendReturnsNilAndDoesNotPublishWhenRepositoryUpsertFails() throws {
        let directory = retain(try TemporaryDirectory())
        let database = retain(VoxtDatabase(databaseURL: directory.url.appendingPathComponent("history.sqlite")))
        let base = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let repository = retain(FailingMeetingHistoryRepository(base: base))
        let store = retain(TranscriptionHistoryStore(repository: repository))
        let entryID = UUID()

        let result = appendMeeting(store: store, entryID: entryID, text: "must remain recoverable")

        XCTAssertNil(result)
        XCTAssertNil(store.entry(id: entryID))
        XCTAssertTrue(store.entries.isEmpty)
    }

    func testDeterministicMeetingEntryIDMakesRecoveryIdempotent() throws {
        let directory = retain(try TemporaryDirectory())
        let database = retain(VoxtDatabase(databaseURL: directory.url.appendingPathComponent("history.sqlite")))
        let repository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let store = retain(TranscriptionHistoryStore(repository: repository))
        let entryID = UUID()

        XCTAssertEqual(appendMeeting(store: store, entryID: entryID, text: "first recovery"), entryID)
        XCTAssertEqual(appendMeeting(store: store, entryID: entryID, text: "duplicate recovery"), entryID)
        XCTAssertEqual(try repository.entryCount(kind: .transcript, query: ""), 1)
        XCTAssertEqual(try repository.entry(id: entryID)?.text, "first recovery")
    }

    private func appendMeeting(
        store: TranscriptionHistoryStore,
        entryID: UUID,
        text: String
    ) -> UUID? {
        store.append(
            entryID: entryID,
            text: text,
            transcriptionEngine: "MLX Audio",
            transcriptionModel: "Visible local model",
            enhancementMode: "Off",
            enhancementModel: "None",
            kind: .transcript,
            isTranslation: false,
            audioDurationSeconds: 10,
            transcriptionProcessingDurationSeconds: nil,
            llmDurationSeconds: nil,
            focusedAppName: nil,
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
            transcriptSegments: [],
            meetingCaptureMode: .meeting,
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        )
    }

    private func retain<Value: AnyObject>(_ value: Value) -> Value {
        Self.retainedObjects.append(value)
        return value
    }
}

private final class FailingMeetingHistoryRepository: HistoryRepositoryProtocol, @unchecked Sendable {
    private let base: HistoryRepositoryProtocol

    init(base: HistoryRepositoryProtocol) {
        self.base = base
    }

    func entries(kind: TranscriptionHistoryKind?, query: String, limit: Int?, offset: Int) throws -> [TranscriptionHistoryEntry] {
        try base.entries(kind: kind, query: query, limit: limit, offset: offset)
    }

    func listEntries(
        kind: TranscriptionHistoryKind?,
        query: String,
        limit: Int?,
        offset: Int
    ) throws -> [TranscriptionHistoryListEntry] {
        try base.listEntries(kind: kind, query: query, limit: limit, offset: offset)
    }

    func entry(id: UUID) throws -> TranscriptionHistoryEntry? { try base.entry(id: id) }
    func latestEntryText() throws -> String? { try base.latestEntryText() }
    func audioRelativePaths() throws -> [String] { try base.audioRelativePaths() }
    func entryCount(kind: TranscriptionHistoryKind?, query: String) throws -> Int {
        try base.entryCount(kind: kind, query: query)
    }
    func pendingNormalEntryCount(after checkpoint: DictionaryHistoryScanCheckpoint?) throws -> Int {
        try base.pendingNormalEntryCount(after: checkpoint)
    }
    func pendingNormalEntries(after checkpoint: DictionaryHistoryScanCheckpoint?) throws -> [TranscriptionHistoryEntry] {
        try base.pendingNormalEntries(after: checkpoint)
    }
    func reportMetrics(dayStarts: [Date], branchStartDate: Date?) throws -> HistoryReportMetrics {
        try base.reportMetrics(dayStarts: dayStarts, branchStartDate: branchStartDate)
    }
    func upsert(_ entry: TranscriptionHistoryEntry) throws { throw MeetingHistoryDurabilityTestError.upsertFailed }
    func delete(id: UUID) throws -> TranscriptionHistoryEntry? { try base.delete(id: id) }
    func clearAll() throws { try base.clearAll() }
    func deleteEntries(kind: TranscriptionHistoryKind) throws -> [TranscriptionHistoryEntry] {
        try base.deleteEntries(kind: kind)
    }
    func deleteEntries(
        olderThan cutoff: Date,
        kinds: Set<TranscriptionHistoryKind>
    ) throws -> [TranscriptionHistoryEntry] {
        try base.deleteEntries(olderThan: cutoff, kinds: kinds)
    }

    func deleteEntries(
        keepingNewest count: Int,
        kind: TranscriptionHistoryKind
    ) throws -> [TranscriptionHistoryEntry] {
        try base.deleteEntries(keepingNewest: count, kind: kind)
    }
}

private enum MeetingHistoryDurabilityTestError: Error {
    case upsertFailed
}
