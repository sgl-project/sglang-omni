// TranscriptionHistoryStoreAsyncTests.swift

import Combine
import Foundation
import XCTest
@testable import Voxt

@MainActor
final class TranscriptionHistoryStoreAsyncTests: XCTestCase {
    func testClearAllInvalidatesInFlightReloadSnapshot() async throws {
        let repository = try makeFixture(entries: [makeEntry(index: 0)])
        let store = TranscriptionHistoryStore(repository: repository)

        repository.blockNextEntriesRequest()
        store.reloadAsync()
        XCTAssertTrue(repository.waitUntilRequestIsBlocked())

        store.clearAll()
        repository.releaseBlockedRequest()
        await drainMainQueue()

        XCTAssertTrue(store.entries.isEmpty)
        XCTAssertFalse(store.hasMore)
    }

    func testClearKindPreservesOtherHistoryKinds() async throws {
        let transcription = makeEntry(index: 0, kind: .normal)
        let translation = makeEntry(index: 1, kind: .translation)
        let rewrite = makeEntry(index: 2, kind: .rewrite)
        let meeting = makeEntry(index: 3, kind: .transcript)
        let repository = try makeFixture(entries: [transcription, translation, rewrite, meeting])
        let store = TranscriptionHistoryStore(repository: repository)

        XCTAssertTrue(store.clear(kind: .translation))

        XCTAssertNil(store.entry(id: translation.id))
        XCTAssertNotNil(store.entry(id: transcription.id))
        XCTAssertNotNil(store.entry(id: rewrite.id))
        XCTAssertNotNil(store.entry(id: meeting.id))
        await drainMainQueue()
    }

    func testRetentionCleanupPreservesMeetingHistory() async throws {
        let defaults = UserDefaults.standard
        let cleanupEnabledKey = AppPreferenceKey.historyCleanupEnabled
        let retentionPeriodKey = AppPreferenceKey.historyRetentionPeriod
        let retentionCountKey = AppPreferenceKey.historyRetentionCount
        let previousCleanupEnabled = defaults.object(forKey: cleanupEnabledKey)
        let previousRetentionPeriod = defaults.object(forKey: retentionPeriodKey)
        let previousRetentionCount = defaults.object(forKey: retentionCountKey)
        defer {
            if let previousCleanupEnabled {
                defaults.set(previousCleanupEnabled, forKey: cleanupEnabledKey)
            } else {
                defaults.removeObject(forKey: cleanupEnabledKey)
            }
            if let previousRetentionPeriod {
                defaults.set(previousRetentionPeriod, forKey: retentionPeriodKey)
            } else {
                defaults.removeObject(forKey: retentionPeriodKey)
            }
            if let previousRetentionCount {
                defaults.set(previousRetentionCount, forKey: retentionCountKey)
            } else {
                defaults.removeObject(forKey: retentionCountKey)
            }
        }

        defaults.set(true, forKey: cleanupEnabledKey)
        defaults.set(HistoryRetentionPeriod.oneDay.rawValue, forKey: retentionPeriodKey)
        defaults.set(HistoryRetentionCount.unlimited.rawValue, forKey: retentionCountKey)
        let oldDate = Date().addingTimeInterval(-2 * 24 * 60 * 60)
        let transcription = makeEntry(index: 0, kind: .normal, createdAt: oldDate)
        let translation = makeEntry(index: 1, kind: .translation, createdAt: oldDate)
        let rewrite = makeEntry(index: 2, kind: .rewrite, createdAt: oldDate)
        let meeting = makeEntry(index: 3, kind: .transcript, createdAt: oldDate)
        let repository = try makeFixture(entries: [transcription, translation, rewrite, meeting])
        let store = TranscriptionHistoryStore(repository: repository)

        XCTAssertNil(store.entry(id: transcription.id))
        XCTAssertNil(store.entry(id: translation.id))
        XCTAssertNil(store.entry(id: rewrite.id))
        XCTAssertNotNil(store.entry(id: meeting.id))
        await drainMainQueue()
    }

    func testRetentionCountCleanupKeepsNewestPerKindAndPreservesMeetings() async throws {
        let defaults = UserDefaults.standard
        let cleanupEnabledKey = AppPreferenceKey.historyCleanupEnabled
        let retentionPeriodKey = AppPreferenceKey.historyRetentionPeriod
        let retentionCountKey = AppPreferenceKey.historyRetentionCount
        let previousCleanupEnabled = defaults.object(forKey: cleanupEnabledKey)
        let previousRetentionPeriod = defaults.object(forKey: retentionPeriodKey)
        let previousRetentionCount = defaults.object(forKey: retentionCountKey)
        defer {
            if let previousCleanupEnabled {
                defaults.set(previousCleanupEnabled, forKey: cleanupEnabledKey)
            } else {
                defaults.removeObject(forKey: cleanupEnabledKey)
            }
            if let previousRetentionPeriod {
                defaults.set(previousRetentionPeriod, forKey: retentionPeriodKey)
            } else {
                defaults.removeObject(forKey: retentionPeriodKey)
            }
            if let previousRetentionCount {
                defaults.set(previousRetentionCount, forKey: retentionCountKey)
            } else {
                defaults.removeObject(forKey: retentionCountKey)
            }
        }

        defaults.set(true, forKey: cleanupEnabledKey)
        defaults.set(HistoryRetentionPeriod.forever.rawValue, forKey: retentionPeriodKey)
        defaults.set(HistoryRetentionCount.fifty.rawValue, forKey: retentionCountKey)

        let now = Date()
        var entries: [TranscriptionHistoryEntry] = []
        var newestNormalIDs: [UUID] = []
        var newestTranslationIDs: [UUID] = []
        var newestRewriteIDs: [UUID] = []

        for index in 0..<55 {
            let createdAt = now.addingTimeInterval(TimeInterval(-index))
            let normal = makeEntry(index: index, kind: .normal, createdAt: createdAt)
            let translation = makeEntry(index: 1000 + index, kind: .translation, createdAt: createdAt)
            let rewrite = makeEntry(index: 2000 + index, kind: .rewrite, createdAt: createdAt)
            entries.append(contentsOf: [normal, translation, rewrite])
            if index < 50 {
                newestNormalIDs.append(normal.id)
                newestTranslationIDs.append(translation.id)
                newestRewriteIDs.append(rewrite.id)
            }
        }

        let meeting = makeEntry(index: 9999, kind: .transcript, createdAt: now.addingTimeInterval(-10_000))
        entries.append(meeting)

        let repository = try makeFixture(entries: entries)
        let store = TranscriptionHistoryStore(repository: repository)

        for id in newestNormalIDs {
            XCTAssertNotNil(store.entry(id: id))
        }
        for id in newestTranslationIDs {
            XCTAssertNotNil(store.entry(id: id))
        }
        for id in newestRewriteIDs {
            XCTAssertNotNil(store.entry(id: id))
        }
        XCTAssertEqual(try repository.entryCount(kind: .normal, query: ""), 50)
        XCTAssertEqual(try repository.entryCount(kind: .translation, query: ""), 50)
        XCTAssertEqual(try repository.entryCount(kind: .rewrite, query: ""), 50)
        XCTAssertNotNil(try repository.entry(id: meeting.id))
        await drainMainQueue()
    }

    func testMutationInvalidatesInFlightNextPage() async throws {
        let entries = (0..<80).map { makeEntry(index: $0) }
        let target = entries[50]
        let repository = try makeFixture(entries: entries)
        let store = TranscriptionHistoryStore(repository: repository)
        XCTAssertEqual(store.entries.count, 40)

        repository.blockNextEntriesRequest()
        store.loadNextPage()
        XCTAssertTrue(repository.waitUntilRequestIsBlocked())

        XCTAssertTrue(store.delete(id: target.id))
        repository.releaseBlockedRequest()
        await drainMainQueue()

        XCTAssertNil(store.entry(id: target.id))
        XCTAssertEqual(store.entries.count, 40)
        XCTAssertTrue(store.hasMore)
    }

    func testNextPageMergesInDescendingOrderAndFinishesPagination() async throws {
        let now = Date()
        let entries = (0..<80).map { makeEntry(index: $0, createdAt: now.addingTimeInterval(TimeInterval(-$0))) }
        let repository = try makeFixture(entries: entries)
        let store = TranscriptionHistoryStore(repository: repository)
        XCTAssertEqual(store.entries, Array(entries.prefix(40)))
        XCTAssertTrue(store.hasMore)

        store.loadNextPage()
        await drainMainQueue()

        XCTAssertEqual(store.entries, entries)
        XCTAssertEqual(store.historyEntries(for: .normal), entries)
        XCTAssertFalse(store.hasMore)
        store.loadNextPage()
        await drainMainQueue()
        XCTAssertEqual(store.entries, entries)
    }

    func testMutationRestartsInterruptedReload() async throws {
        let initialEntries = [makeEntry(index: 0), makeEntry(index: 1)]
        let repository = try makeFixture(entries: initialEntries)
        let store = TranscriptionHistoryStore(repository: repository)
        let externallyAddedEntry = makeEntry(index: 2)
        try repository.upsert(externallyAddedEntry)

        repository.blockNextEntriesRequest()
        store.reloadAsync()
        XCTAssertTrue(repository.waitUntilRequestIsBlocked())

        XCTAssertTrue(store.delete(id: initialEntries[1].id))
        repository.releaseBlockedRequest()
        await drainMainQueue()

        XCTAssertNil(store.entry(id: initialEntries[1].id))
        XCTAssertNotNil(store.entry(id: externallyAddedEntry.id))
        XCTAssertTrue(store.entries.contains { $0.id == externallyAddedEntry.id })
    }

    func testFailedPersistenceDoesNotRestartReloadAndDiscardOptimisticState() async throws {
        let initialEntry = makeEntry(index: 0)
        let repository = try makeFixture(entries: [initialEntry])
        let store = TranscriptionHistoryStore(repository: repository)
        let externallyAddedEntry = makeEntry(index: 1)
        try repository.upsert(externallyAddedEntry)

        repository.blockNextEntriesRequest()
        store.reloadAsync()
        XCTAssertTrue(repository.waitUntilRequestIsBlocked())
        repository.failNextUpsertRequest()

        XCTAssertNotNil(store.updateSummaryChatMessages([], for: initialEntry.id))
        repository.releaseBlockedRequest()
        await drainMainQueue()

        XCTAssertNotNil(store.entry(id: initialEntry.id))
        XCTAssertFalse(store.entries.contains { $0.id == externallyAddedEntry.id })
    }

    func testRecentMenuEntriesAreIndependentOfVisibleHistoryPage() async throws {
        let unrelated = (0..<45).map { makeEntry(index: $0, kind: .rewrite) }
        let transcriptions = (50..<57).map { makeEntry(index: $0, kind: .normal) }
        let translations = (60..<67).map { makeEntry(index: $0, kind: .translation) }
        let repository = try makeFixture(entries: unrelated + transcriptions + translations)
        let store = TranscriptionHistoryStore(repository: repository)
        XCTAssertTrue(store.entries.allSatisfy { $0.kind == .rewrite })

        let candidates = await recentMenuEntries(in: store)
        XCTAssertEqual(candidates.count, 10)
        XCTAssertEqual(Set(candidates.map(\.id)), Set((Array(transcriptions.prefix(5)) + Array(translations.prefix(5))).map(\.id)))

        XCTAssertTrue(store.delete(id: transcriptions[0].id))
        let refreshed = await recentMenuEntries(in: store)
        XCTAssertFalse(refreshed.contains { $0.id == transcriptions[0].id })
        XCTAssertTrue(refreshed.contains { $0.id == transcriptions[5].id })

        store.clearAll()
        let cleared = await recentMenuEntries(in: store)
        XCTAssertTrue(cleared.isEmpty)
    }

    func testRecentMenuPreviewLoadsFullTextOnSelection() async throws {
        let text = String(repeating: "完整文本\n", count: 200)
        let entry = makeEntry(index: 0, text: text)
        let repository = try makeFixture(entries: [entry])
        let store = TranscriptionHistoryStore(repository: repository)
        let candidates = await recentMenuEntries(in: store)
        let candidate = try XCTUnwrap(candidates.first)
        XCTAssertLessThan(candidate.previewText.count, text.count)
        let loaded: TranscriptionHistoryEntry? = await withCheckedContinuation { continuation in
            store.loadEntry(id: candidate.id) { continuation.resume(returning: $0) }
        }
        XCTAssertEqual(loaded?.text, text)
    }

    func testCorrectionMergesTermsAndSnapshotsWithoutNormalizingStoredValues() throws {
        let later = DictionaryCorrectionSnapshot(originalText: "beta", correctedText: "Beta", finalLocation: 10, finalLength: 4)
        let earlier = DictionaryCorrectionSnapshot(originalText: "a", correctedText: "Alpha", finalLocation: 0, finalLength: 5)
        let shorter = DictionaryCorrectionSnapshot(originalText: "al", correctedText: "Al", finalLocation: 0, finalLength: 2)
        let original = makeEntry(index: 0, kind: .translation).updatingDictionaryCorrectionResult(
            text: "Alpha and Beta",
            dictionaryCorrectedTerms: [" Alpha ", "ALPHA", ""],
            dictionaryCorrectionSnapshots: [later, later]
        )
        let repository = try makeFixture(entries: [original])
        let store = TranscriptionHistoryStore(repository: repository)

        store.applyDictionaryCorrectionResult(
            historyID: original.id,
            updatedText: "  Alpha and Beta!\n",
            correctedTerms: ["alpha", " ", " Beta ", "BETA", "Gamma"],
            correctionSnapshots: [earlier, later, shorter, earlier]
        )

        let expected = original.updatingDictionaryCorrectionResult(
            text: "Alpha and Beta!",
            dictionaryCorrectedTerms: [" Alpha ", "ALPHA", "", " Beta ", "Gamma"],
            dictionaryCorrectionSnapshots: [shorter, earlier, later, later]
        )
        XCTAssertEqual(store.entry(id: original.id), expected)
        XCTAssertEqual(store.entries, [expected])
        XCTAssertEqual(try repository.entry(id: original.id), expected)
        XCTAssertEqual(store.historyEntries(for: .translation), [expected])
    }

    func testReplacementKeepsIncomingSnapshotOrderAndDuplicatesAndCanClearSnapshots() throws {
        let snapshot = DictionaryCorrectionSnapshot(originalText: "a", correctedText: "Alpha", finalLocation: 0, finalLength: 5)
        let later = DictionaryCorrectionSnapshot(originalText: "b", correctedText: "Beta", finalLocation: 10, finalLength: 4)
        let original = makeEntry(index: 0).updatingDictionaryCorrectionResult(
            text: "Alpha and Beta",
            dictionaryCorrectedTerms: ["Alpha"],
            dictionaryCorrectionSnapshots: [snapshot]
        )
        let repository = try makeFixture(entries: [original])
        let store = TranscriptionHistoryStore(repository: repository)

        store.replaceDictionaryCorrectionResult(
            historyID: original.id,
            updatedText: original.text,
            correctedTerms: ["alpha", "Beta"],
            correctionSnapshots: [later, snapshot, later]
        )
        let replaced = original.updatingDictionaryCorrectionResult(
            text: original.text,
            dictionaryCorrectedTerms: ["Alpha", "Beta"],
            dictionaryCorrectionSnapshots: [later, snapshot, later]
        )
        XCTAssertEqual(store.entry(id: original.id), replaced)
        XCTAssertEqual(try repository.entry(id: original.id), replaced)

        store.replaceDictionaryCorrectionResult(
            historyID: original.id,
            updatedText: original.text,
            correctedTerms: [],
            correctionSnapshots: []
        )
        XCTAssertEqual(store.entry(id: original.id)?.dictionaryCorrectedTerms, ["Alpha", "Beta"])
        XCTAssertEqual(store.entry(id: original.id)?.dictionaryCorrectionSnapshots, [])
        XCTAssertEqual(try repository.entry(id: original.id), store.entry(id: original.id))
    }

    func testEmptyAndUnchangedCorrectionsDoNotPublishOrPersist() throws {
        let later = DictionaryCorrectionSnapshot(originalText: "b", correctedText: "Beta", finalLocation: 10, finalLength: 4)
        let earlier = DictionaryCorrectionSnapshot(originalText: "a", correctedText: "Alpha", finalLocation: 0, finalLength: 5)
        let original = makeEntry(index: 0).updatingDictionaryCorrectionResult(
            text: "Alpha and Beta",
            dictionaryCorrectedTerms: ["Alpha"],
            dictionaryCorrectionSnapshots: [later, earlier]
        )
        let repository = try makeFixture(entries: [original])
        let store = TranscriptionHistoryStore(repository: repository)
        var publicationCount = 0
        let observation = store.$entries.dropFirst().sink { _ in publicationCount += 1 }
        defer { observation.cancel() }
        repository.failNextUpsertRequest()

        store.applyDictionaryCorrectionResult(
            historyID: original.id,
            updatedText: " \n",
            correctedTerms: ["Beta"],
            correctionSnapshots: [earlier]
        )
        store.replaceDictionaryCorrectionResult(
            historyID: original.id,
            updatedText: "\n",
            correctedTerms: [],
            correctionSnapshots: []
        )
        store.applyDictionaryCorrectionResult(
            historyID: original.id,
            updatedText: "  \(original.text)\n",
            correctedTerms: ["alpha"],
            correctionSnapshots: []
        )
        store.applyDictionaryCorrectedTerms([original.id: ["alpha", " "]])
        XCTAssertEqual(publicationCount, 0)
        XCTAssertEqual(store.entry(id: original.id), original)
        XCTAssertEqual(try repository.entry(id: original.id), original)

        store.applyDictionaryCorrectedTerms([original.id: ["Beta", "beta"]])
        XCTAssertEqual(publicationCount, 1)
        XCTAssertEqual(store.entry(id: original.id)?.dictionaryCorrectedTerms, ["Alpha", "Beta"])
        XCTAssertEqual(try repository.entry(id: original.id), original)
    }

    func testCorrectionPublicationStillPrecedesPersistenceButTermPublicationFollowsIt() throws {
        let original = makeEntry(index: 0)
        let repository = try makeFixture(entries: [original])
        let store = TranscriptionHistoryStore(repository: repository)
        var persistedAtPublication: [TranscriptionHistoryEntry] = []
        let observation = store.$entries.dropFirst().sink { _ in
            if let persisted = try? repository.entry(id: original.id) {
                persistedAtPublication.append(persisted)
            }
        }
        defer { observation.cancel() }

        store.applyDictionaryCorrectionResult(
            historyID: original.id,
            updatedText: "corrected",
            correctedTerms: ["Alpha"],
            correctionSnapshots: []
        )
        XCTAssertEqual(persistedAtPublication, [original])
        let corrected = try XCTUnwrap(store.entry(id: original.id))
        XCTAssertEqual(try repository.entry(id: original.id), corrected)

        store.replaceDictionaryCorrectionResult(
            historyID: original.id,
            updatedText: "replacement",
            correctedTerms: [],
            correctionSnapshots: []
        )
        XCTAssertEqual(persistedAtPublication, [original, corrected])

        store.applyDictionaryCorrectedTerms([original.id: ["Beta"]])
        let withTerms = try XCTUnwrap(store.entry(id: original.id))
        XCTAssertEqual(persistedAtPublication.last, withTerms)
        XCTAssertEqual(try repository.entry(id: original.id), withTerms)
    }

    private func recentMenuEntries(in store: TranscriptionHistoryStore) async -> [TranscriptionHistoryListEntry] {
        await withCheckedContinuation { continuation in
            store.loadRecentMenuEntries(limitPerKind: 5) { continuation.resume(returning: $0) }
        }
    }

    private func makeFixture(entries: [TranscriptionHistoryEntry]) throws -> BlockingHistoryRepository {
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-history-async-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let database = VoxtDatabase(databaseURL: directoryURL.appendingPathComponent("history.sqlite"))
        let baseRepository = HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false)
        try baseRepository.replaceAll(entries)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directoryURL)
        }
        return BlockingHistoryRepository(base: baseRepository)
    }

    private func makeEntry(
        index: Int,
        kind: TranscriptionHistoryKind = .normal,
        createdAt: Date? = nil,
        text: String? = nil
    ) -> TranscriptionHistoryEntry {
        TranscriptionHistoryEntry(
            id: UUID(),
            text: text ?? "entry-\(index)",
            createdAt: createdAt ?? Date().addingTimeInterval(TimeInterval(-index)),
            transcriptionEngine: "engine",
            transcriptionModel: "model",
            enhancementMode: "off",
            enhancementModel: "none",
            kind: kind,
            isTranslation: false,
            audioDurationSeconds: nil,
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
            audioRelativePath: nil,
            whisperWordTimings: nil,
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        )
    }

    private func drainMainQueue() async {
        for _ in 0..<20 {
            await Task.yield()
            try? await Task.sleep(for: .milliseconds(10))
        }
    }
}

private final class BlockingHistoryRepository: HistoryRepositoryProtocol, @unchecked Sendable {
    private let base: HistoryRepositoryProtocol
    private let lock = NSLock()
    private let requestStarted = DispatchSemaphore(value: 0)
    private let requestRelease = DispatchSemaphore(value: 0)
    private var shouldBlockNextEntriesRequest = false
    private var shouldFailNextUpsertRequest = false

    init(base: HistoryRepositoryProtocol) {
        self.base = base
    }

    func blockNextEntriesRequest() {
        lock.withLock {
            shouldBlockNextEntriesRequest = true
        }
    }

    func waitUntilRequestIsBlocked() -> Bool {
        requestStarted.wait(timeout: .now() + 2) == .success
    }

    func releaseBlockedRequest() {
        requestRelease.signal()
    }

    func failNextUpsertRequest() {
        lock.withLock {
            shouldFailNextUpsertRequest = true
        }
    }

    func entries(
        kind: TranscriptionHistoryKind?,
        query: String,
        limit: Int?,
        offset: Int
    ) throws -> [TranscriptionHistoryEntry] {
        let snapshot = try base.entries(kind: kind, query: query, limit: limit, offset: offset)
        let shouldBlock = lock.withLock {
            let value = shouldBlockNextEntriesRequest
            shouldBlockNextEntriesRequest = false
            return value
        }
        if shouldBlock {
            requestStarted.signal()
            _ = requestRelease.wait(timeout: .now() + 2)
        }
        return snapshot
    }

    func listEntries(
        kind: TranscriptionHistoryKind?,
        query: String,
        limit: Int?,
        offset: Int
    ) throws -> [TranscriptionHistoryListEntry] {
        try base.listEntries(kind: kind, query: query, limit: limit, offset: offset)
    }

    func entry(id: UUID) throws -> TranscriptionHistoryEntry? {
        try base.entry(id: id)
    }

    func latestEntryText() throws -> String? {
        try base.latestEntryText()
    }

    func audioRelativePaths() throws -> [String] {
        try base.audioRelativePaths()
    }

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

    func upsert(_ entry: TranscriptionHistoryEntry) throws {
        let shouldFail = lock.withLock {
            let value = shouldFailNextUpsertRequest
            shouldFailNextUpsertRequest = false
            return value
        }
        if shouldFail {
            throw HistoryTestRepositoryError.writeFailed
        }
        try base.upsert(entry)
    }

    func delete(id: UUID) throws -> TranscriptionHistoryEntry? {
        try base.delete(id: id)
    }

    func clearAll() throws {
        try base.clearAll()
    }

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

private enum HistoryTestRepositoryError: Error {
    case writeFailed
}
