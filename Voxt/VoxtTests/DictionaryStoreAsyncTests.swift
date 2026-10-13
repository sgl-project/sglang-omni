// DictionaryStoreAsyncTests.swift

import Foundation
import XCTest
@testable import Voxt

@MainActor
final class DictionaryStoreAsyncTests: XCTestCase {
    func testMutationCompletesPendingReloadBeforeUsingInMemorySnapshot() async throws {
        let repository = try makeFixture()
        let persistedEntry = makeEntry(term: "Persisted")
        try repository.upsert(persistedEntry)
        let store = DictionaryStore(
            defaults: UserDefaults(suiteName: "DictionaryStoreAsyncTests.\(UUID().uuidString)")!,
            fileManager: .default,
            repository: repository
        )
        let externallyAddedEntry = makeEntry(term: "External")
        try repository.upsert(externallyAddedEntry)

        repository.blockNextAllEntriesRequest()
        store.reloadAsync()
        XCTAssertTrue(repository.waitUntilRequestIsBlocked())

        try store.createManualEntry(
            term: "New Entry",
            groupID: nil,
            groupNameSnapshot: nil
        )
        repository.releaseBlockedRequest()
        await drainMainQueue()

        XCTAssertEqual(
            Set(store.entries.map(\.term)),
            Set(["Persisted", "External", "New Entry"])
        )
        XCTAssertEqual(
            Set(try repository.allEntries().map(\.term)),
            Set(["Persisted", "External", "New Entry"])
        )
    }

    func testMutationAbortsWhenPendingReloadCannotReadCompleteSnapshot() async throws {
        let repository = try makeFixture()
        let persistedEntry = makeEntry(term: "Persisted")
        try repository.upsert(persistedEntry)
        let store = DictionaryStore(
            defaults: UserDefaults(suiteName: "DictionaryStoreAsyncTests.\(UUID().uuidString)")!,
            fileManager: .default,
            repository: repository
        )

        repository.blockNextAllEntriesRequest()
        store.reloadAsync()
        XCTAssertTrue(repository.waitUntilRequestIsBlocked())
        repository.failNextAllEntriesRequest()

        XCTAssertThrowsError(
            try store.createManualEntry(
                term: "Must Not Be Written",
                groupID: nil,
                groupNameSnapshot: nil
            )
        ) { error in
            guard let dictionaryError = error as? DictionaryStoreError,
                  case .dataUnavailable = dictionaryError
            else {
                return XCTFail("Expected DictionaryStoreError.dataUnavailable, got \(error)")
            }
        }
        repository.releaseBlockedRequest()
        await drainMainQueue()

        XCTAssertEqual(store.entries.map(\.term), ["Persisted"])
        XCTAssertEqual(try repository.allEntries().map(\.term), ["Persisted"])
    }

    func testNewReplacementEntryIsReturnedByPagedQueryImmediatelyAfterCreation() async throws {
        let repository = try makeFixture()
        let store = DictionaryStore(
            defaults: UserDefaults(suiteName: "DictionaryStoreAsyncTests.\(UUID().uuidString)")!,
            fileManager: .default,
            repository: repository
        )

        try store.createManualEntry(
            term: "Voxt Target",
            replacementTerms: ["vox target"],
            groupID: nil,
            groupNameSnapshot: nil
        )

        let queryExpectation = expectation(description: "replacement entry query completes")
        var queriedEntries: [DictionaryEntry] = []
        store.loadEntries(
            requiringReplacementTerms: true,
            limit: 80,
            offset: 0
        ) { _, entries in
            queriedEntries = entries
            queryExpectation.fulfill()
        }

        await fulfillment(of: [queryExpectation], timeout: 2)
        XCTAssertEqual(queriedEntries.map(\.term), ["Voxt Target"])
        XCTAssertEqual(queriedEntries.first?.replacementTerms.map(\.text), ["vox target"])
    }

    func testReplacementTermsAreAddedWhenTargetEntryAlreadyExists() async throws {
        let repository = try makeFixture()
        let store = DictionaryStore(
            defaults: UserDefaults(suiteName: "DictionaryStoreAsyncTests.\(UUID().uuidString)")!,
            fileManager: .default,
            repository: repository
        )

        try store.createManualEntry(
            term: "Voxt Target",
            groupID: nil,
            groupNameSnapshot: nil
        )
        try store.createManualEntry(
            term: "voxt target",
            replacementTerms: ["vox target"],
            groupID: nil,
            groupNameSnapshot: nil
        )

        let queryExpectation = expectation(description: "replacement entry query completes")
        var queriedEntries: [DictionaryEntry] = []
        store.loadEntries(
            requiringReplacementTerms: true,
            limit: 80,
            offset: 0
        ) { _, entries in
            queriedEntries = entries
            queryExpectation.fulfill()
        }

        await fulfillment(of: [queryExpectation], timeout: 2)
        XCTAssertEqual(try repository.allEntries().count, 1)
        XCTAssertEqual(queriedEntries.map(\.term), ["Voxt Target"])
        XCTAssertEqual(queriedEntries.first?.replacementTerms.map(\.text), ["vox target"])
    }

    private func makeFixture() throws -> BlockingDictionaryRepository {
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-dictionary-async-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let database = VoxtDatabase(databaseURL: directoryURL.appendingPathComponent("dictionary.sqlite"))
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directoryURL)
        }
        return BlockingDictionaryRepository(
            base: DictionaryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false)
        )
    }

    private func makeEntry(term: String) -> DictionaryEntry {
        DictionaryEntry(
            term: term,
            normalizedTerm: DictionaryStore.normalizeTerm(term),
            source: .manual
        )
    }

    private func drainMainQueue() async {
        for _ in 0..<20 {
            await Task.yield()
            try? await Task.sleep(for: .milliseconds(10))
        }
    }

    func testCreateManualEntryReinforcesExistingTermInsteadOfFailing() async throws {
        let repository = try makeFixture()
        try repository.upsert(makeEntry(term: "Hello"))
        let store = DictionaryStore(
            defaults: UserDefaults(suiteName: "DictionaryStoreAsyncTests.\(UUID().uuidString)")!,
            fileManager: .default,
            repository: repository
        )
        await drainMainQueue()

        // A second manual entry whose normalized form already exists in the same scope
        // must not raise SQLite error 19 (idx_dictionary_normalized_scope); it reinforces.
        try store.createManualEntry(
            term: "hello",
            groupID: nil,
            groupNameSnapshot: nil
        )
        await drainMainQueue()

        let persisted = try repository.allEntries()
        XCTAssertEqual(persisted.count, 1)
        XCTAssertEqual(Set(store.entries.map(\.term)), Set(["Hello"]))
        XCTAssertEqual(persisted.first?.matchCount, 1)
        XCTAssertNotNil(persisted.first?.lastMatchedAt)
    }
}

private final class BlockingDictionaryRepository: DictionaryRepositoryProtocol, @unchecked Sendable {
    private let base: DictionaryRepositoryProtocol
    private let lock = NSLock()
    private let requestStarted = DispatchSemaphore(value: 0)
    private let requestRelease = DispatchSemaphore(value: 0)
    private var shouldBlockNextAllEntriesRequest = false
    private var shouldFailNextAllEntriesRequest = false

    init(base: DictionaryRepositoryProtocol) {
        self.base = base
    }

    func blockNextAllEntriesRequest() {
        lock.withLock {
            shouldBlockNextAllEntriesRequest = true
        }
    }

    func waitUntilRequestIsBlocked() -> Bool {
        requestStarted.wait(timeout: .now() + 2) == .success
    }

    func releaseBlockedRequest() {
        requestRelease.signal()
    }

    func failNextAllEntriesRequest() {
        lock.withLock {
            shouldFailNextAllEntriesRequest = true
        }
    }

    func allCategories() throws -> [DictionaryCategory] {
        try base.allCategories()
    }

    func allEntries() throws -> [DictionaryEntry] {
        let shouldFail = lock.withLock {
            let value = shouldFailNextAllEntriesRequest
            shouldFailNextAllEntriesRequest = false
            return value
        }
        if shouldFail {
            throw TestRepositoryError.readFailed
        }
        let snapshot = try base.allEntries()
        let shouldBlock = lock.withLock {
            let value = shouldBlockNextAllEntriesRequest
            shouldBlockNextAllEntriesRequest = false
            return value
        }
        if shouldBlock {
            requestStarted.signal()
            _ = requestRelease.wait(timeout: .now() + 2)
        }
        return snapshot
    }

    func allTerms(limit: Int?) throws -> [String] {
        try base.allTerms(limit: limit)
    }

    func entries(filter: DictionaryFilter, query: String, limit: Int, offset: Int) throws -> [DictionaryEntry] {
        try base.entries(filter: filter, query: query, limit: limit, offset: offset)
    }

    func entryCount(filter: DictionaryFilter, query: String) throws -> Int {
        try base.entryCount(filter: filter, query: query)
    }

    func entries(requiringReplacementTerms: Bool, query: String, limit: Int, offset: Int) throws -> [DictionaryEntry] {
        try base.entries(
            requiringReplacementTerms: requiringReplacementTerms,
            query: query,
            limit: limit,
            offset: offset
        )
    }

    func entryCount(requiringReplacementTerms: Bool, query: String) throws -> Int {
        try base.entryCount(requiringReplacementTerms: requiringReplacementTerms, query: query)
    }

    func matchingEntries(sourceText: String, activeGroupID: UUID?, limit: Int) throws -> [DictionaryEntry] {
        try base.matchingEntries(sourceText: sourceText, activeGroupID: activeGroupID, limit: limit)
    }

    func activeEntriesForRemoteRequest(activeGroupID: UUID?, limit: Int) throws -> [DictionaryEntry] {
        try base.activeEntriesForRemoteRequest(activeGroupID: activeGroupID, limit: limit)
    }

    func upsert(_ entry: DictionaryEntry) throws {
        try base.upsert(entry)
    }

    func upsertCategory(_ category: DictionaryCategory) throws {
        try base.upsertCategory(category)
    }

    func replaceAll(_ entries: [DictionaryEntry]) throws {
        try base.replaceAll(entries)
    }

    func replaceAll(entries: [DictionaryEntry], categories: [DictionaryCategory]) throws {
        try base.replaceAll(entries: entries, categories: categories)
    }

    func delete(id: UUID) throws {
        try base.delete(id: id)
    }

    func deleteCategory(id: UUID, moveEntriesTo category: DictionaryCategory?) throws {
        try base.deleteCategory(id: id, moveEntriesTo: category)
    }

    func clearAll() throws {
        try base.clearAll()
    }

    func hasEntry(normalizedTerm: String, activeGroupID: UUID?) throws -> Bool {
        try base.hasEntry(normalizedTerm: normalizedTerm, activeGroupID: activeGroupID)
    }
}

private enum TestRepositoryError: Error {
    case readFailed
}
