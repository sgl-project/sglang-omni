// SQLiteStorageRepositoryTests.swift
// Provides SQLite Storage Repository Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class SQLiteStorageRepositoryTests: XCTestCase {
    private static var retainedObjects: [AnyObject] = []
    private var temporaryURLs: [URL] = []

    override func tearDownWithError() throws {
        // Keep temporary databases alive until the XCTest process exits. GRDB owns
        // SQLite resources that can outlive individual test methods under the
        // hosted macOS test runner.
        temporaryURLs = []
        try super.tearDownWithError()
    }

    func testFreshDatabaseCreatesStorageTablesAndIndexes() throws {
        let database = try makeDatabase()

        let tables = try database.debugSQLiteObjectNames(type: "table")
        XCTAssertTrue(tables.contains("dictionary_entries"))
        XCTAssertTrue(tables.contains("dictionary_categories"))
        XCTAssertTrue(tables.contains("dictionary_replacement_terms"))
        XCTAssertTrue(tables.contains("dictionary_observed_variants"))
        XCTAssertTrue(tables.contains("history_entries"))
        XCTAssertTrue(tables.contains("dictionary_search"))
        XCTAssertTrue(tables.contains("history_search"))

        let indexes = try database.debugSQLiteObjectNames(type: "index")
        XCTAssertTrue(indexes.contains("idx_dictionary_category_order"))
        XCTAssertTrue(indexes.contains("idx_dictionary_entries_category"))
        XCTAssertTrue(indexes.contains("idx_dictionary_normalized_scope"))
        XCTAssertTrue(indexes.contains("idx_dictionary_active_scope_rank"))
        XCTAssertTrue(indexes.contains("idx_history_kind_created"))
        XCTAssertTrue(indexes.contains("idx_history_browser_host"))
    }

    func testDictionaryJSONMigrationPreservesEntryDetailsAndBacksUpLegacyFile() throws {
        let database = try makeDatabase()
        let legacyURL = try makeTemporaryDirectory().appendingPathComponent("dictionary.json")
        let groupID = UUID()
        let entry = DictionaryEntry(
            term: "Voxt Term",
            normalizedTerm: "voxt term",
            groupID: groupID,
            groupNameSnapshot: "Focused Group",
            source: .auto,
            createdAt: Date(timeIntervalSince1970: 10),
            updatedAt: Date(timeIntervalSince1970: 20),
            lastMatchedAt: Date(timeIntervalSince1970: 30),
            matchCount: 7,
            observedVariants: [
                ObservedVariant(
                    text: "voxt variant",
                    normalizedText: "voxt variant",
                    count: 3,
                    lastSeenAt: Date(timeIntervalSince1970: 40),
                    confidence: .high
                )
            ],
            replacementTerms: [
                DictionaryReplacementTerm(text: "Alias Term", normalizedText: "alias term")
            ]
        )
        try JSONEncoder().encode([entry]).write(to: legacyURL)

        let repository = retain(DictionaryRepository(database: database, legacyJSONURL: legacyURL))
        let migratedEntries = try repository.allEntries()

        XCTAssertEqual(migratedEntries.count, 1)
        XCTAssertEqual(migratedEntries[0].term, "Voxt Term")
        XCTAssertEqual(migratedEntries[0].categoryID, groupID)
        XCTAssertEqual(migratedEntries[0].categoryNameSnapshot, "Focused Group")
        XCTAssertEqual(migratedEntries[0].groupID, groupID)
        XCTAssertEqual(migratedEntries[0].groupNameSnapshot, "Focused Group")
        XCTAssertEqual(migratedEntries[0].source, .auto)
        XCTAssertEqual(migratedEntries[0].lastMatchedAt, Date(timeIntervalSince1970: 30))
        XCTAssertEqual(migratedEntries[0].matchCount, 7)
        XCTAssertEqual(migratedEntries[0].replacementTerms.map(\.text), ["Alias Term"])
        XCTAssertEqual(migratedEntries[0].observedVariants.map(\.text), ["voxt variant"])
        let migratedCategories = try repository.allCategories()
        XCTAssertTrue(migratedCategories.contains {
            $0.id == DictionaryCategory.defaultID && $0.isDefault
        })
        XCTAssertTrue(migratedCategories.contains {
            $0.id == groupID && $0.name == "Focused Group" && !$0.isDefault
        })
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: legacyURL.path + ".migrated-backup"))
    }

    func testDictionaryCategoryPersistenceAndDeletionPolicies() throws {
        let database = try makeDatabase()
        let repository = retain(DictionaryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let namesCategory = DictionaryCategory(
            name: "ArchivedBucket",
            normalizedName: "archivedbucket",
            sortOrder: 1
        )
        let projectsCategory = DictionaryCategory(
            name: "ProjectsBucket",
            normalizedName: "projectsbucket",
            sortOrder: 2
        )
        let nameEntry = DictionaryEntry(
            term: "Custom Term",
            normalizedTerm: "custom term",
            categoryID: namesCategory.id,
            categoryNameSnapshot: namesCategory.name,
            source: .manual
        )
        let projectEntry = DictionaryEntry(
            term: "Project Term",
            normalizedTerm: "project term",
            categoryID: projectsCategory.id,
            categoryNameSnapshot: projectsCategory.name,
            source: .manual
        )

        try repository.upsertCategory(namesCategory)
        try repository.upsertCategory(projectsCategory)
        try repository.replaceAll([nameEntry, projectEntry])

        XCTAssertEqual(try repository.allCategories().map(\.name), ["Default", "ArchivedBucket", "ProjectsBucket"])
        XCTAssertEqual(
            try repository.allEntries().first { $0.id == nameEntry.id }?.categoryNameSnapshot,
            "ArchivedBucket"
        )
        XCTAssertEqual(try repository.entryCount(query: "ArchivedBucket"), 1)

        try repository.deleteCategory(id: namesCategory.id, moveEntriesTo: DictionaryCategory.defaultCategory)
        let movedEntry = try XCTUnwrap(repository.allEntries().first { $0.id == nameEntry.id })
        XCTAssertEqual(movedEntry.categoryID, DictionaryCategory.defaultID)
        XCTAssertEqual(movedEntry.categoryNameSnapshot, DictionaryCategory.defaultName)
        XCTAssertFalse(try repository.allCategories().contains { $0.id == namesCategory.id })
        XCTAssertEqual(try repository.entryCount(query: "ArchivedBucket"), 0)
        XCTAssertEqual(try repository.entries(filter: .all, query: "Default", limit: 10, offset: 0).map(\.id), [nameEntry.id])

        try repository.deleteCategory(id: projectsCategory.id, moveEntriesTo: nil)
        XCTAssertFalse(try repository.allEntries().contains { $0.id == projectEntry.id })
        XCTAssertFalse(try repository.allCategories().contains { $0.id == projectsCategory.id })
    }

    func testDictionaryDuplicateChecksUseDatabaseIndexes() throws {
        let database = try makeDatabase()
        let repository = retain(DictionaryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let groupID = UUID()
        let entry = DictionaryEntry(
            term: "Primary",
            normalizedTerm: "primary",
            groupID: groupID,
            groupNameSnapshot: "Group",
            source: .manual,
            replacementTerms: [
                DictionaryReplacementTerm(text: "Alias", normalizedText: "alias")
            ]
        )

        try repository.replaceAll([entry])

        XCTAssertTrue(try repository.hasEntry(normalizedTerm: "primary", activeGroupID: groupID))
        XCTAssertTrue(try repository.hasEntry(normalizedTerm: "alias", activeGroupID: groupID))
        XCTAssertFalse(try repository.hasEntry(normalizedTerm: "alias", activeGroupID: nil))
    }

    func testDictionaryUpsertDoesNotReplaceExistingEntryOnUniqueTermConflict() throws {
        let database = try makeDatabase()
        let repository = retain(DictionaryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let groupID = UUID()
        let original = DictionaryEntry(
            term: "Primary",
            normalizedTerm: "primary",
            groupID: groupID,
            groupNameSnapshot: "Group",
            source: .manual,
            observedVariants: [
                ObservedVariant(
                    text: "Observed Primary",
                    normalizedText: "observed primary",
                    count: 4,
                    lastSeenAt: Date(timeIntervalSince1970: 40),
                    confidence: .high
                )
            ],
            replacementTerms: [
                DictionaryReplacementTerm(text: "Alias", normalizedText: "alias")
            ]
        )
        let duplicate = DictionaryEntry(
            term: "Duplicate Primary",
            normalizedTerm: "primary",
            groupID: groupID,
            groupNameSnapshot: "Group",
            source: .auto
        )

        try repository.replaceAll([original])
        XCTAssertThrowsError(try repository.upsert(duplicate))

        let persisted = try XCTUnwrap(repository.allEntries().first { $0.id == original.id })
        XCTAssertEqual(persisted.term, "Primary")
        XCTAssertEqual(persisted.replacementTerms.map(\.text), ["Alias"])
        XCTAssertEqual(persisted.observedVariants.map(\.text), ["Observed Primary"])
        XCTAssertFalse(try repository.allEntries().contains { $0.id == duplicate.id })
    }

    func testDictionaryUpsertDoesNotReplaceExistingAliasOnUniqueAliasConflict() throws {
        let database = try makeDatabase()
        let repository = retain(DictionaryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let groupID = UUID()
        let first = DictionaryEntry(
            term: "First",
            normalizedTerm: "first",
            groupID: groupID,
            source: .manual,
            replacementTerms: [
                DictionaryReplacementTerm(text: "Shared Alias", normalizedText: "shared alias")
            ]
        )
        var second = DictionaryEntry(
            term: "Second",
            normalizedTerm: "second",
            groupID: groupID,
            source: .manual
        )

        try repository.replaceAll([first, second])
        second.replacementTerms = [
            DictionaryReplacementTerm(text: "Shared Alias", normalizedText: "shared alias")
        ]
        XCTAssertThrowsError(try repository.upsert(second))

        let persistedEntries = try repository.allEntries()
        let persistedFirst = try XCTUnwrap(persistedEntries.first { $0.id == first.id })
        let persistedSecond = try XCTUnwrap(persistedEntries.first { $0.id == second.id })
        XCTAssertEqual(persistedFirst.replacementTerms.map(\.text), ["Shared Alias"])
        XCTAssertEqual(persistedSecond.replacementTerms, [])
    }

    func testHistoryJSONMigrationPreservesTranscriptEntriesAndBacksUpLegacyFile() throws {
        let database = try makeDatabase()
        let legacyURL = try makeTemporaryDirectory().appendingPathComponent("transcription-history.json")
        let normalEntry = makeHistoryEntry(
            text: "visible history",
            createdAt: Date(timeIntervalSince1970: 1),
            kind: .normal
        )
        let transcriptEntry = makeHistoryEntry(
            text: "transcript history",
            createdAt: Date(timeIntervalSince1970: 2),
            kind: .transcript
        )
        try JSONEncoder().encode([normalEntry, transcriptEntry]).write(to: legacyURL)

        let repository = retain(HistoryRepository(database: database, legacyJSONURL: legacyURL))

        XCTAssertEqual(try repository.entryCount(kind: nil), 2)
        XCTAssertEqual(try repository.entryCount(kind: .normal), 1)
        XCTAssertEqual(try repository.entryCount(kind: .transcript), 1)
        XCTAssertEqual(try repository.entry(id: transcriptEntry.id)?.text, "transcript history")
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: legacyURL.path + ".migrated-backup"))
    }

    func testHistoryRepositoryPaginatesSearchesAndDeletes() throws {
        let database = try makeDatabase()
        let repository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let oldID = UUID()
        let entries = [
            makeHistoryEntry(
                id: oldID,
                text: "old history",
                createdAt: Date(timeIntervalSince1970: 1),
                kind: .normal
            ),
            makeHistoryEntry(
                text: "translation history",
                createdAt: Date(timeIntervalSince1970: 2),
                kind: .translation
            ),
            makeHistoryEntry(
                text: "needle focused app entry",
                createdAt: Date(timeIntervalSince1970: 3),
                kind: .normal,
                focusedAppName: "Safari",
                dictionaryHitTerms: ["NeedleTerm"]
            )
        ]

        try repository.replaceAll(entries)

        XCTAssertEqual(try repository.entryCount(kind: .normal), 2)
        XCTAssertEqual(
            try repository.entries(kind: .normal, limit: 1, offset: 0).map(\.text),
            ["needle focused app entry"]
        )
        XCTAssertEqual(
            try repository.entries(kind: nil, query: "NeedleTerm", limit: 10, offset: 0).map(\.text),
            ["needle focused app entry"]
        )
        XCTAssertEqual(
            try repository.entries(kind: nil, query: "Safari", limit: 10, offset: 0).map(\.text),
            ["needle focused app entry"]
        )

        let removed = try repository.deleteEntries(
            olderThan: Date(timeIntervalSince1970: 2),
            kinds: [.normal, .translation, .rewrite]
        )
        XCTAssertEqual(removed.map(\.id), [oldID])
        XCTAssertNil(try repository.entry(id: oldID))
    }

    func testHistoryListEntriesBoundLargeMeetingPreviewButKeepFullDetailAndSearch() throws {
        let database = try makeDatabase()
        let repository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let text = String(repeating: "meeting transcript line\n", count: 5_000) + "unique-tail-token"
        let meeting = makeHistoryEntry(
            text: text,
            createdAt: Date(),
            kind: .transcript
        )

        try repository.upsert(meeting)

        let listEntries = try repository.listEntries(
            kind: .transcript,
            query: "unique-tail-token",
            limit: 10,
            offset: 0
        )

        XCTAssertEqual(listEntries.count, 1)
        XCTAssertEqual(listEntries[0].id, meeting.id)
        XCTAssertLessThanOrEqual(
            listEntries[0].previewText.count,
            TranscriptionHistoryListEntry.previewCharacterLimit
        )
        XCTAssertEqual(listEntries[0].textLength, text.count)
        XCTAssertNotEqual(listEntries[0].previewText, text)
        XCTAssertEqual(try repository.entry(id: meeting.id)?.text, text)
    }

    func testHistoryRepositoryDeletesOnlyRequestedKind() throws {
        let database = try makeDatabase()
        let repository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let transcription = makeHistoryEntry(text: "transcription", createdAt: Date(), kind: .normal)
        let translation = makeHistoryEntry(text: "translation", createdAt: Date(), kind: .translation)
        let rewrite = makeHistoryEntry(text: "rewrite", createdAt: Date(), kind: .rewrite)
        let meeting = makeHistoryEntry(text: "meeting", createdAt: Date(), kind: .transcript)
        try repository.replaceAll([transcription, translation, rewrite, meeting])

        let removed = try repository.deleteEntries(kind: .translation)

        XCTAssertEqual(removed.map(\.id), [translation.id])
        XCTAssertNil(try repository.entry(id: translation.id))
        XCTAssertNotNil(try repository.entry(id: transcription.id))
        XCTAssertNotNil(try repository.entry(id: rewrite.id))
        XCTAssertNotNil(try repository.entry(id: meeting.id))
    }

    func testHistoryCleanupPreservesMeetings() throws {
        let database = try makeDatabase()
        let repository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let oldDate = Date(timeIntervalSince1970: 1)
        let newDate = Date(timeIntervalSince1970: 10)
        let oldTranscription = makeHistoryEntry(text: "old transcription", createdAt: oldDate, kind: .normal)
        let oldTranslation = makeHistoryEntry(text: "old translation", createdAt: oldDate, kind: .translation)
        let oldRewrite = makeHistoryEntry(text: "old rewrite", createdAt: oldDate, kind: .rewrite)
        let oldMeeting = makeHistoryEntry(text: "old meeting", createdAt: oldDate, kind: .transcript)
        let newTranscription = makeHistoryEntry(text: "new transcription", createdAt: newDate, kind: .normal)
        try repository.replaceAll([
            oldTranscription,
            oldTranslation,
            oldRewrite,
            oldMeeting,
            newTranscription
        ])

        let removed = try repository.deleteEntries(
            olderThan: Date(timeIntervalSince1970: 5),
            kinds: [.normal, .translation, .rewrite]
        )

        XCTAssertEqual(Set(removed.map(\.id)), Set([oldTranscription.id, oldTranslation.id, oldRewrite.id]))
        XCTAssertNotNil(try repository.entry(id: oldMeeting.id))
        XCTAssertNotNil(try repository.entry(id: newTranscription.id))
    }

    func testHistoryCountCleanupKeepsNewestPerKindAndPreservesMeetings() throws {
        let database = try makeDatabase()
        let repository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let newest = makeHistoryEntry(text: "normal-2", createdAt: Date(timeIntervalSince1970: 3), kind: .normal)
        let middle = makeHistoryEntry(text: "normal-1", createdAt: Date(timeIntervalSince1970: 2), kind: .normal)
        let oldest = makeHistoryEntry(text: "normal-0", createdAt: Date(timeIntervalSince1970: 1), kind: .normal)
        let translationNewest = makeHistoryEntry(text: "translation-1", createdAt: Date(timeIntervalSince1970: 5), kind: .translation)
        let translationOldest = makeHistoryEntry(text: "translation-0", createdAt: Date(timeIntervalSince1970: 4), kind: .translation)
        let meeting = makeHistoryEntry(text: "meeting", createdAt: Date(timeIntervalSince1970: 0), kind: .transcript)
        try repository.replaceAll([
            newest,
            middle,
            oldest,
            translationNewest,
            translationOldest,
            meeting
        ])

        let removedNormal = try repository.deleteEntries(keepingNewest: 2, kind: .normal)
        let removedTranslation = try repository.deleteEntries(keepingNewest: 1, kind: .translation)

        XCTAssertEqual(Set(removedNormal.map(\.id)), Set([oldest.id]))
        XCTAssertEqual(Set(removedTranslation.map(\.id)), Set([translationOldest.id]))
        XCTAssertNotNil(try repository.entry(id: newest.id))
        XCTAssertNotNil(try repository.entry(id: middle.id))
        XCTAssertNotNil(try repository.entry(id: translationNewest.id))
        XCTAssertNotNil(try repository.entry(id: meeting.id))
    }

    func testHistoryReportMetricsUseDatabaseAggregates() throws {
        let database = try makeDatabase()
        let repository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let yesterday = calendar.date(byAdding: .day, value: -1, to: today)!
        let dayStarts = [yesterday, today]

        try repository.replaceAll([
            makeHistoryEntry(
                text: "abcd",
                createdAt: today.addingTimeInterval(60),
                kind: .normal,
                audioDurationSeconds: 10
            ),
            makeHistoryEntry(
                text: "translation",
                createdAt: yesterday.addingTimeInterval(60),
                kind: .translation,
                audioDurationSeconds: 20
            )
        ])

        let metrics = try repository.reportMetrics(dayStarts: dayStarts)

        XCTAssertEqual(metrics.totalDictationSeconds, 30)
        XCTAssertEqual(metrics.totalCharacters, 15)
        XCTAssertEqual(metrics.totalTranslationCharacters, 11)
        XCTAssertEqual(metrics.dailyCharacters[today], 4)
        XCTAssertEqual(metrics.dailyCharacters[yesterday], 11)
        XCTAssertTrue(metrics.branchItems.isEmpty)
    }

    func testHistoryReportMetricsAggregateBranchItemsWithinDateRange() throws {
        let database = try makeDatabase()
        let repository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let yesterday = calendar.date(byAdding: .day, value: -1, to: today)!
        let dayStarts = [yesterday, today]

        try repository.replaceAll([
            makeHistoryEntry(
                text: "app chars",
                createdAt: today.addingTimeInterval(60),
                kind: .normal,
                focusedAppName: "Pages",
                focusedAppBundleID: "com.apple.Pages"
            ),
            makeHistoryEntry(
                text: "browser words",
                createdAt: today.addingTimeInterval(120),
                kind: .normal,
                focusedAppName: "Safari",
                focusedAppBundleID: "com.apple.Safari",
                browserURLHost: "example.com",
                browserURLOrigin: "https://example.com"
            ),
            makeHistoryEntry(
                text: "old url",
                createdAt: yesterday.addingTimeInterval(120),
                kind: .normal,
                focusedAppName: "Safari",
                focusedAppBundleID: "com.apple.Safari",
                browserURLHost: "old.example",
                browserURLOrigin: "https://old.example"
            ),
            makeHistoryEntry(
                text: "meeting words",
                createdAt: today.addingTimeInterval(180),
                kind: .transcript,
                focusedAppName: "Pages",
                focusedAppBundleID: "com.apple.Pages",
                browserURLHost: "meeting.example",
                browserURLOrigin: "https://meeting.example"
            )
        ])

        let metrics = try repository.reportMetrics(dayStarts: dayStarts, branchStartDate: today)

        XCTAssertEqual(metrics.branchItems.count, 3)
        XCTAssertTrue(metrics.branchItems.contains {
            $0.kind == .app &&
                $0.title == "Safari" &&
                $0.bundleID == "com.apple.Safari" &&
                $0.characterCount == "browser words".count
        })
        XCTAssertTrue(metrics.branchItems.contains {
            $0.kind == .app &&
                $0.title == "Pages" &&
                $0.bundleID == "com.apple.Pages" &&
                $0.characterCount == "app chars".count
        })
        XCTAssertTrue(metrics.branchItems.contains {
            $0.kind == .url &&
                $0.title == "example.com" &&
                $0.urlOrigin == "https://example.com" &&
                $0.characterCount == "browser words".count
        })
        XCTAssertFalse(metrics.branchItems.contains { $0.title == "old.example" })
        XCTAssertFalse(metrics.branchItems.contains { $0.title == "meeting.example" })
    }

    func testTranscriptionHistoryEntryDecodesLegacyJSONWithoutBrowserContext() throws {
        let entry = makeHistoryEntry(
            text: "legacy",
            createdAt: Date(timeIntervalSince1970: 1),
            kind: .normal
        )
        var json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(entry)) as! [String: Any]
        json.removeValue(forKey: "browserURLHost")
        json.removeValue(forKey: "browserURLOrigin")
        let data = try JSONSerialization.data(withJSONObject: json)

        let decoded = try JSONDecoder().decode(TranscriptionHistoryEntry.self, from: data)

        XCTAssertNil(decoded.browserURLHost)
        XCTAssertNil(decoded.browserURLOrigin)
    }

    func testHistoryEntryOutputDestinationCanBeUpdatedAfterDelayedInjection() throws {
        let database = try makeDatabase()
        let repository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let entry = makeHistoryEntry(
            text: "delayed answer",
            createdAt: Date(),
            kind: .rewrite
        )
        try repository.replaceAll([entry])
        let store = retain(TranscriptionHistoryStore(repository: repository))

        let updated = store.updateOutputDestination(
            for: entry.id,
            focusedAppName: "Safari",
            focusedAppBundleID: "com.apple.Safari",
            browserURLHost: "example.com",
            browserURLOrigin: "https://example.com"
        )

        XCTAssertEqual(updated?.focusedAppName, "Safari")
        XCTAssertEqual(updated?.focusedAppBundleID, "com.apple.Safari")
        XCTAssertEqual(updated?.browserURLHost, "example.com")
        XCTAssertEqual(updated?.browserURLOrigin, "https://example.com")
        XCTAssertEqual(updated?.text, entry.text)
        XCTAssertEqual(updated?.kind, entry.kind)

        let persisted = try XCTUnwrap(repository.entry(id: entry.id))
        XCTAssertEqual(persisted.focusedAppBundleID, "com.apple.Safari")
        XCTAssertEqual(persisted.browserURLHost, "example.com")

        let metrics = try repository.reportMetrics(dayStarts: [])
        XCTAssertTrue(metrics.branchItems.contains {
            $0.kind == .app &&
                $0.bundleID == "com.apple.Safari" &&
                $0.characterCount == entry.text.count
        })
        XCTAssertTrue(metrics.branchItems.contains {
            $0.kind == .url &&
                $0.urlHost == "example.com" &&
                $0.characterCount == entry.text.count
        })
    }

    func testRepositoryHandlesLargeBatchedDictionaryAndHistoryData() throws {
        let database = try makeDatabase()
        let dictionaryRepository = retain(DictionaryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let historyRepository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))

        let dictionaryEntries = (0..<20_000).map {
            DictionaryEntry(
                term: "Term\($0)",
                normalizedTerm: "term\($0)",
                source: .manual,
                replacementTerms: [
                    DictionaryReplacementTerm(text: "Alias\($0)", normalizedText: "alias\($0)")
                ]
            )
        }
        try dictionaryRepository.replaceAll(dictionaryEntries)
        XCTAssertEqual(try dictionaryRepository.entryCount(query: "Alias19999"), 1)
        XCTAssertEqual(
            try dictionaryRepository.activeEntriesForRemoteRequest(activeGroupID: nil, limit: 64).count,
            64
        )
        XCTAssertEqual(
            try dictionaryRepository.matchingEntries(
                sourceText: "Please use Alias19999 today.",
                activeGroupID: nil,
                limit: 20
            ).map(\.term),
            ["Term19999"]
        )
        XCTAssertTrue(
            try dictionaryRepository.matchingEntries(
                sourceText: "No seeded term appears here.",
                activeGroupID: nil,
                limit: 20
            ).isEmpty
        )

        let historyEntries = (0..<20_000).map {
            makeHistoryEntry(
                text: $0 == 19_999 ? "needle history row" : "history row \($0)",
                createdAt: Date(timeIntervalSince1970: TimeInterval($0)),
                kind: $0.isMultiple(of: 2) ? .normal : .translation
            )
        }
        try historyRepository.replaceAll(historyEntries)

        XCTAssertEqual(try historyRepository.entryCount(kind: .normal), 10_000)
        XCTAssertEqual(
            try historyRepository.entries(kind: nil, query: "needle", limit: 10, offset: 0).map(\.text),
            ["needle history row"]
        )
        XCTAssertEqual(try historyRepository.entries(kind: .translation, limit: 25, offset: 50).count, 25)
    }
}

private extension SQLiteStorageRepositoryTests {
    func makeDatabase() throws -> VoxtDatabase {
        let directory = try makeTemporaryDirectory()
        let database = VoxtDatabase(databaseURL: directory.appendingPathComponent("voxt.sqlite"))
        return retain(database)
    }

    func retain<Value: AnyObject>(_ value: Value) -> Value {
        Self.retainedObjects.append(value)
        return value
    }

    func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-storage-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        temporaryURLs.append(url)
        return url
    }

    func makeHistoryEntry(
        id: UUID = UUID(),
        text: String,
        createdAt: Date,
        kind: TranscriptionHistoryKind,
        audioDurationSeconds: TimeInterval? = nil,
        focusedAppName: String? = nil,
        focusedAppBundleID: String? = nil,
        browserURLHost: String? = nil,
        browserURLOrigin: String? = nil,
        dictionaryHitTerms: [String] = []
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
            llmDurationSeconds: nil,
            focusedAppName: focusedAppName,
            focusedAppBundleID: focusedAppBundleID ?? focusedAppName.map { "com.example.\($0.lowercased())" },
            browserURLHost: browserURLHost,
            browserURLOrigin: browserURLOrigin,
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
            dictionaryHitTerms: dictionaryHitTerms,
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        )
    }
}
