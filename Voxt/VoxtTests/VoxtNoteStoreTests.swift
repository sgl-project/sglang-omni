// VoxtNoteStoreTests.swift
// Provides Voxt Note Store Tests for Voxt test coverage.

import XCTest
import SwiftData
@testable import Voxt

@MainActor
final class VoxtNoteStoreTests: XCTestCase {
    private struct StorageOpenError: LocalizedError {
        var errorDescription: String? { "Injected note storage open failure" }
    }

    func testContainerInitializationFailureDoesNotCrashAndCanRetry() throws {
        var shouldFail = true
        let store = VoxtNoteStore(
            inMemory: true,
            containerFactory: { schema, configuration in
                if shouldFail { throw StorageOpenError() }
                return try ModelContainer(for: schema, configurations: [configuration])
            }
        )

        XCTAssertFalse(store.isAvailable)
        XCTAssertEqual(store.availability.errorMessage, "Injected note storage open failure")
        XCTAssertTrue(store.items.isEmpty)
        XCTAssertNil(store.append(
            sessionID: UUID(),
            text: "Must not be accepted while storage is unavailable.",
            title: "Unavailable",
            titleGenerationState: .fallback
        ))

        shouldFail = false
        XCTAssertTrue(store.retryOpeningStorage())
        XCTAssertTrue(store.isAvailable)
        XCTAssertNotNil(store.append(
            sessionID: UUID(),
            text: "Accepted after retry.",
            title: "Recovered",
            titleGenerationState: .fallback
        ))
    }

    func testArchiveAndRebuildPreservesPreviousDatabase() throws {
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let legacyURL = directoryURL.appendingPathComponent("voxt-notes.json")
        let store = VoxtNoteStore(fileURL: legacyURL)
        let original = try XCTUnwrap(store.append(
            sessionID: UUID(),
            text: "Preserve this note in the archived database.",
            title: "Preserved note",
            titleGenerationState: .fallback
        ))

        XCTAssertTrue(store.archiveAndRebuildStorage())
        XCTAssertTrue(store.isAvailable)
        XCTAssertTrue(store.items.isEmpty)

        let archiveURL = try XCTUnwrap(store.lastRecoveryArchiveURL)
        let archivedStoreURL = archiveURL.appendingPathComponent("voxt-notes.store")
        XCTAssertTrue(FileManager.default.fileExists(atPath: archivedStoreURL.path))

        let archivedStore = VoxtNoteStore(
            fileURL: archiveURL.appendingPathComponent("legacy.json"),
            storeURL: archivedStoreURL
        )
        XCTAssertTrue(archivedStore.isAvailable)
        XCTAssertEqual(archivedStore.items.map(\.id), [original.id])
        XCTAssertEqual(archivedStore.items.first?.text, original.text)
    }

    func testAppendUpdateDeleteAndReloadPersistedNotes() async throws {
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let fileURL = directoryURL.appendingPathComponent("voxt-notes.json")

        let sessionID = UUID()
        let store = VoxtNoteStore(fileURL: fileURL)
        let item = try XCTUnwrap(
            store.append(
                sessionID: sessionID,
                text: "Schedule design review for Friday afternoon.",
                title: "Schedule design review",
                titleGenerationState: .pending
            )
        )

        XCTAssertEqual(store.items.count, 1)
        XCTAssertEqual(store.items.first?.titleGenerationState, .pending)

        _ = store.updateTitle("Friday design review", state: .generated, for: item.id)
        XCTAssertEqual(store.items.first?.title, "Friday design review")
        XCTAssertEqual(store.items.first?.titleGenerationState, .generated)

        try await Task.sleep(for: .milliseconds(450))

        let reloadedStore = VoxtNoteStore(fileURL: fileURL)
        XCTAssertEqual(reloadedStore.items.count, 1)
        XCTAssertEqual(reloadedStore.items.first?.title, "Friday design review")
        XCTAssertEqual(reloadedStore.items.first?.sessionID, sessionID)
        XCTAssertFalse(reloadedStore.items.first?.isCompleted ?? true)

        _ = reloadedStore.updateCompletion(true, for: item.id)
        XCTAssertTrue(reloadedStore.items.first?.isCompleted ?? false)
        XCTAssertTrue(reloadedStore.incompleteItems.isEmpty)

        reloadedStore.delete(id: item.id)
        XCTAssertTrue(reloadedStore.items.isEmpty)
    }

    func testSelectedTextSourcePersists() throws {
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let fileURL = directoryURL.appendingPathComponent("voxt-notes.json")
        let store = VoxtNoteStore(fileURL: fileURL)

        let item = try XCTUnwrap(store.append(
            sessionID: UUID(),
            text: "Selected text",
            title: "Selected text",
            titleGenerationState: .fallback,
            source: .selection
        ))

        XCTAssertEqual(item.source, .selection)
        XCTAssertEqual(VoxtNoteStore(fileURL: fileURL).items.first?.source, .selection)
    }

    func testConditionalTextUpdateDoesNotOverwriteUserEdit() throws {
        let store = VoxtNoteStore(inMemory: true)
        let item = try XCTUnwrap(store.append(
            sessionID: UUID(),
            text: "Raw transcription",
            title: "Raw transcription",
            titleGenerationState: .pending
        ))

        let enhanced = try XCTUnwrap(store.updateText(
            "Enhanced transcription",
            ifUnchangedFrom: "Raw transcription",
            for: item.id
        ))
        XCTAssertEqual(enhanced.text, "Enhanced transcription")

        XCTAssertTrue(store.updateDetails(
            item.id,
            title: enhanced.title,
            text: "User-edited note"
        ))
        let preserved = try XCTUnwrap(store.updateText(
            "Late enhancement result",
            ifUnchangedFrom: "Enhanced transcription",
            for: item.id
        ))
        XCTAssertEqual(preserved.text, "User-edited note")
    }

    func testLegacyJSONMigrationPreservesContentAndArchivesSource() throws {
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let legacyURL = directoryURL.appendingPathComponent("voxt-notes.json")
        let sessionID = UUID()
        let incompleteID = UUID()
        let completedID = UUID()
        let createdAt = Date(timeIntervalSince1970: 1_000)
        let legacyPayload: [[String: Any]] = [
            [
                "id": incompleteID.uuidString,
                "sessionID": sessionID.uuidString,
                "createdAt": createdAt.timeIntervalSinceReferenceDate,
                "text": "Keep the original transcript.",
                "title": "Original transcript",
                "titleGenerationState": "generated",
                "isCompleted": false
            ],
            [
                "id": completedID.uuidString,
                "sessionID": sessionID.uuidString,
                "createdAt": createdAt.addingTimeInterval(1).timeIntervalSinceReferenceDate,
                "text": "Already completed.",
                "title": "Completed note",
                "titleGenerationState": "fallback",
                "isCompleted": true
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: legacyPayload)
        try data.write(to: legacyURL)

        let store = VoxtNoteStore(fileURL: legacyURL)

        XCTAssertEqual(store.items.count, 2)
        XCTAssertEqual(store.items.first(where: { $0.id == incompleteID })?.status, .todo)
        XCTAssertEqual(store.items.first(where: { $0.id == completedID })?.status, .done)
        XCTAssertEqual(store.items.first(where: { $0.id == incompleteID })?.source, .migrated)
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyURL.path))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: legacyURL.appendingPathExtension("migrated-v1.bak").path
        ))

        let reloaded = VoxtNoteStore(fileURL: legacyURL)
        XCTAssertEqual(Set(reloaded.items.map(\.id)), Set([incompleteID, completedID]))
    }

    func testStatusPriorityAndReorderingPersist() throws {
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let legacyURL = directoryURL.appendingPathComponent("voxt-notes.json")
        let store = VoxtNoteStore(fileURL: legacyURL)
        let first = try XCTUnwrap(store.append(
            sessionID: UUID(),
            text: "First body",
            title: "First",
            titleGenerationState: .fallback
        ))
        let second = try XCTUnwrap(store.append(
            sessionID: UUID(),
            text: "Second body",
            title: "Second",
            titleGenerationState: .fallback
        ))

        XCTAssertTrue(store.setPriority(.high, for: first.id))
        XCTAssertTrue(store.setPriority(.high, for: second.id))
        XCTAssertTrue(store.setStatus(.inProgress, for: first.id))
        XCTAssertTrue(store.setStatus(.inProgress, for: second.id))
        XCTAssertTrue(store.reorder(noteID: first.id, relativeTo: second.id))
        XCTAssertEqual(store.orderedItems(for: .inProgress).map(\.id), [first.id, second.id])

        let reloaded = VoxtNoteStore(fileURL: legacyURL)
        XCTAssertEqual(reloaded.orderedItems(for: .inProgress).map(\.id), [first.id, second.id])
        XCTAssertEqual(reloaded.items.first(where: { $0.id == first.id })?.priority, .high)
    }

    func testCompletedNotesAreOrderedByMostRecentCompletion() throws {
        var currentDate = Date(timeIntervalSince1970: 1_000)
        let store = VoxtNoteStore(inMemory: true, now: { currentDate })
        let first = try XCTUnwrap(store.append(
            sessionID: UUID(),
            text: "First completed note",
            title: "First",
            titleGenerationState: .fallback
        ))
        currentDate = Date(timeIntervalSince1970: 2_000)
        XCTAssertTrue(store.setStatus(.done, for: first.id))

        currentDate = Date(timeIntervalSince1970: 3_000)
        let second = try XCTUnwrap(store.append(
            sessionID: UUID(),
            text: "Second completed note",
            title: "Second",
            titleGenerationState: .fallback
        ))
        XCTAssertTrue(store.setPriority(.high, for: first.id))
        currentDate = Date(timeIntervalSince1970: 4_000)
        XCTAssertTrue(store.setStatus(.done, for: second.id))

        XCTAssertEqual(store.orderedItems(for: .done).map(\.id), [second.id, first.id])
    }

    func testDoubleClickActionUsesTheSameStatusTransitionsAsTheFloatPanel() throws {
        let store = VoxtNoteStore(inMemory: true)
        let item = try XCTUnwrap(store.append(
            sessionID: UUID(),
            text: "Shared double-click behavior",
            title: "Double click",
            titleGenerationState: .fallback
        ))

        XCTAssertTrue(store.performDoubleClickAction(for: item.id))
        XCTAssertEqual(store.items.first(where: { $0.id == item.id })?.status, .inProgress)

        XCTAssertTrue(store.performDoubleClickAction(for: item.id))
        XCTAssertEqual(store.items.first(where: { $0.id == item.id })?.status, .todo)

        XCTAssertTrue(store.setStatus(.backlog, for: item.id))
        XCTAssertTrue(store.performDoubleClickAction(for: item.id))
        XCTAssertEqual(store.items.first(where: { $0.id == item.id })?.status, .todo)

        XCTAssertTrue(store.setStatus(.done, for: item.id))
        XCTAssertFalse(store.performDoubleClickAction(for: item.id))
        XCTAssertEqual(store.items.first(where: { $0.id == item.id })?.status, .done)
    }

    func testGeneratedTitleDoesNotOverwriteUserRename() throws {
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let store = VoxtNoteStore(fileURL: directoryURL.appendingPathComponent("voxt-notes.json"))
        let item = try XCTUnwrap(store.append(
            sessionID: UUID(),
            text: "Body",
            title: "Pending title",
            titleGenerationState: .pending
        ))

        XCTAssertTrue(store.rename(item.id, to: "My title"))
        _ = store.updateTitle("Late AI title", state: .generated, for: item.id)

        XCTAssertEqual(store.items.first?.title, "My title")
        XCTAssertEqual(store.items.first?.titleGenerationState, .userEdited)
    }

    func testUpdateDetailsSavesTitleAndContentAtomically() throws {
        let store = VoxtNoteStore(inMemory: true)
        let item = try XCTUnwrap(store.append(
            sessionID: UUID(),
            text: "Original content",
            title: "Original title",
            titleGenerationState: .generated
        ))

        XCTAssertTrue(store.updateDetails(
            item.id,
            title: " Updated title ",
            text: " Updated content "
        ))
        let updated = try XCTUnwrap(store.items.first(where: { $0.id == item.id }))
        XCTAssertEqual(updated.title, "Updated title")
        XCTAssertEqual(updated.text, "Updated content")
        XCTAssertEqual(updated.titleGenerationState, .userEdited)

        XCTAssertFalse(store.updateDetails(item.id, title: "", text: "Must not save"))
        let unchanged = try XCTUnwrap(store.items.first(where: { $0.id == item.id }))
        XCTAssertEqual(unchanged.title, "Updated title")
        XCTAssertEqual(unchanged.text, "Updated content")
    }

    func testQueryOrderingHonorsPriorityManualOrderTimestampAndIdentity() {
        let low = queryNote(1, priority: .low, manualOrder: 100, updatedAt: 500)
        let ordered = queryNote(2, priority: .high, manualOrder: 2, updatedAt: 100)
        let newer = queryNote(3, priority: .high, manualOrder: 1, updatedAt: 300)
        let older = queryNote(4, priority: .high, manualOrder: 1, updatedAt: 200)
        let tied = queryNote(5, priority: .high, manualOrder: 1, updatedAt: 200)
        let unordered = queryNote(6, priority: .high, updatedAt: 1_000)
        let backlog = queryNote(7, status: .backlog, priority: .high, manualOrder: 10)
        let notes = [backlog, unordered, tied, older, low, newer, ordered]

        XCTAssertEqual(VoxtNoteQueries.orderedItems(notes, for: .todo).map(\.id),
                       [ordered, newer, older, tied, unordered, low].map(\.id))
        XCTAssertEqual(VoxtNoteQueries.orderedItems(notes, for: .backlog).map(\.id), [backlog.id])
    }

    func testCompletedQueryUsesCompletionFallbackAndIdentityNotPriority() {
        let completed = queryNote(1, status: .done, priority: .low, completedAt: 300)
        let fallback = queryNote(2, status: .done, priority: .high, manualOrder: 10, updatedAt: 200)
        let tied = queryNote(3, status: .done, updatedAt: 200)
        XCTAssertEqual(VoxtNoteQueries.orderedItems([tied, fallback, completed], for: .done).map(\.id),
                       [completed.id, fallback.id, tied.id])
    }

    func testScopeQueryOmitsEmptySectionsAndCountsOnlyActiveStatuses() {
        let todo = queryNote(1)
        let completed = queryNote(2, status: .done)
        let backlog = queryNote(3, status: .backlog)
        let notes = [backlog, completed, todo]
        let snapshot = VoxtNoteQueries.snapshot(notes, for: .notes)
        XCTAssertEqual(snapshot.sections.map(\.status), [.todo, .done])
        XCTAssertEqual(snapshot.visibleCount, 2)
        XCTAssertEqual(snapshot.activeCount, 1)
        let backlogSnapshot = VoxtNoteQueries.snapshot(notes, for: .backlog)
        XCTAssertEqual(backlogSnapshot.sections.flatMap(\.items).map(\.id), [backlog.id])
        XCTAssertEqual(backlogSnapshot.activeCount, 1)
        XCTAssertEqual(VoxtNoteQueries.snapshot([], for: .notes).visibleCount, 0)
        XCTAssertTrue(VoxtNoteQueries.snapshot([], for: .backlog).sections.isEmpty)
    }

    func testPublicationQueryUsesCreationTimeRatherThanSectionOrder() {
        let first = queryNote(1, priority: .low, createdAt: 200)
        let second = queryNote(2, priority: .high, createdAt: 200, updatedAt: 500)
        let newest = queryNote(3, status: .done, createdAt: 300)
        XCTAssertEqual(VoxtNoteQueries.publicationOrder([second, first, newest]).map(\.id),
                       [newest.id, first.id, second.id])
    }

    func testRecordProjectionPreservesValuesAndUnknownRawValueFallbacks() {
        let note = queryNote(1, status: .done, priority: .high, manualOrder: .max, completedAt: 300)
        let record = VoxtNoteRecord(item: note)
        XCTAssertEqual(record.item, note)
        record.titleGenerationStateRaw = "unknown"
        record.statusRaw = "unknown"
        record.priorityRaw = "unknown"
        record.sourceRaw = "unknown"
        XCTAssertEqual(record.item.titleGenerationState, .fallback)
        XCTAssertEqual(record.item.status, .todo)
        XCTAssertEqual(record.item.priority, .none)
        XCTAssertEqual(record.item.source, .transcription)
        XCTAssertEqual(record.item.completedAt, note.completedAt)
        XCTAssertEqual(record.item.manualOrder, .max)
    }

    func testTitleNormalizationPreservesDifferentUserAndGeneratedPolicies() {
        XCTAssertEqual(VoxtNoteTitleSupport.normalizedTitle(" \"User\"\n\t title "), "\"User\" title")
        XCTAssertEqual(VoxtNoteTitleSupport.normalizedTitle(String(repeating: "x", count: 60)).count, 60)
        XCTAssertEqual(VoxtNoteTitleSupport.normalizedGeneratedTitle("\"Generated\"\nignored"), "Generated")
        XCTAssertEqual(VoxtNoteTitleSupport.normalizedGeneratedTitle(String(repeating: "x", count: 60)).count, 48)
        XCTAssertEqual(VoxtNoteTitleSupport.fallbackTitle(from: String(repeating: "中", count: 30)).count, 20)
        XCTAssertEqual(VoxtNoteTitleSupport.fallbackTitle(from: "\n\t"), "Note")
        XCTAssertEqual(VoxtNoteTitleSupport.normalizedTitle("\n\t"), "")
    }

    func testManualOrderOverflowAndReorderRemainScopedAndPersist() throws {
        let directory = try TemporaryDirectory()
        let legacyURL = directory.url.appendingPathComponent("voxt-notes.json")
        let first = queryNote(1, manualOrder: .max, updatedAt: 100)
        let second = queryNote(2, manualOrder: 4, updatedAt: 200)
        let otherPriority = queryNote(3, priority: .high, manualOrder: .max)
        try JSONEncoder().encode([first, second, otherPriority]).write(to: legacyURL)
        let timestamp = Date(timeIntervalSince1970: 900)
        let store = VoxtNoteStore(fileURL: legacyURL, now: { timestamp })
        let created = try XCTUnwrap(store.append(
            sessionID: UUID(), text: " Body ", title: " New\n title ", titleGenerationState: .fallback
        ))
        XCTAssertEqual(created.manualOrder, 3)
        XCTAssertEqual(created.text, "Body")
        XCTAssertEqual(created.title, "New title")
        XCTAssertEqual(store.items.first(where: { $0.id == first.id })?.manualOrder, 2)
        XCTAssertEqual(store.items.first(where: { $0.id == second.id })?.manualOrder, 1)
        XCTAssertEqual(store.items.first(where: { $0.id == otherPriority.id })?.manualOrder, .max)
        XCTAssertFalse(store.reorder(noteID: first.id, relativeTo: otherPriority.id))
        XCTAssertFalse(store.reorder(noteID: first.id, relativeTo: first.id))
        XCTAssertFalse(store.reorder(noteID: UUID(), relativeTo: first.id))
        XCTAssertTrue(store.reorder(noteID: created.id, relativeTo: second.id))
        XCTAssertEqual(store.orderedItems(for: .todo).map(\.id),
                       [otherPriority.id, first.id, second.id, created.id])
        XCTAssertEqual(store.items.first(where: { $0.id == first.id })?.updatedAt, first.updatedAt)
        XCTAssertEqual(store.items.first(where: { $0.id == created.id })?.updatedAt, timestamp)
        let reloaded = VoxtNoteStore(fileURL: legacyURL)
        XCTAssertEqual(reloaded.orderedItems(for: .todo).map(\.id), store.orderedItems(for: .todo).map(\.id))
    }

    func testLegacyCodableDefaultsAndExplicitStatusPrecedence() throws {
        let payload = """
        {"id":"00000000-0000-0000-0000-000000000001",
         "sessionID":"00000000-0000-0000-0000-000000000002",
         "createdAt":100,"text":"Body","title":"Title","titleGenerationState":"fallback",
         "isCompleted":true}
        """
        let legacy = try JSONDecoder().decode(VoxtNoteItem.self, from: Data(payload.utf8))
        XCTAssertEqual(legacy.status, .done)
        XCTAssertEqual(legacy.updatedAt, legacy.createdAt)
        XCTAssertNil(legacy.completedAt)
        XCTAssertNil(legacy.manualOrder)
        XCTAssertEqual(legacy.priority, .none)
        XCTAssertEqual(legacy.source, .migrated)
        let explicit = payload.replacingOccurrences(of: "\"isCompleted\":true", with: "\"isCompleted\":true,\"status\":\"backlog\"")
        XCTAssertEqual(try JSONDecoder().decode(VoxtNoteItem.self, from: Data(explicit.utf8)).status, .backlog)
        let encoded = try JSONEncoder().encode(legacy)
        XCTAssertEqual(try JSONDecoder().decode(VoxtNoteItem.self, from: encoded), legacy)
        struct LegacyCompletion: Decodable { let isCompleted: Bool }
        XCTAssertTrue(try JSONDecoder().decode(LegacyCompletion.self, from: encoded).isCompleted)
    }

    func testFailedReorderRollsBackOrderAndTimestamp() throws {
        var shouldFail = false
        var timestamp = Date(timeIntervalSince1970: 100)
        let store = VoxtNoteStore(inMemory: true, now: { timestamp }, persist: { context in
            if shouldFail { throw StorageOpenError() }
            try context.save()
        })
        let first = try XCTUnwrap(store.append(
            sessionID: UUID(), text: "First", title: "First", titleGenerationState: .fallback
        ))
        let second = try XCTUnwrap(store.append(
            sessionID: UUID(), text: "Second", title: "Second", titleGenerationState: .fallback
        ))
        XCTAssertNil(first.manualOrder)
        XCTAssertNil(second.manualOrder)
        let original = store.items
        let ordered = store.orderedItems(for: .todo)
        let movedID = try XCTUnwrap(ordered.last?.id)
        let targetID = try XCTUnwrap(ordered.first?.id)
        timestamp = Date(timeIntervalSince1970: 200)
        shouldFail = true
        XCTAssertFalse(store.reorder(noteID: movedID, relativeTo: targetID))
        XCTAssertEqual(store.items, original)
        XCTAssertEqual(store.lastErrorMessage, "Injected note storage open failure")
        shouldFail = false
        XCTAssertTrue(store.reorder(noteID: movedID, relativeTo: targetID))
        XCTAssertEqual(store.orderedItems(for: .todo).map(\.id), [movedID, targetID])
        XCTAssertEqual(store.items.first(where: { $0.id == movedID })?.updatedAt, timestamp)
        XCTAssertTrue(store.setStatus(.inProgress, for: movedID))
        XCTAssertNil(store.items.first(where: { $0.id == movedID })?.manualOrder)
    }

    private func queryNote(
        _ identity: Int,
        status: VoxtNoteStatus = .todo,
        priority: VoxtNotePriority = .none,
        manualOrder: Int64? = nil,
        createdAt: TimeInterval = 100,
        updatedAt: TimeInterval = 100,
        completedAt: TimeInterval? = nil
    ) -> VoxtNoteItem {
        VoxtNoteItem(
            id: UUID(uuidString: String(format: "00000000-0000-0000-0000-%012d", identity))!,
            sessionID: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
            createdAt: Date(timeIntervalSince1970: createdAt),
            text: "Body", title: "Title", titleGenerationState: .generated,
            updatedAt: Date(timeIntervalSince1970: updatedAt),
            completedAt: completedAt.map { Date(timeIntervalSince1970: $0) },
            status: status, priority: priority, manualOrder: manualOrder, source: .selection
        )
    }

    func testFallbackTitlePrefersSentencePrefix() {
        XCTAssertEqual(
            VoxtNoteTitleSupport.fallbackTitle(from: "Call Alice about the roadmap. Also share the draft."),
            "Call Alice about the roadmap"
        )
    }
}
