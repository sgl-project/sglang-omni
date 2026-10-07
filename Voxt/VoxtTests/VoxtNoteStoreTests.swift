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

    func testFallbackTitlePrefersSentencePrefix() {
        XCTAssertEqual(
            VoxtNoteTitleSupport.fallbackTitle(from: "Call Alice about the roadmap. Also share the draft."),
            "Call Alice about the roadmap"
        )
    }
}
