// VoxtRemindersSyncCoordinatorTests.swift
// Provides Voxt Reminders Sync Coordinator Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class VoxtRemindersSyncCoordinatorTests: XCTestCase {
    func testDisabledSyncDoesNotCreateReminders() async throws {
        let directory = try TemporaryDirectory()
        let noteStore = VoxtNoteStore(fileURL: directory.url.appendingPathComponent("notes.json"))
        let exportStore = VoxtNoteRemindersExportStore(fileURL: directory.url.appendingPathComponent("exports.json"))
        let backend = FakeVoxtRemindersSyncBackend()
        let settings = RemindersNoteSyncSettings(
            enabled: false,
            selectedListIdentifier: "list-1",
            selectedListTitle: "Voxt"
        )

        let coordinator = VoxtRemindersSyncCoordinator(
            noteStore: noteStore,
            settingsProvider: { settings },
            exportStore: exportStore,
            notificationCenter: NotificationCenter(),
            backendFactory: { backend }
        )

        _ = coordinator
        _ = noteStore.append(
            sessionID: UUID(),
            text: "disabled",
            title: "Disabled",
            titleGenerationState: .generated
        )

        try await Task.sleep(for: .milliseconds(500))
        XCTAssertTrue(backend.reminders.isEmpty)
        XCTAssertTrue(exportStore.recordsByNoteID.isEmpty)
    }

    func testCreateUpdateAndDeleteReminder() async throws {
        let directory = try TemporaryDirectory()
        let noteStore = VoxtNoteStore(fileURL: directory.url.appendingPathComponent("notes.json"))
        let exportStore = VoxtNoteRemindersExportStore(fileURL: directory.url.appendingPathComponent("exports.json"))
        let backend = FakeVoxtRemindersSyncBackend()
        let settings = RemindersNoteSyncSettings(
            enabled: true,
            selectedListIdentifier: "list-1",
            selectedListTitle: "Voxt"
        )

        let coordinator = VoxtRemindersSyncCoordinator(
            noteStore: noteStore,
            settingsProvider: { settings },
            exportStore: exportStore,
            notificationCenter: NotificationCenter(),
            backendFactory: { backend }
        )

        _ = coordinator
        let item = try XCTUnwrap(
            noteStore.append(
                sessionID: UUID(),
                text: "Ship reminders integration.",
                title: "Reminders sync",
                titleGenerationState: .generated
            )
        )

        try await Task.sleep(for: .milliseconds(500))

        let initialRecord = try XCTUnwrap(exportStore.recordsByNoteID[item.id])
        var reminder = try XCTUnwrap(backend.reminder(with: initialRecord.reminderCalendarItemIdentifier))
        XCTAssertEqual(reminder.title, "Reminders sync")
        XCTAssertEqual(reminder.notes, "Ship reminders integration.")
        XCTAssertFalse(reminder.isCompleted)
        XCTAssertEqual(reminder.priority, 0)

        XCTAssertTrue(noteStore.updateDetails(
            item.id,
            title: "Updated title",
            text: "Updated reminder content."
        ))
        XCTAssertTrue(noteStore.setStatus(.inProgress, for: item.id))
        XCTAssertTrue(noteStore.setPriority(.high, for: item.id))
        try await Task.sleep(for: .milliseconds(500))

        reminder = try XCTUnwrap(backend.reminder(with: initialRecord.reminderCalendarItemIdentifier))
        XCTAssertEqual(reminder.title, "Updated title")
        XCTAssertEqual(reminder.notes, "Updated reminder content.")
        XCTAssertFalse(reminder.isCompleted)
        XCTAssertEqual(reminder.priority, 1)

        XCTAssertTrue(noteStore.setStatus(.done, for: item.id))
        try await Task.sleep(for: .milliseconds(500))

        reminder = try XCTUnwrap(backend.reminder(with: initialRecord.reminderCalendarItemIdentifier))
        XCTAssertEqual(reminder.title, "Updated title")
        XCTAssertTrue(reminder.isCompleted)

        noteStore.delete(id: item.id)
        try await Task.sleep(for: .milliseconds(500))

        XCTAssertNil(try backend.reminder(with: initialRecord.reminderCalendarItemIdentifier))
        XCTAssertTrue(backend.deletedIdentifiers.contains(initialRecord.reminderCalendarItemIdentifier))
        XCTAssertTrue(exportStore.recordsByNoteID.isEmpty)
    }

    func testReminderPriorityMappingMatchesEventKitScale() {
        XCTAssertEqual(VoxtRemindersSyncMapping.priority(for: .none), 0)
        XCTAssertEqual(VoxtRemindersSyncMapping.priority(for: .high), 1)
        XCTAssertEqual(VoxtRemindersSyncMapping.priority(for: .medium), 5)
        XCTAssertEqual(VoxtRemindersSyncMapping.priority(for: .low), 9)
        XCTAssertFalse(VoxtRemindersSyncMapping.isCompleted(for: .todo))
        XCTAssertFalse(VoxtRemindersSyncMapping.isCompleted(for: .inProgress))
        XCTAssertFalse(VoxtRemindersSyncMapping.isCompleted(for: .backlog))
        XCTAssertTrue(VoxtRemindersSyncMapping.isCompleted(for: .done))
    }

    func testStaleReminderMappingCreatesReplacementReminder() async throws {
        let directory = try TemporaryDirectory()
        let noteStore = VoxtNoteStore(fileURL: directory.url.appendingPathComponent("notes.json"))
        let exportStore = VoxtNoteRemindersExportStore(fileURL: directory.url.appendingPathComponent("exports.json"))
        let backend = FakeVoxtRemindersSyncBackend()
        let settings = RemindersNoteSyncSettings(
            enabled: true,
            selectedListIdentifier: "list-1",
            selectedListTitle: "Voxt"
        )

        let coordinator = VoxtRemindersSyncCoordinator(
            noteStore: noteStore,
            settingsProvider: { settings },
            exportStore: exportStore,
            notificationCenter: NotificationCenter(),
            backendFactory: { backend }
        )

        _ = coordinator
        let item = try XCTUnwrap(
            noteStore.append(
                sessionID: UUID(),
                text: "Original note body.",
                title: "Original title",
                titleGenerationState: .generated
            )
        )

        try await Task.sleep(for: .milliseconds(500))

        let originalRecord = try XCTUnwrap(exportStore.recordsByNoteID[item.id])
        backend.dropReminder(identifier: originalRecord.reminderCalendarItemIdentifier)

        _ = noteStore.updateTitle("Recovered title", state: .generated, for: item.id)
        try await Task.sleep(for: .milliseconds(500))

        let replacementRecord = try XCTUnwrap(exportStore.recordsByNoteID[item.id])
        XCTAssertNotEqual(
            replacementRecord.reminderCalendarItemIdentifier,
            originalRecord.reminderCalendarItemIdentifier
        )
        let reminder = try XCTUnwrap(backend.reminder(with: replacementRecord.reminderCalendarItemIdentifier))
        XCTAssertEqual(reminder.title, "Recovered title")
    }

    func testUnauthorizedOrMissingListSkipsSyncAndKeepsLocalNote() async throws {
        let directory = try TemporaryDirectory()
        let noteStore = VoxtNoteStore(fileURL: directory.url.appendingPathComponent("notes.json"))
        let exportStore = VoxtNoteRemindersExportStore(fileURL: directory.url.appendingPathComponent("exports.json"))
        let backend = FakeVoxtRemindersSyncBackend()
        let notificationCenter = NotificationCenter()
        var settings = RemindersNoteSyncSettings(
            enabled: true,
            selectedListIdentifier: "list-1",
            selectedListTitle: "Voxt"
        )

        let coordinator = VoxtRemindersSyncCoordinator(
            noteStore: noteStore,
            settingsProvider: { settings },
            exportStore: exportStore,
            notificationCenter: notificationCenter,
            backendFactory: { backend }
        )

        _ = coordinator
        backend.authorizationStateValue = .denied
        let item = try XCTUnwrap(
            noteStore.append(
                sessionID: UUID(),
                text: "Permission blocked note.",
                title: "Blocked",
                titleGenerationState: .generated
            )
        )

        try await Task.sleep(for: .milliseconds(500))
        XCTAssertTrue(backend.reminders.isEmpty)
        XCTAssertTrue(exportStore.recordsByNoteID.isEmpty)
        XCTAssertNotNil(noteStore.items.first(where: { $0.id == item.id }))

        backend.authorizationStateValue = .authorized
        settings.selectedListIdentifier = "missing-list"
        notificationCenter.post(name: .voxtFeatureSettingsDidChange, object: nil)
        try await Task.sleep(for: .milliseconds(500))

        XCTAssertTrue(backend.reminders.isEmpty)
        XCTAssertTrue(exportStore.recordsByNoteID.isEmpty)
        XCTAssertNotNil(noteStore.items.first(where: { $0.id == item.id }))
    }

    func testApplicationTerminationShutdownStopsFutureSyncScheduling() async throws {
        let directory = try TemporaryDirectory()
        let noteStore = VoxtNoteStore(fileURL: directory.url.appendingPathComponent("notes.json"))
        let exportStore = VoxtNoteRemindersExportStore(fileURL: directory.url.appendingPathComponent("exports.json"))
        let backend = FakeVoxtRemindersSyncBackend()
        let notificationCenter = NotificationCenter()
        var settings = RemindersNoteSyncSettings(
            enabled: false,
            selectedListIdentifier: "list-1",
            selectedListTitle: "Voxt"
        )
        let coordinator = VoxtRemindersSyncCoordinator(
            noteStore: noteStore,
            settingsProvider: { settings },
            exportStore: exportStore,
            notificationCenter: notificationCenter,
            backendFactory: { backend }
        )

        coordinator.shutdownForApplicationTermination()
        settings.enabled = true
        _ = noteStore.append(
            sessionID: UUID(),
            text: "Must not sync while terminating.",
            title: "Termination",
            titleGenerationState: .generated
        )
        notificationCenter.post(name: .voxtFeatureSettingsDidChange, object: nil)

        try await Task.sleep(for: .milliseconds(300))
        XCTAssertTrue(backend.reminders.isEmpty)
        XCTAssertTrue(exportStore.recordsByNoteID.isEmpty)
        let didFinishPendingSync = await coordinator.waitForPendingApplicationTerminationSync(timeout: 0.2)
        XCTAssertTrue(didFinishPendingSync)
    }

    func testApplicationTerminationWaitUsesBoundedDeadlineForBlockedBackend() async throws {
        let directory = try TemporaryDirectory()
        let noteStore = VoxtNoteStore(fileURL: directory.url.appendingPathComponent("notes.json"))
        let backend = BlockingVoxtRemindersSyncBackend()
        let settings = RemindersNoteSyncSettings(
            enabled: true,
            selectedListIdentifier: "list-1",
            selectedListTitle: "Voxt"
        )
        let coordinator = VoxtRemindersSyncCoordinator(
            noteStore: noteStore,
            settingsProvider: { settings },
            notificationCenter: NotificationCenter(),
            backendFactory: { backend }
        )
        let didStart = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(returning: backend.started.wait(timeout: .now() + 1) == .success)
            }
        }
        XCTAssertTrue(didStart)

        coordinator.shutdownForApplicationTermination()
        let didFinishBeforeDeadline = await coordinator.waitForPendingApplicationTerminationSync(timeout: 0.05)
        XCTAssertFalse(didFinishBeforeDeadline)

        backend.release.signal()
        let didFinishAfterRelease = await coordinator.waitForPendingApplicationTerminationSync(timeout: 1)
        XCTAssertTrue(didFinishAfterRelease)
    }
}

private final class BlockingVoxtRemindersSyncBackend: VoxtRemindersSyncBackend, @unchecked Sendable {
    let started = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)

    func authorizationState() -> RemindersAuthorizationState {
        started.signal()
        release.wait()
        return .authorized
    }

    func writableLists() throws -> [RemindersListDescriptor] {
        [RemindersListDescriptor(identifier: "list-1", title: "Voxt", sourceTitle: "iCloud")]
    }

    func reminder(with identifier: String) throws -> RemindersReminderRecord? {
        nil
    }

    func saveReminder(_ payload: RemindersReminderPayload, existingIdentifier: String?) throws -> String {
        existingIdentifier ?? "unused"
    }

    func deleteReminder(with identifier: String) throws {}
}

private final class FakeVoxtRemindersSyncBackend: VoxtRemindersSyncBackend {
    private let lock = NSLock()

    var authorizationStateValue: RemindersAuthorizationState = .authorized
    var writableListDescriptors: [RemindersListDescriptor] = [
        RemindersListDescriptor(identifier: "list-1", title: "Voxt", sourceTitle: "iCloud")
    ]

    private var remindersByIdentifier: [String: RemindersReminderRecord] = [:]
    private var nextIdentifier = 1
    private(set) var deletedIdentifiers: [String] = []

    var reminders: [String: RemindersReminderRecord] {
        lock.withLock { remindersByIdentifier }
    }

    func authorizationState() -> RemindersAuthorizationState {
        authorizationStateValue
    }

    func writableLists() throws -> [RemindersListDescriptor] {
        writableListDescriptors
    }

    func reminder(with identifier: String) throws -> RemindersReminderRecord? {
        lock.withLock { remindersByIdentifier[identifier] }
    }

    func saveReminder(_ payload: RemindersReminderPayload, existingIdentifier: String?) throws -> String {
        lock.withLock {
            let identifier: String
            if let existingIdentifier, remindersByIdentifier[existingIdentifier] != nil {
                identifier = existingIdentifier
            } else {
                identifier = "rem-\(nextIdentifier)"
                nextIdentifier += 1
            }

            remindersByIdentifier[identifier] = RemindersReminderRecord(
                calendarItemIdentifier: identifier,
                listIdentifier: payload.listIdentifier,
                title: payload.title,
                notes: payload.notes,
                isCompleted: payload.isCompleted,
                priority: payload.priority
            )
            return identifier
        }
    }

    func deleteReminder(with identifier: String) throws {
        lock.withLock {
            remindersByIdentifier.removeValue(forKey: identifier)
            deletedIdentifiers.append(identifier)
        }
    }

    func dropReminder(identifier: String) {
        lock.withLock {
            _ = remindersByIdentifier.removeValue(forKey: identifier)
        }
    }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}
