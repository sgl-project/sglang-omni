// VoxtNoteStore.swift
// Provides Voxt Note Store for core app behavior.

import Combine
import Foundation
import SwiftData

@MainActor
final class VoxtNoteStore: ObservableObject {
    @Published private(set) var items: [VoxtNoteItem] = []
    @Published private(set) var lastErrorMessage: String?
    @Published private(set) var revision: UInt64 = 0
    @Published private(set) var availability: VoxtNoteStoreAvailability = .unavailable(message: "")
    @Published private(set) var lastRecoveryArchiveURL: URL?

    typealias ContainerFactory = (Schema, ModelConfiguration) throws -> ModelContainer

    private var container: ModelContainer?
    private var context: ModelContext?
    private let fileManager: FileManager
    private let legacyFileURL: URL?
    private let requestedStoreURL: URL?
    private let inMemory: Bool
    private let now: () -> Date
    private let persist: (ModelContext) throws -> Void
    private let containerFactory: ContainerFactory
    private var resolvedStoreURL: URL?
    private var recordsByID: [UUID: VoxtNoteRecord] = [:]
    private var migrationErrorMessage: String?

    nonisolated deinit {}

    init(
        fileManager: FileManager = .default,
        fileURL: URL? = nil,
        storeURL: URL? = nil,
        inMemory: Bool = false,
        now: @escaping () -> Date = Date.init,
        persist: @escaping (ModelContext) throws -> Void = { try $0.save() },
        containerFactory: @escaping ContainerFactory = { schema, configuration in
            try ModelContainer(for: schema, configurations: [configuration])
        }
    ) {
        self.fileManager = fileManager
        self.legacyFileURL = fileURL
        self.requestedStoreURL = storeURL
        self.inMemory = inMemory
        self.now = now
        self.persist = persist
        self.containerFactory = containerFactory

        openStorage()
    }

    var isAvailable: Bool { availability.isReady }

    var incompleteItems: [VoxtNoteItem] {
        items.filter { $0.status != .done }
    }

    @discardableResult
    func retryOpeningStorage() -> Bool {
        if isAvailable {
            reload()
            return isAvailable
        }
        return openStorage()
    }

    @discardableResult
    func archiveAndRebuildStorage() -> Bool {
        guard !inMemory else { return retryOpeningStorage() }

        do {
            let storeURL = try resolvedStoreURL
                ?? requestedStoreURL
                ?? Self.defaultStoreURL(fileManager: fileManager, legacyFileURL: legacyFileURL)
            detachStorage(clearPublishedItems: true)
            lastRecoveryArchiveURL = try archivePersistentStoreFamily(at: storeURL)
            return openStorage()
        } catch {
            markStorageUnavailable(error)
            return false
        }
    }

    @discardableResult
    private func openStorage() -> Bool {
        detachStorage(clearPublishedItems: true)

        do {
            let schema = Schema([VoxtNoteRecord.self])
            let configuration: ModelConfiguration
            if inMemory {
                configuration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
                resolvedStoreURL = nil
            } else {
                let resolvedStoreURL = try requestedStoreURL
                    ?? Self.defaultStoreURL(fileManager: fileManager, legacyFileURL: legacyFileURL)
                try fileManager.createDirectory(
                    at: resolvedStoreURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                self.resolvedStoreURL = resolvedStoreURL
                configuration = ModelConfiguration(
                    "VoxtNotes",
                    schema: schema,
                    url: resolvedStoreURL,
                    allowsSave: true,
                    cloudKitDatabase: .none
                )
            }
            let container = try containerFactory(schema, configuration)
            let context = ModelContext(container)
            context.autosaveEnabled = false
            self.container = container
            self.context = context
            availability = .ready
            lastErrorMessage = nil
        } catch {
            markStorageUnavailable(error)
            return false
        }

        if !inMemory {
            migrateLegacyJSONIfNeeded()
        }
        reload()
        return isAvailable
    }

    func reload() {
        guard let context else { return }
        do {
            let records = try context.fetch(FetchDescriptor<VoxtNoteRecord>())
            recordsByID = Dictionary(uniqueKeysWithValues: records.map { ($0.id, $0) })
            refreshPublishedItems()
            lastErrorMessage = migrationErrorMessage
        } catch {
            markStorageUnavailable(error)
        }
    }

    @discardableResult
    func append(
        sessionID: UUID,
        text: String,
        title: String,
        titleGenerationState: NoteTitleGenerationState,
        source: VoxtNoteSource = .transcription
    ) -> VoxtNoteItem? {
        create(
            sessionID: sessionID,
            text: text,
            title: title,
            titleGenerationState: titleGenerationState,
            status: .todo,
            priority: .none,
            source: source
        )
    }

    @discardableResult
    func create(
        sessionID: UUID,
        text: String,
        title: String,
        titleGenerationState: NoteTitleGenerationState,
        status: VoxtNoteStatus,
        priority: VoxtNotePriority,
        source: VoxtNoteSource
    ) -> VoxtNoteItem? {
        guard let context else { return nil }
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedTitle = VoxtNoteTitleSupport.normalizedTitle(title)
        guard !trimmedText.isEmpty, !normalizedTitle.isEmpty else { return nil }

        let timestamp = now()
        let item = VoxtNoteItem(
            id: UUID(),
            sessionID: sessionID,
            createdAt: timestamp,
            text: trimmedText,
            title: normalizedTitle,
            titleGenerationState: titleGenerationState,
            updatedAt: timestamp,
            status: status,
            priority: priority,
            manualOrder: nextManualOrder(status: status, priority: priority),
            source: source
        )
        let record = VoxtNoteRecord(item: item)
        context.insert(record)
        recordsByID[item.id] = record
        return saveAndReturn(item.id)
    }

    @discardableResult
    func updateTitle(
        _ title: String,
        state: NoteTitleGenerationState,
        for noteID: UUID
    ) -> VoxtNoteItem? {
        let normalizedTitle = VoxtNoteTitleSupport.normalizedTitle(title)
        guard !normalizedTitle.isEmpty, let record = recordsByID[noteID] else { return nil }
        guard record.titleGenerationStateRaw != NoteTitleGenerationState.userEdited.rawValue else {
            return record.item
        }
        record.title = normalizedTitle
        record.titleGenerationStateRaw = state.rawValue
        return saveAndReturn(noteID)
    }

    @discardableResult
    func updateText(
        _ text: String,
        ifUnchangedFrom expectedText: String,
        for noteID: UUID
    ) -> VoxtNoteItem? {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedExpectedText = expectedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty, let record = recordsByID[noteID] else { return nil }
        guard record.text == trimmedExpectedText else { return record.item }
        guard record.text != trimmedText else { return record.item }
        record.text = trimmedText
        record.updatedAt = now()
        return saveAndReturn(noteID)
    }

    @discardableResult
    func rename(_ noteID: UUID, to title: String) -> Bool {
        let normalizedTitle = VoxtNoteTitleSupport.normalizedTitle(title)
        guard !normalizedTitle.isEmpty, let record = recordsByID[noteID] else { return false }
        guard record.title != normalizedTitle else { return true }
        record.title = normalizedTitle
        record.titleGenerationStateRaw = NoteTitleGenerationState.userEdited.rawValue
        record.updatedAt = now()
        return save()
    }

    @discardableResult
    func updateDetails(_ noteID: UUID, title: String, text: String) -> Bool {
        let normalizedTitle = VoxtNoteTitleSupport.normalizedTitle(title)
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedTitle.isEmpty,
              !trimmedText.isEmpty,
              let record = recordsByID[noteID]
        else {
            return false
        }
        guard record.title != normalizedTitle || record.text != trimmedText else { return true }
        if record.title != normalizedTitle {
            record.title = normalizedTitle
            record.titleGenerationStateRaw = NoteTitleGenerationState.userEdited.rawValue
        }
        record.text = trimmedText
        record.updatedAt = now()
        return save()
    }

    @discardableResult
    func setPriority(_ priority: VoxtNotePriority, for noteID: UUID) -> Bool {
        guard let record = recordsByID[noteID] else { return false }
        guard record.priorityRaw != priority.rawValue else { return true }
        let status = VoxtNoteStatus(rawValue: record.statusRaw) ?? .todo
        record.manualOrder = nextManualOrder(status: status, priority: priority, excluding: noteID)
        record.priorityRaw = priority.rawValue
        record.updatedAt = now()
        return save()
    }

    @discardableResult
    func setStatus(_ status: VoxtNoteStatus, for noteID: UUID) -> Bool {
        guard let record = recordsByID[noteID] else { return false }
        guard record.statusRaw != status.rawValue else { return true }
        let priority = VoxtNotePriority(rawValue: record.priorityRaw) ?? .none
        record.manualOrder = nextManualOrder(status: status, priority: priority, excluding: noteID)
        record.statusRaw = status.rawValue
        record.updatedAt = now()
        record.completedAt = status == .done ? now() : nil
        return save()
    }

    @discardableResult
    func performPrimaryAction(for noteID: UUID) -> Bool {
        guard let item = items.first(where: { $0.id == noteID }) else { return false }
        return setStatus(item.status.primaryActionDestination, for: noteID)
    }

    @discardableResult
    func performDoubleClickAction(for noteID: UUID) -> Bool {
        guard let item = items.first(where: { $0.id == noteID }),
              let destination = item.status.doubleClickDestination else {
            return false
        }
        return setStatus(destination, for: noteID)
    }

    @discardableResult
    func updateCompletion(_ isCompleted: Bool, for noteID: UUID) -> VoxtNoteItem? {
        guard setStatus(isCompleted ? .done : .todo, for: noteID) else { return nil }
        return items.first(where: { $0.id == noteID })
    }

    @discardableResult
    func startAfterExternalDrag(noteID: UUID) -> Bool {
        guard let item = items.first(where: { $0.id == noteID }) else { return false }
        switch item.status {
        case .todo, .backlog:
            return setStatus(.inProgress, for: noteID)
        case .inProgress:
            return true
        case .done:
            return false
        }
    }

    @discardableResult
    func reorder(noteID: UUID, relativeTo targetID: UUID) -> Bool {
        guard let updates = VoxtNoteQueries.reorder(noteID: noteID, relativeTo: targetID, in: items) else {
            return false
        }
        for update in updates {
            recordsByID[update.noteID]?.manualOrder = update.manualOrder
        }
        recordsByID[noteID]?.updatedAt = now()
        return save()
    }

    func orderedItems(for status: VoxtNoteStatus) -> [VoxtNoteItem] {
        VoxtNoteQueries.orderedItems(items, for: status)
    }

    func snapshot(for scope: VoxtNoteScope) -> VoxtNoteScopeSnapshot {
        VoxtNoteQueries.snapshot(items, for: scope)
    }

    func delete(id: UUID) {
        guard let context, let record = recordsByID.removeValue(forKey: id) else { return }
        context.delete(record)
        _ = save()
    }

    func clearAll() {
        guard let context else { return }
        recordsByID.values.forEach(context.delete)
        recordsByID.removeAll()
        _ = save()
    }

    @discardableResult
    private func save() -> Bool {
        guard let context else { return false }
        do {
            try persist(context)
            refreshPublishedItems()
            lastErrorMessage = migrationErrorMessage
            return true
        } catch {
            let saveError = error.localizedDescription
            context.rollback()
            reload()
            lastErrorMessage = saveError
            return false
        }
    }

    private func saveAndReturn(_ noteID: UUID) -> VoxtNoteItem? {
        guard save() else { return nil }
        return items.first(where: { $0.id == noteID })
    }

    private func refreshPublishedItems() {
        items = VoxtNoteQueries.publicationOrder(recordsByID.values.map(\.item))
        revision &+= 1
    }

    private func nextManualOrder(
        status: VoxtNoteStatus,
        priority: VoxtNotePriority,
        excluding excludedID: UUID? = nil
    ) -> Int64? {
        let allocation = VoxtNoteQueries.nextManualOrder(
            in: items, status: status, priority: priority, excluding: excludedID
        )
        for update in allocation.updates {
            recordsByID[update.noteID]?.manualOrder = update.manualOrder
        }
        return allocation.nextOrder
    }

    private func migrateLegacyJSONIfNeeded() {
        guard let context else { return }
        do {
            guard let legacyURL = try resolvedLegacyFileURL(),
                  fileManager.fileExists(atPath: legacyURL.path) else {
                return
            }

            let data = try Data(contentsOf: legacyURL)
            let legacyItems = try JSONDecoder().decode([VoxtNoteItem].self, from: data)
            let existingRecords = try context.fetch(FetchDescriptor<VoxtNoteRecord>())
            let existingIDs = Set(existingRecords.map(\.id))

            for item in legacyItems where !existingIDs.contains(item.id) {
                let migratedItem = VoxtNoteItem(
                    id: item.id,
                    sessionID: item.sessionID,
                    createdAt: item.createdAt,
                    text: item.text,
                    title: item.title,
                    titleGenerationState: item.titleGenerationState,
                    updatedAt: item.updatedAt,
                    completedAt: item.completedAt,
                    status: item.status,
                    priority: item.priority,
                    manualOrder: item.manualOrder,
                    source: .migrated
                )
                context.insert(VoxtNoteRecord(item: migratedItem))
            }
            try context.save()
            try archiveMigratedLegacyFile(at: legacyURL)
            migrationErrorMessage = nil
            VoxtLog.history("Migrated legacy Voxt notes. count=\(legacyItems.count)")
        } catch {
            context.rollback()
            migrationErrorMessage = error.localizedDescription
            lastErrorMessage = error.localizedDescription
            VoxtLog.historyWarning("Legacy Voxt note migration failed. error=\(error.localizedDescription)")
        }
    }

    private func resolvedLegacyFileURL() throws -> URL? {
        if let legacyFileURL { return legacyFileURL }
        let directory = try Self.applicationSupportDirectory(fileManager: fileManager)
        return directory.appendingPathComponent("voxt-notes.json")
    }

    private func archiveMigratedLegacyFile(at legacyURL: URL) throws {
        let backupURL = legacyURL.appendingPathExtension("migrated-v1.bak")
        if fileManager.fileExists(atPath: backupURL.path) {
            try fileManager.removeItem(at: legacyURL)
        } else {
            try fileManager.moveItem(at: legacyURL, to: backupURL)
        }
    }

    private func detachStorage(clearPublishedItems: Bool) {
        recordsByID.removeAll()
        context = nil
        container = nil
        migrationErrorMessage = nil
        guard clearPublishedItems else { return }
        items = []
        revision &+= 1
    }

    private func markStorageUnavailable(_ error: Error) {
        let message = error.localizedDescription
        availability = .unavailable(message: message)
        detachStorage(clearPublishedItems: true)
        lastErrorMessage = message
        VoxtLog.historyWarning("Voxt note storage unavailable. error=\(message)")
    }

    private func archivePersistentStoreFamily(at storeURL: URL) throws -> URL? {
        let candidates = [
            storeURL,
            URL(fileURLWithPath: storeURL.path + "-wal"),
            URL(fileURLWithPath: storeURL.path + "-shm")
        ].filter { fileManager.fileExists(atPath: $0.path) }
        guard !candidates.isEmpty else { return nil }

        let recoveryRoot = storeURL.deletingLastPathComponent()
            .appendingPathComponent("Note Store Recovery", isDirectory: true)
        try fileManager.createDirectory(at: recoveryRoot, withIntermediateDirectories: true)
        let archiveURL = recoveryRoot.appendingPathComponent(
            "Notes-\(Int(now().timeIntervalSince1970))-\(UUID().uuidString.prefix(8))",
            isDirectory: true
        )
        try fileManager.createDirectory(at: archiveURL, withIntermediateDirectories: true)

        var movedFiles: [(source: URL, destination: URL)] = []
        do {
            for sourceURL in candidates {
                let destinationURL = archiveURL.appendingPathComponent(sourceURL.lastPathComponent)
                try fileManager.moveItem(at: sourceURL, to: destinationURL)
                movedFiles.append((sourceURL, destinationURL))
            }
        } catch {
            for movedFile in movedFiles.reversed() {
                try? fileManager.moveItem(at: movedFile.destination, to: movedFile.source)
            }
            try? fileManager.removeItem(at: archiveURL)
            throw error
        }

        VoxtLog.history("Archived unavailable Voxt note storage. path=\(archiveURL.path)")
        return archiveURL
    }

    private static func defaultStoreURL(fileManager: FileManager, legacyFileURL: URL?) throws -> URL {
        if let legacyFileURL {
            return legacyFileURL.deletingLastPathComponent().appendingPathComponent("voxt-notes.store")
        }
        return try applicationSupportDirectory(fileManager: fileManager)
            .appendingPathComponent("voxt-notes.store")
    }

    private static func applicationSupportDirectory(fileManager: FileManager) throws -> URL {
        let appSupport = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = appSupport.appendingPathComponent("Voxt", isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
