// VoxtNoteStore.swift
// Provides Voxt Note Store for core app behavior.

import Combine
import Foundation
import SwiftData

nonisolated enum NoteTitleGenerationState: String, Codable, Equatable, Sendable {
    case pending
    case generated
    case fallback
    case userEdited
}

nonisolated enum VoxtNoteStatus: String, Codable, CaseIterable, Identifiable, Sendable {
    case todo
    case inProgress
    case done
    case backlog

    var id: String { rawValue }

    var title: String {
        switch self {
        case .todo:
            return AppLocalization.localizedString("To do")
        case .inProgress:
            return AppLocalization.localizedString("In Progress")
        case .done:
            return AppLocalization.localizedString("Done")
        case .backlog:
            return AppLocalization.localizedString("Backlog")
        }
    }

    var primaryActionDestination: VoxtNoteStatus {
        switch self {
        case .backlog, .done:
            return .todo
        case .todo, .inProgress:
            return .done
        }
    }

    var doubleClickDestination: VoxtNoteStatus? {
        switch self {
        case .backlog:
            return .todo
        case .todo:
            return .inProgress
        case .inProgress:
            return .todo
        case .done:
            return nil
        }
    }

    static let moveMenuOrder: [VoxtNoteStatus] = [.backlog, .inProgress, .todo, .done]
}

nonisolated enum VoxtNoteScope: String, Codable, CaseIterable, Identifiable, Sendable {
    case notes
    case backlog

    var id: String { rawValue }

    var title: String {
        switch self {
        case .notes:
            return AppLocalization.localizedString("Notes")
        case .backlog:
            return AppLocalization.localizedString("Backlog")
        }
    }

    var statuses: [VoxtNoteStatus] {
        switch self {
        case .notes:
            return [.inProgress, .todo, .done]
        case .backlog:
            return [.backlog]
        }
    }

    var activeStatuses: Set<VoxtNoteStatus> {
        switch self {
        case .notes:
            return [.inProgress, .todo]
        case .backlog:
            return [.backlog]
        }
    }
}

nonisolated enum VoxtNotePriority: String, Codable, CaseIterable, Identifiable, Sendable {
    case none
    case low
    case medium
    case high

    var id: String { rawValue }

    var title: String {
        switch self {
        case .none:
            return AppLocalization.localizedString("None")
        case .low:
            return AppLocalization.localizedString("Low")
        case .medium:
            return AppLocalization.localizedString("Medium")
        case .high:
            return AppLocalization.localizedString("High")
        }
    }

    var sortRank: Int {
        switch self {
        case .none:
            return -1
        case .low:
            return 0
        case .medium:
            return 1
        case .high:
            return 2
        }
    }
}

nonisolated enum VoxtNoteSource: String, Codable, Sendable {
    case transcription
    case selection
    case migrated
}

nonisolated struct VoxtNoteItem: Identifiable, Codable, Hashable, Sendable {
    let id: UUID
    let sessionID: UUID
    let createdAt: Date
    let updatedAt: Date
    let completedAt: Date?
    let text: String
    let title: String
    let titleGenerationState: NoteTitleGenerationState
    let status: VoxtNoteStatus
    let priority: VoxtNotePriority
    let manualOrder: Int64?
    let source: VoxtNoteSource

    var isCompleted: Bool { status == .done }

    init(
        id: UUID,
        sessionID: UUID,
        createdAt: Date,
        text: String,
        title: String,
        titleGenerationState: NoteTitleGenerationState,
        isCompleted: Bool = false,
        updatedAt: Date? = nil,
        completedAt: Date? = nil,
        status: VoxtNoteStatus? = nil,
        priority: VoxtNotePriority = .none,
        manualOrder: Int64? = nil,
        source: VoxtNoteSource = .transcription
    ) {
        self.id = id
        self.sessionID = sessionID
        self.createdAt = createdAt
        self.updatedAt = updatedAt ?? createdAt
        self.completedAt = completedAt
        self.text = text
        self.title = title
        self.titleGenerationState = titleGenerationState
        self.status = status ?? (isCompleted ? .done : .todo)
        self.priority = priority
        self.manualOrder = manualOrder
        self.source = source
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case sessionID
        case createdAt
        case updatedAt
        case completedAt
        case text
        case title
        case titleGenerationState
        case status
        case priority
        case manualOrder
        case source
        case isCompleted
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        sessionID = try container.decode(UUID.self, forKey: .sessionID)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? createdAt
        completedAt = try container.decodeIfPresent(Date.self, forKey: .completedAt)
        text = try container.decode(String.self, forKey: .text)
        title = try container.decode(String.self, forKey: .title)
        titleGenerationState = try container.decode(NoteTitleGenerationState.self, forKey: .titleGenerationState)
        if let decodedStatus = try container.decodeIfPresent(VoxtNoteStatus.self, forKey: .status) {
            status = decodedStatus
        } else {
            let legacyCompleted = try container.decodeIfPresent(Bool.self, forKey: .isCompleted) ?? false
            status = legacyCompleted ? .done : .todo
        }
        priority = try container.decodeIfPresent(VoxtNotePriority.self, forKey: .priority) ?? .none
        manualOrder = try container.decodeIfPresent(Int64.self, forKey: .manualOrder)
        source = try container.decodeIfPresent(VoxtNoteSource.self, forKey: .source) ?? .migrated
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(sessionID, forKey: .sessionID)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(updatedAt, forKey: .updatedAt)
        try container.encodeIfPresent(completedAt, forKey: .completedAt)
        try container.encode(text, forKey: .text)
        try container.encode(title, forKey: .title)
        try container.encode(titleGenerationState, forKey: .titleGenerationState)
        try container.encode(status, forKey: .status)
        try container.encode(priority, forKey: .priority)
        try container.encodeIfPresent(manualOrder, forKey: .manualOrder)
        try container.encode(source, forKey: .source)
        try container.encode(isCompleted, forKey: .isCompleted)
    }
}

@Model
nonisolated final class VoxtNoteRecord {
    @Attribute(.unique) var id: UUID
    var sessionID: UUID
    var createdAt: Date
    var updatedAt: Date
    var completedAt: Date?
    var text: String
    var title: String
    var titleGenerationStateRaw: String
    var statusRaw: String
    var priorityRaw: String
    var manualOrder: Int64?
    var sourceRaw: String

    init(item: VoxtNoteItem) {
        id = item.id
        sessionID = item.sessionID
        createdAt = item.createdAt
        updatedAt = item.updatedAt
        completedAt = item.completedAt
        text = item.text
        title = item.title
        titleGenerationStateRaw = item.titleGenerationState.rawValue
        statusRaw = item.status.rawValue
        priorityRaw = item.priority.rawValue
        manualOrder = item.manualOrder
        sourceRaw = item.source.rawValue
    }

    var item: VoxtNoteItem {
        VoxtNoteItem(
            id: id,
            sessionID: sessionID,
            createdAt: createdAt,
            text: text,
            title: title,
            titleGenerationState: NoteTitleGenerationState(rawValue: titleGenerationStateRaw) ?? .fallback,
            updatedAt: updatedAt,
            completedAt: completedAt,
            status: VoxtNoteStatus(rawValue: statusRaw) ?? .todo,
            priority: VoxtNotePriority(rawValue: priorityRaw) ?? .none,
            manualOrder: manualOrder,
            source: VoxtNoteSource(rawValue: sourceRaw) ?? .transcription
        )
    }
}

nonisolated struct VoxtNoteSectionSnapshot: Identifiable, Sendable {
    let status: VoxtNoteStatus
    let items: [VoxtNoteItem]

    var id: VoxtNoteStatus { status }
}

nonisolated struct VoxtNoteScopeSnapshot: Sendable {
    let sections: [VoxtNoteSectionSnapshot]
    let visibleCount: Int
    let activeCount: Int
}

nonisolated enum VoxtNoteStoreAvailability: Equatable, Sendable {
    case ready
    case unavailable(message: String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    var errorMessage: String? {
        guard case .unavailable(let message) = self else { return nil }
        return message
    }
}

nonisolated enum VoxtNoteTitleSupport {
    static func fallbackTitle(from text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Note" }

        let collapsedWhitespace = trimmed
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !collapsedWhitespace.isEmpty else { return "Note" }

        let stopCharacters = CharacterSet(charactersIn: "\n。！？!?;；:.")
        if let boundary = collapsedWhitespace.unicodeScalars.firstIndex(where: { stopCharacters.contains($0) }) {
            let candidate = String(collapsedWhitespace.unicodeScalars[..<boundary])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !candidate.isEmpty {
                return limitedTitle(candidate)
            }
        }

        return limitedTitle(collapsedWhitespace)
    }

    static func normalizedGeneratedTitle(_ title: String) -> String {
        let firstLine = title
            .split(whereSeparator: \.isNewline)
            .first
            .map(String.init) ?? ""
        let trimmed = firstLine
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'“”‘’"))
        guard !trimmed.isEmpty else { return "" }
        return limitedTitle(trimmed)
    }

    private static func limitedTitle(_ value: String) -> String {
        let limit = containsCJK(value) ? 20 : 48
        let candidate = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard candidate.count > limit else { return candidate }
        let endIndex = candidate.index(candidate.startIndex, offsetBy: limit)
        return String(candidate[..<endIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            switch scalar.value {
            case 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF:
                return true
            default:
                return false
            }
        }
    }
}

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

    var latestItem: VoxtNoteItem? { items.first }

    var incompleteItems: [VoxtNoteItem] {
        items.filter { $0.status != .done }
    }

    var latestIncompleteItem: VoxtNoteItem? { incompleteItems.first }

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
        let normalizedTitle = Self.normalizedTitle(title)
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
        let normalizedTitle = Self.normalizedTitle(title)
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
        let normalizedTitle = Self.normalizedTitle(title)
        guard !normalizedTitle.isEmpty, let record = recordsByID[noteID] else { return false }
        guard record.title != normalizedTitle else { return true }
        record.title = normalizedTitle
        record.titleGenerationStateRaw = NoteTitleGenerationState.userEdited.rawValue
        record.updatedAt = now()
        return save()
    }

    @discardableResult
    func updateDetails(_ noteID: UUID, title: String, text: String) -> Bool {
        let normalizedTitle = Self.normalizedTitle(title)
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
        guard noteID != targetID,
              let note = items.first(where: { $0.id == noteID }),
              let target = items.first(where: { $0.id == targetID }),
              note.status == target.status,
              note.priority == target.priority else {
            return false
        }

        var group = orderedItems(for: note.status).filter { $0.priority == note.priority }
        guard let sourceIndex = group.firstIndex(where: { $0.id == noteID }),
              let targetIndex = group.firstIndex(where: { $0.id == targetID }) else {
            return false
        }

        let moved = group.remove(at: sourceIndex)
        group.insert(moved, at: min(targetIndex, group.count))
        for (index, item) in group.enumerated() {
            recordsByID[item.id]?.manualOrder = Int64(group.count - index)
        }
        recordsByID[noteID]?.updatedAt = now()
        return save()
    }

    func orderedItems(for status: VoxtNoteStatus) -> [VoxtNoteItem] {
        items
            .filter { $0.status == status }
            .sorted(by: status == .done ? Self.completedNoteSortOrder : Self.noteSortOrder)
    }

    func snapshot(for scope: VoxtNoteScope) -> VoxtNoteScopeSnapshot {
        let sections = scope.statuses.compactMap { status -> VoxtNoteSectionSnapshot? in
            let statusItems = orderedItems(for: status)
            guard !statusItems.isEmpty else { return nil }
            return VoxtNoteSectionSnapshot(status: status, items: statusItems)
        }
        return VoxtNoteScopeSnapshot(
            sections: sections,
            visibleCount: sections.reduce(0) { $0 + $1.items.count },
            activeCount: sections.reduce(0) { count, section in
                count + (scope.activeStatuses.contains(section.status) ? section.items.count : 0)
            }
        )
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
        items = recordsByID.values.map(\.item).sorted { lhs, rhs in
            if lhs.createdAt != rhs.createdAt { return lhs.createdAt > rhs.createdAt }
            return lhs.id.uuidString < rhs.id.uuidString
        }
        revision &+= 1
    }

    private func nextManualOrder(
        status: VoxtNoteStatus,
        priority: VoxtNotePriority,
        excluding excludedID: UUID? = nil
    ) -> Int64? {
        let group = items.filter {
            $0.id != excludedID && $0.status == status && $0.priority == priority
        }
        guard let maximum = group.compactMap(\.manualOrder).max() else { return nil }
        guard maximum == .max else { return maximum + 1 }

        let orderedGroup = group.sorted(by: Self.noteSortOrder)
        for (index, item) in orderedGroup.enumerated() {
            recordsByID[item.id]?.manualOrder = Int64(orderedGroup.count - index)
        }
        return Int64(orderedGroup.count + 1)
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

    private static func normalizedTitle(_ title: String) -> String {
        title
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private static func noteSortOrder(_ lhs: VoxtNoteItem, _ rhs: VoxtNoteItem) -> Bool {
        if lhs.priority.sortRank != rhs.priority.sortRank {
            return lhs.priority.sortRank > rhs.priority.sortRank
        }
        switch (lhs.manualOrder, rhs.manualOrder) {
        case let (.some(lhsOrder), .some(rhsOrder)) where lhsOrder != rhsOrder:
            return lhsOrder > rhsOrder
        case (.some, .none):
            return true
        case (.none, .some):
            return false
        default:
            break
        }
        if lhs.updatedAt != rhs.updatedAt {
            return lhs.updatedAt > rhs.updatedAt
        }
        return lhs.id.uuidString < rhs.id.uuidString
    }

    private static func completedNoteSortOrder(_ lhs: VoxtNoteItem, _ rhs: VoxtNoteItem) -> Bool {
        let lhsCompletedAt = lhs.completedAt ?? lhs.updatedAt
        let rhsCompletedAt = rhs.completedAt ?? rhs.updatedAt
        if lhsCompletedAt != rhsCompletedAt {
            return lhsCompletedAt > rhsCompletedAt
        }
        return lhs.id.uuidString < rhs.id.uuidString
    }
}
