// VoxtObsidianSyncCoordinator.swift
// Provides Voxt Obsidian Sync Coordinator for core app behavior.

import Foundation
import Combine

nonisolated enum VoxtObsidianSyncMapping {
    static func statusValue(for status: VoxtNoteStatus) -> String {
        switch status {
        case .todo: return "todo"
        case .inProgress: return "in-progress"
        case .done: return "done"
        case .backlog: return "backlog"
        }
    }

    static func priorityValue(for priority: VoxtNotePriority) -> String {
        priority.rawValue
    }
}

final class VoxtObsidianSyncCoordinator {
    private let noteStore: VoxtNoteStore
    private let settingsProvider: () -> ObsidianNoteSyncSettings
    private let exportStore: VoxtNoteObsidianExportStore
    private let fileManagerBox: FileManagerBox
    private let notificationCenter: NotificationCenter
    private let queue = DispatchQueue(label: "com.voxt.obsidian-sync", qos: .utility)
    private let pendingReconciliationGroup = DispatchGroup()
    private var itemsCancellable: AnyCancellable?
    private var settingsObserver: NSObjectProtocol?
    private var latestNotesSnapshot: [VoxtNoteItem]
    private var isShuttingDownForApplicationTermination = false

    @MainActor
    init(
        noteStore: VoxtNoteStore,
        settingsProvider: @escaping () -> ObsidianNoteSyncSettings,
        exportStore: VoxtNoteObsidianExportStore? = nil,
        fileManager: FileManager = .default,
        notificationCenter: NotificationCenter = .default
    ) {
        self.noteStore = noteStore
        self.settingsProvider = settingsProvider
        self.exportStore = exportStore ?? VoxtNoteObsidianExportStore()
        self.fileManagerBox = FileManagerBox(fileManager: fileManager)
        self.notificationCenter = notificationCenter
        self.latestNotesSnapshot = noteStore.items

        itemsCancellable = noteStore.$items
            .dropFirst()
            .receive(on: RunLoop.main)
            .sink { [weak self] items in
                self?.latestNotesSnapshot = items
                self?.scheduleSync(notes: items, settings: settingsProvider(), reason: "notes-updated")
            }

        settingsObserver = notificationCenter.addObserver(
            forName: .voxtFeatureSettingsDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.scheduleSync(
                    notes: self.latestNotesSnapshot,
                    settings: self.settingsProvider(),
                    reason: "settings-updated"
                )
            }
        }

        scheduleSync(
            notes: noteStore.items,
            settings: settingsProvider(),
            reason: "startup"
        )
    }

    deinit {
        if let settingsObserver {
            notificationCenter.removeObserver(settingsObserver)
        }
    }

    @MainActor
    func shutdownForApplicationTermination() {
        guard !isShuttingDownForApplicationTermination else { return }
        isShuttingDownForApplicationTermination = true
        itemsCancellable?.cancel()
        itemsCancellable = nil
        if let settingsObserver {
            notificationCenter.removeObserver(settingsObserver)
            self.settingsObserver = nil
        }
    }

    func waitForPendingApplicationTerminationSync(timeout: TimeInterval) async -> Bool {
        let pendingReconciliationGroup = self.pendingReconciliationGroup
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(
                    returning: pendingReconciliationGroup.wait(timeout: .now() + timeout) == .success
                )
            }
        }
    }

    @MainActor
    private func scheduleSync(
        notes: [VoxtNoteItem],
        settings: ObsidianNoteSyncSettings,
        reason: String
    ) {
        guard !isShuttingDownForApplicationTermination else { return }
        guard noteStore.isAvailable else {
            VoxtLog.warning("Obsidian sync skipped because note storage is unavailable. reason=\(reason)")
            return
        }
        pendingReconciliationGroup.enter()
        queue.async { [exportStore, fileManagerBox, pendingReconciliationGroup] in
            defer { pendingReconciliationGroup.leave() }
            Self.reconcile(
                notes: notes,
                settings: settings,
                exportStore: exportStore,
                fileManager: fileManagerBox.fileManager,
                reason: reason
            )
        }
    }

    private static func reconcile(
        notes: [VoxtNoteItem],
        settings: ObsidianNoteSyncSettings,
        exportStore: VoxtNoteObsidianExportStore,
        fileManager: FileManager,
        reason: String
    ) {
        guard settings.enabled else { return }
        guard let vaultAccess = SecurityScopedBookmarkSupport.accessDirectoryURL(
            bookmarkData: settings.vaultBookmarkData,
            fallbackPath: settings.vaultPath
        ) else {
            VoxtLog.warning("Obsidian sync skipped because vault access is unavailable. reason=\(reason)")
            return
        }
        let vaultURL = vaultAccess.url
        defer { withExtendedLifetime(vaultAccess) {} }

        let previousRecordsByNoteID = exportStore.recordsByNoteID
        var nextRecordsByNoteID: [UUID: VoxtNoteObsidianExportRecord] = [:]
        var occupiedRelativePaths = Set<String>()
        var assignedSessionPaths: [UUID: String] = [:]
        let currentNoteIDs = Set(notes.map(\.id))
        let groupingContext = makeGroupingContext(notes: notes)

        for note in notes.sorted(by: noteSortOrder) {
            if settings.groupingMode == .session,
               let existingSessionPath = assignedSessionPaths[note.sessionID] {
                let record = VoxtNoteObsidianExportRecord(
                    noteID: note.id,
                    groupingMode: .session,
                    relativeFilePath: existingSessionPath
                )
                nextRecordsByNoteID[note.id] = record
                continue
            }

            if let existingRecord = previousRecordsByNoteID[note.id],
               shouldReuseRecord(
                existingRecord,
                for: note,
                settings: settings,
                vaultURL: vaultURL,
                fileManager: fileManager
               ) {
                nextRecordsByNoteID[note.id] = existingRecord
                occupiedRelativePaths.insert(existingRecord.relativeFilePath)
                if settings.groupingMode == .session {
                    assignedSessionPaths[note.sessionID] = existingRecord.relativeFilePath
                }
                continue
            }

            let record = makeExportRecord(
                for: note,
                settings: settings,
                vaultURL: vaultURL,
                occupiedRelativePaths: &occupiedRelativePaths,
                fileManager: fileManager,
                groupingContext: groupingContext
            )
            nextRecordsByNoteID[note.id] = record
            if settings.groupingMode == .session {
                assignedSessionPaths[note.sessionID] = record.relativeFilePath
            }
        }

        let removedRecords = previousRecordsByNoteID.values.filter { !currentNoteIDs.contains($0.noteID) }
        let retiredRelativePaths = Set(previousRecordsByNoteID.values.map(\.relativeFilePath))
            .subtracting(Set(nextRecordsByNoteID.values.map(\.relativeFilePath)))

        exportStore.replaceAll(Array(nextRecordsByNoteID.values))

        let notesByRelativeFilePath = Dictionary(grouping: notes) { note in
            nextRecordsByNoteID[note.id]?.relativeFilePath ?? ""
        }
        let staleRelativePaths = retiredRelativePaths.union(Set(removedRecords.map(\.relativeFilePath)))
        let activeRelativePaths = Set(notesByRelativeFilePath.keys).subtracting([""])

        // Write current files before pruning retired paths so renamed single-note files
        // can still read the previous managed body and preserve user edits.
        for relativePath in activeRelativePaths {
            let fileURL = vaultURL.appendingPathComponent(relativePath, isDirectory: false)
            let assignedNotes = (notesByRelativeFilePath[relativePath] ?? []).sorted { lhs, rhs in
                if lhs.createdAt == rhs.createdAt {
                    return lhs.id.uuidString < rhs.id.uuidString
                }
                return lhs.createdAt < rhs.createdAt
            }

            if assignedNotes.isEmpty {
                try? fileManager.removeItem(at: fileURL)
                pruneEmptyDirectories(
                    startingAt: fileURL.deletingLastPathComponent(),
                    stopAt: vaultURL,
                    fileManager: fileManager
                )
                continue
            }

            guard let record = assignedNotes.first.flatMap({ nextRecordsByNoteID[$0.id] }) else {
                continue
            }

            do {
                try fileManager.createDirectory(
                    at: fileURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )

                switch record.groupingMode {
                case .file:
                    guard let note = assignedNotes.first else { continue }
                    try syncSingleNoteFile(note: note, to: fileURL)
                case .session, .daily:
                    let renderedText = renderGroupedFile(notes: assignedNotes, record: record)
                    try renderedText.write(to: fileURL, atomically: true, encoding: .utf8)
                }
            } catch {
                VoxtLog.warning("Obsidian sync write failed. file=\(relativePath), error=\(error.localizedDescription)")
            }
        }

        for relativePath in staleRelativePaths where !relativePath.isEmpty {
            let fileURL = vaultURL.appendingPathComponent(relativePath, isDirectory: false)
            try? fileManager.removeItem(at: fileURL)
            pruneEmptyDirectories(
                startingAt: fileURL.deletingLastPathComponent(),
                stopAt: vaultURL,
                fileManager: fileManager
            )
        }
    }

    private static func makeExportRecord(
        for note: VoxtNoteItem,
        settings: ObsidianNoteSyncSettings,
        vaultURL: URL,
        occupiedRelativePaths: inout Set<String>,
        fileManager: FileManager,
        groupingContext: GroupingContext
    ) -> VoxtNoteObsidianExportRecord {
        let relativePath = relativeFilePath(
            for: note,
            settings: settings,
            vaultURL: vaultURL,
            occupiedRelativePaths: &occupiedRelativePaths,
            fileManager: fileManager,
            groupingContext: groupingContext
        )
        occupiedRelativePaths.insert(relativePath)
        return VoxtNoteObsidianExportRecord(
            noteID: note.id,
            groupingMode: settings.groupingMode,
            relativeFilePath: relativePath
        )
    }

    private static func relativeFilePath(
        for note: VoxtNoteItem,
        settings: ObsidianNoteSyncSettings,
        vaultURL: URL,
        occupiedRelativePaths: inout Set<String>,
        fileManager: FileManager,
        groupingContext: GroupingContext
    ) -> String {
        let folder = normalizedRelativeFolder(settings.relativeFolder)
        let dateSegment = dayFolderFormatter.string(from: note.createdAt)

        switch settings.groupingMode {
        case .session:
            let representative = groupingContext.representativeNoteBySessionID[note.sessionID] ?? note
            return uniqueRelativePath(
                baseDirectory: "\(folder)/Sessions/\(dateSegment)",
                stem: readableFileStem(for: representative),
                vaultURL: vaultURL,
                occupiedRelativePaths: &occupiedRelativePaths,
                fileManager: fileManager,
                expectedManagedIdentifier: .session(note.sessionID)
            )
        case .daily:
            return "\(folder)/Daily/\(dateSegment) Notes.md"
        case .file:
            return uniqueRelativePath(
                baseDirectory: "\(folder)/Notes/\(dateSegment)",
                stem: readableFileStem(for: note),
                vaultURL: vaultURL,
                occupiedRelativePaths: &occupiedRelativePaths,
                fileManager: fileManager,
                expectedManagedIdentifier: .note(note.id)
            )
        }
    }

    private static func uniqueRelativePath(
        baseDirectory: String,
        stem: String,
        vaultURL: URL,
        occupiedRelativePaths: inout Set<String>,
        fileManager: FileManager,
        expectedManagedIdentifier: ManagedFileIdentifier?
    ) -> String {
        var candidate = "\(baseDirectory)/\(stem).md"
        var suffix = 2

        while occupiedRelativePaths.contains(candidate) ||
                isConflictingExistingPath(
                    candidate,
                    vaultURL: vaultURL,
                    fileManager: fileManager,
                    expectedManagedIdentifier: expectedManagedIdentifier
                ) {
            candidate = "\(baseDirectory)/\(stem) (\(suffix)).md"
            suffix += 1
        }
        return candidate
    }

    private static func isConflictingExistingPath(
        _ relativePath: String,
        vaultURL: URL,
        fileManager: FileManager,
        expectedManagedIdentifier: ManagedFileIdentifier?
    ) -> Bool {
        let fileURL = vaultURL.appendingPathComponent(relativePath)
        guard fileManager.fileExists(atPath: fileURL.path) else {
            return false
        }

        guard let expectedManagedIdentifier else {
            return true
        }

        guard let text = try? String(contentsOf: fileURL, encoding: .utf8),
              let existingIdentifier = managedIdentifier(in: text) else {
            return true
        }

        return existingIdentifier != expectedManagedIdentifier
    }

    private static func normalizedRelativeFolder(_ value: String) -> String {
        let trimmed = value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        return trimmed.isEmpty ? "Voxt" : trimmed
    }

    private static func readableFileStem(for note: VoxtNoteItem) -> String {
        let timestamp = fileNameTimeFormatter.string(from: note.createdAt)
        let title = fileStemTitle(for: note)
        if title.isEmpty {
            return timestamp
        }
        return "\(timestamp) - \(title)"
    }

    private static func fileStemTitle(for note: VoxtNoteItem) -> String {
        sanitizedFileNameComponent(note.title)
    }

    private static func sanitizedFileNameComponent(_ value: String) -> String {
        let scalars = value.unicodeScalars.map { scalar -> Character in
            let disallowed = CharacterSet(charactersIn: "/\\?%*|\"<>")
            if disallowed.contains(scalar) || CharacterSet.controlCharacters.contains(scalar) {
                return " "
            }
            return Character(scalar)
        }
        let candidate = String(scalars)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !candidate.isEmpty else { return "" }
        return String(candidate.prefix(64)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func syncSingleNoteFile(
        note: VoxtNoteItem,
        to fileURL: URL
    ) throws {
        let renderedText = renderSingleNoteFile(note, bodyText: note.text)
        try renderedText.write(to: fileURL, atomically: true, encoding: .utf8)
    }

    private static func renderSingleNoteFile(_ note: VoxtNoteItem, bodyText: String) -> String {
        let normalizedBody = bodyText.trimmingCharacters(in: .newlines)

        return """
        ---
        type: \(yamlValue("voxt-note"))
        source: \(yamlValue("voxt"))
        capture-source: \(yamlValue(note.source.rawValue))
        created: \(yamlValue(isoFormatter.string(from: note.createdAt)))
        updated: \(yamlValue(isoFormatter.string(from: note.updatedAt)))
        completed: \(yamlValue(note.completedAt.map { isoFormatter.string(from: $0) } ?? ""))
        status: \(yamlValue(VoxtObsidianSyncMapping.statusValue(for: note.status)))
        priority: \(yamlValue(VoxtObsidianSyncMapping.priorityValue(for: note.priority)))
        title: \(yamlValue(note.title))
        note-id: \(yamlValue(note.id.uuidString))
        session-id: \(yamlValue(note.sessionID.uuidString))
        ---

        # \(note.title)

        \(normalizedBody)
        """
    }

    private static func renderGroupedFile(
        notes: [VoxtNoteItem],
        record: VoxtNoteObsidianExportRecord
    ) -> String {
        let title = groupedFileTitle(notes: notes, groupingMode: record.groupingMode)
        let noteBlocks = notes.map(renderGroupedNoteBlock).joined(separator: "\n\n")
        let sessionLine = record.groupingMode == .session
            ? "session-id: \(yamlValue(notes.first?.sessionID.uuidString ?? ""))\n"
            : ""

        return """
        ---
        type: \(yamlValue("voxt-note-collection"))
        source: \(yamlValue("voxt"))
        grouping: \(yamlValue(record.groupingMode.rawValue))
        updated: \(yamlValue(isoFormatter.string(from: Date())))
        \(sessionLine)---

        # \(title)

        \(noteBlocks)
        """
    }

    private static func groupedFileTitle(
        notes: [VoxtNoteItem],
        groupingMode: ObsidianNoteGroupingMode
    ) -> String {
        switch groupingMode {
        case .session:
            return notes.first?.title ?? "Voxt Session Notes"
        case .daily:
            guard let date = notes.first?.createdAt else { return "Voxt Daily Notes" }
            return "\(dayFolderFormatter.string(from: date)) Notes"
        case .file:
            return notes.first?.title ?? "Voxt Note"
        }
    }

    nonisolated private static func renderGroupedNoteBlock(_ note: VoxtNoteItem) -> String {
        """
        ## \(note.title)

        > Created: \(displayDateTimeString(from: note.createdAt))
        >
        > Updated: \(displayDateTimeString(from: note.updatedAt))
        >
        > Status: \(VoxtObsidianSyncMapping.statusValue(for: note.status))
        >
        > Priority: \(VoxtObsidianSyncMapping.priorityValue(for: note.priority))

        \(note.text)
        """
    }

    nonisolated private static func displayDateTimeString(from date: Date) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        return formatter.string(from: date)
    }

    private static func managedIdentifier(in text: String) -> ManagedFileIdentifier? {
        guard text.hasPrefix("---\n") else { return nil }
        let bodyStartMarker = "\n---\n"
        guard let frontmatterEndRange = text.range(of: bodyStartMarker) else { return nil }

        let frontmatterText = String(text[text.index(text.startIndex, offsetBy: 4)..<frontmatterEndRange.lowerBound])
        let metadata = parseFrontmatter(frontmatterText)
        guard metadata["source"] == "voxt" else { return nil }

        switch metadata["type"] {
        case "voxt-note":
            guard let noteIDRaw = metadata["note-id"],
                  let noteID = UUID(uuidString: noteIDRaw) else {
                return nil
            }
            return .note(noteID)
        case "voxt-note-collection":
            guard metadata["grouping"] == ObsidianNoteGroupingMode.session.rawValue,
                  let sessionIDRaw = metadata["session-id"],
                  let sessionID = UUID(uuidString: sessionIDRaw) else {
                return nil
            }
            return .session(sessionID)
        default:
            return nil
        }
    }

    private static func parseFrontmatter(_ text: String) -> [String: String] {
        var values: [String: String] = [:]
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            guard let separator = line.firstIndex(of: ":") else { continue }
            let key = line[..<separator].trimmingCharacters(in: .whitespacesAndNewlines)
            let rawValue = line[line.index(after: separator)...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            values[key] = unquoteFrontmatterValue(rawValue)
        }
        return values
    }

    private static func unquoteFrontmatterValue(_ value: String) -> String {
        guard value.count >= 2,
              value.hasPrefix("\""),
              value.hasSuffix("\"")
        else {
            return value
        }

        return String(value.dropFirst().dropLast())
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\\"", with: "\"")
            .replacingOccurrences(of: "\\\\", with: "\\")
    }

    private static func yamlValue(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        return "\"\(escaped)\""
    }

    private static func pruneEmptyDirectories(
        startingAt directoryURL: URL,
        stopAt stopURL: URL,
        fileManager: FileManager
    ) {
        var currentURL = directoryURL.standardizedFileURL
        let normalizedStopURL = stopURL.standardizedFileURL

        while currentURL.path.hasPrefix(normalizedStopURL.path), currentURL != normalizedStopURL {
            guard let contents = try? fileManager.contentsOfDirectory(
                at: currentURL,
                includingPropertiesForKeys: nil,
                options: []
            ), contents.isEmpty else {
                return
            }

            try? fileManager.removeItem(at: currentURL)
            currentURL.deleteLastPathComponent()
        }
    }

    private enum ManagedFileIdentifier: Equatable {
        case note(UUID)
        case session(UUID)
    }

    private struct GroupingContext {
        let representativeNoteBySessionID: [UUID: VoxtNoteItem]
    }

    private struct FileManagerBox: @unchecked Sendable {
        let fileManager: FileManager
    }

    private static let dayFolderFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    private static let fileNameTimeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "HH:mm"
        return formatter
    }()

    private static let isoFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static func makeGroupingContext(notes: [VoxtNoteItem]) -> GroupingContext {
        var representativeNoteBySessionID: [UUID: VoxtNoteItem] = [:]
        for note in notes {
            guard let existing = representativeNoteBySessionID[note.sessionID] else {
                representativeNoteBySessionID[note.sessionID] = note
                continue
            }
            if noteSortOrder(note, existing) {
                representativeNoteBySessionID[note.sessionID] = note
            }
        }
        return GroupingContext(representativeNoteBySessionID: representativeNoteBySessionID)
    }

    private static func shouldReuseRecord(
        _ record: VoxtNoteObsidianExportRecord,
        for note: VoxtNoteItem,
        settings: ObsidianNoteSyncSettings,
        vaultURL: URL,
        fileManager: FileManager
    ) -> Bool {
        guard record.groupingMode == settings.groupingMode else { return false }
        if isRedundantNumericSuffixPath(record.relativeFilePath, vaultURL: vaultURL, fileManager: fileManager) {
            return false
        }
        return !isLegacyRelativeFilePath(record.relativeFilePath, groupingMode: record.groupingMode, note: note)
    }

    private static func isRedundantNumericSuffixPath(
        _ relativePath: String,
        vaultURL: URL,
        fileManager: FileManager
    ) -> Bool {
        guard let baseRelativePath = baseRelativePathByRemovingNumericSuffix(from: relativePath) else {
            return false
        }

        let baseURL = vaultURL.appendingPathComponent(baseRelativePath)
        return !fileManager.fileExists(atPath: baseURL.path)
    }

    private static func baseRelativePathByRemovingNumericSuffix(from relativePath: String) -> String? {
        let pattern = #" \((\d+)\)\.md$"#
        guard relativePath.range(of: pattern, options: .regularExpression) != nil else {
            return nil
        }
        return relativePath.replacingOccurrences(
            of: pattern,
            with: ".md",
            options: .regularExpression
        )
    }

    private static func isLegacyRelativeFilePath(
        _ relativePath: String,
        groupingMode: ObsidianNoteGroupingMode,
        note: VoxtNoteItem
    ) -> Bool {
        let lastComponent = URL(fileURLWithPath: relativePath).lastPathComponent
        switch groupingMode {
        case .session, .file:
            let expectedLegacyName = note.sessionID.uuidString + ".md"
            let expectedLegacyNoteName = note.id.uuidString + ".md"
            if lastComponent == expectedLegacyName || lastComponent == expectedLegacyNoteName {
                return true
            }
            if looksLikeOldDatePrefixedFileName(lastComponent) {
                return true
            }
            return lastComponent.range(
                of: #" \[(完成|已完成)\]\.md$"#,
                options: .regularExpression
            ) != nil
        case .daily:
            return false
        }
    }

    private static func looksLikeOldDatePrefixedFileName(_ value: String) -> Bool {
        value.range(
            of: #"^\d{4}-\d{2}-\d{2} \d{2}\.\d{2} - "#,
            options: .regularExpression
        ) != nil
    }

    nonisolated private static func noteSortOrder(_ lhs: VoxtNoteItem, _ rhs: VoxtNoteItem) -> Bool {
        if lhs.createdAt == rhs.createdAt {
            return lhs.id.uuidString < rhs.id.uuidString
        }
        return lhs.createdAt < rhs.createdAt
    }
}
