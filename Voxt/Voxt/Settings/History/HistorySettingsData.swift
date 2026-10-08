// HistorySettingsData.swift
// Provides History Settings Data for history settings.

import Foundation

enum HistoryContentEmptyState: Equatable {
    case none
    case noNotes
    case noHistory
    case noEntriesInCategory

    var localizedKey: String? {
        switch self {
        case .none:
            return nil
        case .noNotes:
            return "No notes yet."
        case .noHistory:
            return "No history yet."
        case .noEntriesInCategory:
            return "No entries in this category yet."
        }
    }
}

enum HistorySettingsData {
    static let noteSectionOrder: [VoxtNoteStatus] = [.inProgress, .todo, .done, .backlog]
    static let linearNoteSectionOrder: [VoxtNoteStatus] = [.backlog, .inProgress, .todo, .done]

    static func filteredEntries(
        for filter: HistoryFilterTab,
        allEntries: [TranscriptionHistoryEntry]
    ) -> [TranscriptionHistoryEntry] {
        switch filter {
        case .transcription:
            return allEntries.filter { $0.kind == .normal }
        case .translation:
            return allEntries.filter { $0.kind == .translation }
        case .transcript:
            return allEntries.filter { $0.kind == .transcript }
        case .rewrite:
            return allEntries.filter { $0.kind == .rewrite }
        case .note:
            return []
        }
    }

    static func searchEntries(
        _ entries: [TranscriptionHistoryEntry],
        query: String
    ) -> [TranscriptionHistoryEntry] {
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedQuery = DictionaryStore.normalizeTerm(query)
        guard !trimmedQuery.isEmpty || !normalizedQuery.isEmpty else { return entries }

        return entries.filter { entry in
            if entry.text.localizedCaseInsensitiveContains(trimmedQuery) {
                return true
            }
            if entry.displayTitle?.localizedCaseInsensitiveContains(trimmedQuery) == true {
                return true
            }
            if entry.focusedAppName?.localizedCaseInsensitiveContains(trimmedQuery) == true {
                return true
            }
            if !normalizedQuery.isEmpty {
                return entry.dictionaryHitTerms.contains { DictionaryStore.normalizeTerm($0).contains(normalizedQuery) }
                    || entry.dictionaryCorrectedTerms.contains { DictionaryStore.normalizeTerm($0).contains(normalizedQuery) }
            }
            return false
        }
    }

    static func searchNotes(
        _ notes: [VoxtNoteItem],
        query: String
    ) -> [VoxtNoteItem] {
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedQuery.isEmpty else { return notes }

        return notes.filter {
            $0.title.localizedCaseInsensitiveContains(trimmedQuery)
                || $0.text.localizedCaseInsensitiveContains(trimmedQuery)
        }
    }

    static func filteredNotes(
        _ notes: [VoxtNoteItem],
        statuses: Set<VoxtNoteStatus>,
        query: String
    ) -> [VoxtNoteItem] {
        searchNotes(notes, query: query)
            .filter { statuses.contains($0.status) }
            .sorted { lhs, rhs in
                if lhs.updatedAt != rhs.updatedAt {
                    return lhs.updatedAt > rhs.updatedAt
                }
                return lhs.id.uuidString < rhs.id.uuidString
            }
    }

    static func noteSections(from notes: [VoxtNoteItem]) -> [VoxtNoteSectionSnapshot] {
        noteSectionOrder.compactMap { status in
            let items = notes.filter { $0.status == status }
            guard !items.isEmpty else { return nil }
            return VoxtNoteSectionSnapshot(status: status, items: items)
        }
    }

    static func visibleLinearNotes(
        from notes: [VoxtNoteItem],
        status: VoxtNoteStatus,
        completedVisibleLimit: Int
    ) -> [VoxtNoteItem] {
        let matchingNotes = notes.filter { $0.status == status }
        guard status == .done else { return matchingNotes }
        return Array(matchingNotes.prefix(max(0, completedVisibleLimit)))
    }

    static func toggledNoteStatuses(
        _ current: Set<VoxtNoteStatus>,
        status: VoxtNoteStatus
    ) -> Set<VoxtNoteStatus> {
        if current.contains(status) {
            guard current.count > 1 else { return current }
            return current.subtracting([status])
        }
        return current.union([status])
    }

    static func visibleEntries<T>(
        from items: [T],
        visibleLimit: Int
    ) -> [T] {
        Array(items.prefix(max(0, visibleLimit)))
    }

    static func hasMoreItems<T>(
        in items: [T],
        visibleLimit: Int
    ) -> Bool {
        visibleLimit < items.count
    }

    static func nextVisibleLimit(
        currentLimit: Int,
        pageSize: Int,
        totalCount: Int
    ) -> Int {
        min(currentLimit + pageSize, totalCount)
    }

    static func normalizedVisibleLimit(
        currentLimit: Int,
        pageSize: Int,
        totalCount: Int
    ) -> Int {
        min(max(currentLimit, pageSize), max(totalCount, pageSize))
    }

    static func emptyState(
        selectedFilter: HistoryFilterTab,
        allEntries: [TranscriptionHistoryEntry],
        filteredEntries: [TranscriptionHistoryEntry],
        notes: [VoxtNoteItem]
    ) -> HistoryContentEmptyState {
        if selectedFilter == .note {
            return notes.isEmpty ? .noNotes : .none
        }
        if allEntries.isEmpty {
            return .noHistory
        }
        return filteredEntries.isEmpty ? .noEntriesInCategory : .none
    }
}

enum HistoryBulkDeletionTarget: Identifiable {
    case history(HistoryFilterTab)
    case notes

    var id: String {
        switch self {
        case .history(let filter):
            return "history-\(filter.id)"
        case .notes:
            return "notes"
        }
    }
}

enum HistoryListItem: Identifiable {
    case dayHeader(Date)
    case entry(TranscriptionHistoryEntry)
    case meetingEntry(TranscriptionHistoryListEntry)

    var id: String {
        switch self {
        case .dayHeader(let date):
            return "day-\(date.timeIntervalSince1970)"
        case .entry(let entry):
            return "entry-\(entry.id.uuidString)"
        case .meetingEntry(let entry):
            return "entry-\(entry.id.uuidString)"
        }
    }

    var createdAt: Date {
        switch self {
        case .dayHeader(let date):
            return date
        case .entry(let entry):
            return entry.createdAt
        case .meetingEntry(let entry):
            return entry.createdAt
        }
    }
}
