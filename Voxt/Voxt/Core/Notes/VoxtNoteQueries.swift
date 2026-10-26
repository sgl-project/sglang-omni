// VoxtNoteQueries.swift
// Computes note sections and ordering without owning store state.

import Foundation

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

nonisolated struct VoxtNoteManualOrderUpdate {
    let noteID: UUID
    let manualOrder: Int64
}

nonisolated struct VoxtNoteManualOrderAllocation {
    let nextOrder: Int64?
    let updates: [VoxtNoteManualOrderUpdate]
}

nonisolated enum VoxtNoteQueries {
    static func orderedItems(_ items: [VoxtNoteItem], for status: VoxtNoteStatus) -> [VoxtNoteItem] {
        items
            .filter { $0.status == status }
            .sorted(by: status == .done ? completedNoteSortOrder : noteSortOrder)
    }

    static func snapshot(_ items: [VoxtNoteItem], for scope: VoxtNoteScope) -> VoxtNoteScopeSnapshot {
        let sections = scope.statuses.compactMap { status -> VoxtNoteSectionSnapshot? in
            let statusItems = orderedItems(items, for: status)
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

    static func publicationOrder(_ items: [VoxtNoteItem]) -> [VoxtNoteItem] {
        items.sorted { lhs, rhs in
            if lhs.createdAt != rhs.createdAt { return lhs.createdAt > rhs.createdAt }
            return lhs.id.uuidString < rhs.id.uuidString
        }
    }

    static func reorder(
        noteID: UUID,
        relativeTo targetID: UUID,
        in items: [VoxtNoteItem]
    ) -> [VoxtNoteManualOrderUpdate]? {
        guard noteID != targetID,
              let note = items.first(where: { $0.id == noteID }),
              let target = items.first(where: { $0.id == targetID }),
              note.status == target.status,
              note.priority == target.priority else {
            return nil
        }

        var group = orderedItems(items, for: note.status).filter { $0.priority == note.priority }
        guard let sourceIndex = group.firstIndex(where: { $0.id == noteID }),
              let targetIndex = group.firstIndex(where: { $0.id == targetID }) else {
            return nil
        }

        let moved = group.remove(at: sourceIndex)
        group.insert(moved, at: min(targetIndex, group.count))
        return manualOrders(for: group)
    }

    static func nextManualOrder(
        in items: [VoxtNoteItem],
        status: VoxtNoteStatus,
        priority: VoxtNotePriority,
        excluding excludedID: UUID? = nil
    ) -> VoxtNoteManualOrderAllocation {
        let group = items.filter {
            $0.id != excludedID && $0.status == status && $0.priority == priority
        }
        guard let maximum = group.compactMap(\.manualOrder).max() else {
            return VoxtNoteManualOrderAllocation(nextOrder: nil, updates: [])
        }
        guard maximum == .max else {
            return VoxtNoteManualOrderAllocation(nextOrder: maximum + 1, updates: [])
        }

        let orderedGroup = group.sorted(by: noteSortOrder)
        return VoxtNoteManualOrderAllocation(
            nextOrder: Int64(orderedGroup.count + 1),
            updates: manualOrders(for: orderedGroup)
        )
    }

    private static func manualOrders(for group: [VoxtNoteItem]) -> [VoxtNoteManualOrderUpdate] {
        group.enumerated().map { index, item in
            VoxtNoteManualOrderUpdate(noteID: item.id, manualOrder: Int64(group.count - index))
        }
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
