// VoxtNoteModels.swift
// Defines note values, persistence projection and title normalization.

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
    static func normalizedTitle(_ title: String) -> String {
        title
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

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
