import Foundation

nonisolated enum DictionarySuggestionSourceContext: String, Codable, Sendable {
    case history
    case correction
    case repeatObservation
}

nonisolated enum DictionarySuggestionStatus: String, Codable, Sendable {
    case pending
    case dismissed
    case added
}

struct DictionarySuggestionSnapshot: Identifiable, Codable, Hashable {
    let term: String
    let normalizedTerm: String
    let groupID: UUID?
    let groupNameSnapshot: String?

    var id: String {
        "\(normalizedTerm)|\(groupID?.uuidString ?? "global")"
    }
}

nonisolated struct DictionarySuggestion: Identifiable, Codable, Hashable, Sendable {
    let id: UUID
    var term: String
    var normalizedTerm: String
    var sourceContext: DictionarySuggestionSourceContext
    var status: DictionarySuggestionStatus
    var firstSeenAt: Date
    var lastSeenAt: Date
    var seenCount: Int
    var lastHistoryEntryID: UUID?
    var groupID: UUID?
    var groupNameSnapshot: String?
    var evidenceSamples: [String]

    init(
        id: UUID = UUID(),
        term: String,
        normalizedTerm: String,
        sourceContext: DictionarySuggestionSourceContext,
        status: DictionarySuggestionStatus = .pending,
        firstSeenAt: Date = Date(),
        lastSeenAt: Date = Date(),
        seenCount: Int = 1,
        lastHistoryEntryID: UUID? = nil,
        groupID: UUID? = nil,
        groupNameSnapshot: String? = nil,
        evidenceSamples: [String] = []
    ) {
        self.id = id
        self.term = term
        self.normalizedTerm = normalizedTerm
        self.sourceContext = sourceContext
        self.status = status
        self.firstSeenAt = firstSeenAt
        self.lastSeenAt = lastSeenAt
        self.seenCount = seenCount
        self.lastHistoryEntryID = lastHistoryEntryID
        self.groupID = groupID
        self.groupNameSnapshot = groupNameSnapshot
        self.evidenceSamples = evidenceSamples
    }
}

struct DictionaryHistoryScanCheckpoint: Codable, Equatable {
    let lastProcessedAt: Date
    let lastHistoryEntryID: UUID
}

struct DictionaryHistoryScanProgress: Equatable {
    var isRunning = false
    var isCancellationRequested = false
    var processedCount = 0
    var totalCount = 0
    var newSuggestionCount = 0
    var duplicateCount = 0
    var lastProcessedCount = 0
    var lastNewSuggestionCount = 0
    var lastDuplicateCount = 0
    var lastRunAt: Date?
    var errorMessage: String?
}

struct DictionaryHistoryScanCandidate: Hashable {
    let term: String
    let historyEntryIDs: [UUID]
    let groupID: UUID?
    let groupNameSnapshot: String?
    let evidenceSample: String
}

struct DictionaryHistoryScanApplyResult {
    let newSuggestionCount: Int
    let duplicateCount: Int
}
