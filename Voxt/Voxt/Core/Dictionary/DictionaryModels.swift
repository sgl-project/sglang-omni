import Foundation

enum DictionaryEntrySource: String, Codable, CaseIterable {
    case manual
    case auto
    case codex
    case claude

    var titleKey: String {
        switch self {
        case .manual:
            return "Manual"
        case .auto:
            return "Auto"
        case .codex:
            return "Codex"
        case .claude:
            return "Claude"
        }
    }
}

enum DictionaryEntryStatus: String, Codable {
    case active
    case disabled
}

enum DictionaryVariantConfidence: String, Codable {
    case high
    case medium
    case low
}

enum DictionaryFilter: String, CaseIterable, Identifiable {
    case all
    case autoAdded
    case manualAdded

    var id: String { rawValue }

    var titleKey: String {
        switch self {
        case .all:
            return "All"
        case .autoAdded:
            return "Auto"
        case .manualAdded:
            return "Manual"
        }
    }
}

struct DictionaryCategory: Identifiable, Codable, Hashable {
    nonisolated static let defaultID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!
    nonisolated static let defaultName = "Default"

    let id: UUID
    var name: String
    var normalizedName: String
    var isDefault: Bool
    var isExpanded: Bool
    var sortOrder: Int
    var createdAt: Date
    var updatedAt: Date

    nonisolated init(
        id: UUID = UUID(),
        name: String,
        normalizedName: String? = nil,
        isDefault: Bool = false,
        isExpanded: Bool = true,
        sortOrder: Int = 0,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.normalizedName = normalizedName ?? DictionaryStore.normalizeTerm(name)
        self.isDefault = isDefault
        self.isExpanded = isExpanded
        self.sortOrder = sortOrder
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    nonisolated static var defaultCategory: DictionaryCategory {
        DictionaryCategory(
            id: defaultID,
            name: defaultName,
            normalizedName: DictionaryStore.normalizeTerm(defaultName),
            isDefault: true,
            isExpanded: true,
            sortOrder: 0
        )
    }
}

struct ObservedVariant: Identifiable, Codable, Hashable {
    let id: UUID
    var text: String
    var normalizedText: String
    var count: Int
    var lastSeenAt: Date
    var confidence: DictionaryVariantConfidence

    init(
        id: UUID = UUID(),
        text: String,
        normalizedText: String,
        count: Int = 1,
        lastSeenAt: Date = Date(),
        confidence: DictionaryVariantConfidence
    ) {
        self.id = id
        self.text = text
        self.normalizedText = normalizedText
        self.count = count
        self.lastSeenAt = lastSeenAt
        self.confidence = confidence
    }
}

struct DictionaryReplacementTerm: Identifiable, Codable, Hashable {
    let id: UUID
    var text: String
    var normalizedText: String

    init(
        id: UUID = UUID(),
        text: String,
        normalizedText: String
    ) {
        self.id = id
        self.text = text
        self.normalizedText = normalizedText
    }
}

struct DictionaryEntry: Identifiable, Codable, Hashable {
    let id: UUID
    var term: String
    var normalizedTerm: String
    var categoryID: UUID
    var categoryNameSnapshot: String?
    var groupID: UUID?
    var groupNameSnapshot: String?
    var source: DictionaryEntrySource
    var createdAt: Date
    var updatedAt: Date
    var lastMatchedAt: Date?
    var matchCount: Int
    var status: DictionaryEntryStatus
    var observedVariants: [ObservedVariant]
    var replacementTerms: [DictionaryReplacementTerm]

    enum CodingKeys: String, CodingKey {
        case id
        case term
        case normalizedTerm
        case categoryID
        case categoryNameSnapshot
        case groupID
        case groupNameSnapshot
        case source
        case createdAt
        case updatedAt
        case lastMatchedAt
        case matchCount
        case status
        case observedVariants
        case replacementTerms
    }

    init(
        id: UUID = UUID(),
        term: String,
        normalizedTerm: String,
        categoryID: UUID = DictionaryCategory.defaultID,
        categoryNameSnapshot: String? = DictionaryCategory.defaultName,
        groupID: UUID? = nil,
        groupNameSnapshot: String? = nil,
        source: DictionaryEntrySource,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        lastMatchedAt: Date? = nil,
        matchCount: Int = 0,
        status: DictionaryEntryStatus = .active,
        observedVariants: [ObservedVariant] = [],
        replacementTerms: [DictionaryReplacementTerm] = []
    ) {
        self.id = id
        self.term = term
        self.normalizedTerm = normalizedTerm
        self.categoryID = categoryID
        self.categoryNameSnapshot = categoryNameSnapshot
        self.groupID = groupID
        self.groupNameSnapshot = groupNameSnapshot
        self.source = source
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.lastMatchedAt = lastMatchedAt
        self.matchCount = matchCount
        self.status = status
        self.observedVariants = observedVariants
        self.replacementTerms = replacementTerms
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        term = try container.decode(String.self, forKey: .term)
        normalizedTerm = try container.decode(String.self, forKey: .normalizedTerm)
        categoryID = try container.decodeIfPresent(UUID.self, forKey: .categoryID) ?? DictionaryCategory.defaultID
        categoryNameSnapshot = try container.decodeIfPresent(String.self, forKey: .categoryNameSnapshot) ?? DictionaryCategory.defaultName
        groupID = try container.decodeIfPresent(UUID.self, forKey: .groupID)
        groupNameSnapshot = try container.decodeIfPresent(String.self, forKey: .groupNameSnapshot)
        source = try container.decode(DictionaryEntrySource.self, forKey: .source)
        createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? createdAt
        lastMatchedAt = try container.decodeIfPresent(Date.self, forKey: .lastMatchedAt)
        matchCount = try container.decodeIfPresent(Int.self, forKey: .matchCount) ?? 0
        status = try container.decodeIfPresent(DictionaryEntryStatus.self, forKey: .status) ?? .active
        observedVariants = try container.decodeIfPresent([ObservedVariant].self, forKey: .observedVariants) ?? []
        replacementTerms = try container.decodeIfPresent([DictionaryReplacementTerm].self, forKey: .replacementTerms) ?? []
    }

    var matchKeys: [String] {
        [normalizedTerm] + replacementTerms.map(\.normalizedText)
    }

    func visibleMatchKeys(blockedKeys: Set<String>) -> [String] {
        if groupID == nil {
            return matchKeys.filter { !blockedKeys.contains($0) }
        }
        return matchKeys
    }
}

enum DictionaryMatchSource: String, Hashable {
    case term
    case replacementTerm
    case observedVariant
}

enum DictionaryMatchReason: String, Codable {
    case exactTerm
    case exactVariant
    case exactWindow
    case fuzzyWindow
}

struct DictionaryMatchCandidate: Identifiable, Hashable {
    let entryID: UUID
    let term: String
    let matchedText: String
    let normalizedMatchedText: String
    let score: Double
    let reason: DictionaryMatchReason
    let source: DictionaryMatchSource
    let matchRange: NSRange?

    nonisolated var id: String {
        let location = matchRange?.location ?? -1
        let length = matchRange?.length ?? 0
        return "\(entryID.uuidString)|\(normalizedMatchedText)|\(reason.rawValue)|\(source.rawValue)|\(location)|\(length)"
    }

    nonisolated var allowsAutomaticReplacement: Bool {
        if source == .replacementTerm {
            return true
        }

        switch reason {
        case .exactVariant:
            return true
        case .exactWindow:
            return score >= 0.985
        case .fuzzyWindow:
            return score >= 0.97 && normalizedMatchedText.count >= 5
        case .exactTerm:
            return false
        }
    }

    nonisolated var shouldPersistObservedVariant: Bool {
        source != .replacementTerm && reason != .exactTerm
    }
}

struct DictionaryPromptContext {
    let entries: [DictionaryEntry]
    let candidates: [DictionaryMatchCandidate]

    var isEmpty: Bool {
        entries.isEmpty || candidates.isEmpty
    }

    func glossaryText(limit: Int = 12) -> String {
        guard !isEmpty else { return "" }

        let entriesByID = Dictionary(uniqueKeysWithValues: entries.map { ($0.id, $0) })
        var seen = Set<UUID>()
        var lines: [String] = []
        for candidate in candidates.sorted(by: { $0.score > $1.score }) {
            guard let entry = entriesByID[candidate.entryID] else { continue }
            guard seen.insert(entry.id).inserted else { continue }
            lines.append("- \(entry.term)")
            if lines.count >= limit {
                break
            }
        }
        return lines.joined(separator: "\n")
    }

    func glossaryText(for purpose: DictionaryGlossaryPurpose) -> String {
        glossaryText(policy: purpose.selectionPolicy)
    }

    func glossaryText(policy: DictionaryGlossarySelectionPolicy) -> String {
        guard !isEmpty, policy.maxTerms > 0, policy.maxCharacters > 0 else { return "" }

        let entriesByID = Dictionary(uniqueKeysWithValues: entries.map { ($0.id, $0) })
        var seen = Set<UUID>()
        var lines: [String] = []
        var characterCount = 0

        for candidate in candidates.sorted(by: { $0.score > $1.score }) {
            guard let entry = entriesByID[candidate.entryID] else { continue }
            guard seen.insert(entry.id).inserted else { continue }

            let line = "- \(entry.term)"
            let separatorCost = lines.isEmpty ? 0 : 1
            let nextCharacterCount = characterCount + separatorCost + line.count

            if !lines.isEmpty && nextCharacterCount > policy.maxCharacters {
                break
            }
            if lines.isEmpty && line.count > policy.maxCharacters {
                lines.append(line)
                break
            }

            lines.append(line)
            characterCount = nextCharacterCount

            if lines.count >= policy.maxTerms {
                break
            }
        }

        return lines.joined(separator: "\n")
    }
}

struct DictionaryCorrectionResult {
    let text: String
    let candidates: [DictionaryMatchCandidate]
    let correctedTerms: [String]
    let correctionSnapshots: [DictionaryCorrectionSnapshot]
}

struct DictionaryCorrectionSnapshot: Codable, Hashable {
    let originalText: String
    let correctedText: String
    let finalLocation: Int
    let finalLength: Int
}

struct DictionaryImportResult: Equatable {
    let addedCount: Int
    let skippedCount: Int
}

struct DictionaryEntryUpsertResult: Equatable {
    let term: String
    let added: Bool
    let reinforcedCount: Int
}

enum DictionaryStoreError: LocalizedError {
    case dataUnavailable
    case emptyTerm
    case emptyCategoryName
    case duplicateCategory
    case duplicateTerm
    case replacementMatchesDictionaryTerm
    case duplicateReplacementTerm(String)

    var errorDescription: String? {
        switch self {
        case .dataUnavailable:
            return AppLocalization.localizedString("Dictionary data is temporarily unavailable. Please try again.")
        case .emptyTerm:
            return AppLocalization.localizedString("Dictionary term cannot be empty.")
        case .emptyCategoryName:
            return AppLocalization.localizedString("Dictionary category name cannot be empty.")
        case .duplicateCategory:
            return AppLocalization.localizedString("This dictionary category already exists.")
        case .duplicateTerm:
            return AppLocalization.localizedString("This term already exists in the dictionary.")
        case .replacementMatchesDictionaryTerm:
            return AppLocalization.localizedString("Replacement match term cannot be the same as the dictionary term.")
        case .duplicateReplacementTerm(let term):
            return AppLocalization.format(
                "This replacement match term already exists in the dictionary: %@.",
                term
            )
        }
    }
}
