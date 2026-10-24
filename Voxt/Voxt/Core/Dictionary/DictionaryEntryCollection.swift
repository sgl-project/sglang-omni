// DictionaryEntryCollection.swift
// Provides Dictionary Entry Collection for dictionary matching and learning.

import Foundation

enum DictionaryEntryCollection {
    static let asrPromptTermLimit = 500
    static let asrPromptCharacterLimit = 8_000

    static func sortedEntries(_ values: [DictionaryEntry]) -> [DictionaryEntry] {
        values.sorted {
            if $0.updatedAt == $1.updatedAt {
                return $0.term.localizedCaseInsensitiveCompare($1.term) == .orderedAscending
            }
            return $0.updatedAt > $1.updatedAt
        }
    }

    static func filteredEntriesCache(for values: [DictionaryEntry]) -> [DictionaryFilter: [DictionaryEntry]] {
        [
            .all: values,
            .autoAdded: values.filter { $0.source == .auto },
            .manualAdded: values.filter { $0.source == .manual }
        ]
    }

    static func searchEntries(_ entries: [DictionaryEntry], query: String) -> [DictionaryEntry] {
        let normalizedQuery = DictionaryStore.normalizeTerm(query)
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedQuery.isEmpty || !trimmedQuery.isEmpty else { return entries }

        return entries.filter { entry in
            if !normalizedQuery.isEmpty {
                if entry.normalizedTerm.contains(normalizedQuery) {
                    return true
                }
                if entry.replacementTerms.contains(where: { $0.normalizedText.contains(normalizedQuery) }) {
                    return true
                }
                if entry.observedVariants.contains(where: { $0.normalizedText.contains(normalizedQuery) }) {
                    return true
                }
            }

            return entry.term.localizedCaseInsensitiveContains(trimmedQuery)
                || entry.groupNameSnapshot?.localizedCaseInsensitiveContains(trimmedQuery) == true
        }
    }

    static func activeEntriesForRemoteRequest(
        from entries: [DictionaryEntry],
        activeGroupID: UUID?
    ) -> [DictionaryEntry] {
        matcherConfiguration(for: entries, activeGroupID: activeGroupID).entries.filter { $0.status == .active }
    }

    static func blockedGlobalMatchKeys(
        from entries: [DictionaryEntry],
        activeGroupID: UUID?
    ) -> Set<String> {
        matcherConfiguration(for: entries, activeGroupID: activeGroupID).blockedGlobalMatchKeys
    }

    static func promptBiasTermsText(
        from entries: [DictionaryEntry],
        activeGroupID: UUID?,
        maxCount: Int = 24,
        maxCharacters: Int = 320
    ) -> String {
        let candidates = activeEntriesForRemoteRequest(from: entries, activeGroupID: activeGroupID)
        guard !candidates.isEmpty, maxCount > 0, maxCharacters > 0 else { return "" }

        let sortedCandidates = candidates.sorted {
            if $0.matchCount != $1.matchCount {
                return $0.matchCount > $1.matchCount
            }
            switch ($0.lastMatchedAt, $1.lastMatchedAt) {
            case let (lhs?, rhs?) where lhs != rhs:
                return lhs > rhs
            case (.some, .none):
                return true
            case (.none, .some):
                return false
            default:
                break
            }
            if $0.updatedAt != $1.updatedAt {
                return $0.updatedAt > $1.updatedAt
            }
            return $0.term.localizedCaseInsensitiveCompare($1.term) == .orderedAscending
        }

        var seen = Set<String>()
        var selectedTerms: [String] = []
        var totalCharacters = 0

        for entry in sortedCandidates {
            let trimmed = entry.term.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            guard seen.insert(entry.normalizedTerm).inserted else { continue }

            let projectedCharacters = totalCharacters + trimmed.count + (selectedTerms.isEmpty ? 0 : 1)
            if !selectedTerms.isEmpty && projectedCharacters > maxCharacters {
                break
            }

            selectedTerms.append(trimmed)
            totalCharacters = projectedCharacters

            if selectedTerms.count >= maxCount || totalCharacters >= maxCharacters {
                break
            }
        }

        return selectedTerms.joined(separator: "\n")
    }

    static func asrPromptTermsText(
        from entries: [DictionaryEntry],
        maxCount: Int = asrPromptTermLimit,
        maxCharacters: Int = asrPromptCharacterLimit
    ) -> String {
        let candidates = entries.filter { $0.status == .active }
        return rankedTermsText(
            from: candidates,
            maxCount: maxCount,
            maxCharacters: maxCharacters
        )
    }

    static func matcherConfiguration(
        for entries: [DictionaryEntry],
        activeGroupID: UUID?
    ) -> (entries: [DictionaryEntry], blockedGlobalMatchKeys: Set<String>) {
        let globals = entries.filter { $0.status == .active && $0.groupID == nil }
        guard let activeGroupID else {
            return (globals, [])
        }

        let scoped = entries.filter { $0.status == .active && $0.groupID == activeGroupID }
        let blockedKeys = Set(scoped.flatMap(\.matchKeys))
        return (scoped + globals, blockedKeys)
    }

    private static func rankedTermsText(
        from entries: [DictionaryEntry],
        maxCount: Int,
        maxCharacters: Int
    ) -> String {
        guard !entries.isEmpty, maxCount > 0, maxCharacters > 0 else { return "" }

        let sortedCandidates = entries.sorted {
            if $0.matchCount != $1.matchCount {
                return $0.matchCount > $1.matchCount
            }
            switch ($0.lastMatchedAt, $1.lastMatchedAt) {
            case let (lhs?, rhs?) where lhs != rhs:
                return lhs > rhs
            case (.some, .none):
                return true
            case (.none, .some):
                return false
            default:
                break
            }
            if $0.updatedAt != $1.updatedAt {
                return $0.updatedAt > $1.updatedAt
            }
            return $0.term.localizedCaseInsensitiveCompare($1.term) == .orderedAscending
        }

        var seen = Set<String>()
        var selectedTerms: [String] = []
        var totalCharacters = 0

        for entry in sortedCandidates {
            let trimmed = entry.term.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            guard seen.insert(entry.normalizedTerm).inserted else { continue }

            let projectedCharacters = totalCharacters + trimmed.count + (selectedTerms.isEmpty ? 0 : 1)
            if !selectedTerms.isEmpty && projectedCharacters > maxCharacters {
                break
            }

            selectedTerms.append(trimmed)
            totalCharacters = projectedCharacters

            if selectedTerms.count >= maxCount || totalCharacters >= maxCharacters {
                break
            }
        }

        return selectedTerms.joined(separator: "\n")
    }
}
