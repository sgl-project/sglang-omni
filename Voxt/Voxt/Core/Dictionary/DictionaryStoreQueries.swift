import Foundation

// Queries and matcher assembly share the Store's immutable dependencies.
// Reload completion and all mutations remain implemented by DictionaryStore.
extension DictionaryStore {
    func entries(
        filter: DictionaryFilter,
        query: String = "",
        limit: Int,
        offset: Int
    ) -> [DictionaryEntry] {
        if let repository,
           let pagedEntries = try? repository.entries(
            filter: filter,
            query: query,
            limit: limit,
            offset: offset
           ) {
            return pagedEntries
        }

        let filteredEntries = filteredEntries(for: filter)
        let searchedEntries = DictionaryEntryCollection.searchEntries(filteredEntries, query: query)
        guard offset < searchedEntries.count else { return [] }
        return Array(searchedEntries.dropFirst(offset).prefix(limit))
    }

    func entries(
        requiringReplacementTerms: Bool,
        query: String = "",
        limit: Int,
        offset: Int
    ) -> [DictionaryEntry] {
        if let repository,
           let pagedEntries = try? repository.entries(
            requiringReplacementTerms: requiringReplacementTerms,
            query: query,
            limit: limit,
            offset: offset
           ) {
            return pagedEntries
        }

        let filteredEntries = entries.filter { $0.replacementTerms.isEmpty != requiringReplacementTerms }
        let searchedEntries = DictionaryEntryCollection.searchEntries(filteredEntries, query: query)
        guard offset < searchedEntries.count else { return [] }
        return Array(searchedEntries.dropFirst(offset).prefix(limit))
    }

    func entryCount(filter: DictionaryFilter, query: String = "") -> Int {
        if let repository,
           let count = try? repository.entryCount(filter: filter, query: query) {
            return count
        }
        return DictionaryEntryCollection.searchEntries(filteredEntries(for: filter), query: query).count
    }

    func entryCount(requiringReplacementTerms: Bool, query: String = "") -> Int {
        if let repository,
           let count = try? repository.entryCount(
            requiringReplacementTerms: requiringReplacementTerms,
            query: query
           ) {
            return count
        }
        let filteredEntries = entries.filter { $0.replacementTerms.isEmpty != requiringReplacementTerms }
        return DictionaryEntryCollection.searchEntries(filteredEntries, query: query).count
    }

    func loadEntries(
        filter: DictionaryFilter,
        query: String = "",
        limit: Int,
        offset: Int,
        completion: @escaping (Int, [DictionaryEntry]) -> Void
    ) {
        guard let repository else {
            let searchedEntries = DictionaryEntryCollection.searchEntries(filteredEntries(for: filter), query: query)
            let page = offset < searchedEntries.count
                ? Array(searchedEntries.dropFirst(offset).prefix(limit))
                : []
            completion(searchedEntries.count, page)
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            let count = (try? repository.entryCount(filter: filter, query: query)) ?? 0
            let page = (try? repository.entries(filter: filter, query: query, limit: limit, offset: offset)) ?? []
            DispatchQueue.main.async {
                completion(count, page)
            }
        }
    }

    func loadEntries(
        requiringReplacementTerms: Bool,
        query: String = "",
        limit: Int,
        offset: Int,
        completion: @escaping (Int, [DictionaryEntry]) -> Void
    ) {
        guard let repository else {
            let filteredEntries = entries.filter { $0.replacementTerms.isEmpty != requiringReplacementTerms }
            let searchedEntries = DictionaryEntryCollection.searchEntries(filteredEntries, query: query)
            let page = offset < searchedEntries.count
                ? Array(searchedEntries.dropFirst(offset).prefix(limit))
                : []
            completion(searchedEntries.count, page)
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            let count = (try? repository.entryCount(
                requiringReplacementTerms: requiringReplacementTerms,
                query: query
            )) ?? 0
            let page = (try? repository.entries(
                requiringReplacementTerms: requiringReplacementTerms,
                query: query,
                limit: limit,
                offset: offset
            )) ?? []
            DispatchQueue.main.async {
                completion(count, page)
            }
        }
    }

    func allTerms(limit: Int? = nil) -> [String] {
        if let repository,
           let terms = try? repository.allTerms(limit: limit) {
            return terms
        }
        if let limit {
            return Array(entries.map(\.term).prefix(limit))
        }
        return entries.map(\.term)
    }

    func promptBiasTermsText(
        activeGroupID: UUID?,
        maxCount: Int = 24,
        maxCharacters: Int = 320
    ) -> String {
        if let repository,
           let entries = try? repository.activeEntriesForRemoteRequest(
            activeGroupID: activeGroupID,
            limit: max(maxCount * 4, maxCount)
           ) {
            return DictionaryEntryCollection.promptBiasTermsText(
                from: entries,
                activeGroupID: activeGroupID,
                maxCount: maxCount,
                maxCharacters: maxCharacters
            )
        }

        return DictionaryEntryCollection.promptBiasTermsText(
            from: entries,
            activeGroupID: activeGroupID,
            maxCount: maxCount,
            maxCharacters: maxCharacters
        )
    }


    func makeMatcherIfEnabled(for text: String, activeGroupID: UUID?) -> DictionaryMatcher? {
        guard completePendingReloadIfNeeded() else { return nil }
        let configuration = matcherConfiguration(for: activeGroupID, sourceText: text)
        guard !configuration.entries.isEmpty else { return nil }
        return DictionaryMatcher(
            entries: configuration.entries,
            blockedGlobalMatchKeys: configuration.blockedGlobalMatchKeys
        )
    }

    func correctionContext(for text: String, activeGroupID: UUID?) -> DictionaryCorrectionResult? {
        guard let matcher = makeMatcherIfEnabled(for: text, activeGroupID: activeGroupID) else { return nil }
        return matcher.applyCorrections(
            to: text,
            automaticReplacementEnabled: defaults.bool(forKey: AppPreferenceKey.dictionaryHighConfidenceCorrectionEnabled)
        )
    }

    func matchContext(for text: String, activeGroupID: UUID?) -> DictionaryCorrectionResult? {
        guard let matcher = makeMatcherIfEnabled(for: text, activeGroupID: activeGroupID) else { return nil }
        let candidates = matcher.recallCandidates(in: text)
        guard !candidates.isEmpty else { return nil }
        return DictionaryCorrectionResult(
            text: text,
            candidates: candidates,
            correctedTerms: [],
            correctionSnapshots: []
        )
    }

    func glossaryContext(for text: String, activeGroupID: UUID?) -> DictionaryPromptContext? {
        guard let matcher = makeMatcherIfEnabled(for: text, activeGroupID: activeGroupID) else { return nil }
        let context = matcher.promptContext(for: text)
        return context.isEmpty ? nil : context
    }

    func hasEntry(normalizedTerm: String, activeGroupID: UUID?) -> Bool {
        let normalized = normalizedTerm.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return false }

        if let repository,
           let hasEntry = try? repository.hasEntry(normalizedTerm: normalized, activeGroupID: activeGroupID) {
            return hasEntry
        }

        let configuration = matcherConfiguration(for: activeGroupID)
        return configuration.entries.contains { entry in
            entry.visibleMatchKeys(blockedKeys: configuration.blockedGlobalMatchKeys).contains(normalized)
        }
    }

    func activeEntriesForRemoteRequest(activeGroupID: UUID?, limit: Int = 5_000) -> [DictionaryEntry] {
        if let repository,
           let entries = try? repository.activeEntriesForRemoteRequest(
            activeGroupID: activeGroupID,
            limit: limit
           ) {
            return entries
        }

        return Array(
            DictionaryEntryCollection.activeEntriesForRemoteRequest(from: entries, activeGroupID: activeGroupID)
                .prefix(limit)
        )
    }


    private func matcherConfiguration(for activeGroupID: UUID?) -> (entries: [DictionaryEntry], blockedGlobalMatchKeys: Set<String>) {
        (
            entries: DictionaryEntryCollection.activeEntriesForRemoteRequest(
                from: entries,
                activeGroupID: activeGroupID
            ),
            blockedGlobalMatchKeys: DictionaryEntryCollection.blockedGlobalMatchKeys(
                from: entries,
                activeGroupID: activeGroupID
            )
        )
    }

    private func matcherConfiguration(
        for activeGroupID: UUID?,
        sourceText: String
    ) -> (entries: [DictionaryEntry], blockedGlobalMatchKeys: Set<String>) {
        if let repository,
           let candidates = try? repository.matchingEntries(
               sourceText: sourceText,
               activeGroupID: activeGroupID,
               limit: 200
           ) {
            return DictionaryEntryCollection.matcherConfiguration(
                for: candidates,
                activeGroupID: activeGroupID
            )
        }
        return matcherConfiguration(for: activeGroupID)
    }
}
