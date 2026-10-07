// DictionaryStore.swift
// Provides Dictionary Store for dictionary matching and learning.

import Foundation
import Combine

@MainActor
final class DictionaryStore: ObservableObject {
    @Published private(set) var entries: [DictionaryEntry] = []
    @Published private(set) var categories: [DictionaryCategory] = [DictionaryCategory.defaultCategory]
    @Published private(set) var isLoading = false

    let defaults: UserDefaults
    private let fileManager: FileManager
    private var reloadGeneration = 0
    private var filteredEntriesCache: [DictionaryFilter: [DictionaryEntry]] = [:]
    private var validationIndex = DictionaryValidationIndex(entries: [])
    private let persistenceEnabled: Bool
    let repository: DictionaryRepositoryProtocol?

    convenience init() {
        self.init(defaults: .standard, fileManager: .default)
    }

    init(
        defaults: UserDefaults,
        fileManager: FileManager,
        initialEntries: [DictionaryEntry]? = nil,
        persistenceEnabled: Bool = true,
        repository: DictionaryRepositoryProtocol? = nil
    ) {
        self.defaults = defaults
        self.fileManager = fileManager
        self.persistenceEnabled = persistenceEnabled
        self.repository = persistenceEnabled ? (repository ?? DictionaryRepository()) : repository
        if let initialEntries {
            applyReloadedEntries(initialEntries)
        } else if repository == nil, persistenceEnabled {
            reloadAsync()
        } else {
            reload()
        }
    }

    @discardableResult
    func reload() -> Bool {
        invalidatePendingReload()
        do {
            if let repository {
                let decoded = try repository.allEntries()
                let decodedCategories = try repository.allCategories()
                if !decoded.isEmpty || !legacyDictionaryFileExists() {
                    applyReloadedCategories(decodedCategories)
                    applyReloadedEntries(decoded)
                    return true
                }
            }

            let url = try dictionaryFileURL()
            guard fileManager.fileExists(atPath: url.path) else {
                applyReloadedEntries([])
                return true
            }
            let data = try Data(contentsOf: url)
            let decoded = try JSONDecoder().decode([DictionaryEntry].self, from: data)
            applyReloadedEntries(decoded)
            return true
        } catch {
            VoxtLog.dictionary("Dictionary reload failed; preserving the current snapshot. error=\(error.localizedDescription)")
            return false
        }
    }

    func reloadAsync() {
        reloadGeneration += 1
        let generation = reloadGeneration
        isLoading = true

        let repository = repository
        let url: URL?
        do {
            url = try dictionaryFileURL()
        } catch {
            isLoading = false
            applyReloadedEntries([])
            return
        }

        DispatchQueue.global(qos: .utility).async { [weak self, url] in
            let decodedEntries: [DictionaryEntry]
            if let repository,
               let repositoryEntries = try? repository.allEntries(),
               !repositoryEntries.isEmpty || url.map({ !FileManager.default.fileExists(atPath: $0.path) }) == true {
                let repositoryCategories = (try? repository.allCategories()) ?? [DictionaryCategory.defaultCategory]
                DispatchQueue.main.async { [weak self] in
                    guard let self, generation == self.reloadGeneration else { return }
                    self.applyReloadedCategories(repositoryCategories)
                }
                decodedEntries = repositoryEntries
            } else if let url, FileManager.default.fileExists(atPath: url.path) {
                do {
                    let data = try Data(contentsOf: url)
                    decodedEntries = try JSONDecoder().decode([DictionaryEntry].self, from: data)
                } catch {
                    decodedEntries = []
                }
            } else {
                decodedEntries = []
            }

            DispatchQueue.main.async {
                guard let self, generation == self.reloadGeneration else { return }
                self.isLoading = false
                self.applyReloadedEntries(decodedEntries)
            }
        }
    }

    func filteredEntries(for filter: DictionaryFilter) -> [DictionaryEntry] {
        filteredEntriesCache[filter] ?? entries
    }

    func hotwordEntriesByCategory(
        query: String = ""
    ) -> [(category: DictionaryCategory, entries: [DictionaryEntry])] {
        let filtered = DictionaryEntryCollection.searchEntries(
            entries.filter { $0.replacementTerms.isEmpty },
            query: query
        )
        let entriesByCategoryID = Dictionary(grouping: filtered, by: \.categoryID)
        return resolvedCategories().map { category in
            (
                category,
                DictionaryEntryCollection.sortedEntries(entriesByCategoryID[category.id] ?? [])
            )
        }
    }

    func categoryName(for categoryID: UUID) -> String {
        resolvedCategories().first(where: { $0.id == categoryID })?.name
            ?? entries.first(where: { $0.categoryID == categoryID })?.categoryNameSnapshot
            ?? DictionaryCategory.defaultName
    }

    func createCategory(name: String) throws -> DictionaryCategory {
        guard completePendingReloadIfNeeded() else { throw DictionaryStoreError.dataUnavailable }
        let preparedName = try prepareCategoryName(name)
        let now = Date()
        let category = DictionaryCategory(
            name: preparedName.display,
            normalizedName: preparedName.normalized,
            isDefault: false,
            isExpanded: true,
            sortOrder: (categories.map(\.sortOrder).max() ?? 0) + 1,
            createdAt: now,
            updatedAt: now
        )
        try upsertPersistedCategory(category)
        replaceCategories(categories + [category])
        return category
    }

    func ensureCategory(id: UUID?, name: String?) -> DictionaryCategory {
        guard completePendingReloadIfNeeded() else { return resolvedDefaultCategory() }
        guard let id else { return resolvedDefaultCategory() }
        if let existing = categories.first(where: { $0.id == id }) {
            return existing
        }
        let displayName = name?.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedName = (displayName?.isEmpty == false ? displayName : nil) ?? AppLocalization.localizedString("Imported Category")
        let category = DictionaryCategory(
            id: id,
            name: resolvedName,
            normalizedName: Self.normalizeTerm(resolvedName),
            isDefault: false,
            isExpanded: true,
            sortOrder: (categories.map(\.sortOrder).max() ?? 0) + 1
        )
        try? upsertPersistedCategory(category)
        replaceCategories(categories + [category])
        return category
    }

    func updateCategory(id: UUID, name: String) throws {
        guard completePendingReloadIfNeeded() else { throw DictionaryStoreError.dataUnavailable }
        guard let index = categories.firstIndex(where: { $0.id == id }) else { return }
        let preparedName = try prepareCategoryName(name, excluding: id)
        var category = categories[index]
        category.name = preparedName.display
        category.normalizedName = preparedName.normalized
        category.updatedAt = Date()
        try upsertPersistedCategory(category)

        var updatedCategories = categories
        updatedCategories[index] = category
        replaceCategories(updatedCategories)
        refreshCategoryNameSnapshot(category)
    }

    func deleteCategory(id: UUID, deleteEntries: Bool = false) {
        guard completePendingReloadIfNeeded() else { return }
        guard id != DictionaryCategory.defaultID else { return }
        invalidatePendingReload()
        let targetCategory = categories.first(where: { $0.id == id })
        let fallback = resolvedDefaultCategory()
        if deleteEntries {
            let deletedIDs = Set(entries.filter { $0.categoryID == id }.map(\.id))
            if let repository {
                for entryID in deletedIDs {
                    try? repository.delete(id: entryID)
                }
                try? repository.deleteCategory(id: id, moveEntriesTo: nil)
            }
            replaceEntries(entries.filter { !deletedIDs.contains($0.id) }, sort: false)
        } else {
            let movedEntries = entries.map { entry -> DictionaryEntry in
                guard entry.categoryID == id else { return entry }
                var updated = entry
                updated.categoryID = fallback.id
                updated.categoryNameSnapshot = fallback.name
                updated.updatedAt = Date()
                return updated
            }
            try? repository?.deleteCategory(id: id, moveEntriesTo: fallback)
            replaceEntries(movedEntries)
        }
        replaceCategories(categories.filter { $0.id != id })
        if let targetCategory {
            VoxtLog.dictionary("Dictionary category deleted. category=\(targetCategory.name), deleteEntries=\(deleteEntries)")
        }
    }

    func createManualEntry(
        term: String,
        replacementTerms: [String] = [],
        categoryID: UUID = DictionaryCategory.defaultID,
        categoryNameSnapshot: String? = DictionaryCategory.defaultName,
        groupID: UUID?,
        groupNameSnapshot: String?
    ) throws {
        if !replacementTerms.isEmpty {
            try createManualReplacementEntry(
                term: term,
                replacementTerms: replacementTerms,
                categoryID: categoryID,
                categoryNameSnapshot: categoryNameSnapshot,
                groupID: groupID,
                groupNameSnapshot: groupNameSnapshot
            )
            return
        }

        _ = try createOrReinforceManualEntry(
            term: term,
            replacementTerms: replacementTerms,
            categoryID: categoryID,
            categoryNameSnapshot: categoryNameSnapshot,
            groupID: groupID,
            groupNameSnapshot: groupNameSnapshot
        )
    }

    func createManualReplacementEntry(
        term: String,
        replacementTerms: [String],
        categoryID: UUID = DictionaryCategory.defaultID,
        categoryNameSnapshot: String? = DictionaryCategory.defaultName,
        groupID: UUID?,
        groupNameSnapshot: String?
    ) throws {
        guard completePendingReloadIfNeeded() else { throw DictionaryStoreError.dataUnavailable }
        let trimmedTerm = term.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = Self.normalizeTerm(trimmedTerm)
        guard !trimmedTerm.isEmpty, !normalized.isEmpty else {
            throw DictionaryStoreError.emptyTerm
        }

        if let existingIndex = existingTermIndex(normalizedTerm: normalized, groupID: groupID) {
            let existingEntry = entries[existingIndex]
            let combinedReplacementTerms = existingEntry.replacementTerms.map(\.text) + replacementTerms
            let prepared = try prepareEntryInput(
                term: existingEntry.term,
                replacementTerms: combinedReplacementTerms,
                groupID: existingEntry.groupID,
                excluding: existingEntry.id
            )
            var updatedEntry = existingEntry
            updatedEntry.replacementTerms = prepared.replacementTerms
            updatedEntry.updatedAt = Date()

            let reservedKeys = Set([existingEntry.normalizedTerm] + prepared.replacementTerms.map(\.normalizedText))
            updatedEntry.observedVariants.removeAll { reservedKeys.contains($0.normalizedText) }
            try upsertPersistedEntry(updatedEntry)

            var updatedEntries = entries
            updatedEntries[existingIndex] = updatedEntry
            replaceEntries(updatedEntries)
            return
        }

        try createEntry(
            term: term,
            replacementTerms: replacementTerms,
            categoryID: categoryID,
            categoryNameSnapshot: categoryNameSnapshot,
            groupID: groupID,
            groupNameSnapshot: groupNameSnapshot,
            source: .manual
        )
    }

    func createOrReinforceManualEntry(
        term: String,
        replacementTerms: [String] = [],
        categoryID: UUID = DictionaryCategory.defaultID,
        categoryNameSnapshot: String? = DictionaryCategory.defaultName,
        groupID: UUID?,
        groupNameSnapshot: String?
    ) throws -> DictionaryEntryUpsertResult {
        try createOrReinforceEntry(
            term: term,
            replacementTerms: replacementTerms,
            categoryID: categoryID,
            categoryNameSnapshot: categoryNameSnapshot,
            groupID: groupID,
            groupNameSnapshot: groupNameSnapshot,
            source: .manual
        )
    }

    func createAutoEntry(
        term: String,
        replacementTerms: [String] = [],
        categoryID: UUID = DictionaryCategory.defaultID,
        categoryNameSnapshot: String? = DictionaryCategory.defaultName,
        groupID: UUID?,
        groupNameSnapshot: String?
    ) throws {
        try createEntry(
            term: term,
            replacementTerms: replacementTerms,
            categoryID: categoryID,
            categoryNameSnapshot: categoryNameSnapshot,
            groupID: groupID,
            groupNameSnapshot: groupNameSnapshot,
            source: .auto
        )
    }

    func createOrReinforceAutoEntry(
        term: String,
        replacementTerms: [String] = [],
        categoryID: UUID = DictionaryCategory.defaultID,
        categoryNameSnapshot: String? = DictionaryCategory.defaultName,
        groupID: UUID?,
        groupNameSnapshot: String?
    ) throws -> DictionaryEntryUpsertResult {
        try createOrReinforceEntry(
            term: term,
            replacementTerms: replacementTerms,
            categoryID: categoryID,
            categoryNameSnapshot: categoryNameSnapshot,
            groupID: groupID,
            groupNameSnapshot: groupNameSnapshot,
            source: .auto
        )
    }

    private func createEntry(
        term: String,
        replacementTerms: [String],
        categoryID: UUID,
        categoryNameSnapshot: String?,
        groupID: UUID?,
        groupNameSnapshot: String?,
        source: DictionaryEntrySource
    ) throws {
        guard completePendingReloadIfNeeded() else { throw DictionaryStoreError.dataUnavailable }
        let prepared = try prepareEntryInput(
            term: term,
            replacementTerms: replacementTerms,
            groupID: groupID
        )
        let now = Date()
        let entry = DictionaryEntry(
            term: prepared.display,
            normalizedTerm: prepared.normalized,
            categoryID: categoryID,
            categoryNameSnapshot: categoryNameSnapshot,
            groupID: groupID,
            groupNameSnapshot: groupNameSnapshot,
            source: source,
            createdAt: now,
            updatedAt: now,
            replacementTerms: prepared.replacementTerms
        )
        var updatedEntries = entries
        updatedEntries.insert(entry, at: 0)
        try upsertPersistedEntry(entry)
        replaceEntries(updatedEntries)
    }

    private func createOrReinforceEntry(
        term: String,
        replacementTerms: [String],
        categoryID: UUID,
        categoryNameSnapshot: String?,
        groupID: UUID?,
        groupNameSnapshot: String?,
        source: DictionaryEntrySource
    ) throws -> DictionaryEntryUpsertResult {
        guard completePendingReloadIfNeeded() else { throw DictionaryStoreError.dataUnavailable }
        let trimmedTerm = term.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = Self.normalizeTerm(trimmedTerm)
        guard !trimmedTerm.isEmpty, !normalized.isEmpty else {
            throw DictionaryStoreError.emptyTerm
        }

        if let existingIndex = existingTermIndex(normalizedTerm: normalized, groupID: groupID) {
            let reinforcedEntry = reinforceEntry(at: existingIndex, by: 1)
            return DictionaryEntryUpsertResult(
                term: reinforcedEntry.term,
                added: false,
                reinforcedCount: 1
            )
        }

        try createEntry(
            term: term,
            replacementTerms: replacementTerms,
            categoryID: categoryID,
            categoryNameSnapshot: categoryNameSnapshot,
            groupID: groupID,
            groupNameSnapshot: groupNameSnapshot,
            source: source
        )
        return DictionaryEntryUpsertResult(
            term: trimmedTerm,
            added: true,
            reinforcedCount: 0
        )
    }

    func updateEntry(
        id: UUID,
        term: String,
        replacementTerms: [String] = [],
        categoryID: UUID = DictionaryCategory.defaultID,
        categoryNameSnapshot: String? = DictionaryCategory.defaultName,
        groupID: UUID?,
        groupNameSnapshot: String?
    ) throws {
        guard completePendingReloadIfNeeded() else { throw DictionaryStoreError.dataUnavailable }
        let prepared = try prepareEntryInput(
            term: term,
            replacementTerms: replacementTerms,
            groupID: groupID,
            excluding: id
        )
        guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
        var updatedEntry = entries[index]
        updatedEntry.term = prepared.display
        updatedEntry.normalizedTerm = prepared.normalized
        updatedEntry.categoryID = categoryID
        updatedEntry.categoryNameSnapshot = categoryNameSnapshot
        updatedEntry.groupID = groupID
        updatedEntry.groupNameSnapshot = groupNameSnapshot
        updatedEntry.replacementTerms = prepared.replacementTerms
        updatedEntry.updatedAt = Date()

        let reservedKeys = Set([prepared.normalized] + prepared.replacementTerms.map(\.normalizedText))
        updatedEntry.observedVariants.removeAll { reservedKeys.contains($0.normalizedText) }
        try upsertPersistedEntry(updatedEntry)

        var updatedEntries = entries
        updatedEntries[index] = updatedEntry
        replaceEntries(updatedEntries)
    }

    @discardableResult
    func delete(id: UUID) -> Bool {
        guard completePendingReloadIfNeeded() else { return false }
        guard deletePersistedEntry(id: id) else { return false }
        replaceEntries(entries.filter { $0.id != id }, sort: false)
        return true
    }

    func clearAll() {
        guard completePendingReloadIfNeeded() else { return }
        guard clearPersistedEntries() else { return }
        replaceEntries([], sort: false)
    }

    func exportTransferJSONString() throws -> String {
        guard completePendingReloadIfNeeded() else { throw DictionaryStoreError.dataUnavailable }
        return try DictionaryTransferManager.exportJSONString(entries: entries, categories: resolvedCategories())
    }

    func importTransferJSONString(_ json: String) throws -> DictionaryImportResult {
        guard completePendingReloadIfNeeded() else { throw DictionaryStoreError.dataUnavailable }
        let payload = try DictionaryTransferManager.importPayload(from: json)
        return importTransferEntries(payload.entries, categories: payload.categories)
    }

    @discardableResult
    func incrementOccurrences(in text: String, activeGroupID: UUID?) -> [String] {
        guard completePendingReloadIfNeeded() else { return [] }
        let normalizedSource = Self.normalizeTerm(text)
        guard !normalizedSource.isEmpty else { return [] }

        let activeEntries = activeEntriesForRemoteRequest(activeGroupID: activeGroupID, limit: 5_000)
        var incrementsByID: [UUID: Int] = [:]
        for entry in activeEntries {
            let needles = [entry.normalizedTerm] + entry.replacementTerms.map(\.normalizedText)
            guard needles.contains(where: { sourceContainsNeedle($0, normalizedSource: normalizedSource) }) else {
                continue
            }
            incrementsByID[entry.id, default: 0] += 1
        }

        guard !incrementsByID.isEmpty else { return [] }
        let now = Date()
        var updatedEntries = entries
        var changedEntries: [DictionaryEntry] = []
        var reinforcedTerms: [String] = []
        for (entryID, count) in incrementsByID {
            guard let index = updatedEntries.firstIndex(where: { $0.id == entryID }) else { continue }
            updatedEntries[index].matchCount += count
            updatedEntries[index].lastMatchedAt = now
            updatedEntries[index].updatedAt = now
            changedEntries.append(updatedEntries[index])
            reinforcedTerms.append(updatedEntries[index].term)
        }

        guard !changedEntries.isEmpty else { return [] }
        replaceEntries(updatedEntries)
        persistEntries(changedEntries)
        return reinforcedTerms.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
    }

    func recordMatches(_ candidates: [DictionaryMatchCandidate]) {
        guard completePendingReloadIfNeeded() else { return }
        guard !candidates.isEmpty else { return }
        objectWillChange.send()
        let updatedEntries = recordCandidates(candidates)
        guard !updatedEntries.isEmpty else { return }
        filteredEntriesCache = DictionaryEntryCollection.filteredEntriesCache(for: entries)
        persistEntries(updatedEntries)
    }

    nonisolated static func normalizeTerm(_ input: String) -> String {
        DictionaryTermNormalizer.normalize(input)
    }

    private func prepareEntryInput(
        term: String,
        replacementTerms: [String],
        groupID: UUID?,
        excluding excludedID: UUID? = nil,
        existingEntries: [DictionaryEntry]? = nil,
        validationIndex providedValidationIndex: DictionaryValidationIndex? = nil
    ) throws -> DictionaryPreparedEntryInput {
        let resolvedEntries: [DictionaryEntry]?
        if providedValidationIndex == nil, existingEntries == nil, excludedID != nil {
            resolvedEntries = entries
        } else {
            resolvedEntries = existingEntries
        }

        let resolvedValidationIndex = providedValidationIndex
            ?? (resolvedEntries == nil ? validationIndex : nil)

        return try DictionaryEntryInputPreparer.prepare(
            term: term,
            replacementTerms: replacementTerms,
            groupID: groupID,
            excluding: excludedID,
            entries: resolvedEntries,
            validationIndex: resolvedValidationIndex
        )
    }

    private func importTransferEntries(
        _ transferEntries: [DictionaryTransferManager.Entry],
        categories transferCategories: [DictionaryCategory]
    ) -> DictionaryImportResult {
        importTransferCategories(transferCategories, entries: transferEntries)
        var mergedEntries = entries
        var importValidationIndex = validationIndex
        var addedCount = 0
        var skippedCount = 0

        for transferEntry in transferEntries {
            do {
                let prepared = try prepareEntryInput(
                    term: transferEntry.term,
                    replacementTerms: transferEntry.replacementTerms,
                    groupID: transferEntry.groupID,
                    validationIndex: importValidationIndex
                )
                let now = Date()
                let entry = DictionaryEntry(
                    term: prepared.display,
                    normalizedTerm: prepared.normalized,
                    categoryID: transferEntry.categoryID ?? categoryIDForImportedEntry(transferEntry),
                    categoryNameSnapshot: transferEntry.categoryNameSnapshot ?? categoryNameForImportedEntry(transferEntry),
                    groupID: transferEntry.groupID,
                    groupNameSnapshot: transferEntry.groupNameSnapshot,
                    source: .manual,
                    createdAt: now,
                    updatedAt: now,
                    replacementTerms: prepared.replacementTerms
                )
                mergedEntries.append(entry)
                importValidationIndex.insert(entry)
                addedCount += 1
            } catch {
                skippedCount += 1
            }
        }

        replaceEntries(mergedEntries)
        persist()
        return DictionaryImportResult(addedCount: addedCount, skippedCount: skippedCount)
    }

    private func importTransferCategories(
        _ transferCategories: [DictionaryCategory],
        entries transferEntries: [DictionaryTransferManager.Entry]
    ) {
        var mergedCategoriesByID = Dictionary(uniqueKeysWithValues: categories.map { ($0.id, $0) })
        for category in transferCategories {
            mergedCategoriesByID[category.id] = category
        }

        for entry in transferEntries {
            let categoryID = entry.categoryID ?? categoryIDForImportedEntry(entry)
            guard mergedCategoriesByID[categoryID] == nil else { continue }
            mergedCategoriesByID[categoryID] = DictionaryCategory(
                id: categoryID,
                name: entry.categoryNameSnapshot ?? categoryNameForImportedEntry(entry),
                isDefault: categoryID == DictionaryCategory.defaultID,
                isExpanded: true,
                sortOrder: mergedCategoriesByID.count
            )
        }

        let mergedCategories = Array(mergedCategoriesByID.values)
        replaceCategories(mergedCategories)
        guard persistenceEnabled, let repository else { return }
        invalidatePendingReload()
        for category in resolvedCategories() {
            try? repository.upsertCategory(category)
        }
    }

    private func existingTermIndex(normalizedTerm: String, groupID: UUID?) -> Int? {
        entries.firstIndex { entry in
            entry.groupID == groupID && entry.normalizedTerm == normalizedTerm
        }
    }

    @discardableResult
    private func reinforceEntry(at index: Int, by count: Int) -> DictionaryEntry {
        let now = Date()
        var updatedEntries = entries
        updatedEntries[index].matchCount += count
        updatedEntries[index].lastMatchedAt = now
        updatedEntries[index].updatedAt = now
        let updatedEntry = updatedEntries[index]
        replaceEntries(updatedEntries)
        persistEntry(updatedEntry)
        return updatedEntry
    }

    private func recordCandidates(_ candidates: [DictionaryMatchCandidate]) -> [DictionaryEntry] {
        guard !candidates.isEmpty else { return [] }
        let now = Date()
        let grouped = Dictionary(grouping: candidates, by: \.entryID)
        var updatedEntries: [DictionaryEntry] = []

        for (entryID, matches) in grouped {
            guard let index = entries.firstIndex(where: { $0.id == entryID }) else { continue }
            entries[index].lastMatchedAt = now
            entries[index].matchCount += matches.count
            entries[index].updatedAt = now

            for candidate in matches where candidate.shouldPersistObservedVariant {
                let normalizedReservedKeys = Set(
                    [entries[index].normalizedTerm] + entries[index].replacementTerms.map(\.normalizedText)
                )
                guard !normalizedReservedKeys.contains(candidate.normalizedMatchedText) else { continue }
                upsertVariant(
                    into: &entries[index],
                    text: candidate.matchedText,
                    normalizedText: candidate.normalizedMatchedText,
                    confidence: confidence(for: candidate)
                )
            }
            updatedEntries.append(entries[index])
        }

        return updatedEntries
    }

    private func upsertVariant(
        into entry: inout DictionaryEntry,
        text: String,
        normalizedText: String,
        confidence: DictionaryVariantConfidence
    ) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        if let variantIndex = entry.observedVariants.firstIndex(where: { $0.normalizedText == normalizedText }) {
            entry.observedVariants[variantIndex].count += 1
            entry.observedVariants[variantIndex].lastSeenAt = Date()
            entry.observedVariants[variantIndex].confidence = higherConfidence(
                lhs: entry.observedVariants[variantIndex].confidence,
                rhs: confidence
            )
        } else {
            entry.observedVariants.append(
                ObservedVariant(
                    text: text,
                    normalizedText: normalizedText,
                    confidence: confidence
                )
            )
            entry.observedVariants.sort { $0.count > $1.count }
        }
    }

    private func confidence(for candidate: DictionaryMatchCandidate) -> DictionaryVariantConfidence {
        if candidate.score >= 0.985 {
            return .high
        }
        if candidate.score >= 0.92 {
            return .medium
        }
        return .low
    }

    private func higherConfidence(lhs: DictionaryVariantConfidence, rhs: DictionaryVariantConfidence) -> DictionaryVariantConfidence {
        let rank: [DictionaryVariantConfidence: Int] = [
            .low: 0,
            .medium: 1,
            .high: 2
        ]
        return (rank[lhs] ?? 0) >= (rank[rhs] ?? 0) ? lhs : rhs
    }

    private func sourceContainsNeedle(_ needle: String, normalizedSource: String) -> Bool {
        let normalizedNeedle = Self.normalizeTerm(needle)
        guard !normalizedNeedle.isEmpty else { return false }

        var searchRange: Range<String.Index>? = normalizedSource.startIndex..<normalizedSource.endIndex
        while let range = normalizedSource.range(of: normalizedNeedle, options: [], range: searchRange) {
            if hasValidBoundary(
                before: range.lowerBound,
                after: range.upperBound,
                needle: normalizedNeedle,
                source: normalizedSource
            ) {
                return true
            }
            searchRange = range.upperBound..<normalizedSource.endIndex
        }
        return false
    }

    private func hasValidBoundary(
        before lowerBound: String.Index,
        after upperBound: String.Index,
        needle: String,
        source: String
    ) -> Bool {
        let needsLeadingBoundary = needle.unicodeScalars.first.map(Self.isASCIIAlphaNumeric) ?? false
        let needsTrailingBoundary = needle.unicodeScalars.last.map(Self.isASCIIAlphaNumeric) ?? false

        if needsLeadingBoundary,
           lowerBound > source.startIndex,
           let previous = source[..<lowerBound].unicodeScalars.last,
           Self.isASCIIAlphaNumeric(previous) {
            return false
        }

        if needsTrailingBoundary,
           upperBound < source.endIndex,
           let next = source[upperBound...].unicodeScalars.first,
           Self.isASCIIAlphaNumeric(next) {
            return false
        }

        return true
    }

    private nonisolated static func isASCIIAlphaNumeric(_ scalar: UnicodeScalar) -> Bool {
        (65...90).contains(Int(scalar.value))
            || (97...122).contains(Int(scalar.value))
            || (48...57).contains(Int(scalar.value))
    }

    private func persist() {
        guard persistenceEnabled, let repository else { return }
        invalidatePendingReload()
        do {
            try repository.replaceAll(entries)
        } catch {
            // Keep UI responsive even if persistence fails.
        }
    }

    private func persistEntry(_ entry: DictionaryEntry) {
        do {
            try upsertPersistedEntry(entry)
        } catch {
            persist()
        }
    }

    private func upsertPersistedEntry(_ entry: DictionaryEntry) throws {
        guard persistenceEnabled, let repository else { return }
        invalidatePendingReload()
        try repository.upsert(entry)
    }

    private func upsertPersistedCategory(_ category: DictionaryCategory) throws {
        guard persistenceEnabled, let repository else { return }
        invalidatePendingReload()
        try repository.upsertCategory(category)
    }

    private func persistEntries(_ updatedEntries: [DictionaryEntry]) {
        guard persistenceEnabled, let repository else { return }
        invalidatePendingReload()
        do {
            for entry in updatedEntries {
                try repository.upsert(entry)
            }
        } catch {
            persist()
        }
    }

    private func deletePersistedEntry(id: UUID) -> Bool {
        guard persistenceEnabled, let repository else { return true }
        invalidatePendingReload()
        do {
            try repository.delete(id: id)
            return true
        } catch {
            return false
        }
    }

    private func clearPersistedEntries() -> Bool {
        guard persistenceEnabled, let repository else { return true }
        invalidatePendingReload()
        do {
            try repository.clearAll()
            return true
        } catch {
            return false
        }
    }

    private func invalidatePendingReload() {
        reloadGeneration += 1
        isLoading = false
    }

    func completePendingReloadIfNeeded() -> Bool {
        guard isLoading else { return true }
        // Normal startup remains asynchronous. Only a data-dependent action that
        // races that first load pays the synchronous read, so validation and
        // replace-all operations never run against a partial in-memory snapshot.
        return reload()
    }

    private func legacyDictionaryFileExists() -> Bool {
        (try? dictionaryFileURL()).map { fileManager.fileExists(atPath: $0.path) } ?? false
    }

    private func dictionaryFileURL() throws -> URL {
        let appSupport = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return appSupport
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("dictionary.json")
    }

    private func sortEntries(_ values: [DictionaryEntry]) -> [DictionaryEntry] {
        DictionaryEntryCollection.sortedEntries(values)
    }

    private func sortCategories(_ values: [DictionaryCategory]) -> [DictionaryCategory] {
        values.sorted {
            if $0.isDefault != $1.isDefault {
                return $0.isDefault
            }
            if $0.sortOrder != $1.sortOrder {
                return $0.sortOrder < $1.sortOrder
            }
            return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
        }
    }

    private func applyReloadedEntries(_ decodedEntries: [DictionaryEntry]) {
        replaceEntries(decodedEntries)
    }

    private func applyReloadedCategories(_ decodedCategories: [DictionaryCategory]) {
        replaceCategories(decodedCategories)
    }

    private func replaceEntries(_ values: [DictionaryEntry], sort: Bool = true) {
        let resolvedEntries = sort ? sortEntries(values) : values
        entries = resolvedEntries
        filteredEntriesCache = DictionaryEntryCollection.filteredEntriesCache(for: resolvedEntries)
        validationIndex = DictionaryValidationIndex(entries: resolvedEntries)
    }

    private func replaceCategories(_ values: [DictionaryCategory]) {
        var resolved = values
        if !resolved.contains(where: \.isDefault) {
            resolved.append(DictionaryCategory.defaultCategory)
        }
        categories = sortCategories(resolved)
    }

    private func resolvedCategories() -> [DictionaryCategory] {
        let knownIDs = Set(categories.map(\.id))
        let missingCategories = entries
            .filter { !knownIDs.contains($0.categoryID) }
            .reduce(into: [UUID: DictionaryCategory]()) { partialResult, entry in
                partialResult[entry.categoryID] = DictionaryCategory(
                    id: entry.categoryID,
                    name: entry.categoryNameSnapshot ?? DictionaryCategory.defaultName,
                    isDefault: entry.categoryID == DictionaryCategory.defaultID,
                    isExpanded: true,
                    sortOrder: categories.count + partialResult.count + 1,
                    createdAt: entry.createdAt,
                    updatedAt: entry.updatedAt
                )
            }
            .values
        return sortCategories(categories + missingCategories)
    }

    private func resolvedDefaultCategory() -> DictionaryCategory {
        categories.first(where: \.isDefault) ?? DictionaryCategory.defaultCategory
    }

    private func refreshCategoryNameSnapshot(_ category: DictionaryCategory) {
        let updatedEntries = entries.map { entry -> DictionaryEntry in
            guard entry.categoryID == category.id else { return entry }
            var updated = entry
            updated.categoryNameSnapshot = category.name
            return updated
        }
        replaceEntries(updatedEntries)
        persistEntries(updatedEntries.filter { $0.categoryID == category.id })
    }

    private func prepareCategoryName(
        _ name: String,
        excluding excludedID: UUID? = nil
    ) throws -> (display: String, normalized: String) {
        let display = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = Self.normalizeTerm(display)
        guard !display.isEmpty, !normalized.isEmpty else {
            throw DictionaryStoreError.emptyCategoryName
        }
        let duplicate = categories.contains {
            $0.id != excludedID && $0.normalizedName == normalized
        }
        if duplicate {
            throw DictionaryStoreError.duplicateCategory
        }
        return (display, normalized)
    }

    private func categoryIDForImportedEntry(_ entry: DictionaryTransferManager.Entry) -> UUID {
        if let groupID = entry.groupID {
            return groupID
        }
        return DictionaryCategory.defaultID
    }

    private func categoryNameForImportedEntry(_ entry: DictionaryTransferManager.Entry) -> String {
        entry.groupNameSnapshot ?? DictionaryCategory.defaultName
    }
}
