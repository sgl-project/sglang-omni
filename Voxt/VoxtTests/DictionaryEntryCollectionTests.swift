// DictionaryEntryCollectionTests.swift
// Provides Dictionary Entry Collection Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class DictionaryEntryCollectionTests: XCTestCase {
    func testFilteredEntriesSeparatesManualAndAutoSources() {
        let entries = DictionaryEntryCollection.sortedEntries([
            makeEntry(term: "Voxt", source: .manual, updatedAt: Date(timeIntervalSince1970: 100)),
            makeEntry(term: "Waxed", source: .auto, updatedAt: Date(timeIntervalSince1970: 200)),
            makeEntry(term: "Ghostty", source: .manual, updatedAt: Date(timeIntervalSince1970: 300))
        ])

        let cache = DictionaryEntryCollection.filteredEntriesCache(for: entries)

        XCTAssertEqual(cache[.all]?.map(\.term), ["Ghostty", "Waxed", "Voxt"])
        XCTAssertEqual(cache[.manualAdded]?.map(\.term), ["Ghostty", "Voxt"])
        XCTAssertEqual(cache[.autoAdded]?.map(\.term), ["Waxed"])
    }

    func testPromptBiasTermsTextPrioritizesHigherMatchCountAndRespectsLimits() {
        let highPriority = makeEntry(
            term: "Anthropic",
            source: .manual,
            matchCount: 9,
            updatedAt: Date(timeIntervalSince1970: 200)
        )
        let secondary = makeEntry(
            term: "OpenAI",
            source: .manual,
            matchCount: 4,
            updatedAt: Date(timeIntervalSince1970: 150)
        )
        let duplicateNormalized = makeEntry(
            term: "openai",
            source: .manual,
            matchCount: 3,
            updatedAt: Date(timeIntervalSince1970: 100)
        )

        let bias = DictionaryEntryCollection.promptBiasTermsText(
            from: [secondary, duplicateNormalized, highPriority],
            activeGroupID: nil,
            maxCount: 2,
            maxCharacters: 32
        )

        XCTAssertEqual(bias, "Anthropic\nOpenAI")
    }

    func testPromptBiasTermsTextPrefersScopedEntriesOverBlockedGlobals() {
        let sharedKey = "Voxt"
        let groupID = UUID()
        let global = makeEntry(
            term: sharedKey,
            source: .manual,
            groupID: nil,
            updatedAt: Date(timeIntervalSince1970: 100)
        )
        let scoped = makeEntry(
            term: sharedKey,
            source: .manual,
            groupID: groupID,
            matchCount: 5,
            updatedAt: Date(timeIntervalSince1970: 300)
        )
        let extraGlobal = makeEntry(
            term: "Ghostty",
            source: .manual,
            groupID: nil,
            updatedAt: Date(timeIntervalSince1970: 200)
        )

        let bias = DictionaryEntryCollection.promptBiasTermsText(
            from: [global, scoped, extraGlobal],
            activeGroupID: groupID,
            maxCount: 5,
            maxCharacters: 80
        )

        XCTAssertEqual(bias, "Voxt\nGhostty")
    }

    func testASRPromptTermsTextUsesTypefluxSizedLimitAndFrequencyRanking() {
        let entries = (0..<40).map { index in
            makeEntry(
                term: "Term\(index)",
                source: .manual,
                matchCount: index,
                updatedAt: Date(timeIntervalSince1970: TimeInterval(index))
            )
        }

        let bias = DictionaryEntryCollection.asrPromptTermsText(
            from: entries,
            maxCount: 32,
            maxCharacters: 10_000
        )

        let terms = bias.components(separatedBy: "\n")
        XCTAssertEqual(terms.count, 32)
        XCTAssertEqual(terms.first, "Term39")
        XCTAssertEqual(terms.last, "Term8")
    }

    @MainActor
    func testManualReinforcementPreservesDisplayDefaultsAndScope() async throws {
        let store = makeStore()
        try store.createManualEntry(term: "  Café  ", groupID: nil, groupNameSnapshot: nil)
        let entry = try XCTUnwrap(store.entries.first)
        let result = try store.createOrReinforceAutoEntry(
            term: "ＣＡＦＥ", groupID: nil, groupNameSnapshot: nil
        )
        XCTAssertEqual(result, DictionaryEntryUpsertResult(term: "Café", added: false, reinforcedCount: 1))
        XCTAssertEqual(store.entries.first?.id, entry.id)
        XCTAssertEqual(store.entries.first?.source, .manual)
        XCTAssertEqual(store.entries.first?.categoryID, DictionaryCategory.defaultID)
        XCTAssertEqual(store.entries.first?.categoryNameSnapshot, DictionaryCategory.defaultName)
        XCTAssertEqual(store.entries.first?.matchCount, 1)

        let groupID = UUID()
        try store.createManualEntry(term: "cafe", groupID: groupID, groupNameSnapshot: "Project")
        XCTAssertEqual(store.entries.count, 2)
        XCTAssertEqual(store.entries.first(where: { $0.groupID == groupID })?.groupNameSnapshot, "Project")
        XCTAssertThrowsError(try store.createManualEntry(term: "?!", groupID: nil, groupNameSnapshot: nil))
    }

    @MainActor
    func testReplacementValidationExcludesEditedEntryAndKeepsScopeIsolation() async throws {
        let store = makeStore()
        try store.createManualEntry(
            term: "Voxt", replacementTerms: [" vox ", "ＶＯＸ", "?!"],
            groupID: nil, groupNameSnapshot: nil
        )
        let entry = try XCTUnwrap(store.entries.first)
        XCTAssertEqual(entry.replacementTerms.map(\.text), ["vox"])
        try store.updateEntry(
            id: entry.id, term: "Voxt", replacementTerms: ["vox"],
            groupID: nil, groupNameSnapshot: nil
        )
        XCTAssertThrowsError(try store.createManualEntry(
            term: "Other", replacementTerms: ["VOX"], groupID: nil, groupNameSnapshot: nil
        )) { error in
            guard case DictionaryStoreError.duplicateReplacementTerm("VOX") = error else {
                return XCTFail("Unexpected validation error: \(error)")
            }
        }
        try store.createManualEntry(
            term: "Other", replacementTerms: ["VOX"], groupID: UUID(), groupNameSnapshot: "Project"
        )
        XCTAssertThrowsError(try store.updateEntry(
            id: entry.id, term: "Voxt", replacementTerms: ["ＶＯＸＴ"],
            groupID: nil, groupNameSnapshot: nil
        ))
        XCTAssertEqual(store.entries.first(where: { $0.id == entry.id })?.replacementTerms.map(\.text), ["vox"])
    }

    @MainActor
    func testCategoryNamesNormalizeDuplicatesAndAllowExcludedRename() async throws {
        let store = makeStore()
        let category = try store.createCategory(name: "  Café  ")
        XCTAssertEqual(category.name, "Café")
        XCTAssertEqual(category.normalizedName, "cafe")
        XCTAssertThrowsError(try store.createCategory(name: "ＣＡＦＥ")) { error in
            guard case DictionaryStoreError.duplicateCategory = error else {
                return XCTFail("Unexpected category error: \(error)")
            }
        }
        try store.updateCategory(id: category.id, name: "CAFE")
        XCTAssertEqual(store.categoryName(for: category.id), "CAFE")
        XCTAssertThrowsError(try store.createCategory(name: "?!"))
        XCTAssertEqual(store.categories.count, 2)
    }

    @MainActor
    func testRecordedVariantsPreserveIdentityConfidenceAndExistingOrder() async throws {
        let existingVariant = ObservedVariant(text: "vox", normalizedText: "vox", count: 2, confidence: .high)
        let otherVariant = ObservedVariant(text: "other", normalizedText: "other", count: 6, confidence: .low)
        let entry = DictionaryEntry(
            term: "Voxt", normalizedTerm: "voxt", source: .manual,
            observedVariants: [existingVariant, otherVariant],
            replacementTerms: [DictionaryReplacementTerm(text: "alias", normalizedText: "alias")]
        )
        let store = makeStore(entries: [entry])
        func candidate(_ text: String, normalized: String, score: Double) -> DictionaryMatchCandidate {
            DictionaryMatchCandidate(
                entryID: entry.id, term: entry.term, matchedText: text, normalizedMatchedText: normalized,
                score: score, reason: .fuzzyWindow, source: .term, matchRange: nil
            )
        }
        store.recordMatches([candidate("VOX", normalized: "vox", score: 0.1)])
        XCTAssertEqual(store.entries.first?.observedVariants.map(\.id), [existingVariant.id, otherVariant.id])
        XCTAssertEqual(store.entries.first?.observedVariants.first?.count, 3)
        XCTAssertEqual(store.entries.first?.observedVariants.first?.text, "vox")
        XCTAssertEqual(store.entries.first?.observedVariants.first?.confidence, .high)

        store.recordMatches([
            candidate("low", normalized: "low", score: 0.919999),
            candidate("medium", normalized: "medium", score: 0.92),
            candidate("high", normalized: "high", score: 0.985),
            candidate("Voxt", normalized: "voxt", score: 1),
            candidate("alias", normalized: "alias", score: 1),
            candidate("  ", normalized: "blank", score: 1)
        ])
        let updated = try XCTUnwrap(store.entries.first)
        XCTAssertEqual(updated.matchCount, 7)
        XCTAssertNotNil(updated.lastMatchedAt)
        XCTAssertEqual(updated.observedVariants.first?.id, otherVariant.id)
        XCTAssertEqual(updated.observedVariants.first(where: { $0.normalizedText == "low" })?.confidence, .low)
        XCTAssertEqual(updated.observedVariants.first(where: { $0.normalizedText == "medium" })?.confidence, .medium)
        XCTAssertEqual(updated.observedVariants.first(where: { $0.normalizedText == "high" })?.confidence, .high)
        XCTAssertEqual(updated.observedVariants.count, 5)

        try store.updateEntry(
            id: entry.id, term: "vox", replacementTerms: ["other"], groupID: nil, groupNameSnapshot: nil
        )
        XCTAssertEqual(Set(store.entries.first?.observedVariants.map(\.normalizedText) ?? []), ["low", "medium", "high"])
        try store.createManualEntry(
            term: "VOX", replacementTerms: ["low"], groupID: nil, groupNameSnapshot: nil
        )
        XCTAssertEqual(Set(store.entries.first?.observedVariants.map(\.normalizedText) ?? []), ["medium", "high"])
        XCTAssertEqual(store.entries.first?.replacementTerms.map(\.text), ["other", "low"])
        XCTAssertEqual(store.entries.first?.matchCount, 7)
    }

    @MainActor
    func testOccurrenceMatchingUsesASCIIBoundariesAndContinuesPastEmbeddedMatches() async throws {
        let store = makeStore(entries: [
            makeEntry(term: "cat", source: .manual),
            makeEntry(term: "猫", source: .manual),
            makeEntry(term: "café", source: .manual)
        ])
        XCTAssertEqual(store.incrementOccurrences(in: "bobcat cat2", activeGroupID: nil), [])
        let expectedTerms = ["café", "cat", "猫"].sorted {
            $0.localizedCaseInsensitiveCompare($1) == .orderedAscending
        }
        XCTAssertEqual(store.incrementOccurrences(in: "bobcat cat 猫咪 café", activeGroupID: nil), expectedTerms)
        XCTAssertEqual(store.incrementOccurrences(in: "cat cat", activeGroupID: nil), ["cat"])
        XCTAssertEqual(store.entries.first(where: { $0.term == "cat" })?.matchCount, 2)
        XCTAssertEqual(store.incrementOccurrences(in: "?!", activeGroupID: nil), [])
    }

    @MainActor
    func testImportValidationUsesGrowingIndexAndPreservesLegacyCategoryFallbacks() async throws {
        let store = makeStore()
        let groupID = UUID()
        let json = """
        {"entries":[
          {"term":"First","replacementTerms":["alias"],"groupID":"\(groupID)","groupNameSnapshot":"Project"},
          {"term":"Second","replacementTerms":["ALIAS"],"groupID":"\(groupID)"},
          {"term":"Third","replacementTerms":["ALIAS"]},
          {"term":"?!"}
        ]}
        """
        let result = try store.importTransferJSONString(json)
        XCTAssertEqual(result, DictionaryImportResult(addedCount: 2, skippedCount: 2))
        let first = try XCTUnwrap(store.entries.first(where: { $0.term == "First" }))
        XCTAssertEqual(first.categoryID, groupID)
        XCTAssertEqual(first.categoryNameSnapshot, "Project")
        XCTAssertEqual(first.groupID, groupID)
        XCTAssertEqual(first.source, .manual)
        XCTAssertEqual(store.categoryName(for: groupID), "Project")
        XCTAssertEqual(store.entries.first(where: { $0.term == "Third" })?.categoryID, DictionaryCategory.defaultID)
    }
}

private extension DictionaryEntryCollectionTests {
    @MainActor
    func makeStore(entries: [DictionaryEntry] = []) -> DictionaryStore {
        let suiteName = "DictionaryEntryCollectionTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        addTeardownBlock { defaults.removePersistentDomain(forName: suiteName) }
        return DictionaryStore(defaults: defaults, fileManager: .default, initialEntries: entries, persistenceEnabled: false)
    }

    func makeEntry(
        term: String,
        source: DictionaryEntrySource,
        groupID: UUID? = nil,
        matchCount: Int = 0,
        updatedAt: Date = Date()
    ) -> DictionaryEntry {
        DictionaryEntry(
            term: term,
            normalizedTerm: DictionaryStore.normalizeTerm(term),
            groupID: groupID,
            source: source,
            updatedAt: updatedAt,
            matchCount: matchCount
        )
    }
}
