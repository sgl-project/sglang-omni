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

    func testProjectDictionaryScannerExtractsDecoratedTerms() {
        let terms = ProjectDictionaryScanner.candidateTerms(
            in: "Use MLXWhisper, MLXAudio and tmp/typeflux for VoxtProject.",
            allowPlainLowercase: false
        )

        XCTAssertTrue(terms.contains("MLXWhisper"))
        XCTAssertTrue(terms.contains("MLXAudio"))
        XCTAssertTrue(terms.contains("tmp/typeflux"))
        XCTAssertTrue(terms.contains("VoxtProject"))
    }
}

private extension DictionaryEntryCollectionTests {
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
