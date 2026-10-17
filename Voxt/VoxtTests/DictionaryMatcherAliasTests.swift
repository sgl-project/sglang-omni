// DictionaryMatcherAliasTests.swift
// Provides Dictionary Matcher Alias Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class DictionaryMatcherAliasTests: XCTestCase {
    func testReplacementTermsReplaceAllOccurrencesWithoutAutomaticCorrection() {
        let matcher = DictionaryMatcher(
            entries: [
                makeEntry(term: "OpenAI", replacementTerms: ["open ai"])
            ],
            blockedGlobalMatchKeys: []
        )

        let result = matcher.applyCorrections(
            to: "open ai and Open AI",
            automaticReplacementEnabled: false
        )

        XCTAssertEqual(result.text, "OpenAI and OpenAI")
        XCTAssertEqual(result.correctedTerms, ["OpenAI", "OpenAI"])
        XCTAssertEqual(
            Set(result.candidates.map(\.term)),
            ["OpenAI"]
        )
    }

    func testScopedReplacementTermBlocksOnlyConflictingGlobalAlias() {
        let scopedGroupID = UUID()
        let matcher = DictionaryMatcher(
            entries: [
                makeEntry(
                    term: "InternalGPT",
                    replacementTerms: ["chatgpt"],
                    groupID: scopedGroupID
                ),
                makeEntry(
                    term: "OpenAI",
                    replacementTerms: ["chatgpt", "ai assistant"]
                )
            ],
            blockedGlobalMatchKeys: ["chatgpt"]
        )

        let result = matcher.applyCorrections(
            to: "chatgpt and ai assistant",
            automaticReplacementEnabled: false
        )

        XCTAssertEqual(result.text, "InternalGPT and OpenAI")
        XCTAssertEqual(result.correctedTerms, ["OpenAI", "InternalGPT"])
    }

    func testExactWindowCorrectionStillRequiresAutomaticReplacement() {
        let matcher = DictionaryMatcher(
            entries: [
                makeEntry(term: "Open AI")
            ],
            blockedGlobalMatchKeys: []
        )

        let withoutAutomatic = matcher.applyCorrections(
            to: "Open-AI",
            automaticReplacementEnabled: false
        )
        XCTAssertEqual(withoutAutomatic.text, "Open-AI")

        let withAutomatic = matcher.applyCorrections(
            to: "Open-AI",
            automaticReplacementEnabled: true
        )
        XCTAssertEqual(withAutomatic.text, "Open AI")
        XCTAssertEqual(withAutomatic.correctedTerms, ["Open AI"])
    }

    func testGlossaryUsesCanonicalTermForReplacementMatch() {
        let entry = makeEntry(term: "Anthropic", replacementTerms: ["anthropic ai"])
        let matcher = DictionaryMatcher(entries: [entry], blockedGlobalMatchKeys: [])

        let context = matcher.promptContext(for: "anthropic ai")

        XCTAssertEqual(context.glossaryText(), "- Anthropic")
    }

    private func makeEntry(
        term: String,
        replacementTerms: [String] = [],
        groupID: UUID? = nil
    ) -> DictionaryEntry {
        DictionaryEntry(
            term: term,
            normalizedTerm: DictionaryStore.normalizeTerm(term),
            groupID: groupID,
            source: .manual,
            replacementTerms: replacementTerms.map {
                DictionaryReplacementTerm(
                    text: $0,
                    normalizedText: DictionaryStore.normalizeTerm($0)
                )
            }
        )
    }
}
