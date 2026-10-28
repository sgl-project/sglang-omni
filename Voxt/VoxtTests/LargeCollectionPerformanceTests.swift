// LargeCollectionPerformanceTests.swift
// Provides Large Collection Performance Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class LargeCollectionPerformanceTests: XCTestCase {
    func testDictionarySearchFiltersTwentyThousandEntries() {
        let entries = (0..<20_000).map {
            TestFactories.makeEntry(term: "Term\($0)", replacementTerms: ["Alias\($0)"])
        }

        let matches = DictionaryEntryCollection.searchEntries(entries, query: "Alias19999")

        XCTAssertEqual(matches.map(\.term), ["Term19999"])
    }

    func testHistorySearchFiltersTwentyThousandEntries() {
        let entries = (0..<20_000).map {
            makeHistoryEntry(
                kind: .normal,
                text: $0 == 19_999 ? "needle history entry" : "history entry \($0)"
            )
        }

        let matches = HistorySettingsData.searchEntries(entries, query: "needle")

        XCTAssertEqual(matches.map(\.text), ["needle history entry"])
    }

    func testDictionaryMatcherFindsExactTermInTwentyThousandEntries() {
        let entries = (0..<20_000).map {
            TestFactories.makeEntry(term: "Term\($0)")
        }
        let matcher = DictionaryMatcher(entries: entries, blockedGlobalMatchKeys: [])

        let candidates = matcher.recallCandidates(in: "Please keep Term19999 intact.")

        XCTAssertTrue(candidates.contains { $0.term == "Term19999" && $0.reason == .exactTerm })
    }
}

private extension LargeCollectionPerformanceTests {
    func makeHistoryEntry(kind: TranscriptionHistoryKind, text: String) -> TranscriptionHistoryEntry {
        TranscriptionHistoryEntry(
            id: UUID(),
            text: text,
            createdAt: Date(timeIntervalSince1970: 1),
            transcriptionEngine: "engine",
            transcriptionModel: "model",
            enhancementMode: "mode",
            enhancementModel: "enhanced",
            kind: kind,
            isTranslation: kind == .translation,
            audioDurationSeconds: nil,
            transcriptionProcessingDurationSeconds: nil,
            llmDurationSeconds: nil,
            focusedAppName: nil,
            focusedAppBundleID: nil,
            matchedGroupID: nil,
            matchedGroupName: nil,
            matchedAppGroupName: nil,
            matchedURLGroupName: nil,
            remoteASRProvider: nil,
            remoteASRModel: nil,
            remoteASREndpoint: nil,
            remoteLLMProvider: nil,
            remoteLLMModel: nil,
            remoteLLMEndpoint: nil,
            audioRelativePath: nil,
            whisperWordTimings: nil,
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        )
    }
}
