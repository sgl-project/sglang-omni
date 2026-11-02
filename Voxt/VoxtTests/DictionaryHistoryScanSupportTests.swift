// DictionaryHistoryScanSupportTests.swift
// Provides Dictionary History Scan Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class DictionaryHistoryScanSupportTests: XCTestCase {
    func testResponseParserExtractsTermsFromJSONStringWrappedContent() throws {
        let response = """
        Here is the result:
        [
          { "term": "OpenAI" },
          { "term": "MCP" }
        ]
        """

        let terms = try DictionaryHistoryScanResponseParser.parseTerms(from: response)
        XCTAssertEqual(terms, ["OpenAI", "MCP"])
    }

    func testBuildPromptInjectsLanguageAndHistoryXML() throws {
        let entry = TranscriptionHistoryEntry(
            id: UUID(),
            text: "Discuss OpenAI roadmap",
            createdAt: Date(timeIntervalSince1970: 0),
            transcriptionEngine: "test-engine",
            transcriptionModel: "test-model",
            enhancementMode: "none",
            enhancementModel: "none",
            kind: .normal,
            isTranslation: false,
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
            whisperWordTimings: nil,
            dictionaryHitTerms: ["OpenAI"],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        )

        let prompt = try DictionaryHistoryScanSupport.buildPrompt(
            for: [entry],
            filterSettings: .defaultValue,
            groupsByID: [:],
            groupsByLowercasedName: [:],
            userMainLanguage: "English",
            userOtherLanguages: "Chinese"
        )

        XCTAssertTrue(prompt.contains("English"))
        XCTAssertTrue(prompt.contains("Chinese"))
        XCTAssertTrue(prompt.contains("<historyRecords>"))
        XCTAssertTrue(prompt.contains("Discuss OpenAI roadmap"))
        XCTAssertTrue(prompt.contains("<maxCandidateCount>12</maxCandidateCount>"))
        XCTAssertTrue(prompt.contains("Return at most 12 accepted terms."))
    }

    func testResponseParserRecoversTermsFromTruncatedJSONArray() throws {
        let response = """
        [{"term":"OpenAI"},{"term":"button"},{"term":"MCP"},{"term":"station"
        """

        let terms = try DictionaryHistoryScanResponseParser.parseTerms(from: response)
        XCTAssertEqual(terms, ["OpenAI", "MCP"])
    }
}
