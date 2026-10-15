// MeetingTranslationSupportTests.swift
// Provides Meeting Translation Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class MeetingTranslationSupportTests: XCTestCase {
    func testCustomLLMProviderPassesThrough() {
        let resolution = MeetingTranslationSupport.resolvedProvider(
            selectedProvider: .customLLM,
            fallbackProvider: .remoteLLM,
            transcriptionEngine: .mlxAudio,
            targetLanguage: .japanese
        )

        XCTAssertEqual(resolution.provider, .customLLM)
        XCTAssertEqual(resolution.fallbackProvider, .remoteLLM)
    }

    func testRemoteLLMProviderPassesThrough() {
        let resolution = MeetingTranslationSupport.resolvedProvider(
            selectedProvider: .remoteLLM,
            fallbackProvider: .customLLM,
            transcriptionEngine: .mlxAudio,
            targetLanguage: .english
        )

        XCTAssertEqual(resolution.provider, .remoteLLM)
        XCTAssertEqual(resolution.fallbackProvider, .customLLM)
    }
}
