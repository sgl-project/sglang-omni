// TranslationProviderResolverTests.swift
// Provides Translation Provider Resolver Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class TranslationProviderResolverTests: XCTestCase {
    func testResolvePassesThroughConfiguredProvider() {
        let resolution = TranslationProviderResolver.resolve(
            selectedProvider: .remoteLLM,
            fallbackProvider: .customLLM,
            transcriptionEngine: .mlxAudio,
            targetLanguage: .english,
            isSelectedTextTranslation: false
        )

        XCTAssertEqual(resolution.provider, .remoteLLM)
        XCTAssertEqual(resolution.fallbackProvider, .customLLM)
    }

    func testLegacyWhisperProviderRawValueMigratesToCustomLLM() {
        XCTAssertEqual(TranslationModelProvider.resolved(rawValue: "whisperKit"), .customLLM)
    }

    func testLegacyWhisperEngineRawValueMigratesToMLXAudio() {
        XCTAssertEqual(TranscriptionEngine.resolved(rawValue: "whisperKit"), .mlxAudio)
    }
}
