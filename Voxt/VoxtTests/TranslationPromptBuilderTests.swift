// TranslationPromptBuilderTests.swift
// Provides focused coverage for translation prompt construction.

import XCTest
@testable import Voxt

final class TranslationPromptBuilderTests: XCTestCase {
    func testPromptIncludesDefaultTemplateAndRuntimeRules() {
        let prompt = TranslationPromptBuilder.build(
            systemPrompt: AppPromptDefaults.text(for: .translation, language: .english),
            targetLanguage: .english,
            sourceText: "hello world",
            userMainLanguagePromptValue: "English",
            strict: false
        )

        XCTAssertContains(prompt, "Voxt's cleanup and translation assistant")
        XCTAssertContains(prompt, "Translate to English")
        XCTAssertContains(prompt, "Return translated text only.")
    }

    func testStrictPromptKeepsScriptConstraint() {
        let prompt = TranslationPromptBuilder.build(
            systemPrompt: AppPromptDefaults.text(for: .translation, language: .english),
            targetLanguage: .chineseSimplified,
            sourceText: "test",
            userMainLanguagePromptValue: "English",
            strict: true
        )

        XCTAssertContains(prompt, "Use Simplified Chinese characters only.")
        XCTAssertContains(prompt, "Translate every linguistic token")
    }
}
