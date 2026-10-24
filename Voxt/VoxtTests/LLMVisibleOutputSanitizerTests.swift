// LLMVisibleOutputSanitizerTests.swift
// Provides LLMVisible Output Sanitizer Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class LLMVisibleOutputSanitizerTests: XCTestCase {
    func testExtractsFinalOutputFromThinkingProcessLeakForEnhancement() {
        let output = """
        Thinking Process:
        Role: Voxt transcription cleanup assistant.
        Apply Rules:
        1. Preserve final text.

        Final Output:
        测试，测试。
        """

        let sanitized = LLMVisibleOutputSanitizer.sanitize(
            output,
            fallbackText: "测试测试",
            taskKind: .enhancement
        )

        XCTAssertFalse(sanitized.didFallback)
        XCTAssertTrue(sanitized.didExtractFinalOutput)
        XCTAssertEqual(sanitized.text, "测试，测试。")
    }

    func testFallsBackWhenStrictEnhancementOutputOnlyContainsProcessText() {
        let output = """
        Thinking Process:
        The user said a short test phrase, so the answer should be concise.
        """

        let sanitized = LLMVisibleOutputSanitizer.sanitize(
            output,
            fallbackText: "测试测试",
            taskKind: .enhancement
        )

        XCTAssertTrue(sanitized.didFallback)
        XCTAssertEqual(sanitized.text, "测试测试")
    }

    func testStripsThinkTagsForGenericOutput() {
        let sanitized = LLMVisibleOutputSanitizer.sanitize(
            "<think>hidden reasoning</think>\nVisible answer",
            fallbackText: "fallback",
            taskKind: .generic
        )

        XCTAssertFalse(sanitized.didFallback)
        XCTAssertEqual(sanitized.text, "Visible answer")
    }
}
