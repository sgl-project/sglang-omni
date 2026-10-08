// ChineseScriptNormalizerTests.swift
// Provides Chinese Script Normalizer Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class ChineseScriptNormalizerTests: XCTestCase {
    func testNormalizesTraditionalToSimplifiedForSimplifiedChinesePreference() {
        let result = ChineseScriptNormalizer.normalize(
            "這是一個測試，軟件開發。",
            preferredMainLanguage: UserMainLanguageOption.option(for: "zh-Hans")!
        )

        XCTAssertEqual(result, "这是一个测试，软件开发。")
    }

    func testNormalizesSimplifiedToTraditionalForTraditionalChinesePreference() {
        let result = ChineseScriptNormalizer.normalize(
            "这是一个测试，软件开发。",
            preferredMainLanguage: UserMainLanguageOption.option(for: "zh-Hant")!
        )

        XCTAssertEqual(result, "這是一個測試，軟件開發。")
    }

    func testLeavesNonChinesePreferenceUntouched() {
        let text = "这是一个测试 with mixed English."
        let result = ChineseScriptNormalizer.normalize(
            text,
            preferredMainLanguage: UserMainLanguageOption.option(for: "en")!
        )

        XCTAssertEqual(result, text)
    }
}
