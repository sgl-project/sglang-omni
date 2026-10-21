// EnhancementLanguageGuardTests.swift
// Provides Enhancement Language Guard Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class EnhancementLanguageGuardTests: XCTestCase {
    private let chinese = "明天下午三点我们在三楼会议室开会，讨论下个季度的发布计划，请大家提前准备好材料。"
    private let english = "We will meet tomorrow at three in the third floor meeting room to discuss next quarter's release plan. Please prepare your materials in advance."

    func testDominantScriptOfPlainChineseAndEnglish() {
        XCTAssertEqual(EnhancementLanguageGuard.dominantScript(in: chinese), .cjk)
        XCTAssertEqual(EnhancementLanguageGuard.dominantScript(in: english), .latin)
        XCTAssertEqual(EnhancementLanguageGuard.dominantScript(in: "来週の月曜日に資料を送ります。"), .cjk)
        XCTAssertEqual(EnhancementLanguageGuard.dominantScript(in: "다음 주 월요일에 자료를 보내겠습니다"), .cjk)
    }

    func testChineseWithEnglishTermsIsStillChinese() {
        let mixed = "这个 feature 我们下个 sprint 再做，因为 API 文档还没写完，Swift 的部分也要再测一下。"
        XCTAssertEqual(EnhancementLanguageGuard.dominantScript(in: mixed), .cjk)
    }

    func testBalancedCodeSwitchingHasNoDominantScript() {
        XCTAssertNil(EnhancementLanguageGuard.dominantScript(in: "我们 review 一下 this PR 的 diff and merge"))
    }

    func testShortTextHasNoDominantScript() {
        XCTAssertNil(EnhancementLanguageGuard.dominantScript(in: "好的"))
        XCTAssertNil(EnhancementLanguageGuard.dominantScript(in: "OK thanks"))
        XCTAssertNil(EnhancementLanguageGuard.dominantScript(in: "123，。!"))
    }

    func testChineseAnsweredInEnglishIsAFlipAndViceVersa() {
        XCTAssertTrue(EnhancementLanguageGuard.languageFlipped(input: chinese, output: english))
        XCTAssertTrue(EnhancementLanguageGuard.languageFlipped(input: english, output: chinese))
    }

    func testCleanupInTheSameLanguageIsNotAFlip() {
        let cleaned = "明天下午 3 点，我们在三楼会议室开会，讨论下季度的发布计划。请大家提前准备材料。"
        XCTAssertFalse(EnhancementLanguageGuard.languageFlipped(input: chinese, output: cleaned))
        XCTAssertFalse(EnhancementLanguageGuard.languageFlipped(input: english, output: english))
    }

    func testMixedChineseAndEnglishCleanupIsNotAFlip() {
        let input = "嗯那个这个 feature 我们下个 sprint 再做吧，因为 API 文档还没写完，然后 Swift 那部分也要再测一下"
        let output = "这个 feature 我们下个 sprint 再做，因为 API 文档还没写完，Swift 的部分也要再测一下。"
        XCTAssertFalse(EnhancementLanguageGuard.languageFlipped(input: input, output: output))
    }

    func testEnglishWithAChineseNameIsNotAFlip() {
        let input = "Please send the quarterly report to 王小明 before Friday and copy the whole team on it."
        XCTAssertFalse(EnhancementLanguageGuard.languageFlipped(input: input, output: input))
    }

    func testBalancedInputNeverTriggersEvenWithEnglishOutput() {
        XCTAssertFalse(EnhancementLanguageGuard.languageFlipped(
            input: "我们 review 一下 this PR 的 diff and merge",
            output: "Let's review this PR's diff and merge it."
        ))
    }

    func testShortReplyIsNotJudged() {
        XCTAssertFalse(EnhancementLanguageGuard.languageFlipped(input: "好的，谢谢", output: "OK, thanks."))
    }

    func testJapaneseAndChineseShareAScript() {
        XCTAssertFalse(EnhancementLanguageGuard.languageFlipped(input: "来週の月曜日に資料を送ります。", output: chinese))
    }

    func testOutputInTheInputLanguageIsReturnedWithoutRetry() async throws {
        var retries = 0
        var logs: [String] = []
        let result = try await EnhancementLanguageGuard.keepingInputLanguage(
            input: chinese,
            output: "明天下午三点开会。",
            retry: { retries += 1; return nil },
            log: { logs.append($0) }
        )
        XCTAssertEqual(result, "明天下午三点开会。")
        XCTAssertEqual(retries, 0)
        XCTAssertTrue(logs.isEmpty)
    }

    func testFlippedOutputIsRetriedOnceAndTheRetryIsUsed() async throws {
        var retries = 0
        var logs: [String] = []
        let result = try await EnhancementLanguageGuard.keepingInputLanguage(
            input: chinese,
            output: english,
            retry: { retries += 1; return "明天下午三点，我们在三楼会议室开会。" },
            log: { logs.append($0) }
        )
        XCTAssertEqual(result, "明天下午三点，我们在三楼会议室开会。")
        XCTAssertEqual(retries, 1)
        XCTAssertEqual(logs.count, 1)
    }

    func testStillFlippedAfterRetryReturnsTheInput() async throws {
        var retries = 0
        var logs: [String] = []
        let result = try await EnhancementLanguageGuard.keepingInputLanguage(
            input: chinese,
            output: english,
            retry: { retries += 1; return english },
            log: { logs.append($0) }
        )
        XCTAssertEqual(result, chinese)
        XCTAssertEqual(retries, 1)
        XCTAssertEqual(logs.count, 2)
    }

    func testRetryThatCannotRunReturnsTheInput() async throws {
        let result = try await EnhancementLanguageGuard.keepingInputLanguage(
            input: chinese,
            output: english,
            retry: { nil },
            log: { _ in }
        )
        XCTAssertEqual(result, chinese)
    }

    func testRetryInstructionNamesTheSpokenLanguage() {
        XCTAssertTrue(EnhancementLanguageGuard.retryInstruction(for: chinese).hasPrefix("The input is spoken Chinese. Write the cleaned text in Chinese"))
        XCTAssertTrue(EnhancementLanguageGuard.retryInstruction(for: "来週の月曜日に資料を送ります。").contains("spoken Japanese"))
        XCTAssertTrue(EnhancementLanguageGuard.retryInstruction(for: "다음 주 월요일에 자료를 보내겠습니다").contains("spoken Korean"))
        XCTAssertTrue(EnhancementLanguageGuard.retryInstruction(for: english).contains("in that same language"))
    }
}
