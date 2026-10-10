// PromptBuildersTests.swift
// Provides Prompt Builders Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class PromptBuildersTests: XCTestCase {
    func testTranslationPromptBuilderReplacesVariablesAndAddsStrictRules() {
        let prompt = TranslationPromptBuilder.build(
            systemPrompt: "Translate {{SOURCE_TEXT}} to {{TARGET_LANGUAGE}} for {{USER_MAIN_LANGUAGE}}",
            targetLanguage: .japanese,
            sourceText: "hello",
            userMainLanguagePromptValue: "English",
            strict: true
        )

        XCTAssertContains(prompt, "Translate hello to Japanese for English")
        XCTAssertContains(prompt, "Translate every linguistic token into Japanese")
        XCTAssertContains(prompt, "Return translated text only.")
    }

    func testTranslationPromptBuilderAddsTraditionalChineseScriptConstraint() {
        let prompt = TranslationPromptBuilder.build(
            systemPrompt: "Translate {{SOURCE_TEXT}} to {{TARGET_LANGUAGE}}",
            targetLanguage: .chineseTraditional,
            sourceText: "你好",
            userMainLanguagePromptValue: "English",
            strict: true
        )

        XCTAssertContains(prompt, "Translate 你好 to Traditional Chinese")
        XCTAssertContains(prompt, "Use Traditional Chinese characters only.")
        XCTAssertContains(prompt, "Do not output Simplified Chinese characters.")
    }

    func testTranslationPromptBuilderAddsSimplifiedChineseScriptConstraint() {
        let prompt = TranslationPromptBuilder.build(
            systemPrompt: "Translate {{SOURCE_TEXT}} to {{TARGET_LANGUAGE}}",
            targetLanguage: .chineseSimplified,
            sourceText: "你好",
            userMainLanguagePromptValue: "English",
            strict: false
        )

        XCTAssertContains(prompt, "Translate 你好 to Simplified Chinese")
        XCTAssertContains(prompt, "Use Simplified Chinese characters only.")
        XCTAssertContains(prompt, "Do not output Traditional Chinese characters.")
    }

    func testRewritePromptBuilderAppendsConstraintsInStableOrder() {
        let prompt = RewritePromptBuilder.build(
            systemPrompt: "Base {{DICTATED_PROMPT}} / {{SOURCE_TEXT}}",
            dictatedPrompt: "reply politely",
            sourceText: "",
            conversationHistory: [],
            structuredAnswerOutput: true,
            directAnswerMode: true,
            forceNonEmptyAnswer: true
        )

        XCTAssertContains(prompt, "Base reply politely / ")
        XCTAssertTrue(prompt.contains("Direct-answer mode:"))
        XCTAssertContains(prompt, "asking the user to confirm details they already supplied")
        XCTAssertContains(prompt, "A place, subject, date, or other qualifier explicitly present")
        XCTAssertTrue(prompt.contains("Runtime output format rules:"))
        XCTAssertTrue(prompt.contains("Retry rule:"))
        XCTAssertLessThan(
            prompt.range(of: "Direct-answer mode:")!.lowerBound,
            prompt.range(of: "Runtime output format rules:")!.lowerBound
        )
    }

    func testRewritePromptBuilderAddsPlainTextRuntimeRulesWhenNotStructured() {
        let prompt = RewritePromptBuilder.build(
            systemPrompt: "Base {{DICTATED_PROMPT}} / {{SOURCE_TEXT}}",
            dictatedPrompt: "reply",
            sourceText: "source",
            conversationHistory: [],
            structuredAnswerOutput: false,
            directAnswerMode: false,
            forceNonEmptyAnswer: false
        )

        XCTAssertContains(prompt, "Base reply / source")
        XCTAssertContains(prompt, "Runtime output format rules:")
        XCTAssertContains(prompt, "Return plain text only.")
        XCTAssertContains(prompt, "Do not return JSON")
    }

    func testRewritePromptBuilderAppendsConversationHistoryBeforeRuntimeConstraints() {
        let prompt = RewritePromptBuilder.build(
            systemPrompt: "Base {{DICTATED_PROMPT}} / {{SOURCE_TEXT}}",
            dictatedPrompt: "make it shorter",
            sourceText: "",
            conversationHistory: [
                RewriteConversationPromptTurn(
                    userPromptText: "write a reply",
                    sourceText: "Thanks for reaching out.",
                    resultTitle: "Initial Draft",
                    resultContent: "Thanks for your note. Here is the full version."
                ),
                RewriteConversationPromptTurn(
                    userPromptText: "make it warmer",
                    resultTitle: "Warmer Draft",
                    resultContent: "Thanks so much for your note. Here is the full version."
                )
            ],
            structuredAnswerOutput: true,
            directAnswerMode: true,
            forceNonEmptyAnswer: false
        )

        XCTAssertContains(prompt, "Previous conversation:")
        XCTAssertContains(prompt, "Spoken instruction:")
        XCTAssertContains(prompt, "Selected source text:")
        XCTAssertContains(prompt, "Assistant: Thanks for your note. Here is the full version.")
        XCTAssertContains(prompt, "User: make it warmer")
        XCTAssertFalse(prompt.contains("Assistant Title:"))
        XCTAssertLessThan(
            prompt.range(of: "Previous conversation:")!.lowerBound,
            prompt.range(of: "Runtime output format rules:")!.lowerBound
        )
    }

    func testRewritePromptBuilderAddsConversationPlainTextRulesForContinueMode() {
        let prompt = RewritePromptBuilder.build(
            systemPrompt: "Base {{DICTATED_PROMPT}} / {{SOURCE_TEXT}}",
            dictatedPrompt: "继续展开",
            sourceText: "",
            conversationHistory: [
                RewriteConversationPromptTurn(
                    userPromptText: "",
                    resultTitle: "山西省会",
                    resultContent: "山西省的省会是太原。"
                )
            ],
            structuredAnswerOutput: false,
            directAnswerMode: true,
            forceNonEmptyAnswer: true
        )

        XCTAssertContains(prompt, "Conversation mode:")
        XCTAssertContains(prompt, "Treat a short confirmation or correction")
        XCTAssertContains(prompt, "Do not repeat the latest assistant answer")
        XCTAssertContains(prompt, "Return the next assistant reply as plain text only.")
        XCTAssertContains(prompt, "Do not return JSON, markdown fences, labels, or quotes.")
        XCTAssertContains(prompt, "A previous answer was empty or unusable.")
    }

    func testRewriteDirectAnswerRuntimeGuidanceDeclaresDateTimeZoneAndLiveAccess() throws {
        let timeZone = try XCTUnwrap(TimeZone(identifier: "Asia/Shanghai"))
        let date = try XCTUnwrap(
            Calendar(identifier: .gregorian).date(
                from: DateComponents(
                    timeZone: timeZone,
                    year: 2026,
                    month: 7,
                    day: 15,
                    hour: 9,
                    minute: 30
                )
            )
        )

        let guidance = RewriteDirectAnswerRuntimeGuidance.content(
            now: date,
            timeZone: timeZone,
            liveInformationAccess: false
        )

        XCTAssertContains(guidance, "2026-07-15 09:30")
        XCTAssertContains(guidance, "Asia/Shanghai")
        XCTAssertContains(guidance, "Live information lookup: unavailable")
        XCTAssertContains(guidance, "今天")
    }

}
