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

    @MainActor
    func testTemplateDeliveryUsesOnlyExactInputTokens() {
        XCTAssertEqual(
            TextTransformPromptSupport.deliveryForTemplate(
                "Translate {{SOURCE_TEXT}} to {{TARGET_LANGUAGE}}",
                variableTokens: ["{{SOURCE_TEXT}}"]
            ),
            .userMessage
        )
        XCTAssertEqual(
            TextTransformPromptSupport.deliveryForTemplate(
                "Rewrite {{DICTATED_PROMPT}}",
                variableTokens: ["{{DICTATED_PROMPT}}", "{{SOURCE_TEXT}}"]
            ),
            .userMessage
        )
        for template in ["", "Translate to {{TARGET_LANGUAGE}}", "Translate {{source_text}}", "Rewrite {SOURCE_TEXT}"] {
            XCTAssertEqual(
                TextTransformPromptSupport.deliveryForTemplate(
                    template,
                    variableTokens: ["{{SOURCE_TEXT}}", "{{DICTATED_PROMPT}}"]
                ),
                .systemPrompt
            )
        }
    }

    @MainActor
    func testRewriteConversationPromptAppendsPlainTextRetryOnlyWhenRequested() {
        let prompt = TextTransformPromptSupport.rewriteConversationPrompt(forceNonEmptyAnswer: false)
        let retryPrompt = TextTransformPromptSupport.rewriteConversationPrompt(forceNonEmptyAnswer: true)

        XCTAssertContains(prompt, "The previous conversation is provided as chat messages.")
        XCTAssertContains(prompt, "Never repeat or paraphrase the latest assistant reply")
        XCTAssertContains(prompt, "Return plain text only.")
        XCTAssertFalse(prompt.contains("Retry rule:"))
        XCTAssertTrue(retryPrompt.hasPrefix(prompt + "\n\nRetry rule:"))
        XCTAssertContains(retryPrompt, "empty, quoted-empty, or otherwise unusable")
        XCTAssertContains(retryPrompt, "Do not return surrounding quotes.")
    }

    @MainActor
    func testRewriteConversationPromptKeepsNewestFourTurnsInChronologicalOrder() {
        let history = (0..<6).map { index in
            RewriteConversationPromptTurn(
                userPromptText: "instruction-\(index)",
                resultTitle: "title-\(index)",
                resultContent: "answer-\(index)"
            )
        }
        let prompt = RewritePromptBuilder.build(
            systemPrompt: "Base",
            dictatedPrompt: "continue",
            sourceText: "",
            conversationHistory: TextTransformPromptSupport.trimmedRewriteConversationHistory(history),
            structuredAnswerOutput: false,
            directAnswerMode: true,
            forceNonEmptyAnswer: false
        )

        XCTAssertFalse(prompt.contains("instruction-0"))
        XCTAssertFalse(prompt.contains("instruction-1"))
        for index in 2..<6 {
            XCTAssertContains(prompt, "User: instruction-\(index)")
            XCTAssertContains(prompt, "Assistant: answer-\(index)")
        }
        XCTAssertLessThan(
            prompt.range(of: "User: instruction-2")!.lowerBound,
            prompt.range(of: "User: instruction-5")!.lowerBound
        )
    }

    @MainActor
    func testRewriteConversationPromptIncludesExactCharacterBudgetAndKeepsOversizedLatestTurn() {
        let turn = RewriteConversationPromptTurn(
            userPromptText: "x",
            resultTitle: String(repeating: "title", count: 400),
            resultContent: String(repeating: "a", count: 797)
        )
        let exactBudgetHistory = TextTransformPromptSupport.trimmedRewriteConversationHistory([turn, turn, turn])
        let prompt = RewritePromptBuilder.build(
            systemPrompt: "Base",
            dictatedPrompt: "continue",
            sourceText: "",
            conversationHistory: exactBudgetHistory,
            structuredAnswerOutput: false,
            directAnswerMode: true,
            forceNonEmptyAnswer: false
        )
        XCTAssertEqual(prompt.components(separatedBy: "Assistant: ").count - 1, 2)

        let oversizedTurn = RewriteConversationPromptTurn(
            userPromptText: "latest request",
            resultTitle: "Latest",
            resultContent: String(repeating: "界", count: 1_601)
        )
        let oversizedPrompt = RewritePromptBuilder.build(
            systemPrompt: "Base",
            dictatedPrompt: "continue",
            sourceText: "",
            conversationHistory: TextTransformPromptSupport.trimmedRewriteConversationHistory([turn, oversizedTurn]),
            structuredAnswerOutput: false,
            directAnswerMode: true,
            forceNonEmptyAnswer: false
        )
        XCTAssertContains(oversizedPrompt, "User: latest request")
        XCTAssertContains(oversizedPrompt, oversizedTurn.resultContent)
        XCTAssertFalse(oversizedPrompt.contains("User: x"))
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
