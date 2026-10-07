// OverlayStateConversationTests.swift
// Provides Overlay State Conversation Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class OverlayStateConversationTests: XCTestCase {
    func testBeginRewriteConversationSeedsExistingAnswerAndSwitchesSpaceAction() {
        let state = OverlayState()
        state.sessionIconMode = .rewrite
        state.stageConversationUserPrompt("北京今天的天气怎么样？")
        state.rewriteConversationRemoteResponseID = "response-1"
        state.presentAnswer(title: "Draft", content: "First answer", canInject: true)

        XCTAssertEqual(state.answerInteractionMode, .singleResult)
        XCTAssertEqual(state.answerSpaceShortcutAction, .continueAndRecord)

        state.beginRewriteConversationIfNeeded()

        XCTAssertEqual(state.answerInteractionMode, .conversation)
        XCTAssertEqual(state.rewriteConversationTurns.count, 1)
        XCTAssertEqual(state.rewriteConversationTurns[0].userPromptText, "北京今天的天气怎么样？")
        XCTAssertEqual(state.rewriteConversationTurns[0].resultTitle, "Draft")
        XCTAssertEqual(state.rewriteConversationTurns[0].resultContent, "First answer")
        XCTAssertEqual(state.rewriteConversationRemoteResponseID, "response-1")
        XCTAssertEqual(state.answerSpaceShortcutAction, .toggleConversationRecording)
    }

    func testInitialRewriteTurnPreservesSelectedSourceForConversationHistory() {
        let state = OverlayState()
        state.sessionIconMode = .rewrite
        state.stageConversationUserPrompt("帮我写得更礼貌", sourceText: "尽快回复我")

        state.presentAnswer(title: "礼貌回复", content: "方便时请尽快回复我，谢谢。", canInject: true)

        XCTAssertEqual(state.answerInteractionMode, .singleResult)
        XCTAssertEqual(state.rewriteConversationTurns.count, 1)
        XCTAssertEqual(state.rewriteConversationTurns[0].userPromptText, "帮我写得更礼貌")
        XCTAssertEqual(state.rewriteConversationTurns[0].sourceText, "尽快回复我")
        XCTAssertNil(state.pendingConversationUserPrompt)
        XCTAssertNil(state.pendingConversationSourceText)
    }

    func testPresentAnswerInConversationAppendsPendingUserPromptAndUpdatesLatestResult() {
        let state = OverlayState()
        state.sessionIconMode = .rewrite
        state.presentAnswer(title: "Draft", content: "First answer", canInject: true)
        state.beginRewriteConversationIfNeeded()
        state.stageConversationUserPrompt("Make it shorter")

        state.presentAnswer(title: "Shorter", content: "Short answer", canInject: true)

        XCTAssertEqual(state.rewriteConversationTurns.count, 2)
        XCTAssertEqual(state.rewriteConversationTurns[1].userPromptText, "Make it shorter")
        XCTAssertTrue(state.rewriteConversationTurns[1].sourceText.isEmpty)
        XCTAssertEqual(state.rewriteConversationTurns[1].resultTitle, "Shorter")
        XCTAssertEqual(state.rewriteConversationTurns[1].resultContent, "Short answer")
        XCTAssertEqual(state.latestRewriteResult, RewriteAnswerPayload(title: "Shorter", content: "Short answer"))
        XCTAssertNil(state.pendingConversationUserPrompt)
        XCTAssertNil(state.pendingConversationSourceText)
    }

    func testContinueTurnKeepsInitialSelectedTextInHistoryWithoutRestagingIt() {
        let state = OverlayState()
        state.sessionIconMode = .rewrite
        state.stageConversationUserPrompt("帮我回复", sourceText: "明天下午三点可以吗？")
        state.presentAnswer(title: "回复", content: "可以，明天下午三点见。", canInject: true)
        state.beginRewriteConversationIfNeeded()

        state.stageConversationUserPrompt("更正式一点")
        state.presentConversationAnswer(content: "可以，期待明天下午三点与您见面。", canInject: true)

        XCTAssertEqual(state.rewriteConversationTurns.count, 2)
        XCTAssertEqual(state.rewriteConversationTurns[0].sourceText, "明天下午三点可以吗？")
        XCTAssertTrue(state.rewriteConversationTurns[1].sourceText.isEmpty)
        XCTAssertEqual(
            state.rewriteConversationPromptHistory.map(\.modelUserMessage),
            [
                """
                Spoken instruction:
                帮我回复

                Selected source text:
                明天下午三点可以吗？
                """,
                "更正式一点"
            ]
        )
    }

    func testContinueButtonHidesDuringConversationContinuationTurn() {
        let state = OverlayState()
        state.sessionIconMode = .rewrite
        state.presentAnswer(title: "Draft", content: "First answer", canInject: true)
        state.beginRewriteConversationIfNeeded()

        XCTAssertTrue(state.showsRewriteContinueButton)

        state.isRewriteConversationTurnInProgress = true

        XCTAssertFalse(state.showsRewriteContinueButton)
        XCTAssertEqual(state.answerSpaceShortcutAction, .toggleConversationRecording)

        state.presentConversationAnswer(content: "Follow-up answer", canInject: true)

        XCTAssertTrue(state.showsRewriteContinueButton)
    }

    func testAnswerSpaceShortcutUnavailableForNonRewriteAnswer() {
        let state = OverlayState()
        state.sessionIconMode = .translation
        state.presentAnswer(title: "Translation", content: "Bonjour", canInject: false)

        XCTAssertNil(state.answerSpaceShortcutAction)
        XCTAssertFalse(state.canContinueRewriteAnswer)
    }

    func testStreamingAnswerKeepsLatestCompletedPayloadForActions() {
        let state = OverlayState()
        state.sessionIconMode = .rewrite
        state.presentAnswer(title: "Draft", content: "First answer", canInject: true)
        state.beginRewriteConversationIfNeeded()

        state.presentStreamingAnswer(title: "Second Draft", content: "Working...", canInject: true)

        XCTAssertTrue(state.isStreamingAnswer)
        XCTAssertEqual(state.currentAnswerPayload, RewriteAnswerPayload(title: "Second Draft", content: "Working..."))
        XCTAssertEqual(state.latestCompletedAnswerPayload, RewriteAnswerPayload(title: "Draft", content: "First answer"))
        XCTAssertTrue(state.canCopyLatestAnswer)
    }

    func testRestoreConversationAfterFailedTurnKeepsLastAnswerAndDropsPendingPrompt() {
        let state = OverlayState()
        state.sessionIconMode = .rewrite
        state.stageConversationUserPrompt("北京今天的天气怎么样？")
        state.presentAnswer(title: "北京天气", content: "今天晴。", canInject: true)
        state.beginRewriteConversationIfNeeded()
        state.stageConversationUserPrompt("对")
        state.presentStreamingConversationAnswer(content: "正在生成", canInject: true)

        XCTAssertTrue(state.restoreLatestCompletedRewriteConversation(status: "请求失败"))

        XCTAssertEqual(state.answerTitle, "北京天气")
        XCTAssertEqual(state.answerContent, "今天晴。")
        XCTAssertEqual(state.rewriteConversationTurns.count, 1)
        XCTAssertNil(state.pendingConversationUserPrompt)
        XCTAssertFalse(state.isStreamingAnswer)
        XCTAssertFalse(state.isRewriteConversationTurnInProgress)
        XCTAssertEqual(state.statusMessage, "请求失败")
    }

    func testRemoteLLMConfigurationChangeInvalidatesProviderManagedConversationContext() async {
        let notificationCenter = NotificationCenter()
        let state = OverlayState(notificationCenter: notificationCenter)
        state.rewriteConversationRemoteResponseID = "response-1"
        state.rewriteConversationRemoteContextKey = "openAI|https://example.com/v1/responses|gpt-5"
        let requestGeneration = state.rewriteConversationRemoteContextGeneration

        notificationCenter.post(name: .voxtRemoteLLMProviderConfigurationsDidChange, object: nil)
        await Task.yield()

        XCTAssertNil(state.rewriteConversationRemoteResponseID)
        XCTAssertNil(state.rewriteConversationRemoteContextKey)
        XCTAssertFalse(
            state.storeRewriteConversationRemoteContext(
                responseID: "stale-response",
                contextKey: "openAI|https://example.com/v1/responses|gpt-5",
                expectedGeneration: requestGeneration
            )
        )
        XCTAssertNil(state.rewriteConversationRemoteResponseID)
    }
}
