import XCTest
@testable import Voxt

final class RewriteAnswerContentNormalizerTests: XCTestCase {
    private let weatherHistory = [
        RewriteConversationPromptTurn(
            userPromptText: "北京今天的天气情况",
            resultTitle: "北京今日天气",
            resultContent: "请问您需要查询哪一天的天气？"
        )
    ]

    func testStreamingPreviewUnwrapsTruncatedStructuredDraft() {
        let preview = RewriteAnswerContentNormalizer.normalizePlainTextStreamingPreview(
            #"{"title": "路线建议", "content": "建议优先乘坐高铁"#
        )
        XCTAssertEqual(preview, "建议优先乘坐高铁")
    }

    func testRepeatedAssistantClarificationIsRejectedAfterConfirmation() {
        XCTAssertTrue(
            RewriteAnswerContentNormalizer.repeatsLatestAssistantAnswer(
                "请问您需要查询哪一天的天气?",
                dictatedPrompt: "对",
                conversationHistory: weatherHistory
            )
        )
    }

    func testDifferentFollowUpAnswerIsAccepted() {
        XCTAssertFalse(
            RewriteAnswerContentNormalizer.repeatsLatestAssistantAnswer(
                "我无法联网核实北京今天的实时天气，请查看系统天气应用。",
                dictatedPrompt: "对",
                conversationHistory: weatherHistory
            )
        )
    }

    func testExplicitRepeatRequestMayRepeatAssistantAnswer() {
        XCTAssertFalse(
            RewriteAnswerContentNormalizer.repeatsLatestAssistantAnswer(
                "请问您需要查询哪一天的天气？",
                dictatedPrompt: "再说一遍",
                conversationHistory: weatherHistory
            )
        )
    }

    func testKeepUnchangedRequestMayReturnLatestAssistantAnswer() {
        for prompt in [
            "保持原样",
            "不用改了",
            "keep it as is",
            "そのままで"
        ] {
            XCTAssertFalse(
                RewriteAnswerContentNormalizer.repeatsLatestAssistantAnswer(
                    "请问您需要查询哪一天的天气？",
                    dictatedPrompt: prompt,
                    conversationHistory: weatherHistory
                ),
                "Expected unchanged-answer intent to be accepted for: \(prompt)"
            )
        }
    }

    func testPartialPreservationRequestStillRejectsUnchangedAnswer() {
        XCTAssertTrue(
            RewriteAnswerContentNormalizer.repeatsLatestAssistantAnswer(
                "请问您需要查询哪一天的天气？",
                dictatedPrompt: "不要修改语气，但请缩短一些",
                conversationHistory: weatherHistory
            )
        )
    }

    func testTruncatedStructuredJSONIsRejected() {
        XCTAssertTrue(
            RewriteAnswerContentNormalizer.isUnusableStructuredAnswer(
                #"{"title"#,
                dictatedPrompt: "北京今天的天气"
            )
        )
    }

    func testCompleteStructuredAnswerIsAccepted() {
        XCTAssertFalse(
            RewriteAnswerContentNormalizer.isUnusableStructuredAnswer(
                #"{"title":"北京天气","content":"今天晴。"}"#,
                dictatedPrompt: "北京今天的天气"
            )
        )
    }
}
