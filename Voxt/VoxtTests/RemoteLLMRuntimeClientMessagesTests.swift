import XCTest
@testable import Voxt

final class RemoteLLMRuntimeClientMessagesTests: XCTestCase {
    func testResponsesInputMessagesBuildsConversationHistoryAndCurrentTurn() {
        let client = RemoteLLMRuntimeClient()
        let input = client.responsesInputMessages(
            currentUserInput: "看一下大同的经纬度。",
            conversationHistory: [
                RewriteConversationPromptTurn(
                    userPromptText: "北京今天的天气怎么样？",
                    sourceText: "北京行程安排",
                    resultTitle: "大同天气查询",
                    resultContent: "请查看最新天气预报应用或网站获取大同实时天气信息。"
                )
            ]
        )
        XCTAssertEqual(input.count, 3)
        XCTAssertEqual(input.map { $0["role"] as? String }, ["user", "assistant", "user"])
        XCTAssertEqual(input.first?["content"] as? String, "Spoken instruction:\n北京今天的天气怎么样？\n\nSelected source text:\n北京行程安排")
        XCTAssertEqual(input[1]["content"] as? String, "请查看最新天气预报应用或网站获取大同实时天气信息。")
        XCTAssertEqual(input.last?["content"] as? String, "看一下大同的经纬度。")
        XCTAssertTrue(input.allSatisfy { $0["content"] is String })
    }

    func testChatConversationMessagesCarrySelectedTextOnlyInInitialHistoricalTurn() {
        let messages = RemoteLLMRuntimeClient().openAICompatibleConversationMessages(
            systemPrompt: "Answer the follow-up directly.",
            currentUserPrompt: "更简短一点",
            conversationHistory: [
                RewriteConversationPromptTurn(
                    userPromptText: "帮我回复",
                    sourceText: "明天下午三点可以吗？",
                    resultTitle: "回复",
                    resultContent: "可以，明天下午三点见。"
                )
            ]
        )
        XCTAssertEqual(messages.map { $0["role"] }, ["system", "user", "assistant", "user"])
        XCTAssertContains(messages[1]["content"] ?? "", "Spoken instruction:\n帮我回复")
        XCTAssertContains(messages[1]["content"] ?? "", "Selected source text:\n明天下午三点可以吗？")
        XCTAssertEqual(messages[2]["content"], "可以，明天下午三点见。")
        XCTAssertEqual(messages[3]["content"], "更简短一点")
    }

    func testResponsesWithoutHistoryHaveOnlyExplicitText() throws {
        let input = RemoteLLMRuntimeClient().responsesInputMessages(
            currentUserInput: "Write a reply.", conversationHistory: []
        )
        XCTAssertEqual(input.count, 1)
        XCTAssertEqual(input.first?["role"] as? String, "user")
        XCTAssertEqual(input.first?["content"] as? String, "Write a reply.")
        let data = try JSONSerialization.data(withJSONObject: input)
        let json = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertFalse(json.contains("input_image"))
        XCTAssertFalse(json.contains("image_url"))
        XCTAssertFalse(json.contains("data:image"))
    }
}
