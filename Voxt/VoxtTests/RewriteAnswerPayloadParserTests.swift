// RewriteAnswerPayloadParserTests.swift
// Provides Rewrite Answer Payload Parser Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class RewriteAnswerPayloadParserTests: XCTestCase {
    func testExtractParsesInlineJSONPayload() {
        let payload = RewriteAnswerPayloadParser.extract(
            from: #"{"title": "从北京到大同的交通方式", "content": "优先推荐高铁，其次可选普速或自驾。"}"#
        )

        XCTAssertEqual(payload?.title, "从北京到大同的交通方式")
        XCTAssertEqual(payload?.content, "优先推荐高铁，其次可选普速或自驾。")
    }

    func testExtractParsesTruncatedInlineJSONPayload() {
        let payload = RewriteAnswerPayloadParser.extract(
            from: #"{"title": "从北京到大同的交通方式", "content": "优先推荐高铁，其次可选普速列车"#
        )

        XCTAssertEqual(payload?.title, "从北京到大同的交通方式")
        XCTAssertEqual(payload?.content, "优先推荐高铁，其次可选普速列车")
    }

    func testNormalizeUnwrapsNestedStructuredJSONContent() {
        let payload = RewriteAnswerPayloadParser.normalize(
            RewriteAnswerPayload(
                title: "AI 回答",
                content: #"{"title": "路线建议", "content": "建议优先乘坐高铁。"}"#
            )
        )

        XCTAssertEqual(payload.title, "路线建议")
        XCTAssertEqual(payload.content, "建议优先乘坐高铁。")
    }

    func testNormalizeUnwrapsChatCompletionChunkDump() {
        let payload = RewriteAnswerPayloadParser.normalize(
            RewriteAnswerPayload(
                title: "AI 回答",
                content: #"""
                {"choices":[{"delta":{"content":"{\n"},"index":0}],"object":"chat.completion.chunk"}
                {"choices":[{"delta":{"content":"\"title\": \"大同特色美食推荐\", "},"index":0}],"object":"chat.completion.chunk"}
                {"choices":[{"delta":{"content":"\"content\": \"大同美食很多\"}"},"index":0}],"object":"chat.completion.chunk"}
                {"choices":[{"finish_reason":"stop","delta":{"content":""},"index":0}],"object":"chat.completion.chunk"}
                [DONE]
                """#
            )
        )

        XCTAssertEqual(payload.title, "大同特色美食推荐")
        XCTAssertEqual(payload.content, "大同美食很多")
    }

    func testExtractRecoversMalformedChunkDumpLines() {
        let payload = RewriteAnswerPayloadParser.extract(
            from: #"""
            {"choices":[{"delta":{"content":"{""},"index":0}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"title": ""},"finish_reason":null,"index":0}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"大同今日"},"finish_reason":null,"index":0}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"天气", ""},"finish_reason":null,"index":0}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"content": "大同今天"},"finish_reason":null,"index":0}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"天气晴朗。"}"},"finish_reason":null,"index":0}],"object":"chat.completion.chunk"}
            [DONE]
            """#
        )

        XCTAssertEqual(payload?.title, "大同今日天气")
        XCTAssertEqual(payload?.content, "大同今天天气晴朗。")
    }

    func testExtractIgnoresLeadingStreamingFragmentLine() {
        let payload = RewriteAnswerPayloadParser.extract(
            from: #"""
            ","role":"assistant"},"index":0,"logprobs":null,"finish_reason":null}],"object":"chat.completion.chunk","usage":null,"created":1775888214,"system_fingerprint":null,"model":"qwen-plus-latest","id":"chatcmpl-33b142b4-251d-98a8-b4f0-68c0398a1303"}
            {"choices":[{"delta":{"content":"{""},"index":0}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"title": ""},"finish_reason":null,"index":0,"logprobs":null}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"长沙所属"},"finish_reason":null,"index":0,"logprobs":null}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"省份", ""},"finish_reason":null,"index":0,"logprobs":null}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"content": "湖南省"},"finish_reason":null,"index":0,"logprobs":null}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":""}"},"finish_reason":null,"index":0,"logprobs":null}],"object":"chat.completion.chunk"}
            {"choices":[{"finish_reason":"stop","delta":{"content":""},"index":0,"logprobs":null}],"object":"chat.completion.chunk"}
            [DONE]
            """#
        )

        XCTAssertEqual(payload?.title, "长沙所属省份")
        XCTAssertEqual(payload?.content, "湖南省")
    }

    func testExtractRecoversDelimiterPunctuationFromMalformedChunkLines() {
        let payload = RewriteAnswerPayloadParser.extract(
            from: #"""
            {"choices":[{"delta":{"content":"{""},"index":0}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"title": ""},"finish_reason":null,"index":0}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"山西省"},"finish_reason":null,"index":0}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"会", ""},"finish_reason":null,"index":0}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"content": "太原"},"finish_reason":null,"index":0}],"object":"chat.completion.chunk"}
            {"choices":[{"delta":{"content":"市"}"},"finish_reason":null,"index":0}],"object":"chat.completion.chunk"}
            [DONE]
            """#
        )

        XCTAssertEqual(payload?.title, "山西省会")
        XCTAssertEqual(payload?.content, "太原市")
    }
}
