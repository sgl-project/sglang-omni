import XCTest
@testable import Voxt

final class RemoteLLMRuntimeClientStreamingTests: XCTestCase {
    func testExtractStreamingDeltaParsesAnthropicTextDelta() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "type": "content_block_delta",
            "delta": [
                "type": "text_delta",
                "text": "你好"
            ]
        ]

        XCTAssertEqual(client.extractStreamingDelta(from: payload), "你好")
    }

    func testExtractStreamingDeltaParsesOpenAIChoiceDelta() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "choices": [
                [
                    "delta": [
                        "content": " world"
                    ]
                ]
            ]
        ]

        XCTAssertEqual(client.extractStreamingDelta(from: payload), " world")
    }

    func testExtractStreamingDeltaParsesOllamaNativeMessageContent() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "message": [
                "role": "assistant",
                "content": "本地流式输出"
            ],
            "done": false
        ]

        XCTAssertEqual(client.extractStreamingDelta(from: payload), "本地流式输出")
    }

    func testExtractStreamingDeltaParsesOllamaGenerateResponse() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "response": "本地生成增量",
            "done": false
        ]

        XCTAssertEqual(client.extractStreamingDelta(from: payload), "本地生成增量")
    }

    func testExtractPrimaryTextParsesOllamaNativeResponse() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "message": [
                "role": "assistant",
                "content": "这是最终回复"
            ],
            "done": true
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "这是最终回复")
    }

    func testExtractPrimaryTextParsesOpenAICompatibleMessageStringContent() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "choices": [
                [
                    "message": [
                        "role": "assistant",
                        "content": "这是 OpenAI 兼容返回"
                    ]
                ]
            ]
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "这是 OpenAI 兼容返回")
    }

    func testExtractPrimaryTextParsesOpenAICompatibleMessageArrayContent() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "choices": [
                [
                    "message": [
                        "role": "assistant",
                        "content": [
                            [
                                "type": "text",
                                "text": "这是数组 content 返回"
                            ]
                        ]
                    ]
                ]
            ]
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "这是数组 content 返回")
    }

    func testExtractPrimaryTextParsesChoiceDeltaContentFallback() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "object": "chat.completion.chunk",
            "choices": [
                [
                    "delta": [
                        "role": "assistant",
                        "content": "这是 chunk 形状返回"
                    ],
                    "finish_reason": "stop"
                ]
            ]
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "这是 chunk 形状返回")
    }

    func testExtractPrimaryTextParsesGeminiCandidatesParts() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "candidates": [
                [
                    "content": [
                        "parts": [
                            [
                                "text": "这是 Gemini parts 返回"
                            ]
                        ]
                    ]
                ]
            ]
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "这是 Gemini parts 返回")
    }

    func testShouldFlushBufferedEventLinesRecognizesSingleChunkJSONWithoutBlankSeparator() {
        let client = RemoteLLMRuntimeClient()
        let bufferedEventLines = [
            #"{"choices":[{"delta":{"content":"大"},"index":0}]}"#
        ]

        XCTAssertTrue(client.shouldFlushBufferedEventLines(bufferedEventLines))
    }

    func testShouldFlushBufferedEventLinesRecognizesDoneMarkerWithoutBlankSeparator() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertTrue(client.shouldFlushBufferedEventLines(["[DONE]"]))
    }

    func testExtractStreamingDeltaParsesDashScopeChunkPayload() throws {
        let client = RemoteLLMRuntimeClient()
        let raw = #"{"choices":[{"delta":{"content":"大同市的经纬度约为：北纬39.98°，东经113.30°。"},"index":0,"logprobs":null,"finish_reason":null}],"object":"chat.completion.chunk","usage":null,"created":1775896175,"system_fingerprint":null,"model":"qwen-plus-latest","id":"chatcmpl-920daaa7-d5a8-9df8-a704-8078ff684102"}"#
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any])

        XCTAssertEqual(
            client.extractStreamingDelta(from: object),
            "大同市的经纬度约为：北纬39.98°，东经113.30°。"
        )
    }

    func testExtractStreamingDeltaParsesResponsesDeltaEvent() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "type": "response.output_text.delta",
            "delta": "山西大同"
        ]

        XCTAssertEqual(client.extractStreamingDelta(from: payload), "山西大同")
    }

    func testExtractPrimaryTextParsesResponsesOutputArray() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "output": [
                [
                    "content": [
                        [
                            "type": "output_text",
                            "text": "北纬 40.076，东经 113.300"
                        ]
                    ]
                ]
            ]
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "北纬 40.076，东经 113.300")
    }

    func testExtractPrimaryTextMergesAllResponsesMessageItems() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "output": [
                [
                    "type": "message",
                    "content": [["type": "output_text", "text": "第一段"]]
                ],
                [
                    "type": "message",
                    "content": [["type": "output_text", "text": "第二段"]]
                ]
            ]
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "第一段\n第二段")
    }

    func testResponsesCompletionIssueReportsIncompleteReason() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "status": "incomplete",
            "incomplete_details": ["reason": "max_output_tokens"]
        ]

        XCTAssertEqual(
            client.responsesCompletionIssue(from: payload),
            "Responses API returned status 'incomplete' (max_output_tokens)."
        )
        XCTAssertNil(client.responsesCompletionIssue(from: ["status": "completed"]))
    }

    func testStructuredJSONObjectValidationRejectsTruncatedPrefix() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertFalse(client.isValidStructuredJSONObject(#"{"title"#))
        XCTAssertTrue(client.isValidStructuredJSONObject(#"{"title":"天气","content":"晴"}"#))
        XCTAssertFalse(client.isValidStructuredJSONObject(#"["not", "an", "object"]"#))
    }

    func testExtractPrimaryTextParsesResponsesOutputArrayWithStringContent() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "output": [
                [
                    "type": "message",
                    "content": "这是百炼返回的字符串内容"
                ]
            ]
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "这是百炼返回的字符串内容")
    }

    func testExtractPrimaryTextParsesResponsesOutputArrayWithNestedTextValue() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "output": [
                [
                    "type": "message",
                    "content": [
                        [
                            "type": "output_text",
                            "text": [
                                "value": "这是嵌套 text.value 返回"
                            ]
                        ]
                    ]
                ]
            ]
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "这是嵌套 text.value 返回")
    }

    func testExtractPrimaryTextParsesResponsesOutputArrayWithMixedToolAndMessageItems() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "output": [
                [
                    "type": "web_search_call",
                    "output": "{\"ok\":true}"
                ],
                [
                    "type": "message",
                    "content": [
                        [
                            "type": "output_text",
                            "text": "这是最终增强文本"
                        ]
                    ]
                ]
            ]
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "这是最终增强文本")
    }

    func testExtractPrimaryTextIgnoresResponsesReasoningItems() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "output": [
                [
                    "type": "reasoning",
                    "summary": [
                        [
                            "type": "summary_text",
                            "text": "这是推理摘要"
                        ]
                    ]
                ],
                [
                    "type": "message",
                    "content": [
                        [
                            "type": "output_text",
                            "text": "这是最终文本"
                        ]
                    ]
                ]
            ]
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "这是最终文本")
    }

    func testExtractPrimaryTextIgnoresAnthropicThinkingBlocks() {
        let client = RemoteLLMRuntimeClient()
        let payload: [String: Any] = [
            "content": [
                [
                    "type": "thinking",
                    "text": "hidden chain of thought"
                ],
                [
                    "type": "text",
                    "text": "visible answer"
                ]
            ]
        ]

        XCTAssertEqual(client.extractPrimaryText(from: payload), "visible answer")
    }

    func testExtractPrimaryTextIgnoresReasoningContentOnlyMessages() {
        let client = RemoteLLMRuntimeClient()
        let reasoningOnly: [String: Any] = [
            "choices": [
                [
                    "message": [
                        "role": "assistant",
                        "reasoning_content": "hidden reasoning"
                    ]
                ]
            ]
        ]
        let visible: [String: Any] = [
            "choices": [
                [
                    "message": [
                        "role": "assistant",
                        "reasoning_content": "hidden reasoning",
                        "content": "visible content"
                    ]
                ]
            ]
        ]

        XCTAssertNil(client.extractPrimaryText(from: reasoningOnly))
        XCTAssertEqual(client.extractPrimaryText(from: visible), "visible content")
    }

    func testResponsesResponseIDParsesNestedAndTopLevelForms() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertEqual(
            client.responsesResponseID(
                from: [
                    "response": [
                        "id": "resp_nested"
                    ]
                ]
            ),
            "resp_nested"
        )
        XCTAssertEqual(
            client.responsesResponseID(
                from: [
                    "response_id": "resp_top_level"
                ]
            ),
            "resp_top_level"
        )
    }

    func testDecodeResponsesObjectAcceptsJSONResponseObject() throws {
        let client = RemoteLLMRuntimeClient()
        let response = try XCTUnwrap(HTTPURLResponse(
            url: URL(string: "https://api.openai.com/v1/responses")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"]
        ))
        let body = #"{"id":"resp_123","output_text":"优化后的文本"}"#

        XCTAssertEqual(
            try client.decodeResponsesObject(from: Data(body.utf8), response: response)["output_text"] as? String,
            "优化后的文本"
        )
    }

    func testDecodeResponsesObjectRejectsHTMLGatewayPage() throws {
        let client = RemoteLLMRuntimeClient()
        let response = try XCTUnwrap(HTTPURLResponse(
            url: URL(string: "https://api.openai.com/v1/responses")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "text/html; charset=utf-8"]
        ))
        let body = """
        <!doctype html>
        <html><head><title>AI API Gateway</title></head><body></body></html>
        """

        XCTAssertThrowsError(
            try client.decodeResponsesObject(from: Data(body.utf8), response: response)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("returned HTML instead of JSON"))
        }
    }

    func testDecodeResponsesObjectRejectsEventStreamForNonStreamingResponse() throws {
        let client = RemoteLLMRuntimeClient()
        let response = try XCTUnwrap(HTTPURLResponse(
            url: URL(string: "https://api.openai.com/v1/responses")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "text/event-stream"]
        ))
        let body = #"data: {"type":"response.output_text.delta","delta":"你好"}"#

        XCTAssertThrowsError(
            try client.decodeResponsesObject(from: Data(body.utf8), response: response)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("event stream for a non-streaming request"))
        }
    }
}
