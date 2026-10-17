import XCTest
@testable import Voxt

final class RemoteLLMRuntimeClientResponsesRequestTests: XCTestCase {
    func testMakeResponsesRequestBuildsSingleTurnAliyunPayload() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .aliyunBailian,
            endpointValue: "https://dashscope.aliyuncs.com/compatible-mode/v1/responses",
            model: "qwen-plus",
            systemPrompt: "你是助手",
            inputPayload: "山西大同的经纬度是什么？",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.aliyunBailian.rawValue,
                model: "qwen-plus",
                endpoint: "",
                apiKey: "test-key",
                searchEnabled: true
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9),
            textFormat: nil,
            streamingEnabled: true
        )

        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])

        XCTAssertEqual(object["instructions"] as? String, "你是助手")
        XCTAssertEqual(object["input"] as? String, "山西大同的经纬度是什么？")
        let tools = try XCTUnwrap(object["tools"] as? [[String: Any]])
        XCTAssertEqual(tools.first?["type"] as? String, "web_search")
    }

    func testMakeResponsesRequestBuildsContinuePayloadWithPreviousResponseID() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .volcengine,
            endpointValue: "https://ark.cn-beijing.volces.com/api/v3/responses",
            model: "doubao-1-5-pro",
            systemPrompt: "",
            inputPayload: "继续",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.volcengine.rawValue,
                model: "doubao-1-5-pro",
                endpoint: "",
                apiKey: "test-key",
                searchEnabled: true
            ),
            previousResponseID: "resp_123",
            tuning: .init(maxTokens: 256, temperature: 0.1, topP: 0.8),
            textFormat: nil,
            streamingEnabled: false
        )

        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])

        XCTAssertEqual(object["previous_response_id"] as? String, "resp_123")
        XCTAssertEqual(object["input"] as? String, "继续")
        let tools = try XCTUnwrap(object["tools"] as? [[String: Any]])
        XCTAssertEqual(tools.first?["type"] as? String, "web_search")
    }

    func testMakeResponsesRequestAppliesOpenAIOptions() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .openAI,
            endpointValue: "https://api.openai.com/v1/responses",
            model: "gpt-5.2",
            systemPrompt: "",
            inputPayload: "ping",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.openAI.rawValue,
                model: "gpt-5.2",
                endpoint: "",
                apiKey: "test-key",
                openAIReasoningEffort: OpenAIReasoningEffort.high.rawValue,
                openAITextVerbosity: OpenAITextVerbosity.low.rawValue,
                openAIMaxOutputTokens: 2048
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9),
            textFormat: [
                "format": [
                    "type": "json_object"
                ]
            ],
            streamingEnabled: false
        )

        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let reasoning = try XCTUnwrap(object["reasoning"] as? [String: Any])
        let text = try XCTUnwrap(object["text"] as? [String: Any])
        let format = try XCTUnwrap(text["format"] as? [String: Any])

        XCTAssertEqual(object["max_output_tokens"] as? Int, 2048)
        XCTAssertNil(object["temperature"])
        XCTAssertNil(object["top_p"])
        XCTAssertEqual(reasoning["effort"] as? String, "high")
        XCTAssertEqual(text["verbosity"] as? String, "low")
        XCTAssertEqual(format["type"] as? String, "json_object")
    }

    func testMakeVolcengineStructuredResponsesRequestDisablesDefaultThinking() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .volcengine,
            endpointValue: "https://ark.cn-beijing.volces.com/api/v3/responses",
            model: "doubao-seed-2-0-mini-260215",
            systemPrompt: "Return JSON.",
            inputPayload: "北京今天的天气",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.volcengine.rawValue,
                model: "doubao-seed-2-0-mini-260215",
                endpoint: "https://ark.cn-beijing.volces.com/api/v3/responses",
                apiKey: "test-key"
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 384, temperature: 0.1, topP: 0.3),
            textFormat: client.responsesTextFormat(for: .jsonObject),
            streamingEnabled: false
        )

        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let thinking = try XCTUnwrap(object["thinking"] as? [String: Any])
        let text = try XCTUnwrap(object["text"] as? [String: Any])
        let format = try XCTUnwrap(text["format"] as? [String: Any])

        XCTAssertEqual(thinking["type"] as? String, "disabled")
        XCTAssertEqual(format["type"] as? String, "json_object")
    }

    func testMakeVolcengineStructuredResponsesRequestPreservesExplicitThinkingChoice() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .volcengine,
            endpointValue: "https://ark.cn-beijing.volces.com/api/v3/responses",
            model: "doubao-seed-2-0-mini-260215",
            systemPrompt: "Return JSON.",
            inputPayload: "北京今天的天气",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.volcengine.rawValue,
                model: "doubao-seed-2-0-mini-260215",
                endpoint: "https://ark.cn-beijing.volces.com/api/v3/responses",
                apiKey: "test-key",
                generationSettings: LLMGenerationSettings(
                    thinking: LLMThinkingSettings(
                        mode: .on,
                        effort: nil,
                        budgetTokens: nil,
                        exposeReasoning: false
                    )
                )
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 384, temperature: 0.1, topP: 0.3),
            textFormat: client.responsesTextFormat(for: .jsonObject),
            streamingEnabled: false
        )

        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let thinking = try XCTUnwrap(object["thinking"] as? [String: Any])

        XCTAssertEqual(thinking["type"] as? String, "enabled")
    }

    func testMakeResponsesRequestUsesJSONAcceptHeaderWhenNonStreaming() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .openAI,
            endpointValue: "https://api.openai.com/v1/responses",
            model: "gpt-5.2",
            systemPrompt: "",
            inputPayload: "ping",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.openAI.rawValue,
                model: "gpt-5.2",
                endpoint: "",
                apiKey: "test-key"
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9),
            textFormat: nil,
            streamingEnabled: false
        )

        XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "application/json")
    }

    func testMakeResponsesRequestUsesEventStreamAcceptHeaderWhenStreaming() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .openAI,
            endpointValue: "https://api.openai.com/v1/responses",
            model: "gpt-5.2",
            systemPrompt: "",
            inputPayload: "ping",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.openAI.rawValue,
                model: "gpt-5.2",
                endpoint: "",
                apiKey: "test-key"
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9),
            textFormat: nil,
            streamingEnabled: true
        )

        XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "text/event-stream, application/json")
    }

    func testMakeResponsesRequestFiltersOpenAIOptionsByModelFamily() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .openAI,
            endpointValue: "https://api.openai.com/v1/responses",
            model: "gpt-5",
            systemPrompt: "",
            inputPayload: "ping",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.openAI.rawValue,
                model: "gpt-5",
                endpoint: "",
                apiKey: "test-key",
                openAIReasoningEffort: OpenAIReasoningEffort.none.rawValue,
                openAITextVerbosity: OpenAITextVerbosity.high.rawValue
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9),
            textFormat: nil,
            streamingEnabled: false
        )

        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let text = try XCTUnwrap(object["text"] as? [String: Any])

        XCTAssertNil(object["reasoning"])
        XCTAssertEqual(text["verbosity"] as? String, "high")
    }

    func testMakeResponsesRequestOmitsOpenAIModelOptionsForNonSupportingModel() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .openAI,
            endpointValue: "https://api.openai.com/v1/responses",
            model: "gpt-4o",
            systemPrompt: "",
            inputPayload: "ping",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.openAI.rawValue,
                model: "gpt-4o",
                endpoint: "",
                apiKey: "test-key",
                openAIReasoningEffort: OpenAIReasoningEffort.high.rawValue,
                openAITextVerbosity: OpenAITextVerbosity.low.rawValue
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9),
            textFormat: nil,
            streamingEnabled: false
        )

        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])

        XCTAssertNil(object["temperature"])
        XCTAssertNil(object["top_p"])
        XCTAssertNil(object["reasoning"])
        XCTAssertNil(object["text"])
    }

    func testMakeResponsesRequestDoesNotApplyOpenAIOptionsToCompatibleProviders() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .aliyunBailian,
            endpointValue: "https://dashscope.aliyuncs.com/compatible-mode/v1/responses",
            model: "qwen-plus",
            systemPrompt: "",
            inputPayload: "ping",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.aliyunBailian.rawValue,
                model: "qwen-plus",
                endpoint: "",
                apiKey: "test-key",
                openAIReasoningEffort: OpenAIReasoningEffort.high.rawValue,
                openAITextVerbosity: OpenAITextVerbosity.low.rawValue,
                openAIMaxOutputTokens: 2048
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9),
            textFormat: nil,
            streamingEnabled: false
        )

        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])

        XCTAssertEqual(object["max_output_tokens"] as? Int, 512)
        XCTAssertEqual(object["temperature"] as? Double, 0.2)
        XCTAssertEqual(object["top_p"] as? Double, 0.9)
        XCTAssertNil(object["reasoning"])
        XCTAssertNil(object["text"])
    }
}
