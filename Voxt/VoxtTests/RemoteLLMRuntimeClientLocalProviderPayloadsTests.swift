import XCTest
@testable import Voxt

final class RemoteLLMRuntimeClientLocalProviderPayloadsTests: XCTestCase {
    func testOllamaNativePayloadIncludesConfiguredFields() throws {
        let client = RemoteLLMRuntimeClient()

        let payload = try client.ollamaNativePayload(
            model: "qwen3",
            systemPrompt: "你是助手",
            userPrompt: "你好",
            configuration: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.ollama.rawValue,
                model: "qwen3",
                ollamaResponseFormat: OllamaResponseFormat.json.rawValue,
                ollamaThinkMode: OllamaThinkMode.medium.rawValue,
                ollamaKeepAlive: "10m",
                ollamaLogprobsEnabled: true,
                ollamaTopLogprobs: 3,
                ollamaOptionsJSON: #"{"temperature":0.7,"repeat_penalty":1.1}"#
            ),
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: true
        )

        XCTAssertEqual(payload["format"] as? String, "json")
        XCTAssertEqual(payload["think"] as? String, "medium")
        XCTAssertEqual(payload["keep_alive"] as? String, "10m")
        XCTAssertEqual(payload["logprobs"] as? Bool, true)
        XCTAssertEqual(payload["top_logprobs"] as? Int, 3)

        let options = try XCTUnwrap(payload["options"] as? [String: Any])
        XCTAssertEqual(options["temperature"] as? Double, 0.7)
        XCTAssertEqual(options["top_p"] as? Double, 0.9)
        XCTAssertEqual(options["num_predict"] as? Int, 256)
        XCTAssertEqual(options["repeat_penalty"] as? Double, 1.1)
    }

    func testOllamaGeneratePayloadUsesPromptAndSystemFields() throws {
        let client = RemoteLLMRuntimeClient()

        let payload = try client.ollamaNativePayload(
            endpointURL: URL(string: "http://localhost:11434/api/generate"),
            model: "qwen3",
            systemPrompt: "你是助手",
            userPrompt: "你好",
            configuration: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.ollama.rawValue,
                model: "qwen3",
                ollamaThinkMode: OllamaThinkMode.on.rawValue
            ),
            tuning: .init(maxTokens: 128, temperature: 0.2, topP: 0.9),
            streamingEnabled: true
        )

        XCTAssertEqual(payload["prompt"] as? String, "你好")
        XCTAssertEqual(payload["system"] as? String, "你是助手")
        XCTAssertNil(payload["messages"])
        XCTAssertEqual(payload["think"] as? Bool, true)
    }

    func testOllamaGeneratePayloadFlattensConversationMessagesIntoPrompt() throws {
        let client = RemoteLLMRuntimeClient()

        let payload = try client.ollamaNativePayload(
            endpointURL: URL(string: "http://localhost:11434/api/generate"),
            model: "qwen3",
            systemPrompt: "",
            userPrompt: "继续",
            messagesOverride: [
                ["role": "system", "content": "你是助手"],
                ["role": "user", "content": "第一问"],
                ["role": "assistant", "content": "第一答"],
                ["role": "user", "content": "继续"]
            ],
            configuration: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.ollama.rawValue,
                model: "qwen3"
            ),
            tuning: .init(maxTokens: 128, temperature: 0.2, topP: 0.9),
            streamingEnabled: false
        )

        XCTAssertEqual(payload["system"] as? String, "你是助手")
        XCTAssertEqual(
            payload["prompt"] as? String,
            """
            User:
            第一问

            Assistant:
            第一答

            User:
            继续
            """
        )
    }

    func testOllamaNativePayloadSupportsJSONObjectFormatSchema() throws {
        let client = RemoteLLMRuntimeClient()

        let payload = try client.ollamaNativePayload(
            model: "qwen3",
            systemPrompt: "",
            userPrompt: "返回结构化结果",
            configuration: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.ollama.rawValue,
                model: "qwen3",
                ollamaResponseFormat: OllamaResponseFormat.jsonSchema.rawValue,
                ollamaJSONSchema: #"{"type":"object","properties":{"answer":{"type":"string"}}}"#
            ),
            tuning: .init(maxTokens: 128, temperature: 0.2, topP: 0.9),
            streamingEnabled: false
        )

        let schema = try XCTUnwrap(payload["format"] as? [String: Any])
        XCTAssertEqual(schema["type"] as? String, "object")
    }

    func testOllamaCompatibleOverridesMapSupportedOptionKeysOnly() throws {
        let client = RemoteLLMRuntimeClient()
        var payload = client.openAICompatiblePayload(
            model: "qwen3",
            systemPrompt: "",
            userPrompt: "hi",
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: false
        )

        try client.applyOllamaCompatibleOptionOverrides(
            to: &payload,
            configuration: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.ollama.rawValue,
                model: "qwen3",
                ollamaOptionsJSON: #"{"temperature":0.4,"top_p":0.8,"num_predict":64,"repeat_penalty":1.2}"#
            )
        )

        XCTAssertEqual(payload["temperature"] as? Double, 0.4)
        XCTAssertEqual(payload["top_p"] as? Double, 0.8)
        XCTAssertEqual(payload["max_tokens"] as? Int, 64)
        XCTAssertNil(payload["repeat_penalty"])
    }

    func testOMLXGenerationSettingsMapSchemaAndExtraBody() throws {
        let client = RemoteLLMRuntimeClient()
        var payload = client.openAICompatiblePayload(
            model: "Qwen3-Coder-Next-8bit",
            systemPrompt: "",
            userPrompt: "hi",
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: true
        )

        try client.applyOMLXCompatibleConfiguration(
            to: &payload,
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.omlx.rawValue,
                model: "Qwen3-Coder-Next-8bit",
                endpoint: "",
                apiKey: "",
                omlxJSONSchema: #"{"type":"object","properties":{"answer":{"type":"string"}}}"#,
                omlxIncludeUsageStreamOptions: true,
                generationSettings: LLMGenerationSettings(
                    responseFormat: .jsonSchema,
                    extraBodyJSON: #"{"top_k":40,"min_p":0.05}"#
                )
            )
        )

        let responseFormat = try XCTUnwrap(payload["response_format"] as? [String: Any])
        XCTAssertEqual(responseFormat["type"] as? String, "json_schema")
        let streamOptions = try XCTUnwrap(payload["stream_options"] as? [String: Any])
        XCTAssertEqual(streamOptions["include_usage"] as? Bool, true)
        XCTAssertEqual(payload["top_k"] as? Int, 40)
        XCTAssertEqual(payload["min_p"] as? Double, 0.05)
    }

    func testOllamaNativePayloadMapsUnifiedGenerationSettings() throws {
        let client = RemoteLLMRuntimeClient()

        let payload = try client.ollamaNativePayload(
            model: "qwen3",
            systemPrompt: "",
            userPrompt: "hi",
            configuration: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.ollama.rawValue,
                model: "qwen3",
                generationSettings: LLMGenerationSettings(
                    maxOutputTokens: 99,
                    temperature: 0.5,
                    topP: 0.75,
                    topK: 32,
                    minP: 0.04,
                    seed: 123,
                    stop: ["<stop>"],
                    repetitionPenalty: 1.2,
                    responseFormat: .json,
                    thinking: LLMThinkingSettings(
                        mode: .off,
                        effort: nil,
                        budgetTokens: nil,
                        exposeReasoning: false
                    ),
                    extraOptionsJSON: #"{"num_ctx":8192}"#
                )
            ),
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: false
        )

        XCTAssertEqual(payload["format"] as? String, "json")
        XCTAssertEqual(payload["think"] as? Bool, false)
        let options = try XCTUnwrap(payload["options"] as? [String: Any])
        XCTAssertEqual(options["num_predict"] as? Int, 99)
        XCTAssertEqual(options["temperature"] as? Double, 0.5)
        XCTAssertEqual(options["top_p"] as? Double, 0.75)
        XCTAssertEqual(options["top_k"] as? Int, 32)
        XCTAssertEqual(options["min_p"] as? Double, 0.04)
        XCTAssertEqual(options["seed"] as? Int, 123)
        XCTAssertEqual(options["stop"] as? [String], ["<stop>"])
        XCTAssertEqual(options["repeat_penalty"] as? Double, 1.2)
        XCTAssertEqual(options["num_ctx"] as? Int, 8192)
    }
}
