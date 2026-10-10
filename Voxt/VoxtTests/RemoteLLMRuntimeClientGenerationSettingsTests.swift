import XCTest
@testable import Voxt

final class RemoteLLMRuntimeClientGenerationSettingsTests: XCTestCase {
    func testOpenAICompatiblePayloadOmitsResponseFormatByDefault() {
        let client = RemoteLLMRuntimeClient()

        let payload = client.openAICompatiblePayload(
            model: "deepseek-v4-flash",
            systemPrompt: "你是助手",
            userPrompt: "你好",
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: false
        )

        XCTAssertNil(payload["response_format"])
    }

    func testOpenAICompatiblePayloadAddsJSONModeWhenRequested() throws {
        let client = RemoteLLMRuntimeClient()

        let payload = client.openAICompatiblePayload(
            model: "deepseek-v4-flash",
            systemPrompt: "返回 JSON",
            userPrompt: "生成结构化结果",
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: true,
            responseFormat: .jsonObject
        )

        let responseFormat = try XCTUnwrap(payload["response_format"] as? [String: Any])
        XCTAssertEqual(responseFormat["type"] as? String, "json_object")
        XCTAssertEqual(payload["stream"] as? Bool, true)
    }

    func testAnthropicGenerationSettingsMapCoreParametersAndThinkingBudget() throws {
        let client = RemoteLLMRuntimeClient()
        var payload: [String: Any] = [
            "model": "claude-sonnet-4-6",
            "messages": []
        ]

        client.applyAnthropicGenerationSettings(
            to: &payload,
            settings: LLMGenerationSettings(
                maxOutputTokens: 1024,
                temperature: 0.3,
                topP: 0.8,
                topK: 40,
                stop: ["</final>"],
                thinking: LLMThinkingSettings(
                    mode: .budget,
                    effort: nil,
                    budgetTokens: 256,
                    exposeReasoning: false
                )
            ),
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9)
        )

        XCTAssertEqual(payload["max_tokens"] as? Int, 1024)
        XCTAssertEqual(payload["temperature"] as? Double, 0.3)
        XCTAssertEqual(payload["top_p"] as? Double, 0.8)
        XCTAssertEqual(payload["top_k"] as? Int, 40)
        XCTAssertEqual(payload["stop_sequences"] as? [String], ["</final>"])
        let thinking = try XCTUnwrap(payload["thinking"] as? [String: Any])
        XCTAssertEqual(thinking["type"] as? String, "enabled")
        XCTAssertEqual(thinking["budget_tokens"] as? Int, 256)
        XCTAssertEqual(thinking["display"] as? String, "omitted")
    }

    func testAnthropicGenerationSettingsDoNotSendEnabledThinkingWithoutBudget() throws {
        let client = RemoteLLMRuntimeClient()
        var payload: [String: Any] = [
            "model": "claude-sonnet-4-6",
            "messages": []
        ]

        client.applyAnthropicGenerationSettings(
            to: &payload,
            settings: LLMGenerationSettings(
                thinking: LLMThinkingSettings(
                    mode: .on,
                    effort: nil,
                    budgetTokens: nil,
                    exposeReasoning: false
                )
            ),
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9)
        )

        XCTAssertNil(payload["thinking"])
    }

    func testGoogleGenerationSettingsMapConfigAndDisableThinking() throws {
        let client = RemoteLLMRuntimeClient()
        var payload: [String: Any] = [
            "contents": []
        ]

        client.applyGoogleGenerationSettings(
            to: &payload,
            settings: LLMGenerationSettings(
                maxOutputTokens: 768,
                temperature: 0.1,
                topP: 0.7,
                topK: 20,
                stop: ["END"],
                responseFormat: .json,
                thinking: LLMThinkingSettings(
                    mode: .off,
                    effort: nil,
                    budgetTokens: nil,
                    exposeReasoning: false
                )
            ),
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9)
        )

        let generationConfig = try XCTUnwrap(payload["generationConfig"] as? [String: Any])
        XCTAssertEqual(generationConfig["maxOutputTokens"] as? Int, 768)
        XCTAssertEqual(generationConfig["temperature"] as? Double, 0.1)
        XCTAssertEqual(generationConfig["topP"] as? Double, 0.7)
        XCTAssertEqual(generationConfig["topK"] as? Int, 20)
        XCTAssertEqual(generationConfig["stopSequences"] as? [String], ["END"])
        XCTAssertEqual(generationConfig["responseMimeType"] as? String, "application/json")
        XCTAssertNil(payload["thinkingConfig"])
        let thinkingConfig = try XCTUnwrap(generationConfig["thinkingConfig"] as? [String: Any])
        XCTAssertEqual(thinkingConfig["thinkingBudget"] as? Int, 0)
    }

    func testOpenRouterGenerationSettingsExcludeReasoningAndMapOverrides() throws {
        let client = RemoteLLMRuntimeClient()
        var payload = client.openAICompatiblePayload(
            model: "openrouter/auto",
            systemPrompt: "",
            userPrompt: "hi",
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: false
        )

        try client.applyOpenAICompatibleGenerationSettings(
            to: &payload,
            provider: .openrouter,
            configuration: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.openrouter.rawValue,
                model: "openrouter/auto",
                generationSettings: LLMGenerationSettings(
                    maxOutputTokens: 333,
                    temperature: 0.4,
                    topP: 0.6,
                    seed: 42,
                    stop: ["STOP"],
                    presencePenalty: 0.2,
                    frequencyPenalty: 0.1,
                    logprobs: true,
                    topLogprobs: 5,
                    responseFormat: .json,
                    thinking: LLMThinkingSettings(
                        mode: .effort,
                        effort: "high",
                        budgetTokens: nil,
                        exposeReasoning: false
                    ),
                    extraBodyJSON: #"{"provider":{"order":["openai"]}}"#
                )
            ),
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            responseFormat: nil
        )

        XCTAssertEqual(payload["max_tokens"] as? Int, 333)
        XCTAssertEqual(payload["temperature"] as? Double, 0.4)
        XCTAssertEqual(payload["top_p"] as? Double, 0.6)
        XCTAssertEqual(payload["seed"] as? Int, 42)
        XCTAssertEqual(payload["stop"] as? [String], ["STOP"])
        XCTAssertEqual(payload["presence_penalty"] as? Double, 0.2)
        XCTAssertEqual(payload["frequency_penalty"] as? Double, 0.1)
        XCTAssertEqual(payload["logprobs"] as? Bool, true)
        XCTAssertEqual(payload["top_logprobs"] as? Int, 5)
        let reasoning = try XCTUnwrap(payload["reasoning"] as? [String: Any])
        XCTAssertEqual(reasoning["exclude"] as? Bool, true)
        XCTAssertEqual(reasoning["effort"] as? String, "high")
        let responseFormat = try XCTUnwrap(payload["response_format"] as? [String: Any])
        XCTAssertEqual(responseFormat["type"] as? String, "json_object")
        let provider = try XCTUnwrap(payload["provider"] as? [String: Any])
        XCTAssertEqual(provider["order"] as? [String], ["openai"])
    }

    func testDeepSeekDefaultTextGenerationDisablesThinkingForCurrentAndCustomModels() throws {
        let client = RemoteLLMRuntimeClient()
        let tuning = RemoteLLMRuntimeClient.GenerationTuning(maxTokens: 256, temperature: 0.2, topP: 0.9)
        for model in ["deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "custom-deepseek-model"] {
            for streaming in [false, true] {
                var payload = client.openAICompatiblePayload(
                    model: model,
                    systemPrompt: "Polish the transcription.",
                    userPrompt: "hello world",
                    tuning: tuning,
                    streamingEnabled: streaming
                )
                try client.applyOpenAICompatibleGenerationSettings(
                    to: &payload,
                    provider: .deepseek,
                    configuration: TestFactories.makeRemoteConfiguration(
                        providerID: RemoteLLMProvider.deepseek.rawValue,
                        model: model
                    ),
                    tuning: tuning,
                    responseFormat: nil
                )
                XCTAssertEqual(payload["model"] as? String, model)
                XCTAssertEqual(payload["stream"] as? Bool, streaming)
                XCTAssertEqual(payload["max_tokens"] as? Int, 256)
                XCTAssertEqual((payload["thinking"] as? [String: String])?["type"], "disabled")
            }
        }
    }

    func testDeepSeekPreservesExplicitThinkingAndLegacyReasonerWithoutUnsupportedBudget() throws {
        let client = RemoteLLMRuntimeClient()
        for mode in [LLMThinkingMode.on, .off, .budget, .effort, .providerDefault] {
            var payload: [String: Any] = [:]
            try client.applyOpenAICompatibleGenerationSettings(
                to: &payload,
                provider: .deepseek,
                configuration: TestFactories.makeRemoteConfiguration(
                    providerID: RemoteLLMProvider.deepseek.rawValue,
                    model: "deepseek-reasoner",
                    generationSettings: LLMGenerationSettings(
                        maxOutputTokens: 4096,
                        presencePenalty: 0.2,
                        frequencyPenalty: 0.2,
                        thinking: LLMThinkingSettings(
                            mode: mode,
                            effort: "max",
                            budgetTokens: 1024,
                            exposeReasoning: false
                        )
                    )
                ),
                tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
                responseFormat: nil
            )
            let thinking = payload["thinking"] as? [String: Any]
            switch mode {
            case .on, .budget:
                XCTAssertEqual(thinking?["type"] as? String, "enabled")
            case .off:
                XCTAssertEqual(thinking?["type"] as? String, "disabled")
            case .effort:
                XCTAssertEqual(payload["reasoning_effort"] as? String, "max")
            case .providerDefault:
                XCTAssertNil(thinking)
                XCTAssertNil(payload["reasoning_effort"])
            }
            XCTAssertNil(thinking?["budget_tokens"])
            XCTAssertNil(payload["presence_penalty"])
            XCTAssertNil(payload["frequency_penalty"])
            XCTAssertEqual(payload["max_tokens"] as? Int, 4096)
        }
    }

    func testXiaomiMiMoGenerationSettingsMapDocumentedChatCompletionFields() throws {
        let client = RemoteLLMRuntimeClient()
        var payload = client.openAICompatiblePayload(
            model: "mimo-v2.5-pro",
            systemPrompt: "",
            userPrompt: "hi",
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: false
        )

        try client.applyOpenAICompatibleGenerationSettings(
            to: &payload,
            provider: .xiaomiMiMo,
            configuration: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.xiaomiMiMo.rawValue,
                model: "mimo-v2.5-pro",
                generationSettings: LLMGenerationSettings(
                    maxOutputTokens: 333,
                    temperature: 1.0,
                    topP: 0.95,
                    thinking: .off
                )
            ),
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            responseFormat: nil
        )

        XCTAssertNil(payload["max_tokens"])
        XCTAssertEqual(payload["max_completion_tokens"] as? Int, 333)
        XCTAssertEqual(payload["temperature"] as? Double, 1.0)
        XCTAssertEqual(payload["top_p"] as? Double, 0.95)

        let thinking = try XCTUnwrap(payload["thinking"] as? [String: Any])
        XCTAssertEqual(thinking["type"] as? String, "disabled")
    }

    func testXiaomiMiMoGenerationSettingsDoNotSendUnsupportedThinkingTuning() throws {
        let client = RemoteLLMRuntimeClient()
        var payload = client.openAICompatiblePayload(
            model: "mimo-v2.5-pro",
            systemPrompt: "",
            userPrompt: "hi",
            tuning: .init(maxTokens: 256, temperature: 1.0, topP: 0.95),
            streamingEnabled: false
        )

        try client.applyOpenAICompatibleGenerationSettings(
            to: &payload,
            provider: .xiaomiMiMo,
            configuration: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.xiaomiMiMo.rawValue,
                model: "mimo-v2.5-pro",
                generationSettings: LLMGenerationSettings(
                    thinking: LLMThinkingSettings(
                        mode: .effort,
                        effort: "high",
                        budgetTokens: nil,
                        exposeReasoning: false
                    )
                )
            ),
            tuning: .init(maxTokens: 256, temperature: 1.0, topP: 0.95),
            responseFormat: nil
        )

        XCTAssertNil(payload["thinking"])
        XCTAssertNil(payload["reasoning_effort"])
    }

    func testStepFunGenerationSettingsMapDocumentedChatCompletionFields() throws {
        let client = RemoteLLMRuntimeClient()
        var payload = client.openAICompatiblePayload(
            model: "step-3.5-flash-2603",
            systemPrompt: "",
            userPrompt: "hi",
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: false
        )

        try client.applyOpenAICompatibleGenerationSettings(
            to: &payload,
            provider: .stepFun,
            configuration: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.stepFun.rawValue,
                model: "step-3.5-flash-2603",
                generationSettings: LLMGenerationSettings(
                    maxOutputTokens: 333,
                    temperature: 0.4,
                    topP: 0.6,
                    stop: ["STOP"],
                    presencePenalty: 0.2,
                    frequencyPenalty: 0.1,
                    responseFormat: .json,
                    thinking: LLMThinkingSettings(
                        mode: .effort,
                        effort: "high",
                        budgetTokens: nil,
                        exposeReasoning: false
                    )
                )
            ),
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            responseFormat: nil
        )

        XCTAssertEqual(payload["max_tokens"] as? Int, 333)
        XCTAssertEqual(payload["temperature"] as? Double, 0.4)
        XCTAssertEqual(payload["top_p"] as? Double, 0.6)
        XCTAssertEqual(payload["stop"] as? [String], ["STOP"])
        XCTAssertNil(payload["presence_penalty"])
        XCTAssertEqual(payload["frequency_penalty"] as? Double, 0.1)
        XCTAssertEqual(payload["reasoning_effort"] as? String, "high")
        let responseFormat = try XCTUnwrap(payload["response_format"] as? [String: Any])
        XCTAssertEqual(responseFormat["type"] as? String, "json_object")
    }

    func testStepFunReasoningEffortOnlySentForSupportedModel() throws {
        let client = RemoteLLMRuntimeClient()
        var payload = client.openAICompatiblePayload(
            model: "step-3.5-flash",
            systemPrompt: "",
            userPrompt: "hi",
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: false
        )

        try client.applyOpenAICompatibleGenerationSettings(
            to: &payload,
            provider: .stepFun,
            configuration: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.stepFun.rawValue,
                model: "step-3.5-flash",
                generationSettings: LLMGenerationSettings(
                    thinking: LLMThinkingSettings(
                        mode: .effort,
                        effort: "high",
                        budgetTokens: nil,
                        exposeReasoning: false
                    )
                )
            ),
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            responseFormat: nil
        )

        XCTAssertNil(payload["max_tokens"])
        XCTAssertNil(payload["reasoning_effort"])
    }

    func testStepFunDefaultThinkingDoesNotSendReasoningEffort() throws {
        let client = RemoteLLMRuntimeClient()
        var payload = client.openAICompatiblePayload(
            model: "step-3.5-flash-2603",
            systemPrompt: "",
            userPrompt: "hi",
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: false
        )

        try client.applyOpenAICompatibleGenerationSettings(
            to: &payload,
            provider: .stepFun,
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.stepFun.rawValue,
                model: "step-3.5-flash-2603",
                endpoint: "",
                apiKey: ""
            ),
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            responseFormat: nil
        )

        XCTAssertNil(payload["max_tokens"])
        XCTAssertNil(payload["reasoning_effort"])
    }

    func testStepFunTextModelKeepsDefaultMaxTokens() throws {
        let client = RemoteLLMRuntimeClient()
        var payload = client.openAICompatiblePayload(
            model: "step-2-mini",
            systemPrompt: "",
            userPrompt: "hi",
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            streamingEnabled: false
        )

        try client.applyOpenAICompatibleGenerationSettings(
            to: &payload,
            provider: .stepFun,
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.stepFun.rawValue,
                model: "step-2-mini",
                endpoint: "",
                apiKey: ""
            ),
            tuning: .init(maxTokens: 256, temperature: 0.2, topP: 0.9),
            responseFormat: nil
        )

        XCTAssertEqual(payload["max_tokens"] as? Int, 256)
    }
}
