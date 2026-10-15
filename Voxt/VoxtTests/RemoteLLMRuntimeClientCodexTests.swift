import XCTest
@testable import Voxt

final class RemoteLLMRuntimeClientCodexTests: XCTestCase {
    func testMakeResponsesRequestAppliesCodexOAuthHeaders() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .codex,
            endpointValue: "https://chatgpt.com/backend-api/codex/responses",
            model: "gpt-5.3-codex-spark",
            systemPrompt: "",
            inputPayload: "ping",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.codex.rawValue,
                model: "gpt-5.3-codex-spark",
                endpoint: "",
                apiKey: ""
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9),
            textFormat: nil,
            streamingEnabled: false,
            additionalHeaders: [
                "Authorization": "Bearer codex-token",
                "originator": "codex_cli_rs",
                "ChatGPT-Account-ID": "acct_123"
            ]
        )

        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])

        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer codex-token")
        XCTAssertEqual(request.value(forHTTPHeaderField: "originator"), "codex_cli_rs")
        XCTAssertEqual(request.value(forHTTPHeaderField: "ChatGPT-Account-ID"), "acct_123")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "text/event-stream, application/json")
        XCTAssertEqual(object["stream"] as? Bool, true)
        XCTAssertEqual(object["store"] as? Bool, false)
        XCTAssertEqual(object["instructions"] as? String, "Process the input and return only the final answer.")
        XCTAssertNil(object["max_output_tokens"])
        XCTAssertNil(object["temperature"])
        XCTAssertNil(object["top_p"])
        let input = try XCTUnwrap(object["input"] as? [[String: Any]])
        XCTAssertEqual(input.first?["role"] as? String, "user")
        XCTAssertEqual(input.first?["content"] as? String, "ping")
    }

    func testMakeResponsesRequestAppliesCodexFastModeServiceTier() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .codex,
            endpointValue: "https://chatgpt.com/backend-api/codex/responses",
            model: "gpt-5.4",
            systemPrompt: "",
            inputPayload: "ping",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.codex.rawValue,
                model: "gpt-5.4",
                endpoint: "",
                apiKey: "",
                codexFastModeEnabled: true
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9),
            textFormat: nil,
            streamingEnabled: false
        )

        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])

        XCTAssertEqual(object["service_tier"] as? String, "priority")
    }

    func testMakeResponsesRequestOmitsUnsupportedCodexOptions() throws {
        let client = RemoteLLMRuntimeClient()
        let request = try client.makeResponsesRequest(
            provider: .codex,
            endpointValue: "https://chatgpt.com/backend-api/codex/responses",
            model: "gpt-5.4",
            systemPrompt: "",
            inputPayload: "ping",
            configuration: RemoteProviderConfiguration(
                providerID: RemoteLLMProvider.codex.rawValue,
                model: "gpt-5.4",
                endpoint: "",
                apiKey: "",
                openAIReasoningEffort: OpenAIReasoningEffort.xhigh.rawValue,
                openAITextVerbosity: OpenAITextVerbosity.high.rawValue,
                openAIMaxOutputTokens: 4096,
                generationSettings: LLMGenerationSettings(
                    maxOutputTokens: 4096,
                    temperature: 0.7,
                    topP: 0.8,
                    logprobs: true,
                    responseFormat: .json,
                    extraBodyJSON: #"{"service_tier":"flex","temperature":0.1}"#
                )
            ),
            previousResponseID: nil,
            tuning: .init(maxTokens: 512, temperature: 0.2, topP: 0.9),
            textFormat: nil,
            streamingEnabled: false
        )

        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])

        XCTAssertEqual(object["instructions"] as? String, "Process the input and return only the final answer.")
        XCTAssertNil(object["reasoning"])
        XCTAssertNil(object["text"])
        XCTAssertNil(object["max_output_tokens"])
        XCTAssertNil(object["temperature"])
        XCTAssertNil(object["top_p"])
        XCTAssertNil(object["logprobs"])
        XCTAssertNil(object["service_tier"])
    }

    func testCodexModelCatalogParserFlattensBackendResponses() throws {
        let object: [String: Any] = [
            "chat_models": [
                "models": [
                    [
                        "slug": "gpt-5.4",
                        "display_name": "GPT-5.4",
                        "output_modalities": ["text"]
                    ],
                    [
                        "slug": "gpt-image-2",
                        "display_name": "GPT Image 2",
                        "output_modalities": ["image"]
                    ]
                ]
            ],
            "categories": [
                [
                    "models": [
                        [
                            "id": "gpt-5-codex-mini",
                            "name": "GPT-5 Codex Mini",
                            "output_modalities": ["text"]
                        ],
                        [
                            "id": "gpt-5.4",
                            "display_name": "Duplicate",
                            "output_modalities": ["text"]
                        ]
                    ]
                ]
            ]
        ]

        let options = RemoteLLMRuntimeClient.codexModelOptions(from: object)

        XCTAssertEqual(options.map(\.id), ["gpt-5.4", "gpt-5-codex-mini"])
        XCTAssertEqual(options.first?.title, "GPT-5.4")
    }
}
