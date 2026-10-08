import XCTest
@testable import Voxt

final class RemoteLLMRuntimeClientEndpointsTests: XCTestCase {
    func testResolvedLLMEndpointNormalizesAliyunResponsesEndpoints() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .aliyunBailian,
                endpoint: "",
                model: "qwen-plus"
            ),
            "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .aliyunBailian,
                endpoint: "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
                model: "qwen-plus"
            ),
            "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .aliyunBailian,
                endpoint: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                model: "qwen-plus"
            ),
            "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"
        )
    }

    func testResolvedLLMEndpointNormalizesVolcengineResponsesEndpoints() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .volcengine,
                endpoint: "",
                model: "doubao-1-5-pro"
            ),
            "https://ark.cn-beijing.volces.com/api/v3/responses"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .volcengine,
                endpoint: "https://ark.cn-beijing.volces.com/api/v3/models",
                model: "doubao-1-5-pro"
            ),
            "https://ark.cn-beijing.volces.com/api/v3/responses"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .volcengine,
                endpoint: "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                model: "doubao-1-5-pro"
            ),
            "https://ark.cn-beijing.volces.com/api/v3/responses"
        )
    }

    func testStreamingEndpointValueBuildsGoogleStreamEndpoint() {
        let client = RemoteLLMRuntimeClient()

        let endpoint = client.streamingEndpointValue(
            provider: .google,
            endpoint: "https://generativelanguage.googleapis.com/v1beta/models",
            model: "gemini-2.5-pro",
            streamingEnabled: true
        )

        XCTAssertEqual(
            endpoint,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent"
        )
    }

    func testStreamingEndpointValueBuildsOpenAIResponsesEndpoint() {
        let client = RemoteLLMRuntimeClient()

        let endpoint = client.streamingEndpointValue(
            provider: .openAI,
            endpoint: "https://api.openai.com/v1/chat/completions",
            model: "gpt-5.2",
            streamingEnabled: true
        )

        XCTAssertEqual(endpoint, "https://api.openai.com/v1/responses")
    }

    func testOpenAIResolvedEndpointDefaultsToResponsesAPI() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .openAI,
                endpoint: "",
                model: "gpt-5.2"
            ),
            "https://api.openai.com/v1/responses"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .openAI,
                endpoint: "https://api.openai.com",
                model: "gpt-5.2"
            ),
            "https://api.openai.com/v1/responses"
        )
    }

    func testResolvedLLMEndpointBuildsDeepSeekChatCompletionsFromOfficialBaseURL() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .deepseek,
                endpoint: "",
                model: "deepseek-v4-flash"
            ),
            "https://api.deepseek.com/v1/chat/completions"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .deepseek,
                endpoint: "https://api.deepseek.com",
                model: "deepseek-v4-flash"
            ),
            "https://api.deepseek.com/v1/chat/completions"
        )
    }

    func testResolvedLLMEndpointBuildsStepFunChatCompletionsFromOfficialBaseURL() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertEqual(
            client.providerDefaultEndpoint(.stepFun),
            "https://api.stepfun.com/v1/chat/completions"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .stepFun,
                endpoint: "",
                model: "step-3.5-flash"
            ),
            "https://api.stepfun.com/v1/chat/completions"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .stepFun,
                endpoint: "https://api.stepfun.com",
                model: "step-3.5-flash"
            ),
            "https://api.stepfun.com/v1/chat/completions"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .stepFun,
                endpoint: "https://api.stepfun.com/v1/models",
                model: "step-3.5-flash"
            ),
            "https://api.stepfun.com/v1/chat/completions"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .stepFun,
                endpoint: "",
                model: "step-router-v1"
            ),
            "https://api.stepfun.com/step_plan/v1/chat/completions"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .stepFun,
                endpoint: "https://api.stepfun.com/step_plan/v1/chat/completions",
                model: "step-router-v1"
            ),
            "https://api.stepfun.com/step_plan/v1/chat/completions"
        )
    }

    func testResolvedLLMEndpointBuildsXiaomiMiMoChatCompletionsFromOfficialBaseURL() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertEqual(
            client.providerDefaultEndpoint(.xiaomiMiMo),
            "https://api.xiaomimimo.com/v1/chat/completions"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .xiaomiMiMo,
                endpoint: "",
                model: "mimo-v2.5-pro"
            ),
            "https://api.xiaomimimo.com/v1/chat/completions"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .xiaomiMiMo,
                endpoint: "https://api.xiaomimimo.com",
                model: "mimo-v2.5-pro"
            ),
            "https://api.xiaomimimo.com/v1/chat/completions"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .xiaomiMiMo,
                endpoint: "https://api.xiaomimimo.com/v1/models",
                model: "mimo-v2.5-pro"
            ),
            "https://api.xiaomimimo.com/v1/chat/completions"
        )
    }

    func testResolvedLLMEndpointDefaultsOllamaToBaseEndpoint() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertEqual(
            client.providerDefaultEndpoint(.ollama),
            "http://localhost:11434"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .ollama,
                endpoint: "",
                model: "qwen3"
            ),
            "http://localhost:11434"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .ollama,
                endpoint: "http://localhost:11434/api",
                model: "qwen3"
            ),
            "http://localhost:11434"
        )
    }

    func testResolvedLLMEndpointBuildsOMLXChatCompletionsFromBaseURL() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertEqual(
            client.providerDefaultEndpoint(.omlx),
            "http://localhost:8000/v1"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .omlx,
                endpoint: "",
                model: "qwen3"
            ),
            "http://localhost:8000/v1/chat/completions"
        )
        XCTAssertEqual(
            client.resolvedLLMEndpoint(
                provider: .omlx,
                endpoint: "http://localhost:8000/v1/models",
                model: "qwen3"
            ),
            "http://localhost:8000/v1/chat/completions"
        )
    }

    func testResolvedOllamaRequestEndpointSelectsNativeRouteFromBaseEndpoint() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertEqual(
            client.resolvedOllamaRequestEndpoint(
                endpoint: "http://localhost:11434",
                useGenerate: true
            ),
            "http://localhost:11434/api/generate"
        )
        XCTAssertEqual(
            client.resolvedOllamaRequestEndpoint(
                endpoint: "http://localhost:11434",
                useGenerate: false
            ),
            "http://localhost:11434/api/chat"
        )
    }

    func testResolvedOllamaRequestEndpointPreservesExplicitNativeAndCompatibleEndpoints() {
        let client = RemoteLLMRuntimeClient()

        XCTAssertEqual(
            client.resolvedOllamaRequestEndpoint(
                endpoint: "http://localhost:11434/api/chat",
                useGenerate: true
            ),
            "http://localhost:11434/api/chat"
        )
        XCTAssertEqual(
            client.resolvedOllamaRequestEndpoint(
                endpoint: "http://localhost:11434/v1/chat/completions",
                useGenerate: true
            ),
            "http://localhost:11434/v1/chat/completions"
        )
    }
}
