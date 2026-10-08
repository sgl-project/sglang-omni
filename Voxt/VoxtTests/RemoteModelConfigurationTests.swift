import XCTest
@testable import Voxt

final class RemoteModelConfigurationTests: RemoteModelConfigurationTestCase {
    func testOllamaConfigurationIsConfiguredWithoutAPIKeyWhenModelExists() {
        let configuration = TestFactories.makeRemoteConfiguration(
            providerID: RemoteLLMProvider.ollama.rawValue,
            model: "qwen3"
        )

        XCTAssertTrue(configuration.isConfigured)
    }

    func testOMLXConfigurationIsConfiguredWithoutAPIKeyWhenModelExists() {
        let configuration = TestFactories.makeRemoteConfiguration(
            providerID: RemoteLLMProvider.omlx.rawValue,
            model: "qwen3"
        )

        XCTAssertTrue(configuration.isConfigured)
    }

    func testOpenAIConfigurationStillRequiresCredential() {
        let configuration = TestFactories.makeRemoteConfiguration(
            providerID: RemoteLLMProvider.openAI.rawValue,
            model: "gpt-5.2"
        )

        XCTAssertFalse(configuration.isConfigured)
    }

    func testOpenAIModelCatalogUsesOfficialModelIDs() {
        XCTAssertEqual(RemoteLLMProvider.openAI.suggestedModel, "gpt-5.2")

        let ids = RemoteLLMProvider.openAI.modelOptions.map(\.id)
        XCTAssertTrue(ids.contains("gpt-5.2"))
        XCTAssertTrue(ids.contains("gpt-5.2-pro"))
        XCTAssertTrue(ids.contains("gpt-5.1"))
        XCTAssertFalse(ids.contains("gpt-5.5"))
        XCTAssertFalse(ids.contains("gpt-5.4"))
    }

    func testOpenAIReasoningEffortOptionsFollowModelSupport() {
        XCTAssertEqual(
            OpenAIReasoningEffort.supportedCases(forModel: "gpt-5.2"),
            [.automatic, .none, .low, .medium, .high, .xhigh]
        )
        XCTAssertEqual(
            OpenAIReasoningEffort.supportedCases(forModel: "gpt-5.2-pro"),
            [.automatic, .medium, .high, .xhigh]
        )
        XCTAssertEqual(
            OpenAIReasoningEffort.supportedCases(forModel: "gpt-5.1"),
            [.automatic, .none, .low, .medium, .high]
        )
        XCTAssertEqual(
            OpenAIReasoningEffort.supportedCases(forModel: "gpt-5"),
            [.automatic, .minimal, .low, .medium, .high]
        )
    }

    func testLoadSaveRoundTripPreservesOllamaConfigurationFields() {
        let stored: [String: RemoteProviderConfiguration] = [
            RemoteLLMProvider.ollama.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.ollama.rawValue,
                model: "qwen3",
                endpoint: "http://127.0.0.1:11434/api/chat",
                ollamaResponseFormat: OllamaResponseFormat.jsonSchema.rawValue,
                ollamaJSONSchema: #"{"type":"object","properties":{"answer":{"type":"string"}}}"#,
                ollamaThinkMode: OllamaThinkMode.low.rawValue,
                ollamaKeepAlive: "5m",
                ollamaLogprobsEnabled: true,
                ollamaTopLogprobs: 7,
                ollamaOptionsJSON: #"{"num_ctx":8192,"repeat_penalty":1.05}"#
            )
        ]

        let raw = RemoteModelConfigurationStore.saveConfigurations(stored)
        let roundTrip = RemoteModelConfigurationStore.loadConfigurations(from: raw)
        let restored = roundTrip[RemoteLLMProvider.ollama.rawValue]

        XCTAssertEqual(restored?.ollamaResponseFormat, OllamaResponseFormat.jsonSchema.rawValue)
        XCTAssertEqual(restored?.ollamaJSONSchema, #"{"type":"object","properties":{"answer":{"type":"string"}}}"#)
        XCTAssertEqual(restored?.ollamaThinkMode, OllamaThinkMode.low.rawValue)
        XCTAssertEqual(restored?.ollamaKeepAlive, "5m")
        XCTAssertEqual(restored?.ollamaLogprobsEnabled, true)
        XCTAssertEqual(restored?.ollamaTopLogprobs, 7)
        XCTAssertEqual(restored?.ollamaOptionsJSON, #"{"num_ctx":8192,"repeat_penalty":1.05}"#)
        XCTAssertEqual(restored?.generationSettings.responseFormat, .jsonSchema)
        XCTAssertEqual(restored?.generationSettings.thinking.mode, .effort)
        XCTAssertEqual(restored?.generationSettings.thinking.effort, OllamaThinkMode.low.rawValue)
        XCTAssertEqual(restored?.generationSettings.logprobs, true)
        XCTAssertEqual(restored?.generationSettings.topLogprobs, 7)
        XCTAssertEqual(restored?.generationSettings.extraOptionsJSON, #"{"num_ctx":8192,"repeat_penalty":1.05}"#)
    }

    func testLegacyOpenAIConfigurationMigratesToUnifiedGenerationSettings() {
        let configuration = TestFactories.makeRemoteConfiguration(
            providerID: RemoteLLMProvider.openAI.rawValue,
            model: "gpt-5.2",
            openAIReasoningEffort: OpenAIReasoningEffort.high.rawValue,
            openAIMaxOutputTokens: 4096
        )

        XCTAssertEqual(configuration.generationSettings.maxOutputTokens, 4096)
        XCTAssertEqual(configuration.generationSettings.thinking.mode, .effort)
        XCTAssertEqual(configuration.generationSettings.thinking.effort, OpenAIReasoningEffort.high.rawValue)
    }

    func testResolvedASRConfigurationFallsBackToSuggestedModelAndClearsRealtimeFlag() {
        let stored: [String: RemoteProviderConfiguration] = [
            RemoteASRProvider.doubaoASR.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                model: "invalid-model",
                accessToken: "token",
                openAIChunkPseudoRealtimeEnabled: true
            )
        ]

        let resolved = RemoteModelConfigurationStore.resolvedASRConfiguration(
            provider: .doubaoASR,
            stored: stored
        )

        XCTAssertEqual(resolved.model, RemoteASRProvider.doubaoASR.suggestedModel)
        XCTAssertFalse(resolved.openAIChunkPseudoRealtimeEnabled)
    }

    func testResolvedLLMConfigurationDefaultsWhenMissing() {
        let resolved = RemoteModelConfigurationStore.resolvedLLMConfiguration(
            provider: .anthropic,
            stored: [:]
        )

        XCTAssertEqual(resolved.providerID, RemoteLLMProvider.anthropic.rawValue)
        XCTAssertEqual(resolved.model, RemoteLLMProvider.anthropic.suggestedModel)
        XCTAssertEqual(resolved.endpoint, "")
        XCTAssertFalse(resolved.searchEnabled)
    }

    func testResponsesProviderCapabilitiesAreConfiguredPerProvider() {
        XCTAssertTrue(RemoteLLMProvider.openAI.usesResponsesAPI)
        XCTAssertTrue(RemoteLLMProvider.codex.usesResponsesAPI)
        XCTAssertTrue(RemoteLLMProvider.aliyunBailian.usesResponsesAPI)
        XCTAssertTrue(RemoteLLMProvider.volcengine.usesResponsesAPI)
        XCTAssertTrue(RemoteLLMProvider.aliyunBailian.supportsHostedSearch)
        XCTAssertTrue(RemoteLLMProvider.volcengine.supportsHostedSearch)
        XCTAssertTrue(RemoteLLMProvider.aliyunBailian.defaultSearchEnabled)
        XCTAssertFalse(RemoteLLMProvider.volcengine.defaultSearchEnabled)
        XCTAssertTrue(RemoteLLMProvider.omlx.apiKeyIsOptional)
        XCTAssertTrue(RemoteLLMProvider.codex.apiKeyIsOptional)
        XCTAssertFalse(RemoteLLMProvider.omlx.usesResponsesAPI)
    }

    func testDeepSeekUsesCurrentSuggestedModelAndKeepsLegacyAliases() {
        XCTAssertEqual(RemoteLLMProvider.deepseek.suggestedModel, "deepseek-flash")

        let latestIDs = RemoteLLMProvider.deepseek.latestModelOptions.map(\.id)
        XCTAssertTrue(latestIDs.contains("deepseek-flash"))
        XCTAssertTrue(latestIDs.contains("deepseek-v4-pro"))

        let allIDs = RemoteLLMProvider.deepseek.modelOptions.map(\.id)
        XCTAssertTrue(allIDs.contains("deepseek-v4-flash"))
        XCTAssertTrue(allIDs.contains("deepseek-v4-flash-vision-exp"))
        XCTAssertTrue(allIDs.contains("deepseek-chat"))
        XCTAssertTrue(allIDs.contains("deepseek-reasoner"))
    }

    func testDeepSeekEffectiveSettingsPreserveExplicitThinkingChoices() {
        for mode in [LLMThinkingMode.on, .off, .effort, .budget] {
            let thinking = LLMThinkingSettings(
                mode: mode,
                effort: "high",
                budgetTokens: 1024,
                exposeReasoning: false
            )
            let configuration = TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.deepseek.rawValue,
                model: "deepseek-flash",
                generationSettings: LLMGenerationSettings(thinking: thinking)
            )
            XCTAssertEqual(configuration.effectiveGenerationSettings(provider: .deepseek).thinking, thinking)
        }
    }

    func testDeepSeekDefaultThinkingOverrideDoesNotMutateStoredSettingsOrOtherProviders() {
        let configuration = TestFactories.makeRemoteConfiguration(
            providerID: RemoteLLMProvider.deepseek.rawValue,
            model: "deepseek-flash"
        )
        XCTAssertEqual(configuration.effectiveGenerationSettings(provider: .deepseek).thinking.mode, .off)
        XCTAssertEqual(configuration.generationSettings.thinking.mode, .providerDefault)
        XCTAssertEqual(configuration.effectiveGenerationSettings(provider: .openrouter).thinking.mode, .providerDefault)
    }

    func testDeepSeekCapabilitiesMatchDocumentedThinkingControls() {
        let capabilities = LLMProviderCapabilityRegistry.capabilities(for: .deepseek)
        XCTAssertTrue(capabilities.supportsThinkingToggle)
        XCTAssertTrue(capabilities.supportsThinkingEffort)
        XCTAssertFalse(capabilities.supportsThinkingBudget)
        XCTAssertFalse(capabilities.supportsPenalties)
    }

    func testStepFunUsesChatCompletionModelCatalogAndCapabilities() {
        XCTAssertEqual(RemoteLLMProvider.stepFun.suggestedModel, "step-3.5-flash")
        XCTAssertFalse(RemoteLLMProvider.stepFun.usesResponsesAPI)
        XCTAssertFalse(RemoteLLMProvider.stepFun.supportsHostedSearch)

        let ids = RemoteLLMProvider.stepFun.modelOptions.map(\.id)
        XCTAssertTrue(ids.contains("step-3.5-flash"))
        XCTAssertTrue(ids.contains("step-3.5-flash-2603"))
        XCTAssertTrue(ids.contains("step-2-mini"))
        XCTAssertTrue(ids.contains("step-1-32k"))
        XCTAssertTrue(ids.contains("step-router-v1"))

        let capabilities = LLMProviderCapabilityRegistry.capabilities(for: .stepFun)
        XCTAssertTrue(capabilities.supportsThinkingEffort)
        XCTAssertFalse(capabilities.supportsThinkingBudget)
        XCTAssertTrue(capabilities.supportsResponseFormat)
    }

    func testXiaomiMiMoUsesChatCompletionModelCatalogAndCapabilities() {
        XCTAssertEqual(RemoteASRProvider.xiaomiMiMoASR.suggestedModel, "mimo-v2.5-asr")
        XCTAssertEqual(RemoteASRProvider.xiaomiMiMoASR.modelOptions.map(\.id), ["mimo-v2.5-asr"])

        XCTAssertEqual(RemoteLLMProvider.xiaomiMiMo.suggestedModel, "mimo-v2.5-pro")
        XCTAssertFalse(RemoteLLMProvider.xiaomiMiMo.usesResponsesAPI)
        XCTAssertFalse(RemoteLLMProvider.xiaomiMiMo.supportsHostedSearch)

        let ids = RemoteLLMProvider.xiaomiMiMo.modelOptions.map(\.id)
        XCTAssertTrue(ids.contains("mimo-v2.5-pro"))
        XCTAssertTrue(ids.contains("mimo-v2.5"))

        let capabilities = LLMProviderCapabilityRegistry.capabilities(for: .xiaomiMiMo)
        XCTAssertTrue(capabilities.supportsThinkingToggle)
        XCTAssertFalse(capabilities.supportsThinkingEffort)
        XCTAssertFalse(capabilities.supportsThinkingBudget)
        XCTAssertTrue(capabilities.supportsPenalties)
        XCTAssertFalse(capabilities.supportsLogprobs)
        XCTAssertTrue(capabilities.supportsResponseFormat)
    }

    func testStepFunLLMConfigurationDefaultsThinkingToOff() {
        let resolved = RemoteModelConfigurationStore.resolvedLLMConfiguration(
            provider: .stepFun,
            stored: [:]
        )

        XCTAssertEqual(resolved.generationSettings.thinking.mode, .off)
        XCTAssertEqual(resolved.effectiveGenerationSettings(provider: .stepFun).thinking.mode, .off)
    }

    func testStepFunLegacyConfigurationDoesNotInheritOpenAIReasoningEffort() {
        let configuration = TestFactories.makeRemoteConfiguration(
            providerID: RemoteLLMProvider.stepFun.rawValue,
            model: "step-3.5-flash-2603",
            openAIReasoningEffort: OpenAIReasoningEffort.high.rawValue
        )

        XCTAssertEqual(configuration.generationSettings.thinking.mode, .off)
        XCTAssertNil(configuration.generationSettings.thinking.effort)
    }

    func testResolvedAliyunLLMConfigurationDefaultsSearchToEnabled() {
        let resolved = RemoteModelConfigurationStore.resolvedLLMConfiguration(
            provider: .aliyunBailian,
            stored: [:]
        )

        XCTAssertTrue(resolved.searchEnabled)
    }

    func testResolvedVolcengineLLMConfigurationDefaultsSearchToDisabled() {
        let resolved = RemoteModelConfigurationStore.resolvedLLMConfiguration(
            provider: .volcengine,
            stored: [:]
        )

        XCTAssertFalse(resolved.searchEnabled)
    }

    func testDecodeLegacyAliyunLLMConfigurationDefaultsSearchToEnabled() throws {
        let legacyJSON = """
        [
          {
            "providerID": "aliyunBailian",
            "model": "qwen-plus-latest",
            "endpoint": "",
            "apiKey": "",
            "appID": "",
            "accessToken": ""
          }
        ]
        """

        let loaded = RemoteModelConfigurationStore.loadConfigurations(from: legacyJSON)

        XCTAssertEqual(loaded["aliyunBailian"]?.searchEnabled, true)
    }

    func testDecodeLegacyVolcengineLLMConfigurationDefaultsSearchToDisabled() throws {
        let legacyJSON = """
        [
          {
            "providerID": "volcengine",
            "model": "doubao-1-5-pro",
            "endpoint": "",
            "apiKey": "",
            "appID": "",
            "accessToken": ""
          }
        ]
        """

        let loaded = RemoteModelConfigurationStore.loadConfigurations(from: legacyJSON)

        XCTAssertEqual(loaded["volcengine"]?.searchEnabled, false)
    }
}
