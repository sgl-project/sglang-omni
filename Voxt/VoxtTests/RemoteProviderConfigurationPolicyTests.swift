// RemoteProviderConfigurationPolicyTests.swift
// Provides Remote Provider Configuration Policy Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class RemoteProviderConfigurationPolicyTests: XCTestCase {
    func testOpenAIASRSupportsCustomModelSelection() {
        XCTAssertTrue(
            RemoteProviderConfigurationPolicy.supportsCustomModelSelection(
                target: .asr(.openAIWhisper)
            )
        )
        XCTAssertFalse(
            RemoteProviderConfigurationPolicy.supportsCustomModelSelection(
                target: .asr(.glmASR)
            )
        )
    }

    func testOpenAIASRPickerIncludesCustomOption() {
        XCTAssertEqual(
            RemoteProviderConfigurationPolicy.pickerModelOptionIDs(
                target: .asr(.openAIWhisper),
                configuredModel: "whisper-1"
            ),
            [
                "gpt-4o-mini-transcribe",
                "gpt-4o-transcribe",
                "gpt-4o-transcribe-diarize",
                "whisper-1",
                RemoteProviderConfigurationPolicy.customModelOptionID
            ]
        )
    }

    func testResolvedSelectionPrefersKnownSelectionThenConfiguredValue() {
        let target = RemoteProviderTestTarget.llm(.openAI)

        XCTAssertEqual(
            RemoteProviderConfigurationPolicy.resolvedSelection(
                target: target,
                selectedProviderModel: "gpt-5.2",
                configuredModel: "custom-model"
            ),
            "gpt-5.2"
        )

        XCTAssertEqual(
            RemoteProviderConfigurationPolicy.resolvedSelection(
                target: target,
                selectedProviderModel: "unknown",
                configuredModel: "gpt-5.2"
            ),
            "gpt-5.2"
        )
    }

    func testResolvedSelectionUsesCustomForUnknownOpenAIASRConfiguredModel() {
        let resolved = RemoteProviderConfigurationPolicy.resolvedSelection(
            target: .asr(.openAIWhisper),
            selectedProviderModel: "",
            configuredModel: "gpt-4o-transcribe-preview"
        )

        XCTAssertEqual(resolved, RemoteProviderConfigurationPolicy.customModelOptionID)
    }

    func testResolvedSelectionKeepsFirstBuiltinForUnknownNonCustomASRModel() {
        let resolved = RemoteProviderConfigurationPolicy.resolvedSelection(
            target: .asr(.glmASR),
            selectedProviderModel: "",
            configuredModel: "glm-asr-custom"
        )

        XCTAssertEqual(resolved, "glm-asr-2512")
    }

    func testInitialSelectionFallsBackToCustomForUnknownLLMModel() {
        let selection = RemoteProviderConfigurationPolicy.initialSelection(
            target: .llm(.openAI),
            configuredModel: "my-custom-model"
        )

        XCTAssertEqual(selection, RemoteProviderConfigurationPolicy.customModelOptionID)
    }

    func testInitialSelectionFallsBackToCustomForUnknownOpenAIASRModel() {
        let selection = RemoteProviderConfigurationPolicy.initialSelection(
            target: .asr(.openAIWhisper),
            configuredModel: "gpt-4o-transcribe-preview"
        )

        XCTAssertEqual(selection, RemoteProviderConfigurationPolicy.customModelOptionID)
    }

    func testResolvedModelValueUsesSuggestedModelWhenCustomValueEmpty() {
        let resolved = RemoteProviderConfigurationPolicy.resolvedModelValue(
            target: .llm(.anthropic),
            resolvedSelection: RemoteProviderConfigurationPolicy.customModelOptionID,
            customModelID: "   "
        )

        XCTAssertEqual(resolved, RemoteLLMProvider.anthropic.suggestedModel)
    }

    func testResolvedModelValueUsesSuggestedModelWhenOpenAIASRCustomValueEmpty() {
        let resolved = RemoteProviderConfigurationPolicy.resolvedModelValue(
            target: .asr(.openAIWhisper),
            resolvedSelection: RemoteProviderConfigurationPolicy.customModelOptionID,
            customModelID: "   "
        )

        XCTAssertEqual(resolved, RemoteASRProvider.openAIWhisper.suggestedModel)
    }

    func testResolvedASRConfigurationPreservesCustomOpenAIModel() {
        let stored = [
            RemoteASRProvider.openAIWhisper.rawValue: RemoteProviderConfiguration(
                providerID: RemoteASRProvider.openAIWhisper.rawValue,
                model: "whisper-large-v3-turbo",
                endpoint: "https://api.groq.com/openai/v1/audio/transcriptions",
                apiKey: ""
            )
        ]

        let resolved = RemoteModelConfigurationStore.resolvedASRConfiguration(
            provider: .openAIWhisper,
            stored: stored
        )

        XCTAssertEqual(resolved.model, "whisper-large-v3-turbo")
    }

    func testResolvedASRConfigurationStillNormalizesUnknownNonOpenAIModel() {
        let stored = [
            RemoteASRProvider.glmASR.rawValue: RemoteProviderConfiguration(
                providerID: RemoteASRProvider.glmASR.rawValue,
                model: "glm-asr-custom",
                endpoint: "",
                apiKey: ""
            )
        ]

        let resolved = RemoteModelConfigurationStore.resolvedASRConfiguration(
            provider: .glmASR,
            stored: stored
        )

        XCTAssertEqual(resolved.model, RemoteASRProvider.glmASR.suggestedModel)
    }

    func testOpenAIASRSheetShowsCustomFieldForCustomSelection() {
        let sheet = makeSheet(
            target: .asr(.openAIWhisper),
            model: "gpt-4o-transcribe-preview"
        )

        XCTAssertTrue(sheet.supportsCustomProviderModelSelection)
        XCTAssertTrue(sheet.shouldShowCustomProviderModelField)
        XCTAssertEqual(
            sheet.customProviderModelPlaceholder,
            AppLocalization.localizedString("e.g. gpt-4o-transcribe-xxx")
        )
    }

    func testNonOpenAIASRSheetDoesNotShowCustomField() {
        let sheet = makeSheet(
            target: .asr(.glmASR),
            model: "glm-asr-1"
        )

        XCTAssertFalse(sheet.supportsCustomProviderModelSelection)
        XCTAssertFalse(sheet.shouldShowCustomProviderModelField)
    }

    func testOllamaSheetUsesOptionalAPIKeyAndShowsOllamaSection() {
        let sheet = makeSheet(
            target: .llm(.ollama),
            model: "qwen3"
        )

        XCTAssertTrue(sheet.isOllamaLLMProvider)
        XCTAssertEqual(sheet.apiKeyFieldTitle, AppLocalization.localizedString("API Key (Optional)"))
        XCTAssertEqual(sheet.apiKeyFieldPlaceholder, AppLocalization.localizedString("Paste API key (optional)"))
    }

    func testOMLXSheetUsesOptionalAPIKey() {
        let sheet = makeSheet(
            target: .llm(.omlx),
            model: "qwen3"
        )

        XCTAssertFalse(sheet.isOllamaLLMProvider)
        XCTAssertEqual(sheet.apiKeyFieldTitle, AppLocalization.localizedString("API Key (Optional)"))
        XCTAssertEqual(sheet.apiKeyFieldPlaceholder, AppLocalization.localizedString("Paste API key (optional)"))
    }

    func testOllamaSheetShowsJSONSchemaFieldOnlyForJSONSchemaFormat() {
        let sheet = makeSheet(
            target: .llm(.ollama),
            model: "qwen3"
        )

        XCTAssertFalse(sheet.shouldShowOllamaJSONSchemaField(for: OllamaResponseFormat.plain.rawValue))
        XCTAssertTrue(sheet.shouldShowOllamaJSONSchemaField(for: OllamaResponseFormat.jsonSchema.rawValue))
    }

    func testOllamaSheetValidationRejectsInvalidJSONFields() {
        let sheet = makeSheet(
            target: .llm(.ollama),
            model: "qwen3"
        )

        XCTAssertEqual(
            sheet.validationMessageForOllamaSettings(
                responseFormat: OllamaResponseFormat.jsonSchema.rawValue,
                jsonSchema: #"["not-an-object"]"#,
                optionsJSON: #"{"num_ctx":4096}"#,
                logprobsEnabled: false,
                topLogprobsText: ""
            ),
            AppLocalization.format(
                "%@ must be a JSON object.",
                AppLocalization.localizedString("JSON Schema")
            )
        )
    }

    func testOllamaSheetValidationRejectsNegativeTopLogprobs() {
        let sheet = makeSheet(
            target: .llm(.ollama),
            model: "qwen3"
        )

        XCTAssertEqual(
            sheet.validationMessageForOllamaSettings(
                responseFormat: OllamaResponseFormat.plain.rawValue,
                jsonSchema: "",
                optionsJSON: "",
                logprobsEnabled: true,
                topLogprobsText: "-1"
            ),
            AppLocalization.localizedString("Top Logprobs must be a non-negative integer.")
        )
    }

    func testOpenAISheetValidationRejectsInvalidMaxOutputTokens() {
        let invalid = makeSheet(
            target: .llm(.openAI), model: "gpt-5.2", generationMaxOutputTokensText: "0"
        )
        let valid = makeSheet(
            target: .llm(.openAI), model: "gpt-5.2", generationMaxOutputTokensText: "4096"
        )

        XCTAssertEqual(
            invalid.validationMessageForGenerationSettings(),
            AppLocalization.format("%@ must be a positive integer.", AppLocalization.localizedString("Max Output Tokens"))
        )
        XCTAssertNil(valid.validationMessageForGenerationSettings())
    }

    func testGenerationValidationAllowsBlankOrTrimmedPositiveTokenLimit() {
        for text in ["", "  ", " 4096 "] {
            let sheet = makeSheet(
                target: .llm(.openAI), model: "gpt-5.2", generationMaxOutputTokensText: text
            )
            XCTAssertNil(sheet.validationMessageForGenerationSettings(), text)
        }
    }

    func testGenerationValidationRejectsNegativeFractionalAndOverflowingTokenLimits() {
        for text in ["-1", "1.5", "abc", "999999999999999999999999999999"] {
            let sheet = makeSheet(
                target: .llm(.openAI), model: "gpt-5.2", generationMaxOutputTokensText: text
            )
            XCTAssertEqual(
                sheet.validationMessageForGenerationSettings(),
                AppLocalization.format("%@ must be a positive integer.", AppLocalization.localizedString("Max Output Tokens")),
                text
            )
        }
    }

    func testASRSheetDoesNotValidateInactiveGenerationFields() {
        let sheet = makeSheet(
            target: .asr(.openAIWhisper), model: "whisper-1", generationMaxOutputTokensText: "invalid"
        )
        XCTAssertNil(sheet.validationMessageForGenerationSettings())
    }

    func testOMLXValidationRequiresObjectOnlyForSelectedSchemaFormat() {
        let sheet = makeSheet(target: .llm(.omlx), model: "qwen3")
        XCTAssertNil(sheet.validationMessageForOMLXSettings(
            responseFormat: OMLXResponseFormat.plain.rawValue, jsonSchema: "invalid", extraBodyJSON: "{}"
        ))
        XCTAssertNotNil(sheet.validationMessageForOMLXSettings(
            responseFormat: OMLXResponseFormat.jsonSchema.rawValue, jsonSchema: "", extraBodyJSON: "{}"
        ))
        XCTAssertNotNil(sheet.validationMessageForOMLXSettings(
            responseFormat: OMLXResponseFormat.jsonSchema.rawValue, jsonSchema: "[]", extraBodyJSON: "{}"
        ))
        XCTAssertNil(sheet.validationMessageForOMLXSettings(
            responseFormat: OMLXResponseFormat.jsonSchema.rawValue, jsonSchema: "{}", extraBodyJSON: "{}"
        ))
    }

    func testSelectingCustomProviderModelPrefillsCurrentBuiltinModel() {
        let customModelID = RemoteProviderConfigurationPolicy.nextCustomModelID(
            previousResolvedModel: "gpt-4o-transcribe",
            newSelection: RemoteProviderConfigurationPolicy.customModelOptionID,
            currentCustomModelID: "",
            supportsCustomSelection: true
        )

        XCTAssertEqual(customModelID, "gpt-4o-transcribe")
        XCTAssertEqual(
            RemoteProviderConfigurationPolicy.resolvedModelValue(
                target: .asr(.openAIWhisper),
                resolvedSelection: RemoteProviderConfigurationPolicy.customModelOptionID,
                customModelID: customModelID
            ),
            "gpt-4o-transcribe"
        )
    }

    func testSelectingCustomProviderModelKeepsExistingCustomValue() {
        let customModelID = RemoteProviderConfigurationPolicy.nextCustomModelID(
            previousResolvedModel: "gpt-4o-transcribe",
            newSelection: RemoteProviderConfigurationPolicy.customModelOptionID,
            currentCustomModelID: "gpt-4o-transcribe-preview",
            supportsCustomSelection: true
        )

        XCTAssertEqual(customModelID, "gpt-4o-transcribe-preview")
        XCTAssertEqual(
            RemoteProviderConfigurationPolicy.resolvedModelValue(
                target: .asr(.openAIWhisper),
                resolvedSelection: RemoteProviderConfigurationPolicy.customModelOptionID,
                customModelID: customModelID
            ),
            "gpt-4o-transcribe-preview"
        )
    }

    func testAliyunEndpointPresetsDependOnModelType() {
        let qwenPresets = RemoteProviderConfigurationPolicy.endpointPresets(
            target: .asr(.aliyunBailianASR),
            resolvedModel: "qwen3-asr-flash-realtime"
        )
        let funPresets = RemoteProviderConfigurationPolicy.endpointPresets(
            target: .asr(.aliyunBailianASR),
            resolvedModel: "fun-asr-realtime"
        )

        XCTAssertTrue(qwenPresets.allSatisfy { $0.url.contains("/realtime") })
        XCTAssertTrue(funPresets.allSatisfy { $0.url.contains("/inference") })
    }

    func testAliyunLLMEndpointPresetsUseResponsesAPI() {
        let presets = RemoteProviderConfigurationPolicy.endpointPresets(
            target: .llm(.aliyunBailian),
            resolvedModel: "qwen-plus"
        )

        XCTAssertEqual(
            presets.map(\.url),
            [
                "https://dashscope.aliyuncs.com/compatible-mode/v1/responses",
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses",
                "https://dashscope-us.aliyuncs.com/compatible-mode/v1/responses"
            ]
        )
    }

    func testVolcengineLLMEndpointPresetsUseResponsesAPI() {
        let presets = RemoteProviderConfigurationPolicy.endpointPresets(
            target: .llm(.volcengine),
            resolvedModel: "doubao-1-5-pro"
        )

        XCTAssertEqual(
            presets.map(\.url),
            [
                "https://ark.cn-beijing.volces.com/api/v3/responses"
            ]
        )
    }

    func testXiaomiMiMoEndpointPresetsUseOfficialChatCompletionsAPI() {
        let llmPresets = RemoteProviderConfigurationPolicy.endpointPresets(
            target: .llm(.xiaomiMiMo),
            resolvedModel: "mimo-v2.5-pro"
        )

        XCTAssertEqual(
            llmPresets.map(\.url),
            ["https://api.xiaomimimo.com/v1/chat/completions"]
        )

        XCTAssertEqual(
            RemoteProviderConfigurationPolicy.endpointPlaceholder(
                target: .asr(.xiaomiMiMoASR),
                resolvedModel: "mimo-v2.5-asr"
            ),
            "https://api.xiaomimimo.com/v1/chat/completions"
        )
    }

    func testStepFunSheetUsesDocumentedEffortControls() {
        let sheet = makeSheet(
            target: .llm(.stepFun),
            model: "step-3.5-flash-2603"
        )

        XCTAssertEqual(
            sheet.generationThinkingModeMenuOptions.map(\.value),
            [LLMThinkingMode.off.rawValue, LLMThinkingMode.effort.rawValue]
        )
        XCTAssertTrue(sheet.shouldShowGenerationThinking)
        XCTAssertEqual(
            sheet.generationThinkingEffortMenuOptions.map(\.value),
            ["low", "high"]
        )
    }

    func testStepFunSheetHidesEffortControlsForUnsupportedModels() {
        let sheet = makeSheet(
            target: .llm(.stepFun),
            model: "step-3.5-flash"
        )

        XCTAssertEqual(
            sheet.generationThinkingModeMenuOptions.map(\.value),
            [LLMThinkingMode.off.rawValue]
        )
        XCTAssertFalse(sheet.shouldShowGenerationThinking)
        XCTAssertTrue(sheet.generationThinkingEffortMenuOptions.isEmpty)
    }

    func testStepFunSheetDefaultsThinkingSnapshotToOff() {
        let sheet = makeSheet(
            target: .llm(.stepFun),
            model: "step-3.5-flash-2603"
        )

        XCTAssertEqual(sheet.currentGenerationSettingsSnapshot().thinking.mode, .off)
    }

    func testStepFunRouterModelUsesStepPlanEndpointPreset() {
        let presets = RemoteProviderConfigurationPolicy.endpointPresets(
            target: .llm(.stepFun),
            resolvedModel: "step-router-v1"
        )

        XCTAssertEqual(
            presets.map(\.url),
            ["https://api.stepfun.com/step_plan/v1/chat/completions"]
        )

        let endpoint = RemoteProviderConfigurationPolicy.remappedEndpointOnModelChange(
            target: .llm(.stepFun),
            previousModel: "step-3.5-flash",
            newModel: "step-router-v1",
            currentEndpoint: "https://api.stepfun.com/v1/chat/completions"
        )

        XCTAssertEqual(endpoint, "https://api.stepfun.com/step_plan/v1/chat/completions")
    }

    func testStepFunEndpointRemapLeavesCustomAPIPathUntouched() {
        let endpoint = RemoteProviderConfigurationPolicy.remappedEndpointOnModelChange(
            target: .llm(.stepFun),
            previousModel: "step-3.5-flash",
            newModel: "step-router-v1",
            currentEndpoint: "https://api.stepfun.com/custom/chat"
        )

        XCTAssertEqual(endpoint, "https://api.stepfun.com/custom/chat")
    }

    func testEndpointPlaceholderShowsResolvedProviderDefault() {
        XCTAssertEqual(
            RemoteProviderConfigurationPolicy.endpointPlaceholder(
                target: .llm(.openAI),
                resolvedModel: "gpt-5.2"
            ),
            "https://api.openai.com/v1/responses"
        )
        XCTAssertEqual(
            RemoteProviderConfigurationPolicy.endpointPlaceholder(
                target: .asr(.openAIWhisper),
                resolvedModel: "gpt-4o-mini-transcribe"
            ),
            "https://api.openai.com/v1/audio/transcriptions"
        )
    }

    func testAliyunASREndpointRemapsRegionWhenSwitchingModelFamilies() {
        let endpoint = RemoteProviderConfigurationPolicy.remappedEndpointOnModelChange(
            target: .asr(.aliyunBailianASR),
            previousModel: "qwen3-asr-flash-realtime",
            newModel: "fun-asr-realtime",
            currentEndpoint: "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
        )

        XCTAssertEqual(endpoint, "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference")
    }

    func testAliyunASREndpointLeavesCustomHostUntouchedWhenSwitchingModels() {
        let endpoint = RemoteProviderConfigurationPolicy.remappedEndpointOnModelChange(
            target: .asr(.aliyunBailianASR),
            previousModel: "qwen3-asr-flash-realtime",
            newModel: "fun-asr-realtime",
            currentEndpoint: "wss://example.com/custom-realtime"
        )

        XCTAssertEqual(endpoint, "wss://example.com/custom-realtime")
    }

    func testAliyunASREndpointFillsDefaultPresetWhenEmptyAndModelChanges() {
        let endpoint = RemoteProviderConfigurationPolicy.remappedEndpointOnModelChange(
            target: .asr(.aliyunBailianASR),
            previousModel: "qwen3-asr-flash-realtime",
            newModel: "fun-asr-realtime",
            currentEndpoint: ""
        )

        XCTAssertEqual(endpoint, "wss://dashscope.aliyuncs.com/api-ws/v1/inference")
    }

    private func makeSheet(
        target: RemoteProviderTestTarget,
        model: String,
        generationMaxOutputTokensText: String = ""
    ) -> RemoteProviderConfigurationSheet {
        RemoteProviderConfigurationSheet(
            providerTitle: providerTitle(for: target),
            credentialHint: nil,
            showsDoubaoFields: false,
            testTarget: target,
            configuration: RemoteProviderConfiguration(
                providerID: providerID(for: target),
                model: model,
                endpoint: "",
                apiKey: ""
            ),
            onSave: { _ in .success(()) },
            generationMaxOutputTokensText: generationMaxOutputTokensText
        )
    }

    private func providerTitle(for target: RemoteProviderTestTarget) -> String {
        switch target {
        case .asr(let provider):
            return provider.title
        case .llm(let provider):
            return provider.title
        }
    }

    private func providerID(for target: RemoteProviderTestTarget) -> String {
        switch target {
        case .asr(let provider):
            return provider.rawValue
        case .llm(let provider):
            return provider.rawValue
        }
    }
}
