import XCTest
@testable import Voxt
import HuggingFace
import MLX
import MLXAudioSTT

@MainActor
final class CustomLLMModelConfigurationTests: MLXModelManagerTestCase {
    func testKnownCustomLLMRemoteSizeFallbacksRemainAvailable() {
        XCTAssertNotNil(CustomLLMModelManager.fallbackRemoteSizeText(repo: "Qwen/Qwen2-1.5B-Instruct"))
        XCTAssertNotNil(CustomLLMModelManager.fallbackRemoteSizeText(repo: "mlx-community/Qwen3-8B-4bit"))
    }

    func testCustomLLMGenerationSettingsDefaultToThinkingOff() {
        XCTAssertEqual(CustomLLMGenerationSettingsStore.defaultSettings.thinking.mode, .off)
        XCTAssertEqual(CustomLLMGenerationSettingsStore.resolvedSettings(from: nil).thinking.mode, .off)
        XCTAssertEqual(
            CustomLLMGenerationSettingsStore.sanitized(
                LLMGenerationSettings(thinking: .providerDefault)
            ).thinking.mode,
            .off
        )
        for repo in [
            "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
            "mlx-community/LFM2-1.2B-4bit",
            "mlx-community/LFM2-8B-A1B-3bit-MLX",
            "mlx-community/Qwen3.6-27B-4bit",
        ] {
            XCTAssertEqual(
                CustomLLMGenerationSettingsStore.resolvedSettings(
                    for: repo,
                    rawByRepo: nil,
                    legacyRaw: nil
                ).thinking.mode,
                .off,
                repo
            )
        }
        XCTAssertEqual(
            CustomLLMGenerationSettingsStore.resolvedSettings(
                for: "mlx-community/Qwen3.5-4B-OptiQ-4bit",
                rawByRepo: nil,
                legacyRaw: nil
            ).thinking.mode,
            .off
        )
        XCTAssertEqual(
            CustomLLMGenerationSettingsStore.resolvedSettings(
                from: CustomLLMGenerationSettingsStore.defaultStoredValue()
            ).thinking.mode,
            .off
        )
    }

    func testCustomLLMGenerationSettingsStoreKeepsOnlyLocalSupportedFields() {
        let raw = CustomLLMGenerationSettingsStore.storageValue(
            for: LLMGenerationSettings(
                maxOutputTokens: 4096,
                temperature: 0.25,
                topP: 0.9,
                topK: 40,
                minP: 0.05,
                seed: 123,
                stop: ["END"],
                presencePenalty: 1,
                frequencyPenalty: 1,
                repetitionPenalty: 1.08,
                logprobs: true,
                topLogprobs: 5,
                responseFormat: .json,
                thinking: LLMThinkingSettings(mode: .off, effort: "low", budgetTokens: 1024, exposeReasoning: true),
                extraBodyJSON: #"{"unused":true}"#,
                extraOptionsJSON: #"{"unused":true}"#
            )
        )

        let restored = CustomLLMGenerationSettingsStore.resolvedSettings(from: raw)
        XCTAssertEqual(restored.maxOutputTokens, 4096)
        XCTAssertEqual(restored.temperature, 0.25)
        XCTAssertEqual(restored.topP, 0.9)
        XCTAssertEqual(restored.topK, 40)
        XCTAssertEqual(restored.minP, 0.05)
        XCTAssertEqual(restored.repetitionPenalty, 1.08)
        XCTAssertEqual(restored.thinking.mode, .off)
        XCTAssertNil(restored.seed)
        XCTAssertEqual(restored.stop, [])
        XCTAssertNil(restored.presencePenalty)
        XCTAssertNil(restored.frequencyPenalty)
        XCTAssertFalse(restored.logprobs)
        XCTAssertNil(restored.topLogprobs)
        XCTAssertEqual(restored.responseFormat, .plain)
        XCTAssertEqual(restored.extraBodyJSON, "")
        XCTAssertEqual(restored.extraOptionsJSON, "")
    }

    func testCustomLLMGenerationSettingsStoreCanonicalizesRepoKeys() {
        let raw = CustomLLMGenerationSettingsStore.save(
            LLMGenerationSettings(temperature: 0.33),
            for: "mlx-community/Qwen3.5-2B-MLX-4bit",
            rawByRepo: nil
        )
        let values = CustomLLMGenerationSettingsStore.resolvedByRepo(from: raw)

        XCTAssertNil(values["mlx-community/Qwen3.5-2B-MLX-4bit"])
        XCTAssertEqual(values["mlx-community/Qwen3.5-2B-4bit"]?.temperature, 0.33)
        XCTAssertEqual(
            CustomLLMGenerationSettingsStore.resolvedSettings(
                for: "mlx-community/Qwen3.5-2B-MLX-4bit",
                rawByRepo: raw,
                legacyRaw: nil
            ).temperature,
            0.33
        )
    }

    func testCustomLLMGenerationSettingsStoreFallsBackToLegacySettings() {
        let legacyRaw = CustomLLMGenerationSettingsStore.storageValue(
            for: LLMGenerationSettings(temperature: 0.44, topP: 0.8)
        )

        let restored = CustomLLMGenerationSettingsStore.resolvedSettings(
            for: "mlx-community/Qwen3-4B-4bit",
            rawByRepo: nil,
            legacyRaw: legacyRaw
        )

        XCTAssertEqual(restored.temperature, 0.44)
        XCTAssertEqual(restored.topP, 0.8)
    }

    func testCustomLLMTaskKindUsesExpectedTokenBudgetMultipliers() {
        XCTAssertEqual(CustomLLMTaskKind.enhancement.tokenBudgetMultiplier, 1.10, accuracy: 0.0001)
        XCTAssertEqual(CustomLLMTaskKind.translation.tokenBudgetMultiplier, 1.35, accuracy: 0.0001)
        XCTAssertEqual(CustomLLMTaskKind.rewrite.tokenBudgetMultiplier, 1.35, accuracy: 0.0001)
    }

    func testManagerFallsBackForUnsupportedRepo() async {
        await withIsolatedModelStorageRoot { _ in
            let manager = CustomLLMModelManager(modelRepo: "unsupported/repo")
            XCTAssertEqual(manager.currentModelRepo, CustomLLMModelManager.defaultModelRepo)
            await manager.shutdownForApplicationTermination()
        }
    }

    func testManagerPreservesSupportedRepoOnSelection() async throws {
        try await withIsolatedModelStorageRoot { _ in
            let selected = try XCTUnwrap(CustomLLMModelManager.availableModels.last?.id)
            let manager = CustomLLMModelManager(modelRepo: selected)
            XCTAssertEqual(manager.currentModelRepo, selected)
            manager.updateModel(repo: CustomLLMModelManager.defaultModelRepo)
            XCTAssertEqual(manager.currentModelRepo, CustomLLMModelManager.defaultModelRepo)
            await manager.shutdownForApplicationTermination()
        }
    }

    func testCustomLLMRequestPlanBuilderBuildsUserPromptEnhancementRequest() {
        let plan = CustomLLMRequestPlanBuilder.userPromptEnhancement(
            prompt: "rewrite this",
            repo: "Qwen/Qwen2-1.5B-Instruct"
        )

        XCTAssertEqual(plan.kind, .enhancement)
        XCTAssertEqual(plan.instructions, "")
        XCTAssertEqual(plan.prompt, "rewrite this")
        XCTAssertEqual(plan.logMode, "userMessage")
        XCTAssertEqual(plan.contentLogSections.map(\.label), ["system_prompt", "input"])
        XCTAssertEqual(plan.contentLogSections.first?.content, "<empty>")
    }

    func testCustomLLMCompiledPlanPreservesOutputTokenBudgetHint() {
        let compiled = LLMCompiledRequest(
            taskLabel: "enhancement",
            instructions: "system",
            prompt: "prompt",
            debugInput: "input",
            fallbackText: "fallback",
            inputCharacterCount: 5,
            outputTokenBudgetHint: 321,
            conversationHistory: [],
            previousResponseID: nil,
            responseFormat: nil
        )

        let plan = CustomLLMRequestPlanBuilder.compiled(
            request: compiled,
            repo: "mlx-community/Qwen3.5-4B-OptiQ-4bit"
        )

        XCTAssertEqual(plan.maxTokensOverride, 321)
    }

    func testCustomLLMCompiledPlanPreservesRoleBasedConversationHistory() {
        let history = [
            RewriteConversationPromptTurn(
                userPromptText: "北京今天的天气情况",
                resultTitle: "北京天气",
                resultContent: "请问您需要查询哪一天的天气？"
            )
        ]
        let compiled = LLMCompiledRequest(
            taskLabel: "rewrite",
            instructions: "Answer the latest user message.",
            prompt: "对",
            debugInput: "对",
            fallbackText: "",
            inputCharacterCount: 1,
            outputTokenBudgetHint: nil,
            conversationHistory: history,
            previousResponseID: nil,
            responseFormat: nil
        )

        let plan = CustomLLMRequestPlanBuilder.compiled(
            request: compiled,
            repo: "mlx-community/Qwen3.5-4B-OptiQ-4bit"
        )

        XCTAssertEqual(plan.conversationHistory, history)
        XCTAssertFalse(plan.instructions.contains("Previous conversation:"))
        XCTAssertEqual(plan.prompt, "对")
    }

    func testCustomLLMNormalizeResultTextStripsThinkBlocksAndMarkers() {
        let output = """
        <think>
        reason
        </think>

        ```json
        {"resultText":"Hello"}
        ```
        """

        XCTAssertEqual(CustomLLMOutputSanitizer.normalizeResultText(output), #"{"resultText":"Hello"}"#)
        XCTAssertEqual(CustomLLMOutputSanitizer.normalizeResultText("<think>\n\n</think>\n\nHello"), "Hello")
    }
}
