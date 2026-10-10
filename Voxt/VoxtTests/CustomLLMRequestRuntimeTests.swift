import XCTest
import MLXLMCommon
@testable import Voxt

@MainActor
final class CustomLLMRequestRuntimeTests: XCTestCase {
    func testCompiledPlanRoutesCurrentTasksAndPreservesRequestContract() {
        let tasks: [(String, CustomLLMTaskKind)] = [
            ("enhancement", .enhancement), ("translation", .translation),
            ("rewrite", .rewrite), ("transcriptSummary", .enhancement)
        ]
        for (label, kind) in tasks {
            let plan = makePlan(label: label, instructions: "system", prompt: "input", budget: 321)
            XCTAssertEqual(plan.kind, kind, label)
            XCTAssertEqual(plan.repo, "test/repo")
            XCTAssertEqual(plan.instructions, "system")
            XCTAssertEqual(plan.prompt, "input")
            XCTAssertEqual(plan.inputCharacterCount, 5)
            XCTAssertEqual(plan.resultFallback, "fallback")
            XCTAssertEqual(plan.maxTokensOverride, 321)
            XCTAssertEqual(plan.contentLogSections.map(\.label), ["system_prompt", "input", "request_content"])
            XCTAssertNil(plan.logMode)
        }
    }

    func testStructuredResultExtractionPreservesCurrentEnvelopeSupport() {
        for value in [
            #"{"resultText":"hello"}"#,
            "```json\n{\"resultText\":\"hello\"}\n```",
            "Prefix\n{\"resultText\":\"hello\"}\nSuffix"
        ] {
            XCTAssertEqual(CustomLLMRequestRuntime.extractResultText(value), "hello")
        }
    }

    func testInvalidEnvelopeFallsBackToNormalizedText() {
        XCTAssertEqual(CustomLLMRequestRuntime.extractResultText("<think>hidden</think> visible "), "visible")
        XCTAssertEqual(CustomLLMRequestRuntime.extractResultText(#"{"resultText":42}"#), #"{"resultText":42}"#)
        XCTAssertEqual(CustomLLMRequestRuntime.extractResultText("plain answer"), "plain answer")
    }

    func testOutputBudgetPrecedenceIsDebugThenSettingsThenCompiledHint() {
        let plan = makePlan(budget: 321)
        let settings = LLMGenerationSettings(maxOutputTokens: 512)
        let behavior = CustomLLMModelBehavior(family: .qwen3, disablesThinking: true)
        let tuned = CustomLLMRequestRuntime.generationParameters(
            for: plan, behavior: behavior, settings: settings,
            tuning: .init(prefillStepSizeOverride: 64, maxTokensOverride: 77)
        )
        XCTAssertEqual(tuned.maxTokens, 77)
        XCTAssertEqual(tuned.prefill.stepSize, 64)
        let configured = CustomLLMRequestRuntime.generationParameters(for: plan, behavior: behavior, settings: settings, tuning: .default)
        XCTAssertEqual(configured.maxTokens, 512)
        let hinted = CustomLLMRequestRuntime.generationParameters(for: plan, behavior: behavior, settings: .init(), tuning: .default)
        XCTAssertEqual(hinted.maxTokens, 321)
    }

    func testDefaultGenerationAndQwenRepetitionPolicyArePreserved() {
        let plan = makePlan(prompt: "12345678")
        let params = CustomLLMRequestRuntime.generationParameters(
            for: plan,
            behavior: .init(family: .qwen3, disablesThinking: true),
            settings: .init(), tuning: .default
        )
        XCTAssertEqual(params.maxTokens, 136)
        XCTAssertEqual(params.temperature, 0)
        XCTAssertEqual(params.topP, 1)
        XCTAssertEqual(params.repetitionPenalty, Float(1.05))
    }

    func testPrefillWindowUsesTotalPromptCharacters() {
        for (count, expected) in [(999, 256), (1000, 512), (2999, 512), (3000, 768)] {
            let plan = makePlan(prompt: String(repeating: "a", count: count))
            let params = CustomLLMRequestRuntime.generationParameters(
                for: plan,
                behavior: .init(family: .qwen3, disablesThinking: true),
                settings: .init(), tuning: .default
            )
            XCTAssertEqual(params.prefill.stepSize, expected, "characters=\(count)")
        }
    }

    private func makePlan(
        label: String = "enhancement", instructions: String = "", prompt: String = "input", budget: Int? = nil
    ) -> CustomLLMRequestPlan {
        CustomLLMRequestPlanBuilder.compiled(
            request: LLMCompiledRequest(
                taskLabel: label, instructions: instructions, prompt: prompt,
                debugInput: prompt, fallbackText: "fallback", inputCharacterCount: prompt.count,
                outputTokenBudgetHint: budget, conversationHistory: [],
                previousResponseID: nil, responseFormat: nil
            ),
            repo: "test/repo"
        )
    }
}
