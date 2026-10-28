import XCTest
@testable import Voxt

final class LLMExecutionPlanCompilerTests: XCTestCase {
    private func plan(
        task: LLMExecutionTaskPayload,
        delivery: LLMExecutionDelivery = .systemPrompt,
        prompt: String,
        blocks: [LLMContextBlock] = [],
        history: [RewriteConversationPromptTurn] = [],
        previousResponseID: String? = nil
    ) -> LLMExecutionPlan {
        LLMExecutionPlan(
            task: task,
            provider: .customLLM(repo: "test/repo"),
            delivery: delivery,
            promptContent: prompt,
            fallbackText: "fallback",
            executionStrategy: TaskLLMStrategyResolver.resolve(
                taskKind: .rewrite, rawText: "input", promptCharacterCount: prompt.count,
                baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.rewrite.selectionPolicy,
                capabilities: .unknown
            ),
            outputTokenBudgetHint: 321,
            contextBlocks: blocks,
            conversationHistory: history,
            previousResponseID: previousResponseID,
            responseFormat: nil
        )
    }

    func testUserMessageCompilationMovesGlossaryIntoInstructions() {
        let compiled = LLMExecutionPlanCompiler.compile(plan(
            task: .translation(sourceText: "hello", targetLanguage: .english),
            delivery: .userMessage,
            prompt: "Translate hello to English.",
            blocks: [.init(kind: .glossary, title: "Dictionary Guidance", content: "- OpenAI", isStablePrefixCandidate: true)]
        ))
        XCTAssertEqual(compiled.prompt, "Translate hello to English.")
        XCTAssertContains(compiled.instructions, "### Dictionary Guidance")
        XCTAssertContains(compiled.instructions, "- OpenAI")
        XCTAssertEqual(compiled.outputTokenBudgetHint, 321)
    }

    func testSystemPromptCompilationKeepsRequestPromptDynamicAndGlossaryStable() {
        let compiled = LLMExecutionPlanCompiler.compile(plan(
            task: .enhancement(rawText: "raw transcript"),
            prompt: "Clean up the transcript.",
            blocks: [
                .init(kind: .glossary, title: "Dictionary Guidance", content: "- Anthropic", isStablePrefixCandidate: true),
                .init(kind: .input, title: "Raw transcription", content: "raw transcript", isStablePrefixCandidate: false)
            ]
        ))
        XCTAssertContains(compiled.instructions, "Clean up the transcript.")
        XCTAssertContains(compiled.instructions, "### Dictionary Guidance")
        XCTAssertContains(compiled.instructions, "- Anthropic")
        XCTAssertContains(compiled.prompt, "Process this ASR transcription according to the system instructions.")
        XCTAssertContains(compiled.prompt, "raw transcript")
        XCTAssertFalse(compiled.instructions.contains("Raw transcription"))
    }

    func testSystemPromptCompilationIncludesMetadata() {
        let compiled = LLMExecutionPlanCompiler.compile(plan(
            task: .enhancement(rawText: "raw transcript"),
            prompt: "Clean up the transcript.",
            blocks: [.init(kind: .metadata, title: "Runtime rules", content: "quality", isStablePrefixCandidate: true)]
        ))
        XCTAssertContains(compiled.instructions, "### Runtime rules")
        XCTAssertContains(compiled.instructions, "quality")
    }

    func testCompilationKeepsConversationAsExternalRoleMessages() {
        let history = [RewriteConversationPromptTurn(
            userPromptText: "北京今天的天气情况", resultTitle: "北京天气", resultContent: "请问您需要查询哪一天的天气？"
        )]
        let compiled = LLMExecutionPlanCompiler.compile(plan(
            task: .rewrite(dictatedPrompt: "对", sourceText: "", structuredAnswerOutput: false),
            prompt: "Answer the follow-up directly.",
            blocks: [.init(kind: .conversation, title: "Previous conversation", content: "Already in history", isStablePrefixCandidate: false)],
            history: history
        ))
        XCTAssertEqual(compiled.conversationHistory, history)
        XCTAssertFalse(compiled.instructions.contains("Previous conversation"))
        XCTAssertEqual(compiled.prompt, "对")
    }

    func testPreviousResponseIDAndSelectedTextSurviveTextOnlyCompilation() {
        let compiled = LLMExecutionPlanCompiler.compile(plan(
            task: .rewrite(dictatedPrompt: "shorter", sourceText: "Supplied text", structuredAnswerOutput: true),
            prompt: "Rewrite.",
            previousResponseID: "resp_previous"
        ))
        XCTAssertEqual(compiled.previousResponseID, "resp_previous")
        XCTAssertContains(compiled.prompt, "Selected source text:\nSupplied text")
        XCTAssertContains(compiled.prompt, "Spoken instruction:\nshorter")
        XCTAssertFalse(compiled.instructions.contains("Active app context"))
    }

    func testReducedLongInputGlossaryPolicyTightensBudget() {
        let standard = DictionaryGlossaryPurpose.rewrite.selectionPolicy
        let reduced = standard.reducedForLongInput()
        XCTAssertLessThan(reduced.maxTerms, standard.maxTerms)
        XCTAssertLessThan(reduced.maxCharacters, standard.maxCharacters)
    }
}
