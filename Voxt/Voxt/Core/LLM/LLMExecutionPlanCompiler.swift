// LLMExecutionPlanCompiler.swift
// Provides LLMExecution Plan Compiler for LLM requests and response handling.

import Foundation

enum LLMExecutionPlanCompiler {
    static func compile(_ plan: LLMExecutionPlan) -> LLMCompiledRequest {
        let includesExternalConversationState =
            !plan.conversationHistory.isEmpty ||
            !(plan.previousResponseID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "").isEmpty
        let contextInstructions = compiledInstructionSections(
            from: plan.contextBlocks,
            includeConversation: !includesExternalConversationState
        )
        switch plan.delivery {
        case .userMessage:
            return LLMCompiledRequest(
                taskLabel: plan.taskLabel,
                instructions: joinInstructionSections(contextInstructions),
                prompt: plan.promptContent,
                debugInput: debugInput(for: plan.task),
                fallbackText: plan.fallbackText,
                inputCharacterCount: plan.primaryInputCharacterCount,
                outputTokenBudgetHint: plan.outputTokenBudgetHint,
                conversationHistory: plan.conversationHistory,
                previousResponseID: plan.previousResponseID,
                responseFormat: plan.responseFormat
            )

        case .systemPrompt:
            let promptSections = [plan.promptContent] + contextInstructions
            return LLMCompiledRequest(
                taskLabel: plan.taskLabel,
                instructions: joinInstructionSections(promptSections),
                prompt: requestPrompt(for: plan.task),
                debugInput: debugInput(for: plan.task),
                fallbackText: plan.fallbackText,
                inputCharacterCount: plan.primaryInputCharacterCount,
                outputTokenBudgetHint: plan.outputTokenBudgetHint,
                conversationHistory: plan.conversationHistory,
                previousResponseID: plan.previousResponseID,
                responseFormat: plan.responseFormat
            )
        }
    }

    private static func requestPrompt(for task: LLMExecutionTaskPayload) -> String {
        switch task {
        case .enhancement(let rawText):
            return """
            Process this ASR transcription according to the system instructions.
            Return only the final processed text.
            Input:
            \(rawText)
            """

        case .translation(let sourceText, _):
            return """
            Translate this text.
            Input:
            \(sourceText)
            """

        case .rewrite(let dictatedPrompt, let sourceText, _):
            if sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return dictatedPrompt
            }
            return """
            Return the final text to insert.
            Spoken instruction:
            \(dictatedPrompt)

            Selected source text:
            \(sourceText)
            """

        case .transcriptSummary(_, let request):
            return request
        }
    }

    private static func debugInput(for task: LLMExecutionTaskPayload) -> String {
        switch task {
        case .enhancement(let rawText):
            return rawText
        case .translation(let sourceText, _):
            return sourceText
        case .rewrite(let dictatedPrompt, let sourceText, _):
            return """
            Spoken instruction:
            \(dictatedPrompt)

            Selected source text:
            \(sourceText)
            """
        case .transcriptSummary(let transcript, let request):
            return """
            Request:
            \(request)

            Transcript:
            \(transcript)
            """
        }
    }

    private static func compiledInstructionSections(
        from blocks: [LLMContextBlock],
        includeConversation: Bool
    ) -> [String] {
        blocks.compactMap { block in
            guard block.kind != .input else { return nil }
            guard includeConversation || block.kind != .conversation else { return nil }
            let content = block.trimmedContent
            guard !content.isEmpty else { return nil }
            return """
            ### \(block.title)
            \(content)
            """
        }
    }

    private static func joinInstructionSections(_ sections: [String]) -> String {
        sections
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }
}
