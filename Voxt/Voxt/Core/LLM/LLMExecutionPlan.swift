// LLMExecutionPlan.swift
// Provides LLMExecution Plan for LLM requests and response handling.

import Foundation

enum LLMExecutionDelivery: Equatable {
    case systemPrompt
    case userMessage
}

enum LLMExecutionProvider: Equatable {
    case appleIntelligence
    case customLLM(repo: String)
    case localGGUF(modelID: GGUFTranslationModelID)
    case remote(provider: RemoteLLMProvider, configuration: RemoteProviderConfiguration)
}

enum LLMContextBlockKind: String, Equatable {
    case input
    case glossary
    case conversation
    case metadata
}

struct LLMContextBlock: Equatable {
    let kind: LLMContextBlockKind
    let title: String
    let content: String
    let isStablePrefixCandidate: Bool

    var trimmedContent: String {
        content.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

enum LLMExecutionTaskPayload: Equatable {
    case enhancement(rawText: String)
    case translation(sourceText: String, targetLanguage: TranslationTargetLanguage)
    case rewrite(dictatedPrompt: String, sourceText: String, structuredAnswerOutput: Bool)
    case transcriptSummary(transcript: String, request: String)
}

struct LLMExecutionPlan: Equatable {
    let task: LLMExecutionTaskPayload
    let provider: LLMExecutionProvider
    let delivery: LLMExecutionDelivery
    let promptContent: String
    let fallbackText: String
    let executionStrategy: TaskLLMExecutionStrategy
    let outputTokenBudgetHint: Int?
    let contextBlocks: [LLMContextBlock]
    let conversationHistory: [RewriteConversationPromptTurn]
    let previousResponseID: String?
    let responseFormat: RemoteLLMRuntimeClient.OpenAICompatibleResponseFormat?

    var promptCharacterCount: Int {
        promptContent.count
    }

    var primaryInputCharacterCount: Int {
        switch task {
        case .enhancement(let rawText):
            return rawText.count
        case .translation(let sourceText, _):
            return sourceText.count
        case .rewrite(let dictatedPrompt, let sourceText, _):
            return dictatedPrompt.count + sourceText.count
        case .transcriptSummary(let transcript, let request):
            return transcript.count + request.count
        }
    }

    var taskLabel: String {
        switch task {
        case .enhancement:
            return "enhancement"
        case .translation:
            return "translation"
        case .rewrite:
            return "rewrite"
        case .transcriptSummary:
            return "transcriptSummary"
        }
    }
}

struct LLMCompiledRequest: Equatable {
    let taskLabel: String
    let instructions: String
    let prompt: String
    let debugInput: String
    let fallbackText: String
    let inputCharacterCount: Int
    let outputTokenBudgetHint: Int?
    let conversationHistory: [RewriteConversationPromptTurn]
    let previousResponseID: String?
    let responseFormat: RemoteLLMRuntimeClient.OpenAICompatibleResponseFormat?
}
