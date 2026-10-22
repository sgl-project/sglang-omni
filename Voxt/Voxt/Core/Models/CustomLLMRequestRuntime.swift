// Stateless request policy. The manager retains container lifetime, task tracking
// and diagnostic publication; no mutable inference owner is introduced here.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

enum CustomLLMRequestRuntime {
    private struct TextResultPayload: Decodable {
        let resultText: String
    }

    static func generationParameters(
        for request: CustomLLMRequestPlan,
        behavior: CustomLLMModelBehavior,
        settings: LLMGenerationSettings,
        tuning: CustomLLMGenerationTuning
    ) -> GenerateParameters {
        let safeInput = max(1, request.inputCharacterCount)
        let estimated = Int(Double(safeInput) * request.kind.tokenBudgetMultiplier)
        let totalPromptCharacters = request.instructions.count + request.prompt.count
        let budget: Int?
        if let override = tuning.maxTokensOverride {
            budget = max(1, override)
        } else if let override = settings.maxOutputTokens {
            budget = max(1, override)
        } else if let override = request.maxTokensOverride {
            budget = max(1, override)
        } else {
            budget = defaultOutputTokenBudget(for: request.kind, estimated: estimated)
        }

        let prefillStepSize: Int
        if let override = tuning.prefillStepSizeOverride {
            prefillStepSize = override
        } else {
            switch totalPromptCharacters {
            case ..<1000:
                prefillStepSize = 256
            case ..<3000:
                prefillStepSize = 512
            default:
                prefillStepSize = 768
            }
        }

        let repetitionPenalty: Float? =
            settings.repetitionPenalty.map(Float.init) ?? (behavior.family == .qwen3 ? 1.05 : nil)

        return GenerateParameters(
            maxTokens: budget,
            temperature: settings.temperature.map(Float.init) ?? 0,
            topP: settings.topP.map(Float.init) ?? 1.0,
            topK: settings.topK ?? 0,
            minP: settings.minP.map(Float.init) ?? 0,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: 32,
            prefill: .init(stepSize: prefillStepSize, chunking: .remainder)
        )
    }

    private static func defaultOutputTokenBudget(for kind: CustomLLMTaskKind, estimated: Int) -> Int {
        switch kind {
        case .enhancement:
            return max(128, min(estimated + 128, 1024))
        case .translation:
            return max(128, min(estimated + 160, 1024))
        case .rewrite:
            return max(256, min(estimated + 192, 1536))
        case .dictionaryHistoryScan:
            return max(256, min(estimated + 96, 2048))
        }
    }


    static func startLogMessage(
        for request: CustomLLMRequestPlan,
        params: GenerateParameters,
        behavior: CustomLLMModelBehavior
    ) -> String {
        var suffix = ""
        if let mode = request.logMode {
            suffix = ", mode=\(mode)"
        }
        let maxTokens = params.maxTokens.map(String.init) ?? "0"
        let prefillStepSize = params.prefill.stepSize.map(String.init) ?? "0"
        return "Custom LLM \(request.kind.logLabel) started. repo=\(request.repo), inputChars=\(request.inputCharacterCount), maxTokens=\(maxTokens), temperature=\(params.temperature), topP=\(params.topP), prefillStep=\(prefillStepSize)\(suffix), family=\(behavior.family.logLabel), thinkingDisabled=\(behavior.disablesThinking)"
    }

    static func contentLogMessage(for request: CustomLLMRequestPlan) -> String {
        var lines = ["Custom LLM \(request.kind.logLabel) content. repo=\(request.repo)"]
        for section in request.contentLogSections {
            lines.append("[\(section.label)]")
            lines.append(VoxtLog.llmPreview(section.content))
        }
        return lines.joined(separator: "\n")
    }


    static func makeChatSession(
        container: ModelContainer,
        instructions: String,
        conversationHistory: [RewriteConversationPromptTurn],
        repo: String,
        behavior: CustomLLMModelBehavior,
        settings: LLMGenerationSettings
    ) -> ChatSession {
        let additionalContext = localThinkingAdditionalContext(
            behavior: behavior,
            settings: settings
        )
        let history = conversationHistory.flatMap { turn -> [Chat.Message] in
            var messages: [Chat.Message] = []
            let userMessage = turn.modelUserMessage
            if !userMessage.isEmpty {
                messages.append(.user(userMessage))
            }
            let assistantMessage = turn.resultContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if !assistantMessage.isEmpty {
                messages.append(.assistant(assistantMessage))
            }
            return messages
        }
        let session = ChatSession(
            container,
            instructions: instructions,
            history: history,
            additionalContext: additionalContext
        )
        if additionalContext?["enable_thinking"] as? Bool == false {
            VoxtLog.llmDebug("Custom LLM thinking disabled for repo=\(repo) using chat-template additionalContext.")
        } else if additionalContext?["enable_thinking"] as? Bool == true {
            VoxtLog.llmDebug("Custom LLM thinking enabled for repo=\(repo) using chat-template additionalContext.")
        }
        return session
    }

    private static func localThinkingAdditionalContext(
        behavior: CustomLLMModelBehavior,
        settings: LLMGenerationSettings
    ) -> [String: any Sendable]? {
        switch settings.thinking.mode {
        case .providerDefault:
            return CustomLLMModelBehavior.thinkingOffAdditionalContext
        case .off:
            return CustomLLMModelBehavior.thinkingOffAdditionalContext
        case .on:
            return CustomLLMModelBehavior.thinkingOnAdditionalContext
        case .effort, .budget:
            return CustomLLMModelBehavior.thinkingOffAdditionalContext
        }
    }

    static func dictionaryHistoryScanStructuredOutputPrompt(_ prompt: String) -> String {
        """
        Analyze the following task and return only valid JSON.

        Final answer requirements:
        - Return only a JSON array.
        - Every item must be an object with exactly one key: "term".
        - Example: [{"term":"OpenAI"},{"term":"MCP"}]
        - If no term qualifies, return [].
        - Do not wrap the array in another object.
        - Do not return prose, markdown, code fences, or explanations.

        Task:
        \(prompt)
        """
    }

    static func extractResultText(_ output: String) -> String {
        let normalized = sanitizeModelOutput(output)
        if let parsed = decodeStructuredResultText(from: normalized) {
            return parsed
        }
        return normalized
    }

    private static func decodeStructuredResultText(from output: String) -> String? {
        for candidate in jsonCandidates(from: output) {
            guard let data = candidate.data(using: .utf8),
                  let decoded = try? JSONDecoder().decode(TextResultPayload.self, from: data) else {
                continue
            }
            let text = CustomLLMOutputSanitizer.normalizeResultText(decoded.resultText)
            if !text.isEmpty {
                return text
            }
        }
        return nil
    }

    private static func jsonCandidates(from output: String) -> [String] {
        let normalized = output.trimmingCharacters(in: .whitespacesAndNewlines)
        var candidates: [String] = [normalized]

        let unfenced = CustomLLMOutputSanitizer.unwrapCodeFenceIfNeeded(normalized)
        if unfenced != normalized {
            candidates.append(unfenced)
        }

        if let jsonObject = Self.extractFirstJSONObject(in: unfenced),
           !candidates.contains(jsonObject) {
            candidates.append(jsonObject)
        }

        return candidates
    }

    private static func extractFirstJSONObject(in text: String) -> String? {
        guard let start = text.firstIndex(of: "{"),
              let end = text.lastIndex(of: "}"),
              start <= end else {
            return nil
        }
        return String(text[start...end])
    }

    static func sanitizeModelOutput(_ output: String) -> String {
        let cleaned = CustomLLMOutputSanitizer.normalizeResultText(output)
        if cleaned != output.trimmingCharacters(in: .whitespacesAndNewlines) {
            VoxtLog.llm(
                """
                Custom LLM output sanitized.
                [raw]
                \(VoxtLog.llmPreview(output))
                [cleaned]
                \(VoxtLog.llmPreview(cleaned))
                """
            )
        }
        return cleaned
    }
}
