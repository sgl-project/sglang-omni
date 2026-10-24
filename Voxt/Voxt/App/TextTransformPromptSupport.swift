// Pure template delivery and bounded rewrite conversation prompts.

import Foundation

enum TextTransformPromptSupport {
    static func deliveryForTemplate(
        _ template: String,
        variableTokens: [String]
    ) -> AppDelegate.VariablePromptDelivery {
        variableTokens.contains { template.contains($0) } ? .userMessage : .systemPrompt
    }

    static func rewriteConversationPrompt(forceNonEmptyAnswer: Bool) -> String {
        let retryConstraint = forceNonEmptyAnswer
            ? """
            Retry rule:
            - A previous answer was empty, quoted-empty, or otherwise unusable.
            - This time, you must return a non-empty plain-text answer.
            - Do not return surrounding quotes.
            """
            : ""

        let base = """
        You are Voxt's follow-up voice conversation assistant.

        The previous conversation is provided as chat messages.
        Respond to the latest user message directly based on that conversation.

        Rules:
        1. Treat the latest user message as a follow-up to the previous assistant reply.
        2. If the user omits context with a short follow-up like “那大同呢”, infer the missing subject from the conversation history.
        3. Treat short confirmations such as “对”, “是”, or “yes” as answers to the latest assistant question, incorporate them, and continue the original task.
        4. Never repeat or paraphrase the latest assistant reply as the next answer, and never ask the same clarification twice.
        5. Do not ask the user to confirm a place, date, subject, or qualifier that already appears in the conversation.
        6. Ask a clarification question only when essential information is genuinely absent from the whole conversation.
        7. If current information cannot be verified because live lookup is unavailable, state that limitation directly instead of treating the request as ambiguous.
        8. Return plain text only.
        9. Do not return JSON, field names, markdown, or surrounding quotes.
        10. Do not return an empty string.
        """

        let prompt = [base, retryConstraint]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")

        return prompt
    }

    static func trimmedRewriteConversationHistory(
        _ turns: [RewriteConversationPromptTurn]
    ) -> [RewriteConversationPromptTurn] {
        guard !turns.isEmpty else { return [] }
        let budget = (maxTurns: 4, maxCharacters: 1_600)

        let candidates = Array(turns.suffix(budget.maxTurns)).reversed()
        var selectedReversed: [RewriteConversationPromptTurn] = []
        var consumedCharacters = 0

        for turn in candidates {
            let turnCharacters = turn.modelUserMessage.count + turn.resultContent.count
            let separatorCost = selectedReversed.isEmpty ? 0 : 4
            let nextCount = consumedCharacters + separatorCost + turnCharacters

            if !selectedReversed.isEmpty && nextCount > budget.maxCharacters {
                break
            }

            selectedReversed.append(turn)
            consumedCharacters = nextCount
        }

        return selectedReversed.reversed()
    }
}
