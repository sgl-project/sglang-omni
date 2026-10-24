// TextPromptBuilders.swift
// Provides Text Prompt Builders for core app behavior.

import Foundation

struct TranslationPromptBuilder {
    static func build(
        systemPrompt: String,
        targetLanguage: TranslationTargetLanguage,
        sourceText: String,
        userMainLanguagePromptValue: String,
        strict: Bool
    ) -> String {
        let basePrompt = systemPrompt
            .replacingOccurrences(of: "{target_language}", with: targetLanguage.instructionName)
            .replacingOccurrences(of: "{{TARGET_LANGUAGE}}", with: targetLanguage.instructionName)
            .replacingOccurrences(of: "{{SOURCE_TEXT}}", with: sourceText)
            .replacingOccurrences(of: AppDelegate.userMainLanguageTemplateVariable, with: userMainLanguagePromptValue)

        let enforcement = strict
            ? """
            Mandatory translation rules:
            - Translate every linguistic token into \(targetLanguage.instructionName), including very short text.
            \(targetLanguage.translationScriptConstraint.map { "- \($0)" } ?? "")
            - Do not copy source-language wording.
            - Keep proper nouns, product names, URLs, emails, and pure numbers unchanged when needed.
            - Return translated text only.
            """
            : """
            Mandatory translation rules:
            - Translate to \(targetLanguage.instructionName).
            \(targetLanguage.translationScriptConstraint.map { "- \($0)" } ?? "")
            - Translate short linguistic text too.
            - Return translated text only.
            """

        let normalizedEnforcement = enforcement
            .components(separatedBy: .newlines)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .joined(separator: "\n")

        return "\(basePrompt)\n\(normalizedEnforcement)"
    }
}

struct RewritePromptBuilder {
    static func build(
        systemPrompt: String,
        dictatedPrompt: String,
        sourceText: String,
        conversationHistory: [RewriteConversationPromptTurn] = [],
        structuredAnswerOutput: Bool,
        directAnswerMode: Bool,
        forceNonEmptyAnswer: Bool
    ) -> String {
        let basePrompt = systemPrompt
            .replacingOccurrences(of: "{{DICTATED_PROMPT}}", with: dictatedPrompt)
            .replacingOccurrences(of: "{{SOURCE_TEXT}}", with: sourceText)

        let conversationSection = conversationHistorySection(from: conversationHistory)

        let directAnswerConstraint = directAnswerMode
            ? """
            Direct-answer mode:
            - No source text is selected.
            - Treat the spoken instruction itself as the full request; it does not need a separate rewrite target.
            - Answer or perform the request directly instead of restating it or asking the user to confirm details they already supplied.
            - A place, subject, date, or other qualifier explicitly present in the request is not missing context.
            - Ask a clarification question only when an essential detail is genuinely absent and cannot be inferred from the request, selected source text, or conversation.
            - If the request needs live information but live lookup is unavailable, state that limitation directly and briefly; do not pretend that an already supplied place or date is missing.
            """
            : ""
        let conversationConstraint: String
        if conversationHistory.isEmpty {
            conversationConstraint = ""
        } else if structuredAnswerOutput {
            conversationConstraint = """
            Conversation mode:
            - Use the previous conversation as the only context.
            - Treat the spoken instruction as a follow-up to the latest assistant answer.
            - Treat a short confirmation or correction as the user's answer to the latest assistant question, then continue the task.
            - Do not repeat the latest assistant answer or ask the same clarification again.
            """
        } else {
            conversationConstraint = """
            Conversation mode:
            - Use the previous conversation as the only context.
            - Treat the spoken instruction as a follow-up to the latest assistant answer.
            - Treat a short confirmation or correction as the user's answer to the latest assistant question, then continue the task.
            - Do not repeat the latest assistant answer or ask the same clarification again.
            - Return the next assistant reply as plain text only.
            - Do not return JSON, markdown fences, labels, or quotes.
            """
        }
        let runtimeConstraint = structuredAnswerOutput
            ? """
            Runtime output format rules:
            - Return exactly one JSON object with keys "title" and "content".
            - "title" must be a short one-line summary.
            - "content" must contain only the final answer text.
            - "content" must not be empty.
            - Do not wrap the JSON in markdown fences or add extra keys.
            """
            : """
            Runtime output format rules:
            - Return plain text only.
            - Return only the final answer or rewrite content.
            - Do not return JSON, labels, markdown fences, or quotes.
            - Do not leave the answer empty.
            """
        let retryConstraint = forceNonEmptyAnswer
            ? (structuredAnswerOutput
                ? """
                Retry rule:
                - A previous answer returned an empty or unusable "content".
                - This time, you must return a non-empty "content".
                - Do not repeat the latest assistant reply or ask its clarification again.
                - If the instruction is ambiguous, return the most helpful direct answer instead of leaving "content" empty.
                """
                : """
                Retry rule:
                - A previous answer was empty or unusable.
                - This time, you must return a non-empty plain-text answer.
                - Do not repeat the latest assistant reply or ask its clarification again; use the user's latest response to advance the task.
                - Do not return JSON, labels, or quotes.
                - If the instruction is ambiguous, return the most helpful direct answer instead of nothing.
                """)
            : ""

        let extraConstraints = [directAnswerConstraint, conversationConstraint, runtimeConstraint, retryConstraint]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")

        let promptSections = [basePrompt, conversationSection]
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let promptWithHistory = promptSections.joined(separator: "\n\n")

        return extraConstraints.isEmpty ? promptWithHistory : "\(promptWithHistory)\n\n\(extraConstraints)"
    }

    private static func conversationHistorySection(from turns: [RewriteConversationPromptTurn]) -> String {
        let segments = turns.compactMap { turn -> String? in
            let userPrompt = turn.modelUserMessage
            let resultContent = turn.resultContent.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !userPrompt.isEmpty || !resultContent.isEmpty else {
                return nil
            }

            var lines: [String] = []
            if !userPrompt.isEmpty {
                lines.append("User: \(userPrompt)")
            }
            if !resultContent.isEmpty {
                lines.append("Assistant: \(resultContent)")
            }
            return lines.joined(separator: "\n")
        }

        guard !segments.isEmpty else { return "" }
        return """
        Previous conversation:
        \(segments.joined(separator: "\n\n"))
        """
    }
}

enum RewriteDirectAnswerRuntimeGuidance {
    static func content(
        now: Date = Date(),
        timeZone: TimeZone = .current,
        liveInformationAccess: Bool
    ) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "yyyy-MM-dd HH:mm"

        let liveAccessDescription = liveInformationAccess
            ? "available through the configured provider"
            : "unavailable for this request"
        return """
        - Current local date and time: \(formatter.string(from: now)).
        - Current time zone: \(timeZone.identifier).
        - Live information lookup: \(liveAccessDescription).
        - Resolve relative dates such as “today”, “tomorrow”, and “今天” from the date and time above.
        - When live lookup is unavailable, never imply that current facts were verified. State the limitation once and still provide any useful non-live guidance you can.
        """
    }
}
