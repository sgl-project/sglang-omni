// Rewrite answer and conversation values shared by prompts, overlays, and history.

import Foundation

struct RewriteAnswerPayload: Equatable {
    let title: String
    let content: String

    nonisolated var trimmedTitle: String {
        title.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated var trimmedContent: String {
        content.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

struct RewriteConversationTurn: Identifiable, Equatable {
    let id: UUID
    let userPromptText: String
    let sourceText: String
    let resultTitle: String
    let resultContent: String

    init(
        id: UUID = UUID(),
        userPromptText: String,
        sourceText: String = "",
        resultTitle: String,
        resultContent: String
    ) {
        self.id = id
        self.userPromptText = userPromptText
        self.sourceText = sourceText
        self.resultTitle = resultTitle
        self.resultContent = resultContent
    }

    static func seed(from payload: RewriteAnswerPayload) -> RewriteConversationTurn {
        RewriteConversationTurn(
            userPromptText: "",
            sourceText: "",
            resultTitle: payload.title,
            resultContent: payload.content
        )
    }

    var promptTurn: RewriteConversationPromptTurn {
        RewriteConversationPromptTurn(
            userPromptText: userPromptText,
            sourceText: sourceText,
            resultTitle: resultTitle,
            resultContent: resultContent
        )
    }
}

struct RewriteConversationPromptTurn: Equatable {
    let userPromptText: String
    let sourceText: String
    let resultTitle: String
    let resultContent: String

    init(
        userPromptText: String,
        sourceText: String = "",
        resultTitle: String,
        resultContent: String
    ) {
        self.userPromptText = userPromptText
        self.sourceText = sourceText
        self.resultTitle = resultTitle
        self.resultContent = resultContent
    }

    var modelUserMessage: String {
        let prompt = userPromptText.trimmingCharacters(in: .whitespacesAndNewlines)
        let source = sourceText.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !source.isEmpty else { return prompt }
        guard !prompt.isEmpty else {
            return "Selected source text:\n\(source)"
        }
        return """
        Spoken instruction:
        \(prompt)

        Selected source text:
        \(source)
        """
    }
}

enum RewriteAnswerContentNormalizer {
    static func normalizePlainTextStreamingPreview(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        let openingQuotes: Set<Character> = ["\"", "'", "“"]
        let closingQuotes: Set<Character> = ["\"", "'", "”"]

        if looksStructuredAnswerCandidate(trimmed),
           let payload = RewriteAnswerPayloadParser.extract(from: trimmed) {
            let content = payload.trimmedContent
            if !content.isEmpty {
                return content
            }
        }

        var normalized = trimmed

        if ["\"", "'", "“", "”", "\"\"", "''", "{}", "[]"].contains(normalized) {
            return ""
        }

        if let first = normalized.first,
           openingQuotes.contains(first) {
            normalized.removeFirst()
        }

        if let last = normalized.last,
           closingQuotes.contains(last) {
            normalized.removeLast()
        }

        return normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func normalizePlainTextAnswer(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        if looksStructuredAnswerCandidate(trimmed),
           let payload = RewriteAnswerPayloadParser.extract(from: trimmed) {
            let content = payload.trimmedContent
            if !content.isEmpty {
                return content
            }
        }

        var normalized = trimmed
        if normalized.count >= 2 {
            let left = normalized.first
            let right = normalized.last
            let isWrappedByQuotes =
                (left == "\"" && right == "\"") ||
                (left == "'" && right == "'") ||
                (left == "“" && right == "”")
            if isWrappedByQuotes {
                normalized.removeFirst()
                normalized.removeLast()
                normalized = normalized.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        if ["{}", "[]", "\"\"", "''"].contains(normalized) {
            return ""
        }

        return normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func isUnusablePlainTextAnswer(_ text: String, dictatedPrompt: String) -> Bool {
        let normalized = normalizePlainTextAnswer(text)
        guard !normalized.isEmpty else { return true }

        let normalizedPrompt = dictatedPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !normalizedPrompt.isEmpty,
           normalized.caseInsensitiveCompare(normalizedPrompt) == .orderedSame {
            return true
        }

        let lowered = normalized.lowercased()
        return ["null", "nil", "none", "n/a", "na"].contains(lowered)
    }

    static func isUnusableStructuredAnswer(_ text: String, dictatedPrompt: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        if let payload = RewriteAnswerPayloadParser.extract(from: trimmed) {
            let normalizedContent = payload.content.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalizedPrompt = dictatedPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            return normalizedContent.isEmpty || normalizedContent.caseInsensitiveCompare(normalizedPrompt) == .orderedSame
        }

        let normalizedPrompt = dictatedPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.caseInsensitiveCompare(normalizedPrompt) == .orderedSame {
            return true
        }

        let lowered = trimmed.lowercased()
        return trimmed.hasPrefix("{") ||
            trimmed.hasPrefix("[") ||
            lowered.contains("\"title\"") ||
            lowered.contains("\"content\"") ||
            lowered.contains("title:") ||
            lowered.contains("content:")
    }

    static func repeatsLatestAssistantAnswer(
        _ text: String,
        dictatedPrompt: String,
        conversationHistory: [RewriteConversationPromptTurn]
    ) -> Bool {
        guard let latestAssistant = conversationHistory.last?.resultContent else { return false }
        guard !explicitlyAllowsUnchangedAnswer(dictatedPrompt) else { return false }

        let normalizedAnswer = comparisonKey(for: normalizePlainTextAnswer(text))
        let normalizedPrevious = comparisonKey(for: normalizePlainTextAnswer(latestAssistant))
        return !normalizedAnswer.isEmpty && normalizedAnswer == normalizedPrevious
    }

    private static func explicitlyAllowsUnchangedAnswer(_ prompt: String) -> Bool {
        let normalized = prompt.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let repetitionRequests = [
            "repeat", "say that again", "again", "再说一遍", "重复一遍", "重复一下", "再来一次",
            "もう一度", "繰り返して"
        ]
        if repetitionRequests.contains(where: { normalized.contains($0) }) {
            return true
        }

        let normalizedIntent = normalized.trimmingCharacters(
            in: .whitespacesAndNewlines
                .union(.punctuationCharacters)
                .union(.symbols)
        )
        let unchangedAnswerRequests: Set<String> = [
            "保持原样", "请保持原样", "保持原样就好", "保持原样即可", "保持不变", "请保持不变",
            "不用改", "不用改了", "不要修改", "无需修改", "不需要修改", "原样即可", "就这样", "这样就好",
            "keep it as is", "please keep it as is", "leave it as is", "please leave it as is",
            "leave unchanged", "no changes", "no change", "do not change it", "don't change it", "same as before",
            "そのまま", "そのままで", "そのままでいい", "変更しない", "変えないで", "修正不要"
        ]
        return unchangedAnswerRequests.contains(normalizedIntent)
    }

    private static func comparisonKey(for text: String) -> String {
        let ignored = CharacterSet.whitespacesAndNewlines
            .union(.punctuationCharacters)
            .union(.symbols)
        return text.unicodeScalars
            .filter { !ignored.contains($0) }
            .map { String($0).lowercased() }
            .joined()
    }

    private static func looksStructuredAnswerCandidate(_ text: String) -> Bool {
        let lowered = text.lowercased()
        return text.hasPrefix("{") ||
            text.hasPrefix("[") ||
            text.hasPrefix("```") ||
            lowered.contains("\"title\"") ||
            lowered.contains("\"content\"") ||
            (lowered.contains("title:") && lowered.contains("content:"))
    }
}
