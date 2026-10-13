// HistoryConversationSupport.swift
// Provides History Conversation Support for history storage and display support.

import Foundation

enum TranscriptionHistoryConversationSupport {
    static let continuationTimeout: TimeInterval = 10 * 60

    static func supportsDetail(for kind: TranscriptionHistoryKind) -> Bool {
        kind == .rewrite
    }

    static func shouldContinueConversation(
        activeEntryID: UUID?,
        lastUpdatedAt: Date?,
        now: Date = Date()
    ) -> Bool {
        guard activeEntryID != nil, let lastUpdatedAt else { return false }
        return now.timeIntervalSince(lastUpdatedAt) <= continuationTimeout
    }

    static func mergedTranscriptText(existing: String, incoming: String) -> String {
        let trimmedExisting = existing.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedIncoming = incoming.trimmingCharacters(in: .whitespacesAndNewlines)

        if trimmedExisting.isEmpty {
            return trimmedIncoming
        }
        if trimmedIncoming.isEmpty {
            return trimmedExisting
        }
        return "\(trimmedExisting)\n\(trimmedIncoming)"
    }

    static func mergedChatMessages(
        for entry: TranscriptionHistoryEntry,
        appendingTranscript text: String,
        createdAt: Date = Date()
    ) -> [TranscriptSummaryChatMessage] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return bootstrapChatMessages(for: entry)
        }

        var messages = bootstrapChatMessages(for: entry)
        messages.append(
            TranscriptSummaryChatMessage(
                role: .assistant,
                content: trimmed,
                createdAt: createdAt
            )
        )
        return messages
    }

    static func initialChatMessages(
        forTranscript text: String,
        createdAt: Date = Date()
    ) -> [TranscriptSummaryChatMessage]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return [
            TranscriptSummaryChatMessage(
                role: .assistant,
                content: trimmed,
                createdAt: createdAt
            )
        ]
    }

    static func mergedTerms(existing: [String], incoming: [String]) -> [String] {
        var seen = Set<String>()
        var ordered: [String] = []

        for value in existing + incoming {
            let normalized = DictionaryStore.normalizeTerm(value)
            guard !normalized.isEmpty, seen.insert(normalized).inserted else { continue }
            ordered.append(value)
        }

        return ordered
    }

    static func accumulatedDuration(existing: TimeInterval?, incoming: TimeInterval?) -> TimeInterval? {
        switch (existing, incoming) {
        case (.some(let existing), .some(let incoming)):
            return existing + incoming
        case (.some(let existing), .none):
            return existing
        case (.none, .some(let incoming)):
            return incoming
        case (.none, .none):
            return nil
        }
    }

    static func bootstrapChatMessages(for entry: TranscriptionHistoryEntry) -> [TranscriptSummaryChatMessage] {
        if let existingMessages = entry.transcriptionChatMessages, !existingMessages.isEmpty {
            if isCompleteConversation(existingMessages) {
                return existingMessages
            }

            var bootstrapped = [seedMessage(for: entry)]
            bootstrapped.append(contentsOf: existingMessages)
            return bootstrapped
        }

        return [seedMessage(for: entry)]
    }

    static func isCompleteConversation(_ messages: [TranscriptSummaryChatMessage]) -> Bool {
        guard let firstRole = messages.first?.role else { return false }
        if firstRole == .assistant {
            return true
        }
        return firstRole == .user && messages.dropFirst().contains { $0.role == .assistant }
    }

    static func seedMessage(for entry: TranscriptionHistoryEntry) -> TranscriptSummaryChatMessage {
        TranscriptSummaryChatMessage(
            id: entry.id,
            role: .assistant,
            content: entry.text,
            createdAt: entry.createdAt
        )
    }

    static func rewriteConversationMessages(
        from turns: [RewriteConversationTurn],
        createdAt: Date = Date()
    ) -> [TranscriptSummaryChatMessage] {
        var messages: [TranscriptSummaryChatMessage] = []

        for turn in turns {
            let userPrompt = turn.promptTurn.modelUserMessage
            if !userPrompt.isEmpty {
                messages.append(
                    TranscriptSummaryChatMessage(
                        role: .user,
                        content: userPrompt,
                        createdAt: createdAt
                    )
                )
            }

            let assistantContent = turn.resultContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if !assistantContent.isEmpty {
                messages.append(
                    TranscriptSummaryChatMessage(
                        role: .assistant,
                        content: assistantContent,
                        createdAt: createdAt
                    )
                )
            }
        }

        return messages
    }
}
