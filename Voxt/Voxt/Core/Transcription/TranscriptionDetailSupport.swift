// TranscriptionDetailSupport.swift
// Provides Transcription Detail Support for transcription processing.

import Foundation

struct TranscriptionFollowUpProviderStatus: Equatable, Sendable {
    let isAvailable: Bool
    let message: String
}

enum TranscriptionDetailSupport {
    static func title(for kind: TranscriptionHistoryKind) -> String {
        switch kind {
        case .normal:
            return AppLocalization.localizedString("Transcription Details")
        case .translation:
            return AppLocalization.localizedString("Translation Details")
        case .rewrite:
            return AppLocalization.localizedString("Rewrite Details")
        case .transcript:
            return AppLocalization.localizedString("Transcript Details")
        }
    }

    static func followUpPrompt(
        entry: TranscriptionHistoryEntry,
        history: [TranscriptSummaryChatMessage],
        question: String,
        userMainLanguage: String
    ) -> String {
        let trimmedHistory = history
            .map { message in
                let roleLabel = message.role == .user ? "User" : "Assistant"
                return "\(roleLabel): \(message.content)"
            }
            .joined(separator: "\n")

        return AppPromptResourceStore.requiredText(for: .savedTranscriptionFollowUp)
            .replacingOccurrences(of: "{{USER_MAIN_LANGUAGE}}", with: userMainLanguage)
            .replacingOccurrences(of: "{{RESULT_TYPE}}", with: entry.kind.rawValue)
            .replacingOccurrences(
                of: "{{RESULT_CREATED_AT}}",
                with: entry.createdAt.formatted(date: .abbreviated, time: .shortened)
            )
            .replacingOccurrences(of: "{{SAVED_RESULT}}", with: entry.text)
            .replacingOccurrences(
                of: "{{PREVIOUS_CHAT}}",
                with: trimmedHistory.isEmpty ? "None" : trimmedHistory
            )
            .replacingOccurrences(of: "{{QUESTION}}", with: question)
    }
}
