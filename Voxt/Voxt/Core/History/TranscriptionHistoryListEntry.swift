// TranscriptionHistoryListEntry.swift
// Lightweight history data used by the paged history list.

import Foundation

struct TranscriptionHistoryListEntry: Identifiable, Equatable, Hashable, Sendable {
    static let previewCharacterLimit = 320

    let id: UUID
    let previewText: String
    let textLength: Int
    let createdAt: Date
    let kind: TranscriptionHistoryKind
    let displayTitle: String?

    var displayText: String {
        let trimmedPreview = previewText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPreview.isEmpty else {
            return displayTitle ?? localizedFallbackTitle
        }

        if textLength > trimmedPreview.count {
            return trimmedPreview + "…"
        }
        return trimmedPreview
    }

    private var localizedFallbackTitle: String {
        AppLocalization.localizedString("Recording")
    }
}
