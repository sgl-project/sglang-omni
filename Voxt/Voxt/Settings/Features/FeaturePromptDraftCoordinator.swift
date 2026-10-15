// FeaturePromptDraftCoordinator.swift
// Provides Feature Prompt Draft Coordinator for feature settings.

import Foundation

struct FeaturePromptDraftCoordinator: Equatable {
    private(set) var draft: String
    private(set) var lastSyncedText: String

    init(text: String) {
        draft = text
        lastSyncedText = text
    }

    mutating func updateDraft(_ newValue: String) {
        draft = newValue
    }

    mutating func replaceDraft(with newValue: String) {
        draft = newValue
        lastSyncedText = newValue
    }

    mutating func syncExternalText(_ newValue: String) {
        // Ignore the round-trip echo of our own write so the active TextEditor
        // does not receive a redundant string assignment mid-typing.
        guard newValue != lastSyncedText, newValue != draft else { return }
        draft = newValue
        lastSyncedText = newValue
    }

    mutating func takePendingPersist(expectedText: String? = nil) -> String? {
        guard draft != lastSyncedText else { return nil }
        if let expectedText, draft != expectedText {
            return nil
        }

        lastSyncedText = draft
        return draft
    }
}
