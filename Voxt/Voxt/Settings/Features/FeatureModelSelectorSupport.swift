// FeatureModelSelectorSupport.swift
// Provides Feature Model Selector Support for feature settings.

import SwiftUI

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

enum FeatureModelSelectorSheet: String, Identifiable {
    case transcriptionASR
    case transcriptionLLM
    case transcriptionNoteTitle
    case translationASR
    case translationModel
    case rewriteASR
    case rewriteLLM
    case meetingASR
    case meetingSummary

    var id: String { rawValue }

    var title: String {
        switch self {
        case .transcriptionASR: return localized("Choose Transcription Audio Model")
        case .transcriptionLLM: return localized("Choose Transcription Enhancement Model")
        case .transcriptionNoteTitle: return localized("Choose Summary Model")
        case .translationASR: return localized("Choose Translation Audio Model")
        case .translationModel: return localized("Choose Translation Model")
        case .rewriteASR: return localized("Choose Rewrite Audio Model")
        case .rewriteLLM: return localized("Choose Rewrite Enhancement Model")
        case .meetingASR: return localized("Choose Meeting Audio Model")
        case .meetingSummary: return localized("Choose Meeting Summary Model")
        }
    }
}

struct FeatureModelSelectorEntry: Identifiable {
    let selectionID: FeatureModelSelectionID
    let title: String
    let engine: String
    let sizeText: String
    let ratingText: String
    let filterTags: [String]
    let displayTags: [String]
    let statusText: String
    let usageLocations: [String]
    let badgeText: String?
    let isSelectable: Bool
    let disabledReason: String?

    var id: String { selectionID.rawValue }
}
