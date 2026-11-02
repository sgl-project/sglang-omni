// Model debug results, clip presentation and selector adapters.

import Foundation

func modelDebugLocalized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

func modelDebugClipTimestamp(_ date: Date) -> String {
    date.formatted(.dateTime.month(.defaultDigits).day(.defaultDigits).hour().minute().second())
}

func modelDebugClipTitlePreview(_ text: String) -> String? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    let singleLine = trimmed.replacingOccurrences(of: "\n", with: " ")
    return String(singleLine.prefix(28))
}

struct ASRDebugResult: Identifiable, Equatable {
    enum Source: String {
        case liveRecording
        case reusedClip
    }

    let id: UUID
    let clipID: UUID
    let clipTitle: String
    let modelTitle: String
    let source: Source
    let audioDurationText: String
    let runtimeText: String
    let characterCount: Int
    let createdAt: Date
    let outputText: String
    let errorText: String?

    var isError: Bool { errorText != nil }
}

struct ASRDebugClipItem: Identifiable, Equatable {
    let id: UUID
    let clip: DebugAudioClip
    let defaultTitle: String
    var title: String

    var displayTitle: String { title }
}

extension ASRDebugModelOption {
    var selectorEntry: FeatureModelSelectorEntry {
        let selectionID = FeatureModelSelectionID(rawValue: id)
        let locationTag: String
        switch selection {
        case .remote:
            locationTag = modelDebugLocalized("Remote")
        case .mlx:
            locationTag = modelDebugLocalized("Local")
        }
        return FeatureModelSelectorEntry(
            selectionID: selectionID,
            title: title,
            engine: subtitle,
            sizeText: modelDebugLocalized(isSelectable ? "Ready" : "Unavailable"),
            ratingText: "—",
            filterTags: [locationTag] + (isInstalled ? [modelDebugLocalized("Installed"), modelDebugLocalized("Configured")] : []),
            displayTags: [locationTag] + (isInstalled ? [modelDebugLocalized("Installed")] : []),
            statusText: disabledReason ?? "",
            usageLocations: [],
            badgeText: nil,
            isSelectable: isSelectable,
            disabledReason: disabledReason
        )
    }
}

extension LLMDebugModelOption {
    var selectorEntry: FeatureModelSelectorEntry {
        let selectionID = FeatureModelSelectionID(rawValue: id)
        let locationTag: String
        switch selection {
        case .remote:
            locationTag = modelDebugLocalized("Remote")
        case .local:
            locationTag = modelDebugLocalized("Local")
        }
        return FeatureModelSelectorEntry(
            selectionID: selectionID,
            title: title,
            engine: subtitle,
            sizeText: modelDebugLocalized("Ready"),
            ratingText: "—",
            filterTags: [locationTag, modelDebugLocalized("Installed"), modelDebugLocalized("Configured")],
            displayTags: [locationTag, modelDebugLocalized("Installed")],
            statusText: "",
            usageLocations: [],
            badgeText: nil,
            isSelectable: true,
            disabledReason: nil
        )
    }
}

enum ModelDebugWindowStyle {
    static let width: CGFloat = 750
    static let height: CGFloat = 500
    static let minWidth: CGFloat = 650
    static let minHeight: CGFloat = 560
    static let selectorMinWidth: CGFloat = 150
    static let selectorIdealWidth: CGFloat = 220
    static let resultCardBodyHeight: CGFloat = 208
}

struct LLMDebugResult: Identifiable, Equatable {
    let id: UUID
    let modelTitle: String
    let presetTitle: String
    let inputSummary: String
    let requestMetadata: LLMDebugRequestMetadata?
    let durationText: String
    let createdAt: Date
    let outputText: String
    let errorText: String?

    var isError: Bool { errorText != nil }
}

extension ASRDebugResult.Source {
    var localizedTitle: String {
        switch self {
        case .liveRecording:
            return modelDebugLocalized("Recorded")
        case .reusedClip:
            return modelDebugLocalized("Reused Audio")
        }
    }
}
