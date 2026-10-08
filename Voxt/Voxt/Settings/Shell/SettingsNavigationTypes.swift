// SettingsNavigationTypes.swift
// Provides Settings Navigation Types for settings shell.

import SwiftUI

enum SettingsNavigationSection: String, Hashable {
    case generalAudio
    case generalTranscriptionUI
    case generalLanguages
    case generalLogging
    case generalAppBehavior
    case generalAdvanced
    case generalProxy
    case modelEngine
    case modelTextEnhancement
    case modelTranslation
    case modelContentRewrite
    case modelTranscriptionTest
    case dictionarySettings
    case dictionaryEntries
    case appBranchSources
    case appBranchGroups
    case historySettings
    case historyEntries
    case permissionsMain
    case permissionsAppBranchURLAuthorization
    case aboutVoxt
    case aboutProject
    case aboutAuthor
    case aboutThanks
    case aboutLogs

    var tab: SettingsTab {
        switch self {
        case .generalAudio,
             .generalTranscriptionUI,
             .generalLanguages,
             .generalLogging,
             .generalAppBehavior,
             .generalAdvanced,
             .generalProxy:
            return .general
        case .modelEngine,
             .modelTextEnhancement,
             .modelTranslation,
             .modelContentRewrite,
             .modelTranscriptionTest:
            return .model
        case .dictionarySettings,
             .dictionaryEntries:
            return .dictionary
        case .appBranchSources,
             .appBranchGroups:
            return .feature
        case .historySettings,
             .historyEntries:
            return .history
        case .permissionsMain,
             .permissionsAppBranchURLAuthorization:
            return .permissions
        case .aboutVoxt,
             .aboutProject,
             .aboutAuthor,
             .aboutThanks,
             .aboutLogs:
            return .about
        }
    }

    var titleKey: String {
        switch self {
        case .generalAudio: return "Audio"
        case .generalTranscriptionUI: return "Floating Window Style"
        case .generalLanguages: return "Languages"
        case .generalLogging: return "Logging"
        case .generalAppBehavior: return "App Behavior"
        case .generalAdvanced: return "Advanced"
        case .generalProxy: return "Proxy"
        case .modelEngine: return "Engine"
        case .modelTextEnhancement: return "Text Enhancement"
        case .modelTranslation: return "Translation"
        case .modelContentRewrite: return "Content Rewrite"
        case .modelTranscriptionTest: return "Transcription Test"
        case .dictionarySettings: return "Settings"
        case .dictionaryEntries: return "Dictionary Entries"
        case .appBranchSources: return "Sources"
        case .appBranchGroups: return "Groups"
        case .historySettings: return "History Settings"
        case .historyEntries: return "History Entries"
        case .permissionsMain: return "Permissions"
        case .permissionsAppBranchURLAuthorization: return "App Branch URL Authorization"
        case .aboutVoxt: return "Voxt"
        case .aboutProject: return "Project"
        case .aboutAuthor: return "Author"
        case .aboutThanks: return "Thanks"
        case .aboutLogs: return "Logs"
        }
    }

    var title: String {
        AppLocalization.localizedString(titleKey)
    }
}

struct SettingsNavigationTarget: Hashable {
    let tab: SettingsTab
    let section: SettingsNavigationSection?
    let featureTab: FeatureSettingsTab?
    let historyFilter: HistoryFilterTab?
    let modelSelectionID: FeatureModelSelectionID?
    let requestsModelStorageAuthorization: Bool

    init(
        tab: SettingsTab,
        section: SettingsNavigationSection? = nil,
        featureTab: FeatureSettingsTab? = nil,
        historyFilter: HistoryFilterTab? = nil,
        modelSelectionID: FeatureModelSelectionID? = nil,
        requestsModelStorageAuthorization: Bool = false
    ) {
        self.tab = tab
        self.section = section
        self.featureTab = featureTab ?? Self.defaultFeatureTab(for: tab, section: section)
        self.historyFilter = historyFilter
        self.modelSelectionID = modelSelectionID
        self.requestsModelStorageAuthorization = requestsModelStorageAuthorization
    }

    init?(notification: Notification) {
        guard let rawTab = notification.userInfo?["tab"] as? String,
              let tab = SettingsTab(rawValue: rawTab)
        else {
            return nil
        }

        let section: SettingsNavigationSection?
        if let rawSection = notification.userInfo?["section"] as? String,
           !rawSection.isEmpty {
            section = SettingsNavigationSection(rawValue: rawSection)
        } else {
            section = nil
        }

        let featureTab: FeatureSettingsTab?
        if let rawFeatureTab = notification.userInfo?["featureTab"] as? String,
           !rawFeatureTab.isEmpty {
            featureTab = FeatureSettingsTab(rawValue: rawFeatureTab)
        } else {
            featureTab = Self.defaultFeatureTab(for: tab, section: section)
        }

        let historyFilter: HistoryFilterTab?
        if let rawHistoryFilter = notification.userInfo?["historyFilter"] as? String,
           !rawHistoryFilter.isEmpty {
            historyFilter = HistoryFilterTab(rawValue: rawHistoryFilter)
        } else {
            historyFilter = nil
        }

        let modelSelectionID: FeatureModelSelectionID?
        if let rawModelSelectionID = notification.userInfo?["modelSelectionID"] as? String,
           !rawModelSelectionID.isEmpty {
            modelSelectionID = FeatureModelSelectionID(rawValue: rawModelSelectionID)
        } else {
            modelSelectionID = nil
        }

        let requestsModelStorageAuthorization =
            notification.userInfo?["requestsModelStorageAuthorization"] as? String == "true"

        self.init(
            tab: tab == .appEnhancement ? .feature : tab,
            section: section,
            featureTab: featureTab,
            historyFilter: historyFilter,
            modelSelectionID: modelSelectionID,
            requestsModelStorageAuthorization: requestsModelStorageAuthorization
        )
    }

    var userInfo: [String: String] {
        [
            "tab": tab.rawValue,
            "section": section?.rawValue ?? "",
            "featureTab": featureTab?.rawValue ?? "",
            "historyFilter": historyFilter?.rawValue ?? "",
            "modelSelectionID": modelSelectionID?.rawValue ?? "",
            "requestsModelStorageAuthorization": requestsModelStorageAuthorization ? "true" : "false"
        ]
    }

    static func defaultFeatureTab(
        for tab: SettingsTab,
        section: SettingsNavigationSection?
    ) -> FeatureSettingsTab? {
        if tab == .appEnhancement {
            return .appEnhancement
        }
        guard tab == .feature || section?.tab == .feature else { return nil }
        switch section {
        case .appBranchSources, .appBranchGroups:
            return .appEnhancement
        case .none:
            return .features
        default:
            return .features
        }
    }
}

struct SettingsNavigationRequest: Identifiable, Equatable {
    let id: UUID
    let target: SettingsNavigationTarget

    init(id: UUID = UUID(), target: SettingsNavigationTarget) {
        self.id = id
        self.target = target
    }
}

extension View {
    func settingsNavigationAnchor(_ section: SettingsNavigationSection) -> some View {
        id(section.rawValue)
    }
}

enum SettingsTab: String, CaseIterable, Identifiable {
    case report
    case general
    case model
    case feature
    case dictionary
    case appEnhancement
    case history
    case permissions
    case hotkey
    case about

    var id: String { rawValue }

    var titleKey: LocalizedStringKey {
        switch self {
        case .general: return "General"
        case .permissions: return "Permissions"
        case .history: return "History"
        case .report: return "Home"
        case .model: return "Model"
        case .feature: return "Feature"
        case .dictionary: return "Dictionary"
        case .appEnhancement: return "App Branch"
        case .hotkey: return "Hotkey"
        case .about: return "About"
        }
    }

    var title: String { AppLocalization.localizedString(rawTitleKey) }

    private var rawTitleKey: String {
        switch self {
        case .general: return "General"
        case .permissions: return "Permissions"
        case .history: return "History"
        case .report: return "Home"
        case .model: return "Model"
        case .feature: return "Feature"
        case .dictionary: return "Dictionary"
        case .appEnhancement: return "App Branch"
        case .hotkey: return "Hotkey"
        case .about: return "About"
        }
    }

    var iconName: String {
        switch self {
        case .general: return "slider.horizontal.3"
        case .permissions: return "lock.shield"
        case .history: return "clock.arrow.circlepath"
        case .report: return "house"
        case .model: return "waveform"
        case .feature: return "square.grid.2x2"
        case .dictionary: return "book.closed"
        case .appEnhancement: return "sparkles.rectangle.stack"
        case .hotkey: return "keyboard"
        case .about: return "info.circle"
        }
    }

    static func visibleTabs(appEnhancementEnabled: Bool) -> [SettingsTab] {
        [
            .report,
            .feature,
            .dictionary,
            .history
        ]
    }

    static var settingsTabs: [SettingsTab] {
        [
            .general,
            .model,
            .permissions,
            .hotkey,
            .about
        ]
    }
}

enum SettingsSidebarMode: Equatable {
    case root
    case feature
    case history
    case settings
}

enum FeatureSettingsTab: String, CaseIterable, Identifiable {
    case features
    case transcription
    case translation
    case rewrite
    case appEnhancement
    case note
    case meeting
    case files

    var id: String { rawValue }

    var titleKey: LocalizedStringKey {
        switch self {
        case .features: return "Feature"
        case .transcription: return "Transcription"
        case .meeting: return "Meeting"
        case .files: return "Files"
        case .note: return "Notes"
        case .translation: return "Translation"
        case .rewrite: return "Rewrite"
        case .appEnhancement: return "App Enhancement"
        }
    }

    var title: String {
        AppLocalization.localizedString(rawTitleKey)
    }

    private var rawTitleKey: String {
        switch self {
        case .features: return "Feature"
        case .transcription: return "Transcription"
        case .meeting: return "Meeting"
        case .files: return "Files"
        case .note: return "Notes"
        case .translation: return "Translation"
        case .rewrite: return "Rewrite"
        case .appEnhancement: return "App Enhancement"
        }
    }

    var iconName: String {
        switch self {
        case .features: return "switch.2"
        case .transcription: return "waveform.and.mic"
        case .meeting: return "person.2.wave.2"
        case .files: return "doc.on.doc"
        case .note: return "note.text"
        case .translation: return "globe"
        case .rewrite: return "text.badge.star"
        case .appEnhancement: return "sparkles.rectangle.stack"
        }
    }

    func isEnabled(in availability: FeatureAvailabilitySettings) -> Bool {
        switch self {
        case .features, .transcription:
            return true
        case .translation:
            return availability.translationEnabled
        case .rewrite:
            return availability.rewriteEnabled
        case .note:
            return availability.notesEnabled
        case .appEnhancement:
            return availability.appEnhancementEnabled
        case .meeting:
            return availability.meetingEnabled
        case .files:
            return availability.filesEnabled
        }
    }

    static func visibleTabs(
        availability: FeatureAvailabilitySettings,
        appEnhancementEnabled: Bool? = nil,
        noteEnabled: Bool? = nil,
        translationEnabled: Bool? = nil,
        rewriteEnabled: Bool? = nil,
        meetingEnabled: Bool? = nil,
        filesEnabled: Bool? = nil
    ) -> [FeatureSettingsTab] {
        let resolved = FeatureAvailabilitySettings(
            translationEnabled: translationEnabled ?? availability.translationEnabled,
            rewriteEnabled: rewriteEnabled ?? availability.rewriteEnabled,
            notesEnabled: noteEnabled ?? availability.notesEnabled,
            appEnhancementEnabled: appEnhancementEnabled ?? availability.appEnhancementEnabled,
            meetingEnabled: meetingEnabled ?? availability.meetingEnabled,
            filesEnabled: filesEnabled ?? availability.filesEnabled
        )
        return allCases.filter { $0.isEnabled(in: resolved) }
    }

    static func visibleTabs(appEnhancementEnabled: Bool, noteEnabled: Bool) -> [FeatureSettingsTab] {
        visibleTabs(
            availability: .allEnabled,
            appEnhancementEnabled: appEnhancementEnabled,
            noteEnabled: noteEnabled
        )
    }
}

struct SettingsSectionHeader: View {
    let title: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.title2.weight(.semibold))
            Divider()
        }
    }
}
