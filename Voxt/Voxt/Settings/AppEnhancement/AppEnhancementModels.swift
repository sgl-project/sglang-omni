// AppEnhancementModels.swift
// Provides App Enhancement Models for app enhancement settings.

import Foundation
import AppKit
import Carbon

enum SourceTab: String, CaseIterable, Identifiable {
    case apps
    case urls

    var id: String { rawValue }

    var title: String {
        switch self {
        case .apps:
            return AppLocalization.localizedString("Apps")
        case .urls:
            return AppLocalization.localizedString("URLs")
        }
    }
}

struct BranchApp: Identifiable {
    let id: String
    let name: String
    let icon: NSImage
}

struct BranchURLItem: Identifiable, Codable, Equatable {
    let id: UUID
    var pattern: String
}

struct AppBranchAppRef: Codable, Equatable {
    let bundleID: String
    var displayName: String
}

struct AppBranchGroup: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    var prompt: String
    var appBundleIDs: [String]
    var appRefs: [AppBranchAppRef]
    var urlPatternIDs: [UUID]
    var autoKeyPressEnabled: Bool
    var autoKeyPressHotkey: HotkeyPreference.Hotkey
    var isExpanded: Bool

    static let defaultAutoKeyPressHotkey = HotkeyPreference.Hotkey(
        keyCode: UInt16(kVK_Return),
        modifiers: [],
        sidedModifiers: []
    )

    var autoPressReturnEnabled: Bool {
        autoKeyPressEnabled && autoKeyPressHotkey == Self.defaultAutoKeyPressHotkey
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case prompt
        case appBundleIDs
        case appRefs
        case urlPatternIDs
        case autoPressReturnEnabled
        case autoKeyPressEnabled
        case autoKeyPressHotkey
        case isExpanded
    }

    init(
        id: UUID,
        name: String,
        prompt: String,
        appBundleIDs: [String],
        appRefs: [AppBranchAppRef],
        urlPatternIDs: [UUID],
        autoPressReturnEnabled: Bool = false,
        autoKeyPressEnabled: Bool? = nil,
        autoKeyPressHotkey: HotkeyPreference.Hotkey = AppBranchGroup.defaultAutoKeyPressHotkey,
        isExpanded: Bool
    ) {
        self.id = id
        self.name = name
        self.prompt = prompt
        self.appBundleIDs = appBundleIDs
        self.appRefs = appRefs
        self.urlPatternIDs = urlPatternIDs
        self.autoKeyPressEnabled = autoKeyPressEnabled ?? autoPressReturnEnabled
        self.autoKeyPressHotkey = autoKeyPressHotkey
        self.isExpanded = isExpanded
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        prompt = try container.decode(String.self, forKey: .prompt)
        appBundleIDs = try container.decodeIfPresent([String].self, forKey: .appBundleIDs) ?? []
        let decodedRefs = try container.decodeIfPresent([AppBranchAppRef].self, forKey: .appRefs) ?? []
        if decodedRefs.isEmpty {
            appRefs = appBundleIDs.map { AppBranchAppRef(bundleID: $0, displayName: $0) }
        } else {
            appRefs = decodedRefs
        }
        urlPatternIDs = try container.decodeIfPresent([UUID].self, forKey: .urlPatternIDs) ?? []
        let legacyAutoReturnEnabled = try container.decodeIfPresent(Bool.self, forKey: .autoPressReturnEnabled) ?? false
        autoKeyPressEnabled = try container.decodeIfPresent(Bool.self, forKey: .autoKeyPressEnabled) ?? legacyAutoReturnEnabled
        autoKeyPressHotkey = try container.decodeIfPresent(
            HotkeyPreference.Hotkey.self,
            forKey: .autoKeyPressHotkey
        ) ?? Self.defaultAutoKeyPressHotkey
        isExpanded = try container.decodeIfPresent(Bool.self, forKey: .isExpanded) ?? true
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encode(prompt, forKey: .prompt)
        try container.encode(appBundleIDs, forKey: .appBundleIDs)
        try container.encode(appRefs, forKey: .appRefs)
        try container.encode(urlPatternIDs, forKey: .urlPatternIDs)
        try container.encode(autoPressReturnEnabled, forKey: .autoPressReturnEnabled)
        try container.encode(autoKeyPressEnabled, forKey: .autoKeyPressEnabled)
        try container.encode(autoKeyPressHotkey, forKey: .autoKeyPressHotkey)
        try container.encode(isExpanded, forKey: .isExpanded)
    }
}

struct GroupMember: Identifiable {
    let content: GroupMemberContent

    var id: String {
        switch content {
        case .app(let appMember):
            return "app:\(appMember.app.id)"
        case .url(let item):
            return "url:\(item.id.uuidString)"
        }
    }
}

struct GroupAppMember {
    let app: BranchApp
    let isRunning: Bool
}

enum GroupMemberContent {
    case app(GroupAppMember)
    case url(BranchURLItem)
}

enum AppBranchModal: Identifiable {
    case createGroup
    case editGroup(UUID)
    case addURLs
    case editURL(UUID)
    case urlDetail(BranchURLItem)

    var id: String {
        switch self {
        case .createGroup:
            return "create-group"
        case .editGroup(let groupID):
            return "edit-group-\(groupID.uuidString)"
        case .addURLs:
            return "add-urls"
        case .editURL(let urlID):
            return "edit-url-\(urlID.uuidString)"
        case .urlDetail(let item):
            return "url-detail-\(item.id.uuidString)"
        }
    }

    var title: String {
        switch self {
        case .createGroup:
            return AppLocalization.localizedString("Create Group")
        case .editGroup:
            return AppLocalization.localizedString("Edit Group")
        case .addURLs:
            return AppLocalization.localizedString("Add URL Patterns")
        case .editURL:
            return AppLocalization.localizedString("Edit URL Pattern")
        case .urlDetail:
            return AppLocalization.localizedString("URL Detail")
        }
    }

    var actionTitle: String {
        switch self {
        case .createGroup:
            return AppLocalization.localizedString("Create")
        case .editGroup:
            return AppLocalization.localizedString("Save")
        case .addURLs:
            return AppLocalization.localizedString("Add")
        case .editURL:
            return AppLocalization.localizedString("Save")
        case .urlDetail:
            return ""
        }
    }
}
