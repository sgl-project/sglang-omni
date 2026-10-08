// AppEnhancementData.swift
// Provides App Enhancement Data for app enhancement settings.

import SwiftUI
import AppKit
import UniformTypeIdentifiers

extension AppEnhancementSettingsView {
    func handleOnAppear() {
        loadPersistedGroups()
        loadPersistedURLs()
        refreshApps()
    }

    func refreshApps() {
        let runningApps = NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular }

        var uniqueByBundleID: [String: BranchApp] = [:]
        for running in runningApps {
            guard let bundleID = running.bundleIdentifier, !bundleID.isEmpty else { continue }
            let icon = running.icon ?? NSWorkspace.shared.icon(forFile: running.bundleURL?.path ?? "")
            let name = running.localizedName ?? bundleID
            uniqueByBundleID[bundleID] = BranchApp(id: bundleID, name: name, icon: icon)
        }

        apps = uniqueByBundleID.values.sorted {
            $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
        }
    }

    func appRefForBundleID(_ bundleID: String) -> AppBranchAppRef {
        if let running = apps.first(where: { $0.id == bundleID }) {
            return AppBranchAppRef(bundleID: bundleID, displayName: running.name)
        }
        if let app = appFromSystem(bundleID: bundleID, fallbackName: bundleID) {
            return AppBranchAppRef(bundleID: bundleID, displayName: app.name)
        }
        return AppBranchAppRef(bundleID: bundleID, displayName: bundleID)
    }

    func appFromSystem(bundleID: String, fallbackName: String) -> BranchApp? {
        guard let appURL = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) else {
            return nil
        }
        let displayName = Bundle(url: appURL)?.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String
        let bundleName = Bundle(url: appURL)?.object(forInfoDictionaryKey: kCFBundleNameKey as String) as? String
        let resolvedName = displayName ?? bundleName ?? appURL.deletingPathExtension().lastPathComponent
        let icon = NSWorkspace.shared.icon(forFile: appURL.path)
        return BranchApp(
            id: bundleID,
            name: resolvedName.isEmpty ? fallbackName : resolvedName,
            icon: icon
        )
    }

    func defaultAppIcon() -> NSImage {
        NSWorkspace.shared.icon(for: .application)
    }

    func groupForApp(bundleID: String) -> AppBranchGroup? {
        groups.first { $0.appBundleIDs.contains(bundleID) }
    }

    func groupForURL(id: UUID) -> AppBranchGroup? {
        groups.first { $0.urlPatternIDs.contains(id) }
    }

    func groupMembers(group: AppBranchGroup) -> [GroupMember] {
        let runningByBundleID = Dictionary(uniqueKeysWithValues: apps.map { ($0.id, $0) })
        let urlSet = Set(group.urlPatternIDs)

        let appMembers = group.appRefs.map { ref in
            let runningApp = runningByBundleID[ref.bundleID]
            let resolvedApp = runningApp ?? appFromSystem(bundleID: ref.bundleID, fallbackName: ref.displayName)
                ?? BranchApp(
                    id: ref.bundleID,
                    name: ref.displayName.isEmpty ? ref.bundleID : ref.displayName,
                    icon: defaultAppIcon()
                )
            return GroupMember(content: .app(GroupAppMember(app: resolvedApp, isRunning: runningApp != nil)))
        }
        let urlMembers = urlItems
            .filter { urlSet.contains($0.id) }
            .map { GroupMember(content: .url($0)) }
        return appMembers + urlMembers
    }

    func setGroupExpanded(groupID: UUID, expanded: Bool) {
        guard let index = groups.firstIndex(where: { $0.id == groupID }) else { return }
        groups[index].isExpanded = expanded
    }

    func setAllGroupsExpanded(_ expanded: Bool) {
        for index in groups.indices {
            groups[index].isExpanded = expanded
        }
    }

    func assignApp(bundleID: String, to groupID: UUID) {
        for index in groups.indices {
            groups[index].appBundleIDs.removeAll { $0 == bundleID }
            groups[index].appRefs.removeAll { $0.bundleID == bundleID }
        }
        guard let targetIndex = groups.firstIndex(where: { $0.id == groupID }) else { return }
        if !groups[targetIndex].appBundleIDs.contains(bundleID) {
            groups[targetIndex].appBundleIDs.append(bundleID)
        }
        if !groups[targetIndex].appRefs.contains(where: { $0.bundleID == bundleID }) {
            groups[targetIndex].appRefs.append(appRefForBundleID(bundleID))
        }
    }

    func assignURL(urlID: UUID, to groupID: UUID) {
        for index in groups.indices {
            groups[index].urlPatternIDs.removeAll { $0 == urlID }
        }
        guard let targetIndex = groups.firstIndex(where: { $0.id == groupID }) else { return }
        if !groups[targetIndex].urlPatternIDs.contains(urlID) {
            groups[targetIndex].urlPatternIDs.append(urlID)
        }
    }

    func removeAppFromGroup(bundleID: String) {
        for index in groups.indices {
            groups[index].appBundleIDs.removeAll { $0 == bundleID }
            groups[index].appRefs.removeAll { $0.bundleID == bundleID }
        }
    }

    func removeURLFromGroup(urlID: UUID) {
        for index in groups.indices {
            groups[index].urlPatternIDs.removeAll { $0 == urlID }
        }
    }

    func deleteURLItem(id: UUID) {
        urlItems.removeAll { $0.id == id }
        removeURLFromGroup(urlID: id)
    }

    func deleteGroup(groupID: UUID) {
        groups.removeAll { $0.id == groupID }
    }

    func duplicateGroup(groupID: UUID) {
        guard let sourceGroup = groups.first(where: { $0.id == groupID }) else { return }
        let duplicatedGroup = AppBranchGroup(
            id: UUID(),
            name: uniqueDuplicateGroupName(for: sourceGroup.name),
            prompt: sourceGroup.prompt,
            appBundleIDs: sourceGroup.appBundleIDs,
            appRefs: sourceGroup.appRefs,
            urlPatternIDs: sourceGroup.urlPatternIDs,
            autoKeyPressEnabled: sourceGroup.autoKeyPressEnabled,
            autoKeyPressHotkey: sourceGroup.autoKeyPressHotkey,
            isExpanded: true
        )
        groups.append(duplicatedGroup)
    }

    func saveGroup(state: AppBranchModal) {
        let trimmedName = groupNameDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else {
            modalErrorMessage = AppLocalization.localizedString("Group name is required.")
            return
        }

        switch state {
        case .createGroup:
            groups.append(
                AppBranchGroup(
                    id: UUID(),
                    name: trimmedName,
                    prompt: groupPromptDraft,
                    appBundleIDs: [],
                    appRefs: [],
                    urlPatternIDs: [],
                    autoKeyPressEnabled: groupAutoKeyPressEnabledDraft,
                    autoKeyPressHotkey: groupAutoKeyPressHotkeyDraft,
                    isExpanded: true
                )
            )
        case .editGroup(let groupID):
            guard let index = groups.firstIndex(where: { $0.id == groupID }) else { break }
            groups[index].name = trimmedName
            groups[index].prompt = groupPromptDraft
            groups[index].autoKeyPressEnabled = groupAutoKeyPressEnabledDraft
            groups[index].autoKeyPressHotkey = groupAutoKeyPressHotkeyDraft
        default:
            break
        }

        modal = nil
    }

    func uniqueDuplicateGroupName(for baseName: String) -> String {
        let existingNames = Set(groups.map(\.name))
        var candidateIndex = 1
        var candidate = "\(baseName)-\(String(format: "%02d", candidateIndex))"

        while existingNames.contains(candidate) {
            candidateIndex += 1
            candidate = "\(baseName)-\(String(format: "%02d", candidateIndex))"
        }

        return candidate
    }

    func saveAddedURLs() {
        let entries = urlDraft
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard !entries.isEmpty else {
            modalErrorMessage = AppLocalization.localizedString("Enter at least one URL pattern.")
            return
        }

        let canonicalEntries = entries.map(AppBranchURLPatternService.canonicalizedPattern)
        let normalizedInput = canonicalEntries.map(AppBranchURLPatternService.normalizedPattern)
        if Set(normalizedInput).count != normalizedInput.count {
            modalErrorMessage = AppLocalization.localizedString("Duplicate URL patterns detected in input.")
            return
        }

        if let invalid = canonicalEntries.first(where: { !AppBranchURLPatternService.isValidWildcardURLPattern($0) }) {
            modalErrorMessage = AppLocalization.format("Invalid URL pattern: %@. Use wildcard format like google.com/*.", invalid)
            return
        }

        let existing = Set(urlItems.map { AppBranchURLPatternService.normalizedPattern($0.pattern) })
        if normalizedInput.contains(where: { existing.contains($0) }) {
            modalErrorMessage = AppLocalization.localizedString("Some URL patterns already exist.")
            return
        }

        let newItems = canonicalEntries.map { BranchURLItem(id: UUID(), pattern: $0) }
        urlItems.append(contentsOf: newItems)
        modal = nil
    }

    func saveEditedURL(urlID: UUID) {
        let canonical = AppBranchURLPatternService.canonicalizedPattern(urlDraft)
        guard !canonical.isEmpty else {
            modalErrorMessage = AppLocalization.localizedString("URL pattern is required.")
            return
        }

        guard AppBranchURLPatternService.isValidWildcardURLPattern(canonical) else {
            modalErrorMessage = AppLocalization.localizedString("Invalid URL pattern. Use wildcard format like google.com/*.")
            return
        }

        let normalized = AppBranchURLPatternService.normalizedPattern(canonical)
        let others = Set(urlItems.filter { $0.id != urlID }.map { AppBranchURLPatternService.normalizedPattern($0.pattern) })
        if others.contains(normalized) {
            modalErrorMessage = AppLocalization.localizedString("URL pattern already exists.")
            return
        }

        guard let index = urlItems.firstIndex(where: { $0.id == urlID }) else {
            modal = nil
            return
        }
        urlItems[index].pattern = canonical
        modal = nil
    }

    func canonicalizedPattern(_ value: String) -> String {
        AppBranchURLPatternService.canonicalizedPattern(value)
    }

    func normalizedPattern(_ value: String) -> String {
        AppBranchURLPatternService.normalizedPattern(value)
    }

    func isValidWildcardURLPattern(_ pattern: String) -> Bool {
        AppBranchURLPatternService.isValidWildcardURLPattern(pattern)
    }

    func loadPersistedGroups() {
        guard let data = UserDefaults.standard.data(forKey: AppPreferenceKey.appBranchGroups) else {
            return
        }
        do {
            groups = try JSONDecoder().decode([AppBranchGroup].self, from: data)
        } catch {
            groups = []
        }
    }

    func saveGroups() {
        do {
            let data = try JSONEncoder().encode(groups)
            UserDefaults.standard.set(data, forKey: AppPreferenceKey.appBranchGroups)
        } catch {
            VoxtLog.error("Failed to persist app branch groups: \(error.localizedDescription)")
        }
    }

    func loadPersistedURLs() {
        guard let data = UserDefaults.standard.data(forKey: AppPreferenceKey.appBranchURLs) else {
            return
        }
        do {
            urlItems = try JSONDecoder().decode([BranchURLItem].self, from: data).map {
                BranchURLItem(id: $0.id, pattern: canonicalizedPattern($0.pattern))
            }
        } catch {
            urlItems = []
        }
    }

    func saveURLs() {
        do {
            let data = try JSONEncoder().encode(urlItems)
            UserDefaults.standard.set(data, forKey: AppPreferenceKey.appBranchURLs)
        } catch {
            VoxtLog.error("Failed to persist app branch URLs: \(error.localizedDescription)")
        }
    }

    func handleDrop(providers: [NSItemProvider], groupID: UUID) -> Bool {
        guard let provider = providers.first(where: { $0.canLoadObject(ofClass: NSString.self) }) else {
            return false
        }

        provider.loadObject(ofClass: NSString.self) { object, _ in
            guard let rawValue = object as? NSString else { return }
            let value = rawValue as String
            DispatchQueue.main.async {
                if value.hasPrefix("app:") {
                    let bundleID = String(value.dropFirst(4))
                    assignApp(bundleID: bundleID, to: groupID)
                    draggingAppID = nil
                } else if value.hasPrefix("url:") {
                    let rawID = String(value.dropFirst(4))
                    if let urlID = UUID(uuidString: rawID) {
                        assignURL(urlID: urlID, to: groupID)
                    }
                }
            }
        }
        return true
    }
}
