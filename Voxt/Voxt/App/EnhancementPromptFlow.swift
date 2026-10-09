// EnhancementPromptFlow.swift
// Provides Enhancement Prompt Flow for app lifecycle and routing.

import Foundation
import AppKit
import ApplicationServices

extension AppDelegate {
    static let rawTranscriptionTemplateVariable = EnhancementPromptResolver.rawTranscriptionTemplateVariable
    static let userMainLanguageTemplateVariable = EnhancementPromptResolver.userMainLanguageTemplateVariable

    private struct AppBranchAutoKeyPressMatch {
        let group: AppBranchGroup
        let matchedAppGroupName: String?
        let matchedURLGroupName: String?
        let overlayIconMatch: OverlayEnhancementIconMatch?
    }

    struct EnhancementPromptResolution {
        enum Delivery {
            case systemPrompt
            case userMessage
            case skipEnhancement
        }

        let content: String
        let dictionaryGlossary: String?
        let delivery: Delivery
        let source: EnhancementPromptResolver.Source
        let overlayIconMatch: OverlayEnhancementIconMatch?
    }

    func resolveGlobalEnhancementPromptTemplate(_ prompt: String, rawTranscription: String) -> String {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else { return AppPromptDefaults.text(for: .enhancement) }
        return resolveEnhancementPromptVariables(in: trimmedPrompt, rawTranscription: rawTranscription)
    }

    func resolvedGlobalEnhancementPrompt() -> String {
        FeatureSettingsStore.load(defaults: .standard).transcription.prompt
    }

    func resolvedEnhancementPrompt(
        rawTranscription: String,
        glossarySelectionPolicy: DictionaryGlossarySelectionPolicy? = nil
    ) -> EnhancementPromptResolution {
        let groups = loadAppBranchGroups()
        let urlsByID = loadAppBranchURLsByID()
        let context = currentEnhancementContext()
        let frontmostBundleID = context.bundleID
        let focusedAppName = NSWorkspace.shared.frontmostApplication?.localizedName
        let activeBrowserURL = isBrowserBundleID(frontmostBundleID)
            ? activeBrowserTabURL(frontmostBundleID: frontmostBundleID)
            : nil
        let normalizedActiveURL = AppBranchURLPatternService.normalizedURLForMatching(activeBrowserURL)
        let glossary = dictionaryGlossaryText(
            for: rawTranscription,
            purpose: .enhancement,
            selectionPolicy: glossarySelectionPolicy
        )

        let resolution = EnhancementPromptResolver.resolve(
            .init(
                globalPrompt: resolvedGlobalEnhancementPrompt(),
                rawTranscription: rawTranscription,
                userMainLanguagePromptValue: userMainLanguagePromptValue,
                userOtherLanguagesPromptValue: userOtherMainLanguagesPromptValue,
                dictionaryGlossary: glossary,
                appEnhancementEnabled: appEnhancementEnabled,
                groups: groups,
                urlsByID: urlsByID,
                frontmostBundleID: frontmostBundleID,
                focusedAppName: focusedAppName,
                normalizedActiveURL: normalizedActiveURL,
                supportedBrowserBundleIDs: supportedBrowserBundleIDs()
            )
        )

        lastEnhancementPromptContext = EnhancementPromptContext(
            focusedAppName: resolution.promptContext.focusedAppName,
            focusedAppBundleID: frontmostBundleID,
            matchedGroupID: resolution.promptContext.matchedGroupID,
            matchedGroupName: resolution.promptContext.matchedURLGroupName ?? resolution.promptContext.matchedAppGroupName,
            matchedAppGroupName: resolution.promptContext.matchedAppGroupName,
            matchedURLGroupName: resolution.promptContext.matchedURLGroupName,
            overlayIconMatch: overlayIconMatch(
                for: resolution.source,
                frontmostBundleID: frontmostBundleID,
                activeBrowserURL: activeBrowserURL
            )
        )

        switch resolution.source {
        case .globalDefault(.appBranchDisabled):
            VoxtLog.info("Enhancement prompt source: global/default (app branch disabled)")
        case .globalDefault(.noGroups):
            VoxtLog.info("Enhancement prompt source: global/default (no app branch groups)")
        case .globalDefault(.browserURLUnavailable(let bundleID)):
            VoxtLog.info("Enhancement prompt source: global/default (browser url unavailable), bundleID=\(bundleID ?? "nil")")
        case .globalDefault(.browserURLNoMatch(let bundleID, let url)):
            VoxtLog.info("Enhancement prompt source: global/default (browser url no group match), bundleID=\(bundleID ?? "nil"), url=\(url)")
        case .globalDefault(.noGroupMatch(let bundleID)):
            VoxtLog.info("Enhancement prompt source: global/default (no group match), bundleID=\(bundleID ?? "nil")")
        case .appGroup(let groupName, let bundleID):
            VoxtLog.info("Enhancement prompt source: group(app) group=\(groupName), bundleID=\(bundleID)")
        case .appGroupFallback(let groupName, let bundleID):
            VoxtLog.info("Enhancement prompt source: global/default (matched app group has empty prompt), group=\(groupName), bundleID=\(bundleID)")
        case .urlGroup(let groupName, let pattern, let url):
            VoxtLog.info("Enhancement prompt source: group(url) group=\(groupName), pattern=\(pattern), url=\(url)")
        case .urlGroupFallback(let groupName, let pattern, let url):
            VoxtLog.info("Enhancement prompt source: global/default (matched url group has empty prompt), group=\(groupName), pattern=\(pattern), url=\(url)")
        }

        return EnhancementPromptResolution(
            content: resolution.content,
            dictionaryGlossary: glossary,
            delivery: {
                switch resolution.delivery {
                case .systemPrompt:
                    return .systemPrompt
                case .userMessage:
                    return .userMessage
                case .skipEnhancement:
                    return .skipEnhancement
                }
            }(),
            source: resolution.source,
            overlayIconMatch: lastEnhancementPromptContext?.overlayIconMatch
        )
    }

    func captureEnhancementContextSnapshot() -> EnhancementContextSnapshot {
        let frontmostApplication = NSWorkspace.shared.frontmostApplication
        let fallbackAppName = frontmostApplication?.localizedName
        let fallbackBundleID = frontmostApplication?.bundleIdentifier
        let fallbackPID = frontmostApplication?.processIdentifier
        return EnhancementContextSnapshot(
            appName: sessionTargetApplicationBundleID != nil
                ? (NSRunningApplication(processIdentifier: sessionTargetApplicationPID ?? 0)?.localizedName ?? fallbackAppName)
                : fallbackAppName,
            bundleID: sessionTargetApplicationBundleID ?? fallbackBundleID,
            pid: sessionTargetApplicationBundleID != nil ? sessionTargetApplicationPID : fallbackPID,
            capturedAt: Date()
        )
    }

    func currentDictionaryScope() -> (groupID: UUID?, groupName: String?) {
        let groups = loadAppBranchGroups()
        guard !groups.isEmpty else { return (nil, nil) }

        let urlsByID = loadAppBranchURLsByID()
        let context = currentEnhancementContext()
        let frontmostBundleID = context.bundleID

        if isBrowserBundleID(frontmostBundleID) {
            let activeURL = activeBrowserTabURL(frontmostBundleID: frontmostBundleID)
            guard let normalizedActiveURL = AppBranchURLPatternService.normalizedURLForMatching(activeURL) else {
                return (nil, nil)
            }
            if let match = AppBranchURLPatternService.firstGroupMatch(
                groups: groups,
                urlsByID: urlsByID,
                normalizedURL: normalizedActiveURL
            ) {
                return (match.groupID, match.groupName)
            }
            return (nil, nil)
        }

        guard let frontmostBundleID else { return (nil, nil) }
        if let group = groups.first(where: { $0.appBundleIDs.contains(frontmostBundleID) }) {
            return (group.id, group.name)
        }
        return (nil, nil)
    }

    func autoKeyPressHotkeyForCurrentAppBranchSession() -> HotkeyPreference.Hotkey? {
        guard appEnhancementEnabled else { return nil }
        guard let match = currentAppBranchAutoKeyPressMatch() else { return nil }

        if match.group.autoKeyPressEnabled {
            let context = currentEnhancementContext()
            lastEnhancementPromptContext = EnhancementPromptContext(
                focusedAppName: context.appName,
                focusedAppBundleID: context.bundleID,
                matchedGroupID: match.group.id,
                matchedGroupName: match.matchedURLGroupName ?? match.matchedAppGroupName,
                matchedAppGroupName: match.matchedAppGroupName,
                matchedURLGroupName: match.matchedURLGroupName,
                overlayIconMatch: match.overlayIconMatch
            )
            VoxtLog.input(
                "Auto Key enabled for app branch group. group=\(match.group.name), hotkey=\(HotkeyPreference.displayString(for: match.group.autoKeyPressHotkey, distinguishModifierSides: false))",
                verbose: true
            )
        }

        return match.group.autoKeyPressEnabled ? match.group.autoKeyPressHotkey : nil
    }

    private func resolveEnhancementPromptVariables(in prompt: String, rawTranscription: String) -> String {
        prompt
            .replacingOccurrences(of: Self.rawTranscriptionTemplateVariable, with: rawTranscription)
            .replacingOccurrences(of: Self.userMainLanguageTemplateVariable, with: userMainLanguagePromptValue)
    }

    private func loadAppBranchGroups() -> [AppBranchGroup] {
        guard let data = UserDefaults.standard.data(forKey: AppPreferenceKey.appBranchGroups) else { return [] }
        return (try? JSONDecoder().decode([AppBranchGroup].self, from: data)) ?? []
    }

    private func loadAppBranchURLsByID() -> [UUID: String] {
        guard let data = UserDefaults.standard.data(forKey: AppPreferenceKey.appBranchURLs),
              let items = try? JSONDecoder().decode([BranchURLItem].self, from: data)
        else {
            return [:]
        }

        var result: [UUID: String] = [:]
        for item in items {
            result[item.id] = AppBranchURLPatternService.normalizedPattern(item.pattern)
        }
        return result
    }

    private func currentEnhancementContext() -> EnhancementContextSnapshot {
        if let snapshot = enhancementContextSnapshot {
            let age = Date().timeIntervalSince(snapshot.capturedAt)
            if age <= 20 {
                return snapshot
            }
        }
        return captureEnhancementContextSnapshot()
    }

    private func currentAppBranchAutoKeyPressMatch() -> AppBranchAutoKeyPressMatch? {
        let groups = loadAppBranchGroups()
        guard !groups.isEmpty else { return nil }

        let urlsByID = loadAppBranchURLsByID()
        let context = currentEnhancementContext()
        let frontmostBundleID = context.bundleID

        if isBrowserBundleID(frontmostBundleID) {
            let activeURL = activeBrowserTabURL(frontmostBundleID: frontmostBundleID)
            guard let normalizedActiveURL = AppBranchURLPatternService.normalizedURLForMatching(activeURL),
                  let urlMatch = AppBranchURLPatternService.firstGroupMatch(
                    groups: groups,
                    urlsByID: urlsByID,
                    normalizedURL: normalizedActiveURL
                  ),
                  let group = groups.first(where: { $0.id == urlMatch.groupID })
            else {
                return nil
            }

            return AppBranchAutoKeyPressMatch(
                group: group,
                matchedAppGroupName: nil,
                matchedURLGroupName: urlMatch.groupName,
                overlayIconMatch: frontmostBundleID.map {
                    OverlayEnhancementIconMatch(
                        kind: .url,
                        bundleID: $0,
                        urlOrigin: EnhancementOverlayIconResolver.faviconOrigin(fromPageURL: activeURL)
                    )
                }
            )
        }

        guard let frontmostBundleID,
              let group = groups.first(where: { $0.appBundleIDs.contains(frontmostBundleID) })
        else {
            return nil
        }

        return AppBranchAutoKeyPressMatch(
            group: group,
            matchedAppGroupName: group.name,
            matchedURLGroupName: nil,
            overlayIconMatch: OverlayEnhancementIconMatch(
                kind: .app,
                bundleID: frontmostBundleID,
                urlOrigin: nil
            )
        )
    }

    private func overlayIconMatch(
        for source: EnhancementPromptResolver.Source,
        frontmostBundleID: String?,
        activeBrowserURL: String?
    ) -> OverlayEnhancementIconMatch? {
        switch source {
        case .appGroup(_, let bundleID), .appGroupFallback(_, let bundleID):
            return OverlayEnhancementIconMatch(
                kind: .app,
                bundleID: bundleID,
                urlOrigin: nil
            )
        case .urlGroup, .urlGroupFallback:
            guard let bundleID = frontmostBundleID else { return nil }
            return OverlayEnhancementIconMatch(
                kind: .url,
                bundleID: bundleID,
                urlOrigin: EnhancementOverlayIconResolver.faviconOrigin(fromPageURL: activeBrowserURL)
            )
        case .globalDefault:
            return nil
        }
    }

}
