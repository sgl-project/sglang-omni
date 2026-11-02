// FeatureAvailabilitySections.swift
// Provides the Feature master-toggle grid for Custom → Feature.

import SwiftUI
import AppKit

extension FeatureSettingsView {
    var featuresContent: some View {
        let _ = interfaceLanguageRaw
        let _ = appBranchGroupsData
        return featurePage(
            title: AppLocalization.localizedString("Feature"),
            subtitle: AppLocalization.localizedString("Choose which features are available in Voxt."),
            iconKind: .features,
            pills: [],
            showsHeroHeader: false,
            allowsScrolling: false
        ) {
            LazyVGrid(
                columns: [
                    GridItem(.flexible(), spacing: 12),
                    GridItem(.flexible(), spacing: 12)
                ],
                alignment: .leading,
                spacing: 12
            ) {
                ForEach(FeatureAvailabilityCardKind.allCases) { kind in
                    let iconBundleIDs = appIconBundleIDs(for: kind)
                    FeatureAvailabilityCard(
                        kind: kind,
                        isEnabled: binding(for: kind),
                        badges: availabilityBadges(for: kind),
                        appIconBundleIDs: iconBundleIDs,
                        onSelect: { openFeatureAvailabilityDestination(kind) }
                    )
                }
            }
        }
    }

    private func binding(for kind: FeatureAvailabilityCardKind) -> Binding<Bool>? {
        switch kind {
        case .transcription:
            return nil
        case .translation:
            return binding(
                get: { featureSettings.availability.translationEnabled },
                set: { featureSettings.availability.translationEnabled = $0 }
            )
        case .rewrite:
            return binding(
                get: { featureSettings.availability.rewriteEnabled },
                set: { featureSettings.availability.rewriteEnabled = $0 }
            )
        case .note:
            return binding(
                get: { featureSettings.availability.notesEnabled },
                set: { enabled in
                    featureSettings.availability.notesEnabled = enabled
                    featureSettings.transcription.notes.enabled = enabled
                }
            )
        case .appEnhancement:
            return binding(
                get: { featureSettings.availability.appEnhancementEnabled },
                set: { enabled in
                    featureSettings.availability.appEnhancementEnabled = enabled
                    featureSettings.rewrite.appEnhancementEnabled = enabled
                }
            )
        case .meeting:
            return binding(
                get: { featureSettings.availability.meetingEnabled },
                set: { featureSettings.availability.meetingEnabled = $0 }
            )
        case .files:
            return Binding(
                get: { featureSettings.availability.filesEnabled },
                set: { enabled in
                    guard enabled || !meetingFileTaskQueue.hasActiveTasks else {
                        isFilesDisableBlockedAlertPresented = true
                        return
                    }
                    featureSettings.availability.filesEnabled = enabled
                    saveFeatureSettings()
                }
            )
        }
    }

    func availabilityBadges(for kind: FeatureAvailabilityCardKind) -> [FeatureAvailabilityConfigBadge] {
        switch kind {
        case .transcription:
            var badges = [
                modelBadge(
                    id: "transcription-asr",
                    title: selectorBuilder.asrSelectionBadgeTitle(featureSettings.transcription.asrSelectionID),
                    logoKey: selectorBuilder.asrSelectionLogoKey(featureSettings.transcription.asrSelectionID)
                )
            ]
            if featureSettings.transcription.llmEnabled {
                badges.append(
                    modelBadge(
                        id: "transcription-llm",
                        title: selectorBuilder.llmSelectionBadgeTitle(featureSettings.transcription.llmSelectionID),
                        logoKey: selectorBuilder.llmSelectionLogoKey(featureSettings.transcription.llmSelectionID)
                    )
                )
            }
            return badges
        case .translation:
            return [
                modelBadge(
                    id: "translation-asr",
                    title: selectorBuilder.asrSelectionBadgeTitle(featureSettings.translation.asrSelectionID),
                    logoKey: selectorBuilder.asrSelectionLogoKey(featureSettings.translation.asrSelectionID)
                ),
                modelBadge(
                    id: "translation-model",
                    title: selectorBuilder.translationSelectionBadgeTitle(featureSettings.translation.modelSelectionID),
                    logoKey: selectorBuilder.translationSelectionLogoKey(featureSettings.translation.modelSelectionID)
                ),
                FeatureAvailabilityConfigBadge(
                    id: "translation-target",
                    title: featureSettings.translation.targetLanguage.title,
                    systemImageName: "globe"
                )
            ]
        case .rewrite:
            return [
                modelBadge(
                    id: "rewrite-asr",
                    title: selectorBuilder.asrSelectionBadgeTitle(featureSettings.rewrite.asrSelectionID),
                    logoKey: selectorBuilder.asrSelectionLogoKey(featureSettings.rewrite.asrSelectionID)
                ),
                modelBadge(
                    id: "rewrite-llm",
                    title: selectorBuilder.llmSelectionBadgeTitle(featureSettings.rewrite.llmSelectionID),
                    logoKey: selectorBuilder.llmSelectionLogoKey(featureSettings.rewrite.llmSelectionID)
                )
            ]
        case .appEnhancement:
            return [
                FeatureAvailabilityConfigBadge(
                    id: "app-enhancement-groups",
                    title: AppLocalization.format("Groups (%d)", appBranchGroupCount),
                    systemImageName: "square.stack.3d.up"
                )
            ]
        case .note:
            return [
                modelBadge(
                    id: "note-title-model",
                    title: selectorBuilder.llmSelectionBadgeTitle(featureSettings.transcription.notes.titleModelSelectionID),
                    logoKey: selectorBuilder.llmSelectionLogoKey(featureSettings.transcription.notes.titleModelSelectionID)
                ),
                FeatureAvailabilityConfigBadge(
                    id: "note-trigger",
                    title: noteTriggerBadgeTitle,
                    systemImageName: "keyboard"
                )
            ]
        case .meeting:
            return [
                modelBadge(
                    id: "meeting-asr",
                    title: selectorBuilder.asrSelectionBadgeTitle(featureSettings.meeting.asrSelectionID),
                    logoKey: selectorBuilder.asrSelectionLogoKey(featureSettings.meeting.asrSelectionID)
                ),
                modelBadge(
                    id: "meeting-summary",
                    title: selectorBuilder.llmSelectionBadgeTitle(featureSettings.meeting.summaryModelSelectionID),
                    logoKey: selectorBuilder.llmSelectionLogoKey(featureSettings.meeting.summaryModelSelectionID)
                ),
                FeatureAvailabilityConfigBadge(
                    id: "meeting-speakers",
                    title: featureSettings.meeting.speakerDiarizationModel.title,
                    systemImageName: "person.2.wave.2"
                )
            ]
        case .files:
            return [
                FeatureAvailabilityConfigBadge(
                    id: "files-task-count",
                    title: AppLocalization.format("File Tasks (%d)", meetingFileTaskQueue.tasks.count),
                    systemImageName: "list.bullet"
                )
            ]
        }
    }

    func appIconBundleIDs(for kind: FeatureAvailabilityCardKind) -> [String] {
        guard kind == .appEnhancement else { return [] }
        // Collect recent candidates only; icon resolution happens asynchronously in the card.
        return recentAppEnhancementBundleIDs(limit: 24)
    }

    private var appBranchGroups: [AppBranchGroup] {
        let data = appBranchGroupsData.isEmpty
            ? UserDefaults.standard.data(forKey: AppPreferenceKey.appBranchGroups)
            : appBranchGroupsData
        guard let data,
              let groups = try? JSONDecoder().decode([AppBranchGroup].self, from: data)
        else {
            return []
        }
        return groups
    }

    private var appBranchGroupCount: Int {
        appBranchGroups.count
    }

    /// Newest configured apps first (group order, then appRefs order).
    private func recentAppEnhancementBundleIDs(limit: Int) -> [String] {
        var seen = Set<String>()
        var result: [String] = []

        for group in appBranchGroups.reversed() {
            let refs: [AppBranchAppRef]
            if group.appRefs.isEmpty {
                refs = group.appBundleIDs.map { AppBranchAppRef(bundleID: $0, displayName: $0) }
            } else {
                refs = group.appRefs
            }

            for ref in refs.reversed() {
                let bundleID = ref.bundleID.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !bundleID.isEmpty, seen.insert(bundleID).inserted else { continue }
                result.append(bundleID)
                if result.count == limit {
                    return result
                }
            }
        }

        return result
    }

    private var noteTriggerBadgeTitle: String {
        HotkeyPreference.displayString(
            for: HotkeyPreference.loadNoteBindings().first?.hotkey
                ?? HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.defaultNoteKeyCode,
                    modifiers: HotkeyPreference.defaultNoteModifiers,
                    sidedModifiers: HotkeyPreference.defaultNoteSidedModifiers
                ),
            distinguishModifierSides: true
        )
    }

    private func modelBadge(
        id: String,
        title: String,
        logoKey: ModelLogoKey
    ) -> FeatureAvailabilityConfigBadge {
        FeatureAvailabilityConfigBadge(
            id: id,
            title: title,
            logoKey: logoKey
        )
    }

    private func openFeatureAvailabilityDestination(_ kind: FeatureAvailabilityCardKind) {
        if let onSelectFeatureTab {
            onSelectFeatureTab(kind.featureTab)
            return
        }

        NotificationCenter.default.post(
            name: .voxtSettingsNavigate,
            object: nil,
            userInfo: SettingsNavigationTarget(
                tab: .feature,
                featureTab: kind.featureTab
            ).userInfo
        )
    }
}

enum FeatureAvailabilityCardKind: String, CaseIterable, Identifiable {
    case transcription
    case translation
    case rewrite
    case appEnhancement
    case note
    case meeting
    case files

    var id: String { rawValue }

    var titleKey: String {
        switch self {
        case .transcription: return "Transcription"
        case .translation: return "Translation"
        case .rewrite: return "Rewrite"
        case .note: return "Notes"
        case .appEnhancement: return "App Enhancement"
        case .meeting: return "Meeting"
        case .files: return "Files"
        }
    }

    var title: String {
        AppLocalization.localizedString(titleKey)
    }

    var iconKind: SettingsSidebarIconKind? {
        switch self {
        case .transcription: return .transcription
        case .translation: return .translation
        case .rewrite: return .rewrite
        case .note: return .note
        case .appEnhancement: return .appEnhancement
        case .meeting: return .meeting
        case .files: return .files
        }
    }

    var systemImageName: String {
        switch self {
        case .files:
            return "doc.on.doc"
        default:
            return ""
        }
    }

    var featureTab: FeatureSettingsTab {
        switch self {
        case .transcription: return .transcription
        case .translation: return .translation
        case .rewrite: return .rewrite
        case .note: return .note
        case .appEnhancement: return .appEnhancement
        case .meeting: return .meeting
        case .files: return .files
        }
    }

    var showsToggle: Bool {
        self != .transcription
    }
}

struct FeatureAvailabilityConfigBadge: Identifiable, Hashable {
    let id: String
    let title: String
    var logoKey: ModelLogoKey? = nil
    var systemImageName: String? = nil
}

struct FeatureAvailabilityCard: View {
    let kind: FeatureAvailabilityCardKind
    let isEnabled: Binding<Bool>?
    let badges: [FeatureAvailabilityConfigBadge]
    var appIconBundleIDs: [String] = []
    let onSelect: () -> Void

    private var isFeatureEnabled: Bool {
        guard kind.showsToggle, let isEnabled else { return true }
        return isEnabled.wrappedValue
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 8) {
                Button(action: onSelect) {
                    HStack(alignment: .center, spacing: 8) {
                        if let iconKind = kind.iconKind {
                            SettingsSidebarIconView(kind: iconKind)
                                .frame(width: 18, height: 18)
                        } else {
                            Image(systemName: kind.systemImageName)
                                .font(.system(size: 17, weight: .semibold))
                                .foregroundStyle(Color.accentColor)
                                .frame(width: 18, height: 18)
                        }

                        Text(kind.title)
                            .font(.headline.weight(.semibold))
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                    }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(AppLocalization.format("Open %@", kind.title))

                Spacer(minLength: 8)

                if kind.showsToggle, let isEnabled {
                    Toggle("", isOn: isEnabled)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .controlSize(.small)
                        .accessibilityLabel(AppLocalization.format("Enable %@", kind.title))
                }
            }

            if kind == .appEnhancement {
                FeatureAvailabilityAppEnhancementAccessory(
                    candidateBundleIDs: appIconBundleIDs,
                    fallbackBadges: badges,
                    isFeatureEnabled: isFeatureEnabled
                )
            } else if !badges.isEmpty {
                FeatureAvailabilityBadgeFlow(spacing: 6, lineSpacing: 6) {
                    ForEach(badges) { badge in
                        FeatureAvailabilityConfigBadgeView(badge: badge)
                    }
                }
                .opacity(isFeatureEnabled ? 1 : 0.55)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, minHeight: 72, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                .fill(SettingsUIStyle.groupedFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                .stroke(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
        )
    }
}

private struct FeatureAvailabilityAppEnhancementAccessory: View {
    let candidateBundleIDs: [String]
    let fallbackBadges: [FeatureAvailabilityConfigBadge]
    let isFeatureEnabled: Bool

    @State private var resolvedIcons: [ResolvedAppIcon] = []
    @State private var didResolve = false

    private struct ResolvedAppIcon: Identifiable {
        let id: String
        let image: NSImage
    }

    var body: some View {
        Group {
            if candidateBundleIDs.isEmpty {
                fallbackBadgeStrip
            } else if !didResolve {
                Color.clear
                    .frame(height: 20)
            } else if resolvedIcons.isEmpty {
                fallbackBadgeStrip
            } else {
                HStack(spacing: 5) {
                    ForEach(resolvedIcons) { icon in
                        Image(nsImage: icon.image)
                            .resizable()
                            .interpolation(.high)
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 20, height: 20)
                            .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
                            .overlay(
                                RoundedRectangle(cornerRadius: 5, style: .continuous)
                                    .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 0.5)
                            )
                            .accessibilityHidden(true)
                    }
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(AppLocalization.format("Apps (%d)", resolvedIcons.count))
            }
        }
        .opacity(isFeatureEnabled ? 1 : 0.55)
        .task(id: candidateBundleIDs) {
            await resolveIcons()
        }
    }

    @ViewBuilder
    private var fallbackBadgeStrip: some View {
        if !fallbackBadges.isEmpty {
            FeatureAvailabilityBadgeFlow(spacing: 6, lineSpacing: 6) {
                ForEach(fallbackBadges) { badge in
                    FeatureAvailabilityConfigBadgeView(badge: badge)
                }
            }
        }
    }

    @MainActor
    private func resolveIcons() async {
        didResolve = false
        resolvedIcons = []

        guard !candidateBundleIDs.isEmpty else {
            didResolve = true
            return
        }

        var icons: [ResolvedAppIcon] = []
        icons.reserveCapacity(min(candidateBundleIDs.count, 6))

        for bundleID in candidateBundleIDs {
            await Task.yield()
            guard !Task.isCancelled else { return }
            guard let image = EnhancementOverlayIconResolver.appIcon(bundleID: bundleID) else {
                continue
            }
            icons.append(ResolvedAppIcon(id: bundleID, image: image))
            if icons.count == 6 {
                break
            }
        }

        guard !Task.isCancelled else { return }
        resolvedIcons = icons
        didResolve = true
    }
}

private struct FeatureAvailabilityConfigBadgeView: View {
    let badge: FeatureAvailabilityConfigBadge

    var body: some View {
        HStack(spacing: 5) {
            if let logoKey = badge.logoKey {
                ModelLogoView(key: logoKey, fallbackTitle: badge.title, size: 12)
            } else if let systemImageName = badge.systemImageName {
                Image(systemName: systemImageName)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 12, height: 12)
            }

            Text(badge.title)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.primary.opacity(0.82))
                .lineLimit(1)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 4)
        .background(
            Capsule(style: .continuous)
                .fill(SettingsUIStyle.subtleFillColor)
        )
        .overlay(
            Capsule(style: .continuous)
                .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
        )
        .accessibilityLabel(badge.title)
    }
}

private struct FeatureAvailabilityBadgeFlow: Layout {
    var spacing: CGFloat = 6
    var lineSpacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        arrange(proposal: proposal, subviews: subviews).size
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        let result = arrange(proposal: proposal, subviews: subviews)
        for index in subviews.indices {
            let origin = result.origins[index]
            subviews[index].place(
                at: CGPoint(x: bounds.minX + origin.x, y: bounds.minY + origin.y),
                proposal: .unspecified
            )
        }
    }

    private func arrange(
        proposal: ProposedViewSize,
        subviews: Subviews
    ) -> (origins: [CGPoint], size: CGSize) {
        let maxWidth = proposal.width ?? .infinity
        var origins: [CGPoint] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        var totalWidth: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > maxWidth {
                x = 0
                y += rowHeight + lineSpacing
                rowHeight = 0
            }

            origins.append(CGPoint(x: x, y: y))
            rowHeight = max(rowHeight, size.height)
            x += size.width + spacing
            totalWidth = max(totalWidth, x - spacing)
        }

        return (
            origins,
            CGSize(
                width: maxWidth.isFinite ? maxWidth : totalWidth,
                height: y + rowHeight
            )
        )
    }
}
