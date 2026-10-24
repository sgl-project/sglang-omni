import SwiftUI
import AppKit

struct SettingsSidebar: View {
    @Binding var sidebarMode: SettingsSidebarMode
    @Binding var selectedTab: SettingsTab
    @Binding var selectedFeatureTab: FeatureSettingsTab
    @Binding var selectedHistoryFilter: HistoryFilterTab
    let onSelectTab: (SettingsTab) -> Void
    let onSelectFeatureTab: (FeatureSettingsTab) -> Void
    let onSelectHistoryFilter: (HistoryFilterTab) -> Void
    let onReturnToRoot: () -> Void
    let featureAvailability: FeatureAvailabilitySettings
    let hasMissingPermissions: Bool
    let hasNoAvailableMicrophones: Bool
    let modelStorageAuthorizationIssue: String?
    let activeModelDownloadCount: Int
    let hasMissingModelConfigurationIssues: Bool
    let updateBadgeState: UpdateBadgeState
    let hasUnreadNotification: Bool
    let onTapPermissionBadge: () -> Void
    let onTapMicrophoneBadge: () -> Void
    let onTapModelStorageAuthorizationBadge: () -> Void
    let onTapModelBadge: () -> Void
    let onTapUpdateBadge: () -> Void
    let onTapNotification: () -> Void
    let onTapWebsite: () -> Void
    let onTapFeedback: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SettingsSidebarHeader(
                sidebarMode: sidebarMode,
                updateBadgeState: updateBadgeState,
                onTapUpdateBadge: onTapUpdateBadge,
                onReturnToRoot: onReturnToRoot
            )

            SettingsSidebarMenuPager(
                sidebarMode: sidebarMode,
                rootTabs: visibleRootTabs,
                featureTabs: visibleFeatureTabs,
                settingsTabs: visibleSettingsTabs,
                selectedTab: selectedTab,
                selectedFeatureTab: selectedFeatureTab,
                selectedHistoryFilter: selectedHistoryFilter,
                onSelectTab: onSelectTab,
                onSelectFeatureTab: onSelectFeatureTab,
                onSelectHistoryFilter: onSelectHistoryFilter
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)

            VStack(alignment: .leading, spacing: 8) {
                if sidebarMode == .root, let modelStorageAuthorizationIssue {
                    Button(action: onTapModelStorageAuthorizationBadge) {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(.orange)
                            Text(AppLocalization.localizedString("Model Storage"))
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.orange)
                                .lineLimit(1)
                            Spacer(minLength: 0)
                            Text(AppLocalization.localizedString("Authorize"))
                                .font(.system(size: 10, weight: .bold))
                                .foregroundStyle(.orange)
                                .lineLimit(1)
                                .padding(.horizontal, 7)
                                .frame(height: 19)
                                .background(
                                    Capsule(style: .continuous)
                                        .fill(Color.orange.opacity(0.14))
                                )
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .buttonStyle(SettingsStatusButtonStyle(tint: .orange))
                    .help(modelStorageAuthorizationIssue)
                }

                if sidebarMode == .root, hasMissingPermissions {
                    Button(action: onTapPermissionBadge) {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(.red)
                            Text(AppLocalization.localizedString("Permissions Disabled"))
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.red)
                            Spacer(minLength: 0)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .buttonStyle(SettingsStatusButtonStyle(tint: .red))
                }

                if sidebarMode == .root, hasNoAvailableMicrophones {
                    Button(action: onTapMicrophoneBadge) {
                        HStack(spacing: 8) {
                            Image(systemName: "mic.slash.fill")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(.red)
                            Text(AppLocalization.localizedString("No Microphone Available"))
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.red)
                            Spacer(minLength: 0)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .buttonStyle(SettingsStatusButtonStyle(tint: .red))
                }

                if sidebarMode == .root, activeModelDownloadCount > 0 {
                    Button(action: onTapModelBadge) {
                        HStack(spacing: 8) {
                            ProgressView()
                                .controlSize(.small)
                                .tint(.accentColor)
                                .frame(width: 13, height: 13)
                            Text(AppLocalization.localizedString("Downloading"))
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(Color.accentColor)
                                .lineLimit(1)
                            Spacer(minLength: 0)
                            Text("\(activeModelDownloadCount)")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundStyle(Color.accentColor)
                                .padding(.horizontal, 7)
                                .frame(minWidth: 22, minHeight: 20)
                                .background(
                                    Capsule(style: .continuous)
                                        .fill(Color.accentColor.opacity(0.14))
                                )
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .buttonStyle(SettingsStatusButtonStyle(tint: .accentColor))
                }

                if sidebarMode == .root, hasMissingModelConfigurationIssues {
                    Button(action: onTapModelBadge) {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.circle.fill")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(.orange)
                            Text(AppLocalization.localizedString("Model Setup Required"))
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.orange)
                            Spacer(minLength: 0)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .buttonStyle(SettingsStatusButtonStyle(tint: .orange))
                }

                SettingsSidebarInfoBlock(
                    onTapWebsite: onTapWebsite,
                    onTapNotification: onTapNotification,
                    hasUnreadNotification: hasUnreadNotification,
                    onTapFeedback: onTapFeedback
                )
            }
            .frame(maxWidth: .infinity)

        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var visibleRootTabs: [SettingsTab] {
        SettingsTab.visibleTabs(appEnhancementEnabled: featureAvailability.appEnhancementEnabled)
    }

    private var visibleFeatureTabs: [FeatureSettingsTab] {
        FeatureSettingsTab.visibleTabs(availability: featureAvailability)
    }

    private var visibleSettingsTabs: [SettingsTab] {
        SettingsTab.settingsTabs
    }
}

private struct SettingsSidebarMenuPager: View {
    let sidebarMode: SettingsSidebarMode
    let rootTabs: [SettingsTab]
    let featureTabs: [FeatureSettingsTab]
    let settingsTabs: [SettingsTab]
    let selectedTab: SettingsTab
    let selectedFeatureTab: FeatureSettingsTab
    let selectedHistoryFilter: HistoryFilterTab
    let onSelectTab: (SettingsTab) -> Void
    let onSelectFeatureTab: (FeatureSettingsTab) -> Void
    let onSelectHistoryFilter: (HistoryFilterTab) -> Void

    @State private var visibleSubmenuKind: SettingsSidebarSubmenuKind = .feature
    @State private var visiblePageIndex: CGFloat = 0

    var body: some View {
        GeometryReader { proxy in
            let pageWidth = proxy.size.width

            HStack(alignment: .top, spacing: 0) {
                rootMenu
                    .frame(width: pageWidth, alignment: .topLeading)

                subMenu(kind: visibleSubmenuKind)
                    .frame(width: pageWidth, alignment: .topLeading)
            }
            .frame(width: pageWidth * 2, alignment: .leading)
            .offset(x: -pageWidth * visiblePageIndex)
            .onAppear {
                if let submenuKind = sidebarMode.submenuKind {
                    visibleSubmenuKind = submenuKind
                    visiblePageIndex = 1
                } else {
                    visiblePageIndex = 0
                }
            }
            .onChange(of: sidebarMode) { oldMode, newMode in
                if let submenuKind = newMode.submenuKind {
                    updateVisibleSubmenuKind(submenuKind)
                    animateToSubmenu()
                } else {
                    if oldMode != .root, let submenuKind = oldMode.submenuKind {
                        updateVisibleSubmenuKind(submenuKind)
                    }
                    animateToRoot()
                }
            }
        }
        .clipped()
    }

    private func updateVisibleSubmenuKind(_ submenuKind: SettingsSidebarSubmenuKind) {
        var transaction = Transaction()
        transaction.animation = nil
        withTransaction(transaction) {
            visibleSubmenuKind = submenuKind
        }
    }

    private func animateToSubmenu() {
        withAnimation(.easeInOut(duration: 0.22)) {
            visiblePageIndex = 1
        }
    }

    private func animateToRoot() {
        withAnimation(.easeInOut(duration: 0.22)) {
            visiblePageIndex = 0
        }
    }

    private var rootMenu: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(rootTabs) { tab in
                SettingsSidebarTabButton(
                    iconKind: tab.sidebarIconKind,
                    title: tab.titleKey,
                    isActive: tab == selectedTab,
                    action: { onSelectTab(tab) }
                )
            }

            SettingsSidebarTabButton(
                iconKind: .settings,
                title: "Settings",
                isActive: false,
                action: { onSelectTab(.general) }
            )
        }
    }

    @ViewBuilder
    private func subMenu(kind: SettingsSidebarSubmenuKind) -> some View {
        switch kind {
        case .feature:
            featureMenu
        case .history:
            historyMenu
        case .settings:
            settingsMenu
        }
    }

    private var featureMenu: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(featureTabs) { tab in
                SettingsSidebarTabButton(
                    iconKind: tab.sidebarIconKind,
                    systemImageName: tab.sidebarIconKind == nil ? tab.iconName : nil,
                    title: tab.titleKey,
                    badgeText: (tab == .meeting || tab == .note) ? "Experimental" : nil,
                    isActive: tab == selectedFeatureTab,
                    action: { onSelectFeatureTab(tab) }
                )
            }
        }
    }

    private var settingsMenu: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(settingsTabs) { tab in
                SettingsSidebarTabButton(
                    iconKind: tab.sidebarIconKind,
                    title: tab.titleKey,
                    isActive: tab == selectedTab,
                    action: { onSelectTab(tab) }
                )
            }
        }
    }

    private var historyMenu: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(HistoryFilterTab.allCases) { filter in
                let featureTab = filter.correspondingFeatureTab
                SettingsSidebarTabButton(
                    iconKind: featureTab.sidebarIconKind,
                    systemImageName: featureTab.sidebarIconKind == nil ? featureTab.iconName : nil,
                    title: filter.titleKey,
                    isActive: filter == selectedHistoryFilter,
                    action: { onSelectHistoryFilter(filter) }
                )
            }
        }
    }
}

private enum SettingsSidebarSubmenuKind {
    case feature
    case history
    case settings
}

private extension SettingsSidebarMode {
    var submenuKind: SettingsSidebarSubmenuKind? {
        switch self {
        case .root:
            return nil
        case .feature:
            return .feature
        case .history:
            return .history
        case .settings:
            return .settings
        }
    }
}

private struct SettingsSidebarTabButton: View {
    let iconKind: SettingsSidebarIconKind?
    var systemImageName: String?
    let title: LocalizedStringKey
    var badgeText: String? = nil
    let isActive: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                if let iconKind {
                    SettingsSidebarIconView(kind: iconKind)
                        .frame(width: SettingsUIStyle.sidebarItemIconWidth)
                } else if let systemImageName {
                    Image(systemName: systemImageName)
                        .font(.system(size: 13, weight: .semibold))
                        .frame(width: SettingsUIStyle.sidebarItemIconWidth)
                }

                Text(title)
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .allowsTightening(true)
                    .layoutPriority(1)

                if let badgeText {
                    FeatureStatusBadge(text: AppLocalization.localizedString(badgeText))
                        .fixedSize()
                }

                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(SettingsSidebarItemButtonStyle(isActive: isActive))
    }
}
