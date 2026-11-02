import SwiftUI
import AppKit

struct SettingsSidebarInfoBlock: View {
    let onTapWebsite: () -> Void
    let onTapNotification: () -> Void
    let hasUnreadNotification: Bool
    let onTapFeedback: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Button(action: onTapWebsite) {
                HStack(spacing: 6) {
                    SettingsWebsiteIconView()
                        .frame(width: 14, height: 14)
                    Text("Voxt")
                        .font(.system(size: 12, weight: .semibold))
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }
            .buttonStyle(SettingsSidebarInfoTextButtonStyle())

            Spacer(minLength: 2)

            HStack(spacing: 1) {
                Button(action: onTapNotification) {
                    SettingsSidebarIconView(kind: .notification)
                        .frame(width: 14, height: 14)
                }
                .buttonStyle(SettingsSidebarInfoIconButtonStyle())
                .accessibilityLabel(AppLocalization.localizedString("Notifications"))
                .help(AppLocalization.localizedString("Open Notification"))
                .overlay(alignment: .topTrailing) {
                    if hasUnreadNotification {
                        Circle()
                            .fill(Color(red: 1.0, green: 0.42, blue: 0.42).opacity(0.92))
                            .frame(width: 5, height: 5)
                            .offset(x: -3, y: 3)
                            .allowsHitTesting(false)
                    }
                }

                Button(action: onTapFeedback) {
                    SettingsSidebarIconView(kind: .feedback)
                        .frame(width: 14, height: 14)
                }
                .buttonStyle(SettingsSidebarInfoIconButtonStyle())
                .accessibilityLabel(AppLocalization.localizedString("Feedback"))
                .help(AppLocalization.localizedString("Feedback"))
            }
        }
        .padding(.vertical, 4)
    }
}

private struct SettingsSidebarInfoTextButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        SettingsSidebarInfoTextButtonBody(configuration: configuration)
    }
}

private struct SettingsSidebarInfoTextButtonBody: View {
    let configuration: SettingsSidebarInfoTextButtonStyle.Configuration
    @State private var isHovered = false

    var body: some View {
        configuration.label
            .foregroundStyle(Color.primary.opacity(configuration.isPressed ? 0.72 : 0.92))
            .padding(.horizontal, 10)
            .frame(height: 28)
            .background(
                Capsule(style: .continuous)
                    .fill(backgroundFill)
            )
            .contentShape(Capsule(style: .continuous))
            .onHover { isHovered = $0 }
    }

    private var backgroundFill: Color {
        if configuration.isPressed {
            return SettingsUIStyle.sidebarItemPressedFillColor
        }
        if isHovered {
            return SettingsUIStyle.sidebarItemPressedFillColor
        }
        return SettingsUIStyle.sidebarItemFillColor
    }
}

private struct SettingsSidebarInfoIconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        SettingsSidebarInfoIconButtonBody(configuration: configuration)
    }
}

private struct SettingsSidebarInfoIconButtonBody: View {
    let configuration: SettingsSidebarInfoIconButtonStyle.Configuration
    @State private var isHovered = false

    var body: some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(Color.secondary.opacity(configuration.isPressed ? 0.72 : 1))
            .frame(width: 28, height: 28)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(backgroundFill)
            )
            .contentShape(Rectangle())
            .onHover { isHovered = $0 }
    }

    private var backgroundFill: Color {
        if configuration.isPressed {
            return SettingsUIStyle.sidebarItemPressedFillColor
        }
        if isHovered {
            return SettingsUIStyle.sidebarItemFillColor
        }
        return .clear
    }
}

struct HomeShortcutPrompt: View {
    let shortcut: String

    var body: some View {
        HStack(alignment: .center, spacing: 5) {
            let prefix = AppLocalization.localizedString("Home Shortcut Prompt Prefix")
            let suffix = AppLocalization.localizedString("Home Shortcut Prompt Suffix")
            if !prefix.isEmpty {
                Text(prefix)
            }
            Text(shortcut)
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .minimumScaleFactor(0.78)
                .padding(.horizontal, 7)
                .frame(minHeight: 18)
                .background(
                    Capsule(style: .continuous)
                        .fill(SettingsUIStyle.controlFillColor)
                )
            if !suffix.isEmpty {
                Text(suffix)
            }
        }
        .font(.system(size: 12, weight: .medium))
        .foregroundStyle(.secondary)
        .lineLimit(1)
    }
}

struct HomeFooterLinkButton: View {
    let title: String
    var showsCoffeeIcon = false
    let action: () -> Void

    @State private var isHovered = false
    @State private var isBreathingExpanded = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Text(title)

                if showsCoffeeIcon {
                    SettingsCoffeeIconView(
                        color: isHovered ? Color.primary.opacity(0.88) : .secondary,
                        size: 12
                    )
                    .scaleEffect(coffeeIconScale)
                    .opacity(coffeeIconOpacity)
                }
            }
        }
        .buttonStyle(.plain)
        .font(.system(size: 11, weight: .medium))
        .foregroundStyle(isHovered ? Color.primary.opacity(0.88) : Color.secondary)
        .animation(.easeOut(duration: 0.15), value: isHovered)
        .onHover { hovering in
            isHovered = hovering
            guard showsCoffeeIcon else { return }

            if hovering {
                isBreathingExpanded = false
                withAnimation(.easeInOut(duration: 1.05).repeatForever(autoreverses: true)) {
                    isBreathingExpanded = true
                }
            } else {
                withAnimation(.easeOut(duration: 0.15)) {
                    isBreathingExpanded = false
                }
            }
        }
    }

    private var coffeeIconScale: CGFloat {
        guard isHovered else { return 1 }
        return isBreathingExpanded ? 1.14 : 0.96
    }

    private var coffeeIconOpacity: Double {
        guard isHovered else { return 1 }
        return isBreathingExpanded ? 1 : 0.72
    }
}
