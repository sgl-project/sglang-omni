// SettingsButtonStyles.swift
// Provides Settings Button Styles for settings shell.

import SwiftUI

struct SettingsSidebarItemButtonStyle: ButtonStyle {
    var isActive: Bool

    func makeBody(configuration: Configuration) -> some View {
        SettingsSidebarItemButtonBody(configuration: configuration, isActive: isActive)
    }
}

private struct SettingsSidebarItemButtonBody: View {
    let configuration: SettingsSidebarItemButtonStyle.Configuration
    let isActive: Bool
    @State private var isHovered = false

    var body: some View {
        configuration.label
            .font(.system(size: 12.5, weight: .medium))
            .foregroundStyle(Color.primary.opacity(configuration.isPressed ? 0.78 : 0.94))
            .padding(.horizontal, SettingsUIStyle.sidebarItemHorizontalPadding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(height: SettingsUIStyle.sidebarItemHeight)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(backgroundFill)
            )
            .contentShape(Rectangle())
            .opacity(configuration.isPressed ? 0.9 : 1)
            .onHover { isHovered = $0 }
    }

    private var backgroundFill: Color {
        if configuration.isPressed {
            return SettingsUIStyle.sidebarItemPressedFillColor
        }
        if isActive || isHovered {
            return SettingsUIStyle.sidebarItemFillColor
        }
        return .clear
    }
}

struct SettingsPillButtonStyle: ButtonStyle {
    enum Tone {
        case neutral
        case destructive
    }

    var tone: Tone = .neutral
    var horizontalPadding: CGFloat = 12
    var height: CGFloat = 32

    func makeBody(configuration: Configuration) -> some View {
        SettingsPillButtonBody(
            configuration: configuration,
            tone: tone,
            horizontalPadding: horizontalPadding,
            height: height
        )
    }
}

private struct SettingsPillButtonBody: View {
    let configuration: SettingsPillButtonStyle.Configuration
    let tone: SettingsPillButtonStyle.Tone
    let horizontalPadding: CGFloat
    let height: CGFloat

    @State private var isHovered = false

    var body: some View {
        let foreground: Color = tone == .destructive ? .red : .primary

        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(foreground.opacity(configuration.isPressed ? 0.72 : 0.92))
            .padding(.horizontal, horizontalPadding)
            .frame(height: height)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(fill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(stroke, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .opacity(configuration.isPressed ? 0.92 : 1)
            .onHover { isHovered = $0 }
    }

    private var fill: Color {
        if tone == .destructive {
            return .red.opacity(configuration.isPressed ? 0.16 : isHovered ? 0.13 : 0.10)
        }
        if configuration.isPressed {
            return SettingsUIStyle.sidebarItemPressedFillColor
        }
        return isHovered ? SettingsUIStyle.sidebarItemFillColor : SettingsUIStyle.subtleFillColor
    }

    private var stroke: Color {
        if tone == .destructive {
            return .red.opacity(isHovered ? 0.30 : 0.22)
        }
        return isHovered ? SettingsUIStyle.controlHoverBorderColor : SettingsUIStyle.subtleBorderColor
    }
}

struct SettingsPrimaryButtonStyle: ButtonStyle {
    var horizontalPadding: CGFloat = 14
    var height: CGFloat = 34

    func makeBody(configuration: Configuration) -> some View {
        SettingsPrimaryButtonBody(
            configuration: configuration,
            horizontalPadding: horizontalPadding,
            height: height
        )
    }
}

private struct SettingsPrimaryButtonBody: View {
    let configuration: SettingsPrimaryButtonStyle.Configuration
    let horizontalPadding: CGFloat
    let height: CGFloat

    @Environment(\.isEnabled) private var isEnabled
    @State private var isHovered = false

    var body: some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(foreground)
            .padding(.horizontal, horizontalPadding)
            .frame(height: height)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(background)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(stroke, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .opacity(isEnabled && configuration.isPressed ? 0.92 : 1)
            .onHover { isHovered = isEnabled && $0 }
    }

    private var foreground: Color {
        if !isEnabled {
            return .secondary.opacity(0.62)
        }
        return .white.opacity(configuration.isPressed ? 0.82 : 0.96)
    }

    private var background: Color {
        if !isEnabled {
            return Color.primary.opacity(0.055)
        }
        return Color.accentColor.opacity(configuration.isPressed ? 0.82 : isHovered ? 1 : 0.96)
    }

    private var stroke: Color {
        if !isEnabled {
            return SettingsUIStyle.subtleBorderColor.opacity(0.78)
        }
        return Color.accentColor.opacity(isHovered ? 0.38 : 0.28)
    }
}

struct SettingsCompactActionButtonStyle: ButtonStyle {
    enum Tone {
        case neutral
        case destructive
    }

    var tone: Tone = .neutral
    var height: CGFloat = 28
    var horizontalPadding: CGFloat = 9

    func makeBody(configuration: Configuration) -> some View {
        SettingsCompactActionButtonBody(
            configuration: configuration,
            tone: tone,
            height: height,
            horizontalPadding: horizontalPadding
        )
    }
}

private struct SettingsCompactActionButtonBody: View {
    let configuration: SettingsCompactActionButtonStyle.Configuration
    let tone: SettingsCompactActionButtonStyle.Tone
    let height: CGFloat
    let horizontalPadding: CGFloat

    @State private var isHovered = false

    var body: some View {
        let foreground: Color = tone == .destructive ? .red : .primary

        return configuration.label
            .font(.system(size: 11.5, weight: .semibold))
            .foregroundStyle(foreground.opacity(configuration.isPressed ? 0.8 : 0.92))
            .padding(.horizontal, horizontalPadding)
            .frame(height: height)
            .background(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(fill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(stroke, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
            .opacity(configuration.isPressed ? 0.94 : 1)
            .onHover { isHovered = $0 }
    }

    private var fill: Color {
        if tone == .destructive {
            return .red.opacity(configuration.isPressed ? 0.16 : isHovered ? 0.13 : 0.10)
        }
        if configuration.isPressed {
            return SettingsUIStyle.sidebarItemPressedFillColor
        }
        return isHovered ? SettingsUIStyle.sidebarItemFillColor : SettingsUIStyle.subtleFillColor
    }

    private var stroke: Color {
        if tone == .destructive {
            return .red.opacity(isHovered ? 0.30 : 0.22)
        }
        return isHovered ? SettingsUIStyle.controlHoverBorderColor : SettingsUIStyle.subtleBorderColor
    }
}

struct SettingsCompactIconButtonStyle: ButtonStyle {
    var tone: SettingsCompactActionButtonStyle.Tone = .neutral
    var size: CGFloat = 28

    func makeBody(configuration: Configuration) -> some View {
        SettingsCompactIconButtonBody(
            configuration: configuration,
            tone: tone,
            size: size
        )
    }
}

private struct SettingsCompactIconButtonBody: View {
    let configuration: SettingsCompactIconButtonStyle.Configuration
    let tone: SettingsCompactActionButtonStyle.Tone
    let size: CGFloat

    @State private var isHovered = false

    var body: some View {
        configuration.label
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(foreground)
            .frame(width: size, height: size)
            .background(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(fill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(stroke, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
            .opacity(configuration.isPressed ? 0.92 : 1)
            .onHover { isHovered = $0 }
    }

    private var foreground: Color {
        tone == .destructive ? .red : .secondary
    }

    private var fill: Color {
        if tone == .destructive {
            return .red.opacity(configuration.isPressed ? 0.16 : isHovered ? 0.13 : 0.10)
        }
        if configuration.isPressed {
            return SettingsUIStyle.sidebarItemPressedFillColor
        }
        return isHovered ? SettingsUIStyle.sidebarItemFillColor : SettingsUIStyle.subtleFillColor
    }

    private var stroke: Color {
        if tone == .destructive {
            return .red.opacity(isHovered ? 0.30 : 0.22)
        }
        return isHovered ? SettingsUIStyle.controlHoverBorderColor : SettingsUIStyle.subtleBorderColor
    }
}

struct SettingsDialogCloseButtonStyle: ButtonStyle {
    var size: CGFloat = 28

    func makeBody(configuration: Configuration) -> some View {
        SettingsDialogCloseButtonBody(
            configuration: configuration,
            size: size
        )
    }
}

private struct SettingsDialogCloseButtonBody: View {
    let configuration: SettingsDialogCloseButtonStyle.Configuration
    let size: CGFloat

    @State private var isHovered = false

    var body: some View {
        configuration.label
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(SettingsUIStyle.dialogCloseButtonForegroundColor)
            .frame(width: size, height: size)
            .background(
                Circle()
                    .fill(isPressedOrHovering ? closeHoverFill : .clear)
            )
            .contentShape(Circle())
            .opacity(configuration.isPressed ? 0.88 : 1)
            .focusEffectDisabled()
            .onHover { isHovered = $0 }
    }

    private var isPressedOrHovering: Bool {
        configuration.isPressed || isHovered
    }

    private var closeHoverFill: Color {
        SettingsUIStyle.dialogCloseButtonHoverFillColor(isPressed: configuration.isPressed)
    }
}

struct SettingsInlineSelectorButtonStyle: ButtonStyle {
    var isEmphasized = false

    func makeBody(configuration: Configuration) -> some View {
        SettingsInlineSelectorButtonBody(configuration: configuration, isEmphasized: isEmphasized)
    }
}

private struct SettingsInlineSelectorButtonBody: View {
    let configuration: SettingsInlineSelectorButtonStyle.Configuration
    let isEmphasized: Bool

    @State private var isHovered = false

    var body: some View {
        configuration.label
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(isEmphasized ? Color.primary : Color.primary.opacity(0.92))
            .padding(.horizontal, 10)
            .frame(height: 32)
            .background(
                RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                    .fill(configuration.isPressed ? SettingsUIStyle.sidebarItemPressedFillColor : isHovered ? SettingsUIStyle.sidebarItemFillColor : SettingsUIStyle.subtleFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                    .strokeBorder(isHovered ? SettingsUIStyle.controlHoverBorderColor : SettingsUIStyle.subtleBorderColor.opacity(isEmphasized ? 1 : 0.92), lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous))
            .opacity(configuration.isPressed ? 0.92 : 1)
            .onHover { isHovered = $0 }
    }
}

struct SettingsStatusButtonStyle: ButtonStyle {
    let tint: Color

    func makeBody(configuration: Configuration) -> some View {
        SettingsStatusButtonBody(configuration: configuration, tint: tint)
    }
}

private struct SettingsStatusButtonBody: View {
    let configuration: SettingsStatusButtonStyle.Configuration
    let tint: Color

    @State private var isHovered = false

    var body: some View {
        configuration.label
            .padding(.horizontal, 10)
            .frame(height: 32)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(tint.opacity(configuration.isPressed ? 0.16 : isHovered ? 0.13 : 0.10))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(tint.opacity(isHovered ? 0.34 : 0.28), lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .opacity(configuration.isPressed ? 0.9 : 1)
            .onHover { isHovered = $0 }
    }
}

struct SettingsSegmentedButtonStyle: ButtonStyle {
    let isSelected: Bool

    func makeBody(configuration: Configuration) -> some View {
        SettingsSegmentedButtonBody(configuration: configuration, isSelected: isSelected)
    }
}

private struct SettingsSegmentedButtonBody: View {
    let configuration: SettingsSegmentedButtonStyle.Configuration
    let isSelected: Bool

    @State private var isHovered = false

    var body: some View {
        configuration.label
            .font(.system(size: 11.5, weight: .semibold))
            .foregroundStyle(isSelected || isHovered ? Color.accentColor : Color.secondary)
            .padding(.horizontal, 10)
            .frame(height: 28)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(isSelected || isHovered ? Color.accentColor.opacity(0.14) : .clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(isSelected || isHovered ? Color.accentColor.opacity(0.4) : .clear, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .opacity(configuration.isPressed ? 0.9 : 1)
            .onHover { isHovered = $0 }
    }
}
