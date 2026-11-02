// MeetingDetailStyles.swift
// Provides Meeting Detail Styles for meeting detail windows.

import AppKit
import SwiftUI

enum MeetingDetailUIStyle {
    static let windowCornerRadius: CGFloat = SettingsUIStyle.panelCornerRadius
    static let panelCornerRadius: CGFloat = SettingsUIStyle.panelCornerRadius
    static let compactCornerRadius: CGFloat = SettingsUIStyle.compactCornerRadius

    static var windowFillColor: Color {
        SettingsUIStyle.windowBackgroundColor
    }

    static var panelFillColor: Color {
        SettingsUIStyle.panelFillColor
    }

    static var controlFillColor: Color {
        SettingsUIStyle.controlFillColor
    }

    static var mutedFillColor: Color {
        SettingsUIStyle.subtleFillColor
    }

    static var borderColor: Color {
        SettingsUIStyle.panelBorderColor
    }

    static var softBorderColor: Color {
        SettingsUIStyle.subtleBorderColor
    }

    static var dividerColor: Color {
        SettingsUIStyle.subtleBorderColor
    }

    static var primaryButtonFillColor: Color {
        SettingsUIStyle.primaryButtonFillColor
    }

    static var primaryButtonPressedFillColor: Color {
        SettingsUIStyle.primaryButtonPressedFillColor
    }
}

struct MeetingDetailPanelSurface: ViewModifier {
    let cornerRadius: CGFloat

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(MeetingDetailUIStyle.panelFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(MeetingDetailUIStyle.borderColor, lineWidth: 1)
            )
    }
}

extension View {
    func meetingDetailPanelSurface(cornerRadius: CGFloat) -> some View {
        modifier(MeetingDetailPanelSurface(cornerRadius: cornerRadius))
    }
}

struct MeetingToolbarButtonStyle: ButtonStyle {
    var isActive = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(isActive ? Color.white : Color.primary)
            .padding(.horizontal, 12)
            .frame(height: 32)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(
                        isActive
                            ? Color.accentColor.opacity(configuration.isPressed ? 0.84 : 1)
                            : Color.clear
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(
                        isActive ? Color.accentColor.opacity(0.4) : MeetingDetailUIStyle.borderColor,
                        lineWidth: 1
                    )
            )
            .opacity(configuration.isPressed ? 0.88 : 1)
    }
}

struct MeetingPillButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(.primary.opacity(configuration.isPressed ? 0.72 : 0.92))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(MeetingDetailUIStyle.controlFillColor.opacity(configuration.isPressed ? 0.86 : 1))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(MeetingDetailUIStyle.borderColor, lineWidth: 1)
            )
    }
}

struct MeetingPrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(.white.opacity(configuration.isPressed ? 0.82 : 0.96))
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.accentColor.opacity(configuration.isPressed ? 0.82 : 0.96))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(Color.accentColor.opacity(0.28), lineWidth: 1)
            )
    }
}
