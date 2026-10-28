// SettingsUIStyles.swift
// Provides Settings UIStyles for settings shell.

import SwiftUI
import AppKit

enum SettingsUIStyle {
    static let panelCornerRadius: CGFloat = 16
    static let dialogCornerRadius: CGFloat = 6
    static let dialogPadding: CGFloat = 16
    static let modelConfigurationDialogWidth: CGFloat = 520
    static let modelConfigurationDialogMaxHeight: CGFloat = 720
    static let modelConfigurationScrollMaxHeight: CGFloat = 600
    static let compactCornerRadius: CGFloat = 12
    static let controlCornerRadius: CGFloat = 10
    static let sidebarWidth: CGFloat = 172
    static let sidebarItemHorizontalPadding: CGFloat = 14
    static let sidebarItemHeight: CGFloat = 36
    static let sidebarItemIconWidth: CGFloat = 18
    static let contentScrollIndicatorOutset: CGFloat = 10
    static let contentScrollTrailingGutter: CGFloat = 14

    static var windowBackgroundNSColor: NSColor {
        dynamicColor(
            light: NSColor(calibratedRed: 0.980, green: 0.980, blue: 0.980, alpha: 1),
            dark: NSColor(calibratedWhite: 0.09, alpha: 1)
        )
    }

    static var windowBackgroundColor: Color {
        Color(nsColor: windowBackgroundNSColor)
    }

    static var panelFillNSColor: NSColor {
        dynamicColor(
            light: NSColor.white,
            dark: NSColor(calibratedWhite: 0.125, alpha: 1)
        )
    }

    static var panelFillColor: Color {
        Color(nsColor: panelFillNSColor)
    }

    static var controlFillNSColor: NSColor {
        dynamicColor(
            light: NSColor(calibratedRed: 0.965, green: 0.965, blue: 0.965, alpha: 1),
            dark: NSColor(calibratedWhite: 0.20, alpha: 1)
        )
    }

    static var controlFillColor: Color {
        Color(nsColor: controlFillNSColor)
    }

    static var modelGroupFillNSColor: NSColor {
        dynamicColor(
            light: NSColor(calibratedRed: 0.957, green: 0.957, blue: 0.957, alpha: 1),
            dark: NSColor(calibratedWhite: 0.18, alpha: 1)
        )
    }

    static var modelGroupFillColor: Color {
        Color(nsColor: modelGroupFillNSColor)
    }

    static var modelGroupListFillNSColor: NSColor {
        dynamicColor(
            light: NSColor.white,
            dark: NSColor(calibratedWhite: 0.14, alpha: 1)
        )
    }

    static var modelGroupListFillColor: Color {
        Color(nsColor: modelGroupListFillNSColor)
    }

    static var modelCardBorderNSColor: NSColor {
        dynamicColor(
            light: NSColor.black.withAlphaComponent(0.032),
            dark: NSColor.white.withAlphaComponent(0.052)
        )
    }

    static var modelCardBorderColor: Color {
        Color(nsColor: modelCardBorderNSColor)
    }

    static var groupedFillNSColor: NSColor {
        dynamicColor(
            light: NSColor(calibratedRed: 0.980, green: 0.980, blue: 0.980, alpha: 1),
            dark: NSColor(calibratedWhite: 0.12, alpha: 1)
        )
    }

    static var groupedFillColor: Color {
        Color(nsColor: groupedFillNSColor)
    }

    static var subtleFillNSColor: NSColor {
        dynamicColor(
            light: NSColor(calibratedRed: 0.937, green: 0.937, blue: 0.937, alpha: 1),
            dark: NSColor(calibratedWhite: 0.20, alpha: 1)
        )
    }

    static var subtleFillColor: Color {
        Color(nsColor: subtleFillNSColor)
    }

    static var sidebarItemFillNSColor: NSColor {
        dynamicColor(
            light: NSColor(calibratedRed: 0.933, green: 0.933, blue: 0.933, alpha: 1),
            dark: NSColor(calibratedWhite: 0.22, alpha: 1)
        )
    }

    static var sidebarItemFillColor: Color {
        Color(nsColor: sidebarItemFillNSColor)
    }

    static var sidebarItemPressedFillNSColor: NSColor {
        dynamicColor(
            light: NSColor(calibratedRed: 0.902, green: 0.902, blue: 0.902, alpha: 1),
            dark: NSColor(calibratedWhite: 0.27, alpha: 1)
        )
    }

    static var sidebarItemPressedFillColor: Color {
        Color(nsColor: sidebarItemPressedFillNSColor)
    }

    static var subtleBorderNSColor: NSColor {
        dynamicColor(
            light: NSColor.black.withAlphaComponent(0.045),
            dark: NSColor.white.withAlphaComponent(0.075)
        )
    }

    static var subtleBorderColor: Color {
        Color(nsColor: subtleBorderNSColor)
    }

    static var controlHoverBorderNSColor: NSColor {
        dynamicColor(
            light: NSColor.black.withAlphaComponent(0.105),
            dark: NSColor.white.withAlphaComponent(0.145)
        )
    }

    static var controlHoverBorderColor: Color {
        Color(nsColor: controlHoverBorderNSColor)
    }

    static var panelBorderNSColor: NSColor {
        dynamicColor(
            light: NSColor.black.withAlphaComponent(0.06),
            dark: NSColor.white.withAlphaComponent(0.07)
        )
    }

    static var panelBorderColor: Color {
        Color(nsColor: panelBorderNSColor)
    }

    static var dialogBorderNSColor: NSColor {
        dynamicColor(
            light: NSColor.black.withAlphaComponent(0.045),
            dark: NSColor.white.withAlphaComponent(0.04)
        )
    }

    static var dialogBorderColor: Color {
        Color(nsColor: dialogBorderNSColor)
    }

    static var primaryButtonFillColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black.withAlphaComponent(0.92),
            dark: NSColor(calibratedWhite: 0.30, alpha: 1)
        ))
    }

    static var primaryButtonPressedFillColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black.withAlphaComponent(0.86),
            dark: NSColor(calibratedWhite: 0.36, alpha: 1)
        ))
    }

    static var dialogCloseButtonForegroundColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black,
            dark: NSColor.white.withAlphaComponent(0.92)
        ))
    }

    static func dialogCloseButtonHoverFillColor(isPressed: Bool) -> Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black.withAlphaComponent(isPressed ? 0.18 : 0.10),
            dark: NSColor.white.withAlphaComponent(isPressed ? 0.16 : 0.10)
        ))
    }

    static func resolvedSelectWidth(_ width: CGFloat) -> CGFloat {
        max(width - 16, 120)
    }

    private static func dynamicColor(light: NSColor, dark: NSColor) -> NSColor {
        NSColor(name: nil) { appearance in
            switch appearance.bestMatch(from: [.darkAqua, .aqua]) {
            case .darkAqua:
                return dark
            default:
                return light
            }
        }
    }
}

struct SettingsPanelSurface: ViewModifier {
    var cornerRadius: CGFloat = SettingsUIStyle.panelCornerRadius
    var fillOpacity: CGFloat = 0.76
    var backgroundColor: Color = SettingsUIStyle.panelFillColor

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(backgroundColor.opacity(fillOpacity))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(SettingsUIStyle.panelBorderColor, lineWidth: 1)
            )
    }
}

struct SettingsPanelGroupBoxStyle: GroupBoxStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.content
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct SettingsDialogChromeModifier: ViewModifier {
    var width: CGFloat?
    var height: CGFloat?
    var maxHeight: CGFloat?
    var cornerRadius: CGFloat = SettingsUIStyle.dialogCornerRadius
    var onClose: (() -> Void)?
    @Environment(\.dismiss) private var dismiss

    func body(content: Content) -> some View {
        content
            .padding(SettingsUIStyle.dialogPadding)
            .frame(width: width)
            .frame(height: height)
            .frame(maxHeight: maxHeight, alignment: .top)
            .background(SettingsUIStyle.windowBackgroundColor)
            .clipShape(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(SettingsUIStyle.dialogBorderColor, lineWidth: 0.7)
            )
            .overlay(alignment: .topTrailing) {
                Button(action: close) {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .semibold))
                }
                .buttonStyle(SettingsDialogCloseButtonStyle())
                .keyboardShortcut(.cancelAction)
                .padding(.top, SettingsUIStyle.dialogPadding)
                .padding(.trailing, SettingsUIStyle.dialogPadding)
                .help(AppLocalization.localizedString("Close"))
                .accessibilityLabel(AppLocalization.localizedString("Close"))
            }
    }

    private func close() {
        if let onClose {
            onClose()
        } else {
            dismiss()
        }
    }
}

struct SettingsFieldSurfaceModifier: ViewModifier {
    var width: CGFloat?
    var minHeight: CGFloat = 32
    var horizontalPadding: CGFloat = 10
    var alignment: Alignment = .leading

    func body(content: Content) -> some View {
        SettingsFieldSurfaceBody(
            width: width,
            minHeight: minHeight,
            horizontalPadding: horizontalPadding,
            alignment: alignment,
            content: content
        )
    }
}

private struct SettingsFieldSurfaceBody<Content: View>: View {
    var width: CGFloat?
    var minHeight: CGFloat
    var horizontalPadding: CGFloat
    var alignment: Alignment
    let content: Content

    @State private var isHovered = false

    var body: some View {
        content
            .frame(maxWidth: width == nil ? .infinity : nil, alignment: alignment)
            .frame(width: width, alignment: alignment)
            .padding(.horizontal, horizontalPadding)
            .frame(minHeight: minHeight, alignment: alignment)
            .background(
                RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                    .fill(SettingsUIStyle.controlFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                    .strokeBorder(isHovered ? SettingsUIStyle.controlHoverBorderColor : SettingsUIStyle.subtleBorderColor, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous))
            .onHover { isHovered = $0 }
    }
}

struct SettingsPromptEditorModifier: ViewModifier {
    var height: CGFloat
    var contentPadding: CGFloat

    func body(content: Content) -> some View {
        SettingsPromptEditorBody(height: height, contentPadding: contentPadding, content: content)
    }
}

private struct SettingsPromptEditorBody<Content: View>: View {
    var height: CGFloat
    var contentPadding: CGFloat
    let content: Content

    @State private var isHovered = false

    var body: some View {
        content
            .font(.system(size: 13, weight: .medium, design: .monospaced))
            .lineSpacing(4)
            .frame(height: height)
            .scrollContentBackground(.hidden)
            .padding(contentPadding)
            .background(
                RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                    .fill(SettingsUIStyle.controlFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                    .strokeBorder(isHovered ? SettingsUIStyle.controlHoverBorderColor : SettingsUIStyle.subtleBorderColor, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous))
            .onHover { isHovered = $0 }
    }
}

extension View {
    func settingsPanelSurface(
        cornerRadius: CGFloat = SettingsUIStyle.panelCornerRadius,
        fillOpacity: CGFloat = 0.76
    ) -> some View {
        modifier(SettingsPanelSurface(cornerRadius: cornerRadius, fillOpacity: fillOpacity))
    }

    func settingsCardSurface(
        cornerRadius: CGFloat = SettingsUIStyle.panelCornerRadius,
        fillOpacity: CGFloat = 0.92
    ) -> some View {
        modifier(
            SettingsPanelSurface(
                cornerRadius: cornerRadius,
                fillOpacity: fillOpacity,
                backgroundColor: SettingsUIStyle.controlFillColor
            )
        )
    }

    func settingsFieldSurface(
        width: CGFloat? = nil,
        minHeight: CGFloat = 32,
        horizontalPadding: CGFloat = 10,
        alignment: Alignment = .leading
    ) -> some View {
        modifier(
            SettingsFieldSurfaceModifier(
                width: width,
                minHeight: minHeight,
                horizontalPadding: horizontalPadding,
                alignment: alignment
            )
        )
    }

    func settingsPromptEditor(height: CGFloat, contentPadding: CGFloat = 8) -> some View {
        modifier(SettingsPromptEditorModifier(height: height, contentPadding: contentPadding))
    }

    func settingsDialogChrome(
        width: CGFloat? = nil,
        height: CGFloat? = nil,
        maxHeight: CGFloat? = nil,
        cornerRadius: CGFloat = SettingsUIStyle.dialogCornerRadius,
        onClose: (() -> Void)? = nil
    ) -> some View {
        modifier(
            SettingsDialogChromeModifier(
                width: width,
                height: height,
                maxHeight: maxHeight,
                cornerRadius: cornerRadius,
                onClose: onClose
            )
        )
    }

}
