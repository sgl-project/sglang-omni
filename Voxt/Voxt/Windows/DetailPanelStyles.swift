// DetailPanelStyles.swift
// Provides Detail Panel Styles for window and overlay UI.

import AppKit
import SwiftUI

enum DetailPanelUIStyle {
    static let windowCornerRadius: CGFloat = 22
    static let panelCornerRadius: CGFloat = 16
    static let compactCornerRadius: CGFloat = 12

    static var windowFillColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.windowBackgroundColor,
            dark: NSColor(calibratedWhite: 0.14, alpha: 1)
        ))
    }

    static var panelFillColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.windowBackgroundColor,
            dark: NSColor(calibratedWhite: 0.165, alpha: 1)
        ))
    }

    static var controlFillColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor(calibratedWhite: 0.965, alpha: 1),
            dark: NSColor(calibratedWhite: 0.205, alpha: 1)
        ))
    }

    static var mutedFillColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black.withAlphaComponent(0.04),
            dark: NSColor.white.withAlphaComponent(0.045)
        ))
    }

    static var borderColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black.withAlphaComponent(0.06),
            dark: NSColor.white.withAlphaComponent(0.10)
        ))
    }

    static var softBorderColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black.withAlphaComponent(0.05),
            dark: NSColor.white.withAlphaComponent(0.08)
        ))
    }

    static var dividerColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black.withAlphaComponent(0.08),
            dark: NSColor.white.withAlphaComponent(0.10)
        ))
    }

    static var primaryButtonFillColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black.withAlphaComponent(0.92),
            dark: NSColor(calibratedWhite: 0.26, alpha: 1)
        ))
    }

    static var primaryButtonPressedFillColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black.withAlphaComponent(0.86),
            dark: NSColor(calibratedWhite: 0.30, alpha: 1)
        ))
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

struct DetailPanelSurface: ViewModifier {
    let cornerRadius: CGFloat

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(DetailPanelUIStyle.panelFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
            )
    }
}

extension View {
    func detailPanelSurface(cornerRadius: CGFloat) -> some View {
        modifier(DetailPanelSurface(cornerRadius: cornerRadius))
    }
}

struct DetailToolbarButtonStyle: ButtonStyle {
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
                        isActive ? Color.accentColor.opacity(0.4) : DetailPanelUIStyle.borderColor,
                        lineWidth: 1
                    )
            )
            .opacity(configuration.isPressed ? 0.88 : 1)
    }
}

struct DetailPillButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(.primary.opacity(configuration.isPressed ? 0.72 : 0.92))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(DetailPanelUIStyle.controlFillColor.opacity(configuration.isPressed ? 0.86 : 1))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
            )
    }
}

struct DetailPrimaryButtonStyle: ButtonStyle {
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

struct DetailPrimaryIconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .frame(width: 34, height: 34)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(configuration.isPressed ? DetailPanelUIStyle.primaryButtonPressedFillColor : DetailPanelUIStyle.primaryButtonFillColor)
            )
            .opacity(configuration.isPressed ? 0.92 : 1)
    }
}
