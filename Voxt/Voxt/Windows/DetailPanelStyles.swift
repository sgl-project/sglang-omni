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
