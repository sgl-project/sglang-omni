import SwiftUI

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// Shared app and keyboard colors, with matching light and dark appearances.
public enum VoxtTheme {
    public enum Colors {
        public static let ink = adaptive(light: 0x1C1C1E, dark: 0xF2F2F5)
        public static let onInk = adaptive(light: 0xFFFFFF, dark: 0x1C1C1E)
        public static let accent = adaptive(light: 0x2B5BF6, dark: 0x819FFF)
        public static let card = adaptive(light: 0xFFFFFF, dark: 0x1C1C1E)
        public static let screenBackground = adaptive(light: 0xFFFFFF, dark: 0x101012)
        public static let panel = adaptive(light: 0xF2F2F2, dark: 0x252528)
        public static let keyGray = adaptive(light: 0xE4E5EA, dark: 0x3A3A3E)
        public static let caption = adaptive(light: 0x555559, dark: 0xC7C7CC)

        private static func adaptive(light: UInt32, dark: UInt32) -> Color {
            #if canImport(UIKit)
            func native(_ hex: UInt32) -> UIColor {
                UIColor(red: CGFloat((hex >> 16) & 255) / 255,
                        green: CGFloat((hex >> 8) & 255) / 255,
                        blue: CGFloat(hex & 255) / 255, alpha: 1)
            }
            return Color(uiColor: UIColor { traits in
                native(traits.userInterfaceStyle == .dark ? dark : light)
            })
            #else
            func native(_ hex: UInt32) -> NSColor {
                NSColor(srgbRed: CGFloat((hex >> 16) & 255) / 255,
                        green: CGFloat((hex >> 8) & 255) / 255,
                        blue: CGFloat(hex & 255) / 255, alpha: 1)
            }
            return Color(nsColor: NSColor(name: nil) { appearance in
                native(appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light)
            })
            #endif
        }
    }

    public enum Typography {
        public static let screenTitle = Font.system(size: 34, weight: .bold)
        public static let cardTitle = Font.system(size: 17, weight: .semibold)
        public static let body = Font.system(size: 16)
        public static let caption = Font.system(size: 14)
    }
}

/// Capsule button for primary actions.
public struct BlackPillButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 17, weight: .semibold))
            .foregroundStyle(VoxtTheme.Colors.onInk)
            .frame(maxWidth: .infinity)
            .frame(height: 52)
            .background(VoxtTheme.Colors.ink, in: Capsule())
            .opacity(isEnabled ? (configuration.isPressed ? 0.8 : 1) : 0.4)
    }
}

/// Rounded card that follows the current appearance.
public struct CardModifier: ViewModifier {
    public init() {}

    public func body(content: Content) -> some View {
        content
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(VoxtTheme.Colors.card, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(VoxtTheme.Colors.keyGray.opacity(0.45), lineWidth: 0.5))
    }
}

extension View {
    public func voxtCard() -> some View {
        modifier(CardModifier())
    }
}
