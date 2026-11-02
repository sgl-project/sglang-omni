// GeneralSettingsData.swift
// Provides General Settings Data for settings screens.

import Foundation
import AppKit

enum GeneralSettingsData {
    static func userMainLanguageSummary(
        selectedCodes: [String],
        locale: Locale = AppLocalization.locale
    ) -> String {
        guard let primaryCode = selectedCodes.first,
              let primaryOption = UserMainLanguageOption.option(for: primaryCode)
        else {
            return UserMainLanguageOption.fallbackOption().title(locale: locale)
        }

        if selectedCodes.count == 1 {
            return primaryOption.title(locale: locale)
        }

        let format = AppLocalization.localizedString("%@ + %d more")
        return String(format: format, primaryOption.title(locale: locale), selectedCodes.count - 1)
    }

    static func customPasteHotkey(
        keyCode: Int,
        modifiersRawValue: Int,
        sidedModifiersRawValue: Int
    ) -> HotkeyPreference.Hotkey {
        let modifiers = NSEvent.ModifierFlags(rawValue: UInt(modifiersRawValue)).intersection(.hotkeyRelevant)
        return HotkeyPreference.Hotkey(
            keyCode: UInt16(keyCode),
            modifiers: modifiers,
            sidedModifiers: SidedModifierFlags(rawValue: sidedModifiersRawValue).filtered(by: modifiers)
        )
    }

    static func customPasteHotkeyDisplayString(
        keyCode: Int,
        modifiersRawValue: Int,
        sidedModifiersRawValue: Int,
        distinguishModifierSides: Bool
    ) -> String {
        HotkeyPreference.displayString(
            for: customPasteHotkey(
                keyCode: keyCode,
                modifiersRawValue: modifiersRawValue,
                sidedModifiersRawValue: sidedModifiersRawValue
            ),
            distinguishModifierSides: distinguishModifierSides
        )
    }

    static func networkProxyModeTitle(_ mode: VoxtNetworkSession.ProxyMode) -> String {
        switch mode {
        case .system:
            return AppLocalization.localizedString("Follow System")
        case .disabled:
            return AppLocalization.localizedString("Off")
        case .custom:
            return AppLocalization.localizedString("Custom")
        }
    }

    static func proxySchemeTitle(_ scheme: VoxtNetworkSession.ProxyScheme) -> String {
        switch scheme {
        case .http:
            return "HTTP"
        case .https:
            return "HTTPS"
        case .socks5:
            return "SOCKS5"
        }
    }

    static func clampedOverlayOpacity(_ value: Int) -> Int {
        min(max(value, 0), 100)
    }

    static func clampedOverlayCornerRadius(_ value: Int) -> Int {
        min(max(value, 0), 40)
    }

    static func clampedOverlayScreenEdgeInset(_ value: Int) -> Int {
        min(max(value, 0), 120)
    }
}
