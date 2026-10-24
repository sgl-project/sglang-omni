// GeneralSettingsDataTests.swift
// Provides General Settings Data Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class GeneralSettingsDataTests: XCTestCase {
    func testUserMainLanguageSummaryFallsBackWhenSelectionIsEmpty() {
        let summary = GeneralSettingsData.userMainLanguageSummary(
            selectedCodes: [],
            locale: Locale(identifier: "en")
        )

        XCTAssertEqual(summary, UserMainLanguageOption.fallbackOption().title(locale: Locale(identifier: "en")))
    }

    func testUserMainLanguageSummaryIncludesOverflowCount() {
        let locale = Locale(identifier: "en")
        let summary = GeneralSettingsData.userMainLanguageSummary(
            selectedCodes: ["zh-hans", "en", "ja"],
            locale: locale
        )

        let expected = AppLocalization.format(
            "%@ + %d more",
            UserMainLanguageOption.option(for: "zh-hans")?.title(locale: locale) ?? "Chinese (Simplified)",
            2
        )
        XCTAssertEqual(summary, expected)
    }

    func testCustomPasteHotkeyFiltersSidedModifiersByEnabledModifiers() {
        let hotkey = GeneralSettingsData.customPasteHotkey(
            keyCode: 9,
            modifiersRawValue: Int(NSEvent.ModifierFlags.command.rawValue),
            sidedModifiersRawValue: Int(SidedModifierFlags.leftOption.rawValue | SidedModifierFlags.leftCommand.rawValue)
        )

        XCTAssertEqual(hotkey.modifiers, [.command])
        XCTAssertEqual(hotkey.sidedModifiers, [.leftCommand])
    }

    func testProxyTitlesAndOverlayClamping() {
        XCTAssertEqual(GeneralSettingsData.networkProxyModeTitle(.system), AppLocalization.localizedString("Follow System"))
        XCTAssertEqual(GeneralSettingsData.networkProxyModeTitle(.disabled), AppLocalization.localizedString("Off"))
        XCTAssertEqual(GeneralSettingsData.networkProxyModeTitle(.custom), AppLocalization.localizedString("Custom"))
        XCTAssertEqual(GeneralSettingsData.proxySchemeTitle(.http), "HTTP")
        XCTAssertEqual(GeneralSettingsData.proxySchemeTitle(.https), "HTTPS")
        XCTAssertEqual(GeneralSettingsData.proxySchemeTitle(.socks5), "SOCKS5")
        XCTAssertEqual(GeneralSettingsData.clampedOverlayOpacity(140), 100)
        XCTAssertEqual(GeneralSettingsData.clampedOverlayOpacity(-8), 0)
        XCTAssertEqual(GeneralSettingsData.clampedOverlayCornerRadius(55), 40)
        XCTAssertEqual(GeneralSettingsData.clampedOverlayScreenEdgeInset(-1), 0)
    }
}
