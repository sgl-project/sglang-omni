// HotkeyRewriteActivationStateTests.swift
// Provides Hotkey Rewrite Activation State Tests for Voxt test coverage.

import XCTest
import AppKit
@testable import Voxt

final class HotkeyRewriteActivationStateTests: XCTestCase {
    func testInvalidRawValueFallsBackToDedicatedMode() {
        let state = HotkeyRewriteActivationState(rawValue: "invalid")

        XCTAssertEqual(state.mode, .dedicatedHotkey)
        XCTAssertFalse(state.isDoubleTapWakeEnabled)
    }

    func testDisplayTextUsesTranscriptionHotkeyDisplayString() {
        let state = HotkeyRewriteActivationState(
            rawValue: HotkeyPreference.RewriteActivationMode.doubleTapTranscriptionHotkey.rawValue
        )
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )

        XCTAssertEqual(
            state.displayText(for: hotkey, distinguishModifierSides: false),
            AppLocalization.format("Double-tap %@", "fn")
        )
    }

    func testToggledModeSwitchesBetweenDedicatedAndDoubleTap() {
        XCTAssertEqual(
            HotkeyRewriteActivationState(
                rawValue: HotkeyPreference.RewriteActivationMode.dedicatedHotkey.rawValue
            ).toggledMode,
            .doubleTapTranscriptionHotkey
        )
        XCTAssertEqual(
            HotkeyRewriteActivationState(
                rawValue: HotkeyPreference.RewriteActivationMode.doubleTapTranscriptionHotkey.rawValue
            ).toggledMode,
            .dedicatedHotkey
        )
    }
}
