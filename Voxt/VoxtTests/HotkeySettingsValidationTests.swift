// HotkeySettingsValidationTests.swift
// Provides Hotkey Settings Validation Tests for Voxt test coverage.

import XCTest
import AppKit
import Carbon
@testable import Voxt

final class HotkeySettingsValidationTests: XCTestCase {
    func testBareKeyboardShortcutIsAllowed() {
        let messages = HotkeySettingsValidation.messages(
            for: .init(
                transcriptionBindings: [.init(
                    hotkey: HotkeyPreference.Hotkey(
                        keyCode: UInt16(kVK_ANSI_X),
                        modifiers: [],
                        sidedModifiers: []
                    ),
                    behavior: .doubleTap
                )],
                translationBindings: [.init(
                    hotkey: HotkeyPreference.Hotkey(
                        keyCode: HotkeyPreference.modifierOnlyKeyCode,
                        modifiers: [.function, .shift],
                        sidedModifiers: []
                    ),
                    behavior: .tap
                )],
                meetingBindings: [.init(hotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 4), behavior: .tap)],
                rewriteBindings: [.init(hotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 5), behavior: .tap)],
                customPasteHotkey: nil
            )
        )

        XCTAssertFalse(messages.contains { $0.id == "invalid.transcription.0" })
    }

    func testDoubleTapWakeSuppressesRewriteSpecificMessages() {
        let transcriptionHotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )

        let messages = HotkeySettingsValidation.messages(
            for: .init(
                transcriptionHotkey: transcriptionHotkey,
                translationHotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.function, .shift],
                    sidedModifiers: []
                ),
                rewriteHotkey: transcriptionHotkey,
                shouldValidateRewriteHotkey: false,
                customPasteHotkey: nil
            )
        )

        XCTAssertFalse(messages.contains {
            $0.text == AppLocalization.localizedString(
                "Transcription and content rewrite shortcuts should be different."
            )
        })
        XCTAssertFalse(messages.contains {
            $0.id == "conflict.rewrite"
        })
    }

    func testDedicatedRewriteIncludesRewriteDuplicateMessage() {
        let transcriptionHotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )

        let messages = HotkeySettingsValidation.messages(
            for: .init(
                transcriptionHotkey: transcriptionHotkey,
                translationHotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.function, .shift],
                    sidedModifiers: []
                ),
                rewriteHotkey: transcriptionHotkey,
                shouldValidateRewriteHotkey: true,
                customPasteHotkey: nil
            )
        )

        XCTAssertTrue(messages.contains {
            $0.text == AppLocalization.localizedString(
                "Transcription and content rewrite shortcuts should be different."
            )
        })
    }

    func testConflictMessageIsReturnedForEnabledCustomPasteShortcut() {
        let customPasteHotkey = HotkeyPreference.Hotkey(
            keyCode: UInt16(kVK_ANSI_V),
            modifiers: [.command],
            sidedModifiers: []
        )

        let messages = HotkeySettingsValidation.messages(
            for: .init(
                transcriptionHotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.function],
                    sidedModifiers: []
                ),
                translationHotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.function, .shift],
                    sidedModifiers: []
                ),
                rewriteHotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.function, .control],
                    sidedModifiers: []
                ),
                shouldValidateRewriteHotkey: true,
                customPasteHotkey: customPasteHotkey
            )
        )

        XCTAssertTrue(messages.contains {
            $0.id == "conflict.customPaste" &&
            $0.text == AppLocalization.format(
                "Custom paste shortcut: %@",
                AppLocalization.localizedString("Conflicts with Paste (⌘V).")
            )
        })
    }

    func testMouseShortcutDuplicateUsesButtonAndModifiers() {
        let transcriptionHotkey = HotkeyPreference.Hotkey(
            mouseButtonNumber: 4,
            modifiers: [.command],
            sidedModifiers: []
        )

        let messages = HotkeySettingsValidation.messages(
            for: .init(
                transcriptionHotkey: transcriptionHotkey,
                translationHotkey: transcriptionHotkey,
                rewriteHotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 5, modifiers: [.command]),
                shouldValidateRewriteHotkey: true,
                customPasteHotkey: nil
            )
        )

        XCTAssertTrue(messages.contains { $0.id == "duplicate.transcription.translation" })
        XCTAssertFalse(messages.contains { $0.id == "duplicate.transcription.rewrite" })
    }

    func testMouseShortcutsWithDifferentModifiersDoNotConflict() {
        let messages = HotkeySettingsValidation.messages(
            for: .init(
                transcriptionHotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 4, modifiers: [.command]),
                translationHotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 4, modifiers: [.shift]),
                rewriteHotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 4, modifiers: [.option]),
                shouldValidateRewriteHotkey: true,
                customPasteHotkey: nil
            )
        )

        XCTAssertFalse(messages.contains { $0.id.hasPrefix("duplicate.") })
    }

    func testSameHotkeyWithSameBehaviorReportsConflict() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )

        let messages = HotkeySettingsValidation.messages(
            for: .init(
                transcriptionBindings: [.init(hotkey: hotkey, behavior: .tap)],
                translationBindings: [.init(hotkey: hotkey, behavior: .tap)],
                meetingBindings: [.init(hotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 4), behavior: .tap)],
                rewriteBindings: [.init(hotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 5), behavior: .tap)],
                customPasteHotkey: nil
            )
        )

        XCTAssertTrue(messages.contains { $0.id == "duplicate.transcription.translation" })
    }

    func testSameHotkeyWithDifferentBehaviorDoesNotConflict() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )

        let messages = HotkeySettingsValidation.messages(
            for: .init(
                transcriptionBindings: [.init(hotkey: hotkey, behavior: .tap)],
                translationBindings: [.init(hotkey: hotkey, behavior: .doubleTap)],
                meetingBindings: [.init(hotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 4), behavior: .tap)],
                rewriteBindings: [.init(hotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 5), behavior: .tap)],
                customPasteHotkey: nil
            )
        )

        XCTAssertFalse(messages.contains { $0.id.hasPrefix("duplicate.") })
    }

    func testSameWorkflowCannotUseSameHotkeyWithDifferentBehaviors() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )

        let messages = HotkeySettingsValidation.messages(
            for: .init(
                transcriptionBindings: [
                    .init(hotkey: hotkey, behavior: .tap),
                    .init(hotkey: hotkey, behavior: .longPress)
                ],
                translationBindings: [.init(
                    hotkey: HotkeyPreference.Hotkey(
                        keyCode: HotkeyPreference.modifierOnlyKeyCode,
                        modifiers: [.function, .shift],
                        sidedModifiers: []
                    ),
                    behavior: .tap
                )],
                meetingBindings: [.init(hotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 4), behavior: .tap)],
                rewriteBindings: [.init(hotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 5), behavior: .tap)],
                customPasteHotkey: nil
            )
        )

        XCTAssertTrue(messages.contains { $0.id == "duplicate.transcription.transcription" })
    }
}
