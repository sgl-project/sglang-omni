import XCTest
import AppKit
import Carbon
import ApplicationServices
import IOKit.hidsystem
@testable import Voxt

@MainActor
final class HotkeyManagerNoteTests: HotkeyManagerTestCase {
    func testNoteTapBindingRoutesToDedicatedCallback() {
        HotkeyPreference.saveNoteBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: UInt16(kVK_ANSI_N),
                    modifiers: [],
                    sidedModifiers: []
                ),
                behavior: .longPress
            )
        ])

        let manager = makeManager()
        var noteCount = 0
        var transcriptionCount = 0
        manager.onNoteKeyDown = { noteCount += 1 }
        manager.onKeyDown = { transcriptionCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(
            type: .keyDown,
            keyCode: UInt16(kVK_ANSI_N),
            flags: []
        ))
        XCTAssertTrue(manager.testingHandleEvent(
            type: .keyUp,
            keyCode: UInt16(kVK_ANSI_N),
            flags: []
        ))
        XCTAssertEqual(noteCount, 1)
        XCTAssertEqual(transcriptionCount, 0)
    }

    func testRightCommandNoteTapIsCanceledByExternalCommandLChord() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        HotkeyPreference.saveNoteBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.command],
                    sidedModifiers: [.rightCommand]
                ),
                behavior: .tap
            )
        ])

        let manager = makeManager()
        var noteCount = 0
        manager.onNoteKeyDown = { noteCount += 1 }
        let command = commandFlags(for: .rightCommand)

        XCTAssertFalse(manager.testingHandleEventWasConsumed(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: command
        ))
        XCTAssertFalse(manager.testingHandleEventWasConsumed(
            type: .keyDown,
            keyCode: UInt16(kVK_ANSI_L),
            flags: command
        ))
        XCTAssertFalse(manager.testingHandleEventWasConsumed(
            type: .keyUp,
            keyCode: UInt16(kVK_ANSI_L),
            flags: command
        ))
        XCTAssertFalse(manager.testingHandleEventWasConsumed(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: []
        ))

        XCTAssertEqual(noteCount, 0)

        XCTAssertFalse(manager.testingHandleEventWasConsumed(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: command
        ))
        XCTAssertFalse(manager.testingHandleEventWasConsumed(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: []
        ))
        XCTAssertEqual(noteCount, 1)
    }

    func testRightCommandNoteTapIsCanceledByUnassignedExtraModifier() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        HotkeyPreference.saveNoteBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.command],
                    sidedModifiers: [.rightCommand]
                ),
                behavior: .tap
            )
        ])

        let manager = makeManager()
        var noteCount = 0
        manager.onNoteKeyDown = { noteCount += 1 }
        let command = commandFlags(for: .rightCommand)

        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: command
        ))
        _ = manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightShift),
            flags: command.union(.maskShift)
        )
        _ = manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightShift),
            flags: command
        )
        _ = manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: []
        )

        XCTAssertEqual(noteCount, 0)
    }

    func testMoreSpecificModifierBindingCancelsRightCommandNotePrefix() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        HotkeyPreference.saveNoteBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.command],
                    sidedModifiers: [.rightCommand]
                ),
                behavior: .tap
            )
        ])
        HotkeyPreference.saveTranslationBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.command, .shift],
                    sidedModifiers: [.rightCommand, .rightShift]
                ),
                behavior: .tap
            )
        ])

        let manager = makeManager()
        var noteCount = 0
        var translationCount = 0
        manager.onNoteKeyDown = { noteCount += 1 }
        manager.onTranslationKeyDown = { translationCount += 1 }
        let command = commandFlags(for: .rightCommand)

        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: command
        ))
        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightShift),
            flags: command.union(shiftFlags(for: .rightShift))
        ))
        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightShift),
            flags: command
        ))
        _ = manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: []
        )

        XCTAssertEqual(translationCount, 1)
        XCTAssertEqual(noteCount, 0)
    }
}
