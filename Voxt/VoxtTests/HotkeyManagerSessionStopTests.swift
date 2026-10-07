import XCTest
import AppKit
import Carbon
import ApplicationServices
import IOKit.hidsystem
@testable import Voxt

@MainActor
final class HotkeyManagerSessionStopTests: HotkeyManagerTestCase {
    func testMultipleSingleModifierTranscriptionBindingsActAsCommonStopKeysWhenEnabled() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        HotkeyPreference.saveTranscriptionBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.function],
                    sidedModifiers: []
                ),
                behavior: .tap
            ),
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
        var transcriptionDownCount = 0
        var commonStopCount = 0
        manager.setCommonStopKeyEnabled(true)
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onCommonStopKeyDown = { commonStopCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: commandFlags(for: .rightCommand)))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: []))

        XCTAssertEqual(commonStopCount, 2)
        XCTAssertEqual(transcriptionDownCount, 0)
    }

    func testModifierComboTranscriptionBindingEmitsCommonStopWhenEnabled() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.saveTranscriptionBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.function, .command],
                    sidedModifiers: []
                ),
                behavior: .tap
            )
        ])

        let manager = makeManager()
        var transcriptionDownCount = 0
        var commonStopCount = 0
        manager.setCommonStopKeyEnabled(true)
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onCommonStopKeyDown = { commonStopCount += 1 }

        XCTAssertFalse(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Command), flags: combinedFlags(.maskSecondaryFn, .maskCommand)))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Command), flags: .maskSecondaryFn))
        XCTAssertFalse(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(commonStopCount, 1)
        XCTAssertEqual(transcriptionDownCount, 0)
    }
}
