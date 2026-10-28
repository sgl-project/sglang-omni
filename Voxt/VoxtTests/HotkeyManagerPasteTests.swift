import XCTest
import AppKit
import Carbon
import ApplicationServices
import IOKit.hidsystem
@testable import Voxt

@MainActor
final class HotkeyManagerPasteTests: HotkeyManagerTestCase {
    func testModifierOnlyCustomPasteDoesNotBlockFnTapTranscription() async {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: AppPreferenceKey.customPasteHotkeyEnabled)
        defaults.set(Int(HotkeyPreference.modifierOnlyKeyCode), forKey: AppPreferenceKey.customPasteHotkeyKeyCode)
        defaults.set(Int(NSEvent.ModifierFlags.command.rawValue), forKey: AppPreferenceKey.customPasteHotkeyModifiers)
        defaults.set(SidedModifierFlags.rightCommand.rawValue, forKey: AppPreferenceKey.customPasteHotkeySidedModifiers)
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)

        let manager = makeManager()
        var transcriptionDownCount = 0
        let callbackExpectation = expectation(description: "fn transcription callback with modifier-only custom paste enabled")
        manager.onKeyDown = {
            transcriptionDownCount += 1
            callbackExpectation.fulfill()
        }

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskSecondaryFn
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: []
        )

        await fulfillment(of: [callbackExpectation], timeout: 1.0)
        XCTAssertEqual(transcriptionDownCount, 1)
    }

    func testModifierOnlyCustomPasteStillTriggersWithRightCommand() async {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: AppPreferenceKey.customPasteHotkeyEnabled)
        defaults.set(Int(HotkeyPreference.modifierOnlyKeyCode), forKey: AppPreferenceKey.customPasteHotkeyKeyCode)
        defaults.set(Int(NSEvent.ModifierFlags.command.rawValue), forKey: AppPreferenceKey.customPasteHotkeyModifiers)
        defaults.set(SidedModifierFlags.rightCommand.rawValue, forKey: AppPreferenceKey.customPasteHotkeySidedModifiers)
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)

        let manager = makeManager()
        var customPasteDownCount = 0
        let callbackExpectation = expectation(description: "right-command custom paste callback")
        manager.onCustomPasteKeyDown = {
            customPasteDownCount += 1
            callbackExpectation.fulfill()
        }

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: commandFlags(for: .rightCommand)
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: []
        )

        await fulfillment(of: [callbackExpectation], timeout: 1.0)
        XCTAssertEqual(customPasteDownCount, 1)
    }

    func testControlCommandVCustomPasteStillTriggersUnderCommandPresetWithRightCommand() async {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: AppPreferenceKey.customPasteHotkeyEnabled)
        defaults.set(Int(UInt16(kVK_ANSI_V)), forKey: AppPreferenceKey.customPasteHotkeyKeyCode)
        defaults.set(Int(NSEvent.ModifierFlags([.control, .command]).rawValue), forKey: AppPreferenceKey.customPasteHotkeyModifiers)
        defaults.set(0, forKey: AppPreferenceKey.customPasteHotkeySidedModifiers)
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        defaults.set(HotkeyPreference.Preset.commandCombo.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.save(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.command],
            sidedModifiers: [.rightCommand]
        )

        let manager = makeManager()
        var customPasteDownCount = 0
        let callbackExpectation = expectation(description: "control-command-v custom paste callback under command preset")
        manager.onCustomPasteKeyDown = {
            customPasteDownCount += 1
            callbackExpectation.fulfill()
        }

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Control),
            flags: .maskControl
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: commandFlags(for: .rightCommand).union(.maskControl)
        )
        manager.testingHandleEvent(
            type: .keyDown,
            keyCode: UInt16(kVK_ANSI_V),
            flags: commandFlags(for: .rightCommand).union(.maskControl)
        )
        manager.testingHandleEvent(
            type: .keyUp,
            keyCode: UInt16(kVK_ANSI_V),
            flags: commandFlags(for: .rightCommand).union(.maskControl)
        )

        await fulfillment(of: [callbackExpectation], timeout: 1.0)
        XCTAssertEqual(customPasteDownCount, 1)
    }

    func testVoxtInjectedKeyboardEventsBypassHotkeyRouting() {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: AppPreferenceKey.customPasteHotkeyEnabled)
        defaults.set(Int(UInt16(kVK_ANSI_V)), forKey: AppPreferenceKey.customPasteHotkeyKeyCode)
        defaults.set(Int(NSEvent.ModifierFlags([.control, .command]).rawValue), forKey: AppPreferenceKey.customPasteHotkeyModifiers)
        defaults.set(0, forKey: AppPreferenceKey.customPasteHotkeySidedModifiers)

        let manager = makeManager()
        let injectedUserData = voxtInjectedEventSourceUserData()
        var customPasteDownCount = 0
        manager.onCustomPasteKeyDown = {
            customPasteDownCount += 1
        }

        let flags = CGEventFlags.maskCommand.union(.maskControl)
        XCTAssertFalse(
            manager.testingHandleEvent(
                type: .keyDown,
                keyCode: UInt16(kVK_ANSI_V),
                flags: flags,
                eventSourceUserData: injectedUserData
            )
        )
        XCTAssertFalse(
            manager.testingHandleEvent(
                type: .keyUp,
                keyCode: UInt16(kVK_ANSI_V),
                flags: flags,
                eventSourceUserData: injectedUserData
            )
        )
        XCTAssertEqual(customPasteDownCount, 0)
    }

    func testCustomPasteKeyboardTapEmitsOnlyOnRelease() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        defaults.set(true, forKey: AppPreferenceKey.customPasteHotkeyEnabled)
        HotkeyPreference.saveCustomPaste(
            keyCode: UInt16(kVK_ANSI_V),
            modifiers: [.control, .command],
            sidedModifiers: []
        )

        let manager = makeManager()
        var customPasteDownCount = 0
        manager.onCustomPasteKeyDown = { customPasteDownCount += 1 }
        let flags = combinedFlags(.maskControl, .maskCommand)

        XCTAssertTrue(manager.testingHandleEvent(type: .keyDown, keyCode: UInt16(kVK_ANSI_V), flags: flags))
        XCTAssertEqual(customPasteDownCount, 0)
        XCTAssertTrue(manager.testingHandleEvent(type: .keyUp, keyCode: UInt16(kVK_ANSI_V), flags: flags))

        XCTAssertEqual(customPasteDownCount, 1)
    }
}
