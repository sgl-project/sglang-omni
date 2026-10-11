import XCTest
import AppKit
import Carbon
import ApplicationServices
import IOKit.hidsystem
@testable import Voxt

@MainActor
final class HotkeyManagerModifiersTests: HotkeyManagerTestCase {
    func testTapTranscriptionNonModifierDoesNotConsumeReleaseWithoutMatchingKeyDown() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.save(
            keyCode: UInt16(kVK_Space),
            modifiers: [.function],
            sidedModifiers: []
        )

        let manager = makeManager()
        var keyUpCount = 0
        manager.onKeyUp = { keyUpCount += 1 }

        XCTAssertFalse(
            manager.testingHandleEvent(
                type: .keyUp,
                keyCode: UInt16(kVK_Space),
                flags: .maskSecondaryFn
            )
        )
        XCTAssertEqual(keyUpCount, 0)
    }

    func testTapTranscriptionNonModifierConsumesOnlyMatchingRelease() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.save(
            keyCode: UInt16(kVK_Space),
            modifiers: [.function],
            sidedModifiers: []
        )

        let manager = makeManager()
        var keyUpCount = 0
        manager.onKeyUp = { keyUpCount += 1 }

        XCTAssertTrue(
            manager.testingHandleEvent(
                type: .keyDown,
                keyCode: UInt16(kVK_Space),
                flags: .maskSecondaryFn
            )
        )
        XCTAssertFalse(
            manager.testingHandleEvent(
                type: .keyUp,
                keyCode: UInt16(kVK_ANSI_A),
                flags: .maskSecondaryFn
            )
        )
        XCTAssertEqual(keyUpCount, 0)
        XCTAssertTrue(
            manager.testingHandleEvent(
                type: .keyUp,
                keyCode: UInt16(kVK_Space),
                flags: .maskSecondaryFn
            )
        )
        XCTAssertEqual(keyUpCount, 1)
    }

    func testTapTranslationNonModifierDoesNotConsumeReleaseWithoutMatchingKeyDown() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.saveTranslation(
            keyCode: UInt16(kVK_ANSI_Z),
            modifiers: [.function],
            sidedModifiers: []
        )

        let manager = makeManager()
        var translationKeyUpCount = 0
        manager.onTranslationKeyUp = { translationKeyUpCount += 1 }

        XCTAssertFalse(
            manager.testingHandleEvent(
                type: .keyUp,
                keyCode: UInt16(kVK_ANSI_Z),
                flags: .maskSecondaryFn
            )
        )
        XCTAssertEqual(translationKeyUpCount, 0)
    }

    func testTapRewriteNonModifierDoesNotConsumeReleaseWithoutMatchingKeyDown() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.saveRewrite(
            keyCode: UInt16(kVK_ANSI_R),
            modifiers: [.function],
            sidedModifiers: []
        )

        let manager = makeManager()
        var rewriteKeyUpCount = 0
        manager.onRewriteKeyUp = { rewriteKeyUpCount += 1 }

        XCTAssertFalse(
            manager.testingHandleEvent(
                type: .keyUp,
                keyCode: UInt16(kVK_ANSI_R),
                flags: .maskSecondaryFn
            )
        )
        XCTAssertEqual(rewriteKeyUpCount, 0)
    }

    func testTapNonModifierHotkeyRespectsRightCommandDistinction() {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.save(
            keyCode: UInt16(kVK_ANSI_L),
            modifiers: [.command],
            sidedModifiers: [.rightCommand]
        )

        let manager = makeManager()
        var transcriptionDownCount = 0
        var keyUpCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onKeyUp = { keyUpCount += 1 }

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Command),
            flags: commandFlags(for: .leftCommand)
        )
        XCTAssertFalse(
            manager.testingHandleEvent(
                type: .keyDown,
                keyCode: UInt16(kVK_ANSI_L),
                flags: commandFlags(for: .leftCommand)
            )
        )
        XCTAssertEqual(transcriptionDownCount, 0)
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Command),
            flags: []
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: commandFlags(for: .rightCommand)
        )
        XCTAssertTrue(
            manager.testingHandleEvent(
                type: .keyDown,
                keyCode: UInt16(kVK_ANSI_L),
                flags: commandFlags(for: .rightCommand)
            )
        )
        XCTAssertEqual(transcriptionDownCount, 1)
        XCTAssertTrue(
            manager.testingHandleEvent(
                type: .keyUp,
                keyCode: UInt16(kVK_ANSI_L),
                flags: commandFlags(for: .rightCommand)
            )
        )
        XCTAssertEqual(keyUpCount, 1)
    }

    func testDefaultTranslationModifierTapEmitsDedicatedCallback() async {
        let manager = makeManager()
        var transcriptionDownCount = 0
        var translationDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        let callbackExpectation = expectation(description: "translation callback")
        manager.onTranslationKeyDown = {
            translationDownCount += 1
            callbackExpectation.fulfill()
        }

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Shift),
            flags: .maskShift
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: combinedFlags(.maskShift, .maskSecondaryFn)
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskShift
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Shift),
            flags: []
        )
        await fulfillment(of: [callbackExpectation], timeout: 1.0)

        XCTAssertEqual(transcriptionDownCount, 0)
        XCTAssertEqual(translationDownCount, 1)
    }

    func testDefaultMeetingModifierTapDoesNotFallBackToFnTranscriptionOnRelease() async {
        let manager = makeManager()
        var transcriptionDownCount = 0
        var meetingDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        let callbackExpectation = expectation(description: "meeting callback")
        manager.onMeetingKeyDown = {
            meetingDownCount += 1
            callbackExpectation.fulfill()
        }

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskSecondaryFn
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Option),
            flags: combinedFlags(.maskAlternate, .maskSecondaryFn)
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Option),
            flags: .maskSecondaryFn
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: []
        )

        await fulfillment(of: [callbackExpectation], timeout: 1.0)
        XCTAssertEqual(meetingDownCount, 1)
        XCTAssertEqual(transcriptionDownCount, 0)
    }

    func testDefaultRewriteModifierTapEmitsDedicatedCallback() async {
        let manager = makeManager()
        var transcriptionDownCount = 0
        var rewriteDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        let callbackExpectation = expectation(description: "rewrite callback")
        manager.onRewriteKeyDown = {
            rewriteDownCount += 1
            callbackExpectation.fulfill()
        }

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Control),
            flags: .maskControl
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: combinedFlags(.maskControl, .maskSecondaryFn)
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskControl
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Control),
            flags: []
        )
        await fulfillment(of: [callbackExpectation], timeout: 1.0)

        XCTAssertEqual(transcriptionDownCount, 0)
        XCTAssertEqual(rewriteDownCount, 1)
    }

    func testPlainFnTapEmitsSingleTranscriptionCallback() async {
        let manager = makeManager()
        var transcriptionDownCount = 0
        let callbackExpectation = expectation(description: "transcription callback")
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

    func testPlainFnTapStillWorksWhenDistinguishingModifierSidesIsEnabledAndPresetIsCustom() async {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)

        let manager = makeManager()
        var transcriptionDownCount = 0
        let callbackExpectation = expectation(description: "transcription callback with side distinction enabled")
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

    func testLegacyStoredFunctionKeyHotkeyStillTriggersFnTap() async {
        let defaults = UserDefaults.standard
        defaults.set(Int(UInt16(kVK_Function)), forKey: AppPreferenceKey.hotkeyKeyCode)
        defaults.set(Int(NSEvent.ModifierFlags.function.rawValue), forKey: AppPreferenceKey.hotkeyModifiers)
        defaults.set(0, forKey: AppPreferenceKey.hotkeySidedModifiers)
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)

        let manager = makeManager()
        var transcriptionDownCount = 0
        let callbackExpectation = expectation(description: "legacy fn transcription callback")
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

    func testRightCommandTapRemainsStableAcrossDuplicateFlagsChangedEvents() async {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.save(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.command],
            sidedModifiers: [.rightCommand]
        )

        let manager = makeManager()
        var transcriptionDownCount = 0
        let callbackExpectation = expectation(description: "two transcription callbacks")
        callbackExpectation.expectedFulfillmentCount = 2
        manager.onKeyDown = {
            transcriptionDownCount += 1
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
            flags: commandFlags(for: .rightCommand)
        )

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: []
        )

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightCommand),
            flags: commandFlags(for: .rightCommand)
        )
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
        XCTAssertEqual(transcriptionDownCount, 2)
    }

    func testLeftCommandDoesNotTriggerRightCommandTapHotkey() {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.save(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.command],
            sidedModifiers: [.rightCommand]
        )

        let manager = makeManager()
        var transcriptionDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Command),
            flags: commandFlags(for: .leftCommand)
        )

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Command),
            flags: []
        )

        XCTAssertEqual(transcriptionDownCount, 0)
    }

    func testFnTapReleaseIsSuppressedAfterNonModifierChordState() {
        let manager = makeManager()
        var transcriptionDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }

        manager.testingSetTransientState(
            sawNonModifierKeyDuringFunctionChord: true
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: []
        )

        XCTAssertEqual(transcriptionDownCount, 0)
    }

    func testBareKeyboardBindingTriggersWithoutModifiers() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.saveTranscriptionBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: UInt16(kVK_ANSI_X),
                    modifiers: [],
                    sidedModifiers: []
                ),
                behavior: .tap
            )
        ])

        let manager = makeManager()
        var transcriptionDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .keyDown, keyCode: UInt16(kVK_ANSI_X), flags: []))
        XCTAssertTrue(manager.testingHandleEvent(type: .keyUp, keyCode: UInt16(kVK_ANSI_X), flags: []))

        XCTAssertEqual(transcriptionDownCount, 1)
    }

    func testKeyboardChordWinsOverModifierOnlyTapPrefix() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.saveTranscriptionBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.function],
                    sidedModifiers: []
                ),
                behavior: .tap
            )
        ])
        HotkeyPreference.saveTranslationBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: UInt16(kVK_Space),
                    modifiers: [.function],
                    sidedModifiers: []
                ),
                behavior: .tap
            )
        ])

        let manager = makeManager()
        var transcriptionDownCount = 0
        var translationEvents: [String] = []
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onTranslationKeyDown = { translationEvents.append("down") }
        manager.onTranslationKeyUp = { translationEvents.append("up") }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .keyDown, keyCode: UInt16(kVK_Space), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .keyUp, keyCode: UInt16(kVK_Space), flags: .maskSecondaryFn))
        _ = manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: [])

        XCTAssertEqual(transcriptionDownCount, 0)
        XCTAssertEqual(translationEvents, ["down", "up"])
    }

    func testModifierComboTapWorksWhenSpecificModifierIsReleasedBeforeFn() {
        let transcriptionHotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        let translationHotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function, .shift],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([.init(hotkey: transcriptionHotkey, behavior: .tap)])
        HotkeyPreference.saveTranslationBindings([.init(hotkey: translationHotkey, behavior: .tap)])

        let manager = makeManager()
        var transcriptionDownCount = 0
        var translationDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onTranslationKeyDown = { translationDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Shift), flags: combinedFlags(.maskSecondaryFn, .maskShift)))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Shift), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(translationDownCount, 1)
        XCTAssertEqual(transcriptionDownCount, 0)
    }

    func testNonModifierKeyCancelsModifierOnlyTapCandidate() {
        let manager = makeManager()
        var transcriptionDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertFalse(manager.testingHandleEvent(type: .keyDown, keyCode: UInt16(kVK_ANSI_A), flags: .maskSecondaryFn))
        XCTAssertFalse(manager.testingHandleEvent(type: .keyUp, keyCode: UInt16(kVK_ANSI_A), flags: .maskSecondaryFn))
        _ = manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: [])

        XCTAssertEqual(transcriptionDownCount, 0)
    }

    func testAutoRepeatDoesNotRetriggerNonModifierKeyboardHotkey() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.saveTranscriptionBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: UInt16(kVK_Space),
                    modifiers: [.function],
                    sidedModifiers: []
                ),
                behavior: .tap
            )
        ])

        let manager = makeManager()
        var transcriptionDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .keyDown, keyCode: UInt16(kVK_Space), flags: .maskSecondaryFn))
        XCTAssertFalse(manager.testingHandleEvent(type: .keyDown, keyCode: UInt16(kVK_Space), flags: .maskSecondaryFn, isAutoRepeat: true))
        XCTAssertTrue(manager.testingHandleEvent(type: .keyUp, keyCode: UInt16(kVK_Space), flags: .maskSecondaryFn))

        XCTAssertEqual(transcriptionDownCount, 1)
    }
}
