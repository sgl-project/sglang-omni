import XCTest
import AppKit
import Carbon
import ApplicationServices
import IOKit.hidsystem
@testable import Voxt

@MainActor
final class HotkeyManagerTests: HotkeyManagerTestCase {
    func testEscapeKeyDownCanBeConsumedViaCallback() {
        let manager = makeManager()
        var escapeCallbackCount = 0
        manager.onEscapeKeyDown = {
            escapeCallbackCount += 1
            return true
        }

        XCTAssertTrue(
            manager.testingHandleEvent(
                type: .keyDown,
                keyCode: UInt16(kVK_Escape),
                flags: []
            )
        )
        XCTAssertEqual(escapeCallbackCount, 1)
    }

    func testEscapeKeyDownPassesThroughWhenCallbackDeclinesConsumption() {
        let manager = makeManager()
        var escapeCallbackCount = 0
        manager.onEscapeKeyDown = {
            escapeCallbackCount += 1
            return false
        }

        XCTAssertFalse(
            manager.testingHandleEvent(
                type: .keyDown,
                keyCode: UInt16(kVK_Escape),
                flags: []
            )
        )
        XCTAssertEqual(escapeCallbackCount, 1)
    }

    func testProductionEventDispatchDoesNotRunCallbackBeforeReturning() async {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.save(
            keyCode: UInt16(kVK_Space),
            modifiers: [.function],
            sidedModifiers: []
        )

        let manager = makeManager()
        var transcriptionDownCount = 0
        let callbackExpectation = expectation(description: "transcription callback")
        manager.onKeyDown = {
            transcriptionDownCount += 1
            callbackExpectation.fulfill()
        }

        XCTAssertTrue(
            manager.testingHandleEventUsingProductionCallbackDispatch(
                type: .keyDown,
                keyCode: UInt16(kVK_Space),
                flags: .maskSecondaryFn
            )
        )
        XCTAssertEqual(transcriptionDownCount, 0)

        await fulfillment(of: [callbackExpectation], timeout: 1.0)
        XCTAssertEqual(transcriptionDownCount, 1)
    }

    func testTranslationTapCallbackCanReenterEventHandlingWithoutExclusivityViolation() async {
        let manager = makeManager()
        var translationDownCount = 0
        let callbackExpectation = expectation(description: "translation callback")
        manager.onTranslationKeyDown = {
            translationDownCount += 1
            manager.testingHandleEvent(
                type: .keyDown,
                keyCode: UInt16(kVK_ANSI_V),
                flags: self.combinedFlags(.maskShift, .maskSecondaryFn)
            )
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
        XCTAssertEqual(translationDownCount, 1)
    }

    func testCustomRightShiftTapRemainsStableAcrossDuplicateFlagsChangedEvents() async {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.save(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.shift],
            sidedModifiers: [.rightShift]
        )

        let manager = makeManager()
        var transcriptionDownCount = 0
        let callbackExpectation = expectation(description: "two right-shift callbacks")
        callbackExpectation.expectedFulfillmentCount = 2
        manager.onKeyDown = {
            transcriptionDownCount += 1
            callbackExpectation.fulfill()
        }

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightShift),
            flags: shiftFlags(for: .rightShift)
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightShift),
            flags: shiftFlags(for: .rightShift)
        )

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightShift),
            flags: []
        )

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightShift),
            flags: shiftFlags(for: .rightShift)
        )
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightShift),
            flags: shiftFlags(for: .rightShift)
        )

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_RightShift),
            flags: []
        )

        await fulfillment(of: [callbackExpectation], timeout: 1.0)
        XCTAssertEqual(transcriptionDownCount, 2)
    }

    func testMultipleTranscriptionBindingsCanTriggerSameBusiness() {
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
            ),
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: UInt16(kVK_Return),
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
        XCTAssertTrue(manager.testingHandleEvent(type: .keyUp, keyCode: UInt16(kVK_Space), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .keyDown, keyCode: UInt16(kVK_Return), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .keyUp, keyCode: UInt16(kVK_Return), flags: .maskSecondaryFn))

        XCTAssertEqual(transcriptionDownCount, 2)
    }

    func testBareKeyboardBindingDoesNotMatchModifiedKeyPress() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.saveTranscriptionBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: UInt16(kVK_ANSI_F),
                    modifiers: [],
                    sidedModifiers: []
                ),
                behavior: .tap
            )
        ])

        let manager = makeManager()
        var transcriptionDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }

        XCTAssertFalse(manager.testingHandleEvent(type: .keyDown, keyCode: UInt16(kVK_ANSI_F), flags: .maskCommand))
        XCTAssertFalse(manager.testingHandleEvent(type: .keyUp, keyCode: UInt16(kVK_ANSI_F), flags: .maskCommand))

        XCTAssertEqual(transcriptionDownCount, 0)
    }

    func testTranscriptionBindingsKeepIndependentTriggerBehaviors() async {
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
            ),
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.function],
                    sidedModifiers: []
                ),
                behavior: .longPress
            )
        ])

        let manager = makeManager()
        var events: [String] = []
        manager.onKeyDownWithBehavior = { events.append("down:\($0.rawValue)") }
        manager.onKeyUpWithBehavior = { events.append("up:\($0.rawValue)") }

        XCTAssertTrue(manager.testingHandleEvent(type: .keyDown, keyCode: UInt16(kVK_Space), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .keyUp, keyCode: UInt16(kVK_Space), flags: .maskSecondaryFn))

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        try? await Task.sleep(for: .milliseconds(120))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(events, ["down:tap", "up:tap", "down:longPress", "up:longPress"])
    }

    func testSameHotkeySameBehaviorUsesBusinessPriorityDeterministically() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([.init(hotkey: hotkey, behavior: .tap)])
        HotkeyPreference.saveTranslationBindings([.init(hotkey: hotkey, behavior: .tap)])
        HotkeyPreference.saveRewriteBindings([.init(hotkey: hotkey, behavior: .tap)])
        HotkeyPreference.saveMeetingBindings([.init(hotkey: hotkey, behavior: .tap)])

        let manager = makeManager()
        var transcriptionDownCount = 0
        var translationDownCount = 0
        var rewriteDownCount = 0
        var meetingDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onTranslationKeyDown = { translationDownCount += 1 }
        manager.onRewriteKeyDown = { rewriteDownCount += 1 }
        manager.onMeetingKeyDown = { meetingDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(translationDownCount, 1)
        XCTAssertEqual(rewriteDownCount, 0)
        XCTAssertEqual(meetingDownCount, 0)
        XCTAssertEqual(transcriptionDownCount, 0)
    }
}
