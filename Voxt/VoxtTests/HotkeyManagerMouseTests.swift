import XCTest
import AppKit
import Carbon
import ApplicationServices
import IOKit.hidsystem
@testable import Voxt

@MainActor
final class HotkeyManagerMouseTests: HotkeyManagerTestCase {
    func testMouseMiddleTapTriggersTranscriptionCallbacks() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.save(HotkeyPreference.Hotkey(mouseButtonNumber: 2))

        let manager = makeManager()
        var events: [String] = []
        manager.onKeyDown = { events.append("down") }
        manager.onKeyUp = { events.append("up") }

        XCTAssertTrue(manager.testingHandleMouseEvent(type: .otherMouseDown, buttonNumber: 2))
        XCTAssertTrue(manager.testingHandleMouseEvent(type: .otherMouseUp, buttonNumber: 2))

        XCTAssertEqual(events, ["down", "up"])
    }

    func testMouseMiddleDoubleTapCanTriggerRewriteBindingWithoutTapFallback() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        defaults.set(
            HotkeyPreference.RewriteActivationMode.doubleTapTranscriptionHotkey.rawValue,
            forKey: AppPreferenceKey.rewriteHotkeyActivationMode
        )
        HotkeyPreference.save(HotkeyPreference.Hotkey(mouseButtonNumber: 2))
        HotkeyPreference.saveRewrite(HotkeyPreference.Hotkey(mouseButtonNumber: 2))

        let manager = makeManager()
        var transcriptionDownCount = 0
        var rewriteDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onRewriteKeyDown = { rewriteDownCount += 1 }

        manager.testingHandleMouseEvent(type: .otherMouseDown, buttonNumber: 2)
        manager.testingHandleMouseEvent(type: .otherMouseUp, buttonNumber: 2)
        manager.testingHandleMouseEvent(type: .otherMouseDown, buttonNumber: 2)
        manager.testingHandleMouseEvent(type: .otherMouseUp, buttonNumber: 2)

        XCTAssertEqual(transcriptionDownCount, 0)
        XCTAssertEqual(rewriteDownCount, 1)
    }

    func testMouseLongPressEmitsBalancedDownAndUp() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.saveTranscriptionBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 4),
                behavior: .longPress
            )
        ])

        let manager = makeManager()
        var events: [String] = []
        manager.onKeyDownWithBehavior = { events.append("down:\($0.rawValue)") }
        manager.onKeyUpWithBehavior = { events.append("up:\($0.rawValue)") }

        XCTAssertTrue(manager.testingHandleMouseEvent(type: .otherMouseDown, buttonNumber: 4))
        XCTAssertTrue(manager.testingHandleMouseEvent(type: .otherMouseUp, buttonNumber: 4))

        XCTAssertEqual(events, ["down:longPress", "up:longPress"])
    }

    func testMouseLongPressReleaseWorksAfterModifierIsReleased() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.saveTranscriptionBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    mouseButtonNumber: 4,
                    modifiers: [.command],
                    sidedModifiers: []
                ),
                behavior: .longPress
            )
        ])

        let manager = makeManager()
        var events: [String] = []
        manager.onKeyDown = { events.append("down") }
        manager.onKeyUp = { events.append("up") }

        XCTAssertTrue(manager.testingHandleMouseEvent(type: .otherMouseDown, buttonNumber: 4, flags: .maskCommand))
        XCTAssertTrue(manager.testingHandleMouseEvent(type: .otherMouseUp, buttonNumber: 4, flags: []))

        XCTAssertEqual(events, ["down", "up"])
    }

    func testMouseCustomPasteAndTranscriptionButtonBindingsStaySeparatedByModifiers() {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        defaults.set(true, forKey: AppPreferenceKey.customPasteHotkeyEnabled)
        HotkeyPreference.saveTranscriptionBindings([
            .init(hotkey: HotkeyPreference.Hotkey(mouseButtonNumber: 4), behavior: .tap)
        ])
        HotkeyPreference.saveCustomPaste(
            HotkeyPreference.Hotkey(
                mouseButtonNumber: 4,
                modifiers: [.command],
                sidedModifiers: []
            )
        )

        let manager = makeManager()
        var transcriptionEvents: [String] = []
        var customPasteDownCount = 0
        manager.onKeyDown = { transcriptionEvents.append("down") }
        manager.onKeyUp = { transcriptionEvents.append("up") }
        manager.onCustomPasteKeyDown = { customPasteDownCount += 1 }

        XCTAssertTrue(manager.testingHandleMouseEvent(type: .otherMouseDown, buttonNumber: 4))
        XCTAssertTrue(manager.testingHandleMouseEvent(type: .otherMouseUp, buttonNumber: 4))
        XCTAssertTrue(manager.testingHandleMouseEvent(type: .otherMouseDown, buttonNumber: 4, flags: .maskCommand))
        XCTAssertTrue(manager.testingHandleMouseEvent(type: .otherMouseUp, buttonNumber: 4, flags: .maskCommand))

        XCTAssertEqual(transcriptionEvents, ["down", "up"])
        XCTAssertEqual(customPasteDownCount, 1)
    }

    func testMousePresetKeepsFnShiftTranslationHigherPriority() async {
        HotkeyPreference.applyPreset(.mouseMiddleFnShift)

        let manager = makeManager()
        var transcriptionDownCount = 0
        var translationDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        let callbackExpectation = expectation(description: "fn-shift translation callback with mouse transcription")
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
}
