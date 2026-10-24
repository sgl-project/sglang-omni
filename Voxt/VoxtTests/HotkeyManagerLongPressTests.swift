import XCTest
import AppKit
import Carbon
import ApplicationServices
import IOKit.hidsystem
@testable import Voxt

@MainActor
final class HotkeyManagerLongPressTests: HotkeyManagerTestCase {
    func testLongPressFnEmitsDownThenUp() async {
        let defaults = UserDefaults.standard
        defaults.set(HotkeyPreference.TriggerMode.longPress.rawValue, forKey: AppPreferenceKey.hotkeyTriggerMode)
        HotkeyPreference.saveTranscriptionBindings([
            .init(hotkey: HotkeyPreference.load(), behavior: .longPress)
        ])

        let manager = makeManager()
        var events: [String] = []
        manager.onKeyDown = { events.append("down") }
        manager.onKeyUp = { events.append("up") }

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskSecondaryFn
        )
        try? await Task.sleep(for: .milliseconds(120))
        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: []
        )

        try? await Task.sleep(for: .milliseconds(120))

        XCTAssertEqual(events, ["down", "up"])
    }

    func testFnShiftTapWinsOverFnLongPressPrefix() async {
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
        HotkeyPreference.saveTranscriptionBindings([
            .init(hotkey: transcriptionHotkey, behavior: .longPress)
        ])
        HotkeyPreference.saveTranslationBindings([
            .init(hotkey: translationHotkey, behavior: .tap)
        ])

        let manager = makeManager()
        var transcriptionEvents: [String] = []
        var translationDownCount = 0
        manager.onKeyDown = { transcriptionEvents.append("down") }
        manager.onKeyUp = { transcriptionEvents.append("up") }
        manager.onTranslationKeyDown = { translationDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskSecondaryFn
        ))
        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Shift),
            flags: combinedFlags(.maskSecondaryFn, .maskShift)
        ))
        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskShift
        ))
        _ = manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Shift),
            flags: []
        )

        try? await Task.sleep(for: .milliseconds(120))

        XCTAssertEqual(transcriptionEvents, [])
        XCTAssertEqual(translationDownCount, 1)
    }

    func testFnLongPressPrefixStillFiresWhenNoCombinationArrives() async {
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
        HotkeyPreference.saveTranscriptionBindings([
            .init(hotkey: transcriptionHotkey, behavior: .longPress)
        ])
        HotkeyPreference.saveTranslationBindings([
            .init(hotkey: translationHotkey, behavior: .tap)
        ])

        let manager = makeManager()
        var events: [String] = []
        manager.onKeyDown = { events.append("down") }
        manager.onKeyUp = { events.append("up") }

        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskSecondaryFn
        ))
        try? await Task.sleep(for: .milliseconds(120))
        XCTAssertEqual(events, ["down"])

        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: []
        ))
        try? await Task.sleep(for: .milliseconds(120))

        XCTAssertEqual(events, ["down", "up"])
    }

    func testFnLongPressReleaseWorksWithResidualShiftFlag() async {
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
        HotkeyPreference.saveTranscriptionBindings([
            .init(hotkey: transcriptionHotkey, behavior: .longPress)
        ])
        HotkeyPreference.saveTranslationBindings([
            .init(hotkey: translationHotkey, behavior: .tap)
        ])

        let manager = makeManager()
        var events: [String] = []
        var translationDownCount = 0
        manager.onKeyDown = { events.append("down") }
        manager.onKeyUp = { events.append("up") }
        manager.onTranslationKeyDown = { translationDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskSecondaryFn
        ))
        try? await Task.sleep(for: .milliseconds(120))
        XCTAssertEqual(events, ["down"])

        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskShift
        ))
        try? await Task.sleep(for: .milliseconds(120))

        XCTAssertEqual(events, ["down", "up"])
        XCTAssertEqual(translationDownCount, 0)
    }

    func testFnShiftTapCancelsAlreadyStartedFnLongPress() async {
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
        HotkeyPreference.saveTranscriptionBindings([
            .init(hotkey: transcriptionHotkey, behavior: .longPress)
        ])
        HotkeyPreference.saveTranslationBindings([
            .init(hotkey: translationHotkey, behavior: .tap)
        ])

        let manager = makeManager()
        var transcriptionEvents: [String] = []
        var translationDownCount = 0
        manager.onKeyDown = { transcriptionEvents.append("down") }
        manager.onKeyUp = { transcriptionEvents.append("up") }
        manager.onTranslationKeyDown = { translationDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskSecondaryFn
        ))
        try? await Task.sleep(for: .milliseconds(120))
        XCTAssertEqual(transcriptionEvents, ["down"])

        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Shift),
            flags: combinedFlags(.maskSecondaryFn, .maskShift)
        ))
        XCTAssertEqual(transcriptionEvents, ["down", "up"])

        XCTAssertTrue(manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: .maskShift
        ))
        _ = manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Shift),
            flags: []
        )

        try? await Task.sleep(for: .milliseconds(120))

        XCTAssertEqual(transcriptionEvents, ["down", "up"])
        XCTAssertEqual(translationDownCount, 1)
    }

    func testFnTapTranscriptionStillWorksWithRightCommandLongPressBinding() {
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
                behavior: .longPress
            )
        ])

        let manager = makeManager()
        var events: [String] = []
        manager.onKeyDownWithBehavior = { events.append("down:\($0.rawValue)") }
        manager.onKeyUpWithBehavior = { events.append("up:\($0.rawValue)") }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: commandFlags(for: .rightCommand)))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: []))

        XCTAssertEqual(events, ["down:tap", "down:longPress", "up:longPress"])
    }

    func testRightCommandLongPressReleaseWorksWithResidualGenericCommandFlag() {
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
                behavior: .longPress
            )
        ])

        let manager = makeManager()
        var events: [String] = []
        manager.onKeyDownWithBehavior = { events.append("down:\($0.rawValue)") }
        manager.onKeyUpWithBehavior = { events.append("up:\($0.rawValue)") }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: commandFlags(for: .rightCommand)))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: .maskCommand))

        XCTAssertEqual(events, ["down:longPress", "up:longPress"])
    }

    func testRightCommandLongPressReleaseWorksWhenSidedFlagAlsoLingersButKeyIsUp() {
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
                behavior: .longPress
            )
        ])

        let manager = makeManager()
        manager.testingSetModifierKeyStateProvider { _ in false }
        var events: [String] = []
        manager.onKeyDownWithBehavior = { events.append("down:\($0.rawValue)") }
        manager.onKeyUpWithBehavior = { events.append("up:\($0.rawValue)") }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: commandFlags(for: .rightCommand)))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: commandFlags(for: .rightCommand)))

        XCTAssertEqual(events, ["down:longPress", "up:longPress"])
    }

    func testRightCommandLongPressReleaseIsNotClearedByIdleGapRecovery() {
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
                behavior: .longPress
            )
        ])

        let manager = makeManager()
        var events: [String] = []
        manager.onKeyDownWithBehavior = { events.append("down:\($0.rawValue)") }
        manager.onKeyUpWithBehavior = { events.append("up:\($0.rawValue)") }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: commandFlags(for: .rightCommand)))
        manager.testingSetLastEventAt(Date().addingTimeInterval(-6))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: []))

        XCTAssertEqual(events, ["down:longPress", "up:longPress"])
    }

    func testTranscriptionLongPressReleaseDoesNotEmitCommonStopWhenEnabled() {
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
                behavior: .longPress
            )
        ])

        let manager = makeManager()
        var events: [String] = []
        var commonStopCount = 0
        manager.setCommonStopKeyEnabled(true)
        manager.onKeyDownWithBehavior = { events.append("down:\($0.rawValue)") }
        manager.onKeyUpWithBehavior = { events.append("up:\($0.rawValue)") }
        manager.onCommonStopKeyDown = { commonStopCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: commandFlags(for: .rightCommand)))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: []))

        XCTAssertEqual(events, ["down:longPress", "up:longPress"])
        XCTAssertEqual(commonStopCount, 0)
    }
}
