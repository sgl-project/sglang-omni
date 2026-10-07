import XCTest
import AppKit
import Carbon
import ApplicationServices
import IOKit.hidsystem
@testable import Voxt

@MainActor
final class HotkeyManagerDoubleTapTests: HotkeyManagerTestCase {
    func testLegacyDoubleTapWakeMigratesRewriteToDoubleTapBinding() {
        UserDefaults.standard.set(
            HotkeyPreference.RewriteActivationMode.doubleTapTranscriptionHotkey.rawValue,
            forKey: AppPreferenceKey.rewriteHotkeyActivationMode
        )
        UserDefaults.standard.removeObject(forKey: AppPreferenceKey.rewriteHotkeyBindings)

        let manager = makeManager()
        var transcriptionDownCount = 0
        var rewriteDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onRewriteKeyDown = { rewriteDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(transcriptionDownCount, 0)
        XCTAssertEqual(rewriteDownCount, 1)
    }

    func testDoubleTapBindingWinsOverEarlierBusinessTapBindingForSameHotkey() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranslationBindings([.init(hotkey: hotkey, behavior: .tap)])
        HotkeyPreference.saveRewriteBindings([.init(hotkey: hotkey, behavior: .doubleTap)])

        let manager = makeManager()
        var translationDownCount = 0
        var rewriteDownCount = 0
        manager.onTranslationKeyDown = { translationDownCount += 1 }
        manager.onRewriteKeyDown = { rewriteDownCount += 1 }

        manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn)
        manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: [])
        manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn)
        manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: [])

        XCTAssertEqual(translationDownCount, 0)
        XCTAssertEqual(rewriteDownCount, 1)
    }

    func testSameFnTapAndDoubleTapBindingsFallBackToTranscriptionAfterSingleTap() async {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([.init(hotkey: hotkey, behavior: .tap)])
        HotkeyPreference.saveRewriteBindings([.init(hotkey: hotkey, behavior: .doubleTap)])

        let manager = makeManager()
        var transcriptionDownCount = 0
        var rewriteDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onRewriteKeyDown = { rewriteDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertEqual(transcriptionDownCount, 0)
        XCTAssertEqual(rewriteDownCount, 0)

        try? await Task.sleep(for: .milliseconds(Int(NSEvent.doubleClickInterval * 1000) + 80))
        await Task.yield()

        XCTAssertEqual(transcriptionDownCount, 1)
        XCTAssertEqual(rewriteDownCount, 0)
    }

    func testSameFnTapAndDoubleTapBindingsCancelTapFallbackOnDoubleTap() async {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([.init(hotkey: hotkey, behavior: .tap)])
        HotkeyPreference.saveRewriteBindings([.init(hotkey: hotkey, behavior: .doubleTap)])

        let manager = makeManager()
        var transcriptionDownCount = 0
        var rewriteDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onRewriteKeyDown = { rewriteDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(transcriptionDownCount, 0)
        XCTAssertEqual(rewriteDownCount, 1)

        try? await Task.sleep(for: .milliseconds(Int(NSEvent.doubleClickInterval * 1000) + 80))
        await Task.yield()

        XCTAssertEqual(transcriptionDownCount, 0)
        XCTAssertEqual(rewriteDownCount, 1)
    }

    func testSameFnTapAndDoubleTapBindingsSingleTapStopsTranscription() async {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([.init(hotkey: hotkey, behavior: .tap)])
        HotkeyPreference.saveRewriteBindings([.init(hotkey: hotkey, behavior: .doubleTap)])

        let manager = makeManager()
        var transcriptionDownCount = 0
        var rewriteDownCount = 0
        var commonStopCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onRewriteKeyDown = { rewriteDownCount += 1 }
        manager.onCommonStopKeyDown = { commonStopCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        try? await Task.sleep(for: .milliseconds(Int(NSEvent.doubleClickInterval * 1000) + 80))
        await Task.yield()
        XCTAssertEqual(transcriptionDownCount, 1)

        manager.setCommonStopKeyEnabled(true)
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(commonStopCount, 1)
        XCTAssertEqual(rewriteDownCount, 0)

        try? await Task.sleep(for: .milliseconds(Int(NSEvent.doubleClickInterval * 1000) + 80))
        await Task.yield()
        XCTAssertEqual(transcriptionDownCount, 1)
        XCTAssertEqual(rewriteDownCount, 0)
    }

    func testSameFnTapAndDoubleTapBindingsSingleTapStopsRewrite() async {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([.init(hotkey: hotkey, behavior: .tap)])
        HotkeyPreference.saveRewriteBindings([.init(hotkey: hotkey, behavior: .doubleTap)])

        let manager = makeManager()
        var transcriptionDownCount = 0
        var rewriteDownCount = 0
        var commonStopCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onRewriteKeyDown = { rewriteDownCount += 1 }
        manager.onCommonStopKeyDown = { commonStopCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertEqual(rewriteDownCount, 1)

        manager.setCommonStopKeyEnabled(true)
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(commonStopCount, 1)
        XCTAssertEqual(transcriptionDownCount, 0)

        try? await Task.sleep(for: .milliseconds(Int(NSEvent.doubleClickInterval * 1000) + 80))
        await Task.yield()
        XCTAssertEqual(transcriptionDownCount, 0)
        XCTAssertEqual(rewriteDownCount, 1)
    }

    func testTranscriptionModifierOnlyDoubleTapWaitsForSecondRelease() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([.init(hotkey: hotkey, behavior: .doubleTap)])

        let manager = makeManager()
        var transcriptionEvents: [String] = []
        manager.onKeyDownWithBehavior = { transcriptionEvents.append($0.rawValue) }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertEqual(transcriptionEvents, [])
        XCTAssertFalse(manager.testingTransientStateSnapshot().isKeyDown)
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertEqual(transcriptionEvents, [])
        XCTAssertFalse(manager.testingTransientStateSnapshot().isKeyDown)

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertEqual(transcriptionEvents, [])
        XCTAssertFalse(manager.testingTransientStateSnapshot().isKeyDown)
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertEqual(transcriptionEvents, ["doubleTap"])
    }

    func testTranscriptionModifierOnlyDoubleTapIgnoresStaleSameHotkeyTapBindingOnFirstPress() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([
            .init(hotkey: hotkey, behavior: .tap),
            .init(hotkey: hotkey, behavior: .doubleTap)
        ])

        let manager = makeManager()
        var transcriptionEvents: [String] = []
        manager.onKeyDownWithBehavior = { transcriptionEvents.append($0.rawValue) }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertEqual(transcriptionEvents, [])
        XCTAssertFalse(manager.testingTransientStateSnapshot().isKeyDown)
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertEqual(transcriptionEvents, [])
    }

    func testTranscriptionDoubleTapFirstReleaseDoesNotEmitCommonStopWhenIdle() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([.init(hotkey: hotkey, behavior: .doubleTap)])

        let manager = makeManager()
        var transcriptionDownCount = 0
        var commonStopCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onCommonStopKeyDown = { commonStopCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(commonStopCount, 0)
        XCTAssertEqual(transcriptionDownCount, 0)
    }

    func testTranscriptionDoubleTapFirstReleaseEmitsCommonStopWhenEnabled() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([.init(hotkey: hotkey, behavior: .doubleTap)])

        let manager = makeManager()
        var transcriptionDownCount = 0
        var commonStopCount = 0
        manager.setCommonStopKeyEnabled(true)
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onCommonStopKeyDown = { commonStopCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(commonStopCount, 1)
        XCTAssertEqual(transcriptionDownCount, 0)
    }

    func testTranscriptionDoubleTapActiveSessionSingleTapEmitsCommonStopWithoutRestarting() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([.init(hotkey: hotkey, behavior: .doubleTap)])

        let manager = makeManager()
        var transcriptionDownCount = 0
        var commonStopCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onCommonStopKeyDown = {
            commonStopCount += 1
            manager.cancelPendingDoubleTapCandidate(reason: "testCommonStop")
        }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertEqual(transcriptionDownCount, 1)

        manager.setCommonStopKeyEnabled(true)
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(commonStopCount, 1)
        XCTAssertEqual(transcriptionDownCount, 1)
    }

    func testCancelPendingDoubleTapPreventsCommonStopSecondTapFromStartingTranscription() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveTranscriptionBindings([.init(hotkey: hotkey, behavior: .doubleTap)])

        let manager = makeManager()
        var transcriptionDownCount = 0
        manager.setCommonStopKeyEnabled(true)
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.onCommonStopKeyDown = {
            manager.cancelPendingDoubleTapCandidate(reason: "testCommonStop")
        }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(transcriptionDownCount, 0)
    }

    func testFnTapTranscriptionStillWorksWithRightCommandDoubleTapBinding() {
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
                behavior: .doubleTap
            )
        ])

        let manager = makeManager()
        var events: [String] = []
        manager.onKeyDownWithBehavior = { events.append("down:\($0.rawValue)") }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: commandFlags(for: .rightCommand)))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: []))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: commandFlags(for: .rightCommand)))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_RightCommand), flags: []))

        XCTAssertEqual(events, ["down:tap", "down:doubleTap"])
    }

    func testDoubleTapDoesNotTriggerAfterDoubleClickWindowExpires() async {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.function],
            sidedModifiers: []
        )
        HotkeyPreference.saveRewriteBindings([.init(hotkey: hotkey, behavior: .doubleTap)])

        let manager = makeManager()
        var rewriteDownCount = 0
        manager.onRewriteKeyDown = { rewriteDownCount += 1 }

        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))
        try? await Task.sleep(for: .milliseconds(Int(NSEvent.doubleClickInterval * 1000) + 80))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: .maskSecondaryFn))
        XCTAssertTrue(manager.testingHandleEvent(type: .flagsChanged, keyCode: UInt16(kVK_Function), flags: []))

        XCTAssertEqual(rewriteDownCount, 0)
    }
}
