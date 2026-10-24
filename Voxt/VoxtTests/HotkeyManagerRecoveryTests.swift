import XCTest
import AppKit
import Carbon
import ApplicationServices
import IOKit.hidsystem
@testable import Voxt

@MainActor
final class HotkeyManagerRecoveryTests: HotkeyManagerTestCase {
    func testResetTransientStateClearsPendingTapReleaseConsumptionForNonModifierHotkey() {
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
        manager.resetTransientState(reason: "unitTestCancel")

        XCTAssertFalse(
            manager.testingHandleEvent(
                type: .keyUp,
                keyCode: UInt16(kVK_Space),
                flags: .maskSecondaryFn
            )
        )
        XCTAssertEqual(keyUpCount, 0)
    }

    func testStaleFnStateIsResetBeforeFreshTapStartsTranscription() async {
        let manager = makeManager()
        var transcriptionDownCount = 0
        let callbackExpectation = expectation(description: "transcription callback")
        manager.onKeyDown = {
            transcriptionDownCount += 1
            callbackExpectation.fulfill()
        }

        manager.testingSetTransientState(
            isKeyDown: true,
            hasTranscriptionModifierTapCandidate: true
        )

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

    func testResetTransientStateClearsTransientStateWithoutEmittingCallbacks() {
        let manager = makeManager()
        var transcriptionDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        manager.testingSetTransientState(
            isKeyDown: true,
            isTranslationKeyDown: true,
            hasTranscriptionModifierTapCandidate: true,
            hasTranslationModifierTapCandidate: true,
            sawNonModifierKeyDuringFunctionChord: true,
            currentSidedModifiers: .leftShift
        )

        manager.resetTransientState(reason: "unitTest")

        XCTAssertEqual(transcriptionDownCount, 0)
        XCTAssertEqual(
            manager.testingTransientStateSnapshot(),
            .init(
                isKeyDown: false,
                isTranslationKeyDown: false,
                isRewriteKeyDown: false,
                isCustomPasteKeyDown: false,
                hasTranscriptionModifierTapCandidate: false,
                hasTranslationModifierTapCandidate: false,
                hasRewriteModifierTapCandidate: false,
                hasCustomPasteModifierTapCandidate: false,
                sawNonModifierKeyDuringFunctionChord: false,
                currentSidedModifiers: []
            )
        )
    }

    func testTranslationComboStillWinsAfterRecoveryReset() async {
        let manager = makeManager()
        var transcriptionDownCount = 0
        var translationDownCount = 0
        manager.onKeyDown = { transcriptionDownCount += 1 }
        let callbackExpectation = expectation(description: "translation callback")
        manager.onTranslationKeyDown = {
            translationDownCount += 1
            callbackExpectation.fulfill()
        }

        manager.testingSetTransientState(
            isRewriteKeyDown: true,
            hasRewriteModifierTapCandidate: true,
            currentSidedModifiers: .rightControl
        )
        manager.resetTransientState(reason: "unitTest")

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

    func testIdleGapRecoveryClearsStaleChordStateBeforeFnRelease() async {
        let manager = makeManager()
        var transcriptionDownCount = 0
        let callbackExpectation = expectation(description: "transcription callback")
        manager.onKeyDown = {
            transcriptionDownCount += 1
            callbackExpectation.fulfill()
        }

        manager.testingSetTransientState(
            sawNonModifierKeyDuringFunctionChord: true
        )
        manager.testingSetLastEventAt(Date().addingTimeInterval(-5))

        manager.testingHandleEvent(
            type: .flagsChanged,
            keyCode: UInt16(kVK_Function),
            flags: []
        )
        await fulfillment(of: [callbackExpectation], timeout: 1.0)

        XCTAssertEqual(transcriptionDownCount, 1)
    }

    func testIdleGapRecoveryDoesNotSwallowFirstRightCommandTap() async {
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
        let callbackExpectation = expectation(description: "first tap survives idle recovery")
        manager.onKeyDown = {
            transcriptionDownCount += 1
            callbackExpectation.fulfill()
        }

        manager.testingSetTransientState(currentSidedModifiers: .rightCommand)
        manager.testingSetLastEventAt(Date().addingTimeInterval(-5))

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
        XCTAssertEqual(transcriptionDownCount, 1)
    }
}
