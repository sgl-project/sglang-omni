import XCTest
import Carbon
import AppKit
@testable import Voxt

@MainActor
final class HotkeyManagerLifetimeTests: HotkeyManagerTestCase {
    func testStopDropsQueuedBusinessCallbacks() async {
        let manager = configuredManager()
        var count = 0
        manager.onKeyDown = { count += 1 }
        queueKeyDown(manager)
        manager.stop()
        await flushMainQueue()
        XCTAssertEqual(count, 0)
    }

    func testResetDropsOldCallbackWithoutDroppingNewGeneration() async {
        let manager = configuredManager()
        var count = 0
        manager.onKeyDown = { count += 1 }
        queueKeyDown(manager)
        manager.resetTransientState(reason: "test replacement")
        queueKeyDown(manager)
        await flushMainQueue()
        XCTAssertEqual(count, 1)
    }

    func testRecoveryWithoutTapCannotDeliverQueuedOldAction() async {
        let manager = configuredManager()
        var count = 0
        manager.onKeyDown = { count += 1 }
        queueKeyDown(manager)
        XCTAssertEqual(manager.recoverEventTapIfNeeded(disabledEventType: .tapDisabledByTimeout), .unavailable)
        await flushMainQueue()
        XCTAssertEqual(count, 0)
    }

    func testRepeatedStopLeavesRoutingStateEmpty() {
        let manager = configuredManager()
        manager.testingSetTransientState(
            isKeyDown: true, isTranslationKeyDown: true,
            hasTranscriptionModifierTapCandidate: true
        )
        manager.stop()
        manager.stop()
        let snapshot = manager.testingTransientStateSnapshot()
        XCTAssertFalse(snapshot.isKeyDown)
        XCTAssertFalse(snapshot.isTranslationKeyDown)
        XCTAssertFalse(snapshot.hasTranscriptionModifierTapCandidate)
    }

    private func configuredManager() -> HotkeyManager {
        UserDefaults.standard.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        HotkeyPreference.save(keyCode: UInt16(kVK_Space), modifiers: [.function], sidedModifiers: [])
        return makeManager()
    }

    private func queueKeyDown(_ manager: HotkeyManager) {
        _ = manager.testingHandleEventUsingProductionCallbackDispatch(
            type: .keyDown, keyCode: UInt16(kVK_Space), flags: .maskSecondaryFn
        )
    }

    private func flushMainQueue() async {
        await withCheckedContinuation { continuation in
            DispatchQueue.main.async { continuation.resume() }
        }
    }
}
