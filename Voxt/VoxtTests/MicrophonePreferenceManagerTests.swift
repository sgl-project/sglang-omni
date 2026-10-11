// MicrophonePreferenceManagerTests.swift
// Provides Microphone Preference Manager Tests for Voxt test coverage.

import XCTest
import CoreAudio
@testable import Voxt

final class MicrophonePreferenceManagerTests: XCTestCase {
    func testInitialSelectionPrefersHighestTrackedAvailableDevice() {
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(["usb-high", "builtin"], forKey: AppPreferenceKey.microphonePriorityUIDs)

        let state = MicrophonePreferenceManager.syncState(
            defaults: defaults,
            availableDevices: [
                device(id: 1, uid: "builtin", name: "Built-in Mic"),
                device(id: 2, uid: "usb-high", name: "USB Mic")
            ]
        )

        XCTAssertEqual(state.activeUID, "usb-high")
        XCTAssertTrue(state.autoSwitchEnabled)
    }

    func testFocusedDeviceStaysSelectedDuringRegularRefresh() {
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(["usb-high", "builtin"], forKey: AppPreferenceKey.microphonePriorityUIDs)

        _ = MicrophonePreferenceManager.setFocusedDevice(
            uid: "builtin",
            defaults: defaults,
            availableDevices: [
                device(id: 1, uid: "builtin", name: "Built-in Mic"),
                device(id: 2, uid: "usb-high", name: "USB Mic")
            ]
        )

        let state = MicrophonePreferenceManager.syncState(
            defaults: defaults,
            availableDevices: [
                device(id: 1, uid: "builtin", name: "Built-in Mic"),
                device(id: 2, uid: "usb-high", name: "USB Mic")
            ]
        )

        XCTAssertEqual(state.activeUID, "builtin")
    }

    func testHardwareReconnectPromotesHigherPriorityDeviceWhenAutoSwitchEnabled() {
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(["usb-high", "builtin"], forKey: AppPreferenceKey.microphonePriorityUIDs)
        defaults.set("builtin", forKey: AppPreferenceKey.activeInputDeviceUID)

        let state = MicrophonePreferenceManager.syncState(
            defaults: defaults,
            availableDevices: [
                device(id: 1, uid: "builtin", name: "Built-in Mic"),
                device(id: 2, uid: "usb-high", name: "USB Mic")
            ],
            previousAvailableUIDs: ["builtin"]
        )

        XCTAssertEqual(state.activeUID, "usb-high")
    }

    func testHardwareReconnectDoesNotPromoteHigherPriorityDeviceWhenAutoSwitchDisabled() {
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(["usb-high", "builtin"], forKey: AppPreferenceKey.microphonePriorityUIDs)
        defaults.set("builtin", forKey: AppPreferenceKey.activeInputDeviceUID)
        defaults.set(false, forKey: AppPreferenceKey.microphoneAutoSwitchEnabled)

        let state = MicrophonePreferenceManager.syncState(
            defaults: defaults,
            availableDevices: [
                device(id: 1, uid: "builtin", name: "Built-in Mic"),
                device(id: 2, uid: "usb-high", name: "USB Mic")
            ],
            previousAvailableUIDs: ["builtin"]
        )

        XCTAssertEqual(state.activeUID, "builtin")
        XCTAssertFalse(state.autoSwitchEnabled)
    }

    func testUnavailableFocusedDeviceFallsBackToPriorityOrder() {
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(["usb-high", "builtin"], forKey: AppPreferenceKey.microphonePriorityUIDs)
        defaults.set("builtin", forKey: AppPreferenceKey.activeInputDeviceUID)

        let state = MicrophonePreferenceManager.syncState(
            defaults: defaults,
            availableDevices: [
                device(id: 2, uid: "usb-high", name: "USB Mic")
            ]
        )

        XCTAssertEqual(state.activeUID, "usb-high")
    }

    func testLockedActiveUIDPreservesCurrentDeviceDuringHardwareRefresh() {
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(["usb-high", "builtin"], forKey: AppPreferenceKey.microphonePriorityUIDs)
        defaults.set("builtin", forKey: AppPreferenceKey.activeInputDeviceUID)

        let state = MicrophonePreferenceManager.syncState(
            defaults: defaults,
            availableDevices: [
                device(id: 1, uid: "builtin", name: "Built-in Mic"),
                device(id: 2, uid: "usb-high", name: "USB Mic")
            ],
            previousAvailableUIDs: ["builtin"],
            lockedActiveUID: "builtin"
        )

        XCTAssertEqual(state.activeUID, "builtin")
    }

    func testReorderPriorityDeduplicatesAndPersistsTrackedOrder() {
        let defaults = TestDoubles.makeUserDefaults()

        let state = MicrophonePreferenceManager.reorderPriority(
            orderedUIDs: ["usb-high", "builtin", "usb-high", "  ", "headset"],
            defaults: defaults,
            availableDevices: [
                device(id: 1, uid: "builtin", name: "Built-in Mic"),
                device(id: 2, uid: "usb-high", name: "USB Mic"),
                device(id: 3, uid: "headset", name: "Headset")
            ]
        )

        XCTAssertEqual(state.priorityUIDs, ["usb-high", "builtin", "headset"])
        XCTAssertEqual(
            MicrophonePreferenceManager.trackedRecords(defaults: defaults),
            [
                TrackedMicrophoneRecord(uid: "usb-high", lastKnownName: "USB Mic"),
                TrackedMicrophoneRecord(uid: "builtin", lastKnownName: "Built-in Mic"),
                TrackedMicrophoneRecord(uid: "headset", lastKnownName: "Headset")
            ]
        )
    }

    func testTrackedOfflineDevicesStayVisibleAheadOfUntrackedAvailableDevices() {
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(["usb-high"], forKey: AppPreferenceKey.microphonePriorityUIDs)
        defaults.set(
            """
            [{"uid":"usb-high","lastKnownName":"USB Mic"}]
            """,
            forKey: AppPreferenceKey.trackedMicrophoneRecords
        )

        let state = MicrophonePreferenceManager.syncState(
            defaults: defaults,
            availableDevices: [
                device(id: 1, uid: "builtin", name: "Built-in Mic")
            ]
        )

        XCTAssertEqual(state.entries.map(\.uid), ["usb-high", "builtin"])
        XCTAssertEqual(state.entries.first?.status, .offline)
        XCTAssertEqual(state.activeUID, "builtin")
        XCTAssertEqual(state.entries.last?.status, .inUse)
    }

    private func device(id: AudioDeviceID, uid: String, name: String) -> AudioInputDevice {
        AudioInputDevice(id: id, uid: uid, name: name)
    }
}
