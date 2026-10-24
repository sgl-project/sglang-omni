import XCTest
import CoreAudio
@testable import Voxt

@MainActor
final class SystemAudioMuteControllerTests: XCTestCase {
    private let first = SystemAudioOutputDevice(id: 10, uid: "speaker")
    private let second = SystemAudioOutputDevice(id: 20, uid: "headphones")

    private final class Devices: SystemAudioMuteDeviceAccess {
        var current: SystemAudioOutputDevice?
        var states: [AudioDeviceID: (uid: String, muted: Bool)] = [:]
        var canWrite = true
        var canRead = true
        var writes: [String] = []
        var changed: (() -> Void)?

        func defaultOutputDevice() -> SystemAudioOutputDevice? { current }
        func muteState(of device: SystemAudioOutputDevice) -> Bool? {
            guard canRead, let state = states[device.id], state.uid == device.uid else { return nil }
            return state.muted
        }
        func setMuted(_ muted: Bool, on device: SystemAudioOutputDevice) -> Bool {
            guard canWrite, muteState(of: device) != nil else { return false }
            states[device.id] = (device.uid, muted)
            writes.append("\(device.uid):\(muted)")
            return true
        }
        func observe(device: SystemAudioOutputDevice?, onChange: @escaping () -> Void) {
            changed = onChange
        }
        func stopObserving() { changed = nil }
        func install(_ device: SystemAudioOutputDevice, muted: Bool = false) {
            states[device.id] = (device.uid, muted)
        }
        func changeOutput(to device: SystemAudioOutputDevice?) {
            current = device
            changed?()
        }
    }

    private func devices(muted: Bool = false) -> Devices {
        let devices = Devices()
        devices.install(first, muted: muted)
        devices.current = first
        return devices
    }

    func testOnlyOwnMuteIsRestoredAndCallsAreIdempotent() {
        let devices = devices()
        let controller = SystemAudioMuteController(devices: devices)
        XCTAssertTrue(controller.muteSystemAudioIfNeeded())
        XCTAssertTrue(controller.muteSystemAudioIfNeeded())
        controller.restoreSystemAudioIfNeeded()
        controller.restoreSystemAudioIfNeeded()
        XCTAssertEqual(devices.writes, ["speaker:true", "speaker:false"])
        XCTAssertNil(devices.changed)
    }

    func testPreexistingUserMuteIsPreserved() {
        let devices = devices(muted: true)
        let controller = SystemAudioMuteController(devices: devices)
        XCTAssertTrue(controller.muteSystemAudioIfNeeded())
        controller.restoreSystemAudioIfNeeded()
        XCTAssertTrue(devices.writes.isEmpty)
        XCTAssertEqual(devices.muteState(of: first), true)
    }

    func testUnreadableOrUnwritableDeviceDoesNotAcquireOwnership() {
        for readable in [false, true] {
            let devices = devices()
            devices.canRead = readable
            devices.canWrite = false
            let controller = SystemAudioMuteController(devices: devices)
            XCTAssertFalse(controller.muteSystemAudioIfNeeded())
            devices.canRead = true
            devices.canWrite = true
            controller.restoreSystemAudioIfNeeded()
            XCTAssertTrue(devices.writes.isEmpty)
        }
    }

    func testOutputChangeRestoresOldDeviceAndMutesNewDevice() {
        let devices = devices()
        devices.install(second)
        let controller = SystemAudioMuteController(devices: devices)
        XCTAssertTrue(controller.muteSystemAudioIfNeeded())
        devices.changeOutput(to: second)
        XCTAssertEqual(devices.muteState(of: first), false)
        XCTAssertEqual(devices.muteState(of: second), true)
        controller.restoreSystemAudioIfNeeded()
        XCTAssertEqual(devices.writes, ["speaker:true", "speaker:false", "headphones:true", "headphones:false"])
    }

    func testObservedUserUnmuteRelinquishesOwnershipEvenIfUserMutesAgain() {
        let devices = devices()
        let controller = SystemAudioMuteController(devices: devices)
        XCTAssertTrue(controller.muteSystemAudioIfNeeded())
        devices.states[first.id] = (first.uid, false)
        devices.changed?()
        devices.states[first.id] = (first.uid, true)
        devices.changed?()
        controller.restoreSystemAudioIfNeeded()
        XCTAssertEqual(devices.writes, ["speaker:true"])
        XCTAssertEqual(devices.muteState(of: first), true)
    }

    func testReusedHALIdentifierCannotUnmuteAnotherDevice() {
        let devices = devices()
        let controller = SystemAudioMuteController(devices: devices)
        XCTAssertTrue(controller.muteSystemAudioIfNeeded())
        devices.states[first.id] = ("different-device", true)
        controller.restoreSystemAudioIfNeeded()
        XCTAssertEqual(devices.writes, ["speaker:true"])
    }

    func testQueuedNotificationCannotAffectNewSession() {
        let devices = devices()
        devices.install(second)
        let controller = SystemAudioMuteController(devices: devices)
        XCTAssertTrue(controller.muteSystemAudioIfNeeded())
        let stale = devices.changed
        controller.restoreSystemAudioIfNeeded()
        devices.current = second
        XCTAssertTrue(controller.muteSystemAudioIfNeeded())
        // A stale callback must not adopt a changed route in the new session.
        devices.current = first
        let count = devices.writes.count
        stale?()
        XCTAssertEqual(devices.writes.count, count)
        controller.restoreSystemAudioIfNeeded()
        XCTAssertEqual(devices.muteState(of: second), false)
    }

    func testRestoreFailureDoesNotRetryAgainstAUsersLaterState() {
        let devices = devices()
        let controller = SystemAudioMuteController(devices: devices)
        XCTAssertTrue(controller.muteSystemAudioIfNeeded())
        devices.canWrite = false
        controller.restoreSystemAudioIfNeeded()
        devices.canWrite = true
        controller.restoreSystemAudioIfNeeded()
        XCTAssertEqual(devices.writes, ["speaker:true"])
    }

    func testDeviceAvailableAfterInitialFailureCanBeAdopted() {
        let devices = Devices()
        let controller = SystemAudioMuteController(devices: devices)
        XCTAssertFalse(controller.muteSystemAudioIfNeeded())
        devices.install(second)
        devices.changeOutput(to: second)
        XCTAssertEqual(devices.muteState(of: second), true)
        controller.restoreSystemAudioIfNeeded()
        XCTAssertEqual(devices.muteState(of: second), false)
    }
}
