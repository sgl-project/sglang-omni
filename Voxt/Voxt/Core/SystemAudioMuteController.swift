// Device mute changes output state only; it never captures system audio.
import Foundation
import CoreAudio

struct SystemAudioOutputDevice: Equatable {
    let id: AudioDeviceID
    let uid: String
}

protocol SystemAudioMuteDeviceAccess: AnyObject {
    func defaultOutputDevice() -> SystemAudioOutputDevice?
    func muteState(of device: SystemAudioOutputDevice) -> Bool?
    func setMuted(_ muted: Bool, on device: SystemAudioOutputDevice) -> Bool
    func observe(device: SystemAudioOutputDevice?, onChange: @escaping () -> Void)
    func stopObserving()
}

final class SystemAudioMuteController: @unchecked Sendable {
    private let devices: any SystemAudioMuteDeviceAccess
    private let operationQueue = DispatchQueue(label: "com.voxt.systemAudioMute", qos: .userInitiated)
    private var sessionID: UUID?
    private var outputDevice: SystemAudioOutputDevice?
    private var ownedMute: SystemAudioOutputDevice?

    convenience init() {
        self.init(devices: CoreAudioMuteDeviceAccess())
    }

    init(devices: any SystemAudioMuteDeviceAccess) {
        self.devices = devices
    }

    deinit {
        devices.stopObserving()
        restoreOwnedMute()
    }

    @discardableResult
    func muteSystemAudioIfNeeded() -> Bool {
        operationQueue.sync {
            performMuteSystemAudioIfNeeded()
        }
    }

    func muteSystemAudioIfNeededAsync(completion: @escaping @MainActor (Bool) -> Void) {
        operationQueue.async { [weak self] in
            guard let self else { return }
            let result = self.performMuteSystemAudioIfNeeded()
            Task { @MainActor in
                completion(result)
            }
        }
    }

    private func performMuteSystemAudioIfNeeded() -> Bool {
        if sessionID != nil {
            return outputDevice.flatMap { devices.muteState(of: $0) } == true
        }
        let id = UUID()
        sessionID = id
        let muted = adoptCurrentOutput()
        observeOutput(sessionID: id)
        return muted
    }

    func restoreSystemAudioIfNeeded() {
        operationQueue.sync {
            performRestoreSystemAudioIfNeeded()
        }
    }

    private func performRestoreSystemAudioIfNeeded() {
        sessionID = nil // Invalidates already queued property notifications.
        devices.stopObserving()
        restoreOwnedMute()
        outputDevice = nil
    }

    private func adoptCurrentOutput() -> Bool {
        outputDevice = devices.defaultOutputDevice()
        guard let device = outputDevice, let wasMuted = devices.muteState(of: device) else {
            VoxtLog.warning("Output-device mute unavailable: mute state cannot be read.")
            return false
        }
        // Never take ownership of a user's pre-existing mute.
        guard !wasMuted else { return true }
        guard devices.setMuted(true, on: device) else {
            VoxtLog.warning("Output-device mute unavailable: device has no writable mute control.")
            return false
        }
        ownedMute = device
        return true
    }

    private func observeOutput(sessionID id: UUID) {
        devices.observe(device: outputDevice) { [weak self] in
            guard let self else { return }
            self.operationQueue.sync {
                self.handleOutputChange(sessionID: id)
            }
        }
    }

    private func handleOutputChange(sessionID id: UUID) {
        guard self.sessionID == id else { return }
        let current = devices.defaultOutputDevice()
        if current != outputDevice {
            devices.stopObserving()
            restoreOwnedMute()
            _ = adoptCurrentOutput()
            observeOutput(sessionID: id)
        } else if let owned = ownedMute,
                  devices.muteState(of: owned) == false {
            // Respect an observed external unmute; do not fight the user or
            // later unmute a state they set themselves in this session.
            ownedMute = nil
        }
    }

    private func restoreOwnedMute() {
        guard let owned = ownedMute else { return }
        ownedMute = nil
        // Access checks the UID as well as the ID: HAL may reuse an ID after
        // unplugging. Unknown state must not be treated as our original device.
        guard let isMuted = devices.muteState(of: owned) else {
            VoxtLog.warning("Output-device mute could not be restored: device disconnected or state unavailable. Check output mute manually.")
            return
        }
        if isMuted, !devices.setMuted(false, on: owned) {
            VoxtLog.warning("Output-device mute restore failed. Check output mute manually.")
        }
    }
}

private final class CoreAudioMuteDeviceAccess: SystemAudioMuteDeviceAccess {
    private struct Observation {
        let object: AudioObjectID
        var address: AudioObjectPropertyAddress
        let block: AudioObjectPropertyListenerBlock
    }
    private var observations: [Observation] = []

    func defaultOutputDevice() -> SystemAudioOutputDevice? {
        var address = Self.defaultOutputAddress
        var id = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &id) == noErr,
              id != kAudioObjectUnknown, let uid = deviceUID(id) else { return nil }
        return SystemAudioOutputDevice(id: id, uid: uid)
    }

    func muteState(of device: SystemAudioOutputDevice) -> Bool? {
        guard deviceUID(device.id) == device.uid else { return nil }
        var address = Self.muteAddress
        guard AudioObjectHasProperty(device.id, &address) else { return nil }
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(device.id, &address, 0, nil, &size, &value) == noErr else { return nil }
        return value != 0
    }

    func setMuted(_ muted: Bool, on device: SystemAudioOutputDevice) -> Bool {
        guard deviceUID(device.id) == device.uid else { return false }
        var address = Self.muteAddress
        var settable: DarwinBoolean = false
        guard AudioObjectHasProperty(device.id, &address),
              AudioObjectIsPropertySettable(device.id, &address, &settable) == noErr,
              settable.boolValue else { return false }
        var value: UInt32 = muted ? 1 : 0
        return AudioObjectSetPropertyData(device.id, &address, 0, nil, UInt32(MemoryLayout<UInt32>.size), &value) == noErr
    }

    func observe(device: SystemAudioOutputDevice?, onChange: @escaping () -> Void) {
        stopObserving()
        addObservation(object: AudioObjectID(kAudioObjectSystemObject), address: Self.defaultOutputAddress, onChange: onChange)
        if let device {
            addObservation(object: device.id, address: Self.muteAddress, onChange: onChange)
        }
    }

    func stopObserving() {
        for var observation in observations {
            AudioObjectRemovePropertyListenerBlock(observation.object, &observation.address, .main, observation.block)
        }
        observations.removeAll()
    }

    private func addObservation(
        object: AudioObjectID,
        address: AudioObjectPropertyAddress,
        onChange: @escaping @MainActor () -> Void
    ) {
        var address = address
        let block: AudioObjectPropertyListenerBlock = { _, _ in
            Task { @MainActor in onChange() }
        }
        guard AudioObjectAddPropertyListenerBlock(object, &address, .main, block) == noErr else {
            VoxtLog.warning("Output-device mute observation unavailable.")
            return
        }
        observations.append(Observation(object: object, address: address, block: block))
    }

    private func deviceUID(_ id: AudioDeviceID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let storage = UnsafeMutablePointer<CFString?>.allocate(capacity: 1)
        storage.initialize(to: nil)
        defer { storage.deinitialize(count: 1); storage.deallocate() }
        var size = UInt32(MemoryLayout<CFString?>.size)
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, storage) == noErr else { return nil }
        return storage.pointee.map { $0 as String }
    }

    private static var muteAddress: AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyMute,
            mScope: kAudioDevicePropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMain
        )
    }

    private static var defaultOutputAddress: AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
    }
}
