import AppKit
import VoxtNewHelperKit
import CoreAudio
import Foundation

/// output.*: the default output device's mute (kAudioDevicePropertyMute,
/// output scope) through Core Audio, the claims (MuteClaims: what the
/// helper muted, journaled), and the watch: output.changed when the
/// default output device or its mute changes. The claims are undone when
/// the app goes away (stdin closing, helper.shutdown, SIGTERM, SIGINT,
/// SIGHUP) and before the Mac sleeps (then output.released tells the app).
final class Output: @unchecked Sendable {
    let out: Writer
    let hardware = CoreAudioOutput()
    let claims: MuteClaims
    private let watch = OutputWatch()
    private let queue = DispatchQueue(label: "voxtnew.output", qos: .userInitiated)
    private let lock = NSLock()
    private var listening = false
    private var muteListener: (AudioObjectID, AudioObjectPropertyListenerBlock)?
    private var pending = false
    private var signals: [DispatchSourceSignal] = []
    private var started = false

    init(out: Writer) {
        self.out = out
        claims = MuteClaims(hardware: hardware)
    }

    /// Starts the undo on signals and sleep (once, at hello).
    func start() {
        let first = lock.withLock { () -> Bool in
            defer { started = true }
            return !started
        }
        guard first else { return }
        for sig in [SIGTERM, SIGINT, SIGHUP] {
            signal(sig, SIG_IGN)
            let src = DispatchSource.makeSignalSource(signal: sig, queue: queue)
            src.setEventHandler { [self] in
                undo("signal \(sig)")
                exit(0)
            }
            src.resume()
            signals.append(src)
        }
        DispatchQueue.main.async { [self] in
            NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.willSleepNotification, object: nil, queue: nil) { [weak self] _ in
                guard let self else { return }
                let gone = self.claims.releaseAll()
                if !gone.isEmpty {
                    log("output: unmuted \(gone.count) device(s) before sleep")
                    self.out.notify(Method.outputReleased, OutputReleasedParams(devices: gone, reason: OutputReason.sleep))
                }
            }
        }
    }

    /// Undoes the claims as the helper exits.
    func undo(_ why: String) {
        let gone = claims.releaseAll()
        if !gone.isEmpty { log("output: unmuted \(gone.count) device(s): \(why)") }
    }

    func watch(_ p: OutputWatchParams) -> OutputState {
        if p.on { listen() }
        let now = claims.state()
        watch.set(p.on, at: now)
        if p.on { follow(now.device) }
        return now
    }

    /// Listens to the default output device (once), and to the mute of the
    /// one now.
    private func listen() {
        let first = lock.withLock { () -> Bool in
            defer { listening = true }
            return !listening
        }
        guard first else { return }
        var addr = CoreAudioOutput.defaultOutputAddress
        let status = AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &addr, queue) { [weak self] _, _ in
            self?.changed()
        }
        if status != noErr { log("output: no listener for the default output (\(status))") }
    }

    /// Moves the mute listener to device (the default output now).
    private func follow(_ device: OutputDevice?) {
        lock.withLock {
            if let (id, block) = muteListener {
                if id == device?.id { return }
                var addr = CoreAudioOutput.muteAddress
                AudioObjectRemovePropertyListenerBlock(id, &addr, queue, block)
                muteListener = nil
            }
            guard let d = device else { return }
            var addr = CoreAudioOutput.muteAddress
            guard AudioObjectHasProperty(d.id, &addr) else { return }
            let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in self?.changed() }
            if AudioObjectAddPropertyListenerBlock(d.id, &addr, queue, block) == noErr {
                muteListener = (d.id, block)
            }
        }
    }

    /// A change: read once Core Audio settled (20 ms; it tells a change
    /// several times), then tell the app when the state is new.
    private func changed() {
        let schedule = lock.withLock { () -> Bool in
            defer { pending = true }
            return !pending
        }
        guard schedule else { return }
        queue.asyncAfter(deadline: .now() + .milliseconds(20)) { [self] in
            lock.withLock { pending = false }
            let now = claims.state()
            follow(now.device)
            if let s = watch.changed(now) {
                out.notify(Method.outputChanged, s)
            }
        }
    }
}

/// Core Audio's output devices, found by UID: an id may be reused by
/// another device after one is unplugged, and changes when one is
/// reconnected.
final class CoreAudioOutput: OutputHardware, @unchecked Sendable {
    static var defaultOutputAddress: AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice,
                                   mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    }

    static var muteAddress: AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyMute,
                                   mScope: kAudioDevicePropertyScopeOutput, mElement: kAudioObjectPropertyElementMain)
    }

    func defaultOutput() -> OutputDevice? {
        var addr = Self.defaultOutputAddress
        var id = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &id) == noErr,
              id != kAudioObjectUnknown, let uid = string(id, kAudioDevicePropertyDeviceUID) else { return nil }
        return OutputDevice(id: id, uid: uid, name: string(id, kAudioObjectPropertyName))
    }

    func device(uid: String) -> OutputDevice? {
        var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyTranslateUIDToDevice,
                                              mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var cf = uid as CFString
        var id = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = withUnsafeMutablePointer(to: &cf) { p in
            AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr,
                                       UInt32(MemoryLayout<CFString>.size), p, &size, &id)
        }
        guard status == noErr, id != kAudioObjectUnknown, string(id, kAudioDevicePropertyDeviceUID) == uid else { return nil }
        return OutputDevice(id: id, uid: uid, name: string(id, kAudioObjectPropertyName))
    }

    func muted(_ d: OutputDevice) -> Bool? {
        var addr = Self.muteAddress
        guard AudioObjectHasProperty(d.id, &addr) else { return nil }
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(d.id, &addr, 0, nil, &size, &value) == noErr else { return nil }
        return value != 0
    }

    func settable(_ d: OutputDevice) -> Bool {
        var addr = Self.muteAddress
        var settable: DarwinBoolean = false
        return AudioObjectHasProperty(d.id, &addr) && AudioObjectIsPropertySettable(d.id, &addr, &settable) == noErr && settable.boolValue
    }

    func setMuted(_ muted: Bool, _ d: OutputDevice) -> Bool {
        guard settable(d) else { return false }
        var addr = Self.muteAddress
        var value: UInt32 = muted ? 1 : 0
        return AudioObjectSetPropertyData(d.id, &addr, 0, nil, UInt32(MemoryLayout<UInt32>.size), &value) == noErr
    }

    /// A CFString property of a device.
    private func string(_ id: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
        var addr = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &value) == noErr, let v = value else { return nil }
        return v.takeRetainedValue() as String
    }
}
