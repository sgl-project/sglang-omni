import CoreAudio
import Foundation
import VoxtNewHelperKit

/// mic.watch: which processes record from an audio input, from Core
/// Audio's process objects (kAudioHardwarePropertyProcessObjectList, each
/// one's kAudioProcessPropertyIsRunningInput), and mic.changed when they
/// change. Only these properties are read: no audio, no permission.
final class Mic: @unchecked Sendable {
    let out: Writer
    private let watch = MicWatch()
    private let queue = DispatchQueue(label: "voxtnew.mic", qos: .utility)
    private let lock = NSLock()
    private var listening = false
    private var followed: [AudioObjectID: AudioObjectPropertyListenerBlock] = [:]
    private var pending = false
    private let own = [getpid(), getppid()]

    init(out: Writer) {
        self.out = out
    }

    func watch(_ p: MicWatchParams) -> MicUsers {
        if p.on { listen() }
        let now = read()
        watch.set(p.on, at: now)
        return now
    }

    /// Listens to the list of processes (once); read follows each one.
    private func listen() {
        let first = lock.withLock { () -> Bool in
            defer { listening = true }
            return !listening
        }
        guard first else { return }
        var addr = Self.address(kAudioHardwarePropertyProcessObjectList)
        let status = AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &addr, queue) { [weak self] _, _ in
            self?.changed()
        }
        if status != noErr { log("mic: no listener for the audio processes (\(status))") }
    }

    /// A change: read once Core Audio settled (50 ms; it tells a change
    /// several times), then tell the app when the users are new.
    private func changed() {
        let schedule = lock.withLock { () -> Bool in
            defer { pending = true }
            return !pending
        }
        guard schedule else { return }
        queue.asyncAfter(deadline: .now() + .milliseconds(50)) { [self] in
            lock.withLock { pending = false }
            if let users = watch.changed(read()) {
                out.notify(Method.micChanged, users)
            }
        }
    }

    /// The users now; while listening, every process's input is followed.
    private func read() -> MicUsers {
        let ids = Self.processes()
        if lock.withLock({ listening }) { follow(ids) }
        return micUsers(ids.map { id in
            AudioProcess(pid: Int32(bitPattern: Self.number(id, kAudioProcessPropertyPID) ?? 0),
                         bundleId: stringProperty(id, kAudioProcessPropertyBundleID),
                         input: Self.number(id, kAudioProcessPropertyIsRunningInput) == 1)
        }, ownPIDs: own)
    }

    /// Listens to the input of each process in ids, and forgets the ones
    /// that are gone (their objects, and so their listeners, went too).
    private func follow(_ ids: [AudioObjectID]) {
        lock.withLock {
            followed = followed.filter { ids.contains($0.key) }
            for id in ids where followed[id] == nil {
                var addr = Self.address(kAudioProcessPropertyIsRunningInput)
                let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in self?.changed() }
                if AudioObjectAddPropertyListenerBlock(id, &addr, queue, block) == noErr {
                    followed[id] = block
                }
            }
        }
    }

    private static func address(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    }

    private static func processes() -> [AudioObjectID] {
        var addr = address(kAudioHardwarePropertyProcessObjectList)
        var size: UInt32 = 0
        let system = AudioObjectID(kAudioObjectSystemObject)
        guard AudioObjectGetPropertyDataSize(system, &addr, 0, nil, &size) == noErr, size > 0 else { return [] }
        var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(system, &addr, 0, nil, &size, &ids) == noErr else { return [] }
        return Array(ids.prefix(Int(size) / MemoryLayout<AudioObjectID>.size))
    }

    private static func number(_ id: AudioObjectID, _ selector: AudioObjectPropertySelector) -> UInt32? {
        var addr = address(selector)
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        return AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &value) == noErr ? value : nil
    }
}
