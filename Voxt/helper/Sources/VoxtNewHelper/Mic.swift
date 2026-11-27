import CoreAudio
import Foundation
import VoxtNewHelperKit

/// mic.watch: which processes record from an audio input, from Core
/// Audio's process objects (kAudioHardwarePropertyProcessObjectList, each
/// one's kAudioProcessPropertyIsRunningInput), and mic.changed when they
/// change. Only these properties are read: no audio, no permission.
///
/// Core Audio's notices are not enough: on 11-25 Zoom joined a meeting and
/// no notice of its input came for 34 s. So the users are read again after
/// 100 ms, 300 ms, 1 s and 3 s, and every pollEvery while the watch is on.
final class Mic: @unchecked Sendable {
    let out: Writer
    private let watch = MicWatch()
    private let queue = DispatchQueue(label: "voxtnew.mic", qos: .utility)
    private let lock = NSLock()
    private var listening = false
    private var followed: [AudioObjectID: AudioObjectPropertyListenerBlock] = [:]
    private var pending = false
    private var hostNames: Set<String> = []
    private var poll: DispatchSourceTimer?
    private let own = [getpid(), getppid()]
    private static let pollEvery: DispatchTimeInterval = .seconds(2)
    private static let againAfter: [DispatchTimeInterval] = [.milliseconds(100), .milliseconds(300), .seconds(1), .seconds(3)]

    init(out: Writer) {
        self.out = out
    }

    func watch(_ p: MicWatchParams) -> MicUsers {
        lock.withLock { hostNames = Set(p.hosts ?? []) }
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
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + Self.pollEvery, repeating: Self.pollEvery, leeway: .milliseconds(500))
        timer.setEventHandler { [weak self] in self?.again() }
        timer.resume()
        lock.withLock { poll = timer }
    }

    /// Reads the users again while the watch is on, and tells the app
    /// when they are new. On queue.
    private func again() {
        guard watch.isOn, let users = watch.changed(read()) else { return }
        out.notify(Method.micChanged, users)
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
            // An app's input starts a moment after its process shows.
            for delay in Self.againAfter {
                queue.asyncAfter(deadline: .now() + delay) { [weak self] in self?.again() }
            }
        }
    }

    /// The users now; while listening, every process's input is followed.
    /// Only a process that records is asked who it is: most of the dozens
    /// Core Audio lists do not, and each bundle id is a round trip.
    private func read() -> MicUsers {
        let ids = Self.processes()
        if lock.withLock({ listening }) { follow(ids) }
        let recording = ids.filter { Self.number($0, kAudioProcessPropertyIsRunningInput) == 1 }
        var users = micUsers(recording.map { id in
            AudioProcess(pid: Int32(bitPattern: Self.number(id, kAudioProcessPropertyPID) ?? 0),
                         bundleId: stringProperty(id, kAudioProcessPropertyBundleID), input: true)
        }, ownPIDs: own)
        let names = lock.withLock { hostNames }
        if !names.isEmpty {
            // Looked for only while an app records: then it costs one
            // name per process.
            users.hosts = users.apps.isEmpty ? [] : Self.hosts(named: names)
        }
        return users
    }

    /// The processes named one of names, each with the app it serves.
    private static func hosts(named names: Set<String>) -> [MicHost] {
        let count = proc_listallpids(nil, 0)
        guard count > 0 else { return [] }
        var pids = [pid_t](repeating: 0, count: Int(count) + 64)
        let n = Int(proc_listallpids(&pids, Int32(pids.count * MemoryLayout<pid_t>.size)))
        var name = [CChar](repeating: 0, count: 256)
        var found: [MicHost] = []
        for pid in pids.prefix(max(n, 0)) where pid > 0 {
            guard proc_name(pid, &name, UInt32(name.count)) > 0 else { continue }
            let s = String(decoding: name.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
            guard names.contains(s) else { continue }
            found.append(MicHost(pid: Int(pid), name: s, app: servedPID(args(pid)) ?? parent(pid)))
        }
        return found.sorted { $0.pid < $1.pid }
    }

    private static func args(_ pid: pid_t) -> [String] {
        var mib: [Int32] = [CTL_KERN, KERN_ARGMAX]
        var argmax: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctl(&mib, 2, &argmax, &size, nil, 0) == 0, argmax > 0 else { return [] }
        var raw = [UInt8](repeating: 0, count: Int(argmax))
        mib = [CTL_KERN, KERN_PROCARGS2, pid]
        size = raw.count
        guard sysctl(&mib, 3, &raw, &size, nil, 0) == 0 else { return [] }
        return procArgs(Array(raw.prefix(size)))
    }

    private static func parent(_ pid: pid_t) -> Int {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout<proc_bsdinfo>.size)
        guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size else { return 0 }
        return Int(info.pbi_ppid)
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
