import Foundation

// The system's output, muted while VoxtNew records (output.*,
// docs/helper-protocol.md). Only the mute of an output device changes:
// nothing of the system's sound is read, so no permission is needed. The
// app decides when; the helper keeps its claims (the mutes it made) and
// undoes only those: a mute the user made is never claimed. Each claim is
// written to a journal file (<dir>/<pid>.json) before the mute, so that a
// helper that crashed or was killed leaves it for the next one
// (output.recover). The Go fake (internal/helper/helpertest/output.go)
// follows the same rules.
//
// Core Audio is behind OutputHardware (the executable's CoreAudioOutput),
// so the rules are tested with a fake (OutputMuteTests).

/// The output devices as Core Audio has them.
public protocol OutputHardware: AnyObject, Sendable {
    /// The default output device (its id, UID and name), nil for none.
    func defaultOutput() -> OutputDevice?
    /// The device with this UID now (its id may have changed), nil when
    /// none has it.
    func device(uid: String) -> OutputDevice?
    /// Its mute, nil when it has none to read.
    func muted(_ device: OutputDevice) -> Bool?
    /// Whether its mute can be set.
    func settable(_ device: OutputDevice) -> Bool
    /// Sets its mute; false when Core Audio refused.
    func setMuted(_ muted: Bool, _ device: OutputDevice) -> Bool
}

/// A helper's claims on output devices. Thread-safe.
public final class MuteClaims: @unchecked Sendable {
    private let hardware: OutputHardware
    private let pid: Int32
    private let alive: @Sendable (Int32) -> Bool
    private let lock = NSLock()
    private var claims: [OutputDevice] = []
    private var journal: String?

    /// pid names this helper's journal file; alive tells whether another
    /// helper's process runs (its journal is its own).
    public init(hardware: OutputHardware, pid: Int32 = getpid(), alive: @escaping @Sendable (Int32) -> Bool = processRuns) {
        self.hardware = hardware
        self.pid = pid
        self.alive = alive
    }

    /// The devices it muted.
    public var claimed: [OutputDevice] { lock.withLock { claims } }

    /// The default output and its mute.
    public func state() -> OutputState {
        guard let d = hardware.defaultOutput() else { return OutputState() }
        return state(of: d)
    }

    func state(of d: OutputDevice) -> OutputState {
        guard let m = hardware.muted(d) else { return OutputState(device: d) }
        return OutputState(device: d, muted: m, settable: hardware.settable(d))
    }

    /// output.mute: mutes the device and claims it, unless it is muted
    /// already.
    public func mute(_ p: OutputMuteParams) throws -> OutputMuteResult {
        try lock.withLock {
            guard let d = hardware.device(uid: p.device.uid) else {
                throw RPCError(Code.deviceUnavailable, "No output device has the UID \(p.device.uid)",
                               data: .object(["device": .string(p.device.uid)]))
            }
            let st = state(of: d)
            guard let muted = st.muted, st.settable else {
                throw RPCError(Code.unsupported, "\(d.name ?? "The device") has no mute the helper can set",
                               data: .object(["capability": .string("mute")]))
            }
            if muted { return OutputMuteResult(took: false, state: st) }
            if let j = p.journal, !j.isEmpty { journal = j }
            var next = claims.filter { $0.uid != d.uid }
            next.append(d)
            do {
                try writeJournal(next)
            } catch {
                throw RPCError(Code.internal, "Could not write the journal: \(errorMessage(error))")
            }
            guard hardware.setMuted(true, d) else {
                try? writeJournal(claims)
                throw RPCError(Code.unsupported, "\(d.name ?? "The device") did not take the mute",
                               data: .object(["capability": .string("mute")]))
            }
            claims = next
            return OutputMuteResult(took: true, state: state(of: d))
        }
    }

    /// output.release: gives the claim back, unmuting the device if asked
    /// and it is still muted. Without a claim on it, nothing.
    public func release(_ p: OutputReleaseParams) -> OutputReleaseResult {
        lock.withLock {
            guard let i = claims.firstIndex(where: { $0.uid == p.device.uid }) else { return OutputReleaseResult(unmuted: false) }
            claims.remove(at: i)
            try? writeJournal(claims)
            guard p.unmute == true else { return OutputReleaseResult(unmuted: false) }
            return OutputReleaseResult(unmuted: unmuteIfMuted(uid: p.device.uid))
        }
    }

    /// Undoes every claim (stdin closed, helper.shutdown, a signal, sleep)
    /// and says which it gave back.
    @discardableResult
    public func releaseAll() -> [OutputDevice] {
        lock.withLock {
            let gone = claims
            claims = []
            try? writeJournal([])
            for d in gone { _ = unmuteIfMuted(uid: d.uid) }
            return gone
        }
    }

    /// output.recover: undoes the journals of helpers that are gone.
    public func recover(_ p: OutputRecoverParams) -> OutputRecoverResult {
        lock.withLock {
            // No journal is nothing to do: never the working directory.
            guard !p.journal.isEmpty else { return OutputRecoverResult(restored: nil) }
            let dir = URL(fileURLWithPath: p.journal)
            let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
            var restored: [OutputDevice] = []
            for f in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) where f.pathExtension == "json" && Int32(f.deletingPathExtension().lastPathComponent) != nil {
                guard let data = try? Data(contentsOf: f), let j = try? JSONDecoder().decode(JournalFile.self, from: data) else {
                    try? FileManager.default.removeItem(at: f) // unreadable: nothing to undo
                    continue
                }
                if j.pid == pid || alive(j.pid) { continue } // a running helper's claims are its own
                for d in j.devices where unmuteIfMuted(uid: d.uid) {
                    restored.append(d)
                }
                try? FileManager.default.removeItem(at: f)
            }
            return OutputRecoverResult(restored: restored.isEmpty ? nil : restored)
        }
    }

    /// Unmutes the device with uid when it is there and muted. Lock held.
    private func unmuteIfMuted(uid: String) -> Bool {
        guard let d = hardware.device(uid: uid), hardware.muted(d) == true else { return false }
        return hardware.setMuted(false, d)
    }

    /// Keeps claims in the journal; none removes the file. Lock held.
    private func writeJournal(_ list: [OutputDevice]) throws {
        guard let dir = journal, !dir.isEmpty else { return }
        let url = URL(fileURLWithPath: dir).appendingPathComponent("\(pid).json")
        if list.isEmpty {
            if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
            return
        }
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let data = try makeEncoder().encode(JournalFile(pid: pid, devices: list))
        try data.write(to: url, options: .atomic)
    }

    struct JournalFile: Codable {
        var pid: Int32
        var devices: [OutputDevice]
    }
}

/// Whether a process runs (a signal 0 reaches it).
public func processRuns(_ pid: Int32) -> Bool {
    kill(pid, 0) == 0 || errno == EPERM
}

/// output.watch's notifications: a state is sent only when it differs from
/// the last one sent (Core Audio tells a change several times). Thread-safe.
public final class OutputWatch: @unchecked Sendable {
    private let lock = NSLock()
    private var on = false
    private var last: OutputState?

    public init() {}

    public var isOn: Bool { lock.withLock { on } }

    /// Turns it on or off at state now (the result of output.watch).
    public func set(_ on: Bool, at now: OutputState) {
        lock.withLock {
            self.on = on
            last = now
        }
    }

    /// The state to send after a change, nil when off or unchanged.
    public func changed(_ now: OutputState) -> OutputState? {
        lock.withLock {
            guard on, now != last else { return nil }
            last = now
            return now
        }
    }
}
