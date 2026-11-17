// hotkey.record's state: one recording at a time, the keys it heard, and
// how it ends (a chord, a timeout, a cancel). The event tap is the
// helper's (HotkeyRecorder); here it is a handle that starts and stops,
// so the locking is tested without Accessibility.
//
// The lock guards the state only: the tap is started, stopped and the
// recording ended outside it. Taking it again from inside (NSLock is not
// recursive) hung the helper on the first hotkey.record.

import Foundation

/// A key event the recording's tap saw.
public enum KeyRecordInput: Sendable, Equatable {
    case flagsChanged(flags: UInt64)
    case keyDown(keyCode: UInt16, flags: UInt64)
    case mouseDown(button: Int, flags: UInt64)
}

/// A recording's event tap, stopped once when the recording ends.
public protocol KeyRecordTap: AnyObject, Sendable {
    func stop()
}

public final class KeyRecordSession: @unchecked Sendable {
    /// Starts a tap that sends its events to the handler, which says
    /// whether to swallow each one; nil when no tap can be made.
    public typealias MakeTap = (@escaping @Sendable (KeyRecordInput) -> Bool) -> KeyRecordTap?

    private let lock = NSLock()
    private let trusted: @Sendable () -> Bool
    private let makeTap: MakeTap
    // Under lock.
    private var generation = 0
    private var done: (@Sendable (Result<String, RPCError>) -> Void)?
    private var tap: KeyRecordTap?
    private var activeMods = Set<ModKey>()
    private var peakMods = Set<ModKey>()

    public init(trusted: @escaping @Sendable () -> Bool, makeTap: @escaping MakeTap) {
        self.trusted = trusted
        self.makeTap = makeTap
    }

    /// Records one chord; done is called exactly once: with the chord in
    /// internal/hotkey's grammar, or permission_denied, timeout or
    /// cancelled. A recording already running ends cancelled.
    public func begin(timeoutSeconds: Double, done: @escaping @Sendable (Result<String, RPCError>) -> Void) {
        guard trusted() else {
            cancel()
            done(.failure(Self.denied))
            return
        }
        // Taking over and ending the earlier recording in one step: a
        // cancel, then the claim, let another begin slip in between, whose
        // recording this one overwrote, never ended nor stopped.
        let (gen, earlier): (Int, Ended?) = lock.withLock {
            let e = takeLocked()
            generation += 1
            self.done = done
            return (generation, e)
        }
        if let (t, d) = earlier {
            t?.stop()
            d(.failure(RPCError(Code.cancelled, "Cancelled")))
        }
        guard let t = makeTap({ [weak self] input in self?.handle(input, gen) ?? false }) else {
            end(gen, .failure(Self.denied))
            return
        }
        let current = lock.withLock {
            guard gen == generation, self.done != nil else { return false }
            tap = t
            return true
        }
        guard current else {
            t.stop() // ended while the tap started
            return
        }
        let timeout = timeoutSeconds > 0 ? timeoutSeconds : 15
        DispatchQueue.global().asyncAfter(deadline: .now() + timeout) { [weak self] in
            self?.end(gen, .failure(RPCError(Code.timeout, "Recording timed out")))
        }
    }

    /// Ends the recording that runs, if any, cancelled.
    public func cancel() {
        end(nil, .failure(RPCError(Code.cancelled, "Cancelled")))
    }

    /// Whether a recording runs.
    public var isRecording: Bool { lock.withLock { done != nil } }

    /// The tap's event for recording gen: swallowed while it records.
    func handle(_ input: KeyRecordInput, _ gen: Int) -> Bool {
        var chord: String?
        let recording: Bool = lock.withLock {
            guard gen == generation, done != nil else { return false }
            switch input {
            case .flagsChanged(let flags):
                let prev = activeMods
                activeMods = WireHotkeyRecord.mods(from: flags)
                peakMods.formUnion(activeMods)
                if !prev.isEmpty && activeMods.isEmpty {
                    chord = WireHotkeyRecord.chordString(mods: peakMods) // modifiers alone, on release
                }
            case .keyDown(let keyCode, let flags):
                if ModKey.forKeyCode(keyCode) == nil {
                    chord = WireHotkeyRecord.chordString(mods: WireHotkeyRecord.mods(from: flags), keyCode: keyCode)
                }
            case .mouseDown(let button, let flags):
                if button >= 2 {
                    chord = WireHotkeyRecord.chordString(mods: WireHotkeyRecord.mods(from: flags), mouseButton: button)
                }
            }
            return true
        }
        if let chord {
            end(gen, .success(chord))
        }
        return recording
    }

    /// Ends recording gen (nil: whichever runs) with result: outside the
    /// lock, its tap stops and done is called.
    private func end(_ gen: Int?, _ result: Result<String, RPCError>) {
        let ended: Ended? = lock.withLock {
            guard gen == nil || gen == generation else { return nil }
            return takeLocked()
        }
        guard let (t, d) = ended else { return }
        t?.stop()
        d(result)
    }

    /// A recording taken off the session: its tap and done.
    private typealias Ended = (KeyRecordTap?, @Sendable (Result<String, RPCError>) -> Void)

    /// Takes the recording that runs off the session (lock held), for the
    /// caller to stop and end outside the lock.
    private func takeLocked() -> Ended? {
        activeMods = []
        peakMods = []
        guard let d = done else { return nil }
        let t = tap
        done = nil
        tap = nil
        return (t, d)
    }

    private static let denied = RPCError.denied(PermissionName.accessibility, "Accessibility is required to record shortcuts")
}
