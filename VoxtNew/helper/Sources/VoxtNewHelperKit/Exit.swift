import Foundation

// What the helper does before it exits (helper.shutdown, stdin closed, a
// signal): a paste puts the old clipboard back 150–500 ms after ⌘V, so an
// exit in between would leave the dictated text on the clipboard. The
// rules are here, so that they are tested; the helper calls them.

/// The clipboard restores pastes still have to do. Thread-safe.
public final class PendingRestores: @unchecked Sendable {
    private let cond = NSCondition()
    private var pending: [Int: @Sendable () -> Void] = [:]
    private var next = 0
    /// A test seam: called once finish waits.
    var willWait: (@Sendable () -> Void)?

    public init() {}

    /// A paste's restore to come; restore does it now (an exit that cannot
    /// wait). The paste calls done with the id once it restored, or chose
    /// not to.
    public func add(_ restore: @escaping @Sendable () -> Void) -> Int {
        cond.withLock {
            next += 1
            pending[next] = restore
            return next
        }
    }

    public func done(_ id: Int) {
        cond.withLock {
            pending[id] = nil
            cond.broadcast()
        }
    }

    /// Waits up to limit for the pastes to restore, then does the rest.
    public func finish(within limit: TimeInterval) {
        let deadline = Date(timeIntervalSinceNow: limit)
        let rest = cond.withLock { () -> [Int: @Sendable () -> Void] in
            if !pending.isEmpty { willWait?() }
            while !pending.isEmpty, cond.wait(until: deadline) {}
            defer { pending = [:] }
            return pending
        }
        for (_, restore) in rest.sorted(by: { $0.key < $1.key }) { restore() }
    }
}

/// Before an exit: the clipboard first (at most 500 ms), then the
/// recordings, then the output's mutes.
public func beforeExit(_ restores: PendingRestores, within limit: TimeInterval = 0.5, cancelCaptures: () -> Void, undoOutput: () -> Void) {
    restores.finish(within: limit)
    cancelCaptures()
    undoOutput()
}
