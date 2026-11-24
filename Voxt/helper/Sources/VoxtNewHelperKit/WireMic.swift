import Foundation

// mic.*: the apps recording from an audio input now, from Core Audio's
// list of processes (docs/helper-protocol.md, mic.watch). Nothing is
// recorded; the app decides which of them is a meeting.

public struct MicWatchParams: Codable, Sendable {
    public var on: Bool

    public init(on: Bool) {
        self.on = on
    }
}

/// mic.watch's result and mic.changed's params: the processes recording
/// from an input, by pid, each with its pid and bundle id only.
public struct MicUsers: Codable, Sendable, Equatable {
    public var apps: [App]

    public init(apps: [App] = []) {
        self.apps = apps
    }
}

/// A process as Core Audio lists it: whether it runs an input now.
public struct AudioProcess: Equatable, Sendable {
    public var pid: Int32
    public var bundleId: String?
    public var input: Bool

    public init(pid: Int32, bundleId: String?, input: Bool) {
        self.pid = pid
        self.bundleId = bundleId
        self.input = input
    }
}

/// The processes of list that record from an input, without ownPIDs (the
/// helper and the app: a dictation is no meeting), each pid once, by pid.
public func micUsers(_ list: [AudioProcess], ownPIDs: [Int32]) -> MicUsers {
    var seen = Set(ownPIDs)
    var apps: [App] = []
    for p in list.sorted(by: { $0.pid < $1.pid }) where p.input && seen.insert(p.pid).inserted {
        let id = p.bundleId.flatMap { $0.isEmpty ? nil : $0 }
        apps.append(App(pid: Int(p.pid), bundleId: id, name: nil))
    }
    return MicUsers(apps: apps)
}

/// mic.watch's notifications: the users are sent only when they differ
/// from the last ones sent (Core Audio tells a change several times).
/// Thread-safe.
public final class MicWatch: @unchecked Sendable {
    private let lock = NSLock()
    private var on = false
    private var last = MicUsers()

    public init() {}

    /// Turns it on or off at the users now (the result of mic.watch).
    public func set(_ on: Bool, at now: MicUsers) {
        lock.withLock {
            self.on = on
            last = now
        }
    }

    /// The users to send after a change, nil when off or unchanged.
    public func changed(_ now: MicUsers) -> MicUsers? {
        lock.withLock {
            guard on, now != last else { return nil }
            last = now
            return now
        }
    }
}
