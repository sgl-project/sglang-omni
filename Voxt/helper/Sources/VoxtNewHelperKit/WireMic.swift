import Foundation

// mic.*: the apps recording from an audio input now, from Core Audio's
// list of processes (docs/helper-protocol.md, mic.watch). Nothing is
// recorded; the app decides which of them is a meeting.

public struct MicWatchParams: Codable, Sendable {
    public var on: Bool
    /// hosts are the names of processes that mark a meeting (Zoom's
    /// aomhost): MicUsers.hosts lists those that run.
    public var hosts: [String]?

    public init(on: Bool, hosts: [String]? = nil) {
        self.on = on
        self.hosts = hosts
    }
}

/// mic.watch's result and mic.changed's params: the processes recording
/// from an input, by pid, each with its pid and bundle id only; and, when
/// the watch named hosts, those of them that run while an app records
/// (none while none does).
public struct MicUsers: Codable, Sendable, Equatable {
    public var apps: [App]
    public var hosts: [MicHost]?

    public init(apps: [App] = [], hosts: [MicHost]? = nil) {
        self.apps = apps
        self.hosts = hosts
    }
}

/// A process mic.watch was asked about (MicWatchParams.hosts), by its
/// name, and the app it serves: the pid its `-pid` argument names, else
/// its parent's.
public struct MicHost: Codable, Sendable, Equatable {
    public var pid: Int
    public var name: String
    public var app: Int

    public init(pid: Int, name: String, app: Int) {
        self.pid = pid
        self.name = name
        self.app = app
    }
}

/// The pid a process serves, from its arguments ("-pid 758", as Zoom's
/// meeting processes are started), else nil.
public func servedPID(_ args: [String]) -> Int? {
    guard let i = args.firstIndex(of: "-pid"), i + 1 < args.count else { return nil }
    return Int(args[i + 1])
}

/// The arguments in a KERN_PROCARGS2 buffer: argc, the executable's path,
/// NULs, then argc strings (argv[0] first).
public func procArgs(_ raw: [UInt8]) -> [String] {
    let head = MemoryLayout<Int32>.size
    guard raw.count > head else { return [] }
    let argc = Int(raw.withUnsafeBytes { $0.loadUnaligned(as: Int32.self) })
    var i = head
    while i < raw.count && raw[i] != 0 { i += 1 } // the executable's path
    while i < raw.count && raw[i] == 0 { i += 1 }
    var args: [String] = []
    while args.count < argc && i < raw.count {
        let start = i
        while i < raw.count && raw[i] != 0 { i += 1 }
        args.append(String(decoding: raw[start..<i], as: UTF8.self))
        i += 1
    }
    return args
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

    /// Whether the watch is on (the helper reads again only then).
    public var isOn: Bool { lock.withLock { on } }

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
