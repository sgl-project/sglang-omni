import Foundation

/// Which processes get Chromium's AXManualAccessibility, and which have
/// had it: Chromium builds a page's accessibility tree only when a client
/// asks for it, and only a moment later. The helper sets it as an app
/// comes to the front (Electron apps, found by their framework, and the
/// bundle ids app.configure names: the app's Chromium browsers), once per
/// process; browser.url sets it once more, per process, when it finds no
/// web area, then reads again.
///
/// Pure bookkeeping (the platform calls are the caller's), thread-safe.
public final class ManualAccessibility: @unchecked Sendable {
    /// Why the attribute is set: on activation, or by browser.url's retry.
    public enum Reason: Sendable { case activation, retry }

    private let lock = NSLock()
    private var bundles = Set<String>() // lowercased
    private var done: [Reason: Set<Int>] = [:]

    public init() {}

    /// Replaces the bundle ids (app.configure); case does not matter.
    public func configure(_ bundleIds: [String]) {
        let set = Set(bundleIds.map { $0.trimmingCharacters(in: .whitespaces).lowercased() }.filter { !$0.isEmpty })
        lock.withLock { bundles = set }
    }

    /// Whether an app is one of the configured bundle ids.
    public func listed(_ bundleId: String?) -> Bool {
        guard let id = bundleId?.lowercased(), !id.isEmpty else { return false }
        return lock.withLock { bundles.contains(id) }
    }

    /// Whether an app gets the attribute: an Electron app or a listed one.
    public func wants(bundleId: String?, electron: Bool) -> Bool {
        electron || listed(bundleId)
    }

    /// Claims pid for reason: true the first time only, when the caller
    /// sets the attribute (whether that succeeds or not).
    public func claim(pid: Int, _ reason: Reason) -> Bool {
        guard pid > 0 else { return false }
        return lock.withLock { done[reason, default: []].insert(pid).inserted }
    }

    /// Forgets a process that ended (its pid may come back for another).
    public func forget(pid: Int) {
        lock.withLock {
            for r in done.keys { done[r]?.remove(pid) }
        }
    }
}
