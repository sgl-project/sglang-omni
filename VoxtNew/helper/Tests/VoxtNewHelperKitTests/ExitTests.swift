import Foundation
import XCTest
@testable import VoxtNewHelperKit

/// The helper used to exit (helper.shutdown, stdin closed, a signal) while
/// a paste still had to put the old clipboard back, leaving the dictated
/// text on it; a signal also left recordings running.
final class ExitTests: XCTestCase {
    final class Log: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []
        func add(_ s: String) { lock.withLock { lines.append(s) } }
        var all: [String] { lock.withLock { lines } }
    }

    /// A paste whose restore is due soon: the exit waits for it, then
    /// cancels the recordings and undoes the mutes.
    func testExitWaitsForTheClipboard() {
        let r = PendingRestores(), log = Log()
        let id = r.add { log.add("restored by the exit") }
        let due = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            due.wait() // the paste's delay ends
            log.add("restored by the paste")
            r.done(id)
        }
        r.willWait = { due.signal() } // only once the exit waits
        beforeExit(r, within: 30, cancelCaptures: { log.add("captures") }, undoOutput: { log.add("output") })
        XCTAssertEqual(log.all, ["restored by the paste", "captures", "output"])
    }

    /// A restore that does not come in time is done by the exit.
    func testExitDoesALateRestore() {
        let r = PendingRestores(), log = Log()
        _ = r.add { log.add("restored by the exit") }
        let done = r.add { log.add("never") }
        r.done(done)
        beforeExit(r, within: 0, cancelCaptures: { log.add("captures") }, undoOutput: { log.add("output") })
        XCTAssertEqual(log.all, ["restored by the exit", "captures", "output"])
    }
}
