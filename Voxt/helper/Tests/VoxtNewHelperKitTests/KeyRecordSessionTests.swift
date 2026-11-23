import Foundation
import XCTest
@testable import VoxtNewHelperKit

/// A tap that hands its handler to the test and counts its stops.
private final class FakeTap: KeyRecordTap, @unchecked Sendable {
    let handler: @Sendable (KeyRecordInput) -> Bool
    private let lock = NSLock()
    private var stopsCount = 0
    var stops: Int { lock.withLock { stopsCount } }

    init(_ handler: @escaping @Sendable (KeyRecordInput) -> Bool) { self.handler = handler }

    func stop() { lock.withLock { stopsCount += 1 } }
}

/// The taps a session made, in order.
private final class Taps: @unchecked Sendable {
    private let lock = NSLock()
    private var made: [FakeTap] = []
    var fail = false
    var all: [FakeTap] { lock.withLock { made } }
    var last: FakeTap? { all.last }

    func make(_ handler: @escaping @Sendable (KeyRecordInput) -> Bool) -> KeyRecordTap? {
        if fail { return nil }
        let t = FakeTap(handler)
        lock.withLock { made.append(t) }
        return t
    }
}

/// The results a session gave, in order.
private final class Results: @unchecked Sendable {
    private let lock = NSLock()
    private var got: [Result<String, RPCError>] = []
    var all: [Result<String, RPCError>] { lock.withLock { got } }

    func add(_ r: Result<String, RPCError>) { lock.withLock { got.append(r) } }

    func code(_ i: Int) -> String? {
        guard i < all.count, case .failure(let e) = all[i] else { return nil }
        return e.code
    }

    func keys(_ i: Int) -> String? {
        guard i < all.count, case .success(let k) = all[i] else { return nil }
        return k
    }
}

private let leftOption = RawFlags.option | 0x0020

final class KeyRecordSessionTests: XCTestCase {
    private func session(trusted: Bool = true, _ taps: Taps) -> KeyRecordSession {
        KeyRecordSession(trusted: { trusted }, makeTap: { taps.make($0) })
    }

    /// ⌥Q (the user's 11-09 case): the chord ends the recording, the tap
    /// stops once, the keys are swallowed. The first hotkey.record used to
    /// hang the helper on its own lock before it read any key.
    func testOptionQ() {
        let taps = Taps(), results = Results(), s = session(taps)
        s.begin(timeoutSeconds: 15) { results.add($0) }
        guard let tap = taps.last else { return XCTFail("no tap") }
        XCTAssertTrue(tap.handler(.flagsChanged(flags: leftOption)))
        XCTAssertTrue(tap.handler(.keyDown(keyCode: 12, flags: leftOption)))
        XCTAssertEqual(results.keys(0), "LeftOption+Q")
        XCTAssertEqual(tap.stops, 1)
        XCTAssertFalse(s.isRecording)
        XCTAssertFalse(tap.handler(.flagsChanged(flags: 0)), "the release after the chord goes on to the app")
        XCTAssertEqual(results.all.count, 1)
    }

    /// A modifier alone records on its release.
    func testModifierAlone() {
        let taps = Taps(), results = Results(), s = session(taps)
        s.begin(timeoutSeconds: 15) { results.add($0) }
        let tap = taps.last!
        _ = tap.handler(.flagsChanged(flags: RawFlags.command | 0x0010))
        XCTAssertTrue(results.all.isEmpty)
        _ = tap.handler(.flagsChanged(flags: 0))
        XCTAssertEqual(results.keys(0), "RightCommand")
    }

    /// Cancel, then record again: each recording ends once and stops its
    /// own tap; a recording begun while one runs ends that one cancelled.
    func testCancelAndRecordAgain() {
        let taps = Taps(), results = Results(), s = session(taps)
        s.begin(timeoutSeconds: 15) { results.add($0) }
        s.cancel()
        s.cancel() // nothing runs: nothing happens
        XCTAssertEqual(results.code(0), Code.cancelled)
        s.begin(timeoutSeconds: 15) { results.add($0) }
        s.begin(timeoutSeconds: 15) { results.add($0) }
        XCTAssertEqual(results.code(1), Code.cancelled)
        _ = taps.last!.handler(.keyDown(keyCode: 49, flags: RawFlags.control | 0x0001))
        XCTAssertEqual(results.keys(2), "LeftControl+Space")
        XCTAssertEqual(taps.all.map(\.stops), [1, 1, 1])
        _ = taps.all[0].handler(.keyDown(keyCode: 12, flags: 0)) // an old tap's late event
        XCTAssertEqual(results.all.count, 3)
    }

    /// Without Accessibility, or when no tap can be made, it answers
    /// permission_denied at once.
    func testDenied() {
        let taps = Taps(), results = Results()
        session(trusted: false, taps).begin(timeoutSeconds: 15) { results.add($0) }
        taps.fail = true
        let s = session(taps)
        s.begin(timeoutSeconds: 15) { results.add($0) }
        XCTAssertEqual(results.code(0), Code.permissionDenied)
        XCTAssertEqual(results.code(1), Code.permissionDenied)
        XCTAssertFalse(s.isRecording)
    }

    func testTimeout() {
        let taps = Taps(), results = Results(), s = session(taps)
        let ended = expectation(description: "timed out")
        s.begin(timeoutSeconds: 0.05) { results.add($0); ended.fulfill() }
        wait(for: [ended], timeout: 5)
        XCTAssertEqual(results.code(0), Code.timeout)
        XCTAssertEqual(taps.last?.stops, 1)
    }

    /// Events from the tap's thread while the app cancels and records
    /// again: every recording ends exactly once, nothing hangs.
    func testConcurrent() {
        let taps = Taps(), results = Results(), s = session(taps)
        let n = 200
        DispatchQueue.concurrentPerform(iterations: n) { i in
            switch i % 3 {
            case 0: s.begin(timeoutSeconds: 15) { results.add($0) }
            case 1: s.cancel()
            default: _ = taps.last?.handler(.keyDown(keyCode: 12, flags: leftOption))
            }
        }
        s.cancel()
        let begun = (0..<n).filter { $0 % 3 == 0 }.count
        XCTAssertEqual(results.all.count, begun)
        XCTAssertTrue(taps.all.allSatisfy { $0.stops == 1 })
    }
}
