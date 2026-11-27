@testable import VoxtNewHelperKit
import XCTest

/// The engine kept ready after a recording serves the next recording on
/// the same input, bound the same way; a recording that ended because its
/// input failed keeps nothing.
final class WarmInputTests: XCTestCase {
    let parked = ParkedInput(device: 73, bound: false)

    func testSameInputIsServed() {
        XCTAssertTrue(parked.serves(device: 73, bound: false))
    }

    func testOtherInputOrBindingIsNotServed() {
        XCTAssertFalse(parked.serves(device: 41, bound: false), "another input")
        XCTAssertFalse(parked.serves(device: 73, bound: true), "bound now: the default input changed")
    }

    func testOnlyAStopKeepsTheEngine() {
        XCTAssertTrue(parksInput(after: CaptureReason.stopped))
        XCTAssertTrue(parksInput(after: CaptureReason.maxSeconds))
        XCTAssertFalse(parksInput(after: CaptureReason.noCallback))
        XCTAssertFalse(parksInput(after: CaptureReason.deviceLost))
    }
}
