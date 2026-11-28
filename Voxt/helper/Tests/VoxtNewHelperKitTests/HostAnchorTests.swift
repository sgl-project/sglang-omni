import VoxtNewHelperKit
import XCTest

final class HostAnchorTests: XCTestCase {
    func testHostAnchorCarriesStampsAndFindsBreaks() {
        var host = HostAnchor(sampleRate: 16000)
        XCTAssertNil(host.ms(at: 0))
        XCTAssertFalse(host.breaks(1000, at: 0))
        host.set(1000, at: 0)
        XCTAssertEqual(host.ms(at: 1600), 1100)
        XCTAssertFalse(host.breaks(1115, at: 1600)) // jitter
        XCTAssertTrue(host.breaks(1600, at: 1600)) // a restart's gap
        XCTAssertTrue(host.breaks(1050, at: 1600)) // an overlap
        // Padding up to a buffer stamped 1600 ms fits 500 ms, none past it.
        XCTAssertEqual(host.room(before: 1600, at: 1600), 8000)
        XCTAssertEqual(host.room(before: 1050, at: 1600), 0)
    }
}
