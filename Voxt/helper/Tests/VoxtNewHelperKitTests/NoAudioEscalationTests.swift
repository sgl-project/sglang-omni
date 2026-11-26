@testable import VoxtNewHelperKit
import XCTest

/// The limit on engines that deliver no audio: a chosen input that never
/// gives a buffer (silent, or reconfiguring itself forever, as in #4120)
/// moves the recording to the default input, then ends it, instead of
/// restarting without end; an engine that delivers audio proves the
/// input works and the count starts over.
final class NoAudioEscalationTests: XCTestCase {
    func testSecondSilentEngineGoesToTheDefaultInputThirdEnds() {
        var e = NoAudioEscalation()
        XCTAssertEqual(e.silentEngine(), .order)
        XCTAssertEqual(e.silentEngines, 1)
        XCTAssertEqual(e.silentEngine(), .default)
        XCTAssertEqual(e.silentEngine(), .end)
        XCTAssertEqual(e.silentEngine(), .end, "it keeps ending: no fourth engine starts")
    }

    func testAnEngineWithAudioResetsTheCount() {
        var e = NoAudioEscalation()
        XCTAssertEqual(e.silentEngine(), .order)
        e.engineHeard()
        XCTAssertEqual(e.silentEngines, 0)
        XCTAssertEqual(e.silentEngine(), .order)
        e.engineHeard()
        e.engineHeard()
        XCTAssertEqual(e.silentEngine(), .order, "a working input may reconfigure as often as it needs")
    }
}
