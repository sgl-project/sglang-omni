@testable import VoxtNewHelperKit
import XCTest

/// A chosen input that is the default input already starts unbound, on
/// the default input's fast path: binding it explicitly left a USB input
/// silent until two watchdog rebuilds moved the recording to the default
/// input, about 3 s after every start.
final class ChosenInputBindingTests: XCTestCase {
    func testChosenDefaultInputIsNotBound() {
        XCTAssertFalse(bindsChosenInput(73, defaultInput: 73))
    }

    func testChosenOtherInputIsBound() {
        XCTAssertTrue(bindsChosenInput(73, defaultInput: 41))
        XCTAssertTrue(bindsChosenInput(73, defaultInput: 0), "no default input: bind the chosen one")
    }
}
