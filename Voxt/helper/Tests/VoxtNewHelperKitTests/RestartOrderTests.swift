@testable import VoxtNewHelperKit
import XCTest

/// Which inputs a restart tries: an input going away mid-recording moves
/// it to the next of the user's devices still connected, then the default
/// input (Voxt 1.x's RecordingCaptureFlow), instead of ending it.
final class RestartOrderTests: XCTestCase {
    func testChosenInputWentAway() {
        XCTAssertEqual(restartOrder(devices: ["USBMic:1", "AirPodsPro:input"], current: "USBMic:1", gone: true),
                       ["AirPodsPro:input", nil])
        // An input ranked first that could not start earlier is tried again: it may be back.
        XCTAssertEqual(restartOrder(devices: ["Studio", "USBMic:1", "AirPodsPro:input"], current: "USBMic:1", gone: true),
                       ["Studio", "AirPodsPro:input", nil])
        XCTAssertEqual(restartOrder(devices: ["USBMic:1"], current: "USBMic:1", gone: true), [nil])
    }

    func testChosenInputStillThere() {
        XCTAssertEqual(restartOrder(devices: ["USBMic:1", "", "AirPodsPro:input", "USBMic:1"], current: "USBMic:1", gone: false),
                       ["USBMic:1", "AirPodsPro:input", nil])
        XCTAssertEqual(restartOrder(devices: ["USBMic:1", "AirPodsPro:input"], current: "USBMic:1", gone: false, toDefault: true), [nil])
    }

    func testDefaultInputFollowsTheNewDefault() {
        XCTAssertEqual(restartOrder(devices: ["AirPodsPro:input"], current: nil, gone: false), [nil])
    }
}
