import XCTest
@testable import VoxtNewHelperKit

final class HotkeyRecordTests: XCTestCase {
    func testRightOptionSpace() {
        let mods: Set<ModKey> = [.rightOption]
        XCTAssertEqual(WireHotkeyRecord.chordString(mods: mods, keyCode: 49), "RightOption+Space")
    }

    func testRightCommandOnly() {
        XCTAssertEqual(WireHotkeyRecord.chordString(mods: [.rightCommand]), "RightCommand")
    }

    func testFnF5() {
        let mods: Set<ModKey> = [.fn]
        XCTAssertEqual(WireHotkeyRecord.chordString(mods: mods, keyCode: 96), "Fn+F5")
    }

    func testMouse4() {
        XCTAssertEqual(WireHotkeyRecord.chordString(mods: [], mouseButton: 3), "Mouse4")
    }
}
