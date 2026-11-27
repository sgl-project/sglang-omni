import XCTest
@testable import VoxtNewHelperKit

final class SpeakerBorderTests: XCTestCase {
    func testGreenOutlineIsSpeakingButIconAndFillAreNot() {
        let width = 120, height = 80
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        func green(_ x: Int, _ y: Int) { let i = (y * width + x) * 4; pixels[i + 1] = 220; pixels[i + 3] = 255 }
        for y in 20..<35 { for x in 20..<35 { green(x, y) } }
        XCTAssertFalse(hasSpeakingBorder(rgba: pixels, width: width, height: height))
        for x in 0..<width { green(x, 0); green(x, height - 1) }
        for y in 0..<height { green(0, y); green(width - 1, y) }
        XCTAssertTrue(hasSpeakingBorder(rgba: pixels, width: width, height: height))
        for y in 0..<height { for x in 0..<width { green(x, y) } }
        XCTAssertFalse(hasSpeakingBorder(rgba: pixels, width: width, height: height))
    }
}
