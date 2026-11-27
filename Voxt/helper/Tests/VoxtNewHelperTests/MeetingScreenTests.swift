import CoreGraphics
import CoreText
import XCTest
@testable import VoxtNewHelper

final class MeetingScreenTests: XCTestCase {
    func testZoomVideoTileRoleUsesItsCustomDescription() {
        XCTAssertEqual(MeetingWindow.videoRole("AXTabGroup", description: "视频渲染"), "AXVideoRender")
        XCTAssertEqual(MeetingWindow.videoRole("AXTabGroup", description: "video render"), "AXVideoRender")
        XCTAssertEqual(MeetingWindow.videoRole("AXTabGroup", description: "tab group"), "AXTabGroup")
        XCTAssertEqual(MeetingWindow.videoRole("AXGroup", description: "video render"), "AXGroup")
    }

    func testLocalScreenReadMatchesNameToSpeakingOutline() throws {
        let width = 640, height = 360
        let context = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                            bytesPerRow: width*4, space: CGColorSpaceCreateDeviceRGB(),
                                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(gray: 0.08, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setStrokeColor(CGColor(red: 0.1, green: 0.85, blue: 0.2, alpha: 1))
        context.setLineWidth(3)
        context.stroke(CGRect(x: 2, y: 2, width: width-4, height: height-4))
        let text = NSAttributedString(string: "Alice Chen", attributes: [
            NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Helvetica" as CFString, 24, nil),
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 1, alpha: 1),
        ])
        context.textPosition = CGPoint(x: 12, y: 12)
        CTLineDraw(CTLineCreateWithAttributedString(text), context)
        let image = try XCTUnwrap(context.makeImage())
        let frame = CGRect(x: 0, y: 0, width: width, height: height)
        let tiles = MeetingScreen.recognize(image, window: frame, frames: [MeetingTileFrame(rect: frame)])
        XCTAssertEqual(tiles.count, 1)
        XCTAssertEqual(tiles.first?.name, "Alice Chen")
        XCTAssertEqual(tiles.first?.speaking, true)
    }
}
