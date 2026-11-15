import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest

@testable import VoxtNewHelperKit

final class ClipboardPictureTests: XCTestCase {
    func png(_ w: Int, _ h: Int) -> Data {
        let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        let data = NSMutableData()
        let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return data as Data
    }

    let png = UTType.png.identifier, text = UTType.utf8PlainText.identifier

    func testCopiedImage() {
        let img = ClipboardPicture.image(types: [png], data: [png: png(40, 20)], file: nil, maxSide: 1600)
        XCTAssertEqual(img.map { [$0.width, $0.height] }, [40, 20], "a small picture keeps its size")
        let big = ClipboardPicture.image(types: [png], data: [png: png(3200, 100)], file: nil, maxSide: 1600)
        XCTAssertEqual(big.map { [$0.width, $0.height] }, [1600, 50], "a big one shrinks to maxSide")
    }

    func testNoPicture() {
        XCTAssertNil(ClipboardPicture.image(types: [], data: [:], file: nil, maxSide: 1600), "empty clipboard")
        XCTAssertNil(ClipboardPicture.image(types: [text], data: [:], file: nil, maxSide: 1600), "text")
        XCTAssertNil(ClipboardPicture.image(types: [text, png], data: [png: png(10, 10)], file: nil, maxSide: 1600),
                     "text with a picture of itself (a spreadsheet range)")
        XCTAssertNil(ClipboardPicture.image(types: [png], data: [png: Data("not a png".utf8)], file: nil, maxSide: 1600), "broken data")
    }

    func testCopiedFile() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let pic = dir.appendingPathComponent("photo.png"), doc = dir.appendingPathComponent("notes.txt")
        try png(30, 60).write(to: pic)
        try Data("hello".utf8).write(to: doc)
        // Finder puts the file, its name as text and its icon.
        let finder = [UTType.fileURL.identifier, text, UTType.tiff.identifier]
        let img = ClipboardPicture.image(types: finder, data: [:], file: pic, maxSide: 1600)
        XCTAssertEqual(img.map { [$0.width, $0.height] }, [30, 60], "an image file")
        XCTAssertNil(ClipboardPicture.image(types: finder, data: [UTType.tiff.identifier: png(10, 10)], file: doc, maxSide: 1600),
                     "a text file: not its icon")
    }
}
