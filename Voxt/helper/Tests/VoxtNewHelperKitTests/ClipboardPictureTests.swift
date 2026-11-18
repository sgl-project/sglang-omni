import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest

@testable import VoxtNewHelperKit

final class ClipboardPictureTests: XCTestCase {
    func imageData(_ w: Int, _ h: Int, type: UTType = .png) -> Data {
        let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        let data = NSMutableData()
        let dest = CGImageDestinationCreateWithData(data, type.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return data as Data
    }

    let png = UTType.png.identifier, text = UTType.utf8PlainText.identifier

    private func decode(types: [String], data: [String: Data], file: URL?, maxSide: Int) async -> CGImage? {
        await ClipboardPicture.image(types: types, file: file, maxSide: maxSide) { data[$0] }
    }

    func testCopiedImage() async {
        let img = await decode(types: [png], data: [png: imageData(40, 20)], file: nil, maxSide: 1600)
        XCTAssertEqual(img.map { [$0.width, $0.height] }, [40, 20], "a small picture keeps its size")
        let big = await decode(types: [png], data: [png: imageData(3200, 100)], file: nil, maxSide: 1600)
        XCTAssertEqual(big.map { [$0.width, $0.height] }, [1600, 50], "a big one shrinks to maxSide")
    }

    func testFallsBackFromUnreadablePNGToTIFF() async {
        let tiff = UTType.tiff.identifier
        let validTIFF = imageData(30, 60, type: .tiff)
        let types = [png, tiff]
        let validPNG = imageData(40, 20)
        let unreadablePNG = Data(validPNG.prefix(8)) // PNG signature with no pixels.
        let fallback = await decode(types: types, data: [png: unreadablePNG, tiff: validTIFF], file: nil, maxSide: 1600)
        XCTAssertEqual(fallback.map { [$0.width, $0.height] }, [30, 60])

        let preferred = await decode(types: types, data: [png: validPNG, tiff: validTIFF], file: nil, maxSide: 1600)
        XCTAssertEqual(preferred.map { [$0.width, $0.height] }, [40, 20])
    }

    func testReadsLaterFormatOnlyWhenNeeded() async {
        let tiff = UTType.tiff.identifier
        let validPNG = imageData(40, 20)
        let validTIFF = imageData(30, 60, type: .tiff)
        let types = [png, tiff]
        let preferred = await ClipboardPicture.image(types: types, file: nil, maxSide: 1600) { type in
            XCTAssertEqual(type, UTType.png.identifier)
            return validPNG
        }
        XCTAssertEqual(preferred?.width, 40)
        let fallback = await ClipboardPicture.image(types: types, file: nil, maxSide: 1600) { type in
            type == tiff ? validTIFF : nil
        }
        XCTAssertEqual(fallback?.width, 30)
    }

    func testCopiedWebPOnly() async {
        // A 1x1 lossless WebP: ImageIO decodes WebP but cannot write it.
        let webp = UTType.webP.identifier
        let bytes = Data(base64Encoded: "UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==")!
        let img = await decode(types: [webp], data: [webp: bytes], file: nil, maxSide: 1600)
        XCTAssertEqual(img.map { [$0.width, $0.height] }, [1, 1], "a clipboard whose only picture is WebP still sends it")
    }

    private func expectNoImage(types: [String], data: [String: Data], file: URL?, maxSide: Int, _ message: String,
                               line: UInt = #line) async {
        let image = await decode(types: types, data: data, file: file, maxSide: maxSide)
        XCTAssertNil(image, message, line: line)
    }

    func testNoPicture() async {
        await expectNoImage(types: [], data: [:], file: nil, maxSide: 1600, "empty clipboard")
        await expectNoImage(types: [text], data: [:], file: nil, maxSide: 1600, "text")
        await expectNoImage(types: [text, png], data: [png: imageData(10, 10)], file: nil, maxSide: 1600,
                     "text with a picture of itself (a spreadsheet range)")
        await expectNoImage(types: [png], data: [png: Data("not a png".utf8)], file: nil, maxSide: 1600, "broken data")
    }

    func testCopiedFile() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let pic = dir.appendingPathComponent("photo.png"), doc = dir.appendingPathComponent("notes.txt")
        try imageData(30, 60).write(to: pic)
        try Data("hello".utf8).write(to: doc)
        // Finder puts the file, its name as text and its icon.
        let finder = [UTType.fileURL.identifier, text, UTType.tiff.identifier]
        let img = await decode(types: finder, data: [:], file: pic, maxSide: 1600)
        XCTAssertEqual(img.map { [$0.width, $0.height] }, [30, 60], "an image file")
        await expectNoImage(types: finder, data: [UTType.tiff.identifier: imageData(10, 10)], file: doc, maxSide: 1600,
                     "a text file: not its icon")
    }
}
