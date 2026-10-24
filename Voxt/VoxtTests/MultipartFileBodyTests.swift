// MultipartFileBodyTests.swift

import XCTest
@testable import Voxt

final class MultipartFileBodyTests: XCTestCase {
    func testCreatesMultipartFileWithoutChangingSourcePayload() throws {
        let sourceURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("recording-\(UUID().uuidString).wav")
        let source = Data((0..<2_500_000).map { UInt8($0 % 251) })
        try source.write(to: sourceURL)
        defer { try? FileManager.default.removeItem(at: sourceURL) }

        let body = try MultipartFileBody.create(
            sourceFileURL: sourceURL,
            boundary: "Boundary-Test",
            fields: [("model", "whisper-1"), ("prompt", "Voxt")],
            mimeType: "audio/wav"
        )
        defer { body.remove() }

        XCTAssertTrue(FileManager.default.fileExists(atPath: body.url.path))
        XCTAssertGreaterThan(body.byteCount, Int64(source.count))

        let encoded = try Data(contentsOf: body.url)
        XCTAssertTrue(encoded.contains(source))
        XCTAssertTrue(String(decoding: encoded.prefix(256), as: UTF8.self).contains("name=\"model\""))
        XCTAssertTrue(String(decoding: encoded.suffix(64), as: UTF8.self).contains("--Boundary-Test--"))
    }

    func testSanitizesMultipartHeaderValues() throws {
        let sourceURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("unsafe\nname.wav")
        try Data([1, 2, 3]).write(to: sourceURL)
        defer { try? FileManager.default.removeItem(at: sourceURL) }

        let body = try MultipartFileBody.create(
            sourceFileURL: sourceURL,
            boundary: "Boundary-Test",
            fields: [("model\r\nInjected", "value")],
            mimeType: "audio/wav"
        )
        defer { body.remove() }

        let encoded = try String(contentsOf: body.url, encoding: .isoLatin1)
        XCTAssertFalse(encoded.contains("model\r\nInjected"))
        XCTAssertFalse(encoded.contains("unsafe\nname.wav"))
    }
}
