// LocalAPIHTTPTests.swift
// Provides Local API HTTP parsing tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class LocalAPIHTTPTests: XCTestCase {
    private func raw(_ head: String, body: Data = Data()) -> Data {
        Data(head.replacingOccurrences(of: "\n", with: "\r\n").utf8) + body
    }

    func testParsesCompleteRequestWithBodyQueryAndHeaders() throws {
        let body = Data([0x52, 0x49, 0x46, 0x46])
        let data = raw("""
        POST /v1/dictations?enhance=0 HTTP/1.1
        Host: mac.tail1.ts.net:8900
        Authorization: Bearer abc
        Content-Type: audio/wav
        Content-Length: 4


        """, body: body)

        guard case let .complete(request, consumed) = LocalAPIHTTPParser.parse(data, maxBodyBytes: 1024) else {
            return XCTFail("Expected a complete request")
        }
        XCTAssertEqual(consumed, data.count)
        XCTAssertEqual(request.method, "POST")
        XCTAssertEqual(request.path, "/v1/dictations")
        XCTAssertEqual(request.query["enhance"], "0")
        XCTAssertEqual(request.header("authorization"), "Bearer abc")
        XCTAssertEqual(request.header("Content-Type"), "audio/wav")
        XCTAssertEqual(request.body, body)
    }

    func testWaitsForMissingHeaderTerminatorAndBodyBytes() {
        XCTAssertEqual(
            LocalAPIHTTPParser.parse(raw("GET /v1/health HTTP/1.1\nHost: x\n"), maxBodyBytes: 10),
            .incomplete
        )
        let partialBody = raw("POST /v1/dictations HTTP/1.1\nContent-Length: 4\n\n", body: Data([1, 2]))
        XCTAssertEqual(LocalAPIHTTPParser.parse(partialBody, maxBodyBytes: 10), .incomplete)
    }

    func testRequestWithoutBodyIsCompleteAtHeaderEnd() throws {
        guard case let .complete(request, _) = LocalAPIHTTPParser.parse(
            raw("GET /v1/health HTTP/1.1\nHost: x\n\n"),
            maxBodyBytes: 10
        ) else {
            return XCTFail("Expected a complete request")
        }
        XCTAssertEqual(request.method, "GET")
        XCTAssertTrue(request.body.isEmpty)
    }

    func testRejectsOversizedBodyBeforeReadingIt() {
        let data = raw("POST /v1/dictations HTTP/1.1\nContent-Length: 11\n\n")
        XCTAssertEqual(LocalAPIHTTPParser.parse(data, maxBodyBytes: 10), .failure(413))
    }

    func testRejectsChunkedBodiesAndMalformedRequestLines() {
        XCTAssertEqual(
            LocalAPIHTTPParser.parse(raw("POST /v1/dictations HTTP/1.1\nTransfer-Encoding: chunked\n\n"), maxBodyBytes: 10),
            .failure(411)
        )
        XCTAssertEqual(LocalAPIHTTPParser.parse(raw("NONSENSE\n\n"), maxBodyBytes: 10), .failure(400))
        XCTAssertEqual(
            LocalAPIHTTPParser.parse(raw("POST /x HTTP/1.1\nContent-Length: -3\n\n"), maxBodyBytes: 10),
            .failure(400)
        )
    }

    func testRejectsHeadersThatNeverEnd() {
        let data = Data(("GET / HTTP/1.1\r\nX: " + String(repeating: "a", count: 20_000)).utf8)
        XCTAssertEqual(LocalAPIHTTPParser.parse(data, maxBodyBytes: 10), .failure(431))
    }

    func testSerializesJSONResponse() throws {
        let response = LocalAPIHTTPResponse.json(status: 401, ["error": "bad token"])
        let text = String(decoding: response.serialized(), as: UTF8.self)
        XCTAssertTrue(text.hasPrefix("HTTP/1.1 401 Unauthorized\r\n"))
        XCTAssertTrue(text.contains("Content-Type: application/json; charset=utf-8\r\n"))
        XCTAssertTrue(text.contains("Content-Length: 21\r\n"))
        XCTAssertTrue(text.contains("Connection: close\r\n"))
        XCTAssertTrue(text.hasSuffix("\r\n\r\n{\"error\":\"bad token\"}"))
    }
}
