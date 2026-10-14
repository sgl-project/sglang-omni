// LocalAPIRouterTests.swift
// Provides Local API routing and authorization tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class LocalAPIRouterTests: XCTestCase {
    private let router = LocalAPIRouter(token: "s3cret-token", maxBodyBytes: 1_000)

    private func request(
        _ method: String,
        _ path: String,
        query: [String: String] = [:],
        token: String? = "s3cret-token",
        contentType: String? = nil,
        body: Data = Data()
    ) -> LocalAPIHTTPRequest {
        var headers: [String: String] = [:]
        if let token { headers["authorization"] = "Bearer \(token)" }
        if let contentType { headers["content-type"] = contentType }
        return LocalAPIHTTPRequest(method: method, path: path, query: query, headers: headers, body: body)
    }

    private func status(_ route: LocalAPIRoute) -> Int? {
        if case let .reject(response) = route { return response.status }
        return nil
    }

    func testHealthNeedsValidToken() {
        XCTAssertEqual(router.route(request("GET", "/v1/health")), .health)
        XCTAssertEqual(status(router.route(request("GET", "/v1/health", token: nil))), 401)
        XCTAssertEqual(status(router.route(request("GET", "/v1/health", token: "s3cret-tokeN"))), 401)
        XCTAssertEqual(status(router.route(request("GET", "/v1/health", token: "s3cret"))), 401)
    }

    func testEmptyConfiguredTokenRejectsEverything() {
        let open = LocalAPIRouter(token: "", maxBodyBytes: 1_000)
        XCTAssertEqual(status(open.route(request("GET", "/v1/health", token: ""))), 401)
    }

    func testDictationRoutesAudioAndEnhanceFlag() {
        let audio = Data([1, 2, 3])
        XCTAssertEqual(
            router.route(request("POST", "/v1/dictations", contentType: "audio/wav", body: audio)),
            .dictation(LocalAPIDictationRequest(audio: audio, fileExtension: "wav", enhance: true))
        )
        XCTAssertEqual(
            router.route(request("POST", "/v1/dictations", query: ["enhance": "0"], contentType: "audio/x-m4a; codecs=aac", body: audio)),
            .dictation(LocalAPIDictationRequest(audio: audio, fileExtension: "m4a", enhance: false))
        )
    }

    func testDictationRejectsBadInput() {
        XCTAssertEqual(status(router.route(request("POST", "/v1/dictations", contentType: "audio/wav"))), 400)
        XCTAssertEqual(status(router.route(request("POST", "/v1/dictations", contentType: "text/plain", body: Data([1])))), 415)
        XCTAssertEqual(status(router.route(request("POST", "/v1/dictations", body: Data([1])))), 415)
        XCTAssertEqual(status(router.route(request("GET", "/v1/dictations"))), 405)
        XCTAssertEqual(status(router.route(request("GET", "/nope"))), 404)
    }

    func testUnauthorizedBeatsEverythingElse() {
        XCTAssertEqual(status(router.route(request("GET", "/nope", token: nil))), 401)
    }

    func testGeneratedTokensAreLongAndUnique() {
        let a = LocalAPIRouter.generateToken()
        let b = LocalAPIRouter.generateToken()
        XCTAssertGreaterThanOrEqual(a.count, 32)
        XCTAssertNotEqual(a, b)
        XCTAssertNil(a.rangeOfCharacter(from: CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_")).inverted))
    }

    func testCapabilitiesNeedTokenAndGET() {
        XCTAssertEqual(router.route(request("GET", "/v1/capabilities")), .capabilities)
        XCTAssertEqual(status(router.route(request("GET", "/v1/capabilities", token: nil))), 401)
        XCTAssertEqual(status(router.route(request("POST", "/v1/capabilities"))), 405)
    }
}
