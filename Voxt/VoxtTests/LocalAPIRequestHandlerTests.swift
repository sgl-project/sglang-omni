// LocalAPIRequestHandlerTests.swift
// Provides Local API request handler tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class LocalAPIRequestHandlerTests: XCTestCase {
    private static let readyModels: @MainActor @Sendable () -> LocalAPIModelStatus = {
        LocalAPIModelStatus(transcription: .ready, text: .off)
    }

    private func request(_ method: String, _ path: String, token: String = "t") -> LocalAPIHTTPRequest {
        LocalAPIHTTPRequest(method: method, path: path, headers: ["authorization": "Bearer \(token)"])
    }

    func testHealthAnswersWhileMainActorIsBusy() async throws {
        let handler = LocalAPIRequestHandler.make(
            router: LocalAPIRouter(token: "t", maxBodyBytes: 10),
            version: "9.9",
            modelStatus: Self.readyModels,
            dictate: { @MainActor _ in .json(status: 200, ["text": "x"]) }
        )
        let mainBusy = Task { @MainActor in Thread.sleep(forTimeInterval: 1.5) }
        try await Task.sleep(for: .milliseconds(100))

        let started = Date()
        let response = await Task.detached { await handler(self.request("GET", "/v1/health")) }.value
        let elapsed = Date().timeIntervalSince(started)
        await mainBusy.value

        XCTAssertEqual(response.status, 200)
        XCTAssertLessThan(elapsed, 0.5, "health waited for the main actor")
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: response.body) as? [String: Any])
        XCTAssertEqual(body["ok"] as? Bool, true)
        XCTAssertEqual(body["version"] as? String, "9.9")
    }

    func testRejectionsDoNotReachDictation() async {
        let dictated = expectation(description: "dictate")
        dictated.isInverted = true
        let handler = LocalAPIRequestHandler.make(
            router: LocalAPIRouter(token: "t", maxBodyBytes: 10),
            version: "1",
            modelStatus: Self.readyModels,
            dictate: { @MainActor _ in
                dictated.fulfill()
                return .json(status: 200, ["text": "x"])
            }
        )
        let response = await handler(request("POST", "/v1/dictations", token: "wrong"))
        XCTAssertEqual(response.status, 401)
        await fulfillment(of: [dictated], timeout: 0.2)
    }

    func testDictationIsForwarded() async throws {
        let handler = LocalAPIRequestHandler.make(
            router: LocalAPIRouter(token: "t", maxBodyBytes: 10),
            version: "1",
            modelStatus: Self.readyModels,
            dictate: { @MainActor request in .json(status: 200, ["ext": request.fileExtension]) }
        )
        var dictation = request("POST", "/v1/dictations")
        dictation.headers["content-type"] = "audio/wav"
        dictation.body = Data([1])
        let response = await handler(dictation)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: response.body) as? [String: String])
        XCTAssertEqual(body["ext"], "wav")
    }

    func testCapabilitiesDescribeTheAPIAndModelState() async throws {
        let handler = LocalAPIRequestHandler.make(
            router: LocalAPIRouter(token: "t", maxBodyBytes: 4_096),
            version: "2.0",
            modelStatus: Self.readyModels,
            dictate: { @MainActor _ in .json(status: 200, ["text": "x"]) }
        )

        let response = await handler(request("GET", "/v1/capabilities"))

        XCTAssertEqual(response.status, 200)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: response.body) as? [String: Any])
        let api = try XCTUnwrap(body["api"] as? [String: Int])
        XCTAssertEqual(api["major"], 1)
        XCTAssertEqual(api["minor"], 0)
        XCTAssertEqual(body["app"] as? String, "Voxt")
        XCTAssertEqual(body["version"] as? String, "2.0")
        XCTAssertEqual(body["endpoints"] as? [String], ["health", "capabilities", "dictations"])
        let audio = try XCTUnwrap(body["audio"] as? [String: Any])
        XCTAssertEqual(audio["maxBodyBytes"] as? Int, 4_096)
        XCTAssertEqual(audio["contentTypes"] as? [String], LocalAPIRouter.audioExtensions.keys.sorted())
        let models = try XCTUnwrap(body["models"] as? [String: String])
        XCTAssertEqual(models, ["transcription": "ready", "text": "off"])
    }
}
