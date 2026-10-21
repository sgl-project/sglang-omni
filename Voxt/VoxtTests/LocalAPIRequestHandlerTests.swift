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
        XCTAssertEqual(body["endpoints"] as? [String], [
            "health",
            "capabilities",
            "dictations",
            "history/sync",
            "compose",
            "compose/prompts",
            "corrections",
        ])
        let audio = try XCTUnwrap(body["audio"] as? [String: Any])
        XCTAssertEqual(audio["maxBodyBytes"] as? Int, 4_096)
        XCTAssertEqual(audio["contentTypes"] as? [String], LocalAPIRouter.audioExtensions.keys.sorted())
        let models = try XCTUnwrap(body["models"] as? [String: String])
        XCTAssertEqual(models, ["transcription": "ready", "text": "off"])
        XCTAssertEqual(body["locations"] as? [String: String], ["transcription": "unknown", "text": "unknown"],
                       "a handler that is not told where the models run says unknown")
    }

    func testBuiltInPromptsRequireTheTokenAndListEveryMode() async throws {
        let handler = LocalAPIRequestHandler.make(
            router: LocalAPIRouter(token: "t", maxBodyBytes: 10), version: "1",
            modelStatus: Self.readyModels,
            dictate: { _ in .error(status: 500, "unused") },
            compose: { _ in .error(status: 500, "unused") })
        let forbidden = await handler(request("GET", "/v1/compose/prompts", token: "wrong"))
        XCTAssertEqual(forbidden.status, 401)
        let wrongMethod = await handler(request("POST", "/v1/compose/prompts"))
        XCTAssertEqual(wrongMethod.status, 405)
        let response = await handler(request("GET", "/v1/compose/prompts"))
        XCTAssertEqual(response.status, 200)
        let body = try JSONDecoder().decode([String: [String: String]].self, from: response.body)
        XCTAssertEqual(body["prompts"], LocalAPIComposePrompt.builtInPrompts())
    }

    func testComposeIsForwarded() async throws {
        let handler = LocalAPIRequestHandler.make(
            router: LocalAPIRouter(token: "t", maxBodyBytes: 1_000),
            version: "1",
            modelStatus: Self.readyModels,
            dictate: { @MainActor _ in .json(status: 200, ["text": "x"]) },
            compose: { @MainActor request in .json(status: 200, ["mode": request.options.mode.rawValue]) }
        )
        var compose = request("POST", "/v1/compose")
        compose.headers["content-type"] = "application/json"
        compose.body = try JSONSerialization.data(withJSONObject: [
            "audio": Data([1]).base64EncodedString(),
            "contentType": "audio/wav",
            "options": ["mode": "write", "style": "concise", "vocabulary": [], "selectedText": "", "contextBefore": "", "contextAfter": "", "targetLanguage": NSNull()]
        ])
        let response = await handler(compose)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: response.body) as? [String: String])
        XCTAssertEqual(body["mode"], "write")
    }

    func testLocationsAreConservativeAndMatchTheSelectedProviders() {
        func resolve(_ engine: String, _ mode: String, onDevice: Bool = false) -> LocalAPIModelLocations {
            .resolve(transcriptionEngine: engine, enhancementMode: mode, appleSpeechOnDevice: onDevice)
        }
        XCTAssertEqual(resolve("mlxAudio", "customLLM"), .init(transcription: .local, text: .local))
        XCTAssertEqual(resolve("remote", "remoteLLM"), .init(transcription: .remote, text: .remote))
        XCTAssertEqual(resolve("dictation", "appleIntelligence").transcription, .unknown)
        XCTAssertEqual(resolve("dictation", "appleIntelligence", onDevice: true).transcription, .local)
        XCTAssertEqual(resolve("mlxAudio", "appleIntelligence").text, .local)
        XCTAssertEqual(resolve("mlxAudio", "off").text, .local, "no text model sends nothing anywhere")
        XCTAssertEqual(resolve("future", "future"), .init(transcription: .unknown, text: .unknown))
    }

    func testCapabilitiesSayWhereTheSelectedModelsRun() async throws {
        let handler = LocalAPIRequestHandler.make(
            router: LocalAPIRouter(token: "t", maxBodyBytes: 10), version: "1",
            modelStatus: Self.readyModels,
            modelLocations: { .init(transcription: .local, text: .remote) },
            dictate: { _ in .error(status: 500, "unused") })
        let response = await handler(request("GET", "/v1/capabilities"))
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: response.body) as? [String: Any])
        XCTAssertEqual(body["locations"] as? [String: String], ["transcription": "local", "text": "remote"])
    }

    func testLocalOnlyRequestNeverReachesACloudModel() async throws {
        func handler(_ locations: LocalAPIModelLocations) -> LocalAPIServer.Handler {
            LocalAPIRequestHandler.make(
                router: LocalAPIRouter(token: "t", maxBodyBytes: 1_000), version: "1",
                modelStatus: Self.readyModels,
                modelLocations: { locations },
                dictate: { _ in .json(status: 200, ["text": "local"]) },
                compose: { _ in .json(status: 200, ["text": "local"]) })
        }
        var dictation = request("POST", "/v1/dictations")
        dictation.headers["content-type"] = "audio/wav"
        dictation.headers["x-voxt-model-policy"] = "local-only"
        dictation.body = Data([1])
        var compose = request("POST", "/v1/compose")
        compose.headers["content-type"] = "application/json"
        compose.headers["x-voxt-model-policy"] = "local-only"
        compose.body = try JSONSerialization.data(withJSONObject: [
            "audio": Data([1]).base64EncodedString(), "contentType": "audio/wav", "options": ["mode": "dictate"]])
        for audio in [dictation, compose] {
            let cloud = await handler(.init(transcription: .remote, text: .local))(audio)
            XCTAssertEqual(cloud.status, 403)
            XCTAssertEqual(try JSONSerialization.jsonObject(with: cloud.body) as? [String: String], ["error": "Your Mac is set to a cloud model."])
            let unknown = await handler(.init(transcription: .local, text: .unknown))(audio)
            XCTAssertEqual(unknown.status, 403)
            let local = await handler(.init(transcription: .local, text: .local))(audio)
            XCTAssertEqual(local.status, 200)
            var allowed = audio
            allowed.headers["x-voxt-model-policy"] = "cloud-allowed"
            let cloudAllowed = await handler(.init(transcription: .remote, text: .remote))(allowed)
            XCTAssertEqual(cloudAllowed.status, 200)
        }
    }
}
