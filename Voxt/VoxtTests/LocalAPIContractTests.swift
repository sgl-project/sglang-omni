// LocalAPIContractTests.swift
// Checks the Local API's JSON against the fixtures in Voxt/Mobile/contract, which the Voxt Mobile
// tests decode too, so a schema change that forgets one side fails a test on that side.
// Run with VOXT_CONTRACT_RECORD=1 to rewrite the fixtures from the current output.

import XCTest
@testable import Voxt

@MainActor
final class LocalAPIContractTests: XCTestCase {
    private static let contract = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../Mobile/contract", isDirectory: true).standardizedFileURL
    private static let recording = ProcessInfo.processInfo.environment["VOXT_CONTRACT_RECORD"] == "1"
    /// Values that differ on every run, fixed in the fixtures.
    private static let stable: [String: Any] = ["id": "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE", "asrMs": 531, "enhanceMs": 412]

    private let token = "contract-token"
    private var directory: URL!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("LocalAPIContractTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func pipeline(transcript: String = "um tomorrow at three we meet in room four",
                          composed: @escaping (String, LocalAPIComposeOptions) async throws -> String = { _, _ in
                              "Tomorrow at three, we meet in room four."
                          }) -> LocalAPIDictationPipeline {
        let pipeline = LocalAPIDictationPipeline(
            temporaryDirectory: directory, busyReason: { nil }, isReadableAudio: { _ in true },
            transcribe: { _, _ in transcript }, enhance: { _ in "Tomorrow at three, we meet in room four." },
            finalize: { LocalAPIFinalText(text: $0) }, record: { _ in })
        pipeline.composeHandler = composed
        pipeline.correctHandler = { _, _ in #"{"correction":{"original":"Nova","replacement":"Novo"}}"# }
        return pipeline
    }

    private func handler(locations: LocalAPIModelLocations = .init(transcription: .local, text: .local),
                         pipeline: LocalAPIDictationPipeline? = nil) -> LocalAPIServer.Handler {
        let pipeline = pipeline ?? self.pipeline()
        return LocalAPIRequestHandler.make(
            router: LocalAPIRouter(token: token, maxBodyBytes: LocalAPISettings.maxBodyBytes),
            version: "1.16.0",
            modelStatus: { LocalAPIModelStatus(transcription: .ready, text: .ready) },
            modelLocations: { locations },
            dictate: { await pipeline.run($0) },
            compose: { await pipeline.runCompose($0) },
            correct: { await pipeline.runCorrection($0) })
    }

    private func request(_ method: String, _ path: String, contentType: String? = nil, body: Data = Data(),
                         token: String? = nil, headers: [String: String] = [:]) -> LocalAPIHTTPRequest {
        var all = headers
        all["authorization"] = "Bearer \(token ?? self.token)"
        if let contentType { all["content-type"] = contentType }
        return LocalAPIHTTPRequest(method: method, path: path, headers: all, body: body)
    }

    private func fixture(_ name: String) throws -> Any {
        try JSONSerialization.jsonObject(with: Data(contentsOf: Self.contract.appendingPathComponent(name)))
    }

    private func stabilized(_ object: Any) -> Any {
        guard var fields = object as? [String: Any] else { return object }
        for (key, value) in Self.stable where fields[key] != nil { fields[key] = value }
        return fields
    }

    /// The response body equals the fixture, apart from the values in `stable`.
    private func check(_ response: LocalAPIHTTPResponse, against name: String, file: StaticString = #filePath, line: UInt = #line) throws {
        let actual = stabilized(try JSONSerialization.jsonObject(with: response.body))
        try check(actual, against: name, file: file, line: line)
    }

    private func check(_ actual: Any, against name: String, file: StaticString = #filePath, line: UInt = #line) throws {
        if Self.recording {
            try FileManager.default.createDirectory(at: Self.contract, withIntermediateDirectories: true)
            var data = try JSONSerialization.data(withJSONObject: actual, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
            data.append(0x0A)
            try data.write(to: Self.contract.appendingPathComponent(name))
        }
        XCTAssertEqual(actual as? NSDictionary, try fixture(name) as? NSDictionary, name, file: file, line: line)
    }

    private func composeBody() throws -> Data {
        try Data(contentsOf: Self.contract.appendingPathComponent("compose-request.json"))
    }

    func testHealth() async throws {
        try check(await handler()(request("GET", "/v1/health")), against: "health.json")
    }

    func testCapabilities() async throws {
        try check(await handler()(request("GET", "/v1/capabilities")), against: "capabilities.json")
    }

    func testDictationResponse() async throws {
        let response = await handler()(request("POST", "/v1/dictations", contentType: "audio/wav", body: Data([1, 2, 3])))
        XCTAssertEqual(response.status, 200)
        try check(response, against: "dictation-response.json")
    }

    func testComposeRequestIsAcceptedAsWritten() throws {
        let route = LocalAPIRouter(token: token, maxBodyBytes: LocalAPISettings.maxBodyBytes)
            .route(request("POST", "/v1/compose", contentType: "application/json", body: try composeBody()))
        guard case let .compose(compose) = route else { return XCTFail("compose-request.json was refused: \(route)") }
        XCTAssertEqual(compose.options.mode, .translate)
        XCTAssertEqual(compose.options.targetLanguage, "zh-Hans")
        XCTAssertEqual(compose.options.vocabulary, ["Voxt"])
        XCTAssertEqual(compose.language, .request(nil))
        XCTAssertEqual(compose.options.styleExamples, [])
        let options = try XCTUnwrap((try JSONSerialization.jsonObject(with: try composeBody()) as? [String: Any])?["options"] as? [String: Any])
        XCTAssertEqual((options["styleExamples"] as? [Any])?.count, 0, "the phone always sends styleExamples")
    }

    /// The README next to the fixtures promises this: option keys Voxt does not read are ignored, so a
    /// client may send its own (Voxt Mobile sends literalDictation, which only the phone uses).
    func testComposeIgnoresOptionKeysVoxtDoesNotRead() throws {
        var body = try JSONSerialization.jsonObject(with: try composeBody()) as! [String: Any]
        var options = body["options"] as! [String: Any]
        options["literalDictation"] = false
        options["somethingNew"] = ["nested": 1]
        body["options"] = options
        let route = LocalAPIRouter(token: token, maxBodyBytes: LocalAPISettings.maxBodyBytes)
            .route(request("POST", "/v1/compose", contentType: "application/json", body: try JSONSerialization.data(withJSONObject: body)))
        guard case .compose = route else { return XCTFail("unknown option keys were refused: \(route)") }
        let readme = try String(contentsOf: Self.contract.appendingPathComponent("README.md"), encoding: .utf8)
        XCTAssertTrue(readme.contains("ignores option keys it does not read"))
    }

    func testComposeResponses() async throws {
        let body = try composeBody()
        let applied = await handler()(request("POST", "/v1/compose", contentType: "application/json", body: body))
        XCTAssertEqual(applied.status, 200)
        try check(applied, against: "compose-response.json")

        var edit = try JSONSerialization.jsonObject(with: body) as! [String: Any]
        edit["options"] = ["mode": "edit", "selectedText": "See you at the cafe."]
        let notApplied = await handler(pipeline: pipeline { _, _ in throw LocalAPIComposePrompt.EditNotApplied() })(
            request("POST", "/v1/compose", contentType: "application/json", body: try JSONSerialization.data(withJSONObject: edit)))
        XCTAssertEqual(notApplied.status, 200)
        try check(notApplied, against: "compose-not-applied.json")
    }

    func testErrors() async throws {
        let compose = try composeBody()
        let cases: [(String, LocalAPIServer.Handler, LocalAPIHTTPRequest)] = [
            ("badRequest", handler(), request("POST", "/v1/dictations", contentType: "audio/wav")),
            ("unauthorized", handler(), request("GET", "/v1/health", token: "wrong")),
            ("cloudModel", handler(locations: .init(transcription: .remote, text: .local)),
             request("POST", "/v1/compose", contentType: "application/json", body: compose, headers: ["x-voxt-model-policy": "local-only"])),
            ("notFound", handler(), request("GET", "/v1/nothing")),
            ("wrongMethod", handler(), request("GET", "/v1/dictations")),
            ("unsupportedAudio", handler(), request("POST", "/v1/dictations", contentType: "text/plain", body: Data([1]))),
            ("noSpeech", handler(pipeline: pipeline(transcript: "  ")), request("POST", "/v1/dictations", contentType: "audio/wav", body: Data([1]))),
        ]
        var errors: [String: Any] = [:]
        for (name, handler, request) in cases {
            let response = await handler(request)
            errors[name] = ["status": response.status, "body": try JSONSerialization.jsonObject(with: response.body)]
        }
        try check(errors, against: "errors.json")
    }

    func testCorrection() async throws {
        let body = try Data(contentsOf: Self.contract.appendingPathComponent("correction-request.json"))
        let response = await handler()(request("POST", "/v1/corrections", contentType: "application/json", body: body))
        XCTAssertEqual(response.status, 200)
        try check(response, against: "correction-response.json")
    }
}
