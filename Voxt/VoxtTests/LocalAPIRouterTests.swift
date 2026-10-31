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

    func testDictationLanguageComesFromTheQuery() {
        let audio = Data([1])
        func route(_ query: [String: String]) -> LocalAPIRoute {
            router.route(request("POST", "/v1/dictations", query: query, contentType: "audio/wav", body: audio))
        }
        func dictation(_ language: ASRLanguageSelection) -> LocalAPIRoute {
            .dictation(LocalAPIDictationRequest(audio: audio, fileExtension: "wav", enhance: true, language: language))
        }
        XCTAssertEqual(route([:]), dictation(.userSettings))
        XCTAssertEqual(route(["language": "auto"]), dictation(.request(nil)))
        XCTAssertEqual(route(["language": "Chinese"]), dictation(.request("Chinese")))
        XCTAssertEqual(route(["language": "Simplified Chinese"]), dictation(.request("Simplified Chinese")))
        for bad in ["", "  ", "en\"; x", String(repeating: "a", count: 41)] {
            XCTAssertEqual(status(route(["language": bad])), 400, bad)
        }
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

    // MARK: Compose

    private func composeBody(
        mode: String = "dictate",
        targetLanguage: String? = nil,
        selectedText: String? = nil,
        contextBefore: String? = nil,
        contextAfter: String? = nil,
        style: String? = nil,
        vocabulary: [String]? = nil,
        languageVariants: [String]? = nil,
        omitLanguageVariants: Bool = false,
        styleExamples: Any? = nil,
        sendStyleExamples: Bool = false,
        audio: String = Data([9, 8, 7]).base64EncodedString(),
        contentType: String = "audio/wav"
    ) throws -> Data {
        var options: [String: Any] = ["mode": mode]
        options["targetLanguage"] = targetLanguage
        options["selectedText"] = selectedText ?? ""
        options["contextBefore"] = contextBefore ?? ""
        options["contextAfter"] = contextAfter ?? ""
        options["style"] = style ?? "automatic"
        options["vocabulary"] = vocabulary ?? []
        if !omitLanguageVariants {
            options["languageVariants"] = languageVariants ?? []
        }
        if sendStyleExamples {
            options["styleExamples"] = styleExamples ?? NSNull()
        }
        let body: [String: Any] = ["audio": audio, "contentType": contentType, "options": options]
        return try JSONSerialization.data(withJSONObject: body)
    }

    private func composeRequest(_ body: Data, contentType: String = "application/json") -> LocalAPIHTTPRequest {
        request("POST", "/v1/compose", contentType: contentType, body: body)
    }

    func testComposeRoutesDecodedRequest() throws {
        let route = router.route(composeRequest(try composeBody(
            mode: "translate",
            targetLanguage: "zh-Hans",
            selectedText: "hello",
            style: "formal",
            vocabulary: ["Voxt"]
        )))

        guard case let .compose(compose) = route else {
            return XCTFail("expected .compose, got \(route)")
        }
        XCTAssertEqual(compose.audio, Data([9, 8, 7]))
        XCTAssertEqual(compose.fileExtension, "wav")
        XCTAssertEqual(compose.options.mode, .translate)
        XCTAssertEqual(compose.options.targetLanguage, "zh-Hans")
        XCTAssertEqual(compose.options.selectedText, "hello")
        XCTAssertEqual(compose.options.style, .formal)
        XCTAssertEqual(compose.options.vocabulary, ["Voxt"])
    }

    func testComposeRejectsUnauthorizedAndWrongMethod() throws {
        XCTAssertEqual(status(router.route(request("POST", "/v1/compose", token: nil))), 401)
        XCTAssertEqual(status(router.route(request("GET", "/v1/compose"))), 405)
    }

    func testComposeRejectsNonJSONAndBrokenBodies() throws {
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(), contentType: "audio/wav"))), 415)
        XCTAssertEqual(status(router.route(composeRequest(Data("not json".utf8)))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(audio: "%%%not-base64%%%")))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(audio: "")))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(contentType: "text/plain")))), 415)
    }

    func testComposeRejectsUnknownEnums() throws {
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(mode: "summarize")))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(style: "playful")))), 400)
    }

    func testComposeEnforcesModeRequirements() throws {
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(mode: "translate")))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(mode: "translate", targetLanguage: "klingon")))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(mode: "translate", targetLanguage: "zz")))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(mode: "translate", targetLanguage: String(repeating: "a", count: 65))))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(mode: "edit")))), 400)
        if case .reject = router.route(composeRequest(try composeBody(mode: "edit", selectedText: "keep this"))) {
            XCTFail("edit with a selection must pass routing")
        }
        if case .reject = router.route(composeRequest(try composeBody(mode: "translate", targetLanguage: "zh-Hans"))) {
            XCTFail("translate with a BCP-47 tag must pass routing")
        }
    }

    func testComposeEnforcesSizeLimits() throws {
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(mode: "edit", selectedText: String(repeating: "a", count: 8_001))))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(contextBefore: String(repeating: "a", count: 2_001))))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(contextAfter: String(repeating: "a", count: 2_001))))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(vocabulary: Array(repeating: "term", count: 201))))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(vocabulary: [String(repeating: "a", count: 101)])))), 400)
    }

    func testComposeAcceptsOneVariantPerLanguageBase() throws {
        if case .reject = router.route(composeRequest(try composeBody(
            languageVariants: ["en-US", "zh-Hans", "es-MX", "fr-CA", "pt-BR"]
        ))) {
            XCTFail("one variant per base must pass")
        }
    }

    func testComposeRejectsInvalidLanguageVariants() throws {
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(languageVariants: ["en-US", "klingon"])))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(languageVariants: ["banana"])))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(languageVariants: ["en-US", "en-GB"])))), 400)
        XCTAssertEqual(status(router.route(composeRequest(try composeBody(languageVariants: ["zh-Hans", "zh-Hant"])))), 400)
    }

    private func rejection(_ route: LocalAPIRoute) -> String? {
        guard case let .reject(response) = route,
              let object = try? JSONSerialization.jsonObject(with: response.body) as? [String: String] else { return nil }
        return object["error"]
    }

    private func stylePair(_ before: String, _ after: String, extra: String? = nil) -> [String: Any] {
        var pair: [String: Any] = ["before": before, "after": after]
        if let extra { pair["label"] = extra }
        return pair
    }

    func testComposeAbsentStyleExamplesDecodesEmpty() throws {
        guard case let .compose(compose) = router.route(composeRequest(try composeBody())) else {
            return XCTFail("a request without styleExamples must still route")
        }
        XCTAssertEqual(compose.options.styleExamples, [])
    }

    func testComposeAcceptsEightDifferentStyleExamples() throws {
        let examples = (1...8).map { stylePair("Send file \($0).", "Send the file \($0).") }
        guard case let .compose(compose) = router.route(composeRequest(try composeBody(styleExamples: examples, sendStyleExamples: true))) else {
            return XCTFail("eight examples must route")
        }
        XCTAssertEqual(compose.options.styleExamples.count, 8)
        XCTAssertEqual(compose.options.styleExamples[0].before, "Send file 1.")
        XCTAssertEqual(compose.options.styleExamples[0].after, "Send the file 1.")
    }

    func testComposeRejectsInvalidStyleExamples() throws {
        let ninth = (1...9).map { stylePair("Send file \($0).", "Send the file \($0).") }
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: ninth, sendStyleExamples: true)))), "Use at most 8 personal style examples.")
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: [stylePair("", "Cheers")], sendStyleExamples: true)))), "A personal style example must be 1 to 120 characters.")
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: [stylePair(String(repeating: "a", count: 121), "b")], sendStyleExamples: true)))), "A personal style example must be 1 to 120 characters.")
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: [stylePair(" \n\t ", "Cheers")], sendStyleExamples: true)))), "A personal style example must not be blank.")
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: [stylePair("Thanks", "Thanks")], sendStyleExamples: true)))), "The personal style example texts must differ.")
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: [stylePair("at 3", "at 4")], sendStyleExamples: true)))), "Personal style examples must keep the same digit sequences.")
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: [stylePair("$50", "fifty")], sendStyleExamples: true)))), "Personal style examples must keep the same digit sequences.")
        let duplicated = stylePair("Thanks, I'll send Alice the file at 3.", "Cheers — I'll send Alice the file at 3.")
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: [duplicated, duplicated], sendStyleExamples: true)))), "Personal style examples must not repeat.")
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: [stylePair("Thanks", "Cheers", extra: "habit")], sendStyleExamples: true)))), "The personal style request has invalid fields.")
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: "Cheers", sendStyleExamples: true)))), "The personal style request has invalid fields.")
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: NSNull(), sendStyleExamples: true)))), "The personal style request has invalid fields.")
        XCTAssertEqual(rejection(router.route(composeRequest(try composeBody(styleExamples: [["before": 1, "after": "Cheers"]], sendStyleExamples: true)))), "The personal style request has invalid fields.")
    }

    func testComposeAcceptsOmittedLanguageVariants() throws {
        if case .reject = router.route(composeRequest(try composeBody(omitLanguageVariants: true))) {
            XCTFail("requests without languageVariants must keep working")
        }
    }


    func testComposeLanguageFollowsTheDictationRule() throws {
        func language(_ value: Any?) throws -> LocalAPIRoute {
            var body = try JSONSerialization.jsonObject(with: try composeBody()) as! [String: Any]
            body["language"] = value
            return router.route(composeRequest(try JSONSerialization.data(withJSONObject: body)))
        }
        guard case let .compose(settings) = try language(nil),
              case let .compose(automatic) = try language("auto"),
              case let .compose(chinese) = try language("Chinese") else { return XCTFail("expected compose routes") }
        XCTAssertEqual(settings.language, .userSettings)
        XCTAssertEqual(automatic.language, .request(nil))
        XCTAssertEqual(chinese.language, .request("Chinese"))
        XCTAssertEqual(status(try language("en\"; x")), 400)
    }

    // MARK: Corrections

    private func correctionRequest(
        original: Any = "请用星浪处理。",
        edited: Any = "请用星朗处理。",
        method: String = "POST",
        contentType: String? = "application/json",
        raw: Data? = nil
    ) -> LocalAPIHTTPRequest {
        let body = raw ?? (try! JSONSerialization.data(withJSONObject: ["original": original, "edited": edited]))
        return request(method, "/v1/corrections", contentType: contentType, body: body)
    }

    func testCorrectionRoutesAValidPair() throws {
        guard case let .correction(correction) = router.route(correctionRequest()) else {
            return XCTFail("expected .correction")
        }
        XCTAssertEqual(correction.original, "请用星浪处理。")
        XCTAssertEqual(correction.edited, "请用星朗处理。")
    }

    func testCorrectionAcceptsExactly2000Characters() {
        let original = String(repeating: "字", count: 2_000)
        let edited = String(repeating: "文", count: 2_000)
        guard case .correction = router.route(correctionRequest(original: original, edited: edited)) else {
            return XCTFail("2000 characters must pass")
        }
    }

    func testCorrectionRejectsMethodContentTypeAndJSON() {
        XCTAssertEqual(status(router.route(correctionRequest(method: "GET"))), 405)
        XCTAssertEqual(status(router.route(request("POST", "/v1/corrections", token: nil, contentType: "application/json", body: Data("{}".utf8)))), 401)
        XCTAssertEqual(status(router.route(correctionRequest(contentType: "text/plain"))), 415)
        XCTAssertEqual(status(router.route(correctionRequest(contentType: nil))), 415)
        XCTAssertEqual(status(router.route(correctionRequest(raw: Data("not json".utf8)))), 400)
        XCTAssertEqual(status(router.route(correctionRequest(raw: Data("[]".utf8)))), 400)
    }

    func testCorrectionRejectsWrongTypesIdenticalAndOversize() {
        XCTAssertEqual(status(router.route(correctionRequest(original: 3, edited: "星朗"))), 400)
        XCTAssertEqual(status(router.route(correctionRequest(original: "星浪", edited: ["星朗"]))), 400)
        XCTAssertEqual(status(router.route(correctionRequest(original: "", edited: "星朗"))), 400)
        XCTAssertEqual(status(router.route(correctionRequest(original: "星浪", edited: "星浪"))), 400)
        XCTAssertEqual(status(router.route(correctionRequest(original: String(repeating: "字", count: 2_001), edited: "星朗"))), 400)
        XCTAssertEqual(status(router.route(correctionRequest(original: "星浪", edited: String(repeating: "字", count: 2_001)))), 400)
    }
}
