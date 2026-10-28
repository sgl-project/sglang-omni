// DoubaoDictionaryRequestPayloadBuilderTests.swift
// Provides Doubao Dictionary Request Payload Builder Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class DoubaoDictionaryRequestPayloadBuilderTests: XCTestCase {
    func testRequestScopedBuildMapsTermsAndReplacementTerms() {
        let groupID = UUID()
        let entries = [
            TestFactories.makeEntry(term: "OpenAI", replacementTerms: ["open ai"], groupID: nil),
            TestFactories.makeEntry(term: "Doubao", replacementTerms: ["豆包大模型"], groupID: groupID)
        ]
        let configuration = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            model: DoubaoASRConfiguration.modelV2,
            appID: "123",
            accessToken: "token",
            doubaoDictionaryMode: DoubaoDictionaryMode.requestScoped.rawValue
        )

        let payload = DoubaoDictionaryRequestPayloadBuilder.build(
            configuration: configuration,
            entries: entries,
            dictionaryEnabled: true
        )

        XCTAssertEqual(payload.hotwords, ["OpenAI", "Doubao"])
        XCTAssertEqual(payload.correctWords["open ai"], "OpenAI")
        XCTAssertEqual(payload.correctWords["豆包大模型"], "Doubao")
    }

    func testBuildReturnsEmptyPayloadWhenDictionaryDisabled() {
        let configuration = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            model: DoubaoASRConfiguration.modelV2,
            appID: "123",
            accessToken: "token"
        )

        let payload = DoubaoDictionaryRequestPayloadBuilder.build(
            configuration: configuration,
            entries: [TestFactories.makeEntry(term: "OpenAI", replacementTerms: ["open ai"])],
            dictionaryEnabled: false
        )

        XCTAssertTrue(payload.isEmpty)
    }

    func testRequestScopedBuildRespectsPerRequestToggles() {
        let configuration = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            model: DoubaoASRConfiguration.modelV2,
            appID: "123",
            accessToken: "token",
            doubaoDictionaryMode: DoubaoDictionaryMode.requestScoped.rawValue,
            doubaoEnableRequestHotwords: false,
            doubaoEnableRequestCorrections: true
        )

        let payload = DoubaoDictionaryRequestPayloadBuilder.build(
            configuration: configuration,
            entries: [TestFactories.makeEntry(term: "OpenAI", replacementTerms: ["open ai"])],
            dictionaryEnabled: true
        )

        XCTAssertTrue(payload.hotwords.isEmpty)
        XCTAssertEqual(payload.correctWords, ["open ai": "OpenAI"])
    }
}
