import Foundation
import Testing
@testable import VoxtMobileKit

@Suite struct CompositionTests {
    private func optionsWithVariants(_ variants: [String]) throws -> CompositionOptions {
        var object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(CompositionOptions())) as? [String: Any])
        object["languageVariants"] = variants
        return try JSONDecoder().decode(CompositionOptions.self, from: JSONSerialization.data(withJSONObject: object))
    }

    @Test func variantsSurviveTransportAndRejectUnknownOrConflictingChoices() throws {
        let valid = try optionsWithVariants(["en-GB", "zh-Hant", "fr-CA"])
        try valid.validate()
        let encoded = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(valid)) as? [String: Any])
        #expect(encoded["languageVariants"] as? [String] == ["en-GB", "zh-Hant", "fr-CA"])
        for invalid in [["banana"], ["en-GB", "en-US"], ["zh-Hans", "zh-Hant"], ["fr-CA", "fr-CA"]] {
            #expect(throws: CompositionOptions.ValidationError.self) { try optionsWithVariants(invalid).validate() }
        }
    }

    @Test func editRequiresSelectionAndTranslationRequiresTarget() throws {
        #expect(throws: CompositionOptions.ValidationError.self) {
            try CompositionOptions(mode: .edit).validate()
        }
        #expect(throws: CompositionOptions.ValidationError.self) {
            try CompositionOptions(mode: .translate, targetLanguage: "banana").validate()
        }
        try CompositionOptions(mode: .translate, targetLanguage: "ja").validate()
        try CompositionOptions(mode: .edit, selectedText: "Original text").validate()
    }

    @Test func composeSendsAudioAndContextInAuthenticatedJSON() async throws {
        let id = UUID().uuidString
        let client = StubURLProtocol.makeClient(id: id, status: 200,
            json: #"{"id":"c1","text":"Revised","rawText":"Make this shorter","mode":"edit","asrMs":8,"enhanceMs":12}"#)
        let options = CompositionOptions(mode: .edit, selectedText: "Original text", contextBefore: "Dear team,", style: .formal, vocabulary: ["Voxt"])
        let result = try await client.compose(audio: Data([1, 2, 3]), contentType: "audio/wav", options: options)
        #expect(result.text == "Revised")
        let (request, body) = try #require(StubURLProtocol.lastRequest(id: id))
        #expect(request.url?.path == "/v1/compose")
        #expect(request.url?.query == nil)
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret-token")
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
        let data = try #require(body)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["audio"] as? String == "AQID")
        let sent = try #require(object["options"] as? [String: Any])
        #expect(sent["mode"] as? String == "edit")
        #expect(sent["selectedText"] as? String == "Original text")
        #expect(sent["style"] as? String == "formal")
    }

    @Test func notAppliedEditNeverReturnsInsertableTextAndLegacyResponseStillWorks() async throws {
        let options = CompositionOptions(mode: .edit, selectedText: "The server has a cash limit.")
        for status in [",\"status\":\"not_applied\"", ""] {
            let client = StubURLProtocol.makeClient(id: UUID().uuidString, status: 200,
                json: "{\"id\":\"edit\",\"text\":\"The server has a cash limit.\",\"rawText\":\"Fix cash to cash.\",\"mode\":\"edit\"\(status)}")
            do {
                let result = try await client.compose(audio: Data([1]), contentType: "audio/wav", options: options)
                #expect(status.isEmpty, "not_applied must not return insertable text")
                #expect(result.text == options.selectedText)
            } catch {
                #expect(!status.isEmpty)
                #expect(error.localizedDescription == "Couldn't apply that edit")
            }
        }
    }

    @Test func serverCannotSilentlyIgnoreRequestedMode() async throws {
        let id = UUID().uuidString
        let client = StubURLProtocol.makeClient(id: id, status: 200,
            json: #"{"id":"c1","text":"raw speech","rawText":"raw speech","mode":"dictate"}"#)
        await #expect(throws: VoxtClient.Error.invalidResponse) {
            try await client.compose(audio: Data([1]), contentType: "audio/wav", options: .init(mode: .translate, targetLanguage: "en"))
        }
    }

    @Test func invalidOptionsNeverSendAudio() async {
        let id = UUID().uuidString
        let client = StubURLProtocol.makeClient(id: id, status: 200, json: "{}")
        await #expect(throws: CompositionOptions.ValidationError.self) {
            try await client.compose(audio: Data([1]), contentType: "audio/wav", options: .init(mode: .edit))
        }
        #expect(StubURLProtocol.lastRequest(id: id) == nil)
    }
    @Test func defaultMobileDictationUsesComposeCleanup() async throws {
        let id = UUID().uuidString
        let client = StubURLProtocol.makeClient(id: id, status: 200,
            json: #"{"id":"c1","text":"Wednesday at three.","rawText":"Tuesday no Wednesday at three","mode":"dictate"}"#)
        _ = try await client.transcribe(wav: Data([1]), options: .init())
        let (request, _) = try #require(StubURLProtocol.lastRequest(id: id))
        #expect(request.url?.path == "/v1/compose")
    }

}
