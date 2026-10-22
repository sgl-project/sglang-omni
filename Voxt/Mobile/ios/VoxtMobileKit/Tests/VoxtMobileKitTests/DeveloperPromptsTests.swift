import Foundation
import Testing
@testable import VoxtMobileKit

@Suite struct DeveloperPromptsTests {
    private func store() -> DeveloperPromptStore {
        DeveloperPromptStore(defaults: UserDefaults(suiteName: "developer-prompts-\(UUID().uuidString)")!)
    }

    @Test func customPromptIsSentOnlyWhenSetAndLengthLimited() throws {
        let plain = try JSONSerialization.jsonObject(with: JSONEncoder().encode(CompositionOptions())) as? [String: Any]
        #expect(plain?["customPrompt"] == nil)
        var options = CompositionOptions()
        options.customPrompt = "Write it as a haiku."
        let custom = try JSONSerialization.jsonObject(with: JSONEncoder().encode(options)) as? [String: Any]
        #expect(custom?["customPrompt"] as? String == "Write it as a haiku.")
        options.customPrompt = String(repeating: "a", count: 4_000)
        try options.validate()
        options.customPrompt = String(repeating: "a", count: 4_001)
        #expect(throws: CompositionOptions.ValidationError.self) { try options.validate() }
    }

    @Test func promptsAreUsedOnlyWhileDeveloperModeIsOn() {
        let prompts = store()
        #expect(prompts.isEnabled == false)
        prompts.setPrompt("Be brief.", for: .dictate, builtIn: "Clean up the speech.")
        #expect(prompts.prompt(for: .dictate) == "Be brief.")
        #expect(prompts.override(for: .dictate) == nil)
        prompts.isEnabled = true
        #expect(prompts.override(for: .dictate) == "Be brief.")
        #expect(prompts.override(for: .edit) == nil)
        prompts.isEnabled = false
        #expect(prompts.prompt(for: .dictate) == "Be brief.", "turning Developer mode off keeps the prompts")
        #expect(prompts.override(for: .dictate) == nil)
    }

    @Test func builtInOrEmptyTextMeansNoCustomPromptAndResetClearsIt() {
        let prompts = store()
        prompts.isEnabled = true
        prompts.setPrompt("Clean up the speech.", for: .dictate, builtIn: "Clean up the speech.")
        #expect(prompts.prompt(for: .dictate) == nil)
        prompts.setPrompt("   ", for: .dictate, builtIn: "Clean up the speech.")
        #expect(prompts.prompt(for: .dictate) == nil)
        for mode in CompositionOptions.Mode.allCases { prompts.setPrompt("Custom \(mode)", for: mode, builtIn: "") }
        prompts.reset(.edit)
        #expect(prompts.prompt(for: .edit) == nil)
        #expect(prompts.prompt(for: .write) == "Custom write")
        prompts.resetAll()
        #expect(CompositionOptions.Mode.allCases.allSatisfy { prompts.prompt(for: $0) == nil })
        #expect(prompts.isEnabled, "reset keeps Developer mode on")
    }

    @Test func clientAttachesTheCustomPromptOfTheRequestsMode() async throws {
        let id = UUID().uuidString
        let json = "{\"id\":\"\(UUID().uuidString)\",\"text\":\"T\",\"rawText\":\"T\",\"mode\":\"edit\"}"
        StubURLProtocol.lock.withLock { StubURLProtocol.stubs[id] = .init(status: 200, body: Data(json.utf8)) }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        configuration.httpAdditionalHeaders = ["X-Test-Session": id]
        let client = VoxtClient(server: ServerConfig(baseURL: URL(string: "https://mac.tail1.ts.net:8900")!, token: "secret-token"),
                                session: URLSession(configuration: configuration),
                                customPrompt: { mode in mode == .edit ? "Apply the instruction." : nil })
        _ = try await client.transcribeResult(wav: Data([0]), options: .init(mode: .edit, selectedText: "See you Monday."))
        let body = try #require(StubURLProtocol.lastRequest(id: id)?.1)
        let options = try #require((try JSONSerialization.jsonObject(with: body) as? [String: Any])?["options"] as? [String: Any])
        #expect(options["customPrompt"] as? String == "Apply the instruction.")
    }

    @Test func builtInPromptsComeFromTheMacWithThePairingToken() async throws {
        let id = UUID().uuidString
        let client = StubURLProtocol.makeClient(id: id, status: 200, json: #"{"prompts":{"dictate":"D","edit":"E","translate":"T {{TARGET_LANGUAGE}}","write":"W"}}"#)
        let prompts = try await client.builtInPrompts()
        #expect(prompts[.dictate] == "D")
        #expect(prompts[.translate] == "T {{TARGET_LANGUAGE}}")
        let (request, _) = try #require(StubURLProtocol.lastRequest(id: id))
        #expect(request.url?.path == "/v1/compose/prompts")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret-token")
    }
}
