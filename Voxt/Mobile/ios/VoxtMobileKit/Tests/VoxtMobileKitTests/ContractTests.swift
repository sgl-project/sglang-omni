import Foundation
import Testing
@testable import VoxtMobileKit

/// The Mac's Local API as recorded from its router. The phone must read every reply and send
/// requests of the same shape.
enum Contract {
    static let folder = folder(environment: ProcessInfo.processInfo.environment,
                               testsDirectory: URL(fileURLWithPath: #filePath).deletingLastPathComponent())

    /// VOXT_CONTRACT_DIR if set; else Voxt/Mobile/contract when this package sits at
    /// Voxt/Mobile/ios/VoxtMobileKit; else the copy beside these tests.
    static func folder(environment: [String: String], testsDirectory: URL) -> URL {
        if let named = environment["VOXT_CONTRACT_DIR"], !named.isEmpty { return URL(fileURLWithPath: named) }
        let shared = testsDirectory.appendingPathComponent("../../../../contract").standardizedFileURL
        var isFolder: ObjCBool = false
        if FileManager.default.fileExists(atPath: shared.path, isDirectory: &isFolder), isFolder.boolValue { return shared }
        return testsDirectory.appendingPathComponent("Fixtures/contract")
    }
    static func data(_ name: String) throws -> Data { try Data(contentsOf: folder.appendingPathComponent("\(name).json")) }
    static func object(_ name: String) throws -> [String: Any] {
        let bytes = try data(name)
        return try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
    }
    static func text(_ name: String) throws -> String { String(decoding: try data(name), as: UTF8.self) }
    static func json(_ object: Any) throws -> String { String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self) }
    static let legacyCapabilities = #"{"transcription":"local","text":"local"}"#
    /// Endpoints a Mac can leave out.
    static let optionalEndpoints = VoxtClient.Feature.allCases.filter { $0 != .compose }.map(\.rawValue)

    /// The recorded capabilities with `dropped` endpoint names removed, as a Mac without them sends.
    static func capabilities(without dropped: [String]) throws -> String {
        var object = try Contract.object("capabilities")
        object["endpoints"] = ((object["endpoints"] as? [String]) ?? []).filter { !dropped.contains($0) }
        return try Contract.json(object)
    }

    /// A client whose Mac answers /v1/capabilities with `capabilities` and everything else with `json`.
    static func client(id: String, capabilities: String, status: Int = 200, json: String = "{}") -> VoxtClient {
        StubURLProtocol.lock.withLock { StubURLProtocol.capabilities[id] = .init(status: 200, body: Data(capabilities.utf8)) }
        return StubURLProtocol.makeClient(id: id, status: status, json: json)
    }

    static func paths(_ id: String) -> [String] { StubURLProtocol.lock.withLock { StubURLProtocol.paths[id] ?? [] } }
}

@Suite struct ContractTests {
    @Test func capabilitiesSayWhereTheModelsRunAndWhatTheMacServes() throws {
        let capabilities = try JSONDecoder().decode(VoxtClient.Capabilities.self, from: Contract.data("capabilities"))
        #expect(capabilities.isLocal && !capabilities.hasRemote)
        #expect(VoxtClient.Feature.allCases.allSatisfy(capabilities.supports))
        let without = try JSONDecoder().decode(VoxtClient.Capabilities.self,
                                               from: Data(Contract.capabilities(without: ["corrections", "history/sync"]).utf8))
        #expect(without.supports(.compose) && without.supports(.composePrompts))
        #expect(!without.supports(.corrections) && !without.supports(.historySync))
    }

    @Test func correctionSendsTheRecordedPassageAndReadsTheWord() async throws {
        let recorded = try Contract.object("correction-request")
        let id = UUID().uuidString
        let client = Contract.client(id: id, capabilities: try Contract.text("capabilities"), json: try Contract.text("correction-response"))
        let passage = try JSONDecoder().decode(CorrectionLearningRequest.self, from: Contract.data("correction-request"))
        let correction = try await client.correction(for: passage)
        #expect(correction?.original == "Nova" && correction?.replacement == "Novo")
        let (_, body) = try #require(StubURLProtocol.lastRequest(id: id))
        let bytes = try #require(body)
        let sent = try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        #expect(Self.differences(sent: sent, recorded: recorded).isEmpty)
    }

    @Test func aRemoteOrUnknownLocationIsNotLocal() throws {
        for (text, remote) in [("remote", true), ("unknown", false)] {
            var object = try Contract.object("capabilities")
            object["locations"] = ["transcription": "local", "text": text]
            let capabilities = try JSONDecoder().decode(VoxtClient.Capabilities.self, from: Data(Contract.json(object).utf8))
            #expect(!capabilities.isLocal)
            #expect(capabilities.hasRemote == remote)
        }
    }

    /// A Voxt with every endpoint answers with only the two locations.
    @Test func aMacThatListsNoEndpointsKeepsEveryFeature() throws {
        let capabilities = try JSONDecoder().decode(VoxtClient.Capabilities.self, from: Data(Contract.legacyCapabilities.utf8))
        #expect(capabilities.isLocal)
        #expect(VoxtClient.Feature.allCases.allSatisfy(capabilities.supports))
    }

    @Test func healthDecodes() throws {
        let health = try JSONDecoder().decode(VoxtClient.Health.self, from: Contract.data("health"))
        #expect(health.ok && health.app == "Voxt" && health.version == "1.16.0")
    }

    /// Without `language` the Mac uses its own language settings; the phone keeps detecting the spoken one.
    @Test func dictationAsksTheMacToDetectTheLanguageAndReadsTheReply() async throws {
        let id = UUID().uuidString
        let client = Contract.client(id: id, capabilities: try Contract.text("capabilities"), json: try Contract.text("dictation-response"))
        let result = try await client.dictate(audio: Data([1, 2, 3]), contentType: "audio/wav")
        #expect(result.text == "Tomorrow at three, we meet in room four.")
        #expect(result.rawText == "um tomorrow at three we meet in room four")
        #expect(result.asrMs == 531 && result.enhanceMs == 412)
        let (request, _) = try #require(StubURLProtocol.lastRequest(id: id))
        #expect(request.url?.query == "language=auto")
        _ = try await client.dictate(audio: Data([1]), contentType: "audio/wav", enhance: false)
        let (literal, _) = try #require(StubURLProtocol.lastRequest(id: id))
        #expect(literal.url?.query == "enhance=0&language=auto")
    }

    @Test func composeSendsTheRecordedRequestAndReadsTheReply() async throws {
        let recorded = try Contract.object("compose-request")
        let recordedOptions = try #require(recorded["options"])
        let options = try JSONDecoder().decode(CompositionOptions.self, from: Data(Contract.json(recordedOptions).utf8))
        let id = UUID().uuidString
        let client = Contract.client(id: id, capabilities: try Contract.text("capabilities"), json: try Contract.text("compose-response"))
        let result = try await client.compose(audio: Data([1, 2, 3]), contentType: "audio/wav", options: options)
        #expect(result.text == "Tomorrow at three, we meet in room four.")
        let (_, body) = try #require(StubURLProtocol.lastRequest(id: id))
        let bytes = try #require(body)
        let sent = try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        // Voxt ignores keys it doesn't read, so the phone's body matches on the fixture's keys.
        #expect(Self.differences(sent: sent, recorded: recorded).isEmpty, "\(Self.differences(sent: sent, recorded: recorded))")
    }

    /// Keys of `recorded`, nested, whose value in `sent` is missing or different.
    static func differences(sent: [String: Any], recorded: [String: Any], path: String = "") -> [String] {
        recorded.keys.sorted().flatMap { key -> [String] in
            let name = path + key
            if let inner = recorded[key] as? [String: Any] {
                guard let other = sent[key] as? [String: Any] else { return [name] }
                return differences(sent: other, recorded: inner, path: name + ".")
            }
            guard let value = sent[key] else { return [name] }
            return (value as AnyObject).isEqual(recorded[key]) ? [] : [name]
        }
    }

    @Test func anEditTheMacCouldNotApplyIsReported() async throws {
        let id = UUID().uuidString
        let client = Contract.client(id: id, capabilities: try Contract.text("capabilities"), json: try Contract.text("compose-not-applied"))
        await #expect(throws: VoxtClient.Error.editNotApplied) {
            _ = try await client.compose(audio: Data([1]), contentType: "audio/wav", options: CompositionOptions(mode: .edit, selectedText: "See you at the cafe."))
        }
    }

    @Test func everyRecordedErrorKeepsTheMacsMessage() async throws {
        let errors = try #require(Contract.object("errors") as? [String: [String: Any]])
        for (name, entry) in errors {
            let status = try #require(entry["status"] as? Int)
            let message = try #require((entry["body"] as? [String: String])?["error"])
            let body = try #require(entry["body"])
            let client = Contract.client(id: UUID().uuidString, capabilities: try Contract.text("capabilities"),
                                         status: status, json: try Contract.json(body))
            let expected: VoxtClient.Error = status == 401 ? .unauthorized : .server(status: status, message: message)
            await #expect(throws: expected, "\(name)") { _ = try await client.dictate(audio: Data([1]), contentType: "audio/wav") }
        }
    }
}

/// Developer prompts, corrections and history sync are optional on the Mac: the phone uses each
/// only when capabilities list it, or when the Mac lists no endpoints at all (older Voxt).
@Suite struct OptionalEndpointTests {
    private func capabilities(listing kept: [String]) throws -> String {
        try Contract.capabilities(without: Contract.optionalEndpoints.filter { !kept.contains($0) })
    }

    private func call(_ feature: VoxtClient.Feature, _ client: VoxtClient) async throws {
        switch feature {
        case .composePrompts: _ = try await client.builtInPrompts()
        case .corrections: _ = try await client.correction(for: .init(original: "The car is gray.", edited: "The car is grey."))
        case .historySync: _ = try await client.syncHistory(HistorySyncRequest())
        case .compose: break
        }
    }

    @Test(arguments: [VoxtClient.Feature.composePrompts, .corrections, .historySync])
    func aFeatureTheMacDoesNotListIsNotCalled(_ feature: VoxtClient.Feature) async throws {
        let id = UUID().uuidString
        let client = Contract.client(id: id, capabilities: try capabilities(listing: []))
        await #expect(throws: VoxtClient.Error.notOnThisMac) { try await self.call(feature, client) }
        #expect(!Contract.paths(id).contains("/v1/" + feature.rawValue))
    }

    @Test(arguments: [VoxtClient.Feature.composePrompts, .corrections, .historySync])
    func aFeatureTheMacListsIsCalled(_ feature: VoxtClient.Feature) async throws {
        let id = UUID().uuidString
        let client = Contract.client(id: id, capabilities: try capabilities(listing: [feature.rawValue]), status: 500)
        _ = try? await call(feature, client)
        #expect(Contract.paths(id).contains("/v1/" + feature.rawValue))
    }

    @Test(arguments: [VoxtClient.Feature.composePrompts, .corrections, .historySync])
    func aMacThatListsNoEndpointsIsCalled(_ feature: VoxtClient.Feature) async throws {
        let id = UUID().uuidString
        let client = Contract.client(id: id, capabilities: Contract.legacyCapabilities, status: 500)
        _ = try? await call(feature, client)
        #expect(Contract.paths(id).contains("/v1/" + feature.rawValue))
    }
}

/// The app keeps the Mac's endpoint list in the App Group so the keyboard can skip what the Mac lacks.
@Suite struct MacFeaturesTests {
    private func defaults() -> (UserDefaults, String) {
        let name = "mac-features.\(UUID().uuidString)"
        return (UserDefaults(suiteName: name)!, name)
    }

    private func capabilities(_ json: String) throws -> VoxtClient.Capabilities {
        try JSONDecoder().decode(VoxtClient.Capabilities.self, from: Data(json.utf8))
    }

    @Test func aListedEndpointIsServedAndAMissingOneIsNot() throws {
        let (store, name) = defaults()
        defer { store.removePersistentDomain(forName: name) }
        MacFeatures.save(try capabilities(try Contract.capabilities(without: ["corrections", "history/sync"])), to: store)
        #expect(MacFeatures.serves(.composePrompts, in: store))
        #expect(!MacFeatures.serves(.corrections, in: store) && !MacFeatures.serves(.historySync, in: store))
    }

    @Test func aMacThatListsNoEndpointsServesEverything() throws {
        let (store, name) = defaults()
        defer { store.removePersistentDomain(forName: name) }
        MacFeatures.save(try capabilities(try Contract.text("capabilities")), to: store)
        MacFeatures.save(try capabilities(Contract.legacyCapabilities), to: store)
        #expect(VoxtClient.Feature.allCases.allSatisfy { MacFeatures.serves($0, in: store) })
    }

    @Test func beforeTheFirstCheckEverythingIsServed() {
        let (store, name) = defaults()
        defer { store.removePersistentDomain(forName: name) }
        #expect(VoxtClient.Feature.allCases.allSatisfy { MacFeatures.serves($0, in: store) })
    }
}
