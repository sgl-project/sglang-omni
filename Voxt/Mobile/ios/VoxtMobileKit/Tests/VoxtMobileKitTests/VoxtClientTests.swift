import Foundation
import Testing
@testable import VoxtMobileKit

/// Serves canned responses and records the last request. Each test uses its own
/// session id so parallel tests don't share stubs.
final class StubURLProtocol: URLProtocol, @unchecked Sendable {
    struct Stub { var status: Int; var body: Data }
    nonisolated(unsafe) static var stubs: [String: Stub] = [:]
    nonisolated(unsafe) static var requests: [String: (URLRequest, Data?)] = [:]
    nonisolated(unsafe) static var paths: [String: [String]] = [:]
    /// Sessions whose /v1/health gets a fixed healthy reply instead of their own stub.
    nonisolated(unsafe) static var healthySessions: Set<String> = []
    /// Upcoming /v1/compose failures for a session, applied in order before any response.
    nonisolated(unsafe) static var composeFailures: [String: [URLError.Code]] = [:]
    /// How many upcoming /v1/health requests of a session fail with cannotConnectToHost.
    nonisolated(unsafe) static var healthFailures: [String: Int] = [:]
    /// A session's own /v1/capabilities reply; others get the older two-field reply.
    nonisolated(unsafe) static var capabilities: [String: Stub] = [:]
    static let lock = NSLock()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let id = request.value(forHTTPHeaderField: "X-Test-Session") ?? ""
        var body = request.httpBody
        if body == nil, let stream = request.httpBodyStream {
            stream.open()
            var data = Data()
            var buffer = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable {
                let n = stream.read(&buffer, maxLength: buffer.count)
                if n <= 0 { break }
                data.append(buffer, count: n)
            }
            stream.close()
            body = data
        }
        Self.lock.lock()
        Self.requests[id] = (request, body)
        Self.paths[id, default: []].append(request.url?.path ?? "")
        if request.url?.path == "/v1/health", let failures = Self.healthFailures[id], failures > 0 {
            Self.healthFailures[id] = failures - 1
            Self.lock.unlock()
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
            return
        }
        if request.url?.path == "/v1/compose", var failures = Self.composeFailures[id], !failures.isEmpty {
            let code = failures.removeFirst()
            Self.composeFailures[id] = failures
            Self.lock.unlock()
            client?.urlProtocol(self, didFailWithError: URLError(code))
            return
        }
        let stub = request.url?.path == "/v1/capabilities"
            ? Self.capabilities[id] ?? Stub(status: 200, body: Data(#"{"transcription":"local","text":"local"}"#.utf8))
            : request.url?.path == "/v1/health" && Self.healthySessions.contains(id)
            ? Stub(status: 200, body: Data(#"{"ok":true,"app":"Voxt","version":"fixture"}"#.utf8))
            : Self.stubs[id] ?? Stub(status: 500, body: Data())
        Self.lock.unlock()
        let response = HTTPURLResponse(url: request.url!, statusCode: stub.status, httpVersion: nil, headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: stub.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    static func makeClient(id: String, status: Int, json: String) -> VoxtClient {
        lock.lock(); stubs[id] = Stub(status: status, body: Data(json.utf8)); lock.unlock()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        configuration.httpAdditionalHeaders = ["X-Test-Session": id]
        let server = ServerConfig(baseURL: URL(string: "https://mac.tail1.ts.net:8900")!, token: "secret-token")
        return VoxtClient(server: server, session: URLSession(configuration: configuration))
    }

    static func lastRequest(id: String) -> (URLRequest, Data?)? {
        lock.lock(); defer { lock.unlock() }
        return requests[id]
    }
}

@Suite struct VoxtClientTests {
    /// With the app in the background the tunnel to the Mac can be down, and two health checks
    /// back to back both fail. Before any audio leaves, the client resets the tunnel and waits
    /// for it, a few times, so the dictation isn't lost.
    @Test func audioWaitsForTheTunnelBeforeSendingAndNeverSendsWithoutIt() async throws {
        let id = UUID().uuidString
        let okBody = Data("{\"id\":\"\(UUID().uuidString)\",\"text\":\"Hi\",\"rawText\":\"Hi\",\"mode\":\"dictate\"}".utf8)
        StubURLProtocol.lock.withLock {
            StubURLProtocol.stubs[id] = .init(status: 200, body: okBody)
            StubURLProtocol.healthySessions.insert(id)
            StubURLProtocol.healthFailures[id] = 3
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        configuration.httpAdditionalHeaders = ["X-Test-Session": id]
        let server = ServerConfig(baseURL: URL(string: "http://tailcat.invalid")!, token: "fixture-bearer",
                                  tailcat: .init(address: "fixture", clientKey: "fixture", remotePort: 48931))
        let resets = Counter()
        let client = VoxtClient(server: server, session: URLSession(configuration: configuration),
                                resolveBaseURL: { URL(string: "http://127.0.0.1:48010")! },
                                onTransportFailure: { _ in resets.add() }, tunnelRetryDelay: .zero)
        func count(_ path: String) -> Int {
            StubURLProtocol.lock.withLock { StubURLProtocol.paths[id, default: []].filter { $0 == path }.count }
        }
        let result = try await client.transcribeResult(wav: Data([1]), options: CompositionOptions())
        #expect(result.text == "Hi")
        #expect(count("/v1/health") == 4)
        #expect(resets.value == 3, "each failure resets the tunnel")
        #expect(count("/v1/compose") == 1)

        StubURLProtocol.lock.withLock { StubURLProtocol.healthFailures[id] = 4 }
        await #expect(throws: VoxtClient.Error.transportUnavailable) {
            _ = try await client.transcribeResult(wav: Data([2]), options: CompositionOptions())
        }
        #expect(count("/v1/compose") == 1, "no audio is sent without a tunnel")
    }

    /// Health + capabilities before every dictation add a round trip each time. A request
    /// that succeeded recently already proves both.
    @Test func preChecksRunOncePerFreshWindowAndAgainAfterAFailureOrPolicyChange() async throws {
        let id = UUID().uuidString
        let resultID = UUID().uuidString
        let okBody = Data("{\"id\":\"\(resultID)\",\"text\":\"Hi\",\"rawText\":\"Hi\",\"mode\":\"dictate\"}".utf8)
        StubURLProtocol.lock.withLock {
            StubURLProtocol.stubs[id] = .init(status: 200, body: okBody)
            StubURLProtocol.healthySessions.insert(id)
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        configuration.httpAdditionalHeaders = ["X-Test-Session": id]
        let clock = TestClock()
        let cache = PreflightCache(ttl: 60, now: { clock.now })
        let server = ServerConfig(baseURL: URL(string: "http://tailcat.invalid")!, token: "fixture-bearer",
                                  tailcat: .init(address: "fixture", clientKey: "fixture", remotePort: 48931))
        let cloud = CloudSwitch()
        let client = VoxtClient(server: server, session: URLSession(configuration: configuration),
                                resolveBaseURL: { URL(string: "http://127.0.0.1:48010")! }, onTransportFailure: { _ in },
                                allowCloudModels: { cloud.allowed }, preflight: cache)
        func count(_ path: String) -> Int {
            StubURLProtocol.lock.withLock { StubURLProtocol.paths[id, default: []].filter { $0 == path }.count }
        }
        let options = CompositionOptions()
        _ = try await client.transcribeResult(wav: Data([1]), options: options)
        _ = try await client.transcribeResult(wav: Data([2]), options: options)
        #expect(count("/v1/health") == 1)
        #expect(count("/v1/capabilities") == 1)
        #expect(count("/v1/compose") == 2)

        clock.now = clock.now.addingTimeInterval(61)
        _ = try await client.transcribeResult(wav: Data([3]), options: options)
        #expect(count("/v1/capabilities") == 2, "expired window checks again")

        StubURLProtocol.lock.withLock { StubURLProtocol.stubs[id] = .init(status: 503, body: Data(#"{"error":"busy"}"#.utf8)) }
        _ = try? await client.transcribeResult(wav: Data([4]), options: options)
        StubURLProtocol.lock.withLock { StubURLProtocol.stubs[id] = .init(status: 200, body: okBody) }
        _ = try await client.transcribeResult(wav: Data([5]), options: options)
        #expect(count("/v1/capabilities") == 3, "a failed request drops the cache")

        await MainActor.run { cloud.allowed = true }
        _ = try await client.transcribeResult(wav: Data([6]), options: options)
        await MainActor.run { cloud.allowed = false }
        _ = try await client.transcribeResult(wav: Data([7]), options: options)
        #expect(count("/v1/capabilities") == 4, "turning cloud models off again re-checks locality")

        // Cached path, connection refused: the audio never left, so retry once through the checks.
        StubURLProtocol.lock.withLock { StubURLProtocol.composeFailures[id] = [.cannotConnectToHost] }
        let composesBefore = count("/v1/compose"), checksBefore = count("/v1/capabilities")
        let retried = try await client.transcribeResult(wav: Data([8]), options: options)
        #expect(retried.text == "Hi")
        #expect(count("/v1/compose") == composesBefore + 2)
        #expect(count("/v1/capabilities") == checksBefore + 1, "the retry ran the full pre-checks")

        // Cached path, connection lost mid-request: it may have been accepted, so no automatic retry.
        StubURLProtocol.lock.withLock { StubURLProtocol.composeFailures[id] = [.networkConnectionLost] }
        let composesBeforeLost = count("/v1/compose")
        await #expect(throws: VoxtClient.Error.transportUnavailable) {
            _ = try await client.transcribeResult(wav: Data([9]), options: options)
        }
        #expect(count("/v1/compose") == composesBeforeLost + 1)
    }

    @Test func literalOnlyBypassesComposeForDictateAndKeepsDictionaryText() async throws {
        for mode in CompositionOptions.Mode.allCases {
            for literal in [false, true] {
                let session = UUID().uuidString
                let resultID = UUID()
                let json = "{\"id\":\"\(resultID.uuidString)\",\"text\":\"Voxt\",\"rawText\":\"Vox\",\"mode\":\"\(mode.rawValue)\",\"asrMs\":500,\"enhanceMs\":0}"
                let client = StubURLProtocol.makeClient(id: session, status: 200, json: json)
                let audio = Data([1, 2, 3])
                let options = CompositionOptions(mode: mode, targetLanguage: mode == .translate ? "en" : nil, selectedText: "Original", literalDictation: literal)
                let result = try await client.transcribeResult(wav: audio, options: options)
                #expect(result.text == "Voxt")
                #expect(result.rawText == "Vox")
                #expect(result.id == resultID)
                #expect((result.preflightMs ?? -1) >= 0, "phone-side pre-check time is measured")
                #expect((result.requestMs ?? -1) >= 0, "phone-side request time is measured")
                let (request, body) = try #require(StubURLProtocol.lastRequest(id: session))
                if literal && mode == .dictate {
                    #expect(request.url?.path == "/v1/dictations")
                    #expect(request.url?.query == "enhance=0&language=auto")
                    #expect(request.value(forHTTPHeaderField: "Content-Type") == "audio/wav")
                    #expect(body == audio)
                } else {
                    #expect(request.url?.path == "/v1/compose")
                    #expect(request.url?.query == nil)
                }
            }
        }
    }

    @Test func oldRecordingOptionsDefaultLiteralOffAndNewOptionsRoundTrip() throws {
        var options = CompositionOptions()
        #expect(!options.literalDictation)
        var object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(options)) as? [String: Any])
        object.removeValue(forKey: "literalDictation")
        let old = try JSONDecoder().decode(CompositionOptions.self, from: JSONSerialization.data(withJSONObject: object))
        #expect(!old.literalDictation)
        options.literalDictation = true
        #expect(try JSONDecoder().decode(CompositionOptions.self, from: JSONEncoder().encode(options)) == options)
    }

    @Test func transcriberKeepsActualComposeIdentityAndRejectsNonUUID() async throws {
        let id = UUID()
        for wireID in [id.uuidString, "not-a-uuid"] {
            let json = "{\"id\":\"\(wireID)\",\"text\":\"Text\",\"rawText\":\"Text\",\"mode\":\"dictate\"}"
            let client = StubURLProtocol.makeClient(id: UUID().uuidString, status: 200, json: json)
            if wireID == id.uuidString {
                #expect(try await client.transcribeResult(wav: Data([0]), options: .init()).id == id)
            } else {
                await #expect(throws: VoxtClient.Error.invalidResponse) { try await client.transcribeResult(wav: Data([0]), options: .init()) }
            }
        }
    }
    @Test func historySyncUsesAuthenticationAndRejectsInvalidPages() async throws {
        let id = UUID().uuidString, serverID = UUID()
        let response = HistorySyncResponse(storeID: serverID, cursor: 1, hasMore: false, items: [], deletedIDs: [])
        let client = StubURLProtocol.makeClient(id: id, status: 200, json: String(decoding: try JSONEncoder().encode(response), as: UTF8.self))
        let payload = HistorySyncRequest()
        #expect(try await client.syncHistory(payload) == response)
        let (request, body) = try #require(StubURLProtocol.lastRequest(id: id))
        #expect(request.httpMethod == "POST")
        #expect(request.url?.path == "/v1/history/sync")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret-token")
        let bodyData = try #require(body)
        let object = try #require(try JSONSerialization.jsonObject(with: bodyData) as? [String: Any])
        #expect(object["localRevision"] == nil)
        #expect(object["cursor"] as? Int == 0)
        let bad = StubURLProtocol.makeClient(id: UUID().uuidString, status: 200, json: "{}")
        await #expect(throws: (any Error).self) { try await bad.syncHistory(payload) }
        await #expect(throws: HistorySyncError.invalidRequest) { try await client.syncHistory(.init(limit: 0)) }
        let disabled = StubURLProtocol.makeClient(id: UUID().uuidString, status: 403, json: #"{"error":"History sync is off on this Mac."}"#)
        await #expect(throws: VoxtClient.Error.server(status: 403, message: "History sync is off on this Mac.")) { try await disabled.syncHistory(payload) }
    }
    @Test func dictationSendsAudioWithTokenAndParsesText() async throws {
        let id = UUID().uuidString
        let client = StubURLProtocol.makeClient(
            id: id, status: 200,
            json: #"{"id":"d1","text":"明天下午三点开会。","rawText":"嗯那个明天下午三点开会","asrMs":812,"enhanceMs":640}"#
        )
        let audio = Data([1, 2, 3, 4])

        let result = try await client.dictate(audio: audio, contentType: "audio/wav")

        #expect(result.text == "明天下午三点开会。")
        #expect(result.rawText == "嗯那个明天下午三点开会")
        #expect(result.asrMs == 812)
        let (request, body) = try #require(StubURLProtocol.lastRequest(id: id))
        #expect(request.httpMethod == "POST")
        #expect(request.url?.absoluteString == "https://mac.tail1.ts.net:8900/v1/dictations?language=auto")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret-token")
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "audio/wav")
        #expect(body == audio)
    }

    @Test func skipEnhancementAddsQuery() async throws {
        let id = UUID().uuidString
        let client = StubURLProtocol.makeClient(id: id, status: 200, json: #"{"id":"d2","text":"hi","rawText":"hi"}"#)
        _ = try await client.dictate(audio: Data([0]), contentType: "audio/wav", enhance: false)
        let (request, _) = try #require(StubURLProtocol.lastRequest(id: id))
        #expect(request.url?.absoluteString == "https://mac.tail1.ts.net:8900/v1/dictations?enhance=0&language=auto")
    }

    @Test func unauthorizedMapsToTypedError() async {
        let id = UUID().uuidString
        let client = StubURLProtocol.makeClient(id: id, status: 401, json: #"{"error":"bad token"}"#)
        await #expect(throws: VoxtClient.Error.unauthorized) {
            try await client.dictate(audio: Data([0]), contentType: "audio/wav")
        }
    }

    @Test func serverErrorCarriesMessage() async {
        let id = UUID().uuidString
        let client = StubURLProtocol.makeClient(id: id, status: 503, json: #"{"error":"ASR model is not installed"}"#)
        await #expect(throws: VoxtClient.Error.server(status: 503, message: "ASR model is not installed")) {
            try await client.dictate(audio: Data([0]), contentType: "audio/wav")
        }
    }

    @Test func correctionClientValidatesResponseAndSendsOnlyBoundedPassage() async throws {
        let passage = CorrectionLearningRequest(original: "请用星浪处理。", edited: "请用星朗处理。")
        let id = UUID().uuidString
        let client = StubURLProtocol.makeClient(id: id, status: 200, json: #"{"correction":{"original":"星浪","replacement":"星朗"}}"#)
        #expect(try await client.correction(for: passage) == .init(original: "星浪", replacement: "星朗"))
        let (request, data) = try #require(StubURLProtocol.lastRequest(id: id))
        #expect(request.url?.path == "/v1/corrections")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret-token")
        #expect(try JSONDecoder().decode(CorrectionLearningRequest.self, from: #require(data)) == passage)
        for json in [#"{}"#, #"{"correction":{"original":"浪","replacement":"朗"}}"#, #"{"correction":{"original":"星浪","replacement":"幻觉"}}"#] {
            let bad = StubURLProtocol.makeClient(id: UUID().uuidString, status: 200, json: json)
            await #expect(throws: VoxtClient.Error.invalidResponse) { try await bad.correction(for: passage) }
        }
        let none = StubURLProtocol.makeClient(id: UUID().uuidString, status: 200, json: #"{"correction":null}"#)
        #expect(try await none.correction(for: passage) == nil)
    }

    @Test func healthReturnsVersion() async throws {
        let id = UUID().uuidString
        let client = StubURLProtocol.makeClient(id: id, status: 200, json: #"{"ok":true,"app":"Voxt","version":"1.15.0"}"#)
        let health = try await client.health()
        #expect(health.version == "1.15.0")
        let (request, _) = try #require(StubURLProtocol.lastRequest(id: id))
        #expect(request.url?.path == "/v1/health")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret-token")
    }
}

final class TestClock: @unchecked Sendable {
    var now = Date(timeIntervalSince1970: 1_000)
}

@MainActor final class CloudSwitch: Sendable {
    nonisolated(unsafe) var allowed = false
}

final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var value: Int { lock.withLock { count } }
    func add() { lock.withLock { count += 1 } }
}
