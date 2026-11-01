import Foundation
import Testing
@testable import VoxtMobileKit

private actor EndpointProbe {
    var count = 0
    func next() throws -> URL {
        count += 1
        return URL(string: "http://127.0.0.1:\(48000 + count)")!
    }
}

@Suite struct TransportEndpointTests {
    private func client(id: String, json: String, resolve: @escaping @Sendable () async throws -> URL) -> VoxtClient {
        StubURLProtocol.lock.lock()
        StubURLProtocol.stubs[id] = .init(status: 200, body: Data(json.utf8))
        StubURLProtocol.lock.unlock()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        config.httpAdditionalHeaders = ["X-Test-Session": id]
        return VoxtClient(server: .init(baseURL: URL(string: "https://old.example.test")!, token: "fixture-bearer"), session: URLSession(configuration: config), resolveBaseURL: resolve)
    }

    @Test func eachRequestResolvesCurrentEndpointAndPreservesBodyQueryAndBearer() async throws {
        let id = UUID().uuidString, probe = EndpointProbe()
        let c = client(id: id, json: #"{"id":"fixture","text":"hello","rawText":"hello"}"#) { try await probe.next() }
        for port in [48002, 48004] {
            _ = try await c.dictate(audio: Data([0, 1, 255]), contentType: "audio/wav", enhance: false)
            let (req, body) = try #require(StubURLProtocol.lastRequest(id: id))
            #expect(req.url?.absoluteString == "http://127.0.0.1:\(port)/v1/dictations?enhance=0&language=auto")
            #expect(req.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-bearer")
            #expect(body == Data([0, 1, 255]))
        }
        #expect(await probe.count == 4) // capability GET and audio POST each resolve afresh
    }

    @Test func resolverFailureNeverSendsToOriginalAddress() async {
        let id = UUID().uuidString
        let c = client(id: id, json: "{}") { throw URLError(.cannotConnectToHost) }
        await #expect(throws: URLError.self) { try await c.health() }
        #expect(StubURLProtocol.lastRequest(id: id) == nil)
    }

    /// A native bridge that never answers must surface as unavailable, not a request that
    /// waits forever on "Writing it up…".
    @Test func resolverThatNeverReturnsTimesOutAsTransportUnavailable() async {
        let id = UUID().uuidString
        var c = client(id: id, json: "{}") {
            try await Task.sleep(for: .seconds(3600))
            return URL(string: "http://127.0.0.1:48005")!
        }
        c.endpointTimeout = 0.2
        let started = Date()
        await #expect(throws: VoxtClient.Error.transportUnavailable) { try await c.health() }
        #expect(Date().timeIntervalSince(started) < 5)
        #expect(StubURLProtocol.lastRequest(id: id) == nil)
    }

    @Test func cancellationDuringResolutionNeverSendsHTTPRequest() async {
        let id = UUID().uuidString
        let c = client(id: id, json: #"{"ok":true,"app":"Voxt","version":"fixture"}"#) {
            // An adapter may complete its native hop after cancellation.
            withUnsafeCurrentTask { $0?.cancel() }
            return URL(string: "http://127.0.0.1:48003")!
        }
        let operation = Task { try await c.health() }
        await #expect(throws: CancellationError.self) { try await operation.value }
        #expect(StubURLProtocol.lastRequest(id: id) == nil)
    }
}
