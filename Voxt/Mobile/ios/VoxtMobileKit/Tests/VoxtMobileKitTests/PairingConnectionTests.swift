import Foundation
import Testing
@testable import VoxtMobileKit

private actor PendingEndpoint {
    var continuation: CheckedContinuation<URL, Never>?
    var started: CheckedContinuation<Void, Never>?
    var closes = 0
    func resolve() async -> URL {
        await withCheckedContinuation { continuation = $0; started?.resume(); started = nil }
    }
    func waitUntilStarted() async {
        if continuation != nil { return }
        await withCheckedContinuation { started = $0 }
    }
    func finish() { continuation?.resume(returning: URL(string: "http://127.0.0.1:48001")!); continuation = nil }
    func close() { closes += 1 }
}

@Suite @MainActor struct PairingConnectionTests {
    private let config = ServerConfig(baseURL: URL(string: "https://fixture.invalid")!, token: "fixture")
    @Test func directConnectionResolvesUntilRevoked() async throws {
        let subject = PairingConnection(server: config)
        #expect(try await subject.endpoint() == config.baseURL)
        await subject.invalidate().value
        await #expect(throws: CancellationError.self) { try await subject.endpoint() }
    }
    @Test func revokingDuringResolutionRejectsLateEndpointAndClosesOnce() async throws {
        let pending = PendingEndpoint()
        let subject = PairingConnection(server: config, resolve: { await pending.resolve() }, close: { await pending.close() })
        let operation = Task { try await subject.endpoint() }
        await pending.waitUntilStarted()
        await subject.invalidate().value
        await subject.invalidate().value
        await pending.finish()
        await #expect(throws: CancellationError.self) { try await operation.value }
        #expect(await pending.closes == 1)
    }
    @Test func releasingUnusedConnectionClosesItsTransport() async throws {
        let pending = PendingEndpoint()
        var subject: PairingConnection? = PairingConnection(server: config, close: { await pending.close() })
        #expect(subject != nil)
        subject = nil
        // Closing is asynchronous; a bounded wait also makes a missing close fail.
        for _ in 0..<100 {
            if await pending.closes > 0 { break }
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(await pending.closes == 1)
    }
    @Test func tailcatWithoutResolverFailsLocally() async {
        var tailcat = config
        tailcat.tailcat = .init(address: "fixture", clientKey: "fixture", remotePort: 1234)
        let subject = PairingConnection(server: tailcat)
        await #expect(throws: VoxtClient.Error.transportUnavailable) { try await subject.endpoint() }
    }
}

private actor RecoveryEndpoints {
    var attempts = 0
    func next() -> URL {
        attempts += 1
        return URL(string: attempts == 1 ? "http://stale.fixture.invalid" : "http://fresh.fixture.invalid")!
    }
}

private final class RecoveryProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        if request.url?.host == "stale.fixture.invalid" {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
            return
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(#"{"ok":true,"app":"Voxt","version":"restarted"}"#.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@Suite @MainActor struct ConnectionRecoveryTests {
    @Test func healthRedialsFreshEndpointAfterTransportFailure() async throws {
        let endpoints = RecoveryEndpoints()
        let config = ServerConfig(baseURL: URL(string: "http://tailcat.invalid")!, token: "fixture",
                                  tailcat: .init(address: "fixture", clientKey: "fixture", remotePort: 48931))
        let connection = PairingConnection(server: config, resolve: { await endpoints.next() })
        let sessionConfig = URLSessionConfiguration.ephemeral
        sessionConfig.protocolClasses = [RecoveryProtocol.self]
        let session = URLSession(configuration: sessionConfig)
        defer { session.invalidateAndCancel() }
        let health = try await connection.client(session: session).health()
        #expect(health.version == "restarted")
        #expect(await endpoints.attempts == 2)
    }
}
