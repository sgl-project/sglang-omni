import Foundation
import Testing
@testable import VoxtConnectionKit

private final class FixtureProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let authorized = request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-token"
        let credential = request.value(forHTTPHeaderField: "Authorization") ?? ""
        if credential == "Bearer offline" {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost)); return
        }
        let status = credential == "Bearer unavailable" ? 503 : authorized && request.url?.path == "/v1/health" ? 200 : credential == "Bearer malformed" || credential == "Bearer not-ready" ? 200 : 401
        let body = Data((credential == "Bearer malformed" ? "not JSON" : credential == "Bearer not-ready" ? #"{"ok":false,"app":"Voxt","version":"fixture-1"}"# : #"{"ok":true,"app":"Voxt","version":"fixture-1"}"#).utf8)
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: status,
            httpVersion: "HTTP/1.1", headerFields: ["Content-Type": "application/json"])!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

private func fixtureSession() -> URLSession {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [FixtureProtocol.self]
    return URLSession(configuration: configuration)
}

@Test func validLocalTokenShowsVoxtVersion() async throws {
    let client = try LocalInferenceClient(baseURL: URL(string: "http://127.0.0.1:47910")!,
        token: "fixture-token", session: fixtureSession())
    let health = try await client.health()
    #expect(health.ok)
    #expect(health.version == "fixture-1")
}

@Test func onlyExplicitIPv4LoopbackEndpointsAreAccepted() throws {
    for address in ["https://example.com", "http://localhost:47910", "http://192.168.1.2:47910", "http://127.0.0.1.example.com", "http://user@127.0.0.1:47910", "http://127.0.0.1:47910/other", "http://127.0.0.1:47910?token=value", "http://127.0.0.1:47910#fragment"] {
        #expect(throws: LocalInferenceClient.Failure.invalidEndpoint) {
            try LocalInferenceClient(baseURL: URL(string: address)!, token: "fixture-token", session: fixtureSession())
        }
    }
}

@Test func healthDistinguishesUnauthorizedUnavailableAndMalformed() async throws {
    for (token, failure) in [("wrong-token", LocalInferenceClient.Failure.unauthorized), ("unavailable", .unavailable), ("offline", .unavailable), ("malformed", .invalidResponse), ("not-ready", .unavailable)] {
        let client = try LocalInferenceClient(baseURL: URL(string: "http://127.0.0.1:47910")!, token: token, session: fixtureSession())
        await #expect(throws: failure) { try await client.health() }
    }
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["CONNECTOR_REDIRECT_FIXTURE_URL"] != nil))
func localAPIRedirectNeverReachesAnotherListener() async throws {
    guard let endpoint = ProcessInfo.processInfo.environment["CONNECTOR_REDIRECT_FIXTURE_URL"],
          let url = URL(string: endpoint) else { return }
    _ = try await URLSession.shared.data(from: url.appending(path: "reset"))
    let client = try LocalInferenceClient(baseURL: url, token: "synthetic-redirect-token")
    await #expect(throws: LocalInferenceClient.Failure.invalidResponse) { try await client.health() }
    let (data, _) = try await URLSession.shared.data(from: url.appending(path: "counts"))
    struct Counts: Decodable { let received: Int; let credential_received: Bool }
    let counts = try JSONDecoder().decode(Counts.self, from: data)
    #expect(counts.received == 0)
    #expect(!counts.credential_received)
}
