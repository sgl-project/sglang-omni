import Foundation
import Testing
@testable import VoxtMobileKit

private final class CapabilityProtocol: URLProtocol, @unchecked Sendable {
    struct Fixture { var body: String; var status = 200; var paths: [String] = []; var policies: [String] = [] }
    static let lock = NSLock()
    nonisolated(unsafe) static var fixtures: [String: Fixture] = [:]
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let id = request.value(forHTTPHeaderField: "X-Fixture")!
        let path = request.url!.path
        Self.lock.lock()
        Self.fixtures[id]!.paths.append(path)
        Self.fixtures[id]!.policies.append(request.value(forHTTPHeaderField: "X-Voxt-Model-Policy") ?? "missing")
        let fixture = Self.fixtures[id]!
        Self.lock.unlock()
        let isCapabilities = path == "/v1/capabilities"
        let body = isCapabilities ? fixture.body : #"{"id":"fixture","text":"Hello","rawText":"Hello","mode":"dictate"}"#
        let response = HTTPURLResponse(url: request.url!, statusCode: isCapabilities ? fixture.status : 200, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
    static func make(_ body: String, status: Int = 200, allow: Bool = false) -> (VoxtClient, String) {
        let id = UUID().uuidString
        lock.lock(); fixtures[id] = Fixture(body: body, status: status); lock.unlock()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [Self.self]
        config.httpAdditionalHeaders = ["X-Fixture": id]
        return (VoxtClient(server: .init(baseURL: URL(string: "http://127.0.0.1:1234")!, token: "fixture"),
                           session: URLSession(configuration: config), allowCloudModels: { allow }), id)
    }
    static func policies(_ id: String) -> [String] {
        lock.lock(); defer { lock.unlock() }; return fixtures[id]!.policies
    }
    static func paths(_ id: String) -> [String] {
        lock.lock(); defer { lock.unlock() }; return fixtures.removeValue(forKey: id)!.paths
    }
}

@Suite struct LocalModelGateTests {
    @Test func localModelsPermitAudioAfterCapabilities() async throws {
        let (client, id) = CapabilityProtocol.make(#"{"transcription":"local","text":"local"}"#)
        _ = try await client.compose(audio: Data([1]), contentType: "audio/wav", options: .init())
        #expect(CapabilityProtocol.policies(id).last == "local-only")
        #expect(CapabilityProtocol.paths(id) == ["/v1/capabilities", "/v1/compose"])
    }
    @Test func cloudOrUnknownModelBlocksAudioIncludingLiteral() async throws {
        for (asr, text) in [("remote", "local"), ("local", "remote"), ("unknown", "local"), ("local", "unknown")] {
            for literal in [false, true] {
                let (client, id) = CapabilityProtocol.make("{\"transcription\":\"\(asr)\",\"text\":\"\(text)\"}")
                do {
                    _ = try await client.compose(audio: Data([1]), contentType: "audio/wav", options: .init(literalDictation: literal))
                    Issue.record("Audio must not be sent")
                } catch {}
                #expect(CapabilityProtocol.paths(id) == ["/v1/capabilities"])
            }
        }
    }
    @Test func missingOrMalformedCapabilitiesFailClosed() async throws {
        for status in [200, 404, 500] {
            let (client, id) = CapabilityProtocol.make("{}", status: status)
            do { _ = try await client.dictate(audio: Data([1]), contentType: "audio/wav"); Issue.record("Audio sent") }
            catch {}
            #expect(CapabilityProtocol.paths(id) == ["/v1/capabilities"])
        }
    }
    @Test func explicitCloudConsentPermitsAudio() async throws {
        let (client, id) = CapabilityProtocol.make(#"{"transcription":"remote","text":"remote"}"#, allow: true)
        _ = try await client.dictate(audio: Data([1]), contentType: "audio/wav")
        #expect(CapabilityProtocol.paths(id).last == "/v1/dictations")
    }
}
