import Foundation
import Testing
@testable import VoxtMobileKit

struct PairingExchangeTests {
    private func session(status: Int, invitation: String) -> (URLSession, String) {
        let id = UUID().uuidString
        let data = try! JSONEncoder().encode(["invitation": invitation])
        StubURLProtocol.lock.lock()
        StubURLProtocol.stubs[id] = .init(status: status, body: data)
        StubURLProtocol.lock.unlock()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        config.httpAdditionalHeaders = ["X-Test-Session": id]
        return (URLSession(configuration: config), id)
    }
    @Test func exchangesProofForOnlyThisPhonesAuthorization() async throws {
        let key = "nodekey:" + String(repeating: "a", count: 64)
        let (session, id) = session(status: 200, invitation: "voxtmobile://pair?tailcat=tc-fixture&port=48931&token=synthetic&client=" + key)
        let server = try await PairingExchange.exchange(endpoint: URL(string: "http://127.0.0.1:12345")!, proof: "fixture-proof", publicKey: key, privateKey: "fixture-private", session: session)
        #expect(server.tailcat?.address == "tc-fixture")
        #expect(server.tailcat?.clientKey == "fixture-private")
        let (request, body) = try #require(StubURLProtocol.lastRequest(id: id))
        #expect(request.url?.path == "/pair")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-proof")
        #expect(try JSONDecoder().decode([String:String].self, from: #require(body)) == ["phoneKey":key])
    }
    @Test func refusesWrongExpiredSecondPhoneAndMismatchedResponse() async throws {
        let key = "nodekey:" + String(repeating: "a", count: 64)
        for status in [401, 403, 409, 410, 503, 200] {
            let (session, _) = session(status: status, invitation: "voxtmobile://pair?tailcat=tc-fixture&port=48931&token=synthetic&client=nodekey:" + String(repeating: "b", count: 64))
            await #expect(throws: (any Error).self) {
                try await PairingExchange.exchange(endpoint: URL(string: "http://127.0.0.1:12345")!, proof: "fixture-proof", publicKey: key, privateKey: "fixture-private", session: session)
            }
        }
    }
}
