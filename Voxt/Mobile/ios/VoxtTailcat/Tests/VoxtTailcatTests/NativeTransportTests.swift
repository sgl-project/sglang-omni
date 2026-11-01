import Foundation
import Testing
@testable import VoxtTailcat

private struct FixturePair: Decodable {
    let address: String
    let port: UInt16
    let token: String
}

@Suite(.serialized) struct NativeTransportTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["VOXT_TAILCAT_FIXTURE_URL"] != nil))
    func nativeTunnelPreservesHTTPAndClosesThenReconnects() async throws {
        let env = ProcessInfo.processInfo.environment
        let controlURL = try #require(URL(string: env["VOXT_TAILCAT_FIXTURE_URL"] ?? ""))
        let controlToken = try #require(env["VOXT_FIXTURE_CONTROL_TOKEN"])
        let identity = try TailcatIdentity.generate()
        let session = URLSession(configuration: .ephemeral)
        defer { session.invalidateAndCancel() }
        var pairRequest = URLRequest(url: controlURL.appendingPathComponent("pair"))
        pairRequest.httpMethod = "POST"
        pairRequest.setValue("Bearer " + controlToken, forHTTPHeaderField: "Authorization")
        pairRequest.httpBody = try JSONEncoder().encode(["publicKey": identity.publicKey])
        pairRequest.timeoutInterval = 30
        let (pairData, pairResponse) = try await session.data(for: pairRequest)
        #expect((pairResponse as? HTTPURLResponse)?.statusCode == 200)
        let pair = try JSONDecoder().decode(FixturePair.self, from: pairData)
        let options = TailcatOptions(address: pair.address, clientKey: identity.privateKey, remotePort: pair.port)
        let bridge = TailcatBridge()
        do {
            let base = try await bridge.endpoint(for: options)
            let valid = try await request(session, base, "/v1/health", pair.token)
            #expect(valid.0 == 200)
            let health = try JSONDecoder().decode(FixtureHealth.self, from: valid.1)
            #expect(health.version == "native-fixture")
            #expect(try await request(session, base, "/v1/health", "wrong-bearer").0 == 401)
            let payload = Data(String(repeating: "swift-native-fixture", count: 16384).utf8)
            #expect(try await request(session, base, "/v1/dictations", pair.token, payload).0 == 200)
            #expect(try await request(session, base, "/v1/dictations", pair.token, Data([1, 2, 3])).0 == 400)
            let pending = Task { try await request(session, base, "/v1/slow", pair.token) }
            var reachedServer = false
            for _ in 0..<30 {
                if try await request(session, controlURL, "/slow-started", pair.token).0 == 200 {
                    reachedServer = true
                    break
                }
                try await Task.sleep(for: .milliseconds(100))
            }
            #expect(reachedServer)
            let cancelledAt = ContinuousClock.now
            pending.cancel()
            do {
                _ = try await pending.value
                Issue.record("Cancelled live request returned success")
            } catch is CancellationError {
            } catch let error as URLError {
                #expect(error.code == .cancelled)
            }
            #expect(cancelledAt.duration(to: .now) < .seconds(2))
            #expect(try await request(session, base, "/v1/health", pair.token).0 == 200)
            await bridge.close()
            do {
                _ = try await request(session, base, "/v1/health", pair.token)
                Issue.record("Closed bridge accepted an HTTP request")
            } catch is URLError { }
            let reconnected = try await bridge.endpoint(for: options)
            #expect(try await request(session, reconnected, "/v1/health", pair.token).0 == 200)
            await bridge.invalidate()
            await #expect(throws: CancellationError.self) { try await bridge.endpoint(for: options) }
        } catch {
            await bridge.invalidate()
            throw error
        }
    }

    private func request(_ session: URLSession, _ base: URL, _ path: String, _ bearer: String, _ body: Data? = nil) async throws -> (Int, Data) {
        var request = URLRequest(url: base.appendingPathComponent(path))
        request.timeoutInterval = 15
        request.setValue("Bearer " + bearer, forHTTPHeaderField: "Authorization")
        if let body { request.httpMethod = "POST"; request.httpBody = body }
        let (data, response) = try await session.data(for: request)
        return ((response as? HTTPURLResponse)?.statusCode ?? -1, data)
    }
}

private struct FixtureHealth: Decodable { let version: String }
