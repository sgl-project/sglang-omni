import XCTest
import VoxtMobileKit
@testable import VoxtMobile

@MainActor private final class TransportCounts {
    var configs: [ServerConfig] = []
    var closed = 0
    var resolutions = 0
    func make(_ config: ServerConfig) -> PairingConnection {
        configs.append(config)
        return PairingConnection(server: config, resolve: { [self] in
            await resolved()
            throw VoxtClient.Error.transportUnavailable
        }, close: { [self] in await didClose() })
    }
    func resolved() { resolutions += 1 }
    func didClose() { closed += 1 }
}
private final class TransportTokens: TokenStoring {
    var value: String?
    func read() throws -> String? { value }
    func write(_ value: String) throws { self.value = value }
    func delete() throws { value = nil }
}

@MainActor final class AppModelTransportTests: XCTestCase {
    func testTailcatNetworkErrorDoesNotAskForSystemVPN() {
        let failure = URLError(.cannotConnectToHost)
        XCTAssertTrue(AppModel.describe(failure).contains("Tailscale"))
        XCTAssertFalse(AppModel.describe(failure, tailcat: true).contains("Tailscale"))
        XCTAssertTrue(AppModel.describe(failure, tailcat: true).contains("Mac"))
    }

    func testSavedPairingUsesConnectionResolverAndUnpairClosesIt() async throws {
        let name = "transport-test.\(UUID())", counts = TransportCounts(), tokens = TransportTokens()
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let config = ServerConfig(baseURL: URL(string: "http://tailcat.invalid")!, token: "fixture", tailcat: .init(address: "fixture", clientKey: "fixture", remotePort: 1234))
        try PairingStore(defaults: defaults, tokens: tokens).save(config)
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens, injectedConnectionFactory: { counts.make($0) })
        await model.checkConnection(attempts: 1)
        XCTAssertEqual(counts.configs, [config])
        XCTAssertEqual(counts.resolutions, 2, "health retries an unavailable transport once")
        model.unpair()
        for _ in 0..<100 { await Task.yield() }
        XCTAssertEqual(counts.closed, 1)
        XCTAssertEqual(model.connection, .unpaired)
    }
}
