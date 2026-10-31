import XCTest
import VoxtMobileKit
@testable import VoxtMobile

private final class ConnectionTokens: TokenStoring {
    var value: String?
    func read() throws -> String? { value }
    func write(_ value: String) throws { self.value = value }
    func delete() throws { value = nil }
}

@MainActor
private final class PendingHealth {
    private var response: CheckedContinuation<VoxtClient.Health, any Error>?
    private var started: CheckedContinuation<Void, Never>?
    func fetch() async throws -> VoxtClient.Health {
        try await withCheckedThrowingContinuation { response in
            self.response = response
            started?.resume(); started = nil
        }
    }
    func waitUntilStarted() async {
        if response != nil { return }
        await withCheckedContinuation { started = $0 }
    }
    func finish(_ result: Result<VoxtClient.Health, any Error>) { response?.resume(with: result); response = nil }
}

@MainActor
final class AppModelConnectionTests: XCTestCase {
    private func health(_ version: String) throws -> VoxtClient.Health {
        try JSONDecoder().decode(VoxtClient.Health.self, from: Data("{\"ok\":true,\"app\":\"Voxt\",\"version\":\"\(version)\"}".utf8))
    }
    private func makeModel(defaults: UserDefaults, probe: @escaping @MainActor (ServerConfig) async throws -> VoxtClient.Health) throws -> AppModel {
        let tokens = ConnectionTokens()
        try PairingStore(defaults: defaults, tokens: tokens).save(.init(baseURL: URL(string: "https://old.example.test")!, token: "fake-token"))
        return AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens, injectedHealthProbe: probe)
    }

    func testDictationConnectionFailureClearsConnectedAndKeepsRetry() async throws {
        let name = "connection-failure-test.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let tokens = ConnectionTokens()
        let config = ServerConfig(baseURL: URL(string: "http://127.0.0.1:9")!, token: "fixture",
                                  tailcat: .init(address: "fixture", clientKey: "fixture", remotePort: 48931))
        try PairingStore(defaults: defaults, tokens: tokens).save(config)
        let good = try health("before-restart")
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults,
                             injectedTokens: tokens, injectedHealthProbe: { _ in good },
                             injectedConnectionFactory: { config in
            PairingConnection(server: config, resolve: { URL(string: "http://127.0.0.1:9")! })
        })
        await model.checkConnection(attempts: 1)
        XCTAssertEqual(model.connection, .connected(version: "before-restart"))
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(model.phase, .failed)
        guard case .failed = model.connection else { return XCTFail("Home kept stale Connected after request failure") }
        XCTAssertTrue(model.errorMessage?.contains("Mac") == true)
        XCTAssertTrue(model.canRetry)
        XCTAssertEqual(DictationChannel(defaults: defaults).state.errorMessage, model.errorMessage)
    }

    /// The keyboard opens voxtmobile://reconnect after the tunnel failed in the background.
    /// In front, the app checks the connection, then sends the same recording again.
    func testReconnectRetriesTheRecordingOnlyOnceTheMacAnswers() async throws {
        let name = "connection-reconnect-test.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let tokens = ConnectionTokens()
        let config = ServerConfig(baseURL: URL(string: "http://127.0.0.1:9")!, token: "fixture",
                                  tailcat: .init(address: "fixture", clientKey: "fixture", remotePort: 48931))
        try PairingStore(defaults: defaults, tokens: tokens).save(config)
        let good = try health("reconnected")
        let mac = MacAnswers()
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults,
                             injectedTokens: tokens, injectedHealthProbe: { _ in
            guard mac.answers else { throw URLError(.cannotConnectToHost) }
            return good
        }, injectedConnectionFactory: { config in
            PairingConnection(server: config, resolve: { mac.resolved(); return URL(string: "http://127.0.0.1:9")! })
        })
        await model.checkConnection(attempts: 1)
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(model.phase, .failed)
        XCTAssertTrue(model.canRetry)
        let before = mac.resolves
        mac.answers = false
        await model.reconnectAndRetry(attempts: 1)
        XCTAssertEqual(mac.resolves, before, "no resend while the Mac doesn't answer")
        mac.answers = true
        await model.reconnectAndRetry(attempts: 1)
        XCTAssertGreaterThan(mac.resolves, before, "connected: the recording is sent again")
    }

    func testReplacementImmediatelyStopsShowingPreviousConnectedVersion() async throws {
        let name = "connection-test.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let version = try health("existing")
        let model = try makeModel(defaults: defaults) { _ in version }
        await model.checkConnection(attempts: 1)
        try model.save(.init(baseURL: URL(string: "https://next.example.test")!, token: "fixture"))
        XCTAssertEqual(model.connection, .checking)
    }

    func testUnpairDiscardsBothLateSuccessAndLateError() async throws {
        for succeed in [true, false] {
            let name = "connection-test.\(UUID())", pending = PendingHealth()
            let defaults = UserDefaults(suiteName: name)!
            defer { defaults.removePersistentDomain(forName: name) }
            let model = try makeModel(defaults: defaults) { _ in try await pending.fetch() }
            let operation = Task { await model.checkConnection(attempts: 1) }
            await pending.waitUntilStarted()
            model.unpair()
            if succeed { pending.finish(.success(try health("old"))) }
            else { pending.finish(.failure(VoxtClient.Error.unauthorized)) }
            await operation.value
            XCTAssertNil(model.server)
            XCTAssertEqual(model.connection, .unpaired)
        }
    }

    func testOldResponseCannotReplaceNewPairingHealth() async throws {
        let name = "connection-test.\(UUID())", pending = PendingHealth()
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let fresh = try health("new")
        let model = try makeModel(defaults: defaults) { config in
            if config.baseURL.host == "new.example.test" { return fresh }
            return try await pending.fetch()
        }
        let old = Task { await model.checkConnection(attempts: 1) }
        await pending.waitUntilStarted()
        try model.save(.init(baseURL: URL(string: "https://new.example.test")!, token: "fake-new-token"))
        await model.checkConnection(attempts: 1)
        pending.finish(.success(try health("old")))
        await old.value
        XCTAssertEqual(model.connection, .connected(version: "new"))
    }

    /// A dictation refused because the Mac uses a cloud model is shown on the status card as
    /// its own state (not "can't reach"), and the next result clears it.
    func testCloudModelFailureBecomesItsOwnStateAndClearsOnTheNextResult() {
        typealias State = AppModel.ConnectionState
        let connected = State.connected(version: "1.16.0")
        XCTAssertEqual(State.after(phase: .failed, failure: VoxtClient.Error.cloudModelSelected, current: connected),
                       .cloudModel(version: "1.16.0"))
        XCTAssertEqual(State.after(phase: .failed, failure: VoxtClient.Error.editNotApplied, current: connected), connected)
        XCTAssertEqual(State.after(phase: .done, failure: nil, current: .cloudModel(version: "1.16.0")), connected)
        XCTAssertEqual(State.after(phase: .failed, failure: VoxtClient.Error.cloudModelSelected, current: .checking), .checking)
        XCTAssertEqual(AppModel.statusCardState(for: .cloudModel(version: "1.16.0")), .cloudModel(version: "1.16.0"))
        XCTAssertEqual(AppModel.statusCardState(for: .failed("Couldn't reach your Mac.")), .unreachable(reason: "Couldn't reach your Mac."))
        XCTAssertEqual(AppModel.statusCardState(for: .checking), .connecting)
        XCTAssertNil(AppModel.statusCardState(for: .unpaired))
    }
}

/// The fake Mac for the reconnect test: whether it answers, and how often a request looked it up.
private final class MacAnswers: @unchecked Sendable {
    private let lock = NSLock()
    private var answering = true
    private var count = 0
    var answers: Bool {
        get { lock.withLock { answering } }
        set { lock.withLock { answering = newValue } }
    }
    var resolves: Int { lock.withLock { count } }
    func resolved() { lock.withLock { count += 1 } }
}
