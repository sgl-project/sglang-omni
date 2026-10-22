import XCTest
import VoxtMobileKit
import VoxtTailcat
@testable import VoxtMobile

private final class NativePairingTokens: TokenStoring {
    var value: String?
    func read() throws -> String? { value }
    func write(_ value: String) throws { self.value = value }
    func delete() throws { value = nil }
}

@MainActor final class AppModelNativeTailcatTests: XCTestCase {
    func testConnectorQRHandledByAppModelPersistsAndDictates() async throws {
        guard let base = ProcessInfo.processInfo.environment["VOXT_CONNECTOR_FIXTURE_URL"] else {
            throw XCTSkip("Owned connector fixture required")
        }
        let (data, _) = try await URLSession.shared.data(from: XCTUnwrap(URL(string: base + "/fixture/qr")))
        let fields = try JSONDecoder().decode([String: String].self, from: data)
        let url = try XCTUnwrap(URL(string: XCTUnwrap(fields["url"])))
        let name = "connector-app.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let tokens = NativePairingTokens()
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens)
        model.handle(url: url)
        let deadline = Date().addingTimeInterval(35)
        while Date() < deadline {
            if case .connected = model.connection { break }
            if case .failed = model.connection { break }
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertEqual(model.connection, .connected(version: "connector-fixture"))
        XCTAssertNotNil(try PairingStore(defaults: defaults, tokens: tokens).load())
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(model.latestText, "One scan dictation works.")
        model.unpair()
    }

    func testConnectorNativePairingIsolationControls() async throws {
        guard let base = ProcessInfo.processInfo.environment["VOXT_CONNECTOR_FIXTURE_URL"] else {
            throw XCTSkip("Owned connector fixture required")
        }
        let (data, _) = try await URLSession.shared.data(from: XCTUnwrap(URL(string: base + "/fixture/qr")))
        let fields = try JSONDecoder().decode([String: String].self, from: data)
        let url = try XCTUnwrap(URL(string: XCTUnwrap(fields["url"])))
        let bootstrap = try PairingBootstrap(url: url)
        XCTAssertThrowsError(try PairingBootstrap(url: url, now: bootstrap.expiresAt))
        let phone = try TailcatIdentity.generate()
        let other = try TailcatIdentity.generate()
        let bridge = TailcatBridge()
        let endpoint = try await bridge.endpoint(for: .init(address: bootstrap.address, clientKey: bootstrap.temporaryKey, remotePort: bootstrap.port))
        do {
            _ = try await PairingExchange.exchange(endpoint: endpoint, proof: "wrong-proof", publicKey: phone.publicKey, privateKey: phone.privateKey)
            XCTFail("Wrong proof authorized")
        } catch PairingExchange.Failure.rejected(let status) { XCTAssertEqual(status, 401) }
        var forbidden = URLRequest(url: endpoint.appendingPathComponent("v1/compose"))
        forbidden.httpMethod = "POST"
        forbidden.timeoutInterval = 10
        forbidden.setValue("Bearer " + bootstrap.proof, forHTTPHeaderField: "Authorization")
        let (_, response) = try await URLSession.shared.data(for: forbidden)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 403)
        let config = try await PairingExchange.exchange(endpoint: endpoint, proof: bootstrap.proof, publicKey: phone.publicKey, privateKey: phone.privateKey)
        XCTAssertEqual(config.tailcat?.clientKey, phone.privateKey)
        do {
            _ = try await PairingExchange.exchange(endpoint: endpoint, proof: bootstrap.proof, publicKey: other.publicKey, privateKey: other.privateKey)
            XCTFail("Second phone authorized")
        } catch PairingExchange.Failure.rejected(let status) { XCTAssertEqual(status, 409) }
        await bridge.invalidate()
        let wrongBridge = TailcatBridge()
        let wrongEndpoint = try await wrongBridge.endpoint(for: .init(address: bootstrap.address, clientKey: other.privateKey, remotePort: bootstrap.port))
        var wrongRequest = URLRequest(url: wrongEndpoint.appendingPathComponent("pair"))
        wrongRequest.httpMethod = "POST"
        wrongRequest.timeoutInterval = 3
        do {
            _ = try await URLSession.shared.data(for: wrongRequest)
            XCTFail("Wrong temporary key reached pairing endpoint")
        } catch { XCTAssertTrue(error is URLError) }
        await wrongBridge.invalidate()
    }

    func testNativeTailcatRecoversAfterHostRestartWithoutAppRelaunch() async throws {
        guard let base = ProcessInfo.processInfo.environment["VOXT_RECOVERY_FIXTURE_URL"],
              let token = ProcessInfo.processInfo.environment["VOXT_RECOVERY_CONTROL_TOKEN"] else {
            throw XCTSkip("Owned recovery fixture required")
        }
        func control(_ path: String, body: [String: String] = [:]) async throws -> Data {
            var request = URLRequest(url: URL(string: base + path)!)
            request.httpMethod = "POST"
            request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
            request.httpBody = try JSONEncoder().encode(body)
            request.timeoutInterval = 30
            let (data, response) = try await URLSession.shared.data(for: request)
            XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
            return data
        }
        let identity = try TailcatIdentity.generate()
        struct Pair: Decodable { let address: String; let port: UInt16; let token: String }
        let pair = try JSONDecoder().decode(Pair.self, from: await control("/pair", body: ["publicKey": identity.publicKey]))
        let name = "native-recovery.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let tokens = NativePairingTokens()
        let config = ServerConfig(baseURL: URL(string: "http://tailcat.invalid")!, token: pair.token,
                                  tailcat: .init(address: pair.address, clientKey: identity.privateKey, remotePort: pair.port))
        try PairingStore(defaults: defaults, tokens: tokens).save(config)
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens)
        await model.checkConnection(attempts: 1)
        XCTAssertEqual(model.connection, .connected(version: "recovery-fixture"))
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(model.phase, .done)
        _ = try await control("/restart")
        // The same AppModel, native bridge and process stay alive across the helper restart.
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(model.phase, .done)
        XCTAssertEqual(model.latestText, "Recovered fixture dictation.")
        _ = try await control("/drop-compose")
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(model.phase, .failed)
        guard case .failed = model.connection else { return XCTFail("Stale Connected after real TCP failure") }
        XCTAssertTrue(model.canRetry)
        await model.controller.retry()
        XCTAssertEqual(model.phase, .done)
        _ = try await control("/assert-counts")
        model.unpair()
    }

    func testSavedTailcatPairingConnectsThroughNativeLibraryAndUnpairs() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let control = env["VOXT_TAILCAT_FIXTURE_URL"],
              let secret = env["VOXT_FIXTURE_CONTROL_TOKEN"] else { throw XCTSkip("Owned Tailcat fixture required") }
        let identity = try TailcatIdentity.generate()
        var request = URLRequest(url: URL(string: control)!.appendingPathComponent("pair"))
        request.httpMethod = "POST"
        request.setValue("Bearer " + secret, forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONEncoder().encode(["publicKey": identity.publicKey])
        request.timeoutInterval = 30
        let (data, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        struct Invitation: Decodable { let address: String; let port: UInt16; let token: String }
        let invite = try JSONDecoder().decode(Invitation.self, from: data)
        let name = "native-tailcat-test.\(UUID())", tokens = NativePairingTokens()
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let config = ServerConfig(baseURL: URL(string: "http://tailcat.invalid")!, token: invite.token,
                                  tailcat: .init(address: invite.address, clientKey: identity.privateKey, remotePort: invite.port))
        try PairingStore(defaults: defaults, tokens: tokens).save(config)
        let capture: any AudioCapturing = env["VOXT_NATIVE_WAV"].map { FileAudioCapture(url: URL(fileURLWithPath: $0)) } ?? ScriptedCapturePublic()
        let model = AppModel(injectedCapture: capture, injectedDefaults: defaults, injectedTokens: tokens)
        await model.checkConnection(attempts: 1)
        XCTAssertEqual(model.connection, .connected(version: env["VOXT_NATIVE_VERSION"] ?? "native-fixture"))
        if env["VOXT_NATIVE_WAV"] != nil {
            await model.toggleRecording()
            XCTAssertEqual(model.phase, .recording)
            await model.toggleRecording()
            XCTAssertEqual(model.phase, .done)
            XCTAssertTrue(model.latestText?.contains("明天") == true)
        }
        model.unpair()
        XCTAssertNil(model.server)
        XCTAssertEqual(model.connection, .unpaired)
    }
}
