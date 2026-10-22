import Foundation
import Testing
@testable import VoxtConnectionKit

@MainActor private final class AppStoreFixture: ConnectorConfigurationStore {
    var value = ConnectorConfiguration(allowedClients: [], state: nil)
    var saves = 0
    var denySave = false
    func load() throws -> ConnectorConfiguration { value }
    func save(_ value: ConnectorConfiguration) throws {
        if denySave { throw CocoaError(.fileWriteNoPermission) }
        saves += 1; self.value = value
    }
}

@MainActor @Test func uiModelUsesEmbeddedRuntimeAndPersistsItsState() async throws {
    let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let helper = folder.appending(path: "runtime.sh")
    try """
    #!/bin/sh
    read config
    printf '%s\\n' '{"version":1,"address":"tc-owned","port":48931,"state":"new-synthetic-state"}'
    cat >/dev/null
    """.write(to: helper, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: helper.path)
    let store = AppStoreFixture()
    store.value = .init(allowedClients: ["nodekey:" + String(repeating: "a", count: 64)], state: "old-state")
    let model = ConnectorAppModel(runtimeURL: helper, store: store, health: { port, token in
        #expect(port == 47920); #expect(token == "owned-fixture")
        return .init(ok: true, app: "Voxt", version: "fixture")
    })
    model.port = "47920"; model.token = "owned-fixture"
    await model.connect()
    #expect(model.status == .ready(version: "fixture", address: "tc-owned"))
    #expect(store.value.state == "new-synthetic-state"); #expect(store.saves == 1)
    model.disconnect(); #expect(model.status == .disconnected); #expect(model.token.isEmpty)
}

@MainActor @Test func uiModelUnpairedDoesNotLaunchMissingRuntime() async {
    let model = ConnectorAppModel(runtimeURL: URL(fileURLWithPath: "/missing-owned-runtime"), store: AppStoreFixture(), health: { _, _ in .init(ok: true, app: "Voxt", version: "fixture") })
    model.token = "owned-fixture"
    await model.connect()
    #expect(model.status == .localReady(version: "fixture"))
}

@MainActor @Test func uiModelRejectsInvalidConfigurationBeforeHealth() async {
    var calls = 0
    let model = ConnectorAppModel(runtimeURL: URL(fileURLWithPath: "/missing-owned-runtime"), store: AppStoreFixture(), health: { _, _ in calls += 1; throw LocalInferenceClient.Failure.unauthorized })
    model.port = "0"; model.token = "fixture"; await model.connect()
    #expect(calls == 0); #expect(model.configurationError != nil)
    model.port = "47920"; await model.connect()
    #expect(calls == 1); #expect(model.status == .failed(.unauthorized))
}

@MainActor @Test func fileConfigurationStoreRoundTripsPrivatelyAndRejectsUnsafeFiles() throws {
    let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    let url = folder.appending(path: "identity.json")
    let store = FileConnectorConfigurationStore(url: url)
    #expect(try store.load() == .init(allowedClients: [], state: nil))
    let value = ConnectorConfiguration(allowedClients: ["nodekey:" + String(repeating: "b", count: 64)], state: "fixture-state")
    try store.save(value)
    #expect(try store.load() == value)
    let mode = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? NSNumber
    #expect(mode?.intValue == 0o600)
    try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: url.path)
    #expect(throws: (any Error).self) { try store.load() }
    try FileManager.default.removeItem(at: url)
    try FileManager.default.createSymbolicLink(at: url, withDestinationURL: folder.appending(path: "missing"))
    #expect(throws: (any Error).self) { try store.load() }
}

@MainActor private final class PairRuntimeFixture: ConnectorPairingRuntime {
    var onEvent: ((ConnectorPairingProcess.Event) -> Void)?
    var starts = 0, pairs = 0, cancels = 0, stops = 0
    var receivedPort: Int?
    var receivedToken: String?
    func start(localPort: Int, token: String, configuration: ConnectorConfiguration) async throws {
        starts += 1; receivedPort = localPort; receivedToken = token
        onEvent?(.ready(address: ""))
    }
    func beginPairing() throws { pairs += 1 }
    func cancelPairing() throws { cancels += 1; onEvent?(.pairingClosed) }
    func stop() { stops += 1 }
}
@MainActor @Test func uiModelPairingUsesAuthenticatedConfigurationAndRetainsNormalSessionOnCancel() async throws {
    let runtime = PairRuntimeFixture()
    let model = ConnectorAppModel(runtimeURL: URL(fileURLWithPath: "/missing"), store: AppStoreFixture(),
        health: { _, _ in .init(ok: true, app: "Voxt", version: "fixture") }, makePairingRuntime: { runtime })
    model.port = "47920"; model.token = "verified-fixture"; await model.connect()
    model.port = "18080"; model.token = "unverified-edit"
    await model.beginPairing()
    #expect(runtime.starts == 1); #expect(runtime.pairs == 1)
    #expect(runtime.receivedPort == 47920); #expect(runtime.receivedToken == "verified-fixture")
    let expiry = Int(Date().timeIntervalSince1970) + 100
    let code = try ConnectorPairingCode(url: URL(string: "voxtmobile://connect?v=1&tailcat=tc-owned&port=48932&key=privkey:" + String(repeating: "a", count: 64) + "&proof=fixture&expires=\(expiry)")!)
    runtime.onEvent?(.pairing(code)); #expect(model.pairingCode == code)
    runtime.onEvent?(.paired(address: "tc-normal", client: "nodekey:" + String(repeating: "b", count: 64)))
    #expect(model.status == .ready(version: "fixture", address: "tc-normal"))
    model.cancelPairing(); #expect(runtime.cancels == 1); #expect(model.pairingCode == nil)
    #expect(model.status == .ready(version: "fixture", address: "tc-normal"))
    model.disconnect(); #expect(runtime.stops == 1); #expect(model.status == .disconnected)
    runtime.onEvent?(.paired(address: "tc-stale", client: "stale"))
    #expect(model.status == .disconnected)
}
