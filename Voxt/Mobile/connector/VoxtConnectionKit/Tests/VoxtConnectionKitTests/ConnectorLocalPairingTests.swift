import Foundation
import Testing
@testable import VoxtConnectionKit

@MainActor private final class EmptyPhoneStore: ConnectorConfigurationStore {
    func load() throws -> ConnectorConfiguration { .init(allowedClients: [], state: nil) }
    func save(_ value: ConnectorConfiguration) throws {}
}

@MainActor private final class EventRuntime: ConnectorPairingRuntime {
    var onEvent: ((ConnectorPairingProcess.Event) -> Void)?
    func start(localPort: Int, token: String, configuration: ConnectorConfiguration) async throws { onEvent?(.ready(address: "")) }
    func beginPairing() throws {}
    func cancelPairing() throws { onEvent?(.pairingClosed) }
    func stop() {}
}

@MainActor private final class FakeLocalPairing: ConnectorLocalPairing {
    var stops = 0
    func stop() { stops += 1 }
}

@MainActor private final class LocalPairingFactory {
    var started: [(URL, Date)] = []
    var onRequest: (@Sendable (String, String) async -> Bool)?
    var current: FakeLocalPairing?
    var fails = false
    func make(_ invitation: URL, _ expiresAt: Date, _ onRequest: @escaping @Sendable (String, String) async -> Bool) throws -> any ConnectorLocalPairing {
        if fails { throw CocoaError(.featureUnsupported) }
        started.append((invitation, expiresAt)); self.onRequest = onRequest
        let pairing = FakeLocalPairing(); current = pairing; return pairing
    }
}

@MainActor private func pairingModel(_ factory: LocalPairingFactory, runtime: EventRuntime) async -> ConnectorAppModel {
    let model = ConnectorAppModel(runtimeURL: URL(fileURLWithPath: "/missing"), store: EmptyPhoneStore(),
        health: { _, _ in .init(ok: true, app: "Voxt", version: "1.16") }, makePairingRuntime: { runtime },
        preferences: TestPreferences(), makeLocalPairing: { try factory.make($0, $1, $2) })
    model.token = "fixture"
    await model.connect()
    await model.beginPairing()
    return model
}

private func invitation() throws -> ConnectorPairingCode {
    let expiry = Int(Date().timeIntervalSince1970) + 100
    return try ConnectorPairingCode(url: URL(string: "voxtmobile://connect?v=1&tailcat=tc-owned&port=48932&key=privkey:"
        + String(repeating: "a", count: 64) + "&proof=fixture&expires=\(expiry)")!)
}

@MainActor private func eventually(_ condition: () -> Bool) async -> Bool {
    for _ in 0..<10_000 where !condition() { await Task.yield() }
    #expect(condition(), "condition never became true")
    return condition()
}

@MainActor @Test func theQRCodeAlsoStartsSameWiFiPairingWithTheSameInvitation() async throws {
    let factory = LocalPairingFactory(), runtime = EventRuntime()
    let model = await pairingModel(factory, runtime: runtime)
    let code = try invitation()
    runtime.onEvent?(.pairing(code))
    #expect(model.pairingCode == code)
    #expect(factory.started.count == 1)
    #expect(factory.started.first?.0 == code.url); #expect(factory.started.first?.1 == code.expiresAt)
}

@MainActor @Test(arguments: [true, false])
func aPhoneRequestShowsTheCodeAndTheUsersAnswer(allow: Bool) async throws {
    let factory = LocalPairingFactory(), runtime = EventRuntime()
    let model = await pairingModel(factory, runtime: runtime)
    runtime.onEvent?(.pairing(try invitation()))
    let ask = try #require(factory.onRequest)
    let answer = Task { await ask("042917", "Test iPhone") }
    guard await eventually({ model.phoneRequest != nil }) else { answer.cancel(); return }
    #expect(model.phoneRequest?.code == "042 917"); #expect(model.phoneRequest?.deviceName == "Test iPhone")
    model.answerPhoneRequest(allow)
    #expect(await answer.value == allow)
    #expect(model.phoneRequest == nil)
}

@MainActor @Test func aSecondPhoneIsDeniedWhileOneIsAsking() async throws {
    let factory = LocalPairingFactory(), runtime = EventRuntime()
    let model = await pairingModel(factory, runtime: runtime)
    runtime.onEvent?(.pairing(try invitation()))
    let ask = try #require(factory.onRequest)
    let first = Task { await ask("111111", "First iPhone") }
    guard await eventually({ model.phoneRequest != nil }) else { first.cancel(); return }
    #expect(await ask("222222", "Second iPhone") == false)
    #expect(model.phoneRequest?.deviceName == "First iPhone")
    model.answerPhoneRequest(true)
    #expect(await first.value)
}

@MainActor @Test func cancellingAddIPhoneStopsAdvertisingAndDeniesThePendingPhone() async throws {
    let factory = LocalPairingFactory(), runtime = EventRuntime()
    let model = await pairingModel(factory, runtime: runtime)
    runtime.onEvent?(.pairing(try invitation()))
    let ask = try #require(factory.onRequest)
    let pending = Task { await ask("333333", "Test iPhone") }
    guard await eventually({ model.phoneRequest != nil }) else { pending.cancel(); return }
    model.cancelPairing()
    #expect(factory.current?.stops == 1)
    #expect(await pending.value == false)
    #expect(model.phoneRequest == nil)
}

@MainActor @Test func aPairedPhoneStopsAdvertising() async throws {
    let factory = LocalPairingFactory(), runtime = EventRuntime()
    let model = await pairingModel(factory, runtime: runtime)
    runtime.onEvent?(.pairing(try invitation()))
    runtime.onEvent?(.paired(address: "tc-normal", client: "nodekey:" + String(repeating: "b", count: 64)))
    #expect(factory.current?.stops == 1)
    #expect(model.pairingCode == nil)
}

@MainActor @Test func whenTheCoordinatorGivesUpTheSheetCloses() async throws {
    let factory = LocalPairingFactory(), runtime = EventRuntime()
    let model = await pairingModel(factory, runtime: runtime)
    runtime.onEvent?(.pairing(try invitation()))
    let ask = try #require(factory.onRequest)
    let pending = Task { await ask("444444", "Test iPhone") }
    guard await eventually({ model.phoneRequest != nil }) else { pending.cancel(); return }
    pending.cancel()
    #expect(await pending.value == false)
    #expect(await eventually({ model.phoneRequest == nil }))
}

@MainActor @Test func ifSameWiFiPairingCannotStartTheQRCodeStillWorks() async throws {
    let factory = LocalPairingFactory(), runtime = EventRuntime()
    factory.fails = true
    let model = await pairingModel(factory, runtime: runtime)
    let code = try invitation()
    runtime.onEvent?(.pairing(code))
    #expect(model.pairingCode == code)
    #expect(model.pairingError == nil)
}

/// The Allow sheet sits on the Settings window, so that window has to come forward when a phone asks.
@MainActor @Test func aPhoneRequestBringsSettingsForwardOnce() async throws {
    let factory = LocalPairingFactory(), runtime = EventRuntime()
    let model = await pairingModel(factory, runtime: runtime)
    var shown = 0
    model.onPhoneRequest = { shown += 1 }
    runtime.onEvent?(.pairing(try invitation()))
    let ask = try #require(factory.onRequest)
    let first = Task { await ask("111111", "First iPhone") }
    guard await eventually({ model.phoneRequest != nil }) else { first.cancel(); return }
    #expect(shown == 1)
    #expect(await ask("222222", "Second iPhone") == false)
    #expect(shown == 1)
    model.answerPhoneRequest(false)
    _ = await first.value
}
