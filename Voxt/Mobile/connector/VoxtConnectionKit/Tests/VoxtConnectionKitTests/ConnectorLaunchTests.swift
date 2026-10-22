import Foundation
import Testing
@testable import VoxtConnectionKit

@MainActor private final class NoPhonesStore: ConnectorConfigurationStore {
    func load() throws -> ConnectorConfiguration { .init(allowedClients: [], state: nil) }
    func save(_ value: ConnectorConfiguration) throws {}
}

private final class ScriptedAuthorizer: ConnectorAuthorizing, @unchecked Sendable {
    private let lock = NSLock()
    private var results: [Result<ConnectorGrant, ConnectorLink.Failure>]
    private var _calls = 0
    init(_ results: [Result<ConnectorGrant, ConnectorLink.Failure>]) { self.results = results }
    var calls: Int { lock.withLock { _calls } }
    func authorize() async throws -> ConnectorGrant {
        let next = lock.withLock { () -> Result<ConnectorGrant, ConnectorLink.Failure> in
            _calls += 1
            return results.isEmpty ? .failure(.voxtNotRunning) : results.removeFirst()
        }
        return try next.get()
    }
}

@MainActor private final class FakeLoginItem: ConnectorLoginItem {
    private var _status: ConnectorLoginItemStatus = .disabled
    /// Synchronous reads of the system status, which block the main thread in the real app.
    var statusReads = 0
    var status: ConnectorLoginItemStatus {
        get { statusReads += 1; return _status }
        set { _status = newValue }
    }
    func currentStatus() async -> ConnectorLoginItemStatus { _status }
    var registerError: (any Error)?
    var registers = 0, unregisters = 0
    func register() throws {
        registers += 1
        if let registerError { throw registerError }
        status = .enabled
    }
    func unregister() throws { unregisters += 1; status = .disabled }
}

/// Settings redraws read these often; asking SMAppService each time blocked the window from showing.
@MainActor @Test func readingOpenAtLoginNeverAsksTheSystemSynchronously() async {
    let loginItem = FakeLoginItem(); loginItem.status = .requiresApproval
    let model = launchModel(ScriptedAuthorizer([]), loginItem: loginItem)
    for _ in 0..<3 { _ = model.opensAtLogin; _ = model.loginItemMessage }
    #expect(loginItem.statusReads == 0)
    await model.refreshLoginItem()
    #expect(loginItem.statusReads == 0)
    #expect(model.loginItemMessage == "Allow Voxt Connector in System Settings → General → Login Items.")
}

private let approved: Result<ConnectorGrant, ConnectorLink.Failure> = .success(ConnectorGrant(port: 47920, token: "approved-fixture", alwaysAllowed: true))
private let allowedOnce: Result<ConnectorGrant, ConnectorLink.Failure> = .success(ConnectorGrant(port: 47920, token: "approved-fixture", alwaysAllowed: false))

@MainActor private func launchModel(_ authorizer: ScriptedAuthorizer, preferences: TestPreferences = TestPreferences(),
                                    loginItem: FakeLoginItem = FakeLoginItem(), healthy: Bool = true,
                                    sleeps: Recorder = Recorder()) -> ConnectorAppModel {
    ConnectorAppModel(runtimeURL: URL(fileURLWithPath: "/missing"), store: NoPhonesStore(), health: { _, _ in
        guard healthy else { throw LocalInferenceClient.Failure.unavailable }
        return .init(ok: true, app: "Voxt", version: "1.16")
    }, authorizer: authorizer, preferences: preferences, loginItem: loginItem,
       sleep: { duration in await MainActor.run { sleeps.values.append(duration) } })
}

@MainActor private final class Recorder { var values: [Duration] = [] }

@MainActor @Test func beforeAnyApprovalLaunchOpensSettingsAndNeverAsksVoxt() async {
    let authorizer = ScriptedAuthorizer([approved])
    let model = launchModel(authorizer)
    #expect(model.opensSettingsAtLaunch)
    #expect(await model.connectAtLaunch() == false)
    #expect(authorizer.calls == 0)
}

@MainActor @Test func onlyASuccessfulAutomaticConnectIsRemembered() async {
    let preferences = TestPreferences()
    let manual = launchModel(ScriptedAuthorizer([]), preferences: preferences)
    manual.token = "typed-fixture"
    await manual.connect()
    #expect(manual.isConnected); #expect(!preferences.wasApproved)

    let unhealthy = launchModel(ScriptedAuthorizer([approved]), preferences: preferences, healthy: false)
    await unhealthy.connect()
    #expect(!preferences.wasApproved)

    // Allow Once would bring Voxt's dialog back at the next launch, so it isn't remembered.
    let once = launchModel(ScriptedAuthorizer([allowedOnce]), preferences: preferences)
    await once.connect()
    #expect(once.isConnected); #expect(!preferences.wasApproved)

    let automatic = launchModel(ScriptedAuthorizer([approved]), preferences: preferences)
    await automatic.connect()
    #expect(automatic.isConnected); #expect(preferences.wasApproved)
}

@MainActor @Test func allowOnceAtLaunchForgetsTheApproval() async {
    let preferences = TestPreferences(); preferences.wasApproved = true
    let model = launchModel(ScriptedAuthorizer([allowedOnce]), preferences: preferences)
    #expect(await model.connectAtLaunch())
    #expect(!preferences.wasApproved)
}

@MainActor @Test func afterApprovalLaunchConnectsWithoutOpeningSettings() async {
    let preferences = TestPreferences(); preferences.wasApproved = true
    let authorizer = ScriptedAuthorizer([approved])
    let model = launchModel(authorizer, preferences: preferences)
    #expect(!model.opensSettingsAtLaunch)
    #expect(await model.connectAtLaunch())
    #expect(authorizer.calls == 1)
    #expect(model.connectFeedback == .connected("Connected to Voxt 1.16"))
}

@MainActor @Test func launchWaitsForVoxtToStart() async {
    let preferences = TestPreferences(); preferences.wasApproved = true
    let sleeps = Recorder()
    let authorizer = ScriptedAuthorizer([.failure(.voxtNotRunning), .failure(.voxtNotRunning), approved])
    let model = launchModel(authorizer, preferences: preferences, sleeps: sleeps)
    #expect(await model.connectAtLaunch())
    #expect(authorizer.calls == 3)
    #expect(sleeps.values == [.seconds(5), .seconds(5)])
}

@MainActor @Test func launchGivesUpAfterTwoMinutesAndKeepsTheApproval() async {
    let preferences = TestPreferences(); preferences.wasApproved = true
    let sleeps = Recorder()
    let authorizer = ScriptedAuthorizer([])
    let model = launchModel(authorizer, preferences: preferences, sleeps: sleeps)
    #expect(await model.connectAtLaunch() == false)
    #expect(authorizer.calls == 25)
    #expect(sleeps.values.count == 24)
    #expect(model.connectFeedback == .failed("Voxt didn't answer. Open Voxt, or enter its port and Access token."))
    #expect(preferences.wasApproved)
}

@MainActor @Test(arguments: [ConnectorLink.Failure.denied, .rejectedBySignature, .untrustedVoxt])
func refusalAtLaunchForgetsTheApproval(failure: ConnectorLink.Failure) async {
    let preferences = TestPreferences(); preferences.wasApproved = true
    let authorizer = ScriptedAuthorizer([.failure(failure)])
    let model = launchModel(authorizer, preferences: preferences)
    #expect(await model.connectAtLaunch() == false)
    #expect(authorizer.calls == 1)
    #expect(!preferences.wasApproved)
}

@MainActor @Test func openAtLoginIsOffByDefaultAndFollowsTheSystemStatus() {
    let loginItem = FakeLoginItem()
    let model = launchModel(ScriptedAuthorizer([]), loginItem: loginItem)
    #expect(!model.opensAtLogin)
    model.setOpensAtLogin(true)
    #expect(loginItem.registers == 1); #expect(model.opensAtLogin); #expect(model.loginItemMessage == nil)
    model.setOpensAtLogin(false)
    #expect(loginItem.unregisters == 1); #expect(!model.opensAtLogin)
}

@MainActor @Test func openAtLoginExplainsWhenMacOSNeedsApprovalOrRefuses() async {
    let loginItem = FakeLoginItem()
    let model = launchModel(ScriptedAuthorizer([]), loginItem: loginItem)
    loginItem.registerError = CocoaError(.featureUnsupported)
    model.setOpensAtLogin(true)
    #expect(!model.opensAtLogin)
    #expect(model.loginItemMessage == "macOS didn't add Voxt Connector to Login Items.")
    loginItem.registerError = nil
    loginItem.status = .requiresApproval
    await model.refreshLoginItem()
    #expect(model.opensAtLogin == false)
    #expect(model.loginItemMessage == "Allow Voxt Connector in System Settings → General → Login Items.")
}
