import Foundation
import Testing
@testable import VoxtConnectionKit

@MainActor private final class EmptyStore: ConnectorConfigurationStore {
    func load() throws -> ConnectorConfiguration { .init(allowedClients: [], state: nil) }
    func save(_ value: ConnectorConfiguration) throws {}
}

/// Yields until the condition holds; records a failure instead of hanging if it never does.
@MainActor private func eventually(_ condition: () -> Bool, sourceLocation: SourceLocation = #_sourceLocation) async -> Bool {
    for _ in 0..<10_000 where !condition() { await Task.yield() }
    #expect(condition(), "condition never became true", sourceLocation: sourceLocation)
    return condition()
}

/// Holds the health check open until the test releases it.
@MainActor private final class HealthGate {
    private var waiter: CheckedContinuation<Void, Never>?
    var calls = 0
    func wait() async { calls += 1; await withCheckedContinuation { waiter = $0 } }
    func release() { waiter?.resume(); waiter = nil }
}

@MainActor private final class ReadyPairRuntime: ConnectorPairingRuntime {
    var onEvent: ((ConnectorPairingProcess.Event) -> Void)?
    var observed: [ConnectorAppModel.ConnectFeedback] = []
    weak var model: ConnectorAppModel?
    func start(localPort: Int, token: String, configuration: ConnectorConfiguration) async throws {
        if let model { observed.append(model.connectFeedback) }
        onEvent?(.ready(address: ""))
    }
    func beginPairing() throws {}
    func cancelPairing() throws {}
    func stop() {}
}

@MainActor @Test func connectShowsProgressThenTheConnectedVoxtVersion() async {
    let gate = HealthGate()
    let model = ConnectorAppModel(runtimeURL: URL(fileURLWithPath: "/missing"), store: EmptyStore(), health: { _, _ in
        await gate.wait()
        return .init(ok: true, app: "Voxt", version: "1.16")
    })
    model.token = "fixture"
    #expect(model.connectFeedback == .idle)
    #expect(!model.canPair)
    let attempt = Task { await model.connect() }
    guard await eventually({ gate.calls > 0 }) else { gate.release(); await attempt.value; return }
    #expect(model.connectFeedback == .connecting)
    #expect(!model.canPair)
    gate.release()
    await attempt.value
    #expect(model.connectFeedback == .connected("Connected to Voxt 1.16"))
    #expect(model.canPair)
}

@MainActor @Test func connectShowsTheErrorInline() async {
    let model = ConnectorAppModel(runtimeURL: URL(fileURLWithPath: "/missing"), store: EmptyStore(), health: { _, _ in
        throw LocalInferenceClient.Failure.unauthorized
    })
    model.token = "rejected-fixture"
    await model.connect()
    #expect(model.connectFeedback == .failed("The access token was rejected. Copy it from Voxt Settings."))
    #expect(!model.canPair)
    model.port = "0"
    await model.connect()
    #expect(model.connectFeedback == .failed("Enter the port shown in Voxt Settings."))
}

@MainActor @Test func preparingPairingKeepsShowingConnected() async {
    let runtime = ReadyPairRuntime()
    let model = ConnectorAppModel(runtimeURL: URL(fileURLWithPath: "/missing"), store: EmptyStore(),
        health: { _, _ in .init(ok: true, app: "Voxt", version: "1.16") }, makePairingRuntime: { runtime })
    runtime.model = model
    model.token = "fixture"
    await model.connect()
    await model.beginPairing()
    #expect(runtime.observed == [.connected("Connected to Voxt 1.16")])
    #expect(model.connectFeedback == .connected("Connected to Voxt 1.16"))
}

@MainActor @Test func settingsOpenAtLaunchUntilConnected() async {
    var fail = true
    let model = ConnectorAppModel(runtimeURL: URL(fileURLWithPath: "/missing"), store: EmptyStore(), health: { _, _ in
        if fail { throw LocalInferenceClient.Failure.unavailable }
        return .init(ok: true, app: "Voxt", version: "1.16")
    }, preferences: TestPreferences())
    #expect(model.opensSettingsAtLaunch)
    model.token = "fixture"
    await model.connect()
    #expect(model.opensSettingsAtLaunch)
    fail = false
    await model.connect()
    #expect(!model.opensSettingsAtLaunch)
}

/// When Voxt doesn't answer, the status names Voxt's own setting: Settings › General › Local API.
@MainActor @Test func unreachableVoxtNamesItsLocalAPISetting() async {
    let model = ConnectorAppModel(runtimeURL: URL(fileURLWithPath: "/missing"), store: EmptyStore(), health: { _, _ in
        throw LocalInferenceClient.Failure.unavailable
    }, preferences: TestPreferences())
    model.token = "fixture"
    await model.connect()
    #expect(model.statusText == "Could not reach Voxt. Open it and, in Settings › General › Local API, turn on “Allow other apps to dictate through Voxt”.")
}

/// Answers authorize() with a scripted result once the test releases it.
private final class FakeAuthorizer: ConnectorAuthorizing, @unchecked Sendable {
    private let lock = NSLock()
    private var waiter: CheckedContinuation<ConnectorGrant, Error>?
    private var _calls = 0, _cancelled = false
    var calls: Int { lock.withLock { _calls } }
    var cancelled: Bool { lock.withLock { _cancelled } }
    func authorize() async throws -> ConnectorGrant {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.withLock { _calls += 1; waiter = continuation }
            }
        } onCancel: { lock.withLock { _cancelled = true } }
    }
    func finish(_ result: Result<ConnectorGrant, Error>) {
        let continuation = lock.withLock { () -> CheckedContinuation<ConnectorGrant, Error>? in
            defer { waiter = nil }; return waiter
        }
        continuation?.resume(with: result)
    }
}

@MainActor private func makeModel(authorizer: FakeAuthorizer, healthCalls: @escaping (Int, String) -> Void = { _, _ in }) -> ConnectorAppModel {
    ConnectorAppModel(runtimeURL: URL(fileURLWithPath: "/missing"), store: EmptyStore(), health: { port, token in
        healthCalls(port, token)
        return .init(ok: true, app: "Voxt", version: "1.16")
    }, authorizer: authorizer, preferences: TestPreferences())
}

@MainActor @Test func emptyTokenAsksVoxtThenConnectsWithTheApprovedPortAndToken() async {
    let authorizer = FakeAuthorizer()
    var checked: [(Int, String)] = []
    let model = makeModel(authorizer: authorizer) { checked.append(($0, $1)) }
    let attempt = Task { await model.connect() }
    guard await eventually({ authorizer.calls > 0 }) else { return }
    #expect(model.connectFeedback == .waitingForApproval)
    #expect(model.isConnecting)
    #expect(model.statusText == "Waiting for Voxt to allow this Connector")
    authorizer.finish(.success(ConnectorGrant(port: 47920, token: "approved-fixture", alwaysAllowed: true)))
    await attempt.value
    #expect(checked.count == 1)
    #expect(checked.first?.0 == 47920); #expect(checked.first?.1 == "approved-fixture")
    #expect(model.port == "47920")
    #expect(model.connectFeedback == .connected("Connected to Voxt 1.16"))
}

@MainActor @Test func typedTokenSkipsTheAutomaticRequest() async {
    let authorizer = FakeAuthorizer()
    let model = makeModel(authorizer: authorizer)
    model.token = "typed-fixture"
    await model.connect()
    #expect(authorizer.calls == 0)
    #expect(model.connectFeedback == .connected("Connected to Voxt 1.16"))
}

@MainActor @Test(arguments: [
    (ConnectorLink.Failure.voxtNotRunning, "Voxt didn't answer. Open Voxt, or enter its port and Access token."),
    (.denied, "Voxt didn't allow this Connector. Connect again, or enter the port and Access token."),
    (.rejectedBySignature, "Voxt doesn't accept this Connector build. Enter the port and Access token instead."),
    (.untrustedVoxt, "The app answering as Voxt isn't signed by Voxt. Nothing was sent."),
    (.invalidReply, "Voxt sent an unexpected reply. Enter the port and Access token instead."),
])
func automaticFailureShowsWhatToDoAndNeverChecksHealth(failure: ConnectorLink.Failure, message: String) async {
    let authorizer = FakeAuthorizer()
    var healthCalls = 0
    let model = makeModel(authorizer: authorizer) { _, _ in healthCalls += 1 }
    let attempt = Task { await model.connect() }
    guard await eventually({ authorizer.calls > 0 }) else { return }
    authorizer.finish(.failure(failure))
    await attempt.value
    #expect(healthCalls == 0)
    #expect(model.connectFeedback == .failed(message))
    #expect(!model.isConnecting)
    #expect(model.token.isEmpty)
}

@MainActor @Test func cancellingTheWaitIgnoresALateApproval() async {
    let authorizer = FakeAuthorizer()
    var healthCalls = 0
    let model = makeModel(authorizer: authorizer) { _, _ in healthCalls += 1 }
    let attempt = Task { await model.connect() }
    guard await eventually({ authorizer.calls > 0 }) else { return }
    model.cancelAuthorization()
    #expect(model.connectFeedback == .idle)
    #expect(!model.isConnecting)
    // Cancelling the request lets ConnectorLink close its socket right away.
    #expect(await eventually({ authorizer.cancelled }))
    authorizer.finish(.success(ConnectorGrant(port: 47920, token: "late-fixture", alwaysAllowed: true)))
    await attempt.value
    #expect(healthCalls == 0)
    #expect(model.token.isEmpty)
    #expect(model.connectFeedback == .idle)
}
