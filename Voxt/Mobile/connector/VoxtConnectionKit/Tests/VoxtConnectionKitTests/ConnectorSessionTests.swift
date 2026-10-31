import Testing
@testable import VoxtConnectionKit

@MainActor private final class RuntimeFixture: ConnectorRuntime {
    var onUnexpectedExit: (() -> Void)?
    var starts = 0
    var stops = 0
    var fail = false
    var duringStart: (() -> Void)?
    func start() async throws -> String {
        starts += 1
        duringStart?()
        if fail { throw LocalInferenceClient.Failure.unavailable }
        return "fixture-address"
    }
    func stop() { stops += 1 }
}

@MainActor @Test func connectorStartsOnlyAfterAuthenticatedHealthAndStopsOnDisconnect() async {
    let runtime = RuntimeFixture()
    var healthCalls = 0
    let session = ConnectorSession(health: {
        healthCalls += 1
        #expect(runtime.starts == 0)
        return .init(ok: true, app: "Voxt", version: "fixture-1")
    }, makeRuntime: { runtime })
    await session.connect()
    #expect(healthCalls == 1)
    #expect(runtime.starts == 1)
    #expect(session.status == .ready(version: "fixture-1", address: "fixture-address"))
    session.disconnect()
    #expect(runtime.stops == 1)
    #expect(session.status == .disconnected)
}

@MainActor @Test func badLocalTokenNeverStartsRuntime() async {
    let runtime = RuntimeFixture()
    let session = ConnectorSession(health: { throw LocalInferenceClient.Failure.unauthorized }, makeRuntime: { runtime })
    await session.connect()
    #expect(session.status == .failed(.unauthorized))
    #expect(runtime.starts == 0)
}

@MainActor @Test func unexpectedRuntimeExitClearsReadyStatus() async {
    let runtime = RuntimeFixture()
    let session = ConnectorSession(health: { .init(ok: true, app: "Voxt", version: "fixture-1") }, makeRuntime: { runtime })
    await session.connect()
    runtime.onUnexpectedExit?()
    #expect(session.status == .failed(.runtime))
}

@MainActor @Test func disconnectedHealthReplyCannotRestartRuntime() async {
    let runtime = RuntimeFixture()
    var session: ConnectorSession!
    session = ConnectorSession(health: {
        session.disconnect()
        return .init(ok: true, app: "Voxt", version: "fixture-1")
    }, makeRuntime: { runtime })
    await session.connect()
    #expect(session.status == .disconnected)
    #expect(runtime.starts == 0)
}

@MainActor @Test func runtimeStartFailureStopsOwnedRuntime() async {
    let runtime = RuntimeFixture(); runtime.fail = true
    let session = ConnectorSession(health: { .init(ok: true, app: "Voxt", version: "fixture-1") }, makeRuntime: { runtime })
    await session.connect()
    #expect(session.status == .failed(.runtime))
    #expect(runtime.stops == 1)
}

@MainActor @Test func runtimeExitDuringStartCannotBecomeReady() async {
    let runtime = RuntimeFixture()
    runtime.duringStart = { [weak runtime] in runtime?.onUnexpectedExit?() }
    let session = ConnectorSession(health: { .init(ok: true, app: "Voxt", version: "fixture-1") }, makeRuntime: { runtime })
    await session.connect()
    #expect(session.status == .failed(.runtime))
}

@MainActor @Test func unpairedConnectorVerifiesVoxtWithoutOpeningPublicRuntime() async {
    let session = ConnectorSession(health: { .init(ok: true, app: "Voxt", version: "fixture-1") }, makeRuntime: { nil })
    await session.connect()
    #expect(session.status == .localReady(version: "fixture-1"))
}
