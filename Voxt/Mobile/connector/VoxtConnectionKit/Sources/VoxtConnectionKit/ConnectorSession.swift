import Foundation
import Observation

@MainActor public protocol ConnectorRuntime: AnyObject {
    var onUnexpectedExit: (() -> Void)? { get set }
    func start() async throws -> String
    func stop()
}

@MainActor @Observable public final class ConnectorSession {
    public enum Failure: Equatable { case unauthorized, unavailable, invalidResponse, runtime }
    public enum Status: Equatable {
        case disconnected, checking
        case localReady(version: String)
        case ready(version: String, address: String)
        case failed(Failure)
    }
    public private(set) var status: Status = .disconnected
    @ObservationIgnored private let health: () async throws -> LocalInferenceClient.Health
    @ObservationIgnored private let makeRuntime: () -> (any ConnectorRuntime)?
    @ObservationIgnored private var runtime: (any ConnectorRuntime)?
    @ObservationIgnored private var generation = 0

    public init(health: @escaping () async throws -> LocalInferenceClient.Health,
                makeRuntime: @escaping () -> (any ConnectorRuntime)?) {
        self.health = health
        self.makeRuntime = makeRuntime
    }

    public func connect() async {
        disconnect()
        let attempt = generation
        status = .checking
        let result: LocalInferenceClient.Health
        do {
            result = try await health()
            try Task.checkCancellation()
        } catch {
            guard generation == attempt else { return }
            if error is CancellationError || (error as? URLError)?.code == .cancelled {
                status = .disconnected
            } else {
                switch error as? LocalInferenceClient.Failure {
                case .unauthorized: status = .failed(.unauthorized)
                case .invalidEndpoint, .invalidResponse: status = .failed(.invalidResponse)
                default: status = .failed(.unavailable)
                }
            }
            return
        }
        guard generation == attempt else { return }
        guard let owned = makeRuntime() else {
            status = .localReady(version: result.version)
            return
        }
        runtime = owned
        owned.onUnexpectedExit = { [weak self] in
            guard let self, self.generation == attempt else { return }
            self.generation &+= 1
            self.runtime?.stop()
            self.runtime = nil
            self.status = .failed(.runtime)
        }
        do {
            let address = try await owned.start()
            try Task.checkCancellation()
            guard generation == attempt else { owned.stop(); return }
            status = .ready(version: result.version, address: address)
        } catch {
            owned.stop()
            guard generation == attempt else { return }
            runtime = nil
            status = error is CancellationError ? .disconnected : .failed(.runtime)
        }
    }

    public func disconnect() {
        generation &+= 1
        runtime?.stop()
        runtime = nil
        status = .disconnected
    }
}
