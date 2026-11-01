import Foundation

@MainActor public final class TailcatRuntime: ConnectorRuntime {
    public var onUnexpectedExit: (() -> Void)?
    private let process: TailcatProcess
    private let configuration: TailcatConfiguration
    private let persistState: (String) throws -> Void

    public init(executableURL: URL, localPort: Int, allowedClients: [String], state: String?,
                persistState: @escaping (String) throws -> Void, startupTimeoutSeconds: Double = 20) {
        process = TailcatProcess(executableURL: executableURL, startupTimeoutSeconds: startupTimeoutSeconds)
        configuration = TailcatConfiguration(state: state, allowedClients: allowedClients, localPort: localPort)
        self.persistState = persistState
        process.onUnexpectedExit = { [weak self] in self?.onUnexpectedExit?() }
    }

    public func start() async throws -> String {
        do {
            let ready = try await process.start(configuration: configuration)
            try persistState(ready.state)
            return ready.address
        } catch {
            process.stop()
            throw error
        }
    }

    public func stop() { process.stop() }
}
