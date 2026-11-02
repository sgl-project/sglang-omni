import Foundation
import Combine

/// Owns sheet-scoped requests independently of SwiftUI value copies. Replaced
/// tasks remain tracked until exit; only the current invocation may publish.
@MainActor
final class RemoteProviderSheetOperations: ObservableObject {
    struct ConnectionResult: Equatable {
        let id = UUID()
        let message: String
        let succeeded: Bool
    }

    @Published private(set) var modelOptions: [RemoteModelOption]?
    @Published private(set) var isTestingConnection = false
    @Published private(set) var connectionResult: ConnectionResult?
    private let modelTasks = TrackedTaskStore()
    private let connectionTasks = TrackedTaskStore()
    private var modelRequestID = UUID()
    private var connectionRequestID = UUID()
    private var isActive = true

    func activate() {
        isActive = true
    }

    @discardableResult
    func loadModels(_ operation: @escaping @MainActor () async -> [RemoteModelOption]) -> Task<Void, Never>? {
        guard isActive else { return nil }
        modelTasks.cancelAll()
        let id = UUID()
        modelRequestID = id
        modelOptions = nil
        return modelTasks.start { [weak self] in
            let options = await operation()
            guard !Task.isCancelled, let self, self.isActive, self.modelRequestID == id else { return }
            self.modelOptions = options
        }
    }

    @discardableResult
    func testConnection(_ operation: @escaping @MainActor () async throws -> String) -> Task<Void, Never>? {
        guard isActive else { return nil }
        connectionTasks.cancelAll()
        let id = UUID()
        connectionRequestID = id
        connectionResult = nil
        isTestingConnection = true
        return connectionTasks.start { [weak self] in
            let result: ConnectionResult
            do {
                result = ConnectionResult(message: try await operation(), succeeded: true)
            } catch {
                result = ConnectionResult(
                    message: VoxtNetworkSession.directModeConflictMessage(for: error) ?? error.localizedDescription,
                    succeeded: false
                )
            }
            guard !Task.isCancelled, let self, self.isActive, self.connectionRequestID == id else { return }
            self.isTestingConnection = false
            self.connectionResult = result
        }
    }

    func showFailure(_ message: String) {
        guard isActive else { return }
        connectionRequestID = UUID()
        connectionTasks.cancelAll()
        isTestingConnection = false
        connectionResult = ConnectionResult(message: message, succeeded: false)
    }

    func cancel() {
        isActive = false
        modelRequestID = UUID()
        connectionRequestID = UUID()
        modelTasks.cancelAll()
        connectionTasks.cancelAll()
        modelOptions = nil
        isTestingConnection = false
        connectionResult = nil
    }

    func waitForIdle() async {
        await modelTasks.waitForAll()
        await connectionTasks.waitForAll()
    }
}
