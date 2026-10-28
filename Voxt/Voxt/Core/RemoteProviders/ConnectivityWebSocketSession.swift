import Foundation

/// Owns a probe's socket and dedicated URLSession. URLSession cancellation must
/// happen before waiting for a losing receive task; task-group cancellation
/// alone is not a transport-close barrier.
nonisolated final class ConnectivityWebSocketSession: @unchecked Sendable {
    private let sendMessage: @Sendable (URLSessionWebSocketTask.Message) async throws -> Void
    private let receiveMessage: @Sendable () async throws -> URLSessionWebSocketTask.Message
    private let closeTransport: @Sendable () -> Void
    private let lock = NSLock()
    private var isClosed = false

    init(
        send: @escaping @Sendable (URLSessionWebSocketTask.Message) async throws -> Void,
        receive: @escaping @Sendable () async throws -> URLSessionWebSocketTask.Message,
        close: @escaping @Sendable () -> Void
    ) {
        sendMessage = send
        receiveMessage = receive
        closeTransport = close
    }

    @MainActor
    convenience init(managedSocket: VoxtNetworkSession.ManagedWebSocketTask) {
        let task = managedSocket.task
        let session = managedSocket.session
        self.init(
            send: { try await task.send($0) },
            receive: { try await task.receive() },
            close: {
                task.cancel(with: .goingAway, reason: nil)
                session.invalidateAndCancel()
            }
        )
        task.resume()
    }

    func close() {
        let shouldClose = lock.withLock {
            guard !isClosed else { return false }
            isClosed = true
            return true
        }
        if shouldClose { closeTransport() }
    }

    func send(_ message: URLSessionWebSocketTask.Message) async throws {
        try await withTaskCancellationHandler {
            do {
                try checkOpen()
                try await sendMessage(message)
                try Task.checkCancellation()
            } catch {
                close()
                try Task.checkCancellation()
                throw error
            }
        } onCancel: {
            self.close()
        }
    }

    func receive(
        timeoutSeconds: TimeInterval,
        wait: @escaping @Sendable (TimeInterval) async throws -> Void = { try await Task.sleep(for: .seconds($0)) }
    ) async throws -> URLSessionWebSocketTask.Message {
        try await withTaskCancellationHandler {
            try checkOpen()
            return try await withThrowingTaskGroup(of: URLSessionWebSocketTask.Message.self) { group in
                defer { group.cancelAll() }
                group.addTask { try await self.receiveMessage() }
                group.addTask {
                    try await wait(timeoutSeconds)
                    throw NSError(
                        domain: "Voxt.Settings", code: -121,
                        userInfo: [NSLocalizedDescriptionKey: "Connection test timed out waiting for server packet."]
                    )
                }
                do {
                    let result = try await group.next()!
                    try Task.checkCancellation()
                    return result
                } catch {
                    // Wake receive before the group implicitly waits for it.
                    close()
                    try Task.checkCancellation()
                    throw error
                }
            }
        } onCancel: {
            self.close()
        }
    }

    private func checkOpen() throws {
        try Task.checkCancellation()
        guard !lock.withLock({ isClosed }) else { throw CancellationError() }
    }

    deinit { close() }
}
