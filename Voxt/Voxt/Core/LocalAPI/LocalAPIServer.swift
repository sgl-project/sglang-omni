// LocalAPIServer.swift
// Provides the loopback-only HTTP listener behind the Local API. Other devices can reach it only
// through a tunnel the user runs, never directly.

import Foundation
import Network

nonisolated final class LocalAPIServer: @unchecked Sendable {
    typealias Handler = @Sendable (LocalAPIHTTPRequest) async -> LocalAPIHTTPResponse

    enum ServerError: LocalizedError {
        case invalidPort(UInt16)
        case failed(String)

        var errorDescription: String? {
            switch self {
            case let .invalidPort(port): "Port \(port) can't be used."
            case let .failed(reason): reason
            }
        }
    }

    /// A client that stops sending mid-request is dropped after this long.
    static let requestTimeout: TimeInterval = 60

    private let requestedPort: UInt16
    private let maxBodyBytes: Int
    private let handler: Handler
    private let queue = DispatchQueue(label: "com.voxt.remote-input.server")
    private var listener: NWListener?

    init(port: UInt16, maxBodyBytes: Int, handler: @escaping Handler) {
        requestedPort = port
        self.maxBodyBytes = maxBodyBytes
        self.handler = handler
    }

    /// Starts listening on 127.0.0.1 and returns the bound port (useful when `port` is 0).
    func start() async throws -> UInt16 {
        let port: NWEndpoint.Port
        if requestedPort == 0 {
            port = .any
        } else if let fixed = NWEndpoint.Port(rawValue: requestedPort) {
            port = fixed
        } else {
            throw ServerError.invalidPort(requestedPort)
        }

        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: port)
        parameters.allowLocalEndpointReuse = true
        let listener = try NWListener(using: parameters)
        self.listener = listener

        return try await withCheckedThrowingContinuation { continuation in
            // All callbacks run on `queue`; the handler is cleared after the first terminal state
            // so the continuation resumes exactly once.
            listener.stateUpdateHandler = { [weak listener] state in
                switch state {
                case .ready:
                    listener?.stateUpdateHandler = nil
                    continuation.resume(returning: listener?.port?.rawValue ?? 0)
                case let .failed(error):
                    listener?.stateUpdateHandler = nil
                    continuation.resume(throwing: ServerError.failed(error.localizedDescription))
                case .cancelled:
                    listener?.stateUpdateHandler = nil
                    continuation.resume(throwing: CancellationError())
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                self?.accept(connection)
            }
            listener.start(queue: queue)
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        let deadline = DispatchWorkItem { connection.cancel() }
        queue.asyncAfter(deadline: .now() + Self.requestTimeout, execute: deadline)
        receive(on: connection, buffer: Data(), deadline: deadline)
    }

    private func receive(on connection: NWConnection, buffer: Data, deadline: DispatchWorkItem) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 256 * 1024) { [weak self] chunk, _, isComplete, error in
            guard let self else { return connection.cancel() }
            var buffer = buffer
            if let chunk { buffer.append(chunk) }

            switch LocalAPIHTTPParser.parse(buffer, maxBodyBytes: self.maxBodyBytes) {
            case let .complete(request, _):
                deadline.cancel()
                let handler = self.handler
                Task {
                    let response = await handler(request)
                    self.respond(response, on: connection)
                }
            case let .failure(status):
                deadline.cancel()
                self.respond(.error(status: status, LocalAPIHTTPResponse.reasonPhrase(status)), on: connection)
            case .incomplete:
                if isComplete || error != nil {
                    deadline.cancel()
                    connection.cancel()
                } else {
                    self.receive(on: connection, buffer: buffer, deadline: deadline)
                }
            }
        }
    }

    private func respond(_ response: LocalAPIHTTPResponse, on connection: NWConnection) {
        connection.send(content: response.serialized(), contentContext: .finalMessage, isComplete: true, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }
}
