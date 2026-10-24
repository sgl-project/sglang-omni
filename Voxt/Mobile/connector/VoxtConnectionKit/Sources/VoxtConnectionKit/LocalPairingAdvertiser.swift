import Foundation
import Network

/// Advertises `_voxtconnect._tcp` on the local network only while Add iPhone is open, and
/// hands each connection to the coordinator. Not covered by unit tests: needs a real network
/// and, on first use, the macOS Local Network permission (user present).
public final class LocalPairingAdvertiser: @unchecked Sendable {
    public static let serviceType = "_voxtconnect._tcp"
    private let queue = DispatchQueue(label: "voxt.connector.local-pairing")
    private var listener: NWListener?

    public init() {}

    public func start(coordinator: LocalPairingCoordinator) throws {
        stop()
        let listener = try NWListener(using: .tcp)
        listener.service = NWListener.Service(type: Self.serviceType, txtRecord: NWTXTRecord(["v": "1"]))
        listener.newConnectionHandler = { [queue] connection in
            let channel = NWPairingChannel(connection: connection, queue: queue)
            Task {
                await coordinator.serve(channel)
                channel.close()
            }
        }
        listener.start(queue: queue)
        self.listener = listener
    }

    public func stop() {
        listener?.cancel()
        listener = nil
    }
}

/// LocalPairingChannel over an NWConnection.
final class NWPairingChannel: LocalPairingChannel, @unchecked Sendable {
    private let connection: NWConnection
    private var buffer = Data()

    init(connection: NWConnection, queue: DispatchQueue) {
        self.connection = connection
        connection.start(queue: queue)
    }

    func send(_ data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error { continuation.resume(throwing: error) } else { continuation.resume() }
            })
        }
    }

    func readLine() async throws -> Data {
        while true {
            if let newline = buffer.firstIndex(of: 0x0A) {
                let line = Data(buffer[buffer.startIndex..<newline])
                buffer = Data(buffer[buffer.index(after: newline)...])
                return line
            }
            guard buffer.count < 4096 else { throw LocalPairingFailure.invalidReply }
            let chunk: Data = try await withCheckedThrowingContinuation { continuation in
                connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { data, _, complete, error in
                    if let error { continuation.resume(throwing: error) }
                    else if let data, !data.isEmpty { continuation.resume(returning: data) }
                    else if complete { continuation.resume(throwing: LocalPairingFailure.timedOut) }
                    else { continuation.resume(returning: Data()) }
                }
            }
            buffer.append(chunk)
        }
    }

    func close() { connection.cancel() }
}
