import Foundation
import Network

public enum LocalPairingBrowserFailure: Error, Equatable { case notFound }

/// Finds the Voxt Connector on the same Wi-Fi during its 120 s Add iPhone window and opens a
/// line channel to it. Not covered by unit tests: needs a real local network (and, on a
/// phone, the user's Local Network permission). Exercised in the device checklist.
public final class LocalPairingBrowser: @unchecked Sendable {
    public static let serviceType = "_voxtconnect._tcp"
    private let queue = DispatchQueue(label: "voxt.local-pairing")

    public init() {}

    /// The first Connector found within `timeout`, or nil (then the app offers the QR).
    public func firstConnector(timeout: TimeInterval = 10) async -> NWEndpoint? {
        await withCheckedContinuation { continuation in
            let browser = NWBrowser(for: .bonjour(type: Self.serviceType, domain: nil), using: .tcp)
            let once = OnceFlag()
            browser.browseResultsChangedHandler = { results, _ in
                guard let first = results.first else { return }
                if once.claim() { browser.cancel(); continuation.resume(returning: first.endpoint) }
            }
            browser.start(queue: queue)
            queue.asyncAfter(deadline: .now() + timeout) {
                if once.claim() { browser.cancel(); continuation.resume(returning: nil) }
            }
        }
    }

    public func channel(to endpoint: NWEndpoint) -> NWPairingChannel {
        NWPairingChannel(connection: NWConnection(to: endpoint, using: .tcp), queue: queue)
    }
}

/// LocalPairingChannel over an NWConnection.
public final class NWPairingChannel: LocalPairingChannel, @unchecked Sendable {
    private let connection: NWConnection
    private var buffer = Data()

    init(connection: NWConnection, queue: DispatchQueue) {
        self.connection = connection
        connection.start(queue: queue)
    }

    public func send(_ data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error { continuation.resume(throwing: error) } else { continuation.resume() }
            })
        }
    }

    public func readLine() async throws -> Data {
        while true {
            if let newline = buffer.firstIndex(of: 0x0A) {
                let line = Data(buffer[buffer.startIndex..<newline])
                buffer = Data(buffer[buffer.index(after: newline)...])
                return line
            }
            guard buffer.count < 4096 else { throw LocalPairingClient.Failure.invalidReply }
            let chunk: Data = try await withCheckedThrowingContinuation { continuation in
                connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { data, _, complete, error in
                    if let error { continuation.resume(throwing: error) }
                    else if let data, !data.isEmpty { continuation.resume(returning: data) }
                    else if complete { continuation.resume(throwing: LocalPairingClient.Failure.timedOut) }
                    else { continuation.resume(returning: Data()) }
                }
            }
            buffer.append(chunk)
        }
    }

    public func close() { connection.cancel() }
}

private final class OnceFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    func claim() -> Bool { lock.withLock { if done { return false }; done = true; return true } }
}
