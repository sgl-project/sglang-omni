import Foundation

/// Which native server binary to run and how to start it.
nonisolated struct OmniBackendConfiguration: Sendable, Equatable {
    var runtimeExecutable: URL
    var startupTimeoutSeconds: Double = 180
}

/// The last bytes a child process wrote, kept in memory only.
nonisolated final class OmniDiagnosticTail: @unchecked Sendable {
    private let limit: Int
    private let lock = NSLock()
    private var bytes = Data()

    init(limit: Int) {
        self.limit = limit
    }

    func append(_ data: Data) {
        lock.lock()
        defer { lock.unlock() }
        bytes.append(data)
        if bytes.count > limit {
            bytes = Data(bytes.suffix(limit))
        }
    }

    var text: String {
        lock.lock()
        defer { lock.unlock() }
        return String(decoding: bytes, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

nonisolated struct OmniServerEndpoint: Sendable, Equatable {
    let host: String
    let port: Int
    let modelName: String
    let serverProcessIdentifier: Int32

    var baseURL: URL { URL(string: "http://\(host):\(port)")! }
}

nonisolated enum OmniASRRuntimeError: LocalizedError, Equatable {
    case launchFailed(String)
    case serverExited(Int?)
    case retired

    var errorDescription: String? {
        switch self {
        case .launchFailed(let reason):
            return "The local Omni server did not start: \(reason)"
        case .serverExited(let code):
            return "The local Omni server stopped unexpectedly (code \(code.map(String.init) ?? "unknown"))."
        case .retired:
            return "The local Omni server was released before the request finished."
        }
    }
}

/// Whether a launch's startup deadline passed.
nonisolated final class OmniStartupDeadline: @unchecked Sendable {
    private let lock = NSLock()
    private var passed = false

    func expire() {
        lock.lock()
        passed = true
        lock.unlock()
    }

    var expired: Bool {
        lock.lock()
        defer { lock.unlock() }
        return passed
    }
}

/// Splits bytes into newline-terminated lines across reads.
nonisolated final class OmniLineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()

    /// The complete lines in what has arrived so far, without their newlines;
    /// a trailing partial line waits for the next call.
    func append(_ data: Data) -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        pending.append(data)
        var lines: [Data] = []
        while let newline = pending.firstIndex(of: UInt8(ascii: "\n")) {
            let line = pending[pending.startIndex ..< newline]
            if !line.isEmpty {
                lines.append(Data(line))
            }
            pending.removeSubrange(pending.startIndex ... newline)
        }
        return lines
    }
}

nonisolated extension URLSession {
    /// Streams an upload's response so cancellation reaches the server mid-request.
    func streamingUpload(
        _ request: URLRequest,
        body: Data
    ) async throws -> (URLSession.AsyncBytes, URLResponse) {
        var uploadRequest = request
        uploadRequest.httpBody = body
        return try await bytes(for: uploadRequest)
    }
}
