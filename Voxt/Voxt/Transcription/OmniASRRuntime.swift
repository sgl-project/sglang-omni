import Foundation
import OmniModelArtifacts

/// One owned local Omni server for one model: prepare once, share, retire once.
///
/// `ready → retiring → stopped` is a real barrier: `retire()` refuses new
/// leases at once, lets work that already holds a lease finish (a multi-chunk
/// Final or a live session keeps issuing requests), and returns only after the
/// server process has exited.
actor OmniASRRuntime {
    enum State: Equatable {
        case idle
        case starting
        case ready(OmniServerEndpoint)
        case retiring
        case stopped
        case failed(String)
    }

    /// The longest audio the server accepts in one Qwen3-ASR stream request.
    static let qwenMaximumRequestSeconds: Float = 1200
    /// Grace periods for stopping the server process before escalating.
    static let shutdownGrace: Duration = .seconds(15)
    static let terminateGrace: Duration = .seconds(10)
    /// Enough of the server process's stderr to show why a start failed.
    static let diagnosticTailBytes = 2048

    nonisolated let kind: OmniASRModelKind
    nonisolated let modelDirectory: URL
    /// A speculative draft the server verifies against the model; nil decodes
    /// one token per step.
    nonisolated let draftDirectory: URL?
    nonisolated let configuration: OmniBackendConfiguration

    private(set) var state: State = .idle
    private var serverProcess: Process?
    private var modelDirectoryLease: OmniModelDirectoryLease?
    private var draftDirectoryLease: OmniModelDirectoryLease?
    private var controlPipe: Pipe?
    private var servingEndpoint: OmniServerEndpoint?
    private var preparation: Task<OmniServerEndpoint, Error>?
    private var retirement: Task<Void, Never>?
    private let session: URLSession
    private var activeUses = 0
    private var drainWaiters: [CheckedContinuation<Void, Never>] = []

    init(
        kind: OmniASRModelKind,
        modelDirectory: URL,
        draftDirectory: URL? = nil,
        configuration: OmniBackendConfiguration
    ) {
        self.kind = kind
        self.modelDirectory = modelDirectory
        self.draftDirectory = draftDirectory
        self.configuration = configuration
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        // Loopback requests must never be routed through a system proxy.
        sessionConfiguration.connectionProxyDictionary = [:]
        sessionConfiguration.timeoutIntervalForRequest = 600
        sessionConfiguration.timeoutIntervalForResource = 3600
        self.session = URLSession(configuration: sessionConfiguration)
    }

    var isServing: Bool {
        if case .ready = state { return true }
        return false
    }

    /// False once the runtime failed, its server exited, or it was retired: a
    /// caller that wants a server needs a new runtime.
    var canServe: Bool {
        switch state {
        case .idle, .starting, .ready:
            return true
        case .retiring, .stopped, .failed:
            return false
        }
    }

    /// Starts the server once; concurrent callers share the same launch.
    func prepare() async throws -> OmniServerEndpoint {
        switch state {
        case .ready(let endpoint):
            return endpoint
        case .retiring, .stopped:
            throw OmniASRRuntimeError.retired
        case .failed(let reason):
            throw OmniASRRuntimeError.launchFailed(reason)
        case .idle, .starting:
            break
        }
        if let preparation {
            return try await preparation.value
        }
        state = .starting
        let task = Task { try await self.launch() }
        preparation = task
        do {
            let endpoint = try await task.value
            if case .starting = state {
                state = .ready(endpoint)
                servingEndpoint = endpoint
            }
            guard case .ready = state else { throw OmniASRRuntimeError.retired }
            return endpoint
        } catch {
            if case .starting = state {
                state = .failed(error.localizedDescription)
            }
            await stopServerProcess()
            throw error
        }
    }

    /// Holds the server for work that spans requests, such as a live session.
    func beginUse() throws -> OmniServerEndpoint {
        guard case .ready(let endpoint) = state else { throw OmniASRRuntimeError.retired }
        activeUses += 1
        return endpoint
    }

    func endUse() {
        activeUses = max(0, activeUses - 1)
        if activeUses == 0 {
            drainWaiters.forEach { $0.resume() }
            drainWaiters.removeAll()
        }
    }

    /// Idempotent and awaitable: see the type documentation.
    func retire() async {
        if let retirement {
            await retirement.value
            return
        }
        state = .retiring
        let task = Task {
            await self.drainActiveUses()
            await self.stopServerProcess()
        }
        retirement = task
        await task.value
        state = .stopped
        servingEndpoint = nil
        session.invalidateAndCancel()
    }

    private func drainActiveUses() async {
        guard activeUses > 0 else { return }
        await withCheckedContinuation { drainWaiters.append($0) }
    }

    func transcribe(
        _ request: OmniTranscriptionRequest,
        onDelta: (@Sendable (String) -> Void)? = nil
    ) async throws -> OmniTranscriptionResult {
        let endpoint = try beginUse()
        defer { endUse() }
        return try await transcribe(request, holding: endpoint, onDelta: onDelta)
    }

    /// For callers that already hold a lease from `beginUse()`; keeps working
    /// while the runtime drains toward retirement.
    func transcribe(
        _ request: OmniTranscriptionRequest,
        holding endpoint: OmniServerEndpoint,
        onDelta: (@Sendable (String) -> Void)? = nil
    ) async throws -> OmniTranscriptionResult {
        guard activeUses > 0, servingEndpoint == endpoint else {
            throw OmniASRRuntimeError.retired
        }
        let boundary = "voxt-\(UUID().uuidString)"
        var urlRequest = URLRequest(url: endpoint.baseURL.appendingPathComponent("v1/audio/transcriptions"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        let body = OmniMultipartBody.transcription(request, modelName: endpoint.modelName, boundary: boundary)
        let (bytes, response) = try await session.streamingUpload(urlRequest, body: body)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if status != 200 {
            var detail = ""
            for try await line in bytes.lines {
                detail += line
                if detail.count > 2000 { break }
            }
            throw OmniTranscriptionError.httpStatus(status, detail)
        }
        var parser = OmniTranscriptionStreamParser()
        for try await line in bytes.lines {
            try Task.checkCancellation()
            if let delta = try parser.consume(line: line), !delta.isEmpty {
                onDelta?(delta)
            }
        }
        return try parser.finish()
    }

    /// The server process has exited; requests must stop here instead of
    /// reaching whatever later binds the port.
    private func serverProcessExited(processIdentifier: Int32) {
        guard let process = serverProcess, process.processIdentifier == processIdentifier,
              !process.isRunning else { return }
        process.terminationHandler = nil
        modelDirectoryLease = nil
        draftDirectoryLease = nil
        if case .ready = state {
            state = .failed("The local Omni server stopped unexpectedly.")
            servingEndpoint = nil
        }
    }

    private func launch() async throws -> OmniServerEndpoint {
        let directoryLease = try await ModelDiskOperations.perform {
            try OmniModelDirectoryLease.acquire(for: self.modelDirectory, mode: .sharedRead)
        }
        guard case .starting = state else { throw OmniASRRuntimeError.retired }
        modelDirectoryLease = directoryLease
        var arguments = [
            "--supervised",
            "--model-kind", kind.rawValue,
            "--model-directory", directoryLease.directory.path,
            "--startup-timeout-s", String(configuration.startupTimeoutSeconds),
        ]
        if let draftDirectory {
            let draftLease = try await ModelDiskOperations.perform {
                try OmniModelDirectoryLease.acquire(for: draftDirectory, mode: .sharedRead)
            }
            guard case .starting = state else { throw OmniASRRuntimeError.retired }
            draftDirectoryLease = draftLease
            arguments += ["--draft-model-path", draftLease.directory.path]
        }
        let process = Process()
        process.executableURL = configuration.runtimeExecutable
        process.arguments = arguments
        process.environment = Self.runtimeEnvironment(inheriting: ProcessInfo.processInfo.environment)
        let control = Pipe()
        let events = Pipe()
        let diagnostics = Pipe()
        // Note (Jiaxin Deng): writing shutdown to a server that just exited must not raise SIGPIPE in Voxt.
        _ = fcntl(control.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)
        process.standardInput = control
        process.standardOutput = events
        process.standardError = diagnostics
        // Drained for the server's whole life so it never blocks on a full pipe;
        // only the tail is kept, in memory, to explain a failed start.
        let stderrTail = OmniDiagnosticTail(limit: Self.diagnosticTailBytes)
        diagnostics.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
            } else {
                stderrTail.append(data)
            }
        }
        // Note (Omni): a live child must retain its reader even if the last caller drops the runtime.
        process.terminationHandler = { [self] terminated in
            Task { await self.serverProcessExited(processIdentifier: terminated.processIdentifier) }
        }
        // A retire that ran before this launch got the actor must win; nothing
        // suspends between this check and the spawn.
        guard case .starting = state else { throw OmniASRRuntimeError.retired }
        do {
            try process.run()
        } catch {
            process.terminationHandler = nil
            throw OmniASRRuntimeError.launchFailed(error.localizedDescription)
        }
        serverProcess = process
        controlPipe = control
        // Note (Jiaxin Deng): a server that never reports (a stalled model load) is killed at the
        // startup deadline; its stdout then ends and the launch fails.
        let timedOut = OmniStartupDeadline()
        let serverPID = process.processIdentifier
        let seconds = configuration.startupTimeoutSeconds
        let deadline = Task.detached {
            try await Task.sleep(for: .seconds(seconds))
            timedOut.expire()
            kill(serverPID, SIGKILL)
        }
        defer { deadline.cancel() }
        var iterator = Self.eventStream(events.fileHandleForReading).makeAsyncIterator()
        guard let first = try await iterator.next() else {
            if timedOut.expired {
                throw OmniASRRuntimeError.launchFailed(
                    "the server did not report ready within \(String(format: "%g", seconds)) s"
                )
            } else {
                throw OmniASRRuntimeError.launchFailed(
                    "server exited before reporting: \(stderrTail.text)"
                )
            }
        }
        guard first["event"] as? String == "ready",
              let host = first["host"] as? String,
              let port = first["port"] as? Int,
              let modelName = first["model_name"] as? String,
              let serverPID = first["server_pid"] as? Int
        else {
            throw OmniASRRuntimeError.launchFailed(first["reason"] as? String ?? "\(first)")
        }
        return OmniServerEndpoint(
            host: host,
            port: port,
            modelName: modelName,
            serverProcessIdentifier: Int32(serverPID)
        )
    }

    /// Asks for an orderly shutdown, then escalates to SIGTERM and finally SIGKILL.
    private func stopServerProcess() async {
        guard let process = serverProcess else {
            modelDirectoryLease = nil
            draftDirectoryLease = nil
            return
        }
        if let controlPipe {
            let shutdown = Data("{\"command\": \"shutdown\"}\n".utf8)
            try? controlPipe.fileHandleForWriting.write(contentsOf: shutdown)
            try? controlPipe.fileHandleForWriting.close()
        }
        controlPipe = nil
        if !(await Self.waitForExit(process, within: Self.shutdownGrace)) {
            process.terminate()
            if !(await Self.waitForExit(process, within: Self.terminateGrace)) {
                kill(process.processIdentifier, SIGKILL)
                // Note (Omni): an escalation deadline is not proof that the child released model files.
                while process.isRunning {
                    try? await Task.sleep(for: .milliseconds(20))
                }
            }
        }
        process.terminationHandler = nil
        serverProcess = nil
        modelDirectoryLease = nil
        draftDirectoryLease = nil
    }

    /// Voxt's environment without the dynamic loader settings a debugger or XCTest
    /// injects.
    nonisolated static func runtimeEnvironment(inheriting inherited: [String: String]) -> [String: String] {
        inherited.filter {
            !$0.key.hasPrefix("DYLD_") && !$0.key.hasPrefix("__XPC_DYLD_")
        }
    }

    private static func waitForExit(_ process: Process, within limit: Duration) async -> Bool {
        let deadline = ContinuousClock.now + limit
        while process.isRunning {
            guard ContinuousClock.now < deadline else { return false }
            try? await Task.sleep(for: .milliseconds(20))
        }
        return true
    }

    /// The server's stdout as JSON events, read on the file handle's own dispatch
    /// source: a blocking read per live server would starve the concurrency pool.
    nonisolated static func eventStream(
        _ handle: FileHandle
    ) -> AsyncThrowingStream<[String: Any], Error> {
        AsyncThrowingStream { continuation in
            let lines = OmniLineBuffer()
            handle.readabilityHandler = { handle in
                let data = handle.availableData
                if data.isEmpty {
                    handle.readabilityHandler = nil
                    continuation.finish()
                } else {
                    do {
                        for line in lines.append(data) {
                            if let event = try JSONSerialization.jsonObject(with: line) as? [String: Any] {
                                continuation.yield(event)
                            }
                        }
                    } catch {
                        handle.readabilityHandler = nil
                        continuation.finish(throwing: error)
                    }
                }
            }
            continuation.onTermination = { _ in handle.readabilityHandler = nil }
        }
    }
}
