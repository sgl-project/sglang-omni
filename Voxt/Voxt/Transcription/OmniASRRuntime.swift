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
    nonisolated let configuration: OmniBackendConfiguration

    private(set) var state: State = .idle
    private var serverProcess: Process?
    private var controlPipe: Pipe?
    private var servingEndpoint: OmniServerEndpoint?
    private var preparation: Task<OmniServerEndpoint, Error>?
    private var retirement: Task<Void, Never>?
    private let session: URLSession
    private var activeUses = 0
    private var drainWaiters: [CheckedContinuation<Void, Never>] = []

    init(kind: OmniASRModelKind, modelDirectory: URL, configuration: OmniBackendConfiguration) {
        self.kind = kind
        self.modelDirectory = modelDirectory
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
    private func serverProcessExited() {
        guard case .ready = state else { return }
        state = .failed("The local Omni server stopped unexpectedly.")
        servingEndpoint = nil
    }

    private func launch() async throws -> OmniServerEndpoint {
        let process = Process()
        process.executableURL = configuration.runtimeExecutable
        process.arguments = [
            "--supervised",
            "--model-kind", kind.rawValue,
            "--model-directory", modelDirectory.path,
            "--startup-timeout-s", String(configuration.startupTimeoutSeconds),
        ]
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
        process.terminationHandler = { [weak self] _ in
            Task { await self?.serverProcessExited() }
        }
        // A retire that ran before this launch got the actor must win; nothing
        // suspends between this check and the spawn.
        guard case .starting = state else { throw OmniASRRuntimeError.retired }
        do {
            try process.run()
        } catch {
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
        guard let process = serverProcess else { return }
        serverProcess = nil
        if let controlPipe {
            let shutdown = Data("{\"command\": \"shutdown\"}\n".utf8)
            try? controlPipe.fileHandleForWriting.write(contentsOf: shutdown)
            try? controlPipe.fileHandleForWriting.close()
        }
        controlPipe = nil
        if await Self.waitForExit(process, within: Self.shutdownGrace) { return }
        process.terminate()
        if await Self.waitForExit(process, within: Self.terminateGrace) { return }
        kill(process.processIdentifier, SIGKILL)
        _ = await Self.waitForExit(process, within: Self.terminateGrace)
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

extension OmniASRRuntime {
    /// Qwen3-ASR Final as MLXAudio decoded it: ≤1200 s energy-cut chunks share one
    /// token budget, and the first chunk's detected language is forced on the rest.
    /// One Qwen3-ASR Final chunk, built the way Voxt's Swift model built it:
    /// its stop rules, and its prompt audio layout (one more mel frame and its
    /// own audio token count), so transcripts match the original backend.
    /// Like the original, an energy cut may land up to 5 s past the chunk end,
    /// unless that could exceed the server's 1200 s per-request limit.
    nonisolated static func qwenCutMayPassChunkEnd(chunkDurationSeconds: Float) -> Bool {
        chunkDurationSeconds + OmniTranscriptionPlanning.energyCutSearchSeconds <= qwenMaximumRequestSeconds
    }

    nonisolated static func qwenFinalRequest(
        samples: [Float],
        sampleRate: Int,
        language: String?,
        context: String?,
        maxNewTokens: Int
    ) -> OmniTranscriptionRequest {
        OmniTranscriptionRequest(
            samples: samples,
            sampleRate: sampleRate,
            language: language,
            prompt: context,
            maxNewTokens: maxNewTokens,
            stopAtEndOfText: true,
            stopOnTokenLoop: true,
            includeGenerationMetadata: true,
            audioLayout: "voxt_swift"
        )
    }

    func transcribeQwenFinal(
        samples: [Float],
        sampleRate: Int,
        language: String?,
        context: String?,
        maxTokens: Int,
        chunkDurationSeconds: Float = 1200,
        minChunkDurationSeconds: Float = 1
    ) async throws -> (text: String, language: String?) {
        let chunkDuration = min(chunkDurationSeconds, Self.qwenMaximumRequestSeconds)
        let chunks = OmniTranscriptionPlanning.energySplitChunks(
            samples,
            sampleRate: sampleRate,
            chunkDurationSeconds: chunkDuration,
            minChunkDurationSeconds: minChunkDurationSeconds,
            allowsCutPastChunkEnd: Self.qwenCutMayPassChunkEnd(chunkDurationSeconds: chunkDuration)
        )
        let endpoint = try beginUse()
        defer { endUse() }
        var remainingTokens = maxTokens
        var resolvedLanguage = language
        var text = ""
        for chunk in chunks {
            if remainingTokens <= 0 { break }
            try Task.checkCancellation()
            let result = try await transcribe(Self.qwenFinalRequest(
                samples: chunk.samples,
                sampleRate: sampleRate,
                language: resolvedLanguage,
                context: context,
                maxNewTokens: remainingTokens
            ), holding: endpoint)
            guard let metadata = result.generationMetadata else {
                throw OmniTranscriptionError.streamError("generation metadata missing")
            }
            remainingTokens -= metadata.generatedTokenCount
            if resolvedLanguage == nil {
                resolvedLanguage = metadata.language
            }
            text += result.text
        }
        return (text.trimmingCharacters(in: .whitespacesAndNewlines), resolvedLanguage)
    }
}
