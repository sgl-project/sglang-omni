import Foundation

/// Writes protocol lines, one whole line at a time, to a sink (the
/// process's original stdout; a buffer in tests).
public final class LineWriter: @unchecked Sendable {
    private let sink: @Sendable (Data) -> Void
    private let lock = NSLock()
    private let encoder = makeEncoder()

    public init(sink: @escaping @Sendable (Data) -> Void) {
        self.sink = sink
    }

    private func send<P: Encodable>(_ message: Outgoing<P>) {
        guard var data = try? encoder.encode(message) else {
            FileHandle.standardError.write(Data("[llm] could not encode a \(message.method ?? "response") message\n".utf8))
            return
        }
        data.append(0x0A)
        lock.withLock { sink(data) }
    }

    public func result<P: Encodable>(_ id: String, _ value: P) { send(Outgoing.response(id, value)) }
    public func fail(_ id: String, _ error: RPCError) { send(Outgoing<Empty>.failure(id, error)) }
    public func notify<P: Encodable>(_ method: String, _ params: P) { send(Outgoing.notification(method, params)) }

    /// A structured log line for the app's logs panel.
    public func log(_ level: String, _ message: String, job: String? = nil) {
        notify(Method.log, LogParams(level: level, message: message, job: job))
    }
}

/// Runs operations one after another, in the order they were queued.
final class SerialQueue: @unchecked Sendable {
    private let lock = NSLock()
    private var tail: Task<Void, Never>?

    /// Queues op; its task waits for the ones before it. A task cancelled
    /// while it waits throws CancellationError when its turn comes.
    func run<T: Sendable>(_ op: @escaping @Sendable () async throws -> T) -> Task<T, Error> {
        lock.withLock {
            let previous = tail
            let task = Task {
                await previous?.value
                try Task.checkCancellation()
                return try await op()
            }
            tail = Task { _ = await task.result }
            return task
        }
    }
}

/// The process's session: answers requests (docs/llm-protocol.md §3).
/// load, unload and chat go through one queue; the others are answered at
/// once. A chat is a job: job.cancel cancels its task and is answered
/// after the chat's own (cancelled) answer.
public final class Server: @unchecked Sendable {
    public let out: LineWriter
    let engine: any Engine
    let info: @Sendable () -> HelloResult
    let exit: @Sendable (Int32) -> Void
    private let queue = SerialQueue()
    let lock = NSLock()
    private var ready = false
    var jobs: [String: Task<Void, Never>] = [:]

    /// info answers llm.hello; exit ends the process (llm.shutdown).
    public init(out: LineWriter, engine: any Engine, info: @escaping @Sendable () -> HelloResult, exit: @escaping @Sendable (Int32) -> Void) {
        self.out = out
        self.engine = engine
        self.info = info
        self.exit = exit
    }

    /// Handles one line from stdin.
    public func handle(line: Data) {
        let msg: Incoming
        do {
            msg = try Incoming.decode(line)
        } catch {
            FileHandle.standardError.write(Data("[llm] bad line: \(String(decoding: line.prefix(200), as: UTF8.self))\n".utf8))
            return
        }
        guard msg.kind == .request, let id = msg.id, let method = msg.method else {
            FileHandle.standardError.write(Data("[llm] ignored a message that is not a request\n".utf8))
            return
        }
        handle(id: id, method: method, msg)
    }

    func handle(id: String, method: String, _ msg: Incoming) {
        if method == Method.hello {
            return answerHello(id, msg)
        }
        guard lock.withLock({ ready }) else {
            return out.fail(id, RPCError(Code.notInitialized, "llm.hello must come first"))
        }
        switch method {
        case Method.shutdown:
            let running = lock.withLock { Array(jobs.values) }
            running.forEach { $0.cancel() }
            Task {
                for t in running { await t.value }
                out.result(id, Empty())
                exit(0)
            }
        case Method.jobCancel:
            guard let p = decode(id, msg, CancelParams.self) else { return }
            let task = lock.withLock { jobs[p.job] }
            guard let task else { return out.result(id, CancelResult(cancelled: false)) }
            task.cancel()
            Task {
                await task.value
                out.result(id, CancelResult(cancelled: true))
            }
        case Method.status:
            out.result(id, StatusResult(model: engine.loaded(), memory: engine.memory()))
        case Method.load:
            guard let p = decode(id, msg, LoadParams.self) else { return }
            answer(id, queue.run { [engine] in
                let o = try await engine.load(p.model, cacheLimitMB: p.cacheLimitMB)
                return LoadResult(model: p.model, cached: o.cached ? true : nil, loadMs: o.loadMs, memory: engine.memory())
            })
        case Method.unload:
            answer(id, queue.run { [engine] in
                let unloaded = await engine.unload()
                return UnloadResult(unloaded: unloaded, memory: engine.memory())
            })
        case Method.chat:
            guard let p = decode(id, msg, ChatParams.self) else { return }
            startChat(id, p)
        default:
            out.fail(id, RPCError(Code.methodNotFound, "Unknown method \(method)"))
        }
    }

    private func decode<T: Decodable>(_ id: String, _ msg: Incoming, _ type: T.Type) -> T? {
        do {
            return try msg.decodeParams(type)
        } catch {
            out.fail(id, rpcError(error))
            return nil
        }
    }

    private func answer<T: Encodable & Sendable>(_ id: String, _ task: Task<T, Error>) {
        Task {
            switch await task.result {
            case .success(let r): out.result(id, r)
            case .failure(let e): out.fail(id, rpcError(e))
            }
        }
    }

    private func startChat(_ id: String, _ p: ChatParams) {
        if p.messages.isEmpty || p.maxTokens < 0 {
            return out.fail(id, RPCError(Code.badRequest, "A chat needs messages and maxTokens"))
        }
        let work = queue.run { [engine, out] in
            let started = monotonicMs()
            var loadMs: Double?
            if engine.loaded() != p.model {
                if let was = engine.loaded() {
                    out.log("info", "unloading \(was.id); loading \(p.model.id) (\(p.model.format))", job: id)
                }
                loadMs = try await engine.load(p.model, cacheLimitMB: nil).loadMs
            }
            try Task.checkCancellation()
            let o = try await engine.chat(p) { piece in
                out.notify(Method.chatDelta, DeltaParams(job: id, text: piece))
            }
            let total = monotonicMs() - started
            let first = o.firstTokenMs + (loadMs ?? 0)
            return ChatResult(text: o.text, finish: o.finish, usage: o.usage,
                              timings: Timings(loadMs: loadMs, promptMs: o.promptMs, firstTokenMs: first, totalMs: total, tokensPerSecond: o.tokensPerSecond, cache: o.cache),
                              memory: engine.memory())
        }
        // Made under the lock, so that a chat ending at once cannot
        // remove itself before it is registered.
        lock.withLock {
            jobs[id] = Task { [weak self] in
                let result = await withTaskCancellationHandler {
                    await work.result
                } onCancel: {
                    work.cancel()
                }
                guard let self else { return }
                switch result {
                case .success(let r): out.result(id, r)
                case .failure(let e): out.fail(id, rpcError(e))
                }
                lock.withLock { jobs[id] = nil }
            }
        }
    }

    private func answerHello(_ id: String, _ msg: Incoming) {
        guard let p = decode(id, msg, HelloParams.self) else { return }
        guard p.protocol == protocolVersion else {
            return out.fail(id, RPCError(Code.protocolMismatch, "This process speaks protocol \(protocolVersion)",
                                         data: .object(["supported": .array([.int(Int64(protocolVersion))])])))
        }
        lock.withLock { ready = true }
        let hello = info()
        out.log("info", "voxtnew-llm \(hello.version) ready (app \(p.app))")
        out.result(id, hello)
    }
}

extension Server {
    /// Reads requests from stdin on a thread of its own until it closes
    /// (the app quit or crashed), then cancels the chat and exits.
    public func serveStdin() {
        let thread = Thread { [self] in
            while let line = readLine(strippingNewline: true) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.isEmpty { continue }
                handle(line: Data(trimmed.utf8))
            }
            FileHandle.standardError.write(Data("[llm] stdin closed, exiting\n".utf8))
            cancelAll()
            exit(0)
        }
        thread.name = "voxtnew-llm.stdin"
        thread.stackSize = 1 << 20
        thread.start()
    }

    /// Cancels every job.
    func cancelAll() {
        lock.withLock { Array(jobs.values) }.forEach { $0.cancel() }
    }
}
