import VoxtNewHelperKit
import Foundation

/// Writes protocol lines to the original stdout, one whole line at a time.
/// It takes stdout over at startup and points fd 1 at stderr, so stray
/// prints of system libraries cannot corrupt the stream.
final class Writer: @unchecked Sendable {
    private let handle: FileHandle
    private let lock = NSLock()
    private let encoder = makeEncoder()

    init() {
        let fd = dup(STDOUT_FILENO)
        dup2(STDERR_FILENO, STDOUT_FILENO)
        setvbuf(stdout, nil, _IONBF, 0)
        handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    }

    private func send<P: Encodable>(_ message: Outgoing<P>) {
        guard var data = try? encoder.encode(message) else {
            log("could not encode a \(message.method ?? "response") message")
            return
        }
        data.append(0x0A)
        lock.lock()
        defer { lock.unlock() }
        try? handle.write(contentsOf: data)
    }

    func result<P: Encodable>(_ id: String, _ value: P) { send(Outgoing.response(id, value)) }
    func fail(_ id: String, _ error: RPCError) { send(Outgoing<Empty>.failure(id, error)) }
    func notify<P: Encodable>(_ method: String, _ params: P) { send(Outgoing.notification(method, params)) }

    /// A structured log line for the app's logs panel.
    func logLine(_ level: String, _ message: String, job: String? = nil) {
        notify(Method.log, LogParams(level: level, message: message, job: job))
    }
}

/// "12.3 ms", for log lines.
func ms(_ v: Double) -> String { String(format: "%.1f ms", v) }

/// Diagnostics on stderr (the app keeps them in helper.log).
func log(_ message: String) {
    FileHandle.standardError.write(Data("[helper] \(message)\n".utf8))
}

func rpcError(_ error: Error) -> RPCError {
    if let e = error as? RPCError { return e }
    if error is CancellationError { return RPCError(Code.cancelled, "Cancelled") }
    return RPCError(Code.internal, errorMessage(error))
}

/// Where a request's time went inside the helper, for the stderr line of a
/// slow one: the app's supervisor logs its own side (helper.Trace).
final class RequestClock: @unchecked Sendable {
    let received = monotonicMs()
    private let lock = NSLock()
    private var mainWait = 0.0
    private var mainHops = 0

    func addMainWait(_ ms: Double) {
        lock.withLock {
            mainWait += ms
            mainHops += 1
        }
    }

    var mainSummary: String {
        lock.withLock { mainHops == 0 ? "" : ", waited \(ms(mainWait)) for the main thread (\(mainHops) hops)" }
    }

    @TaskLocal static var current: RequestClock?
}

/// Runs body on the main thread (AppKit, NSPasteboard), adding the wait
/// for it to the request's clock: a busy main thread shows in the
/// slow-request line.
func onMain<T: Sendable>(_ body: @MainActor @Sendable () throws -> T) async rethrows -> T {
    let t0 = monotonicMs()
    let clock = RequestClock.current
    return try await MainActor.run {
        clock?.addMainWait(monotonicMs() - t0)
        return try body()
    }
}

/// The session: answers requests from stdin. Control requests each run in
/// a task of their own, so an app slow to answer Accessibility does not
/// hold up capture.stop; capture.start is a job that answers when it ends.
final class Server: @unchecked Sendable {
    let out: Writer
    let front: FrontApps
    let capture: CaptureManager
    let hotkeys: HotkeyTap
    let textio: TextIO
    let output: Output
    private let lock = NSLock()
    private var ready = false
    private var debug = false

    /// Requests slower than this are logged on stderr with where they waited.
    static let slowMs = 250.0

    init(out: Writer) {
        self.out = out
        front = FrontApps(out: out)
        capture = CaptureManager(out: out)
        hotkeys = HotkeyTap(out: out, front: front)
        textio = TextIO(front: front)
        output = Output(out: out)
    }

    /// Reads requests on a thread of its own until stdin closes, then
    /// exits: the app quit or crashed.
    func serve() {
        let thread = Thread { [self] in
            while let line = readLine(strippingNewline: true) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.isEmpty { continue }
                let clock = RequestClock()
                let msg: Incoming
                do {
                    msg = try Incoming.decode(Data(trimmed.utf8))
                } catch {
                    log("bad line: \(trimmed.prefix(200))")
                    continue
                }
                guard msg.kind == .request, let id = msg.id, let method = msg.method else {
                    log("ignored a message that is not a request")
                    continue
                }
                handle(id: id, method: method, msg, clock: clock)
            }
            log("stdin closed, exiting")
            capture.cancelAll()
            output.undo("the app went away")
            exit(0)
        }
        thread.name = "voxtnew.stdin"
        thread.start()
    }

    func handle(id: String, method: String, _ msg: Incoming, clock: RequestClock) {
        if method == Method.hello {
            return answerHello(id, msg)
        }
        guard lock.withLock({ ready }) else {
            return out.fail(id, RPCError(Code.notInitialized, "helper.hello must come first"))
        }
        guard Method.requests.contains(method) else {
            return out.fail(id, RPCError(Code.methodNotFound, "Unknown method \(method)"))
        }
        switch method {
        case Method.shutdown:
            capture.cancelAll()
            output.undo("helper.shutdown")
            out.result(id, Empty())
            exit(0)
        case Method.captureStart:
            // A job: it answers when the recording ends.
            Task {
                do {
                    try await capture.start(job: id, try msg.decodeParams(CaptureParams.self), clock: clock)
                } catch {
                    out.fail(id, rpcError(error))
                }
            }
            return
        default:
            break
        }
        Task {
            await RequestClock.$current.withValue(clock) {
                let started = monotonicMs()
                do {
                    let result = try await self.run(method, msg)
                    self.out.result(id, result)
                } catch {
                    self.out.fail(id, rpcError(error))
                }
                let done = monotonicMs()
                if done - clock.received >= Self.slowMs {
                    log("slow \(method) \(id): \(ms(done - clock.received)) (queued \(ms(started - clock.received)), ran \(ms(done - started))\(clock.mainSummary))")
                }
            }
        }
    }

    /// Runs a control request.
    private func run(_ method: String, _ msg: Incoming) async throws -> any Encodable & Sendable {
        switch method {
        case Method.jobCancel:
            let p = try msg.decodeParams(CancelParams.self)
            return CancelResult(cancelled: capture.cancel(job: p.job))
        case Method.permissionsStatus:
            return Permissions.status()
        case Method.permissionsRequest:
            return try await Permissions.request(try msg.decodeParams(PermissionsRequestParams.self).permission)
        case Method.captureDevices:
            return DevicesResult(devices: captureDevices())
        case Method.captureStop:
            let p = try msg.decodeParams(CaptureStopParams.self)
            guard !p.job.isEmpty else { throw RPCError(Code.badRequest, "Missing job") }
            return CaptureStopResult(stopped: capture.stop(job: p.job))
        case Method.hotkeyConfigure:
            return hotkeys.configure(try msg.decodeParams(HotkeyConfigureParams.self))
        case Method.hotkeyStatus:
            return hotkeys.status()
        case Method.appFrontmost:
            return await front.current()
        case Method.textFocus:
            return try await textio.focus()
        case Method.textSelection:
            return try await textio.selection(try msg.decodeParams(SelectionParams.self))
        case Method.textPaste:
            return try await textio.paste(try msg.decodeParams(PasteParams.self))
        case Method.appList:
            return await AppList.list(installed: try msg.decodeParams(AppListParams.self).installed ?? false)
        case Method.appIcons:
            return try AppIcons.icons(try msg.decodeParams(AppIconsParams.self))
        case Method.appConfigure:
            front.configure(try msg.decodeParams(AppConfigureParams.self))
            return Empty()
        case Method.browserURL:
            return await Browser.url(try msg.decodeParams(BrowserURLParams.self), front: front)
        case Method.outputWatch:
            return output.watch(try msg.decodeParams(OutputWatchParams.self))
        case Method.outputMute:
            return try output.claims.mute(try msg.decodeParams(OutputMuteParams.self))
        case Method.outputRelease:
            return output.claims.release(try msg.decodeParams(OutputReleaseParams.self))
        case Method.outputRecover:
            return output.claims.recover(try msg.decodeParams(OutputRecoverParams.self))
        default:
            throw RPCError(Code.methodNotFound, "Unknown method \(method)")
        }
    }

    private func answerHello(_ id: String, _ msg: Incoming) {
        let p: HelloParams
        do {
            p = try msg.decodeParams(HelloParams.self)
        } catch {
            return out.fail(id, rpcError(error))
        }
        guard p.protocol == protocolVersion else {
            return out.fail(id, RPCError(Code.protocolMismatch, "This helper speaks protocol \(protocolVersion)",
                                         data: .object(["supported": .array([.int(Int64(protocolVersion))])])))
        }
        lock.withLock {
            ready = true
            debug = p.logLevel == "debug"
        }
        front.start()
        output.start()
        let version = helperVersion()
        out.logLine("info", "voxtnew-helper \(version) ready (app \(p.app))")
        out.result(id, HelloResult(protocol: protocolVersion, version: version, pid: Int(getpid()), methods: Method.requests,
                                   device: Device(name: machineModel(), os: osVersion())))
    }
}

/// The helper's version (its Info.plist, kept in step with mygo.json).
func helperVersion() -> String {
    Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev"
}

/// "Mac14,2": the hardware model (sysctl hw.model).
func machineModel() -> String {
    var size = 0
    sysctlbyname("hw.model", nil, &size, nil, 0)
    guard size > 0 else { return "Mac" }
    var buf = [UInt8](repeating: 0, count: size)
    sysctlbyname("hw.model", &buf, &size, nil, 0)
    return String(decoding: buf.prefix { $0 != 0 }, as: UTF8.self)
}

/// "macOS 26.7" (with the patch version when there is one).
func osVersion() -> String {
    let v = ProcessInfo.processInfo.operatingSystemVersion
    return v.patchVersion == 0 ? "macOS \(v.majorVersion).\(v.minorVersion)" : "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
}
