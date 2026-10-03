// SPDX-License-Identifier: Apache-2.0
import Combine
import Darwin
import Foundation

struct WorkerFailure: LocalizedError {
    let message: String
    let rawText: String?
    var errorDescription: String? { message }
}

@MainActor
final class WorkerClient: ObservableObject {
    // Note (Jiaxin Deng): Resolve the default after the store loads the saved language.
    @Published private(set) var statusText: String?
    var status: String { statusText ?? L("worker.notLoaded") }

    private var showingReady = false
    @Published private(set) var isRunning = false
    // Note (Yifei Leng): Only a model download reports a fraction, so any other progress event clears it.
    @Published private(set) var downloadFraction: Double?
    // Note (Yifei Leng): The speech server lives and dies with this worker process.
    @Published private(set) var speechModelReady = false
    var isPreparingSpeech: Bool { pending?.preparesSpeech == true }

    private struct Pending {
        let id: String
        let preparesSpeech: Bool
        let continuation: CheckedContinuation<[String: Any], Error>
    }

    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var errors: FileHandle?
    private var pythonPath: String?
    private var generation = UUID()
    private var pending: Pending?
    private var stdoutBuffer = Data()
    private var stdoutEnded = false
    private var exitStatus: Int32?
    private var timeoutTask: Task<Void, Never>?
    private var exitTask: Task<Void, Never>?
    private let maximumLineBytes = 1_048_576
    // Note (Yucheng Hu): Long enough for the worker to stop its model server. Injectable so a test
    // can exercise the escalation to SIGKILL without waiting the production period out.
    private let gracePeriod: UInt64

    init(gracePeriod: UInt64 = 5_000_000_000) {
        self.gracePeriod = gracePeriod
    }

    func request(_ payload: [String: Any], python: String) async throws -> [String: Any] {
        let requestID = UUID().uuidString
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                do {
                    guard pending == nil else { throw Failure("worker.busy") }
                    var message = payload
                    message["id"] = requestID
                    guard JSONSerialization.isValidJSONObject(message) else {
                        throw Failure("worker.badRequest")
                    }
                    var data = try JSONSerialization.data(withJSONObject: message)
                    guard data.count < 256 * 1_024 else {
                        throw Failure("worker.tooLarge")
                    }
                    data.append(0x0A)
                    try ensureProcess(python: python)
                    guard let input else { throw Failure("worker.noInput") }
                    pending = Pending(id: requestID, preparesSpeech: payload["op"] as? String == "prepare",
                                      continuation: continuation)
                    statusText = payload["op"] as? String == "prepare" ? L("worker.preparing") : L("worker.processing")
                    showingReady = false
                    let workerGeneration = generation
                    let seconds: UInt64 = payload["op"] as? String == "prepare" ? 1_800 : 600
                    timeoutTask = Task { [weak self] in
                        do { try await Task.sleep(nanoseconds: seconds * 1_000_000_000) }
                        catch { return }
                        guard let self, self.pending?.id == requestID else { return }
                        self.failAndStop(Failure("worker.timedOut"))
                    }
                    // Note (Codex): A blocked worker must not block the UI thread writing its pipe.
                    DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                        do { try input.write(contentsOf: data) }
                        catch {
                            DispatchQueue.main.async {
                                guard let self, self.generation == workerGeneration, self.pending?.id == requestID else { return }
                                self.failAndStop(Failure("worker.notAccepting"))
                            }
                        }
                    }
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        } onCancel: {
            Task { @MainActor [weak self] in
                guard let self, self.pending?.id == requestID else { return }
                self.finish(.failure(CancellationError()))
                self.shutdown()
                self.statusText = L("worker.cancelled"); self.showingReady = false
            }
        }
    }

    func stop() {
        finish(.failure(CancellationError()))
        shutdown()
        statusText = nil; showingReady = false
    }

    private func ensureProcess(python: String) throws {
        let executable = try resolvePython(python)
        if let process, process.isRunning, pythonPath == executable.path { return }
        shutdown()
        let bundled = Bundle.main.resourceURL?.appendingPathComponent("backend/worker.py")
        let overridden = ProcessInfo.processInfo.environment["OMNITYPER_WORKER"].map {
            URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)
        }
        guard let worker = [bundled, overridden].compactMap({ $0 }).first(where: {
            FileManager.default.isReadableFile(atPath: $0.path)
        }) else {
            throw Failure("worker.missing")
        }
        let child = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        child.executableURL = executable
        child.arguments = ["-u", worker.path]
        child.currentDirectoryURL = worker.deletingLastPathComponent()
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        child.environment = environment
        child.standardInput = stdin
        child.standardOutput = stdout
        child.standardError = stderr
        let workerGeneration = UUID()
        generation = workerGeneration
        stdoutBuffer.removeAll(keepingCapacity: true)
        stdoutEnded = false
        exitStatus = nil
        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil }
            DispatchQueue.main.async {
                guard let self, self.generation == workerGeneration else { return }
                self.receive(data)
            }
        }
        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { handle.readabilityHandler = nil; return }
            DispatchQueue.main.async {
                guard let self, self.generation == workerGeneration else { return }
                // Note (Codex): Model logs may contain user content; record only their byte count.
                Diagnostics.record("worker.stderr", ["bytes": String(data.count)])
            }
        }
        child.terminationHandler = { [weak self] child in
            let code = child.terminationStatus
            let pid = child.processIdentifier
            let reason = child.terminationReason == .exit ? "exit" : "signal"
            DispatchQueue.main.async {
                guard let self, self.generation == workerGeneration else { return }
                self.exitStatus = code
                self.isRunning = false
                Diagnostics.record("worker.exited", [
                    "pid": String(pid), "reason": reason, "status": String(code),
                ])
                if self.stdoutEnded { self.handleExit() }
                else {
                    // Note (Codex): Process exit can arrive before the pipe's final response.
                    self.exitTask = Task { [weak self] in
                        try? await Task.sleep(nanoseconds: 1_000_000_000)
                        guard !Task.isCancelled, let self, self.generation == workerGeneration else { return }
                        self.handleExit()
                    }
                }
            }
        }
        do { try child.run() }
        catch {
            stdout.fileHandleForReading.readabilityHandler = nil
            stderr.fileHandleForReading.readabilityHandler = nil
            throw Failure("worker.pythonStart", executable.path)
        }
        process = child
        input = stdin.fileHandleForWriting
        output = stdout.fileHandleForReading
        errors = stderr.fileHandleForReading
        pythonPath = executable.path
        isRunning = true
        Diagnostics.record("worker.started", ["pid": String(child.processIdentifier)])
    }

    private func receive(_ data: Data) {
        if data.isEmpty {
            output?.readabilityHandler = nil
            stdoutEnded = true
            if exitStatus != nil { handleExit() }
            else if pending != nil {
                failAndStop(Failure("worker.closed"))
            }
            return
        }
        stdoutBuffer.append(data)
        while let newline = stdoutBuffer.firstIndex(of: 0x0A) {
            let line = stdoutBuffer.prefix(upTo: newline)
            guard line.count <= maximumLineBytes else {
                failAndStop(Failure("worker.oversized"))
                return
            }
            let complete = Data(line)
            stdoutBuffer.removeSubrange(...newline)
            if complete.isEmpty { continue }
            guard let message = (try? JSONSerialization.jsonObject(with: complete)) as? [String: Any],
                  let responseID = message["id"] as? String else {
                failAndStop(Failure("worker.invalid"))
                return
            }
            guard responseID == pending?.id else { continue }
            if message["event"] as? String == "progress" {
                if let progress = message["message"] as? String { statusText = String(progress.prefix(300)); showingReady = false }
                downloadFraction = (message["fraction"] as? Double).map { min(max($0, 0), 1) }
            } else if let ok = message["ok"] as? Bool {
                if ok {
                    if isPreparingSpeech { speechModelReady = true }
                    statusText = L("worker.ready"); showingReady = true
                    finish(.success(message))
                } else {
                    let description = message["error"] as? String ?? L("worker.failed")
                    statusText = L("worker.processFailed"); showingReady = false
                    finish(.failure(WorkerFailure(message: String(description.prefix(2_000)),
                                                  rawText: message["raw_text"] as? String)))
                }
            } else {
                failAndStop(Failure("worker.incomplete"))
                return
            }
        }
        if stdoutBuffer.count > maximumLineBytes {
            failAndStop(Failure("worker.oversized"))
        }
    }

    private func finish(_ result: Result<[String: Any], Error>) {
        downloadFraction = nil
        guard let pending else { return }
        self.pending = nil
        timeoutTask?.cancel()
        timeoutTask = nil
        pending.continuation.resume(with: result)
    }

    private func handleExit() {
        let code = exitStatus ?? -1
        if pending != nil {
            finish(.failure(Failure("worker.exited", String(code))))
            statusText = L("worker.stopped"); showingReady = false
        } else if showingReady { statusText = nil; showingReady = false }
        shutdown()
    }

    private func failAndStop(_ error: Error) {
        finish(.failure(error))
        shutdown()
        statusText = L("worker.stopped"); showingReady = false
    }

    private func shutdown() {
        generation = UUID()
        timeoutTask?.cancel()
        timeoutTask = nil
        exitTask?.cancel()
        exitTask = nil
        let childInput = input
        let childOutput = output
        let childErrors = errors
        input = nil
        output = nil
        errors = nil
        stdoutBuffer.removeAll(keepingCapacity: true)
        stdoutEnded = false
        exitStatus = nil
        isRunning = false
        speechModelReady = false
        pythonPath = nil
        guard let child = process else { return }
        process = nil
        child.terminationHandler = nil
        if child.isRunning {
            child.terminate()
        }
        // Note (Codex): Keep stdin open and drain output so EOF cannot race SIGTERM during cleanup.
        Task { [gracePeriod] in
            let killTask = Task {
                do { try await Task.sleep(nanoseconds: gracePeriod) }
                catch { return }
                if child.isRunning { Darwin.kill(child.processIdentifier, SIGKILL) }
            }
            await withCheckedContinuation { continuation in
                DispatchQueue.global(qos: .utility).async {
                    child.waitUntilExit()
                    continuation.resume()
                }
            }
            killTask.cancel()
            childOutput?.readabilityHandler = nil
            childErrors?.readabilityHandler = nil
            try? childInput?.close()
            Diagnostics.record("worker.stopped", [
                "pid": String(child.processIdentifier),
                "reason": child.terminationReason == .exit ? "exit" : "signal",
                "status": String(child.terminationStatus),
            ])
        }
    }

    private func resolvePython(_ supplied: String) throws -> URL {
        let expanded = (supplied.trimmingCharacters(in: .whitespacesAndNewlines) as NSString).expandingTildeInPath
        guard !expanded.isEmpty else { throw Failure("worker.choosePython") }
        if expanded.contains("/") {
            let url = URL(fileURLWithPath: expanded)
            guard FileManager.default.isExecutableFile(atPath: url.path) else {
                throw Failure("worker.pythonNotExecutable", url.path)
            }
            return url
        }
        let path = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"
        for directory in path.split(separator: ":") {
            let url = URL(fileURLWithPath: String(directory)).appendingPathComponent(expanded)
            if FileManager.default.isExecutableFile(atPath: url.path) { return url }
        }
        throw Failure("worker.pythonMissing", expanded)
    }

}
