import Darwin
import Foundation

@MainActor final class TailcatProcess {
    enum Failure: Error { case unavailable, pipeSetup, failed, timeout, replyTooLarge, invalidReply }
    var onUnexpectedExit: (() -> Void)?
    private let executableURL: URL
    private let startupTimeoutSeconds: Double
    private var child: Child?
    private var monitor: Task<Void, Never>?
    private let maximumMessageBytes = 64 * 1024

    init(executableURL: URL, startupTimeoutSeconds: Double = 20) {
        self.executableURL = executableURL
        self.startupTimeoutSeconds = startupTimeoutSeconds
    }

    isolated deinit { stop() }

    func start(configuration: TailcatConfiguration) async throws -> TailcatReady {
        stop()
        var request = try configuration.encoded()
        request.append(10)
        guard request.count <= maximumMessageBytes, startupTimeoutSeconds.isFinite, startupTimeoutSeconds > 0 else {
            throw Failure.unavailable
        }
        try Task.checkCancellation()
        let running = Child(executableURL: executableURL)
        child = running
        do {
            try running.launch()
            let deadline = ContinuousClock.now.advanced(by: .seconds(startupTimeoutSeconds))
            var sent = 0
            var output = Data()
            var buffer = [UInt8](repeating: 0, count: 4096)
            while true {
                guard child === running else { throw CancellationError() }
                try Task.checkCancellation()
                guard ContinuousClock.now < deadline else { throw Failure.timeout }
                if sent < request.count {
                    let written = request.withUnsafeBytes { bytes in
                        Darwin.write(running.input.fileDescriptor, bytes.baseAddress!.advanced(by: sent), request.count - sent)
                    }
                    if written > 0 { sent += written }
                    else if written < 0 && errno != EAGAIN && errno != EINTR { throw Failure.failed }
                }
                let count = Darwin.read(running.output.fileDescriptor, &buffer, buffer.count)
                if count > 0 {
                    output.append(contentsOf: buffer.prefix(count))
                    guard output.count <= maximumMessageBytes else { throw Failure.replyTooLarge }
                    if let newline = output.firstIndex(of: 10) {
                        guard sent == request.count,
                              let ready = try? TailcatReady(data: output.prefix(upTo: newline)),
                              running.process.isRunning else { throw Failure.invalidReply }
                        startMonitoring(running)
                        return ready
                    }
                } else if count == 0 { throw Failure.failed }
                else if errno != EAGAIN && errno != EINTR { throw Failure.failed }
                guard running.process.isRunning else { throw Failure.failed }
                try await Task.sleep(for: .milliseconds(10))
            }
        } catch {
            if child === running { stop() }
            else { running.close() }
            throw error
        }
    }

    func stop() {
        monitor?.cancel()
        monitor = nil
        child?.close()
        child = nil
    }

    nonisolated static func configureInputPipe(_ descriptor: Int32) throws {
        guard fcntl(descriptor, F_SETNOSIGPIPE, 1) == 0,
              fcntl(descriptor, F_SETFL, O_NONBLOCK) == 0 else { throw Failure.pipeSetup }
    }

    private func startMonitoring(_ running: Child) {
        monitor = Task { @MainActor [weak self, weak running] in
            while !Task.isCancelled {
                do { try await Task.sleep(for: .milliseconds(50)) }
                catch { return }
                guard let self, let running, self.child === running else { return }
                if !running.process.isRunning {
                    self.stop()
                    self.onUnexpectedExit?()
                    return
                }
            }
        }
    }

    @MainActor private final class Child {
        let process = Process()
        let input: FileHandle
        let output: FileHandle
        private let inputPipe = Pipe()
        private let outputPipe = Pipe()
        private var closed = false

        init(executableURL: URL) {
            input = inputPipe.fileHandleForWriting
            output = outputPipe.fileHandleForReading
            process.executableURL = executableURL
            process.environment = ["PATH": "/usr/bin:/bin", "TMPDIR": FileManager.default.temporaryDirectory.path]
            process.standardInput = inputPipe
            process.standardOutput = outputPipe
            process.standardError = FileHandle.nullDevice
        }

        func launch() throws {
            try TailcatProcess.configureInputPipe(input.fileDescriptor)
            guard fcntl(output.fileDescriptor, F_SETFL, O_NONBLOCK) == 0 else { throw Failure.pipeSetup }
            do { try process.run() }
            catch { throw Failure.unavailable }
            try? inputPipe.fileHandleForReading.close()
            try? outputPipe.fileHandleForWriting.close()
        }

        func close() {
            guard !closed else { return }
            closed = true
            try? input.close()
            try? output.close()
            try? inputPipe.fileHandleForReading.close()
            try? outputPipe.fileHandleForWriting.close()
            if process.isRunning {
                process.terminate()
                let ownedProcess = process
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(1))
                    if ownedProcess.isRunning { kill(ownedProcess.processIdentifier, SIGKILL) }
                }
            }
        }
    }
}
