// SPDX-License-Identifier: Apache-2.0
import Testing
import AVFoundation
import Darwin
@testable import OmniTyper

@Suite(.serialized)
struct WorkerClientTests {
    @Test func testAudioResamplingAndRecordingLimit() throws {
        final class Packets: @unchecked Sendable {
            let lock = NSLock()
            var bytes = 0
            var first = Data()
            func receive(_ data: Data) {
                lock.withLock {
                    if first.isEmpty { first = data }
                    bytes += data.count
                }
            }
        }
        let packets = Packets()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let inputFormat = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                                    sampleRate: 48_000, channels: 2, interleaved: false))
        let input = try #require(AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: 48_000))
        input.frameLength = 48_000
        let channels = try #require(input.floatChannelData)
        for frame in 0..<48_000 {
            channels[0][frame] = 0.25
            channels[1][frame] = 0.25
        }
        let sink = try AudioCaptureSink(input: inputFormat, url: url, onPCM: { packets.receive($0) })
        for _ in 0..<302 { sink.consume(input) }
        try sink.close()
        let file = try AVAudioFile(forReading: url)
        #expect(file.fileFormat.sampleRate == 16_000)
        #expect(file.fileFormat.channelCount == 1)
        #expect(file.fileFormat.commonFormat == .pcmFormatInt16)
        #expect(file.length == 300 * 16_000, "The WAV must never exceed the worker's 300-second limit")
        #expect(packets.bytes == Int(file.length) * 2, "Streaming and WAV must contain the same number of samples")
        let pcmSample = Int(packets.first[1024]) | Int(packets.first[1025]) << 8
        #expect(abs(pcmSample - 8192) < 328, "Streaming packets must be little-endian PCM16 at the resampled amplitude")
        let decoded = try #require(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 1_024))
        try file.read(into: decoded)
        let sample = try #require(decoded.floatChannelData)[0][512]
        #expect(abs(sample - 0.25) < 0.01)
    }

    @Test @MainActor
    func testFramingLifecycleAndCancellation() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let script = directory.appendingPathComponent("worker.py")
        let oldWorker = ProcessInfo.processInfo.environment["OMNITYPER_WORKER"]
        let python = ProcessInfo.processInfo.environment["OMNITYPER_TEST_PYTHON"] ?? "/usr/bin/python3"
        guard FileManager.default.isExecutableFile(atPath: python) else {
            Issue.record("Set OMNITYPER_TEST_PYTHON to a Python 3 executable.")
            return
        }
        try #"""
        import json, os, sys, time
        assert sys.dont_write_bytecode, "The worker must not modify the signed app bundle"
        serial = 0
        for line in sys.stdin:
            request = json.loads(line)
            serial += 1
            op = request['op']
            if op == 'crash':
                os._exit(17)
            if op == 'invalid':
                print('{bad json', flush=True)
                continue
            if op == 'oversized':
                os.write(1, b'x' * (2 * 1024 * 1024))
                continue
            if op == 'sleep':
                print(json.dumps(dict(id=request['id'], event='progress', message='waiting')), flush=True)
                time.sleep(30)
            if op == 'failure':
                print(json.dumps(dict(id=request['id'], ok=False, error='Model unavailable')), flush=True)
                continue
            os.write(2, b'PRIVATE_TRANSCRIPT_DO_NOT_LOG\n')
            progress = json.dumps(dict(id=request['id'], event='progress', message='loading')).encode() + b'\n'
            final = json.dumps(dict(id=request['id'], ok=True, text='你好 café',
                                    serial=serial, worker_pid=os.getpid()), ensure_ascii=False).encode() + b'\n'
            cut = final.index('你'.encode()) + 1
            os.write(1, progress + final[:cut])
            time.sleep(0.02)
            os.write(1, final[cut:])
            if op == 'final_exit':
                sys.exit(0)
        """#.write(to: script, atomically: true, encoding: .utf8)
        setenv("OMNITYPER_WORKER", script.path, 1)
        let client = WorkerClient()
        defer {
            client.stop()
            if let oldWorker { setenv("OMNITYPER_WORKER", oldWorker, 1) }
            else { unsetenv("OMNITYPER_WORKER") }
            try? FileManager.default.removeItem(at: directory)
        }

        do {
            _ = try await client.request(["op": "echo", "text": String(repeating: "汉", count: 90_000)], python: python)
            Issue.record("Requests over the worker's 256 KiB limit must fail before sending")
        } catch { #expect(error.localizedDescription.contains("256 KiB")) }
        #expect(!client.isRunning)

        let first = try await client.request(["op": "echo"], python: python)
        #expect(first["text"] as? String == "你好 café")
        #expect(client.isRunning)
        let second = try await client.request(["op": "echo"], python: python)
        #expect(first["worker_pid"] as? Int == second["worker_pid"] as? Int)
        #expect(second["serial"] as? Int == 2)
        #expect(client.status == "Ready")

        do {
            _ = try await client.request(["op": "failure"], python: python)
            Issue.record("Worker error responses must fail the request")
        } catch { #expect(error.localizedDescription == "Model unavailable") }
        #expect(client.isRunning, "An ordinary model failure keeps the protocol usable")
        _ = try await client.request(["op": "echo"], python: python)

        for operation in ["invalid", "oversized", "crash"] {
            do {
                _ = try await client.request(["op": operation], python: python)
                Issue.record("\(operation) must fail the request")
            } catch { #expect(!(error is CancellationError)) }
            #expect(!client.isRunning)
            _ = try await client.request(["op": "echo"], python: python)
        }

        let waiting = Task { try await client.request(["op": "sleep"], python: python) }
        for _ in 0..<100 {
            if client.status == "waiting" { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        #expect(client.status == "waiting")
        do {
            _ = try await client.request(["op": "echo"], python: python)
            Issue.record("Concurrent requests must be rejected")
        } catch { #expect(error.localizedDescription.contains("busy")) }
        waiting.cancel()
        do {
            _ = try await waiting.value
            Issue.record("Cancellation must resume the outstanding request")
        } catch { #expect(error is CancellationError) }
        #expect(!client.isRunning)
        let restarted = try await client.request(["op": "echo"], python: python)
        #expect(restarted["serial"] as? Int == 1)

        let final = try await client.request(["op": "final_exit"], python: python)
        #expect(final["text"] as? String == "你好 café", "A final response must survive an immediate worker exit")
        for _ in 0..<100 {
            if !client.isRunning { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        #expect(!client.isRunning)

        let log = try #require(Diagnostics.fileURL)
        #expect(log.path.hasPrefix(FileManager.default.temporaryDirectory.path),
                "a test run must not append to the real log")
        let diagnostics = try String(contentsOf: log, encoding: .utf8)
        #expect(!diagnostics.contains("PRIVATE_TRANSCRIPT_DO_NOT_LOG"))
        #expect(diagnostics.utf8.count <= 128 * 1024)
        #expect(diagnostics.contains("worker.stderr"), "the worker's own events belong in the same log")
    }

    @Test @MainActor
    func testShutdownAllowsCleanup() async throws {
        guard let harness = try ShutdownHarness() else { return }
        defer { harness.tearDown() }
        let client = harness.client
        for cancelRequest in [false, true] {
            let started = try await harness.start("ready")
            let childPID = try #require(started.child)
            if cancelRequest {
                let waiting = Task { try await client.request(["op": "sleep"], python: harness.python) }
                #expect(try await harness.waitUntil { client.status == "waiting" })
                waiting.cancel()
                do {
                    _ = try await waiting.value
                    Issue.record("Cancellation must finish the outstanding request")
                } catch { #expect(error is CancellationError) }
            } else {
                client.stop()
            }
            // Note (Yucheng Hu): A second stop must be a no-op; cancel followed by quit does this in the app.
            client.stop()

            // Note (Codex): The old worker's cleanup output must not affect its replacement.
            let replacement = try await harness.start("ready")
            let replacementChildPID = try #require(replacement.child)
            #expect(replacement.worker != started.worker)
            let marker = harness.directory.appendingPathComponent("\(started.worker).json")
            #expect(try await harness.waitUntil { FileManager.default.fileExists(atPath: marker.path) })
            let cleanup = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: marker)) as? [String: Any])
            // Note (Yucheng Hu): Asserts the precondition, not the outcome. The stub reaps its descendant on
            // every path, so a test that only checked the outcome would pass against the old ordering too.
            #expect(cleanup["stopped_by"] as? Int == Int(SIGTERM),
                    "Shutdown must not give the worker a second exit trigger while it is cleaning up")
            #expect(cleanup["stdin_open"] as? Bool == true, "EOF must not race SIGTERM or interrupt cleanup")
            #expect(cleanup["stdout_bytes"] as? Int == 131_072, "The parent must keep draining stdout past one pipe buffer")
            #expect(cleanup["stderr_bytes"] as? Int == 131_072, "The parent must keep draining stderr past one pipe buffer")
            #expect(cleanup["error"] == nil)
            #expect(try await harness.waitUntil { Darwin.kill(started.worker, 0) != 0 })
            #expect(Darwin.kill(childPID, 0) != 0, "The worker must finish reaping its descendant")
            #expect(client.isRunning, "An old worker's exit must not stop its replacement")
            client.stop()
            #expect(try await harness.waitUntil { Darwin.kill(replacement.worker, 0) != 0 })
            #expect(Darwin.kill(replacementChildPID, 0) != 0)
        }
    }

    @Test @MainActor
    func testShutdownKillsUnresponsiveWorker() async throws {
        guard let harness = try ShutdownHarness(gracePeriod: 500_000_000) else { return }
        defer { harness.tearDown() }
        let unresponsive = try await harness.start("ignore_term")
        let stoppedAt = Date()
        harness.client.stop()
        try await Task.sleep(nanoseconds: 100_000_000)
        #expect(Darwin.kill(unresponsive.worker, 0) == 0, "Allow the worker its graceful shutdown interval")
        #expect(try await harness.waitUntil { Darwin.kill(unresponsive.worker, 0) != 0 },
                "A worker ignoring SIGTERM must still be killed after the grace period")
        #expect(Date().timeIntervalSince(stoppedAt) >= 0.4)
    }
}

/// Drives a stand-in worker that records what it observed while shutting down,
/// and reaps everything it spawned however the test ends.
@MainActor
private final class ShutdownHarness {
    let client: WorkerClient
    let directory: URL
    let python: String
    private let oldWorker = ProcessInfo.processInfo.environment["OMNITYPER_WORKER"]
    private var workerPIDs: [Int32] = []
    private var childPIDs: [Int32] = []

    init?(gracePeriod: UInt64 = 5_000_000_000) throws {
        client = WorkerClient(gracePeriod: gracePeriod)
        python = ProcessInfo.processInfo.environment["OMNITYPER_TEST_PYTHON"] ?? "/usr/bin/python3"
        guard FileManager.default.isExecutableFile(atPath: python) else {
            Issue.record("Set OMNITYPER_TEST_PYTHON to a Python 3 executable.")
            return nil
        }
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let script = directory.appendingPathComponent("worker.py")
        try Self.source.write(to: script, atomically: true, encoding: .utf8)
        setenv("OMNITYPER_WORKER", script.path, 1)
    }

    /// Sends one request and records the worker and descendant that answered it.
    func start(_ op: String) async throws -> (worker: Int32, child: Int32?) {
        let reply = try await client.request(["op": op], python: python)
        let worker = Int32(try #require(reply["worker_pid"] as? Int))
        let child = (reply["child_pid"] as? Int).map { Int32($0) }
        workerPIDs.append(worker)
        if let child { childPIDs.append(child) }
        return (worker, child)
    }

    func waitUntil(_ condition: () -> Bool, attempts: Int = 150) async throws -> Bool {
        for _ in 0..<attempts {
            if condition() { return true }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        return condition()
    }

    func tearDown() {
        client.stop()
        for pid in workerPIDs where Darwin.kill(pid, 0) == 0 { Darwin.kill(pid, SIGKILL) }
        for pid in childPIDs where Darwin.kill(pid, 0) == 0 { Darwin.kill(-pid, SIGKILL) }
        if let oldWorker { setenv("OMNITYPER_WORKER", oldWorker, 1) }
        else { unsetenv("OMNITYPER_WORKER") }
        try? FileManager.default.removeItem(at: directory)
    }

    private static let source = #"""
    import json, os, pathlib, select, signal, subprocess, sys, time

    child = None
    stopped_by = None

    def terminate(signum: int, frame: object) -> None:
        global stopped_by
        stopped_by = signum
        raise SystemExit(0)

    def stdin_is_open() -> bool:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return not readable or os.read(0, 1) != b''

    signal.signal(signal.SIGTERM, terminate)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request['op'] == 'ignore_term':
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            elif child is None:
                child = subprocess.Popen(
                    [sys.executable, '-c', 'import time; time.sleep(60)'],
                    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if request['op'] == 'sleep':
                print(json.dumps(dict(id=request['id'], event='progress', message='waiting')), flush=True)
                time.sleep(60)
            print(json.dumps(dict(id=request['id'], ok=True, worker_pid=os.getpid(),
                                  child_pid=child.pid if child else None)), flush=True)
    finally:
        # Note (Yucheng Hu): The handler also runs if SIGTERM lands during cleanup; snapshot now so only a loop ended by SIGTERM counts.
        result = dict(stopped_by=stopped_by, stdin_open=False, stdout_bytes=0, stderr_bytes=0)
        try:
            # Note (Yucheng Hu): Give the parent time to close stdin late, so a close that merely trails SIGTERM still fails.
            time.sleep(0.15)
            result['stdin_open'] = stdin_is_open()
            # Note (Yucheng Hu): Twice the largest pipe buffer, so a worker whose reader was detached blocks here.
            for descriptor, field in ((1, 'stdout_bytes'), (2, 'stderr_bytes')):
                for _ in range(16):
                    result[field] += os.write(descriptor, b'x' * 8192)
            result['stdin_open'] = result['stdin_open'] and stdin_is_open()
        except OSError as error:
            result['error'] = str(error)
        finally:
            if child is not None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=2)
            marker = pathlib.Path(__file__).with_name(f'{os.getpid()}.json')
            marker.write_text(json.dumps(result))
    """#
}
