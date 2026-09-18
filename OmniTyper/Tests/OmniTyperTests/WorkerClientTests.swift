// SPDX-License-Identifier: Apache-2.0
import Testing
import AVFoundation
@testable import OmniTyper

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
}
