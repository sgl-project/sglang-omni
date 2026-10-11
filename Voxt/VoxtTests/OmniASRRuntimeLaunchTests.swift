// OmniASRRuntimeLaunchTests.swift
// Covers how the Omni runtime starts its native server process.

import XCTest
@testable import Voxt

final class OmniASRRuntimeLaunchTests: XCTestCase {
    func testRuntimeEnvironmentDropsInjectedDynamicLoaderVariables() {
        let inherited = [
            "PATH": "/usr/bin",
            "HOME": "/Users/someone",
            "DYLD_INSERT_LIBRARIES": "/Xcode/libXCTestBundleInject.dylib",
            "DYLD_LIBRARY_PATH": "/Xcode/usr/lib",
            "DYLD_FRAMEWORK_PATH": "/Xcode/Frameworks",
            "__XPC_DYLD_LIBRARY_PATH": "/Xcode/usr/lib",
        ]

        let environment = OmniASRRuntime.runtimeEnvironment(inheriting: inherited)

        XCTAssertEqual(environment, ["PATH": "/usr/bin", "HOME": "/Users/someone"])
    }

    func testLaunchSettingsNeedTheOmniBackendAndARuntimePath() {
        XCTAssertNil(OmniASRBackend.LaunchSettings(environment: ["VOXT_OMNI_RUNTIME": "/opt/qwen3_asr_server"]))
        XCTAssertNil(OmniASRBackend.LaunchSettings(environment: ["VOXT_ASR_BACKEND": "omni"]))
        XCTAssertNil(OmniASRBackend.LaunchSettings(environment: ["VOXT_ASR_BACKEND": "omni", "VOXT_OMNI_RUNTIME": ""]))
        let settings = OmniASRBackend.LaunchSettings(environment: [
            "VOXT_ASR_BACKEND": "omni",
            "VOXT_OMNI_RUNTIME": "/opt/qwen3_asr_server",
        ])
        XCTAssertEqual(settings?.runtimeExecutable, URL(fileURLWithPath: "/opt/qwen3_asr_server"))
    }

    func testEachKindRunsItsOwnServerBesideTheQwenRuntime() {
        let qwenRuntime = URL(fileURLWithPath: "/opt/voxt/bin/qwen3_asr_server")

        XCTAssertEqual(OmniASRBackend.runtimeExecutable(for: .qwen3ASR, qwenRuntime: qwenRuntime), qwenRuntime)
        XCTAssertEqual(OmniASRBackend.runtimeExecutable(for: .sileroVAD, qwenRuntime: qwenRuntime), qwenRuntime)
        XCTAssertEqual(OmniASRBackend.runtimeExecutable(for: .sortformer, qwenRuntime: qwenRuntime), qwenRuntime)
        XCTAssertEqual(
            OmniASRBackend.runtimeExecutable(for: .whisper, qwenRuntime: qwenRuntime).path,
            "/opt/voxt/bin/whisper_server"
        )
        XCTAssertEqual(OmniASRBackend.modelKindsByRepo["mlx-community/whisper-large-v3-turbo"], .whisper)
        XCTAssertEqual(OmniASRBackend.modelKindsByRepo["mlx-community/whisper-large-v3-mlx"], .whisper)
        XCTAssertEqual(OmniASRBackend.modelKindsByRepo["mlx-community/whisper-small-mlx"], .whisper)
        XCTAssertEqual(
            OmniASRBackend.runtimeExecutable(for: .cohereTranscribe, qwenRuntime: qwenRuntime).path,
            "/opt/voxt/bin/cohere_transcribe_server"
        )
        XCTAssertEqual(OmniASRBackend.modelKindsByRepo["beshkenadze/cohere-transcribe-03-2026-mlx-fp16"], .cohereTranscribe)
        XCTAssertEqual(
            OmniASRBackend.runtimeExecutable(for: .mossTranscribeDiarize, qwenRuntime: qwenRuntime).path,
            "/opt/voxt/bin/moss_transcribe_diarize_server"
        )
        XCTAssertEqual(
            OmniASRBackend.runtimeExecutable(for: .nemotronASR, qwenRuntime: qwenRuntime).path,
            "/opt/voxt/bin/nemotron_asr_server"
        )
        XCTAssertEqual(
            OmniASRBackend.modelKindsByRepo["mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"],
            .nemotronASR
        )
    }

    /// Every Qwen3-ASR checkpoint the native runtime is checked against runs on it;
    /// other sizes and quantizations keep the Swift backend.
    func testQwen3ASRCheckpointsWithGoldenOutputsRunOnTheNativeRuntime() {
        XCTAssertEqual(OmniASRBackend.modelKindsByRepo.filter { $0.value == .qwen3ASR }, [
            "mlx-community/Qwen3-ASR-0.6B-4bit": .qwen3ASR,
            "mlx-community/Qwen3-ASR-1.7B-6bit": .qwen3ASR,
            "mlx-community/Qwen3-ASR-1.7B-8bit": .qwen3ASR,
        ])
    }

    /// The runtime binary is started directly in supervised mode, not through Python.
    func testLaunchRunsTheRuntimeInSupervisedMode() async throws {
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-omni-launch-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let runtimeExecutable = scratch.appendingPathComponent("qwen3_asr_server")
        let arguments = scratch.appendingPathComponent("arguments")
        let environment = scratch.appendingPathComponent("environment")
        let script = """
        #!/bin/sh
        printf '%s\\n' "$0" "$@" > '\(arguments.path)'
        env > '\(environment.path)'
        echo '{"event": "failed", "reason": "recorded"}'
        """
        try script.write(to: runtimeExecutable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: runtimeExecutable.path)
        let modelDirectory = scratch.appendingPathComponent("model", isDirectory: true)
        let configuration = OmniBackendConfiguration(
            runtimeExecutable: runtimeExecutable,
            startupTimeoutSeconds: 42
        )
        let runtime = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: modelDirectory, configuration: configuration)

        do {
            _ = try await runtime.prepare()
            XCTFail("the recording runtime never reports ready")
        } catch {
            XCTAssertEqual(error as? OmniASRRuntimeError, .launchFailed("recorded"))
        }
        await runtime.retire()

        let commandLine = try String(contentsOf: arguments, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: false)
            .dropLast()
            .map(String.init)
        XCTAssertEqual(commandLine, [
            runtimeExecutable.path,
            "--supervised",
            "--model-kind", "qwen3_asr",
            "--model-directory", modelDirectory.path,
            "--startup-timeout-s", "42.0",
        ])
        let variables = try String(contentsOf: environment, encoding: .utf8)
            .split(separator: "\n")
            .compactMap { $0.split(separator: "=", maxSplits: 1).first.map(String.init) }
        XCTAssertFalse(variables.contains("PYTHONPATH"))
        XCTAssertFalse(variables.contains("PYTHONUNBUFFERED"))
        XCTAssertEqual(variables.filter { $0.hasPrefix("DYLD_") || $0.hasPrefix("__XPC_DYLD_") }, [])
    }

    func testAServerThatNeverReportsFailsAtTheStartupDeadline() async throws {
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-omni-deadline-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let runtimeExecutable = scratch.appendingPathComponent("qwen3_asr_server")
        try "#!/bin/sh\nexec sleep 60\n".write(to: runtimeExecutable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: runtimeExecutable.path)
        let configuration = OmniBackendConfiguration(
            runtimeExecutable: runtimeExecutable,
            startupTimeoutSeconds: 0.5
        )
        let runtime = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch, configuration: configuration)

        let startedAt = ContinuousClock.now
        do {
            _ = try await runtime.prepare()
            XCTFail("the server never reports ready")
        } catch {
            XCTAssertEqual(
                error as? OmniASRRuntimeError,
                .launchFailed("the server did not report ready within 0.5 s")
            )
        }
        XCTAssertLessThan(startedAt.duration(to: .now), .seconds(10))
        await runtime.retire()
    }

    func testDiagnosticTailKeepsOnlyTheEndOfLongOutput() {
        let tail = OmniDiagnosticTail(limit: 8)
        tail.append(Data("0123456789".utf8))
        tail.append(Data("ab".utf8))

        XCTAssertEqual(tail.text, "456789ab")
    }

    func testLineBufferJoinsLinesSplitAcrossReads() {
        let buffer = OmniLineBuffer()
        XCTAssertEqual(buffer.append(Data("{\"event\":".utf8)), [])
        XCTAssertEqual(buffer.append(Data("\"ready\"}\n\n{\"a\":1}\n{\"b".utf8)),
                       [Data("{\"event\":\"ready\"}".utf8), Data("{\"a\":1}".utf8)])
        XCTAssertEqual(buffer.append(Data("\":2}\n".utf8)), [Data("{\"b\":2}".utf8)])
    }

    /// Servers that print nothing must not hold Swift concurrency threads: with
    /// a blocking read per server, a few live servers starved every task.
    func testIdleEventStreamsLeaveConcurrencyThreadsFree() throws {
        let pipes = (0 ..< 64).map { _ in Pipe() }
        let readers = pipes.map { pipe in
            Task.detached {
                var iterator = OmniASRRuntime.eventStream(pipe.fileHandleForReading).makeAsyncIterator()
                return try await iterator.next()?["event"] as? String
            }
        }
        // Note (Jiaxin Deng): let every reader block on its pipe before the probe task runs.
        Thread.sleep(forTimeInterval: 0.5)
        let unrelated = expectation(description: "an unrelated task runs")
        Task.detached { unrelated.fulfill() }
        wait(for: [unrelated], timeout: 5)

        try pipes[0].fileHandleForWriting.write(contentsOf: Data("{\"event\":\"ready\"}\n".utf8))
        let first = expectation(description: "the first stream reads its event")
        Task.detached {
            let event = try await readers[0].value
            XCTAssertEqual(event, "ready")
            first.fulfill()
        }
        wait(for: [first], timeout: 5)
        for pipe in pipes {
            try pipe.fileHandleForWriting.close()
        }
    }
}

@MainActor
final class OmniModelManagerRecoveryTests: XCTestCase {
    private actor LoadCounter {
        private(set) var value = 0

        func increment() {
            value += 1
        }
    }

    /// A server that crashed or was killed must not keep failing every request
    /// until the idle unload: the next load replaces the runtime.
    func testALoadedOmniRuntimeThatStoppedServingIsReplacedOnTheNextLoad() async throws {
        let scratch = FileManager.default.temporaryDirectory
        let configuration = OmniBackendConfiguration(
            runtimeExecutable: URL(fileURLWithPath: "/usr/bin/false")
        )
        let loads = LoadCounter()
        let manager = MLXModelManager(modelRepo: "mlx-community/Qwen3-ASR-0.6B-4bit") { _ in
            await loads.increment()
            // Never launched, so never serving: the same answer a dead server gives.
            let runtime = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch, configuration: configuration)
            return MLXLoadedModelBox(loaded: .omni(runtime))
        }

        let first = try await manager.loadModel()
        let second = try await manager.loadModel()

        XCTAssertFalse(first.omniRuntime === second.omniRuntime)
        let loadCount = await loads.value
        XCTAssertEqual(loadCount, 2)
        await manager.shutdownForApplicationTermination()
    }
}
