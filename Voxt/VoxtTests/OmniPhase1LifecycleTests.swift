// OmniPhase1LifecycleTests.swift
// Lifecycle and fault acceptance for a local model on its Omni server.
//
// Opt-in: VOXT_RUN_MODEL_TESTS=1, VOXT_ASR_BACKEND=omni with VOXT_OMNI_RUNTIME
// (the native qwen3_asr_server binary), VOXT_MODEL_STORAGE_ROOT, and VOXT_LIFECYCLE_CLIPS (a directory with
// short.wav and long.wav, 16 kHz mono). VOXT_LIFECYCLE_REPO picks the model
// (Qwen3-ASR 0.6B 4-bit by default). VOXT_LIFECYCLE_OUT receives a JSONL
// record per round; VOXT_LIFECYCLE_ROUNDS defaults to 200.

import Darwin
import XCTest
@testable import Voxt

@MainActor
final class OmniPhase1LifecycleTests: XCTestCase {
    private let repo = ProcessInfo.processInfo.environment["VOXT_LIFECYCLE_REPO"]
        ?? "mlx-community/Qwen3-ASR-0.6B-4bit"

    /// The executable name of the server Voxt launches for `repo`.
    private var serverName: String {
        get throws {
            let kind = try XCTUnwrap(OmniASRBackend.modelKind(for: repo), "\(repo) has no Omni server")
            let configuration = try XCTUnwrap(OmniASRBackend.configuration(for: kind))
            return configuration.runtimeExecutable.lastPathComponent
        }
    }

    private struct Fixtures {
        let short: (samples: [Float], sampleRate: Double)
        let long: (samples: [Float], sampleRate: Double)
        let output: URL?
        let rounds: Int
    }

    private func fixtures() throws -> Fixtures {
        try ModelTestGate.requireEnabled("Omni phase 1 lifecycle")
        let environment = ProcessInfo.processInfo.environment
        guard OmniASRBackend.LaunchSettings(environment: environment) != nil else {
            throw XCTSkip("Set VOXT_ASR_BACKEND=omni and VOXT_OMNI_RUNTIME.")
        }
        guard let clips = environment["VOXT_LIFECYCLE_CLIPS"], !clips.isEmpty else {
            throw XCTSkip("Set VOXT_LIFECYCLE_CLIPS to a directory with short.wav and long.wav.")
        }
        let directory = URL(fileURLWithPath: clips, isDirectory: true)
        return Fixtures(
            short: try DebugAudioClipIO.loadMonoSamples(from: directory.appendingPathComponent("short.wav")),
            long: try DebugAudioClipIO.loadMonoSamples(from: directory.appendingPathComponent("long.wav")),
            output: environment["VOXT_LIFECYCLE_OUT"].map { URL(fileURLWithPath: $0) },
            rounds: Int(environment["VOXT_LIFECYCLE_ROUNDS"] ?? "") ?? 200
        )
    }

    private func makeManager() async throws -> MLXModelManager {
        ModelTestGate.configureStorageRoot(for: self)
        let manager = MLXModelManager(modelRepo: repo)
        _ = try await manager.refreshInstallation(repo: repo)
        guard manager.isModelDownloaded(repo: repo) else {
            throw XCTSkip("\(repo) is not installed under VOXT_MODEL_STORAGE_ROOT.")
        }
        return manager
    }

    // MARK: - Rounds

    /// Load, Final, idle unload; every round must leave no server process behind.
    func testLoadFinalUnloadRoundsLeaveNoProcesses() async throws {
        let fixtures = try fixtures()
        let manager = try await makeManager()
        let transcriber = MLXTranscriber(modelManager: manager)
        let records = try fixtures.output.map { try LifecycleRecords($0) }
        var failures = 0

        for round in 1...fixtures.rounds {
            let startedAt = ContinuousClock.now
            var record: [String: Any] = ["event": "round", "round": round]
            do {
                manager.beginActiveUse()
                let loaded = try await manager.loadModel()
                manager.endActiveUse()
                XCTAssertNotNil(loaded.omniRuntime, "round \(round) did not load an Omni runtime")
                record["load_ms"] = startedAt.duration(to: .now).lifecycleMilliseconds
                record["processes_loaded"] = ProcessTree.descendants().count

                let finalStartedAt = ContinuousClock.now
                let text = try await transcriber.transcribeBufferedResult(
                    samples: fixtures.short.samples,
                    sampleRate: fixtures.short.sampleRate
                )?.text ?? ""
                record["final_ms"] = finalStartedAt.duration(to: .now).lifecycleMilliseconds
                record["final_empty"] = text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty

                let unloadStartedAt = ContinuousClock.now
                manager.releaseLoadedModelIfIdle(reason: "lifecycle")
                let gone = await ProcessTree.waitUntilNoDescendants(timeoutSeconds: 10)
                record["unload_ms"] = unloadStartedAt.duration(to: .now).lifecycleMilliseconds
                record["processes_after_unload"] = ProcessTree.descendants().count
                if !gone || record["final_empty"] as? Bool == true { failures += 1 }
            } catch {
                failures += 1
                record["failure"] = String(describing: error)
                manager.releaseLoadedModelIfIdle(reason: "lifecycle-failure")
                _ = await ProcessTree.waitUntilNoDescendants(timeoutSeconds: 10)
            }
            record["footprint_bytes"] = ProcessTree.footprint(of: getpid())
            records?.write(record)
        }

        await manager.shutdownForApplicationTermination()
        let leftover = ProcessTree.descendants()
        records?.write(["event": "done", "rounds": fixtures.rounds, "failures": failures,
                        "leftover_processes": leftover.count])
        XCTAssertEqual(failures, 0)
        XCTAssertEqual(leftover, [])
    }

    // MARK: - Faults

    /// A server crash mid-Final fails that request, and the next load starts a
    /// fresh server that serves the next Final.
    func testServerKilledDuringFinalFailsThatRequestAndRecovers() async throws {
        let fixtures = try fixtures()
        let manager = try await makeManager()
        let transcriber = MLXTranscriber(modelManager: manager)
        manager.beginActiveUse()
        let firstLoaded = try await manager.loadModel()
        let firstRuntime = try XCTUnwrap(firstLoaded.omniRuntime)
        manager.endActiveUse()

        let final = Task { @MainActor in
            try await transcriber.transcribeBufferedResult(
                samples: fixtures.long.samples,
                sampleRate: fixtures.long.sampleRate
            )
        }
        try await Task.sleep(for: .milliseconds(400))
        let serverName = try serverName
        let servers = ProcessTree.descendants().filter {
            ProcessTree.commandLine(of: $0).contains(serverName)
        }
        XCTAssertFalse(servers.isEmpty, "no \(serverName) process found")
        servers.forEach { kill($0, SIGKILL) }

        let failedAt = ContinuousClock.now
        do {
            _ = try await final.value
            XCTFail("The Final should fail when its server dies")
        } catch {
            XCTAssertLessThan(failedAt.duration(to: .now), .seconds(15))
        }
        _ = await ProcessTree.waitUntilNoDescendants(timeoutSeconds: 15)
        let firstStillServing = await firstRuntime.isServing
        XCTAssertFalse(firstStillServing)

        let text = try await transcriber.transcribeBufferedResult(
            samples: fixtures.short.samples,
            sampleRate: fixtures.short.sampleRate
        )?.text ?? ""
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        let secondRuntime = try await manager.loadModel().omniRuntime
        XCTAssertFalse(secondRuntime === firstRuntime)

        await manager.shutdownForApplicationTermination()
        let gone = await ProcessTree.waitUntilNoDescendants(timeoutSeconds: 10)
        XCTAssertTrue(gone)
    }

    /// Giving up on a cold start stops the server that start was creating.
    func testCancellingAColdStartLeavesNoServer() async throws {
        _ = try fixtures()
        let manager = try await makeManager()
        let load = Task { @MainActor in try await manager.loadModel() }
        let started = await ProcessTree.waitForDescendants(timeoutSeconds: 10)
        XCTAssertTrue(started, "the cold start never spawned the server")
        // Note (Jiaxin Deng): the native server is ready about 150 ms after its process starts;
        // give up as soon as the process appears, inside that window.
        manager.cancelPendingModelLoadForApplicationTermination()
        load.cancel()
        let loaded = try? await load.value
        XCTAssertNil(loaded, "the load finished before it was cancelled, so no cold start was cancelled")
        let gone = await ProcessTree.waitUntilNoDescendants(timeoutSeconds: 5)
        XCTAssertTrue(gone, "processes left: \(ProcessTree.descendants())")
        await manager.shutdownForApplicationTermination()
    }

    /// Termination while a Final runs returns, and leaves no process behind.
    func testTerminationDuringAFinalLeavesNoProcesses() async throws {
        let fixtures = try fixtures()
        let manager = try await makeManager()
        let transcriber = MLXTranscriber(modelManager: manager)
        manager.beginActiveUse()
        _ = try await manager.loadModel()
        manager.endActiveUse()

        let final = Task { @MainActor in
            try? await transcriber.transcribeBufferedResult(
                samples: fixtures.long.samples,
                sampleRate: fixtures.long.sampleRate
            )
        }
        try await Task.sleep(for: .milliseconds(300))
        let shutdownStartedAt = ContinuousClock.now
        await manager.shutdownForApplicationTermination()
        _ = await final.value
        XCTAssertLessThan(shutdownStartedAt.duration(to: .now), .seconds(30))
        let gone = await ProcessTree.waitUntilNoDescendants(timeoutSeconds: 5)
        XCTAssertTrue(gone, "processes left: \(ProcessTree.descendants())")
    }

    /// Cancelling a live session mid-stream leaves the server idle, so the Final
    /// that follows does not wait behind a stale live decode.
    func testCancelledLiveSessionLeavesTheServerIdle() async throws {
        guard OmniASRBackend.modelKind(for: repo) == .qwen3ASR else {
            throw XCTSkip("Only Qwen3-ASR has live sessions on its Omni server.")
        }
        let fixtures = try fixtures()
        let manager = try await makeManager()
        let transcriber = MLXTranscriber(modelManager: manager)
        manager.beginActiveUse()
        let loaded = try await manager.loadModel()
        let runtime = try XCTUnwrap(loaded.omniRuntime)

        let session = try await transcriber.makeMeetingNativeStreamingConfiguration().session
        let samples = fixtures.long.samples
        let chunk = 1600
        var fed = 0
        let feedUntil = min(samples.count, 16000 * 4)
        while fed < feedUntil {
            session.feedAudio(samples: Array(samples[fed..<min(fed + chunk, feedUntil)]))
            fed += chunk
            try await Task.sleep(for: .milliseconds(100))
        }
        session.cancel()

        let endpoint = try await runtime.beginUse()
        var idle = false
        let deadline = ContinuousClock.now + .seconds(3)
        while ContinuousClock.now < deadline {
            let health = try await JSONHealth.fetch(endpoint.baseURL.appendingPathComponent("health"))
            if (health["request_states"] as? [String: Any])?.isEmpty ?? false {
                idle = true
                break
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        // Released before shutdown, which waits for every lease.
        await runtime.endUse()
        XCTAssertTrue(idle, "the server still has live requests after the session was cancelled")

        let text = try await transcriber.transcribeBufferedResult(
            samples: fixtures.short.samples,
            sampleRate: fixtures.short.sampleRate
        )?.text ?? ""
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        manager.endActiveUse()
        await manager.shutdownForApplicationTermination()
    }
}

private enum JSONHealth {
    static func fetch(_ url: URL) async throws -> [String: Any] {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        let (data, _) = try await URLSession(configuration: configuration).data(from: url)
        return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }
}

private extension Duration {
    var lifecycleMilliseconds: Int {
        let parts = components
        return Int(parts.seconds * 1000 + parts.attoseconds / 1_000_000_000_000_000)
    }
}

private final class LifecycleRecords {
    private let handle: FileHandle

    init(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
    }

    deinit {
        try? handle.close()
    }

    func write(_ record: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) else { return }
        handle.write(data)
        handle.write(Data("\n".utf8))
    }
}

/// Processes this test process started, directly or not.
private enum ProcessTree {
    static func descendants(of root: pid_t = getpid()) -> [pid_t] {
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_ALL, 0]
        var size = 0
        guard sysctl(&mib, 4, nil, &size, nil, 0) == 0 else { return [] }
        let stride = MemoryLayout<kinfo_proc>.stride
        var procs = [kinfo_proc](repeating: kinfo_proc(), count: size / stride + 32)
        size = procs.count * stride
        guard sysctl(&mib, 4, &procs, &size, nil, 0) == 0 else { return [] }
        var children: [pid_t: [pid_t]] = [:]
        for proc in procs.prefix(size / stride) where proc.kp_proc.p_stat != SZOMB {
            children[proc.kp_eproc.e_ppid, default: []].append(proc.kp_proc.p_pid)
        }
        var result: [pid_t] = []
        var queue = children[root] ?? []
        while let pid = queue.popLast() {
            result.append(pid)
            queue += children[pid] ?? []
        }
        return result.sorted()
    }

    static func waitUntilNoDescendants(timeoutSeconds: Double) async -> Bool {
        let deadline = ContinuousClock.now + .seconds(timeoutSeconds)
        while !descendants().isEmpty {
            guard ContinuousClock.now < deadline else { return false }
            try? await Task.sleep(for: .milliseconds(50))
        }
        return true
    }

    static func waitForDescendants(timeoutSeconds: Double) async -> Bool {
        let deadline = ContinuousClock.now + .seconds(timeoutSeconds)
        while descendants().isEmpty {
            guard ContinuousClock.now < deadline else { return false }
            try? await Task.sleep(for: .milliseconds(50))
        }
        return true
    }

    static func commandLine(of pid: pid_t) -> String {
        var mib: [Int32] = [CTL_KERN, KERN_PROCARGS2, pid]
        var size = 0
        guard sysctl(&mib, 3, nil, &size, nil, 0) == 0, size > 0 else { return "" }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctl(&mib, 3, &buffer, &size, nil, 0) == 0 else { return "" }
        return String(decoding: buffer.prefix(size).dropFirst(MemoryLayout<Int32>.size).map { $0 == 0 ? 32 : $0 }, as: UTF8.self)
    }

    static func footprint(of pid: pid_t) -> UInt64 {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
            }
        }
        return status == 0 ? info.ri_phys_footprint : 0
    }
}
