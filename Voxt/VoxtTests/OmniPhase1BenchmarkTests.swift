// OmniPhase1BenchmarkTests.swift
// Measures the local Qwen dictation path for the Omni phase 1 A/A/B protocol.
//
// Opt-in: runs only with VOXT_RUN_MODEL_TESTS=1 and VOXT_BENCH_OUT set. The
// same file runs unchanged on the upstream build and on this branch, with or
// without VOXT_ASR_BACKEND=omni, so every arm is measured by identical code.
//
//   VOXT_MODEL_STORAGE_ROOT   model storage root holding mlx-audio/<repo>
//   VOXT_BENCH_MANIFEST       manifest.jsonl (id, stratum, lang, duration)
//   VOXT_BENCH_CLIPS          directory of <id>.wav
//   VOXT_BENCH_OUT            output directory
//   VOXT_BENCH_RUN            run label, e.g. A0-r1
//   VOXT_BENCH_IDS            optional file of clip ids to run, one per line
//   VOXT_BENCH_REPO           default mlx-community/Qwen3-ASR-0.6B-4bit

import Darwin
import XCTest
@testable import Voxt

@MainActor
final class OmniPhase1BenchmarkTests: XCTestCase {
    private struct Clip: Decodable {
        let id: String
        let stratum: String
        let lang: String
        let duration: Double
    }

    private struct Settings {
        let manifest: URL
        let clips: URL
        let output: URL
        let run: String
        let repo: String
        let ids: Set<String>?
    }

    private func settings() throws -> Settings {
        try ModelTestGate.requireEnabled("Omni phase 1 benchmark")
        let environment = ProcessInfo.processInfo.environment
        func value(_ key: String) -> String? {
            let trimmed = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return trimmed.isEmpty ? nil : trimmed
        }
        guard let manifest = value("VOXT_BENCH_MANIFEST"),
              let clips = value("VOXT_BENCH_CLIPS"),
              let output = value("VOXT_BENCH_OUT"),
              let run = value("VOXT_BENCH_RUN") else {
            throw XCTSkip("Set VOXT_BENCH_MANIFEST, VOXT_BENCH_CLIPS, VOXT_BENCH_OUT and VOXT_BENCH_RUN.")
        }
        var ids: Set<String>?
        if let idsPath = value("VOXT_BENCH_IDS") {
            ids = Set(
                try String(contentsOfFile: idsPath, encoding: .utf8)
                    .split(whereSeparator: \.isNewline)
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
            )
        }
        return Settings(
            manifest: URL(fileURLWithPath: manifest),
            clips: URL(fileURLWithPath: clips, isDirectory: true),
            output: URL(fileURLWithPath: output, isDirectory: true),
            run: run,
            repo: value("VOXT_BENCH_REPO") ?? "mlx-community/Qwen3-ASR-0.6B-4bit",
            ids: ids
        )
    }

    private func loadClips(_ settings: Settings) throws -> [Clip] {
        let decoder = JSONDecoder()
        return try String(contentsOf: settings.manifest, encoding: .utf8)
            .split(whereSeparator: \.isNewline)
            .map { try decoder.decode(Clip.self, from: Data($0.utf8)) }
            .filter { settings.ids?.contains($0.id) ?? true }
    }

    private func makeManager(_ settings: Settings) async throws -> MLXModelManager {
        ModelTestGate.configureStorageRoot(for: self)
        let manager = MLXModelManager(modelRepo: settings.repo)
        _ = try await manager.refreshInstallation(repo: settings.repo)
        guard manager.isModelDownloaded(repo: settings.repo) else {
            throw XCTSkip("Benchmark model is not installed: \(settings.repo)")
        }
        return manager
    }

    // MARK: - Final (file) benchmark

    /// Cold load, then every clip through the Final path in manifest order, then
    /// idle unload and a reload. Model state is the app's: one manager, one
    /// transcriber, default preferences.
    func testFinalBenchmark() async throws {
        let settings = try settings()
        let clips = try loadClips(settings)
        let writer = try BenchmarkWriter(settings.output.appendingPathComponent("\(settings.run)-final.jsonl"))
        let sampler = ProcessTreeFootprintSampler()
        sampler.start()
        defer { sampler.stop() }

        let manager = try await makeManager(settings)
        let transcriber = MLXTranscriber(modelManager: manager)
        try await Task.sleep(for: .seconds(2))
        let baseline = sampler.snapshot()
        writer.write(["event": "baseline", "run": settings.run, "footprint_bytes": baseline.current, "processes": baseline.processes])

        sampler.resetPeak()
        var startedAt = ContinuousClock.now
        manager.beginActiveUse()
        _ = try await manager.loadModel()
        manager.endActiveUse()
        let coldLoad = startedAt.duration(to: .now)
        let afterLoad = sampler.snapshot()
        writer.write([
            "event": "cold_load", "run": settings.run, "ms": coldLoad.milliseconds,
            "peak_footprint_bytes": afterLoad.peak, "footprint_bytes": afterLoad.current,
            "processes": afterLoad.processes,
        ])

        sampler.resetPeak()
        for (index, clip) in clips.enumerated() {
            let loaded = try DebugAudioClipIO.loadMonoSamples(
                from: settings.clips.appendingPathComponent("\(clip.id).wav")
            )
            sampler.resetWindowPeak()
            startedAt = ContinuousClock.now
            var text = ""
            var failure: String?
            do {
                text = try await transcriber.transcribeBufferedResult(
                    samples: loaded.samples,
                    sampleRate: loaded.sampleRate
                )?.text ?? ""
            } catch {
                failure = String(describing: type(of: error))
            }
            let elapsed = startedAt.duration(to: .now)
            let window = sampler.snapshot()
            var record: [String: Any] = [
                "event": "final", "run": settings.run, "index": index, "id": clip.id,
                "stratum": clip.stratum, "lang": clip.lang, "audio_seconds": clip.duration,
                "ms": elapsed.milliseconds, "text": text,
                "window_peak_footprint_bytes": window.windowPeak,
            ]
            if let failure { record["failure"] = failure }
            writer.write(record)
        }
        let afterFinals = sampler.snapshot()
        writer.write([
            "event": "finals_done", "run": settings.run, "peak_footprint_bytes": afterFinals.peak,
            "footprint_bytes": afterFinals.current, "processes": afterFinals.processes,
        ])

        try await Task.sleep(for: .seconds(3))
        let loadedIdle = sampler.snapshot()
        startedAt = ContinuousClock.now
        manager.releaseLoadedModelIfIdle(reason: "benchmark")
        try await waitUntil(timeoutSeconds: 30) { !manager.hasLoadedModel }
        try await Task.sleep(for: .seconds(3))
        let afterUnload = sampler.snapshot()
        writer.write([
            "event": "unload", "run": settings.run,
            "loaded_idle_footprint_bytes": loadedIdle.current, "loaded_idle_processes": loadedIdle.processes,
            "after_unload_footprint_bytes": afterUnload.current, "after_unload_processes": afterUnload.processes,
        ])

        sampler.resetPeak()
        startedAt = ContinuousClock.now
        manager.beginActiveUse()
        _ = try await manager.loadModel()
        manager.endActiveUse()
        let reload = startedAt.duration(to: .now)
        writer.write(["event": "reload", "run": settings.run, "ms": reload.milliseconds,
                      "peak_footprint_bytes": sampler.snapshot().peak])

        // Not the transcriber's shutdown: it touches AVAudioEngine.inputNode, and
        // these runs never open an input device.
        await manager.shutdownForApplicationTermination()
        try await Task.sleep(for: .seconds(3))
        let afterShutdown = sampler.snapshot()
        writer.write(["event": "shutdown", "run": settings.run,
                      "footprint_bytes": afterShutdown.current, "processes": afterShutdown.processes])
    }

    // MARK: - Live session benchmark

    /// Dictation sessions without a microphone: the same native Qwen live session
    /// the transcriber installs (built by makeMeetingNativeStreamingConfiguration,
    /// whose Qwen config is identical to dictation's) is fed the clip in real time
    /// at the transcriber's 100 ms feed cadence. Stop then does what
    /// runFinalizationPipeline does in native live mode: cancel the live session
    /// and run the Final pass over the whole recording.
    func testSessionBenchmark() async throws {
        let settings = try settings()
        let clips = try loadClips(settings)
        let writer = try BenchmarkWriter(settings.output.appendingPathComponent("\(settings.run)-session.jsonl"))
        let sampler = ProcessTreeFootprintSampler()
        sampler.start()
        defer { sampler.stop() }

        let manager = try await makeManager(settings)
        let transcriber = MLXTranscriber(modelManager: manager)
        manager.beginActiveUse()
        _ = try await manager.loadModel()
        manager.endActiveUse()
        sampler.resetPeak()

        for (index, clip) in clips.enumerated() {
            let loaded = try DebugAudioClipIO.loadMonoSamples(
                from: settings.clips.appendingPathComponent("\(clip.id).wav")
            )
            XCTAssertEqual(loaded.sampleRate, 16000)
            let samples = loaded.samples
            let timeline = SessionTimeline()
            sampler.resetWindowPeak()
            manager.beginActiveUse()

            timeline.request = .now
            let session: any MLXNativeStreamingSession
            do {
                session = try await transcriber.makeMeetingNativeStreamingConfiguration().session
            } catch {
                manager.endActiveUse()
                writer.write(["event": "session", "run": settings.run, "index": index, "id": clip.id,
                              "stratum": clip.stratum, "failure": "live-setup: \(type(of: error))"])
                continue
            }
            timeline.captureReady = .now
            let consumer = Task { @MainActor in
                for await event in session.events {
                    switch event {
                    case .displayUpdate(let confirmedText, let provisionalText):
                        let visible = (confirmedText + provisionalText)
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !visible.isEmpty else { continue }
                        timeline.previewUpdates += 1
                        if timeline.firstPreview == nil { timeline.firstPreview = .now }
                        if visible != timeline.previews.last?.text {
                            timeline.previews.append((.now, visible))
                        }
                    case .failed:
                        timeline.liveFailed = true
                    default:
                        continue
                    }
                }
            }

            let playbackStart = ContinuousClock.now
            timeline.playbackStart = playbackStart
            var fed = 0
            while fed < samples.count {
                try await Task.sleep(for: .milliseconds(100))
                let elapsed = playbackStart.duration(to: .now)
                let due = min(samples.count, Int(Double(elapsed.milliseconds) * 16))
                if due > fed {
                    session.feedAudio(samples: Array(samples[fed..<due]))
                    fed = due
                }
            }
            timeline.playbackEnd = .now
            try await Task.sleep(for: .milliseconds(300))

            timeline.stop = .now
            session.cancel()
            var text = ""
            var failure: String?
            do {
                text = try await transcriber.transcribeBufferedResult(samples: samples, sampleRate: 16000)?.text ?? ""
            } catch {
                failure = "final: \(type(of: error))"
            }
            timeline.final = .now
            manager.endActiveUse()
            consumer.cancel()

            let window = sampler.snapshot()
            var record: [String: Any] = [
                "event": "session", "run": settings.run, "index": index, "id": clip.id,
                "stratum": clip.stratum, "lang": clip.lang, "audio_seconds": clip.duration,
                "live_setup_ms": timeline.ms(timeline.request, timeline.captureReady) as Any,
                "first_preview_after_audio_ms": timeline.ms(timeline.playbackStart, timeline.firstPreview) as Any,
                "stop_to_final_ms": timeline.ms(timeline.stop, timeline.final) as Any,
                "feed_ms": timeline.ms(timeline.playbackStart, timeline.playbackEnd) as Any,
                "preview_updates": timeline.previewUpdates,
                "previews": timeline.previews.map {
                    ["after_audio_ms": timeline.ms(timeline.playbackStart, $0.arrival) as Any, "text": $0.text]
                },
                "live_failed": timeline.liveFailed,
                "window_peak_footprint_bytes": window.windowPeak,
                "text": text,
            ]
            if let failure { record["failure"] = failure }
            writer.write(record)
            try await Task.sleep(for: .milliseconds(1500))
        }
        let done = sampler.snapshot()
        writer.write(["event": "sessions_done", "run": settings.run, "peak_footprint_bytes": done.peak,
                      "footprint_bytes": done.current, "processes": done.processes])
        // Not the transcriber's shutdown: it touches AVAudioEngine.inputNode, and
        // these runs never open an input device.
        await manager.shutdownForApplicationTermination()
    }

    // MARK: - Meeting live preview with background chunks

    /// A meeting's two consumers of one model: the live session (its own
    /// transcriber, as MeetingMLXNativeLiveSession builds it) fed in real time,
    /// while a second transcriber on the same manager runs the strict chunk
    /// Final for each completed 15 s window, as MeetingSegmentTranscribing does.
    /// Records every live update time and each chunk's latency.
    func testMeetingConcurrencyBenchmark() async throws {
        let settings = try settings()
        let clips = try loadClips(settings)
        let writer = try BenchmarkWriter(settings.output.appendingPathComponent("\(settings.run)-meeting.jsonl"))
        let manager = try await makeManager(settings)
        let liveTranscriber = MLXTranscriber(modelManager: manager, transcriptionPurpose: .meeting)
        let chunkTranscriber = MLXTranscriber(modelManager: manager, transcriptionPurpose: .meeting)
        manager.beginActiveUse()
        _ = try await manager.loadModel()
        let chunkSamples = 15 * 16000

        for (index, clip) in clips.enumerated() {
            let samples = try DebugAudioClipIO.loadMonoSamples(
                from: settings.clips.appendingPathComponent("\(clip.id).wav")
            ).samples
            let timeline = SessionTimeline()
            var updateOffsetsMs: [Int] = []
            let session = try await liveTranscriber.makeMeetingNativeStreamingConfiguration().session
            let start = ContinuousClock.now
            let consumer = Task { @MainActor in
                for await event in session.events {
                    if case .displayUpdate(let confirmedText, let provisionalText) = event,
                       !(confirmedText + provisionalText).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        updateOffsetsMs.append(start.duration(to: .now).milliseconds)
                    } else if case .failed = event {
                        timeline.liveFailed = true
                    }
                }
            }
            var chunkTasks: [Task<[String: Any], Never>] = []
            var fed = 0
            var nextChunkEnd = chunkSamples
            while fed < samples.count {
                try await Task.sleep(for: .milliseconds(100))
                let due = min(samples.count, start.duration(to: .now).milliseconds * 16)
                if due > fed {
                    session.feedAudio(samples: Array(samples[fed..<due]))
                    fed = due
                }
                while fed >= nextChunkEnd {
                    let chunk = Array(samples[(nextChunkEnd - chunkSamples)..<nextChunkEnd])
                    let launchedAtMs = start.duration(to: .now).milliseconds
                    chunkTasks.append(Task { @MainActor in
                        let chunkStart = ContinuousClock.now
                        let text = (try? await chunkTranscriber.transcribeBufferedResult(
                            samples: chunk, sampleRate: 16000
                        ))?.text
                        return ["launched_ms": launchedAtMs, "ms": chunkStart.duration(to: .now).milliseconds,
                                "failed": text == nil]
                    })
                    nextChunkEnd += chunkSamples
                }
            }
            var chunks: [[String: Any]] = []
            for task in chunkTasks { chunks.append(await task.value) }
            session.cancel()
            consumer.cancel()
            writer.write([
                "event": "meeting", "run": settings.run, "index": index, "id": clip.id,
                "audio_seconds": clip.duration, "update_offsets_ms": updateOffsetsMs,
                "chunks": chunks, "live_failed": timeline.liveFailed,
            ])
            try await Task.sleep(for: .milliseconds(1500))
        }
        manager.endActiveUse()
        await manager.shutdownForApplicationTermination()
    }

    private func waitUntil(timeoutSeconds: Double, _ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(timeoutSeconds)
        while !condition() {
            guard ContinuousClock.now < deadline else {
                XCTFail("Timed out after \(timeoutSeconds)s")
                return
            }
            try await Task.sleep(for: .milliseconds(50))
        }
    }
}

@MainActor
private final class SessionTimeline {
    var request: ContinuousClock.Instant?
    var captureReady: ContinuousClock.Instant?
    var playbackStart: ContinuousClock.Instant?
    var playbackEnd: ContinuousClock.Instant?
    var firstPreview: ContinuousClock.Instant?
    var stop: ContinuousClock.Instant?
    var final: ContinuousClock.Instant?
    var previewUpdates = 0
    /// Distinct visible texts with their arrival, to tell a real first preview from one decoded out of silence.
    var previews: [(arrival: ContinuousClock.Instant, text: String)] = []
    var liveFailed = false

    func ms(_ from: ContinuousClock.Instant?, _ to: ContinuousClock.Instant?) -> Int? {
        guard let from, let to else { return nil }
        return from.duration(to: to).milliseconds
    }
}

private extension Duration {
    var milliseconds: Int {
        let parts = components
        return Int(parts.seconds * 1000 + parts.attoseconds / 1_000_000_000_000_000)
    }
}

final class BenchmarkWriter {
    private let handle: FileHandle

    init(_ url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        handle.seekToEndOfFile()
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

/// Physical footprint of this process plus every descendant (the Omni server
/// process), sampled every 100 ms.
final class ProcessTreeFootprintSampler: @unchecked Sendable {
    struct Snapshot {
        let current: UInt64
        let peak: UInt64
        let windowPeak: UInt64
        let processes: Int
    }

    private let lock = NSLock()
    private var current: UInt64 = 0
    private var peak: UInt64 = 0
    private var windowPeak: UInt64 = 0
    private var processes = 0
    private var running = false

    func start() {
        lock.lock()
        running = true
        lock.unlock()
        sample()
        let thread = Thread { [weak self] in
            while let self, self.isRunning {
                self.sample()
                Thread.sleep(forTimeInterval: 0.1)
            }
        }
        thread.qualityOfService = .utility
        thread.start()
    }

    func stop() {
        lock.lock()
        running = false
        lock.unlock()
    }

    private var isRunning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return running
    }

    func resetPeak() {
        sample()
        lock.lock()
        peak = current
        windowPeak = current
        lock.unlock()
    }

    func resetWindowPeak() {
        sample()
        lock.lock()
        windowPeak = current
        lock.unlock()
    }

    func snapshot() -> Snapshot {
        sample()
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(current: current, peak: peak, windowPeak: windowPeak, processes: processes)
    }

    private func sample() {
        let pids = Self.descendants(of: getpid())
        let total = pids.reduce(UInt64(0)) { $0 + Self.footprint(of: $1) }
        lock.lock()
        current = total
        processes = pids.count
        peak = max(peak, total)
        windowPeak = max(windowPeak, total)
        lock.unlock()
    }

    private static func descendants(of root: pid_t) -> [pid_t] {
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_ALL, 0]
        var size = 0
        guard sysctl(&mib, 4, nil, &size, nil, 0) == 0 else { return [root] }
        let stride = MemoryLayout<kinfo_proc>.stride
        var procs = [kinfo_proc](repeating: kinfo_proc(), count: size / stride + 32)
        size = procs.count * stride
        guard sysctl(&mib, 4, &procs, &size, nil, 0) == 0 else { return [root] }
        var children: [pid_t: [pid_t]] = [:]
        for proc in procs.prefix(size / stride) {
            children[proc.kp_eproc.e_ppid, default: []].append(proc.kp_proc.p_pid)
        }
        var result: [pid_t] = []
        var queue = [root]
        while let pid = queue.popLast() {
            result.append(pid)
            queue += children[pid] ?? []
        }
        return result
    }

    private static func footprint(of pid: pid_t) -> UInt64 {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
            }
        }
        return status == 0 ? info.ri_phys_footprint : 0
    }
}
