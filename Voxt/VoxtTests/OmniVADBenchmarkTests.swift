// OmniVADBenchmarkTests.swift
// Measures Voxt's Silero VAD detectors for the original-against-native
// comparison, as OmniPhase1BenchmarkTests does for Qwen3-ASR.
//
// Opt-in: runs only with VOXT_RUN_MODEL_TESTS=1 and VOXT_BENCH_OUT set. The
// backend is the process's: MLXAudioVAD, or the native runtime with
// VOXT_ASR_BACKEND=omni and VOXT_OMNI_RUNTIME, so both arms run identical code.
//
//   VOXT_MODEL_STORAGE_ROOT   model storage root holding mlx-audio/<repo>
//   VOXT_BENCH_MANIFEST       manifest.jsonl (id, stratum, lang, duration)
//   VOXT_BENCH_CLIPS          directory of <id>.wav (16 kHz mono)
//   VOXT_BENCH_OUT            output directory
//   VOXT_BENCH_RUN            run label, e.g. A0-r1

import Foundation
import XCTest
@testable import Voxt

@MainActor
final class OmniVADBenchmarkTests: XCTestCase {
    private struct Settings {
        let clips: [(id: String, url: URL)]
        let writer: BenchmarkWriter
        let run: String
    }

    private func settings(_ mode: String) throws -> Settings {
        try ModelTestGate.requireEnabled("Silero VAD benchmark")
        let environment = ProcessInfo.processInfo.environment
        guard let manifest = environment["VOXT_BENCH_MANIFEST"], let clips = environment["VOXT_BENCH_CLIPS"],
              let output = environment["VOXT_BENCH_OUT"], let run = environment["VOXT_BENCH_RUN"] else {
            throw XCTSkip("Set VOXT_BENCH_MANIFEST, VOXT_BENCH_CLIPS, VOXT_BENCH_OUT and VOXT_BENCH_RUN.")
        }
        ModelTestGate.configureStorageRoot(for: self)
        let ids = try String(contentsOfFile: manifest, encoding: .utf8)
            .split(whereSeparator: \.isNewline)
            .compactMap { line -> String? in
                (try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])?["id"] as? String
            }
        return Settings(
            clips: ids.map { ($0, URL(fileURLWithPath: clips).appendingPathComponent("\($0).wav")) },
            writer: try BenchmarkWriter(URL(fileURLWithPath: output).appendingPathComponent("\(run)-vad-\(mode).jsonl")),
            run: run
        )
    }

    private var backend: String { OmniSileroVADRuntime.isEnabled ? "native" : "swift" }

    /// The streaming detector as dictation and meetings run it: one call per
    /// 512 samples, one stream per clip.
    func testStreamingVADBenchmark() async throws {
        let settings = try settings("stream")
        let sampler = ProcessTreeFootprintSampler()
        sampler.start()
        defer { sampler.stop() }
        let baseline = sampler.snapshot()
        settings.writer.write(["event": "baseline", "run": settings.run, "backend": backend,
                               "footprint_bytes": baseline.current, "processes": baseline.processes])

        let detector = ASRSileroStreamingVoiceActivityDetector()
        var startedAt = ContinuousClock.now
        _ = try await detector.probability(samples: [Float](repeating: 0, count: 512), sampleRate: 16_000, streamID: "cold")
        settings.writer.write(["event": "cold_load", "run": settings.run, "ms": startedAt.duration(to: .now).msDouble,
                               "peak_footprint_bytes": sampler.snapshot().peak])

        sampler.resetPeak()
        for clip in settings.clips {
            let samples = try DebugAudioClipIO.loadMonoSamples(from: clip.url).samples
            var calls: [Double] = []
            let clipStart = ContinuousClock.now
            var offset = 0
            while offset + 512 <= samples.count {
                startedAt = ContinuousClock.now
                _ = try await detector.probability(
                    samples: Array(samples[offset ..< offset + 512]), sampleRate: 16_000, streamID: clip.id
                )
                calls.append(startedAt.duration(to: .now).msDouble)
                offset += 512
            }
            settings.writer.write(["event": "clip", "run": settings.run, "id": clip.id,
                                   "audio_seconds": Double(samples.count) / 16_000,
                                   "ms": clipStart.duration(to: .now).msDouble, "call_ms": calls])
            // Note (Jiaxin Deng): one clip is one session; Voxt resets the detector between sessions.
            await detector.reset()
        }
        let done = sampler.snapshot()
        settings.writer.write(["event": "clips_done", "run": settings.run, "peak_footprint_bytes": done.peak,
                               "processes": done.processes])
        try await recordUnload(settings, sampler: sampler) { await detector.unload() }
    }

    /// The offline detector as meeting audio and imported files use it: speech
    /// ranges of a whole clip with the stored sensitivity profile.
    func testOfflineVADBenchmark() async throws {
        let settings = try settings("offline")
        let sampler = ProcessTreeFootprintSampler()
        sampler.start()
        defer { sampler.stop() }
        let baseline = sampler.snapshot()
        settings.writer.write(["event": "baseline", "run": settings.run, "backend": backend,
                               "footprint_bytes": baseline.current, "processes": baseline.processes])

        let detector = ASRSileroOfflineVoiceActivityDetector()
        let startedAt = ContinuousClock.now
        _ = try await detector.speechRanges(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000)
        settings.writer.write(["event": "cold_load", "run": settings.run, "ms": startedAt.duration(to: .now).msDouble,
                               "peak_footprint_bytes": sampler.snapshot().peak])

        sampler.resetPeak()
        for clip in settings.clips {
            let samples = try DebugAudioClipIO.loadMonoSamples(from: clip.url).samples
            let clipStart = ContinuousClock.now
            let ranges = try await detector.speechRanges(samples: samples, sampleRate: 16_000)
            settings.writer.write(["event": "clip", "run": settings.run, "id": clip.id,
                                   "audio_seconds": Double(samples.count) / 16_000,
                                   "ms": clipStart.duration(to: .now).msDouble, "ranges": ranges.count])
        }
        let done = sampler.snapshot()
        settings.writer.write(["event": "clips_done", "run": settings.run, "peak_footprint_bytes": done.peak,
                               "processes": done.processes])
        try await recordUnload(settings, sampler: sampler) { await detector.unload() }
    }

    private func recordUnload(
        _ settings: Settings,
        sampler: ProcessTreeFootprintSampler,
        unload: () async -> Void
    ) async throws {
        try await Task.sleep(for: .seconds(3))
        let loadedIdle = sampler.snapshot()
        await unload()
        try await Task.sleep(for: .seconds(3))
        let afterUnload = sampler.snapshot()
        settings.writer.write([
            "event": "unload", "run": settings.run,
            "loaded_idle_footprint_bytes": loadedIdle.current, "loaded_idle_processes": loadedIdle.processes,
            "after_unload_footprint_bytes": afterUnload.current, "after_unload_processes": afterUnload.processes,
        ])
    }
}

extension Duration {
    var msDouble: Double {
        let parts = components
        return Double(parts.seconds) * 1000 + Double(parts.attoseconds) / 1e15
    }
}
