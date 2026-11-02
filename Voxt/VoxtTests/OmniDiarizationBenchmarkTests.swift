// OmniDiarizationBenchmarkTests.swift
// Measures Voxt's Sortformer meeting speaker analysis; VOXT_ASR_BACKEND=omni with
// VOXT_OMNI_RUNTIME selects the native arm. Opt-in: VOXT_RUN_MODEL_TESTS=1 with
// VOXT_MODEL_STORAGE_ROOT and VOXT_BENCH_CLIPS, _IDS, _OUT and _RUN.

import Foundation
import XCTest
@testable import Voxt

@MainActor
final class OmniDiarizationBenchmarkTests: XCTestCase {
    // Note (Jiaxin Deng): Voxt's meeting speaker analysis feed (4.96 s).
    private let samplesPerFeed = 79_360

    func testDiarizationBenchmark() async throws {
        try ModelTestGate.requireEnabled("Sortformer benchmark")
        let environment = ProcessInfo.processInfo.environment
        guard let clipsPath = environment["VOXT_BENCH_CLIPS"], let idsPath = environment["VOXT_BENCH_IDS"],
              let output = environment["VOXT_BENCH_OUT"], let run = environment["VOXT_BENCH_RUN"] else {
            throw XCTSkip("Set VOXT_BENCH_CLIPS, VOXT_BENCH_IDS, VOXT_BENCH_OUT and VOXT_BENCH_RUN.")
        }
        ModelTestGate.configureStorageRoot(for: self)
        let ids = try String(contentsOfFile: idsPath, encoding: .utf8)
            .split(whereSeparator: \.isNewline).map(String.init).filter { !$0.isEmpty }
        let assets = try ids.map { id -> (id: String, asset: MeetingAudioAsset) in
            let samples = try DebugAudioClipIO.loadMonoSamples(
                from: URL(fileURLWithPath: clipsPath).appendingPathComponent("\(id).wav")
            ).samples
            return (id, MeetingAudioAsset(source: .systemAudio, samples: samples, sampleRate: 16_000, sessionStartOffset: 0))
        }
        let writer = try BenchmarkWriter(URL(fileURLWithPath: output).appendingPathComponent("\(run)-diarization.jsonl"))
        let sampler = ProcessTreeFootprintSampler()
        sampler.start()
        defer { sampler.stop() }
        let baseline = sampler.snapshot()
        writer.write(["event": "baseline", "run": run, "backend": OmniSortformerRuntime.isEnabled ? "native" : "swift",
                      "footprint_bytes": baseline.current, "processes": baseline.processes])

        sampler.resetPeak()
        for (id, asset) in assets {
            let engine = SortformerMeetingSpeakerDiarizationEngine()
            let descriptor = MeetingAudioAssetDescriptor(source: .systemAudio, sampleRate: 16_000,
                                                         startSample: 0, sampleCount: asset.samples.count)
            let startedAt = ContinuousClock.now
            _ = try await engine.diarizeFile(descriptors: [descriptor], loadAsset: { _ in asset },
                                             options: .init(), progress: nil)
            writer.write(["event": "file", "run": run, "id": id, "audio_seconds": Double(asset.samples.count) / 16_000,
                          "feeds": feeds(asset), "ms": startedAt.duration(to: .now).msDouble])
        }
        let afterFiles = sampler.snapshot()
        try await Task.sleep(for: .seconds(3))
        let released = sampler.snapshot()
        writer.write(["event": "files_done", "run": run, "peak_footprint_bytes": afterFiles.peak,
                      "after_release_footprint_bytes": released.current, "after_release_processes": released.processes])

        let live = SortformerMeetingSpeakerDiarizationEngine()
        let first = assets[0].asset
        let coldAsset = MeetingAudioAsset(source: .systemAudio, samples: Array(first.samples.prefix(samplesPerFeed)),
                                          sampleRate: 16_000, sessionStartOffset: 0)
        sampler.resetPeak()
        var startedAt = ContinuousClock.now
        _ = try await live.diarize(asset: coldAsset, options: .init())
        writer.write(["event": "cold_load", "run": run, "ms": startedAt.duration(to: .now).msDouble,
                      "peak_footprint_bytes": sampler.snapshot().peak])
        sampler.resetPeak()
        for (id, asset) in assets {
            startedAt = ContinuousClock.now
            _ = try await live.diarize(asset: asset, options: .init())
            writer.write(["event": "live", "run": run, "id": id, "audio_seconds": Double(asset.samples.count) / 16_000,
                          "feeds": feeds(asset), "ms": startedAt.duration(to: .now).msDouble])
        }
        let afterLive = sampler.snapshot()
        try await Task.sleep(for: .seconds(3))
        let loadedIdle = sampler.snapshot()
        writer.write(["event": "live_done", "run": run, "peak_footprint_bytes": afterLive.peak,
                      "loaded_idle_footprint_bytes": loadedIdle.current, "loaded_idle_processes": loadedIdle.processes])
    }

    private func feeds(_ asset: MeetingAudioAsset) -> Int {
        (asset.samples.count + samplesPerFeed - 1) / samplesPerFeed
    }
}
