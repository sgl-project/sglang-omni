// OmniVoiceActivityIntegrationTests.swift
// Voxt's Silero detectors on the native runtime against MLXAudioVAD in-process.
//
// Opt-in: VOXT_RUN_MODEL_TESTS=1, VOXT_ASR_BACKEND=omni with VOXT_OMNI_RUNTIME
// (the native qwen3_asr_server binary), VOXT_MODEL_STORAGE_ROOT holding
// mlx-audio/mlx-community_silero-vad-v6, and VOXT_VAD_CLIP (a 16 kHz mono WAV
// with speech).

import MLX
import MLXAudioVAD
import XCTest
@testable import Voxt

@MainActor
final class OmniVoiceActivityIntegrationTests: XCTestCase {
    /// MLX 0.31 (Swift) and 0.32 (runtime) kernels differ in the last bits.
    private let tolerance: Float = 0.01

    private func clip() throws -> [Float] {
        try ModelTestGate.requireEnabled("Silero VAD on the native runtime")
        let environment = ProcessInfo.processInfo.environment
        guard OmniSileroVADRuntime.isEnabled else {
            throw XCTSkip("Set VOXT_ASR_BACKEND=omni and VOXT_OMNI_RUNTIME.")
        }
        guard let path = environment["VOXT_VAD_CLIP"], !path.isEmpty else {
            throw XCTSkip("Set VOXT_VAD_CLIP to a 16 kHz mono WAV.")
        }
        ModelTestGate.configureStorageRoot(for: self)
        let (samples, sampleRate) = try DebugAudioClipIO.loadMonoSamples(from: URL(fileURLWithPath: path))
        XCTAssertEqual(sampleRate, 16_000)
        return samples
    }

    private func swiftModel() async throws -> SileroVAD {
        let directory = try await SileroVADModelProvisioner.shared.ensureModelDirectory()
        return try SileroVADModelSupport.loadModel(from: directory)
    }

    /// The streaming detector, fed in uneven buffers, reports the same
    /// probabilities as MLXAudioVAD fed chunk by chunk.
    func testStreamingDetectorMatchesMLXAudioVAD() async throws {
        let samples = try clip()
        let model = try await swiftModel()
        var state: SileroVADStreamingState?
        var expected = [Float]()
        var offset = 0
        while offset + 512 <= samples.count {
            let (probability, next) = try model.feed(
                chunk: MLXArray(Array(samples[offset ..< offset + 512])), state: state, sampleRate: 16_000
            )
            eval(probability)
            expected.append(probability.item(Float.self))
            state = next
            offset += 512
        }

        let detector = ASRSileroStreamingVoiceActivityDetector()
        let sizes = [700, 4096, 300, 1600]
        var start = 0
        var index = 0
        var compared = 0
        while start < samples.count {
            let end = min(samples.count, start + sizes[index % sizes.count])
            let probability = try await detector.probability(
                samples: Array(samples[start ..< end]), sampleRate: 16_000, streamID: "integration"
            )
            let completed = end / 512
            if completed > start / 512 {
                let reference = expected[completed - 1]
                let probability = try XCTUnwrap(probability)
                XCTAssertEqual(probability, reference, accuracy: tolerance)
                XCTAssertEqual(probability >= 0.5, reference >= 0.5)
                compared += 1
            } else {
                XCTAssertNil(probability)
            }
            start = end
            index += 1
        }
        XCTAssertGreaterThan(compared, 10)
        await detector.unload()
    }

    /// A server that dies costs one failed call; the next call starts a new one.
    func testStreamingDetectorRecoversAfterTheServerDies() async throws {
        let samples = try clip()
        let detector = ASRSileroStreamingVoiceActivityDetector()
        let chunk = { (index: Int) in Array(samples[index * 512 ..< (index + 1) * 512]) }
        let first = try await detector.probability(samples: chunk(0), sampleRate: 16_000, streamID: "recovery")
        XCTAssertNotNil(first)

        let kill = Process()
        kill.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        kill.arguments = ["-KILL", "-f", "model-kind silero_vad"]
        try kill.run()
        kill.waitUntilExit()
        XCTAssertEqual(kill.terminationStatus, 0)

        var recovered = false
        for index in 1 ..< 6 {
            if (try? await detector.probability(samples: chunk(index), sampleRate: 16_000, streamID: "recovery")) != nil {
                recovered = true
                break
            } else {
                try await Task.sleep(for: .milliseconds(100))
            }
        }
        XCTAssertTrue(recovered)
        await detector.unload()
    }

    /// A reset() during a pending exchange is not a server failure: the call
    /// returns quietly and the next one reuses the same server.
    func testStreamingDetectorResetDuringAnExchangeDoesNotFail() async throws {
        let samples = try clip()
        let detector = ASRSileroStreamingVoiceActivityDetector()
        _ = try await detector.probability(samples: Array(samples.prefix(512)), sampleRate: 16_000, streamID: "reset")
        async let pending = detector.probability(samples: samples, sampleRate: 16_000, streamID: "reset")
        try await Task.sleep(for: .milliseconds(20))
        await detector.reset()
        _ = try await pending
        let next = try await detector.probability(samples: Array(samples.prefix(512)), sampleRate: 16_000, streamID: "reset")
        XCTAssertNotNil(next)
        await detector.unload()
    }

    /// The offline detector finds the same speech ranges as getSpeechTimestamps
    /// with the stored meeting profile, to within one chunk.
    func testOfflineDetectorMatchesMLXAudioVAD() async throws {
        let samples = try clip()
        let model = try await swiftModel()
        let profile = MeetingSileroVADSensitivity.stored().configuration()
        let expected = try model.getSpeechTimestamps(
            MLXArray(samples),
            sampleRate: 16_000,
            threshold: profile.onsetProbabilityThreshold,
            minSpeechDurationMs: Int((profile.minSpeechSeconds * 1_000).rounded(.up)),
            minSilenceDurationMs: Int((profile.minSilenceSeconds * 1_000).rounded(.up)),
            speechPadMs: Int((profile.speechPadSeconds * 1_000).rounded(.up))
        )

        let detector = ASRSileroOfflineVoiceActivityDetector()
        let ranges = try await detector.speechRanges(samples: samples, sampleRate: 16_000)
        await detector.unload()

        XCTAssertFalse(expected.isEmpty)
        XCTAssertEqual(ranges.count, expected.count)
        let chunkSeconds = 512.0 / 16_000
        for (range, timestamp) in zip(ranges, expected) {
            XCTAssertEqual(range.startSeconds, Double(timestamp.start) / 16_000, accuracy: chunkSeconds)
            XCTAssertEqual(range.endSeconds, Double(timestamp.end) / 16_000, accuracy: chunkSeconds)
        }
    }
}
