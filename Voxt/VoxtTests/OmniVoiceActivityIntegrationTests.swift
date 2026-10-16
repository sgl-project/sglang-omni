// OmniVoiceActivityIntegrationTests.swift
// Native HTTP fixture tests; numerical parity requires an independently supplied baseline.
// VOXT_RUN_MODEL_TESTS=1, VOXT_OMNI_RUNTIME, VOXT_MODEL_STORAGE_ROOT,
// VOXT_VAD_CLIP (16 kHz mono WAV), VOXT_VAD_REFERENCE_JSON (external MLXAudioVAD capture).

import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import Voxt

@MainActor
final class OmniVoiceActivityIntegrationTests: XCTestCase {
    // Note (assistant): retains the historical cross-kernel parity tolerance, not an inferred golden.
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

    private struct ExternalReference: Decodable {
        let implementation: String
        let revision: String
        let modelSHA256: String
        let samplesSHA256: String
        let sampleRate: Int
        let sampleCount: Int
        let probabilities: [Float]
        let threshold: Float
        let minSpeechDurationMs: Int
        let minSilenceDurationMs: Int
        let speechPadMs: Int
        let timestamps: [Timestamp]

        struct Timestamp: Decodable {
            let start: Int
            let end: Int
        }
    }

    private func externalReference(samples: [Float]) async throws -> ExternalReference {
        guard let path = ProcessInfo.processInfo.environment["VOXT_VAD_REFERENCE_JSON"], !path.isEmpty else {
            throw XCTSkip("Supply an independent MLXAudioVAD capture via VOXT_VAD_REFERENCE_JSON; Native output is not a reference.")
        }
        let reference = try JSONDecoder().decode(ExternalReference.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertEqual(reference.implementation, "MLXAudioVAD")
        XCTAssertFalse(reference.revision.isEmpty)
        XCTAssertEqual(reference.sampleRate, 16_000)
        XCTAssertEqual(reference.sampleCount, samples.count)
        XCTAssertEqual(reference.probabilities.count, samples.count / 512)
        let samplesDigest = SHA256.hash(data: OmniVoiceActivityStream.float32LittleEndian(samples))
        XCTAssertEqual(reference.samplesSHA256, samplesDigest.map { String(format: "%02x", $0) }.joined())
        let directory = try await SileroVADModelProvisioner.shared.ensureModelDirectory()
        let reader = try await SileroVADModelSupport.validatedModelDirectoryLease(from: directory)
        defer { withExtendedLifetime(reader) {} }
        let modelDigest = SHA256.hash(data: try Data(contentsOf: reader.directory.appendingPathComponent("model.safetensors")))
        XCTAssertEqual(reference.modelSHA256, modelDigest.map { String(format: "%02x", $0) }.joined())
        return reference
    }

    func testStreamingDetectorMatchesIndependentMLXAudioVADCapture() async throws {
        let samples = try clip()
        let reference = try await externalReference(samples: samples)
        let expected = reference.probabilities
        guard expected.count == samples.count / 512 else { return }

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
        guard samples.count >= 6 * 512 else { throw XCTSkip("Recovery fixture needs at least six frames.") }
        let detector = ASRSileroStreamingVoiceActivityDetector()
        let chunk = { (index: Int) in Array(samples[index * 512 ..< (index + 1) * 512]) }
        let first = try await detector.probability(samples: chunk(0), sampleRate: 16_000, streamID: "recovery")
        XCTAssertNotNil(first)

        let directory = try await SileroVADModelProvisioner.shared.ensureModelDirectory()
        let endpoint = try await OmniSileroVADRuntime.shared.acquire(modelDirectory: directory)
        await OmniSileroVADRuntime.shared.release()
        XCTAssertEqual(Darwin.kill(endpoint.serverProcessIdentifier, SIGKILL), 0)

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

    func testOfflineDetectorMatchesIndependentMLXAudioVADCapture() async throws {
        let samples = try clip()
        let reference = try await externalReference(samples: samples)
        let profile = MeetingSileroVADSensitivity.stored().configuration()
        XCTAssertEqual(reference.threshold, profile.onsetProbabilityThreshold)
        XCTAssertEqual(reference.minSpeechDurationMs, Int((profile.minSpeechSeconds * 1_000).rounded(.up)))
        XCTAssertEqual(reference.minSilenceDurationMs, Int((profile.minSilenceSeconds * 1_000).rounded(.up)))
        XCTAssertEqual(reference.speechPadMs, Int((profile.speechPadSeconds * 1_000).rounded(.up)))
        let expected = reference.timestamps

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
