// OmniSpeakerDiarizationIntegrationTests.swift
// Voxt's Sortformer speaker analysis on the native runtime against MLXAudioVAD.
// Opt-in: VOXT_RUN_MODEL_TESTS=1, VOXT_ASR_BACKEND=omni, VOXT_OMNI_RUNTIME, VOXT_MODEL_STORAGE_ROOT
// and VOXT_DIARIZATION_CLIP (16 kHz mono speech, a minute or more so the cache compresses).

import MLX
import MLXAudioVAD
import XCTest
@testable import Voxt

@MainActor
final class OmniSpeakerDiarizationIntegrationTests: XCTestCase {
    // Note (Jiaxin Deng): the runtime golden check's tolerances; MLX 0.31 and 0.32 kernels
    // differ in the last bits, and the speaker cache carries that through a clip.
    private let maxProbabilityDifference: Float = 0.35
    private let maxFlippedDecisions = 0.01
    private let maxSpeechMismatch = 0.03

    private struct Reference {
        let policy: MeetingSpeakerFeedPolicy
        var probabilities: [[Float]] = []
        var states: [(fifo: Int, cache: Int, frames: Int)] = []
        var turns: [(speaker: Int, start: Double, end: Double)] = []
    }

    private func clip() throws -> [Float] {
        try ModelTestGate.requireEnabled("Sortformer on the native runtime")
        guard OmniSortformerRuntime.isEnabled else {
            throw XCTSkip("Set VOXT_ASR_BACKEND=omni and VOXT_OMNI_RUNTIME.")
        }
        guard let path = ProcessInfo.processInfo.environment["VOXT_DIARIZATION_CLIP"], !path.isEmpty else {
            throw XCTSkip("Set VOXT_DIARIZATION_CLIP to a 16 kHz mono WAV.")
        }
        ModelTestGate.configureStorageRoot(for: self)
        let (samples, sampleRate) = try DebugAudioClipIO.loadMonoSamples(from: URL(fileURLWithPath: path))
        XCTAssertEqual(sampleRate, 16_000)
        return samples
    }

    private func reference(samples: [Float]) async throws -> Reference {
        let storedDirectory = await MeetingSortformerModelStorage.validatedModelDirectory()
        let directory = try XCTUnwrap(storedDirectory)
        let model = try SortformerModel.fromModelDirectory(directory)
        let config = model.config
        var reference = Reference(policy: try MeetingSpeakerFeedPolicy(
            sampleRate: config.processorConfig.samplingRate,
            hopLength: config.processorConfig.hopLength,
            subsamplingFactor: config.fcEncoderConfig.subsamplingFactor,
            chunkFrames: config.modulesConfig.chunkLen,
            cacheFrames: config.modulesConfig.spkcacheLen,
            updateFrames: config.modulesConfig.spkcacheUpdatePeriod,
            usesAOSC: config.modulesConfig.useAosc
        ))
        let policy = reference.policy
        var state = model.initStreamingState()
        var offset = 0
        while offset < samples.count {
            let end = min(offset + policy.samplesPerFeed, samples.count)
            var padded = Array(samples[offset ..< end])
            padded.append(contentsOf: repeatElement(0, count: max(0, policy.frameSamples - padded.count)))
            let inputFrames = state.framesProcessed
            let (output, next) = try await model.feed(
                chunk: MLXArray(padded), state: state, sampleRate: 16_000,
                threshold: 0.5, minDuration: 0, mergeGap: 0.18,
                spkcacheMax: policy.cacheMaximumFrames, fifoMax: MeetingSpeakerFeedPolicy.fifoMaximumFrames
            )
            state = next
            let probabilities = try XCTUnwrap(output.speakerProbs).asType(.float32)
            let speakers = probabilities.dim(1)
            let flat = probabilities.asArray(Float.self)
            reference.probabilities += stride(from: 0, to: flat.count, by: speakers).map {
                Array(flat[$0 ..< $0 + speakers])
            }
            reference.states.append((state.fifoLen, state.spkcacheLen, state.framesProcessed))
            for segment in output.segments {
                if let range = policy.mappedRange(
                    start: Double(segment.start), end: Double(segment.end),
                    stateFrames: inputFrames, audioOffset: Double(offset) / 16_000, sampleCount: end - offset
                ) {
                    reference.turns.append((segment.speaker, range.lowerBound, range.upperBound))
                }
            }
            offset = end
        }
        return reference
    }

    private func speechMismatch(
        _ expected: [(speaker: Int, start: Double, end: Double)],
        _ actual: [(speaker: Int, start: Double, end: Double)]
    ) -> Double {
        func cells(_ turns: [(speaker: Int, start: Double, end: Double)]) -> Set<Int> {
            var covered = Set<Int>()
            for turn in turns {
                let first = Int((turn.start * 100).rounded())
                let last = Int((turn.end * 100).rounded())
                if last > first {
                    covered.formUnion((first ..< last).map { turn.speaker * 100_000_000 + $0 })
                }
            }
            return covered
        }
        let want = cells(expected)
        let have = cells(actual)
        let union = want.union(have).count
        return union == 0 ? 0 : Double(want.symmetricDifference(have).count) / Double(union)
    }

    /// A server stream fed Voxt's feeds reports MLXAudioVAD's probabilities
    /// and state, feed by feed.
    func testStreamMatchesMLXAudioVAD() async throws {
        let samples = try clip()
        let reference = try await reference(samples: samples)
        let policy = reference.policy

        let storedDirectory = await MeetingSortformerModelStorage.validatedModelDirectory()
        let directory = try XCTUnwrap(storedDirectory)
        let endpoint = try await OmniSortformerRuntime.shared.acquire(modelDirectory: directory)
        let stream = OmniDiarizationStream(endpoint: endpoint)
        var probabilities: [[Float]] = []
        var offset = 0
        var feedIndex = 0
        while offset < samples.count {
            let end = min(offset + policy.samplesPerFeed, samples.count)
            var padded = Array(samples[offset ..< end])
            padded.append(contentsOf: repeatElement(0, count: max(0, policy.frameSamples - padded.count)))
            let feed = try await stream.feed(samples16k: padded)
            probabilities += feed.probabilities
            let state = reference.states[feedIndex]
            XCTAssertEqual(feed.fifoLength, state.fifo)
            XCTAssertEqual(feed.spkcacheLength, state.cache)
            XCTAssertEqual(feed.framesProcessed, state.frames)
            offset = end
            feedIndex += 1
        }
        await stream.close()
        await OmniSortformerRuntime.shared.release()

        XCTAssertEqual(probabilities.count, reference.probabilities.count)
        var flipped = 0
        var decisions = 0
        for (row, expected) in zip(probabilities, reference.probabilities) {
            for (value, original) in zip(row, expected) {
                XCTAssertEqual(value, original, accuracy: maxProbabilityDifference)
                flipped += (value > 0.5) != (original > 0.5) ? 1 : 0
                decisions += 1
            }
        }
        XCTAssertGreaterThan(decisions, 0)
        XCTAssertLessThanOrEqual(Double(flipped), maxFlippedDecisions * Double(decisions))
    }

    /// A server that dies costs the live engine one failed session; the next
    /// session starts a new server.
    func testLiveEngineRecoversAfterTheServerDies() async throws {
        let samples = Array(try clip().prefix(16_000 * 10))
        let engine = SortformerMeetingSpeakerDiarizationEngine()
        let asset = MeetingAudioAsset(source: .systemAudio, samples: samples, sampleRate: 16_000, sessionStartOffset: 0)
        let descriptors = [MeetingAudioAssetDescriptor(source: .systemAudio, sampleRate: 16_000,
                                                       startSample: 0, sampleCount: samples.count)]
        _ = try await engine.diarizeSession(descriptors: descriptors, loadAsset: { _ in asset }, continuousAudioURL: nil, options: .init(), progress: nil)

        let kill = Process()
        kill.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        kill.arguments = ["-KILL", "-f", "model-kind sortformer"]
        try kill.run()
        kill.waitUntilExit()
        XCTAssertEqual(kill.terminationStatus, 0)

        var recovered = false
        for _ in 0 ..< 5 {
            if (try? await engine.diarizeSession(descriptors: descriptors, loadAsset: { _ in asset }, continuousAudioURL: nil, options: .init(), progress: nil)) != nil {
                recovered = true
                break
            } else {
                try await Task.sleep(for: .milliseconds(100))
            }
        }
        XCTAssertTrue(recovered)
    }

    /// The meeting engine, routed to the runtime, finds the same speaker turns
    /// as the Swift engine's feeds.
    func testEngineTurnsMatchMLXAudioVAD() async throws {
        let samples = try clip()
        let reference = try await reference(samples: samples)

        let engine = SortformerMeetingSpeakerDiarizationEngine()
        let asset = MeetingAudioAsset(source: .systemAudio, samples: samples, sampleRate: 16_000, sessionStartOffset: 0)
        let turns = try await engine.diarizeFile(
            descriptors: [MeetingAudioAssetDescriptor(source: .systemAudio, sampleRate: 16_000,
                                                      startSample: 0, sampleCount: samples.count)],
            loadAsset: { _ in asset }, options: .init(), progress: nil
        )

        XCTAssertFalse(reference.turns.isEmpty)
        let actual = try turns.map { turn -> (speaker: Int, start: Double, end: Double) in
            let speaker = try XCTUnwrap(Int(turn.speakerID.replacingOccurrences(of: "sortformer-", with: "")))
            return (speaker, turn.startSeconds, turn.endSeconds)
        }
        XCTAssertLessThanOrEqual(speechMismatch(reference.turns, actual), maxSpeechMismatch)
    }
}
