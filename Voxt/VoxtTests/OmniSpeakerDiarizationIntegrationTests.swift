// Before: b02a3d49 generated its reference with MLXAudioVAD in this test process.
// After: an independently supplied library capture is required; Native never supplies its own oracle.
import CryptoKit
import Foundation
import XCTest
@testable import Voxt

@MainActor
final class OmniSpeakerDiarizationIntegrationTests: XCTestCase {
    // Note (Jiaxin Deng): preserved original tolerances for MLX 0.31/0.32 kernel differences.
    private let maxProbabilityDifference: Float = 0.35
    private let maxFlippedDecisions = 0.01
    private let maxSpeechMismatch = 0.03

    private struct Reference: Decodable {
        struct Feed: Decodable {
            let sampleCount: Int
            let probabilities: [[Float]]
            let fifoLength: Int
            let spkcacheLength: Int
            let framesProcessed: Int
        }
        struct Turn: Decodable {
            let speaker: Int
            let start: Double
            let end: Double
        }
        let producer: String
        let sourceRevision: String
        let captureCommand: String
        let configurationSHA256: String
        let weightsSHA256: String
        let pcmFloat32LESHA256: String
        let threshold: Float
        let minDuration: Float
        let mergeGap: Float
        let spkcacheMax: Int
        let fifoMax: Int
        let feeds: [Feed]
        let turns: [Turn]
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
        XCTAssertFalse(samples.isEmpty)
        return samples
    }

    private func reference(samples: [Float], model: MeetingSortformerModelLease) async throws -> Reference {
        guard let path = ProcessInfo.processInfo.environment["VOXT_SORTFORMER_REFERENCE"], !path.isEmpty else {
            throw XCTSkip("Supply VOXT_SORTFORMER_REFERENCE: an external MLXAudioVAD capture, not Native output.")
        }
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertEqual(reference.producer, "MLXAudioVAD")
        XCTAssertFalse(reference.sourceRevision.isEmpty)
        XCTAssertFalse(reference.captureCommand.isEmpty)
        let storedDirectory = await MeetingSortformerModelStorage.validatedModelDirectory()
        let directory = try XCTUnwrap(storedDirectory)
        for (name, expected) in [("config.json", reference.configurationSHA256), ("model.safetensors", reference.weightsSHA256)] {
            let bytes = try Data(contentsOf: directory.appendingPathComponent(name), options: .mappedIfSafe)
            XCTAssertEqual(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), expected)
        }
        let pcm = OmniVoiceActivityStream.float32LittleEndian(samples)
        XCTAssertEqual(SHA256.hash(data: pcm).map { String(format: "%02x", $0) }.joined(), reference.pcmFloat32LESHA256)
        let policy = try model.configuration.feedPolicy()
        XCTAssertEqual(reference.threshold, 0.5)
        XCTAssertEqual(reference.minDuration, 0)
        XCTAssertEqual(reference.mergeGap, 0.18)
        XCTAssertEqual(reference.spkcacheMax, policy.cacheMaximumFrames)
        XCTAssertEqual(reference.fifoMax, MeetingSpeakerFeedPolicy.fifoMaximumFrames)
        XCTAssertFalse(reference.feeds.isEmpty)
        return reference
    }

    func testStreamMatchesSuppliedLibraryReference() async throws {
        let samples = try clip()
        let model = try await MeetingSortformerModelLease.load()
        var stream: OmniDiarizationStream?
        do {
            let reference = try await reference(samples: samples, model: model)
            let policy = try model.configuration.feedPolicy()
            let activeStream = OmniDiarizationStream(endpoint: model.endpoint, options: OmniDiarizationOptions(
                spkcacheMax: policy.cacheMaximumFrames, fifoMax: MeetingSpeakerFeedPolicy.fifoMaximumFrames
            ))
            stream = activeStream
            var offset = 0
            var feedIndex = 0
            var flipped = 0
            var decisions = 0
            while offset < samples.count {
                let end = min(offset + policy.samplesPerFeed, samples.count)
                var padded = Array(samples[offset..<end])
                padded.append(contentsOf: repeatElement(0, count: max(0, policy.frameSamples - padded.count)))
                let feed = try await activeStream.feed(samples16k: padded)
                guard feedIndex < reference.feeds.count else {
                    XCTFail("Reference has too few feeds")
                    break
                }
                let expected = reference.feeds[feedIndex]
                XCTAssertEqual(padded.count, expected.sampleCount)
                XCTAssertEqual(feed.fifoLength, expected.fifoLength)
                XCTAssertEqual(feed.spkcacheLength, expected.spkcacheLength)
                XCTAssertEqual(feed.framesProcessed, expected.framesProcessed)
                XCTAssertEqual(feed.probabilities.count, expected.probabilities.count)
                for (row, originalRow) in zip(feed.probabilities, expected.probabilities) {
                    XCTAssertEqual(row.count, originalRow.count)
                    for (value, original) in zip(row, originalRow) {
                        XCTAssertTrue(original.isFinite && (0...1).contains(original))
                        XCTAssertEqual(value, original, accuracy: maxProbabilityDifference)
                        flipped += (value > 0.5) != (original > 0.5) ? 1 : 0
                        decisions += 1
                    }
                }
                offset = end
                feedIndex += 1
            }
            XCTAssertEqual(feedIndex, reference.feeds.count)
            XCTAssertGreaterThan(decisions, 0)
            XCTAssertLessThanOrEqual(Double(flipped), maxFlippedDecisions * Double(decisions))
        } catch {
            await stream?.close()
            await model.close()
            throw error
        }
        await stream?.close()
        await model.close()
    }

    func testEngineTurnsMatchSuppliedLibraryReference() async throws {
        let samples = try clip()
        let model = try await MeetingSortformerModelLease.load()
        let reference: Reference
        do {
            reference = try await self.reference(samples: samples, model: model)
        } catch {
            await model.close()
            throw error
        }
        await model.close()
        let engine = SortformerMeetingSpeakerDiarizationEngine()
        let asset = MeetingAudioAsset(source: .systemAudio, samples: samples, sampleRate: 16_000, sessionStartOffset: 0)
        let turns = try await engine.diarizeFile(
            descriptors: [MeetingAudioAssetDescriptor(source: .systemAudio, sampleRate: 16_000,
                                                      startSample: 0, sampleCount: samples.count)],
            loadAsset: { _ in asset }, options: .init(), progress: nil
        )
        XCTAssertFalse(reference.turns.isEmpty)
        let actual = try turns.map { turn -> Reference.Turn in
            let speaker = try XCTUnwrap(Int(turn.speakerID.replacingOccurrences(of: "sortformer-", with: "")))
            return Reference.Turn(speaker: speaker, start: turn.startSeconds, end: turn.endSeconds)
        }
        XCTAssertLessThanOrEqual(speechMismatch(reference.turns, actual), maxSpeechMismatch)
    }

    private func speechMismatch(_ expected: [Reference.Turn], _ actual: [Reference.Turn]) -> Double {
        func cells(_ turns: [Reference.Turn]) -> Set<Int> {
            var covered = Set<Int>()
            for turn in turns {
                let first = Int((turn.start * 100).rounded())
                let last = Int((turn.end * 100).rounded())
                if last > first {
                    covered.formUnion((first..<last).map { turn.speaker * 100_000_000 + $0 })
                }
            }
            return covered
        }
        let want = cells(expected)
        let have = cells(actual)
        let union = want.union(have).count
        return union == 0 ? 0 : Double(want.symmetricDifference(have).count) / Double(union)
    }
}
