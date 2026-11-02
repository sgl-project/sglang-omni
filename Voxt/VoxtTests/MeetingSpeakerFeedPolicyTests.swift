import XCTest
@testable import Voxt

final class MeetingSpeakerFeedPolicyTests: XCTestCase {
    func testFeedIsAlignedAndFitsAOSCCompressionBudget() throws {
        for update in [6, 62, 188] {
            let policy = try makePolicy(update: update)
            XCTAssertLessThanOrEqual(policy.samplesPerFeed, 5 * 16_000)
            XCTAssertEqual(policy.samplesPerFeed % policy.frameSamples, 0)
            XCTAssertLessThanOrEqual(policy.samplesPerFeed / policy.frameSamples + 1, update)
        }
    }

    func testTwelveHourHistoryRemainsBoundedIncludingWindowTailPadding() throws {
        for update in [6, 188] {
            let policy = try makePolicy(update: update)
            var fifo = 0
            // Simulate upstream's AOSC one-pass retirement, including an extra
            // feature-padding frame per feed. Do not allocate twelve hours of PCM.
            for _ in 0..<(12 * 60) {
                var remaining = 60 * 16_000
                while remaining > 0 {
                    let samples = min(remaining, policy.samplesPerFeed)
                    let incoming = (samples + policy.frameSamples - 1) / policy.frameSamples + 1
                    fifo += incoming
                    fifo -= min(max(0, fifo - MeetingSpeakerFeedPolicy.fifoMaximumFrames), update)
                    try policy.validate(fifoFrames: fifo, cacheFrames: 188)
                    remaining -= samples
                }
            }
            XCTAssertLessThanOrEqual(fifo, 188)
        }
    }

    func testOldSixtySecondFeedsOutgrowCompressionBudget() {
        var fifo = 0
        for _ in 0..<20 {
            fifo += 60 * 16_000 / (160 * 8)
            fifo -= min(max(0, fifo - 188), 188)
        }
        XCTAssertEqual(fifo, 11_240)
        XCTAssertGreaterThan(fifo, 188)
    }

    func testInvalidConfigurationAndExcessStateStopRatherThanResetSpeakers() throws {
        XCTAssertThrowsError(try makePolicy(update: 0))
        XCTAssertThrowsError(try makePolicy(update: 2))
        let policy = try makePolicy()
        XCTAssertThrowsError(try policy.validate(fifoFrames: 189, cacheFrames: 188))
        XCTAssertThrowsError(try policy.validate(fifoFrames: 188, cacheFrames: 189))
        XCTAssertNoThrow(try policy.validateFeedDuration(.seconds(30)))
        XCTAssertThrowsError(try policy.validateFeedDuration(.seconds(31)))
    }

    func testTimelineUsesAudioSamplesNotAccumulatedFeaturePadding() throws {
        let policy = try makePolicy()
        let frames = 10_000
        let internalOffset = Double(frames * policy.frameSamples) / 16_000
        let mapped = try XCTUnwrap(policy.mappedRange(
            start: internalOffset + 0.5, end: internalOffset + 9,
            stateFrames: frames, audioOffset: 600, sampleCount: 16_000
        ))
        XCTAssertEqual(mapped.lowerBound, 600.5, accuracy: 0.001)
        XCTAssertEqual(mapped.upperBound, 601, accuracy: 0.001)
        XCTAssertNil(policy.mappedRange(start: .nan, end: 1, stateFrames: 0, audioOffset: 0, sampleCount: 100))
    }

    private func makePolicy(update: Int = 188) throws -> MeetingSpeakerFeedPolicy {
        try MeetingSpeakerFeedPolicy(sampleRate: 16_000, hopLength: 160, subsamplingFactor: 8,
                                     chunkFrames: 188, cacheFrames: 188, updateFrames: update, usesAOSC: true)
    }
}

@MainActor
final class MeetingFileSpeakerFailureTests: XCTestCase {
    func testFileSpeakerFailureDoesNotBecomeSuccessfulTranscriptOnlyHistory() async {
        let segment = MeetingTranscriptSegment(speaker: .them, startSeconds: 0, endSeconds: 2, text: "preserved ASR")
        let descriptor = MeetingAudioAssetDescriptor(source: .mixed, sampleRate: 16_000, startSample: 0, sampleCount: 32_000)
        let asset = MeetingAudioAsset(
            source: .mixed,
            samples: Array(repeating: 0, count: descriptor.sampleCount),
            sampleRate: descriptor.sampleRate,
            sessionStartOffset: 0
        )
        do {
            _ = try await MeetingSpeakerAnalysisPipeline.analyzedFileSegments(
                from: [segment], descriptors: [descriptor], loadAsset: { _ in asset }, engine: FailingSpeakerEngine()
            )
            XCTFail("Speaker errors must propagate so the ASR checkpoint is retained")
        } catch {
            XCTAssertTrue(error is MeetingSpeakerFeedError)
        }
    }
}

private struct FailingSpeakerEngine: MeetingSpeakerDiarizationEngine {
    func diarize(asset: MeetingAudioAsset, options: MeetingSpeakerDiarizationOptions) async throws -> [MeetingSpeakerTurn] {
        throw MeetingSpeakerFeedError.stateLimitExceeded
    }

    func diarizeSession(descriptors: [MeetingAudioAssetDescriptor],
                        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
                        continuousAudioURL: URL?, options: MeetingSpeakerDiarizationOptions,
                        progress: (@Sendable (Double) async -> Void)?) async throws -> [MeetingSpeakerTurn] {
        throw MeetingSpeakerFeedError.stateLimitExceeded
    }

    func diarizeFile(descriptors: [MeetingAudioAssetDescriptor],
                     loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
                     options: MeetingSpeakerDiarizationOptions,
                     progress: (@Sendable (Double) async -> Void)?) async throws -> [MeetingSpeakerTurn] {
        throw MeetingSpeakerFeedError.stateLimitExceeded
    }
}
