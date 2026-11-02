// MeetingAudioChunkingTests.swift
// Provides Meeting Audio Chunking Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class MeetingAudioChunkingTests: XCTestCase {
    func testChunkAccumulatorsUseSharedTimelineAcrossSpeakers() async {
        let me = MeetingChunkAccumulator(speaker: .me, speechThreshold: 0.012, profile: .quality)
        let them = MeetingChunkAccumulator(speaker: .them, speechThreshold: 0.025, profile: .quality)
        let speechSamples = [Float](repeating: 0.2, count: 19_200) // 0.4s @ 48kHz
        let silenceSamples = [Float](repeating: 0, count: 24_000) // 0.5s @ 48kHz

        _ = await them.append(
            samples: speechSamples,
            sampleRate: 48_000,
            level: 0.1,
            bufferEndSeconds: 0.4
        )
        let firstThem = await them.append(
            samples: silenceSamples,
            sampleRate: 48_000,
            level: 0,
            bufferEndSeconds: 0.9
        )

        _ = await me.append(
            samples: speechSamples,
            sampleRate: 48_000,
            level: 0.1,
            bufferEndSeconds: 1.6
        )
        let meChunk = await me.append(
            samples: silenceSamples,
            sampleRate: 48_000,
            level: 0,
            bufferEndSeconds: 2.1
        )

        _ = await them.append(
            samples: speechSamples,
            sampleRate: 48_000,
            level: 0.1,
            bufferEndSeconds: 2.8
        )
        let secondThem = await them.append(
            samples: silenceSamples,
            sampleRate: 48_000,
            level: 0,
            bufferEndSeconds: 3.3
        )

        XCTAssertNotNil(firstThem)
        XCTAssertNotNil(meChunk)
        XCTAssertNotNil(secondThem)
        XCTAssertEqual(firstThem?.speaker, .them)
        XCTAssertEqual(meChunk?.speaker, .me)
        XCTAssertEqual(secondThem?.speaker, .them)
        XCTAssertLessThan(firstThem?.startSeconds ?? .greatestFiniteMagnitude, meChunk?.startSeconds ?? 0)
        XCTAssertLessThan(meChunk?.startSeconds ?? .greatestFiniteMagnitude, secondThem?.startSeconds ?? 0)
    }

    func testQualityAccumulatorEmitsBoundedSevenSecondFinalChunks() async {
        let accumulator = MeetingChunkAccumulator(speaker: .them, speechThreshold: 0.025, profile: .quality)
        let speechSamples = [Float](repeating: 0.2, count: 19_200) // 0.4s @ 48kHz
        let silenceSamples = [Float](repeating: 0, count: 24_000) // 0.5s @ 48kHz

        for index in 1...17 {
            let chunk = await accumulator.append(
                samples: speechSamples,
                sampleRate: 48_000,
                level: 0.1,
                bufferEndSeconds: Double(index) * 0.4
            )
            XCTAssertNil(chunk)
        }

        let boundedChunk = await accumulator.append(
            samples: speechSamples,
            sampleRate: 48_000,
            level: 0.1,
            bufferEndSeconds: 7.2
        )

        XCTAssertNotNil(boundedChunk)
        XCTAssertEqual(boundedChunk?.speaker, .them)
        XCTAssertTrue(boundedChunk?.isFinal ?? false)
        XCTAssertFalse(boundedChunk?.preventsAdjacentMerge ?? true)
        XCTAssertEqual(boundedChunk?.startSeconds ?? -1, 0, accuracy: 0.001)
        XCTAssertEqual(boundedChunk?.endSeconds ?? -1, 7.2, accuracy: 0.001)

        _ = await accumulator.append(
            samples: speechSamples,
            sampleRate: 48_000,
            level: 0.1,
            bufferEndSeconds: 7.6
        )
        let pauseChunk = await accumulator.append(
            samples: silenceSamples,
            sampleRate: 48_000,
            level: 0,
            bufferEndSeconds: 8.1
        )

        XCTAssertNotNil(pauseChunk)
        XCTAssertTrue(pauseChunk?.isFinal ?? false)
        XCTAssertTrue(pauseChunk?.preventsAdjacentMerge ?? false)
        XCTAssertNotEqual(pauseChunk?.segmentID, boundedChunk?.segmentID)
    }

    func testRealtimeAccumulatorEmitsPartialAndShortFinalChunks() async {
        let accumulator = MeetingChunkAccumulator(speaker: .me, speechThreshold: 0.012, profile: .realtime)
        let speechSamples = [Float](repeating: 0.2, count: 14_400) // 0.3s @ 48kHz

        var partial: BufferedMeetingChunk?
        var finalChunk: BufferedMeetingChunk?
        for index in 1...20 {
            let chunk = await accumulator.append(
                samples: speechSamples,
                sampleRate: 48_000,
                level: 0.1,
                bufferEndSeconds: Double(index) * 0.3
            )
            if chunk?.isFinal == false, partial == nil {
                partial = chunk
            }
            if chunk?.isFinal == true {
                finalChunk = chunk
            }
        }

        XCTAssertNotNil(partial)
        XCTAssertFalse(partial?.isFinal ?? true)
        XCTAssertFalse(partial?.preventsAdjacentMerge ?? true)
        XCTAssertNotNil(finalChunk)
        XCTAssertTrue(finalChunk?.isFinal ?? false)
        XCTAssertEqual(finalChunk?.segmentID, partial?.segmentID)
        XCTAssertFalse(finalChunk?.preventsAdjacentMerge ?? true)
        XCTAssertEqual(partial?.endSeconds ?? -1, 2.1, accuracy: 0.001)
        XCTAssertEqual(finalChunk?.endSeconds ?? -1, 6.0, accuracy: 0.001)
    }
}
