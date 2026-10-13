// MeetingWaveformTimelineTests.swift
// Tests waveform generation, timeline mapping, and transcript compatibility.

import XCTest
@testable import Voxt

@MainActor
final class MeetingWaveformTimelineTests: XCTestCase {
    func testTimelineTimeMappingClampsToAudioBounds() {
        XCTAssertEqual(
            MeetingWaveformTimelineSupport.time(forX: -20, width: 100, duration: 10),
            0
        )
        XCTAssertEqual(
            MeetingWaveformTimelineSupport.time(forX: 50, width: 100, duration: 10),
            5
        )
        XCTAssertEqual(
            MeetingWaveformTimelineSupport.time(forX: 140, width: 100, duration: 10),
            10
        )
    }

    func testZoomedTimelineCentersOnCurrentPlaybackTime() {
        let range = MeetingWaveformTimelineSupport.visibleTimeRange(
            currentTime: 50,
            duration: 100,
            zoomScale: 2
        )

        XCTAssertEqual(range.start, 25)
        XCTAssertEqual(range.end, 75)
    }

    func testZoomedTimelineTimeMappingUsesVisibleRange() {
        XCTAssertEqual(
            MeetingWaveformTimelineSupport.time(
                forX: 0,
                width: 100,
                duration: 100,
                currentTime: 50,
                zoomScale: 2
            ),
            25
        )
        XCTAssertEqual(
            MeetingWaveformTimelineSupport.time(
                forX: 50,
                width: 100,
                duration: 100,
                currentTime: 50,
                zoomScale: 2
            ),
            50
        )
        XCTAssertEqual(
            MeetingWaveformTimelineSupport.time(
                forX: 100,
                width: 100,
                duration: 100,
                currentTime: 50,
                zoomScale: 2
            ),
            75
        )
    }

    func testZoomScaleIsClampedToSupportedRange() {
        XCTAssertEqual(
            MeetingWaveformTimelineSupport.clampedZoomScale(0.2),
            MeetingWaveformTimelineSupport.minimumZoomScale
        )
        XCTAssertEqual(
            MeetingWaveformTimelineSupport.clampedZoomScale(20),
            MeetingWaveformTimelineSupport.maximumZoomScale
        )
    }

    func testLegacyTranscriptSegmentDecodesWithoutHighlightField() throws {
        let segment = TranscriptSegment(
            speaker: .them,
            startSeconds: 1,
            endSeconds: 2,
            text: "legacy"
        )
        var json = try XCTUnwrap(
            try JSONSerialization.jsonObject(
                with: JSONEncoder().encode(segment),
                options: []
            ) as? [String: Any]
        )
        json.removeValue(forKey: "isHighlighted")

        let decoded = try JSONDecoder().decode(
            TranscriptSegment.self,
            from: JSONSerialization.data(withJSONObject: json)
        )
        XCTAssertFalse(decoded.isHighlighted)
    }

    func testWaveformBuilderProducesBoundedSamples() async throws {
        let fixtureURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/Audio/short_zh_yang_jiechi.ogg")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: fixtureURL.path))

        let waveformResult = await MeetingWaveformBuilder.load(from: fixtureURL, sampleCount: 64)
        let waveform = try XCTUnwrap(waveformResult)
        XCTAssertEqual(waveform.samples.count, 64)
        XCTAssertGreaterThan(waveform.duration, 0)
        XCTAssertTrue(waveform.samples.allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    func testWaveformCacheKeepsSampleCountInItsKey() async throws {
        let fixtureURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/Audio/short_zh_yang_jiechi.ogg")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: fixtureURL.path))

        let waveform = await MeetingWaveformBuilder.load(from: fixtureURL, sampleCount: 32)
        XCTAssertEqual(waveform?.samples.count, 32)

        let differentSizeWaveform = await MeetingWaveformBuilder.load(from: fixtureURL, sampleCount: 64)
        XCTAssertEqual(differentSizeWaveform?.samples.count, 64)
    }
}
