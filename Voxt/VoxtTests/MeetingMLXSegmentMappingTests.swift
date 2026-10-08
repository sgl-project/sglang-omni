import XCTest
@testable import Voxt
import MLXAudioSTT

final class MeetingMLXSegmentMappingTests: XCTestCase {
    func testStructuredOutputRequiresFinalChunkAndReliableTiming() {
        XCTAssertTrue(
            MeetingMLXSegmentMapping.shouldUseStructuredOutput(
                isFinalChunk: true,
                timingGranularity: .sentence
            )
        )
        XCTAssertFalse(
            MeetingMLXSegmentMapping.shouldUseStructuredOutput(
                isFinalChunk: false,
                timingGranularity: .sentence
            )
        )
        XCTAssertFalse(
            MeetingMLXSegmentMapping.shouldUseStructuredOutput(
                isFinalChunk: true,
                timingGranularity: .chunk
            )
        )
    }

    func testParakeetStructuredSegmentsImproveBoundariesWithoutSpeakerIDs() {
        let segmentID = UUID()
        let result = MLXBufferedTranscriptionResult(
            text: "Hello world",
            structuredSegments: [
                MLXStructuredTranscriptSegment(startSeconds: 0.1, endSeconds: 0.8, text: "Hello"),
                MLXStructuredTranscriptSegment(startSeconds: 0.9, endSeconds: 1.5, text: "world"),
            ]
        )

        let mapped = MeetingMLXSegmentMapping.meetingSegments(
            from: result,
            segmentID: segmentID,
            speaker: .me,
            audioSource: .microphone,
            startSeconds: 10,
            endSeconds: 20,
            usesStructuredOutput: true,
            modelFamily: .parakeet,
            preventsAdjacentMerge: false,
            dictionaryEntries: [],
            speakerDisplayName: { _ in "ShouldNotAppear" }
        )

        XCTAssertEqual(mapped.count, 2)
        XCTAssertEqual(mapped[0].id, segmentID)
        XCTAssertEqual(mapped[0].text, "Hello")
        XCTAssertEqual(mapped[0].startSeconds, 10.1, accuracy: 0.0001)
        XCTAssertEqual(mapped[0].endSeconds ?? -1, 10.8, accuracy: 0.0001)
        XCTAssertNil(mapped[0].speakerID)
        XCTAssertNil(mapped[0].speakerDisplayName)
        XCTAssertNil(mapped[1].speakerID)
    }

    func testWhisperFallsBackToWholeChunkWithoutFakeTimeline() {
        let segmentID = UUID()
        let result = MLXBufferedTranscriptionResult(
            text: "Whole chunk",
            structuredSegments: []
        )

        let mapped = MeetingMLXSegmentMapping.meetingSegments(
            from: result,
            segmentID: segmentID,
            speaker: .me,
            audioSource: .microphone,
            startSeconds: 3,
            endSeconds: 7,
            usesStructuredOutput: false,
            modelFamily: .whisper,
            preventsAdjacentMerge: true,
            dictionaryEntries: [],
            speakerDisplayName: { _ in nil }
        )

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0].text, "Whole chunk")
        XCTAssertEqual(mapped[0].startSeconds, 3)
        XCTAssertEqual(mapped[0].endSeconds, 7)
        XCTAssertNil(mapped[0].speakerID)
    }

    func testMOSSStructuredSegmentsKeepSpeakerLabels() {
        let result = MLXBufferedTranscriptionResult(
            text: "raw",
            structuredSegments: [
                MLXStructuredTranscriptSegment(
                    startSeconds: 0.0,
                    endSeconds: 1.0,
                    speakerID: "1",
                    text: "Hello from moss"
                )
            ]
        )

        let mapped = MeetingMLXSegmentMapping.meetingSegments(
            from: result,
            segmentID: UUID(),
            speaker: .them,
            audioSource: .systemAudio,
            startSeconds: 0,
            endSeconds: 5,
            usesStructuredOutput: true,
            modelFamily: .mossTranscribeDiarize,
            preventsAdjacentMerge: true,
            dictionaryEntries: [],
            speakerDisplayName: { id in "Speaker \(id)" }
        )

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0].speakerID, "moss:1")
        XCTAssertEqual(mapped[0].speakerDisplayName, "Speaker 1")
        XCTAssertEqual(mapped[0].text, "Hello from moss")
    }
}
