import XCTest
@testable import Voxt

final class MeetingLiveTranscriptPresentationTests: XCTestCase {
    func testMeetingModeStripsSpeakerAnalysisButKeepsSourceAndIdentity() {
        let segment = MeetingTranscriptSegment(
            speaker: .me, speakerID: "speaker-1", speakerDisplayName: "Name",
            audioSource: .microphone, speakerConfidence: 0.9,
            startSeconds: 0, endSeconds: 1, text: "hello"
        )
        let events = MeetingLiveTranscriptPresentation.events(for: .partial(segment), captureMode: .meeting, segments: [])
        guard case .partial(let normalized)? = events.first else { return XCTFail("Expected partial") }
        XCTAssertEqual(normalized.id, segment.id)
        XCTAssertEqual(normalized.audioSource, .microphone)
        XCTAssertNil(normalized.speakerID)
        XCTAssertNil(normalized.speakerDisplayName)
        XCTAssertNil(normalized.speakerConfidence)
    }

    func testRecordingModePreservesSpeakerAnalysis() {
        let segment = MeetingTranscriptSegment(speaker: .me, speakerID: "speaker-1", startSeconds: 0, endSeconds: 1, text: "hello")
        let events = MeetingLiveTranscriptPresentation.events(for: .partial(segment), captureMode: .recording, segments: [])
        guard case .partial(let normalized)? = events.first else { return XCTFail("Expected partial") }
        XCTAssertEqual(normalized.speakerID, "speaker-1")
    }

    func testShortAdjacentFinalRetainsPreviousIdentityAndClearsTranslation() {
        let previous = MeetingTranscriptSegment(speaker: .me, startSeconds: 0, endSeconds: 1, text: "hello", translatedText: "old translation")
        let next = MeetingTranscriptSegment(speaker: .me, startSeconds: 1.1, endSeconds: 2, text: "world")
        let events = MeetingLiveTranscriptPresentation.events(for: .final(next), captureMode: .recording, segments: [previous])
        guard case .final(let merged)? = events.first else { return XCTFail("Expected final") }
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(merged.id, previous.id)
        XCTAssertTrue(merged.text.contains("hello"))
        XCTAssertTrue(merged.text.contains("world"))
        XCTAssertNil(merged.translatedText)
        XCTAssertTrue(merged.preventsAdjacentMerge)
    }

    func testDifferentSpeakerOrLongGapDoesNotMerge() {
        let previous = MeetingTranscriptSegment(speaker: .me, startSeconds: 0, endSeconds: 1, text: "hello")
        for next in [
            MeetingTranscriptSegment(speaker: .them, startSeconds: 1.1, endSeconds: 2, text: "world"),
            MeetingTranscriptSegment(speaker: .me, startSeconds: 10, endSeconds: 11, text: "world")
        ] {
            let events = MeetingLiveTranscriptPresentation.events(for: .final(next), captureMode: .recording, segments: [previous])
            guard case .final(let result)? = events.first else { return XCTFail("Expected final") }
            XCTAssertEqual(result.id, next.id)
        }
    }

    func testFinalizationClearsOnlyPendingTranslationFlag() {
        let segment = MeetingTranscriptSegment(
            speaker: .me, startSeconds: 0, endSeconds: 1, text: "hello",
            translatedText: "translation", isTranslationPending: true
        )
        let finalized = MeetingLiveTranscriptPresentation.finalizedSegments(from: [segment])
        XCTAssertEqual(finalized.first?.id, segment.id)
        XCTAssertEqual(finalized.first?.translatedText, "translation")
        XCTAssertEqual(finalized.first?.isTranslationPending, false)
    }
}
