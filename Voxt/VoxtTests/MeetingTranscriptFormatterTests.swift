// MeetingTranscriptFormatterTests.swift
// Provides Meeting Transcript Formatter Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class MeetingTranscriptFormatterTests: XCTestCase {
    func testLLMInputTextUsesSpeakerOrderWithoutTimestampOrTranslation() {
        let segments = [
            MeetingTranscriptSegment(
                speaker: .me,
                startSeconds: 1,
                endSeconds: 3,
                text: "Hello there.",
                translatedText: "你好。"
            ),
            MeetingTranscriptSegment(
                speaker: .them,
                startSeconds: 4,
                endSeconds: 7,
                text: "Let's ship on Friday.",
                translatedText: "我们周五发布吧。"
            )
        ]

        let text = MeetingTranscriptFormatter.llmInputText(for: segments)

        XCTAssertEqual(text, "Me：Hello there.\nThem：Let's ship on Friday.")
        XCTAssertFalse(text.contains("00:01"))
        XCTAssertFalse(text.contains("->"))
        XCTAssertFalse(text.contains("你好"))
    }
}
