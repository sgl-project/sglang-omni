// MeetingSummaryContextPlanningTests.swift
// Verifies bounded summary windows and relevance-based follow-up context.

import XCTest
@testable import Voxt

final class MeetingSummaryContextPlanningTests: XCTestCase {
    func testWindowsNeverExceedConfiguredLimit() {
        let transcript = (0..<20)
            .map { "paragraph-\($0)-" + String(repeating: "x", count: 80) }
            .joined(separator: "\n")

        let windows = MeetingSummaryContextPlanning.windows(
            for: transcript,
            maximumCharacters: 500
        )

        XCTAssertGreaterThan(windows.count, 1)
        XCTAssertTrue(windows.allSatisfy { $0.count <= 500 })
    }

    func testFollowUpContextSelectsRelevantWindow() {
        let transcript = [
            String(repeating: "opening discussion. ", count: 400),
            String(repeating: "budget approval and finance decision. ", count: 250),
            String(repeating: "closing remarks. ", count: 400)
        ].joined(separator: "\n")

        let context = MeetingSummaryContextPlanning.relevantFollowUpContext(
            transcript: transcript,
            question: "What was the budget decision?",
            maximumWindows: 1
        )

        XCTAssertTrue(context.contains("budget approval"))
        XCTAssertLessThanOrEqual(context.count, MeetingSummaryContextPlanning.followUpWindowCharacterLimit)
    }
}
