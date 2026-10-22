import XCTest
@testable import VoxtMobile
import VoxtMobileKit

/// Home: no numbers until there is data, then one "this week" line.
final class HomeSummaryTests: XCTestCase {
    private var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        calendar.firstWeekday = 2
        return calendar
    }
    // A Thursday, 12:00 UTC: two days back is this week's Tuesday, six days back last week's Friday.
    private let now = Date(timeIntervalSince1970: 1_792_065_600)

    func testNothingThisWeekShowsNoLine() {
        XCTAssertNil(WeekSummary(records: [], now: now, calendar: calendar))
        let lastWeek = TranscriptRecord(text: "old words here", date: now.addingTimeInterval(-8 * 86_400))
        XCTAssertNil(WeekSummary(records: [lastWeek], now: now, calendar: calendar))
    }

    func testCountsWordsAndDaysThisWeekOnly() throws {
        let records = [
            TranscriptRecord(text: "one two three", date: now),
            TranscriptRecord(text: "four five", date: now.addingTimeInterval(-3_600)),
            TranscriptRecord(text: "six", date: now.addingTimeInterval(-2 * 86_400)),
            TranscriptRecord(text: "not this week", date: now.addingTimeInterval(-6 * 86_400)),
        ]
        let summary = try XCTUnwrap(WeekSummary(records: records, now: now, calendar: calendar))
        XCTAssertEqual(summary.words, 6)
        XCTAssertEqual(summary.days, 2)
    }

    @MainActor
    func testADictationThatJustLandedReadsAsNowNotInTheFuture() {
        // A result stamped a moment after the clock reads must not show as "in 0 sec.".
        for offset in [-1.0, 0, 1, 20] {
            let label = DictationView.when(now.addingTimeInterval(offset), now: now)
            XCTAssertFalse(label.contains(where: \.isNumber), "\(offset): \(label)")
        }
    }
}
