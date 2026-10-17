// RelativeNoteTimestampFormatterTests.swift
// Provides Relative Note Timestamp Formatter Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class RelativeNoteTimestampFormatterTests: XCTestCase {
    private var originalInterfaceLanguage: String?

    override func setUp() {
        super.setUp()
        originalInterfaceLanguage = UserDefaults.standard.string(forKey: AppPreferenceKey.interfaceLanguage)
        UserDefaults.standard.set(AppInterfaceLanguage.english.rawValue, forKey: AppPreferenceKey.interfaceLanguage)
        AppLocalization.refreshLanguageCache()
    }

    override func tearDown() {
        if let originalInterfaceLanguage {
            UserDefaults.standard.set(originalInterfaceLanguage, forKey: AppPreferenceKey.interfaceLanguage)
        } else {
            UserDefaults.standard.removeObject(forKey: AppPreferenceKey.interfaceLanguage)
        }
        AppLocalization.refreshLanguageCache()
        super.tearDown()
    }

    func testHistoryCardTimestampUsesCalendarDayLabelsThroughSixDays() throws {
        let calendar = utcCalendar()
        let now = try XCTUnwrap(calendar.date(from: DateComponents(year: 2026, month: 7, day: 12, hour: 10)))

        XCTAssertEqual(timestamp(daysBefore: 0, now: now, calendar: calendar), "Today")
        XCTAssertEqual(timestamp(daysBefore: 1, now: now, calendar: calendar), "Yesterday")
        XCTAssertEqual(timestamp(daysBefore: 2, now: now, calendar: calendar), "2 days ago")
        XCTAssertEqual(timestamp(daysBefore: 6, now: now, calendar: calendar), "6 days ago")
    }

    func testHistoryCardTimestampUsesAbsoluteDateAtSevenDays() throws {
        let calendar = utcCalendar()
        let now = try XCTUnwrap(calendar.date(from: DateComponents(year: 2026, month: 7, day: 12, hour: 10)))
        let sevenDaysAgo = try XCTUnwrap(calendar.date(byAdding: .day, value: -7, to: now))

        let text = RelativeNoteTimestampFormatter.historyCardTimestamp(
            for: sevenDaysAgo,
            now: now,
            calendar: calendar
        )

        XCTAssertFalse(text.isEmpty)
        XCTAssertNotEqual(text, "7 days ago")
    }

    func testHistoryCardTimestampUsesCalendarBoundaryForYesterday() throws {
        let calendar = utcCalendar()
        let now = try XCTUnwrap(calendar.date(from: DateComponents(year: 2026, month: 7, day: 12, hour: 0, minute: 30)))
        let previousNight = try XCTUnwrap(calendar.date(from: DateComponents(year: 2026, month: 7, day: 11, hour: 23, minute: 30)))

        XCTAssertEqual(
            RelativeNoteTimestampFormatter.historyCardTimestamp(
                for: previousNight,
                now: now,
                calendar: calendar
            ),
            "Yesterday"
        )
    }

    func testHistoryListTimeUsesTwoDigitHourAndMinuteOnly() throws {
        let calendar = utcCalendar()
        let date = try XCTUnwrap(calendar.date(
            from: DateComponents(year: 2026, month: 7, day: 12, hour: 7, minute: 5, second: 42)
        ))

        XCTAssertEqual(
            RelativeNoteTimestampFormatter.historyListTime(for: date, calendar: calendar),
            "07:05"
        )
    }

    private func timestamp(daysBefore: Int, now: Date, calendar: Calendar) -> String {
        let date = calendar.date(byAdding: .day, value: -daysBefore, to: now)!
        return RelativeNoteTimestampFormatter.historyCardTimestamp(
            for: date,
            now: now,
            calendar: calendar
        )
    }

    private func utcCalendar() -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        return calendar
    }
}
