// RelativeNoteTimestampFormatter.swift
// Provides Relative Note Timestamp Formatter for history storage and display support.

import Foundation

enum RelativeNoteTimestampFormatter {
    static func historyListTime(
        for date: Date,
        calendar: Calendar = .current
    ) -> String {
        let components = calendar.dateComponents([.hour, .minute], from: date)
        return String(
            format: "%02d:%02d",
            locale: Locale(identifier: "en_US_POSIX"),
            components.hour ?? 0,
            components.minute ?? 0
        )
    }

    static func noteCardTimestamp(for date: Date, now: Date = Date()) -> String? {
        historyCardTimestamp(for: date, now: now)
    }

    static func noteHistoryTimestamp(for date: Date, now: Date = Date()) -> String {
        historyCardTimestamp(for: date, now: now)
    }

    static func historyCardTimestamp(
        for date: Date,
        now: Date = Date(),
        calendar: Calendar = .current
    ) -> String {
        let startOfToday = calendar.startOfDay(for: now)
        let startOfDate = calendar.startOfDay(for: date)
        let dayDifference = calendar.dateComponents([.day], from: startOfDate, to: startOfToday).day ?? 0

        switch dayDifference {
        case ...0:
            return AppLocalization.localizedString("Today")
        case 1:
            return AppLocalization.localizedString("Yesterday")
        case 2..<7:
            return AppLocalization.format("%d days ago", dayDifference)
        default:
            return absoluteDate(for: date)
        }
    }

    private static func absoluteDate(for date: Date) -> String {
        date.formatted(
            .dateTime
                .locale(AppLocalization.locale)
                .year()
                .month(.abbreviated)
                .day()
        )
    }
}
