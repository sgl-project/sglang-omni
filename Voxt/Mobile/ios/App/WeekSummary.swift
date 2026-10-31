import Foundation
import VoxtMobileKit

/// Home's one activity line: words and days in this calendar week, or nil before any.
struct WeekSummary: Equatable {
    var words: Int
    var days: Int

    init?(records: [TranscriptRecord], now: Date = Date(), calendar: Calendar = .current) {
        guard let week = calendar.dateInterval(of: .weekOfYear, for: now) else { return nil }
        let thisWeek = records.filter { week.contains($0.date) }
        guard !thisWeek.isEmpty else { return nil }
        let stats = HistoryStatistics(records: thisWeek, now: now, calendar: calendar)
        words = stats.wordCount
        days = stats.activeDays
    }
}
