import Foundation
import Testing
@testable import VoxtMobileKit

@Suite struct HistoryStatisticsTests {
    @Test func timeEstimateUsesMeasuredSessionsAndSubtractsSlowerResults() {
        let now = Date()
        let records = [
            TranscriptRecord(text: Array(repeating: "word", count: 40).joined(separator: " "), date: now, elapsedSeconds: 15),
            TranscriptRecord(text: Array(repeating: "word", count: 10).joined(separator: " "), date: now, elapsedSeconds: 20),
            TranscriptRecord(text: "Old history has no measured duration", date: now),
            TranscriptRecord(text: "Invalid duration", date: now, elapsedSeconds: -5)
        ]
        let stats = HistoryStatistics(records: records, now: now, typingWordsPerMinute: 40)
        #expect(stats.estimatedSecondsSaved == 40)
        #expect(stats.timedDictationCount == 2)
        #expect(HistoryStatistics(records: records, now: now, typingWordsPerMinute: 120).estimatedSecondsSaved == 0)
        #expect(HistoryStatistics(records: [.init(text: "Legacy")]).timedDictationCount == 0)
        #expect(HistoryStatistics(records: records, now: now, typingWordsPerMinute: 0).estimatedSecondsSaved == 40)
    }
    @Test func countsWordsDaysAndStreakFromSavedHistory() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let today = Date(timeIntervalSince1970: 1_700_000_000)
        let yesterday = today.addingTimeInterval(-86400)
        let records = [
            TranscriptRecord(text: "Hello world", date: today),
            TranscriptRecord(text: "One more", date: today),
            TranscriptRecord(text: "Yesterday", date: yesterday),
            TranscriptRecord(text: "Earlier", date: today.addingTimeInterval(-3 * 86400))
        ]
        let stats = HistoryStatistics(records: records, now: today, calendar: calendar)
        #expect(stats.wordCount == 6)
        #expect(stats.dictationCount == 4)
        #expect(stats.activeDays == 3)
        #expect(stats.currentStreak == 2)
        #expect(stats.longestStreak == 2)
        #expect(stats.dailyCounts[calendar.startOfDay(for: today)] == 2)
        #expect(HistoryStatistics(records: records, now: today.addingTimeInterval(2 * 86400), calendar: calendar).currentStreak == 0)
    }
}
