import Foundation
import NaturalLanguage

/// Statistics describe retained local history; deleting history changes these totals.
public struct HistoryStatistics {
    public let wordCount: Int
    public let dictationCount: Int
    public let dailyCounts: [Date: Int]
    public var activeDays: Int { dailyCounts.count }
    public let currentStreak: Int
    public let longestStreak: Int
    public let estimatedSecondsSaved: TimeInterval
    public let timedDictationCount: Int

    public init(records: [TranscriptRecord], now: Date = Date(), calendar: Calendar = .current, typingWordsPerMinute: Double = 40) {
        let records = records.filter { $0.date <= now && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        dictationCount = records.count
        wordCount = records.reduce(0) { $0 + Self.words(in: $1.text) }
        let speed = typingWordsPerMinute.isFinite && (10...120).contains(typingWordsPerMinute) ? typingWordsPerMinute : 40
        let timed = records.filter { $0.elapsedSeconds.map { $0.isFinite && $0 >= 0 } == true }
        timedDictationCount = timed.count
        // This is an estimate against a user-selected typing speed, not measured typing.
        // Slower results reduce the total; legacy rows never inflate it.
        estimatedSecondsSaved = max(0, timed.reduce(0) {
            $0 + Double(Self.words(in: $1.text)) * 60 / speed - ($1.elapsedSeconds ?? 0)
        })
        let days = Dictionary(grouping: records) { calendar.startOfDay(for: $0.date) }.mapValues(\.count)
        dailyCounts = days
        let sorted = days.keys.sorted()
        var longest = 0
        var run = 0
        var previous: Date?
        for day in sorted {
            run = previous.flatMap { calendar.date(byAdding: .day, value: 1, to: $0) } == day ? run + 1 : 1
            longest = max(longest, run)
            previous = day
        }
        longestStreak = longest
        let today = calendar.startOfDay(for: now)
        var cursor = days[today] == nil ? calendar.date(byAdding: .day, value: -1, to: today) : today
        var current = 0
        while let day = cursor, days[day] != nil {
            current += 1
            cursor = calendar.date(byAdding: .day, value: -1, to: day)
        }
        currentStreak = current
    }

    /// Uses the system's word segmentation, including languages without spaces.
    private static func words(in text: String) -> Int {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        return tokenizer.tokens(for: text.startIndex..<text.endIndex).count
    }
}
