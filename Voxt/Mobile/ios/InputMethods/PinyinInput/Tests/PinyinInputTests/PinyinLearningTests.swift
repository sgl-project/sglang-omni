import Foundation
import Testing
@testable import PinyinInput

/// Picks are remembered on the phone and put first next time.
@Suite struct PinyinLearningTests {
    private func temporaryFile() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("pinyin-learning-\(UUID().uuidString).tsv")
    }

    @Test func aPickedWordComesFirstForItsReading() async throws {
        let learning = PinyinLearning(fileURL: temporaryFile())
        defer { learning.clear() }
        let converter = try PinyinConverter(learning: learning)
        #expect(try await converter.candidates(typing: "nihao").first?.text == "你好")
        learning.record(reading: "ni-hao", text: "拟好")
        #expect(try await converter.candidates(typing: "nihao").first?.text == "拟好")
        #expect(try await converter.candidates(typing: "nh").first?.text == "拟好", "the abbreviation finds it too")
    }

    @Test func aPickedShorterWordComesRightAfterTheSentence() async throws {
        let learning = PinyinLearning(fileURL: temporaryFile())
        defer { learning.clear() }
        learning.record(reading: "ni", text: "泥")
        let result = try await PinyinConverter(learning: learning).candidates(typing: "nihao")
        #expect(result.prefix(2).map(\.text) == ["你好", "泥"])
        #expect(result.filter { $0.text == "泥" }.count == 1)
    }

    @Test func theMostRecentPickWins() async throws {
        let learning = PinyinLearning(fileURL: temporaryFile())
        defer { learning.clear() }
        let start = Date(timeIntervalSince1970: 1_000)
        learning.record(reading: "ni-hao", text: "你好", now: start)
        learning.record(reading: "ni-hao", text: "拟好", now: start.addingTimeInterval(60))
        #expect(try await PinyinConverter(learning: learning).candidates(typing: "nihao").first?.text == "拟好")
        learning.record(reading: "ni-hao", text: "你好", now: start.addingTimeInterval(120))
        #expect(try await PinyinConverter(learning: learning).candidates(typing: "nihao").first?.text == "你好")
        #expect(learning.entries.count == 2)
    }

    @Test func picksSurviveAndClearingForgetsThemEverywhere() throws {
        let url = temporaryFile()
        let first = PinyinLearning(fileURL: url)
        first.record(reading: "ni-hao", text: "拟好")
        first.record(reading: "ni-hao", text: "拟好")
        let second = PinyinLearning(fileURL: url)
        #expect(second.entries.map(\.text) == ["拟好"])
        #expect(second.entries.first?.count == 2)
        PinyinLearning.clear(fileURL: url)
        #expect(!FileManager.default.fileExists(atPath: url.path))
        #expect(first.entries.isEmpty, "another instance notices the file is gone")
    }

    @Test func theOldestPicksGoFirstWhenFull() {
        let learning = PinyinLearning(fileURL: temporaryFile(), capacity: 3)
        defer { learning.clear() }
        let start = Date(timeIntervalSince1970: 1_000)
        for (offset, text) in ["一", "二", "三", "四"].enumerated() {
            learning.record(reading: "x", text: text, now: start.addingTimeInterval(Double(offset)))
        }
        #expect(Set(learning.entries.map(\.text)) == ["二", "三", "四"])
    }

    @Test func noFileMeansMemoryOnly() async throws {
        let learning = PinyinLearning(fileURL: nil)
        learning.record(reading: "ni-hao", text: "拟好")
        #expect(try await PinyinConverter(learning: learning).candidates(typing: "nihao").first?.text == "拟好")
    }
}
