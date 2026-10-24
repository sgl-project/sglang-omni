import Foundation
import Testing
import TypingInput
@testable import PinyinInput

/// 简拼: initials stand for whole syllables (nh → 你好).
@Suite struct PinyinAbbreviationTests {
    @Test(arguments: [("nh", "你好"), ("nih", "你好"), ("nhao", "你好"), ("wmen", "我们")])
    func initialsConvertToTheWord(input: String, expected: String) async throws {
        let result = try await PinyinConverter().candidates(typing: input)
        #expect(result.first?.text == expected, "\(result.prefix(5).map(\.text))")
        #expect(result.first?.syllables == 2)
        #expect(result.first?.reading == "\(expected == "我们" ? "wo-men" : "ni-hao")", "the dictionary's reading, for learning")
    }

    @Test func anInitialOnlyMatchesItsOwnInitial() async throws {
        let result = try await PinyinConverter().candidates(typing: "zg")
        #expect(result.prefix(10).map(\.text).contains("中国") == false, "z is not zh: \(result.prefix(5).map(\.text))")
        let zhg = try await PinyinConverter().candidates(typing: "zhg").prefix(5).map(\.text)
        #expect(zhg.contains("中国") && zhg.contains("这个"), "\(zhg)")
    }

    @Test func eachCandidateSaysItsReading() async throws {
        let result = try await PinyinConverter().candidates(typing: "nihao")
        #expect(result.first?.reading == "ni-hao")
        #expect(result.first { $0.text == "你" }?.reading == "ni")
    }

    /// More pieces than the engine takes: the start converts, the rest stays typed.
    @Test func veryLongInputConvertsItsStart() async throws {
        let result = try await PinyinConverter().candidates(typing: String(repeating: "nh", count: 20))
        let first = try #require(result.first)
        #expect(first.syllables <= PinyinConverter.maximumPieces)
        let source = PinyinCandidateSource()
        let candidates = try await source.candidates(for: ConversionRequest(input: [String(repeating: "x", count: 64)]))
        #expect(!candidates.isEmpty)
    }

    /// The keyboard cancels a conversion when the next key arrives; queued work is skipped.
    @Test func aCancelledConversionDoesNoWork() async throws {
        let source = PinyinCandidateSource()
        let task = Task { () -> [Candidate] in
            withUnsafeCurrentTask { $0?.cancel() }
            return try await source.candidates(for: ConversionRequest(input: ["zysbdzsm"]))
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    /// Cold (nothing cached), the worst inputs typed at once; the phone is slower than this Mac.
    @Test func coldAbbreviationsStayFast() async throws {
        var worst = 0.0
        for input in ["zysbdzsm", "wxqsyxxx", String(repeating: "bpmfdtnl", count: 3), String(repeating: "zhshch", count: 8)] {
            let converter = try PinyinConverter()
            let start = Date()
            _ = try await converter.candidates(typing: input)
            worst = max(worst, Date().timeIntervalSince(start))
        }
        print("abbreviation cold worst: \(Int(worst * 1000)) ms")
        #expect(worst < 0.25, "\(worst)s")
    }

    @Test func longAbbreviationsStayFast() async throws {
        let converter = try PinyinConverter()
        let start = Date()
        for input in ["zhshzhsh", "nhnhnhnh", "zysbdzsm", "wxqsyxxx"] {
            _ = try await converter.candidates(typing: input)
        }
        let elapsed = Date().timeIntervalSince(start)
        print("abbreviation timing 4 inputs: \(elapsed)s")
        #expect(elapsed < 2, "\(elapsed)s")
    }
}

/// Fuzzy z/zh, c/ch, s/sh, off unless the user turns it on.
@Suite struct PinyinFuzzyTests {
    @Test func offByDefault() async throws {
        let plain = try await PinyinConverter().candidates(typing: "zongguo").first?.text
        #expect(plain != "中国", "\(plain ?? "-")")
    }

    @Test(arguments: [("zongguo", "中国"), ("sishi", "事实"), ("zuang", "装")])
    func eitherSpellingFindsTheWord(input: String, expected: String) async throws {
        let result = try await PinyinConverter(fuzzy: true).candidates(typing: input).prefix(5).map(\.text)
        #expect(result.contains(expected), "\(result)")
    }

    @Test func bothSpellingsShowUp() async throws {
        let result = try await PinyinConverter(fuzzy: true).candidates(typing: "si").prefix(10).map(\.text)
        #expect(result.contains("四") && result.contains("是"), "\(result)")
    }
}
