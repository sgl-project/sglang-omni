import Testing
@testable import PinyinInput

@Suite struct PinyinSegmenterTests {
    let segmenter = PinyinSegmenter()

    @Test(arguments: [
        ("nihao", ["ni", "hao"]),
        ("woxiangqu", ["wo", "xiang", "qu"]),
        ("weixin", ["wei", "xin"]),
        ("xian", ["xian"]),
        ("xi'an", ["xi", "an"]),
        ("dachuang", ["da", "chuang"]),
        ("wanan", ["wan", "an"]),
        ("lvse", ["lv", "se"]),
        ("nve", ["nve"]),
        ("a'a", ["a", "a"]),
        ("zhuangtai", ["zhuang", "tai"]),
    ])
    func completeInput(input: String, syllables: [String]) {
        #expect(segmenter.split(input) == .init(syllables: syllables, rest: ""))
    }

    /// 简拼: an initial stands for a whole syllable, but only where full syllables don't fit.
    @Test(arguments: [
        ("nh", ["n", "h"]),
        ("nih", ["ni", "h"]),
        ("nizh", ["ni", "zh"]),
        ("xiangq", ["xiang", "q"]),
        ("zhn", ["zh", "n"]),
        ("ng", ["n", "g"]),
        ("zhshr", ["zh", "sh", "r"]),
        ("sh", ["sh"]),
        ("nhao", ["n", "hao"]),
    ])
    func initialsStandForSyllables(input: String, syllables: [String]) {
        #expect(segmenter.split(input) == .init(syllables: syllables, rest: ""))
    }

    @Test(arguments: [
        ("xi'", ["xi"], "'"),
        ("nii", ["ni"], "i"),
        ("u", [], "u"),
        ("", [], ""),
    ])
    func lettersNoSyllableCanStartWithStayRaw(input: String, syllables: [String], rest: String) {
        #expect(segmenter.split(input) == .init(syllables: syllables, rest: rest))
    }

    @Test func fullSyllablesWinOverInitials() {
        #expect(segmenter.splits("xian").map(\.syllables) == [["xian"]])
        #expect(segmenter.splits("fangan").map(\.syllables) == [["fang", "an"], ["fan", "gan"]])
        #expect(segmenter.split("dangan").syllables.allSatisfy(PinyinSegmenter.syllables.contains))
        #expect(PinyinSegmenter.initials.isDisjoint(with: PinyinSegmenter.syllables))
        #expect(PinyinSegmenter.initials.count == 23)
    }

    /// Fuzzy z/zh, c/ch, s/sh: spellings that only exist on the other side split as one syllable.
    @Test func fuzzySplitsTheOtherSpelling() {
        let fuzzy = PinyinSegmenter(fuzzy: true)
        #expect(fuzzy.split("zuang") == .init(syllables: ["zuang"], rest: ""))
        #expect(fuzzy.split("shei") == .init(syllables: ["shei"], rest: ""))
        #expect(fuzzy.split("cuai") == .init(syllables: ["cuai"], rest: ""))
        #expect(segmenter.split("zuang").syllables != ["zuang"], "off by default")
        #expect(fuzzy.split("nihao").syllables == ["ni", "hao"])
    }

    @Test(arguments: [
        ("n", "ni", false, true), ("n", "nv", false, true), ("h", "ni", false, false),
        ("z", "zi", false, true), ("z", "zhi", false, false), ("z", "zhi", true, true),
        ("zh", "zi", true, true), ("zh", "zi", false, false),
        ("ni", "ni", false, true), ("zi", "zhi", false, false), ("zi", "zhi", true, true),
        ("sa", "sha", true, true), ("ca", "sa", true, false),
    ])
    func aTypedPieceMatchesADictionarySyllable(typed: String, syllable: String, fuzzy: Bool, matches: Bool) {
        #expect(PinyinSegmenter.matches(typed, syllable, fuzzy: fuzzy) == matches)
    }

    @Test func everySyllableInTheListIsItsOwnEntry() {
        #expect(PinyinSegmenter.syllables.count == 415)
        #expect(PinyinSegmenter.syllables.isSuperset(of: ["ce", "cei", "chun", "chuo", "dong", "dou", "gen", "geng"]))
    }

    @Test func apostropheIsAHardBoundaryEvenWhenOneSyllableFits() {
        #expect(segmenter.split("xian").syllables.count == 1)
        #expect(segmenter.split("xi'an").syllables.count == 2)
        // "xia" + "n" would also cover "xian"; fewer syllables wins.
        #expect(segmenter.split("xian").syllables == ["xian"])
    }

    @Test func everyFewestSyllableSplitIsListedBestGuessFirst() {
        #expect(segmenter.splits("fangan").map(\.syllables) == [["fang", "an"], ["fan", "gan"]])
        #expect(segmenter.splits("xian").map(\.syllables) == [["xian"]])
        #expect(segmenter.splits("xi'an").map(\.syllables) == [["xi", "an"]])
        #expect(segmenter.splits("nii") == [.init(syllables: ["ni"], rest: "i")])
        #expect(segmenter.splits(String(repeating: "fangan", count: 6)).count <= 8)
    }

    @Test func vowellessInterjectionsAreNotSyllables() {
        // m, n, ng, hm, hng exist in the dictionary but would split words apart.
        #expect(!PinyinSegmenter.syllables.contains("n"))
        #expect(!PinyinSegmenter.syllables.contains("ng"))
        #expect(PinyinSegmenter.syllables.contains("er"))
        #expect((400...420).contains(PinyinSegmenter.syllables.count))
    }
}

/// Typed letters with no apostrophes, against the readings in out/sentences.tsv.
@Suite struct PinyinSegmenterSentenceTests {
    @Test func unbrokenLettersSplitLikeTheReadings() throws {
        let url = sentencesURL
        let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n")
        var same = 0
        var misses: [String] = []
        for line in lines {
            let reading = line.split(separator: "\t")[0].split(separator: "-").map(String.init)
            let split = PinyinSegmenter().split(reading.joined())
            if split.syllables == reading, split.rest.isEmpty { same += 1 } else if misses.count < 15 { misses.append("\(reading.joined(separator: "-")) → \(split.syllables.joined(separator: "-"))") }
        }
        print("segmenter sentences=\(lines.count) same=\(same)\n" + misses.joined(separator: "\n"))
        #expect(Double(same) / Double(lines.count) > 0.9)
    }
}
