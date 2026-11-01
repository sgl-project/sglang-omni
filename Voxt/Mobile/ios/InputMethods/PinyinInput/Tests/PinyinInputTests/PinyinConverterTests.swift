import Foundation
import Testing
@testable import PinyinInput

/// Held-out sentences: Han runs from the app's zh-Hans strings with their pinyin.
let sentencesURL = Bundle.module.url(forResource: "Fixtures/sentences", withExtension: "tsv")!

@Suite struct PinyinConverterTests {
    @Test(arguments: [
        ("nihao", "你好"),
        ("woxiangqu", "我想去"),
        ("weixin", "微信"),
        ("xi'an", "西安"),
    ])
    func bestSentence(input: String, expected: String) async throws {
        let converter = try PinyinConverter()
        let result = try await converter.candidates(typing: input)
        #expect(result.first?.text == expected)
        #expect(result.first?.syllables == PinyinSegmenter().split(input).syllables.count)
    }

    @Test func xianWithoutApostropheIsOneCharacter() async throws {
        let converter = try PinyinConverter()
        let texts = try await converter.candidates(typing: "xian").map(\.text)
        #expect(texts.first.map { ["先", "现", "线", "县", "鲜", "险", "限"].contains($0) } == true, "\(texts.prefix(5))")
        #expect(texts.prefix(10).contains("先"))
        #expect(texts.prefix(10).contains("现"))
        #expect(!texts.contains("西安"))
    }

    @Test func dachuangPutsTheWordFirst() async throws {
        let converter = try PinyinConverter()
        let texts = try await converter.candidates(typing: "dachuang").map(\.text)
        #expect(texts.first == "大床", "\(texts.prefix(5))")
        #expect(texts.contains("大"), "single characters for the first syllable come after the words")
    }

    @Test func candidatesStartAtTheFirstSyllableAndSayHowManyTheyUse() async throws {
        let converter = try PinyinConverter()
        let result = try await converter.candidates(typing: "nihao")
        let ni = try #require(result.first { $0.text == "你" })
        #expect(ni.syllables == 1)
        #expect(Set(result.map(\.text)).count == result.count)
        #expect(result.allSatisfy { $0.syllables >= 1 && $0.syllables <= 2 })
    }

    @Test func lettersNoSyllableCanStartWithAreNotConverted() async throws {
        let converter = try PinyinConverter()
        #expect(try await converter.candidates(typing: "nii").first?.text == "你")
        #expect(try await converter.candidates(typing: "u").isEmpty)
    }

    @Test(arguments: [
        ("deneirong", "的内容", "den-ei-rong ties de-nei-rong; the engine picks"),
        ("wanan", "晚安", "wan-an ties wa-nan"),
        ("fangan", "方案", "fang-an ties fan-gan"),
    ])
    func tiedSplitsAreSettledByTheEngine(input: String, expected: String, note: String) async throws {
        let converter = try PinyinConverter()
        let result = try await converter.candidates(typing: input)
        #expect(result.first?.text == expected, "\(note): \(result.prefix(5).map(\.text))")
    }

    @Test func theOtherSplitStaysInTheList() async throws {
        let converter = try PinyinConverter()
        let result = try await converter.candidates(typing: "fangan")
        let other = try #require(result.first { $0.text == "反感" })
        #expect(other.split == ["fan", "gan"])
        #expect(result.first?.split == ["fang", "an"])
    }

    @Test func typedSentencesEndToEnd() async throws {
        let converter = try PinyinConverter()
        let url = sentencesURL
        let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n")
        var top1 = 0
        for line in lines {
            let fields = line.split(separator: "\t")
            let letters = fields[0].replacingOccurrences(of: "-", with: "")
            if try await converter.candidates(typing: letters).first?.text == String(fields[1]) { top1 += 1 }
        }
        print("typed sentences=\(lines.count) top1=\(top1)")
        #expect(Double(top1) / Double(lines.count) > 0.6)
    }

    @Test func missingAndCorruptDataThrow() throws {
        #expect(throws: PinyinConverter.ConverterError.dictionaryMissing) {
            try PinyinConverter(dictionaryURL: URL(fileURLWithPath: "/nonexistent-voxt-pinyin.data"), sha256: nil)
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        for text in ["", "garbage\n", "# format org.openvanilla.mcbopomofo.sorted\nㄋㄧˇ 你 -1\n"] {
            try Data(text.utf8).write(to: url)
            #expect(throws: PinyinConverter.ConverterError.dictionaryInvalid) {
                try PinyinConverter(dictionaryURL: url, sha256: nil)
            }
        }
    }

    @Test func checksumMismatchThrows() throws {
        #expect(throws: PinyinConverter.ConverterError.dictionaryInvalid) {
            try PinyinConverter(dictionaryURL: try #require(PinyinConverter.bundledDataURL), sha256: String(repeating: "0", count: 64))
        }
    }
}
