import Foundation
import Testing
@testable import ZhuyinInput

@Suite struct ZhuyinConverterTests {
    @Test func realDictionaryAndAlternate() async throws {
        let converter = try ZhuyinConverter()
        let values = try await converter.candidates(for: ["ㄋㄧˇ"])
        #expect(values.first == "你")
        #expect(values.contains("妳"))
        #expect(Set(values).count == values.count)
    }
    @Test func wholeSentenceCandidatesRetainBothSyllables() async throws {
        let converter = try ZhuyinConverter()
        let values = try await converter.candidates(for: ["ㄋㄧˇ", "ㄏㄠˇ"], cursor: 0)
        #expect(values.first == "你好")
        #expect(values.contains("妳好"))
        #expect(!values.contains("妳"))
        #expect(values.allSatisfy { $0.count == 2 })
    }
    @Test func alternatingQueriesDoNotRetainOverrides() async throws {
        let converter = try ZhuyinConverter()
        for _ in 0..<3 {
            #expect(try await converter.candidates(for: ["ㄋㄧˇ"]).first == "你")
            #expect(try await converter.candidates(for: ["ㄏㄠˇ"]).first == "好")
        }
    }
    @Test func missingDataThrows() {
        #expect(throws: ZhuyinConverter.ConverterError.dictionaryMissing) {
            try ZhuyinConverter(dictionaryURL: URL(fileURLWithPath: "/nonexistent-voxt-zhuyin.data"))
        }
    }
    @Test func corruptedDataThrows() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        for text in ["", "garbage\n", "# format org.openvanilla.mcbopomofo.sorted\n"] {
            try Data(text.utf8).write(to: url)
            #expect(throws: ZhuyinConverter.ConverterError.dictionaryInvalid) {
                try ZhuyinConverter(dictionaryURL: url)
            }
        }
    }
    @Test func directoryIsNotData() {
        #expect(throws: ZhuyinConverter.ConverterError.dictionaryInvalid) {
            try ZhuyinConverter(dictionaryURL: FileManager.default.temporaryDirectory)
        }
    }
    @Test func malformedAndUnfinishedReadingsThrow() async throws {
        let converter = try ZhuyinConverter()
        for reading in ["banana", "ㄋ", "ㄋˇ", "ˇㄋㄧ", "ㄧㄋ", "ㄋㄋㄧ", "ㄋㄧˇˇ", "", "_punctuation_list"] {
            await #expect(throws: ZhuyinConverter.ConverterError.invalidReading) {
                try await converter.candidates(for: [reading])
            }
        }
    }
    @Test func emptyCompositionHasNoCandidates() async throws {
        #expect(try await ZhuyinConverter().candidates(for: []).isEmpty)
    }
    @Test func overLimitThrows() async throws {
        let converter = try ZhuyinConverter()
        await #expect(throws: ZhuyinConverter.ConverterError.inputTooLong) {
            try await converter.candidates(for: Array(repeating: "ㄋㄧˇ", count: 33))
        }
    }
    @Test func maximumInputCompletes() async throws {
        let values = try await ZhuyinConverter().candidates(for: Array(repeating: "ㄋㄧˇ", count: 32))
        #expect(!values.isEmpty)
        #expect(values.allSatisfy { $0.count == 32 })
        #expect(values.count <= 10)
    }
    @Test func badCursorThrows() async throws {
        let converter = try ZhuyinConverter()
        for cursor in [-1, 2] {
            await #expect(throws: ZhuyinConverter.ConverterError.invalidReading) {
                try await converter.candidates(for: ["ㄋㄧˇ"], cursor: cursor)
            }
        }
    }
    @Test func cancellationThrows() async throws {
        let converter = try ZhuyinConverter()
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await converter.candidates(for: ["ㄋㄧˇ"])
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }
}
