import XCTest
@testable import JapaneseInput

@MainActor
final class JapaneseConverterTests: XCTestCase {
    func testBundledDictionaryIsValid() {
        XCTAssertNoThrow(try JapaneseConverter())
    }

    func testMissingDictionaryFilesAreRejected() throws {
        let empty = FileManager.default.temporaryDirectory
            .appendingPathComponent("japanese-input-invalid-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }

        XCTAssertThrowsError(try JapaneseConverter(dictionaryURL: empty)) { error in
            XCTAssertEqual(error as? JapaneseConverter.ConverterError, .dictionaryIncomplete("louds/charID.chid"))
        }
        XCTAssertThrowsError(try JapaneseConverter(dictionaryURL: empty.appendingPathComponent("absent")))
    }

    private func makeInvalidDictionary() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("japanese-input-strict-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("louds"), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: root.appendingPathComponent("mm.binary").path, contents: Data("x".utf8))
        return root
    }

    func testZeroByteRequiredFileIsRejected() throws {
        let root = try makeInvalidDictionary()
        defer { try? FileManager.default.removeItem(at: root) }
        FileManager.default.createFile(atPath: root.appendingPathComponent("louds/charID.chid").path, contents: Data())

        XCTAssertThrowsError(try JapaneseConverter(dictionaryURL: root)) { error in
            XCTAssertEqual(error as? JapaneseConverter.ConverterError, .dictionaryIncomplete("louds/charID.chid"))
        }
    }

    func testDirectoryInPlaceOfRequiredFileIsRejected() throws {
        let root = try makeInvalidDictionary()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root.appendingPathComponent("louds/charID.chid"), withIntermediateDirectories: true)

        XCTAssertThrowsError(try JapaneseConverter(dictionaryURL: root)) { error in
            XCTAssertEqual(error as? JapaneseConverter.ConverterError, .dictionaryIncomplete("louds/charID.chid"))
        }
    }

    func testUnreadableRequiredFileIsRejected() throws {
        let root = try makeInvalidDictionary()
        defer { try? FileManager.default.removeItem(at: root) }
        let sentinel = root.appendingPathComponent("louds/charID.chid")
        try Data([1, 2, 3]).write(to: sentinel)
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: sentinel.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: sentinel.path) }

        XCTAssertThrowsError(try JapaneseConverter(dictionaryURL: root)) { error in
            XCTAssertEqual(error as? JapaneseConverter.ConverterError, .dictionaryIncomplete("louds/charID.chid"))
        }
    }

    func testKaigiContainsKanjiCandidates() async throws {
        let converter = try JapaneseConverter()
        let candidates = try await converter.candidates(for: "かいぎ")
        XCTAssertTrue(candidates.contains("会議"), "got: \(candidates)")
    }

    func testKaidanRanksKaidanFirst() async throws {
        let converter = try JapaneseConverter()
        let candidates = try await converter.candidates(for: "かいだん")
        XCTAssertEqual(candidates.first, "階段", "got: \(candidates)")
        XCTAssertTrue(candidates.contains("会談"))
    }

    func testLongPhraseFullConversion() async throws {
        let converter = try JapaneseConverter()
        let candidates = try await converter.candidates(for: "あずーきーはしんじだいのきーぼーどあぷりです")
        XCTAssertTrue(candidates.contains("azooKeyは新時代のキーボードアプリです"), "got: \(candidates)")
    }

    func testEmptyReadingYieldsNoCandidates() async throws {
        let converter = try JapaneseConverter()
        let candidates = try await converter.candidates(for: "")
        XCTAssertTrue(candidates.isEmpty)
    }

    func testAlternatingQueriesDoNotContaminateEachOther() async throws {
        let converter = try JapaneseConverter()
        let firstKaigi = try await converter.candidates(for: "かいぎ")
        let kaidan = try await converter.candidates(for: "かいだん")
        let secondKaigi = try await converter.candidates(for: "かいぎ")

        XCTAssertEqual(kaidan.first, "階段")
        XCTAssertEqual(firstKaigi.first, secondKaigi.first, "kaigi top must survive an interleaved kaidan query: \(firstKaigi) vs \(secondKaigi)")
        XCTAssertEqual(firstKaigi, secondKaigi)
    }

    func testReadingsLongerThan128AreRejected() async throws {
        let converter = try JapaneseConverter()
        let tooLong = String(repeating: "あ", count: 129)
        do {
            _ = try await converter.candidates(for: tooLong)
            XCTFail("129 characters must be rejected")
        } catch let error as JapaneseConverter.ConverterError {
            XCTAssertEqual(error, .inputTooLong)
        }
    }

    func testExactly128CharactersCompletes() async throws {
        let converter = try JapaneseConverter()
        let reading = String(repeating: "あ", count: 128)
        let started = Date()
        let candidates = try await converter.candidates(for: reading)
        let elapsed = Date().timeIntervalSince(started)
        XCTAssertFalse(candidates.isEmpty, "128 identical kana still produce at least one full candidate")
        XCTAssertLessThan(elapsed, 60, "long input must not hang; took \(elapsed)s")
        print("long128_seconds=\(elapsed) candidates=\(candidates.prefix(3))")
    }
}
