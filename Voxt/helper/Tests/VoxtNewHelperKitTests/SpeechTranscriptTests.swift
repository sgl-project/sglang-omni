import XCTest
@testable import VoxtNewHelperKit

/// SpeechTranscript: what was said before the recognizer restarted after a pause stays.
final class SpeechTranscriptTests: XCTestCase {
    func run(_ results: [String]) -> String {
        var t = SpeechTranscript()
        return results.map { t.update($0) }.last ?? ""
    }

    func testKeepsTheTextBeforeARestart() {
        XCTAssertEqual(run(["How", "How are", "How are you?", "Fine", "Fine thanks."]), "How are you? Fine thanks.")
        XCTAssertEqual(run(["我想", "我想去北京。", "明天", "明天出发。"]), "我想去北京。明天出发。")
        XCTAssertEqual(run(["Hello there.", "Hi", "Hi again.", "OK", "OK."]), "Hello there. Hi again. OK.")
    }

    func testRevisionsReplace() {
        XCTAssertEqual(run(["How are", "How are you", "How", "Howard"]), "Howard")
        XCTAssertEqual(run(["我想去北京", "我想"]), "我想")
        XCTAssertEqual(run(["Hello world", "hello world."]), "hello world.")
    }

    func testAFinalWithTheWholeTextIsNotDoubled() {
        XCTAssertEqual(run(["How are you?", "Fine", "How are you? Fine."]), "How are you? Fine.")
    }

    func testEmptyResultsKeepThePreviousTranscript() {
        var t = SpeechTranscript()
        XCTAssertEqual(t.update(""), "")
        XCTAssertEqual(t.update("我想去北京"), "我想去北京")
        XCTAssertEqual(t.update(""), "我想去北京")
        XCTAssertEqual(t.text, "我想去北京")
        XCTAssertEqual(t.update("我想"), "我想")
        XCTAssertEqual(run(["How are you?", "Fine", ""]), "How are you? Fine")
    }

    func testTextIsTheWholeTextSoFar() {
        var t = SpeechTranscript()
        XCTAssertEqual(t.text, "")
        _ = t.update("How are you?")
        _ = t.update("Fine")
        XCTAssertEqual(t.text, "How are you? Fine")
    }
}
