import XCTest
@testable import VoxtMobile
import VoxtMobileKit

/// VOXT_TEST_RESET leaves pinyin as a new install has it, so a test that ran before
/// (fuzzy on, learned words) can't change the next one.
final class TestResetTests: XCTestCase {
    func testResetTurnsFuzzyOffAndClearsLearnedWords() throws {
        let name = "test-reset.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let container = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(name, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: container) }
        PinyinSettings.setFuzzy(true, in: defaults)
        let learned = try XCTUnwrap(PinyinSettings.learnedWordsURL(in: container))
        try FileManager.default.createDirectory(at: learned.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("ni hao\t你好\n".utf8).write(to: learned)
        XCTAssertTrue(PinyinSettings.hasLearnedWords(in: container))

        AppModel.resetPinyinForTests(defaults: defaults, container: container)

        XCTAssertFalse(PinyinSettings.fuzzy(in: defaults))
        XCTAssertFalse(PinyinSettings.hasLearnedWords(in: container))
    }
}
