import Foundation
import Testing
@testable import VoxtMobileKit

/// Pinyin options the app sets and the keyboard reads, in the App Group.
@Suite struct PinyinSettingsTests {
    private func defaults() -> UserDefaults {
        let name = "pinyin-settings.\(UUID().uuidString)"
        return UserDefaults(suiteName: name)!
    }

    @Test func fuzzySyllablesAreOffUntilTurnedOn() {
        let store = defaults()
        #expect(!PinyinSettings.fuzzy(in: store))
        PinyinSettings.setFuzzy(true, in: store)
        #expect(PinyinSettings.fuzzy(in: store))
        #expect(store.bool(forKey: "pinyinFuzzySyllables"), "the key the keyboard reads")
    }

    @Test func learnedWordsLiveInTheSharedContainerAndClearForGood() throws {
        let container = FileManager.default.temporaryDirectory.appendingPathComponent("pinyin-settings-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: container) }
        let url = try #require(PinyinSettings.learnedWordsURL(in: container))
        #expect(url.path.hasPrefix(container.path))
        #expect(!PinyinSettings.hasLearnedWords(in: container))
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("ni-hao\t拟好\t1\t0".utf8).write(to: url)
        #expect(PinyinSettings.hasLearnedWords(in: container))
        PinyinSettings.clearLearnedWords(in: container)
        #expect(!FileManager.default.fileExists(atPath: url.path))
        #expect(!PinyinSettings.hasLearnedWords(in: container))
        #expect(PinyinSettings.learnedWordsURL(in: nil) == nil, "no App Group, nothing saved")
    }
}
