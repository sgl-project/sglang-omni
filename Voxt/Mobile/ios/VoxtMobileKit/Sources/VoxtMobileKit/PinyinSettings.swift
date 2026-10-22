import Foundation

/// Simplified Chinese pinyin options the app sets and the keyboard reads, in the
/// App Group. Learned words are the keyboard's remembered picks, kept on the phone.
public enum PinyinSettings {
    /// Fuzzy z/zh, c/ch, s/sh. Off by default.
    public static let fuzzyKey = "pinyinFuzzySyllables"
    static let learnedWordsPath = "Library/Pinyin/learned-words.tsv"

    public static func fuzzy(in defaults: UserDefaults) -> Bool { defaults.bool(forKey: fuzzyKey) }

    public static func setFuzzy(_ on: Bool, in defaults: UserDefaults) { defaults.set(on, forKey: fuzzyKey) }

    /// Nil without a container (no App Group): picks stay in memory.
    public static func learnedWordsURL(in container: URL?) -> URL? {
        container?.appendingPathComponent(learnedWordsPath, isDirectory: false)
    }

    /// The App Group's file.
    public static var learnedWordsURL: URL? {
        learnedWordsURL(in: FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: DictationChannel.appGroup))
    }

    public static func hasLearnedWords(in container: URL?) -> Bool {
        learnedWordsURL(in: container).map { FileManager.default.fileExists(atPath: $0.path) } ?? false
    }

    /// Settings › Clear learned words. The keyboard notices the file is gone on its next conversion.
    public static func clearLearnedWords(in container: URL?) {
        guard let url = learnedWordsURL(in: container) else { return }
        try? FileManager.default.removeItem(at: url)
    }
}
