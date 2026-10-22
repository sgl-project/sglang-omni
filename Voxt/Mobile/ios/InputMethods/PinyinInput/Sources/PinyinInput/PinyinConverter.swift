import CPinyin
import Foundation
import TypingInput

public actor PinyinConverter {
    public enum ConverterError: Error, Equatable { case dictionaryMissing, dictionaryInvalid, invalidReading, engineUnavailable }
    public struct Candidate: Equatable, Sendable {
        public let text: String
        /// Syllables, from the first, that this candidate replaces.
        public let syllables: Int
        /// The typed syllables and initials this candidate was converted from.
        public var split: [String] = []
        /// The dictionary reading it came from ("ni-hao" for 你好 typed as nh).
        public var reading: String = ""
    }
    /// output_sha256 and bytes in Resources/Notices/DictionaryManifest.json
    /// (Tools/make_data.py defaults: 250k words, 162k of them two-character).
    public static let dataSHA256 = "1af22acc9b2f80aa15a4f778927403d61df3f9b0b7c3f94cc868a5e3d28f37e5"
    public static let dataBytes = 8_059_707

    private final class Engine: @unchecked Sendable {
        let handle: OpaquePointer
        init?(url: URL) {
            guard let value = url.path.withCString({ voxt_pinyin_create($0) }) else { return nil }
            handle = value
        }
        deinit { voxt_pinyin_destroy(handle) }
    }
    private let engine: Engine
    private var segmenter: PinyinSegmenter
    private var fuzzy: Bool
    private let learning: PinyinLearning?

    public static var bundledDataURL: URL? { Bundle.module.url(forResource: "pinyin-data", withExtension: "txt") }

    /// The bundled data, checked against its recorded size and SHA256. `fuzzy`:
    /// z/zh, c/ch, s/sh match each other. `learning`: picks to put first.
    public init(fuzzy: Bool = false, learning: PinyinLearning? = nil) throws {
        guard let url = Self.bundledDataURL else { throw ConverterError.dictionaryMissing }
        try self.init(dictionaryURL: url, byteCount: Self.dataBytes, sha256: Self.dataSHA256, fuzzy: fuzzy, learning: learning)
    }

    /// Other data (tests): nil `sha256` skips the checksum but still requires a pinyin file.
    public init(dictionaryURL: URL, byteCount: Int? = nil, sha256: String?, fuzzy: Bool = false, learning: PinyinLearning? = nil) throws {
        guard FileManager.default.fileExists(atPath: dictionaryURL.path) else { throw ConverterError.dictionaryMissing }
        if let sha256 {
            let size = byteCount ?? ((try? dictionaryURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? -1)
            do { _ = try VerifiedDictionary(url: dictionaryURL, byteCount: size, sha256: sha256) }
            catch { throw ConverterError.dictionaryInvalid }
        }
        guard let engine = Engine(url: dictionaryURL) else { throw ConverterError.dictionaryInvalid }
        self.engine = engine
        self.fuzzy = fuzzy
        self.learning = learning
        segmenter = PinyinSegmenter(fuzzy: fuzzy)
        voxt_pinyin_set_fuzzy(engine.handle, fuzzy ? 1 : 0)
        PinyinSegmenter.syllables.sorted().joined(separator: " ").withCString { voxt_pinyin_set_syllables(engine.handle, $0) }
    }

    public func setFuzzy(_ value: Bool) {
        guard value != fuzzy else { return }
        fuzzy = value
        segmenter = PinyinSegmenter(fuzzy: value)
        voxt_pinyin_set_fuzzy(engine.handle, value ? 1 : 0)
    }

    /// Candidates for typed letters. Letters after the last full syllable are
    /// not converted; no full syllable means no candidates. When letters split
    /// several ways with the same syllable count (fang-an / fan-gan), the split
    /// whose best sentence scores higher wins; the other sentences follow it.
    /// The engine takes at most this many pieces; longer input converts its start.
    public static let maximumPieces = 32

    public func candidates(typing input: String) throws -> [Candidate] {
        let splits = segmenter.splits(input.lowercased()).filter { !$0.syllables.isEmpty }
        var converted: [(score: Double, candidates: [Candidate])] = []
        for split in splits {
            // A newer keystroke cancelled this one: skip the work, the keyboard drops the result.
            try Task.checkCancellation()
            let candidates = try candidates(for: Array(split.syllables.prefix(Self.maximumPieces)))
            converted.append((voxt_pinyin_score(engine.handle), candidates))
        }
        guard let best = converted.indices.max(by: { converted[$0].score < converted[$1].score }) else { return [] }
        var result = converted[best].candidates
        let others = converted.indices.filter { $0 != best }.sorted { converted[$0].score > converted[$1].score }
        for (offset, index) in others.enumerated() {
            guard let sentence = converted[index].candidates.first, !result.contains(where: { $0.text == sentence.text }) else { continue }
            result.insert(sentence, at: min(1 + offset, result.count))
        }
        return learned(first: result)
    }

    /// Picks the user made for these syllables go first, most recent first;
    /// picks that cover only the first few syllables come right after the sentence.
    private func learned(first result: [Candidate]) -> [Candidate] {
        guard let learning, let typed = result.first?.split else { return result }
        var whole: [Candidate] = []
        var partial: [Candidate] = []
        for entry in learning.entries {
            let reading = entry.reading.split(separator: "-").map(String.init)
            guard reading.count <= typed.count,
                  zip(typed, reading).allSatisfy({ PinyinSegmenter.matches($0, $1, fuzzy: fuzzy) }) else { continue }
            let candidate = Candidate(text: entry.text, syllables: reading.count, split: typed, reading: entry.reading)
            if reading.count == typed.count { whole.append(candidate) } else { partial.append(candidate) }
        }
        guard !whole.isEmpty || !partial.isEmpty else { return result }
        var ordered = whole
        ordered += result.prefix(1)
        ordered += partial
        ordered += result.dropFirst()
        var seen = Set<String>()
        return ordered.filter { seen.insert($0.text).inserted }
    }

    public func candidates(for syllables: [String]) throws -> [Candidate] {
        guard !syllables.isEmpty else { return [] }
        let count = syllables.joined(separator: "-").withCString { voxt_pinyin_convert(engine.handle, $0) }
        if count == -1 { throw ConverterError.invalidReading }
        guard count >= 0 else { throw ConverterError.engineUnavailable }
        return (0..<Int(count)).compactMap { index in
            guard let text = voxt_pinyin_candidate(engine.handle, index).flatMap({ String(validatingCString: $0) }) else { return nil }
            let reading = voxt_pinyin_candidate_reading(engine.handle, index).flatMap { String(validatingCString: $0) } ?? ""
            return Candidate(text: text, syllables: voxt_pinyin_candidate_length(engine.handle, index),
                             split: syllables, reading: reading)
        }
    }
}
