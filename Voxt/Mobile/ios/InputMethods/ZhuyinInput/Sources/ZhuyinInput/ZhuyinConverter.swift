import Foundation
import CryptoKit
import CZhuyin

public actor ZhuyinConverter {
    public enum ConverterError: Error, Equatable {
        case dictionaryMissing, dictionaryInvalid, invalidReading, inputTooLong, engineUnavailable
    }
    private let dictionaryURL: URL
    // Only this actor accesses the handle; the holder allows its deinit to free
    // the C++ allocation without moving a non-Sendable pointer between actors.
    private final class Engine: @unchecked Sendable {
        let handle: OpaquePointer
        init(url: URL) throws {
            guard let value = url.path.withCString({ voxt_zhuyin_create($0) }) else {
                throw ConverterError.engineUnavailable
            }
            handle = value
        }
        deinit { voxt_zhuyin_destroy(handle) }
    }
    private var engine: Engine?

    public init(dictionaryURL: URL? = nil) throws {
        guard let url = dictionaryURL ?? Bundle.module.url(forResource: "data", withExtension: "txt"),
              FileManager.default.fileExists(atPath: url.path) else { throw ConverterError.dictionaryMissing }
        guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey, .isReadableKey]),
              values.isRegularFile == true, values.isReadable == true,
              values.fileSize == 7_528_605,
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined()
                == "0deae7b7c1dcde1d7a30d139e7068543e0c0e7112e944b63eb52947bca1db7ac" else {
            throw ConverterError.dictionaryInvalid
        }
        self.dictionaryURL = url
    }

    /// Full-composition replacements. Cursor is a syllable index, defaulting to
    /// the last syllable. Conversion stays on this actor, not the main actor.
    public func candidates(for readings: [String], cursor: Int? = nil) async throws -> [String] {
        try Task.checkCancellation()
        guard readings.count <= 32 else { throw ConverterError.inputTooLong }
        guard !readings.isEmpty else { return [] }
        let position = cursor ?? readings.count - 1
        guard readings.indices.contains(position), readings.allSatisfy(Self.isCompleteReading) else {
            throw ConverterError.invalidReading
        }
        if engine == nil { engine = try Engine(url: dictionaryURL) }
        guard let engine else { throw ConverterError.engineUnavailable }
        let count = readings.joined(separator: "-").withCString { voxt_zhuyin_convert(engine.handle, $0, position) }
        if count == -1 { throw ConverterError.invalidReading }
        guard count >= 0 else { throw ConverterError.engineUnavailable }
        let results = (0..<Int(count)).compactMap { index -> String? in
            guard let text = voxt_zhuyin_candidate(engine.handle, index) else { return nil }
            return String(validatingCString: text)
        }
        try Task.checkCancellation()
        return results
    }

    private static func isCompleteReading(_ reading: String) -> Bool {
        let initials = "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ"
        let medials = "ㄧㄨㄩ", finals = "ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ", tones = "ˊˇˋ˙"
        var previous = -1
        var hasVowel = false
        var initial: Character?
        for symbol in reading {
            let group: Int
            if initials.contains(symbol) { group = 0; initial = symbol }
            else if medials.contains(symbol) { group = 1; hasVowel = true }
            else if finals.contains(symbol) { group = 2; hasVowel = true }
            else if tones.contains(symbol) { group = 3 }
            else { return false }
            guard group > previous else { return false }
            previous = group
        }
        // McBopomofo 3.1.1 includes bare ㄋ → ㄋ symbol rows.
        // Only apical initials can form a complete syllable without a vowel.
        return hasVowel || initial.map { "ㄓㄔㄕㄖㄗㄘㄙ".contains($0) } == true
    }
}
