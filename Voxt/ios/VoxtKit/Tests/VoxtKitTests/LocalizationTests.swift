import Foundation
import Testing

/// The iOS counterpart of internal/i18n's TestKeyParity: every L literal
/// in VoxtUI has an English and a Simplified Chinese string, no string is
/// unused, and a translation keeps its message's format verbs.
@Suite struct LocalizationTests {
    static let ui = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../../Sources/VoxtUI").standardized

    static func table(_ locale: String) throws -> [String: String] {
        let url = ui.appendingPathComponent("Resources/\(locale).lproj/Localizable.strings")
        let data = try Data(contentsOf: url)
        return try #require(PropertyListSerialization.propertyList(from: data, format: nil) as? [String: String])
    }

    static func literals() throws -> Set<String> {
        var keys = Set<String>()
        let files = FileManager.default.enumerator(at: ui, includingPropertiesForKeys: nil)
        let pattern = try NSRegularExpression(pattern: #"\bL\("((?:[^"\\]|\\.)*)""#)
        while let url = files?.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            let source = try String(contentsOf: url, encoding: .utf8)
            for match in pattern.matches(in: source, range: NSRange(source.startIndex..., in: source)) {
                let literal = String(source[Range(match.range(at: 1), in: source)!])
                keys.insert(literal.replacingOccurrences(of: #"\""#, with: "\""))
            }
        }
        return keys
    }

    static func verbs(_ s: String) -> [String] {
        let pattern = try! NSRegularExpression(pattern: #"%(\.\d+)?[@dfs]"#)
        return pattern.matches(in: s, range: NSRange(s.startIndex..., in: s)).map { String(s[Range($0.range, in: s)!]) }
    }

    @Test func everyMessageHasBothLanguages() throws {
        let used = try Self.literals()
        #expect(used.count > 50)
        for locale in ["en", "zh-Hans"] {
            let table = try Self.table(locale)
            #expect(used.subtracting(table.keys).sorted() == [], "missing in \(locale)")
            #expect(Set(table.keys).subtracting(used).sorted() == [], "unused in \(locale)")
        }
    }

    @Test func translationsKeepTheirFormatVerbs() throws {
        let zh = try Self.table("zh-Hans")
        for (key, value) in zh {
            #expect(Self.verbs(key) == Self.verbs(value), "\(key)")
        }
        for (key, value) in try Self.table("en") {
            #expect(key == value)
        }
    }

    @Test func chineseUsesTheProductWords() throws {
        let zh = try Self.table("zh-Hans")
        #expect(zh["Transcribe"] == "转录" && zh["Translate"] == "翻译" && zh["Rewrite"] == "改写")
        #expect(zh["Dictionary"] == "词典" && zh["Language models"] == "语言模型" && zh["Speech models"] == "语音识别模型")
        for value in zh.values {
            #expect(!value.contains("语音模型"), "speech models are 语音识别模型: \(value)")
        }
    }
}
