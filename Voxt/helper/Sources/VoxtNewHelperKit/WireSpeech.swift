import Foundation

// speech.*: internal/helper/proto/speech.go field for field (see the
// conventions in Wire.swift), and the recognizer's locale for a lang.

public struct SpeechParams: Codable, Sendable {
    /// A BCP-47 tag; nil: the Mac's own dictation language.
    public var lang: String?
    public var sampleRate: Int?
    public var hints: [String]?
}

public struct SpeechStartedParams: Codable, Sendable {
    public var job: String
    public var locale: String

    public init(job: String, locale: String) {
        self.job = job
        self.locale = locale
    }
}

public struct SpeechPartialParams: Codable, Sendable {
    public var job: String
    public var text: String

    public init(job: String, text: String) {
        self.job = job
        self.text = text
    }
}

public struct SpeechResult: Codable, Sendable {
    public var text: String
    public init(text: String) { self.text = text }
}

public struct SpeechAudioParams: Codable, Sendable {
    public var job: String
    /// Mono PCM16 little endian; base64 on the wire.
    public var data: Data?
    public var end: Bool?
}

/// The recognizer locale for lang (a BCP-47 tag: "zh", "en-GB",
/// "zh-Hant") among supported (SFSpeechRecognizer's, "zh-CN"): the tag
/// itself, else its language in the user's region (unless the tag names
/// a script, which says its region), else in the language's likely
/// region, else the first of its language; nil when none.
public func speechLocale(_ lang: String, supported: [String], userRegion: String?) -> String? {
    let ids = Set(supported.map { $0.replacingOccurrences(of: "_", with: "-") })
    let tag = Locale.Language(identifier: lang)
    let given = Locale.Language.Components(identifier: lang) // as written: Locale.Language infers a script
    guard let code = tag.languageCode?.identifier else { return nil }
    let likely = Locale.Language(identifier: tag.maximalIdentifier).region?.identifier
    let regions = [given.region?.identifier, given.script == nil ? userRegion : nil, likely]
    for case let region? in regions where ids.contains("\(code)-\(region)") {
        return "\(code)-\(region)"
    }
    return ids.filter { Locale.Language(identifier: $0).languageCode?.identifier == code }.sorted().first
}
